"""
Micro-benchmark harness.

Each adapter calls run_benchmark(setup_fn, run_fn, ...) to sweep N values
and write mean/std rows to results.csv.
"""

import statistics
import time
from typing import Any, Callable, List, Optional

import psutil

from benchmarks.stage_timer import write_csv_row

_BENCH_FIELDS = [
    "operation", "device", "dataset", "N", "repeats",
    "time_mean_s", "time_std_s", "throughput_per_s",
    "peak_mem_mb", "gpu_peak_mem_mb", "timestamp",
]


def run_benchmark(
    operation: str,
    device: str,
    dataset: str,
    sizes: List[int],
    setup_fn: Callable[[int], Any],
    run_fn: Callable[[Any], None],
    repeats: int = 5,
    warmup: int = 1,
    csv_path: Optional[str] = None,
) -> None:
    """
    For each N in sizes: warmup, then time `repeats` calls to run_fn(setup_fn(N)).
    Writes one row per N to results.csv.
    """
    try:
        import torch
        _torch_available = True
    except ImportError:
        _torch_available = False

    for N in sizes:
        args = setup_fn(N)

        for _ in range(warmup):
            run_fn(args)

        if _torch_available and device == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1e6

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_fn(args)
            if _torch_available and device == "cuda":
                import torch
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mem_after = proc.memory_info().rss / 1e6

        gpu_mem = None
        if _torch_available and device == "cuda":
            import torch
            gpu_mem = round(torch.cuda.max_memory_allocated() / 1e6, 1)

        mean_t = statistics.mean(times)
        write_csv_row(
            {
                "operation": operation,
                "device": device,
                "dataset": dataset,
                "N": N,
                "repeats": repeats,
                "time_mean_s": round(mean_t, 4),
                "time_std_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
                "throughput_per_s": round(N / mean_t, 1) if mean_t > 0 else None,
                "peak_mem_mb": round(mem_after - mem_before, 1),
                "gpu_peak_mem_mb": gpu_mem,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            csv_path=csv_path,
            extra_fields=_BENCH_FIELDS,
        )
        print(
            f"  {operation} N={N:>8,} "
            f"mean={mean_t:.3f}s std={times[-1] - times[0]:.3f}s "
            f"({round(N / mean_t):,}/s)"
        )
