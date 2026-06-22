import csv
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import psutil

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

_FIELDS = ["operation", "device", "dataset", "N", "time_s", "peak_mem_mb", "timestamp"]


@dataclass
class _TimerCtx:
    n: Optional[int] = None


@contextmanager
def stage_timer(
    operation: str,
    device: str,
    dataset: str,
    n: Optional[int] = None,
    csv_path: Optional[str] = None,
):
    if csv_path is None:
        csv_path = RESULTS_CSV
    ctx = _TimerCtx(n=n)
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    yield ctx
    elapsed = time.perf_counter() - t0
    mem_after = proc.memory_info().rss / 1e6
    write_csv_row(
        {
            "operation": operation,
            "device": device,
            "dataset": dataset,
            "N": ctx.n,
            "time_s": round(elapsed, 3),
            "peak_mem_mb": round(mem_after - mem_before, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        csv_path=csv_path,
    )


def write_csv_row(
    row: dict,
    csv_path: Optional[str] = None,
    extra_fields: Optional[list] = None,
) -> None:
    if csv_path is None:
        csv_path = RESULTS_CSV
    fields = extra_fields if extra_fields is not None else _FIELDS
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
