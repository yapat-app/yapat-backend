"""
Benchmark: FAISS index build and search per AL cycle.

Times the two FAISS operations used in acquisition scoring:
  - density:   HNSWFlat index build + k-NN search (unlabeled set)
  - diversity: IndexFlatL2 index build + 1-NN search (unlabeled vs labeled)

Sweeps N (unlabeled snippet count) using real embeddings from cache.

Usage:
    python -m benchmarks.bench_faiss \
        --snippet-set-id 8 --embedding-model-id 1 \
        --dataset anuraset --sizes 1000,5000,10000,20000,30687 --repeats 3
"""

import argparse
import time

import faiss
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,10000,20000,30687")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--labeled-fraction", type=float, default=0.05)
    p.add_argument("--density-k", type=int, default=10)
    p.add_argument("--hnsw-m", type=int, default=32)
    return p.parse_args()


def _time_density(z_u: np.ndarray, k: int, M: int) -> tuple[float, float]:
    """Returns (build_time, search_time) for HNSW density index."""
    dim = z_u.shape[1]

    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efSearch = 64
    index.add(z_u)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.search(z_u, k=k)
    search_time = time.perf_counter() - t0

    return build_time, search_time


def _time_diversity(z_u: np.ndarray, z_l: np.ndarray) -> tuple[float, float]:
    """Returns (build_time, search_time) for FlatL2 diversity index."""
    dim = z_l.shape[1]

    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(z_l)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.search(z_u, k=1)
    search_time = time.perf_counter() - t0

    return build_time, search_time


def main():
    args = _parse_args()

    from app.services.pam_al._embedding_cache import get_cache_dir, _embeddings_path
    from benchmarks.stage_timer import write_csv_row
    import time as _time

    cache_dir = get_cache_dir(args.snippet_set_id, args.embedding_model_id)
    X_full = np.load(_embeddings_path(cache_dir), mmap_mode="r").astype(np.float32)
    N_full = X_full.shape[0]
    print(f"Loaded {N_full:,} embeddings from {cache_dir}")

    rng = np.random.default_rng(42)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    _FIELDS = [
        "operation", "device", "dataset", "N", "repeats",
        "time_mean_s", "time_std_s", "throughput_per_s",
        "peak_mem_mb", "gpu_peak_mem_mb", "timestamp",
    ]

    for op_name, time_fn in [
        ("faiss_density_build",  lambda z_u, z_l: _time_density(z_u, args.density_k, args.hnsw_m)[0]),
        ("faiss_density_search", lambda z_u, z_l: _time_density(z_u, args.density_k, args.hnsw_m)[1]),
        ("faiss_diversity_build",  lambda z_u, z_l: _time_diversity(z_u, z_l)[0]),
        ("faiss_diversity_search", lambda z_u, z_l: _time_diversity(z_u, z_l)[1]),
    ]:
        print(f"\nBenchmarking {op_name} on cpu, dataset={args.dataset}")

        for N in sizes:
            idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
            z_u = X_full[idx].copy()
            N_l = max(1, int(N * args.labeled_fraction))
            idx_l = rng.choice(N, size=N_l, replace=False)
            z_l = z_u[idx_l].copy()

            times = []
            for rep in range(args.warmup + args.repeats):
                t = time_fn(z_u, z_l)
                if rep >= args.warmup:
                    times.append(t)

            mean_t = float(np.mean(times))
            std_t = float(np.std(times))
            throughput = N / mean_t if mean_t > 0 else 0

            print(f"  {op_name} N={N:>8,} mean={mean_t:.4f}s std={std_t:+.4f}s ({throughput:,.0f}/s)")

            write_csv_row(
                {
                    "operation": op_name,
                    "device": "cpu",
                    "dataset": args.dataset,
                    "N": N,
                    "repeats": args.repeats,
                    "time_mean_s": round(mean_t, 6),
                    "time_std_s": round(std_t, 6),
                    "throughput_per_s": round(throughput, 1),
                    "peak_mem_mb": 0,
                    "gpu_peak_mem_mb": None,
                    "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                extra_fields=_FIELDS,
            )

    print("\nDone. Results appended to benchmarks/results.csv")


if __name__ == "__main__":
    main()
