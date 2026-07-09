"""
Benchmark: FAISS index build and search per AL cycle.

Times the FAISS operations used in acquisition scoring:
  - density:        HNSWFlat index build + k-NN search (unlabeled set)
  - diversity:      IndexFlatL2 index build + 1-NN search (unlabeled vs labeled)
                    -- the exact path, used below DIVERSITY_HNSW_MIN_NL
  - diversity_hnsw:  HNSWFlat index build + 1-NN search (unlabeled vs labeled)
                    -- the approximate path, used at/above DIVERSITY_HNSW_MIN_NL
                    (see active_learning/samplers.py::diversity())

Sweeps N (unlabeled snippet count) using real embeddings from cache. Bracket
the sweep around the current DIVERSITY_HNSW_MIN_NL config default (500) --
e.g. include labeled-set sizes both below and above it -- so the Flat vs
HNSW crossover point can be read directly off the results instead of guessed.

Also reports diversity_hnsw's approximation quality (recall@1 and mean/max
distance error against exact Flat) at each labeled-set size, via
active_learning.samplers.diversity_approx_error, so results.csv captures
both the speedup and its accuracy cost in one run.

Usage:
    python -m benchmarks.bench_faiss \
        --snippet-set-id 8 --embedding-model-id 1 \
        --dataset anuraset --sizes 1000,5000,10000,20000,30687 --repeats 3
"""

import argparse
import time

import faiss
import numpy as np
import torch


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,10000,20000,30687")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--labeled-fraction", type=float, default=0.05,
                   help="Fraction of N used as labeled set (ignored if --labeled-count set)")
    p.add_argument("--labeled-count", type=int, default=None,
                   help="Absolute number of labeled snippets (overrides --labeled-fraction)")
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
    """Returns (build_time, search_time) for FlatL2 diversity index (exact, O(N*Nl))."""
    dim = z_l.shape[1]

    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(z_l)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.search(z_u, k=1)
    search_time = time.perf_counter() - t0

    return build_time, search_time


def _time_diversity_hnsw(z_u: np.ndarray, z_l: np.ndarray, M: int) -> tuple[float, float]:
    """
    Returns (build_time, search_time) for HNSW diversity index (approximate,
    ~O(N log Nl)) -- the path diversity() switches to at/above
    DIVERSITY_HNSW_MIN_NL. Same index construction as _make_hnsw_index() in
    active_learning/samplers.py.
    """
    dim = z_l.shape[1]

    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efSearch = 64
    index.add(z_l)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.search(z_u, k=1)
    search_time = time.perf_counter() - t0

    return build_time, search_time


def _diversity_accuracy(z_u: np.ndarray, z_l: np.ndarray) -> dict:
    """
    Wraps active_learning.samplers.diversity_approx_error() so recall/error
    against exact Flat search is reported alongside timing, using the same
    (L2-normalized) distance convention diversity() actually uses in
    production -- not a separate, unnormalized comparison.
    """
    from active_learning.samplers import diversity_approx_error

    z_u_t = torch.tensor(z_u, dtype=torch.float32)
    z_l_t = torch.tensor(z_l, dtype=torch.float32)
    return diversity_approx_error(z_u_t, z_l_t)


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
        ("faiss_diversity_hnsw_build",  lambda z_u, z_l: _time_diversity_hnsw(z_u, z_l, args.hnsw_m)[0]),
        ("faiss_diversity_hnsw_search", lambda z_u, z_l: _time_diversity_hnsw(z_u, z_l, args.hnsw_m)[1]),
    ]:
        print(f"\nBenchmarking {op_name} on cpu, dataset={args.dataset}")

        for N in sizes:
            idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
            z_u = X_full[idx].copy()
            N_l = args.labeled_count if args.labeled_count is not None else max(1, int(N * args.labeled_fraction))
            remaining = np.setdiff1d(np.arange(N_full), idx)
            if args.labeled_count is not None and len(remaining) >= N_l:
                idx_l = rng.choice(remaining, size=N_l, replace=False)
                z_l = X_full[idx_l].copy()
            else:
                idx_l = rng.choice(N, size=min(N_l, N), replace=False)
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

    # --- Accuracy of the HNSW approximation vs exact Flat search ---
    # Written to a separate CSV: recall/error columns don't fit the timing
    # schema above, and results.csv's header is fixed at file-creation time
    # (write_csv_row only writes a header for a brand-new file), so mixing
    # schemas into the same file risks a misaligned header for whichever
    # schema didn't create it.
    import os
    accuracy_csv = os.path.join(os.path.dirname(__file__), "diversity_accuracy.csv")
    _ACCURACY_FIELDS = [
        "N", "n_labeled", "recall_at_1", "mean_abs_distance_error",
        "max_abs_distance_error", "dataset", "timestamp",
    ]
    print(f"\nMeasuring diversity HNSW approximation accuracy (vs exact Flat), dataset={args.dataset}")
    for N in sizes:
        idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
        z_u = X_full[idx].copy()
        N_l = args.labeled_count if args.labeled_count is not None else max(1, int(N * args.labeled_fraction))
        remaining = np.setdiff1d(np.arange(N_full), idx)
        if args.labeled_count is not None and len(remaining) >= N_l:
            idx_l = rng.choice(remaining, size=N_l, replace=False)
            z_l = X_full[idx_l].copy()
        else:
            idx_l = rng.choice(N, size=min(N_l, N), replace=False)
            z_l = z_u[idx_l].copy()

        result = _diversity_accuracy(z_u, z_l)
        print(
            f"  N={N:>8,} n_labeled={z_l.shape[0]:>6,} "
            f"recall@1={result['recall_at_1']:.4f} "
            f"mean_abs_err={result['mean_abs_distance_error']:.5f} "
            f"max_abs_err={result['max_abs_distance_error']:.5f}"
        )
        write_csv_row(
            {
                "N": N,
                "n_labeled": z_l.shape[0],
                "recall_at_1": round(result["recall_at_1"], 6) if result["recall_at_1"] is not None else None,
                "mean_abs_distance_error": round(result["mean_abs_distance_error"], 6) if result["mean_abs_distance_error"] is not None else None,
                "max_abs_distance_error": round(result["max_abs_distance_error"], 6) if result["max_abs_distance_error"] is not None else None,
                "dataset": args.dataset,
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            csv_path=accuracy_csv,
            extra_fields=_ACCURACY_FIELDS,
        )

    print("\nDone. Timing results appended to benchmarks/results.csv, accuracy results to benchmarks/diversity_accuracy.csv")


if __name__ == "__main__":
    main()
