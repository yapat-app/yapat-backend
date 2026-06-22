"""
Benchmark: AL acquisition scoring (uncertainty, density, diversity).

Loads real embeddings, sweeps N (unlabeled snippets), times each sampler.

Usage:
    python -m benchmarks.bench_distance \
        --snippet-set-id 7 --embedding-model-id 1 \
        --dataset anuraset --sizes 1000,5000,10000,full --repeats 5
"""

import argparse

import numpy as np
import torch


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,full")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--labeled-fraction", type=float, default=0.05,
                   help="Fraction of N used as labeled set for diversity/density")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--density-k", type=int, default=10)
    return p.parse_args()


def main():
    args = _parse_args()

    # --- Load embeddings ---
    from app.services.pam_al._embedding_cache import get_cache_dir, _embeddings_path
    cache_dir = get_cache_dir(args.snippet_set_id, args.embedding_model_id)
    X_full = np.load(_embeddings_path(cache_dir), mmap_mode="r").astype(np.float32)
    N_full = X_full.shape[0]
    print(f"Loaded {N_full:,} embeddings from {cache_dir}")

    rng = np.random.default_rng(42)

    sizes = []
    for s in args.sizes.split(","):
        s = s.strip()
        if s == "full":
            sizes.append(N_full)
        else:
            sizes.append(min(int(s), N_full))

    from active_learning.samplers import uncertainty, density, diversity
    from benchmarks.runner import run_benchmark

    for operation, run_fn_factory in [
        ("uncertainty", lambda N_l: lambda inp: uncertainty(inp[0])),
        ("density",     lambda N_l: lambda inp: density(inp[1], k=args.density_k)),
        ("diversity",   lambda N_l: lambda inp: diversity(inp[1], inp[2])),
    ]:
        def setup(N, op=operation):
            idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
            X_u = X_full[idx].copy()
            N_l = max(1, int(N * args.labeled_fraction))
            idx_l = rng.choice(N, size=N_l, replace=False)
            Z_l = X_u[idx_l]
            # Synthetic probability matrix
            P = rng.random((N, args.num_classes)).astype(np.float32)
            P = P / P.sum(axis=1, keepdims=True)
            P = torch.tensor(P, dtype=torch.float32)
            return P, X_u, Z_l

        run_fn = run_fn_factory(0)

        print(f"\nBenchmarking {operation} on cpu, dataset={args.dataset}")
        run_benchmark(
            operation=operation,
            device="cpu",
            dataset=args.dataset,
            sizes=sizes,
            setup_fn=setup,
            run_fn=run_fn,
            repeats=args.repeats,
            warmup=args.warmup,
        )

    print("Done. Results appended to benchmarks/results.csv")


if __name__ == "__main__":
    main()
