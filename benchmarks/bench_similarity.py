"""
Benchmark: pgvector cosine search.

Requires a live Postgres connection with embeddings already ingested.
Sweeps N (LIMIT on the embedding table) to simulate different dataset sizes.

Usage:
    python -m benchmarks.bench_similarity \
        --snippet-set-id 7 --embedding-model-id 1 \
        --dataset anuraset --sizes 1000,5000,20000,full --repeats 5 --k 10
"""

import argparse
import os


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,full")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--k", type=int, default=10, help="Number of nearest neighbours")
    return p.parse_args()


def main():
    args = _parse_args()

    # Needs app context for DB / VectorStore
    from app.database import SessionLocal
    from app.services.embedding_service import VectorStore
    from app.services.pam_al._embedding_cache import get_cache_dir, _embeddings_path

    import numpy as np

    cache_dir = get_cache_dir(args.snippet_set_id, args.embedding_model_id)
    X_full = np.load(_embeddings_path(cache_dir), mmap_mode="r").astype(np.float32)
    N_full = X_full.shape[0]
    print(f"Loaded {N_full:,} embeddings for query vectors")

    rng = np.random.default_rng(42)

    sizes = []
    for s in args.sizes.split(","):
        s = s.strip()
        if s == "full":
            sizes.append(N_full)
        else:
            sizes.append(min(int(s), N_full))

    from benchmarks.runner import run_benchmark

    db = SessionLocal()
    store = VectorStore(db)

    # Pick a stable query vector
    query_vec = X_full[0].tolist()

    def setup(N):
        return N  # VectorStore.search uses LIMIT internally via k param

    def run(N):
        # Use k=N to exercise the index at different result set sizes
        store.search(
            query_vector=query_vec,
            embedding_model_id=args.embedding_model_id,
            k=min(N, args.k),
        )

    print(f"\nBenchmarking pgvector search on cpu, dataset={args.dataset}")
    run_benchmark(
        operation="similarity_search",
        device="cpu",
        dataset=args.dataset,
        sizes=sizes,
        setup_fn=setup,
        run_fn=run,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    db.close()
    print("Done. Results appended to benchmarks/results.csv")


if __name__ == "__main__":
    main()
