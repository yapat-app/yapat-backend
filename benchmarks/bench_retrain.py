"""
Benchmark: PAM-AL retraining (linear head).

Sweeps Nl (labeled snippet count) and times model.fit().
Loads real embeddings from on-disk cache, randomly assigns binary labels.

Usage:
    python -m benchmarks.bench_retrain \
        --snippet-set-id 7 --embedding-model-id 1 \
        --device cpu --dataset anuraset \
        --sizes 100,500,2000,full --repeats 5 \
        --num-classes 10
"""

import argparse

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="100,500,2000,full")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = _parse_args()

    # --- Load embeddings from cache ---
    from app.services.pam_al._embedding_cache import get_cache_dir, _embeddings_path
    cache_dir = get_cache_dir(args.snippet_set_id, args.embedding_model_id)
    X_full = np.load(_embeddings_path(cache_dir), mmap_mode="r").astype(np.float32)
    N_full = X_full.shape[0]
    n_dim = X_full.shape[1]
    print(f"Loaded {N_full:,} embeddings (dim={n_dim}) from {cache_dir}")

    rng = np.random.default_rng(42)

    # --- Parse sizes ---
    sizes = []
    for s in args.sizes.split(","):
        s = s.strip()
        if s == "full":
            sizes.append(N_full)
        else:
            sizes.append(min(int(s), N_full))

    from active_learning.model_zoo.linear_multilabel_classifier import MultiLabelLinearClassifier
    from benchmarks.runner import run_benchmark

    def setup(N):
        idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
        X = X_full[idx].copy()
        # Synthetic multi-hot labels: at least one positive per row
        y = rng.integers(0, 2, size=(N, args.num_classes)).astype(np.float32)
        y[y.sum(axis=1) == 0, 0] = 1.0
        return X, y

    def run(args_inner):
        X, y = args_inner
        model = MultiLabelLinearClassifier()
        model.create_classifier(n_dim=n_dim, num_classes=args.num_classes)
        model.model.to(args.device)
        model.fit(
            X=X, y=y,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device,
        )

    print(f"\nBenchmarking retrain on {args.device}, dataset={args.dataset}")
    run_benchmark(
        operation="retrain",
        device=args.device,
        dataset=args.dataset,
        sizes=sizes,
        setup_fn=setup,
        run_fn=run,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print("Done. Results appended to benchmarks/results.csv")


if __name__ == "__main__":
    main()
