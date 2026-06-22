"""
Benchmark: PAM-AL forward pass (inference).

Loads real embeddings from the on-disk cache, sweeps N (total snippets),
and times the forward pass on the given device.

Usage:
    python -m benchmarks.bench_inference \
        --snippet-set-id 7 --embedding-model-id 1 \
        --device cuda --dataset anuraset \
        --sizes 1000,5000,10000,full --repeats 5
"""

import argparse
import sys

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snippet-set-id", type=int, required=True)
    p.add_argument("--embedding-model-id", type=int, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,full")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--checkpoint-path", required=True,
                   help="Path to .pt checkpoint file (model weights)")
    p.add_argument("--label-order", default=None,
                   help="Comma-separated class names (must match checkpoint)")
    return p.parse_args()


def main():
    args = _parse_args()

    # --- Load checkpoint ---
    import torch
    from active_learning.model_zoo.linear_multilabel_classifier import MultiLabelLinearClassifier

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    hyper = ckpt.get("hyperparameters", {})
    label_order = args.label_order.split(",") if args.label_order else hyper.get("label_order", [])
    n_dim = hyper.get("n_dim") or ckpt.get("n_dim")
    num_classes = len(label_order) or hyper.get("num_classes") or ckpt.get("num_classes")

    model = MultiLabelLinearClassifier()
    model.create_classifier(n_dim=n_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device)
    model.eval()

    # --- Load embeddings from cache ---
    from app.services.pam_al._embedding_cache import get_cache_dir, _embeddings_path
    cache_dir = get_cache_dir(args.snippet_set_id, args.embedding_model_id)
    X_full = np.load(_embeddings_path(cache_dir), mmap_mode="r").astype(np.float32)
    N_full = X_full.shape[0]
    print(f"Loaded {N_full:,} embeddings from {cache_dir}")

    # --- Parse sizes ---
    rng = np.random.default_rng(42)
    sizes = []
    for s in args.sizes.split(","):
        s = s.strip()
        if s == "full":
            sizes.append(N_full)
        else:
            sizes.append(min(int(s), N_full))

    # --- Run benchmark ---
    from benchmarks.runner import run_benchmark
    from app.services.pam_al._inference_helpers import forward_pass_full
    batch_size = int(hyper.get("batch_size") or 256)

    def setup(N):
        idx = rng.choice(N_full, size=N, replace=False) if N < N_full else np.arange(N_full)
        return X_full[idx].copy()

    def run(X):
        forward_pass_full(model, X, batch_size=batch_size, threshold=0.5)

    print(f"\nBenchmarking inference on {args.device}, dataset={args.dataset}")
    run_benchmark(
        operation="inference",
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
