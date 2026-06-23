"""
Benchmark: BirdNET embedding forward pass.

Uses synthetic audio (random noise) to isolate model compute from disk I/O.
Sweeps N (number of snippets) on CPU or GPU via TF device placement.

Usage:
    python -m benchmarks.bench_embedding \
        --dataset anuraset --sizes 1000,5000,10000,20000,30687 --repeats 3 --device cpu

Runs in yapat-celery-worker (TF is not installed in pam-al worker).
"""

import argparse
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="anuraset")
    p.add_argument("--sizes", default="1000,5000,10000,20000,30687")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    return p.parse_args()


def main():
    args = _parse_args()

    import tensorflow as tf
    from app.services.birdnet_model import BirdNetEmbedder
    from benchmarks.runner import run_benchmark

    tf_device = "/GPU:0" if args.device == "gpu" else "/CPU:0"

    print("Loading BirdNET model...")
    model = BirdNetEmbedder.instance()
    # Warmup to trigger TF graph compilation
    _dummy = np.zeros((1, BirdNetEmbedder.WINDOW_SAMPLES), dtype=np.float32)
    with tf.device(tf_device):
        model(_dummy)
    print("BirdNET model loaded.")

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    rng = np.random.default_rng(42)

    def setup(N):
        # Generate per-N to avoid OOM (30K × 144K × 4B = ~17 GB if pre-allocated)
        return rng.uniform(-0.1, 0.1, size=(N, BirdNetEmbedder.WINDOW_SAMPLES)).astype(np.float32)

    def run(audio_batch):
        N = audio_batch.shape[0]
        with tf.device(tf_device):
            for start in range(0, N, args.batch_size):
                chunk = audio_batch[start:start + args.batch_size]
                model(chunk)

    print(f"\nBenchmarking embedding on {args.device}, dataset={args.dataset}")
    run_benchmark(
        operation="embedding",
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
