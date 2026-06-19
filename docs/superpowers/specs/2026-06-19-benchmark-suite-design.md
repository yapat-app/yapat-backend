# Benchmark Suite for Research Paper — Design

**Date:** 2026-06-19
**Author:** prathmesh3235
**Status:** Approved (pending spec review)

## Goal

Produce paper-ready performance numbers for the YAPAT active-learning pipeline,
measuring **wall-clock time, throughput, and peak memory** for four operations
across **CPU (local) and GPU (toaster server)**:

1. **Diversity & density** (distance-matrix / nearest-neighbour scoring)
2. **Confidence & uncertainty** (forward pass over the full dataset)
3. **Retraining** (classifier training loop)
4. **Rendering** (number of datapoints in the frontend projection view)

Datasets, in priority order:

- **Tier 1 (core deliverable):** Anuraset, on CPU + GPU
- **Tier 2 (later, if possible):** Anuraset fractions (smaller) + UAnuraset (larger)

## Decisions (locked)

| Decision | Choice |
|---|---|
| Measurement layer | **Direct function calls** now. End-to-end (API→Celery→DB) is a documented TODO. |
| Metrics | Wall-clock time (mean/std), throughput, peak memory |
| Output | CSV (one row per run) + plotting script for scaling-curve figures |
| Data source | **Real cached Anuraset embeddings** (`embeddings.npy`); fractions = seeded row subsample; UAnuraset = its own cache entry |
| Frontend render test | **Playwright + browser Performance API** (Plotly `plotly_afterplot`) |
| Compute isolation | **Refactor to expose a compute-only core**; production behaviour unchanged |

## Architecture

One timing harness, four operation adapters. The runner is the only code that
touches clocks/memory; adapters only know how to `setup(N) -> args` and
`run(args)`.

```
yapat-backend/benchmarks/
  __init__.py
  runner.py          # warmup + N repeats; records time(mean/std), throughput, peak mem; appends CSV
  datasets.py        # dataset name -> (snippet_set_id, embedding_model_id) or cache path; --fraction subsamples rows
  bench_distance.py  # diversity() + density() from active_learning/samplers.py
  bench_inference.py # forward_pass_full() compute core + uncertainty() + _noisy_or_confidence()
  bench_retrain.py   # model.fit() compute core on aligned (X, y)
  plot.py            # reads results.csv -> figures (time vs N per device, throughput, peak mem)
  results.csv        # merged results from all machines
yapat-frontend/benchmarks/
  render_bench.spec.ts  # Playwright: ProjectionView at N = 1k..100k; plotly_afterplot timing -> render.csv
```

### CSV schema (`results.csv`)

```
operation, device, dataset, fraction, N, dim, repeats,
time_mean_s, time_std_s, throughput_per_s, peak_mem_mb, gpu_peak_mem_mb, timestamp, notes
```

- `operation`: `diversity` | `density` | `inference` | `retrain` | `render`
- `device`: `cpu` | `cuda`
- `throughput_per_s`: items/sec (snippets for inference/retrain; query points for distance ops)
- `peak_mem_mb`: process RSS delta via `psutil` (CPU side)
- `gpu_peak_mem_mb`: `torch.cuda.max_memory_allocated()` (blank on CPU runs)

The frontend `render.csv` uses a compatible subset (`operation=render, device=browser, N, time_mean_s, time_std_s`) and is concatenated into the figures by `plot.py`.

## Core runner (`runner.py`)

```python
def benchmark(operation, device, dataset, fraction, setup_fn, run_fn,
              sizes, repeats=5, warmup=1) -> list[Row]:
    for N in sizes:
        args = setup_fn(N)                # build inputs (load + subsample embeddings)
        for _ in range(warmup):
            run_fn(args)                  # warm caches / CUDA kernels
        reset_peak_mem(device)
        times = []
        for _ in range(repeats):
            t0 = perf_counter(); run_fn(args)
            if device == "cuda": torch.cuda.synchronize()
            times.append(perf_counter() - t0)
        record_row(..., mean(times), std(times), N/mean(times), peak_mem(device))
```

- CUDA timing wraps `torch.cuda.synchronize()` so async kernel time is captured.
- `repeats`/`warmup` configurable via CLI; default 5/1.
- One CLI surface per adapter, e.g.:
  `python -m benchmarks.bench_distance --device cuda --dataset anuraset --sizes 1000,5000,20000,full --repeats 5`

## Data layer (`datasets.py`)

- Resolves a dataset name to its embedding cache (reusing `load_embeddings_cached`
  / the `embeddings.npy` on disk). No DB writes; read-only.
- `--fraction f` → deterministic seeded subsample of the loaded rows (so fraction
  runs reuse the Anuraset cache; no re-embedding).
- UAnuraset is a new named entry pointing at its own cache (Tier 2).
- Retrain/inference also load labels: reuse `load_ground_truth_metadata` +
  `align_embeddings_and_labels` for the labelled subset, read-only.

## Compute-only refactor (production behaviour unchanged)

Both refactors extract pure-compute cores that production code and benchmarks call:

1. **Inference** — extract the batch forward-pass loop
   (`run_and_store_inference`, `_inference_helpers.py:315-355`) into:
   ```python
   def forward_pass_full(model, X, batch_size, threshold) -> (features, probs, preds)
   ```
   `run_and_store_inference` then calls this, followed by its existing DB-write
   path. The benchmark calls `forward_pass_full` directly, then `uncertainty()` /
   `_noisy_or_confidence()` on the probs.

2. **Retrain** — `model.fit(X, y, epochs, learning_rate, batch_size, device)` in
   the `model_zoo` classifiers is **already** a clean compute-only method. The
   benchmark loads + aligns `(X_train, y_train)` (read-only), calls
   `model.create_classifier(...)` then `model.fit(...)`. No production refactor
   needed; we only assemble inputs the way `train_from_scratch` does.

3. **Distance** — `diversity(Z_u, Z_l)` and `density(Z_u)` in `samplers.py` are
   already pure tensor functions. Call directly; no refactor.

## Frontend render benchmark (`render_bench.spec.ts`)

- Playwright (headless) mounts `ProjectionView` (`src/components/al/ProjectionView.tsx`)
  with synthetic arrays of N points (`N = 1k, 5k, 10k, 50k, 100k`).
- Hooks Plotly's `plotly_afterplot` event; measures render time via
  `performance.now()` from data-injection to first afterplot, repeated per N.
- Exports `render.csv`. Runs on any machine, independent of backend.
- If `ProjectionView` is not trivially mountable in isolation, a minimal dev
  harness route renders it with injected props (no real backend calls).

## Execution order (fastest path to numbers)

1. Build harness + adapters; validate locally on **CPU** with a small Anuraset
   fraction (fast feedback, correctness check on CSV rows).
2. **Tier 1:** Anuraset × {CPU local, GPU toaster} × {distance, inference, retrain}.
   GPU rows = same scripts run on toaster over SSH; CSVs merge via `device` column.
3. Frontend render test (independent, any machine).
4. **Tier 2:** Anuraset fractions (cheap, reuses cache) + UAnuraset (needs its cache).
5. `plot.py` → figures from merged `results.csv` + `render.csv`.

## Out of scope / TODO

- End-to-end timing (API → Celery → DB) — second measurement layer, deferred.
- Synthetic-embedding fallback — only if a Tier-2 size point has no cache available.
- Statistical accuracy/quality metrics (this suite measures *performance*, not model quality).

## Testing / validation

- Runner unit-tested with a trivial `run_fn` (e.g. `time.sleep`) to confirm
  CSV rows, repeats, and peak-mem capture are correct.
- A CPU smoke run on a small fraction validates each adapter end-to-end before
  committing GPU time on toaster.
- `forward_pass_full` extraction verified by an equality check against the
  pre-refactor output on a small input (features/probs/preds identical).
