# Benchmark Suite for Research Paper — Design

**Date:** 2026-06-19
**Author:** prathmesh3235
**Status:** Approved (pending spec review)

## Goal

Produce paper-ready performance numbers for the **entire** YAPAT pipeline —
from raw audio to active learning — measuring **wall-clock time, throughput,
and peak memory** across **CPU (local) and GPU (toaster server)**.

**Upstream / ingestion stages** (one-shot, end-to-end with in-code stage timers):

1. **Dataset scan** (`scan_recordings` — discover recordings + read durations)
2. **Snippet generation** (segmentation loop inside `run_embedding`)
3. **Embedding generation** (`generate_embeddings_for_recording` forward pass)
4. **Projection / FPV** (`generate_fpv_for_dataset` — render input)
5. **Embedding cache build** (`load_embeddings_cached`: DB → `embeddings.npy`)

**Active-learning compute stages** (direct-call harness):

6. **Diversity & density** (distance-matrix / nearest-neighbour scoring)
7. **Confidence & uncertainty** (forward pass over the full dataset)
8. **Retraining** (classifier training loop)

**Frontend:**

9. **Rendering** (number of datapoints in the frontend projection view)

Key efficiency principle: **ingesting Anuraset on toaster IS the upstream
benchmark.** The platform needs Anuraset ingested anyway; that single run yields
the upstream timings (stages 1–5) *and* the cached embeddings + ground-truth CSVs
that stages 6–8 consume. Nothing is run twice.

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
| Upstream driver | **Real API calls** (`POST /datasets`, `POST /embeddings`) + in-code stage timers writing to `results.csv` |
| Embed config | Platform standard: **BirdNET, window=3.0s, step=3.0s, overlap=0** (confirm against `GET /embeddings/embedding-models` at ingestion) |
| Data access | Mount host `/srv/demos/shared/datasets/AnuraSet` into API + embedding-worker containers under `DATA_ROOT` (`/data`); `source_uri = AnuraSet/raw_data` |

## Architecture

One timing harness, four operation adapters. The runner is the only code that
touches clocks/memory; adapters only know how to `setup(N) -> args` and
`run(args)`.

```
yapat-backend/benchmarks/
  __init__.py
  runner.py          # warmup + N repeats; records time(mean/std), throughput, peak mem; appends CSV
  stage_timer.py     # context manager: times one upstream stage, appends a results.csv row
  datasets.py        # dataset name -> (snippet_set_id, embedding_model_id) or cache path; --fraction subsamples rows
  bench_distance.py  # diversity() + density() from active_learning/samplers.py
  bench_inference.py # forward_pass_full() compute core + uncertainty() + _noisy_or_confidence()
  bench_retrain.py   # model.fit() compute core on aligned (X, y)
  plot.py            # reads results.csv -> figures (time vs N per device, throughput, peak mem)
  results.csv        # merged results from all machines + upstream stages
yapat-frontend/benchmarks/
  render_bench.spec.ts  # Playwright: ProjectionView at N = 1k..100k; plotly_afterplot timing -> render.csv
```

### Upstream stage timer (`stage_timer.py`)

A minimal context manager wrapping each one-shot pipeline stage so it self-reports
to the same `results.csv` schema:

```python
with stage_timer("scan", device="cpu", dataset="anuraset", n=None):
    new_recs = svc.scan_recordings(dataset)   # n set afterward = len(new_recs)
```

Inserted (production code, behaviour unchanged) around:
- `DatasetService.scan_recordings` — stage `scan` (N = recordings)
- the segmentation loop in `run_embedding` — stage `snippet_gen` (N = snippets)
- the per-recording embedding call in `run_embedding` — stage `embedding` (N = snippets)
- `generate_fpv_for_dataset` — stage `fpv` (N = snippets)
- first `load_embeddings_cached` build path — stage `cache_build` (N = snippets)

These stages run **end-to-end** (they write to DB) and are measured once per
dataset during ingestion — there is no separate "compute-only" version because
the whole stage *is* the operation of interest.

### CSV schema (`results.csv`)

```
operation, device, dataset, fraction, N, dim, repeats,
time_mean_s, time_std_s, throughput_per_s, peak_mem_mb, gpu_peak_mem_mb, timestamp, notes
```

- `operation`: `scan` | `snippet_gen` | `embedding` | `fpv` | `cache_build` | `diversity` | `density` | `inference` | `retrain` | `render`
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

## Operational flow (exact, fastest path to numbers)

**Phase 0 — one-time setup (toaster)**
1. Mount `/srv/demos/shared/datasets/AnuraSet` into the API + embedding-worker
   containers under `DATA_ROOT` (`- /srv/demos/shared/datasets/AnuraSet:/data/AnuraSet:ro`).
   Verify `GET /datasets/available-paths` lists `AnuraSet/raw_data`.
2. Add `stage_timer.py` + the inference `forward_pass_full` extraction; build the
   harness. Smoke-test on CPU with a small fraction.

**Phase 1 — ingest Anuraset = upstream benchmark (toaster)**
3. `POST /datasets/` with `source_uri = "AnuraSet/raw_data"` → scan. **→ stage `scan`.**
4. `POST /datasets/{id}/embeddings` (BirdNET, window=3, step=3, overlap=0) →
   `run_embedding` segments + embeds + finalizes; FPV generated.
   **→ stages `snippet_gen`, `embedding`, `fpv`.**
5. First AL load / first distance run builds `embeddings.npy`. **→ stage `cache_build`.**
   *Output: upstream timings + cached embeddings + ground-truth CSVs.*

**Phase 2 — AL compute benchmark (toaster)**
6. `python -m benchmarks.bench_{distance,inference,retrain} --device cuda --dataset anuraset --sizes 1000,5000,20000,full`
7. Repeat with `--device cpu` (same machine, identical data → clean CPU/GPU comparison).

**Phase 3 — frontend rendering (any machine)**
8. `npx playwright test benchmarks/render_bench.spec.ts` → `render.csv`.

**Phase 4 — Tier 2 (later, if time)**
9. Fractions: rerun Phase 2 with `--fraction 0.1/0.25/0.5` (reuses cache, no re-embedding).
10. UAnuraset: repeat Phases 1–2 as its own dataset/cache entry.

**Phase 5 — figures**
11. `python -m benchmarks.plot` → scaling curves + throughput + peak-mem figures
    from merged `results.csv` + `render.csv`.

### CPU-vs-GPU meaningfulness (what to expect)

| Stage | Metric | CPU vs GPU meaningful? |
|---|---|---|
| scan | time, recordings/s | No (IO/CPU) |
| snippet_gen | time, snippets/s | No |
| embedding | time, snippets/s, peak mem | **Yes** (if embed model runs on GPU) |
| fpv | time | Mostly CPU |
| cache_build | time | No |
| distance | time, query-pts/s, peak mem | **Yes** |
| inference | time, snippets/s, GPU mem | **Yes** |
| retrain | time, samples/s, GPU mem | **Yes** |
| render | time vs N | Browser only |

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
