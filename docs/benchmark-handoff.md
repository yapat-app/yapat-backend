# Benchmark Implementation Handoff

**Date:** 2026-06-22
**Purpose:** Context for implementing the benchmark suite in the next session.
**Related spec:** `docs/superpowers/specs/2026-06-19-benchmark-suite-design.md`

---

## Why we're doing this

The research paper has a Section 5.2 "Scalability" with four figures (5a–5d) that need
measured numbers to replace [TODO] placeholders. The paper text and TODOs are in the
conversation history — ask the user to paste them if needed.

| Fig | Operation | X-axis | Paper claim |
|---|---|---|---|
| 5a | BirdNET embedding generation | N snippets | O(N), GPU constant-factor speedup |
| 5b | pgvector cosine search | N snippets in DB | O(N), remains interactive |
| 5c | AL scoring (uncertainty + density + diversity) | N unlabeled snippets | O(N), O(N log N), O(N·Nl) |
| 5d | Retraining (linear head only) | Nl labeled snippets | O(Nl), fast on CPU |

Plus a frontend rendering claim (scattergl scales to tens of thousands of points,
thumbnails subsample to ≤ 25,000).

---

## Hardware

**Toaster (GPU server):**
- 4× NVIDIA RTX PRO 6000 Blackwell, ~98 GB VRAM each
- CUDA 13.2, Driver 595.71.05
- 773 GB RAM, ~98 CPU cores
- SSH: `prathmesh@toaster`
- App: `/srv/demos/apps/yapat-tool/`
- AnuraSet raw data: `/srv/demos/shared/datasets/AnuraSet/raw_data/`
  - Contents: `INCT17/ INCT20955/ INCT4/ INCT41/ all_labels_combined_converted.csv metadata.csv`

**Mac (local, CPU benchmarks only):**
- Standard dev machine, no GPU

**Strategy:** GPU benchmarks on toaster, CPU benchmarks on Mac.
Copy `embeddings.npy` from toaster to Mac after ingestion for AL CPU benchmarks.
For CPU embedding timing: run a small fraction on toaster with `NVIDIA_VISIBLE_DEVICES=none`.

---

## What was completed in this session

### 1. GPU TF fix — DONE, deployed on toaster

BirdNET was running on CPU only (confirmed via `nvidia-smi` showing 0% GPU util).
Root cause: `tensorflow` pip package didn't include CUDA; Docker runtime not configured.

**Fixes made (all committed + pushed to main):**

- `requirements.txt` — removed bare `tensorflow`; Dockerfile handles it conditionally
- `Dockerfile` — added conditional TF install:
  - `DEVICE=gpu` → `tensorflow[and-cuda]`
  - `DEVICE=cpu` → `tensorflow-cpu`
  - Added `ldconfig` step to register pip-bundled CUDA libs dynamically (no hardcoded paths)
- `docker-compose.gpu.yml` — added `celery-worker` GPU access + `NVIDIA_VISIBLE_DEVICES=all`
- On toaster: `nvidia-ctk runtime configure --runtime=docker --set-as-default` run

**Verified working:**
```
GPUs found: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Important caveat — PTX JIT compilation:**
RTX PRO 6000 Blackwell (compute capability 12.0a) is not in TF 2.21.0's precompiled
kernels. First TF GPU operation per container lifetime triggers ~30 min JIT compilation.
**Always do one warm-up embedding call before starting any timed benchmark.**

### 2. Docker GPU config on toaster — DONE

Toaster `.env`:
```
DEVICE=gpu
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docker-compose.gpu.yml
PAM_DEFAULT_DEVICE=cuda
```

### 3. What is NOT done yet (implement in next session)

- Anuraset not yet ingested (platform has no datasets)
- `HOST_DATA_ROOT` on toaster needs to be set to `/srv/demos/shared/datasets`
- Stage timer not yet written (URGENT — needed before ingestion)
- Benchmark harness not yet written
- Plot script not yet written

---

## Operation-to-code mapping

### Fig 5a — Embedding (BirdNET)
- **Entry:** `app/tasks/embedding_tasks.py::generate_embeddings_for_recording()`
- **Model:** `app/services/birdnet_model.py::BirdNetEmbedder.embed_batch_from_recording()`
  - Uses `TFSMLayer` wrapping a TF SavedModel at `assets/models/birdnet/`
  - Always loads audio with `librosa`, runs batched TF inference
- **Device:** GPU if `tensorflow[and-cuda]` + CUDA libs visible (now fixed)
- **Measured by:** stage timer inserted into `run_embedding` task

### Fig 5b — pgvector similarity search
- **Entry:** `app/services/embedding_service.py::VectorStore.search()`
- **SQL:** `SELECT ... FROM embedding_vectors WHERE ... ORDER BY vector <=> :query ASC LIMIT :k`
- **Device:** CPU always (Postgres)
- **Measured by:** `bench_similarity.py` direct calls to `VectorStore.search()` at swept N

### Fig 5c — AL scoring
- **Uncertainty:** `active_learning/samplers.py::uncertainty(P)` — entropy over probs
- **Density:** `active_learning/samplers.py::density(Z_u)` — HNSW k-NN (faiss-cpu, CPU only)
- **Diversity:** `active_learning/samplers.py::diversity(Z_u, Z_l)` — flat L2 vs labeled set
- **Forward pass (feeds probs):** `app/services/pam_al/_inference_helpers.py::run_and_store_inference()` line 266
  - **Refactor needed:** extract lines 315–355 (batch loop) into `forward_pass_full(model, X, batch_size, threshold)`
  - Production code calls `forward_pass_full` then continues with DB writes (unchanged)
- **Device for forward pass:** `PAM_DEFAULT_DEVICE` env var → `cuda` on toaster, `cpu` on Mac
- **Measured by:** `bench_inference.py` + `bench_distance.py`

### Fig 5d — Retraining
- **Entry:** `active_learning/model_zoo/linear_multilabel_classifier.py::fit()` and `mlp_multilabel_classifier.py::fit()`
- **Already clean:** `model.fit(X, y, epochs, learning_rate, batch_size, device)` is pure compute, no DB
- **X-axis is Nl (labeled snippets), not N total** — benchmark sweeps labeled set size
- **Device:** `device` param passed to `fit()` → `xb.to(device)`, `yb.to(device)`
- **Measured by:** `bench_retrain.py`

### Frontend rendering
- **Component:** `src/components/al/ProjectionView.tsx`
- **Key constants:** `DISPLAY_MAX_POINTS = 25000` (line 504), uses `scattergl` (line 726)
- **Measured by:** Playwright `render_bench.spec.ts`

---

## What to build (implementation order)

### STEP 1 — Stage timer (do this FIRST, before ingesting Anuraset)

**File:** `benchmarks/stage_timer.py`

```python
import time, csv, os, psutil
from contextlib import contextmanager

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

@contextmanager
def stage_timer(operation: str, device: str, dataset: str, n: int = None):
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    mem_after = proc.memory_info().rss / 1e6
    row = {
        "operation": operation, "device": device, "dataset": dataset,
        "N": n, "time_s": round(elapsed, 3),
        "peak_mem_mb": round(mem_after - mem_before, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_csv_row(row)

def write_csv_row(row: dict):
    fields = ["operation","device","dataset","N","time_s","peak_mem_mb","timestamp"]
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
```

**Insert into production code at these 4 points:**

1. `app/tasks/processing_tasks.py` around `svc.scan_recordings(dataset)`:
```python
from benchmarks.stage_timer import stage_timer
with stage_timer("scan", "cpu", dataset.source_uri):
    new_recs = svc.scan_recordings(dataset)
# after: pass n=len(new_recs) — need to refactor slightly to capture count
```

2. `app/tasks/embedding_tasks.py` in `run_embedding`, after the segmentation loop:
```python
with stage_timer("snippet_gen", "cpu", str(job.dataset_id), n=total_snippets):
    # the existing segmentation loop goes here
```

3. `app/tasks/embedding_tasks.py` in `run_embedding`, around the `chord(task_group)(finalize)` call:
```python
# Note: chord is async so you can't wrap it directly.
# Instead, record start time before chord, record end in finalize_embedding_job.
# Simplest: add timing fields to the EmbeddingJob model or log to CSV from finalize_embedding_job.
```

4. `app/services/pam_al/_embedding_cache.py` in `_build_cache_from_db`:
```python
with stage_timer("cache_build", "cpu", str(snippet_set_id), n=len(snippet_rows)):
    # existing DB load + npy write
```

### STEP 2 — Ingest Anuraset (do after stage timer is in place)

On toaster:
```bash
# 1. Verify HOST_DATA_ROOT in .env
cat .env | grep HOST_DATA_ROOT
# Should be: HOST_DATA_ROOT=/srv/demos/shared/datasets

# 2. Restart api to pick up volume
docker compose up -d --force-recreate api

# 3. Verify path visible
curl http://localhost:8000/api/datasets/available-paths

# 4. Create dataset via UI or API
POST /api/datasets/
{"name": "AnuraSet", "source_uri": "AnuraSet/raw_data", ...}

# 5. Create embedding job
POST /api/datasets/{id}/embeddings
{"embedding_model_id": <birdnet_id>, "window_size": 3.0, "step_size": 3.0, "overlap": 0.0}

# 6. Watch GPU
watch -n 1 nvidia-smi
```

### STEP 3 — Extract inference compute core (small refactor)

**File:** `app/services/pam_al/_inference_helpers.py`

Extract lines 315–355 from `run_and_store_inference` into:
```python
def forward_pass_full(
    model,
    X: np.ndarray,
    batch_size: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure forward pass — no DB writes. Returns (features, probs, preds)."""
    # move the existing batch loop here
    ...
```

`run_and_store_inference` becomes:
```python
features, probs, preds = forward_pass_full(model, X, batch_size, threshold)
# ... existing DB-write code continues unchanged
```

### STEP 4 — Benchmark runner + adapters

**File:** `benchmarks/runner.py`
```python
def benchmark(setup_fn, run_fn, sizes, device, operation, dataset,
              repeats=5, warmup=1):
    for N in sizes:
        args = setup_fn(N)
        for _ in range(warmup):
            run_fn(args)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_fn(args)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
        write_csv_row({
            "operation": operation, "device": device, "dataset": dataset,
            "N": N, "repeats": repeats,
            "time_mean_s": round(statistics.mean(times), 4),
            "time_std_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0,
            "throughput_per_s": round(N / statistics.mean(times), 1),
            "peak_mem_mb": ...,  # psutil RSS delta
            "gpu_peak_mem_mb": gpu_mem,
        })
```

**Adapters:**
- `benchmarks/bench_distance.py` — calls `diversity(Z_u, Z_l)` and `density(Z_u)` from `active_learning/samplers.py`
- `benchmarks/bench_inference.py` — calls `forward_pass_full()` then `uncertainty()`
- `benchmarks/bench_retrain.py` — calls `model.create_classifier()` then `model.fit()` at swept Nl values
- `benchmarks/bench_similarity.py` — calls `VectorStore.search()` at swept N

**Data loading:** All adapters use `load_embeddings_cached()` from
`app/services/pam_al/_embedding_cache.py` to load the real `embeddings.npy`.
Fractions = seeded `np.random.default_rng(42).choice(N, size=int(N*fraction))`.

**CLI pattern:**
```bash
python -m benchmarks.bench_inference --device cuda --dataset anuraset --sizes 1000,5000,20000,full --repeats 5
python -m benchmarks.bench_retrain   --device cpu  --dataset anuraset --sizes 100,500,2000,full --repeats 5
```

### STEP 5 — Plot script

`benchmarks/plot.py` reads `results.csv` → 4-panel matplotlib figure matching paper layout.
One line per device per operation, x=N, y=time_mean_s, shaded band = ±time_std_s.

---

## CSV schema (unified for all stages)

```
operation, device, dataset, N, repeats, time_s (or time_mean_s), time_std_s,
throughput_per_s, peak_mem_mb, gpu_peak_mem_mb, timestamp
```

Stage timer rows use `time_s` (single measurement, no repeats).
Runner rows use `time_mean_s` / `time_std_s` / `repeats`.
Both append to `benchmarks/results.csv`.

---

## Operations that are CPU-only (no GPU comparison possible)

- `density` and `diversity` — use `faiss-cpu`; `faiss-gpu` not installed
- `scan` — pure IO
- `snippet_gen` — pure Python loop
- `cache_build` — DB read + numpy write
- `pgvector search` — lives inside Postgres

Only `embedding` (BirdNET/TF) and `inference`/`retrain` (PyTorch) have meaningful GPU vs CPU comparisons.

---

## Key files for the implementation

```
yapat-backend/
  active_learning/samplers.py                    # diversity(), density(), uncertainty()
  active_learning/model_zoo/
    linear_multilabel_classifier.py              # fit() — pure compute, already clean
    mlp_multilabel_classifier.py                 # fit() — pure compute, already clean
  app/services/pam_al/_inference_helpers.py      # run_and_store_inference() — needs extraction
  app/services/pam_al/_embedding_cache.py        # load_embeddings_cached() — data loader
  app/services/embedding_service.py              # VectorStore.search() — pgvector bench
  app/tasks/embedding_tasks.py                   # run_embedding, generate_embeddings_for_recording
  app/tasks/processing_tasks.py                  # scan_dataset
  benchmarks/                                    # CREATE THIS DIRECTORY
    stage_timer.py
    runner.py
    datasets.py
    bench_distance.py
    bench_inference.py
    bench_retrain.py
    bench_similarity.py
    plot.py
    results.csv

yapat-frontend/
  src/components/al/ProjectionView.tsx           # DISPLAY_MAX_POINTS=25000, scattergl
  benchmarks/
    render_bench.spec.ts                         # Playwright render test
```

---

## Important notes for implementation

1. **PTX warm-up:** First TF GPU call on Blackwell takes ~30 min (JIT compile). Always
   run one dummy embedding before starting any timed embedding benchmark.

2. **Chord timing:** `run_embedding` uses a Celery chord (async fan-out per recording).
   You can't wrap the chord with a context manager. Instead, record `started_at` on the
   `EmbeddingJob` model (already exists) and `completed_at` in `finalize_embedding_job`.
   The stage timer for `embedding` reads those two timestamps.

3. **Fig 5d x-axis is Nl, not N:** Retraining benchmark sweeps the number of *labeled*
   snippets, not total. Load the full embedding matrix, then subsample rows to simulate
   different labeled set sizes.

4. **pgvector benchmark needs DB populated:** `bench_similarity.py` needs the DB to have
   N embedding vectors. Run it after ingestion; different N values come from the fraction
   ingestion runs (Tier 2) or by querying subsets with `LIMIT`.

5. **Mac CPU run:** Copy `embeddings.npy` + `snippets.json` from toaster's embedding
   cache directory (check `_embedding_cache.py::get_cache_dir()` for the path) to Mac.
   Point the benchmark at the local copy.
