# Reference Data Pool

Branch: `finalUserStudy`. Not harmonized with `main` yet. Implemented
2026-07-24; model-complexity bump (default classifier → MLP) tracked
separately, not part of this change.

For how to actually prepare and register a reference dataset (audio +
`pam_metadata.csv` layout, the one-clip-one-snippet constraint, embedding
job setup), see [reference-data-pool-setup-guide.md](./reference-data-pool-setup-guide.md).

## Goal

Give every dataset access to a persistent pool of pre-labeled, out-of-tool
reference embeddings that gets mixed into every cold-start and retrain, so
the classifier keeps broad species coverage instead of drifting toward
whatever narrow slice of species the current annotation session happens to
cover. This replaces "pre-train a model outside the tool" as the way to seed
and protect against catastrophic forgetting.

## How it works

A reference dataset is an ordinary `Dataset` row (`is_reference=True`) that
goes through the normal ingest pipeline -- audio on disk under `DATA_ROOT`,
registered via `POST /api/datasets/`, scanned/segmented/embedded like any
other dataset -- paired with a `pam_metadata.csv`, same format
`load_ground_truth_metadata` already parses for cold-start bootstrap
(`fname`/`min_t`/`max_t` or one binary column per species, or a
`species`/`label` column; optional `subset` column, ignored here -- all
labeled rows are used regardless of subset).

By default the CSV is expected at `{source_uri}/pam_metadata.csv`, but
`Dataset.reference_metadata_path` can override that per-dataset: a bare
filename still resolves within `source_uri`, while a path containing `/`
resolves relative to `DATA_ROOT` instead, fully independent of
`source_uri`. This keeps a shared raw-audio corpus (e.g. an upstream
benchmark's own folder) untouched by any given reference pool's metadata --
multiple reference `Dataset` rows can point `source_uri` at the *same*
underlying audio folder while each supplying its own
`reference_metadata_path` elsewhere under `DATA_ROOT`, since `Dataset`
uniqueness is only on `(team_id, source_uri)` and admin-owned reference
datasets share `team_id=NULL`.

`is_reference=True` datasets:
- **only register recordings the metadata CSV actually mentions.**
  `DatasetService.scan_recordings` normally walks the entire `source_uri`
  tree and registers every audio file it finds; for reference datasets it
  first filters that list down to files whose basename is a key in
  `pam_metadata.csv` (`_filter_to_referenced_files`, reusing
  `get_referenced_filenames` / `load_ground_truth_metadata` so the filter
  can never drift from what training-time matching actually does). This
  means `source_uri` can point at a much larger directory -- including one
  with unrelated subfolders -- without wasting scan/embed compute on files
  the pool will never use. Falls back to registering everything if no CSV
  exists yet, rather than silently registering zero.
- are excluded from `GET /api/datasets/` unless `include_reference=true` is
  passed, so they never show up as something to annotate
- are still full datasets otherwise -- same scan/snippet/embed pipeline,
  same `EmbeddingVector` storage, no new ingestion code

A target dataset (or every dataset under a team) opts into one or more
reference datasets via `DatasetReferenceLink`
(`app/models/reference_link.py`, table `dataset_reference_links`):
either `dataset_id` (scoped to one dataset) or `team_id` (shared by every
dataset under that team) is set, never both. Managed via
`POST/GET/DELETE /api/reference-links/` (admin-only for now).

At training time (`PAMActiveLearningService._mix_in_reference_pool`, called
from all three cold-start/retrain entry points in
`app/services/pam_al/service.py`: `execute_train_from_scratch` (async,
metadata-CSV-driven, Celery-backed -- what the `/train-from-scratch` API
endpoint dispatches), `train_from_scratch` (sync -- invoked internally from
`_submit_bootstrap_feedback` once accumulated user feedback crosses the
retrain threshold with no metadata CSV involved), and `_execute_retrain`
(manual/auto retrain)):

1. Resolve linked reference datasets for the target dataset
   (`data_h.get_reference_dataset_ids` -- union of dataset-scoped + the
   dataset's team-scoped links).
2. For each, using its **own** species vocabulary discovered from its CSV
   (`data_h.scan_metadata_species` -- a single reference dataset's CSV
   rarely covers every species across every dataset that references it),
   load its embeddings (must share the training run's `embedding_model_id`
   -- mismatched embedding spaces are never mixed) and labels via the
   existing `load_embeddings` / `load_ground_truth_metadata` /
   `align_embeddings_and_labels` functions, unchanged.
3. Union all reference-only species onto the target dataset's own species
   list to get a unified label space; zero-pad the target's own `y` matrix
   to match, and scatter each reference dataset's labels into the unified
   column positions.
4. Concatenate reference `X`/`y` onto the target's own `X_train`/`y_train`
   before `filter_and_balance_classes` -- always on, full pool, no sampling
   ratio (v1 policy).
5. Reference-pool rows carry `snippet_id=None` in the returned list (they
   belong to a different dataset's `snippets` rows). Cold-start's
   `ann_h.store_snippet_annotations` call -- which persists CSV rows as this
   dataset's own `GROUND_TRUTH` annotations -- filters these out first, so a
   reference dataset's snippet ids never get written into another dataset's
   annotation history.
6. Reference-pool provenance (`reference_dataset_ids`,
   `reference_sample_count`, per-dataset counts, and any skipped datasets
   with reasons) is recorded on the new checkpoint's `hyperparameters` and
   the retrain job's `result_metrics` under the `reference_pool` key.

Reference datasets missing a metadata CSV, a `default_snippet_set_id`, or
embeddings for the training run's `embedding_model_id` are skipped with a
logged reason rather than failing the whole training run -- reference data
is supplementary, never required.

## Schema

- `datasets.is_reference` (boolean, default false, indexed)
- `dataset_reference_links`: `id`, `reference_dataset_id` (FK datasets,
  cascade), `dataset_id` (FK datasets, cascade, nullable), `team_id` (FK
  teams, cascade, nullable), `created_at`. Postgres `CHECK` enforces exactly
  one of `dataset_id`/`team_id`; two unique constraints
  (`reference_dataset_id`, `dataset_id`) and (`reference_dataset_id`,
  `team_id`) prevent duplicate links within each scope.
- Migration: `alembic/versions/d3e1f0a9b7c4_reference_data_pool.py`
  (`down_revision = a0c672de4d81`, current head at time of writing).

`align_embeddings_and_labels` (`_data_helpers.py`) was also made time-aware:
an event only applies to a snippet if its `start_time`/`end_time` actually
overlaps that snippet's window, when the metadata CSV provides real event
times (falls back to whole-file matching when it doesn't, unchanged from
before). This means reference data no longer has to be pre-chunked to
one-clip-one-event — arbitrary-length recordings with precise onset/offset
per event now align correctly. See the setup guide for both supported
input shapes.

## API

- `GET /api/datasets/?include_reference=true` -- admin/ops visibility into
  reference datasets; omitted by default everywhere else.
- `POST /api/reference-links/` -- `{reference_dataset_id, dataset_id?,
  team_id?}` (exactly one of the two). Validates the reference dataset has
  `is_reference=True` and the target dataset/team exists. Idempotent
  (returns the existing link if one already matches).
- `GET /api/reference-links/?dataset_id=&team_id=&reference_dataset_id=`
- `DELETE /api/reference-links/{link_id}`

All three are admin-only for now -- reference datasets are provisioned/ops-
managed for the user study, not self-service yet.

## Known gaps / follow-ups

- No UI yet for browsing pool contents/label distribution or managing links
  -- admin API + `is_reference` flag only.
- `filter_and_balance_classes` treats every row equally regardless of
  source. If a reference pool is large relative to a dataset's own
  annotations, it could dominate training -- fine for v1 (mixing policy is
  "always on, fixed pool"), worth revisiting once real proportions are
  observed in the pilot.
- Model-complexity bump (default classifier → `PAM_MLP_MULTILABEL`) is a
  separate, smaller change, not included here.
