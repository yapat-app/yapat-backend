# Setting up a reference dataset

Companion how-to for [reference-data-pool-design.md](./reference-data-pool-design.md).
Answers the practical question that design doc glosses over: how annotations
actually have to line up with snippets.

## The short answer

Two ways to feed in a reference dataset, both supported:

1. **Pre-chunked clips** — one audio file = one labeled event = one
  snippet. Simple, but requires you to cut the clips yourself first.
2. **Arbitrary-length recordings + precise onset/offset per event** — the
  tool auto-windows the recording the normal way, and each event's label
  only gets attached to the snippet(s) it actually overlaps in time. This
  needed a small fix to the alignment logic (below) but is now the
  recommended path if your source data is already annotated with event
  timestamps.

Either way, embeddings still come from BirdNET (the only embedding model
seeded in this codebase — 1024-dim, `window_size=3.0`, `step_size=3.0`,
`overlap=0.0`, all flagged `requires_fixed_*`), so segmentation is always
fixed 3-second, non-overlapping windows regardless of which path you use —
you don't control that part either way.

## Why this needed a fix

Segmentation happens in `run_embedding` (`app/tasks/embedding_tasks.py`),
and it's a genuine sliding window over each `Recording.duration`:

```python
t = 0.0
while t + window <= duration:
    Snippet(recording_id=rec.id, start_time=t, end_time=t + window, ...)
    t += step
```

Ground-truth matching (`align_embeddings_and_labels` in
`app/services/pam_al/_data_helpers.py`) looks up CSV events by
`Recording.file_name`, which is constant across every snippet cut from that
recording. Originally it ignored each snippet's actual `start_time`/
`end_time` entirely and applied *every* event's labels in the file to
*every* snippet from that file — fine for pre-chunked one-clip-one-event
data, silently wrong for a long recording with several time-localized
events.

That's fixed now: `align_embeddings_and_labels` takes each event's
`start_time`/`end_time` (parsed from `onset`/`offset`, `start_time`/
`end_time`, or `min_t`/`max_t` columns — whichever your CSV provides) and
only applies its labels to snippets whose own `[start_time, end_time)`
window actually overlaps it. Both are measured against the same
recording's clock, so they're directly comparable. If an event has no time
info at all (no such column in the CSV), it still falls back to whole-file
matching — that's the correct behavior for pre-chunked clips, where the
"event" legitimately spans the entire (already-trimmed) file.

## Step by step

**1. Decide which of the two input shapes you have**, and prepare audio +
CSV accordingly.

### Option A — pre-chunked clips (one file = one event)

Each audio file is already a 3.0-second labeled clip. Multi-label is fine
(a clip can have more than one species) — that's encoded in the CSV, not
the audio.

```
reference/anuraset_train/
  INCT17_20191002_040000_0_3.wav
  INCT17_20191002_040000_3_6.wav
  ...
  pam_metadata.csv
```

*Wide format* (one binary column per species):

```csv
fname,min_t,max_t,BOAALB,DENMIN,PHYCUV
INCT17_20191002_040000,0,3,1,0,0
INCT17_20191002_040000,3,6,0,1,1
```

With `fname`+`min_t`+`max_t` present, the matching key becomes
`{fname}_{min_t}_{max_t}.wav` — so the CSV row above expects a file named
exactly `INCT17_20191002_040000_0_3.wav`. Integer-valued `min_t`/`max_t` get
formatted without decimals (`0`, not `0.0`) when building that key. Because
each file is already trimmed to exactly one event, `min_t`/`max_t` here
only serve to build the filename to match against — every snippet of that
file (there will be exactly one, since the file's duration equals the
window) gets the label.

*Long format* (one row per species per clip):

```csv
fname,min_t,max_t,species
INCT17_20191002_040000,0,3,BOAALB
INCT17_20191002_040000,3,6,DENMIN
INCT17_20191002_040000,3,6,PHYCUV
```

Multiple rows sharing the same key get OR'd together (`np.maximum`), so this
is how you give one clip more than one label in long format.

### Option B — arbitrary-length recordings + precise onset/offset

Keep your recordings whole. The CSV carries one row per labeled event, with
onset/offset times measured against the *whole recording*, not per-snippet:

```csv
fname,onset,offset,species
INCT17_20191002_040000.wav,124.3,127.9,BOAALB
INCT17_20191002_040000.wav,124.3,127.9,DENMIN
INCT17_20191002_040000.wav,340.0,341.5,PHYCUV
```

(`start_time`/`end_time` column names work identically to `onset`/`offset`
— use whichever your source data already has.) Wide format with binary
species columns works the same way, just add `onset`/`offset` instead of
`min_t`/`max_t`.

Since there's no `min_t`/`max_t` pair, matching falls back to a direct
filename comparison — `fname` here must equal `Recording.file_name` exactly
(include the extension: `INCT17_20191002_040000.wav`, not
`INCT17_20191002_040000`). The recording gets auto-windowed into 3.0s
snippets as usual (`[0,3), [3,6), [6,9), ...`), and each event's labels are
only attached to the snippet(s) whose window actually overlaps
`[onset, offset)`. An event spanning a snippet boundary (e.g. `[2.5, 4.0)`)
correctly lands on both the `[0,3)` and `[3,6)` snippets.

*Simpler alternative for either option*: skip `fname` and use a plain
filename column instead (`file_name`, `sample_name`, `recording_file`,
`recording_name`, or `file_path`) — same direct string match against
`Recording.file_name`. Useful if you'd rather name files however you like.

An optional `subset` column is allowed but ignored for reference data —
every labeled row gets used regardless of subset (that's a deliberate
decision: reference data is never evaluated inside the tool, so there's no
need to hold out a test split).

**2. Lay out the folder** under `DATA_ROOT`, containing your audio (clips or
whole recordings, per whichever option you picked) plus `pam_metadata.csv`.

**3. Register the dataset**, marked as reference-only:

```
POST /api/datasets/
{
  "name": "AnuraSet reference pool",
  "source_uri": "reference/anuraset_train",
  "dataset_type": "PAM",
  "is_reference": true,
  "team_id": null
}
```

This auto-dispatches the scan task, which creates one `Recording` row per
audio file.

**4. Generate embeddings** using the *same* embedding model your target
dataset(s) use (mixing across embedding spaces is refused by the
reference-pool loader, which filters on `embedding_model_id`):

```
POST /api/datasets/{reference_dataset_id}/embeddings
{ "embedding_model_id": <birdnet_model_id> }
```

Since BirdNET's window/step are fixed, this always produces 3.0s snippets —
exactly one per file for Option A, several per file for Option B depending
on recording length. No `window_size`/`step_size` override needed (and
none would apply). This also sets `default_snippet_set_id` on the dataset
automatically, since none existed yet — required, because
`load_reference_pool_training_data` reads `ref_ds.default_snippet_set_id`.

**5. Sanity-check the result** before linking it to anything. For Option A,
`recording_count` should equal your file count and each recording should
have exactly one snippet. For Option B, check the celery worker logs from
the training run once you've triggered it (step 7) — the alignment
summary logs `Matched by file_name`, `Time-filtered (non-overlapping)
events skipped`, and `Positive aligned samples`, which tells you whether
your onset/offset times actually landed inside real snippet windows.

**6. Link it** to whichever dataset or team should draw on it:

```
POST /api/reference-links/
{ "reference_dataset_id": <id>, "dataset_id": <target_dataset_id> }
```

or `{ "team_id": <team_id> }` instead of `dataset_id` to share it across
every dataset under that team.

**7. Trigger a cold-start or retrain** on the linked dataset and check the
resulting checkpoint's `hyperparameters.reference_pool` (or the retrain
job's `result_metrics.reference_pool`) for `reference_sample_count` and any
`skipped` entries with reasons — that confirms the mix actually happened
and tells you if anything got silently skipped (missing metadata file, no
default snippet set, no embeddings for that model).

## Things to still watch for

- **Filename must match exactly**, extension included, when there's no
  `min_t`/`max_t` pair to build a key from — `fname` (or whichever filename
  column you used) has to equal `Recording.file_name` character for
  character.
- **An event with no time columns at all** (Option A without any
  onset/offset/min_t/max_t) still applies to every snippet of its file.
  That's correct for a genuinely single-event clip, but if you meant to use
  Option B and forgot the time columns, you'll silently get whole-file
  matching instead of time-restricted matching — check the alignment log
  summary if results look off.
- **Snippets are always exactly 3.0s** (BirdNET's fixed window). An event
  shorter than that still gets attached to whichever full 3.0s window(s) it
  overlaps; there's no partial-window cropping.
