#!/usr/bin/env python3
"""
Seed a synthetic PAM dataset straight into Postgres, for load-testing at a
scale we have no real data for (2-3 M snippets).

Everything the annotation hub reads is generated: recordings (with the
location/date/time metadata the sidebar filters on), snippets, a base model
checkpoint, one ``al_predictions`` row per snippet, dataset-level ``fpv_vis``
projection coordinates, 1024-d ``embedding_vectors`` (so retraining and
similarity search are exercisable), and a handful of ground-truth labels.

The data is *correlated*, not uniform noise: every snippet is assigned a
dominant species from a Zipf distribution, and its probabilities, projection
coordinates and embedding are all drawn around that species. So the feature
projection shows real clusters, the score histograms are skewed the way real
inference output is, and filtering by a species actually selects a blob.

Writes go through ``COPY`` (binary for the vectors), never the ORM, and ids are
allocated in explicit ranges so the passes stay independent and restartable.

Run it inside the api container, which has the deps and the /app/models_AL
mount::

    docker compose exec api python scripts/seed_mock_dataset.py \\
        --recordings 15000 --snippets-per-recording 200 --labels 24

Start small (``--recordings 250``) and check the hub before committing to a
multi-hour run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import struct
import sys
import time
from contextlib import contextmanager
from typing import Iterable, Sequence

import numpy as np
import psycopg2

# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1024  # fixed by app.models.embedding.VectorType
SCORE_THRESHOLD = 0.5  # probability at/above which a label is "predicted"
PROJECTION_METHODS = ("pca", "umap", "tsne")  # isomap left NULL on purpose
DEFAULT_STEPS = ("catalog", "recordings", "snippets", "predictions", "projection", "embeddings", "labels", "finalize")

_SYL_A = ("DEN", "RHI", "LEP", "SCI", "AME", "BOA", "PHY", "HYL", "ELA", "TUR",
          "MYI", "PIT", "ZON", "COL", "CRY", "ADE", "ISC", "PSE", "TRA", "VAN")
_SYL_B = ("MIN", "NAH", "ICT", "FUS", "ALT", "PIC", "ALB", "CIN", "RUF", "VIR",
          "NIG", "LUT", "GRI", "CAE", "PAL", "SAT", "BRE", "LON", "OCH", "VER")


def species_codes(count: int) -> list[str]:
    """Deterministic 6-letter codes in the style of the real label configs."""
    codes: list[str] = []
    for i in range(count):
        codes.append(_SYL_A[i % len(_SYL_A)] + _SYL_B[(i // len(_SYL_A) + i) % len(_SYL_B)])
    # Guard against collisions once count exceeds 20*20.
    seen: dict[str, int] = {}
    out: list[str] = []
    for code in codes:
        n = seen.get(code, 0)
        seen[code] = n + 1
        out.append(code if n == 0 else f"{code[:4]}{n:02d}")
    return out


# ── COPY helpers ─────────────────────────────────────────────────────────────


def esc(value) -> str:
    """Escape one field for COPY ... FORMAT text."""
    if value is None:
        return r"\N"
    s = str(value)
    return (
        s.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def copy_text(cur, table: str, columns: Sequence[str], rows: Iterable[Sequence]) -> int:
    """COPY an iterable of value tuples into ``table``. Returns the row count."""
    buf = io.StringIO()
    n = 0
    for row in rows:
        buf.write("\t".join(esc(v) for v in row))
        buf.write("\n")
        n += 1
    if n == 0:
        return 0
    buf.seek(0)
    cur.copy_expert(
        f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT text)", buf
    )
    return n


_BINARY_HEADER = b"PGCOPY\n\377\r\n\0" + struct.pack(">ii", 0, 0)
_BINARY_TRAILER = struct.pack(">h", -1)


def _vector_copy_dtype(dim: int) -> np.dtype:
    """Row layout for a binary COPY into embedding_vectors.

    Big-endian throughout, unaligned, matching Postgres' binary COPY wire
    format; the vector field carries pgvector's own header (dim, unused).
    """
    return np.dtype(
        [
            ("nfields", ">i2"),
            ("l_id", ">i4"), ("id", ">i4"),
            ("l_snip", ">i4"), ("snippet_id", ">i4"),
            ("l_job", ">i4"), ("job_id", ">i4"),
            ("l_model", ">i4"), ("model_id", ">i4"),
            ("l_dim", ">i4"), ("dim", ">i4"),
            ("l_vec", ">i4"), ("v_dim", ">i2"), ("v_unused", ">i2"), ("vec", ">f4", (dim,)),
        ],
        align=False,
    )


def copy_embeddings(cur, ids, snippet_ids, job_id: int, model_id: int, vectors: np.ndarray) -> int:
    """Binary COPY of a chunk of embedding vectors (text COPY is far too slow)."""
    m, dim = vectors.shape
    rec = np.zeros(m, dtype=_vector_copy_dtype(dim))
    rec["nfields"] = 6
    rec["l_id"] = rec["l_snip"] = rec["l_job"] = rec["l_model"] = rec["l_dim"] = 4
    rec["id"] = ids
    rec["snippet_id"] = snippet_ids
    rec["job_id"] = job_id
    rec["model_id"] = model_id
    rec["dim"] = dim
    rec["l_vec"] = 4 + 4 * dim  # pgvector header + the float4 payload
    rec["v_dim"] = dim
    rec["v_unused"] = 0
    rec["vec"] = vectors
    payload = io.BytesIO(_BINARY_HEADER + rec.tobytes() + _BINARY_TRAILER)
    cur.copy_expert(
        "COPY embedding_vectors (id, snippet_id, embedding_job_id, embedding_model_id, dim, vector) "
        "FROM STDIN WITH (FORMAT binary)",
        payload,
    )
    return m


def next_id(cur, table: str) -> int:
    cur.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}")
    return int(cur.fetchone()[0])


def fix_sequence(cur, table: str) -> None:
    cur.execute(
        "SELECT setval(pg_get_serial_sequence(%s, 'id'), COALESCE((SELECT MAX(id) FROM "
        + table
        + "), 1))",
        (table,),
    )


HNSW_INDEX = "embedding_vectors_vector_cosine_idx"
HNSW_DDL = (
    f"CREATE INDEX CONCURRENTLY {HNSW_INDEX} ON embedding_vectors "
    "USING hnsw (vector vector_cosine_ops)"
)


def hnsw_index_exists(cur) -> bool:
    cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (HNSW_INDEX,))
    return cur.fetchone() is not None


def drop_hnsw_index(cur) -> bool:
    """The pgvector HNSW index is maintained per inserted row.

    Leaving it in place caps the embedding COPY at a couple of hundred rows a
    second — days, at 3 M. Dropping it first and rebuilding once afterwards is
    the only workable order.
    """
    if not hnsw_index_exists(cur):
        return False
    t0 = time.time()
    cur.execute(f"DROP INDEX {HNSW_INDEX}")
    print(f"  dropped {HNSW_INDEX} ({time.time() - t0:.0f}s)", flush=True)
    return True


# ── Progress ─────────────────────────────────────────────────────────────────


class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.done = 0
        self.start = time.time()
        self._last = 0.0

    def advance(self, n: int) -> None:
        self.done += n
        now = time.time()
        if now - self._last < 5.0 and self.done < self.total:
            return
        self._last = now
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        print(
            f"  {self.label}: {self.done:,}/{self.total:,} "
            f"({self.done / max(self.total, 1) * 100:5.1f}%) "
            f"{rate:,.0f} rows/s  elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m",
            flush=True,
        )

    def finish(self) -> None:
        elapsed = time.time() - self.start
        print(f"  {self.label}: {self.done:,} rows in {elapsed / 60:.1f} min", flush=True)


@contextmanager
def step(name: str, enabled: bool):
    if not enabled:
        print(f"[skip] {name}", flush=True)
        yield False
        return
    print(f"[run ] {name}", flush=True)
    t0 = time.time()
    yield True
    print(f"[done] {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ── Catalog rows ─────────────────────────────────────────────────────────────


def ensure_admin(cur, username: str) -> int:
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(
            f"No user {username!r}. Register one through the API first "
            f"(POST /api/auth/register) and promote it to ADMIN."
        )
    return int(row[0])


def ensure_embedding_model(cur) -> int:
    cur.execute("SELECT id FROM embedding_models WHERE name = 'birdnet' ORDER BY id LIMIT 1")
    row = cur.fetchone()
    if row is not None:
        return int(row[0])
    cur.execute(
        "INSERT INTO embedding_models (name, version, description, window_size, step_size, overlap, "
        "requires_fixed_window, requires_fixed_step, requires_fixed_overlap) "
        "VALUES ('birdnet', '2.4', 'mock', 3, 3, 0, 1, 1, 1) RETURNING id"
    )
    return int(cur.fetchone()[0])


def write_checkpoint_files(checkpoint_dir: str, family: str, labels: Sequence[str]) -> tuple[str, str]:
    """Write a randomly-initialised linear classifier matching the real format."""
    import torch

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{family}_v0_ckpt.pt")
    labels_path = os.path.join(checkpoint_dir, f"{family}_v0_labels.json")
    num_classes = len(labels)
    generator = torch.Generator().manual_seed(17)
    state_dict = {
        "model.weight": torch.randn(num_classes, EMBEDDING_DIM, generator=generator) * 0.02,
        "model.bias": torch.zeros(num_classes),
    }
    torch.save(
        {
            "model_type": "pam_linear_multilabel_classifier",
            "n_dim": EMBEDDING_DIM,
            "num_classes": num_classes,
            "state_dict": state_dict,
            "label_order": list(labels),
        },
        ckpt_path,
    )
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump({"species_list": list(labels)}, fh, indent=2)
    return ckpt_path, labels_path


def create_catalog(cur, args, labels: Sequence[str], admin_id: int) -> dict:
    embedding_model_id = ensure_embedding_model(cur)

    cur.execute(
        "INSERT INTO datasets (name, description, source_uri, dataset_type, team_id, is_reference) "
        "VALUES (%s, %s, %s, 'PAM', NULL, false) RETURNING id",
        (
            args.dataset_name,
            f"Synthetic load-test dataset ({args.recordings:,} recordings x "
            f"{args.snippets_per_recording} snippets)",
            args.audio_dir,
        ),
    )
    dataset_id = int(cur.fetchone()[0])

    cur.execute(
        "INSERT INTO snippet_sets (dataset_id, embedding_model_id, window_size, step_size, overlap, status) "
        "VALUES (%s, %s, %s, %s, 0, 'ready') RETURNING id",
        (dataset_id, embedding_model_id, args.window, args.window),
    )
    snippet_set_id = int(cur.fetchone()[0])
    cur.execute(
        "UPDATE datasets SET default_snippet_set_id = %s WHERE id = %s", (snippet_set_id, dataset_id)
    )
    cur.execute(
        "INSERT INTO user_datasets (user_id, dataset_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (admin_id, dataset_id),
    )

    cur.execute(
        "INSERT INTO embedding_jobs (dataset_id, embedding_model_id, snippet_set_id, status, "
        "started_at, completed_at) VALUES (%s, %s, %s, 'completed', now(), now()) RETURNING id",
        (dataset_id, embedding_model_id, snippet_set_id),
    )
    embedding_job_id = int(cur.fetchone()[0])

    family = args.dataset_name.replace(" ", "_")
    ckpt_dir = os.path.join(args.checkpoints_dir, "pam_active_learning", str(dataset_id))
    ckpt_path, labels_path = write_checkpoint_files(ckpt_dir, family, labels)
    cur.execute(
        "INSERT INTO al_model_checkpoints (dataset_id, model_family_name, version, checkpoint_path, "
        "label_config_path, model_type, hyperparameters, is_base, status) "
        "VALUES (%s, %s, 'v0', %s, %s, 'pam_linear_multilabel_classifier', %s, 1, 'AVAILABLE') RETURNING id",
        (
            dataset_id,
            family,
            ckpt_path,
            labels_path,
            json.dumps(
                {
                    "n_dim": EMBEDDING_DIM,
                    "num_classes": len(labels),
                    "label_order": list(labels),
                    "embedding_model_id": embedding_model_id,
                }
            ),
        ),
    )
    checkpoint_id = int(cur.fetchone()[0])
    cur.execute(
        "INSERT INTO al_model_family_state (dataset_id, model_family_name, active_model_checkpoint_id) "
        "VALUES (%s, %s, %s) ON CONFLICT (dataset_id, model_family_name) DO UPDATE "
        "SET active_model_checkpoint_id = EXCLUDED.active_model_checkpoint_id",
        (dataset_id, family, checkpoint_id),
    )

    return {
        "dataset_id": dataset_id,
        "snippet_set_id": snippet_set_id,
        "embedding_model_id": embedding_model_id,
        "embedding_job_id": embedding_job_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": ckpt_path,
        "label_config_path": labels_path,
        "family": family,
    }


# ── Data passes ──────────────────────────────────────────────────────────────


def seed_recordings(cur, args, ids: dict, rec_id0: int) -> None:
    rng = np.random.default_rng(args.seed + 1)
    locations = [f"SITE_{i:02d}" for i in range(args.locations)]
    start_date = dt.date.fromisoformat(args.start_date)
    duration = args.window * args.snippets_per_recording
    progress = Progress("recordings", args.recordings)

    for lo in range(0, args.recordings, args.chunk):
        hi = min(lo + args.chunk, args.recordings)
        m = hi - lo
        loc_idx = rng.integers(0, len(locations), size=m)
        day_offset = rng.integers(0, args.days, size=m)
        # Dawn/dusk-heavy recording times, like a real PAM schedule.
        peak = rng.integers(0, 2, size=m)
        secs = np.clip(
            rng.normal(np.where(peak == 0, 6 * 3600, 18 * 3600), 2 * 3600, size=m), 0, 86399
        )
        # A small slice of recordings has no usable date/time, as in real imports.
        missing = rng.random(m) < args.missing_metadata_fraction
        rows = []
        for i in range(m):
            rid = rec_id0 + lo + i
            pool_ix = (lo + i) % max(args.audio_pool_size, 1)
            file_name = f"mock_{pool_ix:04d}.wav"
            meta = {"location": locations[int(loc_idx[i])]}
            if not missing[i]:
                date = start_date + dt.timedelta(days=int(day_offset[i]))
                meta["recorded_date"] = date.isoformat()
                meta["recorded_time"] = round(float(secs[i]), 1)
            rows.append(
                (
                    rid,
                    ids["dataset_id"],
                    f"{args.audio_rel_dir}/{file_name}",
                    file_name,
                    duration,
                    args.sample_rate,
                    json.dumps(meta, separators=(",", ":")),
                )
            )
        copy_text(
            cur,
            "recordings",
            ("id", "dataset_id", "file_path", "file_name", "duration", "sample_rate", "extra_metadata"),
            rows,
        )
        progress.advance(m)
    progress.finish()


def chunk_rng(args, chunk_index: int) -> np.random.Generator:
    """Per-chunk generator so each pass reproduces the same values independently."""
    return np.random.default_rng(np.random.SeedSequence([args.seed, chunk_index]))


def seed_snippets_and_derived(cur, conn, args, ids: dict, labels: Sequence[str], offsets: dict, steps: set) -> None:
    total = args.total_snippets
    n_labels = len(labels)
    zipf_p = 1.0 / np.arange(1, n_labels + 1) ** args.zipf_exponent
    zipf_p /= zipf_p.sum()

    global_rng = np.random.default_rng(args.seed)
    centers_2d = {
        "pca": global_rng.normal(0, 26, size=(n_labels, 2)),
        "umap": global_rng.normal(0, 7, size=(n_labels, 2)),
        "tsne": global_rng.normal(0, 40, size=(n_labels, 2)),
    }
    spread_2d = {"pca": 4.5, "umap": 1.1, "tsne": 6.0}
    centers_emb = global_rng.normal(0, 1.0, size=(n_labels, EMBEDDING_DIM)).astype(np.float32)

    want = {
        "snippets": "snippets" in steps,
        "predictions": "predictions" in steps,
        "projection": "projection" in steps,
        "embeddings": "embeddings" in steps and not args.no_embeddings,
    }
    progress = {k: Progress(k, total) for k, v in want.items() if v}

    chunk = args.chunk
    for chunk_index, lo in enumerate(range(0, total, chunk)):
        hi = min(lo + chunk, total)
        m = hi - lo
        rng = chunk_rng(args, chunk_index)
        snippet_ids = np.arange(offsets["snippet"] + lo, offsets["snippet"] + hi, dtype=np.int64)
        rec_local = (np.arange(lo, hi, dtype=np.int64) // args.snippets_per_recording)
        recording_ids = offsets["recording"] + rec_local
        slot = (np.arange(lo, hi, dtype=np.int64) % args.snippets_per_recording).astype(np.float64)
        start_time = slot * args.window
        dom = rng.choice(n_labels, size=m, p=zipf_p)

        if want["snippets"]:
            copy_text(
                cur,
                "snippets",
                ("id", "recording_id", "snippet_set_id", "start_time", "end_time", "duration"),
                (
                    (
                        int(snippet_ids[i]),
                        int(recording_ids[i]),
                        ids["snippet_set_id"],
                        float(start_time[i]),
                        float(start_time[i] + args.window),
                        float(args.window),
                    )
                    for i in range(m)
                ),
            )
            progress["snippets"].advance(m)

        if want["predictions"]:
            probs = rng.beta(0.55, 9.0, size=(m, n_labels)).astype(np.float32)
            strong = rng.beta(5.0, 2.0, size=m).astype(np.float32)
            # A fraction of snippets stay below threshold entirely ("no prediction").
            silent = rng.random(m) < args.silent_fraction
            strong = np.where(silent, rng.beta(1.5, 8.0, size=m).astype(np.float32), strong)
            probs[np.arange(m), dom] = strong
            max_prob = probs.max(axis=1)
            uncertainty = 1.0 - np.abs(2.0 * max_prob - 1.0)
            diversity = rng.random(m)
            density = rng.random(m)
            composite = 0.5 * uncertainty + 0.3 * diversity + 0.2 * density

            above = probs >= SCORE_THRESHOLD
            rows_idx, cols_idx = np.nonzero(above)
            counts = np.bincount(rows_idx, minlength=m)
            bounds = np.concatenate([[0], np.cumsum(counts)])
            rounded = np.round(probs, 4).tolist()
            pred_ids = np.arange(offsets["prediction"] + lo, offsets["prediction"] + hi, dtype=np.int64)

            def prediction_rows():
                for i in range(m):
                    cols = cols_idx[bounds[i]:bounds[i + 1]]
                    yield (
                        int(pred_ids[i]),
                        ids["checkpoint_id"],
                        int(snippet_ids[i]),
                        json.dumps([labels[int(c)] for c in cols], separators=(",", ":")),
                        json.dumps(dict(zip(labels, rounded[i])), separators=(",", ":")),
                        float(uncertainty[i]),
                        float(diversity[i]),
                        float(density[i]),
                        float(composite[i]),
                    )

            copy_text(
                cur,
                "al_predictions",
                (
                    "id", "model_checkpoint_id", "snippet_id", "predicted_labels",
                    "predicted_probabilities", "uncertainty", "diversity", "density", "composite_score",
                ),
                prediction_rows(),
            )
            progress["predictions"].advance(m)

        if want["projection"]:
            coords = {
                method: (centers_2d[method][dom] + rng.normal(0, spread_2d[method], size=(m, 2)))
                for method in PROJECTION_METHODS
            }
            fpv_ids = np.arange(offsets["fpv"] + lo, offsets["fpv"] + hi, dtype=np.int64)

            def fpv_rows():
                for i in range(m):
                    yield (
                        int(fpv_ids[i]),
                        ids["dataset_id"],
                        None,
                        ids["embedding_model_id"],
                        int(snippet_ids[i]),
                        float(coords["pca"][i, 0]), float(coords["pca"][i, 1]),
                        float(coords["umap"][i, 0]), float(coords["umap"][i, 1]),
                        float(coords["tsne"][i, 0]), float(coords["tsne"][i, 1]),
                    )

            copy_text(
                cur,
                "fpv_vis",
                (
                    "id", "dataset_id", "model_checkpoint_id", "embedding_model_id", "snippet_id",
                    "pca_2d_x", "pca_2d_y", "umap_2d_x", "umap_2d_y", "tsne_2d_x", "tsne_2d_y",
                ),
                fpv_rows(),
            )
            progress["projection"].advance(m)

        emb_stop = args.embedding_limit or total
        if want["embeddings"] and lo < emb_stop:
            emb_hi = min(hi, emb_stop) - lo
            vecs = centers_emb[dom[:emb_hi]] + rng.normal(
                0, 0.45, size=(emb_hi, EMBEDDING_DIM)
            ).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            emb_ids = np.arange(offsets["embedding"] + lo, offsets["embedding"] + lo + emb_hi, dtype=np.int64)
            # Sub-chunk: one binary payload for 25k x 1024 floats is ~100 MB.
            for sub in range(0, emb_hi, args.embedding_chunk):
                end = min(sub + args.embedding_chunk, emb_hi)
                copy_embeddings(
                    cur,
                    emb_ids[sub:end],
                    snippet_ids[sub:end],
                    ids["embedding_job_id"],
                    ids["embedding_model_id"],
                    vecs[sub:end],
                )
            progress["embeddings"].advance(emb_hi)

        conn.commit()

    for p in progress.values():
        p.finish()


def seed_labels(cur, args, ids: dict, labels: Sequence[str], offsets: dict) -> None:
    """Ground-truth annotations on a random slice, so the hub has a labelled pool."""
    total = args.total_snippets
    n_labeled = int(total * args.labeled_fraction)
    if n_labeled == 0:
        return
    rng = np.random.default_rng(args.seed + 7)
    n_labels = len(labels)
    zipf_p = 1.0 / np.arange(1, n_labels + 1) ** args.zipf_exponent
    zipf_p /= zipf_p.sum()
    rows_sel = rng.choice(total, size=n_labeled, replace=False)
    rows_sel.sort()
    ann_id = next_id(cur, "al_snippet_annotation")
    progress = Progress("labels", n_labeled)

    for lo in range(0, n_labeled, args.chunk):
        hi = min(lo + args.chunk, n_labeled)
        block = rows_sel[lo:hi]
        picks = rng.choice(n_labels, size=len(block), p=zipf_p)
        rows = []
        for i, row_ix in enumerate(block):
            rows.append(
                (
                    ann_id + lo + i,
                    ids["dataset_id"],
                    int(offsets["snippet"] + int(row_ix)),
                    labels[int(picks[i])],
                    "GROUND_TRUTH",
                    None,
                    ids["checkpoint_id"],
                )
            )
        copy_text(
            cur,
            "al_snippet_annotation",
            ("id", "dataset_id", "snippet_id", "label", "source", "user_id", "model_checkpoint_id"),
            rows,
        )
        progress.advance(len(block))
    progress.finish()


def finalize(cur) -> None:
    for table in (
        "recordings", "snippets", "al_predictions", "fpv_vis",
        "embedding_vectors", "al_snippet_annotation",
    ):
        fix_sequence(cur, table)
    print("  sequences reset", flush=True)

    # The base-layer and labels queries filter snippets by snippet_set_id; without
    # these the planner falls back to a sequential scan of the whole table.
    for name, ddl in (
        ("ix_snippets_snippet_set_id", "CREATE INDEX IF NOT EXISTS ix_snippets_snippet_set_id ON snippets (snippet_set_id)"),
        ("ix_snippets_recording_id", "CREATE INDEX IF NOT EXISTS ix_snippets_recording_id ON snippets (recording_id)"),
    ):
        t0 = time.time()
        cur.execute(ddl)
        print(f"  {name} ({time.time() - t0:.0f}s)", flush=True)

    for table in ("recordings", "snippets", "al_predictions", "fpv_vis", "embedding_vectors"):
        t0 = time.time()
        cur.execute(f"ANALYZE {table}")
        print(f"  ANALYZE {table} ({time.time() - t0:.0f}s)", flush=True)

    if not hnsw_index_exists(cur):
        print(
            "\n  NOTE: the pgvector HNSW index is absent. Similarity search still\n"
            "  works (exact scan, slow). To rebuild it -- hours at 3 M rows, so run\n"
            "  it detached and watch pg_stat_progress_create_index:\n"
            f"    psql -c \"SET maintenance_work_mem='8GB'; SET max_parallel_maintenance_workers=8; {HNSW_DDL};\"",
            flush=True,
        )


def invalidate_explore_cache(ids: dict) -> None:
    """Unpublish any explore layer built from a partial view of this seed.

    Opening the dataset in the UI while seeding is still running triggers a
    layer build against whatever rows are committed at that moment, and the
    result is published as a complete version. Nothing later notices: the
    seeder does not go through the ORM hooks that normally invalidate layers,
    so the stale layer is served indefinitely and every count is wrong.

    Deleting the pointer file is all that is needed -- readers re-stat it on
    every access and rebuild when it is missing.
    """
    root = os.environ.get("EXPLORE_CACHE_DIR", "models_AL/explore_cache")
    keys = [
        ("base", f"ss{ids['snippet_set_id']}"),
        ("model", f"ckpt{ids['checkpoint_id']}_ss{ids['snippet_set_id']}"),
        ("projection", f"ds{ids['dataset_id']}_em{ids['embedding_model_id']}"),
    ]
    for kind, key in keys:
        pointer = os.path.join(root, kind, key, "current.json")
        try:
            os.unlink(pointer)
            print(f"  invalidated stale layer {kind}/{key}", flush=True)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  WARNING: could not invalidate {kind}/{key}: {exc}", flush=True)


def write_audio_pool(path: str, count: int, seconds: float, sample_rate: int) -> None:
    """Band-limited chirp/noise files so playback and spectrograms actually work."""
    import soundfile as sf

    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(3)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    for i in range(count):
        base = rng.uniform(800, 5000)
        signal = 0.05 * rng.normal(0, 1, size=n).astype(np.float32)
        for harmonic in range(1, 4):
            freq = base * harmonic
            if freq > sample_rate / 2:
                break
            envelope = (np.sin(2 * np.pi * (0.2 + 0.05 * harmonic) * t) > 0.6).astype(np.float32)
            signal += 0.25 / harmonic * envelope * np.sin(2 * np.pi * freq * t).astype(np.float32)
        signal = np.clip(signal, -1.0, 1.0)
        out = os.path.join(path, f"mock_{i:04d}.wav")
        sf.write(out, signal, sample_rate, subtype="PCM_16")
        print(f"  wrote {out}", flush=True)


# ── Entry point ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--dataset-name", default="Load Test 3M")
    p.add_argument("--recordings", type=int, default=15_000)
    p.add_argument("--snippets-per-recording", type=int, default=200)
    p.add_argument("--labels", type=int, default=24, help="species in the label set")
    p.add_argument("--locations", type=int, default=24)
    p.add_argument("--days", type=int, default=730, help="date span the recordings cover")
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--window", type=float, default=3.0, help="snippet length in seconds")
    p.add_argument("--sample-rate", type=int, default=32_000)
    p.add_argument("--zipf-exponent", type=float, default=1.1)
    p.add_argument("--silent-fraction", type=float, default=0.35,
                   help="share of snippets with no label above threshold")
    p.add_argument("--missing-metadata-fraction", type=float, default=0.02)
    p.add_argument("--labeled-fraction", type=float, default=0.001)
    p.add_argument("--chunk", type=int, default=25_000)
    p.add_argument("--embedding-chunk", type=int, default=5_000)
    p.add_argument("--no-embeddings", action="store_true",
                   help="skip embedding_vectors (browse path only; no retrain)")
    p.add_argument("--keep-hnsw-index", action="store_true",
                   help="do NOT drop the pgvector HNSW index before seeding embeddings "
                        "(leaves inserts at a few hundred rows/s -- days at 3 M)")
    p.add_argument("--embedding-limit", type=int, default=0,
                   help="seed embeddings for only the first N snippets (0 = all)")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--admin-username", default="admin")
    p.add_argument("--checkpoints-dir", default=os.environ.get("PAM_CHECKPOINTS_DIR", "/app/models_AL/pam/checkpoints"))
    p.add_argument("--audio-dir", default="/data/mock_audio", help="dataset source_uri")
    p.add_argument("--audio-rel-dir", default="mock_audio", help="recording.file_path prefix, relative to DATA_ROOT")
    p.add_argument("--audio-pool-size", type=int, default=8)
    p.add_argument("--write-audio-pool", default=None,
                   help="generate the wav pool at this path and exit")
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                   help=f"comma-separated subset of: {', '.join(DEFAULT_STEPS)}")
    args = p.parse_args(argv)
    args.total_snippets = args.recordings * args.snippets_per_recording
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.write_audio_pool:
        write_audio_pool(
            args.write_audio_pool,
            args.audio_pool_size,
            args.window * args.snippets_per_recording,
            args.sample_rate,
        )
        return 0

    if not args.database_url:
        print("DATABASE_URL is not set and --database-url was not given", file=sys.stderr)
        return 2

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - set(DEFAULT_STEPS)
    if unknown:
        print(f"unknown steps: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    labels = species_codes(args.labels)
    print(
        f"target: {args.total_snippets:,} snippets "
        f"({args.recordings:,} recordings x {args.snippets_per_recording}), "
        f"{args.labels} species, embeddings={'no' if args.no_embeddings else 'yes'}",
        flush=True,
    )

    conn = psycopg2.connect(args.database_url)
    conn.autocommit = False
    cur = conn.cursor()

    admin_id = ensure_admin(cur, args.admin_username)

    with step("catalog", "catalog" in steps) as run:
        if run:
            ids = create_catalog(cur, args, labels, admin_id)
            conn.commit()
            print(f"  {json.dumps({k: v for k, v in ids.items() if isinstance(v, int)})}", flush=True)
        else:
            raise SystemExit("the catalog step is required (it allocates the ids)")

    offsets = {
        "recording": next_id(cur, "recordings"),
        "snippet": next_id(cur, "snippets"),
        "prediction": next_id(cur, "al_predictions"),
        "fpv": next_id(cur, "fpv_vis"),
        "embedding": next_id(cur, "embedding_vectors"),
    }
    print(f"  id offsets: {offsets}", flush=True)

    with step("recordings", "recordings" in steps) as run:
        if run:
            seed_recordings(cur, args, ids, offsets["recording"])
            conn.commit()

    if "embeddings" in steps and not args.no_embeddings and not args.keep_hnsw_index:
        if drop_hnsw_index(cur):
            conn.commit()

    data_steps = steps & {"snippets", "predictions", "projection", "embeddings"}
    with step("snippets/predictions/projection/embeddings", bool(data_steps)) as run:
        if run:
            seed_snippets_and_derived(cur, conn, args, ids, labels, offsets, steps)

    with step("labels", "labels" in steps) as run:
        if run:
            seed_labels(cur, args, ids, labels, offsets)
            conn.commit()

    with step("finalize", "finalize" in steps) as run:
        if run:
            finalize(cur)
            conn.commit()
            invalidate_explore_cache(ids)

    print(
        "\nseeded dataset_id=%s snippet_set_id=%s checkpoint_id=%s"
        % (ids["dataset_id"], ids["snippet_set_id"], ids["checkpoint_id"]),
        flush=True,
    )
    # The explore endpoints are POST with a JSON scope body -- not GET with
    # query params. Printing the real call avoids a 405 hunt.
    print(
        "next: warm the explore layers (POST, not GET):\n"
        "  curl -s -X POST localhost:8000/api/explore/summary \\\n"
        "    -H \"Authorization: Bearer $TOKEN\" -H 'Content-Type: application/json' \\\n"
        "    -d '{\"scope\":{\"dataset_id\":%s,\"snippet_set_id\":%s,"
        "\"checkpoint_id\":%s,\"embedding_model_id\":%s}}'\n"
        "  (202 = building; poll until 200)"
        % (
            ids["dataset_id"],
            ids["snippet_set_id"],
            ids["checkpoint_id"],
            ids["embedding_model_id"],
        ),
        flush=True,
    )
    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
