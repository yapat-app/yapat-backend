"""
"Find similar" neighbour search for the explore engine.

Returns the anchor snippet's nearest neighbours (cosine similarity) inside one
snippet set, using the pgvector HNSW index on ``embedding_vectors``. The result
is turned into a dense per-row column so the pure-numpy query engine can filter,
histogram and sort by it like any other score.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.explore.query import BaseData

logger = logging.getLogger(__name__)

# Fixed neighbour count. Large enough for a labelling session and for the
# similarity slider to have a spread of values, small enough to stay fast.
SIMILARITY_K = 500

# HNSW returns at most ef_search candidates per scan; pgvector's default (40)
# would silently cap the result far below SIMILARITY_K. 1000 is pgvector's max.
_EF_SEARCH = 1000
# Upper bound on tuples visited by an iterative scan (pgvector >= 0.8) when the
# snippet-set filter discards candidates from other datasets sharing the model.
_MAX_SCAN_TUPLES = 200_000

_iterative_scan_supported: bool | None = None
_version_lock = threading.Lock()


class AnchorError(ValueError):
    """The anchor cannot be searched from (missing, foreign or not embedded)."""


def _supports_iterative_scan(db: Session) -> bool:
    """pgvector >= 0.8 can keep scanning the index until LIMIT rows pass the filter."""
    global _iterative_scan_supported
    with _version_lock:
        if _iterative_scan_supported is None:
            version = db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            try:
                major, minor = (int(p) for p in str(version).split(".")[:2])
                _iterative_scan_supported = (major, minor) >= (0, 8)
            except (TypeError, ValueError):
                _iterative_scan_supported = False
        return _iterative_scan_supported


def nearest_neighbours(
    db: Session,
    snippet_set_id: int,
    embedding_model_id: int,
    anchor_snippet_id: int,
    k: int = SIMILARITY_K,
) -> tuple[np.ndarray, np.ndarray]:
    """(snippet_ids int64, similarity float32), closest first, anchor included.

    The anchor is kept (similarity 1.0) so the feed can stay on the snippet the
    user searched from; the next row is then the closest match.
    """
    anchor = db.execute(
        text(
            "SELECT ev.vector::text, s.snippet_set_id "
            "FROM embedding_vectors ev JOIN snippets s ON s.id = ev.snippet_id "
            "WHERE ev.snippet_id = :sid AND ev.embedding_model_id = :mid"
        ),
        {"sid": anchor_snippet_id, "mid": embedding_model_id},
    ).first()
    if anchor is None:
        raise AnchorError(f"Snippet {anchor_snippet_id} has no embedding for this snippet set.")
    if int(anchor[1]) != snippet_set_id:
        raise AnchorError(f"Snippet {anchor_snippet_id} is not part of snippet set {snippet_set_id}.")

    # SET LOCAL lasts until the end of this transaction, so the session-wide
    # defaults are restored by the commit below.
    db.execute(text(f"SET LOCAL hnsw.ef_search = {_EF_SEARCH}"))
    if _supports_iterative_scan(db):
        db.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
        db.execute(text(f"SET LOCAL hnsw.max_scan_tuples = {_MAX_SCAN_TUPLES}"))

    # The query vector is bound as a parameter (not joined in) so the planner
    # sees a constant and can use the HNSW index for ORDER BY ... LIMIT.
    rows = db.execute(
        text(
            "SELECT ev.snippet_id, ev.vector <=> CAST(:q AS vector) AS distance "
            "FROM embedding_vectors ev JOIN snippets s ON s.id = ev.snippet_id "
            "WHERE ev.embedding_model_id = :mid AND s.snippet_set_id = :ssid "
            "ORDER BY ev.vector <=> CAST(:q AS vector) LIMIT :k"
        ),
        {"q": anchor[0], "mid": embedding_model_id, "ssid": snippet_set_id, "k": k},
    ).fetchall()
    db.commit()

    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    sims = np.fromiter((1.0 - float(r[1]) for r in rows), dtype=np.float32, count=len(rows))
    if anchor_snippet_id not in set(ids.tolist()):
        # Approximate search can miss the query point itself; it is by
        # definition the best match.
        ids = np.concatenate([[anchor_snippet_id], ids[: k - 1]]).astype(np.int64)
        sims = np.concatenate([[1.0], sims[: k - 1]]).astype(np.float32)
    # relaxed_order iterative scans may return rows slightly out of order.
    order = np.argsort(-sims, kind="stable")
    if len(rows) < k:
        logger.info(
            "Similarity search for snippet %s returned %d of %d neighbours",
            anchor_snippet_id, len(rows), k,
        )
    return ids[order], sims[order]


def similarity_column(base: BaseData, ids: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Dense float32 [n] column: similarity for neighbours, NaN elsewhere."""
    column = np.full(base.n, np.nan, dtype=np.float32)
    rows = base.index_of(ids)
    found = rows >= 0
    column[rows[found]] = sims[found]
    return column
