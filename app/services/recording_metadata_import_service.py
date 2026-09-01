"""
Enrich a dataset's recordings from an uploaded metadata CSV.

Two operations sharing one parse/match code path:
- ``preview``: parse + match + summarize, NO writes (dry run).
- ``import_metadata``: same, plus merge into ``Recording.extra_metadata`` and
  commit.

Rows are matched to recordings by ``file_name`` (exact match within the dataset,
surrounding whitespace trimmed). See ``app.utils.recording_metadata_csv`` for
the column contract + value normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.models.recording import Recording
from app.utils.recording_metadata_csv import (
    ParsedCsv,
    ParsedRow,
    parse_metadata_csv,
)

logger = logging.getLogger(__name__)

# Cap how much of the (already-small) payload we echo back, per the contract.
_MAX_UNMATCHED_RETURNED = 50
_MAX_ERRORS_RETURNED = 50
_COMMIT_BATCH_SIZE = 500


class RecordingMetadataImportService:
    def __init__(self, db: Session):
        self.db = db

    # -- public API ---------------------------------------------------------

    def preview(self, dataset_id: int, file_bytes: bytes) -> Dict[str, Any]:
        parsed = parse_metadata_csv(
            file_bytes, max_rows=settings.RECORDING_METADATA_MAX_ROWS
        )
        recordings_by_name = self._recordings_by_file_name(dataset_id)

        matched_rows, unmatched_names = self._match(parsed, recordings_by_name)
        affected_ids = {r.id for _row, r in matched_rows}

        errors = list(parsed.errors)
        if parsed.duplicate_file_names:
            errors.insert(
                0,
                "duplicate file_name(s) in CSV (last row wins on import): "
                + ", ".join(parsed.duplicate_file_names[:20]),
            )

        return {
            "total_rows": parsed.total_rows,
            "matched": len(matched_rows),
            "affected_recordings": len(affected_ids),
            "unmatched": len(unmatched_names),
            "unmatched_file_names": unmatched_names[:_MAX_UNMATCHED_RETURNED],
            "columns_present": parsed.columns_present,
            "unique_locations": self._location_counts(matched_rows),
            "errors": errors[:_MAX_ERRORS_RETURNED],
        }

    def import_metadata(
        self,
        dataset_id: int,
        file_bytes: bytes,
        location_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        overrides = location_overrides or {}
        parsed = parse_metadata_csv(
            file_bytes, max_rows=settings.RECORDING_METADATA_MAX_ROWS
        )
        recordings_by_name = self._recordings_by_file_name(dataset_id)

        matched_rows, unmatched_names = self._match(parsed, recordings_by_name)

        errors = list(parsed.errors)
        if parsed.duplicate_file_names:
            errors.insert(
                0,
                "duplicate file_name(s) in CSV (last row applied): "
                + ", ".join(parsed.duplicate_file_names[:20]),
            )

        updated_ids = set()
        pending = 0
        for row, recording in matched_rows:
            if not row.meta:
                continue  # only file_name present -- nothing to merge
            meta = dict(recording.extra_metadata or {})
            new_meta = self._apply_row(row, overrides)
            merged = {**meta, **new_meta}
            if merged == meta:
                continue  # no effective change
            recording.extra_metadata = merged  # reassign so SQLAlchemy marks JSON dirty
            updated_ids.add(recording.id)
            pending += 1
            if pending >= _COMMIT_BATCH_SIZE:
                self.db.commit()
                pending = 0

        if pending:
            self.db.commit()

        return {
            "total_rows": parsed.total_rows,
            "matched": len(updated_ids),
            "unmatched": len(unmatched_names),
            "unmatched_file_names": unmatched_names[:_MAX_UNMATCHED_RETURNED],
            "errors": errors[:_MAX_ERRORS_RETURNED],
        }

    # -- helpers ------------------------------------------------------------

    def _recordings_by_file_name(self, dataset_id: int) -> Dict[str, Recording]:
        """
        All recordings for the dataset keyed by trimmed file_name. On duplicate
        basenames the last one wins -- rare, and the CSV can't disambiguate by
        path anyway.
        """
        rows = (
            self.db.query(Recording)
            .filter(Recording.dataset_id == dataset_id)
            .all()
        )
        return {(r.file_name or "").strip(): r for r in rows}

    @staticmethod
    def _match(parsed: ParsedCsv, recordings_by_name: Dict[str, Recording]):
        matched: List[tuple] = []
        unmatched: List[str] = []
        for row in parsed.rows:
            rec = recordings_by_name.get(row.file_name)
            if rec is None:
                unmatched.append(row.file_name)
            else:
                matched.append((row, rec))
        return matched, unmatched

    @staticmethod
    def _apply_row(row: ParsedRow, overrides: Dict[str, str]) -> Dict[str, Any]:
        """
        The normalized keys to merge for one row, applying the location rename
        and tagging a CSV-supplied location so the filename backfill won't
        clobber it later.
        """
        meta = dict(row.meta)
        if "location" in meta:
            original = meta["location"]
            meta["location"] = overrides.get(original, original)
            meta["location_source"] = "csv_import"
        return meta

    @staticmethod
    def _location_counts(matched_rows) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for row, _rec in matched_rows:
            loc = row.location
            if loc:
                counts[loc] = counts.get(loc, 0) + 1
        return [
            {"name": name, "count": n}
            for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
