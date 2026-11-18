"""
Import annotations into a dataset from an uploaded file.

The file is the one the annotation export produces (CSV or JSON). Two
operations share one parse/match code path:
- ``preview``: parse + check every row, return counts and reasons, NO writes.
- ``import_annotations``: same checks, then add the matching rows.

A row is only imported when it matches an existing snippet of this dataset:
the recording is found by ``recording_file_name`` and the snippet by its
``snippet_start_time`` / ``snippet_end_time``. Rows that do not match are
skipped and reported. Imported annotations are added next to whatever a
snippet already has; a snippet that already has the same species is left
alone. Each imported annotation is attributed to the user who ran the import,
is marked ``source="import"`` (database only) and is mirrored into the
active-learning labels exactly like an annotation created in the app.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import taxonomy
from app.models.annotation import Annotation as AnnotationModel
from app.models.dataset import Dataset
from app.models.recording import Recording
from app.models.snippet import Snippet
from app.models.user import User

logger = logging.getLogger(__name__)

IMPORT_SOURCE = "import"

REQUIRED_COLUMNS = (
    "recording_file_name",
    "snippet_start_time",
    "snippet_end_time",
    "taxon_id",
)

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_MAX_ROWS = 200_000
_MAX_ISSUES_RETURNED = 50
# Rows listed per status in a preview; the counts always cover the whole file.
_MAX_ROWS_RETURNED_PER_STATUS = 500
_MAX_TEXT_LENGTH = 255
_MAX_LABELS_RETURNED = 500
_COMMIT_BATCH_SIZE = 500
# Times in the export are the stored floats; allow for rounding in edited files.
_TIME_TOLERANCE_S = 0.001
_MAX_NAME_LENGTH = 255


class AnnotationImportError(ValueError):
    """The file as a whole cannot be used (not a missing row value)."""


@dataclass
class _Row:
    file_name: str
    start: float
    end: float
    taxon_id: str
    name: Optional[str]
    confidence: Optional[float]


@dataclass
class _Context:
    recordings: Dict[str, List[int]]  # trimmed file_name -> recording ids
    snippets: Dict[int, List[Tuple[int, float, float, int]]]  # recording id -> (id, start, end, set id)
    existing: Set[Tuple[int, str]]  # (snippet id, taxon id) already annotated
    default_snippet_set_id: Optional[int]


class AnnotationImportService:
    def __init__(self, db: Session):
        self.db = db

    # -- public API ---------------------------------------------------------

    def preview(
        self, dataset_id: int, user: User, file_bytes: bytes, filename: str
    ) -> Dict[str, Any]:
        outcome = self._check(dataset_id, user, file_bytes, filename)
        return {
            "total_rows": outcome["total_rows"],
            "will_import": len(outcome["to_create"]),
            "already_exist": outcome["already_exist"],
            "no_match": outcome["no_match"],
            "invalid": outcome["invalid"],
            "rows": outcome["rows"],
            "labels": outcome["labels"],
        }

    def import_annotations(
        self, dataset_id: int, user: User, file_bytes: bytes, filename: str
    ) -> Dict[str, Any]:
        outcome = self._check(dataset_id, user, file_bytes, filename)
        to_create = outcome["to_create"]

        if to_create:
            # Imported here: the API module imports services, not the reverse.
            from app.api.annotations import _mirror_to_al

            try:
                pending = 0
                for snippet_id, row, name in to_create:
                    values: Dict[str, Any] = {}
                    if row.confidence is not None:
                        values["confidence"] = row.confidence
                    annotation = AnnotationModel(
                        snippet_id=snippet_id,
                        user_id=user.id,
                        taxon_id=row.taxon_id,
                        resolved_name_snapshot=name,
                        source=IMPORT_SOURCE,
                        **values,
                    )
                    self.db.add(annotation)
                    _mirror_to_al(self.db, annotation)
                    pending += 1
                    if pending >= _COMMIT_BATCH_SIZE:
                        self.db.commit()
                        pending = 0
                if pending:
                    self.db.commit()
            except IntegrityError:
                self.db.rollback()
                raise AnnotationImportError(
                    "The annotations changed while importing. Rows already saved "
                    "are kept; run the import again to add the rest."
                )

        return {
            "total_rows": outcome["total_rows"],
            "imported": len(to_create),
            "already_exist": outcome["already_exist"],
            "no_match": outcome["no_match"],
            "invalid": outcome["invalid"],
            "issues": outcome["issues"],
        }

    # -- checking -----------------------------------------------------------

    def _check(
        self, dataset_id: int, user: User, file_bytes: bytes, filename: str
    ) -> Dict[str, Any]:
        raw_rows, numbers = self._parse(file_bytes, filename)
        ctx = self._load_context(dataset_id)

        to_create: List[Tuple[int, _Row, str]] = []
        seen: Set[Tuple[int, str]] = set()
        rows_out: List[Dict[str, Any]] = []
        listed = {"will_import": 0, "already_exist": 0, "no_match": 0, "invalid": 0}
        counts = {"already_exist": 0, "no_match": 0, "invalid": 0}
        label_stats: Dict[str, Dict[str, Any]] = {}
        taxa: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        def record(
            status: str,
            index: int,
            raw: Dict[str, Any],
            row: Optional[_Row] = None,
            name: Optional[str] = None,
            reason: Optional[str] = None,
        ) -> None:
            if status != "will_import":
                counts[status] += 1
            clip = lambda value: self._text(value)[:_MAX_TEXT_LENGTH]  # noqa: E731

            # Species summary: covers the whole file, not only the listed rows.
            taxon = clip(raw.get("taxon_id"))
            label = label_stats.setdefault(
                taxon,
                {"taxon_id": taxon, "name": None, "will_import": 0,
                 "already_exist": 0, "no_match": 0, "invalid": 0},
            )
            label[status] += 1
            label["name"] = label["name"] or clip(
                name or raw.get("resolved_name_snapshot")
            ) or None

            if listed[status] >= _MAX_ROWS_RETURNED_PER_STATUS:
                return
            listed[status] += 1
            rows_out.append(
                {
                    "row": index,
                    "status": status,
                    "recording": clip(raw.get("recording_file_name")),
                    "start": clip(raw.get("snippet_start_time")),
                    "end": clip(raw.get("snippet_end_time")),
                    "taxon_id": taxon,
                    "name": clip(name or raw.get("resolved_name_snapshot")) or None,
                    "reason": reason,
                }
            )

        for raw, index in zip(raw_rows, numbers):
            row, problem = self._read_row(raw)
            if row is None:
                record("invalid", index, raw, reason=problem or "unusable row")
                continue

            snippet_id, problem = self._find_snippet(ctx, row)
            if snippet_id is None:
                record("no_match", index, raw, row, reason=problem or "no matching snippet")
                continue

            if row.taxon_id not in taxa:
                taxa[row.taxon_id] = self._resolve_taxon(row.taxon_id, user)
            resolved_name, problem = taxa[row.taxon_id]
            if resolved_name is None:
                record("invalid", index, raw, row, reason=problem or "species could not be resolved")
                continue

            name = row.name or resolved_name
            key = (snippet_id, row.taxon_id)
            if key in ctx.existing:
                record(
                    "already_exist", index, raw, row, name,
                    reason="this snippet already has this species",
                )
                continue
            if key in seen:
                record(
                    "already_exist", index, raw, row, name,
                    reason="same snippet and species as an earlier row in this file",
                )
                continue
            seen.add(key)
            to_create.append((snippet_id, row, name))
            record("will_import", index, raw, row, name)

        labels = sorted(
            label_stats.values(),
            key=lambda l: -(l["will_import"] + l["already_exist"] + l["no_match"] + l["invalid"]),
        )[:_MAX_LABELS_RETURNED]

        issues = [
            {"row": r["row"], "reason": r["reason"]}
            for r in rows_out
            if r["status"] in ("no_match", "invalid")
        ][:_MAX_ISSUES_RETURNED]

        return {
            "total_rows": len(raw_rows),
            "to_create": to_create,
            "rows": rows_out,
            "labels": labels,
            "issues": issues,
            **counts,
        }

    # -- parsing ------------------------------------------------------------

    def _parse(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Return the rows and, for each, its number as the user sees it in the file:
        the line in a CSV (the header is line 1), the position from 1 in a JSON list."""
        if not file_bytes:
            raise AnnotationImportError("Uploaded file is empty")
        if len(file_bytes) > _MAX_UPLOAD_BYTES:
            limit_mb = _MAX_UPLOAD_BYTES / (1024 * 1024)
            raise AnnotationImportError(f"File exceeds the {limit_mb:.0f} MB upload limit")
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise AnnotationImportError("File must be UTF-8 text (CSV or JSON)")

        name = (filename or "").lower()
        is_json = name.endswith(".json") or (
            not name.endswith(".csv") and text.lstrip().startswith("[")
        )

        if is_json:
            try:
                data = json.loads(text)
            except ValueError:
                raise AnnotationImportError("File is not valid JSON")
            if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
                raise AnnotationImportError("JSON must be a list of annotation objects")
            rows = data
            numbers = list(range(1, len(rows) + 1))
            present = set(rows[0].keys()) if rows else set()
        else:
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                raise AnnotationImportError("File has no header row")
            reader.fieldnames = [(h or "").strip() for h in reader.fieldnames]
            rows = []
            numbers = []
            for record in reader:
                rows.append(record)
                numbers.append(reader.line_num)
            present = set(reader.fieldnames)

        if not rows:
            raise AnnotationImportError("The file has no rows to import")
        if len(rows) > _MAX_ROWS:
            raise AnnotationImportError(f"File has more than {_MAX_ROWS:,} rows")

        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            raise AnnotationImportError(
                "File is missing required column(s): " + ", ".join(missing)
            )
        return rows, numbers

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @classmethod
    def _number(cls, value: Any) -> Optional[float]:
        try:
            number = float(cls._text(value))
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    def _read_row(self, raw: Dict[str, Any]) -> Tuple[Optional[_Row], Optional[str]]:
        file_name = self._text(raw.get("recording_file_name"))
        if not file_name:
            return None, "recording_file_name is empty"

        start = self._number(raw.get("snippet_start_time"))
        end = self._number(raw.get("snippet_end_time"))
        if start is None or end is None:
            return None, "snippet_start_time and snippet_end_time must be numbers"
        if start < 0 or end <= start:
            return None, "snippet_end_time must be after snippet_start_time"

        taxon_id = self._text(raw.get("taxon_id"))
        if not taxon_id:
            return None, "taxon_id is empty"

        confidence: Optional[float] = None
        if self._text(raw.get("confidence")):
            confidence = self._number(raw.get("confidence"))
            if confidence is None or not 0 <= confidence <= 1:
                return None, "confidence must be a number between 0 and 1"

        name = self._text(raw.get("resolved_name_snapshot"))
        if len(name) > _MAX_NAME_LENGTH:
            name = ""

        return _Row(file_name, start, end, taxon_id, name or None, confidence), None

    # -- matching -----------------------------------------------------------

    def _load_context(self, dataset_id: int) -> _Context:
        recordings: Dict[str, List[int]] = {}
        for rec_id, file_name in (
            self.db.query(Recording.id, Recording.file_name)
            .filter(Recording.dataset_id == dataset_id)
            .all()
        ):
            recordings.setdefault((file_name or "").strip(), []).append(rec_id)

        snippets: Dict[int, List[Tuple[int, float, float, int]]] = {}
        for snip_id, rec_id, start, end, set_id in (
            self.db.query(
                Snippet.id,
                Snippet.recording_id,
                Snippet.start_time,
                Snippet.end_time,
                Snippet.snippet_set_id,
            )
            .join(Recording, Snippet.recording_id == Recording.id)
            .filter(Recording.dataset_id == dataset_id)
            .all()
        ):
            snippets.setdefault(rec_id, []).append((snip_id, start, end, set_id))

        existing = {
            (snippet_id, taxon_id)
            for snippet_id, taxon_id in (
                self.db.query(AnnotationModel.snippet_id, AnnotationModel.taxon_id)
                .join(Snippet, AnnotationModel.snippet_id == Snippet.id)
                .join(Recording, Snippet.recording_id == Recording.id)
                .filter(Recording.dataset_id == dataset_id)
                .all()
            )
        }

        default_set_id = (
            self.db.query(Dataset.default_snippet_set_id)
            .filter(Dataset.id == dataset_id)
            .scalar()
        )
        return _Context(recordings, snippets, existing, default_set_id)

    @staticmethod
    def _find_snippet(ctx: _Context, row: _Row) -> Tuple[Optional[int], Optional[str]]:
        rec_ids = ctx.recordings.get(row.file_name)
        if not rec_ids:
            return None, f"recording '{row.file_name}' is not in this dataset"

        candidates = [
            snippet
            for rec_id in rec_ids
            for snippet in ctx.snippets.get(rec_id, [])
            if abs(snippet[1] - row.start) <= _TIME_TOLERANCE_S
            and abs(snippet[2] - row.end) <= _TIME_TOLERANCE_S
        ]
        if not candidates:
            return None, (
                f"no snippet at {row.start:g}-{row.end:g} s in '{row.file_name}'"
            )

        # The same window can exist in more than one snippet set: prefer the
        # dataset's default set, otherwise the oldest snippet.
        candidates.sort(
            key=lambda s: (0 if s[3] == ctx.default_snippet_set_id else 1, s[0])
        )
        return candidates[0][0], None

    # -- species ------------------------------------------------------------

    def _resolve_taxon(
        self, taxon_id: str, user: User
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (name, None) when usable, else (None, reason). Same checks as creating an annotation."""
        resolved = taxonomy.resolve_taxon_id(taxon_id, db_session=self.db)
        if not resolved:
            return None, f"species '{taxon_id}' could not be resolved"

        if resolved.get("taxonomy_type") == "custom":
            problem = self._custom_taxonomy_problem(taxon_id, user)
            if problem:
                return None, problem

        name = (
            resolved.get("canonical_name")
            or resolved.get("scientific_name")
            or taxon_id
        )
        return name, None

    def _custom_taxonomy_problem(self, taxon_id: str, user: User) -> Optional[str]:
        from app.core.permissions import check_admin, check_team_member
        from app.models.custom_taxonomy import TaxonomyStatus
        from app.services.custom_taxonomy_service import get_taxonomy_by_id

        custom = get_taxonomy_by_id(taxon_id, self.db)
        if not custom:
            return f"custom taxonomy '{taxon_id}' not found"
        if custom.status != TaxonomyStatus.ACTIVE:
            return f"taxonomy '{taxon_id}' is not active"
        if not custom.is_global and not (
            check_team_member(user, custom.team_id, self.db) or check_admin(user)
        ):
            return f"no access to taxonomy '{taxon_id}'"
        return None
