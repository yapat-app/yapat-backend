"""
Parse + normalize a user-uploaded recording-metadata CSV.

The CSV enriches ``Recording.extra_metadata`` for an existing dataset: rows are
matched to recordings by ``file_name`` and the remaining columns are merged into
the JSON blob. Header names are the contract (English template); values are
normalized here so they line up with what the filename parser already produces
(see ``app.utils.recording_filename_metadata``) and what the frontend filters
expect.

Everything in this module is pure (no DB, no I/O beyond decoding the uploaded
bytes) so preview and import share one code path and it is easy to unit test.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

JOIN_KEY = "file_name"

# Every column the importer understands. `file_name` is the join key; the rest
# are optional and merged into extra_metadata. The template ships a subset --
# any subset is accepted; unknown headers are ignored.
RECOGNIZED_COLUMNS: Tuple[str, ...] = (
    "file_name",
    "catalog_number",
    "taxon_phylum",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "taxon_subspecies",
    "species_as_identified",
    "original_filename",
    "cuts",
    "collector",
    "autonomous",
    "recorded_date",
    "recorded_time",
    "datetime_notes",
    "country",
    "state",
    "city",
    "location",
    "duration",
    "audio_format",
    "sample_rate",
    "bit_depth",
)


class MetadataCsvError(ValueError):
    """Raised for whole-file problems (bad encoding, missing join-key header)."""


# ---------------------------------------------------------------------------
# Value normalizers -- each raises ValueError with a human message on bad input
# ---------------------------------------------------------------------------

def _normalize_recorded_date(raw: str) -> str:
    """`DD/MM/YYYY` -> `YYYY-MM-DD` (matches the filename parser's format)."""
    try:
        return datetime.strptime(raw, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"recorded_date '{raw}' is not a valid DD/MM/YYYY date")


def _normalize_recorded_time(raw: str) -> int:
    """`HH:MM` or `HH:MM:SS` (24h) -> seconds since midnight (0-86399)."""
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return t.hour * 3600 + t.minute * 60 + t.second
    raise ValueError(f"recorded_time '{raw}' is not a valid HH:MM[:SS] time")


def _normalize_int(raw: str, field: str) -> int:
    try:
        return int(float(raw))  # tolerate "32000.0"
    except ValueError:
        raise ValueError(f"{field} '{raw}' is not a valid integer")


def _normalize_duration_seconds(raw: str) -> int:
    """`MM:SS` (or `HH:MM:SS`) -> seconds. A bare number is taken as seconds."""
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            return int(float(parts[0]))
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"duration '{raw}' is not valid MM:SS")
    if len(parts) == 2:
        mm, ss = nums
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = nums
        return hh * 3600 + mm * 60 + ss
    raise ValueError(f"duration '{raw}' is not valid MM:SS")


def _normalize_autonomous(raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("yes", "true", "1", "y"):
        return True
    if v in ("no", "false", "0", "n"):
        return False
    raise ValueError(f"autonomous '{raw}' is not Yes/No")


# Columns that need conversion. Everything else recognised is stored as a
# trimmed string. `duration`, `sample_rate`, `bit_depth` are CSV-reported audio
# facts -- stored under reported_* so they never shadow the measured
# recordings.duration / recordings.sample_rate columns.
_NORMALIZERS = {
    "recorded_date": lambda v: ("recorded_date", _normalize_recorded_date(v)),
    "recorded_time": lambda v: ("recorded_time", _normalize_recorded_time(v)),
    "autonomous": lambda v: ("autonomous", _normalize_autonomous(v)),
    "sample_rate": lambda v: ("reported_sample_rate", _normalize_int(v, "sample_rate")),
    "bit_depth": lambda v: ("bit_depth", _normalize_int(v, "bit_depth")),
    "duration": lambda v: ("reported_duration", _normalize_duration_seconds(v)),
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class ParsedRow:
    """One CSV data row after normalization."""

    __slots__ = ("row_number", "file_name", "meta", "location")

    def __init__(self, row_number: int, file_name: str, meta: Dict[str, Any]):
        self.row_number = row_number
        self.file_name = file_name
        self.meta = meta  # normalized, non-blank keys destined for extra_metadata
        # original (un-renamed) location string, for preview counts + overrides
        self.location: Optional[str] = meta.get("location")


class ParsedCsv:
    def __init__(
        self,
        rows: List[ParsedRow],
        columns_present: List[str],
        errors: List[str],
        total_rows: int,
        duplicate_file_names: List[str],
    ):
        self.rows = rows
        self.columns_present = columns_present
        self.errors = errors
        self.total_rows = total_rows
        self.duplicate_file_names = duplicate_file_names


def _sniff_delimiter(text: str) -> str:
    """
    Detect the column delimiter from the header line. Excel exports in
    non-English locales (e.g. Brazilian/European) use ';' instead of ','; some
    use tabs. Pick whichever candidate appears most in the header row, falling
    back to comma.
    """
    header_line = text.splitlines()[0] if text else ""
    candidates = [",", ";", "\t", "|"]
    best = max(candidates, key=lambda d: header_line.count(d))
    return best if header_line.count(best) > 0 else ","


def _decode(file_bytes: bytes) -> str:
    """Decode UTF-8, tolerating a leading BOM (the template ships one)."""
    try:
        return file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MetadataCsvError("File is not valid UTF-8 text")


def parse_metadata_csv(
    file_bytes: bytes, max_rows: Optional[int] = None
) -> ParsedCsv:
    """
    Parse + normalize the CSV bytes into ParsedRows.

    Raises MetadataCsvError for whole-file problems (bad encoding, empty file,
    missing `file_name` header, or more than ``max_rows`` data rows). Per-cell
    problems are collected into ``errors`` (1-based over data rows) and that
    field is skipped, never failing the row.
    """
    if not file_bytes or not file_bytes.strip():
        raise MetadataCsvError("Uploaded file is empty")

    text = _decode(file_bytes)
    delimiter = _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    if reader.fieldnames is None:
        raise MetadataCsvError("Uploaded file has no header row")

    headers = [(h or "").strip() for h in reader.fieldnames]
    if JOIN_KEY not in headers:
        raise MetadataCsvError(
            f"Missing required '{JOIN_KEY}' column. Found: {', '.join(headers) or '(none)'}"
        )

    recognized = set(RECOGNIZED_COLUMNS)
    columns_present = [h for h in headers if h in recognized]

    rows: List[ParsedRow] = []
    errors: List[str] = []
    seen_file_names: Dict[str, int] = {}
    duplicate_file_names: List[str] = []
    total_rows = 0

    for raw_row in reader:
        total_rows += 1
        row_number = total_rows  # 1-based over data rows

        if max_rows is not None and total_rows > max_rows:
            raise MetadataCsvError(
                f"CSV has more than the allowed {max_rows} data rows"
            )

        file_name = (raw_row.get(JOIN_KEY) or "").strip()
        if not file_name:
            errors.append(f"row {row_number}: missing {JOIN_KEY}")
            continue

        if file_name in seen_file_names:
            if file_name not in duplicate_file_names:
                duplicate_file_names.append(file_name)
        else:
            seen_file_names[file_name] = row_number

        meta: Dict[str, Any] = {}
        for col in columns_present:
            if col == JOIN_KEY:
                continue
            raw_val = raw_row.get(col)
            if raw_val is None:
                continue
            value = raw_val.strip()
            if value == "":
                continue  # blank cell: skip, never overwrite with empty

            normalizer = _NORMALIZERS.get(col)
            if normalizer is None:
                meta[col] = value
                continue
            try:
                key, normalized = normalizer(value)
            except ValueError as e:
                errors.append(f"row {row_number}: {e}")
                continue
            meta[key] = normalized

        rows.append(ParsedRow(row_number, file_name, meta))

    return ParsedCsv(
        rows=rows,
        columns_present=columns_present,
        errors=errors,
        total_rows=total_rows,
        duplicate_file_names=duplicate_file_names,
    )
