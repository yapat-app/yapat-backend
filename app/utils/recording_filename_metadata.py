"""
Derive recording location / site codes from audio file names.

Supported conventions:
- PAM passive monitoring: ``SITE_YYYYMMDD_HHMMSS`` (e.g. CIPAP02_20240515_080000, INCT17_20200323_043000)
- FNJV focal recordings: ``FNJV_<id>_<Genus>_<species>_<Locality>_<ST>_<Collector>``
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

# SITE_YYYYMMDD_HHMMSS — optional extra suffix segments (e.g. lat/lon fragments)
_PAM_SITE_DATETIME_RE = re.compile(
    r"^([A-Za-z]{2,24}\d*)_(\d{8})_(\d{6})(?:_|$)",
)

_FNJV_PREFIX_RE = re.compile(r"^FNJV_(\d+)_", re.IGNORECASE)


def parse_location_from_filename(file_name: str) -> Optional[str]:
    """
    Extract a location / site identifier from a recording file name.

    Returns None when the name does not match a known convention.
    """
    stem, _ext = os.path.splitext(file_name or "")
    stem = stem.strip()
    if not stem:
        return None

    fnjv = _parse_fnjv_location(stem)
    if fnjv:
        return fnjv

    pam = _parse_pam_site_location(stem)
    if pam:
        return pam

    return None


def parse_datetime_from_filename(file_name: str) -> Optional[tuple[str, int]]:
    """
    Extract (recorded_date, recorded_time_seconds) from a PAM-convention
    filename (``SITE_YYYYMMDD_HHMMSS``, e.g. ``CIPAP02_20240515_080000``).

    recorded_date: "YYYY-MM-DD" (calendar date; no timezone conversion —
    these are field-recorder timestamps local to the recording site, not UTC).
    recorded_time_seconds: seconds since midnight (0-86399).

    Returns None for any filename that doesn't match the PAM convention
    (e.g. FNJV, or anything else) — there is no fallback parser, matching
    parse_location_from_filename's behavior of returning None rather than
    guessing at an unrecognized format.
    """
    stem, _ext = os.path.splitext(file_name or "")
    stem = stem.strip()
    if not stem:
        return None

    m = _PAM_SITE_DATETIME_RE.match(stem)
    if not m:
        return None

    date_part = m.group(2)
    time_part = m.group(3)
    try:
        date_obj = datetime.strptime(date_part, "%Y%m%d")
        time_obj = datetime.strptime(time_part, "%H%M%S")
    except ValueError:
        return None

    recorded_date = date_obj.strftime("%Y-%m-%d")
    recorded_time_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return recorded_date, recorded_time_seconds


def location_source_for_filename(file_name: str) -> Optional[str]:
    """Return a short tag describing which parser matched, or None."""
    stem, _ext = os.path.splitext(file_name or "")
    stem = stem.strip()
    if not stem:
        return None
    if _parse_fnjv_location(stem):
        return "filename_fnjv"
    if _parse_pam_site_location(stem):
        return "filename_pam_site"
    return None


def _parse_pam_site_location(stem: str) -> Optional[str]:
    """First underscore-separated token when followed by YYYYMMDD and HHMMSS."""
    m = _PAM_SITE_DATETIME_RE.match(stem)
    if not m:
        return None
    site = m.group(1).strip()
    return site or None


def _parse_fnjv_location(stem: str) -> Optional[str]:
    """
    FNJV focal format: locality is typically ``<City>_<ST>`` after genus/species.

    Example::
        FNJV_0012917_Boana_raniceps_Alcantara_MA_Luis Felipe Toledo
        → Alcantara_MA
    """
    if not _FNJV_PREFIX_RE.match(stem):
        return None

    parts = stem.split("_")
    if len(parts) < 6:
        return None
    if parts[0].upper() != "FNJV" or not parts[1].isdigit():
        return None

    # parts[2], parts[3] = genus, species; parts[4], parts[5] = locality (often City_ST)
    city = parts[4].strip()
    region = parts[5].strip()
    if not city:
        return None
    if len(region) == 2 and region.isalpha():
        return f"{city}_{region}"
    return city
