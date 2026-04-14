"""
Utilities for extracting species identifiers for WSSED/Active Learning.

For FOCAL_RECORDINGS datasets, the expected on-disk layout is:
  <dataset.source_uri>/<species_name>/<audio files...>

We infer the species list from recording file paths created during dataset scan.
"""

from __future__ import annotations

import os
from typing import List, Set

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.recording import Recording


def _normpath(p: str) -> str:
    # Keep behaviour stable across OSes / trailing slashes.
    return os.path.normpath(p).rstrip(os.sep)

def _candidate_bases(source_uri: str, data_root: str = "/data") -> list[str]:
    """
    Build possible dataset-root prefixes that may appear in Recording.file_path.

    - `DatasetService` stores Recording.file_path relative to DATA_ROOT.
    - Some datasets may store `source_uri` as absolute (e.g. /data/...) or relative.
    """
    bases: list[str] = []
    if not source_uri:
        return bases

    src = _normpath(source_uri)
    data_root_n = _normpath(data_root)

    # If source_uri is absolute under DATA_ROOT, convert to DATA_ROOT-relative.
    if os.path.isabs(src):
        try:
            rel = _normpath(os.path.relpath(src, data_root_n))
            if rel and rel != ".":
                bases.append(rel)
        except Exception:
            pass

    # Plain relative source_uri (as provided)
    bases.append(_normpath(source_uri.lstrip(os.sep)))

    # Also include the absolute form for completeness (in case file_path is absolute).
    bases.append(src)

    # Deduplicate while preserving order
    seen = set()
    out: list[str] = []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


def get_dataset_species_list(dataset_id: int, db: Session) -> List[str]:
    """
    Return a sorted list of unique species names for a dataset.

    Species name is inferred as the first path component under dataset.source_uri
    in each Recording.file_path.
    """
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    if not ds.source_uri:
        return []

    bases = _candidate_bases(ds.source_uri, data_root="/data")

    # Pull only the column we need.
    rec_paths = (
        db.query(Recording.file_path)
        .filter(Recording.dataset_id == dataset_id)
        .all()
    )

    species: Set[str] = set()
    for (file_path,) in rec_paths:
        if not file_path:
            continue
        fp = _normpath(file_path)

        # Compute relative path to dataset root (try all plausible bases).
        rel = fp
        for base in bases:
            if fp == base:
                rel = ""
                break
            if fp.startswith(base + os.sep):
                rel = fp[len(base) + 1 :]
                break

        if not rel:
            continue

        # Species is the top-level folder under dataset root.
        first = rel.split(os.sep, 1)[0].strip()
        if first:
            species.add(first)

    return sorted(species)

