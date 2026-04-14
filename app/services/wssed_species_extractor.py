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

    base = _normpath(ds.source_uri)

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

        # Compute relative path to dataset root.
        rel = fp
        if fp == base:
            rel = ""
        elif fp.startswith(base + os.sep):
            rel = fp[len(base) + 1 :]

        if not rel:
            continue

        # Species is the top-level folder under dataset root.
        first = rel.split(os.sep, 1)[0].strip()
        if first:
            species.add(first)

    return sorted(species)

