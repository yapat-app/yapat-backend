"""Resolve PAM cold-start metadata and label-config paths relative to DATA_ROOT."""

from __future__ import annotations

import os
from typing import Optional

_DEFAULT_METADATA = "pam_metadata.csv"
_DEFAULT_LABEL_CONFIG = "pam_label_config.json"


def _pam_file_rel_path(
    source_uri: str,
    user_path: Optional[str],
    default_name: str,
) -> str:
    """Where a PAM file is expected, relative to DATA_ROOT (existence not checked)."""
    source_uri = (source_uri or "").strip().strip("/")
    if not source_uri:
        raise ValueError("dataset source_uri is required to resolve training paths")

    if user_path and str(user_path).strip():
        rel = str(user_path).strip().replace("\\", "/").lstrip("/")
    else:
        rel = f"{source_uri}/{default_name}"

    if "/" not in rel:
        rel = f"{source_uri}/{rel}"
    return rel


def _resolve_pam_file(
    data_root: str,
    source_uri: str,
    user_path: Optional[str],
    default_name: str,
) -> str:
    rel = _pam_file_rel_path(source_uri, user_path, default_name)
    abs_path = os.path.join(data_root, rel)
    if not os.path.isfile(abs_path):
        raise ValueError(f"Training file not found: {abs_path}")
    return rel


def resolve_pam_metadata_path(
    data_root: str,
    source_uri: str,
    metadata_path: Optional[str] = None,
) -> str:
    """Resolve only PAM metadata, without requiring a label-config file."""
    return _resolve_pam_file(data_root, source_uri, metadata_path, _DEFAULT_METADATA)


def expected_pam_metadata_path(source_uri: str, metadata_path: Optional[str] = None) -> str:
    """Where the PAM metadata CSV is looked for, relative to DATA_ROOT."""
    return _pam_file_rel_path(source_uri, metadata_path, _DEFAULT_METADATA)


def resolve_pam_training_paths(
    data_root: str,
    source_uri: str,
    metadata_path: Optional[str] = None,
    label_config_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Return relative paths (under data_root) to metadata CSV and label JSON.

    Bare filenames are resolved inside ``source_uri``. When a path is omitted,
    defaults to ``{source_uri}/pam_metadata.csv`` and ``{source_uri}/pam_label_config.json``.
    """
    return (
        resolve_pam_metadata_path(data_root, source_uri, metadata_path),
        _resolve_pam_file(data_root, source_uri, label_config_path, _DEFAULT_LABEL_CONFIG),
    )
