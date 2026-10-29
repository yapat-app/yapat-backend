"""
Versioned, atomically-published numpy layers on the shared filesystem.

Layout::

    <root>/<kind>/<key>/<version>/<array>.npy
    <root>/<kind>/<key>/<version>/meta.json
    <root>/<kind>/<key>/current.json          -> {"version": "<version>"}

Writers build a version in a temp directory and publish it with two
``os.replace`` calls (directory, then pointer), so readers in other processes
either see the previous complete version or the new complete version — never a
partial one. Readers load arrays with ``mmap_mode="r"`` so several API workers
share one copy through the page cache.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

POINTER_FILE = "current.json"
META_FILE = "meta.json"
KEEP_VERSIONS = 2
STALE_TMP_SECONDS = 3600


def cache_root() -> Path:
    return Path(settings.EXPLORE_CACHE_DIR)


def layer_dir(kind: str, key: str) -> Path:
    return cache_root() / kind / key


@dataclass(frozen=True)
class Pointer:
    version: str
    mtime_ns: int


@dataclass
class Layer:
    """A published layer version. Arrays are read-only memory maps."""

    kind: str
    key: str
    version: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def get(self, name: str) -> np.ndarray | None:
        return self.arrays.get(name)


def read_pointer(kind: str, key: str) -> Pointer | None:
    path = layer_dir(kind, key) / POINTER_FILE
    try:
        stat = path.stat()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    version = data.get("version")
    if not isinstance(version, str) or not version:
        return None
    return Pointer(version=version, mtime_ns=stat.st_mtime_ns)


def load_layer(kind: str, key: str, version: str) -> Layer:
    """Open every array of a published version as a read-only memmap."""
    version_dir = layer_dir(kind, key) / version
    with (version_dir / META_FILE).open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays: dict[str, np.ndarray] = {}
    for name in meta.get("arrays", []):
        arr = np.load(version_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        arrays[name] = arr
    return Layer(kind=kind, key=key, version=version, meta=meta, arrays=arrays)


def write_layer(
    kind: str,
    key: str,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Write and atomically publish a new layer version. Returns the version."""
    base = layer_dir(kind, key)
    base.mkdir(parents=True, exist_ok=True)
    version = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    tmp_dir = base / f".tmp-{version}"
    tmp_dir.mkdir()
    try:
        for name, arr in arrays.items():
            if not name.isidentifier():
                raise ValueError(f"Invalid array name {name!r}")
            np.save(tmp_dir / f"{name}.npy", np.ascontiguousarray(arr), allow_pickle=False)
        full_meta = dict(meta or {})
        full_meta.update(
            {
                "kind": kind,
                "key": key,
                "version": version,
                "arrays": sorted(arrays.keys()),
                "created_at": time.time(),
            }
        )
        with (tmp_dir / META_FILE).open("w", encoding="utf-8") as fh:
            json.dump(full_meta, fh)
        os.replace(tmp_dir, base / version)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    pointer_tmp = base / f".{POINTER_FILE}.{uuid.uuid4().hex}"
    with pointer_tmp.open("w", encoding="utf-8") as fh:
        json.dump({"version": version}, fh)
    os.replace(pointer_tmp, base / POINTER_FILE)

    _prune(base, current=version)
    logger.info("explore: published %s/%s version=%s", kind, key, version)
    return version


def invalidate_layer(kind: str, key: str) -> None:
    """Unpublish the current version so the next reader rebuilds it."""
    try:
        (layer_dir(kind, key) / POINTER_FILE).unlink()
        logger.info("explore: invalidated %s/%s", kind, key)
    except FileNotFoundError:
        pass


def invalidate_prefix(kind: str, key_prefix: str) -> int:
    """Invalidate every layer of ``kind`` whose key starts with ``key_prefix``."""
    root = cache_root() / kind
    if not root.is_dir():
        return 0
    count = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(key_prefix):
            invalidate_layer(kind, child.name)
            count += 1
    return count


def _prune(base: Path, current: str) -> None:
    """Keep the newest KEEP_VERSIONS versions; drop abandoned temp dirs.

    Readers that already memory-mapped an older version keep working after its
    files are unlinked (POSIX semantics); a reader that raced the prune retries
    with the fresh pointer (see registry).
    """
    now = time.time()
    versions: list[str] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".tmp-"):
            try:
                if now - child.stat().st_mtime > STALE_TMP_SECONDS:
                    shutil.rmtree(child, ignore_errors=True)
            except FileNotFoundError:
                pass
            continue
        versions.append(child.name)
    versions.sort(reverse=True)
    keep = set(versions[:KEEP_VERSIONS]) | {current}
    for name in versions:
        if name not in keep:
            shutil.rmtree(base / name, ignore_errors=True)
