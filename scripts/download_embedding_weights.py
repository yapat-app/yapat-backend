#!/usr/bin/env python3
"""Download the weights of the ONNX embedding models into EMBEDDING_WEIGHTS_DIR.

These files are too large for git (see .gitignore), so each environment fetches
them once. Files that are already present and pass their checksum are skipped.

    python scripts/download_embedding_weights.py            # every model
    python scripts/download_embedding_weights.py perch_v2   # just one
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS_ROOT = Path(
    os.getenv("EMBEDDING_WEIGHTS_DIR", Path(__file__).resolve().parent.parent / "embedding_weights")
)

# {implementation_key: [(file name, url, sha256), ...]}
WEIGHTS = {
    "birdnet_v3": [
        (
            "BirdNET+_V3.0-preview3.1_Global_11K_FP16_pruned.onnx",
            "https://zenodo.org/records/20703646/files/"
            "BirdNET%2B_V3.0-preview3.1_Global_11K_FP16_pruned.onnx?download=1",
            "69cfc8db3ebec163feb6329e546eb56e1aadac2a309f1ee99aecfabd1aa9bd24",
        ),
        (
            "TERMS_OF_USE.txt",
            "https://zenodo.org/records/20703646/files/TERMS_OF_USE.txt?download=1",
            "91b474bddbfdbd7cb2f5d85fd4511eee8af151d899c8450cd0b22000c425bd8f",
        ),
    ],
    "perch_v2": [
        (
            "perch_v2.onnx",
            "https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx",
            "bf0c8467a924cb074663970ca4a0ab1e143602121930209657d0dff5d5cefa1f",
        ),
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(name: str, url: str, sha256: str, target_dir: Path) -> None:
    target = target_dir / name
    if target.exists() and _sha256(target) == sha256:
        print(f"ok        {target}")
        return

    partial = target.with_name(target.name + ".part")
    print(f"download  {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)

    actual = _sha256(partial)
    if actual != sha256:
        partial.unlink()
        raise SystemExit(f"Checksum mismatch for {name}: expected {sha256}, got {actual}")
    partial.replace(target)
    print(f"saved     {target} ({target.stat().st_size} bytes, sha256 {actual})")


def main(keys: list[str]) -> None:
    unknown = set(keys) - set(WEIGHTS)
    if unknown:
        raise SystemExit(f"Unknown model(s): {sorted(unknown)}. Known: {sorted(WEIGHTS)}")
    for key in keys or WEIGHTS:
        target_dir = WEIGHTS_ROOT / key
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, url, sha256 in WEIGHTS[key]:
            _download(name, url, sha256, target_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
