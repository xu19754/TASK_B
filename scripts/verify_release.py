"""Verify an extracted release using only Python's standard library."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["files_sha256"]
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "RELEASE_MANIFEST.json"
        and not any(part in {"__pycache__", ".pytest_cache"} for part in p.relative_to(root).parts)
        and p.suffix != ".pyc"
    }
    if actual != set(expected):
        raise RuntimeError(f"Release inventory mismatch: missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}")
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Invalid manifest path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Checksum mismatch: {name}")
    print(f"OK release inventory and SHA-256: {len(expected)} files")
    return manifest


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
