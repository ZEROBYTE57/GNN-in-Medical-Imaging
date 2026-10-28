"""Config loading with a content hash, so results can be traced to settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin wrapper around the parsed YAML.

    Attribute access is deliberately dict-like: the YAML is the source of truth
    and adding a key should not require editing this file.
    """

    raw: dict[str, Any]
    path: Path | None = None
    _hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        self._hash = hashlib.sha256(blob).hexdigest()[:12]

    @property
    def hash(self) -> str:
        """Short digest of the whole config; goes into every output filename."""
        return self._hash

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, key: str) -> dict[str, Any]:
        return self.raw.get(key, {}) or {}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path)


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, truncated. Used to pin the PyRadiomics config."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
