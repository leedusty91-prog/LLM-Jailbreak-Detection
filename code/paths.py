"""Repository-relative paths for data and results.

Training scripts used to read ``/content/...`` and ``gs://...`` locations from
Colab and Cloud Storage. Everything in this repository is resolved from the
repo root, with optional environment overrides for a different checkout layout.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the parent of the ``code`` directory)."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    override = os.environ.get("JAILBREAK_DATA_DIR")
    return Path(override).expanduser() if override else repo_root() / "data"


def results_dir() -> Path:
    override = os.environ.get("JAILBREAK_RESULTS_DIR")
    path = Path(override).expanduser() if override else repo_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def results_path(*parts: str) -> Path:
    return results_dir().joinpath(*parts)
