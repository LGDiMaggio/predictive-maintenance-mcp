"""The gate in front of every test that reads the sample recordings.

Each reader calls :func:`require_recordings` before it opens a file. A test
runs only when ``PMM_RESEARCH_DATA=1`` is set and its files are in the
checkout; otherwise it is skipped, with one reason for each case. The tests
carry the ``research_data`` marker, so ``pytest -m research_data`` selects
them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The variable that enables these tests, and the one value that does.
ENABLE_VARIABLE = "PMM_RESEARCH_DATA"
ENABLED_VALUE = "1"

DISABLED_REASON = (
    f"research-data test: set {ENABLE_VARIABLE}={ENABLED_VALUE} to run it "
    "(see data/LICENSE)"
)


def _display(path: Path) -> str:
    """*path* relative to the repository when it lies inside it."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def skip_reason(
    paths: Iterable[Path], environ: Mapping[str, str] = os.environ
) -> Optional[str]:
    """Why a test that reads *paths* cannot run, or None when it can."""
    if environ.get(ENABLE_VARIABLE) != ENABLED_VALUE:
        return DISABLED_REASON
    for path in paths:
        if not path.is_file():
            return f"research-data test: {_display(path)} is not in this checkout"
    return None


def require_recordings(*paths: Path) -> None:
    """Skip the calling test unless every one of *paths* can be read."""
    reason = skip_reason(paths)
    if reason is not None:
        pytest.skip(reason)
