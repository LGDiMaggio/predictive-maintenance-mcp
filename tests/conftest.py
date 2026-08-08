"""
Test suite for Predictive Maintenance MCP Server.

Tests cover:
- Signal analysis tools (FFT, Envelope, ISO 20816-3)
- Machine learning tools (Feature extraction, Training, Prediction)
- Visualization tools
- Guided workflows
- Safety features (parameter validation)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import json
import logging
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# --------------------------------------------------------------------------
# Import provenance: test the tree these tests live in, not some other one.
#
# `pip install -e .` writes a MetaPathFinder whose MAPPING hardcodes ONE
# absolute path -- the checkout that ran the install. Every git worktree
# sharing that venv therefore imports `predictive_maintenance_mcp` from the
# primary checkout, and the suite silently exercises source the author never
# edited. This has cost four separate sessions real time; the tell was
# nonsense like a tool count or a __version__ that matched no file on disk.
#
# It cannot be fixed by putting src/ on sys.path. The package is imported by
# its installed name, and there is no directory literally named
# `predictive_maintenance_mcp` in this tree -- pyproject maps the name onto
# src/ via [tool.setuptools.package-dir]. (A `sys.path.insert(0, src)` line
# lived here for exactly that reason and did nothing: nothing bare-imports
# `server`, and every intra-src import is relative.)
#
# So bind the name to THIS tree explicitly. setuptools appends its finder to
# sys.meta_path, i.e. behind PathFinder; inserting at position 0 puts this
# one ahead of both. Sub-packages resolve through the parent's __path__ and
# need no special handling.
#
# In CI and in the primary checkout the editable install already points here,
# so this resolves to the same files and changes nothing. It only diverges
# where today's behaviour is wrong.
# --------------------------------------------------------------------------

_PKG = "predictive_maintenance_mcp"
_ALLOW_INSTALLED = os.environ.get("PMM_TEST_ALLOW_INSTALLED") == "1"


class _LocalTreeFinder:
    """Resolve the package from this checkout, ahead of any editable install."""

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname != _PKG:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            SRC_DIR / "__init__.py",
            submodule_search_locations=[str(SRC_DIR)],
        )


if not _ALLOW_INSTALLED and _LocalTreeFinder not in sys.meta_path:
    sys.meta_path.insert(0, _LocalTreeFinder)


def _assert_import_provenance() -> None:
    """Fail loudly at collection if the package resolves outside this tree.

    The pin above should make this unreachable. It stays because the failure
    it guards is silent: without it, a wrong-tree import produces a green
    suite that proves nothing.
    """
    if _ALLOW_INSTALLED:
        return

    import predictive_maintenance_mcp as pkg

    resolved = Path(pkg.__file__).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return

    raise RuntimeError(
        "predictive_maintenance_mcp resolved OUTSIDE this checkout -- these "
        "tests would exercise source you are not editing.\n"
        f"  this checkout : {REPO_ROOT}\n"
        f"  imported from : {resolved}\n"
        "\n"
        "Usual cause: a shared venv whose `pip install -e .` was run from a "
        "different checkout (its finder hardcodes that absolute path), plus "
        "something importing the package before this conftest loaded.\n"
        "\n"
        "Fix: give this worktree its own environment --\n"
        "  python scripts/setup-worktree.py\n"
        "\n"
        "To test an installed distribution on purpose, set "
        "PMM_TEST_ALLOW_INSTALLED=1."
    )


_assert_import_provenance()

# Test data directory
TEST_DATA_DIR = REPO_ROOT / "data" / "signals" / "real_train"

# Fixtures

@pytest.fixture
def sample_healthy_signal():
    """Load baseline healthy signal."""
    signal_path = TEST_DATA_DIR / "baseline_1.csv"
    if not signal_path.exists():
        pytest.skip(f"Sample data not found: {signal_path}")
    
    df = pd.read_csv(signal_path, header=None)
    return df.iloc[:, 0].values


@pytest.fixture
def sample_faulty_signal():
    """Load outer race fault signal."""
    signal_path = TEST_DATA_DIR / "OuterRaceFault_1.csv"
    if not signal_path.exists():
        pytest.skip(f"Sample data not found: {signal_path}")
    
    df = pd.read_csv(signal_path, header=None)
    return df.iloc[:, 0].values


@pytest.fixture
def sample_metadata():
    """Load metadata for baseline signal."""
    metadata_path = TEST_DATA_DIR / "baseline_1_metadata.json"
    if not metadata_path.exists():
        pytest.skip(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def synthetic_sine_signal():
    """Generate synthetic sine wave for controlled testing."""
    fs = 10000  # 10 kHz
    duration = 2.0  # 2 seconds
    freq = 50.0  # 50 Hz
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    
    return signal, fs, freq


@pytest.fixture
def temp_csv_file(tmp_path, synthetic_sine_signal):
    """Create temporary CSV file with synthetic signal."""
    signal, fs, freq = synthetic_sine_signal
    
    csv_path = tmp_path / "test_signal.csv"
    pd.DataFrame(signal).to_csv(csv_path, index=False, header=False)
    
    return csv_path, fs, freq


@pytest.fixture
def package_caplog(caplog):
    """``caplog``, wired to the package logger instead of root.

    ``server.configure_logging`` sets ``propagate = False`` on the package
    logger, deliberately: root's configuration belongs to whoever reached it
    first, and a host that points root at stdout would otherwise move every
    tool's narration onto the stdio transport's JSON-RPC channel.

    The cost is that caplog's handler — which pytest installs on root — never
    sees these records, so it has to be attached directly. A test that skips
    this fixture and asserts on plain ``caplog.records`` is not lenient, it is
    vacuous: it will see nothing from this package at all.
    """
    from predictive_maintenance_mcp import server

    pkg = logging.getLogger(server.PACKAGE_LOGGER)
    pkg.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=server.PACKAGE_LOGGER)
    try:
        yield caplog
    finally:
        pkg.removeHandler(caplog.handler)
