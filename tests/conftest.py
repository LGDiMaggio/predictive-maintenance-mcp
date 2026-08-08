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
import json
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "signals" / "real_train"

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
