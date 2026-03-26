"""Tests for MCP diagnostics tools (ISO 13374 Blocks 3-4)."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = FastMCP("test-diagnostics")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Temp directory with synthetic signals for diagnostics testing."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 2.0, 2 * fs, endpoint=False)

    # Normal signal: low-level broadband noise
    normal = 0.05 * np.random.randn(len(t))
    pd.DataFrame(normal).to_csv(signals_dir / "normal.csv", index=False, header=False)
    with open(signals_dir / "normal_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Signal with known velocity RMS (for ISO 10816 testing)
    # Generate a signal that, when integrated, produces ~5 mm/s RMS velocity
    freq = 50.0  # Hz
    amplitude = 0.5  # g
    accel = amplitude * np.sin(2 * np.pi * freq * t)
    pd.DataFrame(accel).to_csv(signals_dir / "iso_test.csv", index=False, header=False)
    with open(signals_dir / "iso_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Patch directories
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_repository.DATA_DIR", signals_dir)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR", models_dir)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.REPORTS_DIR", reports_dir)

    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()
    (resources_dir / "machine_manuals").mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.RESOURCES_DIR", resources_dir)

    cache_dir = resources_dir / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.CACHE_DIR", cache_dir)

    return signals_dir


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# ISO 20816 evaluation
# ---------------------------------------------------------------------------

class TestEvaluateISO20816:
    """Tests for evaluate_iso_20816 tool."""

    @pytest.mark.asyncio
    async def test_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
        )
        assert result is not None
        # Should have a zone classification
        assert hasattr(result, "zone") or hasattr(result, "severity_zone")


class TestAssessVibrationSeverity:
    """Tests for assess_vibration_severity tool."""

    @pytest.mark.asyncio
    async def test_returns_severity(self, tools, data_dir, mock_ctx):
        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="sev_test", sampling_rate=10000)
        try:
            result = await tools["assess_vibration_severity"](
                ctx=mock_ctx, signal_id="sev_test"
            )
            assert result is not None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Bearing diagnostics
# ---------------------------------------------------------------------------

class TestBearingTools:
    """Tests for bearing fault detection tools."""

    @pytest.mark.asyncio
    async def test_get_bearing_catalog(self, tools, data_dir, mock_ctx):
        if "get_bearing_catalog" not in tools:
            pytest.skip("get_bearing_catalog not registered")
        result = await tools["get_bearing_catalog"](ctx=mock_ctx)
        assert result is not None

    @pytest.mark.asyncio
    async def test_calculate_bearing_frequencies(self, tools, data_dir, mock_ctx):
        if "calculate_bearing_frequencies" not in tools:
            pytest.skip("calculate_bearing_frequencies not registered")
        result = await tools["calculate_bearing_frequencies"](
            ctx=mock_ctx, bearing_id="6205", rpm=1800.0
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_check_bearing_faults_direct(self, tools, data_dir, mock_ctx):
        if "check_bearing_faults_direct" not in tools:
            pytest.skip("check_bearing_faults_direct not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="bearing_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults_direct"](
                ctx=mock_ctx,
                signal_id="bearing_test",
                bearing_id="6205",
                rpm=1800.0,
            )
            assert result is not None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Anomaly detection (train + predict)
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Tests for train_anomaly_model and predict_anomalies tools."""

    @pytest.fixture
    def training_signals(self, data_dir):
        """Create multiple training signal files."""
        fs = 10000
        for i in range(5):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            sig = 0.05 * np.random.randn(len(t))
            pd.DataFrame(sig).to_csv(data_dir / f"train_{i}.csv", index=False, header=False)
            with open(data_dir / f"train_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
        return [f"train_{i}.csv" for i in range(5)]

    @pytest.mark.asyncio
    async def test_train_and_predict(self, tools, data_dir, mock_ctx, training_signals):
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        # Train
        train_result = await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            model_name="test_model",
            sampling_rate=10000.0,
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        assert train_result is not None

        # Predict
        if "predict_anomalies" not in tools:
            pytest.skip("predict_anomalies not registered")

        predict_result = await tools["predict_anomalies"](
            signal_file="train_0.csv",
            model_name="test_model",
            ctx=mock_ctx,
        )
        assert predict_result is not None


# ---------------------------------------------------------------------------
# Integrated diagnosis
# ---------------------------------------------------------------------------

class TestDiagnoseVibration:
    """Tests for diagnose_vibration tool."""

    @pytest.mark.asyncio
    async def test_full_diagnosis(self, tools, data_dir, mock_ctx):
        if "diagnose_vibration" not in tools:
            pytest.skip("diagnose_vibration not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="diag_test", sampling_rate=10000)
        try:
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_test",
                rpm=1800.0,
            )
            assert result is not None
        finally:
            repo.clear_all()
