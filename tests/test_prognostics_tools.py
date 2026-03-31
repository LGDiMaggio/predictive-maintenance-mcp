"""Tests for MCP prognostics tools (ISO 13374 Block 5)."""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.prognostics_tools import register
from predictive_maintenance_mcp.models import (
    RULEstimationResult,
    TrendAnalysisResult,
    DegradationOnsetResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = FastMCP("test-prognostics")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Temp directory with synthetic signals for prognostics testing."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    n_samples = 2 * fs  # 2 seconds

    # Stationary signal (constant RMS)
    rng = np.random.default_rng(42)
    stationary = 0.1 * rng.standard_normal(n_samples)
    pd.DataFrame(stationary).to_csv(
        signals_dir / "stationary.csv", index=False, header=False,
    )
    with open(signals_dir / "stationary_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs}, f)

    # Degrading signal — amplitude ramps up linearly across the signal
    t = np.arange(n_samples, dtype=float)
    envelope = 0.05 + 0.5 * (t / n_samples)
    degrading = envelope * rng.standard_normal(n_samples)
    pd.DataFrame(degrading).to_csv(
        signals_dir / "degrading.csv", index=False, header=False,
    )
    with open(signals_dir / "degrading_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs}, f)

    # Patch DATA_DIR
    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.prognostics_tools.DATA_DIR",
        signals_dir,
    )
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", signals_dir)

    return signals_dir


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# estimate_rul
# ---------------------------------------------------------------------------

class TestEstimateRUL:
    """Tests for estimate_rul tool."""

    @pytest.mark.asyncio
    async def test_linear_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            failure_threshold=1.0,
            method="linear",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, RULEstimationResult)
        assert result.method == "linear"
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_exponential_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            failure_threshold=1.0,
            method="exponential",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, RULEstimationResult)
        assert result.method == "exponential"

    @pytest.mark.asyncio
    async def test_invalid_method_raises(self, tools, data_dir, mock_ctx):
        with pytest.raises(ValueError, match="Unknown method"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                signal_file="degrading.csv",
                failure_threshold=1.0,
                method="invalid_method",
            )

    @pytest.mark.asyncio
    async def test_weibull_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            failure_threshold=1.0,
            method="weibull",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, RULEstimationResult)
        assert result.method == "weibull"
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_kalman_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            failure_threshold=1.0,
            method="kalman",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, RULEstimationResult)
        assert result.method == "kalman"
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_file_not_found(self, tools, data_dir, mock_ctx):
        with pytest.raises(FileNotFoundError):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                signal_file="nonexistent.csv",
                failure_threshold=1.0,
            )

    @pytest.mark.asyncio
    async def test_invalid_feature_raises(self, tools, data_dir, mock_ctx):
        with pytest.raises(ValueError, match="Unknown feature"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                signal_file="degrading.csv",
                failure_threshold=1.0,
                feature_name="nonexistent_feature",
            )

    @pytest.mark.asyncio
    async def test_stationary_signal_returns_inf(self, tools, data_dir, mock_ctx):
        """A stationary signal should not reach a high failure threshold."""
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            signal_file="stationary.csv",
            failure_threshold=100.0,
            method="linear",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, RULEstimationResult)
        # Either inf (no crossing) or very large value
        assert result.rul == float("inf") or result.rul > 100


# ---------------------------------------------------------------------------
# analyze_signal_trend
# ---------------------------------------------------------------------------

class TestAnalyzeSignalTrend:
    """Tests for analyze_signal_trend tool."""

    @pytest.mark.asyncio
    async def test_degrading_shows_increasing(self, tools, data_dir, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, TrendAnalysisResult)
        assert result.feature_name == "rms"
        assert result.trend_direction == "increasing"
        assert result.slope > 0
        assert result.num_segments > 0

    @pytest.mark.asyncio
    async def test_stationary_shows_stable(self, tools, data_dir, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_file="stationary.csv",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, TrendAnalysisResult)
        assert result.trend_direction == "stable"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tools, data_dir, mock_ctx):
        with pytest.raises(FileNotFoundError):
            await tools["analyze_signal_trend"](
                ctx=mock_ctx,
                signal_file="nonexistent.csv",
            )


# ---------------------------------------------------------------------------
# detect_signal_degradation_onset
# ---------------------------------------------------------------------------

class TestDetectDegradationOnset:
    """Tests for detect_signal_degradation_onset tool."""

    @pytest.mark.asyncio
    async def test_degrading_detects_onset(self, tools, data_dir, mock_ctx):
        result = await tools["detect_signal_degradation_onset"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            feature_name="rms",
            threshold_sigma=2.0,
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, DegradationOnsetResult)
        assert result.feature_name == "rms"
        assert result.onset_detected is True
        assert result.onset_segment_index is not None
        assert result.num_segments > 0

    @pytest.mark.asyncio
    async def test_stationary_no_onset(self, tools, data_dir, mock_ctx):
        result = await tools["detect_signal_degradation_onset"](
            ctx=mock_ctx,
            signal_file="stationary.csv",
            feature_name="rms",
            threshold_sigma=3.0,
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, DegradationOnsetResult)
        assert result.onset_detected is False
        assert result.onset_segment_index is None

    @pytest.mark.asyncio
    async def test_custom_threshold_sigma(self, tools, data_dir, mock_ctx):
        result = await tools["detect_signal_degradation_onset"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            feature_name="rms",
            threshold_sigma=10.0,
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, DegradationOnsetResult)
        assert result.threshold_sigma == 10.0
