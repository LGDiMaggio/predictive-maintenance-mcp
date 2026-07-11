"""Tests for MCP prognostics tools (ISO 13374 Block 5).

Covers the honest-prognosis contract:
- estimate_rul works on multi-measurement series only and refuses a
  single recording/point;
- flat series produce a typed "no degradation trend" outcome, never a
  RUL number;
- the trend/onset tools are within-recording screening and echo the
  (truncated) feature series.
"""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.prognostics_tools import register
from predictive_maintenance_mcp.signal_repository import get_repository
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
def measurement_signals(tmp_path):
    """Load 4 recordings with increasing RMS into the signal repository.

    Simulates one recording per measurement session over time.
    Yields the list of signal_ids; cleans the repository afterwards.
    """
    repo = get_repository()
    rng = np.random.default_rng(7)
    fs = 10000
    signal_ids = []
    for i, scale in enumerate([0.1, 0.35, 0.6, 0.85]):
        data = scale * rng.standard_normal(fs)
        path = tmp_path / f"u4_rul_measurement_{i}.csv"
        pd.DataFrame(data).to_csv(path, index=False, header=False)
        sid = f"u4_measurement_{i}"
        repo.load_signal(str(path), signal_id=sid, sampling_rate=fs)
        signal_ids.append(sid)
    yield signal_ids
    for sid in signal_ids:
        repo.clear_signal(sid)


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# estimate_rul — refusal paths
# ---------------------------------------------------------------------------

class TestEstimateRULRefusals:
    """estimate_rul must refuse anything that is not a multi-measure series."""

    @pytest.mark.asyncio
    async def test_single_point_rejected(self, tools, mock_ctx):
        """A single measurement cannot show a degradation trend."""
        with pytest.raises(ValueError) as exc:
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[3.2],
                timestamps=[0.0],
            )
        msg = str(exc.value)
        assert "analyze_signal_trend" in msg
        assert "measurement" in msg.lower()

    @pytest.mark.asyncio
    async def test_single_signal_id_rejected(self, tools, mock_ctx):
        """One recording is one measurement — refused before loading it."""
        with pytest.raises(ValueError) as exc:
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                signal_ids=["one_recording"],
                timestamps=[0.0],
            )
        msg = str(exc.value)
        assert "analyze_signal_trend" in msg
        assert "single recording" in msg.lower() or "measurement" in msg.lower()

    @pytest.mark.asyncio
    async def test_two_points_rejected(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="at least 3"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0],
                timestamps=[0.0, 1.0],
            )

    @pytest.mark.asyncio
    async def test_both_input_routes_rejected(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="exactly one"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0, 3.0],
                signal_ids=["a", "b", "c"],
                timestamps=[0.0, 1.0, 2.0],
            )

    @pytest.mark.asyncio
    async def test_neither_input_route_rejected(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="exactly one"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                timestamps=[0.0, 1.0, 2.0],
            )

    @pytest.mark.asyncio
    async def test_timestamps_length_mismatch(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="timestamp"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0, 3.0],
                timestamps=[0.0, 1.0],
            )

    @pytest.mark.asyncio
    async def test_non_increasing_timestamps(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="strictly increasing"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0, 3.0],
                timestamps=[0.0, 2.0, 2.0],
            )

    @pytest.mark.asyncio
    async def test_unknown_method_rejected(self, tools, mock_ctx):
        """Weibull was removed — it is not a silent alias for anything."""
        with pytest.raises(ValueError, match="Unknown method"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0, 3.0],
                timestamps=[0.0, 1.0, 2.0],
                method="weibull",
            )

    @pytest.mark.asyncio
    async def test_unknown_signal_id_actionable_error(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                signal_ids=["ghost_1", "ghost_2", "ghost_3"],
                timestamps=[0.0, 1.0, 2.0],
            )

    @pytest.mark.asyncio
    async def test_confidence_input_rejected(self, tools, mock_ctx):
        """No tool accepts a caller-dictated confidence."""
        with pytest.raises(TypeError):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=4.5,
                feature_values=[1.0, 2.0, 3.0],
                timestamps=[0.0, 1.0, 2.0],
                confidence=0.9,
            )


# ---------------------------------------------------------------------------
# estimate_rul — outcomes
# ---------------------------------------------------------------------------

class TestEstimateRULOutcomes:
    """estimate_rul outcomes on valid multi-measure series."""

    @pytest.mark.asyncio
    async def test_increasing_series_estimates_rul(self, tools, mock_ctx):
        """10 rising RMS measurements over 30 days → RUL with horizon."""
        timestamps = [float(d) for d in range(0, 30, 3)]  # 10 points, 0..27 days
        values = [1.0 + 0.1 * d for d in timestamps]  # reaches 4.5 at d=35
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=4.5,
            feature_values=values,
            timestamps=timestamps,
            time_unit="days",
        )
        assert isinstance(result, RULEstimationResult)
        assert result.status == "estimated"
        assert result.method == "linear"
        assert result.num_measurements == 10
        assert result.time_unit == "days"
        assert result.observation_horizon == pytest.approx(27.0)
        # Crossing at day 35, last measurement day 27 → RUL 8 days.
        assert result.rul == pytest.approx(8.0, abs=0.5)
        assert result.fit_r_squared is not None and result.fit_r_squared > 0.99
        assert result.trend_p_value is not None and result.trend_p_value < 0.01
        assert result.estimated_rate == pytest.approx(0.1, abs=0.01)
        # Renamed field: no invented confidence anywhere in the output.
        assert not hasattr(result, "confidence")
        assert "confidence" not in result.model_dump_json().lower()

    @pytest.mark.asyncio
    async def test_flat_series_no_degradation_trend(self, tools, mock_ctx):
        """A healthy (flat) machine gets a typed outcome, not a RUL number."""
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=4.5,
            feature_values=[2.0] * 8,
            timestamps=[float(i) for i in range(8)],
        )
        assert isinstance(result, RULEstimationResult)
        assert result.status == "no_degradation_trend"
        assert result.rul is None
        assert "No degradation trend" in result.message

    @pytest.mark.asyncio
    async def test_noisy_stable_series_no_trend(self, tools, mock_ctx):
        rng = np.random.default_rng(11)
        values = (2.0 + rng.normal(0, 0.05, 12)).tolist()
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=4.5,
            feature_values=values,
            timestamps=[float(i) for i in range(12)],
        )
        assert result.status == "no_degradation_trend"
        assert result.rul is None

    @pytest.mark.asyncio
    async def test_threshold_already_exceeded(self, tools, mock_ctx):
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=4.5,
            feature_values=[3.0, 4.0, 5.0],
            timestamps=[0.0, 1.0, 2.0],
        )
        assert result.status == "threshold_already_exceeded"
        assert result.rul is None
        assert "Inspect" in result.message

    @pytest.mark.asyncio
    async def test_decreasing_series_no_trend(self, tools, mock_ctx):
        """Indicator moving away from the threshold is not degradation."""
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=10.0,
            feature_values=[5.0 - 0.3 * i for i in range(10)],
            timestamps=[float(i) for i in range(10)],
        )
        assert result.status == "no_degradation_trend"
        assert result.rul is None

    @pytest.mark.asyncio
    async def test_extrapolation_caution_in_message(self, tools, mock_ctx):
        """RUL far beyond the observed horizon must carry a caution."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        values = [1.0 + 0.01 * t for t in timestamps]
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=2.0,
            feature_values=values,
            timestamps=timestamps,
        )
        # slope significant (perfect line), crossing at t=100 → RUL 96 ≫ horizon 4
        assert result.status == "estimated"
        assert "Caution" in result.message

    @pytest.mark.asyncio
    async def test_kalman_method(self, tools, mock_ctx):
        """Kalman on uniform series exposes interval + heuristic, honestly named."""
        timestamps = [float(i) for i in range(12)]
        values = [1.0 + 0.2 * t for t in timestamps]
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=6.0,
            feature_values=values,
            timestamps=timestamps,
            method="kalman",
        )
        assert result.status == "estimated"
        assert result.method == "kalman"
        assert result.rul is not None and result.rul > 0
        assert result.rul_interval_95 is not None
        assert result.rul_interval_95[0] <= result.rul <= result.rul_interval_95[1]
        assert result.precision_heuristic is not None
        assert 0.0 <= result.precision_heuristic <= 1.0
        assert result.fit_r_squared is None  # no OLS fit in the Kalman path

    @pytest.mark.asyncio
    async def test_kalman_requires_uniform_spacing(self, tools, mock_ctx):
        with pytest.raises(ValueError, match="uniform"):
            await tools["estimate_rul"](
                ctx=mock_ctx,
                failure_threshold=6.0,
                feature_values=[1.0, 1.5, 2.5, 4.0],
                timestamps=[0.0, 1.0, 5.0, 20.0],
                method="kalman",
            )

    @pytest.mark.asyncio
    async def test_exponential_method(self, tools, mock_ctx):
        import math
        timestamps = [float(t) for t in range(10)]
        values = [0.5 * math.exp(0.2 * t) for t in timestamps]
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=0.5 * math.exp(0.2 * 15),
            feature_values=values,
            timestamps=timestamps,
            method="exponential",
        )
        assert result.status == "estimated"
        assert result.method == "exponential"
        # Crossing at t=15, last at t=9 → RUL ≈ 6.
        assert result.rul == pytest.approx(6.0, abs=0.5)
        assert result.fit_r_squared is not None and result.fit_r_squared > 0.99

    @pytest.mark.asyncio
    async def test_signal_ids_route(self, tools, mock_ctx, measurement_signals):
        """Multiple stored recordings + timestamps form a valid series."""
        result = await tools["estimate_rul"](
            ctx=mock_ctx,
            failure_threshold=2.0,
            signal_ids=measurement_signals,
            timestamps=[0.0, 10.0, 20.0, 30.0],
            feature_name="rms",
            time_unit="days",
        )
        assert isinstance(result, RULEstimationResult)
        assert result.status == "estimated"
        assert result.num_measurements == 4
        assert result.observation_horizon == pytest.approx(30.0)
        assert result.rul is not None and result.rul > 0
        assert result.fit_r_squared is not None


# ---------------------------------------------------------------------------
# analyze_signal_trend (within-recording screening)
# ---------------------------------------------------------------------------

class TestAnalyzeSignalTrend:
    """Tests for the analyze_signal_trend screening tool."""

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
        assert result.p_value is not None and result.p_value < 0.05
        assert result.num_segments > 0
        assert result.analysis_scope == "within_recording_screening"

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
    async def test_returns_feature_series(self, tools, data_dir, mock_ctx):
        """The screening tool echoes the per-segment series for follow-up."""
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert len(result.feature_series) > 0
        assert len(result.feature_series) == len(result.segment_times_s)
        assert result.series_truncated is False
        # Times are within the 2-second recording.
        assert all(0.0 <= t <= 2.0 for t in result.segment_times_s)

    @pytest.mark.asyncio
    async def test_series_truncated_to_cap(self, tools, data_dir, mock_ctx):
        """Long recordings are subsampled to at most 50 echoed points."""
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_file="degrading.csv",
            feature_name="rms",
            sampling_rate=10000.0,
            segment_duration=0.05,
            overlap_ratio=0.5,
        )
        assert result.num_segments > 50
        assert len(result.feature_series) <= 50
        assert result.series_truncated is True

    @pytest.mark.asyncio
    async def test_file_not_found(self, tools, data_dir, mock_ctx):
        """Missing file (no metadata, no explicit rate) raises the strict
        sampling-rate error first — never a silent 10 kHz fallback (U6)."""
        with pytest.raises(ValueError, match="sampling_rate"):
            await tools["analyze_signal_trend"](
                ctx=mock_ctx,
                signal_file="nonexistent.csv",
            )

    @pytest.mark.asyncio
    async def test_file_not_found_with_explicit_rate(
        self, tools, data_dir, mock_ctx
    ):
        """With the rate given explicitly, the missing file itself raises."""
        with pytest.raises(FileNotFoundError):
            await tools["analyze_signal_trend"](
                ctx=mock_ctx,
                signal_file="nonexistent.csv",
                sampling_rate=10000.0,
            )


# ---------------------------------------------------------------------------
# detect_signal_degradation_onset (within-recording screening)
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
        # Onset can never be flagged inside the baseline window.
        assert result.onset_segment_index >= result.baseline_segments
        assert result.baseline_segments == result.num_segments // 2

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
