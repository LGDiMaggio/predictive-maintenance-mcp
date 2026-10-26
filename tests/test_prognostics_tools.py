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

from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.prognostics_tools import register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.models import (
    RULEstimationResult,
    TrendAnalysisResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp():
    server = MCPServer("test-prognostics")
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
        signals_dir / "stationary.csv",
        index=False,
        header=False,
    )
    with open(signals_dir / "stationary_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs}, f)

    # Degrading signal — amplitude ramps up linearly across the signal
    t = np.arange(n_samples, dtype=float)
    envelope = 0.05 + 0.5 * (t / n_samples)
    degrading = envelope * rng.standard_normal(n_samples)
    pd.DataFrame(degrading).to_csv(
        signals_dir / "degrading.csv",
        index=False,
        header=False,
    )
    with open(signals_dir / "degrading_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs}, f)

    # Patch DATA_DIR
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir
    )

    return signals_dir


@pytest.fixture
def repo(data_dir):
    """Repository with the screening signals loaded; cleaned afterwards."""
    repo = get_repository()
    repo.clear_all()
    repo.load_signal("stationary.csv")  # metadata: fs=10000
    repo.load_signal("degrading.csv")
    yield repo
    repo.clear_all()


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
    """Tests for the analyze_signal_trend screening tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_degrading_shows_increasing(self, tools, repo, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="degrading",
            feature_name="rms",
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
    async def test_stationary_shows_stable(self, tools, repo, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="stationary",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, TrendAnalysisResult)
        assert result.trend_direction == "stable"

    @pytest.mark.asyncio
    async def test_returns_feature_series(self, tools, repo, mock_ctx):
        """The screening tool echoes the per-segment series for follow-up."""
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="degrading",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert len(result.feature_series) > 0
        assert len(result.feature_series) == len(result.segment_times_s)
        assert result.series_truncated is False
        # Times are within the 2-second recording.
        assert all(0.0 <= t <= 2.0 for t in result.segment_times_s)

    @pytest.mark.asyncio
    async def test_series_truncated_to_cap(self, tools, repo, mock_ctx):
        """Long recordings are subsampled to at most 50 echoed points."""
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="degrading",
            feature_name="rms",
            segment_duration=0.05,
            overlap_ratio=0.5,
        )
        assert result.num_segments > 50
        assert len(result.feature_series) <= 50
        assert result.series_truncated is True

    @pytest.mark.asyncio
    async def test_signal_not_loaded_names_remedy(self, tools, repo, mock_ctx):
        """Unknown signal_id → standard error naming load_signal/list_signals."""
        with pytest.raises(ValueError) as exc:
            await tools["analyze_signal_trend"](
                ctx=mock_ctx,
                signal_id="nonexistent",
            )
        msg = str(exc.value)
        assert "load_signal" in msg
        assert "list_signals" in msg


# ---------------------------------------------------------------------------
# Degradation onset (merged into analyze_signal_trend in U9)
# ---------------------------------------------------------------------------


class TestDegradationOnsetMerged:
    """Onset detection now lives inside analyze_signal_trend (U9 merge)."""

    def test_old_onset_tool_gone(self, tools):
        assert "detect_signal_degradation_onset" not in tools

    @pytest.mark.asyncio
    async def test_degrading_detects_onset(self, tools, repo, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="degrading",
            feature_name="rms",
            onset_threshold_sigma=2.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert isinstance(result, TrendAnalysisResult)
        assert result.feature_name == "rms"
        assert result.onset_detected is True
        assert result.onset_segment_index is not None
        assert result.num_segments > 0
        # Onset can never be flagged inside the baseline window.
        assert result.onset_segment_index >= result.baseline_segments
        assert result.baseline_segments == result.num_segments // 2
        # Onset time maps to the segment center within the recording.
        assert result.onset_time_s is not None
        assert 0.0 <= result.onset_time_s <= 2.0

    @pytest.mark.asyncio
    async def test_stationary_no_onset(self, tools, repo, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="stationary",
            feature_name="rms",
            onset_threshold_sigma=3.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert result.onset_detected is False
        assert result.onset_segment_index is None
        assert result.onset_time_s is None

    @pytest.mark.asyncio
    async def test_custom_threshold_sigma(self, tools, repo, mock_ctx):
        result = await tools["analyze_signal_trend"](
            ctx=mock_ctx,
            signal_id="degrading",
            feature_name="rms",
            onset_threshold_sigma=10.0,
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert result.onset_threshold_sigma == 10.0


# ---------------------------------------------------------------------------
# forecast_asset_indicator (forecast PR 1, U4)
# ---------------------------------------------------------------------------


def _hourly_series(n=200, seed=5, hours_ago=1):
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(hours=hours_ago)
    rng = np.random.default_rng(seed)
    instants = [
        (end - timedelta(hours=n - 1 - i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]
    values = [float(v) for v in np.round(np.cumsum(rng.normal(0, 0.2, n)) + 4.0, 6)]
    return instants, values


class TestForecastAssetIndicator:
    BASE = dict(indicator="velocity RMS", unit="mm/s", resolution="1h")

    @pytest.mark.asyncio
    async def test_forecast_result(self, tools, mock_ctx):
        """Covers AE1."""
        from predictive_maintenance_mcp.models import IndicatorForecastResult

        obs, vals = _hourly_series(300)
        result = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=24, **self.BASE
        )
        assert isinstance(result, IndicatorForecastResult)
        assert result.status == "forecast"
        assert len(result.p10) == len(result.p50) == len(result.p90) == 24
        assert all(p == float(f"{vals[-1]:.6g}") for p in result.p50)
        assert result.backend == "naive"
        assert result.provenance.effective_declaration.aggregation == "mean"
        assert result.interpretation.startswith("A projection of the indicator")

    @pytest.mark.asyncio
    async def test_insufficient_history_is_typed(self, tools, mock_ctx):
        """Covers AE3."""
        obs, vals = _hourly_series(8)
        result = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=6, **self.BASE
        )
        assert result.status == "insufficient_history"
        assert result.reason_code == "too_few_difference_pairs"
        assert (result.available, result.required) == (7, 10)
        assert result.p50 is None and result.method_note is None

    @pytest.mark.asyncio
    async def test_horizon_remedy_can_be_run_as_given(self, tools, mock_ctx):
        obs, vals = _hourly_series(30)
        first = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=40, **self.BASE
        )
        assert first.reason_code == "horizon_exceeds_history"
        second = await tools["forecast_asset_indicator"](
            ctx=mock_ctx,
            observed_at=obs,
            values=vals,
            horizon_steps=first.max_supported_horizon_steps,
            **self.BASE,
        )
        assert second.status == "forecast"

    @pytest.mark.asyncio
    async def test_resolution_remedy_example_works(self, tools, mock_ctx):
        obs, vals = _hourly_series(48)
        with pytest.raises(ValueError, match="'10min' or '1h'"):
            await tools["forecast_asset_indicator"](
                ctx=mock_ctx,
                observed_at=obs,
                values=vals,
                horizon_steps=6,
                indicator="velocity RMS",
                unit="mm/s",
                resolution="1hr",
            )
        result = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=6, **self.BASE
        )
        assert result.status == "forecast"

    @pytest.mark.asyncio
    async def test_seasonal_naive(self, tools, mock_ctx):
        obs, _ = _hourly_series(96)
        rng = np.random.default_rng(1)
        base = np.sin(2 * np.pi * np.arange(24) / 24) * 3 + 20
        vals = [float(v) for v in np.tile(base, 4) + rng.normal(0, 0.2, 96)]
        result = await tools["forecast_asset_indicator"](
            ctx=mock_ctx,
            observed_at=obs,
            values=vals,
            horizon_steps=24,
            backend="seasonal_naive",
            season_length_steps=24,
            **self.BASE,
        )
        assert result.status == "forecast"
        assert result.history.difference_lag == 24
        assert result.method_note.startswith("Seasonal naive reference")

    @pytest.mark.asyncio
    async def test_same_call_twice_is_deterministic(self, tools, mock_ctx):
        obs, vals = _hourly_series(120)
        a = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=12, **self.BASE
        )
        b = await tools["forecast_asset_indicator"](
            ctx=mock_ctx, observed_at=obs, values=vals, horizon_steps=12, **self.BASE
        )
        assert a.p10 == b.p10 and a.p90 == b.p90
        assert a.provenance.input_digest == b.provenance.input_digest


def test_tool_description_states_the_limits_the_code_enforces():
    """The description hard-codes the caps; this keeps it in step with them."""
    from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
        forecast_asset_indicator,
    )
    from predictive_maintenance_mcp.prognostics import indicator_series as s
    from predictive_maintenance_mcp.prognostics import reference_forecast as rf

    doc = " ".join(forecast_asset_indicator.__doc__.split())
    assert f"strictly increasing; at most {s.MAX_INPUT_POINTS}." in doc
    assert f"may span at most {s.MAX_GRID_BINS} bins." in doc
    assert f"Steps to project, 1 to {s.MAX_HORIZON_STEPS}." in doc
    assert f"longer than {s.MAX_HORIZON_DURATION.days} days is refused" in doc
    assert f"Grid step of at most {s.MAX_HORIZON_DURATION.days} days" in doc
    assert f"at least {rf.MIN_DIFFERENCE_PAIRS} difference pairs" in doc
    assert rf.MIN_SEASONS_COVERED == 2 and "also needs two full seasons" in doc
    from predictive_maintenance_mcp.prognostics.indicator_forecast import (
        MAX_RESULT_BYTES,
    )

    assert (
        f"server's conservative {MAX_RESULT_BYTES:,}-byte serialized-output "
        f"limit is refused. When possible, the refusal suggests" in doc
    )


def test_server_instructions_carry_the_projection_section():
    """The model-facing guard against reading a projection as a prognosis."""
    from predictive_maintenance_mcp.prognostics.indicator_forecast import INTERPRETATION
    from predictive_maintenance_mcp.server import mcp as server

    text = " ".join(server.instructions.split())
    assert "Indicator projection (not a prognosis):" in text
    disclaimer = "not a fault detection, a failure probability or a remaining-useful-life estimate"
    assert disclaimer in text and disclaimer in INTERPRETATION
    assert "never pass them to estimate_rul or assess_severity" in text
