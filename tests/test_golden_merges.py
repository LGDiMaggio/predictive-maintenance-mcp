"""U9a golden characterization + scenario tests for the four major merges.

Golden contract
---------------
Before the merges, the outputs of the OLD tools were captured on the
deterministic fixtures of ``tests/_golden_signals.py`` (fixed seeds) and
stored in ``tests/fixtures/golden_merges.json``. The capture ran AFTER the
intentional U9 engine fixes (envelope mean-subtraction + Hann window, band
validation, alerts raise-on-invalid), so the snapshot pins exactly the
behavior the merges must PRESERVE:

- ``assess_severity``    <- evaluate_iso_20816 + assess_vibration_severity
                            + check_vibration_alert + check_custom_vibration_alert
- ``analyze_envelope``   <- compute_envelope_spectrum_tool
- ``check_bearing_faults`` <- check_bearing_fault_peak_tool
                            + check_bearing_faults_direct
                            + lookup_bearing_and_compute_tool
- ``analyze_signal_trend`` <- detect_signal_degradation_onset

Intentionally CHANGED paths (envelope detrend+window, band validation,
alerts unknown->raise) are covered by expected-value tests here and in
tests/test_spectral.py / tests/test_alerts.py — NOT by golden snapshots.

Snapshot regeneration: the capture script requires the pre-merge tools and
exists only in the U9a working notes; the snapshot is regenerable because
the fixtures are fully deterministic. Treat golden_merges.json as frozen
regression armor — never edit it by hand.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import (
    assess_severity,
    check_bearing_faults,
)
from predictive_maintenance_mcp.mcp_tools.analysis_tools import analyze_envelope
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
    analyze_signal_trend,
)

from _golden_signals import load_golden_signals

GOLDEN_FILE = Path(__file__).parent / "fixtures" / "golden_merges.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden_repo(tmp_path_factory):
    """Repository loaded with the deterministic golden signals."""
    repo = get_repository()
    repo.clear_all()
    load_golden_signals(repo, tmp_path_factory.mktemp("golden"))
    yield repo
    repo.clear_all()


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Recursive comparison: every key of the OLD snapshot must match the new
# output (new outputs may carry ADDITIONAL fields; missing/changed = drift).
# ---------------------------------------------------------------------------


def assert_matches(new, old, path="", rel=1e-3, abs_tol=1e-6):
    if isinstance(old, dict):
        assert isinstance(new, dict), f"{path}: expected dict, got {type(new)}"
        for key, old_val in old.items():
            assert key in new, f"{path}.{key}: missing in new output"
            assert_matches(new[key], old_val, path=f"{path}.{key}")
    elif isinstance(old, list):
        assert isinstance(new, (list, tuple)), f"{path}: expected list"
        assert len(new) == len(old), (
            f"{path}: length {len(new)} != golden {len(old)}"
        )
        for i, (n, o) in enumerate(zip(new, old)):
            assert_matches(n, o, path=f"{path}[{i}]")
    elif isinstance(old, bool) or old is None or isinstance(old, str):
        assert new == old, f"{path}: {new!r} != golden {old!r}"
    elif isinstance(old, (int, float)):
        assert new == pytest.approx(old, rel=rel, abs=abs_tol), (
            f"{path}: {new} != golden {old}"
        )
    else:  # pragma: no cover - snapshot only holds JSON types
        assert new == old, f"{path}: {new!r} != golden {old!r}"


def subset(d: dict, keys) -> dict:
    return {k: d[k] for k in keys}


# ===========================================================================
# GOLDEN: assess_severity (4-way merge)
# ===========================================================================


class TestGoldenAssessSeverity:
    @pytest.mark.asyncio
    async def test_signal_route_matches_assess_vibration_severity(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["assess_severity"]["signal_route_default"]
        new = await assess_severity(
            ctx=mock_ctx,
            signal_id="golden_iso",
            machine_group=2,
            support_type="rigid",
        )
        assert_matches(new.model_dump(mode="json"), old, path="signal_route")

    @pytest.mark.asyncio
    async def test_signal_route_group1_flexible(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["assess_severity"]["signal_route_group1_flexible"]
        new = await assess_severity(
            ctx=mock_ctx,
            signal_id="golden_iso",
            machine_group=1,
            support_type="flexible",
        )
        assert_matches(new.model_dump(mode="json"), old, path="g1_flex")

    @pytest.mark.asyncio
    async def test_signal_route_velocity_signal(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["assess_severity"]["signal_route_vel3"]
        new = await assess_severity(ctx=mock_ctx, signal_id="golden_vel3")
        assert_matches(new.model_dump(mode="json"), old, path="vel3")
        assert new.unit_conversion_performed is False

    @pytest.mark.asyncio
    async def test_signal_route_matches_evaluate_iso_20816(
        self, golden, golden_repo, mock_ctx
    ):
        """Field-mapped equivalence with the old ISO20816Result shape
        (zone_description excluded: the old tool prefixed a conversion
        notice that now lives in unit_conversion_performed/original_unit)."""
        old = golden["assess_severity"]["evaluate_iso_default"]
        new = (
            await assess_severity(
                ctx=mock_ctx,
                signal_id="golden_iso",
                machine_group=2,
                support_type="rigid",
            )
        ).model_dump(mode="json")

        mapped = {
            "rms_velocity": new["rms_velocity_mm_s"],
            "machine_group": new["machine_group"],
            "support_type": new["support_type"],
            "zone": new["zone"],
            "severity_level": new["severity_level"],
            "color_code": new["color_code"],
            "boundary_ab": new["boundaries"]["AB"],
            "boundary_bc": new["boundaries"]["BC"],
            "boundary_cd": new["boundaries"]["CD"],
            "frequency_range": new["frequency_range"],
            "operating_speed_rpm": new["operating_speed_rpm"],
            "threshold_provenance": new["threshold_provenance"],
        }
        old_cmp = {k: v for k, v in old.items() if k != "zone_description"}
        assert_matches(mapped, old_cmp, path="evaluate_iso")

    @pytest.mark.asyncio
    async def test_low_rpm_band_matches_evaluate_iso(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["assess_severity"]["evaluate_iso_rpm400"]
        new = await assess_severity(
            ctx=mock_ctx,
            signal_id="golden_iso",
            machine_group=2,
            support_type="rigid",
            operating_speed_rpm=400.0,
        )
        assert new.frequency_range == old["frequency_range"]
        assert "2-1000" in new.frequency_range
        assert new.rms_velocity_mm_s == pytest.approx(
            old["rms_velocity"], rel=1e-3
        )

    @pytest.mark.asyncio
    async def test_rms_route_matches_check_vibration_alert(
        self, golden, golden_repo, mock_ctx
    ):
        for case in golden["assess_severity"]["rms_route_iso"]:
            inp, old = case["input"], case["output"]
            new = await assess_severity(
                ctx=mock_ctx,
                rms_velocity_mm_s=inp["rms_velocity_mm_s"],
                machine_group=inp["machine_group"],
                support_type=inp["support_type"],
            )
            assert new.zone == old["zone"], case
            assert new.alert_level == old["alert_level"], case
            assert new.exceeded_threshold == old["exceeded_threshold"], case
            assert new.rms_velocity_mm_s == pytest.approx(old["rms_velocity"])
            assert new.signal_id is None

    @pytest.mark.asyncio
    async def test_custom_thresholds_match_check_custom_vibration_alert(
        self, golden, golden_repo, mock_ctx
    ):
        for case in golden["assess_severity"]["rms_route_custom"]:
            inp, old = case["input"], case["output"]
            new = await assess_severity(
                ctx=mock_ctx,
                rms_velocity_mm_s=inp["rms_velocity_mm_s"],
                thresholds=inp["thresholds"],
            )
            assert new.zone == old["zone"], case
            assert new.alert_level == old["alert_level"], case
            assert new.exceeded_threshold == old["exceeded_threshold"], case
            assert "custom" in new.threshold_provenance.lower()


# ===========================================================================
# GOLDEN: analyze_envelope (spectrum route, band 500-5000)
# ===========================================================================


class TestGoldenAnalyzeEnvelope:
    @pytest.mark.asyncio
    async def test_matches_compute_envelope_spectrum_tool(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["analyze_envelope"]["spectrum_route_bearing"]
        new = await analyze_envelope(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            filter_low=500.0,
            filter_high=5000.0,
            num_peaks=20,
            segment_duration=None,  # old tool analyzed the full signal
        )
        dump = new.model_dump(mode="json")
        assert_matches(dump["top_peaks"], old["top_peaks"], path="top_peaks")
        assert dump["num_samples"] == old["num_samples"]
        assert dump["signal_id"] == old["signal_id"]
        assert tuple(dump["filter_band"]) == tuple(old["frequency_range"])

    @pytest.mark.asyncio
    async def test_matches_old_tool_on_narrow_band(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["analyze_envelope"]["spectrum_route_noise_band"]
        new = await analyze_envelope(
            ctx=mock_ctx,
            signal_id="golden_noise",
            filter_low=1000.0,
            filter_high=4000.0,
            num_peaks=20,
            segment_duration=None,
        )
        dump = new.model_dump(mode="json")
        assert_matches(dump["top_peaks"], old["top_peaks"], path="top_peaks")
        assert tuple(dump["filter_band"]) == tuple(old["frequency_range"])


# ===========================================================================
# GOLDEN: check_bearing_faults (3-way merge)
# ===========================================================================


class TestGoldenCheckBearingFaults:
    @pytest.mark.asyncio
    async def test_bearing_route_matches_check_bearing_faults_direct(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["check_bearing_faults"]["bearing_route_fault_signal"]
        new = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            bearing_id="6205",
            rpm=1800.0,
        )
        assert_matches(new.model_dump(mode="json"), old, path="bearing_route")
        # New traceability fields on top of the preserved behavior:
        assert new.source, "catalog source must be echoed for bearing_id route"
        canon = {c.fault_type: c.fault_type_canonical for c in new.fault_checks}
        assert canon == {
            "BPFO": "outer_race",
            "BPFI": "inner_race",
            "BSF": "ball",
            "FTF": "cage",
        }

    @pytest.mark.asyncio
    async def test_bearing_route_noise_signal(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["check_bearing_faults"]["bearing_route_noise"]
        new = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_noise",
            bearing_id="6205",
            rpm=1800.0,
        )
        assert_matches(new.model_dump(mode="json"), old, path="noise_route")

    @pytest.mark.asyncio
    async def test_bearing_route_matches_lookup_bearing_and_compute(
        self, golden, golden_repo, mock_ctx
    ):
        """The old lookup_bearing_and_compute_tool route (designation with
        manufacturer prefix) is now just the bearing_id route."""
        old = golden["check_bearing_faults"]["lookup_route"]
        new = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            bearing_id="SKF 6205-2RS",
            rpm=1800.0,
        )
        assert_matches(new.model_dump(mode="json"), old, path="lookup_route")

    @pytest.mark.asyncio
    async def test_single_fault_check_preserved_in_summary(
        self, golden, golden_repo, mock_ctx
    ):
        """The old single-fault tool's BPFO result is the BPFO entry of the
        unified summary (same envelope engine, same matching)."""
        old = golden["check_bearing_faults"]["single_peak_bpfo"]
        new = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            bearing_id="6205",
            rpm=1800.0,
        )
        bpfo = next(
            c for c in new.model_dump(mode="json")["fault_checks"]
            if c["fault_type"] == "BPFO"
        )
        assert_matches(bpfo, old, path="single_bpfo")


# ===========================================================================
# GOLDEN: analyze_signal_trend (trend + onset merge)
# ===========================================================================


class TestGoldenAnalyzeSignalTrend:
    @pytest.mark.asyncio
    async def test_trend_fields_preserved(self, golden, golden_repo, mock_ctx):
        old = golden["analyze_signal_trend"]["trend_degrading"]
        new = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_trend",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert_matches(new.model_dump(mode="json"), old, path="trend")

    @pytest.mark.asyncio
    async def test_trend_stationary_preserved(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["analyze_signal_trend"]["trend_stationary"]
        new = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_noise",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
        )
        assert_matches(new.model_dump(mode="json"), old, path="trend_flat")

    @pytest.mark.asyncio
    async def test_onset_fields_match_detect_signal_degradation_onset(
        self, golden, golden_repo, mock_ctx
    ):
        old = golden["analyze_signal_trend"]["onset_degrading"]
        new = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_trend",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
            onset_threshold_sigma=2.0,  # old threshold_sigma
        )
        assert new.onset_detected == old["onset_detected"]
        assert new.onset_segment_index == old["onset_segment_index"]
        assert new.num_segments == old["num_segments"]
        assert new.baseline_segments == old["baseline_segments"]
        assert new.onset_threshold_sigma == old["threshold_sigma"]

    @pytest.mark.asyncio
    async def test_onset_stationary_none(self, golden, golden_repo, mock_ctx):
        old = golden["analyze_signal_trend"]["onset_stationary"]
        new = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_noise",
            feature_name="rms",
            segment_duration=0.1,
            overlap_ratio=0.5,
            onset_threshold_sigma=3.0,
        )
        assert new.onset_detected is old["onset_detected"] is False
        assert new.onset_segment_index is None
        assert new.onset_time_s is None


# ===========================================================================
# SCENARIOS: assess_severity (new unified behavior)
# ===========================================================================


class TestAssessSeverityScenarios:
    @pytest.mark.asyncio
    async def test_rms_and_signal_routes_agree(self, golden_repo, mock_ctx):
        """A direct 3.0 mm/s reading and a stored mm/s signal with 3.0 mm/s
        RMS land in the same zone (group 2, rigid -> C)."""
        direct = await assess_severity(
            ctx=mock_ctx,
            rms_velocity_mm_s=3.0,
            machine_group=2,
            support_type="rigid",
        )
        stored = await assess_severity(
            ctx=mock_ctx,
            signal_id="golden_vel3",
            machine_group=2,
            support_type="rigid",
        )
        assert direct.zone == stored.zone == "C"
        assert stored.rms_velocity_mm_s == pytest.approx(3.0, rel=0.02)

    @pytest.mark.asyncio
    async def test_declared_power_below_scope_refused(
        self, golden_repo, mock_ctx
    ):
        with pytest.raises(ValueError, match="15 kW"):
            await assess_severity(
                ctx=mock_ctx, rms_velocity_mm_s=3.0, machine_power_kw=5.0
            )
        with pytest.raises(ValueError, match="15 kW"):
            await assess_severity(
                ctx=mock_ctx, signal_id="golden_vel3", machine_power_kw=5.0
            )

    @pytest.mark.asyncio
    async def test_unknown_power_not_refused(self, golden_repo, mock_ctx):
        """The same calls WITHOUT machine_power_kw must not refuse for
        power (unknown power is not assumed to be small)."""
        r1 = await assess_severity(ctx=mock_ctx, rms_velocity_mm_s=3.0)
        r2 = await assess_severity(ctx=mock_ctx, signal_id="golden_vel3")
        assert r1.status == r2.status == "assessed"
        assert r1.machine_power_kw is None

    @pytest.mark.asyncio
    async def test_declared_power_in_scope_echoed(self, golden_repo, mock_ctx):
        r = await assess_severity(
            ctx=mock_ctx, rms_velocity_mm_s=3.0, machine_power_kw=90.0
        )
        assert r.machine_power_kw == 90.0

    @pytest.mark.asyncio
    async def test_both_inputs_is_mutual_exclusion_error(
        self, golden_repo, mock_ctx
    ):
        with pytest.raises(ValueError, match="exactly one"):
            await assess_severity(
                ctx=mock_ctx, signal_id="golden_vel3", rms_velocity_mm_s=3.0
            )

    @pytest.mark.asyncio
    async def test_neither_input_is_error(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="exactly one"):
            await assess_severity(ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_negative_rms_rejected(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="non-negative|>= 0"):
            await assess_severity(ctx=mock_ctx, rms_velocity_mm_s=-1.0)

    @pytest.mark.asyncio
    async def test_invalid_custom_thresholds_rejected(
        self, golden_repo, mock_ctx
    ):
        # Not strictly increasing
        with pytest.raises(ValueError, match="warning < alarm < danger"):
            await assess_severity(
                ctx=mock_ctx,
                rms_velocity_mm_s=2.0,
                thresholds={"warning": 5.0, "alarm": 3.0, "danger": 10.0},
            )
        # Wrong keys (typo)
        with pytest.raises(ValueError, match="thresholds keys"):
            await assess_severity(
                ctx=mock_ctx,
                rms_velocity_mm_s=2.0,
                thresholds={"warn": 1.0, "alarm": 3.0, "danger": 5.0},
            )

    @pytest.mark.asyncio
    async def test_custom_thresholds_on_signal_route(
        self, golden_repo, mock_ctx
    ):
        """Custom thresholds also apply to the RMS computed from a stored
        signal (golden_vel3 has ~3.0 mm/s -> zone B for 2/4/6)."""
        r = await assess_severity(
            ctx=mock_ctx,
            signal_id="golden_vel3",
            thresholds={"warning": 2.0, "alarm": 4.0, "danger": 6.0},
        )
        assert r.zone == "B"
        assert r.boundaries == {"AB": 2.0, "BC": 4.0, "CD": 6.0}
        assert "custom" in r.threshold_provenance.lower()
        assert "ISO" not in r.threshold_provenance.split("—")[0]


# ===========================================================================
# SCENARIOS: check_bearing_faults (new unified behavior)
# ===========================================================================


class TestCheckBearingFaultsScenarios:
    @pytest.mark.asyncio
    async def test_arbitrary_frequency_gmf_route(self, golden_repo, mock_ctx):
        """Gearbox / out-of-catalog path: check an arbitrary labeled
        frequency (GMF) without any catalog entry."""
        r = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_noise",
            frequencies={"GMF": 350.0},
            rpm=1480.0,
        )
        assert r.bearing_id is None
        assert len(r.fault_checks) == 1
        check = r.fault_checks[0]
        assert check.fault_type == "GMF"
        assert check.fault_type_canonical is None
        assert check.expected_frequency_hz == pytest.approx(350.0)
        assert r.shaft_frequency_hz == pytest.approx(1480.0 / 60.0)
        assert r.bearing_frequencies["GMF"] == pytest.approx(350.0)
        assert r.source == "user-provided frequencies"

    @pytest.mark.asyncio
    async def test_geometry_route_matches_catalog_route(
        self, golden_repo, mock_ctx
    ):
        """The 6205 catalog entry IS the CWRU geometry: passing that
        geometry explicitly must produce the same expected frequencies as
        the catalog route."""
        by_catalog = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            bearing_id="6205",
            rpm=1800.0,
        )
        by_geometry = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            num_balls=9,
            ball_diameter_mm=7.94,
            pitch_diameter_mm=39.04,
            contact_angle_deg=0.0,
            rpm=1800.0,
        )
        cat = {c.fault_type: c.expected_frequency_hz for c in by_catalog.fault_checks}
        geo = {c.fault_type: c.expected_frequency_hz for c in by_geometry.fault_checks}
        assert geo == pytest.approx(cat)
        # Same detections too (same engine, same expected frequencies)
        assert {c.fault_type: c.detected for c in by_geometry.fault_checks} == {
            c.fault_type: c.detected for c in by_catalog.fault_checks
        }
        assert by_geometry.source == "user-provided geometry"
        assert by_geometry.bearing_id is None

    @pytest.mark.asyncio
    async def test_exactly_one_route_required(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="exactly ONE"):
            await check_bearing_faults(
                ctx=mock_ctx, signal_id="golden_noise", rpm=1800.0
            )
        with pytest.raises(ValueError, match="exactly ONE"):
            await check_bearing_faults(
                ctx=mock_ctx,
                signal_id="golden_noise",
                rpm=1800.0,
                bearing_id="6205",
                frequencies={"GMF": 350.0},
            )

    @pytest.mark.asyncio
    async def test_partial_geometry_rejected(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="Incomplete bearing geometry"):
            await check_bearing_faults(
                ctx=mock_ctx,
                signal_id="golden_noise",
                rpm=1800.0,
                num_balls=9,
                ball_diameter_mm=7.94,
                # pitch_diameter_mm missing
            )

    @pytest.mark.asyncio
    async def test_empty_or_invalid_frequencies_rejected(
        self, golden_repo, mock_ctx
    ):
        with pytest.raises(ValueError, match="empty"):
            await check_bearing_faults(
                ctx=mock_ctx,
                signal_id="golden_noise",
                rpm=1800.0,
                frequencies={},
            )
        with pytest.raises(ValueError, match="non-positive"):
            await check_bearing_faults(
                ctx=mock_ctx,
                signal_id="golden_noise",
                rpm=1800.0,
                frequencies={"GMF": -10.0},
            )

    @pytest.mark.asyncio
    async def test_unknown_bearing_raises(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="not found"):
            await check_bearing_faults(
                ctx=mock_ctx,
                signal_id="golden_noise",
                bearing_id="NONEXISTENT_999",
                rpm=1800.0,
            )


# ===========================================================================
# SCENARIOS: analyze_envelope (band validation + unified default band)
# ===========================================================================


class TestAnalyzeEnvelopeScenarios:
    @pytest.mark.asyncio
    async def test_band_above_nyquist_raises_no_clamp(
        self, golden_repo, mock_ctx
    ):
        """fs = 10 kHz -> Nyquist 5 kHz: filter_high=6000 is an error, the
        band is never silently clamped."""
        with pytest.raises(ValueError, match="Nyquist"):
            await analyze_envelope(
                ctx=mock_ctx, signal_id="golden_bearing", filter_high=6000.0
            )

    @pytest.mark.asyncio
    async def test_inverted_band_raises(self, golden_repo, mock_ctx):
        with pytest.raises(ValueError, match="filter_high"):
            await analyze_envelope(
                ctx=mock_ctx,
                signal_id="golden_bearing",
                filter_low=4000.0,
                filter_high=500.0,
            )

    @pytest.mark.asyncio
    async def test_valid_band_echoed(self, golden_repo, mock_ctx):
        r = await analyze_envelope(
            ctx=mock_ctx,
            signal_id="golden_bearing",
            filter_low=500.0,
            filter_high=4000.0,
        )
        assert tuple(r.filter_band) == (500.0, 4000.0)
        assert "500-4000" in r.diagnosis

    @pytest.mark.asyncio
    async def test_default_band_is_500_5000(self, golden_repo, mock_ctx):
        """The unified default band is 500-5000 Hz (the era-signal_id
        default), replacing the old analyze_envelope 500-2000 Hz."""
        r = await analyze_envelope(ctx=mock_ctx, signal_id="golden_bearing")
        assert tuple(r.filter_band) == (500.0, 5000.0)

    @pytest.mark.asyncio
    async def test_detects_modulation_frequency(self, golden_repo, mock_ctx):
        """Expected-value: the BPFO-rate modulation of the golden bearing
        fixture appears in the top peaks (detrended + windowed FFT)."""
        r = await analyze_envelope(
            ctx=mock_ctx, signal_id="golden_bearing", num_peaks=10
        )
        freqs = [p.frequency_hz for p in r.top_peaks]
        assert any(abs(f - 107.54) < 3.0 for f in freqs), freqs


# ===========================================================================
# SCENARIOS: analyze_signal_trend (onset merged in)
# ===========================================================================


class TestAnalyzeSignalTrendScenarios:
    @pytest.mark.asyncio
    async def test_returns_trend_and_onset_together(
        self, golden_repo, mock_ctx
    ):
        r = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_trend",
            onset_threshold_sigma=2.0,
        )
        # Trend block
        assert r.trend_direction == "increasing"
        assert r.analysis_scope == "within_recording_screening"
        assert len(r.feature_series) > 0
        assert len(r.feature_series) == len(r.segment_times_s)
        # Onset block (merged): never inside the baseline window
        assert r.onset_detected is True
        assert r.onset_segment_index is not None
        assert r.onset_segment_index >= r.baseline_segments
        assert r.baseline_segments == r.num_segments // 2
        assert r.onset_time_s is not None
        assert r.onset_time_s > 0

    @pytest.mark.asyncio
    async def test_series_truncated_to_cap(self, golden_repo, mock_ctx):
        r = await analyze_signal_trend(
            ctx=mock_ctx,
            signal_id="golden_trend",
            segment_duration=0.05,
        )
        assert r.num_segments > 50
        assert len(r.feature_series) <= 50
        assert r.series_truncated is True
