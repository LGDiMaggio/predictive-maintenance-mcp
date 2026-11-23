"""forecast_asset_indicator with backend 'chronos2', on the fake model libraries.

- The envelope: at least 64 observed bins in the model's window, a horizon of
  at most 144 steps, the observed-span rule capped at 144 steps, the
  truncation warning past 2,048 bins.
- The reference anchors: the reference backend's own numbers, as a direct
  call computes them on the whole grid, at the forecast's anchor steps; the
  call is refused, wrapping the reference's own text, when they cannot be
  computed or reported.
- Warnings, notes, provenance and the result contract, with exact texts.
- The reference backends: unchanged and never touched by the chronos2 checks.
- The tool over MCP: refusals, the output budget, a free event loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

import _chronos2_texts as T
from _chronos2_stub import IN_RANGE, drain_worker, install
from predictive_maintenance_mcp import models as m
from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
    client_result_bytes,
    forecast_asset_indicator,
)
from predictive_maintenance_mcp.models import IndicatorForecastResult
from predictive_maintenance_mcp.prognostics import chronos2_backend as cb
from predictive_maintenance_mcp.prognostics import indicator_forecast as f
from predictive_maintenance_mcp.prognostics.indicator_series import format_duration
from predictive_maintenance_mcp.signal_validation import InvalidInputError

NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
STEP = timedelta(hours=1)


def _series(n=200, level=10.0, seed=1, values=None, drop=(), step=STEP, end=None):
    """n hourly bins ending one step before NOW; ``drop`` omits indices."""
    rng = np.random.default_rng(seed)
    end = end or NOW - step
    if values is None:
        values = level + np.cumsum(rng.normal(0, 0.1, n))
    keep = [i for i in range(n) if i not in set(drop)]
    instants = [(end - (n - 1 - i) * step).strftime("%Y-%m-%dT%H:%M:%SZ") for i in keep]
    return instants, [float(values[i]) for i in keep]


def _kwargs(horizon_steps=24, resolution="1h", **kw):
    return dict(
        indicator="velocity RMS",
        unit="mm/s",
        resolution=resolution,
        horizon_steps=horizon_steps,
        now=NOW,
        server_version="test",
        **kw,
    )


def _chronos2(obs, vals, **kw):
    return asyncio.run(
        f.forecast_indicator_chronos2(observed_at=obs, values=vals, **_kwargs(**kw))
    )


def _reference(obs, vals, backend="naive", **kw):
    return f.forecast_indicator(
        observed_at=obs, values=vals, backend=backend, **_kwargs(**kw)
    )


def _refusal(fn, *args, **kwargs) -> str:
    with pytest.raises(InvalidInputError) as exc:
        fn(*args, **kwargs)
    return str(exc.value)


@pytest.fixture(autouse=True)
def _drain_the_worker(monkeypatch):
    # Set up after monkeypatch, so it runs before the fakes are restored.
    yield
    drain_worker()


@pytest.fixture
def stub(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


class TestResult:
    def test_a_chronos2_forecast_and_its_contract(self, stub):
        obs, vals = _series()
        r = _chronos2(obs, vals)
        assert r["status"] == "forecast" and r["backend"] == "chronos2"
        assert set(r) == set(m.IndicatorForecastResult.model_fields)
        assert set(r["provenance"]) == set(m.IndicatorForecastProvenance.model_fields)
        assert set(r["provenance"]["model"]) == set(
            m.IndicatorForecastModel.model_fields
        )
        assert set(r["reference"]) == set(m.IndicatorForecastReference.model_fields)
        model = IndicatorForecastResult.model_validate(r)
        dumped = model.model_dump(mode="json")
        assert dumped["reference"]["backend"] == "naive"
        assert dumped["provenance"]["model"]["repo_id"] == "amazon/chronos-2"

    def test_the_median_is_the_model_quantile_plus_the_last_value(self, stub):
        obs, vals = _series()
        r = _chronos2(obs, vals, horizon_steps=6)
        assert r["p50"] == [float(f"{vals[-1]:.6g}")] * 6
        assert r["p90"][0] - r["p50"][0] == pytest.approx(0.5, rel=1e-5)

    def test_notes_are_the_approved_texts(self, stub):
        obs, vals = _series()
        r = _chronos2(obs, vals)
        assert r["method_note"] == T.METHOD_NOTE
        assert r["interval_note"] == T.INTERVAL_NOTE
        assert r["interpretation"] == f.INTERPRETATION

    def test_provenance(self, stub, monkeypatch):
        monkeypatch.setenv(cb.ENV_THREADS, "3")
        obs, vals = _series()
        r = _chronos2(obs, vals)
        prov = r["provenance"]
        assert (prov["family"], prov["algorithm_version"]) == (
            "indicator_forecast_chronos2",
            1,
        )
        assert prov["effective_declaration"]["backend"] == "chronos2"
        assert prov["model"] == {
            "repo_id": "amazon/chronos-2",
            "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
            "weights_sha256": cb.MODEL_FILES[1][2],
            "config_sha256": cb.MODEL_FILES[0][2],
            "weights_licence": "Apache-2.0",
            "chronos_forecasting_version": IN_RANGE["chronos-forecasting"],
            "torch_version": IN_RANGE["torch"],
            "transformers_version": IN_RANGE["transformers"],
            "accelerate_version": IN_RANGE["accelerate"],
            "huggingface_hub_version": IN_RANGE["huggingface-hub"],
            "torch_threads": 3,
            "context_bins": 200,
        }
        naive = _reference(obs, vals)
        assert prov["input_digest"] != naive["provenance"]["input_digest"]

    def test_the_digest_follows_the_chronos2_version(self, stub, monkeypatch):
        obs, vals = _series()
        base = _chronos2(obs, vals)["provenance"]["input_digest"]
        monkeypatch.setattr(f, "CHRONOS2_ALGORITHM_VERSION", 2)
        assert _chronos2(obs, vals)["provenance"]["input_digest"] != base

    def test_the_contract_descriptions_are_the_approved_texts(self):
        fields = m.IndicatorForecastResult.model_fields
        assert fields["reference"].description == T.REFERENCE_DESCRIPTION
        assert fields["backend"].description == "Backend that ran"
        assert (
            m.IndicatorForecastDeclaration.model_fields["backend"].description
            == "Backend requested"
        )
        ref = m.IndicatorForecastReference.model_fields
        assert ref["backend"].description == T.REFERENCE_BACKEND_DESCRIPTION
        assert ref["anchors"].description == T.REFERENCE_ANCHORS_DESCRIPTION
        prov = m.IndicatorForecastProvenance.model_fields
        assert prov["family"].description == T.FAMILY_DESCRIPTION
        assert prov["algorithm_version"].description == (
            "Version of the numbers; the same inputs give the same numbers within "
            "one version. " + T.ALGORITHM_VERSION_APPENDED
        )
        assert prov["model"].description == T.MODEL_DESCRIPTION
        assert {
            name: field.description
            for name, field in m.IndicatorForecastModel.model_fields.items()
        } == T.MODEL_FIELDS
        history = m.IndicatorForecastHistory.model_fields
        assert history["difference_lag"].description == T.DIFFERENCE_LAG_DESCRIPTION

    def test_the_difference_pairs_are_those_of_the_reference(self, stub):
        obs, vals = _series(drop=range(50, 60))
        pairs = ("difference_lag", "difference_pairs", "zero_difference_share")
        r = _chronos2(obs, vals)["history"]
        naive = _reference(obs, vals)["history"]
        assert r["difference_lag"] == 1
        assert {k: r[k] for k in pairs} == {k: naive[k] for k in pairs}

    def test_reference_results_carry_neither_new_block(self):
        obs, vals = _series()
        dumped = IndicatorForecastResult.model_validate(_reference(obs, vals))
        wire = dumped.model_dump(mode="json")
        assert "reference" not in wire and "model" not in wire["provenance"]
        text = dumped.model_dump_json()
        assert '"reference"' not in text and '"model"' not in text


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_fewer_than_64_observed_bins_in_the_window_are_refused(self, stub):
        obs, vals = _series(n=63)
        assert _refusal(_chronos2, obs, vals, horizon_steps=12) == (
            T.TOO_FEW_OBSERVED.format(observed_in_window=63)
        )

    def test_the_observed_bins_are_counted_inside_the_window(self, stub):
        # 2,100 bins, the window is the last 2,048: the 40 observed values in
        # the first 52 bins do not count.
        n = 2100
        keep = set(range(0, 40)) | set(range(n - 63, n))
        obs, vals = _series(n=n, drop=[i for i in range(n) if i not in keep])
        assert _refusal(_chronos2, obs, vals, horizon_steps=12) == (
            T.TOO_FEW_OBSERVED.format(observed_in_window=63)
        )

    def test_64_observed_bins_are_enough(self, stub):
        obs, vals = _series(n=64)
        assert _chronos2(obs, vals, horizon_steps=12)["status"] == "forecast"

    @pytest.mark.parametrize("horizon", [145, 256])
    def test_a_horizon_above_144_is_refused(self, stub, horizon):
        obs, vals = _series()
        assert _refusal(_chronos2, obs, vals, horizon_steps=horizon) == (
            T.HORIZON_ABOVE_144.format(horizon_steps=horizon)
        )

    def test_a_horizon_above_256_names_both_limits(self, stub):
        obs, vals = _series()
        text = _refusal(_chronos2, obs, vals, horizon_steps=300)
        assert text == (
            "The call has 2 problems: (1) horizon_steps must be an integer between 1 "
            "and 256, got 300. (2) " + T.HORIZON_ABOVE_144.format(horizon_steps=300)
        )

    def test_144_steps_are_accepted(self, stub):
        obs, vals = _series(n=400)
        assert _chronos2(obs, vals, horizon_steps=144)["status"] == "forecast"

    def test_a_horizon_beyond_the_observed_span_is_insufficient_history(self, stub):
        obs, vals = _series(n=100)
        r = _chronos2(obs, vals, horizon_steps=120)
        assert (r["status"], r["reason_code"]) == (
            "insufficient_history",
            "horizon_exceeds_history",
        )
        assert (r["available"], r["required"]) == (99, 120)
        assert r["max_supported_horizon_steps"] == 99
        assert r["reason"] == T.HORIZON_BEYOND_SPAN.format(
            required=120,
            duration=format_duration(120 * 3600),
            max_supported=99,
            resolution="1h",
        )
        assert r["remedy"] == (
            "Call again with horizon_steps of at most 99, or supply a longer history."
        )
        assert r["reference"] is None and r["provenance"]["model"] is None
        assert stub.loads == []  # no model was needed to say so

    def test_the_span_limit_is_capped_at_144_steps(self, stub):
        obs, vals = _series(n=400)
        r = _chronos2(obs, vals, horizon_steps=144)
        assert r["status"] == "forecast"
        assert (
            f._max_supported_horizon(*_grid_and_declaration(obs, vals, "chronos2"))
            == 144
        )

    def test_the_window_is_the_last_2048_bins_and_a_longer_grid_warns(self, stub):
        # 2,049 bins from 2,048 points (the input limit): one empty bin.
        obs, vals = _series(n=2049, drop=[1])
        r = _chronos2(obs, vals, horizon_steps=12)
        (call,) = stub.pipeline.calls
        assert call["x"].size == 2048
        assert r["provenance"]["model"]["context_bins"] == 2048
        truncated = [w for w in r["warnings"] if w["code"] == "model_context_truncated"]
        assert truncated == [
            {
                "code": "model_context_truncated",
                "detail": T.CONTEXT_TRUNCATED.format(grid_bins="2,049"),
            }
        ]

    def test_a_grid_of_2048_bins_does_not_warn(self, stub):
        obs, vals = _series(n=2048)
        r = _chronos2(obs, vals, horizon_steps=12)
        assert "model_context_truncated" not in [w["code"] for w in r["warnings"]]


def _grid_and_declaration(obs, vals, backend):
    from predictive_maintenance_mcp.prognostics.indicator_series import prepare_series

    declaration, grid = prepare_series(
        observed_at=obs,
        values=vals,
        indicator="velocity RMS",
        unit="mm/s",
        resolution="1h",
        horizon_steps=12,
        backend=backend,
        aggregation="mean",
        season_length_steps=None,
        now=NOW,
    )
    return grid, declaration


# ---------------------------------------------------------------------------
# The reference anchors
# ---------------------------------------------------------------------------


class TestReferenceAnchors:
    @pytest.mark.parametrize("horizon", [1, 2, 3, 4, 5, 24, 144])
    def test_anchors_are_the_direct_reference_at_the_forecast_steps(
        self, stub, horizon
    ):
        obs, vals = _series(n=400)
        r = _chronos2(obs, vals, horizon_steps=horizon)
        naive = _reference(obs, vals, horizon_steps=horizon)
        ref = r["reference"]
        assert ref["backend"] == "naive"
        assert ref["anchors"] == naive["anchors"]
        assert [a["step"] for a in ref["anchors"]] == [a["step"] for a in r["anchors"]]
        assert ref["note"] == T.REFERENCE_NOTE.format(reference_backend="naive")

    def test_a_declared_season_selects_the_seasonal_reference(self, stub):
        rng = np.random.default_rng(3)
        n = 24 * 12
        values = 10 + 2 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.1, n)
        obs, vals = _series(n=n, values=values)
        r = _chronos2(obs, vals, horizon_steps=24, season_length_steps=24)
        seasonal = _reference(
            obs,
            vals,
            backend="seasonal_naive",
            horizon_steps=24,
            season_length_steps=24,
        )
        assert r["reference"]["backend"] == "seasonal_naive"
        assert r["reference"]["anchors"] == seasonal["anchors"]
        assert r["reference"]["note"] == T.REFERENCE_NOTE.format(
            reference_backend="seasonal_naive"
        )
        assert r["provenance"]["effective_declaration"]["season_length_steps"] == 24
        assert r["history"]["difference_lag"] == 24
        # The model never sees the season: its input is the window, nothing more.
        (call,) = stub.pipeline.calls
        assert call["x"].size == n

    def test_the_reference_uses_the_whole_grid_not_only_the_window(self, stub):
        # 2,100 bins, 2,040 points: 40 contiguous, much noisier bins before the
        # model's window, so the whole grid and the window give other bands.
        rng = np.random.default_rng(11)
        values = np.empty(2100)
        values[:40] = 10 + np.cumsum(rng.normal(0, 5.0, 40))
        values[40:] = 10 + np.cumsum(rng.normal(0, 0.1, 2060))
        obs, vals = _series(n=2100, values=values, drop=range(40, 100))
        r = _chronos2(obs, vals, horizon_steps=12)
        whole = _reference(obs, vals, horizon_steps=12)["anchors"]
        window_only = _reference(obs[40:], vals[40:], horizon_steps=12)["anchors"]
        assert whole != window_only
        assert r["reference"]["anchors"] == whole
        assert "model_context_truncated" in [w["code"] for w in r["warnings"]]

    def test_too_few_difference_pairs_refuse_the_call(self, stub):
        # 64 observed bins, every other one empty: no adjacent pair.
        n = 128
        obs, vals = _series(n=n, drop=range(1, n - 1, 2))
        direct = _reference(obs, vals, horizon_steps=12)
        assert direct["reason_code"] == "too_few_difference_pairs"
        assert _refusal(_chronos2, obs, vals, horizon_steps=12) == (
            T.REFERENCE_NOT_COMPUTABLE.format(
                reference_backend="naive",
                reference_reason=direct["reason"],
                reference_remedy=direct["remedy"],
            )
        )
        assert stub.loads == []

    @pytest.mark.parametrize(
        "season, code",
        [(200, "season_coverage_too_short"), (24, "season_phase_unobserved")],
    )
    def test_a_seasonal_reference_that_cannot_be_computed_refuses_the_call(
        self, stub, season, code
    ):
        n = 300
        drop = (
            [i for i in range(n - 48, n - 1) if i % 24 == 5]
            if code.endswith("unobserved")
            else []
        )
        drop = drop + [i for i in range(5, n - 48, 24)] if drop else drop
        obs, vals = _series(n=n, drop=drop)
        direct = _reference(
            obs,
            vals,
            backend="seasonal_naive",
            horizon_steps=24,
            season_length_steps=season,
        )
        assert direct["reason_code"] == code
        assert _refusal(
            _chronos2, obs, vals, horizon_steps=24, season_length_steps=season
        ) == T.REFERENCE_NOT_COMPUTABLE.format(
            reference_backend="seasonal_naive",
            reference_reason=direct["reason"],
            reference_remedy=direct["remedy"],
        )

    def test_a_flat_history_refuses_the_call_with_the_reference_text(self, stub):
        obs, vals = _series(values=np.full(200, 4.2))
        direct = _reference(obs, vals)
        assert direct["reason_code"] == "no_variation"
        assert _refusal(_chronos2, obs, vals) == T.REFERENCE_NOT_COMPUTABLE.format(
            reference_backend="naive",
            reference_reason=direct["reason"],
            reference_remedy=direct["remedy"],
        )

    def test_a_flat_window_after_varying_older_bins_is_refused_before_any_load(
        self, stub
    ):
        # 2,100 bins, 2,048 points: the first 52 bins vary, every observed bin
        # of the model's window (the last 2,048 bins, 52 of them empty) holds
        # 4.2. The reference, on the whole grid, can be computed.
        rng = np.random.default_rng(12)
        values = np.full(2100, 4.2)
        values[:52] = 4.2 + np.cumsum(rng.normal(0, 0.1, 52))
        obs, vals = _series(n=2100, values=values, drop=range(100, 152))
        assert _reference(obs, vals, horizon_steps=12)["status"] == "forecast"
        assert _refusal(_chronos2, obs, vals, horizon_steps=12) == (
            T.FLAT_WINDOW.format(observed_in_window=1996)
        )
        assert stub.loads == [] and stub.pipeline.calls == []

    @pytest.mark.parametrize("where", [0, -1])
    def test_one_different_value_in_the_window_is_variation(self, stub, where):
        values = np.full(2100, 4.2)
        values[:52] = 4.2 + np.cumsum(np.random.default_rng(12).normal(0, 0.1, 52))
        values[52:][where] = 4.2000001
        obs, vals = _series(n=2100, values=values, drop=range(100, 152))
        assert _chronos2(obs, vals, horizon_steps=12)["status"] == "forecast"

    def test_a_reference_band_that_cannot_be_reported_refuses_the_call(self, stub):
        rng = np.random.default_rng(5)
        values = 1e12 + np.cumsum(rng.normal(0, 1e-3, 200))
        obs, vals = _series(values=values)
        with pytest.raises(InvalidInputError) as direct:
            _reference(obs, vals)
        assert _refusal(_chronos2, obs, vals) == T.REFERENCE_BAND_NOT_REPORTABLE.format(
            reference_backend="naive", reference_refusal=str(direct.value)
        )
        assert stub.loads == []


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    @pytest.mark.parametrize(
        "n, empty, fires",
        [(64, 4, True), (64, 3, False), (2048, 103, True), (2048, 102, False)],
    )
    def test_recent_gap_before_origin(self, stub, n, empty, fires):
        tail = -(-n // 10)
        # Empty bins just before the last observed bin, inside the window's tail.
        total = n + empty if n == 64 else n
        drop = list(range(total - 1 - empty, total - 1))
        obs, vals = _series(n=total, drop=drop)
        r = _chronos2(obs, vals, horizon_steps=12)
        gaps = [w for w in r["warnings"] if w["code"] == "recent_gap_before_origin"]
        window = min(total, 2048)
        expected_tail = -(-window // 10)
        if fires:
            assert gaps == [
                {
                    "code": "recent_gap_before_origin",
                    "detail": T.RECENT_GAP.format(
                        empty=empty, window_tail=expected_tail
                    ),
                }
            ]
        else:
            assert gaps == []
        assert tail == -(-n // 10)

    @pytest.mark.parametrize(
        "level, slope, sign",
        [
            (0.2, 0.0, "positive"),
            (-0.2, 0.0, "negative"),
        ],
    )
    def test_the_model_sign_warning(self, monkeypatch, tmp_path, level, slope, sign):
        install(monkeypatch, tmp_path, spread=1.0, slope=slope)
        rng = np.random.default_rng(6)
        base = np.abs(rng.normal(0, 0.01, 200)) + 0.05
        values = base if sign == "positive" else -base
        obs, vals = _series(values=values)
        r = _chronos2(obs, vals, horizon_steps=12)
        crossing = [
            w for w in r["warnings"] if w["code"] == "interval_crosses_observed_sign"
        ]
        assert crossing == [
            {"code": "interval_crosses_observed_sign", "detail": T.SIGN[sign]}
        ]

    @pytest.mark.parametrize("sign", ["zero_or_positive", "zero_or_negative"])
    def test_the_model_sign_warning_with_a_zero_reading(
        self, monkeypatch, tmp_path, sign
    ):
        install(monkeypatch, tmp_path, spread=1.0)
        rng = np.random.default_rng(7)
        base = np.abs(rng.normal(0, 0.01, 200)) + 0.05
        base[50] = 0.0
        values = base if sign == "zero_or_positive" else -base
        obs, vals = _series(values=values)
        r = _chronos2(obs, vals, horizon_steps=12)
        assert {"code": "interval_crosses_observed_sign", "detail": T.SIGN[sign]} in (
            r["warnings"]
        )

    def test_no_sign_warning_when_the_band_keeps_the_sign(self, stub):
        obs, vals = _series(level=100.0)
        r = _chronos2(obs, vals)
        assert "interval_crosses_observed_sign" not in [
            w["code"] for w in r["warnings"]
        ]

    def test_data_warnings_keep_their_reference_texts(self, stub):
        obs, vals = _series(end=NOW - 10 * STEP)
        r = _chronos2(obs, vals, horizon_steps=6)
        naive = _reference(obs, vals, horizon_steps=6)
        stale = [w for w in r["warnings"] if w["code"] == "stale_history"]
        assert stale and stale == [
            w for w in naive["warnings"] if w["code"] == "stale_history"
        ]
        assert r["warnings"][0]["code"] == "stale_history"


# ---------------------------------------------------------------------------
# The call's checks and their order
# ---------------------------------------------------------------------------


class TestCallChecks:
    def test_a_season_with_chronos2_is_validated_as_for_seasonal_naive(self, stub):
        obs, vals = _series()
        for bad in (1, True, "24"):
            assert _refusal(_chronos2, obs, vals, season_length_steps=bad) == (
                f"season_length_steps must be an integer of at least 2, got {bad!r}."
            )

    def test_a_season_with_naive_keeps_its_refusal(self):
        obs, vals = _series()
        assert _refusal(_reference, obs, vals, season_length_steps=24) == (
            "season_length_steps applies only to the seasonal_naive backend: omit it "
            "for the naive backend."
        )

    def test_the_backend_list_names_chronos2(self):
        obs, vals = _series()
        assert _refusal(_reference, obs, vals, backend="arima") == (
            "backend must be one of ['naive', 'seasonal_naive', 'chronos2'], "
            "got 'arima'."
        )

    def test_the_sync_entry_point_refuses_chronos2(self):
        obs, vals = _series()
        with pytest.raises(ValueError, match="forecast_indicator_chronos2"):
            _reference(obs, vals, backend="chronos2")

    def test_availability_comes_before_the_history_checks(self, stub, monkeypatch):
        monkeypatch.delenv(cb.ENV_MODEL_DIR)
        obs, vals = _series(n=20)  # also too short for the model
        assert _refusal(_chronos2, obs, vals, horizon_steps=6) == T.DIR_NOT_SET

    def test_a_malformed_call_is_refused_before_availability(self, stub, monkeypatch):
        monkeypatch.delenv(cb.ENV_MODEL_DIR)
        obs, vals = _series()
        assert _refusal(_chronos2, obs, vals, resolution="1 hour").startswith(
            "resolution '1 hour' is not a grid step"
        )


# ---------------------------------------------------------------------------
# A model band the quantiles cannot carry
# ---------------------------------------------------------------------------


class TestModelBand:
    def test_a_model_band_too_narrow_for_its_level_is_refused(
        self, monkeypatch, tmp_path
    ):
        install(monkeypatch, tmp_path, spread=1e-6)
        rng = np.random.default_rng(8)
        values = 1e5 + np.cumsum(rng.normal(0, 1.0, 200))
        obs, vals = _series(values=values)
        text = _refusal(_chronos2, obs, vals)
        assert text.startswith(
            "The projection's band is too narrow to report at the level of these values: "
            "its narrowest half-width is 1e-06 mm/s on values near"
        )
        assert "subtract the last observed value exactly from every value" in text
        assert "season-to-season spread gives" not in text

    def test_a_model_band_below_the_values_resolution_is_refused(
        self, monkeypatch, tmp_path
    ):
        # The reference's band is reportable (steps of 100 at 1e9); the model's
        # is narrower than a thousand float64 spacings of the values.
        install(monkeypatch, tmp_path, spread=1e-9)
        rng = np.random.default_rng(9)
        values = 1e9 + np.cumsum(rng.normal(0, 100.0, 200))
        obs, vals = _series(values=values)
        text = _refusal(_chronos2, obs, vals)
        assert "less than 1,000 times the floating-point spacing of the values" in text

    @staticmethod
    def _model_run(p50: np.ndarray, half: float) -> cb.ModelRun:
        return cb.ModelRun(
            p10=p50 - half,
            p50=p50,
            p90=p50 + half,
            half_band=half,
            context_bins=200,
            torch_threads=4,
            versions=dict(IN_RANGE),
        )

    def test_a_model_band_never_gets_the_seasonal_or_far_median_text(self):
        # A median spanning 100 across the horizon, far from values near 1,
        # with a half-band of 1e-6: for a reference method this would name the
        # seasonal pattern (first) or the far median. For a model band only
        # the generic level text applies.
        p50 = np.linspace(5.0, 105.0, 24)
        raw = np.full(200, 1.0)
        with pytest.raises(InvalidInputError) as exc:
            f._refuse_unreportable_band(self._model_run(p50, 1e-6), "mm/s", raw)
        assert str(exc.value) == (
            "The projection's band is too narrow to report at the level of these "
            "values: its narrowest half-width is 1e-06 mm/s on values near 105 mm/s, "
            "less than one part in 10,000,000 of the level, and the quantiles carry at "
            "most 10 significant digits, so the band could round to zero width and read "
            "as certainty. The values sent still resolve this variation: subtract the "
            "last observed value exactly from every value, name it in the indicator "
            "(for example 'energy minus 123456 kWh') and add it back when quoting the "
            "quantiles. This removes the level the band is measured against; with the "
            "seasonal_naive backend it may not be enough when the seasonal pattern is "
            "large next to its season-to-season spread."
        )

    def test_a_model_band_under_the_float_resolution_gets_that_text(self):
        p50 = np.full(24, 1e9)
        raw = np.full(200, 1e9)
        with pytest.raises(InvalidInputError) as exc:
            f._refuse_unreportable_band(self._model_run(p50, 1e-9), "mm/s", raw)
        assert str(exc.value).startswith(
            "The projection's band is too narrow to report at the level of these "
            "values: its narrowest half-width is 1e-09 mm/s on values near 1000000000 "
            "mm/s, less than 1,000 times the floating-point spacing of the values"
        )


# ---------------------------------------------------------------------------
# The reference backends stay as they were
# ---------------------------------------------------------------------------


class TestReferenceBackendsUntouched:
    def test_the_chronos2_checks_never_run_for_a_reference_call(
        self, stub, monkeypatch
    ):
        calls: list[str] = []
        for name in ("check_available", "installed_versions", "project"):
            real = getattr(cb, name)
            monkeypatch.setattr(
                cb,
                name,
                lambda *a, _n=name, _r=real, **k: calls.append(_n) or _r(*a, **k),
            )
        obs, vals = _series(n=24 * 12)
        _reference(obs, vals)
        _reference(obs, vals, backend="seasonal_naive", season_length_steps=24)
        assert calls == [] and stub.loads == []

    def test_out_of_range_versions_leave_reference_results_identical(
        self, monkeypatch, tmp_path
    ):
        obs, vals = _series()
        before = json.dumps(_reference(obs, vals), sort_keys=True)
        install(monkeypatch, tmp_path, versions={**IN_RANGE, "torch": "9.0.0"})
        assert json.dumps(_reference(obs, vals), sort_keys=True) == before


# ---------------------------------------------------------------------------
# The tool over MCP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    mcp = MCPServer("test-forecast-chronos2")
    register_all(mcp)
    return mcp


def _arguments(obs, vals, horizon=24, label="velocity RMS", **extra):
    return {
        "observed_at": obs,
        "values": vals,
        "indicator": label,
        "unit": "mm/s",
        "resolution": "1h",
        "horizon_steps": horizon,
        "backend": "chronos2",
        **extra,
    }


async def _call(server, arguments):
    async with Client(server, mode="legacy") as client:
        return await client.call_tool(
            "forecast_asset_indicator", arguments, read_timeout_seconds=60
        )


class TestTool:
    @pytest.mark.asyncio
    async def test_a_refusal_reaches_the_client_with_its_text(
        self, server, stub, monkeypatch
    ):
        monkeypatch.delenv(cb.ENV_MODEL_DIR)
        obs, vals = _series(end=datetime.now(timezone.utc) - STEP)
        result = await _call(server, _arguments(obs, vals))
        assert result.is_error
        assert result.content[0].text == (
            f"Error executing tool forecast_asset_indicator: {T.DIR_NOT_SET}"
        )

    @pytest.mark.asyncio
    async def test_a_forecast_reaches_the_client_with_its_blocks(self, server, stub):
        obs, vals = _series(end=datetime.now(timezone.utc) - STEP)
        result = await _call(server, _arguments(obs, vals))
        assert not result.is_error, result.content[0].text
        model = IndicatorForecastResult.model_validate(result.structured_content)
        assert model.backend == "chronos2"
        assert model.reference is not None and model.provenance.model is not None

    @pytest.mark.asyncio
    async def test_horizon_144_with_the_longest_labels_fits_with_room(
        self, server, monkeypatch, tmp_path
    ):
        from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
            RETRY_RESERVE,
        )
        from predictive_maintenance_mcp.signal_acquisition.measurement import (
            MAX_FREE_TEXT_CHARS,
        )

        install(monkeypatch, tmp_path, spread=0.05)  # the most digits a band needs
        rng = np.random.default_rng(10)
        values = -1.2345678e5 + np.cumsum(rng.normal(0, 1e-2, 2048))
        obs, vals = _series(
            n=2048, values=values, end=datetime.now(timezone.utc) - STEP
        )
        label = "x" * MAX_FREE_TEXT_CHARS
        result = await _call(server, _arguments(obs, vals, horizon=144, label=label))
        # The largest chronos2 result, its reference block included, fits with
        # the retry reserve to spare: growth shows here before calls are refused.
        assert not result.is_error, result.content[0].text
        model = IndicatorForecastResult.model_validate(result.structured_content)
        assert model.reference is not None
        assert client_result_bytes(model) <= f.MAX_RESULT_BYTES - RETRY_RESERVE

    @pytest.mark.asyncio
    async def test_the_size_search_measures_model_results(
        self, server, stub, monkeypatch
    ):
        from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
            RETRY_RESERVE,
        )

        obs, vals = _series(n=400, end=datetime.now(timezone.utc) - STEP)
        args = (None, obs, vals, "velocity RMS", "mm/s", "1h")
        fits = await forecast_asset_indicator(*args, 40, backend="chronos2")
        monkeypatch.setattr(
            f, "MAX_RESULT_BYTES", client_result_bytes(fits) + RETRY_RESERVE + 64
        )
        before = len(stub.pipeline.calls)
        result = await _call(server, _arguments(obs, vals, horizon=144))
        assert result.is_error
        named = int(
            result.content[0]
            .text.split("Call again with horizon_steps of ")[1]
            .split(":")[0]
        )
        assert 40 <= named < 144
        # The requested horizon and every candidate were new inferences.
        assert len(stub.pipeline.calls) - before > 2

    @pytest.mark.asyncio
    async def test_the_event_loop_stays_free_during_a_chronos2_call(self, stub):
        stub.delay = 1.0
        obs, vals = _series(end=datetime.now(timezone.utc) - STEP)
        lags: list[float] = []
        done = False

        async def heartbeat():
            while not done:
                start = time.monotonic()
                await asyncio.sleep(0.01)
                lags.append(time.monotonic() - start - 0.01)

        beat = asyncio.ensure_future(heartbeat())
        await asyncio.sleep(0.05)
        result = await forecast_asset_indicator(
            None, obs, vals, "velocity RMS", "mm/s", "1h", 24, backend="chronos2"
        )
        done = True
        await beat
        assert result.backend == "chronos2"
        assert len(lags) > 20 and max(lags) < 0.25
