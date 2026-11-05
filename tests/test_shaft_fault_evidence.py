"""Diagnostic specificity of the experimental shaft-order evidence assessment.

Adversarial tests against simple but misleading heuristics (a high 1X means
unbalance, a high 2X means misalignment, many harmonics mean looseness, an
ISO zone names a fault), the status rules, and invariants of the
model-facing wording. Synthetic signals only: passing them shows the rules
behave as designed, not that they detect faults on real machines.
"""

import re

import numpy as np
import pytest

from predictive_maintenance_mcp.diagnostics.shaft_fault_evidence import (
    ALTERNATIVES,
    INCONCLUSIVE,
    INSUFFICIENT_DATA,
    MAX_CHANNELS,
    NOT_ASSESSABLE_FROM_RECORD,
    NOT_OBSERVED,
    REASONS,
    SOURCES,
    assess_shaft_fault_evidence,
    with_detail,
)
from predictive_maintenance_mcp.models import ShaftFaultEvidenceResult
from predictive_maintenance_mcp.signal_validation import InvalidInputError

FS = 10000.0
F1 = 24.7  # shaft frequency; declared speed 1500 rpm
T = np.arange(int(10 * FS)) / FS


def _signal(components, noise=0.02, seed=0):
    """Velocity signal in mm/s: [(amplitude, order, phase)] of F1 plus noise."""
    x = np.zeros_like(T)
    for amplitude, order, phase in components:
        x += amplitude * np.sin(2 * np.pi * order * F1 * T + phase)
    return x + noise * np.random.default_rng(seed).standard_normal(T.size)


def _channel(sid, components, direction="horizontal", unit="mm/s", **extra):
    return {
        "signal": _signal(components, seed=sum(map(ord, sid))),
        "fs": FS,
        "signal_id": sid,
        "signal_unit": unit,
        "direction": direction,
        **extra,
    }


def _assess(channels, **kwargs):
    kwargs.setdefault("machine_group", 2)
    kwargs.setdefault("support_type", "rigid")
    return assess_shaft_fault_evidence(channels, 1500, **kwargs)


def _family(result, family):
    return next(h for h in result["hypotheses"] if h["family"] == family)


class TestMisleadingHeuristics:
    def test_high_1x_without_a_severity_context_is_not_unbalance(self):
        r = _assess(
            [_channel("h", [(12.0, 1, 0)])], machine_group=None, support_type=None
        )
        unbalance = _family(r, "unbalance")
        assert unbalance["status"] == INCONCLUSIVE
        assert unbalance["reasons"] == [
            "level_class_undeclared",
            "attribution_not_validated",
        ]
        assert "inconclusive" in unbalance["statement"]

    def test_1x_dominant_at_an_acceptable_level_is_not_unbalance(self):
        r = _assess([_channel("h", [(1.0, 1, 0)])])  # about 0.7 mm/s RMS: zone A
        unbalance = _family(r, "unbalance")
        assert r["level_context"]["zone"] == "A"
        assert unbalance["status"] == INCONCLUSIVE
        assert unbalance["reasons"] == [
            "level_not_elevated",
            "attribution_not_validated",
        ]

    def test_high_2x_on_a_radial_channel_is_not_misalignment(self):
        r = _assess([_channel("h", [(4.0, 1, 0), (9.0, 2, 0.3)])])
        misalignment = _family(r, "misalignment")
        assert r["level_context"]["zone"] == "D"
        assert misalignment["status"] == INCONCLUSIVE
        assert misalignment["reasons"] == [
            "axial_channel_missing",
            "single_record_insufficient",
            "attribution_not_validated",
        ]
        assert misalignment["supporting"][0]["values"]["order"] == 2.0

    def test_a_harmonic_series_is_not_looseness(self):
        series = [(8.0 / k, k, 0.1 * k) for k in range(1, 9)]
        r = _assess([_channel("h", series)])
        assert _family(r, "looseness")["status"] == NOT_OBSERVED

    def test_a_severity_zone_alone_names_no_fault(self):
        # Broadband vibration in zone D with no shaft-order line standing out.
        noisy = {
            "signal": 15.0 * np.random.default_rng(11).standard_normal(T.size),
            "fs": FS,
            "signal_id": "broadband",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([noisy])
        assert r["level_context"]["zone"] == "D"
        statuses = {h["family"]: h["status"] for h in r["hypotheses"]}
        assert "consistent" not in statuses.values()
        assert statuses["unbalance"] in (NOT_OBSERVED, INSUFFICIENT_DATA)


class TestConsistentPatterns:
    def test_unbalance_pattern_with_elevated_vibration_is_not_attributed(self):
        r = _assess(
            [
                _channel("h", [(12.0, 1, 0), (0.5, 2, 0)]),
                _channel("v", [(10.0, 1, 1.5), (0.4, 2, 0)], direction="vertical"),
            ]
        )
        unbalance = _family(r, "unbalance")
        assert unbalance["status"] == INCONCLUSIVE
        assert unbalance["pattern_observed"] is True
        assert unbalance["reasons"] == ["attribution_not_validated"]
        assert "Fault attribution is inconclusive" in unbalance["statement"]
        assert "compatible with rotor unbalance and equally with each of" in (
            unbalance["statement"]
        )
        assert "context that identifies no fault" in unbalance["statement"]
        for alternative in ALTERNATIVES["unbalance"]:
            assert alternative in unbalance["statement"]
        assert unbalance["supporting"][0]["values"]["one_x_share"] > 0.5
        assert unbalance["supporting"][0]["values"]["iso_zone"] == "D"

    def test_misalignment_with_an_axial_channel_stays_inconclusive(self):
        r = _assess(
            [
                _channel("h", [(5.0, 1, 0), (1.0, 2, 0)]),
                _channel("a", [(3.0, 1, 0), (6.0, 2, 0)], direction="axial"),
            ]
        )
        misalignment = _family(r, "misalignment")
        assert misalignment["status"] == INCONCLUSIVE
        assert misalignment["reasons"] == [
            "single_record_insufficient",
            "attribution_not_validated",
        ]
        assert {o["code"] for o in misalignment["supporting"]} == {
            "axial_exceeds_radial"
        }

    def test_half_orders_support_looseness_but_never_make_it_consistent(self):
        # Elevated vibration, located shaft line, declared direction: every
        # other condition of 'consistent' is met.
        r = _assess([_channel("h", [(10.0, 1, 0), (1.0, 1.5, 0), (0.8, 2.5, 0)])])
        looseness = _family(r, "looseness")
        unbalance = _family(r, "unbalance")
        assert r["level_context"]["zone"] == "D"
        assert looseness["status"] == INCONCLUSIVE
        assert looseness["reasons"] == [
            "half_orders_not_specific",
            "attribution_not_validated",
        ]
        assert looseness["supporting"][0]["values"]["orders"] == [1.5, 2.5]
        assert unbalance["status"] == INCONCLUSIVE
        assert "conflicting_observations" in unbalance["reasons"]
        assert unbalance["conflicting"][0]["code"] == "half_orders_present"

    def test_higher_half_orders_are_reported_as_lines_but_not_as_the_pattern(self):
        r = _assess([_channel("h", [(10.0, 1, 0), (1.0, 3.5, 0), (1.0, 6.5, 0)])])
        assert {3.5, 6.5} <= set(r["channels"][0]["significant_orders"])
        assert _family(r, "looseness")["status"] == NOT_OBSERVED
        assert _family(r, "unbalance")["conflicting"] == []


class TestConfoundersAndLimits:
    def test_undeclared_direction_keeps_the_pattern_inconclusive(self):
        r = _assess([_channel("c", [(12.0, 1, 0)], direction=None)])
        assert _family(r, "unbalance")["reasons"] == [
            "direction_undeclared",
            "level_not_evaluable",
            "attribution_not_validated",
        ]

    def test_undeclared_unit_leaves_velocity_patterns_without_data(self):
        r = _assess([_channel("h", [(12.0, 1, 0), (1.0, 1.5, 0)], unit=None)])
        assert _family(r, "unbalance")["status"] == INSUFFICIENT_DATA
        assert _family(r, "unbalance")["reasons"] == ["unit_undeclared"]
        assert _family(r, "misalignment")["status"] == INSUFFICIENT_DATA
        looseness = _family(r, "looseness")
        assert looseness["status"] == INCONCLUSIVE
        assert looseness["reasons"] == [
            "level_unit_undeclared",
            "half_orders_not_specific",
            "attribution_not_validated",
        ]

    def test_a_generated_signal_without_noise_has_insufficient_data(self):
        r = _assess(
            [
                _channel(
                    "h", [(12.0, 1, 0)], acquisition_eligibility="known_inapplicable"
                )
            ]
        )
        for family in ("unbalance", "misalignment", "looseness"):
            assert _family(r, family)["status"] == INSUFFICIENT_DATA
            assert _family(r, family)["reasons"] == ["generated_without_noise"]

    def test_twice_line_frequency_inside_the_2x_window_is_a_confounder(self):
        # Declared speed only (no 1x line), so the 2X window is wide and
        # contains 100 Hz.
        x = {
            "signal": 6.0 * np.sin(2 * np.pi * 100.0 * T)
            + 0.02 * np.random.default_rng(3).standard_normal(T.size)
            + 3.0 * np.sin(2 * np.pi * 49.0 * T)
            + 3.0 * np.sin(2 * np.pi * 50.2 * T),
            "fs": FS,
            "signal_id": "motor",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = assess_shaft_fault_evidence(
            [x], 2970, machine_group=2, support_type="rigid", line_frequency_hz=50.0
        )
        assert r["channels"][0]["speed"]["source"] == "declared"
        misalignment = _family(r, "misalignment")
        assert misalignment["status"] == INCONCLUSIVE
        assert "line_frequency_multiple" in misalignment["reasons"]
        assert "shaft_frequency_declared" in misalignment["reasons"]

    def test_a_record_whose_speed_changed_is_left_out_of_every_pattern(self):
        f = np.where(T < 5.0, 24.7, 25.2)
        phase = 2 * np.pi * np.cumsum(f) / FS
        x = {
            "signal": 12.0 * np.sin(phase)
            + 0.02 * np.random.default_rng(4).standard_normal(T.size),
            "fs": FS,
            "signal_id": "h",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([x])
        assert r["channels"][0]["stationarity"]["speed_changed"] is True
        for family in ("unbalance", "misalignment", "looseness"):
            assert _family(r, family)["status"] == INSUFFICIENT_DATA
            assert _family(r, family)["reasons"] == ["speed_changed"]

    def test_a_steady_channel_is_still_assessed_next_to_one_whose_speed_changed(self):
        f = np.where(T < 5.0, 24.7, 25.2)
        drifting = {
            "signal": 12.0 * np.sin(2 * np.pi * np.cumsum(f) / FS)
            + 0.02 * np.random.default_rng(4).standard_normal(T.size),
            "fs": FS,
            "signal_id": "drift",
            "signal_unit": "mm/s",
            "direction": "vertical",
        }
        r = _assess([drifting, _channel("h", [(12.0, 1, 0)])])
        unbalance = _family(r, "unbalance")
        assert unbalance["status"] == INCONCLUSIVE
        assert unbalance["reasons"] == ["attribution_not_validated"]
        assert [o["channels"] for o in unbalance["supporting"]] == [["h"]]

    def test_a_record_too_short_has_insufficient_data(self):
        short = {
            "signal": 12.0 * np.sin(2 * np.pi * F1 * T[:2000]),
            "fs": FS,
            "signal_id": "short",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([short])
        assert _family(r, "unbalance")["status"] == INSUFFICIENT_DATA
        assert _family(r, "looseness")["status"] == INSUFFICIENT_DATA

    def test_resonance_is_never_assessed_from_a_record(self):
        r = _assess([_channel("h", [(12.0, 1, 0)])])
        resonance = _family(r, "resonance")
        assert resonance["status"] == NOT_ASSESSABLE_FROM_RECORD
        assert "cannot establish or exclude" in resonance["statement"]


class TestRefusals:
    def test_no_channel(self):
        with pytest.raises(InvalidInputError, match="At least one channel"):
            assess_shaft_fault_evidence([], 1500)

    def test_too_many_channels(self):
        channels = [_channel(f"c{n}", [(1.0, 1, 0)]) for n in range(MAX_CHANNELS + 1)]
        with pytest.raises(InvalidInputError, match="At most"):
            assess_shaft_fault_evidence(channels, 1500)

    def test_duplicate_channels(self):
        with pytest.raises(InvalidInputError, match="different signal"):
            assess_shaft_fault_evidence(
                [_channel("h", [(1.0, 1, 0)]), _channel("h", [(1.0, 1, 0)])], 1500
            )


_CASES = [
    [_channel("h", [(12.0, 1, 0)])],
    [_channel("h", [(1.0, 1, 0)])],
    [_channel("h", [(4.0, 1, 0), (9.0, 2, 0)])],
    [
        _channel("h", [(10.0, 1, 0), (1.0, 1.5, 0)]),
        _channel("a", [(3.0, 1, 0), (6.0, 2, 0)], "axial"),
    ],
    [_channel("h", [(12.0, 1, 0)], unit=None, direction=None)],
]


class TestWording:
    @pytest.mark.parametrize("channels", _CASES)
    def test_server_sentences_do_not_overclaim(self, channels):
        r = _assess(channels)
        sentences = [r["summary"], r["scope_statement"]] + [
            h["statement"] for h in r["hypotheses"]
        ]
        for s in sentences:
            assert "—" not in s  # no em-dash in prose
            assert not re.search(r"\bhealthy\b|\bconfirm", s, re.IGNORECASE)
            assert not re.search(r"confidence|probability|likely", s, re.IGNORECASE)
            for match in re.finditer(r"diagnosis", s):
                window = s[max(0, match.start() - 20) : match.end()]
                assert "not a diagnosis" in window or "is a diagnosis" in window, s

    @pytest.mark.parametrize("channels", _CASES)
    def test_every_family_names_alternatives_missing_data_and_sources(self, channels):
        r = _assess(channels)
        for h in r["hypotheses"]:
            assert h["missing"]
            assert set(h["sources"]) <= set(SOURCES)
            for o in h["supporting"] + h["conflicting"]:
                assert set(o["sources"]) <= set(SOURCES)
            if h["family"] != "resonance":
                assert h["alternatives"]
            for reason in h["reasons"]:
                assert reason in REASONS

    @pytest.mark.parametrize("channels", _CASES)
    def test_result_fits_the_public_model(self, channels):
        r = _assess(channels)
        model = ShaftFaultEvidenceResult(
            **with_detail(r, "full"),
            rpm=1500.0,
            parameter_sources={"rpm": "explicit"},
        )
        assert model.experimental is True

    def test_not_observed_never_claims_absence_of_the_fault(self):
        r = _assess([_channel("h", [(1.0, 1, 0)])])
        looseness = _family(r, "looseness")
        assert looseness["status"] == NOT_OBSERVED
        assert "does not exclude" in looseness["statement"]


class TestReviewFixes:
    @pytest.mark.parametrize("max_order", [2, 3])
    def test_low_max_order_does_not_crash(self, max_order):
        r = _assess([_channel("h", [(4.0, 1, 0), (9.0, 2, 0)])], max_order=max_order)
        assert _family(r, "misalignment")["supporting"]
        assert r["channels"][0]["lines"][-1]["order"] == float(max_order)

    def test_the_zone_must_be_on_the_channel_with_the_pattern(self):
        # h: 1X-dominant at zone B; v: no 1X dominance but zone D (broadband).
        broadband = {
            "signal": 15.0 * np.random.default_rng(21).standard_normal(T.size),
            "fs": FS,
            "signal_id": "v",
            "signal_unit": "mm/s",
            "direction": "vertical",
        }
        r = _assess([_channel("h", [(3.0, 1, 0)]), broadband])
        zones = {z["signal_id"]: z["zone"] for z in r["level_context"]["zones"]}
        assert zones == {"h": "B", "v": "D"}
        unbalance = _family(r, "unbalance")
        assert unbalance["status"] == INCONCLUSIVE
        assert unbalance["reasons"] == [
            "level_not_elevated",
            "attribution_not_validated",
        ]
        assert "zone B" in unbalance["statement"]

    def test_a_channel_whose_speed_changed_sets_no_zone(self):
        f = np.where(T < 5.0, 24.7, 25.2)
        drifting = {
            "signal": 12.0 * np.sin(2 * np.pi * np.cumsum(f) / FS),
            "fs": FS,
            "signal_id": "drift",
            "signal_unit": "mm/s",
            "direction": "vertical",
        }
        r = _assess([drifting, _channel("h", [(1.0, 1, 0)])])
        assert [z["signal_id"] for z in r["level_context"]["zones"]] == ["h"]

    def test_power_below_the_iso_scope_is_not_a_valid_context(self):
        r = _assess([_channel("h", [(12.0, 1, 0)])], machine_power_kw=5.0)
        assert r["level_context"]["reason"] == "level_outside_scope"
        assert _family(r, "unbalance")["reasons"] == [
            "level_outside_scope",
            "attribution_not_validated",
        ]

    def test_a_harmonic_above_the_floor_without_1x_is_reported(self):
        r = _assess([_channel("h", [(0.0, 1, 0), (5.0, 2, 0)])])
        codes = {o["code"] for o in _family(r, "misalignment")["supporting"]}
        assert "harmonic_above_floor_without_one_x" in codes

    def test_axial_line_above_a_radial_line_below_the_floor_counts(self):
        r = _assess(
            [
                _channel("h", [(5.0, 1, 0)]),
                _channel("a", [(1.0, 1, 0), (2.0, 2, 0)], direction="axial"),
            ]
        )
        found = [
            o
            for o in _family(r, "misalignment")["supporting"]
            if o["code"] == "axial_exceeds_radial"
        ]
        assert found and found[0]["values"] == {"order": 2.0, "ratio": None}
        assert "assuming the channels are at the same bearing" in found[0]["statement"]

    def test_channels_at_different_locations_are_not_compared(self):
        r = _assess(
            [
                _channel("h", [(5.0, 1, 0), (1.0, 2, 0)], location="P/motor_de"),
                _channel(
                    "a", [(3.0, 1, 0), (6.0, 2, 0)], "axial", location="P/pump_de"
                ),
            ]
        )
        assert {x["status"] for x in r["cross_channel"]} == {"not_comparable_location"}
        codes = {o["code"] for o in _family(r, "misalignment")["supporting"]}
        assert "axial_exceeds_radial" not in codes

    def test_iso_refusal_detail_is_kept(self):
        fs = 2000.0
        t = np.arange(int(10 * fs)) / fs
        low = {
            "signal": 12.0 * np.sin(2 * np.pi * F1 * t)
            + 0.02 * np.random.default_rng(5).standard_normal(t.size),
            "fs": fs,
            "signal_id": "low",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([low])
        assert r["level_context"]["reason"] == "level_not_evaluable"
        assert r["level_context"]["detail"].startswith("low: ")

    def test_directions_are_normalized_and_validated(self):
        r = _assess([_channel("h", [(12.0, 1, 0)], direction="H")])
        assert r["channels"][0]["direction"] == "horizontal"
        with pytest.raises(InvalidInputError, match="Invalid direction"):
            _assess([_channel("h", [(12.0, 1, 0)], direction="sideways")])

    def test_a_constant_record_has_no_screening_statistics(self):
        flat = {
            "signal": np.full(T.size, 0.3),
            "fs": FS,
            "signal_id": "flat",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([flat])
        td = r["channels"][0]["time_domain"]
        assert td["kurtosis_excess"] is None and td["crest_factor"] is not None


class TestVocabularySync:
    def test_reasons_match_the_model(self):
        from typing import get_args

        from predictive_maintenance_mcp.models import ShaftFaultReason

        assert set(REASONS) == set(get_args(ShaftFaultReason))

    def test_observation_codes_match_the_model(self):
        import inspect as _inspect
        from typing import get_args

        from predictive_maintenance_mcp.diagnostics import shaft_fault_evidence
        from predictive_maintenance_mcp.models import ShaftObservationCode

        source = _inspect.getsource(shaft_fault_evidence)
        used = set(re.findall(r'_observation\(\s*"([a-z_]+)"', source))
        assert used == set(get_args(ShaftObservationCode))

    def test_families_statuses_and_directions_match_the_model(self):
        from typing import get_args

        from predictive_maintenance_mcp.diagnostics import shaft_fault_evidence as e
        from predictive_maintenance_mcp.models import (
            ShaftDirection,
            ShaftFaultFamily,
            ShaftFaultStatus,
        )
        from predictive_maintenance_mcp.signal_acquisition.measurement import (
            VALID_DIRECTIONS,
        )

        assert set(e.FAMILIES) == set(get_args(ShaftFaultFamily))
        statuses = {
            e.INSUFFICIENT_DATA,
            e.NOT_OBSERVED,
            e.INCONCLUSIVE,
            e.NOT_ASSESSABLE_FROM_RECORD,
        }
        assert statuses == set(get_args(ShaftFaultStatus))
        assert set(VALID_DIRECTIONS) == set(get_args(ShaftDirection))
        assert set(e.PATTERN_DEFINITIONS) == set(e.FAMILIES)


class TestStepAndPinnedOutcomes:
    def test_a_large_speed_step_makes_the_1x_ambiguous_and_declared(self):
        f = np.where(T < 5.0, 24.7, 26.5)
        x = {
            "signal": 12.0 * np.sin(2 * np.pi * np.cumsum(f) / FS)
            + 0.02 * np.random.default_rng(6).standard_normal(T.size),
            "fs": FS,
            "signal_id": "step",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        r = _assess([x])
        c = r["channels"][0]
        assert c["stationarity"]["status"] == "not_assessed"
        assert c["speed"]["source"] == "declared"
        assert _family(r, "unbalance")["status"] in (INCONCLUSIVE, NOT_OBSERVED)

    def test_broadband_zone_d_leaves_unbalance_not_observed(self):
        noisy = {
            "signal": 15.0 * np.random.default_rng(11).standard_normal(T.size),
            "fs": FS,
            "signal_id": "broadband",
            "signal_unit": "mm/s",
            "direction": "horizontal",
        }
        assert _family(_assess([noisy]), "unbalance")["status"] == NOT_OBSERVED

    def test_a_dead_channel_fits_the_public_model(self):
        dead = {
            "signal": np.zeros(T.size),
            "fs": FS,
            "signal_id": "dead",
            "signal_unit": "mm/s",
            "direction": "vertical",
        }
        r = _assess([_channel("h", [(12.0, 1, 0)]), dead])
        for detail in ("summary", "full"):
            ShaftFaultEvidenceResult(
                **with_detail(r, detail),
                rpm=1500.0,
                parameter_sources={"rpm": "explicit"},
            )


class TestOwnerContract:
    def test_no_family_is_ever_attributed(self):
        cases = [
            [
                _channel("h", [(12.0, 1, 0)]),
                _channel("v", [(10.0, 1, 1.5)], "vertical"),
            ],
            [
                _channel("h", [(5.0, 1, 0), (1.0, 2, 0)]),
                _channel("a", [(3.0, 1, 0), (6.0, 2, 0)], "axial"),
            ],
            [_channel("h", [(10.0, 1, 0), (1.0, 1.5, 0), (0.8, 2.5, 0)])],
        ]
        for channels in cases:
            for h in _assess(channels)["hypotheses"]:
                assert h["status"] in (
                    INSUFFICIENT_DATA,
                    NOT_OBSERVED,
                    INCONCLUSIVE,
                    NOT_ASSESSABLE_FROM_RECORD,
                )
                if h["status"] == INCONCLUSIVE:
                    assert "attribution_not_validated" in h["reasons"]

    def test_pattern_observed_separates_observed_absent_and_unassessed(self):
        r = _assess([_channel("h", [(10.0, 1, 0), (1.0, 1.5, 0)])])
        observed = {h["family"]: h["pattern_observed"] for h in r["hypotheses"]}
        assert observed == {
            "unbalance": True,
            "misalignment": False,
            "looseness": True,
            "resonance": None,
        }
        unassessed = _assess([_channel("h", [(10.0, 1, 0)], unit=None)])
        assert _family(unassessed, "unbalance")["pattern_observed"] is None

    def test_limitations_state_the_measured_false_alarm_rate(self):
        r = _assess([_channel("h", [(12.0, 1, 0)])])
        joined = " ".join(r["limitations"])
        assert "not an empirically demonstrated guarantee" in joined
        assert "1.33 %" in joined
        assert "identifies no fault" in joined

    def test_summary_keeps_the_evidence_and_drops_the_bulk(self):
        r = _assess(
            [
                _channel("h", [(5.0, 1, 0), (1.0, 2, 0)]),
                _channel("a", [(3.0, 1, 0), (6.0, 2, 0)], "axial"),
            ]
        )
        summary, full = with_detail(r, "summary"), with_detail(r, "full")
        assert summary["detail"] == "summary" and full["detail"] == "full"
        assert summary["hypotheses"] == full["hypotheses"]
        assert summary["limitations"] == full["limitations"]
        assert summary["level_context"] == full["level_context"]
        for cs, cf in zip(summary["channels"], full["channels"]):
            assert all(line["significant"] for line in cs["lines"])
            assert [l["order"] for l in cs["lines"]] == cs["significant_orders"]
            assert len(cf["lines"]) == 20
            assert cs["untestable_orders"] == cf["untestable_orders"]
        assert all(x["status"] == "computed" for x in summary["cross_channel"])
        assert set(summary["order_line_policy"]) == {
            "method",
            "version",
            "false_alarm_probability_per_call",
            "measured_false_alarm_rates",
        }
        assert "1.33 %" in summary["order_line_policy"]["measured_false_alarm_rates"]
        assert "assumptions" in full["order_line_policy"]
        assert all(s["supports"] is None for s in summary["sources"])
        assert all(s["supports"] for s in full["sources"])
        assert r["detail"] if "detail" in r else True  # the input is not modified
        assert "detail" not in r

    def test_views_are_deterministic(self):
        channels = [_channel("h", [(12.0, 1, 0), (1.0, 1.5, 0)])]
        a = ShaftFaultEvidenceResult(
            **with_detail(_assess(channels), "summary"),
            rpm=1500.0,
            parameter_sources={"rpm": "explicit"},
        ).model_dump_json()
        b = ShaftFaultEvidenceResult(
            **with_detail(_assess(channels), "summary"),
            rpm=1500.0,
            parameter_sources={"rpm": "explicit"},
        ).model_dump_json()
        assert a == b

    def test_an_unknown_detail_is_refused(self):
        with pytest.raises(InvalidInputError, match="detail must be"):
            with_detail(_assess([_channel("h", [(1.0, 1, 0)])]), "brief")
