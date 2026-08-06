"""Tests for the server-authored advisory layer.

The advisory layer exists because report content was previously authored by
whoever called the report tool. When that caller is an LLM, faithful numbers
arrive under invented labels: a superseded standard name, a machine-class
vocabulary this codebase does not use, and a "confidence" grade the codebase
deliberately refuses to produce.

These tests are therefore assertions about *who wrote the words*, not only
about whether the numbers are right.
"""

import pytest

from predictive_maintenance_mcp.decision_support.advisory import (
    ABSENT,
    ASSESSED,
    REFUSED,
    build_advisory,
    collect_statements,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import THRESHOLD_PROVENANCE


# ---------------------------------------------------------------------------
# Fixtures — diagnose_vibration-shaped results, built explicitly so each test
# states the condition it exercises rather than hiding it in a factory.
# ---------------------------------------------------------------------------


def _iso_assessed(zone="C", rms=5.0):
    return {
        "status": "assessed",
        "signal_id": "sig",
        "rms_velocity_mm_s": rms,
        "machine_group": 2,
        "support_type": "rigid",
        "zone": zone,
        "zone_description": "Unsatisfactory for long-term operation.",
        "severity_level": "Unsatisfactory",
        "color_code": "orange",
        "boundaries": {"A/B": 1.4, "B/C": 2.8, "C/D": 4.5},
        "frequency_range": "10-1000 Hz",
        "unit_conversion_performed": True,
        "original_unit": "g",
        "operating_speed_rpm": 1500.0,
        "machine_power_kw": None,
        "threshold_provenance": THRESHOLD_PROVENANCE,
    }


def _iso_refused():
    return {
        "status": "refused",
        "signal_id": "sig",
        "reason": "Signal unit not declared — ISO 20816-3 severity requires ...",
        "remedy": "Re-load the signal with load_signal(signal_unit='g') ...",
    }


def _bearing_faults(detected=True):
    return {
        "signal_id": "sig",
        "bearing_id": "6205",
        "rpm": 1500.0,
        "shaft_frequency_hz": 25.0,
        "bearing_frequencies": {
            "BPFO": 81.125,
            "BPFI": 118.875,
            "BSF": 63.91,
            "FTF": 14.84,
            "shaft_freq_hz": 25.0,
        },
        "fault_checks": [
            {
                "signal_id": "sig",
                "bearing_id": "6205",
                "fault_type": "BPFO",
                "fault_type_canonical": "outer_race",
                "expected_frequency_hz": 81.125,
                "detected": detected,
                "detected_frequency_hz": 80.5 if detected else None,
                "magnitude": 0.0516 if detected else None,
                "deviation_pct": 0.77 if detected else None,
                "harmonics_detected": [{"expected_hz": 162.25}] if detected else [],
                "evidence_strength": "high" if detected else "none",
            },
            {
                "signal_id": "sig",
                "bearing_id": "6205",
                "fault_type": "BPFI",
                "fault_type_canonical": "inner_race",
                "expected_frequency_hz": 118.875,
                "detected": False,
                "detected_frequency_hz": None,
                "magnitude": None,
                "deviation_pct": None,
                "harmonics_detected": [],
                "evidence_strength": "none",
            },
        ],
        "overall_assessment": "Outer race fault indicated.",
        "most_likely_fault": "BPFO" if detected else None,
        "most_likely_fault_canonical": "outer_race" if detected else None,
    }


def _anomaly(health="Faulty", ratio=1.0):
    return {
        "overall_health": health,
        "anomaly_ratio": ratio,
        "num_segments": 119,
        "model_name": "bearing_health_model",
    }


def _diagnosis(
    iso=None,
    bearing_faults=None,
    anomaly=None,
    evidence_strength="strong",
    recommendations=None,
):
    return {
        "signal_id": "sig",
        "rpm": 1500.0,
        "bearing_id": "6205" if bearing_faults else None,
        "machine_group": 2,
        "support_type": "rigid",
        "fft_summary": {"peak_frequency_hz": 1850.83, "peak_magnitude": 0.0422},
        "psd_summary": {"total_power": 0.507, "top_peaks": [], "freq_resolution": 1.0},
        "stft_summary": {
            "max_power_freq_hz": 5200.0,
            "max_power_time_s": 1.2,
            "energy_per_band": {
                "0-500 Hz": 12.0,
                "500-2000 Hz": 38.0,
                "2000-5000 Hz": 87.5,
                "5000+ Hz": 446.2,
            },
            "num_time_bins": 100,
        },
        "bearing_faults": bearing_faults,
        "iso_severity": iso if iso is not None else _iso_assessed(),
        "anomaly_detection": anomaly,
        "overall_diagnosis": "ISO Severity: Zone C ...",
        "evidence_strength": evidence_strength,
        "recommendations": recommendations or ["Schedule maintenance."],
    }


# ---------------------------------------------------------------------------
# Verdict, evidence, and the standards caveat
# ---------------------------------------------------------------------------


class TestVerdictAndEvidence:
    def test_detected_fault_produces_verdict_naming_the_fault(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        assert "outer race" in advisory["verdict"]["statement"].lower()
        assert advisory["verdict"]["fault_canonical"] == "outer_race"

    def test_evidence_lists_the_frequency_match_and_its_deviation(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        joined = " ".join(advisory["evidence"]["statements"])
        assert "80.5" in joined
        assert "81.1" in joined or "81.12" in joined
        assert "0.77" in joined or "0.8" in joined

    def test_iso_statement_carries_the_provenance_caveat_verbatim(self):
        advisory = build_advisory(_diagnosis())
        assert advisory["iso"]["standard_note"] == THRESHOLD_PROVENANCE

    def test_iso_statement_names_iso_20816_3_not_the_superseded_edition_alone(self):
        advisory = build_advisory(_diagnosis())
        assert "ISO 20816-3" in advisory["iso"]["statement"]

    def test_clean_signal_yields_no_fault_verdict_and_monitoring_action(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="A", rms=0.8),
                bearing_faults=_bearing_faults(detected=False),
                anomaly=_anomaly(health="Healthy", ratio=0.0),
                evidence_strength="none",
            )
        )
        assert advisory["verdict"]["fault_canonical"] is None
        actions = [r["action"] for r in advisory["recommendations"]]
        assert any("monitor" in a.lower() for a in actions)


# ---------------------------------------------------------------------------
# R2 — no confidence, anywhere
# ---------------------------------------------------------------------------


class TestNoConfidenceLabel:
    def test_strong_evidence_exposes_no_confidence_shaped_field(self):
        """Covers AE6."""
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )

        def walk(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert "confidence" not in key.lower(), (
                        f"confidence-shaped key at {path}.{key}"
                    )
                    assert "probability" not in key.lower(), (
                        f"probability-shaped key at {path}.{key}"
                    )
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(advisory)

    def test_evidence_strength_is_accompanied_by_what_it_counts(self):
        """Covers AE6.

        'strong' on its own invites a reader to hear a probability. The
        explanation is what stops that reading.
        """
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        assert advisory["evidence"]["strength"] == "strong"
        explanation = advisory["evidence"]["strength_explanation"]
        assert "corroborat" in explanation.lower()
        assert "not a" in explanation.lower()

    def test_confidence_appears_only_where_it_is_being_denied(self):
        """The word is allowed to say 'this is not one'. Nothing else."""
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        for statement in collect_statements(advisory):
            lowered = statement.lower()
            if "confidence" not in lowered:
                continue
            assert "not a confidence" in lowered, (
                f"statement asserts a confidence rather than denying one: "
                f"{statement}"
            )


# ---------------------------------------------------------------------------
# R6 — indicator disagreement
# ---------------------------------------------------------------------------


class TestIndicatorDisagreement:
    def test_acceptable_zone_with_full_anomaly_ratio_is_reconciled(self):
        """Covers AE1."""
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="B", rms=1.64),
                bearing_faults=_bearing_faults(),
                anomaly=_anomaly(health="Faulty", ratio=1.0),
            )
        )
        assert advisory["disagreements"], "expected a reconciliation statement"
        entry = advisory["disagreements"][0]
        assert "Zone B" in entry["statement"]
        assert entry["governing_indicator"]
        assert entry["governing_indicator"].lower() != "iso"

    def test_agreeing_indicators_produce_no_disagreement_entry(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="D", rms=12.0),
                bearing_faults=_bearing_faults(),
                anomaly=_anomaly(health="Faulty", ratio=1.0),
            )
        )
        assert advisory["disagreements"] == []

    def test_quiet_zone_with_healthy_anomaly_produces_no_disagreement(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="A", rms=0.9),
                bearing_faults=_bearing_faults(detected=False),
                anomaly=_anomaly(health="Healthy", ratio=0.0),
                evidence_strength="none",
            )
        )
        assert advisory["disagreements"] == []


# ---------------------------------------------------------------------------
# R9 — missing inputs are stated, never silently omitted
# ---------------------------------------------------------------------------


class TestAuthoredAbsences:
    def test_missing_bearing_metadata_states_why_matching_was_not_attempted(self):
        """Covers AE3."""
        advisory = build_advisory(_diagnosis(bearing_faults=None, anomaly=_anomaly()))
        block = advisory["bearing_match"]
        assert block["status"] == ABSENT
        assert block["statement"]
        assert "bearing" in block["statement"].lower()
        assert block["remedy"]

    def test_refused_iso_block_carries_reason_and_remedy_verbatim(self):
        refusal = _iso_refused()
        advisory = build_advisory(_diagnosis(iso=refusal, anomaly=_anomaly()))
        assert advisory["iso"]["status"] == REFUSED
        assert refusal["reason"] in advisory["iso"]["statement"]
        assert advisory["iso"]["remedy"] == refusal["remedy"]

    def test_missing_anomaly_model_states_the_unavailable_conclusion(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=None)
        )
        block = advisory["anomaly"]
        assert block["status"] == ABSENT
        assert block["statement"]

    def test_refused_iso_does_not_produce_a_disagreement_entry(self):
        """A refusal is not an indicator — it cannot disagree with one."""
        advisory = build_advisory(
            _diagnosis(iso=_iso_refused(), anomaly=_anomaly(ratio=1.0))
        )
        assert advisory["disagreements"] == []

    def test_every_block_is_present_even_when_its_input_is_missing(self):
        advisory = build_advisory(
            _diagnosis(iso=_iso_refused(), bearing_faults=None, anomaly=None)
        )
        for key in ("iso", "bearing_match", "anomaly", "baseline_comparison"):
            assert key in advisory
            assert advisory[key]["statement"], f"{key} rendered an empty statement"


# ---------------------------------------------------------------------------
# R8 — baseline comparison
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    def test_faulted_against_healthy_yields_a_directional_rms_delta(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="C", rms=5.0),
                bearing_faults=_bearing_faults(),
                anomaly=_anomaly(ratio=1.0),
            ),
            baseline_diagnosis=_diagnosis(
                iso=_iso_assessed(zone="A", rms=1.0),
                bearing_faults=_bearing_faults(detected=False),
                anomaly=_anomaly(health="Healthy", ratio=0.0),
                evidence_strength="none",
            ),
        )
        block = advisory["baseline_comparison"]
        assert block["status"] == ASSESSED
        rms = next(d for d in block["deltas"] if d["indicator"] == "rms_velocity")
        assert rms["direction"] == "higher"
        assert rms["delta"] == pytest.approx(4.0)
        assert "5.00" in rms["statement"] and "1.00" in rms["statement"]

    def test_anomaly_ratio_delta_is_expressed_in_percentage_points(self):
        advisory = build_advisory(
            _diagnosis(anomaly=_anomaly(ratio=1.0)),
            baseline_diagnosis=_diagnosis(
                anomaly=_anomaly(health="Healthy", ratio=0.0)
            ),
        )
        block = advisory["baseline_comparison"]
        ratio = next(d for d in block["deltas"] if d["indicator"] == "anomaly_ratio")
        assert ratio["delta"] == pytest.approx(100.0)
        assert "percentage point" in ratio["statement"]

    def test_missing_baseline_states_the_unavailable_conclusion(self):
        """Covers AE2."""
        advisory = build_advisory(_diagnosis(anomaly=_anomaly()))
        block = advisory["baseline_comparison"]
        assert block["status"] == ABSENT
        assert block["deltas"] == []
        assert "worsening" in block["statement"].lower()
        assert block["remedy"]

    def test_mismatched_declared_unit_refuses_rather_than_computing(self):
        baseline = _diagnosis(anomaly=_anomaly(health="Healthy", ratio=0.0))
        baseline["iso_severity"]["original_unit"] = "mm/s"
        advisory = build_advisory(
            _diagnosis(anomaly=_anomaly()), baseline_diagnosis=baseline
        )
        block = advisory["baseline_comparison"]
        assert block["status"] == REFUSED
        assert block["deltas"] == []
        assert "mm/s" in block["statement"] and "g" in block["statement"]

    def test_mismatched_bearing_refuses(self):
        baseline = _diagnosis(bearing_faults=_bearing_faults(detected=False))
        baseline["bearing_id"] = "6208"
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults()),
            baseline_diagnosis=baseline,
        )
        assert advisory["baseline_comparison"]["status"] == REFUSED

    def test_identical_baseline_reports_no_change_rather_than_an_empty_block(self):
        diagnosis = _diagnosis(
            bearing_faults=_bearing_faults(), anomaly=_anomaly(ratio=1.0)
        )
        advisory = build_advisory(diagnosis, baseline_diagnosis=diagnosis)
        block = advisory["baseline_comparison"]
        assert block["status"] == ASSESSED
        assert block["deltas"], "an unchanged comparison is still a comparison"
        assert "no measurable change" in block["statement"].lower()

    def test_baseline_deltas_appear_in_the_collected_statements(self):
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="C", rms=5.0), anomaly=_anomaly()),
            baseline_diagnosis=_diagnosis(
                iso=_iso_assessed(zone="A", rms=1.0),
                anomaly=_anomaly(health="Healthy", ratio=0.0),
            ),
        )
        statements = collect_statements(advisory)
        for delta in advisory["baseline_comparison"]["deltas"]:
            assert delta["statement"] in statements

    def test_refused_baseline_never_reports_a_delta_number(self):
        baseline = _diagnosis(anomaly=_anomaly(health="Healthy", ratio=0.0))
        baseline["iso_severity"]["original_unit"] = "mm/s"
        advisory = build_advisory(
            _diagnosis(anomaly=_anomaly()), baseline_diagnosis=baseline
        )
        assert advisory["baseline_comparison"]["deltas"] == []


# ---------------------------------------------------------------------------
# R7 — recommendations carry motivation and evidence
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_each_recommendation_carries_a_motivation_and_its_evidence(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        assert advisory["recommendations"]
        for rec in advisory["recommendations"]:
            assert rec["action"]
            assert rec["urgency"]
            assert rec["motivation"], f"{rec['action']} has no motivation"
            assert rec["evidence"], f"{rec['action']} names no evidence"

    def test_fault_specific_action_appears_when_a_fault_is_detected(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        actions = " ".join(r["action"].lower() for r in advisory["recommendations"])
        assert "bearing" in actions

    def test_refused_iso_still_produces_the_remedy_as_an_action(self):
        advisory = build_advisory(_diagnosis(iso=_iso_refused(), anomaly=_anomaly()))
        actions = " ".join(r["action"] for r in advisory["recommendations"])
        assert "load_signal" in actions or "re-load" in actions.lower()


# ---------------------------------------------------------------------------
# Payload shape — provenance and the flat statement list U7 asserts on
# ---------------------------------------------------------------------------


class TestPayloadShape:
    def test_provenance_names_signal_parameters_and_server_version(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        prov = advisory["provenance"]
        assert prov["signal_id"] == "sig"
        assert prov["rpm"] == 1500.0
        assert prov["bearing_id"] == "6205"
        assert prov["machine_group"] == 2
        assert prov["support_type"] == "rigid"
        assert prov["server_version"]

    def test_collect_statements_returns_every_authored_string(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        statements = collect_statements(advisory)
        assert advisory["verdict"]["statement"] in statements
        assert advisory["iso"]["statement"] in statements
        for rec in advisory["recommendations"]:
            assert rec["action"] in statements
        assert len(statements) == len(set(statements)), "statements must be unique"

    def test_advisory_is_deterministic_for_the_same_input(self):
        """Covers AE5 at the payload level."""
        diagnosis = _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        assert build_advisory(diagnosis) == build_advisory(diagnosis)

    def test_status_constants_are_the_only_status_vocabulary(self):
        advisory = build_advisory(
            _diagnosis(iso=_iso_refused(), bearing_faults=None, anomaly=None)
        )
        for key in ("iso", "bearing_match", "anomaly", "baseline_comparison"):
            assert advisory[key]["status"] in (ASSESSED, REFUSED, ABSENT)

    def test_unknown_diagnosis_shape_raises_rather_than_guessing(self):
        with pytest.raises(ValueError):
            build_advisory({"signal_id": "sig"})
