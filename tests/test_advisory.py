"""Tests for the server-authored advisory layer.

The advisory layer exists because report content was previously authored by
whoever called the report tool. When that caller is an LLM, faithful numbers
arrive under invented labels: a superseded standard name, a machine-class
vocabulary this codebase does not use, and a "confidence" grade the codebase
deliberately refuses to produce.

These tests are therefore assertions about *who wrote the words*, not only
about whether the numbers are right.

Since T1 PR C every bearing sentence, badge and action follows the check's
claim (``diagnostics.bearing_claims``) and the central matcher's per-line
states; the legacy per-check fields (``detected``, ``evidence_strength``,
``detected_frequency_hz``, ``magnitude``, ``deviation_pct``,
``harmonics_detected``) drive nothing. The scenario tests at the end run the
engine on signals whose content is known by construction
(``tests/_bearing_scenarios.py``) and on the bundled MFPT and golden
recordings.
"""

import re

import numpy as np
import pytest

from _bearing_scenarios import (
    DEFECT_SEED,
    FAULT_FREE_SEED,
    FREQUENCIES_6205,
    RPM_6205,
    defect,
    load_mfpt,
    white,
)
from _golden_signals import FS as GOLDEN_FS
from _golden_signals import golden_signals
from test_bearing_specificity import BEARING_MAINTENANCE, HARMONICS_RULE_OUT_NOISE
from predictive_maintenance_mcp.decision_support import advisory as advisory_module
from predictive_maintenance_mcp.decision_support.advisory import (
    ABSENT,
    ASSESSED,
    REFUSED,
    build_advisory,
    collect_statements,
)
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    _bearing_check,
    _synthesize_diagnosis,
    diagnose_vibration,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_frequency_set,
)
from predictive_maintenance_mcp.diagnostics.bearing_claims import (
    bearing_claim,
    claim_statement,
    named_candidate,
)
from predictive_maintenance_mcp.diagnostics.bearing_evidence import (
    Hypothesis,
    evidence_policy,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import THRESHOLD_PROVENANCE
from predictive_maintenance_mcp.figures import build_annotated_envelope_figure
from predictive_maintenance_mcp.signal_processing.spectral import (
    envelope_spectrum_arrays,
    resolve_envelope_band,
)

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


#: The MFPT bearing at 1500 rpm, as the hand-built check summaries declare it.
_FREQUENCIES = {"BPFO": 81.125, "BPFI": 118.875, "BSF": 63.91, "FTF": 14.84}
_CANONICAL = {"BPFO": "outer_race", "BPFI": "inner_race", "BSF": "ball", "FTF": "cage"}
_SHAFT_HZ = 25.0

#: The supported lines of the hand-built summaries: (strongest bin in Hz,
#: level above the floor in dB, envelope amplitude, counted further lines).
_SUPPORTED_LINES = {
    "BPFO": (
        80.5,
        22.4,
        0.0516,
        (("harmonic", 2, 161.0, 14.2), ("harmonic", 3, 241.5, 11.8)),
    ),
    "BPFI": (118.0, 17.1, 0.0302, (("sideband", None, 93.0, 10.3),)),
}

_STATE_REASON = {
    "supported": "corroborated",
    "absent": "no_significant_line",
    "unconfirmed": "no_corroboration",
}


def _fault_check(label: str, state: str, legacy_match: bool) -> dict:
    """One per-check dict with the evidence fields of ``state``.

    The legacy fields say a high-evidence match exactly when ``legacy_match``
    is true, whatever the state: the advisory must not read them.
    """
    line_hz = round(
        Hypothesis(label, _FREQUENCIES[label], _CANONICAL[label]).line_hz, 3
    )
    if state == "supported":
        peak_hz, level_db, amplitude, members = _SUPPORTED_LINES[label]
    else:
        peak_hz, level_db, amplitude, members = line_hz * 0.99, 4.1, 0.0011, ()
    return {
        "signal_id": "sig",
        "bearing_id": "6205",
        "fault_type": label,
        "fault_type_canonical": _CANONICAL[label],
        "expected_frequency_hz": _FREQUENCIES[label],
        "detected": legacy_match,
        "detected_frequency_hz": 80.5 if legacy_match else None,
        "magnitude": 0.0516 if legacy_match else None,
        "deviation_pct": 0.77 if legacy_match else None,
        "harmonics_detected": [{"expected_hz": 162.25}] if legacy_match else [],
        "evidence_strength": "high" if legacy_match else "none",
        "state": state,
        "state_reason": _STATE_REASON[state],
        "line_frequency_hz": line_hz,
        "line_peak_hz": round(peak_hz, 3),
        "line_amplitude": amplitude,
        "level_db": level_db,
        "threshold_db": 13.0,
        "line_test": None,
        "corroboration": [
            {
                "kind": kind,
                "harmonic": harmonic,
                "expected_hz": member_hz,
                "peak_hz": member_hz,
                "level_db": member_db,
                "threshold_db": 9.0,
                "significant": True,
                "counted": True,
                "excluded_by": None,
                "untestable_reason": None,
            }
            for kind, harmonic, member_hz, member_db in members
        ],
        "shared_with": [],
    }


def _bearing_faults(
    supported=("BPFO",),
    *,
    unconfirmed=(),
    legacy_match=None,
    eligibility="unknown",
):
    """A check summary as ``check_all_bearing_faults`` returns it.

    ``supported`` and ``unconfirmed`` set the per-line states (the rest are
    absent); the outcome, candidates, claim and the fields that follow the
    claim are computed by the evidence and claim rules. ``legacy_match`` names
    the labels whose legacy fields claim a high-evidence match (default: the
    supported ones).
    """
    legacy = set(supported if legacy_match is None else legacy_match)
    states = {
        label: (
            "supported"
            if label in supported
            else "unconfirmed" if label in unconfirmed else "absent"
        )
        for label in _FREQUENCIES
    }
    checks = [_fault_check(k, s, k in legacy) for k, s in states.items()]
    hypotheses = [Hypothesis(k, v, _CANONICAL[k]) for k, v in _FREQUENCIES.items()]
    candidates = sorted(
        (
            {
                "fault_type": c["fault_type"],
                "fault_type_canonical": c["fault_type_canonical"],
                "line_frequency_hz": c["line_frequency_hz"],
                "line_peak_hz": c["line_peak_hz"],
                "level_db": c["level_db"],
            }
            for c in checks
            if c["state"] == "supported"
        ),
        key=lambda c: -c["level_db"],
    )
    outcome = (
        "indicated"
        if len(candidates) == 1
        else (
            "several_types"
            if candidates
            else "inconclusive" if unconfirmed else "no_indication"
        )
    )
    evidence = {"checks": checks, "outcome": outcome, "candidates": candidates}
    claim = bearing_claim(
        evidence,
        hypotheses,
        shaft_hz=_SHAFT_HZ,
        tolerance_pct=5.0,
        acquisition_eligibility=eligibility,
    )
    named = named_candidate(claim, evidence)
    return {
        "signal_id": "sig",
        "bearing_id": "6205",
        "rpm": 1500.0,
        "shaft_frequency_hz": _SHAFT_HZ,
        "bearing_frequencies": {**_FREQUENCIES, "shaft_freq_hz": _SHAFT_HZ},
        "fault_checks": checks,
        "overall_assessment": claim_statement(claim, evidence, hypotheses),
        "most_likely_fault": named["fault_type"] if named else None,
        "most_likely_fault_canonical": (
            named["fault_type_canonical"] if named else None
        ),
        "source": "test catalog entry",
        "outcome": outcome,
        "outcome_reason": "The outcome reason of the evidence layer.",
        "next_step": None,
        "candidates": candidates,
        "evidence_policy": evidence_policy(5.0, 3),
        "claim": claim,
    }


def _anomaly(health="Faulty", ratio=1.0):
    return {
        "overall_health": health,
        "anomaly_ratio": ratio,
        "num_segments": 119,
        "model_name": "bearing_health_model",
    }


def _not_in_catalog_check():
    """A ``bearing_check`` the pipeline returns for a designation the
    catalog lacks (issue #84). The advisory carries reason and remedy
    verbatim, so the strings here only need to be recognisable."""
    return {
        "status": "not_computed",
        "reason_code": "not_in_catalog",
        "reason": "Bearing '9999' is not in the bearing catalog, so ...",
        "remedy": "Call search_bearing_catalog(bearing_id='9999') ...",
    }


def _diagnosis(
    iso=None,
    bearing_faults=None,
    anomaly=None,
    evidence_strength="strong",
    recommendations=None,
    bearing_check=None,
):
    diagnosis = {
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
            # Shape mirrors compute_stft_spectrogram exactly: a list of
            # {"band", "energy"} entries, not a mapping. A fixture that
            # invents a friendlier shape hides a real integration failure.
            "energy_per_band": [
                {"band": "0-100 Hz", "energy": 4.0},
                {"band": "100-500 Hz", "energy": 8.0},
                {"band": "500-2000 Hz", "energy": 38.0},
                {"band": "2000-5000 Hz", "energy": 87.5},
                {"band": "5000+ Hz", "energy": 446.2},
            ],
            "num_time_bins": 100,
        },
        "bearing_faults": bearing_faults,
        "iso_severity": iso if iso is not None else _iso_assessed(),
        "anomaly_detection": anomaly,
        "overall_diagnosis": "ISO Severity: Zone C ...",
        "evidence_strength": evidence_strength,
        "recommendations": recommendations or ["Schedule maintenance."],
    }
    # Absent unless given: build_advisory reads the block with .get(), so a
    # hand-built dict without it keeps its old reading.
    if bearing_check is not None:
        diagnosis["bearing_check"] = bearing_check
        diagnosis["bearing_id"] = "9999"
    return diagnosis


# ---------------------------------------------------------------------------
# Verdict, evidence, and the standards caveat
# ---------------------------------------------------------------------------


class TestVerdictAndEvidence:
    def test_single_spectrum_evidence_verdict_is_the_claim_with_its_step(self):
        faults = _bearing_faults()
        advisory = build_advisory(_diagnosis(bearing_faults=faults, anomaly=_anomaly()))

        assert faults["claim"]["level"] == "single_spectrum_evidence"
        assert advisory["verdict"]["statement"] == (
            f"{faults['overall_assessment']} Bearing 6205: confirm the "
            "single-spectrum indication with a comparable measurement or an "
            "inspection of the outer race."
        )
        assert "consistent with an outer-race defect" in faults["overall_assessment"]
        assert advisory["verdict"]["fault_canonical"] == "outer_race"
        assert advisory["verdict"]["fault_acronym"] == "BPFO"

    def test_several_types_verdict_names_every_candidate_alike(self):
        faults = _bearing_faults(supported=("BPFO", "BPFI"))
        advisory = build_advisory(_diagnosis(bearing_faults=faults))
        verdict = advisory["verdict"]["statement"]

        assert faults["claim"]["conditions"] == ["several_types"]
        assert verdict.startswith(faults["overall_assessment"])
        assert (
            "inspect the outer race and the inner race with equal priority" in verdict
        )
        assert advisory["verdict"]["fault_canonical"] is None
        assert "most likely" not in verdict.lower()

    def test_withheld_verdict_in_an_elevated_zone_keeps_both_sources_open(self):
        faults = _bearing_faults(supported=())
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="C"), bearing_faults=faults)
        )

        assert advisory["verdict"]["statement"] == (
            f"{faults['overall_assessment']} Broadband vibration is elevated "
            "(Zone C), and the envelope evidence names no bearing fault type, so "
            "bearing and non-bearing sources both remain possible."
        )
        assert "free of defects" in faults["overall_assessment"]
        assert advisory["verdict"]["fault_canonical"] is None

    def test_evidence_lists_the_counted_lines_with_their_levels(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        lines = [
            s
            for s in advisory["evidence"]["statements"]
            if s.startswith("Lines counted for")
        ]

        assert lines == [
            "Lines counted for the outer-race line (BPFO): the line at 80.50 Hz "
            "(22.4 dB above the local noise floor), with harmonic 2 at 161.00 Hz "
            "(14.2 dB) and harmonic 3 at 241.50 Hz (11.8 dB). Each stands above "
            "the local noise floor of this spectrum in its line test: the tested "
            "lines share a nominal false-alarm level of 1 % per check, and each "
            "further line is tested at 1 %. Lines of one spectrum are not "
            "independent evidence and do not confirm a fault."
        ]
        joined = " ".join(advisory["evidence"]["statements"])
        assert HARMONICS_RULE_OUT_NOISE not in joined
        assert "deviation" not in joined

    def test_sideband_only_inner_race_carries_its_limitation(self):
        faults = _bearing_faults(supported=("BPFI",))
        advisory = build_advisory(_diagnosis(bearing_faults=faults))
        (line,) = [
            s
            for s in advisory["evidence"]["statements"]
            if s.startswith("Lines counted for")
        ]

        assert faults["claim"]["limitations"] == ["sideband_only_corroboration"]
        assert "a shaft-rate sideband at 93.00 Hz (10.3 dB)" in line
        assert line.endswith(
            "Its only corroborating lines are shaft-rate sidebands, which shaft-rate "
            "modulation of the background can also produce, so they do not "
            "independently confirm an inner-race defect."
        )

    def test_rows_read_the_line_test_and_carry_closed_list_badges(self):
        faults = _bearing_faults(unconfirmed=("BPFI",))
        advisory = build_advisory(_diagnosis(bearing_faults=faults))
        rows = {r["fault_type"]: r for r in advisory["bearing_match"]["rows"]}

        assert rows["BPFO"]["state"] == "supported"
        assert rows["BPFO"]["badge"] == "above the floor, corroborated"
        assert rows["BPFO"]["emphasised"] is True
        assert rows["BPFI"]["badge"] == "above the floor, not counted"
        assert rows["BSF"]["badge"] == "not above the floor"
        # The rolling element is tested at twice the declared BSF.
        assert rows["BSF"]["line_hz"] == pytest.approx(2 * 63.91, abs=1e-3)
        assert (rows["BPFO"]["peak_hz"], rows["BPFO"]["level_db"]) == (80.5, 22.4)
        for legacy in ("matched", "measured_hz", "deviation_pct", "harmonics"):
            assert legacy not in rows["BPFO"]

    def test_bearing_section_states_each_line_test(self):
        faults = _bearing_faults(unconfirmed=("BPFI",))
        statement = build_advisory(_diagnosis(bearing_faults=faults))["bearing_match"][
            "statement"
        ]

        assert statement.startswith(
            "Envelope line test for bearing 6205 at 25.0 Hz shaft speed. The "
            "outer-race line (BPFO) at 80.50 Hz stands 22.4 dB above the local "
            "noise floor, and further lines of its kinematics also stand above the "
            "floor in the same spectrum."
        )
        assert "but is not counted as support for a fault type" in statement
        assert (
            "The rolling-element line (2 x BSF) does not stand above the local "
            "noise floor in this record and band."
        ) in statement

    def test_legacy_per_check_fields_drive_no_statement_badge_or_action(self):
        """Two summaries that differ only in the legacy fields (one claims a
        high-evidence BPFO match the line test did not find) give the same
        advisory."""
        plain = build_advisory(_diagnosis(bearing_faults=_bearing_faults(supported=())))
        legacy = build_advisory(
            _diagnosis(
                bearing_faults=_bearing_faults(supported=(), legacy_match=("BPFO",))
            )
        )

        assert legacy == plain
        assert legacy["verdict"]["fault_canonical"] is None

    def test_generated_signal_without_noise_withholds_every_bearing_statement(self):
        faults = _bearing_faults(eligibility="known_inapplicable")
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="B"), bearing_faults=faults)
        )
        generated = (
            "The signal's companion metadata declares a generated signal with an "
            "additive noise level of 0; the line test assumes a recording with "
            "measurement noise, so no bearing fault type is assessed."
        )

        assert advisory["verdict"]["statement"] == generated
        assert advisory["verdict"]["fault_canonical"] is None
        assert advisory["bearing_match"]["statement"] == (
            f"Envelope line test for bearing 6205 at 25.0 Hz shaft speed. {generated}"
        )
        assert {r["badge"] for r in advisory["bearing_match"]["rows"]} == {
            "not assessed"
        }
        assert not any(r["emphasised"] for r in advisory["bearing_match"]["rows"])
        assert not any(
            s.startswith("Lines counted for")
            for s in advisory["evidence"]["statements"]
        )
        assert advisory["disagreements"] == []
        assert not [
            r for r in advisory["recommendations"] if r["action"].startswith("Bearing")
        ]

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
                bearing_faults=_bearing_faults(supported=()),
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
                    assert (
                        "confidence" not in key.lower()
                    ), f"confidence-shaped key at {path}.{key}"
                    assert (
                        "probability" not in key.lower()
                    ), f"probability-shaped key at {path}.{key}"
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
        assert "counts the findings" in explanation.lower()
        assert "a bearing finding once" in explanation.lower()
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
        """Covers AE1 (advisory layer). Since T1 PR C each dissenting
        indicator has its own entry: the single-spectrum evidence first, then
        the anomaly model."""
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="B", rms=1.64),
                bearing_faults=_bearing_faults(),
                anomaly=_anomaly(health="Faulty", ratio=1.0),
            )
        )
        assert len(advisory["disagreements"]) == 2, "expected two reconciliations"
        for entry in advisory["disagreements"]:
            assert "Zone B" in entry["statement"]
            assert entry["governing_indicator"]
            assert entry["governing_indicator"].lower() != "iso"

    def test_single_spectrum_evidence_in_an_acceptable_zone_asks_for_confirmation(
        self,
    ):
        """KTD20: the envelope evidence warrants confirmation despite the
        acceptable zone; it is never 'actionable' or a 'developing fault'."""
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="A", rms=0.9), bearing_faults=_bearing_faults()
            )
        )
        (entry,) = advisory["disagreements"]

        assert entry["statement"] == (
            "Indicators differ. Zone A describes overall vibration energy as "
            "acceptable, while the envelope evidence of this single spectrum is "
            "consistent with an outer-race defect. Both can hold: a localised "
            "defect can stand out in the envelope spectrum while adding little "
            "broadband energy. The envelope evidence warrants confirmation despite "
            "the acceptable zone; it is not a confirmed fault."
        )
        actions = [r["action"] for r in advisory["recommendations"]]
        assert not any("actionable" in a.lower() for a in actions)
        assert "developing fault" not in entry["statement"]

    @pytest.mark.parametrize(
        "faults",
        [
            _bearing_faults(supported=("BPFO", "BPFI")),
            _bearing_faults(unconfirmed=("BPFI",), supported=()),
            _bearing_faults(eligibility="known_inapplicable"),
        ],
        ids=["several_types", "inconclusive", "generated_without_noise"],
    )
    def test_only_single_spectrum_evidence_drives_a_bearing_disagreement(self, faults):
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="B"), bearing_faults=faults)
        )

        assert advisory["disagreements"] == []

    def test_an_anomaly_flag_names_and_upgrades_no_bearing_type(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="B"),
                bearing_faults=_bearing_faults(supported=()),
                anomaly=_anomaly(health="Suspicious", ratio=0.2),
            )
        )
        (entry,) = advisory["disagreements"]

        assert entry["statement"] == (
            "Indicators differ. Zone B describes overall vibration energy as "
            "acceptable, while the anomaly model (Suspicious, 20% of segments) "
            "places this signal outside the pattern it learned. The anomaly model "
            "does not identify a fault type or a bearing element."
        )
        assert advisory["verdict"]["fault_canonical"] is None
        assert not re.search(
            r"outer|inner|rolling|cage|BPF|BSF|FTF", entry["statement"]
        )
        assert not [
            r for r in advisory["recommendations"] if r["action"].startswith("Bearing")
        ]

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
                bearing_faults=_bearing_faults(supported=()),
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
        # The remedy names the argument the report tool takes; the companion
        # metadata has no bearing field.
        assert "generate_diagnostic_report with bearing_id" in block["remedy"]
        assert "companion" not in block["remedy"]

    def test_unknown_bearing_says_matching_was_not_computed_not_unsupplied(self):
        """Issue #84: the report said no designation was supplied, and told
        the caller to pass a bearing_id it had passed."""
        check = _not_in_catalog_check()
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="B"), bearing_check=check)
        )
        block = advisory["bearing_match"]
        assert block["status"] == REFUSED
        assert block["rows"] == []
        assert block["statement"].startswith(
            "Bearing characteristic-frequency matching was not computed. "
        )
        assert check["reason"] in block["statement"]
        assert "says nothing about the condition of the bearing" in block["statement"]
        assert block["remedy"] == check["remedy"]
        assert advisory["verdict"]["statement"] == (
            "No bearing fault verdict was reached, because characteristic-"
            "frequency matching was not computed for bearing '9999'."
        )
        statements = collect_statements(advisory)
        assert not any("no bearing designation" in s.lower() for s in statements)
        remedies = [
            r for r in advisory["recommendations"] if r["action"] == check["remedy"]
        ]
        assert len(remedies) == 1
        assert remedies[0]["motivation"] == check["reason"]
        assert remedies[0]["evidence"] == [block["statement"]]

    def test_not_computed_bearing_keeps_the_zone_action_and_is_not_excluded(self):
        """The zone C warning stands on its own evidence; the headline must
        not send the reader to a non-bearing source when the bearing was
        never checked."""
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="C"), bearing_check=_not_in_catalog_check()
            )
        )
        verdict = advisory["verdict"]["statement"]
        assert "(Zone C)" in verdict
        assert "cannot be excluded as the source" in verdict
        assert "non-bearing source" not in verdict
        zone_actions = [
            r
            for r in advisory["recommendations"]
            if r["evidence"] == [advisory["iso"]["statement"]]
        ]
        assert zone_actions, "the ISO zone C action must still be recommended"

    def test_not_computed_bearing_with_an_anomaly_flag_is_not_excluded(self):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="A"),
                bearing_check=_not_in_catalog_check(),
                anomaly=_anomaly(health="Suspicious", ratio=0.2),
            )
        )
        verdict = advisory["verdict"]["statement"]
        assert "anomaly model" in verdict
        assert "cannot be excluded as the source" in verdict
        assert "outside the bearing's characteristic frequencies" not in verdict

    def test_not_computed_without_a_designation_names_no_bearing(self):
        """The fault-orders shape (a check requested by the point's fault
        orders, no bearing_id): not computed, and no bearing named "None"."""
        diagnosis = _diagnosis(bearing_check=_not_in_catalog_check())
        diagnosis["bearing_id"] = None
        verdict = build_advisory(diagnosis)["verdict"]["statement"]
        assert verdict.startswith(
            "No bearing fault verdict was reached, because characteristic-"
            "frequency matching was not computed."
        )
        assert "None" not in verdict

    def test_result_without_a_bearing_check_block_reads_as_before(self):
        """A diagnosis dict without bearing_check: the absence as before."""
        advisory = build_advisory(_diagnosis(bearing_faults=None, anomaly=_anomaly()))
        assert advisory["bearing_match"]["status"] == ABSENT
        assert "no bearing designation was supplied" in (
            advisory["bearing_match"]["statement"]
        )

    def test_not_requested_check_is_the_absent_block(self):
        advisory = build_advisory(
            _diagnosis(
                anomaly=_anomaly(),
                bearing_check={
                    "status": "not_requested",
                    "reason_code": None,
                    "reason": None,
                    "remedy": None,
                },
            )
        )
        assert advisory["bearing_match"]["status"] == ABSENT

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
                bearing_faults=_bearing_faults(supported=()),
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
        baseline = _diagnosis(bearing_faults=_bearing_faults(supported=()))
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
        envelope = next(
            d for d in block["deltas"] if d["indicator"] == "envelope_magnitude"
        )
        assert envelope["statement"] == (
            "Envelope amplitude of the outer-race line (BPFO) is unchanged against "
            "baseline."
        )

    def test_envelope_delta_reads_the_tested_line_of_the_candidate(self):
        """The delta compares line_amplitude at the line the check tested,
        with the strongest bin of the window in both records, and never calls
        it the defect signature."""
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults()),
            baseline_diagnosis=_diagnosis(bearing_faults=_bearing_faults(supported=())),
        )
        (delta,) = [
            d
            for d in advisory["baseline_comparison"]["deltas"]
            if d["indicator"] == "envelope_magnitude"
        ]

        assert delta["fault_type"] == "BPFO"
        assert (delta["value"], delta["baseline_value"]) == (0.0516, 0.0011)
        assert delta["statement"] == (
            "Envelope amplitude of the outer-race line (BPFO) is 33.4 dB higher "
            "than baseline (strongest bin of its search window: 80.50 Hz, "
            "baseline 80.31 Hz)."
        )
        assert "signature" not in advisory["baseline_comparison"]["statement"]

    def test_several_types_give_one_envelope_delta_per_candidate_alike(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(supported=("BPFO", "BPFI"))),
            baseline_diagnosis=_diagnosis(bearing_faults=_bearing_faults(supported=())),
        )
        block = advisory["baseline_comparison"]
        deltas = [d for d in block["deltas"] if d["indicator"] == "envelope_magnitude"]

        assert [d["fault_type"] for d in deltas] == ["BPFO", "BPFI"]
        for delta in deltas:
            assert delta["direction"] == "higher"
            assert delta["statement"] in block["statement"]

    def test_no_envelope_delta_without_a_candidate(self):
        advisory = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(supported=())),
            baseline_diagnosis=_diagnosis(bearing_faults=_bearing_faults()),
        )
        assert not [
            d
            for d in advisory["baseline_comparison"]["deltas"]
            if d["indicator"] == "envelope_magnitude"
        ]

    def test_headline_delta_is_chosen_by_meaning_not_by_magnitude(self):
        """The three deltas are in dB, mm/s and percentage points.

        Ranking them by absolute value would rank them by unit scale: a
        1-point move in anomaly ratio would outrank a 0.5 mm/s rise in RMS
        velocity for no reason other than the axis it sits on.

        Neither record has a bearing candidate, so no envelope delta (which
        would lead by design) takes part.
        """
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_assessed(zone="C", rms=5.5),
                bearing_faults=_bearing_faults(supported=()),
                anomaly=_anomaly(ratio=1.0),
            ),
            baseline_diagnosis=_diagnosis(
                iso=_iso_assessed(zone="A", rms=5.0),
                bearing_faults=_bearing_faults(supported=()),
                anomaly=_anomaly(health="Healthy", ratio=0.0),
            ),
        )
        block = advisory["baseline_comparison"]
        anomaly_delta = next(
            d for d in block["deltas"] if d["indicator"] == "anomaly_ratio"
        )
        assert abs(anomaly_delta["delta"]) == 100.0
        # The anomaly ratio has by far the largest number and is still not
        # the headline: RMS velocity is the more specific movement here.
        assert "RMS velocity" in block["statement"]

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

    def test_single_spectrum_evidence_adds_a_confirmation_step(self):
        faults = _bearing_faults()
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone="D"), bearing_faults=faults)
        )
        actions = [r["action"] for r in advisory["recommendations"]]

        assert actions == [
            "Immediate shutdown recommended",
            "Bearing 6205: confirm the single-spectrum indication with a comparable "
            "measurement or an inspection of the outer race.",
            "Train an anomaly model on healthy signals from this machine, then "
            "re-run the diagnosis.",
        ]
        step = advisory["recommendations"][1]
        assert step["description"] == (
            "Single-spectrum envelope evidence is not a confirmed fault; this step "
            "confirms or excludes it."
        )
        assert step["motivation"] == faults["overall_assessment"]
        assert step["evidence"] == [advisory["bearing_match"]["statement"]]

    @pytest.mark.parametrize(
        "faults",
        [
            _bearing_faults(),
            _bearing_faults(supported=("BPFO", "BPFI")),
            _bearing_faults(unconfirmed=("BPFI",), supported=()),
            _bearing_faults(supported=()),
            _bearing_faults(eligibility="known_inapplicable"),
        ],
        ids=[
            "single_spectrum_evidence",
            "several_types",
            "inconclusive",
            "no_indication",
            "generated_without_noise",
        ],
    )
    @pytest.mark.parametrize("zone", ["A", "B", "C", "D"])
    def test_no_claim_level_reaches_the_fault_type_table(
        self, faults, zone, monkeypatch
    ):
        """No inferred bearing type is passed to the recommendation table, in
        any zone, so no report action asks for replacement."""
        calls = []
        table = advisory_module.generate_recommendations

        def recording(severity_zone, fault_types=None):
            calls.append(fault_types)
            return table(severity_zone, fault_types)

        monkeypatch.setattr(advisory_module, "generate_recommendations", recording)
        advisory = build_advisory(
            _diagnosis(iso=_iso_assessed(zone=zone), bearing_faults=faults)
        )

        assert calls and not any(calls), calls
        assert not [
            s for s in collect_statements(advisory) if re.search(r"replac", s, re.I)
        ]

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

    def test_collect_statements_includes_every_badge_and_the_figure_caption(self):
        """KTD18: the caption and the badges are authored text too."""
        faults = _bearing_faults(unconfirmed=("BPFI",))
        advisory = build_advisory(_diagnosis(bearing_faults=faults))
        freqs = np.linspace(0.0, 500.0, 2000)
        figure = build_annotated_envelope_figure(
            freqs, 0.001 + 0.05 * np.exp(-(((freqs - 80.5) / 0.8) ** 2)), faults
        )

        statements = collect_statements(advisory, figure)

        for row in advisory["bearing_match"]["rows"]:
            assert row["badge"] in statements
        assert figure["caption"] in statements
        assert statements[: len(collect_statements(advisory))] == collect_statements(
            advisory
        )

    def test_advisory_is_deterministic_for_the_same_input(self):
        """Covers AE5 at the payload level."""
        diagnosis = _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        assert build_advisory(diagnosis) == build_advisory(diagnosis)

    @pytest.mark.parametrize("bearing_check", [None, _not_in_catalog_check()])
    def test_status_constants_are_the_only_status_vocabulary(self, bearing_check):
        advisory = build_advisory(
            _diagnosis(
                iso=_iso_refused(),
                bearing_faults=None,
                anomaly=None,
                bearing_check=bearing_check,
            )
        )
        for key in ("iso", "bearing_match", "anomaly", "baseline_comparison"):
            assert advisory[key]["status"] in (ASSESSED, REFUSED, ABSENT)

    def test_unknown_diagnosis_shape_raises_rather_than_guessing(self):
        with pytest.raises(ValueError):
            build_advisory({"signal_id": "sig"})


# ===========================================================================
# T1 PR C scenarios: the engine's claim on signals known by construction
# ===========================================================================

#: Sampling rate of the synthetic 6205 scenarios (the CWRU drive-end rate).
FS = 12000.0
SHAFT_HZ = RPM_6205 / 60.0
GOLDEN_RPM = 1800.0

#: No anomaly model: these scenarios isolate the bearing evidence.
NO_MODEL = "nonexistent_model_advisory_scenarios"

#: Fault-free records of the AE1 sweep (the seeds of the pipeline noise test).
NOISE_RECORDS = 10

#: What no statement of AE13's scenarios may contain. "not a confirmed
#: fault" and "is not a probability" are the allowed negations.
AE13_FORBIDDEN = re.compile(
    r"\bconfirmed\b|damage detected|most likely|probability|\breplac\w*"
    r"|treat this as actionable|developing fault|single noise peak"
    r"|defect signature itself",
    re.IGNORECASE,
)
AE13_ALLOWED = ("not a confirmed fault", "is not a probability")

#: What no bearing statement (verdict, bearing section, badges, caption,
#: counted lines, disagreements, bearing actions) may contain.
BEARING_FORBIDDEN = re.compile(
    r"\bdetected\b|healthy|\bindependent(?:ly)?\b|—", re.IGNORECASE
)
BEARING_ALLOWED = ("not independent evidence", "do not independently confirm")


def _offending(sentence: str, pattern: re.Pattern, allowed: tuple) -> list[str]:
    for phrase in allowed:
        sentence = sentence.replace(phrase, "")
    return pattern.findall(sentence)


def _scenario(signal, fs, rpm, *, bearing_id="6205") -> dict:
    """A diagnose_vibration result plus the signal it was computed from."""
    diagnosis = diagnose_vibration(
        signal,
        fs,
        rpm,
        signal_id="scenario",
        bearing_id=bearing_id,
        signal_unit="g",
        anomaly_model_name=NO_MODEL,
    )
    return {"diagnosis": diagnosis, "signal": signal, "fs": fs}


def _declared_scenario(signal, fs, frequencies, rpm) -> dict:
    """The same with the engine's check of declared frequencies, given to the
    synthesis the pipeline runs (diagnose_vibration checks catalog bearings)."""
    check = check_frequency_set(signal, fs, frequencies, rpm)
    blocks = diagnose_vibration(
        signal, fs, rpm, signal_unit="g", anomaly_model_name=NO_MODEL
    )
    bearing_check = _bearing_check("assessed")
    text, strength, recommendations = _synthesize_diagnosis(
        fft_summary=blocks["fft_summary"],
        psd_summary=blocks["psd_summary"],
        stft_summary=blocks["stft_summary"],
        bearing_faults=check,
        bearing_check=bearing_check,
        iso_severity=blocks["iso_severity"],
        anomaly=blocks["anomaly_detection"],
        rpm=rpm,
    )
    diagnosis = {
        **blocks,
        "bearing_faults": check,
        "bearing_check": bearing_check,
        "overall_diagnosis": text,
        "evidence_strength": strength,
        "recommendations": recommendations,
    }
    return {"diagnosis": diagnosis, "signal": signal, "fs": fs}


def _figure(scenario: dict) -> dict:
    """The report's figure: the spectrum the check used, with its windows."""
    fs = scenario["fs"]
    freqs, mags = envelope_spectrum_arrays(
        scenario["signal"], fs, resolve_envelope_band(fs, None)
    )
    return build_annotated_envelope_figure(
        freqs, mags, scenario["diagnosis"]["bearing_faults"]
    )


def _with_zone(diagnosis: dict, zone: str) -> dict:
    return {**diagnosis, "iso_severity": _iso_assessed(zone=zone)}


def _bearing_statements(advisory: dict, figure: dict) -> list[str]:
    statements = [
        advisory["verdict"]["statement"],
        advisory["bearing_match"]["statement"],
        figure["caption"],
        *(row["badge"] for row in advisory["bearing_match"]["rows"]),
        *(e["statement"] for e in advisory["disagreements"]),
        *(
            s
            for s in advisory["evidence"]["statements"]
            if s.startswith("Lines counted for")
        ),
        *(
            d["statement"]
            for d in advisory["baseline_comparison"]["deltas"]
            if d["indicator"] == "envelope_magnitude"
        ),
    ]
    for rec in advisory["recommendations"]:
        if rec["action"].startswith("Bearing"):
            statements += [rec["action"], rec["description"], rec["motivation"]]
    return statements


@pytest.fixture(scope="module")
def noise_scenarios() -> list[dict]:
    """AE1: white noise, 6205 at 1797 rpm, 1 s at 12 kHz."""
    return [
        _scenario(
            0.1
            * white(
                np.random.default_rng(FAULT_FREE_SEED + 900 + i), FS, 1.0, SHAFT_HZ
            ),
            FS,
            RPM_6205,
        )
        for i in range(NOISE_RECORDS)
    ]


def _combined(seed: int = 0) -> np.ndarray:
    return defect(
        "combined",
        np.random.default_rng((DEFECT_SEED, 42, seed)),
        FS,
        10.0,
        FREQUENCIES_6205,
        SHAFT_HZ,
        -10.0,
    )


@pytest.fixture(scope="module")
def combined_scenario() -> dict:
    """AE4: combined outer- and inner-race defects at -10 dB, 10 s."""
    return _scenario(_combined(), FS, RPM_6205)


@pytest.fixture(scope="module")
def combined_baseline() -> dict:
    """A 10 s white-noise reference for the same bearing and speed."""
    rng = np.random.default_rng(FAULT_FREE_SEED + 950)
    return _scenario(white(rng, FS, 10.0, SHAFT_HZ), FS, RPM_6205)


@pytest.fixture(scope="module")
def mfpt_scenario() -> dict:
    """AE5: the MFPT baseline_3 with its declared frequencies at 1500 rpm,
    whose 120 Hz mains line in the BPFI window has no corroborating line."""
    signal, fs, frequencies, rpm = load_mfpt("real_test/baseline_3.csv")
    return _declared_scenario(signal, fs, frequencies, rpm)


@pytest.fixture(scope="module")
def golden_scenario() -> dict:
    """AE11: golden_bearing, a true BPFO modulation, 1 s at 10 kHz."""
    return _scenario(golden_signals()["golden_bearing"], GOLDEN_FS, GOLDEN_RPM)


class TestScenarioStatements:
    def test_seeded_noise_names_no_fault_and_asks_for_no_bearing_action(
        self, noise_scenarios
    ):
        """AE1: no fault named, no disagreement, no fault-specific action, no
        claim that harmonics rule out noise."""
        wrong = []
        for scenario in noise_scenarios:
            advisory = build_advisory(_with_zone(scenario["diagnosis"], "B"))
            found = {
                "verdict names": advisory["verdict"]["fault_canonical"],
                "disagreements": advisory["disagreements"],
                "maintenance actions": [
                    r["action"]
                    for r in advisory["recommendations"]
                    if BEARING_MAINTENANCE.search(r["action"])
                ],
                "harmonics rule out noise": [
                    s
                    for s in collect_statements(advisory)
                    if HARMONICS_RULE_OUT_NOISE in s
                ],
            }
            if any(found.values()):
                wrong.append({k: v for k, v in found.items() if v})

        assert not wrong, wrong

    def test_combined_defect_lists_candidates_alike_with_a_delta_each(
        self, combined_scenario, combined_baseline
    ):
        """AE4: candidates named as candidates with their levels, both windows
        emphasised alike, an envelope delta per candidate against a baseline,
        no replacement and no 'most likely'."""
        diagnosis = combined_scenario["diagnosis"]
        faults = diagnosis["bearing_faults"]
        advisory = build_advisory(diagnosis, combined_baseline["diagnosis"])
        figure = _figure(combined_scenario)
        candidates = [c["fault_type"] for c in faults["candidates"]]

        assert faults["outcome"] == "several_types"
        assert sorted(candidates) == ["BPFI", "BPFO"]
        verdict = advisory["verdict"]["statement"]
        for candidate in faults["candidates"]:
            assert f"({candidate['level_db']:.1f} dB)" in verdict
        assert advisory["verdict"]["fault_canonical"] is None
        assert sorted(b["label"] for b in figure["bands"] if b["emphasised"]) == [
            "BPFI",
            "BPFO",
        ]
        deltas = [
            d
            for d in advisory["baseline_comparison"]["deltas"]
            if d["indicator"] == "envelope_magnitude"
        ]
        assert [d["fault_type"] for d in deltas] == candidates
        statements = collect_statements(advisory, figure)
        assert not [s for s in statements if re.search(r"replac|most likely", s, re.I)]

    def test_unconfirmed_line_gives_an_inconclusive_verdict_with_its_step(
        self, mfpt_scenario
    ):
        """AE5: the inconclusive sentence and the repeat or reference step."""
        faults = mfpt_scenario["diagnosis"]["bearing_faults"]
        advisory = build_advisory(mfpt_scenario["diagnosis"])

        assert faults["outcome"] == "inconclusive"
        assert advisory["verdict"]["statement"] == (
            "The significance-tested evidence is inconclusive and does not support "
            "a unique fault type. Bearing check: repeat the measurement, or compare "
            "it with a comparable earlier measurement."
        )
        assert advisory["verdict"]["fault_canonical"] is None
        assert advisory["disagreements"] == []
        rows = {r["fault_type"]: r["badge"] for r in advisory["bearing_match"]["rows"]}
        assert rows["BPFI"] == "above the floor, not counted"

    @pytest.mark.parametrize("kind", ["outer", "combined"])
    def test_dict_order_changes_neither_verdict_nor_emphasis(self, kind):
        """AE6: verdict, figure emphasis and fault_canonical agree with the
        outcome in both orders of the frequency dict."""
        if kind == "combined":
            signal = _combined()
        else:
            signal = defect(
                "outer",
                np.random.default_rng((DEFECT_SEED, 40, 0)),
                FS,
                10.0,
                FREQUENCIES_6205,
                SHAFT_HZ,
                -15.0,
            )
        forward = dict(FREQUENCIES_6205)
        backward = dict(reversed(list(FREQUENCIES_6205.items())))
        results = []
        for frequencies in (forward, backward):
            scenario = _declared_scenario(signal, FS, frequencies, RPM_6205)
            advisory = build_advisory(scenario["diagnosis"])
            figure = _figure(scenario)
            results.append(
                (
                    advisory["verdict"]["statement"],
                    advisory["verdict"]["fault_canonical"],
                    [b["label"] for b in figure["bands"] if b["emphasised"]],
                    figure["caption"],
                )
            )

        assert results[0] == results[1]
        verdict, canonical, emphasised, _ = results[0]
        if kind == "outer":
            assert (canonical, emphasised) == ("outer_race", ["BPFO"])
        else:
            assert canonical is None and sorted(emphasised) == ["BPFI", "BPFO"]

    def test_golden_bearing_is_qualified_and_asks_for_confirmation(
        self, golden_scenario
    ):
        """AE11 on the claim this record allows: at 1 s and 10 kHz the cage
        line is not testable and BPFO = 9 x FTF lies in the outer-race window,
        so the outer-race indication is qualified (condition
        cage_comb_not_excluded), not single-spectrum evidence."""
        diagnosis = golden_scenario["diagnosis"]
        faults = diagnosis["bearing_faults"]
        advisory = build_advisory(diagnosis)
        statements = collect_statements(advisory, _figure(golden_scenario))

        assert faults["outcome"] == "indicated"
        assert faults["claim"]["level"] == "qualified"
        assert faults["claim"]["conditions"] == ["cage_comb_not_excluded"]
        assert advisory["verdict"]["statement"].startswith(
            "The envelope evidence of this single spectrum is consistent with an "
            "outer-race defect, but this record is too short to test the cage line"
        )
        assert advisory["verdict"]["fault_canonical"] is None
        bearing_actions = [
            r["action"]
            for r in advisory["recommendations"]
            if r["action"].startswith("Bearing")
        ]
        assert bearing_actions == [
            "Bearing 6205: repeat the measurement with a capture long enough to "
            "test the cage line, or compare with a comparable earlier measurement."
        ]
        assert not [
            s
            for s in statements
            if re.search(r"replace bearing|treat this as actionable", s, re.I)
        ]

    def test_golden_bearing_in_zone_d_keeps_the_zone_action(self, golden_scenario):
        advisory = build_advisory(_with_zone(golden_scenario["diagnosis"], "D"))
        actions = [r["action"] for r in advisory["recommendations"]]

        assert actions[0] == "Immediate shutdown recommended"
        assert any(
            a.startswith("Bearing 6205: repeat the measurement") for a in actions
        )
        assert not [a for a in actions if re.search(r"replac", a, re.I)]

    def test_no_scenario_statement_holds_a_forbidden_phrase(
        self,
        noise_scenarios,
        combined_scenario,
        combined_baseline,
        mfpt_scenario,
        golden_scenario,
    ):
        """AE13 over AE1, AE4, AE5 and AE11 (zone A, B and D), with the
        figure's caption and the badges."""
        cases = [
            (noise_scenarios[0], "B", None),
            (combined_scenario, "A", combined_baseline),
            (combined_scenario, "D", None),
            (mfpt_scenario, "A", None),
            (golden_scenario, "B", None),
            (golden_scenario, "D", None),
        ]
        offending, bearing_offending = [], []
        for scenario, zone, baseline in cases:
            advisory = build_advisory(
                _with_zone(scenario["diagnosis"], zone),
                baseline and baseline["diagnosis"],
            )
            figure = _figure(scenario)
            for statement in collect_statements(advisory, figure):
                if _offending(statement, AE13_FORBIDDEN, AE13_ALLOWED):
                    offending.append(statement)
            for statement in _bearing_statements(advisory, figure):
                if _offending(statement, BEARING_FORBIDDEN, BEARING_ALLOWED):
                    bearing_offending.append(statement)

        assert not offending, offending
        assert not bearing_offending, bearing_offending
