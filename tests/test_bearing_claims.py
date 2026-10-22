"""The claim a bearing check states, from its significance-tested evidence.

``src/diagnostics/bearing_claims.py`` maps the evidence layer's result to one
of three claim levels (withheld, qualified, single-spectrum evidence), with
the conditions that qualify or withhold it and the limitations stated with
single-spectrum evidence. The mapping is checked on constructed evidence
results (states chosen per hypothesis, outcome by the evidence layer's rule)
and, end to end, on seeded spectra whose content is known by construction
(``tests/_bearing_scenarios.py``). Every expectation follows from the
declared frequencies and how the result or the signal was built.
"""

import re

import numpy as np
import pytest
from pydantic import ValidationError

from _bearing_scenarios import DEFECT_SEED, FREQUENCIES_6205, RPM_6205, defect
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
    check_frequency_set,
)
from predictive_maintenance_mcp.diagnostics.bearing_claims import (
    CLAIM_LEVELS,
    CLAIM_SENTENCES,
    acquisition_eligibility_from_companion,
    bearing_claim,
    claim_actions,
    claim_statement,
    counts_as_finding,
    named_candidate,
)
from predictive_maintenance_mcp.diagnostics.bearing_evidence import Hypothesis
from predictive_maintenance_mcp.models import BearingClaim

FS = 12000.0
SHAFT_HZ = RPM_6205 / 60.0

CANONICAL = {"BPFO": "outer_race", "BPFI": "inner_race", "BSF": "ball", "FTF": "cage"}

#: The MFPT bearing at 1500 rpm (the bundled recordings' companions).
FREQUENCIES_MFPT = {"BPFO": 81.125, "BPFI": 118.875, "BSF": 63.91, "FTF": 14.8375}
SHAFT_MFPT_HZ = 1500.0 / 60.0

#: What the claim's sentences never say. "not a confirmed fault" and "is not a
#: probability" are the allowed negations.
FORBIDDEN = re.compile(
    r"\bconfirmed\b|\bdetected\b|damage|\breplac\w*|healthy|most likely"
    r"|faults? detected:|probability|—",
    re.IGNORECASE,
)
ALLOWED_NEGATIONS = ("not a confirmed fault", "is not a probability")

ASSUMPTION = (
    "This assumes the record's background is random measurement noise, which "
    "its metadata cannot confirm; it is not a confirmed fault."
)

_DEFAULT_REASON = {
    "absent": "no_significant_line",
    "supported": "corroborated",
    "unconfirmed": "no_corroboration",
    "not_testable": "search_window_too_narrow",
}


def _offending(sentence: str) -> list[str]:
    for allowed in ALLOWED_NEGATIONS:
        sentence = sentence.replace(allowed, "")
    return FORBIDDEN.findall(sentence)


def _constructed(
    frequencies: dict[str, float],
    states: dict[str, str],
    *,
    reasons: dict[str, str] | None = None,
    counted: tuple[str, ...] = ("harmonic",),
) -> tuple[list[Hypothesis], dict]:
    """Hypotheses and an evidence result with the given state per label.

    Labels not in ``states`` are absent. A supported check has one counted
    corroborating line of each kind in ``counted``; the outcome follows the
    evidence layer's rule (one supported: indicated; two or more: several
    types; else inconclusive when a line is unconfirmed or shared or nothing
    is testable; else no indication).
    """
    hypotheses = [
        Hypothesis(k, float(v), CANONICAL.get(k)) for k, v in frequencies.items()
    ]
    reasons = reasons or {}
    checks = []
    for h in hypotheses:
        state = states.get(h.label, "absent")
        testable = state != "not_testable"
        significant = state in ("supported", "unconfirmed", "shared")
        checks.append(
            {
                "fault_type": h.label,
                "state": state,
                "state_reason": reasons.get(h.label, _DEFAULT_REASON[state]),
                "line_frequency_hz": round(h.line_hz, 3),
                "line_peak_hz": round(h.line_hz, 3) if testable else None,
                "level_db": (20.0 if significant else 3.0) if testable else None,
                "corroboration": (
                    [{"kind": kind, "counted": True} for kind in counted]
                    if state == "supported"
                    else []
                ),
            }
        )
    supported = [c for c in checks if c["state"] == "supported"]
    if len(supported) == 1:
        outcome = "indicated"
    elif supported:
        outcome = "several_types"
    elif any(c["state"] in ("unconfirmed", "shared") for c in checks) or all(
        c["state"] == "not_testable" for c in checks
    ):
        outcome = "inconclusive"
    else:
        outcome = "no_indication"
    candidates = [
        {
            "fault_type": c["fault_type"],
            "fault_type_canonical": CANONICAL.get(c["fault_type"]),
            "line_frequency_hz": c["line_frequency_hz"],
            "line_peak_hz": c["line_peak_hz"],
            "level_db": c["level_db"],
        }
        for c in supported
    ]
    return hypotheses, {"checks": checks, "outcome": outcome, "candidates": candidates}


def _claim(
    hypotheses,
    evidence,
    *,
    eligibility: str = "unknown",
    shaft_hz: float = SHAFT_HZ,
) -> dict:
    return bearing_claim(
        evidence,
        hypotheses,
        shaft_hz=shaft_hz,
        tolerance_pct=5.0,
        acquisition_eligibility=eligibility,
    )


def _summary(hypotheses, evidence, claim, bearing_id="6205", next_step=None) -> dict:
    """The parts of a check summary the actions read."""
    return {
        "claim": claim,
        "bearing_id": bearing_id,
        "candidates": evidence["candidates"],
        "fault_checks": [
            {**c, "fault_type_canonical": CANONICAL.get(c["fault_type"])}
            for c in evidence["checks"]
        ],
        "next_step": next_step,
    }


def _level(claim: dict) -> tuple:
    return claim["level"], claim["conditions"], claim["limitations"]


# ===========================================================================
# Levels and conditions, on constructed evidence
# ===========================================================================


class TestClaimLevels:
    def test_indicated_with_nothing_to_qualify_is_single_spectrum_evidence(self):
        """The cage line is testable and absent, the inner race is absent."""
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"BPFO": "supported"})
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert _level(claim) == ("single_spectrum_evidence", [], [])
        assert claim["acquisition_eligibility"] == "unknown"
        assert claim["basis"] == "significance_tested"
        assert claim["policy_version"] == 1
        assert statement.startswith(
            "The envelope evidence of this single spectrum is consistent with an "
            "outer-race defect: the outer-race line (BPFO) at "
            f"{FREQUENCIES_6205['BPFO']:.2f} Hz stands 20.0 dB above the local "
            "noise floor."
        )
        assert statement.endswith(ASSUMPTION)
        assert named_candidate(claim, evidence)["fault_type"] == "BPFO"

    def test_several_types_is_qualified_and_names_none(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "BPFI": "supported"}
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert _level(claim) == ("qualified", ["several_types"], [])
        assert named_candidate(claim, evidence) is None
        assert "none is singled out" in statement
        assert "the outer-race line (BPFO) at" in statement
        assert "the inner-race line (BPFI) at" in statement
        assert statement.count("(20.0 dB)") == 2
        assert statement.endswith("This is not a confirmed fault.")

    def test_untestable_cage_with_a_cage_harmonic_in_the_window_qualifies(self):
        """M2: on the 6205, BPFO = 9 x FTF, so the ninth cage harmonic lies
        in the outer-race window whenever the cage line cannot be tested."""
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "FTF": "not_testable"}
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert _level(claim) == ("qualified", ["cage_comb_not_excluded"], [])
        assert named_candidate(claim, evidence) is None
        assert statement == (
            "The envelope evidence of this single spectrum is consistent with an "
            "outer-race defect, but this record is too short to test the cage line "
            "(FTF), and a harmonic of the cage frequency falls within the search "
            "window of the outer-race line (BPFO), so a cage defect cannot be "
            "excluded. No single fault type is named from this spectrum."
        )

    def test_a_testable_cage_line_qualifies_nothing(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "FTF": "absent"}
        )

        assert _claim(hypotheses, evidence)["level"] == "single_spectrum_evidence"

    def test_no_cage_harmonic_in_the_window_qualifies_nothing(self):
        """A 54 Hz line searched at +/- 5 % (51.3 to 56.7 Hz) holds no
        multiple of a 12 Hz cage rate (48 and 60 Hz lie outside)."""
        frequencies = {"BPFI": 54.0, "FTF": 12.0}
        assert not any(abs(k * 12.0 - 54.0) <= 0.05 * 54.0 for k in range(1, 6))
        hypotheses, evidence = _constructed(
            frequencies, {"BPFI": "supported", "FTF": "not_testable"}
        )

        assert _level(_claim(hypotheses, evidence)) == (
            "single_spectrum_evidence",
            [],
            [],
        )

    def test_significant_inner_race_with_its_sideband_in_the_window_qualifies(self):
        """M3: on the 6205 at 1797 rpm, BPFI - 2 x shaft rate = 102.3 Hz lies
        inside the outer-race window (107.4 Hz +/- 5.4 Hz)."""
        line = FREQUENCIES_6205["BPFI"] - 2 * SHAFT_HZ - FREQUENCIES_6205["BPFO"]
        assert abs(line) <= 0.05 * FREQUENCIES_6205["BPFO"]
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "BPFI": "unconfirmed"}
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert _level(claim) == ("qualified", ["inner_race_sideband_not_excluded"], [])
        assert statement == (
            "The envelope evidence of this single spectrum is consistent with an "
            "outer-race defect, but the inner-race line (BPFI) also stands above "
            "the local noise floor, and its sideband at BPFI minus twice the shaft "
            "rate falls within the search window of the outer-race line (BPFO), so "
            "an inner-race defect cannot be excluded. No single fault type is named "
            "from this spectrum."
        )

    def test_an_absent_inner_race_line_qualifies_nothing(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "BPFI": "absent"}
        )

        assert _claim(hypotheses, evidence)["level"] == "single_spectrum_evidence"

    def test_an_untestable_inner_race_line_qualifies_nothing(self):
        """M3 needs an inner-race state neither absent nor not_testable: a line
        the record could not test is not said to stand above the floor, even
        with the shaft rate given and its sideband in the outer-race window."""
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFO": "supported", "BPFI": "not_testable"}
        )

        assert _level(_claim(hypotheses, evidence, shaft_hz=SHAFT_HZ)) == (
            "single_spectrum_evidence",
            [],
            [],
        )

    def test_an_inner_race_sideband_outside_the_window_qualifies_nothing(self):
        """MFPT at 1500 rpm: BPFI - 2 x 25 Hz = 68.9 Hz, outside the outer-race
        window (81.1 Hz +/- 4.1 Hz)."""
        hypotheses, evidence = _constructed(
            FREQUENCIES_MFPT, {"BPFO": "supported", "BPFI": "unconfirmed"}
        )
        claim = _claim(hypotheses, evidence, shaft_hz=SHAFT_MFPT_HZ)

        assert claim["level"] == "single_spectrum_evidence"

    def test_both_window_predicates_name_both_alternatives(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205,
            {"BPFO": "supported", "BPFI": "unconfirmed", "FTF": "not_testable"},
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert claim["conditions"] == [
            "cage_comb_not_excluded",
            "inner_race_sideband_not_excluded",
        ]
        assert "a cage defect cannot be excluded; and the inner-race line" in statement
        assert statement.endswith("No single fault type is named from this spectrum.")

    def test_a_cage_held_back_for_the_rolling_element_is_qualified(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205,
            {"FTF": "unconfirmed", "BSF": "unconfirmed"},
            reasons={"FTF": "rolling_element_line_present"},
        )
        claim = _claim(hypotheses, evidence)

        assert _level(claim) == ("qualified", ["rolling_element_held_back"], [])
        assert claim_statement(claim, evidence, hypotheses) == (
            "The significance-tested evidence is inconclusive and does not support "
            "a unique fault type: the cage lines may come from a cage defect or "
            "from a rolling-element defect."
        )

    def test_a_held_back_cage_names_a_shaft_harmonic_at_a_shaft_order(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205,
            {"FTF": "unconfirmed", "BSF": "unconfirmed"},
            reasons={
                "FTF": "rolling_element_line_present",
                "BSF": "line_at_shaft_order",
            },
        )
        claim = _claim(hypotheses, evidence)

        assert claim_statement(claim, evidence, hypotheses).endswith(
            "rolling-element defect, and the line in the rolling-element window "
            "lies at a shaft order and may be a shaft harmonic."
        )

    def test_inconclusive_is_withheld(self):
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"BPFI": "unconfirmed"})
        claim = _claim(hypotheses, evidence)

        assert _level(claim) == ("withheld", ["inconclusive"], [])
        assert claim_statement(claim, evidence, hypotheses) == (
            "The significance-tested evidence is inconclusive and does not support "
            "a unique fault type."
        )

    def test_no_indication_is_withheld_and_never_reads_as_free_of_defects(self):
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"FTF": "not_testable"})
        claim = _claim(hypotheses, evidence)

        assert _level(claim) == ("withheld", ["no_indication"], [])
        assert claim_statement(claim, evidence, hypotheses) == (
            "No eligible line at a tested characteristic frequency stands above the "
            "local noise floor in this record and band. This is not evidence that "
            "the bearing is free of defects. The cage line (FTF) could not be "
            "tested in this record."
        )

    def test_nothing_testable_is_withheld(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {k: "not_testable" for k in FREQUENCIES_6205}
        )
        claim = _claim(hypotheses, evidence)

        assert _level(claim) == ("withheld", ["nothing_testable"], [])
        assert claim_statement(claim, evidence, hypotheses) == (
            "None of the lines could be tested in this record, so no fault type is "
            "assessed."
        )

    def test_a_generated_signal_without_noise_withholds_an_indication(self):
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"BPFO": "supported"})
        claim = _claim(hypotheses, evidence, eligibility="known_inapplicable")

        assert _level(claim) == ("withheld", ["generated_without_noise"], [])
        assert claim["acquisition_eligibility"] == "known_inapplicable"
        assert named_candidate(claim, evidence) is None
        assert claim_statement(claim, evidence, hypotheses) == (
            "The signal's companion metadata declares a generated signal with an "
            "additive noise level of 0; the line test assumes a recording with "
            "measurement noise, so no bearing fault type is assessed."
        )


class TestLabelsOutsideTheVocabulary:
    def test_no_element_and_no_cage_comb(self):
        """A GMF line at 9 x a not-testable FTF: the cage-comb predicate would
        hold for a bearing type, but a label outside the vocabulary gets
        neither the predicate nor an element."""
        frequencies = {
            "GMF": 9 * FREQUENCIES_6205["FTF"],
            "FTF": FREQUENCIES_6205["FTF"],
        }
        hypotheses, evidence = _constructed(
            frequencies, {"GMF": "supported", "FTF": "not_testable"}
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)
        actions = claim_actions(_summary(hypotheses, evidence, claim))

        assert _level(claim) == ("single_spectrum_evidence", [], [])
        assert statement.startswith(
            "The envelope evidence of this single spectrum is consistent with a "
            "periodic line at the declared GMF frequency: the GMF line at"
        )
        assert "defect" not in statement and "race" not in statement
        assert actions == [
            "Bearing 6205: confirm the single-spectrum indication of the GMF line "
            "with a comparable measurement."
        ]

    def test_no_inner_race_sideband(self):
        frequencies = {
            "GMF": FREQUENCIES_6205["BPFO"],
            "BPFI": FREQUENCIES_6205["BPFI"],
        }
        hypotheses, evidence = _constructed(
            frequencies, {"GMF": "supported", "BPFI": "unconfirmed"}
        )

        assert _level(_claim(hypotheses, evidence)) == (
            "single_spectrum_evidence",
            [],
            [],
        )

    def test_several_types_name_the_label_without_an_element(self):
        frequencies = {"BPFO": FREQUENCIES_6205["BPFO"], "GMF": 350.0}
        hypotheses, evidence = _constructed(
            frequencies, {"BPFO": "supported", "GMF": "supported"}
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert "the GMF line at 350.00 Hz (20.0 dB) for a periodic line" in statement
        assert claim_actions(_summary(hypotheses, evidence, claim)) == [
            "Bearing 6205: compare with a comparable earlier measurement before "
            "acting on the BPFO and GMF lines."
        ]

    def test_no_indication_is_label_neutral(self):
        hypotheses, evidence = _constructed({"GMF": 350.0}, {})
        claim = _claim(hypotheses, evidence)

        assert claim_statement(claim, evidence, hypotheses).endswith(
            "This is not evidence that no defect exists."
        )


class TestLimitations:
    def test_sideband_only_inner_race_is_stated_without_a_level_change(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFI": "supported"}, counted=("sideband",)
        )
        claim = _claim(hypotheses, evidence)
        statement = claim_statement(claim, evidence, hypotheses)

        assert _level(claim) == (
            "single_spectrum_evidence",
            [],
            ["sideband_only_corroboration"],
        )
        assert statement.endswith(
            f"{ASSUMPTION} Its only corroborating lines are shaft-rate sidebands, "
            "which shaft-rate modulation of the background can also produce, so they "
            "do not independently confirm an inner-race defect."
        )

    def test_a_harmonic_beside_the_sidebands_is_no_limitation(self):
        hypotheses, evidence = _constructed(
            FREQUENCIES_6205, {"BPFI": "supported"}, counted=("sideband", "harmonic")
        )

        assert _claim(hypotheses, evidence)["limitations"] == []

    def test_an_undeclared_cage_is_stated(self):
        frequencies = {k: v for k, v in FREQUENCIES_6205.items() if k != "FTF"}
        hypotheses, evidence = _constructed(frequencies, {"BPFO": "supported"})
        claim = _claim(hypotheses, evidence)

        assert _level(claim) == ("single_spectrum_evidence", [], ["cage_not_declared"])
        assert claim_statement(claim, evidence, hypotheses).endswith(
            f"{ASSUMPTION} No cage rate was declared, so whether cage-rate lines "
            "could explain this line was not assessed."
        )


class TestEvidencePointsAndActions:
    CASES = {
        "evidence": ({"BPFO": "supported"}, {}, True),
        "several_types": ({"BPFO": "supported", "BPFI": "supported"}, {}, True),
        "cage_comb": ({"BPFO": "supported", "FTF": "not_testable"}, {}, True),
        "inner_race_sideband": ({"BPFO": "supported", "BPFI": "unconfirmed"}, {}, True),
        "held_back": (
            {"FTF": "unconfirmed", "BSF": "unconfirmed"},
            {"FTF": "rolling_element_line_present"},
            False,
        ),
        "inconclusive": ({"BPFI": "unconfirmed"}, {}, False),
        "no_indication": ({}, {}, False),
        "nothing_testable": ({k: "not_testable" for k in FREQUENCIES_6205}, {}, False),
    }

    @pytest.mark.parametrize("case", list(CASES))
    def test_a_finding_counts_only_from_an_indicated_or_several_types_outcome(
        self, case
    ):
        states, reasons, finding = self.CASES[case]
        hypotheses, evidence = _constructed(FREQUENCIES_6205, states, reasons=reasons)

        assert counts_as_finding(_claim(hypotheses, evidence)) is finding

    def test_a_generated_signal_without_noise_counts_nothing(self):
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"BPFO": "supported"})
        claim = _claim(hypotheses, evidence, eligibility="known_inapplicable")

        assert not counts_as_finding(claim)
        assert claim_actions(_summary(hypotheses, evidence, claim)) == []

    @pytest.mark.parametrize(
        "case, expected",
        [
            (
                "evidence",
                [
                    "Bearing 6205: confirm the single-spectrum indication with a "
                    "comparable measurement or an inspection of the outer race."
                ],
            ),
            (
                "several_types",
                [
                    "Bearing 6205: inspect the outer race and the inner race with "
                    "equal priority, and compare with a comparable earlier measurement."
                ],
            ),
            (
                "cage_comb",
                [
                    "Bearing 6205: repeat the measurement with a capture long enough "
                    "to test the cage line, or compare with a comparable earlier "
                    "measurement."
                ],
            ),
            (
                "inner_race_sideband",
                [
                    "Bearing 6205: inspect the outer and inner races with equal "
                    "priority, or compare with a comparable earlier measurement."
                ],
            ),
            (
                "held_back",
                [
                    "Bearing 6205: repeat the measurement, or compare it with a "
                    "comparable earlier measurement."
                ],
            ),
            (
                "inconclusive",
                [
                    "Bearing 6205: repeat the measurement, or compare it with a "
                    "comparable earlier measurement."
                ],
            ),
            ("no_indication", []),
            (
                "nothing_testable",
                ["Bearing 6205: record a longer capture to test the cage line (FTF)."],
            ),
        ],
    )
    def test_actions(self, case, expected):
        states, reasons, _ = self.CASES[case]
        hypotheses, evidence = _constructed(FREQUENCIES_6205, states, reasons=reasons)
        claim = _claim(hypotheses, evidence)
        summary = _summary(
            hypotheses,
            evidence,
            claim,
            next_step="Record a longer capture to test the cage line (FTF).",
        )

        assert claim_actions(summary) == expected

    def test_actions_without_a_bearing_designation(self):
        hypotheses, evidence = _constructed(FREQUENCIES_6205, {"BPFI": "unconfirmed"})
        claim = _claim(hypotheses, evidence)

        assert claim_actions(
            _summary(hypotheses, evidence, claim, bearing_id=None)
        ) == [
            "Bearing check: repeat the measurement, or compare it with a comparable "
            "earlier measurement."
        ]


# ===========================================================================
# The closed list and the reserved level
# ===========================================================================


@pytest.mark.parametrize("key", list(CLAIM_SENTENCES))
def test_no_closed_list_sentence_holds_a_forbidden_phrase(key):
    assert not _offending(CLAIM_SENTENCES[key]), CLAIM_SENTENCES[key]


def test_every_rendered_statement_and_action_is_free_of_forbidden_phrases():
    rendered = []
    for states, reasons, _ in TestEvidencePointsAndActions.CASES.values():
        for eligibility in ("unknown", "known_inapplicable"):
            hypotheses, evidence = _constructed(
                FREQUENCIES_6205, states, reasons=reasons
            )
            claim = _claim(hypotheses, evidence, eligibility=eligibility)
            rendered.append(claim_statement(claim, evidence, hypotheses))
            rendered += claim_actions(
                _summary(hypotheses, evidence, claim, next_step="Record more.")
            )

    assert not [s for s in rendered if _offending(s)]


def test_the_reserved_level_is_never_produced_and_never_valid():
    """``supported_claim`` is reserved for evidence independent of one
    spectrum: no level of the closed vocabulary, and the model refuses it."""
    levels = set()
    for states, reasons, _ in TestEvidencePointsAndActions.CASES.values():
        for eligibility in ("unknown", "known_inapplicable"):
            hypotheses, evidence = _constructed(
                FREQUENCIES_6205, states, reasons=reasons
            )
            claim = _claim(hypotheses, evidence, eligibility=eligibility)
            levels.add(claim["level"])
            BearingClaim(**claim)

    assert levels == set(CLAIM_LEVELS)
    assert "supported_claim" not in CLAIM_LEVELS
    with pytest.raises(ValidationError):
        BearingClaim(
            level="supported_claim",
            conditions=[],
            limitations=[],
            acquisition_eligibility="unknown",
            basis="significance_tested",
            policy_version=1,
        )


# ===========================================================================
# Acquisition eligibility from the companion metadata
# ===========================================================================


@pytest.mark.parametrize(
    "companion, expected",
    [
        ({"generator": "generate_test_signal", "noise_level": 0}, "known_inapplicable"),
        (
            {"generator": "generate_test_signal", "noise_level": 0.0},
            "known_inapplicable",
        ),
        ({"generator": "generate_test_signal", "noise_level": 0.1}, "unknown"),
        ({"generator": "generate_test_signal", "noise_level": "0"}, "unknown"),
        ({"generator": "generate_test_signal"}, "unknown"),
        ({"noise_level": 0}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_only_a_declared_noiseless_generated_signal_is_known_inapplicable(
    companion, expected
):
    assert acquisition_eligibility_from_companion(companion) == expected


# ===========================================================================
# End to end through the analyzer
# ===========================================================================


def _outer_race(seconds: float, snr_db: float, seed: int) -> np.ndarray:
    return defect(
        "outer",
        np.random.default_rng((DEFECT_SEED, 40 if seconds > 1 else 41, seed)),
        FS,
        seconds,
        FREQUENCIES_6205,
        SHAFT_HZ,
        snr_db,
    )


def _catalog(signal, **kwargs) -> dict:
    return check_all_bearing_faults(signal, FS, "6205", RPM_6205, **kwargs)


class TestAnalyzerSummary:
    def test_ten_second_outer_race_defect_is_single_spectrum_evidence(self):
        """At 10 s the cage line is testable, and the record holds no
        inner-race defect: nothing qualifies the outer-race indication."""
        result = _catalog(_outer_race(10.0, -15.0, 0))
        bpfo = next(c for c in result["fault_checks"] if c["fault_type"] == "BPFO")

        assert result["outcome"] == "indicated"
        assert result["claim"]["level"] == "single_spectrum_evidence"
        assert result["most_likely_fault"] == "BPFO"
        assert result["most_likely_fault_canonical"] == "outer_race"
        assert result["overall_assessment"] == (
            "The envelope evidence of this single spectrum is consistent with an "
            f"outer-race defect: the outer-race line (BPFO) at {bpfo['line_peak_hz']:.2f}"
            f" Hz stands {bpfo['level_db']:.1f} dB above the local noise floor. "
            f"{ASSUMPTION}"
        )

    def test_one_second_record_qualifies_an_outer_race_indication(self):
        """AE10 with M2: a 1 s record at 12 kHz cannot test the cage line, and
        BPFO = 9 x FTF lies in the outer-race window."""
        result = _catalog(_outer_race(1.0, 0.0, 0))
        ftf = next(c for c in result["fault_checks"] if c["fault_type"] == "FTF")

        assert result["outcome"] == "indicated"
        assert [c["fault_type"] for c in result["candidates"]] == ["BPFO"]
        assert ftf["state"] == "not_testable"
        assert result["claim"]["level"] == "qualified"
        assert result["claim"]["conditions"] == ["cage_comb_not_excluded"]
        assert result["most_likely_fault"] is None
        assert result["most_likely_fault_canonical"] is None
        assert "so a cage defect cannot be excluded" in result["overall_assessment"]

    def test_combined_defects_name_no_single_type(self):
        """AE4: both races, neither singled out."""
        signal = defect(
            "combined",
            np.random.default_rng((DEFECT_SEED, 42, 0)),
            FS,
            10.0,
            FREQUENCIES_6205,
            SHAFT_HZ,
            -10.0,
        )
        result = _catalog(signal)

        assert result["outcome"] == "several_types"
        assert result["claim"]["conditions"] == ["several_types"]
        assert result["most_likely_fault"] is None
        assert result["most_likely_fault_canonical"] is None
        assert "most likely" not in result["overall_assessment"].lower()

    def test_known_inapplicable_withholds_and_changes_no_evidence_field(self):
        signal = _outer_race(10.0, -15.0, 0)
        unknown = _catalog(signal)
        withheld = _catalog(signal, acquisition_eligibility="known_inapplicable")

        assert withheld["claim"]["conditions"] == ["generated_without_noise"]
        assert withheld["most_likely_fault"] is None
        for key in ("outcome", "outcome_reason", "next_step", "candidates"):
            assert withheld[key] == unknown[key], key
        assert withheld["fault_checks"] == unknown["fault_checks"]

    def test_an_unknown_eligibility_value_is_refused(self):
        with pytest.raises(ValueError, match="acquisition_eligibility"):
            _catalog(_outer_race(1.0, 0.0, 0), acquisition_eligibility="eligible")
        with pytest.raises(ValueError, match="acquisition_eligibility"):
            check_frequency_set(
                _outer_race(1.0, 0.0, 0),
                FS,
                FREQUENCIES_6205,
                RPM_6205,
                acquisition_eligibility="eligible",
            )

    def test_a_lowercase_label_keeps_its_canonical_form(self):
        """The candidate's canonical form upper-cases the label; the summary
        used to look the label up as given and lost it."""
        lower = {k.lower(): v for k, v in FREQUENCIES_6205.items()}
        result = check_frequency_set(_outer_race(10.0, -15.0, 0), FS, lower, RPM_6205)

        assert result["claim"]["level"] == "single_spectrum_evidence"
        assert result["most_likely_fault"] == "bpfo"
        assert result["most_likely_fault_canonical"] == "outer_race"
