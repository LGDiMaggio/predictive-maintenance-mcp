"""Known defects of the two single-acquisition authors, held as strict tripwires.

The advisory (``decision_support/advisory.py``) is meant to be the single
author of single-acquisition verdicts, assessments and maintenance
recommendations, with every other surface projecting what it wrote. Today two
functions author them (``_synthesize_diagnosis`` behind
``diagnose_vibration``, ``build_advisory`` behind
``generate_diagnostic_report``), and for the same input their outputs
diverge. Each divergence that rule already makes a defect is held here as a
``pytest.mark.xfail(strict=True, raises=AssertionError)`` tripwire whose
reason starts with ``audit-known-defect:``:

- the recommended actions;
- the verdict;
- the shaft-order statement, which the report omits.

Each predicate asserts parity only. Which wording, order, recommendation table
or projection format is right is still open, and no predicate prefers one
author's text. Divergences whose fix needs a wording not yet decided (the
pair-baseline sentence, the zone B action next to the zone B description) are
not tripwires here. A satisfiability control beside each predicate shows it
passes when the defect is absent: on real outputs where today's code already
agrees, and on a corrected copy of a divergent case. When a change makes two
surfaces agree, the tripwire XPASSes, the strict run fails, and the marker is
removed with the body unchanged.

Parity is asserted only where both tools receive the same parameters and both
return (``SAME_INPUT_CASES``); the anomaly family's strings are left out
(``_t2_matrix``).
"""

from __future__ import annotations

import copy

import pytest

from _t2_matrix import (
    SAME_INPUT_CASES,
    SHAFT_ORDER_FIRST_WORDS,
    actions_agree,
    diagnosis_actions,
    evidence_strength_agrees,
    report_actions,
    report_recommendations,
    run_case,
    shaft_order_carried,
    shaft_order_sentence,
    t2_sandbox,  # noqa: F401  (fixture)
    verdict_stated_by_diagnosis,
)
from tools.audit.report import KNOWN_DEFECT_PREFIX


def known_defect(short_id: str, text: str) -> str:
    """The xfail reason the weekly audit lists: ``audit-known-defect: <id>: <text>``."""
    return f"{KNOWN_DEFECT_PREFIX} {short_id}: {text}"


#: Same-input cases where the two tools agree on the actions today (the
#: undeclared unit: both recommend the ISO refusal's remedy and nothing else).
ACTIONS_AGREE_TODAY = ("undeclared_unit",)
ACTIONS_DIVERGE_TODAY = tuple(
    c for c in SAME_INPUT_CASES if c not in ACTIONS_AGREE_TODAY
)

#: Same-input cases where the diagnosis text already states the report's
#: verdict today: a withheld bearing claim in zone A with no claim action, so
#: the verdict is the claim's sentence, which both authors take from
#: ``bearing_claims``.
VERDICT_AGREES_TODAY = ("strongest_line_2x_bearing",)
VERDICT_DIVERGES_TODAY = tuple(
    c for c in SAME_INPUT_CASES if c not in VERDICT_AGREES_TODAY
)

#: Same-input cases whose diagnosis states the shaft-order line today.
SHAFT_ORDER_CASES = (
    "strongest_line_1x",
    "strongest_line_2x_bearing",
    "outer_race_defect",
    "combined_defect",
)

ACTION_PARITY_DEFECT = known_defect(
    "diagnosis-report-action-parity",
    "for the same input diagnose_vibration and generate_diagnostic_report recommend "
    "different actions, although one author is meant to write both: the pipeline's "
    "synthesis keeps its own zone ladder (no action in zones A and B, its own wording "
    "in C and D) and its own default action, while the advisory takes the "
    "recommendation table's zone actions",
)
VERDICT_PARITY_DEFECT = known_defect(
    "diagnosis-report-verdict-parity",
    "diagnose_vibration's overall_diagnosis does not state the verdict "
    "generate_diagnostic_report states for the same input: the advisory composes a "
    "headline (the bearing claim with its actions, or the not-attempted or "
    "not-computed headline with its zone addendum) that the pipeline's synthesis "
    "never writes",
)
SHAFT_ORDER_DEFECT = known_defect(
    "report-omits-shaft-order-statement",
    "generate_diagnostic_report omits the shaft-order statement that "
    "diagnose_vibration carries for the same input",
)


class TestActionParity:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=ACTION_PARITY_DEFECT)
    @pytest.mark.parametrize("case_id", ACTIONS_DIVERGE_TODAY)
    def test_both_tools_recommend_the_same_actions(
        self, case_id, t2_sandbox, monkeypatch
    ):
        runs = run_case(case_id, t2_sandbox, monkeypatch)
        diagnosis, report = diagnosis_actions(runs), report_actions(runs)
        assert actions_agree(runs), {
            "diagnose_vibration only": sorted(diagnosis - report),
            "report only": sorted(report - diagnosis),
        }

    @pytest.mark.parametrize("case_id", ACTIONS_AGREE_TODAY)
    def test_control_the_predicate_holds_where_the_tools_already_agree(
        self, case_id, t2_sandbox, monkeypatch
    ):
        assert actions_agree(run_case(case_id, t2_sandbox, monkeypatch))

    def test_control_the_predicate_holds_on_a_corrected_divergent_case(
        self, t2_sandbox, monkeypatch
    ):
        corrected = copy.deepcopy(run_case("zone_b", t2_sandbox, monkeypatch))
        corrected["dv"]["recommendations"] = sorted(report_actions(corrected))
        assert actions_agree(corrected)

    @pytest.mark.parametrize("case_id", ACTIONS_DIVERGE_TODAY)
    def test_the_divergence_is_in_the_zone_and_default_actions_only(
        self, case_id, t2_sandbox, monkeypatch
    ):
        """Why the tripwire fails today, case by case: every report action the
        diagnosis lacks is a zone action of the recommendation table (its
        evidence is the ISO statement alone), and the diagnosis adds at most one
        action of its own, never one of the report's other actions (bearing
        claim actions, remedies). Still true once the two agree."""
        runs = run_case(case_id, t2_sandbox, monkeypatch)
        iso_statement = runs["advisory"]["iso"]["statement"]
        recommendations = report_recommendations(runs)
        zone = {
            r["action"] for r in recommendations if r["evidence"] == [iso_statement]
        }
        other = {r["action"] for r in recommendations} - zone
        report_only = report_actions(runs) - diagnosis_actions(runs)
        diagnosis_only = diagnosis_actions(runs) - report_actions(runs)
        assert runs["advisory"]["iso"]["status"] == "assessed"
        assert report_only <= zone
        assert diagnosis_only.isdisjoint(other)
        assert len(diagnosis_only) <= 1


class TestVerdictParity:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=VERDICT_PARITY_DEFECT)
    @pytest.mark.parametrize("case_id", VERDICT_DIVERGES_TODAY)
    def test_the_diagnosis_states_the_reports_verdict(
        self, case_id, t2_sandbox, monkeypatch
    ):
        runs = run_case(case_id, t2_sandbox, monkeypatch)
        assert verdict_stated_by_diagnosis(runs), runs["report"]["verdict"]

    @pytest.mark.parametrize("case_id", VERDICT_AGREES_TODAY)
    def test_control_the_predicate_holds_where_the_tools_already_agree(
        self, case_id, t2_sandbox, monkeypatch
    ):
        assert verdict_stated_by_diagnosis(run_case(case_id, t2_sandbox, monkeypatch))

    def test_control_the_predicate_holds_on_a_corrected_divergent_case(
        self, t2_sandbox, monkeypatch
    ):
        corrected = copy.deepcopy(run_case("zone_c", t2_sandbox, monkeypatch))
        dv = corrected["dv"]
        dv["overall_diagnosis"] = (
            f"{dv['overall_diagnosis']} {corrected['report']['verdict']}"
        )
        assert verdict_stated_by_diagnosis(corrected)


class TestShaftOrderStatementParity:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=SHAFT_ORDER_DEFECT)
    @pytest.mark.parametrize("case_id", SHAFT_ORDER_CASES)
    def test_the_report_states_the_shaft_order_statement(
        self, case_id, t2_sandbox, monkeypatch
    ):
        assert shaft_order_carried(run_case(case_id, t2_sandbox, monkeypatch))

    def test_control_the_predicate_holds_on_a_corrected_divergent_case(
        self, t2_sandbox, monkeypatch
    ):
        corrected = copy.deepcopy(
            run_case("strongest_line_1x", t2_sandbox, monkeypatch)
        )
        sentence = shaft_order_sentence(corrected["dv"]["overall_diagnosis"])
        corrected["report"]["statements"].append(sentence)
        assert shaft_order_carried(corrected)

    def test_control_a_diagnosis_without_the_line_fails_as_a_setup_error(
        self, t2_sandbox, monkeypatch
    ):
        """The predicate refuses a case it does not apply to with ``Failed``,
        which a ``raises=AssertionError`` tripwire cannot absorb."""
        runs = run_case("zone_a", t2_sandbox, monkeypatch)
        assert shaft_order_sentence(runs["dv"]["overall_diagnosis"]) is None
        with pytest.raises(pytest.fail.Exception, match="states no shaft-order line"):
            shaft_order_carried(runs)

    def test_the_cases_are_exactly_those_whose_diagnosis_states_the_line(
        self, t2_sandbox, monkeypatch
    ):
        """The tripwire's precondition, today: these cases, one statement each."""
        counts = {
            case_id: run_case(case_id, t2_sandbox, monkeypatch)["dv"][
                "overall_diagnosis"
            ].count(SHAFT_ORDER_FIRST_WORDS)
            for case_id in SAME_INPUT_CASES
        }
        assert {c for c, n in counts.items() if n} == set(SHAFT_ORDER_CASES)
        assert {counts[c] for c in SHAFT_ORDER_CASES} == {1}


class TestEvidenceStrengthParity:
    """Parity that holds today (no tripwire): the report copies the value its
    own run of the first author computed."""

    @pytest.mark.parametrize("case_id", SAME_INPUT_CASES)
    def test_both_tools_report_the_same_evidence_strength(
        self, case_id, t2_sandbox, monkeypatch
    ):
        assert evidence_strength_agrees(run_case(case_id, t2_sandbox, monkeypatch))
