"""Characterization of what both single-acquisition authors say today.

Every case of the synthetic matrix (``_t2_matrix.CASES``) runs through
``diagnose_vibration`` and ``generate_diagnostic_report`` (HTML), and each
test compares today's output with a golden in ``tests/fixtures/t2/``.

These goldens are CHARACTERIZATION, not requirements. They record what the
code says on the base they were generated from, so that the work bringing the
two authors together can show what it changes: a change meant to leave every
output byte-identical keeps them green, and an output-changing one regenerates
them and puts the difference in front of the wording review. A failing test
here means "the output changed", never "the output is wrong". The divergences
between the two authors that are already known defects are held separately,
as strict tripwires, in ``test_t2_known_defect_tripwires.py``.

No anomaly model exists in any case and the anomaly family's strings are left
out of every golden (see ``_t2_matrix``). Regenerate with
``T2_CHARACTERIZATION_WRITE=1``.
"""

from __future__ import annotations

import pytest

from _t2_matrix import (
    BASELINE_PRIMARY,
    BASELINES,
    CASES,
    GOLDEN_DIR,
    REPORTED_CASES,
    characterization,
    check_golden,
    controlled_union_route,
    different_bearing_baseline,
    refusals,
    report_projection,
    run_baseline_case,
    run_case,
    t2_sandbox,  # noqa: F401  (fixture)
)

EXTRA_GOLDENS = (
    "corrupt_input_refusals",
    "composer_controlled_union",
    "composer_baseline_other_bearing",
    *BASELINES,
)


@pytest.mark.parametrize("case_id", list(CASES))
def test_today_both_authors_match_the_golden(case_id, t2_sandbox, monkeypatch):
    runs = run_case(case_id, t2_sandbox, monkeypatch)
    check_golden(case_id, CASES[case_id].description, characterization(runs))


@pytest.mark.parametrize("case_id", list(BASELINES))
def test_today_the_report_against_a_baseline_matches_the_golden(
    case_id, t2_sandbox, monkeypatch
):
    runs = run_baseline_case(case_id, t2_sandbox, monkeypatch)
    description = f"{BASELINE_PRIMARY.description}, {BASELINES[case_id].description}"
    check_golden(case_id, description, report_projection(runs))


def test_today_corrupt_inputs_are_refused_as_the_golden_records(t2_sandbox):
    check_golden(
        "corrupt_input_refusals",
        "rpm 0 through both tools; a NaN sample and fs 0 through the pipeline",
        refusals(t2_sandbox),
    )


def test_today_the_composer_on_a_controlled_union_result_matches_the_golden(
    t2_sandbox,
):
    check_golden(
        "composer_controlled_union",
        "The advisory on the outer-race defect with an experimental "
        "controlled_union_v1 bearing result",
        controlled_union_route(t2_sandbox),
    )


def test_today_a_baseline_on_another_bearing_matches_the_golden(t2_sandbox):
    check_golden(
        "composer_baseline_other_bearing",
        "The advisory's baseline comparison against a baseline analysed for 6203",
        different_bearing_baseline(t2_sandbox),
    )


def test_every_golden_belongs_to_a_case_and_every_case_has_one():
    """A golden left behind, or a case without one, is caught here."""
    on_disk = {path.stem for path in GOLDEN_DIR.glob("*.json")}
    assert on_disk == {*CASES, *EXTRA_GOLDENS}


class TestTheReportPathToday:
    """How the report reaches its interpretation today: it runs the first
    author again and keeps only its ``evidence_strength``, and it renders every
    statement it returns. The cases where the report raises instead are
    recorded by their goldens."""

    @pytest.mark.parametrize("case_id", REPORTED_CASES)
    def test_the_report_copies_evidence_strength_from_its_own_pipeline_run(
        self, case_id, t2_sandbox, monkeypatch
    ):
        runs = run_case(case_id, t2_sandbox, monkeypatch)
        (inner,) = runs["report_diagnoses"]
        assert runs["report"]["evidence_strength"] == inner["evidence_strength"]

    @pytest.mark.parametrize("case_id", REPORTED_CASES)
    def test_the_html_renders_every_statement_the_report_returns(
        self, case_id, t2_sandbox, monkeypatch
    ):
        runs = run_case(case_id, t2_sandbox, monkeypatch)
        assert runs["report"]["statements"]
        assert runs["html_missing"] == []
