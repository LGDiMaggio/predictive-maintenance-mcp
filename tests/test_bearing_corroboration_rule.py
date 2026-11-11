"""The experimental, per-call ``controlled_union_v1`` corroboration rule.

The released rule (``any_line``) is the default of every call and must be
reproduced exactly. The experimental rule changes only the corroboration
decision: per-line evidence keeps its released meaning, the rule's decision is
recorded apart from the lines, its results carry their own identifiers, and
the asset ledger never reads them as if they came from the released rule.
"""

import copy
import inspect
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from predictive_maintenance_mcp.asset_ledger import assessment as lg_assessment
from predictive_maintenance_mcp.asset_ledger import snapshot as lg_snapshot
from predictive_maintenance_mcp.decision_support import advisory
from predictive_maintenance_mcp.diagnostics import bearing_analyzer as ba
from predictive_maintenance_mcp.diagnostics import bearing_claims as bc
from predictive_maintenance_mcp.diagnostics import bearing_evidence as be
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import _hypotheses
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools
from predictive_maintenance_mcp.models import (
    BearingCorroboratingLine,
    BearingEvidencePolicy,
    BearingFaultCheckResult,
)

RELEASED = be.CORROBORATION_RULE_ANY_LINE
UNION = be.CORROBORATION_RULE_CONTROLLED_UNION
EXPECTED = [("BPFO", 107.36), ("BPFI", 162.19), ("BSF", 70.58), ("FTF", 11.93)]
# A clear outer-race line with three moderate harmonics, none significant alone.
MODERATE_HARMONICS = (5, {107.4: 9.0, 214.8: 2.6, 322.2: 2.6, 429.6: 2.6})
# A clear outer-race line with one harmonic just above its own threshold.
LONE_HARMONIC = (2002, {107.4: 9.0, 214.8: 3.0})
SEVERAL = (
    5,
    {107.4: 9.0, 214.8: 2.6, 322.2: 2.6, 429.6: 2.6, 350.0: 9.0, 700.0: 9.0},
)


def _spectrum(seed: int, lines: dict[float, float]):
    rng = np.random.default_rng(seed)
    freqs = np.arange(0.0, 2500.0, 0.1)
    mags = np.sqrt(rng.exponential(1.0, freqs.size))
    for hz, amp in lines.items():
        mags[int(round(hz / 0.1))] = amp
    return freqs, mags


def _evaluate(freqs, mags, rule=RELEASED, expected=EXPECTED, shaft_hz=29.95):
    return be.evaluate_hypotheses(
        freqs,
        mags,
        _hypotheses(expected),
        shaft_hz=shaft_hz,
        cage_hz=11.93,
        corroboration_rule=rule,
    )


def _summary(evidence, expected=EXPECTED, shaft_hz=29.95):
    return {
        **evidence,
        "fault_checks": evidence["checks"],
        **ba._summary_evidence(
            evidence,
            expected,
            shaft_hz=shaft_hz,
            tolerance_pct=5.0,
            acquisition_eligibility="unknown",
        ),
    }


def _assert_per_line_evidence_is_truthful(result):
    for check in result["checks"]:
        for entry in check["corroboration"]:
            # The released meaning, whichever rule ran.
            assert entry["counted"] == (
                entry["significant"] and entry["excluded_by"] is None
            )
            if entry["excluded_by"] is not None:
                assert entry["significant"]
            if entry["excluded_by"] == be.EXCLUDED_BY_AGGREGATE_RULE:
                assert not check["aggregate_corroboration"]["corroborated"]
        harmonics = {h["harmonic"] for h in check["harmonics_detected"]}
        counted = {
            e["harmonic"]
            for e in check["corroboration"]
            if e["counted"] and e["kind"] == "harmonic"
        }
        assert harmonics == counted


# --- selection --------------------------------------------------------------------


def test_every_public_entry_defaults_to_the_released_rule():
    for function in (
        be.evaluate_hypotheses,
        be.evidence_policy,
        ba._check_on_one_spectrum,
        ba.check_all_bearing_faults,
        ba.check_frequency_set,
    ):
        parameter = inspect.signature(function).parameters["corroboration_rule"]
        assert parameter.default == RELEASED
    assert lg_snapshot.SnapshotPolicy().corroboration_rule == RELEASED


@pytest.mark.parametrize(
    "rule", ["union", "pooled", "", None, 1, "CONTROLLED_UNION_V1"]
)
def test_an_unknown_rule_is_refused(rule):
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    with pytest.raises(ValueError, match="corroboration_rule must be one of"):
        _evaluate(freqs, mags, rule=rule)
    with pytest.raises(ValueError, match="corroboration_rule must be one of"):
        lg_snapshot.policy_params(lg_snapshot.SnapshotPolicy(corroboration_rule=rule))


def test_the_rule_has_no_tunable_and_no_process_state():
    names = set(vars(be))
    assert not names & {
        "_CORROBORATION",
        "_CORROBORATION_MODES",
        "_set_corroboration_mode",
        "_combined_p_value",
        "_candidate_decision",
    }
    parameters = inspect.signature(be.evaluate_hypotheses).parameters
    assert not {"weight", "scale", "method"} & set(parameters)
    assert set(inspect.signature(be._controlled_union_decision).parameters) == {
        "members",
        "eligible",
    }


def test_rules_do_not_leak_between_calls_or_threads():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    released = _evaluate(freqs, mags)
    union = _evaluate(freqs, mags, rule=UNION)
    assert released != union
    assert _evaluate(freqs, mags) == released

    results: dict[tuple[str, int], dict] = {}

    def run(rule, index):
        results[(rule, index)] = _evaluate(freqs, mags, rule=rule)

    threads = [
        threading.Thread(target=run, args=(rule, index))
        for index in range(6)
        for rule in (RELEASED, UNION)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for (rule, _), result in results.items():
        assert result == (released if rule == RELEASED else union)


# --- default mode -----------------------------------------------------------------


@pytest.mark.parametrize("seed, lines", [MODERATE_HARMONICS, LONE_HARMONIC])
def test_default_result_carries_no_new_key(seed, lines):
    freqs, mags = _spectrum(seed, lines)
    result = _evaluate(freqs, mags)
    for check in result["checks"]:
        assert "aggregate_corroboration" not in check
        assert all(
            entry["excluded_by"] in (None, "shaft_order", "shared")
            for entry in check["corroboration"]
        )
    assert "corroboration_rule" not in result["evidence_policy"]
    assert result["evidence_policy"]["version"] == be.EVIDENCE_POLICY["version"]
    assert result["evidence_policy"] == be.evidence_policy(5.0, 3)
    assert _summary(result)["claim"]["policy_version"] == bc.CLAIM_POLICY_VERSION
    assert "experimental" not in result["outcome_reason"]
    assert "experimental" not in _summary(result)["overall_assessment"]


def test_rule_leaves_the_tests_of_each_line_untouched():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    base = _evaluate(freqs, mags)
    union = _evaluate(freqs, mags, rule=UNION)
    for b, c in zip(base["checks"], union["checks"]):
        keys = ("detected", "level_db", "threshold_db", "line_test")
        assert [b[k] for k in keys] == [c[k] for k in keys]
        ignored = ("counted", "excluded_by")
        assert [
            {k: v for k, v in e.items() if k not in ignored} for e in b["corroboration"]
        ] == [
            {k: v for k, v in e.items() if k not in ignored} for e in c["corroboration"]
        ]
        if b["state"] != c["state"]:
            assert {b["state_reason"], c["state_reason"]} <= {
                "no_corroboration",
                "corroborated",
            }


# --- decisions, end to end --------------------------------------------------------


def test_moderate_harmonics_are_supported_only_as_an_aggregate_decision():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    base = _evaluate(freqs, mags)["checks"][0]
    assert (base["state"], base["state_reason"]) == ("unconfirmed", "no_corroboration")

    result = _evaluate(freqs, mags, rule=UNION)
    check = result["checks"][0]
    assert check["state"] == "supported"
    assert check["aggregate_corroboration"]["basis"] == "combined"
    assert not any(e["counted"] for e in check["corroboration"])
    assert check["harmonics_detected"] == []
    assert check["evidence_strength"] == "low"
    assert "tested together, reach the combined level" in result["outcome_reason"]
    assert "0 further lines" not in result["outcome_reason"]
    _assert_per_line_evidence_is_truthful(result)


@pytest.mark.parametrize(
    "basis, counted, expected",
    [
        ("combined", 0, "low"),
        ("combined", 1, "moderate"),
        ("combined", 2, "high"),
        ("single_line", 1, "moderate"),
        ("single_line", 2, "high"),
        (None, 1, "moderate"),
        (None, 2, "high"),
    ],
)
def test_strength_of_a_supported_check(basis, counted, expected):
    """Low only for support by the combined test with no line counted."""
    line = {
        "significant": True,
        "peak_hz": 107.4,
        "expected_hz": 107.36,
        "amplitude": 9.0,
    }
    members = [{"kind": "sideband"}] * 2
    check = {
        "state": "supported",
        "corroboration": [{"counted": k < counted} for k in range(2)],
    }
    if basis is not None:
        check["aggregate_corroboration"] = {"basis": basis}
    assert be._derived_check_fields(line, check, members)["evidence_strength"] == (
        expected
    )


def test_released_rule_grades_supported_checks_as_before():
    freqs, mags = _spectrum(*LONE_HARMONIC)
    check = _evaluate(freqs, mags)["checks"][0]
    assert check["state"] == "supported"
    assert check["evidence_strength"] == "moderate"


def test_no_pooled_line_of_a_combined_only_check_is_significant():
    """What the sentences state about those lines, for every result reached."""
    reached = 0
    for seed in range(5, 45):
        freqs, mags = _spectrum(seed, MODERATE_HARMONICS[1])
        result = _evaluate(freqs, mags, rule=UNION)
        for check in result["checks"]:
            if check["state"] == "supported" and be._combined_only(check):
                reached += 1
                pooled = check["aggregate_corroboration"]["pooled_members"]
                assert not any(check["corroboration"][k]["significant"] for k in pooled)
                assert check["evidence_strength"] == "low"
                assert "none significant in its own line test" in (
                    result["outcome_reason"]
                )
    assert reached


def test_prose_names_the_rule_by_its_identifier():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    evidence = _evaluate(freqs, mags, rule=UNION)
    text = f"{evidence['outcome_reason']} {_summary(evidence)['overall_assessment']}"
    assert "controlled_union_v1" in text
    assert "controlled-union" not in text


def test_the_note_limits_what_the_evaluation_establishes():
    assert "so that evaluation establishes nothing about real machines" in (
        bc._EXPERIMENTAL_RULE_NOTE
    )


def test_strength_description_states_both_grades():
    description = BearingFaultCheckResult.model_fields["evidence_strength"].description
    assert (
        "'low' (unconfirmed or shared, or, under the experimental controlled_union_v1 "
        "rule only, supported by that rule's combined test with no line counted"
    ) in description
    assert "'moderate' (supported, one counted corroborating line)" in description


def test_a_significant_line_the_rule_rejects_names_why():
    """A significant, unexcluded, uncounted line would leave excluded_by empty."""
    freqs, mags = _spectrum(*LONE_HARMONIC)
    assert _evaluate(freqs, mags)["checks"][0]["state"] == "supported"

    result = _evaluate(freqs, mags, rule=UNION)
    check = result["checks"][0]
    assert check["state_reason"] == "no_corroboration"
    lone = check["corroboration"][0]
    assert (lone["significant"], lone["counted"], lone["excluded_by"]) == (
        True,
        False,
        be.EXCLUDED_BY_AGGREGATE_RULE,
    )
    assert "shaft order or where another tested line" not in result["outcome_reason"]
    assert "does not accept" in result["outcome_reason"]
    _assert_per_line_evidence_is_truthful(result)
    assert BearingCorroboratingLine(**lone).excluded_by == "aggregate_rule"


def test_a_strong_single_line_is_counted():
    freqs, mags = _spectrum(11, {107.4: 9.0, 214.8: 9.0})
    check = _evaluate(freqs, mags, rule=UNION)["checks"][0]
    assert check["state"] == "supported"
    assert check["aggregate_corroboration"]["basis"] == "single_line"
    assert check["corroboration"][0]["counted"]
    assert check["corroboration"][0]["excluded_by"] is None
    assert check["harmonics_detected"][0]["harmonic"] == 2


def test_a_harmonic_at_a_shaft_order_is_not_pooled():
    # 214.8 Hz is 7 x the shaft rate: the strong 2x line is a shaft order.
    shaft_hz = 214.8 / 7
    freqs, mags = _spectrum(21, {107.4: 9.0, 214.8: 12.0})
    base = _evaluate(freqs, mags, shaft_hz=shaft_hz)["checks"][0]
    assert base["corroboration"][0]["excluded_by"] == "shaft_order"

    result = _evaluate(freqs, mags, rule=UNION, shaft_hz=shaft_hz)
    check = result["checks"][0]
    assert check["corroboration"][0]["excluded_by"] == "shaft_order"
    assert not check["corroboration"][0]["counted"]
    # 2x and 4x (429.6 Hz) are both excluded; only 3x stays eligible.
    assert check["aggregate_corroboration"]["eligible_lines"] == 1
    assert (check["state"], check["state_reason"]) == (
        "unconfirmed",
        "no_corroboration",
    )
    _assert_per_line_evidence_is_truthful(result)


def test_a_harmonic_shared_with_another_check_is_not_pooled():
    # A second declared line at 214.8 Hz claims the outer race's 2x window.
    expected = [("BPFO", 107.36), ("GMF", 214.8)]
    freqs, mags = _spectrum(23, {107.4: 9.0, 214.8: 12.0})
    base = _evaluate(freqs, mags, expected=expected)["checks"][0]
    assert base["corroboration"][0]["excluded_by"] == "shared"

    result = _evaluate(freqs, mags, rule=UNION, expected=expected)
    check = result["checks"][0]
    assert check["corroboration"][0]["excluded_by"] == "shared"
    assert not check["corroboration"][0]["counted"]
    assert check["aggregate_corroboration"]["eligible_lines"] == 1
    _assert_per_line_evidence_is_truthful(result)


def test_only_a_significant_line_has_a_record():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    result = _evaluate(freqs, mags, rule=UNION)
    for check in result["checks"]:
        assert ("aggregate_corroboration" in check) == bool(check["corroboration"])


# --- p-values, budgets and pool membership ----------------------------------------


def _member(ratio, search_bins=5, reference_bins=80, reference_rank=40):
    return {
        "ratio": ratio,
        "search_bins": search_bins,
        "reference_bins": reference_bins,
        "reference_rank": reference_rank,
    }


def test_window_p_value_is_exact_and_stays_inside_the_unit_interval():
    per_bin = be.os_cfar_false_alarm_probability(20.0, 80, 40)
    assert be._window_p_value(_member(20.0)) == pytest.approx(1 - (1 - per_bin) ** 5)
    assert be._window_p_value(_member(20.0)) <= be._single_line_p_value(_member(20.0))
    assert 0.0 < be._window_p_value(_member(0.0)) < 1.0


def test_combination_is_fisher():
    p = [0.02, 0.03, 0.5]
    statistic = -2 * sum(math.log(x) for x in p)
    assert be._fisher_p_value(p) == pytest.approx(stats.chi2.sf(statistic, 6))


def _line(window_p, single_p=1.0, local_maximum=True, kind="harmonic"):
    return {
        "window_p": window_p,
        "single_p": single_p,
        "local_maximum": local_maximum,
        "kind": kind,
    }


@pytest.fixture
def given_p_values(monkeypatch):
    monkeypatch.setattr(be, "_window_p_value", lambda m: m["window_p"])
    monkeypatch.setattr(be, "_single_line_p_value", lambda m: m["single_p"])


def test_recorded_budget_and_levels():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    rule = _evaluate(freqs, mags, rule=UNION)["checks"][0]["aggregate_corroboration"]
    lines = rule["eligible_lines"]
    assert lines == 3
    budget = 1.0 - 0.99**lines
    assert rule["rule"] == UNION
    assert rule["budget"] == pytest.approx(budget)
    assert rule["combined_level"] == pytest.approx(0.5 * budget)
    assert rule["single_line_level"] == pytest.approx(0.5 * budget / lines)
    assert rule["pooled_lines"] == len(rule["pooled_members"])


def test_only_eligible_local_maxima_are_pooled(given_p_values):
    members = [
        _line(0.01),
        _line(1e-12, local_maximum=False),
        _line(0.02, kind="sideband"),
        _line(1e-12),
    ]
    rule = be._controlled_union_decision(members, [0, 1, 2])
    assert rule["eligible_lines"] == 3
    assert rule["pooled_lines"] == 2
    assert rule["pooled_members"] == [0, 2]
    assert rule["pooled_kinds"] == ["harmonic", "sideband"]
    assert rule["combined_p_value"] == pytest.approx(
        stats.chi2.sf(-2 * (math.log(0.01) + math.log(0.02)), 4)
    )
    assert rule["budget"] == pytest.approx(1.0 - 0.99**3)


def test_no_eligible_line_and_no_local_maximum_are_not_corroborated(given_p_values):
    empty = be._controlled_union_decision([_line(1e-12)], [])
    assert (empty["budget"], empty["corroborated"], empty["basis"]) == (
        None,
        False,
        None,
    )
    assert empty["pooled_lines"] == 0
    flat = be._controlled_union_decision(
        [_line(1e-12, 1e-12, local_maximum=False)], [0]
    )
    assert flat["budget"] == pytest.approx(0.01)
    assert (flat["combined_p_value"], flat["corroborated"]) == (None, False)
    assert flat["single_lines"] == 0


def test_the_budget_is_split_between_the_two_tests(given_p_values, monkeypatch):
    budget = be._family_level(3)
    single_level = 0.5 * budget / 3
    weak = [_line(0.5, 0.5), _line(0.5, 0.5)]

    monkeypatch.setattr(be, "_fisher_p_value", lambda p: 0.5)
    at_level = be._controlled_union_decision(
        [*weak, _line(0.5, single_level)], [0, 1, 2]
    )
    assert (at_level["corroborated"], at_level["basis"]) == (True, "single_line")
    over = be._controlled_union_decision(
        [*weak, _line(0.5, single_level * 1.01)], [0, 1, 2]
    )
    assert over["corroborated"] is False

    combined_level = 0.5 * budget
    monkeypatch.setattr(be, "_fisher_p_value", lambda p: combined_level)
    joint = be._controlled_union_decision([*weak, _line(0.5, 0.5)], [0, 1, 2])
    assert (joint["corroborated"], joint["basis"]) == (True, "combined")
    assert joint["combined_level"] == pytest.approx(combined_level)
    monkeypatch.setattr(be, "_fisher_p_value", lambda p: combined_level * 1.0001)
    assert (
        be._controlled_union_decision([*weak, _line(0.5, 0.5)], [0, 1, 2])[
            "corroborated"
        ]
        is False
    )


def test_a_single_line_takes_precedence_over_the_combined_basis(
    given_p_values, monkeypatch
):
    monkeypatch.setattr(be, "_fisher_p_value", lambda p: 0.0)
    rule = be._controlled_union_decision([_line(0.5, 0.0), _line(0.5, 0.5)], [0, 1])
    assert rule["basis"] == "single_line"
    assert rule["single_lines"] == 1


# --- identifiers and the public record --------------------------------------------


def test_result_of_the_rule_carries_its_own_identifiers():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    evidence = _evaluate(freqs, mags, rule=UNION)
    policy = evidence["evidence_policy"]
    assert policy["corroboration_rule"] == UNION
    assert policy["version"] == be.EVIDENCE_POLICY["version"] + 1
    assert any("Fisher" in text for text in policy["assumptions"])
    assert any("synthetic signals only" in text for text in policy["exclusions"])
    assert BearingEvidencePolicy(**policy).corroboration_rule == UNION
    summary = _summary(evidence)
    assert (
        summary["claim"]["policy_version"] == bc.CLAIM_POLICY_VERSION_CONTROLLED_UNION
    )
    assert "controlled_union_v1" in summary["overall_assessment"]


def test_released_policy_constant_is_not_mutated_by_the_rule():
    before = copy.deepcopy(be.EVIDENCE_POLICY)
    be.evidence_policy(5.0, 3, UNION)
    assert be.EVIDENCE_POLICY == before


def test_public_model_carries_the_aggregate_record():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    evidence = _evaluate(freqs, mags, rule=UNION)
    fields = {
        "signal_id": "s",
        "bearing_id": None,
        "fault_type_canonical": "outer_race",
        "expected_frequency_hz": 107.36,
        **evidence["checks"][0],
    }
    model = BearingFaultCheckResult(**fields)
    assert model.aggregate_corroboration is not None
    assert model.aggregate_corroboration.basis == "combined"
    assert model.model_dump()["aggregate_corroboration"]["rule"] == UNION


def test_analyzer_routes_pass_the_rule():
    rng = np.random.default_rng(7)
    fs = 12000.0
    signal = rng.normal(size=int(fs * 2))
    for result in (
        ba.check_all_bearing_faults(
            signal, fs, "6205", 1797.0, corroboration_rule=UNION
        ),
        ba.check_frequency_set(
            signal, fs, {"BPFO": 107.36}, 1797.0, corroboration_rule=UNION
        ),
    ):
        assert result["evidence_policy"]["corroboration_rule"] == UNION
    with pytest.raises(ValueError, match="corroboration_rule must be one of"):
        ba.check_all_bearing_faults(signal, fs, "6205", 1797.0, corroboration_rule="x")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        {"bearing_id": "6205"},
        {"frequencies": {"BPFO": 107.36, "BPFI": 162.19}},
        {"num_balls": 9, "ball_diameter_mm": 7.94, "pitch_diameter_mm": 39.04},
    ],
)
async def test_tool_routes_select_the_rule_per_call(route, monkeypatch):
    fs = 12000.0
    signal = np.random.default_rng(9).normal(size=int(fs * 2))
    info = SimpleNamespace(sampling_rate=fs, source_metadata={})
    monkeypatch.setattr(
        diagnostics_tools, "resolve_signal", lambda signal_id: (signal, info)
    )
    tool = diagnostics_tools.check_bearing_faults

    union = await tool(
        ctx=None, signal_id="s", rpm=1797.0, corroboration_rule=UNION, **route
    )
    released = await tool(ctx=None, signal_id="s", rpm=1797.0, **route)

    assert union.evidence_policy.corroboration_rule == UNION
    assert union.claim.policy_version == bc.CLAIM_POLICY_VERSION_CONTROLLED_UNION
    assert released.evidence_policy.corroboration_rule is None
    assert released.claim.policy_version == bc.CLAIM_POLICY_VERSION
    assert all(c.aggregate_corroboration is None for c in released.fault_checks)


def test_advisory_row_says_when_an_aggregate_rule_supports_the_line():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)

    def rows(rule):
        evidence = _evaluate(freqs, mags, rule=rule)
        checks = [
            {
                **c,
                "expected_frequency_hz": h.line_hz,
                "fault_type_canonical": h.canonical,
            }
            for c, h in zip(evidence["checks"], _hypotheses(EXPECTED))
        ]
        summary = _summary(evidence) | {
            "fault_checks": checks,
            "bearing_id": "6205",
            "shaft_frequency_hz": 29.95,
        }
        return advisory._build_bearing_block(summary, None, "6205")["rows"]

    assert all("aggregate_basis" not in row for row in rows(RELEASED))
    outer = rows(UNION)[0]
    assert (outer["state"], outer["counted_lines"]) == ("supported", 0)
    assert outer["aggregate_basis"] == "combined"


# --- wording of an aggregate decision ---------------------------------------------


def _hypothesis(label, canonical):
    return be.Hypothesis(label, 107.36, canonical)


def _combined_check(pooled, peaks, kinds):
    return {
        "line_peak_hz": 107.4,
        "level_db": 20.0,
        "corroboration": [
            {
                "kind": kind,
                "harmonic": 2 + k if kind == "harmonic" else None,
                "peak_hz": hz,
                "level_db": 8.0,
                "counted": False,
            }
            for k, (hz, kind) in enumerate(zip(peaks, kinds))
        ],
        "aggregate_corroboration": {
            "basis": "combined",
            "pooled_lines": len(pooled),
            "pooled_members": pooled,
            "pooled_kinds": sorted({kinds[k] for k in pooled}),
        },
    }


def test_further_count_names_only_the_pooled_lines():
    outer = _hypothesis("BPFO", ba.FAULT_TYPE_CANONICAL["BPFO"])
    check = _combined_check([0, 2], [214.8, 322.2, 429.6], ["harmonic"] * 3)
    assert be._further_count(outer, check) == (
        "2 further lines its kinematics predict (at 214.80 and 429.60 Hz), none "
        "significant in its own line test, tested together, reach the combined level "
        "of the experimental controlled_union_v1 corroboration rule"
    )
    single = _combined_check([1], [214.8, 322.2, 429.6], ["harmonic"] * 3)
    assert be._further_count(outer, single) == (
        "1 further line its kinematics predict (at 322.20 Hz), not significant in "
        "its own line test, reaches the combined level of the experimental "
        "controlled_union_v1 corroboration rule"
    )
    label = _hypothesis("GMF", None)
    assert be._further_count(label, check).startswith("2 of its harmonics (at 214.80")


def test_released_count_wording_is_kept_when_a_line_is_counted():
    outer = _hypothesis("BPFO", ba.FAULT_TYPE_CANONICAL["BPFO"])
    check = _combined_check([0, 1], [214.8, 322.2], ["harmonic"] * 2)
    check["corroboration"][0]["counted"] = True
    assert be._further_count(outer, check) == (
        "1 further line its kinematics predict also stands"
    )


def test_outcome_does_not_say_pooled_lines_stand_above_the_floor():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    reason = _evaluate(freqs, mags, rule=UNION)["outcome_reason"]
    assert "reach the combined level" in reason
    assert "also stand above the floor" not in reason
    assert "above the floor in the same spectrum" not in reason


def test_counted_lines_sentence_for_an_aggregate_decision():
    freqs, mags = _spectrum(*MODERATE_HARMONICS)
    evidence = _evaluate(freqs, mags, rule=UNION)
    check = evidence["checks"][0]
    hz = check["corroboration"][0]["peak_hz"]
    sentence = bc.counted_lines_statements(_summary(evidence))[0]
    assert "No further line is counted on its own" in sentence
    assert f"harmonic 2 at {hz:.2f} Hz" in sentence
    assert "none significant in its own line test, tested together reach" in sentence
    assert "accepted" not in sentence


def test_several_types_sentence_does_not_claim_a_line_of_its_own():
    expected = [("BPFO", 107.36), ("GMF", 350.0)]
    freqs, mags = _spectrum(*SEVERAL)
    evidence = _evaluate(freqs, mags, rule=UNION, expected=expected)
    assert evidence["outcome"] == "several_types"
    assert be._combined_only(evidence["checks"][0])
    assert "each corroborated by a line of its own" not in evidence["outcome_reason"]
    assert (
        "by further lines that reach its combined level, none of them significant in "
        "its own line test"
    ) in evidence["outcome_reason"]
    block = bc.line_test_statement(
        _summary(evidence, expected=expected)
        | {
            "shaft_frequency_hz": 29.95,
            "fault_checks": [
                {**c, "fault_type_canonical": h.canonical}
                for c, h in zip(evidence["checks"], _hypotheses(expected))
            ],
        }
    )
    assert "tested together, reach the combined level of the experimental" in block


def test_released_several_types_sentence_is_unchanged():
    expected = [("BPFO", 107.36), ("GMF", 350.0)]
    freqs, mags = _spectrum(
        5, {107.4: 9.0, 214.8: 9.0, 322.2: 9.0, 350.0: 9.0, 700.0: 9.0}
    )
    evidence = _evaluate(freqs, mags, expected=expected)
    assert evidence["outcome"] == "several_types"
    assert "each corroborated by a line of its own in the same spectrum" in (
        evidence["outcome_reason"]
    )


# --- the inner-race sideband limitation -------------------------------------------


def test_combined_only_inner_race_keeps_the_sideband_limitation():
    expected = [("BPFI", 162.19), ("FTF", 11.93)]
    lines = {162.2: 9.0, 132.2: 2.6, 192.1: 2.6}
    for seed in range(200):
        freqs, mags = _spectrum(seed, lines)
        evidence = be.evaluate_hypotheses(
            freqs,
            mags,
            _hypotheses(expected),
            shaft_hz=29.95,
            cage_hz=11.93,
            envelope_band_hz=300.0,
            corroboration_rule=UNION,
        )
        if evidence["outcome"] == "indicated" and be._combined_only(
            evidence["checks"][0]
        ):
            break
    else:
        pytest.fail("no combined-only inner-race support in 200 seeds")
    record = evidence["checks"][0]["aggregate_corroboration"]
    assert record["pooled_kinds"] == ["sideband"]
    summary = _summary(evidence, expected=expected)
    assert bc.SIDEBAND_ONLY_CORROBORATION in summary["claim"]["limitations"]
    assert "include shaft-rate sidebands" in summary["overall_assessment"]
    assert "include shaft-rate sidebands" in bc.counted_lines_statements(summary)[0]


@pytest.mark.parametrize(
    "canonical, kinds, counted, expected",
    [
        (bc._INNER_RACE, ["sideband", "sideband"], [], True),
        (bc._INNER_RACE, ["harmonic", "sideband"], [], True),
        (bc._INNER_RACE, ["harmonic", "harmonic"], [], False),
        (ba.FAULT_TYPE_CANONICAL["BPFO"], ["sideband", "sideband"], [], False),
        (bc._INNER_RACE, ["harmonic", "sideband"], [0], False),
        (bc._INNER_RACE, ["harmonic", "sideband"], [1], True),
    ],
)
def test_sideband_only_reads_the_pool_only_for_a_combined_only_decision(
    canonical, kinds, counted, expected
):
    check = _combined_check([0, 1], [132.2, 192.1], kinds)
    for k in counted:
        check["corroboration"][k]["counted"] = True
    assert bc._sideband_only(canonical, check) is expected


def test_single_sideband_pool_limitation_is_singular():
    check = _combined_check([1], [132.2, 192.1], ["harmonic", "sideband"])
    assert "The further line tested is a shaft-rate sideband" in (
        bc._combined_sideband_limitation(check)
    )
    pair = _combined_check([0, 1], [132.2, 192.1], ["sideband", "sideband"])
    assert "tested together include shaft-rate sidebands" in (
        bc._combined_sideband_limitation(pair)
    )


# --- asset ledger lineage ---------------------------------------------------------


def test_default_policy_keeps_the_recorded_processing_id():
    """The hash of the parameters the released rule has always had."""
    params = lg_snapshot.policy_params(lg_snapshot.SnapshotPolicy())
    assert "corroboration_rule" not in params
    assert lg_snapshot.processing_id() == (
        f"health_snapshot/{lg_snapshot.ALGORITHM_VERSION}+"
        f"{lg_snapshot._short_digest(params)}"
    )


def test_the_rule_forms_its_own_lineage():
    released = lg_snapshot.SnapshotPolicy()
    union = lg_snapshot.SnapshotPolicy(corroboration_rule=UNION)
    assert lg_snapshot.policy_params(union)["corroboration_rule"] == UNION
    assert lg_snapshot.processing_id(union) != lg_snapshot.processing_id(released)


def _snapshot(processing_id, rule=None):
    params = {} if rule is None else {"corroboration_rule": rule}
    return {"processing": {"processing_id": processing_id, "params": params}}


def _slot(*snapshots):
    return {
        "record": {
            "lineage_positions": {},
            "snapshots": list(snapshots),
            "snapshots_by_lineage": {
                s["processing"]["processing_id"]: s for s in snapshots
            },
        }
    }


def test_released_assessment_never_selects_an_experimental_lineage():
    released, union = _snapshot("a"), _snapshot("b", UNION)
    # The experimental lineage is the most recent one on every slot.
    evaluated = [_slot(released, union), _slot(released, union)]
    assert lg_assessment._select_lineage(evaluated)["processing_id"] == "a"
    assert lg_assessment._select_lineage(evaluated, UNION)["processing_id"] == "b"


def test_lineages_of_different_rules_are_never_mixed():
    released, union = _snapshot("a"), _snapshot("b", UNION)
    evaluated = [_slot(released), _slot(union)]
    assert lg_assessment._select_lineage(evaluated)["processing_id"] is None
    assert lg_assessment._select_lineage(evaluated, UNION)["processing_id"] is None
    only_union = [_slot(union), _slot(union)]
    selected = lg_assessment._select_lineage(only_union)
    assert selected["processing_id"] is None
    assert selected["counts"] == {}
    assert selected["other_rule_lineages"] == ["b"]


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        {"processing": None},
        {"processing": {"params": None}},
        {"processing": {"params": "x"}},
        {"processing": {"params": {"corroboration_rule": 3}}},
    ],
)
def test_a_snapshot_that_names_no_rule_is_of_the_released_rule(snapshot):
    assert lg_assessment._snapshot_rule(snapshot) == RELEASED


_DECLARATION = {
    "asset_id": "A",
    "measurement_point_id": "P",
    "acquired_at": "2026-01-01T00:00:00+00:00",
    "timezone_declared": True,
    "timestamp_suspect": False,
    "rpm": 1797.0,
    "load": None,
    "operating_state": None,
    "sensor_id": "S",
    "direction": "horizontal",
    "declared_by": "test",
    "sampling_rate": 12000.0,
    "signal_unit": "g",
    "raw_format": None,
    "channel_index": 0,
}
_POINT = {
    "measurement_point_id": "P",
    "declaration_version": 1,
    "bearing_id": "6205",
    "fault_orders": None,
    "machine_group": 2,
    "support_type": "rigid",
    "machine_power_kw": None,
    "expected_signal_unit": "g",
    "expected_sensor_id": None,
    "expected_direction": "horizontal",
    "nominal_rpm": 1797.0,
}
_ORDERS = {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357, "FTF": 0.398}


def _noise_record():
    return np.random.default_rng(3).normal(size=int(12000.0 * 2))


def test_snapshot_records_the_rule_and_default_snapshot_does_not():
    default = lg_snapshot.compute_health_snapshot(
        _noise_record(), 12000.0, declaration=_DECLARATION, point=_POINT
    )
    union = lg_snapshot.compute_health_snapshot(
        _noise_record(),
        12000.0,
        declaration=_DECLARATION,
        point=_POINT,
        policy=lg_snapshot.SnapshotPolicy(corroboration_rule=UNION),
    )
    assert "corroboration_rule" not in default["processing"]["params"]
    assert union["processing"]["params"]["corroboration_rule"] == UNION
    assert (
        union["processing"]["processing_id"] != default["processing"]["processing_id"]
    )
    assert lg_assessment._snapshot_rule(default) == RELEASED
    assert lg_assessment._snapshot_rule(union) == UNION


@pytest.mark.parametrize("rule", [RELEASED, UNION])
@pytest.mark.parametrize(
    "target, point",
    [
        ("check_all_bearing_faults", _POINT),
        (
            "check_frequency_set",
            {**_POINT, "bearing_id": None, "fault_orders": _ORDERS},
        ),
    ],
)
def test_snapshot_bearing_block_runs_under_the_rule_of_its_policy(
    target, point, rule, monkeypatch
):
    seen = []
    original = getattr(lg_snapshot, target)

    def spy(*args, **kwargs):
        seen.append(kwargs["corroboration_rule"])
        return original(*args, **kwargs)

    monkeypatch.setattr(lg_snapshot, target, spy)
    lg_snapshot.compute_health_snapshot(
        _noise_record(),
        12000.0,
        declaration=_DECLARATION,
        point=point,
        policy=lg_snapshot.SnapshotPolicy(corroboration_rule=rule),
    )
    assert seen == [rule]


# --- regression pin against the frozen confirmed candidate ------------------------

#: What the confirmed candidate (cls-ae3-sens-c3.patch, union, weight 0.5, scale 1.0)
#: decides on these spectra. A change to any of these is a change of the rule, which
#: needs a new rule name, not an edit of these values.
_FROZEN_DECISIONS = {
    "moderate_harmonics": (
        MODERATE_HARMONICS,
        "supported",
        "corroborated",
        [False, False, False],
        {
            "combined_p_value": 0.006134006461988753,
            "single_lines": 0,
            "corroborated": True,
            "basis": "combined",
        },
    ),
    "lone_harmonic": (
        LONE_HARMONIC,
        "unconfirmed",
        "no_corroboration",
        [False, False, False],
        {
            "combined_p_value": 0.06655189538544651,
            "single_lines": 0,
            "corroborated": False,
            "basis": None,
        },
    ),
    "strong_single": (
        (11, {107.4: 9.0, 214.8: 9.0}),
        "supported",
        "corroborated",
        [True, False, False],
        {
            "combined_p_value": 1.002566332783578e-29,
            "single_lines": 1,
            "corroborated": True,
            "basis": "single_line",
        },
    ),
}


@pytest.mark.parametrize("name", list(_FROZEN_DECISIONS))
def test_decisions_equal_the_frozen_confirmed_candidate(name):
    case, state, reason, counted, record = _FROZEN_DECISIONS[name]
    freqs, mags = _spectrum(*case)
    check = _evaluate(freqs, mags, rule=UNION)["checks"][0]
    decision = check["aggregate_corroboration"]
    assert (check["state"], check["state_reason"]) == (state, reason)
    assert [e["counted"] for e in check["corroboration"]] == counted
    assert decision["eligible_lines"] == decision["pooled_lines"] == 3
    assert decision["budget"] == pytest.approx(0.029700999999999977, rel=1e-12)
    assert decision["combined_level"] == pytest.approx(0.014850499999999989, rel=1e-12)
    assert decision["single_line_level"] == pytest.approx(
        0.004950166666666663, rel=1e-12
    )
    assert decision["combined_p_value"] == pytest.approx(
        record["combined_p_value"], rel=1e-9
    )
    for key in ("single_lines", "corroborated", "basis"):
        assert decision[key] == record[key]


def test_every_statement_of_the_experimental_rule_says_so():
    freqs, mags = _spectrum(11, {107.4: 9.0, 214.8: 9.0})

    def statements(rule):
        evidence = _evaluate(freqs, mags, rule=rule)
        summary = _summary(evidence) | {
            "shaft_frequency_hz": 29.95,
            "fault_checks": [
                {**c, "fault_type_canonical": h.canonical}
                for c, h in zip(evidence["checks"], _hypotheses(EXPECTED))
            ],
        }
        return (
            bc.line_test_statement(summary),
            bc.counted_lines_statements(summary),
            summary["overall_assessment"],
        )

    line_test, counted, overall = statements(UNION)
    assert "experimental controlled_union_v1" in line_test
    assert "experimental controlled_union_v1" in counted[-1]
    assert "not comparable" in overall
    line_test, counted, overall = statements(RELEASED)
    assert "experimental" not in line_test + " ".join(counted) + overall
