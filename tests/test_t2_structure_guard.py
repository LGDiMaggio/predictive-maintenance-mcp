"""The evidence-strength rule and the non-bearing wording stay with their owners.

Two functions interpret one acquisition: the synthesis behind
``diagnose_vibration`` (``diagnosis_pipeline._synthesize_diagnosis``) and the
advisory behind ``generate_diagnostic_report``. The advisory owns the
evidence-strength grade (``score_evidence_strength``), and every non-bearing
sentence both write comes from ``decision_support.statements``. These guards
keep it so:

- no pipeline or advisory function holds a sentence of its own; what a stage
  that could not run writes (the ``reason`` and ``remedy`` it passes on, log
  messages, raised errors) is left out, and one helper that assembles a
  remedy in a local variable is exempt by name;
- the pipeline holds no evidence arithmetic and names no grade, and its
  synthesis takes the grade from the advisory;
- no other module of the diagnosis tools' file set names a grade;
- ``score_evidence_strength`` grades each branch of its rule as the table
  below records.

Each guard is a pure function from source text to its violations, so the red
paths below run it on small sources and assert the exact list.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Optional

import pytest

from predictive_maintenance_mcp.decision_support import advisory, diagnosis_pipeline
from predictive_maintenance_mcp.decision_support.advisory import score_evidence_strength
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    _synthesize_diagnosis,
)
from predictive_maintenance_mcp.decision_support.statements import STATEMENTS

SRC = Path(diagnosis_pipeline.__file__).resolve().parent.parent

#: Pipeline functions exempt as a whole: the remedy of a bearing check that
#: could not run assembles its call example in a local variable before it
#: becomes the ``remedy`` argument. Every other stage outcome is read where it
#: is written, as the ``reason`` or ``remedy`` argument of a call, a log
#: message or a raised error, which the scan leaves out in every function.
STAGE_OUTCOME_FUNCTIONS = frozenset({"_not_in_catalog_check"})

#: Keyword arguments that carry the reason or remedy of a stage that could not
#: run, and the logger whose messages are not output.
STAGE_OUTCOME_KEYWORDS = frozenset({"reason", "remedy"})
LOGGER_NAMES = frozenset({"logger", "logging"})

#: Literals with words that are not statements: the unit of the anomaly-share
#: delta in a baseline comparison.
UNIT_LABELS = frozenset({"percentage points"})

#: The grades a points ladder writes. "none" also names other things
#: (an absent source, a file-access level), so it is not a marker.
GRADE_WORDS = frozenset({"weak", "moderate", "strong"})

#: Modules of the diagnosis tools' file set, other than the advisory.
OTHER_T2_MODULES = (
    "decision_support/alerts.py",
    "decision_support/anomaly_labels.py",
    "decision_support/diagnosis_pipeline.py",
    "decision_support/recommendations.py",
    "decision_support/statements.py",
    "mcp_tools/diagnostics_tools.py",
    "mcp_tools/report_tools.py",
    "mcp_tools/decision_support_tools.py",
    "mcp_tools/prompts.py",
    "integrated_report.py",
    "figures.py",
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                nodes.add(id(body[0].value))
    return nodes


def _is_log_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in LOGGER_NAMES
    )


def _strings(
    node: ast.AST, function: Optional[str], skip: frozenset
) -> Iterator[tuple[ast.Constant, Optional[str]]]:
    """Every string constant under *node* with its enclosing top-level
    function, leaving out raised messages, log messages, ``reason`` and
    ``remedy`` keyword arguments and the functions in *skip*."""
    if isinstance(node, ast.Raise) or _is_log_call(node):
        return
    if isinstance(node, ast.keyword) and node.arg in STAGE_OUTCOME_KEYWORDS:
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if function is None:
            if node.name in skip:
                return
            function = node.name
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node, function
    for child in ast.iter_child_nodes(node):
        yield from _strings(child, function, skip)


def inline_sentence_violations(
    source: str,
    label: str,
    exempt: frozenset = frozenset(),
    allowed: frozenset = frozenset(),
) -> list[str]:
    """String constants that read as words (a letter and a space) outside
    docstrings, raised and logged messages, ``reason`` and ``remedy``
    arguments, the *exempt* top-level functions and the *allowed* literals."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    violations = []
    for node, function in _strings(tree, None, exempt):
        if id(node) in docstrings or node.value in allowed:
            continue
        text = node.value
        if " " in text and any(c.isalpha() for c in text):
            violations.append(
                f"{label}:{node.lineno} {function or '<module>'} holds an inline "
                f"sentence: {text[:40]!r}"
            )
    return violations


def evidence_rule_violations(source: str, label: str) -> list[str]:
    """Evidence arithmetic (a numeric constant added in place) and grade words
    outside docstrings."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) in (int, float)
        ):
            violations.append(
                f"{label}:{node.lineno} adds a constant to {ast.unparse(node.target)}"
            )
        if (
            isinstance(node, ast.Constant)
            and node.value in GRADE_WORDS
            and id(node) not in docstrings
        ):
            violations.append(f"{label}:{node.lineno} names the grade {node.value!r}")
    return violations


def scoring_source_violations(source: str) -> list[str]:
    """The synthesis must call the advisory's ``score_evidence_strength``."""
    tree = ast.parse(source)
    violations = []
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "advisory"
        and any(alias.name == "score_evidence_strength" for alias in node.names)
        for node in tree.body
    )
    if not imported:
        violations.append("score_evidence_strength is not imported from the advisory")
    synthesis = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_synthesize_diagnosis"
        ),
        None,
    )
    if synthesis is None:
        return violations + ["_synthesize_diagnosis is missing"]
    calls = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "score_evidence_strength"
        for node in ast.walk(synthesis)
    )
    if not calls:
        violations.append("_synthesize_diagnosis does not call score_evidence_strength")
    return violations


def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


class TestTheAuthorsHoldNoSentenceOfTheirOwn:
    def test_the_pipeline_holds_none(self):
        assert (
            inline_sentence_violations(
                _source(diagnosis_pipeline),
                "diagnosis_pipeline.py",
                STAGE_OUTCOME_FUNCTIONS,
            )
            == []
        )

    def test_the_advisory_holds_none(self):
        assert (
            inline_sentence_violations(
                _source(advisory), "advisory.py", allowed=UNIT_LABELS
            )
            == []
        )

    def test_every_exempt_stage_outcome_function_exists(self):
        """An exemption for a function that is gone would hide a new one of
        the same name."""
        tree = ast.parse(_source(diagnosis_pipeline))
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert STAGE_OUTCOME_FUNCTIONS <= defined


class TestTheInlineSentenceGuardFires:
    def test_a_sentence_in_a_function(self):
        source = 'def f():\n    return "Plan an inspection."\n'
        assert inline_sentence_violations(source, "m.py") == [
            "m.py:2 f holds an inline sentence: 'Plan an inspection.'"
        ]

    def test_a_sentence_at_module_level(self):
        source = '_REPEAT = "Another analysis repeats it."\n'
        assert inline_sentence_violations(source, "m.py") == [
            "m.py:1 <module> holds an inline sentence: 'Another analysis repeats it.'"
        ]

    def test_the_literal_part_of_an_f_string(self):
        source = 'def f(z):\n    return f"Zone {z} is elevated."\n'
        assert inline_sentence_violations(source, "m.py") == [
            "m.py:2 f holds an inline sentence: 'Zone '",
            "m.py:2 f holds an inline sentence: ' is elevated.'",
        ]

    def test_a_sentence_in_a_nested_helper_names_its_top_level_function(self):
        source = 'def f():\n    def g():\n        return "A nested sentence."\n    return g\n'
        assert inline_sentence_violations(source, "m.py") == [
            "m.py:3 f holds an inline sentence: 'A nested sentence.'"
        ]

    def test_docstrings_raised_messages_keys_and_exempt_functions_pass(self):
        source = (
            '"""Module docstring with words."""\n'
            "def f(x):\n"
            '    """Function docstring with words."""\n'
            "    if x is None:\n"
            '        raise ValueError("A refusal message.")\n'
            '    return {"rms_velocity_mm_s": x, "sep": " ", "unit": "per cent"}\n'
            "def stage():\n"
            '    return "A stage outcome sentence."\n'
        )
        assert (
            inline_sentence_violations(
                source, "m.py", frozenset({"stage"}), frozenset({"per cent"})
            )
            == []
        )
        assert inline_sentence_violations(source, "m.py", frozenset({"stage"})) == [
            "m.py:6 f holds an inline sentence: 'per cent'"
        ]

    def test_stage_outcome_arguments_and_log_messages_pass_elsewhere_fails(self):
        source = (
            "def diagnose(fs):\n"
            '    logger.warning(f"Check not computed at {fs} Hz.")\n'
            '    block = refuse(reason=f"The band is empty at {fs} Hz.", remedy="Re-acquire.")\n'
            '    note = refuse(detail="An interpretation sentence.")\n'
            "    return block, note\n"
        )
        assert inline_sentence_violations(source, "m.py") == [
            "m.py:4 diagnose holds an inline sentence: 'An interpretation sentence.'"
        ]


class TestTheEvidenceRuleLivesInTheAdvisory:
    def test_the_pipeline_holds_no_evidence_arithmetic_and_no_grade(self):
        assert evidence_rule_violations(_source(diagnosis_pipeline), "p.py") == []

    def test_the_synthesis_takes_its_grade_from_the_advisory(self):
        assert scoring_source_violations(_source(diagnosis_pipeline)) == []

    @pytest.mark.parametrize("relative", OTHER_T2_MODULES)
    def test_no_other_module_of_the_file_set_names_a_grade(self, relative):
        source = (SRC / relative).read_text(encoding="utf-8")
        grades = [
            v
            for v in evidence_rule_violations(source, relative)
            if "names the grade" in v
        ]
        assert grades == []


class TestTheEvidenceRuleGuardFires:
    def test_a_constant_added_in_place(self):
        source = (
            "def f():\n    points = 0.0\n    points += 1.0\n    n += 1\n    m += k\n"
        )
        assert evidence_rule_violations(source, "m.py") == [
            "m.py:3 adds a constant to points",
            "m.py:4 adds a constant to n",
        ]

    def test_a_grade_word_outside_a_docstring(self):
        source = 'def f(p):\n    """Grades: weak, strong."""\n    return "moderate"\n'
        assert evidence_rule_violations(source, "m.py") == [
            "m.py:3 names the grade 'moderate'"
        ]

    def test_a_synthesis_without_the_call(self):
        source = (
            "from .advisory import score_evidence_strength\n"
            "def _synthesize_diagnosis():\n"
            '    return "none"\n'
        )
        assert scoring_source_violations(source) == [
            "_synthesize_diagnosis does not call score_evidence_strength"
        ]

    def test_a_scoring_function_not_imported_from_the_advisory(self):
        source = (
            "from .scores import score_evidence_strength\n"
            "def _synthesize_diagnosis(i, b, a):\n"
            "    return score_evidence_strength(i, b, a)\n"
        )
        assert scoring_source_violations(source) == [
            "score_evidence_strength is not imported from the advisory"
        ]


def _iso(zone: Optional[str]) -> dict:
    if zone is None:
        return {"status": "refused", "reason": "r", "remedy": "m"}
    return {"status": "assessed", "zone": zone}


SINGLE = {"claim": {"level": "single_spectrum_evidence", "conditions": []}}
SEVERAL = {"claim": {"level": "qualified", "conditions": ["several_types"]}}
HELD_BACK = {
    "claim": {
        "level": "qualified",
        "conditions": ["inconclusive", "rolling_element_held_back"],
    }
}
WITHHELD = {"claim": {"level": "withheld", "conditions": ["no_indication"]}}
FAULTY = {"overall_health": "Faulty"}
SUSPICIOUS = {"overall_health": "Suspicious"}
HEALTHY = {"overall_health": "Healthy"}


@pytest.mark.parametrize(
    ("zone", "bearing", "anomaly", "grade"),
    [
        (None, None, None, "none"),
        ("A", None, None, "none"),
        ("B", None, None, "none"),
        ("C", None, None, "weak"),
        ("D", None, None, "moderate"),
        ("E", None, None, "moderate"),
        (None, SINGLE, None, "weak"),
        ("A", SEVERAL, None, "weak"),
        ("A", HELD_BACK, None, "none"),
        ("A", WITHHELD, None, "none"),
        ("A", None, FAULTY, "weak"),
        ("A", None, SUSPICIOUS, "weak"),
        ("A", None, HEALTHY, "none"),
        ("C", None, SUSPICIOUS, "weak"),
        ("C", SINGLE, None, "moderate"),
        ("C", SINGLE, SUSPICIOUS, "moderate"),
        ("C", SINGLE, FAULTY, "strong"),
        ("D", SINGLE, None, "strong"),
        ("D", None, SUSPICIOUS, "moderate"),
        ("D", SINGLE, FAULTY, "strong"),
        (None, SINGLE, FAULTY, "moderate"),
    ],
)
def test_the_evidence_grade_of_each_branch(zone, bearing, anomaly, grade):
    """Points: zone C 1, zone D (and any zone outside A to C) 2, refused 0; a
    bearing finding 1; anomaly Faulty 1, Suspicious 0.5. Grades: 0 none, below
    2 weak, below 3 moderate, otherwise strong; the rows include the sums 1.5,
    2, 2.5 and 3 on either side of each bound."""
    assert score_evidence_strength(_iso(zone), bearing, anomaly) == grade


@pytest.mark.parametrize("zone", ["A", "B", "C", "D", "E"])
def test_the_zone_action_and_the_zone_points_agree(zone):
    """The synthesis picks the zone action and the advisory scores the zone,
    on two ladders over the same zones: a zone adds points exactly when it
    adds an action, and it takes the zone D action exactly when it scores as
    zone D (2 points, "moderate" on its own)."""
    iso = {
        "status": "assessed",
        "zone": zone,
        "rms_velocity_mm_s": 1.0,
        "severity_level": "Synthetic",
    }
    _, grade, recommendations = _synthesize_diagnosis(
        fft_summary={"peak_frequency_hz": 157.0},
        psd_summary={},
        stft_summary={},
        bearing_faults=None,
        bearing_check={"status": "not_requested"},
        iso_severity=iso,
        anomaly=None,
        rpm=1500.0,
    )
    zone_c = STATEMENTS["diagnosis_action_zone_c"]
    zone_d = STATEMENTS["diagnosis_action_zone_d"]
    actions = [r for r in recommendations if r in (zone_c, zone_d)]
    assert grade == score_evidence_strength(iso, None, None)
    assert bool(actions) == (grade != "none")
    assert (actions == [zone_d]) == (grade == "moderate")
