"""The closed list of non-bearing statements stays closed, versioned and used.

``decision_support.statements.STATEMENTS`` holds every sentence and action
that the diagnosis synthesis and the advisory write outside the bearing
claim's own list. These tests keep it a closed list:

- its keys are exactly the pinned set, each key names its author, and no two
  keys hold one wording;
- ``INTERPRETATION_POLICY_VERSION`` matches a fingerprint of the list and of
  the advisory's evidence-strength rule (points, grade bounds, acceptable
  zones), so a wording, weight or zone that changes without a new version
  fails here;
- the two authors reference every key and no other;
- every wording the characterization matrix reaches appears in its goldens,
  which were written from the server's outputs before this list existed, so
  the list holds the wordings the server already used; the keys the matrix
  cannot reach are pinned;
- the version reaches no output: no other module reads it.

Each check is a pure function returning its violations, with red paths on
small inputs that assert the exact list.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import string
from pathlib import Path

from predictive_maintenance_mcp import decision_support
from predictive_maintenance_mcp.decision_support import advisory, diagnosis_pipeline
from predictive_maintenance_mcp.decision_support.advisory import (
    ACCEPTABLE_ZONES,
    EVIDENCE_GRADE_BOUNDS,
    EVIDENCE_POINTS,
)
from predictive_maintenance_mcp.decision_support.statements import (
    INTERPRETATION_POLICY_VERSION,
    STATEMENTS,
)

SRC = Path(diagnosis_pipeline.__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "t2"
KEY_SHAPE = re.compile(r"^(diagnosis|report)_[a-z0-9_]+$")

#: The closed list: adding or removing a statement edits this tuple too.
PINNED_KEYS = (
    "diagnosis_iso_refused",
    "diagnosis_iso_assessed",
    "diagnosis_iso_assessed_no_severity",
    "diagnosis_dominant_frequency",
    "diagnosis_shaft_order_line",
    "diagnosis_shaft_order_window_1x",
    "diagnosis_shaft_order_window_2x",
    "diagnosis_shaft_order_readings_1x",
    "diagnosis_shaft_order_readings_2x",
    "diagnosis_shaft_order_discrimination",
    "diagnosis_shaft_order_repeat",
    "diagnosis_bearing_not_computed",
    "diagnosis_anomaly",
    "diagnosis_action_zone_c",
    "diagnosis_action_zone_d",
    "diagnosis_action_anomaly_faulty",
    "diagnosis_action_anomaly_suspicious",
    "diagnosis_action_default",
    "report_iso_refused",
    "report_iso_assessed",
    "report_bearing_not_computed",
    "report_bearing_absent",
    "report_bearing_absent_remedy",
    "report_anomaly_absent",
    "report_anomaly_absent_remedy",
    "report_anomaly",
    "report_anomaly_segments_counted",
    "report_anomaly_segments_share",
    "report_energy_absent",
    "report_energy",
    "report_verdict_not_computed",
    "report_verdict_not_computed_for_bearing",
    "report_verdict_not_attempted",
    "report_verdict_zone_elevated",
    "report_verdict_anomaly_flag",
    "report_verdict_tail_unchecked",
    "report_verdict_tail_non_bearing",
    "report_verdict_tail_outside_bearing",
    "report_dominant_frequency",
    "report_indicator_single_spectrum",
    "report_indicator_anomaly_model",
    "report_indicator_zone",
    "report_description_iso_refused",
    "report_description_bearing_refused",
    "report_description_anomaly_absent",
    "report_motivation_zone_severity",
    "report_motivation_zone",
    "report_baseline_absent",
    "report_baseline_absent_remedy",
    "report_baseline_refused",
    "report_baseline_refused_remedy",
    "report_baseline_unit_mismatch",
    "report_baseline_bearing_mismatch",
    "report_baseline_nothing_compared",
    "report_baseline_nothing_compared_remedy",
    "report_baseline_unchanged",
    "report_baseline_moved",
    "report_delta_rms_unchanged",
    "report_delta_rms_moved",
    "report_delta_anomaly_unchanged",
    "report_delta_anomaly_moved",
)

#: The fingerprint each version of the list and the evidence table carries.
#: A new wording, entry or weight needs a new version with its own pin.
PINNED_FINGERPRINTS = {
    1: "babf82bb7db153a87f8e03cd26971e747f6a201cb8947e2467402d193e025e1d",
}

#: Keys whose wording the characterization matrix never produces: it has no
#: anomaly model, no indicator disagreement, no empty energy breakdown, no
#: unchecked bearing without a designation, and no baseline that moves or
#: shares no assessed block.
PINNED_UNREACHABLE = frozenset(
    {
        "diagnosis_action_anomaly_faulty",
        "diagnosis_action_anomaly_suspicious",
        "diagnosis_anomaly",
        "report_anomaly",
        "report_anomaly_absent",
        "report_anomaly_absent_remedy",
        "report_anomaly_segments_counted",
        "report_anomaly_segments_share",
        "report_baseline_moved",
        "report_baseline_nothing_compared",
        "report_baseline_nothing_compared_remedy",
        "report_delta_anomaly_moved",
        "report_delta_anomaly_unchanged",
        "report_delta_rms_moved",
        "report_description_anomaly_absent",
        "report_energy_absent",
        "report_indicator_anomaly_model",
        "report_indicator_single_spectrum",
        "report_verdict_anomaly_flag",
        "report_verdict_not_computed",
        "report_verdict_tail_outside_bearing",
        "report_verdict_tail_unchecked",
    }
)


def key_set_violations(keys, pinned) -> list[str]:
    keys, pinned = list(keys), list(pinned)
    return [f"key not in the pinned set: {k}" for k in keys if k not in pinned] + [
        f"pinned key missing from the list: {k}" for k in pinned if k not in keys
    ]


def shape_violations(statements: dict) -> list[str]:
    violations = []
    first_key_of = {}
    for key, wording in statements.items():
        if not KEY_SHAPE.match(key):
            violations.append(f"key {key} does not name its author")
        if not isinstance(wording, str) or not wording.strip():
            violations.append(f"key {key} holds no wording")
            continue
        if wording in first_key_of:
            violations.append(
                f"keys {first_key_of[wording]} and {key} hold the same wording"
            )
        else:
            first_key_of[wording] = key
    return violations


def fingerprint(statements: dict, points: dict, bounds, zones) -> str:
    payload = json.dumps(
        {
            "statements": statements,
            "evidence_points": points,
            "grade_bounds": bounds,
            "acceptable_zones": zones,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def version_violations(version: int, digest: str, pinned: dict) -> list[str]:
    violations = []
    seen = {}
    for pinned_version, pinned_digest in sorted(pinned.items()):
        if pinned_digest in seen:
            violations.append(
                f"versions {seen[pinned_digest]} and {pinned_version} share a fingerprint"
            )
        seen.setdefault(pinned_digest, pinned_version)
    if version not in pinned:
        violations.append(f"version {version} has no pinned fingerprint")
    elif pinned[version] != digest:
        violations.append(
            f"version {version} is pinned to {pinned[version]}; "
            f"the list and the table give {digest}"
        )
    return violations


def referenced_keys(source: str) -> set[str]:
    """Every string constant of a module shaped like a statement key."""
    return {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and KEY_SHAPE.match(node.value)
    }


def usage_violations(referenced: set, keys) -> list[str]:
    keys = set(keys)
    return [f"key {k} is used by neither author" for k in sorted(keys - referenced)] + [
        f"{k} is referenced but not in the list" for k in sorted(referenced - keys)
    ]


def wording_pattern(template: str) -> re.Pattern:
    """The template with every field matching any text.

    A field can therefore absorb fixed text: the ISO line without a severity
    word counts as reached through the line that has one.
    """
    parts = []
    for literal, field, _spec, _conversion in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append("(?s:.*?)")
    return re.compile("".join(parts))


def reached_keys(statements: dict, texts) -> set[str]:
    texts = list(texts)
    return {
        key
        for key, template in statements.items()
        if any(wording_pattern(template).search(text) for text in texts)
    }


def coverage_violations(reached: set, keys, unreachable) -> list[str]:
    keys = set(keys)
    violations = [
        f"wording of {k} appears in no golden"
        for k in sorted(keys - reached - set(unreachable))
    ]
    violations += [
        f"{k} is pinned unreachable but a golden holds its wording"
        for k in sorted(reached & set(unreachable))
    ]
    violations += [
        f"pinned unreachable {k} is not a key" for k in sorted(set(unreachable) - keys)
    ]
    return violations


def version_reader_violations(sources: dict) -> list[str]:
    return [
        f"{name} reads the interpretation policy version"
        for name, text in sorted(sources.items())
        if "interpretation_policy_version" in text.lower()
    ]


def _golden_texts() -> list[str]:
    texts = []

    def walk(value):
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in sorted(GOLDEN_DIR.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return texts


def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


class TestTheListIsClosed:
    def test_the_keys_are_the_pinned_set(self):
        assert key_set_violations(STATEMENTS, PINNED_KEYS) == []

    def test_each_key_names_its_author_and_holds_its_own_wording(self):
        assert shape_violations(STATEMENTS) == []

    def test_the_version_carries_the_fingerprint_of_the_list_and_the_table(self):
        digest = fingerprint(
            STATEMENTS, EVIDENCE_POINTS, EVIDENCE_GRADE_BOUNDS, ACCEPTABLE_ZONES
        )
        assert (
            version_violations(
                INTERPRETATION_POLICY_VERSION, digest, PINNED_FINGERPRINTS
            )
            == []
        )

    def test_the_authors_use_every_key_and_no_other(self):
        referenced = referenced_keys(_source(diagnosis_pipeline)) | referenced_keys(
            _source(advisory)
        )
        assert usage_violations(referenced, STATEMENTS) == []

    def test_the_list_holds_the_wordings_the_server_already_used(self):
        reached = reached_keys(STATEMENTS, _golden_texts())
        assert coverage_violations(reached, STATEMENTS, PINNED_UNREACHABLE) == []


class TestTheVersionStaysInternal:
    def test_no_other_module_reads_it(self):
        sources = {
            str(path.relative_to(SRC)): path.read_text(encoding="utf-8")
            for path in SRC.rglob("*.py")
            if path.relative_to(SRC).as_posix() != "decision_support/statements.py"
        }
        assert version_reader_violations(sources) == []

    def test_the_package_does_not_export_it(self):
        assert not hasattr(decision_support, "INTERPRETATION_POLICY_VERSION")


class TestTheRegistryGuardsFire:
    def test_an_added_and_a_removed_key(self):
        assert key_set_violations(["a_x", "new_y"], ["a_x", "old_z"]) == [
            "key not in the pinned set: new_y",
            "pinned key missing from the list: old_z",
        ]

    def test_a_key_without_its_author_an_empty_wording_and_a_duplicate(self):
        statements = {
            "summary_line": "One wording.",
            "report_blank": " ",
            "report_one": "Same wording.",
            "diagnosis_two": "Same wording.",
        }
        assert shape_violations(statements) == [
            "key summary_line does not name its author",
            "key report_blank holds no wording",
            "keys report_one and diagnosis_two hold the same wording",
        ]

    def test_a_reworded_statement_and_a_changed_rule_change_the_fingerprint(self):
        wording, weight, bound, zones = (
            {"report_a": "A wording."},
            {"w": 1.0},
            (("weak", 2.0),),
            ("A", "B"),
        )
        base = fingerprint(wording, weight, bound, zones)
        reworded = fingerprint({"report_a": "A wording!"}, weight, bound, zones)
        reweighted = fingerprint(wording, {"w": 1.5}, bound, zones)
        rebound = fingerprint(wording, weight, (("weak", 2.5),), zones)
        rezoned = fingerprint(wording, weight, bound, ("A",))
        assert len({base, reworded, reweighted, rebound, rezoned}) == 5
        assert version_violations(1, reworded, {1: base}) == [
            f"version 1 is pinned to {base}; the list and the table give {reworded}"
        ]

    def test_a_new_version_without_its_own_pin(self):
        assert version_violations(2, "d", {1: "d"}) == [
            "version 2 has no pinned fingerprint"
        ]
        assert version_violations(2, "d", {1: "d", 2: "d"}) == [
            "versions 1 and 2 share a fingerprint"
        ]

    def test_an_unused_and_an_unknown_key(self):
        source = 'S = {}\ndef f():\n    return S["report_used"] + S["report_typo"]\n'
        referenced = referenced_keys(source)
        assert usage_violations(referenced, ["report_used", "report_unused"]) == [
            "key report_unused is used by neither author",
            "report_typo is referenced but not in the list",
        ]

    def test_a_wording_no_golden_holds_and_a_stale_unreachable_pin(self):
        statements = {
            "report_seen": "Zone {zone} is classified as {severity}.",
            "report_new": "A wording the server never wrote.",
            "report_pinned": "Also seen.",
        }
        texts = ["Prefix. Zone C is classified as unsatisfactory.", "Also seen."]
        reached = reached_keys(statements, texts)
        assert reached == {"report_seen", "report_pinned"}
        assert coverage_violations(
            reached, statements, {"report_pinned", "report_gone"}
        ) == [
            "wording of report_new appears in no golden",
            "report_pinned is pinned unreachable but a golden holds its wording",
            "pinned unreachable report_gone is not a key",
        ]

    def test_a_module_reading_the_version(self):
        sources = {
            "a.py": "x = 1\n",
            "b.py": "from .statements import INTERPRETATION_POLICY_VERSION\n",
            "c.py": 'p = {"interpretation_policy_version": 1}\n',
        }
        assert version_reader_violations(sources) == [
            "b.py reads the interpretation policy version",
            "c.py reads the interpretation policy version",
        ]
