"""U5 tests: the scorer over hand-built outcomes and tiny label tables.

Everything is hand-computable: fixtures are in-test outcome dicts plus a
small synthetic labeled-record set (never the real 64-record tables), so
every rate asserted here can be checked by counting on paper. Coverage:

- happy path: hand-computed per-stratum counts AND rates reproduced
  exactly, one Y1 record per outcome;
- N1/N2 land in their own strata and never move headline (Y1/Y2)
  accuracy;
- the hit criterion is a supported hypothesis: a significant line
  alone, a shared line or a 2x BSF harmonic is not a hit;
- the normal baselines are counted under the same criterion, as counts
  (no indication, inconclusive, bearing alert, named fault), never as a
  rate;
- the secondary classification ranking among supported faults
  (evidence tier high > moderate, then magnitude descending, then
  |deviation_pct| ascending, then fixed BPFO/BPFI/BSF/FTF order);
- what users see (AE9): the outcome distribution, the precision of
  "indicated", the strongest candidate, the fault the claim names and
  the confusion table, every rate with its denominator;
- outcomes measured under another evidence or claim policy are refused;
- ungraded and known-anomalies reporting lines;
- fail-closed refusals: missing/extra/non-ok outcome ids named, empty
  outcomes, empty labels, malformed ok outcome, unknown metadata key;
- metadata injectability (deterministic serialization under overrides)
  and the canonical results write;
- the ``score`` CLI stage end-to-end on tmp paths.

Results are only ever written to tmp_path — never to the committed
benchmarks/cwru/results/ slot (maintainer-run artifact, U7).
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import scipy

import predictive_maintenance_mcp
from benchmarks.cwru import runner, scorer
from benchmarks.cwru.__main__ import main
from benchmarks.cwru.records import LabeledRecord, LabelRecord, OpsRecord
from predictive_maintenance_mcp.diagnostics.bearing_claims import CLAIM_POLICY_VERSION
from predictive_maintenance_mcp.diagnostics.bearing_evidence import EVIDENCE_POLICY

#: Deterministic metadata for every score_results call in this file —
#: the varying values (date, git describe, platform) are pinned so no
#: test depends on the clock, git, or the machine.
META = {
    "date": "2026-08-10T00:00:00+00:00",
    "git_describe": "test-describe",
    "platform": "test-platform",
}


def _ops(opaque_id: str, file_id: int) -> OpsRecord:
    """A minimal valid ops record (operational fields only)."""
    return OpsRecord(
        opaque_id=opaque_id,
        file_id=file_id,
        url=f"https://engineering.case.edu/sites/default/files/{file_id}.mat",
        internal_mat_key=f"X{file_id}_DE_time",
        channel="DE",
        fs_hz=12000,
        nominal_rpm=1797,
        load_hp=0,
        cache_filename=f"{file_id}.mat",
    )


def _labeled(
    opaque_id: str,
    file_id: int,
    fault_type: str,
    *,
    grade: Optional[str] = None,
    known_anomalies: tuple[str, ...] = (),
) -> LabeledRecord:
    """A labeled record with a consistent label for *fault_type*."""
    label = LabelRecord(
        fault_type=fault_type,  # type: ignore[arg-type]
        or_position="centered_6" if fault_type == "outer_race" else None,
        fault_diameter_in=None if fault_type == "normal" else 0.007,
        sr2015_grade=grade,  # type: ignore[arg-type]
        known_anomalies=list(known_anomalies),
    )
    return LabeledRecord(ops=_ops(opaque_id, file_id), label=label)


def _check(fault_key: str, **overrides: Any) -> dict[str, Any]:
    """One fault check in the analyzer's raw shape (default: no finding)."""
    canonical = {
        "BPFO": "outer_race",
        "BPFI": "inner_race",
        "BSF": "ball",
        "FTF": "cage",
    }
    base: dict[str, Any] = {
        "fault_type_canonical": canonical[fault_key],
        "expected_frequency_hz": 100.0,
        "detected": False,
        "detected_frequency_hz": None,
        "magnitude": None,
        "deviation_pct": None,
        "harmonics_detected": [],
        "evidence_strength": "none",
        "state": "absent",
        "state_reason": "no_significant_line",
        "line_frequency_hz": 100.0,
        "line_peak_hz": 100.0,
        "line_amplitude": 0.01,
        "level_db": 3.0,
        "threshold_db": 20.0,
        "shared_with": [],
    }
    base.update(overrides)
    return base


def _harmonic(order: int = 2, magnitude: float = 0.01) -> dict[str, Any]:
    """One harmonics_detected entry in the analyzer's shape."""
    return {
        "harmonic": order,
        "expected_hz": 100.0 * order,
        "detected_hz": 100.0 * order,
        "magnitude": magnitude,
    }


def _supported(evidence: str = "moderate", **overrides: Any) -> dict[str, Any]:
    """Check overrides for a supported hypothesis at *evidence*."""
    values: dict[str, Any] = {
        "detected": True,
        "detected_frequency_hz": 100.0,
        "magnitude": 0.5,
        "deviation_pct": 0.5,
        "evidence_strength": evidence,
        "state": "supported",
        "state_reason": "corroborated",
        "level_db": 30.0,
    }
    values.update(overrides)
    return values


def _significant(state: str = "unconfirmed", **overrides: Any) -> dict[str, Any]:
    """Check overrides for a significant line that is not supported."""
    values: dict[str, Any] = {
        "detected": True,
        "detected_frequency_hz": 100.0,
        "magnitude": 0.5,
        "deviation_pct": 0.5,
        "evidence_strength": "low",
        "state": state,
        "state_reason": "no_corroboration" if state == "unconfirmed" else "line_shared",
        "level_db": 25.0,
    }
    values.update(overrides)
    return values


#: The policy versions the engine produces today; a hand-built outcome
#: carries them unless a test overrides one.
EVIDENCE_VERSION = EVIDENCE_POLICY["version"]
CLAIM_VERSION = CLAIM_POLICY_VERSION


def _outcome(
    opaque_id: str,
    checks: Optional[dict[str, dict[str, Any]]] = None,
    status: str = "ok",
    *,
    outcome: Optional[str] = None,
    candidates: Optional[list[str]] = None,
    claim_level: Optional[str] = None,
    named: Optional[str] = None,
    evidence_policy_version: Any = EVIDENCE_VERSION,
    claim_policy_version: Any = CLAIM_VERSION,
) -> dict[str, Any]:
    """A hand-built runner outcome (only the fields the scorer reads).

    The summary follows the checks unless a test sets it: candidates are
    the supported hypotheses by level, the outcome follows their count,
    and the claim names the single candidate of an indicated outcome.
    """
    if status != runner.OUTCOME_STATUS_OK:
        return {
            "signal_id": opaque_id,
            "status": status,
            "error": "synthetic non-ok outcome",
        }
    fault_checks = {
        key: _check(key, **(checks or {}).get(key, {})) for key in scorer.FAULT_KEYS
    }
    if candidates is None:
        supported = [
            k for k in scorer.FAULT_KEYS if fault_checks[k]["state"] == "supported"
        ]
        candidates = sorted(supported, key=lambda k: -fault_checks[k]["level_db"])
    if outcome is None:
        outcome = {0: "no_indication", 1: "indicated"}.get(
            len(candidates), "several_types"
        )
    if claim_level is None:
        claim_level = {
            "indicated": "single_spectrum_evidence",
            "several_types": "qualified",
        }.get(outcome, "withheld")
    if named is None and claim_level == "single_spectrum_evidence":
        named = candidates[0]
    return {
        "signal_id": opaque_id,
        "status": runner.OUTCOME_STATUS_OK,
        "bearing": {
            "fault_checks": fault_checks,
            "outcome": outcome,
            "candidates": candidates,
            "claim_level": claim_level,
            "most_likely_fault": named,
            "evidence_policy_version": evidence_policy_version,
            "claim_policy_version": claim_policy_version,
        },
    }


def _score(labeled: tuple[LabeledRecord, ...], outcomes: dict) -> dict[str, Any]:
    """score_results with the deterministic metadata pin."""
    return scorer.score_results(outcomes, labeled, metadata_overrides=META)


# ---------------------------------------------------------------------------
# Happy path: hand-computed per-stratum counts and rates
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Fixture outcomes + labels reproduce hand-computed rates exactly."""

    @pytest.fixture
    def scored(self) -> dict[str, Any]:
        """4 Y1 faulted records (one per outcome) + 1 clean ungraded normal."""
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 118, "outer_race", grade="Y1"),
            _labeled("cwru_003", 130, "ball", grade="Y1"),
            _labeled("cwru_004", 144, "inner_race", grade="Y1"),
            _labeled("cwru_005", 97, "normal"),
        )
        outcomes = {
            # Indicated, correct: BPFI alone is supported and named.
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            # Several types, the label among them and the strongest.
            "cwru_002": _outcome(
                "cwru_002",
                {
                    "BPFO": _supported("high", level_db=30.0),
                    "BPFI": _supported("moderate", level_db=20.0),
                },
            ),
            # Inconclusive: the BSF line is significant but unconfirmed.
            # A 2x harmonic no longer makes a ball hit.
            "cwru_003": _outcome(
                "cwru_003",
                {"BSF": _significant(harmonics_detected=[_harmonic(order=2)])},
                outcome="inconclusive",
            ),
            # No indication: nothing significant anywhere.
            "cwru_004": _outcome("cwru_004"),
            # Clean normal baseline.
            "cwru_005": _outcome("cwru_005"),
        }
        return _score(labeled, outcomes)

    def test_y1_counts_and_rates_hand_computed(self, scored):
        y1 = scored["strata"]["Y1"]
        assert y1["n_records"] == 4
        assert y1["n_faulted"] == 4
        assert y1["n_normal"] == 0
        # Counts always published alongside rates.
        assert y1["frequency_detection"] == {"hits": 3, "total": 4, "rate": 0.75}
        assert y1["label_supported"] == {"count": 2, "total": 4, "rate": 0.5}
        assert y1["classification"] == {"correct": 2, "total": 4, "rate": 0.5}
        assert y1["end_to_end"] == {"correct": 2, "total": 4, "rate": 0.5}

    def test_y1_user_visible_outcome_hand_computed(self, scored):
        y1 = scored["strata"]["Y1"]
        assert y1["outcomes"] == {
            "indicated_correct": 1,
            "indicated_wrong": 0,
            "several_types_with_label": 1,
            "several_types_without_label": 0,
            "inconclusive": 1,
            "no_indication": 1,
            "total": 4,
        }
        assert y1["indicated_precision"] == {"correct": 1, "total": 1, "rate": 1.0}
        assert y1["strongest_candidate"] == {"correct": 2, "total": 4, "rate": 0.5}
        assert y1["named_fault"] == {"correct": 1, "wrong": 0, "none": 3, "total": 4}
        assert y1["sensitivity"] == {"alerts": 2, "total": 4, "rate": 0.5}
        assert y1["wrong_type_support"] == {"records": 1, "total": 4, "rate": 0.25}

    def test_normal_lands_in_ungraded_as_counts(self, scored):
        ungraded = scored["strata"]["ungraded"]
        assert ungraded["n_records"] == 1
        assert ungraded["n_normal"] == 1
        assert ungraded["normals"] == {
            "total": 1,
            "no_indication": 1,
            "inconclusive": 0,
            "indicated": 0,
            "several_types": 0,
            "bearing_alert": 0,
            "named_fault": 0,
            "supported_counts": {"BPFO": 0, "BPFI": 0, "BSF": 0, "FTF": 0},
        }
        # The secondary ranking covers fault records only.
        assert ungraded["classification"] == {"correct": 0, "total": 0, "rate": None}

    def test_headline_covers_exactly_y1_y2(self, scored):
        headline = scored["headline"]
        assert headline["strata_included"] == ["Y1", "Y2"]
        # Only the 4 Y1 records — the ungraded normal is not headline.
        assert headline["n_records"] == 4
        assert headline["classification"] == {"correct": 2, "total": 4, "rate": 0.5}
        assert headline["outcomes"]["total"] == 4

    def test_per_record_table_rows(self, scored):
        records = scored["records"]
        assert set(records) == {f"cwru_00{i}" for i in range(1, 6)}
        first = records["cwru_001"]
        assert first["ranked_first"] == "BPFI"
        assert first["classification_correct"] is True
        assert first["outcome_class"] == "indicated_correct"
        assert first["named_fault"] == "BPFI"
        assert first["named_correct"] is True
        second = records["cwru_002"]
        assert second["outcome_class"] == "several_types_with_label"
        assert second["strongest_candidate"] == "BPFO"
        assert second["strongest_candidate_correct"] is True
        assert second["wrong_type_supported"] is True
        assert second["bearing_alert"] is True
        third = records["cwru_003"]
        assert third["ranked_first"] is None
        assert third["frequency_detection_hit"] is True
        assert third["label_supported"] is False
        assert third["outcome_class"] == "inconclusive"
        assert records["cwru_004"]["ranked_first"] is None
        assert records["cwru_004"]["classification_correct"] is False
        normal = records["cwru_005"]
        assert normal["expected_fault"] is None
        assert normal["outcome_class"] == "no_indication"
        assert normal["named_correct"] is None
        assert normal["strongest_candidate_correct"] is None
        assert normal["classification_correct"] is None
        for row in records.values():
            assert row["end_to_end_correct"] is row["classification_correct"]

    def test_empty_strata_report_zero_counts_and_null_rates(self, scored):
        # Every stratum is always present; empty ones are honestly empty.
        assert set(scored["strata"]) == set(scorer.STRATA)
        p1 = scored["strata"]["P1"]
        assert p1["n_records"] == 0
        assert p1["frequency_detection"] == {"hits": 0, "total": 0, "rate": None}
        assert p1["classification"]["rate"] is None
        assert p1["indicated_precision"] == {"correct": 0, "total": 0, "rate": None}
        assert p1["sensitivity"]["rate"] is None


# ---------------------------------------------------------------------------
# Stratification: N1/N2 separate, never in headline; ungraded bucket
# ---------------------------------------------------------------------------


class TestStratification:
    """Known-hard strata are reported, never blended into the headline."""

    def test_n_strata_never_move_headline_accuracy(self):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 118, "inner_race", grade="N1"),
            _labeled("cwru_003", 130, "inner_race", grade="N2"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),  # miss — known-hard
            "cwru_003": _outcome("cwru_003"),  # miss — known-hard
        }
        results = _score(labeled, outcomes)

        # Headline is the Y1 record alone: 1/1, unmoved by the N misses.
        assert results["headline"]["n_records"] == 1
        assert results["headline"]["classification"] == {
            "correct": 1,
            "total": 1,
            "rate": 1.0,
        }
        # The known-hard records are visible in their own strata.
        assert results["strata"]["N1"]["classification"] == {
            "correct": 0,
            "total": 1,
            "rate": 0.0,
        }
        assert results["strata"]["N2"]["n_records"] == 1

    def test_null_grade_lands_in_ungraded_stratum(self):
        labeled = (_labeled("cwru_001", 105, "outer_race", grade=None),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFO": _supported("high")})}
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["stratum"] == "ungraded"
        assert results["strata"]["ungraded"]["frequency_detection"]["hits"] == 1
        assert results["headline"]["n_records"] == 0


# ---------------------------------------------------------------------------
# Hit criterion: a supported hypothesis, nothing weaker
# ---------------------------------------------------------------------------


class TestHitCriterion:
    """A fault hits only when its state is supported: a significant line
    alone, a shared line or a 2x harmonic is not a hit."""

    def test_significant_but_unconfirmed_label_is_detected_not_supported(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {"BPFI": _significant("unconfirmed")},
                outcome="inconclusive",
            )
        }
        row = _score(labeled, outcomes)["records"]["cwru_001"]
        assert row["frequency_detection_hit"] is True
        assert row["label_supported"] is False
        assert row["fault_hits"]["BPFI"] is False
        assert row["ranked_first"] is None
        assert row["classification_correct"] is False

    def test_shared_line_is_not_a_hit(self):
        labeled = (_labeled("cwru_001", 118, "outer_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFO": _significant("shared", shared_with=["BPFI"]),
                    "BPFI": _significant("shared", shared_with=["BPFO"]),
                },
                outcome="inconclusive",
            )
        }
        row = _score(labeled, outcomes)["records"]["cwru_001"]
        assert row["fault_hits"] == {
            "BPFO": False,
            "BPFI": False,
            "BSF": False,
            "FTF": False,
        }
        assert row["wrong_type_supported"] is False

    def test_ball_harmonic_two_alone_is_not_a_hit(self):
        """The 2x BSF leg is gone: the engine itself tests the
        rolling-element line at twice the spin frequency."""
        labeled = (_labeled("cwru_001", 118, "ball", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BSF": {
                        "evidence_strength": "low",
                        "harmonics_detected": [_harmonic(order=2)],
                    }
                },
            )
        }
        row = _score(labeled, outcomes)["records"]["cwru_001"]
        assert row["frequency_detection_hit"] is False
        assert row["fault_hits"]["BSF"] is False
        assert row["classification_correct"] is False

    def test_supported_ball_is_a_hit(self):
        labeled = (_labeled("cwru_001", 118, "ball", grade="Y1"),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BSF": _supported("moderate")})}
        row = _score(labeled, outcomes)["records"]["cwru_001"]
        assert row["fault_hits"]["BSF"] is True
        assert row["ranked_first"] == "BSF"
        assert row["classification_correct"] is True


# ---------------------------------------------------------------------------
# Normal baselines: counts under the same criterion, never a rate
# ---------------------------------------------------------------------------


class TestNormalBaselines:
    """The criterion that counts a hit on a fault record counts it on a
    normal baseline too, and the normals are reported as counts."""

    def test_inconclusive_normal_is_counted_apart_from_alerts(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {"BPFO": _significant("unconfirmed")},
                outcome="inconclusive",
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "inconclusive"
        assert row["bearing_alert"] is False
        assert row["classification_correct"] is None
        normals = results["strata"]["ungraded"]["normals"]
        assert normals["inconclusive"] == 1
        assert normals["bearing_alert"] == 0
        assert normals["named_fault"] == 0

    def test_named_type_on_a_normal_is_an_alert_and_a_named_fault(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFO": _supported("moderate")})}
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "indicated_wrong"
        assert row["named_fault"] == "BPFO"
        assert row["bearing_alert"] is True
        assert results["strata"]["ungraded"]["normals"] == {
            "total": 1,
            "no_indication": 0,
            "inconclusive": 0,
            "indicated": 1,
            "several_types": 0,
            "bearing_alert": 1,
            "named_fault": 1,
            "supported_counts": {"BPFO": 1, "BPFI": 0, "BSF": 0, "FTF": 0},
        }

    def test_several_types_on_a_normal_is_an_alert_without_a_named_fault(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFO": _supported("moderate"),
                    "BPFI": _supported("moderate", level_db=20.0),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "several_types_without_label"
        assert row["named_fault"] is None
        normals = results["strata"]["ungraded"]["normals"]
        assert normals["bearing_alert"] == 1
        assert (normals["several_types"], normals["indicated"]) == (1, 0)
        assert normals["named_fault"] == 0
        assert normals["supported_counts"] == {"BPFO": 1, "BPFI": 1, "BSF": 0, "FTF": 0}

    def test_a_baselines_group_carries_no_rate_but_the_precision(self):
        """On a group of normal baselines every figure is a count; the only
        rate is the precision of "indicated", whose denominator is the
        indicated outcomes and which takes in the baselines as well."""
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFO": _supported("moderate")})}
        ungraded = _score(labeled, outcomes)["strata"]["ungraded"]
        rated = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                if node.get("rate") is not None:
                    rated.append(path)
                for key, value in node.items():
                    walk(value, f"{path}.{key}" if path else key)

        walk(ungraded, "")
        assert rated == ["indicated_precision"]
        assert ungraded["indicated_precision"] == {
            "correct": 0,
            "total": 1,
            "rate": 0.0,
        }
        assert ungraded["classification"] == {"correct": 0, "total": 0, "rate": None}
        assert "rate" not in ungraded["normals"]

    def test_a_fault_record_beside_the_baselines_stays_out_of_their_counts(self):
        labeled = (
            _labeled("cwru_001", 97, "normal"),
            _labeled("cwru_002", 105, "inner_race"),  # grade not transcribed
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001"),
            "cwru_002": _outcome("cwru_002", {"BPFO": _supported("moderate")}),
        }
        ungraded = _score(labeled, outcomes)["strata"]["ungraded"]
        assert (ungraded["n_faulted"], ungraded["n_normal"]) == (1, 1)
        assert ungraded["outcomes"]["indicated_wrong"] == 1
        assert ungraded["normals"]["indicated"] == 0
        assert ungraded["normals"]["bearing_alert"] == 0
        assert ungraded["normals"]["no_indication"] == 1
        assert ungraded["classification"] == {"correct": 0, "total": 1, "rate": 0.0}


# ---------------------------------------------------------------------------
# Classification ranking (documented rule)
# ---------------------------------------------------------------------------


class TestClassificationRanking:
    """Secondary ranking among supported faults: evidence tier (high >
    moderate), then magnitude descending, then |deviation_pct| ascending,
    then fixed BPFO/BPFI/BSF/FTF order."""

    def test_ranked_first_equal_to_label_is_correct(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _supported("high"),
                    "BPFO": _supported("moderate"),  # weaker tier
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["ranked_first"] == "BPFI"
        assert row["classification_correct"] is True

    def test_wrong_fault_stronger_is_incorrect(self):
        """Two faults meet the criterion; the wrong one is stronger."""
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _supported("moderate"),
                    "BPFO": _supported("high"),  # stronger, wrong
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        # The labeled frequency IS detected — but classification ranks.
        assert row["frequency_detection_hit"] is True
        assert row["ranked_first"] == "BPFO"
        assert row["classification_correct"] is False

    def test_magnitude_breaks_equal_evidence_ties(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _supported("moderate", magnitude=0.4),
                    "BPFO": _supported("moderate", magnitude=0.9),
                },
            )
        }
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["ranked_first"] == "BPFO"
        assert results["records"]["cwru_001"]["classification_correct"] is False

    def test_lower_absolute_deviation_breaks_equal_tier_and_magnitude(self):
        """Level 3: with tier and magnitude tied, the smaller
        |deviation_pct| ranks first. The signs are chosen so a signed
        (non-absolute) comparison would pick the OTHER fault, and the
        winner is not first in the fixed order either — isolating the
        |deviation| tie-break from levels 1, 2, and 4."""
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _supported("moderate", magnitude=0.5, deviation_pct=0.3),
                    "BPFO": _supported("moderate", magnitude=0.5, deviation_pct=-2.0),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["ranked_first"] == "BPFI"  # |0.3| < |-2.0|
        assert row["classification_correct"] is True

    def test_fixed_fault_order_is_the_final_deterministic_tie_break(self):
        """Level 4: a full tie on tier, magnitude, AND |deviation_pct| is
        decided by the fixed BPFO/BPFI/BSF/FTF order (scorer.FAULT_KEYS),
        deterministically."""
        labeled = (_labeled("cwru_001", 130, "ball", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    # Identical checks: _detected defaults share
                    # magnitude and deviation_pct exactly.
                    "BPFI": _supported("moderate"),
                    "BSF": _supported("moderate"),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert scorer.FAULT_KEYS.index("BPFI") < scorer.FAULT_KEYS.index("BSF")
        assert row["ranked_first"] == "BPFI"  # earlier in the fixed order
        # The labeled BSF still hits, but ranking is deterministic, so
        # the fixed order names BPFI first and classification misses.
        assert row["frequency_detection_hit"] is True
        assert row["classification_correct"] is False


# ---------------------------------------------------------------------------
# What users see: outcome, precision of indicated, candidates, named fault
# ---------------------------------------------------------------------------


class TestUserVisibleOutcome:
    """AE9 scenarios on hand-built outcomes; every rate has its denominator."""

    def test_several_types_with_the_label_among_the_candidates(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y2"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFO": _supported("high", level_db=35.0),
                    "BPFI": _supported("moderate", level_db=25.0),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "several_types_with_label"
        assert row["strongest_candidate"] == "BPFO"
        assert row["strongest_candidate_correct"] is False
        assert row["named_fault"] is None
        y2 = results["strata"]["Y2"]
        assert y2["outcomes"]["several_types_with_label"] == 1
        assert y2["named_fault"] == {"correct": 0, "wrong": 0, "none": 1, "total": 1}
        assert y2["indicated_precision"] == {"correct": 0, "total": 0, "rate": None}
        assert y2["sensitivity"] == {"alerts": 1, "total": 1, "rate": 1.0}
        assert y2["wrong_type_support"] == {"records": 1, "total": 1, "rate": 1.0}

    def test_wrong_indicated_on_a_partial_stratum_record(self):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="P2"),
            _labeled("cwru_002", 118, "outer_race", grade="P2"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFO": _supported("moderate")}),
            "cwru_002": _outcome("cwru_002", {"BPFO": _supported("moderate")}),
        }
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["outcome_class"] == "indicated_wrong"
        assert results["records"]["cwru_001"]["named_correct"] is False
        p2 = results["strata"]["P2"]
        assert p2["indicated_precision"] == {"correct": 1, "total": 2, "rate": 0.5}
        assert p2["named_fault"] == {"correct": 1, "wrong": 1, "none": 0, "total": 2}
        # A partial stratum never enters the headline.
        assert results["headline"]["n_records"] == 0

    def test_strongest_candidate_follows_the_check_not_the_ranking(self):
        """The check's first candidate (by level over the floor) and the
        secondary ranking (by evidence grade) can disagree; each metric
        reads its own."""
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _supported("moderate", level_db=40.0),
                    "BPFO": _supported("high", level_db=30.0),
                },
            )
        }
        row = _score(labeled, outcomes)["records"]["cwru_001"]
        assert row["strongest_candidate"] == "BPFI"
        assert row["strongest_candidate_correct"] is True
        assert row["ranked_first"] == "BPFO"
        assert row["classification_correct"] is False

    def test_several_types_without_the_label_on_a_fault_record(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="N1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFO": _supported("moderate", level_db=30.0),
                    "BSF": _supported("moderate", level_db=20.0),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "several_types_without_label"
        assert row["label_supported"] is False
        assert row["wrong_type_supported"] is True
        assert results["strata"]["N1"]["outcomes"]["several_types_without_label"] == 1

    def test_qualified_indicated_names_no_fault(self):
        """An indicated outcome under a qualified claim names no type: the
        user is told the readings it cannot exclude, not one fault."""
        labeled = (_labeled("cwru_001", 118, "outer_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001", {"BPFO": _supported("moderate")}, claim_level="qualified"
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["outcome_class"] == "indicated_correct"
        assert row["claim_level"] == "qualified"
        assert row["named_fault"] is None
        assert results["strata"]["Y1"]["named_fault"]["none"] == 1

    def test_confusion_table_counts_named_type_against_label(self):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 118, "outer_race", grade="Y1"),
            _labeled("cwru_003", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFO": _supported("moderate")}),
            "cwru_002": _outcome("cwru_002", {"BPFO": _supported("moderate")}),
            "cwru_003": _outcome("cwru_003", {"BPFO": _supported("moderate")}),
        }
        results = _score(labeled, outcomes)
        y1 = results["strata"]["Y1"]["confusion"]
        assert y1["inner_race"] == {"BPFO": 1, "BPFI": 0, "BSF": 0, "FTF": 0, "none": 0}
        assert y1["outer_race"]["BPFO"] == 1
        assert sum(y1["ball"].values()) == 0
        assert results["strata"]["ungraded"]["confusion"]["normal"]["BPFO"] == 1

    def test_every_rate_carries_its_denominator(self):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),
        }
        results = _score(labeled, outcomes)
        found = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                if "rate" in node:
                    found.append(path)
                    assert "total" in node, path
                for key, value in node.items():
                    walk(value, f"{path}.{key}")

        walk(results["strata"], "strata")
        walk(results["headline"], "headline")
        assert found


# ---------------------------------------------------------------------------
# Policy versions: outcomes measured under another policy are refused
# ---------------------------------------------------------------------------


class TestPolicyVersions:
    """The scorer refuses outcomes measured under another evidence or claim
    policy than the engine's current one."""

    @staticmethod
    def _pair(**versions: Any) -> tuple[tuple[LabeledRecord, ...], dict]:
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}, **versions)
        }
        return labeled, outcomes

    def test_current_versions_are_scored_and_recorded(self):
        labeled, outcomes = self._pair()
        results = _score(labeled, outcomes)
        assert results["criteria"]["policy_versions"] == {
            "evidence": EVIDENCE_VERSION,
            "claim": CLAIM_VERSION,
        }

    @pytest.mark.parametrize(
        "field", ["evidence_policy_version", "claim_policy_version"]
    )
    def test_an_old_policy_version_is_refused(self, field):
        labeled, outcomes = self._pair(**{field: 0})
        with pytest.raises(ValueError, match="cwru_001") as excinfo:
            _score(labeled, outcomes)
        assert "python -m benchmarks.cwru" in str(excinfo.value)

    def test_an_outcome_without_policy_versions_is_refused(self):
        labeled, outcomes = self._pair()
        del outcomes["cwru_001"]["bearing"]["evidence_policy_version"]
        with pytest.raises(ValueError, match="cwru_001"):
            _score(labeled, outcomes)


# ---------------------------------------------------------------------------
# Known anomalies: flagged reported line, never excluded
# ---------------------------------------------------------------------------


class TestKnownAnomalies:
    """Anomalous records are flagged in their own line, not dropped."""

    def test_flagged_line_present_and_record_kept_in_stratum(self):
        labeled = (
            _labeled(
                "cwru_001",
                105,
                "inner_race",
                grade="Y1",
                known_anomalies=("clipping",),
            ),
            _labeled("cwru_002", 118, "inner_race", grade="Y1"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002", {"BPFI": _supported("high")}),
        }
        results = _score(labeled, outcomes)

        flagged = results["known_anomalies_flagged"]
        assert flagged["record_ids"] == ["cwru_001"]
        assert flagged["n_records"] == 1
        assert flagged["classification"] == {"correct": 1, "total": 1, "rate": 1.0}
        assert results["records"]["cwru_001"]["known_anomalies"] is True
        # Flagged, NOT excluded: still counted in its stratum.
        assert results["strata"]["Y1"]["n_records"] == 2

    def test_no_flagged_records_reports_an_empty_line(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {"cwru_001": _outcome("cwru_001")}
        results = _score(labeled, outcomes)
        assert results["known_anomalies_flagged"]["record_ids"] == []
        assert results["known_anomalies_flagged"]["n_records"] == 0


# ---------------------------------------------------------------------------
# Fail closed: partial/degenerate input refuses, ids named
# ---------------------------------------------------------------------------


class TestFailClosed:
    """A partial or empty input is a refusal, never a score."""

    @pytest.fixture
    def pair(self):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),
        }
        return labeled, outcomes

    def test_missing_outcome_refused_naming_the_id(self, pair):
        labeled, outcomes = pair
        del outcomes["cwru_002"]
        with pytest.raises(ValueError, match="cwru_002"):
            _score(labeled, outcomes)

    def test_extra_outcome_refused_naming_the_id(self, pair):
        labeled, outcomes = pair
        outcomes["cwru_099"] = _outcome("cwru_099")
        with pytest.raises(ValueError, match="cwru_099"):
            _score(labeled, outcomes)

    def test_non_ok_outcome_is_a_refusal_not_a_skip(self, pair):
        labeled, outcomes = pair
        outcomes["cwru_002"] = _outcome("cwru_002", status="missing_signal")
        with pytest.raises(ValueError, match="cwru_002") as excinfo:
            _score(labeled, outcomes)
        assert "non-ok" in str(excinfo.value)

    def test_empty_outcomes_refused_never_a_report(self, pair):
        labeled, _ = pair
        with pytest.raises(ValueError, match="[Ee]mpty") as excinfo:
            _score(labeled, {})
        assert "benchmarks.cwru run" in str(excinfo.value)  # remedy

    def test_empty_labels_refused(self, pair):
        _, outcomes = pair
        with pytest.raises(ValueError, match="[Ll]abel"):
            _score((), outcomes)

    def test_duplicate_labeled_ids_refused(self, pair):
        labeled, outcomes = pair
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            _score((labeled[0], labeled[0], labeled[1]), outcomes)

    def test_ok_outcome_without_bearing_block_refused(self, pair):
        labeled, outcomes = pair
        del outcomes["cwru_002"]["bearing"]
        with pytest.raises(ValueError, match="cwru_002") as excinfo:
            _score(labeled, outcomes)
        assert "fault_checks" in str(excinfo.value)

    def test_ok_outcome_missing_a_fault_check_refused(self, pair):
        labeled, outcomes = pair
        del outcomes["cwru_002"]["bearing"]["fault_checks"]["BSF"]
        with pytest.raises(ValueError, match="BSF"):
            _score(labeled, outcomes)

    @pytest.mark.parametrize(
        "field", ["outcome", "candidates", "claim_level", "most_likely_fault"]
    )
    def test_ok_outcome_missing_a_summary_field_refused(self, pair, field):
        labeled, outcomes = pair
        del outcomes["cwru_001"]["bearing"][field]
        with pytest.raises(ValueError, match=f"cwru_001.*{field}"):
            _score(labeled, outcomes)

    def test_unknown_check_outcome_refused(self, pair):
        labeled, outcomes = pair
        outcomes["cwru_001"]["bearing"]["outcome"] = "withheld"
        with pytest.raises(ValueError, match="Unknown check outcome 'withheld'"):
            _score(labeled, outcomes)

    def test_unknown_metadata_override_refused(self, pair):
        labeled, outcomes = pair
        with pytest.raises(ValueError, match="gti_describe"):
            scorer.score_results(
                outcomes, labeled, metadata_overrides={"gti_describe": "typo"}
            )

    def test_read_outcomes_refuses_a_missing_artifact(self, tmp_path):
        missing = tmp_path / "outcomes.json"
        with pytest.raises(ValueError, match="benchmarks.cwru run") as excinfo:
            scorer.read_outcomes(missing)
        assert str(missing) in str(excinfo.value)

    def test_read_outcomes_refuses_a_non_object_artifact(self, tmp_path):
        bad = tmp_path / "outcomes.json"
        bad.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            scorer.read_outcomes(bad)


# ---------------------------------------------------------------------------
# Metadata and the canonical results write
# ---------------------------------------------------------------------------


class TestMetadataAndWrite:
    """Varying metadata is injectable; the write reuses the runner's
    canonical serialization."""

    def test_overrides_pin_varying_values_and_keep_collected_ones(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {"cwru_001": _outcome("cwru_001")}
        results = _score(labeled, outcomes)
        metadata = results["metadata"]
        assert set(metadata) == set(scorer.METADATA_KEYS)
        assert metadata["date"] == META["date"]
        assert metadata["git_describe"] == META["git_describe"]
        assert metadata["platform"] == META["platform"]
        assert metadata["dataset_subset"] == scorer.DATASET_SUBSET
        assert metadata["pipeline_version"] == predictive_maintenance_mcp.__version__
        assert metadata["numpy_version"] == np.__version__
        assert metadata["scipy_version"] == scipy.__version__

    def test_pinned_metadata_makes_serialization_deterministic(self):
        labeled = (_labeled("cwru_001", 105, "ball", grade="Y2"),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BSF": _supported("moderate")})}
        first = runner.serialize_outcomes(_score(labeled, outcomes))
        second = runner.serialize_outcomes(_score(labeled, outcomes))
        assert first == second

    def test_criteria_embedded_for_audit(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        results = _score(labeled, {"cwru_001": _outcome("cwru_001")})
        criteria = results["criteria"]
        assert criteria["hit_state"] == "supported"
        assert "harmonic_leg" not in criteria
        assert list(criteria["outcome_classes"]) == list(scorer.OUTCOME_CLASSES)
        assert criteria["label_to_check_key"] == {
            "inner_race": "BPFI",
            "ball": "BSF",
            "outer_race": "BPFO",
        }

    def test_every_reported_metric_has_its_definition(self):
        """The methodology says every definition is recorded in the
        criteria block: each metric family of a group has an entry there
        (the outcome classes and the secondary ranking under their own
        keys)."""
        labeled = (_labeled("cwru_001", 97, "normal"),)
        results = _score(labeled, {"cwru_001": _outcome("cwru_001")})
        families = set(results["headline"]) - {
            "strata_included",
            "n_records",
            "n_faulted",
            "n_normal",
        }
        covered_elsewhere = {
            "outcomes": "outcome_classes",
            "classification": "ranking_rule",
            "end_to_end": "ranking_rule",
        }
        for family in sorted(families):
            assert covered_elsewhere.get(family, family) in results["criteria"], family

    def test_write_results_is_the_canonical_atomic_write(self, tmp_path):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        results = _score(labeled, {"cwru_001": _outcome("cwru_001")})
        target = tmp_path / "results" / "results.json"

        written = scorer.write_results(results, target)

        assert written == target
        raw = target.read_bytes()
        assert raw == runner.serialize_outcomes(results)
        assert raw.endswith(b"\n")
        assert not target.with_name(target.name + ".part").exists()
        parsed = json.loads(raw.decode("utf-8"))
        assert list(parsed) == sorted(parsed)


# ---------------------------------------------------------------------------
# CLI: the score stage end-to-end on tmp paths
# ---------------------------------------------------------------------------


class TestScoreCLI:
    """`python -m benchmarks.cwru score` reads outcomes, writes results."""

    def test_score_happy_path(self, tmp_path: Path, monkeypatch, capsys):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),
        }
        outcomes_path = tmp_path / "outcomes.json"
        results_path = tmp_path / "results.json"
        runner.write_outcomes(outcomes, outcomes_path)
        # The CLI scores against the vendored 64-record tables by
        # default; point it at the fixture set and pin the metadata.
        monkeypatch.setattr(scorer, "label_view", lambda: labeled)
        monkeypatch.setattr(
            scorer, "collect_metadata", lambda overrides=None: dict(META)
        )

        exit_code = main(
            [
                "score",
                "--outcomes",
                str(outcomes_path),
                "--output",
                str(results_path),
            ]
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "2 record(s)" in out
        parsed = json.loads(results_path.read_text(encoding="utf-8"))
        assert set(parsed["records"]) == {"cwru_001", "cwru_002"}
        assert parsed["strata"]["Y1"]["classification"]["correct"] == 1
        assert parsed["metadata"]["git_describe"] == META["git_describe"]
        # The flat artifact predates provenance: the key is present, null.
        assert parsed["measurement_provenance"] is None

    def test_score_echoes_measurement_provenance_from_the_artifact(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),
        }
        provenance = {
            "date": "2026-08-10T00:00:00+00:00",
            "git_describe": "pin",
            "platform": "pin",
        }
        outcomes_path = tmp_path / "outcomes.json"
        results_path = tmp_path / "results.json"
        runner.write_outcomes(
            runner.compose_outcomes_document(outcomes, provenance), outcomes_path
        )
        monkeypatch.setattr(scorer, "label_view", lambda: labeled)
        monkeypatch.setattr(
            scorer, "collect_metadata", lambda overrides=None: dict(META)
        )

        exit_code = main(
            [
                "score",
                "--outcomes",
                str(outcomes_path),
                "--output",
                str(results_path),
            ]
        )

        assert exit_code == 0
        parsed = json.loads(results_path.read_text(encoding="utf-8"))
        assert parsed["measurement_provenance"] == provenance

    def test_score_partial_outcomes_exit_2_naming_ids(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")})}
        outcomes_path = tmp_path / "outcomes.json"
        results_path = tmp_path / "results.json"
        runner.write_outcomes(outcomes, outcomes_path)
        monkeypatch.setattr(scorer, "label_view", lambda: labeled)

        exit_code = main(
            [
                "score",
                "--outcomes",
                str(outcomes_path),
                "--output",
                str(results_path),
            ]
        )

        assert exit_code == 2
        assert "cwru_002" in capsys.readouterr().err
        assert not results_path.exists()


class TestCommittedResultsDerivedFromOutcomes:
    """The committed results.json is re-derivable from the committed outcomes.

    The drift guard binds the README numbers to results.json; this binds
    results.json to outcomes.json, so the whole README -> results ->
    outcomes chain is tamper-evident. A hand-edited results.json turns
    this red even when the README still agrees with it.

    Reads only the two committed artifacts: no network, no CWRU cache.
    """

    @staticmethod
    def _committed_results_text() -> str:
        return scorer.DEFAULT_RESULTS_PATH.read_text(encoding="utf-8")

    @staticmethod
    def _committed_outcomes_text() -> str:
        return runner.DEFAULT_OUTCOMES_PATH.read_text(encoding="utf-8")

    @staticmethod
    def _rescore(tmp_path: Path) -> str:
        """Re-score the committed outcomes and return the serialized bytes.

        Metadata is pinned from the committed artifact itself rather than
        from a literal: date, git describe, platform and library versions
        record the machine that produced the artifact, so collecting them
        here would compare this run's environment against the maintainer's
        and flap on every machine. Everything outside ``metadata`` is
        recomputed for real.
        """
        committed = json.loads(
            TestCommittedResultsDerivedFromOutcomes._committed_results_text()
        )
        outcomes, provenance = scorer.read_outcomes()
        results = scorer.score_results(
            outcomes,
            metadata_overrides=committed["metadata"],
            measurement_provenance=provenance,
        )
        out = tmp_path / "rescored.json"
        scorer.write_results(results, out)
        return out.read_text(encoding="utf-8")

    def test_rescoring_committed_outcomes_reproduces_results_byte_for_byte(
        self, tmp_path: Path
    ):
        assert self._rescore(tmp_path) == self._committed_results_text()

    def test_every_metadata_key_is_pinned_by_the_committed_artifact(self):
        """The override set above must cover every environment-dependent key.

        If a new metadata key is added and not present in the committed
        artifact, the guard above would silently start collecting it from
        this machine — and pass locally for the maintainer while flapping
        in CI.
        """
        committed = json.loads(self._committed_results_text())
        assert set(committed["metadata"]) == set(scorer.METADATA_KEYS)

    def test_every_provenance_key_is_pinned_by_the_committed_artifact(self):
        """The committed outcomes' provenance keys must equal PROVENANCE_KEYS.

        If a new provenance key is added without regenerating the
        committed outcomes artifact, the derivation guard would silently
        drop it. The guard is dormant while the committed artifact
        predates provenance collection (no ``_provenance`` marker) and
        becomes live once the maintainer re-measures.
        """
        committed = json.loads(self._committed_outcomes_text())
        if runner.PROVENANCE_MARKER_KEY not in committed:
            pytest.skip(
                "committed outcomes predate provenance collection; "
                "regenerate to arm this guard"
            )
        assert set(committed[runner.PROVENANCE_MARKER_KEY]) == set(
            runner.PROVENANCE_KEYS
        )

    def test_a_perturbed_results_copy_fails_the_comparison(self, tmp_path: Path):
        """Mutation leg: the comparison is demonstrated to fail, not assumed to."""
        rescored = self._rescore(tmp_path)
        perturbed = json.loads(rescored)
        before = perturbed["headline"]["n_records"]
        perturbed["headline"]["n_records"] = before + 1

        perturbed_path = tmp_path / "perturbed.json"
        scorer.write_results(perturbed, perturbed_path)
        perturbed_text = perturbed_path.read_text(encoding="utf-8")

        assert perturbed_text != self._committed_results_text()
        # The divergence is nameable, not just a byte count.
        differing = [
            key
            for key in json.loads(perturbed_text)["headline"]
            if json.loads(rescored)["headline"].get(key)
            != json.loads(perturbed_text)["headline"][key]
        ]
        assert differing == ["n_records"]

    def test_guard_needs_no_cache(self):
        """Only the two committed artifacts are read."""
        assert scorer.DEFAULT_RESULTS_PATH.exists()
        assert runner.DEFAULT_OUTCOMES_PATH.exists()

    def test_committed_outcomes_follow_the_published_check_contract(self):
        """The criteria describe what the check does; the hand-built
        fixtures above assume it. Pin it on the measured outcomes: the
        candidates are listed by level over the floor, the outcome follows
        their count, and a fault type is named only at single-spectrum
        evidence."""
        outcomes, _ = scorer.read_outcomes()
        assert outcomes
        for opaque_id, outcome in outcomes.items():
            bearing = outcome["bearing"]
            levels = [
                bearing["fault_checks"][key]["level_db"]
                for key in bearing["candidates"]
            ]
            assert levels == sorted(levels, reverse=True), opaque_id
            count = len(bearing["candidates"])
            if count == 1:
                assert bearing["outcome"] == "indicated", opaque_id
            elif count > 1:
                assert bearing["outcome"] == "several_types", opaque_id
            else:
                assert bearing["outcome"] in ("inconclusive", "no_indication")
            named_level = bearing["claim_level"] == "single_spectrum_evidence"
            assert (bearing["most_likely_fault"] is not None) == named_level, opaque_id


# ---------------------------------------------------------------------------
# Measurement provenance: read split, results echo, schema-stable null
# ---------------------------------------------------------------------------


class TestMeasurementProvenance:
    """Provenance round-trips from the artifact into the results.

    ``read_outcomes`` splits the ``_provenance`` block from the record
    ids; ``score_results`` echoes it verbatim into
    ``measurement_provenance``, which is always present — ``None`` for
    artifacts that predate provenance collection.
    """

    @staticmethod
    def _pair():
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 97, "normal"),
        )
        outcomes = {
            "cwru_001": _outcome("cwru_001", {"BPFI": _supported("high")}),
            "cwru_002": _outcome("cwru_002"),
        }
        return labeled, outcomes

    def test_read_outcomes_round_trips_provenance(self, tmp_path):
        _, outcomes = self._pair()
        provenance = {
            "date": "2026-08-10T00:00:00+00:00",
            "git_describe": "pin",
            "platform": "pin",
        }
        path = tmp_path / "outcomes.json"
        runner.write_outcomes(
            runner.compose_outcomes_document(outcomes, provenance), path
        )
        records, block = scorer.read_outcomes(path)
        assert records == outcomes
        assert block == provenance

    def test_read_outcomes_legacy_artifact_returns_none(self, tmp_path):
        _, outcomes = self._pair()
        path = tmp_path / "outcomes.json"
        runner.write_outcomes(outcomes, path)
        records, block = scorer.read_outcomes(path)
        assert records == outcomes
        assert block is None

    def test_measurement_provenance_echoed_verbatim(self):
        labeled, outcomes = self._pair()
        provenance = {
            "date": "2026-08-10T00:00:00+00:00",
            "git_describe": "pin",
            "platform": "pin",
        }
        results = scorer.score_results(
            outcomes,
            labeled,
            metadata_overrides=META,
            measurement_provenance=provenance,
        )
        assert results["measurement_provenance"] == provenance

    def test_measurement_provenance_null_when_omitted(self):
        labeled, outcomes = self._pair()
        results = scorer.score_results(outcomes, labeled, metadata_overrides=META)
        assert "measurement_provenance" in results
        assert results["measurement_provenance"] is None

    def test_empty_provenance_mapping_honored_verbatim(self):
        labeled, outcomes = self._pair()
        results = scorer.score_results(
            outcomes, labeled, metadata_overrides=META, measurement_provenance={}
        )
        assert results["measurement_provenance"] == {}
