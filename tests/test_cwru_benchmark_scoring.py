"""U5 tests: the scorer over hand-built outcomes and tiny label tables.

Everything is hand-computable: fixtures are in-test outcome dicts plus a
small synthetic labeled-record set (never the real 64-record tables), so
every rate asserted here can be checked by counting on paper. Coverage:

- happy path: hand-computed per-stratum counts AND rates reproduced
  exactly (3 of 4 Y1 hits);
- N1/N2 land in their own strata and never move headline (Y1/Y2)
  accuracy;
- the BSF harmonic leg: harmonic 2 present hits despite
  ``detected=False`` / low evidence; no harmonic 2 misses;
- symmetry: the SAME criterion counts false positives on normal
  baselines (harmonic leg included; low/none evidence does not count);
- classification ranking (documented rule: evidence tier high >
  moderate > harmonic-only, then magnitude descending, then
  |deviation_pct| ascending, then fixed BPFO/BPFI/BSF/FTF order) —
  ranked-first equal to the label is correct, a stronger wrong fault is
  incorrect;
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


def _detected(evidence: str = "moderate", **overrides: Any) -> dict[str, Any]:
    """Check overrides for a detected fundamental at *evidence*."""
    values: dict[str, Any] = {
        "detected": True,
        "detected_frequency_hz": 100.0,
        "magnitude": 0.5,
        "deviation_pct": 0.5,
        "evidence_strength": evidence,
    }
    values.update(overrides)
    return values


def _outcome(
    opaque_id: str,
    checks: Optional[dict[str, dict[str, Any]]] = None,
    status: str = "ok",
) -> dict[str, Any]:
    """A hand-built runner outcome (only the fields the scorer reads)."""
    if status != runner.OUTCOME_STATUS_OK:
        return {
            "signal_id": opaque_id,
            "status": status,
            "error": "synthetic non-ok outcome",
        }
    fault_checks = {
        key: _check(key, **(checks or {}).get(key, {})) for key in scorer.FAULT_KEYS
    }
    return {
        "signal_id": opaque_id,
        "status": runner.OUTCOME_STATUS_OK,
        "bearing": {"fault_checks": fault_checks},
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
        """4 Y1 faulted records (3 hits) + 1 clean ungraded normal."""
        labeled = (
            _labeled("cwru_001", 105, "inner_race", grade="Y1"),
            _labeled("cwru_002", 118, "outer_race", grade="Y1"),
            _labeled("cwru_003", 130, "ball", grade="Y1"),
            _labeled("cwru_004", 144, "inner_race", grade="Y1"),
            _labeled("cwru_005", 97, "normal"),
        )
        outcomes = {
            # Hit: BPFI detected, high evidence.
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
            # Hit: BPFO detected, moderate evidence.
            "cwru_002": _outcome("cwru_002", {"BPFO": _detected("moderate")}),
            # Hit via the harmonic leg only: BSF not detected, low
            # evidence, harmonic 2 present.
            "cwru_003": _outcome(
                "cwru_003",
                {
                    "BSF": {
                        "evidence_strength": "low",
                        "harmonics_detected": [_harmonic(order=2)],
                    }
                },
            ),
            # Miss: nothing found anywhere.
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
        assert y1["classification"] == {"correct": 3, "total": 4, "rate": 0.75}
        assert y1["end_to_end"] == {"correct": 3, "total": 4, "rate": 0.75}

    def test_normal_lands_in_ungraded_with_no_false_positive(self, scored):
        ungraded = scored["strata"]["ungraded"]
        assert ungraded["n_records"] == 1
        assert ungraded["n_normal"] == 1
        assert ungraded["false_positives"]["records_with_any"] == 0
        assert ungraded["false_positives"]["total_normal"] == 1
        assert ungraded["false_positives"]["rate"] == 0.0
        assert ungraded["classification"] == {"correct": 1, "total": 1, "rate": 1.0}

    def test_headline_covers_exactly_y1_y2(self, scored):
        headline = scored["headline"]
        assert headline["strata_included"] == ["Y1", "Y2"]
        # Only the 4 Y1 records — the ungraded normal is not headline.
        assert headline["n_records"] == 4
        assert headline["classification"] == {"correct": 3, "total": 4, "rate": 0.75}

    def test_per_record_table_rows(self, scored):
        records = scored["records"]
        assert set(records) == {f"cwru_00{i}" for i in range(1, 6)}
        assert records["cwru_001"]["ranked_first"] == "BPFI"
        assert records["cwru_001"]["classification_correct"] is True
        assert records["cwru_003"]["ranked_first"] == "BSF"
        assert records["cwru_003"]["frequency_detection_hit"] is True
        assert records["cwru_004"]["ranked_first"] is None
        assert records["cwru_004"]["classification_correct"] is False
        assert records["cwru_005"]["expected_fault"] is None
        assert records["cwru_005"]["false_positive"] is False
        for row in records.values():
            assert row["end_to_end_correct"] is row["classification_correct"]

    def test_empty_strata_report_zero_counts_and_null_rates(self, scored):
        # Every stratum is always present; empty ones are honestly empty.
        assert set(scored["strata"]) == set(scorer.STRATA)
        p1 = scored["strata"]["P1"]
        assert p1["n_records"] == 0
        assert p1["frequency_detection"] == {"hits": 0, "total": 0, "rate": None}
        assert p1["classification"]["rate"] is None


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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
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
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFO": _detected("high")})}
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["stratum"] == "ungraded"
        assert results["strata"]["ungraded"]["frequency_detection"]["hits"] == 1
        assert results["headline"]["n_records"] == 0


# ---------------------------------------------------------------------------
# Harmonic leg: BSF-only, harmonic 2 exactly
# ---------------------------------------------------------------------------


class TestHarmonicLeg:
    """2xBSF alignment: harmonic 2 hits for BSF and for nothing else."""

    def test_ball_harmonic_two_hits_despite_low_evidence(self):
        labeled = (_labeled("cwru_001", 118, "ball", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BSF": {
                        "detected": False,
                        "evidence_strength": "low",
                        "harmonics_detected": [_harmonic(order=2)],
                    }
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["frequency_detection_hit"] is True
        assert row["ranked_first"] == "BSF"
        assert row["classification_correct"] is True

    def test_ball_without_harmonic_two_misses(self):
        """Energy at an unrelated frequency, no 2x harmonic: a miss."""
        labeled = (_labeled("cwru_001", 118, "ball", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    # A 3rd harmonic alone is NOT the leg's membership test.
                    "BSF": {"harmonics_detected": [_harmonic(order=3)]},
                    # An unrelated detection elsewhere does not help ball.
                    "BPFO": _detected("moderate"),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["frequency_detection_hit"] is False
        assert row["ranked_first"] == "BPFO"
        assert row["classification_correct"] is False

    def test_harmonic_leg_is_bsf_only(self):
        """A 2x harmonic on FTF is not a hit — no other fault has the leg."""
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {"FTF": {"harmonics_detected": [_harmonic(order=2)]}},
            )
        }
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["false_positive"] is False


# ---------------------------------------------------------------------------
# Symmetry: the SAME criterion counts false positives on normals
# ---------------------------------------------------------------------------


class TestSymmetryOnNormals:
    """No asymmetry: whatever detects a fault also counts as an FP."""

    def test_normal_with_bsf_harmonic_two_is_false_positive(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {"BSF": {"harmonics_detected": [_harmonic(order=2)]}},
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["false_positive"] is True
        assert row["classification_correct"] is False
        fp = results["strata"]["ungraded"]["false_positives"]
        assert fp["records_with_any"] == 1
        assert fp["rate"] == 1.0
        assert fp["per_fault_counts"] == {"BPFO": 0, "BPFI": 0, "BSF": 1, "FTF": 0}

    def test_normal_with_moderate_detection_is_false_positive(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFI": _detected("moderate")})}
        results = _score(labeled, outcomes)
        assert results["records"]["cwru_001"]["false_positive"] is True
        fp = results["strata"]["ungraded"]["false_positives"]
        assert fp["per_fault_counts"]["BPFI"] == 1

    def test_normal_with_low_or_none_evidence_is_not_false_positive(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    # detected but sub-moderate evidence: below criterion.
                    "BPFO": {
                        "detected": True,
                        "detected_frequency_hz": 100.0,
                        "evidence_strength": "low",
                    },
                    # harmonic 3 only, no fundamental: below criterion.
                    "BSF": {"harmonics_detected": [_harmonic(order=3)]},
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["false_positive"] is False
        assert row["classification_correct"] is True


# ---------------------------------------------------------------------------
# Classification ranking (documented rule)
# ---------------------------------------------------------------------------


class TestClassificationRanking:
    """Rule: evidence tier (high > moderate > harmonic-only), then
    magnitude descending, then |deviation_pct| ascending, then fixed
    BPFO/BPFI/BSF/FTF order."""

    def test_ranked_first_equal_to_label_is_correct(self):
        labeled = (_labeled("cwru_001", 105, "inner_race", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BPFI": _detected("high"),
                    "BPFO": _detected("moderate"),  # weaker tier
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
                    "BPFI": _detected("moderate"),
                    "BPFO": _detected("high"),  # stronger, wrong
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
                    "BPFI": _detected("moderate", magnitude=0.4),
                    "BPFO": _detected("moderate", magnitude=0.9),
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
                    "BPFI": _detected("moderate", magnitude=0.5, deviation_pct=0.3),
                    "BPFO": _detected("moderate", magnitude=0.5, deviation_pct=-2.0),
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
                    "BPFI": _detected("moderate"),
                    "BSF": _detected("moderate"),
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

    def test_detected_fundamental_outranks_harmonic_only(self):
        labeled = (_labeled("cwru_001", 118, "ball", grade="Y1"),)
        outcomes = {
            "cwru_001": _outcome(
                "cwru_001",
                {
                    "BSF": {"harmonics_detected": [_harmonic(order=2)]},
                    "BPFI": _detected("moderate"),
                },
            )
        }
        results = _score(labeled, outcomes)
        row = results["records"]["cwru_001"]
        assert row["frequency_detection_hit"] is True  # harmonic leg
        assert row["ranked_first"] == "BPFI"  # primary leg outranks
        assert row["classification_correct"] is False


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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
            "cwru_002": _outcome("cwru_002", {"BPFI": _detected("high")}),
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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
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
        outcomes = {"cwru_001": _outcome("cwru_001", {"BSF": _detected("moderate")})}
        first = runner.serialize_outcomes(_score(labeled, outcomes))
        second = runner.serialize_outcomes(_score(labeled, outcomes))
        assert first == second

    def test_criteria_embedded_for_audit(self):
        labeled = (_labeled("cwru_001", 97, "normal"),)
        results = _score(labeled, {"cwru_001": _outcome("cwru_001")})
        criteria = results["criteria"]
        assert criteria["harmonic_leg"] == {"fault": "BSF", "harmonic": 2}
        assert criteria["hit_evidence_levels"] == ["high", "moderate"]
        assert criteria["label_to_check_key"] == {
            "inner_race": "BPFI",
            "ball": "BSF",
            "outer_race": "BPFO",
        }

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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
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
        outcomes = {"cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")})}
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
        if provenance is None:
            results.pop("measurement_provenance")
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
            return
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
            "cwru_001": _outcome("cwru_001", {"BPFI": _detected("high")}),
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
