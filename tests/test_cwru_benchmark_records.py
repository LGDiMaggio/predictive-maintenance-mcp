"""U1 tests: vendored CWRU record tables, models, and access views.

Covers the blind-protocol trust boundary at the data level (origin
acceptance example AE3): the ops table parsed by downloader, importer, and
runner must carry no label-bearing key, and the guard refusing one is
demonstrated to fail when a label key is injected (mutation guard).

No network is touched: everything runs against the vendored JSON tables.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks.cwru.records import (
    LABEL_BEARING_KEYS,
    LABELS_PATH,
    OPS_ALLOWED_KEYS,
    RECORDS_OPS_PATH,
    LabeledRecord,
    LabelRecord,
    OpsRecord,
    label_view,
    ops_view,
)

OPAQUE_ID_PATTERN = re.compile(r"^cwru_\d{3}$")

#: Nominal shaft speed per motor load (CWRU-published approximate speeds).
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def _raw_ops() -> list[dict[str, Any]]:
    with open(RECORDS_OPS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _raw_labels() -> dict[str, dict[str, Any]]:
    with open(LABELS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _write_ops_variant(
    tmp_path: Path, mutate: Callable[[list[dict[str, Any]]], None]
) -> Path:
    """Copy the real ops table, apply *mutate*, and write it under tmp_path."""
    entries = _raw_ops()
    mutate(entries)
    out = tmp_path / "records_ops.json"
    out.write_text(json.dumps(entries), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Happy path: both tables load, validate, and cover the v1 subset
# ---------------------------------------------------------------------------


class TestTablesLoadAndValidate:
    """Both vendored tables load and every record validates."""

    def test_both_tables_load_and_align(self):
        ops = ops_view()
        labeled = label_view()
        assert len(ops) == 64
        assert len(labeled) == 64
        # label_view preserves ops-table order
        assert [rec.ops.opaque_id for rec in labeled] == [rec.opaque_id for rec in ops]

    def test_opaque_ids_unique_and_well_formed(self):
        ids = [record.opaque_id for record in ops_view()]
        assert len(set(ids)) == len(ids), "opaque ids must be unique"
        for opaque_id in ids:
            assert OPAQUE_ID_PATTERN.match(opaque_id), (
                f"opaque id {opaque_id!r} does not match "
                f"{OPAQUE_ID_PATTERN.pattern!r}"
            )

    def test_bijection_between_tables(self):
        """Every labels.json key has an ops record and vice versa."""
        ops_ids = {entry["opaque_id"] for entry in _raw_ops()}
        label_ids = set(_raw_labels())
        assert ops_ids == label_ids

    def test_v1_subset_counts(self):
        """Counts match the official CWRU group tables.

        The 12k drive-end table has known gaps: 0.014" outer race exists
        only centered @6:00, and 0.028" has no outer-race records at all.
        """
        labeled = label_view()
        by_type = Counter(rec.label.fault_type for rec in labeled)
        assert by_type == {
            "inner_race": 16,
            "ball": 16,
            "outer_race": 28,
            "normal": 4,
        }

        faults = [rec for rec in labeled if rec.label.fault_type != "normal"]
        assert len(faults) == 60
        by_diameter = Counter(rec.label.fault_diameter_in for rec in faults)
        assert by_diameter == {0.007: 20, 0.014: 12, 0.021: 20, 0.028: 8}

        by_or_position = Counter(
            rec.label.or_position
            for rec in faults
            if rec.label.fault_type == "outer_race"
        )
        assert by_or_position == {
            "centered_6": 12,
            "orthogonal_3": 8,
            "opposite_12": 8,
        }

    def test_sampling_rates_and_nominal_rpm(self):
        """Fault group at 12 kHz, normal baselines at 48 kHz; rpm by load."""
        for rec in label_view():
            expected_fs = 48000 if rec.label.fault_type == "normal" else 12000
            assert rec.ops.fs_hz == expected_fs, rec.ops.opaque_id
            assert (
                rec.ops.nominal_rpm == RPM_BY_LOAD[rec.ops.load_hp]
            ), rec.ops.opaque_id

    def test_url_and_cache_filename_derive_from_file_id(self):
        for record in ops_view():
            assert record.cache_filename == f"{record.file_id}.mat"
            assert record.url == (
                "https://engineering.case.edu/sites/default/files/"
                f"{record.file_id}.mat"
            )


# ---------------------------------------------------------------------------
# Blindness (AE3): ops table carries operational keys only
# ---------------------------------------------------------------------------


class TestOpsBlindness:
    """The ops table is exactly the allowlist — no label-bearing key."""

    def test_ops_entries_carry_exactly_the_allowlist(self):
        for index, entry in enumerate(_raw_ops()):
            assert set(entry) == OPS_ALLOWED_KEYS, (
                f"records_ops.json entry {index} keys diverge from the " f"allowlist"
            )
        # The model mirrors the allowlist, so a field added to one without
        # the other turns red here.
        assert set(OpsRecord.model_fields) == OPS_ALLOWED_KEYS

    def test_file_level_key_disjointness(self):
        """No label-bearing key appears anywhere in records_ops.json."""
        assert OPS_ALLOWED_KEYS.isdisjoint(LABEL_BEARING_KEYS)
        for index, entry in enumerate(_raw_ops()):
            leaked = LABEL_BEARING_KEYS.intersection(entry)
            assert not leaked, (
                f"records_ops.json entry {index} carries label-bearing "
                f"key(s) {sorted(leaked)}"
            )

    def test_label_entries_carry_exactly_the_label_keys(self):
        for opaque_id, entry in _raw_labels().items():
            assert set(entry) == LABEL_BEARING_KEYS, opaque_id
        assert set(LabelRecord.model_fields) == LABEL_BEARING_KEYS


# ---------------------------------------------------------------------------
# Edge case: null S&R grade -> ungraded stratum
# ---------------------------------------------------------------------------


class TestUngradedGrades:
    """Null sr2015_grade validates and is reported as ungraded."""

    def test_null_grade_validates_and_reports_ungraded(self):
        record = LabeledRecord(
            ops=ops_view()[0],
            label=LabelRecord(
                fault_type="inner_race",
                or_position=None,
                fault_diameter_in=0.007,
                sr2015_grade=None,
                known_anomalies=[],
            ),
        )
        assert record.label.sr2015_grade is None
        assert record.grade_stratum == "ungraded"

    def test_vendored_table_grades_are_complete_for_faults(self):
        """Every fault record carries a transcribed S&R 2015 grade (Table B2,
        DE channel, best-of-methods); normal baselines are not fault-graded
        and land in the visible 'ungraded' stratum, never disappearing."""
        for rec in label_view():
            if rec.label.fault_type == "normal":
                assert rec.label.sr2015_grade is None
                assert rec.grade_stratum == "ungraded"
            else:
                assert rec.label.sr2015_grade in {"Y1", "Y2", "P1", "P2", "N1", "N2"}
                assert rec.grade_stratum == rec.label.sr2015_grade


# ---------------------------------------------------------------------------
# Error path: path-hostile names rejected at load
# ---------------------------------------------------------------------------


class TestPathHostileNames:
    """Names from the ops table pass validate_name_component at load."""

    @pytest.mark.parametrize("field", ["opaque_id", "cache_filename"])
    @pytest.mark.parametrize("hostile", ["../evil", ".", "a/b.mat"])
    def test_hostile_name_rejected(self, tmp_path, field, hostile):
        def mutate(entries: list[dict[str, Any]]) -> None:
            entries[0][field] = hostile

        mutated = _write_ops_variant(tmp_path, mutate)
        with pytest.raises(ValueError, match="failed validation"):
            ops_view(mutated)

    @pytest.mark.skipif(os.name != "nt", reason="Windows path separator")
    @pytest.mark.parametrize("field", ["opaque_id", "cache_filename"])
    def test_backslash_name_rejected_on_windows(self, tmp_path, field):
        def mutate(entries: list[dict[str, Any]]) -> None:
            entries[0][field] = "..\\evil"

        mutated = _write_ops_variant(tmp_path, mutate)
        with pytest.raises(ValueError, match="failed validation"):
            ops_view(mutated)


# ---------------------------------------------------------------------------
# Mutation guard: injecting a label key must trip the disjointness guard
# ---------------------------------------------------------------------------


class TestMutationGuard:
    """The disjointness guard is demonstrated to fail, not assumed to."""

    def test_injected_label_key_trips_the_guard(self, tmp_path):
        def mutate(entries: list[dict[str, Any]]) -> None:
            entries[0]["fault_type"] = "inner_race"

        mutated = _write_ops_variant(tmp_path, mutate)
        with pytest.raises(ValueError, match="fault_type") as excinfo:
            ops_view(mutated)
        message = str(excinfo.value)
        assert "label-bearing" in message
        assert "labels.json" in message  # remedy names where labels belong

    def test_unmutated_variant_still_loads(self, tmp_path):
        """The tmp-path plumbing itself is sound: without the injected key
        the same variant loads — so the red above is the guard, not the
        harness."""
        clean = _write_ops_variant(tmp_path, lambda entries: None)
        assert len(ops_view(clean)) == 64
