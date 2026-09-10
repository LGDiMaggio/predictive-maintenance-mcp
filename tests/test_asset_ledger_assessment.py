"""Tests for asset_ledger.assessment (U6) — ISO 13374 Block 3.

Every scenario is built from NUMBERS, not signals: the fixture helpers
fabricate ``measurement_recorded`` and ``health_snapshot_computed``
payloads with chosen indicator values, wrap them in real envelopes with
``make_event`` and project them with the real ``build_asset_view``, so the
assessment is exercised on exactly the view the store would produce. From
the plan's U6 scenarios:

- AE6 (automatic window wording and the ids of the first 10 slots), AE7
  (a declared baseline citing ``declared_by`` and using only its members),
  AE8 (spike -> unconfirmed, return -> isolated, ramp -> persistent with the
  criterion text and the onset id);
- acquisition slots (one spike exported three times), the k-of-n and drift
  persistence rules, a sustained decrease (never "improved"), a reference
  with sigma 0 (the 25% floor), insufficient and provisional histories, the
  requested window (the reference is always used);
- AE3 (undeclared rpm qualifies and participates), AE4 (a velocity
  measurement is excluded and listed), AE5 (lineage counts, the newest
  complete lineage wins, a newest lineage with holes is never used);
- baseline members superseded / non-comparable, too few members (never a
  silent fallback), members without the current lineage;
- the onset coinciding with a sensor change, the bearing evidence
  presence and persistence, the envelope band of an emerging line;
- the error paths (unknown point, unknown asset: closed oracle), the
  parameter validation, determinism, the module purity.
"""

import ast
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from predictive_maintenance_mcp.asset_ledger import assessment
from predictive_maintenance_mcp.asset_ledger.assessment import (
    AMPLITUDE_INDICATORS,
    ASSESSMENT_STATUSES,
    CLASSIFICATIONS,
    MIN_USABLE_SLOTS,
    AssessmentParams,
    assess_change,
    collect_point_slots,
)
from predictive_maintenance_mcp.asset_ledger.snapshot import processing_id
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_BASELINE_DECLARED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    build_asset_view,
    canonical_json,
    make_event,
    short_id,
)

ASSET = "P-101"
POINT = "motor_de_h"
LINEAGE_A = "health_snapshot/1+aaaaaaaaaaaaaaaa"
LINEAGE_B = "health_snapshot/1+bbbbbbbbbbbbbbbb"
FIRST_ACQUIRED = datetime(2026, 1, 5, 9, 0, tzinfo=timezone(timedelta(hours=1)))
WEEK = timedelta(weeks=1)

#: Ten stable values: mean about 1.0, sigma about 0.012, so the 25% floor
#: sets the band ([0.75, 1.25]).
STABLE = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0, 1.01]
SPIKE = 1.8
RAMP = [1.10 + 0.5 * step / 9 for step in range(10)]  # +10% .. +60%
REPROCESS_CALL = (
    "assess_asset_change(asset_id='P-101', measurement_point_id='motor_de_h', "
    "reprocess=True)"
)


# ---------------------------------------------------------------------------
# Fixture helpers: payloads by numbers, envelopes and views by the real code
# ---------------------------------------------------------------------------


def mid(index: int) -> str:
    return f"{index:016x}"


def acquired(index: int) -> str:
    return (FIRST_ACQUIRED + (index - 1) * WEEK).isoformat()


def point_event(
    version: int = 1, *, recorded_at: Optional[datetime] = None, **overrides: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "measurement_point_id": POINT,
        "declaration_version": version,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": 1800.0,
        "declared_by": "test",
        "note": None,
        "changed": [] if version == 1 else ["machine_group"],
    }
    payload.update(overrides)
    return make_event(
        EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload, recorded_at=recorded_at
    )


def measurement_event(
    index: int,
    *,
    measurement_id: Optional[str] = None,
    acquired_at: Optional[str] = None,
    rpm: Optional[float] = 1800.0,
    direction: Optional[str] = "horizontal",
    sensor_id: Optional[str] = "ACC01",
    unit: Optional[str] = "g",
    version: int = 1,
    point: str = POINT,
    location: Optional[str] = None,
    sampling_rate: float = 10000.0,
) -> dict[str, Any]:
    measurement_id = measurement_id or mid(index)
    location = location or f"seq_m{index:02d}.csv"
    payload = {
        "measurement_id": measurement_id,
        "measurement_point_id": point,
        "declaration_version": version,
        "declaration": {
            "asset_id": ASSET,
            "measurement_point_id": point,
            "acquired_at": acquired_at or acquired(index),
            "timezone_declared": True,
            "timestamp_suspect": False,
            "rpm": rpm,
            "load": None,
            "operating_state": None,
            "sensor_id": sensor_id,
            "direction": direction,
            "declared_by": "synthetic",
            "sampling_rate": sampling_rate,
            "signal_unit": unit,
            "raw_format": None,
            "channel_index": 0,
        },
        "file": {
            "location": location,
            "location_is_relative": True,
            "content_sha256": measurement_id * 4,
            "size_bytes": 10,
        },
        "signal_id": f"sig_{measurement_id}",
        "changed": [] if version == 1 else ["measurement_point_id"],
        "locations": [location],
    }
    return make_event(EVENT_MEASUREMENT_RECORDED, ASSET, payload)


def bearing_block(
    detected: bool = False,
    magnitude: Optional[float] = None,
    envelope: float = 0.5,
    *,
    strength: str = "none",
) -> dict[str, Any]:
    """A bearing block with the four catalog labels; BPFO carries the
    scenario's values, the others stay quiet."""
    quiet = {
        "expected_hz": 0.0,
        "detected": False,
        "evidence_strength": "none",
        "magnitude": None,
        "deviation_pct": None,
        "harmonics_detected": [],
        "envelope_amplitude": 0.4,
        "envelope_peak_hz": None,
    }
    labels = {label: dict(quiet) for label in ("BPFO", "BPFI", "BSF", "FTF")}
    labels["BPFO"] = {
        **quiet,
        "expected_hz": 107.54,
        "detected": detected,
        "evidence_strength": strength,
        "magnitude": magnitude,
        "envelope_amplitude": envelope,
    }
    return {
        "source": "catalog",
        "bearing_id": "6205",
        "shaft_hz": 30.0,
        "catalog_source": "test",
        "labels": labels,
    }


def snapshot_event(
    index: int,
    *,
    rms: float,
    measurement_id: Optional[str] = None,
    processing: str = LINEAGE_A,
    unit: Optional[str] = "g",
    one_x: Optional[float] = None,
    iso_velocity: Optional[float] = None,
    bearing: Optional[dict[str, Any]] = None,
    context_digest: str = "ctx",
    point_version: Optional[int] = 1,
    missing: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    measurement_id = measurement_id or mid(index)
    payload = {
        "snapshot_id": short_id(measurement_id, processing, context_digest),
        "measurement_id": measurement_id,
        "measurement_point_id": POINT,
        "processing": {
            "processing_id": processing,
            "algorithm_version": 1,
            "params": {"tolerance_pct": 5.0},
            "effective": {"sampling_rate": 10000.0},
            "provenance": {"pipeline_version": "0.0.0"},
        },
        "context_digest": context_digest,
        "context": {"rpm": 1800.0, "signal_unit": unit},
        "point_declaration_version": point_version,
        "indicators": {
            "rms": rms,
            "peak": 2.5 * rms,
            "crest_factor": 2.5,
            "kurtosis": -0.2,
            "unit": unit,
        },
        "one_x": (
            None
            if one_x is None
            else {
                "target_hz": 30.0,
                "amplitude": one_x,
                "frequency_hz": 30.0,
                "tolerance_pct": 5.0,
                "bins_searched": 3,
                "unit": unit,
            }
        ),
        "bearing": bearing,
        "iso": (
            None
            if iso_velocity is None
            else {
                "velocity_rms_mm_s": iso_velocity,
                "zone": "A",
                "machine_group": 2,
                "support_type": "rigid",
                "direction": "horizontal",
                "operating_speed_rpm": 1800.0,
                "machine_power_kw": None,
                "frequency_range": "10-1000 Hz",
                "boundaries": {"AB": 1.4, "BC": 2.8, "CD": 4.5},
            }
        ),
        "missing": missing or {},
    }
    return make_event(EVENT_HEALTH_SNAPSHOT_COMPUTED, ASSET, payload)


def baseline_event(
    member_indices: Sequence[int],
    *,
    declared_by: str = "L. Rossi",
    declared_at: str = "2026-03-01T00:00:00+00:00",
    note: Optional[str] = None,
    versions: Optional[dict[int, int]] = None,
) -> dict[str, Any]:
    ids = [mid(i) for i in member_indices]
    payload = {
        "baseline_id": short_id(POINT, *sorted(ids), declared_at),
        "measurement_point_id": POINT,
        "measurement_ids": ids,
        "members": [
            {
                "measurement_id": mid(i),
                "declaration_version": (versions or {}).get(i, 1),
                "point_declaration_version": 1,
            }
            for i in member_indices
        ],
        "declared_by": declared_by,
        "note": note,
        "declared_at": declared_at,
    }
    return make_event(EVENT_BASELINE_DECLARED, ASSET, payload)


def view_of(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return build_asset_view(ASSET, list(events), {})


def series_events(
    values: Sequence[float],
    *,
    rpm_missing: Sequence[int] = (),
    processing: str = LINEAGE_A,
    with_point: bool = True,
    with_iso: bool = True,
    with_one_x: bool = True,
    bearing_of: Any = None,
    sensor_of: Any = None,
    unit_of: Any = None,
    start: int = 1,
) -> list[dict[str, Any]]:
    """Point declaration plus one measurement and one snapshot per value.

    ``rms`` is the value itself; peak, 1x and ISO velocity are proportional
    to it; the bearing block is quiet unless ``bearing_of(index)`` says
    otherwise, so every amplitude indicator classifies alike.
    """
    events: list[dict[str, Any]] = [point_event()] if with_point else []
    for offset, value in enumerate(values):
        index = start + offset
        sensor = sensor_of(index) if sensor_of else "ACC01"
        unit = unit_of(index) if unit_of else "g"
        events.append(
            measurement_event(
                index,
                rpm=None if index in rpm_missing else 1800.0,
                sensor_id=sensor,
                unit=unit,
            )
        )
        events.append(
            snapshot_event(
                index,
                rms=value,
                processing=processing,
                unit=unit,
                one_x=0.1 * value if with_one_x else None,
                iso_velocity=0.3 * value if with_iso else None,
                bearing=bearing_of(index) if bearing_of else bearing_block(),
            )
        )
    return events


def assess(view: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    params = kwargs.pop("params", AssessmentParams())
    return assess_change(view, ASSET, POINT, params=params, **kwargs)


def walk(tree: Any):
    """Yield every dict key in a nested structure."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield key
            yield from walk(value)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            yield from walk(item)


def qualification(result: dict[str, Any], code: str) -> Optional[dict[str, Any]]:
    for entry in result["comparability"]["qualifications"]:
        if entry["code"] == code:
            return entry
    return None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestNotFound:
    def test_unknown_asset_lists_known_assets_without_paths(self):
        result = assess_change(view_of([]), ASSET, POINT, known_assets=["Q-7", "P-102"])
        assert result["status"] == "not_found"
        assert result["known_assets"] == ["P-102", "Q-7"]
        assert result["known_points"] == []
        assert "P-101" in result["message"]
        # Closed oracle: ids only, never a path or a directory listing.
        for forbidden in ("\\", "/", ".jsonl", "data"):
            assert forbidden not in result["message"]
        assert result["suggestion"]
        assert "error" not in set(walk(result))

    def test_unknown_point_lists_known_points(self):
        view = view_of(series_events(STABLE))
        result = assess_change(view, ASSET, "pump_nde_v", known_assets=[ASSET])
        assert result["status"] == "not_found"
        assert result["known_points"] == [POINT]
        assert result["known_assets"] == [ASSET]
        assert "pump_nde_v" in result["message"]
        assert "declare_measurement_point" in result["suggestion"]

    def test_point_known_only_through_measurements_is_found(self):
        view = view_of(series_events(STABLE + [1.0], with_point=False))
        result = assess(view)
        assert result["status"] == "assessed"

    def test_view_of_another_asset_is_a_caller_error(self):
        view = view_of(series_events(STABLE))
        with pytest.raises(ValueError, match="belongs to asset"):
            assess_change(view, "P-102", POINT)


class TestParams:
    @pytest.mark.parametrize(
        "field, value",
        [
            ("reference_measurements", 2),
            ("last_k", 0),
            ("persistence_consecutive", 0),
            ("persistence_min_out_of_last", 0),
            ("drift_alpha", 0.0),
            ("drift_alpha", 1.5),
            ("band_sigma", 0.0),
            ("band_relative_floor", -0.1),
            ("iso_change_fraction", 0.0),
            ("acquired_since", "yesterday"),
            ("acquired_until", "2026-13-40"),
        ],
    )
    def test_out_of_range_is_refused(self, field, value):
        params = AssessmentParams(**{field: value})
        with pytest.raises(ValueError, match=field):
            assess_change(view_of(series_events(STABLE)), ASSET, POINT, params=params)

    def test_defaults_are_the_plan_numbers(self):
        params = AssessmentParams()
        assert params.reference_measurements == 10
        assert params.last_k == 5
        assert params.persistence_consecutive == 3
        assert params.persistence_min_out_of_last == 4
        assert params.drift_alpha == 0.05
        assert params.band_sigma == 3.0
        assert params.band_relative_floor == 0.25
        assert params.iso_change_fraction == 0.25
        assert params.acquired_since is None and params.acquired_until is None

    def test_vocabularies(self):
        assert ASSESSMENT_STATUSES == (
            "assessed",
            "not_found",
            "insufficient_history",
            "processing_not_homogeneous",
        )
        assert CLASSIFICATIONS[-1] == "persistent_change"
        assert CLASSIFICATIONS[0] == "no_change"
        assert AMPLITUDE_INDICATORS == ("rms", "peak", "one_x", "iso_velocity")
        assert MIN_USABLE_SLOTS == 4


# ---------------------------------------------------------------------------
# Reference: automatic window (AE6) and history size
# ---------------------------------------------------------------------------


class TestAutomaticWindow:
    def test_ae6_wording_and_the_first_ten_slots(self):
        view = view_of(series_events(STABLE * 3))
        result = assess(view)
        assert result["status"] == "assessed"
        reference = result["reference"]
        assert reference["kind"] == "automatic_window"
        assert reference["health_declared"] is False
        assert reference["measurement_ids"] == [mid(i) for i in range(1, 11)]
        assert reference["count"] == 10
        assert reference["statistics_quality"] == "full"
        assert reference["provisional"] is False
        for text in (reference["message"], result["message"]):
            assert "not declared" in text
            assert "relative comparison" in text
            assert "healthy" not in text
        assert result["assessed"]["classification"] == "no_change"
        assert result["assessed"]["direction"] is None
        assert result["assessed"]["indicators_driving"] == []
        assert result["suggested_verification"] is None
        assert result["observed"]["acquisitions_assessed"] == 20
        assert result["observed"]["slots_assessed"] == 30
        assert len(result["observed"]["measurement_ids_assessed"]) == 5
        assert result["observed"]["measurement_ids_assessed"][-1]["measurement_id"] == (
            mid(30)
        )
        assert result["observed"]["measurement_ids_assessed"][-1]["location"] == (
            "seq_m30.csv"
        )

    def test_sigma_zero_reference_uses_the_relative_floor(self):
        view = view_of(series_events([2.0] * 10 + [2.2]))
        result = assess(view)
        assert result["status"] == "assessed"
        stats = result["observed"]["reference_statistics"]["rms"]
        assert stats["std"] == 0.0
        assert stats["n"] == 10
        assert stats["unit"] == "g"
        assert stats["band"]["half_width"] == pytest.approx(0.5)
        assert stats["band"]["basis"] == "25% of the mean"
        assert result["reference"]["statistics_quality"] == "full"
        assert result["derived"]["deltas"]["rms"]["delta_pct"] == pytest.approx(10.0)
        assert result["assessed"]["classification"] == "no_change"

    def test_three_measurements_are_insufficient(self):
        view = view_of(series_events(STABLE[:3]))
        result = assess(view)
        assert result["status"] == "insufficient_history"
        assert result["available"] == 3
        assert result["required"] == 4
        assert "declare_healthy_baseline" in result["remedy"]
        assert result["reference"]["kind"] == "automatic_window"
        assert result["comparability"]["comparable"] == 3
        assert "error" not in set(walk(result))

    def test_seven_measurements_give_a_provisional_reference_of_six(self):
        view = view_of(series_events(STABLE[:7]))
        result = assess(view)
        assert result["status"] == "assessed"
        assert result["reference"]["provisional"] is True
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 7)]
        assert result["reference"]["statistics_quality"] == "provisional"
        assert "provisional" in result["reference"]["message"]
        assert result["observed"]["acquisitions_assessed"] == 1
        assert result["observed"]["latest_measurement_id"] == mid(7)

    def test_four_measurements_give_a_relative_only_reference(self):
        result = assess(view_of(series_events(STABLE[:4])))
        assert result["status"] == "assessed"
        assert result["reference"]["statistics_quality"] == "relative_only"
        assert result["reference"]["count"] == 3

    def test_eleven_measurements_fill_the_window_exactly(self):
        result = assess(view_of(series_events(STABLE + [1.0])))
        assert result["reference"]["provisional"] is False
        assert result["reference"]["count"] == 10
        assert result["observed"]["acquisitions_assessed"] == 1

    def test_reference_size_is_a_parameter(self):
        params = AssessmentParams(reference_measurements=5)
        result = assess(view_of(series_events(STABLE)), params=params)
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 6)]
        assert result["reference"]["statistics_quality"] == "relative_only"
        assert result["observed"]["acquisitions_assessed"] == 5


class TestRequestedWindow:
    def test_window_excluding_the_reference_still_uses_it(self):
        view = view_of(series_events(STABLE * 3))
        params = AssessmentParams(acquired_since=acquired(15))
        result = assess(view, params=params)
        assert result["status"] == "assessed"
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 11)]
        assert result["reference"]["outside_window"] is True
        assert "outside the requested window" in result["reference"]["message"]
        assert result["observed"]["acquisitions_assessed"] == 16
        assert result["observed"]["slots_assessed"] == 26

    def test_window_inside_the_reference_leaves_nothing_to_assess(self):
        view = view_of(series_events(STABLE * 3))
        params = AssessmentParams(acquired_until=acquired(5))
        result = assess(view, params=params)
        assert result["status"] == "insufficient_history"
        assert result["available"] == 0
        assert result["required"] == 1
        assert "acquired_until" in result["message"]
        assert result["reference"]["outside_window"] is True

    def test_bounds_are_compared_as_instants(self):
        view = view_of(series_events(STABLE * 3))
        # Same instant as M15, spelled in another offset.
        spelled = (FIRST_ACQUIRED + 14 * WEEK).astimezone(timezone.utc).isoformat()
        params = AssessmentParams(acquired_since=spelled, acquired_until=acquired(20))
        result = assess(view, params=params)
        assert result["observed"]["acquisitions_assessed"] == 6


# ---------------------------------------------------------------------------
# Reference: declared baseline (AE7)
# ---------------------------------------------------------------------------


class TestDeclaredBaseline:
    def test_ae7_baseline_cites_declared_by_and_uses_only_its_members(self):
        values = [1.0, 1.1, 0.9, 1.05, 0.95] + STABLE * 2 + STABLE[:5]
        events = series_events(values) + [baseline_event([1, 2, 3, 4, 5])]
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        reference = result["reference"]
        assert reference["kind"] == "declared_baseline"
        assert reference["health_declared"] is True
        assert reference["measurement_ids"] == [mid(i) for i in range(1, 6)]
        assert "declared healthy baseline" in reference["message"]
        assert "declared by L. Rossi" in reference["message"]
        assert "2026-03-01" in reference["message"]
        assert "declared by L. Rossi" in result["message"]
        assert reference["baseline"]["members_used"] == 5
        assert reference["baseline"]["excluded"] == []
        stats = result["observed"]["reference_statistics"]["rms"]
        assert stats["n"] == 5
        assert stats["mean"] == pytest.approx(sum(values[:5]) / 5)
        assert reference["statistics_quality"] == "relative_only"
        assert result["observed"]["acquisitions_assessed"] == 25

    def test_superseded_member_is_excluded_with_a_qualification(self):
        events = series_events(STABLE * 3) + [baseline_event([1, 2, 3, 4, 5])]
        events.append(measurement_event(3, version=2, point="other_point"))
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["reference"]["count"] == 4
        assert result["reference"]["measurement_ids"] == [mid(i) for i in (1, 2, 4, 5)]
        entry = qualification(result, "baseline_member_superseded")
        assert entry is not None and entry["count"] == 1
        assert mid(3) in entry["detail"]
        assert result["reference"]["baseline"]["members_used"] == 4
        assert result["reference"]["baseline"]["excluded"][0]["measurement_id"] == mid(
            3
        )
        assert "4 of 5 members" in result["reference"]["message"]

    def test_member_with_a_new_declaration_version_is_superseded(self):
        events = series_events(STABLE * 3) + [baseline_event([1, 2, 3, 4, 5])]
        events.append(measurement_event(2, version=2, rpm=1500.0))
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        entry = qualification(result, "baseline_member_superseded")
        assert entry is not None and "version 1" in entry["detail"]
        assert mid(2) not in result["reference"]["measurement_ids"]

    def test_non_comparable_member_is_excluded_with_a_qualification(self):
        events = series_events(
            STABLE * 3, unit_of=lambda i: "mm/s" if i == 2 else "g"
        ) + [baseline_event([1, 2, 3, 4, 5])]
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        entry = qualification(result, "baseline_member_non_comparable")
        assert entry is not None and "unit_incompatible" in entry["detail"]
        assert result["reference"]["count"] == 4

    def test_fewer_than_three_members_is_insufficient_naming_the_baseline(self):
        events = series_events(STABLE * 3) + [baseline_event([1, 2, 3, 4, 5])]
        for index in (2, 3, 4):
            events.append(measurement_event(index, version=2, point="other_point"))
        view = view_of(events)
        result = assess(view)
        assert result["status"] == "insufficient_history"
        assert result["available"] == 2
        assert result["required"] == 3
        assert "re-declare the baseline" in result["remedy"]
        baseline_id = view["baselines"][POINT]["current"]["baseline_id"]
        assert baseline_id in result["message"]
        assert "automatic window is not used" in result["message"]
        assert result["reference"]["kind"] == "declared_baseline"

    def test_withdrawn_baseline_falls_back_to_the_window_with_a_note(self):
        events = series_events(STABLE * 3) + [
            baseline_event([1, 2, 3, 4, 5]),
            baseline_event([], declared_at="2026-03-02T00:00:00+00:00"),
        ]
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["reference"]["kind"] == "automatic_window"
        assert result["reference"]["health_declared"] is False
        assert result["reference"]["withdrawn_baseline"]["declared_at"] == (
            "2026-03-02T00:00:00+00:00"
        )
        assert "withdrawn on 2026-03-02" in result["reference"]["message"]
        assert "healthy" not in result["message"]

    def test_baseline_members_without_the_lineage_are_not_homogeneous(self):
        events = [point_event()]
        for index, value in enumerate(STABLE * 3, start=1):
            events.append(measurement_event(index))
            lineage = LINEAGE_A if index <= 5 else LINEAGE_B
            events.append(snapshot_event(index, rms=value, processing=lineage))
        events.append(baseline_event([1, 2, 3, 4, 5]))
        result = assess(view_of(events))
        assert result["status"] == "processing_not_homogeneous"
        assert result["lineages"] == {LINEAGE_A: 5, LINEAGE_B: 25}
        assert "declared baseline" in result["message"]
        assert REPROCESS_CALL in result["remedy"]

    def test_non_members_inside_the_baseline_span_are_not_assessed(self):
        events = series_events(STABLE * 3) + [baseline_event([2, 4, 6, 8])]
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["reference"]["excluded_inside_span"] == [
            mid(i) for i in (1, 3, 5, 7)
        ]
        assert result["observed"]["acquisitions_assessed"] == 22

    def test_baseline_member_wins_a_slot_collision(self):
        """A later export of a baseline member's acquisition does not replace
        the member: the declared baseline is never silently swapped."""
        events = series_events(STABLE * 3) + [baseline_event([1, 2, 3, 4, 5])]
        events.append(measurement_event(3, measurement_id=mid(103)))
        events.append(snapshot_event(3, rms=5.0, measurement_id=mid(103)))
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 6)]
        collapsed = result["comparability"]["collapsed_duplicates"]
        assert [c["measurement_id"] for c in collapsed] == [mid(103)]
        assert collapsed[0]["kept_measurement_id"] == mid(3)


# ---------------------------------------------------------------------------
# Classification (AE8 and the edge cases)
# ---------------------------------------------------------------------------


class TestClassification:
    def test_ae8_spike_then_return_then_ramp(self):
        spike = assess(view_of(series_events(STABLE + [SPIKE])))
        assert spike["status"] == "assessed"
        assert spike["assessed"]["classification"] == "unconfirmed_single_acquisition"
        assert spike["assessed"]["direction"] == "increase"
        assert spike["assessed"]["sudden"] is True
        assert spike["suggested_verification"] == (
            "Repeat the measurement in the same operating conditions."
        )
        assert "rms" in spike["assessed"]["indicators_driving"]
        assert spike["assessed"]["onset_measurement_id"] == mid(11)
        assert spike["assessed"]["onset_acquired_at"] == acquired(11)
        assert spike["assessed"]["onset_coincides_with"] == []
        assert spike["derived"]["deltas"]["rms"]["exceeds"] is True
        assert spike["derived"]["deltas"]["rms"]["side"] == "above"
        assert spike["derived"]["exceedance_runs"]["rms"] == [True]

        returned = assess(view_of(series_events(STABLE + [SPIKE, 1.0])))
        assert returned["assessed"]["classification"] == "isolated_episode"
        assert returned["assessed"]["direction"] == "increase"
        assert returned["suggested_verification"] is None
        assert returned["assessed"]["onset_measurement_id"] == mid(11)
        assert returned["derived"]["exceedance_runs"]["rms"] == [True, False]

        # An unconfirmed run of two: the onset is the start of that run, and
        # the earlier, closed episode is not confused with it.
        two = assess(view_of(series_events(STABLE + [SPIKE, 1.0, 1.0, 1.6, 1.6])))
        assert two["assessed"]["classification"] == "unconfirmed_single_acquisition"
        assert two["assessed"]["onset_measurement_id"] == mid(14)

        values = STABLE + [SPIKE] + STABLE[:9] + RAMP
        ramp = assess(view_of(series_events(values)))
        assert ramp["assessed"]["classification"] == "persistent_change"
        assert ramp["assessed"]["direction"] == "increase"
        criterion = ramp["assessed"]["criterion"]
        assert (
            "consecutive acquisitions outside the band" in criterion
            or "significant drift (p=" in criterion
        )
        assert "mean +/- max(3 sigma, 25%)" in criterion
        assert ramp["assessed"]["onset_measurement_id"] == mid(24)
        assert ramp["assessed"]["onset_acquired_at"] == acquired(24)
        assert ramp["observed"]["acquisitions_assessed"] == 20
        assert ramp["derived"]["per_indicator"]["rms"]["rules"]["consecutive"] is True
        assert ramp["derived"]["per_indicator"]["rms"]["rules"]["consecutive_run"] == 7
        assert ramp["suggested_verification"] is not None
        assert ramp["suggested_verification"].count(".") == 1

    def test_one_spike_exported_three_times_is_one_slot(self):
        events = series_events(STABLE)
        for measurement_id in (mid(11), mid(111), mid(211)):
            events.append(measurement_event(11, measurement_id=measurement_id))
            events.append(snapshot_event(11, rms=SPIKE, measurement_id=measurement_id))
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["assessed"]["classification"] == "unconfirmed_single_acquisition"
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 11)]
        assert result["observed"]["acquisitions_assessed"] == 1
        assert result["observed"]["latest_measurement_id"] == mid(211)
        collapsed = result["comparability"]["collapsed_duplicates"]
        assert [c["measurement_id"] for c in collapsed] == [mid(11), mid(111)]
        assert all(c["kept_measurement_id"] == mid(211) for c in collapsed)
        assert all(c["acquired_at"] == acquired(11) for c in collapsed)

    def test_slow_drift_with_the_latest_outside_is_persistent(self):
        view = view_of(series_events([1.0] * 10 + [1.05, 1.10, 1.15, 1.20, 1.30]))
        result = assess(view)
        assert result["assessed"]["classification"] == "persistent_change"
        assert result["assessed"]["direction"] == "increase"
        assert "significant drift (p=" in result["assessed"]["criterion"]
        assert "latest value outside the band" in result["assessed"]["criterion"]
        drift = result["derived"]["drift"]["rms"]
        assert drift["significant"] is True
        assert drift["p_value"] < 0.05
        assert drift["trend_direction"] == "increasing"
        assert (
            result["derived"]["per_indicator"]["rms"]["rules"]["consecutive"] is False
        )
        assert result["assessed"]["onset_measurement_id"] == mid(15)

    def test_slow_drift_with_the_latest_inside_is_no_change_with_the_drift(self):
        view = view_of(series_events([1.0] * 10 + [1.05, 1.10, 1.15, 1.20, 1.24]))
        result = assess(view)
        assert result["assessed"]["classification"] == "no_change"
        assert result["assessed"]["direction"] is None
        drift = result["derived"]["drift"]["rms"]
        assert drift["p_value"] < 0.05
        assert drift["trend_direction"] == "increasing"
        assert drift["significant"] is True

    def test_intermittent_four_of_the_last_five_is_persistent(self):
        view = view_of(series_events([1.0] * 10 + [1.5, 1.0, 1.5, 1.5, 1.0, 1.5, 1.5]))
        result = assess(view)
        assert result["assessed"]["classification"] == "persistent_change"
        assert (
            "4 of last 5 acquisitions outside the band"
            in result["assessed"]["criterion"]
        )
        assert (
            result["derived"]["per_indicator"]["rms"]["rules"]["intermittent"] is True
        )
        assert (
            result["derived"]["per_indicator"]["rms"]["rules"]["consecutive"] is False
        )
        # The onset is the first exceedance of the window that fired the rule.
        assert result["assessed"]["onset_measurement_id"] == mid(13)
        assert result["derived"]["exceedance_runs"]["rms"] == [
            True,
            True,
            False,
            True,
            True,
        ]

    def test_intermittent_two_of_five_follows_the_latest_slot(self):
        inside = assess(view_of(series_events([1.0] * 10 + [1.5, 1.0, 1.0, 1.5, 1.0])))
        assert inside["assessed"]["classification"] == "isolated_episode"
        outside = assess(view_of(series_events([1.0] * 10 + [1.0, 1.5, 1.0, 1.0, 1.5])))
        assert outside["assessed"]["classification"] == "unconfirmed_single_acquisition"
        assert "without persistence" in outside["assessed"]["criterion"]

    def test_sustained_decrease_is_persistent_and_never_an_improvement(self):
        result = assess(view_of(series_events([1.0] * 10 + [0.7] * 5)))
        assert result["assessed"]["classification"] == "persistent_change"
        assert result["assessed"]["direction"] == "decrease"
        text = canonical_json(result).lower()
        assert "improv" not in text
        assert "healthy" not in text
        assert result["derived"]["deltas"]["rms"]["delta_pct"] == pytest.approx(-30.0)
        assert result["derived"]["deltas"]["rms"]["side"] == "below"

    def test_sudden_step_flag(self):
        gradual = assess(view_of(series_events([1.0] * 10 + [1.2, 1.3, 1.4, 1.5, 1.6])))
        assert gradual["assessed"]["classification"] == "persistent_change"
        assert gradual["assessed"]["sudden"] is False
        stepped = assess(view_of(series_events([1.0] * 10 + [1.0, 1.0, 1.6, 1.6, 1.6])))
        assert stepped["assessed"]["classification"] == "persistent_change"
        assert stepped["assessed"]["sudden"] is True

    def test_onset_coinciding_with_a_sensor_change_is_reported(self):
        sensor_of = lambda index: "ACC02" if index == 11 else "ACC01"  # noqa: E731
        with_change = assess(
            view_of(series_events(STABLE + [SPIKE], sensor_of=sensor_of))
        )
        assert with_change["assessed"]["classification"] == (
            "unconfirmed_single_acquisition"
        )
        assert with_change["assessed"]["onset_coincides_with"] == ["sensor_changed"]
        assert "sensor_changed" in with_change["suggested_verification"]
        without = assess(view_of(series_events(STABLE + [SPIKE])))
        assert without["assessed"]["onset_coincides_with"] == []

    def test_support_indicators_are_reported_not_classified(self):
        result = assess(view_of(series_events(STABLE + [SPIKE])))
        assert set(result["observed"]["reference_statistics"]) >= {
            "crest_factor",
            "kurtosis",
        }
        assert result["derived"]["deltas"]["crest_factor"]["exceeds"] is None
        assert "crest_factor" not in result["derived"]["per_indicator"]
        assert "kurtosis" not in result["assessed"]["indicators_driving"]

    def test_iso_change_against_the_bc_boundary(self):
        result = assess(view_of(series_events([1.0] * 10 + [4.0])))
        iso_change = result["derived"]["iso_change"]
        assert iso_change["machine_group"] == 2
        assert iso_change["support_type"] == "rigid"
        assert iso_change["boundary_bc_mm_s"] == 2.8
        assert iso_change["threshold_mm_s"] == pytest.approx(0.7)
        assert iso_change["delta_mm_s"] == pytest.approx(0.9)
        assert iso_change["exceeded"] is True
        assert result["observed"]["reference_statistics"]["iso_velocity"]["unit"] == (
            "mm/s"
        )

    def test_indicator_missing_in_one_snapshot_is_reported_unavailable(self):
        events = series_events(STABLE + [1.0], with_iso=False)
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert "iso_velocity" not in result["observed"]["reference_statistics"]
        assert result["observed"]["indicators_unavailable"]["iso_velocity"] == {
            "missing_in": 11,
            "of": 11,
            "reason": None,
        }


class TestBearingEvidence:
    @staticmethod
    def evidence(present_indices: set[int]):
        def bearing_of(index: int) -> dict[str, Any]:
            if index in present_indices:
                return bearing_block(True, 120.0, 0.5, strength="high")
            return bearing_block()

        return bearing_of

    def test_present_in_four_of_the_last_five_is_persistent(self):
        events = series_events(STABLE * 2, bearing_of=self.evidence({16, 18, 19, 20}))
        result = assess(view_of(events))
        presence = result["observed"]["evidence_presence"]["BPFO"]
        assert presence["present_in_last_k"] == 4
        assert presence["k"] == 5
        assert presence["presence"] == [True, False, True, True, True]
        assert presence["latest_evidence_strength"] == "high"
        evidence = result["assessed"]["evidence"]["BPFO"]
        assert evidence["persistent"] is True
        assert evidence["consecutive"] == 3
        assert evidence["classification"] == "persistent_change"
        assert "4 of the last 5" in evidence["criterion"]
        assert result["assessed"]["classification"] == "persistent_change"
        assert result["assessed"]["indicators_driving"] == ["evidence_BPFO"]
        # The onset is the start of the run that made the evidence persistent.
        assert result["assessed"]["onset_measurement_id"] == mid(18)
        assert "diagnose_vibration" in result["suggested_verification"]

    def test_present_once_is_reported_but_not_persistent(self):
        events = series_events(STABLE * 2, bearing_of=self.evidence({18}))
        result = assess(view_of(events))
        presence = result["observed"]["evidence_presence"]["BPFO"]
        assert presence["present_in_last_k"] == 1
        evidence = result["assessed"]["evidence"]["BPFO"]
        assert evidence["persistent"] is False
        assert evidence["classification"] == "isolated_episode"
        assert result["assessed"]["classification"] == "isolated_episode"

    def test_detection_without_magnitude_is_not_presence(self):
        def bearing_of(index: int) -> dict[str, Any]:
            if index >= 16:
                return bearing_block(True, None, 0.5, strength="low")
            return bearing_block()

        result = assess(view_of(series_events(STABLE * 2, bearing_of=bearing_of)))
        assert result["observed"]["evidence_presence"]["BPFO"]["present_in_last_k"] == 0
        assert result["assessed"]["evidence"]["BPFO"]["persistent"] is False
        assert result["assessed"]["classification"] == "no_change"

    def test_emerging_line_has_a_defined_envelope_band_and_classifies(self):
        def bearing_of(index: int) -> dict[str, Any]:
            if index >= 18:
                return bearing_block(True, 120.0, 5.0, strength="high")
            return bearing_block(False, None, 0.5 + 0.01 * (index % 3))

        result = assess(view_of(series_events(STABLE * 2, bearing_of=bearing_of)))
        stats = result["observed"]["reference_statistics"]["envelope_BPFO"]
        assert stats["band"]["half_width"] > 0
        assert stats["band"]["degenerate"] is False
        assert stats["unit"] == "g"
        assert result["assessed"]["classification"] == "persistent_change"
        assert "envelope_BPFO" in result["assessed"]["indicators_driving"]
        assert "evidence_BPFO" in result["assessed"]["indicators_driving"]
        assert result["derived"]["per_indicator"]["envelope_BPFO"]["direction"] == (
            "increase"
        )
        assert result["assessed"]["onset_measurement_id"] == mid(18)
        # Quiet labels stay inside their (floor) band.
        assert result["derived"]["per_indicator"]["envelope_BPFI"][
            "classification"
        ] == ("no_change")


# ---------------------------------------------------------------------------
# Comparability (AE3, AE4) and slots
# ---------------------------------------------------------------------------


class TestComparability:
    def test_ae3_undeclared_rpm_qualifies_and_participates(self):
        missing = (3, 6, 9, 13, 16, 19, 23, 26, 29, 30)
        result = assess(view_of(series_events(STABLE * 3, rpm_missing=missing)))
        assert result["status"] == "assessed"
        comparability = result["comparability"]
        assert comparability["comparable"] == 20
        assert comparability["qualified"] == 10
        assert comparability["non_comparable"] == 0
        entry = qualification(result, "rpm_not_declared")
        assert entry is not None
        assert entry["count"] == 10
        assert "constant regime" in entry["detail"]
        assert result["observed"]["slots_assessed"] == 30
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 11)]
        assert result["reference"]["qualification_codes"] == {"rpm_not_declared": 3}
        assert "rpm_not_declared (3)" in result["reference"]["message"]

    def test_ae4_velocity_measurement_is_excluded_and_listed(self):
        events = series_events(
            STABLE * 3 + [1.0], unit_of=lambda i: "mm/s" if i == 31 else "g"
        )
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        comparability = result["comparability"]
        assert comparability["non_comparable"] == 1
        assert comparability["excluded"] == [
            {
                "measurement_id": mid(31),
                "acquired_at": acquired(31),
                "reasons": ["unit_incompatible"],
            }
        ]
        assert comparability["excluded_truncated"] is False
        assert mid(31) not in result["reference"]["measurement_ids"]
        assert result["observed"]["slots_assessed"] == 30
        assert result["observed"]["latest_measurement_id"] == mid(30)

    def test_same_family_unit_is_converted_into_the_expected_unit(self):
        events = series_events(
            STABLE + [1.0], unit_of=lambda i: "m/s2" if i == 11 else "g"
        )
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["comparability"]["comparable"] == 11
        assert result["observed"]["reference_statistics"]["rms"]["unit"] == "g"
        # 1.0 m/s2 is about 0.102 g: 90% below the reference mean.
        assert result["derived"]["deltas"]["rms"]["latest"] == pytest.approx(
            1.0 / 9.80665
        )
        assert result["assessed"]["classification"] == "unconfirmed_single_acquisition"
        assert result["assessed"]["direction"] == "decrease"

    def test_direction_mismatch_is_excluded_and_the_other_axis_collides(self):
        events = series_events(STABLE + [1.0])
        events.append(
            measurement_event(11, measurement_id=mid(111), direction="vertical")
        )
        events.append(snapshot_event(11, rms=3.0, measurement_id=mid(111)))
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["comparability"]["excluded"][0]["reasons"] == [
            "direction_mismatch",
            "timestamp_collision",
        ]
        collision = qualification(result, "timestamp_collision")
        assert collision is not None and collision["count"] == 2
        assert result["assessed"]["classification"] == "no_change"

    def test_point_redeclared_after_the_acquisitions_qualifies_them(self):
        events = series_events(STABLE * 3)
        events.append(
            point_event(
                2,
                machine_group=1,
                recorded_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            )
        )
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        entry = qualification(result, "point_declaration_changed")
        assert entry is not None and entry["count"] == 30
        assert "2026-09-01" in entry["detail"]
        assert result["comparability"]["qualified"] == 30

    def test_collect_point_slots_exposes_the_staging(self):
        staged = collect_point_slots(view_of(series_events(STABLE * 2)), POINT)
        assert staged["found"] is True
        assert staged["point"]["declaration_version"] == 1
        assert [s["measurement_id"] for s in staged["reference"]["slots"]] == [
            mid(i) for i in range(1, 11)
        ]
        assert [s["measurement_id"] for s in staged["sequence"]] == [
            mid(i) for i in range(11, 21)
        ]
        assert len(staged["usable"]) == 20
        assert staged["reference"]["insufficient"] is None


# ---------------------------------------------------------------------------
# Lineage (AE5)
# ---------------------------------------------------------------------------


class TestLineage:
    @staticmethod
    def split(lineage_of) -> list[dict[str, Any]]:
        events = [point_event()]
        for index, value in enumerate(STABLE * 3, start=1):
            events.append(measurement_event(index))
            for lineage in lineage_of(index):
                events.append(snapshot_event(index, rms=value, processing=lineage))
        return events

    def test_ae5_split_lineages_are_not_homogeneous(self):
        events = self.split(lambda i: [LINEAGE_A] if i <= 20 else [LINEAGE_B])
        result = assess(view_of(events))
        assert result["status"] == "processing_not_homogeneous"
        assert result["lineages"] == {LINEAGE_A: 20, LINEAGE_B: 10}
        assert result["evaluated_slots"] == 30
        assert result["missing_for_current"] == 30
        assert result["current_processing_id"] == processing_id()
        assert REPROCESS_CALL in result["remedy"]
        assert "idempotent" in result["remedy"]
        assert result["reference"]["measurement_ids"] == [mid(i) for i in range(1, 11)]
        assert "error" not in set(walk(result))

    def test_ae5_both_lineages_everywhere_the_newest_wins(self):
        newest_b = self.split(lambda i: [LINEAGE_A, LINEAGE_B])
        result = assess(view_of(newest_b))
        assert result["status"] == "assessed"
        assert result["lineage"]["processing_id"] == LINEAGE_B
        assert result["lineage"]["candidates"] == {LINEAGE_A: 30, LINEAGE_B: 30}
        assert result["lineage"]["covered"] == 30
        assert result["lineage"]["is_current"] is False
        newest_a = self.split(lambda i: [LINEAGE_B, LINEAGE_A])
        assert assess(view_of(newest_a))["lineage"]["processing_id"] == LINEAGE_A

    def test_newest_lineage_with_holes_is_never_used(self):
        events = self.split(
            lambda i: [LINEAGE_B] if i <= 10 or i >= 26 else [LINEAGE_A]
        )
        result = assess(view_of(events))
        assert result["status"] == "processing_not_homogeneous"
        assert result["lineages"] == {LINEAGE_A: 15, LINEAGE_B: 15}

    def test_complete_old_lineage_is_used_while_the_new_one_has_holes(self):
        events = self.split(
            lambda i: [LINEAGE_A, LINEAGE_B] if i <= 10 or i >= 26 else [LINEAGE_A]
        )
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["lineage"]["processing_id"] == LINEAGE_A
        assert result["lineage"]["candidates"] == {LINEAGE_A: 30, LINEAGE_B: 15}

    def test_missing_snapshot_is_not_homogeneous(self):
        events = self.split(lambda i: [] if i == 17 else [LINEAGE_A])
        result = assess(view_of(events))
        assert result["status"] == "processing_not_homogeneous"
        assert result["lineages"] == {LINEAGE_A: 29}

    def test_current_lineage_is_recognized(self):
        events = series_events(STABLE + [1.0], processing=processing_id())
        result = assess(view_of(events))
        assert result["lineage"]["is_current"] is True
        assert result["lineage"]["missing_for_current"] == 0
        assert result["lineage"]["algorithm_version"] == 1

    def test_window_shrinks_the_evaluated_set_for_the_lineage(self):
        events = self.split(
            lambda i: [LINEAGE_A] if i <= 20 else [LINEAGE_A, LINEAGE_B]
        )
        params = AssessmentParams(acquired_since=acquired(21))
        result = assess(view_of(events), params=params)
        assert result["status"] == "assessed"
        assert result["lineage"]["processing_id"] == LINEAGE_A
        assert result["lineage"]["candidates"] == {LINEAGE_A: 20, LINEAGE_B: 10}


# ---------------------------------------------------------------------------
# Contract: determinism, serialization, purity
# ---------------------------------------------------------------------------


class TestContract:
    def test_same_view_gives_the_same_canonical_payload(self):
        values = STABLE + [SPIKE] + STABLE[:9] + RAMP
        events = series_events(values, rpm_missing=(3, 6))
        view = view_of(events)
        first = canonical_json(assess(view))
        second = canonical_json(assess(view_of(events)))
        assert first == second
        assert canonical_json(assess(view)) == first

    @pytest.mark.parametrize(
        "values",
        [STABLE[:3], STABLE + [SPIKE], [1.0] * 10 + [0.7] * 5],
    )
    def test_payloads_are_json_serializable_without_an_error_key(self, values):
        result = assess(view_of(series_events(values)))
        json.dumps(result, allow_nan=False)
        assert "error" not in set(walk(result))
        assert result["status"] in ASSESSMENT_STATUSES

    def test_assessed_payload_top_level_keys(self):
        result = assess(view_of(series_events(STABLE + [1.0])))
        assert list(result) == [
            "status",
            "asset_id",
            "measurement_point_id",
            "reference",
            "lineage",
            "observed",
            "derived",
            "assessed",
            "comparability",
            "suggested_verification",
            "message",
        ]
        assert set(result["observed"]) == {
            "reference_statistics",
            "latest",
            "latest_measurement_id",
            "latest_acquired_at",
            "evidence_presence",
            "indicators_unavailable",
            "acquisitions_assessed",
            "slots_assessed",
            "measurement_ids_assessed",
        }
        assert set(result["derived"]) == {
            "deltas",
            "exceedance_runs",
            "drift",
            "iso_change",
            "per_indicator",
        }
        assert set(result["assessed"]) == {
            "classification",
            "direction",
            "sudden",
            "criterion",
            "indicators_driving",
            "onset_measurement_id",
            "onset_acquired_at",
            "onset_coincides_with",
            "evidence",
        }
        assert set(result["comparability"]) == {
            "comparable",
            "qualified",
            "non_comparable",
            "qualifications",
            "excluded",
            "excluded_truncated",
            "collapsed_duplicates",
            "collapsed_truncated",
        }


class TestModulePurity:
    def test_imports_only_the_declared_pure_dependencies(self):
        source = Path(assessment.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        assert modules == {
            "logging",
            "math",
            "datetime",
            "typing",
            "numpy",
            "diagnostics.iso20816",
            "prognostics.trend_analyzer",
            "comparability",
            "snapshot",
        }
        # Never a file, a signal or the clock.
        for forbidden in (
            r"\bopen\(",
            r"\bPath\(",
            r"np\.fromfile",
            r"datetime\.now",
            r"\btime\.",
        ):
            assert re.search(forbidden, source) is None, forbidden

    def test_healthy_appears_only_in_the_declared_baseline_branch(self):
        source = Path(assessment.__file__).read_text(encoding="utf-8")
        code_lines = [
            line
            for line in source.splitlines()
            if "healthy" in line.lower() and not line.lstrip().startswith(("#", "*"))
        ]
        # The declared-baseline wording, the tool name in the remedies, and
        # the docstring sentences that state the rule.
        for line in code_lines:
            assert (
                "declared healthy baseline" in line
                or "declare_healthy_baseline" in line
                or "never called healthy" in line
                or "never" in line
            ), line

    def test_package_reexports(self):
        import predictive_maintenance_mcp.asset_ledger as pkg

        for name in (
            "AssessmentParams",
            "assess_change",
            "collect_point_slots",
            "current_snapshot_id_of",
            "validate_params",
            "ASSESSMENT_STATUSES",
            "CLASSIFICATIONS",
        ):
            assert getattr(pkg, name) is getattr(assessment, name), name
