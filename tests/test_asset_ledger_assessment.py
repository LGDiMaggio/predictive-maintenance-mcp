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
from predictive_maintenance_mcp.asset_ledger.snapshot import (
    context_digest,
    processing_id,
    resolve_context,
)
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
    machine_group: int = 2,
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
                "machine_group": machine_group,
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


def digest_of(measurement: dict[str, Any], point: Optional[dict[str, Any]]) -> str:
    """The ``context_digest`` the snapshot code records for a measurement
    event under a point declaration event (None: point never declared)."""
    declaration = measurement["payload"]["declaration"]
    context = resolve_context(declaration, None if point is None else point["payload"])
    return context_digest(context, declaration)


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
    placeholder_digests: bool = False,
) -> list[dict[str, Any]]:
    """Point declaration plus one measurement and one snapshot per value.

    ``rms`` is the value itself; peak, 1x and ISO velocity are proportional
    to it; the bearing block is quiet unless ``bearing_of(index)`` says
    otherwise, so every amplitude indicator classifies alike. Every
    snapshot carries the ``context_digest`` the snapshot code computes for
    its declaration and the point, as a recorded snapshot does; with
    ``placeholder_digests`` it carries ``"ctx"`` instead, which never
    matches the current context, so every slot reads as stale.
    """
    point = point_event() if with_point else None
    events: list[dict[str, Any]] = [point] if point is not None else []
    for offset, value in enumerate(values):
        index = start + offset
        sensor = sensor_of(index) if sensor_of else "ACC01"
        unit = unit_of(index) if unit_of else "g"
        measurement = measurement_event(
            index,
            rpm=None if index in rpm_missing else 1800.0,
            sensor_id=sensor,
            unit=unit,
        )
        events.append(measurement)
        events.append(
            snapshot_event(
                index,
                rms=value,
                processing=processing,
                unit=unit,
                one_x=0.1 * value if with_one_x else None,
                iso_velocity=0.3 * value if with_iso else None,
                bearing=bearing_of(index) if bearing_of else bearing_block(),
                context_digest=(
                    "ctx" if placeholder_digests else digest_of(measurement, point)
                ),
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
        export = measurement_event(3, measurement_id=mid(103))
        events.append(export)
        events.append(
            snapshot_event(
                3,
                rms=5.0,
                measurement_id=mid(103),
                context_digest=digest_of(export, events[0]),
            )
        )
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
            export = measurement_event(11, measurement_id=measurement_id)
            events.append(export)
            events.append(
                snapshot_event(
                    11,
                    rms=SPIKE,
                    measurement_id=measurement_id,
                    context_digest=digest_of(export, events[0]),
                )
            )
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

    def test_current_context_leaves_no_stale_snapshot(self):
        """Snapshots carrying the digest the current point declaration
        produces are not stale: no count and no re-processing suggestion."""
        result = assess(view_of(series_events(STABLE + [1.0])))
        assert result["status"] == "assessed"
        assert result["lineage"]["stale_context"] == 0
        assert "previous point declaration" not in result["message"]
        assert REPROCESS_CALL not in result["message"]
        # The placeholder digest of the other fixtures never matches the
        # current context: every one of their evaluated slots counts as stale.
        placeholder = assess(
            view_of(series_events(STABLE + [1.0], placeholder_digests=True))
        )
        assert (
            placeholder["lineage"]["stale_context"] == placeholder["lineage"]["covered"]
        )

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
# Restored interpretation: a declaration changed and changed back
# ---------------------------------------------------------------------------

#: Recording instants of the later point declarations, after every
#: acquisition of the fixtures, in declaration order.
REDECLARED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)
RESTORED_AT = datetime(2026, 9, 8, tzinfo=timezone.utc)


def of_type(events: Sequence[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == event_type]


def without(
    events: Sequence[dict[str, Any]], removed: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The event list with *removed* taken out (same order otherwise)."""
    gone = {event["event_id"] for event in removed}
    return [event for event in events if event["event_id"] not in gone]


def b_snapshot(
    measurement: dict[str, Any], point: dict[str, Any], rms: float, **overrides: Any
) -> dict[str, Any]:
    """The snapshot a measurement gets under point declaration *point*: the
    series' proportional indicators, the digest of that context and the
    point's machine group in the ISO block."""
    index = int(measurement["payload"]["measurement_id"], 16)
    fields: dict[str, Any] = {
        "rms": rms,
        "measurement_id": measurement["payload"]["measurement_id"],
        "one_x": 0.1 * rms,
        "iso_velocity": 0.3 * rms,
        "bearing": bearing_block(),
        "context_digest": digest_of(measurement, point),
        "point_version": point["payload"]["declaration_version"],
        "machine_group": point["payload"]["machine_group"],
    }
    fields.update(overrides)
    return snapshot_event(index, **fields)


def revert_events(
    values: Sequence[float],
    b_values: Sequence[float],
    *,
    restore: bool = True,
    processing: str = LINEAGE_A,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Point A and one snapshot per value; point B (machine group 1) with
    every measurement re-processed under it; then point A again (version 3,
    the content of version 1). Returns the events and the snapshot events
    computed under B."""
    events = series_events(values, processing=processing)
    point_b = point_event(2, machine_group=1, recorded_at=REDECLARED_AT)
    events.append(point_b)
    measurements = of_type(events, EVENT_MEASUREMENT_RECORDED)
    b_snapshots = [
        b_snapshot(measurement, point_b, value, processing=processing)
        for measurement, value in zip(measurements, b_values)
    ]
    events.extend(b_snapshots)
    if restore:
        events.append(point_event(3, recorded_at=RESTORED_AT))
    return events, b_snapshots


def recomputed(
    events: Sequence[dict[str, Any]],
    point: dict[str, Any],
    *,
    processing: str = LINEAGE_A,
) -> list[dict[str, Any]]:
    """The snapshots every measurement of *events* gets under *point*, each
    with the rms of its latest snapshot: appended, they bring every slot to
    the current interpretation without changing a value."""
    rms = {
        event["payload"]["measurement_id"]: event["payload"]["indicators"]["rms"]
        for event in of_type(events, EVENT_HEALTH_SNAPSHOT_COMPUTED)
    }
    return [
        b_snapshot(
            measurement,
            point,
            rms[measurement["payload"]["measurement_id"]],
            processing=processing,
        )
        for measurement in of_type(events, EVENT_MEASUREMENT_RECORDED)
    ]


class TestRestoredInterpretation:
    """A declaration changed from A to B and back to A: the snapshot A
    produced is already recorded, so nothing new is appended, and the
    assessment must score it although the lineage's latest snapshot is B's.
    The latest snapshot is used only when no snapshot of the current
    interpretation exists."""

    revert = staticmethod(revert_events)

    def test_point_revert_scores_the_restored_snapshots(self):
        """Covers AE1: the result equals the one the same ledger gives
        without the snapshots computed under B."""
        events, b_snapshots = self.revert(STABLE + [1.0], [2.0] * 11)
        view = view_of(events)
        # The scenario is the real one: A's snapshot was recorded first and
        # the view still reports B's as the lineage's latest.
        slot = view["measurements"][mid(11)]
        assert len(slot["snapshots"]) == 2
        assert slot["snapshots_by_lineage"][LINEAGE_A]["snapshot_id"] == (
            b_snapshots[-1]["payload"]["snapshot_id"]
        )
        result = assess(view)
        assert result["status"] == "assessed"
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest"]["rms"] == pytest.approx(1.0)
        assert result["observed"]["reference_statistics"]["rms"]["mean"] == (
            pytest.approx(sum(STABLE) / len(STABLE))
        )
        assert canonical_json(result) == canonical_json(
            assess(view_of(without(events, b_snapshots)))
        )

    def test_machine_group_revert_restores_the_iso_boundary(self):
        """Only the machine group changes, so every indicator but the ISO
        context is equal: the B/C boundary is group 2's again."""
        events, _ = self.revert(STABLE + [1.0], STABLE + [1.0])
        iso_change = assess(view_of(events))["derived"]["iso_change"]
        assert iso_change["machine_group"] == 2
        assert iso_change["boundary_bc_mm_s"] == pytest.approx(2.8)

    def test_measurement_revert_scores_the_restored_snapshot(self):
        """Covers AE2: rpm 1800, then 1500, then 1800 on the latest slot."""
        events = series_events(STABLE + [1.0])
        point = events[0]
        changed = measurement_event(11, rpm=1500.0, version=2)
        events.append(changed)
        events.append(
            b_snapshot(
                changed,
                point,
                3.0,
                one_x=0.9,
                iso_velocity=0.9,
                bearing=bearing_block(envelope=2.0),
            )
        )
        events.append(measurement_event(11, rpm=1800.0, version=3))
        result = assess(view_of(events))
        latest = result["observed"]["latest"]
        assert result["lineage"]["stale_context"] == 0
        assert latest["rms"] == pytest.approx(1.0)
        assert latest["one_x"] == pytest.approx(0.1)
        assert latest["envelope_BPFO"] == pytest.approx(0.5)
        assert result["assessed"]["classification"] == "no_change"

    def test_interpretation_never_computed_falls_back_to_the_latest(self):
        """A, B, C with no snapshot under C: every slot keeps the lineage's
        latest snapshot (B's) and counts as stale."""
        events, _ = self.revert(STABLE + [1.0], [2.0] * 11, restore=False)
        events.append(point_event(3, machine_group=3, recorded_at=RESTORED_AT))
        result = assess(view_of(events))
        assert result["lineage"]["stale_context"] == 11
        assert result["observed"]["latest"]["rms"] == pytest.approx(2.0)

    def test_changing_back_to_b_matches_the_b_snapshots(self):
        events, _ = self.revert(STABLE + [1.0], [2.0] * 11)
        events.append(point_event(4, machine_group=1, recorded_at=RESTORED_AT + WEEK))
        result = assess(view_of(events))
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest"]["rms"] == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "order, expected",
        [
            (("match", "match_again", "other"), 1.1),
            (("match_again", "match", "other"), 1.0),
            (("match", "other", "other_again"), 1.0),
            (("other", "match", "other_again"), 1.0),
        ],
    )
    def test_matching_snapshot_is_found_wherever_it_was_recorded(self, order, expected):
        """Two payloads can share a snapshot id (another server version
        recorded it): the later one is used. Snapshots of other contexts
        recorded before or after never displace the match."""
        events = series_events(STABLE)
        point = events[0]
        measurement = measurement_event(11)
        events.append(measurement)
        candidates = {
            "match": b_snapshot(measurement, point, 1.0),
            "match_again": b_snapshot(measurement, point, 1.1, point_version=3),
            "other": b_snapshot(measurement, point, 2.0, context_digest="ctx"),
            "other_again": b_snapshot(
                measurement, point, 2.5, context_digest="ctx-other"
            ),
        }
        events.extend(candidates[name] for name in order)
        result = assess(view_of(events))
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest"]["rms"] == pytest.approx(expected)

    def test_older_complete_lineage_is_matched_while_the_current_has_holes(self):
        """The current lineage covers four slots only, so the complete
        lineage is chosen as today; inside it the restored interpretation
        is matched."""
        events, _ = self.revert(STABLE + [1.0], [2.0] * 11, restore=False)
        point_b = of_type(events, EVENT_MEASUREMENT_POINT_DECLARED)[-1]
        measurements = of_type(events, EVENT_MEASUREMENT_RECORDED)
        events.extend(
            b_snapshot(measurement, point_b, 2.0, processing=processing_id())
            for measurement in measurements[:4]
        )
        events.append(point_event(3, recorded_at=RESTORED_AT))
        lineage = assess(view_of(events))["lineage"]
        assert lineage["processing_id"] == LINEAGE_A
        assert lineage["is_current"] is False
        assert lineage["missing_for_current"] == 7
        assert lineage["stale_context"] == 0

    def test_slots_first_loaded_under_b_are_the_only_stale_ones(self):
        """Nine slots reverted, two loaded only while B was declared: the
        reference mixes nine A snapshots and one B snapshot."""
        events, _ = self.revert([1.0] * 9, [2.0] * 9, restore=False)
        point_b = of_type(events, EVENT_MEASUREMENT_POINT_DECLARED)[-1]
        for index in (10, 11):
            measurement = measurement_event(index)
            events.append(measurement)
            events.append(b_snapshot(measurement, point_b, 2.0))
        events.append(point_event(3, recorded_at=RESTORED_AT))
        result = assess(view_of(events))
        assert result["lineage"]["stale_context"] == 2
        assert result["observed"]["reference_statistics"]["rms"]["mean"] == (
            pytest.approx(1.1)
        )
        assert result["observed"]["latest"]["rms"] == pytest.approx(2.0)

    def test_collapsed_duplicate_export_keeps_its_matched_representative(self):
        events = series_events(STABLE + [1.0])
        point_a = events[0]
        export = measurement_event(11, measurement_id=mid(111))
        events.append(export)
        events.append(b_snapshot(export, point_a, 1.0))
        point_b = point_event(2, machine_group=1, recorded_at=REDECLARED_AT)
        events.append(point_b)
        events.extend(
            b_snapshot(measurement, point_b, 2.0)
            for measurement in of_type(events, EVENT_MEASUREMENT_RECORDED)
        )
        events.append(point_event(3, recorded_at=RESTORED_AT))
        result = assess(view_of(events))
        collapsed = result["comparability"]["collapsed_duplicates"]
        assert [c["measurement_id"] for c in collapsed] == [mid(11)]
        assert collapsed[0]["kept_measurement_id"] == mid(111)
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest"]["rms"] == pytest.approx(1.0)

    def test_point_revert_keeps_every_baseline_member(self):
        events, _ = self.revert(STABLE * 3, [2.0] * 30)
        events.append(baseline_event([1, 2, 3, 4, 5]))
        result = assess(view_of(events))
        reference = result["reference"]
        assert reference["kind"] == "declared_baseline"
        assert reference["measurement_ids"] == [mid(i) for i in range(1, 6)]
        assert result["lineage"]["stale_context"] == 0
        assert reference_rms_mean(result) == pytest.approx(sum(STABLE[:5]) / 5)

    def test_measurement_revert_of_a_member_still_supersedes_it(self):
        """Unchanged behavior: baseline membership compares declaration
        versions, so a member re-loaded with B and then with A again is
        superseded although its interpretation is the validated one."""
        events = series_events(STABLE * 2) + [baseline_event([1, 2, 3])]
        point = events[0]
        changed = measurement_event(2, rpm=1500.0, version=2)
        events.append(changed)
        events.append(b_snapshot(changed, point, 3.0))
        events.append(measurement_event(2, rpm=1800.0, version=3))
        result = assess(view_of(events))
        assert result["status"] == "insufficient_history"
        assert "baseline_member_superseded" in result["message"]

    def test_unresolvable_declaration_keeps_todays_handling(self):
        """A declaration the snapshot code cannot resolve (an rpm recorded as
        text) leaves nothing to match: the slot is scored from its latest
        snapshot and is not counted as stale."""
        events = series_events(STABLE)
        events.append(measurement_event(11, rpm="1800"))
        events.append(snapshot_event(11, rms=1.0))
        result = assess(view_of(events))
        assessed_ids = [
            ref["measurement_id"]
            for ref in result["observed"]["measurement_ids_assessed"]
        ]
        assert mid(11) in assessed_ids
        assert result["lineage"]["stale_context"] == 0
        assert result["observed"]["latest"]["rms"] == pytest.approx(1.0)

    def test_revert_assessment_is_deterministic_and_replayable(self):
        events, _ = self.revert(STABLE + [1.0], [2.0] * 11)
        view = view_of(events)
        first = canonical_json(assess(view))
        assert canonical_json(assess(view)) == first
        assert canonical_json(assess(view_of(list(events)))) == first


def reference_rms_mean(result: dict[str, Any]) -> float:
    return float(result["observed"]["reference_statistics"]["rms"]["mean"])


# ---------------------------------------------------------------------------
# Stale evidence: the opening sentence and the remedies that can act
# ---------------------------------------------------------------------------

CURRENT_LINEAGE = processing_id()
REPROCESS_CAVEAT = (
    "in bounded batches across the point's stale measurements, and a "
    "measurement it cannot re-process keeps its previous snapshot (the "
    "reprocess block gives the reason)"
)
STALE_VERIFICATION = (
    "Treat this classification as provisional and confirm it once every "
    "evaluated acquisition is assessed from a snapshot computed under the "
    "current declaration."
)


def stale_opening(stale: int, covered: int) -> str:
    verb = "was" if stale == 1 else "were"
    return (
        f"{stale} of the {covered} evaluated acquisition slots {verb} scored "
        "from snapshots computed under a previous declaration of the point or "
        "the measurement, so this verdict does not reflect the current "
        "declaration yet"
    )


def older_lineage_clause(missing: int, covered: int, actionable: int) -> str:
    return (
        f"; this assessment uses lineage {LINEAGE_A} because {missing} of the "
        f"{covered} evaluated slots still lack a snapshot of the current "
        f"lineage {CURRENT_LINEAGE}, re-processing writes lineage "
        f"{CURRENT_LINEAGE} only and leaves the {LINEAGE_A} snapshots unchanged, "
        f"and the assessment moves to {CURRENT_LINEAGE} once every evaluated "
        f"slot has one; {REPROCESS_CALL} attempts to compute the {actionable} "
        f"missing or outdated {CURRENT_LINEAGE} snapshot(s) of the evaluated "
        f"slots {REPROCESS_CAVEAT}"
    )


def recorded_later_clause() -> str:
    return (
        f"; every evaluated slot already has a snapshot of the current lineage "
        f"{CURRENT_LINEAGE}, but lineage {LINEAGE_A} was recorded later and is "
        f"the one assessed"
    )


class TestStaleEvidence:
    """When stale_context is above 0 the message opens with how many slots
    were scored under a previous declaration, then the message the same view
    would carry with nothing stale; the re-processing call is named only
    when re-processing writes something for the evaluated slots."""

    def test_redeclared_point_counts_every_evaluated_slot_as_stale(self):
        """A re-declaration changing the resolved context (machine group)
        with no snapshot added since. The fixture's lineage is not the
        current one, so the remedy says re-processing writes the current
        lineage and leaves the assessed one unchanged."""
        events = series_events(STABLE + [1.0])
        point_v2 = point_event(2, machine_group=1, recorded_at=REDECLARED_AT)
        events.append(point_v2)
        result = assess(view_of(events))
        assert result["status"] == "assessed"
        assert result["lineage"]["covered"] == 11
        assert result["lineage"]["stale_context"] == 11
        current = assess(view_of(events + recomputed(events, point_v2)))
        assert current["lineage"]["stale_context"] == 0
        assert result["message"] == (
            stale_opening(11, 11)
            + older_lineage_clause(11, 11, 11)
            + ". "
            + current["message"]
        )
        assert result["suggested_verification"] == STALE_VERIFICATION

    def test_unresolvable_declaration_counts_toward_the_remedy_only(self):
        """Re-processing treats a declaration it cannot resolve as stale (it
        then reports it), so the remedy counts it among the snapshots to
        compute, while the stale count, which cannot judge it, does not."""
        events = series_events(STABLE)
        events.append(measurement_event(11, rpm="1800"))
        events.append(snapshot_event(11, rms=1.0))
        events.append(point_event(2, machine_group=1, recorded_at=REDECLARED_AT))
        result = assess(view_of(events))
        assert result["lineage"]["stale_context"] == 10
        assert result["message"].startswith(
            stale_opening(10, 11) + older_lineage_clause(11, 11, 11) + ". "
        )

    def test_current_lineage_names_the_call_with_its_caveat(self):
        """Covers AE4: a slot first loaded while B was declared is the only
        stale one; its classification is no_change, which alone would call
        for no verification."""
        events, _ = revert_events(
            STABLE, STABLE, restore=False, processing=CURRENT_LINEAGE
        )
        point_b = of_type(events, EVENT_MEASUREMENT_POINT_DECLARED)[-1]
        late = measurement_event(11)
        events += [late, b_snapshot(late, point_b, 1.0, processing=CURRENT_LINEAGE)]
        point_a = point_event(3, recorded_at=RESTORED_AT)
        events.append(point_a)
        result = assess(view_of(events))
        assert result["lineage"]["is_current"] is True
        assert result["lineage"]["stale_context"] == 1
        current = assess(
            view_of(
                events + [b_snapshot(late, point_a, 1.0, processing=CURRENT_LINEAGE)]
            )
        )
        assert current["lineage"]["stale_context"] == 0
        assert result["message"] == (
            stale_opening(1, 11)
            + f"; {REPROCESS_CALL} attempts to recompute them {REPROCESS_CAVEAT}. "
            + current["message"]
        )
        assert result["assessed"]["classification"] == "no_change"
        assert result["suggested_verification"] == STALE_VERIFICATION
        assert current["suggested_verification"] is None

    def test_older_lineage_clause_counts_the_slots_still_missing(self):
        """Covers AE6: while re-processing fills the current lineage, the
        stale count on the assessed lineage stays 25 and the clause shows the
        count of slots still missing fall; once the current lineage covers
        every slot the assessment moves to it."""
        values = STABLE * 2 + STABLE[:5]
        events = series_events(values)
        point_b = point_event(2, machine_group=1, recorded_at=REDECLARED_AT)
        events.append(point_b)
        filled = recomputed(events, point_b, processing=CURRENT_LINEAGE)
        for done in (0, 10, 20):
            result = assess(view_of(events + filled[:done]))
            lineage = result["lineage"]
            assert lineage["processing_id"] == LINEAGE_A
            assert lineage["stale_context"] == 25
            assert lineage["missing_for_current"] == 25 - done
            assert result["message"].startswith(
                stale_opening(25, 25)
                + older_lineage_clause(25 - done, 25, 25 - done)
                + ". "
            )
        final = assess(view_of(events + filled))["lineage"]
        assert final["processing_id"] == CURRENT_LINEAGE
        assert final["stale_context"] == 0

    def test_older_lineage_recorded_later_has_nothing_to_write(self):
        """Covers AE7: the current lineage holds every current snapshot and
        an older lineage, recorded later, holds snapshots of another
        context. No call is named."""
        values = STABLE + [1.0]
        events = series_events(values, processing=CURRENT_LINEAGE)
        point = events[0]
        older = [
            b_snapshot(measurement, point, value, context_digest="ctx")
            for measurement, value in zip(
                of_type(events, EVENT_MEASUREMENT_RECORDED), values
            )
        ]
        result = assess(view_of(events + older))
        lineage = result["lineage"]
        assert lineage["processing_id"] == LINEAGE_A
        assert lineage["missing_for_current"] == 0
        assert lineage["stale_context"] == 11
        current = assess(view_of(events + older + recomputed(events, point)))
        assert current["lineage"]["processing_id"] == LINEAGE_A
        assert current["lineage"]["stale_context"] == 0
        assert result["message"] == (
            stale_opening(11, 11)
            + recorded_later_clause()
            + ", so re-processing has nothing to write on lineage "
            + f"{CURRENT_LINEAGE} for the evaluated slots. "
            + current["message"]
        )
        assert REPROCESS_CALL not in result["message"]

    def test_nothing_to_write_ignores_measurements_outside_the_evaluated_set(self):
        """A non-comparable measurement without a current-lineage snapshot is
        something re-processing may still reach; the sentence only speaks of
        the evaluated slots, so it stays as it is."""
        values = STABLE + [1.0]
        events = series_events(values, processing=CURRENT_LINEAGE)
        point = events[0]
        measurements = of_type(events, EVENT_MEASUREMENT_RECORDED)
        older = [
            b_snapshot(measurement, point, value, context_digest="ctx")
            for measurement, value in zip(measurements, values)
        ]
        other_axis = measurement_event(12, direction="vertical")
        events += older + [
            other_axis,
            b_snapshot(other_axis, point, 1.0, context_digest="ctx"),
        ]
        result = assess(view_of(events))
        excluded = [e["measurement_id"] for e in result["comparability"]["excluded"]]
        assert mid(12) in excluded
        assert result["lineage"]["stale_context"] == 11
        assert (
            stale_opening(11, 11)
            + recorded_later_clause()
            + ", so re-processing has nothing to write on lineage "
            + f"{CURRENT_LINEAGE} for the evaluated slots. "
        ) in result["message"]

    def test_older_lineage_recorded_later_names_the_call_for_outdated_ones(self):
        """The current lineage covers every slot, but its snapshot of the
        latest slot was computed under another context: re-processing has
        that one to rewrite, and the assessment moves as soon as it does."""
        events = series_events(STABLE, processing=CURRENT_LINEAGE)
        point = events[0]
        late = measurement_event(11)
        events += [
            late,
            b_snapshot(
                late, point, 1.0, processing=CURRENT_LINEAGE, context_digest="ctx"
            ),
        ]
        older = [
            b_snapshot(measurement, point, value, context_digest="ctx")
            for measurement, value in zip(
                of_type(events, EVENT_MEASUREMENT_RECORDED), STABLE + [1.0]
            )
        ]
        result = assess(view_of(events + older))
        lineage = result["lineage"]
        assert (lineage["processing_id"], lineage["missing_for_current"]) == (
            LINEAGE_A,
            0,
        )
        assert result["message"].startswith(
            stale_opening(11, 11)
            + recorded_later_clause()
            + f"; {REPROCESS_CALL} attempts to recompute the 1 {CURRENT_LINEAGE} "
            f"snapshot(s) computed under a previous declaration {REPROCESS_CAVEAT}, "
            f"and the assessment moves to {CURRENT_LINEAGE} as soon as "
            f"re-processing appends a {CURRENT_LINEAGE} snapshot to one of the "
            "evaluated slots. "
        )


# ---------------------------------------------------------------------------
# Contract: determinism, serialization, purity
# ---------------------------------------------------------------------------

#: Canonical payloads of views no revert touched, recorded before the
#: snapshot selection inside a lineage changed: those views must keep their
#: payloads byte for byte.
PAYLOAD_GOLDEN = Path(__file__).parent / "fixtures" / "assessment_payload_golden.json"


def golden_views() -> dict[str, list[dict[str, Any]]]:
    return {
        "stable": series_events(STABLE + [1.0]),
        "spike": series_events(STABLE + [SPIKE]),
        "ramp_with_rpm_gaps": series_events(
            STABLE + [SPIKE] + STABLE[:9] + RAMP, rpm_missing=(3, 6)
        ),
        "declared_baseline": series_events(STABLE * 3)
        + [baseline_event([1, 2, 3, 4, 5])],
    }


def write_payload_golden(path: Path = PAYLOAD_GOLDEN) -> None:
    """Regenerate the payload golden, only when an output change is intended::

    python -c "from tests.test_asset_ledger_assessment import \\
        write_payload_golden; write_payload_golden()"
    """
    payloads = {
        name: json.loads(canonical_json(assess(view_of(events))))
        for name, events in golden_views().items()
    }
    path.write_text(
        json.dumps(payloads, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class TestContract:
    def test_views_without_a_revert_keep_their_payloads(self):
        golden = json.loads(PAYLOAD_GOLDEN.read_text(encoding="utf-8"))
        views = golden_views()
        assert set(golden) == set(views)
        for name, events in views.items():
            assert canonical_json(assess(view_of(events))) == canonical_json(
                golden[name]
            ), name

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
        assert set(result["lineage"]) == {
            "processing_id",
            "algorithm_version",
            "covered",
            "candidates",
            "current_processing_id",
            "is_current",
            "missing_for_current",
            "stale_context",
        }
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
