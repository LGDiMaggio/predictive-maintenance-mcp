"""Tests for asset_ledger.comparability (U4) — ISO 13374 Block 3.

One rule per test. The module grades a measurement against the current
declaration of its point and against the reference context ("a
contradiction excludes, an absence qualifies"). These tests pin, from the
plan's U4 scenarios:

- units: another family excludes (AE4), the same family converts with the
  ISO factor, an undeclared unit excludes with the re-load remedy;
- direction: a different declared direction is another point (no mapping
  between axes and machine directions), an undeclared one qualifies, and
  the reference direction is the anchor when the point declares none;
- sensor, speed (5% / 10% against nominal_rpm or the reference median, the
  two-pass anchor), operating condition, acquisition setup;
- amplitudes that are not physical (WAV, raw integer without scale);
- timestamps (naive, suspect, collision) and a changed point declaration;
- the reference context, the aggregation, the closed vocabulary (every
  code produced by at least one test and absent from at least one) and
  the module's import purity.
"""

import ast
from pathlib import Path
from typing import Any, Callable

import pytest

from predictive_maintenance_mcp.asset_ledger import comparability as c
from predictive_maintenance_mcp.diagnostics.iso20816 import G_TO_M_S2, M_S_TO_MM_S
from predictive_maintenance_mcp.signal_acquisition.measurement import (
    UNIT_FAMILIES,
    VALID_DIRECTIONS,
)
from predictive_maintenance_mcp.signal_acquisition.repository import (
    VALID_SIGNAL_UNITS,
)

ACQUIRED_AT = "2026-08-20T11:42:00+00:00"
CHANGED_AT = "2026-09-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def declaration(**overrides: Any) -> dict[str, Any]:
    """A bare, fully declared measurement declaration (all rules satisfied)."""
    base: dict[str, Any] = {
        "asset_id": "P-101",
        "measurement_point_id": "motor_de_h",
        "acquired_at": ACQUIRED_AT,
        "timezone_declared": True,
        "timestamp_suspect": False,
        "rpm": 1480.0,
        "load": 75.0,
        "operating_state": "loaded",
        "sensor_id": "STWIN_BOX_001",
        "direction": "horizontal",
        "declared_by": "adapter:stwinbox",
        "measurement_id": "0123456789abcdef",
        "channel_index": 0,
        "sampling_rate": 25600.0,
        "signal_unit": "g",
        "raw_format": None,
    }
    base.update(overrides)
    return base


def payload(location: str = "signals/m01.csv", **overrides: Any) -> dict[str, Any]:
    """A ``measurement_recorded`` payload wrapping :func:`declaration`."""
    block = declaration(**overrides)
    return {
        "measurement_id": block["measurement_id"],
        "declaration": block,
        "file": {"location": location},
    }


def point(**overrides: Any) -> dict[str, Any]:
    """A fully declared ``measurement_point_declared`` payload."""
    base: dict[str, Any] = {
        "measurement_point_id": "motor_de_h",
        "declaration_version": 1,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": "STWIN_BOX_001",
        "expected_direction": "horizontal",
        "nominal_rpm": 1480.0,
        "declared_by": "test",
        "note": None,
    }
    base.update(overrides)
    return base


def reference(n: int = 10, **overrides: Any) -> list[dict[str, Any]]:
    """*n* reference payloads with distinct ids and instants."""
    return [
        payload(
            measurement_id=f"{i:016x}",
            acquired_at=f"2026-08-{10 + i:02d}T11:42:00+00:00",
            **overrides,
        )
        for i in range(n)
    ]


def codes(assessment: dict[str, Any]) -> list[str]:
    return [entry["code"] for entry in assessment["qualifications"]]


def detail(assessment: dict[str, Any], code: str) -> str:
    matches = [e["detail"] for e in assessment["qualifications"] if e["code"] == code]
    assert len(matches) == 1, f"{code} expected exactly once in {assessment}"
    return matches[0]


def assess(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return c.assess_measurement_comparability(*args, **kwargs)


# ---------------------------------------------------------------------------
# Vocabulary and thresholds
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_codes_are_unique(self):
        assert len(set(c.QUALIFICATION_CODES)) == len(c.QUALIFICATION_CODES)

    def test_excluding_and_informational_codes_partition_inside_the_vocabulary(self):
        assert c.EXCLUDING_CODES <= set(c.QUALIFICATION_CODES)
        assert c.INFORMATIONAL_CODES <= set(c.QUALIFICATION_CODES)
        assert not (c.EXCLUDING_CODES & c.INFORMATIONAL_CODES)

    def test_grades(self):
        assert c.COMPARABILITY_GRADES == ("comparable", "qualified", "non_comparable")

    def test_threshold_defaults(self):
        thresholds = c.DEFAULT_THRESHOLDS
        assert thresholds.rpm_comparable_pct == 5.0
        assert thresholds.rpm_qualified_pct == 10.0
        assert thresholds.sampling_rate_tolerance_pct == 1.0
        assert thresholds.load_tolerance_pct == 5.0
        assert c.ComparabilityThresholds() == thresholds

    def test_assessment_shape(self):
        result = assess(payload(), point())
        assert set(result) == {
            "grade",
            "qualifications",
            "conversion",
            "excluded_from_amplitude_trend",
        }
        assert result["grade"] == "comparable"
        assert result["qualifications"] == []
        assert result["conversion"] is None
        assert result["excluded_from_amplitude_trend"] is False

    def test_every_qualification_carries_code_and_detail(self):
        result = assess(
            payload(
                signal_unit=None, direction=None, rpm=None, timezone_declared=False
            ),
            point(),
        )
        for entry in result["qualifications"]:
            assert set(entry) == {"code", "detail"}
            assert entry["code"] in c.QUALIFICATION_CODES
            assert entry["detail"]

    def test_intrinsic_checks_only_without_point_and_context(self):
        clean = assess(payload())
        assert clean["grade"] == "comparable"
        assert clean["qualifications"] == []
        intrinsic = assess(
            payload(
                location="signals/m01.wav",
                signal_unit=None,
                rpm=None,
                timezone_declared=False,
                timestamp_suspect=True,
            )
        )
        assert codes(intrinsic) == [
            "unit_not_declared",
            "rpm_not_declared",
            "amplitude_not_physical",
            "timezone_not_declared",
            "timestamp_suspect",
        ]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnitRule:
    def test_ae4_velocity_measurement_on_acceleration_point_is_non_comparable(self):
        result = assess(payload(signal_unit="mm/s"), point(expected_signal_unit="g"))
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["unit_incompatible"]
        assert "mm/s vs g" in detail(result, "unit_incompatible")
        assert result["conversion"] is None

    def test_same_family_is_comparable_with_conversion_recorded(self):
        result = assess(payload(signal_unit="m/s2"), point(expected_signal_unit="g"))
        assert result["grade"] == "comparable"
        assert codes(result) == ["unit_converted"]
        assert result["conversion"]["from"] == "m/s2"
        assert result["conversion"]["to"] == "g"
        assert result["conversion"]["factor"] == pytest.approx(1.0 / G_TO_M_S2)

    def test_same_unit_needs_no_conversion(self):
        result = assess(payload(signal_unit="g"), point(expected_signal_unit="g"))
        assert result["conversion"] is None
        assert "unit_converted" not in codes(result)

    def test_unit_not_declared_with_identity_is_non_comparable_with_remedy(self):
        result = assess(payload(signal_unit=None), point(expected_signal_unit="g"))
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["unit_not_declared"]
        text = detail(result, "unit_not_declared")
        assert "re-load with signal_unit=g" in text
        assert "supersedes the previous one" in text

    def test_unit_not_declared_without_expected_unit_names_the_vocabulary(self):
        text = detail(assess(payload(signal_unit=None)), "unit_not_declared")
        assert "re-load with signal_unit=" in text
        for unit in VALID_SIGNAL_UNITS:
            assert unit in text

    def test_reference_unit_is_the_anchor_when_the_point_declares_none(self):
        context = c.build_reference_context(
            reference(signal_unit="mm/s"), point(expected_signal_unit=None)
        )
        assert context["unit"] == "mm/s"
        assert context["unit_source"] == "reference_majority"
        result = assess(
            payload(signal_unit="g"), point(expected_signal_unit=None), context
        )
        assert result["grade"] == "non_comparable"
        assert "g vs mm/s" in detail(result, "unit_incompatible")
        converted = assess(
            payload(signal_unit="m/s"), point(expected_signal_unit=None), context
        )
        assert converted["conversion"] == {
            "from": "m/s",
            "to": "mm/s",
            "factor": pytest.approx(M_S_TO_MM_S),
        }

    def test_no_expected_unit_anywhere_means_no_unit_comparison(self):
        result = assess(payload(signal_unit="mm/s"), point(expected_signal_unit=None))
        assert result["grade"] == "comparable"
        assert result["conversion"] is None

    @pytest.mark.parametrize(
        "from_unit, to_unit, factor",
        [
            ("g", "m/s2", G_TO_M_S2),
            ("m/s2", "g", 1.0 / G_TO_M_S2),
            ("m/s", "mm/s", M_S_TO_MM_S),
            ("mm/s", "m/s", 1.0 / M_S_TO_MM_S),
            ("g", "g", 1.0),
            ("mm/s", "mm/s", 1.0),
        ],
    )
    def test_conversion_factors_come_from_the_iso_constants(
        self, from_unit, to_unit, factor
    ):
        assert c.unit_conversion_factor(from_unit, to_unit) == pytest.approx(factor)

    @pytest.mark.parametrize(
        "from_unit, to_unit",
        [("g", "mm/s"), ("m/s", "m/s2"), (None, "g"), ("g", None), ("foo", "g")],
    )
    def test_no_factor_across_families_or_for_unknown_units(self, from_unit, to_unit):
        assert c.unit_conversion_factor(from_unit, to_unit) is None

    def test_conversion_table_covers_the_whole_unit_vocabulary(self):
        every_unit = {unit for units in UNIT_FAMILIES.values() for unit in units}
        assert every_unit == set(VALID_SIGNAL_UNITS)
        for family, units in UNIT_FAMILIES.items():
            for a in units:
                for b in units:
                    assert c.unit_conversion_factor(a, b) is not None, (family, a, b)
        for a in UNIT_FAMILIES["acceleration"]:
            for b in UNIT_FAMILIES["velocity"]:
                assert c.unit_conversion_factor(a, b) is None


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


class TestDirectionRule:
    def test_horizontal_on_vertical_point_is_non_comparable(self):
        result = assess(
            payload(direction="horizontal"), point(expected_direction="vertical")
        )
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["direction_mismatch"]
        assert "horizontal vs vertical" in detail(result, "direction_mismatch")

    def test_absent_direction_is_qualified(self):
        result = assess(payload(direction=None), point(expected_direction="vertical"))
        assert result["grade"] == "qualified"
        assert codes(result) == ["direction_not_declared"]

    def test_axis_x_on_horizontal_point_is_non_comparable_no_mapping(self):
        assert "x" in VALID_DIRECTIONS and "horizontal" in VALID_DIRECTIONS
        result = assess(payload(direction="x"), point(expected_direction="horizontal"))
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["direction_mismatch"]

    def test_same_direction_is_comparable(self):
        result = assess(payload(direction="axial"), point(expected_direction="axial"))
        assert result["grade"] == "comparable"

    def test_reference_direction_is_the_anchor_when_the_point_declares_none(self):
        no_direction = point(expected_direction=None)
        context = c.build_reference_context(reference(direction="x"), no_direction)
        assert context["direction"] == "x"
        assert context["direction_source"] == "reference_majority"
        result = assess(payload(direction="y"), no_direction, context)
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["direction_differs_from_reference"]
        assert "y vs x" in detail(result, "direction_differs_from_reference")

    def test_measurement_along_the_reference_direction_is_comparable(self):
        no_direction = point(expected_direction=None)
        context = c.build_reference_context(reference(direction="x"), no_direction)
        result = assess(payload(direction="x"), no_direction, context)
        assert result["grade"] == "comparable"
        assert result["qualifications"] == []

    def test_declared_expected_direction_wins_over_the_reference(self):
        declared = point(expected_direction="y")
        context = c.build_reference_context(reference(direction="x"), declared)
        assert context["direction"] == "y"
        assert context["direction_source"] == "expected_direction"
        along_point = assess(payload(direction="y"), declared, context)
        assert along_point["grade"] == "comparable"
        along_reference = assess(payload(direction="x"), declared, context)
        assert codes(along_reference) == ["direction_mismatch"]

    def test_no_direction_anchor_anywhere_means_no_direction_comparison(self):
        no_direction = point(expected_direction=None)
        context = c.build_reference_context(reference(direction=None), no_direction)
        assert context["direction"] is None
        result = assess(payload(direction="z"), no_direction, context)
        assert result["grade"] == "comparable"


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


class TestSensorRule:
    def test_different_sensor_is_qualified(self):
        result = assess(payload(sensor_id="STWIN_BOX_002"), point())
        assert result["grade"] == "qualified"
        assert codes(result) == ["sensor_changed"]
        assert "STWIN_BOX_002 vs STWIN_BOX_001" in detail(result, "sensor_changed")

    def test_same_sensor_is_not_qualified(self):
        assert "sensor_changed" not in codes(assess(payload(), point()))

    def test_undeclared_sensor_is_not_graded(self):
        result = assess(payload(sensor_id=None), point())
        assert result["grade"] == "comparable"

    def test_reference_majority_sensor_is_the_anchor_when_the_point_declares_none(
        self,
    ):
        no_sensor = point(expected_sensor_id=None)
        members = reference(6, sensor_id="S-A") + reference(4, sensor_id="S-B")
        context = c.build_reference_context(members, no_sensor)
        assert context["sensor_id"] == "S-A"
        assert context["sensor_id_source"] == "reference_majority"
        result = assess(payload(sensor_id="S-B"), no_sensor, context)
        assert codes(result) == ["sensor_changed"]
        assert assess(payload(sensor_id="S-A"), no_sensor, context)["grade"] == (
            "comparable"
        )


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


class TestRpmRule:
    def test_ae3_absent_rpm_is_qualified_constant_regime(self):
        result = assess(payload(rpm=None), point())
        assert result["grade"] == "qualified"
        assert codes(result) == ["rpm_not_declared"]
        assert "trend assumes constant regime" in detail(result, "rpm_not_declared")

    def test_rpm_equal_to_the_anchor_is_comparable(self):
        result = assess(payload(rpm=1480.0), point(nominal_rpm=1480.0))
        assert result["grade"] == "comparable"
        assert result["qualifications"] == []

    def test_rpm_within_five_percent_is_comparable(self):
        result = assess(payload(rpm=1520.0), point(nominal_rpm=1480.0))
        assert result["grade"] == "comparable"

    def test_rpm_between_five_and_ten_percent_is_qualified_with_percentage(self):
        result = assess(payload(rpm=1560.0), point(nominal_rpm=1480.0))
        assert result["grade"] == "qualified"
        assert codes(result) == ["rpm_deviation_qualified"]
        text = detail(result, "rpm_deviation_qualified")
        assert "1560 vs 1480 rpm" in text
        assert "+5.4%" in text
        assert "±5%" in text

    def test_rpm_beyond_ten_percent_is_non_comparable_with_numbers(self):
        result = assess(payload(rpm=2960.0), point(nominal_rpm=1480.0))
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["rpm_deviation_excluded"]
        text = detail(result, "rpm_deviation_excluded")
        assert "2960 vs 1480 rpm, +100%, tolerance ±10%" in text

    def test_negative_deviation_is_signed(self):
        result = assess(payload(rpm=1369.0), point(nominal_rpm=1480.0))
        assert "-7.5%" in detail(result, "rpm_deviation_qualified")

    def test_thresholds_are_parameters(self):
        wide = c.ComparabilityThresholds(
            rpm_comparable_pct=6.0, rpm_qualified_pct=200.0
        )
        assert assess(payload(rpm=1560.0), point(), thresholds=wide)["grade"] == (
            "comparable"
        )
        big = assess(payload(rpm=2960.0), point(), thresholds=wide)
        assert big["grade"] == "qualified"
        assert "±6%" in detail(big, "rpm_deviation_qualified")

    def test_anchor_is_nominal_rpm_when_the_point_declares_it(self):
        context = c.build_reference_context(
            reference(rpm=1480.0), point(nominal_rpm=1500)
        )
        assert context["rpm_anchor"] == 1500.0
        assert context["rpm_anchor_source"] == "nominal_rpm"
        result = assess(payload(rpm=1700.0), point(nominal_rpm=1500), context)
        assert "1700 vs 1500 rpm, +13.3%" in detail(result, "rpm_deviation_excluded")
        assert "nominal_rpm" in detail(result, "rpm_deviation_excluded")

    def test_tolerance_boundaries_are_inclusive(self):
        p = point(nominal_rpm=1500.0)
        assert assess(payload(rpm=1575.0), p)["qualifications"] == []  # +5%
        at_ten = assess(payload(rpm=1650.0), p)  # +10%
        assert codes(at_ten) == ["rpm_deviation_qualified"]
        assert "+10%" in detail(at_ten, "rpm_deviation_qualified")
        beyond = assess(payload(rpm=1651.0), p)
        assert codes(beyond) == ["rpm_deviation_excluded"]

    def test_anchor_falls_back_to_the_reference_median(self):
        members = [payload(rpm=rpm) for rpm in (1470.0, 1480.0, 1490.0, 3000.0)]
        context = c.build_reference_context(members, point(nominal_rpm=None))
        assert context["rpm_anchor"] == 1485.0
        assert context["rpm_anchor_source"] == "reference_median"
        result = assess(payload(rpm=1485.0), point(nominal_rpm=None), context)
        assert result["grade"] == "comparable"

    def test_no_anchor_at_all_qualifies_a_declared_rpm(self):
        no_nominal = point(nominal_rpm=None)
        context = c.build_reference_context(reference(rpm=None), no_nominal)
        assert context["rpm_anchor"] is None
        assert context["rpm_anchor_source"] is None
        result = assess(payload(rpm=1480.0), no_nominal, context)
        assert result["grade"] == "qualified"
        assert codes(result) == ["no_rpm_anchor"]
        assert "no rpm anchor" in detail(result, "no_rpm_anchor")

    def test_no_anchor_code_is_not_raised_without_a_reference_context(self):
        # Load time: the reference median may exist later, so a declared rpm
        # on a point without nominal_rpm is not graded (no noise per load).
        result = assess(payload(rpm=1480.0), point(nominal_rpm=None))
        assert result["grade"] == "comparable"
        assert assess(payload(rpm=1480.0))["qualifications"] == []

    def test_two_pass_anchor_from_the_first_ten_slots(self):
        rpms = [1480.0] * 9 + [2960.0]
        slots = [
            payload(
                measurement_id=f"{i:016x}",
                acquired_at=f"2026-08-{10 + i:02d}T11:42:00+00:00",
                rpm=rpm,
            )
            for i, rpm in enumerate(rpms)
        ]
        no_nominal = point(nominal_rpm=None)
        context = c.build_reference_context(slots, no_nominal)
        assert context["rpm_anchor"] == 1480.0
        assert context["rpm_anchor_source"] == "reference_median"
        assert context["reference_count"] == 10
        graded = [assess(slot, no_nominal, context) for slot in slots]
        assert [g["grade"] for g in graded[:9]] == ["comparable"] * 9
        assert graded[9]["grade"] == "non_comparable"
        assert codes(graded[9]) == ["rpm_deviation_excluded"]


# ---------------------------------------------------------------------------
# Operating condition
# ---------------------------------------------------------------------------


class TestOperatingConditionRule:
    def test_operating_state_different_from_the_reference_is_qualified(self):
        context = c.build_reference_context(
            reference(operating_state="loaded"), point()
        )
        result = assess(payload(operating_state="idle"), point(), context)
        assert result["grade"] == "qualified"
        assert codes(result) == ["operating_condition_differs"]
        text = detail(result, "operating_condition_differs")
        assert "'idle'" in text and "'loaded'" in text

    def test_load_different_from_the_reference_median_is_qualified(self):
        context = c.build_reference_context(reference(load=75.0), point())
        result = assess(payload(load=20.0), point(), context)
        assert codes(result) == ["operating_condition_differs"]
        assert "load 20 vs 75" in detail(result, "operating_condition_differs")

    def test_load_within_tolerance_is_not_qualified(self):
        context = c.build_reference_context(reference(load=75.0), point())
        assert assess(payload(load=74.0), point(), context)["qualifications"] == []

    def test_both_differences_share_one_qualification(self):
        context = c.build_reference_context(reference(), point())
        result = assess(payload(operating_state="idle", load=0.0), point(), context)
        assert codes(result) == ["operating_condition_differs"]
        text = detail(result, "operating_condition_differs")
        assert "operating_state" in text and "load 0 vs 75" in text

    def test_undeclared_condition_is_not_graded(self):
        context = c.build_reference_context(reference(), point())
        result = assess(payload(operating_state=None, load=None), point(), context)
        assert result["qualifications"] == []

    def test_no_reference_context_means_no_condition_comparison(self):
        result = assess(payload(operating_state="idle", load=0.0), point())
        assert result["qualifications"] == []


# ---------------------------------------------------------------------------
# Acquisition setup
# ---------------------------------------------------------------------------


class TestAcquisitionSetupRule:
    def test_sampling_rate_different_from_the_reference_is_qualified_with_values(
        self,
    ):
        context = c.build_reference_context(reference(sampling_rate=25600.0), point())
        result = assess(payload(sampling_rate=12800.0), point(), context)
        assert result["grade"] == "qualified"
        assert codes(result) == ["acquisition_setup_differs"]
        text = detail(result, "acquisition_setup_differs")
        assert "12800" in text and "25600" in text

    def test_sampling_rate_within_tolerance_is_not_qualified(self):
        context = c.build_reference_context(reference(sampling_rate=25600.0), point())
        result = assess(payload(sampling_rate=25700.0), point(), context)
        assert result["qualifications"] == []

    def test_scale_factor_different_from_the_reference_is_qualified_with_values(
        self,
    ):
        raw = {"sample_format": "int16", "scale_factor": 0.002}
        context = c.build_reference_context(reference(raw_format=raw), point())
        assert context["scale_factor"] == 0.002
        result = assess(
            payload(raw_format={"sample_format": "int16", "scale_factor": 0.001}),
            point(),
            context,
        )
        assert codes(result) == ["acquisition_setup_differs"]
        text = detail(result, "acquisition_setup_differs")
        assert "scale_factor 0.001 vs 0.002" in text

    def test_same_setup_is_not_qualified(self):
        raw = {"sample_format": "float32", "scale_factor": 0.002}
        context = c.build_reference_context(reference(raw_format=raw), point())
        result = assess(payload(raw_format=dict(raw)), point(), context)
        assert result["qualifications"] == []

    def test_scale_factor_is_compared_only_when_both_sides_declare_one(self):
        context = c.build_reference_context(reference(), point())
        assert context["scale_factor"] is None
        raw = {"sample_format": "float32", "scale_factor": 0.002}
        assert assess(payload(raw_format=raw), point(), context)["qualifications"] == []


# ---------------------------------------------------------------------------
# Amplitude
# ---------------------------------------------------------------------------


class TestAmplitudeRule:
    def test_wav_file_is_non_comparable_for_amplitude(self):
        result = assess(payload(location="signals/m01.wav"), point())
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["amplitude_not_physical"]
        assert result["excluded_from_amplitude_trend"] is True
        assert "WAV" in detail(result, "amplitude_not_physical")

    def test_wav_detection_ignores_letter_case(self):
        result = assess(payload(location="C:/captures/M01.WAV"), point())
        assert codes(result) == ["amplitude_not_physical"]

    def test_raw_integer_without_scale_factor_is_non_comparable_for_amplitude(self):
        raw = {"sample_format": "int16", "byte_order": "little", "scale_factor": None}
        result = assess(payload(location="signals/m01.bin", raw_format=raw), point())
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["amplitude_not_physical"]
        assert result["excluded_from_amplitude_trend"] is True
        text = detail(result, "amplitude_not_physical")
        assert "int16" in text and "scale_factor" in text

    def test_raw_integer_with_scale_factor_is_physical(self):
        raw = {"sample_format": "int32", "scale_factor": 0.001}
        result = assess(payload(location="signals/m01.bin", raw_format=raw), point())
        assert result["grade"] == "comparable"
        assert result["excluded_from_amplitude_trend"] is False

    def test_raw_float_without_scale_factor_is_physical(self):
        raw = {"sample_format": "float32", "scale_factor": None}
        result = assess(payload(location="signals/m01.bin", raw_format=raw), point())
        assert result["grade"] == "comparable"

    def test_other_exclusions_do_not_set_the_amplitude_flag(self):
        result = assess(payload(signal_unit="mm/s"), point())
        assert result["grade"] == "non_comparable"
        assert result["excluded_from_amplitude_trend"] is False

    def test_bare_declaration_with_filepath_is_inspected_too(self):
        result = assess(declaration(filepath="/data/signals/m01.wav"), point())
        assert codes(result) == ["amplitude_not_physical"]


# ---------------------------------------------------------------------------
# Timestamps and point declaration
# ---------------------------------------------------------------------------


class TestTimestampRules:
    def test_timezone_not_declared_is_qualified(self):
        result = assess(payload(timezone_declared=False), point())
        assert result["grade"] == "qualified"
        assert codes(result) == ["timezone_not_declared"]
        assert ACQUIRED_AT in detail(result, "timezone_not_declared")

    def test_timestamp_suspect_is_qualified(self):
        suspect = "1970-01-01T00:12:00+00:00"
        result = assess(payload(acquired_at=suspect, timestamp_suspect=True), point())
        assert result["grade"] == "qualified"
        assert codes(result) == ["timestamp_suspect"]
        assert suspect in detail(result, "timestamp_suspect")

    def test_timestamp_collision_is_qualified(self):
        result = assess(payload(), point(), timestamp_collision=True)
        assert result["grade"] == "qualified"
        assert codes(result) == ["timestamp_collision"]
        assert ACQUIRED_AT in detail(result, "timestamp_collision")

    def test_three_timestamp_qualifications_are_distinct_and_never_exclude(self):
        result = assess(
            payload(timezone_declared=False, timestamp_suspect=True),
            point(),
            timestamp_collision=True,
        )
        assert codes(result) == [
            "timezone_not_declared",
            "timestamp_suspect",
            "timestamp_collision",
        ]
        assert result["grade"] == "qualified"
        excluded = assess(
            payload(
                signal_unit="mm/s", timezone_declared=False, timestamp_suspect=True
            ),
            point(),
            timestamp_collision=True,
        )
        assert excluded["grade"] == "non_comparable"
        assert set(codes(result)) <= set(codes(excluded))

    def test_measurement_before_a_changed_point_declaration_is_qualified_with_date(
        self,
    ):
        result = assess(
            payload(),
            point(declaration_version=2),
            point_declaration_changed_at=CHANGED_AT,
        )
        assert result["grade"] == "qualified"
        assert codes(result) == ["point_declaration_changed"]
        text = detail(result, "point_declaration_changed")
        assert CHANGED_AT in text and "version 2" in text

    def test_changed_unit_after_the_measurement_is_graded_by_the_new_context(self):
        result = assess(
            payload(signal_unit="mm/s"),
            point(expected_signal_unit="g", declaration_version=2),
            point_declaration_changed_at=CHANGED_AT,
        )
        assert result["grade"] == "non_comparable"
        assert codes(result) == ["unit_incompatible", "point_declaration_changed"]

    def test_measurement_after_the_point_declaration_is_not_qualified(self):
        result = assess(
            payload(acquired_at="2026-09-02T00:00:00+00:00"),
            point(),
            point_declaration_changed_at=CHANGED_AT,
        )
        assert result["qualifications"] == []

    def test_offsets_are_compared_as_instants(self):
        # 01:00 at +02:00 is 23:00 UTC of the previous day: before CHANGED_AT.
        result = assess(
            payload(acquired_at="2026-09-01T01:00:00+02:00"),
            point(),
            point_declaration_changed_at=CHANGED_AT,
        )
        assert codes(result) == ["point_declaration_changed"]

    def test_unparsable_change_instant_is_ignored(self):
        result = assess(payload(), point(), point_declaration_changed_at="yesterday")
        assert result["qualifications"] == []


# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------


class TestInputForms:
    def test_bare_declaration_is_graded_like_the_wrapped_payload(self):
        wrapped = assess(payload(rpm=1560.0, sensor_id="X"), point())
        bare = assess(declaration(rpm=1560.0, sensor_id="X"), point())
        assert bare == wrapped

    def test_point_none_and_context_none_are_accepted(self):
        assert assess(payload(), None, None)["grade"] == "comparable"

    def test_missing_optional_keys_are_treated_as_undeclared(self):
        minimal = {
            "asset_id": "P-101",
            "measurement_point_id": "motor_de_h",
            "acquired_at": ACQUIRED_AT,
            "signal_unit": "g",
        }
        result = assess(minimal, point())
        assert codes(result) == ["direction_not_declared", "rpm_not_declared"]
        assert result["grade"] == "qualified"


# ---------------------------------------------------------------------------
# Reference context
# ---------------------------------------------------------------------------


class TestReferenceContext:
    def test_keys(self):
        context = c.build_reference_context(reference(), point())
        assert set(context) == {
            "reference_count",
            "measurement_ids",
            "rpm_anchor",
            "rpm_anchor_source",
            "direction",
            "direction_source",
            "sensor_id",
            "sensor_id_source",
            "unit",
            "unit_source",
            "operating_state",
            "load",
            "sampling_rate",
            "scale_factor",
            "point_declaration_version",
            "thresholds",
        }
        assert context["reference_count"] == 10
        assert context["measurement_ids"] == [f"{i:016x}" for i in range(10)]
        assert context["point_declaration_version"] == 1
        assert context["thresholds"] == c.DEFAULT_THRESHOLDS._asdict()

    def test_point_declarations_win_over_the_reference(self):
        members = reference(direction="x", sensor_id="S-B", signal_unit="mm/s")
        context = c.build_reference_context(members, point())
        assert (context["direction"], context["direction_source"]) == (
            "horizontal",
            "expected_direction",
        )
        assert (context["sensor_id"], context["sensor_id_source"]) == (
            "STWIN_BOX_001",
            "expected_sensor_id",
        )
        assert (context["unit"], context["unit_source"]) == (
            "g",
            "expected_signal_unit",
        )

    def test_majority_ties_go_to_the_first_in_order(self):
        no_direction = point(expected_direction=None)
        members = [payload(direction=d) for d in ("x", "y", "x", "y")]
        assert c.build_reference_context(members, no_direction)["direction"] == "x"
        members = [payload(direction=d) for d in ("y", "x", "x", "y")]
        assert c.build_reference_context(members, no_direction)["direction"] == "y"

    def test_majority_ignores_undeclared_values(self):
        no_direction = point(expected_direction=None)
        members = [payload(direction=d) for d in (None, None, None, "z")]
        assert c.build_reference_context(members, no_direction)["direction"] == "z"

    def test_medians_of_load_and_sampling_rate(self):
        members = [
            payload(load=load, sampling_rate=fs)
            for load, fs in ((50.0, 25600.0), (100.0, 12800.0), (75.0, 25600.0))
        ]
        context = c.build_reference_context(members, point())
        assert context["load"] == 75.0
        assert context["sampling_rate"] == 25600.0

    def test_operating_state_majority(self):
        members = reference(3, operating_state="idle") + reference(
            5, operating_state="loaded"
        )
        assert (
            c.build_reference_context(members, point())["operating_state"] == "loaded"
        )

    def test_empty_reference_without_point(self):
        context = c.build_reference_context([], None)
        assert context["reference_count"] == 0
        assert context["measurement_ids"] == []
        for key in (
            "rpm_anchor",
            "rpm_anchor_source",
            "direction",
            "direction_source",
            "sensor_id",
            "unit",
            "operating_state",
            "load",
            "sampling_rate",
            "scale_factor",
            "point_declaration_version",
        ):
            assert context[key] is None, key

    def test_bare_declarations_are_accepted_as_reference(self):
        members = [declaration(rpm=rpm) for rpm in (1480.0, 1500.0, 1490.0)]
        context = c.build_reference_context(members, point(nominal_rpm=None))
        assert context["rpm_anchor"] == 1490.0
        assert context["measurement_ids"] == ["0123456789abcdef"] * 3

    def test_thresholds_are_recorded(self):
        wide = c.ComparabilityThresholds(rpm_comparable_pct=7.5)
        context = c.build_reference_context(reference(), point(), wide)
        assert context["thresholds"]["rpm_comparable_pct"] == 7.5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestSummary:
    def test_counts_per_grade_and_per_code(self):
        p = point()
        assessments = [
            assess(payload(signal_unit="mm/s"), p),  # non_comparable
            assess(payload(direction="vertical"), p),  # non_comparable
            assess(payload(rpm=2960.0), p),  # non_comparable
            assess(payload(rpm=None), p),  # qualified
            assess(payload(rpm=None), p),  # qualified
            assess(payload(sensor_id="OTHER"), p),  # qualified
            assess(payload(direction=None), p),  # qualified
            assess(payload(), p),  # comparable
            assess(payload(), p),  # comparable
            assess(payload(signal_unit="m/s2"), p),  # comparable, converted
        ]
        summary = c.summarize_comparability(assessments)
        assert summary["total"] == 10
        assert summary["grades"] == {
            "comparable": 3,
            "qualified": 4,
            "non_comparable": 3,
        }
        assert summary["codes"] == {
            "unit_incompatible": 1,
            "unit_converted": 1,
            "direction_mismatch": 1,
            "direction_not_declared": 1,
            "sensor_changed": 1,
            "rpm_not_declared": 2,
            "rpm_deviation_excluded": 1,
        }
        assert list(summary["codes"]) == [
            code for code in c.QUALIFICATION_CODES if code in summary["codes"]
        ]

    def test_empty_summary(self):
        assert c.summarize_comparability([]) == {
            "total": 0,
            "grades": {"comparable": 0, "qualified": 0, "non_comparable": 0},
            "codes": {},
        }

    def test_unknown_grade_is_refused(self):
        with pytest.raises(ValueError, match="Unknown comparability grade"):
            c.summarize_comparability([{"grade": "maybe", "qualifications": []}])

    def test_foreign_codes_are_counted_after_the_vocabulary(self):
        assessments = [
            {
                "grade": "qualified",
                "qualifications": [
                    {"code": "baseline_member_superseded", "detail": "x"},
                    {"code": "rpm_not_declared", "detail": "y"},
                ],
            }
        ]
        summary = c.summarize_comparability(assessments)
        assert list(summary["codes"]) == [
            "rpm_not_declared",
            "baseline_member_superseded",
        ]

    def test_grade_of_is_the_rule_used_by_the_assessment(self):
        assert c.grade_of([]) == "comparable"
        assert c.grade_of([{"code": "unit_converted", "detail": ""}]) == "comparable"
        assert c.grade_of([{"code": "sensor_changed", "detail": ""}]) == "qualified"
        both = [
            {"code": "sensor_changed", "detail": ""},
            {"code": "unit_incompatible", "detail": ""},
        ]
        assert c.grade_of(both) == "non_comparable"


# ---------------------------------------------------------------------------
# Every code produced at least once and absent at least once
# ---------------------------------------------------------------------------


def _context(members: list[dict[str, Any]], p: dict[str, Any]) -> dict[str, Any]:
    return c.build_reference_context(members, p)


PRODUCERS: dict[str, Callable[[], dict[str, Any]]] = {
    "unit_not_declared": lambda: assess(payload(signal_unit=None), point()),
    "unit_incompatible": lambda: assess(payload(signal_unit="mm/s"), point()),
    "unit_converted": lambda: assess(payload(signal_unit="m/s2"), point()),
    "direction_mismatch": lambda: assess(payload(direction="vertical"), point()),
    "direction_differs_from_reference": lambda: assess(
        payload(direction="y"),
        point(expected_direction=None),
        _context(reference(direction="x"), point(expected_direction=None)),
    ),
    "direction_not_declared": lambda: assess(payload(direction=None), point()),
    "sensor_changed": lambda: assess(payload(sensor_id="OTHER"), point()),
    "rpm_not_declared": lambda: assess(payload(rpm=None), point()),
    "no_rpm_anchor": lambda: assess(
        payload(),
        point(nominal_rpm=None),
        _context(reference(rpm=None), point(nominal_rpm=None)),
    ),
    "rpm_deviation_qualified": lambda: assess(payload(rpm=1560.0), point()),
    "rpm_deviation_excluded": lambda: assess(payload(rpm=2960.0), point()),
    "operating_condition_differs": lambda: assess(
        payload(operating_state="idle"), point(), _context(reference(), point())
    ),
    "acquisition_setup_differs": lambda: assess(
        payload(sampling_rate=12800.0), point(), _context(reference(), point())
    ),
    "amplitude_not_physical": lambda: assess(payload(location="m.wav"), point()),
    "timezone_not_declared": lambda: assess(payload(timezone_declared=False), point()),
    "timestamp_suspect": lambda: assess(payload(timestamp_suspect=True), point()),
    "timestamp_collision": lambda: assess(payload(), point(), timestamp_collision=True),
    "point_declaration_changed": lambda: assess(
        payload(), point(), point_declaration_changed_at=CHANGED_AT
    ),
}


class TestVocabularyCoverage:
    def test_producers_cover_the_vocabulary_exactly(self):
        assert set(PRODUCERS) == set(c.QUALIFICATION_CODES)

    @pytest.mark.parametrize("code", c.QUALIFICATION_CODES)
    def test_code_is_produced_by_its_scenario_and_absent_from_the_clean_one(self, code):
        produced = PRODUCERS[code]()
        assert codes(produced) == [code]
        expected_grade = (
            "non_comparable"
            if code in c.EXCLUDING_CODES
            else "comparable" if code in c.INFORMATIONAL_CODES else "qualified"
        )
        assert produced["grade"] == expected_grade
        clean = assess(payload(), point(), _context(reference(), point()))
        assert clean["grade"] == "comparable"
        assert code not in codes(clean)


# ---------------------------------------------------------------------------
# Module purity
# ---------------------------------------------------------------------------


class TestModulePurity:
    def test_imports_nothing_from_mcp_repository_or_models(self):
        tree = ast.parse(Path(c.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = {"repository", "models", "mcp", "mcp_tools", "fastmcp", "server"}
        for name in imported:
            assert not (set(name.split(".")) & forbidden), name
        assert "diagnostics.iso20816" in imported
        assert "signal_acquisition.measurement" in imported

    def test_package_reexports(self):
        from predictive_maintenance_mcp import asset_ledger

        for name in (
            "QUALIFICATION_CODES",
            "ComparabilityThresholds",
            "build_reference_context",
            "assess_measurement_comparability",
            "summarize_comparability",
        ):
            assert getattr(asset_ledger, name) is getattr(c, name), name
