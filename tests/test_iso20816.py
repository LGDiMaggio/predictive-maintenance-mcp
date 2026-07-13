"""Tests for the single ISO severity engine (``src/diagnostics/iso20816.py``).

Zone boundary values are those of ISO 10816-3:2009 (four-zone A-D scheme;
ISO 20816-3:2022 supersedes that edition and merges zones A and B). This
file is the tabular specification of the ONLY threshold table allowed in
the codebase.
"""

import numpy as np
import pytest

from predictive_maintenance_mcp.diagnostics.iso20816 import (
    THRESHOLD_PROVENANCE,
    assess_severity_raw,
    assess_severity_with_axis,
    classify_zone,
    get_zone_boundaries,
)


def make_velocity_signal(
    rms_target: float,
    fs: float = 10000,
    duration: float = 1.0,
    freq: float = 50.0,
) -> np.ndarray:
    """Generate a sine signal with a known RMS (mm/s) inside the ISO band."""
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    amplitude = rms_target * np.sqrt(2)
    return amplitude * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# Tabular boundary specification — ISO 10816-3:2009 Table A.1 (velocity)
# ---------------------------------------------------------------------------

# (machine_group, support_type) -> (A/B, B/C, C/D) boundaries in mm/s RMS
EXPECTED_BOUNDARIES = {
    (1, "rigid"): (2.3, 4.5, 7.1),
    (1, "flexible"): (3.5, 7.1, 11.0),
    (2, "rigid"): (1.4, 2.8, 4.5),
    (2, "flexible"): (2.3, 4.5, 7.1),
}

EPS = 1e-6


class TestThresholdTable:
    """The single threshold table and its zone semantics."""

    @pytest.mark.parametrize(
        "group,support",
        sorted(EXPECTED_BOUNDARIES),
        ids=lambda v: str(v),
    )
    def test_boundaries_match_iso_10816_3_2009(self, group, support):
        assert get_zone_boundaries(group, support) == EXPECTED_BOUNDARIES[
            (group, support)
        ]

    @pytest.mark.parametrize(
        "group,support",
        sorted(EXPECTED_BOUNDARIES),
        ids=lambda v: str(v),
    )
    def test_zone_at_and_above_each_boundary(self, group, support):
        """Boundary values belong to the lower zone (<= semantics)."""
        ab, bc, cd = EXPECTED_BOUNDARIES[(group, support)]
        cases = [
            (ab / 2, "A"),
            (ab, "A"),
            (ab + EPS, "B"),
            (bc, "B"),
            (bc + EPS, "C"),
            (cd, "C"),
            (cd + EPS, "D"),
            (cd * 3, "D"),
        ]
        for rms, expected_zone in cases:
            result = classify_zone(rms, machine_group=group, support_type=support)
            assert result["zone"] == expected_zone, (
                f"group={group} support={support} rms={rms}: "
                f"got {result['zone']}, expected {expected_zone}"
            )

    def test_invalid_group_raises(self):
        with pytest.raises(ValueError, match="machine_group"):
            classify_zone(1.0, machine_group=3, support_type="rigid")

    def test_invalid_support_raises(self):
        with pytest.raises(ValueError, match="support_type"):
            classify_zone(1.0, machine_group=2, support_type="floating")

    def test_negative_rms_raises(self):
        with pytest.raises(ValueError):
            classify_zone(-1.0, machine_group=2, support_type="rigid")

    def test_classify_zone_includes_provenance(self):
        result = classify_zone(3.0, machine_group=2, support_type="rigid")
        assert result["threshold_provenance"] == THRESHOLD_PROVENANCE


class TestProvenanceCitation:
    """Honest edition citation: 10816-3:2009 values, 20816-3:2022 merges A/B."""

    def test_provenance_names_both_editions(self):
        assert "10816-3:2009" in THRESHOLD_PROVENANCE
        assert "20816-3:2022" in THRESHOLD_PROVENANCE
        assert "merge" in THRESHOLD_PROVENANCE.lower()

    def test_assessment_output_carries_provenance(self):
        signal = make_velocity_signal(2.0)
        result = assess_severity_raw(signal, fs=10000, signal_unit="mm/s")
        assert "10816-3:2009" in result["threshold_provenance"]
        assert "20816-3:2022" in result["threshold_provenance"]


class TestMachineClassRemoved:
    """The invented machine_class I-IV -> group mapping no longer exists."""

    def test_assess_severity_with_axis_rejects_machine_class(self):
        signal = make_velocity_signal(1.0)
        with pytest.raises(TypeError):
            assess_severity_with_axis(signal, fs=10000, machine_class="II")

    def test_assess_severity_raw_rejects_machine_class(self):
        signal = make_velocity_signal(1.0)
        with pytest.raises(TypeError):
            assess_severity_raw(signal, fs=10000, machine_class="II")

    def test_native_vocabulary_in_output(self):
        # Group 2 rigid: 3.0 mm/s -> Zone C (2.8 < 3.0 <= 4.5)
        signal = make_velocity_signal(3.0)
        result = assess_severity_with_axis(
            signal,
            fs=10000,
            machine_group=2,
            support_type="rigid",
            signal_unit="mm/s",
        )
        assert result["zone"] == "C"
        assert result["machine_group"] == 2
        assert result["support_type"] == "rigid"
        assert "machine_class" not in result


class TestZoneClassificationEndToEnd:
    """Signal-level zone classification through the full assessment path."""

    @pytest.mark.parametrize(
        "rms,expected_zone",
        [(1.0, "A"), (2.0, "B"), (3.5, "C"), (5.0, "D")],
    )
    def test_group2_rigid_zones(self, rms, expected_zone):
        signal = make_velocity_signal(rms)
        result = assess_severity_with_axis(
            signal,
            fs=10000,
            machine_group=2,
            support_type="rigid",
            signal_unit="mm/s",
        )
        assert result["zone"] == expected_zone

    def test_group1_rigid(self):
        # 3.0 mm/s: Zone B for group 1 rigid (2.3 < 3.0 <= 4.5)
        signal = make_velocity_signal(3.0)
        result = assess_severity_with_axis(
            signal,
            fs=10000,
            machine_group=1,
            support_type="rigid",
            signal_unit="mm/s",
        )
        assert result["zone"] == "B"

    def test_group1_flexible(self):
        # 5.0 mm/s: Zone B for group 1 flexible (3.5 < 5.0 <= 7.1)
        signal = make_velocity_signal(5.0)
        result = assess_severity_with_axis(
            signal,
            fs=10000,
            machine_group=1,
            support_type="flexible",
            signal_unit="mm/s",
        )
        assert result["zone"] == "B"

    def test_severity_labels(self):
        signal = make_velocity_signal(1.0)
        result = assess_severity_with_axis(
            signal,
            fs=10000,
            machine_group=2,
            support_type="rigid",
            signal_unit="mm/s",
        )
        assert result["severity_level"] == "Good"
        assert result["color_code"] == "green"


class TestFrequencyBandHonesty:
    """A verdict is only issued when the full 10-1000 Hz ISO band is
    covered; an fs whose clamped upper edge cannot reach 1000 Hz is
    refused rather than assessed over a truncated band."""

    def test_fs_2khz_refused_partial_band(self):
        """At fs=2 kHz the usable upper edge is 950 Hz (0.95 x Nyquist),
        below the 1000 Hz ISO band top. This previously returned a
        '10-950 Hz' verdict over a truncated band; it must now refuse."""
        signal = make_velocity_signal(2.0, fs=2000)
        with pytest.raises(ValueError, match="partially covered"):
            assess_severity_raw(signal, fs=2000, signal_unit="mm/s")

    def test_fs_at_full_band_floor_accepted(self):
        """fs=2106 Hz is the lowest rate whose clamped upper edge
        (0.95 x Nyquist) reaches 1000 Hz — the verdict is issued over the
        full 10-1000 Hz band."""
        signal = make_velocity_signal(2.0, fs=2106)
        result = assess_severity_raw(signal, fs=2106, signal_unit="mm/s")
        assert "10-1000 Hz" in result["frequency_range"]

    def test_fs_just_below_floor_refused(self):
        """fs=2104 Hz (just below the 2106 Hz full-band floor) cannot cover
        the whole ISO band and is refused."""
        signal = make_velocity_signal(2.0, fs=2104)
        with pytest.raises(ValueError, match="partially covered"):
            assess_severity_raw(signal, fs=2104, signal_unit="mm/s")

    def test_fs_10khz_reports_full_band(self):
        signal = make_velocity_signal(2.0, fs=10000)
        result = assess_severity_raw(signal, fs=10000, signal_unit="mm/s")
        assert "10-1000 Hz" in result["frequency_range"]

    def test_low_rpm_uses_2hz_lower_edge(self):
        signal = make_velocity_signal(2.0)
        result = assess_severity_raw(
            signal, fs=10000, signal_unit="mm/s", operating_speed_rpm=300
        )
        assert "2-1000 Hz" in result["frequency_range"]

    def test_high_rpm_uses_10hz_lower_edge(self):
        signal = make_velocity_signal(2.0)
        result = assess_severity_raw(
            signal, fs=10000, signal_unit="mm/s", operating_speed_rpm=3000
        )
        assert "10-1000 Hz" in result["frequency_range"]

    def test_nyquist_below_1khz_refused(self):
        """fs=1.8 kHz cannot cover the ISO band: refuse instead of pretending."""
        signal = make_velocity_signal(2.0, fs=1800)
        with pytest.raises(ValueError, match="Nyquist"):
            assess_severity_raw(signal, fs=1800, signal_unit="mm/s")


class TestPowerScopeRefusal:
    """ISO 20816-3 scope floor: machines below 15 kW are refused when the
    power is declared; unknown power is documented, not guessed."""

    def test_below_15kw_refused(self):
        signal = make_velocity_signal(2.0)
        with pytest.raises(ValueError, match="15 kW"):
            assess_severity_raw(
                signal, fs=10000, signal_unit="mm/s", machine_power_kw=5.0
            )

    def test_at_15kw_accepted(self):
        signal = make_velocity_signal(2.0)
        result = assess_severity_raw(
            signal, fs=10000, signal_unit="mm/s", machine_power_kw=15.0
        )
        assert result["zone"] in ("A", "B", "C", "D")

    def test_unknown_power_not_refused(self):
        signal = make_velocity_signal(2.0)
        result = assess_severity_raw(signal, fs=10000, signal_unit="mm/s")
        assert result["zone"] in ("A", "B", "C", "D")


class TestUnitConversion:
    def test_g_to_velocity(self):
        """Acceleration in g should be converted to mm/s."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 50 * t)  # 0.5 g at 50 Hz

        result = assess_severity_with_axis(signal, fs=fs, signal_unit="g")
        assert result["unit_conversion_performed"] is True
        assert result["original_unit"] == "g"
        assert result["rms_velocity_mm_s"] > 0

    def test_ms2_to_velocity(self):
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 5.0 * np.sin(2 * np.pi * 50 * t)  # 5 m/s²

        result = assess_severity_with_axis(signal, fs=fs, signal_unit="m/s²")
        assert result["unit_conversion_performed"] is True

    def test_mm_s_no_conversion(self):
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 2.0 * np.sin(2 * np.pi * 50 * t)

        result = assess_severity_with_axis(signal, fs=fs, signal_unit="mm/s")
        assert result["unit_conversion_performed"] is False

    def test_m_s_unit(self):
        """m/s should be converted to mm/s."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.002 * np.sin(2 * np.pi * 50 * t)  # 2 mm/s = 0.002 m/s
        result = assess_severity_with_axis(signal, fs=fs, signal_unit="m/s")
        assert result["unit_conversion_performed"] is True
        assert result["original_unit"] == "m/s"

    def test_m_s2_unit(self):
        """m/s2 (without superscript) should work like m/s²."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 5.0 * np.sin(2 * np.pi * 50 * t)
        result = assess_severity_with_axis(signal, fs=fs, signal_unit="m/s2")
        assert result["unit_conversion_performed"] is True

    def test_unknown_unit_raises(self):
        """Unknown unit must be refused — units are never assumed."""
        signal = np.random.randn(10000) * 0.1
        with pytest.raises(ValueError, match="Unknown signal_unit"):
            assess_severity_with_axis(signal, fs=10000, signal_unit="mils")

    def test_none_unit_refused_not_attributeerror(self):
        """Defense-in-depth: a None unit is refused with an actionable
        ValueError, never an opaque AttributeError from .lower()
        (regression: "'NoneType' object has no attribute 'lower'").

        The live MCP boundary already guards None, but the pure conversion
        must fail cleanly if a future caller forgets — refusing, never
        guessing a unit."""
        signal = np.random.randn(10000) * 0.1
        with pytest.raises(ValueError, match="not declared"):
            assess_severity_with_axis(signal, fs=10000, signal_unit=None)


class TestInvalidParameters:
    def test_invalid_group_support(self):
        """Invalid machine_group/support_type combo should raise."""
        with pytest.raises(ValueError):
            assess_severity_raw(np.random.randn(10000), fs=10000, machine_group=3)


class TestResultStructure:
    def test_result_keys(self):
        signal = make_velocity_signal(1.0)
        result = assess_severity_with_axis(signal, fs=10000, signal_unit="mm/s")
        expected_keys = {
            "rms_velocity_mm_s",
            "machine_group",
            "support_type",
            "axis",
            "zone",
            "zone_description",
            "severity_level",
            "color_code",
            "boundaries",
            "frequency_range",
            "unit_conversion_performed",
            "original_unit",
            "threshold_provenance",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_boundaries_dict(self):
        signal = make_velocity_signal(1.0)
        result = assess_severity_with_axis(
            signal, fs=10000, machine_group=2, support_type="rigid",
            signal_unit="mm/s",
        )
        assert "AB" in result["boundaries"]
        assert "BC" in result["boundaries"]
        assert "CD" in result["boundaries"]
        assert (
            result["boundaries"]["AB"]
            < result["boundaries"]["BC"]
            < result["boundaries"]["CD"]
        )
