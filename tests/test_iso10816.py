"""Tests for ISO 10816/20816 severity assessment."""

import numpy as np
import pytest

from predictive_maintenance_mcp.iso10816 import assess_vibration_severity


class TestZoneClassification:
    """Test zone boundaries for all machine classes."""

    def _make_velocity_signal(self, rms_target, fs=10000, duration=1.0):
        """Generate a signal with known RMS velocity in mm/s."""
        n = int(fs * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        # Scale sine wave to achieve target RMS (RMS of sine = A/sqrt(2))
        amplitude = rms_target * np.sqrt(2)
        return amplitude * np.sin(2 * np.pi * 50 * t)

    def test_zone_a_class_ii(self):
        # Class II rigid: Zone A boundary = 1.4 mm/s
        signal = self._make_velocity_signal(1.0)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="II", signal_unit="mm/s"
        )
        assert result["zone"] == "A"
        assert result["severity_level"] == "Good"
        assert result["color_code"] == "green"

    def test_zone_b_class_ii(self):
        signal = self._make_velocity_signal(2.0)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="II", signal_unit="mm/s"
        )
        assert result["zone"] == "B"
        assert result["severity_level"] == "Acceptable"

    def test_zone_c_class_ii(self):
        signal = self._make_velocity_signal(3.5)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="II", signal_unit="mm/s"
        )
        assert result["zone"] == "C"
        assert result["severity_level"] == "Unsatisfactory"

    def test_zone_d_class_ii(self):
        signal = self._make_velocity_signal(5.0)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="II", signal_unit="mm/s"
        )
        assert result["zone"] == "D"
        assert result["severity_level"] == "Unacceptable"

    def test_class_i(self):
        # Class I (small): boundaries are flexible -> 2.3/4.5/7.1
        signal = self._make_velocity_signal(1.5)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="I", signal_unit="mm/s"
        )
        assert result["zone"] == "A"

    def test_class_iii(self):
        # Class III (large rigid): boundaries 2.3/4.5/7.1
        signal = self._make_velocity_signal(3.0)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="III", signal_unit="mm/s"
        )
        assert result["zone"] == "B"

    def test_class_iv(self):
        # Class IV (large flexible): boundaries 3.5/7.1/11.0
        signal = self._make_velocity_signal(5.0)
        result = assess_vibration_severity(
            signal, fs=10000, machine_class="IV", signal_unit="mm/s"
        )
        assert result["zone"] == "B"

    def test_invalid_class(self):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError, match="Invalid machine_class"):
            assess_vibration_severity(signal, fs=10000, machine_class="X")


class TestUnitConversion:
    def test_g_to_velocity(self):
        """Acceleration in g should be converted to mm/s."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 50 * t)  # 0.5g at 50 Hz

        result = assess_vibration_severity(
            signal, fs=fs, signal_unit="g"
        )
        assert result["unit_conversion_performed"] is True
        assert result["original_unit"] == "g"
        assert result["rms_velocity_mm_s"] > 0

    def test_ms2_to_velocity(self):
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 5.0 * np.sin(2 * np.pi * 50 * t)  # 5 m/s²

        result = assess_vibration_severity(
            signal, fs=fs, signal_unit="m/s²"
        )
        assert result["unit_conversion_performed"] is True

    def test_mm_s_no_conversion(self):
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 2.0 * np.sin(2 * np.pi * 50 * t)

        result = assess_vibration_severity(
            signal, fs=fs, signal_unit="mm/s"
        )
        assert result["unit_conversion_performed"] is False


class TestResultStructure:
    def test_result_keys(self):
        signal = np.random.randn(10000)
        result = assess_vibration_severity(signal, fs=10000)
        expected_keys = {
            "rms_velocity_mm_s", "machine_class", "axis", "zone",
            "zone_description", "severity_level", "color_code",
            "boundaries", "unit_conversion_performed", "original_unit",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_boundaries_dict(self):
        signal = np.random.randn(10000)
        result = assess_vibration_severity(signal, fs=10000, machine_class="II")
        assert "AB" in result["boundaries"]
        assert "BC" in result["boundaries"]
        assert "CD" in result["boundaries"]
        assert result["boundaries"]["AB"] < result["boundaries"]["BC"] < result["boundaries"]["CD"]
