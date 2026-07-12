"""Tests for decision_support.alerts — ISO 20816-3 alert thresholds.

The alert path must classify with the SAME zone boundaries as the severity
engine (``diagnostics.iso20816``): boundary values from ISO 10816-3:2009.
Before unification the two paths disagreed (audit finding 2.4).
"""

import numpy as np
import pytest

from predictive_maintenance_mcp.decision_support.alerts import (
    check_alert_thresholds,
    check_custom_alert,
    define_custom_thresholds,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import assess_severity_raw


def _velocity_signal(
    rms_target: float,
    fs: float = 10000.0,
    duration: float = 1.0,
    freq: float = 50.0,
) -> np.ndarray:
    """Sine signal with a known RMS (mm/s) inside the 10-1000 Hz ISO band."""
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    return rms_target * np.sqrt(2) * np.sin(2 * np.pi * freq * t)


class TestSingleThresholdTable:
    """Same reading -> same zone from every path (audit 2.4 reunification)."""

    def test_group2_rigid_3mm_s_zone_c_everywhere(self):
        # 3.0 mm/s, group 2, rigid: Zone C per ISO 10816-3:2009
        # (boundaries 1.4 / 2.8 / 4.5 -> 2.8 < 3.0 <= 4.5).
        # Before unification the alert path said B while the engine said C.
        alert = check_alert_thresholds(3.0, machine_group=2, support_type="rigid")
        engine = assess_severity_raw(
            _velocity_signal(3.0),
            fs=10000,
            machine_group=2,
            support_type="rigid",
            signal_unit="mm/s",
        )
        assert engine["zone"] == "C"
        assert alert["zone"] == "C"
        assert alert["alert_level"] == "alarm"

    @pytest.mark.parametrize(
        "group,support",
        [(1, "rigid"), (1, "flexible"), (2, "rigid"), (2, "flexible")],
        ids=lambda v: str(v),
    )
    @pytest.mark.parametrize("rms", [0.5, 1.5, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0])
    def test_alert_zone_matches_engine_zone(self, group, support, rms):
        alert = check_alert_thresholds(rms, machine_group=group, support_type=support)
        engine = assess_severity_raw(
            _velocity_signal(rms),
            fs=10000,
            machine_group=group,
            support_type=support,
            signal_unit="mm/s",
        )
        assert alert["zone"] == engine["zone"]


class TestCheckAlertThresholds:
    """Zone classification via the alert path (group 2 rigid: 1.4/2.8/4.5)."""

    def test_zone_a_no_alert(self):
        """Low velocity (group 2 rigid) should be Zone A / no alert."""
        result = check_alert_thresholds(1.0, machine_group=2, support_type="rigid")

        assert result["alert_level"] == "none"
        assert result["zone"] == "A"
        assert result["exceeded_threshold"] is None

    def test_zone_d_danger(self):
        """Very high velocity should trigger Zone D / danger."""
        result = check_alert_thresholds(20.0, machine_group=2, support_type="rigid")

        assert result["alert_level"] == "danger"
        assert result["zone"] == "D"
        assert result["exceeded_threshold"] == 4.5  # C/D boundary, group 2 rigid

    def test_different_machine_groups(self):
        """Group 1 has higher thresholds than group 2 (same support)."""
        # 3.0 mm/s: Zone B for group 1 rigid (2.3 < 3.0 <= 4.5),
        # Zone C for group 2 rigid (2.8 < 3.0 <= 4.5)
        g1 = check_alert_thresholds(3.0, machine_group=1, support_type="rigid")
        g2 = check_alert_thresholds(3.0, machine_group=2, support_type="rigid")

        assert g1["zone"] == "B"
        assert g2["zone"] == "C"

        # 2.0 mm/s: Zone A for group 1 (<= 2.3), Zone B for group 2 (> 1.4)
        g1_low = check_alert_thresholds(2.0, machine_group=1, support_type="rigid")
        g2_low = check_alert_thresholds(2.0, machine_group=2, support_type="rigid")

        assert g1_low["zone"] == "A"
        assert g2_low["zone"] == "B"

    def test_flexible_vs_rigid_support(self):
        """Flexible supports have higher boundaries than rigid ones."""
        # 3.0 mm/s group 2: Zone C rigid (2.8 < 3.0), Zone B flexible (2.3 < 3.0 <= 4.5)
        rigid = check_alert_thresholds(3.0, machine_group=2, support_type="rigid")
        flex = check_alert_thresholds(3.0, machine_group=2, support_type="flexible")

        assert rigid["zone"] == "C"
        assert flex["zone"] == "B"

    def test_group1_flexible_supported(self):
        """Group 1 flexible (3.5/7.1/11.0) is a valid combination."""
        result = check_alert_thresholds(3.0, machine_group=1, support_type="flexible")
        assert result["zone"] == "A"
        assert result["alert_level"] == "none"

    def test_unknown_group_raises(self):
        """U9 error contract: invalid group is misuse and RAISES — the old
        soft fallback ('unknown' zone + manual-review warning) returned a
        wrong-looking success instead of an actionable error."""
        with pytest.raises(ValueError, match="machine_group"):
            check_alert_thresholds(5.0, machine_group=99, support_type="rigid")


class TestCustomThresholds:
    """Tests for user-defined threshold alerting."""

    def test_custom_thresholds(self):
        """Custom thresholds should override ISO defaults."""
        thresholds = define_custom_thresholds(warning=5.0, alarm=10.0, danger=15.0)

        low = check_custom_alert(3.0, thresholds)
        assert low["alert_level"] == "none"
        assert low["zone"] == "A"

        mid = check_custom_alert(7.0, thresholds)
        assert mid["alert_level"] == "warning"
        assert mid["zone"] == "B"

        high = check_custom_alert(12.0, thresholds)
        assert high["alert_level"] == "alarm"
        assert high["zone"] == "C"

        extreme = check_custom_alert(20.0, thresholds)
        assert extreme["alert_level"] == "danger"
        assert extreme["zone"] == "D"
