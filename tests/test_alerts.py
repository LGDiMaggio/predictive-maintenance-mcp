"""Tests for decision_support.alerts — ISO 10816/20816 alert thresholds."""

import pytest

from predictive_maintenance_mcp.decision_support.alerts import (
    check_alert_thresholds,
    check_custom_alert,
    define_custom_thresholds,
)


class TestCheckAlertThresholds:
    """Tests for ISO 10816 threshold alerting."""

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
        assert result["exceeded_threshold"] == 7.1  # C upper for group 2 rigid

    def test_different_machine_groups(self):
        """Group 1 rigid has higher thresholds than group 2 rigid."""
        # 3.0 mm/s is Zone B for group 1 rigid (A≤2.8) but Zone B for group 2 rigid (A≤2.3)
        g1 = check_alert_thresholds(3.0, machine_group=1, support_type="rigid")
        g2 = check_alert_thresholds(3.0, machine_group=2, support_type="rigid")

        assert g1["zone"] == "B"
        assert g2["zone"] == "B"

        # 2.5 mm/s: Zone A for group 1, Zone B for group 2
        g1_low = check_alert_thresholds(2.5, machine_group=1, support_type="rigid")
        g2_low = check_alert_thresholds(2.5, machine_group=2, support_type="rigid")

        assert g1_low["zone"] == "A"
        assert g2_low["zone"] == "B"

    def test_flexible_vs_rigid_support(self):
        """Flexible supports have higher Zone A threshold than rigid."""
        # 3.0 mm/s: Zone B for group 2 rigid (A≤2.3), Zone A for group 2 flexible (A≤3.5)
        rigid = check_alert_thresholds(3.0, machine_group=2, support_type="rigid")
        flex = check_alert_thresholds(3.0, machine_group=2, support_type="flexible")

        assert rigid["zone"] == "B"
        assert flex["zone"] == "A"


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
