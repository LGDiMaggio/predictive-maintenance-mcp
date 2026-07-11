"""Tests for MCP decision support tools (ISO 13374 Block 6)."""

import pytest
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.decision_support_tools import register
from predictive_maintenance_mcp.models import AlertResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = FastMCP("test-decision-support")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# check_vibration_alert
# ---------------------------------------------------------------------------

class TestCheckVibrationAlert:
    """Tests for check_vibration_alert tool."""

    @pytest.mark.asyncio
    async def test_zone_a(self, tools, mock_ctx):
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=1.0,
            machine_group=2,
            support_type="rigid",
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "none"
        assert result.zone == "A"
        assert result.rms_velocity == 1.0

    @pytest.mark.asyncio
    async def test_zone_b(self, tools, mock_ctx):
        # Group 2 rigid boundaries (ISO 10816-3:2009): 1.4 / 2.8 / 4.5 mm/s
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=2.0,
            machine_group=2,
            support_type="rigid",
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "warning"
        assert result.zone == "B"

    @pytest.mark.asyncio
    async def test_zone_c(self, tools, mock_ctx):
        # 3.0 mm/s group 2 rigid is Zone C (2.8 < 3.0 <= 4.5); the old
        # duplicated table wrongly called it Zone B.
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=3.0,
            machine_group=2,
            support_type="rigid",
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "alarm"
        assert result.zone == "C"

    @pytest.mark.asyncio
    async def test_zone_d(self, tools, mock_ctx):
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=10.0,
            machine_group=2,
            support_type="rigid",
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "danger"
        assert result.zone == "D"

    @pytest.mark.asyncio
    async def test_flexible_support(self, tools, mock_ctx):
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=4.0,
            machine_group=2,
            support_type="flexible",
        )
        assert isinstance(result, AlertResult)
        assert result.zone == "B"

    @pytest.mark.asyncio
    async def test_unknown_group(self, tools, mock_ctx):
        result = await tools["check_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=5.0,
            machine_group=99,
            support_type="rigid",
        )
        assert isinstance(result, AlertResult)
        assert result.zone == "unknown"


# ---------------------------------------------------------------------------
# check_custom_vibration_alert
# ---------------------------------------------------------------------------

class TestCheckCustomVibrationAlert:
    """Tests for check_custom_vibration_alert tool."""

    @pytest.mark.asyncio
    async def test_normal_zone(self, tools, mock_ctx):
        result = await tools["check_custom_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=0.5,
            warning_threshold=1.0,
            alarm_threshold=3.0,
            danger_threshold=5.0,
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "none"
        assert result.zone == "A"

    @pytest.mark.asyncio
    async def test_warning_zone(self, tools, mock_ctx):
        result = await tools["check_custom_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=2.0,
            warning_threshold=1.0,
            alarm_threshold=3.0,
            danger_threshold=5.0,
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "warning"

    @pytest.mark.asyncio
    async def test_danger_zone(self, tools, mock_ctx):
        result = await tools["check_custom_vibration_alert"](
            ctx=mock_ctx,
            rms_velocity=6.0,
            warning_threshold=1.0,
            alarm_threshold=3.0,
            danger_threshold=5.0,
        )
        assert isinstance(result, AlertResult)
        assert result.alert_level == "danger"
        assert result.zone == "D"


# ---------------------------------------------------------------------------
# generate_maintenance_recommendations
# ---------------------------------------------------------------------------

class TestGenerateMaintenanceRecommendations:
    """Tests for generate_maintenance_recommendations tool."""

    @pytest.mark.asyncio
    async def test_zone_a_recommendations(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="A",
        )
        assert isinstance(result, str)
        assert "monitoring" in result.lower() or "normal" in result.lower()

    @pytest.mark.asyncio
    async def test_zone_d_recommendations(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="D",
        )
        assert isinstance(result, str)
        assert "immediate" in result.lower() or "shutdown" in result.lower()

    @pytest.mark.asyncio
    async def test_with_fault_types(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="C",
            fault_types="outer_race,misalignment",
        )
        assert isinstance(result, str)
        assert "outer_race" in result or "bearing" in result.lower()
        assert "misalignment" in result.lower() or "alignment" in result.lower()
        assert "confidence" not in result.lower()

    @pytest.mark.asyncio
    async def test_confidence_input_rejected(self, tools, mock_ctx):
        """The tool must reject a caller-supplied confidence number."""
        with pytest.raises(TypeError):
            await tools["generate_maintenance_recommendations"](
                ctx=mock_ctx,
                severity_zone="C",
                fault_types="outer_race",
                confidence=0.87,
            )

    @pytest.mark.asyncio
    async def test_empty_fault_types(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="B",
            fault_types="",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_unknown_zone(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="X",
        )
        assert isinstance(result, str)
        assert len(result) > 0
