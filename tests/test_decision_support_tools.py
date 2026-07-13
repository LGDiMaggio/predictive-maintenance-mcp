"""Tests for MCP decision support tools (ISO 13374 Block 6).

The former alert tools (check_vibration_alert / check_custom_vibration_alert)
were absorbed into the unified ``assess_severity`` diagnostics tool in U9 —
their behavior is characterized in tests/test_golden_merges.py. This module
keeps only generate_maintenance_recommendations.
"""

import pytest
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.decision_support_tools import register


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
# Surface: the alert tools are gone (merged into assess_severity)
# ---------------------------------------------------------------------------

class TestAlertToolsMerged:
    def test_alert_tools_not_registered(self, tools):
        assert "check_vibration_alert" not in tools
        assert "check_custom_vibration_alert" not in tools
        assert set(tools) == {"generate_maintenance_recommendations"}


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
            fault_types=["outer_race", "misalignment"],
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
                fault_types=["outer_race"],
                confidence=0.87,
            )

    @pytest.mark.asyncio
    async def test_none_fault_types(self, tools, mock_ctx):
        result = await tools["generate_maintenance_recommendations"](
            ctx=mock_ctx,
            severity_zone="B",
            fault_types=None,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_unknown_fault_type_raises_with_vocabulary(
        self, tools, mock_ctx
    ):
        """U9 loop closure: an acronym like 'BPFO' is NOT silently dropped —
        the error names the canonical vocabulary and the acronym mapping."""
        with pytest.raises(ValueError) as exc:
            await tools["generate_maintenance_recommendations"](
                ctx=mock_ctx,
                severity_zone="C",
                fault_types=["BPFO"],
            )
        msg = str(exc.value)
        assert "BPFO" in msg
        for allowed in (
            "outer_race",
            "inner_race",
            "ball",
            "cage",
            "misalignment",
            "unbalance",
            "looseness",
        ):
            assert allowed in msg

    def test_fault_type_literal_matches_engine_vocabulary(self, mcp):
        """The tool's Literal vocabulary and the engine's VALID_FAULT_TYPES
        are the same closed set (single source of truth, no drift)."""
        from typing import get_args

        from predictive_maintenance_mcp.decision_support.recommendations import (
            VALID_FAULT_TYPES,
        )
        from predictive_maintenance_mcp.mcp_tools.decision_support_tools import (
            FaultType,
        )

        assert set(get_args(FaultType)) == set(VALID_FAULT_TYPES)
