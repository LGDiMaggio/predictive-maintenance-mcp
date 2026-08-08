"""Tests for MCP diagnostic prompts."""

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.prompts import register


@pytest.fixture
def mcp():
    server = MCPServer("test-prompts")
    register(server)
    return server


@pytest.fixture
def prompts(mcp):
    """Get registered prompt functions."""
    return {p.name: p.fn for p in mcp._prompt_manager._prompts.values()}


class TestDiagnoseBearingPrompt:
    """Tests for diagnose_bearing prompt."""

    def test_returns_string(self, prompts):
        result = prompts["diagnose_bearing"](signal_id="test_signal")
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_signal_id(self, prompts):
        result = prompts["diagnose_bearing"](signal_id="my_signal")
        assert "my_signal" in result

    def test_contains_workflow_steps(self, prompts):
        result = prompts["diagnose_bearing"](signal_id="test_signal")
        # Should contain analysis steps
        assert "FFT" in result or "fft" in result or "spectrum" in result.lower()

    def test_with_frequencies(self, prompts):
        result = prompts["diagnose_bearing"](
            signal_id="test_signal",
            bpfo=81.0,
            bpfi=119.0,
            bsf=64.0,
            ftf=6.7,
        )
        assert "81.00" in result
        assert "119.00" in result

    def test_default_machine_group(self, prompts):
        result = prompts["diagnose_bearing"](signal_id="test_signal")
        # Default is machine_group=2
        assert "2" in result or "medium" in result.lower()

    def test_collects_signal_unit_for_iso(self, prompts):
        """STEP 0 must surface signal_unit as a parameter to declare/collect,
        so the ISO 20816-3 verdict is not run on an undeclared unit (U2)."""
        result = prompts["diagnose_bearing"](signal_id="test_signal")
        assert "signal_unit" in result
        # The four accepted units are named for the user to choose from.
        assert "mm/s" in result and "m/s2" in result

    def test_iso_refusal_remedy_does_not_hardcode_g(self, prompts):
        """The ISO-refusal remedy must confirm the ACTUAL unit with the user,
        NOT fabricate signal_unit="g" — following a hardcoded 'g' verbatim
        would integrate a velocity recording as acceleration (wrong ISO zone).
        """
        result = prompts["diagnose_bearing"](signal_id="test_signal")
        assert 'signal_unit="g", overwrite=True' not in result
        assert 'signal_unit="<confirmed-unit>"' in result
        assert "confirm" in result.lower()


class TestDiagnoseGearPrompt:
    """Tests for diagnose_gear prompt."""

    def test_returns_string(self, prompts):
        if "diagnose_gear" not in prompts:
            pytest.skip("diagnose_gear not registered")
        result = prompts["diagnose_gear"](signal_id="test_signal")
        assert isinstance(result, str)
        assert len(result) > 50


class TestQuickDiagnosticPrompt:
    """Tests for quick_diagnostic_report prompt."""

    def test_returns_string(self, prompts):
        if "quick_diagnostic_report" not in prompts:
            pytest.skip("quick_diagnostic_report not registered")
        result = prompts["quick_diagnostic_report"](signal_id="test_signal")
        assert isinstance(result, str)
        assert len(result) > 50
