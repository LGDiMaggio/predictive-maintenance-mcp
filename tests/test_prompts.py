"""Tests for MCP diagnostic prompts."""

import pytest
from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.prompts import register


@pytest.fixture
def mcp():
    server = FastMCP("test-prompts")
    register(server)
    return server


@pytest.fixture
def prompts(mcp):
    """Get registered prompt functions."""
    return {p.name: p.fn for p in mcp._prompt_manager._prompts.values()}


class TestDiagnoseBearingPrompt:
    """Tests for diagnose_bearing prompt."""

    def test_returns_string(self, prompts):
        result = prompts["diagnose_bearing"](signal_file="test.csv")
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_signal_file(self, prompts):
        result = prompts["diagnose_bearing"](signal_file="my_signal.csv")
        assert "my_signal.csv" in result

    def test_contains_workflow_steps(self, prompts):
        result = prompts["diagnose_bearing"](signal_file="test.csv")
        # Should contain analysis steps
        assert "FFT" in result or "fft" in result or "spectrum" in result.lower()

    def test_with_frequencies(self, prompts):
        result = prompts["diagnose_bearing"](
            signal_file="test.csv",
            bpfo=81.0,
            bpfi=119.0,
            bsf=64.0,
            ftf=6.7,
        )
        assert "81.00" in result
        assert "119.00" in result

    def test_default_machine_group(self, prompts):
        result = prompts["diagnose_bearing"](signal_file="test.csv")
        # Default is machine_group=2
        assert "2" in result or "medium" in result.lower()


class TestDiagnoseGearPrompt:
    """Tests for diagnose_gear prompt."""

    def test_returns_string(self, prompts):
        if "diagnose_gear" not in prompts:
            pytest.skip("diagnose_gear not registered")
        result = prompts["diagnose_gear"](signal_file="test.csv")
        assert isinstance(result, str)
        assert len(result) > 50


class TestQuickDiagnosticPrompt:
    """Tests for quick_diagnostic_report prompt."""

    def test_returns_string(self, prompts):
        if "quick_diagnostic_report" not in prompts:
            pytest.skip("quick_diagnostic_report not registered")
        result = prompts["quick_diagnostic_report"](signal_file="test.csv")
        assert isinstance(result, str)
        assert len(result) > 50
