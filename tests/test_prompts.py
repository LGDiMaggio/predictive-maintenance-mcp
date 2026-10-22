"""Tests for MCP diagnostic prompts."""

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.prompts import register
from test_documented_calls import PLUGIN_FILES, REPO_ROOT, extract_calls


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

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"rpm": 1797.0, "bpfo": 107.4, "bpfi": 162.2, "bsf": 70.6, "ftf": 14.3}],
        ids=["minimal", "full"],
    )
    def test_respects_the_server_claim_and_asks_for_no_replacement(
        self, prompts, kwargs
    ):
        """T1 PR C: no "STRONGLY indicated" reading of a 5 % peak, no
        "Confirmed fault ... plan replacement" row, and no inferred fault type
        chained into generate_maintenance_recommendations."""
        result = prompts["diagnose_bearing"](signal_id="test_signal", **kwargs)
        flat = " ".join(result.split())

        assert "STRONGLY" not in result
        assert "plan replacement" not in result
        assert "Confirmed fault" not in result
        assert "fault_types=[" not in result
        assert "claim.level" in flat
        assert "single_spectrum_evidence" in flat
        assert "never request bearing replacement" in flat
        assert 'generate_maintenance_recommendations(severity_zone="A")' in result
        assert (
            "Do NOT pass fault_types taken from an envelope indication of "
            "check_bearing_faults or diagnose_vibration"
        ) in flat

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"rpm": 1797.0, "bpfo": 107.4, "bpfi": 162.2, "bsf": 70.6, "ftf": 14.3}],
        ids=["minimal", "full"],
    )
    def test_withheld_and_qualified_claims_follow_the_claim_not_next_step(
        self, prompts, kwargs
    ):
        """T1 PR C review: a withheld or qualified claim is relayed through
        overall_assessment and the actions the server renders for it; the
        evidence layer's outcome_reason and next_step can still name one
        element, so they are not relayed to name a type or an element."""
        result = prompts["diagnose_bearing"](signal_id="test_signal", **kwargs)
        flat = " ".join(result.split())

        assert "give the claim's reason and next step" not in flat
        assert (
            "qualified → quote overall_assessment verbatim: it names every "
            "reading the claim cannot exclude; single out none"
        ) in flat
        assert (
            "withheld → name no fault type; quote overall_assessment verbatim: "
            "it says why no fault type is assessed"
        ) in flat
        assert (
            "outcome_reason and next_step describe the evidence outcome before "
            "the claim: when claim.level is not single_spectrum_evidence, do not "
            "relay them to name a fault type or a single element."
        ) in flat
        assert (
            "Qualified or withheld claim (any zone) → Only the claim's actions as "
            'the server renders them (the "Bearing" entries of '
            "diagnose_vibration's recommendations, quoted verbatim)"
        ) in flat

    def test_prompt_description_drops_the_strongly_indicated_rule(self, mcp):
        prompt = next(
            p
            for p in mcp._prompt_manager._prompts.values()
            if p.name == "diagnose_bearing"
        )
        assert "STRONGLY" not in (prompt.description or "")


class TestPluginRecommendationCalls:
    """The plugin instructions (agents, skills, commands) follow the same rule
    as the diagnose_bearing prompt: generate_maintenance_recommendations takes
    the zone only, never a fault type an agent read off a bearing check."""

    def test_no_plugin_call_passes_fault_types(self):
        calls = [
            (str(path.relative_to(REPO_ROOT)).replace("\\", "/"), line, args)
            for path in PLUGIN_FILES
            for name, args, line in extract_calls(path.read_text(encoding="utf-8"))
            if name == "generate_maintenance_recommendations"
        ]

        # Anti-rot: the scan must see the documented calls it guards.
        assert calls, "no generate_maintenance_recommendations call found in plugin/"
        offending = [
            f"{source}:{line}: generate_maintenance_recommendations({args})"
            for source, line, args in calls
            if "fault_types" in args
        ]
        assert offending == [], "\n".join(offending)


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
