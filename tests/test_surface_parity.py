"""U9 surface parity: every v0.8.x endpoint has a declared destination.

The v0.8.x surface (46 tools + 4 resources + 4 prompts, frozen in git at
commit 1609ce8: ``git show 1609ce8:tests/fixtures/tool_inventory.json``)
was consolidated to the target surface of 33 tools + 0 resources +
3 prompts. This test is the migration map made executable:

- EVERY old endpoint name appears in OLD_TO_NEW exactly once;
- its destination is a registered endpoint of the NEW surface (kept name,
  renamed name, or the absorbing tool), or ``None`` with a motivation
  string for outright drops;
- the new surface counts exactly 33 migrated tools plus the declared
  post-consolidation additions in POST_U9_ADDITIONS, 0 resources, 3 prompts;
- naming hygiene: no ``_tool`` suffixes, and the retired duplicate-concept
  parameter names (shaft_speed_rpm, rotation_freq, ...) are absent from
  every tool signature and prompt argument list.

This test is STRUCTURAL (endpoint existence + naming), not behavioral —
behavior preservation of the merges is pinned by tests/test_golden_merges.py.
"""

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import register_all


# ---------------------------------------------------------------------------
# The migration table. Keys: every endpoint name that existed in v0.8.x
# (46 tools, 4 resources, 4 prompts). Values: (destination, note) where
# destination is the NEW endpoint the functionality lives in, or None for
# an outright drop (note then carries the motivation).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tools added deliberately AFTER the v0.9.0 consolidation. The migration table
# below describes where the old surface went; it cannot describe a capability
# that did not exist then. Adding a tool is permitted in a minor version (see
# the backward-compatibility invariant in CLAUDE.md) — declaring it here is
# what keeps "no orphan tools" meaningful, since anything registered but
# neither migrated nor declared appeared by accident.
# ---------------------------------------------------------------------------

POST_U9_ADDITIONS: dict[str, str] = {
    "generate_diagnostic_report": (
        "v0.10 — integrated, server-authored diagnostic report. New concept: "
        "generate_diagnostic_report_docx takes its content sections from the "
        "caller, this one authors them."
    ),
}


OLD_TO_NEW: dict[str, tuple[str | None, str]] = {
    # ----- lifecycle -------------------------------------------------------
    "load_signal": ("load_signal", "kept — canonical entry point (+ batch, overwrite)"),
    "list_signals": ("list_signals", "kept — disk listing is scope='disk' (default)"),
    "list_stored_signals": ("list_signals", "merged — scope='memory'"),
    "get_signal_info": ("get_signal_info", "kept — now exposes full source_metadata"),
    "generate_test_signal": (
        "generate_test_signal",
        "kept — now writes metadata, auto-registers, returns StoredSignalInfo",
    ),
    "clear_signal": ("clear_signals", "merged — clear_signals(signal_id=...)"),
    "clear_all_signals": ("clear_signals", "merged — clear_signals() clears all"),
    # ----- analysis --------------------------------------------------------
    "analyze_fft": ("analyze_fft", "kept"),
    "analyze_envelope": ("analyze_envelope", "kept — THE unified envelope tool"),
    "compute_envelope_spectrum_tool": ("analyze_envelope", "merged (U9a)"),
    "analyze_statistics": ("analyze_statistics", "kept"),
    "extract_features_from_signal": (
        "extract_features_from_signal",
        "kept — no longer writes CSV into data/signals/",
    ),
    "compute_power_spectral_density": ("compute_power_spectral_density", "kept"),
    "compute_spectrogram_stft": ("compute_spectrogram_stft", "kept"),
    # ----- diagnostics -----------------------------------------------------
    "evaluate_iso_20816": ("assess_severity", "merged (U9a) — signal route"),
    "assess_vibration_severity": ("assess_severity", "merged (U9a) — signal route"),
    "check_vibration_alert": ("assess_severity", "merged (U9a) — rms_velocity_mm_s route"),
    "check_custom_vibration_alert": (
        "assess_severity",
        "merged (U9a) — thresholds={'warning','alarm','danger'} parameter",
    ),
    "check_bearing_fault_peak_tool": (
        "check_bearing_faults",
        "merged (U9a) — frequencies={label: hz} route with one entry",
    ),
    "check_bearing_faults_direct": ("check_bearing_faults", "merged (U9a) — bearing_id route"),
    "lookup_bearing_and_compute_tool": (
        "check_bearing_faults",
        "merged (U9a) — bearing_id route resolves prefixed designations",
    ),
    "diagnose_vibration_tool": ("diagnose_vibration", "renamed — _tool suffix dropped"),
    "calculate_bearing_characteristic_frequencies": (
        "calculate_bearing_characteristic_frequencies",
        "kept — shaft_speed_rpm parameter renamed to rpm",
    ),
    "search_bearing_catalog": (
        "search_bearing_catalog",
        "kept — bearing_designation parameter renamed to bearing_id",
    ),
    "train_anomaly_model": ("train_anomaly_model", "kept — result echoes model_name"),
    "predict_anomalies": (
        "predict_anomalies",
        "kept — bounded output (counts/percentiles/worst segments)",
    ),
    # ----- documentation ---------------------------------------------------
    "list_machine_manuals": ("list_machine_manuals", "kept"),
    "read_manual_excerpt": (
        "read_manual_excerpt",
        "kept — manual_filename parameter renamed to file_name",
    ),
    "extract_manual_specs": (
        "extract_manual_specs",
        "kept — manual_filename parameter renamed to file_name",
    ),
    "search_documentation": ("search_documentation", "kept"),
    # ----- reports ---------------------------------------------------------
    "plot_signal": ("plot_signal", "kept — timestamped output filename"),
    "plot_spectrum": (
        "generate_fft_report",
        "merged — the FFT report embeds the interactive dB spectrum with peak labels",
    ),
    "plot_envelope": (
        "generate_envelope_report",
        "merged — the envelope report embeds filtered signal + envelope spectrum",
    ),
    "plot_iso_20816_chart": (
        "generate_iso_report",
        "merged — the ISO report embeds the color-coded A-D zone bar chart",
    ),
    "generate_fft_report": ("generate_fft_report", "kept — rotation_freq (Hz) renamed to rpm"),
    "generate_envelope_report": ("generate_envelope_report", "kept"),
    "generate_iso_report": ("generate_iso_report", "kept — delegates to assess_severity"),
    "generate_diagnostic_report_docx": ("generate_diagnostic_report_docx", "kept"),
    "generate_pca_visualization_report": ("generate_pca_visualization_report", "kept"),
    "generate_feature_comparison_report": ("generate_feature_comparison_report", "kept"),
    "list_html_reports": ("list_html_reports", "kept — absorbs get_report_info"),
    "get_report_info": (
        "list_html_reports",
        "merged — list_html_reports(file_name=...) via safe_resolve read path",
    ),
    # ----- prognostics -----------------------------------------------------
    "analyze_signal_trend": ("analyze_signal_trend", "kept — THE unified screening tool"),
    "detect_signal_degradation_onset": ("analyze_signal_trend", "merged (U9a) — onset block"),
    "estimate_rul": ("estimate_rul", "kept — multi-measurement contract (U4)"),
    # ----- decision --------------------------------------------------------
    "generate_maintenance_recommendations": (
        "generate_maintenance_recommendations",
        "kept — fault_types typed as canonical Literal list, raises on unknowns",
    ),
    # ----- resources (dropped as resources; info reachable via tools) ------
    "signal://list": ("list_signals", "resource dropped — list_signals(scope='disk')"),
    "signal://read/{filename}": (
        "get_signal_info",
        "resource dropped — get_signal_info exposes metadata incl. source_metadata",
    ),
    "manual://list": ("list_machine_manuals", "resource dropped — duplicate of the tool"),
    "manual://read/{filename}": (
        "read_manual_excerpt",
        "resource dropped — duplicate of the tool",
    ),
    # ----- prompts ---------------------------------------------------------
    "diagnose_bearing": ("diagnose_bearing", "kept — call templates fixed to valid kwargs"),
    "diagnose_gear": ("diagnose_gear", "kept — call templates fixed to valid kwargs"),
    "quick_diagnostic_report": ("quick_diagnostic_report", "kept"),
    "generate_iso_diagnostic_report": (
        None,
        "dropped — ASCII-art boilerplate contradicting the <=300-word output "
        "policy; the ISO workflow is covered by assess_severity + "
        "generate_iso_report and STEP 1 of the diagnose_bearing prompt",
    ),
}

#: Old endpoint names by kind (frozen v0.8.x inventory, commit 1609ce8).
OLD_RESOURCES = {
    "signal://list",
    "signal://read/{filename}",
    "manual://list",
    "manual://read/{filename}",
}
OLD_PROMPTS = {
    "diagnose_bearing",
    "diagnose_gear",
    "quick_diagnostic_report",
    "generate_iso_diagnostic_report",
}
OLD_TOOLS = set(OLD_TO_NEW) - OLD_RESOURCES - OLD_PROMPTS

#: Retired parameter names: one name per concept on the final surface
#: (signal_id, rpm, sampling_rate, file_name, bearing_id, filepath).
BANNED_PARAM_NAMES = {
    "shaft_speed_rpm",  # -> rpm
    "rotation_freq",  # Hz-valued twin -> rpm (unit-facing rename)
    "rotation_speed_rpm",  # -> rpm
    "operating_speed_rpm",  # -> rpm
    "manual_filename",  # -> file_name
    "bearing_designation",  # -> bearing_id
    "signal_file",  # -> signal_id
    "filename",  # resource-era handle -> file_name / signal_id
    "signal_path",  # never a handle again: signal_id is THE handle
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registered():
    mcp = MCPServer("surface-parity")
    register_all(mcp)
    tools = {t.name: t for t in mcp._tool_manager._tools.values()}
    resources = sorted(
        [str(uri) for uri in mcp._resource_manager._resources.keys()]
        + [str(t.uri_template) for t in mcp._resource_manager._templates.values()]
    )
    prompts = {
        p.name: sorted(a.name for a in (p.arguments or []))
        for p in mcp._prompt_manager._prompts.values()
    }
    return tools, resources, prompts


# ---------------------------------------------------------------------------
# Table completeness
# ---------------------------------------------------------------------------


class TestMappingCompleteness:
    def test_covers_full_old_inventory(self):
        """46 tools + 4 resources + 4 prompts — nothing forgotten, nothing extra."""
        assert len(OLD_TOOLS) == 46
        assert len(OLD_RESOURCES) == 4
        assert len(OLD_PROMPTS) == 4
        assert len(OLD_TO_NEW) == 54

    def test_every_old_name_has_declared_destination(self):
        for old, entry in OLD_TO_NEW.items():
            assert isinstance(entry, tuple) and len(entry) == 2, old
            destination, note = entry
            assert note and isinstance(note, str), f"{old}: motivation missing"
            if destination is None:
                assert "dropped" in note.lower(), (
                    f"{old}: a None destination requires a drop motivation"
                )


# ---------------------------------------------------------------------------
# Destinations exist on the new surface
# ---------------------------------------------------------------------------


class TestDestinationsExist:
    def test_tool_and_resource_destinations_are_registered_tools(self, registered):
        tools, _, _ = registered
        for old in sorted(OLD_TOOLS | OLD_RESOURCES):
            destination, _ = OLD_TO_NEW[old]
            assert destination is not None, (
                f"{old}: tools/resources must map into the tool surface"
            )
            assert destination in tools, (
                f"{old} -> {destination}: destination not registered"
            )

    def test_prompt_destinations_are_registered_prompts(self, registered):
        _, _, prompts = registered
        for old in sorted(OLD_PROMPTS):
            destination, _ = OLD_TO_NEW[old]
            if destination is None:
                continue  # motivated drop
            assert destination in prompts, (
                f"{old} -> {destination}: destination prompt not registered"
            )

    def test_every_current_tool_is_migrated_or_declared(self, registered):
        """No orphan tools.

        Every registered tool either descends from the v0.8.x surface via
        OLD_TO_NEW, or is a deliberate post-consolidation addition declared
        in POST_U9_ADDITIONS. A tool that is neither appeared by accident.
        """
        tools, _, _ = registered
        destinations = {
            dest
            for old, (dest, _) in OLD_TO_NEW.items()
            if dest is not None and old not in OLD_PROMPTS
        }
        assert destinations | set(POST_U9_ADDITIONS) == set(tools)

    def test_declared_additions_are_actually_registered(self, registered):
        """A stale entry in the additions register is as bad as a silent one."""
        tools, _, _ = registered
        missing = [name for name in POST_U9_ADDITIONS if name not in tools]
        assert missing == [], f"declared but not registered: {missing}"

    def test_additions_do_not_shadow_a_migrated_tool(self, registered):
        """An addition must be a new concept, not a second name for an old one."""
        destinations = {
            dest
            for old, (dest, _) in OLD_TO_NEW.items()
            if dest is not None and old not in OLD_PROMPTS
        }
        assert set(POST_U9_ADDITIONS).isdisjoint(destinations)


# ---------------------------------------------------------------------------
# Final surface counts + naming hygiene
# ---------------------------------------------------------------------------


class TestFinalSurface:
    def test_counts_are_the_migrated_surface_plus_declared_additions(
        self, registered
    ):
        tools, resources, prompts = registered
        assert len(tools) == 33 + len(POST_U9_ADDITIONS)
        assert resources == []
        assert len(prompts) == 3

    def test_no_tool_suffixes(self, registered):
        tools, _, prompts = registered
        offenders = [n for n in list(tools) + list(prompts) if n.endswith("_tool")]
        assert offenders == []

    def test_no_duplicate_concept_parameters(self, registered):
        """One name per concept: retired parameter spellings are gone from
        every tool input schema and every prompt argument list."""
        tools, _, prompts = registered
        offenders: list[str] = []
        for name, tool in tools.items():
            params = set(tool.parameters.get("properties", {}))
            for banned in BANNED_PARAM_NAMES & params:
                offenders.append(f"tool {name}({banned}=...)")
        for name, args in prompts.items():
            for banned in BANNED_PARAM_NAMES & set(args):
                offenders.append(f"prompt {name}({banned}=...)")
        assert offenders == [], offenders

    def test_absorbed_and_dropped_names_not_registered(self, registered):
        """Clean cut: no old name survives unless it is its own destination."""
        tools, resources, prompts = registered
        current = set(tools) | set(resources) | set(prompts)
        for old, (destination, _) in OLD_TO_NEW.items():
            if old == destination:
                continue  # kept name
            assert old not in current, f"{old} should no longer be registered"
