"""Known defects of the plugin text, held as strict expected failures.

Each test collects its violations across every plugin file and asserts there
are none. While the defect is present the test fails with ``AssertionError``
and the strict ``xfail`` marker records it; its reason starts with
``audit-known-defect: <finding id>:`` so the weekly audit report lists it.
When a change removes the defect the test passes, the strict marker turns
that into a failure, and the change must remove the marker: the fix is then
proved by the test, not by the merge.

Every check compares plugin text with a fact: the registered server's
parameters, output enums and output fields, the frozen surface of the last
release, the DOCX renderer's code, the file system, or a paragraph the
existing guidance tests already pin. None judges a diagnostic threshold,
label or recommendation.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from _semantic_guards import (
    AVAILABILITY_RE,
    REPO_ROOT,
    literal_arguments,
    missing_example_files,
    missing_pinned_text,
    plugin_markdown,
    renderer_section_keys,
    require,
    unhandled_values,
    unknown_features,
    unmarked_mentions,
    unread_docx_keys,
)
from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.models import RULEstimationResult
from predictive_maintenance_mcp.signal_processing.features import (
    extract_time_domain_features,
)
from test_diagnostic_invariants import known_defect
from test_model_facing_guidance import FALLBACK

RELEASED_FILE = (
    Path(__file__).resolve().parent / "fixtures" / "released_surface_v0.13.0.json"
)

#: Arguments whose values belong to the user's machine (speed, bearing,
#: frequencies, limits); a literal in an example is executed as the machine's.
MACHINE_CONTEXT = {
    "rpm",
    "bearing_id",
    "frequencies",
    "bearing_freqs",
    "failure_threshold",
}

#: Arguments ``diagnose_vibration`` resolves from a declared measurement and
#: point when they are omitted; an explicit literal overrides the declaration.
POINT_RESOLVED = {"machine_group", "support_type", "rpm", "bearing_id"}

#: Surfaces that call the bearing line test or the envelope report and so need
#: the claim-less paragraph already pinned on the bearing workflows.
FALLBACK_SURFACES = (
    "skills/documentation-search/SKILL.md",
    "skills/report-generation/SKILL.md",
    "commands/pm-report.md",
)

#: The paragraph that tells a model the server predates the bearing claim.
CLAIM_FALLBACK_RE = re.compile(
    AVAILABILITY_RE.pattern + r"|predates the bearing claim", re.IGNORECASE
)


def _fail_on(violations: list[str]) -> None:
    assert not violations, "\n".join(violations)


@functools.lru_cache(maxsize=None)
def plugin() -> dict[str, str]:
    return plugin_markdown()


@functools.lru_cache(maxsize=None)
def tool_surface() -> dict[str, dict]:
    """Registered tools as {name: {"parameters": set, "output_properties": set}}."""
    server = MCPServer("semantic-plugin-contract")
    register_all(server)
    surface = {}
    for tool in server._tool_manager._tools.values():
        schema = getattr(tool, "output_schema", None) or getattr(
            getattr(tool, "fn_metadata", None), "output_schema", None
        )
        properties = set((schema or {}).get("properties", {}))
        for definition in (schema or {}).get("$defs", {}).values():
            properties.update(definition.get("properties", {}))
        surface[tool.name] = {
            "parameters": set(tool.parameters.get("properties", {})),
            "output_properties": properties,
        }
    return surface


@functools.lru_cache(maxsize=None)
def released() -> dict:
    return json.loads(RELEASED_FILE.read_text(encoding="utf-8"))


def main_only_output_fields() -> set[str]:
    """Output fields of released tools that the released server never returns."""
    old = set().union(
        *(set(t["output_properties"]) for t in released()["tools"].values())
    )
    current = set().union(
        *(
            t["output_properties"]
            for name, t in tool_surface().items()
            if name in released()["tools"]
        )
    )
    return current - old


def main_only_keywords() -> dict[str, set[str]]:
    """Parameters of released tools that the released server does not accept."""
    extra = {}
    for name, entry in released()["tools"].items():
        added = tool_surface().get(name, {}).get("parameters", set()) - set(
            entry["parameters"]
        )
        if added:
            extra[name] = added
    return extra


class TestPluginExamples:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F1",
            "plugin call examples pass literal machine values (speed, bearing "
            "designation, characteristic frequencies, limits) that a model "
            "executes as the user's machine instead of the declared context",
        ),
    )
    def test_examples_carry_no_literal_machine_context(self):
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in literal_arguments(text, source, MACHINE_CONTEXT)
            ]
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F2",
            "plugin diagnose_vibration examples pass literal machine_group, "
            "support_type, rpm or bearing_id, which override the values the "
            "server resolves from a declared measurement point",
        ),
    )
    def test_diagnose_vibration_examples_leave_point_values_to_the_server(self):
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in literal_arguments(
                    text, source, POINT_RESOLVED, tools={"diagnose_vibration"}
                )
            ]
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F14",
            "plugin examples load files from the shipped sample folders that "
            "do not exist there",
        ),
    )
    def test_example_files_in_sample_folders_exist(self):
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in missing_example_files(text, source)
            ]
        )


class TestPluginValueDomains:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F8",
            "the prognostics skill offers feature names (peak_to_peak) that "
            "the feature extractor does not produce, so the trend and RUL "
            "tools refuse them",
        ),
    )
    def test_offered_feature_names_exist(self):
        signal = np.random.default_rng(0).standard_normal(1000)
        valid = set(extract_time_domain_features(signal))
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in unknown_features(text, source, valid)
            ]
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "N1",
            "the prognostics skill does not handle every estimate_rul status: "
            "model_threshold_crossed and model_not_rising are missing",
        ),
    )
    def test_prognostics_skill_handles_every_rul_status(self):
        statuses = get_args(RULEstimationResult.model_fields["status"].annotation)
        missing = unhandled_values(plugin()["skills/prognostics/SKILL.md"], statuses)
        assert not missing, f"statuses the skill never names: {missing}"

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F9",
            "plugin generate_diagnostic_report_docx examples send section keys "
            "the DOCX renderer never reads, so those sections are dropped",
        ),
    )
    def test_docx_section_keys_are_read_by_the_renderer(self):
        renderer = (REPO_ROOT / "src" / "report_generator.py").read_text(
            encoding="utf-8"
        )
        keys = renderer_section_keys(renderer)
        require(keys, "no sections.get(...) reads found in the renderer")
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in unread_docx_keys(text, source, keys)
            ]
        )


class TestPluginRelayAndCompatibility:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F3",
            "documentation-search, report-generation and /pm-report call the "
            "bearing line test or the envelope report without the claim-less "
            "paragraph the bearing workflows carry for older servers",
        ),
    )
    def test_fallback_paragraph_on_every_report_surface(self):
        calls = {
            "check_bearing_faults",
            "generate_envelope_report",
            "diagnose_vibration",
        }
        _fail_on(
            [
                v
                for source in FALLBACK_SURFACES
                for v in missing_pinned_text(plugin()[source], source, calls, FALLBACK)
            ]
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F11",
            "plugin text tells the model to read output fields the last "
            "released server does not return (for example measurement), with "
            "no availability sentence in that file",
        ),
    )
    def test_main_only_output_fields_carry_an_availability_sentence(self):
        fields = main_only_output_fields()
        require(fields, "no output field differs from the last release")
        patterns = {f"`{name}`": rf"`{re.escape(name)}`" for name in sorted(fields)}
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in unmarked_mentions(
                    text, source, patterns, CLAIM_FALLBACK_RE, per_section=False
                )
            ]
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "A-F12",
            "plugin examples pass keywords the last released server does not "
            "accept (generate_envelope_report rpm), with no availability "
            "sentence in that section; the released server drops them silently",
        ),
    )
    def test_main_only_keywords_carry_an_availability_sentence(self):
        patterns = {
            f"{tool}({keyword}=)": rf"\b{re.escape(tool)}\([^)]*\b{re.escape(keyword)}\s*="
            for tool, keywords in sorted(main_only_keywords().items())
            for keyword in sorted(keywords)
        }
        require(patterns, "no keyword differs from the last release")
        _fail_on(
            [
                v
                for source, text in plugin().items()
                for v in unmarked_mentions(text, source, patterns)
            ]
        )
