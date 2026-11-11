"""Pins for the factual plugin corrections of plugin 1.4.0 (audit A, RC-1a/RC-1b).

The owner approved this wording under OD-3; each test keeps one approved rule
on every plugin file it applies to, found from the file's own calls where it
can be, so a new file that makes the same calls is held to the same rule.
Comparisons use the normalization of the guidance tests (backticks dropped,
whitespace collapsed, single quotes made double).
"""

from __future__ import annotations

import re

import pytest

from _semantic_guards import PLUGIN_DIR, extract_calls, flat, plugin_markdown
from test_model_facing_guidance import FALLBACK

#: W4, audit A-F10: a null bearing block is not a clean check.
NULL_BLOCK = (
    "A null `bearing_faults` block is not a clean bearing check: when a bearing "
    "was given, the check did not run, and the result says nothing about the "
    "bearing. When the result has a `bearing_check` block, quote its `status`, "
    "and its `reason` and `remedy` when the status is `not_computed`; only "
    "`assessed` says anything about the bearing."
)
#: W2 and W3, audit A-F2: the ISO default is disclosed.
DEFAULT_GROUP = "say that the zone uses the default group 2 / rigid"
#: W2: the provenance the server records is relayed.
PARAMETER_SOURCES = "Relay `parameter_sources`, which says where each value came from"
#: W1, audit A-F1: placeholders are not the machine's values.
PLACEHOLDERS = "are placeholders, not"
#: W7: the envelope report's speed only where its schema has one.
ENVELOPE_SCHEMA = "the connected server's schema for"
ENVELOPE_NO_CLAIM = "do not say that the report used"
#: W8: whole-recording RMS and ISO zone boundaries.
ISO_NOT_BROADBAND = (
    "It is not the band-limited velocity RMS that the server's ISO 20816-3 "
    "assessment evaluates"
)
#: G3-O1: the server's band, as `iso20816.py` selects it.
ISO_BAND = (
    "(10-1000 Hz, or 2-1000 Hz when the operating speed supplied or declared "
    "for the assessment is below 600 rpm)"
)
#: G3-O2: the refusal of a diagnosis with no speed from any source.
NO_SPEED_REFUSAL = (
    "If no RPM is supplied and neither the signal's measurement nor its "
    "measurement point declares a speed, `diagnose_vibration` refuses the call "
    "and returns remedies. Ask the user for the operating speed or follow the "
    "declared-data remedies."
)
ISO_JUSTIFIED = (
    "A threshold must be justified for the indicator actually trended: its "
    "feature, how it was computed"
)
ISO_UNIT_ALONE = "A mm/s unit alone does not establish that"

ISO_TOOLS = {"assess_severity", "generate_iso_report", "diagnose_vibration"}


def _plugin() -> dict[str, str]:
    return plugin_markdown()


def _calling(names: set[str]) -> list[str]:
    return sorted(
        source
        for source, text in _plugin().items()
        if any(name in names for name, _, _ in extract_calls(text))
    )


def _missing(sources: list[str], phrase: str) -> list[str]:
    return [s for s in sources if flat(phrase) not in flat(_plugin()[s])]


def test_every_bearing_workflow_carries_the_null_block_clause():
    sources = [
        "skills/bearing-diagnosis/SKILL.md",
        "commands/pm-diagnose.md",
        "agents/diagnostic-pipeline.md",
    ]
    assert _missing(sources, NULL_BLOCK) == []


def test_every_iso_call_site_discloses_the_default_group():
    sources = _calling(ISO_TOOLS)
    assert len(sources) >= 7, sources
    assert _missing(sources, DEFAULT_GROUP) == []


def test_every_one_call_diagnosis_relays_parameter_sources():
    sources = _calling({"diagnose_vibration"})
    assert sources, "no plugin file documents diagnose_vibration"
    assert _missing(sources, PARAMETER_SOURCES) == []


def test_every_one_call_diagnosis_names_the_no_speed_refusal():
    sources = _calling({"diagnose_vibration"})
    assert sources, "no plugin file documents diagnose_vibration"
    assert _missing(sources, NO_SPEED_REFUSAL) == []


def test_prognostics_names_the_server_iso_band_twice():
    """G3-O1: both ISO passages of the prognostics skill give the band."""
    text = flat(_plugin()["skills/prognostics/SKILL.md"])
    assert text.count(flat(ISO_BAND)) == 2


def test_every_file_with_placeholders_says_what_they_are():
    sources = sorted(s for s, t in _plugin().items() if "<rpm>" in t)
    assert len(sources) >= 6, sources
    assert _missing(sources, PLACEHOLDERS) == []


def test_every_envelope_report_speed_is_schema_conditional():
    sources = sorted(
        s
        for s, t in _plugin().items()
        if "generate_envelope_report" in t and "rpm=<rpm>" in t
    )
    assert len(sources) >= 4, sources
    assert _missing(sources, ENVELOPE_SCHEMA) == []
    assert _missing(sources, ENVELOPE_NO_CLAIM) == []


@pytest.mark.parametrize("phrase", [ISO_NOT_BROADBAND, ISO_JUSTIFIED, ISO_UNIT_ALONE])
def test_prognostics_keeps_the_iso_threshold_limits(phrase):
    assert _missing(["skills/prognostics/SKILL.md"], phrase) == []


def test_the_fallback_appears_at_most_once_per_file():
    """Audit A-F15: one copy of the claim-less fallback per surface."""
    counts = {s: flat(t).count(flat(FALLBACK)) for s, t in _plugin().items()}
    assert {s: n for s, n in counts.items() if n > 1} == {}
    assert sum(counts.values()) >= 6, counts


def test_every_skill_declares_its_folder_as_name():
    """Audit A-P-7."""
    wrong = {}
    for skill in sorted((PLUGIN_DIR / "skills").iterdir()):
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        front = text.split("---", 2)[1] if text.startswith("---") else ""
        found = re.search(r"(?m)^name:\s*(\S+)\s*$", front)
        if not found or found.group(1) != skill.name:
            wrong[skill.name] = found.group(1) if found else None
    assert wrong == {}
