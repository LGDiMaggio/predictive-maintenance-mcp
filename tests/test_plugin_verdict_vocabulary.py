"""The plugin authors no health verdict, action or interval of its own.

The verdict-vocabulary guard (``_semantic_guards.verdict_vocabulary``) runs on
every plugin markdown file. Actions and intervals are allowed only when a
server text contains them: the rendered prompts, the registered tool
descriptions and the zone actions of the recommendations table. One surface
is exempt, with its reason; the stale-entry test fails once the exemption
excuses nothing.

A second guard holds that no plugin text claims anomaly detection is verified:
the server-written report relays an anomaly section, which is not evidence
that the detection works end to end.
"""

from __future__ import annotations

import functools
import re

import pytest

from _semantic_guards import plugin_markdown, verdict_vocabulary
from predictive_maintenance_mcp.decision_support.recommendations import (
    generate_recommendations,
)
from test_documented_calls import rendered_prompt_texts
from test_model_facing_guidance import _registered

#: Surfaces not held to the guard yet, with the reason.
EXEMPT = {
    "skills/anomaly-detection/SKILL.md": (
        "Its anomaly-ratio table assigns health labels and actions; the "
        "anomaly-label projection is a pending decision."
    ),
}

FIXTURE_DECISION_LOGIC = (
    "Decision logic:\n"
    "- ISO Zone A + Kurtosis < 1 -> Healthy\n"
    "- ISO Zone B or Kurtosis 1-3 -> Monitor\n"
    "| ISO Zone | Kurtosis | Verdict | Recommendation |\n"
    "|----------|----------|---------|----------------|\n"
    "| B | 1-3 | Monitor | Elevated indicators. Schedule follow-up in 2-4 weeks. |\n"
)

_ANOMALY_VERIFIED_RE = re.compile(
    r"anomal[^.\n]{0,60}\b(?:verified|validated|proven)\b"
    r"|\b(?:verified|validated|proven)\b[^.\n]{0,60}anomal",
    re.I,
)


@functools.lru_cache(maxsize=None)
def server_phrases() -> tuple[str, ...]:
    """Texts the server authors that a plugin surface may project."""
    tools, _ = _registered()
    zone_actions = [
        f"{rec['action']}. {rec['description']}"
        for zone in "ABCD"
        for rec in generate_recommendations(zone, None)
    ]
    return (*rendered_prompt_texts().values(), *tools.values(), *zone_actions)


def anomaly_verified_claims(source: str, text: str) -> list[str]:
    """Lines of *text* that call anomaly detection verified, validated or proven."""
    return [
        f"{source}:{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), start=1)
        if _ANOMALY_VERIFIED_RE.search(line)
    ]


PLUGIN_TEXT = plugin_markdown()
SURFACES = sorted(PLUGIN_TEXT)


@pytest.mark.parametrize("source", [s for s in SURFACES if s not in EXEMPT])
def test_the_plugin_authors_no_verdict(source):
    text = PLUGIN_TEXT[source]
    assert verdict_vocabulary(text, source, server_phrases=server_phrases()) == []


@pytest.mark.parametrize("source", sorted(EXEMPT))
def test_every_exemption_still_excuses_something(source):
    text = PLUGIN_TEXT[source]
    assert verdict_vocabulary(
        text, source, server_phrases=server_phrases()
    ), f"{source} no longer authors a verdict: remove its exemption"


def test_the_old_screening_logic_is_reported_on_a_copy():
    """Re-inserting the retired screening table into a copy of the screening
    skill names each verdict, the action and the interval."""
    source = "skills/quick-screening/SKILL.md"
    text = PLUGIN_TEXT[source] + "\n" + FIXTURE_DECISION_LOGIC
    roles = {
        v.split(": ")[1]
        for v in verdict_vocabulary(text, source, server_phrases=server_phrases())
    }
    assert roles == {
        "arrow-verdict",
        "table-verdict",
        "plugin-action",
        "plugin-interval",
    }


@pytest.mark.parametrize("source", SURFACES)
def test_no_plugin_text_claims_anomaly_detection_is_verified(source):
    assert anomaly_verified_claims(source, PLUGIN_TEXT[source]) == []


def test_a_verified_anomaly_claim_is_reported_on_a_copy():
    source = "skills/report-generation/SKILL.md"
    text = PLUGIN_TEXT[source] + "\nAnomaly detection is verified end to end.\n"
    assert anomaly_verified_claims(source, text) == [
        f"{source}:{len(text.splitlines())}: Anomaly detection is verified end to end."
    ]


#: A single follow-up interval ("in 2 weeks"); the shared guard covers ranges.
_SINGLE_INTERVAL_RE = re.compile(
    r"\b(?:in|within|after|every)\s+\d+\s*(?:hours?|days?|weeks?|months?)\b", re.I
)


def single_interval_violations(source: str, text: str) -> list[str]:
    """Follow-up intervals the text states that no server text contains."""
    phrases = [p.lower() for p in server_phrases()]
    return [
        f"{source}:{number}: plugin-interval: {m.group(0)}"
        for number, line in enumerate(text.splitlines(), start=1)
        for m in _SINGLE_INTERVAL_RE.finditer(line)
        if not any(m.group(0).lower() in p for p in phrases)
    ]


@pytest.mark.parametrize("source", [s for s in SURFACES if s not in EXEMPT])
def test_the_plugin_states_no_interval_of_its_own(source):
    assert single_interval_violations(source, PLUGIN_TEXT[source]) == []


def test_a_single_interval_is_reported_on_a_copy():
    source = "skills/quick-screening/SKILL.md"
    text = (
        PLUGIN_TEXT[source]
        + "\nIf the zone is C, schedule a follow-up measurement in 3 weeks.\n"
    )
    assert single_interval_violations(source, text) == [
        f"{source}:{len(text.splitlines())}: plugin-interval: in 3 weeks"
    ]
