"""Statistical-indicator thresholds in plugin text come from the server, or not at all.

A threshold is an indicator name (kurtosis, crest factor or CF, skewness)
followed on the same line by a comparator and a number (``Kurtosis > 3``) or by
a number range (``CF 4-6``). The bearing and screening workflows, the commands
and the pipeline agent relay the statistics as numbers and state no threshold.
Any other plugin surface may state only the values a rendered server prompt
states for the same indicator. The prompts are harvested with the same rule.
"""

from __future__ import annotations

import functools
import re

import pytest

from _semantic_guards import plugin_markdown
from test_documented_calls import rendered_prompt_texts

#: Surfaces that relay the statistics as numbers only.
NO_THRESHOLD_SURFACES = (
    "agents/diagnostic-pipeline.md",
    "commands/pm-diagnose.md",
    "commands/pm-report.md",
    "commands/pm-screen.md",
    "skills/bearing-diagnosis/SKILL.md",
    "skills/quick-screening/SKILL.md",
)

_INDICATOR_RE = re.compile(r"\b(kurtosis|kurt|crest[ _]factor|cf|skewness)\b", re.I)
PLUGIN_TEXT = plugin_markdown()
_CANONICAL = {"kurt": "kurtosis", "cf": "crest_factor"}
_NUMBER = r"-?\d+(?:\.\d+)?"
_COMPARED_RE = re.compile(
    rf"(?:[<>≤≥]=?|\b(?:above|below|over|under|exceeds?|exceeding|greater than"
    rf"|less than|more than|at least|at most))\s*({_NUMBER})",
    re.I,
)
_RANGE_RE = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*[-–]\s*({_NUMBER})(?![\w.])"
    rf"|\bbetween\s+({_NUMBER})\s+and\s+({_NUMBER})",
    re.I,
)


def _canonical(name: str) -> str:
    name = name.lower().replace(" ", "_")
    return _CANONICAL.get(name, name)


def _values(stretch: str) -> list[float]:
    values = [float(v) for v in _COMPARED_RE.findall(stretch)]
    for groups in _RANGE_RE.findall(stretch):
        values += [float(v) for v in groups if v]
    return values


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def thresholds(text: str) -> list[tuple[int, str, float]]:
    """(line number, indicator, value) for every threshold in *text*.

    Each indicator mention owns the stretch of its line up to the next
    mention, so "CF 4-6 or Kurtosis 3-6" gives CF 4 and 6, kurtosis 3 and 6.
    In a table whose header cell names an indicator, each cell under that
    header is read the same way ("| B | 1-3 |" under "| Zone | Kurtosis |").
    """
    found = []
    header: list[str] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("|"):
            cells = _cells(line)
            if header is None:
                header = cells
            elif not set("".join(cells)) <= set("-: "):
                for title, cell in zip(header, cells):
                    m = _INDICATOR_RE.fullmatch(title.strip("*` "))
                    if m and not _INDICATOR_RE.search(cell):
                        found += [
                            (number, _canonical(m.group(1)), v) for v in _values(cell)
                        ]
        else:
            header = None
        mentions = list(_INDICATOR_RE.finditer(line))
        for i, m in enumerate(mentions):
            end = mentions[i + 1].start() if i + 1 < len(mentions) else len(line)
            name = _canonical(m.group(1))
            found += [(number, name, v) for v in _values(line[m.end() : end])]
    return found


@functools.lru_cache(maxsize=None)
def server_thresholds() -> frozenset[tuple[str, float]]:
    """(indicator, value) pairs the rendered server prompts state."""
    return frozenset(
        (name, value)
        for text in rendered_prompt_texts().values()
        for _, name, value in thresholds(text)
    )


def threshold_violations(source: str, text: str) -> list[str]:
    """Thresholds in a numbers-only surface, or elsewhere stated by no prompt."""
    if source in NO_THRESHOLD_SURFACES:
        return [
            f"{source}:{n}: {name} threshold {value:g} in a numbers-only surface"
            for n, name, value in thresholds(text)
        ]
    return [
        f"{source}:{n}: {name} threshold {value:g} is in no server prompt"
        for n, name, value in thresholds(text)
        if (name, value) not in server_thresholds()
    ]


def test_the_harvest_reads_the_prompts_bands():
    """A satisfiability control: the server prompts do state thresholds, so an
    empty harvest would make the second rule pass vacuously."""
    assert {("kurtosis", 3.0), ("crest_factor", 4.0)} <= server_thresholds()


@pytest.mark.parametrize("source", sorted(PLUGIN_TEXT))
def test_every_threshold_is_absent_or_server_sourced(source):
    assert threshold_violations(source, PLUGIN_TEXT[source]) == []


@pytest.mark.parametrize(
    "source, insert, expected",
    [
        (
            "skills/bearing-diagnosis/SKILL.md",
            "- CF 4-6 or Kurtosis 3-6 → Moderate impulsiveness\n",
            [
                "crest_factor threshold 4",
                "crest_factor threshold 6",
                "kurtosis threshold 3",
                "kurtosis threshold 6",
            ],
        ),
        (
            "skills/bearing-diagnosis/SKILL.md",
            "If Kurtosis < 0 AND Crest Factor < 3, stop.\n",
            ["kurtosis threshold 0", "crest_factor threshold 3"],
        ),
        (
            "skills/quick-screening/SKILL.md",
            "- Kurtosis > 3 → significant\n",
            ["kurtosis threshold 3"],
        ),
        (
            "skills/gear-diagnosis/SKILL.md",
            "- **Kurtosis > 4**: impulsive content\n",
            ["kurtosis threshold 4 is in no server prompt"],
        ),
        (
            "skills/quick-screening/SKILL.md",
            "A kurtosis above 3 means significant impulsiveness.\n",
            ["kurtosis threshold 3"],
        ),
        (
            "skills/quick-screening/SKILL.md",
            "Crest factor between 4 and 6 is moderate.\n",
            ["crest_factor threshold 4", "crest_factor threshold 6"],
        ),
        (
            "skills/quick-screening/SKILL.md",
            "| ISO Zone | Kurtosis | Class |\n|---|---|---|\n| B | 1-3 | Watch |\n",
            ["kurtosis threshold 1", "kurtosis threshold 3"],
        ),
    ],
    ids=[
        "bearing range band",
        "bearing gate",
        "screening band",
        "gear value changed",
        "worded comparator",
        "worded range",
        "indicator column of a table",
    ],
)
def test_an_inserted_threshold_is_reported_on_a_copy(source, insert, expected):
    text = PLUGIN_TEXT[source] + "\n" + insert
    reported = threshold_violations(source, text)
    assert len(reported) == len(expected)
    for fragment, line in zip(expected, reported):
        assert fragment in line, (fragment, line)
