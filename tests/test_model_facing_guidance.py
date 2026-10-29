"""Model-facing bearing guidance stays bound to the runtime vocabulary.

The server instructions, the diagnose_bearing prompt, four tool descriptions
and the plugin's bearing-diagnosis skill, /pm-diagnose command and
diagnostic-pipeline agent tell a model how to relay a bearing or ledger
result. A sentence there that drifts from the server (an outcome name the
check never returns, a claim level it never produces, a "confirmed" the
claim does not support, a lost "not") changes what the model tells the
engineer, and nothing at runtime notices.

Every guard is a pure ``(surface, text) -> violations`` function, run on the
real surfaces and, in ``test_guards_fail_on_mutations``, on mutated copies of
them, so each one has a red path:

- outcome names, claim levels and ledger bases match the runtime Literals
  and ``EVIDENCE_BASES``, both ways;
- retired phrases of the old evidence model stay out;
- "confirm" and "healthy" appear only in listed negated or action forms;
- required sentences stay on their surfaces, and those that echo a server
  sentence are substrings of it;
- the bearing rules end inside the window one Claude Code build was observed
  to show;
- no bearing workflow calls analyze_envelope;
- the plugin relays the claim with the prompt's sentences, the zone sentence
  is the server's, and BSF is tested at twice the spin frequency.

Text is compared after ``flat``: backticks removed, whitespace collapsed,
single quotes made double, so a rewrap or a change of quoting is not drift.
"""

import functools
import re
from typing import Callable, NamedTuple, get_args

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.asset_ledger import assessment
from predictive_maintenance_mcp.diagnostics.bearing_claims import CLAIM_SENTENCES
from predictive_maintenance_mcp.diagnostics.bearing_evidence import Hypothesis
from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.models import (
    BearingCheckOutcomeName,
    BearingClaim,
    BearingClaimLevel,
    BearingFaultCheckResult,
    DiagnosisResult,
)
from test_documented_calls import REPO_ROOT, extract_calls, rendered_prompt_texts

OUTCOMES = frozenset(get_args(BearingCheckOutcomeName))
LEVELS = frozenset(get_args(BearingClaimLevel))

#: Observed truncation of the server instructions in one Claude Code build
#: (an empirical observation, not an MCP contract); recheck when the client
#: changes.
INSTRUCTION_WINDOW = 4096

# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

INSTRUCTIONS = "instructions:rules"
INSTRUCTIONS_FULL = "instructions:full"
PROMPT_MIN = "prompt:diagnose_bearing[minimal]"
PROMPT_FULL = "prompt:diagnose_bearing[full]"
PROMPT_DESC = "prompt:diagnose_bearing:description"
CHECK = "tool:check_bearing_faults"
DIAGNOSIS = "tool:diagnose_vibration"
RECOMMENDATIONS = "tool:generate_maintenance_recommendations"
LEDGER = "tool:assess_asset_change"
SKILL = "plugin:skills/bearing-diagnosis/SKILL.md"
COMMAND = "plugin:commands/pm-diagnose.md"
AGENT = "plugin:agents/diagnostic-pipeline.md"

PROMPTS = (PROMPT_MIN, PROMPT_FULL)
PLUGIN = (SKILL, COMMAND, AGENT)
#: The workflows that run the bearing check.
WORKFLOWS = PROMPTS + PLUGIN
SCANNED = (
    INSTRUCTIONS,
    PROMPT_MIN,
    PROMPT_FULL,
    PROMPT_DESC,
    CHECK,
    DIAGNOSIS,
    RECOMMENDATIONS,
    LEDGER,
    SKILL,
    COMMAND,
    AGENT,
)
#: Surfaces that name the check's outcome, so they must name all four.
NAMES_OUTCOMES = (
    INSTRUCTIONS,
    PROMPT_MIN,
    PROMPT_FULL,
    PROMPT_DESC,
    CHECK,
    SKILL,
    COMMAND,
    AGENT,
)
#: Surfaces that relay the claim level by level, so they name all three.
RELAYS_CLAIM = (PROMPT_MIN, PROMPT_FULL, CHECK, SKILL)

#: Surfaces that relay bearing results but are not held to these rules yet,
#: with the reason. Remove an entry once its surface is corrected: the
#: stale-entry test insists on it.
EXEMPT: dict[str, str] = {
    "prompt:diagnose_gear[minimal]": (
        "Gear workflow: its 'CONFIRMED' gear-tooth wording is gear text. A "
        "bearing claim relayed there still follows the server instructions."
    ),
    "prompt:diagnose_gear[full]": (
        "Gear workflow, rendered with every argument: same gear wording."
    ),
    "plugin:skills/gear-diagnosis/SKILL.md": (
        "Gear skill: its 'strong evidence' scale grades gear findings."
    ),
    "plugin:skills/quick-screening/SKILL.md": (
        "Screening skill: its 'Healthy' row is a screening class of the ISO "
        "zone and statistics, not a bearing result."
    ),
    "plugin:commands/pm-screen.md": (
        "Screening command: same 'Healthy' screening class."
    ),
}


@functools.lru_cache(maxsize=None)
def _registered() -> tuple[dict[str, str], dict[str, str]]:
    """Tool and prompt descriptions as the registered server serves them."""
    mcp = MCPServer("model-facing-guidance")
    register_all(mcp)
    tools = {t.name: t.description or "" for t in mcp._tool_manager._tools.values()}
    prompts = {
        p.name: p.description or "" for p in mcp._prompt_manager._prompts.values()
    }
    return tools, prompts


def full_instructions() -> str:
    from predictive_maintenance_mcp import server

    return server.mcp.instructions or ""


def instruction_rules(instructions: str) -> str:
    """The instructions from the evidence policy to the unit policy."""
    start = instructions.find("Evidence-based inference policy")
    end = instructions.find("Signal unit policy")
    assert 0 <= start < end, "the evidence or unit policy heading is gone"
    return instructions[start:end]


def _plugin(relative: str) -> str:
    return (REPO_ROOT / "plugin" / relative).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def surfaces() -> dict[str, str]:
    tools, prompts = _registered()
    renders = rendered_prompt_texts()
    instructions = full_instructions()
    texts = {
        INSTRUCTIONS: instruction_rules(instructions),
        INSTRUCTIONS_FULL: instructions,
        PROMPT_MIN: renders["prompts.diagnose_bearing[minimal]"],
        PROMPT_FULL: renders["prompts.diagnose_bearing[full]"],
        PROMPT_DESC: prompts["diagnose_bearing"],
        CHECK: tools["check_bearing_faults"],
        DIAGNOSIS: tools["diagnose_vibration"],
        RECOMMENDATIONS: tools["generate_maintenance_recommendations"],
        LEDGER: tools["assess_asset_change"],
        "prompt:diagnose_gear[minimal]": renders["prompts.diagnose_gear[minimal]"],
        "prompt:diagnose_gear[full]": renders["prompts.diagnose_gear[full]"],
    }
    for key in (SKILL, COMMAND, AGENT, *EXEMPT):
        if key.startswith("plugin:"):
            texts[key] = _plugin(key.removeprefix("plugin:"))
    return texts


def flat(text: str) -> str:
    """Backticks removed, whitespace collapsed, single quotes made double."""
    return " ".join(text.replace("`", "").replace("'", '"').split())


# ---------------------------------------------------------------------------
# Vocabularies: outcome names, claim levels, ledger bases
# ---------------------------------------------------------------------------

_WORD = r"[a-z]+(?:_[a-z]+)*"
_TOKEN = rf'"?{_WORD}"?'
#: A list of words joined by commas, "or" or "and": "a, b or c".
_LIST_RE = re.compile(rf"{_TOKEN}(?:(?:, (?:or |and )?| or | and ){_TOKEN})+")


def _named(vocabulary: frozenset[str], text: str) -> set[str]:
    return {v for v in vocabulary if re.search(rf"(?<!\w){v}(?!\w)", text)}


def _list_words(match: str) -> set[str]:
    return set(re.findall(_WORD, match)) - {"or", "and"}


def outcome_violations(surface: str, text: str) -> list[str]:
    """Outcome names: all four or none, and every list of them exact."""
    t = flat(text)
    violations = []
    named = _named(OUTCOMES, t)
    if surface in NAMES_OUTCOMES and not named:
        violations.append(f"{surface}: names no outcome of the check")
    elif named and named != OUTCOMES:
        violations.append(f"{surface}: names only {sorted(named)} of the outcomes")
    for m in _LIST_RE.finditer(t):
        words = _list_words(m.group(0))
        if len(words & OUTCOMES) >= 2 and words != OUTCOMES:
            violations.append(
                f"{surface}: outcome list {m.group(0)!r} is not "
                f"{sorted(OUTCOMES)} (extra {sorted(words - OUTCOMES)}, "
                f"missing {sorted(OUTCOMES - words)})"
            )
    return violations


_LEVEL_SHAPED_RE = re.compile(rf"(?<![\w.])({_WORD}_(?:claim|evidence))(?!\w)")
_CODE_TOKEN_RE = re.compile(rf"[`'\"]({_WORD})[`'\"]")
_LEVEL_IS_RE = re.compile(rf"claim\.level`?\s+is\s+(?:not\s+)?[`'\"]?({_WORD})")
_LEVEL_LIST_RE = re.compile(r"claim\.level`?\s*\(([^)]*)\)")
_RELAY_BULLET_RE = re.compile(rf"•\s*({_WORD})\s*→")
_LEVEL_TABLE_RE = re.compile(r"^\|\s*`?claim\.level`?\s*\|", re.M)


def _level_table_cells(text: str) -> list[str]:
    """First cells of the rows of a table headed by claim.level."""
    cells = []
    for header in _LEVEL_TABLE_RE.finditer(text):
        rows = text[header.end() :].split("\n")[2:]
        for row in rows:
            if not row.startswith("|"):
                break
            cells.append(row.split("|")[1].strip().strip("`"))
    return cells


def claim_level_violations(surface: str, text: str) -> list[str]:
    """Claim levels, read as code tokens or in a level position, are the Literal's."""
    positional: list[str] = []
    positional += _LEVEL_IS_RE.findall(text)
    for inner in _LEVEL_LIST_RE.findall(text):
        positional += sorted(_list_words(inner))
    positional += _RELAY_BULLET_RE.findall(text)
    positional += _level_table_cells(text)
    coded = [t for t in _CODE_TOKEN_RE.findall(text) if t in LEVELS]
    shaped = _LEVEL_SHAPED_RE.findall(text)
    named = (set(positional) | set(coded) | set(shaped)) & LEVELS
    off_list = (set(positional) | set(shaped)) - LEVELS
    violations = []
    if off_list:
        violations.append(
            f"{surface}: claim levels outside the Literal {sorted(off_list)}"
        )
    if surface in RELAYS_CLAIM and named != LEVELS:
        violations.append(
            f"{surface}: relays the claim but names only {sorted(named)} of "
            f"{sorted(LEVELS)}"
        )
    return violations


_BASES_RE = re.compile(
    rf'((?:"{_WORD}"(?:, or |, | or | and ))*"{_WORD}") ledger evidence'
)


def ledger_basis_violations(surface: str, text: str) -> list[str]:
    """Bases quoted before "ledger evidence" are EVIDENCE_BASES; the ledger names all."""
    t = flat(text)
    bases = frozenset(assessment.EVIDENCE_BASES)
    violations = []
    if surface == INSTRUCTIONS:
        quoted = [w for m in _BASES_RE.findall(t) for w in _list_words(m)]
        if not quoted:
            violations.append(f"{surface}: no basis is quoted before 'ledger evidence'")
        if set(quoted) - bases:
            violations.append(
                f"{surface}: ledger bases {sorted(set(quoted) - bases)} are not "
                f"in EVIDENCE_BASES {sorted(bases)}"
            )
    if surface == LEDGER and _named(bases, t) != bases:
        violations.append(
            f"{surface}: names only {sorted(_named(bases, t))} of {sorted(bases)}"
        )
    return violations


# ---------------------------------------------------------------------------
# Retired phrases; "confirm" and "healthy" only in listed forms
# ---------------------------------------------------------------------------

#: (phrase, case-insensitive) of the evidence model the server no longer has.
RETIRED_PHRASES: tuple[tuple[str, bool], ...] = (
    ("STRONGLY", False),
    ("strong evidence", True),
    ("Replace bearing", False),
    ("plan replacement", False),
    ("count of independent", False),
    ("independent corroborating", False),
    ("multiple independent", False),
    ("multiple corroborating", False),
    ("±5%", False),
    ("Check if ANY peak", False),
    ("Compare envelope peaks", False),
    ("Examine envelope spectrum peaks", False),
    ("Evidence matching", True),
    ("Peak matches", False),
    ("No matches", False),
    ("No fault +", False),
    ("No clear bearing fault signatures", False),
    ("Possible [fault type]", False),
    ("health status", False),
    ("Normal / Elevated", False),
    ("consistent with bearing damage", False),
    (", fault identification", False),
    ("definitive fault identification", False),
    ("Frequency matching tolerance", False),
    ("No tool output confirms a fault", False),
)


def retired_phrase_violations(surface: str, text: str) -> list[str]:
    t = flat(text)
    return [
        f"{surface}: retired phrase {phrase!r}"
        for phrase, any_case in RETIRED_PHRASES
        if (phrase.lower() in t.lower() if any_case else flat(phrase) in t)
    ]


#: Every form "confirm" may take: negated for a bearing claim, or an action
#: of kept text that confirms an input with the user.
CONFIRM_FORMS = (
    "Neither check_bearing_faults nor the bearing block of diagnose_vibration "
    "confirms a bearing defect",
    "never 'confirmed', 'damage detected' or 'fault present' for bearing claims",
    'never write "confirmed", "damage detected" or "fault present" for a '
    "bearing claim",
    "Neither an outcome nor a claim level confirms a fault",
    "No outcome confirms a fault",
    "does not identify or confirm a fault type",
    "never as a confirmed fault",
    "not a confirmed fault",
    "ASK USER to confirm machine parameters",
    "If user confirms or provides values",
    "confirm the ACTUAL measurement unit with the user",
    "<confirmed-unit>",
    "Confirm it: compare with a comparable earlier measurement",
    "Confirm with the user; pass machine_power_kw",
    "Confirm the unit with the user",
    "confirm this to the user if they ask",
)
_CONFIRM_RE = re.compile(r"\bconfirm(?:s|ed)?\b", re.I)

#: Every form "healthy" may take: a negation or a quoted retired reading.
HEALTHY_FORMS = (
    "Never report no_indication as 'healthy' or 'no fault'",
    "≠ healthy signal",
    "does not establish that the machine is healthy",
)
_HEALTHY_RE = re.compile(r"\bhealthy\b", re.I)


def _outside_forms(
    surface: str, text: str, forms: tuple[str, ...], pattern: re.Pattern[str]
) -> list[str]:
    t = flat(text)
    for form in sorted((flat(f) for f in forms), key=len, reverse=True):
        t = t.replace(form, " ")
    return [
        f"{surface}: {m.group(0)!r} outside a listed form: "
        f"...{t[max(0, m.start() - 60) : m.end() + 40]}..."
        for m in pattern.finditer(t)
    ]


def confirm_violations(surface: str, text: str) -> list[str]:
    return _outside_forms(surface, text, CONFIRM_FORMS, _CONFIRM_RE)


def healthy_violations(surface: str, text: str) -> list[str]:
    return _outside_forms(surface, text, HEALTHY_FORMS, _HEALTHY_RE)


# ---------------------------------------------------------------------------
# Required phrases, and the server sentences they echo
# ---------------------------------------------------------------------------

FALLBACK = (
    "If a `check_bearing_faults` result, or a non-null `bearing_faults` block "
    "of a `diagnose_vibration` result, has no `claim` field, the server "
    "predates the bearing claim: infer or relay no bearing fault type or "
    "bearing assessment from it (its `overall_assessment`, "
    "`most_likely_fault`, `fault_checks`, bearing sentence or Bearing "
    "recommendations), and tell the user that this bearing guidance needs a "
    "server whose `check_bearing_faults` returns `claim`. The "
    '`evidence_strength` of such a `diagnose_vibration` result, and its "Fault '
    'evidence strength" sentence, also count the bearing finding, so neither '
    "is relayed; results that do not rest on the bearing check, such as the "
    "ISO 20816-3 zone and its action, are relayed as usual."
)
REPORT_FIELDS = (
    "`bearing_matches` and `bearing_assessment` are not the bearing evidence: "
    "report the claim of `check_bearing_faults`, and with a server whose check "
    "returns no `claim`, relay neither"
)
NO_INDICATION = "no_indication is not evidence that the bearing is free of defects"
NOT_FREE_OF_DEFECTS = "is not evidence that the bearing is free of defects"
NOT_CONFIRMED = "not a confirmed fault"
COUNTS_FINDINGS = "counts findings"
NOT_IDENTIFY_OR_CONFIRM = "does not identify or confirm"
ONE_SPECTRUM = "lines of one envelope spectrum"
NOT_HEALTHY = "does not establish that the machine is healthy"
ASKS_CONFIRMATION = "asks for confirmation"
OWN_MATCHING = "never your own matching of envelope peaks"
RELATIVE_EVIDENCE = (
    "supported bearing evidence above the reference band in the acquisitions "
    "its criterion counts"
)

#: (phrase, surfaces that must carry it).
REQUIRED_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Single out a bearing fault type only at claim.level "
        "'single_spectrum_evidence'; quote overall_assessment.",
        (INSTRUCTIONS,),
    ),
    (
        "Never independently match envelope peaks or promote the server's "
        "returned claim level based on statistics, harmonics, or other analyses.",
        (INSTRUCTIONS,),
    ),
    (
        "Neither check_bearing_faults nor the bearing block of diagnose_vibration "
        "confirms a bearing defect",
        (INSTRUCTIONS, SKILL, AGENT),
    ),
    ("('no_indication', 'no_change', 'none') is not health", (INSTRUCTIONS,)),
    ("unclassified, not absent or negative", (INSTRUCTIONS,)),
    (
        "NEVER suggest parameters, thresholds or actions not in a tool output or "
        "prompt workflow.",
        (INSTRUCTIONS,),
    ),
    ("asks for confirmation, never a replacement", (INSTRUCTIONS, COMMAND, AGENT)),
    ("never a replacement", (SKILL, DIAGNOSIS)),
    (
        "'evidence_strength' is not a percentage, probability or confidence",
        (INSTRUCTIONS,),
    ),
    (f"per check it grades {ONE_SPECTRUM}", (INSTRUCTIONS,)),
    (f"per diagnosis it {COUNTS_FINDINGS} and names no fault type", (INSTRUCTIONS,)),
    ("the analysis does not answer it", (INSTRUCTIONS, SKILL, AGENT)),
    (OWN_MATCHING, (*PROMPTS, PROMPT_DESC, SKILL, COMMAND, AGENT)),
    ("never raise the claim", (*PROMPTS, PROMPT_DESC, SKILL)),
    ("No outcome confirms a fault", (PROMPT_DESC,)),
    (NO_INDICATION, (*PROMPTS, PROMPT_DESC, CHECK, SKILL, COMMAND, AGENT)),
    ("claim.level, not outcome, decides", (*PROMPTS, SKILL, COMMAND, AGENT)),
    ("not outcome, decides what may be said about a fault type", (CHECK,)),
    ("Neither an outcome nor a claim level confirms a fault", (CHECK,)),
    ("then the further lines its kinematics predict", (*PROMPTS, SKILL)),
    ("tested at twice it", (*PROMPTS, CHECK, SKILL)),
    (f"grades {ONE_SPECTRUM}", (CHECK, SKILL)),
    ("the false-alarm budget applies to one check", (CHECK,)),
    ("which does not rank them", (CHECK,)),
    (
        "Withheld claim + Zone A/B → The next_step of check_bearing_faults only as "
        "STEP 5 A allows",
        PROMPTS,
    ),
    (
        "No action on a single element. Relay the next_step of "
        "check_bearing_faults only as STEP 5 A allows",
        PROMPTS,
    ),
    ("`next_step` only as Phase 4 allows", (AGENT,)),
    ("no overall health verdict", (AGENT,)),
    (FALLBACK, PLUGIN),
    (REPORT_FIELDS, PLUGIN),
    ("counts the findings of the analyses that ran", (DIAGNOSIS,)),
    (f"{NOT_IDENTIFY_OR_CONFIRM} a fault type", (DIAGNOSIS,)),
    (f"'none' {NOT_HEALTHY}", (DIAGNOSIS,)),
    (
        "or from any other bearing evidence this server reports (such as "
        "assess_asset_change)",
        (RECOMMENDATIONS,),
    ),
    ("neither is the classification", (LEDGER,)),
    ("relative_present_in_last_k of relative_k", (LEDGER,)),
    (f"a no_change classification {NOT_FREE_OF_DEFECTS}", (LEDGER,)),
    (
        "it is not classified, never absent, and never evidence of no fault or of "
        "no change",
        (LEDGER,),
    ),
    (
        "a legacy presence or grade, recorded by an earlier algorithm, is never "
        "current evidence for a fault type",
        (LEDGER,),
    ),
    (f"{RELATIVE_EVIDENCE}, {NOT_CONFIRMED}", (LEDGER,)),
)
#: Phrases a surface carries more than once: the skill gives the fallback at
#: the line test and again at the one-call diagnosis.
REQUIRED_COUNTS: dict[tuple[str, str], int] = {(FALLBACK, SKILL): 2}


def required_phrase_violations(surface: str, text: str) -> list[str]:
    t = flat(text)
    violations = []
    for phrase, where in REQUIRED_PHRASES:
        if surface not in where:
            continue
        wanted = REQUIRED_COUNTS.get((phrase, surface), 1)
        found = t.count(flat(phrase))
        if found < wanted:
            violations.append(
                f"{surface}: carries {found} of {wanted} {phrase[:70]!r}..."
            )
    return violations


#: (phrase, the server sentence it echoes). Each phrase sits inside a
#: required phrase above, so a rewritten server sentence fails here first.
SERVER_ECHOES: tuple[tuple[str, Callable[[], str]], ...] = (
    (NOT_FREE_OF_DEFECTS, lambda: CLAIM_SENTENCES["no_indication"]),
    (NOT_CONFIRMED, lambda: CLAIM_SENTENCES["assumption"]),
    (COUNTS_FINDINGS, lambda: CLAIM_SENTENCES["aggregate"]),
    (NOT_IDENTIFY_OR_CONFIRM, lambda: CLAIM_SENTENCES["evidence_strength_explanation"]),
    (
        ONE_SPECTRUM,
        lambda: BearingFaultCheckResult.model_fields["evidence_strength"].description,
    ),
    (
        NOT_HEALTHY,
        lambda: DiagnosisResult.model_fields["evidence_strength"].description,
    ),
    (ASKS_CONFIRMATION, lambda: BearingClaim.model_fields["level"].description),
    (NOT_CONFIRMED, lambda: BearingClaim.model_fields["level"].description),
    (assessment._RELATIVE_EVIDENCE, lambda: RELATIVE_EVIDENCE),
)


# ---------------------------------------------------------------------------
# Instruction window, workflow calls, prompt/plugin consistency
# ---------------------------------------------------------------------------

#: Starts of the passages that must end inside the window: rules 3, 4 and 6,
#: the evidence_strength bullet and the bullet on unanswered questions.
WINDOW_PASSAGES = (
    "3) Single out",
    "4) Neither check_bearing_faults",
    "6) NEVER suggest",
    "- This server publishes NO",
    "- If a question is not answered",
)


def passage_end(text: str, start: int) -> int:
    """End of a numbered rule (its line) or of a bullet (with its continuation)."""
    if not text.startswith("- ", start):
        return text.index("\n", start)
    end = text.index("\n", start)
    while True:
        rest = text[end + 1 :]
        if (
            rest.lstrip(" ").startswith("-")
            or rest.startswith("\n")
            or not rest.strip()
        ):
            return end
        end = text.index("\n", end + 1)


def window_violations(surface: str, text: str) -> list[str]:
    violations = []
    for marker in WINDOW_PASSAGES:
        start = text.find(marker)
        if start < 0:
            violations.append(f"{surface}: passage {marker!r} is gone")
            continue
        end = passage_end(text, start)
        if end >= INSTRUCTION_WINDOW:
            violations.append(
                f"{surface}: {marker!r} ends at {end}, past the observed "
                f"window of {INSTRUCTION_WINDOW} characters"
            )
    return violations


def envelope_call_violations(surface: str, text: str) -> list[str]:
    called = {name for name, _, _ in extract_calls(text)}
    violations = []
    if "analyze_envelope" in called:
        violations.append(f"{surface}: a bearing workflow calls analyze_envelope")
    if "check_bearing_faults" not in called:
        violations.append(
            f"{surface}: the workflow no longer calls check_bearing_faults"
        )
    return violations


def step_5a_sentences(prompt: str) -> dict[str, str]:
    """The prompt's Step 5 A relay sentence per level, and its next_step rule."""
    start = prompt.find("A) Report the server's bearing claim")
    end = prompt.find("B) Name the outcome", start)
    assert 0 <= start < end, "Step 5 A of diagnose_bearing is gone"
    block = prompt[start:end]
    sentences = {}
    for level in LEVELS:
        m = re.search(
            rf"•\s*{level}\s*→\s*(.*?)(?=\n\s*•|\n\s*most_likely_fault|\Z)",
            block,
            re.S,
        )
        assert m, f"Step 5 A has no relay bullet for {level}"
        sentences[level] = flat(m.group(1))
    m = re.search(
        r"outcome_reason and next_step describe.*?a single element\.", block, re.S
    )
    assert m, "Step 5 A no longer limits outcome_reason and next_step"
    sentences["next_step"] = flat(m.group(0))
    return sentences


#: Where each Step 5 A sentence must also appear.
STEP_5A_SURFACES = {
    "single_spectrum_evidence": (SKILL,),
    "qualified": (SKILL,),
    "withheld": (SKILL,),
    "next_step": (CHECK, SKILL, COMMAND, AGENT),
}


def relay_consistency_violations(surface: str, text: str, prompt: str) -> list[str]:
    t = flat(text).casefold()
    return [
        f"{surface}: lacks the prompt's Step 5 A sentence for {key}: {sentence!r}"
        for key, sentence in step_5a_sentences(prompt).items()
        if surface in STEP_5A_SURFACES[key] and sentence.casefold() not in t
    ]


def zone_sentence_violations(surface: str, text: str) -> list[str]:
    sentence = CLAIM_SENTENCES["verdict_zone_elevated"].format(zone="[C/D]")
    if flat(sentence) in flat(text):
        return []
    return [f"{surface}: the zone sentence is not the server's {sentence!r}"]


# ---------------------------------------------------------------------------
# The guards on the real surfaces
# ---------------------------------------------------------------------------


def _check(guard, surface: str) -> None:
    violations = guard(surface, surfaces()[surface])
    assert not violations, "\n".join(violations)


def test_the_instruction_headings_are_present():
    instructions = full_instructions()
    assert "Evidence-based inference policy" in instructions
    assert "Signal unit policy" in instructions


@pytest.mark.parametrize("surface", SCANNED)
def test_outcome_vocabulary_matches_the_runtime_literal(surface):
    _check(outcome_violations, surface)


@pytest.mark.parametrize("surface", SCANNED)
def test_claim_level_vocabulary_matches_the_runtime_literal(surface):
    _check(claim_level_violations, surface)


@pytest.mark.parametrize("surface", (INSTRUCTIONS, LEDGER))
def test_ledger_bases_match_the_runtime_constant(surface):
    _check(ledger_basis_violations, surface)


@pytest.mark.parametrize("surface", SCANNED)
def test_retired_phrases_absent(surface):
    _check(retired_phrase_violations, surface)


@pytest.mark.parametrize("surface", SCANNED)
def test_confirm_only_in_listed_forms(surface):
    _check(confirm_violations, surface)


@pytest.mark.parametrize("surface", SCANNED)
def test_healthy_only_in_negated_or_quoted_form(surface):
    _check(healthy_violations, surface)


@pytest.mark.parametrize("surface", SCANNED)
def test_required_phrases_per_surface(surface):
    _check(required_phrase_violations, surface)


@pytest.mark.parametrize(
    "phrase, source", SERVER_ECHOES, ids=[e[0] for e in SERVER_ECHOES]
)
def test_pins_are_server_sentences(phrase, source):
    assert flat(phrase) in flat(source()), f"{phrase!r} is not in {source()!r}"
    assert any(
        flat(phrase) in flat(p) for p, _ in REQUIRED_PHRASES
    ), f"{phrase!r} echoes the server but no surface is required to carry it"


def test_bsf_convention_matches_the_line_tested():
    """A BSF value is the spin frequency; the line tested is twice it."""
    assert Hypothesis("BSF", 70.6, "ball").line_hz == pytest.approx(141.2)


def test_instruction_rules_inside_the_client_window():
    """Observed truncation of the server instructions in one Claude Code build
    (an empirical observation, not an MCP contract); recheck when the client
    changes."""
    _check(window_violations, INSTRUCTIONS_FULL)


@pytest.mark.parametrize("surface", WORKFLOWS)
def test_bearing_workflows_call_no_analyze_envelope(surface):
    _check(envelope_call_violations, surface)


@pytest.mark.parametrize("surface", (CHECK, *PLUGIN))
def test_prompt_and_plugin_relay_the_claim_alike(surface):
    violations = relay_consistency_violations(
        surface, surfaces()[surface], surfaces()[PROMPT_FULL]
    )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("surface", PROMPTS)
def test_zone_sentence_is_the_servers(surface):
    _check(zone_sentence_violations, surface)


# ---------------------------------------------------------------------------
# Exemptions and listed forms stay current
# ---------------------------------------------------------------------------

_TEXT_GUARDS = (retired_phrase_violations, confirm_violations, healthy_violations)


def test_exemptions_are_not_scanned_surfaces():
    assert not set(EXEMPT) & set(SCANNED)
    assert all(reason.strip() for reason in EXEMPT.values())


@pytest.mark.parametrize("surface", sorted(EXEMPT))
def test_every_exemption_still_excuses_something(surface):
    """An exemption for a surface that no longer drifts excuses nothing."""
    text = surfaces()[surface]
    assert any(
        guard(surface, text) for guard in _TEXT_GUARDS
    ), f"{surface} no longer drifts: scan it instead of exempting it"


@pytest.mark.parametrize(
    "form, forms",
    [(f, CONFIRM_FORMS) for f in CONFIRM_FORMS]
    + [(f, HEALTHY_FORMS) for f in HEALTHY_FORMS],
)
def test_every_listed_form_is_in_use(form, forms):
    """A listed form no scanned surface uses would excuse a future slip."""
    assert any(flat(form) in flat(surfaces()[s]) for s in SCANNED), form


# ---------------------------------------------------------------------------
# Red paths: each guard fails on a mutated copy of a real surface
# ---------------------------------------------------------------------------


class Mutation(NamedTuple):
    case: str
    surface: str
    guard: Callable[[str, str], list[str]]
    mutate: Callable[[str], str]
    raw: bool = False  # mutate the text as served, not its flat form


def _swap(old: str, new: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert old in text, f"mutation target {old!r} is not in the surface"
        return text.replace(old, new, 1)

    return mutate


def _delete(old: str) -> Callable[[str], str]:
    return _swap(old, "")


def _append(extra: str) -> Callable[[str], str]:
    return lambda text: text + extra


def _move_past_window(marker: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        start = text.rfind("\n", 0, text.index(marker)) + 1
        end = passage_end(text, text.index(marker)) + 1
        return text[:start] + text[end:] + text[start:end]

    return mutate


def _relay_check(surface: str, text: str) -> list[str]:
    return relay_consistency_violations(surface, text, surfaces()[PROMPT_FULL])


MUTATIONS = (
    # outcome names
    Mutation(
        "duplicate: an off-list outcome joins rule 3's list",
        INSTRUCTIONS,
        outcome_violations,
        _swap('"several_types".', '"several_types", "possible_fault".'),
    ),
    Mutation(
        "remove: the skill's outcome list loses inconclusive",
        SKILL,
        outcome_violations,
        _swap("no_indication, inconclusive, indicated", "no_indication, indicated"),
    ),
    # claim levels
    Mutation(
        "substitute: supported for withheld in the skill's claim table",
        SKILL,
        claim_level_violations,
        _swap("| `withheld` |", "| `supported` |"),
        raw=True,
    ),
    Mutation(
        "substitute: supported for withheld in a Step 5 A relay bullet",
        PROMPT_FULL,
        claim_level_violations,
        _swap("• withheld →", "• supported →"),
        raw=True,
    ),
    Mutation(
        "add: the reserved level supported_claim in the agent",
        AGENT,
        claim_level_violations,
        _swap("at `single_spectrum_evidence`", "at `supported_claim`"),
        raw=True,
    ),
    # ledger bases
    Mutation(
        "substitute: an off-list ledger basis in rule 4",
        INSTRUCTIONS,
        ledger_basis_violations,
        _swap('"unavailable" ledger evidence', '"stale" ledger evidence'),
    ),
    Mutation(
        "remove: the ledger description loses reference_relative",
        LEDGER,
        ledger_basis_violations,
        lambda text: text.replace("reference_relative", "reference"),
    ),
    # retired phrases
    Mutation(
        "add: the old evidence scale in the command",
        COMMAND,
        retired_phrase_violations,
        _append(" Strong evidence: peak + harmonics + corroborating statistics."),
    ),
    Mutation(
        "add: peak matching in the skill",
        SKILL,
        retired_phrase_violations,
        _append(" Check if ANY peak falls within ±5% of expected frequencies."),
    ),
    # "confirm": negate and delete-negator
    Mutation(
        "negate: the ledger's 'not a confirmed fault'",
        LEDGER,
        confirm_violations,
        _swap("not a confirmed fault", "a confirmed fault"),
    ),
    Mutation(
        "negate: the skill table's 'not a confirmed fault'",
        SKILL,
        confirm_violations,
        _swap("it is not a confirmed fault", "it is a confirmed fault"),
    ),
    *(
        Mutation(
            f"delete-negator: 'Neither' of {surface}",
            surface,
            confirm_violations,
            _delete("Neither "),
        )
        for surface in (INSTRUCTIONS, CHECK, SKILL, AGENT)
    ),
    Mutation(
        "delete-negator: 'never' of rule 4",
        INSTRUCTIONS,
        confirm_violations,
        _swap('never "confirmed"', '"confirmed"'),
    ),
    *(
        Mutation(
            f"delete-negator: 'never write' of {surface}",
            surface,
            confirm_violations,
            _swap('never write "confirmed"', 'write "confirmed"'),
        )
        for surface in (SKILL, AGENT)
    ),
    Mutation(
        "delete-negator: 'No' of 'No outcome confirms a fault'",
        PROMPT_DESC,
        confirm_violations,
        _swap("No outcome confirms", "Outcome confirms"),
    ),
    # "healthy"
    *(
        Mutation(
            f"delete-negator: 'Never report no_indication as healthy' of {surface}",
            surface,
            healthy_violations,
            _swap("Never report no_indication", "Report no_indication"),
        )
        for surface in (PROMPT_MIN, SKILL)
    ),
    Mutation(
        "add: a health verdict in the command",
        COMMAND,
        healthy_violations,
        _append(" Report the bearing as healthy."),
    ),
    # required phrases: remove the marker, a clause, or a negator
    Mutation(
        "remove-marker: the command loses the fallback",
        COMMAND,
        required_phrase_violations,
        _delete(flat(FALLBACK)),
    ),
    Mutation(
        "remove-marker: the agent loses the envelope-report sentence",
        AGENT,
        required_phrase_violations,
        _delete(flat(REPORT_FIELDS)),
    ),
    *(
        Mutation(
            f"remove-clause: '{clause}' of the fallback in {surface}",
            surface,
            required_phrase_violations,
            _delete(clause),
        )
        for surface in PLUGIN
        for clause in (
            "infer or relay no bearing fault type or bearing assessment from it ",
            ", so neither is relayed",
            "non-null ",
        )
    ),
    Mutation(
        "delete-negator: 'no' of 'relay no bearing fault type'",
        SKILL,
        required_phrase_violations,
        _swap("relay no bearing fault type", "relay bearing fault type"),
    ),
    Mutation(
        "remove-clause: 'relay neither' of the envelope-report sentence",
        SKILL,
        required_phrase_violations,
        _swap("returns no claim, relay neither.", "returns no claim."),
    ),
    Mutation(
        "delete-negator: 'not' of 'are not the bearing evidence'",
        COMMAND,
        required_phrase_violations,
        _swap("are not the bearing evidence", "are the bearing evidence"),
    ),
    Mutation(
        "remove-clause: Step 6 row 4 loses 'only as STEP 5 A allows'",
        PROMPT_MIN,
        required_phrase_violations,
        _swap(
            "next_step of check_bearing_faults only as STEP 5 A allows, else",
            "next_step of check_bearing_faults, else",
        ),
    ),
    Mutation(
        "remove-clause: the agent's 'only as Phase 4 allows'",
        AGENT,
        required_phrase_violations,
        _swap("next_step only as Phase 4 allows", "next_step"),
    ),
    Mutation(
        "delete-negator: 'Never' of rule 3",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("Never independently", "Independently"),
    ),
    Mutation(
        "delete-negator: 'not' of 'is not health'",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("is not health", "is health"),
    ),
    Mutation(
        "delete-negator: 'not' of 'not absent or negative'",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("unclassified, not absent", "unclassified, absent"),
    ),
    Mutation(
        "delete-negator: 'NEVER' of rule 6",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("6) NEVER suggest", "6) suggest"),
    ),
    Mutation(
        "delete-negator: 'never' of rule 6's 'never a replacement'",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("confirmation, never a replacement", "confirmation, a replacement"),
    ),
    Mutation(
        "delete-negator: 'not' of the evidence_strength bullet",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("is not a percentage", "is a percentage"),
    ),
    Mutation(
        "delete-negator: 'no' of 'names no fault type'",
        INSTRUCTIONS,
        required_phrase_violations,
        _swap("names no fault type", "names the fault type"),
    ),
    Mutation(
        "delete-negator: 'never' of 'never your own matching'",
        COMMAND,
        required_phrase_violations,
        _swap("never your own matching", "your own matching"),
    ),
    Mutation(
        "delete-negator: 'never' of 'never raise the claim'",
        PROMPT_DESC,
        required_phrase_violations,
        _swap("never raise the claim", "raise the claim"),
    ),
    Mutation(
        "delete-negator: 'not' of the check's no-indication sentence",
        CHECK,
        required_phrase_violations,
        _swap("no_indication is not evidence", "no_indication is evidence"),
    ),
    Mutation(
        "delete-negator: 'never' of 'never evidence of no fault'",
        LEDGER,
        required_phrase_violations,
        _swap("and never evidence of no fault", "and evidence of no fault"),
    ),
    Mutation(
        "delete-negator: 'never' of 'never absent'",
        LEDGER,
        required_phrase_violations,
        _swap("never absent,", "absent,"),
    ),
    Mutation(
        "delete-negator: 'no' of 'no overall health verdict'",
        AGENT,
        required_phrase_violations,
        _swap("no overall health verdict", "an overall health verdict"),
    ),
    Mutation(
        "delete-negator: 'does not' of 'does not identify or confirm'",
        DIAGNOSIS,
        required_phrase_violations,
        _swap("it does not identify or confirm", "it does identify or confirm"),
    ),
    # window
    Mutation(
        "relocate: rule 3 past the window",
        INSTRUCTIONS_FULL,
        window_violations,
        _move_past_window("3) Single out"),
        raw=True,
    ),
    Mutation(
        "relocate: the unanswered-question bullet past the window",
        INSTRUCTIONS_FULL,
        window_violations,
        _move_past_window("- If a question is not answered"),
        raw=True,
    ),
    # workflow calls
    *(
        Mutation(
            f"add: an analyze_envelope call in {surface}",
            surface,
            envelope_call_violations,
            _append('\nCall `analyze_envelope(signal_id="<id>")`.\n'),
            raw=True,
        )
        for surface in (PROMPT_FULL, SKILL, AGENT)
    ),
    # prompt/plugin consistency and the zone sentence
    Mutation(
        "remove-clause: the command's next_step rule loses 'or a single element'",
        COMMAND,
        _relay_check,
        _swap("a fault type or a single element", "a fault type"),
    ),
    Mutation(
        "substitute: the skill's qualified row singles one out",
        SKILL,
        _relay_check,
        _swap("single out none", "single out the likeliest"),
    ),
    Mutation(
        "substitute: the zone sentence leans to a bearing source",
        PROMPT_MIN,
        zone_sentence_violations,
        _swap("both remain possible", "point to the bearing"),
    ),
)


@pytest.mark.parametrize("m", MUTATIONS, ids=[m.case for m in MUTATIONS])
def test_guards_fail_on_mutations(m):
    text = surfaces()[m.surface]
    base = text if m.raw else flat(text)
    assert m.guard(m.surface, base) == [], "the guard is not green on the real surface"
    mutated = m.mutate(base)
    assert mutated != base, "the mutation changed nothing"
    assert m.guard(m.surface, mutated), f"{m.case}: the guard stayed green"
