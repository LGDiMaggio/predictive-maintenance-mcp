"""Model-facing guidance for assess_shaft_fault_evidence stays bound to its contract.

The plugin and the server prompts tell a client model when to call the
experimental shaft-order evidence tool and how to read it. This guard keeps
that guidance from drifting away from the registered tool and its approved
scientific contract:

- placement: the tool and its skill are named only in the surfaces designed
  for them; outside the skill, every passage that shows a call says the step
  is optional and none says it must or always runs, so no bearing, gear or
  screening workflow makes it mandatory;
- calls: every documented call's literal argument values validate against the
  tool's registered input schema and against the bounds the tool enforces
  (names alone are already guarded by tests/test_documented_calls.py);
- machine parameters: no documented call passes a literal speed, direction,
  machine class or other machine parameter, so no example value can be
  copied into a real diagnosis; the guidance shows signal_ids (and detail)
  and the server resolves the rest from declared context or the user;
- fields: every result field, status and reason the guidance names in
  backticks, in the skill and in the passages about the tool elsewhere, exists
  in the tool's registered schemas;
- semantics: each surface keeps the statements the contract requires (no
  fault attribution, pattern_observed is not attribution, an observed pattern
  stays inconclusive, ISO severity is context only, agreement with
  diagnose_vibration is not corroboration) and no surface uses an affirmative
  fault phrasing.

The wording guard screens the guidance's own sentences, not what a model may
discuss: background on how a fault can show in vibration, named next to the
other explanations, passes; a phrase that attributes a fault from a line
("possible unbalance", "typical of misalignment") does not. It is a heuristic
screen, not a proof: a negation within the
twelve words before a phrase exempts it ("never write that unbalance is
present"), and it reads text, not meaning. Each guard is shown to fail on
mutations of the real text.
"""

import ast
import functools
import re
from typing import Any, Callable, NamedTuple

import jsonschema
import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.diagnostics.shaft_fault_evidence import MAX_CHANNELS
from predictive_maintenance_mcp.diagnostics.shaft_orders import (
    MAX_MAX_ORDER,
    MAX_SPEED_TOLERANCE_PCT,
    MIN_MAX_ORDER,
)
from predictive_maintenance_mcp.mcp_tools import register_all
from test_documented_calls import (
    REPO_ROOT,
    extract_calls,
    rendered_prompt_texts,
    split_top_level_args,
)

TOOL = "assess_shaft_fault_evidence"
SKILL_NAME = "shaft-order-evidence"
SKILL = "plugin/skills/shaft-order-evidence/SKILL.md"
SCREENING = "plugin/skills/quick-screening/SKILL.md"
AGENT = "plugin/agents/diagnostic-pipeline.md"
PLUGIN_README = "plugin/README.md"
PROMPT = "prompts.quick_diagnostic_report"

#: Where the tool may be named. A new mention elsewhere is a design change:
#: review it, then add the surface here with its required statements.
EXPECTED_SURFACES = {SKILL, AGENT, PLUGIN_README, PROMPT}
#: Where the skill may be named (it names itself in its own file).
SKILL_NAME_SURFACES = {SKILL, SCREENING, AGENT, PLUGIN_README}
#: Surfaces whose wording is checked.
CHECKED_SURFACES = EXPECTED_SURFACES | {SCREENING}

#: Statements each surface must keep, matched case-insensitively with
#: whitespace collapsed and backticks dropped (a change of quoting is not drift).
REQUIRED: dict[str, tuple[str, ...]] = {
    SKILL: (
        "attributes no fault",
        "pattern_observed true, status inconclusive",
        "no fault is attributed",
        "reasons always include attribution_not_validated",
        "never, from this result, that unbalance is present, possible or suspected",
        "which does not exclude the fault",
        "says nothing about that family",
        "context only",
        "repeats the observation and is not corroboration",
        "never pass a family from this result to generate_maintenance_recommendations",
        "quote summary and each hypothesis statement verbatim",
        "relay limitations as stated with every observed pattern",
        "not_assessable_from_record",
        "pass null for a direction nobody declares",
        "a direction declared only for the measurement point in the asset ledger is"
        " not read",
        "add a parameter only when the user supplies it and it applies to these"
        " channels",
        "ask the user for the shaft speed and call again with it. never estimate it",
        "is background, never independent evidence that the observed pattern is that"
        " fault",
        "it does not stop the discussion",
        "recommend the measurements in missing",
        "**optional**",
        "not yet in a published release",
        "do not substitute your own reading of fft peaks",
        "a significant line is not abnormality",
    ),
    SCREENING: (
        "names no fault on its own",
        "attributes no fault",
        "it is optional and not part of the screening card",
        "a line at 1x or 2x attributes no unbalance, misalignment or looseness; they"
        " may be named as possible explanations next to the others",
    ),
    AGENT: (
        "(optional; a bearing or gear check never needs it)",
        "and each channel's direction from what its signal declares",
        "if the call is refused because no shaft speed is declared, ask the user for"
        " it and call again with rpm set to the value they give; never estimate it",
        "only when the user supplies them for these channels",
        "is not read: ask the user for it",
        "never a fault",
        "every observed pattern is inconclusive",
        "never write from this result that unbalance, misalignment or looseness is"
        " present, possible or suspected",
        "not_observed does not exclude the fault",
        "insufficient_data says nothing about the family",
        "relay limitations with every observed pattern",
        "context only",
        "repeats the observation and corroborates nothing",
        "verbatim",
        "may be named as possible explanations next to the others, never attributed"
        " from a strongest line at 1x or 2x or from the shaft-order assessment",
        "a re-reading of the same line of the same signal, or a manual describing a"
        " compatible pattern, is not independent evidence",
    ),
    PLUGIN_README: ("attributes no fault",),
    PROMPT: (
        "optional",
        "attributes no fault",
        "a peak at 1×/2× rpm names no fault",
        "if no speed is declared, ask the user for it, and pass rpm or directions only"
        " as the user gives them",
        "unbalance and misalignment also need a declared signal unit",
        "may be named as possible explanations next to the others, never attributed"
        " from a peak at 1×/2× rpm",
        "a re-reading of the same peak of the same signal, or a manual describing a"
        " compatible pattern, is not independent evidence",
    ),
}

#: Affirmative fault phrasings: a finding verb or a likelihood word before a
#: family, a family with a finding or likelihood predicate, 'unbalanced', a
#: confirmation of the pattern, or a 'consistent' status.
FAMILY = r"(?:rotor\s+)?(?:unbalance|imbalance|misalignment|looseness)"
FORBIDDEN = [
    re.compile(
        rf"\b(?:confirm(?:s|ed)?|detect(?:s|ed)?|diagnos(?:e|es|ed|is)|indicat(?:e|es|ed)"
        rf"|identif(?:y|ies|ied)|establish(?:es|ed)?|suggest(?:s|ed)?"
        rf"|points?\s+to|consistent\s+with|possible|suspected|probable|likely"
        rf"|evidence\s+of|signs?\s+of|typical\s+of|characteristic\s+of)"
        rf"\s+(?:an?\s+|the\s+|some\s+)?{FAMILY}"
        rf"(?!\s+(?:patterns?|famil(?:y|ies))\b)",
        re.I,
    ),
    re.compile(
        rf"\b(?:shows?|reveals?)\s+(?:an?\s+|the\s+|some\s+)?{FAMILY}"
        rf"(?!\s+(?:hypothes[ie]s|patterns?|famil(?:y|ies))\b)",
        re.I,
    ),
    re.compile(
        rf"{FAMILY}\s+(?:is\s+|are\s+|was\s+)?(?:confirmed|present|detected|likely"
        rf"|probable|possible|suspected|indicated)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:unbalanced|imbalanced|misaligned)\b"
        r"(?!\s+(?:channels?|sensors?|directions?|axes|accelerometers?)\b)"
        r"|\b(?:is|are|was|were|be|been|looks?|seems?|appears?)\s+(?:\w+\s+)?loose\b"
        r"|\bloose\s+(?:hold-down\s+)?(?:bolts?|foot|feet|fits?|base(?:plate)?|pedestal"
        r"|housing|bearing\s+caps?)\b",
        re.I,
    ),
    re.compile(
        r"\bmost\s+(?:probable|likely)\s+cause\b",
        re.I,
    ),
    re.compile(
        r"\bconfirm(?:s|ed)?\s+(?:\w+\s+){0,3}(?:pattern|finding|fault|hypothes[ie]s)\b",
        re.I,
    ),
    re.compile(
        r"status(?:\s+is)?\s+`?consistent`?|`consistent`|'consistent'|\"consistent\"",
        re.I,
    ),
]
#: Negations that exempt a phrasing when they precede it closely ('no' does
#: not: "no sidebands, so unbalance is likely" stays affirmative).
_NEGATION = re.compile(r"\b(?:not|never|nothing|whether|neither|nor)\b", re.I)
_NEGATION_WORDS = 12
#: The corroboration rule applies to sentences about this tool or the
#: diagnose_vibration reading it could be mistaken to agree with.
_CORROBORATION_SCOPE = re.compile(
    rf"{TOOL}|diagnose_vibration|agree|results?\b|assessments?\b|analys[ie]s|fft", re.I
)
_CORROBORATION = re.compile(
    r"corroborat|independent\s+evidence|strengthens?\s+the\s+case"
    r"|(?:reinforces?|confirms?)\s+each\s+other",
    re.I,
)
_WORKFLOW_NOUN = (
    r"(?:bearing|gear|signal|report|check|run|screening|diagnos\w*|machine|record)s?"
)
_MANDATORY = re.compile(
    r"\b(?:must|mandatory|required\s+(?:step|for\s+(?:every|each|all))"
    r"|(?:step|it|tool|call)\s+is\s+required"
    r"|always\s+(?:run|call|use|invoke|perform|start|begin|do|add|suggest|recommend"
    r"|request|assess)"
    r"|(?:after|before|on)\s+(?:every|each)"
    rf"|(?:for|in)\s+(?:every|each)\s+(?:\w+\s+)?{_WORKFLOW_NOUN}"
    r"|for\s+all\s+(?:signals|machines|reports|checks)"
    r"|every\s+time|each\s+time|never\s+skip|do\s+not\s+skip|don't\s+skip)\b",
    re.I,
)


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def blocks(text: str) -> list[str]:
    """Paragraph blocks (a lead-in line with its list items), flattened."""
    return [flat(p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunks(text: str) -> list[str]:
    """Paragraphs and list items of markdown text, flattened."""
    parts = re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+[.)])\s)", text)
    return [flat(p).strip() for p in parts if p.strip()]


def sentences(text: str) -> list[str]:
    out = []
    for chunk in chunks(text):
        out += [x for x in re.split(r"(?<=[.;!?])\s+", chunk) if x]
    return out


def _negated_before(sentence: str, start: int) -> bool:
    words = sentence[:start].split()[-_NEGATION_WORDS:]
    return bool(_NEGATION.search(" ".join(words)))


@functools.cache
def surfaces() -> dict[str, str]:
    """Every plugin markdown file and rendered prompt, by id."""
    out = {
        str(p.relative_to(REPO_ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in sorted((REPO_ROOT / "plugin").rglob("*.md"))
    }
    out.update(rendered_prompt_texts())
    return out


@functools.cache
def tool_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """The registered (input, output) JSON schemas of the tool."""
    server = MCPServer("shaft-guidance-guard")
    register_all(server)
    tool = server._tool_manager._tools[TOOL]
    return tool.parameters, tool.output_schema


def _schema_vocabulary(schema: Any) -> set[str]:
    """Every property name and every string enum or const value in *schema*."""
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                found |= set(value)
            if key == "enum" and isinstance(value, list):
                found |= {v for v in value if isinstance(v, str)}
            if key == "const" and isinstance(value, str):
                found.add(value)
            found |= _schema_vocabulary(value)
    elif isinstance(schema, list):
        for item in schema:
            found |= _schema_vocabulary(item)
    return found


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def placement_violations(texts: dict[str, str]) -> list[str]:
    out = []
    named = {sid for sid, text in texts.items() if TOOL in text}
    if named != EXPECTED_SURFACES:
        out.append(
            f"{TOOL} is named in {sorted(named)}, expected {sorted(EXPECTED_SURFACES)}"
        )
    skill_named = {sid for sid, text in texts.items() if SKILL_NAME in text}
    if not skill_named <= SKILL_NAME_SURFACES:
        out.append(
            f"the {SKILL_NAME} skill is named in {sorted(skill_named - SKILL_NAME_SURFACES)}"
        )
    for sid in sorted((named | skill_named) - {SKILL, PLUGIN_README}):
        for chunk in blocks(texts[sid]):
            if f"{TOOL}(" not in chunk and SKILL_NAME not in chunk:
                continue
            if "optional" not in chunk.lower():
                out.append(
                    f"{sid}: a passage pointing to the tool does not say it is optional"
                )
            if _MANDATORY.search(chunk):
                out.append(f"{sid}: a passage pointing to the tool makes it mandatory")
    return out


def _bounds_violation(name: str, value: Any) -> str:
    if (
        name == "signal_ids"
        and isinstance(value, list)
        and not 1 <= len(value) <= MAX_CHANNELS
    ):
        return f"signal_ids has {len(value)} entries, outside 1 to {MAX_CHANNELS}"
    if name == "max_order" and not MIN_MAX_ORDER <= value <= MAX_MAX_ORDER:
        return f"max_order {value} is outside {MIN_MAX_ORDER} to {MAX_MAX_ORDER}"
    if name == "speed_tolerance_pct" and not 0 < value <= MAX_SPEED_TOLERANCE_PCT:
        return (
            f"speed_tolerance_pct {value} is outside (0, {MAX_SPEED_TOLERANCE_PCT:g}]"
        )
    if name in ("rpm", "line_frequency_hz", "machine_power_kw") and not value > 0:
        return f"{name} {value} is not positive"
    return ""


def call_violations(source: str, text: str) -> list[str]:
    """Literal values of every documented call: schema, then the tool's bounds."""
    params, _ = tool_schemas()
    out = []
    for name, args, line in extract_calls(text):
        if name != TOOL:
            continue
        literal: dict[str, Any] = {}
        for seg in split_top_level_args(args):
            key, sep, value = seg.partition("=")
            if not sep or value.strip() in ("...", "…"):
                continue
            try:
                literal[key.strip()] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                continue  # an f-string placeholder or expression, not a literal
        literal.setdefault("signal_ids", ["<id>"])
        try:
            jsonschema.validate(literal, params)
        except jsonschema.ValidationError as exc:
            out.append(
                f"{source}:{line}: {TOOL}({args}) violates the schema: {exc.message}"
            )
            continue
        for key, value in literal.items():
            problem = _bounds_violation(key, value)
            if problem:
                out.append(f"{source}:{line}: {TOOL}(...): {problem}")
    return out


#: Parameters whose value describes the machine or the record. A documented
#: call may name them only with a placeholder: the server resolves them from
#: declared context, or the model asks the user (signal_ids and detail are
#: not machine parameters).
MACHINE_PARAMETERS = frozenset(
    {
        "rpm",
        "directions",
        "machine_group",
        "support_type",
        "machine_power_kw",
        "line_frequency_hz",
        "speed_tolerance_pct",
        "max_order",
    }
)
_PLACEHOLDER = re.compile(r"\.\.\.|…|<[^<>]*>")


def hardcoded_parameter_violations(source: str, text: str) -> list[str]:
    """Documented calls pass no literal machine parameter."""
    out = []
    for name, args, line in extract_calls(text):
        if name != TOOL:
            continue
        for position, seg in enumerate(split_top_level_args(args)):
            key, sep, value = seg.partition("=")
            key, value = key.strip(), value.strip()
            if not sep and position > 0 and not _PLACEHOLDER.fullmatch(key):
                out.append(
                    f"{source}:{line}: {TOOL}(...) passes a positional value {key}"
                )
                continue
            if sep and key in MACHINE_PARAMETERS and not _PLACEHOLDER.fullmatch(value):
                out.append(
                    f"{source}:{line}: {TOOL}(...) hard-codes {key}={value}; pass it only"
                    f" when the user supplies it"
                )
    return out


#: Backticked tokens the guidance uses that are not result vocabulary.
_NOT_FIELDS = {TOOL, "diagnose_vibration", "generate_maintenance_recommendations"}


def field_violations(source: str, text: str) -> list[str]:
    """Backticked identifiers about the tool exist in the registered schemas.

    The whole skill is read; elsewhere only the paragraph blocks that show a call
    (a prohibition that merely names the tool is not a reading guide).
    """
    params, output = tool_schemas()
    vocabulary = _schema_vocabulary(output) | set(params.get("properties", {}))
    passages = (
        [text] if source == SKILL else [c for c in blocks(text) if f"{TOOL}(" in c]
    )
    out = []
    for passage in passages:
        for token in re.findall(r"`([a-z_][a-z0-9_]*)`", passage):
            if token not in vocabulary and token not in _NOT_FIELDS:
                out.append(
                    f"{source}: `{token}` is not a field, status or parameter of {TOOL}"
                )
    return out


def semantic_violations(source: str, text: str) -> list[str]:
    """Required statements present; no affirmative fault phrasing or corroboration."""
    normalized = flat(text).replace("`", "").lower()
    out = [
        f"{source}: missing required statement {phrase!r}"
        for phrase in REQUIRED.get(source, ())
        if phrase.replace("`", "").lower() not in normalized
    ]
    for sentence in sentences(text):
        for pattern in FORBIDDEN:
            for m in pattern.finditer(sentence):
                if not _negated_before(sentence, m.start()):
                    out.append(f"{source}: affirmative fault phrasing {m.group(0)!r}")
        if (
            _CORROBORATION.search(sentence)
            and _CORROBORATION_SCOPE.search(sentence)
            and not _NEGATION.search(sentence)
            and "corroborates nothing" not in sentence.lower()
        ):
            out.append(f"{source}: corroboration stated without negation: {sentence!r}")
    return out


# ---------------------------------------------------------------------------
# Tests on the real text
# ---------------------------------------------------------------------------


def test_the_tool_is_named_only_where_designed_and_optional_elsewhere():
    assert placement_violations(surfaces()) == []


@pytest.mark.parametrize("source", sorted(EXPECTED_SURFACES))
def test_documented_calls_validate_against_the_schema_and_bounds(source):
    assert call_violations(source, surfaces()[source]) == []


def test_every_calling_surface_shows_a_call_and_the_skill_names_every_parameter():
    for source in EXPECTED_SURFACES - {PLUGIN_README}:
        assert any(
            name == TOOL for name, _, _ in extract_calls(surfaces()[source])
        ), f"{source} shows no call"
    params, _ = tool_schemas()
    optional = set(params["properties"]) - {"signal_ids", "ctx"}
    named = set(re.findall(r"`([a-z_]+)`", surfaces()[SKILL]))
    assert optional <= named, f"the skill does not name {sorted(optional - named)}"


@pytest.mark.parametrize("source", sorted(EXPECTED_SURFACES))
def test_documented_calls_hard_code_no_machine_parameter(source):
    assert hardcoded_parameter_violations(source, surfaces()[source]) == []


@pytest.mark.parametrize("source", [SKILL, AGENT, PROMPT])
def test_the_guidance_names_only_real_result_fields(source):
    assert field_violations(source, surfaces()[source]) == []


@pytest.mark.parametrize("source", sorted(CHECKED_SURFACES))
def test_required_statements_and_no_affirmative_fault_phrasing(source):
    assert semantic_violations(source, surfaces()[source]) == []


# ---------------------------------------------------------------------------
# The guards bite
# ---------------------------------------------------------------------------


class Mutation(NamedTuple):
    case: str
    source: str
    guard: Callable[[str, str], list[str]]
    mutate: Callable[[str], str]


def _swap(old: str, new: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert old in text, f"mutation target {old!r} is not in the surface"
        return text.replace(old, new, 1)

    return mutate


def _append(extra: str) -> Callable[[str], str]:
    return lambda text: text + extra


_AFFIRMATIVE = (
    "A strong 1X confirms unbalance.",
    "The pattern suggests misalignment.",
    "So unbalance is likely.",
    "The rotor is unbalanced.",
    "This confirms the pattern.",
    "No sidebands appear, so looseness is present.",
    "A 2X line points to misalignment.",
    "Possible unbalance; check the balance.",
    "The axial line is evidence of misalignment.",
    "A half order is a sign of looseness.",
    "The shaft is misaligned.",
    "Unbalance is the most probable cause.",
    "This confirms the unbalance hypothesis.",
    "The 1X line points to the misalignment hypothesis.",
    "The two results reinforce each other.",
    "This assessment and the FFT reinforce each other.",
    "The motor has loose hold-down bolts.",
)

MUTATIONS = (
    Mutation(
        "drop: the skill's non-attribution statement",
        SKILL,
        semantic_violations,
        _swap("It attributes no fault:", "It reports:"),
    ),
    *(
        Mutation(
            f"add: {sentence}", SKILL, semantic_violations, _append(f"\n\n{sentence}\n")
        )
        for sentence in _AFFIRMATIVE
    ),
    Mutation(
        "add: a 'consistent' status",
        SKILL,
        semantic_violations,
        _append("\n\nWhen the status is `consistent`, schedule balancing.\n"),
    ),
    Mutation(
        "weaken: always include becomes may include",
        SKILL,
        semantic_violations,
        _swap("always include", "may include"),
    ),
    Mutation(
        "add: agreement presented as corroboration",
        AGENT,
        semantic_violations,
        _append("\n\nAgreement with diagnose_vibration corroborates the pattern.\n"),
    ),
    *(
        Mutation(
            f"add: {sentence}", AGENT, semantic_violations, _append(f"\n\n{sentence}\n")
        )
        for sentence in (
            "Its result and diagnose_vibration reinforce each other.",
            "diagnose_vibration gives independent evidence for the pattern.",
            "Agreement with diagnose_vibration strengthens the case.",
            "The base is loose.",
        )
    ),
    Mutation(
        "drop: the agent's not_observed rule",
        AGENT,
        semantic_violations,
        _swap("does not exclude the fault", "excludes the fault"),
    ),
    Mutation(
        "add: an unknown detail value",
        SKILL,
        call_violations,
        _swap('detail="full"', 'detail="brief"'),
    ),
    Mutation(
        "add: a direction outside the vocabulary",
        AGENT,
        call_violations,
        _append(
            f'\n\nThen `{TOOL}(signal_ids=["<id>"], directions=["tangential"])`.\n'
        ),
    ),
    Mutation(
        "add: a max_order above the tool's bound",
        SKILL,
        call_violations,
        _swap('detail="full")', 'detail="full", max_order=25)'),
    ),
    Mutation(
        "add: a speed tolerance above the tool's bound",
        SKILL,
        call_violations,
        _swap('detail="full")', 'detail="full", speed_tolerance_pct=20)'),
    ),
    Mutation(
        "add: the agent call hard-codes a speed and a machine class",
        AGENT,
        hardcoded_parameter_violations,
        _swap(
            'signal_ids=["<h>", "<v>", "<a>"])',
            'signal_ids=["<h>", "<v>", "<a>"], rpm=1797, machine_group=2)',
        ),
    ),
    Mutation(
        "add: the skill call hard-codes directions",
        SKILL,
        hardcoded_parameter_violations,
        _swap(
            'signal_ids=["<h>", "<v>", "<a>"])',
            'signal_ids=["<h>", "<v>", "<a>"], directions=["horizontal", "vertical", "axial"])',
        ),
    ),
    Mutation(
        "add: the prompt call hard-codes a speed",
        PROMPT,
        hardcoded_parameter_violations,
        lambda text: re.sub(
            rf"({TOOL}\(signal_ids=\[[^\]]*\])\)", r"\1, rpm=1480)", text, count=1
        ),
    ),
    Mutation(
        "add: a field the result does not have",
        SKILL,
        field_violations,
        _append("\nRead `fault_probability` first.\n"),
    ),
    Mutation(
        "add: an agent passage naming a field the result does not have",
        AGENT,
        field_violations,
        _swap("relay `limitations`", "relay `limitation_text`"),
    ),
    Mutation(
        "drop: the screening skill's optional framing",
        SCREENING,
        semantic_violations,
        _swap(
            "optional and not part of the screening card.",
            "part of the screening card.",
        ),
    ),
)


@pytest.mark.parametrize("m", MUTATIONS, ids=[m.case for m in MUTATIONS])
def test_guards_fail_on_mutations(m):
    text = surfaces()[m.source]
    assert m.guard(m.source, text) == [], "the guard is not green on the real text"
    mutated = m.mutate(text)
    assert mutated != text
    assert m.guard(m.source, mutated), f"{m.case}: the guard stayed green"


def test_placement_guard_fails_when_a_bearing_workflow_requires_the_tool():
    texts = dict(surfaces())
    bearing = "plugin/skills/bearing-diagnosis/SKILL.md"
    texts[bearing] = texts[bearing] + f'\n\nAlways call {TOOL}(signal_ids=["<id>"]).\n'
    assert placement_violations(texts)


def test_placement_guard_fails_when_the_agent_makes_the_step_mandatory():
    texts = dict(surfaces())
    mutated, n = re.subn(
        r"\(optional;\s+a\s+bearing\s+or\s+gear\s+check\s+never\s+needs\s+it\)",
        "(a required step)",
        texts[AGENT],
    )
    assert n == 1
    texts[AGENT] = mutated
    assert placement_violations(texts)


_OPTIONAL_LEAD = r"\(optional;\s+a\s+bearing\s+or\s+gear\s+check\s+never\s+needs\s+it\)"


@pytest.mark.parametrize(
    "lead",
    [
        "(optional; run this step after every bearing check)",
        "(optional; run it every time a bearing check runs)",
        "(optional; after each bearing check)",
        "(optional; required for every report)",
        "(optional; on every signal)",
        "(optional; always run it)",
        "(optional; include it in every report)",
        "(optional; run it for all signals)",
        "(optional; this step is required)",
        "(optional; never skip it)",
    ],
)
def test_placement_guard_fails_when_the_step_recurs(lead):
    texts = dict(surfaces())
    mutated, n = re.subn(_OPTIONAL_LEAD, lead, texts[AGENT])
    assert n == 1
    texts[AGENT] = mutated
    assert placement_violations(texts)


@pytest.mark.parametrize(
    "lead",
    [
        "(optional; always quote summary verbatim)",
        "(optional; one direction for each channel)",
        "(optional; always check the unit first)",
        "(optional; relay limitations for every observed pattern)",
    ],
)
def test_an_instruction_is_not_a_mandatory_step(lead):
    texts = dict(surfaces())
    mutated, n = re.subn(_OPTIONAL_LEAD, lead, texts[AGENT])
    assert n == 1
    texts[AGENT] = mutated
    assert placement_violations(texts) == []


def test_a_required_unit_is_not_a_mandatory_step():
    texts = dict(surfaces())
    mutated, n = re.subn(
        r"never\s+estimate\s+it",
        "never estimate it (a declared unit is required)",
        texts[AGENT],
    )
    assert n == 1
    texts[AGENT] = mutated
    assert placement_violations(texts) == []


def test_placement_guard_fails_when_the_skill_is_pointed_to_from_elsewhere():
    texts = dict(surfaces())
    gear = "plugin/skills/gear-diagnosis/SKILL.md"
    texts[gear] = texts[gear] + f"\n\nThen run the {SKILL_NAME} skill.\n"
    assert placement_violations(texts)


def test_a_prohibition_that_names_the_tool_needs_no_optional_framing():
    """Naming the tool in a prohibition (no call) is not a usage pointer."""
    texts = dict(surfaces())
    texts[AGENT] = texts[AGENT] + (
        f"\n\n- Do not pass a status of {TOOL} to generate_maintenance_recommendations.\n"
    )
    assert placement_violations(texts) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "A loose sensor mount can produce half-order lines.",
        "The table shows the unbalance hypothesis and its reasons.",
        "Faults are established by independent evidence, such as an inspection.",
        "Check for misaligned channel directions before the call.",
    ],
)
def test_benign_sentences_pass_the_wording_guard(sentence):
    text = surfaces()[AGENT] + f"\n\n{sentence}\n"
    assert semantic_violations(AGENT, text) == []


def test_a_placeholder_is_not_a_hard_coded_parameter():
    text = surfaces()[AGENT] + (
        f'\n\n`{TOOL}(signal_ids=["<h>"], rpm=<the speed the user gives>)`\n'
    )
    assert hardcoded_parameter_violations(AGENT, text) == []


def test_a_positional_machine_value_is_hard_coded():
    text = surfaces()[AGENT] + f'\n\n`{TOOL}(["<h>"], 1480)`\n'
    assert hardcoded_parameter_violations(AGENT, text)


@pytest.mark.parametrize(
    "sentence",
    [
        "Unbalance can produce a dominant 1x radial line, and so can a bent shaft"
        " or a resonance.",
        "A manual may describe how misalignment shows in vibration; quote it as"
        " background.",
        "A phase measurement could help distinguish unbalance from misalignment.",
    ],
)
def test_background_on_the_faults_passes_the_wording_guard(sentence):
    for source in (SKILL, AGENT):
        text = surfaces()[source] + f"\n\n{sentence}\n"
        assert semantic_violations(source, text) == []
