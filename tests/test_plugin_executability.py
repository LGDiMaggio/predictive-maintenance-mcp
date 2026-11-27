"""The plugin's agents, commands and skills run as the user's settings allow.

- Every agent runs on the session's model (``model: inherit``).
- No command and no skill pre-approves a tool (``allowed-tools`` absent), so
  every call follows the user's permission settings.
- No agent allowlist and no pre-approval names a tool that needs the user's
  own act: declaring a baseline or a measurement point, importing a signal
  file, clearing signals.
- The plugin README states each agent's number of server tools as its
  ``tools`` field lists them.
- The deprecated signal-explorer agent only points to the report skill.
- Each command hands over to exactly one skill of this plugin and makes no
  call of its own.
- The pipeline agent names a difference between two server results instead
  of reconciling it.

Every guard is a pure ``(source, text) -> violations`` function, run on the
real files and on mutated copies.
"""

from __future__ import annotations

import re

import pytest

from _semantic_guards import PLUGIN_DIR, extract_calls, flat, plugin_markdown
from test_plugin_agent_tools import INVENTORY, PREFIX, _split, tool_violations

USER_ACT_TOOLS = (
    "declare_healthy_baseline",
    "declare_measurement_point",
    "import_signal_file",
    "clear_signals",
)
STUB = "agents/signal-explorer.md"
PIPELINE = "agents/diagnostic-pipeline.md"
THIN = "This command adds no step, threshold, verdict or recommendation of its own."
RELAY_CLAUSES = (
    "say that the server results differ",
    "name any input that differed between the calls",
    "quote both",
    "do not reconcile them or write a verdict of your own",
)
#: The whole rule, so a clause kept but negated ("never say ...") is caught.
RELAY_RULE = (
    "catalog `bearing_id`), say that the server results differ, name any input "
    "that differed between the calls, quote both, and do not reconcile them or "
    "write a verdict of your own"
)


def _front(text: str) -> tuple[dict, str]:
    front, body = _split(text)
    return front or {}, body


def _kind(source: str) -> str:
    return source.split("/", 1)[0]


def executability_violations(source: str, text: str) -> list[str]:
    """A fixed agent model, a pre-approval, or a granted user-act tool."""
    front, _ = _front(text)
    out = []
    if _kind(source) == "agents" and front.get("model") != "inherit":
        out.append(f"{source}: model is {front.get('model')!r}, not inherit")
    if _kind(source) in ("commands", "skills") and "allowed-tools" in front:
        out.append(f"{source}: pre-approves {front['allowed-tools']!r}")
    granted = list(front.get("tools") or []) + list(front.get("allowed-tools") or [])
    for tool in USER_ACT_TOOLS:
        if any(tool in str(entry) for entry in granted):
            out.append(f"{source}: grants {tool}, which needs the user's own act")
    return out


def _server_tool_count(text: str) -> int:
    return sum(
        1 for e in _front(text)[0].get("tools") or [] if str(e).startswith(PREFIX)
    )


def readme_count_violations(readme: str, agents: dict[str, str]) -> list[str]:
    """Agents whose server-tool count the README states wrongly or not at all."""
    out = []
    readme = flat(readme)
    for source, text in agents.items():
        name = source.removeprefix("agents/").removesuffix(".md")
        listed = _server_tool_count(text)
        m = re.search(
            rf"\((?:.*?, )?(\d+|none) for (?:the deprecated )?{name}\b", readme
        )
        stated = None if not m else (0 if m.group(1) == "none" else int(m.group(1)))
        if stated != listed:
            out.append(
                f"README states {stated} server tools for {name}; its file lists {listed}"
            )
    return out


def stub_violations(text: str) -> list[str]:
    """Ways the deprecated agent does more than point to the report skill."""
    front, body = _front(text)
    out = []
    if "Deprecated" not in str(front.get("description")):
        out.append("the stub's description does not say it is deprecated")
    if front.get("tools") != ["Read"]:
        out.append(f"the stub lists {front.get('tools')!r}, not only Read")
    if [name for name, _, _ in extract_calls(body) if name in INVENTORY]:
        out.append("the stub calls a server tool")
    if "predictive-maintenance:report-generation" not in body:
        out.append("the stub does not point to the report-generation skill")
    return out


def command_violations(source: str, text: str) -> list[str]:
    """Ways a command does more than hand over to one skill of this plugin."""
    _, body = _front(text)
    out = []
    skills = set(re.findall(r"predictive-maintenance:([a-z-]+)", body))
    if len(skills) != 1:
        out.append(f"{source}: names {sorted(skills)}, not exactly one skill")
    elif not (PLUGIN_DIR / "skills" / next(iter(skills)) / "SKILL.md").exists():
        out.append(f"{source}: names a skill this plugin does not have")
    calls = [name for name, _, _ in extract_calls(body) if name in INVENTORY]
    if calls:
        out.append(f"{source}: calls {calls} itself")
    if flat(THIN) not in flat(body):
        out.append(f"{source}: does not say it adds nothing of its own")
    return out


def relay_violations(text: str) -> list[str]:
    """Clauses of the rule for two differing server results the text lacks."""
    missing = [
        f"the pipeline lacks {c!r}" for c in RELAY_CLAUSES if flat(c) not in flat(text)
    ]
    if not missing and flat(RELAY_RULE) not in flat(text):
        missing.append("the pipeline's relay rule is altered")
    return missing


PLUGIN_TEXT = plugin_markdown()


def _plugin() -> dict[str, str]:
    return PLUGIN_TEXT


def _sources(kind: str) -> list[str]:
    return [s for s in PLUGIN_TEXT if _kind(s) == kind]


EXECUTABLE = (
    _sources("agents")
    + _sources("commands")
    + [s for s in _sources("skills") if s.endswith("SKILL.md")]
)


@pytest.mark.parametrize("source", EXECUTABLE)
def test_every_agent_command_and_skill_runs_under_the_users_settings(source):
    assert executability_violations(source, _plugin()[source]) == []


def test_the_readme_states_each_agents_server_tool_count():
    agents = {s: _plugin()[s] for s in _sources("agents")}
    assert readme_count_violations(_plugin()["README.md"], agents) == []


def test_the_deprecated_agent_only_points_to_the_report_skill():
    assert stub_violations(_plugin()[STUB]) == []


@pytest.mark.parametrize("source", _sources("commands"))
def test_every_command_hands_over_to_one_skill(source):
    assert command_violations(source, _plugin()[source]) == []


def test_the_pipeline_names_a_difference_between_server_results():
    assert relay_violations(_plugin()[PIPELINE]) == []


# Mutated copies: each guard reports what it exists to catch.


def _set_front(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def test_a_fixed_model_is_reported():
    text = _set_front(_plugin()[PIPELINE], "model: inherit", "model: sonnet")
    assert executability_violations(PIPELINE, text) == [
        f"{PIPELINE}: model is 'sonnet', not inherit"
    ]


@pytest.mark.parametrize(
    "source", ["commands/pm-screen.md", "skills/quick-screening/SKILL.md"]
)
def test_a_pre_approval_is_reported(source):
    text = (
        _plugin()[source]
        .replace("\r\n", "\n")
        .replace("---\n", "---\nallowed-tools:\n  - Bash\n", 1)
    )
    assert executability_violations(source, text) == [
        f"{source}: pre-approves ['Bash']"
    ]


def test_a_user_act_tool_in_an_allowlist_is_reported_although_the_tool_guard_passes():
    tool = f"{PREFIX}declare_healthy_baseline"
    text = (
        _plugin()[PIPELINE]
        .replace(
            f"  - {PREFIX}diagnose_vibration\n",
            f"  - {PREFIX}declare_healthy_baseline\n  - {PREFIX}diagnose_vibration\n",
            1,
        )
        .replace("\r\n", "\n")
    )
    text += '\nCall `declare_healthy_baseline(point_id="<point>")`.\n'
    assert tool in text
    assert tool_violations(text) == []
    assert executability_violations(PIPELINE, text) == [
        f"{PIPELINE}: grants declare_healthy_baseline, which needs the user's own act"
    ]


def test_a_wrong_readme_count_is_reported():
    listed = _server_tool_count(_plugin()[PIPELINE])
    wrong = listed - 2
    readme = _plugin()["README.md"]
    assert f"({listed} for diagnostic-pipeline" in readme
    readme = readme.replace(
        f"({listed} for diagnostic-pipeline", f"({wrong} for diagnostic-pipeline"
    )
    agents = {s: _plugin()[s] for s in _sources("agents")}
    assert readme_count_violations(readme, agents) == [
        f"README states {wrong} server tools for diagnostic-pipeline; "
        f"its file lists {listed}"
    ]


def test_a_stub_that_runs_a_tool_is_reported():
    text = _plugin()[STUB] + '\nCall `analyze_fft(signal_id="<id>")`.\n'
    assert stub_violations(text) == ["the stub calls a server tool"]


def test_a_command_with_its_own_call_or_second_skill_is_reported():
    source = "commands/pm-screen.md"
    text = _plugin()[source] + (
        "\nAlso use `predictive-maintenance:bearing-diagnosis`.\n"
        'Call `analyze_fft(signal_id="<id>")`.\n'
    )
    assert command_violations(source, text) == [
        f"{source}: names ['bearing-diagnosis', 'quick-screening'], not exactly one skill",
        f"{source}: calls ['analyze_fft'] itself",
    ]


@pytest.mark.parametrize("clause", RELAY_CLAUSES)
def test_a_lost_relay_clause_is_reported(clause):
    text = flat(_plugin()[PIPELINE]).replace(flat(clause), "", 1)
    assert relay_violations(text) == [f"the pipeline lacks {clause!r}"]


def test_a_negated_relay_clause_is_reported():
    text = flat(_plugin()[PIPELINE]).replace(
        "say that the server results differ",
        "never say that the server results differ",
        1,
    )
    assert relay_violations(text) == ["the pipeline's relay rule is altered"]


REPORT = "skills/report-generation/SKILL.md"
PRECLAIM_CUE = (
    "without a `bearing_check` field comes from a server that predates the "
    "bearing claim"
)
PRECLAIM_WITHHELD = (
    "its `verdict`, its bearing recommendations or its evidence strength"
)


def preclaim_fallback_violations(source: str, text: str) -> list[str]:
    """The report fallback is missing or keyed on something the route lacks."""
    body = flat(text)
    if source == REPORT:
        body = body.split("Step 5:", 1)[-1].split("## Multi-Signal Comparison", 1)[0]
    return [
        f"{source}: lacks {phrase!r}"
        for phrase in (PRECLAIM_CUE, PRECLAIM_WITHHELD)
        if flat(phrase) not in body
    ]


@pytest.mark.parametrize("source", [REPORT, PIPELINE])
def test_the_report_fallback_is_keyed_on_the_report_result(source):
    assert preclaim_fallback_violations(source, _plugin()[source]) == []


@pytest.mark.parametrize("source", [REPORT, PIPELINE])
def test_a_fallback_keyed_on_the_check_is_reported(source):
    text = flat(_plugin()[source]).replace(
        flat("without a `bearing_check` field"),
        "with a check that returns no claim",
        1,
    )
    assert preclaim_fallback_violations(source, text) == [
        f"{source}: lacks {PRECLAIM_CUE!r}"
    ]
