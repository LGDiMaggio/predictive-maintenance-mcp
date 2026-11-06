"""The plugin agents can call the server tools their workflows use, and no others.

A plugin agent's ``tools`` frontmatter is an allowlist: a tool it does not name
is not available to the agent (Claude Code subagents documentation). Each agent
keeps Read, Glob, Grep and Bash and lists, by name for a server registered as
``predictive-maintenance``, exactly the server tools its own instructions call:
no wildcard, no other built-in tool, no other server, no unused server tool.
"""

import json
import re
from typing import Any, Callable

import pytest
import yaml

from test_documented_calls import REPO_ROOT, extract_calls

SERVER = "predictive-maintenance"
PREFIX = f"mcp__{SERVER}__"
BUILT_INS = ["Read", "Glob", "Grep", "Bash"]
AGENTS = ("plugin/agents/diagnostic-pipeline.md", "plugin/agents/signal-explorer.md")
INVENTORY = frozenset(
    json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "tool_inventory.json").read_text(
            encoding="utf-8"
        )
    )["tools"]
)


def _split(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"---\n(.*?)\n---\n(.*)", text.replace("\r\n", "\n"), re.S)
    assert match, "agent file has no YAML frontmatter"
    return yaml.safe_load(match.group(1)), match.group(2)


def tool_violations(text: str) -> list[str]:
    front, body = _split(text)
    listed = front.get("tools") or []
    out = []
    if listed[: len(BUILT_INS)] != BUILT_INS:
        out.append(f"the built-in tools are not {BUILT_INS}: {listed[:4]}")
    for entry in listed[len(BUILT_INS) :]:
        if not isinstance(entry, str) or not entry.startswith(PREFIX):
            out.append(f"{entry!r} is not a tool of the {SERVER} server")
        elif "*" in entry:
            out.append(f"{entry!r} is a wildcard")
    granted = {
        e[len(PREFIX) :]
        for e in listed
        if isinstance(e, str) and e.startswith(PREFIX) and "*" not in e
    }
    called = {name for name, _, _ in extract_calls(body) if name in INVENTORY}
    out += [
        f"{name} is listed but is not a registered tool"
        for name in sorted(granted - INVENTORY)
    ]
    out += [f"{name} is called but not listed" for name in sorted(called - granted)]
    out += [f"{name} is listed but never called" for name in sorted(granted - called)]
    return out


def _text(agent: str) -> str:
    return (REPO_ROOT / agent).read_text(encoding="utf-8")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_agent_lists_exactly_the_server_tools_it_calls(agent):
    assert tool_violations(_text(agent)) == []


@pytest.mark.parametrize("agent", AGENTS)
def test_the_agent_says_how_it_reaches_the_server_and_never_works_around_it(agent):
    flat = re.sub(r"\s+", " ", _text(agent)).replace("`", "").lower()
    assert "registered as predictive-maintenance" in flat
    assert "tools are not reachable from this agent and stop" in flat
    assert "never run the analysis through bash or simulate a tool result" in flat
    assert "use bash only to inspect files the user points to" in flat
    assert (
        "change the asset ledger, models or signal files, or install packages" in flat
    )


def test_the_plugin_readme_names_the_server_and_the_limit():
    flat = re.sub(
        r"\s+", " ", (REPO_ROOT / "plugin" / "README.md").read_text(encoding="utf-8")
    )
    assert "registered under the name `predictive-maintenance`" in flat
    assert "Under any other name the agents cannot call the server's tools" in flat
    assert "Claude Code asks before each tool call" in flat
    assert "append-only asset ledger" in flat
    assert "instructions limit Bash to inspecting files" in flat


def _add(line: str) -> Callable[[str], str]:
    return lambda text: text.replace("  - Bash\n", f"  - Bash\n  - {line}\n", 1)


def _drop(line: str) -> Callable[[str], str]:
    def mutate(text: str) -> str:
        assert f"  - {line}\n" in text
        return text.replace(f"  - {line}\n", "", 1)

    return mutate


MUTATIONS = {
    "add a built-in tool (Write)": _add("Write"),
    "add the server wildcard": _add(f"{PREFIX}*"),
    "add an unused server tool (clear_signals)": _add(f"{PREFIX}clear_signals"),
    "add a tool of another server": _add("mcp__vibration-analysis__load_signal"),
    "add a tool the server does not register": _add(f"{PREFIX}delete_everything"),
    "drop a called tool (list_signals)": _drop(f"{PREFIX}list_signals"),
    "drop Bash": _drop("Bash"),
}


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("case", sorted(MUTATIONS))
def test_the_permission_guard_fails_on_mutations(agent, case):
    text = _text(agent)
    mutated = MUTATIONS[case](text)
    assert mutated != text
    assert tool_violations(mutated), f"{case}: the guard stayed green"


def test_signal_explorer_does_not_train_a_model_itself():
    flat = re.sub(r"\s+", " ", _text("plugin/agents/signal-explorer.md")).lower()
    assert "do not train one yourself" in flat
    assert "train one first" not in flat
