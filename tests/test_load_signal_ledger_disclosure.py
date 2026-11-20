"""load_signal says, before its arguments, that a load also records in the ledger.

A file whose companion declares a "measurement" object is also recorded in the
local append-only asset ledger when ``load_signal`` loads it. The registered
tool description states this right after its paragraph on signal_id, before
"Args:", and the server instructions state it inside the window a client was
observed to show.
Both quote the ledger statuses the tool returns; the statuses here are driven
through the tool itself, on synthetic signals, and each load succeeds.
"""

import re
from typing import Callable
from unittest.mock import AsyncMock

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.asset_ledger import service as ledger_service
from predictive_maintenance_mcp.mcp_tools import acquisition_tools
from predictive_maintenance_mcp.models import StoredSignalInfo
from predictive_maintenance_mcp.signal_acquisition import repository as repo_module
from test_acquisition_tools import (
    _companion,
    _identity,
    _noise,
    _place,
    _write_companion,
)
from test_model_facing_guidance import _registered, full_instructions, passage_end

LEDGER_STATUSES = frozenset(ledger_service.LEDGER_STATUSES)
DESCRIPTION_PARAGRAPH = "Ledger side effect:"
HANDLE_PARAGRAPH = "Once loaded, reference the signal by its signal_id"
INSTRUCTION_BULLET = "- load_signal also attempts"
_STATUS_RE = re.compile(r"'([a-z]+(?:_[a-z]+)*)'")


def load_signal_description() -> str:
    """The load_signal description as the registered server serves it."""
    return _registered()[0]["load_signal"]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n[ \t]*\n", text.strip())]


def description_passage(description: str) -> str:
    """The paragraph of the description that discloses the ledger record."""
    for paragraph in _paragraphs(description):
        if paragraph.startswith(DESCRIPTION_PARAGRAPH):
            return paragraph
    return ""


def instruction_passage(instructions: str) -> str:
    """The instructions bullet on the ledger record of load_signal."""
    start = instructions.find(INSTRUCTION_BULLET)
    if start < 0:
        return ""
    return instructions[start : passage_end(instructions, start)]


def model_passage() -> str:
    """The ledger_status values named by the returned measurement block."""
    description = StoredSignalInfo.model_fields["measurement"].description or ""
    m = re.search(r"ledger_status \([^)]*\)", description)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Guards: pure functions, run on the real texts and on mutated copies
# ---------------------------------------------------------------------------


def disclosure_order_violations(description: str) -> list[str]:
    """The ledger paragraph directly follows the paragraph on signal_id, before
    "Args:"."""
    start = description.find(DESCRIPTION_PARAGRAPH)
    if start < 0:
        return [f"the description has no {DESCRIPTION_PARAGRAPH!r} paragraph"]
    violations = []
    args = description.find("Args:")
    if args < 0 or start > args:
        violations.append(f"{DESCRIPTION_PARAGRAPH!r} comes after 'Args:'")
    paragraphs = _paragraphs(description)
    handle = [i for i, p in enumerate(paragraphs) if p.startswith(HANDLE_PARAGRAPH)]
    following = paragraphs[handle[0] + 1 : handle[0] + 2] if len(handle) == 1 else []
    if not following or not following[0].startswith(DESCRIPTION_PARAGRAPH):
        violations.append(
            f"{DESCRIPTION_PARAGRAPH!r} does not directly follow {HANDLE_PARAGRAPH!r}"
        )
    return violations


def status_vocabulary_violations(where: str, passage: str) -> list[str]:
    """The statuses listed after ledger_status are the four the tool returns,
    each once, and the passage quotes no other status."""
    if not passage:
        return [f"{where}: the ledger passage is gone"]
    start = passage.find("ledger_status")
    if start < 0:
        return [f"{where}: the passage names no ledger_status"]
    end = passage.find(". ", start)
    listed = _STATUS_RE.findall(passage[start : end if end >= 0 else len(passage)])
    violations = []
    if set(listed) != LEDGER_STATUSES or len(listed) != len(LEDGER_STATUSES):
        violations.append(
            f"{where}: lists {listed}, not each of {sorted(LEDGER_STATUSES)} once"
        )
    extra = set(_STATUS_RE.findall(passage)) - LEDGER_STATUSES
    if extra:
        violations.append(f"{where}: quotes {sorted(extra)}, not ledger statuses")
    return violations


#: The clauses of the ledger paragraph that keep the load apart from the record.
LOAD_AND_RECORD_CLAUSES = (
    "Loading and recording succeed separately",
    "a ledger problem never fails the load",
)


def load_and_record_violations(passage: str) -> list[str]:
    return [
        f"the ledger paragraph no longer states {clause!r}"
        for clause in LOAD_AND_RECORD_CLAUSES
        if clause not in passage
    ]


def _sources() -> dict[str, str]:
    """Each passage with its whitespace collapsed, so wrapping is not drift."""
    passages = {
        "instructions": instruction_passage(full_instructions()),
        "load_signal description": description_passage(load_signal_description()),
        "StoredSignalInfo.measurement": model_passage(),
    }
    return {where: " ".join(text.split()) for where, text in passages.items()}


def test_the_ledger_paragraph_precedes_the_arguments():
    assert disclosure_order_violations(load_signal_description()) == []


@pytest.mark.parametrize(
    "where", ["instructions", "load_signal description", "StoredSignalInfo.measurement"]
)
def test_the_statuses_named_are_the_runtime_statuses(where):
    assert status_vocabulary_violations(where, _sources()[where]) == []


def test_the_ledger_paragraph_keeps_the_load_apart_from_the_record():
    passage = _sources()["load_signal description"]
    assert load_and_record_violations(passage) == []


@pytest.mark.parametrize(
    "old, new, clause",
    [
        (
            "Loading and recording succeed separately: the",
            "The",
            "Loading and recording succeed separately",
        ),
        (
            "a ledger problem never fails the load",
            "a ledger problem can fail the load",
            "a ledger problem never fails the load",
        ),
    ],
)
def test_a_lost_load_and_record_clause_is_named(old, new, clause):
    passage = _sources()["load_signal description"]
    assert old in passage
    assert load_and_record_violations(passage.replace(old, new, 1)) == [
        f"the ledger paragraph no longer states {clause!r}"
    ]


def _paragraph(description: str, start: str) -> str:
    """The paragraph that starts with ``start``, as indented in the description."""
    found = [p for p in _paragraphs(description) if p.startswith(start)]
    assert len(found) == 1, f"no single paragraph starts with {start!r}"
    return "    " + found[0] + "\n\n"


def _move_before(starts: tuple[str, ...], anchor: str) -> Callable[[str], str]:
    """Move the paragraphs that start with ``starts``, in that order, to just
    before the paragraph that starts with ``anchor``."""

    def mutate(description: str) -> str:
        moved = [_paragraph(description, start) for start in starts]
        for paragraph in moved:
            description = description.replace(paragraph, "", 1)
        target = "    " + anchor
        assert description.count(target) == 1, f"{anchor!r} is not a single anchor"
        return description.replace(target, "".join(moved) + target, 1)

    return mutate


def _delete_ledger_paragraph(description: str) -> str:
    return description.replace(_paragraph(description, DESCRIPTION_PARAGRAPH), "", 1)


AFTER_ARGS = f"{DESCRIPTION_PARAGRAPH!r} comes after 'Args:'"
NOT_NEXT = f"{DESCRIPTION_PARAGRAPH!r} does not directly follow {HANDLE_PARAGRAPH!r}"
ORDER_MUTATIONS = {
    "deleted": (
        _delete_ledger_paragraph,
        [f"the description has no {DESCRIPTION_PARAGRAPH!r} paragraph"],
    ),
    "moved after Args": (
        _move_before((DESCRIPTION_PARAGRAPH,), "Asset ledger:"),
        [AFTER_ARGS, NOT_NEXT],
    ),
    "moved below another paragraph, still before Args": (
        _move_before((DESCRIPTION_PARAGRAPH,), "signal_id default:"),
        [NOT_NEXT],
    ),
    "moved after Args with the paragraph it follows": (
        _move_before((HANDLE_PARAGRAPH, DESCRIPTION_PARAGRAPH), "Asset ledger:"),
        [AFTER_ARGS],
    ),
}


@pytest.mark.parametrize("case", sorted(ORDER_MUTATIONS))
def test_the_order_guard_names_a_misplaced_ledger_paragraph(case):
    description = load_signal_description()
    assert disclosure_order_violations(description) == []
    mutate, expected = ORDER_MUTATIONS[case]
    assert disclosure_order_violations(mutate(description)) == expected


VOCABULARY_MUTATIONS = {
    "instructions drop superseded": (
        "instructions",
        "'superseded' (a corrected declaration was added), ",
        "",
    ),
    "instructions add a fifth status": (
        "instructions",
        " or 'not_recorded'",
        ", 'pending' (queued) or 'not_recorded'",
    ),
    "description drop already_recorded": (
        "load_signal description",
        "'already_recorded', ",
        "",
    ),
    "description add a fifth status": (
        "load_signal description",
        "'already_recorded', ",
        "'already_recorded', 'pending', ",
    ),
    "description quote a fifth status after the list": (
        "load_signal description",
        "never fails the load.",
        "never fails the load ('pending').",
    ),
}


@pytest.mark.parametrize("case", sorted(VOCABULARY_MUTATIONS))
def test_the_vocabulary_guard_fails_on_a_mutated_passage(case):
    where, old, new = VOCABULARY_MUTATIONS[case]
    passage = _sources()[where]
    assert status_vocabulary_violations(where, passage) == []
    assert old in passage, f"{case}: {old!r} is not in the {where} passage"
    mutated = passage.replace(old, new, 1)
    assert status_vocabulary_violations(where, mutated), f"{case}: stayed green"


# ---------------------------------------------------------------------------
# The four statuses, driven through load_signal on synthetic signals
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
def tools():
    server = MCPServer("test-ledger-disclosure")
    acquisition_tools.register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


@pytest.fixture
def clean_repo():
    repo = repo_module.get_repository()
    repo.clear_all()
    yield repo
    repo.clear_all()


async def _recorded(tools, ctx, data_dir, tmp_path, monkeypatch):
    _place(data_dir, "m01.csv", _noise(1), _companion(_identity()))
    return await tools["load_signal"](ctx=ctx, filepath="m01.csv")


async def _already_recorded(tools, ctx, data_dir, tmp_path, monkeypatch):
    _place(data_dir, "m01.csv", _noise(1), _companion(_identity()))
    await tools["load_signal"](ctx=ctx, filepath="m01.csv")
    return await tools["load_signal"](ctx=ctx, filepath="m01.csv", overwrite=True)


async def _superseded(tools, ctx, data_dir, tmp_path, monkeypatch):
    path = _place(data_dir, "m01.csv", _noise(1), _companion(_identity(rpm=1800)))
    await tools["load_signal"](ctx=ctx, filepath="m01.csv")
    _write_companion(path, _companion(_identity(rpm=1500)))
    return await tools["load_signal"](ctx=ctx, filepath="m01.csv", overwrite=True)


async def _not_recorded(tools, ctx, data_dir, tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("PMM_LEDGER_DIR", str(blocker))
    _place(data_dir, "m01.csv", _noise(1), _companion(_identity()))
    return await tools["load_signal"](ctx=ctx, filepath="m01.csv")


SCENARIOS = {
    "recorded": _recorded,
    "already_recorded": _already_recorded,
    "superseded": _superseded,
    "not_recorded": _not_recorded,
}


def test_the_scenarios_cover_every_runtime_status():
    assert set(SCENARIOS) == LEDGER_STATUSES


@pytest.mark.asyncio
@pytest.mark.parametrize("status", sorted(SCENARIOS))
async def test_load_signal_returns_each_status_and_keeps_the_signal(
    status, tools, mock_ctx, sandbox_data_dir, tmp_path, monkeypatch, clean_repo
):
    result = await SCENARIOS[status](
        tools, mock_ctx, sandbox_data_dir, tmp_path, monkeypatch
    )
    assert result.signal_id == "m01"
    assert result.measurement["ledger_status"] == status
    if status == "not_recorded":
        assert "PMM_LEDGER_DIR" in result.measurement["reason"]
    stored = await tools["get_signal_info"](ctx=mock_ctx, signal_id="m01")
    assert stored.signal_id == "m01"
    assert stored.measurement["measurement_id"] == result.measurement["measurement_id"]
