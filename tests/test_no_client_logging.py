"""Guards: the server does not log to the MCP client, and its own log
records never touch stdout.

SEP-2577 deprecates the MCP logging capability with no in-protocol
replacement — the SDK names stderr and OpenTelemetry as the alternatives.
Since 0.12.0 the stdlib logger is therefore the ONLY progress channel, which
makes two properties load-bearing:

1. No source file calls the deprecated capability.
2. Those records reach stderr and never stdout, because stdout is the stdio
   transport's JSON-RPC channel.

Both are easy to break silently. ``ctx`` is still injected into every tool
for contract stability, so ``await ctx.info(...)`` still type-checks and
still "works" under mcp 2.x — it just warns, and stops working entirely when
the capability is removed. Every other test in this suite mocks ``ctx``, so
nothing else would notice. And the routing was, before this file existed,
asserted by nobody: the ``logging.basicConfig`` call that claimed to install
a stderr handler was inert, because ``MCPServer(...)`` claims the root logger
at import and ``basicConfig`` is documented to do nothing in that case.
"""

import ast
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
INVENTORY = Path(__file__).parent / "fixtures" / "tool_inventory.json"

#: Methods on the MCP context that send a client-facing log notification.
CLIENT_LOG_METHODS = frozenset({"info", "warning", "debug", "error", "log"})

#: Marker the subprocess routing probes look for on each stream.
PROBE = "routing-probe-marker"


def _run_probe(preamble: str = "") -> tuple[str, str]:
    """Emit one package log record in a clean interpreter; return (out, err).

    The child gets conftest's SUBPROCESS_PIN first. Without it the child
    loads no conftest, so the editable install's hardcoded MAPPING wins and
    the probe answers about whichever checkout ran ``pip install -e .`` --
    which in a git worktree is not this one. These two tests exist precisely
    because in-process capture could not tell us where the bytes go; a probe
    aimed at the wrong tree would be the same class of false confidence.
    """
    from conftest import SUBPROCESS_PIN

    code = (
        f"{SUBPROCESS_PIN}"
        f"{preamble}"
        "import logging;"
        "import predictive_maintenance_mcp.server as s;"
        f"assert s.__file__.startswith({str(SRC)!r}), (\n"
        f"    'probe resolved to ' + s.__file__ + ', not this checkout')\n"
        f"logging.getLogger(s.PACKAGE_LOGGER + '.probe').info({PROBE!r})"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    return proc.stdout, proc.stderr


class _ClientLogCalls(ast.NodeVisitor):
    """Find client-facing log calls, keyed on the Context type, not a name.

    A regex over the literal token ``ctx.`` was the obvious guard and it is
    not enough. It misses three reintroductions that all still notify the
    client: a parameter named anything else (``context.info(...)``), a local
    alias (``emit = ctx.info``), and ``ctx.session.log(...)`` — which carries
    the same SEP-2577 deprecation on the connection object. Binding to the
    annotation instead of the spelling closes the first two; the explicit
    ``.session.log`` chain check closes the third.
    """

    def __init__(self) -> None:
        self.offenders: list[tuple[int, str]] = []
        self._ctx_names: list[set[str]] = [set()]

    # -- scope handling ---------------------------------------------------
    def _enter_function(self, node) -> None:
        names = set(self._ctx_names[-1])
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if arg.annotation is not None and "Context" in ast.unparse(arg.annotation):
                names.add(arg.arg)
        self._ctx_names.append(names)
        self.generic_visit(node)
        self._ctx_names.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def visit_Assign(self, node: ast.Assign) -> None:
        # `emit = ctx.info` — track the alias so the call site is caught.
        if isinstance(node.value, ast.Attribute) and self._is_ctx_attr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.offenders.append(
                        (node.lineno, f"aliases {ast.unparse(node.value)}")
                    )
        self.generic_visit(node)

    # -- detection --------------------------------------------------------
    def _is_ctx_attr(self, node: ast.Attribute) -> bool:
        return (
            node.attr in CLIENT_LOG_METHODS
            and isinstance(node.value, ast.Name)
            and node.value.id in self._ctx_names[-1]
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if self._is_ctx_attr(func):
                self.offenders.append((node.lineno, ast.unparse(func)))
            # `<anything>.session.log(...)` — the connection-level shape.
            elif (
                func.attr == "log"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "session"
            ):
                self.offenders.append((node.lineno, ast.unparse(func)))
        self.generic_visit(node)


def test_no_source_file_logs_to_the_mcp_client():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        finder = _ClientLogCalls()
        finder.visit(ast.parse(path.read_text(encoding="utf-8")))
        offenders += [
            f"{path.relative_to(REPO_ROOT)}:{line}: {what}"
            for line, what in finder.offenders
        ]

    assert offenders == [], (
        "These reach the deprecated MCP logging capability (SEP-2577). Use "
        "the module logger instead — it reaches the operator via stderr and "
        "does not depend on a capability scheduled for removal:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_detects_the_shapes_a_regex_would_miss():
    """The guard is only worth its place if it catches the evasions.

    Asserted directly rather than trusted, because a guard that silently
    stopped matching would look identical to a clean tree.
    """
    source = """
async def a(ctx: Context):
    await ctx.info("plain")

async def b(context: Context):
    await context.warning("renamed parameter")

async def c(ctx: Context):
    emit = ctx.info

async def d(ctx: Context):
    await ctx.session.log("info", "connection level")

async def clean(ctx: Context):
    logger.info("this one is fine")
    ctx.report_progress(0.5)
"""
    finder = _ClientLogCalls()
    finder.visit(ast.parse(source))
    caught = {line for line, _ in finder.offenders}
    assert caught == {
        3,
        6,
        9,
        12,
    }, f"guard missed a reintroduction shape; caught lines {sorted(caught)}"


def test_every_tool_that_had_ctx_injected_still_does():
    """Pin against the frozen inventory, not a magic floor.

    An earlier version asserted ``>= 30`` while 32 tools actually carry a
    context. That tolerated silently dropping two, and its docstring claimed
    to rely on a fixture the code never opened. Comparing the real set makes
    an intentional change require editing this assertion rather than just
    regenerating a snapshot.
    """
    from mcp.server.mcpserver import MCPServer

    from predictive_maintenance_mcp.mcp_tools import register_all

    reference = json.loads(INVENTORY.read_text(encoding="utf-8"))
    expected = {
        name for name, spec in reference["tools"].items() if spec.get("context_kwarg")
    }

    mcp = MCPServer("ctx-injection-guard")
    register_all(mcp)
    actual = {
        tool.name for tool in mcp._tool_manager._tools.values() if tool.context_kwarg
    }

    assert actual == expected, (
        "context injection drifted from the frozen inventory. 0.12.0 changed "
        "how progress is emitted, not whether tools accept a context:\n"
        f"  lost: {sorted(expected - actual)}\n"
        f"  gained: {sorted(actual - expected)}"
    )


class TestLogRecordsStayOffStdout:
    """The routing this whole migration rests on, asserted.

    Before 0.12.0 these records went to the client over JSON-RPC and where
    the process wrote its logs was a cosmetic question. Now the logger is the
    only channel, and stdout is the protocol channel — a record landing there
    puts a non-JSON line between two frames and ends the session.
    """

    def test_package_logger_does_not_propagate_to_root(self):
        """Root belongs to whoever configured it first, and we cannot win.

        The SDK installs a root handler at import; a host is free to install
        one pointing at stdout before importing us. Not propagating is what
        makes the destination independent of both.
        """
        from predictive_maintenance_mcp import server

        pkg = logging.getLogger(server.PACKAGE_LOGGER)
        assert pkg.propagate is False
        assert pkg.handlers, "package logger has no handler — records vanish"

    def test_handler_is_bound_to_stderr_not_stdout(self):
        from predictive_maintenance_mcp import server

        handler = logging.getLogger(server.PACKAGE_LOGGER).handlers[0]
        # Identity against sys.stdout is checked both ways: pytest replaces
        # sys.stdout during collection, so the un-replaced original matters
        # too.
        import sys

        assert handler.stream is not sys.stdout
        assert handler.stream is not sys.__stdout__

    def test_a_record_reaches_fd_2_and_never_fd_1(self):
        """Run in a real subprocess, because nothing else proves this.

        Neither ``capsys`` nor ``capfd`` can: the handler resolves
        ``sys.stderr`` when it is installed, and pytest has already replaced
        that object by then, so an in-process assertion measures pytest's
        capture rather than the file descriptors. A clean interpreter with
        piped stdout/stderr is the only place the real answer exists — and
        the bug this file was written after was invisible precisely because
        no test ever looked at a real process's stdout.
        """
        out, err = _run_probe()
        assert PROBE not in out, (
            "a log record reached stdout — under the stdio transport that is "
            "the JSON-RPC channel, and a non-JSON line between frames ends "
            "the session"
        )
        assert PROBE in err

    def test_a_host_that_owns_root_cannot_redirect_us_onto_stdout(self):
        """The cascade, reproduced.

        A host is free to call ``basicConfig(stream=sys.stdout)`` before
        importing this package, and root's configuration belongs to whoever
        got there first. While these records propagated to root, that host
        silently moved every tool's narration onto the protocol channel.
        """
        out, err = _run_probe(
            preamble="import logging, sys;"
            "logging.basicConfig(level=logging.INFO, stream=sys.stdout);"
        )
        assert PROBE not in out
        assert PROBE in err


class TestRecordsAreBoundedAndSingleLine:
    """Caller-controlled values are no longer framed by the protocol."""

    def _emit(self, caplog, message):
        from predictive_maintenance_mcp import server

        record = logging.LogRecord(
            name=server.PACKAGE_LOGGER,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        server._OneBoundedLine().filter(record)
        return record.getMessage()

    def test_a_newline_cannot_forge_a_second_log_entry(self, caplog):
        forged = "good\n2026-08-08 - root - ERROR - disk failure imminent"
        rendered = self._emit(caplog, forged)
        assert "\n" not in rendered
        assert "\\n" in rendered

    def test_an_unbounded_value_is_truncated(self, caplog):
        rendered = self._emit(caplog, "A" * 1_000_000)
        assert len(rendered) < 2_200
        assert "truncated" in rendered
