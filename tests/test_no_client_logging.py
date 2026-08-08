"""Guard: the server does not push log messages to the MCP client.

SEP-2577 deprecates the MCP logging capability with no in-protocol
replacement — the SDK names stderr and OpenTelemetry as the alternatives.
The server therefore narrates progress on the stdlib logger, which
``src/server.py`` routes to stderr (safe for the stdio transport, whose
protocol channel is stdout).

This is easy to undo by accident: ``ctx`` is still injected into every tool
for contract stability, so ``await ctx.info(...)`` still type-checks and
still "works" under mcp 2.x — it just emits a deprecation warning that is
shown by default, and it stops working entirely when the capability is
removed. Every other test in this suite mocks ``ctx``, so nothing else
would notice.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

FORBIDDEN = re.compile(r"\bctx\.(info|warning|debug|error|log)\s*\(")


def test_no_source_file_logs_through_the_mcp_context():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if FORBIDDEN.search(line):
                offenders.append(
                    f"{path.relative_to(SRC.parent)}:{lineno}: {line.strip()}"
                )

    assert offenders == [], (
        "These call the deprecated MCP logging capability (SEP-2577). Use the "
        "module logger instead — it reaches the operator via stderr and does "
        "not depend on a capability scheduled for removal:\n  "
        + "\n  ".join(offenders)
    )


def test_ctx_is_still_injected_into_tools():
    """The migration must not have removed ctx from any tool signature.

    ``tests/fixtures/tool_inventory.json`` pins ``context_kwarg`` per tool,
    so dropping the parameter would be a protocol-visible change. This
    states the intent directly rather than leaving it implied by the
    fixture: what changed is *how* progress is emitted, not *whether* tools
    accept a context.
    """
    from mcp.server.mcpserver import MCPServer

    from predictive_maintenance_mcp.mcp_tools import register_all

    mcp = MCPServer("ctx-injection-guard")
    register_all(mcp)
    with_ctx = [
        tool.name
        for tool in mcp._tool_manager._tools.values()
        if tool.context_kwarg
    ]
    assert len(with_ctx) >= 30, (
        f"only {len(with_ctx)} tools still receive ctx injection — the "
        f"migration should have changed how logging is emitted, not whether "
        f"tools accept a context"
    )
