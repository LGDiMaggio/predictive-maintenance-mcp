---
title: "Process-global configuration you cannot win, and claims only running can verify"
date: 2026-08-08
category: architecture-patterns
module: predictive-maintenance-mcp
problem_type: architecture_pattern
component: server
severity: high
applies_when:
  - "Code configures a process-global singleton (logging root, warnings filters, locale, signal handlers) that a library may already have claimed"
  - "A setup function is called from one entry point but the package also exports a runnable object that bypasses it"
  - "A safety guarantee is asserted in a PR body, CHANGELOG, or docstring but nothing executes it"
  - "Writing a guard test whose whole job is to fail when someone reintroduces a forbidden pattern"
  - "Rewriting an assertion during a refactor, where the new form is easier to satisfy than the old one"
  - "A dependency's default behaviour, not your own code, is what makes an invariant hold"
tags: [logging, global-state, verification, guard-tests, load-order, stdio-transport, mutation-testing, adversarial-review, fail-closed, ci-guards]
last_updated: 2026-08-09
---

# Process-global configuration you cannot win, and claims only running can verify

## Context

v0.12.0 moved 119 progress-narration calls off the MCP logging capability
(deprecated by SEP-2577 with no in-protocol replacement) onto the stdlib logger.
The migration's entire safety argument was one sentence, repeated in the commit
message, the CHANGELOG, and the PR body:

> `src/server.py` routes logging to stderr, which is what makes this safe for
> the stdio transport — the JSON-RPC channel is stdout.

The code backing it looked unimpeachable:

```python
def _setup_environment() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
```

It had never once taken effect. An eleven-persona review found it; four
independent probes confirmed it:

```
root handlers BEFORE import: []
root handlers AFTER import : [<RichHandler (NOTSET)>]
root handlers AFTER _setup_environment(): [<RichHandler (NOTSET)>]   # unchanged
intended format  : '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
actual formatter : '%(message)s'
```

`MCPServer(...)` is constructed at module scope, and its `__init__` calls
`logging.basicConfig()` first. `basicConfig` is documented to **do nothing when
the root logger already has handlers**. Ours ran second and silently returned.

Logs did still reach stderr — but because the SDK happened to pass
`Console(stderr=True)`, not because of anything this repository did.

## Guidance

### 1. Do not negotiate with a process-global singleton — own your namespace

`logging.basicConfig` is not idempotent-and-last-wins; it is
first-writer-wins-and-silently-no-ops. The root logger is shared mutable state,
and the winner is decided by import order, which a library cannot control.

Configure the logger you own, and stop propagating:

```python
PACKAGE_LOGGER = __package__ or "predictive_maintenance_mcp"

def configure_logging(level: str | None = None, force: bool = False) -> None:
    pkg = logging.getLogger(PACKAGE_LOGGER)
    pkg.setLevel(_resolve_level(level))
    if pkg.handlers and not force:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    pkg.addHandler(handler)
    pkg.propagate = False   # root's destination is no longer our problem
```

`propagate = False` is what converts "we asked politely and hoped" into a
property. Derive the logger name from `__package__` rather than spelling it, so
a package rename cannot silently detach every module logger from its handler.

### 2. Enumerate entry points before believing a setup function runs

`_setup_environment()` was reachable only from `main()`. But `__init__.py`
exported the server object:

```python
from .server import mcp, main
__all__ = ["mcp", "main", "__version__"]
```

An embedder doing `from predictive_maintenance_mcp import mcp; mcp.run()`
reaches every tool without `main()` ever executing. Root then keeps its default
`WARNING` level and **every INFO line is dropped entirely** — not relocated,
gone. Configuration that must hold on all paths belongs at import of the module
that owns it, not inside one entry point.

Ask specifically: console script, `python -m`, Docker `ENTRYPOINT`, direct
import, `mcp dev` / inspector, test harness. Each is an entry point.

### 3. A claim about global state is not verified by reading the code

Every reviewer who read `_setup_environment()` — including the author, twice —
concluded it was correct. The line is correct *in isolation*. Its behaviour
depends on state established by a different package during import.

Reading cannot settle that class of question. Running can, in three lines:

```python
print(logging.getLogger().handlers)        # before import
import predictive_maintenance_mcp.server
print(logging.getLogger().handlers)        # after — who won?
```

The general rule: **when an invariant depends on load order, global mutable
state, or a dependency's default, execute the check.** If the answer would
change depending on what else got imported, reading the diff cannot produce it.

### 4. Test where the bytes go, not where you told them to go

Nothing in the suite looked at a real process's streams, which is precisely why
this survived. Two traps make the in-process version useless here:

- `capsys` swaps the Python-level `sys.stdout` / `sys.stderr` objects, but the
  handler resolved its stream when it was installed. It writes to the old object.
- `capfd` captures file descriptors, but pytest has *already* replaced
  `sys.stderr` before the handler is created, so the handler never holds fd 2.

A subprocess is the only place the real answer exists:

```python
def _run_probe(preamble: str = "") -> tuple[str, str]:
    code = (
        f"{preamble}"
        "import logging;"
        "import predictive_maintenance_mcp.server as s;"
        f"logging.getLogger(s.PACKAGE_LOGGER + '.probe').info({PROBE!r})"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    return proc.stdout, proc.stderr

def test_a_host_that_owns_root_cannot_redirect_us_onto_stdout():
    out, err = _run_probe(
        "import logging, sys;"
        "logging.basicConfig(level=logging.INFO, stream=sys.stdout);"
    )
    assert PROBE not in out
    assert PROBE in err
```

The second test reproduces the actual attack: a host that claims root first and
points it at stdout. Before `propagate = False`, that put every tool's narration
onto the JSON-RPC channel — a non-JSON line between two frames ends the session.

### 5. Guard tests must themselves be tested

The first tripwire was a regex over source text:

```python
FORBIDDEN = re.compile(r"\bctx\.(info|warning|debug|error|log)\s*\(")
```

It binds to the *spelling* `ctx`, and missed three reintroductions that all still
notify the client: a renamed parameter (`context.info(...)`), a local alias
(`emit = ctx.info`), and `ctx.session.log(...)` — the connection-level shape,
carrying the same SEP-2577 deprecation.

Bind to the type instead of the name, by walking the AST and collecting
parameters whose annotation mentions `Context`. Then assert the guard catches
each evasion:

```python
def test_the_guard_detects_the_shapes_a_regex_would_miss():
    source = """
async def a(ctx: Context):
    await ctx.info("plain")
async def b(context: Context):
    await context.warning("renamed parameter")
async def c(ctx: Context):
    emit = ctx.info
async def d(ctx: Context):
    await ctx.session.log("info", "connection level")
"""
    finder = _ClientLogCalls()
    finder.visit(ast.parse(source))
    assert {line for line, _ in finder.offenders} == {3, 6, 9, 12}
```

Without this, a guard that silently stopped matching is indistinguishable from a
clean tree. Both are green.

### 6. A rewritten assertion can be weaker than the one it replaced

Migrating the tests, `mock_ctx.info.assert_awaited()` became:

```python
assert caplog.records, "expected progress on the module logger"
```

`caplog`'s handler is installed on **root** and collects records from every
logger in the process — WeasyPrint and Plotly both emit here — so this passes on
unrelated noise. Worse, `generate_envelope_report` makes three *unconditional*
log calls before reaching the `if result.get('bearing_matches'):` branch, so the
test named `..._bearing_matches` passed with that branch deleted outright.

Assert identity and content:

```python
def _logged(caplog, needle: str) -> bool:
    return any(r.name == REPORT_LOGGER and needle in r.getMessage()
               for r in caplog.records)
```

Then **mutation-test it**: delete the narration, confirm the test goes red,
restore it, confirm green. An assertion rewritten during a refactor deserves that
check specifically, because the failure mode is a test that still passes.

Note the interaction with §1: once `propagate = False`, caplog's root handler
sees *nothing* from the package, so a test on bare `caplog.records` stops being
lenient and becomes vacuous. A fixture that attaches caplog's handler to the
package logger makes the coupling explicit:

```python
@pytest.fixture
def package_caplog(caplog):
    pkg = logging.getLogger(server.PACKAGE_LOGGER)
    pkg.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=server.PACKAGE_LOGGER)
    try:
        yield caplog
    finally:
        pkg.removeHandler(caplog.handler)
```

### 7. Changing a message's destination changes its threat model

The same string is a different object once it moves channels. These records used
to be MCP notifications — framed by the protocol, readable only by the caller
that triggered them. As lines in an operator's log:

- A newline in a caller-supplied `signal_id` stops being a cosmetic wart and
  becomes a **forged log entry**.
- An unbounded value stops being a payload the client must drain and becomes an
  unbounded **synchronous write to a pipe the client is not obliged to drain** —
  which blocks a coroutine that (after the migration) has no await points.
- Encoding stops being the protocol's problem. A piped stderr on Windows decodes
  as the ANSI code page; a non-ASCII identifier raises inside `emit()`, dropping
  the line and printing `--- Logging error ---` in its place.

Neutralise at the boundary, once, rather than at 119 call sites:

```python
class _OneBoundedLine(logging.Filter):
    def filter(self, record):
        text = record.getMessage()
        if len(text) > _MAX_RECORD_CHARS:
            text = f"{text[:_MAX_RECORD_CHARS]}… [truncated, {len(text)} chars total]"
        record.msg = text.replace("\r", "\\r").replace("\n", "\\n")
        record.args = ()
        return True
```

## Why This Matters

The defect was invisible to code review because **the code was right and the
world was wrong**. Six of eleven reviewers read `_setup_environment()` and did
not flag it. The one that did was the adversarial reviewer, and it found it by
executing an import and printing `logging.getLogger().handlers`.

The cost of not catching it was not "logs looked worse". It was:

1. A documented safety guarantee that did not exist, repeated in three places.
2. An invariant that held only by a dependency's accident — `Console(stderr=True)`
   in a `>=2.0.0,<3.0.0` range. Any 2.x that changed that default would have
   moved 119 log lines onto the stdio protocol channel, and nothing in the repo
   would have noticed.
3. A total-silence failure mode on the library-embed path, strictly worse than
   what it replaced.

## The pattern recurred four times in one week

This document was written after the first instance. The next two arrived
within hours, in the pull requests that were *fixing* the first — each shipped
a new guard, and each guard was unable to fail in the environment it ran in.
The fourth was already in the repo and had been for months.

| Where | The guard | Why it could not fail |
|---|---|---|
| #37 | three `caplog` assertions | `caplog` collects every logger in the process, so `assert caplog.records` passed on unrelated noise. One test named `..._bearing_matches` passed with that branch deleted. |
| #37 | `logging.basicConfig(handlers=[StreamHandler(sys.stderr)])` | inert — `MCPServer(...)` claimed the root logger at import, and `basicConfig` no-ops when root already has handlers. |
| #38 | `test_import_provenance.py` | CI installs from a single checkout, so the stock editable finder already resolves correctly. The assertions passed with the pin deleted. |
| #39 | `tools/check_mypy_baseline.py` | `python -m mypy` exits **1** when mypy is absent — the same code as "found errors", and inside the accepted set. Empty output parsed as zero errors: *"OK: 0 mypy error(s)"*, exit 0. |
| pre-existing | the Black CI job | `continue-on-error: true`. Reports pass unconditionally, exactly as the mypy job did. |

Two things generalise.

**The environment decides whether an assertion is an assertion.** Every one of
these is a correct statement about the system. Three of them are also
*unfalsifiable where they execute*, which is not a weaker version of correct —
it is a different thing wearing the same green tick. The question to ask of a
new guard is not "is this true?" but **"what would have to break for this to go
red, and can that happen here?"** If the answer involves an environment CI
never enters, the guard belongs somewhere CI does enter, or it needs a
synthetic version of that environment (see `TestTheGuardItself` in
`tests/test_import_provenance.py`, which drives a fake competing finder because
CI cannot supply a real one).

**Guards fail toward green.** Each of these fails *open*: the missing checker,
the unread config, the noise-satisfied assertion, the flag that suppresses the
exit code. That direction is not a coincidence — a guard is a negative
assertion, and the cheapest way for a negative assertion to be satisfied is for
its input to be empty. So test the empty input specifically: no checker
installed, no records emitted, no finder registered, unparseable output. Those
are the cases in `tests/test_mypy_baseline_gate.py::TestRunMypyFailsClosed`,
and writing them found a real bug in the gate on the first run.

## When to Apply

Reach for this when any of the following is true:

- You are about to write "this is safe because <module>:<line> does X" in a PR
  body, and X is a call into a process-global subsystem.
- A dependency configures the same global you configure, at import.
- Your package exports something runnable that skips your setup function.
- You are writing a guard test — its value is entirely in failing, so prove it
  fails.
- You are rewriting assertions during a migration. Ask of each new one: what
  would have to break for this to go red? If the answer is "nothing plausible",
  it is decoration.
- A message changes destination. Re-derive the threat model; do not carry over
  the old one.

## Related

- [[credibility-refactor-drift-guards-and-honest-outputs-2026-07-13]] — the same
  repository's earlier finding that facts encoded in two places drift apart. This
  is its runtime sibling: a fact encoded in *code* that the *runtime* contradicts.
  Both are cases where a CI guard, not a careful reader, is the thing that holds.
- [[blocking-mypy-behind-a-frozen-baseline-2026-08-08]] — turning the decorative
  mypy job into a real one. Its gate is the fourth instance tabulated above, and
  its own tests are the worked example of testing a guard's empty-input paths.
- [[editable-installs-pin-one-checkout-and-worktrees-silently-inherit-it-2026-08-08]]
  — why a shared venv makes every worktree test the wrong source. The third
  instance: the pin was right, the test proving it was vacuous where it ran.
