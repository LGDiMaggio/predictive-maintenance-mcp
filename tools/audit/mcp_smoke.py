"""Stdio protocol smoke harness: ``python -m tools.audit smoke``.

Why this exists
---------------
Every protocol test in the suite drives the server in-process. A desktop host
does something else: it launches a subprocess and exchanges newline-delimited
JSON-RPC over its stdin and stdout. Three failure classes only exist there:

* a server that does not start (missing package data, an import that only
  resolves inside a clone, a crash before the first frame);
* a server that writes a non-JSON line to stdout. Under mcp 2.x that line
  does NOT end the session: the transport hands it to the client's message
  handler, whose default discards it. In-process tests never see it, and
  stricter hosts break on it. The harness installs a handler that records
  every such item, so corruption before or after the handshake ends the run
  as ``error`` with the offending text as evidence;
* a handshake era the server no longer speaks. Desktop hosts use the
  2025-11-25 ``initialize`` handshake today, while the SDK's default
  negotiation is the 2026-07-28 ``discover`` exchange. Both run here.

What one run checks
-------------------
Per era (``legacy`` = ``Client(..., mode="legacy")``, ``auto`` = the SDK
default): the negotiated protocol version, the tool names against the frozen
inventory (two-way), every input and output schema against the JSON Schema
2020-12 meta-schema, and that an unknown ``signal_id`` is refused with the
project-owned remedy ``load_signal``. In the legacy era only, the documented
diagnostic chain (generated signal, FFT, envelope, explicit-frequency bearing
check, ISO severity, HTML report), with every artifact required to land
inside the sandbox.

Assertions bind to project-owned fields and substrings only, never to SDK
message wording: the SDK's versioning policy excludes wording from its
compatibility promise.

Sandboxing
----------
Through the environment, not monkeypatching: every launch gets a fresh
``tempfile.mkdtemp`` project root as ``cwd`` and ``PDM_PROJECT_DIR``, with
``PMM_LEDGER_DIR`` under it. ``config.py`` resolves every directory from
``PDM_PROJECT_DIR`` at import, so nothing needs patching. The rest of the
caller's environment is inherited on purpose: the SDK's default passes only a
short allow-list, which would hide exactly the variables a real host passes.
The checkout's ``data/`` and ``reports/`` are listed before and after, and
any difference fails the run.

Observations that never fail the run (``--installed`` mode): whether the
installed package finds bearing 6205 in its catalog, and which folders the
server creates when launched once in an empty directory WITHOUT
``PDM_PROJECT_DIR`` (the ``uvx`` and registry install path; the one
deliberate exception to the sandbox rule, still confined to a temporary
directory). The import location of ``predictive_maintenance_mcp`` is always
recorded; a stated ``--expect-installed-under`` or ``--expect-version`` that
does not hold fails the run.

Exit codes: 0 pass, 1 contract failure, 2 the check could not run (server
did not start, timed out, corrupted stdout, or the inventory was unreadable).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple, Optional

import anyio
import anyio.lowlevel
import jsonschema
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from tools.audit import results
from tools.audit.results import Finding

__all__ = [
    "SchemaViolation",
    "SmokeConfig",
    "add_arguments",
    "check_tool_schemas",
    "default_expected_version",
    "diff_inventory",
    "list_all_tools",
    "load_inventory",
    "run",
    "run_smoke",
    "split_server_cmd",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "tests" / "fixtures" / "tool_inventory.json"
PACKAGE = "predictive_maintenance_mcp"
DISTRIBUTION = "predictive-maintenance-mcp"

#: OneDrive cold starts took 66 s locally; CI runners are far faster.
DEFAULT_TIMEOUT_STARTUP = 120.0
DEFAULT_TIMEOUT_REQUEST = 60.0

#: ``legacy`` = the 2025-11-25 initialize handshake; ``auto`` = SDK default.
ERAS = ("legacy", "auto")

#: Trees of the checkout a sandboxed run must never write into.
CHECKOUT_TREES = ("data", "reports")

#: Bounds on what one launch keeps of a misbehaving server's output. Evidence
#: shows at most results.EXCERPT_LIMIT characters anyway; these only stop a
#: server that floods stdout or stderr from filling the harness's memory.
STDOUT_NOISE_LIMIT = 50
STDERR_TAIL_BYTES = 65536

#: The documented chain, on a short deterministic synthetic signal. The
#: generator's bearing_fault signal repeats an impact every 0.1 s, so the
#: explicit-frequency route names that 10 Hz rate.
TEST_SIGNAL = {
    "signal_type": "bearing_fault",
    "duration": 2.0,
    "sampling_rate": 10000.0,
    "noise_level": 0.1,
    "random_seed": 20261015,
}
CHAIN_RPM = 1800.0
IMPACT_FREQUENCIES = {"impact": 10.0}
SEVERITY_CONTEXT = {"machine_group": 2, "support_type": "rigid"}
CHAIN_TOOLS = (
    "generate_test_signal",
    "analyze_fft",
    "analyze_envelope",
    "check_bearing_faults",
    "assess_severity",
    "generate_envelope_report",
)

#: The refusal probe: an id nothing ever loads, and the remedy the project's
#: own not-found message names.
UNKNOWN_SIGNAL_ID = "pmm-audit-no-such-signal"
REFUSAL_REMEDY = "load_signal"

CATALOG_BEARING = "6205"

#: Run by the interpreter under test, from an empty directory: where does the
#: package import from, and which version do source and metadata declare?
_PROBE_CODE = f"""
import json
from importlib.metadata import PackageNotFoundError, version
import {PACKAGE} as package
try:
    distribution = version({DISTRIBUTION!r})
except PackageNotFoundError:
    distribution = None
print(json.dumps({{
    "file": package.__file__,
    "version": getattr(package, "__version__", None),
    "distribution": distribution,
}}))
"""


# ---------------------------------------------------------------------------
# Pure parts (also imported by tests/test_tool_contracts.py)
# ---------------------------------------------------------------------------


class SchemaViolation(NamedTuple):
    """One schema problem: which tool, which schema field, what is wrong."""

    tool: str
    field: str
    problem: str


def load_inventory(path: Path) -> set[str]:
    """Return the tool names of a frozen inventory fixture.

    Raises:
        ValueError: When the file is missing, not JSON, or has no ``tools``
            mapping.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read tool inventory {path}: {exc}") from None
    tools = doc.get("tools") if isinstance(doc, dict) else None
    if not isinstance(tools, dict) or not tools:
        raise ValueError(f"tool inventory {path} has no non-empty 'tools' mapping")
    return set(tools)


def diff_inventory(
    discovered: Iterable[str], expected: Iterable[str]
) -> dict[str, list[str]]:
    """Two-way difference between discovered and frozen tool names.

    Returns:
        ``{"missing": [...], "unexpected": [...]}``: inventory names the
        server does not register, and registered names the inventory lacks.
    """
    found, frozen = set(discovered), set(expected)
    return {
        "missing": sorted(frozen - found),
        "unexpected": sorted(found - frozen),
    }


def check_tool_schemas(tools: Iterable[Mapping[str, Any]]) -> list[SchemaViolation]:
    """Validate tool schemas as the protocol carries them.

    Every ``inputSchema`` and ``outputSchema`` must be a valid JSON Schema
    2020-12 document, and every input schema must declare
    ``"type": "object"`` (tool arguments are always a JSON object).

    Args:
        tools: Wire-shaped tool entries with ``name``, ``inputSchema`` and an
            optional ``outputSchema``.

    Returns:
        One violation per offending schema; empty when all are valid.
    """
    violations: list[SchemaViolation] = []
    for tool in tools:
        name = str(tool.get("name"))
        for field_name in ("inputSchema", "outputSchema"):
            schema = tool.get(field_name)
            if schema is None:
                if field_name == "inputSchema":
                    violations.append(SchemaViolation(name, field_name, "missing"))
                continue
            if not isinstance(schema, Mapping):
                violations.append(
                    SchemaViolation(name, field_name, "not a JSON object")
                )
                continue
            try:
                jsonschema.Draft202012Validator.check_schema(dict(schema))
            except jsonschema.exceptions.SchemaError as exc:
                where = "/".join(str(part) for part in exc.absolute_path) or "<root>"
                violations.append(
                    SchemaViolation(
                        name,
                        field_name,
                        f"not a valid JSON Schema 2020-12 document at {where}: "
                        f"{exc.message}",
                    )
                )
                continue
            if field_name == "inputSchema" and schema.get("type") != "object":
                violations.append(
                    SchemaViolation(
                        name,
                        field_name,
                        'input schema must declare "type": "object", got '
                        f"{schema.get('type')!r}",
                    )
                )
    return violations


def split_server_cmd(values: Sequence[str]) -> list[str]:
    """Turn ``--server-cmd`` values into an argv list.

    Several values are taken verbatim, one per argv element (use the
    ``--server-cmd=-m`` spelling for elements that start with a dash). A
    single value is split shell-style, unless it names an existing file, so
    an interpreter path containing spaces still works on its own.

    Raises:
        ValueError: When no command remains.
    """
    items = [value for value in values if value.strip()]
    if len(items) > 1:
        return items
    if not items:
        raise ValueError("--server-cmd is empty")
    single = items[0]
    if os.path.exists(single):
        return [single]
    windows = os.name == "nt"
    parts = shlex.split(single, posix=not windows)
    if windows:
        # Non-POSIX mode keeps the quotes that protected a spaced path.
        parts = [
            part[1:-1] if len(part) > 1 and part[0] == part[-1] in "\"'" else part
            for part in parts
        ]
    if not parts:
        raise ValueError("--server-cmd is empty")
    return parts


def _read_version_assignment(path: Path, prefix: str) -> Optional[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip().strip("\"'")
    return None


def default_expected_version() -> Optional[str]:
    """``__version__`` from this tree's ``src/__init__.py`` (or pyproject).

    Returns:
        The declared version, or None outside a source tree.
    """
    try:
        found = _read_version_assignment(
            REPO_ROOT / "src" / "__init__.py", "__version__"
        )
    except FileNotFoundError:
        found = None
    if found:
        return found
    try:
        return _read_version_assignment(REPO_ROOT / "pyproject.toml", "version = ")
    except FileNotFoundError:
        return None


def default_server_cmd() -> list[str]:
    """``<this interpreter> -m predictive_maintenance_mcp``."""
    return [sys.executable, "-m", PACKAGE]


def sdk_protocol_versions() -> dict[str, Any]:
    """The installed SDK's protocol-version constants, read defensively."""
    # upstream.sdk_protocol_versions probes the same SDK constants for another
    # purpose: new revisions to review, not this run's handshake evidence.
    try:
        module = importlib.import_module("mcp.types.version")
    except ImportError:
        return {}
    out: dict[str, Any] = {}
    for key, attribute in (
        ("known", "KNOWN_PROTOCOL_VERSIONS"),
        ("handshake", "HANDSHAKE_PROTOCOL_VERSIONS"),
        ("modern", "MODERN_PROTOCOL_VERSIONS"),
    ):
        value = getattr(module, attribute, None)
        if value is not None:
            out[key] = sorted(str(item) for item in value)
    latest = getattr(module, "LATEST_HANDSHAKE_VERSION", None)
    if latest is not None:
        out["latest_handshake"] = str(latest)
    return out


def _snapshot_checkout() -> Optional[dict[str, list[str]]]:
    """Relative file listing of the checkout trees a run must not touch."""
    if not (REPO_ROOT / "pyproject.toml").is_file():
        return None
    listing: dict[str, list[str]] = {}
    for tree in CHECKOUT_TREES:
        base = REPO_ROOT / tree
        listing[tree] = _relative_files(base) if base.is_dir() else []
    return listing


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class SmokeConfig:
    """Inputs of one harness run (the CLI maps its flags onto this).

    ``probe_prelude`` is prepended to the import-location probe, the way
    pytest prepends ``conftest.SUBPROCESS_PIN`` to the server command so a
    worktree probes its own source. ``eras`` exists for tests that only need
    one launch.
    """

    server_cmd: list[str] = field(default_factory=default_server_cmd)
    inventory: Path = DEFAULT_INVENTORY
    expect_version: Optional[str] = None
    installed: bool = False
    expect_installed_under: Optional[Path] = None
    python: Optional[str] = None
    probe_prelude: str = ""
    timeout_startup: float = DEFAULT_TIMEOUT_STARTUP
    timeout_request: float = DEFAULT_TIMEOUT_REQUEST
    variant: Optional[str] = None
    eras: tuple[str, ...] = ERAS

    def probe_python(self) -> Optional[str]:
        """Interpreter for the import probe: explicit, or the server's own."""
        if self.python:
            return self.python
        if len(self.server_cmd) >= 2 and self.server_cmd[1] in ("-m", "-c"):
            return self.server_cmd[0]
        return None

    def resolved_variant(self) -> str:
        if self.variant:
            return self.variant
        kind = "installed" if self.installed else "checkout"
        major, minor = sys.version_info[:2]
        return f"{kind}-{sys.platform}-py{major}.{minor}"


class _Run:
    """Collects findings, gate outcomes and facts across launches."""

    def __init__(self) -> None:
        self.findings: dict[str, Finding] = {}
        self.failures: set[str] = set()
        self.errors: set[str] = set()
        self.facts: dict[str, Any] = {}
        self.sandboxes: list[Path] = []

    def add(
        self,
        finding: Finding,
        gate: Optional[str] = None,
        era: Optional[str] = None,
    ) -> None:
        """Record ``finding``; ``gate`` is ``fail``, ``error`` or None.

        The same fingerprint seen in a second era is merged, with the era
        appended to its evidence, so one problem stays one finding.
        """
        kept = self.findings.get(finding.fingerprint, finding)
        if era is not None:
            eras = list(kept.evidence.get("eras", []))
            if era not in eras:
                eras.append(era)
            kept = replace(kept, evidence={**kept.evidence, "eras": eras})
        self.findings[finding.fingerprint] = kept
        if gate == "fail":
            self.failures.add(finding.fingerprint)
        elif gate == "error":
            self.errors.add(finding.fingerprint)

    def fact(self, key: str, era: str, value: Any) -> None:
        self.facts.setdefault(key, {})[era] = value

    def sandbox(self, prefix: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"pmm-smoke-{prefix}-"))
        self.sandboxes.append(root)
        return root

    @property
    def status(self) -> str:
        if self.errors:
            return "error"
        if self.failures:
            return "fail"
        return "pass"


def _finding(
    rule: str,
    category: str,
    severity: str,
    subject: str,
    title: str,
    classification: Optional[str] = None,
    **evidence: Any,
) -> Finding:
    return Finding(
        rule=rule,
        category=category,
        severity=severity,
        subject=subject,
        title=title,
        classification=classification,
        evidence={key: value for key, value in evidence.items() if value is not None},
    )


def _describe(exc: BaseException) -> str:
    """Flatten exception groups into ``Type: message`` leaves."""
    leaves: list[BaseException] = []

    def walk(item: BaseException) -> None:
        if isinstance(item, BaseExceptionGroup):
            for inner in item.exceptions:
                walk(inner)
        else:
            leaves.append(item)

    walk(exc)
    return "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)


def _stderr_tail(text: str) -> str:
    return results.truncate_evidence(text.strip(), keep="tail")


def _sandbox_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PDM_PROJECT_DIR"] = str(root)
    env["PMM_LEDGER_DIR"] = str(root / "ledger")
    return env


def _inside(path: Any, root: Path) -> bool:
    if not isinstance(path, str) or not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _text(result: Any) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content or [])


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def _read_tail(log: BinaryIO, limit: int = STDERR_TAIL_BYTES) -> str:
    """The last ``limit`` bytes of ``log``, decoded like a text-mode read.

    UTF-8 with replacement and universal newlines, as ``TextIOWrapper``
    decodes them. A cut through a multi-byte character costs at most one
    replacement character at the start.
    """
    size = log.seek(0, os.SEEK_END)
    log.seek(max(0, size - limit))
    text = log.read().decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# One server launch
# ---------------------------------------------------------------------------


class _Deadline:
    """Moves one cancel scope's deadline from phase to phase.

    A single scope encloses the whole session, so the transport's task group
    is entered and left inside it; per-phase timeouts only move its deadline.
    """

    def __init__(self, request_timeout: float) -> None:
        self.request_timeout = request_timeout
        self.scope: Optional[anyio.CancelScope] = None
        self.phase = "spawn"
        self.seconds = request_timeout

    def start(self, phase: str, seconds: Optional[float] = None) -> None:
        self.phase = phase
        self.seconds = self.request_timeout if seconds is None else seconds
        if self.scope is not None:
            self.scope.deadline = anyio.current_time() + self.seconds


@dataclass
class _Launch:
    problem: Optional[str] = None  # finding rule: "timeout" or "session-error"
    phase: str = ""
    detail: str = ""
    stderr: str = ""
    stdout_noise: list[str] = field(default_factory=list)
    stdout_noise_dropped: int = 0  # lines past STDOUT_NOISE_LIMIT, counted only
    handshake_s: Optional[float] = None
    elapsed_s: float = 0.0


Body = Callable[[Client, _Deadline], Awaitable[None]]


async def _launch(
    params: StdioServerParameters,
    era: str,
    config: SmokeConfig,
    body: Body,
) -> _Launch:
    """Spawn the server, handshake in ``era``, run ``body``, always clean up."""
    outcome = _Launch()

    async def record_noise(message: Any) -> None:
        # Exception items are stdout lines the transport could not parse.
        if isinstance(message, Exception):
            if len(outcome.stdout_noise) < STDOUT_NOISE_LIMIT:
                outcome.stdout_noise.append(str(message))
            else:
                outcome.stdout_noise_dropped += 1
        await anyio.lowlevel.checkpoint()

    client_kwargs: dict[str, Any] = {"message_handler": record_noise}
    if era == "legacy":
        client_kwargs["mode"] = "legacy"
    clock = _Deadline(config.timeout_request)
    started = time.perf_counter()
    # The server writes to this file's descriptor and _read_tail reads the
    # bytes back; text mode only because stdio_client types errlog as TextIO.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        try:
            with anyio.CancelScope() as scope:
                clock.scope = scope
                clock.start("startup", config.timeout_startup)
                transport = stdio_client(params, errlog=log)
                async with Client(transport, **client_kwargs) as client:
                    outcome.handshake_s = round(time.perf_counter() - started, 3)
                    await body(client, clock)
            if scope.cancelled_caught:
                outcome.problem = "timeout"
                outcome.detail = f"no response within {clock.seconds:g} s"
        except Exception as exc:  # the transport reports failures as groups
            outcome.problem = "session-error"
            outcome.detail = _describe(exc)
        outcome.phase = clock.phase
        outcome.stderr = _read_tail(log.buffer)
    outcome.elapsed_s = round(time.perf_counter() - started, 3)
    return outcome


def _record_launch_problems(
    outcome: _Launch, label: str, run: _Run, era: Optional[str]
) -> None:
    """Turn a launch's error and stdout noise into error-gated findings."""
    if outcome.problem is not None:
        run.add(
            _finding(
                outcome.problem,
                "mcp-compat",
                "high",
                f"{label} {outcome.phase}",
                f"The server session failed ({label}, {outcome.phase}): "
                f"{results.truncate_evidence(outcome.detail, 300)}",
                excerpt=_stderr_tail(outcome.stderr) or None,
                detail=outcome.detail,
            ),
            gate="error",
            era=era,
        )
    if outcome.stdout_noise:
        noise = list(outcome.stdout_noise)
        if outcome.stdout_noise_dropped:
            noise.append(
                f"[... {outcome.stdout_noise_dropped} more lines not kept ...]"
            )
        run.add(
            _finding(
                "stdout-not-json-rpc",
                "mcp-compat",
                "high",
                f"{label} server stdout",
                "The server wrote a line to stdout that is not JSON-RPC",
                excerpt="\n".join(noise),
                stderr=_stderr_tail(outcome.stderr) or None,
            ),
            gate="error",
            era=era,
        )


# ---------------------------------------------------------------------------
# Per-era checks
# ---------------------------------------------------------------------------


async def list_all_tools(client: Client) -> list[Any]:
    """Every tool ``tools/list`` returns, following ``next_cursor`` pages.

    Raises:
        RuntimeError: When the server still returns a cursor after 100 pages.
    """
    tools: list[Any] = []
    cursor: Optional[str] = None
    for _ in range(100):
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools
    raise RuntimeError("tools/list kept returning a next cursor after 100 pages")


def _record_handshake(era: str, client: Client, run: _Run) -> None:
    version = client.protocol_version
    run.fact("negotiated_protocol", era, version)
    info = client.server_info
    run.fact(
        "server_info",
        era,
        None if info is None else {"name": info.name, "version": info.version},
    )
    if era != "legacy":
        return
    handshake = sdk_protocol_versions().get("handshake")
    if handshake is None:
        run.add(
            _finding(
                "protocol-version-unverified",
                "mcp-compat",
                "medium",
                "legacy era negotiated version",
                "The installed SDK exposes no handshake version list to check against",
                classification="not-checked",
                versions={"negotiated": version},
            ),
            era=era,
        )
    elif version not in handshake:
        run.add(
            _finding(
                "protocol-version-unexpected",
                "mcp-compat",
                "high",
                "legacy era negotiated version",
                "The legacy handshake negotiated a version outside the SDK's "
                "handshake set",
                versions={"negotiated": version, "handshake": handshake},
            ),
            gate="fail",
            era=era,
        )


def _check_discovery(
    era: str, tools: list[Any], expected: Optional[set[str]], run: _Run
) -> None:
    names = [tool.name for tool in tools]
    run.fact("tool_count", era, len(names))
    if expected is not None:
        diff = diff_inventory(names, expected)
        for name in diff["missing"]:
            run.add(
                _finding(
                    "tool-missing",
                    "tool-contract",
                    "high",
                    name,
                    f"Inventory tool {name} is not registered by the server",
                ),
                gate="fail",
                era=era,
            )
        for name in diff["unexpected"]:
            run.add(
                _finding(
                    "tool-unexpected",
                    "tool-contract",
                    "high",
                    name,
                    f"The server registers {name}, which the inventory does not list",
                ),
                gate="fail",
                era=era,
            )
    wire = [
        {
            "name": tool.name,
            "inputSchema": getattr(tool, "input_schema", None),
            "outputSchema": getattr(tool, "output_schema", None),
        }
        for tool in tools
    ]
    for violation in check_tool_schemas(wire):
        run.add(
            _finding(
                "schema-invalid",
                "tool-contract",
                "high",
                f"{violation.tool} {violation.field}",
                f"{violation.tool} publishes an invalid {violation.field}",
                excerpt=violation.problem,
            ),
            gate="fail",
            era=era,
        )


async def _call(
    client: Client, clock: _Deadline, name: str, arguments: dict[str, Any]
) -> Any:
    clock.start(f"tools/call {name}")
    return await client.call_tool(name, arguments)


def _step_failed(name: str, arguments: dict[str, Any], result: Any) -> Finding:
    return _finding(
        "chain-step-failed",
        "tool-contract",
        "high",
        name,
        f"{name} did not return a structured, non-error result in the chain",
        excerpt=_text(result) or None,
        arguments=arguments,
        is_error=bool(result.is_error),
    )


async def _run_chain(client: Client, clock: _Deadline, root: Path, run: _Run) -> None:
    """The documented chain; stops at the first step that fails."""
    status = {name: "not-run" for name in CHAIN_TOOLS}
    run.facts["chain"] = status
    signal_id: Optional[str] = None
    artifacts: dict[str, Any] = {}
    for name in CHAIN_TOOLS:
        if name == "generate_test_signal":
            arguments: dict[str, Any] = dict(TEST_SIGNAL)
        elif name == "check_bearing_faults":
            arguments = {
                "signal_id": signal_id,
                "rpm": CHAIN_RPM,
                "frequencies": dict(IMPACT_FREQUENCIES),
            }
        elif name == "assess_severity":
            arguments = {"signal_id": signal_id, "rpm": CHAIN_RPM, **SEVERITY_CONTEXT}
        else:
            arguments = {"signal_id": signal_id}
        result = await _call(client, clock, name, arguments)
        content = result.structured_content
        if name == "generate_test_signal" and isinstance(content, dict):
            signal_id = content.get("signal_id")
            artifacts["generate_test_signal.filepath"] = content.get("filepath")
        usable = isinstance(content, dict) and isinstance(signal_id, str)
        if result.is_error or not usable:
            status[name] = "failed"
            run.add(_step_failed(name, arguments, result), gate="fail", era="legacy")
            return
        status[name] = "ok"
        if name == "generate_envelope_report":
            artifacts["generate_envelope_report.file_path"] = content.get("file_path")
    for key, path in artifacts.items():
        if not (_inside(path, root) and Path(path).is_file()):
            run.add(
                _finding(
                    "chain-artifact-outside-sandbox",
                    "tool-contract",
                    "high",
                    key,
                    f"{key} is missing or lies outside the sandboxed project root",
                    paths=[str(path)],
                ),
                gate="fail",
                era="legacy",
            )


async def _check_refusal(era: str, client: Client, clock: _Deadline, run: _Run) -> None:
    arguments = {"signal_id": UNKNOWN_SIGNAL_ID}
    result = await _call(client, clock, "analyze_fft", arguments)
    text = _text(result)
    named = REFUSAL_REMEDY in text
    run.fact("refusal", era, {"is_error": bool(result.is_error), "remedy_named": named})
    if not (result.is_error and named):
        run.add(
            _finding(
                "refusal-contract",
                "tool-contract",
                "high",
                "analyze_fft unknown signal_id",
                "An unknown signal_id is not refused with the load_signal remedy",
                excerpt=text or None,
                arguments=arguments,
                is_error=bool(result.is_error),
            ),
            gate="fail",
            era=era,
        )


async def _observe_catalog(client: Client, clock: _Deadline, run: _Run) -> None:
    result = await _call(
        client, clock, "search_bearing_catalog", {"bearing_id": CATALOG_BEARING}
    )
    content = result.structured_content
    if isinstance(content, dict) and isinstance(content.get("result"), dict):
        content = content["result"]
    if result.is_error or not isinstance(content, dict):
        state = "error"
    elif content.get("status") == "not_found":
        state = "not_found"
    else:
        state = "found" if content.get("designation") else "error"
    run.facts["bearing_catalog_6205"] = state
    if state != "found":
        run.add(
            _finding(
                "bearing-catalog-not-shipped",
                "install",
                "medium",
                f"search_bearing_catalog {CATALOG_BEARING}",
                f"The installed package does not find bearing {CATALOG_BEARING} "
                f"in its catalog ({state})",
                excerpt=_text(result) or None,
            )
        )


async def _run_era(
    era: str, config: SmokeConfig, expected: Optional[set[str]], run: _Run
) -> None:
    root = run.sandbox(era)
    params = StdioServerParameters(
        command=config.server_cmd[0],
        args=list(config.server_cmd[1:]),
        env=_sandbox_env(root),
        cwd=str(root),
    )

    async def body(client: Client, clock: _Deadline) -> None:
        _record_handshake(era, client, run)
        clock.start("tools/list")
        _check_discovery(era, await list_all_tools(client), expected, run)
        if era == "legacy":
            await _run_chain(client, clock, root, run)
        await _check_refusal(era, client, clock, run)
        if era == "legacy" and config.installed:
            await _observe_catalog(client, clock, run)

    outcome = await _launch(params, era, config, body)
    run.fact(
        "timing_s", era, {"handshake": outcome.handshake_s, "era": outcome.elapsed_s}
    )
    run.fact("sandbox_files", era, _relative_files(root))
    _record_launch_problems(outcome, f"{era} era", run, era)


async def _observe_default_launch(config: SmokeConfig, run: _Run) -> None:
    """Installed mode: one launch in an empty directory, no project root set."""
    launch_dir = run.sandbox("launch")
    env = dict(os.environ)
    env.pop("PDM_PROJECT_DIR", None)
    env.pop("PMM_LEDGER_DIR", None)
    params = StdioServerParameters(
        command=config.server_cmd[0],
        args=list(config.server_cmd[1:]),
        env=env,
        cwd=str(launch_dir),
    )

    async def body(client: Client, clock: _Deadline) -> None:
        return None

    outcome = await _launch(params, "legacy", config, body)
    _record_launch_problems(outcome, "default launch", run, None)
    folders = sorted(
        path.relative_to(launch_dir).as_posix()
        for path in launch_dir.rglob("*")
        if path.is_dir()
    )
    run.facts["default_launch_folders"] = folders
    run.add(
        _finding(
            "default-launch-folders",
            "install",
            "low" if folders else "info",
            "default launch working directory",
            "Launched without PDM_PROJECT_DIR, the server creates "
            f"{len(folders)} folder(s) in its working directory",
            paths=folders,
        )
    )


def _probe_import(config: SmokeConfig, run: _Run) -> None:
    """Record where the package imports from; check stated expectations."""
    python = config.probe_python()
    run.facts["expected_version"] = config.expect_version
    if python is None:
        run.facts["import_location"] = None
        if config.expect_version or config.expect_installed_under:
            run.add(
                _finding(
                    "import-probe-not-run",
                    "install",
                    "high",
                    "predictive_maintenance_mcp import",
                    "No interpreter to probe the import with; pass --python",
                    classification="not-checked",
                ),
                gate="error",
            )
        return
    cwd = run.sandbox("probe")
    try:
        proc = subprocess.run(
            [python, "-c", config.probe_prelude + _PROBE_CODE],
            cwd=cwd,
            env=_sandbox_env(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.timeout_startup,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        info = json.loads(lines[-1]) if proc.returncode == 0 and lines else None
        detail = proc.stderr
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        info, detail = None, _describe(exc)
    if not isinstance(info, dict):
        run.facts["import_location"] = None
        run.add(
            _finding(
                "import-probe-failed",
                "install",
                "high",
                "predictive_maintenance_mcp import",
                f"{python} cannot import {PACKAGE} outside the checkout",
                excerpt=_stderr_tail(detail) or None,
                command=[python, "-c", "<import probe>"],
            ),
            gate="error",
        )
        return
    location = info.get("file")
    run.facts["import_location"] = location
    run.facts["package_version"] = info.get("version")
    run.facts["distribution_version"] = info.get("distribution")
    run.add(
        _finding(
            "import-location",
            "install",
            "info",
            "predictive_maintenance_mcp import location",
            f"{PACKAGE} imports from the recorded location",
            classification="harmless",
            paths=[location],
        )
    )
    if config.expect_installed_under is not None and not _inside(
        location, Path(config.expect_installed_under)
    ):
        run.add(
            _finding(
                "import-location-unexpected",
                "install",
                "high",
                "predictive_maintenance_mcp import location",
                f"{PACKAGE} imports from outside the expected install prefix",
                paths=[location, str(config.expect_installed_under)],
            ),
            gate="fail",
        )
    if config.expect_version and info.get("version") != config.expect_version:
        run.add(
            _finding(
                "package-version-mismatch",
                "install",
                "high",
                "predictive_maintenance_mcp __version__",
                f"{PACKAGE}.__version__ differs from the expected version",
                versions={
                    "expected": config.expect_version,
                    "found": info.get("version"),
                },
            ),
            gate="fail",
        )


def _check_checkout_untouched(
    before: Optional[dict[str, list[str]]], run: _Run
) -> None:
    after = _snapshot_checkout()
    if before is None or after is None:
        return
    for tree in CHECKOUT_TREES:
        added = sorted(set(after[tree]) - set(before[tree]))
        removed = sorted(set(before[tree]) - set(after[tree]))
        if added or removed:
            run.add(
                _finding(
                    "checkout-written",
                    "tool-contract",
                    "high",
                    f"checkout {tree}/",
                    f"The sandboxed run changed the checkout's {tree}/ tree",
                    paths=[f"+{path}" for path in added]
                    + [f"-{path}" for path in removed],
                ),
                gate="fail",
            )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_smoke(config: SmokeConfig) -> results.Result:
    """Run every check of one harness invocation and return its result."""
    if not config.server_cmd:
        raise ValueError("the server command is empty")
    started_at = results.utc_now()
    started = time.perf_counter()
    run = _Run()
    run.facts["installed_mode"] = config.installed
    run.facts["server_command"] = [
        results.truncate_evidence(part, 200) for part in config.server_cmd
    ]
    run.facts["sdk_protocol_versions"] = sdk_protocol_versions()

    expected: Optional[set[str]] = None
    try:
        expected = load_inventory(config.inventory)
        run.facts["inventory_tool_count"] = len(expected)
    except ValueError as exc:
        run.add(
            _finding(
                "inventory-unreadable",
                "tool-contract",
                "high",
                "tool inventory",
                "The frozen tool inventory could not be read",
                classification="not-checked",
                excerpt=str(exc),
            ),
            gate="error",
        )

    before = _snapshot_checkout()
    try:
        _probe_import(config, run)

        async def all_launches() -> None:
            for era in config.eras:
                await _run_era(era, config, expected, run)
            if config.installed:
                await _observe_default_launch(config, run)

        anyio.run(all_launches)
        _check_checkout_untouched(before, run)
    finally:
        for sandbox in run.sandboxes:
            shutil.rmtree(sandbox, ignore_errors=True)
    run.facts.setdefault("timing_s", {})["total"] = round(
        time.perf_counter() - started, 3
    )
    return results.Result(
        check="smoke",
        variant=config.resolved_variant(),
        status=run.status,
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=results.environment_snapshot(),
        facts=run.facts,
        findings=sorted(
            run.findings.values(), key=lambda item: (item.rule, item.subject)
        ),
    )


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not seconds > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return seconds


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``smoke`` subcommand's flags."""
    parser.add_argument(
        "--server-cmd",
        action="append",
        default=None,
        help=(
            "server command: one string (split shell-style) or one element "
            "per repetition (use --server-cmd=-m for dashed elements); "
            "default: this interpreter -m predictive_maintenance_mcp"
        ),
    )
    parser.add_argument(
        "--expect-version",
        default=None,
        help="expected predictive_maintenance_mcp.__version__ "
        "(default: read from this tree's src/__init__.py when present)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="frozen tool inventory (default: tests/fixtures/tool_inventory.json)",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="installed-package mode: adds the catalog and default-launch "
        "observations",
    )
    parser.add_argument(
        "--expect-installed-under",
        type=Path,
        default=None,
        help="fail unless the package imports from under this prefix",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter for the import probe (default: the server command's "
        "interpreter when it is invoked with -m or -c)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument(
        "--timeout-startup",
        type=_positive_seconds,
        default=DEFAULT_TIMEOUT_STARTUP,
        help="seconds allowed for spawn plus handshake (default: 120)",
    )
    parser.add_argument(
        "--timeout-request",
        type=_positive_seconds,
        default=DEFAULT_TIMEOUT_REQUEST,
        help="seconds allowed per request after the handshake (default: 60)",
    )
    parser.add_argument("--variant", default=None, help="free label for the run")


def run(args: argparse.Namespace) -> int:
    """``smoke`` subcommand: run, optionally write JSON, print a summary."""
    config = SmokeConfig(
        server_cmd=(
            split_server_cmd(args.server_cmd)
            if args.server_cmd
            else default_server_cmd()
        ),
        inventory=args.inventory,
        expect_version=(
            args.expect_version
            if args.expect_version is not None
            else default_expected_version()
        ),
        installed=args.installed,
        expect_installed_under=args.expect_installed_under,
        python=args.python,
        timeout_startup=args.timeout_startup,
        timeout_request=args.timeout_request,
        variant=args.variant,
    )
    result = run_smoke(config)
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    facts = result.facts
    print(
        f"smoke {result.status} ({result.variant}): "
        f"protocol {facts.get('negotiated_protocol')}, "
        f"tools {facts.get('tool_count')}"
    )
    for finding in result.findings:
        print(f"  [{finding.severity}] {finding.rule}: {finding.title}")
    return result.exit_code
