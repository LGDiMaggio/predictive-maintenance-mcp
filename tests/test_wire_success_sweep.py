"""Every registered MCP tool, called once over an in-memory client with valid synthetic input.

Why this module exists
----------------------
A test that calls a tool's Python function directly (``tool.fn``, or the
module-level coroutine with a mocked ``ctx``) never reaches the SDK's
server-side output validation, which checks the returned value against the
tool's declared return type. A result that breaks that contract passes every
such test and fails on every real call. Two registered tools did exactly that
for several releases (``calculate_bearing_characteristic_frequencies`` and
``compute_spectrogram_stft``) while the rest of the suite stayed green.

This module drives the shipped server object through
``Client(server.mcp, mode="legacy")``: initialize handshake, ``tools/call``,
the SDK's output validation, and then a client-side check of every
``structuredContent`` against the ``outputSchema`` the tool advertises.

What it checks
--------------
* Registry closure: the tools in ``tools/list`` equal the tools named in
  ``STEPS``, in both directions, and the prompts in ``prompts/list`` equal
  ``PROMPT_CASES``. A new tool or prompt fails here until it has a case.
* One representative valid call per tool (a few tools are called more than
  once because later steps need their result), each judged against its
  expectation: a success whose structured result conforms to the advertised
  schema, or, for an optional dependency that is not installed, the exact
  documented refusal. ``generate_diagnostic_report_docx`` is the only such
  case: without python-docx it must return ``DOCX_REFUSAL`` word for word,
  with it it must succeed. It is never skipped.
* The two repaired tools at the wire: bearing kinematics against the closed
  form, and the STFT band table against the signal's construction.
* Controls: the judge flags a tool whose output the SDK rejects, a registered
  tool made to return an invalid output, a refusal with different text, and
  the closure check flags a tool without a case and a case without a tool.

Every input is synthetic and every file lives under the test's own temporary
directories (signals, models, reports, resources, ledger, imports); every
by-value import of those directories is patched for the duration of the sweep.
No recording is read and nothing leaves the machine.
"""

from __future__ import annotations

import functools
import json
import math
import re
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

import anyio
import jsonschema
import numpy as np
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from conftest import REPO_ROOT, patch_data_dir
from predictive_maintenance_mcp import document_reader, rag, report_generator
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from tools.audit.mcp_smoke import list_all_tools

#: Anti-rot floor, as in test_tool_contracts: a registry smaller than this
#: means the harness stopped seeing the surface, not that it shrank.
MIN_TOOLS = 30

FS = 10_000.0
SECONDS = 2.0
RPM = 1500.0
GEOMETRY_6205 = {"num_balls": 9, "ball_diameter_mm": 7.94, "pitch_diameter_mm": 39.04}
CALL_TIMEOUT_S = 120

DOCX_TOOL = "generate_diagnostic_report_docx"
#: The documented refusal without the optional extra (report_generator, and
#: the tool's "Requires: pip install predictive-maintenance-mcp[docx]").
DOCX_REFUSAL = (
    "Error executing tool generate_diagnostic_report_docx: python-docx is not "
    "installed. Install with: pip install predictive-maintenance-mcp[docx]"
)

SUCCESS = "success"
REFUSAL = "refusal"
BEARING_FREQUENCY_KEYS = frozenset({"BPFO", "BPFI", "BSF", "FTF", "shaft_freq_hz"})


# ---------------------------------------------------------------------------
# Records and the judge (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallRecord:
    """What the client saw for one ``tools/call``."""

    tool: str
    arguments: Mapping[str, Any]
    is_error: bool = False
    text: str = ""
    structured: Any = None
    advertises_schema: bool = False
    schema_errors: tuple[str, ...] = ()
    client_exception: Optional[str] = None
    precondition_error: Optional[str] = None

    def brief(self) -> str:
        text = json.dumps(dict(self.arguments), default=str, sort_keys=True)
        return text if len(text) <= 100 else text[:97] + "..."


Expectation = tuple[str, Optional[str]]


def expectation(tool: str) -> Expectation:
    """The outcome a valid call of ``tool`` must have in this environment.

    Read at judging time from the implementation's own flag, so the docx case
    follows what the server can actually do here.
    """
    if tool == DOCX_TOOL and not report_generator.HAS_DOCX:
        return (REFUSAL, DOCX_REFUSAL)
    return (SUCCESS, None)


def problems(record: CallRecord, expected: Expectation) -> list[str]:
    """Why ``record`` does not meet ``expected``; empty when it does."""
    kind, refusal_text = expected
    if record.precondition_error is not None:
        return [f"not called, an earlier step failed: {record.precondition_error}"]
    if record.client_exception is not None:
        return [f"the client raised: {record.client_exception}"]
    if kind == REFUSAL:
        if not record.is_error:
            return ["succeeded, but the documented refusal was expected"]
        if record.text != refusal_text:
            return [
                f"refusal text differs: got {record.text!r}, expected {refusal_text!r}"
            ]
        return []
    found: list[str] = []
    if record.is_error:
        found.append(f"returned an error: {record.text[:300]!r}")
    elif record.advertises_schema and record.structured is None:
        found.append("advertises an outputSchema but returned no structuredContent")
    found.extend(
        f"structuredContent breaks the outputSchema: {e}" for e in record.schema_errors
    )
    return found


def closure_problems(
    registered: Sequence[str], cases: Sequence[str], kind: str = "tool"
) -> list[str]:
    """Two-way difference between registered names and the case table."""
    reg, cas = set(registered), set(cases)
    return [
        f"registered {kind} without a case: {name}" for name in sorted(reg - cas)
    ] + [f"case for an unregistered {kind}: {name}" for name in sorted(cas - reg)]


# ---------------------------------------------------------------------------
# Wire calls
# ---------------------------------------------------------------------------


def _schema_errors(
    schema: Optional[Mapping[str, Any]], structured: Any
) -> tuple[str, ...]:
    if not schema or structured is None:
        return ()
    validator = jsonschema.Draft202012Validator(dict(schema))
    return tuple(
        f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message[:200]}"
        for error in validator.iter_errors(structured)
    )


async def output_schemas(client: Client) -> dict[str, Optional[dict[str, Any]]]:
    """Tool name -> advertised ``outputSchema`` (None when absent), wire shape."""
    tools = await list_all_tools(client)
    dumped = [
        t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools
    ]
    return {t["name"]: t.get("outputSchema") for t in dumped}


async def call(
    client: Client,
    tool: str,
    arguments: Mapping[str, Any],
    schemas: Mapping[str, Optional[Mapping[str, Any]]],
) -> CallRecord:
    """One ``tools/call``; every outcome, exceptions included, becomes a record."""
    schema = schemas.get(tool)
    try:
        result = await client.call_tool(
            tool, dict(arguments), read_timeout_seconds=CALL_TIMEOUT_S
        )
    except Exception as exc:  # the client itself refused the response
        return CallRecord(
            tool, arguments, client_exception=f"{type(exc).__name__}: {exc}"
        )
    text = " ".join(getattr(c, "text", "") for c in (result.content or []))
    structured = result.structured_content
    return CallRecord(
        tool,
        arguments,
        is_error=bool(result.is_error),
        text=text,
        structured=structured,
        advertises_schema=schema is not None,
        schema_errors=() if result.is_error else _schema_errors(schema, structured),
    )


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _signal(seed: int, fs: float = FS, seconds: float = SECONDS) -> np.ndarray:
    """White noise (sigma 0.05) plus a 0.02 amplitude 25 Hz tone, seeded."""
    t = np.arange(int(round(fs * seconds))) / fs
    rng = np.random.default_rng(seed)
    return 0.05 * rng.standard_normal(t.size) + 0.02 * np.sin(2 * np.pi * 25.0 * t)


def write_signal(
    signals_dir: Path,
    name: str,
    values: np.ndarray,
    fs: float = FS,
    unit: str = "g",
    measurement: Optional[dict[str, Any]] = None,
) -> str:
    """A header-less one-column CSV and its companion metadata JSON."""
    np.savetxt(signals_dir / f"{name}.csv", values, fmt="%.17g")
    meta: dict[str, Any] = {"sampling_rate": fs, "signal_unit": unit}
    if measurement is not None:
        meta["measurement"] = measurement
    (signals_dir / f"{name}_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return f"{name}.csv"


def _write_export(signals_dir: Path) -> str:
    """A small instrument-style export: semicolons, decimal commas, a header."""
    t = np.arange(8000) / 4000.0
    lines = ["Sample rate [Hz];4000;", "", "Time;CH1", "s;g"]
    lines += [
        f"{ti:.6f};{0.1 * math.sin(2 * math.pi * 100 * ti):.6f}".replace(".", ",")
        for ti in t
    ]
    (signals_dir / "export.csv").write_bytes(("\r\n".join(lines) + "\r\n").encode())
    return "export.csv"


IMPORT_CONFIG: dict[str, Any] = {
    "schema": "pmm-text-import-config/1",
    "interpretation": {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": "utf-8",
            "delimiter": ";",
            "decimal": ",",
            "header_line": 3,
            "units_line": 4,
        },
        "data_lines": {"first": 5},
        "time_base": {"source": "file_declared", "sampling_rate_hz": "4000"},
        "time_columns": [{"column": 0, "time_unit": "s"}],
        "defaults": {},
        "channels": [
            {
                "column": 1,
                "role": "vibration",
                "quantity": "acceleration",
                "unit": "g",
                "scaling": None,
                "binding": None,
            }
        ],
    },
    "attestation": {
        "bases": {
            "time_base": {"basis": "user_accepted_file_value", "line": 1},
            "channels[1].unit": {"basis": "user_accepted_file_value", "line": 4},
        },
        "overrides": {},
    },
}

MANUAL_TEXT = (
    "Pump P-204 service manual (synthetic test document).\n"
    "Drive-end bearing designation: SKF 6205.\n"
    "Rated speed 1480 rpm. Rated power 55 kW.\n"
    "Lubrication interval 2000 hours.\n"
) * 3

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
MEASUREMENT_SIGNALS = [f"m{i}" for i in range(12)]
#: Every signal id the case table loads; cleared after the sweep even if a
#: signal of the same id existed before it (the loads overwrite).
SWEEP_SIGNAL_IDS = frozenset({"w0", "w1", "w2", "w3", *MEASUREMENT_SIGNALS})


# ---------------------------------------------------------------------------
# The case table: one ordered list of steps, closed over the registry
# ---------------------------------------------------------------------------

Arguments = Union[
    Mapping[str, Any], Callable[[Mapping[str, CallRecord]], Mapping[str, Any]]
]


@dataclass(frozen=True)
class Step:
    """One call of the sweep; ``arguments`` may read earlier steps' records."""

    tool: str
    arguments: Arguments
    key: Optional[str] = None


def _measurement_ids(done: Mapping[str, CallRecord]) -> list[str]:
    history = done["history"].structured
    return [m["measurement_id"] for m in history["asset"]["measurements"]][:3]


STEPS: tuple[Step, ...] = (
    Step("list_signals", {}),
    Step(
        "generate_test_signal",
        {
            "signal_type": "normal",
            "duration": 1.0,
            "sampling_rate": FS,
            "noise_level": 0.1,
            "random_seed": 1,
        },
    ),
    Step("load_signal", {"filepath": "w0.csv", "signal_id": "w0", "overwrite": True}),
    Step("get_signal_info", {"signal_id": "w0"}),
    Step("analyze_fft", {"signal_id": "w0"}),
    Step("analyze_envelope", {"signal_id": "w0"}),
    Step("analyze_statistics", {"signal_id": "w0"}),
    Step("extract_features_from_signal", {"signal_id": "w0"}),
    Step("compute_power_spectral_density", {"signal_id": "w0"}),
    Step("compute_spectrogram_stft", {"signal_id": "w0"}),
    Step("assess_severity", {"signal_id": "w0"}),
    Step(
        "load_signal", {"filepath": ["w1.csv", "w2.csv", "w3.csv"], "overwrite": True}
    ),
    Step(
        "train_anomaly_model",
        {"healthy_signal_ids": ["w1", "w2", "w3"], "model_name": "sweep"},
    ),
    Step("predict_anomalies", {"signal_id": "w0", "model_name": "sweep"}),
    Step("list_machine_manuals", {}),
    Step("extract_manual_specs", {"file_name": "sweep_pump_manual.txt"}),
    Step("read_manual_excerpt", {"file_name": "sweep_pump_manual.txt"}),
    Step("search_bearing_catalog", {"bearing_id": "6205"}),
    Step(
        "search_documentation", {"query": "drive-end bearing designation", "top_k": 2}
    ),
    Step("calculate_bearing_characteristic_frequencies", {**GEOMETRY_6205, "rpm": RPM}),
    Step("check_bearing_faults", {"signal_id": "w0", "rpm": RPM, "bearing_id": "6205"}),
    Step("diagnose_vibration", {"signal_id": "w0", "rpm": RPM, "bearing_id": "6205"}),
    Step(
        "assess_shaft_fault_evidence",
        {"signal_ids": ["w0"], "rpm": RPM, "directions": ["horizontal"]},
    ),
    Step(
        "generate_diagnostic_report",
        {"signal_id": "w0", "rpm": RPM, "bearing_id": "6205"},
    ),
    Step(DOCX_TOOL, {"signal_id": "w0", "sections": {"diagnosis": "synthetic sweep"}}),
    Step("plot_signal", {"signal_id": "w0"}),
    Step("generate_fft_report", {"signal_id": "w0", "rpm": RPM}),
    Step("generate_envelope_report", {"signal_id": "w0", "rpm": RPM}),
    Step("generate_iso_report", {"signal_id": "w0", "rpm": RPM}),
    Step("list_html_reports", {}),
    Step("generate_pca_visualization_report", {"model_name": "sweep"}),
    Step(
        "generate_feature_comparison_report",
        {"signal_groups": {"a": ["w1"], "b": ["w2"]}},
    ),
    Step(
        "estimate_rul",
        {
            "failure_threshold": 5.0,
            "timestamps": [float(i) for i in range(20)],
            "feature_values": [1.0 + 0.1 * i for i in range(20)],
        },
    ),
    Step("analyze_signal_trend", {"signal_id": "w0"}),
    Step(
        "forecast_asset_indicator",
        {
            "observed_at": [
                (T0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
                for i in range(48)
            ],
            "values": [1.0 + 0.01 * ((-1) ** i) for i in range(48)],
            "indicator": "velocity_rms",
            "unit": "mm/s",
            "resolution": "1h",
            "horizon_steps": 4,
        },
    ),
    Step("generate_maintenance_recommendations", {"severity_zone": "B"}),
    Step(
        "declare_measurement_point",
        {
            "asset_id": "A1",
            "measurement_point_id": "p1",
            "bearing_id": "6205",
            "nominal_rpm": RPM,
        },
    ),
    *(
        Step(
            "load_signal",
            {"filepath": f"{name}.csv", "signal_id": name, "overwrite": True},
        )
        for name in MEASUREMENT_SIGNALS
    ),
    Step("get_asset_history", {"asset_id": "A1"}, key="history"),
    Step(
        "declare_healthy_baseline",
        lambda done: {
            "asset_id": "A1",
            "measurement_point_id": "p1",
            "measurement_ids": _measurement_ids(done),
            "declared_by": "wire-sweep",
            "note": "synthetic baseline",
        },
    ),
    Step("assess_asset_change", {"asset_id": "A1", "measurement_point_id": "p1"}),
    Step("inspect_signal_file", {"filepath": "export.csv"}),
    Step(
        "validate_signal_import",
        {"filepath": "export.csv", "config": IMPORT_CONFIG},
        key="validation",
    ),
    Step(
        "import_signal_file",
        lambda done: {
            "filepath": "export.csv",
            "config": IMPORT_CONFIG,
            "import_digest": done["validation"].structured["import_digest"],
        },
    ),
    Step("get_signal_imports", {}),
    Step("clear_signals", {"signal_id": "w0"}),
)

#: Tool names of the case table, in first-call order.
CASE_TOOLS: tuple[str, ...] = tuple(dict.fromkeys(step.tool for step in STEPS))

#: One valid argument set per registered prompt.
PROMPT_CASES: dict[str, dict[str, str]] = {
    "diagnose_bearing": {
        "signal_id": "w1",
        "sampling_rate": "10000",
        "machine_group": "2",
        "support_type": "rigid",
        "rpm": "1500",
        "bpfo": "90",
        "bpfi": "135",
        "bsf": "60",
        "ftf": "10",
    },
    "diagnose_gear": {
        "signal_id": "w1",
        "sampling_rate": "10000",
        "num_teeth": "20",
        "rpm": "1500",
    },
    "quick_diagnostic_report": {"signal_id": "w1"},
    "guided_signal_import": {"filepath": "export.csv"},
}


# ---------------------------------------------------------------------------
# Sandbox and sweep
# ---------------------------------------------------------------------------


@contextmanager
def sweep_sandbox(root: Path) -> Iterator[dict[str, Path]]:
    """Every directory the tools read or write, under ``root``, patched in.

    Patches every by-value import site of DATA_DIR (``patch_data_dir``),
    MODELS_DIR, REPORTS_DIR, RESOURCES_DIR and CACHE_DIR used by the tools,
    sets the ledger and import directories, and resets the RAG index cache;
    everything is restored on exit.
    """
    dirs = {
        name: root / name
        for name in ("signals", "models", "reports", "resources", "ledger", "imports")
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    resources = dirs["resources"]
    (resources / "machine_manuals").mkdir()
    (resources / "bearing_catalogs").mkdir()
    (resources / "cache").mkdir()
    (resources / "machine_manuals" / "sweep_pump_manual.txt").write_text(
        MANUAL_TEXT, encoding="utf-8"
    )
    shutil.copyfile(
        REPO_ROOT / "resources" / "bearing_catalogs" / "common_bearings_catalog.json",
        resources / "bearing_catalogs" / "common_bearings_catalog.json",
    )
    with pytest.MonkeyPatch.context() as mp:
        patch_data_dir(mp, dirs["signals"])
        mp.setenv("PMM_LEDGER_DIR", str(dirs["ledger"]))
        mp.setenv("PMM_IMPORT_DIR", str(dirs["imports"]))
        for module in (diagnostics_tools, diagnosis_pipeline, report_tools):
            mp.setattr(module, "MODELS_DIR", dirs["models"])
        for module in (report_tools, report_generator):
            mp.setattr(module, "REPORTS_DIR", dirs["reports"])
        for module in (diagnostics_tools, document_reader, rag):
            mp.setattr(module, "RESOURCES_DIR", resources)
        for module in (document_reader, rag):
            mp.setattr(module, "CACHE_DIR", resources / "cache")
        mp.setattr(rag, "_index", None)
        # The TF-IDF backend: with the optional faiss and sentence-transformers
        # extras installed, a fresh index build would download an embedding
        # model, and this sweep must not touch the network.
        mp.setattr(rag, "_HAS_FAISS", False)
        yield dirs


def write_sweep_inputs(signals_dir: Path) -> None:
    """The files the case table loads, imports or inspects."""
    for seed, name in enumerate(("w0", "w1", "w2", "w3")):
        write_signal(signals_dir, name, _signal(seed))
    for i, name in enumerate(MEASUREMENT_SIGNALS):
        write_signal(
            signals_dir,
            name,
            _signal(100 + i),
            measurement={
                "asset_id": "A1",
                "measurement_point_id": "p1",
                "acquired_at": (T0 + timedelta(days=i)).isoformat(),
                "rpm": RPM,
                "direction": "horizontal",
            },
        )
    _write_export(signals_dir)


@dataclass
class SweepResult:
    """Everything the client saw during one sweep."""

    registered_tools: list[str]
    registered_prompts: list[str]
    records: list[CallRecord]
    prompts: dict[str, tuple[Optional[str], str]] = field(default_factory=dict)


async def run_sweep(server: MCPServer, steps: Sequence[Step] = STEPS) -> SweepResult:
    """Call every step in order over one in-memory client session."""
    records: list[CallRecord] = []
    done: dict[str, CallRecord] = {}
    async with Client(server, mode="legacy") as client:
        schemas = await output_schemas(client)
        for step in steps:
            try:
                arguments = (
                    step.arguments(done) if callable(step.arguments) else step.arguments
                )
            except Exception as exc:
                record = CallRecord(
                    step.tool, {}, precondition_error=f"{type(exc).__name__}: {exc}"
                )
            else:
                record = await call(client, step.tool, arguments, schemas)
            records.append(record)
            if step.key:
                done[step.key] = record
        listed_prompts = await client.list_prompts()
        prompt_names = [p.name for p in listed_prompts.prompts]
        rendered: dict[str, tuple[Optional[str], str]] = {}
        for name in prompt_names:
            if name not in PROMPT_CASES:
                continue
            try:
                prompt = await client.get_prompt(name, PROMPT_CASES[name])
            except Exception as exc:
                rendered[name] = (f"{type(exc).__name__}: {exc}", "")
                continue
            text = "\n".join(getattr(m.content, "text", "") for m in prompt.messages)
            rendered[name] = (None, text)
    return SweepResult(sorted(schemas), sorted(prompt_names), records, rendered)


def _shipped_server() -> MCPServer:
    from predictive_maintenance_mcp import server

    return server.mcp


@pytest.fixture(scope="module")
def sweep(tmp_path_factory) -> Iterator[SweepResult]:
    """The whole case table, run once over the shipped server."""
    repository = get_repository()
    before = {info["signal_id"] for info in repository.list_signals()}
    root = tmp_path_factory.mktemp("wire-sweep")
    try:
        with sweep_sandbox(root) as dirs:
            write_sweep_inputs(dirs["signals"])
            result = anyio.run(run_sweep, _shipped_server())
        yield result
    finally:
        for info in repository.list_signals():
            signal_id = info["signal_id"]
            if signal_id not in before or signal_id in SWEEP_SIGNAL_IDS:
                repository.clear_signal(signal_id)


def _failures(records: Sequence[CallRecord]) -> list[str]:
    return [
        f"{record.tool} {record.brief()}: {problem}"
        for record in records
        for problem in problems(record, expectation(record.tool))
    ]


# ---------------------------------------------------------------------------
# The real surface
# ---------------------------------------------------------------------------


class TestRegistryClosure:
    """The case table and the registry name the same tools and prompts."""

    def test_registry_is_not_trivially_small(self, sweep):
        assert len(sweep.registered_tools) >= MIN_TOOLS

    def test_every_registered_tool_has_a_case_and_every_case_a_tool(self, sweep):
        assert closure_problems(sweep.registered_tools, CASE_TOOLS) == []

    def test_every_registered_prompt_has_a_case_and_every_case_a_prompt(self, sweep):
        assert (
            closure_problems(sweep.registered_prompts, list(PROMPT_CASES), "prompt")
            == []
        )


class TestEveryToolOverTheWire:
    """One valid call per registered tool meets its expectation."""

    @pytest.mark.parametrize("tool", CASE_TOOLS)
    def test_tool_call_meets_its_expectation(self, sweep, tool):
        records = [record for record in sweep.records if record.tool == tool]
        assert records, f"{tool}: the sweep made no call"
        failures = _failures(records)
        assert failures == [], "\n".join(failures)

    def test_every_registered_tool_was_called_with_no_unexpected_outcome(self, sweep):
        called = {record.tool for record in sweep.records}
        assert sorted(called) == sweep.registered_tools
        assert _failures(sweep.records) == []

    def test_docx_case_follows_the_installed_extra(self, sweep):
        """With python-docx a success, without it the exact documented
        refusal: the case is judged either way, never skipped."""
        (record,) = [r for r in sweep.records if r.tool == DOCX_TOOL]
        if report_generator.HAS_DOCX:
            assert problems(record, (SUCCESS, None)) == []
            assert record.structured["file_name"].endswith(".docx")
        else:
            assert record.is_error
            assert record.text == DOCX_REFUSAL

    @pytest.mark.parametrize("prompt", sorted(PROMPT_CASES))
    def test_prompt_renders(self, sweep, prompt):
        error, text = sweep.prompts[prompt]
        assert error is None, f"{prompt}: {error}"
        assert text.strip(), f"{prompt}: rendered no text"


# ---------------------------------------------------------------------------
# The repaired tools at the wire
# ---------------------------------------------------------------------------


def closed_form_bearing_frequencies(
    num_balls: int, ball_mm: float, pitch_mm: float, contact_deg: float, rpm: float
) -> dict[str, float]:
    """Textbook rolling-element kinematics, computed here independently.

    BPFO = Z/2 fr (1 - r), BPFI = Z/2 fr (1 + r), BSF = Pd/(2 Bd) fr (1 - r^2),
    FTF = fr/2 (1 - r), with r = Bd/Pd cos(alpha) and fr = rpm / 60.
    """
    fr = rpm / 60.0
    r = ball_mm / pitch_mm * math.cos(math.radians(contact_deg))
    return {
        "BPFO": num_balls / 2.0 * fr * (1.0 - r),
        "BPFI": num_balls / 2.0 * fr * (1.0 + r),
        "BSF": pitch_mm / (2.0 * ball_mm) * fr * (1.0 - r * r),
        "FTF": fr / 2.0 * (1.0 - r),
        "shaft_freq_hz": fr,
    }


async def _one_call(
    server: MCPServer, tool: str, arguments: Mapping[str, Any]
) -> CallRecord:
    async with Client(server, mode="legacy") as client:
        schemas = await output_schemas(client)
        return await call(client, tool, arguments, schemas)


async def _load_then_call(
    server: MCPServer, signal_id: str, tool: str, arguments: Mapping[str, Any]
) -> tuple[CallRecord, CallRecord]:
    async with Client(server, mode="legacy") as client:
        schemas = await output_schemas(client)
        loaded = await call(
            client,
            "load_signal",
            {"filepath": f"{signal_id}.csv", "signal_id": signal_id, "overwrite": True},
            schemas,
        )
        return loaded, await call(client, tool, arguments, schemas)


class TestBearingKinematicsAtTheWire:
    """``calculate_bearing_characteristic_frequencies`` over MCP (audit B0a-B0c)."""

    @pytest.mark.parametrize(
        "num_balls, ball_mm, pitch_mm, contact_deg, rpm",
        [(9, 7.94, 39.04, 0.0, 1750.0), (12, 10.0, 60.0, 25.0, 1200.0)],
        ids=["6205-1750rpm", "angular-contact-25deg-1200rpm"],
    )
    def test_valid_call_returns_the_closed_form(
        self, num_balls, ball_mm, pitch_mm, contact_deg, rpm
    ):
        record = anyio.run(
            _one_call,
            _shipped_server(),
            "calculate_bearing_characteristic_frequencies",
            {
                "num_balls": num_balls,
                "ball_diameter_mm": ball_mm,
                "pitch_diameter_mm": pitch_mm,
                "contact_angle_deg": contact_deg,
                "rpm": rpm,
            },
        )
        assert problems(record, (SUCCESS, None)) == []
        assert set(record.structured) == BEARING_FREQUENCY_KEYS
        expected = closed_form_bearing_frequencies(
            num_balls, ball_mm, pitch_mm, contact_deg, rpm
        )
        for key, value in record.structured.items():
            assert type(value) in (int, float), f"{key}: {value!r}"
            assert abs(value - expected[key]) <= 0.02, (key, value, expected[key])


def _band_low_edge_hz(label: str) -> float:
    match = re.match(r"^(\d+(?:\.\d+)?)", label)
    assert match, f"band label without a leading frequency: {label!r}"
    return float(match.group(1))


class TestSpectrogramAtTheWire:
    """``compute_spectrogram_stft`` over MCP (audit A8a-A8c)."""

    def _run(self, sandbox_data_dir, name, values, fs, **stft_arguments):
        write_signal(sandbox_data_dir, name, values, fs=fs)
        try:
            loaded, record = anyio.run(
                _load_then_call,
                _shipped_server(),
                name,
                "compute_spectrogram_stft",
                {"signal_id": name, **stft_arguments},
            )
        finally:
            get_repository().clear_signal(name)
        assert problems(loaded, (SUCCESS, None)) == []
        assert problems(record, (SUCCESS, None)) == []
        return record.structured

    def _assert_band_table(self, bands, fs):
        assert bands, "no band in the table"
        for band in bands:
            assert set(band) == {"band", "energy"}, band
            assert isinstance(band["band"], str), band
            assert type(band["energy"]) in (int, float), band
            assert _band_low_edge_hz(band["band"]) < fs / 2.0, (band, fs / 2.0)

    def test_tone_step_peaks_near_2khz_in_the_second_half(self, sandbox_data_dir):
        fs = 10_000.0
        t = np.arange(int(2 * fs)) / fs
        values = np.where(
            t < 1.0, np.sin(2 * np.pi * 500.0 * t), 3.0 * np.sin(2 * np.pi * 2000.0 * t)
        )
        result = self._run(
            sandbox_data_dir, "stft_step", values, fs, nperseg=1024, noverlap=512
        )
        assert abs(result["max_power_freq_hz"] - 2000.0) <= fs / 1024
        assert result["max_power_time_s"] > 1.0
        self._assert_band_table(result["energy_per_band"], fs)

    def test_low_sampling_rate_has_no_band_above_nyquist(self, sandbox_data_dir):
        fs = 2_000.0
        t = np.arange(int(2 * fs)) / fs
        result = self._run(
            sandbox_data_dir, "stft_low_fs", np.sin(2 * np.pi * 700.0 * t), fs
        )
        self._assert_band_table(result["energy_per_band"], fs)
        assert result["freq_range_hz"][1] <= fs / 2.0


# ---------------------------------------------------------------------------
# Controls: the checks detect what they claim to
# ---------------------------------------------------------------------------


def _toy_server() -> MCPServer:
    """Two tools typed like the bearing calculator, one honouring its
    ``dict[str, float]`` contract and one breaking it the way BLA-1 did."""
    toy = MCPServer("wire-sweep-controls")

    def honest() -> dict[str, float]:
        """Returns only numbers."""
        return {"BPFO": 104.56, "shaft_freq_hz": 29.17}

    def nested() -> dict[str, float]:
        """Returns a nested object under a dict[str, float] annotation."""
        return {"BPFO": 104.56, "input_parameters": {"num_balls": 9}}  # type: ignore[dict-item]

    toy.tool()(honest)
    toy.tool()(nested)
    return toy


class TestControls:
    def test_closure_reports_a_registered_tool_without_a_case(self):
        assert closure_problems([*CASE_TOOLS, "ghost_tool"], CASE_TOOLS) == [
            "registered tool without a case: ghost_tool"
        ]

    def test_closure_reports_a_case_without_a_registered_tool(self):
        assert closure_problems(CASE_TOOLS[1:], CASE_TOOLS) == [
            f"case for an unregistered tool: {CASE_TOOLS[0]}"
        ]

    def test_closure_on_the_real_registry_reports_a_dropped_case(self, sweep):
        dropped = [tool for tool in CASE_TOOLS if tool != DOCX_TOOL]
        assert closure_problems(sweep.registered_tools, dropped) == [
            f"registered tool without a case: {DOCX_TOOL}"
        ]

    def test_output_the_sdk_rejects_is_caught_and_honest_output_passes(self):
        toy = _toy_server()
        honest = anyio.run(_one_call, toy, "honest", {})
        nested = anyio.run(_one_call, toy, "nested", {})
        assert problems(honest, (SUCCESS, None)) == []
        assert problems(nested, (SUCCESS, None)) != []

    def test_broken_output_of_a_registered_tool_is_caught(self, monkeypatch):
        """The shipped calculator, made to return a non-number in a key it does
        not log, still passes a direct Python call and fails the judge over the
        wire: the output contract is checked only on the wire."""

        def broken(**_kwargs):
            return {
                "BPFO": 104.56,
                "BPFI": 157.94,
                "BSF": 68.74,
                "FTF": 11.62,
                "shaft_freq_hz": "n/a",
            }

        monkeypatch.setattr(diagnostics_tools, "calculate_bearing_frequencies", broken)
        arguments = {**GEOMETRY_6205, "rpm": RPM}
        direct = anyio.run(
            functools.partial(
                diagnostics_tools.calculate_bearing_characteristic_frequencies,
                **arguments,
            )
        )
        assert direct["shaft_freq_hz"] == "n/a"
        record = anyio.run(
            _one_call,
            _shipped_server(),
            "calculate_bearing_characteristic_frequencies",
            arguments,
        )
        assert record.is_error or record.schema_errors
        assert problems(record, (SUCCESS, None)) != []

    def test_client_side_schema_check_flags_a_nested_object(self):
        """Independent of whether the installed SDK validates on the server."""
        schema = {"type": "object", "additionalProperties": {"type": "number"}}
        assert _schema_errors(schema, {"BPFO": 104.56}) == ()
        assert (
            _schema_errors(schema, {"BPFO": 104.56, "input_parameters": {"z": 9}}) != ()
        )

    def test_refusal_expectation_needs_the_exact_text(self):
        exact = CallRecord(DOCX_TOOL, {}, is_error=True, text=DOCX_REFUSAL)
        bare = CallRecord(
            DOCX_TOOL, {}, is_error=True, text=f"Error executing tool {DOCX_TOOL}"
        )
        success = CallRecord(DOCX_TOOL, {}, structured={"file_name": "r.docx"})
        assert problems(exact, (REFUSAL, DOCX_REFUSAL)) == []
        assert problems(bare, (REFUSAL, DOCX_REFUSAL)) != []
        assert problems(success, (REFUSAL, DOCX_REFUSAL)) != []

    def test_success_expectation_rejects_errors_exceptions_and_schema_breaks(self):
        assert (
            problems(CallRecord("t", {}, is_error=True, text="x"), (SUCCESS, None))
            != []
        )
        assert (
            problems(CallRecord("t", {}, client_exception="boom"), (SUCCESS, None))
            != []
        )
        assert (
            problems(CallRecord("t", {}, precondition_error="x"), (SUCCESS, None)) != []
        )
        assert (
            problems(CallRecord("t", {}, advertises_schema=True), (SUCCESS, None)) != []
        )
        assert (
            problems(CallRecord("t", {}, schema_errors=("bad",)), (SUCCESS, None)) != []
        )

    def test_docx_refusal_reaches_the_client_word_for_word(
        self, sandbox_data_dir, monkeypatch
    ):
        """Even where python-docx is installed, the refusal branch is proven:
        with the implementation's flag off, the shipped tool returns exactly
        ``DOCX_REFUSAL`` over the wire."""
        monkeypatch.setattr(report_generator, "HAS_DOCX", False)
        write_signal(sandbox_data_dir, "docx_refusal", _signal(7))
        try:
            loaded, record = anyio.run(
                _load_then_call,
                _shipped_server(),
                "docx_refusal",
                DOCX_TOOL,
                {"signal_id": "docx_refusal", "sections": {"diagnosis": "x"}},
            )
        finally:
            get_repository().clear_signal("docx_refusal")
        assert problems(loaded, (SUCCESS, None)) == []
        assert problems(record, (REFUSAL, DOCX_REFUSAL)) == []
