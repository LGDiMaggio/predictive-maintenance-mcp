"""U6 contract tests: module-level importability + single error semantic.

Two rails, one contract:
- failures / misuse  -> raised exceptions (MCPServer converts them into MCP
  ``isError`` responses) with "problem — actionable suggestion" messages;
- legitimate negative outcomes (bearing not in catalog, no trend detected)
  -> TYPED results with a ``suggestion``/``status`` field.

Forbidden everywhere: dicts with an ``"error"`` key returned as success and
JSON built by interpolating exception text into f-strings.
"""

import importlib
import inspect
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.models import BearingCatalogMiss

SRC_MCP_TOOLS = Path(__file__).parent.parent / "src" / "mcp_tools"


@pytest.fixture(scope="module")
def mcp():
    server = MCPServer("test-error-contract")
    register_all(server)
    return server


@pytest.fixture(scope="module")
def tools(mcp):
    return {t.name: t for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    return ctx


@pytest.fixture
def sandbox_dirs(sandbox_data_dir, tmp_path, monkeypatch):
    """Point every directory-bearing module at an empty sandbox.

    DATA_DIR goes through conftest's shared ``sandbox_data_dir`` (the
    four-module patch); the model and report directories are patched here.
    """
    signals_dir = sandbox_data_dir
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    for target in (
        "predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR",
        "predictive_maintenance_mcp.mcp_tools.report_tools.MODELS_DIR",
    ):
        monkeypatch.setattr(target, models_dir)
    for target in (
        "predictive_maintenance_mcp.report_generator.REPORTS_DIR",
        "predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR",
    ):
        monkeypatch.setattr(target, reports_dir)
    return signals_dir


# ---------------------------------------------------------------------------
# R10 (importability): every tool is a module-level function
# ---------------------------------------------------------------------------


class TestModuleLevelTools:
    """Every registered endpoint is importable directly from its module."""

    def test_no_tool_is_a_closure(self, mcp):
        for t in mcp._tool_manager._tools.values():
            assert "<locals>" not in t.fn.__qualname__, (
                f"Tool '{t.name}' is still a closure ({t.fn.__qualname__}) — "
                f"U6 requires module-level functions"
            )

    def test_every_tool_importable_by_name(self, mcp):
        for t in mcp._tool_manager._tools.values():
            module = importlib.import_module(t.fn.__module__)
            assert getattr(module, t.name) is t.fn, (
                f"Tool '{t.name}' is not importable as " f"{t.fn.__module__}.{t.name}"
            )

    def test_prompts_are_module_level(self, mcp):
        for p in mcp._prompt_manager._prompts.values():
            # MCPServer wraps prompt fns (validate_call) — unwrap to compare.
            fn = inspect.unwrap(p.fn)
            assert "<locals>" not in fn.__qualname__
            module = importlib.import_module(fn.__module__)
            assert getattr(module, p.name) is fn

    @pytest.mark.asyncio
    async def test_direct_import_happy_path(self, sandbox_dirs, mock_ctx):
        """load_signal + analyze_fft work via direct import, no MCPServer."""
        from predictive_maintenance_mcp.mcp_tools.analysis_tools import (
            analyze_fft,
        )
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(sandbox_dirs / "direct.csv", index=False, header=False)
        with open(sandbox_dirs / "direct_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

        repo = get_repository()
        try:
            repo.load_signal("direct.csv", overwrite=True)
            result = await analyze_fft(ctx=mock_ctx, signal_id="direct")
            assert abs(result.peak_frequency - 50.0) < 2.0
        finally:
            repo.clear_signal("direct")


# ---------------------------------------------------------------------------
# Blanket: failure paths raise — never an error-shaped dict as success
# ---------------------------------------------------------------------------

# Curated obviously-wrong inputs per tool. Tools omitted here either have no
# reachable failure path without heavy side effects (list_signals,
# generate_test_signal, search_documentation/RAG,
# calculate_bearing_characteristic_frequencies) or express their negative
# outcome as a typed result by design (search_bearing_catalog, clear_signals),
# which is covered separately below.
FAILURE_CASES = {
    "analyze_fft": {"signal_id": "__not_loaded__"},
    "analyze_envelope": {"signal_id": "__not_loaded__"},
    "analyze_statistics": {"signal_id": "__not_loaded__"},
    "extract_features_from_signal": {"signal_id": "__not_loaded__"},
    "compute_power_spectral_density": {"signal_id": "__not_loaded__"},
    "compute_spectrogram_stft": {"signal_id": "__not_loaded__"},
    "train_anomaly_model": {
        "healthy_signal_ids": ["x"],
        "model_name": "../evil",
    },
    "predict_anomalies": {"signal_id": "__not_loaded__", "model_name": "__no_model__"},
    "extract_manual_specs": {"file_name": "__missing__.pdf"},
    "read_manual_excerpt": {"file_name": "__missing__.pdf"},
    "check_bearing_faults": {
        "signal_id": "__not_loaded__",
        "bearing_id": "6205",
        "rpm": 1500.0,
    },
    "assess_severity": {"signal_id": "__not_loaded__"},
    # The unresolvable-rpm refusal needs a LOADED signal, so it lives in
    # TestFailuresRaise.test_diagnose_vibration_unresolvable_rpm_raises.
    "diagnose_vibration": {"signal_id": "__not_loaded__", "rpm": 1500.0},
    "plot_signal": {"signal_id": "__not_loaded__"},
    "generate_fft_report": {"signal_id": "__not_loaded__"},
    "generate_envelope_report": {"signal_id": "__not_loaded__"},
    "generate_iso_report": {"signal_id": "__not_loaded__"},
    "generate_diagnostic_report_docx": {
        "signal_id": "__not_loaded__",
        "sections": {"diagnosis": "text"},
    },
    "list_html_reports": {"file_name": "__missing__.html"},
    "generate_maintenance_recommendations": {
        "severity_zone": "C",
        "fault_types": ["BPFO"],  # acronym, not canonical vocabulary
    },
    "generate_pca_visualization_report": {"model_name": "__no_model__"},
    "estimate_rul": {
        "failure_threshold": 10.0,
        "timestamps": [0.0],
        "feature_values": [1.0],
    },
    "analyze_signal_trend": {"signal_id": "__not_loaded__"},
    "load_signal": {"filepath": "__missing__.csv"},
    "get_signal_info": {"signal_id": "__not_loaded__"},
}


def _payload_has_error_key(value) -> bool:
    if isinstance(value, dict):
        return "error" in value
    if isinstance(value, BaseModel):
        return "error" in value.model_dump()
    return False


class TestFailuresRaise:
    """Known failure paths raise instead of returning error-shaped dicts."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name", sorted(FAILURE_CASES))
    async def test_failure_raises(self, tool_name, tools, sandbox_dirs, mock_ctx):
        tool = tools[tool_name]
        kwargs = dict(FAILURE_CASES[tool_name])
        if "ctx" in inspect.signature(tool.fn).parameters:
            kwargs["ctx"] = mock_ctx

        try:
            result = tool.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            return  # raised — correct rail for a failure

        pytest.fail(
            f"{tool_name} returned {type(result).__name__} instead of "
            f"raising on invalid input"
            + (
                " — and the payload is an error-shaped dict"
                if _payload_has_error_key(result)
                else ""
            )
        )

    @pytest.mark.asyncio
    async def test_raw_binary_missing_declaration_raises(
        self, tools, sandbox_dirs, mock_ctx
    ):
        """A raw .bin with no declared parameters is refused with AE1's shape.

        The refusal must name EVERY missing declaration (sample_format AND
        sampling_rate) in ONE message and teach the companion-file remedy.
        A real file must exist in the sandbox first: FileNotFoundError fires
        BEFORE the missing-declaration refusal, so an absent path would test
        the wrong rail.
        """
        raw = np.zeros(64, dtype="<f4")
        (sandbox_dirs / "undeclared.bin").write_bytes(raw.tobytes())

        with pytest.raises(ValueError) as exc_info:
            await tools["load_signal"].fn(ctx=mock_ctx, filepath="undeclared.bin")

        msg = str(exc_info.value)
        assert "sample_format" in msg
        assert "sampling_rate" in msg
        assert "_metadata.json" in msg  # the companion-file alternative

    @pytest.mark.asyncio
    async def test_invalid_measurement_object_raises(
        self, tools, sandbox_dirs, mock_ctx
    ):
        """A readable companion whose "measurement" object is invalid is a
        refusal (raised), not a warning: a declared identity that breaks the
        contract is misuse, unlike an unreadable companion, which loads with
        a ``companion_warning``. Lives here rather than in FAILURE_CASES
        because that table is keyed by tool name (load_signal already holds
        its missing-file case) and the sandbox file must exist first.
        """
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        pd.DataFrame([0.1, 0.2, 0.3]).to_csv(
            sandbox_dirs / "half.csv", index=False, header=False
        )
        with open(sandbox_dirs / "half_metadata.json", "w") as f:
            json.dump({"sampling_rate": 1000, "measurement": {"asset_id": "P-101"}}, f)

        with pytest.raises(ValueError) as exc_info:
            await tools["load_signal"].fn(ctx=mock_ctx, filepath="half.csv")

        msg = str(exc_info.value)
        assert "measurement_point_id" in msg
        assert "acquired_at" in msg
        assert "half_metadata.json" in msg
        loaded = [s["signal_id"] for s in get_repository().list_signals()]
        assert "half" not in loaded  # refused BEFORE insertion

    @pytest.mark.asyncio
    async def test_diagnose_vibration_unresolvable_rpm_raises(
        self, tools, sandbox_dirs, mock_ctx
    ):
        """rpm with no source anywhere is a refusal naming the three remedies.

        Since rpm became optional, diagnose_vibration resolves it from the
        call, then the companion's "measurement" object, then the point's
        nominal_rpm in the asset ledger; with none of them it must raise,
        never default. Lives here rather than in FAILURE_CASES because that
        table is keyed by tool name (diagnose_vibration already holds its
        not-loaded case) and a LOADED signal is needed: on an unknown id the
        not-loaded rail fires first and would test the wrong refusal.
        """
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        fs = 10000
        sig = 0.1 * np.random.default_rng(3).standard_normal(fs)
        pd.DataFrame(sig).to_csv(sandbox_dirs / "no_rpm.csv", index=False, header=False)
        with open(sandbox_dirs / "no_rpm_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)  # no identity

        repo = get_repository()
        try:
            repo.load_signal("no_rpm.csv", overwrite=True)
            with pytest.raises(ValueError) as exc_info:
                await tools["diagnose_vibration"].fn(ctx=mock_ctx, signal_id="no_rpm")
        finally:
            repo.clear_signal("no_rpm")

        msg = str(exc_info.value)
        assert "rpm=" in msg  # remedy 1: the explicit argument
        assert '"measurement"' in msg  # remedy 2: the companion's object
        assert "nominal_rpm" in msg and "declare_measurement_point" in msg  # 3
        assert "error" not in msg.lower().split("cannot resolve")[0]

    @pytest.mark.asyncio
    async def test_docx_missing_dependency_raises(
        self, tools, sandbox_dirs, mock_ctx, monkeypatch
    ):
        """Missing python-docx is a failure — raised, not an error dict."""
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        monkeypatch.setattr(
            "predictive_maintenance_mcp.report_generator.HAS_DOCX", False
        )
        pd.DataFrame([0.1, 0.2, 0.3]).to_csv(
            sandbox_dirs / "docx_sig.csv", index=False, header=False
        )
        repo = get_repository()
        try:
            repo.load_signal("docx_sig.csv", overwrite=True)
            with pytest.raises(ValueError, match="python-docx"):
                await tools["generate_diagnostic_report_docx"].fn(
                    ctx=mock_ctx,
                    signal_id="docx_sig",
                    sections={"diagnosis": "text"},
                )
        finally:
            repo.clear_signal("docx_sig")


# ---------------------------------------------------------------------------
# Typed results for legitimate negative outcomes
# ---------------------------------------------------------------------------


class TestTypedNegativeOutcomes:
    @pytest.mark.asyncio
    async def test_bearing_catalog_miss_is_typed(self, tools, mock_ctx):
        result = await tools["search_bearing_catalog"].fn(
            bearing_id="ZZZ_NOT_A_BEARING_999", ctx=mock_ctx
        )
        assert isinstance(result, BearingCatalogMiss)
        assert result.status == "not_found"
        assert result.suggestion  # actionable next step present
        assert "6205" in result.catalog_contains
        # No invented geometry and no 'error' key masquerading as success
        dumped = result.model_dump()
        assert "error" not in dumped
        for key in ("num_balls", "ball_diameter_mm", "pitch_diameter_mm"):
            assert key not in dumped


# ---------------------------------------------------------------------------
# Source-level guards (cheap tripwires for the audit's offender patterns)
# ---------------------------------------------------------------------------


class TestSourceGuards:
    def test_no_monolith_imports_in_mcp_tools(self):
        # Needle built by concatenation so this guard file itself never
        # shows up in repo-wide greps for the removed monolith's name.
        monolith_name = "machinery_diagnostics" + "_server"
        offenders = [
            p.name
            for p in SRC_MCP_TOOLS.glob("*.py")
            if monolith_name in p.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            f"mcp_tools must not import from the deprecated monolith: " f"{offenders}"
        )

    def test_no_fstring_json_error_returns(self):
        pattern = re.compile(r"f['\"]\{\{\s*\"error\"")
        offenders = [
            p.name
            for p in SRC_MCP_TOOLS.glob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        ]
        assert offenders == [], (
            f"JSON must never be built via f-string interpolation: " f"{offenders}"
        )

    def test_no_error_key_dict_literals_returned(self):
        """No dict literal with an 'error' key anywhere in mcp_tools.

        Legitimate negative outcomes are typed models; failures raise.
        AST-based so comments and docstrings cannot false-positive.
        """
        import ast

        offenders: dict[str, int] = {}
        for path in SRC_MCP_TOOLS.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            hits = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                and any(
                    isinstance(k, ast.Constant) and k.value == "error"
                    for k in node.keys
                )
            )
            if hits:
                offenders[path.name] = hits
        assert offenders == {}, f"error-keyed dict literals found in: {offenders}"
