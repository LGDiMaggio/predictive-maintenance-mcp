"""W1-W8 workflow walkability on the FINAL U9 surface (33 tools).

Each test walks one of the signal-flow workflows end-to-end using ONLY
registered tools with their final names and signatures — proving the
consolidated surface has no dead ends:

- W1 bearing diagnosis:  load -> stats -> envelope -> check_bearing_faults
                         -> assess_severity
- W2 gear diagnosis:     load -> fft -> check_bearing_faults(GMF)
- W3 quick screening:    generate_test_signal -> stats -> fft
- W4 anomaly ML:         batch load -> train -> predict (bounded output)
- W5 report generation:  fft report x2 -> list_html_reports (+ file_name)
- W6 prognostics:        analyze_signal_trend -> estimate_rul (multi-measure)
- W7 documentation:      list_machine_manuals -> read_manual_excerpt
- W8 signal management:  generate_test_signal -> get_signal_info ->
                         list_signals(disk/memory) -> clear_signals
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools():
    mcp = MCPServer("workflow-walkability")
    register_all(mcp)
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Sandboxed data/models/reports/resources dirs + clean repository."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    resources_dir = tmp_path / "resources"
    (resources_dir / "machine_manuals").mkdir(parents=True)

    for target in (
        "predictive_maintenance_mcp.config.DATA_DIR",
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR",
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR",
        "predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR",
    ):
        monkeypatch.setattr(target, signals_dir)
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
    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.diagnostics_tools.RESOURCES_DIR",
        resources_dir,
    )

    repo = get_repository()
    repo.clear_all()
    yield {
        "signals": signals_dir,
        "models": models_dir,
        "reports": reports_dir,
        "resources": resources_dir,
        "repo": repo,
    }
    repo.clear_all()


def _write_signal(signals_dir, name, sig, fs, unit="g", extra_meta=None):
    pd.DataFrame(sig).to_csv(signals_dir / name, index=False, header=False)
    meta = {"sampling_rate": fs, "signal_unit": unit}
    if extra_meta:
        meta.update(extra_meta)
    stem = Path(name).stem
    with open(signals_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# W1 — bearing diagnosis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w1_bearing_diagnosis_walkable(tools, sandbox, mock_ctx):
    fs = 10000
    t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))
    _write_signal(sandbox["signals"], "w1.csv", sig, fs)

    info = await tools["load_signal"](ctx=mock_ctx, filepath="w1.csv")
    sid = info.signal_id

    stats = tools["analyze_statistics"](signal_id=sid)
    assert stats.rms > 0

    env = await tools["analyze_envelope"](ctx=mock_ctx, signal_id=sid)
    assert env.filter_band == (500.0, 5000.0)

    faults = await tools["check_bearing_faults"](
        ctx=mock_ctx, signal_id=sid, bearing_id="6205", rpm=1800.0
    )
    assert len(faults.fault_checks) == 4

    sev = await tools["assess_severity"](
        ctx=mock_ctx, signal_id=sid, machine_group=2, support_type="rigid"
    )
    assert sev.zone in ("A", "B", "C", "D")

    diag = await tools["diagnose_vibration"](
        ctx=mock_ctx, signal_id=sid, rpm=1800.0, bearing_id="6205"
    )
    assert diag.evidence_strength in ("none", "weak", "moderate", "strong")


# ---------------------------------------------------------------------------
# W2 — gear diagnosis (GMF via explicit frequencies)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w2_gear_diagnosis_walkable(tools, sandbox, mock_ctx):
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    gmf = 350.0
    sig = np.sin(2 * np.pi * gmf * t) + 0.05 * np.random.randn(len(t))
    _write_signal(sandbox["signals"], "w2.csv", sig, fs)

    info = await tools["load_signal"](ctx=mock_ctx, filepath="w2.csv")

    fft = await tools["analyze_fft"](ctx=mock_ctx, signal_id=info.signal_id)
    assert abs(fft.peak_frequency - gmf) < 2.0

    gear = await tools["check_bearing_faults"](
        ctx=mock_ctx,
        signal_id=info.signal_id,
        rpm=1480.0,
        frequencies={"GMF": gmf},
    )
    assert gear.fault_checks[0].fault_type == "GMF"
    assert gear.fault_checks[0].fault_type_canonical is None


# ---------------------------------------------------------------------------
# W3 — quick screening on a generated test signal (zero manual steps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w3_screening_on_generated_signal(tools, sandbox, mock_ctx):
    """generate_test_signal returns a StoredSignalInfo that is immediately
    analyzable AND ISO-assessable — no load_signal, no metadata edits."""
    info = await tools["generate_test_signal"](
        signal_type="bearing_fault",
        duration=1.0,
        sampling_rate=10000.0,
        random_seed=7,
        ctx=mock_ctx,
    )
    assert info.sampling_rate == 10000.0
    assert info.signal_unit == "g"

    stats = tools["analyze_statistics"](signal_id=info.signal_id)
    assert stats.rms > 0

    fft = await tools["analyze_fft"](ctx=mock_ctx, signal_id=info.signal_id)
    assert fft.peak_frequency > 0

    sev = await tools["assess_severity"](ctx=mock_ctx, signal_id=info.signal_id)
    assert sev.status == "assessed"
    assert sev.zone in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# W4 — anomaly train + predict (bounded output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w4_train_predict_walkable_and_bounded(tools, sandbox, mock_ctx):
    fs = 10000
    files = []
    rng = np.random.default_rng(0)
    for i in range(3):
        sig = 0.05 * rng.standard_normal(fs)
        _write_signal(sandbox["signals"], f"w4_train_{i}.csv", sig, fs)
        files.append(f"w4_train_{i}.csv")
    infos = await tools["load_signal"](ctx=mock_ctx, filepath=files)
    ids = [i.signal_id for i in infos]

    trained = await tools["train_anomaly_model"](
        healthy_signal_ids=ids,
        model_name="w4_model",
        segment_duration=0.1,
        ctx=mock_ctx,
    )
    assert trained.model_name == "w4_model"

    # A LONG test signal (30 s @ 10 kHz -> 599 segments at 0.1 s / 50%):
    # the response must stay bounded regardless of segment count.
    long_sig = 0.05 * rng.standard_normal(30 * fs)
    _write_signal(sandbox["signals"], "w4_long.csv", long_sig, fs)
    await tools["load_signal"](ctx=mock_ctx, filepath="w4_long.csv")

    pred = await tools["predict_anomalies"](
        signal_id="w4_long", model_name=trained.model_name, ctx=mock_ctx
    )
    assert pred.num_segments > 500
    assert not hasattr(pred, "predictions")
    assert not hasattr(pred, "anomaly_scores")
    assert len(pred.worst_segments) <= 10
    # Bounded output: the serialized payload stays small (< 2 KB) even for
    # thousands of segments.
    assert len(pred.model_dump_json()) < 2048

    # Missing model error lists the models actually on disk.
    with pytest.raises(FileNotFoundError) as exc:
        await tools["predict_anomalies"](
            signal_id="w4_long", model_name="__ghost__", ctx=mock_ctx
        )
    assert "w4_model" in str(exc.value)


# ---------------------------------------------------------------------------
# W5 — report generation (timestamped files, listing, metadata route)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w5_reports_walkable(tools, sandbox, mock_ctx):
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 50 * t)
    _write_signal(sandbox["signals"], "w5.csv", sig, fs)
    info = await tools["load_signal"](ctx=mock_ctx, filepath="w5.csv")

    r1 = await tools["generate_fft_report"](signal_id=info.signal_id)
    r2 = await tools["generate_fft_report"](signal_id=info.signal_id)
    assert r1["file_name"] != r2["file_name"]

    listed = tools["list_html_reports"]()
    names = {r["file_name"] for r in listed}
    assert {r1["file_name"], r2["file_name"]} <= names

    meta = tools["list_html_reports"](file_name=r1["file_name"])
    assert meta["metadata"]["report_type"] == "fft_spectrum"

    # W5 security regression: traversal through the merged metadata route
    # is rejected (U1 fix preserved across the merge).
    with pytest.raises(ValueError, match="Invalid report filename"):
        tools["list_html_reports"](file_name="../../evil.html")


# ---------------------------------------------------------------------------
# W6 — prognostics: within-recording screening feeds multi-measure RUL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w6_trend_to_rul_walkable(tools, sandbox, mock_ctx):
    fs = 5000
    t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
    growing = (0.5 + 0.5 * t / t[-1]) * np.random.default_rng(1).standard_normal(
        len(t)
    )
    _write_signal(sandbox["signals"], "w6.csv", growing, fs)
    info = await tools["load_signal"](ctx=mock_ctx, filepath="w6.csv")

    trend = await tools["analyze_signal_trend"](
        ctx=mock_ctx, signal_id=info.signal_id, feature_name="rms"
    )
    assert trend.analysis_scope == "within_recording_screening"
    assert len(trend.feature_series) > 0

    # Each recording contributes ONE point over time; here we simulate the
    # accumulated series a data collector would produce.
    rul = await tools["estimate_rul"](
        ctx=mock_ctx,
        failure_threshold=4.5,  # ISO 10816-3:2009 C/D boundary, group 2 rigid
        timestamps=[0.0, 100.0, 200.0, 300.0, 400.0],
        feature_values=[1.0, 1.6, 2.3, 2.9, 3.6],
        method="linear",
    )
    assert rul.status == "estimated"
    assert rul.rul is not None and rul.rul > 0


# ---------------------------------------------------------------------------
# W7 — documentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w7_documentation_walkable(tools, sandbox, mock_ctx):
    manuals = sandbox["resources"] / "machine_manuals"
    (manuals / "pump_manual.txt").write_text(
        "Bearing: SKF 6205-2RS\nOperating speed: 1800 RPM", encoding="utf-8"
    )

    listing = await tools["list_machine_manuals"](ctx=mock_ctx)
    assert [m["filename"] for m in listing] == ["pump_manual.txt"]

    text = await tools["read_manual_excerpt"](
        ctx=mock_ctx, file_name="pump_manual.txt"
    )
    assert "SKF 6205-2RS" in text

    catalog_hit = await tools["search_bearing_catalog"](
        ctx=mock_ctx, bearing_id="SKF 6205-2RS"
    )
    assert catalog_hit["num_balls"] == 9


# ---------------------------------------------------------------------------
# W8 — signal management lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w8_signal_management_walkable(tools, sandbox, mock_ctx):
    generated = await tools["generate_test_signal"](
        signal_type="normal", duration=0.5, sampling_rate=5000.0, ctx=mock_ctx
    )

    disk = await tools["list_signals"](ctx=mock_ctx, scope="disk")
    assert disk["count"] >= 1

    memory = await tools["list_signals"](ctx=mock_ctx, scope="memory")
    assert any(
        s["signal_id"] == generated.signal_id for s in memory["signals"]
    )

    info = await tools["get_signal_info"](
        ctx=mock_ctx, signal_id=generated.signal_id
    )
    assert info.source_metadata["signal_type"] == "normal"
    assert info.signal_unit == "g"

    one = await tools["clear_signals"](
        ctx=mock_ctx, signal_id=generated.signal_id
    )
    assert one["status"] == "removed"

    remaining = await tools["clear_signals"](ctx=mock_ctx)
    assert remaining["cleared_count"] >= 0
