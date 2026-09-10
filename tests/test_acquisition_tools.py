"""Tests for MCP acquisition tools (ISO 13374 Block 1).

``TestLoadSignalLedger`` (U5) drives the asset-ledger registration through
``load_signal`` end to end: the returned measurement block carries the
outcome, the repository's cached block keeps the identity only, and a
ledger problem never fails a load.
"""

import functools
import inspect
import json
import shutil
import typing
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from mcp.server.mcpserver import MCPServer

from _golden_signals import golden_signals
from _synthetic_sequence import SEQUENCE_LENGTH, build_measurement_sequence
from _synthetic_sequence import ASSET_ID as SEQUENCE_ASSET
from _synthetic_sequence import companion_path as sequence_companion
from conftest import write_raw_file
from predictive_maintenance_mcp.asset_ledger import service as ledger_service
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    make_event,
)
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools import acquisition_tools, analysis_tools
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import register
from predictive_maintenance_mcp.signal_acquisition import repository as repo_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp():
    """Create a MCPServer test instance with acquisition tools registered."""
    server = MCPServer("test-acquisition")
    register(server)
    return server


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Set up a temp data directory with synthetic signals."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    # Create a sine-wave CSV
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 50 * t)
    pd.DataFrame(sig).to_csv(signals_dir / "test_sine.csv", index=False, header=False)

    # Create metadata
    meta = {"sampling_rate": fs, "signal_unit": "g"}
    with open(signals_dir / "test_sine_metadata.json", "w") as f:
        json.dump(meta, f)

    # Create a subfolder with another signal
    sub = signals_dir / "real_train"
    sub.mkdir()
    pd.DataFrame(sig[:5000]).to_csv(sub / "baseline_1.csv", index=False, header=False)

    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", signals_dir
    )
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir
    )
    return signals_dir


@pytest.fixture
def mock_ctx():
    """Create a mock MCP Context."""
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Registered tools – tested via MCP server internals
# ---------------------------------------------------------------------------


class TestListSignals:
    """Tests for the merged list_signals tool (scope='disk'|'memory')."""

    @pytest.mark.asyncio
    async def test_lists_disk_files_by_default(self, mcp, data_dir):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        result = await tools["list_signals"]()
        assert result["scope"] == "disk"
        assert "test_sine.csv" in result["files"]
        assert "real_train/baseline_1.csv" in result["files"]
        assert result["count"] == len(result["files"])

    @pytest.mark.asyncio
    async def test_memory_scope_lists_loaded_ids(self, mcp, data_dir, mock_ctx):
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        repo = get_repository()
        repo.clear_all()
        try:
            await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", signal_id="mem1"
            )
            result = await tools["list_signals"](ctx=mock_ctx, scope="memory")
            assert result["scope"] == "memory"
            assert result["count"] == 1
            assert result["signals"][0]["signal_id"] == "mem1"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_empty_directory(self, mcp, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", empty_dir
        )
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        result = await tools["list_signals"](scope="disk")
        assert result["count"] == 0
        assert result["files"] == []

    def test_old_lifecycle_tools_gone(self, mcp):
        names = {t.name for t in mcp._tool_manager._tools.values()}
        assert "list_stored_signals" not in names
        assert "clear_signal" not in names
        assert "clear_all_signals" not in names


class TestGenerateTestSignal:
    """generate_test_signal: closed loop — metadata + auto-registration."""

    @pytest.mark.asyncio
    async def test_returns_stored_signal_info_immediately_usable(
        self, mcp, data_dir, mock_ctx
    ):
        """U9 loop closure: the returned StoredSignalInfo has a declared
        rate AND unit and the id is already loaded (ISO-assessable with no
        manual steps)."""
        from predictive_maintenance_mcp.models import StoredSignalInfo
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        repo = get_repository()
        repo.clear_all()
        try:
            result = await tools["generate_test_signal"](
                signal_type="bearing_fault",
                duration=1.0,
                sampling_rate=10000.0,
                noise_level=0.1,
                random_seed=42,
                ctx=mock_ctx,
            )
            assert isinstance(result, StoredSignalInfo)
            assert result.sampling_rate == 10000.0
            assert result.signal_unit == "g"
            # Auto-registered: the array is retrievable by id right away.
            assert len(repo.get_signal(result.signal_id)) == 10000
            # Companion metadata written next to the CSV.
            meta_files = list(data_dir.glob("test_bearing_fault_*_metadata.json"))
            assert meta_files
            meta = json.loads(meta_files[0].read_text())
            assert meta["sampling_rate"] == 10000.0
            assert meta["signal_unit"] == "g"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_consecutive_runs_never_collide(self, mcp, data_dir, mock_ctx):
        """Timestamped filenames: two runs → two distinct files and ids."""
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        repo = get_repository()
        repo.clear_all()
        try:
            a = await tools["generate_test_signal"](
                signal_type="normal", duration=0.2, sampling_rate=5000.0
            )
            b = await tools["generate_test_signal"](
                signal_type="normal", duration=0.2, sampling_rate=5000.0
            )
            assert a.signal_id != b.signal_id
            assert a.filepath != b.filepath
        finally:
            repo.clear_all()


class TestSignalRepository:
    """Tests for load_signal, list_signals(memory), get_signal_info, clear_signals."""

    @pytest.mark.asyncio
    async def test_load_and_list(self, mcp, data_dir, mock_ctx):
        # Get tool functions
        tools = {}
        for tool in mcp._tool_manager._tools.values():
            tools[tool.name] = tool.fn

        # Load signal
        result = await tools["load_signal"](
            ctx=mock_ctx, filepath="test_sine.csv", signal_id="sine1"
        )
        assert result.signal_id == "sine1"
        assert result.num_samples == 10000

        # List stored
        stored = await tools["list_signals"](ctx=mock_ctx, scope="memory")
        assert stored["count"] >= 1
        assert any(s["signal_id"] == "sine1" for s in stored["signals"])

        # Get info
        info = await tools["get_signal_info"](ctx=mock_ctx, signal_id="sine1")
        assert info.signal_id == "sine1"

        # Clear one
        cleared = await tools["clear_signals"](ctx=mock_ctx, signal_id="sine1")
        assert cleared["status"] == "removed"
        assert cleared["cleared_count"] == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        await tools["load_signal"](
            ctx=mock_ctx, filepath="test_sine.csv", signal_id="s1"
        )
        result = await tools["clear_signals"](ctx=mock_ctx)
        assert result["cleared_count"] >= 1

    @pytest.mark.asyncio
    async def test_clear_unknown_id_reports_not_found(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        result = await tools["clear_signals"](ctx=mock_ctx, signal_id="__ghost__")
        assert result["status"] == "not_found"
        assert result["cleared_count"] == 0

    @pytest.mark.asyncio
    async def test_get_signal_info_exposes_source_metadata(
        self, mcp, data_dir, mock_ctx
    ):
        """U9 loop closure: the companion metadata (rpm/shaft_speed, ...)
        is fully exposed via get_signal_info — no resource needed."""
        meta_path = data_dir / "test_sine_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update({"shaft_speed": 1797, "rpm": 1797, "BPFO": 107.36})
        meta_path.write_text(json.dumps(meta))

        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        repo = get_repository()
        repo.clear_all()
        try:
            await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", signal_id="meta_rich"
            )
            info = await tools["get_signal_info"](ctx=mock_ctx, signal_id="meta_rich")
            assert info.source_metadata["shaft_speed"] == 1797
            assert info.source_metadata["rpm"] == 1797
            assert info.source_metadata["BPFO"] == 107.36
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_get_signal_info_unknown_id_raises(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="load_signal"):
            await tools["get_signal_info"](ctx=mock_ctx, signal_id="__nope__")


class TestLoadSignalIdsAndBatch:
    """U8: relative-path default ids, explicit collisions, atomic batch."""

    @pytest.mark.asyncio
    async def test_default_id_from_relative_path(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx, filepath="real_train/baseline_1.csv"
            )
            assert result.signal_id == "real_train_baseline_1"
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_reload_collision_requires_overwrite(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            await tools["load_signal"](ctx=mock_ctx, filepath="test_sine.csv")
            with pytest.raises(ValueError, match="overwrite=True"):
                await tools["load_signal"](ctx=mock_ctx, filepath="test_sine.csv")
            result = await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", overwrite=True
            )
            assert result.signal_id == "test_sine"
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_batch_load_returns_all_infos(self, mcp, data_dir, mock_ctx):
        """Batch form: list in → list of StoredSignalInfo out, one per file."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            results = await tools["load_signal"](
                ctx=mock_ctx,
                filepath=["test_sine.csv", "real_train/baseline_1.csv"],
                sampling_rate=10000.0,
            )
            assert isinstance(results, list)
            assert [r.signal_id for r in results] == [
                "test_sine",
                "real_train_baseline_1",
            ]
            stored = await tools["list_signals"](ctx=mock_ctx, scope="memory")
            assert stored["count"] == 2
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_batch_missing_file_is_atomic(self, mcp, data_dir, mock_ctx):
        """One bad entry → one error naming it, NOTHING loaded."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            with pytest.raises(ValueError, match="__missing__.csv"):
                await tools["load_signal"](
                    ctx=mock_ctx,
                    filepath=["test_sine.csv", "__missing__.csv"],
                )
            stored = await tools["list_signals"](ctx=mock_ctx, scope="memory")
            assert stored["count"] == 0
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_batch_empty_list_raises(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="empty list"):
            await tools["load_signal"](ctx=mock_ctx, filepath=[])

    @pytest.mark.asyncio
    async def test_batch_rejects_custom_signal_id(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="batch"):
            await tools["load_signal"](
                ctx=mock_ctx,
                filepath=["test_sine.csv"],
                signal_id="custom",
            )


class TestLoadSignalUnit:
    """load_signal signal_unit declaration (U5: no ISO verdicts on guessed units)."""

    @pytest.mark.asyncio
    async def test_unit_from_metadata(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", signal_id="meta_unit"
            )
            assert result.signal_unit == "g"  # declared in companion metadata
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_explicit_unit_overrides_metadata(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx,
                filepath="test_sine.csv",
                signal_id="param_unit",
                signal_unit="mm/s",
            )
            assert result.signal_unit == "mm/s"  # declared > metadata ('g')
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_no_declaration_is_none(self, mcp, data_dir, mock_ctx):
        """Without any declaration the unit stays None — never guessed."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx,
                filepath="real_train/baseline_1.csv",
                signal_id="no_unit",
            )
            assert result.signal_unit is None
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_invalid_unit_raises(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="signal_unit"):
            await tools["load_signal"](
                ctx=mock_ctx,
                filepath="test_sine.csv",
                signal_id="bad_unit",
                signal_unit="furlongs",
            )


class TestLoadSignalRaw:
    """U3: raw binary (.bin/.raw/.dat) declaration via the load_signal tool."""

    @pytest.mark.asyncio
    async def test_bin_with_full_explicit_params(self, mcp, data_dir, mock_ctx):
        """Full explicit declaration → StoredSignalInfo with raw_format."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        fs = 10000.0
        t = np.arange(0, 0.1, 1 / fs)
        arr = write_raw_file(data_dir / "raw_full.bin", np.sin(2 * np.pi * 50 * t))
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx,
                filepath="raw_full.bin",
                sampling_rate=fs,
                sample_format="float32",
                byte_order="little",
                n_channels=1,
                channel_index=0,
                header_offset=0,
            )
            assert result.signal_id == "raw_full"
            assert result.num_samples == len(arr)
            assert result.sampling_rate == fs
            assert result.raw_format is not None
            assert result.raw_format["sample_format"] == "float32"
            assert result.raw_format["byte_order"] == "little"
            assert result.raw_format["n_channels"] == 1
            assert result.raw_format["channel_index"] == 0
            assert result.raw_format["header_offset"] == 0
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_batch_bin_with_shared_params(self, mcp, data_dir, mock_ctx):
        """One raw parameter set broadcasts to every .bin in the batch."""
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        write_raw_file(data_dir / "raw_b1.bin", [100, -100, 50, -50], dtype="<i2")
        write_raw_file(data_dir / "raw_b2.bin", [10, -10, 20, -20], dtype="<i2")
        try:
            results = await tools["load_signal"](
                ctx=mock_ctx,
                filepath=["raw_b1.bin", "raw_b2.bin"],
                sampling_rate=5000.0,
                sample_format="int16",
                scale_factor=0.5,
            )
            assert [r.signal_id for r in results] == ["raw_b1", "raw_b2"]
            for r in results:
                assert r.raw_format is not None
                assert r.raw_format["sample_format"] == "int16"
                assert r.raw_format["scale_factor"] == 0.5
            # scale_factor flowed through the tool layer to the decoder.
            np.testing.assert_allclose(
                get_repository().get_signal("raw_b1"), [50.0, -50.0, 25.0, -25.0]
            )
        finally:
            await tools["clear_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_sample_format_on_csv_is_contradiction(self, mcp, data_dir, mock_ctx):
        """Declaring a raw dtype for a self-describing format is refused."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="self-describing"):
            await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", sample_format="float32"
            )

    @pytest.mark.asyncio
    async def test_invalid_sample_format_rejected_by_validation(
        self, mcp, data_dir, mock_ctx
    ):
        """A value outside the Literal vocabulary never reaches the decoder.

        Invoked through the MCP validation layer (tool.run); the exact
        exception type is the server library's business — the contract
        asserted here is only that the call errors instead of loading.
        """
        write_raw_file(data_dir / "raw_enum.bin", [1.0, 2.0])
        tool = mcp._tool_manager._tools["load_signal"]
        with pytest.raises(Exception):
            await tool.run(
                {
                    "filepath": "raw_enum.bin",
                    "sampling_rate": 1000.0,
                    "sample_format": "float99",
                },
                context=mock_ctx,
            )


class TestRawLiteralVocabularySync:
    """The tool's Literal vocabularies and the loader's VALID_* constants
    are the same closed sets (single source of truth, no drift) — the same
    guard pattern as the fault-type Literal in decision_support_tools."""

    @staticmethod
    def _literal_values(annotation) -> set:
        """Unwrap Optional[Literal[...]] to the set of Literal values."""
        literal = next(
            a
            for a in typing.get_args(annotation)
            if typing.get_origin(a) is typing.Literal
        )
        return set(typing.get_args(literal))

    def test_sample_format_literal_matches_loader_vocabulary(self):
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
        from predictive_maintenance_mcp.signal_acquisition.loaders import (
            VALID_SAMPLE_FORMATS,
        )

        ann = inspect.signature(load_signal).parameters["sample_format"].annotation
        assert self._literal_values(ann) == set(VALID_SAMPLE_FORMATS)

    def test_byte_order_literal_matches_loader_vocabulary(self):
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
        from predictive_maintenance_mcp.signal_acquisition.loaders import (
            VALID_BYTE_ORDERS,
        )

        ann = inspect.signature(load_signal).parameters["byte_order"].annotation
        assert self._literal_values(ann) == set(VALID_BYTE_ORDERS)


# ---------------------------------------------------------------------------
# U5: registration in the asset ledger at load time
# ---------------------------------------------------------------------------

ASSET = "P-101"
OTHER_ASSET = "P-102"
POINT = "motor_de_h"
ACQUIRED_AT = "2026-08-20T13:42:00+02:00"
LEDGER_BLOCK_KEYS = set(ledger_service.LOAD_OUTCOME_KEYS)


def _write_signal(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        np.savetxt(fh, values, fmt="%.8f")


def _write_companion(path: Path, payload) -> Path:
    companion = path.parent / f"{path.stem}_metadata.json"
    if isinstance(payload, str):
        companion.write_text(payload, encoding="utf-8")
    else:
        companion.write_text(json.dumps(payload), encoding="utf-8")
    return companion


def _identity(**overrides) -> dict:
    measurement = {
        "asset_id": ASSET,
        "measurement_point_id": POINT,
        "acquired_at": ACQUIRED_AT,
        "direction": "horizontal",
        "sensor_id": "ACC01",
    }
    measurement.update(overrides)
    return {key: value for key, value in measurement.items() if value is not None}


def _companion(measurement=None, **top_level) -> dict:
    payload = {"sampling_rate": 10000, "signal_unit": "g", **top_level}
    if measurement is not None:
        payload["measurement"] = measurement
    return payload


def _noise(seed: int, n: int = 10000) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(n)


def _place(data_dir: Path, name: str, values: np.ndarray, companion: dict) -> Path:
    """Write ``<name>`` (a data_dir-relative path) with its companion."""
    path = data_dir / name
    _write_signal(path, values)
    _write_companion(path, companion)
    return path


def _payloads(event_type: str, asset: str = ASSET) -> list[dict]:
    store = LedgerStore(get_ledger_dir())
    return [
        e["payload"] for e in store.read(asset).events if e["event_type"] == event_type
    ]


def _declare_point(asset: str = ASSET, **overrides) -> None:
    payload = {
        "measurement_point_id": POINT,
        "declaration_version": 1,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": 1800.0,
        "declared_by": "test",
        "note": None,
        "changed": [],
    }
    payload.update(overrides)
    LedgerStore(get_ledger_dir()).append(
        asset, make_event(EVENT_MEASUREMENT_POINT_DECLARED, asset, payload)
    )


@pytest.fixture
def ledger_tools():
    """Acquisition plus analysis tools (AE1 analyzes the loaded signal)."""
    server = MCPServer("test-acquisition-ledger")
    acquisition_tools.register(server)
    analysis_tools.register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


@pytest.fixture
def clean_repo():
    repo = repo_module.get_repository()
    repo.clear_all()
    yield repo
    repo.clear_all()


class TestLoadSignalLedger:
    """load_signal records identified measurements in the asset ledger."""

    @pytest.mark.asyncio
    async def test_full_companion_is_recorded_and_analyzable(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        """Covers AE1: recorded, partial snapshot (no point declared: three
        blocks missing, each with a remedy), the signal usable by the
        analysis tools as before, the cached block identity-only."""
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        result = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")

        block = result.measurement
        assert block["asset_id"] == ASSET
        assert block["measurement_point_id"] == POINT
        assert len(block["measurement_id"]) == 16
        assert LEDGER_BLOCK_KEYS <= set(block)
        assert block["ledger_status"] == "recorded"
        assert block["reason"] is None
        assert block["changed"] == []
        assert block["reattributed_from"] is None
        assert block["declaration_version"] == 1
        assert block["snapshot_status"] == "partial"
        assert set(block["missing"]) == {"one_x", "bearing", "iso"}
        for entry in block["missing"].values():
            assert entry["reason"] and entry["remedy"]
        assert len(block["snapshot_id"]) == 16
        assert block["processing_id"].startswith("health_snapshot/")
        assert block["comparability"]["grade"] == "qualified"
        assert [q["code"] for q in block["comparability"]["qualifications"]] == [
            "rpm_not_declared"
        ]
        assert "error" not in json.dumps(result.model_dump())

        fft = await ledger_tools["analyze_fft"](ctx=mock_ctx, signal_id="m01")
        assert fft.total_bins > 0
        envelope = await ledger_tools["analyze_envelope"](ctx=mock_ctx, signal_id="m01")
        assert envelope.signal_id == "m01"

        cached = await ledger_tools["get_signal_info"](ctx=mock_ctx, signal_id="m01")
        assert cached.measurement["asset_id"] == ASSET
        assert cached.measurement["measurement_id"] == block["measurement_id"]
        assert not (LEDGER_BLOCK_KEYS & set(cached.measurement))
        listed = await ledger_tools["list_signals"](ctx=mock_ctx, scope="memory")
        assert not (LEDGER_BLOCK_KEYS & set(listed["signals"][0]["measurement"]))

        store = LedgerStore(ledger_dir)
        assert [e["event_type"] for e in store.read(ASSET).events] == [
            EVENT_MEASUREMENT_RECORDED,
            EVENT_HEALTH_SNAPSHOT_COMPUTED,
        ]
        assert store.find_measurement_asset(block["measurement_id"]) == ASSET
        recorded = _payloads(EVENT_MEASUREMENT_RECORDED)[0]
        assert recorded["signal_id"] == "m01"
        assert recorded["file"]["location"] == "m01.csv"
        assert recorded["file"]["location_is_relative"] is True
        assert recorded["declaration"]["sampling_rate"] == 10000.0
        assert recorded["declaration"]["signal_unit"] == "g"

    @pytest.mark.asyncio
    async def test_point_declared_first_is_complete_and_comparable(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        _declare_point()
        _place(
            sandbox_data_dir,
            "m01.csv",
            golden_signals()["golden_bearing"],
            _companion(_identity(rpm=1800)),
        )
        result = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        block = result.measurement
        assert block["ledger_status"] == "recorded"
        assert block["snapshot_status"] == "complete"
        assert block["missing"] == {}
        assert block["comparability"] == {"grade": "comparable", "qualifications": []}
        snapshot = _payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)[0]
        assert snapshot["point_declaration_version"] == 1
        assert snapshot["iso"]["direction"] == "horizontal"
        assert snapshot["bearing"]["labels"]["BPFO"]["detected"] is True

    @pytest.mark.asyncio
    async def test_restart_between_loads_keeps_the_history(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        """Covers AE2: load, clear the cache, load another measurement (a new
        store instance is built per call) -> both in acquired_at order with
        their snapshots."""
        _place(
            sandbox_data_dir,
            "m02.csv",
            _noise(2),
            _companion(_identity(acquired_at="2026-08-27T13:42:00+02:00")),
        )
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        later = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m02.csv")
        cleared = await ledger_tools["clear_signals"](ctx=mock_ctx)
        assert cleared["cleared_count"] == 1
        earlier = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        assert earlier.measurement["ledger_status"] == "recorded"

        view = LedgerStore(ledger_dir).read_view(ASSET)
        assert view["ordered_measurement_ids"] == [
            earlier.measurement["measurement_id"],
            later.measurement["measurement_id"],
        ]
        for slot in view["measurements"].values():
            assert len(slot["snapshots"]) == 1

    @pytest.mark.asyncio
    async def test_overwrite_reload_is_already_recorded(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        again = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        assert again.measurement["ledger_status"] == "already_recorded"
        assert again.measurement["measurement_id"] == (
            first.measurement["measurement_id"]
        )
        assert again.measurement["snapshot_id"] == first.measurement["snapshot_id"]
        assert len(_payloads(EVENT_MEASUREMENT_RECORDED)) == 1
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    @pytest.mark.asyncio
    async def test_corrected_rpm_supersedes_with_a_new_snapshot(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        path = _place(
            sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity(rpm=1800))
        )
        first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        _write_companion(path, _companion(_identity(rpm=1500)))
        second = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        block = second.measurement
        assert block["ledger_status"] == "superseded"
        assert block["changed"] == ["rpm"]
        assert block["declaration_version"] == 2
        assert block["measurement_id"] == first.measurement["measurement_id"]
        assert block["snapshot_id"] != first.measurement["snapshot_id"]
        snapshots = _payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)
        assert [s["one_x"]["target_hz"] for s in snapshots] == [30.0, 25.0]
        assert len(_payloads(EVENT_MEASUREMENT_RECORDED)) == 2

    @pytest.mark.asyncio
    async def test_corrected_acquired_at_supersedes_without_a_new_snapshot(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        path = _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        _write_companion(
            path, _companion(_identity(acquired_at="2026-08-20T15:00:00+02:00"))
        )
        second = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        assert second.measurement["ledger_status"] == "superseded"
        assert second.measurement["changed"] == ["acquired_at"]
        assert second.measurement["snapshot_id"] == first.measurement["snapshot_id"]
        view = LedgerStore(ledger_dir).read_view(ASSET)
        slot = view["measurements"][first.measurement["measurement_id"]]
        assert slot["current"]["declaration"]["acquired_at"] == (
            "2026-08-20T13:00:00+00:00"
        )
        assert len(slot["snapshots"]) == 1

    @pytest.mark.asyncio
    async def test_moved_file_supersedes_with_location_only(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        path = _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        archive = sandbox_data_dir / "archive"
        archive.mkdir()
        shutil.move(str(path), str(archive / "m01.csv"))
        shutil.move(
            str(path.parent / "m01_metadata.json"), str(archive / "m01_metadata.json")
        )
        second = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="archive/m01.csv"
        )
        block = second.measurement
        assert second.signal_id == "archive_m01"
        assert block["ledger_status"] == "superseded"
        assert block["changed"] == ["location"]
        assert block["measurement_id"] == first.measurement["measurement_id"]
        recorded = _payloads(EVENT_MEASUREMENT_RECORDED)
        assert recorded[1]["file"]["location"] == "archive/m01.csv"
        assert recorded[1]["locations"] == ["m01.csv", "archive/m01.csv"]
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    @pytest.mark.asyncio
    async def test_unit_parameter_supersedes_with_a_new_snapshot(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        second = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", signal_unit="m/s2", overwrite=True
        )
        assert second.signal_unit == "m/s2"
        assert second.measurement["ledger_status"] == "superseded"
        assert second.measurement["changed"] == ["signal_unit"]
        assert second.measurement["snapshot_id"] != first.measurement["snapshot_id"]
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 2

    @pytest.mark.asyncio
    async def test_two_files_in_the_same_slot_are_both_recorded(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        _place(sandbox_data_dir, "a.csv", _noise(1), _companion(_identity()))
        _place(sandbox_data_dir, "b.csv", _noise(2), _companion(_identity()))
        results = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath=["a.csv", "b.csv"]
        )
        statuses = [r.measurement["ledger_status"] for r in results]
        assert statuses == ["recorded", "recorded"]
        ids = {r.measurement["measurement_id"] for r in results}
        assert len(ids) == 2
        view = LedgerStore(ledger_dir).read_view(ASSET)
        assert set(view["ordered_measurement_ids"]) == ids

    @pytest.mark.asyncio
    async def test_axes_against_a_point_expecting_x(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        _declare_point(expected_direction="x")
        for seed, axis in enumerate(("x", "y", "z"), start=1):
            _place(
                sandbox_data_dir,
                f"axis_{axis}.csv",
                _noise(seed),
                _companion(_identity(direction=axis, rpm=1800)),
            )
        results = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath=["axis_x.csv", "axis_y.csv", "axis_z.csv"]
        )
        by_id = {r.signal_id: r.measurement for r in results}
        assert all(m["ledger_status"] == "recorded" for m in by_id.values())
        assert by_id["axis_x"]["comparability"]["grade"] == "comparable"
        for axis in ("axis_y", "axis_z"):
            grade = by_id[axis]["comparability"]
            assert grade["grade"] == "non_comparable"
            assert [q["code"] for q in grade["qualifications"]] == [
                "direction_mismatch"
            ]
        assert len(_payloads(EVENT_MEASUREMENT_RECORDED)) == 3

    @pytest.mark.asyncio
    async def test_unwritable_ledger_dir_does_not_fail_the_load(
        self,
        ledger_tools,
        sandbox_data_dir,
        tmp_path,
        monkeypatch,
        mock_ctx,
        clean_repo,
    ):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("PMM_LEDGER_DIR", str(blocker))
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        result = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        block = result.measurement
        assert block["ledger_status"] == "not_recorded"
        assert "PMM_LEDGER_DIR" in block["reason"]
        assert block["snapshot_status"] == "skipped"
        assert block["asset_id"] == ASSET
        fft = await ledger_tools["analyze_fft"](ctx=mock_ctx, signal_id="m01")
        assert fft.total_bins > 0
        assert blocker.read_text(encoding="utf-8") == "x"

    @pytest.mark.asyncio
    async def test_busy_lock_does_not_fail_the_load(
        self,
        ledger_tools,
        sandbox_data_dir,
        ledger_dir,
        monkeypatch,
        mock_ctx,
        clean_repo,
    ):
        holder = LedgerStore(ledger_dir)
        holder._ensure_root()
        _, lock_path = holder._paths(ASSET)
        fd = holder._acquire_lock(lock_path, what="test holder")
        monkeypatch.setattr(
            acquisition_tools,
            "LedgerStore",
            functools.partial(LedgerStore, lock_timeout=0.2),
        )
        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        try:
            result = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        finally:
            holder._release_lock(fd)
        assert result.measurement["ledger_status"] == "not_recorded"
        assert "lock" in result.measurement["reason"]
        assert repo_module.get_repository().signal_count == 1

        again = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        assert again.measurement["ledger_status"] == "recorded"
        assert again.measurement["snapshot_status"] == "partial"

    @pytest.mark.asyncio
    async def test_snapshot_failure_then_retry_via_reload(
        self, ledger_tools, sandbox_data_dir, mock_ctx, clean_repo
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("engine exploded")

        _place(sandbox_data_dir, "m01.csv", _noise(1), _companion(_identity()))
        # A scoped patch: the sandbox fixtures must survive the undo.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ledger_service, "compute_health_snapshot", boom)
            first = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        assert first.measurement["ledger_status"] == "recorded"
        assert first.measurement["snapshot_status"] == "failed"
        assert "engine exploded" in first.measurement["reason"]
        assert _payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED) == []

        again = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        assert again.measurement["ledger_status"] == "already_recorded"
        assert again.measurement["snapshot_status"] == "partial"
        assert len(_payloads(EVENT_MEASUREMENT_RECORDED)) == 1
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 1

    @pytest.mark.asyncio
    async def test_batch_with_one_invalid_companion_loads_and_appends_nothing(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        paths = build_measurement_sequence(sandbox_data_dir / "seq")
        broken = sequence_companion(paths[17])
        payload = json.loads(broken.read_text(encoding="utf-8"))
        del payload["measurement"]["acquired_at"]
        broken.write_text(json.dumps(payload), encoding="utf-8")
        names = [f"seq/{p.name}" for p in paths]
        with pytest.raises(ValueError, match="seq_m18_metadata.json"):
            await ledger_tools["load_signal"](ctx=mock_ctx, filepath=names)
        assert repo_module.get_repository().signal_count == 0
        assert not ledger_dir.exists()

    @pytest.mark.asyncio
    async def test_valid_batch_of_thirty_reads_the_asset_once(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        paths = build_measurement_sequence(sandbox_data_dir / "seq")
        reads: list[str] = []
        original = LedgerStore.read

        def spy(self, asset_id):
            reads.append(asset_id)
            return original(self, asset_id)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(LedgerStore, "read", spy)
            results = await ledger_tools["load_signal"](
                ctx=mock_ctx, filepath=[f"seq/{p.name}" for p in paths]
            )
        assert len(results) == SEQUENCE_LENGTH
        assert {r.measurement["ledger_status"] for r in results} == {"recorded"}
        assert {r.measurement["snapshot_status"] for r in results} == {"partial"}
        assert reads == [SEQUENCE_ASSET]

        view = LedgerStore(ledger_dir).read_view(SEQUENCE_ASSET)
        assert view["ordered_measurement_ids"] == [
            r.measurement["measurement_id"] for r in results
        ]
        assert len(_payloads(EVENT_MEASUREMENT_RECORDED, SEQUENCE_ASSET)) == 30
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED, SEQUENCE_ASSET)) == 30

    @pytest.mark.asyncio
    async def test_lru_eviction_keeps_the_snapshots(
        self, ledger_tools, sandbox_data_dir, monkeypatch, mock_ctx, clean_repo
    ):
        """A cache too small for one signal evicts the previous load every
        time; the snapshots were computed at load time and stay."""
        monkeypatch.setenv("PMM_SIGNAL_CACHE_GB", "0.000001")
        monkeypatch.setattr(repo_module, "_repository", None)
        for i in (1, 2, 3):
            _place(
                sandbox_data_dir,
                f"m{i:02d}.csv",
                _noise(i),
                _companion(_identity(acquired_at=f"2026-08-{i:02d}T10:00:00+00:00")),
            )
            result = await ledger_tools["load_signal"](
                ctx=mock_ctx, filepath=f"m{i:02d}.csv"
            )
            assert result.measurement["ledger_status"] == "recorded"
            assert result.measurement["snapshot_status"] == "partial"
        tiny = repo_module.get_repository()
        assert tiny.signal_count == 1
        assert len(_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)) == 3
        tiny.clear_all()

    @pytest.mark.asyncio
    async def test_free_form_companion_keys_never_reach_the_ledger(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        _place(
            sandbox_data_dir,
            "m01.csv",
            _noise(1),
            _companion(_identity(), operator_name="Mario Rossi"),
        )
        result = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        assert result.measurement["ledger_status"] == "recorded"
        assert result.source_metadata["operator_name"] == "Mario Rossi"
        for path in ledger_dir.iterdir():
            data = path.read_bytes()
            assert b"Mario Rossi" not in data
            assert b"operator_name" not in data

    @pytest.mark.asyncio
    async def test_wrong_asset_then_corrected(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        path = _place(
            sandbox_data_dir,
            "m01.csv",
            _noise(1),
            _companion(_identity(asset_id=OTHER_ASSET)),
        )
        wrong = await ledger_tools["load_signal"](ctx=mock_ctx, filepath="m01.csv")
        assert wrong.measurement["asset_id"] == OTHER_ASSET
        wrong_ledger = ledger_dir / "P-102.jsonl"
        bytes_before = wrong_ledger.read_bytes()

        _write_companion(path, _companion(_identity(asset_id=ASSET)))
        right = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="m01.csv", overwrite=True
        )
        block = right.measurement
        assert block["asset_id"] == ASSET
        assert block["ledger_status"] == "recorded"
        assert block["reattributed_from"] == OTHER_ASSET
        assert block["measurement_id"] == wrong.measurement["measurement_id"]

        store = LedgerStore(ledger_dir)
        assert wrong_ledger.read_bytes().startswith(bytes_before)
        superseding = _payloads(EVENT_MEASUREMENT_RECORDED, OTHER_ASSET)
        assert [p["declaration_version"] for p in superseding] == [1, 2]
        assert superseding[1]["changed"] == ["asset_id"]
        assert superseding[1]["declaration"]["asset_id"] == ASSET
        wrong_view = store.read_view(OTHER_ASSET)
        assert wrong_view["ordered_measurement_ids"] == []
        assert wrong_view["reattributed"] == [
            {"measurement_id": block["measurement_id"], "to_asset_id": ASSET}
        ]
        assert store.read_view(ASSET)["ordered_measurement_ids"] == [
            block["measurement_id"]
        ]
        assert store.find_measurement_asset(block["measurement_id"]) == ASSET

    @pytest.mark.asyncio
    async def test_malformed_companion_loads_without_ledger_events(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        path = sandbox_data_dir / "broken.csv"
        _write_signal(path, _noise(1))
        _write_companion(path, '{"sampling_rate": 10000, "measurement": {')
        result = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="broken.csv", sampling_rate=10000.0
        )
        assert result.measurement is None
        assert "broken_metadata.json" in result.companion_warning
        assert not ledger_dir.exists()

    @pytest.mark.asyncio
    async def test_locations_inside_and_outside_the_data_dir(
        self, ledger_tools, sandbox_data_dir, tmp_path, mock_ctx, clean_repo
    ):
        _place(sandbox_data_dir, "site_a/m01.csv", _noise(1), _companion(_identity()))
        inside = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath="site_a/m01.csv"
        )
        outside_path = tmp_path / "elsewhere" / "m02.csv"
        _write_signal(outside_path, _noise(2))
        _write_companion(
            outside_path, _companion(_identity(acquired_at="2026-08-21T10:00:00Z"))
        )
        outside = await ledger_tools["load_signal"](
            ctx=mock_ctx, filepath=str(outside_path)
        )
        assert inside.measurement["ledger_status"] == "recorded"
        assert outside.measurement["ledger_status"] == "recorded"

        by_id = {
            p["measurement_id"]: p["file"]
            for p in _payloads(EVENT_MEASUREMENT_RECORDED)
        }
        assert by_id[inside.measurement["measurement_id"]] == {
            "location": "site_a/m01.csv",
            "location_is_relative": True,
            "content_sha256": inside.measurement["content_sha256"],
            "size_bytes": inside.measurement["size_bytes"],
        }
        assert by_id[outside.measurement["measurement_id"]]["location"] == str(
            outside_path
        )
        assert by_id[outside.measurement["measurement_id"]]["location_is_relative"] is (
            False
        )

    @pytest.mark.asyncio
    async def test_generate_test_signal_is_untouched(
        self, ledger_tools, sandbox_data_dir, ledger_dir, mock_ctx, clean_repo
    ):
        """A generated signal declares no identity: nothing is recorded."""
        result = await ledger_tools["generate_test_signal"](
            signal_type="normal", duration=0.2, sampling_rate=5000.0, ctx=mock_ctx
        )
        assert result.measurement is None
        assert not ledger_dir.exists()
