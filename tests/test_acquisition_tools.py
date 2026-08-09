"""Tests for MCP acquisition tools (ISO 13374 Block 1)."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.acquisition_tools import register

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
