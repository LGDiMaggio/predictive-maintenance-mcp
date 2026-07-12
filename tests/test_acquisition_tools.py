"""Tests for MCP acquisition tools (ISO 13374 Block 1)."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.acquisition_tools import register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    """Create a FastMCP test instance with acquisition tools registered."""
    server = FastMCP("test-acquisition")
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

    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir)
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
    """Tests for list_signals tool."""

    def test_lists_signals(self, mcp, data_dir):
        # Access the tool function directly from server
        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "list_signals":
                tool_fn = tool.fn
                break
        assert tool_fn is not None
        result = tool_fn()
        assert "test_sine.csv" in result
        assert "baseline_1.csv" in result

    def test_empty_directory(self, mcp, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", empty_dir)
        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "list_signals":
                tool_fn = tool.fn
                break
        result = tool_fn()
        assert "No signal files found" in result


class TestGenerateTestSignal:
    """Tests for generate_test_signal tool."""

    @pytest.mark.asyncio
    async def test_generates_bearing_fault(self, mcp, data_dir):
        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "generate_test_signal":
                tool_fn = tool.fn
                break
        assert tool_fn is not None
        result = await tool_fn(
            signal_type="bearing_fault",
            duration=1.0,
            sampling_rate=10000.0,
            noise_level=0.1,
            ctx=None,
        )
        assert "Successfully generated" in result
        assert (data_dir / "test_bearing_fault_10000Hz.csv").exists()

    @pytest.mark.asyncio
    async def test_generates_normal_signal(self, mcp, data_dir):
        tool_fn = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "generate_test_signal":
                tool_fn = tool.fn
                break
        result = await tool_fn(
            signal_type="normal",
            duration=0.5,
            sampling_rate=5000.0,
            ctx=None,
        )
        assert "Successfully generated" in result


class TestSignalRepository:
    """Tests for load_signal, list_stored_signals, get_signal_info, clear tools."""

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
        stored = await tools["list_stored_signals"](ctx=mock_ctx)
        assert len(stored) >= 1
        assert any(s.signal_id == "sine1" for s in stored)

        # Get info
        info = await tools["get_signal_info"](ctx=mock_ctx, signal_id="sine1")
        assert info.signal_id == "sine1"

        # Clear
        cleared = await tools["clear_signal"](ctx=mock_ctx, signal_id="sine1")
        assert cleared["status"] == "removed"

    @pytest.mark.asyncio
    async def test_clear_all(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        await tools["load_signal"](ctx=mock_ctx, filepath="test_sine.csv", signal_id="s1")
        result = await tools["clear_all_signals"](ctx=mock_ctx)
        assert result["cleared_count"] >= 1


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
            await tools["clear_all_signals"](ctx=mock_ctx)

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
            await tools["clear_all_signals"](ctx=mock_ctx)

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
            stored = await tools["list_stored_signals"](ctx=mock_ctx)
            assert len(stored) == 2
        finally:
            await tools["clear_all_signals"](ctx=mock_ctx)

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
            stored = await tools["list_stored_signals"](ctx=mock_ctx)
            assert len(stored) == 0
        finally:
            await tools["clear_all_signals"](ctx=mock_ctx)

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
            await tools["clear_all_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_explicit_unit_overrides_metadata(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", signal_id="param_unit",
                signal_unit="mm/s",
            )
            assert result.signal_unit == "mm/s"  # declared > metadata ('g')
        finally:
            await tools["clear_all_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_no_declaration_is_none(self, mcp, data_dir, mock_ctx):
        """Without any declaration the unit stays None — never guessed."""
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        try:
            result = await tools["load_signal"](
                ctx=mock_ctx, filepath="real_train/baseline_1.csv",
                signal_id="no_unit",
            )
            assert result.signal_unit is None
        finally:
            await tools["clear_all_signals"](ctx=mock_ctx)

    @pytest.mark.asyncio
    async def test_invalid_unit_raises(self, mcp, data_dir, mock_ctx):
        tools = {t.name: t.fn for t in mcp._tool_manager._tools.values()}
        with pytest.raises(ValueError, match="signal_unit"):
            await tools["load_signal"](
                ctx=mock_ctx, filepath="test_sine.csv", signal_id="bad_unit",
                signal_unit="furlongs",
            )
