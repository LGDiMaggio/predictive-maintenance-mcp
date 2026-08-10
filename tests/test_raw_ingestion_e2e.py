"""End-to-end raw binary ingestion through the registered MCP tools (U4).

The fixture mirrors a realistic industrial DAQ capture's exact shape —
1,349,632 float32 little-endian samples (5,398,528 bytes) — but contains a
synthetic sine at a known frequency, so the full flow (declaration →
load_signal → analyze_fft → assess_severity) is provable without committing
vendor data.
"""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import (
    acquisition_tools,
    analysis_tools,
    diagnostics_tools,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

# ---------------------------------------------------------------------------
# Reference-capture fixture geometry
# ---------------------------------------------------------------------------

FS = 25600.0
N_SAMPLES = 1_349_632  # exactly the reference capture's sample count
N_BYTES = 5_398_528  # N_SAMPLES * 4 bytes (float32)
SINE_HZ = 500.0  # integer Hz → exact bin of the 1 s FFT segment (1 Hz grid)
AMPLITUDE = 1.0  # becomes 1.0 mm/s once the unit is declared
RAW_NAME = "vendor_capture.bin"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signals_dir(tmp_path_factory):
    """Write the 5.4 MB reference-shaped sine ONCE for the whole module."""
    d = tmp_path_factory.mktemp("raw_e2e") / "data" / "signals"
    d.mkdir(parents=True)
    t = np.arange(N_SAMPLES) / FS
    payload = (AMPLITUDE * np.sin(2 * np.pi * SINE_HZ * t)).astype("<f4").tobytes()
    assert len(payload) == N_BYTES
    (d / RAW_NAME).write_bytes(payload)
    return d


@pytest.fixture
def mcp():
    """MCPServer with the three tool modules the F1 flow crosses."""
    server = MCPServer("test-raw-e2e")
    acquisition_tools.register(server)
    analysis_tools.register(server)
    diagnostics_tools.register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(signals_dir, monkeypatch):
    """Point every module that captured DATA_DIR at the module fixture dir."""
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
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


async def _load_raw(tools, ctx, **kwargs):
    """Load the reference-shaped file with the minimal raw declaration."""
    return await tools["load_signal"](
        ctx=ctx,
        filepath=RAW_NAME,
        sample_format="float32",
        sampling_rate=FS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


class TestRawIngestionEndToEnd:
    """F1: declaration → load → analysis → ISO verdict, on one raw file."""

    @pytest.mark.asyncio
    async def test_load_reports_exact_sample_count(self, tools, data_dir, mock_ctx):
        """5,398,528 bytes decode to exactly 1,349,632 samples."""
        repo = get_repository()
        repo.clear_all()
        try:
            info = await _load_raw(tools, mock_ctx)
            assert info.signal_id == "vendor_capture"
            assert info.num_samples == N_SAMPLES
            assert info.sampling_rate == FS
            assert info.signal_unit is None  # not declared, never guessed
            assert len(repo.get_signal(info.signal_id)) == N_SAMPLES
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_fft_recovers_declared_sine_frequency(
        self, tools, data_dir, mock_ctx
    ):
        """Covers AE2/F1: spectral analysis is operational on a raw load."""
        repo = get_repository()
        repo.clear_all()
        try:
            info = await _load_raw(tools, mock_ctx)
            fft = await tools["analyze_fft"](ctx=mock_ctx, signal_id=info.signal_id)
            # Default 1.0 s leading segment at fs=25600 → 1 Hz bin grid, so
            # the integer-cycle sine lands on an exact bin.
            assert fft.peak_frequency == pytest.approx(SINE_HZ, abs=1.0)
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_iso_refused_without_unit_then_verdict_after_redeclare(
        self, tools, data_dir, mock_ctx
    ):
        """Covers AE2: the ISO refusal path works through a raw signal.

        No declared unit → the EXISTING structured refusal (ValueError
        naming the undeclared unit and the load_signal remedy); after the
        documented remedy (reload with overwrite=True and a declared unit)
        the severity tool returns a zone verdict.
        """
        repo = get_repository()
        repo.clear_all()
        try:
            info = await _load_raw(tools, mock_ctx)
            with pytest.raises(ValueError, match="signal unit not declared") as exc:
                await tools["assess_severity"](ctx=mock_ctx, signal_id=info.signal_id)
            # The refusal names the remedy, not a guessed unit.
            assert "load_signal" in str(exc.value)

            reloaded = await tools["load_signal"](
                ctx=mock_ctx,
                filepath=RAW_NAME,
                sample_format="float32",
                sampling_rate=FS,
                signal_unit="mm/s",
                overwrite=True,
            )
            assert reloaded.signal_unit == "mm/s"

            verdict = await tools["assess_severity"](
                ctx=mock_ctx, signal_id=reloaded.signal_id
            )
            assert verdict.zone in {"A", "B", "C", "D"}
            # A 1.0 mm/s amplitude sine at 500 Hz (inside the 10-1000 Hz ISO
            # band) has RMS 1/sqrt(2) ≈ 0.707 mm/s → Zone A for group 2 rigid.
            assert verdict.rms_velocity_mm_s == pytest.approx(
                AMPLITUDE / np.sqrt(2), rel=0.05
            )
            assert verdict.zone == "A"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_get_signal_info_reports_effective_raw_format(
        self, tools, data_dir, mock_ctx
    ):
        """raw_format provenance: the six EFFECTIVE decode values are queryable."""
        repo = get_repository()
        repo.clear_all()
        try:
            loaded = await _load_raw(tools, mock_ctx)
            info = await tools["get_signal_info"](
                ctx=mock_ctx, signal_id=loaded.signal_id
            )
            assert info.raw_format == {
                "sample_format": "float32",
                "byte_order": "little",  # effective default, recorded
                "n_channels": 1,
                "channel_index": 0,
                "header_offset": 0,
                "scale_factor": None,  # no scaling declared
            }
        finally:
            repo.clear_all()
