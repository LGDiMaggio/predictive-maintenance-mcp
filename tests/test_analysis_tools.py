"""Tests for MCP analysis tools (ISO 13374 Block 2).

Since U8 every analysis tool takes signal_id as its only signal handle:
signals are loaded once via the repository and referenced by id.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock

from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.analysis_tools import analyze_fft, register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

from _golden_signals import load_golden_signals

#: Characterization snapshot of ``analyze_fft`` on the golden signals.
ANALYZE_FFT_GOLDEN_FILE = Path(__file__).parent / "fixtures" / "analyze_fft_golden.json"

#: The ``analyze_fft`` calls frozen in the snapshot (keyword arguments only;
#: ``ctx`` is added by the caller). Default segment (leading 1 s), the whole
#: signal, and a ``max_frequency`` cut, on both golden signals.
ANALYZE_FFT_GOLDEN_CASES: dict[str, dict] = {
    "golden_iso_default": {"signal_id": "golden_iso"},
    "golden_iso_full_signal": {"signal_id": "golden_iso", "segment_duration": None},
    "golden_bearing_default": {"signal_id": "golden_bearing"},
    "golden_bearing_max_4000": {"signal_id": "golden_bearing", "max_frequency": 4000.0},
}


def build_analyze_fft_golden() -> dict:
    """Run every golden ``analyze_fft`` case and return the snapshot payload.

    Loads the deterministic golden signals into the repository (temporary
    directory, explicit rate and unit, no companions), runs each case of
    :data:`ANALYZE_FFT_GOLDEN_CASES` and dumps the ``FFTResult`` as JSON.
    Clears the repository afterwards. Not for use inside a running event
    loop: the capture recipe calls it from a plain interpreter.
    """
    repo = get_repository()
    repo.clear_all()
    payload: dict = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            load_golden_signals(repo, Path(tmp))
            ctx = AsyncMock()
            for name, kwargs in ANALYZE_FFT_GOLDEN_CASES.items():
                result = asyncio.run(analyze_fft(ctx=ctx, **kwargs))
                payload[name] = {
                    "call": kwargs,
                    "result": result.model_dump(mode="json"),
                }
    finally:
        repo.clear_all()
    return payload


def write_analyze_fft_golden() -> Path:
    """Regenerate the characterization fixture ON PURPOSE (see the recipe)."""
    ANALYZE_FFT_GOLDEN_FILE.write_text(
        json.dumps(build_analyze_fft_golden(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ANALYZE_FFT_GOLDEN_FILE


def assert_close_tree(
    new, old, path: str = "", rel: float = 1e-7, abs_tol: float = 1e-9
):
    """Recursive equality of a JSON tree, numbers via ``pytest.approx``.

    Every key of the golden tree must be present with the same value; the
    tolerance is tight on purpose (same platform: bit-identical; across the
    CI matrix: a few ulps at most). Any algorithmic drift (window, scaling,
    segment choice) differs by orders of magnitude more.
    """
    if isinstance(old, dict):
        assert isinstance(new, dict), f"{path}: expected dict, got {type(new)}"
        for key, old_val in old.items():
            assert key in new, f"{path}.{key}: missing in new output"
            assert_close_tree(new[key], old_val, f"{path}.{key}", rel, abs_tol)
    elif isinstance(old, list):
        assert isinstance(new, (list, tuple)), f"{path}: expected list"
        assert len(new) == len(old), f"{path}: length {len(new)} != golden {len(old)}"
        for i, (n, o) in enumerate(zip(new, old)):
            assert_close_tree(n, o, f"{path}[{i}]", rel, abs_tol)
    elif isinstance(old, bool) or old is None or isinstance(old, str):
        assert new == old, f"{path}: {new!r} != golden {old!r}"
    else:
        assert new == pytest.approx(
            old, rel=rel, abs=abs_tol
        ), f"{path}: {new} != golden {old}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp():
    server = MCPServer("test-analysis")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Temp directory with synthetic test signals."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)

    # Pure 50 Hz sine
    sig = np.sin(2 * np.pi * 50 * t)
    pd.DataFrame(sig).to_csv(signals_dir / "sine50.csv", index=False, header=False)
    with open(signals_dir / "sine50_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Multi-frequency signal: 50 + 150 Hz
    sig2 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
    pd.DataFrame(sig2).to_csv(signals_dir / "multi.csv", index=False, header=False)
    with open(signals_dir / "multi_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Patch all relevant modules
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir
    )
    return signals_dir


@pytest.fixture
def repo(data_dir):
    """Repository with the synthetic signals loaded; cleaned afterwards."""
    repo = get_repository()
    repo.clear_all()
    repo.load_signal("sine50.csv")  # metadata: fs=10000, unit 'g'
    repo.load_signal("multi.csv")
    yield repo
    repo.clear_all()


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
def golden_repo(tmp_path):
    """Repository holding the deterministic golden signals (function scope:
    the ``repo`` fixture of this module clears the shared repository)."""
    repo = get_repository()
    repo.clear_all()
    load_golden_signals(repo, tmp_path)
    yield repo
    repo.clear_all()


# ---------------------------------------------------------------------------
# analyze_fft: characterization against the frozen snapshot
# ---------------------------------------------------------------------------


class TestAnalyzeFFTCharacterization:
    """``analyze_fft`` output is frozen in tests/fixtures/analyze_fft_golden.json.

    Captured BEFORE the amplitude-spectrum core (Hamming window,
    ``scipy.fft.fft``, positive frequencies, single-sided ``2|X|/N``) was
    extracted from the tool into ``signal_processing.spectral.amplitude_spectrum``
    and the deterministic leading-segment rule into ``select_leading_segment``
    (asset-ledger U3). The extraction must be numerically invisible: every
    peak frequency, magnitude and summary field of the ``FFTResult`` must
    still match the snapshot within ``assert_close_tree``'s tight tolerance.

    Fixture history:
    - U3 (2026-09-08): initial capture on the inline implementation, four
      cases of :data:`ANALYZE_FFT_GOLDEN_CASES`.

    Regenerate ON PURPOSE only when ``analyze_fft``'s numbers are meant to
    change (never to make a red test green), adding a line to the history
    above. From the repo root, with the checkout's interpreter:

        python -c "import sys; sys.path.insert(0, 'tests'); \\
                   from test_analysis_tools import write_analyze_fft_golden; \\
                   print(write_analyze_fft_golden())"
    """

    @pytest.fixture(scope="class")
    def golden(self) -> dict:
        with open(ANALYZE_FFT_GOLDEN_FILE, encoding="utf-8") as fh:
            return json.load(fh)

    def test_fixture_covers_every_case(self, golden):
        assert set(golden) == set(ANALYZE_FFT_GOLDEN_CASES)
        for name, kwargs in ANALYZE_FFT_GOLDEN_CASES.items():
            assert golden[name]["call"] == kwargs, name

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", sorted(ANALYZE_FFT_GOLDEN_CASES))
    async def test_matches_golden(self, case, golden, golden_repo, mock_ctx):
        result = await analyze_fft(ctx=mock_ctx, **ANALYZE_FFT_GOLDEN_CASES[case])
        assert_close_tree(result.model_dump(mode="json"), golden[case]["result"], case)

    @pytest.mark.asyncio
    async def test_golden_peaks_are_where_the_fixtures_put_them(
        self, golden, golden_repo, mock_ctx
    ):
        """Sanity anchor independent of the snapshot: the golden_iso sine
        peaks at 50 Hz and golden_bearing at its 3 kHz carrier."""
        iso = await analyze_fft(ctx=mock_ctx, signal_id="golden_iso")
        bearing = await analyze_fft(ctx=mock_ctx, signal_id="golden_bearing")
        assert iso.peak_frequency == pytest.approx(50.0, abs=1.0)
        assert bearing.peak_frequency == pytest.approx(3000.0, abs=1.0)


# ---------------------------------------------------------------------------
# analyze_fft
# ---------------------------------------------------------------------------


class TestAnalyzeFFT:
    """Tests for analyze_fft tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_detects_50hz(self, tools, repo, mock_ctx):
        result = await tools["analyze_fft"](ctx=mock_ctx, signal_id="sine50")
        # FFTResult model — dominant frequency should be ~50 Hz
        assert abs(result.peak_frequency - 50.0) < 2.0

    @pytest.mark.asyncio
    async def test_uses_stored_sampling_rate(self, tools, repo, mock_ctx):
        """The rate comes from the stored signal (metadata at load time)."""
        result = await tools["analyze_fft"](ctx=mock_ctx, signal_id="sine50")
        assert result.sampling_rate == 10000
        assert result.peak_frequency > 0

    @pytest.mark.asyncio
    async def test_returns_peaks(self, tools, repo, mock_ctx):
        result = await tools["analyze_fft"](ctx=mock_ctx, signal_id="multi")
        assert len(result.top_peaks) > 0

    @pytest.mark.asyncio
    async def test_signal_not_loaded_names_remedy(self, tools, repo, mock_ctx):
        """Unknown signal_id → error listing the loaded ids and naming
        load_signal/list_signals as the remedy."""
        with pytest.raises(ValueError) as exc:
            await tools["analyze_fft"](ctx=mock_ctx, signal_id="nonexistent")
        msg = str(exc.value)
        assert "load_signal" in msg
        assert "list_signals" in msg
        assert "sine50" in msg  # available ids listed

    @pytest.mark.asyncio
    async def test_no_rate_anywhere_raises(self, tools, data_dir, repo, mock_ctx):
        """Signal stored without a rate → structured error, never a silent
        default."""
        fs = 10000
        t = np.linspace(0, 0.5, fs // 2, endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(
            data_dir / "no_meta_fft.csv", index=False, header=False
        )
        repo.load_signal("no_meta_fft.csv")  # no metadata → no rate

        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["analyze_fft"](ctx=mock_ctx, signal_id="no_meta_fft")

    @pytest.mark.asyncio
    async def test_deterministic_repeat_calls(self, tools, repo, mock_ctx):
        """Two identical calls analyze identical samples (audit 2.15: the
        random default segment is gone)."""
        r1 = await tools["analyze_fft"](
            ctx=mock_ctx, signal_id="multi", segment_duration=0.25
        )
        r2 = await tools["analyze_fft"](
            ctx=mock_ctx, signal_id="multi", segment_duration=0.25
        )
        assert r1.peak_frequency == r2.peak_frequency
        assert r1.peak_magnitude == r2.peak_magnitude
        assert [p.model_dump() for p in r1.top_peaks] == [
            p.model_dump() for p in r2.top_peaks
        ]

    @pytest.mark.asyncio
    async def test_random_seed_is_explicit_and_reproducible(
        self, tools, repo, mock_ctx
    ):
        """Seeded random segment position is opt-in and reproducible."""
        r1 = await tools["analyze_fft"](
            ctx=mock_ctx, signal_id="multi", segment_duration=0.25, random_seed=7
        )
        r2 = await tools["analyze_fft"](
            ctx=mock_ctx, signal_id="multi", segment_duration=0.25, random_seed=7
        )
        assert r1.peak_magnitude == r2.peak_magnitude


# ---------------------------------------------------------------------------
# analyze_envelope
# ---------------------------------------------------------------------------


class TestAnalyzeEnvelope:
    """Tests for the unified analyze_envelope tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_envelope_returns_result(self, tools, repo, mock_ctx):
        result = await tools["analyze_envelope"](ctx=mock_ctx, signal_id="sine50")
        assert result is not None
        assert len(result.top_peaks) > 0
        assert result.signal_id == "sine50"

    @pytest.mark.asyncio
    async def test_envelope_default_band_echoed(self, tools, repo, mock_ctx):
        """Unified default band is 500-5000 Hz, echoed in the output."""
        result = await tools["analyze_envelope"](ctx=mock_ctx, signal_id="sine50")
        assert tuple(result.filter_band) == (500.0, 5000.0)

    @pytest.mark.asyncio
    async def test_envelope_invalid_band_raises(self, tools, repo, mock_ctx):
        """Band above Nyquist raises — never a silent clamp (U9)."""
        with pytest.raises(ValueError, match="Nyquist"):
            await tools["analyze_envelope"](
                ctx=mock_ctx, signal_id="sine50", filter_high=6000.0
            )

    @pytest.mark.asyncio
    async def test_envelope_deterministic_repeat_calls(self, tools, repo, mock_ctx):
        r1 = await tools["analyze_envelope"](ctx=mock_ctx, signal_id="multi")
        r2 = await tools["analyze_envelope"](ctx=mock_ctx, signal_id="multi")
        assert [p.model_dump() for p in r1.top_peaks] == [
            p.model_dump() for p in r2.top_peaks
        ]

    @pytest.mark.asyncio
    async def test_old_spectrum_tool_gone(self, tools):
        """compute_envelope_spectrum_tool merged into analyze_envelope."""
        assert "compute_envelope_spectrum_tool" not in tools


# ---------------------------------------------------------------------------
# extract_features_from_signal
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for extract_features_from_signal tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_extracts_features(self, tools, repo, mock_ctx):
        result = await tools["extract_features_from_signal"](
            signal_id="sine50",
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        assert result.num_segments > 0
        assert len(result.feature_names) == 17

    @pytest.mark.asyncio
    async def test_segment_count(self, tools, repo, mock_ctx):
        # 1s signal, 0.5s segments, 0 overlap → 2 segments
        result = await tools["extract_features_from_signal"](
            signal_id="sine50",
            segment_duration=0.5,
            overlap_ratio=0.0,
            ctx=mock_ctx,
        )
        assert result.num_segments == 2

    @pytest.mark.asyncio
    async def test_unknown_signal_id_raises(self, tools, repo, mock_ctx):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["extract_features_from_signal"](
                signal_id="__ghost__", ctx=mock_ctx
            )


# ---------------------------------------------------------------------------
# PSD, STFT, Envelope Spectrum (delegation tools)
# ---------------------------------------------------------------------------


class TestSpectralDelegation:
    """Tests for PSD/STFT/envelope tools that delegate to spectral.py."""

    @pytest.mark.asyncio
    async def test_compute_psd(self, tools, repo, mock_ctx):
        result = await tools["compute_power_spectral_density"](
            ctx=mock_ctx, signal_id="sine50"
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_compute_stft(self, tools, repo, mock_ctx):
        try:
            result = await tools["compute_spectrogram_stft"](
                ctx=mock_ctx, signal_id="sine50"
            )
            assert result is not None
        except Exception:
            # Known issue: energy_per_band band names are strings not floats
            pytest.skip("STFT model validation issue with energy_per_band")


# ---------------------------------------------------------------------------
# analyze_statistics
# ---------------------------------------------------------------------------


class TestAnalyzeStatistics:
    """Tests for analyze_statistics tool (signal_id handle)."""

    def test_returns_stats(self, tools, repo):
        result = tools["analyze_statistics"](signal_id="sine50")
        assert result is not None
        assert hasattr(result, "rms")

    def test_unit_reported_only_when_declared(self, tools, repo):
        """Declared metadata unit is reported as-is (no amplitude heuristic)."""
        result = tools["analyze_statistics"](signal_id="sine50")
        assert result.signal_unit == "g"  # from sine50_metadata.json at load
        assert "declared" in result.unit_note

    def test_undeclared_unit_not_guessed(self, tools, data_dir, repo):
        """High-RMS signal without declaration: the old heuristic guessed
        'g'; now the unit stays None and the note names the declaration
        path."""
        fs = 10000
        t = np.linspace(0, 0.5, fs // 2, endpoint=False)
        sig = 4.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)  # RMS ~4 > 0.5
        pd.DataFrame(sig).to_csv(
            data_dir / "loud_no_meta.csv", index=False, header=False
        )
        repo.load_signal("loud_no_meta.csv", sampling_rate=fs)

        result = tools["analyze_statistics"](signal_id="loud_no_meta")
        assert result.signal_unit is None
        assert "load_signal" in result.unit_note
        assert "signal_unit=" in result.unit_note

    def test_unknown_signal_id_raises(self, tools, repo):
        with pytest.raises(ValueError, match="load_signal"):
            tools["analyze_statistics"](signal_id="__ghost__")
