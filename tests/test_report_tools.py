"""Tests for MCP report generation tools (ISO 13374 Block 6).

Since U8 every report tool takes signal_id as its only signal handle:
signals are loaded once via the repository and referenced by id.
"""

import json
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from predictive_maintenance_mcp.mcp_tools.report_tools import register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = FastMCP("test-reports")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 50 * t)
    pd.DataFrame(sig).to_csv(signals_dir / "report_test.csv", index=False, header=False)
    with open(signals_dir / "report_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir)
    # generate_iso_report delegates to the modular evaluate_iso_20816
    # (diagnostics_tools) — patch its DATA_DIR too.
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.DATA_DIR", signals_dir)
    return signals_dir


@pytest.fixture
def repo(data_dir):
    """Repository with the report test signal loaded; cleaned afterwards."""
    repo = get_repository()
    repo.clear_all()
    repo.load_signal("report_test.csv")  # metadata: fs=10000, unit 'g'
    yield repo
    repo.clear_all()


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    rd = tmp_path / "reports"
    rd.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR", rd)
    monkeypatch.setattr("predictive_maintenance_mcp.report_generator.REPORTS_DIR", rd)
    # Also patch the REPORTS_DIR_PATH alias used in diagnostics import
    try:
        monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR_PATH", rd)
    except AttributeError:
        pass
    return rd


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


def _load_extra_signal(repo, signals_dir, name, fs=10000, duration=1.0,
                       freq=50.0, noise=0.0, with_meta=True):
    """Helper: create a CSV signal file and load it into the repository."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
    pd.DataFrame(sig).to_csv(signals_dir / name, index=False, header=False)
    if with_meta:
        meta_path = signals_dir / name.replace(".csv", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)
    return repo.load_signal(name)["signal_id"]


# ---------------------------------------------------------------------------
# Plot signal
# ---------------------------------------------------------------------------

class TestPlotSignal:
    """Tests for plot_signal tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_creates_html(self, tools, repo, reports_dir, mock_ctx):
        result = await tools["plot_signal"](signal_id="report_test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_plot_signal_with_time_range(self, tools, repo, reports_dir):
        """plot_signal with time_range should produce a zoomed plot."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            time_range=[0.1, 0.3],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_no_statistics(self, tools, repo, reports_dir):
        """plot_signal with show_statistics=False omits reference lines."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            show_statistics=False,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_custom_title(self, tools, repo, reports_dir):
        """plot_signal with custom title."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            title="Custom Title Test",
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_not_loaded(self, tools, repo, reports_dir):
        """Unknown signal_id → standard error naming load_signal."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["plot_signal"](signal_id="nonexistent")

    @pytest.mark.asyncio
    async def test_plot_signal_with_ctx(self, tools, repo, reports_dir, mock_ctx):
        """plot_signal with ctx should call ctx.info."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count >= 1
        assert "Interactive plot saved" in result


# ---------------------------------------------------------------------------
# Plot spectrum
# ---------------------------------------------------------------------------

class TestPlotSpectrum:
    """Tests for plot_spectrum tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_creates_spectrum_html(self, tools, repo, reports_dir):
        result = await tools["plot_spectrum"](signal_id="report_test")
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_with_freq_range(self, tools, repo, reports_dir):
        result = await tools["plot_spectrum"](
            signal_id="report_test",
            freq_range=[10.0, 200.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_with_rotation_freq(self, tools, repo, reports_dir):
        """Spectrum should label harmonics when rotation_freq is given."""
        result = await tools["plot_spectrum"](
            signal_id="report_test",
            rotation_freq=50.0,
            num_peaks=5,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_signal_not_loaded(self, tools, repo, reports_dir):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["plot_spectrum"](signal_id="missing")


# ---------------------------------------------------------------------------
# Plot envelope
# ---------------------------------------------------------------------------

class TestPlotEnvelope:
    """Tests for plot_envelope tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_creates_envelope_html(self, tools, repo, reports_dir):
        result = await tools["plot_envelope"](
            signal_id="report_test",
            filter_band=[100.0, 4000.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_default_filter_band(self, tools, repo, reports_dir):
        """When filter_band is None, defaults to [500, 5000] and adjusts if needed."""
        result = await tools["plot_envelope"](signal_id="report_test")
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_with_highlight_freqs(self, tools, repo, reports_dir):
        """Envelope plot should mark highlighted bearing frequencies."""
        result = await tools["plot_envelope"](
            signal_id="report_test",
            filter_band=[100.0, 4000.0],
            freq_range=[0.0, 300.0],
            highlight_freqs=[50.0, 100.0],
            freq_labels=["BPFO", "2xBPFO"],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_highlight_freqs_no_labels(self, tools, repo, reports_dir):
        """When highlight_freqs given but no labels, auto-generate labels."""
        result = await tools["plot_envelope"](
            signal_id="report_test",
            filter_band=[100.0, 4000.0],
            highlight_freqs=[50.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_signal_not_loaded(self, tools, repo, reports_dir):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["plot_envelope"](signal_id="missing")

    @pytest.mark.asyncio
    async def test_envelope_invalid_filter_band(self, tools, repo, reports_dir):
        """Filter band entirely above Nyquist should raise."""
        with pytest.raises(ValueError):
            await tools["plot_envelope"](
                signal_id="report_test",
                filter_band=[6000.0, 7000.0],  # both above Nyquist (5000)
            )


# ---------------------------------------------------------------------------
# List reports
# ---------------------------------------------------------------------------

class TestListReports:
    """Tests for list_html_reports tool."""

    def test_empty_reports(self, tools, reports_dir):
        result = tools["list_html_reports"]()
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 0

    def test_finds_existing_report(self, tools, reports_dir):
        """Verify list_html_reports returns a list (report_generator uses its own REPORTS_DIR)."""
        result = tools["list_html_reports"]()
        # Just verify it returns a list without error
        assert isinstance(result, list)

    def test_lists_html_files_with_metadata(self, tools, reports_dir):
        """list_html_reports should find HTML reports that contain metadata."""
        # Create a fake HTML report with embedded metadata
        metadata = {"report_type": "fft_spectrum", "signal_file": "test.csv"}
        html_content = (
            '<html><head>'
            '<script type="application/json" id="report-metadata">'
            f'{json.dumps(metadata)}'
            '</script></head><body>report</body></html>'
        )
        (reports_dir / "fft_test_report.html").write_text(html_content)
        result = tools["list_html_reports"]()
        assert len(result) == 1
        assert result[0]["report_type"] == "fft_spectrum"
        assert result[0]["signal_file"] == "test.csv"

    def test_lists_ignores_html_without_metadata(self, tools, reports_dir):
        """HTML files without metadata block should be skipped."""
        (reports_dir / "plain.html").write_text("<html><body>no metadata</body></html>")
        result = tools["list_html_reports"]()
        # File has no metadata so list_reports skips it (error branch)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Get report info
# ---------------------------------------------------------------------------

class TestGetReportInfo:
    """Tests for get_report_info tool."""

    def test_report_not_found_raises(self, tools, reports_dir):
        """U6 error contract: a missing report is misuse — raise with an
        actionable message, never an error-shaped dict as success."""
        with pytest.raises(ValueError, match="list_html_reports"):
            tools["get_report_info"](file_name="nonexistent.html")

    def test_report_found_with_metadata(self, tools, reports_dir):
        metadata = {"report_type": "envelope", "signal_file": "sig.csv", "num_peaks": 10}
        html = (
            '<html><head>'
            '<script type="application/json" id="report-metadata">'
            f'{json.dumps(metadata)}'
            '</script></head><body></body></html>'
        )
        (reports_dir / "env_report.html").write_text(html)
        result = tools["get_report_info"](file_name="env_report.html")
        assert "error" not in result
        assert result["metadata"]["report_type"] == "envelope"
        assert result["metadata"]["num_peaks"] == 10

    def test_report_without_metadata_block_raises(self, tools, reports_dir):
        """Report file exists but has no metadata JSON block → raise."""
        (reports_dir / "no_meta.html").write_text("<html><body>plain</body></html>")
        with pytest.raises(ValueError, match="Metadata not found"):
            tools["get_report_info"](file_name="no_meta.html")


# ---------------------------------------------------------------------------
# Generate FFT report
# ---------------------------------------------------------------------------

class TestGenerateFFTReport:
    """Tests for generate_fft_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_fft_report(self, tools, repo, reports_dir):
        result = await tools["generate_fft_report"](signal_id="report_test")
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_fft_report_uses_stored_sampling_rate(self, tools, repo, reports_dir):
        """The rate comes from the stored signal (declared at load time)."""
        result = await tools["generate_fft_report"](signal_id="report_test")
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_fft_report_with_rotation_freq(self, tools, repo, reports_dir):
        result = await tools["generate_fft_report"](
            signal_id="report_test",
            rotation_freq=50.0,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_fft_report_signal_without_rate(self, tools, data_dir, repo, reports_dir):
        """A stored signal without a sampling rate → structured error."""
        sig = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000, endpoint=False))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta.csv", index=False, header=False)
        repo.load_signal("no_meta.csv")  # no metadata → no rate
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_fft_report"](signal_id="no_meta")

    @pytest.mark.asyncio
    async def test_fft_report_signal_not_loaded(self, tools, repo, reports_dir):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_fft_report"](signal_id="does_not_exist")

    @pytest.mark.asyncio
    async def test_fft_report_with_ctx(self, tools, repo, reports_dir, mock_ctx):
        result = await tools["generate_fft_report"](
            signal_id="report_test",
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count >= 1
        assert "file_path" in result


# ---------------------------------------------------------------------------
# Generate envelope report
# ---------------------------------------------------------------------------

class TestGenerateEnvelopeReport:
    """Tests for generate_envelope_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_envelope_report(self, tools, repo, reports_dir):
        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_envelope_report_bearing_freqs_from_companion_metadata(
        self, tools, data_dir, repo, reports_dir
    ):
        """Optional bearing freqs are auto-read from the source file's
        companion metadata when not passed explicitly."""
        meta_path = data_dir / "report_test_metadata.json"
        meta = json.loads(meta_path.read_text())
        # 6205 (CWRU geometry) at 1797 RPM
        meta.update({"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.58, "FTF": 11.93})
        meta_path.write_text(json.dumps(meta))

        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_with_bearing_freqs(self, tools, repo, reports_dir):
        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
            bearing_freqs={"BPFO": 81.0, "BPFI": 119.0, "BSF": 64.0, "FTF": 15.0},
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_signal_without_rate(
        self, tools, data_dir, repo, reports_dir
    ):
        """Stored signal without a rate → structured error."""
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(data_dir / "no_meta_env.csv", index=False, header=False)
        repo.load_signal("no_meta_env.csv")
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_envelope_report"](signal_id="no_meta_env")

    @pytest.mark.asyncio
    async def test_envelope_report_with_ctx_bearing_matches(self, tools, repo, reports_dir, mock_ctx):
        """When bearing matches are found, ctx.info should report them."""
        await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
            bearing_freqs={"BPFO": 50.0, "BPFI": 100.0},
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count >= 1


# ---------------------------------------------------------------------------
# Generate ISO report
# ---------------------------------------------------------------------------

class TestGenerateISOReport:
    """Tests for generate_iso_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_iso_report(self, tools, repo, reports_dir, mock_ctx):
        result = await tools["generate_iso_report"](
            signal_id="report_test",
            machine_group=2,
            support_type="rigid",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_iso_report_signal_without_rate(
        self, tools, data_dir, repo, reports_dir
    ):
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(data_dir / "no_meta_iso.csv", index=False, header=False)
        repo.load_signal("no_meta_iso.csv")
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_iso_report"](signal_id="no_meta_iso")

    @pytest.mark.asyncio
    async def test_iso_report_undeclared_unit_refused(
        self, tools, data_dir, repo, reports_dir
    ):
        """Rate known but unit undeclared → structured error naming the fix."""
        sig = np.random.randn(20000)
        pd.DataFrame(sig).to_csv(data_dir / "no_unit_iso.csv", index=False, header=False)
        repo.load_signal("no_unit_iso.csv", sampling_rate=10000)
        with pytest.raises(ValueError, match="unit not declared"):
            await tools["generate_iso_report"](signal_id="no_unit_iso")


# ---------------------------------------------------------------------------
# Load → analyze → report round-trip (R7 golden path)
# ---------------------------------------------------------------------------

class TestSignalIdRoundTrip:
    """R7: load → analyze → report is walkable with ONE idiom (signal_id)."""

    @pytest.mark.asyncio
    async def test_load_analyze_report_single_idiom(
        self, tools, data_dir, reports_dir, mock_ctx
    ):
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
            load_signal,
        )
        from predictive_maintenance_mcp.mcp_tools.analysis_tools import (
            analyze_fft,
        )

        repo = get_repository()
        repo.clear_all()
        try:
            info = await load_signal(ctx=mock_ctx, filepath="report_test.csv")
            sid = info.signal_id
            assert sid == "report_test"

            fft_result = await analyze_fft(ctx=mock_ctx, signal_id=sid)
            assert abs(fft_result.peak_frequency - 50.0) < 2.0

            report = await tools["generate_fft_report"](signal_id=sid)
            assert Path(report["file_path"]).exists()
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Generate PCA visualization report
# ---------------------------------------------------------------------------

class TestGeneratePCAReport:
    """Tests for generate_pca_visualization_report tool (signal_id handles)."""

    @pytest.fixture
    def models_dir(self, tmp_path, monkeypatch, data_dir):
        """Create a fake trained model with scaler, PCA, and metadata."""
        md = tmp_path / "models"
        md.mkdir()
        monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.report_tools.MODELS_DIR", md)

        # Build a small model from synthetic features
        rng = np.random.RandomState(42)
        n_samples, n_features = 50, 17
        X_train = rng.randn(n_samples, n_features)

        scaler = StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        pca = PCA(n_components=2).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_pca)

        model_name = "test_pca_model"
        with open(md / f"{model_name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(md / f"{model_name}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(md / f"{model_name}_pca.pkl", "wb") as f:
            pickle.dump(pca, f)
        with open(md / f"{model_name}_metadata.json", "w") as f:
            json.dump({"sampling_rate": 10000.0, "model_type": "OneClassSVM"}, f)

        return md

    @pytest.mark.asyncio
    async def test_pca_report_no_test_signals(self, tools, repo, reports_dir, models_dir, mock_ctx):
        """PCA report with no test signals should still produce a report."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["summary"]["total_segments"] == 0

    @pytest.mark.asyncio
    async def test_pca_report_with_test_signals(self, tools, repo, reports_dir, models_dir, mock_ctx):
        """PCA report with test signal_ids should segment, predict, and report."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_ids=["report_test"],
            segment_duration=0.1,
            overlap_ratio=0.5,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert result["summary"]["total_segments"] > 0

    @pytest.mark.asyncio
    async def test_pca_report_with_true_labels(self, tools, repo, reports_dir, models_dir, mock_ctx):
        """PCA report with true_labels (keyed by signal_id) computes metrics."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_ids=["report_test"],
            true_labels={"report_test": "healthy"},
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert result["metadata"]["validation_mode"] is True
        assert "validation_metrics" in result["summary"]
        assert "overall_accuracy" in result["summary"]["validation_metrics"]

    @pytest.mark.asyncio
    async def test_pca_report_model_not_found(self, tools, repo, reports_dir, models_dir):
        with pytest.raises(FileNotFoundError):
            await tools["generate_pca_visualization_report"](
                model_name="nonexistent_model",
            )

    @pytest.mark.asyncio
    async def test_pca_report_signal_not_loaded(self, tools, repo, reports_dir, models_dir):
        """Unknown test signal_id → fail fast, no silent skipping."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_pca_visualization_report"](
                model_name="test_pca_model",
                test_signal_ids=["__ghost__"],
            )


# ---------------------------------------------------------------------------
# Generate feature comparison report
# ---------------------------------------------------------------------------

class TestGenerateFeatureComparisonReport:
    """Tests for generate_feature_comparison_report tool (signal_id handles)."""

    @pytest.fixture
    def multi_signal_ids(self, repo, data_dir):
        """Load two signal groups for comparison."""
        ids = {
            "healthy_1": _load_extra_signal(repo, data_dir, "healthy_1.csv", freq=50.0, noise=0.01),
            "healthy_2": _load_extra_signal(repo, data_dir, "healthy_2.csv", freq=50.0, noise=0.02),
            "faulty_1": _load_extra_signal(repo, data_dir, "faulty_1.csv", freq=50.0, noise=1.0),
        }
        return ids

    @pytest.mark.asyncio
    async def test_feature_comparison_basic(self, tools, multi_signal_ids, reports_dir, mock_ctx):
        result = await tools["generate_feature_comparison_report"](
            signal_groups={
                "Healthy": ["healthy_1", "healthy_2"],
                "Faulty": ["faulty_1"],
            },
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["metadata"]["report_type"] == "feature_comparison"
        assert "Healthy" in result["metadata"]["groups"]

    @pytest.mark.asyncio
    async def test_feature_comparison_subset_features(self, tools, multi_signal_ids, reports_dir):
        """Only plot a subset of features."""
        result = await tools["generate_feature_comparison_report"](
            signal_groups={"A": ["healthy_1"], "B": ["faulty_1"]},
            features_to_plot=["rms", "kurtosis", "crest_factor"],
        )
        assert len(result["metadata"]["features_plotted"]) == 3

    @pytest.mark.asyncio
    async def test_feature_comparison_unknown_id_raises(self, tools, multi_signal_ids, reports_dir):
        """Unknown signal_ids fail fast — no silent skipping (U8 contract)."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_feature_comparison_report"](
                signal_groups={
                    "Mixed": ["healthy_1", "__does_not_exist__"],
                },
            )


# ---------------------------------------------------------------------------
# DOCX report generation
# ---------------------------------------------------------------------------

def _docx_installed() -> bool:
    try:
        import docx  # noqa: F401
        return True
    except ImportError:
        return False


class TestDocxReport:
    """Tests for generate_diagnostic_report_docx tool (signal_id handle).

    Since U6 the tool RAISES ValueError when python-docx is missing (error
    contract: failures raise, never error dicts) — so the happy-path tests
    are skipped when the optional dependency is not installed.
    """

    @pytest.mark.asyncio
    async def test_generates_docx(self, tools, repo, reports_dir, mock_ctx):
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            ctx=mock_ctx,
            signal_id="report_test",
            sections={"diagnosis": "Test diagnosis summary"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_docx_with_all_sections(self, tools, repo, reports_dir, mock_ctx):
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            ctx=mock_ctx,
            signal_id="report_test",
            sections={
                "statistics": {"RMS": 0.707, "Kurtosis": 3.0, "Crest Factor": 1.414},
                "fft_peaks": [{"frequency": 50.0, "magnitude_db": 0.0, "note": "1x"}],
                "diagnosis": "All parameters within normal range.",
            },
            title="Custom DOCX Title",
        )
        assert result is not None
        assert "error" not in result
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_docx_no_ctx(self, tools, repo, reports_dir):
        """DOCX generation without ctx should still work."""
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            signal_id="report_test",
            sections={"diagnosis": "Summary"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_docx_signal_not_loaded_raises(self, tools, repo, reports_dir):
        """Reports are only produced for loaded signals."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_diagnostic_report_docx"](
                signal_id="__ghost__",
                sections={"diagnosis": "Summary"},
            )

    @pytest.mark.asyncio
    async def test_docx_missing_dependency_raises(
        self, tools, repo, reports_dir, mock_ctx, monkeypatch
    ):
        """Missing python-docx → raised ValueError, never an error dict."""
        monkeypatch.setattr(
            "predictive_maintenance_mcp.report_generator.HAS_DOCX", False
        )
        with pytest.raises(ValueError, match="python-docx"):
            await tools["generate_diagnostic_report_docx"](
                ctx=mock_ctx,
                signal_id="report_test",
                sections={"diagnosis": "Summary"},
            )


# ---------------------------------------------------------------------------
# Tool registration completeness
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify all expected tools are registered."""

    def test_all_expected_tools_registered(self, tools):
        expected = {
            "plot_signal",
            "plot_spectrum",
            "plot_envelope",
            "generate_fft_report",
            "generate_envelope_report",
            "generate_iso_report",
            "list_html_reports",
            "get_report_info",
            "generate_pca_visualization_report",
            "generate_feature_comparison_report",
            "generate_diagnostic_report_docx",
        }
        for name in expected:
            assert name in tools, f"Expected tool '{name}' not registered"
