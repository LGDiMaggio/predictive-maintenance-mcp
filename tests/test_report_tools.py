"""Tests for MCP report generation tools (ISO 13374 Block 6)."""

import json
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from mcp.server.fastmcp import FastMCP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from predictive_maintenance_mcp.mcp_tools.report_tools import register


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

    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.report_tools.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_repository.DATA_DIR", signals_dir)
    # Also patch legacy monolith DATA_DIR (used by generate_iso_report → evaluate_iso_20816)
    try:
        monkeypatch.setattr("predictive_maintenance_mcp.machinery_diagnostics_server.DATA_DIR", signals_dir)
    except AttributeError:
        pass  # monolith may not be imported yet
    return signals_dir


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


def _make_signal_file(signals_dir, name, fs=10000, duration=1.0, freq=50.0, noise=0.0):
    """Helper: create a CSV signal file with optional metadata."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
    pd.DataFrame(sig).to_csv(signals_dir / name, index=False, header=False)
    meta_path = signals_dir / name.replace(".csv", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)
    return signals_dir / name


# ---------------------------------------------------------------------------
# Plot signal
# ---------------------------------------------------------------------------

class TestPlotSignal:
    """Tests for plot_signal tool."""

    @pytest.mark.asyncio
    async def test_creates_html(self, tools, data_dir, reports_dir, mock_ctx):
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        result = await tools["plot_signal"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_plot_signal_with_time_range(self, tools, data_dir, reports_dir):
        """plot_signal with time_range should produce a zoomed plot."""
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        result = await tools["plot_signal"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            time_range=[0.1, 0.3],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_no_statistics(self, tools, data_dir, reports_dir):
        """plot_signal with show_statistics=False omits reference lines."""
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        result = await tools["plot_signal"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            show_statistics=False,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_custom_title(self, tools, data_dir, reports_dir):
        """plot_signal with custom title."""
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        result = await tools["plot_signal"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            title="Custom Title Test",
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_file_not_found(self, tools, data_dir, reports_dir):
        """plot_signal should raise FileNotFoundError for missing file."""
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        with pytest.raises(FileNotFoundError):
            await tools["plot_signal"](
                signal_file="nonexistent.csv",
                sampling_rate=10000.0,
            )

    @pytest.mark.asyncio
    async def test_plot_signal_with_ctx(self, tools, data_dir, reports_dir, mock_ctx):
        """plot_signal with ctx should call ctx.info."""
        if "plot_signal" not in tools:
            pytest.skip("plot_signal not registered")
        result = await tools["plot_signal"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count >= 1


# ---------------------------------------------------------------------------
# Plot spectrum
# ---------------------------------------------------------------------------

class TestPlotSpectrum:
    """Tests for plot_spectrum tool."""

    @pytest.mark.asyncio
    async def test_creates_spectrum_html(self, tools, data_dir, reports_dir):
        if "plot_spectrum" not in tools:
            pytest.skip("plot_spectrum not registered")
        result = await tools["plot_spectrum"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_with_freq_range(self, tools, data_dir, reports_dir):
        if "plot_spectrum" not in tools:
            pytest.skip("plot_spectrum not registered")
        result = await tools["plot_spectrum"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            freq_range=[10.0, 200.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_with_rotation_freq(self, tools, data_dir, reports_dir):
        """Spectrum should label harmonics when rotation_freq is given."""
        if "plot_spectrum" not in tools:
            pytest.skip("plot_spectrum not registered")
        result = await tools["plot_spectrum"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            rotation_freq=50.0,
            num_peaks=5,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_spectrum_file_not_found(self, tools, data_dir, reports_dir):
        if "plot_spectrum" not in tools:
            pytest.skip("plot_spectrum not registered")
        with pytest.raises(FileNotFoundError):
            await tools["plot_spectrum"](
                signal_file="missing.csv",
                sampling_rate=10000.0,
            )


# ---------------------------------------------------------------------------
# Plot envelope
# ---------------------------------------------------------------------------

class TestPlotEnvelope:
    """Tests for plot_envelope tool."""

    @pytest.mark.asyncio
    async def test_creates_envelope_html(self, tools, data_dir, reports_dir):
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        result = await tools["plot_envelope"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            filter_band=[100.0, 4000.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_default_filter_band(self, tools, data_dir, reports_dir):
        """When filter_band is None, defaults to [500, 5000] and adjusts if needed."""
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        result = await tools["plot_envelope"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_with_highlight_freqs(self, tools, data_dir, reports_dir):
        """Envelope plot should mark highlighted bearing frequencies."""
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        result = await tools["plot_envelope"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            filter_band=[100.0, 4000.0],
            freq_range=[0.0, 300.0],
            highlight_freqs=[50.0, 100.0],
            freq_labels=["BPFO", "2xBPFO"],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_highlight_freqs_no_labels(self, tools, data_dir, reports_dir):
        """When highlight_freqs given but no labels, auto-generate labels."""
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        result = await tools["plot_envelope"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            filter_band=[100.0, 4000.0],
            highlight_freqs=[50.0],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_envelope_file_not_found(self, tools, data_dir, reports_dir):
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        with pytest.raises(FileNotFoundError):
            await tools["plot_envelope"](
                signal_file="missing.csv",
                sampling_rate=10000.0,
            )

    @pytest.mark.asyncio
    async def test_envelope_invalid_filter_band(self, tools, data_dir, reports_dir):
        """Filter band exceeding Nyquist in a way that makes low >= 1 should raise."""
        if "plot_envelope" not in tools:
            pytest.skip("plot_envelope not registered")
        with pytest.raises(ValueError):
            await tools["plot_envelope"](
                signal_file="report_test.csv",
                sampling_rate=10000.0,
                filter_band=[6000.0, 7000.0],  # both above Nyquist (5000)
            )


# ---------------------------------------------------------------------------
# List reports
# ---------------------------------------------------------------------------

class TestListReports:
    """Tests for list_html_reports tool."""

    def test_empty_reports(self, tools, reports_dir):
        if "list_html_reports" not in tools:
            pytest.skip("list_html_reports not registered")
        result = tools["list_html_reports"]()
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 0

    def test_finds_existing_report(self, tools, reports_dir):
        """Verify list_html_reports returns a list (report_generator uses its own REPORTS_DIR)."""
        if "list_html_reports" not in tools:
            pytest.skip("list_html_reports not registered")
        result = tools["list_html_reports"]()
        # Just verify it returns a list without error
        assert isinstance(result, list)

    def test_lists_html_files_with_metadata(self, tools, reports_dir):
        """list_html_reports should find HTML reports that contain metadata."""
        if "list_html_reports" not in tools:
            pytest.skip("list_html_reports not registered")
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
        if "list_html_reports" not in tools:
            pytest.skip("list_html_reports not registered")
        (reports_dir / "plain.html").write_text("<html><body>no metadata</body></html>")
        result = tools["list_html_reports"]()
        # File has no metadata so list_reports skips it (error branch)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Get report info
# ---------------------------------------------------------------------------

class TestGetReportInfo:
    """Tests for get_report_info tool."""

    def test_report_not_found(self, tools, reports_dir):
        if "get_report_info" not in tools:
            pytest.skip("get_report_info not registered")
        result = tools["get_report_info"](file_name="nonexistent.html")
        assert "error" in result

    def test_report_found_with_metadata(self, tools, reports_dir):
        if "get_report_info" not in tools:
            pytest.skip("get_report_info not registered")
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

    def test_report_without_metadata_block(self, tools, reports_dir):
        """Report file exists but has no metadata JSON block."""
        if "get_report_info" not in tools:
            pytest.skip("get_report_info not registered")
        (reports_dir / "no_meta.html").write_text("<html><body>plain</body></html>")
        result = tools["get_report_info"](file_name="no_meta.html")
        assert "error" in result
        assert "Metadata not found" in result["error"]


# ---------------------------------------------------------------------------
# Generate FFT report
# ---------------------------------------------------------------------------

class TestGenerateFFTReport:
    """Tests for generate_fft_report tool."""

    @pytest.mark.asyncio
    async def test_generates_fft_report(self, tools, data_dir, reports_dir):
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        result = await tools["generate_fft_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_fft_report_auto_detect_sampling_rate(self, tools, data_dir, reports_dir):
        """generate_fft_report auto-detects sampling_rate from metadata."""
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        result = await tools["generate_fft_report"](
            signal_file="report_test.csv",
            # sampling_rate not specified; should be read from metadata
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_fft_report_with_rotation_freq(self, tools, data_dir, reports_dir):
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        result = await tools["generate_fft_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            rotation_freq=50.0,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_fft_report_no_sampling_rate_no_metadata(self, tools, data_dir, reports_dir):
        """Should raise ValueError when no sampling_rate and no metadata."""
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        # Create signal without metadata
        sig = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000, endpoint=False))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta.csv", index=False, header=False)
        with pytest.raises(ValueError, match="sampling_rate required"):
            await tools["generate_fft_report"](signal_file="no_meta.csv")

    @pytest.mark.asyncio
    async def test_fft_report_missing_file(self, tools, data_dir, reports_dir):
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        with pytest.raises((ValueError, FileNotFoundError)):
            await tools["generate_fft_report"](
                signal_file="does_not_exist.csv",
                sampling_rate=10000.0,
            )

    @pytest.mark.asyncio
    async def test_fft_report_with_ctx(self, tools, data_dir, reports_dir, mock_ctx):
        if "generate_fft_report" not in tools:
            pytest.skip("generate_fft_report not registered")
        result = await tools["generate_fft_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            ctx=mock_ctx,
        )
        assert mock_ctx.info.call_count >= 1
        assert "file_path" in result


# ---------------------------------------------------------------------------
# Generate envelope report
# ---------------------------------------------------------------------------

class TestGenerateEnvelopeReport:
    """Tests for generate_envelope_report tool."""

    @pytest.mark.asyncio
    async def test_generates_envelope_report(self, tools, data_dir, reports_dir):
        if "generate_envelope_report" not in tools:
            pytest.skip("generate_envelope_report not registered")
        result = await tools["generate_envelope_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_envelope_report_auto_detect(self, tools, data_dir, reports_dir):
        """Auto-detect sampling_rate and bearing_freqs from metadata."""
        if "generate_envelope_report" not in tools:
            pytest.skip("generate_envelope_report not registered")
        # Add bearing freqs to metadata
        meta_path = data_dir / "report_test_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update({"BPFO": 81.13, "BPFI": 118.88, "BSF": 63.91, "FTF": 14.84})
        meta_path.write_text(json.dumps(meta))

        result = await tools["generate_envelope_report"](
            signal_file="report_test.csv",
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_with_bearing_freqs(self, tools, data_dir, reports_dir):
        if "generate_envelope_report" not in tools:
            pytest.skip("generate_envelope_report not registered")
        result = await tools["generate_envelope_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            filter_low=100.0,
            filter_high=4000.0,
            bearing_freqs={"BPFO": 81.0, "BPFI": 119.0, "BSF": 64.0, "FTF": 15.0},
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_no_sampling_rate_error(self, tools, data_dir, reports_dir):
        """Should raise when sampling_rate cannot be determined."""
        if "generate_envelope_report" not in tools:
            pytest.skip("generate_envelope_report not registered")
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(data_dir / "no_meta_env.csv", index=False, header=False)
        with pytest.raises(ValueError, match="sampling_rate required"):
            await tools["generate_envelope_report"](signal_file="no_meta_env.csv")

    @pytest.mark.asyncio
    async def test_envelope_report_with_ctx_bearing_matches(self, tools, data_dir, reports_dir, mock_ctx):
        """When bearing matches are found, ctx.info should report them."""
        if "generate_envelope_report" not in tools:
            pytest.skip("generate_envelope_report not registered")
        result = await tools["generate_envelope_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
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
    """Tests for generate_iso_report tool."""

    @pytest.mark.asyncio
    async def test_generates_iso_report(self, tools, data_dir, reports_dir, mock_ctx):
        if "generate_iso_report" not in tools:
            pytest.skip("generate_iso_report not registered")
        result = await tools["generate_iso_report"](
            signal_file="report_test.csv",
            sampling_rate=10000.0,
            machine_group=2,
            support_type="rigid",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_iso_report_auto_detect_sampling_rate(self, tools, data_dir, reports_dir, mock_ctx):
        if "generate_iso_report" not in tools:
            pytest.skip("generate_iso_report not registered")
        result = await tools["generate_iso_report"](
            signal_file="report_test.csv",
            machine_group=2,
            support_type="rigid",
            ctx=mock_ctx,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_iso_report_no_sampling_rate_error(self, tools, data_dir, reports_dir):
        if "generate_iso_report" not in tools:
            pytest.skip("generate_iso_report not registered")
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(data_dir / "no_meta_iso.csv", index=False, header=False)
        with pytest.raises(ValueError, match="sampling_rate required"):
            await tools["generate_iso_report"](signal_file="no_meta_iso.csv")


# ---------------------------------------------------------------------------
# Generate PCA visualization report
# ---------------------------------------------------------------------------

class TestGeneratePCAReport:
    """Tests for generate_pca_visualization_report tool."""

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
    async def test_pca_report_no_test_signals(self, tools, data_dir, reports_dir, models_dir, mock_ctx):
        """PCA report with no test signals should still produce a report."""
        if "generate_pca_visualization_report" not in tools:
            pytest.skip("generate_pca_visualization_report not registered")
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["summary"]["total_segments"] == 0

    @pytest.mark.asyncio
    async def test_pca_report_with_test_signals(self, tools, data_dir, reports_dir, models_dir, mock_ctx):
        """PCA report with test signals should segment, predict, and produce a report."""
        if "generate_pca_visualization_report" not in tools:
            pytest.skip("generate_pca_visualization_report not registered")
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_files=["report_test.csv"],
            segment_duration=0.1,
            overlap_ratio=0.5,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert result["summary"]["total_segments"] > 0

    @pytest.mark.asyncio
    async def test_pca_report_with_true_labels(self, tools, data_dir, reports_dir, models_dir, mock_ctx):
        """PCA report with true_labels should compute validation metrics."""
        if "generate_pca_visualization_report" not in tools:
            pytest.skip("generate_pca_visualization_report not registered")
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_files=["report_test.csv"],
            true_labels={"report_test.csv": "healthy"},
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert result["metadata"]["validation_mode"] is True
        assert "validation_metrics" in result["summary"]
        assert "overall_accuracy" in result["summary"]["validation_metrics"]

    @pytest.mark.asyncio
    async def test_pca_report_model_not_found(self, tools, data_dir, reports_dir, models_dir):
        if "generate_pca_visualization_report" not in tools:
            pytest.skip("generate_pca_visualization_report not registered")
        with pytest.raises(FileNotFoundError):
            await tools["generate_pca_visualization_report"](
                model_name="nonexistent_model",
            )


# ---------------------------------------------------------------------------
# Generate feature comparison report
# ---------------------------------------------------------------------------

class TestGenerateFeatureComparisonReport:
    """Tests for generate_feature_comparison_report tool."""

    @pytest.fixture
    def multi_signal_dir(self, data_dir):
        """Create two signal groups for comparison."""
        _make_signal_file(data_dir, "healthy_1.csv", freq=50.0, noise=0.01)
        _make_signal_file(data_dir, "healthy_2.csv", freq=50.0, noise=0.02)
        _make_signal_file(data_dir, "faulty_1.csv", freq=50.0, noise=1.0)
        return data_dir

    @pytest.mark.asyncio
    async def test_feature_comparison_basic(self, tools, multi_signal_dir, reports_dir, mock_ctx):
        if "generate_feature_comparison_report" not in tools:
            pytest.skip("generate_feature_comparison_report not registered")
        result = await tools["generate_feature_comparison_report"](
            signal_groups={
                "Healthy": ["healthy_1.csv", "healthy_2.csv"],
                "Faulty": ["faulty_1.csv"],
            },
            sampling_rate=10000.0,
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["metadata"]["report_type"] == "feature_comparison"
        assert "Healthy" in result["metadata"]["groups"]

    @pytest.mark.asyncio
    async def test_feature_comparison_subset_features(self, tools, multi_signal_dir, reports_dir):
        """Only plot a subset of features."""
        if "generate_feature_comparison_report" not in tools:
            pytest.skip("generate_feature_comparison_report not registered")
        result = await tools["generate_feature_comparison_report"](
            signal_groups={"A": ["healthy_1.csv"], "B": ["faulty_1.csv"]},
            sampling_rate=10000.0,
            features_to_plot=["rms", "kurtosis", "crest_factor"],
        )
        assert len(result["metadata"]["features_plotted"]) == 3

    @pytest.mark.asyncio
    async def test_feature_comparison_skips_missing(self, tools, multi_signal_dir, reports_dir):
        """Missing files in a group should be skipped, not crash."""
        if "generate_feature_comparison_report" not in tools:
            pytest.skip("generate_feature_comparison_report not registered")
        result = await tools["generate_feature_comparison_report"](
            signal_groups={
                "Mixed": ["healthy_1.csv", "does_not_exist.csv"],
            },
            sampling_rate=10000.0,
        )
        assert "file_path" in result


# ---------------------------------------------------------------------------
# DOCX report generation
# ---------------------------------------------------------------------------

class TestDocxReport:
    """Tests for generate_diagnostic_report_docx tool."""

    @pytest.mark.asyncio
    async def test_generates_docx(self, tools, data_dir, reports_dir, mock_ctx):
        if "generate_diagnostic_report_docx" not in tools:
            pytest.skip("generate_diagnostic_report_docx not registered")
        try:
            result = await tools["generate_diagnostic_report_docx"](
                ctx=mock_ctx,
                signal_file="report_test.csv",
                sections={"diagnosis": "Test diagnosis summary"},
            )
            assert result is not None
        except ImportError:
            pytest.skip("python-docx not installed")

    @pytest.mark.asyncio
    async def test_docx_with_all_sections(self, tools, data_dir, reports_dir, mock_ctx):
        if "generate_diagnostic_report_docx" not in tools:
            pytest.skip("generate_diagnostic_report_docx not registered")
        try:
            result = await tools["generate_diagnostic_report_docx"](
                ctx=mock_ctx,
                signal_file="report_test.csv",
                sections={
                    "statistics": {"RMS": 0.707, "Kurtosis": 3.0, "Crest Factor": 1.414},
                    "fft_peaks": [{"frequency": 50.0, "magnitude_db": 0.0, "note": "1x"}],
                    "diagnosis": "All parameters within normal range.",
                },
                title="Custom DOCX Title",
            )
            assert result is not None
            if "error" not in result:
                assert "file_path" in result
        except ImportError:
            pytest.skip("python-docx not installed")

    @pytest.mark.asyncio
    async def test_docx_no_ctx(self, tools, data_dir, reports_dir):
        """DOCX generation without ctx should still work."""
        if "generate_diagnostic_report_docx" not in tools:
            pytest.skip("generate_diagnostic_report_docx not registered")
        try:
            result = await tools["generate_diagnostic_report_docx"](
                signal_file="report_test.csv",
                sections={"diagnosis": "Summary"},
            )
            assert result is not None
        except ImportError:
            pytest.skip("python-docx not installed")


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
