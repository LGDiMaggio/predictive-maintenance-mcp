"""Tests for MCP diagnostics tools (ISO 13374 Blocks 3-4)."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock

from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = FastMCP("test-diagnostics")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Temp directory with synthetic signals for diagnostics testing."""
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 2.0, 2 * fs, endpoint=False)

    # Normal signal: low-level broadband noise
    normal = 0.05 * np.random.randn(len(t))
    pd.DataFrame(normal).to_csv(signals_dir / "normal.csv", index=False, header=False)
    with open(signals_dir / "normal_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Signal with known velocity RMS (for ISO 10816 testing)
    # Generate a signal that, when integrated, produces ~5 mm/s RMS velocity
    freq = 50.0  # Hz
    amplitude = 0.5  # g
    accel = amplitude * np.sin(2 * np.pi * freq * t)
    pd.DataFrame(accel).to_csv(signals_dir / "iso_test.csv", index=False, header=False)
    with open(signals_dir / "iso_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Patch directories
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_loader.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_repository.DATA_DIR", signals_dir)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR", models_dir)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.REPORTS_DIR", reports_dir)

    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()
    (resources_dir / "machine_manuals").mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.RESOURCES_DIR", resources_dir)

    cache_dir = resources_dir / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.CACHE_DIR", cache_dir)

    return signals_dir


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# ISO 20816 evaluation
# ---------------------------------------------------------------------------

class TestEvaluateISO20816:
    """Tests for evaluate_iso_20816 tool."""

    @pytest.mark.asyncio
    async def test_returns_result(self, tools, data_dir, mock_ctx):
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
        )
        assert result is not None
        # Should have a zone classification
        assert hasattr(result, "zone") or hasattr(result, "severity_zone")


class TestAssessVibrationSeverity:
    """Tests for assess_vibration_severity tool."""

    @pytest.mark.asyncio
    async def test_returns_severity(self, tools, data_dir, mock_ctx):
        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="sev_test", sampling_rate=10000)
        try:
            result = await tools["assess_vibration_severity"](
                ctx=mock_ctx, signal_id="sev_test"
            )
            assert result is not None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Bearing diagnostics
# ---------------------------------------------------------------------------

class TestBearingTools:
    """Tests for bearing fault detection tools."""

    @pytest.mark.asyncio
    async def test_get_bearing_catalog(self, tools, data_dir, mock_ctx):
        if "get_bearing_catalog" not in tools:
            pytest.skip("get_bearing_catalog not registered")
        result = await tools["get_bearing_catalog"](ctx=mock_ctx)
        assert result is not None

    @pytest.mark.asyncio
    async def test_calculate_bearing_frequencies(self, tools, data_dir, mock_ctx):
        if "calculate_bearing_frequencies" not in tools:
            pytest.skip("calculate_bearing_frequencies not registered")
        result = await tools["calculate_bearing_frequencies"](
            ctx=mock_ctx, bearing_id="6205", rpm=1800.0
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_check_bearing_faults_direct(self, tools, data_dir, mock_ctx):
        if "check_bearing_faults_direct" not in tools:
            pytest.skip("check_bearing_faults_direct not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="bearing_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults_direct"](
                ctx=mock_ctx,
                signal_id="bearing_test",
                bearing_id="6205",
                rpm=1800.0,
            )
            assert result is not None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Anomaly detection (train + predict)
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Tests for train_anomaly_model and predict_anomalies tools."""

    @pytest.fixture
    def training_signals(self, data_dir):
        """Create multiple training signal files."""
        fs = 10000
        for i in range(5):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            sig = 0.05 * np.random.randn(len(t))
            pd.DataFrame(sig).to_csv(data_dir / f"train_{i}.csv", index=False, header=False)
            with open(data_dir / f"train_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
        return [f"train_{i}.csv" for i in range(5)]

    @pytest.mark.asyncio
    async def test_train_and_predict(self, tools, data_dir, mock_ctx, training_signals):
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        # Train
        train_result = await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            model_name="test_model",
            sampling_rate=10000.0,
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        assert train_result is not None

        # Predict
        if "predict_anomalies" not in tools:
            pytest.skip("predict_anomalies not registered")

        predict_result = await tools["predict_anomalies"](
            signal_file="train_0.csv",
            model_name="test_model",
            ctx=mock_ctx,
        )
        assert predict_result is not None


# ---------------------------------------------------------------------------
# Integrated diagnosis
# ---------------------------------------------------------------------------

class TestDiagnoseVibration:
    """Tests for diagnose_vibration tool."""

    @pytest.mark.asyncio
    async def test_full_diagnosis(self, tools, data_dir, mock_ctx):
        if "diagnose_vibration" not in tools:
            pytest.skip("diagnose_vibration not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="diag_test", sampling_rate=10000)
        try:
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_test",
                rpm=1800.0,
            )
            assert result is not None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Extended ISO 20816 tests
# ---------------------------------------------------------------------------

class TestEvaluateISO20816Extended:
    """Additional coverage for evaluate_iso_20816 tool."""

    @pytest.mark.asyncio
    async def test_zone_classification_is_valid(self, tools, data_dir, mock_ctx):
        """Zone must be one of A/B/C/D."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
        )
        assert result.zone in ("A", "B", "C", "D")
        assert result.rms_velocity > 0

    @pytest.mark.asyncio
    async def test_flexible_support(self, tools, data_dir, mock_ctx):
        """Flexible support uses different zone boundaries."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="flexible",
            sampling_rate=10000.0,
        )
        assert result.zone in ("A", "B", "C", "D")
        # Flexible boundaries are larger than rigid: AB=2.3 vs 1.4
        assert result.boundary_ab == 2.3

    @pytest.mark.asyncio
    async def test_group1_rigid(self, tools, data_dir, mock_ctx):
        """Group 1 rigid machines use different thresholds."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=1,
            support_type="rigid",
            sampling_rate=10000.0,
        )
        assert result.machine_group == 1
        assert result.boundary_ab == 2.3

    @pytest.mark.asyncio
    async def test_group1_flexible(self, tools, data_dir, mock_ctx):
        """Group 1 flexible machines."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=1,
            support_type="flexible",
            sampling_rate=10000.0,
        )
        assert result.boundary_ab == 3.5

    @pytest.mark.asyncio
    async def test_explicit_signal_unit_g(self, tools, data_dir, mock_ctx):
        """Passing signal_unit='g' explicitly should still work."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
            signal_unit="g",
        )
        assert result.rms_velocity > 0

    @pytest.mark.asyncio
    async def test_velocity_signal_mm_s(self, tools, data_dir, mock_ctx):
        """Signal already in mm/s should skip conversion."""
        # Create a velocity signal in mm/s
        fs = 10000
        t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
        vel_signal = 3.0 * np.sin(2 * np.pi * 50 * t)  # ~2.12 mm/s RMS
        pd.DataFrame(vel_signal).to_csv(data_dir / "vel_signal.csv", index=False, header=False)
        with open(data_dir / "vel_signal_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "mm/s"}, f)

        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="vel_signal.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
        )
        assert result.rms_velocity > 0
        # For a 3.0 mm/s amplitude sine, RMS ~ 2.12
        assert result.zone in ("A", "B", "C", "D")

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tools, data_dir, mock_ctx):
        """Requesting a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await tools["evaluate_iso_20816"](
                ctx=mock_ctx,
                signal_file="nonexistent.csv",
                machine_group=2,
                support_type="rigid",
                sampling_rate=10000.0,
            )

    @pytest.mark.asyncio
    async def test_operating_speed_rpm_low(self, tools, data_dir, mock_ctx):
        """Low RPM should use 2-1000 Hz frequency range."""
        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="iso_test.csv",
            machine_group=2,
            support_type="rigid",
            sampling_rate=10000.0,
            operating_speed_rpm=400.0,
        )
        assert result.operating_speed_rpm == 400.0
        assert "2-1000" in result.frequency_range

    @pytest.mark.asyncio
    async def test_no_metadata_no_explicit_rate(self, tools, data_dir, mock_ctx):
        """Signal without metadata and default sampling_rate triggers ctx warnings."""
        # Create a signal with no metadata file
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 0.1 * np.random.randn(len(t))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta.csv", index=False, header=False)

        result = await tools["evaluate_iso_20816"](
            ctx=mock_ctx,
            signal_file="no_meta.csv",
        )
        assert result is not None
        assert result.zone in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Extended bearing diagnostics tests
# ---------------------------------------------------------------------------

class TestBearingToolsExtended:
    """Extended tests for bearing fault detection tools."""

    @pytest.mark.asyncio
    async def test_check_bearing_fault_peak_single(self, tools, data_dir, mock_ctx):
        """Check a single fault type (BPFO)."""
        if "check_bearing_fault_peak_tool" not in tools:
            pytest.skip("check_bearing_fault_peak_tool not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="peak_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_fault_peak_tool"](
                ctx=mock_ctx,
                signal_id="peak_test",
                bearing_id="6205",
                fault_type="BPFO",
                rpm=1800.0,
            )
            assert result is not None
            assert result.fault_type == "BPFO"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_check_bearing_fault_peak_bpfi(self, tools, data_dir, mock_ctx):
        """Check BPFI fault type."""
        if "check_bearing_fault_peak_tool" not in tools:
            pytest.skip("check_bearing_fault_peak_tool not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="bpfi_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_fault_peak_tool"](
                ctx=mock_ctx,
                signal_id="bpfi_test",
                bearing_id="6205",
                fault_type="BPFI",
                rpm=1800.0,
            )
            assert result.fault_type == "BPFI"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_check_bearing_fault_peak_invalid_fault_type(self, tools, data_dir, mock_ctx):
        """Invalid fault_type should raise ValueError."""
        if "check_bearing_fault_peak_tool" not in tools:
            pytest.skip("check_bearing_fault_peak_tool not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="invalid_ft", sampling_rate=10000)
        try:
            with pytest.raises(ValueError, match="Invalid fault_type"):
                await tools["check_bearing_fault_peak_tool"](
                    ctx=mock_ctx,
                    signal_id="invalid_ft",
                    bearing_id="6205",
                    fault_type="INVALID",
                    rpm=1800.0,
                )
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_check_bearing_faults_direct_returns_summary(self, tools, data_dir, mock_ctx):
        """check_bearing_faults_direct should return a BearingFaultsSummary with all 4 checks."""
        if "check_bearing_faults_direct" not in tools:
            pytest.skip("check_bearing_faults_direct not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="faults_all", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults_direct"](
                ctx=mock_ctx,
                signal_id="faults_all",
                bearing_id="6205",
                rpm=1800.0,
            )
            # Should have 4 fault checks (BPFO, BPFI, BSF, FTF)
            assert len(result.fault_checks) == 4
            fault_types = {fc.fault_type for fc in result.fault_checks}
            assert fault_types == {"BPFO", "BPFI", "BSF", "FTF"}
            assert result.bearing_id == "6205"
            assert result.rpm == 1800.0
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_lookup_bearing_and_compute_tool(self, tools, data_dir, mock_ctx):
        """End-to-end: lookup bearing + compute freqs + check signal."""
        if "lookup_bearing_and_compute_tool" not in tools:
            pytest.skip("lookup_bearing_and_compute_tool not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="lookup_test", sampling_rate=10000)
        try:
            result = await tools["lookup_bearing_and_compute_tool"](
                ctx=mock_ctx,
                bearing_type="6205",
                rpm=1800.0,
                signal_id="lookup_test",
            )
            assert result is not None
            assert result.bearing_id == "6205"
            assert len(result.fault_checks) == 4
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_check_bearing_faults_unknown_bearing(self, tools, data_dir, mock_ctx):
        """Unknown bearing ID should raise ValueError."""
        if "check_bearing_faults_direct" not in tools:
            pytest.skip("check_bearing_faults_direct not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="unknown_brg", sampling_rate=10000)
        try:
            with pytest.raises((ValueError, KeyError)):
                await tools["check_bearing_faults_direct"](
                    ctx=mock_ctx,
                    signal_id="unknown_brg",
                    bearing_id="NONEXISTENT_999",
                    rpm=1800.0,
                )
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Extended anomaly detection tests
# ---------------------------------------------------------------------------

class TestAnomalyDetectionExtended:
    """Additional coverage for anomaly detection tools."""

    @pytest.fixture
    def training_signals(self, data_dir):
        """Create multiple training signal files."""
        fs = 10000
        for i in range(5):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            sig = 0.05 * np.random.randn(len(t))
            pd.DataFrame(sig).to_csv(data_dir / f"train_{i}.csv", index=False, header=False)
            with open(data_dir / f"train_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
        return [f"train_{i}.csv" for i in range(5)]

    @pytest.fixture
    def fault_signals(self, data_dir):
        """Create fault signal files with higher amplitude."""
        fs = 10000
        for i in range(3):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            # Much higher amplitude + periodic component to simulate fault
            sig = 2.0 * np.random.randn(len(t)) + 5.0 * np.sin(2 * np.pi * 100 * t)
            pd.DataFrame(sig).to_csv(data_dir / f"fault_{i}.csv", index=False, header=False)
            with open(data_dir / f"fault_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
        return [f"fault_{i}.csv" for i in range(3)]

    @pytest.mark.asyncio
    async def test_train_lof_model(self, tools, data_dir, mock_ctx, training_signals):
        """Train with LocalOutlierFactor model type."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        result = await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            model_name="lof_model",
            sampling_rate=10000.0,
            segment_duration=0.5,
            model_type="LocalOutlierFactor",
            ctx=mock_ctx,
        )
        assert result is not None
        assert result.model_type == "LocalOutlierFactor"

    @pytest.mark.asyncio
    async def test_train_invalid_model_type(self, tools, data_dir, mock_ctx, training_signals):
        """Invalid model type should raise ValueError."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        with pytest.raises(ValueError, match="Unknown model_type"):
            await tools["train_anomaly_model"](
                healthy_signal_files=training_signals,
                model_name="bad_model",
                sampling_rate=10000.0,
                segment_duration=0.5,
                model_type="InvalidModel",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_predict_missing_model(self, tools, data_dir, mock_ctx):
        """Predicting with non-existent model should raise FileNotFoundError."""
        if "predict_anomalies" not in tools:
            pytest.skip("predict_anomalies not registered")

        with pytest.raises(FileNotFoundError, match="Model not found"):
            await tools["predict_anomalies"](
                signal_file="normal.csv",
                model_name="nonexistent_model",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_predict_missing_signal(self, tools, data_dir, mock_ctx, training_signals):
        """Predicting on non-existent signal should raise FileNotFoundError."""
        if "train_anomaly_model" not in tools or "predict_anomalies" not in tools:
            pytest.skip("tools not registered")

        # Train first
        await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            model_name="pred_test_model",
            sampling_rate=10000.0,
            segment_duration=0.5,
            ctx=mock_ctx,
        )

        with pytest.raises(FileNotFoundError):
            await tools["predict_anomalies"](
                signal_file="missing_signal.csv",
                model_name="pred_test_model",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_train_semi_supervised_svm(self, tools, data_dir, mock_ctx, training_signals, fault_signals):
        """Semi-supervised SVM training with fault signals for hyperparameter tuning."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        result = await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            fault_signal_files=fault_signals,
            model_name="semi_svm",
            sampling_rate=10000.0,
            segment_duration=0.5,
            model_type="OneClassSVM",
            ctx=mock_ctx,
        )
        assert result is not None
        assert result.validation_accuracy is not None

    @pytest.mark.asyncio
    async def test_train_semi_supervised_lof(self, tools, data_dir, mock_ctx, training_signals, fault_signals):
        """Semi-supervised LOF training with fault signals."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        result = await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            fault_signal_files=fault_signals,
            model_name="semi_lof",
            sampling_rate=10000.0,
            segment_duration=0.5,
            model_type="LocalOutlierFactor",
            ctx=mock_ctx,
        )
        assert result is not None
        assert result.model_type == "LocalOutlierFactor"

    @pytest.mark.asyncio
    async def test_predict_health_assessment(self, tools, data_dir, mock_ctx, training_signals):
        """Predict on healthy signal should give 'Healthy' or 'Suspicious' health."""
        if "train_anomaly_model" not in tools or "predict_anomalies" not in tools:
            pytest.skip("tools not registered")

        await tools["train_anomaly_model"](
            healthy_signal_files=training_signals,
            model_name="health_model",
            sampling_rate=10000.0,
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        result = await tools["predict_anomalies"](
            signal_file="train_0.csv",
            model_name="health_model",
            ctx=mock_ctx,
        )
        assert result.overall_health in ("Healthy", "Suspicious", "Faulty")
        assert result.confidence in ("High", "Medium")
        assert result.num_segments > 0
        assert 0 <= result.anomaly_ratio <= 1.0


# ---------------------------------------------------------------------------
# Documentation / manual tools
# ---------------------------------------------------------------------------

class TestDocumentationTools:
    """Tests for list_machine_manuals, read_manual_excerpt, extract_manual_specs."""

    @pytest.mark.asyncio
    async def test_list_machine_manuals_empty(self, tools, data_dir, mock_ctx):
        """Listing manuals in empty directory returns empty list."""
        if "list_machine_manuals" not in tools:
            pytest.skip("list_machine_manuals not registered")

        result = await tools["list_machine_manuals"](ctx=mock_ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_machine_manuals_with_txt(self, tools, data_dir, mock_ctx, tmp_path):
        """Text file manuals should appear in the listing."""
        if "list_machine_manuals" not in tools:
            pytest.skip("list_machine_manuals not registered")

        # Write a .txt manual to the resources/machine_manuals dir
        manuals_dir = tmp_path / "resources" / "machine_manuals"
        manuals_dir.mkdir(parents=True, exist_ok=True)
        (manuals_dir / "test_manual.txt").write_text("Bearing 6205 installed on drive end.", encoding="utf-8")

        result = await tools["list_machine_manuals"](ctx=mock_ctx)
        filenames = [m["filename"] for m in result]
        assert "test_manual.txt" in filenames

    @pytest.mark.asyncio
    async def test_read_manual_excerpt_txt(self, tools, data_dir, mock_ctx, tmp_path):
        """Reading a .txt manual should return its content."""
        if "read_manual_excerpt" not in tools:
            pytest.skip("read_manual_excerpt not registered")

        manuals_dir = tmp_path / "resources" / "machine_manuals"
        manuals_dir.mkdir(parents=True, exist_ok=True)
        manual_content = "Bearing: SKF 6205-2RS\nOperating speed: 1800 RPM\nPower: 15 kW"
        (manuals_dir / "pump_manual.txt").write_text(manual_content, encoding="utf-8")

        result = await tools["read_manual_excerpt"](
            ctx=mock_ctx,
            manual_filename="pump_manual.txt",
        )
        assert "SKF 6205-2RS" in result
        assert "1800 RPM" in result

    @pytest.mark.asyncio
    async def test_read_manual_excerpt_missing(self, tools, data_dir, mock_ctx):
        """Reading non-existent manual should raise FileNotFoundError."""
        if "read_manual_excerpt" not in tools:
            pytest.skip("read_manual_excerpt not registered")

        with pytest.raises(FileNotFoundError):
            await tools["read_manual_excerpt"](
                ctx=mock_ctx,
                manual_filename="nonexistent_manual.pdf",
            )

    @pytest.mark.asyncio
    async def test_extract_manual_specs_missing(self, tools, data_dir, mock_ctx):
        """Extracting specs from non-existent manual should raise FileNotFoundError."""
        if "extract_manual_specs" not in tools:
            pytest.skip("extract_manual_specs not registered")

        with pytest.raises(FileNotFoundError):
            await tools["extract_manual_specs"](
                ctx=mock_ctx,
                manual_filename="nonexistent.pdf",
            )


# ---------------------------------------------------------------------------
# Bearing catalog tools
# ---------------------------------------------------------------------------

class TestBearingCatalogTools:
    """Tests for search_bearing_catalog and calculate_bearing_characteristic_frequencies."""

    @pytest.mark.asyncio
    async def test_search_bearing_catalog_known(self, tools, data_dir, mock_ctx):
        """Search for a known bearing (6205) should return specs."""
        if "search_bearing_catalog" not in tools:
            pytest.skip("search_bearing_catalog not registered")

        result = await tools["search_bearing_catalog"](
            ctx=mock_ctx,
            bearing_designation="6205",
        )
        # Should find bearing 6205 (at least in the in-memory fallback)
        assert result is not None
        if "error" not in result:
            assert result["num_balls"] == 9
            assert result["ball_diameter_mm"] > 0

    @pytest.mark.asyncio
    async def test_search_bearing_catalog_unknown(self, tools, data_dir, mock_ctx):
        """Search for a non-existent bearing should return error dict."""
        if "search_bearing_catalog" not in tools:
            pytest.skip("search_bearing_catalog not registered")

        result = await tools["search_bearing_catalog"](
            ctx=mock_ctx,
            bearing_designation="ZZZZZ999",
        )
        assert "error" in result or "suggestion" in result

    @pytest.mark.asyncio
    async def test_search_bearing_catalog_with_prefix(self, tools, data_dir, mock_ctx):
        """Searching with manufacturer prefix (e.g. 'SKF 6205-2RS') should strip it."""
        if "search_bearing_catalog" not in tools:
            pytest.skip("search_bearing_catalog not registered")

        result = await tools["search_bearing_catalog"](
            ctx=mock_ctx,
            bearing_designation="SKF 6205-2RS",
        )
        assert result is not None
        if "error" not in result:
            assert result["num_balls"] == 9

    @pytest.mark.asyncio
    async def test_calculate_bearing_frequencies_tool(self, tools, data_dir, mock_ctx):
        """Calculate frequencies for known SKF 6205 geometry at 1797 RPM."""
        if "calculate_bearing_characteristic_frequencies" not in tools:
            pytest.skip("calculate_bearing_characteristic_frequencies not registered")

        result = await tools["calculate_bearing_characteristic_frequencies"](
            ctx=mock_ctx,
            num_balls=9,
            ball_diameter_mm=7.94,
            pitch_diameter_mm=34.55,
            contact_angle_deg=0.0,
            shaft_speed_rpm=1797.0,
        )
        assert "BPFO" in result
        assert "BPFI" in result
        assert "BSF" in result
        assert "FTF" in result
        # BPFO for 6205 at 1797 RPM should be approximately 107 Hz
        assert result["BPFO"] > 50
        assert result["BPFI"] > result["BPFO"]  # BPFI > BPFO always


# ---------------------------------------------------------------------------
# ISO 20816 chart
# ---------------------------------------------------------------------------

class TestPlotISO20816Chart:
    """Tests for plot_iso_20816_chart tool."""

    @pytest.mark.asyncio
    async def test_generates_html(self, tools, data_dir, mock_ctx, tmp_path):
        """Chart tool should generate an HTML file."""
        if "plot_iso_20816_chart" not in tools:
            pytest.skip("plot_iso_20816_chart not registered")

        result = await tools["plot_iso_20816_chart"](
            ctx=mock_ctx,
            filename="iso_test.csv",
            sampling_rate=10000.0,
            machine_group=2,
            support_type="rigid",
        )
        assert "plot_iso_" in result
        assert ".html" in result


# ---------------------------------------------------------------------------
# Diagnose vibration with bearing
# ---------------------------------------------------------------------------

class TestDiagnoseVibrationExtended:
    """Extended tests for diagnose_vibration tool."""

    @pytest.mark.asyncio
    async def test_diagnosis_with_bearing(self, tools, data_dir, mock_ctx):
        """Full diagnosis including bearing fault detection."""
        if "diagnose_vibration_tool" not in tools:
            pytest.skip("diagnose_vibration_tool not registered")

        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="diag_brg", sampling_rate=10000)
        try:
            result = await tools["diagnose_vibration_tool"](
                ctx=mock_ctx,
                signal_id="diag_brg",
                rpm=1800.0,
                bearing_id="6205",
            )
            assert result is not None
            assert result.bearing_faults is not None
            assert result.iso_severity is not None
            assert result.confidence.lower() in ("low", "medium", "high")
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_diagnosis_missing_signal(self, tools, data_dir, mock_ctx):
        """Diagnosis on non-existent signal_id should raise."""
        if "diagnose_vibration_tool" not in tools:
            pytest.skip("diagnose_vibration_tool not registered")

        with pytest.raises((KeyError, ValueError)):
            await tools["diagnose_vibration_tool"](
                ctx=mock_ctx,
                signal_id="nonexistent_signal_id",
                rpm=1800.0,
            )


# ---------------------------------------------------------------------------
# Assess vibration severity extended
# ---------------------------------------------------------------------------

class TestAssessVibrationSeverityExtended:
    """Additional tests for assess_vibration_severity tool."""

    @pytest.mark.asyncio
    async def test_group1_boundaries(self, tools, data_dir, mock_ctx):
        """machine_group=1 uses the large-machine boundaries (A/B at 2.3 mm/s)."""
        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="class_test", sampling_rate=10000)
        try:
            result = await tools["assess_vibration_severity"](
                ctx=mock_ctx,
                signal_id="class_test",
                machine_group=1,
                support_type="rigid",
            )
            assert result is not None
            assert result.zone in ("A", "B", "C", "D")
            assert result.machine_group == 1
            assert result.support_type == "rigid"
            assert result.boundaries["AB"] == 2.3
            assert "10816-3:2009" in result.threshold_provenance
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_machine_class_parameter_removed(self, tools, data_dir, mock_ctx):
        """The invented machine_class vocabulary is gone from the tool."""
        from predictive_maintenance_mcp.signal_repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="class_test", sampling_rate=10000)
        try:
            with pytest.raises(TypeError):
                await tools["assess_vibration_severity"](
                    ctx=mock_ctx,
                    signal_id="class_test",
                    machine_class="III",
                )
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_signal_without_metadata_sampling_rate(self, tools, data_dir, mock_ctx):
        """Signal with no metadata and no explicit sampling_rate should raise."""
        from predictive_maintenance_mcp.signal_repository import get_repository

        # Create a signal file with NO companion metadata
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 0.1 * np.random.randn(len(t))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta_sr.csv", index=False, header=False)
        # Deliberately do NOT create no_meta_sr_metadata.json

        repo = get_repository()
        try:
            repo.load_signal("no_meta_sr.csv", signal_id="no_sr_test")
            with pytest.raises((ValueError, TypeError)):
                await tools["assess_vibration_severity"](
                    ctx=mock_ctx,
                    signal_id="no_sr_test",
                )
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# RAG / documentation search
# ---------------------------------------------------------------------------

class TestSearchDocumentation:
    """Tests for search_documentation tool."""

    @pytest.mark.asyncio
    async def test_search_empty_resources(self, tools, data_dir, mock_ctx):
        """Search with no documents should return empty results."""
        if "search_documentation" not in tools:
            pytest.skip("search_documentation not registered")

        result = await tools["search_documentation"](
            ctx=mock_ctx,
            query="bearing 6205",
        )
        assert result is not None
        assert "results" in result
        # Empty resources dir -> empty results or note about no documents
        assert isinstance(result["results"], list)
