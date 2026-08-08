"""Tests for MCP diagnostics tools (ISO 13374 Blocks 3-4).

Since U8 every diagnostics tool takes signal_id as its only signal handle:
signals are loaded once via the repository and referenced by id.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock

from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp():
    server = MCPServer("test-diagnostics")
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

    # Signal with known velocity RMS (for ISO 20816-3 testing)
    # Generate a signal that, when integrated, produces ~5 mm/s RMS velocity
    freq = 50.0  # Hz
    amplitude = 0.5  # g
    accel = amplitude * np.sin(2 * np.pi * freq * t)
    pd.DataFrame(accel).to_csv(signals_dir / "iso_test.csv", index=False, header=False)
    with open(signals_dir / "iso_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    # Patch directories (diagnostics_tools now only holds MODELS_DIR and
    # RESOURCES_DIR — DATA_DIR/REPORTS_DIR/CACHE_DIR imports were dead and
    # removed in the U9b naming/import sweep).
    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir)
    monkeypatch.setattr("predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR", models_dir)

    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()
    (resources_dir / "machine_manuals").mkdir()
    monkeypatch.setattr("predictive_maintenance_mcp.mcp_tools.diagnostics_tools.RESOURCES_DIR", resources_dir)

    return signals_dir


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
def repo(data_dir):
    """Repository with the diagnostics signals loaded; cleaned afterwards."""
    repo = get_repository()
    repo.clear_all()
    repo.load_signal("normal.csv")    # metadata: fs=10000, unit 'g'
    repo.load_signal("iso_test.csv")  # metadata: fs=10000, unit 'g'
    yield repo
    repo.clear_all()


# ---------------------------------------------------------------------------
# ISO 20816 evaluation
# ---------------------------------------------------------------------------

class TestAssessSeverity:
    """Tests for the unified assess_severity tool (U9 merge)."""

    @pytest.mark.asyncio
    async def test_signal_route_returns_result(self, tools, repo, mock_ctx):
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=2,
            support_type="rigid",
        )
        assert result is not None
        assert result.status == "assessed"
        assert result.zone in ("A", "B", "C", "D")

    @pytest.mark.asyncio
    async def test_rms_route_returns_result(self, tools, repo, mock_ctx):
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            rms_velocity_mm_s=3.0,
            machine_group=2,
            support_type="rigid",
        )
        assert result.zone == "C"  # 2.8 < 3.0 <= 4.5 (group 2 rigid)
        assert result.alert_level == "alarm"
        assert result.signal_id is None

    def test_old_tool_names_gone(self, tools):
        """Clean cut: the four absorbed severity/alert tools are gone."""
        for old in (
            "evaluate_iso_20816",
            "assess_vibration_severity",
            "check_vibration_alert",
            "check_custom_vibration_alert",
        ):
            assert old not in tools


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
    async def test_check_bearing_faults_catalog_route(self, tools, data_dir, mock_ctx):
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="bearing_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults"](
                ctx=mock_ctx,
                signal_id="bearing_test",
                bearing_id="6205",
                rpm=1800.0,
            )
            assert result is not None
            assert result.source  # catalog source echoed
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Anomaly detection (train + predict)
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    """Tests for train_anomaly_model and predict_anomalies tools."""

    @pytest.fixture
    def training_signals(self, data_dir, repo):
        """Create training signal files and load them as a BATCH (U8)."""
        fs = 10000
        files = []
        for i in range(5):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            sig = 0.05 * np.random.randn(len(t))
            pd.DataFrame(sig).to_csv(data_dir / f"train_{i}.csv", index=False, header=False)
            with open(data_dir / f"train_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
            files.append(f"train_{i}.csv")
        infos = repo.load_signals(files)
        return [info["signal_id"] for info in infos]

    @pytest.mark.asyncio
    async def test_train_and_predict(self, tools, data_dir, mock_ctx, training_signals):
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        # Train
        train_result = await tools["train_anomaly_model"](
            healthy_signal_ids=training_signals,
            model_name="test_model",
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        assert train_result is not None
        # U9 loop closure: the result echoes the name to pass to predict.
        assert train_result.model_name == "test_model"

        # Predict
        if "predict_anomalies" not in tools:
            pytest.skip("predict_anomalies not registered")

        predict_result = await tools["predict_anomalies"](
            signal_id="train_0",
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

        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
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

class TestAssessSeverityExtended:
    """Additional coverage for assess_severity (signal route)."""

    @pytest.mark.asyncio
    async def test_zone_classification_is_valid(self, tools, repo, mock_ctx):
        """Zone must be one of A/B/C/D."""
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=2,
            support_type="rigid",
        )
        assert result.zone in ("A", "B", "C", "D")
        assert result.rms_velocity_mm_s > 0

    @pytest.mark.asyncio
    async def test_flexible_support(self, tools, repo, mock_ctx):
        """Flexible support uses different zone boundaries."""
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=2,
            support_type="flexible",
        )
        assert result.zone in ("A", "B", "C", "D")
        # Flexible boundaries are larger than rigid: AB=2.3 vs 1.4
        assert result.boundaries["AB"] == 2.3

    @pytest.mark.asyncio
    async def test_group1_rigid(self, tools, repo, mock_ctx):
        """Group 1 rigid machines use different thresholds."""
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=1,
            support_type="rigid",
        )
        assert result.machine_group == 1
        assert result.boundaries["AB"] == 2.3

    @pytest.mark.asyncio
    async def test_group1_flexible(self, tools, repo, mock_ctx):
        """Group 1 flexible machines."""
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=1,
            support_type="flexible",
        )
        assert result.boundaries["AB"] == 3.5

    @pytest.mark.asyncio
    async def test_unit_declared_at_load_time(self, tools, data_dir, repo, mock_ctx):
        """The unit declared via load_signal(signal_unit=...) drives the
        evaluation (the tool itself no longer takes a unit parameter)."""
        fs = 10000
        t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(data_dir / "load_unit.csv", index=False, header=False)
        repo.load_signal("load_unit.csv", sampling_rate=fs, signal_unit="g")

        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="load_unit",
            machine_group=2,
            support_type="rigid",
        )
        assert result.rms_velocity_mm_s > 0

    @pytest.mark.asyncio
    async def test_velocity_signal_mm_s(self, tools, data_dir, repo, mock_ctx):
        """Signal already in mm/s should skip conversion."""
        # Create a velocity signal in mm/s
        fs = 10000
        t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
        vel_signal = 3.0 * np.sin(2 * np.pi * 50 * t)  # ~2.12 mm/s RMS
        pd.DataFrame(vel_signal).to_csv(data_dir / "vel_signal.csv", index=False, header=False)
        with open(data_dir / "vel_signal_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "mm/s"}, f)
        repo.load_signal("vel_signal.csv")

        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="vel_signal",
            machine_group=2,
            support_type="rigid",
        )
        assert result.rms_velocity_mm_s > 0
        # For a 3.0 mm/s amplitude sine, RMS ~ 2.12
        assert result.zone in ("A", "B", "C", "D")

    @pytest.mark.asyncio
    async def test_signal_not_loaded_raises(self, tools, repo, mock_ctx):
        """Unknown signal_id → standard error naming load_signal."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["assess_severity"](
                ctx=mock_ctx,
                signal_id="nonexistent",
                machine_group=2,
                support_type="rigid",
            )

    @pytest.mark.asyncio
    async def test_operating_speed_rpm_low(self, tools, repo, mock_ctx):
        """Low RPM should use 2-1000 Hz frequency range."""
        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="iso_test",
            machine_group=2,
            support_type="rigid",
            rpm=400.0,
        )
        assert result.operating_speed_rpm == 400.0
        assert "2-1000" in result.frequency_range

    @pytest.mark.asyncio
    async def test_stored_signal_without_rate_raises(self, tools, data_dir, repo, mock_ctx):
        """Signal stored without a rate → structured error, no silent 10 kHz."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 0.1 * np.random.randn(len(t))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta.csv", index=False, header=False)
        repo.load_signal("no_meta.csv")  # no metadata → no rate

        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["assess_severity"](
                ctx=mock_ctx,
                signal_id="no_meta",
            )

    @pytest.mark.asyncio
    async def test_undeclared_unit_raises(self, tools, data_dir, repo, mock_ctx):
        """Rate known but unit undeclared → structured error naming the fix
        (no RMS-based 'HYPOTHESIS ... PROCEEDING' guess)."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        # RMS ~4 (the amplitude the old heuristic guessed as 'g')
        sig = 4.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(data_dir / "no_unit.csv", index=False, header=False)
        with open(data_dir / "no_unit_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs}, f)  # rate but NO signal_unit
        repo.load_signal("no_unit.csv")

        with pytest.raises(ValueError, match="unit not declared"):
            await tools["assess_severity"](
                ctx=mock_ctx,
                signal_id="no_unit",
            )

    @pytest.mark.asyncio
    async def test_velocity_declared_unit_no_integration(self, tools, data_dir, repo, mock_ctx):
        """mm/s declared → severity without acc→vel integration, correct zone."""
        fs = 10000
        t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
        # RMS = 4.0 mm/s → Group 2 rigid: zone C (2.8 < 4.0 <= 4.5)
        vel = 4.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(vel).to_csv(data_dir / "vel4.csv", index=False, header=False)
        with open(data_dir / "vel4_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "mm/s"}, f)
        repo.load_signal("vel4.csv")

        result = await tools["assess_severity"](
            ctx=mock_ctx,
            signal_id="vel4",
            machine_group=2,
            support_type="rigid",
        )
        assert result.zone == "C"
        assert result.rms_velocity_mm_s == pytest.approx(4.0, rel=0.05)
        assert "CONVERTED" not in result.zone_description


# ---------------------------------------------------------------------------
# Extended bearing diagnostics tests
# ---------------------------------------------------------------------------

class TestBearingToolsExtended:
    """Extended tests for the unified check_bearing_faults tool."""

    @pytest.mark.asyncio
    async def test_single_frequency_check_via_frequencies_route(
        self, tools, data_dir, mock_ctx
    ):
        """The former single-fault check is the frequencies route with one
        labeled entry."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="peak_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults"](
                ctx=mock_ctx,
                signal_id="peak_test",
                frequencies={"BPFO": 107.5},
                rpm=1800.0,
            )
            assert len(result.fault_checks) == 1
            check = result.fault_checks[0]
            assert check.fault_type == "BPFO"
            assert check.fault_type_canonical == "outer_race"
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_catalog_route_returns_summary(self, tools, data_dir, mock_ctx):
        """Catalog route returns a BearingFaultsSummary with all 4 checks."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="faults_all", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults"](
                ctx=mock_ctx,
                signal_id="faults_all",
                bearing_id="6205",
                rpm=1800.0,
            )
            # Should have 4 fault checks (BPFO, BPFI, BSF, FTF)
            assert len(result.fault_checks) == 4
            fault_types = {fc.fault_type for fc in result.fault_checks}
            assert fault_types == {"BPFO", "BPFI", "BSF", "FTF"}
            canonical = {fc.fault_type_canonical for fc in result.fault_checks}
            assert canonical == {"outer_race", "inner_race", "ball", "cage"}
            assert result.bearing_id == "6205"
            assert result.rpm == 1800.0
            assert result.source  # catalog entries carry a source citation
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_designation_with_prefix(self, tools, data_dir, mock_ctx):
        """Manufacturer-prefixed designations resolve via the catalog
        (former lookup_bearing_and_compute_tool path)."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="lookup_test", sampling_rate=10000)
        try:
            result = await tools["check_bearing_faults"](
                ctx=mock_ctx,
                bearing_id="SKF 6205-2RS",
                rpm=1800.0,
                signal_id="lookup_test",
            )
            assert result is not None
            assert len(result.fault_checks) == 4
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_unknown_bearing_raises(self, tools, data_dir, mock_ctx):
        """Unknown bearing ID should raise ValueError."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="unknown_brg", sampling_rate=10000)
        try:
            with pytest.raises((ValueError, KeyError)):
                await tools["check_bearing_faults"](
                    ctx=mock_ctx,
                    signal_id="unknown_brg",
                    bearing_id="NONEXISTENT_999",
                    rpm=1800.0,
                )
        finally:
            repo.clear_all()

    def test_old_bearing_tool_names_gone(self, tools):
        """Clean cut: the three absorbed bearing tools are gone."""
        for old in (
            "check_bearing_fault_peak_tool",
            "check_bearing_faults_direct",
            "lookup_bearing_and_compute_tool",
        ):
            assert old not in tools



# ---------------------------------------------------------------------------
# Extended anomaly detection tests
# ---------------------------------------------------------------------------

class TestAnomalyDetectionExtended:
    """Additional coverage for anomaly detection tools."""

    @pytest.fixture
    def training_signals(self, data_dir, repo):
        """Create training signal files and load them as a BATCH (U8)."""
        fs = 10000
        files = []
        for i in range(5):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            sig = 0.05 * np.random.randn(len(t))
            pd.DataFrame(sig).to_csv(data_dir / f"train_{i}.csv", index=False, header=False)
            with open(data_dir / f"train_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
            files.append(f"train_{i}.csv")
        infos = repo.load_signals(files)
        return [info["signal_id"] for info in infos]

    @pytest.fixture
    def fault_signals(self, data_dir, repo):
        """Create fault signal files (higher amplitude) and load them."""
        fs = 10000
        files = []
        for i in range(3):
            t = np.linspace(0, 1.0, fs, endpoint=False)
            # Much higher amplitude + periodic component to simulate fault
            sig = 2.0 * np.random.randn(len(t)) + 5.0 * np.sin(2 * np.pi * 100 * t)
            pd.DataFrame(sig).to_csv(data_dir / f"fault_{i}.csv", index=False, header=False)
            with open(data_dir / f"fault_{i}_metadata.json", "w") as f:
                json.dump({"sampling_rate": fs}, f)
            files.append(f"fault_{i}.csv")
        infos = repo.load_signals(files)
        return [info["signal_id"] for info in infos]

    @pytest.mark.asyncio
    async def test_train_lof_model(self, tools, data_dir, mock_ctx, training_signals):
        """Train with LocalOutlierFactor model type."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        result = await tools["train_anomaly_model"](
            healthy_signal_ids=training_signals,
            model_name="lof_model",
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
                healthy_signal_ids=training_signals,
                model_name="bad_model",
                segment_duration=0.5,
                model_type="InvalidModel",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_train_unloaded_ids_fail_fast(self, tools, data_dir, repo, mock_ctx):
        """Training with an unloaded signal_id raises the standard error —
        no silent skipping that would train a misleading model."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        with pytest.raises(ValueError, match="load_signal"):
            await tools["train_anomaly_model"](
                healthy_signal_ids=["__not_loaded__"],
                model_name="never_written",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_predict_missing_model(self, tools, data_dir, repo, mock_ctx):
        """Predicting with non-existent model should raise FileNotFoundError."""
        if "predict_anomalies" not in tools:
            pytest.skip("predict_anomalies not registered")

        with pytest.raises(FileNotFoundError, match="models on disk"):
            await tools["predict_anomalies"](
                signal_id="normal",
                model_name="nonexistent_model",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_predict_missing_signal(self, tools, data_dir, mock_ctx, training_signals):
        """Predicting on an unloaded signal_id should raise the standard error."""
        if "train_anomaly_model" not in tools or "predict_anomalies" not in tools:
            pytest.skip("tools not registered")

        # Train first
        await tools["train_anomaly_model"](
            healthy_signal_ids=training_signals,
            model_name="pred_test_model",
            segment_duration=0.5,
            ctx=mock_ctx,
        )

        with pytest.raises(ValueError, match="load_signal"):
            await tools["predict_anomalies"](
                signal_id="missing_signal",
                model_name="pred_test_model",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_train_semi_supervised_svm(self, tools, data_dir, mock_ctx, training_signals, fault_signals):
        """Semi-supervised SVM training with fault signals for hyperparameter tuning."""
        if "train_anomaly_model" not in tools:
            pytest.skip("train_anomaly_model not registered")

        result = await tools["train_anomaly_model"](
            healthy_signal_ids=training_signals,
            fault_signal_ids=fault_signals,
            model_name="semi_svm",
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
            healthy_signal_ids=training_signals,
            fault_signal_ids=fault_signals,
            model_name="semi_lof",
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
            healthy_signal_ids=training_signals,
            model_name="health_model",
            segment_duration=0.5,
            ctx=mock_ctx,
        )
        result = await tools["predict_anomalies"](
            signal_id="train_0",
            model_name="health_model",
            ctx=mock_ctx,
        )
        assert result.overall_health in ("Healthy", "Suspicious", "Faulty")
        # The invented per-prediction "confidence" label was removed.
        assert not hasattr(result, "confidence")
        assert result.num_segments > 0
        assert 0 <= result.anomaly_ratio <= 1.0
        # U9 bounded output: no per-segment arrays, name echoed, worst
        # segments capped at 10.
        assert result.model_name == "health_model"
        assert not hasattr(result, "predictions")
        assert not hasattr(result, "anomaly_scores")
        assert len(result.worst_segments) <= 10
        if result.score_percentiles is not None:
            assert set(result.score_percentiles) == {"p5", "p25", "p50", "p75", "p95"}


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
            file_name="pump_manual.txt",
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
                file_name="nonexistent_manual.pdf",
            )

    @pytest.mark.asyncio
    async def test_extract_manual_specs_missing(self, tools, data_dir, mock_ctx):
        """Extracting specs from non-existent manual should raise FileNotFoundError."""
        if "extract_manual_specs" not in tools:
            pytest.skip("extract_manual_specs not registered")

        with pytest.raises(FileNotFoundError):
            await tools["extract_manual_specs"](
                ctx=mock_ctx,
                file_name="nonexistent.pdf",
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
            bearing_id="6205",
        )
        # Should find bearing 6205 in the verified JSON catalog
        assert isinstance(result, dict)
        assert result["num_balls"] == 9
        assert result["ball_diameter_mm"] > 0
        assert result["source"]  # mandatory source citation

    @pytest.mark.asyncio
    async def test_search_bearing_catalog_unknown(self, tools, data_dir, mock_ctx):
        """Search for a non-existent bearing returns a TYPED miss result
        (U6 error contract), never a dict with an 'error' key."""
        from predictive_maintenance_mcp.models import BearingCatalogMiss

        if "search_bearing_catalog" not in tools:
            pytest.skip("search_bearing_catalog not registered")

        result = await tools["search_bearing_catalog"](
            ctx=mock_ctx,
            bearing_id="ZZZZZ999",
        )
        assert isinstance(result, BearingCatalogMiss)
        assert result.status == "not_found"
        assert result.suggestion

    @pytest.mark.asyncio
    async def test_search_bearing_catalog_with_prefix(self, tools, data_dir, mock_ctx):
        """Searching with manufacturer prefix (e.g. 'SKF 6205-2RS') should strip it."""
        if "search_bearing_catalog" not in tools:
            pytest.skip("search_bearing_catalog not registered")

        result = await tools["search_bearing_catalog"](
            ctx=mock_ctx,
            bearing_id="SKF 6205-2RS",
        )
        assert isinstance(result, dict)
        assert result["num_balls"] == 9

    @pytest.mark.asyncio
    async def test_calculate_bearing_frequencies_tool(self, tools, data_dir, mock_ctx):
        """Calculate frequencies for CWRU-documented 6205 geometry at 1797 RPM."""
        if "calculate_bearing_characteristic_frequencies" not in tools:
            pytest.skip("calculate_bearing_characteristic_frequencies not registered")

        result = await tools["calculate_bearing_characteristic_frequencies"](
            ctx=mock_ctx,
            num_balls=9,
            ball_diameter_mm=7.94,
            pitch_diameter_mm=39.04,
            contact_angle_deg=0.0,
            rpm=1797.0,
        )
        assert "BPFO" in result
        assert "BPFI" in result
        assert "BSF" in result
        assert "FTF" in result
        # CWRU publishes BPFO = 3.5848 x shaft speed -> 107.36 Hz at 1797 RPM
        assert result["BPFO"] == pytest.approx(107.36, rel=0.005)
        assert result["BPFI"] > result["BPFO"]  # BPFI > BPFO always


# ---------------------------------------------------------------------------
# Diagnose vibration with bearing
# ---------------------------------------------------------------------------

class TestDiagnoseVibrationExtended:
    """Extended tests for diagnose_vibration tool."""

    @pytest.mark.asyncio
    async def test_diagnosis_with_bearing(self, tools, data_dir, mock_ctx):
        """Full diagnosis including bearing fault detection."""
        if "diagnose_vibration" not in tools:
            pytest.skip("diagnose_vibration not registered")

        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("normal.csv", signal_id="diag_brg", sampling_rate=10000)
        try:
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_brg",
                rpm=1800.0,
                bearing_id="6205",
            )
            assert result is not None
            assert result.bearing_faults is not None
            assert result.iso_severity is not None
            assert result.evidence_strength in ("none", "weak", "moderate", "strong")
            assert not hasattr(result, "confidence")
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_diagnosis_missing_signal(self, tools, data_dir, mock_ctx):
        """Diagnosis on non-existent signal_id should raise."""
        if "diagnose_vibration" not in tools:
            pytest.skip("diagnose_vibration not registered")

        with pytest.raises((KeyError, ValueError)):
            await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="nonexistent_signal_id",
                rpm=1800.0,
            )


# ---------------------------------------------------------------------------
# Assess vibration severity extended
# ---------------------------------------------------------------------------

class TestAssessSeverityUnitsExtended:
    """Unit/rate discipline tests for assess_severity (signal route)."""

    @pytest.mark.asyncio
    async def test_group1_boundaries(self, tools, data_dir, mock_ctx):
        """machine_group=1 uses the large-machine boundaries (A/B at 2.3 mm/s)."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="class_test", sampling_rate=10000)
        try:
            result = await tools["assess_severity"](
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
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
        repo = get_repository()
        repo.load_signal("iso_test.csv", signal_id="class_test", sampling_rate=10000)
        try:
            with pytest.raises(TypeError):
                await tools["assess_severity"](
                    ctx=mock_ctx,
                    signal_id="class_test",
                    machine_class="III",
                )
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_signal_without_metadata_sampling_rate(self, tools, data_dir, mock_ctx):
        """Signal with no metadata and no explicit sampling_rate should raise."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

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
                await tools["assess_severity"](
                    ctx=mock_ctx,
                    signal_id="no_sr_test",
                )
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_undeclared_unit_refused_names_load_signal(self, tools, data_dir, mock_ctx):
        """Severity on a stored signal without a declared unit → structured
        error naming load_signal(signal_unit=...) — never a silent 'g'."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 4.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)  # RMS ~4
        pd.DataFrame(sig).to_csv(data_dir / "no_unit_sev.csv", index=False, header=False)
        # No metadata → unit undeclared

        repo = get_repository()
        try:
            repo.load_signal("no_unit_sev.csv", signal_id="no_unit_sev", sampling_rate=fs)
            with pytest.raises(ValueError, match=r"load_signal.*signal_unit="):
                await tools["assess_severity"](
                    ctx=mock_ctx,
                    signal_id="no_unit_sev",
                )
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_metadata_unit_g_assessed_without_confirmation(self, tools, data_dir, mock_ctx):
        """Unit 'g' from _metadata.json → acc→vel integration and a verdict,
        no fake confirmation step."""
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

        repo = get_repository()
        try:
            repo.load_signal("iso_test.csv", signal_id="meta_g")  # metadata: fs + 'g'
            result = await tools["assess_severity"](
                ctx=mock_ctx,
                signal_id="meta_g",
            )
            assert result.status == "assessed"
            assert result.zone in ("A", "B", "C", "D")
            assert result.unit_conversion_performed is True
        finally:
            repo.clear_all()


class TestDiagnoseVibrationRefusedISO:
    """diagnose_vibration degrades: refused ISO block, other blocks run."""

    @pytest.mark.asyncio
    async def test_diagnosis_without_unit_iso_refused(self, tools, data_dir, mock_ctx):
        from predictive_maintenance_mcp.models import ISOSeverityRefusal
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(data_dir / "diag_no_unit.csv", index=False, header=False)
        # No metadata → unit undeclared

        repo = get_repository()
        try:
            repo.load_signal("diag_no_unit.csv", signal_id="diag_no_unit", sampling_rate=fs)
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_no_unit",
                rpm=1800.0,
                bearing_id="6205",
            )
            # ISO block is a schema-level refusal
            assert isinstance(result.iso_severity, ISOSeverityRefusal)
            assert result.iso_severity.status == "refused"
            assert "load_signal" in result.iso_severity.remedy
            # The other blocks still ran
            assert result.fft_summary
            assert result.psd_summary
            assert result.bearing_faults is not None
            assert result.evidence_strength in ("none", "weak", "moderate", "strong")
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_diagnosis_nyquist_too_low_iso_refused(self, tools, data_dir, mock_ctx):
        """fs < 2 kHz (Nyquist below the ISO band, U2 refusal) → ISO block
        refused with reason, while the diagnosis itself succeeds."""
        from predictive_maintenance_mcp.models import ISOSeverityRefusal
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

        fs = 1600
        t = np.linspace(0, 1.0, fs, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 50 * t)
        pd.DataFrame(sig).to_csv(data_dir / "diag_low_fs.csv", index=False, header=False)
        with open(data_dir / "diag_low_fs_metadata.json", "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

        repo = get_repository()
        try:
            repo.load_signal("diag_low_fs.csv", signal_id="diag_low_fs")
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_low_fs",
                rpm=1800.0,
            )
            assert isinstance(result.iso_severity, ISOSeverityRefusal)
            assert "Nyquist" in result.iso_severity.reason
            assert result.fft_summary
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_diagnosis_with_declared_unit_assessed(self, tools, data_dir, mock_ctx):
        """Declared unit → assessed ISO block (status discriminator present)."""
        from predictive_maintenance_mcp.models import VibrationSeverityResult
        from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

        repo = get_repository()
        try:
            repo.load_signal("normal.csv", signal_id="diag_ok")  # metadata: fs + 'g'
            result = await tools["diagnose_vibration"](
                ctx=mock_ctx,
                signal_id="diag_ok",
                rpm=1800.0,
            )
            assert isinstance(result.iso_severity, VibrationSeverityResult)
            assert result.iso_severity.status == "assessed"
            assert result.iso_severity.zone in ("A", "B", "C", "D")
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
