"""What train_anomaly_model reports, what it refuses, and the PCA report's rules.

- ``validation_accuracy`` is the balanced accuracy, the mean of the healthy
  rate and the fault detection rate; ``validation_metrics`` keeps the pooled
  ``overall_accuracy`` and gives every rate its window and signal counts.
- A healthy validation set without fault signals returns its specificity.
- The metadata records ``training_mode`` without calling a one-class model
  supervised, how many windows the saved estimator was fit on, and the
  neighbour count a LocalOutlierFactor actually uses.
- ``segment_duration``, ``pca_variance`` and a window of no samples are
  refused with the project's own message before any file is written.
- The PCA report scores a signal only with the model's own segmentation,
  accepts a fixed vocabulary of true labels, and refuses a model that kept one
  PCA component.

Counts the tests compare against are computed here from the saved model and
the windows cut by hand, not read back from the result under test. Everything
runs in a sandbox on seeded synthetic signals (tones plus Gaussian noise,
10 kHz, unit g), with every by-value import of the model and report
directories patched, so nothing is read from or written to the checkout.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from conftest import patch_data_dir
from predictive_maintenance_mcp import report_generator
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.features import (
    extract_time_domain_features,
)

FS = 10_000.0
WINDOW, HOP = 1000, 500  # the default 0.1 s segments with overlap 0.5 at 10 kHz


def synthetic(seed: int, num_samples: int) -> np.ndarray:
    """Three tones with seeded phases plus seeded Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples) / FS
    phases = rng.uniform(0, 2 * np.pi, 3)
    return (
        0.5 * np.sin(2 * np.pi * 25 * t + phases[0])
        + 0.2 * np.sin(2 * np.pi * 50 * t + phases[1])
        + 0.1 * np.sin(2 * np.pi * 1200 * t + phases[2])
        + 0.3 * rng.standard_normal(num_samples)
    )


def bursts(seed: int, num_samples: int) -> np.ndarray:
    """The synthetic record plus a 107 Hz train of decaying 3 kHz bursts."""
    x = synthetic(seed, num_samples)
    burst_t = np.arange(int(0.005 * FS)) / FS
    burst = 2.0 * np.exp(-burst_t / 0.001) * np.sin(2 * np.pi * 3000 * burst_t)
    for start in (np.arange(0, num_samples / FS, 1 / 107) * FS).astype(int):
        end = min(start + burst.size, x.size)
        x[start:end] += burst[: end - start]
    return x


def windows(values: np.ndarray, window: int = WINDOW, hop: int = HOP) -> np.ndarray:
    """Feature rows of every window, cut by hand."""
    rows = [
        extract_time_domain_features(values[start : start + window])
        for start in range(0, len(values) - window + 1, hop)
    ]
    return pd.DataFrame(rows).values


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Module-wide sandbox for signals, models and reports."""
    root = tmp_path_factory.mktemp("anomaly-training-report")
    dirs = {name: root / name for name in ("signals", "models", "reports")}
    for path in dirs.values():
        path.mkdir()
    repo = get_repository()
    loaded: list[str] = []
    values: dict[str, np.ndarray] = {}
    with pytest.MonkeyPatch.context() as mp:
        patch_data_dir(mp, dirs["signals"])
        mp.setattr(diagnostics_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(diagnosis_pipeline, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "REPORTS_DIR", dirs["reports"])
        mp.setattr(report_generator, "REPORTS_DIR", dirs["reports"])

        def load(signal_id: str, data: np.ndarray) -> str:
            path = dirs["signals"] / f"{signal_id}.csv"
            pd.DataFrame(data).to_csv(path, index=False, header=False)
            repo.load_signal(
                path.name,
                signal_id=signal_id,
                sampling_rate=FS,
                signal_unit="g",
                overwrite=True,
            )
            loaded.append(signal_id)
            values[signal_id] = data
            return signal_id

        yield SimpleNamespace(load=load, values=values, **dirs)
        for signal_id in loaded:
            repo.clear_signal(signal_id)


@pytest.fixture(scope="module")
def sets(sandbox):
    """Healthy training (3 x 19 windows), one fault record (29 windows) and
    two healthy validation records (15 windows each)."""
    return SimpleNamespace(
        healthy=[
            sandbox.load(f"tr_h{seed}", synthetic(seed, 10_000)) for seed in (1, 2, 3)
        ],
        fault=[sandbox.load("tr_f1", bursts(11, 15_000))],
        validation=[
            sandbox.load(f"tr_v{seed}", synthetic(seed, 8_000)) for seed in (21, 22)
        ],
    )


def train(**kwargs):
    return asyncio.run(diagnostics_tools.train_anomaly_model(ctx=AsyncMock(), **kwargs))


def saved(models_dir: Path, name: str) -> SimpleNamespace:
    parts = {}
    for part in ("model", "scaler", "pca"):
        with open(models_dir / f"{name}_{part}.pkl", "rb") as handle:
            parts[part] = pickle.load(handle)  # a model this module just trained
    meta = json.loads((models_dir / f"{name}_metadata.json").read_text("utf-8"))
    return SimpleNamespace(metadata=meta, **parts)


def predictions(model: SimpleNamespace, rows: np.ndarray) -> np.ndarray:
    return model.model.predict(model.pca.transform(model.scaler.transform(rows)))


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


class TestValidationReport:

    def test_fault_tuned_training_reports_the_balanced_accuracy(self, sandbox, sets):
        """Without healthy validation signals the healthy rate comes from the
        last 20 % of the training windows, and a OneClassSVM is refit on the
        first 80 % before it is saved."""
        result = train(
            healthy_signal_ids=sets.healthy,
            fault_signal_ids=sets.fault,
            model_name="tr_balanced",
        )
        model = saved(sandbox.models, "tr_balanced")
        training_rows = np.vstack([windows(sandbox.values[s]) for s in sets.healthy])
        split = int(0.8 * len(training_rows))
        assert (len(training_rows), split) == (57, 45)
        healthy_correct = int(np.sum(predictions(model, training_rows[split:]) == 1))
        fault_rows = windows(sandbox.values[sets.fault[0]])
        fault_detected = int(np.sum(predictions(model, fault_rows) == -1))
        healthy_rate, fault_rate = healthy_correct / 12, fault_detected / 29
        assert (
            healthy_rate != fault_rate
        ), "equal rates cannot tell pooled from balanced"

        assert result.validation_accuracy == (healthy_rate + fault_rate) / 2
        assert result.validation_metrics == {
            "healthy_correct": healthy_correct,
            "healthy_total": 12,
            "healthy_accuracy": healthy_rate,
            "healthy_signals": 1,
            "fault_detected": fault_detected,
            "fault_total": 29,
            "fault_accuracy": fault_rate,
            "fault_signals": 1,
            "overall_accuracy": (healthy_correct + fault_detected) / 41,
            "balanced_accuracy": (healthy_rate + fault_rate) / 2,
        }
        assert model.model.shape_fit_[0] == 45
        assert model.metadata["training_windows"] == 57
        assert model.metadata["estimator_fit_windows"] == 45
        assert result.num_training_samples == 57

    def test_healthy_validation_alone_reports_the_specificity(self, sandbox, sets):
        result = train(
            healthy_signal_ids=sets.healthy,
            healthy_validation_ids=sets.validation,
            model_name="tr_specificity",
        )
        model = saved(sandbox.models, "tr_specificity")
        rows = np.vstack([windows(sandbox.values[s]) for s in sets.validation])
        correct = int(np.sum(predictions(model, rows) == 1))

        assert result.validation_accuracy is None
        assert result.validation_metrics == {
            "healthy_correct": correct,
            "healthy_total": 30,
            "healthy_accuracy": correct / 30,
            "healthy_signals": 2,
        }
        assert result.validation_details == (
            f"Healthy: {correct}/30 correct ({correct / 30 * 100:.1f}%)"
        )
        assert model.model.shape_fit_[0] == 57
        assert model.metadata["estimator_fit_windows"] == 57

    def test_a_local_outlier_factor_keeps_its_fit_on_every_window(self, sandbox, sets):
        """A LocalOutlierFactor is not refit: the metadata says it was fit on
        every training window, the ones behind the healthy rate included."""
        train(
            healthy_signal_ids=sets.healthy,
            fault_signal_ids=sets.fault,
            model_type="LocalOutlierFactor",
            model_name="tr_lof_fit",
        )
        model = saved(sandbox.models, "tr_lof_fit")
        assert model.model.n_samples_fit_ == 57
        assert model.metadata["estimator_fit_windows"] == 57
        assert model.metadata["training_windows"] == 57

    def test_signal_counts_leave_out_signals_without_windows(self, sandbox, sets):
        """A signal shorter than one segment contributes no window, so it is not
        counted among the signals behind a rate."""
        short_healthy = sandbox.load("tr_short_h", synthetic(23, 500))
        short_fault = sandbox.load("tr_short_f", bursts(12, 500))
        result = train(
            healthy_signal_ids=sets.healthy,
            healthy_validation_ids=[short_healthy, sets.validation[0]],
            fault_signal_ids=[short_fault, *sets.fault],
            model_name="tr_signal_counts",
        )
        metrics = result.validation_metrics
        assert (metrics["healthy_total"], metrics["healthy_signals"]) == (15, 1)
        assert (metrics["fault_total"], metrics["fault_signals"]) == (29, 1)

    def test_tail_signal_counts_leave_out_signals_without_windows(self, sandbox, sets):
        """Training windows 92 to 114 give the healthy rate: the last 23 of the
        39 windows of the long record, then the 19 of the last record. A record
        shorter than one segment sits between them and is not counted."""
        long_record = sandbox.load("tr_tail_long", synthetic(31, 20_000))
        short_record = sandbox.load("tr_tail_short", synthetic(32, 500))
        last_record = sandbox.load("tr_tail_last", synthetic(33, 10_000))
        result = train(
            healthy_signal_ids=[*sets.healthy, long_record, short_record, last_record],
            fault_signal_ids=sets.fault,
            model_name="tr_tail_counts",
        )
        metrics = result.validation_metrics
        assert result.num_training_samples == 19 * 3 + 39 + 19
        assert (metrics["healthy_total"], metrics["healthy_signals"]) == (23, 2)

    def test_balanced_accuracy_needs_healthy_windows(self, sandbox, sets):
        """When the healthy validation signals give no window, there is no
        balanced accuracy; the pooled value is the fault detection rate."""
        short_healthy = sandbox.load("tr_short_only", synthetic(24, 500))
        result = train(
            healthy_signal_ids=sets.healthy,
            healthy_validation_ids=[short_healthy],
            fault_signal_ids=sets.fault,
            model_name="tr_no_healthy_windows",
        )
        metrics = result.validation_metrics
        assert result.validation_accuracy is None
        assert metrics["balanced_accuracy"] is None
        assert (metrics["healthy_total"], metrics["healthy_signals"]) == (0, 0)
        assert metrics["fault_total"] == 29
        assert metrics["overall_accuracy"] == metrics["fault_detected"] / 29


class TestMetadata:

    def test_training_mode(self, sandbox, sets):
        train(
            healthy_signal_ids=sets.healthy,
            fault_signal_ids=sets.fault,
            model_name="tr_mode_tuned",
        )
        train(healthy_signal_ids=sets.healthy, model_name="tr_mode_plain")
        tuned = saved(sandbox.models, "tr_mode_tuned").metadata
        plain = saved(sandbox.models, "tr_mode_plain").metadata
        assert tuned["training_mode"] == "semi_supervised"
        assert plain["training_mode"] == "unsupervised"

    def test_lof_records_the_neighbour_count_it_uses(self, sandbox):
        """0.2 s at 10 kHz gives 3 windows: scikit-learn lowers the requested
        10 neighbours to 2, and the metadata records both."""
        tiny = sandbox.load("tr_tiny", synthetic(7, 2_000))
        result = train(
            healthy_signal_ids=[tiny],
            model_type="LocalOutlierFactor",
            model_name="tr_tiny_lof",
        )
        model = saved(sandbox.models, "tr_tiny_lof")
        expected = {
            "n_neighbors": 2,
            "contamination": 0.1,
            "mode": "unsupervised_auto",
            "n_neighbors_requested": 10,
        }
        assert model.model.n_neighbors_ == 2
        assert model.metadata["best_params"] == expected
        assert result.model_params == expected

    def test_lof_records_no_requested_count_when_it_is_used(self, sandbox, sets):
        result = train(
            healthy_signal_ids=sets.healthy,
            model_type="LocalOutlierFactor",
            model_name="tr_lof_kept",
        )
        model = saved(sandbox.models, "tr_lof_kept")
        assert model.model.n_neighbors_ == 10
        assert model.metadata["best_params"] == {
            "n_neighbors": 10,
            "contamination": 0.1,
            "mode": "unsupervised_auto",
        }
        assert result.model_params == model.metadata["best_params"]


# ---------------------------------------------------------------------------
# Refusals at training
# ---------------------------------------------------------------------------


def model_files(models_dir: Path, name: str) -> list[Path]:
    return sorted(models_dir.glob(f"{name}_*"))


class TestTrainingRefusals:

    @pytest.mark.parametrize(
        ("value", "shown"),
        [(0.0, "0.0"), (-0.5, "-0.5"), (float("nan"), "nan"), (float("inf"), "inf")],
    )
    def test_segment_duration_must_be_positive_and_finite(
        self, sandbox, sets, value, shown
    ):
        with pytest.raises(ValueError) as caught:
            train(
                healthy_signal_ids=sets.healthy,
                segment_duration=value,
                model_name="tr_refused_segment",
            )
        assert str(caught.value) == (
            f"segment_duration must be a finite number of seconds above 0, got "
            f"{shown} (from the segment_duration parameter). Use the default 0.1, "
            f"or another number of seconds above 0."
        )
        assert model_files(sandbox.models, "tr_refused_segment") == []

    def test_segment_duration_must_be_a_number(self, sandbox, sets):
        with pytest.raises(ValueError) as caught:
            train(
                healthy_signal_ids=sets.healthy,
                segment_duration="0.1",
                model_name="tr_refused_segment",
            )
        assert str(caught.value) == (
            "segment_duration must be a finite number of seconds above 0, got a "
            "value of type str, which is not a number (from the segment_duration "
            "parameter). Use the default 0.1, or another number of seconds above 0."
        )

    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (1.0, "1.0"),
            (0.0, "0.0"),
            (-0.2, "-0.2"),
            (1.5, "1.5"),
            (float("nan"), "nan"),
        ],
    )
    def test_pca_variance_must_be_above_0_and_below_1(
        self, sandbox, sets, value, shown
    ):
        with pytest.raises(ValueError) as caught:
            train(
                healthy_signal_ids=sets.healthy,
                pca_variance=value,
                model_name="tr_refused_pca",
            )
        assert str(caught.value) == (
            f"pca_variance must be above 0 and below 1, got {shown} (from the "
            f"pca_variance parameter). Use the default 0.95, or another fraction "
            f"above 0 and below 1."
        )
        assert model_files(sandbox.models, "tr_refused_pca") == []

    def test_pca_variance_must_be_a_number(self, sandbox, sets):
        with pytest.raises(ValueError) as caught:
            train(
                healthy_signal_ids=sets.healthy,
                pca_variance="0.95",
                model_name="tr_refused_pca",
            )
        assert str(caught.value) == (
            "pca_variance must be above 0 and below 1, got a value of type str, "
            "which is not a number (from the pca_variance parameter). Use the "
            "default 0.95, or another fraction above 0 and below 1."
        )

    def test_a_segment_of_no_samples_is_refused(self, sandbox, sets):
        """5e-05 s is half a sample at 10 kHz."""
        with pytest.raises(ValueError) as caught:
            train(
                healthy_signal_ids=sets.healthy,
                segment_duration=0.00005,
                model_name="tr_refused_window",
            )
        assert str(caught.value) == (
            "A segment of 5e-05 s holds no samples at 10000 Hz, so no features "
            "can be computed from it. Use a segment_duration of at least one "
            "sample period, 0.0001 s at this rate."
        )
        assert model_files(sandbox.models, "tr_refused_window") == []


# ---------------------------------------------------------------------------
# The PCA report
# ---------------------------------------------------------------------------


def pca_report(model_name: str, **kwargs):
    return asyncio.run(
        report_tools.generate_pca_visualization_report(
            model_name, ctx=AsyncMock(), **kwargs
        )
    )


def report_files(reports_dir: Path) -> set[str]:
    return {path.name for path in reports_dir.iterdir()}


def legacy_variant(models_dir: Path, base: str, name: str, overlap: float) -> str:
    """Copy *base*'s model files to *name* with *overlap* in its metadata, the way a
    model saved before the ratio was checked looks on disk."""
    for suffix in ("_model.pkl", "_scaler.pkl", "_pca.pkl"):
        (models_dir / f"{name}{suffix}").write_bytes(
            (models_dir / f"{base}{suffix}").read_bytes()
        )
    meta = json.loads((models_dir / f"{base}_metadata.json").read_text("utf-8"))
    (models_dir / f"{name}_metadata.json").write_text(
        json.dumps({**meta, "overlap_ratio": overlap}), encoding="utf-8"
    )
    return name


OVERLAP_REFUSAL = (
    "overlap_ratio must be at least 0 and below 1, got {!r} (from the overlap_ratio "
    "parameter). Use the default 0.5, or a value from 0 up to but not including 1."
)
SHORT_SIGNAL_REFUSAL = (
    "Signal '{}' (3000 samples) is shorter than one 0.5s segment at 10000 Hz. "
    "Provide a recording at least one segment long, or retrain the model with a "
    "smaller segment_duration."
)


@pytest.fixture(scope="module")
def half_second(sandbox, sets):
    """A model trained on 0.5 s segments, and a fresh record to score."""
    train(
        healthy_signal_ids=sets.healthy,
        segment_duration=0.5,
        model_name="pca_half_second",
    )
    return SimpleNamespace(
        model="pca_half_second", probe=sandbox.load("pca_probe", bursts(31, 20_000))
    )


class TestPcaReport:

    def test_the_report_scores_with_the_models_segmentation(self, half_second):
        report = pca_report(
            half_second.model,
            test_signal_ids=[half_second.probe],
            segment_duration=0.5,
        )
        prediction = asyncio.run(
            diagnostics_tools.predict_anomalies(
                half_second.probe, half_second.model, ctx=AsyncMock()
            )
        )
        assert report["summary"]["total_segments"] == (20_000 - 5_000) // 2_500 + 1
        assert report["summary"]["total_segments"] == prediction.num_segments
        assert report["summary"]["total_anomalies"] == prediction.anomaly_count

    def test_another_segmentation_is_refused(self, sandbox, half_second):
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report(half_second.model, test_signal_ids=[half_second.probe])
        assert str(caught.value) == (
            "Model 'pca_half_second' was trained on segments of 0.5 s with "
            "overlap_ratio 0.5, and this call passed segment_duration=0.1 and "
            "overlap_ratio=0.5. The report scores a signal only with the "
            "segmentation the model was trained on: pass segment_duration=0.5 "
            "and overlap_ratio=0.5."
        )
        assert report_files(sandbox.reports) == before

    def test_another_overlap_is_refused(self, sandbox, half_second):
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report(
                half_second.model,
                test_signal_ids=[half_second.probe],
                segment_duration=0.5,
                overlap_ratio=0.25,
            )
        assert str(caught.value) == (
            "Model 'pca_half_second' was trained on segments of 0.5 s with "
            "overlap_ratio 0.5, and this call passed segment_duration=0.5 and "
            "overlap_ratio=0.25. The report scores a signal only with the "
            "segmentation the model was trained on: pass segment_duration=0.5 "
            "and overlap_ratio=0.5."
        )
        assert report_files(sandbox.reports) == before

    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (0.0, "0.0"),
            (float("nan"), "nan"),
            ("0.5", "a value of type str, which is not a number"),
        ],
    )
    def test_segment_duration_must_be_positive(
        self, sandbox, half_second, value, shown
    ):
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report(
                half_second.model,
                test_signal_ids=[half_second.probe],
                segment_duration=value,
            )
        assert str(caught.value) == (
            f"segment_duration must be a finite number of seconds above 0, got "
            f"{shown} (from the segment_duration parameter). Use the default 0.1, "
            f"or another number of seconds above 0."
        )
        assert report_files(sandbox.reports) == before

    def test_predict_refuses_a_short_signal(self, sandbox, half_second):
        """3000 samples are less than one 0.5 s segment at 10 kHz."""
        short = sandbox.load("pca_short_predict", bursts(34, 3_000))
        with pytest.raises(ValueError) as caught:
            asyncio.run(
                diagnostics_tools.predict_anomalies(
                    short, half_second.model, ctx=AsyncMock()
                )
            )
        assert str(caught.value) == SHORT_SIGNAL_REFUSAL.format("pca_short_predict")

    def test_the_report_refuses_a_short_signal(self, sandbox, half_second):
        short = sandbox.load("pca_short_report", bursts(34, 3_000))
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report(half_second.model, test_signal_ids=[short], segment_duration=0.5)
        assert str(caught.value) == SHORT_SIGNAL_REFUSAL.format("pca_short_report")
        assert report_files(sandbox.reports) == before

    @pytest.mark.parametrize("ratio", [1.0, 1.5])
    def test_a_legacy_model_scores_with_its_recorded_overlap(
        self, sandbox, half_second, ratio
    ):
        """A model saved before the ratio was checked can record 1 or above.
        Passing exactly that value scores one sample apart, as predict_anomalies
        does: 5200 samples and 5000-sample segments give 201 segments."""
        tag = f"{ratio:g}".replace(".", "_")
        name = legacy_variant(
            sandbox.models, half_second.model, f"pca_legacy_{tag}", ratio
        )
        probe = sandbox.load(f"pca_legacy_probe_{tag}", bursts(35, 5_200))
        report = pca_report(
            name, test_signal_ids=[probe], segment_duration=0.5, overlap_ratio=ratio
        )
        prediction = asyncio.run(
            diagnostics_tools.predict_anomalies(probe, name, ctx=AsyncMock())
        )
        assert report["summary"]["total_segments"] == 5_200 - 5_000 + 1
        assert report["summary"]["total_segments"] == prediction.num_segments
        assert report["summary"]["total_anomalies"] == prediction.anomaly_count

    def test_a_legacy_model_refuses_another_overlap(self, sandbox, half_second):
        name = legacy_variant(
            sandbox.models, half_second.model, "pca_legacy_mismatch", 1.0
        )
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as with_default:
            pca_report(name, test_signal_ids=[half_second.probe], segment_duration=0.5)
        with pytest.raises(ValueError) as with_other:
            pca_report(
                name,
                test_signal_ids=[half_second.probe],
                segment_duration=0.5,
                overlap_ratio=1.5,
            )
        assert str(with_default.value) == (
            "Model 'pca_legacy_mismatch' was trained on segments of 0.5 s with "
            "overlap_ratio 1, and this call passed segment_duration=0.5 and "
            "overlap_ratio=0.5. The report scores a signal only with the "
            "segmentation the model was trained on: pass segment_duration=0.5 "
            "and overlap_ratio=1."
        )
        assert str(with_other.value) == OVERLAP_REFUSAL.format(1.5)
        assert report_files(sandbox.reports) == before

    def test_an_overlap_of_one_is_refused_for_a_current_model(
        self, sandbox, half_second
    ):
        with pytest.raises(ValueError) as caught:
            pca_report(
                half_second.model,
                test_signal_ids=[half_second.probe],
                segment_duration=0.5,
                overlap_ratio=1.0,
            )
        assert str(caught.value) == OVERLAP_REFUSAL.format(1.0)

    def test_a_one_component_model_is_refused(self, sandbox, sets):
        train(
            healthy_signal_ids=sets.healthy,
            pca_variance=0.3,
            model_name="pca_one_component",
        )
        assert saved(sandbox.models, "pca_one_component").pca.n_components_ == 1
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report("pca_one_component", test_signal_ids=[sets.validation[0]])
        assert str(caught.value) == (
            "Model 'pca_one_component' kept one PCA component, so the report has "
            "no second component to plot. Retrain it with a higher pca_variance "
            "to keep at least two components."
        )
        assert report_files(sandbox.reports) == before

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("healthy", 1),
            ("Normal", 1),
            ("BASELINE", 1),
            ("faulty", -1),
            ("Fault", -1),
            ("anomaly", -1),
            ("Anomalous", -1),
        ],
    )
    def test_accepted_true_labels(self, half_second, label, expected):
        report = pca_report(
            half_second.model,
            test_signal_ids=[half_second.probe],
            true_labels={half_second.probe: label},
            segment_duration=0.5,
        )
        total = report["summary"]["total_segments"]
        anomalies = report["summary"]["total_anomalies"]
        per_file = report["summary"]["validation_metrics"]["per_file_accuracy"]
        assert per_file[half_second.probe] == {
            "correct": anomalies if expected == -1 else total - anomalies,
            "total": total,
            "accuracy": (anomalies if expected == -1 else total - anomalies) / total,
            "true_label": label.lower(),
        }

    @pytest.mark.parametrize(
        ("label", "shown"),
        [("good", "'good'"), ("helthy", "'helthy'"), ("", "''"), (3, "3")],
    )
    def test_an_unknown_true_label_is_refused(self, sandbox, half_second, label, shown):
        before = report_files(sandbox.reports)
        with pytest.raises(ValueError) as caught:
            pca_report(
                half_second.model,
                test_signal_ids=[half_second.probe],
                true_labels={half_second.probe: label},
                segment_duration=0.5,
            )
        assert str(caught.value) == (
            f"true_labels gives signal 'pca_probe' the label {shown}, which the "
            f"report does not recognize. Use healthy, normal or baseline for a "
            f"healthy signal, or faulty, fault, anomaly or anomalous for a faulty "
            f"one."
        )
        assert report_files(sandbox.reports) == before
