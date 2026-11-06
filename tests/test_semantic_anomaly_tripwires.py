"""Known defects of the anomaly-model tools, held as strict expected failures.

The models are trained here on seeded synthetic records (tones at 25, 50 and
1200 Hz plus Gaussian noise, 10 kHz, unit g) in a sandbox: data, models,
reports and the ledger all live under a temporary directory, and every
by-value import of the model and report directories is patched, so nothing
is read from or written to the checkout.

Each tripwire asserts a predicate taken from a contract the code already
states (a field description, a docstring, a metadata record, the project's
version-pinning rule) or from the agreement of two consumers of one model.
Whether a model should be used at all, how labels are calibrated and which
validation protocol applies are open decisions and are not asserted here.
A control beside each predicate shows a corrected result satisfies it.
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
import sklearn

from _semantic_guards import (
    Outcome,
    capture,
    is_project_refusal,
    no_foreign_error,
    refused_or_accepted,
    require,
)
from conftest import patch_data_dir
from predictive_maintenance_mcp import report_generator
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.mcp_tools.report_tools import (
    generate_diagnostic_report,
    generate_pca_visualization_report,
)
from predictive_maintenance_mcp.models import AnomalyModelResult
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from test_diagnostic_invariants import known_defect

FS = 10_000.0
DIAGNOSIS_MODEL = "bearing_health_model"


def healthy(seed: int, fs: float = FS, seconds: float = 5.0) -> np.ndarray:
    """A stationary record: three tones with seeded phases plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    phases = rng.uniform(0, 2 * np.pi, 3)
    return (
        0.5 * np.sin(2 * np.pi * 25 * t + phases[0])
        + 0.2 * np.sin(2 * np.pi * 50 * t + phases[1])
        + 0.1 * np.sin(2 * np.pi * 1200 * t + phases[2])
        + 0.3 * rng.standard_normal(t.size)
    )


def faulty(seed: int, fs: float = FS, seconds: float = 5.0) -> np.ndarray:
    """The healthy record plus a 107 Hz train of decaying 3 kHz bursts."""
    x = healthy(seed, fs, seconds)
    burst_t = np.arange(int(0.005 * fs)) / fs
    burst = 2.0 * np.exp(-burst_t / 0.001) * np.sin(2 * np.pi * 3000 * burst_t)
    for start in (np.arange(0, seconds, 1 / 107) * fs).astype(int):
        end = min(start + burst.size, x.size)
        x[start:end] += burst[: end - start]
    return x


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Module-wide sandbox for signals, models, reports and the ledger."""
    root = tmp_path_factory.mktemp("semantic-anomaly")
    dirs = {name: root / name for name in ("signals", "models", "reports", "ledger")}
    for path in dirs.values():
        path.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        patch_data_dir(mp, dirs["signals"])
        mp.setenv("PMM_LEDGER_DIR", str(dirs["ledger"]))
        mp.setattr(diagnostics_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(diagnosis_pipeline, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "REPORTS_DIR", dirs["reports"])
        mp.setattr(report_generator, "REPORTS_DIR", dirs["reports"])
        loaded: list[str] = []

        def load(signal_id: str, values, fs: float = FS, unit: str = "g") -> str:
            path = dirs["signals"] / f"{signal_id}.csv"
            pd.DataFrame(np.asarray(values, dtype=float)).to_csv(
                path, index=False, header=False
            )
            loaded.append(signal_id)
            asyncio.run(
                load_signal(
                    AsyncMock(),
                    str(path),
                    signal_id=signal_id,
                    sampling_rate=fs,
                    signal_unit=unit,
                    overwrite=True,
                )
            )
            return signal_id

        yield SimpleNamespace(load=load, **dirs)
        for signal_id in loaded:
            get_repository().clear_signal(signal_id)


def train(**kwargs) -> Outcome:
    return asyncio.run(
        capture(diagnostics_tools.train_anomaly_model(ctx=AsyncMock(), **kwargs))
    )


def predict(signal_id: str, model_name: str) -> Outcome:
    return asyncio.run(
        capture(
            diagnostics_tools.predict_anomalies(signal_id, model_name, ctx=AsyncMock())
        )
    )


def diagnose(signal_id: str) -> Outcome:
    return asyncio.run(
        capture(diagnostics_tools.diagnose_vibration(AsyncMock(), signal_id, rpm=1800))
    )


def metadata(models_dir: Path, name: str) -> dict:
    return json.loads(
        (models_dir / f"{name}_metadata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def trained(sandbox):
    """The healthy training set and the model the diagnosis loads by name."""
    ids = [sandbox.load(f"rc0_h{seed}", healthy(seed)) for seed in (1, 2, 3)]
    outcome = train(healthy_signal_ids=ids, model_name=DIAGNOSIS_MODEL)
    require(
        outcome.error is None, f"training the diagnosis model raised {outcome.error!r}"
    )
    return SimpleNamespace(ids=ids, result=outcome.result)


# ---------------------------------------------------------------------------
# Two consumers of one model, one signal stored at another rate
# ---------------------------------------------------------------------------


def window_count(num_samples: int, window: int, hop: int) -> int:
    return 0 if window > num_samples else (num_samples - window) // hop + 1


def duration_consistent(
    outcome: Outcome, num_samples: int, stored_fs: float, overlap: float
) -> bool:
    """The reported segment duration, at the stored rate, yields the reported
    number of segments (or the call refuses)."""

    def accept(result) -> bool:
        window = int(round(result.segment_duration_s * stored_fs))
        hop = max(1, int(window * (1 - overlap)))
        return window_count(num_samples, window, hop) == result.num_segments

    return refused_or_accepted(outcome, accept)


def consumers_agree(prediction: Outcome, diagnosis: Outcome) -> bool:
    """predict_anomalies and the diagnosis count the same segments for one
    model and one signal, or one of them refuses."""
    if is_project_refusal(prediction.error) or is_project_refusal(diagnosis.error):
        return True
    if prediction.error is not None or diagnosis.error is not None:
        return False
    block = diagnosis.result.anomaly_detection
    return block is None or block.get("num_segments") == prediction.result.num_segments


@pytest.fixture(scope="module")
def resampled(sandbox, trained):
    """A fresh healthy 10 kHz record stored as if sampled at 25.6 kHz."""
    samples = healthy(100)
    return SimpleNamespace(
        signal_id=sandbox.load("rc0_t2", samples, fs=25_600.0),
        num_samples=samples.size,
    )


class TestOneModelTwoConsumers:

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F3-duration",
            "predict_anomalies reports segment_duration_s 0.1 for windows cut "
            "at the training rate, which last 39 ms on a signal stored at "
            "25.6 kHz",
        ),
    )
    def test_reported_segment_duration_matches_the_segments(self, sandbox, resampled):
        outcome = predict(resampled.signal_id, DIAGNOSIS_MODEL)
        no_foreign_error(outcome, "predict_anomalies")
        overlap = metadata(sandbox.models, DIAGNOSIS_MODEL)["overlap_ratio"]
        assert duration_consistent(outcome, resampled.num_samples, 25_600.0, overlap), (
            outcome.error
            or f"segment_duration_s={outcome.result.segment_duration_s}, num_segments={outcome.result.num_segments}"
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F3-consumers",
            "predict_anomalies and the diagnosis segment one signal "
            "differently with the same model (training rate versus stored "
            "rate), so they disagree on the same signal",
        ),
    )
    def test_predict_and_diagnosis_count_the_same_segments(self, resampled):
        prediction, diagnosis = predict(resampled.signal_id, DIAGNOSIS_MODEL), diagnose(
            resampled.signal_id
        )
        no_foreign_error(prediction, "predict_anomalies")
        no_foreign_error(diagnosis, "diagnose_vibration")
        assert consumers_agree(prediction, diagnosis), (
            prediction.error or diagnosis.error,
            getattr(prediction.result, "num_segments", None),
            (getattr(diagnosis.result, "anomaly_detection", None) or {}).get(
                "num_segments"
            ),
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F2e",
            "the diagnosis returns an anomaly verdict from a record shorter "
            "than one training window, which predict_anomalies refuses",
        ),
    )
    def test_diagnosis_gives_no_verdict_where_predict_refuses(self, sandbox, trained):
        short = sandbox.load("rc0_short", healthy(101, seconds=0.08))
        prediction = predict(short, DIAGNOSIS_MODEL)
        require(
            is_project_refusal(prediction.error),
            f"predict_anomalies no longer refuses the short record: {prediction}",
        )
        diagnosis = diagnose(short)
        no_foreign_error(diagnosis, "diagnose_vibration")
        if diagnosis.error is not None:
            return  # a project-authored refusal is an admissible correction
        block = diagnosis.result.anomaly_detection
        assert block is None or "overall_health" not in block, block

    def test_controls_for_the_consumer_predicates(self):
        good = SimpleNamespace(
            segment_duration_s=0.1, num_segments=window_count(50_000, 2560, 1280)
        )
        assert duration_consistent(Outcome(result=good), 50_000, 25_600.0, 0.5)
        assert not duration_consistent(
            Outcome(result=SimpleNamespace(segment_duration_s=0.1, num_segments=99)),
            50_000,
            25_600.0,
            0.5,
        )
        prediction = Outcome(result=SimpleNamespace(num_segments=38))
        assert consumers_agree(
            prediction,
            Outcome(result=SimpleNamespace(anomaly_detection={"num_segments": 38})),
        )
        assert not consumers_agree(
            prediction,
            Outcome(result=SimpleNamespace(anomaly_detection={"num_segments": 99})),
        )


# ---------------------------------------------------------------------------
# A damaged model file
# ---------------------------------------------------------------------------


def damaged_differs_from_absent(damaged: Outcome, absent_result: dict) -> bool:
    """A report built with an unreadable model does not read like the report
    built with no model at all (or the call refuses)."""
    return refused_or_accepted(
        damaged, lambda result: result["statements"] != absent_result["statements"]
    )


class TestDamagedModel:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F2c",
            "a model file that exists but cannot be read is reported as 'no "
            "trained model was found'",
        ),
    )
    def test_a_damaged_model_is_not_reported_as_absent(
        self, sandbox, trained, tmp_path, monkeypatch
    ):
        damaged = tmp_path / "damaged"
        absent = tmp_path / "absent"
        damaged.mkdir()
        absent.mkdir()
        for path in sandbox.models.glob(f"{DIAGNOSIS_MODEL}_*"):
            (damaged / path.name).write_bytes(path.read_bytes())
        model_file = damaged / f"{DIAGNOSIS_MODEL}_model.pkl"
        model_file.write_bytes(model_file.read_bytes()[:64])
        signal_id = sandbox.load("rc0_t3", healthy(102))

        def report_with(models_dir: Path) -> Outcome:
            monkeypatch.setattr(diagnosis_pipeline, "MODELS_DIR", models_dir)
            return asyncio.run(
                capture(
                    generate_diagnostic_report(
                        signal_id, rpm=1800, formats=["html"], ctx=AsyncMock()
                    )
                )
            )

        without_model = report_with(absent)
        require(
            without_model.error is None,
            f"the report without a model raised {without_model.error!r}",
        )
        with_damaged_model = report_with(damaged)
        no_foreign_error(with_damaged_model, "generate_diagnostic_report")
        assert damaged_differs_from_absent(with_damaged_model, without_model.result)

    def test_control_for_the_damaged_model_predicate(self):
        absent = {"statements": ["No anomaly-model verdict is available."]}
        same = Outcome(
            result={"statements": ["No anomaly-model verdict is available."]}
        )
        named = Outcome(
            result={"statements": ["The model file exists but could not be read."]}
        )
        assert not damaged_differs_from_absent(same, absent)
        assert damaged_differs_from_absent(named, absent)


# ---------------------------------------------------------------------------
# What training reports about its own validation and artefacts
# ---------------------------------------------------------------------------


def accuracy_matches_its_name(result, description: str) -> bool:
    """A value described as balanced accuracy is the mean of the two class rates."""
    metrics = result.validation_metrics
    if (
        metrics is None
        or result.validation_accuracy is None
        or "balanced" not in description.lower()
    ):
        return True
    balanced = (metrics["healthy_accuracy"] + metrics["fault_accuracy"]) / 2.0
    return abs(result.validation_accuracy - balanced) < 1e-9


@pytest.fixture(scope="module")
def fault_tuned(sandbox, trained):
    """A one-class model tuned with one fault record."""
    fault = sandbox.load("rc0_f1", faulty(1))
    outcome = train(
        healthy_signal_ids=trained.ids,
        fault_signal_ids=[fault],
        model_name="rc0_fault_tuned",
    )
    require(outcome.error is None, f"fault-tuned training raised {outcome.error!r}")
    return outcome.result


class TestTrainingReport:

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F4-metric",
            "validation_accuracy is described as balanced accuracy but is the "
            "pooled accuracy over unequal healthy and fault window counts",
        ),
    )
    def test_reported_accuracy_is_the_balanced_one_its_description_names(
        self, fault_tuned
    ):
        description = (
            AnomalyModelResult.model_fields["validation_accuracy"].description or ""
        )
        metrics = fault_tuned.validation_metrics or {}
        require(
            metrics.get("healthy_accuracy") != metrics.get("fault_accuracy"),
            "equal class rates cannot tell pooled from balanced accuracy",
        )
        assert accuracy_matches_its_name(fault_tuned, description), (
            fault_tuned.validation_accuracy,
            fault_tuned.validation_metrics,
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F4-mode",
            "a one-class model tuned with fault signals is recorded as "
            "training_mode 'supervised', which its docstring says it is not",
        ),
    )
    def test_metadata_does_not_call_a_one_class_model_supervised(
        self, sandbox, fault_tuned
    ):
        assert (
            metadata(sandbox.models, "rc0_fault_tuned").get("training_mode")
            != "supervised"
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F4-specificity",
            "train_anomaly_model computes the specificity on "
            "healthy_validation_ids but returns no validation value when no "
            "fault signals are given",
        ),
    )
    def test_healthy_validation_returns_a_validation_value(self, sandbox, trained):
        validation = sandbox.load("rc0_hv", healthy(50))
        outcome = train(
            healthy_signal_ids=trained.ids,
            healthy_validation_ids=[validation],
            model_name="rc0_healthy_val",
        )
        require(outcome.error is None, f"training raised {outcome.error!r}")
        assert (
            outcome.result.validation_accuracy is not None
            or outcome.result.validation_metrics is not None
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F8",
            "model metadata records no library or feature versions, so a "
            "verdict cannot be traced to the code that produced it",
        ),
    )
    def test_metadata_records_library_versions(self, sandbox, trained):
        recorded = json.dumps(metadata(sandbox.models, DIAGNOSIS_MODEL))
        assert sklearn.__version__ in recorded, recorded[:400]

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F9",
            "LOF metadata records the requested n_neighbors while the saved "
            "estimator, trained on fewer windows, uses fewer",
        ),
    )
    def test_metadata_describes_the_saved_estimator(self, sandbox):
        tiny = sandbox.load("rc0_tiny", healthy(7, seconds=0.2))
        outcome = train(
            healthy_signal_ids=[tiny],
            model_type="LocalOutlierFactor",
            model_name="rc0_tiny_lof",
        )

        seen = {}

        def accept(result) -> bool:
            seen["recorded"] = metadata(sandbox.models, "rc0_tiny_lof")[
                "best_params"
            ].get("n_neighbors")
            with open(sandbox.models / "rc0_tiny_lof_model.pkl", "rb") as handle:
                seen["effective"] = pickle.load(
                    handle
                ).n_neighbors_  # a model this test just trained
            return seen["recorded"] == seen["effective"]

        assert refused_or_accepted(outcome, accept), outcome.error or seen

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F10-segment",
            "segment_duration=0 reaches a numpy error instead of a refusal "
            "the project wrote",
        ),
    )
    def test_zero_segment_duration_is_refused_by_the_project(self, trained):
        error = train(healthy_signal_ids=trained.ids, segment_duration=0.0).error
        assert is_project_refusal(error), repr(error)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F10-pca",
            "pca_variance=1.0 reaches a scikit-learn parameter error instead "
            "of a refusal the project wrote",
        ),
    )
    def test_full_pca_variance_is_refused_by_the_project(self, trained):
        error = train(healthy_signal_ids=trained.ids, pca_variance=1.0).error
        assert is_project_refusal(error), repr(error)

    def test_controls_for_the_training_predicates(self):
        description = "Overall balanced accuracy on healthy + fault validation data"
        metrics = {"healthy_accuracy": 0.6, "fault_accuracy": 1.0}
        balanced = SimpleNamespace(validation_accuracy=0.8, validation_metrics=metrics)
        pooled = SimpleNamespace(validation_accuracy=0.85, validation_metrics=metrics)
        assert accuracy_matches_its_name(balanced, description)
        assert not accuracy_matches_its_name(pooled, description)
        assert accuracy_matches_its_name(pooled, "Pooled accuracy over all windows")


# ---------------------------------------------------------------------------
# The PCA report as a third consumer
# ---------------------------------------------------------------------------


def pca_counts_agree(report: Outcome, prediction: Outcome) -> bool:
    """The PCA report counts the windows predict_anomalies counts for the same
    model and signal (or one of them refuses)."""
    if is_project_refusal(report.error) or is_project_refusal(prediction.error):
        return True
    if report.error is not None or prediction.error is not None:
        return False
    return report.result["summary"]["total_segments"] == prediction.result.num_segments


class TestPcaReport:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F7-segmentation",
            "the PCA report segments with its own default window instead of "
            "the model's recorded segmentation, so it counts different "
            "windows than predict_anomalies",
        ),
    )
    def test_pca_report_uses_the_models_segmentation(self, sandbox, trained):
        outcome = train(
            healthy_signal_ids=trained.ids,
            segment_duration=0.5,
            model_name="rc0_half_second",
        )
        require(outcome.error is None, f"training raised {outcome.error!r}")
        signal_id = sandbox.load("rc0_t4", healthy(103))
        report = asyncio.run(
            capture(
                generate_pca_visualization_report(
                    "rc0_half_second", test_signal_ids=[signal_id], ctx=AsyncMock()
                )
            )
        )
        prediction = predict(signal_id, "rc0_half_second")
        no_foreign_error(report, "generate_pca_visualization_report")
        no_foreign_error(prediction, "predict_anomalies")
        assert pca_counts_agree(report, prediction), (
            report.error or report.result["summary"]["total_segments"],
            prediction.error or prediction.result.num_segments,
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "B1-F7-one-component",
            "the PCA report fails with an unexpected error on a model whose "
            "PCA kept one component",
        ),
    )
    def test_pca_report_handles_a_one_component_model(self, sandbox, trained):
        outcome = train(
            healthy_signal_ids=trained.ids,
            pca_variance=0.3,
            model_name="rc0_one_component",
        )
        require(outcome.error is None, f"training raised {outcome.error!r}")
        components = metadata(sandbox.models, "rc0_one_component")["num_features_pca"]
        require(components == 1, f"PCA kept {components} components, not one")
        signal_id = sandbox.load("rc0_t5", healthy(104))
        report = asyncio.run(
            capture(
                generate_pca_visualization_report(
                    "rc0_one_component", test_signal_ids=[signal_id], ctx=AsyncMock()
                )
            )
        )
        assert report.error is None or is_project_refusal(report.error), repr(
            report.error
        )

    def test_controls_for_the_pca_predicate(self):
        prediction = Outcome(result=SimpleNamespace(num_segments=19))
        assert pca_counts_agree(
            Outcome(result={"summary": {"total_segments": 19}}), prediction
        )
        assert not pca_counts_agree(
            Outcome(result={"summary": {"total_segments": 99}}), prediction
        )
        assert not pca_counts_agree(
            Outcome(error=IndexError("tuple index out of range")), prediction
        )
