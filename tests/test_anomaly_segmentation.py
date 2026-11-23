"""One segmentation rule and one label rule for every anomaly-model consumer.

Training, ``predict_anomalies``, the diagnosis pipeline and the PCA report cut
a signal into windows with ``plan_segments`` and the two label consumers map an
anomaly ratio with ``health_label``. These tests pin the rule itself (window
length, the one-sample minimum between segment starts, the refusal of a window
of no samples, the label cut-offs) and that the four consumers count the same
windows, and the same anomalous windows, for one model, one signal and one
sampling rate. Training is also pinned as deterministic: two trainings on the
same inputs write byte-identical artefacts and give identical predictions.

Everything runs in a sandbox on seeded synthetic signals (tones plus Gaussian
noise, 10 kHz, unit g): signals, models and reports live under a temporary
directory, and every by-value import of the model and report directories is
patched, so nothing is read from or written to the checkout.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from conftest import patch_data_dir
from predictive_maintenance_mcp import report_generator
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.decision_support.anomaly_labels import health_label
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.features import (
    MODEL_SEGMENT_REMEDY,
    extract_time_domain_features,
    plan_segments,
    segment_features,
)
from predictive_maintenance_mcp.signal_validation import InvalidInputError

FS = 10_000.0


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


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


class TestPlanSegments:

    def test_default_segmentation(self):
        plan = plan_segments(FS, 0.1, 0.5)
        assert (plan.length, plan.hop, plan.hop_s) == (1000, 500, 0.05)

    @pytest.mark.parametrize(
        ("ratio", "hop"),
        [
            (0.0, 1000),
            (0.5, 500),
            # 1 - 0.9 is slightly below 0.1 in floating point: 99 samples.
            (0.9, 99),
        ],
    )
    def test_a_hop_of_at_least_one_sample_is_kept(self, ratio, hop):
        plan = plan_segments(FS, 0.1, ratio)
        assert (plan.length, plan.hop) == (1000, hop)
        assert plan.hop_s == 0.1 * (1 - ratio)

    @pytest.mark.parametrize("ratio", [0.9995, 1.0, 1.5])
    def test_less_than_one_sample_between_starts_advances_one_sample(self, ratio):
        """A ratio just below 1, or 1 and above in a model saved before the
        ratio was checked, advances one sample, reported at that pace."""
        plan = plan_segments(FS, 0.1, ratio)
        assert (plan.length, plan.hop, plan.hop_s) == (1000, 1, 1 / FS)

    def test_a_segment_of_no_samples_is_refused(self):
        with pytest.raises(InvalidInputError) as caught:
            plan_segments(FS, 0.00005, 0.5)
        assert str(caught.value) == (
            "A segment of 5e-05 s holds no samples at 10000 Hz, so no features "
            "can be computed from it. Use a segment_duration of at least one "
            "sample period, 0.0001 s at this rate."
        )

    @pytest.mark.parametrize(
        ("duration", "shown"), [(0.0, "0"), (-0.1, "-0.1"), (0.00005, "5e-05")]
    )
    def test_a_model_segment_of_no_samples_is_refused(self, duration, shown):
        with pytest.raises(InvalidInputError) as caught:
            plan_segments(FS, duration, 0.5, remedy=MODEL_SEGMENT_REMEDY)
        assert str(caught.value) == (
            f"A segment of {shown} s holds no samples at 10000 Hz, so no features "
            "can be computed from it. Retrain the model with a segment_duration "
            "of at least one sample period, 0.0001 s at this rate."
        )

    def test_segment_features_cuts_the_planned_windows(self):
        signal = synthetic(5, 2980)
        plan = plan_segments(FS, 0.1, 0.5)
        features = segment_features(signal, plan)
        assert len(features) == (2980 - 1000) // 500 + 1
        assert features[0] == extract_time_domain_features(signal[:1000])
        assert features[-1] == extract_time_domain_features(signal[1500:2500])


class TestHealthLabel:

    @pytest.mark.parametrize(
        ("ratio", "label"),
        [
            (0.0, "Healthy"),
            (0.0999, "Healthy"),
            (0.1, "Suspicious"),
            (0.2999, "Suspicious"),
            (0.3, "Faulty"),
            (1.0, "Faulty"),
        ],
    )
    def test_cut_offs(self, ratio, label):
        assert health_label(ratio) == label


# ---------------------------------------------------------------------------
# The consumers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Module-wide sandbox for signals, models and reports."""
    root = tmp_path_factory.mktemp("anomaly-segmentation")
    dirs = {name: root / name for name in ("signals", "models", "reports")}
    for path in dirs.values():
        path.mkdir()
    repo = get_repository()
    loaded: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        patch_data_dir(mp, dirs["signals"])
        mp.setattr(diagnostics_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(diagnosis_pipeline, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "MODELS_DIR", dirs["models"])
        mp.setattr(report_tools, "REPORTS_DIR", dirs["reports"])
        mp.setattr(report_generator, "REPORTS_DIR", dirs["reports"])

        def load(signal_id: str, values: np.ndarray) -> str:
            path = dirs["signals"] / f"{signal_id}.csv"
            pd.DataFrame(values).to_csv(path, index=False, header=False)
            repo.load_signal(
                path.name,
                signal_id=signal_id,
                sampling_rate=FS,
                signal_unit="g",
                overwrite=True,
            )
            loaded.append(signal_id)
            return signal_id

        yield SimpleNamespace(load=load, values={}, **dirs)
        for signal_id in loaded:
            repo.clear_signal(signal_id)


def train(**kwargs):
    return asyncio.run(diagnostics_tools.train_anomaly_model(ctx=AsyncMock(), **kwargs))


def predict(signal_id: str, model_name: str):
    return asyncio.run(
        diagnostics_tools.predict_anomalies(signal_id, model_name, ctx=AsyncMock())
    )


def diagnosis_block(values: np.ndarray, model_name: str) -> dict:
    result = diagnosis_pipeline.diagnose_vibration(
        values, FS, rpm=1800, signal_unit="g", anomaly_model_name=model_name
    )
    return result["anomaly_detection"]


def pca_report(model_name: str, signal_id: str, duration: float, ratio: float):
    return asyncio.run(
        report_tools.generate_pca_visualization_report(
            model_name,
            test_signal_ids=[signal_id],
            segment_duration=duration,
            overlap_ratio=ratio,
            ctx=AsyncMock(),
        )
    )


class TestConsumersAgree:
    """Training, predict_anomalies, the diagnosis and the PCA report count the
    same windows, and the same anomalous windows, for one model, one signal
    and one rate."""

    @pytest.mark.parametrize(
        ("duration", "ratio", "num_samples", "num_segments"),
        [
            (0.1, 0.5, 30_000, 59),
            (0.25, 0.3, 30_000, 16),
            # Less than one sample between segment starts: one-sample pace.
            (0.1, 0.9995, 2980, 1981),
        ],
    )
    def test_window_and_anomaly_counts_agree(
        self, sandbox, duration, ratio, num_samples, num_segments
    ):
        tag = f"{duration:g}_{ratio:g}".replace(".", "_")
        model = f"agree_{tag}"
        training = [
            sandbox.load(f"agree_train_{tag}_{seed}", synthetic(seed, num_samples))
            for seed in (1, 2, 3)
        ]
        train(
            healthy_signal_ids=training,
            segment_duration=duration,
            overlap_ratio=ratio,
            model_name=model,
        )
        probe_values = bursts(40, num_samples)
        probe = sandbox.load(f"agree_probe_{tag}", probe_values)
        own = train(
            healthy_signal_ids=[probe],
            segment_duration=duration,
            overlap_ratio=ratio,
            model_name=f"agree_self_{tag}",
        )

        prediction = predict(probe, model)
        block = diagnosis_block(probe_values, model)
        report = pca_report(model, probe, duration, ratio)

        assert own.num_training_samples == num_segments
        assert prediction.num_segments == num_segments
        assert block is not None
        assert block["num_segments"] == num_segments
        assert report["summary"]["total_segments"] == num_segments
        assert block["anomaly_count"] == prediction.anomaly_count
        assert report["summary"]["total_anomalies"] == prediction.anomaly_count
        assert block["overall_health"] == prediction.overall_health


class TestDiagnosisHopFloor:

    def test_the_diagnosis_scores_a_model_with_a_hop_below_one_sample(self, sandbox):
        """0.9995 on 0.1 s segments at 10 kHz leaves int(1000 * 0.0005) = 0
        samples between segment starts. The diagnosis advances one sample, as
        predict_anomalies does, instead of dropping its anomaly block."""
        training = [
            sandbox.load(f"floor_train_{seed}", synthetic(seed, 2000))
            for seed in (1, 2, 3)
        ]
        train(
            healthy_signal_ids=training,
            segment_duration=0.1,
            overlap_ratio=0.9995,
            model_name="floor_model",
        )
        probe_values = synthetic(100, 2980)
        probe = sandbox.load("floor_probe", probe_values)

        block = diagnosis_block(probe_values, "floor_model")

        assert block is not None
        assert block["num_segments"] == 2980 - 1000 + 1
        assert block["num_segments"] == predict(probe, "floor_model").num_segments


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def artefacts(models_dir: Path, name: str) -> dict[str, bytes]:
    return {
        suffix: (models_dir / f"{name}{suffix}").read_bytes()
        for suffix in ("_model.pkl", "_scaler.pkl", "_pca.pkl", "_metadata.json")
    }


class TestTrainingIsDeterministic:

    @pytest.mark.parametrize(
        "options",
        [
            {"model_type": "OneClassSVM"},
            {"model_type": "LocalOutlierFactor"},
            {"model_type": "OneClassSVM", "with_faults": True},
        ],
        ids=["ocsvm", "lof", "ocsvm-fault-tuned"],
    )
    def test_two_trainings_write_identical_artefacts(self, sandbox, options):
        tag = "_".join(str(v) for v in options.values())
        training = [
            sandbox.load(f"det_train_{tag}_{seed}", synthetic(seed, 10_000))
            for seed in (1, 2, 3)
        ]
        extra = {}
        if options.get("with_faults"):
            extra["fault_signal_ids"] = [
                sandbox.load(f"det_fault_{tag}", bursts(9, 10_000))
            ]
        probe = sandbox.load(f"det_probe_{tag}", bursts(41, 10_000))
        name = f"det_{tag}"
        kwargs = dict(
            healthy_signal_ids=training,
            model_type=options["model_type"],
            model_name=name,
            **extra,
        )

        first_result = train(**kwargs)
        first_files = artefacts(sandbox.models, name)
        first_prediction = predict(probe, name)
        second_result = train(**kwargs)
        second_files = artefacts(sandbox.models, name)
        second_prediction = predict(probe, name)

        assert second_files == first_files
        assert second_result.model_dump(exclude={"model_path"}) == (
            first_result.model_dump(exclude={"model_path"})
        )
        assert second_prediction.model_dump() == first_prediction.model_dump()
