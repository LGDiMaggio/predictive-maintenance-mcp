"""predict_anomalies advances at least one sample between segment starts.

A model's metadata can carry an overlap ratio that leaves less than one sample
between segment starts: a ratio just below 1 on a short segment, or, in a
model saved before the ratio was checked, a ratio of 1 or above.
predict_anomalies then advances one sample at a time, as training does,
instead of failing with a Python error. A ratio that already leaves at least
one sample between segment starts segments the signal exactly as before, and
a segment of no samples is refused with the project's own message.

Everything runs in a sandbox on seeded synthetic signals (tones plus Gaussian
noise, 10 kHz, unit g): signals and models live under a temporary directory,
and every by-value import of the model directory is patched, so nothing is
read from or written to the checkout. Models other than the trained one are
written directly as metadata variants of it, the way a model saved by an
earlier version looks on disk; the trainer itself refuses a ratio of 1 or
above.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from conftest import patch_data_dir
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.mcp_tools import diagnostics_tools, report_tools
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

FS = 10_000.0
SEGMENT_S = 0.1
SEGMENT_SAMPLES = 1000  # int(0.1 s * 10 kHz)
HIGH_OVERLAP = 0.9995  # int(1000 * (1 - 0.9995)) == 0 samples between starts
PROBE_SAMPLES = 2980  # 1980 samples after the first segment
TRAINED_MODEL = "hop_trained_high_overlap"


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


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Module-wide sandbox for signals and models."""
    root = tmp_path_factory.mktemp("anomaly-hop-floor")
    signals, models = root / "signals", root / "models"
    signals.mkdir()
    models.mkdir()
    repo = get_repository()
    loaded: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        patch_data_dir(mp, signals)
        mp.setattr(diagnostics_tools, "MODELS_DIR", models)
        mp.setattr(diagnosis_pipeline, "MODELS_DIR", models)
        mp.setattr(report_tools, "MODELS_DIR", models)

        def load(signal_id: str, values: np.ndarray) -> str:
            path = signals / f"{signal_id}.csv"
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

        yield SimpleNamespace(load=load, models=models)
        for signal_id in loaded:
            repo.clear_signal(signal_id)


@pytest.fixture(scope="module")
def trained(sandbox):
    """A model trained at a ratio of 0.9995 on 0.1 s segments at 10 kHz."""
    ids = [
        sandbox.load(f"hop_train_{seed}", synthetic(seed, 2000)) for seed in (1, 2, 3)
    ]
    asyncio.run(
        diagnostics_tools.train_anomaly_model(
            healthy_signal_ids=ids,
            segment_duration=SEGMENT_S,
            overlap_ratio=HIGH_OVERLAP,
            model_name=TRAINED_MODEL,
            ctx=AsyncMock(),
        )
    )
    return TRAINED_MODEL


@pytest.fixture(scope="module")
def probe(sandbox):
    """The signal every prediction reads: 2980 samples at 10 kHz."""
    return sandbox.load("hop_probe", synthetic(100, PROBE_SAMPLES))


def metadata(models_dir: Path, name: str) -> dict:
    return json.loads(
        (models_dir / f"{name}_metadata.json").read_text(encoding="utf-8")
    )


def variant(models_dir: Path, base: str, name: str, **changes) -> str:
    """Copy *base*'s model files to *name*, writing *changes* into its metadata."""
    for suffix in ("_model.pkl", "_scaler.pkl", "_pca.pkl"):
        shutil.copyfile(models_dir / f"{base}{suffix}", models_dir / f"{name}{suffix}")
    (models_dir / f"{name}_metadata.json").write_text(
        json.dumps({**metadata(models_dir, base), **changes}), encoding="utf-8"
    )
    return name


def predict(signal_id: str, model_name: str):
    return asyncio.run(
        diagnostics_tools.predict_anomalies(signal_id, model_name, ctx=AsyncMock())
    )


def assert_one_sample_hop(result) -> None:
    """Every segment start is one sample after the previous one."""
    assert result.num_segments == PROBE_SAMPLES - SEGMENT_SAMPLES + 1
    assert result.anomaly_ratio == result.anomaly_count / result.num_segments
    assert len(result.worst_segments) == 10
    indices = [segment["segment_index"] for segment in result.worst_segments]
    assert len(set(indices)) == 10
    assert all(0 <= index < result.num_segments for index in indices)
    for segment in result.worst_segments:
        assert segment["start_time_s"] == round(segment["segment_index"] / FS, 4)


class TestHopBelowOneSample:

    def test_trained_ratio_just_below_one_advances_one_sample(
        self, sandbox, trained, probe
    ):
        """0.9995 on 0.1 s segments at 10 kHz leaves int(1000 * 0.0005) = 0
        samples between segment starts. The prediction advances one sample
        at a time: 2980 samples and 1000-sample segments give 1981 segments,
        segment k starting at k / 10000 s."""
        stored = metadata(sandbox.models, trained)
        assert (stored["overlap_ratio"], stored["segment_duration"]) == (
            HIGH_OVERLAP,
            SEGMENT_S,
        )
        assert stored["sampling_rate"] == FS
        assert int(int(SEGMENT_S * FS) * (1 - HIGH_OVERLAP)) == 0

        result = predict(probe, trained)

        assert result.model_name == trained
        assert result.segment_duration_s == SEGMENT_S
        assert_one_sample_hop(result)

    @pytest.mark.parametrize("ratio", [1.0, 1.5])
    def test_saved_ratio_of_one_or_above_advances_one_sample(
        self, sandbox, trained, probe, ratio
    ):
        """A model saved before the ratio was checked can carry 1 or above,
        which leaves zero or a negative number of samples between segment
        starts. The prediction advances one sample at a time."""
        name = variant(
            sandbox.models,
            trained,
            f"hop_saved_{ratio:g}".replace(".", "_"),
            overlap_ratio=ratio,
        )

        result = predict(probe, name)

        assert_one_sample_hop(result)


class TestHopOfAtLeastOneSample:

    @pytest.mark.parametrize(
        ("ratio", "hop", "num_segments"),
        [
            (0.0, 1000, 2),
            (0.5, 500, 4),
            # 1 - 0.9 is slightly below 0.1 in floating point, so the segments
            # advance int(1000 * (1 - 0.9)) = 99 samples: 1980 // 99 + 1 = 21.
            (0.9, 99, 21),
        ],
    )
    def test_segmentation_is_unchanged(
        self, sandbox, trained, probe, ratio, hop, num_segments
    ):
        """A ratio that leaves at least one sample between segment starts
        segments the signal as before: the segments advance
        int(1000 * (1 - ratio)) samples, and each start time is the segment
        index times the duration the metadata states, 0.1 * (1 - ratio) s."""
        assert int(SEGMENT_SAMPLES * (1 - ratio)) == hop
        name = variant(
            sandbox.models,
            trained,
            f"hop_valid_{ratio:g}".replace(".", "_"),
            overlap_ratio=ratio,
        )

        result = predict(probe, name)

        assert result.num_segments == num_segments
        assert result.num_segments == (PROBE_SAMPLES - SEGMENT_SAMPLES) // hop + 1
        assert len(result.worst_segments) == min(10, num_segments)
        for segment in result.worst_segments:
            assert segment["start_time_s"] == round(
                float(segment["segment_index"] * (SEGMENT_S * (1 - ratio))), 4
            )


class TestSegmentOfNoSamples:

    @pytest.mark.parametrize("ratio", [0.5, HIGH_OVERLAP, 1.0])
    def test_zero_length_segment_is_refused(self, sandbox, trained, probe, ratio):
        """A segment duration of 0.00005 s at 10 kHz is int(0.5) = 0 samples.
        The one-sample hop applies only to a segment of at least one sample;
        a segment of no samples is refused, whatever the ratio."""
        assert int(0.00005 * FS) == 0
        name = variant(
            sandbox.models,
            trained,
            f"hop_zero_{ratio:g}".replace(".", "_"),
            segment_duration=0.00005,
            overlap_ratio=ratio,
        )

        with pytest.raises(ValueError) as caught:
            predict(probe, name)

        assert str(caught.value) == (
            "A segment of 5e-05 s holds no samples at 10000 Hz, so no features "
            "can be computed from it. Retrain the model with a segment_duration "
            "of at least one sample period, 0.0001 s at this rate."
        )

    def test_negative_length_segment_is_refused(self, sandbox, trained, probe):
        """A negative segment duration is a segment of no samples too."""
        name = variant(
            sandbox.models,
            trained,
            "hop_negative",
            segment_duration=-0.1,
            overlap_ratio=0.5,
        )

        with pytest.raises(ValueError) as caught:
            predict(probe, name)

        assert str(caught.value) == (
            "A segment of -0.1 s holds no samples at 10000 Hz, so no features "
            "can be computed from it. Retrain the model with a segment_duration "
            "of at least one sample period, 0.0001 s at this rate."
        )
