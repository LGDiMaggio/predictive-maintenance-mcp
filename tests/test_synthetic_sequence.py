"""Tests for the deterministic synthetic measurement sequence generator.

The generator (``tests/_synthetic_sequence.py``) feeds the asset-ledger
acceptance test; these tests pin the properties that test relies on: byte
identity across regenerations, the file/companion layout, the fixed rpm
omissions, the shape of the sequence (stable / isolated spike / progressive)
as seen through the RMS of the written files, and the weekly timestamps.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from _golden_signals import FS
from _synthetic_sequence import (
    ASSET_ID,
    DECLARED_BY,
    DIRECTION,
    EXPECTED_KIND,
    INDICES_WITHOUT_RPM,
    ISOLATED_SPIKE,
    NOMINAL_RPM,
    POINT_ID,
    PROGRESSIVE,
    SAMPLES_PER_MEASUREMENT,
    SAMPLING_RATE,
    SENSOR_ID,
    SEQUENCE_LENGTH,
    SIGNAL_UNIT,
    STABLE,
    acquired_at,
    build_measurement_sequence,
    companion_path,
)

STABLE_BLOCKS = {"M01-M10": range(1, 11), "M12-M20": range(12, 21)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digests(directory: Path) -> dict[str, str]:
    """Map every file name in *directory* to the SHA-256 of its bytes."""
    return {p.name: _sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}


def _companion(csv_path: Path) -> dict:
    return json.loads(companion_path(csv_path).read_text(encoding="utf-8"))


def _rms(csv_path: Path) -> float:
    return float(np.sqrt(np.mean(np.loadtxt(csv_path) ** 2)))


@pytest.fixture(scope="module")
def sequence(tmp_path_factory) -> list[Path]:
    """One generation with the default seed, shared by the module."""
    return build_measurement_sequence(tmp_path_factory.mktemp("sequence"))


@pytest.fixture(scope="module")
def rms_by_index(sequence) -> dict[int, float]:
    """RMS of every written CSV, keyed by 1-based index."""
    return {index: _rms(path) for index, path in enumerate(sequence, start=1)}


def test_regeneration_is_byte_identical(sequence, tmp_path):
    again = build_measurement_sequence(tmp_path / "again")

    first = _digests(sequence[0].parent)
    second = _digests(again[0].parent)

    assert [p.name for p in again] == [p.name for p in sequence]
    assert len(first) == 2 * SEQUENCE_LENGTH
    assert first == second


def test_seed_only_affects_waveforms(sequence, tmp_path):
    other = build_measurement_sequence(tmp_path / "other", seed=1)

    for default_csv, other_csv in zip(sequence, other):
        assert _sha256(default_csv) != _sha256(other_csv)
        assert _sha256(companion_path(default_csv)) == _sha256(
            companion_path(other_csv)
        )


def test_thirty_csvs_each_with_companion(sequence):
    assert len(sequence) == SEQUENCE_LENGTH
    assert [p.name for p in sequence] == [
        f"seq_m{index:02d}.csv" for index in range(1, SEQUENCE_LENGTH + 1)
    ]
    for index, csv_path in enumerate(sequence, start=1):
        assert csv_path.is_file()
        assert companion_path(csv_path).name == f"seq_m{index:02d}_metadata.json"
        assert companion_path(csv_path).is_file()


def test_companion_declares_the_contract(sequence):
    assert SAMPLING_RATE == FS  # derived from the golden FS, not truncated
    for index, csv_path in enumerate(sequence, start=1):
        companion = _companion(csv_path)
        assert companion["sampling_rate"] == SAMPLING_RATE
        assert companion["signal_unit"] == SIGNAL_UNIT
        measurement = companion["measurement"]
        assert measurement["asset_id"] == ASSET_ID
        assert measurement["measurement_point_id"] == POINT_ID
        assert measurement["acquired_at"] == acquired_at(index)
        assert measurement["sensor_id"] == SENSOR_ID
        assert measurement["direction"] == DIRECTION
        assert measurement["declared_by"] == DECLARED_BY
        assert set(companion) == {"sampling_rate", "signal_unit", "measurement"}
        assert set(measurement) <= {
            "asset_id",
            "measurement_point_id",
            "acquired_at",
            "rpm",
            "sensor_id",
            "direction",
            "declared_by",
        }


def test_exactly_the_declared_indices_lack_rpm(sequence):
    without_rpm = {
        index
        for index, csv_path in enumerate(sequence, start=1)
        if "rpm" not in _companion(csv_path)["measurement"]
    }

    assert len(INDICES_WITHOUT_RPM) == 10
    assert without_rpm == set(INDICES_WITHOUT_RPM)
    for index, csv_path in enumerate(sequence, start=1):
        if index not in INDICES_WITHOUT_RPM:
            assert _companion(csv_path)["measurement"]["rpm"] == NOMINAL_RPM


def test_expected_kind_table():
    assert sorted(EXPECTED_KIND) == list(range(1, SEQUENCE_LENGTH + 1))
    assert [i for i, kind in EXPECTED_KIND.items() if kind == ISOLATED_SPIKE] == [11]
    assert [i for i, kind in EXPECTED_KIND.items() if kind == PROGRESSIVE] == list(
        range(21, 31)
    )
    assert [i for i, kind in EXPECTED_KIND.items() if kind == STABLE] == [
        *range(1, 11),
        *range(12, 21),
    ]


def test_isolated_spike_stands_out(rms_by_index):
    stable_mean = np.mean([rms_by_index[i] for i in range(1, 11)])

    assert rms_by_index[11] >= 1.5 * stable_mean


def test_progressive_ramp_rises(rms_by_index):
    ramp = [rms_by_index[i] for i in range(21, 31)]

    assert all(later > earlier for earlier, later in zip(ramp, ramp[1:]))
    assert rms_by_index[30] > 1.4 * rms_by_index[20]


@pytest.mark.parametrize("block", sorted(STABLE_BLOCKS))
def test_stable_measurements_within_five_percent(rms_by_index, block):
    values = [rms_by_index[i] for i in STABLE_BLOCKS[block]]

    assert max(values) <= 1.05 * min(values)


def test_timestamps_strictly_increasing_and_weekly(sequence):
    stamps = [
        datetime.fromisoformat(_companion(csv_path)["measurement"]["acquired_at"])
        for csv_path in sequence
    ]

    assert stamps[0] == datetime.fromisoformat("2026-01-05T09:00:00+01:00")
    assert all(stamp.utcoffset() == timedelta(hours=1) for stamp in stamps)
    assert all(
        later - earlier == timedelta(weeks=1)
        for earlier, later in zip(stamps, stamps[1:])
    )


def test_acquired_at_rejects_indices_outside_the_sequence():
    with pytest.raises(ValueError):
        acquired_at(0)
    with pytest.raises(ValueError):
        acquired_at(SEQUENCE_LENGTH + 1)


def test_each_csv_loads_as_one_dimensional_float_array(sequence):
    assert SAMPLES_PER_MEASUREMENT == 20000
    for csv_path in sequence:
        loaded = np.loadtxt(csv_path)
        assert loaded.ndim == 1
        assert loaded.shape == (SAMPLES_PER_MEASUREMENT,)
        assert loaded.dtype == np.float64
        assert np.all(np.isfinite(loaded))
        assert b"\r" not in csv_path.read_bytes()
