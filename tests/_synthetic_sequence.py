"""Deterministic synthetic measurement sequence for the asset-ledger acceptance test.

Thirty vibration measurements of one measurement point, written as headerless
single-column CSV files plus a companion ``<stem>_metadata.json`` each, so the
acceptance test can drive the MCP tools end to end (load, declare, assess)
without any hand-made fixture. The waveform recipe derives from
``golden_bearing`` in ``tests/_golden_signals.py``: a 3 kHz carrier
amplitude-modulated at the BPFO of a 6205 at 1800 rpm (plus its 2nd harmonic)
with seeded gaussian noise; ``FS`` and ``BPFO_6205_1800`` are imported from
there, never retyped.

Sequence layout (1-based index, see :data:`EXPECTED_KIND`):

* M01-M10 ``stable``: amplitude 1.0 with a deterministic jitter within +/-2%,
  modulation depth 0 (no bearing evidence).
* M11 ``isolated_spike``: amplitude x1.8, depth 0.
* M12-M20 ``stable``: as M01-M10.
* M21-M30 ``progressive``: amplitude ramps linearly from +10% to +60% and the
  modulation depth from 0.1 to 1.0, so BPFO evidence emerges over time.

Every measurement draws from its own ``np.random.default_rng(seed + index)``,
the CSV floats use a fixed format with LF line endings on every platform and
the companions are dumped with sorted keys, so two generations with the same
seed produce byte-identical files.

File names are opaque handles: nothing in the core infers anything from them.
Every declared value (asset, point, timestamps, rpm omissions) lives in the
companion, exactly as an adapter would write it.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _golden_signals import BPFO_6205_1800, FS

#: Default seed; ``seed + index`` seeds the generator of measurement ``index``.
DEFAULT_SEED = 20260907

#: Number of measurements in the sequence.
SEQUENCE_LENGTH = 30

#: Duration of every measurement (s); 2 s at ``FS`` gives 20000 samples.
DURATION_S = 2.0

#: Number of samples per measurement.
SAMPLES_PER_MEASUREMENT = int(DURATION_S * FS)

#: Carrier frequency of the ``golden_bearing`` recipe (Hz).
CARRIER_HZ = 3000.0

#: Standard deviation of the additive gaussian noise (unit ``g``).
NOISE_STD = 0.05

#: Modulation coefficients of ``golden_bearing`` at full depth: BPFO
#: fundamental and 2nd harmonic. Depth 1.0 reproduces that recipe exactly.
_MODULATION_FUNDAMENTAL = 0.8
_MODULATION_HARMONIC = 0.3

#: Peak deterministic jitter on the amplitude of stable measurements.
STABLE_JITTER = 0.02

#: Amplitude multiplier of the isolated spike (M11).
SPIKE_AMPLITUDE = 1.8

#: Index of the isolated spike and first index of the progressive ramp.
SPIKE_INDEX = 11
PROGRESSIVE_START = 21

#: Amplitude (relative to 1.0) and modulation depth at both ends of the ramp.
_RAMP_AMPLITUDE = (1.10, 1.60)
_RAMP_DEPTH = (0.1, 1.0)

#: Measurement kinds, as used in :data:`EXPECTED_KIND`.
STABLE = "stable"
ISOLATED_SPIKE = "isolated_spike"
PROGRESSIVE = "progressive"

#: Identity declared in every companion.
ASSET_ID = "P-101"
POINT_ID = "motor_de_h"
BEARING_ID = "6205"
NOMINAL_RPM = 1800
SENSOR_ID = "ACC01"
DIRECTION = "horizontal"
SIGNAL_UNIT = "g"
DECLARED_BY = "synthetic-sequence"

#: Sampling rate declared in every companion (Hz), derived from the golden FS.
SAMPLING_RATE = int(FS)

#: Measurements whose companion omits the ``rpm`` key (exactly ten, fixed).
INDICES_WITHOUT_RPM: frozenset[int] = frozenset({3, 6, 9, 13, 16, 19, 23, 26, 29, 30})

#: Acquisition of M01; later measurements follow weekly. The offset is fixed
#: (+01:00, no DST switch) so the timestamps stay exactly seven days apart.
FIRST_ACQUIRED_AT = datetime(2026, 1, 5, 9, 0, 0, tzinfo=timezone(timedelta(hours=1)))
ACQUISITION_INTERVAL = timedelta(weeks=1)


def _check_index(index: int) -> None:
    """Raise ``ValueError`` unless *index* is a valid 1-based measurement index."""
    if not 1 <= index <= SEQUENCE_LENGTH:
        raise ValueError(f"index must be in 1..{SEQUENCE_LENGTH}, got {index}")


def _kind_of(index: int) -> str:
    """Return the kind of measurement *index* according to the sequence layout."""
    if index == SPIKE_INDEX:
        return ISOLATED_SPIKE
    if index >= PROGRESSIVE_START:
        return PROGRESSIVE
    return STABLE


#: Expected kind per 1-based index: ``stable`` | ``isolated_spike`` |
#: ``progressive``. This is the table the acceptance test asserts against.
EXPECTED_KIND: dict[int, str] = {
    index: _kind_of(index) for index in range(1, SEQUENCE_LENGTH + 1)
}


def acquired_at(index: int) -> str:
    """Return the ISO 8601 acquisition timestamp of measurement *index*.

    Args:
        index: 1-based measurement index (M01 is 1).

    Returns:
        Timestamp with explicit UTC offset, e.g. ``2026-01-05T09:00:00+01:00``
        for M01, one week later for each following measurement.

    Raises:
        ValueError: If *index* is outside ``1..SEQUENCE_LENGTH``.
    """
    _check_index(index)
    return (FIRST_ACQUIRED_AT + (index - 1) * ACQUISITION_INTERVAL).isoformat()


def _amplitude_and_depth(index: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return the carrier amplitude and modulation depth of measurement *index*.

    Stable measurements consume one uniform draw from *rng* for their jitter;
    the spike and the ramp are fully determined by the index.
    """
    kind = EXPECTED_KIND[index]
    if kind == ISOLATED_SPIKE:
        return SPIKE_AMPLITUDE, 0.0
    if kind == PROGRESSIVE:
        fraction = (index - PROGRESSIVE_START) / (SEQUENCE_LENGTH - PROGRESSIVE_START)
        amplitude = (
            _RAMP_AMPLITUDE[0] + (_RAMP_AMPLITUDE[1] - _RAMP_AMPLITUDE[0]) * fraction
        )
        depth = _RAMP_DEPTH[0] + (_RAMP_DEPTH[1] - _RAMP_DEPTH[0]) * fraction
        return amplitude, depth
    jitter = float(rng.uniform(-STABLE_JITTER, STABLE_JITTER))
    return 1.0 + jitter, 0.0


def synthesize_measurement(index: int, *, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Build the waveform of measurement *index* (unit ``g``, float64, 1-D).

    Pure function with no I/O, in the spirit of ``golden_signals()``: a 3 kHz
    carrier amplitude-modulated at ``BPFO_6205_1800`` and its 2nd harmonic
    with the depth of the sequence layout, plus gaussian noise.

    Args:
        index: 1-based measurement index (M01 is 1).
        seed: Sequence seed; the measurement uses ``default_rng(seed + index)``.

    Returns:
        Array of ``SAMPLES_PER_MEASUREMENT`` samples at ``FS`` Hz.

    Raises:
        ValueError: If *index* is outside ``1..SEQUENCE_LENGTH``.
    """
    _check_index(index)
    rng = np.random.default_rng(seed + index)
    amplitude, depth = _amplitude_and_depth(index, rng)
    t = np.arange(SAMPLES_PER_MEASUREMENT) / FS
    modulation = 1.0 + depth * (
        _MODULATION_FUNDAMENTAL * np.sin(2 * np.pi * BPFO_6205_1800 * t)
        + _MODULATION_HARMONIC * np.sin(2 * np.pi * 2 * BPFO_6205_1800 * t)
    )
    carrier = np.sin(2 * np.pi * CARRIER_HZ * t)
    return amplitude * carrier * modulation + NOISE_STD * rng.standard_normal(t.size)


def companion_metadata(index: int) -> dict[str, Any]:
    """Return the companion declared next to measurement *index*.

    The ``rpm`` key is omitted for the indices in :data:`INDICES_WITHOUT_RPM`;
    everything else is constant across the sequence except ``acquired_at``.

    Args:
        index: 1-based measurement index (M01 is 1).

    Returns:
        JSON-serialisable dict with ``sampling_rate``, ``signal_unit`` and the
        ``measurement`` object of the companion contract.
    """
    measurement: dict[str, Any] = {
        "asset_id": ASSET_ID,
        "measurement_point_id": POINT_ID,
        "acquired_at": acquired_at(index),
        "sensor_id": SENSOR_ID,
        "direction": DIRECTION,
        "declared_by": DECLARED_BY,
    }
    if index not in INDICES_WITHOUT_RPM:
        measurement["rpm"] = NOMINAL_RPM
    return {
        "sampling_rate": SAMPLING_RATE,
        "signal_unit": SIGNAL_UNIT,
        "measurement": measurement,
    }


def companion_path(csv_path: Path) -> Path:
    """Return the companion ``<stem>_metadata.json`` path next to *csv_path*."""
    return csv_path.with_name(f"{csv_path.stem}_metadata.json")


def build_measurement_sequence(
    out_dir: Path, *, seed: int = DEFAULT_SEED
) -> list[Path]:
    """Write the 30 measurements (CSV + companion each) into *out_dir*.

    Args:
        out_dir: Target directory, created if missing. Existing files with
            the same names are overwritten.
        seed: Sequence seed (see :func:`synthesize_measurement`). Companions
            do not depend on it.

    Returns:
        The CSV paths in order M01..M30 (``seq_m01.csv`` .. ``seq_m30.csv``);
        the companion of each lives at :func:`companion_path`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(1, SEQUENCE_LENGTH + 1):
        csv_path = out_dir / f"seq_m{index:02d}.csv"
        # An explicit LF handle: given a file name, savetxt opens it in text
        # mode and Windows would translate the row separator to CRLF, which
        # changes the bytes (and any content hash) between platforms.
        with csv_path.open("w", encoding="ascii", newline="\n") as fh:
            np.savetxt(fh, synthesize_measurement(index, seed=seed), fmt="%.8f")
        companion_path(csv_path).write_text(
            json.dumps(companion_metadata(index), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        paths.append(csv_path)
    return paths
