"""Deterministic fixture signals for the U9 golden-merge characterization.

Shared between the one-off capture script (which ran the PRE-MERGE tools and
wrote tests/fixtures/golden_merges.json) and tests/test_golden_merges.py
(which runs the POST-MERGE tools on the SAME signals and asserts
equivalence). Every array is fully deterministic (fixed seeds), so the
snapshot is regenerable bit-for-bit.

Capture recipe (documented in tests/test_golden_merges.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: Sampling rate for every golden fixture signal (Hz).
FS = 10000.0

#: BPFO of the catalog 6205 (CWRU geometry) at 1800 RPM: 3.5848 x 30 Hz.
BPFO_6205_1800 = 107.54


def golden_signals() -> dict[str, np.ndarray]:
    """Build the deterministic fixture signals, keyed by signal_id."""
    signals: dict[str, np.ndarray] = {}

    # ISO severity fixture: pure 50 Hz sine, 0.5 amplitude (unit 'g'), 2 s.
    t2 = np.arange(int(2 * FS)) / FS
    signals["golden_iso"] = 0.5 * np.sin(2 * np.pi * 50.0 * t2)

    # Velocity fixture: 3.0 mm/s RMS sine at 50 Hz (declared 'mm/s'), 2 s.
    signals["golden_vel3"] = 3.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t2)

    # Bearing-fault-like fixture: 3 kHz carrier amplitude-modulated at the
    # BPFO of a 6205 at 1800 RPM (+ 2nd harmonic) + seeded noise, 1 s.
    t1 = np.arange(int(FS)) / FS
    rng = np.random.default_rng(123)
    modulation = (
        1.0
        + 0.8 * np.sin(2 * np.pi * BPFO_6205_1800 * t1)
        + 0.3 * np.sin(2 * np.pi * 2 * BPFO_6205_1800 * t1)
    )
    signals["golden_bearing"] = (
        np.sin(2 * np.pi * 3000.0 * t1) * modulation
        + 0.05 * rng.standard_normal(t1.size)
    )

    # Pure seeded noise (no fault), 1 s.
    rng2 = np.random.default_rng(456)
    signals["golden_noise"] = 0.1 * rng2.standard_normal(int(FS))

    # Degrading recording: amplitude ramp x seeded noise, 2 s.
    rng3 = np.random.default_rng(42)
    n = int(2 * FS)
    ramp = 0.05 + 0.5 * (np.arange(n) / n)
    signals["golden_trend"] = ramp * rng3.standard_normal(n)

    return signals


#: Declared unit per fixture (None = load without a unit declaration).
GOLDEN_UNITS: dict[str, str | None] = {
    "golden_iso": "g",
    "golden_vel3": "mm/s",
    "golden_bearing": "g",
    "golden_noise": "g",
    "golden_trend": "g",
}


def load_golden_signals(repo, tmp_dir: Path) -> list[str]:
    """Write the fixture CSVs into *tmp_dir* and load them into *repo*.

    Uses absolute paths (loaded directly, independent of DATA_DIR) with
    explicit signal_id / sampling_rate / signal_unit so no companion
    metadata files are needed. Returns the loaded signal_ids.
    """
    ids = []
    for sid, arr in golden_signals().items():
        path = Path(tmp_dir) / f"{sid}.csv"
        pd.DataFrame(arr).to_csv(path, index=False, header=False)
        repo.load_signal(
            str(path),
            signal_id=sid,
            sampling_rate=FS,
            signal_unit=GOLDEN_UNITS[sid],
            overwrite=True,
        )
        ids.append(sid)
    return ids
