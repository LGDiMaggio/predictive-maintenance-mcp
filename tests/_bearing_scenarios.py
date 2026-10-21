"""Seeded bearing scenarios whose defects are known by construction.

Used by the bearing acceptance suite, ``tests/test_bearing_specificity.py``.
What a test expects from a scenario follows from how the signal was built
(the defects it contains, ``DEFECTS_PRESENT``), never from what the engine
returns. Every generator draws from a seeded ``numpy.random.Generator``, and
the seed ranges below are used by no other test.

Kinematics: the closed-form rolling-element defect frequencies of Randall &
Antoni (2011), "Rolling element bearing diagnostics - A tutorial", MSSP 25(2),
485-520, for the 6205 geometry the CWRU Bearing Data Center documents (Z = 9,
Bd = 7.94 mm, Pd = 39.04 mm, contact angle 0) at its nominal 1797 rpm.

Fault-free families. None contains a bearing defect:

- ``white``: Gaussian white noise.
- ``shaft_lines``: white noise plus 1x to 5x shaft harmonics and 50 and
  100 Hz lines, all below the 500 Hz lower edge of the default envelope band.
- ``shaft_modulated``: white noise amplitude-modulated at 1x and 2x shaft
  speed, as run-out or unbalance modulates the floor. Its envelope carries
  real lines at shaft orders, none of them a bearing defect.
- ``resonant_coloured``: white noise, a random walk (power falling as 1/f^2)
  with its slow drift removed, and noise through a 3 kHz resonance.
- ``random_impacts``: Poisson (aperiodic) impacts exciting a 3 kHz resonance:
  impulsive, but periodic at no frequency.
- ``gear_tones``: two steady in-band tones at 41x and 82x shaft speed. Their
  envelope beat lies at the 41st shaft order, far above every fault window.

Defects, after the single-point defect model of McFadden & Smith (1984),
"Model for the vibration produced by a single point defect in a rolling
element bearing", JSV 96(1), 69-82: an impulse train at the defect rate with
1 % random slip between impacts, exciting a 3 kHz resonance, scaled to unit
RMS, plus white noise at the stated SNR (defect RMS over noise RMS, in dB) and
1x and 2x shaft lines.

- ``outer``: impacts at BPFO with constant amplitude (the defect is fixed).
- ``inner``: impacts at BPFI, amplitude-modulated at the shaft rate (the
  defect turns with the shaft through the load zone).
- ``rolling_element``: impacts at 2 x BSF, amplitude-modulated at the cage
  rate. A rolling-element defect strikes the inner and the outer race once
  per spin revolution, so its envelope line lies at twice the ball spin
  frequency.
- ``cage``: impacts at FTF with constant amplitude, once per revolution of a
  cage whose damage strikes the rolling elements.
- ``combined``: an outer-race and an inner-race scenario added together, each
  at the stated SNR.

Shared line (``shared_line``): a 3 kHz carrier amplitude-modulated at 1, 2 and
3 x BPFO and at 1.5 x BPFO, plus white noise. The 1.5 x BPFO line is a steady
non-bearing line inside the BPFI search window, as a mains component can be.
The modulation stays positive, so the envelope holds exactly these four
lines, and the 3 x BPFO line is also exactly twice the 1.5 x BPFO line: one
observed line that the outer-race family (third harmonic) and the inner-race
family (second harmonic of the line in its window) both claim, whatever the
width of the harmonic windows. Nothing else in the envelope belongs to BPFI
(no third or fourth harmonic, no shaft-rate sideband), and nothing belongs to
BSF or FTF.
"""

import json
import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.signal import iirpeak, lfilter

REPO_ROOT = Path(__file__).resolve().parent.parent

#: CWRU drive-end bearing 6205: Z, Bd (mm), Pd (mm), and the nominal speed.
Z_6205, BD_6205, PD_6205 = 9, 7.94, 39.04
RPM_6205 = 1797.0

#: Seed ranges no other test uses: fault-free families, then defects.
FAULT_FREE_SEED = 71_000
DEFECT_SEED = 72_000

#: Structural resonance every impulsive scenario excites (Hz).
RESONANCE_HZ = 3000.0

#: The defects each defect scenario contains, by bearing acronym.
DEFECTS_PRESENT: dict[str, frozenset[str]] = {
    "outer": frozenset({"BPFO"}),
    "inner": frozenset({"BPFI"}),
    "rolling_element": frozenset({"BSF"}),
    "cage": frozenset({"FTF"}),
    "combined": frozenset({"BPFO", "BPFI"}),
}


def bearing_kinematics(
    z: int, bd_mm: float, pd_mm: float, alpha_deg: float, rpm: float
) -> dict[str, float]:
    """Rolling-element defect frequencies (Hz), Randall & Antoni (2011).

    fr = rpm/60, c = (Bd/Pd) cos(alpha):
    BPFO = Z/2 fr (1 - c), BPFI = Z/2 fr (1 + c),
    BSF = Pd/(2 Bd) fr (1 - c^2), FTF = fr/2 (1 - c).
    """
    fr = rpm / 60.0
    c = bd_mm / pd_mm * math.cos(math.radians(alpha_deg))
    return {
        "BPFO": z / 2.0 * fr * (1.0 - c),
        "BPFI": z / 2.0 * fr * (1.0 + c),
        "BSF": pd_mm / (2.0 * bd_mm) * fr * (1.0 - c**2),
        "FTF": fr / 2.0 * (1.0 - c),
    }


#: The 6205 at its nominal speed, from the closed form.
FREQUENCIES_6205 = bearing_kinematics(Z_6205, BD_6205, PD_6205, 0.0, RPM_6205)

#: A fault-free family: (rng, fs in Hz, duration in s, shaft speed in Hz).
FaultFreeGenerator = Callable[[np.random.Generator, float, float, float], np.ndarray]

DefectKind = Literal["outer", "inner", "rolling_element", "cage", "combined"]


def _samples(fs: float, seconds: float) -> int:
    return int(round(fs * seconds))


def _resonance_response(
    fs: float, zeta: float = 0.05, length_s: float = 0.01, hz: float = RESONANCE_HZ
):
    t = np.arange(int(length_s * fs)) / fs
    wn = 2.0 * np.pi * hz
    return np.exp(-zeta * wn * t) * np.sin(wn * np.sqrt(1.0 - zeta**2) * t)


# ---------------------------------------------------------------------------
# Fault-free families
# ---------------------------------------------------------------------------


def white(rng, fs, seconds, shaft_hz):
    return rng.standard_normal(_samples(fs, seconds))


def shaft_lines(rng, fs, seconds, shaft_hz):
    n = _samples(fs, seconds)
    t = np.arange(n) / fs
    x = rng.standard_normal(n)
    for h in range(1, 6):
        x += (
            rng.uniform(0.2, 1.5)
            / h
            * np.sin(2 * np.pi * h * shaft_hz * t + rng.uniform(0, 2 * np.pi))
        )
    for line_hz in (50.0, 100.0):
        x += rng.uniform(0.1, 0.5) * np.sin(
            2 * np.pi * line_hz * t + rng.uniform(0, 2 * np.pi)
        )
    return x


def shaft_modulated(rng, fs, seconds, shaft_hz, depth=0.5):
    n = _samples(fs, seconds)
    t = np.arange(n) / fs
    modulation = 1.0 + depth * np.sin(
        2 * np.pi * shaft_hz * t + rng.uniform(0, 2 * np.pi)
    )
    modulation += (
        0.3 * depth * np.sin(2 * np.pi * 2 * shaft_hz * t + rng.uniform(0, 2 * np.pi))
    )
    return modulation * rng.standard_normal(n)


def resonant_coloured(rng, fs, seconds, shaft_hz):
    n = _samples(fs, seconds)
    white_part = rng.standard_normal(n)
    random_walk = np.cumsum(rng.standard_normal(n))
    random_walk -= np.convolve(random_walk, np.ones(256) / 256, mode="same")
    random_walk /= np.std(random_walk)
    b, a = iirpeak(RESONANCE_HZ, 8.0, fs=fs)
    resonant = lfilter(b, a, rng.standard_normal(n))
    resonant /= np.std(resonant)
    return 0.5 * white_part + 0.5 * random_walk + resonant


def random_impacts(rng, fs, seconds, shaft_hz, rate_hz=20.0):
    n = _samples(fs, seconds)
    x = 0.5 * rng.standard_normal(n)
    count = rng.poisson(rate_hz * seconds)
    impacts = np.zeros(n)
    impacts[rng.integers(0, n, size=count)] = rng.uniform(0.5, 2.0, size=count)
    return x + np.convolve(impacts, _resonance_response(fs), mode="full")[:n]


def gear_tones(rng, fs, seconds, shaft_hz):
    n = _samples(fs, seconds)
    t = np.arange(n) / fs
    mesh_hz = 41 * shaft_hz
    return (
        0.5 * rng.standard_normal(n)
        + np.sin(2 * np.pi * mesh_hz * t)
        + 0.5 * np.sin(2 * np.pi * 2 * mesh_hz * t)
    )


FAULT_FREE: dict[str, FaultFreeGenerator] = {
    "white": white,
    "shaft_lines": shaft_lines,
    "shaft_modulated": shaft_modulated,
    "resonant_coloured": resonant_coloured,
    "random_impacts": random_impacts,
    "gear_tones": gear_tones,
}


# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------


def defect(
    kind: DefectKind,
    rng: np.random.Generator,
    fs: float,
    seconds: float,
    frequencies: dict[str, float],
    shaft_hz: float,
    snr_db: float,
    slip: float = 0.01,
    depth: float = 0.8,
    resonance_hz: float = RESONANCE_HZ,
) -> np.ndarray:
    """One defect scenario (see the module docstring for the model).

    ``resonance_hz`` moves the excited resonance, so that a narrow or low
    envelope band can hold the impacts.
    """
    if kind == "combined":
        return sum(
            defect(
                part,
                rng,
                fs,
                seconds,
                frequencies,
                shaft_hz,
                snr_db,
                slip,
                depth,
                resonance_hz,
            )
            for part in ("outer", "inner")
        )
    # Impact rate and amplitude-modulation frequency (None: constant amplitude).
    rate_hz, modulation_hz = {
        "outer": (frequencies["BPFO"], None),
        "inner": (frequencies["BPFI"], shaft_hz),
        "rolling_element": (2.0 * frequencies["BSF"], frequencies["FTF"]),
        "cage": (frequencies["FTF"], None),
    }[kind]
    period = 1.0 / rate_hz
    times = []
    t0 = rng.uniform(0, period)
    while t0 < seconds:
        times.append(t0)
        t0 += period * (1 + slip * rng.standard_normal())
    times = np.array(times)
    if modulation_hz is None:
        amplitude = np.ones_like(times)
    else:
        amplitude = 1 + depth * np.cos(
            2 * np.pi * modulation_hz * times + rng.uniform(0, 2 * np.pi)
        )

    n = _samples(fs, seconds)
    impacts = np.zeros(n)
    np.add.at(impacts, np.clip((times * fs).astype(int), 0, n - 1), amplitude)
    response = np.convolve(
        impacts, _resonance_response(fs, hz=resonance_hz), mode="full"
    )[:n]
    response /= np.std(response)
    t = np.arange(n) / fs
    shaft = 0.3 * np.sin(2 * np.pi * shaft_hz * t) + 0.1 * np.sin(
        2 * np.pi * 2 * shaft_hz * t
    )
    return response + 10 ** (-snr_db / 20.0) * rng.standard_normal(n) + shaft


def shared_line(
    rng: np.random.Generator,
    fs: float,
    seconds: float,
    frequencies: dict[str, float],
    noise_rms: float = 0.05,
) -> np.ndarray:
    """The shared-line scenario (see the module docstring)."""
    n = _samples(fs, seconds)
    t = np.arange(n) / fs
    bpfo = frequencies["BPFO"]
    # Amplitudes sum to 0.9, so the modulation never falls below 0.1.
    modulation = (
        1.0
        + 0.3 * np.sin(2 * np.pi * bpfo * t)
        + 0.2 * np.sin(2 * np.pi * 2 * bpfo * t)
        + 0.2 * np.sin(2 * np.pi * 3 * bpfo * t)
        + 0.2 * np.sin(2 * np.pi * 1.5 * bpfo * t + rng.uniform(0, 2 * np.pi))
    )
    carrier = np.sin(2 * np.pi * RESONANCE_HZ * t)
    return carrier * modulation + noise_rms * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Bundled MFPT recordings
# ---------------------------------------------------------------------------


def load_mfpt(relative: str) -> tuple[np.ndarray, float, dict[str, float], float]:
    """A bundled MFPT recording under ``data/signals``, with what it declares.

    Returns:
        (samples, sampling rate in Hz, declared BPFO/BPFI/BSF/FTF in Hz,
        shaft speed in rpm), the last three from the companion metadata.
    """
    path = REPO_ROOT / "data" / "signals" / relative
    meta = json.loads(path.with_name(f"{path.stem}_metadata.json").read_text())
    samples = pd.read_csv(path, header=None).to_numpy(dtype=float).ravel()
    frequencies = {k: float(meta[k]) for k in ("BPFO", "BPFI", "BSF", "FTF")}
    return samples, float(meta["sampling_rate"]), frequencies, meta["shaft_speed"] * 60
