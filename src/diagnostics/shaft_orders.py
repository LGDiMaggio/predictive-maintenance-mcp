"""Shaft-order lines of the amplitude spectrum: measurement and line test.

Measures, on the server's amplitude spectrum
(:func:`..signal_processing.spectral.amplitude_spectrum`: Hamming window,
``2|X|/N``, no gain correction, the same numbers ``analyze_fft`` reports for
the same samples), the lines at integer and half-integer multiples of the
shaft rotation frequency, and tests each against its local noise floor. Pure
functions over arrays: no I/O, no MCP, no interpretation. What a pattern of
lines may mean is decided elsewhere (:mod:`.shaft_fault_evidence`).

ISO 13374 Block 2 (data manipulation) and the detection half of Block 3: a
line test establishes that a periodic component stands above the local noise
floor at an order, never that it is abnormal, large or caused by a fault.
Shaft-order lines are present in the spectrum of almost every rotating
machine.

Shaft frequency. The declared speed and its declared uncertainty
(``speed_tolerance_pct``) set a search window for the 1x line. When exactly
one significant line stands in that window, its interpolated frequency is
the shaft frequency the orders are measured from, with one frequency bin as
its uncertainty; otherwise (no line, or several competing lines) the
declared speed is used with its full uncertainty, which widens the windows
of the higher orders and can make them overlap. Which of the two was used is
always reported.

Line test. The ordered-statistic CFAR test of :mod:`.bearing_evidence`
(Rohling 1983), reused unchanged through :func:`.bearing_evidence.
os_cfar_threshold`: the strongest bin of the order's window over the
median-rank power of reference bins. The reference is the part of the
spectrum within half a shaft order of the line that lies outside every order
window (integer and half orders) and its two guard bins on each side, so the
other orders of the series never form the floor of a line. A nominal 1 %
family-wise false-alarm budget is split by Bonferroni over every order window
the call could test (two per order up to ``max_order``), fixed before the
data is seen. The model (exponential bin power of a stationary random
background) is the same as the bearing test's, and so are its limits: a
deterministic line that is not a shaft order (mains, a belt, a bearing tone)
in a window is a real line, not a false alarm.

Amplitudes. Each line reports the peak-bin amplitude in the server's
historical convention (a sinusoid of amplitude A on a bin reads about
0.54 A), and, for a significant line, an amplitude estimate of the
sinusoid: the power of the five bins of the Hamming main lobe around the
peak, less the noise floor, over the window's power gain. The estimate does
not depend on where the line falls between bins (no scalloping loss), so
ratios between orders do not either. It is a 0-to-peak amplitude in the
signal's unit and, when the unit is declared, also as a velocity in mm/s
(an acceleration line divided by 2 pi f at its frequency).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import numpy as np

from ..signal_acquisition.repository import normalize_signal_unit
from scipy.stats import kurtosis

from ..signal_processing.spectral import amplitude_spectrum
from ..signal_validation import (
    InvalidInputError,
    validate_positive_finite,
    validate_rpm,
    validate_signal_and_fs,
)
from .bearing_evidence import os_cfar_threshold
from .iso20816 import G_TO_M_S2, M_S_TO_MM_S

#: Line-test geometry, owned here (the values match the bearing envelope
#: test's, but its tuning is not inherited): a local maximum strictly
#: inside a window needs three bins; the floor needs at least 16 reference
#: bins; two guard bins on each side of every window keep the Hamming main
#: lobe (two bins each side) out of the reference; the three lowest bins
#: hold the leakage of any DC offset and never enter a reference.
MIN_SEARCH_BINS = 3
MIN_REFERENCE_BINS = 16
GUARD_BINS = 2
LOWEST_REFERENCE_BIN = 3
#: Nominal family-wise false-alarm budget of one call, split by Bonferroni
#: over every order window the call could test.
FAMILY_FALSE_ALARM_PROBABILITY = 0.01
#: Orders are tested in steps of half a shaft order.
ORDER_STEP = 0.5
#: Highest integer order tested by default, and the bounds of the parameter.
DEFAULT_MAX_ORDER = 10
MIN_MAX_ORDER = 2
MAX_MAX_ORDER = 20
#: Default half-width of the 1x search window, in percent of the declared
#: shaft frequency: the project's convention for a declared speed (the health
#: snapshot's ``one_x_tolerance_pct``). It is the uncertainty of the
#: declaration, not a diagnostic threshold; declare a smaller one when the
#: speed was measured (for example by a tachometer).
DEFAULT_SPEED_TOLERANCE_PCT = 5.0
MAX_SPEED_TOLERANCE_PCT = 15.0
#: Ratio of sampling rate to analysis bandwidth conventional in vibration
#: analyzers (fs = 2.56 x Fmax), which leaves room for the anti-aliasing
#: filter. A line above fs / 2.56 is tested but flagged: an acquisition
#: filter may attenuate it.
CONVENTIONAL_BANDWIDTH_RATIO = 2.56
#: Half-width, in bins, of the Hamming main lobe: the amplitude estimate sums
#: this many bins on each side of the peak.
MAIN_LOBE_HALF_WIDTH_BINS = 2
#: A second significant maximum in the 1x window competes with the strongest
#: only when it is outside the strongest line's main lobe (this many bins
#: away) and within 30 dB of it. Relative to the peak bin of a tone half a
#: bin off the grid, the highest Hamming sidelobe is 40.8 dB down, and 2 %
#: amplitude modulation puts sidebands 40 dB down: 30 dB leaves 10 dB of
#: margin, so one line, clean or slightly modulated, never competes with
#: itself. A weaker line in the window is not separated from leakage and
#: modulation of the strongest and does not count.
COMPETING_LINE_MIN_SEPARATION_BINS = 3
COMPETING_LINE_MIN_LEVEL_DB = -30.0
#: Largest absolute sample value analyzed: squaring larger values overflows
#: double precision.
MAX_ABS_SAMPLE = 1e100
#: Competing lines listed by frequency (their count is always reported).
MAX_COMPETING_LISTED = 5
#: Smallest sampling rate and speed accepted: below them the frequency axis
#: or the shaft orders cannot be represented.
MIN_SAMPLING_RATE_HZ = 1.0
MIN_RPM = 1.0
#: Records longer than this are analyzed on their leading samples (2**22,
#: about 164 s at 25.6 kHz), so time and memory stay bounded.
MAX_ANALYSIS_SAMPLES = 2**22
#: Mains frequencies checked against the order windows when no line
#: frequency is declared.
COMMON_LINE_FREQUENCIES_HZ = (50.0, 60.0)
#: Multiples of the line frequency checked against the order windows.
LINE_FREQUENCY_MULTIPLES = (1, 2, 3, 4)

#: Velocity band of the 1X share: the ISO 20816-3 broadband range the
#: severity evaluation uses (iso20816.assess_severity_raw), 10 Hz to 1 kHz,
#: with a 2 Hz lower edge below 600 rpm.
BAND_LOW_HZ = 10.0
BAND_LOW_HZ_SLOW = 2.0
BAND_HIGH_HZ = 1000.0
SLOW_MACHINE_RPM = 600.0

_ACCELERATION_UNITS = ("g", "m/s2")
_VELOCITY_UNITS = ("mm/s", "m/s")

ORDER_LINE_POLICY: dict[str, Any] = {
    "method": "os_cfar_shaft_order_line_test",
    "version": 1,
    "false_alarm_probability_per_call": FAMILY_FALSE_ALARM_PROBABILITY,
    "false_alarm_probability_meaning": (
        "Nominal family-wise false-alarm budget of one call, not an empirically "
        "demonstrated guarantee. Under the assumptions listed, the model bounds "
        "by this budget the probability that the call finds a significant line in "
        "any order window of a record with no periodic content at the orders; it "
        "is split evenly (Bonferroni) over every order window the call could test "
        "and, within each, over the bins of the window. The realized rate of the "
        "complete procedure is measured by calibration, not asserted: see "
        "measured_false_alarm_rates."
    ),
    "measured_false_alarm_rates": (
        "On 3000 seeded records per family: 0.37 % on white noise and 0.33 % on "
        "pink noise (1x line not located), and 1.33 % (95 % interval 0.95 to "
        "1.81 %) for chance lines at the other orders next to located 1X and 2X "
        "lines, above the nominal 1 %. No rate on real machines is established."
    ),
    "order_step": ORDER_STEP,
    "min_search_bins": MIN_SEARCH_BINS,
    "min_reference_bins": MIN_REFERENCE_BINS,
    "guard_bins": GUARD_BINS,
    "lowest_reference_bin": LOWEST_REFERENCE_BIN,
    "reference_span": "half a shaft order on each side of the line",
    "resolution": (
        "Two components closer than about 3 frequency bins merge into one line; "
        "a line is testable only with at least 16 reference bins within half an "
        "order, which needs a record of about 40 shaft revolutions or more."
    ),
    "conventional_bandwidth_ratio": CONVENTIONAL_BANDWIDTH_RATIO,
    "max_analysis_samples": MAX_ANALYSIS_SAMPLES,
    "assumptions": [
        "Bin power of the amplitude spectrum is approximately exponentially "
        "distributed under a stationary random background.",
        "The noise floor of a line is the median-rank order statistic of the "
        "reference bins within half a shaft order of it, outside every order "
        "window and its guard bins (ordered-statistic CFAR, Rohling 1983).",
        "Bonferroni allocation of the budget over the bins of each window and "
        "over every window the call could test, which bounds the family-wise "
        "probability under any correlation between the tests.",
        "The shaft speed is constant during the record; the record is checked "
        "for a change of the 1x frequency between its halves.",
    ],
    "exclusions": [
        "Shaft-order lines are deterministic in almost every rotating machine: a "
        "significant line shows that a periodic component stands above the "
        "local noise floor at that order, not that it is abnormal, large or "
        "caused by a fault.",
        "A deterministic line that is not a shaft order (mains, a belt, a "
        "bearing or gear tone) inside an order window is a real line: the "
        "budget does not bound how often it is reported at that order.",
        "Non-stationary records, such as speed or load changing during the "
        "capture, are outside the exponential model.",
        "The located 1x line is the strongest significant line within the declared "
        "speed uncertainty when no other line within 30 dB of it competes: a "
        "stronger line that is not the shaft rate (a belt, the supply frequency) "
        "inside the window, or any line there when the true speed lies outside the "
        "declared uncertainty, can be taken as 1x, and every order is then "
        "measured from it.",
        "The budget applies to one call; repeated calls on the same machine "
        "multiply the chances of a chance line.",
        "The budget is not the false-alarm rate or the specificity of any "
        "diagnostic use of the lines.",
    ],
}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _finite(value: float) -> Optional[float]:
    """The value, or None when it is NaN or infinite (a constant record)."""
    return float(value) if math.isfinite(value) else None


def hamming_power_gain(n: int) -> float:
    """``sum(w**2) / n`` of numpy's symmetric Hamming window, in closed form.

    With w[k] = 0.54 - 0.46 cos(2 pi k / (n - 1)), the cosine sums to 1 over
    k = 0 .. n - 1 and its square to (n + 1) / 2 (for n >= 4), so sum(w**2) is
    0.54**2 n - 2 x 0.54 x 0.46 + 0.46**2 (n + 1) / 2.
    """
    if n < 4:  # the period sums below need n - 1 >= 3
        return float(np.sum(np.hamming(n) ** 2) / n) if n > 1 else 1.0
    total = 0.54**2 * n - 2 * 0.54 * 0.46 + 0.46**2 * (n + 1) / 2
    return float(total / n)


def _screening_statistics(x: np.ndarray) -> dict[str, Optional[float]]:
    """RMS, excess kurtosis and crest factor (the analyze_statistics conventions).

    None where a statistic is undefined (a constant or all-zero record).
    """
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    centred = x - np.mean(x)
    flat = not np.any(centred) or float(np.std(x)) <= 1e-12 * max(peak, 1e-300)
    return {
        "rms": rms,
        "kurtosis": None if flat else _finite(float(kurtosis(x, fisher=True))),
        "crest_factor": peak / rms if rms > 0 else None,
    }


def signal_quantity(signal_unit: Optional[str]) -> Optional[str]:
    """'acceleration', 'velocity' or None for an undeclared unit."""
    if signal_unit in _ACCELERATION_UNITS:
        return "acceleration"
    if signal_unit in _VELOCITY_UNITS:
        return "velocity"
    return None


def velocity_amplitude_mm_s(
    amplitude: float, frequency_hz: float, signal_unit: Optional[str]
) -> Optional[float]:
    """The amplitude of a sinusoid at *frequency_hz* as a velocity in mm/s.

    An acceleration amplitude is divided by 2 pi f (single integration of a
    sinusoid); a velocity amplitude is only rescaled. None for an undeclared
    unit: the conversion is never guessed.
    """
    if signal_unit is None or not frequency_hz > 0:
        return None
    if signal_unit == "mm/s":
        return float(amplitude)
    if signal_unit == "m/s":
        return float(amplitude) * M_S_TO_MM_S
    if signal_unit in _ACCELERATION_UNITS:
        accel = amplitude * (G_TO_M_S2 if signal_unit == "g" else 1.0)
        return float(accel / (2.0 * math.pi * frequency_hz) * M_S_TO_MM_S)
    return None


# ---------------------------------------------------------------------------
# One line
# ---------------------------------------------------------------------------


def _bins_within(
    freqs: np.ndarray, centre_hz: float, half_width_hz: float
) -> np.ndarray:
    """Indices of the bins with ``|f - centre| <= half_width`` (float-safe)."""
    slack = half_width_hz * 1e-9 + 1e-12
    lo = int(np.searchsorted(freqs, centre_hz - half_width_hz - slack, side="left"))
    hi = int(np.searchsorted(freqs, centre_hz + half_width_hz + slack, side="right"))
    return np.arange(lo, hi)


def interpolated_peak_hz(freqs: np.ndarray, mags: np.ndarray, peak: int) -> float:
    """Frequency of a spectral peak, by a parabola through the log magnitudes.

    The parabola through the logarithms of the peak bin and its two
    neighbours; for a Hamming-windowed sinusoid its vertex is within a few
    hundredths of a bin of the true frequency. The peak bin's own frequency
    is returned when a neighbour is missing or the three bins do not form a
    maximum.
    """
    if peak <= 0 or peak >= len(mags) - 1:
        return float(freqs[peak])
    a, b, c = (float(m) for m in mags[peak - 1 : peak + 2])
    if min(a, b, c) <= 0:
        return float(freqs[peak])
    la, lb, lc = math.log(a), math.log(b), math.log(c)
    denominator = la - 2.0 * lb + lc
    if not denominator < 0:
        return float(freqs[peak])
    offset = min(0.5, max(-0.5, 0.5 * (la - lc) / denominator))
    return float(freqs[peak]) + offset * float(freqs[0])


def _amplitude_estimate(
    mags: np.ndarray, peak: int, floor_power: float, window_power_gain: float
) -> float:
    """0-to-peak amplitude of a sinusoid from the power of its main lobe.

    ``window_power_gain`` is ``sum(w**2) / N`` of the analysis window. For a
    sinusoid of amplitude A, the five bins of the Hamming main lobe hold
    ``A**2 x window_power_gain`` of ``mags**2`` wherever the sinusoid falls
    between bins (to within the sidelobe energy, about 0.1 %). The mean noise
    power per bin (the median floor over ln 2, for exponential bin power) is
    removed first.
    """
    lo = max(0, peak - MAIN_LOBE_HALF_WIDTH_BINS)
    hi = min(len(mags), peak + MAIN_LOBE_HALF_WIDTH_BINS + 1)
    lobe_power = float(np.sum(mags[lo:hi] ** 2))
    noise_power = (hi - lo) * floor_power / math.log(2.0)
    return math.sqrt(max(lobe_power - noise_power, 0.0) / window_power_gain)


def _measure_line(
    freqs: np.ndarray,
    mags: np.ndarray,
    centre_hz: float,
    half_width_hz: float,
    reference_span_hz: float,
    exclusions: list[tuple[float, float]],
    probability: float,
) -> dict[str, Any]:
    """Window, reference, floor and significance of one line.

    Returns ``testable`` and, when it is not, ``untestable_reason``
    (``above_frequency_range``, ``search_window_too_narrow`` or
    ``noise_floor_not_estimable``); when it is, the peak bin and the test.
    """
    df = float(freqs[0])  # the first positive bin of an FFT axis is fs / N
    out: dict[str, Any] = {"expected_hz": centre_hz, "half_width_hz": half_width_hz}
    if centre_hz + half_width_hz > freqs[-1]:
        return {**out, "testable": False, "untestable_reason": "above_frequency_range"}
    window = _bins_within(freqs, centre_hz, half_width_hz)
    if window.size < MIN_SEARCH_BINS:
        return {
            **out,
            "testable": False,
            "untestable_reason": "search_window_too_narrow",
        }

    span = _bins_within(freqs, centre_hz, reference_span_hz)
    keep = freqs[span] >= LOWEST_REFERENCE_BIN * df * (1 - 1e-9)
    guard = GUARD_BINS * df
    span_freqs = freqs[span]
    for start, stop in exclusions:
        keep &= ~(
            (span_freqs >= start - guard - 1e-9 * df)
            & (span_freqs <= stop + guard + 1e-9 * df)
        )
    reference = mags[span[keep]] ** 2
    rank = (reference.size + 1) // 2
    floor = (
        float(np.partition(reference, rank - 1)[rank - 1]) if reference.size else 0.0
    )
    if reference.size < MIN_REFERENCE_BINS or not floor > 0:
        return {
            **out,
            "testable": False,
            "untestable_reason": "noise_floor_not_estimable",
            "reference_bins": int(reference.size),
        }

    lo, hi = int(window[0]), int(window[-1])
    peak = int(window[np.argmax(mags[window])])
    local_maximum = (
        lo < peak < hi and mags[peak] > mags[peak - 1] and mags[peak] > mags[peak + 1]
    )
    ratio = float(mags[peak]) ** 2 / floor
    threshold = os_cfar_threshold(
        int(reference.size), int(rank), int(window.size), probability
    )
    return {
        **out,
        "testable": True,
        "untestable_reason": None,
        "peak_bin": peak,
        "peak_hz": float(freqs[peak]),
        "amplitude": float(mags[peak]),
        "level_db": 10.0 * math.log10(max(ratio, 1e-30)),
        "threshold_db": 10.0 * math.log10(threshold),
        "local_maximum": bool(local_maximum),
        "significant": bool(local_maximum and ratio > threshold),
        "search_bins": int(window.size),
        "reference_bins": int(reference.size),
        "reference_rank": int(rank),
        "floor_power": floor,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Shaft frequency
# ---------------------------------------------------------------------------


def locate_shaft_line(
    freqs: np.ndarray,
    mags: np.ndarray,
    nominal_hz: float,
    tolerance_pct: float,
    probability: float,
) -> dict[str, Any]:
    """The 1x line within the declared speed's uncertainty, if exactly one.

    Searches ``nominal_hz`` +/- ``tolerance_pct`` (at least three bins) for
    the strongest bin and tests it. Every other local maximum of the window
    that also exceeds the threshold, lies outside the strongest line's main
    lobe and stands above its sidelobes is a competing line; with one, the
    window holds more than one periodic component and none is taken as 1x.

    Returns:
        ``status``: ``located`` (with ``frequency_hz``, the interpolated
        frequency), ``not_significant``, ``ambiguous`` (with
        ``competing_hz``) or ``untestable`` (with ``untestable_reason``);
        plus the line test of the strongest bin.
    """
    df = float(freqs[0])
    uncertainty = tolerance_pct / 100.0 * nominal_hz
    half_width = max(uncertainty, MIN_SEARCH_BINS * df)
    neighbours = []
    for k in (0.5, 1.5):
        hw = max(k * uncertainty, MIN_SEARCH_BINS * df)
        neighbours.append((k * nominal_hz - hw, k * nominal_hz + hw))
    measured = _measure_line(
        freqs,
        mags,
        nominal_hz,
        half_width,
        0.5 * nominal_hz,
        [(nominal_hz - half_width, nominal_hz + half_width), *neighbours],
        probability,
    )
    if not measured["testable"]:
        return {"status": "untestable", **measured}
    if not measured["significant"]:
        return {"status": "not_significant", **measured}

    window = _bins_within(freqs, nominal_hz, half_width)
    peak = measured["peak_bin"]
    threshold_power = measured["threshold"] * measured["floor_power"]
    min_level = mags[peak] * 10.0 ** (COMPETING_LINE_MIN_LEVEL_DB / 20.0)
    competing = []
    for i in window[1:-1]:
        i = int(i)
        if abs(i - peak) < COMPETING_LINE_MIN_SEPARATION_BINS:
            continue
        if mags[i] > mags[i - 1] and mags[i] > mags[i + 1]:
            if mags[i] ** 2 > threshold_power and mags[i] >= min_level:
                competing.append(float(freqs[i]))
    if competing:
        strongest = sorted(
            competing, key=lambda f: -mags[int(np.searchsorted(freqs, f))]
        )
        return {
            "status": "ambiguous",
            "competing_hz": sorted(strongest[:MAX_COMPETING_LISTED]),
            "competing_count": len(competing),
            **measured,
        }
    return {
        "status": "located",
        "frequency_hz": interpolated_peak_hz(freqs, mags, peak),
        **measured,
    }


# ---------------------------------------------------------------------------
# A record
# ---------------------------------------------------------------------------


def _line_frequency_multiples(
    centre_hz: float, half_width_hz: float, line_frequency_hz: Optional[float]
) -> list[dict[str, Any]]:
    """Multiples of the (declared or common) line frequency inside a window."""
    candidates = (
        (line_frequency_hz,)
        if line_frequency_hz is not None
        else COMMON_LINE_FREQUENCIES_HZ
    )
    found = []
    for lf in candidates:
        for m in LINE_FREQUENCY_MULTIPLES:
            if abs(m * lf - centre_hz) <= half_width_hz:
                found.append(
                    {
                        "line_frequency_hz": float(lf),
                        "multiple": m,
                        "frequency_hz": float(m * lf),
                        "declared": line_frequency_hz is not None,
                    }
                )
    return found


def _velocity_band(
    freqs: np.ndarray,
    mags: np.ndarray,
    window_power_gain: float,
    rpm: float,
    signal_unit: Optional[str],
    one_x: dict[str, Any],
) -> dict[str, Any]:
    """Velocity power of the evaluation band and the share the 1X line carries.

    The band is the ISO 20816-3 broadband range (10 Hz to 1 kHz, from 2 Hz
    below 600 rpm), cut at half the sampling rate. The mean-square velocity
    of the band is the sum over its bins of the squared velocity amplitude
    over twice the window power gain, so a sinusoid inside the band counts
    as its own mean square; the 1X share is the 1X line's mean square
    (half its squared amplitude estimate) over the band's. It describes the
    spectrum ("dominated by shaft-rate content" when above one half); it is
    not the ISO severity value, which comes from a filtered time signal.
    """
    # The 1X line's main lobe must lie inside the band, or the band power
    # misses part of it and the share is overstated: near 600 rpm the 2 Hz
    # edge of slower machines is used.
    lobe = (MAIN_LOBE_HALF_WIDTH_BINS + 1) * float(freqs[0])
    one_x_hz = one_x.get("frequency_hz") or rpm / 60.0
    slow = rpm < SLOW_MACHINE_RPM or one_x_hz - lobe < BAND_LOW_HZ
    low = BAND_LOW_HZ_SLOW if slow else BAND_LOW_HZ
    high = min(BAND_HIGH_HZ, float(freqs[-1]))
    out: dict[str, Any] = {
        "status": "not_computed",
        "reason": None,
        "low_hz": low,
        "high_hz": high,
        "truncated": bool(high < BAND_HIGH_HZ),
        "velocity_rms_mm_s": None,
        "one_x_share": None,
    }
    if signal_quantity(signal_unit) is None:
        return {**out, "reason": "unit_undeclared"}
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return {**out, "reason": "band_empty"}
    scale = velocity_amplitude_mm_s(1.0, 1.0, signal_unit)
    assert scale is not None
    if signal_quantity(signal_unit) == "acceleration":
        velocity = mags[band] * scale / freqs[band]
    else:
        velocity = mags[band] * scale
    mean_square = float(np.sum(velocity**2) / (2.0 * window_power_gain))
    share = None
    v1 = one_x.get("velocity_amplitude_mm_s")
    inside = v1 is not None and low <= one_x_hz - lobe and one_x_hz + lobe <= high
    if inside and mean_square > 0:
        share = min(1.0, (v1**2 / 2.0) / mean_square)
    return {
        **out,
        "status": "computed",
        "velocity_rms_mm_s": math.sqrt(mean_square),
        "one_x_share": share,
    }


def _validated_parameters(
    speed_tolerance_pct: Any, max_order: Any, line_frequency_hz: Any
) -> tuple[float, int, Optional[float]]:
    tolerance = validate_positive_finite(
        speed_tolerance_pct,
        name="speed_tolerance_pct",
        remedy=(
            "declare the uncertainty of the shaft speed in percent, above 0 and at "
            f"most {MAX_SPEED_TOLERANCE_PCT:g}"
        ),
    )
    if tolerance > MAX_SPEED_TOLERANCE_PCT:
        raise InvalidInputError(
            f"speed_tolerance_pct must be at most {MAX_SPEED_TOLERANCE_PCT:g}, got "
            f"{speed_tolerance_pct!r} — with a wider uncertainty the windows of the "
            f"low orders approach each other; measure the speed (for example "
            f"with a tachometer) and declare its uncertainty."
        )
    if (
        isinstance(max_order, bool)
        or not isinstance(max_order, (int, np.integer))
        or not MIN_MAX_ORDER <= int(max_order) <= MAX_MAX_ORDER
    ):
        raise InvalidInputError(
            f"max_order must be a whole number from {MIN_MAX_ORDER} to "
            f"{MAX_MAX_ORDER}, got {max_order!r} — pass the highest shaft order to "
            f"test, for example {DEFAULT_MAX_ORDER}."
        )
    lf: Optional[float] = None
    if line_frequency_hz is not None:
        lf = validate_positive_finite(
            line_frequency_hz,
            name="line_frequency_hz",
            remedy="declare the electrical supply frequency in Hz, for example 50 or 60",
        )
    return tolerance, int(max_order), lf


def _half_record_1x(
    signal: np.ndarray,
    fs: float,
    nominal_hz: float,
    tolerance_pct: float,
    probability: float,
) -> dict[str, Any]:
    """Whether the 1x frequency differs between the two halves of the record."""
    half = len(signal) // 2
    located = []
    for part in (signal[:half], signal[half : 2 * half]):
        if part.size < 3:
            return {"status": "not_assessed", "reason": "record_too_short"}
        freqs, mags = amplitude_spectrum(part, fs)
        if freqs.size < 2:
            return {"status": "not_assessed", "reason": "record_too_short"}
        line = locate_shaft_line(freqs, mags, nominal_hz, tolerance_pct, probability)
        if line["status"] != "located":
            return {
                "status": "not_assessed",
                "reason": "1x_not_located_in_both_halves",
            }
        located.append((line["frequency_hz"], line["amplitude"]))
    df_half = fs / half
    drift = abs(located[0][0] - located[1][0])
    return {
        "status": "assessed",
        "frequency_hz": [float(located[0][0]), float(located[1][0])],
        "amplitude": [float(located[0][1]), float(located[1][1])],
        "frequency_resolution_hz": float(df_half),
        "drift_hz": float(drift),
        "speed_changed": bool(drift > df_half),
    }


def measure_shaft_orders(
    signal: Any,
    fs: Any,
    rpm: Any,
    *,
    signal_unit: Optional[str] = None,
    speed_tolerance_pct: Any = DEFAULT_SPEED_TOLERANCE_PCT,
    max_order: Any = DEFAULT_MAX_ORDER,
    line_frequency_hz: Any = None,
) -> dict[str, Any]:
    """Measure and test the shaft-order lines of one vibration record.

    Args:
        signal: 1-D vibration samples.
        fs: Sampling frequency in Hz.
        rpm: Declared shaft speed in revolutions per minute.
        signal_unit: Declared unit ('g', 'm/s2', 'mm/s', 'm/s' or an alias);
            None when undeclared. Without it amplitudes stay in the signal's
            own unit and no velocity is computed.
        speed_tolerance_pct: Declared uncertainty of the speed, in percent.
        max_order: Highest integer order tested (half orders up to it too).
        line_frequency_hz: Declared electrical supply frequency, if any. When
            None, 50 and 60 Hz are checked against the order windows.

    Returns:
        A dict with ``analysis`` (samples, duration, resolution, revolutions,
        truncation), ``speed`` (declared and used shaft frequency, its source
        and uncertainty, the 1x search), ``lines`` (one entry per order, in
        order), ``time_domain`` (RMS, excess kurtosis, crest factor),
        ``stationarity`` (the 1x frequency of the two halves) and ``policy``.

    Raises:
        InvalidInputError: If a sample or a parameter is invalid (NaN or
            infinite samples, fs or rpm not positive, an unknown unit, a
            tolerance or order outside its bounds).
    """
    raw = np.asarray(signal)
    if raw.dtype.kind in "biu":
        # Integer or boolean samples: analyzed as float64 (no integer overflow).
        signal = raw.astype(np.float64)
    x = np.asarray(validate_signal_and_fs(signal, fs), dtype=np.float64)
    if float(fs) < MIN_SAMPLING_RATE_HZ:
        raise InvalidInputError(
            f"fs must be at least {MIN_SAMPLING_RATE_HZ:g} Hz for the shaft-order "
            f"measurement, got {fs!r} — pass the sampling frequency in Hz."
        )
    if x.size and float(np.max(np.abs(x))) > MAX_ABS_SAMPLE:
        raise InvalidInputError(
            f"The signal has samples beyond {MAX_ABS_SAMPLE:g} in magnitude, which the "
            f"spectrum cannot square without overflowing; check the unit and the "
            f"scaling of the record and pass it in a unit that keeps it smaller."
        )
    fs = float(fs)
    if x.ndim != 1 or x.size < 3:
        raise InvalidInputError(
            f"The shaft-order measurement needs a 1-D signal of at least 3 samples, "
            f"got shape {x.shape} — pass the waveform of one channel."
        )
    rpm = validate_rpm(rpm)
    if rpm < MIN_RPM:
        raise InvalidInputError(
            f"rpm must be at least {MIN_RPM:g} for the shaft-order measurement, got "
            f"{rpm!r} — pass the shaft speed in revolutions per minute."
        )
    unit: Optional[str] = None
    if signal_unit is not None:
        unit = normalize_signal_unit(signal_unit)
        if unit is None:
            raise InvalidInputError(
                f"Invalid signal_unit {signal_unit!r} — declare 'g' or 'm/s2' for "
                f"acceleration, 'mm/s' or 'm/s' for velocity, or omit it."
            )
    tolerance, k_max, lf = _validated_parameters(
        speed_tolerance_pct, max_order, line_frequency_hz
    )

    total_samples = int(x.size)
    truncated = total_samples > MAX_ANALYSIS_SAMPLES
    if truncated:
        x = x[:MAX_ANALYSIS_SAMPLES]
    n = int(x.size)
    duration = n / fs
    nominal_hz = rpm / 60.0
    orders = [ORDER_STEP * j for j in range(1, int(round(k_max / ORDER_STEP)) + 1)]
    probability = FAMILY_FALSE_ALARM_PROBABILITY / len(orders)

    freqs, mags = amplitude_spectrum(x, fs)
    df = fs / n
    window_power_gain = hamming_power_gain(n)

    search = locate_shaft_line(freqs, mags, nominal_hz, tolerance, probability)
    if search["status"] == "located":
        shaft_hz = float(search["frequency_hz"])
        uncertainty = df
        source = "located_line"
    else:
        shaft_hz = nominal_hz
        uncertainty = tolerance / 100.0 * nominal_hz
        source = "declared"

    # Windows of every tested order and of the next half order (the upper
    # neighbour of the last), so the reference of every line excludes them.
    def half_width(order: float) -> float:
        return float(max(MIN_SEARCH_BINS * df, order * uncertainty))

    all_orders = [*orders, orders[-1] + ORDER_STEP]
    exclusions = [
        (k * shaft_hz - half_width(k), k * shaft_hz + half_width(k)) for k in all_orders
    ]
    guard = GUARD_BINS * df
    usable_limit = fs / CONVENTIONAL_BANDWIDTH_RATIO

    lines: list[dict[str, Any]] = []
    for k in orders:
        centre = k * shaft_hz
        hw = half_width(k)
        neighbours_overlap = any(
            hw + half_width(j) + 2 * guard >= ORDER_STEP * shaft_hz
            for j in (k - ORDER_STEP, k + ORDER_STEP)
            if j > 0
        )
        entry: dict[str, Any]
        if centre + hw > freqs[-1]:
            entry = {
                "expected_hz": centre,
                "half_width_hz": hw,
                "testable": False,
                "untestable_reason": "above_frequency_range",
            }
        elif neighbours_overlap:
            entry = {
                "expected_hz": centre,
                "half_width_hz": hw,
                "testable": False,
                "untestable_reason": "order_windows_overlap",
            }
        else:
            entry = _measure_line(
                freqs, mags, centre, hw, ORDER_STEP * shaft_hz, exclusions, probability
            )
        line: dict[str, Any] = {
            "order": k,
            "integer_order": float(k).is_integer(),
            "expected_hz": float(centre),
            "half_width_hz": float(hw),
            "testable": bool(entry["testable"]),
            "untestable_reason": entry.get("untestable_reason"),
            "significant": bool(entry.get("significant", False)),
            "peak_hz": None,
            "frequency_hz": None,
            "measured_order": None,
            "amplitude": None,
            "amplitude_estimate": None,
            "velocity_amplitude_mm_s": None,
            "level_db": None,
            "threshold_db": None,
            "local_maximum": None,
            "search_bins": None,
            "reference_bins": entry.get("reference_bins"),
            "beyond_conventional_bandwidth": bool(centre + hw > usable_limit),
            "line_frequency_multiples": (
                _line_frequency_multiples(centre, hw, lf)
                if float(k).is_integer()
                else []
            ),
        }
        if entry["testable"]:
            peak = entry["peak_bin"]
            frequency = interpolated_peak_hz(freqs, mags, peak)
            line.update(
                {
                    "peak_hz": entry["peak_hz"],
                    "frequency_hz": frequency,
                    "measured_order": frequency / shaft_hz,
                    "amplitude": entry["amplitude"],
                    "level_db": entry["level_db"],
                    "threshold_db": entry["threshold_db"],
                    "local_maximum": entry["local_maximum"],
                    "search_bins": entry["search_bins"],
                }
            )
            if entry["significant"]:
                estimate = _amplitude_estimate(
                    mags, peak, entry["floor_power"], window_power_gain
                )
                line["amplitude_estimate"] = estimate
                line["velocity_amplitude_mm_s"] = velocity_amplitude_mm_s(
                    estimate, frequency, unit
                )
        lines.append(line)

    time_domain = _screening_statistics(x)
    one_x = next(line for line in lines if line["order"] == 1.0)
    significant = [line["order"] for line in lines if line["significant"]]
    search_out = {
        "status": search["status"],
        "half_width_hz": float(search["half_width_hz"]),
        "peak_hz": search.get("peak_hz"),
        "level_db": search.get("level_db"),
        "threshold_db": search.get("threshold_db"),
        "competing_hz": search.get("competing_hz", []),
        "competing_count": search.get("competing_count", 0),
        "untestable_reason": search.get("untestable_reason"),
    }
    return {
        "analysis": {
            "samples": n,
            "total_samples": total_samples,
            "truncated": truncated,
            "sampling_rate_hz": fs,
            "duration_s": duration,
            "frequency_resolution_hz": df,
            "revolutions": duration * shaft_hz,
            "conventional_bandwidth_hz": usable_limit,
        },
        "speed": {
            "declared_rpm": rpm,
            "declared_shaft_hz": nominal_hz,
            "tolerance_pct": tolerance,
            "shaft_hz": shaft_hz,
            "source": source,
            "uncertainty_hz": uncertainty,
            "deviation_from_declared_pct": 100.0 * (shaft_hz - nominal_hz) / nominal_hz,
            "search": search_out,
        },
        "unit": {"signal_unit": unit, "quantity": signal_quantity(unit)},
        "line_frequency_hz": lf,
        "lines": lines,
        "significant_orders": significant,
        "velocity_band": _velocity_band(
            freqs, mags, window_power_gain, rpm, unit, one_x
        ),
        "time_domain": {
            "rms": time_domain["rms"],
            "kurtosis_excess": time_domain["kurtosis"],
            "crest_factor": time_domain["crest_factor"],
        },
        "stationarity": _half_record_1x(x, fs, nominal_hz, tolerance, probability),
        "policy": copy.deepcopy(ORDER_LINE_POLICY),
    }
