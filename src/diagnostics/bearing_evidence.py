"""Bearing evidence: line significance, corroboration and the check outcome.

Decides, on one envelope spectrum, which tested fault frequencies carry a
line that stands above the local noise floor, which of those lines further
lines of their kinematics corroborate, and which outcome the whole check
reaches. Pure functions over arrays: no I/O, no MCP, no refusal of input
(the callers in :mod:`.bearing_analyzer` validate the signal first).

Line test (ordered-statistic CFAR, Rohling 1983, "Radar CFAR thresholding in
clutter and multiple target situations", IEEE Trans. AES 19(4), 608-621).
For a line searched over M bins, the statistic is the largest bin power in
the window divided by the k-th smallest of N reference powers on both sides
of the window beyond a guard of two bins, k the median rank (N + 1) // 2.
The per-bin false-alarm probability at threshold T is

    P(T) = prod_{i=0}^{k-1} (N - i) / (N - i + T),

and T is solved so that M * P(T) equals the probability given to the line.
A counted line is also a local maximum strictly inside its window, so the
leakage skirt of a line just outside the window is never counted.

Policy (one owner: ``EVIDENCE_POLICY``). A complete check has a nominal 1 %
family-wise false-alarm budget for finding a significant fundamental line on
a record with no periodic content at any tested frequency, split evenly
(Bonferroni) over the hypotheses it could test. Each corroborating line is tested at 1 %: a type needs a
significant fundamental first, so corroboration only lowers the chance of a
named type, and the inconclusive rate is set by the fundamental tests.

Assumptions (R24):

- bin power of the envelope spectrum is approximately exponential under a
  stationary random envelope (true of the envelope and the squared
  envelope alike);
- the noise floor is the median-rank order statistic of the reference, and
  the threshold is exact under this model: independent, exponentially
  distributed reference and search bins;
- Bonferroni over the bins of a window and over the hypotheses of a check,
  valid under any correlation between the tests.

The Hann window correlates neighbouring bins: that makes the closed form
somewhat optimistic about the floor, while Bonferroni over correlated
window bins and the local-maximum rule err the other way. The net rate per
window is measured on seeded synthetic noise (the calibration test) and is
close to nominal, not conservative; no false-alarm rate on real machines
is established by that calibration.

Exclusions: deterministic non-bearing lines (shaft harmonics, mains, gear
mesh) are real lines, so the probability does not bound how often they
produce an inconclusive or candidate outcome; non-stationary records break
the exponential model; repeated checks of one machine multiply the chances
of a false line (the ledger's persistence rule addresses repetition, not
this probability). The probability is not the system's false-alarm rate or
specificity.

Corroboration, sharing and shaft orders. A detected line's family is the
line plus the corroborating lines its kinematics predict, anchored at the
detected frequency: harmonics h = 2 .. num_harmonics + 1, the shaft-rate
sidebands of the inner race, the cage-rate sidebands and the 1 x BSF line of
the rolling element (whose line is tested at 2 x BSF: a rolling element
strikes both races once per spin revolution). A bin inside the window of a
member of another detected hypothesis's family (the search window for its
detected line, the corroboration window for the others, whether or not a
line was found there) is shared and supports neither, whichever line is
stronger: BPFO = Z x FTF and BPFO + BPFI = Z x shaft rate make such overlaps
structural (on a 6205, 3 x BPFO and 2 x BPFI are 0.71 % apart). A line
inside the window of an integer shaft order supports nothing, because shaft
harmonics are common in envelope spectra (Smith & Randall 2015, MSSP 64-65,
100-131). The shaft-order rule applies to the bearing vocabulary only: a
label outside it (a gear mesh, which is a shaft order by construction) gets
the line test and harmonic corroboration only. A cage supported on its own
is not singled out while the rolling-element line is significant: a
rolling-element defect is modulated at the cage rate (Randall & Antoni 2011,
MSSP 25, 485-520), so its cage-rate content can support the cage. The rule
is conservative: it also holds the cage back when the line in the
rolling-element window lies at a shaft order and may be a shaft harmonic, a
sensitivity cost accepted to avoid naming the wrong type. Harmonics
and sidebands of one spectrum are correlated symptoms, never independent
evidence (ISO 13379-1:2012, 5.1.3).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple, Optional

import numpy as np
from scipy.optimize import brentq

#: Nominal family-wise false-alarm budget of one complete check, split by
#: Bonferroni over the hypotheses it can test and the bins of their windows.
CHECK_FALSE_ALARM_PROBABILITY = 0.01
#: Nominal false-alarm budget of one corroborating window, over its bins.
CORROBORATION_FALSE_ALARM_PROBABILITY = 0.01
#: A local maximum strictly inside a window needs three bins.
MIN_SEARCH_BINS = 3
#: Fewest reference bins from which the floor is estimated.
MIN_REFERENCE_BINS = 16
#: Reference width on each side: this fraction of the line frequency, and
#: at least MIN_REFERENCE_BINS_PER_SIDE bins, where the spectrum allows.
REFERENCE_WIDTH_FRACTION = 0.3
MIN_REFERENCE_BINS_PER_SIDE = 40
#: Bins left out between a window and its reference (Hann main lobe).
GUARD_BINS = 2
#: Reference bins start at this many bins above 0 Hz: the lowest bins hold
#: the leakage of the envelope mean, which is removed before the window.
LOWEST_REFERENCE_BIN = 3
#: Corroborating and shaft-order windows: the larger of this many bins and
#: this fraction of the window's centre frequency, on each side.
MIN_NARROW_WINDOW_BINS = 3
NARROW_WINDOW_FRACTION = 0.01
#: Lines and reference bins are used only below this fraction of the envelope
#: band width (band high minus band low). The envelope of band-limited noise
#: has no content beyond that width and its density falls towards it, so a
#: reference reaching past it sits below the floor at the line and the
#: threshold is too low: 9 to 15 % of noise checks then had a significant
#: line, against 1 % nominal, with a 200 Hz band. 0.9 leaves out the
#: band-pass transition and brings those configurations to about 1 to 1.5 %,
#: measured on seeded noise; it does not restore the nominal rate where a
#: line lies at 0.4 to 0.8 of the width, where the floor slopes (an exclusion
#: of the policy). It also costs sensitivity: further lines beyond the limit
#: cannot be tested, and the outcome then asks for a wider band.
USABLE_BAND_FRACTION = 0.9

#: Fault types whose kinematics add sidebands or a subharmonic.
_INNER_RACE = "inner_race"
_ROLLING_ELEMENT = "ball"

_CAGE = "cage"

EVIDENCE_POLICY: dict[str, Any] = {
    "method": "os_cfar_envelope_line_test",
    "version": 1,
    "false_alarm_probability_per_check": CHECK_FALSE_ALARM_PROBABILITY,
    "false_alarm_probability_meaning": (
        "Nominal family-wise false-alarm budget of one complete check: under "
        "the assumptions listed, the probability that the check finds a "
        "significant line at a tested fault frequency in a record that has no "
        "periodic content there is at most this budget. It is split evenly over "
        "the hypotheses the check could test and, within each, over the bins of "
        "its search window (Bonferroni). The realized rate of the complete "
        "procedure is estimated by calibration, not asserted to equal the budget."
    ),
    "false_alarm_probability_per_corroborating_line": (
        CORROBORATION_FALSE_ALARM_PROBABILITY
    ),
    "min_search_bins": MIN_SEARCH_BINS,
    "min_reference_bins": MIN_REFERENCE_BINS,
    "reference_width_fraction": REFERENCE_WIDTH_FRACTION,
    "min_reference_bins_per_side": MIN_REFERENCE_BINS_PER_SIDE,
    "guard_bins": GUARD_BINS,
    "lowest_reference_bin": LOWEST_REFERENCE_BIN,
    "usable_band_fraction": USABLE_BAND_FRACTION,
    "narrow_window_fraction": NARROW_WINDOW_FRACTION,
    "min_narrow_window_bins": MIN_NARROW_WINDOW_BINS,
    "assumptions": [
        "Bin power of the envelope spectrum is approximately exponentially "
        "distributed under a stationary random envelope.",
        "The noise floor is the median-rank order statistic of reference bins "
        "on both sides of the search window (ordered-statistic CFAR, Rohling "
        "1983). The threshold is exact under this model: independent, "
        "exponentially distributed reference and search bins.",
        "Bonferroni allocation of the budget over the bins of each search "
        "window and over the hypotheses of the check, which bounds the "
        "family-wise probability under any correlation between the tests. The "
        "Hann window correlates neighbouring bins, which makes the "
        "floor estimate somewhat optimistic and the Bonferroni bound somewhat "
        "conservative; the net rate per window, measured on seeded noise, is "
        "close to nominal.",
        "Lines and reference bins are used only below 0.9 of the envelope "
        "band width, where the envelope of band-limited noise carries content.",
    ],
    "exclusions": [
        "Deterministic non-bearing lines (shaft harmonics, mains, gear mesh) "
        "are real lines, not noise: the budget does not bound how often "
        "they produce an inconclusive or candidate outcome.",
        "Non-stationary records, such as speed or load changing during the "
        "capture, are outside the exponential model.",
        "Repeated checks of the same machine multiply the chances of a false "
        "line: the budget applies to one check.",
        "Under amplitude modulation at the shaft rate, a chance line has copies "
        "at plus and minus the shaft rate, where the inner-race sidebands are "
        "tested: those sidebands are then not independent of the line, and the "
        "nominal budget per corroborating line does not bound them.",
        "Where a tested line lies at about 0.4 to 0.8 of the envelope band width, "
        "the envelope floor slopes and the rate per check was measured at up to "
        "about 1.5 times the nominal.",
        "The complete procedure is calibrated on seeded synthetic noise only; "
        "no false-alarm rate on real machines is established by it.",
        "The budget is not the false-alarm rate or the specificity of the " "system.",
    ],
}


class _Names(NamedTuple):
    line: str  # the line, as a sentence names it
    element: str  # what an inspection looks at
    defect: str  # what the evidence is consistent with


_NAMES = {
    "outer_race": _Names(
        "outer-race line (BPFO)", "outer race", "an outer-race defect"
    ),
    "inner_race": _Names(
        "inner-race line (BPFI)", "inner race", "an inner-race defect"
    ),
    "ball": _Names(
        "rolling-element line (2 x BSF)", "rolling elements", "a rolling-element defect"
    ),
    "cage": _Names("cage line (FTF)", "cage", "a cage defect"),
}


#: Closes every outcome that names a type.
_ONE_SPECTRUM = (
    "Lines of one spectrum are not independent evidence and do not confirm a fault."
)


class _Untestable(NamedTuple):
    reason: str  # completes "The {line} at {f} Hz could not be tested: ..."
    step: str  # the next step, with {lines} naming the lines it applies to


_UNTESTABLE = {
    "search_window_too_narrow": _Untestable(
        "its search window holds fewer than 3 frequency bins; a longer capture "
        "gives finer frequency resolution",
        "Record a longer capture to test {lines}.",
    ),
    "noise_floor_not_estimable": _Untestable(
        "the noise floor around it could not be estimated; a longer capture gives "
        "more frequency bins around it",
        "Record a longer capture to test {lines}.",
    ),
    "outside_usable_envelope_band": _Untestable(
        "its search window reaches beyond the usable part of the envelope band, "
        "whose width sets the highest line frequency the envelope can carry",
        "Use a wider envelope band, which may need a higher sampling rate, to test "
        "{lines}.",
    ),
    "above_frequency_range": _Untestable(
        "its search window lies above half the sampling rate",
        "Record at a higher sampling rate, with an envelope band wide enough to "
        "include it, to test {lines}.",
    ),
}


@dataclass(frozen=True)
class Hypothesis:
    """One tested fault frequency.

    Attributes:
        label: The label the caller checks (BPFO, BPFI, BSF, FTF or any
            other, such as "GMF").
        expected_hz: The expected frequency as declared. For the rolling
            element this is the ball spin frequency; its line is tested at
            twice that.
        canonical: outer_race, inner_race, ball or cage; None for a label
            outside the bearing vocabulary, which gets the line test and
            harmonic corroboration only.
    """

    label: str
    expected_hz: float
    canonical: Optional[str]

    @property
    def line_hz(self) -> float:
        """The frequency of the line the hypothesis is tested at."""
        return (
            2.0 * self.expected_hz
            if self.canonical == _ROLLING_ELEMENT
            else self.expected_hz
        )


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def os_cfar_false_alarm_probability(
    threshold: float, reference_bins: int, rank: int
) -> float:
    """Per-bin false-alarm probability of the ordered-statistic test.

    Args:
        threshold: Power ratio T to the rank-th smallest reference power.
        reference_bins: N, the number of reference bins.
        rank: k, 1-based rank of the reference order statistic.

    Returns:
        prod_{i=0}^{k-1} (N - i) / (N - i + T), for exponential bin power.
    """
    i = np.arange(rank, dtype=float)
    terms = (reference_bins - i) / (reference_bins - i + threshold)
    return float(np.exp(np.sum(np.log(terms))))


@lru_cache(maxsize=4096)
def os_cfar_threshold(
    reference_bins: int, rank: int, search_bins: int, probability: float
) -> float:
    """Threshold T such that search_bins x per-bin probability = probability.

    Args:
        reference_bins: N.
        rank: k.
        search_bins: M, the bins searched for the line.
        probability: The probability given to the line (its share of the
            check, or the corroborating-line probability).

    Returns:
        T, a power ratio to the k-th smallest reference power.
    """
    log_target = math.log(probability / search_bins)

    def excess(threshold: float) -> float:
        probability = os_cfar_false_alarm_probability(threshold, reference_bins, rank)
        return math.log(probability) - log_target

    upper = 1.0
    while excess(upper) > 0:
        upper *= 2.0
    return float(brentq(excess, 0.0, upper, xtol=1e-12, rtol=4 * np.finfo(float).eps))


# ---------------------------------------------------------------------------
# One line
# ---------------------------------------------------------------------------


def _measure(
    freqs: np.ndarray,
    mags: np.ndarray,
    df: float,
    centre_hz: float,
    half_width_hz: float,
    usable_hz: Optional[float] = None,
) -> dict[str, Any]:
    """Window, reference and strongest bin of one line, before any threshold.

    ``usable_hz`` is the highest frequency the line and its reference may
    use (``USABLE_BAND_FRACTION`` x the envelope band width); None uses the
    whole spectrum.
    """
    measured: dict[str, Any] = {
        "expected_hz": centre_hz,
        "half_width_hz": half_width_hz,
    }
    if centre_hz - half_width_hz > freqs[-1]:
        return {
            **measured,
            "testable": False,
            "untestable_reason": "above_frequency_range",
        }
    if usable_hz is not None and centre_hz + half_width_hz > usable_hz:
        return {
            **measured,
            "testable": False,
            "untestable_reason": "outside_usable_envelope_band",
        }
    window = np.flatnonzero(np.abs(freqs - centre_hz) <= half_width_hz)
    if window.size < MIN_SEARCH_BINS:
        return {
            **measured,
            "testable": False,
            "untestable_reason": "search_window_too_narrow",
        }

    lo, hi = int(window[0]), int(window[-1])
    side = max(
        math.ceil(REFERENCE_WIDTH_FRACTION * centre_hz / df),
        MIN_REFERENCE_BINS_PER_SIDE,
    )
    lowest = int(np.searchsorted(freqs, LOWEST_REFERENCE_BIN * df * (1 - 1e-9)))
    below = mags[max(lo - GUARD_BINS - side, lowest) : max(lo - GUARD_BINS, lowest)]
    reach = min(hi + GUARD_BINS + 1 + side, freqs.size)
    end = reach if usable_hz is None else min(reach, _bins_below(freqs, usable_hz))
    above = mags[hi + GUARD_BINS + 1 : end]
    reference = np.concatenate([below, above]) ** 2
    rank = (reference.size + 1) // 2
    floor = (
        float(np.partition(reference, rank - 1)[rank - 1]) if reference.size else 0.0
    )
    if reference.size < MIN_REFERENCE_BINS or not floor > 0:
        # Too few bins because the usable band cut the reference short.
        lost = max(0, reach - max(end, hi + GUARD_BINS + 1))
        cut = lost > 0 and reference.size + lost >= MIN_REFERENCE_BINS
        return {
            **measured,
            "testable": False,
            "untestable_reason": (
                "outside_usable_envelope_band" if cut else "noise_floor_not_estimable"
            ),
        }

    peak = int(window[np.argmax(mags[window])])
    local_maximum = (
        lo < peak < hi and mags[peak] > mags[peak - 1] and mags[peak] > mags[peak + 1]
    )
    return {
        **measured,
        "testable": True,
        "untestable_reason": None,
        "peak_hz": float(freqs[peak]),
        "amplitude": float(mags[peak]),
        "ratio": float(mags[peak]) ** 2 / floor,
        "local_maximum": bool(local_maximum),
        "search_bins": int(window.size),
        "reference_bins": int(reference.size),
        "reference_rank": int(rank),
    }


def _bins_below(freqs: np.ndarray, limit_hz: float) -> int:
    """Number of leading bins at or below ``limit_hz``."""
    return int(np.searchsorted(freqs, limit_hz, side="right"))


def _decide(measured: dict[str, Any], probability: Optional[float]) -> dict[str, Any]:
    """Apply the threshold for ``probability`` to a measured line."""
    if not measured["testable"] or probability is None:
        return {**measured, "significant": False}
    threshold = os_cfar_threshold(
        measured["reference_bins"],
        measured["reference_rank"],
        measured["search_bins"],
        probability,
    )
    return {
        **measured,
        "level_db": 10.0 * math.log10(max(measured["ratio"], 1e-30)),
        "threshold_db": 10.0 * math.log10(threshold),
        "false_alarm_probability": probability,
        "significant": measured["local_maximum"] and measured["ratio"] > threshold,
    }


def line_significance(
    freqs: np.ndarray,
    mags: np.ndarray,
    centre_hz: float,
    half_width_hz: float,
    probability: float,
) -> dict[str, Any]:
    """Test one line of an envelope spectrum against its local noise floor.

    Args:
        freqs: Positive frequency axis (Hz), uniform spacing, ascending.
        mags: Envelope spectrum magnitudes aligned with ``freqs``.
        centre_hz: Centre of the search window.
        half_width_hz: Half-width of the search window (inclusive).
        probability: Probability given to this line (M x per-bin).

    Returns:
        ``testable`` and, if not, ``untestable_reason``; if testable, the
        strongest bin (``peak_hz``, ``amplitude``), ``level_db`` and
        ``threshold_db`` (power ratios to the floor), ``local_maximum``,
        ``significant``, ``search_bins``, ``reference_bins``,
        ``reference_rank`` and ``false_alarm_probability``.
    """
    freqs = np.asarray(freqs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    return _decide(
        _measure(freqs, mags, freqs[1] - freqs[0], centre_hz, half_width_hz),
        probability,
    )


# ---------------------------------------------------------------------------
# A check: several hypotheses on one spectrum
# ---------------------------------------------------------------------------


def _narrow_half_width(centre_hz: float, df: float) -> float:
    return max(MIN_NARROW_WINDOW_BINS * df, NARROW_WINDOW_FRACTION * centre_hz)


def _at_shaft_order(frequency_hz: float, shaft_hz: Optional[float], df: float) -> bool:
    if shaft_hz is None:
        return False
    order = max(1, round(frequency_hz / shaft_hz))
    return abs(frequency_hz - order * shaft_hz) <= _narrow_half_width(
        order * shaft_hz, df
    )


def _family(
    hypothesis: Hypothesis,
    detected_hz: float,
    shaft_hz: Optional[float],
    cage_hz: Optional[float],
    num_harmonics: int,
) -> list[tuple[str, Optional[int], float]]:
    """(kind, harmonic, centre) of each corroborating line of a detected line."""
    members: list[tuple[str, Optional[int], float]] = [
        ("harmonic", h, h * detected_hz) for h in range(2, num_harmonics + 2)
    ]
    modulation = {_INNER_RACE: shaft_hz, _ROLLING_ELEMENT: cage_hz}.get(
        hypothesis.canonical or ""
    )
    if modulation is not None:
        members += [
            ("sideband", None, detected_hz + sign * modulation)
            for sign in (-1, 1)
            if detected_hz + sign * modulation > 0
        ]
    if hypothesis.canonical == _ROLLING_ELEMENT:
        members.append(("subharmonic", None, detected_hz / 2.0))
    return members


def _line(hypothesis: Hypothesis) -> str:
    """'outer-race line (BPFO)', or 'GMF line' for a label outside the vocabulary."""
    names = _NAMES.get(hypothesis.canonical or "")
    return names.line if names else f"{hypothesis.label} line"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def evaluate_hypotheses(
    freqs: np.ndarray,
    mags: np.ndarray,
    hypotheses: list[Hypothesis],
    *,
    shaft_hz: Optional[float],
    cage_hz: Optional[float],
    num_harmonics: int = 3,
    tolerance_pct: float = 5.0,
    catalog: bool = False,
    envelope_band_hz: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate every hypothesis of a check on one envelope spectrum.

    Args:
        freqs: Positive frequency axis (Hz) of the envelope spectrum.
        mags: Envelope spectrum magnitudes.
        hypotheses: The tested fault frequencies, in the caller's order.
        shaft_hz: Shaft rate, or None when no speed is known: then no
            shaft-rate sideband or shaft-order test runs, and the reason
            says so.
        cage_hz: Cage rate for the rolling-element sidebands, or None.
        num_harmonics: Harmonics 2x .. (num_harmonics + 1)x tested per line.
        tolerance_pct: Half-width of each fundamental search window, in
            percent of its line frequency.
        catalog: Whether the frequencies come from a catalog bearing, which
            the diagnostic report can compare with a baseline signal; it
            only selects the example in the confirmation step.
        envelope_band_hz: Width of the band-pass band the envelope was taken
            from (band high minus band low). Lines and references are used
            only below ``USABLE_BAND_FRACTION`` of it; None uses the whole
            spectrum.

    Returns:
        ``checks`` (one dict per hypothesis, in input order, with the new
        per-check fields), ``outcome``, ``outcome_reason``, ``next_step``,
        ``candidates`` (supported hypotheses, strongest first) and
        ``evidence_policy``.
    """
    freqs = np.asarray(freqs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    df = float(freqs[1] - freqs[0])
    usable_hz = (
        None if envelope_band_hz is None else USABLE_BAND_FRACTION * envelope_band_hz
    )

    measured = [
        _measure(
            freqs, mags, df, h.line_hz, h.line_hz * tolerance_pct / 100.0, usable_hz
        )
        for h in hypotheses
    ]
    tested = sum(m["testable"] for m in measured)
    share = CHECK_FALSE_ALARM_PROBABILITY / tested if tested else None
    lines = [_decide(m, share) for m in measured]

    # Families of the detected hypotheses, with every member's window.
    families: dict[int, list[dict[str, Any]]] = {}
    for n, (h, line) in enumerate(zip(hypotheses, lines)):
        if not line["significant"]:
            continue
        families[n] = [
            {
                **_decide(
                    _measure(
                        freqs,
                        mags,
                        df,
                        centre,
                        _narrow_half_width(centre, df),
                        usable_hz,
                    ),
                    CORROBORATION_FALSE_ALARM_PROBABILITY,
                ),
                "kind": kind,
                "harmonic": harmonic,
            }
            for kind, harmonic, centre in _family(
                h, line["peak_hz"], shaft_hz, cage_hz, num_harmonics
            )
        ]

    def windows(n: int) -> list[tuple[float, float]]:
        own = lines[n]
        return [(own["expected_hz"], own["half_width_hz"])] + [
            (m["expected_hz"], m["half_width_hz"]) for m in families[n]
        ]

    def sharers(n: int, frequency_hz: float) -> list[str]:
        return [
            hypotheses[other].label
            for other in families
            if other != n and any(abs(frequency_hz - c) <= w for c, w in windows(other))
        ]

    checks = []
    for n, (h, line) in enumerate(zip(hypotheses, lines)):
        # Shaft orders discount bearing lines only: a gear mesh, the typical
        # label outside the vocabulary, is a shaft order by construction.
        order_hz = shaft_hz if h.canonical in _NAMES else None
        corroboration = []
        for member in families.get(n, []):
            excluded_by = None
            if member["significant"]:
                if _at_shaft_order(member["peak_hz"], order_hz, df):
                    excluded_by = "shaft_order"
                elif sharers(n, member["peak_hz"]):
                    excluded_by = "shared"
            corroboration.append(
                {
                    "kind": member["kind"],
                    "harmonic": member["harmonic"],
                    "expected_hz": round(member["expected_hz"], 3),
                    "peak_hz": _rounded(member.get("peak_hz"), 3),
                    "level_db": _rounded(member.get("level_db"), 2),
                    "threshold_db": _rounded(member.get("threshold_db"), 2),
                    "significant": bool(member["significant"]),
                    "counted": bool(member["significant"]) and excluded_by is None,
                    "excluded_by": excluded_by,
                    "untestable_reason": member["untestable_reason"],
                }
            )
        shared_with = sharers(n, line["peak_hz"]) if line["significant"] else []
        if not line["testable"]:
            state, reason = "not_testable", line["untestable_reason"]
        elif not line["significant"]:
            state, reason = "absent", "no_significant_line"
        elif _at_shaft_order(line["peak_hz"], order_hz, df):
            state, reason = "unconfirmed", "line_at_shaft_order"
        elif shared_with:
            state, reason = "shared", "line_shared"
        elif any(c["counted"] for c in corroboration):
            state, reason = "supported", "corroborated"
        else:
            state, reason = "unconfirmed", "no_corroboration"
        checks.append(
            {
                "fault_type": h.label,
                "state": state,
                "state_reason": reason,
                "line_frequency_hz": round(h.line_hz, 3),
                "line_peak_hz": _rounded(line.get("peak_hz"), 3),
                "line_amplitude": line.get("amplitude"),
                "level_db": _rounded(line.get("level_db"), 2),
                "threshold_db": _rounded(line.get("threshold_db"), 2),
                "line_test": (
                    {
                        "search_bins": line["search_bins"],
                        "reference_bins": line["reference_bins"],
                        "reference_rank": line["reference_rank"],
                        "false_alarm_probability": line["false_alarm_probability"],
                    }
                    if line["testable"]
                    else None
                ),
                "corroboration": corroboration,
                "shared_with": shared_with,
            }
        )

    _hold_back_lone_cage(hypotheses, lines, checks)

    supported = sorted(
        (n for n, c in enumerate(checks) if c["state"] == "supported"),
        key=lambda n: (-lines[n]["level_db"], hypotheses[n].label),
    )
    candidates = [
        {
            "fault_type": hypotheses[n].label,
            "fault_type_canonical": hypotheses[n].canonical,
            "line_frequency_hz": checks[n]["line_frequency_hz"],
            "line_peak_hz": checks[n]["line_peak_hz"],
            "level_db": checks[n]["level_db"],
        }
        for n in supported
    ]
    weak = [n for n, c in enumerate(checks) if c["state"] in ("unconfirmed", "shared")]
    if len(supported) == 1:
        outcome = "indicated"
    elif len(supported) > 1:
        outcome = "several_types"
    elif weak or not tested:
        outcome = "inconclusive"
    else:
        outcome = "no_indication"

    reason, step = _outcome_text(
        outcome, hypotheses, checks, supported, weak, shaft_hz, cage_hz, catalog
    )
    return {
        "checks": checks,
        "outcome": outcome,
        "outcome_reason": reason,
        "next_step": step,
        "candidates": candidates,
        "evidence_policy": evidence_policy(tolerance_pct, num_harmonics),
    }


def _hold_back_lone_cage(
    hypotheses: list[Hypothesis],
    lines: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    """Unconfirm a cage that is the only supported type while the
    rolling-element line is significant (in place).

    A rolling-element defect is modulated at the cage rate as the element
    passes through the load zone, so its cage-rate content can support the
    cage line on its own.
    """
    supported = [n for n, c in enumerate(checks) if c["state"] == "supported"]
    if (
        len(supported) == 1
        and hypotheses[supported[0]].canonical == _CAGE
        and any(
            h.canonical == _ROLLING_ELEMENT and line["significant"]
            for h, line in zip(hypotheses, lines)
        )
    ):
        checks[supported[0]]["state"] = "unconfirmed"
        checks[supported[0]]["state_reason"] = "rolling_element_line_present"


def evidence_policy(tolerance_pct: float, num_harmonics: int) -> dict[str, Any]:
    """The policy record echoed in a result, with the call's window settings."""
    return {
        **EVIDENCE_POLICY,
        "assumptions": list(EVIDENCE_POLICY["assumptions"]),
        "exclusions": list(EVIDENCE_POLICY["exclusions"]),
        "search_window_pct": float(tolerance_pct),
        "harmonics_tested": int(num_harmonics),
    }


def _rounded(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------------------
# Outcome reason and next step (closed list of sentences)
# ---------------------------------------------------------------------------


def _outcome_text(
    outcome: str,
    hypotheses: list[Hypothesis],
    checks: list[dict[str, Any]],
    supported: list[int],
    weak: list[int],
    shaft_hz: Optional[float],
    cage_hz: Optional[float],
    catalog: bool,
) -> tuple[str, Optional[str]]:
    """The outcome's reason and next step, from a closed list of sentences.

    Every sentence is built from the states and corroboration fields of the
    same result, so it cannot contradict them.
    """
    reasons: list[str] = []
    steps: list[str] = []
    untested = [n for n, c in enumerate(checks) if c["state"] == "not_testable"]
    compare = "an earlier measurement of the same machine under comparable conditions"

    if outcome == "indicated":
        (n,) = supported
        names = _NAMES.get(hypotheses[n].canonical or "")
        found = (
            f"{_at(hypotheses[n], checks[n])} stands {checks[n]['level_db']:.1f} dB "
            f"above the local noise floor, and {_further_count(hypotheses[n], checks[n])} "
            "above the floor in the same spectrum."
        )
        if names:
            reasons.append(
                f"The envelope evidence of this single spectrum is consistent with "
                f"{names.defect}: {found}"
            )
            example = (
                "generate_diagnostic_report with baseline_signal_id, or the "
                "measurement point's history through get_asset_history"
                if catalog
                else "the measurement point's history through get_asset_history"
            )
            steps.append(
                f"Confirm with {compare} (for example {example}), an inspection of "
                f"the {names.element}, or other independent evidence."
            )
        else:
            reasons.append(_capital(found))
            steps.append(
                f"Confirm with {compare} or other independent evidence before acting "
                f"on the {hypotheses[n].label} line."
            )
        if len(hypotheses) > 1:
            reasons.append("No other tested type is supported in this spectrum.")
        reasons.append(_ONE_SPECTRUM)
    elif outcome == "several_types":
        listed = [
            f"{_at(hypotheses[n], checks[n])} ({checks[n]['level_db']:.1f} dB)"
            for n in supported
        ]
        reasons.append(
            "Several tested lines stand above the local noise floor, each "
            f"corroborated by a line of its own in the same spectrum: {_join(listed)}. "
            "They are listed by level above the floor, which does not rank them, and "
            "the envelope evidence does not single out one of them."
        )
        reasons.append(_ONE_SPECTRUM)
        elements = [_NAMES.get(hypotheses[n].canonical or "") for n in supported]
        if all(elements):
            parts = [f"the {names.element}" for names in elements if names]
            steps.append(
                f"Inspect {_join(parts)} with equal priority, and compare with {compare}."
            )
        else:
            labels = _join([hypotheses[n].label for n in supported])
            steps.append(f"Compare with {compare} before acting on the {labels} lines.")
    elif outcome == "inconclusive" and weak:
        reasons.append("The envelope evidence is inconclusive.")
        steps.append(f"Repeat the measurement, or compare it with {compare}.")
    elif outcome == "inconclusive":
        reasons.append("None of the lines could be tested in this record.")
    else:
        reasons.append(
            "No eligible peak (a local maximum inside a search window) at a tested "
            "frequency satisfies the detection criteria in this record and band. "
            "This does not show that "
            + (
                "the bearing is free of defects: the tested frequencies are only as "
                "accurate as the declared speed and bearing data."
                if all(h.canonical in _NAMES for h in hypotheses)
                else "no defect exists."
            )
        )

    rolling_element_at_shaft_order = any(
        h.canonical == _ROLLING_ELEMENT and c["state_reason"] == "line_at_shaft_order"
        for h, c in zip(hypotheses, checks)
    )
    reasons += [
        _weak_sentence(hypotheses[n], checks[n], rolling_element_at_shaft_order)
        for n in weak
    ]

    def cut(n: int, *untestable: str) -> bool:
        return any(
            m["untestable_reason"] in untestable for m in checks[n]["corroboration"]
        )

    above_range = [n for n in weak if cut(n, "above_frequency_range")]
    # The sampling-rate step also asks for a band wide enough, so a hypothesis
    # it names gets no separate band step.
    beyond_band = [
        n
        for n in weak
        if n not in above_range and cut(n, "outside_usable_envelope_band")
    ]
    too_short = [
        n
        for n in weak
        if cut(n, "search_window_too_narrow", "noise_floor_not_estimable")
    ]
    for named, step in zip((above_range, beyond_band, too_short), _CUT_STEPS):
        if named:
            lines = _join([f"the {_line(hypotheses[n])}" for n in named])
            steps.append(step.format(lines=lines))

    for n in untested:
        reasons.append(
            f"The {_line(hypotheses[n])} at {hypotheses[n].line_hz:.2f} Hz could not "
            f"be tested: {_UNTESTABLE[checks[n]['state_reason']].reason}."
        )
    for reason, untestable in _UNTESTABLE.items():
        named = [n for n in untested if checks[n]["state_reason"] == reason]
        if named:
            lines = _join([f"the {_line(hypotheses[n])}" for n in named])
            steps.append(untestable.step.format(lines=lines))
    if shaft_hz is None:
        reasons.append(
            "The shaft speed was not given, so no shaft-rate sideband or shaft-order "
            "test ran; declare the shaft speed to run them."
        )
        steps.append(
            "Declare the shaft speed to run the sideband and shaft-order tests."
        )
    elif cage_hz is None and any(h.canonical == _ROLLING_ELEMENT for h in hypotheses):
        reasons.append(
            "The cage rate was not given, so no cage-rate sideband test ran for the "
            "rolling-element line; an FTF entry among the frequencies runs it."
        )

    return " ".join(reasons), (" ".join(steps) or None)


def _at(hypothesis: Hypothesis, check: dict[str, Any]) -> str:
    """'the outer-race line (BPFO) at 107.40 Hz' (the detected line)."""
    return f"the {_line(hypothesis)} at {check['line_peak_hz']:.2f} Hz"


def _further_count(hypothesis: Hypothesis, check: dict[str, Any]) -> str:
    """'2 further lines its kinematics predict also stand' (counted lines)."""
    count = sum(c["counted"] for c in check["corroboration"])
    if hypothesis.canonical in _NAMES:
        noun = "further line its kinematics predict"
        nouns = "further lines its kinematics predict"
    else:
        noun = nouns = "of its harmonics"
    return f"1 {noun} also stands" if count == 1 else f"{count} {nouns} also stand"


def _weak_sentence(
    hypothesis: Hypothesis,
    check: dict[str, Any],
    rolling_element_at_shaft_order: bool = False,
) -> str:
    """Why a line that stands above the floor supports no type.

    ``rolling_element_at_shaft_order`` tells a held-back cage that the line
    in the rolling-element window lies at a shaft order.
    """
    found = (
        f"{_capital(_at(hypothesis, check))} stands above the local noise floor, but"
    )
    reason = check["state_reason"]
    bearing = hypothesis.canonical in _NAMES
    further = "the further lines its kinematics predict" if bearing else "its harmonics"
    if reason == "line_at_shaft_order":
        return (
            f"{found} it lies at an integer multiple of the shaft rotation frequency, "
            "where shaft lines are common, so it is not counted as evidence."
        )
    if reason == "line_shared":
        owners = check["shared_with"]
        if len(owners) == 1:
            return (
                f"{found} it lies where the {owners[0]} check also predicts a line, "
                "so it supports neither."
            )
        return (
            f"{found} it lies where the {_join(owners)} checks also predict a line, "
            "so it supports none of them."
        )
    if reason == "rolling_element_line_present":
        ball, cage = _NAMES[_ROLLING_ELEMENT], _NAMES[_CAGE]
        held_back = (
            "Under this conservative ambiguity rule, no single fault type is named "
            "from this spectrum."
        )
        if rolling_element_at_shaft_order:
            return (
                f"{found} another significant line falls within the search window of "
                f"the {ball.line} and lies at a shaft order. That line may be a shaft "
                f"harmonic and does not independently establish {ball.defect}. "
                f"{_capital(ball.defect)} cannot be excluded either: it would also "
                f"produce lines at the cage rate, so the {cage.element} lines may "
                f"come from one. {held_back}"
            )
        return (
            f"{found} another significant line falls within the search window of "
            f"the {ball.line}. {_capital(ball.defect)} also produces lines at the "
            f"cage rate, so the {cage.element} lines may come from one; that line "
            f"alone does not establish {ball.defect}. {held_back}"
        )
    if reason != "no_corroboration":
        raise ValueError(f"no sentence for state reason {reason!r}")
    members = check["corroboration"]
    if not members:
        return f"{found} no further line is tested for it at these settings."
    if any(m["significant"] for m in members):
        return (
            f"{found} those of {further} that also stand above the floor lie at a "
            "shaft order or where another tested line predicts one, so none is counted."
        )
    cut = {m["untestable_reason"] for m in members} - {None}
    if not cut:
        return f"{found} none of {further} stands above the floor."
    why = _cut_phrase(cut)
    if all(m["untestable_reason"] for m in members):
        return f"{found} none of {further} could be tested: {why}."
    return (
        f"{found} none of {further} that could be tested stands above the floor; "
        f"the others could not be tested: {why}."
    )


def _cut_phrase(reasons: set[str]) -> str:
    """Why some corroborating lines could not be tested."""
    if reasons == {"outside_usable_envelope_band"}:
        return "they reach beyond the usable part of the envelope band"
    if reasons == {"above_frequency_range"}:
        return "they lie above half the sampling rate"
    if reasons <= {"search_window_too_narrow", "noise_floor_not_estimable"}:
        return "this record is too short around them"
    return "they lie outside what this record and band can test"


#: Steps for corroborating lines a weak hypothesis could not test: above half
#: the sampling rate, beyond the usable band, too short a record.
_CUT_STEPS = (
    "Record at a higher sampling rate, with an envelope band wide enough to "
    "include them, to test the further lines of {lines}.",
    "Use a wider envelope band, which may need a higher sampling rate, to test "
    "the further lines of {lines}.",
    "Record a longer capture to test the further lines of {lines}.",
)


def _capital(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:]
