"""
Bearing fault detection functions.

Tests the envelope spectrum at expected bearing fault frequencies (BPFO,
BPFI, BSF, FTF, or any declared label) with :mod:`.bearing_evidence`, on one
envelope spectrum per check: per-check ``state``, line level and
corroboration, and per-summary ``outcome``, ``candidates`` and
``evidence_policy``. The per-check ``detected``, ``detected_frequency_hz``,
``magnitude``, ``deviation_pct``, ``harmonics_detected`` and
``evidence_strength`` derive from that evidence (a count of corroborating
lines of one spectrum, never a probability). The summary's ``claim``
(:mod:`.bearing_claims`) says what the check may state from that evidence,
and ``most_likely_fault``, ``most_likely_fault_canonical`` and
``overall_assessment`` follow it.
"""

import logging
from typing import Any, Optional

import numpy as np

from .bearing_catalog import compute_fault_frequencies, lookup_bearing
from .bearing_claims import (
    UNKNOWN,
    bearing_claim,
    claim_statement,
    named_candidate,
    validate_acquisition_eligibility,
)
from .bearing_evidence import Hypothesis, evaluate_hypotheses
from ..signal_processing.spectral import (
    envelope_spectrum_arrays,
    resolve_envelope_band,
)
from ..signal_validation import (
    validate_expected_frequencies,
    validate_expected_frequency,
    validate_rpm,
    validate_signal_and_fs,
)

logger = logging.getLogger(__name__)

#: Remedy for an expected frequency the peak check cannot use. On the catalog
#: route the frequency is computed, so a valid speed can still round it to
#: zero (a fraction of 1 rpm) and the remedy points back at the speed.
_PEAK_FREQUENCY_REMEDY = (
    "pass an expected frequency in Hz above zero; when it is computed from the "
    "shaft speed and the bearing geometry, check that rpm is the measured speed "
    "in revolutions per minute and that the geometry is right"
)

#: Canonical fault vocabulary: acronym -> canonical fault type. Arbitrary
#: labels (e.g. "GMF" for a gear-mesh check) have no canonical mapping.
FAULT_TYPE_CANONICAL: dict[str, str] = {
    "BPFO": "outer_race",
    "BPFI": "inner_race",
    "BSF": "ball",
    "FTF": "cage",
}


def check_bearing_fault_peak(
    signal: np.ndarray,
    fs: float,
    expected_freq: float,
    fault_type: str,
    bearing_id: str,
    signal_id: str = "",
    tolerance_pct: float = 5.0,
    num_harmonics: int = 3,
    envelope_freq_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Test the envelope spectrum for a line at one expected fault frequency.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        expected_freq: Expected fault frequency (Hz).
        fault_type: Fault type label (BPFO, BPFI, BSF, FTF).
        bearing_id: Bearing designation.
        signal_id: Signal identifier (for result labeling).
        tolerance_pct: Frequency tolerance in percent.
        num_harmonics: Number of harmonics to check (2x, 3x, ...).
        envelope_freq_range: Bandpass filter range for envelope. ``None``
            (default) uses the fs-aware default band (500 Hz up to
            min(5000, just below Nyquist)), so a low-fs signal is analyzed
            over its available band instead of raising.

    Returns:
        Dict compatible with BearingFaultCheckResult. The evidence fields
        test this one frequency alone and without a shaft speed, so no
        sideband or shaft-order test runs.

    Raises:
        ValueError: If fs is not a positive finite number, any sample is
            NaN or infinite, or ``expected_freq`` is not a positive finite
            number.
    """
    (check,), _ = _check_on_one_spectrum(
        signal,
        fs,
        [(fault_type, expected_freq)],
        bearing_id=bearing_id,
        signal_id=signal_id,
        tolerance_pct=tolerance_pct,
        num_harmonics=num_harmonics,
        envelope_freq_range=envelope_freq_range,
        shaft_hz=None,
        cage_hz=None,
    )
    return check


def _check_on_one_spectrum(
    signal: np.ndarray,
    fs: float,
    expected: list[tuple[str, float]],
    *,
    bearing_id: Optional[str],
    signal_id: str,
    tolerance_pct: float,
    num_harmonics: int,
    envelope_freq_range: Optional[tuple[float, float]],
    shaft_hz: Optional[float],
    cage_hz: Optional[float],
    catalog: bool = False,
    spectrum: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[list[dict], dict]:
    """Every (label, expected Hz) checked on ONE envelope spectrum.

    ``spectrum``, when given, is that spectrum as (frequencies, magnitudes),
    already computed by :func:`envelope_spectrum_arrays` on this signal and
    band; the signal, fs, frequencies and band are still validated first.

    Returns:
        (per-check dicts in input order, the evidence summary of
        :func:`.bearing_evidence.evaluate_hypotheses`).
    """
    # Every bearing verdict passes through here, so corrupt input is refused
    # before the envelope: a NaN envelope, or a zero, negative or NaN expected
    # frequency, would leave no line to test and read as no indication; an
    # infinite expected frequency has no window. The first frequency is
    # validated before the envelope band and spectrum, the others after, as
    # when the spectrum was computed once per frequency.
    signal = validate_signal_and_fs(signal, fs)
    (first_label, first_freq), *others = expected
    validate_expected_frequency(
        first_freq, label=first_label, remedy=_PEAK_FREQUENCY_REMEDY
    )
    band = resolve_envelope_band(fs, envelope_freq_range)
    # The band as the filter realizes it (an upper edge at Nyquist sits 1 Hz
    # below it, see envelope_spectrum_arrays).
    band_width = min(band[1], fs / 2.0 - 1.0) - band[0]
    if spectrum is None:
        freqs, mags = envelope_spectrum_arrays(signal, fs, band)
    else:
        freqs, mags = spectrum
    for label, expected_freq in others:
        validate_expected_frequency(
            expected_freq, label=label, remedy=_PEAK_FREQUENCY_REMEDY
        )

    evidence = evaluate_hypotheses(
        freqs,
        mags,
        _hypotheses(expected),
        shaft_hz=shaft_hz,
        cage_hz=cage_hz,
        num_harmonics=num_harmonics,
        tolerance_pct=tolerance_pct,
        catalog=catalog,
        envelope_band_hz=band_width,
    )
    checks = [
        {
            "signal_id": signal_id,
            "bearing_id": bearing_id,
            "fault_type": label,
            "fault_type_canonical": FAULT_TYPE_CANONICAL.get(label.upper()),
            "expected_frequency_hz": round(expected_freq, 2),
            **{k: v for k, v in fields.items() if k != "fault_type"},
        }
        for (label, expected_freq), fields in zip(expected, evidence["checks"])
    ]
    return checks, evidence


def _hypotheses(expected: list[tuple[str, float]]) -> list[Hypothesis]:
    """The tested hypotheses of (label, expected Hz) pairs, in input order."""
    return [
        Hypothesis(label, float(f), FAULT_TYPE_CANONICAL.get(label.upper()))
        for label, f in expected
    ]


def _summary_evidence(
    evidence: dict,
    expected: list[tuple[str, float]],
    *,
    shaft_hz: Optional[float],
    tolerance_pct: float,
    acquisition_eligibility: str,
) -> dict[str, Any]:
    """The summary fields of an evidence result, with the claim and the
    fields that follow it."""
    hypotheses = _hypotheses(expected)
    claim = bearing_claim(
        evidence,
        hypotheses,
        shaft_hz=shaft_hz,
        tolerance_pct=tolerance_pct,
        acquisition_eligibility=acquisition_eligibility,
    )
    named = named_candidate(claim, evidence)
    return {
        "overall_assessment": claim_statement(claim, evidence, hypotheses),
        "most_likely_fault": named["fault_type"] if named else None,
        "most_likely_fault_canonical": (
            named["fault_type_canonical"] if named else None
        ),
        **{
            key: evidence[key]
            for key in (
                "outcome",
                "outcome_reason",
                "next_step",
                "candidates",
                "evidence_policy",
            )
        },
        "claim": claim,
    }


def check_all_bearing_faults(
    signal: np.ndarray,
    fs: float,
    bearing_id: str,
    rpm: float,
    signal_id: str = "",
    tolerance_pct: float = 5.0,
    envelope_freq_range: Optional[tuple[float, float]] = None,
    num_harmonics: int = 3,
    *,
    acquisition_eligibility: str = UNKNOWN,
) -> dict:
    """Run all four fault checks (BPFO, BPFI, BSF, FTF) for a bearing.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        bearing_id: Bearing designation (e.g. "6205").
        rpm: Shaft speed in RPM.
        signal_id: Signal identifier.
        tolerance_pct: Frequency tolerance in percent.
        envelope_freq_range: Bandpass filter range for envelope. ``None``
            (default) uses the fs-aware default band (500 Hz up to
            min(5000, just below Nyquist)).
        num_harmonics: Harmonics (2x, 3x, ...) checked per fault type; the
            default reproduces :func:`check_bearing_fault_peak`'s. Exposed
            so the asset ledger's snapshot policy can pin it explicitly.
        acquisition_eligibility: "known_inapplicable" when the signal's own
            companion metadata declares a generated signal without noise
            (the claim is then withheld), else "unknown".

    Returns:
        Dict compatible with BearingFaultsSummary, with its ``claim``.

    Raises:
        ValueError: If rpm is not a positive finite number, if
            acquisition_eligibility is not one of its two values, if the
            bearing is not found in the catalog, or if
            :func:`check_bearing_fault_peak` refuses the signal, fs or a
            computed frequency (non-finite samples, a non-finite or
            non-positive fs, a frequency that rounds to zero at a fraction
            of 1 rpm).
    """
    # Before the catalog lookup, so the speed is reported even for an
    # unknown designation.
    validate_rpm(rpm)
    validate_acquisition_eligibility(acquisition_eligibility)
    freq_data = compute_fault_frequencies(bearing_id, rpm)
    if freq_data is None:
        raise ValueError(
            f"Bearing '{bearing_id}' not found in catalog. "
            f"Check designation or provide bearing geometry manually."
        )

    shaft_freq = freq_data["shaft_freq_hz"]
    fault_types = {
        "BPFO": freq_data["BPFO"],
        "BPFI": freq_data["BPFI"],
        "BSF": freq_data["BSF"],
        "FTF": freq_data["FTF"],
    }

    expected = list(fault_types.items())
    checks, evidence = _check_on_one_spectrum(
        signal,
        fs,
        expected,
        bearing_id=bearing_id,
        signal_id=signal_id,
        tolerance_pct=tolerance_pct,
        num_harmonics=num_harmonics,
        envelope_freq_range=envelope_freq_range,
        shaft_hz=shaft_freq,
        cage_hz=fault_types["FTF"],
        catalog=True,
    )

    bearing_freqs = {k: v for k, v in fault_types.items()}
    bearing_freqs["shaft_freq_hz"] = shaft_freq

    return {
        "signal_id": signal_id,
        "bearing_id": bearing_id,
        "rpm": rpm,
        "shaft_frequency_hz": shaft_freq,
        "bearing_frequencies": bearing_freqs,
        "fault_checks": checks,
        # Catalog traceability: every catalog entry carries a mandatory
        # source citation — echo it so results stay auditable.
        "source": (freq_data.get("bearing_info") or {}).get("source"),
        **_summary_evidence(
            evidence,
            expected,
            shaft_hz=shaft_freq,
            tolerance_pct=tolerance_pct,
            acquisition_eligibility=acquisition_eligibility,
        ),
    }


def check_frequency_set(
    signal: np.ndarray,
    fs: float,
    frequencies: dict[str, float],
    rpm: Optional[float],
    signal_id: str = "",
    tolerance_pct: float = 5.0,
    envelope_freq_range: Optional[tuple[float, float]] = None,
    num_harmonics: int = 3,
    *,
    acquisition_eligibility: str = UNKNOWN,
    spectrum: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Check an arbitrary set of labeled expected frequencies in the envelope.

    Covers the out-of-catalog and gearbox paths: labels may be the bearing
    acronyms (BPFO/BPFI/BSF/FTF — mapped to the canonical fault vocabulary)
    or any user label such as "GMF" (no canonical mapping).

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        frequencies: {label: expected frequency in Hz}, all values positive
            and finite.
        rpm: Shaft speed in RPM (echoed; shaft frequency = rpm/60), positive
            and finite; None when no speed is declared, which runs no
            shaft-rate sideband or shaft-order test (the outcome reason says
            so) and echoes ``rpm`` and ``shaft_frequency_hz`` as None.
        signal_id: Signal identifier (for result labeling).
        tolerance_pct: Frequency tolerance in percent.
        envelope_freq_range: Bandpass filter range for envelope. ``None``
            (default) uses the fs-aware default band (500 Hz up to
            min(5000, just below Nyquist)).
        num_harmonics: Harmonics (2x, 3x, ...) checked per label; the
            default reproduces :func:`check_bearing_fault_peak`'s.
        acquisition_eligibility: "known_inapplicable" when the signal's own
            companion metadata declares a generated signal without noise
            (the claim is then withheld), else "unknown".
        spectrum: The envelope spectrum of this signal on this band as
            (frequencies, magnitudes), as :func:`envelope_spectrum_arrays`
            returns it, when the caller has already computed it (the
            envelope report); None computes it here. The signal and fs are
            validated first either way.

    Returns:
        Dict compatible with BearingFaultsSummary (bearing_id/source None),
        with its ``claim``.

    Raises:
        ValueError: If a given rpm is not a positive finite number, if
            acquisition_eligibility is not one of its two values, if
            frequencies is empty or contains a value that is not a positive
            finite number (every such label is listed), or if
            :func:`check_bearing_fault_peak` refuses the signal or fs
            (non-finite samples, a non-finite or non-positive fs).
    """
    if rpm is not None:
        validate_rpm(rpm)
    shaft_hz = None if rpm is None else rpm / 60.0
    validate_acquisition_eligibility(acquisition_eligibility)
    if not frequencies:
        raise ValueError(
            "frequencies is empty — pass at least one {label: hz} entry, "
            "e.g. {'BPFO': 107.4} or {'GMF': 350.0}."
        )
    validate_expected_frequencies(frequencies)

    expected = [(label, float(freq)) for label, freq in frequencies.items()]
    checks, evidence = _check_on_one_spectrum(
        signal,
        fs,
        expected,
        bearing_id="",
        signal_id=signal_id,
        tolerance_pct=tolerance_pct,
        num_harmonics=num_harmonics,
        envelope_freq_range=envelope_freq_range,
        shaft_hz=shaft_hz,
        # A declared cage frequency is the rolling element's modulation rate.
        cage_hz=next(
            (
                float(freq)
                for label, freq in frequencies.items()
                if FAULT_TYPE_CANONICAL.get(label.upper()) == "cage"
            ),
            None,
        ),
        spectrum=spectrum,
    )

    freq_echo = {k: float(v) for k, v in frequencies.items()}
    if shaft_hz is not None:
        freq_echo["shaft_freq_hz"] = shaft_hz

    return {
        "signal_id": signal_id,
        "bearing_id": None,
        "rpm": rpm,
        "shaft_frequency_hz": shaft_hz,
        "bearing_frequencies": freq_echo,
        "fault_checks": checks,
        "source": None,
        **_summary_evidence(
            evidence,
            expected,
            shaft_hz=shaft_hz,
            tolerance_pct=tolerance_pct,
            acquisition_eligibility=acquisition_eligibility,
        ),
    }


def lookup_bearing_and_compute(
    signal: np.ndarray,
    fs: float,
    bearing_type: str,
    rpm: float,
    signal_id: str = "",
    tolerance_pct: float = 5.0,
) -> dict:
    """End-to-end: look up bearing, compute frequencies, check all faults.

    Convenience function combining catalog lookup and analysis.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        bearing_type: Bearing designation (e.g. "SKF 6205-2RS").
        rpm: Shaft speed in RPM.
        signal_id: Signal identifier.
        tolerance_pct: Frequency tolerance in percent.

    Returns:
        Dict compatible with BearingFaultsSummary.

    Raises:
        ValueError: As :func:`check_all_bearing_faults`.
    """
    return check_all_bearing_faults(
        signal=signal,
        fs=fs,
        bearing_id=bearing_type,
        rpm=rpm,
        signal_id=signal_id,
        tolerance_pct=tolerance_pct,
    )
