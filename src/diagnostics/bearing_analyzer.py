"""
Bearing fault detection functions.

Checks envelope spectrum peaks against expected bearing fault frequencies
(BPFO, BPFI, BSF, FTF) with tolerance matching, harmonic detection,
and evidence-strength rating (derived from detected peaks/harmonics,
never presented as a probability).
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .bearing_catalog import compute_fault_frequencies, lookup_bearing
from ..signal_processing.spectral import compute_envelope_spectrum

logger = logging.getLogger(__name__)

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
    envelope_freq_range: tuple[float, float] = (500, 5000),
) -> dict:
    """Check if envelope spectrum has a peak near an expected fault frequency.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        expected_freq: Expected fault frequency (Hz).
        fault_type: Fault type label (BPFO, BPFI, BSF, FTF).
        bearing_id: Bearing designation.
        signal_id: Signal identifier (for result labeling).
        tolerance_pct: Frequency tolerance in percent.
        num_harmonics: Number of harmonics to check (2x, 3x, ...).
        envelope_freq_range: Bandpass filter range for envelope.

    Returns:
        Dict compatible with BearingFaultCheckResult.
    """
    env = compute_envelope_spectrum(
        signal, fs, frequency_range=envelope_freq_range, num_peaks=50
    )

    peaks = env["top_peaks"]
    tolerance = expected_freq * tolerance_pct / 100.0

    # Search for fundamental
    detected = False
    detected_freq = None
    detected_mag = None
    deviation = None

    for p in peaks:
        freq = p["frequency_hz"]
        if abs(freq - expected_freq) <= tolerance:
            detected = True
            detected_freq = freq
            detected_mag = p["magnitude"]
            deviation = (freq - expected_freq) / expected_freq * 100.0
            break

    # Check harmonics
    harmonics = []
    for h in range(2, num_harmonics + 2):
        harmonic_freq = expected_freq * h
        harmonic_tol = harmonic_freq * tolerance_pct / 100.0
        for p in peaks:
            if abs(p["frequency_hz"] - harmonic_freq) <= harmonic_tol:
                harmonics.append({
                    "harmonic": h,
                    "expected_hz": round(harmonic_freq, 2),
                    "detected_hz": round(p["frequency_hz"], 2),
                    "magnitude": round(p["magnitude"], 6),
                })
                break

    # Evidence-strength rating from detected fundamental + harmonics.
    if detected and len(harmonics) >= 2:
        evidence_strength = "high"
    elif detected and len(harmonics) >= 1:
        evidence_strength = "moderate"
    elif detected:
        evidence_strength = "moderate"
    elif len(harmonics) > 0:
        evidence_strength = "low"
    else:
        evidence_strength = "none"

    return {
        "signal_id": signal_id,
        "bearing_id": bearing_id,
        "fault_type": fault_type,
        "fault_type_canonical": FAULT_TYPE_CANONICAL.get(fault_type.upper()),
        "expected_frequency_hz": round(expected_freq, 2),
        "detected": detected,
        "detected_frequency_hz": round(detected_freq, 2) if detected_freq else None,
        "magnitude": round(detected_mag, 6) if detected_mag else None,
        "deviation_pct": round(deviation, 2) if deviation is not None else None,
        "harmonics_detected": harmonics,
        "evidence_strength": evidence_strength,
    }


def check_all_bearing_faults(
    signal: np.ndarray,
    fs: float,
    bearing_id: str,
    rpm: float,
    signal_id: str = "",
    tolerance_pct: float = 5.0,
    envelope_freq_range: tuple[float, float] = (500, 5000),
) -> dict:
    """Run all four fault checks (BPFO, BPFI, BSF, FTF) for a bearing.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        bearing_id: Bearing designation (e.g. "6205").
        rpm: Shaft speed in RPM.
        signal_id: Signal identifier.
        tolerance_pct: Frequency tolerance in percent.
        envelope_freq_range: Bandpass filter range for envelope.

    Returns:
        Dict compatible with BearingFaultsSummary.

    Raises:
        ValueError: If bearing not found in catalog.
    """
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

    checks = []
    for ft, expected in fault_types.items():
        result = check_bearing_fault_peak(
            signal=signal,
            fs=fs,
            expected_freq=expected,
            fault_type=ft,
            bearing_id=bearing_id,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
            envelope_freq_range=envelope_freq_range,
        )
        checks.append(result)

    most_likely, assessment = _summarize_fault_checks(checks)

    bearing_freqs = {k: v for k, v in fault_types.items()}
    bearing_freqs["shaft_freq_hz"] = shaft_freq

    return {
        "signal_id": signal_id,
        "bearing_id": bearing_id,
        "rpm": rpm,
        "shaft_frequency_hz": shaft_freq,
        "bearing_frequencies": bearing_freqs,
        "fault_checks": checks,
        "overall_assessment": assessment,
        "most_likely_fault": most_likely,
        "most_likely_fault_canonical": (
            FAULT_TYPE_CANONICAL.get(most_likely) if most_likely else None
        ),
        # Catalog traceability: every catalog entry carries a mandatory
        # source citation — echo it so results stay auditable.
        "source": (freq_data.get("bearing_info") or {}).get("source"),
    }


def _summarize_fault_checks(checks: list[dict]) -> tuple[Optional[str], str]:
    """Rank detected checks by evidence strength -> (most_likely, assessment)."""
    evidence_order = {"high": 3, "moderate": 2, "low": 1, "none": 0}
    detected_faults = [
        c for c in checks if c["detected"] and c["evidence_strength"] != "none"
    ]
    detected_faults.sort(
        key=lambda c: evidence_order.get(c["evidence_strength"], 0), reverse=True
    )

    most_likely = detected_faults[0]["fault_type"] if detected_faults else None

    if not detected_faults:
        assessment = (
            "No bearing fault frequencies detected within tolerance. "
            "Bearing appears healthy based on envelope spectrum analysis."
        )
    else:
        fault_list = ", ".join(
            f"{c['fault_type']} ({c['evidence_strength']} evidence)"
            for c in detected_faults
        )
        assessment = (
            f"Bearing fault frequencies detected: {fault_list}. "
            f"Most likely fault: {most_likely}."
        )
    return most_likely, assessment


def check_frequency_set(
    signal: np.ndarray,
    fs: float,
    frequencies: dict[str, float],
    rpm: float,
    signal_id: str = "",
    tolerance_pct: float = 5.0,
    envelope_freq_range: tuple[float, float] = (500, 5000),
) -> dict:
    """Check an arbitrary set of labeled expected frequencies in the envelope.

    Covers the out-of-catalog and gearbox paths: labels may be the bearing
    acronyms (BPFO/BPFI/BSF/FTF — mapped to the canonical fault vocabulary)
    or any user label such as "GMF" (no canonical mapping).

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        frequencies: {label: expected frequency in Hz}, all values > 0.
        rpm: Shaft speed in RPM (echoed; shaft frequency = rpm/60).
        signal_id: Signal identifier (for result labeling).
        tolerance_pct: Frequency tolerance in percent.
        envelope_freq_range: Bandpass filter range for envelope.

    Returns:
        Dict compatible with BearingFaultsSummary (bearing_id/source None).

    Raises:
        ValueError: If frequencies is empty or contains non-positive values.
    """
    if not frequencies:
        raise ValueError(
            "frequencies is empty — pass at least one {label: hz} entry, "
            "e.g. {'BPFO': 107.4} or {'GMF': 350.0}."
        )
    bad = {k: v for k, v in frequencies.items() if not v or v <= 0}
    if bad:
        raise ValueError(
            f"frequencies contains non-positive values: {bad} — every "
            f"expected frequency must be > 0 Hz."
        )

    checks = [
        check_bearing_fault_peak(
            signal=signal,
            fs=fs,
            expected_freq=float(freq),
            fault_type=label,
            bearing_id="",
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
            envelope_freq_range=envelope_freq_range,
        )
        for label, freq in frequencies.items()
    ]
    most_likely, assessment = _summarize_fault_checks(checks)

    freq_echo = {k: float(v) for k, v in frequencies.items()}
    freq_echo["shaft_freq_hz"] = rpm / 60.0

    return {
        "signal_id": signal_id,
        "bearing_id": None,
        "rpm": rpm,
        "shaft_frequency_hz": rpm / 60.0,
        "bearing_frequencies": freq_echo,
        "fault_checks": checks,
        "overall_assessment": assessment,
        "most_likely_fault": most_likely,
        "most_likely_fault_canonical": (
            FAULT_TYPE_CANONICAL.get(most_likely) if most_likely else None
        ),
        "source": None,
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
    """
    return check_all_bearing_faults(
        signal=signal,
        fs=fs,
        bearing_id=bearing_type,
        rpm=rpm,
        signal_id=signal_id,
        tolerance_pct=tolerance_pct,
    )
