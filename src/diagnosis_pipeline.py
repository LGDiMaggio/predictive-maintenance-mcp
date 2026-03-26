"""
Integrated vibration diagnosis pipeline.

Combines FFT, PSD, STFT, bearing fault detection, and ISO severity
into a single coherent diagnostic result with confidence scoring
and actionable recommendations.
"""

import logging
from typing import Any, Optional

import numpy as np
from scipy.fft import fft, fftfreq

from .spectral import compute_psd, compute_stft_spectrogram
from .bearing_analyzer import check_all_bearing_faults
from .iso10816 import assess_vibration_severity

logger = logging.getLogger(__name__)


def _compute_fft_summary(
    signal: np.ndarray, fs: float, num_peaks: int = 10
) -> dict[str, Any]:
    """Compute FFT and return compact summary."""
    N = len(signal)
    window = np.hamming(N)
    windowed = signal * window

    fft_vals = fft(windowed)
    freqs = fftfreq(N, 1 / fs)

    pos = freqs > 0
    freqs = freqs[pos]
    mags = 2.0 * np.abs(fft_vals[pos]) / N

    peak_idx = int(np.argmax(mags))
    peak_freq = float(freqs[peak_idx])
    peak_mag = float(mags[peak_idx])
    rms_spectral = float(np.sqrt(np.mean(mags**2)))

    # Top peaks
    order = np.argsort(mags)[::-1][:num_peaks]
    top_peaks = [
        {"frequency_hz": round(float(freqs[i]), 2), "magnitude": round(float(mags[i]), 6)}
        for i in np.sort(order)
    ]

    return {
        "peak_frequency_hz": round(peak_freq, 2),
        "peak_magnitude": round(peak_mag, 6),
        "rms_spectral": round(rms_spectral, 6),
        "num_bins": len(freqs),
        "top_peaks": top_peaks,
    }


def diagnose_vibration(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    signal_id: str = "",
    bearing_id: Optional[str] = None,
    machine_class: str = "II",
    signal_unit: str = "g",
) -> dict:
    """Integrated vibration diagnosis pipeline.

    Runs:
    1. FFT analysis
    2. PSD (Welch method)
    3. STFT spectrogram (time-varying behavior)
    4. Bearing fault detection (if bearing_id provided)
    5. ISO 10816/20816 severity assessment
    6. Synthesis into overall diagnosis

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        rpm: Machine operating speed (RPM).
        signal_id: Signal identifier for result labeling.
        bearing_id: Bearing designation for fault detection (optional).
        machine_class: ISO machine class ('I', 'II', 'III', 'IV').
        signal_unit: Signal unit ('g', 'm/s²', 'mm/s').

    Returns:
        Dict compatible with DiagnosisResult.
    """
    # 1. FFT
    fft_summary = _compute_fft_summary(signal, fs)

    # 2. PSD
    psd_result = compute_psd(signal, fs, nperseg=min(1024, len(signal)))
    psd_summary = {
        "total_power": psd_result["total_power"],
        "top_peaks": psd_result["top_peaks"][:5],
        "freq_resolution": psd_result["frequency_resolution"],
    }

    # 3. STFT
    stft_result = compute_stft_spectrogram(signal, fs, nperseg=min(512, len(signal)))
    stft_summary = {
        "max_power_freq_hz": stft_result["max_power_freq_hz"],
        "max_power_time_s": stft_result["max_power_time_s"],
        "energy_per_band": stft_result["energy_per_band"],
        "num_time_bins": stft_result["num_time_bins"],
    }

    # 4. Bearing faults
    bearing_faults = None
    if bearing_id:
        try:
            bearing_faults = check_all_bearing_faults(
                signal=signal,
                fs=fs,
                bearing_id=bearing_id,
                rpm=rpm,
                signal_id=signal_id,
            )
        except ValueError as e:
            logger.warning(f"Bearing analysis skipped: {e}")
            bearing_faults = None

    # 5. ISO severity
    iso_result = assess_vibration_severity(
        signal=signal,
        fs=fs,
        machine_class=machine_class,
        signal_unit=signal_unit,
        operating_speed_rpm=rpm,
    )
    iso_result["signal_id"] = signal_id

    # 6. Synthesis
    diagnosis_text, confidence, recommendations = _synthesize_diagnosis(
        fft_summary=fft_summary,
        psd_summary=psd_summary,
        stft_summary=stft_summary,
        bearing_faults=bearing_faults,
        iso_severity=iso_result,
        rpm=rpm,
    )

    return {
        "signal_id": signal_id,
        "rpm": rpm,
        "bearing_id": bearing_id,
        "machine_class": machine_class,
        "fft_summary": fft_summary,
        "psd_summary": psd_summary,
        "stft_summary": stft_summary,
        "bearing_faults": bearing_faults,
        "iso_severity": iso_result,
        "overall_diagnosis": diagnosis_text,
        "confidence": confidence,
        "recommendations": recommendations,
    }


def _synthesize_diagnosis(
    fft_summary: dict,
    psd_summary: dict,
    stft_summary: dict,
    bearing_faults: Optional[dict],
    iso_severity: dict,
    rpm: float,
) -> tuple[str, str, list[str]]:
    """Combine analysis results into diagnosis, confidence, and recommendations."""
    lines = []
    recommendations = []
    severity_score = 0  # 0=good, 1=watch, 2=action, 3=critical

    # ISO severity assessment
    zone = iso_severity["zone"]
    rms_vel = iso_severity["rms_velocity_mm_s"]

    if zone == "A":
        lines.append(f"ISO Severity: Zone A (Good) — RMS velocity {rms_vel:.2f} mm/s.")
    elif zone == "B":
        lines.append(f"ISO Severity: Zone B (Acceptable) — RMS velocity {rms_vel:.2f} mm/s.")
        severity_score = max(severity_score, 1)
    elif zone == "C":
        lines.append(f"ISO Severity: Zone C (Unsatisfactory) — RMS velocity {rms_vel:.2f} mm/s.")
        severity_score = max(severity_score, 2)
        recommendations.append("Schedule maintenance within next planned shutdown.")
    else:
        lines.append(f"ISO Severity: Zone D (Unacceptable) — RMS velocity {rms_vel:.2f} mm/s.")
        severity_score = max(severity_score, 3)
        recommendations.append("IMMEDIATE ACTION: Stop machine and inspect.")

    # FFT dominant frequency
    peak_f = fft_summary["peak_frequency_hz"]
    shaft_freq = rpm / 60.0
    lines.append(f"Dominant frequency: {peak_f:.1f} Hz (shaft: {shaft_freq:.1f} Hz).")

    # Check for shaft-related frequencies
    if shaft_freq > 0:
        ratio = peak_f / shaft_freq
        if 0.9 < ratio < 1.1:
            lines.append("Dominant peak at 1x shaft speed — possible unbalance.")
            severity_score = max(severity_score, 1)
            recommendations.append("Check rotor balance.")
        elif 1.9 < ratio < 2.1:
            lines.append("Dominant peak at 2x shaft speed — possible misalignment.")
            severity_score = max(severity_score, 1)
            recommendations.append("Check alignment and coupling condition.")

    # Bearing fault analysis
    if bearing_faults:
        most_likely = bearing_faults.get("most_likely_fault")
        if most_likely:
            detected = [
                c for c in bearing_faults["fault_checks"]
                if c["detected"] and c["confidence"] != "none"
            ]
            fault_text = ", ".join(
                f"{c['fault_type']} ({c['confidence']})" for c in detected
            )
            lines.append(f"Bearing faults detected: {fault_text}.")
            lines.append(f"Most likely: {most_likely}.")

            # Severity based on confidence
            high_conf = any(c["confidence"] == "high" for c in detected)
            if high_conf:
                severity_score = max(severity_score, 2)
                recommendations.append(
                    f"Bearing {bearing_faults['bearing_id']}: "
                    f"plan replacement at next opportunity."
                )
            else:
                severity_score = max(severity_score, 1)
                recommendations.append(
                    f"Bearing {bearing_faults['bearing_id']}: "
                    f"monitor closely, confirm with additional measurements."
                )
        else:
            lines.append("No bearing fault frequencies detected — bearing appears healthy.")

    # Overall confidence
    if severity_score >= 2:
        confidence = "high"
    elif severity_score == 1:
        confidence = "moderate"
    else:
        confidence = "high"

    # Default recommendation if none generated
    if not recommendations:
        recommendations.append("Continue routine monitoring per maintenance schedule.")

    diagnosis_text = " ".join(lines)

    return diagnosis_text, confidence, recommendations
