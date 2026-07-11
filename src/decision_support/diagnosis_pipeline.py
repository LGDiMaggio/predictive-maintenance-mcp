"""
Integrated vibration diagnosis pipeline.

Combines FFT, PSD, STFT, bearing fault detection, ISO severity,
and anomaly detection (OneClassSVM) into a single coherent diagnostic
result with an evidence-strength rating and actionable recommendations.

The ``evidence_strength`` field is categorical (none/weak/moderate/strong)
and derived from the number and quality of independent corroborating
findings — never from severity alone and never presented as a probability.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

from ..config import MODELS_DIR
from ..path_safety import resolve_model_paths
from ..signal_processing.spectral import compute_psd, compute_stft_spectrogram
from ..signal_processing.features import extract_time_domain_features as _extract_time_domain_features
from ..diagnostics.bearing_analyzer import check_all_bearing_faults
from ..diagnostics.iso10816 import assess_vibration_severity

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


def _run_anomaly_detection(
    signal: np.ndarray,
    fs: float,
    model_name: str = "bearing_health_model",
) -> Optional[dict]:
    """Run anomaly detection using a pre-trained OneClassSVM model.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        model_name: Name prefix for model/scaler/pca/metadata files.

    Returns:
        Dict with anomaly_ratio, overall_health, anomaly_score, num_segments,
        or None if model not found or prediction fails.
    """
    try:
        # Defense in depth: model_name is currently a constant, but route it
        # through the same containment helper so a future user-controlled caller
        # stays safe. Kept inside the try so an invalid name degrades to None
        # (this function's contract) instead of aborting the whole diagnosis.
        _model_paths = resolve_model_paths(MODELS_DIR, model_name)
        model_path = _model_paths.model

        if not model_path.exists():
            logger.info(f"Anomaly model not found at {model_path}, skipping.")
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(_model_paths.scaler, "rb") as f:
            scaler = pickle.load(f)
        with open(_model_paths.pca, "rb") as f:
            pca = pickle.load(f)
        with open(_model_paths.metadata, "r") as f:
            meta = json.load(f)

        # Use model's training parameters for segmentation
        segment_duration = meta.get("segment_duration", 0.1)
        overlap_ratio = meta.get("overlap_ratio", 0.5)
        segment_samples = int(segment_duration * fs)
        hop = int(segment_samples * (1 - overlap_ratio))

        if segment_samples > len(signal):
            segment_samples = len(signal)
            hop = segment_samples

        # Extract features from segments
        features_list = []
        for start in range(0, len(signal) - segment_samples + 1, hop):
            seg = signal[start : start + segment_samples]
            features_list.append(_extract_time_domain_features(seg))

        if not features_list:
            return None

        X = pd.DataFrame(features_list).values
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        predictions = model.predict(X_pca)
        anomaly_count = int(np.sum(predictions == -1))
        anomaly_ratio = float(anomaly_count / len(predictions))

        # Anomaly score (distance from decision boundary)
        anomaly_score = None
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_pca)
            anomaly_score = float(np.mean(scores))

        if anomaly_ratio < 0.1:
            health = "Healthy"
        elif anomaly_ratio < 0.3:
            health = "Suspicious"
        else:
            health = "Faulty"

        return {
            "anomaly_ratio": round(anomaly_ratio, 4),
            "anomaly_count": anomaly_count,
            "num_segments": len(predictions),
            "overall_health": health,
            "anomaly_score": round(anomaly_score, 4) if anomaly_score is not None else None,
        }

    except Exception as e:
        logger.warning(f"Anomaly detection failed: {e}")
        return None


def diagnose_vibration(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    signal_id: str = "",
    bearing_id: Optional[str] = None,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
    signal_unit: str = "g",
    anomaly_model_name: str = "bearing_health_model",
) -> dict:
    """Integrated vibration diagnosis pipeline.

    Runs:
    1. FFT analysis
    2. PSD (Welch method)
    3. STFT spectrogram (time-varying behavior)
    4. Bearing fault detection (if bearing_id provided)
    5. ISO 20816-3 severity assessment (zone boundaries from
       ISO 10816-3:2009)
    6. Anomaly detection (OneClassSVM, if model available)
    7. Synthesis into overall diagnosis

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        rpm: Machine operating speed (RPM).
        signal_id: Signal identifier for result labeling.
        bearing_id: Bearing designation for fault detection (optional).
        machine_group: ISO 20816-3 machine group — 1 (large, >300 kW) or
            2 (medium, 15-300 kW).
        support_type: Support type — 'rigid' or 'flexible'.
        signal_unit: Signal unit ('g', 'm/s²', 'mm/s').
        anomaly_model_name: Name of trained anomaly model (default: 'bearing_health_model').

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
        machine_group=machine_group,
        support_type=support_type,
        signal_unit=signal_unit,
        operating_speed_rpm=rpm,
    )
    iso_result["signal_id"] = signal_id

    # 6. Anomaly detection
    anomaly_result = _run_anomaly_detection(signal, fs, model_name=anomaly_model_name)

    # 7. Synthesis
    diagnosis_text, evidence_strength, recommendations = _synthesize_diagnosis(
        fft_summary=fft_summary,
        psd_summary=psd_summary,
        stft_summary=stft_summary,
        bearing_faults=bearing_faults,
        iso_severity=iso_result,
        anomaly=anomaly_result,
        rpm=rpm,
    )

    return {
        "signal_id": signal_id,
        "rpm": rpm,
        "bearing_id": bearing_id,
        "machine_group": machine_group,
        "support_type": support_type,
        "fft_summary": fft_summary,
        "psd_summary": psd_summary,
        "stft_summary": stft_summary,
        "bearing_faults": bearing_faults,
        "iso_severity": iso_result,
        "anomaly_detection": anomaly_result,
        "overall_diagnosis": diagnosis_text,
        "evidence_strength": evidence_strength,
        "recommendations": recommendations,
    }


def _synthesize_diagnosis(
    fft_summary: dict,
    psd_summary: dict,
    stft_summary: dict,
    bearing_faults: Optional[dict],
    iso_severity: dict,
    anomaly: Optional[dict],
    rpm: float,
) -> tuple[str, str, list[str]]:
    """Combine analysis results into diagnosis, evidence strength, and recommendations.

    The evidence strength is categorical (none/weak/moderate/strong) and
    accumulates points from independent corroborating findings:

    - ISO severity: zone C +1.0, zone D +2.0 (elevated broadband vibration
      is direct physical evidence of a problem, but on its own it is not
      corroborated).
    - Shaft signature: dominant peak at 1x or 2x shaft speed +1.0.
    - Bearing fault frequencies: best detected check contributes
      high +2.0 / moderate +1.0 / low +0.5.
    - Anomaly model: Faulty +1.0, Suspicious +0.5.

    Mapping: 0 -> "none", <2 -> "weak", <3 -> "moderate", >=3 -> "strong".
    A quiet machine with no findings therefore gets "none" — severity alone
    can never produce "strong".
    """
    lines = []
    recommendations = []
    evidence_points = 0.0

    # ISO severity assessment
    zone = iso_severity["zone"]
    rms_vel = iso_severity["rms_velocity_mm_s"]

    if zone == "A":
        lines.append(f"ISO Severity: Zone A (Good) — RMS velocity {rms_vel:.2f} mm/s.")
    elif zone == "B":
        lines.append(f"ISO Severity: Zone B (Acceptable) — RMS velocity {rms_vel:.2f} mm/s.")
    elif zone == "C":
        lines.append(f"ISO Severity: Zone C (Unsatisfactory) — RMS velocity {rms_vel:.2f} mm/s.")
        evidence_points += 1.0
        recommendations.append("Schedule maintenance within next planned shutdown.")
    else:
        lines.append(f"ISO Severity: Zone D (Unacceptable) — RMS velocity {rms_vel:.2f} mm/s.")
        evidence_points += 2.0
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
            evidence_points += 1.0
            recommendations.append("Check rotor balance.")
        elif 1.9 < ratio < 2.1:
            lines.append("Dominant peak at 2x shaft speed — possible misalignment.")
            evidence_points += 1.0
            recommendations.append("Check alignment and coupling condition.")

    # Bearing fault analysis
    if bearing_faults:
        most_likely = bearing_faults.get("most_likely_fault")
        if most_likely:
            detected = [
                c for c in bearing_faults["fault_checks"]
                if c["detected"] and c["evidence_strength"] != "none"
            ]
            fault_text = ", ".join(
                f"{c['fault_type']} ({c['evidence_strength']} evidence)"
                for c in detected
            )
            lines.append(f"Bearing faults detected: {fault_text}.")
            lines.append(f"Most likely: {most_likely}.")

            strengths = {c["evidence_strength"] for c in detected}
            if "high" in strengths:
                evidence_points += 2.0
                recommendations.append(
                    f"Bearing {bearing_faults['bearing_id']}: "
                    f"plan replacement at next opportunity."
                )
            else:
                evidence_points += 1.0 if "moderate" in strengths else 0.5
                recommendations.append(
                    f"Bearing {bearing_faults['bearing_id']}: "
                    f"monitor closely, confirm with additional measurements."
                )
        else:
            lines.append("No bearing fault frequencies detected — bearing appears healthy.")

    # Anomaly detection
    if anomaly:
        health = anomaly["overall_health"]
        ratio = anomaly["anomaly_ratio"]
        lines.append(
            f"Anomaly detection: {health} "
            f"({ratio * 100:.1f}% anomalous segments)."
        )
        if health == "Faulty":
            evidence_points += 1.0
            recommendations.append(
                "Anomaly model flagged signal as faulty — investigate root cause."
            )
        elif health == "Suspicious":
            evidence_points += 0.5
            recommendations.append(
                "Anomaly model detected suspicious patterns — increase monitoring frequency."
            )

    # Overall evidence strength — from corroborating findings, NOT severity.
    if evidence_points == 0.0:
        evidence_strength = "none"
    elif evidence_points < 2.0:
        evidence_strength = "weak"
    elif evidence_points < 3.0:
        evidence_strength = "moderate"
    else:
        evidence_strength = "strong"

    lines.append(f"Fault evidence strength: {evidence_strength}.")

    # Default recommendation if none generated
    if not recommendations:
        recommendations.append("Continue routine monitoring per maintenance schedule.")

    diagnosis_text = " ".join(lines)

    return diagnosis_text, evidence_strength, recommendations
