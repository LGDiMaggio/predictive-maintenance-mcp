"""
Integrated vibration diagnosis pipeline.

Combines FFT, PSD, STFT, bearing fault detection, ISO severity,
and anomaly detection (OneClassSVM) into a single coherent diagnostic
result with confidence scoring and actionable recommendations.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy

from .config import MODELS_DIR
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


def _extract_time_domain_features(segment: np.ndarray) -> dict[str, float]:
    """Extract 17 time-domain features from a signal segment.

    Mirrors extract_time_domain_features in machinery_diagnostics_server.py
    to ensure compatibility with trained anomaly models.
    """
    mean_val = float(np.mean(segment))
    std_val = float(np.std(segment))
    var_val = float(np.var(segment))
    mean_abs_val = float(np.mean(np.abs(segment)))
    rms_val = float(np.sqrt(np.mean(segment**2)))
    max_val = float(np.max(segment))
    min_val = float(np.min(segment))
    range_val = max_val - min_val
    skewness_val = float(skew(segment))
    kurtosis_val = float(kurtosis(segment, fisher=True))

    if rms_val == 0:
        rms_val = 1e-10
    if mean_abs_val == 0:
        mean_abs_val = 1e-10

    shape_factor_val = rms_val / mean_abs_val
    crest_factor_val = float(np.max(np.abs(segment))) / rms_val
    impulse_factor_val = float(np.max(np.abs(segment))) / mean_abs_val
    sqrt_mean = float(np.mean(np.sqrt(np.abs(segment))))
    clearance_factor_val = float(np.max(np.abs(segment))) / (sqrt_mean**2) if sqrt_mean > 0 else 0.0
    power_val = float(np.mean(segment**2))
    hist, _ = np.histogram(segment, bins=50, density=True)
    hist = hist + 1e-10
    entropy_val = float(entropy(hist))
    zero_crossings = np.where(np.diff(np.sign(segment)))[0]
    zero_crossing_rate_val = float(len(zero_crossings) / len(segment))

    return {
        "mean": mean_val, "std": std_val, "var": var_val,
        "mean_abs": mean_abs_val, "rms": rms_val,
        "max": max_val, "min": min_val, "range": range_val,
        "skewness": skewness_val, "kurtosis": kurtosis_val,
        "shape_factor": shape_factor_val, "crest_factor": crest_factor_val,
        "impulse_factor": impulse_factor_val, "clearance_factor": clearance_factor_val,
        "power": power_val, "entropy": entropy_val,
        "zero_crossing_rate": zero_crossing_rate_val,
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
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    pca_path = MODELS_DIR / f"{model_name}_pca.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    if not model_path.exists():
        logger.info(f"Anomaly model not found at {model_path}, skipping.")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(metadata_path, "r") as f:
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
    machine_class: str = "II",
    signal_unit: str = "g",
    anomaly_model_name: str = "bearing_health_model",
) -> dict:
    """Integrated vibration diagnosis pipeline.

    Runs:
    1. FFT analysis
    2. PSD (Welch method)
    3. STFT spectrogram (time-varying behavior)
    4. Bearing fault detection (if bearing_id provided)
    5. ISO 10816/20816 severity assessment
    6. Anomaly detection (OneClassSVM, if model available)
    7. Synthesis into overall diagnosis

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        rpm: Machine operating speed (RPM).
        signal_id: Signal identifier for result labeling.
        bearing_id: Bearing designation for fault detection (optional).
        machine_class: ISO machine class ('I', 'II', 'III', 'IV').
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
        machine_class=machine_class,
        signal_unit=signal_unit,
        operating_speed_rpm=rpm,
    )
    iso_result["signal_id"] = signal_id

    # 6. Anomaly detection
    anomaly_result = _run_anomaly_detection(signal, fs, model_name=anomaly_model_name)

    # 7. Synthesis
    diagnosis_text, confidence, recommendations = _synthesize_diagnosis(
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
        "machine_class": machine_class,
        "fft_summary": fft_summary,
        "psd_summary": psd_summary,
        "stft_summary": stft_summary,
        "bearing_faults": bearing_faults,
        "iso_severity": iso_result,
        "anomaly_detection": anomaly_result,
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
    anomaly: Optional[dict],
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

    # Anomaly detection
    if anomaly:
        health = anomaly["overall_health"]
        ratio = anomaly["anomaly_ratio"]
        lines.append(
            f"Anomaly detection: {health} "
            f"({ratio * 100:.1f}% anomalous segments)."
        )
        if health == "Faulty":
            severity_score = max(severity_score, 2)
            recommendations.append(
                "Anomaly model flagged signal as faulty — investigate root cause."
            )
        elif health == "Suspicious":
            severity_score = max(severity_score, 1)
            recommendations.append(
                "Anomaly model detected suspicious patterns — increase monitoring frequency."
            )

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
