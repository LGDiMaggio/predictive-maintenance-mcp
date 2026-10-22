"""
Integrated vibration diagnosis pipeline.

Combines FFT, PSD, STFT, bearing fault detection, ISO severity,
and anomaly detection (OneClassSVM) into a single coherent diagnostic
result with an evidence-strength rating and actionable recommendations.

The ``evidence_strength`` field is categorical (none/weak/moderate/strong)
and counts the findings of the analyses that ran, a bearing finding once. It
never rests on severity alone, is never presented as a probability, and never
names or confirms a fault type.
"""

import json
import logging
import pickle
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from ..path_safety import resolve_model_paths
from ..signal_processing.spectral import (
    amplitude_spectrum,
    compute_psd,
    compute_stft_spectrogram,
    resolve_envelope_band,
)
from ..signal_processing.features import (
    extract_time_domain_features as _extract_time_domain_features,
)
from ..signal_acquisition.repository import VALID_SIGNAL_UNITS, normalize_signal_unit
from ..signal_validation import (
    InvalidInputError,
    validate_rpm,
    validate_signal_and_fs,
)
from ..diagnostics.bearing_analyzer import check_all_bearing_faults
from ..diagnostics.bearing_catalog import lookup_bearing
from ..diagnostics.bearing_claims import (
    UNKNOWN,
    aggregate_statement,
    claim_actions,
    counts_as_finding,
    pipeline_statement,
)
from ..diagnostics.iso20816 import assess_severity_with_axis

logger = logging.getLogger(__name__)


def _refused_iso_block(signal_id: str, reason: str, remedy: str) -> dict:
    """Build the schema-level refused ISO severity block.

    Compatible with :class:`~predictive_maintenance_mcp.models.ISOSeverityRefusal`.
    The refusal lives in the result SCHEMA (status/reason/remedy fields), not
    in log prose an LLM client can lose.
    """
    return {
        "status": "refused",
        "signal_id": signal_id,
        "reason": reason,
        "remedy": remedy,
    }


def _bearing_check(
    status: str,
    reason_code: Optional[str] = None,
    reason: Optional[str] = None,
    remedy: Optional[str] = None,
) -> dict:
    """Build the ``bearing_check`` block (``models.BearingCheckOutcome``)."""
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "remedy": remedy,
    }


def _not_in_catalog_check(bearing_id: str, rpm: float, signal_id: str) -> dict:
    """The check cannot run: the catalog has no geometry for *bearing_id*."""
    call = f"check_bearing_faults(signal_id={signal_id!r}, rpm={rpm:.10g}"
    return _bearing_check(
        "not_computed",
        reason_code="not_in_catalog",
        reason=(
            f"Bearing {bearing_id!r} is not in the bearing catalog, so its "
            f"characteristic frequencies (BPFO, BPFI, BSF, FTF) could not be "
            f"computed."
        ),
        remedy=(
            f"The bearing check of this diagnosis runs only for a designation "
            f"in the bearing catalog. search_bearing_catalog(bearing_id="
            f"{bearing_id!r}) lists the designations the catalog holds, in "
            f"case the bearing is there under another one. Otherwise check "
            f"the bearing with {call}, num_balls=..., ball_diameter_mm=..., "
            f"pitch_diameter_mm=..., contact_angle_deg=...), taking the "
            f"geometry from the bearing manufacturer's data and never "
            f"estimating it, or with {call}, frequencies={{'BPFO': ..., "
            f"'BPFI': ..., 'BSF': ..., 'FTF': ...}}) giving its characteristic "
            f"frequencies in Hz."
        ),
    )


def _empty_envelope_band_check(bearing_id: str, fs: float) -> dict:
    """The check cannot run: its default envelope band is empty at *fs*.

    The band is 500 Hz up to just below Nyquist (``resolve_envelope_band``),
    so it is empty at fs <= 1002 Hz and reaches 500-5000 Hz only above
    10 kHz (10002 Hz): the numbers the remedy names, pinned by the tests.
    """
    return _bearing_check(
        "not_computed",
        reason_code="envelope_band_unusable",
        reason=(
            f"At fs={fs:g} Hz the envelope band of the bearing check, from "
            f"500 Hz to just below the Nyquist frequency ({fs / 2:g} Hz), "
            f"is empty, so no envelope spectrum could be computed for "
            f"bearing {bearing_id!r}."
        ),
        remedy=(
            "Re-acquire the signal at a higher sampling rate, then re-run the "
            "diagnosis. The envelope band of the bearing check runs from "
            "500 Hz to just below half the sampling rate: it is empty at "
            "1002 Hz or less, and it reaches its full width, 500 to 5000 Hz, "
            "only above 10 kHz."
        ),
    )


#: Fewest samples the bearing check's envelope band-pass filter runs on:
#: ``sosfiltfilt`` pads 27 samples (scipy's default padlen for the 4th-order
#: band) and needs a longer signal. Pinned against scipy by the tests.
_ENVELOPE_FILTER_MIN_SAMPLES = 28


def _refuse_signal_too_short_for_the_bearing_check(
    signal: np.ndarray, fs: float
) -> None:
    """Refuse, with the project's own message, a signal the filter cannot take.

    The filter's minimum only lets it run; the message says so, because a
    record that just meets it resolves the envelope spectrum far too coarsely
    for a meaningful check.

    Raises:
        InvalidInputError: If the signal has fewer samples than the filter needs.
    """
    n = len(signal)
    if n >= _ENVELOPE_FILTER_MIN_SAMPLES:
        return
    duration_s = n / fs
    raise InvalidInputError(
        f"The bearing check cannot run on this signal: its envelope band-pass "
        f"filter runs forward and backward and needs at least "
        f"{_ENVELOPE_FILTER_MIN_SAMPLES} samples, and the signal has {n}. "
        f"Meeting that minimum only lets the filter run; it does not make the "
        f"check meaningful. The envelope spectrum resolves frequencies only to "
        f"1 / duration ({1 / duration_s:.4g} Hz for this {duration_s:.4g} s "
        f"signal), while the characteristic bearing frequencies depend on "
        f"shaft speed and bearing geometry, so a meaningful check needs a "
        f"record many shaft revolutions long. Load or re-acquire a "
        f"longer record of the measurement, then re-run the diagnosis."
    )


def _unrunnable_bearing_check(
    bearing_id: str, fs: float, rpm: float, signal_id: str
) -> Optional[dict]:
    """The not-computed block when the check cannot run, else None.

    The sampling rate comes first: an empty envelope band blocks every route
    of the check, the geometry and frequency routes the catalog remedy names
    included, so re-acquiring is the first thing to fix.
    """
    try:
        resolve_envelope_band(fs)
    except ValueError:
        return _empty_envelope_band_check(bearing_id, fs)
    if lookup_bearing(bearing_id) is None:
        return _not_in_catalog_check(bearing_id, rpm, signal_id)
    return None


def _compute_fft_summary(
    signal: np.ndarray, fs: float, num_peaks: int = 10
) -> dict[str, Any]:
    """Compute FFT and return compact summary.

    The spectrum is ``spectral.amplitude_spectrum`` (Hamming, ``2|X|/N``),
    the same core ``analyze_fft`` and the asset ledger's 1x amplitude use.
    """
    freqs, mags = amplitude_spectrum(signal, fs)

    peak_idx = int(np.argmax(mags))
    peak_freq = float(freqs[peak_idx])
    peak_mag = float(mags[peak_idx])
    rms_spectral = float(np.sqrt(np.mean(mags**2)))

    # Top peaks
    order = np.argsort(mags)[::-1][:num_peaks]
    top_peaks = [
        {
            "frequency_hz": round(float(freqs[i]), 2),
            "magnitude": round(float(mags[i]), 6),
        }
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
            "anomaly_score": (
                round(anomaly_score, 4) if anomaly_score is not None else None
            ),
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
    signal_unit: Optional[str] = None,
    anomaly_model_name: str = "bearing_health_model",
    unchecked_bearing: Optional[dict] = None,
    acquisition_eligibility: str = UNKNOWN,
) -> dict:
    """Integrated vibration diagnosis pipeline.

    Runs:
    1. FFT analysis
    2. PSD (Welch method)
    3. STFT spectrogram (time-varying behavior)
    4. Bearing fault detection (if bearing_id provided; ``bearing_check``
       says whether it ran)
    5. ISO 20816-3 severity assessment (zone boundaries from
       ISO 10816-3:2009)
    6. Anomaly detection (OneClassSVM, if model available)
    7. Synthesis into overall diagnosis

    The pipeline DEGRADES instead of failing when the ISO verdict cannot be
    produced honestly: with an undeclared signal unit, or a sampling rate
    that cannot cover the ISO evaluation band (Nyquist < 1 kHz), the
    ``iso_severity`` block becomes a structured refusal
    (``status: "refused"`` + ``reason`` + ``remedy``) while the spectral,
    bearing, and anomaly blocks still run. Units are never guessed from
    signal amplitude. The bearing check degrades the same way when it
    cannot run: a sampling rate that leaves the envelope band empty
    (fs <= 1002 Hz), a ``bearing_id`` the catalog lacks, or a caller's own
    ``unchecked_bearing``, gives ``bearing_faults`` None and a
    ``bearing_check`` with ``status: "not_computed"``, ``reason_code``,
    ``reason`` and ``remedy``, and the text and the recommendations say so.
    Corrupt input is the exception: a NaN or infinite sample, a non-finite
    or non-positive fs, or a shaft speed that is not a positive finite
    number invalidates every block, so the whole call raises before any
    block is computed. Anything else the bearing check refuses is raised
    too, never turned into a skipped bearing block: a fault frequency that
    rounds to zero at a fraction of 1 rpm, or a signal too short for the
    envelope filter (fewer than 28 samples). ``bearing_check`` is
    ``"not_requested"`` only without a ``bearing_id`` and without an
    ``unchecked_bearing``.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        rpm: Machine operating speed (RPM), positive and finite.
        signal_id: Signal identifier for result labeling.
        bearing_id: Bearing designation for fault detection (optional).
        machine_group: ISO 20816-3 machine group — 1 (large, >300 kW) or
            2 (medium, 15-300 kW).
        support_type: Support type — 'rigid' or 'flexible'.
        signal_unit: DECLARED signal unit — 'g', 'm/s2', 'mm/s', or 'm/s'
            (aliases like 'm/s²' are normalized). None means undeclared:
            the ISO severity block is refused with a remedy naming
            ``load_signal(signal_unit=...)``.
        anomaly_model_name: Name of trained anomaly model (default: 'bearing_health_model').
        unchecked_bearing: A ``not_computed`` ``bearing_check`` block the
            caller determined without a ``bearing_id`` (the MCP tool's point
            that declares fault orders only). Reported, synthesized and
            recommended like the pipeline's own; ignored when ``bearing_id``
            is given.
        acquisition_eligibility: "known_inapplicable" when the signal's own
            companion metadata declares a generated signal without noise
            (the bearing claim is then withheld), else "unknown"; passed to
            the bearing check.

    Returns:
        Dict compatible with DiagnosisResult — ``iso_severity`` is either an
        assessed block (``status: "assessed"``) or a refusal
        (``status: "refused"``); ``bearing_check`` has ``status``
        ``"assessed"``, ``"not_requested"`` or ``"not_computed"``.

    Raises:
        ValueError: If signal_unit is provided but is not a recognized unit
            (caller misuse — distinct from the honest "undeclared" refusal),
            if fs or rpm is not a positive finite number, if any sample is
            NaN or infinite, or if the bearing check refuses its input (an
            expected frequency computed from rpm that rounds to zero, a
            signal too short for the envelope filter: ``InvalidInputError``).
    """
    # Validate a provided unit up front; None stays None (undeclared).
    declared_unit: Optional[str] = None
    if signal_unit is not None:
        declared_unit = normalize_signal_unit(signal_unit)
        if declared_unit is None:
            raise ValueError(
                f"Invalid signal_unit '{signal_unit}' — declare one of "
                f"{list(VALID_SIGNAL_UNITS)} ('g'/'m/s2' for acceleration, "
                f"'mm/s'/'m/s' for velocity), or omit it to get a structured "
                f"ISO refusal instead of a verdict."
            )

    # Corrupt input is refused here, not per block: the ISO block below turns
    # its ValueError into a refused block, so without this check a NaN
    # sample would still yield a "diagnosis". The
    # speed is checked unconditionally: without a bearing or a declared unit
    # no block looks at it, yet the synthesis still reads its 1x/2x lines.
    signal = validate_signal_and_fs(signal, fs)
    validate_rpm(rpm)

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

    # 4. Bearing faults. A requested check runs, or bearing_check says why it
    # could not: bearing_faults=None alone reads as "no check requested".
    bearing_faults = None
    if not bearing_id:
        bearing_check = unchecked_bearing or _bearing_check("not_requested")
    elif (
        unrunnable := _unrunnable_bearing_check(bearing_id, fs, rpm, signal_id)
    ) is not None:
        bearing_check = unrunnable
        logger.warning(f"Bearing analysis not computed: {unrunnable['reason']}")
    else:
        # The conditions the check cannot run under are handled above, so
        # what it raises now refuses the caller's own input (a fault frequency
        # that rounds to zero at a fraction of 1 rpm, or a signal too short
        # for the envelope filter, refused just below in the project's own
        # words): raised, never skipped.
        _refuse_signal_too_short_for_the_bearing_check(signal, fs)
        bearing_faults = check_all_bearing_faults(
            signal=signal,
            fs=fs,
            bearing_id=bearing_id,
            rpm=rpm,
            signal_id=signal_id,
            acquisition_eligibility=acquisition_eligibility,
        )
        bearing_check = _bearing_check("assessed")

    # 5. ISO severity — assessed only on a DECLARED unit and a sufficient
    # sampling rate; otherwise a schema-level refusal (the other blocks
    # above/below still run).
    if declared_unit is None:
        iso_result = _refused_iso_block(
            signal_id=signal_id,
            reason=(
                "Signal unit not declared — ISO 20816-3 severity requires "
                "knowing whether the signal is acceleration ('g'/'m/s2') or "
                "velocity ('mm/s'/'m/s'); units are never guessed from "
                "amplitude."
            ),
            remedy=(
                "Re-load the signal with load_signal(filepath=..., "
                "signal_unit='g'|'m/s2'|'mm/s'|'m/s') or add a 'signal_unit' "
                "field to the companion _metadata.json, then re-run the "
                "diagnosis."
            ),
        )
    else:
        try:
            iso_result = assess_severity_with_axis(
                signal=signal,
                fs=fs,
                machine_group=machine_group,
                support_type=support_type,
                signal_unit=declared_unit,
                operating_speed_rpm=rpm,
            )
            iso_result["signal_id"] = signal_id
            iso_result["status"] = "assessed"
        except ValueError as e:
            # E.g. Nyquist below the 1 kHz ISO evaluation band (U2 refusal).
            iso_result = _refused_iso_block(
                signal_id=signal_id,
                reason=str(e),
                remedy=(
                    "Resolve the condition in 'reason' — e.g. re-acquire the "
                    "signal at fs >= 2106 Hz so the full ISO 20816-3 "
                    "evaluation band (up to 1 kHz) is covered — then re-run "
                    "the diagnosis. The spectral/bearing/anomaly blocks in "
                    "this result are unaffected."
                ),
            )

    # 6. Anomaly detection
    anomaly_result = _run_anomaly_detection(signal, fs, model_name=anomaly_model_name)

    # 7. Synthesis
    diagnosis_text, evidence_strength, recommendations = _synthesize_diagnosis(
        fft_summary=fft_summary,
        psd_summary=psd_summary,
        stft_summary=stft_summary,
        bearing_faults=bearing_faults,
        bearing_check=bearing_check,
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
        "bearing_check": bearing_check,
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
    bearing_check: dict,
    iso_severity: dict,
    anomaly: Optional[dict],
    rpm: float,
) -> tuple[str, str, list[str]]:
    """Combine analysis results into diagnosis, evidence strength, and recommendations.

    The evidence strength is categorical (none/weak/moderate/strong) and
    accumulates points from the findings of the analyses that ran:

    - ISO severity: zone C +1.0, zone D +2.0 (elevated broadband vibration
      is direct physical evidence of a problem, but on its own it is not
      corroborated). A refused ISO block (undeclared unit, insufficient
      sampling rate) contributes NO evidence — no verdict, no points.
    - Shaft signature: dominant peak at 1x or 2x shaft speed +1.0.
    - Bearing claim (:mod:`..diagnostics.bearing_claims`): one finding,
      +1.0, at single-spectrum evidence or qualified from an indicated or
      several-types outcome; a withheld claim and one held back as
      inconclusive add nothing. The bearing sentence and actions come from
      the same claim (confirmation, inspection of every candidate alike, a
      repeat or acquisition step; never a replacement). A bearing check that
      was not computed (``bearing_check``) contributes NO evidence either;
      the text states why and the remedy is recommended, as for a refused
      ISO block.
    - Anomaly model: Faulty +1.0, Suspicious +0.5.

    Mapping: 0 -> "none", <2 -> "weak", <3 -> "moderate", >=3 -> "strong".
    A quiet machine with no findings therefore gets "none" — severity alone
    can never produce "strong". The sentence that reports it says it counts
    findings, is not a probability and identifies no fault type.
    """
    lines = []
    recommendations = []
    evidence_points = 0.0

    # ISO severity assessment (may be a schema-level refusal)
    if iso_severity.get("status") == "refused":
        lines.append(f"ISO Severity: not assessed — {iso_severity['reason']}")
        recommendations.append(iso_severity["remedy"])
    else:
        zone = iso_severity["zone"]
        rms_vel = iso_severity["rms_velocity_mm_s"]
        # The severity word (Good/Acceptable/Unsatisfactory/Unacceptable) is
        # owned by iso20816._ZONE_INFO and arrives on the classification
        # result as ``severity_level`` — render it instead of re-hardcoding
        # the four labels here (they silently drift otherwise). The zone-keyed
        # evidence/recommendation ladder below is unchanged.
        severity_word = iso_severity.get("severity_level", "")
        severity_label = f" ({severity_word})" if severity_word else ""
        lines.append(
            f"ISO Severity: Zone {zone}{severity_label} — "
            f"RMS velocity {rms_vel:.2f} mm/s."
        )

        if zone in ("A", "B"):
            pass  # Good/Acceptable: no fault evidence, no recommendation.
        elif zone == "C":
            evidence_points += 1.0
            recommendations.append("Schedule maintenance within next planned shutdown.")
        else:  # Zone D — unacceptable (also the ladder's catch-all).
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

    # Bearing analysis: the sentence, the actions and the point follow the
    # check's claim (bearing_claims), never the legacy per-check grades.
    if bearing_faults:
        lines.append(pipeline_statement(bearing_faults))
        if counts_as_finding(bearing_faults["claim"]):
            evidence_points += 1.0
        recommendations.extend(claim_actions(bearing_faults))
    elif bearing_check["status"] == "not_computed":
        lines.append(
            f"Bearing analysis: not computed. {bearing_check['reason']} No "
            f"bearing check ran, so this diagnosis says nothing about the "
            f"condition of the bearing."
        )
        recommendations.append(bearing_check["remedy"])

    # Anomaly detection
    if anomaly:
        health = anomaly["overall_health"]
        ratio = anomaly["anomaly_ratio"]
        lines.append(
            f"Anomaly detection: {health} " f"({ratio * 100:.1f}% anomalous segments)."
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

    lines.append(aggregate_statement(evidence_strength))

    # Default recommendation if none generated
    if not recommendations:
        recommendations.append("Continue routine monitoring per maintenance schedule.")

    diagnosis_text = " ".join(lines)

    return diagnosis_text, evidence_strength, recommendations
