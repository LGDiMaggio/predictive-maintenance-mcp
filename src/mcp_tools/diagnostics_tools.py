"""MCP tools for diagnostics, anomaly detection, and documentation (ISO 13374 Blocks 3-4)."""

import logging
import json
import pickle
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from mcp.server.mcpserver import MCPServer, Context

from ..config import MODELS_DIR, RESOURCES_DIR
from ..models import (
    AnomalyModelResult, AnomalyPredictionResult,
    BearingCatalogMiss, BearingFaultCheckResult, BearingFaultsSummary,
    VibrationSeverityResult, ISOSeverityRefusal, DiagnosisResult,
)
from ..document_reader import (
    calculate_bearing_frequencies,
    extract_machine_specs,
    extract_text_from_pdf,
    lookup_bearing_in_catalog,
)
from ..diagnostics.bearing_catalog import (
    list_catalog_bearings as _list_catalog,
)
from ..diagnostics.bearing_analyzer import (
    check_all_bearing_faults as _check_all_faults,
    check_frequency_set as _check_frequency_set,
)
from ..diagnostics.iso20816 import (
    assess_severity_raw,
    classify_zone,
    describe_zone,
    check_power_scope,
)
from ..decision_support.alerts import check_custom_alert as _check_custom_alert
from ..decision_support.diagnosis_pipeline import (
    diagnose_vibration as _diagnose_vibration,
)

from ..signal_processing.features import (
    extract_time_domain_features,
    segment_and_extract_features as _segment_and_extract_features,
)
from ._utils import (
    resolve_signal,
    safe_resolve,
    resolve_model_paths,
    validate_name_component,
)

logger = logging.getLogger(__name__)


async def _extract_features_from_ids(
    signal_ids: list[str],
    segment_duration: float,
    overlap_ratio: float,
    ctx: Context = None,
) -> tuple[list[dict], dict[str, float]]:
    """
    Resolve stored signals, segment them, and extract features.

    Every signal must already be in the repository (load_signal — the batch
    form accepts a list of files) and must carry a sampling rate; a missing
    id or rate raises the standard actionable error (fail fast: a training
    set silently missing entries would train a misleading model).

    Args:
        signal_ids: Stored signal IDs (from load_signal)
        segment_duration: Segment duration in seconds
        overlap_ratio: Overlap ratio (0-1)
        ctx: MCP context for logging

    Returns:
        Tuple of (all_features_list, {signal_id: sampling_rate})
    """
    all_features = []
    detected_rates: dict[str, float] = {}

    for sid in signal_ids:
        signal_data, info = resolve_signal(sid)
        file_rate = info.sampling_rate

        if ctx:
            await ctx.info(f"  '{sid}': {file_rate} Hz")

        detected_rates[sid] = file_rate

        features = _segment_and_extract_features(signal_data, file_rate, segment_duration, overlap_ratio)
        all_features.extend(features)

    return all_features, detected_rates


async def _extract_and_transform_validation_features(
    signal_ids: list[str],
    segment_duration: float,
    overlap_ratio: float,
    scaler,
    pca,
    ctx: Context = None,
) -> Optional[np.ndarray]:
    """
    Extract features from stored validation signals and apply scaler + PCA.

    Args:
        signal_ids: Stored signal IDs (from load_signal)
        segment_duration: Segment duration in seconds
        overlap_ratio: Overlap ratio (0-1)
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer
        ctx: MCP context for logging

    Returns:
        PCA-transformed feature matrix, or None if no features extracted
    """
    features_list, _ = await _extract_features_from_ids(
        signal_ids, segment_duration, overlap_ratio, ctx=ctx
    )

    if not features_list:
        return None

    features_df = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df.values)
    features_pca = pca.transform(features_scaled)
    return features_pca


# ------------------------------------------------------------------
# Unified severity assessment (U9 merge)
# ------------------------------------------------------------------

#: Provenance note attached when user-defined custom thresholds are used.
_CUSTOM_THRESHOLD_PROVENANCE = (
    "User-defined custom thresholds (warning/alarm/danger boundaries) — "
    "ISO 10816-3:2009 zone values NOT applied."
)


def _validate_custom_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Validate a custom {warning, alarm, danger} threshold dict, or raise."""
    expected = {"warning", "alarm", "danger"}
    keys = set(thresholds)
    if keys != expected:
        raise ValueError(
            f"Invalid custom thresholds keys {sorted(keys)} — provide "
            f"exactly {{'warning': <mm/s>, 'alarm': <mm/s>, "
            f"'danger': <mm/s>}} (zone boundaries A/B, B/C, C/D)."
        )
    w, a, d = (
        float(thresholds["warning"]),
        float(thresholds["alarm"]),
        float(thresholds["danger"]),
    )
    if not (0 < w < a < d):
        raise ValueError(
            f"Invalid custom thresholds: need 0 < warning < alarm < danger, "
            f"got warning={w:g}, alarm={a:g}, danger={d:g} mm/s."
        )
    return {"warning": w, "alarm": a, "danger": d}


async def assess_severity(
    ctx: Context,
    signal_id: Optional[str] = None,
    rms_velocity_mm_s: Optional[float] = None,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
    thresholds: Optional[dict[str, float]] = None,
    machine_power_kw: Optional[float] = None,
    rpm: Optional[float] = None,
) -> VibrationSeverityResult:
    """Assess vibration severity (ISO 20816-3 zones A-D) and alert level.

    THE unified severity tool: ISO zone assessment and alert
    classification in one call. Zone boundary values are those of
    ISO 10816-3:2009 (ISO 20816-3:2022 merges zones A/B — provenance is
    noted in the result). Scope: machines rated above 15 kW; a declared
    machine_power_kw below 15 kW is refused.

    Input routes (exactly ONE required):
    - signal_id: a stored signal (load_signal first). Sampling rate AND
      declared unit come from the stored metadata; an undeclared unit is
      refused, never guessed from amplitude.
    - rms_velocity_mm_s: a direct broadband RMS velocity reading in mm/s
      (e.g. from a portable instrument) — no unit declaration needed.

    Args:
        ctx: MCP context.
        signal_id: ID of the stored signal (mutually exclusive with
            rms_velocity_mm_s).
        rms_velocity_mm_s: Direct broadband RMS velocity in mm/s
            (mutually exclusive with signal_id).
        machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW).
            Ignored when custom thresholds are given. Default 2.
        support_type: 'rigid' or 'flexible'. Ignored when custom
            thresholds are given. Default 'rigid'.
        thresholds: Optional custom zone boundaries {'warning': A/B,
            'alarm': B/C, 'danger': C/D} in mm/s, strictly increasing —
            replaces the ISO table for this call.
        machine_power_kw: Rated machine power, if known. Declared values
            below 15 kW are refused (out of ISO scope); None means
            unknown and is not refused.
        rpm: Operating speed in RPM (signal route only: selects the 2 Hz
            band lower edge below 600 RPM).

    Returns:
        VibrationSeverityResult (status='assessed') with zone, severity,
        boundaries, derived alert_level/exceeded_threshold, and threshold
        provenance.

    Raises:
        ValueError: On route misuse (both/neither inputs), undeclared
            signal unit, missing sampling rate, Nyquist below the ISO
            band, declared power below 15 kW, negative RMS, or invalid
            custom thresholds.
    """
    if (signal_id is None) == (rms_velocity_mm_s is None):
        raise ValueError(
            "Provide exactly one of signal_id or rms_velocity_mm_s — "
            "signal_id for a stored recording (load_signal first), "
            "rms_velocity_mm_s for a direct broadband RMS velocity "
            "reading in mm/s (e.g. from a portable instrument)."
        )

    # Scope refusal fails fast on BOTH routes (the engine re-checks it on
    # the signal route).
    check_power_scope(machine_power_kw)

    custom = (
        _validate_custom_thresholds(thresholds) if thresholds is not None else None
    )

    if signal_id is not None:
        # --- Stored-signal route (former evaluate_iso_20816 /
        # assess_vibration_severity) ----------------------------------
        signal_data, info = resolve_signal(signal_id)
        fs = info.sampling_rate

        unit = info.signal_unit
        if unit is None:
            raise ValueError(
                f"ISO severity refused for '{signal_id}': signal unit not "
                f"declared — units are never guessed from amplitude. "
                f"Re-load with load_signal(filepath=..., "
                f"signal_unit='g'|'m/s2'|'mm/s'|'m/s', overwrite=True) or "
                f"add a 'signal_unit' field to the companion "
                f"_metadata.json."
            )
        if ctx:
            await ctx.info(
                f"Assessing severity for '{signal_id}' (group "
                f"{machine_group}, {support_type}, unit: {unit})"
            )

        raw = assess_severity_raw(
            signal=signal_data,
            fs=fs,
            machine_group=machine_group,
            support_type=support_type,
            signal_unit=unit,
            operating_speed_rpm=rpm,
            machine_power_kw=machine_power_kw,
        )
        rms = raw["rms_velocity_mm_s"]
        frequency_range = raw["frequency_range"]
        unit_conversion = raw["unit_conversion_performed"]
        original_unit = raw["original_unit"]
        zone_block = {
            "zone": raw["zone"],
            "zone_description": raw["zone_description"],
            "severity_level": raw["severity_level"],
            "color_code": raw["color_code"],
            "boundaries": raw["boundaries"],
            "threshold_provenance": raw["threshold_provenance"],
        }
    else:
        # --- Direct-RMS route (former check_vibration_alert) ----------
        rms = float(rms_velocity_mm_s)
        frequency_range = "not applicable — direct RMS velocity reading"
        unit_conversion = False
        original_unit = "mm/s"
        if ctx:
            await ctx.info(
                f"Assessing severity for a direct reading of {rms:.2f} mm/s"
            )
        zone_info = classify_zone(rms, machine_group, support_type)
        zone_block = {
            "zone": zone_info["zone"],
            "zone_description": zone_info["zone_description"],
            "severity_level": zone_info["severity_level"],
            "color_code": zone_info["color_code"],
            "boundaries": zone_info["boundaries"],
            "threshold_provenance": zone_info["threshold_provenance"],
        }

    if custom is not None:
        # --- Custom thresholds override the zone classification
        # (former check_custom_vibration_alert) ------------------------
        zone = _check_custom_alert(rms, custom)["zone"]
        zone_block = {
            "zone": zone,
            **describe_zone(zone),
            "boundaries": {
                "AB": custom["warning"],
                "BC": custom["alarm"],
                "CD": custom["danger"],
            },
            "threshold_provenance": _CUSTOM_THRESHOLD_PROVENANCE,
        }

    if ctx:
        await ctx.info(
            f"Zone {zone_block['zone']} ({zone_block['severity_level']}): "
            f"{rms:.2f} mm/s"
        )

    return VibrationSeverityResult(
        signal_id=signal_id,
        rms_velocity_mm_s=round(rms, 4),
        machine_group=machine_group,
        support_type=str(support_type).lower(),
        frequency_range=frequency_range,
        unit_conversion_performed=unit_conversion,
        original_unit=original_unit,
        operating_speed_rpm=rpm,
        machine_power_kw=machine_power_kw,
        **zone_block,
    )

# ------------------------------------------------------------------
# ML anomaly detection – training
# ------------------------------------------------------------------

async def train_anomaly_model(
    healthy_signal_ids: list[str],
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    model_type: str = "OneClassSVM",
    pca_variance: float = 0.95,
    fault_signal_ids: Optional[list[str]] = None,
    healthy_validation_ids: Optional[list[str]] = None,
    model_name: str = "anomaly_model",
    ctx: Context = None
) -> AnomalyModelResult:
    """
        Train ML-based anomaly detection model on healthy data (UNSUPERVISED/SEMI-SUPERVISED).

        All signals are referenced by signal_id: load them first with
        load_signal — its batch form accepts a list of file paths, e.g.
        load_signal(filepath=["real_train/baseline_1.csv", ...]). Each
        signal's sampling rate comes from its stored metadata.

        Complete pipeline:
        1. Extract features from healthy signals (segmentation + time-domain features)
        2. Standardize features (StandardScaler - fitted on training data only)
        3. Dimensionality reduction (PCA with specified variance explained)
        4. Train novelty detection model (OneClassSVM or LocalOutlierFactor) on HEALTHY DATA ONLY
        5. Optional hyperparameter tuning using validation data (semi-supervised)
        6. Save model, scaler, and PCA transformer

        **Training Mode:**
        - UNSUPERVISED: Train only on healthy data with automatic hyperparameters
        - SEMI-SUPERVISED: Train on healthy data, tune hyperparameters using validation set (healthy + fault)

        **Note:** This is NOT supervised learning. OneClassSVM/LOF are trained ONLY on healthy data.
        Fault data (if provided) is used ONLY for hyperparameter tuning after training.

        **Validation Strategy:**
        - If healthy_validation_ids provided: Use those explicitly (no split)
        - If healthy_validation_ids NOT provided: Automatic 80/20 split of training data
        - If fault_signal_ids provided: Enable semi-supervised mode (hyperparameter tuning)

        Args:
            healthy_signal_ids: Stored signal IDs with healthy machine data (for training)
            segment_duration: Segment duration in seconds (default: 0.1)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            model_type: 'OneClassSVM' or 'LocalOutlierFactor' (default: 'OneClassSVM')
            pca_variance: Cumulative variance to explain with PCA (default: 0.95)
            fault_signal_ids: Optional stored signal IDs for HYPERPARAMETER TUNING (semi-supervised)
            healthy_validation_ids: Optional stored healthy signal IDs for validation (specificity check).
                                      If not provided, 20% of training data will be used.
            model_name: Name for saved model files (default: 'anomaly_model')
            ctx: MCP context for progress/logging

        Returns:
            AnomalyModelResult with model paths and performance metrics

        Raises:
            ValueError: If a signal_id is not loaded or has no sampling rate,
                or model_name/model_type is invalid.
        """
    # Fail fast on an unsafe model_name before doing any expensive work or
    # touching the filesystem (the write happens in step 6).
    validate_name_component(model_name, kind="model_name")

    if ctx:
        await ctx.info(f"Training {model_type} model on {len(healthy_signal_ids)} healthy signals...")

    # Step 1: Extract features from all healthy signals
    all_features, detected_rates = await _extract_features_from_ids(
        healthy_signal_ids, segment_duration, overlap_ratio, ctx=ctx
    )

    features_df = pd.DataFrame(all_features)
    X_train = features_df.values

    if ctx:
        await ctx.info(f"Extracted {X_train.shape[0]} feature vectors from healthy data")
        await ctx.info(f"Original feature dimension: {X_train.shape[1]}")

    # Step 2: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Step 3: PCA for dimensionality reduction
    pca = PCA(n_components=pca_variance)
    X_pca = pca.fit_transform(X_scaled)

    if ctx:
        await ctx.info(f"PCA components: {pca.n_components_}")
        await ctx.info(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 4: Train anomaly detection model
    # Strategy: Train on healthy data only (unsupervised), then use validation for hyperparameter tuning

    if model_type == "OneClassSVM":
        if fault_signal_ids:
            # SEMI-SUPERVISED MODE: Train on healthy, tune hyperparameters with validation (healthy + fault)
            if ctx:
                await ctx.info("Training in SEMI-SUPERVISED mode")
                await ctx.info("- Training: Healthy data only (unsupervised)")
                await ctx.info("- Hyperparameter tuning: Using validation set (healthy + fault)")
                await ctx.info("Evaluating hyperparameter grid...")

            # Prepare validation features for fault signals
            X_fault = await _extract_and_transform_validation_features(
                fault_signal_ids, segment_duration, overlap_ratio,
                scaler, pca, ctx=ctx
            )

            # Prepare validation features for healthy signals
            X_healthy_val = None
            if healthy_validation_ids:
                X_healthy_val = await _extract_and_transform_validation_features(
                    healthy_validation_ids, segment_duration, overlap_ratio,
                    scaler, pca, ctx=ctx
                )

            # Hyperparameter grid
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf']
            }

            # Manual hyperparameter search with validation scoring
            best_score = -np.inf
            best_params = None
            best_model = None

            for nu in param_grid['nu']:
                for gamma in param_grid['gamma']:
                    # Train on HEALTHY DATA ONLY
                    model_candidate = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                    model_candidate.fit(X_pca)  # Only healthy training data

                    # Evaluate on validation set (healthy + fault)
                    validation_score = 0.0
                    validation_count = 0

                    # Score healthy validation (should predict +1)
                    if X_healthy_val is not None:
                        healthy_predictions = model_candidate.predict(X_healthy_val)
                        healthy_accuracy = np.mean(healthy_predictions == 1)
                        validation_score += healthy_accuracy
                        validation_count += 1

                    # Score fault validation (should predict -1)
                    if X_fault is not None:
                        fault_predictions = model_candidate.predict(X_fault)
                        fault_accuracy = np.mean(fault_predictions == -1)
                        validation_score += fault_accuracy
                        validation_count += 1

                    # Balanced accuracy across healthy + fault
                    if validation_count > 0:
                        validation_score /= validation_count

                    if validation_score > best_score:
                        best_score = validation_score
                        best_params = {'kernel': 'rbf', 'nu': nu, 'gamma': gamma}
                        best_model = model_candidate

            model = best_model

            if ctx:
                await ctx.info(f"Best hyperparameters: nu={best_params['nu']}, gamma={best_params['gamma']}")
                await ctx.info(f"Validation balanced accuracy: {best_score:.3f}")

        else:
            # UNSUPERVISED MODE: No fault data -> Use automatic parameters
            if ctx:
                await ctx.info("Training in UNSUPERVISED mode (novelty detection)")
                await ctx.info("Using automatic parameters: nu='auto', gamma='scale'")

            # Auto-calculate nu based on expected outlier fraction (rule of thumb: 5%)
            nu_auto = min(0.1, max(0.01, 1.0 / np.sqrt(len(X_pca))))

            model = OneClassSVM(
                kernel='rbf',
                nu=nu_auto,  # Adaptive based on sample size
                gamma='scale'  # Automatic scaling based on features
            )
            model.fit(X_pca)

            best_params = {
                'kernel': 'rbf',
                'nu': float(nu_auto),
                'gamma': 'scale',
                'mode': 'unsupervised_auto'
            }

            if ctx:
                await ctx.info(f"Auto-calculated nu={nu_auto:.4f} based on sample size")

    elif model_type == "LocalOutlierFactor":
        if fault_signal_ids:
            # SEMI-SUPERVISED MODE with LOF
            if ctx:
                await ctx.info("Training LOF in SEMI-SUPERVISED mode")
                await ctx.info("- Training: Healthy data only (unsupervised)")
                await ctx.info("- Hyperparameter tuning: Using validation set")

            # Prepare validation features for fault signals
            X_fault = await _extract_and_transform_validation_features(
                fault_signal_ids, segment_duration, overlap_ratio,
                scaler, pca, ctx=ctx
            )

            # Prepare healthy validation features
            X_healthy_val = None
            if healthy_validation_ids:
                X_healthy_val = await _extract_and_transform_validation_features(
                    healthy_validation_ids, segment_duration, overlap_ratio,
                    scaler, pca, ctx=ctx
                )

            # Hyperparameter search for LOF
            best_score = -np.inf
            best_params = None
            best_model = None

            for n_neighbors in [10, 20, 30, 50]:
                for contamination in [0.05, 0.1, 0.15, 0.2]:
                    model_candidate = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination,
                        novelty=True
                    )
                    model_candidate.fit(X_pca)  # Only healthy training data

                    # Evaluate on validation set
                    validation_score = 0.0
                    validation_count = 0

                    # Score healthy validation
                    if X_healthy_val is not None:
                        healthy_predictions = model_candidate.predict(X_healthy_val)
                        healthy_accuracy = np.mean(healthy_predictions == 1)
                        validation_score += healthy_accuracy
                        validation_count += 1

                    # Score fault validation
                    if X_fault is not None:
                        fault_predictions = model_candidate.predict(X_fault)
                        fault_accuracy = np.mean(fault_predictions == -1)
                        validation_score += fault_accuracy
                        validation_count += 1

                    # Balanced accuracy
                    if validation_count > 0:
                        validation_score /= validation_count

                    if validation_score > best_score:
                        best_score = validation_score
                        best_params = {'n_neighbors': n_neighbors, 'contamination': contamination}
                        best_model = model_candidate

            model = best_model

            if ctx:
                await ctx.info(f"Best parameters: n_neighbors={best_params['n_neighbors']}, contamination={best_params['contamination']}")

        else:
            # UNSUPERVISED MODE: Auto parameters
            if ctx:
                await ctx.info("Training LOF in UNSUPERVISED mode")
                await ctx.info("Using automatic parameters based on sample size")

            # Auto-calculate n_neighbors (rule of thumb: sqrt(n) or ~5% of samples)
            n_auto = max(10, min(50, int(np.sqrt(len(X_pca)))))
            contamination_auto = 0.1  # Conservative 10% outlier assumption

            model = LocalOutlierFactor(
                n_neighbors=n_auto,
                contamination=contamination_auto,
                novelty=True
            )
            model.fit(X_pca)

            best_params = {
                'n_neighbors': int(n_auto),
                'contamination': contamination_auto,
                'mode': 'unsupervised_auto'
            }

            if ctx:
                await ctx.info(f"Auto-calculated n_neighbors={n_auto}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'OneClassSVM' or 'LocalOutlierFactor'")

    # Step 5: Optional validation on healthy + fault data
    validation_accuracy = None
    validation_details = None
    validation_metrics = None

    if fault_signal_ids or healthy_validation_ids:
        # Part A: Validate on HEALTHY data
        # Two options:
        # 1. User provides explicit healthy_validation_ids -> Use those
        # 2. User doesn't provide -> Auto-split training data 80/20

        if healthy_validation_ids:
            # Option 1: User provided explicit healthy validation files
            if ctx:
                await ctx.info(f"Using {len(healthy_validation_ids)} explicitly provided healthy validation files")

            # Extract and transform features from validation signals
            X_pca_healthy_val = await _extract_and_transform_validation_features(
                healthy_validation_ids, segment_duration, overlap_ratio,
                scaler, pca, ctx=ctx
            )

            if X_pca_healthy_val is not None:
                healthy_predictions = model.predict(X_pca_healthy_val)
                healthy_correct = np.sum(healthy_predictions == 1)
                healthy_total = len(healthy_predictions)
                healthy_accuracy = healthy_correct / healthy_total

                if ctx:
                    await ctx.info(f"Healthy validation: {healthy_correct}/{healthy_total} correctly classified ({healthy_accuracy*100:.1f}%)")
            else:
                healthy_correct = 0
                healthy_total = 0
                healthy_accuracy = 0.0

        else:
            # Option 2: Auto-split training data 80/20
            if ctx:
                await ctx.info("No healthy validation files provided - using 80/20 split of training data")

            split_idx = int(0.8 * len(X_pca))
            X_pca_train = X_pca[:split_idx]
            X_pca_healthy_val = X_pca[split_idx:]

            # Retrain model on 80% split for proper validation
            if model_type == "OneClassSVM":
                model_retrained = OneClassSVM(
                    kernel=best_params.get('kernel', 'rbf'),
                    nu=best_params['nu'],
                    gamma=best_params['gamma']
                )
                model_retrained.fit(X_pca_train)
                model = model_retrained  # Use retrained model

                if ctx:
                    await ctx.info("Model retrained on 80% of data for validation")

            # Validate on 20% split
            healthy_predictions = model.predict(X_pca_healthy_val)
            healthy_correct = np.sum(healthy_predictions == 1)
            healthy_total = len(healthy_predictions)
            healthy_accuracy = healthy_correct / healthy_total

            if ctx:
                await ctx.info(f"Healthy validation: {healthy_correct}/{healthy_total} correctly classified ({healthy_accuracy*100:.1f}%)")

        # Part B: Validate on FAULT data (only if fault files were provided)
        X_fault_pca = None
        if fault_signal_ids:
            X_fault_pca = await _extract_and_transform_validation_features(
                fault_signal_ids, segment_duration, overlap_ratio,
                scaler, pca, ctx=ctx
            )

        if X_fault_pca is not None:
            # Predict (should be -1 for anomalies)
            fault_predictions = model.predict(X_fault_pca)

            # Calculate fault detection rate
            anomaly_detected = np.sum(fault_predictions == -1)
            fault_total = len(fault_predictions)
            fault_accuracy = anomaly_detected / fault_total

            # Calculate overall balanced accuracy
            # Overall accuracy = (healthy_correct + fault_correct) / (healthy_total + fault_total)
            total_correct = healthy_correct + anomaly_detected
            total_samples = healthy_total + fault_total
            validation_accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

            validation_details = (
                f"Healthy: {healthy_correct}/{healthy_total} correct ({healthy_accuracy*100:.1f}%), "
                    f"Fault: {anomaly_detected}/{fault_total} detected ({fault_accuracy*100:.1f}%)"
            )

            validation_metrics = {
                'healthy_correct': int(healthy_correct),
                'healthy_total': int(healthy_total),
                'healthy_accuracy': float(healthy_accuracy),
                'fault_detected': int(anomaly_detected),
                'fault_total': int(fault_total),
                'fault_accuracy': float(fault_accuracy),
                'overall_accuracy': float(validation_accuracy)
            }

            if ctx:
                await ctx.info(f"Fault validation: {anomaly_detected}/{fault_total} detected as anomalies ({fault_accuracy*100:.1f}%)")
                await ctx.info(f"Overall validation accuracy: {validation_accuracy*100:.1f}%")

    # Step 6: Save model, scaler, and PCA
    # Validate model_name and contain every derived path before writing —
    # these are pickle files, so an unvalidated name is an arbitrary-write
    # (and later arbitrary-code-execution) primitive.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    _model_paths = resolve_model_paths(MODELS_DIR, model_name)
    model_path = _model_paths.model
    scaler_path = _model_paths.scaler
    pca_path = _model_paths.pca

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)

    # Save metadata. Each training signal carries its own stored sampling
    # rate; a single uniform rate is recorded numerically, mixed rates as
    # 'per_file' (prediction then uses the test signal's own stored rate).
    unique_rates = sorted(set(detected_rates.values()))
    metadata = {
        'model_type': model_type,
        'training_mode': 'supervised' if fault_signal_ids else 'unsupervised',
        'feature_names': list(features_df.columns),
        'num_features_original': X_train.shape[1],
        'num_features_pca': X_pca.shape[1],
        'pca_variance': float(pca.explained_variance_ratio_.sum()),
        'best_params': best_params,
        'sampling_rate': unique_rates[0] if len(unique_rates) == 1 else 'per_file',
        'sampling_rates_detected': detected_rates,
        'segment_duration': segment_duration,
        'overlap_ratio': overlap_ratio,
        'multi_rate_training': len(unique_rates) > 1,
        'validation_with_faults': fault_signal_ids is not None,
        'num_validation_files': len(fault_signal_ids) if fault_signal_ids else 0
    }

    metadata_path = _model_paths.metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if ctx:
        await ctx.info(f"Model saved to {model_path}")
        await ctx.info(f"Scaler saved to {scaler_path}")
        await ctx.info(f"PCA saved to {pca_path}")

    return AnomalyModelResult(
        model_name=model_name,
        model_type=model_type,
        num_training_samples=X_train.shape[0],
        num_features_original=X_train.shape[1],
        num_features_pca=X_pca.shape[1],
        variance_explained=float(pca.explained_variance_ratio_.sum()),
        model_params=best_params,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        pca_path=str(pca_path),
        validation_accuracy=validation_accuracy,
        validation_details=validation_details,
        validation_metrics=validation_metrics
    )

# ------------------------------------------------------------------
# ML anomaly detection – prediction
# ------------------------------------------------------------------

async def predict_anomalies(
    signal_id: str,
    model_name: str = "anomaly_model",
    ctx: Context = None
) -> AnomalyPredictionResult:
    """
        Predict anomalies in a stored signal using a trained model.

        Requires the signal loaded via load_signal() first and a model
        trained via train_anomaly_model (its result echoes the model_name
        to pass here). Pipeline: segment -> features -> scaler -> PCA ->
        predict -> aggregate.

        Output is BOUNDED: counts, anomaly ratio, score percentiles, and
        up to 10 worst segments — never per-segment arrays, regardless of
        signal length.

        Args:
            signal_id: ID of the stored signal to analyze (from load_signal)
            model_name: Name of trained model (default: 'anomaly_model')
            ctx: MCP context for progress/logging

        Returns:
            AnomalyPredictionResult with aggregate statistics and health
            assessment.

        Raises:
            FileNotFoundError: If the model does not exist (the message
                lists the models actually on disk).
            ValueError: If the signal_id is not loaded, or no sampling rate
                is available for segmentation.
        """
    if ctx:
        await ctx.info(f"Predicting anomalies in '{signal_id}'...")

    # Validate model_name and contain every derived path (single source of
    # truth shared with the training/PCA/pipeline model-load sites).
    _model_paths = resolve_model_paths(MODELS_DIR, model_name)
    model_path = _model_paths.model
    scaler_path = _model_paths.scaler
    pca_path = _model_paths.pca
    metadata_path = _model_paths.metadata

    if not model_path.exists():
        available = sorted(
            p.name[: -len("_model.pkl")]
            for p in MODELS_DIR.glob("*_model.pkl")
        ) if MODELS_DIR.exists() else []
        raise FileNotFoundError(
            f"Model '{model_name}' not found — models on disk: "
            f"{available if available else 'none'}. Train one with "
            f"train_anomaly_model(model_name=...) first."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Resolve the stored signal (fail fast with the standard message).
    # The stored rate is only REQUIRED when the model was trained with
    # per-file rates; a uniform-rate model segments at its training rate.
    signal_data, info = resolve_signal(signal_id, require_sampling_rate=False)

    # Extract features
    sampling_rate_val = metadata['sampling_rate']
    if isinstance(sampling_rate_val, str):
        # Model was trained with per-file rates: use the stored signal's rate.
        if info.sampling_rate is None:
            raise ValueError(
                f"Model '{model_name}' was trained with per-file sampling "
                    f"rates, but signal '{signal_id}' has no stored rate — "
                    f"re-load it with load_signal(filepath=..., "
                    f"sampling_rate=..., overwrite=True) or add a "
                    f"'sampling_rate' field to the companion _metadata.json."
            )
        sampling_rate_val = info.sampling_rate
    sampling_rate_val = float(sampling_rate_val)
    segment_duration = metadata['segment_duration']
    overlap_ratio = metadata['overlap_ratio']

    segment_length_samples = int(segment_duration * sampling_rate_val)
    hop_length = int(segment_length_samples * (1 - overlap_ratio))

    # Guard: a signal shorter than one segment yields an empty feature matrix
    # and an opaque sklearn "X has 0 features" error at scaler.transform.
    # Fail early with an actionable message instead.
    if len(signal_data) < segment_length_samples:
        raise ValueError(
            f"Signal '{signal_id}' ({len(signal_data)} samples) is shorter than "
            f"one {segment_duration}s segment at {sampling_rate_val:g} Hz — "
            f"provide a longer recording or a smaller segment_duration."
        )

    features_list = []
    for start in range(0, len(signal_data) - segment_length_samples + 1, hop_length):
        segment = signal_data[start:start + segment_length_samples]
        features = extract_time_domain_features(segment)
        features_list.append(features)

    X_test = pd.DataFrame(features_list).values

    # Apply preprocessing
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)

    # Predict
    predictions = model.predict(X_pca)

    # Decision scores when the model exposes them (negative = anomalous).
    scores = None
    if hasattr(model, 'decision_function'):
        scores = np.asarray(model.decision_function(X_pca), dtype=float)

    # Aggregate statistics — the per-segment arrays never leave this
    # function (bounded output regardless of signal length).
    anomaly_count = int(np.sum(predictions == -1))
    anomaly_ratio = float(anomaly_count / len(predictions))

    score_percentiles = None
    if scores is not None:
        score_percentiles = {
            f"p{p}": round(float(np.percentile(scores, p)), 6)
            for p in (5, 25, 50, 75, 95)
        }

    # Worst segments: lowest decision score (or, without scores, the
    # first predicted anomalies), capped at 10 entries.
    hop_s = segment_duration * (1 - overlap_ratio)
    if scores is not None:
        worst_idx = np.argsort(scores)[:10]
    else:
        worst_idx = np.flatnonzero(predictions == -1)[:10]
    worst_segments = [
        {
            "segment_index": int(i),
            "start_time_s": round(float(i * hop_s), 4),
            **(
                {"score": round(float(scores[i]), 6)}
                if scores is not None
                else {}
            ),
        }
        for i in worst_idx
    ]

    # Assess overall health (thresholds on the anomaly ratio itself —
    # no separate "confidence" label is derived from them).
    if anomaly_ratio < 0.1:
        overall_health = "Healthy"
    elif anomaly_ratio < 0.3:
        overall_health = "Suspicious"
    else:
        overall_health = "Faulty"

    if ctx:
        await ctx.info(f"Analyzed {len(predictions)} segments")
        await ctx.info(f"Anomalies detected: {anomaly_count} ({anomaly_ratio*100:.1f}%)")
        await ctx.info(f"Health status: {overall_health}")

    return AnomalyPredictionResult(
        model_name=model_name,
        num_segments=len(predictions),
        anomaly_count=anomaly_count,
        anomaly_ratio=anomaly_ratio,
        segment_duration_s=float(segment_duration),
        score_percentiles=score_percentiles,
        worst_segments=worst_segments,
        overall_health=overall_health,
    )

# ------------------------------------------------------------------
# Machine documentation tools
# ------------------------------------------------------------------

async def list_machine_manuals(ctx: Context | None = None) -> list[dict[str, Any]]:
    """
        List all machine manuals in resources/machine_manuals/ (PDF/TXT).

        Use before read_manual_excerpt / extract_manual_specs, and pass the
        returned filenames exactly as-is.

        Returns:
            List of dicts with filename, size_mb, modified, and path.
        """
    manuals_dir = RESOURCES_DIR / "machine_manuals"
    manuals = []

    # Support both PDF and TXT files
    for manual_file in list(manuals_dir.glob("*.pdf")) + list(manuals_dir.glob("*.txt")):
        stat = manual_file.stat()
        manuals.append({
            "filename": manual_file.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "path": str(manual_file.relative_to(RESOURCES_DIR))
        })

    if ctx:
        await ctx.info(f"Found {len(manuals)} machine manuals in resources/machine_manuals/")


    return sorted(manuals, key=lambda x: x['filename'])

async def extract_manual_specs(
    file_name: str,
    use_cache: bool = True,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Extract machine specifications from an equipment manual (PDF).

        Extracts bearing designations (e.g. SKF 6205), operating speeds
        (RPM), power ratings (kW/HP/MW), and a text excerpt. Results are
        cached. If a bearing's geometry is not in the manual, follow up
        with search_bearing_catalog(bearing_id=...); if it is not in the
        catalog either, ask the user for the geometry — never invent it.

        Args:
            file_name: Manual filename in resources/machine_manuals/
            use_cache: Use cached extraction if available (default: True)
            ctx: MCP context

        Returns:
            Dictionary with extracted specifications and text excerpt.

        Raises:
            FileNotFoundError: If the manual does not exist (the message
                lists the available manuals).
        """
    if ctx:
        await ctx.info(f"Extracting specifications from: {file_name}")

    manual_path = safe_resolve(RESOURCES_DIR / "machine_manuals", file_name)

    if not manual_path.exists():
        raise FileNotFoundError(
            f"Manual not found: {file_name}\n"
                f"Available manuals: {[f.name for f in (RESOURCES_DIR / 'machine_manuals').glob('*.pdf')]}"
        )

    # Extract specs (with caching)
    specs = extract_machine_specs(manual_path, use_cache=use_cache)

    if ctx:
        await ctx.info(f"Found {len(specs['bearings'])} bearing designations")
        await ctx.info(f"Found {len(specs['rpm_values'])} RPM values")
        if specs['bearings']:
            await ctx.info(f"Bearings: {', '.join(specs['bearings'][:5])}")

    return specs

async def calculate_bearing_characteristic_frequencies(
    num_balls: int,
    ball_diameter_mm: float,
    pitch_diameter_mm: float,
    contact_angle_deg: float = 0.0,
    rpm: float = 1500.0,
    ctx: Context | None = None
) -> dict[str, float]:
    """
        Calculate bearing characteristic frequencies from geometry.

        Standard rolling-element kinematic formulas (Randall & Antoni 2011,
        "Rolling element bearing diagnostics — A tutorial", MSSP 25(2)).
        Requires the EXACT geometry — from the manual, the catalog
        (search_bearing_catalog), or the user; never guessed. Deep-groove
        ball bearings have contact_angle_deg = 0.

        Args:
            num_balls: Number of rolling elements (Z)
            ball_diameter_mm: Ball/roller diameter (Bd) in mm
            pitch_diameter_mm: Pitch circle diameter (Pd) in mm
            contact_angle_deg: Contact angle (alpha) in degrees
            rpm: Shaft rotation speed in RPM
            ctx: MCP context

        Returns:
            Dictionary with BPFO, BPFI, BSF, FTF in Hz.

        Example:
            >>> # 6205 geometry (CWRU Bearing Data Center) at 1797 RPM
            >>> freqs = calculate_bearing_characteristic_frequencies(
            ...     num_balls=9, ball_diameter_mm=7.94,
            ...     pitch_diameter_mm=39.04, rpm=1797
            ... )
            >>> round(freqs['BPFO'], 2)
            107.36
        """
    if ctx:
        await ctx.info(f"Calculating bearing frequencies for {num_balls} balls at {rpm} RPM")

    freqs = calculate_bearing_frequencies(
        num_balls=num_balls,
        ball_diameter_mm=ball_diameter_mm,
        pitch_diameter_mm=pitch_diameter_mm,
        contact_angle_deg=contact_angle_deg,
        shaft_speed_rpm=rpm
    )

    if ctx:
        await ctx.info(f"BPFO (outer race): {freqs['BPFO']:.2f} Hz")
        await ctx.info(f"BPFI (inner race): {freqs['BPFI']:.2f} Hz")
        await ctx.info(f"BSF (ball spin): {freqs['BSF']:.2f} Hz")
        await ctx.info(f"FTF (cage): {freqs['FTF']:.2f} Hz")

    return freqs

async def read_manual_excerpt(
    file_name: str,
    max_pages: int = 10,
    ctx: Context | None = None
) -> str:
    """
        Read a text excerpt from a machine manual (PDF or TXT).

        Use for consecutive-page reading; for targeted questions prefer
        search_documentation. Start with max_pages=10 and increase only if
        needed (pages consume tokens).

        Args:
            file_name: Manual filename in resources/machine_manuals/
                (PDF or TXT)
            max_pages: Maximum pages to extract (ignored for TXT files)
            ctx: MCP context

        Returns:
            Extracted text from the manual.

        Raises:
            FileNotFoundError: If the manual does not exist.
        """
    if ctx:
        await ctx.info(f"Reading from: {file_name}")

    manual_path = safe_resolve(RESOURCES_DIR / "machine_manuals", file_name)

    if not manual_path.exists():
        available = [
            f.name
            for f in (RESOURCES_DIR / "machine_manuals").glob("*")
            if f.suffix.lower() in (".pdf", ".txt")
        ]
        raise FileNotFoundError(
            f"Manual not found: {file_name} — available manuals: "
            f"{available if available else 'none'} "
            f"(see list_machine_manuals())."
        )

    # Read based on file type
    if manual_path.suffix.lower() == '.txt':
        with open(manual_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if ctx:
            await ctx.info(f"Extracted {len(text)} characters from text file")
    else:
        text = extract_text_from_pdf(manual_path, max_pages=max_pages)
        if ctx:
            await ctx.info(f"Extracted {len(text)} characters from {max_pages} pages")

    return text

async def search_bearing_catalog(
    bearing_id: str,
    ctx: Context | None = None
) -> dict[str, Any] | BearingCatalogMiss:
    """
        Search for bearing specifications in the local verified catalog.

        Fallback for when the machine manual names a bearing but not its
        geometry. The catalog is small BY DESIGN: only entries whose
        geometry is traceable to a public source (mandatory `source`
        citation). A miss is a legitimate negative outcome — ask the user
        for the geometry; never guess it.

        Args:
            bearing_id: Bearing designation (e.g. "6205", "SKF 6205-2RS")
            ctx: MCP context

        Returns:
            Dictionary with bearing specifications if found, or a
            BearingCatalogMiss (status='not_found', suggestion,
            catalog_contains) when the bearing is not in the catalog.

        Raises:
            Exception: If the catalog itself cannot be read (missing or
                malformed common_bearings_catalog.json).
        """
    if ctx:
        await ctx.info(f"Searching catalog for bearing: {bearing_id}")

    bearing_specs = lookup_bearing_in_catalog(bearing_id)

    if bearing_specs:
        if ctx:
            await ctx.info(f"Found {bearing_specs['designation']} in catalog (source: {bearing_specs.get('source', 'unknown')})")
            await ctx.info(f"  Type: {bearing_specs.get('type', 'N/A')}")
            await ctx.info(f"  Balls: {bearing_specs['num_balls']}, Ball diameter: {bearing_specs['ball_diameter_mm']} mm")
            await ctx.info(f"  Pitch diameter: {bearing_specs['pitch_diameter_mm']} mm")
        return bearing_specs

    # Not in catalog: a legitimate negative outcome — typed result, never an
    # ad-hoc {"error": ...} dict returned as success. No geometry is invented.
    if ctx:
        await ctx.warning(f"Bearing {bearing_id} not found in catalog")
        await ctx.warning("  LLM should ask user for bearing geometry or suggest uploading manufacturer catalog")

    available = sorted(b["designation"] for b in _list_catalog())
    return BearingCatalogMiss(
        bearing_id=bearing_id,
        suggestion=(
            "Ask user for bearing geometry (num_balls, ball_diameter_mm, "
            "pitch_diameter_mm, contact_angle_deg) or upload manufacturer "
            "catalog PDF to resources/bearing_catalogs/"
        ),
        catalog_contains=available,
    )

# ------------------------------------------------------------------
# Document search (RAG)
# ------------------------------------------------------------------

async def search_documentation(
    query: str,
    top_k: int = 5,
    force_reindex: bool = False,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Semantic search across all machine manuals and bearing catalogs.

        Uses vector retrieval (RAG) to find the most relevant passages from
        PDFs, text files, and JSON catalogs in resources/.

        Backends (chosen automatically):
          - FAISS + sentence-transformers  (pip install predictive-maintenance-mcp[vector-search])
          - TF-IDF keyword search          (default, zero extra deps)

        The index is built lazily on first call and cached on disk.  It is
        automatically rebuilt when source files change.

        Args:
            query: Natural-language question or keywords
                   (e.g. "bearing 6205 geometry", "maintenance interval pump")
            top_k: Number of passages to return (default: 5)
            force_reindex: Rebuild the index even if cache is fresh (default: False)
            ctx: MCP context

        Returns:
            Dictionary with ranked results, each containing text passage, source
            file, relevance score, and chunk index.
        """
    # Deliberately lazy: the RAG backend may pull in FAISS /
    # sentence-transformers, which are heavy optional dependencies —
    # importing them at module load would slow server startup for
    # everyone, including users who never search documentation.
    from ..rag import get_or_build_index, backend_name

    if ctx:
        await ctx.info(f"Searching documentation for: {query!r} (backend: {backend_name()})")

    idx = get_or_build_index(force_rebuild=force_reindex)

    if idx.num_chunks == 0:
        return {
            "results": [],
            "backend": backend_name(),
            "note": "No documents indexed. Add PDFs or TXT files to resources/machine_manuals/ or resources/bearing_catalogs/."
        }

    results = idx.query(query, top_k=top_k)

    if ctx:
        await ctx.info(f"Found {len(results)} relevant passages from {len(set(r['source'] for r in results))} documents")

    return {
        "query": query,
        "num_results": len(results),
        "total_chunks_indexed": idx.num_chunks,
        "backend": backend_name(),
        "results": results,
    }

# ------------------------------------------------------------------
# Unified bearing fault check (U9 merge)
# ------------------------------------------------------------------

async def check_bearing_faults(
    ctx: Context,
    signal_id: str,
    rpm: float,
    bearing_id: Optional[str] = None,
    frequencies: Optional[dict[str, float]] = None,
    num_balls: Optional[int] = None,
    ball_diameter_mm: Optional[float] = None,
    pitch_diameter_mm: Optional[float] = None,
    contact_angle_deg: float = 0.0,
    tolerance_pct: float = 5.0,
) -> BearingFaultsSummary:
    """Check expected fault frequencies in a stored signal's envelope spectrum.

    THE unified bearing-check tool: catalog lookup, explicit
    frequencies, or explicit geometry in one call. Requires the signal
    loaded via load_signal() first.

    Expected-frequency routes (exactly ONE required):
    - bearing_id: catalog lookup (verified entries only) — BPFO/BPFI/BSF/
      FTF computed from the catalog geometry; the entry's source citation
      is echoed in the result.
    - frequencies: explicit {label: hz} dict — for bearings not in the
      catalog or non-bearing checks such as a gearbox GMF
      (e.g. {"GMF": 350.0}). Labels BPFO/BPFI/BSF/FTF map to the
      canonical fault vocabulary; other labels have no canonical form.
    - explicit geometry: num_balls + ball_diameter_mm + pitch_diameter_mm
      (+ contact_angle_deg) — frequencies computed from user-provided
      geometry (out-of-catalog path).

    Each check reports fault_type_canonical (outer_race / inner_race /
    ball / cage) alongside the acronym.

    Args:
        ctx: MCP context.
        signal_id: ID of the stored signal.
        rpm: Shaft speed in RPM.
        bearing_id: Bearing designation (e.g. '6205', 'SKF 6205-2RS').
        frequencies: Explicit expected frequencies {label: hz}, all > 0.
        num_balls: Number of rolling elements (explicit-geometry route).
        ball_diameter_mm: Ball/roller diameter Bd in mm.
        pitch_diameter_mm: Pitch circle diameter Pd in mm.
        contact_angle_deg: Contact angle in degrees (default 0.0).
        tolerance_pct: Frequency matching tolerance in percent (default 5).

    Returns:
        BearingFaultsSummary with one check per expected frequency,
        overall assessment, most likely fault (+ canonical form), and the
        provenance of the expected frequencies (`source`).

    Raises:
        ValueError: If the signal is not loaded / has no sampling rate, if
            not exactly one route is given, if the geometry is incomplete,
            if the bearing is not in the catalog, or if frequencies is
            empty / non-positive.
    """
    geometry_given = [
        p is not None for p in (num_balls, ball_diameter_mm, pitch_diameter_mm)
    ]
    routes = sum(
        [bearing_id is not None, frequencies is not None, any(geometry_given)]
    )
    if routes != 1:
        raise ValueError(
            "Provide exactly ONE expected-frequency route — bearing_id "
            "(catalog lookup), frequencies (explicit {label: hz} dict, "
            "e.g. {'GMF': 350.0}), or the full bearing geometry "
            "(num_balls + ball_diameter_mm + pitch_diameter_mm)."
        )
    if any(geometry_given) and not all(geometry_given):
        missing = [
            name
            for name, given in zip(
                ("num_balls", "ball_diameter_mm", "pitch_diameter_mm"),
                geometry_given,
            )
            if not given
        ]
        raise ValueError(
            f"Incomplete bearing geometry: missing {', '.join(missing)} — "
            f"the explicit-geometry route needs num_balls, "
            f"ball_diameter_mm, and pitch_diameter_mm together."
        )

    signal_data, info = resolve_signal(signal_id)
    fs = info.sampling_rate

    if bearing_id is not None:
        # --- Catalog route (former check_bearing_faults_direct /
        # lookup_bearing_and_compute_tool) -----------------------------
        if ctx:
            await ctx.info(
                f"Checking catalog bearing {bearing_id} at {rpm} RPM"
            )
        result = _check_all_faults(
            signal=signal_data,
            fs=fs,
            bearing_id=bearing_id,
            rpm=rpm,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )
    elif frequencies is not None:
        # --- Explicit-frequencies route (gearbox / out-of-catalog) -----
        if ctx:
            await ctx.info(
                f"Checking {len(frequencies)} explicit frequencies at "
                f"{rpm} RPM: {sorted(frequencies)}"
            )
        result = _check_frequency_set(
            signal=signal_data,
            fs=fs,
            frequencies=frequencies,
            rpm=rpm,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )
        result["source"] = "user-provided frequencies"
    else:
        # --- Explicit-geometry route (bearing not in catalog) ----------
        if ctx:
            await ctx.info(
                f"Checking user geometry (Z={num_balls}, Bd="
                f"{ball_diameter_mm} mm, Pd={pitch_diameter_mm} mm) at "
                f"{rpm} RPM"
            )
        freqs = calculate_bearing_frequencies(
            num_balls=num_balls,
            ball_diameter_mm=ball_diameter_mm,
            pitch_diameter_mm=pitch_diameter_mm,
            contact_angle_deg=contact_angle_deg,
            shaft_speed_rpm=rpm,
        )
        result = _check_frequency_set(
            signal=signal_data,
            fs=fs,
            frequencies={k: freqs[k] for k in ("BPFO", "BPFI", "BSF", "FTF")},
            rpm=rpm,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )
        result["source"] = "user-provided geometry"

    if ctx:
        await ctx.info(result["overall_assessment"])

    return BearingFaultsSummary(
        signal_id=result["signal_id"],
        bearing_id=result["bearing_id"],
        rpm=result["rpm"],
        shaft_frequency_hz=result["shaft_frequency_hz"],
        bearing_frequencies=result["bearing_frequencies"],
        fault_checks=[BearingFaultCheckResult(**c) for c in result["fault_checks"]],
        overall_assessment=result["overall_assessment"],
        most_likely_fault=result["most_likely_fault"],
        most_likely_fault_canonical=result["most_likely_fault_canonical"],
        source=result["source"],
    )

# ------------------------------------------------------------------
# Integrated diagnosis (signal_id based)
# ------------------------------------------------------------------

async def diagnose_vibration(
    ctx: Context,
    signal_id: str,
    rpm: float,
    bearing_id: Optional[str] = None,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
) -> DiagnosisResult:
    """Full integrated diagnosis: FFT + PSD + STFT + bearing faults + ISO severity.

        Comprehensive vibration diagnostic pipeline. Loads signal from repository,
        runs all analyses, and synthesizes results into an actionable report.
        The ISO severity block uses ISO 20816-3 machine group/support type
        (zone boundaries from ISO 10816-3:2009, provenance noted in output).

        The diagnosis DEGRADES instead of failing when the ISO verdict cannot
        be produced honestly: if the stored signal has no declared unit (or
        the sampling rate cannot cover the ISO evaluation band), the
        iso_severity block is a structured refusal (status='refused' with
        reason and remedy) while the spectral, bearing, and anomaly blocks
        still run. Units are never guessed from amplitude — declare them via
        load_signal(signal_unit=...) or the companion _metadata.json.

        Args:
            signal_id: ID of the stored signal.
            rpm: Machine operating speed in RPM.
            bearing_id: Bearing designation for fault detection (optional).
            machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW).
                Default 2.
            support_type: 'rigid' or 'flexible'. Default 'rigid'.

        Raises:
            ValueError: If the stored signal has no sampling rate.
        """
    signal_data, info = resolve_signal(signal_id)
    fs = info.sampling_rate

    # None = undeclared: pipeline degrades to a refused ISO block while
    # the other diagnosis blocks still run (no unit guessing).
    signal_unit = info.signal_unit
    if ctx:
        await ctx.info(f"Running full diagnosis for '{signal_id}' at {rpm} RPM")
        if bearing_id:
            await ctx.info(f"Bearing analysis: {bearing_id}")

    result = _diagnose_vibration(
        signal=signal_data,
        fs=fs,
        rpm=rpm,
        signal_id=signal_id,
        bearing_id=bearing_id,
        machine_group=machine_group,
        support_type=support_type,
        signal_unit=signal_unit,
    )

    # Convert nested dicts to Pydantic models
    bearing_faults_model = None
    if result["bearing_faults"]:
        bf = result["bearing_faults"]
        bearing_faults_model = BearingFaultsSummary(
            signal_id=bf["signal_id"],
            bearing_id=bf["bearing_id"],
            rpm=bf["rpm"],
            shaft_frequency_hz=bf["shaft_frequency_hz"],
            bearing_frequencies=bf["bearing_frequencies"],
            fault_checks=[BearingFaultCheckResult(**c) for c in bf["fault_checks"]],
            overall_assessment=bf["overall_assessment"],
            most_likely_fault=bf["most_likely_fault"],
            most_likely_fault_canonical=bf.get("most_likely_fault_canonical"),
            source=bf.get("source"),
        )

    # ISO block: assessed result or schema-level refusal (reason + remedy)
    iso_block = result["iso_severity"]
    if iso_block.get("status") == "refused":
        iso_model: VibrationSeverityResult | ISOSeverityRefusal = (
            ISOSeverityRefusal(
                signal_id=iso_block.get("signal_id", signal_id),
                reason=iso_block["reason"],
                remedy=iso_block["remedy"],
            )
        )
    else:
        iso_model = VibrationSeverityResult(**iso_block)

    if ctx:
        await ctx.info(
            f"Diagnosis complete: {result['evidence_strength']} fault evidence"
        )
        if iso_block.get("status") == "refused":
            await ctx.info(f"ISO severity: refused — {iso_block['reason']}")
        else:
            await ctx.info(f"ISO Zone: {iso_block['zone']}")
        for rec in result["recommendations"]:
            await ctx.info(f"  -> {rec}")

    return DiagnosisResult(
        signal_id=result["signal_id"],
        rpm=result["rpm"],
        bearing_id=result["bearing_id"],
        machine_group=result["machine_group"],
        support_type=result["support_type"],
        fft_summary=result["fft_summary"],
        psd_summary=result["psd_summary"],
        stft_summary=result["stft_summary"],
        bearing_faults=bearing_faults_model,
        iso_severity=iso_model,
        anomaly_detection=result.get("anomaly_detection"),
        overall_diagnosis=result["overall_diagnosis"],
        evidence_strength=result["evidence_strength"],
        recommendations=result["recommendations"],
    )


def register(mcp: MCPServer) -> None:
    """Register diagnostics, anomaly-detection, and documentation tools on *mcp*."""
    mcp.tool()(assess_severity)
    mcp.tool()(train_anomaly_model)
    mcp.tool()(predict_anomalies)
    mcp.tool()(list_machine_manuals)
    mcp.tool()(extract_manual_specs)
    mcp.tool()(calculate_bearing_characteristic_frequencies)
    mcp.tool()(read_manual_excerpt)
    mcp.tool()(search_bearing_catalog)
    mcp.tool()(search_documentation)
    mcp.tool()(check_bearing_faults)
    mcp.tool()(diagnose_vibration)
