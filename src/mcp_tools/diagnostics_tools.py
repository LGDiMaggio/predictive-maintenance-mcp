"""MCP tools for diagnostics, anomaly detection, and documentation (ISO 13374 Blocks 3-4)."""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR, MODELS_DIR, RESOURCES_DIR, REPORTS_DIR, CACHE_DIR
from ..signal_loader import load_signal_data, get_metadata_path, extract_segment, SUPPORTED_EXTENSIONS
from ..signal_repository import (
    get_repository,
    normalize_signal_unit,
    VALID_SIGNAL_UNITS,
)
from ..models import (
    ISO20816Result, AnomalyModelResult, AnomalyPredictionResult,
    BearingFaultCheckResult, BearingFaultsSummary, VibrationSeverityResult,
    ISOSeverityRefusal, DiagnosisResult, StoredSignalInfo
)
from ..report_generator import save_iso_report, REPORTS_DIR as REPORTS_DIR_PATH
from ..document_reader import (
    calculate_bearing_frequencies,
    extract_machine_specs,
    extract_bearing_designation,
    extract_rpm_values,
    extract_power_ratings,
    extract_text_from_pdf,
)
from ..spectral import (
    compute_psd as _compute_psd,
    compute_stft_spectrogram as _compute_stft,
    compute_envelope_spectrum as _compute_envelope,
)
from ..bearing_catalog import (
    lookup_bearing as _lookup_bearing,
    compute_fault_frequencies as _compute_fault_freqs,
    list_catalog_bearings as _list_catalog,
)
from ..bearing_analyzer import (
    check_bearing_fault_peak as _check_fault_peak,
    check_all_bearing_faults as _check_all_faults,
    lookup_bearing_and_compute as _lookup_and_compute,
)
from ..iso10816 import assess_vibration_severity as _assess_severity
from ..diagnosis_pipeline import diagnose_vibration as _diagnose_vibration

from ..signal_processing.features import (
    extract_time_domain_features,
    segment_and_extract_features as _segment_and_extract_features,
    resolve_sampling_rate as _resolve_sampling_rate,
)
from ._utils import (
    safe_resolve,
    resolve_model_paths,
    validate_name_component,
)

logger = logging.getLogger(__name__)


async def _extract_features_from_files(
    signal_files: list[str],
    sampling_rate: Optional[float],
    segment_duration: float,
    overlap_ratio: float,
    strict: bool = True,
    ctx: Context = None,
) -> tuple[list[dict], list[float]]:
    """
    Load signals, resolve sampling rates, segment, and extract features.

    Args:
        signal_files: List of signal filenames relative to DATA_DIR
        sampling_rate: User-provided sampling rate (None = auto-detect per file)
        segment_duration: Segment duration in seconds
        overlap_ratio: Overlap ratio (0-1)
        strict: If True, raise on missing files/rates; if False, skip them
        ctx: MCP context for logging

    Returns:
        Tuple of (all_features_list, detected_rates)
    """
    all_features = []
    detected_rates = []

    for signal_file in signal_files:
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            if strict:
                raise FileNotFoundError(f"File not found: {signal_file}")
            else:
                if ctx:
                    await ctx.warning(f"File not found: {signal_file}, skipping")
                continue

        file_rate = _resolve_sampling_rate(signal_file, sampling_rate, strict=strict)
        if file_rate is None:
            if ctx:
                await ctx.warning(f"No sampling rate for {signal_file}, skipping")
            continue

        if ctx and sampling_rate is None:
            await ctx.info(f"  {signal_file}: detected {file_rate} Hz from metadata")

        detected_rates.append(file_rate)

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            if strict:
                raise ValueError(f"Could not load signal data from: {signal_file}")
            else:
                if ctx:
                    await ctx.warning(f"Could not load {signal_file}, skipping")
                continue

        features = _segment_and_extract_features(signal_data, file_rate, segment_duration, overlap_ratio)
        all_features.extend(features)

    return all_features, detected_rates


async def _extract_and_transform_validation_features(
    signal_files: list[str],
    sampling_rate: Optional[float],
    segment_duration: float,
    overlap_ratio: float,
    scaler,
    pca,
    strict: bool = False,
    ctx: Context = None,
) -> Optional[np.ndarray]:
    """
    Extract features from validation files and apply scaler + PCA transform.

    Args:
        signal_files: List of signal filenames relative to DATA_DIR
        sampling_rate: User-provided sampling rate (None = auto-detect)
        segment_duration: Segment duration in seconds
        overlap_ratio: Overlap ratio (0-1)
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer
        strict: If True, raise on errors; if False, skip problematic files
        ctx: MCP context for logging

    Returns:
        PCA-transformed feature matrix, or None if no features extracted
    """
    features_list, _ = await _extract_features_from_files(
        signal_files, sampling_rate, segment_duration, overlap_ratio,
        strict=strict, ctx=ctx
    )

    if not features_list:
        return None

    features_df = pd.DataFrame(features_list)
    features_scaled = scaler.transform(features_df.values)
    features_pca = pca.transform(features_scaled)
    return features_pca


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register(mcp: FastMCP) -> None:
    """Register diagnostics, anomaly-detection, and documentation tools on *mcp*."""

    # Import ServerSession for type hints used in some signatures
    from mcp.server.session import ServerSession

    # ------------------------------------------------------------------
    # ISO 20816 evaluation
    # ------------------------------------------------------------------

    @mcp.tool()
    async def evaluate_iso_20816(
        ctx: Context,
        signal_file: str,
        sampling_rate: Optional[float] = None,
        machine_group: int = 2,  # Default 2 (medium) - most common industrial case
        support_type: str = "rigid",  # Default rigid - most common for horizontal machines
        operating_speed_rpm: Optional[float] = None,
        signal_unit: Optional[str] = None  # 'g'/'m/s2' (acceleration) or 'mm/s'/'m/s' (velocity); None = read from metadata
    ) -> ISO20816Result:
        """
        Evaluate vibration severity according to the ISO 20816-3 standard.

        ISO 20816-3 defines vibration severity zones for rotating machinery based on
        broadband RMS velocity measurements on non-rotating parts (bearings, housings).
        Zone boundary values are those of ISO 10816-3:2009 (four-zone A-D scheme;
        ISO 20816-3:2022 merges zones A/B) — the provenance note is included in the
        result. Scope: machines rated above 15 kW.

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type or severity from filename** (e.g., "OuterRaceFault_1.csv" does NOT mean outer race fault)
        - **NEVER assume baseline/healthy from filename** (e.g., "baseline" does NOT guarantee Zone A)
        - Treat ALL filenames as opaque identifiers
        - Report ONLY the ISO zone returned by measurement, regardless of filename
        - If filename suggests "baseline" but measurement shows Zone C/D, report Zone C/D

        **DEFAULTS** (use if user doesn't specify):
        - machine_group = 2 (medium-sized machines, most common)
        - support_type = "rigid" (horizontal machines on foundations)

        **Machine Group Selection Guide** (ask user if unsure):
        - Group 1: Large machines (power >300 kW OR shaft height H >= 315 mm)
          Examples: Large turbines, generators, compressors, large pumps
        - Group 2: Medium machines (15-300 kW OR 160mm <= H < 315mm) [DEFAULT]
          Examples: Industrial motors, fans, pumps, gearboxes

        **Support Type Selection Guide** (ask user if unsure):
        - "rigid": Machine on stiff foundation, horizontal orientation [DEFAULT]
          Rule: Lowest natural frequency > 1.25 x main excitation frequency
          Examples: Motors/pumps on concrete, horizontal compressors
        - "flexible": Machine on soft supports, vertical, or large turbine-generator sets
          Examples: Vertical pumps, machines on springs, large turbogenerators

        **When to ask user**:
        - If power/dimensions unknown -> use defaults (Group 2, rigid)
        - If clearly large turbine (>10 MW) -> suggest Group 1, flexible
        - If vertical machine -> suggest flexible
        - If user provides machine specs -> use guide above

        Evaluation Zones:
        - Zone A (Green): New machine condition - excellent
        - Zone B (Yellow): Acceptable for long-term unrestricted operation
        - Zone C (Orange): Unsatisfactory - limited operation, plan maintenance
        - Zone D (Red): Sufficient severity to cause damage - immediate action

        **Parameter discipline (no guessing):**
        - Sampling rate: explicit parameter > companion metadata file >
          structured error. There is NO silent default rate.
        - Signal unit: explicit parameter > companion metadata 'signal_unit'
          field > structured error. The unit is NEVER inferred from signal
          amplitude \u2014 a wrong unit completely invalidates ISO 20816-3 results.

        Args:
            signal_file: Name of the CSV file in data/signals/
            sampling_rate: Sampling frequency in Hz. If None, it is read from
                the companion _metadata.json; if neither is available the tool
                raises ValueError.
            machine_group: Machine group 1 (large) or 2 (medium) (default: 2 - medium)
            support_type: 'rigid' or 'flexible' (default: 'rigid')
            operating_speed_rpm: Operating speed in RPM (optional, for frequency range selection)
            signal_unit: DECLARED signal unit - 'g' or 'm/s2' (acceleration) or
                'mm/s' or 'm/s' (velocity). If None, it is read from the
                companion metadata 'signal_unit' field; if neither declares a
                recognized unit the tool raises ValueError naming how to
                declare it.

        Returns:
            ISO20816Result with evaluation zone, severity level, and recommendations

        Raises:
            FileNotFoundError: If the signal file does not exist.
            ValueError: If the sampling rate or the signal unit is not
                declared anywhere (parameter or metadata), if the declared
                unit is invalid, or if the sampling rate cannot cover the
                ISO evaluation band (Nyquist < 1 kHz).

        Example:
            await evaluate_iso_20816(
                ctx,
                "motor_vibration.csv",
                sampling_rate=10000,
                machine_group=2,
                support_type="rigid",
                operating_speed_rpm=1500,
                signal_unit="g"  # Explicitly declare: 'g'/'m/s2' or 'mm/s'/'m/s'
            )
        """
        from ..iso10816 import assess_severity_raw

        # Load signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

        # Companion metadata (single read for sampling_rate + signal_unit)
        metadata_file = filepath.parent / (filepath.stem + "_metadata.json")
        metadata: dict = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Sampling rate: explicit parameter > metadata > structured error.
        if sampling_rate is not None:
            if ctx:
                await ctx.info(f"Using provided sampling_rate = {sampling_rate} Hz")
        elif metadata.get('sampling_rate') is not None:
            sampling_rate = metadata['sampling_rate']
            if ctx:
                await ctx.info(f"Using sampling_rate = {sampling_rate} Hz from {metadata_file.name}")
        else:
            raise ValueError(
                f"No sampling rate for '{signal_file}' \u2014 pass the sampling_rate "
                f"parameter explicitly or add a 'sampling_rate' field to "
                f"'{metadata_file.name}'. ISO 20816-3 evaluation never proceeds "
                f"on a guessed rate."
            )

        # Signal unit: explicit parameter > metadata > structured error.
        # NEVER guessed from signal amplitude.
        if signal_unit is not None:
            unit = normalize_signal_unit(signal_unit)
            if unit is None:
                raise ValueError(
                    f"Invalid signal_unit '{signal_unit}' \u2014 declare one of "
                    f"{list(VALID_SIGNAL_UNITS)} ('g'/'m/s2' for acceleration, "
                    f"'mm/s'/'m/s' for velocity)."
                )
            if ctx:
                await ctx.info(f"Signal unit '{unit}' (declared via parameter)")
        else:
            unit = normalize_signal_unit(metadata.get('signal_unit'))
            if unit is not None and ctx:
                await ctx.info(f"Signal unit '{unit}' (declared in {metadata_file.name})")
        if unit is None:
            raise ValueError(
                f"Signal unit not declared for '{signal_file}' \u2014 ISO 20816-3 "
                f"severity requires a declared unit and never guesses it from "
                f"amplitude. Pass signal_unit='g'|'m/s2'|'mm/s'|'m/s' or add a "
                f"'signal_unit' field to '{metadata_file.name}'."
            )

        # Notify about machine parameters
        if ctx:
            await ctx.info(f"Machine parameters: Group {machine_group} ({'Large' if machine_group == 1 else 'Medium'}), Support '{support_type}'")
            await ctx.info(f"   If incorrect, provide machine_group and support_type parameters")

        # ========================================================================
        # DELEGATE MATH to iso10816.assess_severity_raw (single source of truth)
        # ========================================================================
        result = assess_severity_raw(
            signal=signal_data,
            fs=sampling_rate,
            machine_group=machine_group,
            support_type=support_type,
            signal_unit=unit,
            operating_speed_rpm=operating_speed_rpm,
        )

        # Post-computation ctx messages
        if ctx:
            if result["unit_conversion_performed"]:
                await ctx.info(
                    f"Acceleration ({unit}) integrated to velocity: "
                    f"{result['rms_velocity_mm_s']:.2f} mm/s RMS"
                )
            elif unit == 'mm/s':
                await ctx.info(f"Signal already in velocity (mm/s) - no conversion needed")

        # Build zone description with conversion notice
        zone_desc = result["zone_description"]
        if result["unit_conversion_performed"]:
            zone_desc = f"SIGNAL CONVERTED: Acceleration ({unit}) -> Velocity (mm/s). {zone_desc}"

        return ISO20816Result(
            rms_velocity=result["rms_velocity_mm_s"],
            machine_group=machine_group,
            support_type=support_type.lower(),
            zone=result["zone"],
            zone_description=zone_desc,
            severity_level=result["severity_level"],
            color_code=result["color_code"],
            boundary_ab=result["boundaries"]["AB"],
            boundary_bc=result["boundaries"]["BC"],
            boundary_cd=result["boundaries"]["CD"],
            frequency_range=result["frequency_range"],
            operating_speed_rpm=operating_speed_rpm,
            threshold_provenance=result["threshold_provenance"],
        )

    # ------------------------------------------------------------------
    # ISO 20816 chart
    # ------------------------------------------------------------------

    @mcp.tool()
    async def plot_iso_20816_chart(
        filename: str,
        sampling_rate: float,
        machine_group: int = 1,
        support_type: str = "rigid",
        operating_speed_rpm: Optional[float] = None,
        signal_unit: Optional[str] = None,
        ctx: Context | None = None
    ) -> str:
        """
        Generate visual chart showing ISO 20816-3 zone position for the analyzed signal.

        Creates an interactive HTML plot with:
        - Horizontal bar chart showing zones A/B/C/D with boundaries
        - Marker indicating actual RMS velocity position
        - Color-coded zones (green/yellow/orange/red)
        - Zone descriptions

        Args:
            filename: Name of the signal file
            sampling_rate: Sampling frequency (Hz)
            machine_group: 1 (large >300kW) or 2 (medium 15-300kW)
            support_type: 'rigid' or 'flexible'
            operating_speed_rpm: Operating speed in RPM (optional)
            signal_unit: DECLARED signal unit ('g', 'm/s2', 'mm/s', 'm/s').
                If None, it must be declared in the companion metadata —
                units are never guessed from amplitude.
            ctx: MCP context

        Returns:
            Path to generated HTML file with ISO chart

        Raises:
            ValueError: If the signal unit is not declared anywhere
                (parameter or metadata).
        """
        if ctx:
            await ctx.info(f"Evaluating ISO 20816-3 for {filename}...")

        # First, perform ISO evaluation
        iso_result = await evaluate_iso_20816(
            ctx=ctx,
            signal_file=filename,
            sampling_rate=sampling_rate,
            machine_group=machine_group,
            support_type=support_type,
            operating_speed_rpm=operating_speed_rpm,
            signal_unit=signal_unit,
        )

        # Create figure
        fig = go.Figure()

        # Zone boundaries
        boundaries = [0, iso_result.boundary_ab, iso_result.boundary_bc, iso_result.boundary_cd, iso_result.boundary_cd * 1.3]
        zone_names = ["Zone A<br>(Good)", "Zone B<br>(Acceptable)", "Zone C<br>(Unsatisfactory)", "Zone D<br>(Unacceptable)"]
        zone_colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]  # green, yellow, orange, red

        # Add horizontal bars for each zone
        for i in range(4):
            fig.add_trace(go.Bar(
                y=[f"ISO 20816-3<br>Group {machine_group}<br>{support_type.title()}"],
                x=[boundaries[i+1] - boundaries[i]],
                base=boundaries[i],
                orientation='h',
                name=zone_names[i],
                marker=dict(color=zone_colors[i], opacity=0.7),
                text=[zone_names[i]],
                textposition='inside',
                hovertemplate=f"{zone_names[i]}<br>Range: {boundaries[i]:.2f} - {boundaries[i+1]:.2f} mm/s<extra></extra>"
            ))

        # Add marker for actual value
        fig.add_trace(go.Scatter(
            x=[iso_result.rms_velocity],
            y=[f"ISO 20816-3<br>Group {machine_group}<br>{support_type.title()}"],
            mode='markers+text',
            name='Measured RMS',
            marker=dict(
                symbol='diamond',
                size=20,
                color='black',
                line=dict(width=2, color='white')
            ),
            text=[f'<b>{iso_result.rms_velocity:.2f} mm/s</b>'],
            textposition="top center",
            textfont=dict(size=14, color='black'),
            hovertemplate=f"Measured: {iso_result.rms_velocity:.2f} mm/s<br>Zone: {iso_result.zone}<br>{iso_result.severity_level}<extra></extra>"
        ))

        # Layout
        max_x = boundaries[-1]
        fig.update_layout(
            title=dict(
                text=f"ISO 20816-3 Evaluation: {filename}<br>" +
                     f"<span style='font-size:14px'>RMS Velocity: {iso_result.rms_velocity:.2f} mm/s | " +
                     f"Zone <b>{iso_result.zone}</b> ({iso_result.severity_level})</span>",
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="RMS Velocity (mm/s)",
                range=[0, max_x],
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="",
                showticklabels=True
            ),
            barmode='stack',
            height=400,
            width=1000,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text=f"Zone Boundaries: A/B={iso_result.boundary_ab:.1f} | B/C={iso_result.boundary_bc:.1f} | C/D={iso_result.boundary_cd:.1f} mm/s",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=11, color="gray")
                )
            ]
        )

        # Save HTML to reports directory
        safe_name = Path(filename).stem.replace('/', '_').replace('\\', '_')
        output_file = REPORTS_DIR / f"plot_iso_{safe_name}.html"
        fig.write_html(str(output_file))

        if ctx:
            await ctx.info(f"ISO chart saved to {output_file.name}")
            await ctx.info(f"To view report metadata: list_html_reports() or get_report_info()")

        return f"ISO 20816-3 chart saved to: {output_file}\nUse list_html_reports() to see all reports, or open file in browser"

    # ------------------------------------------------------------------
    # ML anomaly detection – training
    # ------------------------------------------------------------------

    @mcp.tool()
    async def train_anomaly_model(
        healthy_signal_files: list[str],
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
        model_type: str = "OneClassSVM",
        pca_variance: float = 0.95,
        fault_signal_files: Optional[list[str]] = None,
        healthy_validation_files: Optional[list[str]] = None,
        model_name: str = "anomaly_model",
        ctx: Context = None
    ) -> AnomalyModelResult:
        """
        Train ML-based anomaly detection model on healthy data (UNSUPERVISED/SEMI-SUPERVISED).

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
        - If healthy_validation_files provided: Use those explicitly (no split)
        - If healthy_validation_files NOT provided: Automatic 80/20 split of training data
        - If fault_signal_files provided: Enable semi-supervised mode (hyperparameter tuning)

        Args:
            healthy_signal_files: List of CSV files with healthy machine data (for training)
            sampling_rate: Sampling frequency in Hz (auto-detect from metadata if None)
            segment_duration: Segment duration in seconds (default: 0.1)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            model_type: 'OneClassSVM' or 'LocalOutlierFactor' (default: 'OneClassSVM')
            pca_variance: Cumulative variance to explain with PCA (default: 0.95)
            fault_signal_files: Optional list of fault signals for HYPERPARAMETER TUNING (semi-supervised)
            healthy_validation_files: Optional list of healthy signals for validation (specificity check).
                                      If not provided, 20% of training data will be used.
            model_name: Name for saved model files (default: 'anomaly_model')
            ctx: MCP context for progress/logging

        Returns:
            AnomalyModelResult with model paths and performance metrics
        """
        # Fail fast on an unsafe model_name before doing any expensive work or
        # touching the filesystem (the write happens in step 6).
        validate_name_component(model_name, kind="model_name")

        if ctx:
            await ctx.info(f"Training {model_type} model on {len(healthy_signal_files)} healthy signals...")

        # Step 1: Extract features from all healthy signals
        all_features, detected_rates = await _extract_features_from_files(
            healthy_signal_files, sampling_rate, segment_duration, overlap_ratio,
            strict=True, ctx=ctx
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
            if fault_signal_files:
                # SEMI-SUPERVISED MODE: Train on healthy, tune hyperparameters with validation (healthy + fault)
                if ctx:
                    await ctx.info("Training in SEMI-SUPERVISED mode")
                    await ctx.info("- Training: Healthy data only (unsupervised)")
                    await ctx.info("- Hyperparameter tuning: Using validation set (healthy + fault)")
                    await ctx.info("Evaluating hyperparameter grid...")

                # Prepare validation features for fault signals
                X_fault = await _extract_and_transform_validation_features(
                    fault_signal_files, sampling_rate, segment_duration, overlap_ratio,
                    scaler, pca, strict=False, ctx=ctx
                )

                # Prepare validation features for healthy signals
                X_healthy_val = None
                if healthy_validation_files:
                    X_healthy_val = await _extract_and_transform_validation_features(
                        healthy_validation_files, sampling_rate, segment_duration, overlap_ratio,
                        scaler, pca, strict=False, ctx=ctx
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
            if fault_signal_files:
                # SEMI-SUPERVISED MODE with LOF
                if ctx:
                    await ctx.info("Training LOF in SEMI-SUPERVISED mode")
                    await ctx.info("- Training: Healthy data only (unsupervised)")
                    await ctx.info("- Hyperparameter tuning: Using validation set")

                # Prepare validation features for fault signals
                X_fault = await _extract_and_transform_validation_features(
                    fault_signal_files, sampling_rate, segment_duration, overlap_ratio,
                    scaler, pca, strict=False, ctx=ctx
                )

                # Prepare healthy validation features
                X_healthy_val = None
                if healthy_validation_files:
                    X_healthy_val = await _extract_and_transform_validation_features(
                        healthy_validation_files, sampling_rate, segment_duration, overlap_ratio,
                        scaler, pca, strict=False, ctx=ctx
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

        if fault_signal_files or healthy_validation_files:
            # Part A: Validate on HEALTHY data
            # Two options:
            # 1. User provides explicit healthy_validation_files -> Use those
            # 2. User doesn't provide -> Auto-split training data 80/20

            if healthy_validation_files:
                # Option 1: User provided explicit healthy validation files
                if ctx:
                    await ctx.info(f"Using {len(healthy_validation_files)} explicitly provided healthy validation files")

                # Extract and transform features from validation files
                X_pca_healthy_val = await _extract_and_transform_validation_features(
                    healthy_validation_files, sampling_rate, segment_duration, overlap_ratio,
                    scaler, pca, strict=True, ctx=ctx
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
            if fault_signal_files:
                X_fault_pca = await _extract_and_transform_validation_features(
                    fault_signal_files, sampling_rate, segment_duration, overlap_ratio,
                    scaler, pca, strict=True, ctx=ctx
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

        # Save metadata
        metadata = {
            'model_type': model_type,
            'training_mode': 'supervised' if fault_signal_files else 'unsupervised',
            'feature_names': list(features_df.columns),
            'num_features_original': X_train.shape[1],
            'num_features_pca': X_pca.shape[1],
            'pca_variance': float(pca.explained_variance_ratio_.sum()),
            'best_params': best_params,
            'sampling_rate': sampling_rate if sampling_rate is not None else 'per_file',
            'sampling_rates_detected': detected_rates if sampling_rate is None else None,
            'segment_duration': segment_duration,
            'overlap_ratio': overlap_ratio,
            'multi_rate_training': sampling_rate is None,
            'validation_with_faults': fault_signal_files is not None,
            'num_validation_files': len(fault_signal_files) if fault_signal_files else 0
        }

        metadata_path = _model_paths.metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if ctx:
            await ctx.info(f"Model saved to {model_path}")
            await ctx.info(f"Scaler saved to {scaler_path}")
            await ctx.info(f"PCA saved to {pca_path}")

        return AnomalyModelResult(
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

    @mcp.tool()
    async def predict_anomalies(
        signal_file: str,
        model_name: str = "anomaly_model",
        ctx: Context = None
    ) -> AnomalyPredictionResult:
        """
        Predict anomalies in new signal using trained model.

        Applies the complete pipeline:
        1. Segment signal
        2. Extract features
        3. Apply scaler (from training)
        4. Apply PCA (from training)
        5. Predict with trained model
        6. Calculate anomaly ratio and overall health

        Args:
            signal_file: Name of CSV file to analyze
            model_name: Name of trained model (default: 'anomaly_model')
            ctx: MCP context for progress/logging

        Returns:
            AnomalyPredictionResult with predictions and health assessment
        """
        if ctx:
            await ctx.info(f"Predicting anomalies in {signal_file}...")

        # Validate model_name and contain every derived path (single source of
        # truth shared with the training/PCA/pipeline model-load sites).
        _model_paths = resolve_model_paths(MODELS_DIR, model_name)
        model_path = _model_paths.model
        scaler_path = _model_paths.scaler
        pca_path = _model_paths.pca
        metadata_path = _model_paths.metadata

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train model first.")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

        # Extract features
        sampling_rate_val = metadata['sampling_rate']
        if isinstance(sampling_rate_val, str):
            # Model was trained with per-file metadata detection; read from signal metadata
            meta_path = get_metadata_path(signal_file)
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    signal_meta = json.load(f)
                meta_rate = signal_meta.get('sampling_rate')
                if meta_rate is None:
                    raise ValueError(
                        f"No 'sampling_rate' field in {meta_path.name} — add it "
                        f"to the metadata file; prediction never proceeds on a "
                        f"guessed rate."
                    )
                sampling_rate_val = float(meta_rate)
            else:
                raise ValueError(
                    f"Model was trained with per-file sampling rates but no metadata found for {signal_file}. "
                    "Please provide a _metadata.json file alongside the signal CSV."
                )
        sampling_rate_val = float(sampling_rate_val)
        segment_duration = metadata['segment_duration']
        overlap_ratio = metadata['overlap_ratio']

        segment_length_samples = int(segment_duration * sampling_rate_val)
        hop_length = int(segment_length_samples * (1 - overlap_ratio))

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

        # Get anomaly scores if available
        anomaly_scores = None
        if hasattr(model, 'decision_function'):
            anomaly_scores = model.decision_function(X_pca).tolist()

        # Calculate statistics
        anomaly_count = int(np.sum(predictions == -1))
        anomaly_ratio = float(anomaly_count / len(predictions))

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
            num_segments=len(predictions),
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_ratio,
            predictions=predictions.tolist(),
            anomaly_scores=anomaly_scores,
            overall_health=overall_health,
        )

    # ------------------------------------------------------------------
    # Machine documentation tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def list_machine_manuals(ctx: Context | None = None) -> list[dict[str, Any]]:
        """
        List all available machine manuals in resources/machine_manuals/.

        Returns list of PDFs and text files with filename, size, and modification date.
        Use this to see what manuals are available before extracting specs.

        **IMPORTANT - LLM Usage Guidelines:**
        - This tool returns ONLY the list of available files
        - DO NOT make assumptions about manual content without reading it
        - DO NOT infer specifications without using extract_manual_specs() or read_manual_excerpt()
        - ALWAYS use the returned filenames exactly as-is when calling other tools
        - If user asks about manual content, use read_manual_excerpt() or extract_manual_specs()

        Returns:
            List of dictionaries with manual information

        Example:
            >>> manuals = list_machine_manuals()
            >>> print(f"Found {len(manuals)} manuals")
            >>> for m in manuals:
            ...     print(f"- {m['filename']}: {m['size_mb']:.2f} MB")
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

    @mcp.tool()
    async def extract_manual_specs(
        manual_filename: str,
        use_cache: bool = True,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Extract machine specifications from equipment manual PDF.

        Automatically extracts:
        - Bearing designations (e.g., SKF 6205, FAG NU2205)
        - Operating speeds (RPM values)
        - Power ratings (kW, HP, MW)
        - Text excerpt for LLM context

        Results are cached for fast repeated access.

        **IMPORTANT - LLM Usage Guidelines:**
        - This tool returns ONLY data extracted from the manual text
        - DO NOT add information not present in the extraction results
        - DO NOT make assumptions about missing specifications
        - If a specification is not in the results, tell the user it was not found
        - ALWAYS base your response exclusively on the returned dictionary
        - If user needs more detail, suggest using read_manual_excerpt() to read full text
        - DO NOT invent bearing geometries, frequencies, or other technical data

        **WORKFLOW for missing bearing geometry:**
        1. Check if bearing geometry is in extraction results (rare in manuals)
        2. If not found, use search_bearing_catalog(bearing_designation) tool
        3. If bearing not in catalog, ask user to provide:
           - All geometric parameters (num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg)
           - OR upload manufacturer catalog to bearing_catalogs/ directory

        Args:
            manual_filename: PDF filename in resources/machine_manuals/
            use_cache: Use cached extraction if available (default: True)
            ctx: MCP context

        Returns:
            Dictionary with extracted specifications and text excerpt

        Example:
            >>> specs = extract_manual_specs("pump_XYZ_manual.pdf")
            >>> print(f"Bearings: {specs['bearings']}")
            >>> print(f"RPM: {specs['rpm_values']}")
            >>> print(f"Power: {specs['power_ratings']}")
        """
        if ctx:
            await ctx.info(f"Extracting specifications from: {manual_filename}")

        manual_path = safe_resolve(RESOURCES_DIR / "machine_manuals", manual_filename)

        if not manual_path.exists():
            raise FileNotFoundError(
                f"Manual not found: {manual_filename}\n"
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

    @mcp.tool()
    async def calculate_bearing_characteristic_frequencies(
        num_balls: int,
        ball_diameter_mm: float,
        pitch_diameter_mm: float,
        contact_angle_deg: float = 0.0,
        shaft_speed_rpm: float = 1500.0,
        ctx: Context | None = None
    ) -> dict[str, float]:
        """
        Calculate bearing characteristic frequencies from geometry.

        Uses the standard rolling-element bearing kinematic formulas (see
        Randall & Antoni 2011, "Rolling element bearing diagnostics - A
        tutorial", Mech. Systems and Signal Processing 25(2), 485-520).
        Essential for bearing fault diagnosis when you know bearing geometry
        but don't have pre-calculated frequencies.

        **IMPORTANT - LLM Usage Guidelines:**
        - This tool REQUIRES exact bearing geometry parameters
        - DO NOT guess or estimate bearing geometry if not provided
        - DO NOT use "typical" or "standard" values without user confirmation
        - If geometry is unknown, tell user to:
          1. Check manual using read_manual_excerpt()
          2. Look up bearing in manufacturer catalog (e.g., SKF, FAG, NSK)
          3. Use search_bearing_catalog() if bearing designation is known
          4. Measure the bearing physically if necessary
        - ONLY calculate with geometry explicitly provided by user or found in manual
        - DO NOT make assumptions about contact angle (use 0 deg if unknown and inform user)

        Args:
            num_balls: Number of rolling elements (Z)
            ball_diameter_mm: Ball/roller diameter (Bd) in mm
            pitch_diameter_mm: Pitch circle diameter (Pd) in mm
            contact_angle_deg: Contact angle (alpha) in degrees (0 deg for deep groove ball bearings)
            shaft_speed_rpm: Shaft rotation speed in RPM
            ctx: MCP context

        Returns:
            Dictionary with BPFO, BPFI, BSF, FTF in Hz

        Example:
            >>> # 6205 geometry (CWRU Bearing Data Center) at 1797 RPM
            >>> freqs = calculate_bearing_characteristic_frequencies(
            ...     num_balls=9,
            ...     ball_diameter_mm=7.94,
            ...     pitch_diameter_mm=39.04,
            ...     contact_angle_deg=0.0,
            ...     shaft_speed_rpm=1797
            ... )
            >>> print(f"BPFO: {freqs['BPFO']:.2f} Hz")
            BPFO: 107.36 Hz

        Common bearing geometries:
        - Deep groove ball bearings: contact_angle = 0 deg
        - Angular contact bearings: contact_angle = 15-40 deg
        - Cylindrical roller bearings: contact_angle = 0 deg
        """
        if ctx:
            await ctx.info(f"Calculating bearing frequencies for {num_balls} balls at {shaft_speed_rpm} RPM")

        freqs = calculate_bearing_frequencies(
            num_balls=num_balls,
            ball_diameter_mm=ball_diameter_mm,
            pitch_diameter_mm=pitch_diameter_mm,
            contact_angle_deg=contact_angle_deg,
            shaft_speed_rpm=shaft_speed_rpm
        )

        if ctx:
            await ctx.info(f"BPFO (outer race): {freqs['BPFO']:.2f} Hz")
            await ctx.info(f"BPFI (inner race): {freqs['BPFI']:.2f} Hz")
            await ctx.info(f"BSF (ball spin): {freqs['BSF']:.2f} Hz")
            await ctx.info(f"FTF (cage): {freqs['FTF']:.2f} Hz")

        return freqs

    @mcp.tool()
    async def read_manual_excerpt(
        manual_filename: str,
        max_pages: int = 10,
        ctx: Context | None = None
    ) -> str:
        """
        Read text excerpt from machine manual (PDF or TXT).

        Useful for providing context to LLM for questions about
        specific machine parameters, maintenance procedures, etc.

        **Token Warning**: Reading many pages can consume significant tokens.
        Start with max_pages=10 and increase if needed.

        **IMPORTANT - LLM Usage Guidelines:**
        - This tool returns ONLY the text extracted from the manual
        - Base your answers EXCLUSIVELY on the returned text
        - DO NOT add information not present in the extracted text
        - If information is not found in the text, clearly state "Not found in manual"
        - DO NOT make assumptions or fill gaps with general knowledge
        - If user needs more pages, suggest increasing max_pages parameter
        - ALWAYS cite the manual when answering: "According to the manual..."

        Args:
            manual_filename: Manual filename in resources/machine_manuals/ (PDF or TXT)
            max_pages: Maximum number of pages to extract (default: 10, ignored for TXT files)
            ctx: MCP context

        Returns:
            Extracted text from manual

        Example:
            >>> text = read_manual_excerpt("pump_manual.pdf", max_pages=5)
            >>> # LLM can now answer: "What bearings are recommended for this pump?"
        """
        if ctx:
            await ctx.info(f"Reading from: {manual_filename}")

        manual_path = safe_resolve(RESOURCES_DIR / "machine_manuals", manual_filename)

        if not manual_path.exists():
            raise FileNotFoundError(f"Manual not found: {manual_filename}")

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

    @mcp.tool()
    async def search_bearing_catalog(
        bearing_designation: str,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Search for bearing specifications in local bearing catalogs.

        This is a FALLBACK tool. LLM should use this ONLY when:
        1. Bearing designation found in machine manual
        2. Bearing geometry NOT found in machine manual
        3. Need geometry to calculate characteristic frequencies

        **IMPORTANT - LLM Usage Guidelines:**
        - Use this tool ONLY after checking machine manual first
        - DO NOT use this as primary source - manual takes precedence
        - If bearing not found here, ask user for specifications
        - DO NOT guess or estimate if bearing not in catalog
        - The catalog is small BY DESIGN: it contains only bearings whose
          geometry is traceable to a public source (each entry carries a
          mandatory `source` citation). Unverifiable entries were removed.
        - For bearings not in the catalog, tell user: "Bearing {X} not in catalog. Please provide geometry or upload manufacturer catalog to bearing_catalogs/"

        Search order:
        1. JSON catalog (common_bearings_catalog.json) — verified entries only
        2. Structured not-found payload if the designation is absent

        Args:
            bearing_designation: Bearing designation (e.g., "6205", "SKF 6205-2RS", "UER204")
            ctx: MCP context

        Returns:
            Dictionary with bearing specifications if found, None otherwise

        Example:
            >>> specs = search_bearing_catalog("SKF 6205-2RS")
            >>> print(f"Balls: {specs['num_balls']}, Diameter: {specs['ball_diameter_mm']} mm")
            Balls: 9, Diameter: 7.94 mm
        """
        if ctx:
            await ctx.info(f"Searching catalog for bearing: {bearing_designation}")

        try:
            from ..document_reader import lookup_bearing_in_catalog

            bearing_specs = lookup_bearing_in_catalog(bearing_designation)

            if bearing_specs:
                if ctx:
                    await ctx.info(f"Found {bearing_specs['designation']} in catalog (source: {bearing_specs.get('source', 'unknown')})")
                    await ctx.info(f"  Type: {bearing_specs.get('type', 'N/A')}")
                    await ctx.info(f"  Balls: {bearing_specs['num_balls']}, Ball diameter: {bearing_specs['ball_diameter_mm']} mm")
                    await ctx.info(f"  Pitch diameter: {bearing_specs['pitch_diameter_mm']} mm")
                return bearing_specs
            else:
                if ctx:
                    await ctx.warning(f"Bearing {bearing_designation} not found in catalog")
                    await ctx.warning("  LLM should ask user for bearing geometry or suggest uploading manufacturer catalog")
                from ..diagnostics.bearing_catalog import list_catalog_bearings

                available = sorted(
                    b["designation"] for b in list_catalog_bearings()
                )
                return {
                    "error": f"Bearing {bearing_designation} not found in catalog",
                    "suggestion": "Ask user for bearing geometry (num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg) or upload manufacturer catalog PDF to resources/bearing_catalogs/",
                    "catalog_contains": available
                }

        except Exception as e:
            logger.error(f"Error searching bearing catalog: {e}")
            if ctx:
                await ctx.error(f"Error searching catalog: {str(e)}")
            return {
                "error": str(e),
                "suggestion": "Check that bearing_catalogs directory exists and contains common_bearings_catalog.json"
            }

    # ------------------------------------------------------------------
    # Document search (RAG)
    # ------------------------------------------------------------------

    @mcp.tool()
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
    # Bearing diagnostics (Phase 1 -- signal_id based)
    # ------------------------------------------------------------------

    @mcp.tool()
    async def check_bearing_fault_peak_tool(
        ctx: Context,
        signal_id: str,
        bearing_id: str,
        fault_type: str,
        rpm: float,
        tolerance_pct: float = 5.0,
    ) -> BearingFaultCheckResult:
        """Check for a specific bearing fault frequency in a stored signal.

        Args:
            signal_id: ID of the stored signal.
            bearing_id: Bearing designation (e.g. '6205').
            fault_type: Which fault to check: 'BPFO', 'BPFI', 'BSF', or 'FTF'.
            rpm: Shaft speed in RPM.
            tolerance_pct: Frequency matching tolerance (default 5%).
        """
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'.")

        # Look up expected frequency
        freq_data = _compute_fault_freqs(bearing_id, rpm)
        if freq_data is None:
            raise ValueError(f"Bearing '{bearing_id}' not found in catalog.")

        fault_type_upper = fault_type.upper()
        if fault_type_upper not in freq_data:
            raise ValueError(f"Invalid fault_type '{fault_type}'. Use BPFO, BPFI, BSF, or FTF.")

        expected = freq_data[fault_type_upper]

        if ctx:
            await ctx.info(f"Checking {fault_type_upper} at {expected:.2f} Hz for bearing {bearing_id}")

        result = _check_fault_peak(
            signal=signal_data,
            fs=fs,
            expected_freq=expected,
            fault_type=fault_type_upper,
            bearing_id=bearing_id,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )

        return BearingFaultCheckResult(**result)

    @mcp.tool()
    async def check_bearing_faults_direct(
        ctx: Context,
        signal_id: str,
        bearing_id: str,
        rpm: float,
        tolerance_pct: float = 5.0,
    ) -> BearingFaultsSummary:
        """Run all bearing fault checks (BPFO, BPFI, BSF, FTF) on a stored signal.

        Args:
            signal_id: ID of the stored signal.
            bearing_id: Bearing designation (e.g. '6205').
            rpm: Shaft speed in RPM.
            tolerance_pct: Frequency matching tolerance (default 5%).
        """
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'.")

        if ctx:
            await ctx.info(f"Running all bearing fault checks for {bearing_id} at {rpm} RPM")

        result = _check_all_faults(
            signal=signal_data,
            fs=fs,
            bearing_id=bearing_id,
            rpm=rpm,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )

        return BearingFaultsSummary(
            signal_id=result["signal_id"],
            bearing_id=result["bearing_id"],
            rpm=result["rpm"],
            shaft_frequency_hz=result["shaft_frequency_hz"],
            bearing_frequencies=result["bearing_frequencies"],
            fault_checks=[BearingFaultCheckResult(**c) for c in result["fault_checks"]],
            overall_assessment=result["overall_assessment"],
            most_likely_fault=result["most_likely_fault"],
        )

    @mcp.tool()
    async def lookup_bearing_and_compute_tool(
        ctx: Context,
        bearing_type: str,
        rpm: float,
        signal_id: str,
        tolerance_pct: float = 5.0,
    ) -> BearingFaultsSummary:
        """Look up bearing, compute fault frequencies, and check signal -- all in one call.

        End-to-end bearing analysis: catalog lookup + frequency calculation +
        envelope spectrum fault detection.

        Args:
            bearing_type: Bearing designation (e.g. 'SKF 6205-2RS', '6205').
            rpm: Shaft speed in RPM.
            signal_id: ID of the stored signal.
            tolerance_pct: Frequency matching tolerance (default 5%).
        """
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'.")

        if ctx:
            await ctx.info(f"End-to-end bearing analysis: {bearing_type} at {rpm} RPM")

        result = _lookup_and_compute(
            signal=signal_data,
            fs=fs,
            bearing_type=bearing_type,
            rpm=rpm,
            signal_id=signal_id,
            tolerance_pct=tolerance_pct,
        )

        return BearingFaultsSummary(
            signal_id=result["signal_id"],
            bearing_id=result["bearing_id"],
            rpm=result["rpm"],
            shaft_frequency_hz=result["shaft_frequency_hz"],
            bearing_frequencies=result["bearing_frequencies"],
            fault_checks=[BearingFaultCheckResult(**c) for c in result["fault_checks"]],
            overall_assessment=result["overall_assessment"],
            most_likely_fault=result["most_likely_fault"],
        )

    # ------------------------------------------------------------------
    # ISO severity & diagnosis (Phase 1 -- signal_id based)
    # ------------------------------------------------------------------

    @mcp.tool()
    async def assess_vibration_severity(
        ctx: Context,
        signal_id: str,
        machine_group: Literal[1, 2] = 2,
        support_type: Literal["rigid", "flexible"] = "rigid",
        axis: str = "vertical",
    ) -> VibrationSeverityResult:
        """Assess vibration severity per ISO 20816-3 for a stored signal.

        Zone boundary values come from ISO 10816-3:2009 (four-zone A-D
        scheme; ISO 20816-3:2022 merges zones A/B) — the provenance note is
        included in the result. Scope: machines rated above 15 kW; results
        for smaller machines are not covered by these boundaries.

        Uses the signal_id pattern (load once, reference by ID). The signal
        unit must be DECLARED — via load_signal(signal_unit=...) or the
        companion _metadata.json — and is never guessed from amplitude; an
        undeclared unit is a structured error, not a silent assumption.

        Args:
            signal_id: ID of the stored signal.
            machine_group: 1 (large machines, >300 kW) or
                2 (medium machines, 15-300 kW). Default 2.
            support_type: 'rigid' or 'flexible'. Default 'rigid'.
            axis: Measurement axis (informational).

        Raises:
            ValueError: If the stored signal has no sampling rate or no
                declared unit, or if the sampling rate cannot cover the ISO
                evaluation band (Nyquist < 1 kHz).
        """
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(
                f"No sampling rate for signal '{signal_id}' — re-load with "
                f"load_signal(filepath=..., sampling_rate=...) or add a "
                f"'sampling_rate' field to the companion _metadata.json."
            )

        signal_unit = info.get("signal_unit")
        if signal_unit is None:
            raise ValueError(
                f"ISO severity refused for '{signal_id}': signal unit not "
                f"declared — units are never guessed from amplitude. Re-load "
                f"with load_signal(filepath=..., signal_unit='g'|'m/s2'|"
                f"'mm/s'|'m/s') or add a 'signal_unit' field to the companion "
                f"_metadata.json."
            )
        if ctx:
            await ctx.info(
                f"Assessing ISO severity for '{signal_id}' "
                f"(group {machine_group}, {support_type}, unit: {signal_unit})"
            )

        result = _assess_severity(
            signal=signal_data,
            fs=fs,
            machine_group=machine_group,
            support_type=support_type,
            axis=axis,
            signal_unit=signal_unit,
        )
        result["signal_id"] = signal_id

        if ctx:
            await ctx.info(f"Zone {result['zone']} ({result['severity_level']}): {result['rms_velocity_mm_s']:.2f} mm/s")

        return VibrationSeverityResult(**result)

    @mcp.tool()
    async def diagnose_vibration_tool(
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
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(
                f"No sampling rate for signal '{signal_id}' — re-load with "
                f"load_signal(filepath=..., sampling_rate=...) or add a "
                f"'sampling_rate' field to the companion _metadata.json."
            )

        # None = undeclared: pipeline degrades to a refused ISO block while
        # the other diagnosis blocks still run (no unit guessing).
        signal_unit = info.get("signal_unit")
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
