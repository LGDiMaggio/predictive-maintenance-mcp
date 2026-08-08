"""MCP tools for report generation and visualization (ISO 13374 Block 6)."""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, sosfiltfilt
from mcp.server.mcpserver import MCPServer, Context

from ..config import MODELS_DIR, REPORTS_DIR
from ..models import StoredSignalInfo
from ..report_generator import (
    save_fft_report,
    save_envelope_report,
    save_iso_report,
    read_report_metadata,
    list_reports,
    save_diagnostic_report_docx,
    timestamped_report_name,
)
from ..signal_processing.features import extract_time_domain_features
from ._utils import resolve_model_paths, resolve_signal

# Canonical modular twin of the ISO evaluation tool (module-level function
# since U6) — replaces the former runtime import from the deprecated monolith.
from .diagnostics_tools import assess_severity
from ..signal_processing.spectral import validate_bandpass_band

logger = logging.getLogger(__name__)


def _companion_metadata(info: StoredSignalInfo) -> dict:
    """Read the companion _metadata.json of a stored signal's source file.

    Used only for OPTIONAL extras (e.g. reference bearing frequencies) —
    sampling rate and unit always come from the repository entry itself.
    """
    src = Path(info.filepath)
    meta_path = src.parent / f"{src.stem}_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # pragma: no cover - corrupt metadata is rare
            logger.warning(f"Error reading metadata {meta_path}: {e}")
    return {}



async def generate_diagnostic_report_docx(
    signal_id: str,
    sections: dict[str, Any],
    title: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
        Generate a structured Word (.docx) diagnostic report for a stored signal.

        Requires: ``pip install predictive-maintenance-mcp[docx]``

        ``sections`` is a dict whose keys define what to include (all optional):
          - statistics:           dict  (RMS, Kurtosis, Crest Factor …)
          - fft_peaks:            list  [{frequency, magnitude_db, note}, …]
          - envelope_peaks:       list  [{frequency, magnitude_db, match}, …]
          - bearing_frequencies:  dict  {BPFO, BPFI, BSF, FTF}
          - iso:                  dict  (mapped from assess_severity output)
          - diagnosis:            str   (free-text diagnostic summary)

        Args:
            signal_id: ID of the stored signal (from load_signal); used for
                the report title / filename.
            sections: Content sections to include (see above)
            title: Optional custom report title
            ctx: MCP context

        Returns:
            Dictionary with file_path, file_name, and per-section summary.

        Raises:
            ValueError: If the signal_id is not loaded, or python-docx is
                not installed.
        """
    # Validate the handle: reports are only produced for loaded signals.
    resolve_signal(signal_id, require_sampling_rate=False)

    if ctx:
        await ctx.info(f"Generating DOCX diagnostic report for '{signal_id}'")

    # Raises ValueError when the optional python-docx dependency is missing
    # (error contract: failures raise, never error-shaped dicts).
    result = save_diagnostic_report_docx(signal_id, sections, title=title)

    if ctx:
        await ctx.info(f"DOCX report saved: {result['file_name']}")

    return result

async def plot_signal(
    signal_id: str,
    time_range: Optional[list[float]] = None,
    show_statistics: bool = True,
    title: Optional[str] = None,
    ctx: Context | None = None
) -> str:
    """
        Generate interactive time-domain plot for a stored signal.

        Creates an interactive HTML plot showing the signal in the time domain.
        Useful for inspecting signal quality, identifying anomalies, and
        visualizing transients. Requires the signal loaded via load_signal()
        first; the sampling rate comes from the stored signal metadata.

        Args:
            signal_id: ID of the stored signal (from load_signal).
            time_range: [start_time, end_time] in seconds to zoom on a portion (optional)
            show_statistics: Show RMS, peak levels as horizontal lines (default: True)
            title: Custom plot title (optional)
            ctx: MCP context for progress/logging

        Returns:
            Path to generated HTML file

        Raises:
            ValueError: If the signal_id is not loaded, or the stored signal
                has no sampling rate.

        Example:
            plot_signal(
                "bearing_signal",
                time_range=[0.1, 0.3],  # Zoom on 100-300 ms
                show_statistics=True
            )
        """
    if ctx:
        await ctx.info(f"Generating time-domain plot for '{signal_id}'...")

    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate

    # Time array
    n = len(signal_data)
    time = np.arange(n) / sampling_rate

    # Apply time range filter if specified
    if time_range:
        mask = (time >= time_range[0]) & (time <= time_range[1])
        time_plot = time[mask]
        signal_plot = signal_data[mask]
    else:
        time_plot = time
        signal_plot = signal_data

    # Calculate statistics
    rms = np.sqrt(np.mean(signal_plot**2))
    peak_pos = np.max(signal_plot)
    peak_neg = np.min(signal_plot)
    mean_val = np.mean(signal_plot)

    # Create plot
    fig = go.Figure()

    # Main signal
    fig.add_trace(go.Scatter(
        x=time_plot,
        y=signal_plot,
        mode='lines',
        name='Signal',
        line=dict(color='blue', width=1),
        hovertemplate='Time: %{x:.4f} s<br>Amplitude: %{y:.4f}<extra></extra>'
    ))

    # Add statistical reference lines if requested
    if show_statistics:
        # RMS lines
        fig.add_trace(go.Scatter(
            x=[time_plot[0], time_plot[-1]],
            y=[rms, rms],
            mode='lines',
            name=f'RMS (+{rms:.4f})',
            line=dict(color='green', width=2, dash='dash'),
            hovertemplate=f'RMS: {rms:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[time_plot[0], time_plot[-1]],
            y=[-rms, -rms],
            mode='lines',
            name=f'RMS (−{rms:.4f})',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False,
            hovertemplate=f'RMS: -{rms:.4f}<extra></extra>'
        ))

        # Peak lines
        fig.add_trace(go.Scatter(
            x=[time_plot[0], time_plot[-1]],
            y=[peak_pos, peak_pos],
            mode='lines',
            name=f'Peak (+{peak_pos:.4f})',
            line=dict(color='red', width=1, dash='dot'),
            hovertemplate=f'Peak: {peak_pos:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[time_plot[0], time_plot[-1]],
            y=[peak_neg, peak_neg],
            mode='lines',
            name=f'Peak (−{abs(peak_neg):.4f})',
            line=dict(color='red', width=1, dash='dot'),
            hovertemplate=f'Peak: {peak_neg:.4f}<extra></extra>'
        ))

        # Mean line
        if abs(mean_val) > 1e-6:  # Only show if mean is significant
            fig.add_trace(go.Scatter(
                x=[time_plot[0], time_plot[-1]],
                y=[mean_val, mean_val],
                mode='lines',
                name=f'Mean ({mean_val:.4f})',
                line=dict(color='orange', width=1, dash='dashdot'),
                hovertemplate=f'Mean: {mean_val:.4f}<extra></extra>'
            ))

    # Layout
    plot_title = title or f"Time-Domain Signal - {signal_id}"
    duration = time_plot[-1] - time_plot[0]

    fig.update_layout(
        title=plot_title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        showlegend=True,
        annotations=[
            dict(
                text=f"Duration: {duration:.3f} s | Samples: {len(signal_plot)} | Fs: {sampling_rate} Hz",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        ]
    )

    # Save HTML to reports directory (timestamped: runs never overwrite)
    output_file = REPORTS_DIR / timestamped_report_name("plot_signal", signal_id)
    fig.write_html(str(output_file))

    if ctx:
        await ctx.info(f"Plot saved to {output_file.name}")
        await ctx.info(
            "To view report metadata: list_html_reports(file_name=...)"
        )

    return f"Interactive plot saved to: {output_file}\nUse list_html_reports() to see all reports, or open file in browser"

async def generate_fft_report(
    signal_id: str,
    max_freq: float = 5000.0,
    num_peaks: int = 15,
    rpm: Optional[float] = None,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Generate an interactive FFT spectrum report (HTML) for a stored signal.

        Saves a self-contained Plotly HTML report (spectrum in dB, automatic
        peak detection, harmonic labels) to the reports/ directory with a
        timestamped filename — consecutive runs produce distinct files.
        Requires the signal loaded via load_signal() first; the sampling
        rate comes from the stored signal metadata.

        Args:
            signal_id: ID of the stored signal (from load_signal).
            max_freq: Maximum frequency to display (Hz). Default 5000 Hz
            num_peaks: Number of peaks to detect and label. Default 15
            rpm: Optional shaft speed in RPM — peaks at integer multiples
                of rpm/60 Hz are labeled as 1x/2x/... harmonics.
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Raises:
            ValueError: If the signal_id is not loaded, or the stored signal
                has no sampling rate.
        """
    if ctx:
        await ctx.info(f"Generating FFT report for '{signal_id}'...")

    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate

    # Perform FFT
    N = len(signal_data)
    window = np.hamming(N)
    signal_windowed = signal_data * window

    fft_values = fft(signal_windowed)
    frequencies = fftfreq(N, 1/sampling_rate)

    # Positive frequencies only
    positive_idx = frequencies > 0
    frequencies = frequencies[positive_idx]
    magnitudes = 2.0 * np.abs(fft_values[positive_idx]) / N

    # Generate and save report (signal_id is the report's signal label).
    # rpm is user-facing; the engine labels harmonics in Hz.
    result = save_fft_report(
        signal_file=signal_id,
        sampling_rate=sampling_rate,
        frequencies=frequencies,
        magnitudes=magnitudes,
        signal_data=signal_data,
        max_freq=max_freq,
        num_peaks=num_peaks,
        rotation_freq=(rpm / 60.0) if rpm is not None else None
    )

    if ctx:
        await ctx.info(result['message'])
        await ctx.info(f"Report location: {result['file_path']}")

    return result

async def generate_envelope_report(
    signal_id: str,
    filter_low: float = 500.0,
    filter_high: Optional[float] = None,
    max_freq: float = 500.0,
    num_peaks: int = 15,
    bearing_freqs: Optional[dict[str, float]] = None,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Generate professional envelope analysis report (HTML) for a stored signal.

        Generates a professional HTML report file instead of inline content.
        Saves to reports/ directory. Requires the signal loaded via
        load_signal() first; the sampling rate comes from the stored signal
        metadata. Reference bearing frequencies (BPFO/BPFI/BSF/FTF) can be
        passed explicitly or, if omitted, are read from the source file's
        companion _metadata.json when present.

        Args:
            signal_id: ID of the stored signal (from load_signal).
            filter_low: Bandpass filter low cutoff (Hz). Default 500 Hz
            filter_high: Bandpass filter high cutoff (Hz). Default (None)
                adapts to the signal: min(5000, Nyquist-1). An explicit value
                above Nyquist is rejected, never clamped.
            max_freq: Max envelope spectrum frequency to display. Default 500 Hz
            num_peaks: Number of peaks to detect. Default 15
            bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Raises:
            ValueError: If the signal_id is not loaded, or the stored signal
                has no sampling rate.

        Example:
            >>> # Bearing frequencies computed for YOUR bearing/rpm (here: 6205
            >>> # per CWRU geometry at 1797 RPM)
            >>> result = generate_envelope_report(
            ...     "real_train_OuterRaceFault_1",
            ...     bearing_freqs={"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.58, "FTF": 11.93}
            ... )
        """
    if ctx:
        await ctx.info(f"Generating envelope analysis report for '{signal_id}'...")

    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate

    # Resolve the default upper edge fs-aware: the fixed 5000 Hz default used
    # to exceed Nyquist on sub-10 kHz signals and raise. A None (default)
    # filter_high adapts to the signal; an explicit value is honored as-is and
    # still validated (an over-Nyquist band the caller asked for is never
    # clamped silently). min(5000, Nyquist-1) matches the digital-filter
    # corner realized below and compute_envelope_spectrum's own default.
    if filter_high is None:
        filter_high = min(5000.0, sampling_rate / 2 - 1.0)

    # U9 band-validation sweep: an invalid band vs Nyquist raises — this
    # used to feed scipy an unnormalizable corner (or silently mis-filter).
    validate_bandpass_band(filter_low, filter_high, sampling_rate)

    # Optional extras: reference bearing frequencies from the companion
    # metadata of the source file (never required).
    if bearing_freqs is None:
        metadata = _companion_metadata(info)
        if any(k in metadata for k in ("BPFO", "BPFI", "BSF", "FTF")):
            bearing_freqs = {
                "BPFO": metadata.get("BPFO"),
                "BPFI": metadata.get("BPFI"),
                "BSF": metadata.get("BSF"),
                "FTF": metadata.get("FTF")
            }

    # Bandpass filter (a corner exactly AT Nyquist is realized 1 Hz below
    # it — a digital filter corner cannot sit at Nyquist)
    nyquist = sampling_rate / 2
    high_norm = min(filter_high, nyquist - 1.0) / nyquist
    sos = butter(4, [filter_low / nyquist, high_norm], btype='band', output='sos')
    filtered_signal = sosfiltfilt(sos, signal_data)

    # Envelope via Hilbert
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # Envelope spectrum
    N = len(envelope)
    env_fft = fft(envelope)
    env_frequencies = fftfreq(N, 1/sampling_rate)

    positive_idx = env_frequencies > 0
    env_frequencies = env_frequencies[positive_idx]
    env_magnitudes = 2.0 * np.abs(env_fft[positive_idx]) / N

    # Generate and save report (signal_id is the report's signal label)
    result = save_envelope_report(
        signal_file=signal_id,
        sampling_rate=sampling_rate,
        filter_band=(filter_low, filter_high),
        filtered_signal=filtered_signal,
        envelope=envelope,
        env_frequencies=env_frequencies,
        env_magnitudes=env_magnitudes,
        bearing_freqs=bearing_freqs,
        max_freq=max_freq,
        num_peaks=num_peaks
    )

    if ctx:
        await ctx.info(result['message'])
        await ctx.info(f"Report location: {result['file_path']}")
        if result.get('bearing_matches'):
            await ctx.info(f"Bearing frequency matches: {', '.join(result['bearing_matches'])}")

    return result

async def generate_iso_report(
    signal_id: str,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
    rpm: Optional[float] = None,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Generate an ISO 20816-3 evaluation report (HTML) for a stored signal.

        Saves a self-contained Plotly HTML report (color-coded A-D zone
        chart with the measured RMS marker, boundaries, severity text) to
        the reports/ directory with a timestamped filename. The evaluation
        itself is delegated to assess_severity — requires the signal loaded
        via load_signal() first with sampling rate AND a declared unit
        (units are never guessed).

        Args:
            signal_id: ID of the stored signal (from load_signal).
            machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW)
            support_type: 'rigid' or 'flexible'
            rpm: Operating speed in RPM (optional; selects the ISO band's
                lower edge below 600 RPM)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Raises:
            ValueError: If the signal_id is not loaded, or the stored signal
                has no sampling rate or no declared unit.
        """
    if ctx:
        await ctx.info(f"Generating ISO 20816-3 report for '{signal_id}'...")

    # Perform ISO evaluation via the unified severity tool (U9 merge).
    sev = await assess_severity(
        ctx=ctx,
        signal_id=signal_id,
        machine_group=machine_group,
        support_type=support_type,
        rpm=rpm
    )

    # Map the unified model onto the report template's expected keys.
    iso_dict = {
        "rms_velocity": sev.rms_velocity_mm_s,
        "zone": sev.zone,
        "zone_description": sev.zone_description,
        "severity_level": sev.severity_level,
        "color_code": sev.color_code,
        "machine_group": sev.machine_group,
        "support_type": sev.support_type,
        "boundary_ab": sev.boundaries["AB"],
        "boundary_bc": sev.boundaries["BC"],
        "boundary_cd": sev.boundaries["CD"],
        "frequency_range": sev.frequency_range,
        "operating_speed_rpm": sev.operating_speed_rpm,
        "threshold_provenance": sev.threshold_provenance,
    }

    # Generate and save report (signal_id is the report's signal label)
    result = save_iso_report(
        signal_file=signal_id,
        iso_result=iso_dict
    )

    if ctx:
        await ctx.info(result['message'])
        await ctx.info(f"Report location: {result['file_path']}")

    return result

def list_html_reports(
    file_name: Optional[str] = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
        List HTML reports, or get one report's embedded metadata.

        Without file_name: lists every report in reports/ with file name,
        type, signal, and size. With file_name: returns that report's
        embedded metadata block (absorbed get_report_info). Never returns
        HTML content — metadata only, to avoid token consumption.

        Args:
            file_name: Optional report filename inside reports/ — returns
                its metadata instead of the listing.

        Returns:
            List of report summaries (no file_name), or a dict with the
            single report's metadata (file_name given).

        Raises:
            ValueError: If file_name escapes the reports directory, does
                not exist, or carries no metadata block.
        """
    if file_name is not None:
        # Single-report route (former get_report_info). The read path stays
        # on read_report_metadata, which contains the user-supplied name via
        # safe_resolve(REPORTS_DIR, file_name) — U1 security fix preserved.
        return read_report_metadata(file_name)
    return list_reports()

async def generate_pca_visualization_report(
    model_name: str,
    test_signal_ids: Optional[list[str]] = None,
    true_labels: Optional[dict[str, str]] = None,
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Generate PCA visualization HTML report showing test data in 2D PCA space.

        Creates interactive scatter plot with:
        - Test/prediction data (green = predicted healthy, red = predicted anomaly)
        - PC1 vs PC2 axes with variance explained
        - Hover information showing segment details and prediction status

        **IMPORTANT**: Labels show MODEL PREDICTIONS, not ground truth. Use `true_labels`
        parameter to provide actual labels for validation visualization.

        Requires the test signals loaded via load_signal() first; each
        signal's sampling rate comes from its stored metadata.

        Args:
            model_name: Name of trained model (e.g., 'bearing_health_model')
            test_signal_ids: Optional list of stored signal IDs to predict and visualize
            true_labels: Optional dict mapping signal_ids to true labels.
                        Format: {"real_test_baseline_3": "healthy",
                                 "real_test_InnerRaceFault_vload_6": "faulty"}
                        When provided, legend shows both true and predicted labels for validation.
            segment_duration: Segment duration in seconds (default: 0.1s for ML)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (includes validation metrics if true_labels provided)

        Raises:
            FileNotFoundError: If the model does not exist.
            ValueError: If a signal_id is not loaded or has no sampling rate.

        Example (with validation):
            >>> generate_pca_visualization_report(
            ...     model_name="bearing_health_model",
            ...     test_signal_ids=["real_test_baseline_3", "real_test_InnerRaceFault_vload_6"],
            ...     true_labels={"real_test_baseline_3": "healthy",
            ...                  "real_test_InnerRaceFault_vload_6": "faulty"}
            ... )
        """
    if ctx:
        await ctx.info(f"Generating PCA visualization for model '{model_name}'...")

    # Load model, scaler, PCA — validate the name and contain every derived
    # path before un-pickling (single source of truth in path_safety).
    _model_paths = resolve_model_paths(MODELS_DIR, model_name)
    model_path = _model_paths.model
    scaler_path = _model_paths.scaler
    pca_path = _model_paths.pca
    metadata_path = _model_paths.metadata

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)

    # Collect training data (reconstruct from model metadata if available)
    # For now, we'll just note that training data would be visualized
    # In production, you'd save training features during train_anomaly_model

    training_pca_data = []  # Placeholder - would load from saved training features

    # Process test signals if provided
    test_data_list = []
    last_sampling_rate: Optional[float] = None

    if test_signal_ids:
        for sid in test_signal_ids:
            signal_data, sig_info = resolve_signal(sid)
            fs = sig_info.sampling_rate
            last_sampling_rate = fs

            # Segment and extract features (per-signal sampling rate)
            segment_length_samples = int(segment_duration * fs)
            hop_length = int(segment_length_samples * (1 - overlap_ratio))

            # Guard: a signal shorter than one segment yields an empty feature
            # matrix and an opaque sklearn "X has 0 features" error at
            # scaler.transform. Fail early with an actionable message instead.
            if len(signal_data) < segment_length_samples:
                raise ValueError(
                    f"Signal '{sid}' ({len(signal_data)} samples) is shorter "
                    f"than one {segment_duration}s segment at {fs:g} Hz — "
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

            test_data_list.append({
                'signal_id': sid,
                'pca_data': X_pca,
                'predictions': predictions
            })

    # Create Plotly figure
    fig = go.Figure()

    # Plot test data
    for test_data in test_data_list:
        X_pca = test_data['pca_data']
        predictions = test_data['predictions']
        sid = test_data['signal_id']

        # Determine true label if provided (keyed by signal_id)
        true_label = None
        if true_labels and sid in true_labels:
            true_label = true_labels[sid].lower()

        # Separate healthy and anomalous predictions
        healthy_idx = predictions == 1
        anomaly_idx = predictions == -1

        # Create legend labels
        if true_label:
            # Show both true and predicted labels for validation
            healthy_legend = f'{sid} (True: {true_label}, Predicted: Healthy)'
            anomaly_legend = f'{sid} (True: {true_label}, Predicted: Anomaly)'

            # Update hover template to show both
            healthy_hover = f'<b>{sid}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>True Label: {true_label}<br>Predicted: Healthy<extra></extra>'
            anomaly_hover = f'<b>{sid}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>True Label: {true_label}<br>Predicted: ANOMALY<extra></extra>'
        else:
            # Show only predictions (no ground truth assumed)
            healthy_legend = f'{sid} (Predicted: Healthy)'
            anomaly_legend = f'{sid} (Predicted: Anomaly)'

            # Hover template clarifies these are predictions
            healthy_hover = f'<b>{sid}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Predicted: Healthy<extra></extra>'
            anomaly_hover = f'<b>{sid}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Predicted: ANOMALY<extra></extra>'

        if np.any(healthy_idx):
            fig.add_trace(go.Scatter(
                x=X_pca[healthy_idx, 0],
                y=X_pca[healthy_idx, 1],
                mode='markers',
                name=healthy_legend,
                marker=dict(color='green', size=8, opacity=0.6),
                hovertemplate=healthy_hover
            ))

        if np.any(anomaly_idx):
            fig.add_trace(go.Scatter(
                x=X_pca[anomaly_idx, 0],
                y=X_pca[anomaly_idx, 1],
                mode='markers',
                name=anomaly_legend,
                marker=dict(color='red', size=8, opacity=0.6, symbol='x'),
                hovertemplate=anomaly_hover
            ))

    # Layout
    variance_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=f"PCA Visualization - {model_name}",
        xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}% variance)",
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=700,
        showlegend=True
    )

    # Save HTML report (timestamped: consecutive runs never overwrite)
    output_file = REPORTS_DIR / timestamped_report_name(
        "pca_visualization", model_name
    )
    fig.write_html(str(output_file))

    # Prepare metadata - convert all numpy types to Python natives
    metadata = {
        'report_type': 'pca_visualization',
        'model_name': model_name,
        'test_signals': test_signal_ids or [],
        'pca_components': int(pca.n_components_),
        'variance_explained_pc1': float(variance_explained[0]),
        'variance_explained_pc2': float(variance_explained[1]),
        'total_variance_2d': float(variance_explained[0] + variance_explained[1]),
        'segment_duration': float(segment_duration),
        'sampling_rate': (
            float(last_sampling_rate) if last_sampling_rate is not None else None
        ),
        'validation_mode': true_labels is not None
    }

    # Calculate summary statistics - convert numpy int to Python int
    total_segments = int(sum(len(td['predictions']) for td in test_data_list))
    total_anomalies = int(sum(np.sum(td['predictions'] == -1) for td in test_data_list))

    summary = {
        'total_segments': int(total_segments),
        'total_anomalies': int(total_anomalies),
        'anomaly_ratio': float(total_anomalies / total_segments) if total_segments > 0 else 0.0
    }

    # Calculate validation metrics if true labels provided
    if true_labels:
        correct_predictions = 0
        total_with_labels = 0
        per_file_accuracy = {}

        for test_data in test_data_list:
            sid = test_data['signal_id']

            if sid in true_labels:
                predictions = test_data['predictions']
                true_label = true_labels[sid].lower()

                # Determine expected predictions (1 = healthy, -1 = anomaly)
                expected_prediction = 1 if true_label in ['healthy', 'normal', 'baseline'] else -1

                # Count correct predictions
                file_correct = int(np.sum(predictions == expected_prediction))
                file_total = len(predictions)
                correct_predictions += file_correct
                total_with_labels += file_total

                per_file_accuracy[sid] = {
                    'correct': file_correct,
                    'total': file_total,
                    'accuracy': float(file_correct / file_total) if file_total > 0 else 0.0,
                    'true_label': true_label
                }

        overall_accuracy = float(correct_predictions / total_with_labels) if total_with_labels > 0 else 0.0

        summary['validation_metrics'] = {
            'overall_accuracy': overall_accuracy,
            'total_labeled_segments': total_with_labels,
            'correct_predictions': correct_predictions,
            'per_file_accuracy': per_file_accuracy
        }

        if ctx:
            await ctx.info(f"Validation Mode: Overall accuracy = {overall_accuracy*100:.2f}%")
            for fname, acc_info in per_file_accuracy.items():
                await ctx.info(f"  - {fname}: {acc_info['accuracy']*100:.1f}% ({acc_info['correct']}/{acc_info['total']})")

    message = f"PCA visualization report saved: {output_file.name}"
    if ctx:
        await ctx.info(message)
        await ctx.info(f"PC1+PC2 explain {metadata['total_variance_2d']*100:.1f}% of variance")
        await ctx.info(f"Analyzed {total_segments} segments, {total_anomalies} anomalies detected")

    return {
        'file_path': str(output_file),
        'file_name': output_file.name,
        'message': message,
        'metadata': metadata,
        'summary': summary
    }

async def generate_feature_comparison_report(
    signal_groups: dict[str, list[str]],
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    features_to_plot: Optional[list[str]] = None,
    ctx: Context | None = None
) -> dict[str, Any]:
    """
        Generate feature comparison report with violin plots comparing time-domain features.

        Creates interactive HTML report with violin plots showing distribution of 17
        time-domain features across different signal groups (e.g., Healthy vs Faulty).
        Requires every signal loaded via load_signal() first; each signal's
        sampling rate comes from its stored metadata.

        **Strategy**: Same HTML report approach as other reports. Useful for understanding
        which features are most discriminative for fault detection.

        Args:
            signal_groups: Dictionary mapping group names to lists of stored
                          signal IDs.
                          Example: {"Healthy": ["real_train_baseline_1"],
                                   "Faulty": ["real_train_OuterRaceFault_1"]}
            segment_duration: Segment duration in seconds (default: 0.1s for ML)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            features_to_plot: List of feature names to plot (default: all 17 features)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary

        Raises:
            ValueError: If a signal_id is not loaded or has no sampling rate.

        Example:
            >>> generate_feature_comparison_report(
            ...     signal_groups={
            ...         "Healthy": ["real_train_baseline_1", "real_train_baseline_2"],
            ...         "Inner Fault": ["real_train_InnerRaceFault_vload_1"],
            ...         "Outer Fault": ["real_train_OuterRaceFault_1"]
            ...     }
            ... )
        """
    if ctx:
        await ctx.info(f"Generating feature comparison report for {len(signal_groups)} groups...")

    # All possible features
    all_feature_names = [
        'mean', 'std', 'var', 'mean_abs', 'rms', 'max', 'min', 'range',
        'crest_factor', 'kurtosis', 'skewness', 'shape_factor', 'impulse_factor',
        'clearance_factor', 'power', 'entropy', 'zero_crossing_rate'
    ]

    if features_to_plot is None:
        features_to_plot = all_feature_names

    # Extract features from all signal groups
    group_features = {}
    last_sampling_rate: Optional[float] = None

    for group_name, signal_ids in signal_groups.items():
        all_features_for_group = []

        for sid in signal_ids:
            signal_data, sig_info = resolve_signal(sid)
            fs = sig_info.sampling_rate
            last_sampling_rate = fs

            # Segment and extract features (per-signal sampling rate)
            segment_length_samples = int(segment_duration * fs)
            hop_length = int(segment_length_samples * (1 - overlap_ratio))

            for start in range(0, len(signal_data) - segment_length_samples + 1, hop_length):
                segment = signal_data[start:start + segment_length_samples]
                features = extract_time_domain_features(segment)
                all_features_for_group.append(features)

        group_features[group_name] = pd.DataFrame(all_features_for_group)

    # Create subplots - one violin plot per feature
    num_features = len(features_to_plot)
    rows = (num_features + 2) // 3  # 3 columns
    cols = min(3, num_features)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=features_to_plot,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    for idx, feature in enumerate(features_to_plot):
        row = idx // 3 + 1
        col = idx % 3 + 1

        for group_idx, (group_name, features_df) in enumerate(group_features.items()):
            if feature not in features_df.columns:
                continue

            fig.add_trace(
                go.Violin(
                    y=features_df[feature],
                    name=group_name,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[group_idx % len(colors)],
                    opacity=0.6,
                    showlegend=(idx == 0),  # Show legend only once
                    hovertemplate=f'<b>{group_name}</b><br>{feature}: %{{y:.4f}}<extra></extra>'
                ),
                row=row,
                col=col
            )

    # Update layout
    fig.update_layout(
        title="Time-Domain Feature Comparison (Violin Plots)",
        height=400 * rows,
        width=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Save HTML report (timestamped: consecutive runs never overwrite)
    group_names_safe = "_vs_".join(
        [name.replace(" ", "_") for name in signal_groups.keys()]
    )
    output_file = REPORTS_DIR / timestamped_report_name(
        "feature_comparison", group_names_safe
    )
    fig.write_html(str(output_file))

    # Prepare metadata
    metadata = {
        'report_type': 'feature_comparison',
        'groups': {name: len(ids) for name, ids in signal_groups.items()},
        'features_plotted': features_to_plot,
        'segment_duration': segment_duration,
        'sampling_rate': last_sampling_rate,
        'segments_per_group': {name: len(df) for name, df in group_features.items()}
    }

    message = f"Feature comparison report saved: {output_file.name}"
    if ctx:
        await ctx.info(message)
        await ctx.info(f"Compared {len(signal_groups)} groups across {len(features_to_plot)} features")

    return {
        'file_path': str(output_file),
        'file_name': output_file.name,
        'message': message,
        'metadata': metadata
    }


def register(mcp: MCPServer) -> None:
    """Register report generation and visualization tools with the MCP server."""
    mcp.tool()(generate_diagnostic_report_docx)
    mcp.tool()(plot_signal)
    mcp.tool()(generate_fft_report)
    mcp.tool()(generate_envelope_report)
    mcp.tool()(generate_iso_report)
    mcp.tool()(list_html_reports)
    mcp.tool()(generate_pca_visualization_report)
    mcp.tool()(generate_feature_comparison_report)
