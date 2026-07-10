"""MCP tools for report generation and visualization (ISO 13374 Block 6)."""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR, MODELS_DIR, RESOURCES_DIR, REPORTS_DIR
from ..signal_loader import load_signal_data, get_metadata_path, extract_segment, SUPPORTED_EXTENSIONS
from ..report_generator import (
    save_fft_report,
    save_envelope_report,
    save_iso_report,
    read_report_metadata,
    list_reports,
    save_diagnostic_report_docx,
    REPORTS_DIR as REPORTS_DIR_PATH,
)
from ..document_reader import calculate_bearing_frequencies
from ._utils import resolve_model_paths

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """Register report generation and visualization tools with the MCP server."""

    @mcp.tool()
    async def generate_diagnostic_report_docx(
        signal_file: str,
        sections: dict[str, Any],
        title: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured Word (.docx) diagnostic report.

        Requires: ``pip install predictive-maintenance-mcp[docx]``

        ``sections`` is a dict whose keys define what to include (all optional):
          - statistics:           dict  (RMS, Kurtosis, Crest Factor …)
          - fft_peaks:            list  [{frequency, magnitude_db, note}, …]
          - envelope_peaks:       list  [{frequency, magnitude_db, match}, …]
          - bearing_frequencies:  dict  {BPFO, BPFI, BSF, FTF}
          - iso:                  dict  (from evaluate_iso_20816 output)
          - diagnosis:            str   (free-text diagnostic summary)

        Args:
            signal_file: Signal filename used for the report title / filename
            sections: Content sections to include (see above)
            title: Optional custom report title
            ctx: MCP context

        Returns:
            Dictionary with file_path, file_name, and per-section summary.
        """
        if ctx:
            await ctx.info(f"Generating DOCX diagnostic report for {signal_file}")

        result = save_diagnostic_report_docx(signal_file, sections, title=title)

        if ctx and "error" not in result:
            await ctx.info(f"DOCX report saved: {result['file_name']}")

        return result

    @mcp.tool()
    async def plot_signal(
        signal_file: str,
        sampling_rate: float = 10000.0,
        time_range: Optional[list[float]] = None,
        show_statistics: bool = True,
        title: Optional[str] = None,
        ctx: Context | None = None
    ) -> str:
        """
        Generate interactive time-domain signal plot.

        Creates an interactive HTML plot showing the signal in the time domain.
        Useful for inspecting signal quality, identifying anomalies, and visualizing transients.

        Args:
            signal_file: Name of the CSV file in data/signals/
            sampling_rate: Sampling frequency in Hz (default: 10000)
            time_range: [start_time, end_time] in seconds to zoom on a portion (optional)
            show_statistics: Show RMS, peak levels as horizontal lines (default: True)
            title: Custom plot title (optional)
            ctx: MCP context for progress/logging

        Returns:
            Path to generated HTML file

        Example:
            plot_signal(
                "bearing_signal.csv",
                sampling_rate=10000,
                time_range=[0.1, 0.3],  # Zoom on 100-300 ms
                show_statistics=True
            )
        """
        if ctx:
            await ctx.info(f"Generating time-domain plot for {signal_file}...")

        # Read signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

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
        plot_title = title or f"Time-Domain Signal - {signal_file}"
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

        # Save HTML to reports directory
        safe_name = Path(signal_file).stem.replace('/', '_').replace('\\', '_')
        output_file = REPORTS_DIR / f"plot_signal_{safe_name}.html"
        fig.write_html(str(output_file))

        if ctx:
            await ctx.info(f"Plot saved to {output_file.name}")
            await ctx.info(f"To view report metadata: list_html_reports() or get_report_info()")

        return f"Interactive plot saved to: {output_file}\nUse list_html_reports() to see all reports, or open file in browser"

    @mcp.tool()
    async def plot_spectrum(
        signal_file: str,
        sampling_rate: float = 10000.0,
        freq_range: Optional[list[float]] = None,
        num_peaks: int = 10,
        min_peak_distance: float = 1.0,
        rotation_freq: Optional[float] = None,
        title: Optional[str] = None,
        ctx: Context | None = None
    ) -> str:
        """
        Generate interactive FFT spectrum plot with automatic peak detection.

        Creates an interactive HTML plot showing the frequency spectrum up to Nyquist frequency (Fs/2).
        Automatically identifies and labels the most significant peaks. If rotation frequency is provided,
        identifies harmonics as 1x, 2x, 3x RPM.

        Args:
            signal_file: Name of the CSV file in data/signals/
            sampling_rate: Sampling frequency in Hz (default: 10000)
            freq_range: [min_freq, max_freq] to limit the plot range (default: [0, Fs/2])
            num_peaks: Number of peaks to identify and label (default: 10)
            min_peak_distance: Minimum distance between peaks in Hz (default: 1.0)
            rotation_freq: Rotation frequency in Hz for RPM harmonic labeling (optional)
            title: Custom plot title (optional)
            ctx: MCP context for progress/logging

        Returns:
            Path to generated HTML file with peak information

        Example:
            plot_spectrum(
                "bearing_signal.csv",
                sampling_rate=10000,
                rotation_freq=25.0,  # 1500 RPM = 25 Hz
                num_peaks=15
            )
        """
        if ctx:
            await ctx.info(f"Generating spectrum plot for {signal_file}...")

        # Read signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

        # Apply Hamming window to reduce spectral leakage
        n = len(signal_data)
        window = np.hamming(n)
        signal_windowed = signal_data * window

        # Compute FFT
        freqs = fftfreq(n, d=1/sampling_rate)
        fft_values = fft(signal_windowed)

        # Keep only positive frequencies up to Nyquist (Fs/2)
        nyquist = sampling_rate / 2.0
        positive_freq_mask = (freqs > 0) & (freqs <= nyquist)
        freqs = freqs[positive_freq_mask]

        # Correct normalization for single-sided spectrum
        amplitude = 2.0 * np.abs(fft_values[positive_freq_mask]) / n

        # Convert to dB scale (normalized to maximum)
        # Peak will be at 0 dB, everything else negative
        max_amplitude = np.max(amplitude)
        amplitude_db = 20 * np.log10(np.maximum(amplitude / max_amplitude, 1e-10))

        # Default frequency range: 0 to Nyquist
        if freq_range is None:
            freq_range = [0, nyquist]

        # Apply frequency range filter
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_plot = freqs[mask]
        amplitude_plot = amplitude[mask]
        amplitude_db_plot = amplitude_db[mask]

        # Find peaks using scipy
        from scipy.signal import find_peaks

        # Convert min_peak_distance to number of samples
        freq_resolution = sampling_rate / n
        min_distance_samples = int(min_peak_distance / freq_resolution)

        # Find peaks in the plot range
        peak_indices, properties = find_peaks(
            amplitude_db_plot,
            distance=min_distance_samples,
            prominence=2  # Only peaks with >2 dB prominence
        )

        # Sort by amplitude and keep top num_peaks
        if len(peak_indices) > num_peaks:
            sorted_indices = np.argsort(amplitude_db_plot[peak_indices])[::-1]
            peak_indices = peak_indices[sorted_indices[:num_peaks]]

        # Create plot
        fig = go.Figure()

        # Main spectrum in dB
        fig.add_trace(go.Scatter(
            x=freqs_plot,
            y=amplitude_db_plot,
            mode='lines',
            name='Spectrum',
            line=dict(color='blue', width=1),
            hovertemplate='Frequency: %{x:.2f} Hz<br>Amplitude: %{y:.2f} dB<extra></extra>'
        ))

        # Mark detected peaks
        for idx in peak_indices:
            freq = freqs_plot[idx]
            amp_db = amplitude_db_plot[idx]

            # Generate label
            if rotation_freq:
                # Check if it's a harmonic of rotation frequency
                harmonic_ratio = freq / rotation_freq
                if abs(harmonic_ratio - round(harmonic_ratio)) < 0.1:  # Within 10% tolerance
                    harmonic_num = int(round(harmonic_ratio))
                    label = f"{harmonic_num}xRPM ({freq:.1f} Hz)"
                else:
                    label = f"{freq:.1f} Hz"
            else:
                label = f"{freq:.1f} Hz"

            # Add marker
            fig.add_trace(go.Scatter(
                x=[freq],
                y=[amp_db],
                mode='markers+text',
                name=label,
                marker=dict(color='red', size=8, symbol='diamond'),
                text=[label],
                textposition="top center",
                textfont=dict(size=9, color='red'),
                hovertemplate=f'{label}<br>Amplitude: {amp_db:.2f} dB<extra></extra>'
            ))

        # Layout
        plot_title = title or f"FFT Spectrum - {signal_file}"
        fig.update_layout(
            title=plot_title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude (dB re. max)",
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600,
            showlegend=False,  # Hide legend since we have text labels
            yaxis=dict(range=[-80, 5])  # From -80 dB to +5 dB (peak at 0 dB)
        )

        # Save HTML to reports directory
        safe_name = Path(signal_file).stem.replace('/', '_').replace('\\', '_')
        output_file = REPORTS_DIR / f"plot_spectrum_{safe_name}.html"
        fig.write_html(str(output_file))

        if ctx:
            await ctx.info(f"Plot saved to {output_file.name}")
            await ctx.info(f"Detected {len(peak_indices)} significant peaks")
            await ctx.info(f"To view report metadata: list_html_reports() or get_report_info()")

        return f"Interactive plot saved to: {output_file}\nUse list_html_reports() to see all reports, or open file in browser"

    @mcp.tool()
    async def plot_envelope(
        signal_file: str,
        sampling_rate: float = 10000.0,
        filter_band: Optional[list[float]] = None,
        freq_range: Optional[list[float]] = None,
        highlight_freqs: Optional[list[float]] = None,
        freq_labels: Optional[list[str]] = None,
        title: Optional[str] = None,
        ctx: Context | None = None
    ) -> str:
        """
        Generate interactive envelope spectrum plot.

        Creates an interactive HTML plot showing both the envelope spectrum and optionally
        the filtered signal. Can highlight bearing/gear frequencies.

        Args:
            signal_file: Name of the CSV file in data/signals/
            sampling_rate: Sampling frequency in Hz (default: 10000)
            filter_band: [low_freq, high_freq] for bandpass filter (optional, default: [500, 5000])
            freq_range: [min_freq, max_freq] to limit the envelope spectrum plot (optional)
            highlight_freqs: List of frequencies (Hz) to mark (e.g., BPFO, BPFI) (optional)
            freq_labels: Labels for highlighted frequencies (optional)
            title: Custom plot title (optional)
            ctx: MCP context for progress/logging

        Returns:
            Path to generated HTML file

        Example:
            plot_envelope(
                "bearing_signal.csv",
                sampling_rate=10000,
                filter_band=[500, 5000],
                freq_range=[0, 300],
                highlight_freqs=[120.5, 241.0],
                freq_labels=["BPFO", "2xBPFO"]
            )
        """
        if ctx:
            await ctx.info(f"Generating envelope plot for {signal_file}...")

        # Read signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

        # Default filter band if not specified
        if filter_band is None:
            filter_band = [500, 5000]

        # Validate filter band
        nyquist = sampling_rate / 2
        if filter_band[1] >= nyquist:
            # Adjust upper frequency to be below Nyquist
            filter_band[1] = nyquist * 0.95
            if ctx:
                await ctx.info(f"Adjusted filter upper limit to {filter_band[1]:.0f} Hz (< Nyquist)")

        # Bandpass filter
        low = filter_band[0] / nyquist
        high = filter_band[1] / nyquist

        # Ensure valid range
        if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
            raise ValueError(f"Invalid filter band [{filter_band[0]}, {filter_band[1]}] Hz for Fs={sampling_rate} Hz (Nyquist={nyquist} Hz)")

        sos = butter(4, [low, high], btype='band', output='sos')
        filtered_signal = sosfiltfilt(sos, signal_data)

        # Envelope using Hilbert transform
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)

        # Apply Hamming window to envelope before FFT
        n = len(envelope)
        window = np.hamming(n)
        envelope_windowed = envelope * window

        # FFT of envelope
        freqs = fftfreq(n, d=1/sampling_rate)
        fft_envelope = fft(envelope_windowed)

        # Keep only positive frequencies (excluding DC at freq=0)
        positive_freq_mask = freqs > 0
        freqs = freqs[positive_freq_mask]

        # Correct normalization for single-sided spectrum
        amplitude = 2.0 * np.abs(fft_envelope[positive_freq_mask]) / n

        # Convert to dB scale (normalized to maximum)
        max_amplitude = np.max(amplitude)
        amplitude_db = 20 * np.log10(np.maximum(amplitude / max_amplitude, 1e-10))

        # Apply frequency range filter if specified
        if freq_range:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_plot = freqs[mask]
            amplitude_plot = amplitude[mask]
            amplitude_db_plot = amplitude_db[mask]
        else:
            freqs_plot = freqs
            amplitude_plot = amplitude
            amplitude_db_plot = amplitude_db

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'Filtered Signal ({filter_band[0]}-{filter_band[1]} Hz)',
                'Envelope Spectrum'
            ),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.6]
        )

        # Time array for signal plot
        time = np.arange(len(filtered_signal)) / sampling_rate

        # Plot 1: Filtered signal with envelope
        fig.add_trace(
            go.Scatter(
                x=time,
                y=filtered_signal,
                mode='lines',
                name='Filtered Signal',
                line=dict(color='lightblue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=envelope,
                mode='lines',
                name='Envelope',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )

        # Plot 2: Envelope spectrum in dB
        fig.add_trace(
            go.Scatter(
                x=freqs_plot,
                y=amplitude_db_plot,
                mode='lines',
                name='Envelope Spectrum',
                line=dict(color='darkblue', width=1),
                hovertemplate='Frequency: %{x:.2f} Hz<br>Amplitude: %{y:.2f} dB<extra></extra>'
            ),
            row=2, col=1
        )

        # Highlight specific frequencies in envelope spectrum
        if highlight_freqs:
            if not freq_labels:
                freq_labels = [f"{f:.1f} Hz" for f in highlight_freqs]

            for freq, label in zip(highlight_freqs, freq_labels):
                # Find nearest frequency
                idx = np.argmin(np.abs(freqs_plot - freq))

                # Add vertical line
                fig.add_vline(
                    x=freq,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=label,
                    annotation_position="top",
                    row=2, col=1
                )

                # Add marker
                fig.add_trace(
                    go.Scatter(
                        x=[freqs_plot[idx]],
                        y=[amplitude_db_plot[idx]],
                        mode='markers',
                        name=label,
                        marker=dict(color='red', size=10, symbol='diamond'),
                        hovertemplate=f'{label}<br>Frequency: %{{x:.2f}} Hz<br>Amplitude: %{{y:.2f}} dB<extra></extra>',
                        showlegend=True
                    ),
                    row=2, col=1
                )

        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (dB re. max)", row=2, col=1)

        # Set dB range for envelope spectrum (normalized to max)
        fig.update_yaxes(range=[-80, 5], row=2, col=1)

        # Layout
        plot_title = title or f"Envelope Analysis - {signal_file}"
        fig.update_layout(
            title_text=plot_title,
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=900,
            showlegend=True
        )

        # Save HTML to reports directory
        safe_name = Path(signal_file).stem.replace('/', '_').replace('\\', '_')
        output_file = REPORTS_DIR / f"plot_envelope_{safe_name}.html"
        fig.write_html(str(output_file))

        if ctx:
            await ctx.info(f"Plot saved to {output_file.name}")
            await ctx.info(f"To view report metadata: list_html_reports() or get_report_info()")

        return f"Interactive plot saved to: {output_file}\nUse list_html_reports() to see all reports, or open file in browser"

    @mcp.tool()
    async def generate_fft_report(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        max_freq: float = 5000.0,
        num_peaks: int = 15,
        rotation_freq: Optional[float] = None,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Generate professional FFT spectrum report as HTML file.

        **NEW PREFERRED METHOD**: Generates a professional HTML report file
        instead of inline content. Saves to reports/ directory.

        Args:
            signal_file: Signal filename in data/signals/
            sampling_rate: Sampling rate in Hz (auto-detect if None)
            max_freq: Maximum frequency to display (Hz). Default 5000 Hz
            num_peaks: Number of peaks to detect and label. Default 15
            rotation_freq: Optional shaft rotation frequency for harmonic labels
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Example:
            >>> result = generate_fft_report("real_train/baseline_1.csv")
            >>> # User can open: result['file_path']
        """
        if ctx:
            await ctx.info(f"Generating FFT report for {signal_file}...")

        # Load signal
        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Unable to load signal from {signal_file}")

        # Auto-detect sampling rate
        if sampling_rate is None:
            metadata_path = get_metadata_path(signal_file)
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    sampling_rate = metadata.get("sampling_rate")

            if sampling_rate is None:
                raise ValueError("sampling_rate required (not found in metadata)")

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

        # Generate and save report
        result = save_fft_report(
            signal_file=signal_file,
            sampling_rate=sampling_rate,
            frequencies=frequencies,
            magnitudes=magnitudes,
            signal_data=signal_data,
            max_freq=max_freq,
            num_peaks=num_peaks,
            rotation_freq=rotation_freq
        )

        if ctx:
            await ctx.info(result['message'])
            await ctx.info(f"Report location: {result['file_path']}")

        return result

    @mcp.tool()
    async def generate_envelope_report(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        filter_low: float = 500.0,
        filter_high: float = 5000.0,
        max_freq: float = 500.0,
        num_peaks: int = 15,
        bearing_freqs: Optional[dict[str, float]] = None,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Generate professional envelope analysis report as HTML file.

        **NEW PREFERRED METHOD**: Generates a professional HTML report file
        instead of inline content. Saves to reports/ directory.

        Args:
            signal_file: Signal filename in data/signals/
            sampling_rate: Sampling rate in Hz (auto-detect if None)
            filter_low: Bandpass filter low cutoff (Hz). Default 500 Hz
            filter_high: Bandpass filter high cutoff (Hz). Default 5000 Hz
            max_freq: Max envelope spectrum frequency to display. Default 500 Hz
            num_peaks: Number of peaks to detect. Default 15
            bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Example:
            >>> result = generate_envelope_report(
            ...     "real_train/OuterRaceFault_1.csv",
            ...     bearing_freqs={"BPFO": 81.13, "BPFI": 118.88, "BSF": 63.91, "FTF": 14.84}
            ... )
        """
        if ctx:
            await ctx.info(f"Generating envelope analysis report for {signal_file}...")

        # Load signal
        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Unable to load signal from {signal_file}")

        # Auto-detect sampling rate and bearing freqs
        if sampling_rate is None or bearing_freqs is None:
            metadata_path = get_metadata_path(signal_file)
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    if sampling_rate is None:
                        sampling_rate = metadata.get("sampling_rate")
                    if bearing_freqs is None:
                        bearing_freqs = {
                            "BPFO": metadata.get("BPFO"),
                            "BPFI": metadata.get("BPFI"),
                            "BSF": metadata.get("BSF"),
                            "FTF": metadata.get("FTF")
                        }

            if sampling_rate is None:
                raise ValueError("sampling_rate required")

        # Bandpass filter
        nyquist = sampling_rate / 2
        sos = butter(4, [filter_low / nyquist, filter_high / nyquist], btype='band', output='sos')
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

        # Generate and save report
        result = save_envelope_report(
            signal_file=signal_file,
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

    @mcp.tool()
    async def generate_iso_report(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        machine_group: int = 2,
        support_type: str = "rigid",
        operating_speed_rpm: Optional[float] = None,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Generate professional ISO 20816-3 evaluation report as HTML file.

        **NEW PREFERRED METHOD**: Generates a professional HTML report file
        instead of inline content. Saves to reports/ directory.

        Args:
            signal_file: Signal filename in data/signals/
            sampling_rate: Sampling rate in Hz (auto-detect if None)
            machine_group: ISO machine group (1=large >300kW, 2=medium 15-300kW)
            support_type: 'rigid' or 'flexible'
            operating_speed_rpm: Operating speed in RPM (optional)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (NO HTML content)

        Example:
            >>> result = generate_iso_report(
            ...     "real_train/baseline_1.csv",
            ...     machine_group=2,
            ...     support_type="rigid"
            ... )
        """
        if ctx:
            await ctx.info(f"Generating ISO 20816-3 report for {signal_file}...")

        # Auto-detect sampling rate
        if sampling_rate is None:
            metadata_path = get_metadata_path(signal_file)
            if metadata_path.exists():
                with open(metadata_path) as f:
                    sampling_rate = json.load(f).get("sampling_rate")

            if sampling_rate is None:
                raise ValueError("sampling_rate required")

        # Use the monolith's evaluate_iso_20816 (shares same logic as diagnostics_tools)
        from ..machinery_diagnostics_server import evaluate_iso_20816

        # Perform ISO evaluation
        iso_result = await evaluate_iso_20816(
            ctx=ctx,
            signal_file=signal_file,
            sampling_rate=sampling_rate,
            machine_group=machine_group,
            support_type=support_type,
            operating_speed_rpm=operating_speed_rpm
        )

        # Convert Pydantic model to dict
        iso_dict = iso_result.model_dump()

        # Generate and save report
        result = save_iso_report(
            signal_file=signal_file,
            iso_result=iso_dict
        )

        if ctx:
            await ctx.info(result['message'])
            await ctx.info(f"Report location: {result['file_path']}")

        return result

    @mcp.tool()
    def list_html_reports() -> list[dict[str, Any]]:
        """
        List all available HTML reports in reports/ directory.

        Returns list of reports with metadata (file name, type, signal, size).
        Does NOT return HTML content - only metadata to avoid token consumption.

        Returns:
            List of dicts with report information

        Example:
            >>> reports = list_html_reports()
            >>> print(f"Found {len(reports)} reports")
            >>> for r in reports:
            ...     print(f"- {r['file_name']}: {r['report_type']} for {r['signal_file']}")
        """
        return list_reports()

    @mcp.tool()
    def get_report_info(file_name: str) -> dict[str, Any]:
        """
        Get metadata from HTML report without loading entire file.

        Extracts metadata JSON from HTML report file. This allows LLM to
        understand report content without consuming tokens for HTML.

        Args:
            file_name: Report filename in reports/ directory

        Returns:
            Dictionary with metadata (NO HTML content)

        Example:
            >>> info = get_report_info("fft_spectrum_baseline_1.html")
            >>> print(f"Signal: {info['metadata']['signal_file']}")
            >>> print(f"Peaks detected: {info['metadata']['num_peaks']}")
        """
        return read_report_metadata(file_name)

    @mcp.tool()
    async def generate_pca_visualization_report(
        model_name: str,
        test_signal_files: Optional[list[str]] = None,
        true_labels: Optional[dict[str, str]] = None,
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Generate PCA visualization HTML report showing training and test data in 2D PCA space.

        Creates interactive scatter plot with:
        - Training data (blue dots) - healthy baseline
        - Test/prediction data (green = predicted healthy, red = predicted anomaly)
        - PC1 vs PC2 axes with variance explained
        - Hover information showing segment details and prediction status

        **IMPORTANT**: Labels show MODEL PREDICTIONS, not ground truth. Use `true_labels`
        parameter to provide actual labels for validation visualization.

        **Strategy**: Same HTML report approach as FFT/Envelope/ISO reports.
        Saved to reports/ directory for LLM to reference without consuming tokens.

        Args:
            model_name: Name of trained model (e.g., 'bearing_health_model')
            test_signal_files: Optional list of signals to predict and visualize
            true_labels: Optional dict mapping signal filenames to true labels.
                        Format: {"baseline_3.csv": "healthy", "InnerRaceFault_vload_6.csv": "faulty"}
                        When provided, legend shows both true and predicted labels for validation.
            sampling_rate: Sampling rate (auto-detect from metadata if None)
            segment_duration: Segment duration in seconds (default: 0.1s for ML)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary (includes validation metrics if true_labels provided)

        Example (predictions only):
            >>> generate_pca_visualization_report(
            ...     model_name="bearing_health_model",
            ...     test_signal_files=["real_test/baseline_3.csv", "real_test/InnerRaceFault_vload_6.csv"]
            ... )

        Example (with validation):
            >>> generate_pca_visualization_report(
            ...     model_name="bearing_health_model",
            ...     test_signal_files=["real_test/baseline_3.csv", "real_test/InnerRaceFault_vload_6.csv"],
            ...     true_labels={"baseline_3.csv": "healthy", "InnerRaceFault_vload_6.csv": "faulty"}
            ... )
        """
        # Import extract_time_domain_features from the monolith (will be refactored later)
        from ..machinery_diagnostics_server import extract_time_domain_features

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

        # Use model's sampling rate if not provided
        if sampling_rate is None:
            sampling_rate = model_metadata.get('sampling_rate', 10000.0)

        # Collect training data (reconstruct from model metadata if available)
        # For now, we'll just note that training data would be visualized
        # In production, you'd save training features during train_anomaly_model

        training_pca_data = []  # Placeholder - would load from saved training features

        # Process test signals if provided
        test_data_list = []

        if test_signal_files:
            for signal_file in test_signal_files:
                filepath = DATA_DIR / signal_file
                if not filepath.exists():
                    logger.warning(f"File not found: {signal_file}, skipping...")
                    continue

                signal_data = load_signal_data(signal_file)
                if signal_data is None:
                    raise ValueError(f"Could not load signal data from: {signal_file}")

                # Segment and extract features
                segment_length_samples = int(segment_duration * sampling_rate)
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

                test_data_list.append({
                    'signal_file': signal_file,
                    'pca_data': X_pca,
                    'predictions': predictions
                })

        # Create Plotly figure
        fig = go.Figure()

        # Plot test data
        for test_data in test_data_list:
            X_pca = test_data['pca_data']
            predictions = test_data['predictions']
            signal_file = test_data['signal_file']

            # Extract filename without path for label matching
            file_basename = signal_file.split('/')[-1]

            # Determine true label if provided
            true_label = None
            if true_labels and file_basename in true_labels:
                true_label = true_labels[file_basename].lower()

            # Separate healthy and anomalous predictions
            healthy_idx = predictions == 1
            anomaly_idx = predictions == -1

            # Create legend labels
            if true_label:
                # Show both true and predicted labels for validation
                healthy_legend = f'{signal_file} (True: {true_label}, Predicted: Healthy)'
                anomaly_legend = f'{signal_file} (True: {true_label}, Predicted: Anomaly)'

                # Update hover template to show both
                healthy_hover = f'<b>{signal_file}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>True Label: {true_label}<br>Predicted: Healthy<extra></extra>'
                anomaly_hover = f'<b>{signal_file}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>True Label: {true_label}<br>Predicted: ANOMALY<extra></extra>'
            else:
                # Show only predictions (no ground truth assumed)
                healthy_legend = f'{signal_file} (Predicted: Healthy)'
                anomaly_legend = f'{signal_file} (Predicted: Anomaly)'

                # Hover template clarifies these are predictions
                healthy_hover = f'<b>{signal_file}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Predicted: Healthy<extra></extra>'
                anomaly_hover = f'<b>{signal_file}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Predicted: ANOMALY<extra></extra>'

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

        # Save HTML report
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        output_file = REPORTS_DIR / f"pca_visualization_{safe_name}.html"
        fig.write_html(str(output_file))

        # Prepare metadata - convert all numpy types to Python natives
        metadata = {
            'report_type': 'pca_visualization',
            'model_name': model_name,
            'test_signals': test_signal_files or [],
            'pca_components': int(pca.n_components_),
            'variance_explained_pc1': float(variance_explained[0]),
            'variance_explained_pc2': float(variance_explained[1]),
            'total_variance_2d': float(variance_explained[0] + variance_explained[1]),
            'segment_duration': float(segment_duration),
            'sampling_rate': float(sampling_rate) if sampling_rate is not None else None,
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
                signal_file = test_data['signal_file']
                file_basename = signal_file.split('/')[-1]

                if file_basename in true_labels:
                    predictions = test_data['predictions']
                    true_label = true_labels[file_basename].lower()

                    # Determine expected predictions (1 = healthy, -1 = anomaly)
                    expected_prediction = 1 if true_label in ['healthy', 'normal', 'baseline'] else -1

                    # Count correct predictions
                    file_correct = int(np.sum(predictions == expected_prediction))
                    file_total = len(predictions)
                    correct_predictions += file_correct
                    total_with_labels += file_total

                    per_file_accuracy[file_basename] = {
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

    @mcp.tool()
    async def generate_feature_comparison_report(
        signal_groups: dict[str, list[str]],
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
        features_to_plot: Optional[list[str]] = None,
        ctx: Context | None = None
    ) -> dict[str, Any]:
        """
        Generate feature comparison report with violin plots comparing time-domain features.

        Creates interactive HTML report with violin plots showing distribution of 17
        time-domain features across different signal groups (e.g., Healthy vs Faulty).

        **Strategy**: Same HTML report approach as other reports. Useful for understanding
        which features are most discriminative for fault detection.

        Args:
            signal_groups: Dictionary mapping group names to list of signal files.
                          Example: {"Healthy": ["baseline_1.csv", "baseline_2.csv"],
                                   "Faulty": ["InnerRaceFault_1.csv", "OuterRaceFault_1.csv"]}
            sampling_rate: Sampling rate (auto-detect from metadata if None)
            segment_duration: Segment duration in seconds (default: 0.1s for ML)
            overlap_ratio: Overlap ratio 0-1 (default: 0.5)
            features_to_plot: List of feature names to plot (default: all 17 features)
            ctx: MCP context

        Returns:
            Dictionary with file path, metadata, and summary

        Example:
            >>> generate_feature_comparison_report(
            ...     signal_groups={
            ...         "Healthy": ["real_train/baseline_1.csv", "real_train/baseline_2.csv"],
            ...         "Inner Fault": ["real_train/InnerRaceFault_vload_1.csv"],
            ...         "Outer Fault": ["real_train/OuterRaceFault_1.csv"]
            ...     }
            ... )
        """
        # Import extract_time_domain_features from the monolith (will be refactored later)
        from ..machinery_diagnostics_server import extract_time_domain_features

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

        for group_name, signal_files in signal_groups.items():
            all_features_for_group = []

            for signal_file in signal_files:
                filepath = DATA_DIR / signal_file
                if not filepath.exists():
                    logger.warning(f"File not found: {signal_file}, skipping...")
                    continue

                # Auto-detect sampling rate if not provided
                if sampling_rate is None:
                    metadata_path = get_metadata_path(signal_file)
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                            sampling_rate = metadata.get("sampling_rate", 10000.0)
                    else:
                        sampling_rate = 10000.0  # fallback

                signal_data = load_signal_data(signal_file)
                if signal_data is None:
                    raise ValueError(f"Could not load signal data from: {signal_file}")

                # Segment and extract features
                segment_length_samples = int(segment_duration * sampling_rate)
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

        # Save HTML report
        group_names_safe = "_vs_".join([name.replace(" ", "_") for name in signal_groups.keys()])
        output_file = REPORTS_DIR / f"feature_comparison_{group_names_safe}.html"
        fig.write_html(str(output_file))

        # Prepare metadata
        metadata = {
            'report_type': 'feature_comparison',
            'groups': {name: len(files) for name, files in signal_groups.items()},
            'features_plotted': features_to_plot,
            'segment_duration': segment_duration,
            'sampling_rate': sampling_rate,
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
