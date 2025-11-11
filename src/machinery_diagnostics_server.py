"""
Predictive Maintenance MCP Server

This server provides tools and resources for predictive maintenance,
vibration signal analysis, and industrial machinery diagnostics.

Features:
- Resources: Reading signals from files (CSV, binary)
- Tools: FFT Analysis, Envelope Analysis, Statistical Analysis
- Prompts: Diagnostic workflows for bearings, gears, etc.
"""

import logging
from pathlib import Path
from typing import Any, Optional, List, Dict
from dataclasses import dataclass
import pickle
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession


# Logging configuration (use stderr to not interfere with stdio)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Goes to stderr by default
)
logger = logging.getLogger(__name__)

# MCP server initialization
mcp = FastMCP(
    "Predictive Maintenance",
    instructions="""
    MCP server for predictive maintenance and industrial machinery diagnostics.
    
    Capabilities:
    - Reading and managing vibration signals
    - Spectral analysis (FFT with dB normalization)
    - Envelope analysis for bearing fault detection
    - Statistical analysis (RMS, Kurtosis, Crest Factor)
    - ISO 20816-3 vibration severity evaluation
    - HTML artifact generation (interactive Plotly charts, no file I/O)
    - Automatic peak detection and harmonic identification
    - Guided diagnostic workflows (prompts)

    Evidence-based inference policy (hard rules):
    1) Do NOT infer fault type from filenames, paths, or user-provided labels. Treat filenames as opaque identifiers.
    2) Do NOT make diagnostic claims based solely on statistical parameters (RMS/CF/Kurtosis). Use them for screening only.
    3) Bearing fault identification (inner/outer/ball/cage) must be supported by frequency-domain evidence (envelope peaks at characteristic frequencies) and at least one additional indicator (e.g., high kurtosis or distinct harmonics). If this corroboration is missing, mark the result as "inconclusive" and recommend further analysis.
    4) Use cautious language: say "possible" or "consistent with" when evidence is partial; say "confirmed" only if multiple independent analyses agree.
    5) Always cite which analyses and thresholds support each conclusion. If data or parameters are missing, ask for them instead of guessing.
    6) NEVER suggest parameters, thresholds, or recommendations not explicitly provided in tool outputs or prompt workflows. Do NOT invent frequency ranges, filter settings, or maintenance actions. Only use guidance from STEP 6 of diagnostic prompts.

    Output formatting rules:
    - Keep responses brief (≤300 words, bullet points)
    - Use HTML charts for visualization (render as artifacts, NOT code)
    - If truncated, retry with save_*_to_file() tools
    - NEVER print large data directly

    Filename resolution policy:
    - FIRST call list_available_signals() to verify exact filename
    - Do NOT auto-correct or guess filenames
    - If ambiguous, ask user to clarify
    
    Workflow Prompts (use these for guided analysis):
    - diagnose_bearing() - Complete bearing diagnostic workflow with evidence-based decision tree
    - diagnose_gear() - Gear fault detection workflow
    - quick_diagnostic_report() - Fast screening analysis (non-definitive)
    """
)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "signals"
MODELS_DIR = Path(__file__).parent.parent / "models"


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class FFTResult(BaseModel):
    """FFT analysis result with structured output."""
    frequencies: list[float] = Field(description="Frequency array (Hz)")
    magnitudes: list[float] = Field(description="Magnitude array")
    peak_frequency: float = Field(description="Dominant peak frequency (Hz)")
    peak_magnitude: float = Field(description="Dominant peak magnitude")
    sampling_rate: float = Field(description="Sampling frequency (Hz)")
    num_samples: int = Field(description="Number of analyzed samples")
    frequency_resolution: float = Field(description="Frequency resolution (Hz)")


class EnvelopeResult(BaseModel):
    """Envelope analysis result - optimized for chat display."""
    # Summary statistics instead of full arrays
    num_samples: int = Field(description="Number of samples in envelope signal")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    filter_band: tuple[float, float] = Field(description="Bandpass filter band (Hz)")
    
    # Only top peaks (not full spectrum)
    peak_frequencies: list[float] = Field(description="Top peak frequencies (Hz)")
    peak_magnitudes: list[float] = Field(description="Top peak magnitudes")
    
    # Human-readable diagnosis
    diagnosis: str = Field(description="Interpretive diagnosis text with bearing frequency analysis")
    
    # Optional: small preview of spectrum (first 100 points for visualization hint)
    spectrum_preview_freq: list[float] = Field(default=[], description="First 100 freq points (Hz)")
    spectrum_preview_mag: list[float] = Field(default=[], description="First 100 magnitude points")


class StatisticalResult(BaseModel):
    """Statistical analysis result of the signal."""
    rms: float = Field(description="Root Mean Square (effective value)")
    peak_to_peak: float = Field(description="Peak-to-peak value")
    peak: float = Field(description="Peak value")
    crest_factor: float = Field(description="Crest Factor (Peak/RMS)")
    kurtosis: float = Field(description="Kurtosis (measure of impulsiveness)")
    skewness: float = Field(description="Skewness (asymmetry)")
    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")


class SignalInfo(BaseModel):
    """Information about an available signal."""
    filename: str = Field(description="File name")
    path: str = Field(description="Full path")
    size_bytes: int = Field(description="File size in bytes")
    num_samples: Optional[int] = Field(None, description="Number of samples (if available)")


class ISO20816Result(BaseModel):
    """ISO 20816-3 vibration severity evaluation result."""
    rms_velocity: float = Field(description="RMS velocity in mm/s (broadband)")
    machine_group: int = Field(description="Machine group (1 or 2)")
    support_type: str = Field(description="Support type: 'rigid' or 'flexible'")
    zone: str = Field(description="Evaluation zone: 'A', 'B', 'C', or 'D'")
    zone_description: str = Field(description="Zone description and recommendation")
    severity_level: str = Field(description="Severity level: 'Good', 'Acceptable', 'Unsatisfactory', 'Unacceptable'")
    color_code: str = Field(description="Color code: 'green', 'yellow', 'orange', 'red'")
    boundary_ab: float = Field(description="Zone A/B boundary (mm/s)")
    boundary_bc: float = Field(description="Zone B/C boundary (mm/s)")
    boundary_cd: float = Field(description="Zone C/D boundary (mm/s)")
    frequency_range: str = Field(description="Frequency range used for measurement")
    operating_speed_rpm: Optional[float] = Field(None, description="Operating speed in RPM")


class FeatureExtractionResult(BaseModel):
    """Result of time-domain feature extraction from signal segments."""
    num_segments: int = Field(description="Number of segments extracted")
    segment_length_samples: int = Field(description="Samples per segment")
    segment_duration_s: float = Field(description="Duration of each segment in seconds")
    overlap_ratio: float = Field(description="Overlap ratio between segments")
    features_shape: List[int] = Field(description="Shape of feature matrix [num_segments, num_features]")
    feature_names: List[str] = Field(description="Names of extracted features")
    features_preview: List[Dict[str, float]] = Field(description="First 5 segments features (preview)")


class AnomalyModelResult(BaseModel):
    """Result of anomaly detection model training."""
    model_type: str = Field(description="Type of model: 'OneClassSVM' or 'LocalOutlierFactor'")
    num_training_samples: int = Field(description="Number of healthy samples used for training")
    num_features_original: int = Field(description="Number of original features")
    num_features_pca: int = Field(description="Number of PCA components (features after dimensionality reduction)")
    variance_explained: float = Field(description="Cumulative variance explained by PCA components")
    model_params: Dict[str, Any] = Field(description="Best model hyperparameters")
    model_path: str = Field(description="Path to saved model file (.pkl)")
    scaler_path: str = Field(description="Path to saved scaler file (.pkl)")
    pca_path: str = Field(description="Path to saved PCA file (.pkl)")
    validation_accuracy: Optional[float] = Field(None, description="Accuracy on fault data if provided")
    validation_details: Optional[str] = Field(None, description="Validation details")


class AnomalyPredictionResult(BaseModel):
    """Result of anomaly detection prediction on new data."""
    num_segments: int = Field(description="Number of segments analyzed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    anomaly_ratio: float = Field(description="Ratio of anomalies (0-1)")
    predictions: List[int] = Field(description="Predictions per segment: 1=normal, -1=anomaly")
    anomaly_scores: Optional[List[float]] = Field(None, description="Anomaly scores if available")
    overall_health: str = Field(description="Overall health status: 'Healthy', 'Suspicious', 'Faulty'")
    confidence: str = Field(description="Confidence level: 'High', 'Medium', 'Low'")



# ============================================================================
# RESOURCES - SIGNAL READING
# ============================================================================

@mcp.resource("signal://list")
def list_available_signals() -> str:
    """
    List all available signals in the data/signals directory.
    
    Returns:
        JSON with the list of available signal files
    """
    try:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            return "[]"
        
        signals = []
        for file_path in DATA_DIR.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".csv", ".txt", ".npy", ".dat"]:
                signals.append({
                    "filename": str(file_path.relative_to(DATA_DIR)).replace("\\", "/"),
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "extension": file_path.suffix
                })
        
        return pd.DataFrame(signals).to_json(orient="records", indent=2)
    
    except Exception as e:
        logger.error(f"Error listing signals: {e}")
        return f'{{"error": "{str(e)}"}}'


@mcp.resource("signal://read/{filename}")
def read_signal_file(filename: str) -> str:
    """
    Read a signal file and return the data.
    
    Supports formats: CSV, TXT (newline-separated values), NPY (numpy array)
    
    Args:
        filename: Name of the file to read
        
    Returns:
        JSON with signal data
    """
    try:
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            return f'{{"error": "File {filename} not found"}}'
        
        # Lettura in base all'estensione
        if file_path.suffix == ".npy":
            data = np.load(file_path)
            signal_data = data.tolist()
        
        elif file_path.suffix in [".csv", ".txt"]:
            df = pd.read_csv(file_path, header=None)
            signal_data = df.iloc[:, 0].tolist()
        
        else:
            return f'{{"error": "Unsupported file format: {file_path.suffix}"}}'
        
        result = {
            "filename": filename,
            "num_samples": len(signal_data),
            "data": signal_data[:1000],  # First 1000 samples to avoid overload
            "total_samples": len(signal_data),
            "preview_only": len(signal_data) > 1000
        }
        
        return pd.Series(result).to_json(indent=2)
    
    except Exception as e:
        logger.error(f"Error reading signal {filename}: {e}")
        return f'{{"error": "{str(e)}"}}'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_signal_data(filename: str) -> Optional[np.ndarray]:
    """
    Load signal data from file.
    
    Args:
        filename: File name
        
    Returns:
        Numpy array with data or None if error
    """
    try:
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            return None
        
        if file_path.suffix == ".npy":
            return np.load(file_path)
        
        elif file_path.suffix in [".csv", ".txt"]:
            df = pd.read_csv(file_path, header=None)
            return df.iloc[:, 0].values
        
        return None
    
    except Exception as e:
        logger.error(f"Error loading signal {filename}: {e}")
        return None


# ============================================================================

@mcp.tool()
def list_signals() -> str:
    """
    List all available signal files in the data/signals directory.
    
    Returns:
        A formatted text listing of all available signals
    """
    try:
        if not DATA_DIR.exists():
            return "No signals directory found"
        
        signals = []
        for file_path in DATA_DIR.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".csv", ".txt", ".npy", ".dat"]:
                relative_path = file_path.relative_to(DATA_DIR)
                signals.append(str(relative_path).replace("\\", "/"))
        
        if not signals:
            return "No signal files found"
        
        # Group by directory
        from collections import defaultdict
        by_dir = defaultdict(list)
        for sig in signals:
            dir_name = sig.split('/')[0] if '/' in sig else "root"
            by_dir[dir_name].append(sig)
        
        # Format output
        output = [f"Found {len(signals)} signal files:"]
        for dir_name in sorted(by_dir.keys()):
            files = by_dir[dir_name]
            output.append(f"\n{dir_name}/ ({len(files)} files):")
            for f in sorted(files):
                output.append(f"  - {f}")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def read_plot_html(filename: str, max_chars: int = 15000) -> str:
    """
    Read and return the content of an HTML plot file for inline rendering in clients.
    
    This tool allows clients like Claude to display interactive plots inline by retrieving
    the HTML content. The content is truncated if it exceeds max_chars to avoid context overflow.
    
    Args:
        filename: Name of the plot file (e.g., "plot_envelope_baseline_1.html")
        max_chars: Maximum characters to return (default: 15000)
    
    Returns:
        HTML content of the plot file (truncated if necessary) or an error message
    
    Example:
        read_plot_html("plot_envelope_baseline_1.html")
    """
    try:
        # Check in data/signals directory first
        plot_path = DATA_DIR / filename
        
        # If not found, check in project root
        if not plot_path.exists():
            plot_path = Path(__file__).parent.parent / filename
        
        # If still not found, return error
        if not plot_path.exists():
            return f"Error: Plot file '{filename}' not found in data/signals/ or project root."
        
        # Read HTML content
        with open(plot_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Get file size info
        file_size = len(html_content)
        
        # Truncate if necessary
        if file_size > max_chars:
            html_content = html_content[:max_chars]
            truncation_note = f"\n\n<!-- TRUNCATED: File is {file_size} chars, showing first {max_chars} chars. -->\n"
            html_content += truncation_note
        
        return html_content
    
    except Exception as e:
        return f"Error reading plot file '{filename}': {str(e)}"


# ============================================================================
# # TOOLS - FFT ANALYSIS
# ============================================================================

@mcp.tool()
def analyze_fft(
    filename: str,
    sampling_rate: float = 1000.0,
    max_frequency: Optional[float] = None
) -> FFTResult:
    """
    Perform FFT (Fast Fourier Transform) analysis on a signal.
    
    FFT analysis converts the signal from time domain to frequency domain,
    allowing identification of harmonic components and faults that manifest
    at specific frequencies.
    
    IMPORTANT: Do NOT use the default sampling_rate (1000 Hz) unless explicitly 
    confirmed correct for this signal. Check signal metadata first or ask user.
    
    Args:
        filename: Name of the file containing the signal
        sampling_rate: Sampling frequency in Hz (default: 1000 Hz - VERIFY BEFORE USE)
        max_frequency: Maximum frequency to analyze (default: Nyquist frequency)
        
    Returns:
        FFTResult with frequencies, magnitudes and dominant peak
    """
    # Load data
    signal_data = load_signal_data(filename)
    
    if signal_data is None:
        raise ValueError(f"Unable to load signal from {filename}")
    
    # Number of samples
    N = len(signal_data)
    
    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(N)
    signal_windowed = signal_data * window
    
    # Calculate FFT
    fft_values = fft(signal_windowed)
    frequencies = fftfreq(N, 1/sampling_rate)
    
    # Take only positive frequencies (excluding DC component at index 0)
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    
    # Correct normalization for single-sided spectrum:
    # - Multiply by 2 (energy from negative frequencies)
    # - Divide by N (FFT normalization)
    # Note: DC component (freq=0) should not be multiplied by 2, but we exclude it with frequencies > 0
    magnitudes = 2.0 * np.abs(fft_values[positive_freq_idx]) / N
    
    # Apply maximum frequency limit if specified
    if max_frequency is not None:
        freq_limit_idx = frequencies <= max_frequency
        frequencies = frequencies[freq_limit_idx]
        magnitudes = magnitudes[freq_limit_idx]
    
    # Find dominant peak
    peak_idx = np.argmax(magnitudes)
    peak_frequency = float(frequencies[peak_idx])
    peak_magnitude = float(magnitudes[peak_idx])
    
    # Calculate frequency resolution
    frequency_resolution = sampling_rate / N
    
    return FFTResult(
        frequencies=frequencies.tolist(),
        magnitudes=magnitudes.tolist(),
        peak_frequency=peak_frequency,
        peak_magnitude=peak_magnitude,
        sampling_rate=sampling_rate,
        num_samples=N,
        frequency_resolution=frequency_resolution
    )


# ============================================================================
# TOOLS - ENVELOPE ANALYSIS
# ============================================================================

@mcp.tool()
def analyze_envelope(
    filename: str,
    sampling_rate: float = 1000.0,
    filter_low: float = 500.0,
    filter_high: float = 2000.0,
    num_peaks: int = 5
) -> EnvelopeResult:
    """
    Perform Envelope Analysis to detect bearing faults.
    
    Envelope analysis is particularly effective for detecting faults in ball/roller bearings.
    The signal is high-pass filtered, then the envelope is calculated via Hilbert transform,
    and finally the envelope spectrum is analyzed.
    
    Returns ONLY peak information and diagnosis text (no full arrays) to avoid context overflow.
    
    IMPORTANT: Do NOT use the default sampling_rate (1000 Hz) unless explicitly 
    confirmed correct for this signal. Check signal metadata first or ask user.
    
    Args:
        filename: Name of the file containing the signal
        sampling_rate: Sampling frequency in Hz (default: 1000 Hz - VERIFY BEFORE USE)
        filter_low: Low frequency of bandpass filter in Hz (default: 500 Hz)
        filter_high: High frequency of bandpass filter in Hz (default: 2000 Hz)
        num_peaks: Number of main peaks to identify (default: 5)
        
    Returns:
        EnvelopeResult with peak information and diagnosis (optimized for chat display)
    """
    # Load data
    signal_data = load_signal_data(filename)
    
    if signal_data is None:
        raise ValueError(f"Unable to load signal from {filename}")
    
    # Design Butterworth bandpass filter using SOS (numerically stable)
    nyquist = sampling_rate / 2
    low = filter_low / nyquist
    high = filter_high / nyquist
    
    sos = butter(4, [low, high], btype='band', output='sos')
    
    # Apply filter
    filtered_signal = sosfiltfilt(sos, signal_data)
    
    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # Calculate envelope spectrum
    N = len(envelope)
    envelope_fft = fft(envelope)
    envelope_frequencies = fftfreq(N, 1/sampling_rate)
    
    # Take only positive frequencies
    positive_idx = envelope_frequencies > 0
    envelope_frequencies = envelope_frequencies[positive_idx]
    envelope_magnitudes = np.abs(envelope_fft[positive_idx])
    
    # Find main peaks
    peak_indices = np.argsort(envelope_magnitudes)[-num_peaks:][::-1]
    peak_frequencies = envelope_frequencies[peak_indices].tolist()
    peak_magnitudes = envelope_magnitudes[peak_indices].tolist()
    
    # Create diagnosis text
    diagnosis_lines = [
        f"Envelope Analysis Results:",
        f"Filter band: {filter_low}-{filter_high} Hz",
        f"",
        f"Top {num_peaks} peaks in envelope spectrum:"
    ]
    
    for i, (freq, mag) in enumerate(zip(peak_frequencies, peak_magnitudes), 1):
        diagnosis_lines.append(f"  {i}. {freq:7.2f} Hz  (magnitude: {mag:.2e})")
    
    diagnosis_lines.extend([
        "",
        "Bearing frequency reference (typical SKF 6205 @ 1500 RPM):",
        "  • BPFO (outer race): ~81.13 Hz",
        "  • BPFI (inner race): ~118.88 Hz",
        "  • BSF (ball spin):   ~63.91 Hz",
        "  • FTF (cage):        ~14.84 Hz",
        "",
        "⚠️ Compare peaks above with actual bearing frequencies for your system.",
        "💡 Use plot_envelope(...) for visual analysis and harmonic identification."
    ])
    
    diagnosis = "\n".join(diagnosis_lines)
    
    # Small preview (first 100 points for hint/context)
    preview_size = min(100, len(envelope_frequencies))
    
    return EnvelopeResult(
        num_samples=len(envelope),
        sampling_rate=sampling_rate,
        filter_band=(filter_low, filter_high),
        peak_frequencies=peak_frequencies,
        peak_magnitudes=peak_magnitudes,
        diagnosis=diagnosis,
        spectrum_preview_freq=envelope_frequencies[:preview_size].tolist(),
        spectrum_preview_mag=envelope_magnitudes[:preview_size].tolist()
    )


# ============================================================================
# TOOLS - STATISTICAL ANALYSIS
# ============================================================================

@mcp.tool()
def analyze_statistics(filename: str) -> StatisticalResult:
    """
    Calculate statistical parameters of the signal for diagnostics.
    
    Statistical parameters are key indicators for diagnostics:
    - RMS: Effective value, correlated to signal energy
    - Crest Factor: Indicates presence of impulses (high = possible faults)
    - Kurtosis: Measures impulsiveness (>3 = presence of impulses)
    - Peak-to-Peak: Signal range
    
    Args:
        filename: Name of the file containing the signal
        
    Returns:
        StatisticalResult with all statistical parameters
    """
    # Load data
    signal_data = load_signal_data(filename)
    
    if signal_data is None:
        raise ValueError(f"Unable to load signal from {filename}")
    
    # Calculate statistical parameters
    rms = float(np.sqrt(np.mean(signal_data**2)))
    peak = float(np.max(np.abs(signal_data)))
    peak_to_peak = float(np.max(signal_data) - np.min(signal_data))
    mean_val = float(np.mean(signal_data))
    std_dev = float(np.std(signal_data))
    
    # Crest Factor
    crest_factor = peak / rms if rms > 0 else 0.0
    
    # Kurtosis (using scipy)
    from scipy.stats import kurtosis, skew
    kurtosis_val = float(kurtosis(signal_data, fisher=True))  # Fisher=True for excess kurtosis
    skewness_val = float(skew(signal_data))
    
    return StatisticalResult(
        rms=rms,
        peak_to_peak=peak_to_peak,
        peak=peak,
        crest_factor=crest_factor,
        kurtosis=kurtosis_val,
        skewness=skewness_val,
        mean=mean_val,
        std_dev=std_dev
    )


@mcp.tool()
def evaluate_iso_20816(
    signal_file: str,
    sampling_rate: float = 10000.0,
    machine_group: int = 2,  # CHANGED: Default 2 (medium) - most common industrial case
    support_type: str = "rigid",  # Default rigid - most common for horizontal machines
    operating_speed_rpm: Optional[float] = None
) -> ISO20816Result:
    """
    Evaluate vibration severity according to ISO 20816-3 standard.
    
    ISO 20816-3 defines vibration severity zones for rotating machinery based on
    broadband RMS velocity measurements on non-rotating parts (bearings, housings).
    
    **DEFAULTS** (use if user doesn't specify):
    - machine_group = 2 (medium-sized machines, most common)
    - support_type = "rigid" (horizontal machines on foundations)
    
    **Machine Group Selection Guide** (ask user if unsure):
    - Group 1: Large machines (power >300 kW OR shaft height H ≥ 315 mm)
      Examples: Large turbines, generators, compressors, large pumps
    - Group 2: Medium machines (15-300 kW OR 160mm ≤ H < 315mm) [DEFAULT]
      Examples: Industrial motors, fans, pumps, gearboxes
    
    **Support Type Selection Guide** (ask user if unsure):
    - "rigid": Machine on stiff foundation, horizontal orientation [DEFAULT]
      Rule: Lowest natural frequency > 1.25 × main excitation frequency
      Examples: Motors/pumps on concrete, horizontal compressors
    - "flexible": Machine on soft supports, vertical, or large turbine-generator sets
      Examples: Vertical pumps, machines on springs, large turbogenerators
    
    **When to ask user**:
    - If power/dimensions unknown → use defaults (Group 2, rigid)
    - If clearly large turbine (>10 MW) → suggest Group 1, flexible
    - If vertical machine → suggest flexible
    - If user provides machine specs → use guide above
    
    Evaluation Zones:
    - Zone A (Green): New machine condition - excellent
    - Zone B (Yellow): Acceptable for long-term unrestricted operation
    - Zone C (Orange): Unsatisfactory - limited operation, plan maintenance
    - Zone D (Red): Sufficient severity to cause damage - immediate action
    
    Args:
        signal_file: Name of the CSV file in data/signals/
        sampling_rate: Sampling frequency in Hz (default: 10000)
        machine_group: Machine group 1 (large) or 2 (medium) (default: 2 - medium)
        support_type: 'rigid' or 'flexible' (default: 'rigid')
        operating_speed_rpm: Operating speed in RPM (optional, for frequency range selection)
    
    Returns:
        ISO20816Result with evaluation zone, severity level, and recommendations
        
    Example:
        evaluate_iso_20816(
            "motor_vibration.csv",
            sampling_rate=10000,
            machine_group=2,
            support_type="rigid",
            operating_speed_rpm=1500
        )
    """
    # Load signal
    filepath = DATA_DIR / signal_file
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {signal_file}")
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
    # Try to read metadata JSON for sampling rate
    metadata_file = filepath.parent / (filepath.stem + "_metadata.json")
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if 'sampling_rate' in metadata:
                sampling_rate = metadata['sampling_rate']
    
    # Auto-detect and convert acceleration to velocity if needed
    # Heuristic: acceleration signals typically have RMS > 0.5 g
    rms_raw = np.sqrt(np.mean(signal_data**2))
    if rms_raw > 0.5:  # Likely acceleration in g
        # Convert acceleration (g) to velocity (mm/s)
        # Integration: v(t) = ∫ a(t) dt
        signal_ac = signal_data - np.mean(signal_data)  # Remove DC
        
        # Convert from g to m/s²
        g_const = 9.80665  # m/s²
        accel_ms2 = signal_ac * g_const
        
        # Integrate in frequency domain
        n = len(accel_ms2)
        dt = 1.0 / sampling_rate
        
        # FFT
        accel_fft = np.fft.rfft(accel_ms2)
        freqs = np.fft.rfftfreq(n, dt)
        
        # Integrate: V(f) = A(f) / (j*2πf)
        vel_fft = np.zeros_like(accel_fft, dtype=complex)
        vel_fft[1:] = accel_fft[1:] / (1j * 2 * np.pi * freqs[1:])
        
        # IFFT to get velocity in m/s
        vel_ms = np.fft.irfft(vel_fft, n=n)
        
        # Convert to mm/s
        signal_data = vel_ms * 1000.0
    
    # Determine frequency range based on operating speed
    # ISO 20816-3: 10-1000 Hz for speeds ≥ 600 rpm
    #              2-1000 Hz for speeds 120-600 rpm
    if operating_speed_rpm and operating_speed_rpm < 600:
        freq_low = 2.0
        freq_high = 1000.0
        freq_range_desc = "2-1000 Hz (speed < 600 RPM)"
    else:
        freq_low = 10.0
        freq_high = 1000.0
        freq_range_desc = "10-1000 Hz (speed ≥ 600 RPM)"
    
    # Apply bandpass filter using SOS (more numerically stable)
    nyquist = sampling_rate / 2.0
    
    # Ensure filter frequencies are within valid range
    freq_low = max(freq_low, 1.0)
    freq_high = min(freq_high, nyquist * 0.95)
    
    # Use SOS format for numerical stability with high sampling rates
    sos = butter(4, [freq_low / nyquist, freq_high / nyquist], btype='band', output='sos')
    signal_filtered = sosfiltfilt(sos, signal_data)
    
    # Calculate RMS velocity in mm/s
    # Assuming input signal is already in mm/s (or convert if needed)
    rms_velocity = float(np.sqrt(np.mean(signal_filtered**2)))
    
    # ISO 20816-3 zone boundaries (mm/s RMS velocity)
    # Table A.1: Group 1 (Large machines, >300 kW, H ≥ 315 mm)
    # Table A.2: Group 2 (Medium machines, 15-300 kW, 160 mm ≤ H < 315 mm)
    
    if machine_group == 1:
        if support_type.lower() == "rigid":
            boundary_ab = 2.3
            boundary_bc = 4.5
            boundary_cd = 7.1
        else:  # flexible
            boundary_ab = 3.5
            boundary_bc = 7.1
            boundary_cd = 11.0
    elif machine_group == 2:
        if support_type.lower() == "rigid":
            boundary_ab = 1.4
            boundary_bc = 2.8
            boundary_cd = 4.5
        else:  # flexible
            boundary_ab = 2.3
            boundary_bc = 4.5
            boundary_cd = 7.1
    else:
        raise ValueError(f"Invalid machine_group: {machine_group}. Must be 1 or 2.")
    
    # Determine zone
    if rms_velocity <= boundary_ab:
        zone = "A"
        zone_desc = "New machine condition. Vibration is excellent."
        severity = "Good"
        color = "green"
    elif rms_velocity <= boundary_bc:
        zone = "B"
        zone_desc = "Acceptable for unrestricted long-term operation."
        severity = "Acceptable"
        color = "yellow"
    elif rms_velocity <= boundary_cd:
        zone = "C"
        zone_desc = "Unsatisfactory for long-term operation. Plan maintenance soon."
        severity = "Unsatisfactory"
        color = "orange"
    else:
        zone = "D"
        zone_desc = "Vibration severity may cause damage. Immediate action required!"
        severity = "Unacceptable"
        color = "red"
    
    return ISO20816Result(
        rms_velocity=rms_velocity,
        machine_group=machine_group,
        support_type=support_type.lower(),
        zone=zone,
        zone_description=zone_desc,
        severity_level=severity,
        color_code=color,
        boundary_ab=boundary_ab,
        boundary_bc=boundary_bc,
        boundary_cd=boundary_cd,
        frequency_range=freq_range_desc,
        operating_speed_rpm=operating_speed_rpm
    )


@mcp.tool()
async def plot_iso_20816_chart(
    filename: str,
    sampling_rate: float,
    machine_group: int = 1,
    support_type: str = "rigid",
    operating_speed_rpm: Optional[float] = None,
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
        ctx: MCP context
    
    Returns:
        Path to generated HTML file with ISO chart
    """
    if ctx:
        await ctx.info(f"Evaluating ISO 20816-3 for {filename}...")
    
    # First, perform ISO evaluation
    iso_result = evaluate_iso_20816(
        signal_file=filename,
        sampling_rate=sampling_rate,
        machine_group=machine_group,
        support_type=support_type,
        operating_speed_rpm=operating_speed_rpm
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
    
    # Save HTML
    output_file = DATA_DIR / f"plot_iso_{filename.replace('.csv', '')}.html"
    fig.write_html(str(output_file))
    
    if ctx:
        await ctx.info(f"ISO chart saved to {output_file.name}")
        await ctx.info(f"💡 To view inline, use: read_plot_html('{output_file.name}')")
    
    return f"ISO 20816-3 chart saved to: {output_file}\n💡 For inline rendering: read_plot_html('{output_file.name}')"


# ============================================================================
# TOOLS - HTML ARTIFACT GENERATION (Inline rendering for LLMs)
# ============================================================================

@mcp.tool()
def generate_iso_chart_html(
    signal_file: str,
    sampling_rate: float,
    machine_group: int = 1,
    support_type: str = "rigid",
    operating_speed_rpm: Optional[float] = None
) -> str:
    """
    Generate standalone HTML artifact for ISO 20816-3 visualization.
    
    Returns complete HTML document (not a file path) that can be rendered directly
    as an artifact in LLM interfaces like Claude, ChatGPT, etc.
    
    This is PREFERRED over plot_iso_20816_chart for inline rendering because:
    - No file I/O required
    - Works with any LLM supporting HTML artifacts
    - Self-contained (includes Plotly.js CDN)
    - Responsive and interactive
    
    Args:
        signal_file: Name of the signal file
        sampling_rate: Sampling frequency (Hz)
        machine_group: 1 (large >300kW) or 2 (medium 15-300kW)
        support_type: 'rigid' or 'flexible'
        operating_speed_rpm: Operating speed in RPM (optional)
    
    Returns:
        Complete HTML document as string
    """
    # Perform ISO evaluation
    iso_result = evaluate_iso_20816(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        machine_group=machine_group,
        support_type=support_type,
        operating_speed_rpm=operating_speed_rpm
    )
    
    # Color mapping
    zone_colors = {
        "A": ("#27ae60", "zone-a"),
        "B": ("#f39c12", "zone-b"),
        "C": ("#e67e22", "zone-c"),
        "D": ("#c0392b", "zone-d")
    }
    
    zone_color, zone_class = zone_colors.get(iso_result.zone, ("#95a5a6", "zone-unknown"))
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISO 20816-3 Vibration Chart</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .info-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .info-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 18px;
            margin: 20px auto;
            text-align: center;
            width: fit-content;
            display: block;
        }}
        .zone-a {{ background: #27ae60; color: white; }}
        .zone-b {{ background: #f39c12; color: white; }}
        .zone-c {{ background: #e67e22; color: white; }}
        .zone-d {{ background: #c0392b; color: white; }}
        .legend {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .legend-color {{
            width: 40px;
            height: 40px;
            border-radius: 5px;
            margin-right: 15px;
        }}
        .legend-text {{
            flex: 1;
        }}
        .legend-label {{
            font-weight: bold;
            margin-bottom: 3px;
        }}
        .legend-desc {{
            font-size: 11px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ISO 20816-3 Vibration Severity Chart</h1>
        <div class="subtitle">{signal_file} | Machine Group {machine_group} | {support_type.title()} Support</div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="info-label">RMS Velocity</div>
                <div class="info-value">{iso_result.rms_velocity:.2f} mm/s</div>
            </div>
            <div class="info-card">
                <div class="info-label">Evaluation Zone</div>
                <div class="info-value">Zone {iso_result.zone}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Severity Level</div>
                <div class="info-value">{iso_result.severity_level}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Frequency Range</div>
                <div class="info-value">{iso_result.frequency_range}</div>
            </div>
        </div>

        <div class="status-badge {zone_class}">
            {'✓' if iso_result.zone == 'A' else '⚠️' if iso_result.zone in ['B', 'C'] else '🚨'} {iso_result.zone_description}
        </div>

        <div class="chart-container">
            <div id="chart"></div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #27ae60;"></div>
                <div class="legend-text">
                    <div class="legend-label">Zone A (0 - {iso_result.boundary_ab:.1f} mm/s)</div>
                    <div class="legend-desc">New machine condition - excellent</div>
                </div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f39c12;"></div>
                <div class="legend-text">
                    <div class="legend-label">Zone B ({iso_result.boundary_ab:.1f} - {iso_result.boundary_bc:.1f} mm/s)</div>
                    <div class="legend-desc">Acceptable for long-term operation</div>
                </div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e67e22;"></div>
                <div class="legend-text">
                    <div class="legend-label">Zone C ({iso_result.boundary_bc:.1f} - {iso_result.boundary_cd:.1f} mm/s)</div>
                    <div class="legend-desc">Unsatisfactory - plan maintenance</div>
                </div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #c0392b;"></div>
                <div class="legend-text">
                    <div class="legend-label">Zone D (> {iso_result.boundary_cd:.1f} mm/s)</div>
                    <div class="legend-desc">Severe - immediate action required</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const boundaries = {{
            AB: {iso_result.boundary_ab},
            BC: {iso_result.boundary_bc},
            CD: {iso_result.boundary_cd},
            max: {iso_result.boundary_cd * 1.3}
        }};
        
        const rmsVelocity = {iso_result.rms_velocity};
        
        const trace1 = {{
            x: [boundaries.AB],
            y: ['Vibration Severity'],
            name: 'Zone A',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#27ae60',
                line: {{ color: '#229954', width: 2 }}
            }},
            hovertemplate: 'Zone A: 0 - ' + boundaries.AB + ' mm/s<br>New machine condition<extra></extra>'
        }};
        
        const trace2 = {{
            x: [boundaries.BC - boundaries.AB],
            y: ['Vibration Severity'],
            name: 'Zone B',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#f39c12',
                line: {{ color: '#e67e22', width: 2 }}
            }},
            hovertemplate: 'Zone B: ' + boundaries.AB + ' - ' + boundaries.BC + ' mm/s<br>Acceptable operation<extra></extra>'
        }};
        
        const trace3 = {{
            x: [boundaries.CD - boundaries.BC],
            y: ['Vibration Severity'],
            name: 'Zone C',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#e67e22',
                line: {{ color: '#d35400', width: 2 }}
            }},
            hovertemplate: 'Zone C: ' + boundaries.BC + ' - ' + boundaries.CD + ' mm/s<br>Unsatisfactory<extra></extra>'
        }};
        
        const trace4 = {{
            x: [boundaries.max - boundaries.CD],
            y: ['Vibration Severity'],
            name: 'Zone D',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#c0392b',
                line: {{ color: '#a93226', width: 2 }}
            }},
            hovertemplate: 'Zone D: > ' + boundaries.CD + ' mm/s<br>Severe condition<extra></extra>'
        }};
        
        const marker = {{
            x: [rmsVelocity],
            y: ['Vibration Severity'],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Measured',
            marker: {{
                color: '#2c3e50',
                size: 20,
                symbol: 'circle',
                line: {{ color: '#fff', width: 3 }}
            }},
            text: [rmsVelocity.toFixed(2) + ' mm/s'],
            textposition: 'top center',
            textfont: {{ size: 14, color: '#2c3e50', family: 'Arial Black' }},
            hovertemplate: 'Measured RMS Velocity: ' + rmsVelocity.toFixed(2) + ' mm/s<extra></extra>'
        }};
        
        const data = [trace1, trace2, trace3, trace4, marker];
        
        const layout = {{
            title: {{
                text: 'Vibration Severity according to ISO 20816-3',
                font: {{ size: 18, color: '#2c3e50' }}
            }},
            barmode: 'stack',
            xaxis: {{
                title: 'RMS Velocity (mm/s)',
                range: [0, boundaries.max],
                showgrid: true,
                gridcolor: '#ecf0f1',
                zeroline: true
            }},
            yaxis: {{
                showticklabels: false
            }},
            height: 350,
            margin: {{ l: 50, r: 50, t: 80, b: 80 }},
            showlegend: true,
            legend: {{
                x: 1.02,
                y: 0.5,
                orientation: 'v'
            }},
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white'
        }};
        
        const config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('chart', data, layout, config);
    </script>
</body>
</html>"""
    
    return html


@mcp.tool()
def generate_fft_chart_html(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    max_freq: float = 5000.0,
    num_peaks: int = 8,
    rotation_freq: Optional[float] = None,
    use_db: bool = False,
    segment_duration: Optional[float] = 1.0  # NEW: Use 1.0s segment by default (None = full signal)
) -> str:
    """
    Generate interactive FFT spectrum chart as HTML artifact (no file I/O).
    
    **IMPORTANT FOR LLMs**: After calling this tool, render the returned HTML string
    as an interactive artifact/preview, NOT as code. The HTML contains a Plotly chart
    that must be displayed to the user, not shown as text.
    
    Returns complete HTML document with Plotly visualization showing frequency spectrum
    in dB scale, identified peaks, and optional harmonic markers.
    
    Analysis is performed on FULL signal (no downsampling) for maximum accuracy.
    Only display data is downsampled for efficient rendering.
    
    Args:
        signal_file: Signal filename in data/signals/ or subdirectory (e.g., "real_train/baseline_1.csv")
        sampling_rate: Sampling rate in Hz. If None, will auto-detect from metadata JSON
        max_freq: Maximum frequency to display (Hz). Default 5000 Hz for bearing analysis
        num_peaks: Number of top peaks to identify and label. Default 8 (reduced from 10 to limit output size)
        rotation_freq: Optional shaft rotation frequency (Hz) for harmonic labeling
    
    Returns:
        Complete HTML document as string (~25-30KB) with:
        - Interactive Plotly line chart (spectrum magnitude in dB)
        - Peak markers (red diamonds) with frequency labels
        - Harmonic notes if rotation_freq provided (e.g., "Harmonic 74× shaft")
        - Top 8 peaks listed with dB values
        - Self-contained (Plotly.js from CDN)
    
    Example:
        >>> html = generate_fft_chart_html(
        ...     "real_train/inner_fault_1.csv",
        ...     max_freq=2000,
        ...     num_peaks=8,
        ...     rotation_freq=25.0
        ... )
        >>> # LLM should render this HTML as interactive artifact
    """
    # Load signal
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    # Auto-detect sampling rate if not provided
    if sampling_rate is None:
        metadata_path = signal_path.parent / f"{signal_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                sampling_rate = metadata.get("sampling_rate")
        
        if sampling_rate is None:
            raise ValueError("sampling_rate required (not found in metadata)")
    
    # Extract segment if requested (default: 1.0s segment from middle)
    if segment_duration is not None and segment_duration > 0:
        segment_samples = int(segment_duration * sampling_rate)
        if len(signal_data) > segment_samples:
            # Take segment from middle (more representative than start)
            start_idx = (len(signal_data) - segment_samples) // 2
            signal_data = signal_data[start_idx:start_idx + segment_samples]
    
    # STEP 1: FFT on signal (full resolution, no decimation!)
    n = len(signal_data)
    freq_full = np.fft.rfftfreq(n, 1/sampling_rate)
    fft_full = np.fft.rfft(signal_data)
    mag_full = np.abs(fft_full) * 2 / n  # Full spectrum magnitude
    
    # Convert to dB scale (normalized to max: 20*log10(mag/max))
    max_mag = np.max(mag_full)
    mag_full_db = 20 * np.log10((mag_full + 1e-12) / max_mag)  # Normalized dB
    
    # STEP 2: Find peaks on FULL spectrum (accurate detection)
    from scipy.signal import find_peaks
    freq_resolution = freq_full[1] - freq_full[0]
    min_peak_distance = max(1, int(10 / freq_resolution))  # Min 10 Hz spacing, but ≥1
    peak_indices, properties = find_peaks(
        mag_full_db,  # Use dB for peak detection
        height=-40,  # Peaks within 40 dB of max (max is at 0 dB)
        distance=min_peak_distance
    )
    
    # Sort by magnitude and take top N
    peak_mags_db = properties['peak_heights']
    top_peak_idx = np.argsort(peak_mags_db)[::-1][:num_peaks]
    peak_indices = peak_indices[top_peak_idx]
    peak_freqs = freq_full[peak_indices]
    peak_mags_db = mag_full_db[peak_indices]
    
    # STEP 3: Prepare display data (NO decimation for accurate peak alignment)
    mask = freq_full <= max_freq
    freq_display = freq_full[mask]  # Full resolution for accurate peaks
    mag_display_db = mag_full_db[mask]
    
    # Filter peaks within display range
    peak_mask = peak_freqs <= max_freq
    display_peak_freqs = peak_freqs[peak_mask]
    display_peak_mags_db = peak_mags_db[peak_mask]
    
    # Format peak info for display with CLEAR labels
    peak_info_lines = []
    for i, (pf, pm_db) in enumerate(zip(display_peak_freqs, display_peak_mags_db), 1):
        # Check for harmonics (if rotation_freq provided)
        harmonic_note = ""
        if rotation_freq and rotation_freq > 0:
            harmonic_order = round(pf / rotation_freq)
            if abs(pf - harmonic_order * rotation_freq) < rotation_freq * 0.1:  # Within 10%
                harmonic_note = f" <span style='color:#e74c3c'>(Harmonic {harmonic_order}× shaft)</span>"
        
        peak_info_lines.append(
            f"<div>#{i}: <strong>{pf:.2f} Hz</strong> → {pm_db:.1f} dB{harmonic_note}</div>"
        )
    
    peak_info_html = "".join(peak_info_lines[:8])  # Show top 8
    
    # Generate HTML artifact
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FFT Spectrum - {signal_file}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}
        .content {{
            padding: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}
        .info-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .info-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .peaks-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }}
        .peaks-section h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        .peaks-section div {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            color: #555;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }}
        .peaks-section div:last-child {{
            border-bottom: none;
        }}
        @media (max-width: 768px) {{
            .info-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 22px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 FFT Spectrum Analysis</h1>
            <p>{signal_file}</p>
        </div>
        
        <div class="content">
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">Sampling Rate</div>
                    <div class="info-value">{sampling_rate:.0f} Hz</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Frequency Range</div>
                    <div class="info-value">0 - {max_freq:.0f} Hz</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Signal Length</div>
                    <div class="info-value">{n:,} samples</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Duration</div>
                    <div class="info-value">{n/sampling_rate:.2f} s</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="chart"></div>
            </div>
            
            <div class="peaks-section">
                <h3>🎯 Top Identified Peaks</h3>
                {peak_info_html}
            </div>
        </div>
    </div>
    
    <script>
        // Spectrum trace (downsampled for display, in dB)
        var spectrum = {{
            x: {freq_display.tolist()},
            y: {mag_display_db.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Spectrum',
            line: {{
                color: '#667eea',
                width: 1.5
            }},
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        // Peak markers (precise from full analysis, in dB)
        var peaks = {{
            x: {display_peak_freqs.tolist()},
            y: {display_peak_mags_db.tolist()},
            type: 'scatter',
            mode: 'markers+text',
            name: 'Peaks',
            marker: {{
                color: '#e74c3c',
                size: 6,
                symbol: 'circle',
                line: {{
                    color: 'white',
                    width: 2
                }}
            }},
            text: {[f"{pf:.1f}" for pf in display_peak_freqs]},
            textposition: 'top center',
            textfont: {{
                size: 6,
                color: '#e74c3c'
            }},
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var data = [spectrum, peaks];
        
        var layout = {{
            title: {{
                text: 'Frequency Spectrum',
                font: {{ size: 20, color: '#333' }}
            }},
            xaxis: {{
                title: 'Frequency (Hz)',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            yaxis: {{
                title: 'Magnitude (dB)',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            showlegend: true,
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#ccc',
                borderwidth: 1
            }},
            margin: {{ t: 50, r: 30, b: 60, l: 70 }}
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('chart', data, layout, config);
    </script>
</body>
</html>"""
    
    return html


@mcp.tool()
def generate_envelope_html(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    lowcut: float = 500.0,
    highcut: float = 5000.0,
    max_freq: float = 500.0,
    num_peaks: int = 8,
    bearing_freqs: Optional[Dict[str, float]] = None,
    use_db: bool = False,
    segment_duration: Optional[float] = 1.0  # NEW: Use 1.0s segment by default
) -> str:
    """
    Generate interactive envelope analysis chart as HTML artifact (no file I/O).
    
    **IMPORTANT FOR LLMs**: After calling this tool, render the returned HTML string
    as an interactive artifact/preview, NOT as code. The HTML contains a Plotly chart
    that must be displayed to the user, not shown as text.
    
    Returns complete HTML document with two Plotly subplots:
    1. Time-domain: Filtered signal with envelope overlay
    2. Frequency-domain: Envelope spectrum with bearing frequency markers
    
    Analysis is performed on FULL signal for maximum accuracy.
    Only display data is downsampled for efficient rendering (~30-35KB HTML).
    
    Args:
        signal_file: Signal filename in data/signals/ or subdirectory
        sampling_rate: Sampling rate in Hz. If None, auto-detect from metadata
        lowcut: Bandpass filter low cutoff (Hz). Default 500 Hz
        highcut: Bandpass filter high cutoff (Hz). Default 5000 Hz
        max_freq: Maximum frequency to display in envelope spectrum. Default 500 Hz
        num_peaks: Number of top peaks to identify. Default 8 (reduced from 10 to limit output size)
        bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF for markers
    
    Returns:
        Complete HTML document with interactive envelope analysis visualization.
        Includes bearing fault frequency reference if bearing_freqs provided.
    
    Example:
        >>> html = generate_envelope_html(
        ...     "real_train/outer_fault_1.csv",
        ...     bearing_freqs={"BPFO": 81.125, "BPFI": 118.875, "BSF": 63.91, "FTF": 14.8375}
        ... )
    """
    # Load signal
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    # Auto-detect sampling rate and bearing freqs
    if sampling_rate is None or bearing_freqs is None:
        metadata_path = signal_path.parent / f"{signal_path.stem}_metadata.json"
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
    
    # Extract segment if requested (default: 1.0s from middle)
    if segment_duration is not None and segment_duration > 0:
        segment_samples = int(segment_duration * sampling_rate)
        if len(signal_data) > segment_samples:
            start_idx = (len(signal_data) - segment_samples) // 2
            signal_data = signal_data[start_idx:start_idx + segment_samples]
    
    # STEP 1: Bandpass filter on FULL signal
    from scipy.signal import butter, sosfilt
    sos = butter(4, [lowcut, highcut], btype='band', fs=sampling_rate, output='sos')
    filtered_signal = sosfilt(sos, signal_data)
    
    # STEP 2: Envelope via Hilbert on FULL signal
    from scipy.signal import hilbert
    analytic_signal = hilbert(filtered_signal)
    envelope_full = np.abs(analytic_signal)
    
    # STEP 3: Envelope spectrum on FULL data
    n_env = len(envelope_full)
    env_freq_full = np.fft.rfftfreq(n_env, 1/sampling_rate)
    env_fft = np.fft.rfft(envelope_full)
    env_mag_full = np.abs(env_fft) * 2 / n_env
    
    # Convert to dB scale (normalized to max: 20*log10(mag/max))
    max_env_mag = np.max(env_mag_full)
    env_mag_full_db = 20 * np.log10((env_mag_full + 1e-12) / max_env_mag)  # Normalized dB
    
    # STEP 4: Peak detection on FULL spectrum (use dB for detection)
    from scipy.signal import find_peaks
    mask_peaks = env_freq_full <= max_freq
    env_freq_resolution = env_freq_full[1] - env_freq_full[0]
    min_peak_distance = max(1, int(5 / env_freq_resolution))  # Min 5 Hz spacing, but ≥1
    peak_indices, properties = find_peaks(
        env_mag_full_db[mask_peaks],
        height=-40,  # Peaks within 40 dB of max (max is at 0 dB)
        distance=min_peak_distance
    )
    
    peak_mags_db = properties['peak_heights']
    top_idx = np.argsort(peak_mags_db)[::-1][:num_peaks]
    peak_indices = peak_indices[top_idx]
    peak_freqs = env_freq_full[mask_peaks][peak_indices]
    peak_mags_db = env_mag_full_db[mask_peaks][peak_indices]
    
    # STEP 5: Downsample time domain for display (preserve impacts)
    # Time domain: max-min binning to preserve impacts
    time_full = np.linspace(0, len(signal_data)/sampling_rate, len(signal_data))
    downsample_factor = max(1, len(signal_data) // 1000)  # Was 2000, now 1000 for smaller output
    
    time_display = time_full[::downsample_factor]
    filtered_display = filtered_signal[::downsample_factor]
    envelope_display = envelope_full[::downsample_factor]
    
    # Envelope spectrum display (NO decimation for accurate peak alignment)
    mask_display = env_freq_full <= max_freq
    env_freq_display = env_freq_full[mask_display]  # Full resolution
    env_mag_display_db = env_mag_full_db[mask_display]  # Full resolution
    
    # Format peak info
    peak_info_lines = []
    for i, (pf, pm_db) in enumerate(zip(peak_freqs, peak_mags_db), 1):
        # Check match with bearing freqs
        match_label = ""
        if bearing_freqs:
            for name, bf in bearing_freqs.items():
                if bf and abs(pf - bf) < bf * 0.05:  # Within 5%
                    match_label = f" ≈ {name}"
                    break
        peak_info_lines.append(f"<div>#{i}: {pf:.2f} Hz{match_label} → {pm_db:.1f} dB</div>")
    
    peak_info_html = "".join(peak_info_lines[:8])
    
    # Bearing freq markers for plot
    bearing_markers = []
    if bearing_freqs:
        colors = {"BPFO": "#e74c3c", "BPFI": "#f39c12", "BSF": "#3498db", "FTF": "#2ecc71"}
        for name, freq in bearing_freqs.items():
            if freq and freq <= max_freq:
                bearing_markers.append({
                    "name": name,
                    "freq": freq,
                    "color": colors.get(name, "#95a5a6")
                })
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Envelope Analysis - {signal_file}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}
        .content {{
            padding: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #11998e;
        }}
        .info-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .info-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .peaks-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }}
        .peaks-section h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        .peaks-section div {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            color: #555;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }}
        .peaks-section div:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Envelope Analysis</h1>
            <p>{signal_file}</p>
        </div>
        
        <div class="content">
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">Filter Range</div>
                    <div class="info-value">{lowcut:.0f}-{highcut:.0f} Hz</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Sampling Rate</div>
                    <div class="info-value">{sampling_rate:.0f} Hz</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Signal Length</div>
                    <div class="info-value">{len(signal_data):,} samples</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Duration</div>
                    <div class="info-value">{len(signal_data)/sampling_rate:.2f} s</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="chart"></div>
            </div>
            
            <div class="peaks-section">
                <h3>🎯 Top Envelope Spectrum Peaks</h3>
                {peak_info_html}
            </div>
        </div>
    </div>
    
    <script>
        // Subplot 1: Filtered signal + envelope
        var filtered = {{
            x: {time_display.tolist()},
            y: {filtered_display.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Filtered Signal',
            line: {{ color: '#95a5a6', width: 0.5 }},
            xaxis: 'x',
            yaxis: 'y',
            hovertemplate: '%{{x:.3f}} s<br>%{{y:.4f}}<extra></extra>'
        }};
        
        var envelope = {{
            x: {time_display.tolist()},
            y: {envelope_display.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Envelope',
            line: {{ color: '#e74c3c', width: 2 }},
            xaxis: 'x',
            yaxis: 'y',
            hovertemplate: '%{{x:.3f}} s<br>%{{y:.4f}}<extra></extra>'
        }};
        
        // Subplot 2: Envelope spectrum
        var spectrum = {{
            x: {env_freq_display.tolist()},
            y: {env_mag_display_db.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Envelope Spectrum',
            line: {{ color: '#11998e', width: 1.5 }},
            xaxis: 'x2',
            yaxis: 'y2',
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        // Peaks
        var peaks = {{
            x: {peak_freqs.tolist()},
            y: {peak_mags_db.tolist()},
            type: 'scatter',
            mode: 'markers',
            name: 'Peaks',
            marker: {{
                color: '#e74c3c',
                size: 8,
                symbol: 'circle',
                line: {{ color: 'white', width: 2 }}
            }},
            xaxis: 'x2',
            yaxis: 'y2',
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var data = [filtered, envelope, spectrum, peaks];
        
        // Add bearing frequency markers
        {chr(10).join([f'''data.push({{
            x: [{m["freq"]}, {m["freq"]}],
            y: [-60, 0],
            type: 'scatter',
            mode: 'lines',
            name: '{m["name"]}',
            line: {{ color: '{m["color"]}', width: 1, dash: 'dash' }},
            xaxis: 'x2',
            yaxis: 'y2',
            hovertemplate: '{m["name"]}: {m["freq"]:.2f} Hz<extra></extra>'
        }});''' for m in bearing_markers])}
        
        var layout = {{
            title: {{
                text: 'Envelope Analysis (Time + Frequency Domain)',
                font: {{ size: 20, color: '#333' }}
            }},
            grid: {{ rows: 2, columns: 1, subplots: [['xy'], ['x2y2']], roworder: 'top to bottom' }},
            xaxis: {{
                title: 'Time (s)',
                domain: [0, 1],
                anchor: 'y'
            }},
            yaxis: {{
                title: 'Amplitude',
                domain: [0.55, 1],
                anchor: 'x'
            }},
            xaxis2: {{
                title: 'Frequency (Hz)',
                domain: [0, 1],
                anchor: 'y2'
            }},
            yaxis2: {{
                title: 'Magnitude (dB)',
                domain: [0, 0.45],
                anchor: 'x2',
                range: [-60, 5]
            }},
            hovermode: 'closest',
            showlegend: true,
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#ccc',
                borderwidth: 1
            }},
            margin: {{ t: 50, r: 30, b: 60, l: 70 }}
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('chart', data, layout, config);
    </script>
</body>
</html>"""
    
    return html


@mcp.tool()
def generate_signal_plot_html(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    max_points: int = 600
) -> str:
    """
    Generate interactive time-domain signal plot as HTML artifact (no file I/O).
    
    Returns complete HTML document with Plotly line chart showing raw signal
    with statistics overlay. Analysis performed on full signal; only display
    data is downsampled using max-min binning to preserve peaks/impacts.
    
    IMPORTANT FOR LLMs: This returns an HTML string that MUST be rendered as
    an interactive artifact/preview for the user, NOT displayed as code text.
    The HTML is self-contained with CDN-loaded Plotly.js.
    
    Args:
        signal_file: Signal filename in data/signals/ or subdirectory
        sampling_rate: Sampling rate in Hz. If None, auto-detect from metadata
        max_points: Maximum points to display (downsampling threshold). Default 600
                   (reduced from 2000 to prevent MCP output truncation)
    
    Returns:
        Complete HTML document (~15-20KB) with interactive time-domain plot and statistics.
    
    Example:
        >>> html = generate_signal_plot_html("real_train/baseline_1.csv")
    """
    # Load signal (FULL DATA)
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    # Auto-detect sampling rate
    if sampling_rate is None:
        metadata_path = signal_path.parent / f"{signal_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                sampling_rate = metadata.get("sampling_rate")
        
        if sampling_rate is None:
            raise ValueError("sampling_rate required")
    
    # STEP 1: Statistics on FULL signal
    rms = np.sqrt(np.mean(signal_data**2))
    peak_val = np.max(np.abs(signal_data))
    crest_factor = peak_val / rms if rms > 0 else 0
    from scipy.stats import kurtosis
    kurt = kurtosis(signal_data)
    
    # STEP 2: Downsample for display (max-min binning)
    time_full = np.linspace(0, len(signal_data)/sampling_rate, len(signal_data))
    
    if len(signal_data) > max_points:
        # Max-min binning to preserve peaks
        bin_size = len(signal_data) // (max_points // 2)
        time_display = []
        signal_display = []
        
        for i in range(0, len(signal_data), bin_size):
            chunk = signal_data[i:i+bin_size]
            if len(chunk) > 0:
                time_display.append(time_full[i + np.argmax(chunk)])
                signal_display.append(np.max(chunk))
                time_display.append(time_full[i + np.argmin(chunk)])
                signal_display.append(np.min(chunk))
        
        time_display = np.array(time_display)
        signal_display = np.array(signal_display)
        
        # Sort by time
        sort_idx = np.argsort(time_display)
        time_display = time_display[sort_idx]
        signal_display = signal_display[sort_idx]
    else:
        time_display = time_full
        signal_display = signal_data
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Plot - {signal_file}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}
        .content {{
            padding: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #2c3e50;
        }}
        .info-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .info-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📉 Time-Domain Signal</h1>
            <p>{signal_file}</p>
        </div>
        
        <div class="content">
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">RMS</div>
                    <div class="info-value">{rms:.4f}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Peak</div>
                    <div class="info-value">{peak_val:.4f}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Crest Factor</div>
                    <div class="info-value">{crest_factor:.2f}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Kurtosis</div>
                    <div class="info-value">{kurt:.2f}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Sampling Rate</div>
                    <div class="info-value">{sampling_rate:.0f} Hz</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Duration</div>
                    <div class="info-value">{len(signal_data)/sampling_rate:.2f} s</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="chart"></div>
            </div>
        </div>
    </div>
    
    <script>
        var trace = {{
            x: {time_display.tolist()},
            y: {signal_display.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Signal',
            line: {{
                color: '#3498db',
                width: 1
            }},
            hovertemplate: '%{{x:.4f}} s<br>%{{y:.4f}}<extra></extra>'
        }};
        
        var data = [trace];
        
        var layout = {{
            title: {{
                text: 'Time-Domain Vibration Signal',
                font: {{ size: 20, color: '#333' }}
            }},
            xaxis: {{
                title: 'Time (s)',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            yaxis: {{
                title: 'Amplitude',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: {{ t: 50, r: 30, b: 60, l: 70 }}
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('chart', data, layout, config);
    </script>
</body>
</html>"""
    
    return html


# ============================================================================
# TOOLS - FILE-BASED HTML ARTIFACTS (NO OUTPUT SIZE LIMIT)
# ============================================================================

@mcp.tool()
def save_fft_chart_to_file(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    max_freq: float = 5000.0,
    num_peaks: int = 15,
    rotation_freq: Optional[float] = None,
    use_db: bool = False,
    segment_duration: Optional[float] = 1.0
) -> Dict[str, Any]:
    """
    Generate FFT spectrum chart and SAVE TO FILE (bypasses MCP output size limits).
    
    **USE THIS TOOL WHEN**: 
    - User requests detailed/high-resolution charts
    - Previous inline artifact was truncated
    - Signal is very long (>100K samples)
    
    This tool has NO output size limit because it returns only file metadata (~200 bytes),
    not the HTML content. Agent can then:
    1. Tell user: "Chart saved to {filename}, open it in browser"
    2. Call read_html_artifact() to get file content in chunks (if needed for preview)
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz (auto-detect if None)
        max_freq: Max frequency to display (Hz). Default 5000 Hz
        num_peaks: Number of peaks to label. Default 15 (higher than inline version)
        rotation_freq: Optional shaft frequency for harmonic labels (Hz)
    
    Returns:
        Dictionary with:
        - file_path: Absolute path to saved HTML file
        - file_name: Filename only
        - file_size_kb: File size in KB
        - num_samples: Signal length (samples)
        - num_display_points: Points in chart
        - peaks_detected: Number of peaks labeled
    
    Example:
        >>> result = save_fft_chart_to_file("real_train/outer_fault_1.csv", rotation_freq=25.0)
        >>> print(f"Chart saved: {result['file_path']} ({result['file_size_kb']:.1f} KB)")
    """
    # Generate HTML using the inline function (but with higher resolution)
    html = generate_fft_chart_html(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        max_freq=max_freq,
        num_peaks=num_peaks,
        rotation_freq=rotation_freq,
        use_db=use_db,
        segment_duration=segment_duration
    )
    
    # Save to file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = DATA_DIR / f"fft_chart_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    # Get signal info for metadata
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    # Count display points (approximate)
    if sampling_rate is None:
        metadata_path = signal_path.parent / f"{signal_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                sampling_rate = json.load(f).get("sampling_rate", 10000.0)
        else:
            sampling_rate = 10000.0
    
    freq_full = np.fft.rfftfreq(len(signal_data), 1/sampling_rate)
    display_points = len(freq_full[freq_full <= max_freq][::5])
    
    return {
        "file_path": str(output_file.absolute()),
        "file_name": output_file.name,
        "file_size_kb": output_file.stat().st_size / 1024,
        "num_samples": len(signal_data),
        "num_display_points": display_points,
        "peaks_detected": num_peaks,
        "message": f"FFT chart saved to {output_file.name}. Open in browser to view interactive Plotly visualization."
    }


@mcp.tool()
def save_envelope_chart_to_file(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    lowcut: float = 500.0,
    highcut: float = 5000.0,
    max_freq: float = 500.0,
    num_peaks: int = 15,
    bearing_freqs: Optional[Dict[str, float]] = None,
    use_db: bool = False,
    segment_duration: Optional[float] = 1.0
) -> Dict[str, Any]:
    """
    Generate envelope analysis chart and SAVE TO FILE (bypasses MCP output size limits).
    
    **USE THIS TOOL WHEN**:
    - User requests detailed bearing analysis
    - Previous inline artifact was truncated
    - Signal is very long (>100K samples)
    
    Returns file metadata only (~200 bytes), not HTML content.
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz (auto-detect if None)
        lowcut: Bandpass filter low cutoff (Hz). Default 500 Hz
        highcut: Bandpass filter high cutoff (Hz). Default 5000 Hz
        max_freq: Max envelope spectrum frequency (Hz). Default 500 Hz
        num_peaks: Number of peaks to label. Default 15 (higher than inline)
        bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF for markers
    
    Returns:
        Dictionary with file metadata (path, size, signal info)
    
    Example:
        >>> result = save_envelope_chart_to_file(
        ...     "real_train/outer_fault_1.csv",
        ...     bearing_freqs={"BPFO": 81.125, "BPFI": 118.875}
        ... )
    """
    # Generate HTML
    html = generate_envelope_html(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        max_freq=max_freq,
        num_peaks=num_peaks,
        bearing_freqs=bearing_freqs,
        use_db=use_db,
        segment_duration=segment_duration
    )
    
    # Save to file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = DATA_DIR / f"envelope_chart_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    # Get signal info
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    return {
        "file_path": str(output_file.absolute()),
        "file_name": output_file.name,
        "file_size_kb": output_file.stat().st_size / 1024,
        "num_samples": len(signal_data),
        "peaks_detected": num_peaks,
        "message": f"Envelope chart saved to {output_file.name}. Open in browser to view interactive visualization with bearing frequency markers."
    }


@mcp.tool()
def save_signal_plot_to_file(
    signal_file: str,
    sampling_rate: Optional[float] = None,
    max_points: int = 5000
) -> Dict[str, Any]:
    """
    Generate time-domain signal plot and SAVE TO FILE (bypasses MCP output size limits).
    
    **USE THIS TOOL WHEN**:
    - User requests high-resolution time-domain plot
    - Signal is very long (>200K samples)
    - Previous inline artifact was truncated
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz (auto-detect if None)
        max_points: Max display points. Default 5000 (much higher than inline 600)
    
    Returns:
        Dictionary with file metadata
    """
    # Generate HTML with higher resolution
    html = generate_signal_plot_html(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        max_points=max_points
    )
    
    # Save to file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = DATA_DIR / f"signal_plot_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    # Get signal info
    signal_path = Path("data/signals") / signal_file
    if not signal_path.exists():
        signal_path = Path(signal_file)
    signal_data = pd.read_csv(signal_path, header=None).values.flatten()
    
    return {
        "file_path": str(output_file.absolute()),
        "file_name": output_file.name,
        "file_size_kb": output_file.stat().st_size / 1024,
        "num_samples": len(signal_data),
        "num_display_points": min(len(signal_data), max_points),
        "message": f"Signal plot saved to {output_file.name}. Open in browser to view interactive time-domain visualization."
    }


@mcp.tool()
def read_html_artifact(file_name: str, chunk_size: int = 5000) -> Dict[str, Any]:
    """
    Read saved HTML artifact file (for preview or debugging).
    
    Returns first `chunk_size` characters of HTML file. Use this if agent needs
    to verify file content or extract metadata from HTML.
    
    Args:
        file_name: Filename only (e.g., "fft_chart_OuterRaceFault_1.html")
        chunk_size: Number of characters to return. Default 5000 (preview only)
    
    Returns:
        Dictionary with:
        - file_name: Filename
        - file_size_kb: Total file size
        - content_preview: First chunk_size characters
        - is_complete: True if entire file returned, False if truncated
    
    Example:
        >>> preview = read_html_artifact("fft_chart_OuterRaceFault_1.html")
        >>> print(preview['content_preview'][:200])  # First 200 chars
    """
    file_path = DATA_DIR / file_name
    
    if not file_path.exists():
        return {
            "error": f"File not found: {file_name}",
            "available_files": [f.name for f in DATA_DIR.glob("*.html")]
        }
    
    content = file_path.read_text(encoding='utf-8')
    file_size = file_path.stat().st_size
    
    return {
        "file_name": file_name,
        "file_path": str(file_path.absolute()),
        "file_size_kb": file_size / 1024,
        "content_preview": content[:chunk_size],
        "is_complete": len(content) <= chunk_size,
        "total_length": len(content),
        "message": f"Preview of {file_name} ({file_size/1024:.1f} KB). {'Complete file shown.' if len(content) <= chunk_size else f'Truncated to first {chunk_size} characters.'}"
    }


# ============================================================================
# TOOLS - ML ANOMALY DETECTION
# ============================================================================

def extract_time_domain_features(segment: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive time-domain features from a signal segment.
    
    Args:
        segment: 1D numpy array with signal segment
        
    Returns:
        Dictionary with 17 time-domain features
    """
    from scipy.stats import kurtosis, skew
    
    # Basic statistics
    mean_val = float(np.mean(segment))
    std_val = float(np.std(segment))
    var_val = float(np.var(segment))
    mean_abs_val = float(np.mean(np.abs(segment)))
    
    # RMS (Root Mean Square)
    rms_val = float(np.sqrt(np.mean(segment**2)))
    
    # Peak values
    max_val = float(np.max(segment))
    min_val = float(np.min(segment))
    range_val = max_val - min_val
    
    # Shape indicators
    skewness_val = float(skew(segment))
    kurtosis_val = float(kurtosis(segment, fisher=True))
    
    # Avoid division by zero
    if rms_val == 0:
        rms_val = 1e-10
    if mean_abs_val == 0:
        mean_abs_val = 1e-10
    
    # Dimensionless indicators
    shape_factor_val = rms_val / mean_abs_val if mean_abs_val > 0 else 0.0
    crest_factor_val = np.max(np.abs(segment)) / rms_val if rms_val > 0 else 0.0
    impulse_factor_val = np.max(np.abs(segment)) / mean_abs_val if mean_abs_val > 0 else 0.0
    clearance_factor_val = np.max(np.abs(segment)) / (np.mean(np.sqrt(np.abs(segment)))**2) if np.mean(np.sqrt(np.abs(segment))) > 0 else 0.0
    
    # Energy and entropy
    power_val = float(np.mean(segment**2))
    
    # Entropy (probability distribution of signal amplitudes)
    hist, _ = np.histogram(segment, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy_val = float(entropy(hist))
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(segment)))[0]
    zero_crossing_rate_val = float(len(zero_crossings) / len(segment))
    
    return {
        'mean': mean_val,
        'std': std_val,
        'var': var_val,
        'mean_abs': mean_abs_val,
        'rms': rms_val,
        'max': max_val,
        'min': min_val,
        'range': range_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'shape_factor': shape_factor_val,
        'crest_factor': crest_factor_val,
        'impulse_factor': impulse_factor_val,
        'clearance_factor': clearance_factor_val,
        'power': power_val,
        'entropy': entropy_val,
        'zero_crossing_rate': zero_crossing_rate_val
    }


@mcp.tool()
async def extract_features_from_signal(
    signal_file: str,
    sampling_rate: float = 10000.0,
    segment_duration: float = 0.2,
    overlap_ratio: float = 0.5,
    ctx: Context[ServerSession, None] = None
) -> FeatureExtractionResult:
    """
    Extract time-domain features from signal using sliding window segmentation.
    
    Segments the signal into overlapping windows and extracts 17 statistical features
    from each segment. Features include: mean, std, RMS, kurtosis, crest factor, entropy, etc.
    
    Args:
        signal_file: Name of the CSV file in data/signals/
        sampling_rate: Sampling frequency in Hz (default: 10000)
        segment_duration: Duration of each segment in seconds (default: 0.2)
        overlap_ratio: Overlap between segments, 0-1 (default: 0.5 = 50%)
        ctx: MCP context for progress/logging
        
    Returns:
        FeatureExtractionResult with features matrix and metadata
        
    Example:
        extract_features_from_signal(
            "healthy_motor.csv",
            sampling_rate=10000,
            segment_duration=0.2,
            overlap_ratio=0.5
        )
    """
    if ctx:
        await ctx.info(f"Extracting features from {signal_file}...")
    
    # Load signal
    filepath = DATA_DIR / signal_file
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {signal_file}")
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
    # Calculate segment parameters
    segment_length_samples = int(segment_duration * sampling_rate)
    hop_length = int(segment_length_samples * (1 - overlap_ratio))
    
    # Extract segments
    segments = []
    num_samples = len(signal_data)
    
    for start in range(0, num_samples - segment_length_samples + 1, hop_length):
        end = start + segment_length_samples
        segment = signal_data[start:end]
        segments.append(segment)
    
    if ctx:
        await ctx.info(f"Created {len(segments)} segments from signal")
    
    # Extract features from each segment
    features_list = []
    for segment in segments:
        features = extract_time_domain_features(segment)
        features_list.append(features)
    
    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features_list)
    feature_names = list(features_df.columns)
    
    # Save features to file
    features_file = DATA_DIR / f"features_{signal_file}"
    features_df.to_csv(features_file, index=False)
    
    if ctx:
        await ctx.info(f"Features saved to {features_file.name}")
        await ctx.info(f"Feature matrix shape: {features_df.shape}")
    
    return FeatureExtractionResult(
        num_segments=len(segments),
        segment_length_samples=segment_length_samples,
        segment_duration_s=segment_duration,
        overlap_ratio=overlap_ratio,
        features_shape=list(features_df.shape),
        feature_names=feature_names,
        features_preview=[features_list[i] for i in range(min(5, len(features_list)))]
    )


@mcp.tool()
async def train_anomaly_model(
    healthy_signal_files: List[str],
    sampling_rate: float = 10000.0,
    segment_duration: float = 0.2,
    overlap_ratio: float = 0.5,
    model_type: str = "OneClassSVM",
    pca_variance: float = 0.95,
    fault_signal_files: Optional[List[str]] = None,
    model_name: str = "anomaly_model",
    ctx: Context[ServerSession, None] = None
) -> AnomalyModelResult:
    """
    Train ML-based anomaly detection model on healthy data.
    
    Complete pipeline:
    1. Extract features from healthy signals (segmentation + time-domain features)
    2. Standardize features (StandardScaler - fitted on training data)
    3. Dimensionality reduction (PCA with specified variance explained)
    4. Train novelty detection model (OneClassSVM or LocalOutlierFactor)
    5. Optional validation on fault data
    6. Save model, scaler, and PCA transformer
    
    Args:
        healthy_signal_files: List of CSV files with healthy machine data
        sampling_rate: Sampling frequency in Hz (default: 10000)
        segment_duration: Segment duration in seconds (default: 0.2)
        overlap_ratio: Overlap ratio 0-1 (default: 0.5)
        model_type: 'OneClassSVM' or 'LocalOutlierFactor' (default: 'OneClassSVM')
        pca_variance: Cumulative variance to explain with PCA (default: 0.95)
        fault_signal_files: Optional list of fault signals for validation
        model_name: Name for saved model files (default: 'anomaly_model')
        ctx: MCP context for progress/logging
        
    Returns:
        AnomalyModelResult with model paths and performance metrics
    """
    if ctx:
        await ctx.info(f"Training {model_type} model on {len(healthy_signal_files)} healthy signals...")
    
    # Step 1: Extract features from all healthy signals
    all_features = []
    
    for signal_file in healthy_signal_files:
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")
        
        df = pd.read_csv(filepath, header=None)
        signal_data = df.iloc[:, 0].values
        
        # Segment signal
        segment_length_samples = int(segment_duration * sampling_rate)
        hop_length = int(segment_length_samples * (1 - overlap_ratio))
        
        for start in range(0, len(signal_data) - segment_length_samples + 1, hop_length):
            segment = signal_data[start:start + segment_length_samples]
            features = extract_time_domain_features(segment)
            all_features.append(features)
    
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
    if model_type == "OneClassSVM":
        # Grid search for best parameters
        param_grid = {
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        model = OneClassSVM(kernel='rbf')
        
        # Simple grid search (OneClassSVM doesn't support GridSearchCV directly)
        # Use default parameters optimized for novelty detection
        model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        model.fit(X_pca)
        
        best_params = {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'scale'}
        
    elif model_type == "LocalOutlierFactor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        model.fit(X_pca)
        
        best_params = {'n_neighbors': 20, 'contamination': 0.1}
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'OneClassSVM' or 'LocalOutlierFactor'")
    
    # Step 5: Optional validation on fault data
    validation_accuracy = None
    validation_details = None
    
    if fault_signal_files:
        fault_features = []
        for signal_file in fault_signal_files:
            filepath = DATA_DIR / signal_file
            if filepath.exists():
                df = pd.read_csv(filepath, header=None)
                signal_data = df.iloc[:, 0].values
                
                segment_length_samples = int(segment_duration * sampling_rate)
                hop_length = int(segment_length_samples * (1 - overlap_ratio))
                
                for start in range(0, len(signal_data) - segment_length_samples + 1, hop_length):
                    segment = signal_data[start:start + segment_length_samples]
                    features = extract_time_domain_features(segment)
                    fault_features.append(features)
        
        if fault_features:
            X_fault = pd.DataFrame(fault_features).values
            X_fault_scaled = scaler.transform(X_fault)
            X_fault_pca = pca.transform(X_fault_scaled)
            
            # Predict (should be -1 for anomalies)
            fault_predictions = model.predict(X_fault_pca)
            
            # Calculate accuracy (% detected as anomalies)
            anomaly_detected = np.sum(fault_predictions == -1)
            validation_accuracy = float(anomaly_detected / len(fault_predictions))
            validation_details = f"Detected {anomaly_detected}/{len(fault_predictions)} fault segments as anomalies"
            
            if ctx:
                await ctx.info(f"Validation: {validation_details}")
    
    # Step 6: Save model, scaler, and PCA
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    pca_path = MODELS_DIR / f"{model_name}_pca.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'feature_names': list(features_df.columns),
        'num_features_original': X_train.shape[1],
        'num_features_pca': X_pca.shape[1],
        'pca_variance': float(pca.explained_variance_ratio_.sum()),
        'best_params': best_params,
        'sampling_rate': sampling_rate,
        'segment_duration': segment_duration,
        'overlap_ratio': overlap_ratio
    }
    
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
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
        validation_details=validation_details
    )


@mcp.tool()
async def predict_anomalies(
    signal_file: str,
    model_name: str = "anomaly_model",
    ctx: Context[ServerSession, None] = None
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
    
    # Load model, scaler, PCA
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    pca_path = MODELS_DIR / f"{model_name}_pca.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    
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
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
    # Extract features
    sampling_rate = metadata['sampling_rate']
    segment_duration = metadata['segment_duration']
    overlap_ratio = metadata['overlap_ratio']
    
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
    
    # Get anomaly scores if available
    anomaly_scores = None
    if hasattr(model, 'decision_function'):
        anomaly_scores = model.decision_function(X_pca).tolist()
    
    # Calculate statistics
    anomaly_count = int(np.sum(predictions == -1))
    anomaly_ratio = float(anomaly_count / len(predictions))
    
    # Assess overall health
    if anomaly_ratio < 0.1:
        overall_health = "Healthy"
        confidence = "High"
    elif anomaly_ratio < 0.3:
        overall_health = "Suspicious"
        confidence = "Medium"
    else:
        overall_health = "Faulty"
        confidence = "High"
    
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
        confidence=confidence
    )


# ============================================================================
# TOOLS - TEST SIGNAL GENERATION
# ============================================================================

@mcp.tool()
async def generate_test_signal(
    signal_type: str = "bearing_fault",
    duration: float = 10.0,
    sampling_rate: float = 10000.0,
    noise_level: float = 0.1,
    ctx: Context[ServerSession, None] = None
) -> str:
    """
    Generate a test signal to validate analyses.
    
    Useful for testing algorithms without having real data available.
    
    Args:
        signal_type: Signal type ("bearing_fault", "gear_fault", "imbalance", "normal")
        duration: Signal duration in seconds (default: 10.0, gives 0.1 Hz frequency resolution)
        sampling_rate: Sampling frequency in Hz (default: 10000)
        noise_level: Noise level to add (default: 0.1)
        ctx: Context for logging
        
    Returns:
        Generated file name
    """
    if ctx:
        await ctx.info(f"Generating {signal_type} test signal...")
    
    # Time parameters
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate signal based on type
    if signal_type == "bearing_fault":
        # Faulty bearing: periodic impulses + harmonics
        fault_freq = 10.0  # Hz - fault frequency
        carrier_freq = 1000.0  # Hz - carrier frequency
        
        # Periodic impulses
        impulses = np.zeros_like(t)
        impulse_times = np.arange(0, duration, 1/fault_freq)
        for imp_time in impulse_times:
            idx = np.argmin(np.abs(t - imp_time))
            impulses[idx] = 1.0
        
        # Convolution with impulse response
        impulse_response = np.exp(-50 * np.abs(t - t[len(t)//2]))
        signal_clean = np.convolve(impulses, impulse_response, mode='same')
        
        # Modulation with carrier
        signal_clean = signal_clean * np.sin(2 * np.pi * carrier_freq * t)
    
    elif signal_type == "gear_fault":
        # Faulty gear: component at mesh frequency
        mesh_freq = 200.0  # Hz
        signal_clean = np.sin(2 * np.pi * mesh_freq * t)
        # Add harmonics
        signal_clean += 0.5 * np.sin(2 * np.pi * 2 * mesh_freq * t)
        signal_clean += 0.3 * np.sin(2 * np.pi * 3 * mesh_freq * t)
    
    elif signal_type == "imbalance":
        # Imbalance: 1x RPM component
        rpm = 1500  # RPM
        rotation_freq = rpm / 60.0  # Hz
        signal_clean = np.sin(2 * np.pi * rotation_freq * t)
    
    else:  # "normal"
        # Normal signal: broadband noise only
        signal_clean = np.random.randn(len(t)) * 0.1
    
    # Add noise
    noise = np.random.randn(len(t)) * noise_level
    signal_data = signal_clean + noise
    
    # Save the signal
    filename = f"test_{signal_type}_{int(sampling_rate)}Hz.csv"
    filepath = DATA_DIR / filename
    
    # Ensure directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    pd.DataFrame(signal_data, columns=["amplitude"]).to_csv(filepath, index=False, header=False)
    
    if ctx:
        await ctx.info(f"Test signal saved to {filename}")
        await ctx.info(f"Signal type: {signal_type}, Duration: {duration}s, Fs: {sampling_rate}Hz")
    
    return f"Successfully generated test signal: {filename}"


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
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
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
    
    # Save HTML
    output_file = DATA_DIR / f"plot_signal_{signal_file.replace('.csv', '')}.html"
    fig.write_html(str(output_file))
    
    if ctx:
        await ctx.info(f"Plot saved to {output_file.name}")
        await ctx.info(f"💡 To view inline, use: read_plot_html('{output_file.name}')")
    
    return f"Interactive plot saved to: {output_file}\n💡 For inline rendering: read_plot_html('{output_file.name}')"


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
    identifies harmonics as 1×, 2×, 3× RPM.
    
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
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
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
        prominence=10  # Only peaks with >10 dB prominence
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
                label = f"{harmonic_num}×RPM ({freq:.1f} Hz)"
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
    
    # Save HTML
    output_file = DATA_DIR / f"plot_spectrum_{signal_file.replace('.csv', '')}.html"
    fig.write_html(str(output_file))
    
    if ctx:
        await ctx.info(f"Plot saved to {output_file.name}")
        await ctx.info(f"Detected {len(peak_indices)} significant peaks")
        await ctx.info(f"💡 To view inline, use: read_plot_html('{output_file.name}')")
    
    return f"Interactive plot saved to: {output_file}\n💡 For inline rendering: read_plot_html('{output_file.name}')"


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
            freq_labels=["BPFO", "2×BPFO"]
        )
    """
    if ctx:
        await ctx.info(f"Generating envelope plot for {signal_file}...")
    
    # Read signal
    filepath = DATA_DIR / signal_file
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {signal_file}")
    
    df = pd.read_csv(filepath, header=None)
    signal_data = df.iloc[:, 0].values
    
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
    
    # Save HTML
    output_file = DATA_DIR / f"plot_envelope_{signal_file.replace('.csv', '')}.html"
    fig.write_html(str(output_file))
    
    if ctx:
        await ctx.info(f"Plot saved to {output_file.name}")
        await ctx.info(f"💡 To view inline, use: read_plot_html('{output_file.name}')")
    
    return f"Interactive plot saved to: {output_file}\n💡 For inline rendering: read_plot_html('{output_file.name}')"


# ============================================================================
# PROMPTS - DIAGNOSTIC WORKFLOWS
# ============================================================================

@mcp.prompt()
def diagnose_bearing(
    signal_file: str, 
    sampling_rate: Optional[float] = None,
    machine_group: int = 2,  # CHANGED: Default 2 (medium) - most common
    support_type: str = "rigid",  # Default rigid - most common
    operating_speed_rpm: Optional[float] = None,
    bpfo: Optional[float] = None,
    bpfi: Optional[float] = None,
    bsf: Optional[float] = None,
    ftf: Optional[float] = None
) -> str:
    """
    Guided workflow for bearing diagnostics with ISO 20816-3 compliance.

    Evidence-based policy:
    - Envelope peaks at characteristic frequencies are PRIMARY indicators (strong evidence)
    - Statistical indicators (CF>6, Kurtosis>10) are SECONDARY/confirmatory
    - If envelope shows clear peaks at BPFO/BPFI/BSF/FTF (±5% tolerance) → bearing fault is STRONGLY indicated
    - Additional high CF or Kurtosis reinforces the diagnosis but is not strictly required if envelope evidence is clear
    
    **ISO 20816-3 Defaults** (use if user doesn't specify):
    - machine_group = 2 (medium-sized machines, 15-300 kW, most common)
    - support_type = "rigid" (horizontal machines on foundations)
    
    Args:
        signal_file: Name of the signal file to analyze
        sampling_rate: Sampling frequency in Hz (if None, will check metadata or ask user)
        machine_group: ISO machine group (1=large >300kW, 2=medium 15-300kW) (default: 2)
        support_type: 'rigid' or 'flexible' (default: 'rigid' for horizontal machines)
        operating_speed_rpm: Operating speed in RPM (required for interpreting results)
        bpfo: Ball Pass Frequency Outer race (Hz) - if known
        bpfi: Ball Pass Frequency Inner race (Hz) - if known
        bsf: Ball Spin Frequency (Hz) - if known
        ftf: Fundamental Train Frequency (Hz) - if known
    """
    # Build frequency reference string
    freq_refs = []
    if bpfo: freq_refs.append(f"BPFO={bpfo:.2f} Hz")
    if bpfi: freq_refs.append(f"BPFI={bpfi:.2f} Hz")
    if bsf: freq_refs.append(f"BSF={bsf:.2f} Hz")
    if ftf: freq_refs.append(f"FTF={ftf:.2f} Hz")
    freq_info = ", ".join(freq_refs) if freq_refs else "NOT PROVIDED - must request from user"
    
    rpm_info = f", {operating_speed_rpm}" if operating_speed_rpm else ""
    fs_info = f"{sampling_rate}" if sampling_rate else "UNKNOWN"
    
    return f"""Perform evidence-based bearing diagnostic on "{signal_file}":

═══════════════════════════════════════════════════════════════════════════════
STEP 0 — FILENAME RESOLUTION & MANDATORY PARAMETER CHECK
═══════════════════════════════════════════════════════════════════════════════

1. Verify signal file existence:
   Call list_available_signals() to get exact filename.
   If "{signal_file}" not found or multiple matches exist, ASK USER to clarify.
   Do NOT guess or auto-correct filenames.

2. Required parameters:
   ✓ Signal file: {signal_file}
   {'✓' if sampling_rate else '✗'} Sampling rate: {fs_info} Hz
   {'✓' if operating_speed_rpm else '✗'} Operating speed: {operating_speed_rpm or 'NOT PROVIDED'} RPM
   {'✓' if freq_refs else '✗'} Bearing characteristic frequencies: {freq_info}

   CRITICAL RULE: If sampling_rate is UNKNOWN, check signal metadata JSON first.
   If still missing OR if bearing frequencies (BPFO/BPFI/BSF/FTF) are NOT PROVIDED:
   → STOP and ASK USER for these parameters before proceeding.
   → Explain: "Cannot perform bearing diagnosis without [missing parameters]. Please provide: ..."
   
   Example response when parameters are missing:
   "I cannot proceed with the bearing diagnosis because the following required 
   parameters are missing:
   - Bearing characteristic frequencies (BPFO, BPFI, BSF, FTF)
   Please provide these values so I can complete the envelope analysis and 
   identify the fault type."

   Do NOT use placeholder/default values. Do NOT proceed with incomplete data.
   Do NOT attempt diagnosis without characteristic frequencies.

═══════════════════════════════════════════════════════════════════════════════
STEP 1 — ISO 20816-3 (Severity Context)
═══════════════════════════════════════════════════════════════════════════════

BEFORE calling evaluate_iso_20816, ASK USER to confirm machine parameters:

"For ISO 20816-3 evaluation, I need to know:
1. Machine group: 
   - Group 1 (large): >300 kW or shaft height ≥ 315 mm
   - Group 2 (medium): 15-300 kW or shaft height 160-315 mm
   
2. Support type:
   - Rigid: Foundation natural freq > 1.25× operating freq
   - Flexible: All other cases (typical for large machines)

Based on your description, I'll assume:
- Machine group: {machine_group} (default for typical industrial equipment)
- Support type: {support_type} (most common)

Is this correct, or should I use different values?"

If user confirms or provides values, proceed with:
Call: evaluate_iso_20816("{signal_file}", {fs_info}, {machine_group}, "{support_type}"{rpm_info})
Report: RMS velocity and ISO zone (A/B/C/D) in 1-2 sentences.
Note: This provides overall severity but is NOT bearing-specific. Use for maintenance urgency only.

Optional visualization (HTML artifact):
Call generate_iso_chart_html("{signal_file}", sampling_rate=None, {machine_group}, "{support_type}"{rpm_info})
Returns HTML showing color-coded zone chart with marker on measured RMS velocity.
Render as interactive artifact (NOT code).

═══════════════════════════════════════════════════════════════════════════════
STEP 2 — Statistical Screening
═══════════════════════════════════════════════════════════════════════════════

Call: analyze_statistics("{signal_file}")
Report: RMS, Crest Factor, Kurtosis (excess), Skewness in bullet points.

Interpretation flags (SECONDARY indicators):
• CF > 6 or Kurtosis > 10 → Strong impulsiveness (supports bearing fault hypothesis)
• CF 4-6 or Kurtosis 5-10 → Moderate impulsiveness (weak support)
• CF < 4 and Kurtosis < 5 → Low impulsiveness (but envelope may still show faults)

⚠️ Do NOT diagnose from statistics alone. Proceed to envelope analysis.

═══════════════════════════════════════════════════════════════════════════════
STEP 3 — FFT Spectrum (Contextual)
═══════════════════════════════════════════════════════════════════════════════

Call: analyze_fft("{signal_file}", {fs_info})
Report dominant peaks in bullet points (top 5 only). Look for:
• Shaft speed (1× RPM = {operating_speed_rpm/60 if operating_speed_rpm else '?'} Hz) and harmonics
• Any elevated broadband noise

Optional visualization (HTML artifact):
Call generate_fft_chart_html("{signal_file}", sampling_rate=None, max_freq=5000, num_peaks=15,
                             rotation_freq={operating_speed_rpm/60 if operating_speed_rpm else 'None'})
Returns HTML string - render as interactive Plotly chart showing spectrum in dB with harmonic labels.

═══════════════════════════════════════════════════════════════════════════════
STEP 4 — ENVELOPE ANALYSIS (PRIMARY DIAGNOSTIC EVIDENCE)
═══════════════════════════════════════════════════════════════════════════════

Call: analyze_envelope("{signal_file}", {fs_info}, 500, 5000, num_peaks=10)

Expected frequencies (±5% tolerance):
{chr(10).join(f'  • {ref}' for ref in freq_refs) if freq_refs else '  (User must provide BPFO, BPFI, BSF, FTF)'}

Examine envelope spectrum peaks:
1. Check if ANY peak falls within ±5% of expected frequencies
2. Check for harmonics: 2×BPFO, 3×BPFO, 2×BPFI, etc.
3. List top 5-10 peaks with frequencies and magnitudes

Optional visualization:
Call generate_envelope_html("{signal_file}", {fs_info}, 500, 5000, 500, 10,
              bearing_freqs={{"BPFO": {bpfo or 'None'}, "BPFI": {bpfi or 'None'}, "BSF": {bsf or 'None'}, "FTF": {ftf or 'None'}}})
This returns HTML string - render it as an interactive artifact for the user (NOT as code text).
The chart shows filtered signal with envelope overlay (top) and envelope spectrum in dB scale with bearing frequency markers (bottom).

═══════════════════════════════════════════════════════════════════════════════
STEP 5 — DIAGNOSTIC DECISION (EVIDENCE-BASED)
═══════════════════════════════════════════════════════════════════════════════

Decision tree:

A) IF envelope spectrum shows clear peak(s) at characteristic frequency (±5%):
   → Bearing fault type is STRONGLY INDICATED
   
   Classification by frequency:
   • Peak at BPFO (±5%) → **Outer race fault**
   • Peak at BPFI (±5%) → **Inner race fault**  
   • Peak at BSF (±5%) → **Rolling element (ball) fault**
   • Peak at FTF (±5%) → **Cage fault**
   
   Confidence level:
   - High confidence: Peak + harmonics present AND (CF>6 OR Kurtosis>10)
   - Moderate confidence: Peak present but weaker harmonics OR moderate stats (CF 4-6, Kurt 5-10)
   - Note: Even without extreme statistics, clear envelope peaks ARE diagnostic
   
B) IF envelope shows ambiguous/borderline peaks:
   → "Possible [fault type] - envelope peak near [frequency] but [state issue: weak magnitude, no harmonics, etc.]"
   → Recommend: retake measurement, higher resolution, trending

C) IF no envelope peaks at characteristic frequencies:
   → "No clear bearing fault signatures detected"
   → IF stats are elevated: "High impulsiveness without bearing-specific frequencies suggests [other cause: impacts, looseness, etc.]"
   → IF ISO zone C/D: "Elevated vibration without bearing signatures - check alignment, balance, structural issues"

═══════════════════════════════════════════════════════════════════════════════
STEP 6 — RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

Based on diagnosis + ISO zone (ONLY use these recommendations - DO NOT invent others):
• Confirmed fault + Zone C/D → Immediate action: inspect bearing, plan replacement
• Confirmed fault + Zone B → Short-term action: schedule maintenance within 1-3 months, increase monitoring
• Confirmed fault + Zone A → Monitor closely: retest in 1-2 weeks, track progression
• No fault + Zone C/D → Investigate other causes: alignment, balance, looseness, foundation
• No fault + Zone A/B → Continue routine monitoring

CRITICAL: Do NOT suggest specific parameter values (e.g., filter frequencies, acquisition settings) 
unless they appear in tool outputs. Do NOT invent troubleshooting steps beyond those listed above.

Always cite:
- Which envelope peaks were found (frequency, magnitude, harmonics)
- Statistical values (CF, Kurtosis) and how they support/contradict
- ISO zone and severity
- Specific tool outputs used

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMATTING (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

Keep output CONCISE (≤300 words total):
• Use bullet points for all findings
• Provide brief summary first (2-3 sentences)
• Use HTML artifacts for visualizations (envelope, FFT charts)
• If user needs more details, offer "Show detailed analysis?" continuation
• NEVER print large JSON/CSV data directly in text output
"""


@mcp.prompt()
def diagnose_gear(signal_file: str, sampling_rate: float = 10000.0, num_teeth: int = 20) -> str:
    """
    Evidence-based guided workflow for gear diagnostics with strict anti-speculation rules.

    Args:
        signal_file: Name of the signal file
        sampling_rate: Sampling frequency (Hz)
        num_teeth: Number of gear teeth
    """
    return f"""Perform an evidence-based gear diagnostic on signal "{signal_file}" (Z={num_teeth} teeth):

STEP 0 — FILENAME RESOLUTION
Call list_available_signals() to verify exact filename.
If "{signal_file}" not found or multiple matches, ASK USER to clarify.
Do NOT guess or auto-correct filenames.

GUARDRAILS (apply throughout):
- Do NOT infer faults from filename, path, or labels.
- Ask for missing operating speed (rotation_speed_rpm) instead of guessing. If unknown, mark results as provisional.
- A gear tooth fault (localized damage) requires BOTH:
  • Clear Gear Mesh Frequency (GMF) harmonics AND
  • Sidebands spaced by shaft rotation frequency (f_rot) around GMF or its harmonics
  • (Optional but reinforcing) Elevated Kurtosis (>5) or modulation energy
- Without sidebands: DO NOT claim tooth damage; consider uniform wear only if GMF elevated but stable statistics.

STEP 1 — INPUT & CONTEXT
If rotation_speed_rpm provided by user: f_rot = rotation_speed_rpm / 60
Else: request it politely before final classification (you may proceed with screening but mark conclusions "inconclusive").
Compute theoretical GMF = f_rot × {num_teeth} (once f_rot known).

STEP 2 — STATISTICS (screening only)
Call: analyze_statistics("{signal_file}")
Report RMS, Crest Factor, Kurtosis in bullet points (brief).
Indicators:
- Elevated RMS: possible general load / imbalance
- High Kurtosis (>5): impulsive impacts (may correlate with chipped tooth)
- High Crest Factor (>4): impulsiveness
(Do NOT diagnose from stats alone.)

STEP 3 — SPECTRUM (frequency evidence)
Call: analyze_fft("{signal_file}", {sampling_rate})
Extract dominant peaks up to, e.g., 5× expected GMF (once f_rot known). Identify:
- GMF and its harmonics: GMF, 2×GMF, 3×GMF
- Sidebands: GMF ± n·f_rot (n=1..3). Log their presence, spacing consistency, and relative amplitudes.
If operating speed unknown: still list dominant peaks; flag need for speed to confirm GMF.
Report top 5 peaks only (brief).
Optional plots: Use generate_fft_chart_html() for interactive visualization (render as artifact).

STEP 4 — OPTIONAL ENVELOPE (if strong modulation or impacts)
If stats suggest impulsiveness OR sideband pattern partial: Call analyze_envelope("{signal_file}", {sampling_rate}, 500, 5000) to inspect modulation signature. (Not mandatory if FFT already conclusive.)

STEP 5 — CLASSIFICATION (apply confirmation rule)
Decision categories (choose exactly one):
- "Gear tooth fault CONFIRMED" → Requires: (GMF harmonics present) AND (≥1 clear sideband pair with spacing ≈ f_rot) AND (supporting stat: Kurtosis>5 or CF>4)
- "Possible localized tooth damage" → Partial sidebands OR ambiguous spacing; list missing evidence required for confirmation.
- "Uniform wear / increased load" → Elevated GMF amplitude WITHOUT sidebands, normal/low impulsiveness.
- "Inconclusive" → Lacking rotation speed, or frequency pattern insufficient; request speed or higher-resolution data.

Each conclusion MUST cite: tools used (statistics, FFT, envelope), specific numeric peaks (frequencies & magnitudes), sideband spacing vs expected f_rot (difference in Hz), and any supporting statistical thresholds.

STEP 6 — RECOMMENDATIONS (brief bullet points)
Provide actionable items aligned with category:
- Confirmed fault: plan inspection, tooth visual check, lubrication review, short-term monitoring interval suggestion.
- Possible fault: acquire operating speed, higher-resolution spectrum, trend GMF amplitude.
- Uniform wear: continue monitoring; schedule routine inspection.
- Inconclusive: list exact missing parameters/data.

OUTPUT FORMATTING: Keep output ≤300 words. Use bullet points. Offer "Show more?" if user needs detailed explanation.
"""


@mcp.prompt()
def quick_diagnostic_report(signal_file: str) -> str:
    """
    Quick, evidence-aware screening report (non-definitive).

    Args:
        signal_file: Name of the signal file
    """
    return f"""Generate a concise screening report for "{signal_file}" using only observable evidence:

STEP 0 — FILENAME RESOLUTION
Call list_available_signals() to verify exact filename.
If "{signal_file}" not found or multiple matches, ASK USER to clarify.

Guardrails:
- Ignore filenames/paths as diagnostic evidence.
- Do NOT diagnose faults from statistics alone; use them for screening only.
- Use cautious language: "possible/consistent with" unless corroborated by multiple indicators.

1) Load & sanity checks
- Report number of samples, duration (s), min/max values (brief, 1 line).

2) Statistics (screening)
Call: analyze_statistics("{signal_file}")
Report: RMS, Crest Factor, Kurtosis, Skewness (bullet points only).
Flags (screening thresholds, not definitive):
- CF > 4 → impulsiveness present; CF > 6 → strong impulsiveness
- Kurtosis > 3 → impulsive content; > 5 → significant; > 8 → severe
Note: These flags alone are insufficient for fault identification.

3) Spectral snapshot
Call: analyze_fft("{signal_file}", 10000)
- Report peak frequency, magnitude (top 3 peaks only).
- If operating speed is known, relate peaks to 1×/2× RPM; otherwise, request it for deeper interpretation.

4) Next-step guidance (evidence-first)
- If strong impulsiveness (CF>6 or Kurtosis>8), suggest: "Use diagnose_bearing prompt for targeted bearing analysis"
- If tonal/harmonic pattern dominates, suggest: "Use diagnose_gear prompt if gear suspected"
- If broadband increase, suggest: ISO 20816-3 check with evaluate_iso_20816()

Output format (≤200 words):
- Screening summary with measured values (bullet points)
- No definitive fault labels
- List recommended targeted analyses and required missing parameters
"""


@mcp.prompt()
def generate_iso_diagnostic_report(
    signal_file: str,
    sampling_rate: float = 10000.0,
    machine_group: int = 1,
    support_type: str = "rigid",
    operating_speed_rpm: Optional[float] = None,
    machine_name: str = "Machine",
    measurement_location: str = "Bearing"
) -> str:
    """
    Generate comprehensive diagnostic report with ISO 20816-3 compliance evaluation.
    
    Creates a structured diagnostic report including:
    - ISO 20816-3 vibration severity assessment
    - Statistical indicators
    - Spectral analysis
    - Fault detection (bearing/gear)
    - Maintenance recommendations
    
    Args:
        signal_file: Name of the signal file to analyze
        sampling_rate: Sampling frequency in Hz
        machine_group: ISO 20816 group (1=large >300kW, 2=medium 15-300kW)
        support_type: 'rigid' or 'flexible'
        operating_speed_rpm: Operating speed in RPM
        machine_name: Machine identifier
        measurement_location: Measurement point description
    """
    rpm_param = f", operating_speed_rpm={operating_speed_rpm}" if operating_speed_rpm else ""
    
    return f"""Generate a comprehensive diagnostic report for {machine_name} - {measurement_location}

SIGNAL: {signal_file}
SAMPLING RATE: {sampling_rate} Hz
MACHINE GROUP: {machine_group} ({'Large >300kW' if machine_group == 1 else 'Medium 15-300kW'})
SUPPORT TYPE: {support_type.title()}
OPERATING SPEED: {operating_speed_rpm if operating_speed_rpm else 'Not specified'} RPM

================================================================================
SECTION 1: ISO 20816-3 VIBRATION SEVERITY ASSESSMENT
================================================================================

Execute: evaluate_iso_20816("{signal_file}", {sampling_rate}, {machine_group}, "{support_type}"{rpm_param})

Present results in this format:

┌─────────────────────────────────────────────────────────────────────┐
│ ISO 20816-3 EVALUATION RESULT                                       │
├─────────────────────────────────────────────────────────────────────┤
│ RMS Velocity (broadband):     [VALUE] mm/s                          │
│ Frequency Range:               [RANGE] Hz                           │
│ Evaluation Zone:               Zone [A/B/C/D]                       │
│ Severity Level:                [Good/Acceptable/Unsatisfactory/     │
│                                 Unacceptable]                        │
│ Color Code:                    🟢/🟡/🟠/🔴                          │
├─────────────────────────────────────────────────────────────────────┤
│ ZONE BOUNDARIES (mm/s):                                             │
│   Zone A/B: [VALUE]  |  Zone B/C: [VALUE]  |  Zone C/D: [VALUE]    │
├─────────────────────────────────────────────────────────────────────┤
│ INTERPRETATION:                                                     │
│ [Zone description from result]                                      │
└─────────────────────────────────────────────────────────────────────┘

ISO COMPLIANCE STATUS:
• If Zone A (Green): ✅ COMPLIANT - Machine in excellent condition
• If Zone B (Yellow): ⚠️  ACCEPTABLE - Continue normal operation, monitor
• If Zone C (Orange): ⚠️  NON-COMPLIANT - Plan maintenance within 1-3 months
• If Zone D (Red): 🚨 CRITICAL - Immediate action required, risk of damage

================================================================================
SECTION 2: STATISTICAL INDICATORS
================================================================================

Execute: analyze_statistics("{signal_file}")

Report the following parameters:

┌─────────────────────────────────────────────────────────────────────┐
│ STATISTICAL ANALYSIS                                                │
├─────────────────────────────────────────────────────────────────────┤
│ RMS:                  [VALUE] (Energy level)                        │
│ Peak:                 [VALUE] (Maximum amplitude)                   │
│ Peak-to-Peak:         [VALUE] (Total excursion)                     │
│ Crest Factor:         [VALUE] (Peak/RMS ratio)                      │
│ Kurtosis:             [VALUE] (Impulsiveness indicator)             │
│ Skewness:             [VALUE] (Asymmetry indicator)                 │
└─────────────────────────────────────────────────────────────────────┘

DIAGNOSTIC INDICATORS:
• Crest Factor > 4: ⚠️  Possible presence of impulses (bearing faults)
• Crest Factor > 6: 🚨 High probability of bearing defects
• Kurtosis > 3: ⚠️  Presence of impulses
• Kurtosis > 5: ⚠️  Significant impulsive content (bearing damage)
• Kurtosis > 8: 🚨 Severe bearing damage

================================================================================
SECTION 3: SPECTRAL ANALYSIS
================================================================================

Execute: analyze_fft("{signal_file}", {sampling_rate}, max_frequency=1000)

Identify:
• Peak frequency and magnitude
• Frequency resolution
• Energy distribution across spectrum

Execute: plot_spectrum("{signal_file}", {sampling_rate}, freq_range=[0, 1000], num_peaks=15)

Look for:
• Dominant frequencies (possible fault indicators)
• Harmonics pattern (multiples of rotation frequency)
• Sidebands (modulation indicators)
• Broadband noise level

================================================================================
SECTION 4: BEARING FAULT DETECTION
================================================================================

Execute: analyze_envelope("{signal_file}", {sampling_rate}, filter_low=500, filter_high=5000, num_peaks=10)

Execute: plot_envelope("{signal_file}", {sampling_rate}, filter_band=[500, 5000], freq_range=[0, 100])

Analyze envelope spectrum peaks and compare with:
• BPFO (Ball Pass Frequency - Outer race): Outer race defect
• BPFI (Ball Pass Frequency - Inner race): Inner race defect
• BSF (Ball Spin Frequency): Rolling element defect
• FTF (Fundamental Train Frequency): Cage defect

Note: Envelope peaks at harmonics of these frequencies indicate bearing damage

================================================================================
SECTION 5: OVERALL ASSESSMENT AND RECOMMENDATIONS
================================================================================

Based on all analyses, provide:

MACHINE CONDITION SUMMARY:
├─ ISO 20816-3 Status: [Compliant/Non-compliant]
├─ Vibration Severity: [Zone A/B/C/D - Color code]
├─ Fault Indicators: [Present/Absent]
└─ Urgency Level: [Normal/Monitor/Plan Maintenance/Immediate Action]

IDENTIFIED ISSUES (if any):
• [List any detected faults based on statistical/spectral/envelope analysis]

RECOMMENDATIONS:
1. IMMEDIATE ACTIONS (if Zone D or critical indicators):
   - [Specific actions needed]

2. SHORT-TERM (1-3 months, if Zone C):
   - [Maintenance planning recommendations]

3. MONITORING (if Zone B):
   - [Suggested monitoring frequency and parameters]

4. ROUTINE OPERATION (if Zone A):
   - [Continue normal operation, periodic checks]

ADDITIONAL DIAGNOSTICS (if needed):
• Consider trending analysis for Zone B/C
• Perform time-domain analysis if high Crest Factor
• Check alignment if high 1× RPM component
• Inspect lubrication if broadband noise increase

================================================================================
REPORT GENERATED: [Current date/time]
ANALYZED BY: ISO 20816-3 Diagnostic System
================================================================================
"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting Predictive Maintenance MCP Server...")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Run server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
