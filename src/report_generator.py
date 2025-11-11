"""
HTML Report Generation Tools

Professional report generation system for machinery diagnostics.
All reports are saved as standalone HTML files in the reports/ directory.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Import HTML templates
try:
    from src.html_templates import (
        create_fft_report,
        create_envelope_report,
        create_iso_report
    )
except ImportError:
    from html_templates import (
        create_fft_report,
        create_envelope_report,
        create_iso_report
    )

logger = logging.getLogger(__name__)

# Reports output directory
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fft_report(
    signal_file: str,
    sampling_rate: float,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    signal_data: np.ndarray,
    max_freq: Optional[float] = None,
    num_peaks: int = 15,
    rotation_freq: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate and save professional FFT spectrum report.
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz
        frequencies: Frequency array (positive frequencies)
        magnitudes: Magnitude array
        signal_data: Original signal data
        max_freq: Maximum frequency to display (default: Nyquist)
        num_peaks: Number of peaks to detect and label
        rotation_freq: Optional shaft rotation frequency for harmonic labeling
    
    Returns:
        Dictionary with file path, metadata, and summary
    """
    # Apply frequency limit
    if max_freq is None:
        max_freq = sampling_rate / 2.0
    
    mask = frequencies <= max_freq
    freq_display = frequencies[mask]
    mag_display = magnitudes[mask]
    
    # Convert to dB scale (normalized to max)
    max_mag = np.max(mag_display)
    mag_display_db = 20 * np.log10((mag_display + 1e-12) / max_mag)
    
    # Peak detection
    freq_resolution = frequencies[1] - frequencies[0]
    min_peak_distance = max(1, int(10 / freq_resolution))
    
    peak_indices, properties = find_peaks(
        mag_display_db,
        height=-40,  # Within 40 dB of max
        distance=min_peak_distance
    )
    
    # Sort by magnitude and take top N
    peak_mags_db = properties['peak_heights']
    top_peak_idx = np.argsort(peak_mags_db)[::-1][:num_peaks]
    peak_indices = peak_indices[top_peak_idx]
    
    # Build peaks list with harmonic detection
    peaks = []
    for idx in peak_indices:
        freq = float(freq_display[idx])
        mag_db = float(mag_display_db[idx])
        
        # Check if harmonic of rotation frequency
        note = ""
        if rotation_freq and rotation_freq > 0:
            harmonic_order = round(freq / rotation_freq)
            if abs(freq - harmonic_order * rotation_freq) < rotation_freq * 0.1:
                note = f"Harmonic {harmonic_order}× shaft"
        
        peaks.append({
            'frequency': freq,
            'magnitude_db': mag_db,
            'note': note
        })
    
    # Metadata
    metadata = {
        'signal_file': signal_file,
        'sampling_rate': sampling_rate,
        'num_samples': len(signal_data),
        'duration': len(signal_data) / sampling_rate,
        'max_frequency': max_freq,
        'num_peaks': len(peaks),
        'rotation_freq': rotation_freq,
        'report_type': 'fft_spectrum'
    }
    
    # Generate HTML
    html = create_fft_report(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        frequencies=freq_display.tolist(),
        magnitudes_db=mag_display_db.tolist(),
        peaks=peaks,
        metadata=metadata
    )
    
    # Save HTML file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = REPORTS_DIR / f"fft_spectrum_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    logger.info(f"FFT report saved: {output_file.name}")
    
    return {
        'file_path': str(output_file.absolute()),
        'file_name': output_file.name,
        'file_size_kb': output_file.stat().st_size / 1024,
        'report_type': 'fft_spectrum',
        'num_peaks_detected': len(peaks),
        'peak_frequencies': [p['frequency'] for p in peaks[:5]],  # Top 5 for summary
        'metadata': metadata,
        'message': f"✓ FFT spectrum report saved: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)"
    }


def save_envelope_report(
    signal_file: str,
    sampling_rate: float,
    filter_band: tuple,
    filtered_signal: np.ndarray,
    envelope: np.ndarray,
    env_frequencies: np.ndarray,
    env_magnitudes: np.ndarray,
    bearing_freqs: Optional[Dict[str, float]] = None,
    max_freq: float = 500.0,
    num_peaks: int = 15
) -> Dict[str, Any]:
    """
    Generate and save professional envelope analysis report.
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz
        filter_band: (low, high) Hz tuple
        filtered_signal: Bandpass filtered signal
        envelope: Envelope signal
        env_frequencies: Envelope spectrum frequencies
        env_magnitudes: Envelope spectrum magnitudes
        bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF
        max_freq: Max frequency to display in envelope spectrum
        num_peaks: Number of peaks to detect
    
    Returns:
        Dictionary with file path, metadata, and summary
    """
    # Apply frequency limit
    mask = env_frequencies <= max_freq
    env_freq_display = env_frequencies[mask]
    env_mag_display = env_magnitudes[mask]
    
    # Convert to dB scale (normalized to max)
    max_mag = np.max(env_mag_display)
    env_mag_display_db = 20 * np.log10((env_mag_display + 1e-12) / max_mag)
    
    # Peak detection
    freq_resolution = env_frequencies[1] - env_frequencies[0]
    min_peak_distance = max(1, int(5 / freq_resolution))
    
    peak_indices, properties = find_peaks(
        env_mag_display_db,
        height=-40,
        distance=min_peak_distance
    )
    
    # Sort and take top N
    peak_mags_db = properties['peak_heights']
    top_idx = np.argsort(peak_mags_db)[::-1][:num_peaks]
    peak_indices = peak_indices[top_idx]
    
    # Build peaks list with bearing frequency matching
    peaks = []
    for idx in peak_indices:
        freq = float(env_freq_display[idx])
        mag_db = float(env_mag_display_db[idx])
        
        # Check match with bearing frequencies
        match = ""
        if bearing_freqs:
            for name, bf in bearing_freqs.items():
                if bf and abs(freq - bf) < bf * 0.05:  # Within 5%
                    match = f"≈ {name}"
                    break
        
        peaks.append({
            'frequency': freq,
            'magnitude_db': mag_db,
            'match': match
        })
    
    # Time arrays for plotting (downsample for file size)
    downsample_factor = max(1, len(filtered_signal) // 1000)
    time_data = np.linspace(0, len(filtered_signal) / sampling_rate, len(filtered_signal))
    time_display = time_data[::downsample_factor].tolist()
    filtered_display = filtered_signal[::downsample_factor].tolist()
    envelope_display = envelope[::downsample_factor].tolist()
    
    # Metadata
    metadata = {
        'signal_file': signal_file,
        'sampling_rate': sampling_rate,
        'filter_band': filter_band,
        'num_samples': len(filtered_signal),
        'duration': len(filtered_signal) / sampling_rate,
        'max_frequency': max_freq,
        'num_peaks': len(peaks),
        'bearing_frequencies': bearing_freqs,
        'report_type': 'envelope_analysis'
    }
    
    # Generate HTML
    html = create_envelope_report(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        filter_band=filter_band,
        time_data=time_display,
        filtered_signal=filtered_display,
        envelope=envelope_display,
        env_freq=env_freq_display.tolist(),
        env_mag_db=env_mag_display_db.tolist(),
        peaks=peaks,
        bearing_freqs=bearing_freqs,
        metadata=metadata
    )
    
    # Save HTML file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = REPORTS_DIR / f"envelope_analysis_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    logger.info(f"Envelope report saved: {output_file.name}")
    
    # Summary of matches
    matches_found = [p['match'] for p in peaks if p['match']]
    
    return {
        'file_path': str(output_file.absolute()),
        'file_name': output_file.name,
        'file_size_kb': output_file.stat().st_size / 1024,
        'report_type': 'envelope_analysis',
        'num_peaks_detected': len(peaks),
        'peak_frequencies': [p['frequency'] for p in peaks[:5]],
        'bearing_matches': matches_found,
        'metadata': metadata,
        'message': f"✓ Envelope analysis report saved: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)"
    }


def save_iso_report(
    signal_file: str,
    iso_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate and save professional ISO 20816-3 evaluation report.
    
    Args:
        signal_file: Signal filename
        iso_result: ISO evaluation result from evaluate_iso_20816()
    
    Returns:
        Dictionary with file path, metadata, and summary
    """
    # Metadata
    metadata = {
        'signal_file': signal_file,
        'rms_velocity': iso_result['rms_velocity'],
        'zone': iso_result['zone'],
        'severity_level': iso_result['severity_level'],
        'machine_group': iso_result['machine_group'],
        'support_type': iso_result['support_type'],
        'report_type': 'iso_20816'
    }
    
    # Generate HTML
    html = create_iso_report(
        signal_file=signal_file,
        iso_result=iso_result,
        metadata=metadata
    )
    
    # Save HTML file
    safe_name = signal_file.replace("/", "_").replace("\\", "_").replace(".csv", "")
    output_file = REPORTS_DIR / f"iso_20816_{safe_name}.html"
    output_file.write_text(html, encoding='utf-8')
    
    logger.info(f"ISO report saved: {output_file.name}")
    
    return {
        'file_path': str(output_file.absolute()),
        'file_name': output_file.name,
        'file_size_kb': output_file.stat().st_size / 1024,
        'report_type': 'iso_20816',
        'zone': iso_result['zone'],
        'severity': iso_result['severity_level'],
        'rms_velocity': iso_result['rms_velocity'],
        'metadata': metadata,
        'message': f"✓ ISO 20816-3 report saved: {output_file.name} - Zone {iso_result['zone']} ({iso_result['severity_level']})"
    }


def read_report_metadata(file_name: str) -> Dict[str, Any]:
    """
    Read metadata from HTML report without loading entire file.
    
    Args:
        file_name: Report filename in reports/ directory
    
    Returns:
        Dictionary with metadata, or error dict if file not found
    """
    file_path = REPORTS_DIR / file_name
    
    if not file_path.exists():
        available = [f.name for f in REPORTS_DIR.glob("*.html")]
        return {
            'error': f"Report not found: {file_name}",
            'available_reports': available
        }
    
    try:
        # Read file and extract JSON metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find metadata JSON block
        start_marker = '<script type="application/json" id="report-metadata">'
        end_marker = '</script>'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return {
                'error': 'Metadata not found in report',
                'file_name': file_name,
                'file_size_kb': file_path.stat().st_size / 1024
            }
        
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        
        if end_idx == -1:
            return {'error': 'Malformed metadata in report'}
        
        metadata_json = content[start_idx:end_idx].strip()
        metadata = json.loads(metadata_json)
        
        return {
            'file_name': file_name,
            'file_path': str(file_path.absolute()),
            'file_size_kb': file_path.stat().st_size / 1024,
            'metadata': metadata,
            'message': f"Metadata loaded from {file_name}"
        }
    
    except Exception as e:
        logger.error(f"Error reading report metadata: {e}")
        return {
            'error': f"Failed to read metadata: {str(e)}",
            'file_name': file_name
        }


def list_reports() -> List[Dict[str, Any]]:
    """
    List all available HTML reports in reports/ directory.
    
    Returns:
        List of dicts with report information
    """
    reports = []
    
    for html_file in REPORTS_DIR.glob("*.html"):
        # Try to read metadata
        metadata_info = read_report_metadata(html_file.name)
        
        if 'error' not in metadata_info:
            meta = metadata_info.get('metadata', {})
            reports.append({
                'file_name': html_file.name,
                'file_size_kb': html_file.stat().st_size / 1024,
                'report_type': meta.get('report_type', 'unknown'),
                'signal_file': meta.get('signal_file', 'unknown'),
                'created': html_file.stat().st_mtime
            })
    
    # Sort by creation time (newest first)
    reports.sort(key=lambda x: x['created'], reverse=True)
    
    return reports
