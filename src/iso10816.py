"""
ISO 10816 / ISO 20816-3 vibration severity assessment.

Single source of truth for zone classification, unit conversion, and
bandpass filtering. Used by:
- assess_vibration_severity (new signal_id MCP tool)
- evaluate_iso_20816 (legacy filename-based MCP tool)
- diagnose_vibration (integrated diagnosis pipeline)

Pure functions — no MCP context, no file I/O.
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# Machine class → (machine_group, support_type) mapping
# ISO 20816-3 Table 1
_CLASS_MAP = {
    "I": (2, "flexible"),    # Small machines <15 kW
    "II": (2, "rigid"),      # Medium machines 15-300 kW
    "III": (1, "rigid"),     # Large rigid-foundation >300 kW
    "IV": (1, "flexible"),   # Large flexible-foundation (turbines)
}

# Zone boundaries per (machine_group, support_type) in mm/s RMS
# ISO 20816-3:2022 Table C.1
_ZONE_BOUNDARIES = {
    (1, "rigid"):    (2.3, 4.5, 7.1),    # Group 1 rigid
    (1, "flexible"): (3.5, 7.1, 11.0),   # Group 1 flexible
    (2, "rigid"):    (1.4, 2.8, 4.5),    # Group 2 rigid
    (2, "flexible"): (2.3, 4.5, 7.1),    # Group 2 flexible
}

_ZONE_INFO = {
    "A": ("New machine condition. Vibration is excellent.", "Good", "green"),
    "B": ("Acceptable for unrestricted long-term operation.", "Acceptable", "yellow"),
    "C": ("Unsatisfactory for long-term operation. Plan maintenance soon.", "Unsatisfactory", "orange"),
    "D": ("Vibration severity may cause damage. Immediate action required!", "Unacceptable", "red"),
}


def _convert_to_velocity_mm_s(
    signal: np.ndarray,
    fs: float,
    signal_unit: str,
) -> tuple[np.ndarray, bool, str]:
    """Convert signal to velocity in mm/s if needed.

    Returns:
        (velocity_mm_s, conversion_performed, original_unit)
    """
    unit = signal_unit.lower()

    if unit in ("g", "m/s²", "m/s2"):
        # Remove DC offset
        signal_ac = signal - np.mean(signal)

        # Convert to m/s²
        if unit == "g":
            accel_ms2 = signal_ac * 9.80665
        else:
            accel_ms2 = signal_ac

        # Frequency-domain integration: V(f) = A(f) / (j·2π·f)
        n = len(accel_ms2)
        dt = 1.0 / fs
        accel_fft = np.fft.rfft(accel_ms2)
        freqs = np.fft.rfftfreq(n, dt)

        vel_fft = np.zeros_like(accel_fft, dtype=complex)
        vel_fft[1:] = accel_fft[1:] / (1j * 2 * np.pi * freqs[1:])

        vel_ms = np.fft.irfft(vel_fft, n=n)
        return vel_ms * 1000.0, True, unit

    elif unit == "m/s":
        return signal * 1000.0, True, unit

    elif unit == "mm/s":
        return signal.copy(), False, unit

    else:
        logger.warning(f"Unknown signal_unit '{signal_unit}', assuming mm/s")
        return signal.copy(), False, unit


def _classify_zone(
    rms_velocity: float,
    boundary_ab: float,
    boundary_bc: float,
    boundary_cd: float,
) -> tuple[str, str, str, str]:
    """Classify into ISO zone. Returns (zone, description, severity, color)."""
    if rms_velocity <= boundary_ab:
        zone = "A"
    elif rms_velocity <= boundary_bc:
        zone = "B"
    elif rms_velocity <= boundary_cd:
        zone = "C"
    else:
        zone = "D"

    desc, severity, color = _ZONE_INFO[zone]
    return zone, desc, severity, color


def assess_severity_raw(
    signal: np.ndarray,
    fs: float,
    machine_group: int = 2,
    support_type: str = "rigid",
    signal_unit: str = "g",
    operating_speed_rpm: Optional[float] = None,
) -> dict:
    """Core severity assessment using raw machine_group/support_type parameters.

    This is the single implementation of ISO 20816-3 zone classification.
    Both the legacy evaluate_iso_20816 tool and the new assess_vibration_severity
    tool delegate to this function.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        machine_group: 1 (large >300kW) or 2 (medium 15-300kW).
        support_type: 'rigid' or 'flexible'.
        signal_unit: 'g', 'm/s²', 'mm/s', or 'm/s'.
        operating_speed_rpm: Operating speed for frequency range selection.

    Returns:
        Dict with zone, severity, RMS velocity, boundaries, frequency range, etc.
    """
    support_type = support_type.lower()
    key = (machine_group, support_type)
    if key not in _ZONE_BOUNDARIES:
        raise ValueError(
            f"Invalid machine_group={machine_group}, support_type='{support_type}'. "
            f"machine_group must be 1 or 2, support_type must be 'rigid' or 'flexible'."
        )

    boundary_ab, boundary_bc, boundary_cd = _ZONE_BOUNDARIES[key]

    # Unit conversion
    velocity_mm_s, unit_conversion, original_unit = _convert_to_velocity_mm_s(
        signal, fs, signal_unit
    )

    # Frequency range per ISO 20816-3
    # 10-1000 Hz for speeds >= 600 RPM (or unknown)
    # 2-1000 Hz for speeds 120-600 RPM
    if operating_speed_rpm is not None and operating_speed_rpm < 600:
        low_hz = 2.0
        freq_range_desc = "2-1000 Hz (speed < 600 RPM)"
    else:
        low_hz = 10.0
        freq_range_desc = "10-1000 Hz (speed >= 600 RPM)"
    high_hz = 1000.0

    # Bandpass filter
    nyquist = fs / 2.0
    low_hz = max(low_hz, 1.0)
    high_hz = min(high_hz, nyquist * 0.95)

    if high_hz > low_hz:
        sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
        velocity_filtered = sosfiltfilt(sos, velocity_mm_s)
    else:
        velocity_filtered = velocity_mm_s

    # RMS velocity
    rms_velocity = float(np.sqrt(np.mean(velocity_filtered**2)))

    # Zone classification
    zone, zone_desc, severity, color = _classify_zone(
        rms_velocity, boundary_ab, boundary_bc, boundary_cd
    )

    return {
        "rms_velocity_mm_s": round(rms_velocity, 4),
        "machine_group": machine_group,
        "support_type": support_type,
        "zone": zone,
        "zone_description": zone_desc,
        "severity_level": severity,
        "color_code": color,
        "boundaries": {
            "AB": boundary_ab,
            "BC": boundary_bc,
            "CD": boundary_cd,
        },
        "frequency_range": freq_range_desc,
        "unit_conversion_performed": unit_conversion,
        "original_unit": original_unit,
        "operating_speed_rpm": operating_speed_rpm,
    }


def assess_vibration_severity(
    signal: np.ndarray,
    fs: float,
    machine_class: str = "II",
    axis: str = "vertical",
    signal_unit: str = "g",
    operating_speed_rpm: Optional[float] = None,
) -> dict:
    """Assess vibration severity using machine_class (I/II/III/IV) interface.

    Higher-level wrapper that maps machine_class to the underlying
    machine_group/support_type used by ISO 20816-3.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        machine_class: 'I' (small), 'II' (medium), 'III' (large rigid),
            'IV' (large flexible).
        axis: Measurement axis (informational).
        signal_unit: Unit of input signal ('g', 'm/s²', 'mm/s', 'm/s').
        operating_speed_rpm: Operating speed for frequency range selection.

    Returns:
        Dict with zone, severity, RMS velocity, boundaries, etc.
    """
    machine_class = machine_class.upper()
    if machine_class not in _CLASS_MAP:
        raise ValueError(
            f"Invalid machine_class '{machine_class}'. "
            f"Must be one of: {list(_CLASS_MAP.keys())}"
        )

    group, support = _CLASS_MAP[machine_class]

    result = assess_severity_raw(
        signal=signal,
        fs=fs,
        machine_group=group,
        support_type=support,
        signal_unit=signal_unit,
        operating_speed_rpm=operating_speed_rpm,
    )

    # Replace group/support with class in result
    result["machine_class"] = machine_class
    result["axis"] = axis

    return result
