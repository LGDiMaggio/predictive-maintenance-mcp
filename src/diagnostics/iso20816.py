"""
ISO 20816-3 vibration severity assessment — single source of truth.

Zone boundary values are those published in ISO 10816-3:2009 (velocity
criteria). The four-zone A-D scheme of that edition is kept because it is
what practitioners know; ISO 20816-3:2022, which supersedes it, merges
zones A and B into a single acceptance region. Every result carries this
provenance note (see ``THRESHOLD_PROVENANCE``).

Scope: industrial machines with rated power above 15 kW measured on
non-rotating parts. When the machine power is declared and falls below
15 kW the assessment is refused; when it is unknown the scope limit is
documented but not enforceable.

This module is the ONLY place in the codebase where zone boundaries live.
Consumers (import, never redefine):
- ``assess_severity`` / ``diagnose_vibration`` MCP tools
- ``decision_support.alerts.check_alert_thresholds``

Pure functions — no MCP context, no file I/O.
"""

from typing import Literal, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

#: Provenance note attached to every assessment result.
THRESHOLD_PROVENANCE = (
    "Zone boundaries from ISO 10816-3:2009 (four-zone A-D scheme). "
    "ISO 20816-3:2022 supersedes that edition and merges zones A and B; "
    "the A/B boundary is kept here for practitioner familiarity."
)

# Zone boundaries per (machine_group, support_type) in mm/s RMS velocity.
# ISO 10816-3:2009 velocity criteria — each tuple is the upper limit of
# zones (A, B, C); zone D is anything above. THE single threshold table of
# the codebase: import via get_zone_boundaries()/classify_zone(), never
# redefine these values elsewhere.
_ZONE_BOUNDARIES: dict[tuple[int, str], tuple[float, float, float]] = {
    (1, "rigid"): (2.3, 4.5, 7.1),  # Group 1 (large, >300 kW) rigid
    (1, "flexible"): (3.5, 7.1, 11.0),  # Group 1 flexible
    (2, "rigid"): (1.4, 2.8, 4.5),  # Group 2 (medium, 15-300 kW) rigid
    (2, "flexible"): (2.3, 4.5, 7.1),  # Group 2 flexible
}

_ZONE_INFO = {
    "A": ("New machine condition. Vibration is excellent.", "Good", "green"),
    "B": (
        "Acceptable for unrestricted long-term operation.",
        "Acceptable",
        "yellow",
    ),
    "C": (
        "Unsatisfactory for long-term operation. Plan maintenance soon.",
        "Unsatisfactory",
        "orange",
    ),
    "D": (
        "Vibration severity may cause damage. Immediate action required!",
        "Unacceptable",
        "red",
    ),
}

# ISO 20816-3 evaluation band upper edge (Hz).
_ISO_BAND_UPPER_HZ = 1000.0
# ISO 20816-3 scope floor for rated machine power (kW).
_ISO_POWER_FLOOR_KW = 15.0


def get_zone_boundaries(
    machine_group: Literal[1, 2],
    support_type: Literal["rigid", "flexible"],
) -> tuple[float, float, float]:
    """Return the (A/B, B/C, C/D) zone boundaries in mm/s RMS velocity.

    Args:
        machine_group: 1 (large machines, >300 kW) or 2 (medium, 15-300 kW).
        support_type: 'rigid' or 'flexible'.

    Returns:
        Tuple of (A/B, B/C, C/D) boundaries per ISO 10816-3:2009.

    Raises:
        ValueError: If the (machine_group, support_type) combination is not
            in the ISO table.
    """
    key = (machine_group, str(support_type).lower())
    if key not in _ZONE_BOUNDARIES:
        raise ValueError(
            f"Invalid machine_group={machine_group}, support_type='{support_type}' — "
            f"machine_group must be 1 (large, >300 kW) or 2 (medium, 15-300 kW) "
            f"and support_type must be 'rigid' or 'flexible'."
        )
    return _ZONE_BOUNDARIES[key]


def describe_zone(zone: Literal["A", "B", "C", "D"]) -> dict:
    """Return the standard description/severity/color for an ISO zone letter.

    Used by consumers that classify against NON-ISO boundaries (custom
    thresholds) but still report the familiar zone vocabulary.

    Raises:
        ValueError: If zone is not one of 'A', 'B', 'C', 'D'.
    """
    if zone not in _ZONE_INFO:
        raise ValueError(
            f"Unknown zone '{zone}' — valid zones are A, B, C, D."
        )
    desc, severity, color = _ZONE_INFO[zone]
    return {
        "zone_description": desc,
        "severity_level": severity,
        "color_code": color,
    }


def check_power_scope(machine_power_kw: Optional[float]) -> None:
    """Refuse a DECLARED machine power below the 15 kW ISO scope floor.

    ``None`` means "unknown" and passes: without the power figure the
    scope condition is not detectable, so the limit stays documented but
    unenforced.

    Raises:
        ValueError: If machine_power_kw is declared and below 15 kW.
    """
    if machine_power_kw is not None and machine_power_kw < _ISO_POWER_FLOOR_KW:
        raise ValueError(
            f"ISO assessment refused: declared machine power "
            f"{machine_power_kw:g} kW is below the 15 kW scope floor of "
            f"ISO 20816-3 (thresholds from ISO 10816-3:2009) — these zone "
            f"boundaries do not apply to small machines; use manufacturer "
            f"acceptance limits instead, or omit machine_power_kw only if "
            f"the power is genuinely unknown."
        )


def classify_zone(
    rms_velocity_mm_s: float,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
) -> dict:
    """Classify an RMS velocity reading into an ISO zone (A/B/C/D).

    Boundary values belong to the lower zone (<= semantics). This is the
    single zone-classification path of the codebase; alerting and severity
    assessment both delegate here.

    Args:
        rms_velocity_mm_s: Broadband RMS velocity in mm/s.
        machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW).
        support_type: 'rigid' or 'flexible'.

    Returns:
        Dict with ``zone``, ``zone_description``, ``severity_level``,
        ``color_code``, ``boundaries`` ({AB, BC, CD} in mm/s), and
        ``threshold_provenance``.

    Raises:
        ValueError: If rms_velocity_mm_s is negative or the machine
            group/support combination is invalid.
    """
    if rms_velocity_mm_s < 0:
        raise ValueError(
            f"rms_velocity_mm_s must be >= 0, got {rms_velocity_mm_s} — RMS is "
            f"non-negative by definition; check the upstream computation."
        )

    boundary_ab, boundary_bc, boundary_cd = get_zone_boundaries(
        machine_group, support_type
    )

    if rms_velocity_mm_s <= boundary_ab:
        zone = "A"
    elif rms_velocity_mm_s <= boundary_bc:
        zone = "B"
    elif rms_velocity_mm_s <= boundary_cd:
        zone = "C"
    else:
        zone = "D"

    desc, severity, color = _ZONE_INFO[zone]
    return {
        "zone": zone,
        "zone_description": desc,
        "severity_level": severity,
        "color_code": color,
        "boundaries": {
            "AB": boundary_ab,
            "BC": boundary_bc,
            "CD": boundary_cd,
        },
        "threshold_provenance": THRESHOLD_PROVENANCE,
    }


def _convert_to_velocity_mm_s(
    signal: np.ndarray,
    fs: float,
    signal_unit: str,
) -> tuple[np.ndarray, bool, str]:
    """Convert signal to velocity in mm/s if needed.

    Returns:
        (velocity_mm_s, conversion_performed, original_unit)

    Raises:
        ValueError: If ``signal_unit`` is not one of the declared vocabulary
            ('g', 'm/s²'/'m/s2', 'm/s', 'mm/s'). Units are never assumed.
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
        raise ValueError(
            f"Unknown signal_unit '{signal_unit}' — declare one of "
            f"'g'/'m/s2' (acceleration) or 'mm/s'/'m/s' (velocity). "
            f"Units are never assumed; a wrong unit invalidates the "
            f"ISO 20816-3 verdict."
        )


def assess_severity_raw(
    signal: np.ndarray,
    fs: float,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
    signal_unit: str = "g",
    operating_speed_rpm: Optional[float] = None,
    machine_power_kw: Optional[float] = None,
) -> dict:
    """Core severity assessment using native ISO machine_group/support_type.

    This is the single implementation of ISO zone classification for
    signals. All severity/diagnosis tools delegate to this function.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz). Must give Nyquist >= 1000 Hz so the
            ISO evaluation band can be covered.
        machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW).
        support_type: 'rigid' or 'flexible'.
        signal_unit: 'g', 'm/s²', 'mm/s', or 'm/s'.
        operating_speed_rpm: Operating speed for frequency range selection
            (2 Hz lower edge below 600 RPM, 10 Hz otherwise).
        machine_power_kw: Rated machine power, if known. Values below the
            15 kW ISO 20816-3 scope floor are refused; ``None`` means
            "unknown" and is not refused (the scope limit stays documented).

    Returns:
        Dict with zone, severity, RMS velocity, boundaries, the REAL
        integration band used (``frequency_range``), and the threshold
        provenance note.

    Raises:
        ValueError: If the machine group/support combination is invalid,
            if Nyquist < 1000 Hz (ISO band cannot be evaluated), or if the
            declared machine power is below 15 kW (out of ISO scope).
    """
    support_type = str(support_type).lower()
    # Validate group/support before any signal processing.
    get_zone_boundaries(machine_group, support_type)

    check_power_scope(machine_power_kw)

    nyquist = fs / 2.0
    if nyquist < _ISO_BAND_UPPER_HZ:
        raise ValueError(
            f"ISO assessment not possible: Nyquist frequency {nyquist:g} Hz "
            f"(fs={fs:g} Hz) is below the {_ISO_BAND_UPPER_HZ:g} Hz upper "
            f"edge of the ISO 20816-3 evaluation band — re-acquire the "
            f"signal at fs >= {2 * _ISO_BAND_UPPER_HZ:g} Hz."
        )

    # Unit conversion
    velocity_mm_s, unit_conversion, original_unit = _convert_to_velocity_mm_s(
        signal, fs, signal_unit
    )

    # Evaluation band. Lower edge per ISO 20816-3: 10 Hz for speeds
    # >= 600 RPM (or unknown), 2 Hz for 120-600 RPM. Upper edge is the
    # nominal 1000 Hz, reduced to 0.95 x Nyquist when fs cannot support
    # it — the reported band is ALWAYS the band actually used.
    if operating_speed_rpm is not None and operating_speed_rpm < 600:
        low_hz = 2.0
        speed_note = "speed < 600 RPM"
    else:
        low_hz = 10.0
        speed_note = "speed >= 600 RPM"
    high_hz = min(_ISO_BAND_UPPER_HZ, nyquist * 0.95)

    freq_range_desc = f"{low_hz:g}-{high_hz:g} Hz ({speed_note})"
    if high_hz < _ISO_BAND_UPPER_HZ:
        freq_range_desc += (
            f" — upper edge limited by fs={fs:g} Hz; the ISO nominal band "
            f"extends to {_ISO_BAND_UPPER_HZ:g} Hz"
        )

    # Bandpass filter (zero-phase)
    sos = butter(
        4, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos"
    )
    velocity_filtered = sosfiltfilt(sos, velocity_mm_s)

    # RMS velocity
    rms_velocity = float(np.sqrt(np.mean(velocity_filtered**2)))

    # Zone classification via the single classification path
    zone_result = classify_zone(rms_velocity, machine_group, support_type)

    return {
        "rms_velocity_mm_s": round(rms_velocity, 4),
        "machine_group": machine_group,
        "support_type": support_type,
        "zone": zone_result["zone"],
        "zone_description": zone_result["zone_description"],
        "severity_level": zone_result["severity_level"],
        "color_code": zone_result["color_code"],
        "boundaries": zone_result["boundaries"],
        "frequency_range": freq_range_desc,
        "unit_conversion_performed": unit_conversion,
        "original_unit": original_unit,
        "operating_speed_rpm": operating_speed_rpm,
        "machine_power_kw": machine_power_kw,
        "threshold_provenance": THRESHOLD_PROVENANCE,
    }


def assess_severity_with_axis(
    signal: np.ndarray,
    fs: float,
    machine_group: Literal[1, 2] = 2,
    support_type: Literal["rigid", "flexible"] = "rigid",
    axis: str = "vertical",
    signal_unit: str = "g",
    operating_speed_rpm: Optional[float] = None,
    machine_power_kw: Optional[float] = None,
) -> dict:
    """Assess vibration severity using native ISO vocabulary.

    Thin wrapper over :func:`assess_severity_raw` that records the
    measurement axis. Renamed from ``assess_vibration_severity`` in U9b —
    that name belonged to a removed MCP tool (merged into the unified
    ``assess_severity``), and an engine function must not shadow a dead
    endpoint name. The former ``machine_class`` (I-IV) parameter was
    removed: that mapping is not part of ISO 10816-3:2009 or ISO 20816-3:2022, whose
    vocabulary is machine group + support type.

    Args:
        signal: 1D vibration signal.
        fs: Sampling frequency (Hz).
        machine_group: 1 (large, >300 kW) or 2 (medium, 15-300 kW).
        support_type: 'rigid' or 'flexible'.
        axis: Measurement axis (informational).
        signal_unit: Unit of input signal ('g', 'm/s²', 'mm/s', 'm/s').
        operating_speed_rpm: Operating speed for frequency range selection.
        machine_power_kw: Rated machine power, if known (<15 kW is refused
            as out of ISO scope).

    Returns:
        Dict with zone, severity, RMS velocity, boundaries, real frequency
        band, and threshold provenance.
    """
    result = assess_severity_raw(
        signal=signal,
        fs=fs,
        machine_group=machine_group,
        support_type=support_type,
        signal_unit=signal_unit,
        operating_speed_rpm=operating_speed_rpm,
        machine_power_kw=machine_power_kw,
    )
    result["axis"] = axis
    return result
