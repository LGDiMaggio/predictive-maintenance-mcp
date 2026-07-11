"""
ISO 13374 Block 6 — Decision Support: Alert Thresholds.

Threshold-based alerting for vibration monitoring per ISO 20816-3.
Zone boundaries come from the single severity engine
(``diagnostics.iso10816``, values from ISO 10816-3:2009); this module
holds NO threshold table of its own.
"""

from __future__ import annotations

from ..diagnostics.iso10816 import classify_zone

# Zone letter -> alert level for the ISO path.
_ZONE_TO_ALERT = {"A": "none", "B": "warning", "C": "alarm", "D": "danger"}


def check_alert_thresholds(
    rms_velocity: float,
    machine_group: int = 2,
    support_type: str = "rigid",
) -> dict:
    """Classify an RMS velocity reading against ISO 20816-3 zone thresholds.

    Delegates zone classification to the single severity engine
    (boundary values from ISO 10816-3:2009) so that the same reading maps
    to the same zone on every code path.

    Args:
        rms_velocity: Velocity RMS in mm/s.
        machine_group: ISO 20816-3 machine group — 1 (large, >300 kW) or
            2 (medium, 15-300 kW).
        support_type: ``"rigid"`` or ``"flexible"``.

    Returns:
        Dict with ``alert_level``, ``zone``, ``exceeded_threshold``,
        and ``message``.
    """
    try:
        zone_info = classify_zone(rms_velocity, machine_group, support_type)
    except ValueError as exc:
        return {
            "alert_level": "warning",
            "zone": "unknown",
            "exceeded_threshold": None,
            "message": f"{exc} Manual review required.",
        }

    zone = zone_info["zone"]
    bounds = zone_info["boundaries"]
    exceeded = {
        "A": None,
        "B": bounds["AB"],
        "C": bounds["BC"],
        "D": bounds["CD"],
    }[zone]
    messages = {
        "A": "Vibration within normal limits (Zone A).",
        "B": f"Vibration exceeds {bounds['AB']} mm/s — elevated level (Zone B).",
        "C": (
            f"Vibration exceeds {bounds['BC']} mm/s — "
            "unsatisfactory level (Zone C)."
        ),
        "D": (
            f"Vibration exceeds {bounds['CD']} mm/s — "
            "unacceptable level (Zone D). Immediate action required."
        ),
    }

    return {
        "alert_level": _ZONE_TO_ALERT[zone],
        "zone": zone,
        "exceeded_threshold": exceeded,
        "message": messages[zone],
    }


def define_custom_thresholds(
    warning: float,
    alarm: float,
    danger: float,
) -> dict:
    """Create a custom threshold dictionary.

    Args:
        warning: Upper boundary of the *normal* zone (Zone A→B transition).
        alarm: Upper boundary of the *warning* zone (Zone B→C transition).
        danger: Upper boundary of the *alarm* zone (Zone C→D transition).

    Returns:
        Threshold dict suitable for :func:`check_custom_alert`.
    """
    return {"warning": warning, "alarm": alarm, "danger": danger}


def check_custom_alert(rms_velocity: float, thresholds: dict) -> dict:
    """Classify an RMS velocity reading against custom (non-ISO) thresholds.

    Args:
        rms_velocity: Velocity RMS in mm/s.
        thresholds: Dict returned by :func:`define_custom_thresholds`.

    Returns:
        Same structure as :func:`check_alert_thresholds`.
    """
    return _classify_custom(
        rms_velocity,
        thresholds["warning"],
        thresholds["alarm"],
        thresholds["danger"],
    )


# ---- internal helper -------------------------------------------------------

def _classify_custom(
    rms_velocity: float,
    a_upper: float,
    b_upper: float,
    c_upper: float,
) -> dict:
    """Return alert dict for a value given three user-defined boundaries."""
    if rms_velocity <= a_upper:
        return {
            "alert_level": "none",
            "zone": "A",
            "exceeded_threshold": None,
            "message": "Vibration within normal limits (Zone A).",
        }
    if rms_velocity <= b_upper:
        return {
            "alert_level": "warning",
            "zone": "B",
            "exceeded_threshold": a_upper,
            "message": (
                f"Vibration exceeds {a_upper} mm/s — "
                "elevated level (Zone B)."
            ),
        }
    if rms_velocity <= c_upper:
        return {
            "alert_level": "alarm",
            "zone": "C",
            "exceeded_threshold": b_upper,
            "message": (
                f"Vibration exceeds {b_upper} mm/s — "
                "unsatisfactory level (Zone C)."
            ),
        }
    return {
        "alert_level": "danger",
        "zone": "D",
        "exceeded_threshold": c_upper,
        "message": (
            f"Vibration exceeds {c_upper} mm/s — "
            "unacceptable level (Zone D). Immediate action required."
        ),
    }
