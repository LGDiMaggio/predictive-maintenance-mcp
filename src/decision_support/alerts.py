"""
ISO 13374 Block 6 — Decision Support: Alert Thresholds.

Threshold-based alerting for vibration monitoring per ISO 10816/20816.
"""

from __future__ import annotations

# ISO 10816 zone boundaries (mm/s RMS) indexed by (group, support_type).
# Each tuple is (A_upper, B_upper, C_upper).  Zone D is anything above C.
_ISO_THRESHOLDS: dict[tuple[int, str], tuple[float, float, float]] = {
    (1, "rigid"): (2.8, 7.1, 18.0),
    (2, "rigid"): (2.3, 4.5, 7.1),
    (2, "flexible"): (3.5, 7.1, 11.0),
}


def check_alert_thresholds(
    rms_velocity: float,
    machine_group: int = 2,
    support_type: str = "rigid",
) -> dict:
    """Classify an RMS velocity reading against ISO 10816 thresholds.

    Args:
        rms_velocity: Velocity RMS in mm/s.
        machine_group: ISO 10816 machine group (1 or 2).
        support_type: ``"rigid"`` or ``"flexible"``.

    Returns:
        Dict with ``alert_level``, ``zone``, ``exceeded_threshold``,
        and ``message``.
    """
    key = (machine_group, support_type.lower())
    thresholds = _ISO_THRESHOLDS.get(key)

    if thresholds is None:
        return {
            "alert_level": "warning",
            "zone": "unknown",
            "exceeded_threshold": None,
            "message": (
                f"No ISO 10816 thresholds for group {machine_group}, "
                f"support '{support_type}'. Manual review required."
            ),
        }

    a_upper, b_upper, c_upper = thresholds
    return _classify(rms_velocity, a_upper, b_upper, c_upper)


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
    """Classify an RMS velocity reading against custom thresholds.

    Args:
        rms_velocity: Velocity RMS in mm/s.
        thresholds: Dict returned by :func:`define_custom_thresholds`.

    Returns:
        Same structure as :func:`check_alert_thresholds`.
    """
    return _classify(
        rms_velocity,
        thresholds["warning"],
        thresholds["alarm"],
        thresholds["danger"],
    )


# ---- internal helper -------------------------------------------------------

def _classify(
    rms_velocity: float,
    a_upper: float,
    b_upper: float,
    c_upper: float,
) -> dict:
    """Return alert dict for a value given three zone boundaries."""
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
