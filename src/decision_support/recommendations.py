"""
ISO 13374 Block 6 — Decision Support: Maintenance Recommendations.

Rule-based recommendation engine for vibration-based maintenance decisions.
"""

from __future__ import annotations

# Mapping from fault type keyword to specific maintenance advice.
_FAULT_RECOMMENDATIONS: dict[str, str] = {
    "outer_race": "Replace bearing, check alignment",
    "inner_race": "Replace bearing, inspect shaft condition",
    "ball": "Replace bearing, check lubrication system",
    "cage": "Replace bearing, investigate contamination",
    "misalignment": "Realign coupling, check foundation bolts",
    "unbalance": "Balance rotor, check for deposit buildup",
    "looseness": "Tighten foundation bolts, inspect mounting",
}


def generate_recommendations(
    severity_zone: str,
    fault_types: list[str] | None = None,
) -> list[dict]:
    """Generate maintenance recommendations based on severity and faults.

    Deliberately takes no "confidence" input: a caller-supplied
    confidence number would be repeated verbatim in advisory output
    without any evidential basis.

    Args:
        severity_zone: ISO 20816-3 zone letter — ``"A"``, ``"B"``,
            ``"C"``, or ``"D"``.
        fault_types: Optional list of detected fault keywords (e.g.
            ``["outer_race", "misalignment"]``).

    Returns:
        List of recommendation dicts, each containing ``action``,
        ``urgency``, and ``description``.
    """
    zone = severity_zone.upper()

    zone_map: dict[str, tuple[str, str, str]] = {
        "A": (
            "Continue normal monitoring",
            "low",
            "Vibration levels are within acceptable limits. "
            "Maintain regular monitoring schedule.",
        ),
        "B": (
            "Schedule inspection",
            "medium",
            "Vibration levels are elevated. "
            "Schedule a visual and operational inspection.",
        ),
        "C": (
            "Plan maintenance within 2 weeks",
            "high",
            "Vibration levels are unsatisfactory. "
            "Plan corrective maintenance within two weeks.",
        ),
        "D": (
            "Immediate shutdown recommended",
            "critical",
            "Vibration levels are unacceptable and may cause damage. "
            "Immediate shutdown and inspection recommended.",
        ),
    }

    if zone not in zone_map:
        action, urgency, description = (
            "Review vibration data manually",
            "medium",
            f"Unknown severity zone '{severity_zone}'. "
            "Manual review of vibration data is recommended.",
        )
    else:
        action, urgency, description = zone_map[zone]

    recommendations: list[dict] = [
        {"action": action, "urgency": urgency, "description": description}
    ]

    # Append fault-specific recommendations when available.
    if fault_types:
        for fault in fault_types:
            key = fault.lower()
            if key in _FAULT_RECOMMENDATIONS:
                recommendations.append(
                    {
                        "action": _FAULT_RECOMMENDATIONS[key],
                        "urgency": urgency,
                        "description": f"Detected fault: {fault}.",
                    }
                )

    return recommendations
