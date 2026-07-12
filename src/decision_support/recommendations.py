"""
ISO 13374 Block 6 — Decision Support: Maintenance Recommendations.

Rule-based recommendation engine for vibration-based maintenance decisions.
"""

from __future__ import annotations

from ..diagnostics.bearing_analyzer import FAULT_TYPE_CANONICAL

# Mapping from fault type keyword to specific maintenance advice. Bearing
# keys are the canonical fault vocabulary (bearing_analyzer
# FAULT_TYPE_CANONICAL values); the rest are machine-level faults.
_FAULT_RECOMMENDATIONS: dict[str, str] = {
    "outer_race": "Replace bearing, check alignment",
    "inner_race": "Replace bearing, inspect shaft condition",
    "ball": "Replace bearing, check lubrication system",
    "cage": "Replace bearing, investigate contamination",
    "misalignment": "Realign coupling, check foundation bolts",
    "unbalance": "Balance rotor, check for deposit buildup",
    "looseness": "Tighten foundation bolts, inspect mounting",
}

# The bearing part of the vocabulary MUST mirror the canonical fault types
# produced by check_bearing_faults / diagnose_vibration.
assert set(FAULT_TYPE_CANONICAL.values()) <= set(_FAULT_RECOMMENDATIONS), (
    "recommendation vocabulary out of sync with FAULT_TYPE_CANONICAL"
)

#: Closed fault-type vocabulary accepted by generate_recommendations.
VALID_FAULT_TYPES: tuple[str, ...] = tuple(sorted(_FAULT_RECOMMENDATIONS))


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
        fault_types: Optional list of detected fault keywords from the
            closed vocabulary ``VALID_FAULT_TYPES`` (e.g.
            ``["outer_race", "misalignment"]``).

    Returns:
        List of recommendation dicts, each containing ``action``,
        ``urgency``, and ``description``.

    Raises:
        ValueError: If any fault type is outside ``VALID_FAULT_TYPES`` —
            unknown values used to be dropped silently, which hid typos
            (e.g. 'BPFO' instead of 'outer_race').
    """
    if fault_types:
        unknown = [
            f for f in fault_types if f.lower() not in _FAULT_RECOMMENDATIONS
        ]
        if unknown:
            raise ValueError(
                f"Unknown fault type(s) {unknown} — allowed values: "
                f"{list(VALID_FAULT_TYPES)}. Bearing acronyms map to the "
                f"canonical vocabulary as BPFO=outer_race, BPFI=inner_race, "
                f"BSF=ball, FTF=cage."
            )

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
