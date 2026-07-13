"""
Bearing catalog wrapper for lookup and fault frequency computation.

Thin layer over document_reader.py functions, combining bearing lookup
with characteristic frequency calculation in a single call.
"""

import logging
from typing import Optional

from ..document_reader import (
    calculate_bearing_frequencies,
    load_bearing_catalog,
    lookup_bearing_in_catalog,
)

logger = logging.getLogger(__name__)


def lookup_bearing(designation: str) -> Optional[dict]:
    """Look up bearing specifications by designation.

    Delegates to document_reader.lookup_bearing_in_catalog which searches
    the JSON catalog and in-memory fallback.

    Args:
        designation: Bearing designation (e.g. "6205", "SKF 6205-2RS").

    Returns:
        Dict with bearing geometry or None if not found.
    """
    return lookup_bearing_in_catalog(designation)


def compute_fault_frequencies(designation: str, rpm: float) -> Optional[dict]:
    """Look up bearing and compute characteristic fault frequencies.

    Combines catalog lookup with frequency calculation in one call.

    Args:
        designation: Bearing designation.
        rpm: Shaft speed in RPM.

    Returns:
        Dict with BPFO, BPFI, BSF, FTF, shaft_freq_hz, bearing_info,
        or None if bearing not found in catalog.
    """
    bearing = lookup_bearing(designation)
    if bearing is None:
        return None

    freqs = calculate_bearing_frequencies(
        num_balls=bearing["num_balls"],
        ball_diameter_mm=bearing["ball_diameter_mm"],
        pitch_diameter_mm=bearing["pitch_diameter_mm"],
        contact_angle_deg=bearing.get("contact_angle_deg", 0.0),
        shaft_speed_rpm=rpm,
    )

    return {
        **freqs,
        "bearing_info": bearing,
    }


def list_catalog_bearings() -> list[dict]:
    """List all bearings in the catalog.

    Reads the same JSON file used by lookup_bearing_in_catalog (single
    source of truth — see document_reader.load_bearing_catalog).

    Returns:
        List of dicts with designation, type, bore_mm, outer_diameter_mm,
        num_balls, and the mandatory source citation.
    """
    catalog = load_bearing_catalog()
    bearings = catalog.get("bearings", {})
    return [
        {
            "designation": b.get("designation", key),
            "type": b.get("type", "Unknown"),
            "bore_mm": b.get("bore_mm"),
            "outer_diameter_mm": b.get("outer_diameter_mm"),
            "num_balls": b.get("num_balls"),
            "source": b.get("source"),
        }
        for key, b in bearings.items()
    ]
