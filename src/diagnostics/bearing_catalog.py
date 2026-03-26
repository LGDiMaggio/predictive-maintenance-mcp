"""
Bearing catalog wrapper for lookup and fault frequency computation.

Thin layer over document_reader.py functions, combining bearing lookup
with characteristic frequency calculation in a single call.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ..config import RESOURCES_DIR
from ..document_reader import (
    calculate_bearing_frequencies,
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

    Returns:
        List of dicts with designation, type, bore_mm, outer_diameter_mm.
    """
    catalog_path = RESOURCES_DIR / "bearing_catalogs" / "common_bearings_catalog.json"
    if not catalog_path.exists():
        return []

    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        bearings = catalog.get("bearings", {})
        return [
            {
                "designation": b.get("designation", key),
                "type": b.get("type", "Unknown"),
                "bore_mm": b.get("bore_mm"),
                "outer_diameter_mm": b.get("outer_diameter_mm"),
                "num_balls": b.get("num_balls"),
            }
            for key, b in bearings.items()
        ]
    except Exception as e:
        logger.error(f"Error reading bearing catalog: {e}")
        return []
