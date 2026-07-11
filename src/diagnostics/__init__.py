"""
ISO 13374 Blocks 3-4 — State Detection & Health Assessment.

Bearing fault detection, catalog lookup, ISO 20816-3 severity.
"""

from .bearing_analyzer import (  # noqa: F401
    check_bearing_fault_peak,
    check_all_bearing_faults,
    lookup_bearing_and_compute,
)
from .bearing_catalog import (  # noqa: F401
    lookup_bearing,
    compute_fault_frequencies,
    list_catalog_bearings,
)
from .iso10816 import (  # noqa: F401
    THRESHOLD_PROVENANCE,
    assess_severity_raw,
    assess_vibration_severity,
    classify_zone,
    get_zone_boundaries,
)
