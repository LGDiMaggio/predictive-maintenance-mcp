"""
ISO 13374 Block 6 — Decision Support (Advisory Generation).

Integrated diagnosis pipeline combining spectral, bearing, and severity analyses,
plus rule-based recommendations and threshold alerting.
"""

from .diagnosis_pipeline import (  # noqa: F401
    diagnose_vibration,
)
from .advisory import (  # noqa: F401
    build_advisory,
    collect_statements,
)
from .recommendations import generate_recommendations  # noqa: F401
from .alerts import (  # noqa: F401
    check_alert_thresholds,
    define_custom_thresholds,
    check_custom_alert,
)
