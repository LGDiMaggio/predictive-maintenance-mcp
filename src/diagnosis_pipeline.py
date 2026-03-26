"""Backward compatibility shim — use predictive_maintenance_mcp.decision_support.diagnosis_pipeline."""
from .decision_support.diagnosis_pipeline import *  # noqa: F401,F403
from .decision_support.diagnosis_pipeline import (  # noqa: F401
    _run_anomaly_detection,
    _extract_time_domain_features,
)
