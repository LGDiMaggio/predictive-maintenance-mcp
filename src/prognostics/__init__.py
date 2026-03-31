"""
ISO 13374 Block 5 — Prognostic Assessment.

Trend analysis and Remaining Useful Life estimation for
degradation-based maintenance planning.
"""

from .trend_analyzer import analyze_trend, detect_degradation_onset  # noqa: F401
from .rul_estimator import estimate_rul_linear, estimate_rul_exponential, estimate_rul_weibull  # noqa: F401
from .kalman_rul import estimate_rul_kalman  # noqa: F401
