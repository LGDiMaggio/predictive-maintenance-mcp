"""
The signal import engine: the steps above the strict decoder.

ISO 13374 Block 1 (data acquisition). ``signal_acquisition`` reads a
delimited-text export exactly as an interpretation says
(``import_config``, ``text_scan``, ``text_decoder``); this package judges
what was read before anything is imported. It starts with the time-base
assessment (:mod:`.timebase`); inspection, configuration and validation
follow in later units of the import plan.
"""

from .timebase import (  # noqa: F401
    EVIDENCE_LEVELS,
    FINDING_CODES,
    MAX_LISTED_LINES,
    OUTCOMES,
    RATE_SOURCES,
    REGIMES,
    DeclaredTimeBase,
    RateComparison,
    RateSource,
    TimeBaseAssessment,
    TimeBaseRefused,
    TimeFinding,
    assess_time_base,
    assess_time_column,
    parse_time_tokens,
)

__all__ = [
    "EVIDENCE_LEVELS",
    "FINDING_CODES",
    "MAX_LISTED_LINES",
    "OUTCOMES",
    "RATE_SOURCES",
    "REGIMES",
    "DeclaredTimeBase",
    "RateComparison",
    "RateSource",
    "TimeBaseAssessment",
    "TimeBaseRefused",
    "TimeFinding",
    "assess_time_base",
    "assess_time_column",
    "parse_time_tokens",
]
