"""
The signal import engine: the steps above the strict decoder.

ISO 13374 Block 1 (data acquisition). ``signal_acquisition`` reads a
delimited-text export exactly as an interpretation says
(``import_config``, ``text_scan``, ``text_decoder``); this package judges
what was read before anything is imported: the time-base assessment
(:mod:`.timebase`), the read-only inspection (:mod:`.inspection`, results in
:mod:`.contracts`) and the append-only import record store
(:mod:`.records`).
"""

from .contracts import InspectionResult  # noqa: F401
from .inspection import inspect_signal_file  # noqa: F401
from .records import (  # noqa: F401
    ImportRecordDamaged,
    ImportRecordError,
    ImportRecordMissing,
    ImportRecordStore,
)
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
    "ImportRecordDamaged",
    "ImportRecordError",
    "ImportRecordMissing",
    "ImportRecordStore",
    "InspectionResult",
    "inspect_signal_file",
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
