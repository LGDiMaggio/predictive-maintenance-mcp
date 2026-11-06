"""
Interpretation contract of a delimited-text signal import.

ISO 13374 Block 1 (data acquisition). An *interpretation* says how the bytes
of one delimited-text export (CSV or TXT) become vibration signals: the
dialect, the data line range, the time base, the time columns to assess, the
file-level declaration defaults and one entry per imported channel. It never
says who confirmed what (the attestation: bases, overrides, ``declared_by``,
the confirmation channel), and it never names a signal id. Those live in the
import record, outside every identity, so re-confirming the same
interpretation through another channel never changes a measurement.

This module owns:

- the closed interpretation vocabularies, each defined here once;
- the inspection value-status labels and the closed issue-code list (the
  text scanner and decoder raise most issues, and they cannot import the
  ``signal_import`` package);
- :func:`normalize_interpretation`, which returns the canonical form of an
  interpretation or refuses it with every problem in one message, and
  :func:`loads_interpretation`, its strict JSON entry point;
- the identities and check values: :func:`config_digest`,
  :func:`import_id`, :func:`channel_interpretation` and its digest,
  :func:`samples_digest`;
- the ledger's ``import_format`` keys and :func:`build_import_format`, the
  ``raw_format`` marker that makes an older server refuse an imported
  measurement, and :func:`is_raw_declaration`, the one predicate every
  ``raw_format`` reader goes through.

Dependencies: the standard library, numpy (for :func:`samples_digest`), the
leaf modules ``canonical_json`` and ``file_text``, and the measurement leaf.
Never the repository, the ledger or the import service: the ledger reads this
contract when it replays an imported measurement.

Canonical form (schema ``pmm-text-import/1``)::

    {
      "schema": "pmm-text-import/1",
      "dialect": {"encoding": "utf-8", "delimiter": ";", "decimal": ",",
                  "header_line": 7, "units_line": 8},
      "data_lines": {"first": 9, "last": null},
      "time_base": {"source": "stated", "sampling_rate_hz": "25600",
                    "time_column": null},
      "time_columns": [{"column": 0, "time_unit": "s"}],
      "defaults": {"asset_id": "P-101",
                   "acquired_at": "2026-08-20T11:42:00+00:00",
                   "rpm": "1782", "load": null, "operating_state": null},
      "channels": [
        {"column": 3, "role": "vibration", "quantity": "acceleration",
         "unit": "g",
         "scaling": {"source_unit": "mV",
                     "sensitivity": {"value": "100", "unit": "mV/g",
                                     "source": "calibration_certificate"},
                     "dc_policy": "keep"},
         "binding": {"measurement_point_id": "motor_de_h",
                     "direction": "horizontal", "sensor_id": null,
                     "asset_id": null}}
      ]
    }

Rules of the canonical form:

- Line numbers are physical and 1-based; column indices are physical and
  0-based. Header labels never identify a column.
- Every other number is a decimal string normalized through
  :class:`decimal.Decimal` (``"25600.0"``, ``"25600"`` and ``"2.56e4"`` all
  become ``"25600"``), so an identity never depends on how a float prints.
  A JSON float is refused with that remedy.
- Every optional key is present with its default, so an omitted key and an
  explicit null have the same digest; ``channels`` and ``time_columns`` are
  sorted by column.
- Vocabulary tokens are exact ASCII spellings with no aliases; the micro
  signs U+00B5 and U+03BC are refused in tokens (write ``us``).
- Free text (``sensor_id``, ``operating_state``) follows the measurement
  contract's single free-text rule; ids follow the ledger id grammar.
- ``acquired_at`` is written in UTC when it declares an offset, as the
  ledger records it (two spellings of one instant are one interpretation),
  and stays naive when it declares none.
- A refusal is an :class:`InterpretationError`: one message, plus every
  problem typed as missing, unknown key or invalid, so a caller can ask for
  a missing value instead of parsing the message.

Two scaling rules are narrower than a general unit converter, on purpose: a
sensitivity is stated per the channel's own unit (``mV/g`` for a channel in
``g``), so the divisor between the electrical source unit and the stored
unit is always an exact power of ten times the stated value; and the only DC
policy is ``keep`` (no offset is subtracted).
"""

import decimal
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple, Optional

import numpy as np

from ..canonical_json import content_hash
from ..file_text import TOKEN_CHARS, escape_file_text
from .measurement import (
    MAX_FREE_TEXT_CHARS,
    UNIT_FAMILIES,
    normalize_direction,
    validate_free_text,
    validate_ledger_id,
)

__all__ = [
    "INTERPRETATION_SCHEMA",
    "ENCODINGS",
    "DELIMITERS",
    "DECIMAL_MARKS",
    "TIME_UNITS",
    "TIME_BASE_SOURCES",
    "ROLES",
    "IMPORTABLE_ROLES",
    "QUANTITIES",
    "ELECTRICAL_UNITS",
    "SENSITIVITY_UNITS",
    "SENSITIVITY_SOURCES",
    "DC_POLICIES",
    "PREVIEW_UNIT_TOKENS",
    "VALUE_STATUSES",
    "ISSUE_CODES",
    "MAX_FIELDS",
    "INTERPRETATION_DEFAULTS",
    "DIALECT_DEFAULTS",
    "DATA_LINES_DEFAULTS",
    "TIME_BASE_DEFAULTS",
    "DECLARATION_DEFAULTS",
    "CHANNEL_DEFAULTS",
    "BINDING_DEFAULTS",
    "IMPORT_FORMAT_KEYS",
    "IMPORT_RAW_MARKER_SAMPLE_FORMAT",
    "normalize_decimal",
    "PROBLEM_KINDS",
    "MAX_INTERPRETATION_CHARS",
    "InterpretationProblem",
    "InterpretationError",
    "normalize_interpretation",
    "loads_interpretation",
    "config_digest",
    "import_id",
    "is_import_id",
    "channel_interpretation",
    "channel_interpretation_digest",
    "samples_digest",
    "scaling_divisor",
    "build_import_format",
    "import_raw_marker",
    "is_import_marker",
    "is_raw_declaration",
]

#: Schema id of the canonical interpretation this version writes and reads.
INTERPRETATION_SCHEMA = "pmm-text-import/1"

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

#: Text encodings. A BOM decides UTF-8 with signature; otherwise strict UTF-8,
#: and cp1252 only when confirmed. UTF-16 and UTF-32 are a named limit.
ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")

#: Field delimiters: a comma, a semicolon, one tab, or a run of spaces and
#: tabs. Tab and whitespace are spelled as words so no token is invisible.
DELIMITERS: tuple[str, ...] = (",", ";", "tab", "whitespace")

#: Decimal marks. Never the same character as the delimiter.
DECIMAL_MARKS: tuple[str, ...] = (".", ",")

#: Units of a time column (ASCII ``us`` for microseconds).
TIME_UNITS: tuple[str, ...] = ("s", "ms", "us", "ns")

#: Where the sampling rate comes from: a time column, a value the user
#: stated, or a value the file declares (its preamble).
TIME_BASE_SOURCES: tuple[str, ...] = ("time_column", "stated", "file_declared")

#: Column roles. Inspection reports every role; only ``vibration`` channels
#: of an importable quantity become signals in this version.
ROLES: tuple[str, ...] = (
    "vibration",
    "time",
    "tachometer",
    "displacement",
    "temperature",
    "current",
    "voltage",
    "index",
    "other",
    "ignore",
)

#: Roles a channel entry may carry.
IMPORTABLE_ROLES: tuple[str, ...] = ("vibration",)

#: Importable quantities, each with the units of its family.
QUANTITIES: tuple[str, ...] = tuple(UNIT_FAMILIES)

#: Electrical source units a sensitivity scales from.
ELECTRICAL_UNITS: tuple[str, ...] = ("mV", "V")

#: Sensitivity units, each mapped to ``(electrical unit, physical unit)``.
SENSITIVITY_UNITS: dict[str, tuple[str, str]] = {
    "mV/g": ("mV", "g"),
    "V/g": ("V", "g"),
    "mV/(m/s2)": ("mV", "m/s2"),
    "V/(m/s2)": ("V", "m/s2"),
    "mV/(mm/s)": ("mV", "mm/s"),
    "V/(m/s)": ("V", "m/s"),
}

#: Where a sensitivity value comes from.
SENSITIVITY_SOURCES: tuple[str, ...] = (
    "datasheet_nominal",
    "calibration_certificate",
    "teds",
    "user_estimate",
)

#: DC policies of an electrical channel: the mean is kept as recorded.
DC_POLICIES: tuple[str, ...] = ("keep",)

#: Unit tokens inspection recognizes but this version never imports
#: (displacement, imperial velocity, speed, temperature, current). They never
#: enter ``VALID_SIGNAL_UNITS`` or ``UNIT_FAMILIES``.
PREVIEW_UNIT_TOKENS: tuple[str, ...] = ("um", "mil", "in/s", "rpm", "degC", "A")

#: How an inspected value was obtained. ``detected``: stated in the file;
#: ``derived``: computed from the data; ``proposed``: structural, and the only
#: candidate that parses the scanned data; ``ambiguous``: several candidates;
#: ``absent``: nothing found. Inspection never labels a value confirmed.
VALUE_STATUSES: tuple[str, ...] = (
    "detected",
    "derived",
    "proposed",
    "ambiguous",
    "absent",
)

#: Every issue code an inspection, validation, import or replay reports.
ISSUE_CODES: tuple[str, ...] = (
    # path and file refusals
    "unsafe_path",
    "path_outside_roots",
    "file_not_found",
    "file_unreadable",
    "not_a_regular_file",
    "unsupported_suffix",
    "file_too_large",
    "binary_content",
    "unsupported_encoding",
    "compressed_input",
    "line_too_long",
    "too_many_fields",
    # structure
    "ambiguous_dialect",
    "encoding_needs_confirmation",
    "no_numeric_region",
    "not_a_waveform",
    "region_too_short",
    "field_count_mismatch",
    "duplicate_label",
    "empty_label",
    "trailing_delimiter_column",
    "repeated_header_block",
    # tokens
    "non_numeric_token",
    "non_finite_token",
    "quoted_data",
    "thousands_separator",
    "blank_line_in_data",
    "ragged_row",
    "truncated_last_line",
    # time
    "time_grammar",
    "time_gap",
    "time_reversal",
    "time_residual",
    "time_off_pattern",
    "time_resolution_insufficient",
    "rate_sources_disagree",
    "time_column_unassessed",
    # units and scaling
    "unit_token_unrecognized",
    "unit_token_preview_only",
    "double_scaling",
    "missing_sensitivity",
    "contradicts_file_declaration",
    # bases, binding and history
    "missing_basis",
    "basis_mismatch",
    "binding_invalid",
    "binding_collision",
    "prior_import_differs",
    "supersedes_baseline_member",
    "data_range_not_overlapping",
    "restores_superseded_interpretation",
    "companion_present",
    "history_unreadable",
    "history_incomplete",
    # replay
    "source_missing",
    "source_changed",
    # remote access
    "remote_import_disabled",
)

#: Most fields a data row may hold; every column index is below it.
MAX_FIELDS = 1024

# ---------------------------------------------------------------------------
# Field defaults: a field is required exactly when it has no default
# ---------------------------------------------------------------------------

#: Top level. ``schema``, ``dialect``, ``data_lines``, ``time_base`` and
#: ``channels`` are required.
INTERPRETATION_DEFAULTS: dict[str, Any] = {"time_columns": [], "defaults": {}}

#: ``dialect``: ``encoding``, ``delimiter`` and ``decimal`` are required.
DIALECT_DEFAULTS: dict[str, Any] = {"header_line": None, "units_line": None}

#: ``data_lines``: ``first`` is required; ``last`` null reads to the end of
#: the file.
DATA_LINES_DEFAULTS: dict[str, Any] = {"last": None}

#: ``time_base``: ``source`` and ``sampling_rate_hz`` are required;
#: ``time_column`` is required exactly when the source is ``time_column``.
TIME_BASE_DEFAULTS: dict[str, Any] = {"time_column": None}

#: The ``defaults`` block: file-level declaration values every bound channel
#: inherits (asset, acquisition instant, shaft speed, load, operating state).
#: Every field is optional.
DECLARATION_DEFAULTS: dict[str, Any] = {
    "asset_id": None,
    "acquired_at": None,
    "rpm": None,
    "load": None,
    "operating_state": None,
}

#: A channel entry: ``column``, ``role``, ``quantity`` and ``unit`` are
#: required.
CHANNEL_DEFAULTS: dict[str, Any] = {"scaling": None, "binding": None}

#: A channel ``binding``: ``measurement_point_id`` is required; ``asset_id``
#: overrides the file-level default for this channel.
BINDING_DEFAULTS: dict[str, Any] = {
    "direction": None,
    "sensor_id": None,
    "asset_id": None,
}

_INTERPRETATION_FIELDS = (
    "schema",
    "dialect",
    "data_lines",
    "time_base",
    "time_columns",
    "defaults",
    "channels",
)
_DIALECT_FIELDS = ("encoding", "delimiter", "decimal", "header_line", "units_line")
_DATA_LINES_FIELDS = ("first", "last")
_TIME_BASE_FIELDS = ("source", "sampling_rate_hz", "time_column")
_TIME_COLUMN_FIELDS = ("column", "time_unit")
_CHANNEL_FIELDS = ("column", "role", "quantity", "unit", "scaling", "binding")
_SCALING_FIELDS = ("source_unit", "sensitivity", "dc_policy")
_SENSITIVITY_FIELDS = ("value", "unit", "source")
_BINDING_FIELDS = ("measurement_point_id", "direction", "sensor_id", "asset_id")

#: The fields of a channel entry that shape its samples (the binding is
#: carried by the ledger declaration's own keys).
_SAMPLE_SHAPING_CHANNEL_FIELDS = ("column", "role", "quantity", "unit", "scaling")

#: Keys of a recorded ``import_format``, in order: what the ledger replays
#: and fingerprints. ``sampling_rate``, ``signal_unit`` and ``channel_index``
#: stay in the declaration's own top-level keys and are not repeated here.
IMPORT_FORMAT_KEYS: tuple[str, ...] = (
    "schema",
    "dialect",
    "data_lines",
    "time_base",
    "quantity",
    "scaling",
    "samples_digest",
    "sample_count",
)

#: ``sample_format`` of the ``raw_format`` marker every imported declaration
#: carries. It lies outside the raw decoder's vocabulary, so a server that
#: predates the import refuses the measurement before reading its file
#: instead of decoding the first column. A frozen constant: no version, no
#: ``scale_factor``, no ``int``/``uint`` prefix (the amplitude rule reads
#: that prefix), because it enters fingerprints and snapshot digests for good.
IMPORT_RAW_MARKER_SAMPLE_FORMAT = "pmm-text-import"

# ---------------------------------------------------------------------------
# Decimal strings
# ---------------------------------------------------------------------------

#: Lexical caps checked BEFORE any ``Decimal`` conversion, so an extreme
#: exponent or a very long digit string costs nothing. The character cap
#: admits every canonical form the other two caps allow (the longest,
#: ``-0.`` with 29 zeros and 20 digits, has 52 characters), so a canonical
#: interpretation always normalizes to itself.
_MAX_DECIMAL_CHARS = 64
_MAX_SIGNIFICANT_DIGITS = 20
_MAX_ADJUSTED_EXPONENT = 30

_DECIMAL_TOKEN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?"
)

#: A bounded local context: every signal is trapped and becomes a refusal.
_DECIMAL_CONTEXT = decimal.Context(
    prec=_MAX_DECIMAL_CHARS,
    rounding=decimal.ROUND_HALF_EVEN,
    Emax=99,
    Emin=-99,
    traps=[
        decimal.InvalidOperation,
        decimal.DivisionByZero,
        decimal.Overflow,
        decimal.Underflow,
        decimal.Subnormal,
        decimal.Inexact,
        decimal.Rounded,
        decimal.Clamped,
        decimal.FloatOperation,
    ],
)

_DECIMAL_HELP = (
    'pass a decimal string such as "25600" or "10.2" (numbers are kept as '
    "strings, so an identity never depends on how a float prints)"
)


def _significant_digits(token: str) -> int:
    """Digits between the first and the last non-zero digit (at least 1)."""
    mantissa = token.lstrip("+-").split("e")[0].split("E")[0].replace(".", "")
    stripped = mantissa.strip("0")
    return len(stripped) if stripped else 1


def _format_decimal(value: decimal.Decimal) -> str:
    """Plain notation, no exponent, no trailing zeros; zero is ``"0"``."""
    with decimal.localcontext(_DECIMAL_CONTEXT):
        normalized = value.normalize()
    if normalized.is_zero():
        return "0"
    return format(normalized, "f")


def normalize_decimal(value: object, *, field: str) -> str:
    """Return the canonical decimal string of *value*.

    Args:
        value: A decimal string. JSON numbers, booleans and other types are
            refused.
        field: Field name used in messages.

    Returns:
        The value in plain notation without trailing zeros: ``"25600.0"``,
        ``"25600"`` and ``"2.56e4"`` all give ``"25600"``.

    Raises:
        ValueError: For a non-string, a string over 64 characters or 20
            significant digits, a malformed string, or a magnitude beyond
            ``1e30`` either way; the message names the field. No ``decimal``
            exception escapes.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        kind = "a JSON number" if isinstance(value, (int, float)) else _kind(value)
        raise ValueError(f"{field} is {kind}: {_DECIMAL_HELP}")
    if len(value) > _MAX_DECIMAL_CHARS:
        raise ValueError(
            f"{field} is {len(value)} characters long, over the "
            f"{_MAX_DECIMAL_CHARS}-character cap of a decimal string: {_DECIMAL_HELP}"
        )
    if _DECIMAL_TOKEN.fullmatch(value) is None:
        raise ValueError(
            f"{field} {_shown(value)} is not a plain decimal number (digits, one "
            f"optional '.', an optional exponent such as 'e4'; no spaces, no "
            f"thousands separators, no NaN or Infinity): {_DECIMAL_HELP}"
        )
    if _significant_digits(value) > _MAX_SIGNIFICANT_DIGITS:
        raise ValueError(
            f"{field} has more than {_MAX_SIGNIFICANT_DIGITS} significant digits; "
            f"state it with at most {_MAX_SIGNIFICANT_DIGITS}"
        )
    try:
        with decimal.localcontext(_DECIMAL_CONTEXT) as ctx:
            parsed = ctx.create_decimal(value)
    except decimal.DecimalException:
        raise ValueError(
            f"{field} {_shown(value)} is out of range: {_DECIMAL_HELP}"
        ) from None
    if not parsed.is_zero() and abs(parsed.adjusted()) > _MAX_ADJUSTED_EXPONENT:
        raise ValueError(
            f"{field} {_shown(value)} is out of range: its magnitude must lie "
            f"between 1e-{_MAX_ADJUSTED_EXPONENT} and 1e{_MAX_ADJUSTED_EXPONENT}"
        )
    return _format_decimal(parsed)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

_REMEDY = "Correct every listed field and validate the interpretation again."

_MICRO_SIGNS = ("\u00b5", "\u03bc")


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, (list, tuple)):
        return "a list"
    if isinstance(value, float):
        return "a JSON float"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, str):
        return "a string"
    return type(value).__name__


def _shown(value: object) -> str:
    """A caller value as it may appear in a refusal: short, escaped and
    quoted, with its own quotes escaped so it cannot close the quotation."""
    if isinstance(value, str):
        text, truncated = escape_file_text(value, cap=TOKEN_CHARS)
        text = text.replace("'", "\\'")
        return f"'{text}{'...' if truncated else ''}'"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    return _kind(value)


#: Kinds of an interpretation problem: a required value that is absent (a
#: caller can ask for it), a key the contract does not know, or a value given
#: but not acceptable.
PROBLEM_KINDS: tuple[str, ...] = ("missing", "unknown_key", "invalid")


class InterpretationProblem(NamedTuple):
    """One problem of a refused interpretation.

    Attributes:
        path: Dotted path of the field (``channels[0].scaling.dc_policy``).
        kind: One of :data:`PROBLEM_KINDS`.
        text: The problem and its remedy, caller values short and escaped.
    """

    path: str
    kind: str
    text: str


class InterpretationError(ValueError):
    """A refused interpretation: one message, plus every problem typed.

    ``str()`` is the one "problem, remedy" message; :attr:`problems` lets a
    caller tell a missing value it can ask for from an invalid one without
    parsing the message.
    """

    def __init__(self, problems: list[InterpretationProblem]) -> None:
        self.problems: tuple[InterpretationProblem, ...] = tuple(problems)
        super().__init__(
            "Invalid import interpretation: "
            + "; ".join(f"{p.path} {p.text}" for p in self.problems)
            + f". {_REMEDY}"
        )


class _Problems:
    """Accumulates typed problems; refuses once at the end."""

    def __init__(self) -> None:
        self.items: list[InterpretationProblem] = []

    def add(self, path: str, text: str, kind: str = "invalid") -> None:
        self.items.append(InterpretationProblem(path, kind, text))

    def collect(self, path: str, value: Any, normalize: Callable[[Any], Any]) -> Any:
        """Apply *normalize*; turn its ValueError into a problem, escaped."""
        try:
            return normalize(value)
        except ValueError as exc:
            text, truncated = escape_file_text(str(exc), cap=600)
            self.add(path, f"is refused: {text}{'...' if truncated else ''}")
            return None

    def raise_if_any(self) -> None:
        if self.items:
            raise InterpretationError(self.items)


def _object(
    value: object,
    path: str,
    fields: tuple[str, ...],
    defaults: Mapping[str, Any],
    problems: _Problems,
) -> Optional[dict[str, Any]]:
    """Check an object's keys and fill its defaults; None when unusable."""
    if not isinstance(value, Mapping):
        problems.add(
            path, f"is {_kind(value)}: pass an object with the keys {list(fields)}"
        )
        return None
    for key in value:
        if key not in fields:
            shown = _shown(key) if isinstance(key, str) else _kind(key)
            problems.add(
                path,
                f"has the unknown key {shown}: the allowed keys are {list(fields)}",
                kind="unknown_key",
            )
    for key in fields:
        if key not in defaults and value.get(key) is None:
            child = key if path == "interpretation" else f"{path}.{key}"
            problems.add(child, "is required", kind="missing")
    result: dict[str, Any] = {}
    for key in fields:
        raw = value.get(key)
        if raw is None and key in defaults:
            default = defaults[key]
            result[key] = (
                list(default)
                if isinstance(default, list)
                else (dict(default) if isinstance(default, dict) else default)
            )
        else:
            result[key] = raw
    return result


def _token(
    value: object, path: str, vocabulary: tuple[str, ...], problems: _Problems
) -> Optional[str]:
    """An exact vocabulary token, or a problem listing the vocabulary."""
    if isinstance(value, str) and value in vocabulary:
        return value
    hint = ""
    if isinstance(value, str) and any(sign in value for sign in _MICRO_SIGNS):
        hint = (
            " (the micro sign U+00B5 and the Greek mu U+03BC are never "
            "normalized in tokens: write 'us' for microseconds)"
        )
    elif isinstance(value, str) and not value.isascii():
        hint = " (tokens are ASCII)"
    problems.add(
        path,
        f"{_shown(value)} is not one of {list(vocabulary)}{hint}",
    )
    return None


def _integer(
    value: object, path: str, problems: _Problems, *, minimum: int, below: int
) -> Optional[int]:
    """An int (never a bool, a float or a string) in ``[minimum, below)``."""
    if isinstance(value, bool) or not isinstance(value, int):
        problems.add(path, f"is {_kind(value)}: pass an integer")
        return None
    if not minimum <= value < below:
        problems.add(
            path,
            f"{value} is out of range: pass an integer from {minimum} to {below - 1}",
        )
        return None
    return value


def _positive_decimal(value: object, path: str, problems: _Problems) -> Optional[str]:
    text: Optional[str] = problems.collect(
        path, value, lambda v: normalize_decimal(v, field=path)
    )
    if text is not None and decimal.Decimal(text) <= 0:
        problems.add(path, f"{_shown(text)} must be greater than zero")
        return None
    return text


_LINE_LIMIT = 2**53


def _normalize_dialect(value: object, problems: _Problems) -> Optional[dict[str, Any]]:
    dialect = _object(value, "dialect", _DIALECT_FIELDS, DIALECT_DEFAULTS, problems)
    if dialect is None:
        return None
    out: dict[str, Any] = {key: None for key in _DIALECT_FIELDS}
    for key, vocabulary in (
        ("encoding", ENCODINGS),
        ("delimiter", DELIMITERS),
        ("decimal", DECIMAL_MARKS),
    ):
        if dialect[key] is not None:
            out[key] = _token(dialect[key], f"dialect.{key}", vocabulary, problems)
    for key in ("header_line", "units_line"):
        if dialect[key] is not None:
            out[key] = _integer(
                dialect[key], f"dialect.{key}", problems, minimum=1, below=_LINE_LIMIT
            )
    if out["delimiter"] is not None and out["delimiter"] == out["decimal"]:
        problems.add(
            "dialect.delimiter",
            f"{_shown(out['delimiter'])} equals the decimal mark: a field "
            f"delimiter and a decimal mark must differ",
        )
    if out["header_line"] is not None and out["header_line"] == out["units_line"]:
        problems.add("dialect.units_line", "names the same line as header_line")
    return out


def _normalize_data_lines(
    value: object, dialect: Optional[dict[str, Any]], problems: _Problems
) -> Optional[dict[str, Any]]:
    lines = _object(
        value, "data_lines", _DATA_LINES_FIELDS, DATA_LINES_DEFAULTS, problems
    )
    if lines is None:
        return None
    first = None
    if lines["first"] is not None:
        first = _integer(
            lines["first"], "data_lines.first", problems, minimum=1, below=_LINE_LIMIT
        )
    last = None
    if lines["last"] is not None:
        last = _integer(
            lines["last"], "data_lines.last", problems, minimum=1, below=_LINE_LIMIT
        )
    if first is not None and last is not None and last < first:
        problems.add("data_lines.last", f"{last} is before data_lines.first {first}")
    if first is not None and dialect is not None:
        for key in ("header_line", "units_line"):
            line = dialect.get(key)
            if line is not None and line >= first:
                problems.add(
                    f"dialect.{key}",
                    f"{line} is not before data_lines.first {first}: the header "
                    f"and units rows precede the data region",
                )
    return {"first": first, "last": last}


def _normalize_time_columns(value: object, problems: _Problems) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        problems.add("time_columns", f"is {_kind(value)}: pass a list")
        return []
    if len(value) > MAX_FIELDS:
        problems.add("time_columns", f"lists {len(value)} columns, over {MAX_FIELDS}")
        return []
    out = []
    for index, entry in enumerate(value):
        path = f"time_columns[{index}]"
        item = _object(entry, path, _TIME_COLUMN_FIELDS, {}, problems)
        if item is None:
            continue
        column = (
            None
            if item["column"] is None
            else _integer(
                item["column"], f"{path}.column", problems, minimum=0, below=MAX_FIELDS
            )
        )
        unit = (
            None
            if item["time_unit"] is None
            else _token(item["time_unit"], f"{path}.time_unit", TIME_UNITS, problems)
        )
        out.append({"column": column, "time_unit": unit})
    return out


def _normalize_time_base(
    value: object, time_columns: list[dict[str, Any]], problems: _Problems
) -> Optional[dict[str, Any]]:
    base = _object(value, "time_base", _TIME_BASE_FIELDS, TIME_BASE_DEFAULTS, problems)
    if base is None:
        return None
    source = None
    if base["source"] is not None:
        source = _token(base["source"], "time_base.source", TIME_BASE_SOURCES, problems)
    rate = None
    if base["sampling_rate_hz"] is not None:
        rate = _positive_decimal(
            base["sampling_rate_hz"], "time_base.sampling_rate_hz", problems
        )
    column = None
    if base["time_column"] is not None:
        column = _integer(
            base["time_column"],
            "time_base.time_column",
            problems,
            minimum=0,
            below=MAX_FIELDS,
        )
    if source == "time_column":
        if base["time_column"] is None:
            problems.add(
                "time_base.time_column",
                "is required when time_base.source is 'time_column'",
                kind="missing",
            )
        elif column is not None and column not in {c["column"] for c in time_columns}:
            problems.add(
                "time_base.time_column",
                f"{column} is not listed in time_columns: list it there with its time_unit",
            )
    elif source is not None and base["time_column"] is not None:
        problems.add(
            "time_base.time_column",
            f"is set but time_base.source is {_shown(source)}: set it only when "
            f"the source is 'time_column'",
        )
    return {"source": source, "sampling_rate_hz": rate, "time_column": column}


def _normalize_acquired_at(value: object) -> str:
    """An ISO 8601 instant: in UTC when an offset is declared (as the ledger
    records it, so two spellings of one instant are one interpretation),
    naive when none is (the ledger flags it ``timezone_declared=false``)."""
    if not isinstance(value, str) or not value.isascii() or len(value) > 64:
        raise ValueError(
            "declare an ASCII ISO 8601 timestamp of at most 64 characters, "
            "such as '2026-08-20T13:42:00+02:00'"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            "not an ISO 8601 timestamp; declare one such as "
            "'2026-08-20T13:42:00+02:00' (an offset or 'Z' declares the timezone)"
        ) from None
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return parsed.astimezone(timezone.utc).isoformat()
    return parsed.isoformat()


def _bounded_text(field: str) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if isinstance(value, str) and len(value) > MAX_FREE_TEXT_CHARS:
            raise ValueError(
                f"Invalid {field}: {len(value)} characters, over the "
                f"{MAX_FREE_TEXT_CHARS}-character limit."
            )
        return validate_free_text(field, value)

    return check


def _ledger_id(field: str) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if isinstance(value, str) and len(value) > 256:
            raise ValueError(f"Invalid {field}: {len(value)} characters long.")
        return validate_ledger_id(value, kind=field)

    return check


def _direction(value: Any) -> str:
    if isinstance(value, str) and len(value) > 32:
        raise ValueError(
            f"Invalid direction: {len(value)} characters long; declare one of "
            f"the measurement contract's directions."
        )
    return normalize_direction(value)


def _normalize_defaults(value: object, problems: _Problems) -> dict[str, Any]:
    fields = tuple(DECLARATION_DEFAULTS)
    given = _object(value, "defaults", fields, DECLARATION_DEFAULTS, problems)
    out = dict(DECLARATION_DEFAULTS)
    if given is None:
        return out
    if given["asset_id"] is not None:
        out["asset_id"] = problems.collect(
            "defaults.asset_id", given["asset_id"], _ledger_id("asset_id")
        )
    if given["acquired_at"] is not None:
        out["acquired_at"] = problems.collect(
            "defaults.acquired_at", given["acquired_at"], _normalize_acquired_at
        )
    if given["rpm"] is not None:
        out["rpm"] = _positive_decimal(given["rpm"], "defaults.rpm", problems)
    if given["load"] is not None:
        out["load"] = problems.collect(
            "defaults.load",
            given["load"],
            lambda v: normalize_decimal(v, field="defaults.load"),
        )
    if given["operating_state"] is not None:
        out["operating_state"] = problems.collect(
            "defaults.operating_state",
            given["operating_state"],
            _bounded_text("operating_state"),
        )
    return out


def _normalize_scaling(
    value: object, path: str, unit: Optional[str], problems: _Problems
) -> Optional[dict[str, Any]]:
    scaling = _object(value, path, _SCALING_FIELDS, {}, problems)
    if scaling is None:
        return None
    source_unit = None
    if scaling["source_unit"] is not None:
        source_unit = _token(
            scaling["source_unit"], f"{path}.source_unit", ELECTRICAL_UNITS, problems
        )
    dc_policy = None
    if scaling["dc_policy"] is not None:
        dc_policy = _token(
            scaling["dc_policy"], f"{path}.dc_policy", DC_POLICIES, problems
        )
    sensitivity = None
    if scaling["sensitivity"] is not None:
        given = _object(
            scaling["sensitivity"],
            f"{path}.sensitivity",
            _SENSITIVITY_FIELDS,
            {},
            problems,
        )
        if given is not None:
            sensitivity = {
                "value": (
                    None
                    if given["value"] is None
                    else _positive_decimal(
                        given["value"], f"{path}.sensitivity.value", problems
                    )
                ),
                "unit": (
                    None
                    if given["unit"] is None
                    else _token(
                        given["unit"],
                        f"{path}.sensitivity.unit",
                        tuple(SENSITIVITY_UNITS),
                        problems,
                    )
                ),
                "source": (
                    None
                    if given["source"] is None
                    else _token(
                        given["source"],
                        f"{path}.sensitivity.source",
                        SENSITIVITY_SOURCES,
                        problems,
                    )
                ),
            }
            physical = (
                SENSITIVITY_UNITS[sensitivity["unit"]][1]
                if sensitivity["unit"] is not None
                else None
            )
            if physical is not None and unit is not None and physical != unit:
                problems.add(
                    f"{path}.sensitivity.unit",
                    f"{_shown(sensitivity['unit'])} is stated per {_shown(physical)} "
                    f"but the channel's unit is {_shown(unit)}: store the channel in "
                    f"{_shown(physical)}, or state the sensitivity per {_shown(unit)} "
                    f"(no unit conversion is applied to a sensitivity)",
                )
    return {
        "source_unit": source_unit,
        "sensitivity": sensitivity,
        "dc_policy": dc_policy,
    }


def _normalize_binding(
    value: object, path: str, problems: _Problems
) -> Optional[dict[str, Any]]:
    binding = _object(value, path, _BINDING_FIELDS, BINDING_DEFAULTS, problems)
    if binding is None:
        return None
    out: dict[str, Any] = {key: None for key in _BINDING_FIELDS}
    if binding["measurement_point_id"] is not None:
        out["measurement_point_id"] = problems.collect(
            f"{path}.measurement_point_id",
            binding["measurement_point_id"],
            _ledger_id("measurement_point_id"),
        )
    if binding["asset_id"] is not None:
        out["asset_id"] = problems.collect(
            f"{path}.asset_id", binding["asset_id"], _ledger_id("asset_id")
        )
    if binding["direction"] is not None:
        out["direction"] = problems.collect(
            f"{path}.direction", binding["direction"], _direction
        )
    if binding["sensor_id"] is not None:
        out["sensor_id"] = problems.collect(
            f"{path}.sensor_id", binding["sensor_id"], _bounded_text("sensor_id")
        )
    return out


def _normalize_channel(
    entry: object, path: str, problems: _Problems
) -> Optional[dict[str, Any]]:
    channel = _object(entry, path, _CHANNEL_FIELDS, CHANNEL_DEFAULTS, problems)
    if channel is None:
        return None
    column = None
    if channel["column"] is not None:
        column = _integer(
            channel["column"], f"{path}.column", problems, minimum=0, below=MAX_FIELDS
        )
    role = None
    if channel["role"] is not None:
        if channel["role"] in IMPORTABLE_ROLES:
            role = channel["role"]
        elif channel["role"] in ROLES:
            problems.add(
                f"{path}.role",
                f"{_shown(channel['role'])} is a preview role: inspection reports "
                f"it, but only {list(IMPORTABLE_ROLES)} channels are imported in "
                f"this version",
            )
        else:
            _token(channel["role"], f"{path}.role", ROLES, problems)
    quantity = None
    if channel["quantity"] is not None:
        quantity = _token(channel["quantity"], f"{path}.quantity", QUANTITIES, problems)
    unit = None
    if channel["unit"] is not None:
        every_unit = tuple(u for units in UNIT_FAMILIES.values() for u in units)
        unit = _token(channel["unit"], f"{path}.unit", every_unit, problems)
        if (
            unit is not None
            and quantity is not None
            and unit not in UNIT_FAMILIES[quantity]
        ):
            problems.add(
                f"{path}.unit",
                f"{_shown(unit)} is not a unit of {_shown(quantity)}: use one of "
                f"{list(UNIT_FAMILIES[quantity])}",
            )
            unit = None
    scaling = None
    if channel["scaling"] is not None:
        scaling = _normalize_scaling(
            channel["scaling"], f"{path}.scaling", unit, problems
        )
    binding = None
    if channel["binding"] is not None:
        binding = _normalize_binding(channel["binding"], f"{path}.binding", problems)
    return {
        "column": column,
        "role": role,
        "quantity": quantity,
        "unit": unit,
        "scaling": scaling,
        "binding": binding,
    }


def _normalize_channels(value: object, problems: _Problems) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        problems.add("channels", f"is {_kind(value)}: pass a list of channel entries")
        return []
    if not value:
        problems.add(
            "channels",
            "is empty: select at least one vibration channel",
            kind="missing",
        )
        return []
    if len(value) > MAX_FIELDS:
        problems.add("channels", f"lists {len(value)} channels, over {MAX_FIELDS}")
        return []
    channels = []
    for index, entry in enumerate(value):
        channel = _normalize_channel(entry, f"channels[{index}]", problems)
        if channel is not None:
            channels.append(channel)
    return channels


def _sort_by_column(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda entry: entry["column"])


def _check_columns(
    channels: list[dict[str, Any]],
    time_columns: list[dict[str, Any]],
    problems: _Problems,
) -> None:
    """Refuse a column given two roles (entries all carry a valid column)."""
    seen: dict[int, str] = {}
    for label, entries in (("channels", channels), ("time_columns", time_columns)):
        for entry in entries:
            column = entry["column"]
            if column in seen:
                problems.add(
                    label,
                    f"lists column {column} more than once (also in {seen[column]}): "
                    f"each physical column has one role",
                )
            else:
                seen[column] = label


def normalize_interpretation(interpretation: object) -> dict[str, Any]:
    """Return the canonical form of an import interpretation.

    Pure function. Idempotent: the canonical form normalizes to itself.

    Args:
        interpretation: A mapping in the shape of the module docstring
            (parsed JSON, a model dump, or a canonical form).

    Returns:
        A new dict with every key of the canonical form, in that order:
        defaults filled, decimal strings normalized, ``direction``
        normalized, ``acquired_at`` in UTC when it declares an offset and
        naive when it does not, and ``channels`` and ``time_columns``
        sorted by column.

    Raises:
        InterpretationError: A ``ValueError`` whose message lists every
            problem and whose ``problems`` type each one (missing, unknown
            key, invalid):
            unknown or missing keys, floats or strings where an integer is
            expected, JSON numbers where a decimal string is expected,
            out-of-vocabulary or non-ASCII tokens, out-of-range line or
            column numbers, a delimiter equal to the decimal mark, a column
            given two roles, a time base inconsistent with its time column,
            a preview role on a channel, a unit outside the channel's
            quantity, a sensitivity stated per another unit, and every
            refusal of the measurement contract's id, direction and
            free-text rules. Caller values are cited short and escaped.
    """
    problems = _Problems()
    top = _object(
        interpretation,
        "interpretation",
        _INTERPRETATION_FIELDS,
        INTERPRETATION_DEFAULTS,
        problems,
    )
    if top is None:
        problems.raise_if_any()
        raise AssertionError("unreachable: _object reports every unusable value")
    if top["schema"] is not None and top["schema"] != INTERPRETATION_SCHEMA:
        problems.add(
            "schema",
            f"{_shown(top['schema'])} is not {INTERPRETATION_SCHEMA!r}, the "
            f"only schema this version reads",
        )
    dialect = (
        None if top["dialect"] is None else _normalize_dialect(top["dialect"], problems)
    )
    data_lines = (
        None
        if top["data_lines"] is None
        else _normalize_data_lines(top["data_lines"], dialect, problems)
    )
    time_columns = _sort_by_column(
        [
            c
            for c in _normalize_time_columns(top["time_columns"], problems)
            if c["column"] is not None
        ]
    )
    time_base = (
        None
        if top["time_base"] is None
        else _normalize_time_base(top["time_base"], time_columns, problems)
    )
    defaults = _normalize_defaults(top["defaults"], problems)
    channels = (
        []
        if top["channels"] is None
        else _sort_by_column(
            [
                c
                for c in _normalize_channels(top["channels"], problems)
                if c["column"] is not None
            ]
        )
    )
    _check_columns(channels, time_columns, problems)
    problems.raise_if_any()
    return {
        "schema": INTERPRETATION_SCHEMA,
        "dialect": dialect,
        "data_lines": data_lines,
        "time_base": time_base,
        "time_columns": time_columns,
        "defaults": defaults,
        "channels": channels,
    }


def _refuse_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Invalid import interpretation: the key {_shown(key)} appears "
                f"twice in one object. Keep one value per key and validate the "
                f"interpretation again."
            )
        result[key] = value
    return result


def _refuse_constant(name: str) -> Any:
    raise ValueError(
        f"Invalid import interpretation: {name} is not a JSON number. "
        f"{_DECIMAL_HELP[0].upper()}{_DECIMAL_HELP[1:]}."
    )


#: Largest interpretation text parsed; a real one is a few kilobytes.
MAX_INTERPRETATION_CHARS = 1 << 20

#: Longest integer literal parsed (line numbers stay below 2**53).
_MAX_INT_DIGITS = 20


def _parse_int(token: str) -> int:
    if len(token.lstrip("-")) > _MAX_INT_DIGITS:
        raise ValueError(
            f"Invalid import interpretation: an integer of {len(token)} "
            f"characters is out of range. {_REMEDY}"
        )
    return int(token)


def loads_interpretation(text: str) -> dict[str, Any]:
    """Parse JSON text strictly, then normalize it.

    The entry point wherever the server parses the text itself: a key that
    appears twice in one object is refused (the default parser would keep
    the last value silently), and so are NaN, Infinity, integer literals over
    20 digits, nesting deeper than the parser can follow, and text over
    :data:`MAX_INTERPRETATION_CHARS`.

    Raises:
        ValueError: For a non-string or oversized text, malformed JSON, any
            of the refusals above, or any refusal of
            :func:`normalize_interpretation`. No other exception escapes.
    """
    if not isinstance(text, str):
        raise ValueError(
            f"Invalid import interpretation: expected JSON text, got "
            f"{_kind(text)}. {_REMEDY}"
        )
    if len(text) > MAX_INTERPRETATION_CHARS:
        raise ValueError(
            f"Invalid import interpretation: {len(text)} characters, over the "
            f"{MAX_INTERPRETATION_CHARS}-character cap. {_REMEDY}"
        )
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_refuse_duplicate_keys,
            parse_constant=_refuse_constant,
            parse_int=_parse_int,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid import interpretation: not valid JSON ({exc.msg} at line "
            f"{exc.lineno}, column {exc.colno}). {_REMEDY}"
        ) from None
    except RecursionError:
        raise ValueError(
            f"Invalid import interpretation: nested too deeply to parse. {_REMEDY}"
        ) from None
    return normalize_interpretation(parsed)


# ---------------------------------------------------------------------------
# Identities and check values
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"[0-9a-f]{64}")

#: Domain separator of :func:`import_id`.
_IMPORT_ID_DOMAIN = b"pmm-import/1\x00"


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def is_import_id(value: object) -> bool:
    """Whether *value* has the form of an ``import_id``: exactly 64 lowercase
    hexadecimal characters (checked before any path is built from it)."""
    return _is_hex64(value)


def config_digest(interpretation: Mapping[str, Any]) -> str:
    """SHA-256 (64 hex) of the canonical JSON of the canonical interpretation.

    Raises:
        ValueError: From :func:`normalize_interpretation`.
    """
    return content_hash(normalize_interpretation(interpretation))


def import_id(
    source_sha256: str, size_bytes: int, interpretation: Mapping[str, Any]
) -> str:
    """Identity of one interpretation of one file's bytes.

    ``sha256(b"pmm-import/1\\0" + <source sha256 hex> + b"\\0" + <size> +
    b"\\0" + <config digest hex>)``, all ASCII. It excludes signal ids,
    confirmation details and package versions.

    Args:
        source_sha256: SHA-256 of the file bytes, 64 lowercase hex.
        size_bytes: Size of the file in bytes.
        interpretation: The interpretation (normalized here).

    Returns:
        64 lowercase hex characters.

    Raises:
        ValueError: For a malformed digest or size, or from the normalizer.
    """
    if not _is_hex64(source_sha256):
        raise ValueError(
            "source_sha256 must be the 64-character lowercase hex SHA-256 of the "
            "file bytes."
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ValueError("size_bytes must be a non-negative integer.")
    payload = b"\x00".join(
        (
            source_sha256.encode("ascii"),
            str(size_bytes).encode("ascii"),
            config_digest(interpretation).encode("ascii"),
        )
    )
    return hashlib.sha256(_IMPORT_ID_DOMAIN + payload).hexdigest()


def channel_interpretation(
    interpretation: Mapping[str, Any], column: int
) -> dict[str, Any]:
    """The part of an interpretation that shapes one channel's samples.

    The file-level fields that decide its samples (schema, dialect, data line
    range, time base) plus the channel's own entry without its binding (the
    ledger declaration carries the binding in its own keys). Editing or
    adding another channel, a time column or a declaration default leaves it
    unchanged.

    Raises:
        ValueError: From the normalizer, or when no channel reads *column*.
    """
    canonical = normalize_interpretation(interpretation)
    for channel in canonical["channels"]:
        if channel["column"] == column and not isinstance(column, bool):
            return {
                "schema": canonical["schema"],
                "dialect": canonical["dialect"],
                "data_lines": canonical["data_lines"],
                "time_base": canonical["time_base"],
                "channel": {
                    key: channel[key] for key in _SAMPLE_SHAPING_CHANNEL_FIELDS
                },
            }
    raise ValueError(
        f"No channel of the interpretation reads column {column!r}; the channels "
        f"read columns {[c['column'] for c in canonical['channels']]}."
    )


def channel_interpretation_digest(
    interpretation: Mapping[str, Any], column: int
) -> str:
    """SHA-256 (64 hex) of :func:`channel_interpretation`."""
    return content_hash(channel_interpretation(interpretation, column))


def samples_digest(samples: np.ndarray) -> str:
    """SHA-256 of a channel's samples as C-contiguous little-endian float64.

    A check value, not an identity: replay must reproduce it bit for bit.

    Raises:
        ValueError: If *samples* is not one-dimensional.
    """
    array = np.ascontiguousarray(samples, dtype="<f8")
    if array.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {tuple(array.shape)}."
        )
    return hashlib.sha256(array.data).hexdigest()


#: Power-of-ten exponent of each electrical unit, in volts.
_ELECTRICAL_EXPONENT = {"mV": -3, "V": 0}


def scaling_divisor(scaling: Mapping[str, Any]) -> decimal.Decimal:
    """The exact divisor of a scaled channel, in source unit per stored unit.

    Each scaled sample is ``raw / divisor`` (one division, in float64, by the
    float nearest to this value). With the sensitivity stated per the
    channel's own unit, the divisor is the stated value times a power of ten
    (``100 mV/g`` on a column in ``V`` gives ``0.1`` V/g), computed exactly.

    Args:
        scaling: A canonical ``scaling`` block.

    Raises:
        ValueError: For a block that is not canonical.
    """
    try:
        sensitivity = scaling["sensitivity"]
        electrical, _physical = SENSITIVITY_UNITS[sensitivity["unit"]]
        shift = (
            _ELECTRICAL_EXPONENT[electrical]
            - _ELECTRICAL_EXPONENT[scaling["source_unit"]]
        )
        with decimal.localcontext(_DECIMAL_CONTEXT):
            return decimal.Decimal(sensitivity["value"]).scaleb(shift)
    except (KeyError, TypeError, decimal.DecimalException):
        raise ValueError(
            "scaling is not a canonical scaling block; normalize the "
            "interpretation first."
        ) from None


def build_import_format(
    interpretation: Mapping[str, Any],
    column: int,
    *,
    samples_digest: str,
    sample_count: int,
) -> dict[str, Any]:
    """The ``import_format`` a recorded imported channel carries.

    Everything replay needs besides the declaration's own ``sampling_rate``,
    ``signal_unit`` and ``channel_index``: the channel interpretation, the
    time base without its rate, the exact scaling divisor (as a decimal
    string and as ``float.hex()``), and the samples digest and count replay
    must reproduce.

    Args:
        interpretation: The interpretation (normalized here).
        column: The channel's physical column.
        samples_digest: :func:`samples_digest` of the decoded, scaled samples.
        sample_count: Number of those samples.

    Returns:
        A dict with exactly :data:`IMPORT_FORMAT_KEYS`, in that order.

    Raises:
        ValueError: From the normalizer, for an unknown column, or a
            malformed digest or count.
    """
    if not _is_hex64(samples_digest):
        raise ValueError("samples_digest must be 64 lowercase hex characters.")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer.")
    selected = channel_interpretation(interpretation, column)
    channel = selected["channel"]
    scaling = channel["scaling"]
    recorded_scaling = None
    if scaling is not None:
        divisor = scaling_divisor(scaling)
        recorded_scaling = {
            **scaling,
            "divisor": _format_decimal(divisor),
            "divisor_hex": float(divisor).hex(),
        }
    return {
        "schema": selected["schema"],
        "dialect": selected["dialect"],
        "data_lines": selected["data_lines"],
        "time_base": {
            "source": selected["time_base"]["source"],
            "time_column": selected["time_base"]["time_column"],
        },
        "quantity": channel["quantity"],
        "scaling": recorded_scaling,
        "samples_digest": samples_digest,
        "sample_count": sample_count,
    }


# ---------------------------------------------------------------------------
# The raw_format marker and the raw-declaration predicate
# ---------------------------------------------------------------------------


def import_raw_marker() -> dict[str, str]:
    """A fresh copy of the ``raw_format`` marker of an imported declaration.

    The ledger writes it beside ``import_format`` and nowhere else: it never
    appears in ``StoredSignalInfo.raw_format``.
    """
    return {"sample_format": IMPORT_RAW_MARKER_SAMPLE_FORMAT}


def is_import_marker(raw_format: object) -> bool:
    """Whether *raw_format* names the import marker's ``sample_format``."""
    return (
        isinstance(raw_format, Mapping)
        and raw_format.get("sample_format") == IMPORT_RAW_MARKER_SAMPLE_FORMAT
    )


_NOT_DECODABLE_REMEDY = (
    "it is never decoded as the file's first column; re-import the original "
    "file through the signal import"
)


def is_raw_declaration(declaration: Mapping[str, Any]) -> bool:
    """Whether a recorded declaration is decoded by the raw binary decoder.

    The one predicate every ``raw_format`` reader goes through.

    Returns:
        True for a ``raw_format`` block of the raw decoder (complete or not:
        an incomplete block keeps its own refusal); False for a declaration
        without ``raw_format`` and for an imported declaration (the marker
        plus an ``import_format``).

    Raises:
        ValueError: For the marker without an ``import_format``, a marker
            with any other key, or an ``import_format`` without the marker.
            Such a declaration is refused, never read as column 0.
    """
    raw_format = declaration.get("raw_format")
    import_format = declaration.get("import_format")
    if is_import_marker(raw_format):
        if dict(raw_format) != import_raw_marker():
            raise ValueError(
                "recorded raw_format carries the import marker together with "
                f"other keys, so its interpretation is unknown; {_NOT_DECODABLE_REMEDY}"
            )
        if not isinstance(import_format, Mapping):
            raise ValueError(
                "recorded raw_format is the import marker but the declaration "
                f"has no import_format; {_NOT_DECODABLE_REMEDY}"
            )
        return False
    if import_format is not None:
        raise ValueError(
            "recorded declaration has an import_format but its raw_format is not "
            f"the import marker; {_NOT_DECODABLE_REMEDY}"
        )
    return isinstance(raw_format, Mapping)
