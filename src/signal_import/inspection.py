"""
Read-only inspection of a delimited-text signal export.

ISO 13374 Block 1 (data acquisition). The first step of an import: open a
file nobody has vouched for, read a bounded head and tail of it, and report
what it appears to hold, labelling every value with how it was obtained.
Inspection never writes anything (the file, its folder, the signal store,
the ledger and the import record store are untouched), never confirms a
value, and never echoes file text except as ``FileText``.

Order of work:

1. The path is gated and confined to the import roots before any
   operating-system call, then opened through one verified descriptor
   (``path_safety.open_import_file``); binary, compressed, UTF-16 and UTF-32
   content is refused before any text is read back (``text_scan``). Those
   refusals raise; they echo no file text beyond an escaped name.
2. Every (delimiter, decimal mark) reading is scored over the scanned lines:
   the longest run of consecutive rows with the same field count and the
   same numeric columns, then the share of that run's fields that are
   numbers. A row wider than the field cap is inconsistent under that
   reading only. Readings that split the run into the same fields are one
   reading, spelled with the first delimiter of the closed vocabulary. A
   unique best reading is ``proposed``; a tie is ``ambiguous`` and every
   reading is listed.
3. Regions follow from the best reading: the data run; above it, a units row
   (every token a unit token) and a header row; above those, the preamble,
   where ``key<separator>value`` lines become key-value pairs; below the
   data, a trailer.
4. A page of at most :data:`COLUMNS_PER_PAGE` columns is profiled over the
   scanned data rows: label, unit token, role candidate, numeric share,
   statistics, at most :data:`HEAD_TOKENS` verbatim tokens and, for a time
   candidate (monotonic, time-labelled, carrying a time unit, or printed as
   dates), the time-base rules applied to the scanned rows only.
5. The result is kept within :data:`OUTPUT_BUDGET_BYTES` of compact JSON:
   when it would exceed it, the column page shrinks first (the caller pages
   on with ``next_columns_from``), then preamble lines are dropped and
   counted.

With no numeric data region, the result echoes no preamble or label text.
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Optional

from .. import config
from ..file_text import LABEL_CHARS, PREAMBLE_LINE_CHARS, TOKEN_CHARS, FileText
from ..path_safety import open_import_file
from ..signal_acquisition.import_config import (
    DECIMAL_MARKS,
    DELIMITERS,
    ELECTRICAL_UNITS,
    MAX_FIELDS,
    SENSITIVITY_UNITS,
    TIME_UNITS,
    normalize_decimal,
)
from ..signal_acquisition.measurement import SIGNAL_UNIT_ALIASES, UNIT_FAMILIES
from ..signal_acquisition.text_scan import ScannedLine, TextScan, scan_text
from .contracts import (
    Candidate,
    ColumnStats,
    Dialect,
    DialectReading,
    EncodingEvidence,
    InspectedColumn,
    InspectedFile,
    InspectionResult,
    InspectionStatus,
    Issue,
    KeyValue,
    Preamble,
    PreambleLine,
    Region,
    ScanSummary,
    TimeStats,
    ValueStatus,
)
from .timebase import TimeBaseRefused, assess_time_base, parse_time_tokens

__all__ = [
    "COLUMNS_PER_PAGE",
    "HEAD_TOKENS",
    "MAX_PREAMBLE_LINES",
    "MIN_WAVEFORM_ROWS",
    "OUTPUT_BUDGET_BYTES",
    "PREVIEW_ROLES",
    "inspect_signal_file",
]

#: Columns profiled per page (fewer when the size budget requires).
COLUMNS_PER_PAGE = 32

#: Verbatim tokens shown per column.
HEAD_TOKENS = 3

#: Preamble lines shown at most.
MAX_PREAMBLE_LINES = 40

#: Fewest data rows this version requires of a fully scanned file before it
#: offers it for import: the 256-sample default segment of the spectral
#: tools. A default requirement of the analysis, not a definition of what a
#: waveform is.
MIN_WAVEFORM_ROWS = 256

#: Largest inspection result, in bytes of compact JSON.
OUTPUT_BUDGET_BYTES = 24 * 1024

#: Preview-only unit tokens (lower case) and the role each suggests.
PREVIEW_ROLES: dict[str, str] = {
    "um": "displacement",
    "mil": "displacement",
    "in/s": "other",
    "rpm": "tachometer",
    "degc": "temperature",
    "a": "current",
}

_NUMBER = {
    ".": re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"),
    ",": re.compile(r"[+-]?(?:\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?"),
}
_ANY_NUMBER = re.compile(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?")
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_SEPARATOR = re.compile("[:=]")
_DELIMITER_CHAR = re.compile("[ \t]")
# Patterns on stripped text with no two quantifiers competing for the same
# characters, so a long run of spaces cannot make matching quadratic.
_BRACKETED = re.compile(r"[\[(]([^\[\]()]{1,64})[\])]")
_LABEL_UNIT = re.compile(r".*[\[(]([^\[\]()]{1,64})[\])]")
_MAX_UNIT_CHARS = 16
_BRACKET_PART = re.compile(r"[\[(][^\[\]()]*[\])]")
_TIME_LABEL = re.compile(
    r"(?:^|[\s_\-./])(?:time|t|zeit|tempo|temps|tiempo|timestamp|x_value|sec|"
    r"seconds?)(?:$|[\s_\-./\[(])",
    re.IGNORECASE,
)

#: Preamble keys (lower case, separators as single spaces) and the field
#: each names.
_PREAMBLE_KEYS: dict[str, str] = {
    "sampling rate": "sampling_rate_hz",
    "sample rate": "sampling_rate_hz",
    "sampling frequency": "sampling_rate_hz",
    "samplerate": "sampling_rate_hz",
    "fs": "sampling_rate_hz",
    "delta x": "sample_interval",
    "dt": "sample_interval",
    "sample interval": "sample_interval",
    "sampling interval": "sample_interval",
    "sensitivity": "sensitivity",
    "start time": "acquired_at",
    "date": "acquired_at",
    "time": "acquired_at",
    "acquired at": "acquired_at",
    "rpm": "rpm",
    "speed": "rpm",
    "shaft speed": "rpm",
    "unit": "unit",
    "units": "unit",
}
#: Numeric preamble fields and the unit spellings (lower case) after the
#: number that let the number stand as that field's value; any other text
#: after it (``25.6 kHz``, ``30 Hz`` for a speed) leaves it unnormalized.
_FIELD_UNITS: dict[str, frozenset[str]] = {
    "sampling_rate_hz": frozenset({"", "hz"}),
    "sample_interval": frozenset({"", "s"}),
    "sensitivity": frozenset({""} | {unit.lower() for unit in SENSITIVITY_UNITS}),
    "rpm": frozenset({"", "rpm", "1/min"}),
}

_IMPORTABLE_UNITS = {unit for units in UNIT_FAMILIES.values() for unit in units}
_ELECTRICAL = {unit.lower(): unit for unit in ELECTRICAL_UNITS}

_CONFIRM_CHANNELS = "each imported channel's column index, role, quantity and unit"
_CONFIRM_RATE = "the sampling rate and the source it comes from"
_CONFIRM_BINDING = "the asset, measurement point and direction of each channel"
_CONFIRM_DIALECT = "the dialect (delimiter and decimal mark)"
_CONFIRM_ENCODING = (
    "the text encoding (a Windows-1252 decode alone does not establish it)"
)
_CONFIRM_COMPANION = "that the companion metadata file next to this file is not read"


# ---------------------------------------------------------------------------
# Rows and readings
# ---------------------------------------------------------------------------


def _split(text: str, delimiter: str) -> Optional[list[str]]:
    """The fields of *text* under *delimiter*; None over the field cap."""
    if delimiter == "whitespace":
        stripped = text.strip(" \t")
        fields = _WHITESPACE_RUN.split(stripped) if stripped else [""]
    else:
        fields = text.split("\t" if delimiter == "tab" else delimiter)
    return None if len(fields) > MAX_FIELDS else fields


def _number(token: str, decimal: str) -> Optional[float]:
    token = token.strip(" ")
    if _NUMBER[decimal].fullmatch(token) is None:
        return None
    value = float(token.replace(",", ".") if decimal == "," else token)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class _Row:
    fields: tuple[str, ...]
    numeric: frozenset[int]
    trailing_empty: bool


def _row(text: str, delimiter: str, decimal: str) -> Optional[_Row]:
    """A row with at least one number under the reading, else None."""
    if not text.strip(" \t"):
        return None
    fields = _split(text, delimiter)
    if fields is None:
        return None
    trailing = len(fields) > 1 and fields[-1].strip(" ") == ""
    if trailing:
        fields = fields[:-1]
    numeric = frozenset(
        index
        for index, token in enumerate(fields)
        if _number(token, decimal) is not None
    )
    if not numeric:
        return None
    return _Row(tuple(field.strip(" ") for field in fields), numeric, trailing)


@dataclass(frozen=True)
class _Reading:
    delimiter: str
    decimal: str
    first: int  # index into the head lines
    last: int
    fields: int
    numeric: frozenset[int]
    trailing_empty: bool
    rows: tuple[tuple[str, ...], ...]

    @property
    def numeric_ratio(self) -> float:
        return len(self.numeric) / self.fields

    @property
    def score(self) -> tuple[int, float]:
        return (len(self.rows), round(self.numeric_ratio, 9))

    def matches(self, row: Optional[_Row]) -> bool:
        return (
            row is not None
            and len(row.fields) == self.fields
            and row.numeric == self.numeric
        )


def _best_run(
    lines: tuple[ScannedLine, ...], delimiter: str, decimal: str
) -> Optional[_Reading]:
    best: Optional[_Reading] = None
    start = 0
    while start < len(lines):
        first = _row(lines[start].text, delimiter, decimal)
        if first is None:
            start += 1
            continue
        rows = [first.fields]
        trailing = first.trailing_empty
        end = start
        while end + 1 < len(lines):
            nxt = _row(lines[end + 1].text, delimiter, decimal)
            if (
                nxt is None
                or len(nxt.fields) != len(first.fields)
                or (nxt.numeric != first.numeric)
            ):
                break
            rows.append(nxt.fields)
            trailing = trailing or nxt.trailing_empty
            end += 1
        reading = _Reading(
            delimiter=delimiter,
            decimal=decimal,
            first=start,
            last=end,
            fields=len(first.fields),
            numeric=first.numeric,
            trailing_empty=trailing,
            rows=tuple(rows),
        )
        if best is None or reading.score > best.score:
            best = reading
        start = end + 1
    return best


def _readings(lines: tuple[ScannedLine, ...]) -> list[_Reading]:
    """The best-scoring readings, one per distinct split of the data run."""
    scored = [
        reading
        for delimiter in DELIMITERS
        for decimal in DECIMAL_MARKS
        if delimiter != decimal
        for reading in [_best_run(lines, delimiter, decimal)]
        if reading is not None
    ]
    if not scored:
        return []
    top = max(reading.score for reading in scored)
    distinct: dict[tuple, _Reading] = {}
    for reading in scored:  # vocabulary order picks each group's spelling
        if reading.score == top:
            distinct.setdefault((reading.first, reading.rows), reading)
    return list(distinct.values())


# ---------------------------------------------------------------------------
# Units, labels and the preamble
# ---------------------------------------------------------------------------


def _bracketed(text: str, pattern: "re.Pattern[str]") -> Optional[str]:
    """The unit inside the brackets *pattern* finds, 1 to 16 characters."""
    match = pattern.fullmatch(text.strip())
    if match is None:
        return None
    inner = match.group(1).strip()
    return inner if 1 <= len(inner) <= _MAX_UNIT_CHARS else None


def _unit_text(token: str) -> str:
    inner = _bracketed(token, _BRACKETED)
    return inner if inner is not None else token.strip()


def _classify_unit(token: str) -> tuple[Optional[str], Optional[str], str]:
    """``(importable unit, suggested role, kind)`` of a unit token; *kind* is
    ``importable``, ``electrical``, ``time``, ``preview`` or
    ``unrecognized``. Only ASCII spellings map; any other token is evidence
    as written."""
    text = _unit_text(token)
    folded = text.lower()
    folded = SIGNAL_UNIT_ALIASES.get(folded, folded)
    if folded in _IMPORTABLE_UNITS:
        return folded, "vibration", "importable"
    if not folded.isascii():
        return None, None, "unrecognized"
    if folded in _ELECTRICAL:
        return None, "voltage", "electrical"
    if folded in TIME_UNITS:
        return None, "time", "time"
    if folded in PREVIEW_ROLES:
        return None, PREVIEW_ROLES[folded], "preview"
    return None, None, "unrecognized"


def _is_units_row(fields: list[str]) -> bool:
    tokens = [field.strip() for field in fields if field.strip()]
    return bool(tokens) and all(
        _bracketed(token, _BRACKETED) is not None
        or _classify_unit(token)[2] != "unrecognized"
        for token in tokens
    )


def _cited(raw: str, cap: int) -> FileText:
    return FileText.of(raw, cap=cap)


def _key_value(text: str, delimiter: str) -> Optional[tuple[str, str]]:
    """``(key, value)`` of a preamble line, or None.

    A ``:`` or ``=`` separates the key when it comes before the first data
    delimiter, or anywhere in a whitespace export, whose keys hold spaces
    (``Sampling rate: 25600``);
    otherwise the data delimiter does (the key is the first field, the value
    the rest, a trailing ``:`` or ``=`` dropped from the key). A key holds a
    letter. Linear in the line's length.
    """
    stripped = text.strip()
    separator = _SEPARATOR.search(stripped)
    if delimiter == "whitespace":
        boundary = _DELIMITER_CHAR.search(stripped)
    else:
        boundary = re.search(
            re.escape("\t" if delimiter == "tab" else delimiter), stripped
        )
    pair: Optional[tuple[str, str]] = None
    if separator is not None and (
        delimiter == "whitespace"  # a key of several words holds spaces
        or boundary is None
        or separator.start() < boundary.start()
    ):
        pair = (
            stripped[: separator.start()].strip(),
            stripped[separator.end() :].strip(),
        )
    else:
        fields = _split(stripped, delimiter)
        filled = [field.strip() for field in fields or [] if field.strip()]
        if len(filled) >= 2:
            joiner = {"tab": "\t", "whitespace": " "}.get(delimiter, delimiter)
            pair = filled[0].rstrip(":=").strip(), joiner.join(filled[1:])
        elif separator is not None:
            pair = (
                stripped[: separator.start()].strip(),
                stripped[separator.end() :].strip(),
            )
    if (
        pair is None
        or not pair[0]
        or not pair[1]
        or len(pair[0]) > LABEL_CHARS
        or not re.search("[A-Za-z]", pair[0])
    ):
        return None
    return pair


def _normalized(field: str, value: str, decimal: str) -> Optional[str]:
    """The value as a canonical decimal string, only when it is a number in
    the data's decimal mark followed by nothing or the field's own unit."""
    text = value.strip()
    number = _ANY_NUMBER.match(text)
    if (
        number is None
        or text[number.end() :].strip().lower() not in _FIELD_UNITS[field]
    ):
        return None
    digits = number.group(0)
    other_mark = "," if decimal == "." else "."
    if other_mark in digits:
        return None
    try:
        return normalize_decimal(digits.replace(",", "."), field=field)
    except ValueError:
        return None


def _candidate(key: str, value: str, line: int, decimal: str) -> Optional[Candidate]:
    folded = _BRACKET_PART.sub(" ", key)
    folded = re.sub(r"[\s_\-.]+", " ", folded).strip().lower()
    field = _PREAMBLE_KEYS.get(folded)
    if field is None:
        return None
    normalized = _normalized(field, value, decimal) if field in _FIELD_UNITS else None
    return Candidate(
        field=field,
        status="detected",
        value=_cited(value, PREAMBLE_LINE_CHARS),
        normalized=normalized,
        line=line,
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def _rounded(value: float) -> float:
    return float(f"{value:.6g}")


def _stats(values: list[float]) -> Optional[ColumnStats]:
    if not values:
        return None
    low, high = min(values), max(values)
    count = len(values)
    # Scaled so that finite values near the float maximum cannot overflow.
    scale = max(abs(low), abs(high)) or 1.0
    return ColumnStats(
        n=count,
        min=_rounded(low),
        max=_rounded(high),
        mean=_rounded(math.fsum(v / count for v in values)),
        rms=_rounded(
            scale * math.sqrt(math.fsum((v / scale) ** 2 for v in values) / count)
        ),
        count_at_min=values.count(low),
        count_at_max=values.count(high),
    )


def _is_date_text(token: str, decimal: str) -> bool:
    try:
        parse_time_tokens([token], decimal_mark=decimal)
    except TimeBaseRefused as refused:
        return refused.datetime_text
    return False


def _time_stats(
    tokens: list[str],
    decimal: str,
    first_line: int,
    time_unit: Optional[str],
    unit_status: Optional[ValueStatus],
) -> tuple[TimeStats, Optional[str]]:
    """The scanned rows' time statistics and, when the column's unit token
    spells a time unit exactly and its rows support one, the rate they
    derive (a unit read only with letter case ignored derives none)."""
    try:
        mantissa, exponent = parse_time_tokens(
            tokens, decimal_mark=decimal, first_line=first_line
        )
        assessment = assess_time_base(
            mantissa, exponent, time_unit=time_unit or "s", first_line=first_line
        )
    except TimeBaseRefused as refused:
        stats = TimeStats(
            time_unit=time_unit,
            time_unit_status=unit_status,
            rows=len(tokens),
            refused_code=refused.code,
            datetime_text=refused.datetime_text,
        )
        return stats, None
    stats = TimeStats(
        time_unit=time_unit,
        time_unit_status=unit_status,
        rows=assessment.row_count,
        evidence=assessment.evidence,
        regime=assessment.regime,
        resolution=assessment.resolution if time_unit else None,
        modal_steps=list(assessment.modal_steps),
        finding_codes=[finding.code for finding in assessment.findings],
    )
    return stats, assessment.proposed_rate_hz if unit_status == "detected" else None


def _index_list(indexes: list[int]) -> str:
    shown = ", ".join(str(i) for i in indexes[:10])
    return shown + (f" and {len(indexes) - 10} more" if len(indexes) > 10 else "")


# ---------------------------------------------------------------------------
# The result and its size budget
# ---------------------------------------------------------------------------


@dataclass
class _Draft:
    """Everything a result is built from, before the budget is applied."""

    status: InspectionStatus
    file: InspectedFile
    regions: list[Region]
    preamble_lines: list[PreambleLine]
    key_values: list[KeyValue]
    preamble_omitted: int
    dialect: Dialect
    columns_reading: DialectReading
    columns: list[InspectedColumn]
    columns_total: int
    columns_from: int
    candidates: list[Candidate]
    issues: list[Issue]
    confirmations: list[str]

    def result(self, page: int, kept: int, pairs: int) -> InspectionResult:
        """The result with *page* columns, *kept* preamble lines and *pairs*
        key-value pairs; a candidate read from a dropped pair is dropped with
        it."""
        shown = self.columns[:page]
        end = self.columns_from + len(shown)
        dropped_lines = {pair.line for pair in self.key_values[pairs:]}
        candidates = [c for c in self.candidates if c.line not in dropped_lines]
        omitted = (
            self.preamble_omitted
            + len(self.preamble_lines)
            - kept
            + len(self.key_values)
            - pairs
        )
        return InspectionResult(
            status=self.status,
            file=self.file,
            regions=self.regions,
            preamble=Preamble(
                lines=self.preamble_lines[:kept],
                key_values=self.key_values[:pairs],
                omitted_lines=omitted,
            ),
            dialect=self.dialect,
            columns_reading=self.columns_reading,
            columns=shown,
            columns_total=self.columns_total,
            columns_from=self.columns_from,
            next_columns_from=end if end < self.columns_total else None,
            candidates=candidates,
            issues=self.issues,
            required_confirmations=self.confirmations,
        )


def _size(result: InspectionResult) -> int:
    return len(result.model_dump_json().encode("utf-8"))


def _within_budget(draft: _Draft) -> InspectionResult:
    """Shrink the column page first (the caller pages on), then drop
    preamble lines and then key-value pairs from the end, counting them."""
    page = len(draft.columns)
    kept = len(draft.preamble_lines)
    pairs = len(draft.key_values)
    result = draft.result(page, kept, pairs)
    while _size(result) > OUTPUT_BUDGET_BYTES and page > 1:
        page -= 1
        result = draft.result(page, kept, pairs)
    while _size(result) > OUTPUT_BUDGET_BYTES and kept > 0:
        kept -= 1
        result = draft.result(page, kept, pairs)
    while _size(result) > OUTPUT_BUDGET_BYTES and pairs > 0:
        pairs -= 1
        result = draft.result(page, kept, pairs)
    return result


def _encoding(scan: TextScan) -> EncodingEvidence:
    encoding = scan.encoding_candidates[0]
    if scan.bom is not None:
        return EncodingEvidence(value=encoding, status="detected")
    if encoding == "cp1252":
        return EncodingEvidence(value=encoding, status="ambiguous")
    return EncodingEvidence(value=encoding, status="proposed")


def _reading_model(reading: _Reading, lines: tuple[ScannedLine, ...]) -> DialectReading:
    return DialectReading(
        delimiter=reading.delimiter,
        decimal=reading.decimal,
        fields=reading.fields,
        data_first_line=lines[reading.first].number,
        rows=len(reading.rows),
        numeric_ratio=round(reading.numeric_ratio, 4),
    )


def _empty_result(
    file: InspectedFile, issues: list[Issue], confirmations: list[str]
) -> InspectionResult:
    return InspectionResult(
        status="not_importable",
        file=file,
        regions=[],
        preamble=Preamble(lines=[], key_values=[], omitted_lines=0),
        dialect=Dialect(status="absent", proposal=None, candidates=[]),
        columns_reading=None,
        columns=[],
        columns_total=0,
        columns_from=0,
        next_columns_from=None,
        candidates=[],
        issues=issues,
        required_confirmations=confirmations,
    )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _file_issues(
    encoding: EncodingEvidence, ends_with_break: Optional[bool], companion: bool
) -> tuple[list[Issue], list[str]]:
    issues: list[Issue] = []
    confirmations: list[str] = []
    if encoding.status == "ambiguous":
        issues.append(
            Issue(
                code="encoding_needs_confirmation",
                message=(
                    "The file is not valid UTF-8. It decodes as Windows-1252, "
                    "but a successful decode does not establish the encoding "
                    "the export was written in, so the encoding needs "
                    "confirmation."
                ),
            )
        )
        confirmations.append(_CONFIRM_ENCODING)
    if ends_with_break is False:
        issues.append(
            Issue(
                code="truncated_last_line",
                message=(
                    "The file does not end with a line break. Validation "
                    "refuses a last line without one as a policy, because it "
                    "cannot tell a complete export from one cut short or still "
                    "being written; confirm at the source that the export is "
                    "complete."
                ),
            )
        )
    if companion:
        issues.append(
            Issue(
                code="companion_present",
                message=(
                    "A companion metadata file (this file's name without its "
                    "extension, followed by _metadata.json) lies next to it. The "
                    "import never reads "
                    "it, and the envelope report would draw its bearing "
                    "markers on every imported channel; an import needs it "
                    "acknowledged first."
                ),
            )
        )
        confirmations.append(_CONFIRM_COMPANION)
    return issues, confirmations


def _inspect(
    scan: TextScan,
    *,
    name: str,
    ends_with_break: Optional[bool],
    companion: bool,
    columns_from: int,
) -> InspectionResult:
    lines = scan.head_lines
    encoding = _encoding(scan)
    issues, confirmations = _file_issues(encoding, ends_with_break, companion)
    file = InspectedFile(
        name=_cited(name, LABEL_CHARS),
        size_bytes=scan.size_bytes,
        encoding=encoding,
        newline_census=scan.newline_census,
        ends_with_line_break=ends_with_break,
        scan=ScanSummary(
            bytes_read=scan.bytes_read,
            lines_scanned=len(lines),
            complete=scan.complete,
        ),
    )

    readings = _readings(lines)
    if not readings:
        issues.append(
            Issue(
                code="no_numeric_region",
                message=(
                    "The file is empty."
                    if scan.size_bytes == 0
                    else "No row of numbers was found in the scanned lines: the "
                    "file holds text only (for example a header without data "
                    "rows)."
                ),
            )
        )
        return _empty_result(file, issues, confirmations)

    best = readings[0]
    data_first = lines[best.first].number
    if len(readings) > 1:
        issues.append(
            Issue(
                code="ambiguous_dialect",
                message=(
                    f"{len(readings)} readings of the data rows parse equally "
                    f"well (listed under the dialect candidates), so the "
                    f"delimiter and decimal mark need confirmation."
                ),
                line=data_first,
            )
        )
        confirmations.append(_CONFIRM_DIALECT)
    dialect = Dialect(
        status="ambiguous" if len(readings) > 1 else "proposed",
        proposal=None if len(readings) > 1 else _reading_model(best, lines),
        candidates=[_reading_model(reading, lines) for reading in readings],
    )

    # The data run, and whether it goes on past the scanned head.
    reaches_head_end = best.last == len(lines) - 1
    continues = reaches_head_end and scan.head_truncated
    data_last = None if continues else lines[best.last].number

    # Units row and header: the non-blank lines just above the data.
    above = [line for line in lines[: best.first] if line.text.strip(" \t")]
    units_line: Optional[ScannedLine] = None
    header_line: Optional[ScannedLine] = None
    if above and _is_units_row(_split(above[-1].text, best.delimiter) or []):
        units_line = above.pop()
    if above:
        fields = _split(above[-1].text, best.delimiter) or []
        if any(f.strip() and _number(f, best.decimal) is None for f in fields):
            header_line = above.pop()
    preamble_source = above

    def row_fields(line: Optional[ScannedLine]) -> list[str]:
        if line is None:
            return []
        fields = [f.strip() for f in _split(line.text, best.delimiter) or []]
        if best.trailing_empty and len(fields) == best.fields + 1 and not fields[-1]:
            fields = fields[:-1]
        return fields

    header_fields = row_fields(header_line)
    units_fields = row_fields(units_line)

    regions: list[Region] = []
    if preamble_source:
        first, last = preamble_source[0].number, preamble_source[-1].number
        regions.append(
            Region(
                kind="preamble",
                first_line=first,
                last_line=last,
                lines=last - first + 1,
            )
        )
    for kind, line in (("header", header_line), ("units", units_line)):
        if line is not None:
            regions.append(
                Region(
                    kind=kind, first_line=line.number, last_line=line.number, lines=1
                )
            )
    regions.append(
        Region(
            kind="data",
            first_line=data_first,
            last_line=data_last,
            lines=None if data_last is None else data_last - data_first + 1,
        )
    )
    trailer_first: Optional[int] = None
    trailer_count = 0
    if not reaches_head_end:
        after = [line for line in lines[best.last + 1 :] if line.text.strip(" \t")]
        if after:
            trailer_first, trailer_count = after[0].number, len(after)
    elif scan.tail_lines:
        for text in reversed(scan.tail_lines):
            if not text.strip(" \t"):
                continue
            if best.matches(_row(text, best.delimiter, best.decimal)):
                break
            trailer_count += 1
    if trailer_count:
        regions.append(
            Region(
                kind="trailer",
                first_line=trailer_first,
                last_line=None,
                lines=trailer_count,
            )
        )

    # Field counts decide whether labels and unit tokens can be matched.
    labels: list[Optional[str]] = [None] * best.fields
    unit_tokens: list[Optional[str]] = [None] * best.fields
    unit_source: Optional[int] = None
    for kind, line, fields in (
        ("header", header_line, header_fields),
        ("units row", units_line, units_fields),
    ):
        if line is None:
            continue
        if len(fields) != best.fields:
            issues.append(
                Issue(
                    code="field_count_mismatch",
                    message=(
                        f"The {kind} on line {line.number} has {len(fields)} "
                        f"fields but the data rows from line {data_first} have "
                        f"{best.fields}, so its entries cannot be matched to "
                        f"columns and are not shown: map every channel by its "
                        f"column index."
                    ),
                    line=line.number,
                )
            )
        elif kind == "header":
            labels = [field or None for field in fields]
        else:
            unit_tokens = [field or None for field in fields]
            unit_source = line.number
    if header_line is not None and len(header_fields) == best.fields:
        filled = [label for label in labels if label]
        if len(filled) < best.fields:
            issues.append(
                Issue(
                    code="empty_label",
                    message="Some header fields are empty; those columns have no label.",
                    line=header_line.number,
                )
            )
        if len(set(filled)) < len(filled):
            issues.append(
                Issue(
                    code="duplicate_label",
                    message=(
                        "Two or more columns share a header label; refer to "
                        "columns by index."
                    ),
                    line=header_line.number,
                )
            )
        if units_line is None:
            for index, label in enumerate(labels):
                bracketed = _bracketed(label, _LABEL_UNIT) if label else None
                if bracketed is not None:
                    unit_tokens[index] = bracketed
                    unit_source = header_line.number
    if best.trailing_empty:
        issues.append(
            Issue(
                code="trailing_delimiter_column",
                message=(
                    "The data rows end with a delimiter, which adds an empty "
                    "last field; it is not a column and is never imported."
                ),
                line=data_first,
            )
        )

    # The preamble, each line shown once, and the values it states.
    candidates: list[Candidate] = []
    preamble_lines: list[PreambleLine] = []
    key_values: list[KeyValue] = []
    for line in preamble_source[:MAX_PREAMBLE_LINES]:
        pair = _key_value(line.text, best.delimiter)
        if pair is None:
            preamble_lines.append(
                PreambleLine(
                    line=line.number, text=_cited(line.text, PREAMBLE_LINE_CHARS)
                )
            )
            continue
        key, value = pair
        key_values.append(
            KeyValue(
                line=line.number,
                key=_cited(key, LABEL_CHARS),
                value=_cited(value, PREAMBLE_LINE_CHARS),
            )
        )
        stated = _candidate(key, value, line.number, best.decimal)
        if stated is not None:
            candidates.append(stated)

    # Unit issues name every column, not only this page's.
    by_kind: dict[str, list[int]] = {"unrecognized": [], "preview": []}
    for index, token in enumerate(unit_tokens):
        kind = _classify_unit(token)[2] if token else "absent"
        if kind in by_kind:
            by_kind[kind].append(index)
    if by_kind["unrecognized"]:
        issues.append(
            Issue(
                code="unit_token_unrecognized",
                message=(
                    f"The unit tokens of columns {_index_list(by_kind['unrecognized'])} "
                    f"are not units this version imports; they are reported as "
                    f"written and never mapped to a default."
                ),
                columns=by_kind["unrecognized"],
            )
        )
    if by_kind["preview"]:
        issues.append(
            Issue(
                code="unit_token_preview_only",
                message=(
                    f"Columns {_index_list(by_kind['preview'])} carry units this "
                    f"version shows but never imports (displacement, imperial "
                    f"velocity, speed, temperature or current)."
                ),
                columns=by_kind["preview"],
            )
        )

    # Every column is profiled (the scanned rows bound the work), so a rate a
    # time column derives is found whatever page is shown; the result carries
    # one page of the profiles.
    start = min(columns_from, best.fields)
    profiles: list[InspectedColumn] = []
    case_folded_time_units: list[int] = []
    for index in range(best.fields):
        tokens = [row[index] for row in best.rows]
        values = [
            v for v in (_number(t, best.decimal) for t in tokens) if v is not None
        ]
        token = unit_tokens[index]
        unit, role, kind = _classify_unit(token) if token else (None, None, "absent")
        label = labels[index]
        monotonic = len(values) == len(tokens) >= 3 and all(
            later > earlier for earlier, later in zip(values, values[1:])
        )
        time_like = (
            kind == "time"
            or bool(label and _TIME_LABEL.search(label))
            or (not values and _is_date_text(tokens[0], best.decimal))
        )
        time_stats: Optional[TimeStats] = None
        if time_like or monotonic:
            role = "time"
            # The vocabulary spelling the classification matched ("S" is read
            # as s); the token itself stays in unit_token as written.
            time_unit = _unit_text(token).lower() if token and kind == "time" else None
            unit_status: Optional[ValueStatus] = None
            if time_unit is not None:
                exact = token is not None and _unit_text(token) == time_unit
                unit_status = "detected" if exact else "ambiguous"
                if not exact:
                    case_folded_time_units.append(index)
            time_stats, rate = _time_stats(
                tokens, best.decimal, data_first, time_unit, unit_status
            )
            if rate is not None:
                candidates.append(
                    Candidate(
                        field="sampling_rate_hz",
                        status="derived",
                        normalized=rate,
                        column=index,
                    )
                )
        profiles.append(
            InspectedColumn(
                index=index,
                label=_cited(label, LABEL_CHARS) if label else None,
                unit_token=_cited(token, TOKEN_CHARS) if token else None,
                unit_line=unit_source if token else None,
                unit=unit,
                role_candidate=role,
                numeric_ratio=round(len(values) / len(tokens), 4),
                stats=_stats(values),
                time_stats=time_stats,
                head_tokens=[_cited(t, TOKEN_CHARS) for t in tokens[:HEAD_TOKENS]],
            )
        )

    columns = profiles[start : start + COLUMNS_PER_PAGE]
    if not any(c.field in ("sampling_rate_hz", "sample_interval") for c in candidates):
        candidates.append(Candidate(field="sampling_rate_hz", status="absent"))

    # The row count is exact only when every line of the file was scanned.
    status: InspectionStatus = "importable"
    if scan.complete and not scan.head_truncated and len(best.rows) < MIN_WAVEFORM_ROWS:
        status = "not_importable"
        issues.append(
            Issue(
                code="not_a_waveform",
                message=(
                    f"The data region holds {len(best.rows)} rows, fewer than "
                    f"the {MIN_WAVEFORM_ROWS}-sample default segment of the "
                    f"spectral tools, which this version requires of an "
                    f"imported signal; the file may hold a table of features "
                    f"rather than a waveform."
                ),
                line=data_first,
            )
        )
    else:
        confirmations[:0] = [_CONFIRM_CHANNELS, _CONFIRM_RATE, _CONFIRM_BINDING]
        if case_folded_time_units:
            confirmations.append(
                f"the time unit of columns {_index_list(case_folded_time_units)}, "
                f"whose unit tokens name a time unit only when letter case is "
                f"ignored"
            )

    return _within_budget(
        _Draft(
            status=status,
            file=file,
            regions=regions,
            preamble_lines=preamble_lines,
            key_values=key_values,
            preamble_omitted=max(0, len(preamble_source) - MAX_PREAMBLE_LINES),
            dialect=dialect,
            columns_reading=_reading_model(best, lines),
            columns=columns,
            columns_total=best.fields,
            columns_from=start,
            candidates=candidates,
            issues=issues,
            confirmations=confirmations,
        )
    )


def inspect_signal_file(path: str, *, columns_from: int = 0) -> InspectionResult:
    """Inspect a delimited-text export without writing anything.

    Args:
        path: The file, absolute or relative to the data directory, under the
            data directory or a folder listed in ``PMM_IMPORT_ROOTS``.
        columns_from: Index of the first column to profile (pages of at most
            :data:`COLUMNS_PER_PAGE`).

    Returns:
        The :class:`InspectionResult`.

    Raises:
        PathRefused: The path is unsafe, outside the import roots, not a
            regular ``.csv`` or ``.txt`` file, too large, or unreadable.
        TextScanRefused: Binary, compressed, UTF-16/UTF-32 or undecodable
            content, a line over the line cap, or rows too wide under every
            delimiter.
        ValueError: A negative *columns_from*, or a malformed
            ``PMM_IMPORT_ROOTS``.
    """
    if (
        isinstance(columns_from, bool)
        or not isinstance(columns_from, int)
        or columns_from < 0
    ):
        raise ValueError("columns_from must be an integer, 0 or more.")
    roots = config.get_import_roots()
    with open_import_file(
        path, roots, base=config.DATA_DIR, max_size=config.get_max_signal_size()
    ) as file:
        scan = scan_text(file)
        ends_with_break = (
            None if file.size == 0 else file.read(file.size - 1, 1) in (b"\n", b"\r")
        )
        companion = os.path.exists(
            file.path.with_name(f"{file.path.stem}_metadata.json")
        )
        name = file.path.name
    return _inspect(
        scan,
        name=name,
        ends_with_break=ends_with_break,
        companion=companion,
        columns_from=columns_from,
    )
