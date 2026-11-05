"""
Typed results of the signal import service (Pydantic v2).

ISO 13374 Block 1 (data acquisition). The tool layer imports these models
from here; ``src/models.py`` stays free of package imports. Every string a
model takes from a file is a :class:`FileText` (escaped, capped and carried
under the key ``file_text``), so a reader can always tell file text from the
server's own words, and file text never speaks in the server's voice.

This module holds the inspection result. Input models (the configuration a
caller sends) set ``extra="forbid"`` so that a misspelled key is refused
instead of silently dropped; output models are built by the server only.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..file_text import FileText
from ..signal_acquisition.import_config import ROLES, VALUE_STATUSES

__all__ = [
    "FileText",
    "ValueStatus",
    "Role",
    "InspectionStatus",
    "ScanSummary",
    "EncodingEvidence",
    "InspectedFile",
    "Region",
    "PreambleLine",
    "KeyValue",
    "Preamble",
    "DialectReading",
    "Dialect",
    "ColumnStats",
    "TimeStats",
    "InspectedColumn",
    "Candidate",
    "Issue",
    "InspectionResult",
]

#: How an inspected value was obtained (see ``import_config.VALUE_STATUSES``).
ValueStatus = Literal[VALUE_STATUSES]  # type: ignore[valid-type]

#: A column role (see ``import_config.ROLES``).
Role = Literal[ROLES]  # type: ignore[valid-type]

#: Whether an inspected file holds a waveform an import can read.
InspectionStatus = Literal["importable", "not_importable"]


class _Output(BaseModel):
    """Base of every output model: frozen, and a stray key is a bug."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScanSummary(_Output):
    """How much of the file the inspection read.

    Attributes:
        bytes_read: Bytes read through the verified descriptor.
        lines_scanned: Head lines examined (at most 300).
        complete: Every byte of the file was read.
    """

    bytes_read: int
    lines_scanned: int
    complete: bool


class EncodingEvidence(_Output):
    """The text encoding: ``detected`` from a UTF-8 byte-order mark,
    ``proposed`` when strict UTF-8 decodes every byte read, ``ambiguous`` when
    only Windows-1252 does (a successful decode does not establish the
    encoding an export was written in, so it needs confirmation)."""

    value: str
    status: ValueStatus


class InspectedFile(_Output):
    """The file as found.

    Attributes:
        name: The file's name, as file text.
        size_bytes: Size at open time (no full-file digest is computed).
        encoding: The encoding evidence.
        newline_census: Line breaks in the bytes read, by kind (``lf``,
            ``crlf``, ``cr``).
        ends_with_line_break: The last byte is a line break (None for an
            empty file).
        scan: How much was read.
    """

    name: FileText
    size_bytes: int
    encoding: EncodingEvidence
    newline_census: dict[str, int]
    ends_with_line_break: Optional[bool]
    scan: ScanSummary


class Region(_Output):
    """One region of the file, by physical line (1-based).

    Attributes:
        kind: ``preamble``, ``header``, ``units``, ``data`` or ``trailer``.
        first_line: First physical line (None when unknown: a trailer found
            only in the tail, past lines that were not read).
        last_line: Last physical line (None when the region continues past
            the scanned head, or is unknown).
        lines: Lines in the region when known.
    """

    kind: Literal["preamble", "header", "units", "data", "trailer"]
    first_line: Optional[int]
    last_line: Optional[int]
    lines: Optional[int]


class PreambleLine(_Output):
    """A preamble line that is not a key-value pair."""

    line: int
    text: FileText


class KeyValue(_Output):
    """A preamble line read as a key and a value."""

    line: int
    key: FileText
    value: FileText


class Preamble(_Output):
    """The lines before the header, each shown once.

    Attributes:
        lines: Lines that are not key-value pairs.
        key_values: Lines read as a key and a value.
        omitted_lines: Preamble lines not shown (over the cap of 40 lines, or
            dropped to keep the result within its size budget).
    """

    lines: list[PreambleLine]
    key_values: list[KeyValue]
    omitted_lines: int


class DialectReading(_Output):
    """One reading of the data rows.

    Attributes:
        delimiter: ``,``, ``;``, ``tab`` or ``whitespace``.
        decimal: ``.`` or ``,``.
        fields: Fields per data row under this reading.
        data_first_line: First line of its longest run of consistent rows.
        rows: Rows in that run among the scanned lines.
        numeric_ratio: Share of the run's fields that are numbers.
    """

    delimiter: str
    decimal: str
    fields: int
    data_first_line: int
    rows: int
    numeric_ratio: float


class Dialect(_Output):
    """The dialect proposal: ``proposed`` when one reading scores best,
    ``ambiguous`` when several tie (each listed in ``candidates``),
    ``absent`` when none finds a row of numbers."""

    status: ValueStatus
    proposal: Optional[DialectReading]
    candidates: list[DialectReading]


class ColumnStats(_Output):
    """Statistics over the column's numbers among the scanned data rows,
    rounded to six significant digits."""

    n: int
    min: float
    max: float
    mean: float
    rms: float
    count_at_min: int
    count_at_max: int


class TimeStats(_Output):
    """The time-base rules applied to a time candidate's scanned rows only
    (validation assesses the whole column).

    Attributes:
        time_unit: The unit read from the column's unit token, or None when
            it has none (the statistics are then in ticks of the print and
            no rate is derived). The token itself, as written, is the
            column's ``unit_token``.
        time_unit_status: ``detected`` when the token spells the unit
            exactly; ``ambiguous`` when it names a time unit only with letter
            case ignored (``S``, ``MS``): no rate is derived then, and the
            unit is listed among the required confirmations. None without a
            time unit.
        rows: Stamps assessed.
        evidence: ``full``, ``partial`` or ``duration_only`` (None when
            refused).
        regime: The print regime (None when refused).
        resolution: One tick of the print grid, in *time_unit*.
        modal_steps: The modal pair of tick steps.
        finding_codes: Codes of the findings on these rows.
        refused_code: ``time_grammar`` or ``region_too_short`` when the rows
            cannot be assessed.
        datetime_text: The stamps are dates or times of day, which this
            version does not assess.
    """

    time_unit: Optional[str]
    time_unit_status: Optional[ValueStatus] = None
    rows: int
    evidence: Optional[str] = None
    regime: Optional[str] = None
    resolution: Optional[str] = None
    modal_steps: Optional[list[int]] = None
    finding_codes: list[str] = Field(default_factory=list)
    refused_code: Optional[str] = None
    datetime_text: bool = False


class InspectedColumn(_Output):
    """Evidence for one column, by physical 0-based index (labels are display
    text only and never identify a column).

    Attributes:
        index: Physical column index.
        label: The header label, as file text (None without a header, or
            when the header's field count differs from the data's).
        unit_token: The unit token from the units row or the label's
            brackets, verbatim, as file text (detected evidence, never a
            confirmed unit).
        unit_line: Physical line the unit token was read from.
        unit: The importable unit the token maps to (``g``, ``m/s2``,
            ``mm/s``, ``m/s``), None otherwise.
        role_candidate: The role the evidence suggests (None without any).
        numeric_ratio: Share of the scanned rows whose field is a number.
        stats: Statistics of the numbers (None when there are none).
        time_stats: Time-base statistics, for a time candidate.
        head_tokens: The first three fields of the column, as file text.
    """

    index: int
    label: Optional[FileText]
    unit_token: Optional[FileText]
    unit_line: Optional[int]
    unit: Optional[str]
    role_candidate: Optional[Role]
    numeric_ratio: float
    stats: Optional[ColumnStats]
    time_stats: Optional[TimeStats]
    head_tokens: list[FileText]


class Candidate(_Output):
    """An engineering value the file states or the data suggests.

    Attributes:
        field: ``sampling_rate_hz``, ``sample_interval``, ``sensitivity``,
            ``acquired_at``, ``rpm`` or ``unit``.
        status: ``detected`` (stated in the file), ``derived`` (computed
            from the data), or ``absent`` (for ``sampling_rate_hz`` when
            neither the file nor a time column gives a rate).
        value: The value as the file prints it (``detected``).
        normalized: The value as a canonical decimal string, only when it is
            a number in the data's decimal mark followed by nothing or by the
            field's own unit (``Hz``, ``s``, ``rpm``, a sensitivity unit);
            ``25.6 kHz`` stays unnormalized.
        line: Physical line it was read from.
        column: Column it was derived from.
    """

    field: str
    status: ValueStatus
    value: Optional[FileText] = None
    normalized: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


class Issue(_Output):
    """One finding, in the server's words; file text never enters *message*.

    Attributes:
        code: One of ``import_config.ISSUE_CODES``.
        message: What was found and what it means, in the server's words.
        line: The physical line it concerns, when there is one.
        column: The column it concerns, when there is exactly one.
        columns: Every column it concerns, when it names several.
    """

    code: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    columns: list[int] = Field(default_factory=list)


class InspectionResult(_Output):
    """What a read-only inspection found (the file is never written).

    Attributes:
        status: ``importable`` when a numeric data region was found that can
            hold a waveform, else ``not_importable``.
        file: The file as found.
        regions: Preamble, header, units row, data and trailer.
        preamble: Preamble lines and key-value pairs (empty when no numeric
            data region was found).
        dialect: The dialect proposal.
        columns_reading: The reading the regions and columns were profiled
            under: the proposal, or, when the dialect is ambiguous, the first
            candidate (the others may split the rows differently). None when
            no numeric data region was found.
        columns: A page of columns, from *columns_from*.
        columns_total: Columns in the data rows.
        columns_from: Index of the first column on this page.
        next_columns_from: Index to pass for the next page (None when this
            page reaches the last column).
        candidates: Engineering values found.
        issues: Findings.
        required_confirmations: What an import of this file would need the
            user to confirm, in the server's words.
    """

    status: InspectionStatus
    file: InspectedFile
    regions: list[Region]
    preamble: Preamble
    dialect: Dialect
    columns_reading: Optional[DialectReading]
    columns: list[InspectedColumn]
    columns_total: int
    columns_from: int
    next_columns_from: Optional[int]
    candidates: list[Candidate]
    issues: list[Issue]
    required_confirmations: list[str]
