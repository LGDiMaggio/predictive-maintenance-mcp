"""
Typed results of the signal import service (Pydantic v2).

ISO 13374 Block 1 (data acquisition). The tool layer imports these models
from here; ``src/models.py`` stays free of package imports. Every string a
model takes from a file is a :class:`FileText` (escaped, capped and carried
under the key ``file_text``), so a reader can always tell file text from the
server's own words, and file text never speaks in the server's voice.

This module holds the inspection result, the draft a configuration starts
from, and the validation result. The configuration a caller sends is
normalized by ``attestation.normalize_configuration`` (plain data, refused
key by key); output models are built by the server only.
"""

from typing import Any, Literal, Optional

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
    "Prefill",
    "ImportDraft",
    "ValidationOutcome",
    "IssueEffect",
    "ValidationIssue",
    "Statement",
    "TimeColumnCheck",
    "ValidatedChannel",
    "ChannelPrecheck",
    "ValidationResult",
    "ImportStatus",
    "ImportedChannel",
    "ImportResult",
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


# ---------------------------------------------------------------------------
# Configuration draft
# ---------------------------------------------------------------------------


class Prefill(_Output):
    """A configured value a draft took from evidence, not from the user.

    A prefill is never a basis: the user still confirms it, citing the same
    evidence (the remaining confirmations say how).

    Attributes:
        field: The field path the value fills (``time_base``,
            ``channels[3].unit``, ``defaults.rpm``, ...).
        value: The value, as the configuration writes it.
        origin: One of ``attestation.ORIGINS``.
        line: The file line it was read from (``file_declared``).
        column: The time column it was derived from (``derived``).
        import_id: The earlier import it came from (``prior_import``).
        declaration_version: The point declaration it came from
            (``point_declaration``).
    """

    field: str
    value: str
    origin: str
    line: Optional[int] = None
    column: Optional[int] = None
    import_id: Optional[str] = None
    declaration_version: Optional[int] = None


class ImportDraft(_Output):
    """A configuration to complete: never a confirmation.

    Attributes:
        configuration: The draft (``attestation.CONFIGURATION_SCHEMA``); its
            attestation carries no basis, since only the user gives one.
        prefills: Every value the draft took from evidence, with its origin.
        remaining_confirmations: What the user still has to state or accept,
            in the server's words.
    """

    configuration: dict[str, Any]
    prefills: list[Prefill]
    remaining_confirmations: list[str]


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

#: ``ready``: every value has a basis its evidence supports and the whole
#: file parses; ``needs_confirmation``: the user has to state, accept or
#: choose something; ``invalid``: the file or the configuration is refused
#: as it stands.
ValidationOutcome = Literal["ready", "needs_confirmation", "invalid"]

#: What an issue does to the outcome.
IssueEffect = Literal["invalid", "needs_confirmation", "notice"]


class ValidationIssue(_Output):
    """One finding of a validation, in the server's words.

    Attributes:
        code: One of ``import_config.ISSUE_CODES``.
        effect: ``invalid`` and ``needs_confirmation`` keep the import from
            ``ready``; a ``notice`` informs and is recorded.
        message: What was found and what to do; file text never enters it.
        field: The configuration field it concerns, when there is one.
        line: The physical file line it concerns, when there is one.
        column: The physical column it concerns, when there is one.
        token: The file's own text the issue is about (a printed value or
            row as written), escaped and capped, when it names one; the
            message refers to it and never repeats it.
    """

    code: str
    effect: IssueEffect
    message: str
    field: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    token: Optional[FileText] = None


class Statement(_Output):
    """One server-authored confirmation statement per engineering value.

    The prose names a channel by its index and its 1-based position and
    states the consequence; the column's first tokens, and its label only
    when the header's field count matches the data, travel in their own
    file-text fields, never inside the prose.

    Attributes:
        field: The configuration field the statement confirms.
        column: The channel's physical column, when it concerns one.
        text: The statement, in the server's words.
        label: The column's header label as file text.
        head_tokens: The column's first data tokens as file text.
        origin: Where the value comes from (``attestation.ORIGINS``).
        basis: The user's act behind it (``attestation.BASES``).
        caller_asserted: The basis arrived as an MCP tool argument: it
            records what the caller claimed, never who the caller is.
    """

    field: str
    column: Optional[int] = None
    text: str
    label: Optional[FileText] = None
    head_tokens: list[FileText] = Field(default_factory=list)
    origin: Optional[str] = None
    basis: Optional[str] = None
    caller_asserted: bool = False


class TimeColumnCheck(_Output):
    """The assessment of one time column over the whole data region.

    Attributes:
        column: Physical column index.
        time_unit: The declared unit of its stamps.
        outcome: ``accepted``, ``needs_confirmation`` or ``invalid``.
        evidence: ``full``, ``partial`` or ``duration_only``.
        regime: How its stamps are printed.
        resolution: One tick of its print, in *time_unit*.
        fitted_rate_hz: The least-squares rate, when one was fitted.
        proposed_rate_hz: The rate it supports proposing, when it does.
        finding_codes: Its findings (issue codes).
        statements: Its findings in plain words, with their statistics.
    """

    column: int
    time_unit: str
    outcome: str
    evidence: str
    regime: str
    resolution: str
    fitted_rate_hz: Optional[str] = None
    proposed_rate_hz: Optional[str] = None
    finding_codes: list[str] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)


class ValidatedChannel(_Output):
    """One selected vibration channel, as the whole-file parse found it.

    Attributes:
        column: Physical column index.
        position: Its 1-based position in words ("the 4th field from the
            left").
        role: ``vibration``.
        quantity: ``acceleration`` or ``velocity``.
        unit: The unit it is stored in.
        scaling_divisor: Source units per stored unit, as an exact decimal
            string, for an electrical column.
        sample_count: Samples in the data region.
        samples_digest: SHA-256 of the stored samples (little-endian
            float64), the value replay must reproduce.
        label: Its header label as file text (None when the header's field
            count differs from the data's).
        unit_token: Its unit token as file text.
        head_tokens: Its first data tokens as file text.
        mean: Mean of the stored samples (electrical columns, for the DC
            policy).
        mean_to_std: Mean divided by standard deviation (electrical columns).
        measurement_id: The measurement it binds, when it has a binding.
    """

    column: int
    position: str
    role: str
    quantity: str
    unit: str
    scaling_divisor: Optional[str] = None
    sample_count: int
    samples_digest: str
    label: Optional[FileText] = None
    unit_token: Optional[FileText] = None
    head_tokens: list[FileText] = Field(default_factory=list)
    mean: Optional[float] = None
    mean_to_std: Optional[float] = None
    measurement_id: Optional[str] = None


class ChannelPrecheck(_Output):
    """The read-only ledger pre-check of one bound channel.

    Attributes:
        column: Physical column index.
        measurement_id: The measurement the channel binds.
        asset_id: The asset it binds.
        asset_status: ``known`` (the ledger holds events of the asset) or
            ``new``.
        point_status: ``declared`` or ``undeclared``.
        point_declaration_version: The point's current declaration version.
        measurement_status: ``new``, ``already_recorded`` (an identical
            declaration is recorded) or ``supersedes`` (it would replace the
            recorded declaration).
        changed: The declaration keys a supersession changes.
        recorded_asset_id: The asset the measurement is recorded under, when
            it is recorded.
        baseline_member: The measurement is a member of its point's active
            baseline.
        restores_superseded: The new interpretation is one an earlier,
            later superseded, declaration of the measurement used.
        data_range_overlaps: Whether the new data line range overlaps the
            recorded one (None when nothing comparable is recorded).
        column_swap: The measurement of another column of the same bytes
            recorded at the same point and acquisition slot.
    """

    column: int
    measurement_id: str
    asset_id: str
    asset_status: Literal["known", "new"]
    point_status: Literal["declared", "undeclared"]
    point_declaration_version: Optional[int] = None
    measurement_status: Literal["new", "already_recorded", "supersedes"]
    changed: list[str] = Field(default_factory=list)
    recorded_asset_id: Optional[str] = None
    baseline_member: bool = False
    restores_superseded: bool = False
    data_range_overlaps: Optional[bool] = None
    column_swap: Optional[str] = None


class ValidationResult(_Output):
    """What a validation decided (nothing is ever written).

    Attributes:
        outcome: ``ready``, ``needs_confirmation`` or ``invalid``.
        file: The file name as file text.
        source_sha256: SHA-256 of every byte of the file (None when the
            parse stopped before the end).
        size_bytes: The file size.
        first_line: Physical line of the first data row.
        last_line: Physical line of the last data row.
        row_count: Data rows (samples per channel).
        sampling_rate_hz: The configured sampling rate.
        time_base_source: Where it comes from (``time_column``, ``stated``
            or ``file_declared``).
        time_columns: Every assessed time column.
        channels: Every selected channel the parse decoded.
        statements: One statement per engineering value.
        issues: Every finding, each with its effect on the outcome.
        required_confirmations: What the user still has to confirm, in the
            server's words (empty when ``ready``): entry *i* is the message of
            the *i*-th issue whose effect is ``needs_confirmation``, in issue
            order, and no two such issues share a message. That issue carries
            the field, the line and, for a value the file prints, the text as
            printed (``token``) its message refers to.
        ledger_precheck: The read-only pre-check of every bound channel.
        confirmation_channel: Through which channel the configuration came.
        caller_asserted: Its bases record what the caller claimed.
        import_id: The interpretation's identity (None when the parse
            stopped before the end).
        import_digest: What an import must present; set only when
            ``ready``.
    """

    outcome: ValidationOutcome
    file: FileText
    source_sha256: Optional[str] = None
    size_bytes: int
    first_line: Optional[int] = None
    last_line: Optional[int] = None
    row_count: Optional[int] = None
    sampling_rate_hz: str
    time_base_source: str
    time_columns: list[TimeColumnCheck] = Field(default_factory=list)
    channels: list[ValidatedChannel] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    required_confirmations: list[str] = Field(default_factory=list)
    ledger_precheck: list[ChannelPrecheck] = Field(default_factory=list)
    confirmation_channel: str
    caller_asserted: bool
    import_id: Optional[str] = None
    import_digest: Optional[str] = None


#: What an import call did: a new import, a repeat of a recorded one, or a
#: re-materialization from the import record alone.
ImportStatus = Literal["imported", "already_imported", "rematerialized"]


class ImportedChannel(_Output):
    """One channel of an import, as it ended.

    Attributes:
        column: Physical column index.
        position: Its 1-based position in words.
        signal_id: The id it is stored under (None when it was not stored).
        materialized: Whether its samples are in the signal store after
            this call.
        samples_digest: SHA-256 of its stored samples (little-endian
            float64).
        sample_count: Its number of samples.
        measurement_id: The measurement it binds (None without a binding).
        asset_id: The asset it is recorded under (None without a binding).
        ledger_status: ``recorded``, ``already_recorded``, ``superseded`` or
            ``not_recorded``; None without a binding.
        snapshot_status: ``complete``, ``partial``, ``failed`` or
            ``skipped``; None without a binding.
        changed: The declaration keys a supersession changed.
        reattributed_from: The asset the measurement was recorded under
            before, when the import moved it.
        reason: Why the channel is not stored or not recorded, in the
            server's words.
    """

    column: int
    position: str
    signal_id: Optional[str] = None
    materialized: bool
    samples_digest: str
    sample_count: int
    measurement_id: Optional[str] = None
    asset_id: Optional[str] = None
    ledger_status: Optional[str] = None
    snapshot_status: Optional[str] = None
    changed: list[str] = Field(default_factory=list)
    reattributed_from: Optional[str] = None
    reason: Optional[str] = None


class ImportResult(_Output):
    """What an import or a re-materialization did.

    Attributes:
        status: ``imported``, ``already_imported`` or ``rematerialized``.
        import_id: The interpretation's identity.
        file: The file name as file text.
        record_created: Whether this call wrote the import record's first
            line.
        caller_asserted: The bases of this import record what the caller
            claimed (MCP tool arguments), not who the caller is.
        channels: Every channel of the import, in column order.
        notices: What the caller should know, in the server's words.
    """

    status: ImportStatus
    import_id: str
    file: FileText
    record_created: bool
    caller_asserted: bool
    channels: list[ImportedChannel] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
