"""
Evidence a validation reads from the bounded scan.

Read under the configured dialect: the header and units rows (the
configured ones, else the ones the inspection rule finds above the
data), recognized preamble values and the first data rows, with the
rules that read a column's unit token and time-likeness from them.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ...signal_acquisition.text_scan import TextScan
from ..inspection import (
    _ANY_NUMBER,
    _FIELD_UNITS,
    _LABEL_UNIT,
    _TIME_LABEL,
    _bracketed,
    _candidate,
    _classify_unit,
    _is_date_text,
    _is_units_row,
    _key_value,
    _number,
    _readings,
    _split,
)

#: Data rows of the scanned head used for first tokens and time candidates.
_HEAD_ROWS = 300


@dataclass
class _Preamble:
    """One recognized preamble value: its field, raw value and line."""

    field: str
    key: str
    raw: str
    line: int
    normalized: Optional[str]
    printed: Optional[str]  # the number exactly as printed, decimal point


@dataclass
class _Evidence:
    header: Optional[list[str]]
    units: Optional[list[str]]
    preamble: list[_Preamble]
    head_rows: list[list[str]]
    unique_reading: Optional[tuple[str, str, int]]
    encoding: str
    #: The lines *header* and *units* come from: the configured rows, else
    #: the rows the inspection rule finds just above the data region, so a
    #: units row the configuration leaves out is still evidence.
    header_line: Optional[int] = None
    units_line: Optional[int] = None
    #: ``(header_line, units_line)`` as the inspection rule finds them above
    #: the configured data region (R9 compares the configured rows with it).
    scanned_structure: tuple[Optional[int], Optional[int]] = (None, None)
    #: The units row the inspection rule finds when the configuration names
    #: another line for it: still evidence, so a misassigned units line
    #: cannot hide it.
    extra_units: Optional[list[str]] = None
    extra_units_line: Optional[int] = None
    #: The header the inspection rule finds when the configuration names
    #: another line for it: its bracketed unit tokens are still evidence.
    extra_header: Optional[list[str]] = None
    extra_header_line: Optional[int] = None
    #: The raw text of each header or units row read, by line.
    row_text: dict[int, str] = field(default_factory=dict)
    #: ``(kind, line, fields, data fields)`` of each header or units row
    #: whose unit tokens cannot be matched to the data's fields.
    lost: list[tuple[str, int, int, int]] = field(default_factory=list)


def _printed_number(field_name: str, value: str, decimal_mark: str) -> Optional[str]:
    """The number of a preamble value as printed (decimal point), when it is
    a number in the data's decimal mark followed by nothing or the field's
    own unit; the same rule inspection normalizes by."""
    text = value.strip()
    number = _ANY_NUMBER.match(text)
    if number is None or field_name not in _FIELD_UNITS:
        return None
    if text[number.end() :].strip().lower() not in _FIELD_UNITS[field_name]:
        return None
    digits = number.group(0)
    if ("," if decimal_mark == "." else ".") in digits:
        return None
    return digits.replace(",", ".")


def _evidence(scan: TextScan, interpretation: Mapping[str, Any]) -> _Evidence:
    dialect = interpretation["dialect"]
    delimiter, decimal_mark = dialect["delimiter"], dialect["decimal"]
    lines = {line.number: line.text for line in scan.head_lines}
    first = interpretation["data_lines"]["first"]

    def fields_of(number: Optional[int]) -> Optional[list[str]]:
        if number is None or number not in lines:
            return None
        split = _split(lines[number], delimiter)
        return None if split is None else [f.strip() for f in split]

    # The inspection rule: a units row, then a header, just above the data
    # (R9 compares the configured rows with these); for evidence, the same
    # rule above any data rows a trimmed range leaves out.
    scanned_header, scanned_units = _rows_above(
        scan, first, delimiter, decimal_mark, skip_data=False
    )
    found_header, found_units = _rows_above(
        scan, first, delimiter, decimal_mark, skip_data=True
    )
    header_line = (
        dialect["header_line"] if dialect["header_line"] is not None else found_header
    )
    units_line = (
        dialect["units_line"] if dialect["units_line"] is not None else found_units
    )

    # Every other line above the data region is read for printed values: a
    # header or units line the configuration names elsewhere hides none.
    structure = {found_header, found_units}
    preamble: list[_Preamble] = []
    for number in sorted(n for n in lines if n < first and n not in structure):
        pair = _key_value(lines[number], delimiter)
        if pair is None:
            continue
        candidate = _candidate(pair[0], pair[1], number, decimal_mark)
        if candidate is None:
            continue
        preamble.append(
            _Preamble(
                field=candidate.field,
                key=pair[0],
                raw=pair[1],
                line=number,
                normalized=candidate.normalized,
                printed=_printed_number(candidate.field, pair[1], decimal_mark),
            )
        )
    head_rows: list[list[str]] = []
    for number in sorted(n for n in lines if n >= first):
        if len(head_rows) >= _HEAD_ROWS:
            break
        last = interpretation["data_lines"]["last"]
        if last is not None and number > last:
            break
        split = _split(lines[number], delimiter)
        if split is None or not lines[number].strip(" \t"):
            break
        head_rows.append([f.strip(" ") for f in split])
    readings = _readings(scan.head_lines)
    unique = None
    if len(readings) == 1:
        best = readings[0]
        unique = (best.delimiter, best.decimal, scan.head_lines[best.first].number)
    header = fields_of(header_line)
    units = fields_of(units_line)
    extra_line = (
        found_units
        if dialect["units_line"] is not None and found_units != units_line
        else None
    )
    extra = fields_of(extra_line)
    extra_header_line = (
        found_header
        if dialect["header_line"] is not None and found_header != header_line
        else None
    )
    extra_header = fields_of(extra_header_line)
    return _Evidence(
        header=header,
        units=units,
        preamble=preamble,
        head_rows=head_rows,
        unique_reading=unique,
        encoding=scan.encoding_candidates[0],
        header_line=header_line if header is not None else None,
        units_line=units_line if units is not None else None,
        scanned_structure=(scanned_header, scanned_units),
        extra_units=extra,
        extra_units_line=extra_line if extra is not None else None,
        extra_header=extra_header,
        extra_header_line=extra_header_line if extra_header is not None else None,
        row_text={
            number: lines[number]
            for number in (header_line, units_line, extra_line, extra_header_line)
            if number is not None and number in lines
        },
    )


def _rows_above(
    scan: TextScan, first: int, delimiter: str, decimal_mark: str, *, skip_data: bool
) -> tuple[Optional[int], Optional[int]]:
    """``(header line, units line)`` by the inspection rule: a units row,
    then a header, just above the data region; with *skip_data*, above the
    numeric rows just before it too (data rows a trimmed range leaves out)."""
    above = [
        line
        for line in scan.head_lines
        if line.number < first and line.text.strip(" \t")
    ]
    if skip_data:
        while above:
            fields = [f for f in (_split(above[-1].text, delimiter) or []) if f.strip()]
            if not fields or any(_number(f, decimal_mark) is None for f in fields):
                break
            above.pop()
    units: Optional[int] = None
    header: Optional[int] = None
    if above and _is_units_row(_split(above[-1].text, delimiter) or []):
        units = above.pop().number
    if above:
        fields = _split(above[-1].text, delimiter) or []
        if any(f.strip() and _number(f, decimal_mark) is None for f in fields):
            header = above.pop().number
    return header, units


def _time_like(evidence: _Evidence, column: int, decimal_mark: str) -> bool:
    """Whether a column looks like time, by the inspection rule: a time unit
    token, a time label, increasing head values, or date text."""
    label = (
        evidence.header[column]
        if evidence.header and column < len(evidence.header)
        else ""
    )
    token = _unit_token(evidence, column)
    tokens = [row[column] for row in evidence.head_rows if column < len(row)]
    values = [_number(t, decimal_mark) for t in tokens]
    numeric = [v for v in values if v is not None]
    monotonic = len(numeric) == len(tokens) >= 3 and all(
        later > earlier for earlier, later in zip(numeric, numeric[1:])
    )
    return (
        (token is not None and _classify_unit(token)[2] == "time")
        or bool(label and _TIME_LABEL.search(label))
        or monotonic
        or (bool(tokens) and not numeric and _is_date_text(tokens[0], decimal_mark))
    )


def _aligned(
    row: Optional[list[str]], evidence: _Evidence, field_count: int
) -> Optional[list[str]]:
    """A header or units row matched to the data's fields: one field short of
    data rows that all end with a delimiter, it gains the empty last field
    (the inspection rule for a trailing delimiter); longer than the data
    rows only by empty trailing fields, it loses them (a trailing delimiter
    on the row itself)."""
    if row is None:
        return row
    if len(row) > field_count and not any(row[field_count:]):
        return row[:field_count]
    if len(row) != field_count - 1:
        return row
    trailing = bool(evidence.head_rows) and all(
        bool(cells) and cells[-1] == "" for cells in evidence.head_rows
    )
    return [*row, ""] if trailing else row


def _unit_token(evidence: _Evidence, column: int) -> Optional[str]:
    """The column's unit token: its units-row field, else the bracketed unit
    of its header label (the rule inspection reports)."""
    if evidence.units is not None:
        if column < len(evidence.units) and evidence.units[column]:
            return evidence.units[column]
        return None
    if evidence.header is not None and column < len(evidence.header):
        return _bracketed(evidence.header[column], _LABEL_UNIT)
    return None


def _unit_tokens(evidence: _Evidence, column: int) -> list[tuple[str, Optional[int]]]:
    """Every unit token the file prints for a column, with its line: the
    column's token (:func:`_unit_token`), and the field of the units row the
    inspection rule finds when the configuration names another line."""
    found: list[tuple[str, Optional[int]]] = []
    token = _unit_token(evidence, column)
    if token is not None:
        found.append((token, _token_line(evidence)))
    extra = evidence.extra_units
    if extra is not None and column < len(extra) and extra[column]:
        found.append((extra[column], evidence.extra_units_line))
    header = evidence.extra_header
    if header is not None and column < len(header):
        bracketed = _bracketed(header[column], _LABEL_UNIT)
        if bracketed:
            found.append((bracketed, evidence.extra_header_line))
    return found


def _token_line(evidence: _Evidence) -> Optional[int]:
    if evidence.units is not None:
        return evidence.units_line
    if evidence.header is not None:
        return evidence.header_line
    return None


def _printed_unit(found: _Preamble) -> Optional[str]:
    """The unit a preamble value is printed in: the text after its number,
    else the bracketed unit of its key (``Sensitivity [mV/g]: 100``)."""
    text = found.raw.strip()
    number = _ANY_NUMBER.match(text)
    rest = text[number.end() :].strip() if number is not None else ""
    if rest:
        return rest
    return _bracketed(found.key, _LABEL_UNIT)
