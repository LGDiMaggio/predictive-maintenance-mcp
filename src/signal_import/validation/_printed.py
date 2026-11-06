"""
Printed declarations an import validation compares with the configuration.

A time column's unit token, every printed sensitivity and preamble unit,
and header or units rows whose tokens cannot be matched to the data (R11).
A declaration the file ties to one channel is compared with it, and a
contradiction is refused unless an override names a reason. A declaration
that cannot be read, or that the file does not tie to one channel, is never
taken as agreeing: it needs the user, and only the override of the field it
concerns records the user's reason.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

from decimal import Decimal
from typing import Any, Literal, Mapping, Optional

from ...signal_acquisition.import_config import (
    SENSITIVITY_UNITS,
    TIME_UNITS,
    scaling_divisor,
)
from .. import statements as st
from ..inspection import _classify_unit, _unit_text
from ._common import _ASK, _Run, _decimal, _printed
from ._evidence import _Evidence, _Preamble, _printed_unit, _unit_tokens

_Agreement = Literal["agree", "differ", "unknown"]

#: The quantity each physical unit of a sensitivity measures.
_QUANTITY = {
    "g": "acceleration",
    "m/s2": "acceleration",
    "mm/s": "velocity",
    "m/s": "velocity",
}


def _check_printed(run: _Run, evidence: _Evidence, field_count: int) -> None:
    """Every printed declaration this module compares."""
    _check_time_units(run, evidence)
    _check_sensitivities(run, evidence, field_count)
    _check_preamble_units(run, evidence, field_count)
    _check_lost_rows(run, evidence)


_RECORDED = "; the override with its reason is recorded."


def _check_time_units(run: _Run, evidence: _Evidence) -> None:
    """A time column's printed unit token against its configured time unit:
    seconds are never read as milliseconds."""
    for entry in run.interpretation["time_columns"]:
        column, unit = entry["column"], entry["time_unit"]
        path = f"time_columns[{column}].time_unit"
        override = run.override(path)
        where = f"The time column at {st.channel_words(column)}"
        for token, line in _unit_tokens(evidence, column):
            printed = _unit_text(token)
            if printed == unit:
                continue
            if _classify_unit(token)[2] == "time" and printed in TIME_UNITS:
                run.issue(
                    "contradicts_file_declaration",
                    "notice" if override is not None else "invalid",
                    f"{where}: its unit token on line {line} is {printed}, not the "
                    f"configured time unit {unit}"
                    + (
                        _RECORDED
                        if override is not None
                        else f". Correct the configuration, or name an override "
                        f"with a reason for {path!r} if the file is wrong."
                    ),
                    field_path=path,
                    line=line,
                    column=column,
                )
                continue
            run.issue(
                "printed_value_unverified",
                "notice" if override is not None else "needs_confirmation",
                f"{where}: its unit token on line {line} (its text as printed is "
                f"in this issue's token) is not a time unit this version reads "
                f"exactly ({', '.join(TIME_UNITS)}), so the configured time unit {unit} "
                f"cannot be checked against it"
                + (
                    _RECORDED
                    if override is not None
                    else f": {_ASK} to confirm the time unit, and name an override "
                    f"with a reason for {path!r}."
                ),
                field_path=path,
                line=line,
                column=column,
                token=_printed(token),
            )


def _sensitivity_reading(found: _Preamble) -> Optional[tuple[Decimal, Optional[str]]]:
    """The printed sensitivity's value and its unit (None when it prints
    none), or None when it cannot be read as a number with a sensitivity
    unit."""
    value = _decimal(found.normalized)
    if value is None:
        return None
    printed_unit = _printed_unit(found)
    if not printed_unit:
        return value, None
    units = {unit.lower(): unit for unit in SENSITIVITY_UNITS}
    unit = units.get(printed_unit.lower())
    return None if unit is None else (value, unit)


def _agreement(
    reading: tuple[Decimal, Optional[str]], scaling: Mapping[str, Any]
) -> _Agreement:
    """Whether a printed sensitivity is the configured one: equal in volts
    per the same physical unit. A printed number without a unit agrees only
    when it equals the configured number; otherwise which unit it is in
    cannot be told."""
    value, unit = reading
    configured = scaling["sensitivity"]
    if unit is None:
        return "agree" if value == _decimal(configured["value"]) else "unknown"
    printed_per, stated_per = (
        SENSITIVITY_UNITS[unit][1],
        SENSITIVITY_UNITS[configured["unit"]][1],
    )
    if printed_per != stated_per:
        # Per another unit of the same quantity (g and m/s2) the values
        # cannot be compared exactly; per another quantity they differ.
        same_quantity = _QUANTITY[printed_per] == _QUANTITY[stated_per]
        return "unknown" if same_quantity else "differ"
    printed = {
        "sensitivity": {"value": format(value, "f"), "unit": unit},
        "source_unit": "V",
    }
    stated = {"sensitivity": configured, "source_unit": "V"}
    return "agree" if scaling_divisor(printed) == scaling_divisor(stated) else "differ"


def _sole_electrical_column(
    run: _Run, evidence: _Evidence, field_count: int, scaled: list[int]
) -> Optional[int]:
    """The one column every printed sensitivity can only describe: the sole
    scaled channel, when every other column of the data is a configured
    time column or prints a unit token that is not electrical."""
    if len(scaled) != 1:
        return None
    time_columns = {c["column"] for c in run.interpretation["time_columns"]}
    for column in range(field_count):
        if column == scaled[0] or column in time_columns:
            continue
        kinds = [
            _classify_unit(token)[2] for token, _line in _unit_tokens(evidence, column)
        ]
        if not kinds or any(
            kind not in ("importable", "time", "preview") for kind in kinds
        ):
            return None
    return scaled[0]


def _check_sensitivities(run: _Run, evidence: _Evidence, field_count: int) -> None:
    """Every printed sensitivity against the scaled channels it may describe.

    The file ties a printed sensitivity to a channel only when that channel
    is the sole electrical column (:func:`_sole_electrical_column`); a
    contradiction then is refused unless an override names a reason. A
    channel whose scaling basis cites a line is tied to that line by the
    user and checked by its basis. A printed sensitivity that cannot be
    read, or that differs from a channel the file does not tie it to, needs
    the user."""
    channels = {
        c["column"]: c
        for c in run.interpretation["channels"]
        if c["scaling"] is not None and c["scaling"].get("sensitivity")
    }
    if not channels:
        return  # nothing configured that a printed sensitivity describes
    citing = {}
    for column in channels:
        entry = run.basis(f"channels[{column}].scaling")
        if entry is not None and entry["basis"] == "user_accepted_file_value":
            citing[column] = entry["line"]
    sole = _sole_electrical_column(run, evidence, field_count, sorted(channels))
    for found in evidence.preamble:
        if found.field != "sensitivity":
            continue
        # A channel citing this line is compared by its basis; every other
        # scaled channel is one this line may describe.
        targets = [
            column
            for column in ([sole] if sole is not None else sorted(channels))
            if citing.get(column) != found.line
        ]
        if not targets:
            continue
        reading = _sensitivity_reading(found)
        verdicts: dict[int, _Agreement] = {
            column: (
                "unknown"
                if reading is None
                else _agreement(reading, channels[column]["scaling"])
            )
            for column in targets
        }
        if sole is not None and verdicts[sole] == "differ":
            path = f"channels[{sole}].scaling"
            override = run.override(path)
            run.issue(
                "contradicts_file_declaration",
                "notice" if override is not None else "invalid",
                f"{st.channel_words(sole).capitalize()}: its configured sensitivity "
                f"is not the one line {found.line} of the file prints, and every "
                f"other column of the data is a time column or carries a unit "
                f"token that is not electrical"
                + (
                    _RECORDED
                    if override is not None
                    else f". Correct the configuration, or name an override with "
                    f"a reason for {path!r} if the file is wrong."
                ),
                field_path=path,
                line=found.line,
                column=sole,
            )
            continue
        open_columns = [c for c, verdict in verdicts.items() if verdict != "agree"]
        if not open_columns:
            continue
        paths = [f"channels[{column}].scaling" for column in open_columns]
        missing = [path for path in paths if run.override(path) is None]
        words = "; ".join(st.channel_words(column) for column in open_columns)
        confirm = (
            f"{_ASK} to confirm the configured sensitivity, and name an override "
            f"with a reason for each of {missing}."
        )
        if reading is None:
            why = (
                f"a sensitivity that this version cannot read as a number with a "
                f"sensitivity unit (its text as printed is in this issue's token), "
                f"so the configured sensitivity of {words} cannot be checked "
                f"against it"
            )
            ask = confirm
        elif sole is not None:
            # Tied to the sole electrical column, but in another unit of the
            # same quantity, or in none: equal or not cannot be told exactly.
            why = (
                f"a sensitivity (its text as printed is in this issue's token) in "
                f"another unit, or in none, so it cannot be compared exactly with "
                f"the configured sensitivity of {words}"
            )
            ask = confirm
        else:
            why = (
                f"a sensitivity (its text as printed is in this issue's token) that "
                f"the file does not tie to one channel, and it is not shown to be "
                f"the configured sensitivity of {words}"
            )
            ask = (
                f"{_ASK} which channel it describes, cite line {found.line} in "
                f"that channel's scaling basis, and name an override with a "
                f"reason for each of {missing}."
            )
        run.issue(
            "printed_value_unverified",
            "notice" if not missing else "needs_confirmation",
            f"Line {found.line} prints {why}"
            + (
                "; the overrides with their reasons are recorded."
                if not missing
                else f": {ask}"
            ),
            field_path=missing[0] if missing else paths[0],
            line=found.line,
            token=_printed(found.raw),
        )


def _unit_fits(channel: Mapping[str, Any], token: str) -> bool:
    """Whether a printed unit is the configured unit of *channel*: its
    physical unit when unscaled, its source unit when scaled."""
    unit, _role, kind = _classify_unit(token)
    scaling = channel["scaling"]
    if kind == "importable":
        return scaling is None and unit == channel["unit"]
    if kind == "electrical":
        source = (scaling or {}).get("source_unit")
        return source is not None and source.lower() == _unit_text(token).lower()
    return False


def _check_preamble_units(run: _Run, evidence: _Evidence, field_count: int) -> None:
    """A unit the preamble prints (``Unit: mV``) against the configured units.
    It describes the one channel when the data holds no other column than
    configured time columns; otherwise which channel it describes cannot be
    told."""
    channels = run.interpretation["channels"]
    time_columns = {c["column"] for c in run.interpretation["time_columns"]}
    data_columns = [c for c in range(field_count) if c not in time_columns]
    sole = (
        channels[0]["column"]
        if len(channels) == 1 and data_columns == [channels[0]["column"]]
        else None
    )
    for found in evidence.preamble:
        if found.field != "unit":
            continue
        token = found.raw.strip()
        kind = _classify_unit(token)[2]
        if kind == "time":
            continue  # it describes the time axis, never a vibration channel
        readable = kind != "unrecognized"
        open_columns = [
            c["column"] for c in channels if not readable or not _unit_fits(c, token)
        ]
        if not open_columns:
            continue
        if sole is not None and readable:
            path = f"channels[{sole}].unit"
            override = run.override(path)
            run.issue(
                "contradicts_file_declaration",
                "notice" if override is not None else "invalid",
                f"{st.channel_words(sole).capitalize()}: line {found.line} prints a "
                f"unit that is not its configured one, and the data holds no other "
                f"column it could describe"
                + (
                    _RECORDED
                    if override is not None
                    else f". Correct the configuration, or name an override with "
                    f"a reason for {path!r} if the file is wrong."
                ),
                field_path=path,
                line=found.line,
                column=sole,
                token=_printed(token),
            )
            continue
        paths = [f"channels[{column}].unit" for column in open_columns]
        missing = [path for path in paths if run.override(path) is None]
        words = "; ".join(st.channel_words(column) for column in open_columns)
        why = (
            "a unit that this version does not read"
            if not readable
            else "a unit that the file does not tie to one channel"
        )
        run.issue(
            "printed_value_unverified",
            "notice" if not missing else "needs_confirmation",
            f"Line {found.line} prints {why} (its text as printed is in this "
            f"issue's token), and it is not shown to be the configured unit of {words}"
            + (
                "; the overrides with their reasons are recorded."
                if not missing
                else f": {_ASK} to confirm each unit, and name an override with a "
                f"reason for each of {missing}."
            ),
            field_path=missing[0] if missing else paths[0],
            line=found.line,
            token=_printed(token),
        )


def _check_lost_rows(run: _Run, evidence: _Evidence) -> None:
    """A header or units row whose unit tokens cannot be matched to the
    data's fields: the configured units cannot be checked against them."""
    paths = [f"channels[{c['column']}].unit" for c in run.interpretation["channels"]]
    paths += [
        f"time_columns[{c['column']}].time_unit"
        for c in run.interpretation["time_columns"]
    ]
    for kind, line, fields, data_fields in evidence.lost:
        missing = [path for path in paths if run.override(path) is None]
        run.issue(
            "printed_value_unverified",
            "notice" if not missing else "needs_confirmation",
            f"The {kind} row on line {line} (its text as printed is in this issue's "
            f"token) has {fields} fields but the data rows have {data_fields}, so "
            f"its unit tokens cannot be matched to columns and the configured units "
            f"cannot be checked against them"
            + (
                "; the overrides with their reasons are recorded."
                if not missing
                else f": correct the dialect, or {_ASK} to confirm the units and "
                f"name an override with a reason for each of {missing}."
            ),
            field_path=missing[0] if missing else (paths[0] if paths else None),
            line=line,
            token=_printed(evidence.row_text.get(line, "")),
        )
