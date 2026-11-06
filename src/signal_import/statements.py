"""
Confirmation statements of a signal import validation.

ISO 13374 Block 1 (data acquisition). Validation writes one statement per
engineering value the import would write, in the server's words: the
sampling rate and its source, each channel's role, quantity, unit and
scaling, a DC policy, a binding, the acquisition instant, the shaft speed,
load and operating state, the dialect and data range, a companion file's
acknowledgement. The assistant presents them verbatim before an import.

Rules every statement follows:

- a channel is named by its physical index and its 1-based position in
  words ("index 3, the 4th field from the left"), never by its label;
- the prose states the consequence ("its samples, in millivolts as
  configured, are divided by 100 mV per g");
- a value's declared provenance (a sensitivity's source) is kept apart from
  its basis, which says whether the user stated it or accepted it from
  evidence validation checked it against;
- file text (labels, first tokens) never enters the prose: it travels in the
  statement's own file-text fields, the label only when the header's field
  count matches the data's;
- the basis is stated, and a basis that arrived as an MCP tool argument is
  marked caller-asserted.
"""

from typing import Any, Mapping, Optional, Sequence

from .contracts import FileText, Statement

__all__ = [
    "ASK",
    "ordinal",
    "position_words",
    "channel_words",
    "basis_words",
    "time_base_statement",
    "channel_unit_statement",
    "scaling_statement",
    "dc_policy_statement",
    "binding_statement",
    "default_statement",
    "structure_statement",
    "companion_statement",
]

#: How a server message hands a confirmation to the assistant.
ASK = "ask the user"

_SOURCE_WORDS = {
    "stated": "stated",
    "file_declared": "printed in the file",
    "time_column": "derived from a time column",
}

_SENSITIVITY_SOURCE_WORDS = {
    "datasheet_nominal": "a datasheet nominal value",
    "calibration_certificate": "a calibration certificate",
    "teds": "the sensor's TEDS",
    "user_estimate": "an estimate",
}

_ELECTRICAL_WORDS = {"mV": "millivolts", "V": "volts"}

_DEFAULT_WORDS = {
    "acquired_at": "The acquisition instant of the first selected sample is",
    "rpm": "The shaft speed is",
    "load": "The load is",
    "operating_state": "The operating state is",
}


def ordinal(number: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``4th``, ``11th``, ``21st``, ..."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def position_words(column: int) -> str:
    """A physical 0-based column as its 1-based position in words."""
    return f"the {ordinal(column + 1)} field from the left"


def channel_words(column: int) -> str:
    """``index 3, the 4th field from the left``."""
    return f"index {column}, {position_words(column)}"


def basis_words(
    basis: Optional[str], caller_asserted: bool, *, citation: str = ""
) -> str:
    """The user's act behind a value, in words, with its citation."""
    if basis is None:
        return "no basis is given yet"
    words = {
        "user_stated": "as stated",
        "user_accepted_file_value": "accepted from the file",
        "user_accepted_derived_value": "accepted as derived from the data",
        "user_accepted_prior_value": "accepted from an earlier import of the same bytes",
        "user_accepted_point_value": "accepted from the measurement point's declaration",
        "user_accepted_profile_value": "accepted from an import profile",
    }[basis]
    if citation:
        words = f"{words} ({citation})"
    if caller_asserted:
        words = f"{words}, as the caller asserted (not verified as the user's own act)"
    return words


def _statement(
    field: str,
    text: str,
    *,
    column: Optional[int] = None,
    label: Optional[FileText] = None,
    head_tokens: Sequence[FileText] = (),
    origin: Optional[str],
    basis: Optional[str],
    caller_asserted: bool,
) -> Statement:
    return Statement(
        field=field,
        column=column,
        text=text,
        label=label,
        head_tokens=list(head_tokens),
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted and basis is not None,
    )


def time_base_statement(
    *,
    rate_hz: str,
    source: str,
    time_column: Optional[int],
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    citation: str,
    discrepancy: str = "",
    override: Optional[str] = None,
) -> Statement:
    """The sampling rate, its source, its basis and any recorded discrepancy."""
    source_words = _SOURCE_WORDS[source]
    if source == "time_column" and time_column is not None:
        source_words = f"derived from the time column at {channel_words(time_column)}"
    text = (
        f"Every imported channel is stored at {rate_hz} Hz, {source_words}, "
        f"{basis_words(basis, caller_asserted, citation=citation)}."
    )
    if discrepancy:
        text = f"{text} {discrepancy}"
    if override is not None:
        text = f"{text} An override with a reason is recorded for the time base."
    return _statement(
        "time_base",
        text,
        column=time_column,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def channel_unit_statement(
    *,
    column: int,
    quantity: str,
    unit: str,
    scaling: Optional[Mapping[str, Any]],
    label: Optional[FileText],
    head_tokens: Sequence[FileText],
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    citation: str,
    override: Optional[str] = None,
) -> Statement:
    """A channel's role, quantity and unit. A scaled channel's sensitivity has
    its own statement (:func:`scaling_statement`) with its own basis."""
    text = f"{channel_words(column).capitalize()} is imported as vibration {quantity} in {unit}"
    if scaling is not None and scaling.get("source_unit"):
        electrical = _ELECTRICAL_WORDS.get(scaling["source_unit"], "the source unit")
        text += f", converted from {electrical} by its sensitivity"
    text += f", {basis_words(basis, caller_asserted, citation=citation)}."
    if override is not None:
        text += (
            " The column's own unit token says otherwise; an override with a "
            "reason is recorded."
        )
    return _statement(
        f"channels[{column}].unit",
        text,
        column=column,
        label=label,
        head_tokens=head_tokens,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def scaling_statement(
    *,
    column: int,
    unit: str,
    scaling: Mapping[str, Any],
    divisor: Optional[str],
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    citation: str,
) -> Statement:
    """A scaled channel's sensitivity, the transformation it applies, and the
    basis it was confirmed on. The sensitivity's source (datasheet,
    certificate, TEDS, estimate) is what the configuration declares; the
    basis says whether the value was stated or accepted from evidence that
    validation checked it against."""
    sensitivity = scaling.get("sensitivity") or {}
    source_unit = scaling.get("source_unit")
    electrical = _ELECTRICAL_WORDS.get(source_unit, "the source unit")
    source = _SENSITIVITY_SOURCE_WORDS.get(sensitivity.get("source"), "a stated value")
    transformation = (
        f"divided by {divisor} {source_unit} per {unit}"
        if divisor is not None
        else "divided by this sensitivity"
    )
    # The basis covers the sensitivity's value and unit only; the source
    # unit is the configuration's, said as such.
    text = (
        f"{channel_words(column).capitalize()} has the sensitivity "
        f"{sensitivity.get('value')} {sensitivity.get('unit')}, "
        f"{basis_words(basis, caller_asserted, citation=citation)}; its declared "
        f"source is {source}. Its samples, in {electrical} as configured, are "
        f"{transformation}."
    )
    return _statement(
        f"channels[{column}].scaling",
        text,
        column=column,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def dc_policy_statement(
    *,
    column: int,
    unit: str,
    mean: Optional[float],
    mean_to_std: Optional[float],
    head_tokens: Sequence[FileText],
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
) -> Statement:
    """An electrical channel's DC level, kept as recorded."""
    measured = ""
    if mean is not None:
        measured = f" (mean {mean:.6g} {unit}"
        measured += (
            f", mean divided by standard deviation {mean_to_std:.6g})"
            if mean_to_std is not None
            else ", constant samples)"
        )
    text = (
        f"{channel_words(column).capitalize()} keeps its DC level as recorded"
        f"{measured}: no offset is subtracted, "
        f"{basis_words(basis, caller_asserted)}."
    )
    return _statement(
        f"channels[{column}].dc_policy",
        text,
        column=column,
        head_tokens=head_tokens,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def binding_statement(
    *,
    column: int,
    identity: Mapping[str, Any],
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    citation: str,
) -> Statement:
    """A channel's asset, measurement point, direction and sensor."""
    parts = [
        f"measurement point {identity['measurement_point_id']} of asset "
        f"{identity['asset_id']}"
    ]
    parts.append(
        f"direction {identity['direction']}"
        if identity.get("direction")
        else "no direction"
    )
    parts.append(
        f"sensor '{identity['sensor_id']}'"
        if identity.get("sensor_id")
        else "no sensor id"
    )
    text = (
        f"{channel_words(column).capitalize()} is recorded as measurement "
        f"{identity['measurement_id']} at {', '.join(parts)}, "
        f"{basis_words(basis, caller_asserted, citation=citation)}."
    )
    return _statement(
        f"channels[{column}].binding",
        text,
        column=column,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def default_statement(
    *,
    field: str,
    value: str,
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    citation: str,
    shift: Optional[Mapping[str, Any]] = None,
    override: Optional[str] = None,
) -> Statement:
    """A file-level declaration value every bound channel inherits."""
    unit = " rpm" if field == "rpm" else ""
    text = (
        f"{_DEFAULT_WORDS[field]} {value}{unit}, "
        f"{basis_words(basis, caller_asserted, citation=citation)}."
    )
    if shift is not None:
        exact = "" if shift["shift_exact"] else ", rounded to the microsecond"
        text += (
            f" The data range starts {shift['offset_samples']} samples after the "
            f"data region, so the stated instant is moved by "
            f"{shift['exact_shift_seconds']} s{exact}."
        )
    if override is not None:
        text += " The file states another value; an override with a reason is recorded."
    return _statement(
        f"defaults.{field}",
        text,
        origin=origin,
        basis=basis,
        caller_asserted=caller_asserted,
    )


def structure_statement(
    *,
    field: str,
    text: str,
    basis: Optional[str],
    origin: Optional[str],
    caller_asserted: bool,
    proven: bool,
) -> Statement:
    """The dialect or the data range: confirmed, proven by the parse, or
    neither (*proven* says whether the parse proves it)."""
    if basis is not None:
        text = f"{text}, {basis_words(basis, caller_asserted)}."
    elif proven:
        text = (
            f"{text}: the whole file parses this way and no other reading of "
            f"the scanned data does."
        )
    else:
        text = f"{text}, {basis_words(None, caller_asserted)}."
    return _statement(
        field, text, origin=origin, basis=basis, caller_asserted=caller_asserted
    )


def companion_statement(*, basis: Optional[str], caller_asserted: bool) -> Statement:
    """The acknowledgement a companion metadata file needs."""
    text = (
        "A companion metadata file lies next to this file. The import never "
        "reads it, and the envelope report would draw its bearing markers on "
        f"every imported channel; {basis_words(basis, caller_asserted)}."
    )
    return _statement(
        "companion",
        text,
        origin="stated" if basis else None,
        basis=basis,
        caller_asserted=caller_asserted,
    )
