"""
Draft import configurations from an inspection.

ISO 13374 Block 1 (data acquisition). :func:`configure_import` is pure: it
builds a draft configuration from an inspection result, the user's explicit
edits and optional prefills (an earlier import of the same bytes, the
measurement point's declaration), and lists what still needs the user. Every
value it takes from evidence is listed as a prefill with its origin, and it
never writes a basis: only the user gives one, and validation checks that an
accepted value equals the evidence it cites.
"""

import copy
from collections.abc import Mapping
from typing import Any, Optional

from ..signal_acquisition.import_config import INTERPRETATION_SCHEMA
from ..signal_acquisition.measurement import unit_family
from . import statements as st
from .attestation import ATTESTATION_SCHEMA, CONFIGURATION_SCHEMA
from .contracts import ImportDraft, InspectionResult, Prefill

__all__ = ["configure_import"]

_ASK = st.ASK


def _region_line(inspection: InspectionResult, kind: str) -> Optional[int]:
    for region in inspection.regions:
        if region.kind == kind:
            return region.first_line
    return None


def _merge(base: dict[str, Any], edits: Mapping[str, Any]) -> dict[str, Any]:
    """The user's edits over the draft: blocks merge key by key, lists and
    values replace."""
    merged = dict(base)
    for key, value in edits.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def configure_import(
    inspection: InspectionResult,
    *,
    edits: Optional[Mapping[str, Any]] = None,
    prior: Optional[Mapping[str, Any]] = None,
    point: Optional[Mapping[str, Any]] = None,
) -> ImportDraft:
    """Build a draft configuration and list what still needs the user.

    Args:
        inspection: The inspection of the file.
        edits: The user's explicit edits, in the configuration's shape
            (``{"interpretation": {...}, "attestation": {...}}``); they win
            over every prefill. Blocks merge key by key; ``channels`` and
            ``time_columns`` replace the drafted lists.
        prior: An earlier import of the same bytes: ``{"import_id",
            "interpretation"}`` (an import record's line 1). Its channel
            entries prefill the same columns.
        point: A measurement point declaration (the ledger payload, with
            ``measurement_point_id`` and ``declaration_version``). Its
            expected unit, direction and sensor prefill the channels bound to
            it, and its nominal rpm the file-level speed.

    Returns:
        The :class:`ImportDraft`. Its attestation names no basis.
    """
    prefills: list[Prefill] = []
    remaining: list[str] = []
    # Inputs are deep-copied: the draft never shares or mutates them.
    edits = copy.deepcopy(dict(edits or {}))
    prior = copy.deepcopy(dict(prior)) if prior is not None else None
    edited = dict(edits.get("interpretation") or {})

    dialect: dict[str, Any] = {
        "encoding": inspection.file.encoding.value,
        "delimiter": None,
        "decimal": None,
        "header_line": _region_line(inspection, "header"),
        "units_line": _region_line(inspection, "units"),
    }
    data_lines: dict[str, Any] = {"first": None, "last": None}
    proposal = inspection.dialect.proposal
    if proposal is not None:
        dialect["delimiter"], dialect["decimal"] = proposal.delimiter, proposal.decimal
        data_lines["first"] = proposal.data_first_line
    elif "dialect" not in edited:
        remaining.append(
            f"The dialect: several readings of the data rows parse equally well; "
            f"{_ASK} which delimiter and decimal mark are right."
        )
    if inspection.file.encoding.status == "ambiguous" and "dialect" not in edited:
        remaining.append(
            f"The text encoding: a Windows-1252 decode alone does not establish "
            f"it; {_ASK} which encoding the export was saved in."
        )

    time_columns = [
        {"column": column.index, "time_unit": column.time_stats.time_unit}
        for column in inspection.columns
        if column.role_candidate == "time"
        and column.time_stats is not None
        and column.time_stats.time_unit is not None
        and column.time_stats.time_unit_status == "detected"
    ]

    time_base = _time_base(inspection, prior, prefills)

    defaults: dict[str, Any] = {
        "asset_id": None,
        "acquired_at": None,
        "rpm": None,
        "load": None,
        "operating_state": None,
    }
    speed = next(
        (
            c
            for c in inspection.candidates
            if c.field == "rpm" and c.status == "detected" and c.normalized
        ),
        None,
    )
    if speed is not None:
        defaults["rpm"] = speed.normalized
        prefills.append(
            Prefill(
                field="defaults.rpm",
                value=speed.normalized or "",
                origin="file_declared",
                line=speed.line,
            )
        )

    channels: list[dict[str, Any]] = []
    for column in inspection.columns:
        family = unit_family(column.unit) if column.unit else None
        if column.role_candidate == "vibration" and family is not None:
            channels.append(
                {
                    "column": column.index,
                    "role": "vibration",
                    "quantity": family,
                    "unit": column.unit,
                    "scaling": None,
                    "binding": None,
                }
            )
            prefills.append(
                Prefill(
                    field=f"channels[{column.index}].unit",
                    value=column.unit or "",
                    origin="file_declared",
                    line=column.unit_line,
                )
            )
        elif column.role_candidate == "voltage":
            remaining.append(
                f"{st.channel_words(column.index).capitalize()} carries an "
                f"electrical unit: to import it, {_ASK} for its quantity, the "
                f"stored unit, the sensitivity with its unit and source, and the DC "
                f"policy."
            )
    if prior is not None:
        channels = _prior_channels(channels, prior, prefills)

    interpretation: dict[str, Any] = {
        "schema": INTERPRETATION_SCHEMA,
        "dialect": dialect,
        "data_lines": data_lines,
        "time_base": time_base,
        "time_columns": time_columns,
        "defaults": defaults,
        "channels": channels,
    }
    interpretation = _merge(interpretation, edited)
    if point is not None:
        _point_prefills(interpretation, point, prefills)

    remaining.extend(_confirmations(interpretation, prefills))
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "bases": {},
        "overrides": {},
        "declared_by": None,
    }
    attestation.update(edits.get("attestation") or {})
    return ImportDraft(
        configuration={
            "schema": CONFIGURATION_SCHEMA,
            "interpretation": interpretation,
            "attestation": attestation,
        },
        prefills=prefills,
        remaining_confirmations=remaining,
    )


def _time_base(
    inspection: InspectionResult,
    prior: Optional[Mapping[str, Any]],
    prefills: list[Prefill],
) -> dict[str, Any]:
    """The time base a draft proposes: a rate a time column derives, else one
    the preamble prints, else an earlier import's."""
    time_base: dict[str, Any] = {
        "source": None,
        "sampling_rate_hz": None,
        "time_column": None,
    }
    derived = next(
        (
            c
            for c in inspection.candidates
            if c.field == "sampling_rate_hz" and c.status == "derived"
        ),
        None,
    )
    printed = next(
        (
            c
            for c in inspection.candidates
            if c.field == "sampling_rate_hz" and c.status == "detected" and c.normalized
        ),
        None,
    )
    if derived is not None and derived.normalized and derived.column is not None:
        time_base = {
            "source": "time_column",
            "sampling_rate_hz": derived.normalized,
            "time_column": derived.column,
        }
        prefills.append(
            Prefill(
                field="time_base",
                value=derived.normalized,
                origin="derived",
                column=derived.column,
            )
        )
    elif printed is not None:
        time_base = {
            "source": "file_declared",
            "sampling_rate_hz": printed.normalized,
            "time_column": None,
        }
        prefills.append(
            Prefill(
                field="time_base",
                value=printed.normalized or "",
                origin="file_declared",
                line=printed.line,
            )
        )
    elif prior is not None:
        rate = prior["interpretation"]["time_base"]["sampling_rate_hz"]
        time_base = {"source": "stated", "sampling_rate_hz": rate, "time_column": None}
        prefills.append(
            Prefill(
                field="time_base",
                value=rate,
                origin="prior_import",
                import_id=prior["import_id"],
            )
        )
    return time_base


def _prior_channels(
    channels: list[dict[str, Any]],
    prior: Mapping[str, Any],
    prefills: list[Prefill],
) -> list[dict[str, Any]]:
    """An earlier import's channels: all of them when the inspection proposes
    none, else their scaling and binding for the same columns."""
    earlier = {c["column"]: c for c in prior["interpretation"]["channels"]}
    if not channels:
        channels = [dict(c) for c in earlier.values()]
        for channel in channels:
            prefills.append(
                Prefill(
                    field=f"channels[{channel['column']}].unit",
                    value=channel["unit"],
                    origin="prior_import",
                    import_id=prior["import_id"],
                )
            )
        return channels
    for channel in channels:
        match = earlier.get(channel["column"])
        if match is None:
            continue
        for part in ("scaling", "binding"):
            if channel[part] is None and match.get(part) is not None:
                channel[part] = match[part]
                prefills.append(
                    Prefill(
                        field=f"channels[{channel['column']}].{part}",
                        value=str(match[part]),
                        origin="prior_import",
                        import_id=prior["import_id"],
                    )
                )
    return channels


def _point_prefills(
    interpretation: dict[str, Any],
    point: Mapping[str, Any],
    prefills: list[Prefill],
) -> None:
    """A point declaration's expected direction and sensor on the channels
    bound to it, and its nominal rpm when the draft states no speed."""
    version = point.get("declaration_version")
    point_id = point.get("measurement_point_id")
    for channel in interpretation.get("channels") or []:
        binding = channel.get("binding") or {}
        if binding.get("measurement_point_id") != point_id:
            continue
        for key, declared in (
            ("direction", "expected_direction"),
            ("sensor_id", "expected_sensor_id"),
        ):
            if binding.get(key) is None and point.get(declared) is not None:
                binding[key] = point[declared]
                prefills.append(
                    Prefill(
                        field=f"channels[{channel['column']}].binding",
                        value=str(point[declared]),
                        origin="point_declaration",
                        declaration_version=version,
                    )
                )
        channel["binding"] = binding
    nominal = point.get("nominal_rpm")
    if (
        nominal is not None
        and (interpretation.get("defaults") or {}).get("rpm") is None
    ):
        speed = format(float(nominal), "g")
        interpretation.setdefault("defaults", {})["rpm"] = speed
        prefills.append(
            Prefill(
                field="defaults.rpm",
                value=speed,
                origin="point_declaration",
                declaration_version=version,
            )
        )


def _accept(prefill: Optional[Prefill]) -> str:
    """How the user would accept a prefilled value."""
    if prefill is None:
        return f"{_ASK} to state it (basis 'user_stated')"
    citation = {
        "file_declared": ("user_accepted_file_value", f"citing line {prefill.line}"),
        "derived": ("user_accepted_derived_value", f"citing column {prefill.column}"),
        "prior_import": (
            "user_accepted_prior_value",
            f"citing import {prefill.import_id}",
        ),
        "point_declaration": (
            "user_accepted_point_value",
            f"citing point declaration version {prefill.declaration_version}",
        ),
    }.get(prefill.origin)
    if citation is None:
        return f"{_ASK} to state it (basis 'user_stated')"
    basis, cited = citation
    return (
        f"{_ASK} to accept {prefill.value!r} (basis {basis!r}, {cited}) or to "
        f"state another value (basis 'user_stated')"
    )


def _confirmations(
    interpretation: Mapping[str, Any], prefills: list[Prefill]
) -> list[str]:
    """Every value the draft would write, and how the user confirms it."""
    by_field: dict[str, Prefill] = {}
    for prefill in prefills:
        by_field.setdefault(prefill.field, prefill)
    out: list[str] = []
    time_base = interpretation.get("time_base") or {}
    if time_base.get("sampling_rate_hz") is None:
        out.append(
            f"The sampling rate: no time column or preamble value of the file "
            f"supports one; {_ASK} for the rate and its source."
        )
    else:
        prefill = by_field.get("time_base")
        text = f"The sampling rate and its source: {_accept(prefill)}."
        if prefill is not None and prefill.origin == "derived":
            text += (
                " The inspection derived this rate from the scanned head of the "
                "time column; validation derives it from the whole column and "
                "states that value, which is the one to accept."
            )
        out.append(text)
    channels = interpretation.get("channels") or []
    if not channels:
        out.append(f"The channels to import: {_ASK} which columns hold vibration.")
    for channel in channels:
        column = channel.get("column")
        where = st.channel_words(column) if isinstance(column, int) else "a channel"
        out.append(
            f"The role, quantity and unit of {where}: "
            f"{_accept(by_field.get(f'channels[{column}].unit'))}."
        )
        if channel.get("scaling") is not None:
            out.append(
                f"The sensitivity of {where}: {_accept(by_field.get(f'channels[{column}].scaling'))}."
            )
        if channel.get("binding") is not None:
            out.append(
                f"The binding of {where}: {_accept(by_field.get(f'channels[{column}].binding'))}."
            )
    for name, value in (interpretation.get("defaults") or {}).items():
        if value is not None and name != "asset_id":
            out.append(
                f"The file-level {name.replace('_', ' ')}: "
                f"{_accept(by_field.get(f'defaults.{name}'))}."
            )
    return out
