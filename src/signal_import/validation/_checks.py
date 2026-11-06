"""
Channel, default and structure checks of an import validation.

Unit tokens, time-like channels, printed defaults, each basis against
the value it confirms, the parse proof of the structure (R9), the bases
every written value needs, and the channel bindings.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

from datetime import datetime
from typing import Any, Mapping, Optional

from ...signal_acquisition.measurement import (
    binding_collisions,
    build_import_identity,
    shift_acquired_at,
)
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..inspection import _classify_unit, _unit_text
from ._common import _ASK, _Run, _decimal
from ._evidence import (
    _Evidence,
    _Preamble,
    _printed_unit,
    _time_like,
    _token_line,
    _unit_token,
)
from ._history import _Prior


def _check_time_like_channel(
    run: _Run, evidence: _Evidence, channel: Mapping[str, Any]
) -> None:
    """A selected channel that looks like time keeps a preview role (R16):
    refused unless an override names a reason. A time unit token is the unit
    check's contradiction already."""
    column = channel["column"]
    if not _time_like(evidence, column, run.interpretation["dialect"]["decimal"]):
        return
    token = _unit_token(evidence, column)
    if token is not None and _classify_unit(token)[2] == "time":
        return
    path = f"channels[{column}].unit"
    override = run.override(path)
    run.issue(
        "contradicts_file_declaration",
        "notice" if override is not None else "invalid",
        f"{st.channel_words(column).capitalize()}: its label or first values look "
        f"like a time column, which this version never imports as vibration"
        + (
            "; the override with its reason is recorded."
            if override is not None
            else f". Deselect it (list it in time_columns to assess it as time), "
            f"or name an override with a reason for {path!r} if it is a "
            f"vibration channel."
        ),
        field_path=path,
        column=column,
    )


def _check_stated_defaults(run: _Run, evidence: _Evidence) -> None:
    """A speed or acquisition instant the preamble prints otherwise (R11):
    refused unless an override names a reason. A value accepted from the
    file is checked against its cited line instead."""
    defaults = run.interpretation["defaults"]
    for name, words in (("rpm", "shaft speed"), ("acquired_at", "acquisition instant")):
        value = defaults[name]
        entry = run.basis(f"defaults.{name}")
        if value is None or (
            entry is not None and entry["basis"] == "user_accepted_file_value"
        ):
            continue
        path = f"defaults.{name}"
        override = run.override(path)
        for found in evidence.preamble:
            if found.field != name or not _contradicts(name, found, value):
                continue
            run.issue(
                "contradicts_file_declaration",
                "notice" if override is not None else "invalid",
                f"The configured {words} is not the one line {found.line} of the "
                f"file prints"
                + (
                    "; the override with its reason is recorded."
                    if override is not None
                    else f". Correct the configuration, or name an override with "
                    f"a reason for {path!r} if the file is wrong."
                ),
                field_path=path,
                line=found.line,
            )
            break


def _contradicts(name: str, found: _Preamble, value: str) -> bool:
    """A printed speed or instant that differs from the configured one; a
    printed value the comparison cannot read contradicts nothing."""
    if name == "rpm":
        printed = _decimal(found.normalized)
        return printed is not None and printed != _decimal(value)
    try:
        left = datetime.fromisoformat(found.raw.strip())
        right = datetime.fromisoformat(value)
    except ValueError:
        return False
    if left.tzinfo is None or right.tzinfo is None:
        # A naive print states a wall-clock time only.
        return left.replace(tzinfo=None) != right.replace(tzinfo=None)
    return left != right


def _check_units(run: _Run, evidence: _Evidence, channel: Mapping[str, Any]) -> None:
    """A unit token on a selected column against the configured unit."""
    column = channel["column"]
    token = _unit_token(evidence, column)
    scaling = channel["scaling"]
    path = f"channels[{column}].unit"
    override = run.override(path)
    where = st.channel_words(column).capitalize()
    if token is not None:
        unit, _role, kind = _classify_unit(token)
        line = _token_line(evidence)
        contradiction: Optional[str] = None
        if kind == "importable":
            if scaling is not None:
                run.issue(
                    "double_scaling",
                    "invalid",
                    f"{where} already states a physical unit on line {line}, so a "
                    f"sensitivity would scale it twice: remove the scaling.",
                    field_path=f"channels[{column}].scaling",
                    line=line,
                    column=column,
                )
            elif unit != channel["unit"]:
                contradiction = (
                    f"its unit token on line {line} maps to {unit}, not to the "
                    f"configured {channel['unit']}"
                )
        elif kind == "electrical":
            electrical = _unit_text(token).lower()
            source_unit = scaling["source_unit"] if scaling is not None else None
            if scaling is None:
                contradiction = (
                    f"its unit token on line {line} is electrical, so storing it "
                    f"in {channel['unit']} needs a sensitivity and its source unit"
                )
            elif source_unit is None or source_unit.lower() != electrical:
                contradiction = (
                    f"its unit token on line {line} names another electrical unit "
                    f"than the scaling's source unit"
                )
        elif kind == "preview":
            if override is None:
                run.issue(
                    "unit_token_preview_only",
                    "invalid",
                    f"{where} carries a unit token on line {line} that names a "
                    f"quantity this version shows but never imports, so it is not "
                    f"imported as vibration; deselect it, or name an override with "
                    f"a reason for {path!r} if the token is wrong.",
                    field_path=path,
                    line=line,
                    column=column,
                )
            else:
                run.issue(
                    "unit_token_preview_only",
                    "notice",
                    f"{where} carries a preview-only unit token on line {line}; "
                    f"the override with its reason is recorded.",
                    field_path=path,
                    line=line,
                    column=column,
                )
            return
        elif kind == "time":
            contradiction = f"its unit token on line {line} is a time unit"
        else:
            contradiction = (
                f"its unit token on line {line} does not map to the configured "
                f"{channel['unit']}"
            )
        if contradiction is not None:
            run.issue(
                "contradicts_file_declaration",
                "notice" if override is not None else "invalid",
                f"{where}: {contradiction}"
                + (
                    "; the override with its reason is recorded."
                    if override is not None
                    else f". Correct the configuration, or name an override with "
                    f"a reason for {path!r} if the file is wrong."
                ),
                field_path=path,
                line=line,
                column=column,
            )


def _check_value_bases(
    run: _Run,
    evidence: _Evidence,
    prior: _Prior,
    points: Mapping[int, Optional[Mapping[str, Any]]],
) -> None:
    """Every basis that accepts a file, earlier-import or point value holds
    the configured value."""
    interpretation = run.interpretation
    for channel in interpretation["channels"]:
        column = channel["column"]
        for part in ("unit", "scaling", "dc_policy", "binding"):
            path = f"channels[{column}].{part}"
            entry = run.basis(path)
            if entry is None or entry["basis"] == "user_stated":
                continue
            problem = _channel_basis_problem(
                run, evidence, prior, points.get(column), channel, part, entry
            )
            if problem is not None:
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"{st.channel_words(column).capitalize()}: {problem}",
                    field_path=path,
                    line=entry["line"],
                    column=column,
                )
    defaults = interpretation["defaults"]
    for name in ("acquired_at", "rpm", "load", "operating_state"):
        path = f"defaults.{name}"
        entry = run.basis(path)
        if entry is None or entry["basis"] == "user_stated" or defaults[name] is None:
            continue
        problem = _default_basis_problem(
            evidence, prior, points, name, defaults[name], entry
        )
        if problem is not None:
            run.issue(
                "basis_mismatch",
                "invalid",
                problem,
                field_path=path,
                line=entry["line"],
            )
    for path in ("dialect", "data_lines", "companion"):
        entry = run.basis(path)
        if entry is None or entry["basis"] == "user_stated":
            continue
        if entry["basis"] == "user_accepted_prior_value" and path != "companion":
            earlier = prior.interpretation(entry["import_id"])
            if earlier is not None and earlier[path] == interpretation[path]:
                continue
            if earlier is None and prior.unread(entry["import_id"]):
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"The cited earlier import was not read in this validation, so "
                    f"the {path.replace('_', ' ')} cannot be compared with it.",
                    field_path=path,
                )
                continue
        run.issue(
            "basis_mismatch",
            "invalid",
            f"The {path.replace('_', ' ')} is confirmed by stating it (basis "
            f"'user_stated') or, for the dialect and the data range, by citing an "
            f"earlier import of these bytes that used the same.",
            field_path=path,
        )


def _channel_basis_problem(
    run: _Run,
    evidence: _Evidence,
    prior: _Prior,
    point: Optional[Mapping[str, Any]],
    channel: Mapping[str, Any],
    part: str,
    entry: Mapping[str, Any],
) -> Optional[str]:
    basis = entry["basis"]
    column = channel["column"]
    if basis == "user_accepted_prior_value":
        earlier = prior.channel(entry["import_id"], column)
        if earlier is None and prior.unread(entry["import_id"]):
            return "the cited earlier import was not read in this validation."
        if earlier is None:
            return "the cited earlier import of these bytes reads no such column."
        compared = {
            "unit": ("role", "quantity", "unit"),
            "dc_policy": ("scaling",),
        }.get(part, (part,))
        if any(earlier.get(k) != channel.get(k) for k in compared):
            return f"the cited earlier import used another {part.replace('_', ' ')}."
        return None
    if basis == "user_accepted_point_value":
        if point is None:
            return "no declared measurement point is bound, so no point value can be cited."
        if point.get("declaration_version") != entry["declaration_version"]:
            return (
                f"the bound point's current declaration is version "
                f"{point.get('declaration_version')}, not the cited one."
            )
        if part == "unit":
            return (
                None
                if point.get("expected_signal_unit") == channel["unit"]
                else "the point declaration expects another signal unit."
            )
        if part == "binding":
            binding = channel["binding"] or {}
            expected = (
                ("direction", "expected_direction"),
                ("sensor_id", "expected_sensor_id"),
            )
            for key, declared_key in expected:
                declared, stated = point.get(declared_key), binding.get(key)
                words = key.replace("_", " ")
                if declared is not None and declared != stated:
                    return f"the point declaration expects another {words}."
                if declared is None and stated is not None:
                    return f"the point declaration holds no {words}; state it."
            return None
        return f"the point declaration holds no {part.replace('_', ' ')}."
    if basis == "user_accepted_file_value":
        return _file_value_problem(evidence, channel, part, entry["line"])
    return f"the basis {basis!r} cannot confirm a channel's {part.replace('_', ' ')}."


def _file_value_problem(
    evidence: _Evidence, channel: Mapping[str, Any], part: str, cited: Optional[int]
) -> Optional[str]:
    """Why a channel value accepted from the file is not what line *cited*
    holds, or None when it is."""
    column = channel["column"]
    if part == "unit":
        line = _token_line(evidence)
        token = _unit_token(evidence, column)
        if line is None or cited != line or token is None:
            return "the cited line holds no unit token for this column."
        unit, _role, kind = _classify_unit(token)
        if kind != "importable" or unit != channel["unit"] or channel["scaling"]:
            return (
                "the unit token on the cited line is not the configured unit "
                "(a scaled channel's unit comes from its sensitivity: state it)."
            )
        return None
    if part == "scaling":
        sensitivity = (channel["scaling"] or {}).get("sensitivity") or {}
        for value in evidence.preamble:
            if value.line != cited or value.field != "sensitivity":
                continue
            if value.normalized is None or _decimal(value.normalized) != _decimal(
                sensitivity.get("value")
            ):
                return "the sensitivity on the cited line is not the configured value."
            printed_unit = _printed_unit(value)
            if (
                printed_unit is None
                or printed_unit.lower() != str(sensitivity.get("unit") or "").lower()
            ):
                return (
                    "the cited line prints the sensitivity in another unit, or "
                    "in none; state the sensitivity with its unit."
                )
            return None
        return "the cited line prints no sensitivity."
    return f"the file states no {part.replace('_', ' ')}; state it."


def _default_basis_problem(
    evidence: _Evidence,
    prior: _Prior,
    points: Mapping[int, Optional[Mapping[str, Any]]],
    name: str,
    value: str,
    entry: Mapping[str, Any],
) -> Optional[str]:
    basis = entry["basis"]
    if basis == "user_accepted_file_value":
        for found in evidence.preamble:
            if found.line != entry["line"] or found.field != name:
                continue
            if name == "rpm":
                if found.normalized is not None and _decimal(
                    found.normalized
                ) == _decimal(value):
                    return None
                return f"The configured rpm {value} is not the speed line {found.line} prints."
            if name == "acquired_at":
                if _same_instant(found.raw, value):
                    return None
                return f"The configured acquisition instant is not the one line {found.line} prints."
        return f"Line {entry['line']} prints no {name.replace('_', ' ')}."
    if basis == "user_accepted_prior_value":
        earlier = prior.interpretation(entry["import_id"])
        if earlier is not None and earlier["defaults"][name] == value:
            return None
        if earlier is None and prior.unread(entry["import_id"]):
            return "The cited earlier import was not read in this validation."
        return f"The cited earlier import used another {name.replace('_', ' ')}."
    if basis == "user_accepted_point_value" and name == "rpm":
        # The file-level speed every bound channel inherits: every bound
        # point must declare it, at the cited version.
        bound = list(points.values())
        if bound and all(
            point is not None
            and point.get("declaration_version") == entry["declaration_version"]
            and point.get("nominal_rpm") is not None
            and _decimal(str(point["nominal_rpm"])) == _decimal(value)
            for point in bound
        ):
            return None
        return (
            "Not every bound point's declaration at the cited version holds this "
            "nominal rpm."
        )
    return f"The basis {basis!r} cannot confirm the {name.replace('_', ' ')}."


def _same_instant(printed: str, configured: str) -> bool:
    """The printed instant is the configured one: both naive, or both with an
    offset (a naive print never confirms an instant in UTC)."""
    try:
        left = datetime.fromisoformat(printed.strip())
        right = datetime.fromisoformat(configured)
    except ValueError:
        return False
    if (left.tzinfo is None) != (right.tzinfo is None):
        return False
    return left == right


def _structure_proofs(
    interpretation: Mapping[str, Any],
    evidence: _Evidence,
    decode: Optional[DecodeResult],
) -> dict[str, bool]:
    """Whether the parse proves ``dialect`` and ``data_lines`` (R9).

    The whole file must decode as configured and the scanned data must have
    exactly one reading, the configured one. The dialect also needs an
    unambiguous encoding (a Windows-1252 decode alone proves none) and the
    header and units rows the inspection rule finds above the data; the data
    range needs no stated last line.
    """
    dialect = interpretation["dialect"]
    parsed = (
        decode is not None
        and evidence.unique_reading is not None
        and evidence.unique_reading
        == (
            dialect["delimiter"],
            dialect["decimal"],
            interpretation["data_lines"]["first"],
        )
    )
    return {
        "dialect": parsed
        and evidence.encoding == dialect["encoding"]
        and evidence.encoding != "cp1252"
        and evidence.scanned_structure
        == (dialect["header_line"], dialect["units_line"]),
        "data_lines": parsed and interpretation["data_lines"]["last"] is None,
    }


def _required_bases(
    run: _Run,
    evidence: _Evidence,
    proofs: Mapping[str, bool],
    companion: bool,
) -> None:
    interpretation = run.interpretation
    for path, what in (
        (
            "dialect",
            "The encoding, delimiter, decimal mark and header and units rows",
        ),
        ("data_lines", "The data line range"),
    ):
        if not proofs[path] and run.basis(path) is None:
            if path == "dialect" and evidence.unique_reading is None:
                run.issue(
                    "ambiguous_dialect",
                    "needs_confirmation",
                    f"Several readings of the data rows parse equally well, so "
                    f"{what.lower()} need confirmation: {_ASK} which reading is "
                    f"right and name a basis for 'dialect'.",
                    field_path=path,
                )
            elif path == "dialect" and evidence.encoding == "cp1252":
                run.issue(
                    "encoding_needs_confirmation",
                    "needs_confirmation",
                    f"The file is not valid UTF-8, so its encoding needs "
                    f"confirmation: {_ASK} which encoding the export was saved in "
                    f"and name a basis for 'dialect'.",
                    field_path=path,
                )
            else:
                run.missing_basis(path, what)
    for channel in interpretation["channels"]:
        column = channel["column"]
        where = st.channel_words(column).capitalize()
        if run.basis(f"channels[{column}].unit") is None:
            run.missing_basis(
                f"channels[{column}].unit",
                f"{where} as vibration {channel['quantity']} in {channel['unit']}",
            )
        scaling = channel["scaling"]
        if scaling is not None and run.basis(f"channels[{column}].scaling") is None:
            run.missing_basis(
                f"channels[{column}].scaling", f"The sensitivity of {where.lower()}"
            )
        if scaling is not None and scaling["source_unit"] is not None:
            if column in run.pending_dc:
                mean, ratio = run.stats.get(column, (None, None))
                measured = ""
                if mean is not None:
                    measured = (
                        f" Its stored samples have a mean of {mean:.6g} "
                        f"{channel['unit']}"
                        + (
                            f" and a mean divided by standard deviation of {ratio:.6g}."
                            if ratio is not None
                            else " and do not vary."
                        )
                    )
                run.issue(
                    "missing_basis",
                    "needs_confirmation",
                    f"{where} is a voltage channel without a stated DC policy."
                    f"{measured} {_ASK.capitalize()} whether to keep its DC level "
                    f"as recorded (dc_policy 'keep', the only policy of this "
                    f"version: no offset is subtracted) and name a basis for "
                    f"'channels[{column}].dc_policy'.",
                    field_path=f"channels[{column}].dc_policy",
                    column=column,
                )
            elif run.basis(f"channels[{column}].dc_policy") is None:
                run.missing_basis(
                    f"channels[{column}].dc_policy", f"The DC policy of {where.lower()}"
                )
        if (
            channel["binding"] is not None
            and run.basis(f"channels[{column}].binding") is None
        ):
            run.missing_basis(
                f"channels[{column}].binding", f"The binding of {where.lower()}"
            )
    for name, value in interpretation["defaults"].items():
        if (
            value is not None
            and name != "asset_id"
            and run.basis(f"defaults.{name}") is None
        ):
            run.missing_basis(
                f"defaults.{name}", f"The file-level {name.replace('_', ' ')}"
            )
    if companion and run.basis("companion") is None:
        run.issue(
            "companion_present",
            "needs_confirmation",
            f"A companion metadata file lies next to this file. The import never "
            f"reads it, and the envelope report would draw its bearing markers on "
            f"every imported channel: {_ASK} to acknowledge it, with a basis for "
            f"'companion'.",
            field_path="companion",
        )


def _effective_defaults(
    run: _Run, evidence: _Evidence
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """The file-level defaults, with the acquisition instant moved to the
    first selected sample when the data range starts after the data region."""
    defaults = dict(run.interpretation["defaults"])
    first = run.interpretation["data_lines"]["first"]
    unique = evidence.unique_reading
    dialect = run.interpretation["dialect"]
    if (
        defaults["acquired_at"] is None
        or unique is None
        or unique[:2] != (dialect["delimiter"], dialect["decimal"])
        or first <= unique[2]
    ):
        return defaults, None
    shift = shift_acquired_at(
        defaults["acquired_at"],
        offset_samples=first - unique[2],
        sampling_rate_hz=run.interpretation["time_base"]["sampling_rate_hz"],
    )
    defaults["acquired_at"] = shift["acquired_at"]
    return defaults, shift


def _identities(
    run: _Run, decode: DecodeResult, defaults: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    identities: dict[int, dict[str, Any]] = {}
    for channel in run.interpretation["channels"]:
        if channel["binding"] is None:
            continue
        column = channel["column"]
        try:
            identities[column] = build_import_identity(
                defaults,
                channel["binding"],
                channel_index=column,
                content_sha256=decode.source_sha256,
                size_bytes=decode.size_bytes,
            )
        except ValueError as exc:
            missing = "missing required field" in str(exc)
            run.issue(
                "binding_invalid",
                "needs_confirmation" if missing else "invalid",
                f"{exc}"
                + (f" {_ASK.capitalize()} for the missing value." if missing else ""),
                field_path=f"channels[{column}].binding",
                column=column,
            )
    for group in binding_collisions(identities.values()):
        run.issue(
            "binding_collision",
            "invalid",
            f"Channels at indices {list(group)} are bound to the same asset, "
            f"measurement point, direction and sensor, so they would record one "
            f"measurement twice: bind each to its own point, direction or sensor.",
            field_path=f"channels[{group[0]}].binding",
            column=group[0],
        )
    return identities
