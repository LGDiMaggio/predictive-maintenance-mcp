"""
Channel, default and structure checks of an import validation.

Unit tokens, time-like channels, printed defaults, each basis against
the value it confirms, the parse proof of the structure (R9), the bases
every written value needs, and the channel bindings.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Optional

from ...signal_acquisition.measurement import (
    binding_collisions,
    build_import_identity,
    shift_acquired_at,
)
from ...file_text import PREAMBLE_LINE_CHARS, FileText
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..inspection import _classify_unit, _unit_text
from ..attestation import MAX_EXCLUSIONS
from ..region_audit import BlockTally, ExcludedBlock, RegionAudit, three_significant
from ._common import _ASK, _Run, _decimal, _delimiter_words, _printed
from ._evidence import (
    _Evidence,
    _Preamble,
    _printed_unit,
    _time_like,
    _token_line,
    _unit_token,
    _unit_tokens,
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


_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Why a printed speed or instant cannot be compared, by verdict.
_UNVERIFIED = {
    "unreadable_rpm": (
        "Line {line} prints a shaft speed that this version cannot read as a "
        "number in rpm (its text as printed is in this issue's token), so the "
        "configured {value} rpm cannot be checked against it"
    ),
    "unreadable_acquired_at": (
        "Line {line} prints an acquisition instant that this version cannot read "
        "as an ISO 8601 date and time (its text as printed is in this issue's "
        "token), so the configured instant cannot be checked against it"
    ),
    "naive_printed": (
        "Line {line} prints the acquisition instant without a time zone while the "
        "configured instant declares an offset, so the two cannot be compared "
        "without assuming a zone"
    ),
    "aware_printed": (
        "Line {line} prints the acquisition instant with a time zone while the "
        "configured instant declares none, so the two cannot be compared without "
        "assuming a zone"
    ),
    "date_only": (
        "Line {line} prints only a date, with no time zone, while the configured "
        "instant declares an offset, so its local date cannot be compared without "
        "assuming a zone"
    ),
}


def _check_stated_defaults(run: _Run, evidence: _Evidence) -> None:
    """Every speed or acquisition instant the preamble prints against the
    configured one (R11): a contradiction is refused unless an override names
    a reason, and a printed value that cannot be read or compared needs the
    user. The line a file-value basis cites is checked by that basis; every
    other printed line is compared here."""
    defaults = run.interpretation["defaults"]
    for name, words in (("rpm", "shaft speed"), ("acquired_at", "acquisition instant")):
        value = defaults[name]
        if value is None:
            continue
        path = f"defaults.{name}"
        entry = run.basis(path)
        cited = (
            entry["line"]
            if entry is not None and entry["basis"] == "user_accepted_file_value"
            else None
        )
        override = run.override(path)
        contradicted = False
        for found in evidence.preamble:
            if found.field != name or found.line == cited:
                continue
            verdict = _printed_verdict(name, found, value)
            if verdict == "agree" or (verdict == "differ" and contradicted):
                continue
            if verdict == "differ":
                contradicted = True
                run.issue(
                    "contradicts_file_declaration",
                    "notice" if override is not None else "invalid",
                    f"The configured {words} is not the one line {found.line} of "
                    f"the file prints"
                    + (
                        "; the override with its reason is recorded."
                        if override is not None
                        else f". Correct the configuration, or name an override "
                        f"with a reason for {path!r} if the file is wrong."
                    ),
                    field_path=path,
                    line=found.line,
                )
                continue
            run.issue(
                "printed_value_unverified",
                "notice" if override is not None else "needs_confirmation",
                _UNVERIFIED[verdict].format(line=found.line, value=value)
                + (
                    "; the override with its reason is recorded."
                    if override is not None
                    else f": {_ASK} to confirm the configured value, and name an "
                    f"override with a reason for {path!r}."
                ),
                field_path=path,
                line=found.line,
                token=_printed(found.raw),
            )


def _printed_verdict(name: str, found: _Preamble, value: str) -> str:
    """``agree``, ``differ``, or why a printed speed or instant cannot be
    compared with the configured one. A naive instant is compared only with
    a naive one, and a date alone only with a naive instant's date: no zone
    is ever assumed."""
    if name == "rpm":
        printed = _decimal(found.normalized)
        if printed is None:
            return "unreadable_rpm"
        return "agree" if printed == _decimal(value) else "differ"
    text = found.raw.strip()
    configured = datetime.fromisoformat(value)
    if _DATE_ONLY.fullmatch(text):
        if configured.tzinfo is not None:
            return "date_only"
        try:
            day = date.fromisoformat(text)
        except ValueError:
            return "unreadable_acquired_at"
        return "agree" if day == configured.date() else "differ"
    try:
        printed_at = datetime.fromisoformat(text)
    except ValueError:
        return "unreadable_acquired_at"
    if printed_at.tzinfo is None and configured.tzinfo is not None:
        return "naive_printed"
    if printed_at.tzinfo is not None and configured.tzinfo is None:
        return "aware_printed"
    return "agree" if printed_at == configured else "differ"


def _check_units(run: _Run, evidence: _Evidence, channel: Mapping[str, Any]) -> None:
    """Every unit token the file prints for a selected column against the
    configured unit."""
    for token, line in _unit_tokens(evidence, channel["column"]):
        _check_unit_token(run, channel, token, line)


def _check_unit_token(
    run: _Run, channel: Mapping[str, Any], token: str, line: Optional[int]
) -> None:
    column = channel["column"]
    scaling = channel["scaling"]
    path = f"channels[{column}].unit"
    override = run.override(path)
    where = st.channel_words(column).capitalize()
    if token is not None:
        unit, _role, kind = _classify_unit(token)
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
                # The scaling's source unit is what disagrees, so only the
                # scaling's override records a reason for it.
                scaling_path = f"channels[{column}].scaling"
                scaling_override = run.override(scaling_path)
                run.issue(
                    "contradicts_file_declaration",
                    "notice" if scaling_override is not None else "invalid",
                    f"{where}: its unit token on line {line} names another "
                    f"electrical unit than the scaling's source unit"
                    + (
                        "; the override with its reason is recorded."
                        if scaling_override is not None
                        else f". Correct the configuration, or name an override "
                        f"with a reason for {scaling_path!r} if the file is wrong."
                    ),
                    field_path=scaling_path,
                    line=line,
                    column=column,
                )
                return
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
    audit: Optional[RegionAudit] = None,
) -> dict[str, bool]:
    """Whether the parse proves ``dialect`` and ``data_lines`` (R9).

    The whole file must decode as configured and the scanned data must have
    exactly one reading, the configured one. The dialect also needs an
    unambiguous encoding (a Windows-1252 decode alone proves none), the
    header and units rows the inspection rule finds above the data, and
    configured header and units lines that are not damaged data rows; the
    data range needs no stated last line and a census that found no line
    above the data start that could be part of the acquisition.
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
    kinds = audit.kinds if audit is not None else {}
    undamaged = all(
        kinds.get(line) != "damaged"
        for line in (dialect["header_line"], dialect["units_line"])
        if line is not None
    )
    return {
        "dialect": parsed
        and evidence.encoding == dialect["encoding"]
        and evidence.encoding != "cp1252"
        and evidence.scanned_structure
        == (dialect["header_line"], dialect["units_line"])
        and undamaged,
        "data_lines": parsed
        and interpretation["data_lines"]["last"] is None
        and audit is not None
        and audit.complete
        and audit.clean,
    }


# ---------------------------------------------------------------------------
# Rows the data range leaves out
# ---------------------------------------------------------------------------

_SEPARATORS = {
    "above": {
        "header": "a header line",
        "units": "a units line",
        "blank": "a blank line",
        "text": "a text line",
        "nothing": "nothing (they are directly above the data)",
    },
    "below": {
        "header": "a text line",
        "units": "a text line",
        "blank": "a blank line",
        "text": "a text line",
        "nothing": "nothing (they are directly below the data)",
    },
}

#: The same, for a block of one row.
_SEPARATOR_ONE_ROW = {
    side: {
        **phrases,
        "nothing": f"nothing (it is directly {side} the data)",
    }
    for side, phrases in _SEPARATORS.items()
}

_SIDE_WORDS = {"above": "above the data start", "below": "below the data end"}

#: Blocks named one by one when more blocks are left out than an
#: acknowledgment can name; one summary finding counts the rest.
_NAMED_PAST_LIMIT = 10


def _listed(entries: list[str], more: int) -> str:
    return ", ".join(entries) + (f", and {more} more" if more else "")


def _composition(block: ExcludedBlock) -> str:
    damaged = [row for row in block.listed if row.kind == "damaged"]
    ragged = [row for row in block.listed if row.kind == "ragged"]
    parts = []
    if block.data_rows:
        parts.append(f"{block.data_rows} with the data's columns")
    if block.damaged:
        named = [f"line {r.line}, where field {r.field} {r.problem}" for r in damaged]
        parts.append(
            f"{block.damaged} damaged ({_listed(named, block.damaged - len(damaged))})"
        )
    if block.ragged:
        named = [
            f"line {r.line}, {r.count} field{'' if r.count == 1 else 's'}"
            for r in ragged
        ]
        parts.append(
            f"{block.ragged} with numbers in another number of fields "
            f"({_listed(named, block.ragged - len(ragged))})"
        )
    return "; ".join(parts)


def _excluded_finding(block: ExcludedBlock, rate: str, next_step: str) -> str:
    side = _SIDE_WORDS[block.side]
    if block.rows == 1:
        head = (
            f"Line {block.first_line} {side} holds 1 row that could be part of the "
            f"acquisition"
        )
    else:
        head = (
            f"Lines {block.first_line} to {block.last_line} {side} hold {block.rows} "
            f"rows that could be part of the acquisition"
        )
    seconds = three_significant(Decimal(block.rows) / Decimal(rate))
    if block.rows == 1:
        body = (
            f"At {rate} Hz it would span {seconds} s. It is separated from the data "
            f"rows by {_SEPARATOR_ONE_ROW[block.side][block.separator]}. The import "
            f"would leave this row out, so it might not be the complete acquisition."
        )
    else:
        body = (
            f"At {rate} Hz they would span {seconds} s. They are separated from the "
            f"data rows by {_SEPARATORS[block.side][block.separator]}. The import "
            f"would leave them out, so it might not be the complete acquisition."
        )
    return f"{head}: {_composition(block)}. {body} {next_step}"


def _too_many_step() -> str:
    return (
        f"This configuration leaves out more blocks than an acknowledgment can name "
        f"({MAX_EXCLUSIONS}), so no acknowledgment can resolve it: re-export the "
        f"acquisition so that these rows can be read, or configure a data line range "
        f"that leaves fewer blocks out, then validate again."
    )


def _remaining_summary(
    blocks: list[ExcludedBlock], tallies: tuple[BlockTally, ...]
) -> str:
    """The summary finding for every block past the named ones: exact
    counts, and per side the lines from the first to the last of them."""
    spans: dict[str, list[int]] = {}
    count = rows = 0
    pieces = [BlockTally(b.side, 1, b.rows, b.first_line, b.last_line) for b in blocks]
    pieces += tallies
    for piece in pieces:
        span = spans.setdefault(piece.side, [piece.first_line, piece.last_line])
        span[0] = min(span[0], piece.first_line)
        span[1] = max(span[1], piece.last_line)
        count += piece.blocks
        rows += piece.rows
    where = " and ".join(
        f"lines {spans[side][0]} to {spans[side][1]} {_SIDE_WORDS[side]}"
        for side in ("above", "below")
        if side in spans
    )
    return (
        f"Beyond the blocks named above, {where} hold {count} more blocks of rows "
        f"that could be part of the acquisition ({rows} rows in all); they are not "
        f"named one by one."
    )


def _ask_about(run: _Run, block: ExcludedBlock, rate: str, next_step: str) -> None:
    """One block's excluded_data_rows finding, asking with *next_step*."""
    run.issue(
        "excluded_data_rows",
        "needs_confirmation",
        _excluded_finding(block, rate, next_step),
        field_path="data_lines",
        line=block.first_line,
        token=_separator_token(block),
    )


def _report_past_the_limit(
    run: _Run, audit: RegionAudit, rate: str, next_step: str
) -> None:
    """More blocks than an acknowledgment can name: the first
    :data:`_NAMED_PAST_LIMIT` in line order, each with *next_step*, then one
    summary finding for the rest. Every finding asks; none can be resolved."""
    blocks = list(audit.blocks)
    for block in blocks[:_NAMED_PAST_LIMIT]:
        _ask_about(run, block, rate, next_step)
    rest = blocks[_NAMED_PAST_LIMIT:]
    run.issue(
        "excluded_data_rows",
        "needs_confirmation",
        _remaining_summary(rest, audit.tallies),
        field_path="data_lines",
        line=rest[0].first_line if rest else audit.tallies[0].first_line,
    )


def _acknowledge_step(block: ExcludedBlock, sha: str, import_id: str) -> str:
    if block.rows == 1:
        rows, they, them = "this row", "it", "it"
    else:
        rows, they, them = "these rows", "they", "them"
    return (
        f"Ask the user whether {rows} should be left out of the import. If {they} "
        f"should be imported, do not import: re-export the acquisition so that {they} "
        f"can be read. If {they} should be left out, the user can confirm the data "
        f"line range with a basis for 'data_lines' and acknowledge the exclusion in "
        f"attestation.exclusions with source_sha256 {sha}, import_id {import_id}, "
        f"first_line {block.first_line}, last_line {block.last_line}, rows "
        f"{block.rows} and their reason for leaving {them} out."
    )


def _excluded_notice(block: ExcludedBlock) -> str:
    if block.rows == 1:
        head = (
            f"Line {block.first_line} (1 row that could be part of the acquisition) "
            f"is left out"
        )
    else:
        head = (
            f"Lines {block.first_line} to {block.last_line} ({block.rows} rows that "
            f"could be part of the acquisition) are left out"
        )
    return (
        f"{head} as the user acknowledged; the acknowledgment and its reason are "
        f"recorded. This import is recorded as a partial acquisition and may not be "
        f"the complete acquisition."
    )


def _separator_token(block: ExcludedBlock):
    if block.separator in ("header", "units", "text") and block.separator_text:
        # The census kept the line stripped and cut one character past the
        # cap; stripping it again could drop whitespace at the cut and, with
        # it, the truncation flag.
        return FileText.of(block.separator_text, cap=PREAMBLE_LINE_CHARS)
    return None


def _dialect_gate(
    run: _Run, evidence: _Evidence, proofs: Mapping[str, bool]
) -> Optional[str]:
    """Why no acknowledgment may resolve an exclusion yet, or None: the
    dialect or the encoding still needs confirmation, or the scanned rows
    read better under another delimiter and decimal mark than the configured
    ones (tied readings settled by a dialect basis pass)."""
    dialect = run.interpretation["dialect"]
    unique = evidence.unique_reading
    if unique is not None and unique[:2] != (dialect["delimiter"], dialect["decimal"]):
        return (
            f"the scanned rows read better with {_delimiter_words(unique[0])} between "
            f"fields and '{unique[1]}' as the decimal mark than with the configured "
            f"reading"
        )
    if not proofs["dialect"] and run.basis("dialect") is None:
        if unique is not None and evidence.encoding == "cp1252":
            return "the text encoding needs confirmation"
        return "the delimiter and decimal mark need confirmation"
    return None


def _check_excluded_rows(
    run: _Run,
    audit: RegionAudit,
    decode: DecodeResult,
    evidence: _Evidence,
    proofs: Mapping[str, bool],
    import_id: str,
) -> list[ExcludedBlock]:
    """Every row the data range leaves out that could be part of the
    acquisition asks before ``ready``; only a basis for ``data_lines``
    together with an acknowledgment bound to this file, this interpretation
    and the exact block resolves it, kept as a notice. Returns the resolved
    blocks."""
    rate = run.interpretation["time_base"]["sampling_rate_hz"]
    blocks = list(audit.blocks)
    # The dialect gate first: past the limit too, an unsettled dialect is the
    # step the named blocks give.
    gate = _dialect_gate(run, evidence, proofs)
    gate_step = (
        None
        if gate is None
        else f"No acknowledgment can resolve this while {gate}: settle the dialect "
        f"first, then validate again."
    )
    if audit.block_count > MAX_EXCLUSIONS:
        _report_past_the_limit(run, audit, rate, gate_step or _too_many_step())
        return []
    if gate_step is not None:
        for block in blocks:
            _ask_about(run, block, rate, gate_step)
        return []
    by_lines = {(b.first_line, b.last_line): b for b in blocks}
    matched: set[tuple[int, int]] = set()
    for ack in run.attestation.get("exclusions") or []:
        first, last = ack["first_line"], ack["last_line"]
        block = by_lines.get((first, last))
        span = f"line {first}" if first == last else f"lines {first} to {last}"
        problem: Optional[str] = None
        if ack["source_sha256"] != decode.source_sha256:
            problem = (
                f"The exclusion acknowledgment for {span} was given for another file "
                f"(source_sha256 {ack['source_sha256']}). Remove it, or validate the "
                f"file it was given for."
            )
        elif ack["import_id"] != import_id:
            problem = (
                f"The exclusion acknowledgment for {span} was given for another "
                f"configuration of this file (import_id {ack['import_id']}). Validate "
                f"this configuration and acknowledge the rows it leaves out."
            )
        elif not blocks:
            problem = (
                f"The exclusion acknowledgment names {span}, but this configuration "
                f"leaves out no row that could be part of the data. Remove it."
            )
        elif block is None:
            problem = (
                f"The exclusion acknowledgment names {span}, which this configuration "
                f"does not leave out as one block. Use the lines the "
                f"excluded_data_rows finding gives."
            )
        elif ack["rows"] != block.rows:
            named = "1 row" if ack["rows"] == 1 else f"{ack['rows']} rows"
            problem = (
                f"The exclusion acknowledgment names {named} for {span}, where this "
                f"configuration leaves out {block.rows}. Use the count the "
                f"excluded_data_rows finding gives."
            )
        if problem is not None:
            run.issue(
                "basis_mismatch",
                "invalid",
                problem,
                field_path="attestation.exclusions",
                line=first,
            )
        else:
            matched.add((first, last))
    based = run.basis("data_lines") is not None
    resolved: list[ExcludedBlock] = []
    for block in blocks:
        if based and (block.first_line, block.last_line) in matched:
            run.issue(
                "excluded_data_rows",
                "notice",
                _excluded_notice(block),
                field_path="data_lines",
                line=block.first_line,
                token=_separator_token(block),
            )
            resolved.append(block)
        else:
            _ask_about(
                run,
                block,
                rate,
                _acknowledge_step(block, decode.source_sha256, import_id),
            )
    if blocks and len(resolved) == len(blocks):
        _check_instant_beside_exclusions(run, audit, evidence)
    return resolved


def _check_instant_beside_exclusions(
    run: _Run, audit: RegionAudit, evidence: _Evidence
) -> None:
    """An acquisition instant accepted from the preamble may describe a row
    the data range leaves out: it needs its own confirmation when such a row
    lies above the run inspection found under the configured reading, or no
    unique reading exists."""
    if run.interpretation["defaults"]["acquired_at"] is None:
        return
    entry = run.basis("defaults.acquired_at")
    if entry is None or entry["basis"] != "user_accepted_file_value":
        return
    if run.override("defaults.acquired_at") is not None:
        return
    above = [b for b in audit.blocks if b.side == "above"]
    if not above:
        return
    dialect = run.interpretation["dialect"]
    unique = evidence.unique_reading
    found = (
        unique is not None
        and unique[:2] == (dialect["delimiter"], dialect["decimal"])
        and not audit.rows_above(unique[2])
    )
    if found:
        return
    first = run.interpretation["data_lines"]["first"]
    left_out = sum(b.rows for b in above)
    if left_out == 1:
        rows, which = "1 row", "that row"
    else:
        rows, which = f"{left_out} rows", "one of those rows"
    run.issue(
        "printed_value_unverified",
        "needs_confirmation",
        f"The data start leaves out {rows} above it that could be part of the "
        f"acquisition, so the acquisition instant printed on line {entry['line']} "
        f"may describe {which} rather than the first imported sample. Ask the user "
        f"whether it applies to line {first}; to keep it, name an override for "
        f"'defaults.acquired_at' with a reason.",
        field_path="defaults.acquired_at",
        line=entry["line"],
    )


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
