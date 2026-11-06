"""
Attestation of a signal import: who confirmed what, on which basis.

ISO 13374 Block 1 (data acquisition). An import *configuration* is an
interpretation (``import_config``: how the bytes become signals) plus an
attestation: the basis the user gave for each value the import writes, the
overrides of file declarations with their reasons, and an optional free-text
``declared_by``. The attestation never enters an identity: re-confirming the
same interpretation through another channel leaves every ``import_id`` and
``measurement_id`` unchanged. It enters only the ``import_digest``, which
binds an import to the exact bytes and configuration that were validated.

This module owns:

- the attestation vocabularies: origins, bases, confirmation channels and
  transports, each defined here once;
- the closed grammar of the field paths a basis or an override names;
- :func:`normalize_configuration`, which returns the canonical form of a
  configuration or refuses it with every problem in one message;
- :func:`import_digest`.

Canonical form (schema ``pmm-text-import-config/1``)::

    {
      "schema": "pmm-text-import-config/1",
      "interpretation": {...},          # import_config canonical form
      "attestation": {
        "schema": "pmm-import-attestation/1",
        "bases": {
          "time_base": {"basis": "user_stated", "line": null,
                        "column": null, "import_id": null,
                        "declaration_version": null},
          "channels[3].unit": {"basis": "user_accepted_file_value",
                               "line": 8, ...}
        },
        "overrides": {"channels[3].unit": "header unit is a typo"},
        "declared_by": "J. Engineer"
      }
    }

A basis names the user's act, never the server's or the assistant's: a value
the file states, the data derive, an earlier import or the point declaration
holds is accepted only by citing that evidence (a preamble or units line, a
time column, an ``import_id``, a point declaration version), and validation
checks that the configured value equals what the cited evidence holds. A
basis that arrives as an MCP tool argument is recorded as caller-asserted:
it states what the caller claimed, never who the caller is.
"""

import re
from collections.abc import Mapping
from typing import Any, Optional

from ..canonical_json import canonical_json, content_hash
from ..file_text import escape_file_text
from ..signal_acquisition.import_config import (
    InterpretationError,
    InterpretationProblem,
    is_import_id,
    normalize_interpretation,
)
from ..signal_acquisition.measurement import validate_free_text

__all__ = [
    "CONFIGURATION_SCHEMA",
    "ATTESTATION_SCHEMA",
    "ORIGINS",
    "BASES",
    "BASIS_ORIGIN",
    "CONFIRMATION_CHANNELS",
    "SUPPORTED_CHANNELS",
    "CALLER_ASSERTED_CHANNELS",
    "TRANSPORTS",
    "BASIS_CITATIONS",
    "OVERRIDABLE_FIELDS",
    "ConfigurationError",
    "parse_field",
    "is_caller_asserted",
    "check_channel",
    "check_transport",
    "normalize_configuration",
    "import_digest",
]

#: Schema id of the canonical configuration this version writes and reads.
CONFIGURATION_SCHEMA = "pmm-text-import-config/1"

#: Schema id of the canonical attestation.
ATTESTATION_SCHEMA = "pmm-import-attestation/1"

#: Where a configured value comes from: stated in the file, derived from the
#: data, an import profile, an earlier import of the same bytes, the
#: measurement point's declaration, or stated by the user.
ORIGINS: tuple[str, ...] = (
    "file_declared",
    "derived",
    "profile",
    "prior_import",
    "point_declaration",
    "stated",
)

#: The user's act behind a configured value.
BASES: tuple[str, ...] = (
    "user_stated",
    "user_accepted_file_value",
    "user_accepted_derived_value",
    "user_accepted_profile_value",
    "user_accepted_prior_value",
    "user_accepted_point_value",
)

#: The origin each basis attests.
BASIS_ORIGIN: dict[str, str] = {
    "user_stated": "stated",
    "user_accepted_file_value": "file_declared",
    "user_accepted_derived_value": "derived",
    "user_accepted_profile_value": "profile",
    "user_accepted_prior_value": "prior_import",
    "user_accepted_point_value": "point_declaration",
}

#: The citation key each basis requires (None: no citation).
BASIS_CITATIONS: dict[str, Optional[str]] = {
    "user_stated": None,
    "user_accepted_file_value": "line",
    "user_accepted_derived_value": "column",
    "user_accepted_profile_value": None,
    "user_accepted_prior_value": "import_id",
    "user_accepted_point_value": "declaration_version",
}

#: Through which channel a confirmation arrived. The calling adapter sets
#: it; no request field can. ``mcp_elicitation`` is reserved.
CONFIRMATION_CHANNELS: tuple[str, ...] = (
    "mcp_tool_argument",
    "playground_ui",
    "python_api",
    "mcp_elicitation",
)

#: The confirmation channels this version accepts.
SUPPORTED_CHANNELS: tuple[str, ...] = (
    "mcp_tool_argument",
    "playground_ui",
    "python_api",
)

#: Channels whose bases are what the caller claimed: the server cannot tell
#: an assistant's argument from the engineer's words.
CALLER_ASSERTED_CHANNELS: tuple[str, ...] = ("mcp_tool_argument",)

#: The MCP transports a confirmation can arrive over.
TRANSPORTS: tuple[str, ...] = ("stdio", "sse", "streamable-http")

#: Fields an override may name: the stated value contradicts a file
#: declaration (a unit token, a printed rate, speed or instant), or a time
#: candidate column is left unassessed.
OVERRIDABLE_FIELDS: tuple[str, ...] = (
    "channel_unit",
    "time_base",
    "time_column",
    "defaults.rpm",
    "defaults.acquired_at",
)

_ATTESTATION_FIELDS = ("schema", "bases", "overrides", "declared_by")
_CONFIGURATION_FIELDS = ("schema", "interpretation", "attestation")
_ENTRY_FIELDS = ("basis", "line", "column", "import_id", "declaration_version")
_CHANNEL_PARTS = ("unit", "scaling", "dc_policy", "binding")
_DEFAULT_FIELDS = ("acquired_at", "rpm", "load", "operating_state")
_MAX_FIELD_CHARS = 64
_MAX_BASES = 4096
_LINE_LIMIT = 2**53

_FIELD = re.compile(
    r"(?P<plain>dialect|data_lines|time_base|companion)"
    r"|defaults\.(?P<default>acquired_at|rpm|load|operating_state)"
    r"|channels\[(?P<channel>0|[1-9][0-9]{0,3})\]\.(?P<part>unit|scaling|dc_policy|binding)"
    r"|time_columns\[(?P<time>0|[1-9][0-9]{0,3})\]"
)

_DIGEST_DOMAIN = "pmm-import-digest/1"

_REMEDY = "Correct every listed field and validate the configuration again."


class ConfigurationError(InterpretationError):
    """A refused configuration: one message, every problem typed.

    The interpretation's problems keep their paths under ``interpretation``;
    the attestation's under ``attestation``. A subclass of
    :class:`~import_config.InterpretationError`, so a caller that handles a
    refused interpretation handles a refused configuration too.
    """

    def __init__(self, problems: list[InterpretationProblem]) -> None:
        ValueError.__init__(
            self,
            "Invalid import configuration: "
            + "; ".join(f"{p.path} {p.text}" for p in problems)
            + f". {_REMEDY}",
        )
        self.problems = tuple(problems)


def parse_field(path: object) -> Optional[tuple[str, Optional[int]]]:
    """``(kind, column)`` of a field path, or None outside the grammar.

    Kinds: ``dialect``, ``data_lines``, ``time_base``, ``companion``,
    ``defaults.<field>``, ``channel_<part>`` (unit, scaling, dc_policy,
    binding) and ``time_column``; *column* is the physical column of the
    two indexed kinds.
    """
    if not isinstance(path, str) or len(path) > _MAX_FIELD_CHARS:
        return None
    match = _FIELD.fullmatch(path)
    if match is None:
        return None
    if match["plain"]:
        return match["plain"], None
    if match["default"]:
        return f"defaults.{match['default']}", None
    if match["channel"] is not None:
        return f"channel_{match['part']}", int(match["channel"])
    return "time_column", int(match["time"])


def is_caller_asserted(channel: str) -> bool:
    """Whether bases from *channel* record only what the caller claimed."""
    return channel in CALLER_ASSERTED_CHANNELS


def check_channel(channel: object) -> str:
    """The confirmation channel an adapter passed.

    Raises:
        ValueError: For a channel outside :data:`SUPPORTED_CHANNELS`
            (``mcp_elicitation`` is reserved for a later version).
    """
    if channel == "mcp_elicitation":
        raise ValueError(
            "confirmation channel 'mcp_elicitation' is reserved: this version "
            "takes confirmations as tool arguments, from the playground or "
            "from the Python API."
        )
    if not isinstance(channel, str) or channel not in SUPPORTED_CHANNELS:
        raise ValueError(
            f"confirmation channel must be one of {list(SUPPORTED_CHANNELS)}; "
            f"the calling adapter sets it."
        )
    return channel


def check_transport(transport: object) -> Optional[str]:
    """The MCP transport a confirmation arrived over, or None.

    Raises:
        ValueError: For a transport outside :data:`TRANSPORTS`.
    """
    if transport is None:
        return None
    if not isinstance(transport, str) or transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {list(TRANSPORTS)} or None.")
    return transport


def _shown(value: object) -> str:
    if isinstance(value, str):
        text, truncated = escape_file_text(value, cap=48)
        return f"'{text}{'...' if truncated else ''}'"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    return f"a {type(value).__name__}"


class _Problems:
    def __init__(self) -> None:
        self.items: list[InterpretationProblem] = []

    def add(self, path: str, text: str, kind: str = "invalid") -> None:
        self.items.append(InterpretationProblem(path, kind, text))


def _free_text(
    field: str, value: object, path: str, problems: _Problems
) -> Optional[str]:
    try:
        return validate_free_text(field, value)
    except ValueError as exc:
        text, truncated = escape_file_text(str(exc), cap=400)
        problems.add(path, f"is refused: {text}{'...' if truncated else ''}")
        return None


def _positive_int(value: object, path: str, problems: _Problems) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        problems.add(path, f"is {_shown(value)}: pass a positive integer")
        return None
    if not 1 <= value < _LINE_LIMIT:
        problems.add(path, f"{value} is out of range: pass a positive integer")
        return None
    return value


def _entry(value: object, path: str, problems: _Problems) -> Optional[dict[str, Any]]:
    """One basis entry: the basis and the one citation it requires."""
    if isinstance(value, str):
        value = {"basis": value}
    if not isinstance(value, Mapping):
        problems.add(
            path,
            f"is {_shown(value)}: pass an object with the keys {list(_ENTRY_FIELDS)}",
        )
        return None
    unknown = [key for key in value if key not in _ENTRY_FIELDS]
    for key in unknown:
        problems.add(
            path,
            f"has the unknown key {_shown(key)}: the allowed keys are "
            f"{list(_ENTRY_FIELDS)}",
            kind="unknown_key",
        )
    basis = value.get("basis")
    if basis is None:
        problems.add(f"{path}.basis", "is required", kind="missing")
        return None
    if basis not in BASES:
        problems.add(f"{path}.basis", f"{_shown(basis)} is not one of {list(BASES)}")
        return None
    if basis == "user_accepted_profile_value":
        problems.add(
            f"{path}.basis",
            "names an import profile, which this version does not offer: state "
            "the value, or accept it from the file, the data, an earlier import "
            "or the point declaration",
        )
        return None
    entry: dict[str, Any] = {key: None for key in _ENTRY_FIELDS}
    entry["basis"] = basis
    required = BASIS_CITATIONS[basis]
    for key in ("line", "column", "import_id", "declaration_version"):
        given = value.get(key)
        if key == required:
            if given is None:
                problems.add(
                    f"{path}.{key}",
                    f"is required by the basis {basis!r}: cite the evidence the "
                    f"accepted value comes from",
                    kind="missing",
                )
                continue
            if key == "import_id":
                if not is_import_id(given):
                    problems.add(
                        f"{path}.import_id",
                        f"{_shown(given)} is not an import_id: pass exactly 64 "
                        f"lowercase hexadecimal characters",
                    )
                    continue
                entry[key] = given
            elif key == "column":
                if isinstance(given, bool) or not isinstance(given, int):
                    problems.add(
                        f"{path}.column", f"is {_shown(given)}: pass an integer"
                    )
                    continue
                if not 0 <= given < 1024:
                    problems.add(f"{path}.column", f"{given} is out of range")
                    continue
                entry[key] = given
            else:
                entry[key] = _positive_int(given, f"{path}.{key}", problems)
        elif given is not None:
            problems.add(
                f"{path}.{key}",
                f"is set but the basis {basis!r} cites no {key}: remove it, or "
                f"name the basis that cites one",
            )
    return entry


def _check_field(
    path: str,
    parsed: tuple[str, Optional[int]],
    interpretation: Optional[Mapping[str, Any]],
    problems: _Problems,
    *,
    for_override: bool,
) -> bool:
    """Whether *path* names a value the interpretation has to confirm."""
    kind, column = parsed
    if interpretation is None:
        return True  # the interpretation's own problems are reported already
    channels = {c["column"]: c for c in interpretation["channels"]}
    if kind.startswith("channel_"):
        channel = channels.get(column)
        if channel is None:
            problems.add(
                path,
                f"names column {column}, which no channel of the interpretation "
                f"reads",
            )
            return False
        part = kind[len("channel_") :]
        if part in ("scaling", "dc_policy") and channel["scaling"] is None:
            problems.add(path, f"names the scaling of column {column}, which has none")
            return False
        if part == "dc_policy" and channel["scaling"]["dc_policy"] is None:
            problems.add(
                path,
                f"names the DC policy of column {column}, which states none: set "
                f"channels' scaling.dc_policy first",
            )
            return False
        if part == "binding" and channel["binding"] is None:
            problems.add(path, f"names the binding of column {column}, which has none")
            return False
    elif kind.startswith("defaults."):
        if interpretation["defaults"][kind[len("defaults.") :]] is None:
            problems.add(
                path, "names a file-level default the interpretation leaves empty"
            )
            return False
    elif kind == "time_column":
        if not for_override:
            problems.add(
                path,
                "takes an override (a reason to leave the column unassessed), not "
                "a basis",
            )
            return False
        listed = {c["column"] for c in interpretation["time_columns"]}
        if column in listed or column in channels:
            problems.add(
                path,
                f"names column {column}, which the interpretation already reads; "
                f"an override leaves only an unlisted time column unassessed",
            )
            return False
    if for_override and kind not in OVERRIDABLE_FIELDS:
        problems.add(
            path,
            f"takes no override: an override is named for a channel's unit, the "
            f"time base, the rpm or acquisition instant default, or an unlisted "
            f"time column (fields {list(OVERRIDABLE_FIELDS)})",
        )
        return False
    return True


def _normalize_attestation(
    value: object,
    interpretation: Optional[Mapping[str, Any]],
    problems: _Problems,
) -> Optional[dict[str, Any]]:
    path = "attestation"
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        problems.add(path, f"is {_shown(value)}: pass an object")
        return None
    for key in value:
        if key not in _ATTESTATION_FIELDS:
            problems.add(
                path,
                f"has the unknown key {_shown(key)}: the allowed keys are "
                f"{list(_ATTESTATION_FIELDS)}",
                kind="unknown_key",
            )
    schema = value.get("schema", ATTESTATION_SCHEMA)
    if schema != ATTESTATION_SCHEMA:
        problems.add(
            f"{path}.schema",
            f"{_shown(schema)} is not {ATTESTATION_SCHEMA!r}, the only schema this "
            f"version reads",
        )
    bases: dict[str, dict[str, Any]] = {}
    given_bases = value.get("bases")
    if given_bases is None:
        given_bases = {}
    if not isinstance(given_bases, Mapping):
        problems.add(f"{path}.bases", f"is {_shown(given_bases)}: pass an object")
    elif len(given_bases) > _MAX_BASES:
        problems.add(
            f"{path}.bases", f"names {len(given_bases)} fields, over {_MAX_BASES}"
        )
    else:
        for field, entry_value in given_bases.items():
            entry_path = f"{path}.bases[{_shown(field)}]"
            parsed = parse_field(field)
            if parsed is None:
                problems.add(
                    entry_path,
                    "is not a field this version confirms: name 'dialect', "
                    "'data_lines', 'time_base', 'companion', 'defaults.<field>' or "
                    "'channels[<column>].<unit|scaling|dc_policy|binding>'",
                )
                continue
            entry = _entry(entry_value, entry_path, problems)
            if entry is not None and _check_field(
                field, parsed, interpretation, problems, for_override=False
            ):
                bases[field] = entry
    overrides: dict[str, str] = {}
    given_overrides = value.get("overrides")
    if given_overrides is None:
        given_overrides = {}
    if not isinstance(given_overrides, Mapping):
        problems.add(
            f"{path}.overrides", f"is {_shown(given_overrides)}: pass an object"
        )
    else:
        for field, reason in given_overrides.items():
            entry_path = f"{path}.overrides[{_shown(field)}]"
            parsed = parse_field(field)
            if parsed is None:
                problems.add(entry_path, "is not a field this version confirms")
                continue
            if not _check_field(
                field, parsed, interpretation, problems, for_override=True
            ):
                continue
            if reason is None:
                problems.add(entry_path, "needs a reason", kind="missing")
                continue
            text = _free_text("override reason", reason, entry_path, problems)
            if text is not None:
                overrides[field] = text
    declared_by = value.get("declared_by")
    if declared_by is not None:
        declared_by = _free_text(
            "declared_by", declared_by, f"{path}.declared_by", problems
        )
    return {
        "schema": ATTESTATION_SCHEMA,
        "bases": dict(sorted(bases.items())),
        "overrides": dict(sorted(overrides.items())),
        "declared_by": declared_by,
    }


def normalize_configuration(configuration: object) -> dict[str, Any]:
    """Return the canonical form of an import configuration.

    Pure function. Idempotent: the canonical form normalizes to itself.

    Args:
        configuration: A mapping in the shape of the module docstring.

    Returns:
        ``{"schema", "interpretation", "attestation"}``, each canonical.

    Raises:
        ConfigurationError: Every problem of the interpretation (paths under
            ``interpretation.``) and of the attestation, in one message: an
            unknown key, a missing required value, a basis outside the
            vocabulary or without the citation it requires, an import profile
            basis (not offered in this version), a field path outside the
            grammar or naming a value the interpretation does not hold, an
            override on a field that takes none or without a reason, and
            free text the measurement contract's single rule refuses.
    """
    problems = _Problems()
    if not isinstance(configuration, Mapping):
        problems.add(
            "configuration",
            f"is {_shown(configuration)}: pass an object with the keys "
            f"{list(_CONFIGURATION_FIELDS)}",
        )
        raise ConfigurationError(problems.items)
    for key in configuration:
        if key not in _CONFIGURATION_FIELDS:
            problems.add(
                "configuration",
                f"has the unknown key {_shown(key)}: the allowed keys are "
                f"{list(_CONFIGURATION_FIELDS)}",
                kind="unknown_key",
            )
    schema = configuration.get("schema")
    if schema is None:
        problems.add("schema", "is required", kind="missing")
    elif schema != CONFIGURATION_SCHEMA:
        problems.add(
            "schema",
            f"{_shown(schema)} is not {CONFIGURATION_SCHEMA!r}, the only schema "
            f"this version reads",
        )
    interpretation: Optional[dict[str, Any]] = None
    if configuration.get("interpretation") is None:
        problems.add("interpretation", "is required", kind="missing")
    else:
        try:
            interpretation = normalize_interpretation(configuration["interpretation"])
        except InterpretationError as exc:
            for problem in exc.problems:
                problems.add(
                    f"interpretation.{problem.path}", problem.text, problem.kind
                )
    attestation = _normalize_attestation(
        configuration.get("attestation"), interpretation, problems
    )
    if problems.items:
        raise ConfigurationError(problems.items)
    return {
        "schema": CONFIGURATION_SCHEMA,
        "interpretation": interpretation,
        "attestation": attestation,
    }


def import_digest(
    import_id: str,
    attestation: Mapping[str, Any],
    *,
    channel: str,
    transport: Optional[str],
) -> str:
    """SHA-256 (64 hex) binding an import to what was validated.

    Covers the ``import_id`` (the file's bytes and the interpretation), the
    canonical attestation, and the confirmation channel and transport, so an
    import through another channel, or with another basis or override,
    validates again first. Canonical JSON escapes non-ASCII free text (a
    declared deviation from RFC 8785, as in the ledger).

    Raises:
        ValueError: For a malformed ``import_id``, channel or transport.
    """
    if not is_import_id(import_id):
        raise ValueError("import_id must be 64 lowercase hexadecimal characters.")
    payload = {
        "domain": _DIGEST_DOMAIN,
        "import_id": import_id,
        "attestation": dict(attestation),
        "channel": check_channel(channel),
        "transport": check_transport(transport),
    }
    canonical_json(payload)  # refuses a value that is not plain JSON
    return content_hash(payload)
