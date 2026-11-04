"""
Measurement declaration contract of the companion ``<stem>_metadata.json``.

ISO 13374 Block 1 (data acquisition): the identity of a measurement, i.e.
"who measured what, where and when, on which asset", is DECLARED by the
producer of the file in a ``"measurement"`` object of the companion and
validated here with the same discipline as the raw-binary declaration:
explicit > companion > refusal with a remedy, nothing inferred from file
names or signal content.

Leaf module: only the standard library and :mod:`..path_safety`. The signal
repository calls it while preparing an entry (before insertion, so batches
stay all-or-nothing), the asset ledger imports its vocabularies, and the
documentation guard derives the declaration table from it.

Companion contract (the presence of the object activates it)::

    {
      "sampling_rate": 26585.3,                    # honored as today
      "signal_unit": "g",                          # honored as today
      "measurement": {
        "asset_id": "P-101",                       # required, ledger id
        "measurement_point_id": "motor_de_h",      # required, ledger id
        "acquired_at": "2026-08-20T13:42:00+02:00",  # required, ISO 8601
        "rpm": 1482, "load": 75, "operating_state": "loaded",
        "sensor_id": "STWIN_BOX_001", "direction": "horizontal",
        "declared_by": "adapter:stwinbox"
      }
    }

Free-form keys stay at the top level of the companion (they pass verbatim
into ``source_metadata``); inside the object only the fields listed in
``MEASUREMENT_FIELDS`` are accepted, so a misspelled field is refused
instead of silently dropped. ``sampling_rate`` and ``signal_unit`` stay
where they are today. Historical top-level keys such as ``shaft_speed``
(Hz) or ``load`` are never interpreted as measurement fields.

The signal import binds a measurement by the same rules (plan KTD15): one
channel's binding (measurement point, direction, sensor, an asset override)
over the import's file-level defaults (asset, acquisition instant, rpm,
load, operating state), with refusals that name "import configuration,
channel <index>" instead of a companion (``source_kind="import"``, see
:data:`SOURCE_KINDS`). Every function takes plain values, never an import
configuration object, so the module stays a leaf.
"""

import decimal
import hashlib
import math
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..path_safety import validate_name_component

__all__ = [
    "MEASUREMENT_KEY",
    "VALID_DIRECTIONS",
    "DIRECTION_ALIASES",
    "REQUIRED_MEASUREMENT_FIELDS",
    "MEASUREMENT_FIELD_DEFAULTS",
    "MEASUREMENT_FIELDS",
    "FREE_TEXT_FIELDS",
    "MAX_FREE_TEXT_CHARS",
    "MAX_LEDGER_ID_CHARS",
    "UNIT_FAMILIES",
    "SIGNAL_UNIT_ALIASES",
    "WIN32_RESERVED_NAMES",
    "LEDGER_ID_GRAMMAR",
    "MEASUREMENT_ID_HEX_CHARS",
    "MEASUREMENT_DECLARATION_KEYS",
    "MEASUREMENT_IDENTITY_KEYS",
    "SOURCE_KINDS",
    "IMPORT_BINDING_FIELDS",
    "IMPORT_DEFAULT_FIELDS",
    "TRIM_SHIFT_KEYS",
    "unit_family",
    "validate_ledger_id",
    "validate_free_text",
    "normalize_direction",
    "normalize_acquired_at",
    "validate_measurement_declaration",
    "digest_file",
    "measurement_id_from_digest",
    "compute_measurement_id",
    "build_measurement_identity",
    "build_import_identity",
    "binding_collisions",
    "shift_acquired_at",
]

#: Companion key whose presence activates the measurement contract.
MEASUREMENT_KEY = "measurement"

#: Closed direction vocabulary. ``x``/``y``/``z`` (sensor axes) and
#: ``horizontal``/``vertical``/``axial`` (machine directions) are DIFFERENT
#: declarations: an axis is a direction only if the producer says so.
VALID_DIRECTIONS: tuple[str, ...] = ("horizontal", "vertical", "axial", "x", "y", "z")

#: Form-only aliases (the initial of the word, any letter case). There is
#: deliberately no alias between the two halves of the vocabulary.
DIRECTION_ALIASES: dict[str, str] = {"h": "horizontal", "v": "vertical", "a": "axial"}

#: Fields the object MUST declare to be accepted.
REQUIRED_MEASUREMENT_FIELDS: tuple[str, ...] = (
    "asset_id",
    "measurement_point_id",
    "acquired_at",
)

#: Effective defaults of the OPTIONAL fields (every one "not declared").
#: Encoded like ``loaders.RAW_PARAM_DEFAULTS``: a field is required exactly
#: when it is absent from this mapping, so the documentation guard and the
#: validator read optional-ness from one place.
MEASUREMENT_FIELD_DEFAULTS: dict[str, object] = {
    "rpm": None,
    "load": None,
    "operating_state": None,
    "sensor_id": None,
    "direction": None,
    "declared_by": None,
}

#: Every accepted field of the object, required first, in canonical order.
MEASUREMENT_FIELDS: tuple[str, ...] = (
    *REQUIRED_MEASUREMENT_FIELDS,
    *MEASUREMENT_FIELD_DEFAULTS,
)

#: Free-text fields: bounded, one line, never interpreted by the server.
FREE_TEXT_FIELDS: tuple[str, ...] = ("operating_state", "sensor_id", "declared_by")

#: Length cap of the free-text fields, in characters.
MAX_FREE_TEXT_CHARS = 200

#: Ledger ids become file names (``data/ledger/<asset_id>.jsonl``); the cap
#: keeps them well inside every filesystem's component limit.
MAX_LEDGER_ID_CHARS = 100

#: Unit families: a PARTITION of the repository's ``VALID_SIGNAL_UNITS``
#: (asserted by test, which imports both; the repository is not imported
#: here so this module stays a leaf). Measurements in different families
#: are never comparable; within a family a conversion is recorded.
UNIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "acceleration": ("g", "m/s2"),
    "velocity": ("mm/s", "m/s"),
}

#: Accepted spellings of a signal unit, mapped onto the canonical vocabulary
#: (applied after ``strip().lower()``). The one alias table: the repository's
#: ``normalize_signal_unit`` and the signal import's unit-token evidence both
#: read it, so a spelling cannot be accepted on one route and not the other.
SIGNAL_UNIT_ALIASES: dict[str, str] = {
    "m/s²": "m/s2",
    "m/s^2": "m/s2",
    "mm/sec": "mm/s",
}

#: Win32 device names. Refused on EVERY operating system, with or without
#: an extension: a ledger can be copied to a Windows machine, where a bare
#: ``NUL`` is the device (``open("NUL", "ab")`` succeeds and the bytes
#: vanish) and older builds treat ``NUL.jsonl`` the same way; newer Windows
#: 11 builds create a real file instead, which is no safer to rely on.
WIN32_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)

#: The grammar of a ledger id, spelled out in every refusal.
LEDGER_ID_GRAMMAR = (
    "start with a letter or digit, use only letters, digits, '_', '-' and "
    "'.', end with neither a dot nor a space, have at most "
    f"{MAX_LEDGER_ID_CHARS} characters, and not be a Windows reserved device "
    "name (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9, with or without an "
    "extension)"
)

#: Length of ``measurement_id``: the first 16 hex digits of a SHA-256.
MEASUREMENT_ID_HEX_CHARS = 16

#: Keys of the normalized declaration (what the pure validator returns).
MEASUREMENT_DECLARATION_KEYS: tuple[str, ...] = (
    "asset_id",
    "measurement_point_id",
    "acquired_at",
    "timezone_declared",
    "timestamp_suspect",
    "rpm",
    "load",
    "operating_state",
    "sensor_id",
    "direction",
    "declared_by",
)

#: Keys of the full identity block stored on ``StoredSignalInfo.measurement``:
#: the declaration plus the four file-derived keys (the identity hash, the
#: decoded channel, and the full content digest with the file size, which
#: the asset ledger records so a moved file can be re-verified without
#: hashing it twice at load time).
MEASUREMENT_IDENTITY_KEYS: tuple[str, ...] = (
    *MEASUREMENT_DECLARATION_KEYS,
    "measurement_id",
    "channel_index",
    "content_sha256",
    "size_bytes",
)

#: Instants before this are flagged ``timestamp_suspect``: a real-time clock
#: that was never set reports the 1970 epoch plus its uptime.
_EPOCH_YEAR_END = datetime(1971, 1, 1, tzinfo=timezone.utc)

#: Instants further than this after "now" are flagged ``timestamp_suspect``.
_FUTURE_TOLERANCE = timedelta(days=1)

_EXAMPLE_OBJECT = (
    '{"asset_id": "P-101", "measurement_point_id": "motor_de_h", '
    '"acquired_at": "2026-08-20T13:42:00+02:00"}'
)

_REMEDY = (
    'Fix the "measurement" object and re-load the file, or remove the object '
    "to load the file without asset identity."
)

_ACQUIRED_AT_HELP = (
    "declare an ISO 8601 timestamp string such as '2026-08-20T13:42:00+02:00' "
    "(an offset or 'Z' declares the timezone; without one the instant is "
    "ordered as UTC and flagged timezone_declared=false)"
)

#: Where a declaration comes from: a companion ``<stem>_metadata.json``, or
#: one channel of a signal import. Each kind selects a whole message bundle
#: (framing and remedies); the field rules and their messages are shared.
SOURCE_KINDS: tuple[str, ...] = ("companion", "import")

#: Fields of one import channel's ``binding`` (``asset_id`` overrides the
#: file-level default for that channel). Equal to the import contract's
#: binding fields (asserted by test; ``import_config`` is not imported here).
IMPORT_BINDING_FIELDS: tuple[str, ...] = (
    "measurement_point_id",
    "direction",
    "sensor_id",
    "asset_id",
)

#: Fields of an import's file-level ``defaults``, which every bound channel
#: inherits. Equal to the import contract's ``DECLARATION_DEFAULTS`` keys.
IMPORT_DEFAULT_FIELDS: tuple[str, ...] = (
    "asset_id",
    "acquired_at",
    "rpm",
    "load",
    "operating_state",
)

#: Keys of :func:`shift_acquired_at`'s result, in order.
TRIM_SHIFT_KEYS: tuple[str, ...] = (
    "acquired_at",
    "offset_samples",
    "shift_seconds",
    "exact_shift_seconds",
    "shift_exact",
)

_IMPORT_REMEDY = (
    "Fix the channel's binding or the file-level defaults in the import "
    "configuration and validate it again, or remove the channel's binding to "
    "import the channel without asset identity."
)

# One bundle per source kind: ``where`` opens every refusal of the object,
# the others are the problems only the framing can word. The companion
# bundle is the wording that existed before imports did, byte for byte.
_MESSAGES: dict[str, dict[str, str]] = {
    "companion": {
        "where": 'Invalid "measurement" object in companion {source}: ',
        "absent": (
            'Companion {source} declares no "measurement" object: nothing to '
            "validate. Add the object to declare the asset identity of the "
            "file (required fields: {required})."
        ),
        "not_object": "expected a JSON object with the fields {fields}, got {kind}",
        "unknown": (
            "unknown field(s) {unknown}: the object accepts only {fields}; keep "
            "free-form keys at the top level of the companion, where they pass "
            "verbatim into source_metadata"
        ),
        "missing": (
            "missing required field(s) {missing}: declare them in the object, "
            "e.g. {example}"
        ),
        "remedy": _REMEDY,
    },
    "import": {
        "where": (
            "Invalid measurement binding in the import configuration, channel "
            "{source}: "
        ),
        "absent": (
            "The import configuration binds no measurement to channel {source}: "
            "nothing to validate. Bind the channel to declare its asset identity "
            "(required fields: {required})."
        ),
        "not_object": "expected an object with the fields {fields}, got {kind}",
        "unknown": (
            "unknown field(s) {unknown}: a channel's measurement accepts only "
            "{fields}"
        ),
        "missing": (
            "missing required field(s) {missing}: declare measurement_point_id in "
            "the channel's binding, asset_id in the binding or in the file-level "
            "defaults, and acquired_at in the file-level defaults"
        ),
        "remedy": _IMPORT_REMEDY,
    },
}


def unit_family(unit: Optional[str]) -> Optional[str]:
    """Return the family ("acceleration" or "velocity") of a canonical unit.

    Args:
        unit: A canonical signal unit ("g", "m/s2", "mm/s", "m/s") or None.

    Returns:
        The family name, or None when the unit is None or not in any family
        (no guessing: an unknown spelling is not normalized here).
    """
    if unit is None:
        return None
    for family, units in UNIT_FAMILIES.items():
        if unit in units:
            return family
    return None


def validate_ledger_id(name: object, *, kind: str) -> str:
    """Return *name* unchanged if it satisfies the ledger id grammar.

    Built on :func:`path_safety.validate_name_component` (single safe path
    component, no rewriting) and stricter than it: the id must start with a
    letter or digit, must not end with a dot, must respect
    ``MAX_LEDGER_ID_CHARS`` and must not be a Win32 reserved device name,
    even with an extension, on every operating system. Letter case is
    preserved: ``P-101`` and ``p-101`` are different declarations (the
    ledger store refuses ids that differ only by case).

    Args:
        name: The declared id (any JSON value; only strings can pass).
        kind: Field name used in the message ("asset_id", ...).

    Returns:
        The same string.

    Raises:
        ValueError: Naming the offending value, the specific problem and
            the full grammar.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Invalid {kind} {name!r}: must be a string. A ledger id must "
            f"{LEDGER_ID_GRAMMAR}."
        )
    try:
        validate_name_component(name, kind=kind)
    except ValueError as exc:
        raise ValueError(f"{exc} A ledger id must {LEDGER_ID_GRAMMAR}.") from None

    problem: Optional[str] = None
    if len(name) > MAX_LEDGER_ID_CHARS:
        problem = (
            f"is {len(name)} characters long, over the {MAX_LEDGER_ID_CHARS}-"
            f"character cap"
        )
    elif not name[0].isalnum():
        problem = "must start with a letter or digit"
    elif name.endswith("."):
        problem = "ends with a dot"
    elif name.split(".", 1)[0].upper() in WIN32_RESERVED_NAMES:
        problem = "is a Windows reserved device name"
    if problem is not None:
        raise ValueError(
            f"Invalid {kind} {name!r}: {problem}. A ledger id must "
            f"{LEDGER_ID_GRAMMAR}."
        )
    return name


def normalize_direction(value: object) -> str:
    """Normalize a declared direction to the closed vocabulary.

    Only letter case and the form aliases ``h``/``v``/``a`` are normalized;
    ``x`` stays ``x`` and is never mapped onto ``horizontal``.

    Raises:
        ValueError: Listing the vocabulary when *value* is outside it.
    """
    if isinstance(value, str):
        folded = value.lower()
        folded = DIRECTION_ALIASES.get(folded, folded)
        if folded in VALID_DIRECTIONS:
            return folded
    raise ValueError(
        f"Invalid direction {value!r}: declare one of {list(VALID_DIRECTIONS)} "
        f"(aliases {DIRECTION_ALIASES} and any letter case are accepted; 'x' "
        f"and 'horizontal' are different declarations and are never mapped "
        f"onto each other)."
    )


def normalize_acquired_at(
    value: object, *, now: Optional[datetime] = None
) -> tuple[str, bool, bool]:
    """Parse a declared acquisition instant.

    Args:
        value: ISO 8601 string (anything ``datetime.fromisoformat`` accepts).
        now: Reference instant for the future check; defaults to the wall
            clock (UTC). Tests inject a fixed value.

    Returns:
        ``(acquired_at, timezone_declared, timestamp_suspect)``:
        the instant normalized to UTC in ISO 8601 form (so two exports of
        the same acquisition with different offsets order and collapse as
        one instant; the verbatim string stays in ``source_metadata``),
        whether an offset was declared (a naive timestamp is accepted,
        ordered as UTC and flagged), and whether the instant is implausible
        (in the 1970 epoch year, or more than one day after *now*).

    Raises:
        ValueError: If *value* is not a string or not ISO 8601.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid acquired_at {value!r}: {_ACQUIRED_AT_HELP}.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"Invalid acquired_at {value!r}: {_ACQUIRED_AT_HELP}."
        ) from None

    declared = parsed.tzinfo is not None and parsed.utcoffset() is not None
    if declared:
        utc = parsed.astimezone(timezone.utc)
    else:
        utc = parsed.replace(tzinfo=timezone.utc)

    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    suspect = utc < _EPOCH_YEAR_END or utc > reference + _FUTURE_TOLERANCE
    return utc.isoformat(), declared, suspect


def _is_control(char: str) -> bool:
    """True for control, format, surrogate, private-use, unassigned and
    line/paragraph separator characters (U+2028/U+2029 break line-based
    readers)."""
    category = unicodedata.category(char)
    return category.startswith("C") or category in ("Zl", "Zp")


def validate_free_text(field: str, value: object) -> str:
    """Validate one free-text field: non-empty string, bounded, one line.

    The single free-text rule of the contract, shared by the companion's
    ``operating_state`` / ``sensor_id`` / ``declared_by`` and by the ledger
    declarations (``declared_by``, ``note``, ``expected_sensor_id``): at most
    ``MAX_FREE_TEXT_CHARS`` characters, no control, format, separator or
    unassigned characters, never interpreted by the server.

    Args:
        field: Field name used in the message.
        value: The declared value (any JSON value; only strings can pass).

    Returns:
        The same string.

    Raises:
        ValueError: Naming the field, the limit or the offending characters.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Invalid {field} {value!r}: declare a non-empty string of at most "
            f"{MAX_FREE_TEXT_CHARS} characters, or omit the field."
        )
    if len(value) > MAX_FREE_TEXT_CHARS:
        raise ValueError(
            f"Invalid {field}: {len(value)} characters, over the "
            f"{MAX_FREE_TEXT_CHARS}-character limit."
        )
    control = [f"U+{ord(ch):04X}" for ch in value if _is_control(ch)]
    if control:
        raise ValueError(
            f"Invalid {field}: contains control character(s) {control[:3]}; "
            f"use plain text on one line."
        )
    return value


def _validate_number(field: str, value: object, *, positive: bool) -> float:
    """Validate a numeric field (bool and NaN/Inf are not numbers here)."""
    wanted = "positive number" if positive else "number"
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"Invalid {field} {value!r}: declare a {wanted} (e.g. "
            f"{'1480' if positive else '75'}), not a string or a flag; omit "
            f"the field when unknown."
        )
    if positive and value <= 0:
        raise ValueError(
            f"Invalid {field} {value!r}: declare a {wanted}, or omit the field "
            f"when unknown."
        )
    return float(value)


def _collect(problems: list[str], value: Any, normalize: Callable[[Any], Any]) -> Any:
    """Apply *normalize*, turning its ValueError into an accumulated problem."""
    try:
        return normalize(value)
    except ValueError as exc:
        problems.append(str(exc))
        return None


def _source(
    source_kind: str, companion_path: Union[str, Path, None], channel_index: object
) -> tuple[dict[str, str], str]:
    """The message bundle of *source_kind* and the source it names."""
    if source_kind == "companion":
        if channel_index is not None:
            raise ValueError(
                "channel_index names the channel of an import; a companion "
                "declaration takes none."
            )
        return _MESSAGES["companion"], Path(str(companion_path)).name
    if source_kind == "import":
        return _MESSAGES["import"], str(_validate_channel_index(channel_index))
    raise ValueError(
        f"Invalid source_kind {source_kind!r}: declare one of {list(SOURCE_KINDS)}."
    )


def validate_measurement_declaration(
    companion: dict[str, Any],
    companion_path: Union[str, Path, None],
    *,
    now: Optional[datetime] = None,
    source_kind: str = "companion",
    channel_index: Optional[int] = None,
) -> dict[str, Any]:
    """Validate the companion's ``"measurement"`` object and normalize it.

    Pure function (no file access). Every problem is accumulated into ONE
    ``ValueError`` that names the companion file, each offending field with
    its remedy, and the way out (fix the object, or remove it to load the
    file without asset identity), mirroring the raw-declaration refusal.

    Args:
        companion: The parsed companion dict (the whole file, not the
            object): the object is looked up under ``MEASUREMENT_KEY``.
            For an import, a dict holding one channel's merged declaration
            under the same key (see :func:`build_import_identity`).
        companion_path: Path or name of the companion file; only its name
            appears in messages. Unused for an import.
        now: Reference instant for the timestamp plausibility check (see
            :func:`normalize_acquired_at`).
        source_kind: ``"companion"`` (the default, with the wording files
            have always been refused with) or ``"import"``, whose messages
            name "import configuration, channel <channel_index>" and the
            configuration fix.
        channel_index: For an import, the physical column of the channel;
            None for a companion.

    Returns:
        A dict with exactly ``MEASUREMENT_DECLARATION_KEYS``, in that order:
        ids unchanged, ``acquired_at`` normalized to UTC, ``rpm``/``load``
        as floats, ``direction`` normalized, undeclared optionals ``None``.

    Raises:
        ValueError: If the object is missing, is not a JSON object, has
            unknown fields, lacks a required field, or any field fails its
            rule; or for an unknown *source_kind*, a *channel_index* given
            to a companion, or a bad one given to an import.
    """
    messages, source = _source(source_kind, companion_path, channel_index)
    return _validate_object(companion, messages, source, now=now, problems=[])


def _validate_object(
    container: Mapping[str, Any],
    messages: dict[str, str],
    source: str,
    *,
    now: Optional[datetime],
    problems: list[str],
    overflow_is_problem: bool = False,
) -> dict[str, Any]:
    """The rules of :func:`validate_measurement_declaration`, worded by
    *messages*; *problems* may already hold problems to report with them.

    An instant that cannot be moved to UTC (the first or last hours of the
    calendar) makes `datetime` raise `OverflowError`; the import route
    reports it as a problem, the companion route keeps its behaviour.
    """
    where = messages["where"].format(source=source)
    if MEASUREMENT_KEY not in container:
        raise ValueError(
            messages["absent"].format(
                source=source, required=list(REQUIRED_MEASUREMENT_FIELDS)
            )
        )
    declaration = container[MEASUREMENT_KEY]
    if not isinstance(declaration, dict):
        problem = messages["not_object"].format(
            fields=list(MEASUREMENT_FIELDS), kind=type(declaration).__name__
        )
        raise ValueError(
            where + "; ".join([*problems, problem]) + f". {messages['remedy']}"
        )

    unknown = sorted(str(key) for key in declaration if key not in MEASUREMENT_FIELDS)
    if unknown:
        problems.append(
            messages["unknown"].format(unknown=unknown, fields=list(MEASUREMENT_FIELDS))
        )
    missing = [f for f in REQUIRED_MEASUREMENT_FIELDS if declaration.get(f) is None]
    if missing:
        problems.append(
            messages["missing"].format(
                missing=", ".join(missing), example=_EXAMPLE_OBJECT
            )
        )

    identity: dict[str, Any] = {key: None for key in MEASUREMENT_DECLARATION_KEYS}
    identity["timezone_declared"] = False
    identity["timestamp_suspect"] = False

    for field in ("asset_id", "measurement_point_id"):
        if declaration.get(field) is not None:
            identity[field] = _collect(
                problems, declaration[field], partial(validate_ledger_id, kind=field)
            )
    if declaration.get("acquired_at") is not None:
        try:
            parsed = _collect(
                problems,
                declaration["acquired_at"],
                partial(normalize_acquired_at, now=now),
            )
        except OverflowError:
            if not overflow_is_problem:
                raise
            problems.append(
                f"Invalid acquired_at {declaration['acquired_at']!r}: the instant "
                f"lies beyond the instants UTC can hold (years 1 to 9999)"
            )
            parsed = None
        if parsed is not None:
            identity["acquired_at"] = parsed[0]
            identity["timezone_declared"] = parsed[1]
            identity["timestamp_suspect"] = parsed[2]
    if declaration.get("rpm") is not None:
        identity["rpm"] = _collect(
            problems,
            declaration["rpm"],
            partial(_validate_number, "rpm", positive=True),
        )
    if declaration.get("load") is not None:
        identity["load"] = _collect(
            problems,
            declaration["load"],
            partial(_validate_number, "load", positive=False),
        )
    for field in FREE_TEXT_FIELDS:
        if declaration.get(field) is not None:
            identity[field] = _collect(
                problems, declaration[field], partial(validate_free_text, field)
            )
    if declaration.get("direction") is not None:
        identity["direction"] = _collect(
            problems, declaration["direction"], normalize_direction
        )

    if problems:
        raise ValueError(where + "; ".join(problems) + f". {messages['remedy']}")
    return identity


def _validate_channel_index(channel_index: object) -> int:
    """Return *channel_index* if it is a non-negative int (bool excluded)."""
    if (
        isinstance(channel_index, bool)
        or not isinstance(channel_index, int)
        or channel_index < 0
    ):
        raise ValueError(
            f"Invalid channel_index {channel_index!r}: the measurement identity "
            f"needs the 0-based integer index of the decoded channel (0 for "
            f"single-channel files)."
        )
    return channel_index


def digest_file(path: Union[str, Path]) -> tuple[str, int]:
    """SHA-256 of a file's bytes and its size, read once.

    The single place the file is hashed: the identity hash, the ledger's
    ``file.content_sha256`` and the re-verification of a moved file all
    derive from this digest.

    Args:
        path: The file to digest.

    Returns:
        ``(sha256_hex, size_bytes)``: 64 lowercase hex characters and the
        number of bytes hashed.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        content_digest = hashlib.file_digest(fh, "sha256").hexdigest()
    return content_digest, size


def measurement_id_from_digest(content_sha256: str, channel_index: int) -> str:
    """The identity of a measurement from an already computed content digest.

    ``measurement_id`` is the first 16 hex digits of
    ``sha256("<sha256 of the file bytes, hex>:<channel_index>")``.

    Args:
        content_sha256: Hex digest of the file bytes (see :func:`digest_file`).
        channel_index: 0-based index of the decoded channel.

    Returns:
        16 lowercase hex characters.

    Raises:
        ValueError: If *channel_index* is not a non-negative int or the
            digest is not a hex string.
    """
    channel = _validate_channel_index(channel_index)
    if not isinstance(content_sha256, str) or not content_sha256:
        raise ValueError(
            f"Invalid content_sha256 {content_sha256!r}: pass the hex digest "
            f"returned by digest_file()."
        )
    combined = f"{content_sha256}:{channel}".encode("ascii")
    return hashlib.sha256(combined).hexdigest()[:MEASUREMENT_ID_HEX_CHARS]


def compute_measurement_id(path: Union[str, Path], channel_index: int) -> str:
    """Identity of a measurement: the file's bytes plus the decoded channel.

    ``measurement_id`` is the first 16 hex digits of
    ``sha256("<sha256 of the file bytes, hex>:<channel_index>")``. The same
    bytes copied or moved elsewhere keep their id; two channels of one
    multi-channel file get different ids; a re-export with different bytes
    is a different measurement.

    Args:
        path: The signal file.
        channel_index: 0-based index of the decoded channel (0 for
            single-channel and self-describing files).

    Returns:
        16 lowercase hex characters.

    Raises:
        ValueError: If *channel_index* is not a non-negative int.
        FileNotFoundError: If *path* does not exist.
    """
    channel = _validate_channel_index(channel_index)
    content_digest, _ = digest_file(path)
    return measurement_id_from_digest(content_digest, channel)


def build_measurement_identity(
    companion: dict[str, Any],
    companion_path: Union[str, Path, None],
    *,
    signal_path: Union[str, Path],
    channel_index: int,
    now: Optional[datetime] = None,
    source_kind: str = "companion",
) -> dict[str, Any]:
    """Validate the declaration, then bind it to the file: the full block.

    The single composition point the repository uses: the pure validation
    runs first (an invalid object is refused before the file is touched),
    then the file is digested ONCE: the identity hash is derived from the
    content digest and the EFFECTIVE channel index, and the full digest
    with the file size is kept in the block for the asset ledger.

    Args:
        companion: Parsed companion dict (see
            :func:`validate_measurement_declaration`).
        companion_path: Companion path or name, for messages.
        signal_path: The signal file whose bytes identify the measurement.
        channel_index: Effective decoded channel (0 unless a multi-channel
            raw file declared otherwise); for an import, the physical column.
        now: Reference instant for the timestamp plausibility check.
        source_kind: The message bundle (see
            :func:`validate_measurement_declaration`); an import that has
            already digested the file uses :func:`build_import_identity`.

    Returns:
        A dict with exactly ``MEASUREMENT_IDENTITY_KEYS``, in that order:
        the declaration, ``measurement_id``, ``channel_index``,
        ``content_sha256`` (64 hex) and ``size_bytes`` (file size).

    Raises:
        ValueError: From the validator, or for a bad *channel_index*.
        FileNotFoundError: If *signal_path* does not exist.
    """
    identity = validate_measurement_declaration(
        companion,
        companion_path,
        now=now,
        source_kind=source_kind,
        channel_index=channel_index if source_kind == "import" else None,
    )
    channel = _validate_channel_index(channel_index)
    content_digest, size_bytes = digest_file(signal_path)
    return _bound(identity, content_digest, size_bytes, channel)


def _bound(
    identity: dict[str, Any], content_sha256: str, size_bytes: int, channel: int
) -> dict[str, Any]:
    """The declaration plus the four file-derived keys."""
    identity["measurement_id"] = measurement_id_from_digest(content_sha256, channel)
    identity["channel_index"] = channel
    identity["content_sha256"] = content_sha256
    identity["size_bytes"] = size_bytes
    return identity


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

# A decimal number as the import configuration writes one (a canonical
# decimal string), bounded before any conversion.
_DECIMAL_TEXT = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?"
)
_MAX_DECIMAL_TEXT = 64


# The configuration's caps on a decimal string (``import_config``).
_MAX_DECIMAL_DIGITS = 20
_MAX_DECIMAL_ADJUSTED = 30


def _decimal_text(value: object) -> Optional[decimal.Decimal]:
    """*value* as an exact decimal when it is a decimal string within the
    configuration's caps (length, digits, magnitude); else None."""
    if (
        not isinstance(value, str)
        or len(value) > _MAX_DECIMAL_TEXT
        or _DECIMAL_TEXT.fullmatch(value) is None
    ):
        return None
    number = decimal.Decimal(value)  # exact; the grammar bounds the exponent
    # Significant digits as the configuration counts them: between the first
    # and the last non-zero digit ("1480000000000000000000000" has three).
    digits = "".join(map(str, number.as_tuple().digits)).strip("0")
    if len(digits) > _MAX_DECIMAL_DIGITS:
        return None
    if not number.is_zero() and abs(number.adjusted()) > _MAX_DECIMAL_ADJUSTED:
        return None
    return number


def _import_number(field: str, value: object, problems: list[str]) -> object:
    """rpm or load of an import: the configuration writes a decimal string,
    which becomes a float for the shared number rule; a string that is not
    one is refused here, in the import's words."""
    if not isinstance(value, str):
        return value
    number = _decimal_text(value)
    if number is None:
        problems.append(
            f"Invalid {field} {value!r}: the import configuration writes it as a "
            f"decimal string such as '1480', of at most {_MAX_DECIMAL_DIGITS} "
            f"significant digits and a magnitude within 1e-"
            f"{_MAX_DECIMAL_ADJUSTED} to 1e{_MAX_DECIMAL_ADJUSTED}"
        )
        return None
    return float(number)


def build_import_identity(
    defaults: Optional[Mapping[str, Any]],
    binding: Mapping[str, Any],
    *,
    channel_index: int,
    content_sha256: str,
    size_bytes: int,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """One import channel's identity block, from a file already digested.

    The channel's binding over the import's file-level defaults, validated
    by the companion's rules with the import bundle (refusals name "import
    configuration, channel <channel_index>"), then bound through
    :func:`measurement_id_from_digest`, so the file is never hashed twice.
    ``rpm`` and ``load`` may be the configuration's decimal strings.

    Args:
        defaults: The file-level defaults (fields
            :data:`IMPORT_DEFAULT_FIELDS`, any of them null), or None.
        binding: The channel's binding (fields
            :data:`IMPORT_BINDING_FIELDS`); its ``asset_id`` overrides the
            default.
        channel_index: Physical 0-based column of the channel.
        content_sha256: SHA-256 of every byte of the file, 64 hex digits.
        size_bytes: Size of the file.
        now: Reference instant for the timestamp plausibility check.

    Returns:
        A dict with exactly ``MEASUREMENT_IDENTITY_KEYS``, in that order.

    Raises:
        ValueError: One message with every problem: a field in the wrong
            block, a field that fails its rule, a missing required field, or
            a bad *channel_index*, digest or size.
    """
    channel = _validate_channel_index(channel_index)
    if not isinstance(content_sha256, str) or not _SHA256_HEX.fullmatch(content_sha256):
        raise ValueError(
            f"Invalid content_sha256 {content_sha256!r}: pass the SHA-256 of the "
            f"file bytes as 64 lowercase hex digits."
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise ValueError(
            f"Invalid size_bytes {size_bytes!r}: pass the file size as a "
            f"non-negative integer."
        )
    messages = _MESSAGES["import"]
    problems: list[str] = []
    merged: dict[str, Any] = {}
    for block, fields, value in (
        ("the file-level defaults", IMPORT_DEFAULT_FIELDS, defaults or {}),
        ("the channel's binding", IMPORT_BINDING_FIELDS, binding),
    ):
        if not isinstance(value, Mapping):
            problems.append(
                f"{block} must be an object with the fields {list(fields)}, got "
                f"{type(value).__name__}"
            )
            continue
        misplaced = sorted(str(key) for key in value if key not in fields)
        if misplaced:
            problems.append(
                f"field(s) {misplaced} do not belong in {block}, which accepts "
                f"only {list(fields)}"
            )
        merged.update(
            {
                key: item
                for key, item in value.items()
                if key in fields and item is not None
            }
        )
    for field in ("rpm", "load"):
        if field in merged:
            merged[field] = _import_number(field, merged[field], problems)
            if merged[field] is None:
                del merged[field]
    identity = _validate_object(
        {MEASUREMENT_KEY: merged},
        messages,
        str(channel),
        now=now,
        problems=problems,
        overflow_is_problem=True,
    )
    return _bound(identity, content_sha256, size_bytes, channel)


def binding_collisions(
    identities: Iterable[Mapping[str, Any]],
) -> tuple[tuple[int, ...], ...]:
    """Channels of one import bound to the same measurement.

    Two channels with the same asset, measurement point, direction and
    sensor would record one measurement twice (KTD15 refuses them; the
    caller reports ``binding_collision``). The asset is part of the key
    because a measurement point belongs to an asset.

    Args:
        identities: Identity blocks of :func:`build_import_identity`.

    Returns:
        Each group of colliding channel indices, sorted, groups in order of
        their first channel; empty when every binding is distinct.
    """
    groups: dict[tuple[Any, ...], list[int]] = {}
    for identity in identities:
        key = tuple(
            identity[field]
            for field in ("asset_id", "measurement_point_id", "direction", "sensor_id")
        )
        groups.setdefault(key, []).append(identity["channel_index"])
    collisions = [tuple(sorted(group)) for group in groups.values() if len(group) > 1]
    return tuple(sorted(collisions))


def _exact_seconds(value: Fraction) -> str:
    """An exact number of seconds: a decimal when it terminates, else p/q."""
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    with decimal.localcontext(decimal.Context(prec=10_000)):
        exact = decimal.Decimal(value.numerator) / decimal.Decimal(value.denominator)
    exact = exact.normalize()
    return "0" if exact.is_zero() else format(exact, "f")


# Fractional seconds beyond the microsecond, which ``fromisoformat`` drops.
_SUBMICROSECOND = re.compile(r"[.,][0-9]{7,}")


def shift_acquired_at(
    acquired_at: str, *, offset_samples: int, sampling_rate_hz: str
) -> dict[str, Any]:
    """The acquisition instant of a trimmed range's first sample.

    The shift is ``offset_samples / sampling_rate_hz`` seconds, computed
    exactly and stated exactly; the instant keeps the microsecond resolution
    of ISO 8601 as Python writes it, so a shift that is not a whole number
    of microseconds is applied rounded half to even, and reported as not
    exact.

    Args:
        acquired_at: The instant of the data region's first sample, ISO
            8601 (as :func:`normalize_acquired_at` returns it, or naive).
        offset_samples: Samples between the region's start and the first
            selected sample.
        sampling_rate_hz: The sampling rate, a positive decimal string.

    Returns:
        A dict with exactly ``TRIM_SHIFT_KEYS``: the shifted instant (UTC
        when it declares an offset, naive when it does not),
        *offset_samples*, the applied shift in seconds as a decimal string,
        the exact shift (a decimal when it terminates, else ``p/q``), and
        whether the shifted instant is exactly the stated instant plus the
        exact shift.

    Raises:
        ValueError: For an instant that is not ISO 8601, a negative or
            non-integer offset, a rate that is not a positive decimal
            string within the configuration's caps, or a shifted instant
            outside the years 1 to 9999.
    """
    if not isinstance(acquired_at, str):
        raise ValueError(f"Invalid acquired_at {acquired_at!r}: {_ACQUIRED_AT_HELP}.")
    try:
        start = datetime.fromisoformat(acquired_at)
        if start.tzinfo is not None and start.utcoffset() is not None:
            start = start.astimezone(timezone.utc)
    except ValueError:
        raise ValueError(
            f"Invalid acquired_at {acquired_at!r}: {_ACQUIRED_AT_HELP}."
        ) from None
    except OverflowError:
        raise ValueError(
            f"Invalid acquired_at {acquired_at!r}: the instant lies beyond the "
            f"instants UTC can hold (years 1 to 9999)."
        ) from None
    if (
        isinstance(offset_samples, bool)
        or not isinstance(offset_samples, int)
        or offset_samples < 0
    ):
        raise ValueError(
            f"Invalid offset_samples {offset_samples!r}: pass the number of samples "
            f"between the data region's start and the first selected sample."
        )
    number = _decimal_text(sampling_rate_hz)
    if number is None or number <= 0:
        raise ValueError(
            f"Invalid sampling_rate_hz {sampling_rate_hz!r}: pass the sampling rate "
            f"as a positive decimal string such as '25600'."
        )
    exact = Fraction(offset_samples) / Fraction(number)
    microseconds = round(exact * 1_000_000)
    try:
        shifted = start + timedelta(microseconds=microseconds)
    except OverflowError:
        raise ValueError(
            f"Invalid trim: {offset_samples} samples at {sampling_rate_hz} Hz move "
            f"the instant {acquired_at!r} beyond the year 9999."
        ) from None
    shift = decimal.Decimal(microseconds).scaleb(-6).normalize()
    return {
        "acquired_at": shifted.isoformat(),
        "offset_samples": offset_samples,
        "shift_seconds": "0" if shift.is_zero() else format(shift, "f"),
        "exact_shift_seconds": _exact_seconds(exact),
        "shift_exact": exact * 1_000_000 == microseconds
        and _SUBMICROSECOND.search(acquired_at) is None,
    }
