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
"""

import hashlib
import math
import os
import unicodedata
from datetime import datetime, timedelta, timezone
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
    "WIN32_RESERVED_NAMES",
    "LEDGER_ID_GRAMMAR",
    "MEASUREMENT_ID_HEX_CHARS",
    "MEASUREMENT_DECLARATION_KEYS",
    "MEASUREMENT_IDENTITY_KEYS",
    "unit_family",
    "validate_ledger_id",
    "normalize_direction",
    "normalize_acquired_at",
    "validate_measurement_declaration",
    "digest_file",
    "measurement_id_from_digest",
    "compute_measurement_id",
    "build_measurement_identity",
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


def _validate_free_text(field: str, value: object) -> str:
    """Validate one free-text field: non-empty string, bounded, one line."""
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


def validate_measurement_declaration(
    companion: dict[str, Any],
    companion_path: Union[str, Path],
    *,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Validate the companion's ``"measurement"`` object and normalize it.

    Pure function (no file access). Every problem is accumulated into ONE
    ``ValueError`` that names the companion file, each offending field with
    its remedy, and the way out (fix the object, or remove it to load the
    file without asset identity), mirroring the raw-declaration refusal.

    Args:
        companion: The parsed companion dict (the whole file, not the
            object): the object is looked up under ``MEASUREMENT_KEY``.
        companion_path: Path or name of the companion file; only its name
            appears in messages.
        now: Reference instant for the timestamp plausibility check (see
            :func:`normalize_acquired_at`).

    Returns:
        A dict with exactly ``MEASUREMENT_DECLARATION_KEYS``, in that order:
        ids unchanged, ``acquired_at`` normalized to UTC, ``rpm``/``load``
        as floats, ``direction`` normalized, undeclared optionals ``None``.

    Raises:
        ValueError: If the object is missing, is not a JSON object, has
            unknown fields, lacks a required field, or any field fails its
            rule.
    """
    name = Path(str(companion_path)).name
    if MEASUREMENT_KEY not in companion:
        raise ValueError(
            f'Companion {name} declares no "measurement" object: nothing to '
            f"validate. Add the object to declare the asset identity of the "
            f"file (required fields: {list(REQUIRED_MEASUREMENT_FIELDS)})."
        )
    declaration = companion[MEASUREMENT_KEY]
    if not isinstance(declaration, dict):
        raise ValueError(
            f'Invalid "measurement" object in companion {name}: expected a '
            f"JSON object with the fields {list(MEASUREMENT_FIELDS)}, got "
            f"{type(declaration).__name__}. {_REMEDY}"
        )

    problems: list[str] = []
    unknown = sorted(str(key) for key in declaration if key not in MEASUREMENT_FIELDS)
    if unknown:
        problems.append(
            f"unknown field(s) {unknown}: the object accepts only "
            f"{list(MEASUREMENT_FIELDS)}; keep free-form keys at the top level "
            f"of the companion, where they pass verbatim into source_metadata"
        )
    missing = [f for f in REQUIRED_MEASUREMENT_FIELDS if declaration.get(f) is None]
    if missing:
        problems.append(
            f"missing required field(s) {', '.join(missing)}: declare them in "
            f"the object, e.g. {_EXAMPLE_OBJECT}"
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
        parsed = _collect(
            problems,
            declaration["acquired_at"],
            partial(normalize_acquired_at, now=now),
        )
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
                problems, declaration[field], partial(_validate_free_text, field)
            )
    if declaration.get("direction") is not None:
        identity["direction"] = _collect(
            problems, declaration["direction"], normalize_direction
        )

    if problems:
        raise ValueError(
            f'Invalid "measurement" object in companion {name}: '
            + "; ".join(problems)
            + f". {_REMEDY}"
        )
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
    companion_path: Union[str, Path],
    *,
    signal_path: Union[str, Path],
    channel_index: int,
    now: Optional[datetime] = None,
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
            raw file declared otherwise).
        now: Reference instant for the timestamp plausibility check.

    Returns:
        A dict with exactly ``MEASUREMENT_IDENTITY_KEYS``, in that order:
        the declaration, ``measurement_id``, ``channel_index``,
        ``content_sha256`` (64 hex) and ``size_bytes`` (file size).

    Raises:
        ValueError: From the validator, or for a bad *channel_index*.
        FileNotFoundError: If *signal_path* does not exist.
    """
    identity = validate_measurement_declaration(companion, companion_path, now=now)
    channel = _validate_channel_index(channel_index)
    content_digest, size_bytes = digest_file(signal_path)
    identity["measurement_id"] = measurement_id_from_digest(content_digest, channel)
    identity["channel_index"] = channel
    identity["content_sha256"] = content_digest
    identity["size_bytes"] = size_bytes
    return identity
