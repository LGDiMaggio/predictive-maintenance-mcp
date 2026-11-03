"""
Canonical JSON and its content hash: one form shared by the asset ledger
and the signal import.

Leaf module (standard library only). The asset ledger store re-exports both
functions under their historical names, so every existing import keeps
working and the bytes it writes are unchanged. The signal import computes
its configuration digest and ``import_id`` with the same function, so the two
never drift apart; it lives here rather than in the store because the
import's interpretation contract (``signal_acquisition.import_config``) is
imported while ``signal_acquisition`` is still initializing, before the
ledger store could be.

The form is compact, key-sorted JSON with ``ensure_ascii=True`` and no NaN
or Infinity. For an ASCII-only value made of strings without control
characters, integers below 2**53, booleans, null, lists and objects, it is
byte-identical to RFC 8785 (JSON Canonicalization Scheme). Non-ASCII text is
escaped as ``\\uXXXX``, a declared deviation from RFC 8785 that keeps
U+2028, U+2029 and every other line separator out of the ledger's records.
(``asset_ledger.snapshot`` keeps its own private copy for the snapshot
digests, so those can never move with this module.)
"""

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "content_hash"]


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to compact, key-sorted, ASCII-only JSON.

    ``allow_nan=False`` refuses NaN/Infinity (a ValueError, before any byte
    touches the disk); ``ensure_ascii=True`` keeps line separators and every
    non-ASCII character escaped, so a record can never contain a literal
    0x0A, 0x0D, U+2028 or U+2029.

    Raises:
        ValueError: For non-finite floats or non-serializable values (``json``
            reports foreign types such as numpy scalars, sets and datetimes
            as ``TypeError``; a refused input is a ``ValueError`` everywhere
            in this package, and the message names the offending type).
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        )
    except TypeError as exc:
        raise ValueError(
            f"Cannot serialize to canonical JSON: {exc}. Convert numpy scalars "
            f"with .item(), sets to lists and datetimes to ISO 8601 strings "
            f"before building the event."
        ) from exc


def content_hash(obj: Any) -> str:
    """SHA-256 (64 hex chars) of the canonical JSON of *obj*."""
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()
