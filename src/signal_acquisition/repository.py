"""
In-memory signal repository with LRU eviction.

Provides a singleton cache for loaded vibration signals, enabling the
signal_id reference pattern: load once, reference by ID across multiple
MCP tool calls without re-reading from disk.

Thread-safe via threading.Lock.
"""

import copy
import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import DATA_DIR
from ..path_safety import safe_resolve
from .loaders import (
    RAW_EXTENSIONS,
    VALID_BYTE_ORDERS,
    VALID_SAMPLE_FORMATS,
    load_raw_binary,
    load_signal_data,
)

logger = logging.getLogger(__name__)

#: Canonical signal-unit vocabulary. ISO severity verdicts require one of
#: these to be DECLARED (never guessed from signal amplitude).
VALID_SIGNAL_UNITS: tuple[str, ...] = ("g", "m/s2", "mm/s", "m/s")

#: Accepted spellings mapped to the canonical vocabulary.
_UNIT_ALIASES: dict[str, str] = {
    "m/s²": "m/s2",
    "m/s^2": "m/s2",
    "mm/sec": "mm/s",
}


def normalize_signal_unit(unit: Optional[str]) -> Optional[str]:
    """Normalize a signal-unit string to the canonical vocabulary.

    Args:
        unit: Raw unit string (e.g. "G", "m/s²", "mm/s") or None.

    Returns:
        The canonical unit ("g", "m/s2", "mm/s", "m/s"), or None when the
        input is None or not a recognized unit. No guessing is performed —
        unrecognized strings map to None, never to a default unit.
    """
    if unit is None:
        return None
    u = str(unit).strip().lower()
    u = _UNIT_ALIASES.get(u, u)
    return u if u in VALID_SIGNAL_UNITS else None


#: The six raw-binary decode parameters threaded from load_signal to the
#: decoder, in canonical order (also the ``raw_format`` provenance key order).
_RAW_PARAM_NAMES: tuple[str, ...] = (
    "sample_format",
    "byte_order",
    "n_channels",
    "channel_index",
    "header_offset",
    "scale_factor",
)

#: Effective defaults for the OPTIONAL raw decode parameters, applied only
#: AFTER the explicit > companion merge. ``sample_format`` (like
#: ``sampling_rate``) has NO default — for a raw file it is a REQUIRED
#: declaration, refused when unavailable after the merge.
_RAW_PARAM_DEFAULTS: dict[str, Any] = {
    "byte_order": "little",
    "n_channels": 1,
    "channel_index": 0,
    "header_offset": 0,
    "scale_factor": None,
}


def _explicit_raw_params(
    sample_format: Optional[str],
    byte_order: Optional[str],
    n_channels: Optional[int],
    channel_index: Optional[int],
    header_offset: Optional[int],
    scale_factor: Optional[float],
) -> dict[str, Any]:
    """Bundle the explicitly declared raw decode parameters.

    ``None`` means "not explicitly declared", so the explicit > companion >
    default/refusal merge in :meth:`SignalRepository._merge_raw_params` can
    tell an omitted parameter from a declared one.
    """
    return {
        "sample_format": sample_format,
        "byte_order": byte_order,
        "n_channels": n_channels,
        "channel_index": channel_index,
        "header_offset": header_offset,
        "scale_factor": scale_factor,
    }


class SignalRepository:
    """In-memory signal cache with LRU eviction.

    Signals are stored as numpy arrays keyed by a user-chosen or auto-generated
    signal_id. Least-recently-used eviction keeps total memory under a
    configurable cap.
    """

    def __init__(self, max_memory_bytes: int = 10 * 1024**3):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max_memory = max_memory_bytes
        self._current_memory = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_signal(
        self,
        filepath: str,
        signal_id: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        signal_unit: Optional[str] = None,
        overwrite: bool = False,
        *,
        sample_format: Optional[str] = None,
        byte_order: Optional[str] = None,
        n_channels: Optional[int] = None,
        channel_index: Optional[int] = None,
        header_offset: Optional[int] = None,
        scale_factor: Optional[float] = None,
    ) -> dict:
        """Load a signal file into the repository.

        For headerless raw binary files (extensions in ``RAW_EXTENSIONS``:
        .bin/.raw/.dat) the decode parameters below apply, with precedence
        explicit parameter > companion ``<stem>_metadata.json`` field >
        documented default. ``sample_format`` AND ``sampling_rate`` are
        REQUIRED for raw files (a headerless file carries no
        self-description and parameters are never guessed); a load missing
        either is refused with one message naming everything missing.
        Declaring raw parameters on a self-describing format is refused as
        a contradiction. ``None`` on any raw parameter means "not
        explicitly declared".

        Args:
            filepath: Filename relative to DATA_DIR, or absolute path.
            signal_id: Custom ID. Defaults to the file's path relative to
                DATA_DIR with separators replaced by underscores (e.g.
                'real_train/baseline_1.csv' -> 'real_train_baseline_1'),
                or the filename stem for files outside DATA_DIR. When the
                effective n_channels > 1, the derived id gains a
                ``_ch<channel_index>`` suffix so channels never collide.
            sampling_rate: Override sampling rate (Hz). Takes precedence
                over the companion metadata file. Required (here or in the
                companion) for raw binary files.
            signal_unit: Declared signal unit — 'g', 'm/s2', 'mm/s', or
                'm/s'. Takes precedence over the companion metadata file.
                Required (here or in metadata) for ISO severity verdicts;
                units are never guessed from amplitude.
            overwrite: Replace an existing entry with the same signal_id.
                Without it a collision is an explicit error, so two files
                that derive the same id can never silently shadow each
                other.
            sample_format: Declared raw sample dtype — one of
                ``VALID_SAMPLE_FORMATS`` ('float32', 'float64', 'int16',
                'int32'). Raw files only; REQUIRED for them.
            byte_order: Declared raw endianness — 'little' (effective
                default) or 'big'.
            n_channels: Interleaved channels in the raw file (effective
                default 1).
            channel_index: 0-based channel to extract (effective default
                0).
            header_offset: Bytes to skip before the first raw sample
                (effective default 0).
            scale_factor: Optional multiplier applied after decoding (e.g.
                ADC counts -> physical unit). Default: no scaling.

        Returns:
            Metadata dict compatible with StoredSignalInfo (raw loads
            record the effective decode parameters under ``raw_format``).

        Raises:
            FileNotFoundError: If signal file does not exist.
            ValueError: If signal data cannot be loaded, if signal_unit is
                not one of the valid units, if signal_id collides with
                an existing entry and overwrite is False, if a required
                raw declaration is missing, or if raw parameters are
                declared for a self-describing format.
        """
        declared_unit = self._validate_unit(signal_unit)
        explicit_raw = _explicit_raw_params(
            sample_format,
            byte_order,
            n_channels,
            channel_index,
            header_offset,
            scale_factor,
        )
        entry = self._prepare_entry(
            filepath, signal_id, sampling_rate, declared_unit, explicit_raw
        )
        return self._insert_entry(entry, overwrite=overwrite)

    def load_signals(
        self,
        filepaths: list[str],
        sampling_rate: Optional[float] = None,
        signal_unit: Optional[str] = None,
        overwrite: bool = False,
        *,
        sample_format: Optional[str] = None,
        byte_order: Optional[str] = None,
        n_channels: Optional[int] = None,
        channel_index: Optional[int] = None,
        header_offset: Optional[int] = None,
        scale_factor: Optional[float] = None,
    ) -> list[dict]:
        """Load a batch of signal files atomically (fail-fast, all-or-nothing).

        All paths, derived signal_ids, and collisions are validated up
        front; the arrays are then all read from disk BEFORE anything is
        stored. On the first problem a single actionable ValueError names
        the offending entries and nothing is loaded — the repository is
        never left half-populated.

        The raw decode parameters (sample_format, byte_order, n_channels,
        channel_index, header_offset, scale_factor) broadcast to EVERY file
        in the batch, exactly like sampling_rate/signal_unit; per-file
        values come from each file's companion ``<stem>_metadata.json``.
        See :meth:`load_signal` for the per-file merge and refusal rules.

        Args:
            filepaths: Signal file paths (relative to DATA_DIR or absolute).
                signal_ids are derived from each file's relative path.
            sampling_rate: One sampling rate applied to every file
                (overrides companion metadata). None = per-file metadata.
            signal_unit: One declared unit applied to every file. None =
                per-file metadata.
            overwrite: Allow replacing existing entries with the same ids.
            sample_format: One declared raw sample dtype applied to every
                file (see :meth:`load_signal`). None = per-file metadata.
            byte_order: One declared raw endianness applied to every file.
            n_channels: Interleaved channel count applied to every file.
            channel_index: 0-based channel extracted from every file.
            header_offset: Header bytes skipped in every file.
            scale_factor: One decode multiplier applied to every file.

        Returns:
            One StoredSignalInfo-compatible dict per file, in input order.

        Raises:
            ValueError: Empty list, invalid unit, missing files, duplicate
                or colliding signal_ids, or an invalid/missing raw
                declaration — raised BEFORE any signal is stored.
        """
        if not filepaths:
            raise ValueError(
                "load_signal received an empty list — pass at least one "
                "signal file path (see list_signals(scope='disk') for the "
                "files on disk)."
            )
        declared_unit = self._validate_unit(signal_unit)
        explicit_raw = _explicit_raw_params(
            sample_format,
            byte_order,
            n_channels,
            channel_index,
            header_offset,
            scale_factor,
        )

        # Phase 1: validate every path and every derived id up front.
        problems: list[str] = []
        planned: list[tuple[str, str]] = []  # (filepath, signal_id)
        batch_ids: dict[str, str] = {}
        for raw in filepaths:
            fp = Path(raw)
            if not fp.is_absolute():
                fp = DATA_DIR / raw
            # Channel-aware id derivation MUST agree with _prepare_entry's
            # fallback: both run the same explicit > companion merge, so a
            # multi-channel raw load derives the same _ch<k>-suffixed id on
            # the batch route and the single route.
            channel: Optional[int] = None
            if fp.suffix.lower() in RAW_EXTENSIONS:
                source = self._read_metadata(fp).get("source_metadata", {})
                merged = self._merge_raw_params(fp, explicit_raw, source)
                channel = self._channel_for_id(merged)
            sid = self._default_signal_id(fp, channel)
            if not fp.exists():
                problems.append(f"'{raw}': file not found")
                continue
            if sid in batch_ids:
                problems.append(
                    f"'{raw}': derives signal_id '{sid}' already taken by "
                    f"'{batch_ids[sid]}' in this batch"
                )
                continue
            if not overwrite:
                with self._lock:
                    already = sid in self._store
                if already:
                    problems.append(
                        f"'{raw}': signal_id '{sid}' is already loaded "
                        f"(pass overwrite=True to replace it)"
                    )
                    continue
            batch_ids[sid] = raw
            planned.append((raw, sid))
        if problems:
            raise ValueError(
                "Batch load aborted, nothing was loaded — "
                + "; ".join(problems)
                + ". Fix these entries and retry (the repository is unchanged)."
            )

        # Phase 2: read every array from disk before storing anything. Any
        # read/validation failure here raises BEFORE the insert phase runs,
        # so a partially-prepared batch never reaches the store.
        entries = [
            self._prepare_entry(raw, sid, sampling_rate, declared_unit, explicit_raw)
            for raw, sid in planned
        ]

        # Phase 3: insert every entry in a SINGLE continuous locked section
        # (re-checks all ids for collisions, then inserts all-or-nothing).
        return self._insert_batch(entries, overwrite=overwrite)

    def get_signal(self, signal_id: str) -> np.ndarray:
        """Get signal array by ID. Updates LRU order.

        The returned array is a READ-ONLY view of the cached data: normal
        in-place writes raise ``ValueError: assignment destination is
        read-only``, so tools cannot ACCIDENTALLY corrupt the cache while
        analyzing it. This guards against mistakes, not deliberate tampering
        — a caller that knowingly re-enables writes on the underlying buffer
        (``view.base.setflags(write=True)``) can still reach the cached data.
        If you need to mutate, copy first (``np.array(view)``).

        Raises:
            KeyError: If signal_id not found (message names the remedy).
        """
        with self._lock:
            if signal_id not in self._store:
                raise KeyError(self._not_found_message(signal_id))
            self._store.move_to_end(signal_id)
            view = self._store[signal_id]["array"].view()
            view.flags.writeable = False
            return view

    def get_signal_info(self, signal_id: str) -> dict:
        """Get metadata for a stored signal without touching LRU order.

        Returns a DEEP copy: the stored ``info`` holds mutable nested state
        (``shape`` list, ``source_metadata`` dict). A shallow ``.copy()``
        would share those references, so a caller mutating
        ``info['source_metadata']['rpm']`` would corrupt the cache.

        Raises:
            KeyError: If signal_id not found (message names the remedy).
        """
        with self._lock:
            if signal_id not in self._store:
                raise KeyError(self._not_found_message(signal_id))
            return copy.deepcopy(self._store[signal_id]["info"])

    def list_signals(self) -> list[dict]:
        """List all cached signals with metadata.

        Each entry is a DEEP copy (see :meth:`get_signal_info`) so mutating a
        returned dict's nested ``shape``/``source_metadata`` cannot leak into
        the cache.
        """
        with self._lock:
            return [copy.deepcopy(entry["info"]) for entry in self._store.values()]

    def clear_signal(self, signal_id: str) -> bool:
        """Remove one signal. Returns True if found and removed."""
        with self._lock:
            if signal_id not in self._store:
                return False
            self._remove_entry(signal_id)
            return True

    def clear_all(self) -> int:
        """Clear entire cache. Returns count of removed signals."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._current_memory = 0
            return count

    @property
    def current_memory_bytes(self) -> int:
        """Current total memory usage in bytes."""
        return self._current_memory

    @property
    def signal_count(self) -> int:
        """Number of signals currently stored."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_unit(self, signal_unit: Optional[str]) -> Optional[str]:
        """Normalize an explicitly declared unit or raise (fail fast)."""
        if signal_unit is None:
            return None
        declared = normalize_signal_unit(signal_unit)
        if declared is None:
            raise ValueError(
                f"Invalid signal_unit '{signal_unit}' — declare one of "
                f"{list(VALID_SIGNAL_UNITS)} ('g'/'m/s2' for acceleration, "
                f"'mm/s'/'m/s' for velocity)."
            )
        return declared

    def _default_signal_id(self, fp: Path, channel: Optional[int] = None) -> str:
        """Derive the default signal_id from the path relative to DATA_DIR.

        'real_train/baseline_1.csv' -> 'real_train_baseline_1', so files
        with the same name in different directories get DISTINCT ids
        (stem-only ids let real_train/baseline_1 and real_test/baseline_1
        silently shadow each other). Files outside DATA_DIR fall back to
        the filename stem.

        Args:
            fp: Resolved file path the id derives from.
            channel: Channel index to embed as a ``_ch<k>`` suffix — pass
                it only when the EFFECTIVE n_channels > 1 (see
                :meth:`_channel_for_id`), so single-channel ids stay
                byte-identical to the pre-raw behavior while channels of
                one multi-channel file can never collide.
        """
        try:
            rel = fp.relative_to(DATA_DIR)
        except ValueError:
            base = fp.stem
        else:
            base = "_".join(rel.with_suffix("").parts)
        if channel is not None:
            return f"{base}_ch{channel}"
        return base

    def _not_found_message(self, signal_id: str) -> str:
        """Standard actionable not-found message. Caller must hold lock."""
        available = list(self._store.keys())
        return (
            f"Signal '{signal_id}' is not in the in-memory repository — it "
            f"was never loaded, or it was evicted (least-recently-used "
            f"signals are dropped when the cache exceeds its "
            f"PMM_SIGNAL_CACHE_GB cap). Currently loaded: "
            f"{available if available else 'none'}. Load it with "
            f"load_signal(filepath=...); use list_signals(scope='disk') to "
            f"see the files available on disk."
        )

    @staticmethod
    def _channel_for_id(raw_params: dict[str, Any]) -> Optional[int]:
        """Channel index to embed in a derived signal_id, or None.

        Returns the effective channel_index only when the EFFECTIVE
        n_channels > 1; a single-channel load returns None so the derived
        id stays identical to the pre-raw behavior (backward compatible).
        """
        n = raw_params.get("n_channels")
        idx = raw_params.get("channel_index")
        if isinstance(n, int) and n > 1 and isinstance(idx, int):
            return idx
        return None

    def _validate_companion_raw_value(
        self, name: str, value: Any, companion: str
    ) -> Any:
        """Validate one companion-sourced raw decode value.

        ``json.load`` applies no ``Literal``, so companion values are
        re-validated against the SAME closed vocabularies (and types) as
        the explicit parameters — mirroring the
        normalize_signal_unit/VALID_SIGNAL_UNITS approach. An invalid value
        raises a ValueError naming the offending value, its companion-file
        source, and the valid vocabulary — never a raw KeyError.
        """
        if name == "sample_format":
            if value not in VALID_SAMPLE_FORMATS:
                raise ValueError(
                    f"Invalid sample_format '{value}' in companion "
                    f"{companion} — declare one of "
                    f"{list(VALID_SAMPLE_FORMATS)}."
                )
        elif name == "byte_order":
            if value not in VALID_BYTE_ORDERS:
                raise ValueError(
                    f"Invalid byte_order '{value}' in companion "
                    f"{companion} — declare one of {list(VALID_BYTE_ORDERS)} "
                    f"('little' is the default)."
                )
        elif name in ("n_channels", "channel_index", "header_offset"):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Invalid {name} {value!r} in companion {companion} — "
                    f"declare an integer."
                )
        elif name == "scale_factor":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Invalid scale_factor {value!r} in companion "
                    f"{companion} — declare a number."
                )
        return value

    def _merge_raw_params(
        self,
        fp: Path,
        explicit: dict[str, Any],
        companion_source: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge the raw decode declaration for one file.

        Precedence: explicit parameter > companion ``<stem>_metadata.json``
        field > documented default. Effective defaults (byte_order 'little',
        n_channels 1, channel_index 0, header_offset 0, no scale_factor)
        are applied only AFTER the merge; ``sample_format`` has no default
        and stays None when neither route declared it, so the caller's
        required-declaration check can refuse.

        Args:
            fp: Resolved file path (names the companion file in errors).
            explicit: Explicitly declared parameters (None = undeclared).
            companion_source: Complete companion metadata dict (the
                ``source_metadata`` of :meth:`_read_metadata`).

        Returns:
            Dict with the six decode parameters in canonical order.

        Raises:
            ValueError: If a companion-sourced value is outside its closed
                vocabulary or of the wrong type.
        """
        companion = f"{fp.stem}_metadata.json"
        merged: dict[str, Any] = {}
        for name in _RAW_PARAM_NAMES:
            value = explicit.get(name)
            if value is None:
                companion_value = companion_source.get(name)
                if companion_value is not None:
                    value = self._validate_companion_raw_value(
                        name, companion_value, companion
                    )
            if value is None:
                value = _RAW_PARAM_DEFAULTS.get(name)
            merged[name] = value
        return merged

    def _require_raw_declaration(
        self,
        filepath: str,
        fp: Path,
        raw_params: dict[str, Any],
        effective_sampling_rate: Optional[float],
    ) -> None:
        """Refuse a raw load whose required declaration is incomplete (R2).

        A headerless raw file has zero self-description, so
        ``sample_format`` AND ``sampling_rate`` must be available after the
        explicit > companion merge. ALL missing parameters are accumulated
        into ONE ValueError (the batch-validator pattern) naming the exact
        re-call and the companion-file alternative — never one refusal per
        parameter.
        """
        missing = []
        if raw_params.get("sample_format") is None:
            missing.append("sample_format")
        if effective_sampling_rate is None:
            missing.append("sampling_rate")
        if not missing:
            return
        call_examples = {
            "sample_format": "sample_format='float32'",
            "sampling_rate": "sampling_rate=25600.0",
        }
        json_examples = {
            "sample_format": '"sample_format": "float32"',
            "sampling_rate": '"sampling_rate": 25600.0',
        }
        recall = ", ".join(call_examples[m] for m in missing)
        fields = ", ".join(json_examples[m] for m in missing)
        message = (
            f"Raw binary file '{filepath}' cannot be decoded — missing "
            f"required declaration(s): {', '.join(missing)}. A headerless "
            f"raw file carries no self-description and parameters are never "
            f"guessed. Declare them explicitly, e.g. "
            f"load_signal(filepath='{filepath}', {recall}), or create a "
            f"companion {fp.stem}_metadata.json next to the file with "
            f"{{{fields}}}."
        )
        if "sample_format" in missing:
            message += f" Valid sample_format values: {list(VALID_SAMPLE_FORMATS)}."
        raise ValueError(message)

    def _reject_raw_params_on_self_describing(
        self, filepath: str, fp: Path, explicit: dict[str, Any]
    ) -> None:
        """Refuse raw decode parameters declared for a non-raw format.

        Declaring how to decode a self-describing file contradicts the
        file's own header — the declared-never-guessed policy cuts both
        ways, so the contradiction is refused instead of silently ignored.
        """
        declared = [name for name, value in explicit.items() if value is not None]
        if not declared:
            return
        raise ValueError(
            f"File '{filepath}' has extension '{fp.suffix}', a "
            f"self-describing format — declaring raw decode parameter(s) "
            f"{declared} contradicts the file's own header (the "
            f"declared-never-guessed policy cuts both ways). Omit them for "
            f"this file; raw parameters apply only to headerless raw binary "
            f"files ({sorted(RAW_EXTENSIONS)})."
        )

    def _prepare_entry(
        self,
        filepath: str,
        signal_id: Optional[str],
        sampling_rate: Optional[float],
        declared_unit: Optional[str],
        raw_declared: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Resolve path/id, read the array and metadata — no store mutation.

        Args:
            filepath: Filename relative to DATA_DIR, or absolute path.
            signal_id: Caller-chosen id, or None to derive the default.
            sampling_rate: Explicitly declared rate (Hz), or None.
            declared_unit: Already-normalized declared unit, or None.
            raw_declared: Explicitly declared raw decode parameters (see
                :func:`_explicit_raw_params`); None means none declared.
        """
        # Reject a caller-supplied non-positive rate before doing any I/O:
        # a stored 0/negative rate breaks downstream fftfreq(N, 1/rate) with
        # a ZeroDivisionError. (Metadata-derived rates are backstopped by the
        # StoredSignalInfo schema and resolve_signal.)
        if sampling_rate is not None and sampling_rate <= 0:
            raise ValueError(
                f"sampling rate must be positive, got {sampling_rate} — "
                f"pass a sampling_rate > 0 (Hz) or omit it to use the "
                f"companion _metadata.json."
            )

        fp = Path(filepath)
        if not fp.is_absolute():
            fp = DATA_DIR / filepath

        if not fp.exists():
            raise FileNotFoundError(
                f"Signal file not found: {fp} — use list_signals("
                f"scope='disk') to see the files available on disk."
            )

        if raw_declared is None:
            raw_declared = {}
        is_raw = fp.suffix.lower() in RAW_EXTENSIONS
        raw_params: Optional[dict[str, Any]] = None
        meta: dict = {}
        if is_raw:
            # Raw branch: read the companion BEFORE decoding, so
            # companion-declared decode parameters reach the decoder
            # (non-raw formats keep the after-load order below).
            meta = self._read_metadata(fp)
            raw_params = self._merge_raw_params(
                fp, raw_declared, meta.get("source_metadata", {})
            )
            effective_sr = (
                sampling_rate
                if sampling_rate is not None
                else meta.get("sampling_rate")
            )
            self._require_raw_declaration(filepath, fp, raw_params, effective_sr)
        else:
            self._reject_raw_params_on_self_describing(filepath, fp, raw_declared)

        if signal_id is None:
            channel = (
                self._channel_for_id(raw_params) if raw_params is not None else None
            )
            signal_id = self._default_signal_id(fp, channel)

        data = self._load_array(fp, raw_params)
        if data is None:
            raise ValueError(
                f"Unable to load signal from {filepath} — unsupported "
                f"format or read error."
            )

        # Companion metadata. Precedence: explicitly declared parameter >
        # companion _metadata.json > None (undeclared).
        if not is_raw:
            meta = self._read_metadata(fp)
        if sampling_rate is not None:
            meta["sampling_rate"] = sampling_rate
        elif "sampling_rate" not in meta:
            meta["sampling_rate"] = None
        if declared_unit is not None:
            meta["signal_unit"] = declared_unit

        # Freeze the cached array so tools cannot ACCIDENTALLY corrupt the
        # cache in place (read-only views raise on assignment). The
        # repository must OWN the memory (copy once if the loader returned a
        # view into e.g. a pandas block); otherwise the ultimate base would
        # stay writeable and even accidental writes through it could slip
        # past the frozen view. This is not a hard boundary against a caller
        # that deliberately re-enables writes on the base.
        data = np.asarray(data)
        if not data.flags.owndata:
            data = data.copy()
        data.setflags(write=False)

        return {
            "signal_id": signal_id,
            "fp": fp,
            "data": data,
            "meta": meta,
            "raw": raw_params,
        }

    def _insert_entry(self, entry: dict, overwrite: bool) -> dict:
        """Insert a single prepared entry under the lock (collision-checked)."""
        with self._lock:
            return self._insert_locked(entry, overwrite=overwrite)

    def _insert_batch(self, entries: list[dict], overwrite: bool) -> list[dict]:
        """Insert a prepared batch atomically inside one continuous lock.

        The lock is held across the WHOLE critical section: all ids are
        re-checked for collisions first (a concurrent insert could have taken
        one after phase-1 released its lock), and only if every id is clear
        (or overwrite=True) is anything stored. On a collision without
        overwrite nothing is inserted, so the store is never half-populated.
        Memory accounting and LRU eviction stay correct because every insert
        runs through :meth:`_insert_locked` while the lock is held.
        """
        with self._lock:
            if not overwrite:
                collisions = [
                    e["signal_id"] for e in entries if e["signal_id"] in self._store
                ]
                if collisions:
                    ids = ", ".join(f"'{c}'" for c in collisions)
                    raise ValueError(
                        f"Batch load aborted, nothing was loaded — signal_id(s) "
                        f"{ids} were loaded concurrently after validation. Pass "
                        f"overwrite=True to replace them, or retry (the "
                        f"repository is unchanged)."
                    )
            return [self._insert_locked(e, overwrite=overwrite) for e in entries]

    def _insert_locked(self, entry: dict, overwrite: bool) -> dict:
        """Insert a prepared entry into the store. Caller MUST hold the lock."""
        signal_id = entry["signal_id"]
        data = entry["data"]
        meta = entry["meta"]
        size_bytes = data.nbytes

        if signal_id in self._store:
            if not overwrite:
                existing = self._store[signal_id]["info"]["filepath"]
                raise ValueError(
                    f"signal_id '{signal_id}' is already loaded (from "
                    f"'{existing}') — pass overwrite=True to replace it, "
                    f"or choose a different signal_id=..."
                )
            self._remove_entry(signal_id)

        self._evict_if_needed(size_bytes)

        now = datetime.now(timezone.utc).isoformat()
        sr = meta.get("sampling_rate")
        duration = float(len(data)) / sr if sr else None

        info = {
            "signal_id": signal_id,
            "filepath": str(entry["fp"]),
            "load_timestamp": now,
            "shape": list(data.shape),
            "num_samples": len(data),
            "sampling_rate": sr,
            "duration_s": round(duration, 4) if duration else None,
            "size_bytes": size_bytes,
            "signal_unit": meta.get("signal_unit"),
            "source_metadata": meta.get("source_metadata", {}),
            # Provenance: the EFFECTIVE raw decode parameters (after the
            # explicit > companion > default merge); None for
            # self-describing formats.
            "raw_format": entry.get("raw"),
        }
        self._store[signal_id] = {"array": data, "info": info}
        self._current_memory += size_bytes

        logger.info(
            f"Loaded signal '{signal_id}': {len(data)} samples, "
            f"{size_bytes / 1024:.1f} KB"
        )
        return info

    def _load_array(
        self, fp: Path, raw_params: Optional[dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """Read the signal array for an already-resolved absolute path.

        Files under DATA_DIR go through the shared loader; absolute paths
        OUTSIDE DATA_DIR are read directly from THAT path — never via a
        same-named file inside DATA_DIR (the old fallback passed fp.name to
        the DATA_DIR-relative loader, silently loading the wrong file when
        a name collision existed).

        Args:
            fp: Already-resolved absolute path.
            raw_params: EFFECTIVE raw decode parameters for a raw binary
                file, or None for self-describing formats. When set, the
                raw decoder's typed errors PROPAGATE — the raw route never
                goes through load_signal_data's return-None shell.
        """
        try:
            rel = fp.relative_to(DATA_DIR)
        except ValueError:
            return self._load_direct(fp, raw_params)
        if raw_params is not None:
            # Same safe_resolve containment as load_signal_data applies to
            # DATA_DIR-relative names, but OUTSIDE any try/except-None
            # shell: both the containment error and the decoder's typed
            # "problem — remedy" errors must reach the caller intact.
            resolved = safe_resolve(DATA_DIR, str(rel))
            return load_raw_binary(resolved, **raw_params)
        return load_signal_data(str(rel))

    def _remove_entry(self, signal_id: str) -> None:
        """Remove entry and update memory counter. Caller must hold lock."""
        entry = self._store.pop(signal_id)
        self._current_memory -= entry["info"]["size_bytes"]

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict LRU entries until new_size fits. Caller must hold lock."""
        while self._store and self._current_memory + new_size > self._max_memory:
            oldest_id, oldest_entry = next(iter(self._store.items()))
            logger.info(
                f"Evicting signal '{oldest_id}' "
                f"({oldest_entry['info']['size_bytes'] / 1024:.1f} KB) "
                f"to make room"
            )
            self._remove_entry(oldest_id)

    def _read_metadata(self, filepath: Path) -> dict:
        """Read companion _metadata.json if it exists.

        The metadata 'signal_unit' is normalized to the canonical vocabulary;
        an unrecognized value is treated as undeclared (None) with a warning —
        it is never coerced to a default unit. The COMPLETE raw metadata dict
        (rpm, shaft_speed, reference frequencies, ...) is preserved under
        'source_metadata' so get_signal_info can expose it.
        """
        meta_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                raw_unit = meta.get("signal_unit")
                unit = normalize_signal_unit(raw_unit)
                if raw_unit is not None and unit is None:
                    logger.warning(
                        f"Unrecognized signal_unit '{raw_unit}' in {meta_path.name} — "
                        f"treating as undeclared; valid units: {list(VALID_SIGNAL_UNITS)}"
                    )
                return {
                    "sampling_rate": meta.get("sampling_rate"),
                    "signal_unit": unit,
                    "source_metadata": meta if isinstance(meta, dict) else {},
                }
            except Exception as e:
                logger.warning(f"Error reading metadata {meta_path}: {e}")
        return {}

    def _load_direct(
        self, filepath: Path, raw_params: Optional[dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """Fallback loader for absolute paths outside DATA_DIR."""
        # Raw branch FIRST and OUTSIDE the try/except-return-None shell:
        # it delegates to the SAME pure decoder as the DATA_DIR route (no
        # third copy of the logic), and its typed "problem — remedy"
        # errors must propagate instead of collapsing into None.
        if raw_params is not None:
            return load_raw_binary(filepath, **raw_params)
        try:
            if filepath.suffix == ".npy":
                return np.load(filepath)
            elif filepath.suffix in (".csv", ".txt"):
                import pandas as pd

                df = pd.read_csv(filepath, header=None)
                return df.iloc[:, 0].values
            elif filepath.suffix == ".mat":
                from scipy.io import loadmat

                mat = loadmat(str(filepath))
                for k, v in mat.items():
                    if not k.startswith("__") and isinstance(v, np.ndarray):
                        data = v.flatten()
                        if len(data) > 0:
                            return data.astype(np.float64)
        except Exception as e:
            logger.error(f"Direct load failed for {filepath}: {e}")
        return None


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_repository: Optional[SignalRepository] = None
_repo_lock = threading.Lock()


def get_repository() -> SignalRepository:
    """Get or create the global SignalRepository singleton."""
    global _repository
    if _repository is None:
        with _repo_lock:
            # Double-checked locking
            if _repository is None:
                max_gb = float(os.environ.get("PMM_SIGNAL_CACHE_GB", "10"))
                _repository = SignalRepository(max_memory_bytes=int(max_gb * 1024**3))
    return _repository
