"""
In-memory signal repository with LRU eviction.

Provides a singleton cache for loaded vibration signals, enabling the
signal_id reference pattern: load once, reference by ID across multiple
MCP tool calls without re-reading from disk.

Thread-safe via threading.Lock.
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import DATA_DIR
from .loaders import load_signal_data

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
    ) -> dict:
        """Load a signal file into the repository.

        Args:
            filepath: Filename relative to DATA_DIR, or absolute path.
            signal_id: Custom ID. Defaults to the file's path relative to
                DATA_DIR with separators replaced by underscores (e.g.
                'real_train/baseline_1.csv' -> 'real_train_baseline_1'),
                or the filename stem for files outside DATA_DIR.
            sampling_rate: Override sampling rate (Hz). Takes precedence
                over the companion metadata file.
            signal_unit: Declared signal unit — 'g', 'm/s2', 'mm/s', or
                'm/s'. Takes precedence over the companion metadata file.
                Required (here or in metadata) for ISO severity verdicts;
                units are never guessed from amplitude.
            overwrite: Replace an existing entry with the same signal_id.
                Without it a collision is an explicit error, so two files
                that derive the same id can never silently shadow each
                other.

        Returns:
            Metadata dict compatible with StoredSignalInfo.

        Raises:
            FileNotFoundError: If signal file does not exist.
            ValueError: If signal data cannot be loaded, if signal_unit is
                not one of the valid units, or if signal_id collides with
                an existing entry and overwrite is False.
        """
        declared_unit = self._validate_unit(signal_unit)
        entry = self._prepare_entry(filepath, signal_id, sampling_rate, declared_unit)
        return self._insert_entry(entry, overwrite=overwrite)

    def load_signals(
        self,
        filepaths: list[str],
        sampling_rate: Optional[float] = None,
        signal_unit: Optional[str] = None,
        overwrite: bool = False,
    ) -> list[dict]:
        """Load a batch of signal files atomically (fail-fast, all-or-nothing).

        All paths, derived signal_ids, and collisions are validated up
        front; the arrays are then all read from disk BEFORE anything is
        stored. On the first problem a single actionable ValueError names
        the offending entries and nothing is loaded — the repository is
        never left half-populated.

        Args:
            filepaths: Signal file paths (relative to DATA_DIR or absolute).
                signal_ids are derived from each file's relative path.
            sampling_rate: One sampling rate applied to every file
                (overrides companion metadata). None = per-file metadata.
            signal_unit: One declared unit applied to every file. None =
                per-file metadata.
            overwrite: Allow replacing existing entries with the same ids.

        Returns:
            One StoredSignalInfo-compatible dict per file, in input order.

        Raises:
            ValueError: Empty list, invalid unit, missing files, duplicate
                or colliding signal_ids — raised BEFORE any signal is
                stored.
        """
        if not filepaths:
            raise ValueError(
                "load_signal received an empty list — pass at least one "
                "signal file path (see list_signals() for the files on disk)."
            )
        declared_unit = self._validate_unit(signal_unit)

        # Phase 1: validate every path and every derived id up front.
        problems: list[str] = []
        planned: list[tuple[str, str]] = []  # (filepath, signal_id)
        batch_ids: dict[str, str] = {}
        for raw in filepaths:
            fp = Path(raw)
            if not fp.is_absolute():
                fp = DATA_DIR / raw
            sid = self._default_signal_id(fp)
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

        # Phase 2: read every array from disk before storing anything.
        entries = [
            self._prepare_entry(raw, sid, sampling_rate, declared_unit)
            for raw, sid in planned
        ]

        # Phase 3: insert all.
        return [self._insert_entry(e, overwrite=overwrite) for e in entries]

    def get_signal(self, signal_id: str) -> np.ndarray:
        """Get signal array by ID. Updates LRU order.

        The returned array is a READ-ONLY view of the cached data: tools
        may analyze it freely but cannot corrupt the cache in place
        (in-place writes raise ``ValueError: assignment destination is
        read-only``).

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

        Raises:
            KeyError: If signal_id not found (message names the remedy).
        """
        with self._lock:
            if signal_id not in self._store:
                raise KeyError(self._not_found_message(signal_id))
            return self._store[signal_id]["info"].copy()

    def list_signals(self) -> list[dict]:
        """List all cached signals with metadata."""
        with self._lock:
            return [entry["info"].copy() for entry in self._store.values()]

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

    def _default_signal_id(self, fp: Path) -> str:
        """Derive the default signal_id from the path relative to DATA_DIR.

        'real_train/baseline_1.csv' -> 'real_train_baseline_1', so files
        with the same name in different directories get DISTINCT ids
        (stem-only ids let real_train/baseline_1 and real_test/baseline_1
        silently shadow each other). Files outside DATA_DIR fall back to
        the filename stem.
        """
        try:
            rel = fp.relative_to(DATA_DIR)
        except ValueError:
            return fp.stem
        return "_".join(rel.with_suffix("").parts)

    def _not_found_message(self, signal_id: str) -> str:
        """Standard actionable not-found message. Caller must hold lock."""
        available = list(self._store.keys())
        return (
            f"Signal '{signal_id}' is not in the in-memory repository — it "
            f"was never loaded, or it was evicted (least-recently-used "
            f"signals are dropped when the cache exceeds its "
            f"PMM_SIGNAL_CACHE_GB cap). Currently loaded: "
            f"{available if available else 'none'}. Load it with "
            f"load_signal(filepath=...); use list_signals() to see the "
            f"files available on disk."
        )

    def _prepare_entry(
        self,
        filepath: str,
        signal_id: Optional[str],
        sampling_rate: Optional[float],
        declared_unit: Optional[str],
    ) -> dict:
        """Resolve path/id, read the array and metadata — no store mutation."""
        fp = Path(filepath)
        if not fp.is_absolute():
            fp = DATA_DIR / filepath

        if not fp.exists():
            raise FileNotFoundError(
                f"Signal file not found: {fp} — use list_signals() to see "
                f"the files available on disk."
            )

        if signal_id is None:
            signal_id = self._default_signal_id(fp)

        data = self._load_array(fp)
        if data is None:
            raise ValueError(
                f"Unable to load signal from {filepath} — unsupported "
                f"format or read error."
            )

        # Companion metadata. Precedence: explicitly declared parameter >
        # companion _metadata.json > None (undeclared).
        meta = self._read_metadata(fp)
        if sampling_rate is not None:
            meta["sampling_rate"] = sampling_rate
        elif "sampling_rate" not in meta:
            meta["sampling_rate"] = None
        if declared_unit is not None:
            meta["signal_unit"] = declared_unit

        # Freeze the cached array: tools receive read-only views and can
        # never corrupt the cache in place. The repository must OWN the
        # memory (copy once if the loader returned a view into e.g. a
        # pandas block), otherwise a caller could re-enable writing
        # through the still-writeable ultimate base.
        data = np.asarray(data)
        if not data.flags.owndata:
            data = data.copy()
        data.setflags(write=False)

        return {"signal_id": signal_id, "fp": fp, "data": data, "meta": meta}

    def _insert_entry(self, entry: dict, overwrite: bool) -> dict:
        """Insert a prepared entry into the store (collision-checked)."""
        signal_id = entry["signal_id"]
        data = entry["data"]
        meta = entry["meta"]
        size_bytes = data.nbytes

        with self._lock:
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
            }
            self._store[signal_id] = {"array": data, "info": info}
            self._current_memory += size_bytes

        logger.info(
            f"Loaded signal '{signal_id}': {len(data)} samples, "
            f"{size_bytes / 1024:.1f} KB"
        )
        return info

    def _load_array(self, fp: Path) -> Optional[np.ndarray]:
        """Read the signal array for an already-resolved absolute path.

        Files under DATA_DIR go through the shared loader; absolute paths
        OUTSIDE DATA_DIR are read directly from THAT path — never via a
        same-named file inside DATA_DIR (the old fallback passed fp.name to
        the DATA_DIR-relative loader, silently loading the wrong file when
        a name collision existed).
        """
        try:
            rel = fp.relative_to(DATA_DIR)
        except ValueError:
            return self._load_direct(fp)
        return load_signal_data(str(rel))

    def _remove_entry(self, signal_id: str) -> None:
        """Remove entry and update memory counter. Caller must hold lock."""
        entry = self._store.pop(signal_id)
        self._current_memory -= entry["info"]["size_bytes"]

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict LRU entries until new_size fits. Caller must hold lock."""
        while (
            self._store
            and self._current_memory + new_size > self._max_memory
        ):
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
        it is never coerced to a default unit.
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
                }
            except Exception as e:
                logger.warning(f"Error reading metadata {meta_path}: {e}")
        return {}

    def _load_direct(self, filepath: Path) -> Optional[np.ndarray]:
        """Fallback loader for absolute paths outside DATA_DIR."""
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
                _repository = SignalRepository(
                    max_memory_bytes=int(max_gb * 1024**3)
                )
    return _repository
