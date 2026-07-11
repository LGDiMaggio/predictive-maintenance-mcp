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

from .config import DATA_DIR
from .signal_loader import load_signal_data

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
    ) -> dict:
        """Load a signal file into the repository.

        Args:
            filepath: Filename relative to DATA_DIR, or absolute path.
            signal_id: Custom ID. Defaults to the filename stem.
            sampling_rate: Override sampling rate (Hz). Takes precedence
                over the companion metadata file.
            signal_unit: Declared signal unit — 'g', 'm/s2', 'mm/s', or
                'm/s'. Takes precedence over the companion metadata file.
                Required (here or in metadata) for ISO severity verdicts;
                units are never guessed from amplitude.

        Returns:
            Metadata dict compatible with StoredSignalInfo.

        Raises:
            FileNotFoundError: If signal file does not exist.
            ValueError: If signal data cannot be loaded, or if signal_unit
                is not one of the valid units.
        """
        # Validate the explicitly declared unit up front (fail fast).
        declared_unit: Optional[str] = None
        if signal_unit is not None:
            declared_unit = normalize_signal_unit(signal_unit)
            if declared_unit is None:
                raise ValueError(
                    f"Invalid signal_unit '{signal_unit}' — declare one of "
                    f"{list(VALID_SIGNAL_UNITS)} ('g'/'m/s2' for acceleration, "
                    f"'mm/s'/'m/s' for velocity)."
                )
        # Resolve path
        fp = Path(filepath)
        if not fp.is_absolute():
            fp = DATA_DIR / filepath

        if not fp.exists():
            raise FileNotFoundError(f"Signal file not found: {fp}")

        # Determine signal_id
        if signal_id is None:
            signal_id = fp.stem

        # Load data via existing loader (handles CSV, MAT, WAV, NPY, Parquet)
        # load_signal_data expects filename relative to DATA_DIR
        try:
            rel = fp.relative_to(DATA_DIR)
            data = load_signal_data(str(rel))
        except (ValueError, TypeError):
            # Absolute path outside DATA_DIR — load directly
            data = load_signal_data(fp.name)
            if data is None:
                # Try loading with numpy/pandas directly
                data = self._load_direct(fp)

        if data is None:
            raise ValueError(f"Unable to load signal from {filepath}")

        # Read companion metadata. Precedence: explicitly declared
        # parameter > companion _metadata.json > None (undeclared).
        meta = self._read_metadata(fp)
        if sampling_rate is not None:
            meta["sampling_rate"] = sampling_rate
        elif "sampling_rate" not in meta:
            meta["sampling_rate"] = None
        if declared_unit is not None:
            meta["signal_unit"] = declared_unit

        size_bytes = data.nbytes

        with self._lock:
            # If signal_id already exists, remove old entry first
            if signal_id in self._store:
                self._remove_entry(signal_id)

            # Evict if needed
            self._evict_if_needed(size_bytes)

            # Store
            now = datetime.now(timezone.utc).isoformat()
            sr = meta.get("sampling_rate")
            duration = float(len(data)) / sr if sr else None

            info = {
                "signal_id": signal_id,
                "filepath": str(fp),
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

    def get_signal(self, signal_id: str) -> np.ndarray:
        """Get signal array by ID. Updates LRU order.

        Raises:
            KeyError: If signal_id not found.
        """
        with self._lock:
            if signal_id not in self._store:
                raise KeyError(
                    f"Signal '{signal_id}' not found. "
                    f"Available: {list(self._store.keys())}"
                )
            self._store.move_to_end(signal_id)
            return self._store[signal_id]["array"]

    def get_signal_info(self, signal_id: str) -> dict:
        """Get metadata for a stored signal without touching LRU order."""
        with self._lock:
            if signal_id not in self._store:
                raise KeyError(f"Signal '{signal_id}' not found.")
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
