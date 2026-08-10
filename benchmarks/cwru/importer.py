"""Importer: cached CWRU ``.mat`` files → opaque signals in the repository.

This is the adapter between the vendored CWRU metadata and the server's
real ingestion path. All vendor semantics (MATLAB variable naming, cache
filenames) stay inside this module; everything downstream sees only
opaque ids and a minimal, label-free companion (origin acceptance
example AE3).

Per record the importer:

1. reads the cached ``.mat`` (downloaded and verified by
   :mod:`benchmarks.cwru.download`) and selects EXACTLY the declared
   ``internal_mat_key`` — never "the first numeric variable" (Smith &
   Randall 2015 document stray and duplicated variables in some files;
   the maintainer freeze step pins the right key per record);
2. refuses, fail closed with a "problem — remedy" error and nothing
   written, when the key is unpinned, absent from the file, or not a
   1-D-coercible numeric array. **Coercibility rule:** a 1-D vector or a
   2-D column/row vector (``Nx1``/``1xN``) is coercible via ``ravel``;
   an empty array, a true multi-column matrix, and any non-numeric or
   complex variable are refused — there is no unambiguous single channel
   in them;
3. writes ``data/signals/cwru/<opaque_id>.npy`` (1-D float64) plus a
   companion ``<opaque_id>_metadata.json`` containing ONLY
   ``sampling_rate`` (declared in the ops table from the dataset
   documentation) and ``signal_unit`` (:data:`SIGNAL_UNIT`) — nothing
   else, because ``get_signal_info`` surfaces the whole companion and
   the blind protocol depends on it. Both files are promoted atomically
   from ``.part`` temporaries, and every path is built through the
   repo's ``path_safety`` choke point;
4. loads the converted file through the REAL ingestion path
   (``SignalRepository.load_signal``) under the opaque ``signal_id``,
   with sampling rate and unit coming from the companion — exercising
   the same companion-metadata contract production loads use.

Re-import is explicit: an existing converted file or repository entry is
refused unless ``overwrite=True``, which is threaded through to
``load_signal``'s own overwrite flag so files and cache replace together.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.io import loadmat

from benchmarks.cwru.download import DEFAULT_CACHE_ROOT
from benchmarks.cwru.records import OpsRecord
from predictive_maintenance_mcp.config import DATA_DIR
from predictive_maintenance_mcp.path_safety import safe_resolve
from predictive_maintenance_mcp.signal_acquisition.repository import (
    SignalRepository,
    get_repository,
)

__all__ = [
    "DEFAULT_SIGNALS_DIR",
    "SIGNAL_UNIT",
    "import_record",
    "import_records",
]

#: Converted opaque signals live INSIDE the repository containment
#: boundary (``DATA_DIR = data/signals``) so relative loads and signal
#: listing can reach them; the directory is gitignored (no CWRU data is
#: ever committed).
DEFAULT_SIGNALS_DIR: Path = Path(DATA_DIR) / "cwru"

#: Declared signal unit for every CWRU record: acceleration in g, per the
#: community convention for this dataset. A declaration sourced from
#: documentation, not a guess from amplitude — stated as an assumption in
#: the benchmark methodology.
SIGNAL_UNIT: str = "g"

_FREEZE_KEY_REMEDY = (
    "The declared .mat variable is pinned per record by the maintainer "
    "freeze step (python -m benchmarks.cwru freeze), when the real files "
    "are first inspected — pin internal_mat_key in records_ops.json "
    "before importing."
)


def _resolve_mat_path(record: OpsRecord, mat_path: Optional[Path]) -> Path:
    """Resolve the cached ``.mat`` path for *record* and require it exists.

    Args:
        record: The ops record being imported.
        mat_path: Explicit override (tests/maintainer only); ``None``
            resolves ``cache_filename`` under the U2 cache root through
            the path-safety choke point.

    Returns:
        The existing ``.mat`` path.

    Raises:
        ValueError: If the file is not in the cache — the remedy is the
            download step, never a silent fetch from here (download and
            import are separate, individually verifiable stages).
    """
    resolved = (
        Path(mat_path)
        if mat_path is not None
        else safe_resolve(DEFAULT_CACHE_ROOT, record.cache_filename)
    )
    if not resolved.exists():
        raise ValueError(
            f"Cached .mat for record '{record.opaque_id}' not found at "
            f"{resolved} — run the download step first (python -m "
            f"benchmarks.cwru download, or "
            f"benchmarks.cwru.download.ensure_cached) so a "
            f"checksum-verified file is in the cache."
        )
    return resolved


def _extract_declared_channel(record: OpsRecord, mat_path: Path) -> np.ndarray:
    """Read *mat_path* and return EXACTLY the declared channel, 1-D float64.

    Args:
        record: The ops record whose ``internal_mat_key`` names the
            channel (guaranteed non-``None`` by the caller).
        mat_path: The cached ``.mat`` file to read.

    Returns:
        The declared channel flattened to a 1-D ``float64`` array.

    Raises:
        ValueError: "Problem — remedy", nothing written, when the file
            cannot be parsed, the declared key is absent, or the variable
            is not a 1-D-coercible numeric array (see the module
            docstring for the coercibility rule).
    """
    try:
        mat = loadmat(str(mat_path))
    except Exception as exc:
        raise ValueError(
            f"Cached .mat for record '{record.opaque_id}' at {mat_path} "
            f"cannot be read ({exc}) — the cache file is corrupt or not a "
            f"MATLAB file. Delete it and re-run the download step so a "
            f"checksum-verified copy replaces it."
        ) from exc

    candidates = sorted(key for key in mat if not key.startswith("__"))
    declared = record.internal_mat_key
    if declared not in mat:
        raise ValueError(
            f"Record '{record.opaque_id}': declared .mat variable "
            f"'{declared}' is not present in {mat_path} — available "
            f"candidate variables: {candidates}. Fix internal_mat_key in "
            f"records_ops.json (S&R 2015 document stray/duplicated "
            f"variables in some files; the freeze step pins the right key "
            f"per record). Nothing was written."
        )

    value = mat[declared]
    problem: Optional[str] = None
    if not isinstance(value, np.ndarray):
        problem = f"is {type(value).__name__}, not an array"
    elif not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        problem = f"has non-real dtype {value.dtype}"
    elif value.size == 0:
        problem = f"is empty (shape {value.shape})"
    elif value.ndim > 2 or (value.ndim == 2 and 1 not in value.shape):
        problem = (
            f"has shape {value.shape} — a multi-dimensional matrix has no "
            f"unambiguous single channel (only 1-D vectors and Nx1/1xN "
            f"columns are coercible)"
        )
    if problem is not None:
        raise ValueError(
            f"Record '{record.opaque_id}': declared .mat variable "
            f"'{declared}' in {mat_path} is not a 1-D-coercible numeric "
            f"array — it {problem}. Fix internal_mat_key in "
            f"records_ops.json to name the intended accelerometer "
            f"channel; available candidate variables: {candidates}. "
            f"Nothing was written."
        )
    flat: np.ndarray = value.ravel().astype(np.float64)
    return flat


def _refuse_existing(
    record: OpsRecord,
    npy_path: Path,
    companion_path: Path,
    repository: SignalRepository,
) -> None:
    """Refuse a re-import that was not made explicit with ``overwrite``.

    Both layers are checked: converted files on disk (which survive a
    process restart) and the in-memory repository entry (which an
    unrelated load could occupy). Checked BEFORE anything is written, so
    a refusal leaves the previous import byte-identical.

    Raises:
        ValueError: Naming exactly what already exists and the
            ``overwrite=True`` remedy.
    """
    existing: list[str] = []
    if npy_path.exists():
        existing.append(f"converted signal {npy_path.name}")
    if companion_path.exists():
        existing.append(f"companion {companion_path.name}")
    try:
        repository.get_signal_info(record.opaque_id)
    except KeyError:
        pass
    else:
        existing.append(f"repository entry '{record.opaque_id}'")
    if existing:
        raise ValueError(
            f"Record '{record.opaque_id}' is already imported "
            f"({'; '.join(existing)}) — re-import must be explicit. Pass "
            f"overwrite=True to replace the converted signal, its "
            f"companion, and the repository entry together. Nothing was "
            f"written."
        )


def _atomic_write_npy(path: Path, samples: np.ndarray) -> None:
    """Write *samples* to *path* via a ``.part`` temporary + atomic rename."""
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as sink:
        np.save(sink, samples)
    os.replace(part, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write *payload* to *path* via a ``.part`` temporary + atomic rename."""
    part = path.with_name(path.name + ".part")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _verify_declarations_honored(
    record: OpsRecord, companion_path: Path, info: dict
) -> None:
    """Fail closed if the load did not carry the declared rate and unit.

    The companion is the single declaration source on this path; the
    repository swallows an unreadable companion into a logged warning, so
    without this check a corrupt write would produce a silently
    undeclared signal instead of a refusal.

    Raises:
        ValueError: If the loaded metadata diverges from the declaration.
    """
    if (
        info.get("sampling_rate") != record.fs_hz
        or info.get("signal_unit") != SIGNAL_UNIT
    ):
        raise ValueError(
            f"Record '{record.opaque_id}' loaded with "
            f"sampling_rate={info.get('sampling_rate')!r}, "
            f"signal_unit={info.get('signal_unit')!r} instead of the "
            f"declared ({record.fs_hz}, '{SIGNAL_UNIT}') — the companion "
            f"{companion_path.name} was not honored (corrupt or "
            f"unreadable write?). Re-import the record with "
            f"overwrite=True."
        )


def import_record(
    record: OpsRecord,
    *,
    mat_path: Optional[Path] = None,
    overwrite: bool = False,
    signals_dir: Optional[Path] = None,
    repository: Optional[SignalRepository] = None,
) -> dict[str, object]:
    """Import one record's declared channel into the signal repository.

    Extracts exactly ``record.internal_mat_key`` from the cached ``.mat``,
    writes the opaque ``.npy`` + minimal companion (see module docstring),
    and loads it through ``SignalRepository.load_signal`` under
    ``record.opaque_id`` — the real ingestion path, never a bypass.

    Args:
        record: Ops record to import (the sole source of key, rate, and
            names — no label ever reaches this function).
        mat_path: Cached ``.mat`` override (tests only); defaults to the
            record's file under the U2 cache root.
        overwrite: Replace an existing import (files AND repository
            entry). Without it any existing trace refuses, fail closed.
        signals_dir: Converted-signal directory override (tests only);
            defaults to :data:`DEFAULT_SIGNALS_DIR`.
        repository: Repository override (tests only); defaults to the
            process singleton ``get_repository()``.

    Returns:
        Summary dict: ``signal_id``, ``npy_path``, ``companion_path``,
        ``num_samples``, ``sampling_rate``, ``signal_unit``.

    Raises:
        ValueError: "Problem — remedy", with nothing written, when the
            key is unpinned/absent/not coercible, the cache file is
            missing or unreadable, or the record is already imported
            without ``overwrite``.
    """
    if record.internal_mat_key is None:
        raise ValueError(
            f"Record '{record.opaque_id}' has no internal_mat_key pinned "
            f"(null in records_ops.json) — the importer never guesses "
            f"which MATLAB variable is the signal. {_FREEZE_KEY_REMEDY}"
        )

    resolved_mat = _resolve_mat_path(record, mat_path)
    samples = _extract_declared_channel(record, resolved_mat)

    target_dir = Path(signals_dir) if signals_dir is not None else DEFAULT_SIGNALS_DIR
    npy_path = safe_resolve(target_dir, f"{record.opaque_id}.npy")
    companion_path = safe_resolve(target_dir, f"{record.opaque_id}_metadata.json")
    repo = repository if repository is not None else get_repository()

    if not overwrite:
        _refuse_existing(record, npy_path, companion_path, repo)

    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(npy_path, samples)
    # ONLY the two declarations — get_signal_info surfaces the whole
    # companion, so any extra key here would leak straight to the system
    # under test (AE3).
    _atomic_write_json(
        companion_path,
        {"sampling_rate": record.fs_hz, "signal_unit": SIGNAL_UNIT},
    )

    info = repo.load_signal(
        str(npy_path), signal_id=record.opaque_id, overwrite=overwrite
    )
    _verify_declarations_honored(record, companion_path, info)

    return {
        "signal_id": record.opaque_id,
        "npy_path": str(npy_path),
        "companion_path": str(companion_path),
        "num_samples": int(samples.size),
        "sampling_rate": record.fs_hz,
        "signal_unit": SIGNAL_UNIT,
    }


def import_records(
    records: Iterable[OpsRecord],
    *,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
    signals_dir: Optional[Path] = None,
    repository: Optional[SignalRepository] = None,
) -> list[dict[str, object]]:
    """Import a batch of records, failing fast on the first problem.

    Convenience wrapper for the runner: each record goes through
    :func:`import_record` in order. The first failure propagates
    immediately (its message already names the offending record);
    records imported earlier in the batch stay imported — re-run with
    ``overwrite=True`` after fixing the problem, and the U4 ``all``
    dispatch refuses to score on a partial import either way.

    Args:
        records: Ops records to import, in order.
        cache_dir: Cache root override (tests only); each record's
            ``cache_filename`` is resolved under it through the
            path-safety choke point. ``None`` uses the U2 default cache.
        overwrite: Threaded to every :func:`import_record` call.
        signals_dir: Threaded to every :func:`import_record` call.
        repository: Threaded to every :func:`import_record` call.

    Returns:
        One :func:`import_record` summary dict per record, in order.

    Raises:
        ValueError: The first per-record failure, unchanged.
    """
    results: list[dict[str, object]] = []
    for record in records:
        mat_path = (
            safe_resolve(Path(cache_dir), record.cache_filename)
            if cache_dir is not None
            else None
        )
        results.append(
            import_record(
                record,
                mat_path=mat_path,
                overwrite=overwrite,
                signals_dir=signals_dir,
                repository=repository,
            )
        )
    return results
