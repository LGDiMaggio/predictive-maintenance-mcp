"""MCP tools for signal acquisition and data management (ISO 13374 Block 1)."""

import itertools
import logging
import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR
from ..signal_acquisition.loaders import SUPPORTED_EXTENSIONS
from ..signal_acquisition.repository import get_repository
from ..models import StoredSignalInfo

logger = logging.getLogger(__name__)

#: Process-local sequence for generated test-signal filenames — the Windows
#: clock can return identical timestamps for back-to-back calls, so a
#: monotonic counter guarantees unique filenames (and thus unique ids).
_test_signal_sequence = itertools.count()


# ------------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------------

async def list_signals(
    ctx: Context = None,
    scope: Literal["disk", "memory"] = "disk",
) -> dict[str, Any]:
    """List signal files on disk or signals loaded in the repository.

    scope='disk' (default): files under data/signals/ that load_signal can
    open — use before loading. scope='memory': signals currently cached in
    the in-memory repository with their metadata (signal_id, sampling_rate,
    declared unit) — use to see which signal_ids are available for analysis.

    Args:
        ctx: MCP context.
        scope: 'disk' for loadable files, 'memory' for loaded signal_ids.

    Returns:
        Dict with scope, count, and either 'files' (relative paths, disk)
        or 'signals' (StoredSignalInfo entries, memory).
    """
    if scope == "memory":
        repo = get_repository()
        signals = repo.list_signals()
        if ctx:
            await ctx.info(f"{len(signals)} signal(s) in repository")
        return {
            "scope": "memory",
            "count": len(signals),
            "signals": [StoredSignalInfo(**s).model_dump() for s in signals],
        }

    # scope == "disk"
    files: list[str] = []
    if DATA_DIR.exists():
        for file_path in DATA_DIR.glob("**/*"):
            if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
                rel = file_path.relative_to(DATA_DIR)
                files.append(str(rel).replace("\\", "/"))
    files.sort()
    if ctx:
        await ctx.info(f"{len(files)} signal file(s) on disk")
    return {"scope": "disk", "count": len(files), "files": files}

async def generate_test_signal(
    signal_type: Literal[
        "bearing_fault", "gear_fault", "imbalance", "normal"
    ] = "bearing_fault",
    duration: float = 10.0,
    sampling_rate: float = 10000.0,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None,
    ctx: Context = None,
) -> StoredSignalInfo:
    """Generate a synthetic test signal, save it, and load it into the repository.

    The signal is written to data/signals/ with a timestamped filename and a
    companion _metadata.json declaring sampling_rate and signal_unit='g'
    (synthetic acceleration), then auto-registered in the repository — the
    returned signal_id is immediately usable by every analysis, diagnosis,
    and ISO severity tool with no manual steps.

    Signal content: 'bearing_fault' = 10 Hz impacts modulating a 1 kHz
    carrier; 'gear_fault' = 200 Hz mesh tone + harmonics; 'imbalance' =
    25 Hz (1500 RPM) tone; 'normal' = broadband noise.

    Args:
        signal_type: Synthetic fault pattern to generate.
        duration: Signal duration in seconds (10 s gives 0.1 Hz resolution).
        sampling_rate: Sampling frequency in Hz.
        noise_level: Additive white-noise amplitude.
        random_seed: Seed for reproducible noise (None = non-deterministic).
        ctx: MCP context.

    Returns:
        StoredSignalInfo of the auto-loaded signal (signal_id, declared
        sampling_rate and unit 'g').
    """
    if ctx:
        await ctx.info(f"Generating {signal_type} test signal...")

    rng = np.random.default_rng(random_seed)

    # Time parameters
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # Generate signal based on type
    if signal_type == "bearing_fault":
        # Faulty bearing: periodic impulses + harmonics
        fault_freq = 10.0  # Hz - fault frequency
        carrier_freq = 1000.0  # Hz - carrier frequency

        # Periodic impulses
        impulses = np.zeros_like(t)
        impulse_times = np.arange(0, duration, 1 / fault_freq)
        for imp_time in impulse_times:
            idx = np.argmin(np.abs(t - imp_time))
            impulses[idx] = 1.0

        # Convolution with impulse response
        impulse_response = np.exp(-50 * np.abs(t - t[len(t) // 2]))
        signal_clean = np.convolve(impulses, impulse_response, mode='same')

        # Modulation with carrier
        signal_clean = signal_clean * np.sin(2 * np.pi * carrier_freq * t)

    elif signal_type == "gear_fault":
        # Faulty gear: component at mesh frequency
        mesh_freq = 200.0  # Hz
        signal_clean = np.sin(2 * np.pi * mesh_freq * t)
        # Add harmonics
        signal_clean += 0.5 * np.sin(2 * np.pi * 2 * mesh_freq * t)
        signal_clean += 0.3 * np.sin(2 * np.pi * 3 * mesh_freq * t)

    elif signal_type == "imbalance":
        # Imbalance: 1x RPM component
        rpm = 1500  # RPM
        rotation_freq_hz = rpm / 60.0
        signal_clean = np.sin(2 * np.pi * rotation_freq_hz * t)

    else:  # "normal"
        # Normal signal: broadband noise only
        signal_clean = rng.standard_normal(len(t)) * 0.1

    # Add noise
    noise = rng.standard_normal(len(t)) * noise_level
    signal_data = signal_clean + noise

    # Save the signal with a timestamped filename (consecutive runs never
    # overwrite each other) plus companion metadata so the loaded entry
    # carries a declared rate AND unit — ISO-assessable with no extra steps.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    seq = next(_test_signal_sequence)
    filename = f"test_{signal_type}_{stamp}-{seq:06d}.csv"
    filepath = DATA_DIR / filename
    pd.DataFrame(signal_data, columns=["amplitude"]).to_csv(
        filepath, index=False, header=False
    )

    metadata = {
        "sampling_rate": sampling_rate,
        "signal_unit": "g",
        "generator": "generate_test_signal",
        "signal_type": signal_type,
        "duration_s": duration,
        "noise_level": noise_level,
        "random_seed": random_seed,
    }
    meta_path = DATA_DIR / f"{filepath.stem}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Auto-register: the loop closes here — the caller gets a signal_id
    # that every analysis/diagnosis/report tool accepts immediately.
    repo = get_repository()
    info = repo.load_signal(filename)

    if ctx:
        await ctx.info(
            f"Test signal saved as {filename} and loaded as signal_id "
            f"'{info['signal_id']}' ({signal_type}, {duration}s @ "
            f"{sampling_rate:g} Hz, unit 'g')"
        )

    return StoredSignalInfo(**info)

async def load_signal(
    ctx: Context,
    filepath: str | list[str],
    signal_id: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    signal_unit: Optional[Literal["g", "m/s2", "mm/s", "m/s"]] = None,
    overwrite: bool = False,
) -> StoredSignalInfo | list[StoredSignalInfo]:
    """Load one signal — or a batch — into the in-memory repository.

        Once loaded, reference the signal by its signal_id in every analysis,
        diagnosis, report, and prognostics tool (the load -> analyze ->
        diagnose -> report flow uses signal_id as the single handle).

        Batch form: pass a LIST of file paths (e.g. for training sets). The
        batch is fail-fast and atomic — all paths and derived ids are
        validated up front, and on the first problem ONE error names the
        offending entries and nothing is loaded. One declared sampling_rate/
        signal_unit applies to all files; per-file metadata wins only when
        the parameter is omitted. Custom signal_id is not allowed for a
        batch (ids derive from each file's relative path).

        signal_id default: the path relative to data/signals/ with separators
        replaced by underscores — 'real_train/baseline_1.csv' loads as
        'real_train_baseline_1', so same-named files in different folders
        never collide silently. Re-loading a path whose id already exists is
        an explicit error unless overwrite=True.

        Signal unit discipline: ISO 20816-3 severity verdicts require a
        DECLARED unit — either via this parameter or a 'signal_unit' field in
        the companion _metadata.json (explicit parameter wins). Units are
        never guessed from signal amplitude; without a declared unit the ISO
        severity block is refused with a structured reason and remedy.

        Args:
            filepath: Filename relative to data/signals/ or absolute path —
                or a list of such paths for an atomic batch load.
            signal_id: Custom ID (single-file loads only; default derives
                from the relative path).
            sampling_rate: Sampling rate in Hz (overrides metadata file).
            signal_unit: Declared signal unit — 'g' or 'm/s2' (acceleration),
                'mm/s' or 'm/s' (velocity). Overrides the metadata file.
            overwrite: Replace existing entries on signal_id collision
                instead of raising.

        Returns:
            StoredSignalInfo for a single load; a list of StoredSignalInfo
            (input order) for a batch.

        Raises:
            ValueError: If signal_unit is invalid, the signal data cannot be
                loaded, a signal_id collides without overwrite=True, or a
                batch contains any invalid entry (nothing is loaded).
        """
    repo = get_repository()

    if isinstance(filepath, list):
        if signal_id is not None:
            raise ValueError(
                "signal_id cannot be combined with a batch load — batch ids "
                "derive from each file's relative path. Load files "
                "individually to assign custom ids."
            )
        infos = repo.load_signals(
            filepath,
            sampling_rate=sampling_rate,
            signal_unit=signal_unit,
            overwrite=overwrite,
        )
        if ctx:
            ids = [i["signal_id"] for i in infos]
            await ctx.info(f"Loaded {len(infos)} signals: {ids}")
            undeclared = [
                i["signal_id"] for i in infos if not i.get("signal_unit")
            ]
            if undeclared:
                await ctx.info(
                    f"Signal unit not declared for {undeclared} — ISO "
                        f"severity verdicts will be refused for these until "
                        f"the unit is declared."
                )
        return [StoredSignalInfo(**i) for i in infos]

    info = repo.load_signal(
        filepath,
        signal_id=signal_id,
        sampling_rate=sampling_rate,
        signal_unit=signal_unit,
        overwrite=overwrite,
    )
    if ctx:
        await ctx.info(f"Loaded signal '{info['signal_id']}': {info['num_samples']} samples, {info['size_bytes'] / 1024:.1f} KB")
        if info.get("signal_unit"):
            await ctx.info(f"Signal unit: '{info['signal_unit']}' (declared)")
        else:
            await ctx.info(
                "Signal unit: not declared — ISO severity verdicts will be "
                    "refused until the unit is declared "
                    "(load_signal(signal_unit=...) or metadata 'signal_unit')."
            )
    return StoredSignalInfo(**info)

async def get_signal_info(ctx: Context, signal_id: str) -> StoredSignalInfo:
    """Get metadata for a stored signal without loading the full array.

    Includes the COMPLETE companion-metadata dict (source_metadata: rpm/
    shaft_speed, reference frequencies, ...) alongside the repository
    fields (sampling_rate, declared signal_unit, shape, timestamps).

    Args:
        ctx: MCP context.
        signal_id: ID of a signal previously loaded via load_signal.

    Returns:
        StoredSignalInfo with source_metadata populated from the companion
        _metadata.json (empty dict when the file has none).

    Raises:
        ValueError: If the signal_id is not in the repository.
    """
    repo = get_repository()
    try:
        info = repo.get_signal_info(signal_id)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None
    return StoredSignalInfo(**info)

async def clear_signals(
    ctx: Context, signal_id: Optional[str] = None
) -> dict[str, Any]:
    """Remove one signal — or all signals — from the in-memory repository.

    Args:
        ctx: MCP context.
        signal_id: ID to remove; None (default) clears the whole cache.

    Returns:
        Dict with cleared_count, plus signal_id and status ('removed' or
        'not_found') for single-signal calls.
    """
    repo = get_repository()
    if signal_id is None:
        count = repo.clear_all()
        if ctx:
            await ctx.info(f"Cleared {count} signal(s) from repository")
        return {"cleared_count": count}

    removed = repo.clear_signal(signal_id)
    status = "removed" if removed else "not_found"
    if ctx:
        await ctx.info(f"Signal '{signal_id}': {status}")
    return {
        "signal_id": signal_id,
        "status": status,
        "cleared_count": 1 if removed else 0,
    }


def register(mcp: FastMCP) -> None:
    """Register acquisition tools on the given FastMCP instance."""
    mcp.tool()(list_signals)
    mcp.tool()(generate_test_signal)
    mcp.tool()(load_signal)
    mcp.tool()(get_signal_info)
    mcp.tool()(clear_signals)
