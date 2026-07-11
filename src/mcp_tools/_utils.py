"""Shared utilities for MCP tool modules.

Path-safety helpers (``safe_resolve``, ``sanitize_filename``) are re-exported
from :mod:`predictive_maintenance_mcp.path_safety`, the canonical single source
of truth, so the containment logic lives in exactly one place.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server.fastmcp import Context

# Canonical path-safety helpers — do not re-implement here (see path_safety.py).
from ..path_safety import (  # noqa: F401
    safe_resolve,
    sanitize_filename,
    validate_name_component,
    resolve_model_paths,
    ModelPaths,
)

logger = logging.getLogger(__name__)


async def load_and_validate_metadata(
    ctx: Context,
    filename: str,
    data_dir: Path,
    load_signal_data_fn,
    provided_sampling_rate: Optional[float],
    provided_segment_duration: Optional[float],
    default_segment_duration: float,
) -> tuple[float, float]:
    """Resolve sampling rate and segment duration for a signal file.

    Sampling-rate discipline (no silent defaults, no sentinel comparisons):

    1. An explicitly provided ``provided_sampling_rate`` wins (``None``
       means "not provided" — never a magic default value).
    2. Otherwise the companion ``<stem>_metadata.json`` value is used.
    3. Otherwise a ValueError is raised naming the fix — analysis never
       proceeds on a guessed sampling rate.

    Args:
        ctx: MCP context for user communication.
        filename: Signal filename.
        data_dir: Base data directory.
        load_signal_data_fn: Callable to load signal arrays (for info logging).
        provided_sampling_rate: Explicitly provided sampling rate, or None.
        provided_segment_duration: Explicitly provided segment duration, or None.
        default_segment_duration: Default segment duration (e.g., 1.0).

    Returns:
        Tuple of (sampling_rate, segment_duration).

    Raises:
        ValueError: If no sampling rate is provided and none can be read
            from the companion metadata file.
    """
    filepath = data_dir / filename
    metadata_file = filepath.parent / (filepath.stem + "_metadata.json")

    # Read companion metadata (if any)
    metadata_sampling_rate: Optional[float] = None
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            metadata_sampling_rate = metadata.get("sampling_rate")
        except Exception as e:
            logger.warning(f"Error reading metadata {metadata_file}: {e}")

    # Resolve sampling rate: explicit parameter > metadata > structured error.
    if provided_sampling_rate is not None:
        sampling_rate = provided_sampling_rate
        if (
            metadata_sampling_rate is not None
            and abs(sampling_rate - metadata_sampling_rate) > 0.1
        ):
            await ctx.info(
                f"Using explicitly provided sampling_rate = {sampling_rate} Hz "
                f"(overrides metadata value {metadata_sampling_rate} Hz)"
            )
        else:
            await ctx.info(f"Using provided sampling_rate = {sampling_rate} Hz")
    elif metadata_sampling_rate is not None:
        sampling_rate = metadata_sampling_rate
        await ctx.info(
            f"Using sampling_rate = {sampling_rate} Hz from {metadata_file.name}"
        )
    else:
        raise ValueError(
            f"No sampling rate for '{filename}' — pass the sampling_rate "
            f"parameter explicitly or add a 'sampling_rate' field to the "
            f"companion metadata file '{metadata_file.name}'. Analysis never "
            f"proceeds on a guessed sampling rate."
        )

    # Segment duration: explicit or documented default (non-critical parameter).
    segment_duration = (
        provided_segment_duration
        if provided_segment_duration is not None
        else default_segment_duration
    )
    await ctx.info(f"Using segment_duration = {segment_duration}s")

    # Informational: signal length/duration
    try:
        signal_data = load_signal_data_fn(filename)
        if signal_data is not None:
            signal_duration_sec = len(signal_data) / sampling_rate
            await ctx.info(
                f"Signal info: {len(signal_data)} samples, "
                f"{signal_duration_sec:.2f}s at {sampling_rate} Hz"
            )
    except Exception as e:
        logger.warning(f"Could not load signal for info: {e}")

    return sampling_rate, segment_duration
