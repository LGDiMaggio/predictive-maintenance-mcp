"""Shared utilities for MCP tool modules."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


def safe_resolve(base_dir: Path, user_input: str) -> Path:
    """Resolve *user_input* under *base_dir* and verify containment.

    Prevents path-traversal attacks (e.g. ``../../etc/passwd``).

    Args:
        base_dir: The directory the resolved path must stay inside.
        user_input: User-supplied filename / relative path.

    Returns:
        The resolved, validated :class:`Path`.

    Raises:
        ValueError: If the resolved path escapes *base_dir*.
    """
    candidate = (base_dir / user_input).resolve()
    allowed = base_dir.resolve()
    # Use Path.is_relative_to (Python 3.9+) to avoid sibling-directory bypass
    # e.g. /data/signals_evil would pass a naive startswith("/data/signals") check
    if not candidate.is_relative_to(allowed):
        raise ValueError(f"Invalid path — escapes base directory: {user_input}")
    return candidate


def sanitize_filename(name: str) -> str:
    """Strip directory separators and dangerous characters from *name*.

    Useful when constructing output filenames from user-provided strings.

    Returns:
        A safe filename component containing only ``[a-zA-Z0-9_.-]``.
    """
    stem = Path(name).name  # drop any directory components
    return re.sub(r"[^a-zA-Z0-9_.\-]", "_", stem)


async def load_and_validate_metadata(
    ctx: Context,
    filename: str,
    data_dir: Path,
    load_signal_data_fn,
    provided_sampling_rate: Optional[float],
    default_sampling_rate: float,
    provided_segment_duration: Optional[float],
    default_segment_duration: float,
) -> tuple[float, float]:
    """Load metadata and validate/confirm analysis parameters with user.

    Critical parameter validation strategy:
    1. SAMPLING RATE:
       - Check metadata file first
       - If metadata exists: use it, notify user
       - If no metadata AND user provided: use user value, warn no verification
       - If no metadata AND no user input: CRITICAL WARNING, ask user to confirm

    2. SEGMENT DURATION:
       - Always notify user of value being used
       - Suggest they can modify if needed

    Args:
        ctx: MCP context for user communication.
        filename: Signal filename.
        data_dir: Base data directory.
        load_signal_data_fn: Callable to load signal arrays.
        provided_sampling_rate: Sampling rate provided by user (None if using default).
        default_sampling_rate: Default sampling rate (e.g., 1000.0).
        provided_segment_duration: Segment duration provided by user (None if using default).
        default_segment_duration: Default segment duration (e.g., 1.0).

    Returns:
        Tuple of (validated_sampling_rate, validated_segment_duration).
    """
    filepath = data_dir / filename
    metadata_file = filepath.parent / (filepath.stem + "_metadata.json")

    # Initialize with provided or default values
    sampling_rate = provided_sampling_rate if provided_sampling_rate is not None else default_sampling_rate
    segment_duration = provided_segment_duration if provided_segment_duration is not None else default_segment_duration

    # Check if user explicitly provided values (not using defaults)
    user_provided_sampling_rate = (provided_sampling_rate is not None and provided_sampling_rate != default_sampling_rate)
    user_provided_segment_duration = (provided_segment_duration is not None and provided_segment_duration != default_segment_duration)

    # STEP 1: Validate SAMPLING RATE (CRITICAL)
    metadata_found = False
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if 'sampling_rate' in metadata:
                metadata_sampling_rate = metadata['sampling_rate']
                metadata_found = True

                if user_provided_sampling_rate and abs(sampling_rate - metadata_sampling_rate) > 0.1:
                    await ctx.info(f"⚠️  CONFLICT: User provided {sampling_rate} Hz, but metadata says {metadata_sampling_rate} Hz")
                    await ctx.info(f"   Using METADATA value: {metadata_sampling_rate} Hz (more reliable)")
                    sampling_rate = metadata_sampling_rate
                else:
                    await ctx.info(f"✅ Metadata found: sampling_rate = {metadata_sampling_rate} Hz")
                    sampling_rate = metadata_sampling_rate

    # CRITICAL: No metadata found
    if not metadata_found:
        if user_provided_sampling_rate:
            await ctx.info(f"📌 Using user-provided sampling_rate = {sampling_rate} Hz")
            await ctx.info(f"   ⚠️  No metadata file to verify - cannot confirm correctness")
        else:
            await ctx.info(f"")
            await ctx.info(f"❌ CRITICAL: No metadata found and no sampling_rate provided!")
            await ctx.info(f"")
            await ctx.info(f"   File: {filename}")
            await ctx.info(f"   Expected metadata: {metadata_file.name}")
            await ctx.info(f"")
            await ctx.info(f"   Sampling rate is CRITICAL for frequency analysis accuracy.")
            await ctx.info(f"   Using default {sampling_rate} Hz may give COMPLETELY WRONG results!")
            await ctx.info(f"")
            await ctx.info(f"⚠️  PLEASE CONFIRM:")
            await ctx.info(f"   • Do you know the sampling rate for '{filename}'?")
            await ctx.info(f"   • If YES: Please provide sampling_rate parameter and re-run")
            await ctx.info(f"   • If NO: Results will be UNRELIABLE - interpretation requires caution")
            await ctx.info(f"")
            await ctx.info(f"⚠️  PROCEEDING WITH DEFAULT {sampling_rate} Hz (likely incorrect!)")
            await ctx.info(f"")

    # STEP 2: Validate SEGMENT DURATION (important but less critical)
    if user_provided_segment_duration:
        await ctx.info(f"📊 Using segment_duration = {segment_duration}s (user-provided)")
    else:
        await ctx.info(f"📊 Using segment_duration = {segment_duration}s (default)")
        await ctx.info(f"   💡 You can modify by providing segment_duration parameter")

    # Calculate signal info
    try:
        signal_data = load_signal_data_fn(filename)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {filename}")
        signal_duration_sec = len(signal_data) / sampling_rate
        await ctx.info(f"")
        await ctx.info(f"📏 Signal info: {len(signal_data)} samples, {signal_duration_sec:.2f}s duration at {sampling_rate} Hz")

        if segment_duration is not None and segment_duration < signal_duration_sec:
            await ctx.info(f"   Analyzing {segment_duration}s segment from {signal_duration_sec:.2f}s total")
        else:
            await ctx.info(f"   Analyzing full signal")
        await ctx.info(f"")
    except Exception as e:
        logger.warning(f"Could not load signal for info: {e}")

    return sampling_rate, segment_duration
