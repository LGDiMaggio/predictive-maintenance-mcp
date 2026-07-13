"""Shared utilities for MCP tool modules.

Path-safety helpers (``safe_resolve``, ``sanitize_filename``) are re-exported
from :mod:`predictive_maintenance_mcp.path_safety`, the canonical single source
of truth, so the containment logic lives in exactly one place.
"""

import logging

import numpy as np

from ..models import StoredSignalInfo

# Canonical path-safety helpers — do not re-implement here (see path_safety.py).
from ..path_safety import (  # noqa: F401
    safe_resolve,
    sanitize_filename,
    validate_name_component,
    resolve_model_paths,
    ModelPaths,
)
from ..signal_acquisition.repository import get_repository

logger = logging.getLogger(__name__)


def resolve_signal(
    signal_id: str, *, require_sampling_rate: bool = True
) -> tuple[np.ndarray, StoredSignalInfo]:
    """Resolve a stored signal_id to its array and metadata, or raise.

    Single canonical replacement for the repeated
    get-signal-or-explain-what-to-do idiom in the signal_id-based tools.
    The returned array is a READ-ONLY view of the repository cache.

    Args:
        signal_id: ID of a signal previously loaded via load_signal.
        require_sampling_rate: If True (default), raise when the stored
            signal has no sampling rate OR a non-positive one — analysis
            never proceeds on a guessed or invalid rate (a 0/negative rate
            would break downstream fftfreq(N, 1/rate) with a
            ZeroDivisionError).

    Returns:
        Tuple of (signal array, StoredSignalInfo metadata).

    Raises:
        ValueError: If the signal_id is not in the repository (the message
            lists the loaded IDs, explains LRU eviction, and names
            load_signal/list_signals as the remedy), or if
            require_sampling_rate is True and no sampling rate is stored.
    """
    repo = get_repository()
    try:
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
    except KeyError as exc:
        # The repository's standard not-found message (available ids,
        # eviction explanation, load_signal/list_signals remedy).
        raise ValueError(str(exc.args[0])) from None

    stored = StoredSignalInfo(**info)
    if require_sampling_rate and (
        stored.sampling_rate is None or stored.sampling_rate <= 0
    ):
        raise ValueError(
            f"No sampling rate for signal '{signal_id}' — re-load with "
            f"load_signal(filepath=..., sampling_rate=..., overwrite=True) "
            f"or add a 'sampling_rate' field to the companion _metadata.json."
        )
    return signal_data, stored
