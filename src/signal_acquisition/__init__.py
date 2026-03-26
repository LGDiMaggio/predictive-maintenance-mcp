"""
ISO 13374 Block 1 — Signal Acquisition.

Data loading, format handling, and in-memory signal repository.
"""

from .loaders import (  # noqa: F401
    load_signal_data,
    extract_segment,
    get_metadata_path,
    get_metadata_path_from_dir,
    SUPPORTED_EXTENSIONS,
)
from .repository import (  # noqa: F401
    SignalRepository,
    get_repository,
)
