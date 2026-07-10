"""Canonical path-safety helpers — the single source of truth for containment.

Every filesystem path built from a user-supplied string (signal filename, model
name, report filename) must be resolved through :func:`safe_resolve` so it cannot
escape its intended base directory. Model files, which are additionally pickled
and un-pickled, go through :func:`resolve_model_paths` so the name is validated
*and* every derived path is contained in one call.

Housing these here (rather than under ``mcp_tools/``) keeps the deprecated
monolith, the report generator, and the decision-support pipeline all importing
the same implementation instead of each rolling — or drifting from — its own.

.. note::
    ``safe_resolve`` uses :meth:`pathlib.Path.is_relative_to`, not
    ``str.startswith``. The startswith form (used in commit ``61627b0``) was
    bypassable via a sibling directory: ``/data/signals_evil`` passes
    ``startswith("/data/signals")``. See ``tests/test_security_paths.py``.
"""

import re
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "safe_resolve",
    "sanitize_filename",
    "validate_name_component",
    "resolve_model_paths",
    "ModelPaths",
]


class ModelPaths(NamedTuple):
    """The four contained filesystem paths that make up a saved ML model."""

    model: Path
    scaler: Path
    pca: Path
    metadata: Path


def safe_resolve(base_dir: Path, user_input: str) -> Path:
    """Resolve *user_input* under *base_dir* and verify containment.

    Prevents path-traversal attacks (e.g. ``../../etc/passwd``) and the
    sibling-directory bypass (e.g. ``<base>_evil``).

    Args:
        base_dir: The directory the resolved path must stay inside.
        user_input: User-supplied filename / relative path.

    Returns:
        The resolved, validated :class:`Path`.

    Raises:
        ValueError: If the resolved path escapes *base_dir*.
    """
    candidate = (Path(base_dir) / user_input).resolve()
    allowed = Path(base_dir).resolve()
    # Path.is_relative_to (Python 3.9+) avoids the sibling-directory bypass:
    # /data/signals_evil would pass a naive startswith("/data/signals") check.
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


def validate_name_component(name: str, *, kind: str = "name") -> str:
    """Return *name* unchanged if it is already a safe single path component.

    Unlike :func:`sanitize_filename`, this does not silently rewrite the input —
    it rejects anything that is not already clean, so a caller cannot be tricked
    into believing ``../../evil`` was accepted and then writing to a mangled
    ``.._.._evil`` path. Use it as the guard before building model/output paths.

    Args:
        name: The user-supplied component (e.g. a model name).
        kind: Human-readable label used in the error message.

    Raises:
        ValueError: If *name* is empty or contains path separators / unsafe
            characters.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid {kind}: must be a non-empty string.")
    # "." / ".." survive sanitize_filename (they contain only allowed chars) but
    # are never valid names — reject them explicitly.
    if name in (".", ".."):
        raise ValueError(f"Invalid {kind} '{name}': reserved path name.")
    if sanitize_filename(name) != name:
        raise ValueError(
            f"Invalid {kind} '{name}'. Use only alphanumeric, underscore, "
            f"hyphen, or dot characters (no path separators)."
        )
    return name


def resolve_model_paths(models_dir: Path, model_name: str) -> ModelPaths:
    """Validate *model_name* and return the four contained model file paths.

    Single entry point for every model read/write site (train, predict, PCA
    report, diagnosis pipeline) so the containment check cannot be forgotten at
    one call site and present at another.

    Args:
        models_dir: Base directory for model artifacts.
        model_name: User-supplied model name (validated as a single component).

    Returns:
        A :class:`ModelPaths` named tuple (``.model``, ``.scaler``, ``.pca``,
        ``.metadata``) of resolved, contained paths — attribute access is
        statically checkable, unlike a stringly-keyed dict.

    Raises:
        ValueError: If *model_name* is not a safe single path component.
    """
    safe = validate_name_component(model_name, kind="model_name")
    return ModelPaths(
        model=safe_resolve(models_dir, f"{safe}_model.pkl"),
        scaler=safe_resolve(models_dir, f"{safe}_scaler.pkl"),
        pca=safe_resolve(models_dir, f"{safe}_pca.pkl"),
        metadata=safe_resolve(models_dir, f"{safe}_metadata.json"),
    )
