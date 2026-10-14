"""Shared utilities for MCP tool modules.

Path-safety helpers (``safe_resolve``, ``sanitize_filename``) are re-exported
from :mod:`predictive_maintenance_mcp.path_safety`, the canonical single source
of truth, so the containment logic lives in exactly one place.

``refusal_boundary`` is the one adapter between the tools' plain exceptions and
the type the MCP SDK forwards to the model; every ``register()`` applies it.
"""

import functools
import inspect
import logging
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver.exceptions import ToolError

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


# ---------------------------------------------------------------------------
# Refusal boundary (issue #73)
#
# Since mcp 2.3.0, ``Tool.run`` (mcp/server/mcpserver/tools/base.py) forwards
# the text of ``ToolError`` / ``ResourceError`` only; any other exception
# reaches the model as the bare ``Error executing tool <name>`` and its message
# survives only in the server log (mcp 2.0.x appended the text for every
# exception). The tools raise plain ``ValueError`` / ``FileNotFoundError`` for
# user-fixable conditions, with "problem — remedy" messages the model needs,
# and direct callers and the test suite catch those types. So the boundary
# adds ``ToolError`` without taking the original type away.
# ---------------------------------------------------------------------------


class ToolValueError(ToolError, ValueError):
    """A refusal the SDK forwards to the model, still a ``ValueError`` to callers.

    Deriving from both makes the SDK treat the refusal as anticipated (message
    on the wire, INFO log without traceback) while ``except ValueError`` keeps
    working for direct Python callers and for the test suite.
    """


class ToolFileNotFoundError(ToolError, FileNotFoundError):
    """``FileNotFoundError`` counterpart of :class:`ToolValueError`."""


#: The ``src/`` tree this package is installed from. A ``ValueError`` raised
#: below it is a refusal the project wrote for the model; one raised outside
#: it came from a library or the stdlib and nobody wrote it for the model.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _raised_in_this_package(filename: str) -> bool:
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_ROOT)
    except (OSError, ValueError):
        return False  # ``<string>`` and friends: not a project frame


def _log_conversion(
    tool_name: str, exc: BaseException, forwarded_as: type[ToolError]
) -> None:
    """Keep the trail the SDK drops once a refusal is typed as ``ToolError``.

    mcp >= 2.3 logs a ``ToolError`` as one INFO line with no traceback, so a
    ``ValueError`` a library raised (``json.JSONDecodeError`` on a corrupt
    model metadata file, a numpy shape error) would lose the frame that
    locates it. One bounded line records the original type and its
    innermost frame: at WARNING when that frame lies outside this package,
    because such a message was never written for the model, and at DEBUG for
    the refusals the project raised itself. Never with ``exc_info``: the
    ``_OneBoundedLine`` filter in ``server.py`` normalises ``record.msg``
    only, and a formatted traceback would bring back the multi-line record
    it exists to prevent.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        origin = f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}"
        level = (
            logging.DEBUG
            if _raised_in_this_package(frames[-1].filename)
            else logging.WARNING
        )
    else:
        origin, level = "unknown frame", logging.WARNING
    logger.log(
        level,
        "Tool %r: %s.%s raised at %s, forwarded to the model as %s",
        tool_name,
        type(exc).__module__,
        type(exc).__qualname__,
        origin,
        forwarded_as.__name__,
    )


@contextmanager
def _refusals_as_tool_errors(tool_name: str) -> Iterator[None]:
    """Re-raise the refusal types as their ``ToolError`` counterparts.

    ``ToolError`` is checked first because ``ToolValueError`` is itself a
    ``ValueError``: a refusal already typed for the SDK passes through.
    """
    try:
        yield
    except ToolError:
        raise
    except FileNotFoundError as exc:
        _log_conversion(tool_name, exc, ToolFileNotFoundError)
        raise ToolFileNotFoundError(str(exc)) from exc
    except ValueError as exc:
        _log_conversion(tool_name, exc, ToolValueError)
        raise ToolValueError(str(exc)) from exc


def refusal_boundary(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so its refusals cross the wire as ``ToolError``.

    Apply at registration, ``mcp.tool()(refusal_boundary(fn))``, not on the
    module-level function: direct Python callers keep the plain exceptions and
    only the MCP edge learns about the SDK's type. ``functools.wraps`` keeps
    name, docstring, signature and annotations, so the generated input schema,
    the ``ctx`` detection and the frozen tool inventory are unchanged; the
    wrapper is sync or async to match *fn*, because the SDK records
    ``is_async`` and runs sync tools on a worker thread.

    Caught: ``ValueError`` (including pydantic ``ValidationError`` and the
    ledger's ``LedgerError``) and ``FileNotFoundError``, the types the tools
    raise for user-fixable conditions. The SDK then logs them at INFO without
    a traceback, so ``_log_conversion`` keeps the origin frame in this
    package's log: WARNING when a library raised the ``ValueError``, DEBUG
    for the project's own refusals. Any other exception type is a genuine
    crash and keeps the SDK's masking and ERROR traceback.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_boundary(*args: Any, **kwargs: Any) -> Any:
            with _refusals_as_tool_errors(fn.__name__):
                return await fn(*args, **kwargs)

        return async_boundary

    @functools.wraps(fn)
    def sync_boundary(*args: Any, **kwargs: Any) -> Any:
        with _refusals_as_tool_errors(fn.__name__):
            return fn(*args, **kwargs)

    return sync_boundary
