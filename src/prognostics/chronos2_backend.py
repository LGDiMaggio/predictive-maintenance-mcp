"""
ISO 13374 Block 5 — Prognostic Assessment: the optional Chronos-2 backend.

The adapter behind ``forecast_asset_indicator(backend="chronos2")``:

- availability: the optional ``forecast-chronos2`` extra is installed, its
  ``chronos-forecasting``, ``torch`` and ``transformers`` versions are inside
  the supported ranges, and ``PMM_FORECAST_MODEL_DIR`` names a local directory
  holding ``config.json`` and ``model.safetensors``;
- integrity: both files match the size and SHA-256 published for the pinned
  revision before anything is loaded;
- execution: one adapter-owned worker thread runs, for its whole life, the
  ``torch`` and ``chronos`` imports, the digest checks, the load and every
  inference. Nothing of it runs on the event loop, and the reference backends
  never wait on it;
- precision: the window is centred on its last observed value in float64
  before the model's float32 cast, and the value is added back to the
  quantiles in float64.

Nothing heavy is imported at module import. The model is loaded only from the
verified local directory, with ``local_files_only``, safetensors only and no
remote code, so no request reaches the Hugging Face Hub. The adapter sets no
environment variable (no process-wide offline switch), so other tools keep
their own Hugging Face settings.

Settings are read at each call, like ``PMM_MAX_SIGNAL_SIZE``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import logging
import math
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..signal_validation import InvalidInputError

logger = logging.getLogger(__name__)

REPO_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
#: Size in bytes and SHA-256 of each file of REVISION as published on the Hub,
#: checked in this order. Matching them is how the revision is verified.
MODEL_FILES: tuple[tuple[str, int, str], ...] = (
    (
        CONFIG_FILE,
        1_067,
        "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
    ),
    (
        WEIGHTS_FILE,
        477_930_472,
        "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
    ),
)
#: Licence declared in the model card metadata of REPO_ID.
WEIGHTS_LICENCE = "Apache-2.0"

#: Operating envelope of the backend: at most this many of the most recent grid
#: bins go to the model, at least MIN_OBSERVED_BINS of them observed, and the
#: horizon is at most MAX_HORIZON_STEPS. These are the evaluated limits, not
#: validated forecasting thresholds.
MAX_CONTEXT_BINS = 2048
MIN_OBSERVED_BINS = 64
MAX_HORIZON_STEPS = 144
QUANTILE_LEVELS = (0.1, 0.5, 0.9)

#: Supported versions: the bounds of the forecast-chronos2 extra in
#: pyproject.toml. Change both together.
SUPPORTED_RANGES: tuple[tuple[str, str], ...] = (
    ("chronos-forecasting", "==2.3.2"),
    ("torch", ">=2.13,<2.15"),
    ("transformers", ">=5.14,<5.21"),
)
#: Distributions whose installed versions every projection records; each must
#: be installed for the backend to be available.
RECORDED_DISTRIBUTIONS = (
    "chronos-forecasting",
    "torch",
    "transformers",
    "accelerate",
    "huggingface-hub",
)

ENV_MODEL_DIR = "PMM_FORECAST_MODEL_DIR"
ENV_THREADS = "PMM_FORECAST_MODEL_THREADS"
ENV_PRELOAD = "PMM_FORECAST_MODEL_PRELOAD"
ENV_LOAD_WAIT = "PMM_FORECAST_MODEL_LOAD_WAIT_S"
DEFAULT_THREADS = 4
THREADS_RANGE = (1, 64)
DEFAULT_LOAD_WAIT_S = 30.0
LOAD_WAIT_RANGE = (0.0, 120.0)
_ON_VALUES = frozenset({"1", "true", "yes", "on"})

#: Load attempts per server process: the first and one retry after a failure.
MAX_LOAD_ATTEMPTS = 2

_GUIDE = "the forecasting model guide"

EXTRA_NOT_INSTALLED = (
    "backend 'chronos2' is not available on this server: the optional "
    "'forecast-chronos2' dependencies are not installed. An operator can install "
    f"them as described in {_GUIDE}; on Linux, install the CPU-only build of torch "
    "first to avoid the CUDA libraries. The extra cannot be installed on macOS with "
    "an Intel processor or on Python 3.15 or later, because no torch build in the "
    "supported range is published for them. Until then, use backend 'naive' or "
    "'seasonal_naive'."
)
DIR_NOT_SET = (
    f"backend 'chronos2' is not available: {ENV_MODEL_DIR} is not set. An operator "
    f"must download {REPO_ID} at revision {REVISION} into a local directory, check "
    f"its SHA-256 values as described in {_GUIDE}, and set {ENV_MODEL_DIR} to that "
    "directory. The server never downloads model files itself. Until then, use "
    "backend 'naive' or 'seasonal_naive'."
)
NOT_A_MODEL_DIR = (
    f"backend 'chronos2' is not available: {ENV_MODEL_DIR} is {{path}}, which is not "
    f"a local directory holding {CONFIG_FILE} and {WEIGHTS_FILE}. Download {REPO_ID} "
    f"at revision {REVISION} into that directory as described in {_GUIDE}, or point "
    f"{ENV_MODEL_DIR} to the directory that holds it. The server never downloads "
    "model files itself."
)
DIGEST_MISMATCH = (
    f"backend 'chronos2' did not load the model: {{file}} in {ENV_MODEL_DIR} does not "
    f"match the SHA-256 published for revision {REVISION[:8]} of {REPO_ID}, so the "
    "files cannot be verified as that revision. Download that exact revision again "
    f"into an empty directory and check both SHA-256 values as described in {_GUIDE}. "
    "Unverified model files are never loaded."
)
VERSION_OUTSIDE_RANGE = (
    "backend 'chronos2' is not available: the installed {package} {version} is "
    "outside the range this backend supports ({range}). Reinstall the "
    "'forecast-chronos2' extra so that its declared versions are used, as described "
    f"in {_GUIDE}. Until then, use backend 'naive' or 'seasonal_naive'."
)
LOAD_FAILED = (
    "backend 'chronos2' could not load the verified model files. The server log "
    "records the cause. Check that the 'forecast-chronos2' dependencies are "
    f"installed as described in {_GUIDE}, then restart the server. Until then, use "
    "backend 'naive' or 'seasonal_naive'."
)
STILL_LOADING = (
    "backend 'chronos2' is still loading its model in the background, and no "
    "projection was made from this call. The first load can take from several "
    "seconds to more than a minute, depending on the machine and the disk. Call "
    f"again in about a minute. An operator can set {ENV_PRELOAD}=1 so that the "
    "model loads when the server starts."
)


class Chronos2Refusal(InvalidInputError):
    """The chronos2 backend cannot serve this call; the text says why and what to do."""


class _WaitTimeout(Exception):
    """The wait for a worker job ended before the job did; the job goes on."""


class _DigestMismatch(Exception):
    """A model file does not match the published size and SHA-256."""

    def __init__(self, file: str) -> None:
        super().__init__(file)
        self.file = file


# ---------------------------------------------------------------------------
# Settings, read at each call
# ---------------------------------------------------------------------------


def _bounded_setting(
    name: str, default: float, bounds: tuple[float, float], kind: Callable[[Any], Any]
) -> Any:
    """An operator setting: the default when unset or not a number, clamped to bounds."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        value = None
    if value is None or not math.isfinite(value):
        logger.warning("%s=%r is not a usable number; using %s", name, raw, default)
        return default
    low, high = bounds
    if not low <= value <= high:
        clamped = kind(min(max(value, low), high))
        logger.warning(
            "%s=%r is outside %s to %s; using %s", name, raw, low, high, clamped
        )
        return clamped
    return value


def model_threads() -> int:
    """PyTorch threads for each projection: ``PMM_FORECAST_MODEL_THREADS``."""
    return int(_bounded_setting(ENV_THREADS, DEFAULT_THREADS, THREADS_RANGE, int))


def load_wait_seconds() -> float:
    """Seconds a call waits for a load in progress: ``PMM_FORECAST_MODEL_LOAD_WAIT_S``."""
    return float(
        _bounded_setting(ENV_LOAD_WAIT, DEFAULT_LOAD_WAIT_S, LOAD_WAIT_RANGE, float)
    )


def preload_requested() -> bool:
    """Whether ``PMM_FORECAST_MODEL_PRELOAD`` asks for a load at server start."""
    return os.environ.get(ENV_PRELOAD, "").strip().lower() in _ON_VALUES


# ---------------------------------------------------------------------------
# Availability (steps 1 to 4): no import of torch, transformers or chronos
# ---------------------------------------------------------------------------


def installed_versions() -> dict[str, str]:
    """Installed version of each recorded distribution, from package metadata.

    Raises:
        Chronos2Refusal: When one of them is not installed (the extra is not).
    """
    versions = {}
    for name in RECORDED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise Chronos2Refusal(EXTRA_NOT_INSTALLED) from None
    return versions


def _outside(version: str, spec: str) -> bool:
    # packaging is a dependency of transformers, so it is present whenever the
    # extra is; it is imported here so that the default install never needs it.
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        return not SpecifierSet(spec).contains(Version(version))
    except InvalidVersion:
        return True


def check_available() -> Path:
    """Run the availability checks a chronos2 call needs before any load.

    In order: the extra is installed; its pinned and bounded versions are
    inside the supported ranges; ``PMM_FORECAST_MODEL_DIR`` is set; it is a
    directory holding both model files. A value shaped like a repository id is
    not a local directory, so it is refused here and never reaches a loader.

    Returns:
        The model directory, absolute.

    Raises:
        Chronos2Refusal: With the text of the first check that fails.
    """
    versions = installed_versions()
    for package, spec in SUPPORTED_RANGES:
        if _outside(versions[package], spec):
            raise Chronos2Refusal(
                VERSION_OUTSIDE_RANGE.format(
                    package=package, version=versions[package], range=spec
                )
            )
    raw = os.environ.get(ENV_MODEL_DIR)
    if raw is None or not raw.strip():
        raise Chronos2Refusal(DIR_NOT_SET)
    directory = Path(raw)
    try:
        holds_files = directory.is_dir() and all(
            (directory / name).is_file() for name, _, _ in MODEL_FILES
        )
    except (OSError, ValueError):
        holds_files = False
    if not holds_files:
        raise Chronos2Refusal(NOT_A_MODEL_DIR.format(path=raw))
    return directory.resolve()


# ---------------------------------------------------------------------------
# Worker-side steps (5 to 7, and inference)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify_files(directory: Path) -> None:
    """Steps 5 and 6: each file's size, then its SHA-256, in MODEL_FILES order.

    Raises:
        _DigestMismatch: Naming the first file that does not match.
    """
    for name, size, sha256 in MODEL_FILES:
        path = directory / name
        if path.stat().st_size != size or _sha256(path) != sha256:
            raise _DigestMismatch(name)


#: Files whose presence makes ``Chronos2Pipeline.from_pretrained`` load other
#: files than the two verified ones (an adapter names its own base model and
#: weights), so a directory holding one is never loaded.
UNVERIFIED_REDIRECTS = ("adapter_config.json",)


def _refuse_redirects(directory: Path) -> None:
    """Raise when the directory holds a file that would redirect the load."""
    for name in UNVERIFIED_REDIRECTS:
        if (directory / name).exists():
            raise RuntimeError(
                f"{name} in {directory} would make the loader read files that are "
                f"not verified; remove it to load the pinned model files"
            )


def _import_and_load(directory: Path) -> Any:
    """Step 7: import the libraries and load the pipeline from local files only."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        str(directory),
        device_map="cpu",
        local_files_only=True,
        use_safetensors=True,
    )


def _set_threads(threads: int) -> int:
    """Set the PyTorch thread count on the calling (worker) thread; return it as read there."""
    import torch

    torch.set_num_threads(threads)
    return int(torch.get_num_threads())


def _predict(pipeline: Any, context: np.ndarray, horizon: int) -> np.ndarray:
    """One inference: quantiles at QUANTILE_LEVELS, shape (horizon, 3), as returned."""
    quantiles, _ = pipeline.predict_quantiles(
        [{"target": context}],
        prediction_length=horizon,
        quantile_levels=list(QUANTILE_LEVELS),
        batch_size=1,
        cross_learning=False,
    )
    return np.asarray(quantiles[0][0], dtype=np.float64)


@dataclass(frozen=True)
class LoadedModel:
    """A pipeline loaded from verified files, with the versions read at load."""

    pipeline: Any
    versions: dict[str, str]


def _load_job(directory: Path, attempt: "_LoadAttempt") -> None:
    """Worker job: verify, then load. Reports through the attempt's futures."""
    try:
        _verify_files(directory)
    except _DigestMismatch as exc:
        logger.warning(
            "chronos2 model file %s does not match the pinned SHA-256; not loaded",
            exc.file,
        )
        # The slot settles (and records the mismatch) before any waiter wakes.
        attempt.loaded.set_exception(exc)
        attempt.verified.set_exception(exc)
        return
    except BaseException as exc:  # an unreadable file is a load failure
        logger.error("chronos2 model files could not be read", exc_info=True)
        attempt.verified.set_result(None)
        attempt.loaded.set_exception(exc)
        return
    attempt.verified.set_result(None)
    try:
        _refuse_redirects(directory)
        pipeline = _import_and_load(directory)
        versions = installed_versions()
    except BaseException as exc:
        logger.error("chronos2 model load failed", exc_info=True)
        attempt.loaded.set_exception(exc)
        return
    attempt.loaded.set_result(LoadedModel(pipeline, versions))


def _predict_job(
    model: LoadedModel, context: np.ndarray, horizon: int, threads: int
) -> tuple[np.ndarray, int]:
    """Worker job: set the thread count here, read it back, run one inference."""
    read = _set_threads(threads)
    return _predict(model.pipeline, context, horizon), read


# ---------------------------------------------------------------------------
# The worker and the load state
# ---------------------------------------------------------------------------


class _Worker:
    """One daemon thread that runs every job of the adapter, in submission order."""

    def __init__(self, name: str = "pmm-chronos2-worker") -> None:
        self._name = name
        self._jobs: "queue.SimpleQueue[tuple[Future, Callable[..., Any], tuple]]" = (
            queue.SimpleQueue()
        )
        self._lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        future: Future = Future()
        with self._lock:
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._run, name=self._name, daemon=True
                )
                self.thread.start()
        self._jobs.put((future, fn, args))
        return future

    def _run(self) -> None:
        while True:
            future, fn, args = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # keep the worker alive for the next job
                future.set_exception(exc)


@dataclass
class _LoadAttempt:
    """One verify-then-load run in the worker, shared by every waiting call."""

    verified: Future = field(default_factory=Future)
    loaded: Future = field(default_factory=Future)


class _LoadSlot:
    """The process's one model: not_loaded, loading, ready or failed.

    Single flight: concurrent calls wait on the same attempt. A failed load is
    reported once, then retried once by the next call; a second failure stays
    until the server restarts. A digest mismatch is a refusal before any load:
    it leaves the slot not_loaded, so a corrected directory is verified again;
    when no call was waiting for it, the next call reports it once first.
    """

    def __init__(self, worker: _Worker) -> None:
        self._worker = worker
        self._lock = threading.Lock()
        self._state = "not_loaded"
        self._attempt: Optional[_LoadAttempt] = None
        self._model: Optional[LoadedModel] = None
        self._failures = 0
        self._failure_reported = False
        self._mismatch: Optional[str] = None  # a file no call has reported yet

    @property
    def state(self) -> str:
        return self._state

    def _start(self, directory: Path) -> _LoadAttempt:
        # Called with the lock held.
        attempt = _LoadAttempt()
        self._attempt = attempt
        self._state = "loading"
        attempt.loaded.add_done_callback(self._settle)
        self._worker.submit(_load_job, directory, attempt)
        return attempt

    def _settle(self, loaded: Future) -> None:
        # A new attempt starts only once this one has settled, so the
        # attempt settling here is always the current one.
        with self._lock:
            exc = loaded.exception()
            if exc is None:
                self._model = loaded.result()
                self._state = "ready"
            elif isinstance(exc, _DigestMismatch):
                self._attempt = None
                self._mismatch = exc.file
                self._state = "not_loaded"
            else:
                self._failures += 1
                self._failure_reported = False
                self._state = "failed"

    def start_if_idle(self, directory: Path) -> None:
        """Start a load unless one has run or is running (preload)."""
        with self._lock:
            if self._state == "not_loaded":
                self._start(directory)

    async def acquire(self, directory: Path, wait_s: float) -> LoadedModel:
        """The loaded model, starting or joining the load as needed.

        The call waits at most ``wait_s`` in all, for the digest checks and the
        load together; a mismatch found within it is reported at once.

        Raises:
            Chronos2Refusal: Digest mismatch, load failure or still loading.
        """
        with self._lock:
            if self._state == "ready":
                assert self._model is not None
                return self._model
            if self._state == "failed":
                if not self._failure_reported or self._failures >= MAX_LOAD_ATTEMPTS:
                    self._failure_reported = True
                    raise Chronos2Refusal(LOAD_FAILED)
                self._state = "not_loaded"
            if self._state == "not_loaded" and self._mismatch is not None:
                file, self._mismatch = self._mismatch, None
                raise Chronos2Refusal(DIGEST_MISMATCH.format(file=file))
            if self._state == "not_loaded":
                self._start(directory)
            attempt = self._attempt
            assert attempt is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        try:
            await _await_future(attempt.verified, wait_s)
        except _WaitTimeout:
            raise Chronos2Refusal(STILL_LOADING) from None
        except _DigestMismatch as exc:
            with self._lock:
                if self._mismatch == exc.file:
                    self._mismatch = None
            raise Chronos2Refusal(DIGEST_MISMATCH.format(file=exc.file)) from None
        try:
            model: LoadedModel = await _await_future(
                attempt.loaded, max(0.0, deadline - loop.time())
            )
        except _WaitTimeout:
            raise Chronos2Refusal(STILL_LOADING) from None
        except Exception:
            with self._lock:
                if self._attempt is attempt:
                    self._failure_reported = True
            raise Chronos2Refusal(LOAD_FAILED) from None
        return model


async def _await_future(future: Future, timeout: Optional[float]) -> Any:
    """Wait for a worker future without blocking the event loop or cancelling it."""
    if not future.done():
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def wake(_: Future) -> None:
            def mark() -> None:
                if not done.done():
                    done.set_result(None)

            try:
                loop.call_soon_threadsafe(mark)
            except RuntimeError:  # the loop is closed: nobody is waiting
                pass

        future.add_done_callback(wake)
        try:
            await asyncio.wait_for(done, timeout)
        except asyncio.TimeoutError:
            raise _WaitTimeout from None
    return future.result()


_WORKER = _Worker()
_SLOT = _LoadSlot(_WORKER)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRun:
    """The model's quantiles for one call, in the declared unit, and their record.

    ``half_band`` is the narrowest half-band of the model's centred quantiles,
    before the window's last value is added back: the model's own band width,
    which a far-away level cannot round away.
    """

    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    half_band: float
    context_bins: int
    torch_threads: int
    versions: dict[str, str]

    def provenance(self) -> dict[str, Any]:
        """The result's ``provenance.model`` block: the digests verified at load."""
        verified = {name: sha256 for name, _, sha256 in MODEL_FILES}
        return {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "weights_sha256": verified[WEIGHTS_FILE],
            "config_sha256": verified[CONFIG_FILE],
            "weights_licence": WEIGHTS_LICENCE,
            "chronos_forecasting_version": self.versions["chronos-forecasting"],
            "torch_version": self.versions["torch"],
            "transformers_version": self.versions["transformers"],
            "accelerate_version": self.versions["accelerate"],
            "huggingface_hub_version": self.versions["huggingface-hub"],
            "torch_threads": self.torch_threads,
            "context_bins": self.context_bins,
        }


def centre(window: np.ndarray) -> tuple[np.ndarray, float]:
    """The model input: the window minus its last observed value, as float32.

    The subtraction is done in float64 and empty bins stay NaN, so a large
    offset does not cost the float32 input its resolution.
    """
    observed = window[~np.isnan(window)]
    offset = float(observed[-1])
    return (window - offset).astype(np.float32), offset


async def project(directory: Path, window: np.ndarray, horizon: int) -> ModelRun:
    """Run the model on ``window`` (the most recent grid bins, NaN when empty).

    Waits for the model as ``PMM_FORECAST_MODEL_LOAD_WAIT_S`` allows, then runs
    one inference in the worker with ``PMM_FORECAST_MODEL_THREADS`` set there.

    Raises:
        Chronos2Refusal: Digest mismatch, load failure or still loading.
        RuntimeError: When the model returns non-finite quantiles.
    """
    model = await _SLOT.acquire(directory, load_wait_seconds())
    context, offset = centre(window)
    job = _WORKER.submit(_predict_job, model, context, horizon, model_threads())
    try:
        raw, threads = await _await_future(job, None)
    except asyncio.CancelledError:
        job.cancel()  # an abandoned call's queued inference never runs
        raise
    centred = np.sort(raw, axis=1)
    if centred.shape != (horizon, len(QUANTILE_LEVELS)) or not np.all(
        np.isfinite(centred)
    ):
        raise RuntimeError(
            f"chronos2 returned unusable quantiles (shape {centred.shape})"
        )
    half_band = min(
        float(np.min(centred[:, 2] - centred[:, 1])),
        float(np.min(centred[:, 1] - centred[:, 0])),
    )
    return ModelRun(
        p10=centred[:, 0] + offset,
        p50=centred[:, 1] + offset,
        p90=centred[:, 2] + offset,
        half_band=half_band,
        context_bins=int(window.size),
        torch_threads=int(threads),
        versions=dict(model.versions),
    )


def preload_if_requested() -> None:
    """Start the load in the worker at server start when the operator asked for it.

    Never blocks and never raises: a refusal is logged and given to the first
    chronos2 call instead.
    """
    if not preload_requested():
        return
    try:
        directory = check_available()
    except Chronos2Refusal as exc:
        logger.warning("chronos2 preload not started: %s", exc)
        return
    _SLOT.start_if_idle(directory)
