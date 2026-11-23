"""A stand-in for the Chronos-2 libraries, for tests that run without the extra.

``install`` plants two fake modules, ``chronos`` (its ``Chronos2Pipeline``)
and ``torch`` (its thread setting), and answers the installed-version lookup.
Everything of the adapter runs for real against them: the availability
checks, the digest checks (against two small files written here), the worker
thread, the load state, the import-and-load call with its arguments, the
thread setting, the centring and every payload rule.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from predictive_maintenance_mcp.prognostics import chronos2_backend as cb

#: Installed versions inside the supported ranges.
IN_RANGE = {
    "chronos-forecasting": "2.3.2",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
    "accelerate": "1.16.0",
    "huggingface-hub": "1.27.0",
}

_names = itertools.count()


class StubPipeline:
    """Returns centred quantiles: P50 = slope * step, a band of
    +/- spread * sqrt(step), as float32 like the real library, optionally in a
    scrambled column order. Records every call."""

    def __init__(self, spread: float = 0.5, slope: float = 0.0, scramble: bool = False):
        self.spread = spread
        self.slope = slope
        self.scramble = scramble
        self.calls: list[dict[str, Any]] = []

    def predict_quantiles(
        self, inputs, prediction_length, quantile_levels, batch_size, cross_learning
    ):
        (item,) = inputs
        x = item["target"]
        self.calls.append(
            {
                "dtype": x.dtype,
                "x": np.array(x, copy=True),
                "horizon": prediction_length,
                "levels": list(quantile_levels),
                "batch_size": batch_size,
                "cross_learning": cross_learning,
                "thread": threading.current_thread().name,
            }
        )
        k = np.arange(1, prediction_length + 1, dtype=float)
        half = self.spread * np.sqrt(k)
        mid = self.slope * k
        q = np.stack([mid - half, mid, mid + half], axis=1)
        if self.scramble:
            q = q[:, [2, 0, 1]]
        return [q[np.newaxis].astype(np.float32)], [mid[np.newaxis].astype(np.float32)]


class ThreadSetting:
    """PyTorch's thread count as it behaves: per thread, starting at DEFAULT
    on every thread, so a setting made on one thread does not reach another."""

    DEFAULT = 16

    def __init__(self) -> None:
        self._local = threading.local()
        self.calls: list[tuple[str, int]] = []

    def set(self, threads: int) -> None:
        self._local.value = threads
        self.calls.append((threading.current_thread().name, threads))

    def read(self) -> int:
        return getattr(self._local, "value", self.DEFAULT)


@dataclass
class Load:
    thread: str
    path: Any
    args: tuple
    kwargs: dict
    environ: dict


@dataclass
class Stub:
    directory: Path
    pipeline: StubPipeline
    threads: ThreadSetting
    worker_name: str
    loads: list[Load] = field(default_factory=list)
    gate: Optional[threading.Event] = None
    errors: list[BaseException] = field(default_factory=list)
    delay: float = 0.0


def write_model_files(directory: Path) -> tuple[tuple[str, int, str], ...]:
    """Two small stand-in model files and their (name, size, sha256) entries."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, content in (
        (cb.CONFIG_FILE, b'{"architectures": ["Chronos2Model"]}\n'),
        (cb.WEIGHTS_FILE, b"stand-in weights " * 64),
    ):
        (directory / name).write_bytes(content)
        entries.append((name, len(content), hashlib.sha256(content).hexdigest()))
    return tuple(entries)


def install(monkeypatch, tmp_path: Path, versions: Optional[dict] = None, **pipeline):
    """Make the chronos2 backend available on fake libraries; return the record."""
    directory = tmp_path / "chronos-2"
    monkeypatch.setattr(cb, "MODEL_FILES", write_model_files(directory))
    monkeypatch.setenv(cb.ENV_MODEL_DIR, str(directory))
    for name in (cb.ENV_THREADS, cb.ENV_PRELOAD, cb.ENV_LOAD_WAIT):
        monkeypatch.delenv(name, raising=False)
    set_versions(monkeypatch, versions or IN_RANGE)

    worker_name = f"pmm-chronos2-test-{next(_names)}"
    worker = cb._Worker(worker_name)
    monkeypatch.setattr(cb, "_WORKER", worker)
    monkeypatch.setattr(cb, "_SLOT", cb._LoadSlot(worker))

    stub = Stub(directory, StubPipeline(**pipeline), ThreadSetting(), worker_name)

    class Chronos2Pipeline:
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            stub.loads.append(
                Load(
                    threading.current_thread().name,
                    path,
                    args,
                    kwargs,
                    dict(os.environ),
                )
            )
            if stub.gate is not None:
                assert stub.gate.wait(30), "the test never opened the load gate"
            if stub.delay:
                threading.Event().wait(stub.delay)
            if stub.errors:
                raise stub.errors.pop(0)
            return stub.pipeline

    chronos = types.ModuleType("chronos")
    chronos.Chronos2Pipeline = Chronos2Pipeline  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.set_num_threads = stub.threads.set  # type: ignore[attr-defined]
    torch.get_num_threads = stub.threads.read  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chronos", chronos)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return stub


def drain_worker(timeout: float = 30.0) -> None:
    """Wait until every job the current adapter worker holds has finished.

    Call it at the end of a test, before its fake modules are restored: a
    load still queued would otherwise import the next test's fake modules.
    """
    worker = cb._WORKER
    if worker.thread is not None:
        worker.submit(lambda: None).result(timeout)


def set_versions(monkeypatch, versions: dict) -> None:
    """Answer importlib.metadata.version from ``versions`` for the recorded
    distributions; a name missing from it is not installed."""
    real = importlib.metadata.version

    def version(name: str) -> str:
        if name in cb.RECORDED_DISTRIBUTIONS:
            if name not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
