"""Gated: the chronos2 backend with the real model files (opt-in).

Runs only with ``PMM_TEST_CHRONOS2_WEIGHTS=1``, the ``forecast-chronos2``
extra installed and ``PMM_FORECAST_MODEL_DIR`` naming a local copy of
amazon/chronos-2 at the pinned revision; otherwise every test here is skipped
with that reason. Nothing is downloaded: the operator's directory is read.

- Numerical equivalence: the synthetic request set regenerated from seed
  20261117 goes through the adapter at 4 threads and is compared with the
  recorded reference (``tests/fixtures/chronos2_equivalence_reference.json``)
  within 1e-3 of each request's narrowest half-band, never bit for bit.
- Centring with the real model: offsets 1e5, 1e6 and 1e8 give the offset-0
  quantiles plus the offset.
- The load opens no network connection and leaves every Hugging Face and
  transformers setting of the process, and ``huggingface_hub``'s offline
  mode, as they were. Variables the libraries set on their own import are
  reported.
- The thread count read inside the worker is the configured one, after a
  different value was set on the calling thread.
- One end-to-end tool call.

Rerun it whenever the resolved versions of torch, transformers,
huggingface-hub, accelerate or chronos-forecasting change (operator guide,
section 8).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import re
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics import chronos2_backend as cb

FIXTURE = Path(__file__).parent / "fixtures" / "chronos2_equivalence_reference.json"
TOLERANCE = 1e-3  # of the narrowest half-band
OFFSETS = (1e5, 1e6, 1e8)

_ENABLED = os.environ.get("PMM_TEST_CHRONOS2_WEIGHTS") == "1"
_REASON = (
    "chronos2 real-weight tests are opt-in: set PMM_TEST_CHRONOS2_WEIGHTS=1, install "
    "the forecast-chronos2 extra and set PMM_FORECAST_MODEL_DIR"
)

pytestmark = [
    pytest.mark.chronos2_weights,
    pytest.mark.skipif(
        not _ENABLED or importlib.util.find_spec("chronos") is None, reason=_REASON
    ),
]


def synthetic_requests():
    """The P1 synthetic set, regenerated from its seed exactly as recorded."""
    rng = np.random.default_rng(20261117)
    reqs = []

    def fam(name, n):
        t = np.arange(n, dtype=float)
        if name == "rw":
            return 50 + np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 0.1, n)
        if name == "level":
            return 2.0 + rng.normal(0, 0.1, n)
        if name == "daily":
            return (
                40 + 5 * np.sin(2 * np.pi * t / 144) + 0.001 * t + rng.normal(0, 0.3, n)
            )
        if name == "step":
            x = 10 + rng.normal(0, 0.2, n)
            x[int(0.7 * n) :] += 2.0
            return x
        if name == "rising":
            return (
                0.05 * np.exp(t / max(n, 1) * 1.5)
                + 0.02
                + np.abs(rng.normal(0, 0.003, n))
            )
        raise ValueError(name)

    for name in ("rw", "level", "daily", "step", "rising"):
        for n, hs in ((64, (30,)), (512, (30, 144)), (2048, (30, 144))):
            x = fam(name, n)
            for h in hs:
                reqs.append((f"syn-{name}-n{n}-h{h}", x.copy(), h))
    for name in ("rw", "daily"):
        x = fam(name, 512)
        x[-51:] = np.nan  # last 10 % of the context missing
        for h in (30, 144):
            reqs.append((f"syn-{name}-n512-gap10-h{h}", x.copy(), h))
    base = np.cumsum(rng.normal(0, 0.01, 512))
    for off in (0.0, 1e5, 1e6):
        for h in (30, 144):
            reqs.append((f"syn-offset{int(off)}-n512-h{h}", base + off, h))
    return {rid: (x, h) for rid, x, h in reqs}


REQUESTS = synthetic_requests()
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _half_band(q: np.ndarray) -> float:
    return min(float(np.min(q[:, 2] - q[:, 1])), float(np.min(q[:, 1] - q[:, 0])))


def _quantiles(run) -> np.ndarray:
    return np.stack([run.p10, run.p50, run.p90], axis=1)


@pytest.fixture(scope="module")
def loaded():
    """Load the real model once, with the network blocked, and record what
    the load changed. Waits for the load however long it takes."""
    directory = cb.check_available()
    attempts: list[object] = []
    local = ("127.0.0.1", "::1", "localhost")
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] in local:
            return real_connect(sock, address)
        attempts.append(address)
        raise OSError("network blocked during the chronos2 load")

    def getaddrinfo(host, *args, **kwargs):
        if host in local:
            return real_getaddrinfo(host, *args, **kwargs)
        attempts.append(host)
        raise OSError("network blocked during the chronos2 load")

    environ = dict(os.environ)
    offline_before = os.environ.get("HF_HUB_OFFLINE")
    socket.socket.connect, socket.getaddrinfo = connect, getaddrinfo  # type: ignore[method-assign]
    start = time.monotonic()
    try:
        cb._SLOT.start_if_idle(directory)
        while cb._SLOT.state == "loading":
            time.sleep(0.05)
    finally:
        socket.socket.connect, socket.getaddrinfo = real_connect, real_getaddrinfo  # type: ignore[method-assign]
    hub = sys.modules.get("huggingface_hub.constants")
    after = dict(os.environ)
    changed = {
        key: (environ.get(key), after.get(key))
        for key in environ.keys() | after.keys()
        if environ.get(key) != after.get(key)
    }
    return {
        "directory": directory,
        "state": cb._SLOT.state,
        "seconds": time.monotonic() - start,
        "attempts": attempts,
        "changed": changed,
        "offline_before": offline_before,
        "hub_offline": None if hub is None else hub.HF_HUB_OFFLINE,
    }


#: The settings that decide whether any tool of the process can reach the
#: Hugging Face Hub. The load must leave every one of them as it was.
_HUB_SETTINGS = re.compile(r"^(HF_|HUGGINGFACE|TRANSFORMERS_|HF_HUB_)")


@pytest.fixture
def four_threads(monkeypatch):
    monkeypatch.setenv(cb.ENV_THREADS, "4")
    monkeypatch.setenv(cb.ENV_LOAD_WAIT, "120")


def test_the_model_loaded_offline_without_changing_the_process(loaded):
    assert loaded["state"] == "ready", loaded
    assert loaded["attempts"] == []
    hub_changes = {k: v for k, v in loaded["changed"].items() if _HUB_SETTINGS.match(k)}
    assert hub_changes == {}
    if loaded["offline_before"] is None:
        assert loaded["hub_offline"] is False
    # Other variables the libraries set on their own import (scikit-learn's
    # OpenMP settings, torch's inductor cache) are reported, not refused.
    print(
        f"\nchronos2 load: {loaded['seconds']:.1f} s; other changes: {loaded['changed']}"
    )


def test_the_reference_was_recorded_with_the_pinned_files():
    assert REFERENCE["revision"] == cb.REVISION
    assert REFERENCE["weights_sha256"] == cb.MODEL_FILES[1][2]
    assert REFERENCE["threads"] == 4
    assert sorted(REFERENCE["requests"]) == sorted(
        rid
        for rid in REQUESTS
        if not rid.startswith("syn-offset") or rid.startswith("syn-offset0-")
    )


@pytest.mark.parametrize("rid", sorted(REFERENCE["requests"]))
def test_equivalence_with_the_recorded_reference(loaded, four_threads, rid):
    window, horizon = REQUESTS[rid]
    run = asyncio.run(cb.project(loaded["directory"], window, horizon))
    reference = np.sort(np.asarray(REFERENCE["requests"][rid], dtype=float), axis=1)
    worst = float(np.max(np.abs(_quantiles(run) - reference)))
    assert run.torch_threads == 4
    assert worst <= TOLERANCE * _half_band(reference), (rid, worst)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("horizon", (30, 144))
def test_centring_keeps_large_offsets_equivalent(loaded, four_threads, offset, horizon):
    base, _ = REQUESTS[f"syn-offset0-n512-h{horizon}"]
    reference = np.sort(
        np.asarray(REFERENCE["requests"][f"syn-offset0-n512-h{horizon}"]), axis=1
    )
    run = asyncio.run(cb.project(loaded["directory"], base + offset, horizon))
    worst = float(np.max(np.abs(_quantiles(run) - offset - reference)))
    assert worst <= TOLERANCE * _half_band(reference), (offset, worst)


def test_the_thread_count_is_applied_inside_the_worker(loaded, monkeypatch):
    import torch

    window, horizon = REQUESTS["syn-rw-n512-h30"]
    for configured, elsewhere in ((4, 2), (2, 4)):
        torch.set_num_threads(elsewhere)  # on this thread: must not reach the worker
        monkeypatch.setenv(cb.ENV_THREADS, str(configured))
        run = asyncio.run(cb.project(loaded["directory"], window, horizon))
        assert run.torch_threads == configured


def test_a_tool_call_end_to_end(loaded, four_threads):
    from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
        forecast_asset_indicator,
    )

    window, _ = REQUESTS["syn-daily-n512-h144"]
    end = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=10)
    obs = [
        (end - (len(window) - 1 - i) * timedelta(minutes=10)).isoformat()
        for i in range(len(window))
    ]
    start = time.monotonic()
    result = asyncio.run(
        forecast_asset_indicator(
            None,
            obs,
            [float(v) for v in window],
            "temperature",
            "degC",
            "10min",
            144,
            backend="chronos2",
        )
    )
    elapsed = time.monotonic() - start
    assert result.status == "forecast" and result.backend == "chronos2"
    assert result.reference is not None and result.reference.backend == "naive"
    model = result.provenance.model
    assert model.torch_threads == 4 and model.context_bins == 512
    assert model.torch_version == importlib.metadata.version("torch")
    assert model.chronos_forecasting_version == "2.3.2"
    print(f"\nchronos2 warm tool call (512 bins, 144 steps): {elapsed * 1000:.0f} ms")
