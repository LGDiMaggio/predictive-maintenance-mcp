"""The chronos2 adapter (``prognostics.chronos2_backend``) without the extra.

Every test runs the adapter's real code against fake ``chronos`` and
``torch`` modules (``_chronos2_stub``):

- availability in the handoff's order, each refusal with its exact text;
- integrity: both model files checked by size and SHA-256 before any load;
- the load state: single flight, the bounded wait and its "still loading"
  refusal, one retry after a failed load, the opt-in preload;
- the worker: loading and inference never run on the event loop, and the
  thread count is set and read inside the worker;
- float64 centring before the float32 cast;
- the process environment is left as it was, and nothing reaches the network;
- the default path imports neither torch nor chronos (subprocess probe).
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import _chronos2_texts as T
from _chronos2_stub import IN_RANGE, drain_worker, install, set_versions
from predictive_maintenance_mcp.prognostics import chronos2_backend as cb
from predictive_maintenance_mcp.prognostics import indicator_forecast as f
from predictive_maintenance_mcp.signal_validation import InvalidInputError

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)


def _series(n=200, level=10.0, seed=1):
    rng = np.random.default_rng(seed)
    end = NOW - timedelta(hours=1)
    instants = [
        (end - (n - 1 - i) * timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]
    values = [float(v) for v in level + np.cumsum(rng.normal(0, 0.1, n))]
    return instants, values


def _chronos2(obs, vals, horizon_steps=24, **kw):
    return asyncio.run(
        f.forecast_indicator_chronos2(
            observed_at=obs,
            values=vals,
            indicator="velocity RMS",
            unit="mm/s",
            resolution="1h",
            horizon_steps=horizon_steps,
            now=NOW,
            server_version="test",
            **kw,
        )
    )


def _window(n=128, level=5.0, seed=2):
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.normal(0, 0.01, n))


def _project(window, horizon=12):
    return asyncio.run(cb.project(cb.check_available(), window, horizon))


def _refusal(fn, *args, **kwargs) -> str:
    with pytest.raises(InvalidInputError) as exc:
        fn(*args, **kwargs)
    return str(exc.value)


def _load_failure_causes(caplog) -> list[str]:
    """The causes the adapter logged with 'chronos2 model load failed', from
    this test's own worker only (an earlier test's worker may still log)."""
    worker = cb._WORKER.thread.name if cb._WORKER.thread else None
    return [
        str(r.exc_info[1])
        for r in caplog.records
        if r.name == cb.__name__
        and r.getMessage() == "chronos2 model load failed"
        and r.threadName == worker
    ]


def _wait_for_state(state, timeout=10.0):
    end = time.monotonic() + timeout
    while cb._SLOT.state != state:
        assert time.monotonic() < end, f"slot stayed {cb._SLOT.state}"
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _drain_the_worker(monkeypatch):
    # Set up after monkeypatch, so it runs before the fakes are restored.
    yield
    drain_worker()


@pytest.fixture
def stub(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# Availability, in the handoff's order (steps 1 to 4)
# ---------------------------------------------------------------------------


class TestAvailability:
    @pytest.mark.parametrize("missing", cb.RECORDED_DISTRIBUTIONS)
    def test_a_missing_distribution_means_the_extra_is_not_installed(
        self, stub, monkeypatch, missing
    ):
        set_versions(monkeypatch, {k: v for k, v in IN_RANGE.items() if k != missing})
        assert _refusal(cb.check_available) == T.EXTRA_NOT_INSTALLED

    @pytest.mark.parametrize(
        "package, version, spec",
        [
            ("torch", "2.15.0", ">=2.13,<2.15"),
            ("torch", "2.12.1+cpu", ">=2.13,<2.15"),
            ("torch", "2.15.0rc1", ">=2.13,<2.15"),
            ("torch", "not-a-version", ">=2.13,<2.15"),
            ("transformers", "5.21.0", ">=5.14,<5.21"),
            ("transformers", "5.13.2", ">=5.14,<5.21"),
            ("chronos-forecasting", "2.3.1", "==2.3.2"),
            ("chronos-forecasting", "2.4.0", "==2.3.2"),
        ],
    )
    def test_a_version_outside_its_range_is_refused(
        self, stub, monkeypatch, package, version, spec
    ):
        set_versions(monkeypatch, {**IN_RANGE, package: version})
        assert _refusal(cb.check_available) == T.VERSION_OUTSIDE_RANGE.format(
            package=package, version=version, range=spec
        )

    @pytest.mark.parametrize(
        "versions",
        [
            {"torch": "2.14.1+cpu", "transformers": "5.20.2"},
            {"torch": "2.13.0", "transformers": "5.14.0"},
            {"chronos-forecasting": "2.3.2+local"},
        ],
    )
    def test_versions_inside_the_ranges_pass(self, stub, monkeypatch, versions):
        set_versions(monkeypatch, {**IN_RANGE, **versions})
        assert cb.check_available() == stub.directory.resolve()

    def test_the_ranges_equal_the_extra_in_pyproject(self):
        import tomllib

        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        extra = project["project"]["optional-dependencies"]["forecast-chronos2"]
        assert sorted(extra) == sorted(f"{p}{s}" for p, s in cb.SUPPORTED_RANGES)
        assert not any(
            "forecast-chronos2" in item
            for item in project["project"]["optional-dependencies"]["full"]
        )

    def test_the_checks_import_no_model_library(self, stub, monkeypatch):
        imported: list[str] = []
        real = builtins.__import__

        def spy(name, *args, **kwargs):
            imported.append(name.split(".")[0])
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", spy)
        cb.check_available()
        assert not {"torch", "transformers", "chronos"} & set(imported)

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_an_unset_or_blank_directory_is_refused(self, stub, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(cb.ENV_MODEL_DIR)
        else:
            monkeypatch.setenv(cb.ENV_MODEL_DIR, value)
        assert _refusal(cb.check_available) == T.DIR_NOT_SET

    @pytest.mark.parametrize(
        "make",
        ["missing", "a_file", "no_weights", "no_config", "repository_id"],
    )
    def test_not_a_directory_holding_both_files_is_refused(
        self, stub, monkeypatch, make
    ):
        if make == "missing":
            value = str(stub.directory / "absent")
        elif make == "a_file":
            value = str(stub.directory / cb.CONFIG_FILE)
        elif make == "repository_id":
            value = "amazon/chronos-2"
        else:
            name = cb.WEIGHTS_FILE if make == "no_weights" else cb.CONFIG_FILE
            (stub.directory / name).unlink()
            value = str(stub.directory)
        monkeypatch.setenv(cb.ENV_MODEL_DIR, value)
        assert _refusal(cb.check_available) == T.NOT_A_MODEL_DIR.format(path=value)

    def test_the_directory_is_returned_absolute(self, stub, monkeypatch):
        monkeypatch.chdir(stub.directory.parent)
        monkeypatch.setenv(cb.ENV_MODEL_DIR, stub.directory.name)
        assert cb.check_available() == stub.directory.resolve()


class TestNoNetwork:
    def test_a_repository_id_never_reaches_a_loader_or_the_network(
        self, stub, monkeypatch
    ):
        attempts: list[object] = []
        local = ("127.0.0.1", "::1", "localhost")
        real_connect = socket.socket.connect
        real_getaddrinfo = socket.getaddrinfo

        # The event loop's own self-pipe connects over loopback on Windows;
        # anything else is recorded and refused.
        def connect(sock, address):
            if isinstance(address, tuple) and address[0] in local:
                return real_connect(sock, address)
            attempts.append(address)
            raise OSError("network blocked in this test")

        def getaddrinfo(host, *args, **kwargs):
            if host in local:
                return real_getaddrinfo(host, *args, **kwargs)
            attempts.append(host)
            raise OSError("network blocked in this test")

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setenv(cb.ENV_MODEL_DIR, "amazon/chronos-2")
        obs, vals = _series()
        assert _refusal(_chronos2, obs, vals) == T.NOT_A_MODEL_DIR.format(
            path="amazon/chronos-2"
        )
        assert stub.loads == [] and attempts == []

    def test_the_loader_gets_the_verified_local_directory_only(self, stub):
        _project(_window())
        (load,) = stub.loads
        assert load.path == str(stub.directory.resolve())
        assert load.args == ()
        assert load.kwargs == {
            "device_map": "cpu",
            "local_files_only": True,
            "use_safetensors": True,
        }

    def test_the_process_environment_is_left_as_it_was(self, stub):
        before = dict(os.environ)
        _project(_window())
        (load,) = stub.loads
        assert load.environ == before  # as the loader saw it
        assert dict(os.environ) == before  # and after


# ---------------------------------------------------------------------------
# Integrity (steps 5 and 6)
# ---------------------------------------------------------------------------


class TestIntegrity:
    @pytest.mark.parametrize("name", [cb.CONFIG_FILE, cb.WEIGHTS_FILE])
    def test_a_changed_file_is_refused_before_any_load(self, stub, name):
        path = stub.directory / name
        data = bytearray(path.read_bytes())
        data[0] ^= 1  # same size, other content
        path.write_bytes(bytes(data))
        assert _refusal(_project, _window()) == T.DIGEST_MISMATCH.format(file=name)
        assert stub.loads == []
        assert cb._SLOT.state == "not_loaded"

    def test_a_file_of_another_size_is_refused(self, stub):
        (stub.directory / cb.WEIGHTS_FILE).write_bytes(b"short")
        assert _refusal(_project, _window()) == T.DIGEST_MISMATCH.format(
            file=cb.WEIGHTS_FILE
        )

    def test_the_config_is_named_first_when_both_differ(self, stub):
        for name in (cb.CONFIG_FILE, cb.WEIGHTS_FILE):
            (stub.directory / name).write_bytes(b"x")
        assert _refusal(_project, _window()) == T.DIGEST_MISMATCH.format(
            file=cb.CONFIG_FILE
        )

    def test_a_corrected_directory_is_verified_again_and_loads(self, stub):
        path = stub.directory / cb.WEIGHTS_FILE
        good = path.read_bytes()
        path.write_bytes(b"x" * len(good))
        _refusal(_project, _window())
        path.write_bytes(good)
        assert _project(_window()).p50.size == 12
        assert len(stub.loads) == 1

    def test_an_adapter_file_in_the_directory_is_never_loaded(self, stub, caplog):
        # An adapter_config.json makes the loader read files nobody verified.
        (stub.directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger=cb.__name__):
            assert _refusal(_project, _window()) == T.LOAD_FAILED
        assert stub.loads == []
        assert _load_failure_causes(caplog) == [
            f"adapter_config.json in {stub.directory.resolve()} would make the loader "
            f"read files that are not verified; remove it to load the pinned model "
            f"files"
        ]

    def test_a_slow_digest_check_counts_against_the_wait(self, stub, monkeypatch):
        real = cb._sha256

        def slow(path):
            time.sleep(1.0)
            return real(path)

        monkeypatch.setattr(cb, "_sha256", slow)
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "0.2")
        start = time.monotonic()
        assert _refusal(_project, _window()) == T.STILL_LOADING
        assert time.monotonic() - start < 0.9
        _wait_for_state("ready")
        assert _project(_window()).p50.size == 12
        assert len(stub.loads) == 1

    def test_a_mismatch_found_after_the_wait_is_reported_once(self, stub, monkeypatch):
        path = stub.directory / cb.CONFIG_FILE
        data = bytearray(path.read_bytes())
        data[0] ^= 1  # same size, so the file is hashed
        path.write_bytes(bytes(data))
        hashed: list[object] = []
        real = cb._sha256

        def slow(path):
            hashed.append(path)
            time.sleep(0.5)
            return real(path)

        monkeypatch.setattr(cb, "_sha256", slow)
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "0.1")
        assert _refusal(_project, _window()) == T.STILL_LOADING
        end = time.monotonic() + 10
        while not hashed or cb._SLOT.state == "loading":
            assert time.monotonic() < end
            time.sleep(0.01)
        # Reported by the next call without hashing again ...
        assert _refusal(_project, _window()) == T.DIGEST_MISMATCH.format(
            file=cb.CONFIG_FILE
        )
        assert len(hashed) == 1
        # ... and only once: the call after it verifies the files again.
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "10")
        assert _refusal(_project, _window()) == T.DIGEST_MISMATCH.format(
            file=cb.CONFIG_FILE
        )
        assert len(hashed) == 2 and stub.loads == []

    def test_a_preload_mismatch_is_logged(self, stub, monkeypatch, caplog):
        (stub.directory / cb.WEIGHTS_FILE).write_bytes(b"x")
        monkeypatch.setenv(cb.ENV_PRELOAD, "1")
        with caplog.at_level(logging.WARNING, logger=cb.__name__):
            cb.preload_if_requested()
            end = time.monotonic() + 10
            while cb._SLOT.state == "loading":
                assert time.monotonic() < end
                time.sleep(0.01)
        assert (
            "chronos2 model file model.safetensors does not match the pinned "
            "SHA-256; not loaded"
        ) in caplog.messages

    def test_an_unreadable_model_file_is_a_load_failure(
        self, stub, monkeypatch, caplog
    ):
        def unreadable(path):
            raise PermissionError(f"cannot read {path}")

        monkeypatch.setattr(cb, "_sha256", unreadable)
        with caplog.at_level(logging.ERROR, logger=cb.__name__):
            assert _refusal(_project, _window()) == T.LOAD_FAILED
        assert "chronos2 model files could not be read" in caplog.messages
        assert stub.loads == []

    def test_a_path_the_file_system_cannot_check_is_refused(self, stub, monkeypatch):
        def broken(self):
            raise OSError("the path cannot be checked")

        monkeypatch.setattr(Path, "is_dir", broken)
        assert _refusal(cb.check_available) == T.NOT_A_MODEL_DIR.format(
            path=str(stub.directory)
        )

    def test_the_published_digests_are_those_of_the_pinned_revision(self):
        assert cb.REVISION == "29ec3766d36d6f73f0696f85560a422f50e8498c"
        assert cb.MODEL_FILES == (
            (
                "config.json",
                1_067,
                "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
            ),
            (
                "model.safetensors",
                477_930_472,
                "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
            ),
        )


# ---------------------------------------------------------------------------
# The load state: single flight, bounded wait, one retry, preload
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_the_first_call_loads_and_projects_in_the_worker(self, stub):
        run = _project(_window())
        assert [load.thread for load in stub.loads] == [stub.worker_name]
        assert [call["thread"] for call in stub.pipeline.calls] == [stub.worker_name]
        assert cb._SLOT.state == "ready"
        assert run.versions == IN_RANGE

    def test_concurrent_calls_share_one_load(self, stub):
        stub.gate = threading.Event()

        async def three():
            directory = cb.check_available()
            calls = [cb.project(directory, _window(), 6) for _ in range(3)]
            asyncio.get_running_loop().call_later(0.3, stub.gate.set)
            return await asyncio.gather(*calls)

        runs = asyncio.run(three())
        assert len(runs) == 3 and len(stub.loads) == 1

    def test_a_load_longer_than_the_wait_is_refused_and_goes_on(
        self, stub, monkeypatch
    ):
        stub.gate = threading.Event()
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "0.2")
        assert _refusal(_project, _window()) == T.STILL_LOADING
        assert cb._SLOT.state == "loading"
        stub.gate.set()
        _wait_for_state("ready")
        assert _project(_window()).p50.size == 12
        assert len(stub.loads) == 1

    def test_a_load_ending_after_a_refusal_serves_the_same_loop(
        self, stub, monkeypatch
    ):
        stub.gate = threading.Event()

        async def both():
            directory = cb.check_available()
            monkeypatch.setenv(cb.ENV_LOAD_WAIT, "0.1")
            with pytest.raises(cb.Chronos2Refusal) as first:
                await cb.project(directory, _window(), 6)
            stub.gate.set()  # the load finishes while this loop still runs
            monkeypatch.setenv(cb.ENV_LOAD_WAIT, "10")
            return str(first.value), await cb.project(directory, _window(), 6)

        refusal, run = asyncio.run(both())
        assert refusal == T.STILL_LOADING
        assert run.p50.size == 6 and len(stub.loads) == 1

    def test_a_wait_of_zero_refuses_at_once(self, stub, monkeypatch):
        stub.gate = threading.Event()
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "0")
        start = time.monotonic()
        assert _refusal(_project, _window()) == T.STILL_LOADING
        assert time.monotonic() - start < 1.0
        stub.gate.set()

    def test_a_failed_load_is_reported_then_retried_once(self, stub, caplog):
        stub.errors.append(RuntimeError("weights could not be mapped"))
        with caplog.at_level(logging.ERROR, logger=cb.__name__):
            assert _refusal(_project, _window()) == T.LOAD_FAILED
        assert _load_failure_causes(caplog) == ["weights could not be mapped"]
        assert _project(_window()).p50.size == 12
        assert len(stub.loads) == 2

    def test_a_second_failure_stays_until_the_server_restarts(self, stub):
        stub.errors += [RuntimeError("first"), RuntimeError("second")]
        for _ in range(4):
            assert _refusal(_project, _window()) == T.LOAD_FAILED
        assert len(stub.loads) == cb.MAX_LOAD_ATTEMPTS == 2

    def test_a_preload_failure_is_reported_by_the_first_call(self, stub, monkeypatch):
        stub.errors.append(RuntimeError("preload"))
        monkeypatch.setenv(cb.ENV_PRELOAD, "1")
        cb.preload_if_requested()
        _wait_for_state("failed")
        assert _refusal(_project, _window()) == T.LOAD_FAILED
        assert len(stub.loads) == 1  # reported, not retried yet
        assert _project(_window()).p50.size == 12
        assert len(stub.loads) == 2

    def test_preload_starts_the_load_without_blocking(self, stub, monkeypatch):
        stub.gate = threading.Event()
        monkeypatch.setenv(cb.ENV_PRELOAD, "1")
        start = time.monotonic()
        cb.preload_if_requested()
        assert time.monotonic() - start < 0.5
        assert cb._SLOT.state == "loading"
        stub.gate.set()
        _wait_for_state("ready")
        _project(_window())
        assert len(stub.loads) == 1

    def test_without_the_setting_nothing_is_preloaded(self, stub):
        cb.preload_if_requested()
        assert cb._WORKER.thread is None and stub.loads == []

    def test_a_preload_refusal_is_logged_and_starts_nothing(
        self, stub, monkeypatch, caplog
    ):
        monkeypatch.setenv(cb.ENV_PRELOAD, "1")
        monkeypatch.delenv(cb.ENV_MODEL_DIR)
        with caplog.at_level(logging.WARNING, logger=cb.__name__):
            cb.preload_if_requested()
        assert f"chronos2 preload not started: {T.DIR_NOT_SET}" in caplog.messages
        assert cb._WORKER.thread is None

    def test_a_reference_call_does_not_wait_for_the_load(self, stub, monkeypatch):
        stub.gate = threading.Event()
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "10")
        obs, vals = _series()

        async def both():
            loading = asyncio.ensure_future(
                cb.project(cb.check_available(), _window(), 6)
            )
            await asyncio.sleep(0.1)
            start = time.monotonic()
            naive = f.forecast_indicator(
                observed_at=obs,
                values=vals,
                indicator="velocity RMS",
                unit="mm/s",
                resolution="1h",
                horizon_steps=24,
                now=NOW,
                server_version="test",
            )
            elapsed = time.monotonic() - start
            stub.gate.set()
            await loading
            return naive, elapsed

        naive, elapsed = asyncio.run(both())
        assert naive["status"] == "forecast" and elapsed < 1.0


# ---------------------------------------------------------------------------
# The worker and the thread count
# ---------------------------------------------------------------------------


class TestWorker:
    def test_the_event_loop_keeps_running_while_the_model_loads(self, stub):
        stub.delay = 1.0

        async def measure():
            lags: list[float] = []
            done = False

            async def heartbeat():
                while not done:
                    start = time.monotonic()
                    await asyncio.sleep(0.01)
                    lags.append(time.monotonic() - start - 0.01)

            beat = asyncio.ensure_future(heartbeat())
            await asyncio.sleep(0.05)
            await cb.project(cb.check_available(), _window(), 6)
            done = True
            await beat
            return max(lags), len(lags)

        worst, beats = asyncio.run(measure())
        assert beats > 20  # the heartbeat kept running during the 1 s load
        assert worst < 0.25

    def test_the_thread_count_is_set_and_read_inside_the_worker(
        self, stub, monkeypatch
    ):
        stub.threads.set(2)  # on the test's own thread: must not reach the worker
        monkeypatch.setenv(cb.ENV_THREADS, "4")
        run = _project(_window())
        assert run.torch_threads == 4
        assert stub.threads.calls[1:] == [(stub.worker_name, 4)]
        monkeypatch.setenv(cb.ENV_THREADS, "1")
        assert _project(_window()).torch_threads == 1

    def test_the_call_follows_the_documented_inference_settings(self, stub):
        _project(_window(), horizon=7)
        (call,) = stub.pipeline.calls
        assert call["horizon"] == 7
        assert call["levels"] == [0.1, 0.5, 0.9]
        assert (call["batch_size"], call["cross_learning"]) == (1, False)

    def test_quantiles_are_sorted_per_step(self, monkeypatch, tmp_path):
        install(monkeypatch, tmp_path, scramble=True)
        run = _project(_window())
        assert np.all(run.p10 < run.p50) and np.all(run.p50 < run.p90)

    def test_an_inference_error_reaches_the_caller_and_the_worker_goes_on(
        self, stub, monkeypatch
    ):
        real = stub.pipeline.predict_quantiles
        failures = [RuntimeError("inference failed")]

        def predict(*args, **kwargs):
            if failures:
                raise failures.pop()
            return real(*args, **kwargs)

        monkeypatch.setattr(stub.pipeline, "predict_quantiles", predict)
        with pytest.raises(RuntimeError, match="inference failed"):
            _project(_window())
        assert _project(_window()).p50.size == 12

    def test_unusable_model_quantiles_are_never_a_forecast(self, stub, monkeypatch):
        def predict(inputs, prediction_length, **kwargs):
            q = np.full((1, prediction_length, 3), np.nan, dtype=np.float32)
            return [q], [q[..., 1]]

        monkeypatch.setattr(stub.pipeline, "predict_quantiles", predict)
        with pytest.raises(RuntimeError, match="unusable quantiles"):
            _project(_window())

    def test_an_abandoned_call_leaves_no_queued_inference(self, stub):
        _project(_window())  # load the model
        gate = threading.Event()

        async def abandon():
            busy = cb._WORKER.submit(gate.wait, 10)  # the worker is busy
            task = asyncio.ensure_future(cb.project(cb.check_available(), _window(), 6))
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            gate.set()
            busy.result(10)
            cb._WORKER.submit(lambda: None).result(10)  # the queue has drained

        asyncio.run(abandon())
        assert len(stub.pipeline.calls) == 1

    def test_a_cancelled_queued_job_is_skipped(self):
        worker = cb._Worker("pmm-chronos2-test-cancel")
        gate = threading.Event()
        first = worker.submit(gate.wait, 10)
        second = worker.submit(lambda: "second")
        third = worker.submit(lambda: "third")
        assert second.cancel()
        gate.set()
        assert third.result(10) == "third" and first.result(10) is True
        assert second.cancelled()


# ---------------------------------------------------------------------------
# Centring (offsets of the P1 probe)
# ---------------------------------------------------------------------------


class TestCentring:
    @pytest.mark.parametrize("offset", [0.0, 1e5, 1e6, 1e8])
    def test_the_model_sees_the_window_minus_its_last_value_as_float32(
        self, stub, offset
    ):
        window = offset + _window(n=512, level=0.0)
        window[100:110] = np.nan
        run = _project(window)
        (call,) = stub.pipeline.calls
        x = call["x"]
        assert call["dtype"] == np.float32
        assert x[-1] == 0.0
        assert np.array_equal(np.isnan(x), np.isnan(window))
        expected = window - window[-1]
        assert np.nanmax(np.abs(x - expected)) <= 1e-6 * np.nanmax(np.abs(expected))
        # The offset is added back in float64: the band keeps its width.
        assert run.half_band == pytest.approx(0.5, rel=1e-6)
        assert np.min(run.p90 - run.p50) == pytest.approx(0.5, rel=1e-6)
        assert run.p50[0] == window[-1]


# ---------------------------------------------------------------------------
# Settings, read at each call
# ---------------------------------------------------------------------------


class TestSettings:
    @pytest.mark.parametrize(
        "raw, expected",
        [(None, 4), ("8", 8), ("0", 1), ("65", 64), ("abc", 4), ("4.5", 4)],
    )
    def test_threads(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv(cb.ENV_THREADS, raising=False)
        else:
            monkeypatch.setenv(cb.ENV_THREADS, raw)
        assert cb.model_threads() == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, 30.0),
            ("0", 0.0),
            ("2.5", 2.5),
            ("121", 120.0),
            ("-1", 0.0),
            ("nan", 30.0),
            ("x", 30.0),
        ],
    )
    def test_load_wait(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv(cb.ENV_LOAD_WAIT, raising=False)
        else:
            monkeypatch.setenv(cb.ENV_LOAD_WAIT, raw)
        assert cb.load_wait_seconds() == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [("1", True), ("true", True), ("on", True), ("0", False), ("", False)],
    )
    def test_preload(self, monkeypatch, raw, expected):
        monkeypatch.setenv(cb.ENV_PRELOAD, raw)
        assert cb.preload_requested() is expected

    @pytest.mark.parametrize(
        "name, raw, used, reason",
        [
            (cb.ENV_THREADS, "abc", 4, "is not a usable number"),
            (cb.ENV_THREADS, "4.5", 4, "is not a usable number"),
            (cb.ENV_THREADS, "0", 1, "is outside 1 to 64"),
            (cb.ENV_THREADS, "65", 64, "is outside 1 to 64"),
            (cb.ENV_LOAD_WAIT, "x", 30.0, "is not a usable number"),
            (cb.ENV_LOAD_WAIT, "nan", 30.0, "is not a usable number"),
            (cb.ENV_LOAD_WAIT, "inf", 30.0, "is not a usable number"),
            (cb.ENV_LOAD_WAIT, "-1", 0.0, "is outside 0.0 to 120.0"),
            (cb.ENV_LOAD_WAIT, "121", 120.0, "is outside 0.0 to 120.0"),
        ],
    )
    def test_an_unusable_setting_is_logged_with_the_value_used(
        self, monkeypatch, caplog, name, raw, used, reason
    ):
        """An operator sees each unusable setting in the server log: the
        variable, the value given and the value used instead."""
        monkeypatch.setenv(name, raw)
        read = cb.model_threads if name == cb.ENV_THREADS else cb.load_wait_seconds
        with caplog.at_level(logging.WARNING, logger=cb.logger.name):
            assert read() == used
        logged = [r for r in caplog.records if r.name == cb.logger.name]
        assert [(r.levelno, r.getMessage()) for r in logged] == [
            (logging.WARNING, f"{name}={raw!r} {reason}; using {used}")
        ]

    @pytest.mark.parametrize(
        "name, raw",
        [(cb.ENV_THREADS, "8"), (cb.ENV_THREADS, ""), (cb.ENV_LOAD_WAIT, "120")],
    )
    def test_a_usable_or_unset_setting_logs_nothing(
        self, monkeypatch, caplog, name, raw
    ):
        monkeypatch.setenv(name, raw)
        read = cb.model_threads if name == cb.ENV_THREADS else cb.load_wait_seconds
        with caplog.at_level(logging.DEBUG, logger=cb.logger.name):
            read()
        assert [r for r in caplog.records if r.name == cb.logger.name] == []

    def test_unusable_settings_never_refuse_a_call(self, stub, monkeypatch, caplog):
        """The call runs with the values used instead; nothing is refused."""
        monkeypatch.setenv(cb.ENV_THREADS, "many")
        monkeypatch.setenv(cb.ENV_LOAD_WAIT, "999")
        obs, vals = _series()
        with caplog.at_level(logging.WARNING, logger=cb.logger.name):
            r = _chronos2(obs, vals)
        assert r["status"] == "forecast" and r["backend"] == "chronos2"
        assert r["provenance"]["model"]["torch_threads"] == cb.DEFAULT_THREADS
        assert (stub.worker_name, cb.DEFAULT_THREADS) in stub.threads.calls
        messages = {r.getMessage() for r in caplog.records if r.name == cb.logger.name}
        assert f"{cb.ENV_THREADS}='many' is not a usable number; using 4" in messages
        assert (
            f"{cb.ENV_LOAD_WAIT}='999' is outside 0.0 to 120.0; using 120.0" in messages
        )


def test_server_start_starts_the_opt_in_preload_once(monkeypatch):
    """server.main asks for the preload after setting up the environment and
    before the transport runs; the adapter decides whether one starts."""
    from unittest.mock import MagicMock

    import predictive_maintenance_mcp.server as srv

    order: list[str] = []
    monkeypatch.setattr(srv, "_setup_environment", lambda: order.append("setup"))
    monkeypatch.setattr(
        srv.chronos2_backend, "preload_if_requested", lambda: order.append("preload")
    )
    monkeypatch.setattr(
        srv.mcp, "run", MagicMock(side_effect=lambda **_: order.append("run"))
    )
    for name in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["server"])
    srv.main()
    assert order == ["setup", "preload", "run"]


# ---------------------------------------------------------------------------
# The default path stays free of the model libraries (subprocess probe)
# ---------------------------------------------------------------------------

_SENTINEL = """\
import os
with open(os.environ["PMM_SENTINEL_RECORD"], "a", encoding="utf-8") as _f:
    _f.write(__name__ + "\\n")
"""

_PROBE = """
import sys
sys.path.insert(0, {sentinels!r})
import asyncio, json
from datetime import datetime, timedelta, timezone
from predictive_maintenance_mcp.server import mcp
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
    forecast_asset_indicator,
)
end = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=1)
obs = [(end - (199 - i) * timedelta(hours=1)).isoformat() for i in range(200)]
vals = [10 + ((i * 7) % 5) * 0.1 + (i % 24) * 0.05 for i in range(200)]
kinds = []
for backend, season in (("naive", None), ("seasonal_naive", 24)):
    result = asyncio.run(forecast_asset_indicator(
        None, obs, vals, "velocity RMS", "mm/s", "1h", 12,
        backend=backend, season_length_steps=season))
    kinds.append(result.status)
print(json.dumps({{
    "status": kinds,
    "loaded": sorted(m for m in ("torch", "transformers", "chronos") if m in sys.modules),
}}))
"""


def test_the_default_path_imports_no_model_library(tmp_path):
    """Sentinel torch, transformers and chronos packages sit first on the
    path and record any import; the server import and two reference calls
    must import none of them, with or without the real libraries installed."""
    from conftest import SUBPROCESS_PIN

    sentinels = tmp_path / "sentinels"
    for name in ("torch", "transformers", "chronos"):
        (sentinels / name).mkdir(parents=True)
        (sentinels / name / "__init__.py").write_text(_SENTINEL, encoding="utf-8")
    record = tmp_path / "imports.txt"
    record.write_text("", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            SUBPROCESS_PIN + _PROBE.format(sentinels=str(sentinels)),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        env={**os.environ, "PMM_SENTINEL_RECORD": str(record)},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    assert outcome == {"status": ["forecast", "forecast"], "loaded": []}
    assert record.read_text(encoding="utf-8") == ""
