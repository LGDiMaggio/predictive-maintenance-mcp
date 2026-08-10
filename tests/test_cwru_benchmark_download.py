"""U2 tests: downloader, cache, and fail-closed checksum pinning.

Every guard is exercised through the injectable fetch seam — no test
performs network I/O, so the suite is CI-safe by construction. Each
fail-closed guard is demonstrated to fire (wrong hash, empty body,
missing pin, size-cap breach, scheme/redirect refusal, differing freeze
pin), and each has a green counterpart proving the harness itself is
sound, mirroring the U1 mutation-guard style.
"""

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Optional

import pytest

from benchmarks.cwru import download
from benchmarks.cwru.download import (
    SameOriginHttpsRedirectHandler,
    build_https_opener,
    ensure_cached,
    freeze_checksums,
)
from benchmarks.cwru.records import OpsRecord

PAYLOAD = bytes(range(256)) * 4  # 1024 deterministic bytes
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()

CWRU_URL = "https://engineering.case.edu/sites/default/files/105.mat"


def _record(
    url: str = CWRU_URL,
    opaque_id: str = "cwru_001",
    file_id: int = 105,
    cache_filename: str = "105.mat",
) -> OpsRecord:
    return OpsRecord(
        opaque_id=opaque_id,
        file_id=file_id,
        url=url,
        internal_mat_key=None,
        channel="DE",
        fs_hz=12000,
        nominal_rpm=1797,
        load_hp=0,
        cache_filename=cache_filename,
    )


class _FakeResponse:
    """Seam response serving pre-cut chunks, policing read counts.

    Args:
        chunks: Chunks handed out one per ``read`` call, then ``b""``.
        max_reads: If set, a read beyond this count raises
            AssertionError — proof the downloader stopped consuming the
            body once its size cap tripped.
        fail_reads_with: If set, every ``read`` raises this exception —
            simulates a connection dying mid-transfer.
    """

    def __init__(
        self,
        chunks: list[bytes],
        *,
        max_reads: Optional[int] = None,
        fail_reads_with: Optional[BaseException] = None,
    ) -> None:
        self._chunks = list(chunks)
        self._max_reads = max_reads
        self._fail_reads_with = fail_reads_with
        self.reads = 0

    def read(self, amt: int = -1) -> bytes:
        if self._fail_reads_with is not None:
            raise self._fail_reads_with
        if self._max_reads is not None and self.reads >= self._max_reads:
            raise AssertionError(
                "downloader read past its size cap — the oversized body "
                "must never be fully consumed"
            )
        self.reads += 1
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _chunked(body: bytes, size: int) -> list[bytes]:
    return [body[i : i + size] for i in range(0, len(body), size)]


def _serving(
    payloads: dict[str, bytes],
    *,
    chunk_size: int = 512,
    max_reads: Optional[int] = None,
    calls: Optional[list[str]] = None,
):
    """Build a fetch seam serving *payloads* keyed by URL."""

    def fetch(url: str, timeout: float) -> _FakeResponse:
        if calls is not None:
            calls.append(url)
        return _FakeResponse(_chunked(payloads[url], chunk_size), max_reads=max_reads)

    return fetch


def _endless(chunk: bytes, *, max_reads: int, calls: Optional[list[str]] = None):
    """Build a fetch seam serving an effectively unbounded body."""

    def fetch(url: str, timeout: float) -> _FakeResponse:
        if calls is not None:
            calls.append(url)
        # More chunks than max_reads allows: the cap must trip first.
        return _FakeResponse([chunk] * (max_reads + 8), max_reads=max_reads)

    return fetch


def _refusing_fetch(calls: list[str]):
    """Build a fetch seam that must never be reached."""

    def fetch(url: str, timeout: float) -> _FakeResponse:
        calls.append(url)
        raise AssertionError("fetch seam must not be called")

    return fetch


def _write_pins(path: Path, mapping: dict) -> None:
    path.write_text(json.dumps(mapping), encoding="utf-8")


def _good_pins(tmp_path: Path, filename: str = "105.mat") -> Path:
    pins = tmp_path / "checksums.json"
    _write_pins(pins, {filename: {"sha256": PAYLOAD_SHA256, "bytes": len(PAYLOAD)}})
    return pins


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# Happy path: verified download lands under the final name
# ---------------------------------------------------------------------------


class TestVerifyHappyPath:
    """Seam download verifies against a correct pin and is promoted."""

    def test_download_verifies_and_lands_under_final_name(self, tmp_path, cache_dir):
        record = _record()
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_serving({record.url: PAYLOAD}),
        )
        assert result == cache_dir / "105.mat"
        assert result.read_bytes() == PAYLOAD
        assert not (cache_dir / "105.mat.part").exists()

    def test_existing_good_file_is_reverified_not_redownloaded(
        self, tmp_path, cache_dir
    ):
        record = _record()
        cache_dir.mkdir(parents=True)
        (cache_dir / "105.mat").write_bytes(PAYLOAD)
        calls: list[str] = []
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_refusing_fetch(calls),
        )
        assert result.read_bytes() == PAYLOAD
        assert calls == [], "a pin-matching cache file must not be re-fetched"

    def test_existing_bad_file_is_redownloaded(self, tmp_path, cache_dir):
        """A tampered/corrupt final file fails re-verification and is
        replaced by a fresh, verified download."""
        record = _record()
        cache_dir.mkdir(parents=True)
        (cache_dir / "105.mat").write_bytes(b"tampered garbage")
        calls: list[str] = []
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_serving({record.url: PAYLOAD}, calls=calls),
        )
        assert result.read_bytes() == PAYLOAD
        assert calls == [record.url]


# ---------------------------------------------------------------------------
# Fail-closed verification guards (each shown to fire)
# ---------------------------------------------------------------------------


class TestVerifyGuards:
    """Missing pin, wrong hash, wrong size, and empty body all refuse."""

    def test_wrong_pinned_hash_rejects_and_cleans_up(self, tmp_path, cache_dir):
        record = _record()
        pins = tmp_path / "checksums.json"
        _write_pins(
            pins,
            {"105.mat": {"sha256": "0" * 64, "bytes": len(PAYLOAD)}},
        )
        with pytest.raises(ValueError, match="sha256 verification") as excinfo:
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_serving({record.url: PAYLOAD}),
            )
        message = str(excinfo.value)
        assert "cwru_001" in message
        assert "0" * 64 in message  # expected value named
        assert PAYLOAD_SHA256 in message  # actual value named
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_size_mismatch_rejects_with_expected_and_actual(self, tmp_path, cache_dir):
        record = _record()
        pins = tmp_path / "checksums.json"
        bigger = len(PAYLOAD) + 200
        _write_pins(pins, {"105.mat": {"sha256": PAYLOAD_SHA256, "bytes": bigger}})
        with pytest.raises(ValueError, match="wrong size") as excinfo:
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_serving({record.url: PAYLOAD}),
            )
        message = str(excinfo.value)
        assert str(bigger) in message
        assert str(len(PAYLOAD)) in message
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_empty_body_rejects(self, tmp_path, cache_dir):
        record = _record()
        with pytest.raises(ValueError, match="empty body"):
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=_good_pins(tmp_path),
                fetch=_serving({record.url: b""}),
            )
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_missing_pin_entry_names_freeze_remedy(self, tmp_path, cache_dir):
        record = _record()
        pins = tmp_path / "checksums.json"
        _write_pins(pins, {"118.mat": {"sha256": "a" * 64, "bytes": 10}})
        calls: list[str] = []
        with pytest.raises(ValueError, match="freeze") as excinfo:
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_refusing_fetch(calls),
            )
        message = str(excinfo.value)
        assert "cwru_001" in message
        assert "105.mat" in message
        assert calls == [], "an unpinned record must not be downloaded"

    def test_missing_checksums_file_names_freeze_remedy(self, tmp_path, cache_dir):
        record = _record()
        calls: list[str] = []
        with pytest.raises(ValueError, match="freeze"):
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=tmp_path / "absent.json",
                fetch=_refusing_fetch(calls),
            )
        assert calls == []

    def test_unknown_mode_refused(self, tmp_path, cache_dir):
        with pytest.raises(ValueError, match="Unknown download mode"):
            ensure_cached(
                _record(),
                mode="yolo",  # type: ignore[arg-type]
                cache_dir=cache_dir,
                checksums_path=_good_pins(tmp_path),
                fetch=_serving({CWRU_URL: PAYLOAD}),
            )


# ---------------------------------------------------------------------------
# Streaming size cap: abort mid-stream, never read the full body
# ---------------------------------------------------------------------------


class TestStreamingSizeCap:
    """The cap trips on cumulative bytes, before the body is consumed."""

    def test_oversize_response_aborts_mid_stream_in_verify(self, tmp_path, cache_dir):
        record = _record()
        pins = tmp_path / "checksums.json"
        _write_pins(pins, {"105.mat": {"sha256": PAYLOAD_SHA256, "bytes": 1000}})
        # 512-byte chunks against a 1000-byte pin: read 1 -> 512 ok,
        # read 2 -> 1024 > 1000 must abort; a third read would raise
        # AssertionError inside the fake, proving the body was never
        # fully consumed.
        with pytest.raises(ValueError, match="exceeded the pinned size"):
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_endless(b"x" * 512, max_reads=2),
            )
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_freeze_ceiling_aborts_mid_stream(self, tmp_path, cache_dir, monkeypatch):
        monkeypatch.setattr(download, "FREEZE_CEILING_BYTES", 1000)
        with pytest.raises(ValueError, match="freeze-mode ceiling"):
            ensure_cached(
                _record(),
                mode="freeze",
                cache_dir=cache_dir,
                fetch=_endless(b"x" * 512, max_reads=2),
            )
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_freeze_checksums_shares_the_ceiling(
        self, tmp_path, cache_dir, monkeypatch
    ):
        monkeypatch.setattr(download, "FREEZE_CEILING_BYTES", 1000)
        pins = tmp_path / "checksums.json"
        with pytest.raises(ValueError, match="freeze-mode ceiling"):
            freeze_checksums(
                [_record()],
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_endless(b"x" * 512, max_reads=2),
            )
        assert not pins.exists(), "no pins may be written after an abort"

    def test_body_at_exactly_the_pinned_size_passes(self, tmp_path, cache_dir):
        """Green counterpart: the cap refuses only bytes BEYOND the pin."""
        record = _record()
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_serving({record.url: PAYLOAD}, chunk_size=512),
        )
        assert result.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Scheme and redirect enforcement, in every mode
# ---------------------------------------------------------------------------


class TestSchemeAndRedirectEnforcement:
    """http URLs and cross-origin redirects are refused fail-closed."""

    @pytest.mark.parametrize("mode", ["verify", "freeze"])
    def test_http_url_refused_in_both_modes(self, tmp_path, cache_dir, mode):
        record = _record(url="http://engineering.case.edu/sites/default/files/105.mat")
        calls: list[str] = []
        with pytest.raises(ValueError, match="only https"):
            ensure_cached(
                record,
                mode=mode,
                cache_dir=cache_dir,
                checksums_path=_good_pins(tmp_path),
                fetch=_refusing_fetch(calls),
            )
        assert calls == [], "a non-https URL must never be fetched"

    def test_http_url_refused_by_freeze_checksums(self, tmp_path, cache_dir):
        record = _record(url="http://engineering.case.edu/sites/default/files/105.mat")
        calls: list[str] = []
        pins = tmp_path / "checksums.json"
        with pytest.raises(ValueError, match="only https"):
            freeze_checksums(
                [record],
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_refusing_fetch(calls),
            )
        assert calls == []
        assert not pins.exists()

    @pytest.mark.parametrize(
        "newurl",
        [
            "https://evil.example/105.mat",  # host change
            "http://engineering.case.edu/sites/default/files/105.mat",  # scheme
        ],
    )
    def test_cross_origin_redirect_refused(self, newurl):
        handler = SameOriginHttpsRedirectHandler()
        request = urllib.request.Request(CWRU_URL)
        with pytest.raises(ValueError, match="Refusing redirect"):
            handler.redirect_request(request, None, 302, "Found", None, newurl)

    def test_same_origin_https_redirect_allowed(self):
        """Green counterpart: a same-host https redirect still works."""
        handler = SameOriginHttpsRedirectHandler()
        request = urllib.request.Request(CWRU_URL)
        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            None,
            "https://engineering.case.edu/sites/moved/105.mat",
        )
        assert isinstance(redirected, urllib.request.Request)
        assert redirected.full_url == (
            "https://engineering.case.edu/sites/moved/105.mat"
        )

    def test_relative_redirect_resolves_against_origin_and_is_allowed(self):
        handler = SameOriginHttpsRedirectHandler()
        request = urllib.request.Request(CWRU_URL)
        redirected = handler.redirect_request(
            request, None, 302, "Found", None, "/sites/moved/105.mat"
        )
        assert isinstance(redirected, urllib.request.Request)
        assert redirected.full_url == (
            "https://engineering.case.edu/sites/moved/105.mat"
        )

    def test_default_opener_is_wired_with_the_guard(self):
        """The hardened handler REPLACES the stock redirect handler."""
        opener = build_https_opener()
        redirect_handlers = [
            handler
            for handler in opener.handlers
            if isinstance(handler, urllib.request.HTTPRedirectHandler)
        ]
        assert redirect_handlers, "opener must have a redirect handler"
        assert all(
            isinstance(handler, SameOriginHttpsRedirectHandler)
            for handler in redirect_handlers
        )


# ---------------------------------------------------------------------------
# Interrupted downloads: .part semantics and re-runs
# ---------------------------------------------------------------------------


class TestInterruptedDownloads:
    """Stale .part files never block a re-run; failures clean up."""

    def test_stale_part_from_killed_run_is_replaced_on_rerun(self, tmp_path, cache_dir):
        record = _record()
        cache_dir.mkdir(parents=True)
        (cache_dir / "105.mat.part").write_bytes(b"leftover from a killed process")
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_serving({record.url: PAYLOAD}),
        )
        assert result.read_bytes() == PAYLOAD
        assert not (cache_dir / "105.mat.part").exists()

    def test_transport_failure_is_retried_then_cleaned_up(
        self, tmp_path, cache_dir, monkeypatch
    ):
        record = _record()
        monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
        calls: list[str] = []

        def fetch(url: str, timeout: float) -> _FakeResponse:
            calls.append(url)
            return _FakeResponse(
                [], fail_reads_with=ConnectionResetError("connection reset")
            )

        with pytest.raises(ValueError, match="after 3 attempts") as excinfo:
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=_good_pins(tmp_path),
                fetch=fetch,
            )
        assert "connection reset" in str(excinfo.value)
        assert len(calls) == 3, "retry must be bounded to 3 attempts"
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()

    def test_rerun_after_failure_succeeds(self, tmp_path, cache_dir, monkeypatch):
        """Green counterpart: the failed run leaves nothing behind that
        stops a later, healthy run from succeeding."""
        record = _record()
        monkeypatch.setattr(download.time, "sleep", lambda seconds: None)

        def dying_fetch(url: str, timeout: float) -> _FakeResponse:
            return _FakeResponse([], fail_reads_with=ConnectionResetError("reset"))

        with pytest.raises(ValueError, match="after 3 attempts"):
            ensure_cached(
                record,
                cache_dir=cache_dir,
                checksums_path=_good_pins(tmp_path),
                fetch=dying_fetch,
            )

        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=_good_pins(tmp_path),
            fetch=_serving({record.url: PAYLOAD}),
        )
        assert result.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Freeze mode: pin creation, subset merging, and overwrite refusal
# ---------------------------------------------------------------------------


class TestFreeze:
    """freeze_checksums pins correctly and refuses silent overwrites."""

    def test_freeze_writes_sorted_pins_and_promotes_files(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        record_a = _record()
        record_b = _record(
            url="https://engineering.case.edu/sites/default/files/118.mat",
            opaque_id="cwru_002",
            file_id=118,
            cache_filename="118.mat",
        )
        other_payload = b"different body"
        # Deliberately passed in reverse order: the written table must
        # still be key-sorted.
        result = freeze_checksums(
            [record_b, record_a],
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_serving({record_a.url: PAYLOAD, record_b.url: other_payload}),
        )
        written = json.loads(pins.read_text(encoding="utf-8"))
        assert list(written) == ["105.mat", "118.mat"]
        assert written["105.mat"] == {
            "bytes": len(PAYLOAD),
            "sha256": PAYLOAD_SHA256,
        }
        assert written["118.mat"] == {
            "bytes": len(other_payload),
            "sha256": hashlib.sha256(other_payload).hexdigest(),
        }
        assert (cache_dir / "105.mat").read_bytes() == PAYLOAD
        assert (cache_dir / "118.mat").read_bytes() == other_payload
        assert set(result) == {"105.mat", "118.mat"}

    def test_freeze_then_verify_roundtrip_without_refetch(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        record = _record()
        freeze_checksums(
            [record],
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_serving({record.url: PAYLOAD}),
        )
        calls: list[str] = []
        result = ensure_cached(
            record,
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_refusing_fetch(calls),
        )
        assert result.read_bytes() == PAYLOAD
        assert calls == [], "the frozen file must satisfy verify without refetch"

    def test_freeze_subset_preserves_existing_pins(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        _write_pins(pins, {"999.mat": {"sha256": "b" * 64, "bytes": 42}})
        record = _record()
        freeze_checksums(
            [record],
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_serving({record.url: PAYLOAD}),
        )
        written = json.loads(pins.read_text(encoding="utf-8"))
        assert written["999.mat"] == {"bytes": 42, "sha256": "b" * 64}
        assert written["105.mat"]["sha256"] == PAYLOAD_SHA256

    def test_freeze_refuses_differing_pin_without_force(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        _write_pins(pins, {"105.mat": {"sha256": "c" * 64, "bytes": 7}})
        before = pins.read_text(encoding="utf-8")
        record = _record()
        with pytest.raises(ValueError, match="force=True") as excinfo:
            freeze_checksums(
                [record],
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_serving({record.url: PAYLOAD}),
            )
        assert "refusing to overwrite" in str(excinfo.value)
        assert pins.read_text(encoding="utf-8") == before, "table must be untouched"
        assert not (cache_dir / "105.mat").exists(), "conflicting file not promoted"
        assert not (cache_dir / "105.mat.part").exists()

    def test_freeze_force_overwrites_differing_pin(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        _write_pins(pins, {"105.mat": {"sha256": "c" * 64, "bytes": 7}})
        record = _record()
        freeze_checksums(
            [record],
            force=True,
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_serving({record.url: PAYLOAD}),
        )
        written = json.loads(pins.read_text(encoding="utf-8"))
        assert written["105.mat"] == {
            "bytes": len(PAYLOAD),
            "sha256": PAYLOAD_SHA256,
        }

    def test_freeze_with_identical_pin_needs_no_force(self, tmp_path, cache_dir):
        """Green counterpart: re-freezing an unchanged record is not a
        conflict — the guard fires only on a DIFFERING pin."""
        pins = _good_pins(tmp_path)
        record = _record()
        freeze_checksums(
            [record],
            cache_dir=cache_dir,
            checksums_path=pins,
            fetch=_serving({record.url: PAYLOAD}),
        )
        written = json.loads(pins.read_text(encoding="utf-8"))
        assert written["105.mat"]["sha256"] == PAYLOAD_SHA256

    def test_freeze_refuses_empty_body(self, tmp_path, cache_dir):
        pins = tmp_path / "checksums.json"
        record = _record()
        with pytest.raises(ValueError, match="empty body"):
            freeze_checksums(
                [record],
                cache_dir=cache_dir,
                checksums_path=pins,
                fetch=_serving({record.url: b""}),
            )
        assert not pins.exists(), "an empty body must never be pinned"
        assert not (cache_dir / "105.mat").exists()
        assert not (cache_dir / "105.mat.part").exists()
