"""Stdlib downloader for the CWRU benchmark: cache, pins, fail-closed verify.

Raw ``.mat`` records are downloaded on demand into the gitignored cache
``data/cache/cwru/`` (never committed, never redistributed). Two modes:

- **verify** (default): every record must already have a pin (SHA-256 +
  byte size) in ``benchmarks/cwru/checksums.json``. Missing pin, size
  mismatch, hash mismatch, and empty body each fail closed with a
  distinct "problem — remedy" error. A pre-existing final cache file is
  re-verified (size + hash) on every :func:`ensure_cached` call and
  re-downloaded if it no longer matches its pin, so a corrupted or
  tampered cache can never satisfy the benchmark silently.
- **freeze** (maintainer-only): :func:`freeze_checksums` downloads each
  record, computes its pin, and writes ``checksums.json``. It refuses to
  overwrite an existing pin with a different one unless ``force=True``,
  and freezing a subset never deletes pins outside that subset.

Hardening decisions (documented here because they are load-bearing):

- **Chunked streaming, 1 MiB chunks** (:data:`CHUNK_SIZE_BYTES`): the
  response is streamed to ``<name>.part`` and hashed as it arrives; the
  final cache name is created only by an atomic :func:`os.replace` after
  every check passes. On any abort the ``.part`` is removed and the
  final name is never created.
- **Streaming size cap**: the transfer aborts the moment cumulative
  bytes exceed the pinned size (verify) or
  :data:`FREEZE_CEILING_BYTES` (freeze, where no pin exists yet). The
  ceiling is 64 MiB — the largest v1 record is ~4 MB, so the ceiling
  never refuses a legitimate CWRU file while bounding what a hostile or
  corrupt response can write to disk.
- **Scheme and redirect enforcement in every mode, freeze included**
  (a poisoned freeze would pin a poisoned file permanently): URLs must
  be ``https``, and the default opener's
  :class:`SameOriginHttpsRedirectHandler` refuses any redirect that
  changes scheme or host.
- **Checksum key = ``cache_filename``** from the ops view — the one
  name the downloader is allowed to build paths from, already validated
  as a single safe path component by the ops table model. Every cache
  path is resolved through the repo's ``path_safety`` choke point.
- **No new dependencies**: ``urllib.request`` + ``hashlib`` only, with
  a bounded retry (2 retries, short linear backoff) around transient
  network errors. Guard failures (cap, scheme, verification) are never
  retried.

The network fetch is an injectable seam (``fetch=`` parameter) so tests
exercise every guard without any network I/O.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.request
from contextlib import AbstractContextManager
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Callable, Iterable, Literal, Optional, Protocol
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchmarks.cwru.records import OpsRecord
from predictive_maintenance_mcp.config import PROJECT_ROOT
from predictive_maintenance_mcp.path_safety import safe_resolve

__all__ = [
    "CHECKSUMS_PATH",
    "CHUNK_SIZE_BYTES",
    "DEFAULT_CACHE_ROOT",
    "FREEZE_CEILING_BYTES",
    "ChecksumPin",
    "Fetcher",
    "Mode",
    "ResponseLike",
    "SameOriginHttpsRedirectHandler",
    "build_https_opener",
    "ensure_cached",
    "freeze_checksums",
]

#: Bounded streaming chunk size: 1 MiB per read keeps memory flat while
#: hashing multi-MB records in a handful of iterations.
CHUNK_SIZE_BYTES: int = 1 << 20

#: Freeze-mode size ceiling (64 MiB). No pin exists yet during freeze, so
#: this is the only bound on what a response may write to disk. The
#: largest real v1 record is ~4 MB; 64 MiB never refuses a legitimate
#: CWRU file. Raise it here, deliberately, if a larger record is added.
FREEZE_CEILING_BYTES: int = 64 * 1024 * 1024

#: Vendored pin table, written by the maintainer freeze step and then
#: committed; required by every verify-mode download.
CHECKSUMS_PATH: Path = Path(__file__).resolve().parent / "checksums.json"

#: Raw ``.mat`` cache root — gitignored, outside ``data/signals`` so
#: cached CWRU filenames are never surfaced by signal listing.
DEFAULT_CACHE_ROOT: Path = Path(PROJECT_ROOT) / "data" / "cache" / "cwru"

#: Download mode: ``"verify"`` requires a pin; ``"freeze"`` creates one.
Mode = Literal["verify", "freeze"]

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 0.5
_DEFAULT_TIMEOUT_S = 30.0

#: Transient transport errors worth a bounded retry. ``URLError`` is an
#: ``OSError`` subclass, so both are covered; our own ``ValueError``
#: guards are deliberately NOT in this tuple and are never retried.
_RETRYABLE_ERRORS = (OSError, http.client.HTTPException)

_FREEZE_REMEDY = (
    "Run the maintainer freeze step (python -m benchmarks.cwru freeze) "
    "and commit checksums.json first."
)


class ResponseLike(Protocol):
    """Minimal read interface the downloader needs from a response."""

    def read(self, amt: int = ..., /) -> bytes:
        """Read up to *amt* bytes; empty bytes signals end of stream."""
        ...


#: Injectable fetch seam: ``(url, timeout) -> context manager`` yielding a
#: :class:`ResponseLike`. Tests replace it; production uses
#: :func:`build_https_opener` via the default seam.
Fetcher = Callable[[str, float], AbstractContextManager[ResponseLike]]


class ChecksumPin(BaseModel):
    """One committed pin: the only acceptable content for a cache file.

    Attributes:
        sha256: Lowercase hex SHA-256 of the exact file bytes.
        bytes: Exact byte size. Strictly positive — a 0-byte pin could
            turn an upstream failure into a "verified" empty file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)


class SameOriginHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses scheme or host changes, fail closed.

    Installed in every mode, freeze included: during freeze no pin exists
    yet, so a redirect to an attacker host would otherwise be pinned
    permanently as the ground truth.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        """Follow same-origin https redirects only.

        Raises:
            ValueError: If the redirect target changes scheme or host.
        """
        resolved = urljoin(req.full_url, newurl)
        original = urlsplit(req.full_url)
        target = urlsplit(resolved)
        if target.scheme.lower() != "https" or (
            target.netloc.lower() != original.netloc.lower()
        ):
            raise ValueError(
                f"Refusing redirect from {req.full_url} to {resolved} — "
                f"redirects that change scheme or host are refused in "
                f"every mode (fail closed). If the CWRU site really "
                f"moved, update records_ops.json and re-freeze checksums "
                f"deliberately."
            )
        redirected: Optional[urllib.request.Request] = super().redirect_request(
            req, fp, code, msg, headers, resolved
        )
        return redirected


def build_https_opener() -> urllib.request.OpenerDirector:
    """Build the hardened opener used by the default fetch seam.

    Passing a :class:`SameOriginHttpsRedirectHandler` instance to
    ``build_opener`` replaces the default redirect handler, so every
    redirect goes through the same-origin guard.

    Returns:
        An opener whose redirect handling refuses scheme/host changes.
    """
    return urllib.request.build_opener(SameOriginHttpsRedirectHandler())


def _open_https(url: str, timeout: float) -> AbstractContextManager[ResponseLike]:
    """Default fetch seam: open *url* with the hardened opener."""
    response: AbstractContextManager[ResponseLike] = build_https_opener().open(
        url, timeout=timeout
    )
    return response


def _require_https(record: OpsRecord) -> None:
    """Refuse any non-https URL up front, in every mode.

    Args:
        record: The ops record whose URL is about to be fetched.

    Raises:
        ValueError: If the record's URL scheme is not exactly ``https``.
    """
    scheme = urlsplit(record.url).scheme.lower()
    if scheme != "https":
        raise ValueError(
            f"Record '{record.opaque_id}' URL '{record.url}' uses scheme "
            f"'{scheme or '<none>'}' — only https downloads are allowed, "
            f"in freeze mode too (a poisoned freeze would pin a poisoned "
            f"file). Fix the URL in records_ops.json."
        )


def _cache_paths(record: OpsRecord, cache_dir: Optional[Path]) -> tuple[Path, Path]:
    """Resolve the record's final and ``.part`` cache paths, contained.

    Both names derive from the ops view's ``cache_filename`` (already
    validated as a single safe component) and are resolved through the
    repo's path-safety choke point, never by string concatenation.

    Args:
        record: The ops record being cached.
        cache_dir: Cache root override (tests only); defaults to
            :data:`DEFAULT_CACHE_ROOT`.

    Returns:
        ``(final_path, part_path)`` under the cache root, which is
        created if absent.
    """
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    final_path: Path = safe_resolve(root, record.cache_filename)
    part_path: Path = safe_resolve(root, record.cache_filename + ".part")
    return final_path, part_path


def _load_pins(pins_path: Path, *, required: bool) -> dict[str, ChecksumPin]:
    """Load the committed pin table.

    Args:
        pins_path: Path to ``checksums.json``.
        required: Verify mode passes ``True``: a missing table is a
            fail-closed error naming the freeze remedy. Freeze mode
            passes ``False``: a missing table is simply empty.

    Returns:
        Mapping of ``cache_filename`` to :class:`ChecksumPin`.

    Raises:
        ValueError: If the table is required but missing, is not valid
            JSON, is not an object, or contains an invalid entry.
    """
    if not pins_path.exists():
        if required:
            raise ValueError(
                f"Checksum table not found at {pins_path} — verify-mode "
                f"downloads require pinned hashes. {_FREEZE_REMEDY}"
            )
        return {}
    try:
        raw = json.loads(pins_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Checksum table {pins_path} is not valid JSON ({exc}) — "
            f"restore it from git, or regenerate it with the freeze step."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"Checksum table {pins_path} must be a JSON object keyed by "
            f"cache filename, got {type(raw).__name__} — restore it from "
            f"git, or regenerate it with the freeze step."
        )
    pins: dict[str, ChecksumPin] = {}
    for name, entry in raw.items():
        try:
            pins[name] = ChecksumPin.model_validate(entry)
        except ValidationError as exc:
            raise ValueError(
                f"Checksum table {pins_path} entry '{name}' failed "
                f"validation — restore it from git, or regenerate it "
                f"with the freeze step. Details: {exc}"
            ) from exc
    return pins


def _write_pins(pins_path: Path, pins: dict[str, ChecksumPin]) -> None:
    """Serialize *pins* to *pins_path* with sorted keys, atomically."""
    payload = {
        name: {"bytes": pin.bytes, "sha256": pin.sha256} for name, pin in pins.items()
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = pins_path.with_name(pins_path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, pins_path)


def _stream_to_part(
    response: ResponseLike,
    part_path: Path,
    *,
    byte_cap: int,
    cap_description: str,
    cap_remedy: str,
    record: OpsRecord,
) -> tuple[str, int]:
    """Stream *response* to *part_path* in bounded chunks, hashing as it goes.

    Aborts the moment cumulative bytes exceed *byte_cap*: the oversized
    remainder of the body is never read, let alone written.

    Args:
        response: Open response to read from.
        part_path: The ``.part`` destination (truncated if it exists,
            e.g. a stale leftover from a killed process).
        byte_cap: Hard cumulative-byte limit for this transfer.
        cap_description: Human-readable name of the cap for the error.
        cap_remedy: Remedy sentence appended to the cap error.
        record: The record being downloaded, for error messages.

    Returns:
        ``(sha256_hexdigest, total_bytes)`` of the streamed content.

    Raises:
        ValueError: If the cumulative size exceeds *byte_cap*.
    """
    digest = hashlib.sha256()
    total = 0
    with open(part_path, "wb") as sink:
        while True:
            chunk = response.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > byte_cap:
                raise ValueError(
                    f"Download for record '{record.opaque_id}' exceeded "
                    f"{cap_description} after {total} bytes — aborting "
                    f"mid-stream and removing the partial file. "
                    f"{cap_remedy}"
                )
            digest.update(chunk)
            sink.write(chunk)
    return digest.hexdigest(), total


def _download_to_part(
    record: OpsRecord,
    part_path: Path,
    *,
    byte_cap: int,
    cap_description: str,
    cap_remedy: str,
    fetch: Fetcher,
    timeout: float,
) -> tuple[str, int]:
    """Download the record to its ``.part`` path with bounded retry.

    Only transient transport errors are retried; guard failures (cap,
    redirect refusal) propagate immediately. On any failure the
    ``.part`` file is removed; on success it is left in place for the
    caller to verify and promote.

    Args:
        record: The ops record to download.
        part_path: Destination ``.part`` path.
        byte_cap: Hard cumulative-byte limit for the transfer.
        cap_description: Human-readable name of the cap for the error.
        cap_remedy: Remedy sentence appended to the cap error.
        fetch: The fetch seam.
        timeout: Per-attempt socket timeout in seconds.

    Returns:
        ``(sha256_hexdigest, total_bytes)`` of the downloaded content.

    Raises:
        ValueError: If the cap trips, or every attempt fails.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with fetch(record.url, timeout) as response:
                return _stream_to_part(
                    response,
                    part_path,
                    byte_cap=byte_cap,
                    cap_description=cap_description,
                    cap_remedy=cap_remedy,
                    record=record,
                )
        except _RETRYABLE_ERRORS as exc:
            last_error = exc
            part_path.unlink(missing_ok=True)
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS * attempt)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
    raise ValueError(
        f"Download failed for record '{record.opaque_id}' from "
        f"{record.url} after {_MAX_ATTEMPTS} attempts (last error: "
        f"{last_error}) — check network connectivity and the CWRU "
        f"Bearing Data Center site, then retry."
    ) from last_error


def _check_not_empty(record: OpsRecord, total: int) -> None:
    """Refuse an empty body — 0 bytes is never a valid record.

    Raises:
        ValueError: If *total* is zero.
    """
    if total == 0:
        raise ValueError(
            f"Download for record '{record.opaque_id}' from {record.url} "
            f"returned an empty body (0 bytes) — zero bytes is never a "
            f"valid record; the server likely errored. Retry, and check "
            f"the URL in records_ops.json if it persists."
        )


def _verify_streamed(
    record: OpsRecord, pin: ChecksumPin, digest: str, total: int
) -> None:
    """Check a completed transfer against its pin, fail closed.

    Args:
        record: The record that was downloaded.
        pin: The committed pin it must match.
        digest: SHA-256 hexdigest of the streamed bytes.
        total: Number of bytes streamed.

    Raises:
        ValueError: Distinct errors for empty body, size mismatch, and
            hash mismatch, each naming expected and actual values.
    """
    _check_not_empty(record, total)
    if total != pin.bytes:
        raise ValueError(
            f"Download for record '{record.opaque_id}' has wrong size: "
            f"expected {pin.bytes} bytes (pinned), got {total} — the "
            f"upstream file changed or the transfer is corrupt; the "
            f"partial file was removed. If the CWRU dataset legitimately "
            f"changed, re-run the freeze step with force=True and review "
            f"the diff."
        )
    if digest != pin.sha256:
        raise ValueError(
            f"Download for record '{record.opaque_id}' failed sha256 "
            f"verification: expected {pin.sha256}, got {digest} — the "
            f"upstream file changed or the transfer is corrupt; the "
            f"partial file was removed. If the CWRU dataset legitimately "
            f"changed, re-run the freeze step with force=True and review "
            f"the diff."
        )


def _file_matches_pin(path: Path, pin: ChecksumPin) -> bool:
    """Return whether the file at *path* matches *pin* (size and hash)."""
    if path.stat().st_size != pin.bytes:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest() == pin.sha256


def ensure_cached(
    record: OpsRecord,
    *,
    mode: Mode = "verify",
    cache_dir: Optional[Path] = None,
    checksums_path: Optional[Path] = None,
    fetch: Optional[Fetcher] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Path:
    """Ensure the record's raw ``.mat`` file is present and trustworthy.

    In verify mode (the default) the record must have a committed pin;
    a pre-existing final cache file is re-verified (size + SHA-256) on
    every call and deleted + re-downloaded if it no longer matches, so
    a corrupt or tampered cache can never satisfy the benchmark. A fresh
    download is streamed to ``<name>.part`` under the pinned size cap
    and promoted to the final name by an atomic rename only after the
    empty-body, size, and hash checks all pass.

    In freeze mode no pin exists yet: the download runs under
    :data:`FREEZE_CEILING_BYTES` with the same scheme/redirect
    enforcement, always re-downloads (the pin must come from the live
    official source, never from a stale cache file), and is promoted
    after the empty-body check. Pin bookkeeping itself lives in
    :func:`freeze_checksums`.

    Args:
        record: Ops record to cache (the sole source of URL + filename).
        mode: ``"verify"`` (default) or ``"freeze"``.
        cache_dir: Cache root override (tests only).
        checksums_path: Pin table override (tests only).
        fetch: Fetch seam override (tests only); defaults to the
            hardened https opener.
        timeout: Per-attempt socket timeout in seconds.

    Returns:
        The final cache path of the verified file.

    Raises:
        ValueError: Fail-closed, with a "problem — remedy" message, on:
            unknown mode, non-https URL, cross-origin redirect, missing
            pin, size-cap breach, empty body, size mismatch, hash
            mismatch, or exhausted retries.
    """
    if mode not in ("verify", "freeze"):
        raise ValueError(
            f"Unknown download mode '{mode}' — use 'verify' (default, "
            f"requires pinned checksums) or 'freeze' (maintainer pin "
            f"step)."
        )
    _require_https(record)
    final_path, part_path = _cache_paths(record, cache_dir)
    fetch_seam = fetch if fetch is not None else _open_https

    if mode == "freeze":
        digest, total = _download_to_part(
            record,
            part_path,
            byte_cap=FREEZE_CEILING_BYTES,
            cap_description=(f"the freeze-mode ceiling ({FREEZE_CEILING_BYTES} bytes)"),
            cap_remedy=(
                "If a legitimately larger record was added, raise "
                "FREEZE_CEILING_BYTES in benchmarks/cwru/download.py "
                "deliberately."
            ),
            fetch=fetch_seam,
            timeout=timeout,
        )
        try:
            _check_not_empty(record, total)
        except ValueError:
            part_path.unlink(missing_ok=True)
            raise
        os.replace(part_path, final_path)
        return final_path

    pins_path = checksums_path if checksums_path is not None else CHECKSUMS_PATH
    pins = _load_pins(pins_path, required=True)
    pin = pins.get(record.cache_filename)
    if pin is None:
        raise ValueError(
            f"No checksum pin for record '{record.opaque_id}' (cache "
            f"file '{record.cache_filename}') in {pins_path} — every "
            f"verify-mode download requires a pinned sha256 and byte "
            f"size. {_FREEZE_REMEDY}"
        )

    if final_path.exists():
        if _file_matches_pin(final_path, pin):
            return final_path
        # Fail closed: drop the bad file now so an aborted re-download
        # cannot leave an unverified file under the final name.
        final_path.unlink()

    digest, total = _download_to_part(
        record,
        part_path,
        byte_cap=pin.bytes,
        cap_description=f"the pinned size ({pin.bytes} bytes)",
        cap_remedy=(
            "If the CWRU dataset legitimately changed, re-run the freeze "
            "step with force=True and review the diff."
        ),
        fetch=fetch_seam,
        timeout=timeout,
    )
    try:
        _verify_streamed(record, pin, digest, total)
    except ValueError:
        part_path.unlink(missing_ok=True)
        raise
    os.replace(part_path, final_path)
    return final_path


def freeze_checksums(
    records: Iterable[OpsRecord],
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    checksums_path: Optional[Path] = None,
    fetch: Optional[Fetcher] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, ChecksumPin]:
    """Maintainer-only: download records and pin their checksums.

    Each record is downloaded fresh (ceiling-capped, https-only,
    same-origin redirects only), hashed, and promoted into the cache;
    the resulting pins are merged into ``checksums.json`` with sorted
    keys. Freezing a subset never deletes pins outside that subset.

    A record whose fresh download disagrees with its existing pin is
    refused unless ``force=True`` — a silent overwrite would let an
    upstream (or man-in-the-middle) change redefine the benchmark's
    ground truth without anyone noticing. On refusal nothing is written:
    the conflicting download is discarded and the existing table is left
    byte-identical.

    Args:
        records: Ops records to pin (typically the full ops view).
        force: Allow overwriting an existing, differing pin.
        cache_dir: Cache root override (tests only).
        checksums_path: Pin table override (tests only).
        fetch: Fetch seam override (tests only).
        timeout: Per-attempt socket timeout in seconds.

    Returns:
        The full pin mapping as written (existing pins merged with the
        newly frozen ones).

    Raises:
        ValueError: Fail-closed on non-https URL, cross-origin redirect,
            ceiling breach, empty body, exhausted retries, or a
            differing existing pin without ``force``.
    """
    pins_path = checksums_path if checksums_path is not None else CHECKSUMS_PATH
    existing = _load_pins(pins_path, required=False)
    updated = dict(existing)
    fetch_seam = fetch if fetch is not None else _open_https

    for record in records:
        _require_https(record)
        final_path, part_path = _cache_paths(record, cache_dir)
        digest, total = _download_to_part(
            record,
            part_path,
            byte_cap=FREEZE_CEILING_BYTES,
            cap_description=(f"the freeze-mode ceiling ({FREEZE_CEILING_BYTES} bytes)"),
            cap_remedy=(
                "If a legitimately larger record was added, raise "
                "FREEZE_CEILING_BYTES in benchmarks/cwru/download.py "
                "deliberately."
            ),
            fetch=fetch_seam,
            timeout=timeout,
        )
        try:
            _check_not_empty(record, total)
        except ValueError:
            part_path.unlink(missing_ok=True)
            raise
        new_pin = ChecksumPin(sha256=digest, bytes=total)
        old_pin = existing.get(record.cache_filename)
        if old_pin is not None and old_pin != new_pin and not force:
            part_path.unlink(missing_ok=True)
            raise ValueError(
                f"checksums table {pins_path} already pins "
                f"'{record.cache_filename}' (record "
                f"'{record.opaque_id}') to sha256 {old_pin.sha256} "
                f"({old_pin.bytes} bytes) but the fresh download "
                f"produced {new_pin.sha256} ({new_pin.bytes} bytes) — "
                f"refusing to overwrite the pin silently. If the "
                f"upstream dataset legitimately changed, re-run freeze "
                f"with force=True and document the change; nothing was "
                f"written."
            )
        os.replace(part_path, final_path)
        updated[record.cache_filename] = new_pin

    _write_pins(pins_path, updated)
    return updated
