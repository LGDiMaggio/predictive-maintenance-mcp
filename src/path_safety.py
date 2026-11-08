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

import errno
import ntpath
import os
import posixpath
import re
import stat
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, Optional

__all__ = [
    "safe_resolve",
    "sanitize_filename",
    "validate_name_component",
    "resolve_under",
    "resolve_model_paths",
    "ModelPaths",
    "WINDOWS_DEVICE_NAMES",
    "IMPORT_SUFFIXES",
    "PathRefused",
    "GatedPath",
    "gate_import_path",
    "VerifiedFile",
    "open_import_file",
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
    candidate = _without_extended_prefix((Path(base_dir) / user_input).resolve())
    allowed = _without_extended_prefix(Path(base_dir).resolve())
    # Path.is_relative_to (Python 3.9+) avoids the sibling-directory bypass:
    # /data/signals_evil would pass a naive startswith("/data/signals") check.
    if not candidate.is_relative_to(allowed):
        raise ValueError(f"Invalid path — escapes base directory: {user_input}")
    return candidate


def resolve_under(resolved_base: Path, user_input: str) -> Path:
    """:func:`safe_resolve` for a base the caller resolved once already.

    A loop over many locations resolves its base directory once instead of
    once per location.

    Raises:
        ValueError: If the resolved path escapes *resolved_base*.
    """
    candidate = _without_extended_prefix((resolved_base / user_input).resolve())
    if not candidate.is_relative_to(resolved_base):
        raise ValueError(f"Invalid path — escapes base directory: {user_input}")
    return candidate


_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


def _without_extended_prefix(path: Path) -> Path:
    """Drop the Windows extended-length prefix ``\\\\?\\`` when present.

    ``Path.resolve`` on Windows normally strips the prefix that the Win32
    final-path API returns, but it keeps it when the check it performs to do
    so hits a transient error (a sharing violation while another process or
    an antivirus scanner holds the file open). The prefix is only a namespace
    marker: comparing a prefixed candidate against an unprefixed base would
    report a false escape, so containment is checked on the plain form.
    """
    text = str(path)
    if text.startswith(_EXTENDED_UNC_PREFIX):
        return Path("\\\\" + text[len(_EXTENDED_UNC_PREFIX) :])
    if text.startswith(_EXTENDED_PREFIX):
        return Path(text[len(_EXTENDED_PREFIX) :])
    return path


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


# ---------------------------------------------------------------------------
# Import paths (Signal Import Engine, KTD5): a lexical gate that runs before
# any operating-system call, then checks on the one open descriptor through
# which every later read goes.
# ---------------------------------------------------------------------------

#: Win32 device names. A path component whose stem (before the first dot,
#: trailing spaces ignored) is one of them names a device on Windows, with or
#: without an extension, so it is refused in import paths on every operating
#: system. A superset of ``measurement.WIN32_RESERVED_NAMES`` (the console
#: names and the superscript port numbers Windows also treats as devices).
WINDOWS_DEVICE_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{digit}" for port in ("COM", "LPT") for digit in "123456789"}
    | {f"{port}{chr(code)}" for port in ("COM", "LPT") for code in (0xB9, 0xB2, 0xB3)}
)

#: Suffixes the import reads, compared case-insensitively on the resolved
#: final name (which also catches a Windows 8.3 alias of another extension).
IMPORT_SUFFIXES: tuple[str, ...] = (".csv", ".txt")

#: Longest path string the gate considers.
_MAX_PATH_CHARS = 4096

#: Longest path component (the limit of NTFS, ext4 and APFS names).
_MAX_COMPONENT_CHARS = 255

_CHANGED_REMEDY = (
    "the file changed while it was being read, so the export may still be in "
    "progress: wait until it is complete, then validate it again"
)


class PathRefused(ValueError):
    """An import path refused by the gate or by the open-file checks.

    Attributes:
        code: The issue code of the import contract (``unsafe_path``,
            ``path_outside_roots``, ``file_not_found``, ``file_unreadable``,
            ``not_a_regular_file``, ``unsupported_suffix``, ``file_too_large``,
            ``source_changed``).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GatedPath(NamedTuple):
    """A path that passed the lexical gate (nothing resolved, nothing opened).

    Attributes:
        path: The lexically normalized absolute path.
        root: The normalized allowed root that contains it (the most specific
            one when roots nest).
        relative: *path* relative to *root*, with the platform separator.
    """

    path: str
    root: str
    relative: str


def _unsafe(problem: str) -> PathRefused:
    return PathRefused(
        "unsafe_path",
        f"The path is refused: {problem}. Pass a plain local path "
        f"to a file under the data directory or a folder listed in "
        f"PMM_IMPORT_ROOTS (a network share through a mapped drive letter).",
    )


def _comparable(path: str, windows: bool) -> str:
    return path.replace("/", "\\").lower() if windows else path


def gate_import_path(
    raw: object,
    roots: Sequence[object],
    *,
    base: Optional[object] = None,
    windows: bool = os.name == "nt",
) -> GatedPath:
    """Refuse a dangerous import path before any operating-system call.

    Pure string work: nothing is resolved, stat-ed or opened, so a path that
    would make the host contact the network or a device (a UNC share, the
    ``\\\\?\\``, ``\\\\.\\`` or ``\\??\\`` namespaces, a device name) is refused
    without ever reaching the filesystem. Then the lexically normalized path
    must lie under one of *roots*.

    Refused on Windows: UNC and namespace prefixes, drive-relative
    (``C:data.csv``) and root-relative (``\\data.csv``) forms, and any ``:``
    after the drive (alternate data streams). Refused everywhere: control,
    format and line-separator characters, a leading ``//``, components that
    end with a dot or a space (Windows drops them silently, so the name
    would alias another file) and device names in any component.

    Args:
        raw: The path as given.
        roots: The allowed roots (absolute paths).
        base: Directory a relative *raw* is joined to (the data directory).
        windows: Apply the Windows path grammar; the default follows the host.

    Returns:
        The :class:`GatedPath`.

    Raises:
        PathRefused: ``unsafe_path`` or ``path_outside_roots``.
    """
    flavor = ntpath if windows else posixpath
    if not isinstance(raw, str) or not raw.strip():
        raise PathRefused("unsafe_path", "Pass the file path as a non-empty string.")
    if len(raw) > _MAX_PATH_CHARS:
        raise _unsafe(f"it is {len(raw)} characters long")
    if any(unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") for ch in raw):
        raise _unsafe("it contains control or invisible formatting characters")

    if windows:
        text = raw.replace("/", "\\")
        if text.startswith("\\\\"):
            raise _unsafe(
                "network shares and the \\\\?\\ and \\\\.\\ forms are not read"
            )
        if text.startswith("\\??\\"):
            raise _unsafe("the \\??\\ namespace is not read")
        drive, rest = ntpath.splitdrive(text)
        if drive and not re.fullmatch(r"[A-Za-z]:", drive):
            raise _unsafe("only a drive letter may precede the path")
        if drive and rest and not rest.startswith("\\"):
            raise _unsafe("a drive-relative path (no separator after the drive)")
        if not drive and text.startswith("\\"):
            raise _unsafe("a root-relative path (no drive letter)")
        if ":" in rest:
            raise _unsafe("a ':' after the drive (an alternate data stream)")
        if any(char in rest for char in '<>"|?*'):
            raise _unsafe('a character Windows reserves in names (< > " | ? *)')
        components = rest.split("\\")
    else:
        text = raw
        if text.startswith("//"):
            raise _unsafe("a path starting with '//'")
        components = text.split("/")

    for component in components:
        if component in ("", ".", ".."):
            continue
        if len(component) > _MAX_COMPONENT_CHARS:
            raise _unsafe(f"a path component is over {_MAX_COMPONENT_CHARS} characters")
        if component.endswith((".", " ")):
            raise _unsafe("a path component ends with a dot or a space")
        if component.split(".")[0].rstrip(" ").upper() in WINDOWS_DEVICE_NAMES:
            raise _unsafe("a path component is a Windows device name")

    if not flavor.isabs(text):
        if base is None:
            raise _unsafe("a relative path with no base directory")
        text = flavor.join(str(base), text)
    normalized = flavor.normpath(text)
    if not flavor.isabs(normalized):
        raise _unsafe("it does not form an absolute path")

    sep = "\\" if windows else "/"
    match: Optional[str] = None
    target = _comparable(normalized, windows)
    for root in roots:
        candidate = flavor.normpath(str(root))
        prefix = _comparable(candidate, windows).rstrip(sep) + sep
        if target == _comparable(candidate, windows) or target.startswith(prefix):
            if match is None or len(candidate) > len(match):
                match = candidate
    if match is None:
        raise PathRefused(
            "path_outside_roots",
            "The path is outside the folders the import reads: the "
            "data directory and the folders listed in PMM_IMPORT_ROOTS. Copy "
            "the file into one of them, or ask the operator to list its "
            "folder in PMM_IMPORT_ROOTS.",
        )
    relative = normalized[len(match.rstrip(sep)) :].lstrip(sep)
    return GatedPath(path=normalized, root=match, relative=relative)


class VerifiedFile:
    """One open, verified descriptor; every read goes through it.

    Reads never go past the size :func:`os.fstat` reported when the file was
    opened. :meth:`verify_unchanged` probes one byte beyond it and compares a
    fresh ``fstat``, so a file that grew, shrank or was rewritten while it
    was read is refused (the export may still be in progress).

    Attributes:
        path: The resolved path that was opened.
        size: The size in bytes at open time.
        bytes_read: Bytes returned by :meth:`read` so far.
    """

    def __init__(self, fd: int, path: Path, status: os.stat_result) -> None:
        self._fd: Optional[int] = fd
        self.path = path
        self.size = int(status.st_size)
        self._mtime_ns = status.st_mtime_ns
        self.bytes_read = 0

    def read(self, offset: int, length: int) -> bytes:
        """Up to *length* bytes from *offset*, never past :attr:`size`."""
        end = min(offset + max(length, 0), self.size)
        if offset >= end:
            return b""
        fd = self._require_open()
        os.lseek(fd, offset, os.SEEK_SET)
        parts: list[bytes] = []
        wanted = end - offset
        while wanted:
            data = os.read(fd, wanted)
            if not data:
                break
            parts.append(data)
            wanted -= len(data)
        result = b"".join(parts)
        self.bytes_read += len(result)
        return result

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        """Every byte of the file in order, then :meth:`verify_unchanged`."""
        offset = 0
        while offset < self.size:
            data = self.read(offset, chunk_size)
            if not data:
                raise PathRefused(
                    "source_changed",
                    f"The file became shorter while it "
                    f"was being read: {_CHANGED_REMEDY}.",
                )
            offset += len(data)
            yield data
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        """Refuse when the file grew past, or changed from, its open state."""
        fd = self._require_open()
        os.lseek(fd, self.size, os.SEEK_SET)
        grew = bool(os.read(fd, 1))
        now = os.fstat(fd)
        if grew or now.st_size != self.size or now.st_mtime_ns != self._mtime_ns:
            raise PathRefused(
                "source_changed",
                f"The file changed while it was being read: {_CHANGED_REMEDY}.",
            )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _require_open(self) -> int:
        if self._fd is None:
            raise ValueError("VerifiedFile is closed.")
        return self._fd

    def __enter__(self) -> "VerifiedFile":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


def open_import_file(
    raw: object,
    roots: Sequence[object],
    *,
    base: Optional[object],
    max_size: int,
) -> VerifiedFile:
    """Gate, resolve, check and open an import file (KTD5, in this order).

    1. :func:`gate_import_path` (no operating-system call before it passes);
    2. containment again after resolving links (:func:`safe_resolve`);
    3. the suffix of the resolved final name, ``.csv`` or ``.txt`` in any
       letter case;
    4. ``lstat``: a regular file within *max_size*;
    5. open read-only (no-follow and non-blocking on POSIX) and ``fstat`` the
       descriptor: still a regular file, the same file the ``lstat`` saw,
       within *max_size*.

    Returns:
        The open :class:`VerifiedFile`; the caller closes it.

    Raises:
        PathRefused: With the code of the failed check; an operating-system
            error never escapes with its own message (which names the full
            path).
    """
    gated = gate_import_path(raw, roots, base=base)
    try:
        resolved = safe_resolve(Path(gated.root), gated.relative)
    except ValueError:
        raise PathRefused(
            "path_outside_roots",
            "The path leads outside the folders the import reads once its "
            "links are followed. Copy the file into the data directory or a "
            "folder listed in PMM_IMPORT_ROOTS.",
        ) from None
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symbolic link loop
        raise _unreadable(exc) from None
    name = resolved.name
    if not name.lower().endswith(IMPORT_SUFFIXES):
        raise PathRefused(
            "unsupported_suffix",
            "The file is not read by the import: it reads .csv and "
            ".txt files (any letter case). If it is delimited text, import a "
            "copy renamed to .csv; the identity follows the bytes, not the name.",
        )
    try:
        seen = os.lstat(resolved)
    except (FileNotFoundError, NotADirectoryError):
        raise PathRefused(
            "file_not_found",
            "The file does not exist; check the path (list_signals shows the "
            "files under the data directory).",
        ) from None
    except OSError as exc:
        raise _unreadable(exc) from None
    if not stat.S_ISREG(seen.st_mode):
        raise PathRefused(
            "not_a_regular_file",
            "The path is not a regular file (a folder, a link, a "
            "pipe or a device); pass the path of the exported file itself.",
        )
    _refuse_oversized(seen.st_size, max_size)
    try:
        fd = os.open(resolved, _OPEN_FLAGS)
    except (FileNotFoundError, NotADirectoryError):
        raise PathRefused(
            "source_changed",
            f"The file disappeared between its check and its "
            f"opening: {_CHANGED_REMEDY}.",
        ) from None
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", None):
            raise PathRefused(
                "source_changed",
                f"The file was replaced by a link between its check "
                f"and its opening: {_CHANGED_REMEDY}.",
            ) from None
        raise _unreadable(exc) from None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            seen.st_dev,
            seen.st_ino,
        ):
            raise PathRefused(
                "source_changed",
                f"The file was replaced between its check and its "
                f"opening: {_CHANGED_REMEDY}.",
            )
        _refuse_oversized(opened.st_size, max_size)
    except BaseException:
        os.close(fd)
        raise
    return VerifiedFile(fd, resolved, opened)


def _unreadable(exc: BaseException) -> PathRefused:
    """A refusal for a file the operating system will not let the import
    read; the system's own message (which carries the full path) is not
    repeated."""
    reason = (
        "permission is denied, or another program holds the file open (a "
        "spreadsheet program locks the files it shows)"
        if isinstance(exc, PermissionError)
        else "the operating system refused to read it"
    )
    return PathRefused(
        "file_unreadable",
        f"The file cannot be read: {reason}. Close the program that holds "
        f"it, check the permissions, or copy the export into the data directory.",
    )


def _refuse_oversized(size: int, max_size: int) -> None:
    if size > max_size:
        raise PathRefused(
            "file_too_large",
            f"The file is {size} bytes, over the {max_size}-byte cap; "
            f"raise the cap via the PMM_MAX_SIGNAL_SIZE environment variable "
            f"(bytes) if this size is intentional.",
        )
