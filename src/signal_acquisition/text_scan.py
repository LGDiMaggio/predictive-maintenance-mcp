"""
Bounded, read-only scan of an untrusted delimited-text file.

ISO 13374 Block 1 (data acquisition). Inspection of a signal import reads a
file nobody has vouched for, so this scan reads only a bounded head and tail
through the one verified descriptor of ``path_safety.open_import_file``,
refuses binary, compressed and UTF-16/UTF-32 content before returning any
text, and caps every line it returns. It never writes anything.

What it reads:

- the head: at most :data:`HEAD_BYTES` bytes, plus, when the head ends inside
  a line, the rest of that line up to :data:`LINE_CAP` (a longer line is
  refused, never materialized); at most :data:`HEAD_LINES` of its lines are
  returned;
- the tail: the last :data:`TAIL_BYTES` bytes, without their first, partial
  line. When the head already reached the end of the file but holds more
  than :data:`HEAD_LINES` lines, the tail is the lines after the returned
  ones that lie within the last :data:`TAIL_BYTES` (each counted with one
  break), taken from the bytes already read, so the end of a file is seen
  whatever its size.

What it refuses, before any text is returned: known binary and compressed
signatures, UTF-16 and UTF-32 byte-order marks (a named limit of this
version; UTF-32 is checked first because its mark starts with the UTF-16
one), a NUL byte, bytes that are neither UTF-8 nor cp1252, a line over
:data:`LINE_CAP`, and a line with more than ``MAX_FIELDS`` fields under every
candidate delimiter. Refusals carry an issue code and cite file text only
escaped (``file_text``).

Line breaks follow ``line_breaks``, so the head's line numbers are the
physical line numbers every other import step reports.
"""

import codecs
import re
from dataclasses import dataclass
from typing import Optional

from ..path_safety import VerifiedFile
from .import_config import MAX_FIELDS
from .line_breaks import normalize_line_breaks

__all__ = [
    "HEAD_BYTES",
    "HEAD_LINES",
    "TAIL_BYTES",
    "LINE_CAP",
    "SIGNATURES",
    "TextScanRefused",
    "ScannedLine",
    "TextScan",
    "scan_text",
]

#: Bytes read from the start of the file (before finishing its last line).
HEAD_BYTES = 64 * 1024

#: Head lines returned.
HEAD_LINES = 300

#: Bytes read from the end of the file.
TAIL_BYTES = 16 * 1024

#: Longest line, in bytes, the import reads.
LINE_CAP = 1 << 20

#: First read size while finishing the head's last line (doubles up to
#: :data:`_EXTEND_MAX_BYTES`), so a line of ordinary length costs little.
_EXTEND_BYTES = 4 * 1024
_EXTEND_MAX_BYTES = 64 * 1024

#: Leading byte signatures of formats that are not delimited text, each with
#: its issue code and the format named in the refusal.
SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x1f\x8b", "compressed_input", "gzip"),
    (b"PK\x03\x04", "compressed_input", "zip"),
    (b"PK\x05\x06", "compressed_input", "zip"),
    (b"PK\x07\x08", "compressed_input", "zip"),
    (b"\xfd7zXZ\x00", "compressed_input", "xz"),
    (b"\x28\xb5\x2f\xfd", "compressed_input", "zstd"),
    (b"7z\xbc\xaf\x27\x1c", "compressed_input", "7-Zip"),
    (b"MATLAB 5.0 MAT-file", "binary_content", "MATLAB MAT (v5)"),
    (b"MATLAB 7.3 MAT-file", "binary_content", "MATLAB MAT (v7.3, HDF5)"),
    (b"\x89HDF\r\n\x1a\n", "binary_content", "HDF5"),
    (b"TDSm", "binary_content", "TDMS"),
    (b"RIFF", "binary_content", "RIFF (WAV)"),
    (b"\x93NUMPY", "binary_content", "NumPy NPY"),
    (b"PAR1", "binary_content", "Parquet"),
)

#: bzip2: ``BZh``, a block-size digit, then the block magic.
_BZIP2 = re.compile(rb"BZh[1-9]1AY&SY")

#: Byte-order marks this version refuses, UTF-32 before UTF-16.
_UNSUPPORTED_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "UTF-32"),
    (b"\x00\x00\xfe\xff", "UTF-32"),
    (b"\xff\xfe", "UTF-16"),
    (b"\xfe\xff", "UTF-16"),
)
_UTF8_BOM = b"\xef\xbb\xbf"

_FIELD_SPLITTERS = (
    re.compile(","),
    re.compile(";"),
    re.compile("\t"),
    re.compile("[ \t]+"),
)


class TextScanRefused(ValueError):
    """A file the scan refuses; ``code`` is an issue code of the import."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ScannedLine:
    """One complete line of the head.

    Attributes:
        number: Physical line number (1-based).
        text: The decoded line without its break. Raw file text: callers
            wrap it in ``FileText`` before it reaches any result.
    """

    number: int
    text: str


@dataclass(frozen=True)
class TextScan:
    """What a bounded scan saw.

    Attributes:
        size_bytes: File size at open time.
        bom: ``"utf-8-sig"`` when the file starts with a UTF-8 byte-order
            mark, else None.
        encoding_candidates: The encoding that decodes every byte read:
            ``("utf-8-sig",)`` or ``("utf-8",)``, which the bytes decide, or
            ``("cp1252",)``, which needs confirmation.
        newline_census: Breaks in the bytes read, by kind: ``"lf"`` (a lone
            ``\\n``), ``"crlf"`` (a run of ``\\r`` then ``\\n``) and ``"cr"``
            (every other ``\\r``).
        head_lines: The first complete lines, in order (at most
            :data:`HEAD_LINES`).
        tail_lines: The complete lines of the tail, without line numbers (the
            middle of the file is not read). Empty when the returned head
            lines reach the end of the file, or when no complete line fits in
            the tail (a last line longer than :data:`TAIL_BYTES`).
        tail_follows_head: The first tail line is the line right after the
            last head line, so no line lies unread between them and the tail
            lines can be numbered on from the head.
        head_truncated: More of the file follows the returned head lines.
        complete: Every byte of the file was read.
        bytes_read: Bytes read through the descriptor.
    """

    size_bytes: int
    bom: Optional[str]
    encoding_candidates: tuple[str, ...]
    newline_census: dict[str, int]
    head_lines: tuple[ScannedLine, ...]
    tail_lines: tuple[str, ...]
    tail_follows_head: bool
    head_truncated: bool
    complete: bool
    bytes_read: int


def _refuse_signatures(head: bytes) -> None:
    for prefix, code, label in SIGNATURES:
        if head.startswith(prefix):
            how = (
                "uncompressed CSV or TXT"
                if code == "compressed_input"
                else "CSV or TXT"
            )
            raise TextScanRefused(
                code,
                f"The file is {label} content, not delimited text; "
                f"export the signal as {how}, or load the original format with "
                f"load_signal.",
            )
    if _BZIP2.match(head):
        raise TextScanRefused(
            "compressed_input",
            "The file is bzip2 content, not delimited text; "
            "decompress it and import the CSV or TXT file.",
        )
    for mark, label in _UNSUPPORTED_BOMS:
        if head.startswith(mark):
            raise TextScanRefused(
                "unsupported_encoding",
                f"The file is {label} text, a limit of this version: "
                f"save it again as UTF-8 (with or without a byte-order mark) "
                f"or as Windows-1252, then import that copy.",
            )


def _refuse_nul(data: bytes) -> None:
    if b"\x00" in data:
        raise TextScanRefused(
            "binary_content",
            "The file contains NUL bytes, so it is binary content, "
            "not delimited text; export the signal as CSV or TXT.",
        )


def _decodes(encoding: str, *parts: bytes) -> bool:
    try:
        for part in parts:
            codecs.decode(part, encoding, errors="strict")
    except UnicodeDecodeError:
        return False
    return True


def _census(data: bytes) -> dict[str, int]:
    crlf = 0
    cr = 0
    for run in re.finditer(rb"\r+", data):
        if data[run.end() : run.end() + 1] == b"\n":
            crlf += 1  # the whole run and its \n are one break
        else:
            cr += run.end() - run.start()
    return {"lf": data.count(b"\n") - crlf, "crlf": crlf, "cr": cr}


def _refuse_wide(where: str, text: str) -> None:
    stripped = text.strip()
    if all(len(splitter.split(stripped)) > MAX_FIELDS for splitter in _FIELD_SPLITTERS):
        raise TextScanRefused(
            "too_many_fields",
            f"The file has more than {MAX_FIELDS} fields on {where} "
            f"under every delimiter the import reads; it is not a signal export "
            f"this version reads.",
        )


def _read_head(file: VerifiedFile) -> bytes:
    """The head, ending at a line break or at the end of the file."""
    head = file.read(0, HEAD_BYTES)
    _refuse_signatures(head)
    _refuse_nul(head)
    if len(head) < file.size and head[-1:] not in (b"\n", b"\r"):
        line_start = max(head.rfind(b"\n"), head.rfind(b"\r")) + 1
        step = _EXTEND_BYTES
        while len(head) < file.size:
            more = file.read(len(head), step)
            step = min(step * 2, _EXTEND_MAX_BYTES)
            if not more:
                break
            _refuse_nul(more)
            stop = re.search(rb"[\r\n]", more)
            # The line's length without its break, as the decoder counts it.
            length = (
                len(head) - line_start + (len(more) if stop is None else stop.start())
            )
            head += more if stop is None else more[: stop.end()]
            if length > LINE_CAP:
                line = normalize_line_breaks(head[:line_start]).count(b"\n") + 1
                raise TextScanRefused(
                    "line_too_long",
                    f"The file has a line longer than {LINE_CAP} bytes "
                    f"at line {line}; it is not a signal export this version "
                    f"reads.",
                )
            if stop is not None:
                break
    return _finish_break(file, head)


def _finish_break(file: VerifiedFile, head: bytes) -> bytes:
    """When *head* stops inside a run of ``\\r``, take the rest of the run
    and the ``\\n`` that may follow, so the break is not split in two (a run
    of ``\\r`` then ``\\n`` is ONE break). Bounded by :data:`LINE_CAP`."""
    taken = 0
    while head.endswith(b"\r") and len(head) < file.size and taken <= LINE_CAP:
        more = file.read(len(head), _EXTEND_BYTES)
        _refuse_nul(more)
        run = len(more) - len(more.lstrip(b"\r"))
        head += more[:run]
        taken += run
        if run < len(more):
            if more[run : run + 1] == b"\n":
                head += b"\n"
            break
    return head


def _tail_body(file: VerifiedFile, head_end: int) -> tuple[bytes, bool]:
    """The tail bytes after its first, partial line, and whether they start
    exactly where the head ended (nothing skipped)."""
    start = max(head_end, file.size - TAIL_BYTES)
    tail = file.read(start, file.size - start)
    _refuse_nul(tail)
    if start == head_end:
        return tail, True
    first_break = re.search(rb"[\r\n]", tail)
    if first_break is None:
        return b"", False
    return tail[first_break.start() :], False


def _last_lines(lines: list[str], encoding: str) -> tuple[str, ...]:
    """The last of *lines* (decoded from a head that read the whole file)
    that fit in :data:`TAIL_BYTES`, each counted with one break: the tail a
    larger file would have, without reading anything more."""
    codec = "utf-8" if encoding == "utf-8-sig" else encoding
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        size += len(line.encode(codec)) + 1
        if size > TAIL_BYTES:
            break
        kept.append(line)
    return tuple(reversed(kept))


def scan_text(file: VerifiedFile) -> TextScan:
    """Scan the head and tail of an opened import file.

    Args:
        file: The verified descriptor (``path_safety.open_import_file``).

    Returns:
        The :class:`TextScan`.

    Raises:
        TextScanRefused: For binary, compressed, UTF-16/UTF-32, NUL-bearing
            or undecodable content, a line over :data:`LINE_CAP`, or a line
            too wide under every delimiter.
    """
    head = _read_head(file)
    head_is_all = len(head) >= file.size
    tail, contiguous = (b"", True) if head_is_all else _tail_body(file, len(head))

    bom = "utf-8-sig" if head.startswith(_UTF8_BOM) else None
    if bom is not None:
        if not _decodes("utf-8", head[len(_UTF8_BOM) :], tail):
            raise TextScanRefused(
                "unsupported_encoding",
                "The file starts with a UTF-8 byte-order mark but is "
                "not valid UTF-8; save it again as UTF-8 and import that copy.",
            )
        encoding = "utf-8-sig"
    elif _decodes("utf-8", head, tail):
        encoding = "utf-8"
    elif _decodes("cp1252", head, tail):
        encoding = "cp1252"
    else:
        raise TextScanRefused(
            "unsupported_encoding",
            "The file is neither UTF-8 nor Windows-1252 text; save it "
            "again in one of them and import that copy.",
        )

    text = normalize_line_breaks(head).decode(encoding)
    pieces = text.split("\n")
    if text.endswith("\n") or not head_is_all:
        pieces = pieces[:-1]
    if not text:
        pieces = []
    head_lines = tuple(
        ScannedLine(number=index + 1, text=line)
        for index, line in enumerate(pieces[:HEAD_LINES])
    )
    for line in head_lines:
        _refuse_wide(f"line {line.number}", line.text)

    tail_lines: tuple[str, ...] = ()
    follows = False
    if tail:
        tail_encoding = "utf-8" if encoding == "utf-8-sig" else encoding
        decoded = normalize_line_breaks(tail).decode(tail_encoding)
        lines = decoded.split("\n")
        if not contiguous:
            lines = lines[1:]  # the remainder of the partial first line
        if decoded.endswith("\n"):
            lines = lines[:-1]
        tail_lines = tuple(lines)
        # The head ends at a line break, so a tail read from where it ended
        # starts on the next line; it follows the head lines when all of the
        # head's lines were returned.
        follows = contiguous and len(pieces) <= HEAD_LINES
    elif head_is_all and len(pieces) > HEAD_LINES:
        tail_lines = _last_lines(pieces[HEAD_LINES:], encoding)
        follows = len(tail_lines) == len(pieces) - HEAD_LINES
    for tail_line in tail_lines:
        _refuse_wide("a line near the end", tail_line)

    return TextScan(
        size_bytes=file.size,
        bom=bom,
        encoding_candidates=(encoding,),
        newline_census=_census(head + tail),
        head_lines=head_lines,
        tail_lines=tail_lines,
        tail_follows_head=bool(tail_lines) and follows,
        head_truncated=not head_is_all or len(pieces) > HEAD_LINES,
        complete=head_is_all or contiguous,
        bytes_read=file.bytes_read,
    )
