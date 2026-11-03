"""
One definition of a physical line break, applied in linear time.

ISO 13374 Block 1 (data acquisition). The blank-row scan of the signal store,
the bounded text scan and the strict text decoder of the signal import count
physical lines the same way, so a line number in any of their messages names
the same line:

- a run of ``\\r`` immediately followed by ``\\n`` is ONE break (Python's
  ``csv`` module on Windows writes ``\\r\\r\\n``);
- a lone ``\\n`` is one break, and so is each ``\\r`` of a run that is not
  followed by ``\\n``.

:class:`LineBreakNormalizer` rewrites every break as a single ``\\n`` while
the bytes stream through in chunks of any size, so line ``k`` of the output is
physical line ``k`` of the file. Each run of ``\\r`` is consumed in one step
(the backtracking pattern ``\\r*\\n|\\r`` used before restarted at every
position of a long run, quadratic in its length), and a run that straddles two
chunks is carried as a count, not as bytes, so memory stays bounded whatever
the run's length.

Leaf module: standard library only.
"""

import re
from collections.abc import Iterator

__all__ = ["MAX_PIECE_BYTES", "LineBreakNormalizer", "normalize_line_breaks"]

#: Largest piece of output produced for a run of breaks at once.
MAX_PIECE_BYTES = 1 << 20

#: One whole run of ``\r``. Greedy with no requirement after it, so each run
#: is matched once and never retried from inside: linear time.
_CR_RUN = re.compile(rb"\r+")


def _collapse_cr_run(match: "re.Match[bytes]") -> bytes:
    """A run followed by ``\\n`` joins that break; otherwise each ``\\r`` is one."""
    end = match.end()
    if match.string[end : end + 1] == b"\n":
        return b""
    return b"\n" * (end - match.start())


def _normalize_body(body: bytes) -> bytes:
    """Normalize a piece that does not end inside a run of ``\\r``."""
    if b"\r" not in body:
        return body
    # Two plain replaces turn \r\n and \r\r\n into \n (the common cases, at C
    # speed). A \r still right before a \n means a longer run, which the
    # single-pass run pattern collapses; with none left, every \r is a lone
    # break.
    body = body.replace(b"\r\n", b"\n").replace(b"\r\n", b"\n")
    if b"\r\n" in body:
        return _CR_RUN.sub(_collapse_cr_run, body)
    return body.replace(b"\r", b"\n")


def _newlines(count: int) -> Iterator[bytes]:
    while count > 0:
        size = min(count, MAX_PIECE_BYTES)
        yield b"\n" * size
        count -= size


class LineBreakNormalizer:
    """Rewrite every physical line break as one ``\\n``, across chunks.

    Feed the bytes in order with :meth:`feed` and call :meth:`finish` once at
    the end; the concatenated output has exactly one ``\\n`` per physical
    break and every other byte unchanged.
    """

    def __init__(self) -> None:
        self._pending_cr = 0  # trailing \r of the input so far, fate unknown

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        """Normalize the next chunk; yields zero or more output pieces."""
        if not chunk:
            return
        if self._pending_cr:
            lead = len(chunk) - len(chunk.lstrip(b"\r"))
            self._pending_cr += lead
            if lead == len(chunk):
                return
            chunk = chunk[lead:]
            if chunk[:1] != b"\n":
                yield from _newlines(self._pending_cr)
            self._pending_cr = 0
        body = chunk.rstrip(b"\r")
        self._pending_cr = len(chunk) - len(body)
        if body:
            yield _normalize_body(body)

    def finish(self) -> Iterator[bytes]:
        """Flush a final run of ``\\r``: each is a break at the end of input."""
        pending, self._pending_cr = self._pending_cr, 0
        yield from _newlines(pending)


def normalize_line_breaks(data: bytes) -> bytes:
    """The whole of *data* normalized at once (for small, complete inputs)."""
    normalizer = LineBreakNormalizer()
    return b"".join([*normalizer.feed(data), *normalizer.finish()])
