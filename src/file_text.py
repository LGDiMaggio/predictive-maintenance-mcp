"""
File-derived text as data: one escaping rule and one value type.

Every string the signal import takes from a file it reads (content, column
labels, unit tokens, preamble lines, file and folder names) reaches a tool
result, an issue, a statement or a refusal only as a :class:`FileText`:
escaped by allowlist, capped on its escaped length, and carried under the key
``file_text`` so a reader can always tell it apart from the server's own
words. File text is data, never instructions, and it is never logged.

Leaf module (standard library only), shared by the text scanner, the
decoder, the import service and the signal store, so all of them escape the
same way.

The allowlist keeps:

- printable ASCII (U+0020 to U+007E), except the backslash, which is written
  as ``\\\\`` so a backslash in the output always starts an escape;
- the printable Latin-1 characters U+00A1 to U+00FF except the soft hyphen
  U+00AD (this covers unit signs such as ``µ`` U+00B5, ``°`` and ``²``, and
  the accented letters of European channel labels);
- the Greek small letter mu U+03BC, the other way exports spell "micro".

Everything else is escaped, which is why a block list is not enough here:
controls, line and paragraph separators, NBSP, bidirectional overrides and
isolates, zero-width characters, variation selectors (U+FE0F, U+E0100), the
combining grapheme joiner U+034F, Hangul fillers (U+115F, U+3164), the
Braille blank U+2800, private-use and unassigned code points all print as
nothing or as something else. Escapes are ``\\t``, ``\\n``, ``\\r``, then
``\\xNN``, ``\\uNNNN`` or ``\\UNNNNNNNN`` with lowercase hex. Bytes are escaped
byte by byte (printable ASCII kept, everything else ``\\xNN``), because the
encoding of a field that failed to decode is unknown.
"""

import re
from dataclasses import dataclass
from typing import Union

__all__ = [
    "LABEL_CHARS",
    "PREAMBLE_LINE_CHARS",
    "TOKEN_CHARS",
    "MAX_FILE_TEXT_CHARS",
    "escape_file_text",
    "is_escaped_file_text",
    "FileText",
]

#: Cap of a column label, in escaped characters.
LABEL_CHARS = 120

#: Cap of one preamble line, in escaped characters.
PREAMBLE_LINE_CHARS = 200

#: Cap of a token cited by a refusal or a preview, in escaped characters.
TOKEN_CHARS = 32

#: Ceiling of every :class:`FileText`, whatever cap a caller asks for.
MAX_FILE_TEXT_CHARS = 1024

_SHORT_ESCAPES = {"\t": "\\t", "\n": "\\n", "\r": "\\r", "\\": "\\\\"}

#: Characters outside printable ASCII that are kept as they are.
_KEPT_NON_ASCII = frozenset(
    {chr(code) for code in range(0xA1, 0x100) if code != 0xAD} | {"\u03bc"}
)

_ESCAPED_TEXT = re.compile(
    "(?:[\\x20-\\x5b\\x5d-\\x7e"
    + "".join(sorted(_KEPT_NON_ASCII))
    + "]|\\\\(?:\\\\|[tnr]|x[0-9a-f]{2}|u[0-9a-f]{4}|U[0-9a-f]{8}))*"
)


def _escape_char(char: str) -> str:
    if char in _SHORT_ESCAPES:
        return _SHORT_ESCAPES[char]
    code = ord(char)
    if 0x20 <= code <= 0x7E or char in _KEPT_NON_ASCII:
        return char
    if code < 0x100:
        return f"\\x{code:02x}"
    if code < 0x10000:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def _escape_byte(byte: int) -> str:
    if byte == 0x5C:
        return "\\\\"
    if 0x20 <= byte <= 0x7E:
        return chr(byte)
    return f"\\x{byte:02x}"


def escape_file_text(raw: Union[str, bytes], *, cap: int) -> tuple[str, bool]:
    """Escape *raw* by the allowlist and cap the escaped form.

    Args:
        raw: File-derived text, or raw bytes of a field whose encoding is
            unknown.
        cap: Largest escaped length kept (at most
            :data:`MAX_FILE_TEXT_CHARS`). The cut never splits an escape.

    Returns:
        ``(escaped, truncated)``: the escaped text and whether anything was
        cut to respect *cap*.

    Raises:
        ValueError: If *raw* is neither ``str`` nor ``bytes``, or *cap* is
            not an int in ``1..MAX_FILE_TEXT_CHARS``.
    """
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or not (1 <= cap <= MAX_FILE_TEXT_CHARS)
    ):
        raise ValueError(
            f"cap must be an int from 1 to {MAX_FILE_TEXT_CHARS}, got {cap!r}."
        )
    if isinstance(raw, str):
        pieces = map(_escape_char, raw)
    elif isinstance(raw, (bytes, bytearray)):
        pieces = map(_escape_byte, raw)
    else:
        raise ValueError(f"File text must be str or bytes, got {type(raw).__name__}.")
    kept: list[str] = []
    length = 0
    for piece in pieces:
        if length + len(piece) > cap:
            return "".join(kept), True
        kept.append(piece)
        length += len(piece)
    return "".join(kept), False


def is_escaped_file_text(text: object) -> bool:
    """Whether *text* is well-formed escaped text.

    Every character is allowlisted, and every backslash starts an escape of
    one of the forms the escaper writes. That is the safety property: no
    character outside the allowlist can be present. It does not prove that
    *text* is the canonical escape of some input (``\\x41`` passes although
    the escaper writes ``A``), and a ``\\xNN`` escape names a byte in text
    escaped from bytes but a code point in text escaped from ``str``.
    """
    return isinstance(text, str) and _ESCAPED_TEXT.fullmatch(text) is not None


@dataclass(frozen=True, repr=False)
class FileText:
    """Text taken from a file: escaped, capped, and marked as file text.

    Build it with :meth:`of`. The constructor accepts only text that is
    already in escaped form, so raw file text cannot slip in through it.
    ``str()`` and ``repr()`` withhold the text: an f-string or a log line
    can never put file text into the server's own words by accident; read
    :attr:`file_text` where the text is meant to be shown as data.

    Attributes:
        file_text: The escaped text (allowlisted characters and escapes).
        truncated: Whether the escaped text was cut at its cap.
    """

    file_text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.file_text, str)
            and len(self.file_text) > MAX_FILE_TEXT_CHARS
        ):
            raise ValueError(
                f"FileText is capped at {MAX_FILE_TEXT_CHARS} escaped characters."
            )
        if not is_escaped_file_text(self.file_text):
            raise ValueError(
                "FileText holds escaped text only; build it with "
                "FileText.of(raw, cap=...)."
            )
        if not isinstance(self.truncated, bool):
            raise ValueError("FileText.truncated must be a bool.")

    @classmethod
    def of(cls, raw: Union[str, bytes], *, cap: int) -> "FileText":
        """Escape and cap *raw* (see :func:`escape_file_text`)."""
        text, truncated = escape_file_text(raw, cap=cap)
        return cls(file_text=text, truncated=truncated)

    def as_dict(self) -> dict[str, object]:
        """``{"file_text": ..., "truncated": ...}``, the form results carry."""
        return {"file_text": self.file_text, "truncated": self.truncated}

    def __str__(self) -> str:
        return f"<file text withheld: {len(self.file_text)} escaped characters>"

    __repr__ = __str__
