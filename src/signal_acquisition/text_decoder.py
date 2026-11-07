"""
The strict text decoder of the signal import: one decoder for validation,
import and ledger replay.

ISO 13374 Block 1 (data acquisition). It turns the data region of a
delimited-text export into float64 channels exactly as a canonical
interpretation (``import_config``) says, or refuses with the physical line,
the field and a short escaped token. Nothing is repaired, resampled or
guessed.

How a file is read (plan KTD8):

- through the one verified descriptor of ``path_safety.open_import_file``,
  in blocks of :data:`BLOCK_BYTES`, never past the size seen at open time;
  every byte is hashed in the same pass, and the file must be unchanged at
  the end (a file still being written is refused);
- every byte is checked against the confirmed encoding, strictly, and a NUL
  byte anywhere is refused;
- line breaks follow ``line_breaks``, so line numbers are the physical line
  numbers every other import step reports; a line over ``LINE_CAP`` is
  refused;
- inside the data region: a blank line followed by more data is refused (a
  declared last line makes every blank line in the region a refusal), every
  row must have the same number of fields, and a last line without a line
  break is refused as possibly cut short;
- only the selected fields are gated: their bytes must be digits, a sign, an
  exponent letter, spaces at either end and the confirmed decimal mark (a
  ``.`` in a decimal-comma field is refused). An unselected field (a dotted
  date, a comment, the empty field after a trailing delimiter) only counts
  toward the row width, so replaying one channel never depends on another
  column's content;
- a vibration channel is parsed by numpy's bytes-to-float64 conversion,
  which agrees with ``float()`` bit for bit on every token the gate lets
  through; a value that is not finite is refused; the samples are divided by
  the exact scaling divisor (``import_config.scaling_divisor``) and checked
  finite again;
- a time column is parsed exactly, without per-token Python objects, into an
  int64 mantissa and a decimal exponent per row (value = mantissa *
  10**exponent in the declared time unit); the time-base assessment decides
  its print regime from those;
- a decoded-output budget (rows times kept arrays times bytes) is checked
  per block, so a file of very short rows cannot exhaust memory.

Two modes: ``"full"`` keeps every channel's samples; ``"digest"`` (for
validation) keeps only the time columns and streams each channel's samples
digest, which equals ``import_config.samples_digest`` of the full array.
In both modes every channel also carries the mean and population standard
deviation of its scaled samples, merged block by block (three numbers per
channel, whatever the file's length), so validation can state a voltage
channel's DC level without keeping its samples.

Refusals are :class:`TextDecodeError` (a ``ValueError``) with an issue code
of the import contract, the physical line, the 0-based field and the
token's byte position in its line. Library messages, which embed raw field
text, never reach the caller; a token is cited short and escaped.

Every decoder contract ever issued stays in :data:`DECODERS`, so a recorded
measurement replays under the contract it was decoded with.
"""

import codecs
import hashlib
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Optional

import numpy as np

from ..config import get_max_signal_size
from ..file_text import LABEL_CHARS, TOKEN_CHARS, FileText, escape_file_text
from ..path_safety import VerifiedFile
from ..signal_validation import validate_finite_samples
from .import_config import MAX_FIELDS, normalize_interpretation, scaling_divisor
from .line_breaks import LineBreakNormalizer
from .text_scan import LINE_CAP

__all__ = [
    "DECODER_CONTRACT",
    "BLOCK_BYTES",
    "DECODED_BUDGET_FACTOR",
    "MAX_TOKEN_BYTES",
    "TIME_TOKEN_BYTES",
    "TIME_MAX_DIGITS",
    "TIME_MAX_EXPONENT_DIGITS",
    "parse_time_matrix",
    "TextDecodeError",
    "ChannelResult",
    "TimeColumnResult",
    "DecodeResult",
    "decode_text",
    "DECODERS",
    "get_decoder",
]

#: Contract id of this decoder; bumped only when its output can change.
DECODER_CONTRACT = "pmm-text-decoder/1"

#: Bytes read per block. The row and field gates keep a few bytes of working
#: memory per byte of block, so a much larger block would raise the peak for
#: no gain.
BLOCK_BYTES = 4 * 1024 * 1024

#: Rows whose tokens are gated and parsed together: the token matrix of one
#: slice is at most this many rows times the longest token allowed.
_SLICE_ROWS = 65536

#: Rows checked together for blank lines, width and fields (about a hundred
#: bytes of index arrays per row).
_BATCH_ROWS = 262144

#: The decoded-output budget is this many times ``PMM_MAX_SIGNAL_SIZE``.
DECODED_BUDGET_FACTOR = 2

#: Longest token of a vibration channel, in bytes.
MAX_TOKEN_BYTES = 64

#: Longest token of a time column, in bytes.
TIME_TOKEN_BYTES = 32

#: Most significant digits of a time stamp (an int64 mantissa holds 18).
TIME_MAX_DIGITS = 18

#: Most digits of a time stamp's exponent.
TIME_MAX_EXPONENT_DIGITS = 3

_DELIMITER_BYTES = {",": ord(","), ";": ord(";"), "tab": ord("\t")}

_SPACE = ord(" ")
_TAB = ord("\t")
_NEWLINE = ord("\n")
_DOT = ord(".")
_COMMA = ord(",")
_MINUS = ord("-")
_UTF8_BOM = b"\xef\xbb\xbf"

#: Bytes that make a line hold data (anything but ASCII whitespace).
_DATA_BYTE = np.ones(256, dtype=bool)
_DATA_BYTE[list(b" \t\x0b\x0c\n")] = False


def _merge_moments(
    moments: tuple[int, float, float, float], block: np.ndarray
) -> tuple[int, float, float, float]:
    """``(count, shift, mean, sum of squared deviations)`` after one more
    block: Chan's pairwise update on the samples minus *shift* (the
    channel's first sample), so the moments stay exact to rounding without
    keeping the samples, a constant channel has exactly zero spread, and the
    same blocks always give the same result. The channel's mean is
    ``shift + mean``."""
    count, shift, mean, squares = moments
    size = int(block.size)
    if size == 0:
        return moments
    if count == 0:
        shift = float(block[0])
    shifted = block - shift
    block_mean = float(shifted.mean())
    block_squares = float(np.square(shifted - block_mean).sum())
    total = count + size
    delta = block_mean - mean
    return (
        total,
        shift,
        mean + delta * size / total,
        squares + block_squares + delta * delta * count * size / total,
    )


def _alphabet(decimal_mark: str) -> np.ndarray:
    allowed = np.zeros(256, dtype=bool)
    allowed[0] = True  # padding of the token matrix
    allowed[list(b"0123456789+-eE ")] = True
    allowed[_DOT if decimal_mark == "." else _COMMA] = True
    return allowed


# A time stamp, [+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?, as an automaton over
# character classes, run on all rows at once (no per-token Python object).
_PAD, _DIGIT, _POINT, _SIGN, _EXP, _OTHER = range(6)
(
    _START,
    _SIGNED,
    _INT,
    _INT_DOT,
    _BARE_DOT,
    _FRAC,
    _E,
    _E_SIGN,
    _E_DIGITS,
    _END,
    _ERROR,
) = range(11)
_TRANSITIONS = np.full((11, 6), _ERROR, dtype=np.int8)
for _state, _cls, _next in (
    (_START, _DIGIT, _INT),
    (_START, _POINT, _BARE_DOT),
    (_START, _SIGN, _SIGNED),
    (_SIGNED, _DIGIT, _INT),
    (_SIGNED, _POINT, _BARE_DOT),
    (_INT, _PAD, _END),
    (_INT, _DIGIT, _INT),
    (_INT, _POINT, _INT_DOT),
    (_INT, _EXP, _E),
    (_INT_DOT, _PAD, _END),
    (_INT_DOT, _DIGIT, _FRAC),
    (_INT_DOT, _EXP, _E),
    (_BARE_DOT, _DIGIT, _FRAC),
    (_FRAC, _PAD, _END),
    (_FRAC, _DIGIT, _FRAC),
    (_FRAC, _EXP, _E),
    (_E, _DIGIT, _E_DIGITS),
    (_E, _SIGN, _E_SIGN),
    (_E_SIGN, _DIGIT, _E_DIGITS),
    (_E_DIGITS, _PAD, _END),
    (_E_DIGITS, _DIGIT, _E_DIGITS),
    (_END, _PAD, _END),
):
    _TRANSITIONS[_state, _cls] = _next
_CLASS = np.full(256, _OTHER, dtype=np.int8)
_CLASS[0] = _PAD
_CLASS[list(b"0123456789")] = _DIGIT
_CLASS[_DOT] = _POINT
_CLASS[list(b"+-")] = _SIGN
_CLASS[list(b"eE")] = _EXP


def parse_time_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse time stamps exactly, one per row of a token matrix.

    The one time-stamp grammar of the import, shared by the decoder and the
    time-base assessment (``signal_import.timebase``), so a preview and a
    full decode read a stamp the same way.

    Args:
        matrix: uint8 matrix, one token per row, left-aligned and padded with
            zero bytes, with ``.`` as the decimal mark and no spaces.

    Returns:
        ``(mantissa, exponent, valid)``: the int64 digits and int16 decimal
        exponent of each stamp (value = mantissa * 10**exponent), and whether
        the row is a stamp of at most :data:`TIME_MAX_DIGITS` significant
        digits and :data:`TIME_MAX_EXPONENT_DIGITS` exponent digits. The
        first two are meaningless where ``valid`` is false.
    """
    count, span = matrix.shape
    classes = _CLASS[matrix]
    state = np.full(count, _START, dtype=np.int8)
    mantissa = np.zeros(count, dtype=np.int64)
    significant = np.zeros(count, dtype=np.int64)
    fraction = np.zeros(count, dtype=np.int64)
    exponent = np.zeros(count, dtype=np.int64)
    exponent_digits = np.zeros(count, dtype=np.int64)
    negative = np.zeros(count, dtype=bool)
    negative_exponent = np.zeros(count, dtype=bool)
    for j in range(span):
        cls = classes[:, j]
        digit = matrix[:, j].astype(np.int64) - ord("0")
        following = _TRANSITIONS[state, cls]
        mantissa_digit = (following == _INT) | (following == _FRAC)
        counted = mantissa_digit & ((significant > 0) | (digit > 0))
        significant += counted
        fits = mantissa_digit & (significant <= TIME_MAX_DIGITS)
        mantissa = np.where(fits, mantissa * 10 + digit, mantissa)
        fraction += following == _FRAC
        negative |= (state == _START) & (matrix[:, j] == _MINUS)
        negative_exponent |= (state == _E) & (matrix[:, j] == _MINUS)
        in_exponent = following == _E_DIGITS
        exponent = np.where(in_exponent, exponent * 10 + digit, exponent)
        exponent_digits += in_exponent
        state = following
    state = _TRANSITIONS[state, _PAD]
    valid = (
        (state == _END)
        & (significant <= TIME_MAX_DIGITS)
        & (exponent_digits <= TIME_MAX_EXPONENT_DIGITS)
    )
    mantissa = np.where(negative, -mantissa, mantissa)
    scale = np.where(negative_exponent, -exponent, exponent) - fraction
    return mantissa, scale.astype(np.int16), valid


class TextDecodeError(ValueError):
    """A refusal of the strict decoder.

    Attributes:
        code: Issue code of the import contract.
        line: Physical line (1-based), when the refusal has one.
        field: 0-based field index, when it concerns one field.
        position: Byte position of the token in its line, when it has one.
        token: The offending token, short and escaped, when it has one.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: Optional[int] = None,
        field: Optional[int] = None,
        position: Optional[int] = None,
        token: Optional[FileText] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line = line
        self.field = field
        self.position = position
        self.token = token


@dataclass(frozen=True)
class ChannelResult:
    """One decoded vibration channel.

    Attributes:
        column: Physical column index.
        samples: The float64 samples after scaling (None in digest mode).
        samples_digest: SHA-256 of the samples as little-endian float64.
        sample_count: Number of samples.
        mean: Mean of the scaled samples, merged block by block.
        std: Population standard deviation of the scaled samples, merged
            block by block (Chan's pairwise update), in both modes.
    """

    column: int
    samples: Optional[np.ndarray]
    samples_digest: str
    sample_count: int
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass(frozen=True)
class TimeColumnResult:
    """One time column, exactly as printed.

    Each stamp is ``mantissa[i] * 10**exponent[i]`` in ``time_unit``.

    Attributes:
        column: Physical column index.
        time_unit: The declared unit (``s``, ``ms``, ``us``, ``ns``).
        mantissa: int64 digits of each stamp, signed.
        exponent: int16 decimal exponent of each stamp's last digit.
    """

    column: int
    time_unit: str
    mantissa: np.ndarray
    exponent: np.ndarray


@dataclass(frozen=True)
class DecodeResult:
    """What one decode produced.

    Attributes:
        contract: The decoder contract (:data:`DECODER_CONTRACT`).
        source_sha256: SHA-256 of every byte of the file.
        size_bytes: The file size.
        first_line: Physical line of the first decoded row.
        last_line: Physical line of the last decoded row.
        row_count: Number of decoded rows (samples per channel).
        field_count: Fields per row.
        channels: The vibration channels, by column.
        time_columns: The time columns, by column.
    """

    contract: str
    source_sha256: str
    size_bytes: int
    first_line: int
    last_line: int
    row_count: int
    field_count: int
    channels: tuple[ChannelResult, ...]
    time_columns: tuple[TimeColumnResult, ...]


DecodeMode = Literal["full", "digest"]


def _field_words(field: int) -> str:
    position = field + 1
    if 10 <= position % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    return f"field {field} (the {position}{suffix} from the left)"


def _parses(text: bytes) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


_THOUSANDS = {
    ".": re.compile(rb"[+-]?\d{1,3}(,\d{3})+(\.\d*)?"),
    ",": re.compile(rb"[+-]?\d{1,3}(\.\d{3})+(,\d*)?"),
}
_NON_FINITE_WORDS = {b"nan", b"inf", b"infinity", b"na", b"n/a", b"#n/a", b"null"}


def _token_problem(token: bytes, decimal_mark: str, *, time: bool) -> tuple[str, str]:
    """The issue code and the problem of a token the gates refused."""
    stripped = token.strip(b" ")
    words = stripped.lower().lstrip(b"+-")
    plain = "time_grammar" if time else "non_numeric_token"
    if not stripped:
        return plain, "is an empty field"
    if b'"' in stripped or b"'" in stripped:
        return "quoted_data", "is quoted; numbers are read unquoted"
    if _THOUSANDS[decimal_mark].fullmatch(stripped):
        return "thousands_separator", "uses a thousands separator, which is not read"
    if words in _NON_FINITE_WORDS:
        return (
            "time_grammar" if time else "non_finite_token"
        ), "is not a finite number"
    if decimal_mark == "," and b"." in stripped:
        return plain, "has a '.' in a decimal-comma field"
    return plain, "is not a plain decimal number"


@dataclass(frozen=True)
class _Plan:
    delimiter: Optional[int]  # None: a run of spaces and tabs
    decimal_mark: str
    first: int
    last: Optional[int]
    header_lines: tuple[int, ...]
    channels: tuple[int, ...]
    divisors: Mapping[int, Optional[float]]
    time_columns: tuple[tuple[int, str], ...]
    encoding: str

    @property
    def selected(self) -> tuple[int, ...]:
        return (*self.channels, *(column for column, _ in self.time_columns))


def _plan(interpretation: Mapping[str, Any]) -> _Plan:
    canonical = normalize_interpretation(interpretation)
    dialect = canonical["dialect"]
    divisors: dict[int, Optional[float]] = {}
    for channel in canonical["channels"]:
        scaling = channel["scaling"]
        divisors[channel["column"]] = (
            None if scaling is None else float(scaling_divisor(scaling))
        )
    return _Plan(
        delimiter=_DELIMITER_BYTES.get(dialect["delimiter"]),
        decimal_mark=dialect["decimal"],
        first=canonical["data_lines"]["first"],
        last=canonical["data_lines"]["last"],
        header_lines=tuple(
            line
            for line in (dialect["header_line"], dialect["units_line"])
            if line is not None
        ),
        channels=tuple(divisors),
        divisors=MappingProxyType(divisors),
        time_columns=tuple(
            (entry["column"], entry["time_unit"]) for entry in canonical["time_columns"]
        ),
        encoding=dialect["encoding"],
    )


@dataclass
class _Rows:
    """The data rows of one block: bytes and the span of every field."""

    segment: np.ndarray  # the rows' bytes, each ending with \n
    line0: int  # physical line of the first row
    starts: np.ndarray  # row starts in ``segment``
    field_start: np.ndarray  # (rows, width)
    field_end: np.ndarray  # (rows, width)


class _Decoder:
    """State of one decode across blocks."""

    def __init__(self, plan: _Plan, mode: DecodeMode, name: str) -> None:
        self.plan = plan
        self.mode = mode
        self.name = name
        self.alphabet = _alphabet(plan.decimal_mark)
        self.width: Optional[int] = None
        self.rows = 0
        self.first_row_line: Optional[int] = None
        self.last_row_line: Optional[int] = None
        self.pending_blank: Optional[int] = None
        self.header_texts: set[bytes] = set()
        self.next_line = 1
        self.partial = b""
        self.digests = {column: hashlib.sha256() for column in plan.channels}
        # Per channel: sample count, shift (the first sample), running mean of
        # the shifted samples, sum of squared deviations.
        self.moments: dict[int, tuple[int, float, float, float]] = {
            c: (0, 0.0, 0.0, 0.0) for c in plan.channels
        }
        self.blocks: dict[int, list[np.ndarray]] = {c: [] for c in plan.channels}
        self.time_blocks: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            column: [] for column, _ in plan.time_columns
        }
        self.budget = DECODED_BUDGET_FACTOR * get_max_signal_size()
        self.kept_bytes = 0

    def refuse(
        self,
        code: str,
        problem: str,
        remedy: str,
        *,
        line: Optional[int] = None,
        field: Optional[int] = None,
        position: Optional[int] = None,
        token: Optional[bytes] = None,
    ) -> TextDecodeError:
        where = [f"File {self.name}"]
        if line is not None:
            where.append(f"line {line}")
        if field is not None:
            where.append(_field_words(field))
        cited = None
        text = ", ".join(where) + ": "
        if token is not None:
            cited = FileText.of(token, cap=TOKEN_CHARS)
            # The token's own quotes are escaped, so it cannot close the
            # quotation and speak in the server's voice.
            quoted = cited.file_text.replace("'", "\\'")
            shown = f"'{quoted}{'...' if cited.truncated else ''}'"
            text += f"the value {shown} {problem}"
        else:
            text += problem
        return TextDecodeError(
            code,
            f"{text} \N{EM DASH} {remedy}.",
            line=line,
            field=field,
            position=position,
            token=cited,
        )

    # -- lines --------------------------------------------------------------

    def feed(self, piece: bytes) -> None:
        data = self.partial + piece if self.partial else piece
        cut = data.rfind(b"\n")
        if cut < 0:
            self.partial = data
        else:
            self.partial = data[cut + 1 :]
            self.lines(data[: cut + 1], unterminated=False)
        if len(self.partial) > LINE_CAP:
            raise self.too_long(self.next_line)

    def finish(self) -> None:
        if self.partial:
            data, self.partial = self.partial + b"\n", b""
            self.lines(data, unterminated=True)

    def too_long(self, line: int) -> TextDecodeError:
        return self.refuse(
            "line_too_long",
            f"is longer than {LINE_CAP} bytes",
            "this is not a signal export the import reads",
            line=line,
        )

    def lines(self, batch: bytes, *, unterminated: bool) -> None:
        plan = self.plan
        line0 = self.next_line
        array = np.frombuffer(batch, dtype=np.uint8)
        ends = np.flatnonzero(array == _NEWLINE)
        count = ends.size
        self.next_line += count
        starts = np.empty(count, dtype=np.int64)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        for header in plan.header_lines:
            if line0 <= header < line0 + count:
                index = header - line0
                self.header_texts.add(batch[starts[index] : ends[index]].strip())
        too_long = np.flatnonzero(ends - starts > LINE_CAP)
        stop = int(too_long[0]) if too_long.size else count
        lo = max(plan.first - line0, 0)
        hi = count if plan.last is None else min(count, plan.last - line0 + 1)
        # The lines before an over-long one are checked first, so a refusal
        # always names the first offending line. They go in batches of at
        # most _BATCH_ROWS rows: the row and field arrays stay bounded even
        # when a block holds millions of very short rows.
        end = min(hi, stop)
        for first in range(lo, end, _BATCH_ROWS):
            last = min(first + _BATCH_ROWS, end)
            self.region(
                array,
                starts[first:last],
                ends[first:last],
                line0 + first,
                unterminated=unterminated and last == count,
            )
        if stop < count:
            raise self.too_long(line0 + stop)

    def region(
        self,
        array: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        line0: int,
        *,
        unterminated: bool,
    ) -> None:
        offset = int(starts[0])
        has_data = np.logical_or.reduceat(
            _DATA_BYTE[array[offset : int(ends[-1]) + 1]], starts - offset
        )
        blank = np.flatnonzero(~has_data)
        data = np.flatnonzero(has_data)
        if self.pending_blank is not None and data.size:
            raise self.blank(self.pending_blank, declared=False)
        # The first structural problem of these lines, raised only once the
        # rows before it have passed every check.
        stop: Optional[int] = None
        problem: Optional[TextDecodeError] = None
        if blank.size and (
            self.plan.last is not None or (data.size and blank[0] < data[-1])
        ):
            stop = int(blank[0])
            problem = self.blank(line0 + stop, declared=self.plan.last is not None)
        elif data.size and unterminated and int(data[-1]) == ends.size - 1:
            stop = int(data[-1])
            problem = self.refuse(
                "truncated_last_line",
                "is the last line and has no line break, so it may be cut short",
                "the export may still be in progress; when it is complete, end the "
                "file with a line break, or end data_lines before that line",
                line=line0 + stop,
            )
        rows = (int(data[-1]) + 1 if data.size else 0) if stop is None else stop
        if rows:
            segment = array[offset : int(ends[rows - 1]) + 1]
            self.data_rows(segment, starts[:rows] - offset, ends[:rows] - offset, line0)
        if problem is not None:
            raise problem
        if blank.size and self.pending_blank is None:
            self.pending_blank = line0 + int(blank[0])

    def blank(self, line: int, *, declared: bool) -> TextDecodeError:
        remedy = (
            "a blank line in the declared data region is a missing sample; end "
            "data_lines before it"
            if declared
            else "a blank line followed by more data is a missing sample, so the "
            "file is refused rather than read with a shifted time axis"
        )
        return self.refuse(
            "blank_line_in_data", "is blank inside the data region", remedy, line=line
        )

    # -- rows ---------------------------------------------------------------

    def data_rows(
        self, segment: np.ndarray, starts: np.ndarray, ends: np.ndarray, line0: int
    ) -> None:
        plan = self.plan
        if plan.delimiter is not None:
            is_delimiter = segment == plan.delimiter
            fields = np.add.reduceat(is_delimiter, starts, dtype=np.int64) + 1
        else:
            content = (segment != _SPACE) & (segment != _TAB) & (segment != _NEWLINE)
            begins = content.copy()
            begins[1:] &= ~content[:-1]
            fields = np.add.reduceat(begins, starts, dtype=np.int64)
        if self.width is None:
            self.first_row_line = line0
            self.set_width(int(fields[0]), line0)
        width = self.width
        assert width is not None
        ragged = np.flatnonzero(fields != width)
        count = int(ragged[0]) if ragged.size else starts.size
        if count:
            end = int(ends[count - 1]) + 1
            field_start = np.empty((count, width), dtype=np.int64)
            field_end = np.empty((count, width), dtype=np.int64)
            if plan.delimiter is not None:
                positions = np.flatnonzero(is_delimiter[:end]).reshape(count, width - 1)
                field_start[:, 0] = starts[:count]
                field_start[:, 1:] = positions + 1
                field_end[:, :-1] = positions
                field_end[:, -1] = ends[:count]
            else:
                stops = content[:end].copy()
                stops[:-1] &= ~content[1:end]
                field_start[:] = np.flatnonzero(begins[:end]).reshape(count, width)
                field_end[:] = np.flatnonzero(stops).reshape(count, width) + 1
            rows = _Rows(segment, line0, starts[:count], field_start, field_end)
            if self.rows == 0:
                self.check_trailing_column(rows)
            self.charge(count)
            self.columns(rows)
        if ragged.size:
            index = int(ragged[0])
            raise self.refuse(
                "ragged_row",
                f"has {int(fields[index])} fields where the first data row has {width}",
                "every data row must have the same fields; check the delimiter and "
                "the data line range",
                line=line0 + index,
            )

    def columns(self, rows: _Rows) -> None:
        """Decode every selected column of *rows*; when several refuse, the
        first line (then the first field) is the one reported."""
        plan = self.plan
        refusals: list[TextDecodeError] = []
        values: dict[int, np.ndarray] = {}
        times: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for column in plan.channels:
            try:
                values[column] = self.channel_values(rows, column)
            except TextDecodeError as refusal:
                refusals.append(refusal)
        for column, _unit in plan.time_columns:
            try:
                times[column] = self.time_values(rows, column)
            except TextDecodeError as refusal:
                refusals.append(refusal)
        if refusals:
            raise min(refusals, key=lambda r: (r.line or 0, r.field or 0))
        for column, samples in values.items():
            self.digests[column].update(samples.data)
            self.moments[column] = _merge_moments(self.moments[column], samples)
            if self.mode == "full":
                self.blocks[column].append(samples)
        for column, pair in times.items():
            self.time_blocks[column].append(pair)
        self.rows += rows.starts.size
        self.last_row_line = rows.line0 + rows.starts.size - 1

    def set_width(self, width: int, line: int) -> None:
        if width > MAX_FIELDS:
            raise self.refuse(
                "too_many_fields",
                f"has {width} fields, more than the {MAX_FIELDS} the import reads",
                "this is not a signal export the import reads",
                line=line,
            )
        missing = [column for column in self.plan.selected if column >= width]
        if missing:
            raise self.refuse(
                "field_count_mismatch",
                f"has {width} fields, so the selected column {missing[0]} does not "
                f"exist",
                "select columns by their 0-based index among the data fields",
                line=line,
            )
        self.width = width

    def check_trailing_column(self, rows: _Rows) -> None:
        width = self.width
        if self.plan.delimiter is None or width is None or width < 2:
            return
        last = width - 1
        if (
            last in self.plan.selected
            and rows.field_start[0, last] == rows.field_end[0, last]
        ):
            raise self.refuse(
                "trailing_delimiter_column",
                f"ends with a delimiter, so its column {last} is empty",
                "the empty column after a trailing delimiter holds no data; select "
                "the columns before it",
                line=rows.line0,
                field=last,
            )

    def charge(self, count: int) -> None:
        per_row = 10 * len(self.plan.time_columns)
        if self.mode == "full":
            per_row += 8 * len(self.plan.channels)
        self.kept_bytes += count * per_row
        if self.kept_bytes > self.budget:
            raise self.refuse(
                "file_too_large",
                f"decodes to more than the {self.budget}-byte output budget "
                f"({DECODED_BUDGET_FACTOR} times PMM_MAX_SIGNAL_SIZE)",
                "select fewer channels or a shorter data line range, or raise "
                "PMM_MAX_SIGNAL_SIZE if this size is intentional",
            )

    # -- tokens -------------------------------------------------------------

    def spans(
        self, rows: _Rows, column: int, *, time: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Start and end of the column's token in every row; a time stamp's
        spaces at either end are left out."""
        segment = rows.segment
        start = rows.field_start[:, column].copy()
        end = rows.field_end[:, column].copy()
        if time:
            last_byte = segment.size - 1
            for _ in range(TIME_TOKEN_BYTES + 1):
                lead = (start < end) & (segment[np.minimum(start, last_byte)] == _SPACE)
                trail = (start < end) & (segment[np.maximum(end - 1, 0)] == _SPACE)
                if not (lead.any() or trail.any()):
                    break
                start += lead
                end -= trail & ~lead
        return start, end

    def gated_slices(
        self, rows: _Rows, column: int, cap: int, *, time: bool
    ) -> Iterator[tuple[int, np.ndarray, Optional[TextDecodeError]]]:
        """The column's tokens in slices of at most :data:`_SLICE_ROWS` rows,
        each as a uint8 matrix (decimal mark as ``.``) of the rows before the
        slice's first gate refusal, with that refusal (or None). Memory stays
        bounded whatever the rows' lengths."""
        segment = rows.segment
        start_all, end_all = self.spans(rows, column, time=time)
        last_byte = segment.size - 1
        for lo in range(0, start_all.size, _SLICE_ROWS):
            start = start_all[lo : lo + _SLICE_ROWS]
            lengths = end_all[lo : lo + _SLICE_ROWS] - start
            long_rows = np.flatnonzero(lengths > cap)
            limit = int(long_rows[0]) if long_rows.size else start.size
            span = max(int(lengths[:limit].max()) if limit else 1, 1)
            matrix = np.zeros((limit, span), dtype=np.uint8)
            for offset in range(span):
                matrix[:, offset] = np.where(
                    lengths[:limit] > offset,
                    segment[np.minimum(start[:limit] + offset, last_byte)],
                    0,
                )
            refusal: Optional[TextDecodeError] = None
            refused = np.flatnonzero(~self.alphabet[matrix].all(axis=1))
            if refused.size:
                limit = int(refused[0])
                matrix = matrix[:limit]
                refusal = self.token_refusal(rows, column, lo + limit, time=time)
            elif long_rows.size:
                row = lo + limit
                refusal = self.refuse(
                    "time_grammar" if time else "non_numeric_token",
                    f"is longer than {cap} bytes",
                    "this is not a number the import reads",
                    line=rows.line0 + row,
                    field=column,
                    position=int(start_all[row] - rows.starts[row]),
                    token=segment[start_all[row] : end_all[row]].tobytes(),
                )
            if self.plan.decimal_mark == ",":
                matrix[matrix == _COMMA] = _DOT
            yield lo, matrix, refusal

    def token_refusal(
        self, rows: _Rows, column: int, row: int, *, time: bool
    ) -> TextDecodeError:
        segment = rows.segment
        line_start = int(rows.starts[row])
        line_end = (
            int(rows.starts[row + 1]) - 1
            if row + 1 < rows.starts.size
            else int(rows.field_end[row, -1])
        )
        if segment[line_start:line_end].tobytes().strip() in self.header_texts:
            return self.refuse(
                "repeated_header_block",
                "repeats the header inside the data region",
                "end data_lines before the repeated header, or import each block "
                "separately",
                line=rows.line0 + row,
            )
        start = int(rows.field_start[row, column])
        token = segment[start : int(rows.field_end[row, column])].tobytes()
        code, problem = _token_problem(token, self.plan.decimal_mark, time=time)
        return self.refuse(
            code,
            problem,
            "the import reads plain decimal numbers with the confirmed decimal mark "
            "and never repairs a value",
            line=rows.line0 + row,
            field=column,
            position=start - line_start,
            token=token,
        )

    def channel_values(self, rows: _Rows, column: int) -> np.ndarray:
        """The column's samples, parsed, scaled and checked finite."""
        parts = []
        for lo, matrix, refusal in self.gated_slices(
            rows, column, MAX_TOKEN_BYTES, time=False
        ):
            parts.append(self.parse_floats(rows, column, lo, matrix))
            if refusal is not None:
                raise refusal
        values = np.concatenate(parts) if len(parts) > 1 else parts[0]
        divisor = self.plan.divisors[column]
        if divisor is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                values = values / divisor  # checked finite right below
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                raise self.refuse(
                    "non_finite_token",
                    "is not finite once divided by the scaling divisor",
                    "check the sensitivity and the source unit",
                    line=rows.line0 + int(bad[0]),
                    field=column,
                )
        values = np.ascontiguousarray(values, dtype="<f8")
        # The shared finite-sample rule (R47); the line-precise checks above
        # already refused anything it would.
        return validate_finite_samples(values, source=f"File {self.name}")

    def parse_floats(
        self, rows: _Rows, column: int, lo: int, matrix: np.ndarray
    ) -> np.ndarray:
        if not matrix.shape[0]:
            return np.empty(0, dtype=np.float64)
        texts = matrix.view(f"S{matrix.shape[1]}").ravel()
        unparsable: Optional[int] = None
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                values = texts.astype(np.float64)
        except ValueError:
            unparsable = next(
                row for row, text in enumerate(texts) if not _parses(text)
            )
            # A non-finite value before the unparsable one comes first.
            with np.errstate(over="ignore", invalid="ignore"):
                values = texts[:unparsable].astype(np.float64)
        bad = np.flatnonzero(~np.isfinite(values))
        if unparsable is not None and not bad.size:
            raise self.token_refusal(rows, column, lo + unparsable, time=False)
        if bad.size:
            row = lo + int(bad[0])
            start = int(rows.field_start[row, column])
            raise self.refuse(
                "non_finite_token",
                "is not a finite number",
                "the import reads finite decimal numbers only and never repairs a "
                "value",
                line=rows.line0 + row,
                field=column,
                position=start - int(rows.starts[row]),
                token=rows.segment[start : int(rows.field_end[row, column])].tobytes(),
            )
        return values

    def time_values(self, rows: _Rows, column: int) -> tuple[np.ndarray, np.ndarray]:
        """The column's stamps as int64 mantissas and int16 exponents."""
        mantissas, exponents = [], []
        for lo, matrix, refusal in self.gated_slices(
            rows, column, TIME_TOKEN_BYTES, time=True
        ):
            mantissa, exponent = self.parse_time(rows, column, lo, matrix)
            mantissas.append(mantissa)
            exponents.append(exponent)
            if refusal is not None:
                raise refusal
        return np.concatenate(mantissas), np.concatenate(exponents)

    def parse_time(
        self, rows: _Rows, column: int, lo: int, matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact decimal parse: int64 mantissa and int16 exponent per row."""
        mantissa, exponent, valid = parse_time_matrix(matrix)
        refused = np.flatnonzero(~valid)
        if refused.size:
            row = lo + int(refused[0])
            start = int(rows.field_start[row, column])
            token = rows.segment[start : int(rows.field_end[row, column])].tobytes()
            code, problem = _token_problem(token, self.plan.decimal_mark, time=True)
            if problem == "is not a plain decimal number":
                problem = (
                    f"is not a time stamp the import reads: a decimal number with at "
                    f"most {TIME_MAX_DIGITS} significant digits and an exponent of at "
                    f"most {TIME_MAX_EXPONENT_DIGITS} digits"
                )
            raise self.refuse(
                code,
                problem,
                "a time column holds plain decimal stamps; leave the column "
                "unassessed, with a reason, if it is not a time stamp",
                line=rows.line0 + row,
                field=column,
                position=start - int(rows.starts[row]),
                token=token,
            )
        return mantissa, exponent

    # -- result -------------------------------------------------------------

    def result(self, digest: str, size: int) -> DecodeResult:
        plan = self.plan
        lines = self.next_line - 1
        if plan.last is not None and plan.last > lines:
            raise self.refuse(
                "no_numeric_region",
                f"has {lines} lines, so data_lines.last {plan.last} is past its end",
                "set data_lines.last to a line of the file, or leave it unset to read "
                "to the end",
            )
        if self.rows == 0:
            problem = (
                f"has {lines} lines, so data_lines.first {plan.first} is past its end"
                if plan.first > lines
                else "holds no data row in its data region"
            )
            raise self.refuse(
                "no_numeric_region", problem, "set data_lines to the data rows"
            )
        assert self.first_row_line is not None and self.last_row_line is not None
        assert self.width is not None
        channels = []
        for column in plan.channels:
            # One channel at a time, dropping its blocks right after, so the
            # peak holds the kept samples plus one channel, not two copies.
            blocks = self.blocks.pop(column)
            samples = np.concatenate(blocks) if self.mode == "full" else None
            blocks.clear()
            count, shift, mean, squares = self.moments[column]
            channels.append(
                ChannelResult(
                    column=column,
                    samples=samples,
                    samples_digest=self.digests[column].hexdigest(),
                    sample_count=self.rows,
                    mean=shift + mean if count else None,
                    std=math.sqrt(squares / count) if count else None,
                )
            )
        times = tuple(
            TimeColumnResult(
                column=column,
                time_unit=unit,
                mantissa=np.concatenate([m for m, _ in self.time_blocks[column]]),
                exponent=np.concatenate([e for _, e in self.time_blocks[column]]),
            )
            for column, unit in plan.time_columns
        )
        return DecodeResult(
            contract=DECODER_CONTRACT,
            source_sha256=digest,
            size_bytes=size,
            first_line=self.first_row_line,
            last_line=self.last_row_line,
            row_count=self.rows,
            field_count=self.width,
            channels=tuple(channels),
            time_columns=times,
        )


def decode_text(
    file: VerifiedFile,
    interpretation: Mapping[str, Any],
    *,
    mode: DecodeMode = "full",
) -> DecodeResult:
    """Decode the data region of an opened file as *interpretation* says.

    Args:
        file: The verified descriptor (``path_safety.open_import_file``).
        interpretation: The import interpretation (normalized here).
        mode: ``"full"`` keeps the samples; ``"digest"`` keeps only the
            time columns and the samples digests.

    Returns:
        The :class:`DecodeResult`.

    Raises:
        TextDecodeError: For anything the strict reading refuses.
        ValueError: From the normalizer, or for an unknown *mode*.
        path_safety.PathRefused: When the file changed while it was read.
    """
    if mode not in ("full", "digest"):
        raise ValueError(f"mode must be 'full' or 'digest', got {mode!r}.")
    plan = _plan(interpretation)
    shown, truncated = escape_file_text(file.path.name, cap=LABEL_CHARS)
    decoder = _Decoder(plan, mode, f"'{shown}{'...' if truncated else ''}'")
    hasher = hashlib.sha256()
    checker = codecs.getincrementaldecoder(plan.encoding)(errors="strict")
    normalizer = LineBreakNormalizer()
    first = True
    for chunk in file.iter_chunks(BLOCK_BYTES):
        hasher.update(chunk)
        if b"\x00" in chunk:
            raise decoder.refuse(
                "binary_content",
                "contains NUL bytes, so it is binary content",
                "export the signal as CSV or TXT",
            )
        try:
            checker.decode(chunk, final=False)
        except UnicodeDecodeError:
            raise decoder.refuse(
                "unsupported_encoding",
                f"has bytes that are not valid {plan.encoding} text",
                "confirm the encoding the file was saved in",
            ) from None
        if first and plan.encoding == "utf-8-sig" and chunk.startswith(_UTF8_BOM):
            chunk = chunk[len(_UTF8_BOM) :]
        first = False
        for piece in normalizer.feed(chunk):
            decoder.feed(piece)
    try:
        checker.decode(b"", final=True)
    except UnicodeDecodeError:
        raise decoder.refuse(
            "unsupported_encoding",
            f"ends inside a {plan.encoding} character",
            "confirm the encoding the file was saved in",
        ) from None
    for piece in normalizer.finish():
        decoder.feed(piece)
    decoder.finish()
    return decoder.result(hasher.hexdigest(), file.size)


#: Every decoder contract ever issued, by id (never shrinks).
DECODERS: Mapping[str, Callable[..., DecodeResult]] = MappingProxyType(
    {DECODER_CONTRACT: decode_text}
)


def get_decoder(contract: str) -> Callable[..., DecodeResult]:
    """The decoder of a recorded contract id.

    Raises:
        ValueError: For a contract this build does not know.
    """
    try:
        return DECODERS[contract]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown text decoder contract {contract!r}; this build replays "
            f"{sorted(DECODERS)}. Re-import the file with this build."
        ) from None
