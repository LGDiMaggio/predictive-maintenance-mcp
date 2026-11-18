"""
Rows the configured data range leaves out: one row classifier and a census.

ISO 13374 Block 1 (data acquisition). An import that leaves out rows which
could be part of the acquisition must never pass as complete. This module
holds the one rule both import layers use to tell such rows from the lines
around them, and the census validation runs over every line above the
configured data start and, when a last data line is stated, every line after
it.

The row classifier reads a line under a reading (delimiter and decimal mark)
and the shape of the data region (field count, numeric columns, the classes
of the other columns) and files it as:

- ``data``: the data's shape, every numeric column a finite number as the
  strict decoder reads it (ASCII digits, the configured decimal mark);
- ``damaged``: the data's field count, every numeric column a finite number
  or a number-like token, at least one of each; or, when no text or blank
  line separates it from the data, every numeric column a number-like token
  (a one-column ``NaN`` row, an all-``NaN`` row);
- ``ragged``: another field count, every non-empty field a finite number;
  or, when the data rows hold a date column, every non-empty field a finite
  number or date text and at least one a finite number (a dated row with a
  channel fewer);
- ``blank``: no data byte, or only empty fields;
- ``text``: everything else.

Two cases are read as the decoder reads them, so an annotated row is never
taken for text: when the data rows end with an empty field, a value in that
last field is an annotation in the data's own column, not another field; and
a column that is empty in every data row accepts any value.

"Number-like" is the decoder's own refusal taxonomy under the reading's
decimal mark (an empty field, a quoted number, a thousands form, a
non-finite word, an overflow, a ``.`` in a decimal-comma field), plus a
closed list of look-alikes (non-ASCII digits, U+2212, digit groups with
no-break, narrow no-break, thin-space or underscore separators, ``0x``
hexadecimal, ``d`` exponents) and of non-finite spellings the decoder reports
only as "not a plain decimal number" (``1.#QNAN``, ``#VALUE!``, ...).
Placeholders outside both lists (``--``, ``OVL``) are read as text, a named
limit. No ratio or threshold enters the rule.

The census streams the file through the verified descriptor validation
already holds, counting physical lines as the decoder does
(``line_breaks``), so its line numbers are the decoder's; the decoder's own
read then ends with ``verify_unchanged``, so both reads saw the same bytes.
It keeps four bytes per classified line (its kind, and for a damaged or
ragged row its field and problem); for the first ``MAX_EXCLUSIONS + 1``
blocks of each side, the text of the line that separates each from the data,
cut to the length a finding shows; and a summary of each block with its first
damaged or ragged rows for the first ``MAX_EXCLUSIONS + 1`` blocks; blocks
past those are counted, never dropped. It claims only what it classified.

Nothing here writes anywhere or imports the signal store or the ledger.
"""

import re
import unicodedata
from array import array
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

from ..file_text import PREAMBLE_LINE_CHARS
from ..path_safety import VerifiedFile
from ..signal_acquisition.line_breaks import LineBreakNormalizer
from ..signal_acquisition.text_decoder import (
    _NON_FINITE_WORDS,
    _THOUSANDS,
    _UTF8_BOM,
    _token_problem,
)
from ..signal_acquisition.text_scan import HEAD_BYTES, HEAD_LINES, LINE_CAP
from .attestation import MAX_EXCLUSIONS
from .timebase import _DATETIME

__all__ = [
    "LineKind",
    "UNEXPLAINED",
    "MAX_LISTED_ROWS",
    "RowShape",
    "Classified",
    "IrregularRow",
    "ExcludedBlock",
    "BlockTally",
    "KEPT_BLOCKS",
    "RegionAudit",
    "fields_of",
    "column_classes",
    "shape_of",
    "classify",
    "skip_unexplained",
    "census",
    "three_significant",
]

#: What a line is to the data region.
LineKind = Literal["data", "damaged", "ragged", "blank", "text"]

#: The kinds of a row that could be part of the acquisition.
UNEXPLAINED: tuple[str, ...] = ("data", "damaged", "ragged")

#: Damaged and ragged rows a block names at most, in line order.
MAX_LISTED_ROWS = 10

_WHITESPACE_RUN = re.compile(r"[ \t]+")
_DATA_BYTE = re.compile(rb"[^ \t\x0b\x0c]")


def fields_of(text: str, delimiter: str) -> list[str]:
    """The fields of *text* under *delimiter*, however many (a whitespace
    reading splits on runs of spaces and tabs after trimming them)."""
    if delimiter == "whitespace":
        stripped = text.strip(" \t")
        return _WHITESPACE_RUN.split(stripped) if stripped else [""]
    return text.split("\t" if delimiter == "tab" else delimiter)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _number_pattern(
    decimal: str, exponent: str = "[0-9]+", integer: str = "[0-9]+"
) -> str:
    mark = re.escape(decimal)
    return rf"[+-]?(?:{integer}(?:{mark}[0-9]*)?|{mark}[0-9]+)(?:[eE][+-]?{exponent})?"


_FINITE = {mark: re.compile(_number_pattern(mark)) for mark in (".", ",")}
_HEX = re.compile(r"[+-]?0[xX][0-9a-fA-F]+")
_D_EXPONENT = {
    mark: re.compile(
        rf"[+-]?(?:[0-9]+(?:{re.escape(mark)}[0-9]*)?|{re.escape(mark)}[0-9]+)"
        rf"[dD][+-]?[0-9]+"
    )
    for mark in (".", ",")
}
_GROUPED = {
    mark: re.compile(
        rf"[+-]?[0-9]{{1,3}}(?:[   _][0-9]{{3}})+" rf"(?:{re.escape(mark)}[0-9]*)?"
    )
    for mark in (".", ",")
}
_SPELLINGS = frozenset(
    {
        "1.#QNAN",
        "1.#SNAN",
        "1.#INF",
        "-1.#INF",
        "-1.#IND",
        "#VALUE!",
        "#DIV/0!",
        "#NUM!",
        "#REF!",
        "#NAME?",
        "#NULL!",
    }
)


def _finite(token: str, decimal: str) -> Optional[bool]:
    """True: a finite number the decoder reads; False: the decoder's number
    grammar, but it overflows; None: not the grammar."""
    if _FINITE[decimal].fullmatch(token) is None:
        return None
    value = float(token.replace(",", ".") if decimal == "," else token)
    return value not in (float("inf"), float("-inf"))


def _ascii_digits(token: str) -> Optional[str]:
    """*token* with every non-ASCII decimal digit replaced by its value, or
    None when it holds none."""
    out = []
    changed = False
    for char in token:
        if not char.isascii() and char.isdigit():
            value = unicodedata.decimal(char, None)
            if value is None:
                return None
            out.append(str(value))
            changed = True
        else:
            out.append(char)
    return "".join(out) if changed else None


def _number_like(token: str, decimal: str) -> bool:
    """A token the decoder refuses that keeps a number's place: its own
    refusal taxonomy under *decimal*, and the closed lists of look-alikes and
    of non-finite spellings."""
    if not token:
        return True  # an empty field
    if '"' in token or "'" in token:
        inner = token.strip("\"'").strip(" ")
        if inner and _finite(inner, decimal) is not None:
            return True
    raw = token.encode("utf-8", "replace")
    if _THOUSANDS[decimal].fullmatch(raw):
        return True
    if raw.lower().lstrip(b"+-") in _NON_FINITE_WORDS:
        return True
    if _finite(token, decimal) is False:
        return True  # an overflow
    if decimal == "," and _finite(token, ".") is True:
        return True  # a '.' in a decimal-comma field
    digits = _ascii_digits(token)
    if digits is not None and _finite(digits, decimal) is not None:
        return True
    if "−" in token and _finite(token.replace("−", "-"), decimal):
        return True
    if (
        _GROUPED[decimal].fullmatch(token)
        or _HEX.fullmatch(token)
        or _D_EXPONENT[decimal].fullmatch(token)
    ):
        return True
    spelled = token.upper().replace(",", ".") if decimal == "," else token.upper()
    return spelled in _SPELLINGS


def _problem(token: str, decimal: str) -> tuple[str, str]:
    """The issue code and the decoder's phrase for a number-like token."""
    if _finite(token, decimal) is False:
        return "non_finite_token", "is not a finite number"
    return _token_problem(token.encode("utf-8", "replace"), decimal, time=False)


def _is_date_text(token: str, decimal: str) -> bool:
    """The time grammar refuses *token* as a date or a time of day.

    The grammar marks a refused stamp so when it matches the date and
    time-of-day pattern, and no stamp the grammar reads matches that pattern
    (two date separators or a colon), so the pattern alone decides under
    either decimal mark. Parsing and refusing the token would decide the
    same at hundreds of times the cost, once per row of a dated shape.
    """
    return _DATETIME.fullmatch(token.strip(" ")) is not None


def _text_class(token: str, decimal: str) -> str:
    if not token:
        return "empty"
    if _finite(token, decimal):
        return "number"
    return "date" if _is_date_text(token, decimal) else "word"


# ---------------------------------------------------------------------------
# Shape and classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowShape:
    """The shape of the data region under one reading.

    Attributes:
        delimiter: ``,``, ``;``, ``tab`` or ``whitespace``.
        decimal: ``.`` or ``,``.
        fields: Fields per data row, a trailing empty field not counted.
        numeric: The columns that hold numbers in the data rows.
        text_classes: For every other column, the classes its fields show in
            the data rows (``date``, ``word``, ``empty``, ``number``).
        trailing: The data rows end with an empty field (a trailing
            delimiter).
    """

    delimiter: str
    decimal: str
    fields: int
    numeric: frozenset[int]
    text_classes: Mapping[int, frozenset[str]]
    trailing: bool = False


@dataclass(frozen=True)
class Classified:
    """What one line is to the data region.

    Attributes:
        kind: One of :data:`LineKind`.
        field: For a damaged row, its first number-like numeric field.
        code: For a damaged row, the decoder's issue code for that field.
        problem: For a damaged row, the decoder's phrase for that field.
        columns: For a damaged row, every number-like numeric field.
        count: For a ragged row, its number of fields.
    """

    kind: LineKind
    field: Optional[int] = None
    code: Optional[str] = None
    problem: Optional[str] = None
    columns: tuple[int, ...] = ()
    count: Optional[int] = None


_BLANK = Classified("blank")
_TEXT = Classified("text")
_DATA = Classified("data")


def _cells(text: str, delimiter: str, fields: int) -> list[str]:
    """The stripped fields of *text*, without trailing empty fields past
    *fields* (a trailing delimiter)."""
    cells = [cell.strip(" ") for cell in fields_of(text, delimiter)]
    if len(cells) > fields and not any(cells[fields:]):
        cells = cells[:fields]
    return cells


def _shaped_cells(text: str, shape: RowShape) -> list[str]:
    """The cells of *text* under *shape*: a value in the empty last field the
    data rows end with is dropped, as the decoder never reads that column. A
    line with nothing but that value keeps it: it is no data row, and only
    its other cells tell text from a ragged row."""
    cells = _cells(text, shape.delimiter, shape.fields)
    if (
        shape.trailing
        and len(cells) > shape.fields
        and not any(cells[shape.fields + 1 :])
        and any(cells[: shape.fields])
    ):
        cells = cells[: shape.fields]
    return cells


def column_classes(
    rows: Sequence[Sequence[str]], fields: int, numeric: frozenset[int], decimal: str
) -> dict[int, frozenset[str]]:
    """For every column of *fields* outside *numeric*, the classes its
    stripped fields show in *rows* (``date``, ``word``, ``empty``,
    ``number``)."""
    return {
        column: frozenset(
            _text_class(row[column], decimal) for row in rows if len(row) > column
        )
        for column in range(fields)
        if column not in numeric
    }


def _has_data(text: str) -> bool:
    return any(char not in " \t\x0b\x0c" for char in text)


def _annotates(
    first: Sequence[str], following: str, delimiter: str, decimal: str
) -> bool:
    """The first row ends with a value in the field the next row leaves
    empty (a trailing delimiter), their other finite columns the same: that
    value is an annotation in the data's empty last column, as inspection
    reads the first row of a run. Inspection starts no single-field run on
    such a row (it may be a value with its unit); a range configured to start
    there is still read so here, which only makes more rows outside it ask."""
    after = [cell.strip(" ") for cell in fields_of(following, delimiter)]
    if len(after) != len(first) or len(after) < 2 or after[-1]:
        return False
    finite = {i for i, cell in enumerate(after[:-1]) if _finite(cell, decimal)}
    return bool(finite) and finite == {
        i for i, cell in enumerate(first[:-1]) if _finite(cell, decimal)
    }


def shape_of(rows: Sequence[str], delimiter: str, decimal: str) -> Optional[RowShape]:
    """The data region's shape from its first rows, or None when they hold
    no number.

    The field count is the first row's, a trailing empty field not counted,
    nor an annotation the first row holds in the empty last field the next
    row ends with; a column is numeric when most rows of that field count
    hold a finite number there.
    """
    texts = [text for text in rows if _has_data(text)]
    if not texts:
        return None
    first = [cell.strip(" ") for cell in fields_of(texts[0], delimiter)]
    trailing = len(first) > 1 and not first[-1]
    if trailing or (len(texts) > 1 and _annotates(first, texts[1], delimiter, decimal)):
        first, trailing = first[:-1], True
    count = len(first)
    same = [
        cells
        for cells in (_cells(text, delimiter, count) for text in texts)
        if len(cells) == count
    ]
    numeric = frozenset(
        column
        for column in range(count)
        if 2 * sum(1 for cells in same if _finite(cells[column], decimal)) > len(same)
    )
    if not numeric:
        return None
    classes = column_classes(same, count, numeric, decimal)
    return RowShape(delimiter, decimal, count, numeric, classes, trailing)


def _raw_class(text: str, shape: RowShape) -> tuple[Classified, bool]:
    """The class of *text* before the position rule, and whether the
    position rule decides it (every numeric column number-like: damaged
    beside the data, text away from it)."""
    if not _has_data(text):
        return _BLANK, False
    cells = _shaped_cells(text, shape)
    if not any(cells):
        return _BLANK, False
    decimal = shape.decimal
    if len(cells) != shape.fields:
        filled = [cell for cell in cells if cell]
        finite = [cell for cell in filled if _finite(cell, decimal)]
        if filled and (
            len(finite) == len(filled)
            or (finite and _dated(shape) and _numbers_and_dates(filled, decimal))
        ):
            return Classified("ragged", count=len(cells)), False
        return _TEXT, False
    finite_columns: list[int] = []
    like: list[int] = []
    for column in sorted(shape.numeric):
        cell = cells[column]
        if _finite(cell, decimal):
            finite_columns.append(column)
        elif _number_like(cell, decimal):
            like.append(column)
        else:
            return _TEXT, False
    if not like:
        if all(
            classes == _EMPTY_ONLY or _text_class(cells[column], decimal) in classes
            for column, classes in shape.text_classes.items()
        ):
            return _DATA, False
        return _TEXT, False
    code, problem = _problem(cells[like[0]], decimal)
    damaged = Classified(
        "damaged", field=like[0], code=code, problem=problem, columns=tuple(like)
    )
    return damaged, not finite_columns


#: The classes of a column that is empty in every data row: an annotation
#: there is still a data row's.
_EMPTY_ONLY = frozenset({"empty"})


def _dated(shape: RowShape) -> bool:
    """The data rows hold a date column."""
    return any("date" in classes for classes in shape.text_classes.values())


def _numbers_and_dates(cells: Sequence[str], decimal: str) -> bool:
    return all(_finite(cell, decimal) or _is_date_text(cell, decimal) for cell in cells)


def classify(text: str, shape: RowShape, *, adjacent: bool) -> Classified:
    """The class of one line.

    Args:
        text: The line, without its break.
        shape: The data region's shape.
        adjacent: No text or blank line separates the line from the data
            (directly above or below it, inside a block of rows beside it, or
            between data rows).
    """
    found, positional = _raw_class(text, shape)
    if positional and not adjacent:
        return _TEXT
    return found


def skip_unexplained(
    texts_upward: Sequence[str],
    shape: Optional[RowShape],
    *,
    include_data: bool = False,
) -> int:
    """How many lines, read upward from just above the data, are damaged or
    ragged rows (and, with *include_data*, rows of the data's shape) before
    the first other non-blank line. Blank lines are passed over, and they end
    the position rule's adjacency."""
    if shape is None:
        return 0
    skipped = 0
    adjacent = True
    for text in texts_upward:
        if not _has_data(text):
            adjacent = False
            skipped += 1
            continue
        kind = classify(text, shape, adjacent=adjacent).kind
        if kind in ("damaged", "ragged") or (include_data and kind == "data"):
            skipped += 1
            continue
        break
    # Trailing blank lines passed over belong to whatever lies above them.
    while skipped and not _has_data(texts_upward[skipped - 1]):
        skipped -= 1
    return skipped


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IrregularRow:
    """A damaged or ragged row of a block.

    Attributes:
        line: Its physical line.
        kind: ``damaged`` or ``ragged``.
        field: For a damaged row, its first number-like numeric field.
        problem: For a damaged row, the decoder's phrase for that field.
        count: For a ragged row, its number of fields.
    """

    line: int
    kind: Literal["damaged", "ragged"]
    field: Optional[int] = None
    problem: Optional[str] = None
    count: Optional[int] = None


@dataclass(frozen=True)
class ExcludedBlock:
    """A maximal run of rows that could be part of the acquisition, on one
    side of the data region, that no text or blank line interrupts.

    Attributes:
        side: ``above`` (before the data start) or ``below`` (after a
            stated last data line).
        first_line: Its first physical line.
        last_line: Its last physical line.
        rows: Its rows (``last_line - first_line + 1``).
        data_rows: Rows with the data's shape.
        damaged: Damaged rows.
        ragged: Ragged rows.
        listed: The first :data:`MAX_LISTED_ROWS` damaged or ragged rows.
        separator: What separates it from the data: ``header``, ``units``,
            ``blank``, ``text`` or ``nothing``.
        separator_line: The separating line (None for ``nothing``).
        separator_text: The separating line's text, for a header, units or
            text line, stripped and cut to one character past the length a
            finding shows (raw file text: wrap it before it reaches a
            result).
    """

    side: Literal["above", "below"]
    first_line: int
    last_line: int
    rows: int
    data_rows: int
    damaged: int
    ragged: int
    listed: tuple[IrregularRow, ...]
    separator: str
    separator_line: Optional[int]
    separator_text: Optional[str]

    def as_range(self) -> dict[str, int]:
        """``{first_line, last_line, rows}``, as records and statements hold it."""
        return {
            "first_line": self.first_line,
            "last_line": self.last_line,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class BlockTally:
    """The blocks of one side counted past the ones the census keeps.

    Attributes:
        side: ``above`` or ``below``.
        blocks: Blocks counted.
        rows: Their rows in all.
        first_line: The first line of the first of them.
        last_line: The last line of the last of them.
    """

    side: Literal["above", "below"]
    blocks: int
    rows: int
    first_line: int
    last_line: int


#: Blocks the census keeps in full, in line order: one more than an
#: acknowledgment can name, so a range past that limit is always known as such.
KEPT_BLOCKS = MAX_EXCLUSIONS + 1


@dataclass(frozen=True)
class RegionAudit:
    """The census of the lines a data range leaves out.

    Attributes:
        shape: The data region's shape the lines were classified under (None
            when the region's first rows hold no number: nothing classified).
        blocks: The first :data:`KEPT_BLOCKS` blocks, above the data first,
            in line order.
        kinds: The class of each line the caller asked about (the header and
            units lines), when it lies in a classified zone.
        complete: Every line outside the data range was classified (False
            only when the region's first rows hold no number to classify by).
        tallies: Per side, the blocks counted past :attr:`blocks` (empty when
            every block is kept).
    """

    shape: Optional[RowShape]
    blocks: tuple[ExcludedBlock, ...]
    kinds: Mapping[int, str]
    complete: bool = True
    tallies: tuple[BlockTally, ...] = ()

    @property
    def clean(self) -> bool:
        """No line outside the data range could be part of the acquisition."""
        return not self.blocks

    @property
    def block_count(self) -> int:
        """Every block, kept or counted."""
        return len(self.blocks) + sum(tally.blocks for tally in self.tallies)

    def rows_above(self, line: int) -> bool:
        """Whether a block lies above physical *line*."""
        return any(b.side == "above" and b.first_line < line for b in self.blocks)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


#: Bytes the census reads at a time (a line cut across two reads is held
#: until its end).
CENSUS_BLOCK_BYTES = 64 * 1024

#: Line kinds as stored, one byte per line.
_K_DATA, _K_DAMAGED, _K_RAGGED, _K_BLANK, _K_TEXT, _K_CANDIDATE = range(6)
_KIND_CODES = {
    "data": _K_DATA,
    "damaged": _K_DAMAGED,
    "ragged": _K_RAGGED,
    "blank": _K_BLANK,
    "text": _K_TEXT,
}
_KIND_NAMES = {code: name for name, code in _KIND_CODES.items()}


def _iter_lines(
    file: VerifiedFile, encoding: str, ranges: Sequence[tuple[int, Optional[int]]]
) -> Iterator[tuple[int, bytes]]:
    """The physical lines of *file* whose numbers fall in *ranges*
    (inclusive, an upper bound of None reads to the end), without their
    breaks, numbered as the strict decoder numbers them. A line longer than
    ``LINE_CAP`` is cut there (the decoder refuses such a file)."""
    reader = _LineReader(ranges)
    normalizer = LineBreakNormalizer()
    offset = 0
    first = True
    while offset < file.size and not reader.done:
        chunk = file.read(offset, CENSUS_BLOCK_BYTES)
        if not chunk:
            break
        offset += len(chunk)
        if first and encoding == "utf-8-sig" and chunk.startswith(_UTF8_BOM):
            chunk = chunk[len(_UTF8_BOM) :]
        first = False
        for piece in normalizer.feed(chunk):
            yield from reader.process(piece)
            if reader.done:
                return
    if reader.done:
        return
    for piece in normalizer.finish():
        yield from reader.process(piece)
        if reader.done:
            return
    if reader.carry and reader.wanted(reader.line):
        yield reader.line, reader.carry


class _LineReader:
    """Splits normalized pieces into numbered lines, keeping only the lines
    in its ranges and counting the others without splitting them."""

    def __init__(self, ranges: Sequence[tuple[int, Optional[int]]]) -> None:
        self.ranges = list(ranges)
        self.end: Optional[int] = (
            None
            if any(hi is None for _, hi in self.ranges)
            else max(hi for _, hi in self.ranges if hi is not None)
        )
        self.starts = sorted(lo for lo, _ in self.ranges)
        self.line = 1  # the line the next byte belongs to
        self.carry = b""  # the start of that line, when it is wanted
        self.done = False

    def wanted(self, number: int) -> bool:
        return any(
            lo <= number and (hi is None or number <= hi) for lo, hi in self.ranges
        )

    def next_wanted(self, number: int) -> Optional[int]:
        for lo in self.starts:
            if lo > number:
                return lo
        return None

    def process(self, piece: bytes) -> Iterator[tuple[int, bytes]]:
        pos = 0
        size = len(piece)
        while pos < size:
            line = self.line
            if self.end is not None and line > self.end:
                self.done = True
                return
            if not self.wanted(line):
                target = self.next_wanted(line)
                skippable = piece.count(b"\n", pos)
                if target is None or line + skippable < target:
                    if target is None and self.end is not None:
                        self.done = True
                        return
                    self.line = line + skippable
                    self.carry = b""
                    return
            newline = piece.find(b"\n", pos)
            if newline < 0:
                if self.wanted(line) and len(self.carry) <= LINE_CAP:
                    self.carry = (self.carry + piece[pos:])[: LINE_CAP + 1]
                return
            if self.wanted(line):
                yield line, (self.carry + piece[pos:newline])[: LINE_CAP + 1]
            self.carry = b""
            self.line = line + 1
            pos = newline + 1


def _text(raw: bytes, encoding: str) -> str:
    codec = "utf-8" if encoding == "utf-8-sig" else encoding
    return raw.decode(codec, errors="replace")


def _region_rows(
    file: VerifiedFile, encoding: str, first: int, last: Optional[int]
) -> list[str]:
    """The data region's first rows: from line *first*, at most
    ``HEAD_LINES`` rows within ``HEAD_BYTES``, up to a blank line or line
    *last*."""
    stop = first + HEAD_LINES - 1
    if last is not None:
        stop = min(stop, last)
    rows: list[str] = []
    size = 0
    for _number, raw in _iter_lines(file, encoding, [(first, stop)]):
        if not _DATA_BYTE.search(raw):
            break
        size += len(raw) + 1
        if rows and size > HEAD_BYTES:
            break
        rows.append(_text(raw, encoding))
    return rows


def _fast_data(shape: RowShape) -> Optional["re.Pattern[bytes]"]:
    """A byte pattern that matches only rows of the data's shape, for a
    shape whose every column is numeric (at most 200 integer digits and a
    two-digit exponent, so no match overflows: anything longer is left to the
    full classifier)."""
    if len(shape.numeric) != shape.fields:
        return None
    number = _number_pattern(shape.decimal, "[0-9]{1,2}", "[0-9]{1,200}")
    if shape.delimiter == "whitespace":
        body = rf"[ \t]*{number}(?:[ \t]+{number}){{{shape.fields - 1}}}[ \t]*"
    else:
        separator = re.escape("\t" if shape.delimiter == "tab" else shape.delimiter)
        cell = rf" *{number} *"
        body = rf"{cell}(?:{separator}{cell}){{{shape.fields - 1}}}(?:{separator} *)?"
    return re.compile(body.encode("ascii"))


def _resolved(kind: int, adjacent: bool) -> tuple[int, bool]:
    """The position rule for one line, walking away from the data: the
    line's kind, and whether the next line is still beside the data."""
    if kind == _K_CANDIDATE:
        kind = _K_DAMAGED if adjacent else _K_TEXT
    if kind in (_K_TEXT, _K_BLANK):
        return kind, False
    return kind, adjacent or kind == _K_DATA


class _Zone:
    """The classified lines of one side of the data region, and the texts of
    the lines that separate its first ``KEPT_BLOCKS`` blocks from the data
    (no other line's, so their number never grows with the file)."""

    def __init__(self, side: Literal["above", "below"], base: int) -> None:
        self.side = side
        self.base = base  # physical line of index 0
        self.kinds = bytearray()
        self.fields = array("h")
        self.problems = bytearray()
        self.texts: dict[int, str] = {}
        # Above the data: whether the previous line is a row the position
        # rule does not decide, and the position-ruled rows right after such
        # a row whose text is held until the rule decides them.
        self._row_before = False
        self._held: list[tuple[int, str]] = []
        # Below the data: the position rule as the census reads, and the
        # previous line when it is text.
        self._adjacent = True
        self._previous: Optional[tuple[int, str]] = None

    def add_data(self, number: int) -> None:
        """A row the byte pattern reads as data: :meth:`add` and :meth:`note`
        in one step, the census's most common line."""
        self.kinds.append(_K_DATA)
        self.fields.append(0)
        self.problems.append(0)
        if self.side == "above":
            if self._held:
                self._held.clear()
            self._row_before = True
            return
        self._adjacent = True
        previous, self._previous = self._previous, None
        if (
            previous is not None
            and previous[0] == number - 1
            and len(self.texts) < KEPT_BLOCKS
        ):
            self.texts[previous[0]] = _separator_text(previous[1])

    def note(self, number: int, kind: int, text: str) -> None:
        """Keep *text* when line *number* (of *kind* before the position
        rule) separates one of the first ``KEPT_BLOCKS`` blocks from the
        data: above the data the line after a block, below it the line
        before one."""
        if self.side == "above":
            self._note_above(number, kind, text)
        else:
            self._note_below(number, kind, text)

    def _note_above(self, number: int, kind: int, text: str) -> None:
        # The rule decides a position-ruled row walking up from the data, so
        # here, reading down, by the next text, blank or data row (damaged
        # and ragged rows pass the decision on): text or blank makes the
        # held rows text, each the separator of the row before it; a data
        # row, or the data start, makes them damaged. Each kept text is a
        # different block's, so at most KEPT_BLOCKS are kept or held.
        if kind == _K_CANDIDATE:
            if self._row_before and len(self.texts) + len(self._held) < KEPT_BLOCKS:
                self._held.append((number, _separator_text(text)))
        else:
            if kind in (_K_TEXT, _K_BLANK):
                self.texts.update(self._held)
                self._held.clear()
            elif kind == _K_DATA:
                self._held.clear()
            if kind == _K_TEXT and self._row_before and len(self.texts) < KEPT_BLOCKS:
                self.texts[number] = _separator_text(text)
        self._row_before = kind in (_K_DATA, _K_DAMAGED, _K_RAGGED)

    def _note_below(self, number: int, kind: int, text: str) -> None:
        # Walking down from the data, the rule decides each row as it is
        # read (the same steps as resolve()).
        kind, self._adjacent = _resolved(kind, self._adjacent)
        previous, self._previous = self._previous, None
        if kind in (_K_DATA, _K_DAMAGED, _K_RAGGED):
            if (
                previous is not None
                and previous[0] == number - 1
                and len(self.texts) < KEPT_BLOCKS
            ):
                self.texts[previous[0]] = _separator_text(previous[1])
        elif kind == _K_TEXT:
            self._previous = (number, text)

    def add(self, found: Classified, candidate: bool, problems: list[str]) -> None:
        if candidate:
            kind = _K_CANDIDATE
        else:
            kind = _KIND_CODES[found.kind]
        self.kinds.append(kind)
        if found.kind == "damaged":
            if found.problem not in problems:
                problems.append(found.problem or "")
            self.fields.append(min(found.field or 0, 32767))
            self.problems.append(problems.index(found.problem or ""))
        elif found.kind == "ragged":
            self.fields.append(min(found.count or 0, 32767))
            self.problems.append(0)
        else:
            self.fields.append(0)
            self.problems.append(0)

    def resolve(self, upward: bool) -> None:
        """Apply the position rule, walking away from the data."""
        order: Sequence[int] = (
            range(len(self.kinds) - 1, -1, -1) if upward else range(len(self.kinds))
        )
        adjacent = True
        for index in order:
            self.kinds[index], adjacent = _resolved(self.kinds[index], adjacent)

    def kind_of(self, line: int) -> Optional[str]:
        index = line - self.base
        if 0 <= index < len(self.kinds):
            return _KIND_NAMES[self.kinds[index]]
        return None


def _blocks(
    zone: _Zone,
    problems: list[str],
    *,
    data_edge: int,
    header_line: Optional[int],
    units_line: Optional[int],
    keep: int,
) -> tuple[list[ExcludedBlock], Optional[BlockTally]]:
    """The zone's blocks in line order: the first *keep* in full, the rest
    counted in one tally."""
    kinds = zone.kinds
    out: list[ExcludedBlock] = []
    counted = counted_rows = 0
    counted_first = counted_last = 0
    index = 0
    total = len(kinds)
    unexplained = (_K_DATA, _K_DAMAGED, _K_RAGGED)
    while index < total:
        if kinds[index] not in unexplained:
            index += 1
            continue
        start = index
        while index < total and kinds[index] in unexplained:
            index += 1
        stop = index  # exclusive
        if len(out) >= keep:
            if not counted:
                counted_first = zone.base + start
            counted += 1
            counted_rows += stop - start
            counted_last = zone.base + stop - 1
            continue
        span = bytes(kinds[start:stop])
        damaged = span.count(_K_DAMAGED)
        ragged = span.count(_K_RAGGED)
        listed: list[IrregularRow] = []
        position = 0
        while len(listed) < MAX_LISTED_ROWS and (damaged or ragged):
            hits = [
                found
                for found in (
                    span.find(bytes([_K_DAMAGED]), position),
                    span.find(bytes([_K_RAGGED]), position),
                )
                if found >= 0
            ]
            if not hits:
                break
            at = min(hits)
            row = start + at
            line = zone.base + row
            if span[at] == _K_DAMAGED:
                listed.append(
                    IrregularRow(
                        line,
                        "damaged",
                        field=zone.fields[row],
                        problem=problems[zone.problems[row]],
                    )
                )
            else:
                listed.append(IrregularRow(line, "ragged", count=zone.fields[row]))
            position = at + 1
        first_line = zone.base + start
        last_line = zone.base + stop - 1
        neighbour = last_line + 1 if zone.side == "above" else first_line - 1
        if neighbour == data_edge:
            separator, separator_line, separator_text = "nothing", None, None
        else:
            kind = zone.kind_of(neighbour)
            separator_line = neighbour
            separator_text = zone.texts.get(neighbour)
            if kind == "blank":
                separator, separator_text = "blank", None
            elif neighbour == header_line:
                separator = "header"
            elif neighbour == units_line:
                separator = "units"
            else:
                separator = "text"
        out.append(
            ExcludedBlock(
                side=zone.side,
                first_line=first_line,
                last_line=last_line,
                rows=stop - start,
                data_rows=span.count(_K_DATA),
                damaged=damaged,
                ragged=ragged,
                listed=tuple(listed),
                separator=separator,
                separator_line=separator_line,
                separator_text=separator_text,
            )
        )
    tally = (
        BlockTally(zone.side, counted, counted_rows, counted_first, counted_last)
        if counted
        else None
    )
    return out, tally


def _separator_text(text: str) -> str:
    """A possible separator's text as a finding can show it: stripped, and
    cut one character past the cap, so the shown text and its truncation
    flag are the full line's."""
    return text.strip()[: PREAMBLE_LINE_CHARS + 1]


def census(
    file: VerifiedFile,
    interpretation: Mapping[str, Any],
    *,
    header_line: Optional[int] = None,
    units_line: Optional[int] = None,
) -> RegionAudit:
    """Classify every line above the configured data start and, when a last
    data line is stated, every line after it.

    Args:
        file: The verified descriptor validation holds (nothing is closed
            or written; the decoder's later read verifies the file
            unchanged).
        interpretation: The canonical interpretation.
        header_line: The header line, for the separator's kind.
        units_line: The units line, for the separator's kind.

    Returns:
        The :class:`RegionAudit`.
    """
    dialect = interpretation["dialect"]
    encoding = dialect["encoding"]
    first = interpretation["data_lines"]["first"]
    last = interpretation["data_lines"]["last"]
    shape = shape_of(
        _region_rows(file, encoding, first, last),
        dialect["delimiter"],
        dialect["decimal"],
    )
    if shape is None:
        return RegionAudit(None, (), {}, complete=first == 1 and last is None)
    fast = _fast_data(shape)
    problems: list[str] = []
    above = _Zone("above", 1)
    below = _Zone("below", (last or 0) + 1)
    ranges: list[tuple[int, Optional[int]]] = []
    if first > 1:
        ranges.append((1, first - 1))
    if last is not None:
        ranges.append((last + 1, None))
    for number, raw in _iter_lines(file, encoding, ranges) if ranges else ():
        zone = above if number < first else below
        if fast is not None and fast.fullmatch(raw):
            zone.add_data(number)
            continue
        text = _text(raw, encoding)
        found, candidate = _raw_class(text, shape)
        zone.add(found, candidate, problems)
        zone.note(number, zone.kinds[-1], text)
    above.resolve(upward=True)
    below.resolve(upward=False)
    blocks, above_tally = _blocks(
        above,
        problems,
        data_edge=first,
        header_line=header_line,
        units_line=units_line,
        keep=KEPT_BLOCKS,
    )
    tallies = [above_tally] if above_tally is not None else []
    if last is not None:
        below_blocks, below_tally = _blocks(
            below,
            problems,
            data_edge=last,
            header_line=header_line,
            units_line=units_line,
            keep=KEPT_BLOCKS - len(blocks),
        )
        blocks += below_blocks
        if below_tally is not None:
            tallies.append(below_tally)
    kinds = {
        line: kind
        for line in (header_line, units_line)
        if line is not None
        for kind in [above.kind_of(line) if line < first else below.kind_of(line)]
        if kind is not None
    }
    return RegionAudit(shape, tuple(blocks), kinds, tallies=tuple(tallies))


def three_significant(value: Decimal) -> str:
    """*value* rounded to three significant digits, as a plain decimal."""
    if value == 0:
        return "0"
    exponent = value.adjusted() - 2
    return format(value.quantize(Decimal(1).scaleb(exponent)), "f")
