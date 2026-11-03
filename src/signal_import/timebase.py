"""
The time-base assessment of the signal import (plan KTD9; R12 to R15).

ISO 13374 Block 1 (data acquisition). A pure function from one time column's
stamps, exactly as printed, a declared time unit and optional rate evidence
to a typed assessment: the evidence level the column's print supports, its
findings, statistics and, when the column supports one, a proposed sampling
rate. No signal is ever resampled, interpolated, gap-filled or repaired, and
no wording here suggests it.

Every bound comes from the printed resolution; there is no free tolerance.

1. *Print regime.* A stamp is ``mantissa * 10**exponent`` (the decoder's
   output, :mod:`..signal_acquisition.text_decoder`). A print whose
   coarsest rounding is finer than float64 resolution at its largest stamp
   is ``shortest_repr`` (repr, ``%.17g``): the float summation bound sets
   the grid, one decade above it. A print with a row cut at its widest
   significant-digit count above the finest decimal place is
   ``significant_digits`` (``%e``, ``%g``): each row keeps its own quantum,
   the rate needs a stated or file-declared basis, and every stamp is
   checked against it. Anything else is ``fixed_decimals``, trailing zeros
   stripped or not.
2. *Grid.* Stamps become exact int64 ticks of one grid (the finest decimal
   place, the commonest quantum, or the float-bound decade), rounded half
   to even in integer arithmetic. Every stamp has a rounding half-width in
   ticks: half a tick on the grid, more for a coarser quantum or float32
   values.
3. *Effective resolution.* Stamps that are all float32 values, tested
   exactly, whose float32 spacing is coarser than the decimals the print
   actually uses (time accumulated in float32) carry that spacing: evidence
   ``duration_only``, no derived rate.
4. *Reference interval.* ``b`` is the mean of the modal pair of adjacent
   tick steps ``{m, m+1}`` between stamps on the grid, never a fitted slope,
   which absorbs the defect it should find.
5. *Evidence level.* ``full`` when ``b >= 2``; ``partial`` when
   ``1 <= b < 2``; ``duration_only`` when ``b < 1`` or as in 3.
6. *Steps.* At ``full`` and ``partial``, a step of ``1.5 b`` or more is a
   gap (``time_gap``) and a step below ``0.5 b`` a duplicate or reversal
   (``time_reversal``), both widened by any extra rounding of the two
   stamps; a step of the modal pair is the reference, never an offence. At
   ``duration_only`` a backward step is a reversal (rounding never reverses
   a clock), and a step is a gap only when it is longer than 1.5 intervals
   whatever the rounding.
7. *Residual.* After the step tests pass, a least-squares line; a stamp
   further than ``b/2`` from it, or than the print-and-fit bound when that
   is wider, plus its extra rounding, is a timing defect
   (``time_residual``).
8. *Pattern.* Every step a uniform clock can print and every residual
   within the print-and-fit bound (one tick for fixed decimals: half a tick
   of print, half a tick of fit) is uniform; anything else is jitter or
   wander (``time_off_pattern``), which needs a choice of time-base source.
9. *Rate sources* (R15). A stated rate must predict every stamp within its
   rounding plus the print-and-fit half; a printed preamble rate or
   interval carries its half-unit interval, and a stated rate must lie
   inside it; at ``duration_only`` only the total duration is checked,
   within one effective tick. Any disagreement is
   ``rate_sources_disagree``.

The fitted rate is reported with 12 significant digits and the exact
interval the print's rounding alone allows it; it is checked with
``validate_sampling_rate``. It is proposed as a rate only at ``full``
evidence, outside ``significant_digits``, with no finding at all; even then
a time column alone never sets the rate (R12): the caller records a basis.

Memory: the step array (8 bytes per row) and, outside fixed decimals, the
tick array; the floating-point passes run in chunks of :data:`_CHUNK_ROWS`.
"""

import decimal
import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal, Optional

import numpy as np

from ..file_text import TOKEN_CHARS, FileText
from ..signal_acquisition.import_config import (
    DECIMAL_MARKS,
    TIME_UNITS,
    normalize_decimal,
)
from ..signal_acquisition.text_decoder import (
    TIME_MAX_DIGITS,
    TIME_MAX_EXPONENT_DIGITS,
    TIME_TOKEN_BYTES,
    TimeColumnResult,
    parse_time_matrix,
)
from ..signal_validation import InvalidInputError, validate_sampling_rate

__all__ = [
    "REGIMES",
    "EVIDENCE_LEVELS",
    "OUTCOMES",
    "FINDING_CODES",
    "RATE_SOURCES",
    "MAX_LISTED_LINES",
    "RATE_DIGITS",
    "TimeBaseRefused",
    "DeclaredTimeBase",
    "RateSource",
    "RateComparison",
    "TimeFinding",
    "TimeBaseAssessment",
    "parse_time_tokens",
    "assess_time_base",
    "assess_time_column",
]

#: Print regimes of a time column.
REGIMES: tuple[str, ...] = ("fixed_decimals", "shortest_repr", "significant_digits")

#: What a column's print can show: single-sample gaps (``full``), duplicates
#: and reversals only (``partial``), or the total duration only.
EVIDENCE_LEVELS: tuple[str, ...] = ("full", "partial", "duration_only")

#: ``invalid`` refuses the column (R13); ``needs_confirmation`` needs the
#: user's choice of time-base source (R13, R15).
OUTCOMES: tuple[str, ...] = ("accepted", "needs_confirmation", "invalid")

#: Issue codes of the findings, all in ``import_config.ISSUE_CODES``.
FINDING_CODES: tuple[str, ...] = (
    "time_gap",
    "time_reversal",
    "time_residual",
    "time_off_pattern",
    "rate_sources_disagree",
)

#: Where a sampling rate can come from, in the order they are listed.
RATE_SOURCES: tuple[str, ...] = ("time_column", "file_declared", "stated")

#: Offending lines listed per finding; every offence is counted.
MAX_LISTED_LINES = 10

#: Significant digits of the fitted rate.
RATE_DIGITS = 12

# Rows per chunk of the floating-point passes.
_CHUNK_ROWS = 1 << 20

# Largest step span counted with a histogram instead of a sort.
_HISTOGRAM_SPAN = 1 << 20

# Ticks stay within this magnitude, so every difference fits an int64.
_TICK_LIMIT = 1 << 61

# One tick, in seconds, lies within these powers of ten, so every derived
# rate is a decimal the configuration can hold (within 1e-30 to 1e30).
_TICK_EXPONENTS = (-30, 9)

# What the decoder produces: at most 18 digits, a 3-digit exponent less the
# fraction digits of a 32-byte token.
_MAX_MANTISSA = 10**18 - 1
_MAX_EXPONENT = 1100

_POW10 = np.array([10**i for i in range(19)], dtype=np.int64)

# Power of ten of one unit, in seconds.
_UNIT_EXPONENT = {"s": 0, "ms": -3, "us": -6, "ns": -9}

# Rate arithmetic: exact enough for 12 digits and every signal trapped.
_RATE_CONTEXT = decimal.Context(
    prec=40,
    rounding=decimal.ROUND_HALF_EVEN,
    Emax=999_999,
    Emin=-999_999,
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
)

_NON_FINITE_WORDS = {"nan", "inf", "infinity", "na", "n/a", "#n/a", "null"}
_PRINTABLE = re.compile(r"[\x21-\x7e]*")
_DATETIME = re.compile(
    r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"
    r"|\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)
_GRAMMAR_REMEDY = (
    "a time column holds plain decimal stamps; leave the column unassessed, with "
    "a reason, if it is not a time stamp the import reads"
)


class TimeBaseRefused(ValueError):
    """A time column, or a rate, the assessment cannot read.

    Attributes:
        code: Issue code of the import contract (``time_grammar`` or
            ``region_too_short``).
        line: Physical line of the offending stamp, when there is one.
        token: The offending text, short and escaped, when there is one.
        datetime_text: The stamp is a date or a time of day, which this
            version does not assess: the column blocks ``ready`` until the
            R13 override names it.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: Optional[int] = None,
        token: Optional[FileText] = None,
        datetime_text: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line = line
        self.token = token
        self.datetime_text = datetime_text


@dataclass(frozen=True)
class DeclaredTimeBase:
    """A sampling rate or sample interval printed in the file's preamble.

    Attributes:
        kind: ``"rate"`` (in Hz) or ``"interval"`` (per sample, in
            *time_unit*), for example LabVIEW's ``Delta_X``.
        token: The number exactly as printed; its last digit sets the
            half-unit interval it carries.
        time_unit: Unit of an interval (``s``, ``ms``, ``us``, ``ns``).
    """

    kind: Literal["rate", "interval"]
    token: str
    time_unit: str = "s"


@dataclass(frozen=True)
class RateSource:
    """One source of the sampling rate and the interval it supports.

    Attributes:
        source: One of :data:`RATE_SOURCES`.
        rate_hz: Its rate, as a decimal string.
        low_hz: Lowest rate it supports.
        high_hz: Highest rate it supports (equal to *low_hz* for a stated
            rate, which is exact).
    """

    source: str
    rate_hz: str
    low_hz: str
    high_hz: str


@dataclass(frozen=True)
class RateComparison:
    """Whether two rate sources agree within the precision each carries.

    Attributes:
        first: The source compared against.
        second: The other source.
        consistent: They agree.
        difference_ppm: How far the second source's rate lies from the
            first's, in parts per million (None when the column has no
            rate to compare).
    """

    first: str
    second: str
    consistent: bool
    difference_ppm: Optional[float]


@dataclass(frozen=True)
class TimeFinding:
    """One kind of offence of a time column.

    Attributes:
        code: One of :data:`FINDING_CODES`.
        count: How many rows (or source pairs) offend.
        lines: The first :data:`MAX_LISTED_LINES` offending physical lines.
        measures: Per listed line, the offending step (gaps and reversals)
            or distance from the fitted line (residuals) in sample
            intervals; empty when a line has no such measure.
        statement: The finding in plain words, with its statistics.
    """

    code: str
    count: int
    lines: tuple[int, ...]
    measures: tuple[float, ...]
    statement: str


@dataclass(frozen=True)
class TimeBaseAssessment:
    """The assessment of one time column.

    Attributes:
        outcome: One of :data:`OUTCOMES`.
        evidence: One of :data:`EVIDENCE_LEVELS`.
        regime: One of :data:`REGIMES`.
        time_unit: The declared unit of the stamps.
        row_count: Number of stamps.
        first_line: Physical line of the first stamp.
        last_line: Physical line of the last stamp.
        resolution: One tick of the assessment grid, in *time_unit*.
        effective_resolution: The resolution the values actually carry
            (coarser than *resolution* only for float32 values).
        float32_values: The stamps are float32 values coarser than their
            print.
        duration: Last stamp minus first, in *time_unit*.
        modal_steps: The modal pair ``(m, m + 1)`` of tick steps.
        reference_interval_ticks: ``b``, the mean step of the modal pair.
        step_range: Smallest and largest tick step.
        max_residual_ticks: Largest distance from the fitted line (None
            when no line was fitted).
        rms_residual_ticks: Root mean square of those distances.
        fitted_rate_hz: The least-squares rate, :data:`RATE_DIGITS`
            significant digits (None at ``duration_only`` or before the step
            tests pass).
        rate_uncertainty_hz: Worst-case effect on it of the print rounding
            alone, rounded up to two significant digits.
        proposed_rate_hz: The fitted rate in canonical configuration form,
            when the column supports proposing it.
        rate_sources: The rate sources given and found.
        rate_comparisons: Every pairwise comparison made.
        findings: The findings, in :data:`FINDING_CODES` order.
    """

    outcome: str
    evidence: str
    regime: str
    time_unit: str
    row_count: int
    first_line: int
    last_line: int
    resolution: str
    effective_resolution: str
    float32_values: bool
    duration: str
    modal_steps: tuple[int, int]
    reference_interval_ticks: float
    step_range: tuple[int, int]
    max_residual_ticks: Optional[float]
    rms_residual_ticks: Optional[float]
    fitted_rate_hz: Optional[str]
    rate_uncertainty_hz: Optional[str]
    proposed_rate_hz: Optional[str]
    rate_sources: tuple[RateSource, ...]
    rate_comparisons: tuple[RateComparison, ...]
    findings: tuple[TimeFinding, ...]

    @property
    def needs_rate_basis(self) -> bool:
        """The rate needs a stated or file-declared basis (R12, R14)."""
        return self.proposed_rate_hz is None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _quoted(cited: FileText) -> str:
    """File text as a refusal cites it: escaped, capped and quoted, with its own
    quotes escaped so it cannot close the quotation (as `import_config`)."""
    text = cited.file_text.replace("'", "\\'")
    return f"'{text}{'...' if cited.truncated else ''}'"


def _token_refusal(token: str, line: int, decimal_mark: str) -> TimeBaseRefused:
    stripped = token.strip(" ")
    words = stripped.lower().lstrip("+-")
    datetime_text = _DATETIME.fullmatch(stripped) is not None
    if not stripped:
        problem = "is an empty field"
    elif datetime_text:
        problem = "is a date or a time of day, which this version does not assess"
    elif words in _NON_FINITE_WORDS:
        problem = "is not a finite number"
    elif len(stripped) > TIME_TOKEN_BYTES:
        problem = f"is longer than {TIME_TOKEN_BYTES} characters"
    elif decimal_mark == "." and "," in stripped:
        problem = "has a ',' in a dot-decimal column"
    elif decimal_mark == "," and "." in stripped:
        problem = "has a '.' in a decimal-comma column"
    else:
        problem = (
            f"is not a time stamp the import reads: a decimal number with at most "
            f"{TIME_MAX_DIGITS} significant digits and an exponent of at most "
            f"{TIME_MAX_EXPONENT_DIGITS} digits"
        )
    cited = FileText.of(token, cap=TOKEN_CHARS)
    shown = _quoted(cited)
    return TimeBaseRefused(
        "time_grammar",
        f"Line {line}: the time stamp {shown} {problem} \N{EM DASH} {_GRAMMAR_REMEDY}.",
        line=line,
        token=cited,
        datetime_text=datetime_text,
    )


def parse_time_tokens(
    tokens: Sequence[str], *, decimal_mark: str = ".", first_line: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Parse time tokens exactly, by the decoder's own grammar.

    For tokens already in hand (an inspection preview); a full decode
    produces the same arrays (``text_decoder``).

    Args:
        tokens: One stamp per row, as text; spaces at either end are allowed.
        decimal_mark: ``"."`` or ``","``.
        first_line: Physical line of the first token.

    Returns:
        ``(mantissa, exponent)``: int64 digits and int16 decimal exponent of
        each stamp.

    Raises:
        TimeBaseRefused: ``time_grammar`` for the first token that is not a
            plain decimal stamp within the caps (NaN, ``inf``, empty, the
            wrong decimal mark, a date or time of day, over
            :data:`TIME_TOKEN_BYTES` characters, too many digits, an
            exponent of more than three digits), naming its line.
        TypeError: For a token that is not text.
        ValueError: For an unknown decimal mark.
    """
    if decimal_mark not in DECIMAL_MARKS:
        raise ValueError(
            f"decimal_mark must be one of {DECIMAL_MARKS}, got {decimal_mark!r}"
        )
    encoded: list[bytes] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise TypeError(f"time tokens must be str, got {type(token).__name__}")
        stripped = token.strip(" ")
        if len(stripped) > TIME_TOKEN_BYTES or _PRINTABLE.fullmatch(stripped) is None:
            raise _token_refusal(token, first_line + index, decimal_mark)
        raw = stripped.encode("ascii")
        if decimal_mark == ",":
            if b"." in raw:
                raise _token_refusal(token, first_line + index, decimal_mark)
            raw = raw.replace(b",", b".")
        encoded.append(raw)
    if not encoded:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16)
    width = max(1, max(len(raw) for raw in encoded))
    matrix = np.frombuffer(
        b"".join(raw.ljust(width, b"\0") for raw in encoded), dtype=np.uint8
    ).reshape(len(encoded), width)
    mantissa, exponent, valid = parse_time_matrix(matrix)
    refused = np.flatnonzero(~valid)
    if refused.size:
        index = int(refused[0])
        raise _token_refusal(tokens[index], first_line + index, decimal_mark)
    return mantissa, exponent


# ---------------------------------------------------------------------------
# Rate evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    source: str
    rate: Decimal
    low: Decimal
    high: Decimal


def _display(value: Decimal) -> str:
    """At most :data:`RATE_DIGITS` significant digits, plain, no trailing zeros."""
    with decimal.localcontext(_RATE_CONTEXT) as ctx:
        ctx.prec = RATE_DIGITS
        rounded = (+value).normalize()
    return "0" if rounded.is_zero() else format(rounded, "f")


def _with_digits(value: Decimal, digits: int, rounding: str) -> str:
    """Exactly *digits* significant digits, plain notation."""
    with decimal.localcontext(_RATE_CONTEXT) as ctx:
        ctx.prec = digits
        ctx.rounding = rounding
        rounded = +value
        exact = rounded.quantize(Decimal(1).scaleb(rounded.adjusted() - digits + 1))
    return format(exact, "f")


def _stated_rate(value: object) -> Optional[_Source]:
    if value is None:
        return None
    text = normalize_decimal(value, field="the stated sampling rate")
    rate = Decimal(text)
    if rate <= 0:
        raise ValueError(f"the stated sampling rate must be above zero, got {text}")
    return _Source("stated", rate, rate, rate)


def _declared_rate(declared: Optional[DeclaredTimeBase]) -> Optional[_Source]:
    if declared is None:
        return None
    if not isinstance(declared, DeclaredTimeBase):
        raise TypeError(
            f"declared must be a DeclaredTimeBase, got {type(declared).__name__}"
        )
    if declared.kind not in ("rate", "interval"):
        raise ValueError(
            f"a declared time base is a 'rate' or an 'interval', got {declared.kind!r}"
        )
    if declared.time_unit not in TIME_UNITS:
        raise ValueError(f"time unit {declared.time_unit!r} is not one of {TIME_UNITS}")
    token = declared.token
    cited = FileText.of(token, cap=TOKEN_CHARS) if isinstance(token, str) else None
    shown = _quoted(cited) if cited else "(not text)"
    try:
        normalize_decimal(token, field="the declared value")
        with decimal.localcontext(_RATE_CONTEXT):
            value = Decimal(token)  # validated just above
        if value <= 0:
            raise ValueError("is not above zero")
    except ValueError as exc:
        reason = str(exc).replace("the declared value ", "", 1)
        raise TimeBaseRefused(
            "time_grammar",
            f"The file declares a sample {declared.kind} {shown} that the import cannot "
            f"use: {reason} \N{EM DASH} state the sampling rate, or correct the "
            f"declaration.",
            token=cited,
        ) from None
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    with decimal.localcontext(_RATE_CONTEXT):
        half = Decimal(5).scaleb(exponent - 1)
        if declared.kind == "rate":
            return _Source("file_declared", value, value - half, value + half)
        unit = _UNIT_EXPONENT[declared.time_unit]
        seconds = value.scaleb(unit)
        spread = half.scaleb(unit)
        return _Source(
            "file_declared",
            1 / seconds,
            1 / (seconds + spread),
            1 / (seconds - spread),
        )


# ---------------------------------------------------------------------------
# Print regime and grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Print:
    regime: str
    grid: int  # power of ten of one tick, in the time unit
    base: float  # rounding of a stamp printed on the grid, in ticks
    quantum: Optional[np.ndarray]  # per-row quantum exponents (significant digits)


def _chunks(rows: int) -> Iterator[tuple[int, int]]:
    for lo in range(0, rows, _CHUNK_ROWS):
        yield lo, min(lo + _CHUNK_ROWS, rows)


def _digit_counts(mantissa: np.ndarray) -> np.ndarray:
    """Decimal digits of each mantissa (0 for zero)."""
    digits: np.ndarray = np.searchsorted(_POW10, np.abs(mantissa), side="right")
    return digits


def _largest_value(mantissa: np.ndarray, exponent: np.ndarray) -> float:
    """The largest stamp magnitude, as a float."""
    top = 0.0
    with np.errstate(over="ignore"):
        for lo, hi in _chunks(mantissa.size):
            values = (
                np.abs(mantissa[lo:hi].astype(np.float64)) * 10.0 ** exponent[lo:hi]
            )
            top = max(top, float(values.max()))
    if not math.isfinite(top):
        raise TimeBaseRefused(
            "time_grammar",
            f"The time stamps reach beyond the float64 range \N{EM DASH} {_GRAMMAR_REMEDY}.",
        )
    return top


def _print_model(mantissa: np.ndarray, exponent: np.ndarray) -> _Print:
    """Step 1 of the module docstring."""
    rows = mantissa.size
    finest = int(exponent.min())
    widest = 0
    cut = False  # some row has `widest` digits and stops above the finest place
    for lo, hi in _chunks(rows):
        digits = _digit_counts(mantissa[lo:hi])
        widest = max(widest, int(digits.max()))
    if widest == 0:  # every stamp is zero
        return _Print("fixed_decimals", finest, 0.5, None)
    # Exponents are within int16, and so is every quantum.
    quantum = np.empty(rows, dtype=np.int16)
    for lo, hi in _chunks(rows):
        part = mantissa[lo:hi]
        digits = _digit_counts(part)
        lead = exponent[lo:hi] + digits - 1
        quantum[lo:hi] = np.where(
            part != 0, np.maximum(lead - widest + 1, finest), finest
        )
        cut = cut or bool(np.any((digits == widest) & (exponent[lo:hi] > finest)))
    low, high = int(quantum.min()), int(quantum.max())
    spacing = float(np.spacing(_largest_value(mantissa, exponent)))
    if high < math.log10(spacing):
        # Digits beyond float64 precision: the float summation bound sets the
        # grid, one decade above it (shortest repr, %.17g, over-long prints).
        bound = (rows + 2) * spacing
        grid = max(math.ceil(math.log10(bound)) + 1, finest)
        return _Print("shortest_repr", grid, 0.5 + bound / 10.0**grid, None)
    if low != high and cut:
        # A row cut at `widest` significant digits above the finest decimal
        # place: a significant-digit print. The grid is the commonest quantum.
        counts = np.bincount(quantum.astype(np.int64) - low)
        grid = low + int(np.flatnonzero(counts == counts.max())[-1])
        return _Print("significant_digits", grid, 0.5, quantum)
    # Fixed decimals, trailing zeros stripped or not: every stamp has the
    # finest place, whatever it printed.
    return _Print("fixed_decimals", finest, 0.5, None)


def _on_grid(
    mantissa: np.ndarray, exponent: np.ndarray, grid: int, *, first_line: int
) -> np.ndarray:
    """Each stamp as int64 ticks of ``10**grid``, rounded half to even."""
    if int(exponent.min()) == int(exponent.max()) == grid:
        return mantissa  # within 10**18, so every difference fits
    ticks = np.empty(mantissa.size, dtype=np.int64)
    for lo, hi in _chunks(mantissa.size):
        shift = exponent[lo:hi].astype(np.int64) - grid
        part = mantissa[lo:hi]
        # Up: exact multiplication, refused before it could overflow.
        places = np.clip(shift, 0, 18)
        limit = np.where(shift > 18, 0, _TICK_LIMIT // _POW10[places])
        over = np.flatnonzero((shift >= 0) & (np.abs(part) > limit))
        if over.size:
            line = first_line + lo + int(over[0])
            raise TimeBaseRefused(
                "time_grammar",
                f"Line {line}: the time stamp is too far in scale from the others to "
                f"share one grid of ticks \N{EM DASH} {_GRAMMAR_REMEDY}.",
                line=line,
            )
        raised = part * _POW10[places]
        # Down: exact division, rounded half to even. Beyond 18 places every
        # mantissa (under 10**18) rounds to zero.
        divisor = _POW10[np.clip(-shift, 0, 18)]
        quotient = part // divisor
        twice = 2 * (part - quotient * divisor)
        quotient += (twice > divisor) | ((twice == divisor) & (quotient % 2 == 1))
        lowered = np.where(shift < -18, 0, quotient)
        ticks[lo:hi] = np.where(shift >= 0, raised, lowered)
    return ticks


def _float32_spacing(ticks: np.ndarray, grid: int) -> Optional[float]:
    """The float32 spacing at the largest stamp, when every stamp is a
    float32 value and that spacing is coarser than the decimals the print
    actually uses (step 3 of the module docstring).

    The test is exact: with at most 12 decimals and ticks below 2**53, the
    nearest float32 times ``10**-grid`` and its distance from the tick are
    exact in float64, so a print that lies exactly half a tick from its
    float32 value (a tie) is still recognized. Other grids are not tested.
    """
    top = max(int(ticks.max()), -int(ticks.min()))
    if top == 0 or top >= 1 << 53 or not -12 <= grid <= 0:
        return None
    zeros = 18  # decimal zeros every nonzero stamp ends with
    for lo, hi in _chunks(ticks.size):
        part = ticks[lo:hi]
        nonzero = part[part != 0]
        found = 0
        while (
            found < zeros and nonzero.size and not np.any(nonzero % _POW10[found + 1])
        ):
            found += 1
        zeros = min(zeros, found) if nonzero.size else zeros
        if not zeros:
            break
    scale = 10.0**-grid  # exact: at most 10**12
    with np.errstate(over="ignore"):
        spacing = float(np.spacing(np.float32(top / scale)))
    if not math.isfinite(spacing) or spacing * scale <= 10.0**zeros:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        for lo, hi in _chunks(ticks.size):
            exact = ticks[lo:hi].astype(np.float64)
            nearest = (exact / scale).astype(np.float32).astype(np.float64)
            if not np.all(np.abs(nearest * scale - exact) <= 0.5):
                return None
    return spacing


# ---------------------------------------------------------------------------
# The column on its grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Column:
    """One column on its grid, with what every rule reads.

    Every stamp has a rounding half-width ``base + extra``: ``base`` for a
    stamp printed on the grid; the extra is its own coarser quantum, or its
    rounding onto the grid when finer (significant digits), plus half its
    float32 spacing (float32 values).
    """

    printed: _Print
    ticks: np.ndarray
    steps: np.ndarray
    m: int  # the modal pair is (m, m + 1)
    reference: Fraction  # b, exact
    spacing: Optional[float]  # float32 spacing at the largest stamp
    largest_extra: float
    evidence: str
    first_line: int
    tick_exponent: int  # one tick is 10**tick_exponent seconds

    @property
    def rows(self) -> int:
        return int(self.ticks.size)

    @property
    def b(self) -> float:
        return float(self.reference)

    @property
    def base(self) -> float:
        return self.printed.base

    @property
    def tick_bound(self) -> float:
        """Print and fit: one tick for fixed decimals."""
        return 2 * self.printed.base

    @property
    def span(self) -> int:
        return int(self.ticks[-1]) - int(self.ticks[0])

    @property
    def last_line(self) -> int:
        return self.first_line + self.rows - 1

    def extra(self, lo: int, hi: int) -> "np.ndarray | float":
        return _extra(self.printed, self.ticks, self.spacing is not None, lo, hi)

    def step_extra(self, lo: int, hi: int) -> "np.ndarray | float":
        """Extra width of steps ``lo`` to ``hi - 1`` (step ``i`` joins rows
        ``i`` and ``i + 1``)."""
        rows = self.extra(lo, hi + 1)
        if isinstance(rows, float):
            return 2 * rows
        widths: np.ndarray = rows[:-1] + rows[1:]
        return widths


def _extra(
    printed: _Print, ticks: np.ndarray, float32: bool, lo: int, hi: int
) -> "np.ndarray | float":
    extra: "np.ndarray | float" = 0.0
    if printed.quantum is not None:
        shift = printed.quantum[lo:hi].astype(np.float64) - printed.grid
        power = 10.0**shift
        extra = np.where(
            shift > 0, 0.5 * (power - 1), np.where(shift < 0, 0.5 * power, 0.0)
        )
    if float32:
        scale = 10.0**-printed.grid
        values = (ticks[lo:hi].astype(np.float64) / scale).astype(np.float32)
        extra = extra + 0.5 * np.abs(np.spacing(values)).astype(np.float64) * scale
    return extra


def _column(
    mantissa: np.ndarray, exponent: np.ndarray, time_unit: str, first_line: int
) -> _Column:
    printed = _print_model(mantissa, exponent)
    tick_exponent = printed.grid + _UNIT_EXPONENT[time_unit]
    if not _TICK_EXPONENTS[0] <= tick_exponent <= _TICK_EXPONENTS[1]:
        raise TimeBaseRefused(
            "time_grammar",
            f"The time stamps resolve 1e{tick_exponent} s, outside the 1e"
            f"{_TICK_EXPONENTS[0]} s to 1e{_TICK_EXPONENTS[1]} s the assessment "
            f"holds \N{EM DASH} {_GRAMMAR_REMEDY}.",
        )
    ticks = _on_grid(mantissa, exponent, printed.grid, first_line=first_line)
    steps = np.diff(ticks)
    spacing = _float32_spacing(ticks, printed.grid)
    largest_extra = 0.0
    clean: Optional[list[np.ndarray]] = None
    if printed.quantum is not None or spacing is not None:
        clean = []
        for lo, hi in _chunks(ticks.size):
            rows = _extra(printed, ticks, spacing is not None, lo, hi)
            largest_extra = max(largest_extra, float(np.max(rows)))
        for lo, hi in _chunks(steps.size):
            rows = _extra(printed, ticks, spacing is not None, lo, hi + 1)
            assert not isinstance(rows, float)
            clean.append((rows[:-1] + rows[1:]) == 0)
    m, total, count = _modal_pair(
        steps, None if clean is None else np.concatenate(clean)
    )
    reference = Fraction(total, count)
    if spacing is not None or reference < 1:
        evidence = "duration_only"
    elif reference < 2:
        evidence = "partial"
    else:
        evidence = "full"
    return _Column(
        printed=printed,
        ticks=ticks,
        steps=steps,
        m=m,
        reference=reference,
        spacing=spacing,
        largest_extra=largest_extra,
        evidence=evidence,
        first_line=first_line,
        tick_exponent=tick_exponent,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _histogram_pair(steps: np.ndarray) -> tuple[int, int, int]:
    low, high = int(steps.min()), int(steps.max())
    if high - low <= _HISTOGRAM_SPAN:
        counts = np.zeros(high - low + 2, dtype=np.int64)
        for lo, hi in _chunks(steps.size):
            counts += np.bincount(steps[lo:hi] - low, minlength=counts.size)
        scores = np.where(counts[:-1] > 0, counts[:-1] + counts[1:], -1)
        index = int(np.argmax(scores))
        return low + index, int(counts[index]), int(counts[index + 1])
    values, counts = np.unique(steps, return_counts=True)
    following = np.zeros(counts.size, dtype=np.int64)
    adjacent = values[1:] == values[:-1] + 1
    following[:-1] = np.where(adjacent, counts[1:], 0)
    index = int(np.argmax(counts + following))
    return int(values[index]), int(counts[index]), int(following[index])


def _modal_pair(steps: np.ndarray, clean: Optional[np.ndarray]) -> tuple[int, int, int]:
    """``(m, S, C)``: the modal pair ``{m, m+1}`` of steps that occur, the sum
    and the count of the steps in it; ties go to the smaller ``m``.

    Only steps between two stamps printed on the grid count, when there are
    any (*clean*). A pair of zero steps alone, next to positive steps, is not
    a coarse print but repeated stamps: the pair is then taken among the
    positive steps, so the repeats are refused as duplicates.
    """
    chosen = steps if clean is None or not clean.any() else steps[clean]
    m, below, above = _histogram_pair(chosen)
    if m == 0 and above == 0 and chosen.max() > 0:
        m, below, above = _histogram_pair(chosen[chosen > 0])
    return m, m * below + (m + 1) * above, below + above


@dataclass
class _Offence:
    """Count, first lines and measures of one kind of offending step."""

    count: int = 0
    lines: tuple[int, ...] = ()
    measures: tuple[float, ...] = ()


def _offending_steps(
    column: _Column,
    test: Callable[[np.ndarray, "np.ndarray | float"], np.ndarray],
    *,
    measured: bool,
) -> _Offence:
    """The steps *test* flags, in chunks; step ``i`` ends at row ``i + 1``."""
    found = _Offence()
    lines: list[int] = []
    measures: list[float] = []
    for lo, hi in _chunks(column.steps.size):
        part = column.steps[lo:hi]
        flagged = np.flatnonzero(test(part, column.step_extra(lo, hi)))
        found.count += int(flagged.size)
        for i in flagged[: MAX_LISTED_LINES - len(lines)]:
            lines.append(column.first_line + 1 + lo + int(i))
            if measured:
                measures.append(float(Fraction(int(part[i])) / column.reference))
    found.lines, found.measures = tuple(lines), tuple(measures)
    return found


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _in_pair(
    column: _Column, part: np.ndarray, extra: "np.ndarray | float"
) -> np.ndarray:
    """Steps a uniform clock can print: the modal pair itself between two
    stamps on the grid; within one tick of ``b`` widened by the stamps'
    extra rounding otherwise. A backward step never is."""
    m, b = column.m, column.b
    exact = (part == m) | (part == m + 1)
    widened = (part >= 0) & (part >= b - 1 - extra) & (part <= b + 1 + extra)
    flagged: np.ndarray = np.where(np.asarray(extra) == 0, exact, widened)
    return flagged


def _step_findings(column: _Column) -> list[TimeFinding]:
    """Step 6 of the module docstring."""
    b = column.b
    if column.evidence == "duration_only":
        # Rounding never reverses a clock. A gap is refused only when it is
        # one whatever the rounding: the shortest true step against the
        # longest true interval (whose estimate float32 rounding may bias).
        spread = 0.0
        if column.spacing is not None:
            spread = column.tick_bound + 2 * column.largest_extra
        rules = [
            (
                "time_reversal",
                lambda part, extra: part < 0,
                "earlier than the stamp before",
                False,
            )
        ]
        if b > 0:
            rules.insert(
                0,
                (
                    "time_gap",
                    lambda part, extra: part - (column.tick_bound + extra)
                    >= 1.5 * (b + spread),
                    "of more than 1.5 sample intervals whatever the print's "
                    "rounding: samples are missing",
                    True,
                ),
            )
    else:
        # Exact thresholds of integer steps against b = S/C between stamps on
        # the grid; a step of the modal pair is the reference itself, never
        # an offence (this only matters at partial evidence, where 1.5 b can
        # fall below m + 1). A stamp's extra rounding widens both.
        gap_from = math.ceil(3 * column.reference / 2)
        reversal_below = math.ceil(column.reference / 2)

        def gap(part: np.ndarray, extra: "np.ndarray | float") -> np.ndarray:
            flagged: np.ndarray = np.where(
                np.asarray(extra) == 0, part >= gap_from, part - extra >= 1.5 * b
            ) & ~_in_pair(column, part, extra)
            return flagged

        def reversal(part: np.ndarray, extra: "np.ndarray | float") -> np.ndarray:
            short = np.where(
                np.asarray(extra) == 0, part < reversal_below, part + extra < 0.5 * b
            )
            flagged: np.ndarray = ((part < 0) | short) & ~_in_pair(column, part, extra)
            return flagged

        rules = [
            (
                "time_gap",
                gap,
                "of 1.5 sample intervals or more: samples are missing",
                True,
            ),
            (
                "time_reversal",
                reversal,
                "below half a sample interval: a repeated, reordered or restarted "
                "stamp, or a change of rate",
                True,
            ),
        ]
    findings = []
    for code, test, words, measured in rules:
        offence = _offending_steps(column, test, measured=measured)
        if offence.count:
            findings.append(
                TimeFinding(
                    code,
                    offence.count,
                    offence.lines,
                    offence.measures,
                    f"{_plural(offence.count, 'step')} {words}; first at line "
                    f"{offence.lines[0]}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass
class _Fit:
    slope: float  # ticks per sample
    max_residual: float
    rms_residual: float
    defects: int
    defect_lines: tuple[int, ...]
    defect_measures: tuple[float, ...]
    off: int
    off_lines: tuple[int, ...]
    stated_spread: Optional[tuple[float, float]]


def _fit(column: _Column, stated_beta: Optional[float]) -> _Fit:
    """The least-squares line through the stamps and the residual, pattern
    and stated-rate tests against it, in chunks (steps 7 to 9). Deviations
    from ``i * m`` are built by an integer running sum (exact, never
    overflowing), and only the small remainder is floating point."""
    rows, steps, m, b = column.rows, column.steps, column.m, column.b
    centre = (rows - 1) / 2
    spread = rows * (rows * rows - 1) / 12
    residual_bound = max(b / 2, column.tick_bound)

    def deviations() -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        carry = 0
        for lo, hi in _chunks(rows):
            part = steps[max(lo - 1, 0) : hi - 1] - m
            running = np.cumsum(part, dtype=np.int64) + carry
            exact = np.concatenate([[carry], running]) if lo == 0 else running
            carry = int(running[-1]) if running.size else carry
            index = np.arange(lo, hi, dtype=np.float64)
            yield lo, hi, index, exact.astype(np.float64) - index * (b - m)

    total = moment = 0.0
    for _, _, index, error in deviations():
        total += float(error.sum())
        moment += float(((index - centre) * error).sum())
    tilt = moment / spread
    level = total / rows

    largest = squares = 0.0
    defects = off = 0
    defect_lines: list[int] = []
    defect_measures: list[float] = []
    off_lines: list[int] = []
    high = -math.inf
    low = math.inf
    for lo, hi, index, error in deviations():
        extra = column.extra(lo, hi)
        residual = error - level - tilt * (index - centre)
        distance = np.abs(residual)
        largest = max(largest, float(distance.max()))
        squares += float((residual * residual).sum())
        defect = distance > residual_bound + extra
        defects += int(defect.sum())
        for i in np.flatnonzero(defect)[: MAX_LISTED_LINES - len(defect_lines)]:
            defect_lines.append(column.first_line + lo + int(i))
            defect_measures.append(float(distance[i]) / b)
        outside = np.asarray(distance > column.tick_bound + extra)
        first = max(lo - 1, 0)
        step_part = steps[first : hi - 1]
        stepped = ~_in_pair(column, step_part, column.step_extra(first, hi - 1))
        if lo == 0:
            outside[1:] |= stepped
        else:
            outside |= stepped
        off += int(outside.sum())
        for i in np.flatnonzero(outside)[: MAX_LISTED_LINES - len(off_lines)]:
            off_lines.append(column.first_line + lo + int(i))
        if stated_beta is not None:
            # A stated rate must predict every stamp within its own rounding
            # plus the print-and-fit half: one tick for fixed decimals.
            predicted = error + index * (b - stated_beta)
            width = column.tick_bound + extra
            high = max(high, float(np.max(predicted - width)))
            low = min(low, float(np.min(predicted + width)))
    return _Fit(
        slope=b + tilt,
        max_residual=largest,
        rms_residual=math.sqrt(squares / rows),
        defects=defects,
        defect_lines=tuple(defect_lines),
        defect_measures=tuple(defect_measures),
        off=off,
        off_lines=tuple(off_lines),
        stated_spread=(high, low) if stated_beta is not None else None,
    )


def _beta(rate: Decimal, tick_exponent: int) -> Decimal:
    """Ticks per sample at *rate*."""
    with decimal.localcontext(_RATE_CONTEXT):
        return Decimal(1).scaleb(-tick_exponent) / rate


def _ppm(reference: Optional[Decimal], other: Decimal) -> Optional[float]:
    if reference is None:
        return None
    with decimal.localcontext(_RATE_CONTEXT):
        return float((other - reference) / reference * 1_000_000)


def _fit_findings(
    column: _Column, stated: Optional[_Source]
) -> tuple[_Fit, Optional[TimeFinding]]:
    """Steps 7 and 8 of the module docstring, once the step tests passed."""
    stated_beta = None
    if stated is not None:
        stated_beta = float(_beta(stated.rate, column.tick_exponent))
    fit = _fit(column, stated_beta)
    b = column.b
    if fit.defects:
        return fit, TimeFinding(
            "time_residual",
            fit.defects,
            fit.defect_lines,
            fit.defect_measures,
            f"{_plural(fit.defects, 'stamp')} more than half a sample interval from "
            f"the fitted line (largest {fit.max_residual / b:.3g} intervals): the "
            f"clock drifts or changes rate; if the column is not the sampling "
            f"clock, leave it unassessed with a reason",
        )
    if fit.off:
        return fit, TimeFinding(
            "time_off_pattern",
            fit.off,
            fit.off_lines,
            (),
            f"{_plural(fit.off, 'stamp')} off a uniform clock (steps from "
            f"{int(column.steps.min())} to {int(column.steps.max())} ticks against "
            f"{column.m} and {column.m + 1}; largest residual "
            f"{fit.max_residual:.3g} ticks, RMS residual {fit.rms_residual:.3g} "
            f"ticks, bound {column.tick_bound:.3g}): jitter or wander; choose the "
            f"time-base source",
        )
    return fit, None


def _fitted_rate(column: _Column, fit: _Fit) -> tuple[str, Optional[_Source]]:
    """The fitted rate with :data:`RATE_DIGITS` digits, checked with
    ``validate_sampling_rate``, and the interval the print's rounding alone
    allows it (None when the rounding could reverse the slope)."""
    rows = column.rows
    with decimal.localcontext(_RATE_CONTEXT):
        one = Decimal(1).scaleb(-column.tick_exponent)
        slope = Decimal(fit.slope)
        fitted = one / slope
        # Worst case of the least-squares slope when every stamp is off by up
        # to its half-width: half * sum |w_i|, in closed form.
        weights = Fraction(12 * (rows * rows // 4), rows * (rows * rows - 1))
        error = Fraction(column.base + column.largest_extra) * weights
        worst = Decimal(error.numerator) / Decimal(error.denominator)
        bounds = (
            (one / (slope + worst), one / (slope - worst)) if slope > worst else None
        )
    fitted_text = _with_digits(fitted, RATE_DIGITS, decimal.ROUND_HALF_EVEN)
    try:
        validate_sampling_rate(
            float(fitted_text), name="derived sampling rate", source="the time column"
        )
    except InvalidInputError:
        raise TimeBaseRefused(
            "time_grammar",
            f"The time stamps give a sampling rate of {fitted_text} Hz, which is not "
            f"a usable rate \N{EM DASH} {_GRAMMAR_REMEDY}.",
        ) from None
    if bounds is None:
        return fitted_text, None
    return fitted_text, _Source("time_column", Decimal(fitted_text), *bounds)


def _column_comparison(
    column: _Column, fit: Optional[_Fit], own: Optional[_Source], other: _Source
) -> RateComparison:
    """The column against one other rate source (step 9)."""
    if column.evidence == "duration_only":
        # Only the total duration, within one effective tick: the rounding
        # of the first and the last stamp.
        first = column.extra(0, 1)
        last = column.extra(column.rows - 1, column.rows)
        tolerance = Decimal(
            column.tick_bound + float(np.max(first)) + float(np.max(last))
        )
        with decimal.localcontext(_RATE_CONTEXT):
            longest = (column.rows - 1) * _beta(other.low, column.tick_exponent)
            shortest = (column.rows - 1) * _beta(other.high, column.tick_exponent)
            consistent = shortest - tolerance <= column.span <= longest + tolerance
            implied = (
                Decimal(column.rows - 1).scaleb(-column.tick_exponent) / column.span
                if column.span > 0
                else None
            )
        return RateComparison(
            "time_column", other.source, consistent, _ppm(implied, other.rate)
        )
    assert fit is not None
    if other.source == "stated" and fit.stated_spread is not None:
        high, low = fit.stated_spread
        consistent = high <= low
    elif own is None:
        consistent = True  # the column bounds no rate: nothing to contradict
    else:
        consistent = own.low <= other.high and other.low <= own.high
    fitted = own.rate if own is not None else None
    return RateComparison(
        "time_column", other.source, consistent, _ppm(fitted, other.rate)
    )


def _disagreement(comparisons: Sequence[RateComparison]) -> Optional[TimeFinding]:
    disagreements = [c for c in comparisons if not c.consistent]
    if not disagreements:
        return None
    return TimeFinding(
        "rate_sources_disagree",
        len(disagreements),
        (),
        tuple(c.difference_ppm or 0.0 for c in disagreements),
        "the rate sources disagree beyond the precision each carries: "
        + "; ".join(
            (
                f"{c.second} lies {c.difference_ppm:+.6g} ppm from {c.first}"
                if c.difference_ppm is not None
                else f"{c.second} does not fit {c.first}"
            )
            for c in disagreements
        )
        + "; choose the time-base source",
    )


def _in_unit(value: Decimal) -> str:
    return format(value, "f")


def _arrays(
    mantissa: np.ndarray, exponent: np.ndarray, time_unit: str, first_line: int
) -> tuple[np.ndarray, np.ndarray]:
    if time_unit not in TIME_UNITS:
        raise ValueError(f"time unit {time_unit!r} is not one of {TIME_UNITS}")
    if (
        isinstance(first_line, bool)
        or not isinstance(first_line, int)
        or first_line < 1
    ):
        raise ValueError(f"first_line must be a line number from 1, got {first_line!r}")
    mantissa = np.asarray(mantissa)
    exponent = np.asarray(exponent)
    if (
        mantissa.ndim != 1
        or exponent.shape != mantissa.shape
        or mantissa.dtype.kind not in "iu"
        or exponent.dtype.kind not in "iu"
    ):
        raise ValueError(
            "mantissa and exponent must be 1-D integer arrays of one length"
        )
    if mantissa.size < 2:
        raise TimeBaseRefused(
            "region_too_short",
            f"The time column holds {_plural(int(mantissa.size), 'stamp')}; a time "
            f"base needs at least two \N{EM DASH} widen the data region, or leave "
            f"the column unassessed.",
        )
    # The decoder's own ranges: at most 18 digits, exponents within int16.
    if (
        int(mantissa.max()) > _MAX_MANTISSA
        or int(mantissa.min()) < -_MAX_MANTISSA
        or int(exponent.max()) > _MAX_EXPONENT
        or int(exponent.min()) < -_MAX_EXPONENT
    ):
        raise ValueError(
            "mantissa must hold at most 18 digits and exponent lie within "
            f"+-{_MAX_EXPONENT}, as text_decoder produces them"
        )
    return mantissa.astype(np.int64), exponent.astype(np.int16)


def assess_time_base(
    mantissa: np.ndarray,
    exponent: np.ndarray,
    *,
    time_unit: str,
    first_line: int,
    stated_rate_hz: Optional[str] = None,
    declared: Optional[DeclaredTimeBase] = None,
) -> TimeBaseAssessment:
    """Assess one time column by the rules of KTD9.

    Args:
        mantissa: int64 digits of each stamp (``text_decoder`` output).
        exponent: Decimal exponent of each stamp's last digit.
        time_unit: The declared unit: ``s``, ``ms``, ``us`` or ``ns``.
        first_line: Physical line of the first stamp; row ``i`` is line
            ``first_line + i``.
        stated_rate_hz: A stated sampling rate, as a decimal string, to
            check against the column.
        declared: A rate or interval printed in the file.

    Returns:
        The assessment.

    Raises:
        TimeBaseRefused: ``region_too_short`` for fewer than two stamps;
            ``time_grammar`` for stamps the assessment cannot hold or an
            unusable declared value.
        ValueError: For a malformed argument, including a stated rate that
            is not a positive decimal string.
    """
    mantissa, exponent = _arrays(mantissa, exponent, time_unit, first_line)
    stated = _stated_rate(stated_rate_hz)
    file_declared = _declared_rate(declared)
    column = _column(mantissa, exponent, time_unit, first_line)

    findings = _step_findings(column)
    fit: Optional[_Fit] = None
    if column.evidence != "duration_only" and not findings:
        fit, fit_finding = _fit_findings(column, stated)
        if fit_finding is not None:
            findings.append(fit_finding)
    fitted_text: Optional[str] = None
    own: Optional[_Source] = None
    uncertainty_text: Optional[str] = None
    if fit is not None:
        fitted_text, own = _fitted_rate(column, fit)
    if own is not None:
        with decimal.localcontext(_RATE_CONTEXT):
            uncertainty = own.high - own.rate  # the wider side of 1 / slope
        uncertainty_text = _with_digits(uncertainty, 2, decimal.ROUND_UP)

    # A column with a finding is no rate evidence: jitter, a gap or a defect
    # needs a choice or a refusal first.
    comparisons = [
        _column_comparison(column, fit, own, other)
        for other in (file_declared, stated)
        if other is not None and not findings
    ]
    if file_declared is not None and stated is not None:
        comparisons.append(
            RateComparison(
                "file_declared",
                "stated",
                file_declared.low <= stated.rate <= file_declared.high,
                _ppm(file_declared.rate, stated.rate),
            )
        )
    disagreement = _disagreement(comparisons)
    if disagreement is not None:
        findings.append(disagreement)
    findings.sort(key=lambda f: FINDING_CODES.index(f.code))

    codes = {f.code for f in findings}
    if codes & {"time_gap", "time_reversal", "time_residual"}:
        outcome = "invalid"
    elif codes:
        outcome = "needs_confirmation"
    else:
        outcome = "accepted"
    proposed = None
    if (
        outcome == "accepted"
        and column.evidence == "full"
        and column.printed.regime != "significant_digits"
        and fitted_text is not None
        and own is not None
    ):
        proposed = normalize_decimal(fitted_text, field="the derived sampling rate")
    resolution = _in_unit(Decimal(1).scaleb(column.printed.grid))
    return TimeBaseAssessment(
        outcome=outcome,
        evidence=column.evidence,
        regime=column.printed.regime,
        time_unit=time_unit,
        row_count=column.rows,
        first_line=first_line,
        last_line=column.last_line,
        resolution=resolution,
        effective_resolution=(
            _in_unit(Decimal(column.spacing)) if column.spacing else resolution
        ),
        float32_values=column.spacing is not None,
        duration=_in_unit(Decimal(column.span).scaleb(column.printed.grid)),
        modal_steps=(column.m, column.m + 1),
        reference_interval_ticks=column.b,
        step_range=(int(column.steps.min()), int(column.steps.max())),
        max_residual_ticks=fit.max_residual if fit is not None else None,
        rms_residual_ticks=fit.rms_residual if fit is not None else None,
        fitted_rate_hz=fitted_text,
        rate_uncertainty_hz=uncertainty_text,
        proposed_rate_hz=proposed,
        rate_sources=tuple(
            RateSource(s.source, _display(s.rate), _display(s.low), _display(s.high))
            for s in (own, file_declared, stated)
            if s is not None
        ),
        rate_comparisons=tuple(comparisons),
        findings=tuple(findings),
    )


def assess_time_column(
    column: TimeColumnResult,
    *,
    first_line: int,
    stated_rate_hz: Optional[str] = None,
    declared: Optional[DeclaredTimeBase] = None,
) -> TimeBaseAssessment:
    """:func:`assess_time_base` of one decoded time column.

    Args:
        column: A time column of ``text_decoder.decode_text``.
        first_line: ``DecodeResult.first_line``.
        stated_rate_hz: As for :func:`assess_time_base`.
        declared: As for :func:`assess_time_base`.

    Returns:
        The assessment.
    """
    return assess_time_base(
        column.mantissa,
        column.exponent,
        time_unit=column.time_unit,
        first_line=first_line,
        stated_rate_hz=stated_rate_hz,
        declared=declared,
    )
