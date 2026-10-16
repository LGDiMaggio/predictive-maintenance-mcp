"""Canonical finiteness checks for vibration samples and sampling rates.

The single source of truth for "is this signal fit to analyze": every place
that lets an array or a sampling rate into the system calls these two
validators instead of rolling its own comparison. That covers the signal
store's loads, the asset ledger's re-reads, the raw-binary decoder, the ledger
snapshot, test-signal generation, and the public diagnostic entry points that
accept arrays directly from Python.

A comparison such as ``fs <= 0`` or ``not fs > 0`` is not enough: every
comparison with NaN is False, so ``nan <= 0`` lets NaN through, and
``not inf > 0`` lets infinity through. Non-finite input then turns into
confident verdicts (a NaN RMS falls through the ISO 20816-3 zone ladder to
Zone D; an all-NaN envelope matches no fault frequency and reads as a healthy
bearing).

Both validators raise a plain ``ValueError`` whose message states the problem
and the remedy, from this module (project code), so the MCP refusal boundary
forwards the text verbatim and direct Python callers can catch ``ValueError``.
They never repair input: a missing or corrupt sample is refused, not filled.

Numpy and the standard library only, so every layer can import it without
adding a dependency edge.
"""

import math
import numbers
from typing import Any

import numpy as np

__all__ = [
    "BAD_SAMPLE_REMEDY",
    "validate_finite_samples",
    "validate_positive_finite",
    "validate_sampling_rate",
    "validate_signal_and_fs",
]

#: Remedy for a capture with missing, non-finite or non-numeric samples. It
#: reaches an assistant that acts on tool output, so it says what is NOT a fix.
BAD_SAMPLE_REMEDY = (
    "re-export or re-acquire the capture, or cut it at the source to a "
    "contiguous run of valid samples; filling, interpolating or deleting "
    "individual samples is not a fix, because it invents a measurement or "
    "shifts every later sample in time"
)

_DEFAULT_RATE_REMEDY = "declare the sampling frequency in Hz as a positive number"

#: Remedy the direct-call diagnostic boundaries give a Python caller for fs.
_DIRECT_CALL_FS_REMEDY = "pass the sampling frequency the signal was acquired at, in Hz"


def validate_positive_finite(
    value: Any,
    *,
    name: str,
    source: str = "",
    remedy: str,
) -> float:
    """Return *value* as a float when it is a finite real number above zero.

    Args:
        value: The declared quantity. Booleans are refused even though Python
            treats them as integers.
        name: What the quantity is called in the caller's vocabulary; it
            appears as ``"<name> must be positive"`` and ``"<name> > 0"``.
        source: Where the value came from (a parameter or a companion file),
            appended to the message when given.
        remedy: What the caller should do instead.

    Returns:
        The value as a Python float.

    Raises:
        ValueError: If the value is not a real number, is NaN or infinite
            (an integer too large for a float counts as infinite), or is zero
            or negative.
    """
    where = f" (from {source})" if source else ""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be positive and finite ({name} > 0), got {value!r} "
            f"of type {type(value).__name__}, which is not a number{where} — "
            f"{remedy}."
        )
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{name} must be positive and finite ({name} > 0), got "
            f"{value!r}{where} — {remedy}."
        )
    return number


def validate_sampling_rate(
    value: Any,
    *,
    name: str = "sampling rate",
    source: str = "",
    remedy: str = _DEFAULT_RATE_REMEDY,
) -> float:
    """Return *value* as a float when it is a valid sampling rate.

    The rate-specific defaults of :func:`validate_positive_finite`.

    Args:
        value: The declared rate.
        name: What the rate is called in the caller's vocabulary (for
            example ``"fs"``).
        source: Where the value came from, appended to the message when
            given.
        remedy: What the caller should do instead.

    Returns:
        The rate as a Python float.

    Raises:
        ValueError: If the rate is not a real number, is NaN or infinite, or
            is zero or negative.
    """
    return validate_positive_finite(value, name=name, source=source, remedy=remedy)


def validate_signal_and_fs(signal: Any, fs: Any) -> np.ndarray:
    """The check of a diagnostic function that takes an array from Python.

    Used by the three direct-call boundaries (ISO severity, the bearing
    fault peak check and the diagnosis pipeline), which refuse before
    computing anything: the sampling rate first, then the samples.

    Args:
        signal: The samples the caller passed.
        fs: The sampling frequency the caller passed, in Hz.

    Returns:
        The validated samples (see :func:`validate_finite_samples`).

    Raises:
        ValueError: If *fs* is not a finite number above zero, or a sample
            is NaN, infinite or not a number.
    """
    validate_sampling_rate(fs, name="fs", remedy=_DIRECT_CALL_FS_REMEDY)
    return validate_finite_samples(signal, source="signal")


def validate_finite_samples(
    data: Any,
    *,
    source: str = "Signal",
    remedy: str = BAD_SAMPLE_REMEDY,
) -> np.ndarray:
    """Return *data* as an array when every sample is a finite number.

    Integer and boolean arrays are always finite and pass untouched; float
    and complex arrays pass untouched when finite. A text column (what the
    CSV decoder returns once any token fails to parse) is parsed element by
    element: only the elements that are not numbers are reported, and a
    column whose every element parses is returned as float64 and then checked
    for finiteness.

    Args:
        data: Samples as an array, a masked array (judged by its data, not
            its mask) or a sequence.
        source: Subject of the message, for example ``"File 'pump.csv'"``.
        remedy: What the caller should do instead; defaults to
            :data:`BAD_SAMPLE_REMEDY`.

    Returns:
        The validated array: the input itself when it was already a finite
        numeric array, otherwise the float64 array parsed from text.

    Raises:
        ValueError: If any sample is NaN, infinite or not a number; the
            message gives the count, the total and the first offending
            0-based index (flat index plus shape for multi-dimensional data).
    """
    arr = np.asarray(data)
    kind = arr.dtype.kind
    if kind in "biu":
        return arr
    if kind in "OSU":
        arr = _parse_text_samples(arr, source=source, remedy=remedy)
    elif kind not in "fc":
        raise ValueError(
            f"{source} holds {arr.dtype} values, which are not samples — {remedy}."
        )

    finite = np.isfinite(arr)
    if finite.all():
        return arr
    bad = np.flatnonzero(~finite)
    raise ValueError(
        f"{source} contains {bad.size} non-finite sample(s) (NaN or infinity) "
        f"out of {arr.size}; first at {_position(int(bad[0]), arr.shape)} — "
        f"{remedy}."
    )


def _parse_text_samples(arr: np.ndarray, *, source: str, remedy: str) -> np.ndarray:
    """Parse a text or object array into float64, refusing what does not parse."""
    flat = arr.ravel()
    parsed = np.empty(flat.size, dtype=np.float64)
    bad_count = 0
    first_bad = -1
    for i, item in enumerate(flat):
        try:
            parsed[i] = float(item)
        except OverflowError:
            # A number too large for a float: refused below as non-finite.
            parsed[i] = math.inf
        except (TypeError, ValueError):
            if bad_count == 0:
                first_bad = i
            bad_count += 1
    if bad_count:
        raise ValueError(
            f"{source} contains {bad_count} non-numeric sample(s) out of "
            f"{flat.size}; first at {_position(first_bad, arr.shape)}, value "
            f"{flat[first_bad]!r} (a header row or a marker token) — {remedy}."
        )
    return parsed.reshape(arr.shape)


def _position(flat_index: int, shape: tuple[int, ...]) -> str:
    """Where a sample sits, phrased for the message."""
    if len(shape) <= 1:
        return f"sample index {flat_index}"
    return f"sample index {flat_index} (flat, array shape {shape})"
