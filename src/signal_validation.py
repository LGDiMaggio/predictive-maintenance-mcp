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
    "validate_sampling_rate",
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


def validate_sampling_rate(
    value: Any,
    *,
    name: str = "sampling rate",
    source: str = "",
    remedy: str = _DEFAULT_RATE_REMEDY,
) -> float:
    """Return *value* as a float when it is a finite real number above zero.

    Args:
        value: The declared rate. Booleans are refused even though Python
            treats them as integers.
        name: What the rate is called in the caller's vocabulary (for
            example ``"fs"``); it appears as ``"<name> must be positive"``
            and ``"<name> > 0"``.
        source: Where the value came from (a parameter or a companion file),
            appended to the message when given.
        remedy: What the caller should do instead.

    Returns:
        The rate as a Python float.

    Raises:
        ValueError: If the rate is not a real number, is NaN or infinite, or
            is zero or negative.
    """
    where = f" (from {source})" if source else ""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be positive and finite ({name} > 0), got {value!r} "
            f"of type {type(value).__name__}, which is not a number{where} — "
            f"{remedy}."
        )
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(
            f"{name} must be positive and finite ({name} > 0), got "
            f"{value!r}{where} — {remedy}."
        )
    return rate


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
            f"{source} holds {arr.dtype} values, which are not samples — " f"{remedy}."
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
    bad: list[int] = []
    for i, item in enumerate(flat):
        try:
            parsed[i] = float(item)
        except (TypeError, ValueError):
            bad.append(i)
    if bad:
        raise ValueError(
            f"{source} contains {len(bad)} non-numeric sample(s) out of "
            f"{flat.size}; first at {_position(bad[0], arr.shape)}, value "
            f"{flat[bad[0]]!r} (a header row or a marker token) — {remedy}."
        )
    return parsed.reshape(arr.shape)


def _position(flat_index: int, shape: tuple[int, ...]) -> str:
    """Where a sample sits, phrased for the message."""
    if len(shape) <= 1:
        return f"sample index {flat_index}"
    return f"sample index {flat_index} (flat, array shape {shape})"
