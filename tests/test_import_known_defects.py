"""Known defects the Signal Import Engine makes more visible (plan U1).

Each test states the correct behavior and is a strict ``xfail`` tripwire
built with :func:`known_defect`, so the audit report lists it as a
diagnostic-correctness finding and a fix XPASSes, which fails the strict run
and asks for the marker to be removed (the convention of
``tests/test_diagnostic_invariants.py``):

- ``load-signal-decimal-comma-truncated``: ``load_signal`` reads a
  headerless decimal-comma CSV as two integer columns and keeps the first, so
  ``0,5`` loads as ``0``. The import route reads such files exactly; the fix
  of ``load_signal`` itself is its own change (plan, Deferred to Follow-Up
  Work).
- ``timebase-significant-digit-print-ties``: the time-base assessment's
  print-tie check covers fixed decimals only, so a uniform clock on print
  ties in a significant-digit print (``%.4g`` of mid-interval stamps) is
  still refused.

Retired:

- ``timebase-print-ties-misread`` (PR #108, owner decision 2026-11-04): a
  fixed-decimal column compatible with exact half-to-even rounding of one
  uniform clock with its stamps on print ties is no longer refused; an
  exact check finds that candidate clock and reports the ambiguity as
  ``needs_confirmation``, without selecting or validating a rate. Its
  regression tests live in ``tests/test_timebase.py`` (``TestPrintTies``).

- ``blank-row-scan-quadratic-on-cr-runs`` (plan U3, 2026-11-03): the
  blank-row scan is linear on long runs of ``\\r``. Its regression tests live
  in ``tests/test_signal_loader.py`` (``TestBlankRowScanIsLinear``).
"""

import re
from unittest.mock import AsyncMock

import numpy as np
import pytest

from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from test_diagnostic_invariants import known_defect

#: One second at 10 kHz of decimal-comma samples.
DECIMAL_COMMA_SAMPLES = 10000


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=known_defect(
        "load-signal-decimal-comma-truncated",
        "load_signal reads a headerless decimal-comma CSV as the integer "
        "parts of its samples",
    ),
)
async def test_decimal_comma_csv_is_read_exactly_or_refused(sandbox_data_dir):
    """One second of samples written with a decimal comma (``0,512``,
    ``-0,731``, ...) either loads as those values or is refused for what it
    is; it must never load as their integer parts.

    Only a refusal that names the decimal comma or ``inspect_signal_file``
    counts as the fix: any other refusal fails the test outright (it is not
    the expected ``AssertionError``), so an unrelated new rule cannot XPASS
    this tripwire while the truncation is still there."""
    thousandths = [((n * 7919) % 2001) - 1000 for n in range(DECIMAL_COMMA_SAMPLES)]
    expected = np.array(thousandths, dtype=np.float64) / 1000
    text = "".join(f"{value:.3f}\n".replace(".", ",") for value in expected)
    (sandbox_data_dir / "decimal_comma.csv").write_bytes(text.encode("ascii"))
    repo = get_repository()
    try:
        try:
            await load_signal(
                ctx=AsyncMock(),
                filepath="decimal_comma.csv",
                signal_id="decimal_comma",
                sampling_rate=float(DECIMAL_COMMA_SAMPLES),
            )
        except ValueError as exc:
            if re.search(r"decimal|inspect_signal_file", str(exc), re.IGNORECASE):
                return  # refused for what the file is: a correct outcome
            pytest.fail(f"decimal-comma file refused for another reason: {exc}")
        loaded = repo.get_signal("decimal_comma")
        assert np.array_equal(
            loaded, expected
        ), f"decimal-comma samples loaded as {loaded[:4].tolist()}..."
    finally:
        repo.clear_all()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=known_defect(
        "timebase-significant-digit-print-ties",
        "a uniform clock on print ties is refused in a significant-digit print",
    ),
)
def test_a_uniform_clock_on_significant_digit_ties_is_not_refused():
    """1 kHz stamps at the middle of each interval, (n + 0.5) ms, printed with
    ``%.4g``: past 1 s each stamp keeps three decimals and lies on a print
    tie, as in the fixed-decimal case the check now reads as ambiguous."""
    from predictive_maintenance_mcp.signal_import import timebase

    tokens = ["%.4g" % ((n + 0.5) / 1000) for n in range(1, 3000)]
    mantissa, exponent = timebase.parse_time_tokens(tokens)
    result = timebase.assess_time_base(mantissa, exponent, time_unit="s", first_line=2)
    assert result.outcome != "invalid", result.findings
