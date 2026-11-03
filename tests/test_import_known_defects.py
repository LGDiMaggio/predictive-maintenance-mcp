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
- `timebase-print-ties-misread`: a uniform clock whose every stamp lies
  exactly half a tick off the print grid (mid-interval stamps printed at one
  tick per sample) rounds to steps of 0 and 2, which KTD9's thresholds read
  as duplicates or gaps and refuse; at larger whole-number intervals the same
  ties read as off-pattern (needs confirmation, never a refusal). A
  tie-aware rule changes KTD9's thresholds, so it waits for a plan decision.

Retired:

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
        "timebase-print-ties-misread",
        "a uniform clock whose stamps all lie on print ties is refused at one "
        "or two ticks per sample",
    ),
)
def test_a_uniform_clock_on_print_ties_is_not_refused():
    """1 kHz stamps at the middle of each sample interval, (n + 0.5) ms,
    printed with 3 decimals and rounded half to even: 0.000, 0.002, 0.002,
    0.004, ... The clock is uniform; only the print's ties make its steps
    0 and 2."""
    from decimal import ROUND_HALF_EVEN, Decimal

    from predictive_maintenance_mcp.signal_import import timebase

    tick = Decimal("0.001")
    tokens = [
        str(((n + Decimal("0.5")) * tick).quantize(tick, rounding=ROUND_HALF_EVEN))
        for n in range(2000)
    ]
    mantissa, exponent = timebase.parse_time_tokens(tokens)
    result = timebase.assess_time_base(mantissa, exponent, time_unit="s", first_line=2)
    assert result.outcome != "invalid", result.findings
