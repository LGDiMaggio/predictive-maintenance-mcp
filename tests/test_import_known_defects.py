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
- ``blank-row-scan-quadratic-on-cr-runs``: ``check_text_blank_rows`` hands a
  chunk to the backtracking pattern ``\\r*\\n|\\r`` when the chunk holds a
  run of three or more ``\\r`` before a ``\\n``; on a long run of lone
  ``\\r`` that pattern restarts at every position, so the scan's time grows
  with the square of the run (about 0.1 s for 16 KiB and 1.6 s for 64 KiB
  measured on 2026-11-02). Plan U3 makes the scan linear and retires this
  tripwire.
"""

import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.signal_acquisition.loaders import check_text_blank_rows
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from test_diagnostic_invariants import known_defect

#: Length of the lone-``\r`` run of the quadratic-scan tripwire. A linear
#: scan reads it in about a millisecond; the quadratic one takes seconds.
CR_RUN_BYTES = 64 * 1024

#: Time a linear scan of :data:`CR_RUN_BYTES` stays far below on any host.
CR_RUN_BOUND_S = 0.25


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
    """A file whose samples are ``0,5``, ``1,25``, ``-0,75``, ``2,5`` either
    loads as those values or is refused; it must never load as integers."""
    (sandbox_data_dir / "decimal_comma.csv").write_bytes(b"0,5\n1,25\n-0,75\n2,5\n")
    repo = get_repository()
    try:
        try:
            await load_signal(
                ctx=AsyncMock(),
                filepath="decimal_comma.csv",
                signal_id="decimal_comma",
                sampling_rate=1000.0,
            )
        except ValueError:
            return  # a refusal is a correct outcome
        loaded = repo.get_signal("decimal_comma")
        assert np.array_equal(
            loaded, [0.5, 1.25, -0.75, 2.5]
        ), f"decimal-comma samples loaded as {loaded.tolist()}"
    finally:
        repo.clear_all()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=known_defect(
        "blank-row-scan-quadratic-on-cr-runs",
        "check_text_blank_rows takes quadratic time on a long run of lone "
        "carriage returns",
    ),
)
def test_blank_row_scan_is_linear_on_carriage_return_runs(tmp_path):
    """A ``\\r\\r\\r\\n`` break followed by 64 KiB of lone ``\\r`` between two
    data lines: the scan refuses the blank lines, and must do so in time
    linear in the file size."""
    path = tmp_path / "cr_run.csv"
    path.write_bytes(b"1\r\r\r\n" + b"\r" * CR_RUN_BYTES + b"2\n")

    start = time.perf_counter()
    with pytest.raises(ValueError, match="empty line at line 2 between data lines"):
        check_text_blank_rows(path)
    elapsed = time.perf_counter() - start

    assert elapsed < CR_RUN_BOUND_S, (
        f"blank-row scan of a {CR_RUN_BYTES}-byte run of lone CR took "
        f"{elapsed:.2f} s"
    )
