"""Rows left out of the data range never leave an import silently partial.

One damaged row near the start of an export used to move the proposed data
start forward; the parse proof then waived the bases for the dialect and the
data range, so the import reached ``ready`` with the earlier rows dropped and
no warning. These tests pin, through the inspection's draft and through
hand-written configurations, on the Python service and on the MCP route:

- a damaged row inside the data no longer moves the proposed start: the draft
  starts at the true first row and validation refuses the damaged line;
- one damaged row no longer decides the delimiter and decimal mark;
- rows that could be part of the acquisition and lie above the configured
  data start, or below a stated last line, ask before anything is imported,
  name their lines, row count, duration and separator, and are left out only
  with a basis for the data range and an acknowledgment bound to this file,
  this configuration and these exact rows;
- the resolved finding stays as a notice and the import is recorded as a
  partial acquisition;
- files with nothing such outside their data range keep their outcome.

Every fixture is synthetic and built from a closed formula.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import import_corpus as corpus
import pytest

from predictive_maintenance_mcp.mcp_tools import import_tools
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
    get_signal_info,
    list_signals,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    import_id as compute_import_id,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    normalize_interpretation,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_import import (
    ConfigurationError,
    SignalImportService,
    configure_import,
    inspect_signal_file,
    validate_import,
)

RATE = 25600
ROWS = 2048
STATED = {"basis": "user_stated"}
REASON = "rows recorded before the measurement run started"
ROUTES = ("service", "mcp")
CODE = "excluded_data_rows"


# ---------------------------------------------------------------------------
# Closed-formula files
# ---------------------------------------------------------------------------


def t(row: int) -> str:
    return f"{row / RATE:.6f}"


def plain(row: int) -> str:
    return f"{t(row)},{corpus.sample(row, 0):.6f}"


def to_bytes(lines: list[str]) -> bytes:
    return corpus.lines_to_bytes(lines)


def clean_lines(rows: int = ROWS, header: str = "time,acc") -> list[str]:
    return [header] + [plain(row) for row in range(rows)]


def replaced(line: int, text: str, rows: int = ROWS) -> bytes:
    """The clean file with physical *line* (data row ``line - 2``) replaced."""
    lines = clean_lines(rows)
    lines[line - 1] = text
    return to_bytes(lines)


def inserted(line: int, text: str, rows: int = ROWS) -> bytes:
    """The clean file with *text* inserted so that it is physical *line*."""
    lines = clean_lines(rows)
    lines.insert(line - 1, text)
    return to_bytes(lines)


def token_rows(first: int, last: int, token: str, rows: int = ROWS) -> bytes:
    """The acceleration field of lines *first* to *last* replaced by *token*."""
    lines = clean_lines(rows)
    for line in range(first, last + 1):
        lines[line - 1] = f"{t(line - 2)},{token}"
    return to_bytes(lines)


def one_column(lines_by_number: dict[int, str], rows: int = ROWS) -> bytes:
    lines = ["acc"] + [f"{corpus.sample(row, 0):.6f}" for row in range(rows)]
    for number, text in lines_by_number.items():
        lines[number - 1] = text
    return to_bytes(lines)


def european_nan(line: int) -> bytes:
    def dc(text: str) -> str:
        return text.replace(".", ",")

    head = [
        "Device;DAQ-01",
        "Sampling rate;25600 Hz",
        "Operator;A. Tester",
        "Time;AccX;AccY;AccZ",
        "s;g;g;g",
    ]
    data = []
    for row in range(ROWS):
        values = [dc(t(row))] + [dc(f"{corpus.sample(row, c):.6f}") for c in range(3)]
        if row == line - 6:
            values[2] = "NaN"
        data.append(";".join(values))
    return to_bytes(head + data)


def calibration_table() -> bytes:
    calib = [f"{p}.0,{p * 0.0102:.4f}" for p in (0, 10, 20, 50, 100)]
    return to_bytes(["Calibration", *calib, "time,acc"] + clean_lines()[1:])


def numeric_preamble() -> bytes:
    return to_bytes(
        ["Sampling rate,25600", "Samples,2048", "Channel,1"] + clean_lines()
    )


def instant_file(*, repeated_header: bool = False, warm_up: bool = False) -> bytes:
    """A preamble instant on line 1, a header on line 2 (whose label is no
    preamble key), and either a repeated header at line 103 or NaN in the
    acceleration field of lines 3 to 51."""
    lines = ["Start time,2026-08-20T11:42:00+00:00"] + clean_lines(header="elapsed,acc")
    if repeated_header:
        lines.insert(102, "elapsed,acc")
    if warm_up:
        for line in range(3, 52):
            lines[line - 1] = f"{t(line - 3)},NaN"
    return to_bytes(lines)


def three_decimal_file() -> bytes:
    lines = ["a;b"] + [
        f"{corpus.sample(row, 0):.3f};{corpus.sample(row, 1):.3f}"
        for row in range(ROWS)
    ]
    lines[500] = "1;2"
    return to_bytes(lines)


def ragged_block_file() -> bytes:
    lines = ["a,b,c"]
    lines += [
        f"{t(r)},{corpus.sample(r, 0):.6f},{corpus.sample(r, 1):.6f}"
        for r in range(100)
    ]
    lines += ["time,x,y,z"]
    lines += [
        f"{t(r)},{corpus.sample(r, 0):.6f},{corpus.sample(r, 1):.6f},{corpus.sample(r, 2):.6f}"
        for r in range(ROWS)
    ]
    return to_bytes(lines)


def tied_decimal_comma(rows: int = ROWS) -> bytes:
    return to_bytes(["acc"] + [f"{row % 9 + 1},5" for row in range(rows)])


def trailer_file(*tail: str) -> bytes:
    return to_bytes(clean_lines() + list(tail))


# ---------------------------------------------------------------------------
# Configurations and routes
# ---------------------------------------------------------------------------


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.write_bytes(data)
        return path

    return _write


@pytest.fixture(autouse=True)
def empty_store():
    repo = get_repository()
    repo.clear_all()
    yield
    repo.clear_all()


def drafted(path: Path) -> dict[str, Any]:
    inspected = inspect_signal_file(str(path))
    assert inspected.status == "importable", inspected.issues
    return configure_import(inspected).configuration["interpretation"]


def interpretation(
    dialect: dict[str, Any],
    first: int,
    *,
    last: Optional[int] = None,
    channels: tuple[int, ...] = (1,),
    time_column: Optional[int] = 0,
) -> dict[str, Any]:
    return {
        "schema": "pmm-text-import/1",
        "dialect": dict(dialect),
        "data_lines": {"first": first, "last": last},
        "time_base": {"source": "stated", "sampling_rate_hz": str(RATE)},
        "time_columns": (
            [] if time_column is None else [{"column": time_column, "time_unit": "s"}]
        ),
        "defaults": {},
        "channels": [
            {"column": c, "role": "vibration", "quantity": "acceleration", "unit": "g"}
            for c in channels
        ],
    }


def from_draft(path: Path, **overrides: Any) -> dict[str, Any]:
    """The draft's dialect and data start, with a stated rate and channels."""
    draft = drafted(path)
    return interpretation(draft["dialect"], draft["data_lines"]["first"], **overrides)


def dialect(
    delimiter: str = ",",
    decimal: str = ".",
    header: Optional[int] = 1,
    units: Optional[int] = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "decimal": decimal,
        "header_line": header,
        "units_line": units,
    }


def rate_and_units(interp: dict[str, Any]) -> dict[str, Any]:
    bases = {"time_base": STATED}
    for channel in interp["channels"]:
        bases[f"channels[{channel['column']}].unit"] = STATED
    return bases


def every_basis(interp: dict[str, Any]) -> dict[str, Any]:
    return {**rate_and_units(interp), "dialect": STATED, "data_lines": STATED}


def configuration(
    interp: dict[str, Any],
    bases: dict[str, Any],
    *,
    exclusions: Optional[list[dict[str, Any]]] = None,
    overrides: Optional[dict[str, str]] = None,
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    interp = copy.deepcopy(interp)
    if defaults:
        interp["defaults"] = dict(defaults)
    attestation: dict[str, Any] = {"bases": dict(bases), "overrides": overrides or {}}
    if exclusions is not None:
        attestation["exclusions"] = exclusions
    return {
        "schema": "pmm-text-import-config/1",
        "interpretation": interp,
        "attestation": attestation,
    }


def validate(path: Path, conf: dict[str, Any], route: str = "service"):
    if route == "service":
        return validate_import(str(path), conf, channel="python_api")
    sent = import_tools._plain(import_tools.ImportConfiguration.model_validate(conf))
    return validate_import(
        str(path), sent, channel="mcp_tool_argument", transport="stdio"
    )


def findings(result, effect: Optional[str] = None) -> list[Any]:
    return [
        issue
        for issue in result.issues
        if issue.code == CODE and (effect is None or issue.effect == effect)
    ]


def summary(result) -> tuple[str, list[str]]:
    return result.outcome, sorted(
        {f"{i.code}:{i.effect}@{i.line}" for i in result.issues}
    )


_BLOCK = re.compile(
    r"^(?:Lines (\d+) to (\d+)|Line (\d+)) (above the data start|below the data end) "
    r"holds? (\d+) rows? that could be part of the acquisition"
)
_VALUES = re.compile(
    r"source_sha256 ([0-9a-f]{64}), import_id ([0-9a-f]{64}), first_line (\d+), "
    r"last_line (\d+), rows (\d+) and their reason"
)


def block_of(issue) -> tuple[int, int, int, str]:
    match = _BLOCK.match(issue.message)
    assert match is not None, issue.message
    first = int(match.group(1) or match.group(3))
    last = int(match.group(2) or match.group(3))
    return first, last, int(match.group(5)), match.group(4).split()[0]


def blocks(
    result, effect: str = "needs_confirmation"
) -> list[tuple[int, int, int, str]]:
    return [block_of(issue) for issue in findings(result, effect)]


def acknowledgments_from(result, reason: str = REASON) -> list[dict[str, Any]]:
    """What a caller copies from each finding of *result*."""
    out = []
    for issue in findings(result, "needs_confirmation"):
        match = _VALUES.search(issue.message)
        assert match is not None, issue.message
        out.append(
            {
                "source_sha256": match.group(1),
                "import_id": match.group(2),
                "first_line": int(match.group(3)),
                "last_line": int(match.group(4)),
                "rows": int(match.group(5)),
                "reason": reason,
            }
        )
    return out


def acknowledged(conf: dict[str, Any], acks: list[dict[str, Any]]) -> dict[str, Any]:
    conf = copy.deepcopy(conf)
    conf["attestation"]["bases"]["data_lines"] = STATED
    conf["attestation"]["exclusions"] = acks
    return conf


def ask_then_acknowledge(path: Path, conf: dict[str, Any], route: str):
    asked = validate(path, conf, route)
    assert asked.outcome == "needs_confirmation", summary(asked)
    resolved = validate(path, acknowledged(conf, acknowledgments_from(asked)), route)
    return asked, resolved


_NOTICE = re.compile(
    r"^(?:Lines (\d+) to (\d+) \((\d+) rows|Line (\d+) \(1 row) that could be part "
    r"of the acquisition\) (?:are|is) left out as the user acknowledged"
)


def notice_of(issue) -> tuple[int, int, int]:
    match = _NOTICE.match(issue.message)
    assert match is not None, issue.message
    if match.group(4):
        line = int(match.group(4))
        return line, line, 1
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def assert_partial_ready(result, rows: int, expected_blocks) -> None:
    assert result.outcome == "ready", summary(result)
    assert result.import_digest is not None
    assert result.row_count == rows
    notices = findings(result, "notice")
    assert [notice_of(i) for i in notices] == [b[:3] for b in expected_blocks]
    for issue, block in zip(notices, expected_blocks):
        assert issue.message == expected_notice(*block[:3])


def statement(result, field: str = "data_lines") -> str:
    return next(s.text for s in result.statements if s.field == field)


# ---------------------------------------------------------------------------
# Exact wording of the approved findings
# ---------------------------------------------------------------------------


def seconds(rows: int) -> str:
    from decimal import Decimal

    value = Decimal(rows) / Decimal(RATE)
    exponent = value.adjusted() - 2
    return format(value.quantize(Decimal(1).scaleb(exponent)), "f")


def expected_finding(
    first: int,
    last: int,
    rows: int,
    composition: str,
    separator: str,
    sha: str,
    iid: str,
    side: str = "above the data start",
) -> str:
    one = first == last
    head = (
        f"Line {first} {side} holds 1 row that could be part of the acquisition"
        if one
        else f"Lines {first} to {last} {side} hold {rows} rows that could be part "
        f"of the acquisition"
    )
    rows_words = "this row" if one else "these rows"
    return (
        f"{head}: {composition}. At {RATE} Hz they would span {seconds(rows)} s. "
        f"They are separated from the data rows by {separator}. The import would "
        f"leave them out, so it might not be the complete acquisition. Ask the user "
        f"whether {rows_words} should be left out of the import. If they should be "
        f"imported, do not import: re-export the acquisition so that they can be "
        f"read. If they should be left out, the user can confirm the data line range "
        f"with a basis for 'data_lines' and acknowledge the exclusion in "
        f"attestation.exclusions with source_sha256 {sha}, import_id {iid}, "
        f"first_line {first}, last_line {last}, rows {rows} and their reason for "
        f"leaving them out."
    )


def expected_notice(first: int, last: int, rows: int) -> str:
    head = (
        f"Line {first} (1 row that could be part of the acquisition) is left out"
        if first == last
        else f"Lines {first} to {last} ({rows} rows that could be part of the "
        f"acquisition) are left out"
    )
    return (
        f"{head} as the user acknowledged; the acknowledgment and its reason are "
        f"recorded. This import is recorded as a partial acquisition and may not be "
        f"the complete acquisition."
    )


# ---------------------------------------------------------------------------
# A damaged row inside the data: the draft starts at the true first row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestDamagedRowInsideTheData:
    def test_short_row_is_refused_at_its_line(self, write, route):
        path = write("short.csv", corpus.short_row())
        draft = drafted(path)
        assert draft["data_lines"]["first"] == 2
        assert draft["dialect"]["header_line"] == 1
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert result.outcome == "invalid"
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("ragged_row", 102)
        ]

    @pytest.mark.parametrize(
        "token, code",
        [
            ("inf", "non_finite_token"),
            ("NaN", "non_finite_token"),
            ("N/A", "non_finite_token"),
            ("1e400", "non_finite_token"),
            ("1d3", "non_numeric_token"),
            ("0x10", "non_numeric_token"),
            ("1_000", "non_numeric_token"),
            ("−0.5", "non_numeric_token"),
            ("1 000", "non_numeric_token"),
            ("1.#QNAN", "non_numeric_token"),
            ("-1.#IND", "non_numeric_token"),
            ("#VALUE!", "non_numeric_token"),
            ("#DIV/0!", "non_numeric_token"),
        ],
    )
    def test_damaged_token_is_refused_at_its_line(self, write, route, token, code):
        path = write("token.csv", corpus.bad_token(token))
        draft = drafted(path)
        assert draft["data_lines"]["first"] == 2
        assert draft["dialect"]["header_line"] == 1
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert result.outcome == "invalid"
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            (code, 102)
        ]

    def test_empty_field_is_refused_as_an_empty_field(self, write, route):
        path = write("empty.csv", replaced(102, f"{t(100)},"))
        assert drafted(path)["data_lines"]["first"] == 2
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        refused = [i for i in result.issues if i.effect == "invalid"]
        assert [(i.code, i.line) for i in refused] == [("non_numeric_token", 102)]
        assert "is an empty field" in refused[0].message

    def test_damaged_row_never_chooses_the_reading(self, write, route):
        path = write("thousands.csv", corpus.thousands_comma_decimal())
        draft = drafted(path)
        assert (draft["dialect"]["delimiter"], draft["dialect"]["decimal"]) == (
            ";",
            ",",
        )
        assert draft["data_lines"]["first"] == 2
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("thousands_separator", 102)
        ]

    def test_european_export_with_nan_in_a_selected_channel(self, write, route):
        path = write("european.csv", european_nan(106))
        draft = drafted(path)
        assert draft["data_lines"]["first"] == 6
        assert draft["dialect"]["header_line"] == 4
        interp = from_draft(path, channels=(1, 2, 3))
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("non_finite_token", 106)
        ]

    @pytest.mark.parametrize("line", [3, 150, 151])
    def test_short_row_anywhere_in_the_head_is_refused_at_its_line(
        self, write, route, line
    ):
        path = write("short.csv", replaced(line, t(line - 2)))
        assert drafted(path)["data_lines"]["first"] == 2
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("ragged_row", line)
        ]

    def test_one_column_nan_inside_the_data(self, write, route):
        path = write("one.csv", one_column({102: "NaN"}))
        draft = drafted(path)
        assert draft["data_lines"]["first"] == 2
        assert draft["dialect"]["header_line"] == 1
        interp = from_draft(path, channels=(0,), time_column=None)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("non_finite_token", 102)
        ]


# ---------------------------------------------------------------------------
# Rows above the data start ask, with their evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestRowsAboveTheDataStart:
    def test_blank_line_separated_block(self, write, route):
        path = write("blank.csv", inserted(102, ""))
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 103
        conf = configuration(interp, rate_and_units(interp))
        asked, resolved = ask_then_acknowledge(path, conf, route)
        assert blocks(asked) == [(2, 101, 100, "above")]
        issue = findings(asked)[0]
        assert issue.line == 2 and issue.token is None
        assert issue.message == expected_finding(
            2,
            101,
            100,
            "100 with the data's columns",
            "a blank line",
            asked.source_sha256,
            asked.import_id,
        )
        assert asked.import_digest is None
        assert_partial_ready(resolved, 1948, [(2, 101, 100)])
        assert findings(resolved, "notice")[0].message == expected_notice(2, 101, 100)

    def test_repeated_header_separated_block(self, write, route):
        path = write("header.csv", inserted(102, "time,acc"))
        interp = from_draft(path)
        assert interp["dialect"]["header_line"] == 102
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 101, 100, "above")]
        issue = findings(asked)[0]
        assert "separated from the data rows by a header line" in issue.message
        assert (
            issue.token is not None and issue.token.as_dict()["file_text"] == "time,acc"
        )
        assert_partial_ready(resolved, 1948, [(2, 101, 100)])

    def test_explicit_trim_of_a_clean_file(self, write, route):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = interpretation(dialect(), 103)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, every_basis(interp)), route
        )
        assert blocks(asked) == [(2, 102, 101, "above")]
        assert (
            "nothing (they are directly above the data)" in findings(asked)[0].message
        )
        assert_partial_ready(resolved, 1947, [(2, 102, 101)])
        assert "lines 2 to 102 (101 rows)" in statement(resolved)
        assert "the configured data rows parse this way" not in statement(resolved)

    def test_trim_past_a_short_row_with_every_basis(self, write, route):
        path = write("short.csv", corpus.short_row())
        interp = interpretation(dialect(), 103)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, every_basis(interp)), route
        )
        assert blocks(asked) == [(2, 102, 101, "above")]
        message = findings(asked)[0].message
        assert (
            ": 100 with the data's columns; 1 with numbers in another number of "
            "fields (line 102, 1 field)." in message
        )
        assert "nothing (they are directly above the data)" in message
        assert_partial_ready(resolved, 1947, [(2, 102, 101)])

    def test_nan_in_the_first_data_row(self, write, route):
        path = write("first.csv", replaced(2, f"{t(0)},NaN"))
        draft = drafted(path)
        assert draft["dialect"]["header_line"] == 1
        assert draft["data_lines"]["first"] == 3
        interp = from_draft(path)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 2, 1, "above")]
        assert (
            ": 1 damaged (line 2, where field 1 is not a finite number)."
            in findings(asked)[0].message
        )
        assert not [i for i in asked.issues if i.code == "printed_value_unverified"]
        assert_partial_ready(resolved, 2047, [(2, 2, 1)])

    def test_short_first_data_row(self, write, route):
        path = write("first.csv", replaced(2, t(0)))
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 3
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 2, 1, "above")]
        assert (
            ": 1 with numbers in another number of fields (line 2, 1 field)."
            in findings(asked)[0].message
        )
        assert "nothing (it is directly above the data)" in findings(asked)[0].message
        assert_partial_ready(resolved, 2047, [(2, 2, 1)])

    def test_calibration_table_above_a_header(self, write, route):
        path = write("calibration.csv", calibration_table())
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 8
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 6, 5, "above")]
        assert (
            "separated from the data rows by a header line"
            in findings(asked)[0].message
        )
        assert_partial_ready(resolved, 2048, [(2, 6, 5)])

    def test_warm_up_block_of_nan_rows(self, write, route):
        path = write("warmup.csv", token_rows(2, 50, "NaN"))
        draft = drafted(path)
        assert draft["dialect"]["header_line"] == 1
        assert draft["data_lines"]["first"] == 51
        interp = from_draft(path)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 50, 49, "above")]
        message = findings(asked)[0].message
        listed = ", ".join(
            f"line {line}, where field 1 is not a finite number"
            for line in range(2, 12)
        )
        assert f": 49 damaged ({listed}, and 39 more)." in message
        assert_partial_ready(resolved, 1999, [(2, 50, 49)])

    def test_ragged_numeric_block_above_a_new_header(self, write, route):
        path = write("ragged.csv", ragged_block_file())
        interp = from_draft(path, channels=(1, 2, 3))
        assert interp["data_lines"]["first"] == 103
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 101, 100, "above")]
        listed = ", ".join(f"line {line}, 3 fields" for line in range(2, 12))
        assert (
            f": 100 with numbers in another number of fields ({listed}, and 90 more)."
            in findings(asked)[0].message
        )
        assert_partial_ready(resolved, ROWS, [(2, 101, 100)])

    @pytest.mark.parametrize(
        "lines_by_number, block",
        [
            ({2: "NaN"}, (2, 2, 1)),
            ({line: "NaN" for line in range(2, 51)}, (2, 50, 49)),
        ],
        ids=["first-row", "warm-up"],
    )
    def test_one_column_nan_above_the_data(self, write, route, lines_by_number, block):
        path = write("one.csv", one_column(lines_by_number))
        draft = drafted(path)
        assert draft["dialect"]["header_line"] == 1
        interp = from_draft(path, channels=(0,), time_column=None)
        asked = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert blocks(asked) == [(*block, "above")]

    def test_all_nan_warm_up_in_two_columns(self, write, route):
        lines = clean_lines()
        for line in range(2, 51):
            lines[line - 1] = "NaN,NaN"
        path = write("allnan.csv", to_bytes(lines))
        draft = drafted(path)
        assert draft["dialect"]["header_line"] == 1
        assert draft["data_lines"]["first"] == 51
        interp = from_draft(path)
        asked = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert blocks(asked) == [(2, 50, 49, "above")]

    @pytest.mark.parametrize("token", ["1.#QNAN", "-1.#IND", "#VALUE!", "#DIV/0!"])
    def test_spreadsheet_and_runtime_spellings_above_the_data(
        self, write, route, token
    ):
        first_row = write("first.csv", replaced(2, f"{t(0)},{token}"))
        draft = drafted(first_row)
        assert draft["dialect"]["header_line"] == 1
        interp = from_draft(first_row)
        asked = validate(
            first_row, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(2, 2, 1, "above")]
        warm = write("warm.csv", token_rows(2, 50, token))
        interp = from_draft(warm)
        asked = validate(warm, configuration(interp, rate_and_units(interp)), route)
        assert blocks(asked) == [(2, 50, 49, "above")]

    def test_numeric_metadata_line_above_a_blank_line(self, write, route):
        lines = ["25600,1", ""] + [plain(row) for row in range(ROWS)]
        path = write("meta.csv", to_bytes(lines))
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 3
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), route
        )
        assert blocks(asked) == [(1, 1, 1, "above")]
        assert_partial_ready(resolved, ROWS, [(1, 1, 1)])

    def test_trim_past_the_scanned_head_examines_every_line(self, write, route):
        path = write("long.csv", to_bytes(clean_lines(10000)))
        interp = interpretation(dialect(), 5001)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, every_basis(interp)), route
        )
        assert blocks(asked) == [(2, 5000, 4999, "above")]
        assert ": 4999 with the data's columns." in findings(asked)[0].message
        assert not re.search(
            r"not (?:examined|read|scanned)", findings(asked)[0].message
        )
        assert_partial_ready(resolved, 5001, [(2, 5000, 4999)])


# ---------------------------------------------------------------------------
# Only the basis together with a bound acknowledgment resolves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestResolution:
    def test_basis_alone_acknowledgment_alone_and_override_do_not_resolve(
        self, write, route
    ):
        path = write("blank.csv", inserted(102, ""))
        interp = from_draft(path)
        conf = configuration(interp, rate_and_units(interp))
        asked = validate(path, conf, route)
        acks = acknowledgments_from(asked)

        basis_only = copy.deepcopy(conf)
        basis_only["attestation"]["bases"]["data_lines"] = STATED
        result = validate(path, basis_only, route)
        assert result.outcome == "needs_confirmation"
        assert blocks(result) == [(2, 101, 100, "above")]

        ack_only = configuration(interp, rate_and_units(interp), exclusions=acks)
        result = validate(path, ack_only, route)
        assert result.outcome == "needs_confirmation"
        assert blocks(result) == [(2, 101, 100, "above")]

        override = configuration(
            interp, rate_and_units(interp), overrides={"data_lines": REASON}
        )
        with pytest.raises(ConfigurationError) as refused:
            validate(path, override, route)
        assert (
            "attestation.overrides['data_lines'] takes no override: rows the data range "
            "leaves out are acknowledged in attestation.exclusions, bound to this file, "
            "this configuration and the exact rows" in str(refused.value)
        )

    def test_acknowledgment_given_on_the_first_validation(self, write, route):
        data = inserted(102, "")
        path = write("blank.csv", data)
        interp = from_draft(path)
        canonical = normalize_interpretation(interp)
        sha = hashlib.sha256(data).hexdigest()
        ack = {
            "source_sha256": sha,
            "import_id": compute_import_id(sha, len(data), canonical),
            "first_line": 2,
            "last_line": 101,
            "rows": 100,
            "reason": REASON,
        }
        conf = acknowledged(configuration(interp, rate_and_units(interp)), [ack])
        result = validate(path, conf, route)
        assert_partial_ready(result, 1948, [(2, 101, 100)])
        assert findings(result, "notice")[0].message == expected_notice(2, 101, 100)

    @pytest.mark.parametrize(
        "case", ["file", "interpretation", "lines", "rows", "none"]
    )
    def test_a_reused_acknowledgment_is_refused_naming_what_differs(
        self, write, route, case
    ):
        data = inserted(102, "")
        path = write("blank.csv", data)
        interp = from_draft(path)
        asked = validate(path, configuration(interp, rate_and_units(interp)), route)
        ack = acknowledgments_from(asked)[0]
        target = path
        target_interp = interp
        expected: str
        if case == "file":
            other = bytearray(data)
            other[-3] = ord("9") if other[-3] != ord("9") else ord("8")
            target = write("other.csv", bytes(other))
            expected = (
                f"The exclusion acknowledgment for lines 2 to 101 was given for another "
                f"file (source_sha256 {ack['source_sha256']}). Remove it, or validate "
                f"the file it was given for."
            )
        elif case == "interpretation":
            target_interp = copy.deepcopy(interp)
            target_interp["channels"][0]["unit"] = "m/s2"
            expected = (
                f"The exclusion acknowledgment for lines 2 to 101 was given for another "
                f"configuration of this file (import_id {ack['import_id']}). Validate "
                f"this configuration and acknowledge the rows it leaves out."
            )
        elif case == "lines":
            ack = {**ack, "first_line": 3, "rows": 99}
            expected = (
                "The exclusion acknowledgment names lines 3 to 101, which this "
                "configuration does not leave out as one block. Use the lines the "
                "excluded_data_rows finding gives."
            )
        elif case == "rows":
            ack = {**ack, "rows": 99}
            expected = (
                "The exclusion acknowledgment names 99 rows for lines 2 to 101, where "
                "this configuration leaves out 100. Use the count the "
                "excluded_data_rows finding gives."
            )
        else:
            clean = to_bytes(clean_lines())
            target = write("clean.csv", clean)
            target_interp = interpretation(dialect(), 2)
            sha = hashlib.sha256(clean).hexdigest()
            ack = {
                **ack,
                "source_sha256": sha,
                "import_id": compute_import_id(
                    sha, len(clean), normalize_interpretation(target_interp)
                ),
            }
            expected = (
                "The exclusion acknowledgment names lines 2 to 101, but this "
                "configuration leaves out no row that could be part of the data. "
                "Remove it."
            )
        conf = acknowledged(
            configuration(target_interp, rate_and_units(target_interp)), [ack]
        )
        result = validate(target, conf, route)
        assert result.outcome == "invalid"
        assert result.import_digest is None
        mismatches = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [(i.effect, i.field, i.line, i.message) for i in mismatches] == [
            ("invalid", "attestation.exclusions", ack["first_line"], expected)
        ]
        if case != "none":
            assert blocks(result) == [(2, 101, 100, "above")]

    def test_prior_import_basis_does_not_resolve(self, write, route):
        path = write("blank.csv", inserted(102, ""))
        interp = from_draft(path)
        conf = configuration(interp, rate_and_units(interp))
        asked, resolved = ask_then_acknowledge(path, conf, route)
        SignalImportService(get_repository()).import_(
            str(path),
            acknowledged(conf, acknowledgments_from(asked)),
            resolved.import_digest,
            channel="python_api" if route == "service" else "mcp_tool_argument",
            transport=None if route == "service" else "stdio",
        )
        prior = copy.deepcopy(conf)
        prior["attestation"]["bases"]["data_lines"] = {
            "basis": "user_accepted_prior_value",
            "import_id": resolved.import_id,
        }
        result = validate(path, prior, route)
        assert result.outcome == "needs_confirmation"
        assert blocks(result) == [(2, 101, 100, "above")]


# ---------------------------------------------------------------------------
# No acknowledgment while the dialect is unsettled or misread
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestDialectGate:
    def _ack(self, data: bytes, interp: dict[str, Any], first: int, last: int):
        sha = hashlib.sha256(data).hexdigest()
        return {
            "source_sha256": sha,
            "import_id": compute_import_id(
                sha, len(data), normalize_interpretation(interp)
            ),
            "first_line": first,
            "last_line": last,
            "rows": last - first + 1,
            "reason": REASON,
        }

    def test_unconfirmed_encoding(self, write, route):
        lines = clean_lines(header="time,accé")
        lines.insert(101, "")
        data = "\n".join(lines).encode("cp1252") + b"\n"
        path = write("cp1252.csv", data)
        interp = from_draft(path)
        assert interp["dialect"]["encoding"] == "cp1252"
        conf = acknowledged(
            configuration(interp, rate_and_units(interp)),
            [self._ack(data, interp, 2, 101)],
        )
        result = validate(path, conf, route)
        assert result.outcome == "needs_confirmation"
        assert blocks(result) == [(2, 101, 100, "above")]
        assert findings(result)[0].message.endswith(
            "No acknowledgment can resolve this while the text encoding needs "
            "confirmation: settle the dialect first, then validate again."
        )
        assert "encoding_needs_confirmation" in {i.code for i in result.issues}

    def test_ambiguous_dialect(self, write, route):
        lines = [f"{row % 9 + 1},5" for row in range(ROWS)]
        lines.insert(100, "")
        data = to_bytes(lines)
        path = write("tied.csv", data)
        interp = interpretation(
            dialect(";", ",", None), 102, channels=(0,), time_column=None
        )
        conf = acknowledged(
            configuration(interp, rate_and_units(interp)),
            [self._ack(data, interp, 1, 100)],
        )
        result = validate(path, conf, route)
        assert result.outcome == "needs_confirmation"
        assert findings(result)[0].message.endswith(
            "No acknowledgment can resolve this while the delimiter and decimal mark "
            "need confirmation: settle the dialect first, then validate again."
        )
        assert "ambiguous_dialect" in {i.code for i in result.issues}

    def test_configured_reading_other_than_the_best_one(self, write, route):
        data = corpus.thousands_comma_decimal()
        path = write("thousands.csv", data)
        interp = interpretation(
            dialect(",", ".", 1), 103, channels=(2,), time_column=None
        )
        conf = acknowledged(
            configuration(interp, every_basis(interp)),
            [self._ack(data, interp, 2, 102)],
        )
        result = validate(path, conf, route)
        assert result.outcome == "needs_confirmation"
        assert findings(result)[0].message.endswith(
            "No acknowledgment can resolve this while the scanned rows read better "
            "with ';' between fields and ',' as the decimal mark than with the "
            "configured reading: settle the dialect first, then validate again."
        )

    def test_tied_readings_settled_by_a_dialect_basis_pass(self, write, route):
        data = tied_decimal_comma()
        path = write("tied.csv", data)
        interp = interpretation(
            dialect(";", ",", 1), 103, channels=(0,), time_column=None
        )
        conf = acknowledged(
            configuration(interp, every_basis(interp)),
            [self._ack(data, interp, 2, 102)],
        )
        result = validate(path, conf, route)
        assert_partial_ready(result, ROWS - 101, [(2, 102, 101)])


# ---------------------------------------------------------------------------
# The acquisition instant may describe a row left out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestAcquisitionInstant:
    INSTANT = "2026-08-20T11:42:00+00:00"

    def _conf(self, interp: dict[str, Any]) -> dict[str, Any]:
        bases = {
            **every_basis(interp),
            "defaults.acquired_at": {"basis": "user_accepted_file_value", "line": 1},
        }
        return configuration(interp, bases, defaults={"acquired_at": self.INSTANT})

    @pytest.mark.parametrize(
        "data, found, first",
        [
            (instant_file(repeated_header=True), 104, 104),
            (instant_file(warm_up=True), 52, 52),
            (instant_file(repeated_header=True), 104, 107),
            (instant_file(warm_up=True), 52, 55),
        ],
        ids=["header", "warm-up", "header-trimmed", "warm-up-trimmed"],
    )
    def test_preamble_instant_needs_its_own_confirmation(
        self, write, route, data, found, first
    ):
        path = write("instant.csv", data)
        draft = drafted(path)
        assert draft["data_lines"]["first"] == found
        interp = interpretation(draft["dialect"], first)
        conf = self._conf(interp)
        asked = validate(path, conf, route)
        resolved = validate(
            path, acknowledged(conf, acknowledgments_from(asked)), route
        )
        assert resolved.outcome == "needs_confirmation", summary(resolved)
        instant = [
            i
            for i in resolved.issues
            if i.code == "printed_value_unverified"
            and i.field == "defaults.acquired_at"
        ]
        excluded = sum(rows for _, _, rows, _ in blocks(asked))
        assert [(i.effect, i.line, i.message) for i in instant] == [
            (
                "needs_confirmation",
                1,
                f"The data start leaves out {excluded} rows above it that could be part "
                f"of the acquisition, so the acquisition instant printed on line 1 may "
                f"describe one of those rows rather than the first imported sample. Ask "
                f"the user whether it applies to line {first}; to keep it, name an "
                f"override for 'defaults.acquired_at' with a reason.",
            )
        ]
        overridden = copy.deepcopy(acknowledged(conf, acknowledgments_from(asked)))
        overridden["attestation"]["overrides"]["defaults.acquired_at"] = REASON
        assert validate(path, overridden, route).outcome == "ready"


# ---------------------------------------------------------------------------
# A stated last line is checked like a stated first line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestStatedLastLine:
    def test_rows_below_a_stated_last_line(self, write, route):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = interpretation(dialect(), 2, last=1949)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, every_basis(interp)), route
        )
        assert blocks(asked) == [(1950, 2049, 100, "below")]
        assert findings(asked)[0].message == expected_finding(
            1950,
            2049,
            100,
            "100 with the data's columns",
            "nothing (they are directly below the data)",
            asked.source_sha256,
            asked.import_id,
            side="below the data end",
        )
        assert_partial_ready(resolved, 1948, [(1950, 2049, 100)])

    def test_a_text_trailer_is_never_counted(self, write, route):
        path = write("trailer.csv", trailer_file("", "End of record", "Checksum,A1B2"))
        interp = interpretation(dialect(), 2, last=2049)
        result = validate(path, configuration(interp, every_basis(interp)), route)
        assert result.outcome == "ready", summary(result)
        assert result.row_count == ROWS
        assert findings(result) == []

    def test_without_a_last_line_no_trailing_check_runs(self, write, route):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = interpretation(dialect(), 2)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert result.outcome == "ready", summary(result)
        assert findings(result) == []

    def test_a_summary_row_with_the_data_shape_after_a_blank_line(self, write, route):
        path = write("summary.csv", trailer_file("", "0.5,0.25"))
        interp = interpretation(dialect(), 2, last=2049)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, every_basis(interp)), route
        )
        assert blocks(asked) == [(2051, 2051, 1, "below")]
        message = findings(asked)[0].message
        assert message.startswith(
            "Line 2051 below the data end holds 1 row that could be part of the "
            "acquisition: 1 with the data's columns."
        )
        assert "separated from the data rows by a blank line" in message
        assert_partial_ready(resolved, ROWS, [(2051, 2051, 1)])

    def test_all_nan_row_directly_below_and_after_a_text_line(self, write, route):
        direct = write("direct.csv", trailer_file("NaN,NaN"))
        interp = interpretation(dialect(), 2, last=2049)
        asked = validate(direct, configuration(interp, every_basis(interp)), route)
        assert blocks(asked) == [(2050, 2050, 1, "below")]
        assert (
            ": 1 damaged (line 2050, where field 0 is not a finite number)."
            in findings(asked)[0].message
        )
        separated = write("separated.csv", trailer_file("Summary", "NaN,NaN"))
        result = validate(separated, configuration(interp, every_basis(interp)), route)
        assert result.outcome == "ready", summary(result)
        assert findings(result) == []

    def test_blocks_above_and_below_need_one_acknowledgment_each(self, write, route):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = interpretation(dialect(), 103, last=1949)
        conf = configuration(interp, every_basis(interp))
        asked = validate(path, conf, route)
        assert blocks(asked) == [(2, 102, 101, "above"), (1950, 2049, 100, "below")]
        acks = acknowledgments_from(asked)
        one = validate(path, acknowledged(conf, acks[:1]), route)
        assert one.outcome == "needs_confirmation"
        assert blocks(one) == [(1950, 2049, 100, "below")]
        both = validate(path, acknowledged(conf, acks), route)
        assert_partial_ready(both, 1847, [(2, 102, 101), (1950, 2049, 100)])
        text = statement(both)
        assert "lines 2 to 102 (101 rows); lines 1950 to 2049 (100 rows)" in text


# ---------------------------------------------------------------------------
# Files with nothing outside their data range keep their outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
class TestUnchanged:
    def test_numeric_key_value_preamble(self, write, route):
        path = write("preamble.csv", numeric_preamble())
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 5
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert result.outcome == "ready", summary(result)
        assert result.row_count == ROWS
        assert findings(result) == []

    def test_short_row_past_the_first_half_of_the_head(self, write, route):
        path = write("late.csv", replaced(160, t(158)))
        interp = from_draft(path)
        assert interp["data_lines"]["first"] == 2
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert [(i.code, i.line) for i in result.issues if i.effect == "invalid"] == [
            ("ragged_row", 160)
        ]

    def test_three_decimal_values_keep_their_reading(self, write, route):
        path = write("three.csv", three_decimal_file())
        inspected = inspect_signal_file(str(path))
        assert inspected.dialect.status == "proposed"
        assert (
            inspected.dialect.proposal.delimiter,
            inspected.dialect.proposal.decimal,
        ) == (
            ";",
            ".",
        )
        assert "ambiguous_dialect" not in {i.code for i in inspected.issues}

    def test_clean_file_keeps_its_identity(self, write, route):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = from_draft(path)
        result = validate(path, configuration(interp, rate_and_units(interp)), route)
        assert result.outcome == "ready", summary(result)
        assert result.import_id == compute_import_id(
            result.source_sha256, result.size_bytes, normalize_interpretation(interp)
        )
        assert statement(result).endswith(
            "no line above line 2 has the data's columns, is a damaged data row, or is "
            "a row of numbers in another number of fields."
        )


# ---------------------------------------------------------------------------
# One-row wording and the block limit
# ---------------------------------------------------------------------------


def alternating_tail(blocks: int, head: Optional[list[str]] = None) -> bytes:
    """The clean file after the lines *head*, then *blocks* one-row blocks
    after its last data line, each a row with the data's columns followed by
    a text line."""
    tail: list[str] = []
    for _ in range(blocks):
        tail += ["0.5,0.25", "note"]
    return to_bytes((head or []) + clean_lines() + tail)


class TestOneRowWording:
    def test_a_one_row_block_reads_in_the_singular(self, write):
        path = write("first.csv", replaced(2, t(0)))
        interp = interpretation(dialect(), 3)
        asked, resolved = ask_then_acknowledge(
            path, configuration(interp, rate_and_units(interp)), "service"
        )
        message = findings(asked)[0].message
        assert (
            f"At {RATE} Hz it would span {seconds(1)} s. It is separated from the data "
            f"rows by nothing (it is directly above the data). The import would leave "
            f"this row out, so it might not be the complete acquisition. Ask the user "
            f"whether this row should be left out" in message
        )
        ack = acknowledgments_from(asked)[0]
        assert message.endswith(
            f"Ask the user whether this row should be left out of the import. If it "
            f"should be imported, do not import: re-export the acquisition so that it "
            f"can be read. If it should be left out, the user can confirm the data line "
            f"range with a basis for 'data_lines' and acknowledge the exclusion in "
            f"attestation.exclusions with source_sha256 {ack['source_sha256']}, "
            f"import_id {ack['import_id']}, first_line 2, last_line 2, rows 1 and their "
            f"reason for leaving it out."
        )
        assert statement(resolved).endswith(
            "This import is recorded as a partial acquisition: line 2 (1 row) could be "
            "part of the acquisition and is left out as the user acknowledged, and the "
            "acknowledgment with its reason is recorded."
        )

    def test_the_partial_notice_reads_each_block_in_its_number(self):
        from predictive_maintenance_mcp.signal_import.service import _partial_notice

        one = {"first_line": 2, "last_line": 2, "rows": 1}
        more = {"first_line": 5, "last_line": 9, "rows": 5}
        assert _partial_notice([one]) == (
            "This import is recorded as a partial acquisition: line 2 (1 row) could be "
            "part of the acquisition and is left out as the user acknowledged."
        )
        assert _partial_notice([one, more]) == (
            "This import is recorded as a partial acquisition: line 2 (1 row); lines 5 "
            "to 9 (5 rows) could be part of the acquisition and are left out as the "
            "user acknowledged."
        )

    def test_an_acknowledgment_naming_one_row(self, write):
        path = write("clean.csv", to_bytes(clean_lines()))
        interp = interpretation(dialect(), 4)
        conf = configuration(interp, every_basis(interp))
        acks = acknowledgments_from(validate(path, conf))
        assert [(a["first_line"], a["last_line"], a["rows"]) for a in acks] == [
            (2, 3, 2)
        ]
        acks[0]["rows"] = 1
        refused = validate(path, acknowledged(conf, acks))
        assert [
            i.message
            for i in refused.issues
            if i.code == "basis_mismatch" and i.field == "attestation.exclusions"
        ] == [
            "The exclusion acknowledgment names 1 row for lines 2 to 3, where this "
            "configuration leaves out 2. Use the count the excluded_data_rows finding "
            "gives."
        ]

    def test_a_refused_acknowledgment_of_one_line_names_that_line(self, write):
        path = write("first.csv", replaced(2, t(0)))
        interp = interpretation(dialect(), 3)
        conf = configuration(interp, every_basis(interp))
        good = acknowledgments_from(validate(path, conf))[0]
        assert (good["first_line"], good["last_line"], good["rows"]) == (2, 2, 1)

        def refusals(ack: dict[str, Any], on: Path = path, using=conf) -> list[str]:
            result = validate(on, acknowledged(using, [ack]))
            return [
                i.message
                for i in result.issues
                if i.code == "basis_mismatch" and i.field == "attestation.exclusions"
            ]

        other = {**good, "source_sha256": "c" * 64}
        assert refusals(other) == [
            f"The exclusion acknowledgment for line 2 was given for another file "
            f"(source_sha256 {'c' * 64}). Remove it, or validate the file it was "
            f"given for."
        ]
        moved = {**good, "import_id": "d" * 64}
        assert refusals(moved) == [
            f"The exclusion acknowledgment for line 2 was given for another "
            f"configuration of this file (import_id {'d' * 64}). Validate this "
            f"configuration and acknowledge the rows it leaves out."
        ]
        elsewhere = {**good, "first_line": 5, "last_line": 5}
        assert refusals(elsewhere) == [
            "The exclusion acknowledgment names line 5, which this configuration does "
            "not leave out as one block. Use the lines the excluded_data_rows finding "
            "gives."
        ]
        # A one-line acknowledgment naming another row count is refused by the
        # configuration check before validation, so the 4.4 (d) count message
        # cannot be reached for one line.
        clean = write("clean.csv", to_bytes(clean_lines()))
        clean_interp = interpretation(dialect(), 2)
        clean_conf = configuration(clean_interp, every_basis(clean_interp))
        nothing = acknowledged(clean_conf, [])
        source = validate(clean, nothing)
        stale = {
            **good,
            "source_sha256": source.source_sha256,
            "import_id": source.import_id,
        }
        assert refusals(stale, on=clean, using=clean_conf) == [
            "The exclusion acknowledgment names line 2, but this configuration leaves "
            "out no row that could be part of the data. Remove it."
        ]

    def test_the_instant_beside_one_row_left_out(self, write):
        lines = ["Start time,2026-08-20T11:42:00+00:00"] + clean_lines(
            header="elapsed,acc"
        )
        lines[2] = f"{t(0)},NaN"
        path = write("instant.csv", to_bytes(lines))
        interp = interpretation(dialect(header=2), 4)
        bases = {
            **every_basis(interp),
            "defaults.acquired_at": {"basis": "user_accepted_file_value", "line": 1},
        }
        conf = configuration(
            interp, bases, defaults={"acquired_at": TestAcquisitionInstant.INSTANT}
        )
        asked = validate(path, conf)
        assert blocks(asked) == [(3, 3, 1, "above")]
        resolved = validate(path, acknowledged(conf, acknowledgments_from(asked)))
        assert [
            i.message for i in resolved.issues if i.code == "printed_value_unverified"
        ] == [
            "The data start leaves out 1 row above it that could be part of the "
            "acquisition, so the acquisition instant printed on line 1 may describe "
            "that row rather than the first imported sample. Ask the user whether it "
            "applies to line 4; to keep it, name an override for "
            "'defaults.acquired_at' with a reason."
        ]


class TestBlockLimit:
    def test_past_the_limit_ten_blocks_are_named_and_one_summary_counts_the_rest(
        self, write
    ):
        path = write("many.csv", alternating_tail(300))
        interp = interpretation(dialect(), 2, last=ROWS + 1)
        result = validate(path, configuration(interp, every_basis(interp)))
        assert result.outcome == "needs_confirmation"
        found = findings(result)
        assert len(found) == 11
        assert all(i.effect == "needs_confirmation" for i in found)
        assert [i.line for i in found[:10]] == [ROWS + 2 + 2 * k for k in range(10)]
        assert found[0].message == (
            f"Line {ROWS + 2} below the data end holds 1 row that could be part of the "
            f"acquisition: 1 with the data's columns. At {RATE} Hz it would span "
            f"{seconds(1)} s. It is separated from the data rows by nothing (it is "
            f"directly below the data). The import would leave this row out, so it "
            f"might not be the complete acquisition. This configuration leaves out "
            f"more blocks than an acknowledgment can name (256), so no acknowledgment "
            f"can resolve it: re-export the acquisition so that these rows can be "
            f"read, or configure a data line range that leaves fewer blocks out, then "
            f"validate again."
        )
        summary_issue = found[10]
        assert summary_issue.line == ROWS + 2 + 2 * 10
        assert summary_issue.message == (
            f"Beyond the blocks named above, lines {ROWS + 22} to "
            f"{ROWS + 2 + 2 * 299} below the data end hold 290 more blocks of rows "
            f"that could be part of the acquisition (290 rows in all); they are not "
            f"named one by one."
        )
        assert len(result.required_confirmations) < 20

    def test_at_the_limit_every_block_is_named_and_can_be_acknowledged(self, write):
        path = write("limit.csv", alternating_tail(256))
        interp = interpretation(dialect(), 2, last=ROWS + 1)
        conf = configuration(interp, every_basis(interp))
        asked = validate(path, conf)
        assert len(findings(asked)) == 256
        assert not [i for i in findings(asked) if "more blocks" in i.message]
        resolved = validate(path, acknowledged(conf, acknowledgments_from(asked)))
        assert resolved.outcome == "ready", summary(resolved)
        assert len(findings(resolved, "notice")) == 256

    def test_past_the_limit_an_unsettled_dialect_is_the_named_blocks_step(self, write):
        from predictive_maintenance_mcp.signal_import.validation import _checks

        path = write("gated.csv", alternating_tail(300))
        settled_interp = interpretation(dialect(), 2, last=ROWS + 1)
        settled = findings(
            validate(path, configuration(settled_interp, every_basis(settled_interp)))
        )
        # No dialect basis, and a configured header line that the scan does not
        # find: the dialect is unproven, so the gate applies.
        interp = interpretation(dialect(header=None), 2, last=ROWS + 1)
        bases = {k: v for k, v in every_basis(interp).items() if k != "dialect"}
        gated = findings(validate(path, configuration(interp, bases)))
        gate_step = (
            "No acknowledgment can resolve this while the delimiter and decimal mark "
            "need confirmation: settle the dialect first, then validate again."
        )
        assert len(gated) == 11
        assert all(i.message.endswith(gate_step) for i in gated[:10])
        assert [
            i.message.replace(gate_step, _checks._too_many_step()) for i in gated[:10]
        ] == [i.message for i in settled[:10]]
        assert gated[10].message == settled[10].message
        assert gated[10].line == settled[10].line

    def test_past_the_limit_the_summary_joins_both_sides(self, write):
        head: list[str] = []
        for _ in range(20):
            head += ["0.5,0.25", "note"]
        path = write("both.csv", alternating_tail(300, head=head))
        end = 41 + ROWS  # the last data line
        interp = interpretation(dialect(header=41), 42, last=end)
        found = findings(validate(path, configuration(interp, every_basis(interp))))
        assert len(found) == 11
        assert [i.line for i in found[:10]] == [1 + 2 * k for k in range(10)]
        assert found[10].line == 21
        assert found[10].message == (
            f"Beyond the blocks named above, lines 21 to 39 above the data start and "
            f"lines {end + 1} to {end + 599} below the data end hold 310 more blocks of "
            f"rows that could be part of the acquisition (310 rows in all); they are "
            f"not named one by one."
        )

    def test_past_the_limit_acknowledgments_resolve_nothing(self, write):
        path = write("many.csv", alternating_tail(300))
        interp = interpretation(dialect(), 2, last=ROWS + 1)
        conf = configuration(interp, every_basis(interp))
        asked = validate(path, conf)
        acks = [
            {
                "source_sha256": asked.source_sha256,
                "import_id": asked.import_id,
                "first_line": ROWS + 2 + 2 * k,
                "last_line": ROWS + 2 + 2 * k,
                "rows": 1,
                "reason": REASON,
            }
            for k in range(256)
        ]
        result = validate(path, acknowledged(conf, acks))
        assert result.outcome == "needs_confirmation", summary(result)
        assert [(i.effect, i.message) for i in findings(result)] == [
            (i.effect, i.message) for i in findings(asked)
        ]
        assert not findings(result, "notice")


class TestValueWithItsUnit:
    """A number and a word above single-field rows that end with a delimiter
    may be a value with its unit: it never becomes data without a question."""

    @staticmethod
    def lines(top: str = "25600,Hz", end: str = ",") -> list[str]:
        return [top] + [f"{corpus.sample(r, 0):.6f}{end}" for r in range(ROWS)]

    @pytest.mark.parametrize(
        "top, end",
        [
            ("25600,Hz", ","),
            ("25600,,Hz", ",,"),
            ("25600,Hz", ",,"),
            ("25600,,2", ",,"),
        ],
        ids=["unit", "padded", "padded-rows", "two-values"],
    )
    def test_the_draft_starts_after_it_and_validation_asks_about_it(
        self, write, top, end
    ):
        path = write("unit_line.csv", to_bytes(self.lines(top, end)))
        draft = drafted(path)
        assert draft["data_lines"]["first"] == 2
        interp = interpretation(draft["dialect"], 2, channels=(0,), time_column=None)
        result = validate(path, configuration(interp, every_basis(interp)))
        assert result.outcome == "needs_confirmation", summary(result)
        assert blocks(result) == [(1, 1, 1, "above")]

    def test_a_note_on_the_first_single_field_row_keeps_the_header(self, write):
        lines = ["acc,", f"{corpus.sample(0, 0):.6f},first sample"] + [
            f"{corpus.sample(r, 0):.6f}," for r in range(1, ROWS)
        ]
        path = write("noted_one.csv", to_bytes(lines))
        draft = drafted(path)
        assert (draft["dialect"]["header_line"], draft["data_lines"]["first"]) == (1, 3)
        interp = interpretation(draft["dialect"], 3, channels=(0,), time_column=None)
        result = validate(path, configuration(interp, rate_and_units(interp)))
        assert result.outcome == "needs_confirmation", summary(result)
        assert blocks(result) == [(2, 2, 1, "above")]
        missing = {i.field for i in result.issues if i.code == "missing_basis"}
        assert "dialect" not in missing
        assert "data_lines" in missing

    def test_a_range_configured_to_start_on_it_is_not_proven(self, write):
        path = write("unit_line.csv", to_bytes(self.lines()))
        interp = interpretation(
            dialect(header=None), 1, channels=(0,), time_column=None
        )
        result = validate(path, configuration(interp, rate_and_units(interp)))
        assert result.outcome == "needs_confirmation", summary(result)
        assert ("missing_basis", "data_lines") in {
            (i.code, i.field) for i in result.issues
        }


class TestSeparatorEvidence:
    def test_a_long_separator_cut_at_a_space_is_shown_as_cut(self, write):
        long = "a" * 200 + " b"
        path = write("long.csv", inserted(7, long))
        interp = interpretation(dialect(), 8)
        result = validate(path, configuration(interp, every_basis(interp)))
        assert blocks(result) == [(2, 6, 5, "above")]
        token = findings(result)[0].token
        assert token is not None
        assert (token.file_text, token.truncated) == ("a" * 200, True)


# ---------------------------------------------------------------------------
# The partial acquisition reads back as partial
# ---------------------------------------------------------------------------


class TestReadBack:
    @pytest.mark.asyncio
    async def test_imported_partial_acquisition_reads_back_as_partial(self, write):
        path = write("blank.csv", inserted(102, ""))
        interp = from_draft(path)
        conf = configuration(interp, rate_and_units(interp))
        asked, resolved = ask_then_acknowledge(path, conf, "service")
        service = SignalImportService(get_repository())
        imported = service.import_(
            str(path),
            acknowledged(conf, acknowledgments_from(asked)),
            resolved.import_digest,
            channel="python_api",
            transport=None,
        )
        expected = {
            "partial_acquisition": True,
            "excluded_rows": [{"first_line": 2, "last_line": 101, "rows": 100}],
        }
        signal_id = imported.channels[0].signal_id
        info = await get_signal_info(ctx=AsyncMock(), signal_id=signal_id)
        provenance = info.model_dump()["import_provenance"]
        assert {k: provenance[k] for k in expected} == expected
        listed = await list_signals(ctx=AsyncMock(), scope="memory")
        provenance = listed["signals"][0]["import_provenance"]
        assert {k: provenance[k] for k in expected} == expected
        view = import_tools._record_view(service.get_import(imported.import_id))
        assert view.acquisition_completeness.model_dump() == expected
        get_repository().clear_all()
        restored = service.rematerialize(imported.import_id)
        info = await get_signal_info(ctx=AsyncMock(), signal_id=signal_id)
        provenance = info.model_dump()["import_provenance"]
        assert {k: provenance[k] for k in expected} == expected
        assert restored.status == "rematerialized"
