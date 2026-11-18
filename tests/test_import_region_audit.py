"""The row classifier and the census of the lines a data range leaves out.

The classifier must agree with the strict decoder on every token class it
reads, keep labels, units, key-value preambles and tag lines out of the
data, apply its position rule on both sides of the data region, and the
census must number lines exactly as the decoder does and examine every line
it reports on, past the scanned head included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import import_corpus as corpus
import pytest

from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.path_safety import open_import_file
from predictive_maintenance_mcp.signal_acquisition.text_decoder import (
    TextDecodeError,
    decode_text,
)
from predictive_maintenance_mcp.signal_import import region_audit
from predictive_maintenance_mcp.signal_import.region_audit import (
    RowShape,
    census,
    classify,
    shape_of,
)

TWO = RowShape(",", ".", 2, frozenset({0, 1}), {})
ONE = RowShape(",", ".", 1, frozenset({0}), {})


def kind(text: str, shape: RowShape = TWO, *, adjacent: bool = True) -> str:
    return classify(text, shape, adjacent=adjacent).kind


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.write_bytes(data)
        return path

    return _write


def opened(path: Path):
    return open_import_file(
        str(path),
        config.get_import_roots(),
        base=config.DATA_DIR,
        max_size=config.get_max_signal_size(),
    )


def interp(
    first: int,
    last: Optional[int] = None,
    delimiter: str = ",",
    decimal: str = ".",
    *,
    header: Optional[int] = None,
    channels: tuple[int, ...] = (1,),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": encoding,
            "delimiter": delimiter,
            "decimal": decimal,
            "header_line": header,
            "units_line": None,
        },
        "data_lines": {"first": first, "last": last},
        "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
        "time_columns": [],
        "defaults": {},
        "channels": [
            {"column": c, "role": "vibration", "quantity": "acceleration", "unit": "g"}
            for c in channels
        ],
    }


def audit(path: Path, conf: dict[str, Any], **kwargs: Any):
    with opened(path) as file:
        return census(file, conf, **kwargs)


def spans(found) -> list[tuple[str, int, int, int, str]]:
    return [
        (b.side, b.first_line, b.last_line, b.rows, b.separator) for b in found.blocks
    ]


def rows(count: int, header: str = "time,acc") -> list[str]:
    return [header] + [corpus._plain_row(row) for row in range(count)]


# ---------------------------------------------------------------------------
# Tokens: the decoder's own taxonomy, under the reading's decimal mark
# ---------------------------------------------------------------------------

DECODER_TOKENS = [
    (",", ".", "inf"),
    (",", ".", "NaN"),
    (",", ".", "N/A"),
    (",", ".", "1e400"),
    (",", ".", "1d3"),
    (",", ".", "0x10"),
    (",", ".", "1_000"),
    (",", ".", "−0.5"),
    (",", ".", "1 000"),
    (",", ".", "١٢"),
    (",", ".", '"1.5"'),
    (",", ".", ""),
    (";", ".", "1,234.5"),
    (";", ",", "1.234,5"),
    (";", ",", "0.5"),
    (",", ".", "1.#QNAN"),
    (",", ".", "#VALUE!"),
]


@pytest.mark.parametrize("delimiter, decimal, token", DECODER_TOKENS)
def test_each_damaged_token_reports_the_decoders_own_code(
    write, delimiter, decimal, token
):
    def number(value: str) -> str:
        return value.replace(".", decimal) if decimal == "," else value

    good = [f"{number(corpus._time(r))}{delimiter}{number('0.25')}" for r in range(3)]
    bad = f"{number(corpus._time(3))}{delimiter}{token}"
    path = write(
        "token.csv", corpus.lines_to_bytes(["time" + delimiter + "acc", *good, bad])
    )
    with opened(path) as file, pytest.raises(TextDecodeError) as refused:
        decode_text(file, interp(2, None, delimiter, decimal))
    shape = RowShape(delimiter, decimal, 2, frozenset({0, 1}), {})
    found = classify(bad, shape, adjacent=False)
    assert found.kind == "damaged"
    assert (found.field, found.code) == (1, refused.value.code)
    assert f"the value {found.problem}" in str(refused.value)


def test_arabic_indic_digits_are_number_like_and_overflow_is_non_finite():
    assert kind("0.1,١٢") == "damaged"
    assert classify("0.1,1e400", TWO, adjacent=False).code == "non_finite_token"
    assert kind("0.1,0.2") == "data"


@pytest.mark.parametrize(
    "token",
    [
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
    ],
)
def test_listed_non_finite_spellings_beside_a_number_are_damaged(token):
    assert classify(f"0.5,{token}", TWO, adjacent=False).kind == "damaged"


@pytest.mark.parametrize("token", ["--", "OVL"])
def test_placeholders_outside_the_lists_are_text(token):
    assert kind(f"0.5,{token}") == "text"


# ---------------------------------------------------------------------------
# What is never data
# ---------------------------------------------------------------------------


def test_labels_units_key_values_tags_and_dates_are_text():
    assert kind("time,acc") == "text"
    assert kind("s,g") == "text"
    assert kind("Sampling rate,25600") == "text"
    assert kind("Sample rate,25600") == "text"
    tab = RowShape("tab", ".", 2, frozenset({0, 1}), {})
    assert kind("Delta_X\t0.000039", tab) == "text"
    assert kind("2026-08-20 13:42:00", ONE) == "text"
    dated = shape_of(["2026-08-20T13:42:00,0.25", "2026-08-20T13:42:01,0.26"], ",", ".")
    assert dated is not None and dated.numeric == frozenset({1})
    assert kind("Sampling rate,25600", dated) == "text"
    assert kind("2026-08-20T13:42:02,0.27", dated) == "data"


def test_position_rule_for_hex_labels_and_all_nan_rows():
    for row in ("0x01,0x02", "NaN,NaN"):
        assert kind(row, adjacent=False) == "text"
        assert kind(row, adjacent=True) == "damaged"
    assert kind("NaN", ONE, adjacent=True) == "damaged"
    assert kind("NaN", ONE, adjacent=False) == "text"


def test_ragged_rows_and_the_absorbed_numeric_line():
    assert kind("0.5", adjacent=False) == "ragged"
    assert classify("0.5", TWO, adjacent=False).count == 1
    assert kind("25600", ONE, adjacent=False) == "data"
    assert kind("25600,1", ONE, adjacent=False) == "ragged"


def test_trailing_delimiters_and_leading_blanks_keep_one_shape():
    trailing = shape_of(["0.1,0.2,", "0.3,0.4,"], ",", ".")
    assert trailing is not None and trailing.fields == 2
    assert kind("0.5,0.6,", trailing) == "data"
    assert kind("0.5,0.6", trailing) == "data"
    assert kind("0.5,", trailing) == "damaged"
    spaced = shape_of(["  0.1   0.2", "0.3 0.4"], "whitespace", ".")
    assert spaced is not None and spaced.fields == 2
    assert kind("   0.5\t0.6  ", spaced) == "data"


def test_an_annotation_in_the_empty_last_column_is_a_data_row():
    trailing = shape_of(["0.1,0.2,", "0.3,0.4,"], ",", ".")
    assert trailing is not None and trailing.trailing
    assert kind("0.5,0.6,first sample", trailing) == "data"
    assert kind("0.5,0.6,note,", trailing) == "data"
    assert kind("0.5,NaN,note", trailing) == "damaged"
    assert kind("time,acc,comment", trailing) == "text"
    assert kind("Comment,,note", trailing) == "text"
    assert kind("", trailing) == "blank"
    # Nothing but a value in that column: no data row, so not blank either.
    assert kind(",,note", trailing) == "text"
    marker = classify(",,5", trailing, adjacent=False)
    assert (marker.kind, marker.count) == ("ragged", 3)
    plain = shape_of(["0.1,0.2", "0.3,0.4"], ",", ".")
    assert plain is not None and not plain.trailing
    assert kind("0.5,0.6,note", plain) == "text"


def test_an_annotated_first_row_takes_the_shape_of_the_rows_after_it():
    for note in ("first sample", "7"):
        annotated = shape_of([f"0.1,0.2,{note}", "0.3,0.4,", "0.5,0.6,"], ",", ".")
        assert annotated is not None
        assert (annotated.fields, annotated.trailing) == (2, True), note
        assert annotated.numeric == frozenset({0, 1})
        assert kind("0.7,0.8,7", annotated, adjacent=False) == "data"
        assert kind("0.7,0.8,2026-08-20", annotated, adjacent=False) == "data"
    # Not an annotation: the next row has the same fields filled, or other
    # finite columns.
    for after in ("0.3,0.4,note", "0.3,word,"):
        three = shape_of(["0.1,0.2,note", after], ",", ".")
        assert three is not None
        assert (three.fields, three.trailing) == (3, False), after


def test_a_column_empty_in_every_data_row_accepts_an_annotation():
    gap = shape_of(["0.1,,0.2", "0.3,,0.4"], ",", ".")
    assert gap is not None and gap.text_classes == {1: frozenset({"empty"})}
    assert kind("0.5,note,0.6", gap) == "data"
    assert kind("0.5,,0.6", gap) == "data"
    assert kind("time,,acc", gap) == "text"


def test_a_dated_row_with_a_channel_fewer_is_ragged():
    dated = shape_of(
        ["2026-08-20T13:42:00,0.25,0.5", "2026-08-20T13:42:01,0.26,0.5"], ",", "."
    )
    assert dated is not None and dated.numeric == frozenset({1, 2})
    fewer = classify("2026-08-20T13:42:02,0.27", dated, adjacent=False)
    assert (fewer.kind, fewer.count) == ("ragged", 2)
    assert kind("2026-08-20T13:42:02", dated, adjacent=False) == "text"
    assert kind("Start,2026-08-20T13:42:02", dated, adjacent=False) == "text"
    assert kind("Sampling rate,25600", dated, adjacent=False) == "text"
    undated = shape_of(["0.0,0.25,0.5", "0.1,0.26,0.5"], ",", ".")
    assert undated is not None
    assert kind("2026-08-20T13:42:02,0.27", undated, adjacent=False) == "text"


DATE_TOKENS = [
    "2026-08-20T13:42:00",
    "2026-08-20 13:42:00.5",
    "2026-08-20T13:42:00Z",
    "08-20-2026T13:42:00+02:00",
    "20/08/2026",
    "20.08.2026",
    " 2026-08-20 ",
    "13:42",
    "13:42:00,25",
    "2026-08-20T13:42:00." + "1" * 40,
    "٢٠٢٦-٠٨-٢٠",
    "1.2.3",
    "1-2-3",
    "1.2",
    "0.5",
    "-1.5",
    "1e5",
    "1,5",
    "12:3",
    "T13:42",
    "NaN",
    "",
    "word",
]


@pytest.mark.parametrize("decimal", [".", ","])
def test_the_date_check_agrees_with_the_time_grammar(decimal):
    import random

    from predictive_maintenance_mcp.signal_import.timebase import (
        TimeBaseRefused,
        parse_time_tokens,
    )

    def by_grammar(token: str) -> bool:
        try:
            parse_time_tokens([token], decimal_mark=decimal)
        except TimeBaseRefused as refused:
            return refused.datetime_text
        return False

    picker = random.Random(17)
    alphabet = "0123456789.-/:,TZe+ "
    drawn = [
        "".join(picker.choice(alphabet) for _ in range(picker.randint(1, 14)))
        for _ in range(3000)
    ]
    for token in DATE_TOKENS + drawn:
        assert region_audit._is_date_text(token, decimal) is by_grammar(token), token


def test_three_decimal_values_are_damaged_under_the_other_decimal_mark_only():
    dot = RowShape(";", ".", 2, frozenset({0, 1}), {})
    comma = RowShape(";", ",", 2, frozenset({0, 1}), {})
    assert kind("0.123;0.456", dot) == "data"
    assert kind("1;0.456", comma, adjacent=False) == "damaged"
    assert kind("0.123;0.456", comma) == "damaged"


def test_trailer_lines_after_the_data():
    assert kind("End of record", adjacent=True) == "text"
    assert kind("Checksum,A1B2", adjacent=True) == "text"
    assert kind("Mean,abc", adjacent=True) == "text"
    assert kind("", adjacent=True) == "blank"
    assert kind(",", adjacent=True) == "blank"
    assert kind("0.5,0.25", adjacent=False) == "data"


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def test_short_row_trim_is_one_block_directly_above_the_data(write):
    path = write("short.csv", corpus.short_row())
    found = audit(path, interp(103, header=1), header_line=1)
    assert spans(found) == [("above", 2, 102, 101, "nothing")]
    block = found.blocks[0]
    assert (block.data_rows, block.damaged, block.ragged) == (100, 0, 1)
    assert [(r.line, r.kind, r.count) for r in block.listed] == [(102, "ragged", 1)]


def test_calibration_table_is_separated_by_the_header(write):
    calib = [f"{p}.0,{p * 0.0102:.4f}" for p in (0, 10, 20, 50, 100)]
    data = corpus.lines_to_bytes(["Calibration", *calib] + rows(400))
    path = write("calibration.csv", data)
    assert spans(audit(path, interp(8, header=7), header_line=7)) == [
        ("above", 2, 6, 5, "header")
    ]
    found = audit(path, interp(8))
    assert spans(found) == [("above", 2, 6, 5, "text")]
    assert found.blocks[0].separator_text == "time,acc"


def test_lines_are_numbered_as_the_decoder_numbers_them(write):
    lines = rows(600)
    for line in range(30, 41):
        lines[line - 1] = f"{corpus._time(line - 2)},NaN"
    breaks = ("\r\n", "\n", "\r")
    data = "".join(text + breaks[i % 3] for i, text in enumerate(lines)).encode()
    path = write("mixed.csv", data)
    found = audit(path, interp(300, header=1))
    assert spans(found) == [("above", 2, 299, 298, "nothing")]
    assert [r.line for r in found.blocks[0].listed] == list(range(30, 40))
    with opened(path) as file:
        decoded = decode_text(file, interp(300, header=1), mode="digest")
    assert decoded.first_line == 300


def test_every_line_above_a_start_past_the_head_is_examined(write):
    lines = rows(10000)
    lines[7000 - 1] = f"{corpus._time(6998)},NaN"
    lines[7001 - 1] = f"{corpus._time(6999)},N/A"
    path = write("long.csv", corpus.lines_to_bytes(lines))
    found = audit(path, interp(9001, header=1))
    assert spans(found) == [("above", 2, 9000, 8999, "nothing")]
    block = found.blocks[0]
    assert (block.damaged, block.ragged) == (2, 0)
    assert [(r.line, r.kind) for r in block.listed] == [
        (7000, "damaged"),
        (7001, "damaged"),
    ]


MIXED = {
    5: f"{corpus._time(3)},NaN",
    6: corpus._time(4),
    7: "note,abc",
    8: "",
    9: "0x01,0x02",
    10: "NaN,NaN",
    11: f"{corpus._time(9)},1e400",
    12: f"{corpus._time(10)}, 0.5 ",
    13: f"{corpus._time(11)},0.5,",
    14: f"{corpus._time(12)},1e-300",
    # An integer part that overflows: the byte pattern must leave it to the
    # full classifier.
    15: f"{corpus._time(13)},1" + "0" * 309,
}


def test_the_fast_path_and_the_classifier_agree(write, monkeypatch):
    lines = rows(300)
    for line, text in MIXED.items():
        lines[line - 1] = text
    path = write("mixed.csv", corpus.lines_to_bytes(lines))
    shape = shape_of([corpus._plain_row(0), corpus._plain_row(1)], ",", ".")
    assert shape is not None
    fast = region_audit._fast_data(shape)
    assert fast is not None
    for text in lines:
        if fast.fullmatch(text.encode()):
            assert region_audit._raw_class(text, shape)[0].kind == "data", text
    with_fast = audit(path, interp(250, header=1), header_line=1)
    monkeypatch.setattr(region_audit, "_fast_data", lambda shape: None)
    without = audit(path, interp(250, header=1), header_line=1)
    assert with_fast == without
    assert spans(with_fast) == [
        ("above", 2, 6, 5, "text"),
        ("above", 9, 249, 241, "nothing"),
    ]


def test_the_fast_path_keeps_every_separator_text_on_both_sides(write, monkeypatch):
    above = ["note", "0.5,0.25", "NaN,NaN", "0.5,0.25", "comment", "0.5,0.25"]
    above += ["NaN,NaN", ""]
    tail = ["note", "0.5,0.25", "NaN,NaN", "comment", "NaN,NaN", "0.5,0.25"]
    for _ in range(300):
        tail += ["note", "0.5,0.25"]
    path = write("fast_texts.csv", corpus.lines_to_bytes(above + rows(300) + tail))
    conf = interp(10, 309, header=9)

    def census_and_texts() -> tuple[Any, list[dict[int, str]]]:
        zones: list[Any] = []

        class Recording(region_audit._Zone):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                zones.append(self)

        monkeypatch.setattr(region_audit, "_Zone", Recording)
        found = audit(path, conf, header_line=9)
        return found, [dict(zone.texts) for zone in zones]

    shape = shape_of([corpus._plain_row(0), corpus._plain_row(1)], ",", ".")
    assert shape is not None
    fast = region_audit._fast_data(shape)
    assert fast is not None and fast.fullmatch(b"0.5,0.25")
    with_fast = census_and_texts()
    monkeypatch.setattr(region_audit, "_fast_data", lambda shape: None)
    without = census_and_texts()
    assert with_fast == without
    assert with_fast[0].block_count == 2 + 302
    assert [len(texts) for texts in with_fast[1]] == [2, region_audit.KEPT_BLOCKS]


def test_rows_below_a_stated_last_line(write):
    lines = rows(300) + [corpus._plain_row(r) for r in range(300, 400)]
    data = corpus.lines_to_bytes(lines, final=False)
    path = write("tail.csv", data)
    found = audit(path, interp(2, 301, header=1))
    assert spans(found) == [("below", 302, 401, 100, "nothing")]


def test_mirrored_position_rule_below_the_data(write):
    direct = write(
        "direct.csv", corpus.lines_to_bytes(rows(300) + ["NaN,NaN", "NaN,NaN"])
    )
    assert spans(audit(direct, interp(2, 301))) == [("below", 302, 303, 2, "nothing")]
    one = corpus.lines_to_bytes(
        ["acc"] + [f"{corpus.sample(r, 0):.6f}" for r in range(300)] + ["NaN"]
    )
    one_path = write("one.csv", one)
    assert spans(audit(one_path, interp(2, 301, channels=(0,)))) == [
        ("below", 302, 302, 1, "nothing")
    ]
    for separator in ("", "Totals"):
        path = write(
            "sep.csv", corpus.lines_to_bytes(rows(300) + [separator, "NaN,NaN"])
        )
        assert spans(audit(path, interp(2, 301))) == []


def test_trailer_after_a_stated_last_line_is_never_a_block(write):
    path = write(
        "trailer.csv",
        corpus.lines_to_bytes(
            rows(300) + ["", "End of record", "Checksum,A1B2", "Mean,abc"]
        ),
    )
    assert spans(audit(path, interp(2, 301))) == []
    summary = write("summary.csv", corpus.lines_to_bytes(rows(300) + ["", "0.5,0.25"]))
    assert spans(audit(summary, interp(2, 301))) == [("below", 303, 303, 1, "blank")]


def test_blocks_never_span_the_data_region(write):
    path = write("both.csv", corpus.lines_to_bytes(rows(400)))
    assert spans(audit(path, interp(103, 300, header=1), header_line=1)) == [
        ("above", 2, 102, 101, "nothing"),
        ("below", 301, 401, 101, "nothing"),
    ]


def test_a_last_line_at_the_end_and_an_unset_last_line_find_nothing(write):
    path = write("clean.csv", corpus.lines_to_bytes(rows(300)))
    assert spans(audit(path, interp(2, 301, header=1), header_line=1)) == []
    assert spans(audit(path, interp(2, header=1), header_line=1)) == []


def test_a_byte_order_mark_is_not_part_of_line_one(write):
    lines = [f"{corpus._time(0)},NaN"] + [corpus._plain_row(r) for r in range(1, 400)]
    path = write("bom.csv", b"\xef\xbb\xbf" + corpus.lines_to_bytes(lines))
    conf = interp(3, encoding="utf-8-sig")
    found = audit(path, conf)
    assert spans(found) == [("above", 1, 2, 2, "nothing")]
    assert [(r.line, r.kind) for r in found.blocks[0].listed] == [(1, "damaged")]
    with opened(path) as file:
        assert decode_text(file, conf, mode="digest").first_line == 3


def test_lines_after_a_last_line_past_the_first_read_block(write):
    lines = rows(5000)
    lines[4500 - 1] = f"{corpus._time(4498)},NaN"
    lines[4800 - 1] = f"{corpus._time(4798)},N/A"
    breaks = ("\r\n", "\n", "\r")
    data = "".join(text + breaks[i % 3] for i, text in enumerate(lines)).encode()
    tail_starts = sum(len(t) + len(breaks[i % 3]) for i, t in enumerate(lines[:4001]))
    assert tail_starts > region_audit.CENSUS_BLOCK_BYTES
    path = write("long_tail.csv", data)
    conf = interp(2, 4001, header=1)
    found = audit(path, conf, header_line=1)
    assert spans(found) == [("below", 4002, 5001, 1000, "nothing")]
    assert [(r.line, r.kind) for r in found.blocks[0].listed] == [
        (4500, "damaged"),
        (4800, "damaged"),
    ]
    with opened(path) as file:
        assert decode_text(file, conf, mode="digest").last_line == 4001


def test_a_block_against_the_units_line_is_separated_by_it(write):
    block = [corpus._plain_row(r) for r in range(5)]
    data = corpus.lines_to_bytes(["time,acc", *block, "s,g"] + rows(400)[1:])
    path = write("units.csv", data)
    found = audit(path, interp(8, header=1), header_line=1, units_line=7)
    assert spans(found) == [("above", 2, 6, 5, "units")]
    assert found.blocks[0].separator_text == "s,g"


def test_an_annotated_first_sample_is_a_block_beside_the_data(write):
    data = [f"{corpus._plain_row(r)}," for r in range(1, 400)]
    lines = ["time,acc,", f"{corpus._plain_row(0)},first sample"] + data
    path = write("annotated.csv", corpus.lines_to_bytes(lines))
    found = audit(path, interp(3, header=1), header_line=1)
    assert spans(found) == [("above", 2, 2, 1, "nothing")]
    assert found.blocks[0].data_rows == 1


def test_a_start_at_an_annotated_row_reads_a_marked_row_above_as_data(write):
    data = [f"{corpus._plain_row(r)}," for r in range(2, 400)]
    lines = [
        "time,acc,",
        f"{corpus._plain_row(0)},7",
        f"{corpus._plain_row(1)},first sample",
    ] + data
    path = write("annotated_start.csv", corpus.lines_to_bytes(lines))
    found = audit(path, interp(3, header=1), header_line=1)
    assert spans(found) == [("above", 2, 2, 1, "nothing")]
    assert found.blocks[0].data_rows == 1


def test_a_dated_row_with_a_channel_fewer_above_the_data_is_a_block(write):
    data = [
        f"2026-08-20T13:42:00,{corpus.sample(r, 0):.6f},{corpus.sample(r, 1):.6f}"
        for r in range(400)
    ]
    lines = ["date,acc_x,acc_y", "2026-08-20T13:41:59,0.125000"] + data
    path = write("dated.csv", corpus.lines_to_bytes(lines))
    found = audit(path, interp(3, header=1, channels=(1, 2)), header_line=1)
    assert spans(found) == [("above", 2, 2, 1, "nothing")]
    assert [(r.line, r.kind, r.count) for r in found.blocks[0].listed] == [
        (2, "ragged", 2)
    ]


def test_blocks_past_the_kept_ones_are_counted_not_dropped(write):
    tail: list[str] = []
    for _ in range(300):
        tail += ["0.5,0.25", "note"]
    path = write("many.csv", corpus.lines_to_bytes(rows(300) + tail))
    found = audit(path, interp(2, 301))
    assert len(found.blocks) == region_audit.KEPT_BLOCKS == 257
    assert found.block_count == 300
    assert found.tallies == (
        region_audit.BlockTally("below", 43, 43, 302 + 2 * 257, 302 + 2 * 299),
    )
    assert [b.first_line for b in found.blocks[:3]] == [302, 304, 306]


def test_a_separator_line_is_kept_only_as_long_as_a_finding_shows(write):
    block = [corpus._plain_row(r) for r in range(5)]
    long = "  " + "x" * 5000
    data = corpus.lines_to_bytes(["time,acc", *block, long] + rows(400)[1:])
    path = write("long_separator.csv", data)
    found = audit(path, interp(8, header=1), header_line=1)
    assert spans(found) == [("above", 2, 6, 5, "text")]
    assert found.blocks[0].separator_text == "x" * 201


def test_separator_texts_are_kept_for_the_kept_blocks_only(write, monkeypatch):
    zones: list[Any] = []

    class Recording(region_audit._Zone):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            zones.append(self)

    monkeypatch.setattr(region_audit, "_Zone", Recording)
    head: list[str] = []
    for _ in range(300):
        head += ["0.5,0.25", "note"]
    tail: list[str] = []
    for _ in range(300):
        tail += ["note", "0.5,0.25"]
    path = write("both_sides.csv", corpus.lines_to_bytes(head + rows(300) + tail))
    found = audit(path, interp(602, 901, header=601), header_line=601)
    assert found.block_count == 600
    assert [b.first_line for b in found.blocks[:2]] == [1, 3]
    assert [b.separator_text for b in found.blocks] == ["note"] * 257
    assert found.tallies == (
        region_audit.BlockTally("above", 43, 43, 515, 599),
        region_audit.BlockTally("below", 300, 300, 903, 1501),
    )
    assert len(zones) == 2
    assert all(len(zone.texts) <= region_audit.KEPT_BLOCKS for zone in zones)


def test_separator_texts_follow_the_position_rule_on_both_sides(write):
    above = [
        "0.5,0.25",  # 1: a block, ended by line 2, which the rule makes text
        "NaN,NaN",  # 2
        "0.5",  # 3: a block of its own (ragged), ended by line 4
        "note",  # 4
        "0.5,0.25",  # 5: a block with line 6 (damaged) and line 7
        "NaN,NaN",  # 6
        "0.5,0.25",  # 7
        "comment",  # 8
    ]
    below = [
        "note",  # 310
        "0.5,0.25",  # 311: a block with line 312 (damaged)
        "NaN,NaN",  # 312
        "comment",  # 313
        "NaN,NaN",  # 314: text, the rule decides, so it separates line 315
        "0.5",  # 315: a block of its own (ragged)
    ]
    path = write("rule.csv", corpus.lines_to_bytes(above + rows(300) + below))
    found = audit(path, interp(10, 309, header=9), header_line=9)
    assert spans(found) == [
        ("above", 1, 1, 1, "text"),
        ("above", 3, 3, 1, "text"),
        ("above", 5, 7, 3, "text"),
        ("below", 311, 312, 2, "text"),
        ("below", 315, 315, 1, "text"),
    ]
    assert [b.separator_text for b in found.blocks] == [
        "NaN,NaN",
        "note",
        "comment",
        "note",
        "NaN,NaN",
    ]


def test_three_significant_digits():
    from decimal import Decimal

    assert region_audit.three_significant(Decimal(100) / Decimal(25600)) == "0.00391"
    assert region_audit.three_significant(Decimal(1) / Decimal(25600)) == "0.0000391"
    assert region_audit.three_significant(Decimal(4999) / Decimal(25600)) == "0.195"
    assert region_audit.three_significant(Decimal(256000) / Decimal(25600)) == "10.0"
