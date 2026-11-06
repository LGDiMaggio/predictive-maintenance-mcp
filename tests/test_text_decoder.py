"""Tests for the strict text decoder (Signal Import Engine plan, U5):
``signal_acquisition/text_decoder.py``.

Every scenario of the plan's U5 list, plus the time-stamp parser and the
contract table. Files are written into ``tmp_path`` and opened through the
verified descriptor, as the import will. Exotic characters are built with
``chr`` so the source stays ASCII.
"""

import hashlib
import random
import re
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.file_text import TOKEN_CHARS
from predictive_maintenance_mcp.path_safety import PathRefused, open_import_file
from predictive_maintenance_mcp.signal_acquisition import import_config as ic
from predictive_maintenance_mcp.signal_acquisition import text_decoder as td


def interpretation(
    *,
    delimiter=";",
    decimal=",",
    first=2,
    last=None,
    header_line=1,
    channels=((1, None),),
    time_columns=(),
    encoding="utf-8",
) -> dict:
    """An interpretation of acceleration channels; *channels* is a sequence
    of ``(column, scaling)``."""
    return {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": encoding,
            "delimiter": delimiter,
            "decimal": decimal,
            "header_line": header_line,
        },
        "data_lines": {"first": first, "last": last},
        "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
        "time_columns": [{"column": c, "time_unit": u} for c, u in time_columns],
        "channels": [
            {
                "column": column,
                "role": "vibration",
                "quantity": "acceleration",
                "unit": "g",
                "scaling": scaling,
            }
            for column, scaling in channels
        ],
    }


def decode(tmp_path: Path, data: bytes, interp: dict, *, mode="full", name="x.csv"):
    (tmp_path / name).write_bytes(data)
    with open_import_file(name, [tmp_path], base=tmp_path, max_size=1 << 40) as file:
        return td.decode_text(file, interp, mode=mode)


def refusal(tmp_path: Path, data: bytes, interp: dict, **kwargs) -> td.TextDecodeError:
    with pytest.raises(td.TextDecodeError) as caught:
        decode(tmp_path, data, interp, **kwargs)
    assert caught.value.code in ic.ISSUE_CODES
    return caught.value


def channel(result: td.DecodeResult, column: int) -> np.ndarray:
    [found] = [c for c in result.channels if c.column == column]
    return found.samples


SCALING_100_MV_PER_G = {
    "source_unit": "mV",
    "sensitivity": {
        "value": "100",
        "unit": "mV/g",
        "source": "calibration_certificate",
    },
    "dc_policy": "keep",
}


# ---------------------------------------------------------------------------
# Exact decoding
# ---------------------------------------------------------------------------


class TestExactValues:
    def test_decimal_comma_semicolon_decodes_to_float_of_each_token(self, tmp_path):
        rng = random.Random(1)
        tokens = [f"{rng.uniform(-50, 50):.{rng.randint(0, 9)}f}" for _ in range(2000)]
        body = "".join(f"{i};{t.replace('.', ',')}\n" for i, t in enumerate(tokens))
        result = decode(tmp_path, ("t;acc\n" + body).encode(), interpretation())
        assert np.array_equal(channel(result, 1), [float(t) for t in tokens])
        assert result.first_line == 2 and result.last_line == 2001
        assert result.row_count == 2000 and result.field_count == 2

    def test_17_significant_digit_tokens_are_bit_for_bit_float(self, tmp_path):
        rng = random.Random(17)
        values = [rng.uniform(-1, 1) * 10 ** rng.randint(-20, 20) for _ in range(5000)]
        tokens = [f"{v:.16e}" if i % 2 else repr(v) for i, v in enumerate(values)]
        body = "".join(f"{t}\n" for t in tokens).encode()
        interp = interpretation(
            delimiter=",", decimal=".", first=1, header_line=None, channels=((0, None),)
        )
        decoded = channel(decode(tmp_path, body, interp), 0)
        expected = np.array([float(t) for t in tokens])
        assert decoded.tobytes() == expected.tobytes()

    def test_whitespace_delimited_columns_with_padding(self, tmp_path):
        data = b"  1.5   -2.25\n 3   4e-3\n\t5\t6\n"
        interp = interpretation(
            delimiter="whitespace",
            decimal=".",
            first=1,
            header_line=None,
            channels=((0, None), (1, None)),
        )
        result = decode(tmp_path, data, interp)
        assert channel(result, 0).tolist() == [1.5, 3.0, 5.0]
        assert channel(result, 1).tolist() == [-2.25, 0.004, 6.0]

    def test_spaces_around_a_fixed_delimited_value_are_read(self, tmp_path):
        data = b"a; b\n1 ; 2,5 \n"
        result = decode(tmp_path, data, interpretation())
        assert channel(result, 1).tolist() == [2.5]

    def test_utf8_bom_is_stripped_before_line_one(self, tmp_path):
        data = b"\xef\xbb\xbf1,5\n2,5\n"
        interp = interpretation(
            first=1, header_line=None, channels=((0, None),), encoding="utf-8-sig"
        )
        assert channel(decode(tmp_path, data, interp), 0).tolist() == [1.5, 2.5]

    def test_the_file_digest_is_every_byte(self, tmp_path):
        data = b"head\r\n1;2,5\r\n3;4,5\r\ntrailer text\r\n"
        result = decode(tmp_path, data, interpretation(last=3))
        assert result.source_sha256 == hashlib.sha256(data).hexdigest()
        assert result.size_bytes == len(data)
        assert channel(result, 1).tolist() == [2.5, 4.5]


# ---------------------------------------------------------------------------
# Token refusals
# ---------------------------------------------------------------------------


class TestTokenRefusals:
    @pytest.mark.parametrize(
        ("token", "code"),
        [
            (b"NaN", "non_finite_token"),
            (b"NA", "non_finite_token"),
            (b"inf", "non_finite_token"),
            (b"-Infinity", "non_finite_token"),
            (b"", "non_numeric_token"),
            (b'"2,5"', "quoted_data"),
            (b"1_000", "non_numeric_token"),
            (chr(0x2212).encode() + b"1", "non_numeric_token"),
            (b"1" + chr(0xA0).encode(), "non_numeric_token"),
            (chr(0x661).encode(), "non_numeric_token"),
            (b"1.234,5", "thousands_separator"),
            (b"0x10", "non_numeric_token"),
            (b"1d3", "non_numeric_token"),
            (b"1e400", "non_finite_token"),
            (b"2.5", "non_numeric_token"),
            (b"1,2,3", "non_numeric_token"),
        ],
    )
    def test_each_bad_token_is_refused_with_line_and_field(self, tmp_path, token, code):
        data = b"t;acc\n0;1,5\n1;" + token + b"\n2;3,5\n"
        caught = refusal(tmp_path, data, interpretation())
        assert caught.code == code
        assert caught.line == 3 and caught.field == 1
        assert caught.position == 2
        assert "line 3, field 1 (the 2nd from the left)" in str(caught)

    def test_dot_decimal_thousands_separator(self, tmp_path):
        data = b"t;acc\n1;1,234.5\n"
        caught = refusal(tmp_path, data, interpretation(decimal="."))
        assert caught.code == "thousands_separator"

    def test_a_dot_in_a_decimal_comma_field_is_refused(self, tmp_path):
        caught = refusal(tmp_path, b"t;acc\n1;2,5\n2;2.5\n", interpretation())
        assert caught.line == 3
        assert "'.' in a decimal-comma field" in str(caught)

    def test_a_300_character_token_is_cited_short_and_escaped(self, tmp_path):
        token = ("x" + chr(0x202E)).encode() * 150
        caught = refusal(tmp_path, b"t;acc\n1;" + token + b"\n", interpretation())
        message = str(caught)
        assert caught.token is not None
        assert len(caught.token.file_text) <= TOKEN_CHARS
        assert chr(0x202E) not in message
        assert "could not convert" not in message and "numpy" not in message.lower()
        assert len(message) < 600

    def test_a_repeated_header_block_is_refused_at_its_line(self, tmp_path):
        data = b"t;acc\n1;2,5\nt;acc\n3;4,5\n"
        caught = refusal(tmp_path, data, interpretation())
        assert caught.code == "repeated_header_block"
        assert caught.line == 3

    def test_text_in_an_unselected_column_is_ignored_but_counted(self, tmp_path):
        data = b"t;acc;note\n1;2,5;ok, fine.\n2;3,5;n/a\n"
        assert channel(decode(tmp_path, data, interpretation()), 1).tolist() == [
            2.5,
            3.5,
        ]
        caught = refusal(tmp_path, b"t;acc;note\n1;2,5;ok\n2;3,5\n", interpretation())
        assert caught.code == "ragged_row" and caught.line == 3

    def test_an_unselected_dotted_date_column_is_ignored(self, tmp_path):
        data = b"date;acc\n17.10.2026;1,5\n17.10.2026;-2,5\n"
        assert channel(decode(tmp_path, data, interpretation()), 1).tolist() == [
            1.5,
            -2.5,
        ]


# ---------------------------------------------------------------------------
# Lines and rows
# ---------------------------------------------------------------------------


class TestLinesAndRows:
    def test_a_blank_line_inside_the_data_is_refused(self, tmp_path):
        caught = refusal(tmp_path, b"t;acc\n1;2,5\n\n3;4,5\n", interpretation())
        assert caught.code == "blank_line_in_data" and caught.line == 3

    def test_a_blank_line_in_the_preamble_is_not_read(self, tmp_path):
        data = b"rate;25600\n\nt;acc\n1;2,5\n"
        result = decode(tmp_path, data, interpretation(first=4, header_line=3))
        assert channel(result, 1).tolist() == [2.5]

    def test_trailing_blank_lines_are_accepted_when_reading_to_the_end(self, tmp_path):
        result = decode(tmp_path, b"t;acc\n1;2,5\n\n  \n\n", interpretation())
        assert result.row_count == 1

    def test_a_declared_last_line_makes_any_blank_a_refusal(self, tmp_path):
        caught = refusal(tmp_path, b"t;acc\n1;2,5\n\n", interpretation(last=3))
        assert caught.code == "blank_line_in_data" and caught.line == 3

    @pytest.mark.parametrize(("row", "line"), [(b"3", 3), (b"3;4,5;6", 3)])
    def test_short_and_long_rows_are_refused_with_their_line(self, tmp_path, row, line):
        caught = refusal(tmp_path, b"t;acc\n1;2,5\n" + row + b"\n", interpretation())
        assert caught.code == "ragged_row" and caught.line == line

    def test_a_truncated_last_line_is_refused(self, tmp_path):
        caught = refusal(tmp_path, b"t;acc\n1;2,5\n3;4,", interpretation())
        assert caught.code == "truncated_last_line" and caught.line == 3

    def test_an_unterminated_trailer_after_the_data_is_fine(self, tmp_path):
        result = decode(tmp_path, b"t;acc\n1;2,5\nEnd of data", interpretation(last=2))
        assert result.row_count == 1

    def test_trailing_delimiter_column_decodes_alone_and_with_others(self, tmp_path):
        data = b"t;a;b;\n1;2,5;3,5;\n2;4,5;5,5;\n"
        both = decode(tmp_path, data, interpretation(channels=((1, None), (2, None))))
        alone = decode(tmp_path, data, interpretation(channels=((2, None),)))
        assert both.field_count == 4
        [b_both] = [c for c in both.channels if c.column == 2]
        [b_alone] = alone.channels
        assert b_both.samples_digest == b_alone.samples_digest
        caught = refusal(tmp_path, data, interpretation(channels=((3, None),)))
        assert caught.code == "trailing_delimiter_column"

    def test_a_column_past_the_row_width_is_refused(self, tmp_path):
        caught = refusal(
            tmp_path, b"t;acc\n1;2,5\n", interpretation(channels=((5, None),))
        )
        assert caught.code == "field_count_mismatch"

    def test_data_lines_past_the_end_are_refused(self, tmp_path):
        assert refusal(tmp_path, b"t;acc\n1;2,5\n", interpretation(first=9)).code == (
            "no_numeric_region"
        )
        assert refusal(tmp_path, b"t;acc\n1;2,5\n", interpretation(last=9)).code == (
            "no_numeric_region"
        )

    def test_a_line_over_the_cap_is_refused(self, tmp_path):
        data = b"t;acc\n1;" + b"2" * (td.LINE_CAP + 10) + b"\n"
        assert refusal(tmp_path, data, interpretation()).code == "line_too_long"

    def test_the_line_cap_matches_the_scan(self, tmp_path):
        exact = b"1;2,5;" + b"x" * (td.LINE_CAP - 6)
        interp = interpretation()
        assert decode(tmp_path, b"t;a;note\n" + exact + b"\n", interp).row_count == 1
        caught = refusal(tmp_path, b"t;a;note\n" + exact + b"x\n", interp)
        assert caught.code == "line_too_long" and caught.line == 2

    def test_mixed_line_breaks_keep_physical_line_numbers(self, tmp_path):
        data = b"t;acc\r\r\n1;2,5\r3;4,5\n5;x\r\n"
        caught = refusal(tmp_path, data, interpretation())
        assert caught.line == 4


class TestStreamingEdges:
    """Behaviors that only show across block boundaries or at the end of a
    file; blocks are made small to reach them with small files."""

    @pytest.fixture
    def small_blocks(self, monkeypatch):
        monkeypatch.setattr(td, "BLOCK_BYTES", 64)

    def test_a_line_longer_than_the_cap_without_a_break_is_refused_while_streaming(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(td, "BLOCK_BYTES", 4096)
        monkeypatch.setattr(td, "LINE_CAP", 10_000)
        data = b"t;acc\n1;" + b"2" * 20_000
        caught = refusal(tmp_path, data, interpretation())
        assert caught.code == "line_too_long" and caught.line == 2

    def test_a_blank_line_ending_one_block_and_data_in_the_next(
        self, tmp_path, small_blocks
    ):
        data = b"t;acc\n" + b"1;2,5\n" * 9 + b"\n" + b"1;2,5\n" * 20
        caught = refusal(tmp_path, data, interpretation())
        assert (caught.code, caught.line) == ("blank_line_in_data", 11)

    def test_every_block_size_gives_the_same_samples(self, tmp_path, monkeypatch):
        data = b"t;a;b\r\r\n" + b"".join(
            f"{k};{k / 7:.5f};-{k % 13},5\r\n".replace(".", ",", 1).encode()
            for k in range(500)
        )
        interp = interpretation(
            channels=((1, None), (2, None)), time_columns=((0, "s"),)
        )
        reference = decode(tmp_path, data, interp)
        for size in (1, 7, 64, 1000):
            monkeypatch.setattr(td, "BLOCK_BYTES", size)
            result = decode(tmp_path, data, interp)
            assert [c.samples_digest for c in result.channels] == [
                c.samples_digest for c in reference.channels
            ]
            assert np.array_equal(
                result.time_columns[0].mantissa, reference.time_columns[0].mantissa
            )

    def test_a_trailing_carriage_return_ends_the_last_line(self, tmp_path):
        result = decode(tmp_path, b"t;acc\r1;2,5\r", interpretation())
        assert result.row_count == 1

    def test_input_ending_inside_a_utf8_character_is_refused(self, tmp_path):
        data = b"t;acc\n1;2,5\nend \xc3"
        caught = refusal(tmp_path, data, interpretation(last=2))
        assert caught.code == "unsupported_encoding"

    def test_more_than_the_field_cap_is_refused(self, tmp_path):
        row = ";".join(["1"] * 1100) + "\n"
        caught = refusal(tmp_path, ("t\n" + row).encode(), interpretation())
        assert caught.code == "too_many_fields"

    def test_time_stamps_padded_with_spaces_are_read(self, tmp_path):
        data = b"t;acc\n  0,5 ;1\n1,5   ;2\n"
        result = decode(tmp_path, data, interpretation(time_columns=((0, "s"),)))
        assert result.time_columns[0].mantissa.tolist() == [5, 15]

    def test_an_eleventh_field_is_named_eleventh(self, tmp_path):
        row = ";".join(["1"] * 10 + ["x"]) + "\n"
        caught = refusal(
            tmp_path, ("t\n" + row).encode(), interpretation(channels=((10, None),))
        )
        assert "field 10 (the 11th from the left)" in str(caught)


class TestFirstOffendingLine:
    """R20: whichever check finds it, the refusal names the first offending
    physical line (then the first field)."""

    TWO = interpretation(channels=((1, None), (2, None)))

    @pytest.mark.parametrize(
        ("rows", "line", "code"),
        [
            # a bad token in column 2 before a bad token in column 1
            (b"0;1;2\n0;1;x\n0;y;2\n", 3, "non_numeric_token"),
            # a bad token before a ragged row
            (b"0;1;2\n0;x;2\n0;1\n", 3, "non_numeric_token"),
            # a ragged row before a bad token
            (b"0;1;2\n0;1\n0;x;2\n", 3, "ragged_row"),
            # a bad token before a blank line followed by data
            (b"0;1;2\n0;x;2\n\n0;1;2\n", 3, "non_numeric_token"),
            # a bad token before a truncated last line
            (b"0;1;2\n0;x;2\n0;1;2", 3, "non_numeric_token"),
            # an over-long token after a bad one in the same column
            (b"0;x;2\n0;" + b"1" * 70 + b";2\n", 2, "non_numeric_token"),
            # an over-long token before a bad one in the same column
            (b"0;" + b"1" * 70 + b";2\n0;x;2\n", 2, "non_numeric_token"),
            # a non-finite value before an unparsable one
            (b"0;1e999;2\n0;1e5e;2\n", 2, "non_finite_token"),
        ],
    )
    def test_the_first_line_wins(self, tmp_path, rows, line, code):
        caught = refusal(tmp_path, b"t;a;b\n" + rows, self.TWO)
        assert (caught.line, caught.code) == (line, code)

    def test_a_token_before_an_over_long_line_wins(self, tmp_path):
        data = b"t;a;b\n0;x;2\n0;1;" + b"2" * (td.LINE_CAP + 5) + b"\n"
        caught = refusal(tmp_path, data, self.TWO)
        assert (caught.line, caught.code) == (2, "non_numeric_token")

    def test_the_first_field_wins_on_one_line(self, tmp_path):
        caught = refusal(tmp_path, b"t;a;b\n0;x;y\n", self.TWO)
        assert (caught.line, caught.field) == (2, 1)


class TestBoundedTokenMemory:
    @staticmethod
    def _peak(tmp_path, data: bytes) -> int:
        interp = interpretation(
            delimiter=",", decimal=".", first=1, header_line=None, channels=((0, None),)
        )
        (tmp_path / "x.csv").write_bytes(data)
        with open_import_file(
            "x.csv", [tmp_path], base=tmp_path, max_size=1 << 30
        ) as file:
            tracemalloc.start()
            result = td.decode_text(file, interp, mode="digest")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        assert result.row_count == data.count(b"\n")
        return peak

    def test_one_long_token_does_not_widen_every_row(self, tmp_path):
        """Two million one-byte rows (a whole 4 MiB block), with and without
        one 64-byte token: the token matrix is built in slices and the row
        arrays in batches, so the long token adds almost nothing and the peak
        stays bounded."""
        rows = 2_000_000
        plain = b"1\n" * rows
        with_long = b"1\n" * (rows // 2) + b"1" * 64 + b"\n" + b"1\n" * (rows // 2)
        baseline = self._peak(tmp_path, plain)
        peak = self._peak(tmp_path, with_long)
        assert peak < baseline * 1.2
        assert peak < 96 * 1024 * 1024

    def test_an_overflowing_token_is_refused_without_a_warning(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            caught = refusal(tmp_path, b"t;acc\n1;1e400\n", interpretation(decimal="."))
        assert caught.code == "non_finite_token"


# ---------------------------------------------------------------------------
# File-level refusals
# ---------------------------------------------------------------------------


class TestFileRefusals:
    def test_a_delimiter_equal_to_the_decimal_mark_is_refused_before_reading(
        self, tmp_path
    ):
        (tmp_path / "x.csv").write_bytes(b"1,5\n")
        with open_import_file("x.csv", [tmp_path], base=tmp_path, max_size=99) as file:
            with pytest.raises(ValueError, match="equals the decimal mark"):
                td.decode_text(file, interpretation(delimiter=","))
            assert file.bytes_read == 0

    def test_a_nul_byte_anywhere_is_refused(self, tmp_path):
        data = b"t;acc\n1;2,5\n" + b"trailer\x00\n"
        assert refusal(tmp_path, data, interpretation(last=2)).code == "binary_content"

    def test_bytes_invalid_in_the_confirmed_encoding_are_refused(self, tmp_path):
        data = b"Beschleunigung m/s\xb2;x\n1;2,5\n"
        assert refusal(tmp_path, data, interpretation()).code == "unsupported_encoding"
        result = decode(tmp_path, data, interpretation(encoding="cp1252"))
        assert channel(result, 1).tolist() == [2.5]

    def test_a_file_that_grows_during_the_read_is_refused(self, tmp_path, monkeypatch):
        path = tmp_path / "x.csv"
        path.write_bytes(b"t;acc\n1;2,5\n")
        real_feed = td._Decoder.feed

        def feed_then_append(self, piece):
            with path.open("ab") as appender:
                appender.write(b"3;4,5\n")
            return real_feed(self, piece)

        monkeypatch.setattr(td._Decoder, "feed", feed_then_append)
        with open_import_file("x.csv", [tmp_path], base=tmp_path, max_size=99) as file:
            with pytest.raises(PathRefused) as caught:
                td.decode_text(file, interpretation())
        assert caught.value.code == "source_changed"

    def test_the_output_budget_is_checked_before_it_is_crossed(
        self, tmp_path, monkeypatch
    ):
        """1500 one-byte rows decode to 12000 bytes, over a budget of twice a
        4000-byte cap; digest mode keeps no sample array and passes."""
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "4000")
        data = b"1\n" * 1500
        interp = interpretation(
            delimiter=",", decimal=".", first=1, header_line=None, channels=((0, None),)
        )
        caught = refusal(tmp_path, data, interp)
        assert caught.code == "file_too_large"
        assert "8000-byte output budget" in str(caught)
        assert decode(tmp_path, data, interp, mode="digest").row_count == 1500

    def test_unknown_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            decode(tmp_path, b"t;acc\n1;2\n", interpretation(), mode="fast")


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScaling:
    def test_a_v_column_with_a_mv_per_g_sensitivity_divides_by_volts_per_g(
        self, tmp_path
    ):
        """100 mV/g on a column in volts: each sample divided by exactly
        0.1 V/g (the corrected U5 wording: not by 1000 times the value)."""
        scaling = {**SCALING_100_MV_PER_G, "source_unit": "V"}
        tokens = ["0,25", "-0,1", "1,3", "0,0001"]
        data = (
            "t;acc\n" + "".join(f"{i};{t}\n" for i, t in enumerate(tokens))
        ).encode()
        samples = channel(
            decode(tmp_path, data, interpretation(channels=((1, scaling),))), 1
        )
        raw = np.array([float(t.replace(",", ".")) for t in tokens])
        assert samples.tobytes() == (raw / 0.1).tobytes()
        assert ic.scaling_divisor(scaling) == ic.scaling_divisor(
            {**scaling, "source_unit": "V"}
        )

    def test_a_mv_column_divides_by_the_stated_value(self, tmp_path):
        data = b"t;acc\n1;250\n2;-50\n"
        samples = channel(
            decode(
                tmp_path, data, interpretation(channels=((1, SCALING_100_MV_PER_G),))
            ),
            1,
        )
        assert samples.tolist() == [2.5, -0.5]

    def test_a_scaled_value_that_overflows_is_refused(self, tmp_path):
        scaling = {
            "source_unit": "V",
            "sensitivity": {
                "value": "1e-30",
                "unit": "mV/g",
                "source": "user_estimate",
            },
            "dc_policy": "keep",
        }
        caught = refusal(
            tmp_path, b"t;acc\n1;1e300\n", interpretation(channels=((1, scaling),))
        )
        assert caught.code == "non_finite_token" and caught.line == 2

    @pytest.mark.parametrize("value", ["0", "-100", "NaN"])
    def test_a_zero_negative_or_non_finite_sensitivity_is_refused_before_decoding(
        self, tmp_path, value
    ):
        scaling = {
            **SCALING_100_MV_PER_G,
            "sensitivity": {**SCALING_100_MV_PER_G["sensitivity"], "value": value},
        }
        (tmp_path / "x.csv").write_bytes(b"t;acc\n1;2\n")
        with open_import_file("x.csv", [tmp_path], base=tmp_path, max_size=99) as file:
            with pytest.raises(ValueError, match="sensitivity.value"):
                td.decode_text(file, interpretation(channels=((1, scaling),)))
            assert file.bytes_read == 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _rows(count: int, columns: int = 3) -> bytes:
    rng = np.random.default_rng(3)
    values = np.round(rng.standard_normal((count, columns)), 6)
    lines = [";".join(f"{v:.6f}".replace(".", ",") for v in row) for row in values]
    return ("a;b;c\n" + "\n".join(lines) + "\n").encode()


class TestModes:
    def test_digest_mode_matches_full_mode_and_keeps_no_samples(self, tmp_path):
        data = _rows(5000)
        interp = interpretation(channels=((0, None), (1, None), (2, None)))
        full = decode(tmp_path, data, interp)
        digest = decode(tmp_path, data, interp, mode="digest")
        assert [c.samples_digest for c in full.channels] == [
            c.samples_digest for c in digest.channels
        ]
        assert all(c.samples is None for c in digest.channels)
        for result in full.channels:
            assert result.samples_digest == ic.samples_digest(result.samples)

    @pytest.mark.parametrize("block_bytes", [256, 4096, 1 << 20])
    def test_both_modes_carry_the_streamed_moments(
        self, tmp_path, monkeypatch, block_bytes
    ):
        """Mean and population standard deviation merged block by block equal
        the full array's, whatever the block size, in both modes."""
        monkeypatch.setattr(td, "BLOCK_BYTES", block_bytes)
        data = _rows(5000)
        interp = interpretation(channels=((0, None), (1, None), (2, None)))
        full = decode(tmp_path, data, interp)
        digest = decode(tmp_path, data, interp, mode="digest")
        for kept, streamed in zip(full.channels, digest.channels):
            assert kept.samples is not None and streamed.samples is None
            assert (streamed.mean, streamed.std) == (kept.mean, kept.std)
            assert streamed.mean == pytest.approx(
                float(np.mean(kept.samples)), abs=1e-12
            )
            assert streamed.std == pytest.approx(float(np.std(kept.samples)), rel=1e-12)

    def test_moments_are_of_the_scaled_samples(self, tmp_path):
        rows = [f"{1000 + (r % 7) * 10}" for r in range(600)]
        data = ("v\n" + "\n".join(rows) + "\n").encode()
        scaling = {
            "source_unit": "mV",
            "sensitivity": {
                "value": "100",
                "unit": "mV/g",
                "source": "datasheet_nominal",
            },
            "dc_policy": "keep",
        }
        interp = interpretation(delimiter=",", decimal=".", channels=((0, scaling),))
        result = decode(tmp_path, data, interp)
        [channel] = result.channels
        assert channel.mean == pytest.approx(float(np.mean(channel.samples)), rel=1e-12)
        assert channel.std == pytest.approx(float(np.std(channel.samples)), rel=1e-12)
        assert 10.0 < channel.mean < 10.7  # millivolts divided by 100 mV/g

    @pytest.mark.parametrize("value", ["1.5", "3.3", "1000000000000.1"])
    def test_a_constant_channel_has_exactly_zero_spread(
        self, tmp_path, monkeypatch, value
    ):
        """Across block boundaries and at a large offset: the shifted
        accumulator gives exactly 0, so no ratio is formed from rounding."""
        monkeypatch.setattr(td, "BLOCK_BYTES", 256)
        data = ("v\n" + f"{value}\n" * 3000).encode()
        interp = interpretation(delimiter=",", decimal=".", channels=((0, None),))
        [channel] = decode(tmp_path, data, interp, mode="digest").channels
        assert (channel.mean, channel.std) == (float(value), 0.0)

    def test_digest_mode_holds_no_sample_arrays_only_the_time_array(self, tmp_path):
        """What a decode leaves allocated: the samples in full mode; in
        digest mode only the time column's mantissa and exponent arrays."""
        rows = 200_000
        data = _rows(rows)
        interp = interpretation(
            channels=((1, None), (2, None)), time_columns=((0, "s"),)
        )
        (tmp_path / "x.csv").write_bytes(data)
        held = {}
        for mode in ("full", "digest"):
            with open_import_file(
                "x.csv", [tmp_path], base=tmp_path, max_size=1 << 30
            ) as file:
                tracemalloc.start()
                result = td.decode_text(file, interp, mode=mode)
                held[mode], _ = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            del result
        samples, time_arrays = rows * 2 * 8, rows * (8 + 2)
        assert held["digest"] < time_arrays + 1024 * 1024
        assert held["full"] >= samples + time_arrays


# ---------------------------------------------------------------------------
# Time columns
# ---------------------------------------------------------------------------


class TestTimeColumns:
    @pytest.mark.parametrize(
        ("tokens", "mantissa", "exponent"),
        [
            (["0,000000", "0,000039", "0,000078"], [0, 39, 78], [-6, -6, -6]),
            (["1,50", "-2", "+0,5"], [150, -2, 5], [-2, 0, -1]),
            (["3,9e-05", "1E+3", ",5", "5,"], [39, 1, 5, 5], [-6, 3, -1, 0]),
            (["1,234560e+00", "1,234599e+00"], [1234560, 1234599], [-6, -6]),
        ],
    )
    def test_stamps_are_read_exactly(self, tmp_path, tokens, mantissa, exponent):
        data = ("t;acc\n" + "".join(f"{t};1\n" for t in tokens)).encode()
        result = decode(tmp_path, data, interpretation(time_columns=((0, "s"),)))
        [time] = result.time_columns
        assert time.mantissa.tolist() == mantissa
        assert time.exponent.tolist() == exponent
        assert time.mantissa.dtype == np.int64 and time.exponent.dtype == np.int16
        assert time.time_unit == "s"

    @pytest.mark.parametrize(
        "token",
        [
            "NaN",
            "",
            "1,2,3",
            "e5",
            ",",
            "1e",
            "1e+1234",
            "1234567890123456789",
            "1 2",
            "2.5",
            "inf",
        ],
    )
    def test_bad_stamps_are_refused_as_time_grammar(self, tmp_path, token):
        data = f"t;acc\n0;1\n{token};2\n".encode()
        caught = refusal(tmp_path, data, interpretation(time_columns=((0, "ms"),)))
        assert caught.code in ("time_grammar", "quoted_data", "thousands_separator")
        assert caught.code == "time_grammar"
        assert caught.line == 3 and caught.field == 0

    def test_time_columns_count_toward_the_budget_in_digest_mode(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "4000")
        data = ("t;acc\n" + "".join(f"{i};1\n" for i in range(1000))).encode()
        caught = refusal(
            tmp_path, data, interpretation(time_columns=((0, "s"),)), mode="digest"
        )
        assert caught.code == "file_too_large"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestContracts:
    def test_every_issued_contract_is_registered(self):
        assert td.DECODER_CONTRACT == "pmm-text-decoder/1"
        assert td.get_decoder(td.DECODER_CONTRACT) is td.decode_text
        assert dict(td.DECODERS) == {td.DECODER_CONTRACT: td.decode_text}

    @pytest.mark.parametrize("contract", ["pmm-text-decoder/0", "", None])
    def test_an_unknown_contract_is_refused(self, contract):
        with pytest.raises(ValueError, match="Unknown text decoder contract"):
            td.get_decoder(contract)

    def test_refusals_never_carry_library_text(self, tmp_path):
        caught = refusal(tmp_path, b"t;acc\n1;1e5e5\n", interpretation())
        assert not re.search(r"could not convert|ValueError|float64", str(caught))


# ---------------------------------------------------------------------------
# Large inputs (slow)
# ---------------------------------------------------------------------------


def _fixed_decimal_tokens(ticks: np.ndarray, decimals: int) -> np.ndarray:
    """``ticks / 10**decimals`` printed with *decimals* places, as a uint8
    matrix of digits (no per-token Python object)."""
    integer = ticks // 10**decimals
    width = int(np.floor(np.log10(max(int(integer.max()), 1)))) + 1
    columns = width + 1 + decimals
    out = np.empty((ticks.size, columns), dtype=np.uint8)
    value = ticks.copy()
    for j in range(columns - 1, width, -1):
        out[:, j] = ord("0") + value % 10
        value //= 10
    out[:, width] = ord(".")
    for j in range(width - 1, -1, -1):
        out[:, j] = ord("0") + value % 10
        value //= 10
    return out


@pytest.mark.slow
def test_twelve_million_time_stamps_decode_to_ticks_without_python_tokens(tmp_path):
    rows = 12_000_000
    ticks = np.arange(rows, dtype=np.int64) * 39
    stamps = _fixed_decimal_tokens(ticks, 6)
    tail = np.frombuffer(b",1\n", dtype=np.uint8)
    matrix = np.concatenate([stamps, np.broadcast_to(tail, (rows, 3))], axis=1)
    (tmp_path / "x.csv").write_bytes(matrix.tobytes())
    interp = interpretation(
        delimiter=",",
        decimal=".",
        first=1,
        header_line=None,
        channels=((1, None),),
        time_columns=((0, "s"),),
    )
    with open_import_file("x.csv", [tmp_path], base=tmp_path, max_size=1 << 31) as file:
        tracemalloc.start()
        result = td.decode_text(file, interp, mode="digest")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    [time] = result.time_columns
    assert np.array_equal(time.mantissa, ticks)
    assert (time.exponent == -6).all()
    # The kept tick arrays are 10 bytes a row; Python tokens would be ~66.
    assert peak < rows * 10 + 400 * 1024 * 1024


@pytest.mark.slow
def test_peak_memory_on_100_mb_stays_below_twice_the_result(tmp_path):
    rows = 4_000_000
    rng = np.random.default_rng(9)
    values = np.round(rng.standard_normal((rows, 3)), 5)
    text = "\n".join(";".join(f"{v:.5f}" for v in row) for row in values[:1000])
    block = (text + "\n").encode()
    data = block * (rows // 1000)
    (tmp_path / "x.csv").write_bytes(data)
    interp = interpretation(
        delimiter=";",
        decimal=".",
        first=1,
        header_line=None,
        channels=((0, None), (1, None), (2, None)),
    )
    with open_import_file("x.csv", [tmp_path], base=tmp_path, max_size=1 << 31) as file:
        tracemalloc.start()
        result = td.decode_text(file, interp)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    arrays = sum(c.samples.nbytes for c in result.channels)
    assert len(data) > 100 * 1000 * 1000
    assert peak < 2 * arrays
