"""Tests for the bounded text scan and the shared line-break rule
(Signal Import Engine plan, U3): ``signal_acquisition/text_scan.py`` and
``signal_acquisition/line_breaks.py``.

Pinned here: one physical-line rule streamed in any chunk size and linear on
long ``\\r`` runs; a scan that reads only its head and tail caps (counted
through the descriptor), refuses binary, compressed, NUL-bearing and
UTF-16/UTF-32 content before returning any text, refuses an over-long or
over-wide line, keeps a multibyte character split at the head cap, decodes a
cp1252 preamble, and writes nothing. Exotic characters are built with
``chr`` so the source stays ASCII.
"""

import os
import random
import time
from pathlib import Path

import pytest

from predictive_maintenance_mcp.path_safety import VerifiedFile, open_import_file
from predictive_maintenance_mcp.signal_acquisition import import_config
from predictive_maintenance_mcp.signal_acquisition import line_breaks as lb
from predictive_maintenance_mcp.signal_acquisition import text_scan as ts


def _open(root: Path, name: str) -> VerifiedFile:
    return open_import_file(name, [root], base=root, max_size=1 << 40)


def _scan(root: Path, name: str, data: bytes) -> ts.TextScan:
    (root / name).write_bytes(data)
    with _open(root, name) as file:
        return ts.scan_text(file)


def _refusal(root: Path, name: str, data: bytes) -> ts.TextScanRefused:
    (root / name).write_bytes(data)
    with _open(root, name) as file, pytest.raises(ts.TextScanRefused) as caught:
        ts.scan_text(file)
    assert caught.value.code in import_config.ISSUE_CODES
    return caught.value


# ---------------------------------------------------------------------------
# Line breaks
# ---------------------------------------------------------------------------


class TestLineBreaks:
    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            (b"a\nb\n", b"a\nb\n"),
            (b"a\r\nb\r\n", b"a\nb\n"),
            (b"a\r\r\nb", b"a\nb"),
            (b"a\r\r\r\r\nb", b"a\nb"),
            (b"a\rb\r", b"a\nb\n"),
            (b"a\r\rb", b"a\n\nb"),
            (b"a\r\n\rb", b"a\n\nb"),
            (b"\r", b"\n"),
            (b"", b""),
        ],
    )
    def test_every_break_becomes_one_newline(self, raw, normalized):
        assert lb.normalize_line_breaks(raw) == normalized

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64])
    def test_any_chunking_gives_the_same_output(self, chunk):
        rng = random.Random(chunk)
        for _ in range(300):
            data = bytes(rng.choice(b"ab\r\r\n ") for _ in range(rng.randint(0, 40)))
            normalizer = lb.LineBreakNormalizer()
            pieces = []
            for offset in range(0, len(data), chunk):
                pieces.extend(normalizer.feed(data[offset : offset + chunk]))
            pieces.extend(normalizer.finish())
            assert b"".join(pieces) == lb.normalize_line_breaks(data), data

    def test_long_cr_run_is_linear_and_streamed_in_bounded_pieces(self):
        """16 MiB of lone \\r fed in 1 MiB chunks: carried as a count, then
        emitted in pieces of at most MAX_PIECE_BYTES."""
        normalizer = lb.LineBreakNormalizer()
        chunk = b"\r" * (1 << 20)
        start = time.perf_counter()
        sizes = [len(piece) for _ in range(16) for piece in normalizer.feed(chunk)]
        sizes += [len(piece) for piece in normalizer.finish()]
        assert time.perf_counter() - start < 2.0
        assert sum(sizes) == 16 << 20
        assert max(sizes) <= lb.MAX_PIECE_BYTES

    def test_cr_run_then_lf_across_chunks_is_one_break(self):
        normalizer = lb.LineBreakNormalizer()
        out = b"".join(
            [
                *normalizer.feed(b"1\r"),
                *normalizer.feed(b"\r\r"),
                *normalizer.feed(b"\n2\n"),
                *normalizer.finish(),
            ]
        )
        assert out == b"1\n2\n"


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class TestScanContent:
    def test_lines_census_and_encoding(self, tmp_path):
        scan = _scan(tmp_path, "a.csv", b"rate;25600\r\n\r\nt;x\r\n0,1;2,5\r\n")
        assert [(line.number, line.text) for line in scan.head_lines] == [
            (1, "rate;25600"),
            (2, ""),
            (3, "t;x"),
            (4, "0,1;2,5"),
        ]
        assert scan.newline_census == {"lf": 0, "crlf": 4, "cr": 0}
        assert scan.encoding_candidates == ("utf-8",)
        assert scan.bom is None
        assert scan.complete and not scan.head_truncated
        assert scan.tail_lines == ()

    def test_mixed_breaks_are_counted_by_kind(self, tmp_path):
        scan = _scan(tmp_path, "a.csv", b"1\n2\r\n3\r4\r\r\n5")
        assert scan.newline_census == {"lf": 1, "crlf": 2, "cr": 1}
        assert [line.text for line in scan.head_lines] == ["1", "2", "3", "4", "5"]

    def test_utf8_bom_decides_the_encoding(self, tmp_path):
        scan = _scan(tmp_path, "a.csv", b"\xef\xbb\xbfTime;Acc\n0;1\n")
        assert scan.bom == "utf-8-sig"
        assert scan.encoding_candidates == ("utf-8-sig",)
        assert scan.head_lines[0].text == "Time;Acc"

    def test_cp1252_preamble_is_offered_not_raised(self, tmp_path):
        scan = _scan(tmp_path, "a.csv", b"Beschleunigung m/s\xb2;Temp \xb0C\n1;2\n")
        assert scan.encoding_candidates == ("cp1252",)
        assert (
            scan.head_lines[0].text
            == "Beschleunigung m/s" + chr(0xB2) + ";Temp " + chr(0xB0) + "C"
        )

    def test_neither_utf8_nor_cp1252_is_refused(self, tmp_path):
        refusal = _refusal(tmp_path, "a.csv", b"\xe9\x81\n1\n")
        assert refusal.code == "unsupported_encoding"

    def test_utf8_character_split_at_the_head_cap_is_kept_whole(self, tmp_path):
        prefix = b"x" * (ts.HEAD_BYTES - 1)
        data = prefix + "é".encode("utf-8") + b"\n1\n2\n"
        scan = _scan(tmp_path, "a.csv", data)
        assert scan.encoding_candidates == ("utf-8",)
        assert scan.head_lines[0].text.endswith(chr(0xE9))

    def test_head_stops_at_its_line_cap_and_tail_has_the_last_lines(self, tmp_path):
        data = b"".join(f"{n},{n * 2}\n".encode() for n in range(1, 20001))
        scan = _scan(tmp_path, "a.csv", data)
        assert len(scan.head_lines) == ts.HEAD_LINES
        assert scan.head_lines[-1].text == f"{ts.HEAD_LINES},{ts.HEAD_LINES * 2}"
        assert scan.head_truncated and not scan.complete
        assert scan.tail_lines[-1] == "20000,40000"
        assert all("," in line for line in scan.tail_lines)

    def test_a_small_file_past_the_line_cap_has_the_lines_after_the_head_as_tail(
        self, tmp_path
    ):
        """The head read the whole file (under HEAD_BYTES) but returns only
        HEAD_LINES lines: the rest are the tail, read from the same bytes."""
        lines = [f"{n},{n * 2}" for n in range(1, 401)]
        data = "".join(f"{line}\n" for line in lines).encode()
        scan = _scan(tmp_path, "a.csv", data)
        assert len(data) < ts.HEAD_BYTES
        assert scan.complete and scan.head_truncated
        assert len(scan.head_lines) == ts.HEAD_LINES
        assert scan.tail_lines == tuple(lines[ts.HEAD_LINES :])
        assert scan.tail_follows_head
        assert scan.bytes_read == len(data)  # nothing read twice or beyond
        assert scan.newline_census == {"lf": 400, "crlf": 0, "cr": 0}

    def test_that_tail_is_bounded_like_any_other(self, tmp_path):
        lines = [f"{n:08d},{n * 2:010d}" for n in range(1, 3001)]  # 60,000 bytes
        data = "".join(f"{line}\n" for line in lines).encode()
        scan = _scan(tmp_path, "a.csv", data)
        assert len(data) < ts.HEAD_BYTES and scan.complete
        kept = len(scan.tail_lines)
        assert 0 < kept < len(lines) - ts.HEAD_LINES
        assert scan.tail_lines == tuple(lines[-kept:])  # whole lines, the last ones
        assert sum(len(line) + 1 for line in scan.tail_lines) <= ts.TAIL_BYTES
        assert sum(len(line) + 1 for line in lines[-kept - 1 :]) > ts.TAIL_BYTES
        assert not scan.tail_follows_head  # lines were skipped before it

    def test_a_tail_read_from_where_a_short_head_ended_follows_it(self, tmp_path):
        """A head of long lines holds fewer than HEAD_LINES of them; the tail
        read from where it ended starts on the next line."""
        lines = [f"{n:05d}," + "7" * 273 for n in range(1, 251)]  # 70,000 bytes
        data = "".join(f"{line}\n" for line in lines).encode()
        scan = _scan(tmp_path, "a.csv", data)
        assert scan.complete and len(scan.head_lines) < ts.HEAD_LINES
        assert scan.tail_follows_head
        head = [line.text for line in scan.head_lines]
        assert head + list(scan.tail_lines) == lines  # every line once, in order

    def test_a_large_file_tail_does_not_follow_the_head(self, tmp_path):
        data = b"".join(f"{n},{n * 2}\n".encode() for n in range(1, 20001))
        scan = _scan(tmp_path, "a.csv", data)
        assert scan.tail_lines and not scan.tail_follows_head

    @pytest.mark.parametrize(
        ("line", "size"),
        [(b"1;23\r\r\n", 73_500), (b"1;2\r\r\n", 72_000), (b"1;23\r\n", 73_500)],
    )
    def test_a_break_split_at_the_head_cap_is_not_a_phantom_line(
        self, tmp_path, line, size
    ):
        """Python's csv module on Windows writes \\r\\r\\n; when the head cap
        falls inside that break, the tail must not start with an empty line."""
        data = line * (size // len(line))
        scan = _scan(tmp_path, "a.csv", data)
        expected = line.rstrip(b"\r\n").decode()
        assert all(text == expected for text in scan.tail_lines), scan.tail_lines[:3]
        assert all(entry.text == expected for entry in scan.head_lines)

    def test_the_line_cap_counts_the_line_without_its_break(self, tmp_path):
        """Like the decoder: a line of exactly LINE_CAP bytes is read."""
        exact = b"1;" + b"x" * (ts.LINE_CAP - 2)
        scan = _scan(tmp_path, "a.csv", exact + b"\n2;3\n")
        assert len(scan.head_lines[0].text) == ts.LINE_CAP
        assert scan.tail_lines == ("2;3",)
        refused = _refusal(tmp_path, "b.csv", exact + b"x\n2;3\n")
        assert refused.code == "line_too_long"

    def test_an_empty_file_has_no_lines(self, tmp_path):
        scan = _scan(tmp_path, "a.csv", b"")
        assert scan.head_lines == () and scan.tail_lines == () and scan.complete

    def test_a_tail_without_any_break_has_no_lines(self, tmp_path):
        data = b"1;2\n" + b"x" * (ts.HEAD_BYTES + ts.TAIL_BYTES + 10)
        scan = _scan(tmp_path, "a.csv", data)
        assert scan.tail_lines == ()

    def test_a_bom_before_invalid_utf8_is_refused(self, tmp_path):
        refusal = _refusal(tmp_path, "a.csv", b"\xef\xbb\xbf1;2\n\xff\n")
        assert refusal.code == "unsupported_encoding"
        assert "byte-order mark" in str(refusal)

    def test_scan_writes_nothing(self, tmp_path):
        target = tmp_path / "a.csv"
        target.write_bytes(b"1;2\n3;4\n")
        before = sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.iterdir()
        )
        with _open(tmp_path, "a.csv") as file:
            ts.scan_text(file)
        after = sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.iterdir()
        )
        assert before == after


class TestScanRefusals:
    def test_nul_byte_is_refused_before_any_text(self, tmp_path):
        refusal = _refusal(tmp_path, "a.csv", b"1;2\n3\x00;4\n")
        assert refusal.code == "binary_content"
        assert "NUL" in str(refusal)

    def test_nul_byte_in_the_tail_is_refused(self, tmp_path):
        data = b"1;2\n" * 30000 + b"5;\x006\n"
        assert _refusal(tmp_path, "a.csv", data).code == "binary_content"

    def test_utf16_bom_is_a_named_limit(self, tmp_path):
        refusal = _refusal(tmp_path, "a.csv", "1;2\n".encode("utf-16"))
        assert refusal.code == "unsupported_encoding"
        assert "UTF-16" in str(refusal)

    @pytest.mark.parametrize("codec", ["utf-32-le", "utf-32-be"])
    def test_utf32_bom_is_recognized_as_utf32(self, tmp_path, codec):
        data = "﻿1;2\n".replace("﻿", chr(0xFEFF)).encode(codec)
        refusal = _refusal(tmp_path, "a.csv", data)
        assert refusal.code == "unsupported_encoding"
        assert "UTF-32" in str(refusal) and "UTF-16" not in str(refusal)

    @pytest.mark.parametrize(
        ("data", "code", "label"),
        [
            (b"\x1f\x8b\x08\x00rest", "compressed_input", "gzip"),
            (b"PK\x03\x04rest", "compressed_input", "zip"),
            (b"BZh91AY&SYrest", "compressed_input", "bzip2"),
            (b"MATLAB 5.0 MAT-file, Platform: x", "binary_content", "MATLAB MAT (v5)"),
            (b"\x89HDF\r\n\x1a\nrest", "binary_content", "HDF5"),
            (b"TDSm rest", "binary_content", "TDMS"),
            (b"RIFF\x24\x00\x00\x00WAVE", "binary_content", "RIFF (WAV)"),
        ],
    )
    def test_known_signatures_are_refused_naming_the_format(
        self, tmp_path, data, code, label
    ):
        refusal = _refusal(tmp_path, "a.csv", data)
        assert refusal.code == code
        assert label in str(refusal)

    def test_a_4_mb_single_line_is_refused_by_the_line_cap(self, tmp_path):
        data = b"1," * (2 * 1024 * 1024) + b"\n"
        (tmp_path / "a.csv").write_bytes(data)
        with _open(tmp_path, "a.csv") as file:
            with pytest.raises(ts.TextScanRefused) as caught:
                ts.scan_text(file)
            assert file.bytes_read <= ts.HEAD_BYTES + ts.LINE_CAP + 64 * 1024
        assert caught.value.code == "line_too_long"
        assert "line 1" in str(caught.value)

    def test_too_many_fields_under_every_delimiter_is_refused(self, tmp_path):
        row = ";".join(["1,2\t3 4"] * 1100) + "\n"
        assert _refusal(tmp_path, "a.csv", row.encode()).code == "too_many_fields"

    def test_many_commas_are_fine_when_another_delimiter_fits(self, tmp_path):
        row = (";".join(["1"] * 4) + "," * 2000 + "\n").encode()
        scan = _scan(tmp_path, "a.csv", row)
        assert len(scan.head_lines) == 1

    def test_refusals_do_not_repeat_the_file_name(self, tmp_path):
        """A Hangul filler passes the path gate (it is a letter) but prints as
        nothing; a refusal carries no file text, so it cannot show it."""
        name = "evil" + chr(0x3164) + ".csv"
        try:
            (tmp_path / name).write_bytes(b"\x00")
        except OSError:
            pytest.skip("filesystem refuses the character")
        with _open(tmp_path, name) as file, pytest.raises(ts.TextScanRefused) as caught:
            ts.scan_text(file)
        assert chr(0x3164) not in str(caught.value)
        assert "\\u3164" not in str(caught.value)
        assert "evil" not in str(caught.value)
        assert str(caught.value).startswith("The file ")


class _VirtualFile(VerifiedFile):
    """A 400 MB file that exists only as a pattern: it counts the bytes the
    scan asks for, without touching the disk."""

    LINE = b"0.123456;1.234567;2.345678\n"

    def __init__(self, size: int) -> None:  # noqa: D401 - test double
        self._fd = None
        self.path = Path("virtual.csv")
        self.size = size
        self.bytes_read = 0

    def read(self, offset: int, length: int) -> bytes:
        end = min(offset + max(length, 0), self.size)
        if offset >= end:
            return b""
        period = len(self.LINE)
        first = offset // period
        last = (end - 1) // period + 1
        data = self.LINE * (last - first)
        result = data[offset - first * period : end - first * period]
        self.bytes_read += len(result)
        return result


def test_a_400_mb_file_reads_only_head_and_tail():
    file = _VirtualFile(400 * 1000 * 1000)
    scan = ts.scan_text(file)
    assert scan.bytes_read == file.bytes_read
    assert file.bytes_read <= ts.HEAD_BYTES + 4096 + ts.TAIL_BYTES + 1
    assert scan.head_lines[0].text == "0.123456;1.234567;2.345678"
    assert scan.tail_lines and scan.head_truncated and not scan.complete


def test_every_scan_code_is_an_issue_code():
    for _, code, _ in ts.SIGNATURES:
        assert code in import_config.ISSUE_CODES
    assert os.path.basename(ts.__file__) == "text_scan.py"
