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

    def test_refusals_escape_the_file_name(self, tmp_path):
        """A Hangul filler passes the path gate (it is a letter) but prints as
        nothing, so the refusal shows it escaped."""
        name = "evil" + chr(0x3164) + ".csv"
        try:
            (tmp_path / name).write_bytes(b"\x00")
        except OSError:
            pytest.skip("filesystem refuses the character")
        with _open(tmp_path, name) as file, pytest.raises(ts.TextScanRefused) as caught:
            ts.scan_text(file)
        assert chr(0x3164) not in str(caught.value)
        assert "\\u3164" in str(caught.value)


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
