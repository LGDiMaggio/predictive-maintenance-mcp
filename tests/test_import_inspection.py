"""Tests for ``signal_import.inspection``: the read-only inspection.

Files are synthesized (``import_corpus``) into a sandboxed data directory, so
every claim runs against real bytes on disk through the real path gate.
"""

import builtins
import io
import json
import os
import time
from pathlib import Path

import import_corpus as corpus
import pytest

from predictive_maintenance_mcp.file_text import is_escaped_file_text
from predictive_maintenance_mcp.path_safety import PathRefused
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    ISSUE_CODES,
    PREVIEW_UNIT_TOKENS,
)
from predictive_maintenance_mcp.signal_acquisition.text_scan import TextScanRefused
from predictive_maintenance_mcp.signal_import import inspection as inspection_module
from predictive_maintenance_mcp.signal_import.inspection import (
    COLUMNS_PER_PAGE,
    HEAD_TOKENS,
    OUTPUT_BUDGET_BYTES,
    PREVIEW_ROLES,
    inspect_signal_file,
)


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.write_bytes(data)
        return path

    return _write


def codes(result) -> list[str]:
    return [issue.code for issue in result.issues]


def column(result, index):
    return next(c for c in result.columns if c.index == index)


def region(result, kind):
    return next((r for r in result.regions if r.kind == kind), None)


def dumped(result) -> str:
    return result.model_dump_json()


class TestAcceptance:
    def test_semicolon_decimal_comma_export_with_preamble(self, write):
        """A six-line preamble, a names row and a units row: the dialect is
        proposed, data starts on line 9, unit tokens are detected evidence
        with their line."""
        path = write("export.csv", corpus.european_export())
        result = inspect_signal_file(str(path))

        assert result.status == "importable"
        assert result.dialect.status == "proposed"
        proposal = result.dialect.proposal
        assert (proposal.delimiter, proposal.decimal, proposal.data_first_line) == (
            ";",
            ",",
            9,
        )
        assert [(r.kind, r.first_line, r.last_line) for r in result.regions] == [
            ("preamble", 1, 6),
            ("header", 7, 7),
            ("units", 8, 8),
            ("data", 9, None),
        ]
        units = [
            (c.index, c.unit_token.file_text, c.unit_line, c.unit)
            for c in result.columns
        ]
        assert units == [
            (0, "s", 8, None),
            (1, "g", 8, "g"),
            (2, "g", 8, "g"),
            (3, "g", 8, "g"),
        ]
        assert [c.label.file_text for c in result.columns] == [
            "Time",
            "AccX",
            "AccY",
            "AccZ",
        ]
        assert [kv.line for kv in result.preamble.key_values] == [1, 2, 3, 4, 5, 6]
        assert result.preamble.lines == []
        stated = {(c.field, c.status, c.normalized, c.line) for c in result.candidates}
        assert ("sampling_rate_hz", "detected", "25600", 2) in stated
        assert ("sensitivity", "detected", "100", 3) in stated
        assert ("acquired_at", "detected", None, 4) in stated

    def test_rows_like_1_5_are_ambiguous_with_both_readings(self, write):
        result = inspect_signal_file(
            str(write("ambiguous.csv", corpus.ambiguous_rows()))
        )
        assert result.dialect.status == "ambiguous"
        assert result.dialect.proposal is None
        readings = {
            (c.delimiter, c.decimal, c.fields) for c in result.dialect.candidates
        }
        assert readings == {(",", ".", 2), (";", ",", 1)}
        assert "ambiguous_dialect" in codes(result)
        assert any("dialect" in item for item in result.required_confirmations)

    def test_header_one_field_short_is_a_mismatch_and_shows_no_label(self, write):
        result = inspect_signal_file(
            str(write("short.csv", corpus.header_short_by_one()))
        )
        assert "field_count_mismatch" in codes(result)
        mismatch = next(i for i in result.issues if i.code == "field_count_mismatch")
        assert mismatch.line == 1 and "column index" in mismatch.message
        assert result.columns_total == 4
        assert all(c.label is None for c in result.columns)
        assert [c.head_tokens[0].file_text for c in result.columns][:2] == [
            "0,000",
            "0,010",
        ]

    def test_instruction_like_preamble_is_escaped_capped_file_text(self, write):
        path = write("inj.csv", corpus.instruction_preamble())
        result = inspect_signal_file(str(path))
        texts = [line.text for line in result.preamble.lines]
        assert [t.file_text for t in texts] == ["Recorder export", corpus.INJECTION]
        payload = json.loads(dumped(result))

        # The text appears only as the value of a "file_text" key.
        def occurrences(node, under_file_text=False):
            if isinstance(node, dict):
                return sum(occurrences(v, k == "file_text") for k, v in node.items())
            if isinstance(node, list):
                return sum(occurrences(v) for v in node)
            if isinstance(node, str) and corpus.INJECTION in node:
                assert under_file_text, node
                return 1
            return 0

        assert occurrences(payload) == 1
        assert all(corpus.INJECTION not in issue.message for issue in result.issues)


class TestReadOnly:
    def _snapshot(self, *folders: Path) -> dict:
        state = {}
        for folder in folders:
            if not folder.exists():
                state[str(folder)] = None
                continue
            for path in sorted(folder.rglob("*")):
                status = path.stat()
                state[str(path)] = (
                    path.is_dir(),
                    status.st_size,
                    status.st_mtime_ns,
                    None if path.is_dir() else path.read_bytes(),
                )
        return state

    def test_inspection_writes_nothing_anywhere(
        self, write, sandbox_data_dir, ledger_dir, import_dir
    ):
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            get_repository,
        )

        write("export.csv", corpus.european_export())
        write("export_metadata.json", b'{"signal_unit": "g"}')
        write("short.csv", corpus.header_short_by_one())
        repository = get_repository()
        signals_before = repository.list_signals()
        before = self._snapshot(sandbox_data_dir, ledger_dir, import_dir)
        inspect_signal_file(str(sandbox_data_dir / "export.csv"))
        inspect_signal_file("short.csv")
        with pytest.raises(PathRefused):
            inspect_signal_file(str(sandbox_data_dir / "export_metadata.json"))
        assert self._snapshot(sandbox_data_dir, ledger_dir, import_dir) == before
        assert repository.list_signals() == signals_before

    def test_companion_is_reported_without_being_read(self, write, monkeypatch):
        path = write("export.csv", corpus.european_export())
        companion = write("export_metadata.json", b'{"signal_unit": "MARKER-123"}')
        opened: list[str] = []

        def recorder(real):
            def record(target, *args, **kwargs):
                if isinstance(target, (str, bytes, os.PathLike)):
                    opened.append(os.fsdecode(target))
                return real(target, *args, **kwargs)

            return record

        # Every way to open a file: the descriptor call, the built-in and io.
        with monkeypatch.context() as scoped:
            scoped.setattr(os, "open", recorder(os.open))
            scoped.setattr(builtins, "open", recorder(builtins.open))
            scoped.setattr(io, "open", recorder(io.open))
            result = inspect_signal_file(str(path))
        assert "companion_present" in codes(result)
        assert all(Path(p).name != companion.name for p in opened)
        assert "MARKER-123" not in dumped(result)
        assert any("companion" in item for item in result.required_confirmations)


class TestPathAndFileRefusals:
    def test_path_outside_the_roots_is_refused_before_any_stat_or_open(
        self, sandbox_data_dir, tmp_path, monkeypatch
    ):
        outside = tmp_path / "elsewhere" / "data.csv"
        outside.parent.mkdir()
        outside.write_bytes(corpus.plain_columns(10, 2))
        calls: list[str] = []
        for name in ("open", "stat", "lstat"):
            real = getattr(os, name)

            def recorder(*args, _real=real, _name=name, **kwargs):
                calls.append(_name)
                return _real(*args, **kwargs)

            monkeypatch.setattr(os, name, recorder)
        try:
            with pytest.raises(PathRefused) as caught:
                inspect_signal_file(str(outside))
        finally:
            monkeypatch.undo()
        assert caught.value.code == "path_outside_roots"
        assert calls == []

    def test_upper_case_suffix_is_accepted(self, write):
        result = inspect_signal_file(str(write("EXPORT.CSV", corpus.european_export())))
        assert result.status == "importable"

    def test_dat_suffix_is_refused_with_the_rename_remedy(self, write):
        path = write("export.dat", corpus.european_export())
        with pytest.raises(PathRefused) as caught:
            inspect_signal_file(str(path))
        assert caught.value.code == "unsupported_suffix"
        assert "renamed to .csv" in str(caught.value)

    def test_binary_content_is_refused_before_any_text(self, write):
        path = write("zero.csv", b"1.0,2.0\n3.0,\x004.0\n")
        with pytest.raises(TextScanRefused) as caught:
            inspect_signal_file(str(path))
        assert caught.value.code == "binary_content"

    def test_negative_page_start_is_refused(self, write):
        path = write("export.csv", corpus.european_export())
        with pytest.raises(ValueError, match="columns_from"):
            inspect_signal_file(str(path), columns_from=-1)


class TestNotImportable:
    def test_feature_table_is_not_a_waveform(self, write):
        result = inspect_signal_file(str(write("features.csv", corpus.feature_table())))
        assert result.status == "not_importable"
        assert "not_a_waveform" in codes(result)
        assert result.columns_total == 17

    def test_header_only_file(self, write):
        result = inspect_signal_file(str(write("header.csv", b"time;acc_x;acc_y\n")))
        assert result.status == "not_importable"
        assert codes(result) == ["no_numeric_region"]
        assert "text only" in result.issues[0].message

    def test_rows_over_the_field_cap_are_not_called_text(self, write):
        # F2: 1,025 numeric columns, one more than this version reads; rows
        # of about 9 KB, so the 64 KB head holds 8 of them.
        result = inspect_signal_file(
            str(write("wide.csv", corpus.plain_columns(400, 1025)))
        )
        assert result.status == "not_importable"
        assert codes(result) == ["no_numeric_region"]
        assert result.issues[0].message == (
            "No row of numbers within the 1024-field limit of this version was "
            "found in the first 8 lines, the only lines searched for rows: lines "
            "1, 2, 3, 4, 5, 6, 7, 8 have more fields than that, so they are not "
            "read as rows, and the file may hold more columns than this version "
            "imports; rows further on in the file are not ruled out."
        )

    def test_one_row_over_the_field_cap_in_a_fully_scanned_file(self, write):
        wide = ",".join(f"{value}.5" for value in range(1025))
        data = corpus.lines_to_bytes(["channels follow", wide])
        result = inspect_signal_file(str(write("one-wide.csv", data)))
        assert result.issues[0].message == (
            "No row of numbers within the 1024-field limit of this version was "
            "found in the scanned lines: line 2 has more fields than that, so it "
            "is not read as a row, and the file may hold more columns than this "
            "version imports."
        )

    def test_wide_text_header_is_not_counted_as_a_wide_row(self, write):
        # 600 labels of three words: over the cap only when split at spaces,
        # and no number among the fields, so it is no row at all.
        header = ",".join(f"Ch {chr(65 + index % 26)} acc" for index in range(600))
        notes = ["note text here" for _ in range(400)]
        result = inspect_signal_file(
            str(write("wide-header.csv", corpus.lines_to_bytes([header] + notes)))
        )
        assert result.issues[0].message == (
            "No row of numbers was found in the first 300 lines, the only lines "
            "searched for rows; rows further on in the file are not ruled out."
        )

    def test_text_head_of_a_longer_file_claims_nothing_past_it(self, write):
        notes = [f"note {index}: plain text" for index in range(350)]
        data = corpus.lines_to_bytes(notes + ["time,value"] + _rows(400))
        result = inspect_signal_file(str(write("late-data.csv", data)))
        assert codes(result) == ["no_numeric_region"]
        assert result.issues[0].message == (
            "No row of numbers was found in the first 300 lines, the only lines "
            "searched for rows; rows further on in the file are not ruled out."
        )

    def test_empty_file(self, write):
        result = inspect_signal_file(str(write("empty.csv", b"")))
        assert result.status == "not_importable"
        assert result.issues[0].code == "no_numeric_region"
        assert result.issues[0].message == "The file is empty."
        assert result.file.ends_with_line_break is None

    def test_no_numeric_region_echoes_no_preamble_or_label_text(self, write):
        data = corpus.lines_to_bytes(
            ["Recorder SECRET-LABEL", corpus.INJECTION, "time;acc;SECRET-HEADER"]
        )
        result = inspect_signal_file(str(write("text.csv", data)))
        text = dumped(result)
        assert "SECRET" not in text and corpus.INJECTION not in text
        assert result.preamble.lines == [] and result.preamble.key_values == []
        assert result.columns == []


class TestFileText:
    INVISIBLE = [0xFE0F, 0xE0100, 0x034F, 0x3164, 0x115F, 0x2800]

    def test_invisible_characters_in_labels_come_back_escaped(self, write):
        labels = [
            f"acc{chr(code)}x{index}" for index, code in enumerate(self.INVISIBLE)
        ]
        path = write(
            "labels.csv", corpus.plain_columns(400, len(labels), labels=labels)
        )
        result = inspect_signal_file(str(path))
        for code, col in zip(self.INVISIBLE, result.columns):
            escaped = col.label.file_text
            assert chr(code) not in escaped
            assert ("\\u%04x" % code if code < 0x10000 else "\\U%08x" % code) in escaped
            assert is_escaped_file_text(escaped)

    def test_label_cap_counts_the_escaped_length(self, write):
        label = chr(0x2800) * 100  # 600 escaped characters
        path = write("long.csv", corpus.plain_columns(400, 1, labels=[label]))
        result = inspect_signal_file(str(path))
        shown = result.columns[0].label
        assert shown.truncated and len(shown.file_text) <= 120

    def test_micro_sign_unit_is_kept_verbatim_and_never_mapped(self, write):
        data = corpus.lines_to_bytes(
            ["a;b", f"[{chr(0xB5)}m];[m/s{chr(0xB2)}]"]
            + [f"{i},0;{i},5" for i in range(400)]
        )
        result = inspect_signal_file(str(write("micro.csv", data)))
        micro, metric = result.columns
        assert micro.unit is None and micro.unit_token.file_text == f"[{chr(0xB5)}m]"
        assert metric.unit == "m/s2"
        assert "unit_token_unrecognized" in codes(result)


class TestUnitsAndRoles:
    def test_unit_tokens_map_to_roles(self, write):
        tokens = ["s", "[g]", "mm/s", "mV", "rpm", "[mg]", "degC"]
        header = ";".join(f"c{i}" for i in range(len(tokens)))
        rows = [
            ";".join(
                [f"{row / 1000:.3f}".replace(".", ",")] + ["1,5"] * (len(tokens) - 1)
            )
            for row in range(400)
        ]
        data = corpus.lines_to_bytes([header, ";".join(tokens)] + rows)
        result = inspect_signal_file(str(write("units.csv", data)))
        roles = [(c.unit, c.role_candidate) for c in result.columns]
        assert roles == [
            (None, "time"),
            ("g", "vibration"),
            ("mm/s", "vibration"),
            (None, "voltage"),
            (None, "tachometer"),
            (None, None),
            (None, "temperature"),
        ]
        assert "unit_token_preview_only" in codes(result)
        assert "unit_token_unrecognized" in codes(result)

    def test_preview_roles_cover_the_preview_vocabulary(self):
        assert set(PREVIEW_ROLES) == {token.lower() for token in PREVIEW_UNIT_TOKENS}

    def test_label_brackets_supply_the_unit_token(self, write):
        path = write(
            "brackets.csv", corpus.plain_columns(400, 2, labels=["Acc [g]", "Acc (mV)"])
        )
        result = inspect_signal_file(str(path))
        assert [
            (c.unit_token.file_text, c.unit_line, c.unit) for c in result.columns
        ] == [
            ("g", 1, "g"),
            ("mV", 1, None),
        ]


class TestTimeCandidates:
    def test_time_column_with_unit_derives_a_rate_from_the_scanned_rows(self, write):
        result = inspect_signal_file(str(write("export.csv", corpus.european_export())))
        time = column(result, 0)
        assert time.role_candidate == "time"
        assert time.time_stats.evidence == "full"
        assert time.time_stats.modal_steps == [39, 40]
        derived = [c for c in result.candidates if c.status == "derived"]
        assert [(c.field, c.column) for c in derived] == [("sampling_rate_hz", 0)]
        # Fitted to the scanned rows only, so close to, not exactly, 25600.
        assert abs(float(derived[0].normalized) - 25600) < 0.1

    def test_four_decimals_at_25600_hz_derive_no_rate(self, write):
        rows = [f"{row / 25600:.4f},{corpus.sample(row, 0)}" for row in range(400)]
        data = corpus.lines_to_bytes(["time [s],acc [g]"] + rows)
        result = inspect_signal_file(str(write("coarse.csv", data)))
        assert column(result, 0).time_stats.evidence == "duration_only"
        assert not [c for c in result.candidates if c.status == "derived"]

    def test_monotonic_column_without_unit_gets_no_rate(self, write):
        rows = [f"{row},{corpus.sample(row, 0)}" for row in range(400)]
        result = inspect_signal_file(
            str(write("index.csv", corpus.lines_to_bytes(rows)))
        )
        first = column(result, 0)
        assert first.role_candidate == "time"
        assert first.time_stats.time_unit is None
        assert not [c for c in result.candidates if c.status == "derived"]

    def test_date_time_text_column_is_a_time_candidate_not_assessed(self, write):
        rows = [f"2026-08-20 13:42:{row % 60:02d};{row}.5" for row in range(400)]
        data = corpus.lines_to_bytes(["when;acc"] + rows)
        result = inspect_signal_file(str(write("dates.csv", data)))
        when = column(result, 0)
        assert when.role_candidate == "time"
        assert (
            when.time_stats.datetime_text
            and when.time_stats.refused_code == "time_grammar"
        )


class TestStructure:
    def test_trailing_delimiter_column_and_unselected_text_column(self, write):
        rows = [f"20.08.2026;{row},5;{row},25;" for row in range(400)]
        result = inspect_signal_file(str(write("f15.csv", corpus.lines_to_bytes(rows))))
        assert (result.dialect.proposal.delimiter, result.dialect.proposal.decimal) == (
            ";",
            ",",
        )
        assert result.columns_total == 3
        assert "trailing_delimiter_column" in codes(result)

    def test_missing_final_line_break_is_reported_up_front(self, write):
        data = corpus.lines_to_bytes([f"{row},0.5" for row in range(400)], final=False)
        result = inspect_signal_file(str(write("open.csv", data)))
        assert result.file.ends_with_line_break is False
        assert "truncated_last_line" in codes(result)

    def test_mixed_line_breaks_are_counted(self, write):
        data = b"a,b\r\n" + b"".join(
            f"{i},0.5".encode() + (b"\n", b"\r\n", b"\r")[i % 3] for i in range(400)
        )
        result = inspect_signal_file(str(write("mixed.csv", data)))
        census = result.file.newline_census
        assert census["lf"] > 0 and census["crlf"] > 0 and census["cr"] > 0
        assert result.dialect.proposal.data_first_line == 2

    def test_excel_sep_line_is_reported_as_preamble(self, write):
        data = corpus.lines_to_bytes(
            ["sep=;", "a;b"] + [f"{i};0,5" for i in range(400)]
        )
        result = inspect_signal_file(str(write("sep.csv", data)))
        assert [kv.key.file_text for kv in result.preamble.key_values] == ["sep"]
        assert result.dialect.proposal.data_first_line == 3

    def test_windows_1252_needs_encoding_confirmation(self, write):
        data = "Température;acc\n".encode("cp1252") + b"".join(
            f"{i};0,5\n".encode() for i in range(400)
        )
        result = inspect_signal_file(str(write("cp.csv", data)))
        assert result.file.encoding.status == "ambiguous"
        assert "encoding_needs_confirmation" in codes(result)

    def test_utf8_bom_is_detected(self, write):
        data = b"\xef\xbb\xbf" + corpus.plain_columns(400, 2)
        result = inspect_signal_file(str(write("bom.csv", data)))
        assert (result.file.encoding.value, result.file.encoding.status) == (
            "utf-8-sig",
            "detected",
        )

    def test_trailer_in_the_tail_is_counted(self, write):
        data = corpus.plain_columns(5000, 2) + b"END OF DATA\nchecksum,abc\n"
        result = inspect_signal_file(str(write("trailer.csv", data)))
        trailer = region(result, "trailer")
        assert trailer is not None and trailer.lines == 2 and trailer.first_line is None
        assert region(result, "data").last_line is None

    def test_blank_lines_in_the_tail_trailer_are_not_counted(self, write):
        data = corpus.plain_columns(5000, 2) + b"END OF DATA\n\nchecksum,abc\n\n"
        result = inspect_signal_file(str(write("trailer2.csv", data)))
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (None, 2)

    def test_trailer_after_the_data_of_a_fully_scanned_file(self, write):
        # 285 lines: the whole file is in the scanned head.
        lines = ["time,value"] + _rows(280) + ["", "END OF DATA", "", "rows,280"]
        result = inspect_signal_file(
            str(write("short.csv", corpus.lines_to_bytes(lines)))
        )
        assert result.file.scan.complete and result.status == "importable"
        data = region(result, "data")
        assert (data.first_line, data.last_line, data.lines) == (2, 281, 280)
        trailer = region(result, "trailer")
        # Counted from its first non-blank line; blank lines are not trailer lines.
        assert (trailer.first_line, trailer.last_line, trailer.lines) == (283, None, 2)

    def test_trailer_past_line_300_of_a_file_under_64_kb_is_found(self, write):
        # F1: the head read the whole file, but only its first 300 lines are
        # head lines; the trailer lies after them.
        lines = ["time,value"] + _rows(400) + ["", "END OF DATA", "", "rows,400"]
        data = corpus.lines_to_bytes(lines)
        result = inspect_signal_file(str(write("mid.csv", data)))
        assert len(data) < 64 * 1024 and result.file.scan.complete
        assert region(result, "data").last_line is None  # past the scanned head
        trailer = region(result, "trailer")
        # The tail follows the head, so its lines are numbered: 402 is blank.
        assert (trailer.first_line, trailer.last_line, trailer.lines) == (
            403,
            None,
            2,
        )

    def test_trailer_after_blank_lines_past_line_300_is_found(self, write):
        # The data ends at line 291; only blank lines follow it in the head.
        lines = ["time,value"] + _rows(290) + [""] * 20 + ["END OF DATA", "rows,290"]
        result = inspect_signal_file(
            str(write("gap.csv", corpus.lines_to_bytes(lines)))
        )
        assert region(result, "data").last_line == 291
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (312, 2)

    def test_trailer_starting_at_line_301_is_counted_whole(self, write):
        lines = ["time,value"] + _rows(299) + ["END", "x y z"]
        result = inspect_signal_file(
            str(write("edge301.csv", corpus.lines_to_bytes(lines)))
        )
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (301, 2)

    def test_trailer_across_line_300_of_a_small_file_is_counted_whole(self, write):
        # Data ends at line 271; the trailer runs from line 272 to line 371,
        # all of it read (the tail follows the head).
        notes = [f"note {index}: summary text" for index in range(100)]
        lines = ["time,value"] + _rows(270) + notes
        result = inspect_signal_file(
            str(write("long-trailer.csv", corpus.lines_to_bytes(lines)))
        )
        assert region(result, "data").last_line == 271
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (272, 100)

    def test_trailer_running_past_the_scanned_head_has_no_stated_length(self, write):
        # A large file: lines between the head and the tail are not read.
        notes = [f"note {index:05d}: summary text" for index in range(5000)]
        lines = ["time,value"] + _rows(270) + notes
        result = inspect_signal_file(
            str(write("long-trailer2.csv", corpus.lines_to_bytes(lines)))
        )
        assert not result.file.scan.complete
        assert region(result, "data").last_line == 271
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (272, None)

    def test_trailer_filling_the_whole_tail_has_no_stated_length(self, write):
        # No data row in the last 16 KB: where the trailer starts is unknown.
        notes = [f"note {index:05d}: summary text" for index in range(1500)]
        data = corpus.plain_columns(5000, 2) + corpus.lines_to_bytes(notes)
        result = inspect_signal_file(str(write("tail-trailer.csv", data)))
        trailer = region(result, "trailer")
        assert (trailer.first_line, trailer.lines) == (None, None)

    def test_blank_line_between_preamble_and_header(self, write):
        lines = ["Sampling rate: 25600", "", "time,value"] + _rows(400)
        result = inspect_signal_file(
            str(write("blank.csv", corpus.lines_to_bytes(lines)))
        )
        preamble, header = region(result, "preamble"), region(result, "header")
        assert (preamble.first_line, preamble.last_line) == (1, 1)
        assert header.first_line == 3
        assert result.dialect.proposal.data_first_line == 4
        assert [c.label.file_text for c in result.columns] == ["time", "value"]
        stated = [c for c in result.candidates if c.status == "detected"]
        assert [(c.field, c.normalized, c.line) for c in stated] == [
            ("sampling_rate_hz", "25600", 1)
        ]

    def test_header_and_units_row_ending_with_the_delimiter(self, write):
        rows = [f"{r / 25600:.6f},{corpus.sample(r, 0):.6f}," for r in range(400)]
        data = corpus.lines_to_bytes(["time,acc,", "s,g,"] + rows)
        result = inspect_signal_file(str(write("trail.csv", data)))
        # The empty last field of every line is dropped, so labels and units
        # still match the two columns.
        assert codes(result) == ["trailing_delimiter_column"]
        assert result.columns_total == 2
        assert [c.label.file_text for c in result.columns] == ["time", "acc"]
        assert [c.unit_token.file_text for c in result.columns] == ["s", "g"]
        assert [c.unit_line for c in result.columns] == [2, 2]


class TestPagingAndBudget:
    def test_columns_are_paged(self, write):
        path = write("wide.csv", corpus.plain_columns(400, 40))
        first = inspect_signal_file(str(path))
        assert [c.index for c in first.columns] == list(range(COLUMNS_PER_PAGE))
        assert first.next_columns_from == COLUMNS_PER_PAGE
        second = inspect_signal_file(str(path), columns_from=first.next_columns_from)
        assert [c.index for c in second.columns] == list(range(32, 40))
        assert second.next_columns_from is None

    def test_every_cap_filled_stays_within_the_budget(self, write):
        result = inspect_signal_file(str(write("capped.csv", corpus.capped_export())))
        size = len(dumped(result).encode("utf-8"))
        assert size <= OUTPUT_BUDGET_BYTES
        assert result.columns_total == 200
        assert 1 <= len(result.columns) <= COLUMNS_PER_PAGE
        assert result.next_columns_from == result.columns_from + len(result.columns)
        assert all(len(c.head_tokens) <= HEAD_TOKENS for c in result.columns)
        # Preamble text appears once: key-value lines are not repeated as lines.
        assert len(result.preamble.key_values) == 40
        assert result.preamble.lines == []
        assert dumped(result).count("v" * 193) == 40

    def test_pages_cover_every_column_once_and_repeat_identically(self, write):
        path = write("capped.csv", corpus.capped_export())
        seen: list[int] = []
        start = 0
        while start is not None:
            first = inspect_signal_file(str(path), columns_from=start)
            again = inspect_signal_file(str(path), columns_from=start)
            assert dumped(first) == dumped(again)  # deterministic
            assert len(dumped(first).encode("utf-8")) <= OUTPUT_BUDGET_BYTES
            seen.extend(c.index for c in first.columns)
            start = first.next_columns_from
        assert seen == list(range(200))

    def test_omitted_count_is_exact_and_confirmations_survive(self, write):
        preamble = ["fs: " + "\\" * 300 for _ in range(45)]  # 5 past the line cap
        data = corpus.lines_to_bytes(preamble + ["time,value"] + _rows(400))
        result = inspect_signal_file(str(write("heavy2.csv", data)))
        shown = len(result.preamble.lines) + len(result.preamble.key_values)
        assert result.preamble.omitted_lines == 45 - shown
        assert len(result.required_confirmations) == 3
        # The column page shrank first; the caller pages on to the rest.
        assert len(result.columns) >= 1
        assert result.next_columns_from == result.columns_from + len(result.columns)

    def test_plain_preamble_lines_drop_before_key_values(self, write):
        plain = ["\\" * 300 for _ in range(20)]  # lines 1 to 20, no key
        pairs = [f"k{i:02d}" + "\\" * 100 + ": " + "\\" * 300 for i in range(20)]
        path = write(
            "mixed.csv",
            corpus.lines_to_bytes(plain + pairs + ["time,value"] + _rows(400)),
        )
        result = inspect_signal_file(str(path))
        assert len(dumped(result).encode("utf-8")) <= OUTPUT_BUDGET_BYTES
        # The column page shrank first; the caller pages on to the rest.
        assert len(result.columns) == 1 and result.next_columns_from == 1
        # Then plain lines went from the end, before any key-value pair.
        shown = [line.line for line in result.preamble.lines]
        assert 0 < len(shown) < 20 and shown == list(range(1, len(shown) + 1))
        assert [pair.line for pair in result.preamble.key_values] == list(range(21, 41))
        assert result.preamble.omitted_lines == 20 - len(shown)
        assert len(result.required_confirmations) == 3
        assert dumped(inspect_signal_file(str(path))) == dumped(result)

    def test_issue_codes_belong_to_the_closed_vocabulary(self, write):
        for name, data in (
            ("a.csv", corpus.european_export()),
            ("b.csv", corpus.ambiguous_rows()),
            ("c.csv", corpus.header_short_by_one()),
            ("d.csv", corpus.feature_table()),
        ):
            result = inspect_signal_file(str(write(name, data)))
            assert set(codes(result)) <= set(ISSUE_CODES)


def _rows(count: int, *, sep: str = ",") -> list[str]:
    return [
        f"{row / 25600:.6f}{sep}{corpus.sample(row, 0):.6f}" for row in range(count)
    ]


class TestCraftedInputStaysBounded:
    """Long runs of spaces in untrusted text must not make matching quadratic."""

    LIMIT_SECONDS = 2.0

    def _timed(self, write, name: str, lines: list[str]) -> tuple[float, object]:
        path = write(name, corpus.lines_to_bytes(lines))
        started = time.perf_counter()
        result = inspect_signal_file(str(path))
        return time.perf_counter() - started, result

    def test_space_padded_preamble_line(self, write):
        elapsed, result = self._timed(
            write, "pad.csv", [" " * 60000 + "x", "time,value"] + _rows(300)
        )
        assert elapsed < self.LIMIT_SECONDS
        assert result.status == "importable"

    def test_space_padded_key_value_line(self, write):
        elapsed, _ = self._timed(
            write, "kv.csv", ["a:x" + " " * 60000 + "y", "time,value"] + _rows(300)
        )
        assert elapsed < self.LIMIT_SECONDS

    def test_space_padded_bracket_in_a_header_field(self, write):
        elapsed, _ = self._timed(
            write, "bracket.csv", ["(" + " " * 30000 + "x,value"] + _rows(300)
        )
        assert elapsed < self.LIMIT_SECONDS


class TestReviewedEdgeCases:
    @pytest.mark.parametrize(
        "head, unit",
        [
            (["Time [S],acc [g]"], "s"),
            (["Time (uS),Acc (g)"], "us"),
            (["time,acc", "MS,g"], "ms"),
        ],
    )
    def test_time_unit_in_another_letter_case_stays_ambiguous(self, write, head, unit):
        result = inspect_signal_file(
            str(write("case.csv", corpus.lines_to_bytes(head + _rows(400))))
        )
        time_column = column(result, 0)
        assert time_column.role_candidate == "time"
        assert time_column.time_stats.time_unit == unit
        assert time_column.time_stats.time_unit_status == "ambiguous"
        # The token is kept as written, and no rate rests on the folded unit.
        assert time_column.unit_token.file_text.lower() == unit
        assert time_column.unit_token.file_text != unit
        assert not [c for c in result.candidates if c.status == "derived"]
        assert any(
            "time unit of columns 0" in item for item in result.required_confirmations
        )

    def test_time_unit_spelled_exactly_is_detected(self, write):
        data = corpus.lines_to_bytes(["time [s],acc [g]"] + _rows(400))
        result = inspect_signal_file(str(write("exact.csv", data)))
        assert column(result, 0).time_stats.time_unit_status == "detected"
        assert [c.status for c in result.candidates if c.column == 0] == ["derived"]
        assert not any("time unit" in item for item in result.required_confirmations)

    def test_long_preamble_does_not_shrink_a_complete_file(self, write):
        preamble = [f"note {index}: plain text" for index in range(60)]
        data = corpus.lines_to_bytes(preamble + ["time,value"] + _rows(400))
        result = inspect_signal_file(str(write("long.csv", data)))
        assert result.file.scan.complete
        assert result.status == "importable"
        assert "not_a_waveform" not in codes(result)

    @pytest.mark.parametrize(
        "rows, status", [(255, "not_importable"), (256, "importable")]
    )
    def test_waveform_minimum_boundary_on_a_fully_scanned_file(
        self, write, rows, status
    ):
        result = inspect_signal_file(
            str(write("edge.csv", corpus.lines_to_bytes(_rows(rows))))
        )
        assert result.status == status

    def test_escape_heavy_recognized_preamble_stays_within_the_budget(self, write):
        preamble = ["fs: " + "\\" * 300 for _ in range(40)]
        data = corpus.lines_to_bytes(preamble + ["time,value"] + _rows(400))
        result = inspect_signal_file(str(write("heavy.csv", data)))
        assert len(dumped(result).encode("utf-8")) <= OUTPUT_BUDGET_BYTES
        kept = {pair.line for pair in result.preamble.key_values}
        assert result.preamble.omitted_lines == 40 - len(kept) > 0
        assert all(c.line in kept for c in result.candidates if c.status == "detected")

    @pytest.mark.parametrize(
        "line, sep",
        [("Sampling rate: 25600", " "), ("Sampling rate:\t25600", "\t")],
    )
    def test_colon_key_in_whitespace_and_tab_exports(self, write, line, sep):
        header = sep.join(["time", "value"])
        data = corpus.lines_to_bytes([line, header] + _rows(400, sep=sep))
        result = inspect_signal_file(str(write("ws.txt", data)))
        assert [kv.key.file_text for kv in result.preamble.key_values] == [
            "Sampling rate"
        ]
        stated = [c for c in result.candidates if c.status == "detected"]
        assert [(c.field, c.normalized) for c in stated] == [
            ("sampling_rate_hz", "25600")
        ]

    @pytest.mark.parametrize(
        "line, sep, field, normalized",
        [
            (
                "Sample rate\t25,600",
                "\t",
                "sampling_rate_hz",
                None,
            ),  # comma is no decimal here
            ("Sampling rate: 25.6 kHz", "\t", "sampling_rate_hz", None),
            ("Speed: 30 Hz", "\t", "rpm", None),
            ("Sampling rate: 25600 Hz", "\t", "sampling_rate_hz", "25600"),
            ("Sensitivity: 100 mV/g", "\t", "sensitivity", "100"),
        ],
    )
    def test_preamble_numbers_follow_the_decimal_mark_and_unit(
        self, write, line, sep, field, normalized
    ):
        header = sep.join(["time", "value"])
        data = corpus.lines_to_bytes([line, header] + _rows(400, sep=sep))
        result = inspect_signal_file(str(write("norm.txt", data)))
        stated = [c for c in result.candidates if c.status == "detected"]
        assert [(c.field, c.normalized) for c in stated] == [(field, normalized)]

    @pytest.mark.parametrize(
        "value",
        ["1e40", "25600.0000000000000000001"],
        ids=["beyond-1e30", "over-20-significant-digits"],
    )
    def test_stated_number_the_decimal_rules_refuse_is_shown_not_normalized(
        self, write, value
    ):
        data = corpus.lines_to_bytes([f"fs: {value}", "time,value"] + _rows(400))
        result = inspect_signal_file(str(write("refused.csv", data)))
        stated = [c for c in result.candidates if c.field == "sampling_rate_hz"]
        # Kept as written, never rounded into a value it does not state.
        assert [(c.status, c.value.file_text, c.normalized) for c in stated] == [
            ("detected", value, None)
        ]

    def test_decimal_comma_preamble_number(self, write):
        rows = [f"{r},5;{r},25" for r in range(400)]
        data = corpus.lines_to_bytes(["Delta_X;0,000039", "x;y"] + rows)
        result = inspect_signal_file(str(write("comma.csv", data)))
        stated = [c for c in result.candidates if c.status == "detected"]
        assert [(c.field, c.normalized) for c in stated] == [
            ("sample_interval", "0.000039")
        ]

    def test_missing_rate_is_stated_as_absent(self, write):
        result = inspect_signal_file(
            str(write("norate.csv", corpus.plain_columns(400, 2)))
        )
        rates = [c for c in result.candidates if c.field == "sampling_rate_hz"]
        assert [(c.status, c.normalized) for c in rates] == [("absent", None)]

    def test_profiled_reading_is_named(self, write):
        proposed = inspect_signal_file(str(write("p.csv", corpus.european_export())))
        assert proposed.columns_reading == proposed.dialect.proposal
        ambiguous = inspect_signal_file(str(write("a.csv", corpus.ambiguous_rows())))
        assert ambiguous.dialect.proposal is None
        assert ambiguous.columns_reading == ambiguous.dialect.candidates[0]

    def test_unit_issues_list_their_columns(self, write):
        data = corpus.lines_to_bytes(
            ["a;b;c;d", "[mg];rpm;g;[kg]"] + [f"{i},5;1;2;3" for i in range(400)]
        )
        result = inspect_signal_file(str(write("units2.csv", data)))
        by_code = {issue.code: issue.columns for issue in result.issues}
        assert by_code["unit_token_unrecognized"] == [0, 3]
        assert by_code["unit_token_preview_only"] == [1]

    def test_values_near_the_float_maximum_do_not_overflow(self, write):
        rows = [f"{1.6e308 if i % 2 else 1.7e308!r},{i}" for i in range(300)]
        result = inspect_signal_file(str(write("big.csv", corpus.lines_to_bytes(rows))))
        stats = column(result, 0).stats
        assert stats.n == 300 and stats.mean == pytest.approx(1.65e308)

    def test_empty_and_duplicate_labels(self, write):
        data = corpus.lines_to_bytes(["x,,x"] + [f"{i},1,2" for i in range(400)])
        result = inspect_signal_file(str(write("labels2.csv", data)))
        assert {"empty_label", "duplicate_label"} <= set(codes(result))
        assert [c.label.file_text if c.label else None for c in result.columns] == [
            "x",
            None,
            "x",
        ]

    def test_units_row_one_field_short_shows_no_unit_tokens(self, write):
        data = corpus.lines_to_bytes(
            ["a,b,c", "g,g"] + [f"{i},1,2" for i in range(400)]
        )
        result = inspect_signal_file(str(write("units3.csv", data)))
        mismatch = [i for i in result.issues if i.code == "field_count_mismatch"]
        assert [i.line for i in mismatch] == [2]
        assert all(c.unit_token is None for c in result.columns)


def test_waveform_minimum_is_the_spectral_tools_default_segment():
    """The not-a-waveform threshold is the PSD and STFT default segment."""
    import inspect as introspect

    from predictive_maintenance_mcp.signal_import.inspection import MIN_WAVEFORM_ROWS
    from predictive_maintenance_mcp.signal_processing import spectral

    for function in (spectral.compute_psd, spectral.compute_stft_spectrogram):
        default = introspect.signature(function).parameters["nperseg"].default
        assert default == MIN_WAVEFORM_ROWS


def test_module_has_no_writer():
    """Inspection opens files read-only through the path gate and has no
    write call of its own."""
    source = Path(inspection_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "write_text",
        "write_bytes",
        ".mkdir(",
        "os.remove",
        "unlink(",
        "open(",
    ):
        assert forbidden not in source.replace("open_import_file(", "")
