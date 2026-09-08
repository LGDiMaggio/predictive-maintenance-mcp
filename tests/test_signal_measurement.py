"""Tests for signal_acquisition.measurement (U1): the companion ``"measurement"``
contract and the identity of a measurement.

The contract is a leaf module (stdlib plus path_safety only) consumed by the
signal repository at load time and, later, by the asset ledger. These tests
pin, from the plan's U1 scenarios:

- the ledger id grammar built on ``path_safety.validate_name_component``
  (reserved Windows device names refused on every OS, no trailing dot);
- the closed direction vocabulary and its form-only aliases (never x to
  horizontal);
- rpm/load typing and the free-text limits;
- ISO 8601 timestamps (naive accepted but flagged, aware normalized to UTC,
  suspect epoch/future, unparsable refused);
- the pure validator: ONE accumulated ValueError naming the companion file;
- ``measurement_id`` from the file bytes plus the channel index;
- unit families as a partition of ``VALID_SIGNAL_UNITS``;
- repository integration: the block on ``StoredSignalInfo``, refusal BEFORE
  insertion (batches stay all-or-nothing), ``companion_warning`` instead of
  refusal for unusable companions;
- tool-level visibility (``get_signal_info`` / ``list_signals``);
- the per-test ledger sandbox and packaging parity.
"""

import ast
import hashlib
import json
import os
import shutil
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver import MCPServer

from conftest import write_raw_file
from predictive_maintenance_mcp.signal_acquisition import measurement as m
from predictive_maintenance_mcp.signal_acquisition.repository import (
    VALID_SIGNAL_UNITS,
    SignalRepository,
    get_repository,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

FULL_MEASUREMENT = {
    "asset_id": "P-101",
    "measurement_point_id": "motor_de_h",
    "acquired_at": "2026-08-20T13:42:00+02:00",
    "rpm": 1482,
    "load": 75,
    "operating_state": "loaded",
    "sensor_id": "STWIN_BOX_001",
    "direction": "horizontal",
    "declared_by": "adapter:stwinbox",
}

MINIMAL_MEASUREMENT = {
    "asset_id": "P-101",
    "measurement_point_id": "motor_de_h",
    "acquired_at": "2026-08-20T13:42:00+02:00",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Write a deterministic single-column CSV signal and return the values."""
    values = np.random.default_rng(seed).standard_normal(n)
    pd.DataFrame(values).to_csv(path, index=False, header=False)
    return values


def write_companion(signal_path: Path, payload) -> Path:
    """Write ``<stem>_metadata.json`` next to ``signal_path``.

    ``payload`` is a dict (serialized as JSON) or a raw string (written
    verbatim, for malformed companions).
    """
    companion = signal_path.parent / f"{signal_path.stem}_metadata.json"
    if isinstance(payload, str):
        companion.write_text(payload, encoding="utf-8")
    else:
        companion.write_text(json.dumps(payload), encoding="utf-8")
    return companion


def companion_with(measurement=None, **top_level) -> dict:
    """A companion declaring rate and unit, plus an optional measurement."""
    payload: dict = {"sampling_rate": 10000, "signal_unit": "g", **top_level}
    if measurement is not None:
        payload["measurement"] = measurement
    return payload


def validate(measurement: dict, name: str = "sig_metadata.json", **kwargs) -> dict:
    return m.validate_measurement_declaration(
        {"measurement": measurement}, name, **kwargs
    )


@pytest.fixture
def repo() -> SignalRepository:
    return SignalRepository()


# ---------------------------------------------------------------------------
# Vocabularies and field tables
# ---------------------------------------------------------------------------


class TestVocabularies:
    def test_direction_vocabulary_is_closed(self):
        assert m.VALID_DIRECTIONS == ("horizontal", "vertical", "axial", "x", "y", "z")

    def test_aliases_are_form_only(self):
        """h/v/a spell out the word; x/y/z never map to horizontal/vertical."""
        assert m.DIRECTION_ALIASES == {"h": "horizontal", "v": "vertical", "a": "axial"}
        assert not ({"x", "y", "z"} & set(m.DIRECTION_ALIASES))
        assert not ({"x", "y", "z"} & set(m.DIRECTION_ALIASES.values()))

    def test_required_and_optional_fields_partition_the_contract(self):
        """Optional-ness is encoded like RAW_PARAM_DEFAULTS: required fields are
        absent from the defaults mapping."""
        assert m.REQUIRED_MEASUREMENT_FIELDS == (
            "asset_id",
            "measurement_point_id",
            "acquired_at",
        )
        assert set(m.MEASUREMENT_FIELD_DEFAULTS) == {
            "rpm",
            "load",
            "operating_state",
            "sensor_id",
            "direction",
            "declared_by",
        }
        assert all(v is None for v in m.MEASUREMENT_FIELD_DEFAULTS.values())
        assert not set(m.REQUIRED_MEASUREMENT_FIELDS) & set(
            m.MEASUREMENT_FIELD_DEFAULTS
        )
        assert m.MEASUREMENT_FIELDS == (
            *m.REQUIRED_MEASUREMENT_FIELDS,
            *m.MEASUREMENT_FIELD_DEFAULTS,
        )

    def test_free_text_fields_are_optional_fields(self):
        assert m.FREE_TEXT_FIELDS == ("operating_state", "sensor_id", "declared_by")
        assert set(m.FREE_TEXT_FIELDS) <= set(m.MEASUREMENT_FIELD_DEFAULTS)
        assert m.MAX_FREE_TEXT_CHARS == 200

    def test_identity_keys_extend_declaration_keys(self):
        assert m.MEASUREMENT_IDENTITY_KEYS == (
            "asset_id",
            "measurement_point_id",
            "acquired_at",
            "timezone_declared",
            "timestamp_suspect",
            "rpm",
            "load",
            "operating_state",
            "sensor_id",
            "direction",
            "declared_by",
            "measurement_id",
            "channel_index",
        )
        assert m.MEASUREMENT_DECLARATION_KEYS == m.MEASUREMENT_IDENTITY_KEYS[:-2]

    def test_unit_families_partition_valid_signal_units(self):
        """Every unit the repository accepts belongs to EXACTLY one family, and
        the families contain nothing else (imports both modules on purpose:
        the leaf module must not import the repository)."""
        seen: list[str] = []
        for units in m.UNIT_FAMILIES.values():
            seen.extend(units)
        assert sorted(seen) == sorted(VALID_SIGNAL_UNITS)
        assert len(seen) == len(set(seen))
        assert m.UNIT_FAMILIES == {
            "acceleration": ("g", "m/s2"),
            "velocity": ("mm/s", "m/s"),
        }

    def test_unit_family_lookup(self):
        assert m.unit_family("g") == "acceleration"
        assert m.unit_family("m/s2") == "acceleration"
        assert m.unit_family("mm/s") == "velocity"
        assert m.unit_family("m/s") == "velocity"
        assert m.unit_family(None) is None
        assert m.unit_family("furlong/fortnight") is None


class TestLeafModule:
    def test_imports_only_stdlib_and_path_safety(self):
        """The contract is a leaf: no repository, no models, no numpy."""
        tree = ast.parse(
            (SRC_DIR / "signal_acquisition" / "measurement.py").read_text("utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert (
                        alias.name.split(".")[0] in sys.stdlib_module_names
                    ), alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert (
                        node.level == 2 and node.module == "path_safety"
                    ), f"relative import of {node.module!r} at level {node.level}"
                else:
                    assert (
                        node.module.split(".")[0] in sys.stdlib_module_names
                    ), node.module

    def test_public_names_reexported_from_package(self):
        from predictive_maintenance_mcp import signal_acquisition as pkg

        for name in (
            "MEASUREMENT_KEY",
            "VALID_DIRECTIONS",
            "REQUIRED_MEASUREMENT_FIELDS",
            "MEASUREMENT_FIELD_DEFAULTS",
            "UNIT_FAMILIES",
            "unit_family",
            "validate_ledger_id",
            "validate_measurement_declaration",
            "compute_measurement_id",
            "build_measurement_identity",
        ):
            assert getattr(pkg, name) is getattr(m, name), name

    def test_asset_ledger_package_is_importable_with_iso_header(self):
        import predictive_maintenance_mcp.asset_ledger as ledger_pkg

        assert "ISO 13374" in (ledger_pkg.__doc__ or "")


# ---------------------------------------------------------------------------
# Ledger id grammar
# ---------------------------------------------------------------------------


class TestLedgerIdGrammar:
    @pytest.mark.parametrize(
        "name", ["P-101", "motor_de_h", "A1.b2", "x", "0asset", "P.101-a_b", "a" * 100]
    )
    def test_accepted_ids_are_returned_unchanged(self, name):
        assert m.validate_ledger_id(name, kind="asset_id") == name

    @pytest.mark.parametrize(
        "name",
        [
            "P 101",  # space
            "P/101",  # separator
            "..",  # reserved path name
            "P-101.",  # trailing dot
            "NUL",  # Windows device
            "nul",  # case-insensitive on Windows
            "com1.old",  # device name with an extension
            "LPT9",
            "CON.jsonl",
            "-x",  # must start alphanumeric
            "_x",
            ".hidden",
            "",  # empty
            None,  # not a string
            101,  # JSON number
            "a" * 101,  # over the length cap
        ],
    )
    def test_rejected_ids_name_the_grammar(self, name):
        with pytest.raises(ValueError) as exc_info:
            m.validate_ledger_id(name, kind="measurement_point_id")
        message = str(exc_info.value)
        assert "measurement_point_id" in message
        assert "letter or digit" in message  # the grammar is spelled out
        assert "NUL" in message  # reserved names listed

    def test_reserved_names_refused_on_every_os(self):
        """A ledger can be copied to Windows, so the refusal is unconditional."""
        for name in ("CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9"):
            with pytest.raises(ValueError, match="reserved"):
                m.validate_ledger_id(name, kind="asset_id")
            with pytest.raises(ValueError, match="reserved"):
                m.validate_ledger_id(name.lower() + ".jsonl", kind="asset_id")
        assert "COM0" not in m.WIN32_RESERVED_NAMES
        assert "COM10" not in m.WIN32_RESERVED_NAMES

    def test_builds_on_validate_name_component(self, monkeypatch):
        """The base guard is called, not re-implemented: a name it rejects is
        rejected here too, with the ledger grammar appended."""
        calls: list[str] = []
        original = m.validate_name_component

        def spy(name, *, kind="name"):
            calls.append(name)
            return original(name, kind=kind)

        monkeypatch.setattr(m, "validate_name_component", spy)
        assert m.validate_ledger_id("P-101", kind="asset_id") == "P-101"
        with pytest.raises(ValueError):
            m.validate_ledger_id("../evil", kind="asset_id")
        assert calls == ["P-101", "../evil"]

    def test_reserved_name_is_refused_before_any_file_is_opened(
        self, tmp_path, monkeypatch
    ):
        """The grammar runs before any path is built: no open() call, no
        directory change. (On older Windows open('NUL.jsonl', 'ab') succeeds
        and the bytes vanish; this Windows 11 build creates a real file
        instead, so the hazard itself is not asserted, only that the
        refusal never reaches the filesystem on any OS.)"""
        opened: list[str] = []
        real_open = open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", spy)
        before = sorted(p.name for p in tmp_path.iterdir())
        for name in ("NUL", "NUL.jsonl", "com1.old"):
            with pytest.raises(ValueError, match="reserved"):
                m.validate_ledger_id(name, kind="asset_id")
        assert opened == []
        assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# Direction vocabulary
# ---------------------------------------------------------------------------


class TestDirectionNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("H", "horizontal"),
            ("h", "horizontal"),
            ("V", "vertical"),
            ("a", "axial"),
            ("AXIAL", "axial"),
            ("Horizontal", "horizontal"),
            ("x", "x"),
            ("X", "x"),
            ("Y", "y"),
            ("z", "z"),
        ],
    )
    def test_form_aliases_and_case_folding(self, raw, expected):
        assert m.normalize_direction(raw) == expected

    def test_x_and_horizontal_are_distinct_declarations(self):
        assert m.normalize_direction("x") != m.normalize_direction("horizontal")

    @pytest.mark.parametrize("raw", ["sideways", "", "hor", 5, None, ["x"]])
    def test_outside_vocabulary_lists_the_vocabulary(self, raw):
        with pytest.raises(ValueError) as exc_info:
            m.normalize_direction(raw)
        message = str(exc_info.value)
        for word in m.VALID_DIRECTIONS:
            assert word in message
        assert "direction" in message


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


class TestAcquiredAt:
    NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

    def test_naive_is_accepted_and_flagged(self):
        normalized, declared, suspect = m.normalize_acquired_at(
            "2026-08-20T13:42:00", now=self.NOW
        )
        assert normalized == "2026-08-20T13:42:00+00:00"
        assert declared is False
        assert suspect is False

    def test_zulu_suffix_is_a_declared_timezone(self):
        normalized, declared, suspect = m.normalize_acquired_at(
            "2026-08-20T13:42:00Z", now=self.NOW
        )
        assert normalized == "2026-08-20T13:42:00+00:00"
        assert declared is True
        assert suspect is False

    def test_offset_is_normalized_to_utc(self):
        normalized, declared, _ = m.normalize_acquired_at(
            "2026-08-20T13:42:00+02:00", now=self.NOW
        )
        assert normalized == "2026-08-20T11:42:00+00:00"
        assert declared is True

    def test_normalized_strings_order_chronologically(self):
        """Different declared offsets sort by instant once normalized."""
        a, _, _ = m.normalize_acquired_at("2026-08-20T13:42:00+02:00", now=self.NOW)
        b, _, _ = m.normalize_acquired_at("2026-08-20T12:00:00Z", now=self.NOW)
        assert a < b  # 11:42 UTC precedes 12:00 UTC

    @pytest.mark.parametrize(
        "raw", ["20/08/2026 13:42", "yesterday", "", 1724160120, None, 13.42]
    )
    def test_unparsable_is_refused(self, raw):
        with pytest.raises(ValueError, match="ISO 8601"):
            m.normalize_acquired_at(raw, now=self.NOW)

    @pytest.mark.parametrize(
        "raw", ["1970-01-01T00:00:00Z", "1970-06-15T08:00:00", "1969-12-31T23:59:59Z"]
    )
    def test_epoch_year_is_suspect(self, raw):
        _, _, suspect = m.normalize_acquired_at(raw, now=self.NOW)
        assert suspect is True

    def test_first_instant_after_epoch_year_is_not_suspect(self):
        _, _, suspect = m.normalize_acquired_at("1971-01-01T00:00:00Z", now=self.NOW)
        assert suspect is False

    def test_future_beyond_one_day_is_suspect(self):
        _, _, within = m.normalize_acquired_at("2026-09-09T11:00:00Z", now=self.NOW)
        _, _, beyond = m.normalize_acquired_at("2026-09-09T13:00:00Z", now=self.NOW)
        assert within is False
        assert beyond is True

    def test_wall_clock_default_flags_far_future(self):
        _, _, suspect = m.normalize_acquired_at("2099-01-01T00:00:00Z")
        assert suspect is True


# ---------------------------------------------------------------------------
# The pure validator
# ---------------------------------------------------------------------------


class TestValidateMeasurementDeclaration:
    def test_happy_path_normalizes_every_field(self):
        identity = validate(FULL_MEASUREMENT)
        assert tuple(identity) == m.MEASUREMENT_DECLARATION_KEYS
        assert identity == {
            "asset_id": "P-101",
            "measurement_point_id": "motor_de_h",
            "acquired_at": "2026-08-20T11:42:00+00:00",
            "timezone_declared": True,
            "timestamp_suspect": False,
            "rpm": 1482.0,
            "load": 75.0,
            "operating_state": "loaded",
            "sensor_id": "STWIN_BOX_001",
            "direction": "horizontal",
            "declared_by": "adapter:stwinbox",
        }

    def test_minimal_declaration_leaves_optionals_none(self):
        identity = validate(MINIMAL_MEASUREMENT)
        for field in m.MEASUREMENT_FIELD_DEFAULTS:
            assert identity[field] is None
        assert identity["timezone_declared"] is True

    def test_missing_required_fields_are_one_error_naming_the_companion(self):
        with pytest.raises(ValueError) as exc_info:
            validate({"asset_id": "P-101"}, name="pump_metadata.json")
        message = str(exc_info.value)
        assert "measurement_point_id" in message
        assert "acquired_at" in message
        assert "pump_metadata.json" in message
        assert '"measurement"' in message  # the remedy names the object
        assert message.count("Invalid") == 1  # one accumulated refusal

    def test_all_problems_accumulate_into_one_message(self):
        with pytest.raises(ValueError) as exc_info:
            validate(
                {
                    "asset_id": "P 101",
                    "measurement_point_id": "motor_de_h",
                    "direction": "sideways",
                    "rpm": "fast",
                }
            )
        message = str(exc_info.value)
        assert "acquired_at" in message  # missing
        assert "asset_id" in message  # grammar
        assert "sideways" in message  # vocabulary
        assert "rpm" in message  # type

    def test_null_required_field_counts_as_missing(self):
        with pytest.raises(ValueError, match="acquired_at"):
            validate({**MINIMAL_MEASUREMENT, "acquired_at": None})

    def test_unknown_field_is_refused_with_the_accepted_list(self):
        """The object is a closed contract; free-form keys live at the top
        level of the companion, where they pass verbatim into source_metadata."""
        with pytest.raises(ValueError) as exc_info:
            validate({**MINIMAL_MEASUREMENT, "note": "checked by hand"})
        message = str(exc_info.value)
        assert "note" in message
        assert "top level" in message
        for field in m.MEASUREMENT_FIELDS:
            assert field in message

    @pytest.mark.parametrize("bad", ["P-101", ["P-101"], 7, None])
    def test_measurement_that_is_not_an_object_is_refused(self, bad):
        with pytest.raises(ValueError, match="JSON object"):
            validate(bad)

    def test_companion_without_object_is_refused(self):
        with pytest.raises(ValueError, match='"measurement"'):
            m.validate_measurement_declaration({"sampling_rate": 1}, "x_metadata.json")

    def test_message_names_the_file_not_its_directory(self, tmp_path):
        companion = tmp_path / "secret_dir" / "sig_metadata.json"
        with pytest.raises(ValueError) as exc_info:
            m.validate_measurement_declaration(
                {"measurement": {"asset_id": "P-101"}}, companion
            )
        message = str(exc_info.value)
        assert "sig_metadata.json" in message
        assert "secret_dir" not in message

    def test_timestamp_flags_flow_through(self):
        identity = validate(
            {**MINIMAL_MEASUREMENT, "acquired_at": "1970-01-01T00:00:00"}
        )
        assert identity["timezone_declared"] is False
        assert identity["timestamp_suspect"] is True

    def test_now_override_is_honored(self):
        far = datetime(2099, 6, 1, tzinfo=timezone.utc)
        identity = validate(
            {**MINIMAL_MEASUREMENT, "acquired_at": "2099-05-31T12:00:00Z"}, now=far
        )
        assert identity["timestamp_suspect"] is False


class TestNumericFields:
    @pytest.mark.parametrize("rpm", ["1480 rpm", "1480", -5, 0, True, float("nan"), []])
    def test_rpm_must_be_a_positive_number(self, rpm):
        with pytest.raises(ValueError) as exc_info:
            validate({**MINIMAL_MEASUREMENT, "rpm": rpm})
        message = str(exc_info.value)
        assert "rpm" in message
        assert "positive number" in message

    def test_rpm_absent_or_null_is_none(self):
        assert validate(MINIMAL_MEASUREMENT)["rpm"] is None
        assert validate({**MINIMAL_MEASUREMENT, "rpm": None})["rpm"] is None

    def test_rpm_is_normalized_to_float(self):
        assert validate({**MINIMAL_MEASUREMENT, "rpm": 1482})["rpm"] == 1482.0
        assert validate({**MINIMAL_MEASUREMENT, "rpm": 1482.5})["rpm"] == 1482.5

    @pytest.mark.parametrize("load", ["75%", True, float("inf"), {"kw": 5}])
    def test_load_must_be_a_number(self, load):
        with pytest.raises(ValueError, match="load"):
            validate({**MINIMAL_MEASUREMENT, "load": load})

    def test_load_accepts_any_finite_number(self):
        assert validate({**MINIMAL_MEASUREMENT, "load": 0})["load"] == 0.0
        assert validate({**MINIMAL_MEASUREMENT, "load": -12.5})["load"] == -12.5


class TestFreeText:
    @pytest.mark.parametrize("field", m.FREE_TEXT_FIELDS)
    def test_limit_is_named_when_exceeded(self, field):
        assert validate({**MINIMAL_MEASUREMENT, field: "x" * 200})[field] == "x" * 200
        with pytest.raises(ValueError) as exc_info:
            validate({**MINIMAL_MEASUREMENT, field: "x" * 201})
        message = str(exc_info.value)
        assert field in message
        assert "200" in message

    @pytest.mark.parametrize("field", m.FREE_TEXT_FIELDS)
    @pytest.mark.parametrize(
        "text",
        ["line\nbreak", "tab\there", "nul\x00", "para\u2028sep", "bom\ufeff"],
    )
    def test_control_characters_are_refused(self, field, text):
        with pytest.raises(ValueError) as exc_info:
            validate({**MINIMAL_MEASUREMENT, field: text})
        message = str(exc_info.value)
        assert field in message
        assert "control" in message

    @pytest.mark.parametrize("field", m.FREE_TEXT_FIELDS)
    @pytest.mark.parametrize("value", ["", "   ", 5, ["a"]])
    def test_empty_or_non_string_is_refused(self, field, value):
        with pytest.raises(ValueError, match=field):
            validate({**MINIMAL_MEASUREMENT, field: value})

    def test_unicode_letters_and_spaces_are_fine(self):
        text = "sensore lato accoppiamento, città"
        assert validate({**MINIMAL_MEASUREMENT, "sensor_id": text})["sensor_id"] == text


# ---------------------------------------------------------------------------
# Identity hash
# ---------------------------------------------------------------------------


class TestMeasurementId:
    def test_recipe_is_sha256_of_content_digest_and_channel(self, tmp_path):
        """Pinned so a later re-verification of a moved file can recompute it."""
        path = tmp_path / "sig.csv"
        path.write_bytes(b"1.0\n2.0\n3.0\n")
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = hashlib.sha256(f"{content}:0".encode("ascii")).hexdigest()[:16]
        assert m.compute_measurement_id(path, 0) == expected

    def test_is_sixteen_hex_and_stable(self, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        first = m.compute_measurement_id(path, 0)
        second = m.compute_measurement_id(str(path), 0)
        assert first == second
        assert len(first) == m.MEASUREMENT_ID_HEX_CHARS == 16
        assert int(first, 16) >= 0

    def test_copy_in_another_folder_has_same_id(self, tmp_path):
        path = tmp_path / "a" / "sig.csv"
        path.parent.mkdir()
        write_csv(path)
        copy = tmp_path / "b" / "renamed.csv"
        copy.parent.mkdir()
        shutil.copyfile(path, copy)
        assert m.compute_measurement_id(path, 0) == m.compute_measurement_id(copy, 0)

    def test_channel_index_distinguishes_channels(self, tmp_path):
        path = tmp_path / "multi.bin"
        write_raw_file(path, np.arange(20, dtype=np.float32))
        assert m.compute_measurement_id(path, 0) != m.compute_measurement_id(path, 1)

    def test_content_change_changes_id(self, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path, seed=1)
        before = m.compute_measurement_id(path, 0)
        write_csv(path, seed=2)
        assert m.compute_measurement_id(path, 0) != before

    @pytest.mark.parametrize("channel", [-1, "0", 0.0, True, None])
    def test_channel_index_must_be_a_non_negative_int(self, tmp_path, channel):
        path = tmp_path / "sig.csv"
        write_csv(path)
        with pytest.raises(ValueError, match="channel_index"):
            m.compute_measurement_id(path, channel)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            m.compute_measurement_id(tmp_path / "absent.csv", 0)


class TestBuildMeasurementIdentity:
    def test_composes_declaration_and_hash(self, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        identity = m.build_measurement_identity(
            {"measurement": FULL_MEASUREMENT},
            "sig_metadata.json",
            signal_path=path,
            channel_index=0,
        )
        assert tuple(identity) == m.MEASUREMENT_IDENTITY_KEYS
        assert identity["measurement_id"] == m.compute_measurement_id(path, 0)
        assert identity["channel_index"] == 0
        assert identity["asset_id"] == "P-101"
        assert "error" not in identity

    def test_validation_failure_precedes_hashing(self, tmp_path):
        """An invalid object is refused without touching the (absent) file."""
        with pytest.raises(ValueError, match="acquired_at"):
            m.build_measurement_identity(
                {"measurement": {"asset_id": "P-101"}},
                "sig_metadata.json",
                signal_path=tmp_path / "absent.csv",
                channel_index=0,
            )


# ---------------------------------------------------------------------------
# Repository integration
# ---------------------------------------------------------------------------


class TestRepositoryIntegration:
    def test_identity_block_is_stored_and_stable(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, companion_with(FULL_MEASUREMENT))

        info = repo.load_signal(str(path))
        block = info["measurement"]
        assert tuple(block) == m.MEASUREMENT_IDENTITY_KEYS
        assert block["asset_id"] == "P-101"
        assert block["measurement_point_id"] == "motor_de_h"
        assert block["acquired_at"] == "2026-08-20T11:42:00+00:00"
        assert block["direction"] == "horizontal"
        assert block["channel_index"] == 0
        assert len(block["measurement_id"]) == 16
        assert info["companion_warning"] is None
        assert info["sampling_rate"] == 10000  # the rest of the companion still honored
        assert info["signal_unit"] == "g"

        again = repo.load_signal(str(path), overwrite=True)
        assert again["measurement"]["measurement_id"] == block["measurement_id"]

    def test_block_agrees_with_verbatim_source_metadata(self, repo, tmp_path):
        """source_metadata keeps the object verbatim; the normalized block is
        the authoritative one (documented on the model field)."""
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, companion_with(FULL_MEASUREMENT))
        info = repo.load_signal(str(path))
        verbatim = info["source_metadata"]["measurement"]
        assert verbatim == FULL_MEASUREMENT
        assert info["measurement"]["asset_id"] == verbatim["asset_id"]
        assert (
            info["measurement"]["measurement_point_id"]
            == verbatim["measurement_point_id"]
        )
        # Normalization is visible in the block, never in the verbatim copy.
        assert verbatim["acquired_at"] == "2026-08-20T13:42:00+02:00"
        assert info["measurement"]["acquired_at"] == "2026-08-20T11:42:00+00:00"

    def test_get_signal_info_returns_a_copy_of_the_block(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, companion_with(FULL_MEASUREMENT))
        repo.load_signal(str(path), signal_id="s")
        info = repo.get_signal_info("s")
        info["measurement"]["asset_id"] = "tampered"
        assert repo.get_signal_info("s")["measurement"]["asset_id"] == "P-101"

    @pytest.mark.parametrize(
        "payload",
        [
            {"sampling_rate": 10000, "signal_unit": "g"},
            {
                "sampling_rate": 10000,
                "signal_unit": "g",
                "load": 75,
                "shaft_speed": 24.7,
            },
        ],
    )
    def test_companion_without_object_behaves_as_today(self, repo, tmp_path, payload):
        """Covers AE1 (no asset fields): free keys such as load or shaft_speed
        at the top level are never interpreted as a measurement."""
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, payload)
        info = repo.load_signal(str(path))
        assert info["measurement"] is None
        assert info["companion_warning"] is None
        assert info["source_metadata"] == payload
        assert info["sampling_rate"] == 10000
        assert info["signal_unit"] == "g"

    def test_file_without_companion_has_both_keys_none(self, repo, tmp_path):
        path = tmp_path / "bare.csv"
        write_csv(path)
        info = repo.load_signal(str(path))
        assert info["measurement"] is None
        assert info["companion_warning"] is None

    def test_missing_required_fields_refuse_before_insertion(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, companion_with({"asset_id": "P-101"}))
        with pytest.raises(ValueError) as exc_info:
            repo.load_signal(str(path))
        message = str(exc_info.value)
        assert "measurement_point_id" in message
        assert "acquired_at" in message
        assert "sig_metadata.json" in message
        assert repo.signal_count == 0

    def test_invalid_object_refuses_even_when_the_rest_is_fine(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(
            path, companion_with({**MINIMAL_MEASUREMENT, "direction": "sideways"})
        )
        with pytest.raises(ValueError, match="sideways"):
            repo.load_signal(str(path))
        assert repo.signal_count == 0

    def test_malformed_companion_is_a_warning_not_a_refusal(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, '{"sampling_rate": 10000, "measurement": {')
        info = repo.load_signal(str(path))
        assert info["measurement"] is None
        assert "sig_metadata.json" in info["companion_warning"]
        assert "not valid JSON" in info["companion_warning"]
        assert info["sampling_rate"] is None  # exactly as a companion-less load
        assert info["source_metadata"] == {}
        assert repo.signal_count == 1

    def test_non_object_companion_is_a_warning_not_a_refusal(self, repo, tmp_path):
        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, [MINIMAL_MEASUREMENT])
        info = repo.load_signal(str(path))
        assert info["measurement"] is None
        assert "sig_metadata.json" in info["companion_warning"]
        assert "not a JSON object" in info["companion_warning"]
        assert repo.signal_count == 1

    def test_batch_with_one_invalid_companion_loads_nothing(
        self, repo, sandbox_data_dir
    ):
        names = ["first.csv", "second.csv", "third.csv"]
        for i, name in enumerate(names):
            path = sandbox_data_dir / name
            write_csv(path, seed=i)
            declaration = dict(MINIMAL_MEASUREMENT, measurement_point_id=f"pt_{i}")
            if name == "second.csv":
                declaration.pop("acquired_at")
            write_companion(path, companion_with(declaration))

        with pytest.raises(ValueError) as exc_info:
            repo.load_signals(names)
        assert "second_metadata.json" in str(exc_info.value)
        assert repo.signal_count == 0

    def test_valid_batch_records_an_identity_per_file(self, repo, sandbox_data_dir):
        names = ["first.csv", "second.csv"]
        for i, name in enumerate(names):
            path = sandbox_data_dir / name
            write_csv(path, seed=i)
            write_companion(
                path,
                companion_with(
                    dict(MINIMAL_MEASUREMENT, measurement_point_id=f"pt_{i}")
                ),
            )
        infos = repo.load_signals(names)
        ids = {info["measurement"]["measurement_id"] for info in infos}
        assert len(ids) == 2
        assert [info["measurement"]["measurement_point_id"] for info in infos] == [
            "pt_0",
            "pt_1",
        ]

    def test_raw_channels_get_distinct_ids(self, repo, tmp_path):
        """A two-channel raw file loaded with channel_index 0 and 1 yields two
        measurement ids; the channel index is part of the identity."""
        path = tmp_path / "multi.bin"
        interleaved = np.empty(200, dtype=np.float32)
        interleaved[0::2] = 5.0
        interleaved[1::2] = 9.0
        write_raw_file(path, interleaved)
        write_companion(
            path,
            {
                "sampling_rate": 1000,
                "sample_format": "float32",
                "n_channels": 2,
                "measurement": MINIMAL_MEASUREMENT,
            },
        )
        i0 = repo.load_signal(str(path), channel_index=0)
        i1 = repo.load_signal(str(path), channel_index=1)
        assert i0["signal_id"] == "multi_ch0" and i1["signal_id"] == "multi_ch1"
        assert i0["measurement"]["channel_index"] == 0
        assert i1["measurement"]["channel_index"] == 1
        assert (
            i0["measurement"]["measurement_id"] != i1["measurement"]["measurement_id"]
        )
        assert i0["measurement"]["measurement_id"] == m.compute_measurement_id(path, 0)

    def test_copied_file_keeps_its_measurement_id(self, repo, tmp_path):
        original = tmp_path / "site_a" / "sig.csv"
        original.parent.mkdir()
        write_csv(original)
        write_companion(original, companion_with(FULL_MEASUREMENT))
        copy_dir = tmp_path / "site_b"
        shutil.copytree(original.parent, copy_dir)

        a = repo.load_signal(str(original), signal_id="a")
        b = repo.load_signal(str(copy_dir / "sig.csv"), signal_id="b")
        assert a["measurement"]["measurement_id"] == b["measurement"]["measurement_id"]
        assert a["filepath"] != b["filepath"]


# ---------------------------------------------------------------------------
# Tool-level visibility
# ---------------------------------------------------------------------------


class TestToolVisibility:
    @pytest.fixture
    def tools(self):
        from predictive_maintenance_mcp.mcp_tools import acquisition_tools

        server = MCPServer("test-measurement")
        acquisition_tools.register(server)
        return {t.name: t.fn for t in server._tool_manager._tools.values()}

    @pytest.mark.asyncio
    async def test_get_signal_info_and_list_signals_show_the_block(
        self, tools, sandbox_data_dir
    ):
        ctx = AsyncMock()
        with_id = sandbox_data_dir / "with_id.csv"
        write_csv(with_id, seed=1)
        write_companion(with_id, companion_with(FULL_MEASUREMENT))
        plain = sandbox_data_dir / "plain.csv"
        write_csv(plain, seed=2)
        write_companion(plain, companion_with())

        repo = get_repository()
        repo.clear_all()
        try:
            await tools["load_signal"](ctx=ctx, filepath="with_id.csv")
            await tools["load_signal"](ctx=ctx, filepath="plain.csv")

            info = await tools["get_signal_info"](ctx=ctx, signal_id="with_id")
            assert info.measurement["asset_id"] == "P-101"
            assert info.measurement["measurement_point_id"] == "motor_de_h"
            assert "error" not in info.measurement
            assert info.companion_warning is None
            assert "error" not in info.model_dump()

            bare = await tools["get_signal_info"](ctx=ctx, signal_id="plain")
            assert bare.measurement is None
            assert bare.companion_warning is None

            listed = await tools["list_signals"](ctx=ctx, scope="memory")
            by_id = {s["signal_id"]: s for s in listed["signals"]}
            assert by_id["with_id"]["measurement"]["asset_id"] == "P-101"
            assert by_id["plain"]["measurement"] is None
        finally:
            repo.clear_all()

    @pytest.mark.asyncio
    async def test_load_signal_relays_companion_warning(self, tools, sandbox_data_dir):
        ctx = AsyncMock()
        broken = sandbox_data_dir / "broken.csv"
        write_csv(broken)
        write_companion(broken, "not json at all")
        repo = get_repository()
        repo.clear_all()
        try:
            info = await tools["load_signal"](
                ctx=ctx, filepath="broken.csv", sampling_rate=10000.0
            )
            assert "broken_metadata.json" in info.companion_warning
            assert info.measurement is None
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Sandbox: no test may write into the checkout's data/ledger
# ---------------------------------------------------------------------------


def _listing(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*"))


class TestLedgerSandbox:
    def test_env_points_every_test_at_its_own_tmp_path(self, tmp_path, ledger_dir):
        from predictive_maintenance_mcp.config import get_ledger_dir

        assert Path(os.environ["PMM_LEDGER_DIR"]) == ledger_dir
        assert ledger_dir.is_relative_to(tmp_path)
        assert get_ledger_dir() == ledger_dir
        assert not ledger_dir.exists()  # nothing creates it up front

    def test_loading_an_identity_never_touches_the_checkout_ledger(
        self, repo, tmp_path
    ):
        from predictive_maintenance_mcp.config import get_ledger_dir

        checkout_ledger = REPO_ROOT / "data" / "ledger"
        before = _listing(checkout_ledger)

        path = tmp_path / "sig.csv"
        write_csv(path)
        write_companion(path, companion_with(FULL_MEASUREMENT))
        info = repo.load_signal(str(path))
        assert info["measurement"]["asset_id"] == "P-101"

        assert _listing(checkout_ledger) == before
        assert get_ledger_dir().is_relative_to(tmp_path)
        # Nothing is written next to the user's data either.
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "sig.csv",
            "sig_metadata.json",
        ]


# ---------------------------------------------------------------------------
# Packaging parity
# ---------------------------------------------------------------------------


class TestPackaging:
    @pytest.fixture(scope="class")
    def setuptools_config(self) -> dict:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        return data["tool"]["setuptools"]

    def test_asset_ledger_package_is_declared(self, setuptools_config):
        assert (
            "predictive_maintenance_mcp.asset_ledger" in setuptools_config["packages"]
        )
        assert (
            setuptools_config["package-dir"]["predictive_maintenance_mcp.asset_ledger"]
            == "src/asset_ledger"
        )

    def test_every_src_subpackage_is_declared_both_ways(self, setuptools_config):
        on_disk = {p.parent.name for p in SRC_DIR.glob("*/__init__.py")}
        declared = {
            name.split(".", 1)[1]
            for name in setuptools_config["packages"]
            if name.startswith("predictive_maintenance_mcp.")
        }
        assert declared == on_disk
        for sub in on_disk:
            assert (
                setuptools_config["package-dir"][f"predictive_maintenance_mcp.{sub}"]
                == f"src/{sub}"
            )

    @pytest.mark.parametrize("ignore_file", [".gitignore", ".dockerignore"])
    def test_ledger_directory_is_ignored(self, ignore_file):
        lines = {
            line.strip()
            for line in (REPO_ROOT / ignore_file).read_text("utf-8").splitlines()
        }
        assert "data/ledger/" in lines

    def test_manifest_prunes_the_ledger(self):
        lines = {
            line.strip()
            for line in (REPO_ROOT / "MANIFEST.in").read_text("utf-8").splitlines()
        }
        assert "prune data/ledger" in lines
