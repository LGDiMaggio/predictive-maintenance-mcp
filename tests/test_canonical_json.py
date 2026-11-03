"""Tests for ``src/canonical_json.py`` (Signal Import Engine plan, U2).

The canonical JSON form moved out of the asset ledger store into a leaf
module the signal import shares. What must hold:

- the store re-exports the very same functions, so every ledger byte and
  every existing import is unchanged (the snapshot golden and the import
  compatibility golden stay green);
- the module is a leaf: standard library only;
- on the ASCII-only values an import interpretation consists of (strings
  without control characters, integers below 2**53, booleans, null, lists and
  objects), the output equals RFC 8785, the JSON Canonicalization Scheme,
  checked against a reference written here from the RFC.
"""

import ast
import sys
from pathlib import Path

import pytest

from predictive_maintenance_mcp import asset_ledger
from predictive_maintenance_mcp import canonical_json as leaf
from predictive_maintenance_mcp.asset_ledger import store

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _jcs_string(text: str) -> str:
    """RFC 8785 section 3.2.2.2: escape the quote, the backslash and
    U+0000 to U+001F (the five short forms, else lowercase ``\\u00XX``)."""
    short = {"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    out = []
    for char in text:
        if char in ('"', "\\"):
            out.append("\\" + char)
        elif char in short:
            out.append(short[char])
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def jcs(value: object) -> str:
    """RFC 8785 for null, booleans, integers below 2**53, strings, lists and
    objects (members sorted by the UTF-16 code units of their names)."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        assert abs(value) < 2**53
        return str(value)
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, list):
        return "[" + ",".join(jcs(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return "{" + ",".join(f"{_jcs_string(k)}:{jcs(value[k])}" for k in keys) + "}"
    raise TypeError(type(value).__name__)


#: Representative canonical interpretations (ASCII only, like the real ones).
REPRESENTATIVE = [
    {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": "utf-8",
            "delimiter": ";",
            "decimal": ",",
            "header_line": 7,
            "units_line": 8,
        },
        "data_lines": {"first": 9, "last": None},
        "time_base": {
            "source": "time_column",
            "sampling_rate_hz": "25600",
            "time_column": 0,
        },
        "time_columns": [{"column": 0, "time_unit": "s"}],
        "defaults": {
            "asset_id": "P-101",
            "acquired_at": "2026-08-20T13:42:00+02:00",
            "rpm": "1782",
            "load": "-0.5",
            "operating_state": "loaded, 75 %",
        },
        "channels": [
            {
                "column": 3,
                "role": "vibration",
                "quantity": "acceleration",
                "unit": "g",
                "scaling": {
                    "source_unit": "mV",
                    "sensitivity": {
                        "value": "100",
                        "unit": "mV/g",
                        "source": "calibration_certificate",
                    },
                    "dc_policy": "keep",
                },
                "binding": {
                    "measurement_point_id": "motor_de_h",
                    "direction": "horizontal",
                    "sensor_id": 'ACC "01" \\ rack',
                    "asset_id": None,
                },
            }
        ],
    },
    {"a": [], "B": {}, "_": [True, False, None, 0, -1, 2**53 - 1], "\t\n": "x"},
    {"": "", "aa": "a", "a": "aa", "b": [{"z": 1, "y": [2, {"x": "tab\there"}]}]},
]


class TestCanonicalJsonLeaf:
    def test_store_reexports_the_same_functions(self):
        assert store.canonical_json is leaf.canonical_json
        assert store.content_hash is leaf.content_hash
        assert asset_ledger.canonical_json is leaf.canonical_json
        assert asset_ledger.content_hash is leaf.content_hash

    def test_module_imports_only_the_standard_library(self):
        tree = ast.parse((SRC_DIR / "canonical_json.py").read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a leaf module has no relative imports"
                assert node.module.split(".")[0] in sys.stdlib_module_names

    @pytest.mark.parametrize("value", REPRESENTATIVE)
    def test_equals_rfc_8785_on_ascii_values(self, value):
        assert leaf.canonical_json(value) == jcs(value)

    def test_reference_is_not_vacuous(self):
        """The reference sorts and escapes; the comparison could fail."""
        assert jcs({"b": 1, "a": "\n"}) == '{"a":"\\n","b":1}'
        assert jcs({"b": 1, "a": "\n"}) != '{"b":1,"a":"\\n"}'

    def test_non_ascii_is_escaped_unlike_rfc_8785(self):
        """The declared deviation: non-ASCII text is escaped as \\uXXXX."""
        value = {"label": "Beschleunigung " + chr(0xB5) + "m" + chr(0x2028)}
        text = leaf.canonical_json(value)
        assert text.isascii()
        assert text != jcs(value)

    def test_content_hash_of_canonical_json(self):
        import hashlib

        value = REPRESENTATIVE[0]
        expected = hashlib.sha256(leaf.canonical_json(value).encode("ascii"))
        assert leaf.content_hash(value) == expected.hexdigest()
