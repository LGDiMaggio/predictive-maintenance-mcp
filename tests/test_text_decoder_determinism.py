"""Determinism of the strict text decoder (Signal Import Engine plan, U5).

Ledger replay must reproduce a recorded samples digest bit for bit, on every
operating system and dependency stack, so the decoder's output is pinned here
before it is optimized. Each case is a synthetic export built from integer
arithmetic and Python's correctly rounded ``repr``/format (no random stream,
no vectorized math), decoded under ``pmm-text-decoder/1``; the fixture holds
the file digest, each channel's samples digest and count, and a digest of
each time column's mantissa and exponent arrays.

Fixture history:
- U5 (2026-11-03): initial capture, five cases.

Regenerate ON PURPOSE only, never to make a red test green: a changed digest
means the decoder's output changed, which needs a new contract id in
``text_decoder.DECODERS`` (the old one stays). From the repo root, with the
checkout's interpreter:

    python -c "import sys; sys.path.insert(0, 'tests'); import conftest; \\
               from test_text_decoder_determinism import write_digests; \\
               print(write_digests())"
"""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from predictive_maintenance_mcp.path_safety import open_import_file
from predictive_maintenance_mcp.signal_acquisition import text_decoder as td

FIXTURE = Path(__file__).parent / "fixtures" / "text_decoder_digests.json"
RECIPE = "write_digests"

_SCALING_V_100_MV_PER_G = {
    "source_unit": "V",
    "sensitivity": {
        "value": "100",
        "unit": "mV/g",
        "source": "calibration_certificate",
    },
    "dc_policy": "keep",
}


def _interpretation(delimiter, decimal, first, channels, time_columns=(), header=None):
    return {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": "utf-8",
            "delimiter": delimiter,
            "decimal": decimal,
            "header_line": header,
        },
        "data_lines": {"first": first, "last": None},
        "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
        "time_columns": [{"column": c, "time_unit": "s"} for c in time_columns],
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


def cases() -> dict[str, tuple[bytes, dict]]:
    """The pinned exports and their interpretations."""
    rows = 3000
    repr_rows = "".join(f"{repr(k / 7)},{repr(-k / 13 + 0.1)}\n" for k in range(rows))
    european = "Time;Acc X [V];Acc Y [g];Date\r\n" + "".join(
        f"{k * 39 / 1e6:.6f};{(k % 401 - 200) / 1000:.3f};{(k * 7 % 1001 - 500) / 997:.6f};"
        f"17.10.2026\r\n".replace(".", ",", 3)
        for k in range(rows)
    )
    spaced = "".join(f"  {k / 3:.9e}\t {(k % 11) - 5}\n" for k in range(rows))
    trailing = "a;b;\n" + "".join(f"{k};{k % 5},25;\n" for k in range(rows))
    short = "".join(f"{(k * 104729) % 2001 - 1000}\n" for k in range(rows))
    return {
        "repr_17_digits": (
            repr_rows.encode(),
            _interpretation(",", ".", 1, ((0, None), (1, None))),
        ),
        "european_scaled_with_time": (
            european.encode(),
            _interpretation(
                ";",
                ",",
                2,
                ((1, _SCALING_V_100_MV_PER_G), (2, None)),
                time_columns=(0,),
                header=1,
            ),
        ),
        "whitespace_exponent": (
            spaced.encode(),
            _interpretation("whitespace", ".", 1, ((0, None), (1, None))),
        ),
        "trailing_delimiter": (
            trailing.encode(),
            _interpretation(";", ",", 2, ((1, None),), time_columns=(0,), header=1),
        ),
        "integers_one_column": (
            short.encode(),
            _interpretation(",", ".", 1, ((0, None),)),
        ),
    }


def _array_digest(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).data).hexdigest()


def capture(root: Path) -> dict[str, Any]:
    """Decode every case under *root* and return its pinned values."""
    out: dict[str, Any] = {}
    for name, (data, interp) in cases().items():
        (root / f"{name}.csv").write_bytes(data)
        with open_import_file(
            f"{name}.csv", [root], base=root, max_size=1 << 30
        ) as file:
            result = td.decode_text(file, interp)
        out[name] = {
            "contract": result.contract,
            "source_sha256": result.source_sha256,
            "row_count": result.row_count,
            "channels": {
                str(c.column): {
                    "samples_digest": c.samples_digest,
                    "count": c.sample_count,
                }
                for c in result.channels
            },
            "time_columns": {
                str(t.column): {
                    "mantissa": _array_digest(t.mantissa, "<i8"),
                    "exponent": _array_digest(t.exponent, "<i2"),
                }
                for t in result.time_columns
            },
        }
    return out


def write_digests() -> Path:
    """Regenerate the fixture ON PURPOSE (see the module docstring)."""
    with tempfile.TemporaryDirectory() as tmp:
        payload = capture(Path(tmp))
    FIXTURE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return FIXTURE


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    if not FIXTURE.exists():
        pytest.fail(f"{FIXTURE.name} is missing; regenerate it with {RECIPE}()")
    return json.loads(FIXTURE.read_text("utf-8"))


def test_decoded_output_matches_the_pinned_digests(tmp_path, pinned):
    actual = capture(tmp_path)
    assert actual == pinned, (
        "the decoder's output changed: a new contract id is needed before the "
        f"fixture is regenerated with {RECIPE}()"
    )


def test_the_pinned_cases_cover_scaling_time_and_every_delimiter(pinned):
    assert set(pinned) == set(cases())
    assert {case["contract"] for case in pinned.values()} == {td.DECODER_CONTRACT}
    assert pinned["european_scaled_with_time"]["time_columns"]
    assert all(case["row_count"] == 3000 for case in pinned.values())


def test_the_scaled_channel_is_exactly_raw_over_the_divisor(tmp_path):
    data, interp = cases()["european_scaled_with_time"]
    (tmp_path / "e.csv").write_bytes(data)
    with open_import_file("e.csv", [tmp_path], base=tmp_path, max_size=1 << 30) as file:
        result = td.decode_text(file, interp)
    [scaled] = [c for c in result.channels if c.column == 1]
    raw = np.array([(k % 401 - 200) / 1000 for k in range(3000)])
    assert scaled.samples.tobytes() == (raw / 0.1).tobytes()
