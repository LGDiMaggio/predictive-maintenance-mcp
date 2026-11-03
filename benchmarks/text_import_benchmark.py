"""Throughput and peak memory of the strict text decoder (Signal Import
Engine plan, U5).

Synthesizes a five-column export (a six-decimal time column and four
acceleration channels, semicolon-delimited with a decimal comma, a short
preamble and a header) of the requested size in a temporary folder, then
decodes it in full and in digest mode through the verified descriptor and
reports the throughput and the tracemalloc peak. The figures drive the
deferred size-cap decision; nothing is asserted here.

Usage, from the repo root with the checkout's interpreter::

    python -m benchmarks.text_import_benchmark --size-mb 900
"""

import argparse
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import conftest  # noqa: E402,F401  (pins the package to this checkout)

from predictive_maintenance_mcp.path_safety import open_import_file  # noqa: E402
from predictive_maintenance_mcp.signal_acquisition.text_decoder import (  # noqa: E402
    decode_text,
)

ROW_BYTES = 53  # "0,000039;-0,12345;0,54321;-0,99999;0,00001\r\n" is about this long
CHUNK_ROWS = 1_000_000


def _digits(values: np.ndarray, decimals: int) -> np.ndarray:
    """``values / 10**decimals`` printed with a decimal comma, as uint8 rows."""
    negative = values < 0
    magnitude = np.abs(values)
    integer = magnitude // 10**decimals
    width = int(np.floor(np.log10(max(int(integer.max()), 1)))) + 1
    out = np.empty((values.size, 1 + width + 1 + decimals), dtype=np.uint8)
    out[:, 0] = np.where(negative, ord("-"), ord("0"))
    work = magnitude.copy()
    for j in range(out.shape[1] - 1, width + 1, -1):
        out[:, j] = ord("0") + work % 10
        work //= 10
    out[:, width + 1] = ord(",")
    for j in range(width, 0, -1):
        out[:, j] = ord("0") + work % 10
        work //= 10
    return out


def write_export(path: Path, size_bytes: int) -> int:
    """Write the synthetic export; returns the number of data rows."""
    rows = max(size_bytes // ROW_BYTES, 1)
    rng = np.random.default_rng(13)
    with path.open("wb") as out:
        out.write(b"Sample rate;25600\r\nSensor;ACC\r\n\r\nTime;A1;A2;A3;A4\r\n")
        for first in range(0, rows, CHUNK_ROWS):
            count = min(CHUNK_ROWS, rows - first)
            ticks = np.arange(first, first + count, dtype=np.int64) * 39
            parts = [_digits(ticks, 6)]
            for _ in range(4):
                parts.append(np.full((count, 1), ord(";"), dtype=np.uint8))
                parts.append(_digits(rng.integers(-99999, 100000, count), 5))
            parts.append(np.broadcast_to(np.frombuffer(b"\r\n", np.uint8), (count, 2)))
            out.write(np.concatenate(parts, axis=1).tobytes())
    return rows


def interpretation() -> dict:
    return {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": "utf-8",
            "delimiter": ";",
            "decimal": ",",
            "header_line": 4,
        },
        "data_lines": {"first": 5},
        "time_base": {
            "source": "time_column",
            "sampling_rate_hz": "25600",
            "time_column": 0,
        },
        "time_columns": [{"column": 0, "time_unit": "s"}],
        "channels": [
            {"column": c, "role": "vibration", "quantity": "acceleration", "unit": "g"}
            for c in (1, 2, 3, 4)
        ],
    }


def measure(path: Path, mode: str, trace: bool) -> tuple[float, int, int]:
    root = path.parent
    with open_import_file(path.name, [root], base=root, max_size=1 << 40) as file:
        if trace:
            tracemalloc.start()
        start = time.perf_counter()
        result = decode_text(file, interpretation(), mode=mode)
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if trace else 0
        if trace:
            tracemalloc.stop()
    kept = sum(c.samples.nbytes for c in result.channels if c.samples is not None)
    kept += sum(t.mantissa.nbytes + t.exponent.nbytes for t in result.time_columns)
    return elapsed, peak, kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size-mb", type=float, default=900.0)
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="time without tracemalloc (it slows numpy allocations)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.csv"
        rows = write_export(path, int(args.size_mb * 1e6))
        size = path.stat().st_size
        print(f"file: {size / 1e6:.1f} MB, {rows} rows, 5 columns")
        for mode in ("digest", "full"):
            elapsed, peak, kept = measure(path, mode, trace=not args.no_trace)
            print(
                f"{mode:6s}: {elapsed:6.1f} s, {size / 1e6 / elapsed:6.1f} MB/s, "
                f"kept {kept / 1e6:7.1f} MB"
                + (
                    ""
                    if args.no_trace
                    else f", peak {peak / 1e6:7.1f} MB "
                    f"({peak / max(kept, 1):.2f} x kept)"
                )
            )


if __name__ == "__main__":
    main()
