"""Tests for the STWIN.box reference adapter (``examples/adapters/stwinbox``).

The decoder runs in-process on synthetic DATALOG2 folders built here with
the packet and frame layout of the ST SDK (4-byte packet counters, frames
of ``samples_per_ts`` interleaved int16 samples closed by a float64
timestamp, a final block without timestamp). The command line runs as a
subprocess with the plain ``sys.executable`` pattern: the adapter imports
nothing from the package, so ``conftest.SUBPROCESS_PIN`` is not needed.
AE10 drives two converted acquisitions through ``load_signal`` with a
restart (repository cleared, fresh store) in between.
"""

import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from examples.adapters.stwinbox import stwinbox_to_measurement as adapter
from predictive_maintenance_mcp.asset_ledger.store import LedgerStore
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = REPO_ROOT / "examples" / "adapters" / "stwinbox"
SCRIPT = ADAPTER_DIR / "stwinbox_to_measurement.py"
README = ADAPTER_DIR / "README.md"

ASSET = "P-101"
PREFIX = "motor_de"
SENSOR = "iis3dwb_acc"
SERIAL = "0054TEST000000000000001"
ALIAS = "STWIN_BOX_T"
SENSITIVITY = 0.0004879999905824661
MEASODR = 26585.318359375
SAMPLES_PER_TS = 1000
DIM = 3
N_SAMPLES = 2500
#: 2500 samples x 3 axes x 2 bytes + 2 timestamps x 8 bytes = 15016 = 4 x 3754,
#: so the synthetic stream ends on a packet boundary, as real files do.
USB_DPS = 3754
START_TIME = "2026-06-01T10:00:00.000Z"
START_TIME_WEEK_LATER = "2026-06-08T10:00:00.000Z"
FIRST_TIMESTAMP = 0.0572
WINDOW_HEADER = "Time [s],A_x [g],A_y [g],A_z [g]"


# ---------------------------------------------------------------------------
# Synthetic DATALOG2 folders (layout verified against the ST SDK)
# ---------------------------------------------------------------------------


def synthetic_counts(seed: int = 0, n: int = N_SAMPLES) -> np.ndarray:
    """``(n, 3)`` int16 counts with a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32768, size=(n, DIM), dtype=np.int16)


def expected_samples(counts: np.ndarray) -> np.ndarray:
    """What the adapter must write: counts times sensitivity, in float32."""
    return counts.astype(np.float32) * np.float32(SENSITIVITY)


def build_payload(
    counts: np.ndarray, *, spts: int = SAMPLES_PER_TS, timestamps=None
) -> bytes:
    """Frames of *spts* samples closed by a float64 timestamp, then the
    samples of an incomplete final block without timestamp."""
    period = spts / MEASODR
    complete = len(counts) // spts
    if timestamps is None:
        timestamps = [FIRST_TIMESTAMP + k * period for k in range(complete)]
    chunks = []
    for k in range(complete):
        chunks.append(counts[k * spts : (k + 1) * spts].astype("<i2").tobytes())
        chunks.append(struct.pack("<d", timestamps[k]))
    tail = counts[complete * spts :]
    if len(tail):
        chunks.append(tail.astype("<i2").tobytes())
    return b"".join(chunks)


def packetize(payload: bytes, dps: int, *, counters=None) -> bytes:
    """Prefix every *dps*-byte payload chunk with its uint32 byte counter."""
    offsets = list(range(0, len(payload), dps))
    if counters is None:
        counters = offsets
    return b"".join(
        struct.pack("<I", counter) + payload[offset : offset + dps]
        for counter, offset in zip(counters, offsets)
    )


def write_datalog2_folder(
    folder: Path,
    *,
    counts: np.ndarray,
    start_time: Optional[str] = START_TIME,
    component_overrides: Optional[dict[str, Any]] = None,
    interface: int = 1,
    dps: int = USB_DPS,
    dat: Optional[bytes] = None,
) -> Path:
    """Write ``acquisition_info.json``, ``device_config.json`` and the
    ``.dat`` of one synthetic acquisition. A ``None`` override removes the
    key; ``start_time=None`` removes ``start_time``."""
    folder.mkdir(parents=True, exist_ok=True)
    component: dict[str, Any] = {
        "odr": 0,
        "fs": 3,
        "enable": True,
        "samples_per_ts": SAMPLES_PER_TS,
        "dim": DIM,
        "ioffset": 0.0197,
        "measodr": MEASODR,
        "usb_dps": dps,
        "sd_dps": dps + 4,
        "sensitivity": SENSITIVITY,
        "data_type": "int16",
        "sensor_category": 0,
        "st_ble_stream": {
            "id": 7,
            "acc": {"enable": False, "unit": "g", "format": "int16_t", "channels": 3},
        },
        "c_type": 0,
    }
    for key, value in (component_overrides or {}).items():
        if value is None:
            component.pop(key, None)
        else:
            component[key] = value
    config = {
        "devices": [
            {
                "sn": SERIAL,
                "components": [
                    {
                        "DeviceInformation": {
                            "manufacturer": "STMicroelectronics",
                            "model": "STEVAL-STWINBX1",
                        }
                    },
                    {
                        "firmware_info": {
                            "alias": ALIAS,
                            "fw_name": "FP-SNS-DATALOG2_Datalog2",
                            "fw_version": "3.2.0",
                        }
                    },
                    {SENSOR: component},
                ],
            }
        ]
    }
    info: dict[str, Any] = {
        "name": "synthetic",
        "uuid": "00000000-0000-0000-0000-000000000000",
        "data_ext": ".dat",
        "data_fmt": "HSD_2.0.0",
        "interface": interface,
        "schema_version": "2.0.0",
    }
    if start_time is not None:
        info["start_time"] = start_time
        info["end_time"] = start_time
    (folder / "device_config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "acquisition_info.json").write_text(json.dumps(info), encoding="utf-8")
    if dat is None:
        dat = packetize(build_payload(counts), dps)
    (folder / f"{SENSOR}.dat").write_bytes(dat)
    return folder


def decode_folder(folder: Path) -> adapter.DecodedSensor:
    acquisition = adapter.read_datalog_acquisition(folder, sensor=SENSOR)
    raw = (folder / f"{SENSOR}.dat").read_bytes()
    return adapter.decode_dat(raw, acquisition.sensor)


def write_window_csv(
    path: Path, *, rows: int = 200, rate: float = 26667.0
) -> np.ndarray:
    """A recorder window with *rows* samples; returns the three columns."""
    rng = np.random.default_rng(3)
    columns = np.round(rng.standard_normal((rows, 3)) * 0.1, 5)
    lines = [WINDOW_HEADER]
    for i in range(rows):
        lines.append(
            f"{i / rate:.6f},{columns[i, 0]:.5f},{columns[i, 1]:.5f},"
            f"{columns[i, 2]:.5f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return columns


# ---------------------------------------------------------------------------
# Command line helpers
# ---------------------------------------------------------------------------


def run_cli(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def datalog2_args(folder: Path, out: Path, *extra: str) -> list[str]:
    return [
        str(folder),
        "--asset-id",
        ASSET,
        "--measurement-point-prefix",
        PREFIX,
        "--out",
        str(out),
        *extra,
    ]


def window_args(csv: Path, out: Path, *extra: str) -> list[str]:
    return [
        str(csv),
        "--input-format",
        "window-csv",
        "--asset-id",
        ASSET,
        "--measurement-point-prefix",
        PREFIX,
        "--out",
        str(out),
        *extra,
    ]


def output_paths(out: Path, axis: str, stamp: str) -> tuple[Path, Path]:
    stem = f"{ASSET}_{PREFIX}_{axis}_{stamp}"
    return out / f"{stem}.bin", out / f"{stem}_metadata.json"


def read_companion(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nothing_written(out: Path) -> bool:
    return not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------
# Decoder (in-process)
# ---------------------------------------------------------------------------


class TestDecoder:
    def test_happy_path_recovers_every_sample(self, tmp_path):
        counts = synthetic_counts(0)
        folder = write_datalog2_folder(tmp_path / "20260601_10_00_00", counts=counts)
        decoded = decode_folder(folder)

        assert decoded.samples.shape == (N_SAMPLES, DIM)
        assert decoded.samples.dtype == np.float32
        assert decoded.packets == 4
        assert decoded.complete_frames == 2
        assert decoded.trailing_samples == 500
        assert decoded.dropped_tail_bytes == 0
        assert np.array_equal(decoded.samples, expected_samples(counts))
        np.testing.assert_allclose(
            decoded.samples, counts.astype(np.float64) * SENSITIVITY, rtol=2e-7
        )

    def test_last_partial_block_keeps_no_timestamp_bytes(self, tmp_path):
        """The two timestamps sit between samples 1000/1001 and 2000/2001;
        a decoder that let them through would corrupt exactly those rows."""
        counts = synthetic_counts(1)
        folder = write_datalog2_folder(tmp_path / "acq", counts=counts)
        decoded = decode_folder(folder)
        period = SAMPLES_PER_TS / MEASODR
        np.testing.assert_allclose(
            decoded.timestamps, [FIRST_TIMESTAMP, FIRST_TIMESTAMP + period]
        )
        for boundary in (SAMPLES_PER_TS, 2 * SAMPLES_PER_TS):
            window = slice(boundary - 2, boundary + 2)
            assert np.array_equal(
                decoded.samples[window], expected_samples(counts)[window]
            )
        assert np.array_equal(decoded.samples[2000:], expected_samples(counts)[2000:])

    def test_sd_card_interface_subtracts_the_counter_from_sd_dps(self, tmp_path):
        counts = synthetic_counts(2)
        folder = write_datalog2_folder(
            tmp_path / "DL2_00001", counts=counts, interface=0
        )
        acquisition = adapter.read_datalog_acquisition(folder, sensor=SENSOR)
        assert acquisition.sensor.packet_payload_bytes == USB_DPS
        decoded = decode_folder(folder)
        assert np.array_equal(decoded.samples, expected_samples(counts))

    def test_incomplete_trailing_packet_is_dropped_not_decoded(self, tmp_path):
        counts = synthetic_counts(4)
        dat = packetize(build_payload(counts), 4000)  # 15016 = 3 x 4000 + 3016
        folder = write_datalog2_folder(
            tmp_path / "acq", counts=counts, dps=4000, dat=dat
        )
        decoded = decode_folder(folder)
        assert decoded.packets == 3
        assert decoded.dropped_tail_bytes == 3016 + adapter.PACKET_COUNTER_BYTES
        # 12000 payload bytes: one 6008-byte frame, then 5992 bytes of samples
        assert decoded.complete_frames == 1
        assert decoded.trailing_samples == 998
        assert np.array_equal(decoded.samples, expected_samples(counts)[:1998])
        conversion = adapter.convert_datalog2(
            folder, out_dir=tmp_path / "out", asset_id=ASSET, point_prefix=PREFIX
        )
        assert any("trailing bytes" in w for w in conversion.warnings)

    def test_packet_counter_gap_is_refused(self, tmp_path):
        counts = synthetic_counts(5)
        counters = [0, USB_DPS, 3 * USB_DPS, 4 * USB_DPS]  # one packet lost
        dat = packetize(build_payload(counts), USB_DPS, counters=counters)
        folder = write_datalog2_folder(tmp_path / "acq", counts=counts, dat=dat)
        with pytest.raises(adapter.AdapterError, match="lost data"):
            decode_folder(folder)

    def test_timestamp_step_outside_band_is_refused(self, tmp_path):
        counts = synthetic_counts(6)
        period = SAMPLES_PER_TS / MEASODR
        payload = build_payload(
            counts, timestamps=[FIRST_TIMESTAMP, FIRST_TIMESTAMP + 50 * period]
        )
        folder = write_datalog2_folder(
            tmp_path / "acq", counts=counts, dat=packetize(payload, USB_DPS)
        )
        with pytest.raises(adapter.AdapterError, match="frame period"):
            decode_folder(folder)

    def test_missing_sensitivity_names_the_field(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            component_overrides={"sensitivity": None},
        )
        with pytest.raises(adapter.AdapterError, match="sensitivity"):
            adapter.read_datalog_acquisition(folder, sensor=SENSOR)

    def test_non_int16_data_type_names_the_field(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            component_overrides={"data_type": "int24"},
        )
        with pytest.raises(adapter.AdapterError, match="data_type"):
            adapter.read_datalog_acquisition(folder, sensor=SENSOR)

    def test_odr_index_is_never_read_as_hertz(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            component_overrides={"measodr": None, "odr": 26667},
        )
        with pytest.raises(adapter.AdapterError, match="enumeration index"):
            adapter.convert_datalog2(
                folder, out_dir=tmp_path / "out", asset_id=ASSET, point_prefix=PREFIX
            )
        conversion = adapter.convert_datalog2(
            folder,
            out_dir=tmp_path / "out",
            asset_id=ASSET,
            point_prefix=PREFIX,
            sampling_rate=26667.0,
        )
        companion = conversion.outputs[0].companion
        assert companion["sampling_rate"] == 26667.0
        assert companion["adapter"]["sampling_rate_source"] == "argument"
        assert companion["adapter"]["nominal_odr_index"] == 26667
        assert companion["adapter"]["measured_odr_hz"] is None

    def test_start_time_offset_relabels_the_digits(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        conversion = adapter.convert_datalog2(
            folder,
            out_dir=tmp_path / "out",
            asset_id=ASSET,
            point_prefix=PREFIX,
            start_time_offset="+02:00",
        )
        companion = conversion.outputs[0].companion
        assert companion["measurement"]["acquired_at"] == "2026-06-01T10:00:00+02:00"
        block = companion["adapter"]
        assert (
            block["acquired_at_source"]
            == "acquisition_info.start_time_at_declared_offset"
        )
        assert block["start_time_offset_declared"] == "+02:00"
        assert block["acquisition"]["start_time"] == START_TIME
        assert not any("Z suffix" in w for w in conversion.warnings)
        # the output stamp is the instant in UTC
        assert conversion.outputs[0].bin_path.name.endswith("_20260601T080000Z.bin")

    def test_start_time_copied_verbatim_warns_about_the_label(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        conversion = adapter.convert_datalog2(
            folder, out_dir=tmp_path / "out", asset_id=ASSET, point_prefix=PREFIX
        )
        assert (
            conversion.outputs[0].companion["measurement"]["acquired_at"] == START_TIME
        )
        assert any(
            "Z suffix" in w and "--start-time-offset" in w for w in conversion.warnings
        )

    def test_offset_and_instant_together_are_refused(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        with pytest.raises(adapter.AdapterError, match="not both"):
            adapter.convert_datalog2(
                folder,
                out_dir=tmp_path / "out",
                asset_id=ASSET,
                point_prefix=PREFIX,
                acquired_at="2026-06-01T08:00:00Z",
                start_time_offset="+02:00",
            )

    @pytest.mark.parametrize("bad", ["+2:00", "02:00", "+15:00", "+01:60", "UTC", ""])
    def test_utc_offset_grammar(self, bad):
        with pytest.raises(adapter.AdapterError):
            adapter.parse_utc_offset(bad, source="--start-time-offset")

    def test_utc_offset_signs(self):
        from datetime import timedelta

        assert adapter.parse_utc_offset("+05:30", source="x").utcoffset(
            None
        ) == timedelta(hours=5, minutes=30)
        assert adapter.parse_utc_offset("-05:00", source="x").utcoffset(
            None
        ) == timedelta(hours=-5)

    def test_never_set_clock_warns(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            start_time="2000-01-01T00:00:05.000Z",
        )
        conversion = adapter.convert_datalog2(
            folder, out_dir=tmp_path / "out", asset_id=ASSET, point_prefix=PREFIX
        )
        assert any("real-time clock" in w for w in conversion.warnings)

    def test_window_filename_instant_is_utc_with_offset(self):
        instant = adapter.acquired_at_from_window_filename(
            Path("P-201_20260629_132700_123456.csv")
        )
        assert instant == "2026-06-29T13:27:00.123456+00:00"
        with pytest.raises(adapter.AdapterError, match="--acquired-at"):
            adapter.acquired_at_from_window_filename(Path("window.csv"))

    @pytest.mark.parametrize("bad", ["", "-P-101", "P 101", "CON", "P-101.", "a" * 101])
    def test_ledger_id_grammar_mirrors_the_server(self, bad):
        with pytest.raises(adapter.AdapterError, match="ledger id"):
            adapter.validate_ledger_id(bad, kind="asset_id")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class TestCommandLine:
    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "window-csv" in result.stdout
        assert "--start-time-offset" in result.stdout

    def test_datalog2_writes_three_axes_with_companions(self, tmp_path):
        counts = synthetic_counts(0)
        folder = write_datalog2_folder(tmp_path / "20260601_10_00_00", counts=counts)
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out))
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.splitlines()) == 6
        assert "Z suffix" in result.stderr

        for column, axis in enumerate("xyz"):
            bin_path, companion_path = output_paths(out, axis, "20260601T100000Z")
            assert bin_path.is_file() and companion_path.is_file()
            samples = np.fromfile(bin_path, dtype="<f4")
            assert samples.shape == (N_SAMPLES,)
            assert np.array_equal(samples, expected_samples(counts)[:, column])

            companion = read_companion(companion_path)
            assert companion["sample_format"] == "float32"
            assert companion["byte_order"] == "little"
            assert companion["n_channels"] == 1
            assert companion["sampling_rate"] == MEASODR
            assert companion["signal_unit"] == "g"
            measurement = companion["measurement"]
            assert measurement["asset_id"] == ASSET
            assert measurement["measurement_point_id"] == f"{PREFIX}_{axis}"
            assert measurement["acquired_at"] == START_TIME
            assert measurement["direction"] == axis
            assert measurement["sensor_id"] == SERIAL
            assert measurement["declared_by"] == "adapter:stwinbox"
            assert "rpm" not in measurement
            block = companion["adapter"]
            assert block["firmware"] == {
                "alias": ALIAS,
                "name": "FP-SNS-DATALOG2_Datalog2",
                "version": "3.2.0",
            }
            assert block["source"] == folder.name
            assert block["decoder"] == adapter.DATALOG2_DECODER
            assert block["measured_odr_hz"] == MEASODR
            assert block["nominal_odr_index"] == 0
            assert block["nominal_odr_hz"] is None
            assert block["sensitivity_g_per_lsb"] == SENSITIVITY
            assert block["acquired_at_source"] == "acquisition_info.start_time"
            assert block["sampling_rate_source"] == "device_config.measodr"
            assert block["axis"] == axis
            assert block["samples"] == N_SAMPLES

    def test_rpm_is_recorded_only_when_given(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out, "--rpm", "1480"))
        assert result.returncode == 0, result.stderr
        _, companion_path = output_paths(out, "x", "20260601T100000Z")
        assert read_companion(companion_path)["measurement"]["rpm"] == 1480.0

    def test_missing_sensitivity_exits_nonzero_and_writes_nothing(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            component_overrides={"sensitivity": None},
        )
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out))
        assert result.returncode != 0
        assert "sensitivity" in result.stderr
        assert result.stdout == ""
        assert nothing_written(out)

    def test_non_int16_data_type_is_refused(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq",
            counts=synthetic_counts(),
            component_overrides={"data_type": "float32"},
        )
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out))
        assert result.returncode != 0
        assert "data_type" in result.stderr
        assert nothing_written(out)

    def test_missing_start_time_is_refused_unless_acquired_at(self, tmp_path):
        folder = write_datalog2_folder(
            tmp_path / "acq", counts=synthetic_counts(), start_time=None
        )
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out))
        assert result.returncode != 0
        assert "start_time" in result.stderr and "--acquired-at" in result.stderr
        assert nothing_written(out)

        result = run_cli(
            *datalog2_args(folder, out, "--acquired-at", "2026-06-01T12:00:00+02:00")
        )
        assert result.returncode == 0, result.stderr
        _, companion_path = output_paths(out, "x", "20260601T100000Z")
        companion = read_companion(companion_path)
        assert companion["measurement"]["acquired_at"] == "2026-06-01T12:00:00+02:00"
        assert companion["adapter"]["acquired_at_source"] == "argument"

    def test_start_time_offset_option(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        out = tmp_path / "out"
        result = run_cli(*datalog2_args(folder, out, "--start-time-offset", "+02:00"))
        assert result.returncode == 0, result.stderr
        assert "Z suffix" not in result.stderr
        _, companion_path = output_paths(out, "x", "20260601T080000Z")
        companion = read_companion(companion_path)
        assert companion["measurement"]["acquired_at"] == "2026-06-01T10:00:00+02:00"
        assert companion["adapter"]["start_time_offset_declared"] == "+02:00"

    def test_existing_output_is_refused_without_force(self, tmp_path):
        folder = write_datalog2_folder(tmp_path / "acq", counts=synthetic_counts())
        out = tmp_path / "out"
        assert run_cli(*datalog2_args(folder, out)).returncode == 0
        bin_path, _ = output_paths(out, "x", "20260601T100000Z")
        before = bin_path.stat().st_mtime_ns

        result = run_cli(*datalog2_args(folder, out))
        assert result.returncode != 0
        assert "--force" in result.stderr
        assert bin_path.stat().st_mtime_ns == before

        result = run_cli(*datalog2_args(folder, out, "--force"))
        assert result.returncode == 0, result.stderr

    def test_invalid_asset_id_is_refused_before_reading(self, tmp_path):
        out = tmp_path / "out"
        result = run_cli(
            str(tmp_path / "missing"),
            "--asset-id",
            "CON",
            "--measurement-point-prefix",
            PREFIX,
            "--out",
            str(out),
        )
        assert result.returncode != 0
        assert "asset_id" in result.stderr
        assert nothing_written(out)

    def test_window_csv_happy_path(self, tmp_path):
        csv = tmp_path / "window.csv"
        columns = write_window_csv(csv)
        out = tmp_path / "out"
        result = run_cli(
            *window_args(
                csv,
                out,
                "--sampling-rate",
                "26667",
                "--acquired-at",
                "2026-06-29T13:27:00Z",
            )
        )
        assert result.returncode == 0, result.stderr
        for column, axis in enumerate("xyz"):
            bin_path, companion_path = output_paths(out, axis, "20260629T132700Z")
            samples = np.fromfile(bin_path, dtype="<f4")
            assert samples.shape == (200,)
            assert np.array_equal(samples, columns[:, column].astype(np.float32))
            companion = read_companion(companion_path)
            assert companion["sampling_rate"] == 26667.0
            assert companion["signal_unit"] == "g"
            assert companion["measurement"]["acquired_at"] == "2026-06-29T13:27:00Z"
            assert (
                companion["measurement"]["measurement_point_id"] == f"{PREFIX}_{axis}"
            )
            assert companion["measurement"]["direction"] == axis
            assert "sensor_id" not in companion["measurement"]
            block = companion["adapter"]
            assert block["decoder"] == adapter.WINDOW_CSV_DECODER
            assert block["columns"] == ["A_x [g]", "A_y [g]", "A_z [g]"]
            assert block["rows"] == 200
            assert block["acquired_at_source"] == "argument"

    def test_window_csv_without_sampling_rate_is_refused(self, tmp_path):
        csv = tmp_path / "window.csv"
        write_window_csv(csv)
        out = tmp_path / "out"
        result = run_cli(
            *window_args(csv, out, "--acquired-at", "2026-06-29T13:27:00Z")
        )
        assert result.returncode != 0
        assert "--sampling-rate" in result.stderr
        assert nothing_written(out)

    def test_window_csv_without_instant_is_refused(self, tmp_path):
        csv = tmp_path / "window.csv"
        write_window_csv(csv)
        out = tmp_path / "out"
        result = run_cli(*window_args(csv, out, "--sampling-rate", "26667"))
        assert result.returncode != 0
        assert "--acquired-at" in result.stderr
        assert nothing_written(out)

    def test_window_csv_instant_from_filename(self, tmp_path):
        csv = tmp_path / "P-201_20260629_132700_123456.csv"
        write_window_csv(csv)
        out = tmp_path / "out"
        result = run_cli(
            *window_args(
                csv, out, "--sampling-rate", "26667", "--acquired-at-from-filename"
            )
        )
        assert result.returncode == 0, result.stderr
        _, companion_path = output_paths(out, "x", "20260629T132700Z")
        companion = read_companion(companion_path)
        assert (
            companion["measurement"]["acquired_at"]
            == "2026-06-29T13:27:00.123456+00:00"
        )
        assert companion["adapter"]["acquired_at_source"] == "filename"

    def test_start_time_offset_is_datalog2_only(self, tmp_path):
        csv = tmp_path / "window.csv"
        write_window_csv(csv)
        out = tmp_path / "out"
        result = run_cli(
            *window_args(
                csv,
                out,
                "--sampling-rate",
                "26667",
                "--acquired-at",
                "2026-06-29T13:27:00Z",
                "--start-time-offset",
                "+01:00",
            )
        )
        assert result.returncode != 0
        assert nothing_written(out)


# ---------------------------------------------------------------------------
# AE10: two acquisitions a week apart, through load_signal, restart between
# ---------------------------------------------------------------------------


class TestEndToEndAE10:
    @pytest.mark.asyncio
    async def test_two_acquisitions_form_one_history(self, tmp_path, sandbox_data_dir):
        first = write_datalog2_folder(
            tmp_path / "20260601_10_00_00", counts=synthetic_counts(11)
        )
        second = write_datalog2_folder(
            tmp_path / "20260608_10_00_00",
            counts=synthetic_counts(12),
            start_time=START_TIME_WEEK_LATER,
        )
        out = sandbox_data_dir / "stwinbox"
        for folder in (first, second):
            result = run_cli(*datalog2_args(folder, out, "--rpm", "1480"))
            assert result.returncode == 0, result.stderr
        first_bin, _ = output_paths(out, "x", "20260601T100000Z")
        second_bin, _ = output_paths(out, "x", "20260608T100000Z")

        server = MCPServer("test-stwinbox-adapter")
        register(server)
        tools = {t.name: t.fn for t in server._tool_manager._tools.values()}
        ctx = AsyncMock()
        repo = get_repository()
        repo.clear_all()
        try:
            # the later acquisition first: the history must still order by acquired_at
            later = await tools["load_signal"](
                ctx=ctx, filepath=f"stwinbox/{second_bin.name}"
            )
            assert later.measurement["ledger_status"] == "recorded"
            assert later.measurement["measurement_point_id"] == f"{PREFIX}_x"

            # restart: the repository is emptied and a fresh store reads the files
            repo.clear_all()
            store = LedgerStore(get_ledger_dir())

            earlier = await tools["load_signal"](
                ctx=ctx, filepath=f"stwinbox/{first_bin.name}"
            )
            assert earlier.measurement["ledger_status"] == "recorded"
            assert earlier.measurement["asset_id"] == ASSET

            view = store.read_view(ASSET)
            assert view["ordered_measurement_ids"] == [
                earlier.measurement["measurement_id"],
                later.measurement["measurement_id"],
            ]
            instants = []
            for measurement_id in view["ordered_measurement_ids"]:
                current = view["measurements"][measurement_id]["current"]
                assert current["measurement_point_id"] == f"{PREFIX}_x"
                declaration = current["declaration"]
                assert declaration["direction"] == "x"
                assert declaration["sensor_id"] == SERIAL
                assert declaration["rpm"] == 1480.0
                assert declaration["sampling_rate"] == MEASODR
                assert declaration["signal_unit"] == "g"
                instants.append(declaration["acquired_at"])
            assert instants == sorted(instants)
            assert instants[0].startswith("2026-06-01T10:00:00")
            assert instants[1].startswith("2026-06-08T10:00:00")
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadme:
    @pytest.fixture(scope="class")
    def text(self) -> str:
        return README.read_text(encoding="utf-8")

    def test_no_em_dash(self, text):
        assert "—" not in text

    def test_checklist_section_with_entries_to_record(self, text):
        assert "## Real-device checklist" in text
        assert text.count("to be recorded") >= 5

    def test_no_call_snippets_for_ledger_tools(self, text):
        """The documented-calls guard executes ``name(kw=...)`` snippets; the
        tools of the ledger surface are named in prose here, never called."""
        named_in_prose = (
            "declare_measurement_point",
            "get_asset_history",
            "assess_asset_change",
            "load_signal",
        )
        for name in named_in_prose:
            assert name in text
        for name in (*named_in_prose, "declare_healthy_baseline"):
            assert re.search(rf"\b{name}\(", text) is None, name

    def test_states_the_documented_caveats(self, text):
        lowered = text.lower()
        assert "one file per axis" in lowered
        assert "enumeration index" in lowered
        assert "--start-time-offset" in text
        for phrase in ("in minutes", "instantly", "no expertise"):
            assert phrase not in lowered
