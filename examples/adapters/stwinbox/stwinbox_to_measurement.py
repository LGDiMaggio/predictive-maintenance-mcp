#!/usr/bin/env python3
"""Convert STWIN.box acquisitions into measurement files for the server.

The STEVAL-STWINBX1 board running the FP-SNS-DATALOG2 firmware writes one
folder per capture: ``acquisition_info.json``, ``device_config.json`` and one
``<sensor>.dat`` per active sensor. The USB bridge recorder of the same board
writes short windows as CSV files with the header
``Time [s],A_x [g],A_y [g],A_z [g]``. Neither form is loadable by the server
as it is: the ``.dat`` stream carries packet counters and timestamps between
the samples, and the CSV puts time in its first column.

This script reads either form and writes, for every accelerometer axis, a
headerless float32 little-endian ``.bin`` in g plus the companion
``<stem>_metadata.json`` that the server's ``load_signal`` tool reads: the
raw decode declaration (``sample_format``, ``byte_order``, ``sampling_rate``,
``signal_unit``), the ``measurement`` object that identifies the asset, the
measurement point, the instant and the direction, and an ``adapter`` block
with the provenance of every declared value.

It imports nothing from the server: the companion file is the whole
integration surface. Standard library and NumPy only.

DATALOG2 layout, as decoded by the ST ``stdatalog-pysdk`` (``HSDatalog_v2``):
a ``.dat`` file is a sequence of packets, each opened by a 4-byte
little-endian byte counter. The packet payload size is ``usb_dps`` (USB),
``sd_dps - 4`` (SD card), ``ble_dps`` or ``serial_dps`` according to
``acquisition_info.json::interface``. The concatenated payloads form frames
of ``samples_per_ts * dim`` interleaved int16 samples followed by one
float64 timestamp in seconds. Samples are scaled by the component's
``sensitivity`` (g per LSB). The component's ``odr`` field is a firmware
enumeration index, not a frequency: the sampling rate declared here is
``measodr``, the rate the firmware measured during the capture.

``acquisition_info.json::start_time`` is written with a ``Z`` suffix, but
when the host starts the log the SDK sets the board clock from the host's
local wall-clock time without an offset (``HSDLink_v2.set_rtc_time``), so
the digits are host local time and the ``Z`` is not evidence. The adapter
copies the value as written and warns; ``--start-time-offset`` declares the
offset the clock actually held, and ``--acquired-at`` declares the instant.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ADAPTER_NAME = "stwinbox_to_measurement"
ADAPTER_VERSION = "1.0"
DATALOG2_DECODER = "datalog2-packet-frames-v1"
WINDOW_CSV_DECODER = "window-csv-columns-v1"

INPUT_DATALOG2 = "datalog2"
INPUT_WINDOW_CSV = "window-csv"
INPUT_FORMATS = (INPUT_DATALOG2, INPUT_WINDOW_CSV)

DEFAULT_SENSOR = "iis3dwb_acc"
DEFAULT_DECLARED_BY = "adapter:stwinbox"
AXES = ("x", "y", "z")
SIGNAL_UNIT = "g"
SAMPLE_FORMAT = "float32"
BYTE_ORDER = "little"

#: Every packet of a ``.dat`` file opens with a uint32 little-endian byte
#: counter (``data_protocol_size`` in the SDK). The counter of packet ``n+1``
#: equals the counter of packet ``n`` plus the payload size: a different step
#: means the link lost data.
PACKET_COUNTER_BYTES = 4
#: Every frame of the payload stream closes with a float64 timestamp.
TIMESTAMP_BYTES = 8
#: ``acquisition_info.json::interface`` selects the packet-size field of the
#: component. On the SD card the field counts the counter bytes as well.
INTERFACE_PACKET_KEYS = {0: "sd_dps", 1: "usb_dps", 2: "ble_dps", 3: "serial_dps"}
INTERFACE_NAMES = {0: "SD card", 1: "USB", 2: "BLE", 3: "serial"}
SUPPORTED_DATA_TYPE = "int16"
SAMPLE_DTYPE = np.dtype("<i2")
#: Timestamp steps outside this band around the expected frame period are
#: what the SDK's timestamp recovery treats as corrupted frames.
TIMESTAMP_STEP_BAND = (0.1, 10.0)
#: A real-time clock that was never set reports the year 2000 (the DATALOG2
#: example acquisitions start on 2000-01-01).
RTC_SUSPECT_BEFORE_YEAR = 2010
#: ``--start-time-offset`` form: a signed ``HH:MM`` UTC offset.
UTC_OFFSET_RE = re.compile(r"^(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")
MAX_UTC_OFFSET = timedelta(hours=14)

WINDOW_CSV_HEADER = ("Time [s]", "A_x [g]", "A_y [g]", "A_z [g]")
WINDOW_CSV_FILENAME_RE = re.compile(
    r"^(?P<asset>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d{6})$"
)
#: Declared and implied rates further apart than this get a warning.
WINDOW_CSV_RATE_TOLERANCE = 0.05

#: The server's ledger-id grammar, mirrored so a refusal happens here, before
#: any file is written, with the same wording the server would use.
LEDGER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
MAX_LEDGER_ID_CHARS = 100
MAX_FREE_TEXT_CHARS = 200
WIN32_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
LEDGER_ID_GRAMMAR = (
    "start with a letter or digit, use only letters, digits, '_', '-' and '.', "
    "end with neither a dot nor a space, have at most "
    f"{MAX_LEDGER_ID_CHARS} characters, and not be a Windows reserved device "
    "name (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9)"
)


class AdapterError(Exception):
    """A refusal. The message names the problem and the remedy; nothing is written."""


# ---------------------------------------------------------------------------
# Declarations shared by both input formats
# ---------------------------------------------------------------------------


def validate_ledger_id(value: object, *, kind: str) -> str:
    """Return *value* if it satisfies the server's ledger-id grammar.

    Args:
        value: The declared id.
        kind: Field name used in the refusal (``asset_id``, ...).

    Raises:
        AdapterError: Naming the value, the problem and the grammar.
    """
    if not isinstance(value, str) or not value:
        raise AdapterError(
            f"Invalid {kind} {value!r}: must be a non-empty string. A ledger id "
            f"must {LEDGER_ID_GRAMMAR}."
        )
    text = value
    problem: Optional[str] = None
    if len(text) > MAX_LEDGER_ID_CHARS:
        problem = f"is {len(text)} characters long, over the cap"
    elif not LEDGER_ID_RE.match(text):
        problem = "contains characters outside the grammar or starts with one"
    elif text.endswith("."):
        problem = "ends with a dot"
    elif text.split(".", 1)[0].upper() in WIN32_RESERVED_NAMES:
        problem = "is a Windows reserved device name"
    if problem is not None:
        raise AdapterError(
            f"Invalid {kind} {text!r}: {problem}. A ledger id must "
            f"{LEDGER_ID_GRAMMAR}."
        )
    return text


def validate_free_text(value: object, *, kind: str) -> str:
    """Return *value* if it is one bounded line without control characters."""
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"Invalid {kind} {value!r}: must be a non-empty string.")
    if len(value) > MAX_FREE_TEXT_CHARS:
        raise AdapterError(
            f"Invalid {kind}: {len(value)} characters, over the "
            f"{MAX_FREE_TEXT_CHARS}-character cap."
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise AdapterError(f"Invalid {kind} {value!r}: control characters.")
    return value


def parse_instant(value: object, *, source: str) -> datetime:
    """Parse an ISO 8601 instant that declares its UTC offset.

    Args:
        value: The date-time string (``2026-02-28T23:48:45.000Z`` and
            ``2026-08-20T13:42:00+02:00`` are both accepted).
        source: Where the value came from, for the refusal.

    Raises:
        AdapterError: Not a date-time, or a date-time without an offset.
    """
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{source}: {value!r} is not an ISO 8601 date-time.")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        raise AdapterError(
            f"{source}: {value!r} is not an ISO 8601 date-time "
            f"(expected a form like 2026-06-29T13:27:00Z)."
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdapterError(
            f"{source}: {value!r} declares no UTC offset. Declare the instant "
            f"with its offset (2026-06-29T13:27:00Z or "
            f"2026-06-29T15:27:00+02:00) so the asset history orders it "
            f"against measurements from other sources."
        )
    return parsed


def output_stamp(instant: datetime) -> str:
    """Compact UTC form of *instant* used in output file names."""
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_utc_offset(text: object, *, source: str) -> timezone:
    """Parse a ``+HH:MM`` / ``-HH:MM`` offset into a fixed-offset timezone.

    Raises:
        AdapterError: Not of that form, or beyond 14 hours.
    """
    match = UTC_OFFSET_RE.match(text.strip()) if isinstance(text, str) else None
    if match is None:
        raise AdapterError(
            f"{source}: {text!r} is not a UTC offset of the form +HH:MM or "
            f"-HH:MM (for example +01:00, or =-05:00 after the option name)."
        )
    delta = timedelta(hours=int(match["hours"]), minutes=int(match["minutes"]))
    if int(match["minutes"]) > 59 or delta > MAX_UTC_OFFSET:
        raise AdapterError(f"{source}: {text!r} is not a valid UTC offset.")
    return timezone(-delta if match["sign"] == "-" else delta)


def at_declared_offset(instant: datetime, offset: timezone) -> str:
    """Keep the wall-clock digits of *instant* and attach *offset* to them.

    Used for a ``start_time`` whose ``Z`` suffix is known to be a label
    rather than a fact: the board clock held local time at *offset*.
    """
    return instant.replace(tzinfo=offset).isoformat()


def acquired_at_from_window_filename(path: Path) -> str:
    """Instant encoded by the recorder's ``<asset>_<YYYYmmdd>_<HHMMSS>_<ffffff>.csv``.

    The recorder names windows by their UTC start. Returns the ISO 8601 form
    with the ``+00:00`` offset.

    Raises:
        AdapterError: The name does not follow the convention.
    """
    match = WINDOW_CSV_FILENAME_RE.match(path.stem)
    if match is None:
        raise AdapterError(
            f"{path.name}: the file name does not follow the recorder "
            f"convention <asset>_<YYYYmmdd>_<HHMMSS>_<ffffff>.csv, so the "
            f"acquisition instant cannot be read from it. Pass --acquired-at."
        )
    stamp = f"{match['date']}_{match['time']}_{match['micro']}"
    try:
        instant = datetime.strptime(stamp, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        raise AdapterError(
            f"{path.name}: {stamp!r} is not a valid UTC date-time. "
            f"Pass --acquired-at."
        ) from None
    return instant.replace(tzinfo=timezone.utc).isoformat()


def build_measurement(
    *,
    asset_id: str,
    measurement_point_id: str,
    acquired_at: str,
    direction: str,
    sensor_id: Optional[str],
    rpm: Optional[float],
    declared_by: str,
) -> dict[str, Any]:
    """The ``measurement`` object of the companion, in the server's field order."""
    measurement: dict[str, Any] = {
        "asset_id": asset_id,
        "measurement_point_id": measurement_point_id,
        "acquired_at": acquired_at,
    }
    if rpm is not None:
        measurement["rpm"] = rpm
    if sensor_id is not None:
        measurement["sensor_id"] = sensor_id
    measurement["direction"] = direction
    measurement["declared_by"] = declared_by
    return measurement


def build_companion(
    *, sampling_rate: float, measurement: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, Any]:
    """The companion file: raw declaration, identity and provenance."""
    return {
        "sample_format": SAMPLE_FORMAT,
        "byte_order": BYTE_ORDER,
        "n_channels": 1,
        "channel_index": 0,
        "header_offset": 0,
        "sampling_rate": sampling_rate,
        "signal_unit": SIGNAL_UNIT,
        "measurement": measurement,
        "adapter": adapter,
    }


# ---------------------------------------------------------------------------
# DATALOG2: metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorLayout:
    """What ``device_config.json`` declares about one accelerometer component.

    Attributes:
        name: Component name (``iis3dwb_acc``).
        dim: Interleaved axes per sample.
        samples_per_ts: Samples between two timestamps (0: no timestamps).
        sensitivity: g per LSB.
        packet_payload_bytes: Payload bytes per packet on the acquisition link.
        measured_odr_hz: ``measodr``, or ``None`` when the firmware did not
            record it.
        odr_index: The ``odr`` enumeration index (never a frequency).
        unit: Unit reported by the component, when declared.
    """

    name: str
    dim: int
    samples_per_ts: int
    sensitivity: float
    packet_payload_bytes: int
    measured_odr_hz: Optional[float]
    odr_index: Optional[int]
    unit: Optional[str]

    @property
    def sample_bytes(self) -> int:
        """Bytes of one multi-axis sample."""
        return self.dim * SAMPLE_DTYPE.itemsize

    @property
    def frame_data_bytes(self) -> int:
        """Data bytes between two timestamps."""
        return self.samples_per_ts * self.sample_bytes


@dataclass(frozen=True)
class DatalogAcquisition:
    """The declarations of one DATALOG2 folder that the adapter uses."""

    folder: Path
    interface: int
    start_time: Optional[str]
    end_time: Optional[str]
    acquisition_name: Optional[str]
    acquisition_uuid: Optional[str]
    serial_number: Optional[str]
    alias: Optional[str]
    firmware_name: Optional[str]
    firmware_version: Optional[str]
    model: Optional[str]
    sensor: SensorLayout


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterError(
            f"{what} not found: {path}. A DATALOG2 acquisition folder holds "
            f"acquisition_info.json, device_config.json and one <sensor>.dat "
            f"per active sensor."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AdapterError(f"{what} could not be read ({exc}): {path}") from None
    # The firmware may terminate the file with a NUL byte; the SDK strips it.
    text = text.rstrip("\x00")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{what} is not valid JSON ({exc}): {path}") from None
    if not isinstance(data, dict):
        raise AdapterError(f"{what} is not a JSON object: {path}")
    return data


def _positive_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _positive_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _named_component(device: dict[str, Any], key: str) -> dict[str, Any]:
    """The body of the component keyed *key* (``firmware_info``...), or ``{}``."""
    for component in device.get("components") or ():
        if isinstance(component, dict):
            body = component.get(key)
            if isinstance(body, dict):
                return body
    return {}


def find_sensor_component(
    device_config: dict[str, Any], sensor: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(device, component body)`` for *sensor*.

    Raises:
        AdapterError: No device, or no component of that name; the message
            lists the sensor components the configuration declares.
    """
    devices = device_config.get("devices")
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        raise AdapterError("device_config.json declares no device under 'devices'.")
    device = devices[0]
    body = _named_component(device, sensor)
    if body:
        return device, body
    declared = sorted(
        name
        for component in device.get("components") or ()
        if isinstance(component, dict)
        for name, value in component.items()
        if isinstance(value, dict) and "data_type" in value
    )
    raise AdapterError(
        f"Sensor {sensor!r} is not a component of device_config.json. Sensor "
        f"components declared: {declared}. Pass --sensor with one of them."
    )


def sensor_layout(
    component: dict[str, Any], *, name: str, interface: int
) -> SensorLayout:
    """Read the decoding parameters of one sensor component.

    Raises:
        AdapterError: A field that decoding needs is missing or unusable;
            the message names the field.
    """
    if component.get("enable") is False:
        raise AdapterError(
            f"Sensor {name!r}: 'enable' is false in device_config.json, so the "
            f"acquisition holds no data for it. Pass --sensor with an enabled "
            f"accelerometer."
        )
    data_type = component.get("data_type")
    if data_type != SUPPORTED_DATA_TYPE:
        raise AdapterError(
            f"Sensor {name!r}: 'data_type' is {data_type!r}; this adapter decodes "
            f"{SUPPORTED_DATA_TYPE!r} samples only."
        )
    sensitivity = _positive_number(component.get("sensitivity"))
    if sensitivity is None:
        raise AdapterError(
            f"Sensor {name!r}: device_config.json declares no positive "
            f"'sensitivity' (g per LSB), so the counts cannot be scaled to g. "
            f"Refusing rather than assuming a full-scale range."
        )
    dim = _positive_int(component.get("dim"))
    if dim is None:
        raise AdapterError(
            f"Sensor {name!r}: 'dim' (axes per sample) is missing or not a "
            f"positive integer in device_config.json."
        )
    spts_value = component.get("samples_per_ts")
    if isinstance(spts_value, dict):  # older schema: {"val": N, ...}
        spts_value = spts_value.get("val")
    if (
        isinstance(spts_value, bool)
        or not isinstance(spts_value, int)
        or spts_value < 0
    ):
        raise AdapterError(
            f"Sensor {name!r}: 'samples_per_ts' is missing or not a non-negative "
            f"integer in device_config.json."
        )
    packet_key = INTERFACE_PACKET_KEYS.get(interface)
    if packet_key is None:
        raise AdapterError(
            f"acquisition_info.json: 'interface' {interface!r} is not one of "
            f"{sorted(INTERFACE_PACKET_KEYS)} (SD card, USB, BLE, serial), so "
            f"the packet size of the .dat stream is unknown."
        )
    packet_size = _positive_int(component.get(packet_key))
    if packet_size is None:
        raise AdapterError(
            f"Sensor {name!r}: {packet_key!r} (packet size on the "
            f"{INTERFACE_NAMES[interface]} link) is missing or not a positive "
            f"integer in device_config.json."
        )
    payload = packet_size - PACKET_COUNTER_BYTES if interface == 0 else packet_size
    if payload <= 0:
        raise AdapterError(
            f"Sensor {name!r}: {packet_key!r} = {packet_size} leaves no payload "
            f"after the {PACKET_COUNTER_BYTES}-byte packet counter."
        )
    unit: Optional[str] = None
    stream = component.get("st_ble_stream")
    if isinstance(stream, dict):
        for value in stream.values():
            if isinstance(value, dict) and isinstance(value.get("unit"), str):
                unit = value["unit"]
                break
    if unit is not None and unit != SIGNAL_UNIT:
        raise AdapterError(
            f"Sensor {name!r}: device_config.json reports its unit as {unit!r}; "
            f"this adapter declares accelerations in {SIGNAL_UNIT!r} only."
        )
    odr_index = component.get("odr")
    return SensorLayout(
        name=name,
        dim=dim,
        samples_per_ts=spts_value,
        sensitivity=sensitivity,
        packet_payload_bytes=payload,
        measured_odr_hz=_positive_number(component.get("measodr")),
        odr_index=odr_index if isinstance(odr_index, int) else None,
        unit=unit,
    )


def read_datalog_acquisition(folder: Path, *, sensor: str) -> DatalogAcquisition:
    """Read the two JSON files of a DATALOG2 folder for one sensor."""
    if not folder.is_dir():
        raise AdapterError(
            f"{folder} is not a directory. A DATALOG2 input is the acquisition "
            f"folder (YYYYMMDD_HH_MM_SS) written by the firmware; for a recorder "
            f"CSV pass --input-format {INPUT_WINDOW_CSV}."
        )
    info = _load_json_object(folder / "acquisition_info.json", "acquisition_info.json")
    config = _load_json_object(folder / "device_config.json", "device_config.json")
    interface = info.get("interface")
    if isinstance(interface, bool) or not isinstance(interface, int):
        raise AdapterError(
            "acquisition_info.json: 'interface' (0 SD card, 1 USB, 2 BLE, 3 "
            "serial) is missing, so the packet size of the .dat stream is "
            "unknown."
        )
    device, component = find_sensor_component(config, sensor)
    layout = sensor_layout(component, name=sensor, interface=interface)
    firmware = _named_component(device, "firmware_info")
    device_info = _named_component(device, "DeviceInformation")

    def text(source: dict[str, Any], key: str) -> Optional[str]:
        value = source.get(key)
        return value if isinstance(value, str) and value else None

    return DatalogAcquisition(
        folder=folder,
        interface=interface,
        start_time=text(info, "start_time"),
        end_time=text(info, "end_time"),
        acquisition_name=text(info, "name"),
        acquisition_uuid=text(info, "uuid"),
        serial_number=text(device, "sn"),
        alias=text(firmware, "alias"),
        firmware_name=text(firmware, "fw_name"),
        firmware_version=text(firmware, "fw_version"),
        model=text(device_info, "model"),
        sensor=layout,
    )


# ---------------------------------------------------------------------------
# DATALOG2: the .dat decoder (pure functions over bytes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedSensor:
    """Samples recovered from one ``<sensor>.dat`` file.

    Attributes:
        samples: ``(n_samples, dim)`` float32 in g (counts times sensitivity).
        timestamps: One float64 per complete frame, seconds since the
            acquisition started, as written by the firmware.
        packets: Whole packets read.
        complete_frames: Frames that carried their timestamp.
        trailing_samples: Samples recovered from a final block that has no
            timestamp yet (the ST SDK discards that block).
        dropped_tail_bytes: Bytes after the last whole packet (never data).
    """

    samples: np.ndarray
    timestamps: np.ndarray
    packets: int
    complete_frames: int
    trailing_samples: int
    dropped_tail_bytes: int


def strip_packet_counters(
    raw: bytes, payload_bytes: int
) -> tuple[np.ndarray, int, int]:
    """Remove the per-packet counters and check their progression.

    Args:
        raw: The whole ``.dat`` file.
        payload_bytes: Payload bytes per packet (see ``SensorLayout``).

    Returns:
        ``(payload, packets, dropped_tail_bytes)``: the concatenated payloads
        as a uint8 array, the number of whole packets, and the bytes of an
        incomplete trailing packet, which are ignored as the SDK does.

    Raises:
        AdapterError: Less than one packet, or a counter step that differs
            from the payload size (data lost on the link).
    """
    if payload_bytes <= 0:
        raise AdapterError("Packet payload size must be positive.")
    packet_bytes = payload_bytes + PACKET_COUNTER_BYTES
    packets = len(raw) // packet_bytes
    if packets == 0:
        raise AdapterError(
            f"The .dat file holds {len(raw)} bytes, less than one "
            f"{packet_bytes}-byte packet; nothing to decode."
        )
    used = packets * packet_bytes
    table = np.frombuffer(raw, dtype=np.uint8, count=used).reshape(
        packets, packet_bytes
    )
    counters = np.ascontiguousarray(table[:, :PACKET_COUNTER_BYTES]).view("<u4").ravel()
    steps = np.diff(counters.astype(np.int64)) % (1 << 32)
    gaps = np.nonzero(steps != payload_bytes)[0]
    if gaps.size:
        index = int(gaps[0])
        raise AdapterError(
            f"Packet counter gap between packets {index} and {index + 1} "
            f"({int(counters[index])} then {int(counters[index + 1])}, expected "
            f"a step of {payload_bytes}): the acquisition lost data on the "
            f"link, so its samples are not contiguous. Refusing to write a "
            f"measurement from it."
        )
    payload = np.ascontiguousarray(table[:, PACKET_COUNTER_BYTES:]).reshape(-1)
    return payload, packets, len(raw) - used


def split_frames(
    payload: np.ndarray, *, frame_data_bytes: int, sample_bytes: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Separate sample bytes from the timestamps that close every frame.

    Args:
        payload: Concatenated packet payloads (uint8).
        frame_data_bytes: Data bytes per frame (``samples_per_ts * dim * 2``);
            0 means the stream carries no timestamps.
        sample_bytes: Bytes of one multi-axis sample.

    Returns:
        ``(data, timestamps, trailing_samples)``: the sample bytes in order
        (uint8, a whole number of samples), the float64 timestamps of the
        complete frames, and how many samples came from a final block that
        has no timestamp (kept: those bytes are samples, not timestamps).
    """
    if frame_data_bytes == 0:
        whole = (payload.size // sample_bytes) * sample_bytes
        return payload[:whole], np.empty(0, dtype=np.float64), 0
    frame_bytes = frame_data_bytes + TIMESTAMP_BYTES
    frames = payload.size // frame_bytes
    table = payload[: frames * frame_bytes].reshape(frames, frame_bytes)
    data = np.ascontiguousarray(table[:, :frame_data_bytes]).reshape(-1)
    timestamps = np.ascontiguousarray(table[:, frame_data_bytes:]).view("<f8").ravel()
    tail = payload[frames * frame_bytes :][:frame_data_bytes]
    usable = (tail.size // sample_bytes) * sample_bytes
    if usable:
        data = np.concatenate([data, tail[:usable]])
    return data, timestamps, usable // sample_bytes


def timestamp_anomalies(
    timestamps: np.ndarray, frame_period: Optional[float]
) -> list[int]:
    """Indexes of frames whose timestamp step leaves the SDK's tolerance band.

    A NaN timestamp, or a step below 0.1 or above 10 times the expected
    frame period, is what the SDK's timestamp recovery rewrites and zeroes.
    Without a known frame period only NaN values are reported.
    """
    if timestamps.size == 0:
        return []
    flagged = set(int(i) for i in np.nonzero(~np.isfinite(timestamps))[0])
    if frame_period and timestamps.size > 1:
        steps = np.abs(np.diff(timestamps))
        low, high = TIMESTAMP_STEP_BAND
        bad = (steps < low * frame_period) | (steps > high * frame_period)
        flagged.update(int(i) + 1 for i in np.nonzero(bad)[0])
    return sorted(flagged)


def decode_dat(raw: bytes, layout: SensorLayout) -> DecodedSensor:
    """Decode a ``<sensor>.dat`` file into scaled samples.

    Raises:
        AdapterError: Structural problems (see ``strip_packet_counters``) or
            timestamps the SDK would treat as corrupted.
    """
    payload, packets, dropped = strip_packet_counters(raw, layout.packet_payload_bytes)
    data, timestamps, trailing = split_frames(
        payload,
        frame_data_bytes=layout.frame_data_bytes,
        sample_bytes=layout.sample_bytes,
    )
    if data.size == 0:
        raise AdapterError("The .dat file holds no complete sample.")
    period = (
        layout.samples_per_ts / layout.measured_odr_hz
        if layout.measured_odr_hz and layout.samples_per_ts
        else None
    )
    flagged = timestamp_anomalies(timestamps, period)
    if flagged:
        shown = ", ".join(str(i) for i in flagged[:5])
        more = "" if len(flagged) <= 5 else f" and {len(flagged) - 5} more"
        raise AdapterError(
            f"Frame timestamps at index {shown}{more} (of {timestamps.size}) step "
            f"outside {TIMESTAMP_STEP_BAND[0]} to {TIMESTAMP_STEP_BAND[1]} times "
            f"the expected frame period; the ST SDK treats such frames as "
            f"corrupted. Refusing to write a measurement from this file."
        )
    counts = np.ascontiguousarray(data).view(SAMPLE_DTYPE).reshape(-1, layout.dim)
    samples = counts.astype(np.float32) * np.float32(layout.sensitivity)
    return DecodedSensor(
        samples=samples,
        timestamps=timestamps,
        packets=packets,
        complete_frames=int(timestamps.size),
        trailing_samples=trailing,
        dropped_tail_bytes=dropped,
    )


# ---------------------------------------------------------------------------
# Window CSV of the USB bridge recorder
# ---------------------------------------------------------------------------


def read_window_csv(path: Path) -> np.ndarray:
    """Parse a recorder window into a ``(rows, 4)`` float64 table.

    The header must be exactly ``Time [s],A_x [g],A_y [g],A_z [g]``.

    Raises:
        AdapterError: Missing file, another header, no rows, or rows that
            are not four numbers.
    """
    if not path.is_file():
        raise AdapterError(f"{path} is not a file.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header_line = handle.readline()
            rows = [line for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise AdapterError(f"{path.name} could not be read ({exc}).") from None
    header = tuple(cell.strip() for cell in header_line.strip().split(","))
    if header != WINDOW_CSV_HEADER:
        raise AdapterError(
            f"{path.name}: header {header_line.strip()!r} is not the recorder's "
            f"{','.join(WINDOW_CSV_HEADER)!r}."
        )
    if not rows:
        raise AdapterError(f"{path.name}: no sample rows after the header.")
    try:
        table = np.loadtxt(rows, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise AdapterError(f"{path.name}: rows are not four numbers ({exc}).") from None
    if table.ndim != 2 or table.shape[1] != len(WINDOW_CSV_HEADER):
        raise AdapterError(
            f"{path.name}: expected {len(WINDOW_CSV_HEADER)} columns per row."
        )
    if not np.all(np.isfinite(table)):
        raise AdapterError(f"{path.name}: non-finite values in the sample rows.")
    return table


def implied_rate_hz(time_column: np.ndarray) -> Optional[float]:
    """Rate implied by the median step of the ``Time [s]`` column, if any."""
    if time_column.size < 2:
        return None
    step = float(np.median(np.diff(time_column)))
    return 1.0 / step if step > 0 else None


# ---------------------------------------------------------------------------
# Conversion: from an input to planned outputs, then to files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisOutput:
    """One axis of one acquisition, ready to be written."""

    axis: str
    bin_path: Path
    companion_path: Path
    samples: np.ndarray
    companion: dict[str, Any]


@dataclass(frozen=True)
class Conversion:
    """Planned outputs plus the warnings raised while planning them."""

    outputs: list[AxisOutput]
    warnings: list[str]


def _plan_axes(
    *,
    samples: np.ndarray,
    out_dir: Path,
    asset_id: str,
    point_prefix: str,
    acquired_at: str,
    sampling_rate: float,
    sensor_id: Optional[str],
    rpm: Optional[float],
    declared_by: str,
    adapter: dict[str, Any],
) -> list[AxisOutput]:
    if samples.ndim != 2 or samples.shape[1] != len(AXES):
        raise AdapterError(
            f"Expected {len(AXES)} axes per sample, got {samples.shape[-1]}; "
            f"one file per axis is the only layout this adapter writes."
        )
    stamp = output_stamp(parse_instant(acquired_at, source="acquired_at"))
    outputs = []
    for index, axis in enumerate(AXES):
        point_id = validate_ledger_id(
            f"{point_prefix}_{axis}", kind="measurement_point_id"
        )
        measurement = build_measurement(
            asset_id=asset_id,
            measurement_point_id=point_id,
            acquired_at=acquired_at,
            direction=axis,
            sensor_id=sensor_id,
            rpm=rpm,
            declared_by=declared_by,
        )
        provenance = dict(adapter)
        provenance["axis"] = axis
        provenance["axis_index"] = index
        provenance["samples"] = int(samples.shape[0])
        stem = f"{asset_id}_{point_prefix}_{axis}_{stamp}"
        outputs.append(
            AxisOutput(
                axis=axis,
                bin_path=out_dir / f"{stem}.bin",
                companion_path=out_dir / f"{stem}_metadata.json",
                samples=np.ascontiguousarray(samples[:, index], dtype=np.float32),
                companion=build_companion(
                    sampling_rate=sampling_rate,
                    measurement=measurement,
                    adapter=provenance,
                ),
            )
        )
    return outputs


def convert_datalog2(
    folder: Path,
    *,
    out_dir: Path,
    asset_id: str,
    point_prefix: str,
    sensor: str = DEFAULT_SENSOR,
    rpm: Optional[float] = None,
    declared_by: str = DEFAULT_DECLARED_BY,
    sensor_id: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    acquired_at: Optional[str] = None,
    start_time_offset: Optional[str] = None,
    nominal_odr_hz: Optional[float] = None,
) -> Conversion:
    """Plan the outputs of one DATALOG2 folder (nothing is written here).

    Args:
        folder: The acquisition folder.
        out_dir: Where the ``.bin`` and companion files will go.
        asset_id: Declared asset.
        point_prefix: Measurement points are ``<prefix>_x``, ``_y``, ``_z``.
        sensor: Component name; its ``.dat`` file is ``<sensor>.dat``.
        rpm: Shaft speed in rev/min, when known.
        declared_by: Attribution recorded in the companion.
        sensor_id: Overrides the device serial number.
        sampling_rate: Overrides ``measodr``.
        acquired_at: Overrides ``acquisition_info.json::start_time``.
        start_time_offset: ``+HH:MM`` offset the board clock held when it
            stamped ``start_time``; replaces the ``Z`` label (see the module
            docstring). Not combinable with ``acquired_at``.
        nominal_odr_hz: The configured ODR in Hz, recorded as provenance
            only (the firmware stores an enumeration index, not Hz).

    Raises:
        AdapterError: Any refusal, with its remedy.
    """
    acquisition = read_datalog_acquisition(folder, sensor=sensor)
    layout = acquisition.sensor
    warnings: list[str] = []

    if acquired_at is not None:
        if start_time_offset is not None:
            raise AdapterError(
                "Pass either --acquired-at (the instant) or --start-time-offset "
                "(the offset of the board clock), not both."
            )
        acquired_source = "argument"
        parse_instant(acquired_at, source="--acquired-at")
    elif acquisition.start_time is not None:
        instant = parse_instant(
            acquisition.start_time, source="acquisition_info.json::start_time"
        )
        if start_time_offset is not None:
            acquired_source = "acquisition_info.start_time_at_declared_offset"
            offset = parse_utc_offset(start_time_offset, source="--start-time-offset")
            acquired_at = at_declared_offset(instant, offset)
        else:
            acquired_source = "acquisition_info.start_time"
            acquired_at = acquisition.start_time
            warnings.append(
                f"acquisition_info.json::start_time {acquired_at} is copied as "
                f"written. Its Z suffix is a label: when the host starts the "
                f"log, the board clock is set from the host's local time "
                f"without an offset, so the instant may be off by that offset. "
                f"Pass --start-time-offset <+HH:MM> to declare the offset the "
                f"clock held, or --acquired-at to declare the instant."
            )
        if instant.year < RTC_SUSPECT_BEFORE_YEAR:
            warnings.append(
                f"acquisition_info.json::start_time is {acquisition.start_time}: "
                f"the board's real-time clock was probably not set. Pass "
                f"--acquired-at with the true instant if you know it."
            )
    else:
        raise AdapterError(
            "acquisition_info.json declares no 'start_time', so the acquisition "
            "instant is unknown. Pass --acquired-at <ISO 8601 with offset>."
        )

    if sampling_rate is not None:
        rate_source = "argument"
    elif layout.measured_odr_hz is not None:
        rate_source = "device_config.measodr"
        sampling_rate = layout.measured_odr_hz
    else:
        raise AdapterError(
            f"Sensor {sensor!r}: device_config.json has no positive 'measodr' "
            f"(measured output data rate), and 'odr' is an enumeration index, "
            f"not a frequency. Pass --sampling-rate <Hz>."
        )

    dat_path = folder / f"{sensor}.dat"
    if not dat_path.is_file():
        raise AdapterError(
            f"{dat_path.name} not found in {folder}: the sensor was declared but "
            f"its data file is missing."
        )
    try:
        raw = dat_path.read_bytes()
    except OSError as exc:
        raise AdapterError(f"{dat_path} could not be read ({exc}).") from None
    decoded = decode_dat(raw, layout)
    if decoded.dropped_tail_bytes:
        warnings.append(
            f"{dat_path.name}: {decoded.dropped_tail_bytes} trailing bytes do not "
            f"form a whole packet and were ignored (the ST SDK ignores them too)."
        )

    if sensor_id is None:
        sensor_id = acquisition.serial_number or acquisition.alias
    if sensor_id is not None:
        validate_free_text(sensor_id, kind="sensor_id")

    adapter: dict[str, Any] = {
        "name": ADAPTER_NAME,
        "version": ADAPTER_VERSION,
        "input_format": INPUT_DATALOG2,
        "source": folder.name,
        "decoder": DATALOG2_DECODER,
        "sensor": sensor,
        "device": {
            "model": acquisition.model,
            "serial_number": acquisition.serial_number,
        },
        "firmware": {
            "alias": acquisition.alias,
            "name": acquisition.firmware_name,
            "version": acquisition.firmware_version,
        },
        "acquisition": {
            "name": acquisition.acquisition_name,
            "uuid": acquisition.acquisition_uuid,
            "interface": INTERFACE_NAMES.get(acquisition.interface),
            "start_time": acquisition.start_time,
            "end_time": acquisition.end_time,
        },
        "acquired_at_source": acquired_source,
        "start_time_offset_declared": start_time_offset,
        "sampling_rate_source": rate_source,
        "measured_odr_hz": layout.measured_odr_hz,
        "nominal_odr_hz": nominal_odr_hz,
        "nominal_odr_index": layout.odr_index,
        "sensitivity_g_per_lsb": layout.sensitivity,
        "samples_per_timestamp": layout.samples_per_ts,
        "packets": decoded.packets,
        "complete_frames": decoded.complete_frames,
        "trailing_samples": decoded.trailing_samples,
    }
    outputs = _plan_axes(
        samples=decoded.samples,
        out_dir=out_dir,
        asset_id=asset_id,
        point_prefix=point_prefix,
        acquired_at=acquired_at,
        sampling_rate=sampling_rate,
        sensor_id=sensor_id,
        rpm=rpm,
        declared_by=declared_by,
        adapter=adapter,
    )
    return Conversion(outputs=outputs, warnings=warnings)


def convert_window_csv(
    path: Path,
    *,
    out_dir: Path,
    asset_id: str,
    point_prefix: str,
    sampling_rate: Optional[float],
    acquired_at: Optional[str],
    acquired_at_from_filename: bool = False,
    rpm: Optional[float] = None,
    declared_by: str = DEFAULT_DECLARED_BY,
    sensor_id: Optional[str] = None,
) -> Conversion:
    """Plan the outputs of one recorder window (nothing is written here).

    ``sampling_rate`` is required: the CSV declares none. The instant comes
    from ``acquired_at`` (primary) or, on request, from the file name.

    Raises:
        AdapterError: Any refusal, with its remedy.
    """
    warnings: list[str] = []
    if sampling_rate is None:
        raise AdapterError(
            "A recorder window declares no sampling rate. Pass --sampling-rate "
            "<Hz> (the rate the recorder was configured for, or the measured "
            "rate when known)."
        )
    if acquired_at is not None and acquired_at_from_filename:
        raise AdapterError(
            "Pass either --acquired-at or --acquired-at-from-filename, not both."
        )
    if acquired_at is not None:
        acquired_source = "argument"
        parse_instant(acquired_at, source="--acquired-at")
    elif acquired_at_from_filename:
        acquired_source = "filename"
        acquired_at = acquired_at_from_window_filename(path)
    else:
        raise AdapterError(
            "A recorder window declares no acquisition instant. Pass "
            "--acquired-at <ISO 8601 with offset>, or "
            "--acquired-at-from-filename to trust the recorder's "
            "<asset>_<UTC stamp>.csv naming."
        )
    if sensor_id is not None:
        validate_free_text(sensor_id, kind="sensor_id")

    table = read_window_csv(path)
    implied = implied_rate_hz(table[:, 0])
    if implied is not None and abs(implied - sampling_rate) > (
        WINDOW_CSV_RATE_TOLERANCE * sampling_rate
    ):
        warnings.append(
            f"{path.name}: the Time column implies about {implied:.1f} Hz while "
            f"--sampling-rate declares {sampling_rate:g} Hz; the declared value "
            f"is written. Check the recorder configuration."
        )
    adapter: dict[str, Any] = {
        "name": ADAPTER_NAME,
        "version": ADAPTER_VERSION,
        "input_format": INPUT_WINDOW_CSV,
        "source": path.name,
        "decoder": WINDOW_CSV_DECODER,
        "columns": list(WINDOW_CSV_HEADER[1:]),
        "acquired_at_source": acquired_source,
        "sampling_rate_source": "argument",
        "time_column_implied_rate_hz": implied,
        "rows": int(table.shape[0]),
    }
    outputs = _plan_axes(
        samples=table[:, 1:],
        out_dir=out_dir,
        asset_id=asset_id,
        point_prefix=point_prefix,
        acquired_at=acquired_at,
        sampling_rate=sampling_rate,
        sensor_id=sensor_id,
        rpm=rpm,
        declared_by=declared_by,
        adapter=adapter,
    )
    return Conversion(outputs=outputs, warnings=warnings)


def write_outputs(outputs: Sequence[AxisOutput], *, force: bool = False) -> list[Path]:
    """Write every planned file, refusing first if any target exists.

    Returns:
        The written paths, ``.bin`` before companion, axis by axis.

    Raises:
        AdapterError: An existing target without ``force``, or an OS error.
    """
    existing = [
        target
        for output in outputs
        for target in (output.bin_path, output.companion_path)
        if target.exists()
    ]
    if existing and not force:
        listed = ", ".join(str(p) for p in existing)
        raise AdapterError(
            f"Refusing to overwrite existing output: {listed}. Pass --force to "
            f"replace it, or choose another --out directory."
        )
    written: list[Path] = []
    try:
        for output in outputs:
            output.bin_path.parent.mkdir(parents=True, exist_ok=True)
            output.samples.astype("<f4").tofile(output.bin_path)
            written.append(output.bin_path)
            output.companion_path.write_text(
                json.dumps(output.companion, indent=2) + "\n", encoding="utf-8"
            )
            written.append(output.companion_path)
    except OSError as exc:
        raise AdapterError(
            f"Could not write {exc.filename or 'output'}: {exc}"
        ) from None
    return written


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="stwinbox_to_measurement",
        description=(
            "Convert an STWIN.box DATALOG2 acquisition folder, or a window CSV "
            "of its USB bridge recorder, into one float32 .bin per axis plus "
            "the companion metadata the server's load_signal tool reads."
        ),
    )
    parser.add_argument(
        "input",
        help="acquisition folder (datalog2) or CSV file (window-csv)",
    )
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default=INPUT_DATALOG2,
        help=f"input kind (default: {INPUT_DATALOG2})",
    )
    parser.add_argument(
        "--asset-id", required=True, help="asset the measurement belongs to"
    )
    parser.add_argument(
        "--measurement-point-prefix",
        required=True,
        help="measurement points become <prefix>_x, <prefix>_y, <prefix>_z",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory (for example data/signals/<asset>)",
    )
    parser.add_argument(
        "--sensor",
        default=DEFAULT_SENSOR,
        help=f"DATALOG2 component to decode (default: {DEFAULT_SENSOR})",
    )
    parser.add_argument(
        "--rpm", type=_positive_float, default=None, help="shaft speed in rev/min"
    )
    parser.add_argument(
        "--declared-by",
        default=DEFAULT_DECLARED_BY,
        help=f"attribution recorded in the companion (default: {DEFAULT_DECLARED_BY})",
    )
    parser.add_argument(
        "--sensor-id",
        default=None,
        help="sensor identifier (DATALOG2 default: the device serial number)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=_positive_float,
        default=None,
        help="Hz; required for window-csv, overrides measodr for datalog2",
    )
    instant = parser.add_mutually_exclusive_group()
    instant.add_argument(
        "--acquired-at",
        default=None,
        help=(
            "ISO 8601 instant with offset; required for window-csv, overrides "
            "acquisition_info.json start_time for datalog2"
        ),
    )
    instant.add_argument(
        "--acquired-at-from-filename",
        action="store_true",
        help="window-csv only: read the instant from <asset>_<UTC stamp>.csv",
    )
    instant.add_argument(
        "--start-time-offset",
        default=None,
        metavar="+HH:MM",
        help=(
            "datalog2 only: UTC offset the board clock held when it stamped "
            "start_time; replaces the Z suffix (write --start-time-offset=-05:00 "
            "for a negative offset)"
        ),
    )
    parser.add_argument(
        "--nominal-odr-hz",
        type=_positive_float,
        default=None,
        help="datalog2 only: the configured ODR in Hz, recorded as provenance",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing output files"
    )
    return parser


def convert(args: argparse.Namespace) -> Conversion:
    """Validate the arguments and plan the outputs for either input format."""
    asset_id = validate_ledger_id(args.asset_id, kind="asset_id")
    validate_ledger_id(args.measurement_point_prefix, kind="measurement_point_prefix")
    declared_by = validate_free_text(args.declared_by, kind="declared_by")
    source = Path(args.input)
    out_dir = Path(args.out)
    if args.input_format == INPUT_WINDOW_CSV:
        if args.start_time_offset is not None:
            raise AdapterError(
                "--start-time-offset applies to --input-format datalog2 only; a "
                "recorder window declares its instant through --acquired-at."
            )
        return convert_window_csv(
            source,
            out_dir=out_dir,
            asset_id=asset_id,
            point_prefix=args.measurement_point_prefix,
            sampling_rate=args.sampling_rate,
            acquired_at=args.acquired_at,
            acquired_at_from_filename=args.acquired_at_from_filename,
            rpm=args.rpm,
            declared_by=declared_by,
            sensor_id=args.sensor_id,
        )
    if args.acquired_at_from_filename:
        raise AdapterError(
            "--acquired-at-from-filename applies to --input-format window-csv only; "
            "a DATALOG2 folder declares start_time in acquisition_info.json."
        )
    return convert_datalog2(
        source,
        out_dir=out_dir,
        asset_id=asset_id,
        point_prefix=args.measurement_point_prefix,
        sensor=args.sensor,
        rpm=args.rpm,
        declared_by=declared_by,
        sensor_id=args.sensor_id,
        sampling_rate=args.sampling_rate,
        acquired_at=args.acquired_at,
        start_time_offset=args.start_time_offset,
        nominal_odr_hz=args.nominal_odr_hz,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: 0 on success, 1 on a refusal (message on stderr)."""
    args = build_parser().parse_args(argv)
    try:
        conversion = convert(args)
        for message in conversion.warnings:
            print(f"warning: {message}", file=sys.stderr)
        written = write_outputs(conversion.outputs, force=args.force)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
