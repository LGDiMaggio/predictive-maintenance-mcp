# STWIN.box adapter

`stwinbox_to_measurement.py` converts an acquisition of the STEVAL-STWINBX1 board (STWIN.box, FP-SNS-DATALOG2 firmware) into files the server loads as declared measurements: one headerless float32 `.bin` per accelerometer axis, in g, each with the companion `<stem>_metadata.json` that the `load_signal` tool reads. It follows the [Adapter Guide](../../../docs/ADAPTER_GUIDE.md): it runs before the server is involved, imports nothing from it, and needs only the standard library and NumPy.

Two inputs are accepted:

- a DATALOG2 acquisition folder (`YYYYMMDD_HH_MM_SS/`) holding `acquisition_info.json`, `device_config.json` and one `<sensor>.dat` per active sensor, as written by the firmware on the SD card or by the ST host tools over USB;
- a window CSV of the USB bridge recorder of the same board, header `Time [s],A_x [g],A_y [g],A_z [g]`, selected with `--input-format window-csv`.

## Usage

```text
python examples/adapters/stwinbox/stwinbox_to_measurement.py <acquisition folder> \
    --asset-id P-101 --measurement-point-prefix motor_de --out data/signals/P-101 \
    [--rpm 1480] [--start-time-offset +02:00]

python examples/adapters/stwinbox/stwinbox_to_measurement.py <window>.csv \
    --input-format window-csv --asset-id P-101 --measurement-point-prefix motor_de \
    --out data/signals/P-101 --sampling-rate 26667 --acquired-at 2026-06-29T13:27:00Z
```

| Option | Applies to | Meaning |
|--------|------------|---------|
| `--asset-id` | both | The asset the measurement belongs to (ledger id grammar: starts with a letter or digit, then letters, digits, `_`, `-`, `.`). |
| `--measurement-point-prefix` | both | Measurement points become `<prefix>_x`, `<prefix>_y`, `<prefix>_z`. |
| `--out` | both | Output directory, typically under the server's `data/signals/`. |
| `--sensor` | datalog2 | Component to decode; default `iis3dwb_acc`. |
| `--rpm` | both | Shaft speed in rev/min; recorded only when given. |
| `--declared-by` | both | Attribution written in the companion; default `adapter:stwinbox`. |
| `--sensor-id` | both | Overrides the sensor identifier (datalog2 default: the board serial number). |
| `--sampling-rate` | both | Hz. Required for a window CSV; overrides `measodr` for a folder. |
| `--acquired-at` | both | ISO 8601 instant with offset. Required for a window CSV (unless the option below is chosen); overrides `start_time` for a folder. |
| `--acquired-at-from-filename` | window-csv | Reads the instant from the recorder's `<asset>_<YYYYmmdd>_<HHMMSS>_<ffffff>.csv` naming, taken as UTC. Secondary to `--acquired-at`. |
| `--start-time-offset` | datalog2 | UTC offset the board clock held when it stamped `start_time`; replaces the `Z` suffix. Write `--start-time-offset=-05:00` for a negative offset. |
| `--nominal-odr-hz` | datalog2 | The configured output data rate in Hz, recorded as provenance only. |
| `--force` | both | Replace existing output files. |

Outputs are `<asset>_<prefix>_<axis>_<UTC stamp>.bin` and `<asset>_<prefix>_<axis>_<UTC stamp>_metadata.json` for `x`, `y` and `z`, where the stamp is the acquisition instant in UTC. An existing output is never replaced without `--force`. A refusal exits with status 1, prints one message naming the problem and the remedy on stderr, and writes nothing.

## What it reads and what it declares

| Companion field | Source | Notes |
|-----------------|--------|-------|
| `sampling_rate` | `device_config.json`, sensor `measodr` | The rate the firmware measured during the capture. The sensor's `odr` field is a firmware enumeration index (0 for the IIS3DWB), never a frequency, and is never read as one; without `measodr` the adapter refuses unless `--sampling-rate` is given. |
| `signal_unit` | fixed, `g` | The IIS3DWB reports acceleration in g. A component that declares another unit is refused. |
| `sample_format`, `byte_order` | fixed, `float32`, `little` | The adapter writes the scaled samples (counts times `sensitivity`), not the counts. |
| `measurement.acquired_at` | `acquisition_info.json`, `start_time` | Copied as written, see [The acquisition clock](#the-acquisition-clock). `--acquired-at` overrides it; `--start-time-offset` relabels it. A folder without `start_time` is refused unless `--acquired-at` is given. |
| `measurement.sensor_id` | `device_config.json`, device `sn` (firmware `alias` when there is no serial) | `--sensor-id` overrides. |
| `measurement.asset_id`, `measurement_point_id`, `direction`, `declared_by`, `rpm` | command line | One point per axis, `direction` `x`, `y` or `z`. `rpm` is present only when `--rpm` was given. |
| `adapter` (top level) | provenance | Adapter name and version, decoder name, source folder or file name, firmware alias, name and version, device model and serial, measured and nominal ODR, the `odr` index, sensitivity, frame layout, packet and frame counts, and where each declared value came from. The server keeps it under `source_metadata`. |

Decoding a folder needs `sensitivity` (g per LSB), `data_type` (only `int16` is supported), `dim` (axes per sample), `samples_per_ts` (samples between two timestamps) and the packet size of the acquisition link (`usb_dps` over USB, `sd_dps` on the SD card, selected by `interface` in `acquisition_info.json`). A missing or unusable field is a refusal that names the field.

### The `.dat` stream

A `<sensor>.dat` file is a sequence of packets. Each packet opens with a 4-byte little-endian byte counter and carries `usb_dps` bytes of payload (`sd_dps` minus 4 on the SD card). The concatenated payloads form frames of `samples_per_ts` samples, each sample being `dim` interleaved int16 values, followed by one float64 timestamp in seconds. The adapter strips the counters, checks that they advance by exactly one payload per packet (a different step means the link lost data and the file is refused rather than written as a contiguous signal), strips the timestamps, and keeps the samples of a final block that has no timestamp yet. Timestamps that step outside 0.1 to 10 times the expected frame period are refused as well: the ST SDK zeroes such frames, this adapter does not write a measurement from them.

### Window CSV

The recorder writes short windows with the header `Time [s],A_x [g],A_y [g],A_z [g]`. The adapter takes `A_x`, `A_y` and `A_z` as the three axes and ignores the `Time` column, which the server does not need. A window declares neither a sampling rate nor an acquisition instant, so both are declared on the command line: `--sampling-rate` is required, and `--acquired-at` is the primary way to declare the instant. `--acquired-at-from-filename` is the secondary option for recorders that follow the `<asset>_<UTC stamp>.csv` naming; it declares trust in that convention. When the rate implied by the `Time` column differs from the declared one by more than 5 %, the adapter writes the declared value and prints a warning.

## The acquisition clock

`acquisition_info.json` carries `start_time` and `end_time` with a `Z` suffix. That suffix is a label, not a measurement: when the host starts the log, the ST SDK sets the board clock from the host's local wall-clock time without an offset (`HSDLink_v2.start_log` calls `set_rtc_time` with `datetime.now()`), and the firmware writes those digits with `Z`. On host-driven acquisitions the digits are therefore the host's local time. A board that logs to the SD card on its own uses whatever its clock last held; a clock that was never set reports 2000-01-01, and the adapter warns when the year is before 2010.

The adapter never corrects the value on its own. It copies `start_time` as written, prints a warning that says so, and offers two declarations:

- `--start-time-offset +02:00` keeps the digits and replaces the `Z` with the offset the clock actually held (`2026-05-11T11:06:00.000Z` becomes `2026-05-11T11:06:00+02:00`);
- `--acquired-at 2026-05-11T09:06:00Z` declares the instant outright.

Either choice is recorded in the companion (`adapter.acquired_at_source`, `adapter.start_time_offset_declared`). An offset that is wrong by hours changes the order of measurements against sources that declare true UTC, so the checklist below records what the clock held on the device in use.

## Limitations

- One file per axis. The companion is per file stem and cannot map channels to directions, so a three-axis acquisition becomes three measurements on three points.
- Only `int16` components with a positive `sensitivity` are decoded. Microphones, the magnetometer and other components are refused when named with `--sensor`.
- Only acceleration in g. The adapter does not convert units.
- Tags, the `end_time`, and the other sensors of the acquisition are not used.
- The nominal output data rate in Hz is not derived from the `odr` index (that mapping lives in the firmware's device template); pass `--nominal-odr-hz` to record it.
- A packet counter gap or a timestamp anomaly is a refusal, not a repair.

## Verification against the ST SDK

The decoder was compared with `stdatalog-pysdk` (`HSDatalog.get_dataframe`, timestamp recovery enabled) on four acquisitions of the `iis3dwb_acc` component on 2026-09-10: three taken over USB with FP-SNS-DATALOG2 3.2.0 and the SD-card example shipped with the SDK (firmware 2.3.0). On every acquisition the SDK returned exactly the samples of the complete frames (541000, 128000, 262000 and 274000 samples, 1000 per frame), the adapter returned the same samples plus those of the final block without a timestamp (778, 162, 150 and 728 samples), and the largest absolute difference on any sample was 3.5e-7 g, which is float32 resolution at the amplitudes involved. The adapter's frame timestamps matched the SDK's frame boundaries. The comparison is not part of the test suite because the SDK is not a dependency of this repository; `tests/test_examples_stwinbox_adapter.py` covers the same layout on synthetic folders.

## End-to-end flow with the server

1. Convert the first acquisition with the command above, writing under the server's data directory, for example `data/signals/P-101/`. The three companions declare asset `P-101`, points `motor_de_x`, `motor_de_y`, `motor_de_z`, the instant, the sensor and the rate.
2. Declare the measurement point you intend to trend, once, with the `declare_measurement_point` tool: asset `P-101`, point `motor_de_x`, the bearing (catalog id or fault orders), machine group and support type for the ISO block, expected unit `g` and expected direction `x`. The declaration is an event in the asset ledger; a later change is a new declaration, not an edit.
3. Load the first `.bin` of that point with the `load_signal` tool, naming the file. The declaration comes from the companion; the returned measurement block reports whether the measurement was recorded and whether its health snapshot is complete.
4. Restart the server. The ledger lives in `data/ledger/` (or `PMM_LEDGER_DIR`) and survives the restart; the in-memory signal store does not, and does not need to.
5. Convert and load the second acquisition, taken some time later, the same way.
6. Ask for the history of `P-101` with the `get_asset_history` tool: the two measurements of `motor_de_x` appear in `acquired_at` order, each with the context declared by its companion (sensor, direction, rpm when given) and its snapshot. The `assess_asset_change` tool compares the later measurement against the earlier ones with the criteria stated in its response.

The same flow with the window CSV input replaces step 1 with the second command above.

## Real-device checklist

The adapter is verified against the SDK on stored acquisitions (above). The round trip on a connected board is a manual step for the maintainer, recorded here when done; the entries stay "to be recorded" until then.

| Item | Result |
|------|--------|
| Date of the run | to be recorded |
| Firmware (`fw_name`, `fw_version`) and board alias | to be recorded |
| Output data rate: `measodr` against the nominal rate of the `odr` index | to be recorded |
| `start_time`: UTC or host local time, checked against the host clock at the start of the log | to be recorded |
| Two acquisitions taken at a distance in time, converted and loaded, with the server restarted in between: same history from `get_asset_history` | to be recorded |
| Anomalies found (packet counter gaps, timestamp steps, unit or sensitivity surprises) | to be recorded |

Observations from stored acquisitions, FP-SNS-DATALOG2 3.2.0 over USB, IIS3DWB at `odr` index 0, read on 2026-09-10: `measodr` between 26578 Hz and 26592 Hz across acquisitions against a nominal 26667 Hz (about 0.3 % below); `sensitivity` 0.000488 g per LSB; `samples_per_ts` 1000; `usb_dps` 7000; first packet counter 0; the `start_time` digits equal to the host's local wall clock (folder name and file times) in both winter and summer time, which is what the clock-setting path above predicts. The SD-card example from the SDK, logged with a clock that was never set, starts at 2000-01-01.
