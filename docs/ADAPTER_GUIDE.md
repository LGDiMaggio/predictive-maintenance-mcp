# Adapter Guide — Loading Vendor Data

This guide documents the **ingestion boundary** of the Predictive Maintenance MCP Server: what the core will and will not do with vendor data files, and how to write an **adapter** for an acquisition system whose format the server does not read directly.

> **The boundary in one sentence**: no vendor format parsers ship in the core and nothing is inferred from file content or names — translating a vendor's metadata into the server's explicit declaration is the user's (or an external adapter's) job.

---

## Table of Contents

1. [The Principle: Declared, Never Guessed](#the-principle-declared-never-guessed)
2. [Supported Formats](#supported-formats)
3. [The Declaration Parameters](#the-declaration-parameters)
4. [The Companion File](#the-companion-file)
5. [Declaring a Measurement](#declaring-a-measurement)
6. [Worked Example: A Headerless DAQ Recording](#worked-example-a-headerless-daq-recording)
7. [When the Declaration Is Wrong](#when-the-declaration-is-wrong)
8. [What to Contribute](#what-to-contribute)

---

## The Principle: Declared, Never Guessed

A headerless raw binary file carries zero self-description: nothing in the bytes says whether a sample is `float32` or `int16`, little- or big-endian, one interleaved channel or four. Rather than guessing — and silently producing wrong spectra — the server requires the caller to **declare** the decode contract, then validates that declaration against the file before a single sample reaches any analysis tool.

An **adapter** is anything that translates a vendor's metadata — an XML sidecar, a proprietary header, an exported settings file — into that explicit declaration: either keyword parameters on the `load_signal` tool, or a companion JSON file placed next to the signal. There is nothing to register and no plugin API — **the declaration is the integration surface**.

Three consequences follow:

- **Adapters run outside the server.** An adapter can be a ten-line script in any language. It runs before the server is involved and imports nothing from it.
- **The core stays vendor-neutral.** New vendor formats never require changes to the server, and no vendor-specific parsing code has to be reviewed, secured, or maintained in the core.
- **The declaration is validated, never trusted blindly.** A declaration that contradicts the file fails loudly, with the arithmetic shown (see [When the Declaration Is Wrong](#when-the-declaration-is-wrong)).

---

## Supported Formats

The server distinguishes two classes of signal file:

| Class | Extensions | Declaration |
|-------|------------|-------------|
| Self-describing | `.csv`, `.txt`, `.npy`, `.mat`, `.wav`, `.parquet` | Not needed for decoding — and not allowed: declaring raw decode parameters for these is refused as a contradiction of the file's own header |
| Raw headerless | `.bin`, `.raw`, `.dat` | Required — the file cannot be decoded without it |

Self-describing formats still benefit from a declared `sampling_rate` and `signal_unit` (via parameter or companion file): ISO 20816-3 severity verdicts are refused until a unit is declared, because units are never guessed from signal amplitude.

---

## The Declaration Parameters

Raw files load through the standard `load_signal` tool; the declaration is carried by additive keyword parameters. Two of them are **required** for a raw file — a load missing either is refused with one message naming everything missing and both remedies. The rest carry documented defaults.

<!-- adapter-declaration:start -->

| Parameter | Scope | Allowed values | Default |
|-----------|-------|----------------|---------|
| `sample_format` | Raw files — **required** | `float32`, `float64`, `int16`, `int32` | none — must be declared |
| `sampling_rate` | Raw files — **required** (recommended for every format) | positive number, in Hz | none — must be declared |
| `signal_unit` | Any format — required for ISO severity verdicts | `g`, `m/s2` (acceleration), `mm/s`, `m/s` (velocity) | none — severity verdicts are refused until declared |
| `byte_order` | Raw files | `little`, `big` | `little` |
| `n_channels` | Raw files | integer, 1 or more (interleaved channel count) | `1` |
| `channel_index` | Raw files | integer, 0-based, below `n_channels` | `0` |
| `header_offset` | Raw files | integer, bytes to skip before the first sample | `0` |
| `scale_factor` | Raw files | number — multiplier applied after decoding | none (no scaling) |

<!-- adapter-declaration:end -->

Notes:

- **Integer formats decode to raw ADC counts.** There is no implicit normalization. Declare `scale_factor` (the sensor/DAQ calibration multiplier) to convert counts into the declared physical unit — declaring a `signal_unit` on unscaled counts would misrepresent amplitudes to every severity assessment.
- **Batch loads broadcast the declaration.** When `load_signal` receives a list of file paths, the raw parameters apply to every file in the batch, exactly like `sampling_rate` and `signal_unit`; per-file values come from each file's companion metadata.
- **Provenance is recorded.** Stored signals record the six effective decode parameters under `raw_format`, so `get_signal_info(signal_id="...")` can answer "how was this file decoded" after the fact.

---

## The Companion File

Instead of repeating parameters on every call, place a JSON file named after the signal file's stem next to it — for `motor_de_001.raw`, the companion is `motor_de_001_metadata.json`:

```json
{
  "sampling_rate": 25600.0,
  "signal_unit": "g",
  "sample_format": "float32",
  "byte_order": "little",
  "n_channels": 1,
  "channel_index": 0,
  "header_offset": 0,
  "rpm": 1480
}
```

Honored keys are exactly the declaration parameters above (`sampling_rate`, `signal_unit`, `sample_format`, `byte_order`, `n_channels`, `channel_index`, `header_offset` and `scale_factor`) plus the `measurement` object described in [Declaring a Measurement](#declaring-a-measurement). Companion values are validated against the same closed vocabularies as explicit parameters: an invalid value is refused with a message naming the offending value, its companion-file source, and the valid vocabulary.

Any other top-level keys (like `rpm` above, shaft speeds, reference frequencies) are not decode parameters and are never read as measurement fields: they are preserved verbatim under `source_metadata` and exposed by `get_signal_info(signal_id="...")`. The speed that enters an asset's history is the `rpm` inside the `measurement` object, not a top-level key.

### Merge precedence

When a parameter is available from more than one place, the effective value is resolved in this order:

| Precedence | Source | Notes |
|------------|--------|-------|
| 1 — wins | Explicit `load_signal` parameter | Overrides everything |
| 2 | Companion `<stem>_metadata.json` field | Validated against the same vocabularies as explicit parameters |
| 3 | Documented default | Optional parameters only — `sample_format` and `sampling_rate` have no default |

If `sample_format` or `sampling_rate` is still missing after the merge, the raw load is refused with a single message naming everything missing and both remedies (the explicit re-call and the companion-file alternative).

---

## Declaring a Measurement

The parameters above say how to decode a file. A `measurement` object in the same companion says whose measurement it is: which asset, which measurement point, when it was acquired and under which conditions. It is the input of the asset health ledger, the local append-only history that `get_asset_history` and `assess_asset_change` read and that `declare_measurement_point` and `declare_healthy_baseline` extend.

```json
{
  "sampling_rate": 26585.3,
  "signal_unit": "g",
  "sample_format": "float32",
  "byte_order": "little",
  "measurement": {
    "asset_id": "P-101",
    "measurement_point_id": "motor_de_h",
    "acquired_at": "2026-08-20T13:42:00+02:00",
    "rpm": 1482,
    "load": 75,
    "operating_state": "loaded",
    "sensor_id": "STWIN_BOX_001",
    "direction": "horizontal",
    "declared_by": "adapter:stwinbox"
  }
}
```

The presence of the object activates the contract. A valid object records the measurement in the ledger when the file loads. An object with a missing required field, an unknown field or a value outside its rule refuses the load with one message that names the companion file, every problem and the way out (fix the object, or remove it to load the file without asset identity). A companion without the object behaves exactly as before. A companion that is not valid JSON, or not a JSON object, does not block the load either: the file loads as if it had no companion, and the returned `companion_warning` names the file and the case.

<!-- measurement-declaration:start -->

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `asset_id` | yes | string | Ledger id: starts with a letter or digit; then letters, digits, `_`, `-` and `.`; no trailing dot or space; at most 100 characters; not a Windows reserved device name (CON, PRN, AUX, NUL, COM1 to COM9, LPT1 to LPT9, with or without an extension). Letter case is preserved and significant |
| `measurement_point_id` | yes | string | Same grammar as `asset_id`, at most 100 characters. One point per axis for multi-axis captures |
| `acquired_at` | yes | string | ISO 8601 instant of the acquisition; an offset or `Z` is recommended. Stored normalized to UTC. A naive timestamp is accepted, ordered as UTC and flagged `timezone_declared: false`; an instant in 1970 or earlier, or more than a day in the future, is flagged `timestamp_suspect` |
| `rpm` | no | number | Shaft speed in revolutions per minute, positive. Not the shaft frequency in hertz |
| `load` | no | number | Load in the producer's own convention (percent, kW, ...), any finite number. A load more than 5 percent from the reference median qualifies the measurement |
| `operating_state` | no | string | Free text, at most 200 characters, one line, never interpreted (for example `loaded`, `idle`) |
| `sensor_id` | no | string | Free text, at most 200 characters, one line. A sensor that differs from the one the point expects, or from the reference, qualifies the measurement |
| `direction` | no | string | One of `horizontal`, `vertical`, `axial` (machine directions) or `x`, `y`, `z` (sensor axes); the aliases `h`, `v`, `a` and any letter case are accepted. `x` and `horizontal` are different declarations and are never mapped onto each other |
| `declared_by` | no | string | Attribution identifier of who or what declared the measurement (a name, a role, an adapter tag); free text, at most 200 characters, one line. Written to the ledger |

<!-- measurement-declaration:end -->

Rules that follow from the table:

- **Ids become file names.** The ledger of an asset is `data/ledger/<asset_id>.jsonl`, which is why the grammar is stricter than a plain path component. An id that differs from an existing one only by letter case is refused on every operating system.
- **Only the listed fields are accepted inside the object.** A misspelled field is refused, not dropped. Free-form keys stay at the top level of the companion, where they pass verbatim into `source_metadata`; historical top-level keys such as `shaft_speed` (Hz) or `rpm` are never read as measurement fields.
- **`rpm` is revolutions per minute, not shaft frequency in hertz.** A speed declared in Hz would be read as a machine sixty times slower, and every expected fault frequency would be wrong. Declare `rpm: 1482`, not `rpm: 24.7`.
- **One measurement point per axis.** The companion belongs to one file stem, so it cannot assign different points or directions to the channels of one interleaved raw file. Write one file per axis, each with its own companion naming its own point (`motor_de_x`, `motor_de_y`, `motor_de_z`) and its own `direction`.
- **Timestamps carry the offset.** Two exports of the same acquisition with different offsets order and collapse as one instant. A naive timestamp is accepted, and every assessment reports it as a qualification (`timezone_not_declared`).
- **The declared context also feeds `diagnose_vibration`.** When a call passes no `rpm`, `bearing_id`, `machine_group` or `support_type`, the tool takes the measurement's declared `rpm`, then the point's declaration made with `declare_measurement_point`; an explicit argument always wins, and the result names where each value came from.

### Missing or contradictory context

After the load, and again at every assessment, the measurement is graded against the current declaration of its point and against the reference measurements it will be trended with. The rule: **a contradiction excludes, an absence qualifies.**

| Grade | When | Effect |
|-------|------|--------|
| `non_comparable` | The unit belongs to another family than the point's (acceleration against velocity) or no unit is declared; the `direction` differs from the point's `expected_direction` or from the reference; the speed is more than 10 percent from the anchor (the point's `nominal_rpm`, else the median of the reference); the amplitudes are not physical (integer samples without `scale_factor`, normalized integer WAV) | Recorded, excluded from the trend, listed with the reason |
| `qualified` | `direction`, `rpm` or the timezone is not declared; the `sensor_id` differs from the reference; the speed is between 5 and 10 percent from the anchor; `load` (beyond 5 percent) or `operating_state` differ from the reference; the sampling rate (beyond 1 percent) or the `scale_factor` differs; the measurement predates a change of the point's declaration | In the trend, with the qualification attached to every result |
| `comparable` | Everything declared and consistent; a same-family unit difference (`g` against `m/s2`) is converted and noted | In the trend |

The grade is reported by `load_signal` (against the point alone, in the returned `measurement` block), by `get_asset_history` (against the point, per measurement) and by `assess_asset_change` (against point and reference). It is computed at query time against the current declarations and never stored as a fact.

### What the ledger persists

The ledger lives under `data/ledger/` (or `PMM_LEDGER_DIR`): one JSON Lines file per asset, `<asset_id>.jsonl`, a `_measurements.jsonl` index that maps each measurement id to the asset it was recorded under, and one `.jsonl.lock` sidecar per file. Files are only ever appended to.

What is written:

- **Declarations**: the validated `measurement` fields, `sampling_rate`, `signal_unit`, the effective `raw_format` and the channel index, the `signal_id`, and a file reference (location relative to the data directory, or absolute with a flag; SHA-256 of the bytes; size). The free-form top-level keys of the companion (`source_metadata`) are never written to the ledger.
- **Snapshots**: the indicators derived at load time (time-domain indicators, 1x amplitude, envelope amplitude at each expected bearing fault frequency, ISO 20816-3 severity) with their processing lineage, and a `{reason, remedy}` entry for every block the declared context could not support.
- **Point declarations** (versioned per point) **and baselines**, each with `declared_by` and the note.

What is never written: the waveform. Consequences:

- **Raw files remain the source.** Re-processing after a change of the snapshot algorithm (`assess_asset_change(..., reprocess=True)`) re-reads the file after verifying its hash against the ledger. A history whose files are gone keeps its old snapshots but cannot be re-processed.
- **A moved file, re-loaded, records its new location.** The re-load is reported as `superseded` with `changed: ["location"]`; no new snapshot is computed, and every location ever declared stays in the record.
- **The same file re-loaded is recognized** (`already_recorded`, nothing appended). A corrected declaration (an rpm fixed in the companion) is a new version of the measurement, `superseded` with the changed keys named, and yields a new snapshot; the old one stays.
- **A measurement recorded under the wrong asset** (a mistyped `asset_id`) is corrected by fixing the companion and re-loading the file: it is recorded under the correct asset, and the ledger of the previous asset receives a superseding declaration that lists it under `reattributed`. Nothing is deleted.

### Privacy and operations

- `declared_by` is an attribution identifier. It is written to the ledger and quoted verbatim in later assessments; choose a name, a role or an adapter tag accordingly.
- The ledger is plain text (JSON Lines, ASCII-escaped) under `data/ledger/`: readable, diffable, copyable. A backup is a copy of the directory.
- Deletion is per asset: remove `<asset_id>.jsonl` (and its `.jsonl.lock` sidecar) while the server is stopped. Events inside a file are never removed individually; the `_measurements.jsonl` index is append-only as well.
- Keep `PMM_LEDGER_DIR` on a local, non-synced path. The server warns at startup when the directory is not writable or lies under OneDrive, Dropbox or iCloud Drive, where a sync client can lock the append-only files or replace them with stale copies. `PDM_PROJECT_DIR` keeps the relative file locations valid across restarts.
- Every ledger tool reads the whole file of the asset at each call. The documented threshold beyond which an index becomes necessary is 5,000 events or 20 MB per ledger; `get_asset_history` reports `event_count` and `ledger_bytes` so the threshold is observable.

### The integration model

A file plus its companion is the integration path for any acquisition system, and the only one the core exposes: an adapter converts the device's output into a file the server reads and a companion that declares the decode parameters and the `measurement` object. It runs before the server is involved and imports nothing from it, exactly like every adapter described in this guide. The STWIN.box adapter in [`examples/adapters/stwinbox/`](../examples/adapters/stwinbox/README.md) is the reference implementation of this path: it converts FP-SNS-DATALOG2 acquisition folders and USB window CSVs of the STEVAL-STWINBX1 board into one float32 `.bin` per axis, in g, each with a companion that declares the decode parameters and the measurement.

---

## Worked Example: A Headerless DAQ Recording

Scenario: an industrial DAQ unit writes headerless raw files — `float32`, little-endian, single channel, sampled at 25,600 Hz from an accelerometer calibrated in g. A recording lands as `data/signals/motor_de_001.raw`.

### The adapter's output

An adapter for this DAQ reads the unit's own metadata (settings export, sidecar, or fixed configuration) and emits one companion file per recording — `motor_de_001_metadata.json` next to the signal:

```json
{
  "sampling_rate": 25600.0,
  "signal_unit": "g",
  "sample_format": "float32",
  "byte_order": "little"
}
```

`byte_order` could be omitted (it is the documented default), but an adapter should emit it anyway — an explicit companion file is self-explanatory to whoever opens the folder later.

### Loading — natural language

With the companion file in place, the ask to the assistant needs no technical parameters:

> "Load motor_de_001.raw, run an envelope analysis, and show me the bearing fault evidence."

The assistant calls `load_signal(filepath="motor_de_001.raw")` and the full declaration comes from the companion file.

### Loading — direct call

The equivalent explicit call, with no companion file involved:

```python
load_signal(
    filepath="motor_de_001.raw",
    sample_format="float32",
    sampling_rate=25600.0,
    signal_unit="g",
)
```

### Multi-channel variant

If the DAQ writes four interleaved channels into one file, the companion declares the layout:

```json
{
  "sampling_rate": 25600.0,
  "signal_unit": "g",
  "sample_format": "float32",
  "n_channels": 4
}
```

Each load extracts **one** channel:

```python
load_signal(filepath="motor_de_001.raw", channel_index=2)
```

When the effective `n_channels` is greater than 1, the derived signal id gains a `_ch<k>` suffix — here `motor_de_001_ch2` — so channels of the same file never collide. An explicit `signal_id` is used verbatim, with no suffix applied.

### Integer formats and calibration

A DAQ that stores 16-bit ADC counts needs a declared calibration multiplier to yield physical units:

```python
load_signal(
    filepath="motor_nde_001.raw",
    sample_format="int16",
    sampling_rate=25600.0,
    scale_factor=0.000488,
    signal_unit="g",
)
```

Here `scale_factor` is the counts-to-g conversion from the sensor/DAQ datasheet. Without it, the samples stay raw counts and a declared unit would be a lie.

---

## When the Declaration Is Wrong

A wrong declaration does not silently produce wrong analysis — the loader validates the declared shape against the actual file size before decoding, and refusal messages show the arithmetic. For example, a 6,144,002-byte file declared as single-channel `float32` (4-byte samples) is not a whole number of 4-byte frames — 2 bytes remain — so the load is refused, and the error shows exactly that arithmetic: file size minus `header_offset`, the frame size (sample size times channel count), and the remainder. That remainder is the best available detector of a wrong sample format, a wrong channel count, or a forgotten header.

Other loud failures:

- A float payload that decodes to NaN/Inf samples is refused as a likely `byte_order` or `sample_format` mismatch.
- A file larger than the `PMM_MAX_SIGNAL_SIZE` cap (bytes, default 500 MB) is refused before a single byte is read, with the environment-variable remedy named in the message.
- Declaring raw parameters for a self-describing format is refused as a contradiction — the declared-never-guessed policy cuts both ways.

**What validation cannot catch**: a headerless file gives the loader nothing to check `sampling_rate` against — a wrong rate rescales every frequency in every downstream analysis. The same holds for `signal_unit` and `scale_factor`, which set the physical meaning of amplitudes. Getting those three right from the vendor's metadata is precisely an adapter's most valuable job.

---

## What to Contribute

Three kinds of contribution keep this boundary useful without moving it:

1. **An adapter script for a vendor format** — a standalone script (any language) that reads the vendor's sidecar, header, or settings export and emits companion `<stem>_metadata.json` files next to the signal files. It runs before the server is involved and imports nothing from it.
2. **A worked format mapping** — documentation of which fields in a vendor's metadata map to which declaration keys, with an anonymized sample layout.
3. **Improvements to this guide** — corrections, clearer examples, additional edge cases.

The STWIN.box adapter in [`examples/adapters/stwinbox/`](../examples/adapters/stwinbox/README.md) shows the shape of the first kind, including the `measurement` object that feeds the asset history.

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for the general workflow, then [open an issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) describing the format (byte layout, where the metadata lives, a sample declaration) — or [start a discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) if the approach is still open.
