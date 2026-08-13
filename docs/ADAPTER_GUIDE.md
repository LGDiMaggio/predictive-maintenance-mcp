# Adapter Guide — Loading Vendor Data

This guide documents the **ingestion boundary** of the Predictive Maintenance MCP Server: what the core will and will not do with vendor data files, and how to write an **adapter** for an acquisition system whose format the server does not read directly.

> **The boundary in one sentence**: no vendor format parsers ship in the core and nothing is inferred from file content or names — translating a vendor's metadata into the server's explicit declaration is the user's (or an external adapter's) job.

---

## Table of Contents

1. [The Principle: Declared, Never Guessed](#the-principle-declared-never-guessed)
2. [Supported Formats](#supported-formats)
3. [The Declaration Parameters](#the-declaration-parameters)
4. [The Companion File](#the-companion-file)
5. [Worked Example: A Headerless DAQ Recording](#worked-example-a-headerless-daq-recording)
6. [When the Declaration Is Wrong](#when-the-declaration-is-wrong)
7. [What to Contribute](#what-to-contribute)

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

Honored keys are exactly the declaration parameters above: `sampling_rate`, `signal_unit`, `sample_format`, `byte_order`, `n_channels`, `channel_index`, `header_offset`, and `scale_factor`. Companion values are validated against the same closed vocabularies as explicit parameters — an invalid value is refused with a message naming the offending value, its companion-file source, and the valid vocabulary.

Any other keys (like `rpm` above, shaft speeds, reference frequencies) are not decode parameters: they are preserved verbatim under `source_metadata` and exposed by `get_signal_info(signal_id="...")`.

### Merge precedence

When a parameter is available from more than one place, the effective value is resolved in this order:

| Precedence | Source | Notes |
|------------|--------|-------|
| 1 — wins | Explicit `load_signal` parameter | Overrides everything |
| 2 | Companion `<stem>_metadata.json` field | Validated against the same vocabularies as explicit parameters |
| 3 | Documented default | Optional parameters only — `sample_format` and `sampling_rate` have no default |

If `sample_format` or `sampling_rate` is still missing after the merge, the raw load is refused with a single message naming everything missing and both remedies (the explicit re-call and the companion-file alternative).

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

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for the general workflow, then [open an issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) describing the format (byte layout, where the metadata lives, a sample declaration) — or [start a discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) if the approach is still open.
