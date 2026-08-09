---
description: >
  Signal loading, generation, and cache management using the
  predictive-maintenance-mcp server. Use this skill when the user says "load
  signal", "import signal", "list signals", "available signals", "generate test
  signal", "create test data", "synthetic signal", "clear cache", "signal info",
  "what signals are loaded", "show signals", "signal formats", or needs help
  managing vibration signal files and the in-memory signal repository.
---

# Signal Management

Load, generate, inspect, and manage vibration signals in the
predictive-maintenance-mcp in-memory repository. Supports CSV, TXT, NPY, WAV,
MAT (MATLAB), and Parquet formats, plus headerless raw binary (`.bin`, `.raw`,
`.dat`) with a declared decode contract. The `signal_id` returned by loading is
the single handle every analysis, diagnosis, report, and prognostics tool
accepts.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Core Operations

### List Available Signals

- **On disk**: `list_signals(scope="disk")` — files under data/signals/ that
  load_signal can open (this is also the default scope).
- **In memory**: `list_signals(scope="memory")` — signals currently loaded,
  with their signal_id, sampling rate, and declared unit.

### Load a Signal

Call `load_signal(filepath="real_train/baseline_1.csv", signal_unit="g")`.

- **filepath**: path relative to data/signals/ or an absolute path
- **signal_id** (optional): custom handle. Default: the relative path with
  separators replaced by underscores — `real_train/baseline_1.csv` becomes
  `real_train_baseline_1`, so same-named files in different folders never
  collide silently
- **sampling_rate** (optional): Hz — overrides the companion metadata file.
  Required if no `_metadata.json` declares it (ask the user; never guess)
- **signal_unit** (optional): `"g"`, `"m/s2"`, `"mm/s"`, or `"m/s"` — declare
  it if known. ISO severity verdicts are REFUSED for signals without a
  declared unit; units are never guessed from amplitude
- **overwrite**: re-loading a path whose signal_id already exists is an
  explicit error unless `overwrite=True`

**Batch loading** (e.g. for model training) — pass a list; the batch is
atomic and fail-fast (one error names the bad entries, nothing is loaded):

Call `load_signal(filepath=["real_train/baseline_1.csv", "real_train/baseline_2.csv"], signal_unit="g")`

### Load a Raw Binary Signal (.bin / .raw / .dat)

A headerless raw file carries no self-description, so the decode contract
must be declared. **Required**: `sample_format` (`"float32"`, `"float64"`,
`"int16"`, or `"int32"`) AND `sampling_rate` — either as explicit parameters
or declared in a companion `<stem>_metadata.json` next to the file (explicit
parameters take precedence). Optional: `byte_order` (default `"little"`),
`n_channels` (default 1), `channel_index`, `header_offset` (bytes),
`scale_factor`.

Call `load_signal(filepath="motor.bin", sample_format="float32", sampling_rate=25600, signal_unit="mm/s")`.

- All declared values must come from the user's actual acquisition setup
  (DAQ configuration, sensor datasheet) — ask the user for THEIR real
  values; the server never guesses them from the file content or name
- Integer formats (`int16`/`int32`) decode to raw ADC counts — declare
  `scale_factor` (the sensor chain's calibration factor) to convert counts
  into the declared physical unit; without it the values stay raw counts
- A load missing a required declaration is refused with one message naming
  everything missing and both remedies (re-call with parameters, or create
  the companion metadata file)

### Inspect Signal Metadata

Call `get_signal_info(signal_id="real_train_baseline_1")` — sampling rate,
duration, sample count, declared unit, and the full companion metadata
(source_metadata: rpm, reference frequencies, ...) without loading the array
into the conversation.

### Generate Test Signals

Call `generate_test_signal(signal_type="bearing_fault", duration=10.0, sampling_rate=10000, random_seed=42)`.

Parameters:
- **signal_type**: `"bearing_fault"` (10 Hz impacts on a 1 kHz carrier),
  `"gear_fault"` (200 Hz mesh tone + harmonics), `"imbalance"` (25 Hz tone,
  1500 RPM), or `"normal"` (broadband noise)
- **duration**: seconds (default 10.0)
- **sampling_rate**: Hz (default 10000)
- **noise_level**: additive white-noise amplitude (default 0.1)
- **random_seed**: set for reproducible signals

The tool writes a timestamped CSV plus companion metadata (sampling rate and
unit "g"), auto-registers it, and returns the StoredSignalInfo — the returned
signal_id is immediately analyzable and ISO-assessable with no manual steps.

This is useful for:
- Demonstrating the diagnostic workflow without real sensor data
- Testing skills and reports with known fault signatures
- Training users on vibration analysis

### Clear Signals

- `clear_signals(signal_id="real_train_baseline_1")` — remove one signal
- `clear_signals()` — clear the whole in-memory cache

### Visualize a Signal

Call `plot_signal(signal_id="real_train_baseline_1")` to generate an
interactive HTML time-domain plot of the raw waveform.

## Workflow for New Users

1. `list_signals(scope="disk")` to see available data files
2. `load_signal(filepath="<file>", signal_unit="g")` — declare the unit if known
3. `get_signal_info(signal_id="<id>")` to verify it loaded correctly
4. `plot_signal(signal_id="<id>")` for a quick visual check
5. Proceed with analysis (quick-screening, bearing-diagnosis, ...)

If no data files are available, generate a test signal:
1. `generate_test_signal(signal_type="bearing_fault", random_seed=42)`
2. Analyze the signal_id returned in the result

## Important Notes

- Signals are cached in memory; they persist for the session but not across
  server restarts (re-load after a restart)
- CSV files: the first numeric column is used
- Raw binary files decode exactly as declared (`get_signal_info` reports the
  effective decode parameters under `raw_format`); a wrong declaration is
  usually caught by the divisibility or non-finite-payload checks, but the
  fix is always the user's real acquisition values, never a guess
- Always confirm sampling_rate and signal_unit for formats that do not embed
  them — ask the user, never guess
- All signal data stays local — nothing is transmitted externally
- Signal management prepares data for analysis that supports — and never
  replaces — the engineer's judgment
