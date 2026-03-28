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
MAT (MATLAB), and Parquet formats.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Core Operations

### List Available Signals

**On disk**: Call `list_signals()` to browse files in the data/ directory.

**In memory**: Call `list_stored_signals()` to see signals currently loaded in
the cache with their metadata (sampling rate, duration, sample count).

### Load a Signal

Call `load_signal(file_path=..., signal_id=..., sampling_rate=...)`.

- **file_path**: path to the signal file (CSV, TXT, NPY, WAV, MAT, Parquet)
- **signal_id**: a short identifier for referencing the signal later (e.g.,
  "bearing_normal_01")
- **sampling_rate**: required for formats that don't embed it (CSV, TXT, NPY).
  WAV files include the sampling rate automatically.

If the user provides a file path without specifying a signal_id, derive one from
the filename (e.g., "bearing_data.csv" -> "bearing_data").

### Inspect Signal Metadata

Call `get_signal_info(signal_id=...)` to get metadata without loading the full
array: sampling rate, duration, sample count, min/max values.

### Generate Test Signals

Call `generate_test_signal(...)` to create synthetic signals for testing.

Parameters:
- **fault_type**: "normal", "outer_race", "inner_race", "ball_defect",
  "cage_fault", "gear_mesh", "unbalance", "misalignment"
- **duration**: signal duration in seconds (default 1.0)
- **sampling_rate**: Hz (default 10000)
- **shaft_rpm**: rotational speed (default 1800)
- **snr_db**: signal-to-noise ratio in dB (default 20)
- **signal_id**: identifier for the generated signal

This is useful for:
- Demonstrating the diagnostic workflow without real sensor data
- Testing skills and reports with known fault signatures
- Training users on vibration analysis

### Clear Signals

- `clear_signal(signal_id=...)` — remove a specific signal from cache
- `clear_all_signals()` — clear all cached signals

### Visualize a Signal

Call `plot_signal(signal_id=...)` to generate an HTML time-domain plot of the
raw waveform.

## Workflow for New Users

1. Call `list_signals()` to see what data files are available
2. Load one with `load_signal(file_path=..., signal_id=...)`
3. Call `get_signal_info(signal_id=...)` to verify it loaded correctly
4. Call `plot_signal(signal_id=...)` for a quick visual check
5. Proceed with analysis (quick-screening, bearing-diagnosis, etc.)

If no data files are available, generate a test signal:
1. Call `generate_test_signal(fault_type="outer_race", signal_id="test_outer")`
2. Proceed with analysis on "test_outer"

## Important Notes

- Signals are cached in memory; they persist for the session but not across
  server restarts
- Large signals (>100 MB) may take a few seconds to load
- CSV files must have a single numeric column or the first numeric column is used
- Always confirm sampling_rate for formats that don't embed it
- All signal data stays local — nothing is transmitted externally
