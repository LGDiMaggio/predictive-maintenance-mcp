# Tool Catalog

The complete endpoint reference for the Predictive Maintenance MCP server: every registered MCP tool and guided prompt, grouped by category.

**Total: 37 MCP endpoints — 34 tools and 3 prompts.**

## Signal Lifecycle (5)

| Tool | Description |
|------|-------------|
| `load_signal` | Load vibration file(s) (CSV, WAV, MAT, NPY, Parquet, raw binary `.bin`/`.raw`/`.dat` with declared decode metadata) with declared sampling rate and unit — returns the `signal_id` handle |
| `list_signals` | Browse signal files on disk (`scope="disk"`) or loaded signals in memory (`scope="memory"`) |
| `get_signal_info` | Signal metadata (sampling rate, duration, declared unit, source metadata) |
| `generate_test_signal` | Create a synthetic signal, auto-registered and immediately analyzable |
| `clear_signals` | Remove one signal or the whole in-memory cache |

## Spectral & Statistical Analysis (6)

| Tool | Description |
|------|-------------|
| `analyze_fft` | Frequency spectrum with automatic peak detection |
| `analyze_envelope` | Envelope analysis for bearing fault detection (default band 500–5000 Hz) |
| `analyze_statistics` | Time-domain features (RMS, kurtosis, crest factor) |
| `extract_features_from_signal` | Segmented statistical feature extraction |
| `compute_power_spectral_density` | Power spectral density (Welch method) |
| `compute_spectrogram_stft` | Time-frequency spectrogram |

## Diagnostics & Health Assessment (7)

| Tool | Description |
|------|-------------|
| `assess_severity` | Unified ISO 20816-3 severity assessment (signal or direct RMS reading, custom thresholds) — requires a declared signal unit, never guesses |
| `check_bearing_faults` | Unified fault-frequency matching (catalog bearing, explicit frequencies, or explicit geometry) |
| `diagnose_vibration` | Integrated evidence-based diagnosis pipeline (one call) |
| `calculate_bearing_characteristic_frequencies` | Expected fault frequencies from bearing geometry |
| `search_bearing_catalog` | Look up verified, source-traced bearing geometry |
| `train_anomaly_model` | Train novelty detection on healthy baselines |
| `predict_anomalies` | Score a signal against a trained model (bounded output) |

## Documentation (4)

| Tool | Description |
|------|-------------|
| `search_documentation` | Semantic search over equipment manuals |
| `read_manual_excerpt` | Read pages from a manual |
| `extract_manual_specs` | Extract structured specs from PDFs |
| `list_machine_manuals` | Browse available documentation |

## Reporting (9)

| Tool | Description |
|------|-------------|
| `plot_signal` | Interactive time-domain plot |
| `generate_fft_report` | Interactive frequency analysis report |
| `generate_envelope_report` | Envelope analysis with fault markers |
| `generate_iso_report` | Severity zone visualization |
| `generate_diagnostic_report` | Integrated diagnostic report, HTML and PDF, wording authored by the server |
| `generate_diagnostic_report_docx` | Structured Word document report |
| `generate_pca_visualization_report` | PCA anomaly projection |
| `generate_feature_comparison_report` | Cross-signal feature comparison |
| `list_html_reports` | Report management (list all or inspect one) |

## Prognostics (2)

| Tool | Description |
|------|-------------|
| `analyze_signal_trend` | Within-recording screening: feature trend + degradation onset in one call |
| `estimate_rul` | Remaining Useful Life from repeated measurements over time (linear, exponential, Kalman) — refuses single-recording extrapolation |

## Decision Support (1)

| Tool | Description |
|------|-------------|
| `generate_maintenance_recommendations` | Maintenance recommendations from severity zone + canonical fault types |

## Guided Workflows (3 prompts)

| Prompt | Description |
|--------|-------------|
| `diagnose_bearing` | Complete bearing fault diagnostic decision tree |
| `diagnose_gear` | Gear fault detection workflow |
| `quick_diagnostic_report` | Fast health screening |

---

Back to the [README](../README.md) · developer setup and extension guide: [Developer's Quickstart](QUICKSTART_DEVELOPER.md)
