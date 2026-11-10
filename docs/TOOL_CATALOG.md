# Tool Catalog

The complete endpoint reference for the Predictive Maintenance MCP server: every registered MCP tool and guided prompt, grouped by category.

**Total on the main branch: 48 MCP endpoints, 44 tools and 4 prompts.** Tools listed under Unreleased in the [CHANGELOG](../CHANGELOG.md) are not yet in a published PyPI release; their rows below say so.

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

## Diagnostics & Health Assessment (8)

| Tool | Description |
|------|-------------|
| `assess_severity` | Unified ISO 20816-3 severity assessment (signal or direct RMS reading, custom thresholds) — requires a declared signal unit, never guesses |
| `check_bearing_faults` | Unified fault-frequency matching (catalog bearing, explicit frequencies, or explicit geometry); optional experimental `corroboration_rule` |
| `diagnose_vibration` | Integrated evidence-based diagnosis pipeline (one call) |
| `calculate_bearing_characteristic_frequencies` | Expected fault frequencies from bearing geometry |
| `search_bearing_catalog` | Look up verified, source-traced bearing geometry |
| `train_anomaly_model` | Train novelty detection on healthy baselines |
| `predict_anomalies` | Score a signal against a trained model (bounded output) |
| `assess_shaft_fault_evidence` | Experimental evidence assessment for unbalance, misalignment and looseness on one to six channels: significance-tested lines at integer and half shaft orders, a severity context and, per fault family, whether the pattern this assessment uses is observed, matching and conflicting observations, missing measurements and alternatives. Attributes no fault; never a diagnosis. Summary or full detail. Unreleased: main branch only |

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

## Prognostics (3)

| Tool | Description |
|------|-------------|
| `analyze_signal_trend` | Within-recording screening: feature trend + degradation onset in one call |
| `estimate_rul` | Remaining Useful Life from repeated measurements over time (linear, exponential, Kalman) — refuses single-recording extrapolation |
| `forecast_asset_indicator` | Projection of one indicator series you supply (ISO 8601 instants with an offset, declared unit and resolution) with a naive or seasonal-naive reference method: P10, P50 and P90 per step. A projection only, never a fault finding, a failure probability or a remaining life. Unreleased: main branch only |

## Decision Support (1)

| Tool | Description |
|------|-------------|
| `generate_maintenance_recommendations` | Maintenance recommendations from severity zone + canonical fault types |

## Asset Health Ledger (4)

Measurements whose companion declares a `measurement` object (asset, point, acquisition instant) are recorded in a local append-only ledger at load time, with a health snapshot each. These tools declare the context of a point, declare its healthy reference, read the recorded history and assess the change of a point across acquisitions. Nothing leaves the machine.

| Tool | Description |
|------|-------------|
| `declare_measurement_point` | Versioned declaration of a point's context: bearing or fault orders, nominal speed, ISO 20816-3 group and support, expected unit, sensor and direction; a re-declaration reports the changed keys and how many recorded measurements need re-processing. Example: `declare_measurement_point(asset_id="P-101", measurement_point_id="motor_de_h", bearing_id="6205", nominal_rpm=1800, machine_group=2, support_type="rigid", expected_signal_unit="g", expected_direction="horizontal")` Unreleased: main branch only |
| `declare_healthy_baseline` | Declares which recorded measurements are the healthy reference of a point, attributed to the declarer; an empty list withdraws it. Example: `declare_healthy_baseline(asset_id="P-101", measurement_point_id="motor_de_h", measurement_ids=["3f2a9c1d8e7b6a50", "b81c2d3e4f5a6978", "c0ffee1234567890"], declared_by="vibration-analyst", note="after bearing replacement")` Unreleased: main branch only |
| `get_asset_history` | Index of the recorded assets (no arguments) or the history of one asset read from its ledger alone: measurements newest first with an indicator preview, point declarations, baselines, integrity. Example: `get_asset_history(asset_id="P-101", max_measurements=20)` Unreleased: main branch only |
| `assess_asset_change` | Change of one point against its reference (declared baseline, or the first comparable acquisitions with health not declared): band per indicator, classification with its criterion, bearing evidence over the last K acquisitions classified against the reference band on the current lineage (legacy evidence reported as recorded, unclassifiable evidence as unavailable), at most one suggested verification; `reprocess=True` runs one call of a re-processing sequence (at most 10 attempts, 500 file checks and 50 failures per call) and its block gives `sequence_state`. A call without `reprocess_cursor` starts a new re-processing sequence, including the call a message, a remedy or a declaration names, and a named call is an attempt, not a promise: start one only when no sequence of the point has ended in the conversation, after the user took an action a sequence listed, after a load or a declaration changed the point, once after an environment fix, or when the user asks to list the blocked measurements again. While the block gives a `next_call`, make exactly that call; when it gives none, `sequence_state` is `complete` or `exhausted` (the measurements under `needs_action` stay stale until their action is taken). A new sequence attempts again the failures only an attempt finds. Example: `assess_asset_change(asset_id="P-101", measurement_point_id="motor_de_h", last_k=5)`; a `next_call` with a token: `assess_asset_change(asset_id="P-101", measurement_point_id="motor_de_h", reprocess=True, reprocess_cursor=...)` Unreleased: main branch only |

## Signal Import (4)

Delimited-text exports (`.csv`, `.txt`) with a header, a units row, a preamble or several channels are imported in four steps, and every value the import writes needs the user's own confirmation. See the [Signal Import Guide](IMPORT_GUIDE.md).

| Tool | Description |
|------|-------------|
| `inspect_signal_file` | Read-only inspection of a text export: regions with line numbers, dialect, columns by physical index, the values the file states, and a draft configuration with the confirmations still missing. File text only in `file_text` fields. Example: `inspect_signal_file(filepath="exports/pump_export.csv")` Unreleased: main branch only |
| `validate_signal_import` | Read-only validation of a configuration against the whole file: every value's basis checked against the evidence it cites, one statement per value, the asset-ledger pre-check, and `ready` (with an `import_digest`), `needs_confirmation` or `invalid`. Bases sent through MCP are recorded as caller-asserted. Unreleased: main branch only |
| `import_signal_file` | Imports a `ready` configuration with its `import_digest` (record first, then every channel stored as `<stem>_c<index>` or none, then each bound channel recorded in the asset ledger), or restores a recorded import with `import_id` alone. Example: `import_signal_file(import_id="<64 hex>")` Unreleased: main branch only |
| `get_signal_imports` | Index of the recorded imports, newest first and paged, or one import record (locations as file text, confirmations, stored signals). Example: `get_signal_imports(signal_id="pump_export_c1")` Unreleased: main branch only |

## Guided Workflows (4 prompts)

| Prompt | Description |
|--------|-------------|
| `diagnose_bearing` | Complete bearing fault diagnostic decision tree |
| `diagnose_gear` | Gear fault detection workflow |
| `quick_diagnostic_report` | Fast health screening |
| `guided_signal_import` | Step-by-step import of a text export: inspect, confirmations, validation, statements, import (unreleased: main branch only) |

---

Back to the [README](../README.md) · developer setup and extension guide: [Developer's Quickstart](QUICKSTART_DEVELOPER.md)
