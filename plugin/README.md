# Predictive Maintenance Plugin for Claude Code

Industrial-grade predictive maintenance skills, agents, and workflows for Claude Code. Works with the [predictive-maintenance-mcp](https://github.com/LGDiMaggio/predictive-maintenance-mcp) server. Every workflow is designed to support and accelerate expert decision-making — never to replace it.

## Prerequisites

The `predictive-maintenance-mcp` MCP server must be installed and connected. See the [installation guide](https://github.com/LGDiMaggio/predictive-maintenance-mcp/blob/main/INSTALL.md) for setup instructions.

The bearing workflows read the `claim` and `outcome` fields of
`check_bearing_faults`. With a server that does not return them (0.13.0 and
earlier), the workflows relay no bearing fault type or bearing assessment, no
evidence strength of a diagnosis that ran the bearing check and no bearing
matches of an envelope report; they say why, and relay results that do not
rest on the bearing check, such as the ISO 20816-3 zone, as usual.

## Installation

### From Marketplace

```shell
/plugin marketplace add LGDiMaggio/predictive-maintenance-mcp
/plugin install predictive-maintenance@predictive-maintenance-marketplace
```

> This repo also includes `.claude-plugin/marketplace.json` at repository root for hosted distribution (`owner/repo`).

### Local Development

```shell
/plugin marketplace add .
/plugin install predictive-maintenance@predictive-maintenance-marketplace

# Optional: test only the plugin subdirectory marketplace
/plugin marketplace add ./plugin
/plugin install predictive-maintenance@predictive-maintenance-marketplace
```

## Components

### MCP server name for the agents

The two agents call the server's tools by name, as
`mcp__predictive-maintenance__<tool>`, so they reach them only when the server
is registered under the name `predictive-maintenance`, the name the
installation instructions use (register it under that name in Claude Code
too). Under any other name the agents cannot call the server's
tools and say so; skills and commands, which run in the main conversation, are
not affected. Each agent lists in the `tools:` field of its file only the
server tools its workflow uses (19 for diagnostic-pipeline, 10 for
signal-explorer); other server tools, such as `clear_signals` or
`train_anomaly_model`, are not available to the agents. Some listed tools
change state: `load_signal` stores signals in the server's memory, where
`overwrite=True` replaces the signal with the same id, and, for a file whose
companion declares an asset and measurement point, adds the measurement with
its declared unit, sampling rate and raw-format settings to the append-only
asset ledger (a re-load with changed settings adds a superseding entry; the
earlier one stays in the history); the report tools write new files under
`reports/`. Both agents also keep Read, Glob, Grep and Bash; their
instructions limit Bash to inspecting files. Claude Code asks before each
tool call according to your permission settings.

### Skills (9)

Skills activate automatically when Claude detects relevant context.

| Skill | Trigger Examples | Description |
|---|---|---|
| **bearing-diagnosis** | "diagnose bearing", "outer race fault" | Full bearing fault diagnostic workflow |
| **gear-diagnosis** | "gear fault", "gear mesh frequency" | Gear fault detection via spectral analysis |
| **quick-screening** | "quick check", "is this machine healthy" | Fast health screening (<30s) |
| **report-generation** | "generate report", "create PDF" | Professional HTML/DOCX report generation |
| **anomaly-detection** | "train model", "detect anomalies" | ML-based anomaly detection (SVM, LOF) |
| **signal-management** | "load signal", "generate test signal" | Signal loading, generation, cache management |
| **documentation-search** | "search manual", "look up bearing" | RAG search across manuals and catalogs |
| **prognostics** | "trend analysis", "remaining useful life" | Within-recording screening + multi-measurement RUL |
| **shaft-order-evidence** | "is this unbalance", "misalignment or looseness" | Experimental shaft-order evidence; attributes no fault (unreleased tool) |

### Agents (2)

Agents run autonomously for complex, multi-step tasks.

| Agent | When to Use | Description |
|---|---|---|
| **diagnostic-pipeline** | "Run a full diagnostic on this signal" | Complete ISO 13374 pipeline: load -> analyze -> diagnose -> report |
| **signal-explorer** | "Explore these signals and compare them" | Signal characterization, comparison, and outlier detection |

### Commands (3)

Quick entry points for common workflows.

| Command | Usage | Description |
|---|---|---|
| `/pm-diagnose` | `/pm-diagnose real_train_baseline_1` | Start bearing fault diagnosis |
| `/pm-screen` | `/pm-screen real_train_baseline_1` | Quick health screening |
| `/pm-report` | `/pm-report real_train_baseline_1 full` | Generate diagnostic reports |

## MCP Tool Coverage

This plugin provides domain expertise for the predictive-maintenance-mcp server on the main branch, which registers 44 tools (plus 4 guided prompts, 48 endpoints total); this plugin's skills do not yet cover the signal import tools, and tools listed under Unreleased in the server's [CHANGELOG](../CHANGELOG.md) are not yet in a published release. Every signal is referenced by the `signal_id` returned by `load_signal`.

- **Signal Lifecycle**: load_signal, list_signals, get_signal_info, generate_test_signal, clear_signals
- **Spectral & Statistical Analysis**: analyze_fft, analyze_envelope, analyze_statistics, extract_features_from_signal, compute_power_spectral_density, compute_spectrogram_stft
- **Diagnostics & Health Assessment**: assess_severity, check_bearing_faults, diagnose_vibration, calculate_bearing_characteristic_frequencies, search_bearing_catalog, train_anomaly_model, predict_anomalies, assess_shaft_fault_evidence (unreleased, experimental)
- **Documentation**: list_machine_manuals, read_manual_excerpt, extract_manual_specs, search_documentation
- **Reports**: plot_signal, generate_fft_report, generate_envelope_report, generate_iso_report, generate_diagnostic_report, generate_diagnostic_report_docx, generate_pca_visualization_report, generate_feature_comparison_report, list_html_reports
- **Prognostics**: analyze_signal_trend, estimate_rul, forecast_asset_indicator (unreleased)
- **Decision Support**: generate_maintenance_recommendations
- **Asset Health Ledger** (unreleased): declare_measurement_point, declare_healthy_baseline, get_asset_history, assess_asset_change
- **Signal Import** (unreleased): inspect_signal_file, validate_signal_import, import_signal_file, get_signal_imports
- **Guided Prompts**: diagnose_bearing, diagnose_gear, quick_diagnostic_report, guided_signal_import (unreleased)

## Standards Compliance

- **ISO 13374** — Six-block diagnostic architecture
- **ISO 20816-3** — Vibration severity classification (zone boundary values from ISO 10816-3:2009; provenance noted in tool output)
- **MIMOSA OSA-CBM** — Condition-based maintenance framework

## License

MIT
