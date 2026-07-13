# Predictive Maintenance Plugin for Claude Code

Industrial-grade predictive maintenance skills, agents, and workflows for Claude Code. Works with the [predictive-maintenance-mcp](https://github.com/LGDiMaggio/predictive-maintenance-mcp) server. Every workflow is designed to support and accelerate expert decision-making — never to replace it.

## Prerequisites

The `predictive-maintenance-mcp` MCP server must be installed and connected. See the [installation guide](https://github.com/LGDiMaggio/predictive-maintenance-mcp/blob/main/INSTALL.md) for setup instructions.

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

### Skills (8)

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

This plugin provides domain expertise for all 33 tools (plus 3 guided prompts — 36 endpoints total) of the predictive-maintenance-mcp server. Every signal is referenced by the `signal_id` returned by `load_signal`.

- **Signal Lifecycle**: load_signal, list_signals, get_signal_info, generate_test_signal, clear_signals
- **Spectral & Statistical Analysis**: analyze_fft, analyze_envelope, analyze_statistics, extract_features_from_signal, compute_power_spectral_density, compute_spectrogram_stft
- **Diagnostics & Health Assessment**: assess_severity, check_bearing_faults, diagnose_vibration, calculate_bearing_characteristic_frequencies, search_bearing_catalog, train_anomaly_model, predict_anomalies
- **Documentation**: list_machine_manuals, read_manual_excerpt, extract_manual_specs, search_documentation
- **Reports**: plot_signal, generate_fft_report, generate_envelope_report, generate_iso_report, generate_diagnostic_report_docx, generate_pca_visualization_report, generate_feature_comparison_report, list_html_reports
- **Prognostics**: analyze_signal_trend, estimate_rul
- **Decision Support**: generate_maintenance_recommendations
- **Guided Prompts**: diagnose_bearing, diagnose_gear, quick_diagnostic_report

## Standards Compliance

- **ISO 13374** — Six-block diagnostic architecture
- **ISO 20816-3** — Vibration severity classification (zone boundary values from ISO 10816-3:2009; provenance noted in tool output)
- **MIMOSA OSA-CBM** — Condition-based maintenance framework

## License

MIT
