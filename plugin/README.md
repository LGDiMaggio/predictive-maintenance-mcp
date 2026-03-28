# Predictive Maintenance Plugin for Claude Code

Industrial-grade predictive maintenance skills, agents, and workflows for Claude Code. Works with the [predictive-maintenance-mcp](https://github.com/LGDiMaggio/predictive-maintenance-mcp) server.

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

### Skills (7)

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
| `/pm-diagnose` | `/pm-diagnose bearing_001` | Start bearing fault diagnosis |
| `/pm-screen` | `/pm-screen bearing_001` | Quick health screening |
| `/pm-report` | `/pm-report bearing_001 full` | Generate diagnostic reports |

## MCP Tool Coverage

This plugin provides domain expertise for all 38 tools in the predictive-maintenance-mcp server:

- **Signal Acquisition**: load_signal, generate_test_signal, list_signals, list_stored_signals, get_signal_info, clear_signal, clear_all_signals
- **Spectral Analysis**: analyze_fft, analyze_envelope, compute_power_spectral_density, compute_spectrogram_stft, extract_features_from_signal
- **Visualization**: plot_signal, plot_spectrum, plot_envelope, plot_iso_20816_chart
- **Bearing Diagnostics**: calculate_bearing_characteristic_frequencies, check_bearing_fault_peak_tool, check_bearing_faults_direct, diagnose_vibration_tool, search_bearing_catalog, lookup_bearing_and_compute_tool
- **ISO Standards**: evaluate_iso_20816, assess_vibration_severity
- **ML / Anomaly Detection**: train_anomaly_model, predict_anomalies
- **Documentation**: search_documentation, read_manual_excerpt, extract_manual_specs, list_machine_manuals, load_and_validate_metadata
- **Reports**: generate_fft_report, generate_envelope_report, generate_iso_report, generate_feature_comparison_report, generate_pca_visualization_report, generate_diagnostic_report_docx

## Standards Compliance

- **ISO 13374** — Six-block diagnostic architecture
- **ISO 20816-3** — Vibration severity classification
- **MIMOSA OSA-CBM** — Condition-based maintenance framework

## License

MIT
