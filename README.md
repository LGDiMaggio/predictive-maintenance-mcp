# 🏭 Predictive Maintenance MCP Server

<!-- mcp-name: io.github.LGDiMaggio/predictive-maintenance-mcp -->

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17611542.svg)](https://doi.org/10.5281/zenodo.17611542)
[![Tests](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp)
[![FastMCP](https://img.shields.io/badge/FastMCP-powered-green.svg)](https://github.com/jlowin/fastmcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Luigi%20Di%20Maggio-0077B5?logo=linkedin)](https://www.linkedin.com/in/luigi-gianpio-di-maggio)

> **Transform raw vibration data into actionable maintenance insights through natural conversation with AI.**

A [Model Context Protocol](https://modelcontextprotocol.io/) server that brings **industrial machinery diagnostics** directly to LLMs like Claude, enabling AI-powered vibration analysis, bearing fault detection, and predictive maintenance workflows — all through natural language.

![Predictive Maintenance MCP Server Cover](assets/predictive-maintenance-mcp_com.jpg)

---

## 🎯 Our Mission

Predictive maintenance is critical for Industry 4.0, yet expert-level machinery diagnostics remains inaccessible to most engineers. Complex diagnostic workflows — FFT spectrum analysis, envelope demodulation, ISO severity assessment — require years of specialized training.

**We believe that AI can democratize this expertise.**

By combining the reasoning capabilities of Large Language Models with specialized diagnostic tools through the Model Context Protocol (MCP), we create a bridge: engineers can describe a problem in plain language and receive professional-grade analysis. No signal processing PhD required.

This project is an **open-source framework** that proves this vision works. It's a foundation — a set of building blocks — that the community can extend, customize, and deploy for any industrial diagnostics scenario.

> 📖 **Read the full story**: [Building an AI-Powered Predictive Maintenance System with MCP and Claude](https://medium.com/@luigigianpio.dimaggio/building-an-ai-powered-predictive-maintenance-system-with-model-context-protocol-and-claude-1b0ed588e574)

---

## 🏗️ Ecosystem Architecture

This project is built around the **Model Context Protocol (MCP)** — an open standard that lets you package any software tool and make it instantly queryable by an LLM. Think of it as a USB port for AI: plug in a tool, and the LLM immediately knows how to use it.

```
┌──────────────────────────────────────────────────────────────┐
│   YOU (Natural Language)                                     │
│   "Is this bearing failing? Show me the envelope analysis."  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│   LLM (Claude, GPT, local models...)                         │
│   Understands your question, selects the right tools         │
└──────────────┬───────────────────────────────────────────────┘
               │  Model Context Protocol (MCP)
               ▼
┌──────────────────────────────────────────────────────────────┐
│   PREDICTIVE MAINTENANCE MCP SERVER                          │
│   ┌────────────────┐  ┌───────────────┐  ┌───────────────┐  │
│   │ FFT Analysis   │  │ Envelope      │  │ ISO 20816-3   │  │
│   │ & Reporting    │  │ Demodulation  │  │ Compliance    │  │
│   └────────────────┘  └───────────────┘  └───────────────┘  │
│   ┌────────────────┐  ┌───────────────┐  ┌───────────────┐  │
│   │ ML Anomaly     │  │ Manual/PDF    │  │ Bearing       │  │
│   │ Detection      │  │ Reader        │  │ Catalog       │  │
│   └────────────────┘  └───────────────┘  └───────────────┘  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│   YOUR DATA (raw files stay on your machine)                 │
│   Vibration signals · Equipment manuals · Trained models     │
│                                                              │
│   ⚠️  Analysis results (peaks, RMS, diagnoses) flow back     │
│   to the LLM provider. Use a local LLM for full air-gap.     │
└──────────────────────────────────────────────────────────────┘
```

**Key insight**: The LLM doesn't know signal processing. It knows how to *call tools* that do signal processing. MCP is the standard that makes this plug-and-play — you can add new diagnostic tools without changing the LLM.

### What This Means For You

- **🔌 Plug-and-play** — Add new analysis tools (thermography, oil analysis, acoustics) as simple Python functions — the LLM discovers them automatically
- **🔒 Local processing** — Raw signals never leave your machine; only computed results (peaks, RMS, diagnoses) flow to the LLM. Use a [local LLM](https://ollama.com/) for full air-gapped privacy
- **🤖 LLM-agnostic** — Works with Claude, ChatGPT, or any MCP-compatible client
- **🧱 Modular** — Use only the tools you need, extend with your own

---

## 📑 Table of Contents

- [🎯 Our Mission](#-our-mission)
- [🏗️ Ecosystem Architecture](#️-ecosystem-architecture)
- [🚪 Choose Your Path](#-choose-your-path)
- [✨ What Makes This Special](#-what-makes-this-special)
- [🎬 Quick Examples](#-quick-examples)
- [🚀 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🔧 Available Tools & Resources](#-available-tools--resources)
- [🤖 Copilot Skills](#copilot-skills-guided-workflows)
- [📐 Detailed Architecture](#-detailed-architecture)
- [📊 Sample Dataset](#-sample-dataset)
- [💡 Usage Examples](#-usage-examples)
- [📊 Professional Reports](#-professional-reports)
- [📖 Documentation](#-documentation)
- [🧪 Testing](#-testing)
- [🛠️ Development](#️-development)
- [🚀 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📚 Citation](#-citation)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🚪 Choose Your Path

This project serves two audiences. Pick the door that fits you:

<table>
<tr>
<td width="50%" valign="top">

### 🔧 I'm a Maintenance / Reliability Engineer

*"I want to use AI for my vibration analysis and diagnostics."*

**You don't need to write code.** This tool turns your vibration data into professional reports through simple conversation.

**What you'll get:**
- ISO 20816-3 compliance reports in one sentence
- Bearing fault detection from your real data
- Interactive HTML reports you can share with your team
- ML anomaly detection trained on your healthy baselines

➡️ **Start here**: [**Quickstart for Engineers**](docs/QUICKSTART_ENGINEER.md)

</td>
<td width="50%" valign="top">

### 💻 I'm an AI / Software Developer

*"I want to understand MCP, extend this server, or build my own."*

**The code is your playground.** Learn how MCP works, add new diagnostic tools, or use this as a template for your own domain-specific server.

**What you'll learn:**
- How MCP bridges LLMs and specialized tools
- How to create new tools as Python functions
- How to design resource/tool hybrid architectures
- How to contribute to an industrial open-source project

➡️ **Start here**: [**Quickstart for Developers**](docs/QUICKSTART_DEVELOPER.md)

</td>
</tr>
</table>

> 💡 **Don't fit either profile?** That's fine — read on for the full documentation, or jump to [Quick Examples](#-quick-examples) to see the server in action.

---

## ✨ What Makes This Special

- **🎯 Real Bearing Fault Data Included** — 20 production-quality vibration signals from real machinery tests (3 healthy, 17 faulty)
- **📊 Professional HTML Reports** — Interactive Plotly visualizations with automatic peak detection and frequency markers
- **🤖 ML Anomaly Detection** — Train unsupervised/semi-supervised models (OneClassSVM/LOF) on healthy baselines with optional hyperparameter tuning
- **📏 ISO 20816-3 Compliance** — Industry-standard vibration severity assessment built-in
  ![ISO Compliance](assets/iso.png)
- **🔍 Advanced Diagnostics** — FFT spectrum analysis, envelope analysis for bearing faults, time-domain feature extraction
  <details>

  <summary><b>Example analysis</b></summary>
  
  ![Envelope analysis 1](assets/envelope_analysis.png)
  ![Envelope analysis 2](assets/envelope_signals.png)
  ![Envelope analysis 3](assets/envelope_list.png)
  
  </details>
- **📁 Multi-Format Support** — Load signals from CSV, MAT (MATLAB), WAV, NPY, and Parquet files
- **🚀 Zero Configuration** — Works out of the box with sample data, auto-detects sampling rates from metadata

---

## 🎬 Quick Examples

### Example 1: Bearing Fault Detection

```
Generate envelope report for real_train/OuterRaceFault_1.csv
```

**Result**: AI automatically:
1. Detects sampling rate from metadata (97,656 Hz)
2. Applies bandpass filter (500-5000 Hz)
3. Generates interactive HTML report with bearing fault frequencies marked
4. Identifies outer race fault at ~81 Hz with harmonics
5. Saves report to `reports/envelope_OuterRaceFault_1_*.html`

### Example 2: ISO 20816-3 Vibration Assessment

```
Evaluate real_train/OuterRaceFault_1.csv against ISO 20816-3 standard
```

**Result**: 
- RMS velocity: 4.5 mm/s → Zone B (Acceptable for long-term operation)
- Interactive HTML report with zone visualization
- Compliance assessment and recommendations

### Example 3: Machine Manual Integration + Diagnosis

```
1. Extract specifications from test_pump_manual.pdf
2. Calculate bearing frequencies for SKF 6205-2RS at 1475 RPM
3. Diagnose bearing fault in signal_from_pump.csv using calculated frequencies
```

**Result**: Complete zero-knowledge diagnosis:
- Extracts: Drive end bearing SKF 6205-2RS, operating speed 1475 RPM
- Calculates: BPFO=85.20 Hz, BPFI=136.05 Hz, BSF=101.32 Hz
- Diagnoses: Outer race fault detected with 3 harmonics

📚 **More examples**: See [Usage Examples](#-usage-examples) section below or [EXAMPLES.md](EXAMPLES.md) for complete workflows

---

## 🚀 Installation

### Quick Start (Python Package)

```bash
# 1. Clone repository
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp

# 2. Run automated setup
python setup_venv.py

# 3. Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 4. Verify installation
python validate_server.py
```

📖 **Detailed Installation Guide**: See [INSTALL.md](INSTALL.md) for troubleshooting, Claude Desktop setup, and developer instructions.

### From Source (Advanced)

```bash
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
pip install -e .
```

### Run as Module

After installation, you can also run the server directly:

```bash
python -m predictive_maintenance_mcp
```

---

## ⚙️ Configuration

### Claude Desktop

Add to your Claude Desktop config:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "C:/path/to/predictive-maintenance-mcp/.venv/Scripts/python.exe",
      "args": ["C:/path/to/predictive-maintenance-mcp/src/machinery_diagnostics_server.py"]
    }
  }
}
```

> **Important Notes**:
> - Replace `C:/path/to/predictive-maintenance-mcp` with your actual project path
> - Use **absolute paths** for both `command` and `args`
> - On macOS/Linux, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`
> - Forward slashes (`/`) work on all platforms, including Windows

After configuration, **restart Claude Desktop** completely.

### VS Code

Add to your MCP configuration (`.vscode/mcp.json` or user settings):

```json
{
  "servers": {
    "predictive-maintenance": {
      "command": "/path/to/predictive-maintenance-mcp/.venv/bin/python",
      "args": ["/path/to/predictive-maintenance-mcp/src/machinery_diagnostics_server.py"]
    }
  }
}
```

> Adjust paths according to your system (use `.venv/Scripts/python.exe` on Windows)

---

## 🔧 Available Tools & Resources

### MCP Resources (Direct Data Access)

Resources provide **direct read access** for Claude to examine data:

<details>
<summary><b>📊 Vibration Signals</b></summary>

- **`signal://list`** — Browse all available signal files with metadata
- **`signal://read/{filename}`** — Read signal data directly (supports CSV, MAT, WAV, NPY, Parquet)

**Usage:** Claude can directly read signals without calling tools first.

</details>

<details>
<summary><b>📖 Machine Manuals</b></summary>

- **`manual://list`** — Browse available equipment manuals (PDF/TXT)
- **`manual://read/{filename}`** — Read manual text (first 20 pages)

**Usage:** Claude can answer ANY question about manual content by reading directly.

</details>

---

### MCP Tools (Analysis & Processing)

Tools perform **computations and generate outputs**:

<details>
<summary><b>📊 Analysis & Diagnostics</b></summary>

- **`analyze_fft`** — FFT spectrum analysis with automatic peak detection
- **`analyze_envelope`** — Envelope analysis for bearing fault detection
- **`analyze_statistics`** — Time-domain statistical indicators (RMS, Crest Factor, Kurtosis, etc.)
- **`evaluate_iso_20816`** — ISO 20816-3 vibration severity assessment
- **`diagnose_bearing`** — Guided 6-step bearing diagnostic workflow
- **`diagnose_gear`** — Evidence-based gear fault diagnostic workflow

</details>

<details>
<summary><b>🤖 Machine Learning</b></summary>

- **`extract_features_from_signal`** — Extract 17+ statistical features from vibration data
- **`train_anomaly_model`** — Train novelty detection models (OneClassSVM/LOF) on healthy data only, with optional semi-supervised hyperparameter tuning
- **`predict_anomalies`** — Detect anomalies in new signals with confidence scores

</details>

<details>
<summary><b>📄 Professional Report Generation</b></summary>

- **`generate_fft_report`** — Interactive FFT spectrum HTML report with peak table
- **`generate_envelope_report`** — Envelope analysis report with bearing fault markers
- **`generate_iso_report`** — ISO 20816-3 evaluation with zone visualization
- **`generate_pca_visualization_report`** — 2D/3D PCA projection report for anomaly exploration
- **`generate_feature_comparison_report`** — Feature-level comparison report across signals/classes
- **`list_html_reports`** — List all generated reports with metadata
- **`get_report_info`** — Get report details without loading full HTML

> 💡 **All reports are interactive Plotly visualizations saved to `reports/` directory**

</details>

<details>
<summary><b>📖 Machine Documentation Reader</b></summary>

- **`list_machine_manuals`** — List available equipment manuals (PDF/TXT)
- **`extract_manual_specs`** — Extract bearings, RPM, power from manual (with caching)
- **`calculate_bearing_characteristic_frequencies`** — Calculate BPFO/BPFI/BSF/FTF from geometry
- **`read_manual_excerpt`** — Read manual text excerpt (configurable page limit)
- **`search_bearing_catalog`** — Search bearing geometry in local catalog (20+ common bearings)

**MCP Resources:**
- `manual://list` — Browse available manuals
- `manual://read/{filename}` — Read manual for LLM context

> 🎯 **Upload pump manual → Extract bearing specs → Auto-calculate frequencies → Diagnose signal**

</details>

<details>
<summary><b>🔍 Data Management & Visualization</b></summary>

- **`list_signals`** — Browse available signal files with metadata
- **`generate_test_signal`** — Create synthetic signals for testing
- **`plot_signal`** — Generate time-domain signal plot
- **`plot_spectrum`** — Generate FFT spectrum plot
- **`plot_envelope`** — Generate envelope spectrum plot

</details>

---

### Copilot Skills (Guided Workflows)

The `skills/` directory contains pre-built guided workflows that orchestrate multiple MCP tools into structured diagnostic procedures:

| Skill | Steps | Description |
|-------|:-----:|-------------|
| [**bearing-diagnosis**](skills/bearing-diagnosis/SKILL.md) | 8 | Complete bearing fault detection: statistics → FFT → envelope → frequency matching → ISO severity → report |
| [**quick-screening**](skills/quick-screening/SKILL.md) | 5 | Fast health screening with clear Healthy/Suspicious/Critical classification |
| [**report-generation**](skills/report-generation/SKILL.md) | 6 | Professional HTML report generation with composite multi-report option |

> 💡 Skills are standalone markdown files that any MCP-compatible LLM client can use as system instructions to coordinate multi-step diagnostic workflows.

---

## 📐 Detailed Architecture

The system follows a **hybrid MCP architecture** combining Resources (direct data access) and Tools (computational processing):

![MCP Server Architecture](assets/MCPserver.png)

<details>
<summary><b>Detailed Structure</b></summary>

```
┌─────────────────────────────────────────────────────────────┐
│                    CLAUDE / LLM CLIENT                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP SERVER (FastMCP)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RESOURCES (Direct Data Access)                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Vibration Signals                             │  │  │
│  │  │  • signal://list                               │  │  │
│  │  │  • signal://read/{filename}                    │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Machine Manuals                               │  │  │
│  │  │  • manual://list                               │  │  │
│  │  │  • manual://read/{filename}                    │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TOOLS (Analysis & Processing)                       │  │
│  │  • FFT, Envelope, ISO 20816-3                        │  │
│  │  • ML Anomaly Detection                              │  │
│  │  • Report Generation (HTML)                          │  │
│  │  • Manual Spec Extraction                            │  │
│  │  • Bearing Frequency Calculation                     │  │
│  │  • Bearing Catalog Search                            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  SIGNAL ANALYSIS │   │  DOCUMENT READER MODULE          │
│  MODULE          │   │  ┌────────────┐  ┌────────────┐  │
│  • FFT Engine    │   │  │ PDF Extract│  │ ISO Formulas│  │
│  • Envelope      │   │  │ (pypdf)    │  │ BPFO/BPFI   │  │
│  • Filters       │   │  └────────────┘  └────────────┘  │
│  • Statistics    │   │  ┌─────────────────────────────┐  │
│  • ML Models     │   │  │  Bearing Catalog DB         │  │
│  • Plotly Charts │   │  │  • 20+ ISO bearings         │  │
│                  │   │  └─────────────────────────────┘  │
└────────┬─────────┘   └────────┬─────────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   LOCAL FILE SYSTEM                         │
│  ┌──────────────────────┐   ┌──────────────────────────┐   │
│  │  data/signals/       │   │  resources/              │   │
│  │  ├── real_train/     │   │  ├── machine_manuals/    │   │
│  │  ├── real_test/      │   │  ├── bearing_catalogs/   │   │
│  │  └── samples/        │   │  ├── datasheets/         │   │
│  └──────────────────────┘   │  └── cache/ (auto)       │   │
│  ┌──────────────────────┐   └──────────────────────────┘   │
│  │  reports/            │   ┌──────────────────────────┐   │
│  │  • FFT reports       │   │  models/                 │   │
│  │  • Envelope reports  │   │  • Trained ML models     │   │
│  │  • ISO reports       │   │  • Scalers, PCA          │   │
│  └──────────────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

</details>

**Key Features:**
- ✅ **4 MCP Resources** — Direct read access to signals and manuals
- ✅ **25 MCP Tools** — Complete diagnostic workflow (analysis, plotting, ML, reporting, manuals)
- ✅ **4 MCP Prompts** — Guided diagnostic workflows
- ✅ **Hybrid Architecture** — Resources for reading, Tools for processing
- ✅ **Local-First** — All data stays on your machine (privacy-preserving)

---

## 📊 Sample Dataset

The server includes **20 real bearing vibration signals** from production machinery:

**Training Set (14 signals)**:
- ✅ **2 Healthy Baselines** — Normal operation data
- 🔴 **7 Inner Race Faults** — Variable load conditions
- ⚠️ **5 Outer Race Faults** — Various severity levels

**Test Set (6 signals)**:
- ✅ **1 Healthy Baseline** — Validation data
- 🔴 **2 Inner Race Faults** — Test conditions
- ⚠️ **3 Outer Race Faults** — Test conditions

> **Note**: Sampling rates and durations vary by signal (48.8-97.7 kHz, 3-6 seconds). All parameters auto-detected from metadata files.

📖 **Full dataset documentation**: [data/README.md](data/README.md)

---

## 💡 Usage Examples

### Quick Fault Detection

```
Diagnose bearing fault in real_train/OuterRaceFault_1.csv
BPFO=81.13 Hz, BPFI=118.88 Hz, BSF=63.91 Hz, FTF=14.84 Hz
```

**Result:** ✅ Outer race fault detected at ~81 Hz with harmonics

### Generate Professional Report

```
Generate envelope report for real_train/OuterRaceFault_1.csv
```

**Result:** Interactive HTML saved to `reports/` with bearing fault markers

### Train ML Anomaly Detector

```
Train anomaly model on baseline_1.csv and baseline_2.csv
Validate on OuterRaceFault_1.csv
```

**Result:** Model detects fault with 95%+ confidence

📚 **More examples**: [EXAMPLES.md](EXAMPLES.md) for complete diagnostic workflows

---

## 📊 Professional Reports

All analysis tools generate **interactive HTML reports** with Plotly visualizations:

### Why HTML Reports?

✅ **Universal** — Works with any LLM (Claude, ChatGPT, local models)  
✅ **Zero tokens** — Files saved locally, not in chat  
✅ **Interactive** — Pan, zoom, hover for details  
✅ **Professional** — Publication-ready visualizations  
✅ **Persistent** — Save for documentation and sharing

### Report Types

| Report | Tool | Contents |
|--------|------|----------|
| 🔊 **FFT Spectrum** | `generate_fft_report()` | Frequency analysis, peak detection, harmonic markers |
| 🎯 **Envelope Analysis** | `generate_envelope_report()` | Bearing fault frequencies, modulation detection |
| 📏 **ISO 20816-3** | `generate_iso_report()` | Vibration severity zones, compliance assessment |

All reports include:
- Interactive Plotly charts (pan/zoom/hover)
- Automatic peak detection with frequency tables
- Metadata (signal info, analysis parameters)
- Timestamp and file references

**Usage:**
```
Generate FFT report for baseline_1.csv
```
→ Opens `reports/fft_spectrum_baseline_1_20251111_143022.html` in browser

---

## 📖 Documentation

| Document | Audience | Description |
|----------|----------|-------------|
| [**Quickstart for Engineers**](docs/QUICKSTART_ENGINEER.md) | 🔧 Engineers | Get results fast, no coding required |
| [**Quickstart for Developers**](docs/QUICKSTART_DEVELOPER.md) | 💻 Developers | Understand MCP, extend the server |
| [EXAMPLES.md](EXAMPLES.md) | Everyone | Complete diagnostic workflows with step-by-step tutorials |
| [INSTALL.md](INSTALL.md) | Everyone | Detailed installation and troubleshooting guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributors | How to contribute (for every skill level) |
| [Ollama Guide](docs/OLLAMA_GUIDE.md) | Engineers | Use with local LLMs (fully air-gapped) |
| [CHANGELOG.md](CHANGELOG.md) | Everyone | Version history |
| [data/README.md](data/README.md) | Everyone | Dataset documentation |
| [skills/](skills/) | 🤖 LLM Clients | Copilot Skills — guided diagnostic workflows (bearing, screening, reporting) |

---

## 🧪 Testing

This project includes a comprehensive test suite covering all analysis tools:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fft_analysis.py

# Run with verbose output
pytest -v
```

**Test coverage includes:**
- ✅ FFT analysis and peak detection
- ✅ Envelope analysis and bearing fault detection
- ✅ ISO 20816-3 evaluation and zone classification
- ✅ ML tools (feature extraction, training, prediction)
- ✅ Report generation system (HTML outputs)
- ✅ Real bearing fault data validation

See [tests/README.md](tests/README.md) for detailed testing documentation.

---

## 🛠️ Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Debugging

Use MCP Inspector for interactive testing:

```bash
npx @modelcontextprotocol/inspector npx predictive-maintenance-mcp
```

Or from source:

```bash
uv run mcp dev src/machinery_diagnostics_server.py
```

---

## 🚀 Roadmap

### ✨ Recent: v0.5.0 — Code Quality & Multi-Format Support

- 📂 **Multi-format signal loading** — CSV, MAT, WAV, NPY, Parquet via unified `load_signal_data()`
- 🔧 **ML code deduplication** — 4 helper functions reduce ~163 statements
- 📦 **pypdf migration** — Replaced deprecated PyPDF2 with pypdf
- ▶️ **`python -m` support** — Run as `python -m predictive_maintenance_mcp`
- 🧹 **Consolidated metadata reads** — ISO evaluation no longer double-reads metadata files

### 🔮 Planned Enhancements

Each item below links to an open issue where you can **discuss, contribute, or claim the task**:

| Priority | Enhancement | Status | Get Involved |
|----------|-------------|--------|--------------|
| ✅ Done | **Parquet/MAT/WAV/NPY data format support** | v0.5.0 | — |
| 🔴 High | **Customizable ISO report thresholds** | Open | [Good First Issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| 🔴 High | **Docker image for zero-install setup** | Open | [Help Wanted](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| 🟡 Medium | **Vector search for large documents** (ChromaDB/FAISS) | Planned | [Discuss](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) |
| 🟡 Medium | **OCR for scanned PDF manuals** (Tesseract) | Planned | [Discuss](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) |
| 🟡 Medium | **Multi-signal trending** — Compare historical data | Planned | [Discuss](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) |
| 🟢 Future | **Real-time streaming** — Live vibration monitoring | Concept | — |
| 🟢 Future | **Dashboard** — Multi-asset fleet monitoring | Concept | — |
| 🟢 Future | **Online bearing catalog** — Web search for unknown bearings | Concept | — |
| 🟢 Future | **Multimodal fusion** — Vibration + temperature + acoustic | Concept | — |

> 💡 **Have ideas?** [Open a discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) or [create an issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues/new/choose)!

---

## 🤝 Contributing

We welcome contributions from **everyone** — not just programmers. See our full [CONTRIBUTING.md](CONTRIBUTING.md) guide, which includes specific paths for:

- **🔧 Domain experts** — Validate signals, add datasets, review diagnostic logic
- **💻 Developers** — Add tools, fix bugs, improve architecture
- **📖 Technical writers** — Improve docs, add tutorials, translate content
- **🧪 Testers** — Edge cases, validation with ground truth data

### Quick Start for Contributors

1. Browse [Issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) — look for `good first issue` or `help wanted` labels
2. Comment on the issue to claim it
3. Fork → Branch → Code → Test → PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup and guidelines.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Note**: Sample data is licensed CC BY-NC-SA 4.0 (non-commercial). For commercial use, replace with your own machinery data.

---

## 📚 Citation

If you use this server in your research or projects, please cite:

```bibtex
@software{dimaggio_predictive_maintenance_mcp_2025,
  title = {Predictive Maintenance MCP Server: An open-source framework for integrating Large Language Models with predictive maintenance and fault diagnosis workflows},
  author = {Di Maggio, Luigi Gianpio},
  year = {2025},
  version = {0.5.0},
  url = {https://github.com/LGDiMaggio/predictive-maintenance-mcp},
  doi = {10.5281/zenodo.17611542}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17611542.svg)](https://doi.org/10.5281/zenodo.17611542)

---

## 🙏 Acknowledgments

- **FastMCP** framework by [@jlowin](https://github.com/jlowin)
- **Model Context Protocol** by [Anthropic](https://www.anthropic.com/)
- **Sample Data** from [MathWorks](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)
- **Development Assistance**: Core codebase and demonstration examples were developed with assistance from [Claude](https://claude.ai) by Anthropic to rapidly prototype and demonstrate the Model Context Protocol (MCP) concept for predictive maintenance applications.

> ⚠️ **Development Notice**: This codebase was generated using Claude AI under human supervision to explore and validate MCP-based approaches for industrial diagnostics and predictive maintenance workflows. While the implementation demonstrates the potential of AI-assisted development for specialized engineering domains, **thorough testing and validation are required** before any production or safety-critical use.

---

## 📬 Support

- **Issues**: https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues
- **Discussions**: https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions
- **Blog post**: [Building an AI-Powered Predictive Maintenance System with MCP and Claude](https://medium.com/@luigigianpio.dimaggio/building-an-ai-powered-predictive-maintenance-system-with-model-context-protocol-and-claude-1b0ed588e574)

---

**Built with ❤️ for condition monitoring professionals and the open-source community**
