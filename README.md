# 🏭 Predictive Maintenance MCP Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp)
[![FastMCP](https://img.shields.io/badge/FastMCP-powered-green.svg)](https://github.com/jlowin/fastmcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Luigi%20Di%20Maggio-0077B5?logo=linkedin)](https://www.linkedin.com/in/luigi-gianpio-di-maggio)

A Model Context Protocol server that brings **industrial machinery diagnostics** directly to LLMs like Claude, enabling AI-powered vibration analysis, bearing fault detection, and predictive maintenance—all through natural conversation.

> 🔧 **From Vibration Data to Actionable Insights**: Transform raw sensor data into professional diagnostics reports with FFT analysis, envelope analysis, ISO compliance checks, and ML anomaly detection—no engineering degree required.

## ✨ What Makes This Special

- **🎯 Real Bearing Fault Data Included** - 15 production-quality vibration signals from real machinery tests
- **📊 Professional HTML Reports** - Interactive Plotly visualizations with automatic peak detection and frequency markers
- **🤖 ML Anomaly Detection** - Train OneClassSVM/LocalOutlierFactor models with automatic parameter optimization
- **📏 ISO 20816-3 Compliance** - Industry-standard vibration severity assessment built-in
- **🔍 Advanced Diagnostics** - FFT spectrum analysis, envelope analysis for bearing faults, time-domain feature extraction
- **🚀 Zero Configuration** - Works out of the box with sample data, auto-detects sampling rates from metadata

## 🎬 Quick Example

```
Generate envelope report for real_train/OuterRaceFault_1.csv
```

**Result**: AI automatically:
1. Detects sampling rate from metadata (97,656 Hz)
2. Applies bandpass filter (500-5000 Hz)
3. Generates interactive HTML report with bearing fault frequencies marked
4. Identifies outer race fault at ~81 Hz with harmonics
5. Saves report to `reports/envelope_OuterRaceFault_1_*.html`

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

## Configuration

### Claude Desktop

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json` on Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

<details>
<summary>Using NPX</summary>

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "npx",
      "args": ["-y", "predictive-maintenance-mcp"]
    }
  }
}
```

</details>

<details>
<summary>Using UV</summary>

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "uvx",
      "args": ["predictive-maintenance-mcp"]
    }
  }
}
```

</details>

<details>
<summary>Using Python (from source)</summary>

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "python",
      "args": ["-m", "machinery_diagnostics_server"],
      "cwd": "C:/path/to/predictive-maintenance-mcp"
    }
  }
}
```

</details>

### VS Code

For manual installation, add to your MCP configuration:

**Method 1: User Configuration (Recommended)**
Open Command Palette (`Ctrl + Shift + P`) → `MCP: Open User Configuration`

**Method 2: Workspace Configuration**
Create `.vscode/mcp.json` in your workspace

```json
{
  "servers": {
    "predictive-maintenance": {
      "command": "npx",
      "args": ["-y", "predictive-maintenance-mcp"]
    }
  }
}
```

## 🔧 Available Tools

<details>
<summary><b>📊 Analysis & Diagnostics</b></summary>

- **`analyze_fft`** - FFT spectrum analysis with automatic peak detection
- **`analyze_envelope`** - Envelope analysis for bearing fault detection
- **`analyze_statistics`** - Time-domain statistical indicators (RMS, Crest Factor, Kurtosis, etc.)
- **`evaluate_iso_20816`** - ISO 20816-3 vibration severity assessment
- **`diagnose_bearing`** - Guided 6-step bearing diagnostic workflow
- **`diagnose_gear`** - Evidence-based gear fault diagnostic workflow

</details>

<details>
<summary><b>🤖 Machine Learning</b></summary>

- **`extract_features_from_signal`** - Extract 17+ statistical features from vibration data
- **`train_anomaly_model`** - Train OneClassSVM/LocalOutlierFactor on healthy baseline
- **`predict_anomalies`** - Detect anomalies in new signals with confidence scores

</details>

<details>
<summary><b>📄 Professional Report Generation</b></summary>

- **`generate_fft_report`** - Interactive FFT spectrum HTML report with peak table
- **`generate_envelope_report`** - Envelope analysis report with bearing fault markers
- **`generate_iso_report`** - ISO 20816-3 evaluation with zone visualization
- **`list_html_reports`** - List all generated reports with metadata
- **`get_report_info`** - Get report details without loading full HTML

> 💡 **All reports are interactive Plotly visualizations saved to `reports/` directory**

</details>

<details>
<summary><b>🔍 Data Management</b></summary>

- **`list_signals`** - Browse available signal files with metadata
- **`generate_test_signal`** - Create synthetic signals for testing

</details>

## 📊 Sample Dataset

The server includes **15 real bearing vibration signals** from production machinery:

- ✅ **3 Healthy Baselines** - Normal operation data
- ⚠️ **7 Outer Race Faults** - Various severity levels  
- 🔴 **5 Inner Race Faults** - Variable load conditions

**Specifications**: 97.7 kHz sampling rate, 6-second duration, BPFO=81.13 Hz

📖 **Full dataset documentation**: [data/README.md](data/README.md)

## 💡 Usage Examples

### Quick Fault Detection

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

📚 **More examples**: [EXAMPLES.md](EXAMPLES.md) | **Step-by-step tutorial**: [QUICKSTART.md](QUICKSTART.md)

## 📊 Professional Reports

All analysis tools generate **interactive HTML reports** with Plotly visualizations:

### Why HTML Reports?

✅ **Universal** - Works with any LLM (Claude, ChatGPT, local models)  
✅ **Zero tokens** - Files saved locally, not in chat  
✅ **Interactive** - Pan, zoom, hover for details  
✅ **Professional** - Publication-ready visualizations  
✅ **Persistent** - Save for documentation and sharing

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

## Documentation

- [EXAMPLES.md](EXAMPLES.md) - Complete diagnostic workflows
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step tutorials
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Debugging

Use MCP Inspector for interactive testing:

```bash
npx @modelcontextprotocol/inspector npx predictive-maintenance-mcp
```

Or from source:

```bash
uv run mcp dev src/machinery_diagnostics_server.py
```

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

## 🚀 Roadmap

### Coming in v0.3.0

#### 🤖 AI-Powered Machine Documentation Reader
**Automatic extraction of machine specifications from manuals and datasheets**

The next major feature will enable LLMs to automatically read and extract critical parameters from:
- 📄 **Equipment Manuals** (PDF) - Bearing specifications, operating speeds, power ratings
- 📊 **Bearing Catalogs** - Automatic BPFO/BPFI/BSF/FTF calculation from bearing geometry
- 🔧 **Technical Datasheets** - Machine parameters for ISO 20816-3 evaluation
- 📖 **OEM Documentation** - Parts specifications and maintenance requirements

**Benefits:**
- ✅ Eliminate manual parameter entry
- ✅ Reduce diagnostic errors from incorrect specifications
- ✅ Enable diagnostics on unknown equipment (just upload the manual!)
- ✅ Automatic bearing frequency calculations from geometry (pitch diameter, ball diameter, contact angle)

**Example workflow:**
```
"Upload the bearing datasheet and diagnose the vibration signal"
→ LLM extracts: Z=9 balls, Bd=7.94mm, Pd=34.55mm, α=0°
→ Calculates: BPFO=81.13 Hz, BPFI=118.88 Hz, BSF=63.91 Hz
→ Performs envelope analysis automatically
```

**Technical approach:** Combine LLM vision capabilities (Claude 3.5 Sonnet) with structured data extraction and validation.

### Future Enhancements
- Real-time signal streaming support
- Multi-signal comparison and trending
- Dashboard for multi-asset monitoring
- Mobile-friendly report viewing
- Cloud integration options
- Multimodal diagnostics (vibration + temperature + acoustic data)

💡 **Have ideas?** Open an issue or discussion to suggest features!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Sample data is licensed CC BY-NC-SA 4.0 (non-commercial). For commercial use, replace with your own machinery data.

## Citation

If you use this server in your research or projects:

```bibtex
@software{machinery_diagnostics_mcp,
  title = {Predictive Maintenance MCP Server},
  author = {Di Maggio, Luigi Gianpio},
  year = {2025},
  url = {https://github.com/LGDiMaggio/predictive-maintenance-mcp}
}
```

## Acknowledgments

- **FastMCP** framework by [@jlowin](https://github.com/jlowin)
- **Model Context Protocol** by [Anthropic](https://www.anthropic.com/)
- **Sample Data** from [MathWorks](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)

## Support

- **Issues**: https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues
- **Discussions**: https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions _(Enable in repo Settings → Features → Discussions)_

> 💬 **GitHub Discussions not yet enabled.** Repository owner can enable in Settings to allow community Q&A and feature requests.

---

**Built with ❤️ for condition monitoring professionals**
