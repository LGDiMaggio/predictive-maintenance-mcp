# Predictive Maintenance MCP Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-powered-green.svg)](https://github.com/jlowin/fastmcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Model Context Protocol server that enables AI assistants like Claude to perform professional vibration analysis, bearing diagnostics, and anomaly detection for industrial machinery.

> 🏭 **Industrial AI Made Accessible**: Connect Claude to real machinery diagnostics—from vibration analysis to predictive maintenance—without writing code.

## Features

- **FFT Analysis** - Spectrum analysis with automatic peak detection
- **Envelope Analysis** - Advanced bearing fault detection (BPFO, BPFI, BSF, FTF)
- **ISO 20816-3 Compliance** - Industry-standard vibration severity evaluation
- **Machine Learning** - Anomaly detection with OneClassSVM/LocalOutlierFactor
- **Feature Extraction** - 17 time-domain statistical features (RMS, Kurtosis, Crest Factor, etc.)
- **Interactive Visualizations** - Plotly charts with peak markers and frequency annotations
- **Sample Datasets** - Real bearing fault data included (baseline, inner race, outer race faults)
- **Safety Features** - Evidence-based inference policy prevents AI hallucination

## Installation

### NPM (Recommended)

```bash
npx predictive-maintenance-mcp
```

### UV

```bash
uvx predictive-maintenance-mcp
```

### From Source

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

## Quick Start

### Test with Sample Data

The server includes real bearing vibration signals from MathWorks. After restarting Claude Desktop:

```
Diagnose bearing fault in real_train/OuterRaceFault_1.csv with 97656 Hz sampling rate.
The bearing has BPFO=81.13 Hz, BPFI=118.88 Hz, BSF=63.91 Hz, FTF=14.84 Hz.
```

**Expected result:** ✅ Outer race fault detected with harmonics at ~81 Hz, 162 Hz, 243 Hz

### Available Tools

#### Analysis Tools

- `analyze_fft` - FFT spectrum analysis with peak detection
- `analyze_envelope` - Envelope analysis for bearing fault detection
- `evaluate_iso_20816` - ISO 20816-3 vibration severity evaluation
- `diagnose_bearing` - Complete 6-step bearing diagnostic workflow

#### Machine Learning Tools

- `extract_features_from_signal` - Extract 17 statistical features
- `train_anomaly_model` - Train ML model on healthy baseline data
- `predict_anomalies` - Detect anomalies in new signals

#### Visualization Tools

- `generate_fft_chart_html` - Interactive FFT spectrum chart
- `generate_envelope_html` - Interactive envelope spectrum chart
- `save_fft_chart_to_file` - Save FFT chart to HTML file
- `save_envelope_chart_to_file` - Save envelope chart to HTML file

#### Utility Tools

- `list_signals` - List all available signal files
- `generate_test_signal` - Generate synthetic test signals
- `read_plot_html` - Read HTML plot files for rendering

## Sample Datasets

**Training Set** (`data/signals/real_train/`):
- `baseline_1.csv`, `baseline_2.csv` - Healthy bearing operation
- `OuterRaceFault_1.csv`, `OuterRaceFault_2.csv` - Outer race defects
- `InnerRaceFault_vload_1-5.csv` - Inner race defects (variable load)

**Test Set** (`data/signals/real_test/`):
- `baseline_3.csv` - Healthy bearing
- `OuterRaceFault_3.csv` - Outer race fault
- `InnerRaceFault_vload_6-7.csv` - Inner race faults

**Specifications:**
- **Sampling Rate**: 97,656 Hz (~98 kHz)
- **Duration**: 6.0 seconds each
- **Bearing Frequencies**: BPFO=81.13 Hz, BPFI=118.88 Hz, BSF=63.91 Hz, FTF=14.84 Hz

**Data Source**: [MathWorks RollingElementBearingFaultDiagnosis-Data](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)  
**License**: CC BY-NC-SA 4.0 (Non-commercial use only)

## Examples

### Example 1: Detect Outer Race Fault

```
Analyze envelope spectrum of real_train/OuterRaceFault_1.csv at 97656 Hz.
Compare peaks with BPFO = 81.13 Hz.
```

### Example 2: Train ML Model

```
Train anomaly model on real_train/baseline_1.csv and real_train/baseline_2.csv.
Validate on real_train/OuterRaceFault_1.csv.
```

### Example 3: Compare Healthy vs Faulty

```
Compare FFT spectra of real_train/baseline_1.csv and real_train/OuterRaceFault_1.csv
at 97656 Hz sampling rate.
```

📚 **Full workflows**: See [EXAMPLES.md](EXAMPLES.md) for complete diagnostic procedures  
🚀 **Quick start guide**: See [QUICKSTART.md](QUICKSTART.md) for step-by-step tutorials

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

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Sample data is licensed CC BY-NC-SA 4.0 (non-commercial). For commercial use, replace with your own machinery data.

## Citation

If you use this server in your research or projects:

```bibtex
@software{machinery_diagnostics_mcp,
  title = {Predictive Maintenance MCP Server},
  author = {Di Maggio, Luigi},
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
- **Discussions**: https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions

---

**Built with ❤️ for condition monitoring professionals**
