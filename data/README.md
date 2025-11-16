# 🔊 Real Bearing Vibration Dataset

This directory contains **production-quality bearing vibration data** from real machinery tests - ready for immediate analysis, ML training, and fault detection demonstrations.

## ✨ What's Included

- **21 high-quality vibration signals** with varying sampling rates and durations
- **3 fault types**: Healthy baselines, inner race faults, outer race faults
- **Train/test split**: Pre-organized for ML workflow
  - **Training set**: 2 healthy, 5 inner race faults, 7 outer race faults
  - **Test set**: 1 healthy, 2 inner race faults, 3 outer race faults
- **Complete metadata**: Each signal has JSON file with sampling rate, duration, bearing frequencies, load conditions
- **Professional analysis ready**: Works with all MCP diagnostic tools

Perfect for:
- 🎓 Learning predictive maintenance techniques
- 🔬 Testing diagnostic algorithms
- 🤖 Training ML anomaly detection models
- 📊 Generating professional analysis reports
- 🚀 Demonstrating MCP server capabilities

## 📁 Directory Structure

- **`signals/`** - Processed CSV signals ready for analysis (exposed via MCP resources)
  - `real_train/` - Training dataset (2 healthy + 12 faulty signals)
  - `real_test/` - Test dataset for validation (1 healthy + 5 faulty signals)
- **`real_bearings/`** - Source MAT files from MathWorks (archive only, not used by MCP server)
  - `train/` - Original MATLAB .mat files
  - `test/` - Original MATLAB .mat files

> **Note**: The MCP server only uses CSV files in `signals/` directory. The `real_bearings/` folder is kept as source archive.

## 📊 Dataset Information

**Source**: [MathWorks RollingElementBearingFaultDiagnosis-Data](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)  
**License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike 4.0 International)

### ⚠️ License Summary

This data is licensed under **CC BY-NC-SA 4.0**, which means:

✅ **You CAN:**
- Use for learning, research, and educational purposes
- Share and redistribute the data
- Adapt and build upon the data

❌ **You CANNOT:**
- Use for commercial purposes without separate licensing
- Distribute derivative works under different license terms

📄 **Full License**: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

### 📝 Citation

When using this data, please cite:

```
The MathWorks, Inc. (2023). Rolling Element Bearing Fault Diagnosis Dataset.
GitHub Repository: https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data
License: CC BY-NC-SA 4.0
```

## 📁 Available Signals

### Training Set (`real_train/`) - 14 signals

**Healthy Baselines (2 files)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `baseline_1.csv` | 97,656 Hz | 6.0s | 585,936 |
| `baseline_2.csv` | 97,656 Hz | 6.0s | 585,936 |

**Inner Race Faults (5 files)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `InnerRaceFault_vload_1.csv` | 48,828 Hz | 3.0s | 146,484 |
| `InnerRaceFault_vload_2.csv` | 48,828 Hz | 3.0s | 146,484 |
| `InnerRaceFault_vload_3.csv` | 48,828 Hz | 3.0s | 146,484 |
| `InnerRaceFault_vload_4.csv` | 48,828 Hz | 3.0s | 146,484 |
| `InnerRaceFault_vload_5.csv` | 48,828 Hz | 3.0s | 146,484 |

**Outer Race Faults (7 files)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `OuterRaceFault_1.csv` | 97,656 Hz | 6.0s | 585,936 |
| `OuterRaceFault_2.csv` | 97,656 Hz | 6.0s | 585,936 |
| `OuterRaceFault_vload_1.csv` | 48,828 Hz | 3.0s | 146,484 |
| `OuterRaceFault_vload_2.csv` | 48,828 Hz | 3.0s | 146,484 |
| `OuterRaceFault_vload_3.csv` | 48,828 Hz | 3.0s | 146,484 |
| `OuterRaceFault_vload_4.csv` | 48,828 Hz | 3.0s | 146,484 |
| `OuterRaceFault_vload_5.csv` | 48,828 Hz | 3.0s | 146,484 |

### Test Set (`real_test/`) - 7 signals

**Healthy Baseline (1 file)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `baseline_3.csv` | 97,656 Hz | 6.0s | 585,936 |

**Inner Race Faults (2 files)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `InnerRaceFault_vload_6.csv` | 48,828 Hz | 3.0s | 146,484 |
| `InnerRaceFault_vload_7.csv` | 48,828 Hz | 3.0s | 146,484 |

**Outer Race Faults (3 files)**
| File | Sampling Rate | Duration | Samples |
|------|--------------|----------|---------|
| `OuterRaceFault_3.csv` | 97,656 Hz | 6.0s | 585,936 |
| `OuterRaceFault_vload_6.csv` | 48,828 Hz | 3.0s | 146,484 |
| `OuterRaceFault_vload_7.csv` | 48,828 Hz | 3.0s | 146,484 |

> **Note**: Sampling rates and durations vary between signals. All parameters are stored in corresponding `*_metadata.json` files and automatically detected by the MCP server.

## 🔧 Signal Specifications

- **Format**: CSV (single column, no header)
- **Units**: Acceleration (g)
- **Sampling Rates**: 97,656 Hz or 48,828 Hz (varies by signal)
- **Durations**: 3.0s or 6.0s (varies by signal)
- **Data Points**: 146,484 or 585,936 samples (varies by signal)

> **Important**: All signal parameters (sampling rate, duration, samples) are stored in corresponding `*_metadata.json` files and automatically detected by the MCP server. Do not assume fixed values - always check metadata!

### Bearing Characteristic Frequencies

| Frequency | Value (Hz) | Description |
|-----------|------------|-------------|
| **Shaft Speed** | 25.0 Hz | Rotation frequency |
| **FTF** | 14.84 Hz | Fundamental Train Frequency (cage) |
| **BSF** | 63.91 Hz | Ball Spin Frequency |
| **BPFO** | 81.13 Hz | Ball Pass Frequency Outer Race |
| **BPFI** | 118.88 Hz | Ball Pass Frequency Inner Race |

## 📊 Analysis Workflow

The MCP server provides comprehensive diagnostic tools that automatically detect signal parameters from metadata files. All analysis tools generate **interactive HTML reports** with Plotly visualizations.

### Available Report Types

| Report Type | Tool | Description | Output Location |
|-------------|------|-------------|-----------------|
| **FFT Analysis** | `generate_fft_report()` | Frequency spectrum analysis with peak detection | `reports/fft_*.html` |
| **Envelope Spectrum** | `generate_envelope_report()` | Bearing fault detection with modulation analysis | `reports/envelope_*.html` |
| **ISO 20816-3** | `generate_iso_report()` | Vibration severity assessment and zone classification | `reports/iso_*.html` |

### Typical Workflow

```
1. List available signals → list_signals()
2. Generate analysis report → generate_fft_report(signal_file)
3. Review interactive HTML → Open in browser (zoom, pan, hover)
4. Train ML model → train_anomaly_model() with healthy baselines
5. Detect anomalies → predict_anomalies() on new signals
```

### Key Features

- ✅ **Automatic parameter detection** - Sampling rates, durations, and frequencies read from metadata
- ✅ **Interactive visualizations** - Plotly charts with zoom, pan, hover capabilities
- ✅ **Professional reports** - HTML format suitable for documentation and sharing
- ✅ **ML-ready** - Pre-split train/test sets for anomaly detection workflows

## 📚 Metadata Files

Each `.csv` signal has a corresponding `*_metadata.json` file containing:

```json
{
  "sampling_rate": 97656.0,
  "signal_unit": "g",
  "shaft_speed": 25.0,
  "load": 270.0,
  "BPFI": 118.875,
  "BPFO": 81.125,
  "FTF": 14.8375,
  "BSF": 63.91,
  "num_samples": 585936,
  "duration_sec": 6.0
}
```

**Usage**: These files provide all necessary parameters for analysis (no need to manually enter frequencies!).

## ⚠️ Usage Notes

### For Academic/Research Use ✅
- ✅ Free to use for learning, research, education
- ✅ Cite the MathWorks repository in publications
- ✅ Share derivative works under CC BY-NC-SA 4.0

### For Commercial Use ❌
- ❌ **Not permitted** under CC BY-NC-SA 4.0 license without separate licensing
- ✅ This MCP server (MIT license) can be used commercially, but **replace sample signals** with your own data

### Recommended Approach for Commercial Projects

1. **Development/Testing**: Use these sample signals freely
2. **Production Deployment**: Replace with your own vibration data or obtain commercial license from MathWorks
3. **MCP Server Code**: MIT licensed, use freely in commercial projects
4. **Sample Data**: For demonstration and educational purposes only

## 🎓 Citation

If you use this data in research or publications, please cite:

```
The MathWorks, Inc. (2023). Rolling Element Bearing Fault Diagnosis Dataset.
GitHub Repository: https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data
License: CC BY-NC-SA 4.0
```

## 📖 Additional Resources

- [MathWorks Repository](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data) - Dataset source
- [MathWorks Predictive Maintenance Toolbox](https://www.mathworks.com/help/predmaint/) - MATLAB examples
- [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/) - Full license terms

---

**Note**: This MCP server is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc. Sample data is provided under CC BY-NC-SA 4.0 license for educational and non-commercial demonstration purposes only.
