# 🔊 Real Bearing Vibration Dataset

This directory contains **production-quality bearing vibration data** from real machinery tests - ready for immediate analysis, ML training, and fault detection demonstrations.

## ✨ What's Included

- **15 high-quality vibration signals** (6 seconds each @ 97.6 kHz)
- **3 fault types**: Healthy baselines, inner race faults, outer race faults
- **Train/test split**: Pre-organized for ML workflow
- **Complete metadata**: Sampling rates, bearing frequencies, load conditions
- **Professional analysis ready**: Works with all MCP diagnostic tools

Perfect for:
- 🎓 Learning predictive maintenance techniques
- 🔬 Testing diagnostic algorithms
- 🤖 Training ML anomaly detection models
- 📊 Generating professional analysis reports
- 🚀 Demonstrating MCP server capabilities

## 📁 Directory Structure

- **`signals/`** - Processed CSV signals ready for analysis (exposed via MCP resources)
  - `real_train/` - Training dataset (healthy + faults)
  - `real_test/` - Test dataset for validation
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

### Training Set (`real_train/`)

| File | Type | Fault Location | Load (lbs) | Samples | Duration |
|------|------|----------------|------------|---------|----------|
| `baseline_1.csv` | Healthy | None | 270 | 585,936 | 6.0s |
| `baseline_2.csv` | Healthy | None | 270 | 585,936 | 6.0s |
| `OuterRaceFault_1.csv` | Fault | Outer Race | 270 | 585,936 | 6.0s |
| `OuterRaceFault_2.csv` | Fault | Outer Race | 270 | 585,936 | 6.0s |
| `InnerRaceFault_vload_1.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |
| `InnerRaceFault_vload_2.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |
| `InnerRaceFault_vload_3.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |
| `InnerRaceFault_vload_4.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |
| `InnerRaceFault_vload_5.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |

### Test Set (`real_test/`)

| File | Type | Fault Location | Load (lbs) | Samples | Duration |
|------|------|----------------|------------|---------|----------|
| `baseline_3.csv` | Healthy | None | 270 | 585,936 | 6.0s |
| `OuterRaceFault_3.csv` | Fault | Outer Race | 270 | 585,936 | 6.0s |
| `InnerRaceFault_vload_6.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |
| `InnerRaceFault_vload_7.csv` | Fault | Inner Race | Variable | 585,936 | 6.0s |

## 🔧 Signal Specifications

- **Sampling Rate**: 97,656 Hz (~98 kHz)
- **Duration**: 6.0 seconds per signal
- **Data Points**: 585,936 samples per file
- **Format**: CSV (single column, no header)
- **Units**: Acceleration (g)
- **Bearing Type**: Deep groove ball bearing
- **Shaft Speed**: 25 Hz (1500 RPM)
- **Load**: 270 lbs (constant) or variable

### Bearing Characteristic Frequencies

| Frequency | Value (Hz) | Description |
|-----------|------------|-------------|
| **Shaft Speed** | 25.0 Hz | Rotation frequency |
| **FTF** | 14.84 Hz | Fundamental Train Frequency (cage) |
| **BSF** | 63.91 Hz | Ball Spin Frequency |
| **BPFO** | 81.13 Hz | Ball Pass Frequency Outer Race |
| **BPFI** | 118.88 Hz | Ball Pass Frequency Inner Race |

## � Analysis Workflow

The MCP server generates **professional analysis reports** in two formats:

### Output Formats

1. **📊 HTML Reports** (Interactive)
   - Plotly-based interactive charts
   - Zoom, pan, hover for detailed inspection
   - Ideal for analysis and exploration
   - Location: `reports/` directory

2. **🖼️ PNG Images** (Static)
   - High-resolution static images (1200x800px)
   - Perfect for documentation, presentations, GitHub README
   - Same filename as HTML with `.png` extension
   - Generated automatically alongside HTML

### Available Report Types

| Report Type | Tool | Description | Output Files |
|-------------|------|-------------|--------------|
| **FFT Analysis** | `generate_fft_report()` | Frequency spectrum analysis | `fft_*.html`, `fft_*.png` |
| **Envelope Spectrum** | `generate_envelope_report()` | Bearing fault detection | `envelope_*.html`, `envelope_*.png` |
| **ISO 20816-3** | `generate_iso_report()` | Vibration severity assessment | `iso_*.html`, `iso_*.png` |

### Typical Workflow

```
1. List available signals → list_signals()
2. Generate analysis report → generate_fft_report(signal_file, ...)
3. Review interactive HTML → Open in browser
4. Use PNG in documentation → Perfect for README, reports
5. Share/collaborate → HTML for interactivity, PNG for static viewing
```

## �🚀 Quick Start Examples

### Example 1: Detect Outer Race Fault (Envelope Analysis)

In Claude Desktop:
```
Generate envelope spectrum report for real_train/OuterRaceFault_1.csv.
Use bandpass filter 500-5000 Hz and mark BPFO = 81.13 Hz.
```

**Output**:
- 📊 `reports/envelope_OuterRaceFault_1_YYYYMMDD_HHMMSS.html` - Interactive Plotly chart
- 🖼️ `reports/envelope_OuterRaceFault_1_YYYYMMDD_HHMMSS.png` - Static image for documentation

**Expected Result**: ✅ Strong peak at ~81 Hz with harmonics at 162 Hz, 243 Hz (outer race fault confirmed)

### Example 2: Compare Healthy vs Faulty (FFT Analysis)

```
Generate FFT report for real_train/baseline_1.csv and real_train/OuterRaceFault_1.csv.
Compare frequency content.
```

**Output**:
- 📊 `reports/fft_baseline_1_YYYYMMDD_HHMMSS.html` - Healthy baseline spectrum
- 📊 `reports/fft_OuterRaceFault_1_YYYYMMDD_HHMMSS.html` - Faulty bearing spectrum
- 🖼️ `.png` versions for both (ideal for README, presentations, reports)

**Expected Result**: ✅ Faulty signal shows elevated high-frequency content and modulation sidebands

### Example 3: Train ML Anomaly Detector

```
Train anomaly detection model:
- Training: real_train/baseline_1.csv, real_train/baseline_2.csv
- Validation (healthy): real_test/baseline_3.csv
- Validation (fault): real_train/OuterRaceFault_1.csv, real_train/InnerRaceFault_vload_1.csv
- Model: OneClassSVM
- Model name: "bearing_health_model"
```

**Expected Result**: 
- ✅ Model detects >90% of fault segments (high sensitivity)
- ✅ <10% false positives on healthy data (high specificity)
- ✅ Validation metrics show balanced accuracy >90%

**Note**: The tool automatically:
- Detects sampling rates from metadata files
- Uses 0.1s segments with 50% overlap for ML
- Validates on BOTH healthy and fault data for comprehensive evaluation

### Example 4: ISO 20816-3 Compliance Check

```
Generate ISO 20816-3 evaluation report for real_train/baseline_1.csv.
Machine: Group 2 (medium, 100 kW), rigid foundation.
```

**Output**:
- 📊 `reports/iso_baseline_1_YYYYMMDD_HHMMSS.html` - Interactive severity chart
- 🖼️ `reports/iso_baseline_1_YYYYMMDD_HHMMSS.png` - Static severity visualization

**Expected Result**: ✅ Zone A or B (acceptable vibration level)

## 📚 Metadata Files

Each `.csv` signal has a corresponding `*_metadata.json` file containing:

```json
{
  "sampling_rate": 97656.0,
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

## 💡 Pro Tips

### Working with Reports

1. **Interactive Analysis** 📊
   - Open `.html` files in any browser
   - Use Plotly controls: zoom (drag box), pan (click+drag), reset (home icon)
   - Hover over peaks to see exact frequency and amplitude
   - Save custom views or screenshots

2. **Documentation & Sharing** 🖼️
   - Use `.png` files in GitHub README, wiki, documentation
   - Perfect for presentations, technical reports, publications
   - High resolution (1200x800px) suitable for printing
   - Consistent professional appearance

3. **Batch Analysis** 🔄
   ```
   Generate FFT reports for all training signals:
   real_train/baseline_1.csv, baseline_2.csv, OuterRaceFault_1.csv
   ```
   - Compare multiple signals side-by-side
   - Identify patterns across dataset
   - Build comprehensive diagnostic library

4. **ML Model Validation** 🤖
   - Use `real_train/` for training (healthy baselines)
   - Use `real_test/` for validation (separate test set)
   - Include explicit `healthy_validation_files` for proper train/test split
   - Evaluate both specificity (healthy) and sensitivity (fault detection)

### Best Practices

- ✅ Always check metadata files for correct sampling rates
- ✅ Use bandpass filters 500-5000 Hz for envelope analysis
- ✅ Compare against theoretical bearing frequencies (BPFO, BPFI)
- ✅ Validate ML models on BOTH healthy and fault signals
- ✅ Keep HTML reports for interactive analysis, PNG for documentation
- ✅ Organize reports by date/analysis type in `reports/` directory

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

