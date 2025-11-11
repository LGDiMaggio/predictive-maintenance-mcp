# Sample Bearing Vibration Signals

This directory contains **real bearing vibration data** included with this MCP server for immediate testing and demonstration.

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

## 🚀 Quick Start Examples

### Example 1: Detect Outer Race Fault

In Claude Desktop:
```
Analyze envelope spectrum of real_train/OuterRaceFault_1.csv with 97656 Hz sampling rate.
Use filters 500-5000 Hz. Compare peaks with BPFO = 81.13 Hz.
```

**Expected Result**: ✅ Strong peak at ~81 Hz with harmonics at 162 Hz, 243 Hz (outer race fault confirmed)

### Example 2: Compare Healthy vs Faulty

```
Compare FFT spectra of real_train/baseline_1.csv and real_train/OuterRaceFault_1.csv, 
both at 97656 Hz sampling rate.
```

**Expected Result**: ✅ Faulty signal shows elevated high-frequency content

### Example 3: Train ML Anomaly Detector

```
Train anomaly detection model on:
- real_train/baseline_1.csv
- real_train/baseline_2.csv

Validate on:
- real_train/OuterRaceFault_1.csv
- real_train/InnerRaceFault_vload_1.csv

Use OneClassSVM with sampling rate 97656 Hz.
```

**Expected Result**: ✅ Model detects >90% of fault segments, <5% false positives on healthy data

### Example 4: ISO 20816-3 Compliance

```
Evaluate ISO 20816-3 severity for real_train/baseline_1.csv at 97656 Hz.
Machine is Group 2 (medium machine, 100 kW) with rigid foundation.
```

**Expected Result**: ✅ Zone A or B (acceptable vibration)

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

