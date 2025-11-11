# Machinery Diagnostics - Complete Examples

This guide provides step-by-step examples of complete diagnostic workflows using the Machinery Diagnostics MCP Server.

## Table of Contents

- [Example 1: Simple FFT Analysis](#example-1-simple-fft-analysis)
- [Example 2: Bearing Fault Detection](#example-2-bearing-fault-detection)
- [Example 3: ISO 20816-3 Compliance Check](#example-3-iso-20816-3-compliance-check)
- [Example 4: Complete Bearing Diagnosis](#example-4-complete-bearing-diagnosis)
- [Example 5: Working with Different Segment Durations](#example-5-working-with-different-segment-durations)
- [Example 6: Machine Learning-Based Anomaly Detection](#example-6-machine-learning-based-anomaly-detection)

---

## Example 1: Simple FFT Analysis

### Objective
Analyze the frequency content of a vibration signal to identify dominant frequencies.

### Scenario
A pump is showing increased vibration. You have captured a vibration signal and want to identify the frequency components.

### Step-by-Step

**Step 1: Prepare the signal file**

Ensure your CSV file is in the `data/signals/` directory:
```
data/signals/pump_vibration.csv
```

**Step 2: Run FFT analysis**

In Claude Desktop, use this prompt:
```
Analyze the FFT spectrum of pump_vibration.csv with sampling rate 10000 Hz
```

**Step 3: Interpret results**

The tool will return:
```json
{
  "peaks": [
    {"frequency": 50.0, "magnitude": 0.85},
    {"frequency": 100.0, "magnitude": 0.32},
    {"frequency": 150.0, "magnitude": 0.18}
  ],
  "frequency_resolution": 1.0,
  "sampling_rate": 10000,
  "segment_duration": 1.0
}
```

**Interpretation:**
- **50 Hz**: Dominant peak likely corresponds to shaft rotation (3000 RPM)
- **100 Hz, 150 Hz**: Harmonics of shaft frequency
- **Frequency resolution**: 1 Hz (adequate for most applications)

**Step 4: Generate visualization**

Request an interactive chart:
```
Generate an interactive FFT chart for pump_vibration.csv
```

This opens an HTML chart where you can:
- Zoom into frequency regions of interest
- Hover over peaks to see exact values
- Export the chart for reports

### Expected Outcome
✅ Identified shaft rotation frequency and harmonics  
✅ No unusual frequencies detected  
✅ Vibration appears normal

---

## Example 2: Bearing Fault Detection

### Objective
Detect and localize bearing faults using envelope analysis.

### Scenario
A motor bearing is suspected of having a defect. The motor runs at 1800 RPM (30 Hz shaft frequency), and the bearing has known characteristic frequencies.

### Bearing Specifications
- **BPFO** (Ball Pass Frequency Outer): 120.5 Hz
- **BPFI** (Ball Pass Frequency Inner): 198.2 Hz
- **BSF** (Ball Spin Frequency): 89.3 Hz
- **FTF** (Fundamental Train Frequency): 12.1 Hz

### Step-by-Step

**Step 1: Perform envelope analysis**

```
Perform envelope analysis on motor_bearing.csv with:
- Sampling rate: 12000 Hz
- Filter band: 500 to 5000 Hz
```

**Step 2: Examine envelope spectrum**

The tool returns:
```json
{
  "envelope_peaks": [
    {"frequency": 120.5, "magnitude": 0.92},
    {"frequency": 241.0, "magnitude": 0.48},
    {"frequency": 361.5, "magnitude": 0.25}
  ],
  "filter_band": [500, 5000],
  "dominant_frequency": 120.5
}
```

**Step 3: Compare with bearing frequencies**

| Detected Peak | Theoretical | Match | Fault Type |
|--------------|-------------|-------|------------|
| 120.5 Hz | 120.5 Hz (BPFO) | ✅ | Outer race fault |
| 241.0 Hz | 241.0 Hz (2×BPFO) | ✅ | Harmonic |
| 361.5 Hz | 361.5 Hz (3×BPFO) | ✅ | Harmonic |

**Step 4: Generate envelope chart**

```
Generate envelope chart for motor_bearing.csv with:
- Sampling rate: 12000 Hz
- Filter band: 500 to 5000 Hz
- Highlight frequencies: [120.5, 241.0, 361.5]
- Labels: ["BPFO", "2×BPFO", "3×BPFO"]
```

### Diagnosis
🔴 **OUTER RACE BEARING FAULT DETECTED**

**Evidence:**
- Strong peak at BPFO (120.5 Hz)
- Clear harmonics (2×, 3×)
- Pattern consistent with localized outer race defect

**Recommendation:**
- Schedule bearing replacement
- Monitor vibration levels weekly
- Avoid operating at high loads

### Expected Outcome
✅ Bearing fault localized to outer race  
✅ Severity assessed (moderate to severe)  
✅ Actionable maintenance plan created

---

## Example 3: ISO 20816-3 Compliance Check

### Objective
Evaluate vibration severity according to international standard ISO 20816-3.

### Scenario
A large industrial pump (400 kW, rigid foundation) needs vibration assessment for compliance reporting.

### Machine Specifications
- **Power**: 400 kW → Group 1 (large machine)
- **Foundation**: Rigid concrete → Support type: rigid
- **Operating speed**: 1500 RPM (25 Hz)

### Step-by-Step

**Step 1: Evaluate against ISO standard**

```
Evaluate pump_vibration.csv against ISO 20816-3 with:
- Sampling rate: 10000 Hz
- Machine group: 1
- Support type: rigid
```

**Step 2: Review results**

```json
{
  "rms_velocity": 3.2,
  "unit": "mm/s",
  "severity_zone": "B",
  "machine_group": 1,
  "support_type": "rigid",
  "thresholds": {
    "zone_A_B": 2.3,
    "zone_B_C": 4.5,
    "zone_C_D": 7.1
  },
  "interpretation": "Acceptable for long-term operation"
}
```

**Step 3: Interpret zones**

```
Zone A (🟢): 0 - 2.3 mm/s     → New machine condition
Zone B (🟡): 2.3 - 4.5 mm/s   → Acceptable (CURRENT: 3.2 mm/s)
Zone C (🟠): 4.5 - 7.1 mm/s   → Unsatisfactory
Zone D (🔴): > 7.1 mm/s       → Unacceptable
```

**Step 4: Document for compliance**

Create report:
```
Vibration Assessment Report
---------------------------
Machine: Industrial Pump (400 kW)
Standard: ISO 20816-3
Date: [Current Date]

Measurement:
- RMS Velocity: 3.2 mm/s
- Severity Zone: B (Yellow)
- Status: ACCEPTABLE

Classification:
- Machine Group: 1 (Large, >300 kW)
- Support Type: Rigid
- Zone B Threshold: 2.3 - 4.5 mm/s

Recommendation:
✅ Machine acceptable for continuous operation
✅ Monitor vibration quarterly
✅ No immediate action required
```

### Expected Outcome
✅ Compliance documented  
✅ Machine within acceptable limits  
✅ Monitoring schedule established

---

## Example 4: Complete Bearing Diagnosis

### Objective
Perform comprehensive bearing diagnostics using the guided workflow prompt.

### Scenario
A compressor bearing shows signs of wear. Need complete diagnostic workup with severity assessment and recommendations.

### Machine Information
- **Type**: Centrifugal compressor
- **Power**: 250 kW → Group 2 (medium machine)
- **Support**: Flexible coupling → Support type: flexible
- **Speed**: 3600 RPM → Shaft frequency: 60 Hz
- **Bearing**: SKF 6316
  - BPFO: 156.3 Hz
  - BPFI: 243.7 Hz
  - BSF: 98.6 Hz
  - FTF: 25.5 Hz

### Step-by-Step

**Step 1: Initiate guided diagnosis**

```
Run complete bearing diagnosis on compressor_bearing.csv with:
- Sampling rate: 12000 Hz
- Shaft frequency: 60 Hz
- Bearing frequencies:
  BPFO: 156.3 Hz
  BPFI: 243.7 Hz
  BSF: 98.6 Hz
  FTF: 25.5 Hz
- Machine group: 2
- Support type: flexible
```

**Step 2: Workflow executes automatically**

The `diagnose_bearing` prompt performs 6 steps:

```
📋 STEP 1: Parameter Validation
✅ All required parameters present
✅ Bearing frequencies verified

📊 STEP 2: ISO 20816-3 Severity Evaluation
User confirms machine parameters:
- Group 2: Medium machine (15-300 kW) ✅
- Support: Flexible ✅

ISO Result:
- RMS Velocity: 5.8 mm/s
- Zone: C (Orange)
- Interpretation: Unsatisfactory, plan maintenance

🔍 STEP 3: FFT Analysis
Primary frequencies detected:
- 60 Hz: Shaft rotation (3600 RPM)
- 120 Hz: 2× shaft (normal)
- 243.7 Hz: BPFI (ALERT!)

📈 STEP 4: Envelope Analysis
Envelope spectrum peaks:
- 243.7 Hz: Strong peak (BPFI)
- 487.4 Hz: 2×BPFI harmonic
- 731.1 Hz: 3×BPFI harmonic

🎯 STEP 5: Fault Pattern Recognition
Pattern Match:
- Peak at BPFI: ✅ CONFIRMED
- Harmonics present: ✅ YES (2×, 3×)
- No shaft modulation: ✅ Localized defect

Diagnosis: INNER RACE BEARING FAULT

💡 STEP 6: Recommendations

FAULT DETECTED: Inner Race Bearing Defect
Severity: HIGH (Zone C + Clear BPFI pattern)

Immediate Actions:
1. Schedule bearing replacement within 1-2 weeks
2. Reduce operating speed if possible
3. Increase monitoring frequency to daily
4. Prepare for planned shutdown

Long-term:
5. Investigate root cause (lubrication, misalignment, overload)
6. Improve preventive maintenance procedures
7. Consider vibration monitoring system installation
```

### Diagnosis Summary

| Aspect | Finding | Severity |
|--------|---------|----------|
| ISO 20816-3 | Zone C (5.8 mm/s) | 🟠 Unsatisfactory |
| FFT Analysis | BPFI at 243.7 Hz | 🔴 Fault detected |
| Envelope | Strong BPFI + harmonics | 🔴 Advanced stage |
| Fault Type | Inner race defect | 🔴 Critical |
| Recommendation | Replace within 1-2 weeks | ⚠️ Urgent |

### Expected Outcome
✅ Comprehensive diagnosis completed  
✅ Fault localized to inner race  
✅ Severity assessed (high/urgent)  
✅ Detailed action plan provided  
✅ Root cause investigation initiated

---

## Example 5: Working with Different Segment Durations

### Objective
Understand when and how to use different segment durations for optimal results.

### Scenario Comparison

#### Scenario A: High-Speed Bearing (Fast Processing Needed)

**Requirements:**
- Bearing speed: 10,000 RPM (166.7 Hz)
- BPFO: ~500 Hz
- Need quick screening of multiple signals

**Solution: Use default 1.0s segments**

```
analyze_fft
file_path: high_speed_bearing.csv
sampling_rate: 20000
segment_duration: 1.0
```

**Results:**
- Frequency resolution: ~1 Hz (adequate for 500 Hz detection)
- Processing time: Fast (10× faster than full signal)
- Output size: ~25 KB
- ✅ Perfect for quick screening

---

#### Scenario B: Low-Speed Machine (Better Resolution Needed)

**Requirements:**
- Shaft speed: 300 RPM (5 Hz)
- Need to detect 0.5 Hz frequency differences
- Gear mesh analysis requires high resolution

**Solution: Use longer 5.0s segments**

```
analyze_fft
file_path: low_speed_gear.csv
sampling_rate: 10000
segment_duration: 5.0
```

**Results:**
- Frequency resolution: ~0.2 Hz (excellent for low frequencies)
- Processing time: Moderate
- Output size: ~40 KB
- ✅ Excellent resolution for low-speed analysis

---

#### Scenario C: Research/Detailed Analysis (Maximum Detail)

**Requirements:**
- Complete frequency information needed
- Preparing for publication or detailed report
- No performance constraints

**Solution: Use full signal**

```
analyze_fft
file_path: research_signal.csv
sampling_rate: 10000
segment_duration: None
```

**Results:**
- Frequency resolution: Maximum (depends on signal length)
- Processing time: Slowest
- Output size: Largest (~40-100 KB)
- ✅ Maximum detail for research

---

### Segment Duration Recommendations

| Application | Segment Duration | Reason |
|-------------|------------------|--------|
| **Quick screening** | 1.0s (default) | Fast, adequate resolution |
| **Routine monitoring** | 1.0s | Balances speed and accuracy |
| **Bearing diagnostics** | 1.0-2.0s | Good resolution for BPFO/BPFI |
| **Low-speed machines** | 2.0-5.0s | Better low-frequency resolution |
| **Gear analysis** | 2.0-5.0s | Resolves sidebands |
| **Research/validation** | None (full) | Maximum information |
| **Automated systems** | 1.0s | Fast batch processing |

---

## Example 6: Machine Learning-Based Anomaly Detection

### Objective
Train an ML model on healthy machine data to automatically detect anomalies in new measurements.

### Scenario
You have collected vibration data from a motor fleet. Some motors are known to be healthy. You want to build a model that can automatically detect when a motor starts to develop a fault.

### Step-by-Step

**Step 1: Prepare training data**

Collect signals from healthy motors:
```
data/signals/healthy_motor_1.csv
data/signals/healthy_motor_2.csv
data/signals/healthy_motor_3.csv
data/signals/healthy_motor_4.csv
```

Optionally, prepare fault data for validation:
```
data/signals/fault_motor_1.csv
data/signals/fault_motor_2.csv
```

**Step 2: Extract features and train model**

In Claude Desktop:
```
Train an anomaly detection model on these healthy signals:
- healthy_motor_1.csv
- healthy_motor_2.csv
- healthy_motor_3.csv
- healthy_motor_4.csv

Use OneClassSVM with PCA variance 0.95 and validate on:
- fault_motor_1.csv
- fault_motor_2.csv

Save the model as "motor_fleet_model"
```

**Step 3: Review training results**

The tool will return:
```json
{
  "model_path": "models/motor_fleet_model_model.pkl",
  "scaler_path": "models/motor_fleet_model_scaler.pkl",
  "pca_path": "models/motor_fleet_model_pca.pkl",
  "training_samples": 2400,
  "pca_components": 8,
  "pca_variance_explained": 0.956,
  "validation_accuracy": 0.98,
  "validation_details": {
    "healthy_correct": "100% (1200/1200)",
    "fault_detected": "96% (576/600)"
  }
}
```

**Interpretation:**
- ✅ Model trained on 2,400 segments from healthy motors
- ✅ PCA reduced 17 features to 8 components (95.6% variance)
- ✅ Validation: 98% accuracy
  - Healthy data: 100% correctly classified (no false alarms)
  - Fault data: 96% detected (24 missed faults out of 600 segments)

**Step 4: Predict anomalies in new motor**

Now test a new motor:
```
Predict anomalies in motor_unknown_5.csv using motor_fleet_model
```

**Step 5: Review predictions**

Results:
```json
{
  "num_segments": 600,
  "anomaly_count": 42,
  "anomaly_ratio": 0.07,
  "overall_health": "Healthy",
  "confidence": "High",
  "predictions": [1, 1, 1, ..., -1, -1, ...],
  "anomaly_scores": [-0.23, -0.18, -0.15, ..., 0.82, 0.91, ...]
}
```

**Interpretation:**
- ✅ 600 segments analyzed
- ✅ 42 anomalous segments detected (7% of signal)
- ✅ **Overall: Healthy** (< 10% anomalies)
- 🔵 **Action**: Continue normal operation, monitor weekly

**Step 6: Monitor over time**

Track anomaly ratio trends:
- **Week 1**: 7% anomalies → Healthy
- **Week 2**: 12% anomalies → Suspicious
- **Week 3**: 28% anomalies → Suspicious  
  → Schedule inspection within 2 weeks
- **Week 4**: 45% anomalies → **Faulty**  
  → **Replace bearing immediately**

### Expected Outcome

✅ **Automated monitoring system** that:
- Detects anomalies without manual feature engineering
- Provides early warning (weeks before failure)
- Scales to large motor fleets
- Requires only healthy baseline data

### When to Use ML Approach

| Use Case | Traditional Analysis | ML Approach |
|----------|---------------------|-------------|
| **Single machine diagnosis** | ✅ Better (interpretable) | ⚠️ Overkill |
| **Fleet monitoring (>10 machines)** | ⚠️ Labor intensive | ✅ Automated |
| **Trend detection** | ⚠️ Manual comparison | ✅ Automatic alerts |
| **Unknown fault patterns** | ❌ May miss novel faults | ✅ Detects deviations |
| **Root cause diagnosis** | ✅ Clear (FFT, Envelope) | ❌ Black box |

**Recommendation:** Use ML for **screening and monitoring**, then use traditional analysis (FFT, Envelope, ISO) for **root cause diagnosis** when anomalies are detected.

---

## Pro Tips

### Tip 1: Always Check Peak Alignment
After generating a chart, verify that red peak markers align precisely with spectrum lines. If not aligned, you may be using an old version—update to v0.2.0+.

### Tip 2: Use Bearing Frequency Calculators
Many online calculators can determine BPFO, BPFI, BSF, FTF from bearing geometry. Always verify these before diagnosis.

### Tip 3: Combine Multiple Analysis Methods
For best results, use:
1. **ISO 20816-3** → Overall severity
2. **FFT** → Primary frequencies
3. **Envelope** → Bearing fault localization

### Tip 4: Trend Analysis
Save analysis results over time to track:
- RMS velocity trends
- Peak amplitude changes
- Appearance of new frequencies

### Tip 5: Safety First
If ISO returns Zone D or envelope shows strong BPFI/BPFO:
- **Stop operation immediately**
- **Inspect bearing visually**
- **Do not restart until repaired**

---

## Next Steps

- **Practice**: Try these examples with your own signals
- **Learn**: Study the diagnostic reasoning in each example
- **Contribute**: Share your diagnostic workflows in [GitHub Discussions](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions)

---

**Need help?** Open an issue or start a discussion on GitHub!

**Found this useful?** ⭐ Star the repository to show support!
