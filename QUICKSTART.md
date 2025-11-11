# 🎯 Quick Start Guide - Machinery Diagnostics MCP Server

Get started with machinery diagnostics in under 5 minutes! This guide uses the real bearing dataset included with the server.

---

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
cd "C:\\path\\to\\predictive-maintenance-mcp"

# Install dependencies
uv sync

# Verify installation
uv run python tests/verify.py
```

### 2. Test the Server

```bash
# Run comprehensive tests with real bearing data
uv run python tests/test_real_data.py

# Or run the full test suite
uv run python tests/test_suite.py

# Start MCP Inspector for interactive testing
uv run mcp dev src/machinery_diagnostics_server.py
```

### 3. Claude Desktop Configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "machinery-diagnostics": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\predictive-maintenance-mcp",
        "run",
        "src/machinery_diagnostics_server.py"
      ]
    }
  }
}
```

**Restart Claude Desktop** to apply changes.

---

## 🎓 Real-World Dataset

The server includes a real bearing fault dataset for testing:

**Dataset:** Rolling Element Bearing Fault Diagnosis  
**Source:** [MathWorks Repository](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data)  
**License:** CC BY-NC-SA 4.0

**Available Files:**
- 📁 `data/signals/real_train/` - 14 training files (2 baseline, 5 inner fault, 7 outer fault)
- 📁 `data/signals/real_test/` - 6 test files (1 baseline, 2 inner fault, 3 outer fault)

**Signal Characteristics:**
- Acceleration signals (g units)
- Sampling rates: 97656 Hz (baseline) or 48828 Hz (vload files)
- Duration: 3-6 seconds per file
- Bearing frequencies: BPFI=118.875 Hz, BPFO=81.125 Hz, FTF=14.8375 Hz, BSF=63.91 Hz

---

## 📖 Usage Examples

### Example 1: List Available Signals

**Prompt in Claude Desktop:**
```
What signals do I have available for analysis?
```

**What happens:**
- Claude reads the `signal://list` resource
- Shows all files in the `data/signals/` directory
- Includes name, size, file type

---

### Example 2: Generate FFT Report for Baseline Signal

**Prompt:**
```
Generate an FFT report for real_train/baseline_1.csv
```

**What happens:**
1. Claude calls `generate_fft_report` tool
2. Server performs FFT analysis (sampling rate auto-detected: 97,656 Hz)
3. Creates HTML file in `reports/`:
   - **Interactive Plotly chart**: `fft_spectrum_real_train_baseline_1.html`

**Claude's response includes:**
```
✓ FFT spectrum report saved: fft_spectrum_real_train_baseline_1.html (150.9 KB)

Report Details:
- Signal: real_train/baseline_1.csv
- Sampling Rate: 97,656 Hz
- 15 peaks detected
- Dominant peak: 29.95 Hz (shaft frequency, 1797 RPM)

Open the HTML file in your browser for interactive exploration!
```

**Open the HTML file** in your browser to:
- Zoom into frequency regions
- Hover over peaks for exact values
- Identify shaft frequency harmonics

---

### Example 3: Detect Outer Race Fault

**Prompt:**
```
Generate envelope analysis report for real_train/OuterRaceFault_1.csv with bearing frequencies BPFO=81.13 Hz
```

**What happens:**
1. Claude calls `generate_envelope_report` tool
2. Server performs:
   - Bandpass filtering (2000-8000 Hz)
   - Hilbert transform envelope extraction
   - FFT of envelope signal
   - Peak detection with bearing frequency matching
3. Creates professional HTML report

**Claude's response:**
```
✓ Envelope analysis report saved: envelope_analysis_real_train_OuterRaceFault_1.html (199.8 KB)

Report Details:
- Signal: real_train/OuterRaceFault_1.csv
- Filter Band: 500-5000 Hz
- 10 peaks detected
- Bearing matches found: ≈ BPFO (81.13 Hz), ≈ BPFI

🔴 OUTER RACE FAULT DETECTED
- Strong peak at ~80 Hz (BPFO: 81.13 Hz)
- Clear harmonics present
- Severity: Advanced stage

Recommendation: Schedule bearing replacement within 1-2 weeks
```

**The HTML report shows:**
- Filtered signal and envelope in time domain
- Envelope spectrum with automatic BPFO markers
- Peak annotations with bearing frequency matches

---

### Example 4: ISO 20816-3 Compliance Check

**Prompt:**
```
Generate ISO 20816-3 report for real_train/baseline_1.csv with machine group 3 and rigid support
```

**What happens:**
1. Claude calls `generate_iso_report` tool
2. Server evaluates vibration severity per ISO 20816-3
3. Creates zone visualization HTML report

**Claude's response:**
```
✓ ISO 20816-3 report saved: iso_20816_real_train_baseline_1.html - Zone A (Good)

Evaluation Result:
- RMS Velocity: 1.8 mm/s
- Severity Zone: A (Green)
- Machine Group: 3 (Small machine)
- Support Type: Rigid

Status: NEW MACHINE CONDITION ✅
- Within acceptable limits
- Suitable as baseline reference
- No action required
```

**The report displays:**
- Color-coded zones (A=Green, B=Yellow, C=Orange, D=Red)
- Current RMS velocity marker
- Severity interpretation

---

### Example 5: Complete Bearing Diagnostic

**Prompt:**
```
Run complete bearing diagnosis on real_train/InnerRaceFault_vload_1.csv
```

**What happens:**
1. Claude uses the `diagnose_bearing` prompt (guided workflow)
2. Automatically performs 6 diagnostic steps:
   - Parameter validation
   - ISO 20816-3 severity evaluation
   - FFT analysis for frequency identification
   - Envelope analysis for fault localization
   - Fault pattern recognition
   - Recommendations generation
3. Generates all reports (FFT + Envelope + ISO)

**Claude's comprehensive response:**
```
📋 BEARING DIAGNOSTIC REPORT
Signal: real_train/InnerRaceFault_vload_1.csv

📊 ISO 20816-3 EVALUATION
- RMS Velocity: 4.2 mm/s
- Zone: C (Orange) - Unsatisfactory
- Action required: Plan maintenance

🔍 FFT ANALYSIS
- Shaft frequency: 28.83 Hz (1730 RPM)
- BPFI detected: 138.87 Hz ⚠️
- Harmonics: 2×BPFI, 3×BPFI present

📈 ENVELOPE ANALYSIS
- Strong peak at BPFI: 138.87 Hz (magnitude: 0.91)
- Harmonics confirmed: 2×, 3×BPFI
- Modulation sidebands at ±28.83 Hz (shaft frequency)

🎯 DIAGNOSIS: INNER RACE BEARING FAULT
- Severity: HIGH/URGENT
- Pattern: Localized defect with load-dependent modulation
- Zone C + Strong BPFI + Harmonics = Advanced stage

💡 RECOMMENDATIONS
Immediate (1 week):
- ⚠️ URGENT: Schedule bearing replacement
- Reduce speed and load immediately
- Implement daily monitoring
- Prepare spare bearing and crew

Long-term:
- Investigate root cause (lubrication, alignment, overload)
- Improve predictive maintenance procedures
- Consider automated monitoring system

📁 REPORTS GENERATED:
✓ fft_spectrum_real_train_InnerRaceFault_vload_1.html
✓ envelope_analysis_real_train_InnerRaceFault_vload_1.html
✓ iso_20816_real_train_InnerRaceFault_vload_1.html
```

---

## 💡 Pro Tips

### Tip 1: Explore Reports Interactively
- **HTML files**: Open in browser for interactive Plotly charts (zoom, hover, pan)
- View reports directly or drag them into your browser

### Tip 2: List All Generated Reports
```
Show me all generated reports
```
Claude calls `list_html_reports()` to show metadata without loading HTML content.

### Tip 3: View Report Details
```
Get details for envelope_analysis_real_train_OuterRaceFault_1.html
```
Claude calls `get_report_info()` to extract embedded metadata (signal info, peaks, bearing matches).

### Tip 4: Compare Baseline vs Fault
Generate reports for both:
- `real_train/baseline_1.csv` (healthy)
- `real_train/OuterRaceFault_1.csv` (faulty)

Compare HTML reports side-by-side in your browser to spot differences!

### Tip 5: Auto-Detection
The server automatically detects:
- Sampling rate (from `_metadata.json` files)
- Signal duration
- File format (acceleration vs velocity)

You don't need to specify sampling rate for real dataset files!

---

## 🎯 Next Steps

1. **Try the examples above** with different fault types
2. **Generate reports** for training and test datasets
3. **Compare spectra** between baseline and fault signals
4. **Read EXAMPLES.md** for advanced workflows
5. **Star the repository** ⭐ if you find it useful!

---

### Example 1: Complete Bearing Diagnostic with Real DataRun an FFT analysis on the test_bearing_fault_10000Hz.csv signal

```

**Prompt:**

```**Structured Output:**

Analyze the baseline bearing signal: real_train/baseline_1.csv```json

```{

  "frequencies": [0.0, 1.0, 2.0, ...],

**Output:**  "magnitudes": [0.05, 0.12, 0.82, ...],

```  "peak_frequency": 1200.5,

📊 STATISTICAL ANALYSIS  "peak_magnitude": 0.82,

- RMS: 0.89 g  "sampling_rate": 10000,

- Crest Factor: 5.22 (normal for rotating machinery)  "num_samples": 10000,

- Kurtosis: 0.01 (healthy, Gaussian distribution)  "frequency_resolution": 1.0

- Status: ✅ NORMAL}

```

📈 FFT ANALYSIS

- Sampling rate: 97656 Hz (auto-detected)---

- Peak frequency: 0.17 Hz

- Spectrum: Normal broadband noise### Example 4: Complete Bearing Diagnostic



🎯 ISO 20816-3 EVALUATION**Prompt:**

- RMS Velocity: 1.54 mm/s (auto-converted from acceleration)```

- Zone: B (Acceptable)Run a complete bearing diagnostic on the test_bearing_fault_10000Hz.csv file with 10000 Hz sampling frequency

- Machine Condition: Good```



💡 DIAGNOSIS: Healthy bearing, normal operating conditions**Automatic Workflow:**

```1. ✅ Statistical analysis (RMS, Crest Factor, Kurtosis)

2. ✅ FFT analysis for harmonics

---3. ✅ Envelope analysis for fault frequencies

4. ✅ Interpretation and diagnosis

### Example 2: Inner Race Fault Detection5. ✅ Action recommendations



**Prompt:****Example Output:**

``````

Analyze the inner race fault signal: real_train/InnerRaceFault_vload_1.csv📊 STATISTICAL ANALYSIS

```- RMS: 2.34 (moderate)

- Crest Factor: 6.2 ⚠️ (high, indicates impulses)

**Output:**- Kurtosis: 8.5 🔴 (very high, defect present)

```

📊 STATISTICAL ANALYSIS📈 FFT ANALYSIS

- RMS: 1.81 g (2× baseline)- Dominant peak: 120 Hz

- Crest Factor: 25.34 (5× baseline - HIGH!)- Harmonics present: 240, 360, 480 Hz

- Kurtosis: 48.05 (4800× baseline - CRITICAL!)

- Status: 🔴 ALARM🔍 ENVELOPE ANALYSIS

- Main peaks: 156.3 Hz, 312.6 Hz

📈 FFT ANALYSIS- Interpretation: BPFO (Ball Pass Frequency Outer race)

- Sampling rate: 48828 Hz (auto-detected from metadata)

- Peak frequencies: 118.5 Hz, 237 Hz, 356 Hz🎯 DIAGNOSIS

- Pattern: BPFI harmonics detectedStatus: 🔴 ALARM

Defect: Damaged bearing outer race

🔍 ENVELOPE ANALYSISSeverity: Critical

- Dominant frequency: 118.88 Hz (matches BPFI)

- Harmonics: 2×BPFI, 3×BPFI present💡 RECOMMENDATIONS

- Diagnosis: INNER RACE FAULT CONFIRMED- ⚠️ Urgent intervention required

- 🔧 Bearing replacement

🎯 ISO 20816-3 EVALUATION- 📅 Do not schedule, immediate intervention

- RMS Velocity: 1.62 mm/s```

- Zone: B (Acceptable for velocity, but...)

- Note: Acceleration metrics show severe fault!---



⚠️ CRITICAL FINDING:### Example 5: Quick Screening

- Kurtosis increase of 4800× indicates severe impacting

- Inner race fault with advanced damage**Prompt:**

- Immediate maintenance required!```

Do a quick screening of the motor_vibration.csv signal

💡 RECOMMENDATION:```

- 🔴 URGENT: Replace bearing immediately

- Monitor adjacent components for secondary damage**Output:**

- Do NOT continue operation```

```🔍 QUICK SCREENING - motor_vibration.csv



---📊 Parameters:

- Samples: 10000

### Example 3: Train ML Model on Baseline Data- Duration: 1.0 s

- Range: -3.2 to +3.5

**Prompt:**

```📈 Quick Analysis:

Train an anomaly detection model using the baseline signals- RMS: 1.85

```- Crest Factor: 3.8

- Kurtosis: 4.2

**What happens:**

1. Loads both baseline files (baseline_1.csv, baseline_2.csv)🎯 Classification: 🟡 MONITORING

2. Segments signals (1.0s windows, 50% overlap)- Vibration level within normal range

3. Extracts 17 time-domain features per segment- Crest Factor slightly elevated

4. Applies PCA dimensionality reduction (95% variance)- In-depth analysis recommended if trend increases

5. Trains OneClassSVM model

6. Saves model to `models/` directory💡 Next steps:

- Monitor parameters weekly

**Output:**- If CF > 5 or Kurtosis > 6: complete diagnostic

``````

✅ Model trained successfully!

---

📊 Training Details:

- Training samples: 232 segments### Example 6: Interactive Spectrum Visualization

- Original features: 17 (RMS, Kurtosis, Crest Factor, etc.)

- PCA components: 6 features**Prompt:**

- Variance explained: 95.8%```

- Model type: OneClassSVMGenerate a spectrum plot for test_bearing_fault_10000Hz.csv and highlight the expected bearing frequencies

```

💾 Model saved: models/real_baseline_model_model.pkl

**What happens:**

🎯 Ready for anomaly detection on new signals!1. Claude runs FFT analysis to identify peak frequencies

```2. Calls `plot_spectrum` with detected peaks

3. Creates interactive HTML plot with markers

---4. Returns path to the plot file



### Example 4: Detect Anomalies with ML**Result:**

```

**Prompt:**✅ Interactive plot created: data/signals/plot_spectrum_test_bearing_fault_10000Hz.html

```

Use the trained model to detect anomalies in the inner fault signalPlot features:

```- Frequency range: 0-5000 Hz

- Highlighted frequencies:

**Output:**  * 120.5 Hz - BPFO (Ball Pass Frequency Outer)

```  * 241.0 Hz - 2×BPFO

🤖 ML ANOMALY DETECTION  * 361.5 Hz - 3×BPFO

- Interactive zoom and pan enabled

📊 Analysis Results:- Hover for exact values

- Signal: real_train/InnerRaceFault_vload_1.csv

- Total segments: 28Open the file in your browser to explore the spectrum interactively!

- Anomalies detected: 28```

- Anomaly rate: 100.0%

---

🎯 PREDICTION: FAULT DETECTED

### Example 7: Envelope Analysis with Visualization

💡 Interpretation:

- 100% of segments classified as anomalous**Prompt:**

- Model trained on healthy baseline```

- Strong indication of machinery faultPerform envelope analysis on the bearing signal and create a plot showing the envelope spectrum with BPFO markers

- Consistent with statistical analysis (Kurtosis = 48)```



✅ ML model successfully detects bearing fault!**What happens:**

```1. Claude performs envelope analysis

2. Identifies characteristic frequencies

---3. Calls `plot_envelope` with frequency markers

4. Generates two-subplot interactive plot

### Example 5: ISO 20816-3 Compliance Evaluation

**Result:**

**Prompt:**```

```✅ Envelope analysis complete!

Evaluate the bearing against ISO 20816-3 standard

```📊 Analysis Results:

- Filter band: 500-5000 Hz

**What the server does:**- Peak frequencies detected: 120.5, 241.0, 361.5 Hz

1. **Auto-detects signal type**: Acceleration (g) or velocity (mm/s)- Matches BPFO pattern ✓

2. **Converts if needed**: Integrates acceleration → velocity using FFT

3. **Applies bandpass filter**: 10-1000 Hz (uses SOS filter for stability)📈 Interactive plot created: data/signals/plot_envelope_test_bearing_fault_10000Hz.html

4. **Calculates RMS velocity**

5. **Classifies into zones**: A (Good), B (Acceptable), C (Unsatisfactory), D (Unacceptable)Plot includes:

- Top: Filtered signal + envelope overlay

**Output:**- Bottom: Envelope spectrum with BPFO markers

```- All frequencies labeled and highlighted

📋 ISO 20816-3 EVALUATION

Diagnosis: Outer race bearing fault confirmed by envelope analysis

Signal: real_train/baseline_1.csv```

- Detected: Acceleration signal (0.89 g RMS)

- Auto-conversion: g → m/s² → velocity (FFT integration)---

- Sampling rate: 97656 Hz (from metadata)

### Example 8: Gear Diagnostic

📊 Results:

- RMS Velocity: 1.54 mm/s**Prompt:**

- Machine Group: 2 (Medium machines, 15-300 kW)```

- Support Type: RigidAnalyze the gear with 24 teeth, file gearbox_vibration.csv, sampling frequency 20000 Hz

- Operating Speed: 1500 RPM```



🎯 Classification:**Workflow:**

- Zone: B (Acceptable)1. GMF (Gear Mesh Frequency) calculation

- Severity: Newly commissioned or overhauled machine2. FFT analysis for GMF and harmonics

- Color Code: Yellow3. Sideband search

4. Localized defect diagnosis

💡 Recommendation:

- Condition acceptable for continuous operation---

- Continue routine monitoring

- Schedule next measurement in 3 months## 🔧 Available Tools

```

### 1. `analyze_fft`

---Fast Fourier Transform spectral analysis



### Example 6: Envelope Analysis with Bearing Frequencies**Parameters:**

- `filename`: Signal file name

**Prompt:**- `sampling_rate`: Sampling frequency (Hz)

```- `max_frequency`: Maximum analysis frequency (optional)

Perform envelope analysis on the outer race fault signal and identify bearing frequencies

```**Example:**

```python

**Output:**analyze_fft("signal.csv", sampling_rate=10000, max_frequency=5000)

``````

🔍 ENVELOPE ANALYSIS

---

Signal: real_train/OuterRaceFault_1.csv

Filter band: 500-10000 Hz (high-pass for bearing impacts)### 2. `analyze_envelope`

Sampling rate: 97656 HzEnvelope analysis for bearing defects



📊 Detected Frequencies:**Parameters:**

1. 81.13 Hz - BPFO (Ball Pass Frequency Outer) ✓- `filename`: Signal file name

2. 162.26 Hz - 2×BPFO- `sampling_rate`: Sampling frequency (Hz)

3. 243.39 Hz - 3×BPFO- `filter_low`: Low filter frequency (Hz)

- `filter_high`: High filter frequency (Hz)

🎯 DIAGNOSIS:- `num_peaks`: Number of peaks to identify

- Fault Type: OUTER RACE DEFECT

- Severity: Moderate (BPFO harmonics present)**Example:**

- Confidence: High (frequency match within 0.1 Hz)```python

analyze_envelope("bearing.csv", 10000, 500, 5000, num_peaks=5)

📈 Interactive Plot Created:```

- File: data/signals/plot_envelope_OuterRaceFault_1.html

- Top subplot: Filtered signal + envelope---

- Bottom subplot: Envelope spectrum with BPFO markers

### 3. `analyze_statistics`

💡 RECOMMENDATION:Diagnostic statistical parameters

- Monitor trending (check weekly)

- Plan bearing replacement within 1-2 months**Parameters:**

- Avoid sudden load increases- `filename`: Signal file name

```

**Output:**

---- RMS (Root Mean Square)

- Peak-to-Peak

### Example 7: Compare Multiple Signals- Crest Factor

- Kurtosis

**Prompt:**- Skewness

```- Mean, Std Dev

Compare the statistical parameters of baseline vs inner fault vs outer fault

```---



**Output:**### 4. `generate_test_signal`

```Generate synthetic test signals

📊 COMPARATIVE ANALYSIS

**Parameters:**

| Signal | RMS (g) | Crest Factor | Kurtosis | Status |- `signal_type`: "bearing_fault", "gear_fault", "imbalance", "normal"

|--------|---------|--------------|----------|--------|- `duration`: Duration (s)

| Baseline | 0.89 | 5.22 | 0.01 | ✅ Normal |- `sampling_rate`: Sampling frequency (Hz)

| Inner Fault | 1.81 | 25.34 | 48.05 | 🔴 Critical |- `noise_level`: Noise level (0-1)

| Outer Fault | 0.72 | 6.10 | 0.17 | 🟡 Monitor |

**Available Types:**

📈 Key Findings:- **bearing_fault**: Modulated periodic impulses (bearing defect)

1. Inner fault shows DRAMATIC increase in Kurtosis (4800×)- **gear_fault**: Mesh frequency + harmonics (gear defect)

   → Severe impacting, advanced damage- **imbalance**: 1x RPM component (imbalance)

   - **normal**: Broadband noise only (normal conditions)

2. Outer fault shows moderate increase in Crest Factor

   → Early stage defect, still developing---

   

3. RMS alone is NOT sufficient indicator### 5. `plot_spectrum`

   → Inner fault RMS only 2× baselineGenerate interactive FFT spectrum plot

   → But Kurtosis shows 4800× increase!

**Parameters:**

💡 Best Indicators for Bearing Faults:- `signal_file`: Signal file name

1. 🥇 Kurtosis - Most sensitive (especially inner race)- `sampling_rate`: Sampling frequency (Hz)

2. 🥈 Crest Factor - Good for impacting- `freq_range`: [min, max] frequency range (optional)

3. 🥉 Envelope Analysis - Confirms fault location- `highlight_freqs`: Frequencies to highlight (optional)

4. 📊 ISO 20816-3 - Overall machine condition- `freq_labels`: Labels for highlighted frequencies (optional)

- `title`: Custom plot title (optional)

🎯 RECOMMENDATION:

- Use multi-parameter approach**Example:**

- Kurtosis > 10 → Investigate immediately```python

- Combine with envelope analysis for fault localizationplot_spectrum(

```    "bearing_signal.csv",

    sampling_rate=10000,

---    freq_range=[0, 500],

    highlight_freqs=[120.5, 241.0, 361.5],

## 🔧 Available Tools    freq_labels=["BPFO", "2×BPFO", "3×BPFO"]

)

### 1. Statistical Analysis```

**Tool:** `analyze_statistics`

**Output:** Interactive HTML file in `data/signals/` directory

```python

analyze_statistics(filename="real_train/baseline_1.csv")**Prompt Example:**

``````

Create an FFT plot for bearing_signal.csv, highlight the BPFO frequency at 120.5 Hz

**Returns:**```

- RMS (Root Mean Square)

- Peak-to-Peak---

- Peak value

- Crest Factor (Peak/RMS)### 6. `plot_envelope`

- **Kurtosis** (Excess kurtosis, best for fault detection!)Generate interactive envelope analysis plot

- Skewness

- Mean, Standard Deviation**Parameters:**

- `signal_file`: Signal file name

**Use case:** First-line screening, trend monitoring- `sampling_rate`: Sampling frequency (Hz)

- `filter_band`: [low, high] bandpass filter (optional, default: [500, 5000])

---- `freq_range`: [min, max] envelope spectrum range (optional)

- `highlight_freqs`: Frequencies to highlight (optional)

### 2. FFT Spectrum Analysis- `freq_labels`: Labels for highlighted frequencies (optional)

**Tool:** `analyze_fft`- `title`: Custom plot title (optional)



```python**Example:**

analyze_fft(```python

    filename="signal.csv",plot_envelope(

    sampling_rate=10000,  # Optional: auto-detects from metadata    "bearing_signal.csv",

    max_frequency=5000    sampling_rate=10000,

)    filter_band=[500, 5000],

```    freq_range=[0, 300],

    highlight_freqs=[120.5, 241.0],

**Features:**    freq_labels=["BPFO", "2×BPFO"]

- Auto-detects sampling rate from metadata JSON)

- Identifies dominant frequencies```

- Full spectrum output

- Harmonic analysis**Output:** Interactive HTML file with 2 subplots:

1. Filtered signal with envelope overlay

**Use case:** Frequency identification, imbalance, misalignment2. Envelope spectrum with frequency markers



---**Prompt Example:**

```

### 3. Envelope AnalysisGenerate an envelope plot for bearing_signal.csv and mark the expected BPFO at 120.5 Hz

**Tool:** `analyze_envelope````



```python---

analyze_envelope(

    filename="bearing_signal.csv",## 📚 Resources

    sampling_rate=10000,

    filter_low=500,

    filter_high=10000,### `signal://list`

    operating_speed_rpm=1500List all available signals

)

```**Example:**

```

**Features:**Show me all available signals

- **SOS filters** (numerically stable at high sampling rates)```

- Automatic bearing frequency calculation

- BPFO, BPFI, FTF, BSF detection---

- Severity assessment

### `signal://read/{filename}`

**Use case:** Bearing fault detection and localizationRead a specific signal file



---**Example:**

```

### 4. ISO 20816-3 EvaluationRead the bearing_test.csv file

**Tool:** `evaluate_iso_20816````



```python**Supported Formats:**

evaluate_iso_20816(- `.csv` - Comma Separated Values

    signal_file="bearing_signal.csv",- `.txt` - Text with newline-separated values

    sampling_rate=None,  # Auto-detects from metadata- `.npy` - Binary NumPy arrays

    machine_group=2,

    support_type="rigid",---

    operating_speed_rpm=1500

)## 💡 Guided Prompts

```

### `diagnose_bearing`

**New Features:**Complete bearing diagnostic workflow

- ✅ **Auto-converts acceleration → velocity** (FFT integration)

- ✅ **Auto-detects sampling rate** from metadata JSON**What it does:**

- ✅ **SOS filters** for numerical stability1. Statistical analysis (RMS, CF, Kurtosis)

- ✅ Supports variable sampling rates (48 kHz - 100 kHz)2. FFT analysis (harmonics, frequencies)

3. Envelope analysis (BPFO, BPFI, BSF, FTF)

**Use case:** ISO compliance, overall machine condition assessment4. Results interpretation

5. Severity classification

---6. Action recommendations



### 5. ML Anomaly Detection---



#### Train Model### `diagnose_gear`

**Tool:** `train_anomaly_model`Gear diagnostic workflow



```python**What it does:**

train_anomaly_model(1. GMF (Gear Mesh Frequency) calculation

    healthy_signal_files=[2. FFT analysis for GMF and harmonics

        "real_train/baseline_1.csv",3. Sideband search (localized defects)

        "real_train/baseline_2.csv"4. Defect type diagnosis

    ],5. Recommendations

    segment_duration=1.0,

    overlap_ratio=0.5,---

    model_name="baseline_model"

)### `quick_diagnostic_report`

```Quick screening report



#### Predict Anomalies**What it does:**

**Tool:** `predict_anomalies`1. Signal loading and verification

2. Quick statistical analysis

```python3. Spectral screening

predict_anomalies(4. Status classification (🟢🟡🟠🔴)

    signal_file="test_signal.csv",5. In-depth analysis suggestions

    model_name="baseline_model"

)---

```

## 🎨 Advanced Workflows

**Features:**

- 17 time-domain features### Workflow 1: Complete Predictive Diagnostics

- PCA dimensionality reduction

- OneClassSVM / LocalOutlierFactor```

- Segment-level predictionsStep 1: Generate baseline signal

"Generate a normal signal for reference"

**Use case:** Predictive maintenance, early fault detection

Step 2: Baseline analysis

---"Run quick screening on test_normal_10000Hz.csv"



### 6. Interactive PlottingStep 3: Generate signal with defect

"Generate a signal with bearing defect"

#### FFT Spectrum Plot

**Tool:** `plot_spectrum`Step 4: Comparison

"Compare statistical parameters between the two signals"

```python

plot_spectrum(Step 5: Complete diagnostic

    signal_file="signal.csv","Run complete bearing diagnostic on the faulty signal"

    sampling_rate=10000,```

    freq_range=[0, 500],

    highlight_freqs=[120.5, 241.0],---

    freq_labels=["BPFO", "2×BPFO"]

)### Workflow 2: Trend Monitoring

```

```

#### Envelope PlotStep 1: Initial analysis

**Tool:** `plot_envelope`"Analyze statistics of signal_week1.csv"



```pythonStep 2: Follow-up analysis

plot_envelope("Analyze statistics of signal_week2.csv"

    signal_file="bearing_signal.csv",

    sampling_rate=10000,Step 3: Evaluate trend

    filter_band=[500, 5000],"Crest Factor went from 3.2 to 5.8, what does this mean?"

    highlight_freqs=[118.88],

    freq_labels=["BPFI"]Step 4: Decision

)"Based on the trend, is maintenance necessary?"

``````



**Features:**---

- Interactive HTML plots (Plotly)

- Zoom, pan, hover tooltips### Workflow 3: Root Cause Analysis

- Frequency markers with labels

- Professional styling```

Step 1: Screening

---"Quick screening of motor_A.csv"



## 📊 Interpretation GuideStep 2: Identify anomalies

"High kurtosis detected, what in-depth analyses are needed?"

### Statistical Parameters

Step 3: Targeted analyses

| Parameter | Formula | Normal | Caution | Alarm | Critical |"Run envelope analysis to identify the faulty component"

|-----------|---------|--------|---------|-------|----------|

| **RMS** | √(Σx²/n) | Baseline | +20% | +50% | +100% |Step 4: Confirmation

| **Crest Factor** | Peak/RMS | < 3 | 3-4 | 4-6 | > 6 |"Do the found frequencies correspond to typical defects?"

| **Kurtosis*** | E[(x-μ)⁴]/σ⁴ - 3 | < 1 | 1-3 | 3-10 | > 10 |```



*Excess kurtosis (fisher=True): 0 = Gaussian distribution---



### Kurtosis Interpretation (Most Important for Bearings!)## 📊 Results Interpretation



```### Statistical Parameters

Kurtosis < 0.5:  ✅ Healthy (normal distribution)

Kurtosis 0.5-3:  🟡 Slight impulsivity (monitor)| Parameter | Normal | Caution | Alarm | Critical |

Kurtosis 3-10:   🟠 Moderate fault (investigate)|-----------|--------|---------|-------|----------|

Kurtosis > 10:   🔴 Severe fault (urgent action)| **Crest Factor** | < 3 | 3-4 | 4-6 | > 6 |

Kurtosis > 20:   ⚠️ CRITICAL (immediate shutdown)| **Kurtosis** | < 3 | 3-5 | 5-8 | > 8 |

```| **RMS** | Base | +20% | +50% | +100% |



**Real Data Example:**### Bearing Fault Frequencies

- Baseline: Kurtosis = 0.01 ✅

- Inner Fault: Kurtosis = 48.05 🔴 (4800× increase!)Standard formulas (to be adapted to specific bearing):



### ISO 20816-3 Zones```

BPFO = N/2 × fr × (1 + (d/D) × cos(α))

| Zone | RMS Velocity | Description | Action |BPFI = N/2 × fr × (1 - (d/D) × cos(α))

|------|--------------|-------------|--------|BSF = (D/d) × fr × (1 - (d/D × cos(α))²) / 2

| **A** | < 1.4 mm/s | Good | Routine monitoring |FTF = fr/2 × (1 - (d/D) × cos(α))

| **B** | 1.4-2.8 mm/s | Acceptable | Increased monitoring |

| **C** | 2.8-4.5 mm/s | Unsatisfactory | Corrective action soon |Where:

| **D** | > 4.5 mm/s | Unacceptable | Immediate action |- N = number of rolling elements

- fr = rotation frequency (Hz)

*Values for Machine Group 2 (15-300 kW), Rigid support- d = rolling element diameter

- D = pitch diameter

---- α = contact angle

```

## 🎯 Best Practices

---

### 1. Multi-Parameter Approach

Don't rely on a single indicator:## 🐛 Troubleshooting



```### Problem: Server won't start

✅ GOOD: RMS + Crest Factor + Kurtosis + Envelope

❌ BAD: RMS alone**Solution:**

``````bash

# Check environment

**Example:** Inner race faultuv run python --version

- RMS: Only 2× increase (might be missed)

- Kurtosis: 4800× increase (unmistakable!)# Reinstall dependencies

uv sync

### 2. Use Appropriate Tools for Each Fault Type

# Manual test

| Fault Type | Best Indicators |uv run python src/machinery_diagnostics_server.py

|------------|-----------------|```

| **Bearings** | Kurtosis, Crest Factor, Envelope Analysis |

| **Imbalance** | RMS, FFT (1× RPM) |---

| **Misalignment** | FFT (1×, 2× RPM) |

| **Looseness** | Kurtosis, harmonics |### Problem: File not found

| **Gears** | Envelope, GMF analysis |

**Check:**

### 3. Trend Monitoring1. File in `data/signals/` ?

Track parameters over time:2. Filename correct (case-sensitive)?

3. Supported extension (.csv, .txt, .npy)?

```python

# Week 1---

Kurtosis = 0.5

### Problem: Envelope analysis errors

# Week 2

Kurtosis = 1.2  (2.4× increase - investigate!)**Common causes:**

- `filter_high >= sampling_rate/2` ❌

# Week 3- Signal too short

Kurtosis = 3.5  (7× increase - urgent!)- Corrupted file

```

**Solution:**

**Trending is more important than absolute values!**```

filter_high < sampling_rate / 2  (Nyquist)

### 4. Combine with ISO StandardExample: fs=10000 Hz → filter_high max = 4999 Hz

Use ISO 20816-3 for:```

- Compliance reporting

- Overall machine health---

- Maintenance decisions

## 📞 Support

But remember:

- Velocity-based (less sensitive to bearing faults)For problems or questions:

- Kurtosis is better for early detection1. Check README.md

- Use both for complete assessment2. Run test_suite.py

3. Check server logs

---

---

## 🐛 Troubleshooting

**Happy diagnostics! 🔧**

### Problem: NaN values in ISO 20816-3

**Cause:** Signal is acceleration but filter fails  
**Solution:** ✅ **FIXED** - Server auto-converts acceleration → velocity

### Problem: Different sampling rates

**Cause:** Dataset has 97656 Hz and 48828 Hz files  
**Solution:** ✅ **FIXED** - Server auto-reads from metadata JSON

### Problem: "File not found" error

**Check:**
```bash
# List available signals
ls data/signals/real_train/

# Check metadata exists
ls data/signals/real_train/*_metadata.json
```

### Problem: Filter instability at high sampling rates

**Cause:** Regular Butterworth filter numerical instability  
**Solution:** ✅ **FIXED** - Server uses SOS (Second-Order Sections) filters

---

## 📚 Additional Resources

### Dataset Information
- **Download script:** `download_real_data.py`
- **Conversion script:** `convert_mat_to_csv.py`
- **Test script:** `tests/test_real_data.py`

### Documentation
- Full API reference: `README.md`
- Technical details: Check inline documentation in `machinery_diagnostics_server.py`

### Real Data Files
```
data/signals/real_train/
├── baseline_1.csv (+ _metadata.json)
├── baseline_2.csv (+ _metadata.json)
├── InnerRaceFault_vload_1.csv (+ _metadata.json)
├── InnerRaceFault_vload_2-5.csv
├── OuterRaceFault_1.csv (+ _metadata.json)
└── OuterRaceFault_vload_1-7.csv

Each metadata JSON contains:
- sampling_rate
- duration_sec
- num_samples
- BPFI, BPFO, FTF, BSF
- shaft_speed, load
```

---

## 🎓 Example Workflows

### Workflow 1: Complete Bearing Diagnostic

```
1. "List available signals"
   → Shows all real_train and real_test files

2. "Analyze statistics of real_train/baseline_1.csv"
   → Baseline: Kurtosis = 0.01 ✅

3. "Analyze statistics of real_train/InnerRaceFault_vload_1.csv"
   → Inner Fault: Kurtosis = 48.05 🔴

4. "Perform envelope analysis on the inner fault signal"
   → Confirms BPFI at 118.88 Hz

5. "Evaluate against ISO 20816-3 standard"
   → Zone B but acceleration metrics show severe fault

6. "What's the diagnosis and recommended action?"
   → CRITICAL: Immediate bearing replacement required
```

### Workflow 2: Train and Test ML Model

```
1. "Train anomaly detection model on baseline signals"
   → Model trained on baseline_1 and baseline_2

2. "Test the model on baseline_1.csv"
   → 9.5% anomalies (expected - some segments differ)

3. "Test the model on InnerRaceFault_vload_1.csv"
   → 100% anomalies detected! ✅

4. "Test the model on OuterRaceFault_1.csv"
   → 100% anomalies detected! ✅

5. "What does this mean?"
   → ML successfully distinguishes faults from baseline
```

### Workflow 3: Comparative Analysis

```
1. "Compare all three signal types statistically"
   → Table with RMS, CF, Kurtosis side-by-side

2. "Which parameter is most sensitive?"
   → Kurtosis: 4800× increase for inner fault!

3. "Show me the envelope spectra for both faults"
   → Inner: BPFI peaks, Outer: BPFO peaks

4. "Generate ISO report for all signals"
   → All in Zone B (velocity), but acceleration shows faults
```

---

## 💡 Pro Tips

1. **Always check Kurtosis first** - Most sensitive for bearings
2. **Use metadata auto-detection** - No need to specify sampling rate
3. **Combine statistical + envelope** - Best accuracy
4. **Track trends over time** - More important than absolute values
5. **ISO 20816-3 for compliance** - But use Kurtosis for detection
6. **ML for predictive maintenance** - Train on healthy baseline
7. **Interactive plots for reports** - Professional HTML output

---

**Ready for world-class machinery diagnostics! 🔧**


