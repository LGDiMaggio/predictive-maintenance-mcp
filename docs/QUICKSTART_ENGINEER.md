# 🔧 Quickstart for Maintenance & Reliability Engineers

> **Goal**: Get your first AI-powered diagnostic report in under 10 minutes — no coding experience needed.

---

## What You'll Achieve

By the end of this guide, you will have:

1. ✅ A running AI diagnostics server on your computer
2. ✅ An interactive bearing fault detection report (HTML)
3. ✅ An ISO 20816-3 vibration severity assessment
4. ✅ The ability to ask questions about your machinery in plain language

**No Python knowledge required.** You'll copy-paste a few commands, and then everything happens through conversation.

---

## What This Tool Does

Imagine having a vibration analyst available 24/7, who can:

- **Read your vibration data** (CSV files from accelerometers, DAQ systems)
- **Run FFT spectrum analysis** and detect dominant frequencies
- **Perform envelope analysis** to identify bearing fault signatures (BPFO, BPFI, BSF)
- **Evaluate ISO 20816-3 compliance** and classify vibration severity zones
- **Read equipment manuals** (PDF/TXT) and extract bearing specifications automatically
- **Train anomaly detection models** on your healthy baselines and flag deviations
- **Generate professional HTML reports** you can share with your team or management

You interact with all of this through **natural language** — just describe what you need.

---

## Step 1: Install the Server (One-Time Setup)

### What You Need

- A computer with **Python 3.11 or 3.12** installed ([Download Python](https://www.python.org/downloads/))
- **Claude Desktop** app ([Download Claude Desktop](https://claude.ai/download))

### Installation (Copy-Paste Each Line)

Open **Command Prompt** (Windows) or **Terminal** (macOS/Linux) and run:

```bash
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
python setup_venv.py
```

Wait for the setup to complete (1-2 minutes). Then verify:

```bash
python validate_server.py
```

You should see: `✅ Server validation passed`

> **Don't have Git?** Download the project as a ZIP from GitHub and extract it instead of running `git clone`.

### Connect to Claude Desktop

**Windows**: Run the automated setup:
```powershell
.\setup_claude.ps1
```

**Or configure manually**: See [INSTALL.md](../INSTALL.md) → "For Claude Desktop Users" section.

After configuration, **restart Claude Desktop** completely (quit and reopen).

---

## Step 2: Your First Analysis (5 Minutes)

The server comes with **20 real bearing vibration signals** for you to explore immediately.

### Open Claude Desktop and Try These

#### 2.1 — See What Data Is Available

Type in Claude:
```
List all available vibration signals
```

Claude will show you the 20 included signals: healthy baselines, inner race faults, and outer race faults.

#### 2.2 — Generate a Bearing Fault Report

Type:
```
Generate an envelope analysis report for real_train/OuterRaceFault_1.csv
```

**What happens:**
1. Claude reads the vibration signal (97,656 samples/second!)
2. Applies bandpass filtering to isolate fault frequencies
3. Performs envelope demodulation
4. Generates a professional interactive HTML report
5. Saves it to your `reports/` folder

**Open the HTML file** in your browser — you'll see an interactive Plotly chart with bearing fault frequencies marked. You can zoom, pan, and hover for details.

#### 2.3 — Check ISO Compliance

Type:
```
Evaluate the vibration severity of real_train/OuterRaceFault_1.csv according to ISO 20816-3
```

Claude will:
- Calculate RMS velocity
- Classify it into ISO zones (A/B/C/D)
- Generate a color-coded compliance report
- Give you a clear recommendation

#### 2.4 — Ask Questions in Plain Language

You're not limited to specific commands. Try conversational questions:

```
Compare the FFT spectrum of baseline_1.csv with OuterRaceFault_1.csv. 
What differences do you see?
```

```
Train an anomaly detection model using the two healthy baselines, 
then check if OuterRaceFault_1.csv is anomalous
```

```
What bearing frequencies should I look for if the shaft speed is 1500 RPM 
and I'm using a 6205 bearing?
```

---

## Step 3: Use Your Own Data

### Prepare Your Data

Your vibration data needs to be in **CSV format** with one column of acceleration values. Example:

```csv
acceleration
0.0023
0.0045
-0.0012
0.0067
...
```

### Add Your Files

1. Copy your CSV files to: `data/signals/real_train/` (or `real_test/`)
2. (Optional) Create a metadata JSON file with the same name:

```json
{
    "sampling_rate_hz": 10000,
    "shaft_speed_rpm": 1475,
    "unit": "g",
    "description": "Pump A - Drive end bearing"
}
```

> **No metadata file? No problem.** You can provide the sampling rate directly in conversation:  
> *"Analyze the FFT of my_signal.csv with a sampling rate of 10,000 Hz"*

### Add Equipment Manuals

Copy PDF or TXT manuals to: `resources/machine_manuals/`

Then ask Claude:
```
What bearings are specified in my_pump_manual.pdf?
```

Claude will extract bearing designations, operating speeds, and automatically calculate fault frequencies.

---

## Step 4: Common Workflow Examples

### Complete Bearing Diagnosis Workflow

```
1. Read the specifications from pump_manual.pdf
2. Look up the bearing geometry for SKF 6205-2RS
3. Calculate the characteristic frequencies at 1475 RPM
4. Run envelope analysis on pump_vibration.csv looking for those frequencies
5. Generate a report with your findings
```

Claude handles all five steps automatically, chaining the tools together.

### Batch Analysis

```
Generate ISO 20816-3 reports for all the training signals
```

### ML-Based Anomaly Detection

```
Train an anomaly detection model using baseline_1.csv and baseline_2.csv 
as healthy references. Then evaluate all the fault signals against it.
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Claude doesn't see the server | Restart Claude Desktop completely. Check paths in config. |
| "Server not found" error | Make sure you used **absolute paths** in the config file |
| Python not found | Download from [python.org](https://www.python.org/downloads/) — version 3.11 or 3.12 |
| Reports don't open | Navigate to the `reports/` folder and double-click the HTML file |

📖 **More help**: See [INSTALL.md](../INSTALL.md) → Troubleshooting section

---

## What's Next?

- 📚 **Explore more workflows**: [EXAMPLES.md](../EXAMPLES.md) has 7 complete step-by-step tutorials
- 📊 **Add your own data**: Copy CSV files and manuals to start analyzing your real machinery
- 🤝 **Give feedback**: Your real-world experience is invaluable — [tell us what worked and what didn't](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions)
- 🔧 **Help validate**: Even without coding, you can help by [testing diagnostic accuracy with your data](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues)

---

**You are the expert on your machines. This tool gives you an AI assistant that speaks your language.**
