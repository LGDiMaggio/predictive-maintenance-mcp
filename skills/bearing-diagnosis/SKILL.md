---
name: bearing-diagnosis
description: >
  Complete bearing fault diagnostic workflow using vibration analysis.
  Use when user says "diagnose bearing", "bearing fault", "bearing check",
  "detect bearing damage", "bearing vibration analysis", "inner race", 
  "outer race", "ball defect", or "cage fault". Coordinates statistical 
  screening, envelope analysis, and characteristic frequency matching 
  via the predictive-maintenance-mcp server.
metadata:
  author: Luigi Di Maggio
  version: 1.0.0
  mcp-server: predictive-maintenance-mcp
  category: predictive-maintenance
compatibility: >
  Requires predictive-maintenance-mcp MCP server connected.
  Works on Claude.ai, Claude Code, and API.
---

# Bearing Fault Diagnosis Skill

Evidence-based bearing fault detection using vibration signals. This skill 
orchestrates MCP tools in a precise diagnostic sequence with decision gates 
at each step.

## Instructions

### Step 1: Signal Discovery
Call `list_signals()` to find available vibration signals.
Ask user which signal to analyze if multiple are available.

### Step 2: Statistical Screening
Call `analyze_statistics("{signal_file}")`.

Evaluate screening flags (excess kurtosis, Fisher convention):
- Kurtosis > 0 → non-Gaussian content (possible impulses)
- Kurtosis > 3 → significant impulsive content
- Kurtosis > 6 → severe, likely bearing damage
- Crest Factor > 4 → impulsiveness present
- Crest Factor > 6 → strong impulsiveness

**Decision gate:** If Kurtosis < 0 AND Crest Factor < 3, the signal 
shows no impulsive content. Report "No bearing fault indicators in 
time-domain screening" and ask if user wants to continue anyway.

### Step 3: Spectral Analysis (FFT)
Call `analyze_fft("{signal_file}")`.

Identify:
- Dominant frequencies and their harmonics
- Shaft frequency (1× RPM) and multiples
- Any broadband energy increase

If operating speed is unknown, ask user for RPM or shaft frequency.

### Step 4: Bearing Characteristic Frequencies
If bearing designation is known:
Call `search_bearing_catalog("{bearing_designation}")`
or `calculate_bearing_characteristic_frequencies(...)`.

Expected frequencies:
- BPFO: Ball Pass Frequency Outer race
- BPFI: Ball Pass Frequency Inner race  
- BSF: Ball Spin Frequency
- FTF: Fundamental Train (cage) Frequency

### Step 5: Envelope Analysis
Call `analyze_envelope("{signal_file}", filter_low=500, filter_high=5000)`.

Adjust filter band based on Step 3:
- If resonance found in FFT → center band around resonance
- Default band: 500–5000 Hz for general bearing analysis
- High-speed machines: shift band higher

Call `plot_envelope(...)` for visual report.

### Step 6: Evidence Matching
Compare envelope peaks against characteristic frequencies:

| Peak matches | Harmonics present | + High Kurtosis | Diagnosis |
|---|---|---|---|
| BPFO ± 2% | Yes (2×, 3×) | Yes | **Possible outer race fault** |
| BPFI ± 2% | Yes + sidebands | Yes | **Possible inner race fault** |
| BSF ± 2% | Yes (2×) | Yes | **Possible ball defect** |
| FTF ± 2% | Irregular spacing | Moderate | **Possible cage fault** |
| No matches | — | — | **Inconclusive** |

**Confidence levels:**
- "Confirmed": Peak + harmonics + high kurtosis + additional indicator
- "Possible/consistent with": Peak present but missing corroboration
- "Inconclusive": Insufficient evidence

### Step 7: ISO 20816-3 Severity
Call `evaluate_iso_20816("{signal_file}", machine_group=2, support_type="rigid")`.

Report vibration zone (A/B/C/D) and urgency level.

### Step 8: Report Generation
Call `generate_envelope_report(...)` and `generate_fft_report(...)`.
Inform user of HTML report locations.

## Troubleshooting

### No peaks in envelope spectrum
Cause: Wrong filter band or signal too short.
Solution: Try wider filter band (200–10000 Hz), or analyze full signal 
(set segment_duration to None).

### Unknown bearing type
Solution: Ask user for bearing designation (e.g., SKF 6205) and use 
`search_bearing_catalog()`. If unavailable, check machine manual with 
`extract_manual_specs()`.

### Kurtosis high but no envelope peaks
Cause: Impulses may be from non-bearing source (gear mesh, electrical).
Solution: Suggest `diagnose_gear` prompt or check for electrical frequencies.
