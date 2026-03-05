---
name: quick-screening
description: >
  Fast vibration health screening for industrial machinery.
  Use when user says "quick check", "health screening", "vibration check",
  "is this machine healthy", "overall health", "quick diagnostic", 
  "machine status", or "condition monitoring". Performs rapid statistical 
  and spectral screening without full fault identification.
metadata:
  author: Luigi Di Maggio
  version: 1.0.0
  mcp-server: predictive-maintenance-mcp
  category: predictive-maintenance
compatibility: >
  Requires predictive-maintenance-mcp MCP server connected.
  Works on Claude.ai, Claude Code, and API.
---

# Quick Vibration Screening Skill

Fast health status assessment for rotating machinery. Produces a screening 
report in under 30 seconds with clear next-step recommendations.

## Instructions

### Step 1: Signal Selection
Call `list_signals()` to show available signals.
If user hasn't specified, ask which signal to analyze.

### Step 2: Statistics
Call `analyze_statistics("{signal_file}")`.

Report as bullet points:
- RMS: {value} (energy level)
- Crest Factor: {value}
- Kurtosis: {value} (excess kurtosis)

### Step 3: FFT Snapshot
Call `analyze_fft("{signal_file}")`.

Report:
- Peak frequency: {value} Hz
- Top 3 spectral peaks (frequency + magnitude)

### Step 4: ISO 20816-3 Assessment
Call `evaluate_iso_20816("{signal_file}", machine_group=2, support_type="rigid")`.

Report:
- RMS Velocity: {value} mm/s
- Zone: {A/B/C/D} ({color})
- Severity: {level}

### Step 5: Summary and Recommendations

Format as a brief screening card:

```
VIBRATION HEALTH SCREENING
===========================
Signal: {filename}
Overall: {Healthy/Suspicious/Critical}

Key Indicators:
  RMS: {value}  |  CF: {value}  |  Kurt: {value}
  ISO Zone: {zone} ({severity})

Recommendation:
  {One of the following:}
  - "No immediate concerns. Schedule routine monitoring."
  - "Elevated indicators. Consider detailed bearing analysis 
     (use bearing-diagnosis skill)."  
  - "Critical levels detected. Immediate detailed analysis 
     recommended."
```

**Decision logic:**
- ISO Zone A + Kurtosis < 1 → Healthy
- ISO Zone B OR Kurtosis 1-3 → Monitor
- ISO Zone C OR Kurtosis 3-6 → Suspicious → Recommend bearing-diagnosis
- ISO Zone D OR Kurtosis > 6 → Critical → Urgent detailed analysis

### Important Notes
- This is a SCREENING tool, NOT a diagnostic tool
- Never make definitive fault claims from screening alone
- Always recommend targeted analysis for suspicious/critical results
- Use cautious language: "indicators suggest" not "confirmed fault"
