---
description: >
  Fast vibration health screening for industrial machinery using the
  predictive-maintenance-mcp server. Use this skill when the user says "quick
  check", "health screening", "vibration check", "is this machine healthy",
  "overall health", "quick diagnostic", "machine status", "condition monitoring",
  "health assessment", or wants a rapid overview of machine condition without
  full fault identification.
---

# Quick Vibration Screening

Fast health status assessment for rotating machinery. Produces a screening
report in under 30 seconds with clear next-step recommendations.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Step 1 — Signal Selection

Call `list_stored_signals()` or `list_signals()` to show available signals. If
the user has not specified one, ask which signal to analyze. Load with
`load_signal(...)` if needed.

### Step 2 — Statistical Features

Call `extract_features_from_signal(signal_id=...)`.

Report as bullet points:
- **RMS**: energy level
- **Crest Factor**: impulsiveness indicator
- **Kurtosis**: excess kurtosis (Fisher convention)
- **Peak-to-Peak**: overall vibration range

### Step 3 — FFT Snapshot

Call `analyze_fft(signal_id=...)`.

Report:
- Peak frequency and magnitude
- Top 3 spectral peaks (frequency + magnitude)
- Any obvious harmonic patterns

### Step 4 — ISO 20816-3 Assessment

Call `evaluate_iso_20816(signal_id=..., machine_group=2, support_type="rigid")`.

Report:
- RMS Velocity in mm/s
- Zone: A / B / C / D
- Severity description

### Step 5 — Summary Card

Format the output as a concise screening card:

```
VIBRATION HEALTH SCREENING
===========================
Signal: {filename}
Overall: {Healthy / Monitor / Suspicious / Critical}

Key Indicators:
  RMS: {value}  |  CF: {value}  |  Kurt: {value}
  ISO Zone: {zone} ({severity})

Recommendation:
  {appropriate next step}
```

**Decision logic:**

| ISO Zone | Kurtosis | Verdict | Recommendation |
|----------|----------|---------|----------------|
| A | < 1 | Healthy | No immediate concerns. Schedule routine monitoring. |
| B | 1–3 | Monitor | Elevated indicators. Schedule follow-up in 2–4 weeks. |
| C | 3–6 | Suspicious | Consider detailed bearing analysis (bearing-diagnosis skill). |
| D | > 6 | Critical | Immediate detailed analysis recommended. |

## Important Notes

- This is a SCREENING tool, NOT a diagnostic tool
- Never make definitive fault claims from screening alone
- Always recommend targeted analysis for Suspicious/Critical results
- Use cautious language: "indicators suggest", not "confirmed fault"
- Confirm signal units with the user before ISO evaluation
