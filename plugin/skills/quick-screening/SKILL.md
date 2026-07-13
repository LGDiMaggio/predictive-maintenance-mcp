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
card in under 30 seconds with clear next-step recommendations. Screening
informs the engineer's decision — it is never a diagnosis by itself.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Step 1 — Signal Selection

Call `list_signals(scope="memory")` to show loaded signal_ids, or
`list_signals(scope="disk")` for loadable files. If the user has not
specified one, ask. Load with
`load_signal(filepath="real_test/baseline_1.csv", signal_unit="g")` if
needed — declaring `signal_unit` up front makes the ISO step work without a
re-load.

### Step 2 — Statistical Features

Call `analyze_statistics(signal_id="<id>")`.

Report as bullet points:
- **RMS**: energy level
- **Crest Factor**: impulsiveness indicator
- **Kurtosis**: excess kurtosis (Fisher convention)
- **Peak-to-Peak**: overall vibration range

### Step 3 — FFT Snapshot

Call `analyze_fft(signal_id="<id>")`.

Report:
- Peak frequency and magnitude
- Top 3 spectral peaks (frequency + magnitude)
- Any obvious harmonic patterns

### Step 4 — ISO Severity Assessment

Call `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`.

Report:
- RMS velocity in mm/s
- Zone: A / B / C / D
- Severity description

Notes:
- Confirm machine_group (1 = large >300 kW, 2 = medium 15-300 kW) and
  support_type with the user when known.
- The verdict requires a DECLARED signal unit. If it is refused, re-load with
  `load_signal(filepath="<file>", signal_unit="g", overwrite=True)` after
  confirming the unit with the user — units are never guessed.
- If only a portable-instrument reading is available (no signal file), use
  `assess_severity(rms_velocity_mm_s=3.2, machine_group=2, support_type="rigid")`.

### Step 5 — Summary Card

Format the output as a concise screening card:

```
VIBRATION HEALTH SCREENING
===========================
Signal: {signal_id}
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
- The screening supports the engineer's judgment; it does not replace it
- All processing happens locally — raw data never leaves the machine
