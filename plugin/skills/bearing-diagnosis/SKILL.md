---
description: >
  Complete bearing fault diagnostic workflow using vibration analysis and the
  predictive-maintenance-mcp server. Use this skill when the user says "diagnose
  bearing", "bearing fault", "bearing check", "detect bearing damage", "bearing
  vibration analysis", "inner race fault", "outer race fault", "ball defect",
  "cage fault", "BPFO", "BPFI", "BSF", "FTF", or asks to identify bearing
  problems from vibration data.
---

# Bearing Fault Diagnosis

Evidence-based bearing fault detection using vibration signals. Orchestrate MCP
tools in a precise diagnostic sequence with decision gates at each step.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Step 1 — Signal Discovery

Call `list_stored_signals()` to check cached signals, or `list_signals()` to
browse the data/ directory. If no signal is loaded, ask the user for a file path
and call `load_signal(file_path=..., signal_id=...)`. Ask which signal to
analyze if multiple are available.

### Step 2 — Statistical Screening

Call `extract_features_from_signal(signal_id=...)`.

Evaluate screening flags (excess kurtosis, Fisher convention):

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| Kurtosis > 0 | Mild | Non-Gaussian content (possible impulses) |
| Kurtosis > 3 | Moderate | Significant impulsive content |
| Kurtosis > 6 | Severe | Likely bearing damage |
| Crest Factor > 4 | Mild | Impulsiveness present |
| Crest Factor > 6 | Strong | Strong impulsiveness |

**Decision gate**: If Kurtosis < 0 AND Crest Factor < 3 the signal shows no
impulsive content. Report "No bearing fault indicators in time-domain screening"
and ask if the user wants to continue anyway.

### Step 3 — Spectral Analysis (FFT)

Call `analyze_fft(signal_id=...)`.

Identify:
- Dominant frequencies and their harmonics
- Shaft frequency (1x RPM) and multiples
- Broadband energy increase

If operating speed is unknown, ask the user for RPM or shaft frequency.

### Step 4 — Bearing Characteristic Frequencies

If bearing designation is known, call
`lookup_bearing_and_compute_tool(bearing_query=..., shaft_rpm=...)` to look up
the bearing in the catalog AND compute fault frequencies in one step.

Otherwise call `calculate_bearing_characteristic_frequencies(...)` with manual
geometry (n_balls, d_ball, d_pitch, contact_angle, shaft_rpm).

If the bearing designation is unknown, ask the user or try
`extract_manual_specs(manual_name=...)` to pull it from a machine manual.

Expected frequencies:
- **BPFO** — Ball Pass Frequency Outer race
- **BPFI** — Ball Pass Frequency Inner race
- **BSF** — Ball Spin Frequency
- **FTF** — Fundamental Train (cage) Frequency

### Step 5 — Envelope Analysis

Call `analyze_envelope(signal_id=..., filter_low=..., filter_high=...)`.

Adjust the filter band based on Step 3:
- If a resonance was found in FFT, center the band around it
- Default band: 500–5000 Hz for general bearing analysis
- High-speed machines: shift band higher

Call `plot_envelope(...)` for a visual report.

### Step 6 — Evidence Matching

Compare envelope peaks against characteristic frequencies using
`check_bearing_faults_direct(signal_id=..., shaft_rpm=..., ...)`.

| Peak matches | Harmonics | + High Kurtosis | Diagnosis |
|---|---|---|---|
| BPFO +/- 2% | 2x, 3x present | Yes | **Possible outer race fault** |
| BPFI +/- 2% | + sidebands at shaft freq | Yes | **Possible inner race fault** |
| BSF +/- 2% | 2x present | Yes | **Possible ball defect** |
| FTF +/- 2% | Irregular spacing | Moderate | **Possible cage fault** |
| No matches | — | — | **Inconclusive** |

Confidence levels:
- **Confirmed**: Peak + harmonics + high kurtosis + additional indicator
- **Possible/consistent with**: Peak present but missing corroboration
- **Inconclusive**: Insufficient evidence

### Step 7 — ISO 20816-3 Severity

Call `evaluate_iso_20816(signal_id=..., machine_group=2, support_type="rigid")`.

Report vibration zone (A/B/C/D) and urgency level. Confirm signal units with the
user before calling — wrong units invalidate the result.

### Step 8 — Report Generation

Call `generate_envelope_report(...)` and `generate_fft_report(...)`.
Optionally call `generate_diagnostic_report_docx(...)` for a full DOCX report.
Inform the user of report file locations.

## Troubleshooting

**No peaks in envelope spectrum** — Wrong filter band or signal too short. Try a
wider band (200–10000 Hz) or analyze the full signal without segmentation.

**Unknown bearing type** — Ask for bearing designation (e.g., SKF 6205) and call
`search_bearing_catalog()`. If unavailable, check the machine manual with
`extract_manual_specs()`.

**Kurtosis high but no envelope peaks** — Impulses may come from a non-bearing
source (gear mesh, electrical). Suggest using the gear-diagnosis skill or check
for electrical frequencies (line frequency harmonics).

## Important Notes

- Always use cautious diagnostic language: "consistent with", "possible fault"
- Never claim a definitive fault without multiple corroborating indicators
- Confirm signal units before ISO severity assessment
- All processing happens locally — raw signal data never leaves the machine
