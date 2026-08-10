---
description: >
  Gear fault diagnosis workflow using vibration analysis via the
  predictive-maintenance-mcp server. Use this skill when the user says "gear
  fault", "gear diagnosis", "gear mesh", "gearbox analysis", "gear vibration",
  "tooth damage", "gear defect", "diagnose gear", "gear mesh frequency",
  "sideband analysis", or wants to detect gear faults from vibration signals.
---

# Gear Fault Diagnosis

Detect gear faults through spectral analysis of vibration signals, focusing on
gear mesh frequency (GMF) harmonics, sidebands, and modulation patterns. This
workflow supports expert judgment — it never replaces it.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Background

Gear faults produce characteristic vibration patterns:
- **Gear Mesh Frequency (GMF)** = number_of_teeth x shaft_RPM / 60
- Healthy gears show GMF and low-level harmonics
- Faulty gears show increased GMF harmonics, sidebands spaced at the shaft
  frequency, and broadband noise

## Workflow

### Step 1 — Signal Selection and Parameters

Load the signal with `load_signal(filepath="gearbox_run1.csv", signal_unit="g")`
or pick a loaded one from `list_signals(scope="memory")`. Verify sampling rate
and metadata with `get_signal_info(signal_id="<id>")`. For raw binary files
(`.bin`/`.raw`/`.dat`), also declare `sample_format` and `sampling_rate` (or
provide a companion `<stem>_metadata.json`) — see the signal-management skill.

Gather from the user (do NOT guess):
- **Shaft RPM** (or shaft frequency in Hz)
- **Number of teeth** on the gear of interest
- **Gear ratio** (if analyzing a gear pair)

Calculate:
- GMF = teeth x RPM / 60
- Shaft frequency = RPM / 60

### Step 2 — Statistical Screening

Call `analyze_statistics(signal_id="<id>")`.

Gear fault indicators (screening only, never diagnostic alone):
- **Kurtosis > 3**: impulsive content (possible tooth damage)
- **Crest Factor > 5**: localized damage
- **RMS increase**: overall vibration energy rise

### Step 3 — FFT Analysis

Call `analyze_fft(signal_id="<id>")`.

Inspect the spectrum for:

| Pattern | Indicates |
|---|---|
| GMF with low harmonics, no sidebands | Normal gear operation |
| Increased GMF harmonics (2x, 3x, 4x GMF) | Gear wear or misalignment |
| Sidebands around GMF at shaft frequency spacing | Localized tooth damage |
| Broadband noise increase | Advanced wear or pitting |
| Ghost frequencies (non-harmonic peaks) | Manufacturing defects |

Systematic check of the expected GMF against the envelope spectrum:

Call `check_bearing_faults(signal_id="<id>", rpm=1500, frequencies={"GMF": 800.0})`

(The `frequencies` route accepts any labeled frequency — it is the
out-of-catalog / gearbox path.)

### Step 4 — Envelope Analysis (for Modulation)

Call `analyze_envelope(signal_id="<id>", filter_low=500, filter_high=5000)`.

Center the demodulation band around GMF or its harmonics (keep it below
Nyquist) to extract amplitude modulation. Sidebands in the envelope spectrum
at shaft-frequency spacing corroborate gear fault modulation.

### Step 5 — Advanced Spectral Analysis

Call `compute_spectrogram_stft(signal_id="<id>")` to check for time-varying
patterns — localized tooth damage shows periodic energy bursts at the shaft
rotation rate.

Call `compute_power_spectral_density(signal_id="<id>")` for a smoothed
spectral view that highlights persistent gear-mesh patterns.

### Step 6 — ISO Severity Context

Call `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`.

Report the zone (A/B/C/D) and urgency. The verdict requires a DECLARED signal
unit — if refused, confirm the unit with the user and re-load with
`load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`.

### Step 7 — Diagnosis Summary

Classification rule (choose exactly one, cite the evidence):
- **Gear tooth fault — strong evidence**: GMF harmonics present AND at least
  one clear sideband pair spaced at the shaft frequency AND a supporting
  statistic (Kurtosis > 3 or CF > 4)
- **Possible localized tooth damage**: partial sidebands or ambiguous
  spacing — list the missing evidence needed for confirmation
- **Uniform wear / increased load**: elevated GMF amplitude WITHOUT sidebands
  and normal impulsiveness

Each conclusion MUST cite: tools used, specific peaks (frequency and
magnitude), sideband spacing vs the expected shaft frequency, and supporting
statistics.

### Step 8 — Report Generation

Call `generate_fft_report(signal_id="<id>", max_freq=5000, num_peaks=15, rpm=1500)`
and optionally `generate_diagnostic_report_docx(signal_id="<id>", sections={"summary": "..."})`.

## Important Notes

- GMF calculation requires an accurate tooth count and RPM — ask, never assume
- Multi-stage gearboxes: analyze each stage separately
- Gear and bearing faults can coexist; if bearing frequencies also appear,
  recommend the bearing-diagnosis skill as well
- Use cautious language for all diagnoses; the final call rests with a
  qualified engineer
- All processing is local — no data leaves the machine
