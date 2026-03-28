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
gear mesh frequency (GMF) harmonics, sidebands, and modulation patterns.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Background

Gear faults produce characteristic vibration patterns:
- **Gear Mesh Frequency (GMF)** = number_of_teeth x shaft_RPM / 60
- Healthy gears show GMF and low-level harmonics
- Faulty gears show increased GMF harmonics, sidebands at shaft speed, and
  broadband noise

## Workflow

### Step 1 — Signal Selection and Parameters

Load the signal with `load_signal(...)` or select from `list_stored_signals()`.

Gather from the user:
- **Shaft RPM** (or shaft frequency in Hz)
- **Number of teeth** on the gear of interest
- **Gear ratio** (if analyzing a gear pair)

Calculate:
- GMF = teeth x RPM / 60
- Shaft frequency = RPM / 60

### Step 2 — Statistical Screening

Call `extract_features_from_signal(signal_id=...)`.

Gear fault indicators:
- **Kurtosis > 3**: impulsive content (tooth damage)
- **Crest Factor > 5**: localized damage
- **RMS increase**: overall vibration energy rise

### Step 3 — FFT Analysis

Call `analyze_fft(signal_id=...)`.

Inspect the spectrum for:

| Pattern | Indicates |
|---|---|
| GMF with low harmonics, no sidebands | Normal gear operation |
| Increased GMF harmonics (2x, 3x, 4x GMF) | Gear wear or misalignment |
| Sidebands around GMF at shaft frequency spacing | Localized tooth damage |
| Broadband noise increase | Advanced wear or pitting |
| Ghost frequencies (non-harmonic peaks) | Manufacturing defects |

### Step 4 — Envelope Analysis (for Modulation)

Call `analyze_envelope(signal_id=..., filter_low=..., filter_high=...)`.

Center the filter band around GMF or its harmonics to extract amplitude
modulation patterns. Sidebands in the envelope spectrum at shaft frequency
confirm gear fault modulation.

### Step 5 — Advanced Spectral Analysis

Call `compute_spectrogram_stft(signal_id=...)` to check for time-varying
patterns. Gear faults with localized damage show periodic energy bursts at the
shaft rotation rate.

Call `compute_power_spectral_density(signal_id=...)` for a smoothed spectral
view that reduces noise and highlights persistent gear mesh patterns.

### Step 6 — ISO 20816-3 Severity

Call `evaluate_iso_20816(signal_id=..., machine_group=..., support_type=...)`.

Report vibration zone and urgency level.

### Step 7 — Diagnosis Summary

Present findings:

```
GEAR FAULT DIAGNOSIS
====================
Signal: {signal_id}
Gear Mesh Frequency: {gmf} Hz ({teeth} teeth x {rpm} RPM)

Spectral Findings:
  GMF amplitude: {value} — {normal/elevated/high}
  Harmonics: {count} significant harmonics detected
  Sidebands: {present/absent} at {shaft_freq} Hz spacing

Assessment: {diagnosis}
Confidence: {level}
ISO Zone: {zone}

Recommendation: {action}
```

### Step 8 — Report Generation

Call `generate_fft_report(signal_id=...)` and optionally
`generate_diagnostic_report_docx(...)`.

## Diagnosis Decision Table

| GMF Level | Harmonics | Sidebands | Kurtosis | Diagnosis |
|---|---|---|---|---|
| Normal | 1-2 harmonics | None | < 3 | **Healthy** |
| Elevated | 3+ harmonics | None | < 3 | **Wear / misalignment** |
| High | Multiple | Present | 3-6 | **Localized tooth damage** |
| High | Many + broadband | Many | > 6 | **Advanced gear damage** |

## Important Notes

- GMF calculation requires accurate tooth count and RPM
- Some gearboxes have multiple gear stages — analyze each separately
- Gear and bearing faults can coexist; if bearing frequencies are also present,
  recommend running the bearing-diagnosis skill as well
- Use cautious language for all diagnoses
- All processing is local — no data leaves the machine
