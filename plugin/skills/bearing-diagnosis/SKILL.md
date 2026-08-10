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
tools in a precise diagnostic sequence (ISO 13374 blocks 1-4) with decision
gates at each step. This workflow supports the judgment of a qualified
vibration engineer — it never replaces it.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

**Never infer a fault from a signal id or filename** — "OuterRaceFault" in an
id is not evidence. Base every conclusion exclusively on tool outputs.

## Workflow

### Step 1 — Signal Discovery (ISO 13374 Block 1)

Call `list_signals(scope="memory")` to check which signal_ids are already
loaded, or `list_signals(scope="disk")` to browse loadable files under the
data/signals/ directory. If the target signal is not loaded, load it:

Call `load_signal(filepath="real_train/OuterRaceFault_1.csv", signal_unit="g")`

- The returned `signal_id` (derived from the relative path, e.g.
  `real_train_OuterRaceFault_1`) is the single handle for every later call.
- For raw binary files (`.bin`/`.raw`/`.dat`), also declare `sample_format` and
  `sampling_rate` (or provide a companion `<stem>_metadata.json`) — see the
  signal-management skill.
- Declare `signal_unit` ("g", "m/s2", "mm/s", or "m/s") if you know it — the
  ISO severity verdict in Step 7 is REFUSED without a declared unit. Units are
  never guessed. Ask the user; do not invent one.
- Verify with `get_signal_info(signal_id="<id>")`: sampling rate, duration,
  declared unit, and any companion metadata (rpm, reference frequencies).

If the sampling rate is missing, STOP and ask the user — do not guess.

### Step 2 — Statistical Screening (Block 2)

Call `analyze_statistics(signal_id="<id>")`.

Evaluate screening flags (excess kurtosis, Fisher convention):

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| Kurtosis > 0 | Mild | Non-Gaussian content (possible impulses) |
| Kurtosis > 3 | Moderate | Significant impulsive content |
| Kurtosis > 6 | Severe | Strong impulsiveness, consistent with bearing damage |
| Crest Factor > 4 | Mild | Impulsiveness present |
| Crest Factor > 6 | Strong | Strong impulsiveness |

**Decision gate**: If Kurtosis < 0 AND Crest Factor < 3, the signal shows no
impulsive content. Report "No bearing fault indicators in time-domain
screening" and ask whether to continue anyway — envelope analysis can still
reveal early faults.

### Step 3 — Spectral Analysis (Block 2)

Call `analyze_fft(signal_id="<id>")`.

Identify:
- Dominant frequencies and their harmonics
- Shaft frequency (1x RPM) and multiples
- Broadband energy increase

If the operating speed is unknown, ask the user for the RPM — it is required
for Steps 4 and 6.

### Step 4 — Bearing Characteristic Frequencies

Establish the expected BPFO/BPFI/BSF/FTF frequencies by ONE of these routes:

1. **Catalog**: `search_bearing_catalog(bearing_id="6205")` — returns verified
   geometry with its source citation, plus per-RPM fault frequency multipliers.
2. **Explicit geometry** (bearing not in catalog):
   `calculate_bearing_characteristic_frequencies(num_balls=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04, contact_angle_deg=0.0, rpm=1797)`
3. **Machine manual**: `extract_manual_specs(file_name="pump_manual.pdf")` to
   pull the bearing designation from documentation, then route 1.

If the bearing is not in the catalog and no geometry is available, ask the
user. Never fabricate geometry or frequencies.

Expected frequencies:
- **BPFO** — Ball Pass Frequency, Outer race (canonical fault: `outer_race`)
- **BPFI** — Ball Pass Frequency, Inner race (canonical fault: `inner_race`)
- **BSF** — Ball Spin Frequency (canonical fault: `ball`)
- **FTF** — Fundamental Train Frequency, cage (canonical fault: `cage`)

### Step 5 — Envelope Analysis (Block 2, primary evidence)

Call `analyze_envelope(signal_id="<id>", filter_low=500, filter_high=5000)`.

- Default demodulation band 500-5000 Hz suits general bearing analysis.
- If Step 3 found a structural resonance, center the band around it.
- The band must stay below Nyquist — an invalid band is an explicit error,
  never a silent clamp.

### Step 6 — Evidence Matching (Block 3)

Compare envelope peaks against the expected frequencies systematically:

Call `check_bearing_faults(signal_id="<id>", rpm=1797, bearing_id="6205")`

or, when frequencies came from geometry or the user:

Call `check_bearing_faults(signal_id="<id>", rpm=1797, frequencies={"BPFO": 107.4, "BPFI": 162.2})`

| Peak matches (±5%) | Harmonics | + High Kurtosis | Diagnosis |
|---|---|---|---|
| BPFO | 2x, 3x present | Yes | **Possible outer race fault** |
| BPFI | + sidebands at shaft freq | Yes | **Possible inner race fault** |
| BSF | 2x present | Yes | **Possible ball defect** |
| FTF | Irregular spacing | Moderate | **Possible cage fault** |
| No matches | — | — | **Inconclusive** |

Evidence strength language:
- **Strong evidence**: peak + harmonics + corroborating statistics
- **Possible / consistent with**: peak present but missing corroboration
- **Inconclusive**: insufficient evidence — say so plainly

### Step 7 — ISO Severity (Block 4)

Call `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`

- `machine_group`: 1 (large, >300 kW) or 2 (medium, 15-300 kW). Confirm with
  the user; pass `machine_power_kw` if known (< 15 kW is out of ISO scope and
  is refused).
- Report the zone (A/B/C/D) and RMS velocity in mm/s.
- If the verdict is refused because no unit is declared, re-load with
  `load_signal(filepath="<file>", signal_unit="g", overwrite=True)` after
  confirming the unit with the user.
- Severity is machine-level context, NOT bearing-specific evidence.

**One-call alternative**: `diagnose_vibration(signal_id="<id>", rpm=1797, bearing_id="6205", machine_group=2, support_type="rigid")`
runs FFT + PSD + STFT + bearing checks + ISO severity in one pass and degrades
to a structured ISO refusal (reason + remedy) when the unit is undeclared.

### Step 8 — Report Generation (Block 6)

Call `generate_envelope_report(signal_id="<id>", bearing_freqs={"BPFO": 107.4, "BPFI": 162.2})`
and `generate_fft_report(signal_id="<id>", rpm=1797)`.
Optionally `generate_diagnostic_report_docx(signal_id="<id>", sections={"summary": "..."})`
for a Word document. Report the returned file paths to the user.

## Troubleshooting

**No peaks in envelope spectrum** — Wrong demodulation band or signal too
short. Try a different band within Nyquist (e.g. `filter_low=1000,
filter_high=4000`) guided by the FFT resonances.

**Unknown bearing type** — Ask for the designation (e.g. "6205") and call
`search_bearing_catalog(bearing_id="6205")`. If it is not in the catalog, the
result says so with a suggestion — ask the user for geometry; never invent it.

**Kurtosis high but no envelope peaks** — Impulses may come from a non-bearing
source (gear mesh, electrical). Suggest the gear-diagnosis skill or check for
line-frequency harmonics.

**ISO verdict refused** — The signal has no declared unit. Confirm the unit
with the user and re-load with `load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`.

## Important Notes

- Always use cautious diagnostic language: "consistent with", "possible fault"
- Never claim a definitive fault without multiple corroborating indicators
- This skill augments and accelerates expert judgment; the final maintenance
  decision rests with a qualified engineer
- All processing happens locally — raw signal data never leaves the machine
