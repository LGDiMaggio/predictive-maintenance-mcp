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

### Step 1: Signal Discovery (ISO 13374 Block 1)

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
  ISO severity verdict in Step 6 is REFUSED without a declared unit. Units are
  never guessed. Ask the user; do not invent one.
- Verify with `get_signal_info(signal_id="<id>")`: sampling rate, duration,
  declared unit, and any companion metadata (rpm, reference frequencies).

If the sampling rate is missing, STOP and ask the user — do not guess.

### Step 2: Statistical Screening (Block 2)

Call `analyze_statistics(signal_id="<id>")`.

Evaluate screening flags (excess kurtosis, Fisher convention):

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| Kurtosis > 0 | Mild | Non-Gaussian content (possible impulses) |
| Kurtosis > 3 | Moderate | Significant impulsive content |
| Kurtosis > 6 | Severe | Strong impulsiveness |
| Crest Factor > 4 | Mild | Impulsiveness present |
| Crest Factor > 6 | Strong | Strong impulsiveness |

These flags are screening only: they name no source and never raise the
bearing claim of Step 5.

**Decision gate**: If Kurtosis < 0 AND Crest Factor < 3, the signal shows no
impulsive content. Report "No impulsive content in time-domain screening; the
bearing is not assessed by it" and ask whether to continue anyway: the
envelope line test of Step 5 can still find lines that screening misses.

### Step 3: Spectral Analysis (Block 2)

Call `analyze_fft(signal_id="<id>")`.

Identify:
- Dominant frequencies and their harmonics
- Shaft frequency (1x RPM) and multiples
- Broadband energy increase

If the operating speed is unknown, ask the user for the RPM — it is required
for Steps 4 and 5.

### Step 4: Bearing Characteristic Frequencies

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

### Step 5: Envelope Line Test (Block 3)

The check tests the line at each expected frequency against the local noise
floor of one envelope spectrum, then the further lines its kinematics
predict. Its result, never your own matching of envelope peaks, decides what
may be said about a fault type.

Call `check_bearing_faults(signal_id="<id>", rpm=1797, bearing_id="6205")`

or, when frequencies came from geometry or the user:

Call `check_bearing_faults(signal_id="<id>", rpm=1797, frequencies={"BPFO": 107.4, "BPFI": 162.2})`

A `BSF` value is the ball spin frequency: the rolling-element line is tested
at twice it (`line_frequency_hz`).

Report the result in the server's words:

| `claim.level` | What to report |
|---|---|
| `single_spectrum_evidence` | Quote `overall_assessment`: "the envelope evidence of this single spectrum is consistent with [type]", with its levels and its stated assumption; it is not a confirmed fault and asks for confirmation (a comparable earlier measurement or an inspection of that element), never a replacement |
| `qualified` | Quote `overall_assessment` verbatim: it names every reading the claim cannot exclude; single out none |
| `withheld` | Name no fault type; quote `overall_assessment` verbatim: it says why no fault type is assessed |

- Name the outcome as returned: `no_indication`, `inconclusive`, `indicated`
  or `several_types`. `claim.level`, not `outcome`, decides whether a fault
  type is named: an `indicated` outcome can carry a qualified claim.
- `outcome_reason` and `next_step` describe the evidence outcome before the
  claim: when `claim.level` is not `single_spectrum_evidence`, do not relay
  them to name a fault type or a single element.
- Never report `no_indication` as "healthy" or "no fault": quote its
  `overall_assessment` verbatim; `no_indication` is not evidence that the
  bearing is free of defects.
- Each check's `evidence_strength` (`none`, `low`, `moderate`, `high`) grades
  lines of one envelope spectrum: it is not a probability. Statistics and
  harmonics you read elsewhere never raise the claim.
- If a `check_bearing_faults` result, or a non-null `bearing_faults` block of
  a `diagnose_vibration` result, has no `claim` field, the server predates
  the bearing claim: infer or relay no bearing fault type or bearing
  assessment from it (its `overall_assessment`, `most_likely_fault`,
  `fault_checks`, bearing sentence or Bearing recommendations), and tell the
  user that this bearing guidance needs a server whose `check_bearing_faults`
  returns `claim`. The `evidence_strength` of such a `diagnose_vibration`
  result, and its "Fault evidence strength" sentence, also count the bearing
  finding, so neither is relayed; results that do not rest on the bearing
  check, such as the ISO 20816-3 zone and its action, are relayed as usual.

### Step 6: ISO Severity (Block 4)

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

Its `overall_diagnosis` contains the claim's sentence
("Bearing analysis: ..."), and the "Bearing" entries of its `recommendations`
are the claim's actions: quote them when its `bearing_faults` block carries a
`claim`. Its `evidence_strength` (`none`, `weak`, `moderate`, `strong`)
counts the findings of every analysis that ran, the ISO zone among them: it
is not a probability and not the strength of a bearing finding.

If a `check_bearing_faults` result, or a non-null `bearing_faults` block of a
`diagnose_vibration` result, has no `claim` field, the server predates the
bearing claim: infer or relay no bearing fault type or bearing assessment
from it (its `overall_assessment`, `most_likely_fault`, `fault_checks`,
bearing sentence or Bearing recommendations), and tell the user that this
bearing guidance needs a server whose `check_bearing_faults` returns
`claim`. The `evidence_strength` of such a `diagnose_vibration` result, and
its "Fault evidence strength" sentence, also count the bearing finding, so
neither is relayed; results that do not rest on the bearing check, such as
the ISO 20816-3 zone and its action, are relayed as usual.

### Step 7: Report Generation (Block 6)

Call `generate_envelope_report(signal_id="<id>", bearing_freqs={"BPFO": 107.4, "BPFI": 162.2}, rpm=1797)`
and `generate_fft_report(signal_id="<id>", rpm=1797)`.
Optionally `generate_diagnostic_report_docx(signal_id="<id>", sections={"summary": "..."})`
for a Word document. Report the returned file paths to the user.

The envelope report's `bearing_matches` and `bearing_assessment` are not the
bearing evidence: report the claim of `check_bearing_faults`, and with a
server whose check returns no `claim`, relay neither.

## Troubleshooting

**The check returns `no_indication`**: quote `overall_assessment`. Do not
re-run the check with another `tolerance_pct` or frequency set to get a
different outcome: a repeated check is a new test. Change a declared value
(speed, bearing data) only when the user corrects it.

**Unknown bearing type**: Ask for the designation (e.g. "6205") and call
`search_bearing_catalog(bearing_id="6205")`. If it is not in the catalog, the
result says so with a suggestion — ask the user for geometry; never invent it.

**Kurtosis high but the claim names no fault type**: the statistics do not
identify a source; bearing and non-bearing sources (for example gear mesh or
electrical) both remain possible. Suggest the gear-diagnosis skill or check
for line-frequency harmonics.

**ISO verdict refused**: The signal has no declared unit. Confirm the unit
with the user and re-load with `load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`.

## Important Notes

- Always use cautious diagnostic language: "consistent with", "possible fault"
- Neither `check_bearing_faults` nor the bearing block of `diagnose_vibration`
  confirms a bearing defect: never write "confirmed", "damage detected" or
  "fault present" for a bearing claim, however many indicators agree. If the
  results do not answer a question (for example whether the bearing is
  damaged), say that the analysis does not answer it
- This skill augments and accelerates expert judgment; the final maintenance
  decision rests with a qualified engineer
- All processing happens locally — raw signal data never leaves the machine
