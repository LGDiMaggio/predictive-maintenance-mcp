---
name: quick-screening
description: >
  Fast vibration health screening for industrial machinery using the
  predictive-maintenance-mcp server. Use this skill when the user says "quick
  check", "health screening", "vibration check", "is this machine healthy",
  "overall health", "quick diagnostic", "machine status", "condition monitoring",
  "health assessment", or wants a rapid overview of machine condition without
  full fault identification.
---

# Quick Vibration Screening

Fast vibration screening for rotating machinery. Produces a screening
card in under 30 seconds with clear next-step recommendations. Screening
informs the engineer's decision — it is never a diagnosis by itself.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Step 1 — Signal Selection

Call `list_signals(scope="memory")` to show loaded signal_ids, or
`list_signals(scope="disk")` for loadable files. If the user has not
specified one, ask. Load with
`load_signal(filepath="<file>", signal_unit="g")` if
needed — declaring `signal_unit` up front makes the ISO step work without a
re-load. For raw binary files (`.bin`/`.raw`/`.dat`), also declare
`sample_format` and `sampling_rate` (or provide a companion
`<stem>_metadata.json`) — see the signal-management skill.

### Step 2 — Statistical Features

Call `analyze_statistics(signal_id="<id>")`.

Report as bullet points:
- **RMS**: energy level
- **Crest Factor**: impulsiveness indicator
- **Kurtosis**: excess kurtosis (Fisher convention)
- **Peak-to-Peak**: overall vibration range

Report them as the numbers returned. They are screening context only: they
identify no fault and are not a health verdict.

### Step 3 — FFT Snapshot

Call `analyze_fft(signal_id="<id>")`.

Report:
- Peak frequency and magnitude
- Top 3 spectral peaks (frequency + magnitude)
- Any obvious harmonic patterns

A strongest line at 1x or 2x, or a harmonic series, names no fault on its own.
If the user asks about unbalance, misalignment or looseness, the
shaft-order-evidence skill gives an experimental evidence assessment that
attributes no fault. It is optional and not part of the screening card.

### Step 4 — ISO Severity Assessment

Call `assess_severity(signal_id="<id>", machine_group=<group>, support_type="<support type>")`.

Report:
- RMS velocity in mm/s
- The zone (A, B, C or D) with its `zone_description`, as returned

If the assessment is refused, relay its reason and remedy as returned, and
give no zone, zone action or verdict in its place.

Notes:
- Pass machine_group (1 = large >300 kW, 2 = medium 15-300 kW) and
  support_type (rigid or flexible) as the user states them; `assess_severity`
  does not read a declared measurement point. When they were not stated,
  leave them out and say that the zone uses the default group 2 / rigid.
- The verdict requires a DECLARED signal unit. If it is refused, re-load with
  `load_signal(filepath="<file>", signal_unit="g", overwrite=True)` after
  confirming the unit with the user — units are never guessed.
- If only a portable-instrument reading is available (no signal file), use
  `assess_severity(rms_velocity_mm_s=3.2, machine_group=2, support_type="rigid")`.

### Step 5 — Summary Card

Format the output as a concise screening card, with every value as the tools
return it:

```
VIBRATION HEALTH SCREENING
===========================
Signal: {signal_id}

Key Indicators:
  RMS: {value}  |  CF: {value}  |  Kurt: {value}
  ISO Zone: {zone} ({zone_description}), or the refusal's reason and remedy

Zone action:
  {the output of generate_maintenance_recommendations}

Next steps:
  {the next steps below that apply}
```

For the zone action, call `generate_maintenance_recommendations(severity_zone="C")`
with the zone letter the assessment returned, and relay its output as
returned. Pass no `fault_types`: a screening establishes no fault. A refused
assessment gives no zone: skip the call.

Next steps:
- A bearing question, or a bearing designation the user gives: the
  bearing-diagnosis skill
- A question about unbalance, misalignment or looseness: the optional
  shaft-order-evidence skill (Step 3)

## Important Notes

- This is a SCREENING tool, NOT a diagnostic tool
- Never make definitive fault claims from screening alone
- Suggest targeted analysis only as the next steps of Step 5 give it
- Use cautious language ("indicators suggest"): a screening result is not a
  confirmed fault, and a line at
  1x or 2x attributes no unbalance, misalignment or looseness; they may be
  named as possible explanations next to the others (Step 3)
- The screening supports the engineer's judgment; it does not replace it
- All processing happens locally — raw data never leaves the machine
