---
description: >
  Experimental shaft-order evidence assessment for unbalance, misalignment and
  looseness using the predictive-maintenance-mcp server. Use this skill when the
  user asks whether a machine shows unbalance (imbalance), misalignment or
  mechanical looseness, asks what a 1x or 2x line, a harmonic series of the
  shaft speed or half-order lines could mean, or wants machine-level evidence
  next to the bearing check. It reports which spectral patterns are observed
  and what they cannot establish; it never attributes or confirms a fault.
---

# Shaft-Order Evidence (experimental)

`assess_shaft_fault_evidence` measures the lines at integer and half shaft
orders of one to six channels of one machine at one operating condition. For
unbalance, misalignment and looseness it reports whether the pattern the server
uses for each family is observed, what conflicts with it, which measurements
are missing and what else could explain it. It attributes no fault: the
patterns are not validated as fault detectors. The result supports the
engineer's judgment; it never replaces it.

**Prerequisite**: a predictive-maintenance-mcp server that registers
`assess_shaft_fault_evidence` (main branch; not yet in a published release). If
the tool is not available, tell the user that the connected server does not
provide it. Do not substitute your own reading of FFT peaks or a strongest
line at 1x or 2x reported by `diagnose_vibration`.

**Optional**: use it when the question is about unbalance, misalignment or
looseness, or about what shaft-order lines could mean. A bearing check, a gear
check, anomaly detection, prognostics or a quick screening does not need it.

## Workflow

### Step 1 — Channels and context

Find the channels with `list_signals(scope="memory")`. Use channels of one
machine at one operating condition, for example horizontal, vertical and axial
at one bearing. Ask the user for what is missing; never guess:

- the shaft speed in rpm, unless the signals or their measurement point declare
  it;
- the direction of each channel (horizontal, vertical or axial), unless the
  signal's measurement metadata declares it; a direction declared only for the
  measurement point in the asset ledger is not read, so ask the user for it.
  Pass null for a direction nobody declares: the patterns of that channel
  then stay `inconclusive` with `direction_undeclared`;
- the signal unit, declared at load time with
  `load_signal(filepath="<file>", signal_unit="<unit the user gives>")`.
  Without it, unbalance and misalignment have `insufficient_data`
  (`unit_undeclared`); looseness is still assessed;
- optionally the ISO 20816-3 machine group, support type and machine power, for
  the severity context;
- for a motor on a variable-frequency drive, the drive's output frequency.

### Step 2 — Assessment

Call `assess_shaft_fault_evidence(signal_ids=["<h>", "<v>", "<a>"])`. The
server takes the speed, the machine group, the support type and the machine
power from what the signals or their measurement point declare, and each
channel's direction from what its signal declares; `parameter_sources` and
each `direction_source` say where each value came from.

Add a parameter only when the user supplies it and it applies to these
channels:

- `rpm`: without a declared speed the call is refused; ask the user for the
  shaft speed and call again with it. Never estimate it.
- `directions`: one entry per channel, in the order of `signal_ids`, with
  null for a channel whose direction the user does not give;
- `machine_group`, `support_type`, `machine_power_kw`: the ISO 20816-3
  machine class, for the severity context;
- `line_frequency_hz`: the supply frequency; for an inverter-fed motor, the
  drive's output frequency;
- `speed_tolerance_pct`: smaller than the default only when the user states
  the accuracy of a measured speed, for example from a tachometer;
- `max_order`: only when the user asks for a different highest order;
- `detail`: keep the default `"summary"`; use
  `assess_shaft_fault_evidence(signal_ids=["<id>"], detail="full")` only when
  the user needs every order line or the full policy and source texts.

### Step 3 — Read the result

- Quote `summary` and each hypothesis `statement` verbatim; never restate them
  more strongly.
- For each family in `hypotheses` (unbalance, misalignment, looseness,
  resonance):
  - `pattern_observed` true, status `inconclusive`: the pattern is observed and
    no fault is attributed; its `reasons` always include
    `attribution_not_validated`. Say "the pattern this assessment uses for
    unbalance is observed", never, from this result, that unbalance is
    present, possible or suspected.
  - `pattern_observed` false, status `not_observed`: the pattern is absent,
    which does not exclude the fault.
  - status `insufficient_data` (`pattern_observed` null): the assessment says
    nothing about that family; relay its `reasons`.
  - resonance is always `not_assessable_from_record` (`pattern_observed` null).
- Relay the `supporting` and `conflicting` observations with their values, and
  the `alternatives`, which the records cannot exclude, with every observed
  pattern.
- Relay `limitations` as stated with every observed pattern: they say what the
  result cannot establish, including how often the patterns appeared in
  records of the normal condition and the measured false-alarm rate of the
  line test.
- `level_context` holds the ISO 20816-3 zone: context only. A zone never
  identifies a fault, and zone A or B does not exclude one.
- `parameter_sources` and each channel's `direction_source` say where the speed,
  the machine class and the directions came from. Mention any declared value
  the user did not give.

### Step 4 — Next steps

Name only the measurements in `missing` and the checks the result itself lists,
as input to the engineer's decision.

## What stays open

This result attributes nothing, and it does not stop the discussion: you may
explain how unbalance, misalignment and looseness typically show in
vibration, name them as possible explanations next to the `alternatives`,
quote a manual as background, and recommend the measurements in `missing`
that would distinguish them.

## Never

- Never write that unbalance, misalignment or looseness is present, possible,
  suspected, detected, confirmed or likely on the strength of this result.
- `diagnose_vibration` reads the same spectral lines of the same signal: the
  agreement of the two repeats the observation and is not corroboration.
- A manual, a catalogue or another reference that describes a compatible
  pattern is background, never independent evidence that the observed pattern
  is that fault.
- Never pass a family from this result to `generate_maintenance_recommendations`
  as a fault type: its fault types are for faults established independently,
  for example by inspection.
- A significant line is not abnormality: shaft-order lines are present in almost
  every machine.
- Never guess the rpm, the unit, the direction or the machine class.
- All processing is local; raw data never leaves the machine.
