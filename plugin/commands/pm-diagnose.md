---
description: Start a bearing fault diagnosis workflow on a vibration signal
argument-hint: "[signal_id or file_path]"
allowed-tools:
  - Read
  - Bash
---

Run the bearing fault diagnosis workflow on the specified signal.

If the user provided a signal_id or file path as argument, use it directly.
Otherwise, call `list_signals(scope="memory")` to show loaded signals (and
`list_signals(scope="disk")` for loadable files) and ask the user to choose.

Follow the bearing-diagnosis skill workflow:
1. Load/verify the signal: `load_signal(filepath="<file>", signal_unit="g")`
   (declare the unit only if known; ask, never guess; raw binary
   `.bin`/`.raw`/`.dat` files also need `sample_format` and `sampling_rate`, or
   a companion `<stem>_metadata.json`), then
   `get_signal_info(signal_id="<id>")`
2. Statistical screening: `analyze_statistics(signal_id="<id>")`
3. FFT analysis: `analyze_fft(signal_id="<id>")`
4. Bearing characteristic frequencies: ask for the bearing designation if
   unknown, then `search_bearing_catalog(bearing_id="6205")` or
   `calculate_bearing_characteristic_frequencies(num_balls=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04, rpm=1797)`
5. Envelope line test (the bearing evidence): `check_bearing_faults(signal_id="<id>", rpm=1797, bearing_id="6205")`
6. ISO 20816-3 severity: `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`
7. Reports: `generate_envelope_report(signal_id="<id>", rpm=1797)` +
   `generate_fft_report(signal_id="<id>")`, optionally
   `generate_diagnostic_report_docx(signal_id="<id>", sections={"Summary": "..."})`.
   The envelope report's `bearing_matches` and `bearing_assessment` are not
   the bearing evidence: report the claim of `check_bearing_faults`, and with
   a server whose check returns no `claim`, relay neither.

Present a concise summary at the end: the bearing claim in the server's words
(`claim.level`, with `overall_assessment` quoted verbatim), never your own
matching of envelope peaks; the outcome as returned (`no_indication`,
`inconclusive`, `indicated` or `several_types`); the ISO zone; and only the
actions the tools return or the workflow lists. `claim.level`, not `outcome`,
decides whether a fault type is named: single one out only at
`single_spectrum_evidence`, as envelope evidence of one spectrum, never as a
confirmed fault. `outcome_reason` and `next_step` describe the evidence
outcome before the claim: when `claim.level` is not
`single_spectrum_evidence`, do not relay them to name a fault type or a
single element. A single-spectrum indication asks for confirmation, never a
replacement, and `no_indication` is not evidence that the bearing is free of
defects. If a `check_bearing_faults` result, or a non-null `bearing_faults`
block of a `diagnose_vibration` result, has no `claim` field, the server
predates the bearing claim: infer or relay no bearing fault type or bearing
assessment from it (its `overall_assessment`, `most_likely_fault`,
`fault_checks`, bearing sentence or Bearing recommendations), and tell the
user that this bearing guidance needs a server whose `check_bearing_faults`
returns `claim`. The `evidence_strength` of such a `diagnose_vibration`
result, and its "Fault evidence strength" sentence, also count the bearing
finding, so neither is relayed; results that do not rest on the bearing
check, such as the ISO 20816-3 zone and its action, are relayed as usual.
Use cautious language: the result supports the engineer's decision, it does
not replace it.
