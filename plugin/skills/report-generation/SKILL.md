---
name: report-generation
description: >
  Generate diagnostic reports (the server-written report in HTML or PDF, report
  files, DOCX) and multi-signal comparisons for vibration analysis using the
  predictive-maintenance-mcp server. Use this skill when the user says
  "generate report", "create report", "export report", "diagnostic report",
  "save analysis", "create PDF", "archive results", "full documentation",
  "DOCX report", "HTML report", "comparison report", "PCA report", "compare
  these signals", or "explore these signals".
---

# Report Generation

Orchestrate predictive-maintenance-mcp tools to produce professional diagnostic
reports. Supports the server-written diagnostic report, time-domain plots,
FFT spectrum, envelope analysis, ISO 20816-3 evaluation, feature comparison,
PCA visualization, and DOCX documents.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Available Report Types

| Report Type | Tool | When to Use |
|---|---|---|
| Diagnostic report (HTML, PDF) | `generate_diagnostic_report()` | Full, comprehensive or PDF report; the server writes every sentence |
| Time-domain plot | `plot_signal()` | Waveform inspection |
| FFT Spectrum | `generate_fft_report()` | Frequency content inspection |
| Envelope Analysis | `generate_envelope_report()` | Bearing fault frequency detection |
| ISO 20816-3 | `generate_iso_report()` | Vibration severity classification |
| Feature Comparison | `generate_feature_comparison_report()` | Multi-signal comparison |
| PCA Visualization | `generate_pca_visualization_report()` | Anomaly-model cluster overview |
| Full DOCX | `generate_diagnostic_report_docx()` | Complete diagnostic document |

## Workflow

### Step 1 — Determine Report Type

Ask the user or infer from context which report(s) are needed. A "full",
"comprehensive" or "PDF" request is the diagnostic report of Step 5. A request
to compare or explore several signals is the Multi-Signal Comparison below.

### Step 2 — Verify Signal Availability

Call `list_signals(scope="memory")` to confirm the signals are loaded. Load
with `load_signal(filepath="<file>", signal_unit="g")` if needed (raw binary
`.bin`/`.raw`/`.dat` files additionally need `sample_format` and
`sampling_rate` — see the signal-management skill). All report
tools take `signal_id` — the handle returned by load_signal.

### Step 3 — Gather Parameters and Generate

Values in angle brackets are placeholders, not your machine's values: fill
them only from what `get_signal_info` reports for the signal or from what the
user states (`rpm` in revolutions per minute), and ask when neither gives
them.

**Time-domain plot**:
`plot_signal(signal_id="<id>", time_range=[0.0, 2.0], show_statistics=True)`

**FFT report**:
`generate_fft_report(signal_id="<id>", max_freq=5000, num_peaks=15, rpm=<rpm>)`
(`rpm` is optional — when given, harmonic markers are drawn at 1x/2x/3x RPM)

**Envelope report**:
`generate_envelope_report(signal_id="<id>", filter_low=500, filter_high=5000, max_freq=500, bearing_freqs=<reference frequencies>)`
(`bearing_freqs` overlays expected fault frequency markers. When the
connected server's schema for this tool lists an `rpm` parameter, the report
takes the shaft speed the signal declares, and `rpm=<rpm>` passes a measured
speed the user states, which overrides the declared one; without a speed the
shaft-order and sideband tests do not run. If the schema does not list `rpm`,
omit it, and do not say that the report used a shaft speed or performed
shaft-order or sideband checks)

`bearing_matches` and `bearing_assessment` are not the bearing evidence:
report the claim of `check_bearing_faults`, and with a server whose check
returns no `claim`, relay neither.

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

**ISO 20816-3 report**:
`generate_iso_report(signal_id="<id>", machine_group=<group>, support_type="<support type>")`
- machine_group is 1 (large, >300 kW) or 2 (medium, 15-300 kW) and
  support_type is rigid or flexible: pass them as the user states them;
  `generate_iso_report` does not read a declared measurement point. When they
  were not stated, leave them out and say that the zone uses the default
  group 2 / rigid
- The signal must have a DECLARED unit; if the assessment is refused, re-load
  with `load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`
  after confirming the unit with the user.

**Feature comparison and PCA visualization**: see Multi-Signal Comparison
below.

**Full DOCX** (sections maps section titles to content):
`generate_diagnostic_report_docx(signal_id="<id>", title="Pump P-101 diagnostic", sections={"Summary": "...", "Findings": "..."})`

### Step 4 — Report to User

After generation:
```
Report generated successfully:
  Type: {report_type}
  File: {returned file path}
  Signal: {signal_id}

Key findings:
  {2-3 bullet summary}

Open the HTML file in a browser for the full interactive report.
```

Report filenames are timestamped — consecutive runs never overwrite each
other. Call `list_html_reports()` to list all generated reports, or
`list_html_reports(file_name="<report>.html")` for one report's details.

### Step 5: Full, Comprehensive or PDF Report

For a "full", "comprehensive" or "PDF" request, call
`generate_diagnostic_report(signal_id="<id>", rpm=<rpm>, bearing_id="<bearing designation>", formats=["html"])`:
one document whose every sentence the server writes.

- `formats` takes "html", "pdf" or both. A PDF needs the server's optional
  PDF extra; when the server refuses it, relay its message.
- `rpm` is required. Pass `bearing_id` only when a bearing designation is
  known, `baseline_signal_id` only when the user names a stored signal of the
  same machine in a known-good state, and `machine_group` and `support_type`
  as the user states them: `generate_diagnostic_report` does not read a
  declared measurement point. When they were not stated, leave them out and
  say that the zone uses the default group 2 / rigid.
- Relay its `statements` verbatim and in order, its `verdict` and
  `recommendations` as returned, and the paths in `files`. Add no sentence,
  zone, confidence or fault type of your own to them; when the result does
  not answer a question, say so.
- A result without a `bearing_check` field comes from a server that predates
  the bearing claim: relay none of its bearing sentences, its `verdict`, its
  bearing recommendations or its evidence strength, and say that this bearing
  guidance needs a server whose `check_bearing_faults` returns `claim`. Its
  other statements, such as the ISO 20816-3 zone, are relayed as usual.

## Multi-Signal Comparison

For a request to compare or explore several signals:

1. Load every signal in one batch, which is atomic:
   `load_signal(filepath=["<file A>", "<file B>", "<file C>"])`, adding
   `signal_unit` only when the user states it (raw binary files also need
   `sample_format` and `sampling_rate`, or a companion
   `<stem>_metadata.json`)
2. Call `analyze_statistics(signal_id="<id>")` for each signal
3. Call `generate_feature_comparison_report(signal_groups={"<group name>": ["<id>", "<id>"], "<other group name>": ["<id>"]}, features_to_plot=["rms", "kurtosis", "crest_factor"])`.
   `signal_groups` maps a group name to the signal_ids in it. The group names
   are the user's: ask which signals belong together, and never name a group
   after a condition the tools did not establish
4. Only when a trained anomaly model exists and the user names it, add
   `generate_pca_visualization_report(model_name="<model name>", test_signal_ids=["<id>", "<id>"])`.
   It segments each signal with the model's training `segment_duration` and
   `overlap_ratio`, and its `true_labels`, when the user supplies them, take
   only the labels its description lists. Training a model is a step of the
   anomaly-detection skill: suggest it to the user, and do not train one
   yourself in this workflow

Report each feature's values per signal, and their range across the signals,
as the numbers the tools return, with the report paths. Label no signal and
no group by its values (such as normal, impulsive, tonal or outlier), and do
not single out a signal as different from the others: the comparison report
shows the values for the engineer to judge.

## Important Notes

- NEVER display raw HTML content in chat — always provide the file path
- Signal unit declaration is CRITICAL for ISO reports (no unit = refusal)
- Reports are self-contained HTML with embedded Plotly charts
- DOCX reports include embedded images and can be shared externally
- Reports document the evidence for an engineer's decision — they do not
  replace engineering judgment
- All processing is local — reports contain only processed results, not raw data
