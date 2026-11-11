---
name: report-generation
description: >
  Generate professional diagnostic reports (HTML, DOCX) for vibration analysis
  using the predictive-maintenance-mcp server. Use this skill when the user says
  "generate report", "create report", "export report", "diagnostic report",
  "save analysis", "create PDF", "archive results", "full documentation",
  "DOCX report", "HTML report", "comparison report", or "PCA report".
---

# Report Generation

Orchestrate predictive-maintenance-mcp tools to produce professional diagnostic
reports. Supports time-domain plots, FFT spectrum, envelope analysis,
ISO 20816-3 evaluation, feature comparison, PCA visualization, and full DOCX
documents.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Available Report Types

| Report Type | Tool | When to Use |
|---|---|---|
| Time-domain plot | `plot_signal()` | Waveform inspection |
| FFT Spectrum | `generate_fft_report()` | Frequency content inspection |
| Envelope Analysis | `generate_envelope_report()` | Bearing fault frequency detection |
| ISO 20816-3 | `generate_iso_report()` | Vibration severity classification |
| Feature Comparison | `generate_feature_comparison_report()` | Multi-signal comparison |
| PCA Visualization | `generate_pca_visualization_report()` | Anomaly-model cluster overview |
| Full DOCX | `generate_diagnostic_report_docx()` | Complete diagnostic document |

## Workflow

### Step 1 — Determine Report Type

Ask the user or infer from context which report(s) are needed. For a "full" or
"comprehensive" request, generate multiple reports in sequence.

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

**Feature comparison** (signal_groups maps a label to a list of signal_ids):
`generate_feature_comparison_report(signal_groups={"baseline": ["baseline_1"], "faulty": ["OuterRaceFault_1"]}, features_to_plot=["rms", "kurtosis"])`

**PCA visualization** (requires a model trained via train_anomaly_model):
`generate_pca_visualization_report(model_name="anomaly_model", test_signal_ids=["<id>"])`

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

### Step 5 — Composite Reports (Optional)

For "full" or "comprehensive" requests, generate in sequence:
1. FFT spectrum report
2. Envelope analysis report
3. ISO 20816-3 report
4. Full DOCX report (synthesizes everything)

Provide a combined summary referencing all report paths.

## Important Notes

- NEVER display raw HTML content in chat — always provide the file path
- Signal unit declaration is CRITICAL for ISO reports (no unit = refusal)
- Reports are self-contained HTML with embedded Plotly charts
- DOCX reports include embedded images and can be shared externally
- Reports document the evidence for an engineer's decision — they do not
  replace engineering judgment
- All processing is local — reports contain only processed results, not raw data
