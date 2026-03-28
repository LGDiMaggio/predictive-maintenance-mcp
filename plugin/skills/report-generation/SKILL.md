---
description: >
  Generate professional diagnostic reports (HTML, DOCX) for vibration analysis
  using the predictive-maintenance-mcp server. Use this skill when the user says
  "generate report", "create report", "export report", "diagnostic report",
  "save analysis", "create PDF", "archive results", "full documentation",
  "DOCX report", "HTML report", "comparison report", or "PCA report".
---

# Report Generation

Orchestrate predictive-maintenance-mcp tools to produce professional diagnostic
reports. Supports FFT spectrum, envelope analysis, ISO 20816-3 evaluation,
feature comparison, PCA visualization, and full DOCX reports.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Available Report Types

| Report Type | Tool | When to Use |
|---|---|---|
| FFT Spectrum | `generate_fft_report()` | Frequency content inspection |
| Envelope Analysis | `generate_envelope_report()` | Bearing fault frequency detection |
| ISO 20816-3 | `generate_iso_report()` | Vibration severity classification |
| Feature Comparison | `generate_feature_comparison_report()` | Multi-signal comparison |
| PCA Visualization | `generate_pca_visualization_report()` | Cluster/anomaly overview |
| Full DOCX | `generate_diagnostic_report_docx()` | Complete diagnostic document |

## Workflow

### Step 1 — Determine Report Type

Ask the user or infer from context which report(s) are needed. If the user asks
for a "full" or "comprehensive" report, generate multiple reports.

### Step 2 — Verify Signal Availability

Call `list_stored_signals()` to confirm signals are loaded. Load with
`load_signal(...)` if needed. For comparison reports, identify all signals.

### Step 3 — Gather Parameters

**FFT Report**: signal_id (required), max_freq (optional Hz)

**Envelope Report**: signal_id (required), bearing_name (optional for frequency
overlay), filter_low / filter_high (optional bandpass bounds)

**ISO 20816-3 Report**: signal_id (required), machine_group (1–4, default 2),
support_type ("rigid" or "flexible"), signal_unit (critical — ask user)

**Feature Comparison**: signal_ids (list), label (category label)

**PCA Visualization**: signal_ids (list of signal IDs for projection)

**Full DOCX**: signal_id (required), title, machine_name, additional notes

### Step 4 — Generate Report

Call the appropriate tool. Each returns a report_path and key analysis data.

### Step 5 — Report to User

After generation:
```
Report generated successfully:
  Type: {report_type}
  File: reports/{filename}
  Signal: {signal_id}

Key findings:
  {2-3 bullet summary}

Open the HTML file in a browser for the full interactive report.
```

### Step 6 — Composite Reports (Optional)

For "full" or "comprehensive" requests, generate in sequence:
1. FFT Spectrum report
2. Envelope Analysis report
3. ISO 20816-3 report
4. Full DOCX report (combines everything)

Provide a combined summary referencing all reports.

## Important Notes

- NEVER display raw HTML content in chat — always provide the file path
- Signal unit confirmation is CRITICAL for ISO reports
- Reports are self-contained HTML with embedded Plotly charts
- DOCX reports include embedded images and can be shared externally
- All processing is local — reports contain only processed results, not raw data
