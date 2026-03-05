---
name: report-generation
description: >
  Generate professional HTML diagnostic reports for vibration analysis.
  Use when user says "generate report", "create report", "export report",
  "diagnostic report", "save analysis", "create PDF", "archive results",
  or "full documentation". Orchestrates multiple analysis tools and
  produces self-contained HTML reports saved in reports/ directory.
metadata:
  author: Luigi Di Maggio
  version: 1.0.0
  mcp-server: predictive-maintenance-mcp
  category: predictive-maintenance
compatibility: >
  Requires predictive-maintenance-mcp MCP server connected.
  Works on Claude.ai, Claude Code, and API.
  Reports saved as HTML files in reports/ directory.
---

# Report Generation Skill

Orchestrates predictive-maintenance-mcp tools to produce professional
HTML diagnostic reports. Supports FFT spectrum, envelope analysis,
ISO 20816-3 evaluation, and feature comparison reports.

## Instructions

### Step 1: Determine Report Type

Ask user or infer from context which report(s) are needed:

| Report Type          | Tool                           | When to Use                      |
|---------------------|-------------------------------|----------------------------------|
| FFT Spectrum        | `generate_fft_report()`       | Frequency content inspection     |
| Envelope Analysis   | `generate_envelope_report()`  | Bearing fault frequency detection|
| ISO 20816-3         | `generate_iso_report()`       | Vibration severity classification|
| Feature Comparison  | `generate_feature_comparison_report()` | Multi-signal comparison  |
| PCA Visualization   | `generate_pca_visualization_report()`  | Cluster/anomaly overview |

### Step 2: Verify Signal Availability

Call `list_signals()` to confirm the target signal exists.
If user wants a comparison report, identify all signals to include.

### Step 3: Gather Parameters

Each report type needs specific parameters:

**FFT Report:**
- signal_file (required)
- max_freq (optional, Hz — default auto)

**Envelope Report:**
- signal_file (required)
- bearing_name (optional — for characteristic frequency overlay)
- lowcut, highcut (optional — bandpass filter bounds)

**ISO 20816-3 Report:**
- signal_file (required)
- machine_group (1-4, default 2)
- support_type ("rigid" or "flexible")
- signal_unit (critical — ask user if unknown)

**Feature Comparison:**
- signal_files (list of signals)
- label (category label)

### Step 4: Generate Report

Call the appropriate tool. Each tool returns:
- report_path: file location in reports/
- key analysis data in structured format

### Step 5: Report to User

After generation, inform user:

```
Report generated successfully:
  Type: {report_type}
  File: reports/{filename}.html
  Signal: {signal_file}
  
Key findings:
  {2-3 bullet summary of results}

Open the HTML file in a browser for the full interactive report.
```

### Step 6: Composite Reports (Optional)

If user requests a "full" or "comprehensive" report, generate 
multiple reports in sequence:

1. FFT Spectrum report (frequency content)
2. Envelope Analysis report (bearing fault detection)
3. ISO 20816-3 report (severity assessment)

Then provide a combined summary:

```
COMPREHENSIVE DIAGNOSTIC REPORTS
=================================
Signal: {filename}

Generated Reports:
  1. reports/{fft_report}.html — Frequency spectrum
  2. reports/{envelope_report}.html — Envelope analysis  
  3. reports/{iso_report}.html — ISO 20816-3 evaluation

Combined Assessment:
  {3-5 line summary referencing all three analyses}
```

### Important Notes
- NEVER display HTML content in chat (wastes tokens)
- Always provide file path so user can open in browser
- Use `list_html_reports()` to show all previously generated reports
- Use `get_report_info("{report_file}")` to read metadata without consuming tokens
- Reports are self-contained HTML with embedded Plotly charts
- Signal unit confirmation is CRITICAL for ISO reports — wrong units invalidate results
