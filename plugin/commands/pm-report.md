---
description: Generate a diagnostic report for a vibration signal
argument-hint: "[signal_id] [report_type: fft|envelope|iso|comparison|pca|docx|full]"
allowed-tools:
  - Read
  - Bash
---

Generate a diagnostic report for the specified signal.

Parse the arguments:
- First argument: signal_id or file_path
- Second argument (optional): report type. Defaults to "full" if not specified.

If no arguments provided, call `list_stored_signals()` and ask the user which
signal and report type they want.

Report types:
- **fft**: Call `generate_fft_report(signal_id=...)`
- **envelope**: Call `generate_envelope_report(signal_id=...)`
- **iso**: Call `generate_iso_report(signal_id=..., machine_group=2, support_type="rigid")`
- **comparison**: Call `generate_feature_comparison_report(signal_ids=[...])` (ask for multiple signals)
- **pca**: Call `generate_pca_visualization_report(signal_ids=[...])` (ask for multiple signals)
- **docx**: Call `generate_diagnostic_report_docx(signal_id=...)`
- **full**: Generate FFT + envelope + ISO + DOCX reports in sequence

For ISO reports, confirm signal units with the user before generating.

After generation, report the file paths and key findings from each report.
