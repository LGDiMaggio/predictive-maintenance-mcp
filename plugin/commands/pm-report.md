---
description: Generate a diagnostic report for a vibration signal
argument-hint: "[signal_id] [report_type: fft|envelope|iso|comparison|pca|docx|full]"
allowed-tools:
  - Read
  - Bash
---

Generate a diagnostic report for the specified signal.

Parse the arguments:
- First argument: signal_id (or a file path to load first)
- Second argument (optional): report type. Defaults to "full".

If no arguments are provided, call `list_signals(scope="memory")` and ask the
user which signal and report type they want. Load files first with
`load_signal(filepath="<file>", signal_unit="g")` (declare the unit only if
known).

Report types:
- **fft**: `generate_fft_report(signal_id="<id>", max_freq=5000, num_peaks=15)`
- **envelope**: `generate_envelope_report(signal_id="<id>", filter_low=500, filter_high=5000)`
- **iso**: `generate_iso_report(signal_id="<id>", machine_group=2, support_type="rigid")`
- **comparison**: `generate_feature_comparison_report(signal_groups={"baseline": ["baseline_1"], "test": ["fault_1"]})`
  (signal_groups maps a group label to a list of signal_ids — ask which
  signals belong to which group)
- **pca**: `generate_pca_visualization_report(model_name="anomaly_model", test_signal_ids=["<id>"])`
  (requires a model trained via train_anomaly_model — ask for the model name)
- **docx**: `generate_diagnostic_report_docx(signal_id="<id>", sections={"Summary": "..."})`
- **full**: generate FFT + envelope + ISO + DOCX reports in sequence

For ISO reports the signal must have a DECLARED unit — if the assessment is
refused, confirm the unit with the user and re-load with
`load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`.

After generation, report the file paths and key findings from each report.
Report filenames are timestamped; `list_html_reports()` lists everything
generated so far.
