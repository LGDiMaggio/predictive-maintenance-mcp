---
description: Generate a diagnostic report for a vibration signal
argument-hint: "[signal_id] [report_type: fft|envelope|iso|comparison|pca|docx|full|pdf]"
---

Run the report-generation skill of this plugin
(`predictive-maintenance:report-generation`) for the signal and the report type
the user named.

The first argument is a signal_id or a file path; the second, optional, is the
report type, and a missing report type means a full report. Without a signal,
the skill's own signal check applies. Follow that skill's workflow, relay rules
and reporting exactly. This command adds no step, threshold, verdict or
recommendation of its own.
