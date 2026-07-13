---
description: Run a quick vibration health screening on a signal
argument-hint: "[signal_id or file_path]"
allowed-tools:
  - Read
  - Bash
---

Run a quick vibration health screening on the specified signal.

If the user provided a signal_id or file path as argument, use it directly.
Otherwise, call `list_signals(scope="memory")` to show loaded signals and ask
the user to choose one.

Follow the quick-screening skill workflow:
1. Load/verify the signal — `load_signal(filepath="<file>", signal_unit="g")`
   if not loaded (declare the unit only if known)
2. Statistical features — `analyze_statistics(signal_id="<id>")` (RMS, Crest
   Factor, Kurtosis)
3. FFT snapshot — `analyze_fft(signal_id="<id>")` (top spectral peaks)
4. ISO 20816-3 severity — `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`
   (if refused for an undeclared unit, confirm the unit with the user and
   re-load with `load_signal(filepath="<file>", signal_unit="mm/s", overwrite=True)`)
5. Produce a screening card with overall health status

Decision logic:
- ISO Zone A + Kurtosis < 1 -> Healthy
- ISO Zone B or Kurtosis 1-3 -> Monitor
- ISO Zone C or Kurtosis 3-6 -> Suspicious
- ISO Zone D or Kurtosis > 6 -> Critical

Keep the output concise and actionable — this is a screening, not a
diagnosis; recommend targeted analysis for Suspicious/Critical results. It
should complete in under 30 seconds.
