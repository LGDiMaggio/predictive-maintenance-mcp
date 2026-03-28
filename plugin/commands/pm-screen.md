---
description: Run a quick vibration health screening on a signal
argument-hint: "[signal_id or file_path]"
allowed-tools:
  - Read
  - Bash
---

Run a quick vibration health screening on the specified signal.

If the user provided a signal_id or file_path as argument, use it directly.
Otherwise, call `list_stored_signals()` to show available signals and ask the
user to choose one.

Follow the quick-screening skill workflow:
1. Load/verify the signal
2. Extract statistical features (RMS, Crest Factor, Kurtosis)
3. FFT snapshot (top spectral peaks)
4. ISO 20816-3 severity assessment
5. Produce a screening card with overall health status

Decision logic:
- ISO Zone A + Kurtosis < 1 -> Healthy
- ISO Zone B or Kurtosis 1-3 -> Monitor
- ISO Zone C or Kurtosis 3-6 -> Suspicious
- ISO Zone D or Kurtosis > 6 -> Critical

Keep the output concise and actionable. This should complete in under 30 seconds.
