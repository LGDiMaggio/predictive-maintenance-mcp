---
description: Start a bearing fault diagnosis workflow on a vibration signal
argument-hint: "[signal_id or file_path]"
allowed-tools:
  - Read
  - Bash
---

Run the bearing fault diagnosis workflow on the specified signal.

If the user provided a signal_id or file_path as argument, use it directly.
Otherwise, call `list_stored_signals()` to show available signals and ask the
user to choose one.

Follow the bearing-diagnosis skill workflow:
1. Load/verify the signal
2. Statistical screening (features extraction)
3. FFT analysis
4. Bearing characteristic frequency calculation (ask for bearing designation if
   unknown)
5. Envelope analysis with appropriate filter band
6. Evidence matching against fault frequencies
7. ISO 20816-3 severity assessment
8. Generate reports (envelope + FFT + optionally DOCX)

Present a concise diagnosis summary at the end with confidence level and
recommended actions.
