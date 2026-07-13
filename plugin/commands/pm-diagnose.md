---
description: Start a bearing fault diagnosis workflow on a vibration signal
argument-hint: "[signal_id or file_path]"
allowed-tools:
  - Read
  - Bash
---

Run the bearing fault diagnosis workflow on the specified signal.

If the user provided a signal_id or file path as argument, use it directly.
Otherwise, call `list_signals(scope="memory")` to show loaded signals (and
`list_signals(scope="disk")` for loadable files) and ask the user to choose.

Follow the bearing-diagnosis skill workflow:
1. Load/verify the signal — `load_signal(filepath="<file>", signal_unit="g")`
   (declare the unit only if known; ask, never guess), then
   `get_signal_info(signal_id="<id>")`
2. Statistical screening — `analyze_statistics(signal_id="<id>")`
3. FFT analysis — `analyze_fft(signal_id="<id>")`
4. Bearing characteristic frequencies — ask for the bearing designation if
   unknown, then `search_bearing_catalog(bearing_id="6205")` or
   `calculate_bearing_characteristic_frequencies(num_balls=9, ball_diameter_mm=7.94, pitch_diameter_mm=39.04, rpm=1797)`
5. Envelope analysis — `analyze_envelope(signal_id="<id>", filter_low=500, filter_high=5000)`
6. Evidence matching — `check_bearing_faults(signal_id="<id>", rpm=1797, bearing_id="6205")`
7. ISO 20816-3 severity — `assess_severity(signal_id="<id>", machine_group=2, support_type="rigid")`
8. Reports — `generate_envelope_report(signal_id="<id>")` +
   `generate_fft_report(signal_id="<id>")`, optionally
   `generate_diagnostic_report_docx(signal_id="<id>", sections={"Summary": "..."})`

Present a concise diagnosis summary at the end with evidence strength and
recommended actions. Use cautious language — the result supports the
engineer's decision, it does not replace it.
