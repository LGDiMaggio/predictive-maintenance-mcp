"""The closed list of non-bearing statements of a single-acquisition diagnosis.

Two functions interpret one acquisition today: the synthesis behind
``diagnose_vibration`` (:mod:`.diagnosis_pipeline`) and the advisory behind
``generate_diagnostic_report`` (:mod:`.advisory`). Every sentence and action
they write that the bearing claim does not own comes from ``STATEMENTS``, by
key; the bearing sentences come from ``CLAIM_SENTENCES`` in
:mod:`..diagnostics.bearing_claims`, and the reasons and remedies of a stage
that could not run are written by the stage that refused.

Keys start with the function that writes the statement: ``diagnosis_`` for the
synthesis behind ``diagnose_vibration`` and ``report_`` for the advisory. Where
the two say the same thing in different words, both wordings are kept, each
under its own key, until a reviewed wording change brings them together.

``INTERPRETATION_POLICY_VERSION`` identifies this list together with the
evidence-strength rule of :mod:`.advisory`. It changes whenever a statement, a
table entry or a composition rule changes. It is internal: no tool output,
report or rendering carries it.
"""

INTERPRETATION_POLICY_VERSION = 1

STATEMENTS: dict[str, str] = {
    # diagnose_vibration: overall_diagnosis lines
    "diagnosis_iso_refused": "ISO Severity: not assessed — {reason}",
    "diagnosis_iso_assessed": (
        "ISO Severity: Zone {zone} ({severity}) — RMS velocity "
        "{rms_velocity:.2f} mm/s."
    ),
    "diagnosis_iso_assessed_no_severity": (
        "ISO Severity: Zone {zone} — RMS velocity {rms_velocity:.2f} mm/s."
    ),
    "diagnosis_dominant_frequency": (
        "Dominant frequency: {peak_hz:.1f} Hz (shaft: {shaft_hz:.1f} Hz)."
    ),
    # The measured-line sentence for a strongest line near 1x or 2x shaft
    # speed: the window, what the line cannot separate (normal running first),
    # what would narrow it down, and why a second reading is no corroboration.
    "diagnosis_shaft_order_line": (
        "The strongest line of the spectrum is at {ratio:.2f} times the "
        "declared shaft frequency ({window}). Its dominance does not identify "
        "a fault, and it is not counted as a finding. This check cannot "
        "distinguish {readings}. {discrimination} {repeat}"
    ),
    "diagnosis_shaft_order_window_1x": "within 10 % of 1x",
    "diagnosis_shaft_order_window_2x": "within 5 % of 2x",
    "diagnosis_shaft_order_readings_1x": (
        "the 1x response of a machine in normal condition from unbalance, a "
        "bent or bowed rotor, eccentricity, misalignment, mechanical looseness "
        "or a soft foot, a worn or damaged coupling, a resonance near running "
        "speed, vibration transmitted from another machine, or a component at "
        "the electrical supply frequency, which lies close to 1x on a two-pole "
        "motor"
    ),
    "diagnosis_shaft_order_readings_2x": (
        "the 2x response of a machine in normal condition from misalignment, "
        "mechanical looseness, a rub, a shaft crack, a shaft or rotor with "
        "asymmetric stiffness, a resonance near twice running speed, the "
        "passing tone of a rotor with two blades or lobes, vibration "
        "transmitted from another machine, a component at the electrical "
        "supply frequency, which lies close to 2x on a four-pole motor, or, on "
        "a two-pole motor, the electromagnetic vibration at twice the supply "
        "frequency, which lies close to 2x (exactly at 2x on a synchronous "
        "motor)"
    ),
    "diagnosis_shaft_order_discrimination": (
        "Narrowing these down needs more than this check, such as a finer "
        "analysis of the line's frequency, a baseline from the same machine at "
        "the same condition, phase, a second radial or an axial channel, or "
        "several speeds. Establishing a cause needs a physical check of the "
        "machine."
    ),
    "diagnosis_shaft_order_repeat": (
        "Another analysis that reads this same line of the same signal repeats "
        "the observation and does not corroborate it."
    ),
    "diagnosis_bearing_not_computed": (
        "Bearing analysis: not computed. {reason} No bearing check ran, so "
        "this diagnosis says nothing about the condition of the bearing."
    ),
    "diagnosis_anomaly": (
        "Anomaly detection: {health} ({ratio_pct:.1f}% anomalous segments)."
    ),
    # diagnose_vibration: recommendations
    "diagnosis_action_zone_c": "Schedule maintenance within next planned shutdown.",
    "diagnosis_action_zone_d": "IMMEDIATE ACTION: Stop machine and inspect.",
    "diagnosis_action_anomaly_faulty": (
        "Anomaly model flagged signal as faulty — investigate root cause."
    ),
    "diagnosis_action_anomaly_suspicious": (
        "Anomaly model detected suspicious patterns — increase monitoring frequency."
    ),
    "diagnosis_action_default": "Continue routine monitoring per maintenance schedule.",
    # generate_diagnostic_report: ISO severity block
    "report_iso_refused": "ISO severity was not assessed. {reason}",
    "report_iso_assessed": (
        "ISO 20816-3: RMS velocity {rms_velocity:.2f} mm/s places this machine "
        "in Zone {zone} ({severity}) for machine group {machine_group} on a "
        "{support_type} support. {description}"
    ),
    # generate_diagnostic_report: bearing line-test block
    "report_bearing_not_computed": (
        "Bearing characteristic-frequency matching was not computed. {reason} "
        "No bearing check ran, so this report says nothing about the condition "
        "of the bearing."
    ),
    "report_bearing_absent": (
        "Bearing characteristic-frequency matching was not attempted: no "
        "bearing designation was supplied for this signal, so BPFO, BPFI, BSF "
        "and FTF could not be computed. Without it, a spectral peak cannot be "
        "attributed to a specific bearing element."
    ),
    "report_bearing_absent_remedy": (
        "Re-run generate_diagnostic_report with bearing_id set to the "
        "bearing's designation (search_bearing_catalog lists the designations "
        "the bearing catalog holds)."
    ),
    # generate_diagnostic_report: anomaly block
    "report_anomaly_absent": (
        "No anomaly-model verdict is available for this signal: no trained "
        "model was found. The pattern-level check that would corroborate or "
        "contradict the severity reading is therefore missing from this "
        "assessment."
    ),
    "report_anomaly_absent_remedy": (
        "Train an anomaly model on healthy signals from this machine, then "
        "re-run the diagnosis."
    ),
    "report_anomaly": (
        "Anomaly model verdict: {health} — {counted} ({ratio_pct:.1f}%) fall "
        "outside the learned healthy pattern."
    ),
    "report_anomaly_segments_counted": "{count} of {segments} segments",
    "report_anomaly_segments_share": "{ratio_pct:.1f}% of segments",
    # generate_diagnostic_report: spectral energy block. Which band holds the
    # most energy is a measurement and names no source.
    "report_energy_absent": (
        "No spectral energy distribution is available: the STFT band breakdown "
        "was not computed, so the frequency region carrying the signal's "
        "energy cannot be named."
    ),
    "report_energy": (
        "The largest share of STFT energy is in the {band} band ({share:.0f}% "
        "of the total). This percentage describes the measured signal and does "
        "not identify a fault. The relative distribution across frequency "
        "bands can change with the measured physical quantity (for example, "
        "acceleration versus velocity)."
    ),
    # generate_diagnostic_report: verdict when no bearing check ran
    "report_verdict_not_computed": (
        "No bearing fault verdict was reached, because characteristic-frequency "
        "matching was not computed."
    ),
    "report_verdict_not_computed_for_bearing": (
        "No bearing fault verdict was reached, because characteristic-frequency "
        "matching was not computed for bearing {bearing_id!r}."
    ),
    "report_verdict_not_attempted": (
        "No bearing fault verdict was reached, because characteristic-frequency "
        "matching was not attempted."
    ),
    "report_verdict_zone_elevated": (
        " Broadband vibration is nonetheless elevated (Zone {zone}){tail}"
    ),
    "report_verdict_anomaly_flag": (
        " The anomaly model nonetheless flags this signal as departing from "
        "the learned healthy pattern{tail}"
    ),
    "report_verdict_tail_unchecked": (
        ", and the bearing was not checked, so it cannot be excluded as the source."
    ),
    "report_verdict_tail_non_bearing": (
        ", so a non-bearing source should be considered."
    ),
    "report_verdict_tail_outside_bearing": (
        ", so a source outside the bearing's characteristic frequencies should "
        "be considered."
    ),
    # generate_diagnostic_report: evidence
    "report_dominant_frequency": "Dominant frequency in the raw spectrum: {peak_hz:.1f} Hz.",
    # generate_diagnostic_report: indicator disagreements
    "report_indicator_single_spectrum": "single-spectrum envelope evidence",
    "report_indicator_anomaly_model": "anomaly model",
    "report_indicator_zone": "ISO zone {zone}",
    # generate_diagnostic_report: recommendations
    "report_description_iso_refused": (
        "The severity verdict is unavailable until this is resolved."
    ),
    "report_description_bearing_refused": (
        "The bearing check is unavailable until this is resolved."
    ),
    "report_description_anomaly_absent": (
        "Pattern-level corroboration is unavailable without a trained model."
    ),
    "report_motivation_zone_severity": "ISO Zone {zone} is classified as {severity}.",
    "report_motivation_zone": "ISO Zone {zone}.",
    # generate_diagnostic_report: baseline comparison
    "report_baseline_absent": (
        "No healthy baseline was supplied for this machine, so the readings "
        "below are absolute rather than relative. Whether this condition is "
        "new, stable, or worsening cannot be determined from a single "
        "acquisition."
    ),
    "report_baseline_absent_remedy": (
        "Re-run the diagnosis with a baseline signal id from the same machine "
        "in a known-good state."
    ),
    "report_baseline_refused": (
        "Baseline comparison was refused: {reason} A delta between "
        "measurements taken under different conditions would look like a "
        "change in machine condition."
    ),
    "report_baseline_refused_remedy": (
        "Supply a baseline acquired from the same measurement point under the "
        "same declared conditions."
    ),
    "report_baseline_unit_mismatch": (
        "the signal declares its unit as '{signal_unit}' while the baseline "
        "declares '{baseline_unit}'."
    ),
    "report_baseline_bearing_mismatch": (
        "the signal was analysed against bearing '{signal_bearing}' and the "
        "baseline against '{baseline_bearing}'."
    ),
    "report_baseline_nothing_compared": (
        "A baseline was supplied ({baseline_id}), but no indicator could be "
        "compared: the two diagnoses share no assessed block. Whether this "
        "condition is new, stable, or worsening cannot be determined."
    ),
    "report_baseline_nothing_compared_remedy": (
        "Ensure both signals declare their unit so the severity and anomaly "
        "blocks are assessed rather than refused."
    ),
    "report_baseline_unchanged": (
        "Compared with baseline {baseline_id}: no measurable change in any "
        "compared indicator. The condition described above is stable, not "
        "developing."
    ),
    "report_baseline_moved": (
        "Compared with baseline {baseline_id}: {moved} of {total} indicators "
        "moved. {headline}"
    ),
    "report_delta_rms_unchanged": (
        "RMS velocity is unchanged against baseline ({now:.2f} mm/s, was "
        "{then:.2f} mm/s)."
    ),
    "report_delta_rms_moved": (
        "RMS velocity is {delta:.2f} mm/s {direction} than baseline ({now:.2f} "
        "mm/s, was {then:.2f} mm/s)."
    ),
    "report_delta_anomaly_unchanged": (
        "The share of anomalous segments is unchanged against baseline "
        "({now:.0f}%, was {then:.0f}%) — a difference of under one percentage "
        "point."
    ),
    "report_delta_anomaly_moved": (
        "The share of anomalous segments is {delta:.0f} percentage points "
        "{direction} than baseline ({now:.0f}%, was {then:.0f}%)."
    ),
}
