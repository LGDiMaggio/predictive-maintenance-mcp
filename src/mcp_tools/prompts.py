"""MCP diagnostic prompts for guided analysis workflows.

Every tool call written in these templates must be executable against the
registered inventory (valid tool name + valid kwargs) — a CI guard (U10)
validates them. Signals are referenced by signal_id: the templates start
by loading the signal if it is not already in the repository.
"""

from typing import Optional

from mcp.server.fastmcp import FastMCP



def diagnose_bearing(
    signal_id: str,
    sampling_rate: Optional[float] = None,
    machine_group: int = 2,  # Default 2 (medium) - most common
    support_type: str = "rigid",  # Default rigid - most common
    rpm: Optional[float] = None,
    bpfo: Optional[float] = None,
    bpfi: Optional[float] = None,
    bsf: Optional[float] = None,
    ftf: Optional[float] = None
) -> str:
    """
        Guided workflow for bearing diagnostics with ISO 20816-3 compliance.

        Evidence-based policy:
        - Envelope peaks at characteristic frequencies are PRIMARY indicators (strong evidence)
        - Statistical indicators (CF>6, Kurtosis>6) are SECONDARY/confirmatory
        - If envelope shows clear peaks at BPFO/BPFI/BSF/FTF (±5% tolerance) → bearing fault is STRONGLY indicated
        - Additional high CF or Kurtosis reinforces the diagnosis but is not strictly required if envelope evidence is clear

        **ISO 20816-3 Defaults** (use if user doesn't specify):
        - machine_group = 2 (medium-sized machines, 15-300 kW, most common)
        - support_type = "rigid" (horizontal machines on foundations)

        Args:
            signal_id: ID of the stored signal (or the file to load in STEP 0)
            sampling_rate: Sampling frequency in Hz (if None, will check metadata or ask user)
            machine_group: ISO machine group (1=large >300kW, 2=medium 15-300kW) (default: 2)
            support_type: 'rigid' or 'flexible' (default: 'rigid' for horizontal machines)
            rpm: Operating speed in RPM (required for interpreting results)
            bpfo: Ball Pass Frequency Outer race (Hz) - if known
            bpfi: Ball Pass Frequency Inner race (Hz) - if known
            bsf: Ball Spin Frequency (Hz) - if known
            ftf: Fundamental Train Frequency (Hz) - if known
        """
    # Build frequency reference string
    freq_refs = []
    if bpfo: freq_refs.append(f"BPFO={bpfo:.2f} Hz")
    if bpfi: freq_refs.append(f"BPFI={bpfi:.2f} Hz")
    if bsf: freq_refs.append(f"BSF={bsf:.2f} Hz")
    if ftf: freq_refs.append(f"FTF={ftf:.2f} Hz")
    freq_info = ", ".join(freq_refs) if freq_refs else "NOT PROVIDED - must request from user"

    bearing_freqs_dict = (
        "{"
        + ", ".join(
            f'"{name}": {val}'
            for name, val in (
                ("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf), ("FTF", ftf)
            )
            if val
        )
        + "}"
    )
    freqs_placeholder = (
        '{"BPFO": <hz>, "BPFI": <hz>, "BSF": <hz>, "FTF": <hz>}'
    )

    rpm_kwarg = f", rpm={rpm}" if rpm else ""
    fs_info = f"{sampling_rate}" if sampling_rate else "UNKNOWN"
    fs_kwarg = f", sampling_rate={sampling_rate}" if sampling_rate else ""

    return f"""Perform evidence-based bearing diagnostic on signal_id "{signal_id}":

    ⚠️  CRITICAL INFERENCE POLICY ⚠️
    ═══════════════════════════════════════════════════════════════════════════════
    **NEVER INFER FAULT TYPE OR CONDITION FROM A SIGNAL ID OR FILENAME**

    - The id "{signal_id}" is an OPAQUE IDENTIFIER ONLY
    - "OuterRaceFault" in an id ≠ outer race fault exists
    - "baseline" in an id ≠ healthy signal

    **BASE DIAGNOSIS EXCLUSIVELY ON:**
    1. Envelope spectrum peaks matching BPFO/BPFI/BSF/FTF (±5% tolerance)
    2. Statistical indicators (CF, Kurtosis) as SECONDARY confirmation
    3. ISO 20816-3 zone measurement

    **IF THE ID CONTRADICTS ANALYSIS:**
    Report: "Despite the id suggesting [X], analysis shows [Y]"

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 0 — SIGNAL RESOLUTION & MANDATORY PARAMETER CHECK
    ═══════════════════════════════════════════════════════════════════════════════

    1. Verify the signal is loaded:
       Call: list_signals(scope="memory")
       If "{signal_id}" is not among the loaded ids, find its file with
       list_signals(scope="disk") and load it:
       Call: load_signal(filepath="<file>"{fs_kwarg})
       If the file is not found or multiple matches exist, ASK USER to clarify.
       Do NOT guess or auto-correct names.

    2. Required parameters:
       ✓ Signal: {signal_id}
       {'✓' if sampling_rate else '✗'} Sampling rate: {fs_info} Hz
       {'✓' if rpm else '✗'} Operating speed: {rpm or 'NOT PROVIDED'} RPM
       {'✓' if freq_refs else '✗'} Bearing characteristic frequencies: {freq_info}

       CRITICAL RULE: The sampling rate comes from the stored signal metadata
       (get_signal_info(signal_id="{signal_id}") shows it). If it is missing
       OR if bearing frequencies (BPFO/BPFI/BSF/FTF) are NOT PROVIDED and no
       bearing designation is known:
       → STOP and ASK USER for these parameters before proceeding.
       → Explain: "Cannot perform bearing diagnosis without [missing parameters]."

       Do NOT use placeholder/default values. Do NOT proceed with incomplete data.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 1 — ISO 20816-3 (Severity Context)
    ═══════════════════════════════════════════════════════════════════════════════

    BEFORE calling assess_severity, ASK USER to confirm machine parameters:

    "For ISO 20816-3 evaluation, I need to know:
    1. Machine group:
       - Group 1 (large): >300 kW or shaft height ≥ 315 mm
       - Group 2 (medium): 15-300 kW or shaft height 160-315 mm

    2. Support type:
       - Rigid: Foundation natural freq > 1.25× operating freq
       - Flexible: All other cases (typical for large machines)

    Based on your description, I'll assume:
    - Machine group: {machine_group} (default for typical industrial equipment)
    - Support type: {support_type} (most common)

    Is this correct, or should I use different values?"

    If user confirms or provides values, proceed with:
    Call: assess_severity(signal_id="{signal_id}", machine_group={machine_group}, support_type="{support_type}"{rpm_kwarg})
    Report: RMS velocity and ISO zone (A/B/C/D) in 1-2 sentences.
    Note: This provides overall severity but is NOT bearing-specific. Use for maintenance urgency only.
    Note: The verdict requires a DECLARED signal unit — if it is refused,
    re-load with load_signal(filepath="<file>", signal_unit="g", overwrite=True).

    Optional visualization:
    Call: generate_iso_report(signal_id="{signal_id}", machine_group={machine_group}, support_type="{support_type}"{rpm_kwarg})
    This saves an interactive HTML report (color-coded zone chart with the
    measured RMS marker) to the reports/ directory. Tell user to open the
    returned file path in their browser.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 2 — Statistical Screening
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_statistics(signal_id="{signal_id}")
    Report: RMS, Crest Factor, Kurtosis (excess), Skewness in bullet points.

    Interpretation flags (SECONDARY indicators):
    • CF > 6 or Kurtosis > 6 → Strong impulsiveness (supports bearing fault hypothesis)
    • CF 4-6 or Kurtosis 3-6 → Moderate impulsiveness (weak support)
    • CF < 4 and Kurtosis < 3 → Low impulsiveness (but envelope may still show faults)

    ⚠️ Do NOT diagnose from statistics alone. Proceed to envelope analysis.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 3 — FFT Spectrum (Contextual)
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_fft(signal_id="{signal_id}")
    Report dominant peaks in bullet points (top 5 only). Look for:
    • Shaft speed (1× RPM = {rpm/60 if rpm else '?'} Hz) and harmonics
    • Any elevated broadband noise

    Optional visualization:
    Call: generate_fft_report(signal_id="{signal_id}", max_freq=5000, num_peaks=15{rpm_kwarg})
    This saves an interactive HTML FFT report (dB spectrum, automatic peak
    detection, harmonic markers) to the reports/ directory. Tell user to
    open the returned file path in their browser.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 4 — ENVELOPE ANALYSIS (PRIMARY DIAGNOSTIC EVIDENCE)
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_envelope(signal_id="{signal_id}", filter_low=500, filter_high=5000, num_peaks=10)

    Expected frequencies (±5% tolerance):
    {chr(10).join(f'  • {ref}' for ref in freq_refs) if freq_refs else '  (User must provide BPFO, BPFI, BSF, FTF)'}

    Examine envelope spectrum peaks:
    1. Check if ANY peak falls within ±5% of expected frequencies
    2. Check for harmonics: 2×BPFO, 3×BPFO, 2×BPFI, etc.
    3. List top 5-10 peaks with frequencies and magnitudes

    Systematic check (when rpm is known):
    Call: check_bearing_faults(signal_id="{signal_id}", rpm={rpm if rpm else '<rpm>'}, frequencies={bearing_freqs_dict if freq_refs else freqs_placeholder})
    (Or pass bearing_id="<designation>" for a catalog bearing instead of frequencies.)

    Optional visualization:
    Call: generate_envelope_report(signal_id="{signal_id}", filter_low=500, filter_high=5000, max_freq=500, bearing_freqs={bearing_freqs_dict if freq_refs else freqs_placeholder})
    This saves an interactive HTML envelope report (filtered signal +
    envelope spectrum with bearing frequency markers) to the reports/
    directory. Tell user to open the returned file path in their browser.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 5 — DIAGNOSTIC DECISION (EVIDENCE-BASED)
    ═══════════════════════════════════════════════════════════════════════════════

    Decision tree:

    A) IF envelope spectrum shows clear peak(s) at characteristic frequency (±5%):
       → Bearing fault type is STRONGLY INDICATED

       Classification by frequency:
       • Peak at BPFO (±5%) → **Outer race fault** (canonical: outer_race)
       • Peak at BPFI (±5%) → **Inner race fault** (canonical: inner_race)
       • Peak at BSF (±5%) → **Rolling element (ball) fault** (canonical: ball)
       • Peak at FTF (±5%) → **Cage fault** (canonical: cage)

       Evidence strength:
       - High evidence: Peak + harmonics present AND (CF>6 OR Kurtosis>6)
       - Moderate evidence: Peak present but weaker harmonics OR moderate stats (CF 4-6, Kurt 3-6)
       - Note: Even without extreme statistics, clear envelope peaks ARE diagnostic

    B) IF envelope shows ambiguous/borderline peaks:
       → "Possible [fault type] - envelope peak near [frequency] but [state issue: weak magnitude, no harmonics, etc.]"
       → Recommend: retake measurement, higher resolution, trending

    C) IF no envelope peaks at characteristic frequencies:
       → "No clear bearing fault signatures detected"
       → IF stats are elevated: "High impulsiveness without bearing-specific frequencies suggests [other cause: impacts, looseness, etc.]"
       → IF ISO zone C/D: "Elevated vibration without bearing signatures - check alignment, balance, structural issues"

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 6 — RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════════════════════

    Based on diagnosis + ISO zone (ONLY use these recommendations - DO NOT invent others):
    • Confirmed fault + Zone C/D → Immediate action: inspect bearing, plan replacement
    • Confirmed fault + Zone B → Short-term action: schedule maintenance within 1-3 months, increase monitoring
    • Confirmed fault + Zone A → Monitor closely: retest in 1-2 weeks, track progression
    • No fault + Zone C/D → Investigate other causes: alignment, balance, looseness, foundation
    • No fault + Zone A/B → Continue routine monitoring

    Optionally formalize:
    Call: generate_maintenance_recommendations(severity_zone="A", fault_types=["outer_race"])
    (Use the actual zone letter; fault_types uses the canonical vocabulary:
    outer_race, inner_race, ball, cage, misalignment, unbalance, looseness —
    NOT the BPFO/BPFI acronyms.)

    CRITICAL: Do NOT suggest specific parameter values (e.g., filter frequencies, acquisition settings)
    unless they appear in tool outputs. Do NOT invent troubleshooting steps beyond those listed above.

    Always cite:
    - Which envelope peaks were found (frequency, magnitude, harmonics)
    - Statistical values (CF, Kurtosis) and how they support/contradict
    - ISO zone and severity
    - Specific tool outputs used

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT FORMATTING (CRITICAL)
    ═══════════════════════════════════════════════════════════════════════════════

    Keep output CONCISE (≤300 words total):
    • Use bullet points for all findings
    • Provide brief summary first (2-3 sentences)
    • Use generate_fft_report / generate_envelope_report / generate_iso_report
      to create HTML reports (saved to the reports/ directory)
    • Tell user to open the HTML file path in browser for interactive visualizations
    • If user needs more details, offer "Show detailed analysis?" continuation
    • NEVER print large JSON/CSV data directly in text output
    • Frame every conclusion as decision support for a qualified engineer:
      this analysis AUGMENTS expert judgment — the maintenance decision
      rests with the engineer, not with this workflow
    """


def diagnose_gear(
    signal_id: str,
    sampling_rate: Optional[float] = None,
    num_teeth: Optional[int] = None,
    rpm: Optional[float] = None
) -> str:
    """
        Evidence-based guided workflow for gear diagnostics with strict anti-speculation rules.

        Args:
            signal_id: ID of the stored signal (or the file to load in STEP 0)
            sampling_rate: Sampling frequency in Hz (if None, will check metadata or ask user)
            num_teeth: Number of gear teeth (REQUIRED for GMF calculation)
            rpm: Shaft rotation speed in RPM (REQUIRED for GMF and sideband identification)
        """
    fs_info = f"{sampling_rate}" if sampling_rate else "UNKNOWN"
    fs_kwarg = f", sampling_rate={sampling_rate}" if sampling_rate else ""
    teeth_info = f"{num_teeth}" if num_teeth else "NOT PROVIDED"
    rpm_info = f"{rpm}" if rpm else "NOT PROVIDED"
    gmf_value = (
        f"{rpm/60 * num_teeth:.2f}" if (rpm and num_teeth) else "<gmf_hz>"
    )

    return f"""Perform an evidence-based gear diagnostic on signal_id "{signal_id}":

    ⚠️  CRITICAL INFERENCE POLICY ⚠️
    ═══════════════════════════════════════════════════════════════════════════════
    **NEVER INFER FAULT TYPE OR CONDITION FROM A SIGNAL ID OR FILENAME**

    - The id "{signal_id}" is an OPAQUE IDENTIFIER ONLY
    - "GearFault" in an id ≠ gear fault exists
    - "baseline" in an id ≠ healthy signal

    **BASE DIAGNOSIS EXCLUSIVELY ON:**
    1. FFT spectrum showing GMF harmonics
    2. Sidebands spaced by shaft rotation frequency (f_rot)
    3. Statistical indicators (Kurtosis) as SECONDARY confirmation

    **IF THE ID CONTRADICTS ANALYSIS:**
    Report: "Despite the id suggesting [X], analysis shows [Y]"

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 0 — SIGNAL RESOLUTION & MANDATORY PARAMETER CHECK
    ═══════════════════════════════════════════════════════════════════════════════

    1. Verify the signal is loaded:
       Call: list_signals(scope="memory")
       If "{signal_id}" is not among the loaded ids, find its file with
       list_signals(scope="disk") and load it:
       Call: load_signal(filepath="<file>"{fs_kwarg})
       If the file is not found or multiple matches exist, ASK USER to clarify.

    2. Required parameters:
       ✓ Signal: {signal_id}
       {'✓' if sampling_rate else '✗'} Sampling rate: {fs_info} Hz
       {'✓' if num_teeth else '✗'} Number of teeth (Z): {teeth_info}
       {'✓' if rpm else '✗'} Rotation speed: {rpm_info} RPM

       CRITICAL RULE: The sampling rate comes from the stored signal metadata.
       If it is missing OR if num_teeth OR rpm are NOT PROVIDED:
       → STOP and ASK USER for these parameters before proceeding.
       → Explain: "Cannot perform gear diagnosis without [missing parameters]."

       Do NOT use placeholder/default values. Do NOT proceed with incomplete data.

    GUARDRAILS (apply throughout):
    - Do NOT infer faults from signal ids, paths, or labels.
    - A gear tooth fault (localized damage) requires BOTH:
      • Clear Gear Mesh Frequency (GMF) harmonics AND
      • Sidebands spaced by shaft rotation frequency (f_rot) around GMF or its harmonics
      • (Optional but reinforcing) Elevated Kurtosis (>3) or modulation energy
    - Without sidebands: DO NOT claim tooth damage; consider uniform wear only if GMF elevated but stable statistics.

    STEP 1 — INPUT & CONTEXT
    Once all parameters confirmed:
    - f_rot = rpm / 60 = {f"{rpm/60:.2f}" if rpm else "?"} Hz
    - Theoretical GMF = f_rot × Z = {f"{rpm/60 * num_teeth:.2f}" if (rpm and num_teeth) else "?"} Hz

    STEP 2 — STATISTICS (screening only)
    Call: analyze_statistics(signal_id="{signal_id}")
    Report RMS, Crest Factor, Kurtosis in bullet points (brief).
    Indicators:
    - Elevated RMS: possible general load / imbalance
    - High Kurtosis (>3): impulsive impacts (may correlate with chipped tooth)
    - High Crest Factor (>4): impulsiveness
    (Do NOT diagnose from stats alone.)

    STEP 3 — SPECTRUM (frequency evidence)
    Call: analyze_fft(signal_id="{signal_id}")
    Extract dominant peaks up to, e.g., 5× expected GMF. Identify:
    - GMF and its harmonics: GMF, 2×GMF, 3×GMF
    - Sidebands: GMF ± n·f_rot (n=1..3). Log their presence, spacing consistency, and relative amplitudes.
    Report top 5 peaks only (brief).

    Systematic check (when rpm and GMF are known):
    Call: check_bearing_faults(signal_id="{signal_id}", rpm={rpm if rpm else '<rpm>'}, frequencies={{"GMF": {gmf_value}}})

    Optional visualization:
    Call: generate_fft_report(signal_id="{signal_id}", max_freq=5000, num_peaks=15)
    This saves an interactive HTML FFT report to the reports/ directory.
    Tell user to open the returned file path in their browser.

    STEP 4 — OPTIONAL ENVELOPE (if strong modulation or impacts)
    If stats suggest impulsiveness OR sideband pattern partial:
    Call: analyze_envelope(signal_id="{signal_id}", filter_low=500, filter_high=5000)
    to inspect modulation signature. (Not mandatory if FFT already conclusive.)

    STEP 5 — CLASSIFICATION (apply confirmation rule)
    Decision categories (choose exactly one):
    - "Gear tooth fault CONFIRMED" → Requires: (GMF harmonics present) AND (≥1 clear sideband pair with spacing ≈ f_rot) AND (supporting stat: Kurtosis>3 or CF>4)
    - "Possible localized tooth damage" → Partial sidebands OR ambiguous spacing; list missing evidence required for confirmation.
    - "Uniform wear / increased load" → Elevated GMF amplitude WITHOUT sidebands, normal/low impulsiveness.

    Each conclusion MUST cite: tools used (statistics, FFT, envelope), specific numeric peaks (frequencies & magnitudes), sideband spacing vs expected f_rot (difference in Hz), and any supporting statistical thresholds.

    STEP 6 — RECOMMENDATIONS (brief bullet points)
    Provide actionable items aligned with category:
    - Confirmed fault: plan inspection, tooth visual check, lubrication review, short-term monitoring interval suggestion.
    - Possible fault: higher-resolution spectrum, trend GMF amplitude.
    - Uniform wear: continue monitoring; schedule routine inspection.

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT FORMATTING (CRITICAL)
    ═══════════════════════════════════════════════════════════════════════════════

    Keep output CONCISE (≤300 words total):
    • Use bullet points for all findings
    • Provide brief summary first (2-3 sentences)
    • Use generate_fft_report to create HTML reports (saved to the reports/
      directory)
    • Tell user to open the HTML file path in browser for interactive visualizations
    • If user needs more details, offer "Show detailed analysis?" continuation
    • NEVER print large JSON/CSV data directly in text output
    • Frame every conclusion as decision support for a qualified engineer:
      this analysis AUGMENTS expert judgment — the maintenance decision
      rests with the engineer, not with this workflow
    """


def quick_diagnostic_report(signal_id: str) -> str:
    """
        Quick, evidence-aware screening report (non-definitive).

        Args:
            signal_id: ID of the stored signal (or the file to load in STEP 0)
        """
    return f"""Generate a concise screening report for signal_id "{signal_id}" using only observable evidence:

    STEP 0 — SIGNAL RESOLUTION
    Call: list_signals(scope="memory")
    If "{signal_id}" is not among the loaded ids, find its file with
    list_signals(scope="disk") and load it:
    Call: load_signal(filepath="<file>")
    If not found or multiple matches, ASK USER to clarify.

    Guardrails:
    - Ignore signal ids/filenames as diagnostic evidence.
    - Do NOT diagnose faults from statistics alone; use them for screening only.
    - Use cautious language: "possible/consistent with" unless corroborated by multiple indicators.

    1) Load & sanity checks
    Call: get_signal_info(signal_id="{signal_id}")
    - Report number of samples, duration (s), sampling rate (brief, 1 line).

    2) Statistics (screening)
    Call: analyze_statistics(signal_id="{signal_id}")
    Report: RMS, Crest Factor, Kurtosis, Skewness (bullet points only).
    Flags (screening thresholds, not definitive):
    - CF > 4 → impulsiveness present; CF > 6 → strong impulsiveness
    - Kurtosis > 0 → non-Gaussian / impulsive content (excess kurtosis); > 3 → significant; > 6 → severe
    Note: These flags alone are insufficient for fault identification.

    3) Spectral snapshot
    Call: analyze_fft(signal_id="{signal_id}")
    - Report peak frequency, magnitude (top 3 peaks only).
    - If operating speed is known, relate peaks to 1×/2× RPM; otherwise, request it for deeper interpretation.

    4) Next-step guidance (evidence-first)
    - If strong impulsiveness (CF>6 or Kurtosis>6), suggest: "Use diagnose_bearing prompt for targeted bearing analysis"
    - If tonal/harmonic pattern dominates, suggest: "Use diagnose_gear prompt if gear suspected"
    - If broadband increase, suggest: ISO 20816-3 check with assess_severity(signal_id="{signal_id}")

    Output format (≤200 words):
    - Screening summary with measured values (bullet points)
    - No definitive fault labels
    - List recommended targeted analyses and required missing parameters
    - Frame the result as screening input for a qualified engineer: it
      AUGMENTS expert judgment, it never replaces it
    """


def register(mcp: FastMCP) -> None:
    """Register diagnostic prompts on the MCP server."""
    mcp.prompt()(diagnose_bearing)
    mcp.prompt()(diagnose_gear)
    mcp.prompt()(quick_diagnostic_report)
