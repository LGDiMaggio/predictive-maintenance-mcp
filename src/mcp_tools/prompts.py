"""MCP diagnostic prompts for guided analysis workflows."""

from typing import Optional

from mcp.server.fastmcp import FastMCP


def register(mcp):
    """Register diagnostic prompts on the MCP server."""

    @mcp.prompt()
    def diagnose_bearing(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        machine_group: int = 2,  # CHANGED: Default 2 (medium) - most common
        support_type: str = "rigid",  # Default rigid - most common
        operating_speed_rpm: Optional[float] = None,
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
            signal_file: Name of the signal file to analyze
            sampling_rate: Sampling frequency in Hz (if None, will check metadata or ask user)
            machine_group: ISO machine group (1=large >300kW, 2=medium 15-300kW) (default: 2)
            support_type: 'rigid' or 'flexible' (default: 'rigid' for horizontal machines)
            operating_speed_rpm: Operating speed in RPM (required for interpreting results)
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

        rpm_info = f", {operating_speed_rpm}" if operating_speed_rpm else ""
        fs_info = f"{sampling_rate}" if sampling_rate else "UNKNOWN"

        return f"""Perform evidence-based bearing diagnostic on "{signal_file}":

    ⚠️  CRITICAL INFERENCE POLICY ⚠️
    ═══════════════════════════════════════════════════════════════════════════════
    **NEVER INFER FAULT TYPE OR CONDITION FROM FILENAME**

    - Filename "{signal_file}" is an OPAQUE IDENTIFIER ONLY
    - "OuterRaceFault" in filename ≠ outer race fault exists
    - "baseline" in filename ≠ healthy signal
    - "InnerRaceFault" in filename ≠ inner race fault exists

    **BASE DIAGNOSIS EXCLUSIVELY ON:**
    1. Envelope spectrum peaks matching BPFO/BPFI/BSF/FTF (±5% tolerance)
    2. Statistical indicators (CF, Kurtosis) as SECONDARY confirmation
    3. ISO 20816-3 zone measurement

    **IF FILENAME CONTRADICTS ANALYSIS:**
    Report: "Despite filename suggesting [X], analysis shows [Y]"
    Example: "Despite 'OuterRaceFault' in filename, envelope analysis shows NO peaks at BPFO"

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 0 — FILENAME RESOLUTION & MANDATORY PARAMETER CHECK
    ═══════════════════════════════════════════════════════════════════════════════

    1. Verify signal file existence:
       Call list_available_signals() to get exact filename.
       If "{signal_file}" not found or multiple matches exist, ASK USER to clarify.
       Do NOT guess or auto-correct filenames.

    2. Required parameters:
       ✓ Signal file: {signal_file}
       {'✓' if sampling_rate else '✗'} Sampling rate: {fs_info} Hz
       {'✓' if operating_speed_rpm else '✗'} Operating speed: {operating_speed_rpm or 'NOT PROVIDED'} RPM
       {'✓' if freq_refs else '✗'} Bearing characteristic frequencies: {freq_info}

       CRITICAL RULE: If sampling_rate is UNKNOWN, check signal metadata JSON first.
       If still missing OR if bearing frequencies (BPFO/BPFI/BSF/FTF) are NOT PROVIDED:
       → STOP and ASK USER for these parameters before proceeding.
       → Explain: "Cannot perform bearing diagnosis without [missing parameters]. Please provide: ..."

       Example response when parameters are missing:
       "I cannot proceed with the bearing diagnosis because the following required
       parameters are missing:
       - Bearing characteristic frequencies (BPFO, BPFI, BSF, FTF)
       Please provide these values so I can complete the envelope analysis and
       identify the fault type."

       Do NOT use placeholder/default values. Do NOT proceed with incomplete data.
       Do NOT attempt diagnosis without characteristic frequencies.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 1 — ISO 20816-3 (Severity Context)
    ═══════════════════════════════════════════════════════════════════════════════

    BEFORE calling evaluate_iso_20816, ASK USER to confirm machine parameters:

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
    Call: evaluate_iso_20816("{signal_file}", {fs_info}, {machine_group}, "{support_type}"{rpm_info})
    Report: RMS velocity and ISO zone (A/B/C/D) in 1-2 sentences.
    Note: This provides overall severity but is NOT bearing-specific. Use for maintenance urgency only.

    Optional visualization:
    Call: generate_iso_report("{signal_file}", {machine_group}, "{support_type}"{rpm_info})
    This saves an interactive HTML report to the reports/ directory showing:
    - Color-coded ISO zone chart with marker on measured RMS velocity
    - Time-domain signal plot
    - Detailed severity assessment
    The tool returns the file path. Tell user to open the returned HTML file path in their browser to view the interactive report.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 2 — Statistical Screening
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_statistics("{signal_file}")
    Report: RMS, Crest Factor, Kurtosis (excess), Skewness in bullet points.

    Interpretation flags (SECONDARY indicators):
    • CF > 6 or Kurtosis > 6 → Strong impulsiveness (supports bearing fault hypothesis)
    • CF 4-6 or Kurtosis 3-6 → Moderate impulsiveness (weak support)
    • CF < 4 and Kurtosis < 3 → Low impulsiveness (but envelope may still show faults)

    ⚠️ Do NOT diagnose from statistics alone. Proceed to envelope analysis.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 3 — FFT Spectrum (Contextual)
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_fft("{signal_file}", {fs_info})
    Report dominant peaks in bullet points (top 5 only). Look for:
    • Shaft speed (1× RPM = {operating_speed_rpm/60 if operating_speed_rpm else '?'} Hz) and harmonics
    • Any elevated broadband noise

    Optional visualization:
    Call: generate_fft_report("{signal_file}", max_freq=5000, num_peaks=15)
    This saves an interactive HTML report to the reports/ directory showing:
    - FFT spectrum in dB scale with automatic peak detection
    - Harmonic markers (if rotation frequency provided)
    - Top frequency peaks table
    The tool returns the file path. Tell user to open the returned HTML file path in their browser to view the interactive FFT analysis.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 4 — ENVELOPE ANALYSIS (PRIMARY DIAGNOSTIC EVIDENCE)
    ═══════════════════════════════════════════════════════════════════════════════

    Call: analyze_envelope("{signal_file}", {fs_info}, 500, 5000, num_peaks=10)

    Expected frequencies (±5% tolerance):
    {chr(10).join(f'  • {ref}' for ref in freq_refs) if freq_refs else '  (User must provide BPFO, BPFI, BSF, FTF)'}

    Examine envelope spectrum peaks:
    1. Check if ANY peak falls within ±5% of expected frequencies
    2. Check for harmonics: 2×BPFO, 3×BPFO, 2×BPFI, etc.
    3. List top 5-10 peaks with frequencies and magnitudes

    Optional visualization:
    Call: generate_envelope_report("{signal_file}",
                                  bpfo={bpfo or 'None'},
                                  bpfi={bpfi or 'None'},
                                  bsf={bsf or 'None'},
                                  ftf={ftf or 'None'},
                                  filter_low=500,
                                  filter_high=5000,
                                  max_freq=500)
    This saves an interactive HTML report to the reports/ directory showing:
    - Filtered signal with envelope overlay (time domain)
    - Envelope spectrum in dB scale with bearing frequency markers
    - Automatic bearing fault detection with confidence levels
    The tool returns the file path. Tell user to open the returned HTML file path in their browser to view the interactive envelope analysis with bearing fault markers.

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 5 — DIAGNOSTIC DECISION (EVIDENCE-BASED)
    ═══════════════════════════════════════════════════════════════════════════════

    Decision tree:

    A) IF envelope spectrum shows clear peak(s) at characteristic frequency (±5%):
       → Bearing fault type is STRONGLY INDICATED

       Classification by frequency:
       • Peak at BPFO (±5%) → **Outer race fault**
       • Peak at BPFI (±5%) → **Inner race fault**
       • Peak at BSF (±5%) → **Rolling element (ball) fault**
       • Peak at FTF (±5%) → **Cage fault**

       Confidence level:
       - High confidence: Peak + harmonics present AND (CF>6 OR Kurtosis>6)
       - Moderate confidence: Peak present but weaker harmonics OR moderate stats (CF 4-6, Kurt 3-6)
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
    • Use generate_*_report() tools to create HTML reports (saved to reports/ directory)
    • Tell user to open the HTML file path in browser for interactive visualizations
    • If user needs more details, offer "Show detailed analysis?" continuation
    • NEVER print large JSON/CSV data directly in text output
    """


    @mcp.prompt()
    def diagnose_gear(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        num_teeth: Optional[int] = None,
        rotation_speed_rpm: Optional[float] = None
    ) -> str:
        """
        Evidence-based guided workflow for gear diagnostics with strict anti-speculation rules.

        Args:
            signal_file: Name of the signal file
            sampling_rate: Sampling frequency in Hz (if None, will check metadata or ask user)
            num_teeth: Number of gear teeth (REQUIRED for GMF calculation)
            rotation_speed_rpm: Shaft rotation speed in RPM (REQUIRED for GMF and sideband identification)
        """
        fs_info = f"{sampling_rate}" if sampling_rate else "UNKNOWN"
        teeth_info = f"{num_teeth}" if num_teeth else "NOT PROVIDED"
        rpm_info = f"{rotation_speed_rpm}" if rotation_speed_rpm else "NOT PROVIDED"

        return f"""Perform an evidence-based gear diagnostic on signal "{signal_file}":

    ⚠️  CRITICAL INFERENCE POLICY ⚠️
    ═══════════════════════════════════════════════════════════════════════════════
    **NEVER INFER FAULT TYPE OR CONDITION FROM FILENAME**

    - Filename "{signal_file}" is an OPAQUE IDENTIFIER ONLY
    - "GearFault" in filename ≠ gear fault exists
    - "baseline" in filename ≠ healthy signal
    - "ToothDamage" in filename ≠ tooth damage exists

    **BASE DIAGNOSIS EXCLUSIVELY ON:**
    1. FFT spectrum showing GMF harmonics
    2. Sidebands spaced by shaft rotation frequency (f_rot)
    3. Statistical indicators (Kurtosis) as SECONDARY confirmation

    **IF FILENAME CONTRADICTS ANALYSIS:**
    Report: "Despite filename suggesting [X], analysis shows [Y]"
    Example: "Despite 'GearFault' in filename, FFT analysis shows NO GMF harmonics or sidebands"

    ═══════════════════════════════════════════════════════════════════════════════
    STEP 0 — FILENAME RESOLUTION & MANDATORY PARAMETER CHECK
    ═══════════════════════════════════════════════════════════════════════════════

    1. Verify signal file existence:
       Call list_available_signals() to get exact filename.
       If "{signal_file}" not found or multiple matches exist, ASK USER to clarify.
       Do NOT guess or auto-correct filenames.

    2. Required parameters:
       ✓ Signal file: {signal_file}
       {'✓' if sampling_rate else '✗'} Sampling rate: {fs_info} Hz
       {'✓' if num_teeth else '✗'} Number of teeth (Z): {teeth_info}
       {'✓' if rotation_speed_rpm else '✗'} Rotation speed: {rpm_info} RPM

       CRITICAL RULE: If sampling_rate is UNKNOWN, check signal metadata JSON first.
       If still missing OR if num_teeth OR rotation_speed_rpm are NOT PROVIDED:
       → STOP and ASK USER for these parameters before proceeding.
       → Explain: "Cannot perform gear diagnosis without [missing parameters]. Please provide: ..."

       Example response when parameters are missing:
       "I cannot proceed with the gear diagnosis because the following required
       parameters are missing:
       - Number of gear teeth (Z): needed to calculate Gear Mesh Frequency (GMF)
       - Rotation speed (RPM): needed to identify GMF and sidebands
       Please provide these values so I can complete the spectral analysis and
       identify gear faults."

       Do NOT use placeholder/default values. Do NOT proceed with incomplete data.
       Do NOT attempt diagnosis without num_teeth and rotation_speed_rpm.

    GUARDRAILS (apply throughout):
    - Do NOT infer faults from filename, path, or labels.
    - A gear tooth fault (localized damage) requires BOTH:
      • Clear Gear Mesh Frequency (GMF) harmonics AND
      • Sidebands spaced by shaft rotation frequency (f_rot) around GMF or its harmonics
      • (Optional but reinforcing) Elevated Kurtosis (>3) or modulation energy
    - Without sidebands: DO NOT claim tooth damage; consider uniform wear only if GMF elevated but stable statistics.

    STEP 1 — INPUT & CONTEXT
    Once all parameters confirmed:
    - f_rot = rotation_speed_rpm / 60 = {f"{rotation_speed_rpm/60:.2f}" if rotation_speed_rpm else "?"} Hz
    - Theoretical GMF = f_rot × Z = {f"{rotation_speed_rpm/60 * num_teeth:.2f}" if (rotation_speed_rpm and num_teeth) else "?"} Hz

    STEP 2 — STATISTICS (screening only)
    Call: analyze_statistics("{signal_file}")
    Report RMS, Crest Factor, Kurtosis in bullet points (brief).
    Indicators:
    - Elevated RMS: possible general load / imbalance
    - High Kurtosis (>3): impulsive impacts (may correlate with chipped tooth)
    - High Crest Factor (>4): impulsiveness
    (Do NOT diagnose from stats alone.)

    STEP 3 — SPECTRUM (frequency evidence)
    Call: analyze_fft("{signal_file}", {fs_info})
    Extract dominant peaks up to, e.g., 5× expected GMF. Identify:
    - GMF and its harmonics: GMF, 2×GMF, 3×GMF
    - Sidebands: GMF ± n·f_rot (n=1..3). Log their presence, spacing consistency, and relative amplitudes.
    Report top 5 peaks only (brief).

    Optional visualization:
    Call: generate_fft_report("{signal_file}", max_freq=5000, num_peaks=15)
    This saves an interactive HTML report to the reports/ directory showing FFT spectrum with automatic peak detection.
    Tell user to open the returned HTML file path in their browser to view the interactive FFT analysis.

    STEP 4 — OPTIONAL ENVELOPE (if strong modulation or impacts)
    If stats suggest impulsiveness OR sideband pattern partial:
    Call: analyze_envelope("{signal_file}", {fs_info}, 500, 5000)
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
    • Use generate_fft_report() tool to create HTML reports (saved to reports/ directory)
    • Tell user to open the HTML file path in browser for interactive visualizations
    • If user needs more details, offer "Show detailed analysis?" continuation
    • NEVER print large JSON/CSV data directly in text output
    """


    @mcp.prompt()
    def quick_diagnostic_report(signal_file: str) -> str:
        """
        Quick, evidence-aware screening report (non-definitive).

        Args:
            signal_file: Name of the signal file
        """
        return f"""Generate a concise screening report for "{signal_file}" using only observable evidence:

    STEP 0 — FILENAME RESOLUTION
    Call list_available_signals() to verify exact filename.
    If "{signal_file}" not found or multiple matches, ASK USER to clarify.

    Guardrails:
    - Ignore filenames/paths as diagnostic evidence.
    - Do NOT diagnose faults from statistics alone; use them for screening only.
    - Use cautious language: "possible/consistent with" unless corroborated by multiple indicators.

    1) Load & sanity checks
    - Report number of samples, duration (s), min/max values (brief, 1 line).

    2) Statistics (screening)
    Call: analyze_statistics("{signal_file}")
    Report: RMS, Crest Factor, Kurtosis, Skewness (bullet points only).
    Flags (screening thresholds, not definitive):
    - CF > 4 → impulsiveness present; CF > 6 → strong impulsiveness
    - Kurtosis > 0 → non-Gaussian / impulsive content (excess kurtosis); > 3 → significant; > 6 → severe
    Note: These flags alone are insufficient for fault identification.

    3) Spectral snapshot
    Call: analyze_fft("{signal_file}", 10000)
    - Report peak frequency, magnitude (top 3 peaks only).
    - If operating speed is known, relate peaks to 1×/2× RPM; otherwise, request it for deeper interpretation.

    4) Next-step guidance (evidence-first)
    - If strong impulsiveness (CF>6 or Kurtosis>6), suggest: "Use diagnose_bearing prompt for targeted bearing analysis"
    - If tonal/harmonic pattern dominates, suggest: "Use diagnose_gear prompt if gear suspected"
    - If broadband increase, suggest: ISO 20816-3 check with evaluate_iso_20816()

    Output format (≤200 words):
    - Screening summary with measured values (bullet points)
    - No definitive fault labels
    - List recommended targeted analyses and required missing parameters
    """


    @mcp.prompt()
    def generate_iso_diagnostic_report(
        signal_file: str,
        sampling_rate: float = 10000.0,
        machine_group: int = 1,
        support_type: str = "rigid",
        operating_speed_rpm: Optional[float] = None,
        machine_name: str = "Machine",
        measurement_location: str = "Bearing"
    ) -> str:
        """
        Generate comprehensive diagnostic report with ISO 20816-3 compliance evaluation.

        Creates a structured diagnostic report including:
        - ISO 20816-3 vibration severity assessment
        - Statistical indicators
        - Spectral analysis
        - Fault detection (bearing/gear)
        - Maintenance recommendations

        Args:
            signal_file: Name of the signal file to analyze
            sampling_rate: Sampling frequency in Hz
            machine_group: ISO 20816 group (1=large >300kW, 2=medium 15-300kW)
            support_type: 'rigid' or 'flexible'
            operating_speed_rpm: Operating speed in RPM
            machine_name: Machine identifier
            measurement_location: Measurement point description
        """
        rpm_param = f", operating_speed_rpm={operating_speed_rpm}" if operating_speed_rpm else ""

        return f"""Generate a comprehensive diagnostic report for {machine_name} - {measurement_location}

    SIGNAL: {signal_file}
    SAMPLING RATE: {sampling_rate} Hz
    MACHINE GROUP: {machine_group} ({'Large >300kW' if machine_group == 1 else 'Medium 15-300kW'})
    SUPPORT TYPE: {support_type.title()}
    OPERATING SPEED: {operating_speed_rpm if operating_speed_rpm else 'Not specified'} RPM

    ================================================================================
    SECTION 1: ISO 20816-3 VIBRATION SEVERITY ASSESSMENT
    ================================================================================

    Execute: evaluate_iso_20816("{signal_file}", {sampling_rate}, {machine_group}, "{support_type}"{rpm_param})

    Present results in this format:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ ISO 20816-3 EVALUATION RESULT                                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │ RMS Velocity (broadband):     [VALUE] mm/s                          │
    │ Frequency Range:               [RANGE] Hz                           │
    │ Evaluation Zone:               Zone [A/B/C/D]                       │
    │ Severity Level:                [Good/Acceptable/Unsatisfactory/     │
    │                                 Unacceptable]                        │
    │ Color Code:                    🟢/🟡/🟠/🔴                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │ ZONE BOUNDARIES (mm/s):                                             │
    │   Zone A/B: [VALUE]  |  Zone B/C: [VALUE]  |  Zone C/D: [VALUE]    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ INTERPRETATION:                                                     │
    │ [Zone description from result]                                      │
    └─────────────────────────────────────────────────────────────────────┘

    ISO COMPLIANCE STATUS:
    • If Zone A (Green): ✅ COMPLIANT - Machine in excellent condition
    • If Zone B (Yellow): ⚠️  ACCEPTABLE - Continue normal operation, monitor
    • If Zone C (Orange): ⚠️  NON-COMPLIANT - Plan maintenance within 1-3 months
    • If Zone D (Red): 🚨 CRITICAL - Immediate action required, risk of damage

    ================================================================================
    SECTION 2: STATISTICAL INDICATORS
    ================================================================================

    Execute: analyze_statistics("{signal_file}")

    Report the following parameters:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ STATISTICAL ANALYSIS                                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │ RMS:                  [VALUE] (Energy level)                        │
    │ Peak:                 [VALUE] (Maximum amplitude)                   │
    │ Peak-to-Peak:         [VALUE] (Total excursion)                     │
    │ Crest Factor:         [VALUE] (Peak/RMS ratio)                      │
    │ Kurtosis:             [VALUE] (Impulsiveness indicator)             │
    │ Skewness:             [VALUE] (Asymmetry indicator)                 │
    └─────────────────────────────────────────────────────────────────────┘

    DIAGNOSTIC INDICATORS:
    • Crest Factor > 4: ⚠️  Possible presence of impulses (bearing faults)
    • Crest Factor > 6: 🚨 High probability of bearing defects
    • Kurtosis > 0: ⚠️  Non-Gaussian / impulsive content (excess kurtosis, Fisher convention)
    • Kurtosis > 3: ⚠️  Significant impulsive content (bearing damage likely)
    • Kurtosis > 6: 🚨 Severe bearing damage (strong impulses)

    ================================================================================
    SECTION 3: SPECTRAL ANALYSIS
    ================================================================================

    Execute: analyze_fft("{signal_file}", {sampling_rate}, max_frequency=1000)

    Identify:
    • Peak frequency and magnitude
    • Frequency resolution
    • Energy distribution across spectrum

    Execute: plot_spectrum("{signal_file}", {sampling_rate}, freq_range=[0, 1000], num_peaks=15)

    Look for:
    • Dominant frequencies (possible fault indicators)
    • Harmonics pattern (multiples of rotation frequency)
    • Sidebands (modulation indicators)
    • Broadband noise level

    ================================================================================
    SECTION 4: BEARING FAULT DETECTION
    ================================================================================

    Execute: analyze_envelope("{signal_file}", {sampling_rate}, filter_low=500, filter_high=5000, num_peaks=10)

    Execute: plot_envelope("{signal_file}", {sampling_rate}, filter_band=[500, 5000], freq_range=[0, 100])

    Analyze envelope spectrum peaks and compare with:
    • BPFO (Ball Pass Frequency - Outer race): Outer race defect
    • BPFI (Ball Pass Frequency - Inner race): Inner race defect
    • BSF (Ball Spin Frequency): Rolling element defect
    • FTF (Fundamental Train Frequency): Cage defect

    Note: Envelope peaks at harmonics of these frequencies indicate bearing damage

    ================================================================================
    SECTION 5: OVERALL ASSESSMENT AND RECOMMENDATIONS
    ================================================================================

    Based on all analyses, provide:

    MACHINE CONDITION SUMMARY:
    ├─ ISO 20816-3 Status: [Compliant/Non-compliant]
    ├─ Vibration Severity: [Zone A/B/C/D - Color code]
    ├─ Fault Indicators: [Present/Absent]
    └─ Urgency Level: [Normal/Monitor/Plan Maintenance/Immediate Action]

    IDENTIFIED ISSUES (if any):
    • [List any detected faults based on statistical/spectral/envelope analysis]

    RECOMMENDATIONS:
    1. IMMEDIATE ACTIONS (if Zone D or critical indicators):
       - [Specific actions needed]

    2. SHORT-TERM (1-3 months, if Zone C):
       - [Maintenance planning recommendations]

    3. MONITORING (if Zone B):
       - [Suggested monitoring frequency and parameters]

    4. ROUTINE OPERATION (if Zone A):
       - [Continue normal operation, periodic checks]

    ADDITIONAL DIAGNOSTICS (if needed):
    • Consider trending analysis for Zone B/C
    • Perform time-domain analysis if high Crest Factor
    • Check alignment if high 1× RPM component
    • Inspect lubrication if broadband noise increase

    ================================================================================
    REPORT GENERATED: [Current date/time]
    ANALYZED BY: ISO 20816-3 Diagnostic System
    ================================================================================
    """
