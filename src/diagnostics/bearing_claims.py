"""Bearing claims: what a bearing check may state from its evidence.

The evidence layer (:mod:`.bearing_evidence`) decides, on one envelope
spectrum, which tested lines stand above the local noise floor, which further
lines corroborate them and which outcome the check reaches. This module
decides what the server may claim from that result, at one of three levels:

- ``withheld``: no fault type is named, and the statement says why the record
  supports none;
- ``qualified``: the evidence and every competing explanation the result
  cannot exclude are named, and no single type is singled out;
- ``single_spectrum_evidence``: the envelope evidence of this single spectrum
  is consistent with one type, under the stated assumption that the record's
  background is random measurement noise; its actions are confirmation steps,
  never a maintenance action on the bearing.

A fourth level, ``supported_claim``, is reserved for evidence independent of
one spectrum (such as a reference-relative rise on comparable measurements of
the asset ledger). This module never produces it.

The conditions that qualify an indicated type are predeclared window
predicates on the declared frequencies, with the check's own search tolerance
and no tuning. They qualify the claim only: they never change the outcome, the
candidates, a state or a line test, and they add no sharing or detection rule.

- Cage comb (``cage_comb_not_excluded``): a bearing type other than the cage
  is indicated, the cage line was not testable in this record, and an integer
  multiple of the declared cage frequency lies inside the indicated line's
  search window. BPFO = Z x FTF makes this structural for the outer race.
- Inner-race sideband (``inner_race_sideband_not_excluded``): the outer race
  is indicated, the inner-race line stands above the floor without being
  supported, and the declared BPFI minus twice the shaft rate lies inside the
  outer-race search window.
- Rolling element held back (``rolling_element_held_back``): the evidence
  layer held a lone cage back because the rolling-element line is significant
  (its state reason ``rolling_element_line_present``).

A label outside the bearing vocabulary never gets these predicates or a
bearing element in its sentences. An acquisition known to be outside the line
test's model (a generated signal without noise, as the signal's own companion
metadata declares it) withholds every claim. That marker is unvalidated, so it
may only make a claim more cautious; an ``unknown`` acquisition is not a
validation of the model's assumptions.

Every sentence and action this module writes comes from ``CLAIM_SENTENCES``,
one closed list. The report surfaces (the advisory of
``generate_diagnostic_report``, its envelope figure and badges, the envelope
report and the caller-supplied fault types of
``generate_maintenance_recommendations``) take their bearing wording from the
same list, through the functions at the end of this module.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .bearing_evidence import (
    _CAGE,
    _INNER_RACE,
    _NAMES,
    _ROLLING_ELEMENT,
    Hypothesis,
    _capital,
    _join,
)

#: Claim levels, from the most cautious. ``supported_claim`` is reserved for
#: evidence independent of one spectrum and is never produced here.
WITHHELD = "withheld"
QUALIFIED = "qualified"
SINGLE_SPECTRUM_EVIDENCE = "single_spectrum_evidence"
CLAIM_LEVELS = (WITHHELD, QUALIFIED, SINGLE_SPECTRUM_EVIDENCE)

#: What the record's own metadata says about the line test's model. There is
#: no value for a record known to fit it: no route records what would show it.
KNOWN_INAPPLICABLE = "known_inapplicable"
UNKNOWN = "unknown"
ACQUISITION_ELIGIBILITIES = (KNOWN_INAPPLICABLE, UNKNOWN)

CLAIM_BASIS = "significance_tested"
CLAIM_POLICY_VERSION = 1

# Conditions: why a claim is qualified or withheld.
GENERATED_WITHOUT_NOISE = "generated_without_noise"
SEVERAL_TYPES = "several_types"
CAGE_COMB_NOT_EXCLUDED = "cage_comb_not_excluded"
INNER_RACE_SIDEBAND_NOT_EXCLUDED = "inner_race_sideband_not_excluded"
ROLLING_ELEMENT_HELD_BACK = "rolling_element_held_back"
INCONCLUSIVE = "inconclusive"
NOTHING_TESTABLE = "nothing_testable"
NO_INDICATION = "no_indication"

# Limitations stated with single-spectrum evidence, without a level change.
SIDEBAND_ONLY_CORROBORATION = "sideband_only_corroboration"
CAGE_NOT_DECLARED = "cage_not_declared"

#: Qualified claims that still rest on a supported type: they count as one
#: bearing finding in the diagnosis's evidence points.
_FINDING_CONDITIONS = frozenset(
    {SEVERAL_TYPES, CAGE_COMB_NOT_EXCLUDED, INNER_RACE_SIDEBAND_NOT_EXCLUDED}
)

_OUTER_RACE = "outer_race"
#: Bearing types a cage-rate comb or an undeclared cage can concern.
_NOT_CAGE = (_OUTER_RACE, _INNER_RACE, _ROLLING_ELEMENT)

#: Untestable reasons a longer capture removes.
_TOO_SHORT = frozenset({"search_window_too_narrow", "noise_floor_not_estimable"})

#: Every sentence and action this module writes (one closed list).
CLAIM_SENTENCES: dict[str, str] = {
    # single_spectrum_evidence
    "evidence": (
        "The envelope evidence of this single spectrum is consistent with "
        "{defect}: the {line} at {peak_hz:.2f} Hz stands {level_db:.1f} dB above "
        "the local noise floor. This assumes the record's background is random "
        "measurement noise, which its metadata cannot confirm; it is not a "
        "confirmed fault."
    ),
    "evidence_label": (
        "The envelope evidence of this single spectrum is consistent with a "
        "periodic line at the declared {label} frequency: the {label} line at "
        "{peak_hz:.2f} Hz stands {level_db:.1f} dB above the local noise floor. "
        "This assumes the record's background is random measurement noise, which "
        "its metadata cannot confirm; it is not a confirmed fault."
    ),
    "limitation_sideband_only": (
        "Its only corroborating lines are shaft-rate sidebands, which shaft-rate "
        "modulation of the background can also produce, so they do not "
        "independently confirm an inner-race defect."
    ),
    "limitation_cage_not_declared": (
        "No cage rate was declared, so whether cage-rate lines could explain this "
        "line was not assessed."
    ),
    # qualified
    "several_types": (
        "The envelope evidence of this single spectrum is consistent with more "
        "than one fault type, and none is singled out: {candidates}. They are "
        "listed by level above the local noise floor, which does not rank them. "
        "This is not a confirmed fault."
    ),
    "several_types_candidate": (
        "the {line} at {peak_hz:.2f} Hz ({level_db:.1f} dB) for {defect}"
    ),
    "several_types_candidate_label": (
        "the {label} line at {peak_hz:.2f} Hz ({level_db:.1f} dB) for a periodic "
        "line at the declared {label} frequency"
    ),
    "not_excluded": (
        "The envelope evidence of this single spectrum is consistent with "
        "{defect}, but {alternatives}. No single fault type is named from this "
        "spectrum."
    ),
    "not_excluded_joiner": "; and ",
    "cage_comb_too_short": (
        "this record is too short to test the cage line (FTF), and a harmonic of "
        "the cage frequency falls within the search window of the {line}, so a "
        "cage defect cannot be excluded"
    ),
    "cage_comb_untested": (
        "the cage line (FTF) could not be tested in this record, and a harmonic of "
        "the cage frequency falls within the search window of the {line}, so a "
        "cage defect cannot be excluded"
    ),
    "inner_race_sideband": (
        "the inner-race line (BPFI) also stands above the local noise floor, and "
        "its sideband at BPFI minus twice the shaft rate falls within the search "
        "window of the outer-race line (BPFO), so an inner-race defect cannot be "
        "excluded"
    ),
    "rolling_element_held_back": (
        "The significance-tested evidence is inconclusive and does not support a "
        "unique fault type: the cage lines may come from a cage defect or from a "
        "rolling-element defect."
    ),
    "rolling_element_held_back_shaft_order": (
        "The significance-tested evidence is inconclusive and does not support a "
        "unique fault type: the cage lines may come from a cage defect or from a "
        "rolling-element defect, and the line in the rolling-element window lies "
        "at a shaft order and may be a shaft harmonic."
    ),
    # withheld
    "inconclusive": (
        "The significance-tested evidence is inconclusive and does not support a "
        "unique fault type."
    ),
    "no_indication": (
        "No eligible line at a tested characteristic frequency stands above the "
        "local noise floor in this record and band. This is not evidence that the "
        "bearing is free of defects."
    ),
    "no_indication_label": (
        "No eligible line at a tested frequency stands above the local noise floor "
        "in this record and band. This is not evidence that no defect exists."
    ),
    "untested": "{lines} could not be tested in this record.",
    "nothing_testable": (
        "None of the lines could be tested in this record, so no fault type is "
        "assessed."
    ),
    "generated_without_noise": (
        "This is a generated signal without noise; the line test assumes a "
        "recording with measurement noise, so no bearing fault type is assessed."
    ),
    "generated_without_noise_label": (
        "This is a generated signal without noise; the line test assumes a "
        "recording with measurement noise, so no fault type is assessed."
    ),
    # diagnose_vibration: the bearing statement, actions and the aggregate
    "pipeline_statement": "Bearing analysis: {statement}",
    "action_prefix": "Bearing {bearing_id}: ",
    "action_prefix_unnamed": "Bearing check: ",
    "action_evidence": (
        "{prefix}confirm the single-spectrum indication with a comparable "
        "measurement or an inspection of the {element}."
    ),
    "action_evidence_label": (
        "{prefix}confirm the single-spectrum indication of the {label} line with a "
        "comparable measurement."
    ),
    "action_several_types": (
        "{prefix}inspect {elements} with equal priority, and compare with a "
        "comparable earlier measurement."
    ),
    "action_several_types_label": (
        "{prefix}compare with a comparable earlier measurement before acting on "
        "the {labels} lines."
    ),
    "action_cage_comb_too_short": (
        "{prefix}repeat the measurement with a capture long enough to test the "
        "cage line, or compare with a comparable earlier measurement."
    ),
    "action_cage_comb_untested": (
        "{prefix}repeat the measurement so that the cage line can be tested, or "
        "compare with a comparable earlier measurement."
    ),
    "action_inner_race_sideband": (
        "{prefix}inspect the outer and inner races with equal priority, or compare "
        "with a comparable earlier measurement."
    ),
    "action_repeat": (
        "{prefix}repeat the measurement, or compare it with a comparable earlier "
        "measurement."
    ),
    "action_nothing_testable": "{prefix}{step}",
    "aggregate": (
        "Combined evidence across the analyses that ran: {strength}. It counts "
        "findings, is not a probability, and does not identify or confirm a fault "
        "type."
    ),
    # generate_diagnostic_report: the advisory's bearing section, one sentence
    # per tested line
    "block_intro": (
        "Envelope line test for bearing {bearing_id} at {shaft_hz:.1f} Hz shaft speed."
    ),
    "block_intro_unnamed": "Envelope line test at {shaft_hz:.1f} Hz shaft speed.",
    "block_supported": (
        "The {line} at {peak_hz:.2f} Hz stands {level_db:.1f} dB above the local "
        "noise floor, and further lines of its kinematics also stand above the "
        "floor in the same spectrum."
    ),
    "block_supported_label": (
        "The {line} at {peak_hz:.2f} Hz stands {level_db:.1f} dB above the local "
        "noise floor, and harmonics of it also stand above the floor in the same "
        "spectrum."
    ),
    "block_not_counted": (
        "The {line} at {peak_hz:.2f} Hz stands {level_db:.1f} dB above the local "
        "noise floor but is not counted as support for a fault type."
    ),
    "block_absent": (
        "The {line} does not stand above the local noise floor in this record and "
        "band."
    ),
    "block_not_testable": "The {line} could not be tested in this record.",
    # badges of the per-line table (integrated report, envelope report)
    "badge_supported": "above the floor, corroborated",
    "badge_not_counted": "above the floor, not counted",
    "badge_absent": "not above the floor",
    "badge_not_testable": "not testable",
    "badge_not_assessed": "not assessed",
    # the advisory's verdict, after the claim statement and its actions
    "verdict_zone_elevated": (
        "Broadband vibration is elevated (Zone {zone}), and the envelope evidence "
        "names no bearing fault type, so bearing and non-bearing sources both "
        "remain possible."
    ),
    "verdict_anomaly_flag": (
        "The anomaly model places this signal outside the pattern it learned, and "
        "the envelope evidence names no bearing fault type, so bearing and "
        "non-bearing sources both remain possible."
    ),
    # the advisory's evidence: the counted lines of each candidate
    "evidence_lines": (
        "Lines counted for the {line}: the line at {peak_hz:.2f} Hz "
        "({level_db:.1f} dB above the local noise floor), with {members}. Each "
        "stands above the local noise floor of this spectrum in its line test: "
        "the tested lines share a nominal false-alarm level of {check_pct:g} % per "
        "check, and each further line is tested at {line_pct:g} %. Lines of one "
        "spectrum are not independent evidence and do not confirm a fault."
    ),
    "member_harmonic": "harmonic {harmonic} at {peak_hz:.2f} Hz ({level_db:.1f} dB)",
    "member_shaft_sideband": (
        "a shaft-rate sideband at {peak_hz:.2f} Hz ({level_db:.1f} dB)"
    ),
    "member_cage_sideband": (
        "a cage-rate sideband at {peak_hz:.2f} Hz ({level_db:.1f} dB)"
    ),
    "member_subharmonic": "the 1 x BSF line at {peak_hz:.2f} Hz ({level_db:.1f} dB)",
    # the advisory's disagreements (ISO zone A or B)
    "disagreement_evidence": (
        "Indicators differ. Zone {zone} describes overall vibration energy as "
        "acceptable, while the envelope evidence of this single spectrum is "
        "consistent with {defect}. Both can hold: a localised defect can stand out "
        "in the envelope spectrum while adding little broadband energy. The "
        "envelope evidence warrants confirmation despite the acceptable zone; it "
        "is not a confirmed fault."
    ),
    "periodic_line": "a periodic line at the declared {label} frequency",
    "disagreement_anomaly": (
        "Indicators differ. Zone {zone} describes overall vibration energy as "
        "acceptable, while the anomaly model ({health}, {ratio_pct:.0f}% of "
        "segments) places this signal outside the pattern it learned. The anomaly "
        "model does not identify a fault type or a bearing element."
    ),
    # the advisory's bearing recommendations (the actions are claim_actions)
    "action_description_evidence": (
        "Single-spectrum envelope evidence is not a confirmed fault; this step "
        "confirms or excludes it."
    ),
    "action_description_qualified": (
        "The envelope evidence does not single out one fault type; this step "
        "addresses the readings it cannot exclude."
    ),
    "action_description_withheld": (
        "The envelope evidence supports no fault type in this record; this step "
        "can make the record assessable."
    ),
    "evidence_strength_explanation": (
        "Evidence strength counts the findings of the analyses that ran, a bearing "
        "finding once. It is not a confidence score, it is not a probability, and "
        "it does not identify or confirm a fault type."
    ),
    # the advisory's baseline comparison, per candidate
    "delta_envelope": (
        "Envelope amplitude of the {line} is {delta_db:.1f} dB {direction} than "
        "baseline (strongest bin of its search window: {peak_hz:.2f} Hz, baseline "
        "{baseline_peak_hz:.2f} Hz)."
    ),
    "delta_envelope_unchanged": (
        "Envelope amplitude of the {line} is unchanged against baseline."
    ),
    # the annotated envelope figure
    "figure_caption": (
        "Envelope spectrum with the {labels} search windows overlaid. Each shaded "
        "band is the ±{tolerance_pct:g}% window around the line its check "
        "tests{rolling_element}. A peak inside a window is not a finding by "
        "itself: the check counts a line only when it stands above the local noise "
        "floor and a further line it predicts does too."
    ),
    "figure_rolling_element": " (2 x BSF for the rolling element)",
    "figure_emphasis_one": (
        "The emphasised window ({labels}) holds the candidate line of the bearing "
        "statement in this report."
    ),
    "figure_emphasis_many": (
        "The emphasised windows ({labels}) hold the candidate lines of the bearing "
        "statement in this report, emphasised alike."
    ),
    "figure_emphasis_none": (
        "No window is emphasised: the bearing statement in this report names no "
        "candidate line."
    ),
    "figure_annotation": "{name} {line_hz:.2f} Hz",
    "figure_annotation_candidate": (
        "{name} {line_hz:.2f} Hz: strongest bin {peak_hz:.1f} Hz, {level_db:.1f} dB "
        "above the floor"
    ),
    # generate_envelope_report: one label per candidate line
    "envelope_label": (
        "{name} line at {peak_hz:.2f} Hz, {level_db:.1f} dB above the local noise "
        "floor and corroborated in this spectrum"
    ),
    "envelope_label_no_speed": (
        "{name} line at {peak_hz:.2f} Hz, {level_db:.1f} dB above the local noise "
        "floor and corroborated in this spectrum; no shaft speed is declared, so "
        "the sideband test and the shaft-order test were not run"
    ),
    "envelope_no_speed": (
        "No shaft speed is declared for this signal (the rpm of its measurement or "
        "the nominal rpm of its measurement point), so the line test ran without "
        "the sideband test and the shaft-order test."
    ),
    # generate_maintenance_recommendations: a fault type the caller supplies
    "caller_fault_type": (
        "Fault type supplied by the caller ({fault}); this tool does not verify it."
    ),
}

_S = CLAIM_SENTENCES


def validate_acquisition_eligibility(value: str) -> str:
    """Refuse an acquisition eligibility outside the closed vocabulary.

    Args:
        value: The eligibility the caller passes.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value is not "known_inapplicable" or "unknown".
    """
    if value not in ACQUISITION_ELIGIBILITIES:
        raise ValueError(
            f"acquisition_eligibility must be 'known_inapplicable' or 'unknown', "
            f"got {value!r}; pass 'unknown' unless the signal's own metadata "
            f"declares a generated signal without noise."
        )
    return value


def acquisition_eligibility_from_companion(source_metadata: Mapping[str, Any]) -> str:
    """The acquisition eligibility a signal's companion metadata declares.

    Only a companion that declares a generated signal (a ``generator`` key)
    with a numeric ``noise_level`` of exactly 0 is known to be outside the line
    test's model. Everything else, a positive noise level and a signal without
    a companion included, is unknown: the marker can only withhold a claim.

    Args:
        source_metadata: The signal's stored companion metadata.

    Returns:
        "known_inapplicable" or "unknown".
    """
    noise = source_metadata.get("noise_level")
    numeric = isinstance(noise, (int, float)) and not isinstance(noise, bool)
    if "generator" in source_metadata and numeric and noise == 0:
        return KNOWN_INAPPLICABLE
    return UNKNOWN


def _index(hypotheses: Sequence[Hypothesis], label: str) -> int:
    return next(n for n, h in enumerate(hypotheses) if h.label == label)


def _first(hypotheses: Sequence[Hypothesis], canonical: str) -> Optional[int]:
    return next((n for n, h in enumerate(hypotheses) if h.canonical == canonical), None)


def _sideband_only(canonical: Optional[str], check: Mapping[str, Any]) -> bool:
    """Whether an inner-race line's only counted corroboration is sidebands."""
    counted = {m["kind"] for m in check["corroboration"] if m["counted"]}
    return canonical == _INNER_RACE and counted == {"sideband"}


def _cage_comb_in_window(line_hz: float, cage_hz: float, tolerance: float) -> bool:
    """Whether some k x cage_hz, k = 1 .. floor(line (1 + tol) / cage) + 1,
    lies within tolerance x line_hz of line_hz."""
    if not cage_hz > 0:
        return False
    k_max = math.floor(line_hz * (1.0 + tolerance) / cage_hz) + 1
    return any(
        abs(k * cage_hz - line_hz) <= tolerance * line_hz for k in range(1, k_max + 1)
    )


def _cage_comb_not_excluded(
    hypotheses: Sequence[Hypothesis],
    checks: Sequence[Mapping[str, Any]],
    x: Hypothesis,
    cage: Optional[int],
    tolerance: float,
) -> bool:
    """Whether the cage line was not testable and a cage multiple lies in x's window."""
    return (
        cage is not None
        and checks[cage]["state"] == "not_testable"
        and _cage_comb_in_window(x.line_hz, hypotheses[cage].line_hz, tolerance)
    )


def _inner_race_sideband_not_excluded(
    hypotheses: Sequence[Hypothesis],
    checks: Sequence[Mapping[str, Any]],
    x: Hypothesis,
    inner: Optional[int],
    shaft_hz: Optional[float],
    tolerance: float,
) -> bool:
    """Whether the inner-race line stands above the floor and its sideband at
    BPFI minus twice the shaft rate lies in the outer-race window of x."""
    return (
        x.canonical == _OUTER_RACE
        and inner is not None
        and shaft_hz is not None
        and checks[inner]["state"] not in ("absent", "not_testable")
        and abs(hypotheses[inner].line_hz - 2.0 * shaft_hz - x.line_hz)
        <= tolerance * x.line_hz
    )


def bearing_claim(
    evidence: Mapping[str, Any],
    hypotheses: Sequence[Hypothesis],
    *,
    shaft_hz: Optional[float],
    tolerance_pct: float,
    acquisition_eligibility: str,
) -> dict[str, Any]:
    """The claim a check's significance-tested evidence allows.

    Args:
        evidence: The result of :func:`.bearing_evidence.evaluate_hypotheses`
            (``checks`` in hypothesis order, ``outcome``, ``candidates``).
        hypotheses: The tested hypotheses, in the order of ``evidence``.
        shaft_hz: Shaft rate (Hz); None skips the inner-race sideband test.
        tolerance_pct: Half-width of the fundamental search windows, in
            percent of the line frequency (the check's own tolerance).
        acquisition_eligibility: "known_inapplicable" or "unknown".

    Returns:
        ``level``, ``conditions``, ``limitations``, ``acquisition_eligibility``,
        ``basis`` and ``policy_version``.
    """
    validate_acquisition_eligibility(acquisition_eligibility)
    checks = evidence["checks"]
    outcome = evidence["outcome"]
    tolerance = tolerance_pct / 100.0
    conditions: list[str] = []
    limitations: list[str] = []

    if acquisition_eligibility == KNOWN_INAPPLICABLE:
        level, conditions = WITHHELD, [GENERATED_WITHOUT_NOISE]
    elif outcome == "indicated":
        n = _index(hypotheses, evidence["candidates"][0]["fault_type"])
        x = hypotheses[n]
        cage = _first(hypotheses, _CAGE)
        if x.canonical in _NOT_CAGE:
            if _cage_comb_not_excluded(hypotheses, checks, x, cage, tolerance):
                conditions.append(CAGE_COMB_NOT_EXCLUDED)
            inner = _first(hypotheses, _INNER_RACE)
            if _inner_race_sideband_not_excluded(
                hypotheses, checks, x, inner, shaft_hz, tolerance
            ):
                conditions.append(INNER_RACE_SIDEBAND_NOT_EXCLUDED)
        if conditions:
            level = QUALIFIED
        else:
            level = SINGLE_SPECTRUM_EVIDENCE
            if _sideband_only(x.canonical, checks[n]):
                limitations.append(SIDEBAND_ONLY_CORROBORATION)
            if x.canonical in _NOT_CAGE and cage is None:
                limitations.append(CAGE_NOT_DECLARED)
    elif outcome == "several_types":
        level, conditions = QUALIFIED, [SEVERAL_TYPES]
    elif outcome == "inconclusive":
        if any(c["state_reason"] == "rolling_element_line_present" for c in checks):
            level, conditions = QUALIFIED, [ROLLING_ELEMENT_HELD_BACK]
        elif all(c["state"] == "not_testable" for c in checks):
            level, conditions = WITHHELD, [NOTHING_TESTABLE]
        else:
            level, conditions = WITHHELD, [INCONCLUSIVE]
    else:
        level, conditions = WITHHELD, [NO_INDICATION]

    return {
        "level": level,
        "conditions": conditions,
        "limitations": limitations,
        "acquisition_eligibility": acquisition_eligibility,
        "basis": CLAIM_BASIS,
        "policy_version": CLAIM_POLICY_VERSION,
    }


def named_candidate(
    claim: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """The one candidate a claim names: only at single-spectrum evidence."""
    if claim["level"] != SINGLE_SPECTRUM_EVIDENCE:
        return None
    (candidate,) = evidence["candidates"]
    return dict(candidate)


def counts_as_finding(claim: Mapping[str, Any]) -> bool:
    """Whether a claim adds one bearing finding to the diagnosis's points.

    Single-spectrum evidence does, and so does a qualified claim from an
    indicated or several-types outcome; a withheld claim and a qualified one
    from an inconclusive outcome do not.
    """
    return claim["level"] == SINGLE_SPECTRUM_EVIDENCE or (
        claim["level"] == QUALIFIED
        and bool(_FINDING_CONDITIONS.intersection(claim["conditions"]))
    )


def _line_name(canonical: Optional[str], label: str) -> str:
    names = _NAMES.get(canonical or "")
    return names.line if names else f"{label} line"


def claim_statement(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    hypotheses: Sequence[Hypothesis],
) -> str:
    """The overall assessment of a check, from its claim.

    Args:
        claim: The result of :func:`bearing_claim`.
        evidence: The evidence result the claim was computed from.
        hypotheses: The tested hypotheses, in the order of ``evidence``.

    Returns:
        One statement from ``CLAIM_SENTENCES``.
    """
    checks = evidence["checks"]
    conditions = claim["conditions"]
    vocabulary = all(h.canonical in _NAMES for h in hypotheses)

    if GENERATED_WITHOUT_NOISE in conditions:
        key = (
            "generated_without_noise" if vocabulary else "generated_without_noise_label"
        )
        return _S[key]
    if NO_INDICATION in conditions:
        sentence = _S["no_indication" if vocabulary else "no_indication_label"]
        untested = [
            f"the {_line_name(h.canonical, h.label)}"
            for h, c in zip(hypotheses, checks)
            if c["state"] == "not_testable"
        ]
        if untested:
            sentence += " " + _S["untested"].format(lines=_capital(_join(untested)))
        return sentence
    if NOTHING_TESTABLE in conditions:
        return _S["nothing_testable"]
    if INCONCLUSIVE in conditions:
        return _S["inconclusive"]
    if ROLLING_ELEMENT_HELD_BACK in conditions:
        at_shaft_order = any(
            h.canonical == _ROLLING_ELEMENT
            and c["state_reason"] == "line_at_shaft_order"
            for h, c in zip(hypotheses, checks)
        )
        key = (
            "rolling_element_held_back_shaft_order"
            if at_shaft_order
            else "rolling_element_held_back"
        )
        return _S[key]
    if SEVERAL_TYPES in conditions:
        listed = []
        for candidate in evidence["candidates"]:
            n = _index(hypotheses, candidate["fault_type"])
            listed.append(_candidate_phrase(hypotheses[n], checks[n]))
        return _S["several_types"].format(candidates=_join(listed))

    (candidate,) = evidence["candidates"]
    n = _index(hypotheses, candidate["fault_type"])
    x, check = hypotheses[n], checks[n]
    names = _NAMES.get(x.canonical or "")
    if claim["level"] == QUALIFIED:
        alternatives = []
        if CAGE_COMB_NOT_EXCLUDED in conditions:
            cage = checks[_first(hypotheses, _CAGE)]
            key = (
                "cage_comb_too_short"
                if cage["state_reason"] in _TOO_SHORT
                else "cage_comb_untested"
            )
            alternatives.append(_S[key].format(line=names.line))
        if INNER_RACE_SIDEBAND_NOT_EXCLUDED in conditions:
            alternatives.append(_S["inner_race_sideband"])
        return _S["not_excluded"].format(
            defect=names.defect,
            alternatives=_S["not_excluded_joiner"].join(alternatives),
        )

    if names:
        sentence = _S["evidence"].format(
            defect=names.defect,
            line=names.line,
            peak_hz=check["line_peak_hz"],
            level_db=check["level_db"],
        )
    else:
        sentence = _S["evidence_label"].format(
            label=x.label, peak_hz=check["line_peak_hz"], level_db=check["level_db"]
        )
    limitations = claim["limitations"]
    if SIDEBAND_ONLY_CORROBORATION in limitations:
        sentence += " " + _S["limitation_sideband_only"]
    if CAGE_NOT_DECLARED in limitations:
        sentence += " " + _S["limitation_cage_not_declared"]
    return sentence


def _candidate_phrase(hypothesis: Hypothesis, check: Mapping[str, Any]) -> str:
    names = _NAMES.get(hypothesis.canonical or "")
    if names:
        return _S["several_types_candidate"].format(
            defect=names.defect,
            line=names.line,
            peak_hz=check["line_peak_hz"],
            level_db=check["level_db"],
        )
    return _S["several_types_candidate_label"].format(
        label=hypothesis.label,
        peak_hz=check["line_peak_hz"],
        level_db=check["level_db"],
    )


def pipeline_statement(summary: Mapping[str, Any]) -> str:
    """The bearing sentence of ``diagnose_vibration``, from a check summary."""
    return _S["pipeline_statement"].format(statement=summary["overall_assessment"])


def claim_actions(summary: Mapping[str, Any]) -> list[str]:
    """The bearing actions of ``diagnose_vibration``, from a check summary.

    Confirmation steps at single-spectrum evidence; inspection of every
    candidate alike, an acquisition step or a comparison at qualified; a
    repeat or acquisition step for an inconclusive or untestable record; none
    for no indication and for a generated signal without noise. Never a
    replacement.

    Args:
        summary: A check summary (``check_all_bearing_faults`` or
            ``check_frequency_set``) with its ``claim``.

    Returns:
        The actions, in a fixed order; empty when the claim asks for none.
    """
    claim = summary["claim"]
    conditions = claim["conditions"]
    bearing_id = summary.get("bearing_id")
    prefix = (
        _S["action_prefix"].format(bearing_id=bearing_id)
        if bearing_id
        else _S["action_prefix_unnamed"]
    )
    candidates = summary["candidates"]

    if claim["level"] == SINGLE_SPECTRUM_EVIDENCE:
        (candidate,) = candidates
        names = _NAMES.get(candidate["fault_type_canonical"] or "")
        if names:
            return [_S["action_evidence"].format(prefix=prefix, element=names.element)]
        return [
            _S["action_evidence_label"].format(
                prefix=prefix, label=candidate["fault_type"]
            )
        ]
    if SEVERAL_TYPES in conditions:
        elements = [_NAMES.get(c["fault_type_canonical"] or "") for c in candidates]
        if all(elements):
            parts = [f"the {names.element}" for names in elements if names]
            return [
                _S["action_several_types"].format(prefix=prefix, elements=_join(parts))
            ]
        labels = _join([c["fault_type"] for c in candidates])
        return [_S["action_several_types_label"].format(prefix=prefix, labels=labels)]

    actions = []
    if CAGE_COMB_NOT_EXCLUDED in conditions:
        cage = next(
            c for c in summary["fault_checks"] if c["fault_type_canonical"] == _CAGE
        )
        key = (
            "action_cage_comb_too_short"
            if cage["state_reason"] in _TOO_SHORT
            else "action_cage_comb_untested"
        )
        actions.append(_S[key].format(prefix=prefix))
    if INNER_RACE_SIDEBAND_NOT_EXCLUDED in conditions:
        actions.append(_S["action_inner_race_sideband"].format(prefix=prefix))
    if ROLLING_ELEMENT_HELD_BACK in conditions or INCONCLUSIVE in conditions:
        actions.append(_S["action_repeat"].format(prefix=prefix))
    if NOTHING_TESTABLE in conditions and summary.get("next_step"):
        step = summary["next_step"]
        actions.append(
            _S["action_nothing_testable"].format(
                prefix=prefix, step=step[0].lower() + step[1:]
            )
        )
    return actions


def aggregate_statement(strength: str) -> str:
    """The diagnosis's sentence on its combined evidence strength."""
    return _S["aggregate"].format(strength=strength)


#: The badge of each per-line state (unconfirmed and shared lines stand
#: above the floor but support no type).
_BADGE_KEYS = {
    "supported": "badge_supported",
    "unconfirmed": "badge_not_counted",
    "shared": "badge_not_counted",
    "absent": "badge_absent",
    "not_testable": "badge_not_testable",
}

#: How a counted corroborating line is named, per kind.
_MEMBER_KEYS = {
    ("harmonic", None): "member_harmonic",
    ("sideband", _INNER_RACE): "member_shaft_sideband",
    ("sideband", _ROLLING_ELEMENT): "member_cage_sideband",
    ("subharmonic", _ROLLING_ELEMENT): "member_subharmonic",
}


def emphasised_candidates(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The candidates a report may mark, in the check's level order.

    Every candidate alike at single-spectrum evidence and at a qualified
    claim; none when the claim is withheld (a generated signal without noise
    keeps its candidates in the evidence fields, but no report marks them).

    Args:
        summary: A check summary with its ``claim`` and ``candidates``.

    Returns:
        Copies of the candidates to mark.
    """
    if summary["claim"]["level"] == WITHHELD:
        return []
    return [dict(c) for c in summary["candidates"]]


def line_display_name(label: str, canonical: Optional[str]) -> str:
    """The short name of the line a check tests: '2 x BSF' for the rolling
    element, whose line is tested at twice the declared ball spin frequency."""
    return f"2 x {label}" if canonical == _ROLLING_ELEMENT else label


def state_badge(state: str, claim: Mapping[str, Any]) -> str:
    """The badge of one tested line, from its state and the claim.

    A claim withheld because the signal is generated without noise makes
    every badge "not assessed": the line test's model does not apply.
    """
    if GENERATED_WITHOUT_NOISE in claim["conditions"]:
        return _S["badge_not_assessed"]
    return _S[_BADGE_KEYS[state]]


def _check_of(summary: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return next(c for c in summary["fault_checks"] if c["fault_type"] == label)


def line_test_statement(summary: Mapping[str, Any]) -> str:
    """What the line test found on each tested line, for a report section.

    One sentence per hypothesis from its state; for a generated signal
    without noise, the claim's withheld statement instead.

    Args:
        summary: A check summary with ``shaft_frequency_hz``, ``fault_checks``
            and ``claim``.

    Returns:
        The section statement.
    """
    bearing_id = summary.get("bearing_id")
    shaft_hz = summary["shaft_frequency_hz"]
    sentences = [
        (
            _S["block_intro"].format(bearing_id=bearing_id, shaft_hz=shaft_hz)
            if bearing_id
            else _S["block_intro_unnamed"].format(shaft_hz=shaft_hz)
        )
    ]
    if GENERATED_WITHOUT_NOISE in summary["claim"]["conditions"]:
        return " ".join([*sentences, summary["overall_assessment"]])
    for check in summary["fault_checks"]:
        canonical = check.get("fault_type_canonical")
        line = _line_name(canonical, check["fault_type"])
        state = check["state"]
        if state == "supported":
            key = "block_supported" if canonical in _NAMES else "block_supported_label"
        elif state in ("unconfirmed", "shared"):
            key = "block_not_counted"
        else:
            key = "block_absent" if state == "absent" else "block_not_testable"
        sentences.append(
            _S[key].format(
                line=line, peak_hz=check["line_peak_hz"], level_db=check["level_db"]
            )
        )
    return " ".join(sentences)


def _member_phrase(member: Mapping[str, Any], canonical: Optional[str]) -> str:
    kind = member["kind"]
    key = _MEMBER_KEYS[(kind, None if kind == "harmonic" else canonical)]
    return _S[key].format(
        harmonic=member["harmonic"],
        peak_hz=member["peak_hz"],
        level_db=member["level_db"],
    )


def counted_lines_statements(summary: Mapping[str, Any]) -> list[str]:
    """One sentence per marked candidate: its line and its counted lines.

    Each sentence claims only that the lines stand above the local noise
    floor at the evidence policy's nominal false-alarm levels, and that
    lines of one spectrum are not independent evidence; an inner-race line
    corroborated by sidebands alone carries the sideband limitation.

    Args:
        summary: A check summary with ``evidence_policy``.

    Returns:
        The sentences, in the candidates' level order; none at withheld.
    """
    policy = summary["evidence_policy"]
    sentences = []
    for candidate in emphasised_candidates(summary):
        check = _check_of(summary, candidate["fault_type"])
        canonical = candidate["fault_type_canonical"]
        members = [
            _member_phrase(m, canonical) for m in check["corroboration"] if m["counted"]
        ]
        sentence = _S["evidence_lines"].format(
            line=_line_name(canonical, candidate["fault_type"]),
            peak_hz=check["line_peak_hz"],
            level_db=check["level_db"],
            members=_join(members),
            check_pct=100.0 * policy["false_alarm_probability_per_check"],
            line_pct=100.0 * policy["false_alarm_probability_per_corroborating_line"],
        )
        if _sideband_only(canonical, check):
            sentence += " " + _S["limitation_sideband_only"]
        sentences.append(sentence)
    return sentences


def _defect_phrase(label: str, canonical: Optional[str]) -> str:
    names = _NAMES.get(canonical or "")
    return names.defect if names else _S["periodic_line"].format(label=label)


def evidence_disagreement_statement(summary: Mapping[str, Any], zone: str) -> str:
    """The disagreement of single-spectrum evidence with an acceptable zone.

    Args:
        summary: A check summary whose claim is single-spectrum evidence.
        zone: The acceptable ISO zone ("A" or "B").

    Returns:
        The statement: the envelope evidence warrants confirmation despite
        the acceptable zone, and is not a confirmed fault.
    """
    (candidate,) = summary["candidates"]
    return _S["disagreement_evidence"].format(
        zone=zone,
        defect=_defect_phrase(
            candidate["fault_type"], candidate["fault_type_canonical"]
        ),
    )


def envelope_delta_statement(
    check: Mapping[str, Any],
    *,
    delta_db: float,
    direction: str,
    baseline_peak_hz: float,
) -> str:
    """The baseline comparison of a candidate's envelope amplitude.

    Args:
        check: The candidate's per-check dict in the signal's check.
        delta_db: Amplitude change at the tested line, in dB.
        direction: "higher", "lower" or "unchanged".
        baseline_peak_hz: The strongest bin of the same window in the baseline.

    Returns:
        The delta sentence.
    """
    line = _line_name(check.get("fault_type_canonical"), check["fault_type"])
    if direction == "unchanged":
        return _S["delta_envelope_unchanged"].format(line=line)
    return _S["delta_envelope"].format(
        line=line,
        delta_db=abs(delta_db),
        direction=direction,
        peak_hz=check["line_peak_hz"],
        baseline_peak_hz=baseline_peak_hz,
    )


def envelope_labels(summary: Mapping[str, Any]) -> list[str]:
    """One envelope-report label per marked candidate line.

    A check run without a shaft speed (``shaft_frequency_hz`` None) ran no
    sideband or shaft-order test, and every label says so.

    Args:
        summary: A check summary (``check_frequency_set``).

    Returns:
        The labels, in the candidates' level order; none at withheld.
    """
    key = (
        "envelope_label_no_speed"
        if summary.get("shaft_frequency_hz") is None
        else "envelope_label"
    )
    return [
        _S[key].format(
            name=line_display_name(c["fault_type"], c["fault_type_canonical"]),
            peak_hz=c["line_peak_hz"],
            level_db=c["level_db"],
        )
        for c in emphasised_candidates(summary)
    ]
