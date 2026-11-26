"""ISO 13374 Block 6 — server-authored advisory.

Every evaluative sentence a report shows a human is written *here*, by the
server that computed the numbers, from the closed lists of
:mod:`.statements` and :mod:`..diagnostics.bearing_claims`. Renderers place
these strings; they never compose their own. The evidence-strength grade of one
acquisition is scored here too (:func:`score_evidence_strength`), for the
diagnosis pipeline and for this payload alike.

This exists because the previous arrangement let the caller author report
content (see ``generate_diagnostic_report_docx``'s ``sections`` argument).
When the caller is a language model, the numbers survive the trip and the
labels do not: a superseded standard name, a machine-class vocabulary this
codebase does not use, and a "confidence" grade that
:mod:`..decision_support.recommendations` deliberately refuses to produce.

Three rules hold throughout:

1. Standard names, zone descriptions, and threshold provenance are read from
   :mod:`..diagnostics.iso20816` — the single source of that vocabulary — and
   never re-typed here.
2. No field, value, or sentence is a confidence or a probability. Evidential
   weight is conveyed by naming the facts that support a finding, and the
   categorical ``evidence_strength`` always travels with a sentence saying
   what it counts.
3. A missing input produces an authored statement about the absence and the
   conclusion it makes unavailable. It never produces a missing section — a
   silently absent block reads as "nothing to report".

Every bearing sentence, badge and action follows the bearing check's claim
(:mod:`..diagnostics.bearing_claims`) and the central matcher's per-line
states, from the same closed list of sentences. The per-check fields derived
from those states (``detected``, ``evidence_strength``,
``detected_frequency_hz``, ``magnitude``, ``deviation_pct``,
``harmonics_detected``) drive nothing here, and no inferred bearing fault type
is passed to the recommendation table.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..diagnostics.bearing_claims import (
    CLAIM_SENTENCES,
    QUALIFIED,
    SINGLE_SPECTRUM_EVIDENCE,
    WITHHELD,
    claim_actions,
    counted_lines_statements,
    counts_as_finding,
    emphasised_candidates,
    envelope_delta_statement,
    evidence_disagreement_statement,
    line_test_statement,
    named_candidate,
    state_badge,
)
from .recommendations import generate_recommendations
from .statements import STATEMENTS

#: Block status vocabulary. ``ASSESSED`` means the block reached a verdict;
#: ``REFUSED`` means the engine declined to answer and said why; ``ABSENT``
#: means an input the block needs was never supplied.
ASSESSED = "assessed"
REFUSED = "refused"
ABSENT = "absent"

#: Always rendered next to ``evidence_strength``. The categorical value on its
#: own invites a reader, or a model, to hear a probability; this sentence is
#: what stops that reading.
EVIDENCE_STRENGTH_EXPLANATION = CLAIM_SENTENCES["evidence_strength_explanation"]

#: The description of a bearing action, per claim level.
_ACTION_DESCRIPTION = {
    SINGLE_SPECTRUM_EVIDENCE: CLAIM_SENTENCES["action_description_evidence"],
    QUALIFIED: CLAIM_SENTENCES["action_description_qualified"],
    WITHHELD: CLAIM_SENTENCES["action_description_withheld"],
}

#: Urgency of a bearing action: a confirmation, inspection, repeat or
#: acquisition step, never a maintenance action on the bearing.
_BEARING_ACTION_URGENCY = "medium"

#: Keys a ``diagnose_vibration`` result must carry. Guessing at a partial
#: result would produce authored sentences about numbers that were never
#: computed, which is the failure mode this module exists to prevent.
_REQUIRED_KEYS = ("signal_id", "iso_severity", "fft_summary", "stft_summary")

#: ISO zones that are no evidence of a problem: no evidence points and no zone
#: action in the diagnosis, no elevated-zone verdict, and the only zones an
#: indicator disagreement is named against.
ACCEPTABLE_ZONES = ("A", "B")

#: Which baseline delta leads the comparison sentence, most specific first.
#: Chosen by what the movement MEANS, not by magnitude: the three deltas are
#: measured in dB, mm/s and percentage points, so "the biggest number" would
#: rank them by unit scale rather than by diagnostic weight — the anomaly
#: ratio, living on a 0-100 scale, would win almost every time.
_HEADLINE_ORDER = ("envelope_magnitude", "rms_velocity", "anomaly_ratio")

#: Anomaly-model verdicts that flag a departure from the learned pattern.
_ANOMALY_FLAGS = ("Faulty", "Suspicious")

#: Points each finding of one acquisition adds to its evidence strength.
#: Elevated broadband vibration is direct physical evidence of a problem but is
#: not corroborated on its own; a bearing claim counts once; the anomaly model
#: counts its own verdict. Changing a weight changes a statistical decision
#: rule, not a wording.
EVIDENCE_POINTS = {
    "iso_zone_c": 1.0,
    "iso_zone_d": 2.0,
    "bearing_finding": 1.0,
    "anomaly_faulty": 1.0,
    "anomaly_suspicious": 0.5,
}

#: The grades above "none", each with the points it stays below; at or above
#: the last bound the grade is "strong". Severity alone (zone D, 2 points)
#: therefore stays below "strong".
EVIDENCE_GRADE_BOUNDS = (("weak", 2.0), ("moderate", 3.0))


# ---------------------------------------------------------------------------
# ISO severity
# ---------------------------------------------------------------------------


def _build_iso_block(iso: dict) -> dict:
    """Author the ISO severity block, or carry the engine's refusal forward."""
    if iso.get("status") == "refused":
        return {
            "status": REFUSED,
            "statement": STATEMENTS["report_iso_refused"].format(reason=iso["reason"]),
            "reason": iso["reason"],
            "remedy": iso["remedy"],
            "standard_note": None,
        }

    zone = iso["zone"]
    rms = iso["rms_velocity_mm_s"]
    severity = iso.get("severity_level", "")
    description = iso.get("zone_description", "")
    boundaries = iso.get("boundaries", {})

    statement = (
        STATEMENTS["report_iso_assessed"]
        .format(
            rms_velocity=rms,
            zone=zone,
            severity=severity,
            machine_group=iso.get("machine_group"),
            support_type=iso.get("support_type"),
            description=description,
        )
        .strip()
    )

    return {
        "status": ASSESSED,
        "statement": statement,
        "zone": zone,
        "severity_level": severity,
        "rms_velocity_mm_s": rms,
        "boundaries": boundaries,
        "evaluation_band": iso.get("frequency_range"),
        # Verbatim from iso20816 — the caveat that the 2022 edition merges
        # zones A and B travels with every verdict or it travels with none.
        "standard_note": iso.get("threshold_provenance"),
    }


# ---------------------------------------------------------------------------
# Bearing characteristic-frequency matching
# ---------------------------------------------------------------------------


def _build_bearing_block(
    bearing_faults: Optional[dict],
    bearing_check: Optional[dict],
    bearing_id: Optional[str],
) -> dict:
    """Author the bearing line-test block, or state why it was not attempted.

    A designation that was supplied but could not be checked (the pipeline's
    ``bearing_check`` is ``not_computed``) is a refusal with the engine's
    reason and remedy, never the "no designation supplied" absence. An
    assessed block states each tested line by its state and carries one row
    per line with its closed-list badge.
    """
    if bearing_check and bearing_check.get("status") == "not_computed":
        return {
            "status": REFUSED,
            "statement": STATEMENTS["report_bearing_not_computed"].format(
                reason=bearing_check["reason"]
            ),
            "reason": bearing_check["reason"],
            "remedy": bearing_check["remedy"],
            "bearing_id": bearing_id,
            "rows": [],
        }

    if not bearing_faults:
        return {
            "status": ABSENT,
            "statement": STATEMENTS["report_bearing_absent"],
            "remedy": STATEMENTS["report_bearing_absent_remedy"],
            "rows": [],
        }

    claim = bearing_faults["claim"]
    marked = {c["fault_type"] for c in emphasised_candidates(bearing_faults)}
    rows = []
    for check in bearing_faults["fault_checks"]:
        row = {
            "fault_type": check["fault_type"],
            "fault_canonical": check.get("fault_type_canonical"),
            "expected_hz": check["expected_frequency_hz"],
            "line_hz": check["line_frequency_hz"],
            "peak_hz": check.get("line_peak_hz"),
            "amplitude": check.get("line_amplitude"),
            "level_db": check.get("level_db"),
            "counted_lines": sum(
                1 for m in check.get("corroboration") or [] if m["counted"]
            ),
            "state": check["state"],
            "emphasised": check["fault_type"] in marked,
            "badge": state_badge(check["state"], claim),
        }
        # An aggregate rule can support a line with no line counted on its own.
        aggregate = check.get("aggregate_corroboration")
        if aggregate is not None:
            row["aggregate_basis"] = aggregate["basis"]
        rows.append(row)

    return {
        "status": ASSESSED,
        "statement": line_test_statement(bearing_faults),
        "bearing_id": bearing_faults.get("bearing_id"),
        "shaft_frequency_hz": bearing_faults.get("shaft_frequency_hz"),
        "bearing_frequencies": bearing_faults.get("bearing_frequencies", {}),
        "claim_level": claim["level"],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def _build_anomaly_block(anomaly: Optional[dict]) -> dict:
    """Author the anomaly block, or state that no model verdict exists."""
    if not anomaly:
        return {
            "status": ABSENT,
            "statement": STATEMENTS["report_anomaly_absent"],
            "remedy": STATEMENTS["report_anomaly_absent_remedy"],
        }

    health = anomaly["overall_health"]
    ratio = anomaly["anomaly_ratio"]
    segments = anomaly.get("num_segments")
    counted = (
        STATEMENTS["report_anomaly_segments_counted"].format(
            count=round(ratio * segments), segments=segments
        )
        if segments
        else STATEMENTS["report_anomaly_segments_share"].format(ratio_pct=ratio * 100)
    )
    return {
        "status": ASSESSED,
        "statement": STATEMENTS["report_anomaly"].format(
            health=health, counted=counted, ratio_pct=ratio * 100
        ),
        "overall_health": health,
        "anomaly_ratio": ratio,
        "num_segments": segments,
    }


# ---------------------------------------------------------------------------
# Spectral energy distribution
# ---------------------------------------------------------------------------


def _build_energy_block(stft_summary: dict) -> dict:
    """Author the spectral-energy block from the STFT band breakdown."""
    # compute_stft_spectrogram emits a list of {"band", "energy"} entries.
    entries = stft_summary.get("energy_per_band") or []
    bands = {entry["band"]: entry["energy"] for entry in entries}
    if not bands:
        return {
            "status": ABSENT,
            "statement": STATEMENTS["report_energy_absent"],
            "bands": {},
        }

    total = sum(bands.values())
    dominant = max(bands, key=lambda k: bands[k])
    share = (bands[dominant] / total * 100) if total else 0.0
    return {
        "status": ASSESSED,
        # Which band holds the most energy is a measurement: shaft orders,
        # resonances, impacts, gear or bearing tones and electrical sources
        # can each dominate a band, so the statement names no source. A unit
        # change (g to m/s2) scales every band alike; a different measured
        # quantity (acceleration versus velocity) can reorder them.
        "statement": STATEMENTS["report_energy"].format(band=dominant, share=share),
        "bands": dict(bands),
        "dominant_band": dominant,
        "dominant_share_pct": round(share, 1),
    }


# ---------------------------------------------------------------------------
# Evidence strength
# ---------------------------------------------------------------------------


def score_evidence_strength(
    iso_severity: dict,
    bearing_faults: Optional[dict],
    anomaly: Optional[dict],
) -> str:
    """Grade the evidence of one acquisition: "none", "weak", "moderate" or "strong".

    The grade counts the findings of the analyses that ran, with the points of
    ``EVIDENCE_POINTS``:

    - ISO severity: zone C and zone D add their points; zones A and B add
      none. A zone outside A to C counts as zone D. A refused ISO block
      (undeclared unit, insufficient sampling rate) adds nothing: no verdict,
      no points.
    - Bearing claim (:mod:`..diagnostics.bearing_claims`): one finding when
      ``counts_as_finding`` holds for the claim (single-spectrum evidence, or
      qualified from an indicated or several-types outcome); a withheld claim
      and one held back as inconclusive add nothing. A bearing check that was
      not computed adds nothing.
    - Anomaly model: a "Faulty" or "Suspicious" verdict adds its points.
    - A strongest spectral line at 1x or 2x shaft speed adds nothing: a
      shaft-rate line is expected on a rotating machine and identifies no
      fault.

    Zero points grade "none"; otherwise the first grade of
    ``EVIDENCE_GRADE_BOUNDS`` whose bound the points stay below, else
    "strong". A quiet machine with no findings therefore grades "none", and
    severity alone never grades "strong". The grade is never a probability and
    never names or confirms a fault type.

    Args:
        iso_severity: The pipeline's ISO block, assessed (with ``zone``) or
            refused (``status: "refused"``).
        bearing_faults: The bearing check result with its ``claim``, or None
            when no check ran.
        anomaly: The anomaly model's result with ``overall_health``, or None
            when no verdict exists.

    Returns:
        The categorical grade.
    """
    points = 0.0
    if iso_severity.get("status") != "refused":
        zone = iso_severity["zone"]
        if zone == "C":
            points += EVIDENCE_POINTS["iso_zone_c"]
        elif zone not in ACCEPTABLE_ZONES:
            points += EVIDENCE_POINTS["iso_zone_d"]
    if bearing_faults and counts_as_finding(bearing_faults["claim"]):
        points += EVIDENCE_POINTS["bearing_finding"]
    if anomaly:
        health = anomaly["overall_health"]
        if health == "Faulty":
            points += EVIDENCE_POINTS["anomaly_faulty"]
        elif health == "Suspicious":
            points += EVIDENCE_POINTS["anomaly_suspicious"]

    if points == 0.0:
        return "none"
    for grade, bound in EVIDENCE_GRADE_BOUNDS:
        if points < bound:
            return grade
    return "strong"


# ---------------------------------------------------------------------------
# Verdict and evidence
# ---------------------------------------------------------------------------


def _anomaly_flagged(anomaly_block: dict) -> bool:
    return anomaly_block.get("overall_health") in _ANOMALY_FLAGS


def _build_verdict(
    bearing_faults: Optional[dict],
    bearing_block: dict,
    iso_block: dict,
    anomaly_block: dict,
) -> dict:
    """Author the headline verdict from the blocks that reached one.

    For an assessed bearing check: the claim's statement and its actions, one
    branch per claim level. A type is named (``fault_canonical``) only at
    single-spectrum evidence; at several types every candidate is named
    alike; a withheld claim keeps bearing and non-bearing sources open when
    the zone or the anomaly model points to a problem.
    """
    if bearing_block["status"] == ASSESSED and bearing_faults:
        claim = bearing_faults["claim"]
        statement = " ".join(
            [bearing_faults["overall_assessment"], *claim_actions(bearing_faults)]
        )
        named = named_candidate(claim, bearing_faults)
        if named:
            return {
                "statement": statement,
                "fault_canonical": named["fault_type_canonical"],
                "fault_acronym": named["fault_type"],
            }
        if claim["level"] == WITHHELD:
            if (
                iso_block["status"] == ASSESSED
                and iso_block["zone"] not in ACCEPTABLE_ZONES
            ):
                statement += " " + CLAIM_SENTENCES["verdict_zone_elevated"].format(
                    zone=iso_block["zone"]
                )
            elif _anomaly_flagged(anomaly_block):
                statement += " " + CLAIM_SENTENCES["verdict_anomaly_flag"]
        return {"statement": statement, "fault_canonical": None, "fault_acronym": ""}

    not_computed = bearing_block["status"] == REFUSED
    if not_computed:
        bearing_id = bearing_block.get("bearing_id")
        headline = (
            STATEMENTS["report_verdict_not_computed_for_bearing"].format(
                bearing_id=bearing_id
            )
            if bearing_id
            else STATEMENTS["report_verdict_not_computed"]
        )
    else:
        headline = STATEMENTS["report_verdict_not_attempted"]

    # A bearing named but never checked cannot be ruled out as the source.
    unchecked = STATEMENTS["report_verdict_tail_unchecked"]
    if iso_block["status"] == ASSESSED and iso_block["zone"] not in ACCEPTABLE_ZONES:
        tail = (
            unchecked if not_computed else STATEMENTS["report_verdict_tail_non_bearing"]
        )
        headline += STATEMENTS["report_verdict_zone_elevated"].format(
            zone=iso_block["zone"], tail=tail
        )
    elif _anomaly_flagged(anomaly_block):
        tail = (
            unchecked
            if not_computed
            else STATEMENTS["report_verdict_tail_outside_bearing"]
        )
        headline += STATEMENTS["report_verdict_anomaly_flag"].format(tail=tail)

    return {"statement": headline, "fault_canonical": None, "fault_acronym": ""}


def _build_evidence(
    diagnosis: dict,
    bearing_block: dict,
    iso_block: dict,
    anomaly_block: dict,
    energy_block: dict,
) -> dict:
    """Collect the facts that support the verdict, as authored sentences.

    For the bearing: the counted lines of each candidate the claim lets a
    report mark, with their levels, claiming only that they stand above the
    local noise floor at the stated nominal false-alarm levels.
    """
    statements: list[str] = []

    faults = diagnosis.get("bearing_faults")
    if bearing_block["status"] == ASSESSED and faults:
        statements += counted_lines_statements(faults)

    if iso_block["status"] == ASSESSED:
        statements.append(iso_block["statement"])
    if anomaly_block["status"] == ASSESSED:
        statements.append(anomaly_block["statement"])
    if energy_block["status"] == ASSESSED:
        statements.append(energy_block["statement"])

    peak = diagnosis.get("fft_summary", {}).get("peak_frequency_hz")
    if peak is not None:
        statements.append(STATEMENTS["report_dominant_frequency"].format(peak_hz=peak))

    return {
        "strength": diagnosis.get("evidence_strength", "none"),
        "strength_explanation": EVIDENCE_STRENGTH_EXPLANATION,
        "statements": statements,
    }


# ---------------------------------------------------------------------------
# Indicator disagreement (R6)
# ---------------------------------------------------------------------------


def _build_disagreements(
    iso_block: dict,
    bearing_faults: Optional[dict],
    bearing_block: dict,
    anomaly_block: dict,
) -> list[dict]:
    """Name the indicators that disagree with an acceptable ISO zone.

    A refused ISO block is not an indicator and cannot disagree with one — an
    absent verdict is silence, not a contradicting opinion. Only
    single-spectrum evidence drives a bearing entry, which asks for
    confirmation; an anomaly-model flag has its own entry, which names no
    bearing type.
    """
    if iso_block["status"] != ASSESSED:
        return []
    zone = iso_block["zone"]
    if zone not in ACCEPTABLE_ZONES:
        return []

    entries: list[dict] = []
    if (
        bearing_block["status"] == ASSESSED
        and bearing_faults
        and bearing_faults["claim"]["level"] == SINGLE_SPECTRUM_EVIDENCE
    ):
        entries.append(
            {
                "statement": evidence_disagreement_statement(bearing_faults, zone),
                "governing_indicator": STATEMENTS["report_indicator_single_spectrum"],
                "deferring_indicator": STATEMENTS["report_indicator_zone"].format(
                    zone=zone
                ),
            }
        )
    if _anomaly_flagged(anomaly_block):
        entries.append(
            {
                "statement": CLAIM_SENTENCES["disagreement_anomaly"].format(
                    zone=zone,
                    health=anomaly_block["overall_health"],
                    ratio_pct=anomaly_block["anomaly_ratio"] * 100,
                ),
                "governing_indicator": STATEMENTS["report_indicator_anomaly_model"],
                "deferring_indicator": STATEMENTS["report_indicator_zone"].format(
                    zone=zone
                ),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Recommendations (R7)
# ---------------------------------------------------------------------------


def _refusal_recommendation(block: dict, description: str) -> dict:
    """The action that lifts a refused block: its remedy, motivated by its reason."""
    return {
        "action": block["remedy"],
        "urgency": "medium",
        "description": description,
        "motivation": block["reason"],
        "evidence": [block["statement"]],
    }


def _build_recommendations(
    iso_block: dict,
    bearing_faults: Optional[dict],
    bearing_block: dict,
    anomaly_block: dict,
) -> list[dict]:
    """Attach a motivation and its supporting evidence to each action.

    The zone actions come from the recommendation table, by the ISO zone
    alone. The bearing actions are the claim's (confirmation, inspection of
    every candidate alike, a repeat or acquisition step); no inferred bearing
    fault type is ever passed to the table.
    """
    recommendations: list[dict] = []

    if iso_block["status"] == REFUSED:
        recommendations.append(
            _refusal_recommendation(
                iso_block, STATEMENTS["report_description_iso_refused"]
            )
        )
    else:
        zone = iso_block["zone"]
        severity = iso_block.get("severity_level")
        motivation = (
            STATEMENTS["report_motivation_zone_severity"].format(
                zone=zone, severity=severity.lower()
            )
            if severity
            else STATEMENTS["report_motivation_zone"].format(zone=zone)
        )
        for entry in generate_recommendations(severity_zone=zone):
            recommendations.append(
                {
                    **entry,
                    "motivation": motivation,
                    "evidence": [iso_block["statement"]],
                }
            )

    if bearing_block["status"] == ASSESSED and bearing_faults:
        description = _ACTION_DESCRIPTION[bearing_faults["claim"]["level"]]
        for action in claim_actions(bearing_faults):
            recommendations.append(
                {
                    "action": action,
                    "urgency": _BEARING_ACTION_URGENCY,
                    "description": description,
                    "motivation": bearing_faults["overall_assessment"],
                    "evidence": [bearing_block["statement"]],
                }
            )

    if bearing_block["status"] == REFUSED:
        recommendations.append(
            _refusal_recommendation(
                bearing_block, STATEMENTS["report_description_bearing_refused"]
            )
        )

    if anomaly_block["status"] == ABSENT:
        recommendations.append(
            {
                "action": anomaly_block["remedy"],
                "urgency": "low",
                "description": STATEMENTS["report_description_anomaly_absent"],
                "motivation": anomaly_block["statement"],
                "evidence": [anomaly_block["statement"]],
            }
        )

    return recommendations


# ---------------------------------------------------------------------------
# Baseline comparison (R8)
# ---------------------------------------------------------------------------


def build_baseline_comparison(
    diagnosis: dict, baseline: Optional[dict]
) -> dict[str, Any]:
    """Author the comparison against a healthy reference signal.

    An absolute figure tells a technician where the machine is; a delta tells
    them where it is going. When no baseline is supplied the block still
    renders — it states the absence and the conclusion that absence costs.
    """
    if not baseline:
        return {
            "status": ABSENT,
            "statement": STATEMENTS["report_baseline_absent"],
            "remedy": STATEMENTS["report_baseline_absent_remedy"],
            "deltas": [],
        }

    incompatible = _baseline_incompatibility(diagnosis, baseline)
    if incompatible:
        return {
            "status": REFUSED,
            "statement": STATEMENTS["report_baseline_refused"].format(
                reason=incompatible
            ),
            "remedy": STATEMENTS["report_baseline_refused_remedy"],
            "deltas": [],
        }

    deltas = [
        delta
        for delta in (
            _rms_delta(diagnosis, baseline),
            _anomaly_delta(diagnosis, baseline),
        )
        if delta is not None
    ] + _envelope_deltas(diagnosis, baseline)

    if not deltas:
        return {
            "status": ABSENT,
            "statement": STATEMENTS["report_baseline_nothing_compared"].format(
                baseline_id=baseline.get("signal_id")
            ),
            "remedy": STATEMENTS["report_baseline_nothing_compared_remedy"],
            "deltas": [],
        }

    moved = [d for d in deltas if d["direction"] != "unchanged"]
    if not moved:
        statement = STATEMENTS["report_baseline_unchanged"].format(
            baseline_id=baseline.get("signal_id")
        )
    else:
        lead = min(_HEADLINE_ORDER.index(d["indicator"]) for d in moved)
        headline = " ".join(
            d["statement"]
            for d in moved
            if _HEADLINE_ORDER.index(d["indicator"]) == lead
        )
        statement = STATEMENTS["report_baseline_moved"].format(
            baseline_id=baseline.get("signal_id"),
            moved=len(moved),
            total=len(deltas),
            headline=headline,
        )

    return {
        "status": ASSESSED,
        "statement": statement,
        "baseline_signal_id": baseline.get("signal_id"),
        "remedy": "",
        "deltas": deltas,
    }


def _baseline_incompatibility(diagnosis: dict, baseline: dict) -> str:
    """Return a reason the two signals cannot be compared, or an empty string."""
    signal_iso = diagnosis.get("iso_severity", {})
    baseline_iso = baseline.get("iso_severity", {})
    if (
        signal_iso.get("status") == "assessed"
        and baseline_iso.get("status") == "assessed"
    ):
        signal_unit = signal_iso.get("original_unit")
        baseline_unit = baseline_iso.get("original_unit")
        if signal_unit != baseline_unit:
            return STATEMENTS["report_baseline_unit_mismatch"].format(
                signal_unit=signal_unit, baseline_unit=baseline_unit
            )

    signal_bearing = diagnosis.get("bearing_id")
    baseline_bearing = baseline.get("bearing_id")
    if signal_bearing and baseline_bearing and signal_bearing != baseline_bearing:
        return STATEMENTS["report_baseline_bearing_mismatch"].format(
            signal_bearing=signal_bearing, baseline_bearing=baseline_bearing
        )

    return ""


def _direction(delta: float, tolerance: float) -> str:
    if abs(delta) < tolerance:
        return "unchanged"
    return "higher" if delta > 0 else "lower"


def _rms_delta(diagnosis: dict, baseline: dict) -> Optional[dict]:
    signal_iso = diagnosis.get("iso_severity", {})
    baseline_iso = baseline.get("iso_severity", {})
    if (
        signal_iso.get("status") != "assessed"
        or baseline_iso.get("status") != "assessed"
    ):
        return None

    now = signal_iso["rms_velocity_mm_s"]
    then = baseline_iso["rms_velocity_mm_s"]
    delta = now - then
    direction = _direction(delta, 0.005)
    if direction == "unchanged":
        statement = STATEMENTS["report_delta_rms_unchanged"].format(now=now, then=then)
    else:
        statement = STATEMENTS["report_delta_rms_moved"].format(
            delta=abs(delta), direction=direction, now=now, then=then
        )
    return {
        "indicator": "rms_velocity",
        "unit": "mm/s",
        "value": now,
        "baseline_value": then,
        "delta": round(delta, 4),
        "direction": direction,
        "statement": statement,
    }


def _anomaly_delta(diagnosis: dict, baseline: dict) -> Optional[dict]:
    signal_anomaly = diagnosis.get("anomaly_detection")
    baseline_anomaly = baseline.get("anomaly_detection")
    if not signal_anomaly or not baseline_anomaly:
        return None

    now = signal_anomaly["anomaly_ratio"] * 100
    then = baseline_anomaly["anomaly_ratio"] * 100
    delta = now - then
    direction = _direction(delta, 0.05)
    if direction == "unchanged":
        statement = STATEMENTS["report_delta_anomaly_unchanged"].format(
            now=now, then=then
        )
    else:
        statement = STATEMENTS["report_delta_anomaly_moved"].format(
            delta=abs(delta), direction=direction, now=now, then=then
        )
    return {
        "indicator": "anomaly_ratio",
        "unit": "percentage points",
        "value": round(now, 2),
        "baseline_value": round(then, 2),
        "delta": round(delta, 2),
        "direction": direction,
        "statement": statement,
    }


def _envelope_deltas(diagnosis: dict, baseline: dict) -> list[dict]:
    """Compare envelope amplitude at each candidate's tested line.

    One delta per candidate the claim lets a report mark, all alike, each at
    the line its check tested: the strongest bin of the search window
    (``line_amplitude``) in the signal against the same window in the
    baseline, which reports it for every testable line, significant or not.
    """
    signal_faults = diagnosis.get("bearing_faults")
    baseline_faults = baseline.get("bearing_faults")
    if not signal_faults or not baseline_faults:
        return []

    deltas = []
    for candidate in emphasised_candidates(signal_faults):
        label = candidate["fault_type"]
        check = next(
            c for c in signal_faults["fault_checks"] if c["fault_type"] == label
        )
        reference = next(
            (
                c
                for c in baseline_faults["fault_checks"]
                if c["fault_type"] == label and c.get("line_amplitude")
            ),
            None,
        )
        if not check.get("line_amplitude") or reference is None:
            continue
        now = check["line_amplitude"]
        then = reference["line_amplitude"]
        delta_db = 20 * math.log10(now / then)
        direction = _direction(delta_db, 0.05)
        deltas.append(
            {
                "indicator": "envelope_magnitude",
                "fault_type": label,
                "unit": "dB",
                "value": now,
                "baseline_value": then,
                "delta": round(delta_db, 2),
                "direction": direction,
                "statement": envelope_delta_statement(
                    check,
                    delta_db=delta_db,
                    direction=direction,
                    baseline_peak_hz=reference["line_peak_hz"],
                ),
            }
        )
    return deltas


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _build_provenance(diagnosis: dict) -> dict:
    """Record what was analysed and by which server version."""
    # Imported inside the function: the package __init__ pulls in the MCP
    # server, and this module is imported from within that import chain.
    from .. import __version__

    return {
        "signal_id": diagnosis.get("signal_id"),
        "rpm": diagnosis.get("rpm"),
        "bearing_id": diagnosis.get("bearing_id"),
        "machine_group": diagnosis.get("machine_group"),
        "support_type": diagnosis.get("support_type"),
        "server_version": __version__,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_advisory(
    diagnosis: dict,
    baseline_diagnosis: Optional[dict] = None,
) -> dict[str, Any]:
    """Turn a ``diagnose_vibration`` result into server-authored statements.

    Every string in the returned payload is written here or by a module this
    one reads from. Renderers place these strings; a renderer that composes
    its own evaluative sentence has broken the contract this function exists
    to hold.

    Args:
        diagnosis: A result from
            :func:`..decision_support.diagnosis_pipeline.diagnose_vibration`.
        baseline_diagnosis: Optional result for a healthy reference signal on
            the same machine. When omitted, the comparison block states the
            absence and the conclusion it makes unavailable.

    Returns:
        The advisory payload: verdict, evidence, per-indicator blocks,
        indicator disagreements, baseline comparison, recommendations, and
        provenance.

    Raises:
        ValueError: If ``diagnosis`` is not a recognisable diagnosis result.
            Authoring sentences about numbers that were never computed is the
            failure this module exists to prevent, so a partial input is
            refused rather than filled in.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in diagnosis]
    if missing:
        raise ValueError(
            f"Not a diagnose_vibration result — missing {missing}. Pass the "
            f"dict returned by diagnose_vibration(); the advisory layer does "
            f"not re-run analysis and cannot author statements about values "
            f"it was not given."
        )

    bearing_faults = diagnosis.get("bearing_faults")
    iso_block = _build_iso_block(diagnosis["iso_severity"])
    bearing_block = _build_bearing_block(
        bearing_faults,
        diagnosis.get("bearing_check"),
        diagnosis.get("bearing_id"),
    )
    anomaly_block = _build_anomaly_block(diagnosis.get("anomaly_detection"))
    energy_block = _build_energy_block(diagnosis["stft_summary"])

    verdict = _build_verdict(bearing_faults, bearing_block, iso_block, anomaly_block)
    evidence = _build_evidence(
        diagnosis, bearing_block, iso_block, anomaly_block, energy_block
    )
    disagreements = _build_disagreements(
        iso_block, bearing_faults, bearing_block, anomaly_block
    )
    baseline_block = build_baseline_comparison(diagnosis, baseline_diagnosis)
    recommendations = _build_recommendations(
        iso_block, bearing_faults, bearing_block, anomaly_block
    )

    return {
        "signal_id": diagnosis["signal_id"],
        "verdict": verdict,
        "evidence": evidence,
        "iso": iso_block,
        "bearing_match": bearing_block,
        "anomaly": anomaly_block,
        "spectral_energy": energy_block,
        "disagreements": disagreements,
        "baseline_comparison": baseline_block,
        "recommendations": recommendations,
        "provenance": _build_provenance(diagnosis),
    }


def collect_statements(
    advisory: dict, figure: Optional[dict[str, Any]] = None
) -> list[str]:
    """Return every authored statement in the payload, in document order.

    This is the surface the cross-rendering parity test asserts on: a
    statement present here and absent from a rendering means that rendering
    dropped something the server said. The badges of the bearing line-test
    table are included, and so is the caption of ``figure`` (the report's
    annotated envelope spectrum) when it is given.

    Args:
        advisory: Payload from :func:`build_advisory`.
        figure: Optional description from
            :func:`..figures.build_annotated_envelope_figure`.

    Returns:
        The statements, each once.
    """
    statements: list[str] = []

    def add(value: Optional[str]) -> None:
        if value and value not in statements:
            statements.append(value)

    add(advisory["verdict"]["statement"])
    for block_key in ("iso", "bearing_match", "anomaly", "spectral_energy"):
        add(advisory[block_key].get("statement"))
        add(advisory[block_key].get("remedy"))
    for row in advisory["bearing_match"].get("rows", []):
        add(row["badge"])
    for entry in advisory["disagreements"]:
        add(entry["statement"])
    add(advisory["baseline_comparison"].get("statement"))
    for sentence in advisory["baseline_comparison"].get("deltas", []):
        add(sentence.get("statement"))
    for sentence in advisory["evidence"]["statements"]:
        add(sentence)
    add(advisory["evidence"]["strength_explanation"])
    for rec in advisory["recommendations"]:
        add(rec["action"])
        add(rec["description"])
        add(rec["motivation"])
    if figure is not None:
        add(figure.get("caption"))

    return statements
