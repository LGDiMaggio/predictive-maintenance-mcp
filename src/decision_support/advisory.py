"""ISO 13374 Block 6 — server-authored advisory.

Every evaluative sentence a report shows a human is written *here*, by the
server that computed the numbers. Renderers place these strings; they never
compose their own.

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
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..diagnostics.bearing_analyzer import FAULT_TYPE_CANONICAL
from .recommendations import generate_recommendations

#: Block status vocabulary. ``ASSESSED`` means the block reached a verdict;
#: ``REFUSED`` means the engine declined to answer and said why; ``ABSENT``
#: means an input the block needs was never supplied.
ASSESSED = "assessed"
REFUSED = "refused"
ABSENT = "absent"

#: Always rendered next to ``evidence_strength``. The categorical value on its
#: own invites a reader — or a model — to hear a probability; this sentence is
#: what stops that reading.
EVIDENCE_STRENGTH_EXPLANATION = (
    "Evidence strength counts independent corroborating findings — it is "
    "not a confidence score and not a probability."
)

#: Keys a ``diagnose_vibration`` result must carry. Guessing at a partial
#: result would produce authored sentences about numbers that were never
#: computed, which is the failure mode this module exists to prevent.
_REQUIRED_KEYS = ("signal_id", "iso_severity", "fft_summary", "stft_summary")

_ACCEPTABLE_ZONES = ("A", "B")


def _fault_label(canonical: Optional[str]) -> str:
    """Render a canonical fault type as prose ('outer_race' -> 'outer race')."""
    return canonical.replace("_", " ") if canonical else ""


def _acronym_for(canonical: Optional[str]) -> str:
    """Reverse-map a canonical fault type to its bearing acronym."""
    for acronym, value in FAULT_TYPE_CANONICAL.items():
        if value == canonical:
            return acronym
    return ""


# ---------------------------------------------------------------------------
# ISO severity
# ---------------------------------------------------------------------------


def _build_iso_block(iso: dict) -> dict:
    """Author the ISO severity block, or carry the engine's refusal forward."""
    if iso.get("status") == "refused":
        return {
            "status": REFUSED,
            "statement": (
                f"ISO severity was not assessed. {iso['reason']}"
            ),
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
        f"ISO 20816-3: RMS velocity {rms:.2f} mm/s places this machine in "
        f"Zone {zone} ({severity}) for machine group "
        f"{iso.get('machine_group')} on a {iso.get('support_type')} support. "
        f"{description}"
    ).strip()

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


def _build_bearing_block(bearing_faults: Optional[dict]) -> dict:
    """Author the frequency-matching block, or state why it was not attempted."""
    if not bearing_faults:
        return {
            "status": ABSENT,
            "statement": (
                "Bearing characteristic-frequency matching was not attempted: "
                "no bearing designation was supplied for this signal, so BPFO, "
                "BPFI, BSF and FTF could not be computed. Without it, a "
                "spectral peak cannot be attributed to a specific bearing "
                "element."
            ),
            "remedy": (
                "Re-run the diagnosis with a bearing_id present in the "
                "catalog, or add the bearing designation to the signal's "
                "companion metadata."
            ),
            "rows": [],
        }

    rows = []
    for check in bearing_faults.get("fault_checks", []):
        rows.append(
            {
                "fault_type": check["fault_type"],
                "fault_canonical": check.get("fault_type_canonical"),
                "expected_hz": check["expected_frequency_hz"],
                "measured_hz": check.get("detected_frequency_hz"),
                "deviation_pct": check.get("deviation_pct"),
                "magnitude": check.get("magnitude"),
                "harmonics": len(check.get("harmonics_detected") or []),
                "matched": bool(check.get("detected")),
                "evidence_strength": check.get("evidence_strength", "none"),
            }
        )

    matched = [r for r in rows if r["matched"]]
    if matched:
        parts = [
            f"{r['fault_type']} expected at {r['expected_hz']:.2f} Hz, "
            f"measured at {r['measured_hz']:.2f} Hz "
            f"({r['deviation_pct']:.2f}% deviation)"
            for r in matched
        ]
        statement = (
            f"Bearing {bearing_faults.get('bearing_id')} at "
            f"{bearing_faults.get('shaft_frequency_hz', 0.0):.1f} Hz shaft "
            f"speed: {'; '.join(parts)}. The remaining characteristic "
            f"frequencies did not match."
        )
    else:
        statement = (
            f"Bearing {bearing_faults.get('bearing_id')}: none of the four "
            f"characteristic frequencies (BPFO, BPFI, BSF, FTF) matched a "
            f"peak in the envelope spectrum within tolerance."
        )

    return {
        "status": ASSESSED,
        "statement": statement,
        "bearing_id": bearing_faults.get("bearing_id"),
        "shaft_frequency_hz": bearing_faults.get("shaft_frequency_hz"),
        "bearing_frequencies": bearing_faults.get("bearing_frequencies", {}),
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
            "statement": (
                "No anomaly-model verdict is available for this signal: no "
                "trained model was found. The pattern-level check that would "
                "corroborate or contradict the severity reading is therefore "
                "missing from this assessment."
            ),
            "remedy": (
                "Train an anomaly model on healthy signals from this machine, "
                "then re-run the diagnosis."
            ),
        }

    health = anomaly["overall_health"]
    ratio = anomaly["anomaly_ratio"]
    segments = anomaly.get("num_segments")
    counted = (
        f"{round(ratio * segments)} of {segments} segments"
        if segments
        else f"{ratio * 100:.1f}% of segments"
    )
    return {
        "status": ASSESSED,
        "statement": (
            f"Anomaly model verdict: {health} — {counted} "
            f"({ratio * 100:.1f}%) fall outside the learned healthy pattern."
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
            "statement": (
                "No spectral energy distribution is available: the STFT band "
                "breakdown was not computed, so the frequency region carrying "
                "the signal's energy cannot be named."
            ),
            "bands": {},
        }

    total = sum(bands.values())
    dominant = max(bands, key=lambda k: bands[k])
    share = (bands[dominant] / total * 100) if total else 0.0
    return {
        "status": ASSESSED,
        "statement": (
            f"Spectral energy is concentrated in the {dominant} band "
            f"({share:.0f}% of total STFT energy). High-frequency dominance is "
            f"consistent with impulsive excitation of structural resonances; "
            f"low-frequency dominance is consistent with shaft-order sources "
            f"such as unbalance or misalignment."
        ),
        "bands": dict(bands),
        "dominant_band": dominant,
        "dominant_share_pct": round(share, 1),
    }


# ---------------------------------------------------------------------------
# Verdict and evidence
# ---------------------------------------------------------------------------


def _build_verdict(
    bearing_block: dict, iso_block: dict, anomaly_block: dict
) -> dict:
    """Author the headline verdict from the blocks that reached one."""
    matched = [r for r in bearing_block.get("rows", []) if r["matched"]]
    if matched:
        primary = max(
            matched,
            key=lambda r: {"high": 3, "moderate": 2, "low": 1}.get(
                r["evidence_strength"], 0
            ),
        )
        canonical = primary["fault_canonical"]
        label = _fault_label(canonical)
        return {
            "statement": (
                f"A {label} fault is indicated: the envelope spectrum peak "
                f"matches the {primary['fault_type']} characteristic frequency "
                f"of this bearing within {primary['deviation_pct']:.2f}%."
            ),
            "fault_canonical": canonical,
            "fault_acronym": primary["fault_type"],
        }

    if bearing_block["status"] == ASSESSED:
        headline = (
            "No bearing fault is indicated: no characteristic frequency "
            "matched a peak in the envelope spectrum."
        )
    else:
        headline = (
            "No bearing fault verdict was reached, because characteristic-"
            "frequency matching was not attempted."
        )

    if iso_block["status"] == ASSESSED and iso_block["zone"] not in _ACCEPTABLE_ZONES:
        headline += (
            f" Broadband vibration is nonetheless elevated "
            f"(Zone {iso_block['zone']}), so a non-bearing source should be "
            f"considered."
        )
    elif anomaly_block.get("overall_health") in ("Faulty", "Suspicious"):
        headline += (
            " The anomaly model nonetheless flags this signal as departing "
            "from the learned healthy pattern, so a source outside the "
            "bearing's characteristic frequencies should be considered."
        )

    return {"statement": headline, "fault_canonical": None, "fault_acronym": ""}


def _build_evidence(
    diagnosis: dict,
    bearing_block: dict,
    iso_block: dict,
    anomaly_block: dict,
    energy_block: dict,
) -> dict:
    """Collect the facts that support the verdict, as authored sentences."""
    statements: list[str] = []

    for row in bearing_block.get("rows", []):
        if not row["matched"]:
            continue
        acronym = row["fault_type"]
        sentence = (
            f"{acronym} match: expected {row['expected_hz']:.2f} Hz, measured "
            f"{row['measured_hz']:.2f} Hz, deviation {row['deviation_pct']:.2f}%."
        )
        if row["harmonics"]:
            sentence += (
                f" {row['harmonics']} harmonic(s) of {acronym} are also "
                f"present, which a single noise peak would not produce."
            )
        statements.append(sentence)

    if iso_block["status"] == ASSESSED:
        statements.append(iso_block["statement"])
    if anomaly_block["status"] == ASSESSED:
        statements.append(anomaly_block["statement"])
    if energy_block["status"] == ASSESSED:
        statements.append(energy_block["statement"])

    peak = diagnosis.get("fft_summary", {}).get("peak_frequency_hz")
    if peak is not None:
        statements.append(
            f"Dominant frequency in the raw spectrum: {peak:.1f} Hz."
        )

    return {
        "strength": diagnosis.get("evidence_strength", "none"),
        "strength_explanation": EVIDENCE_STRENGTH_EXPLANATION,
        "statements": statements,
    }


# ---------------------------------------------------------------------------
# Indicator disagreement (R6)
# ---------------------------------------------------------------------------


def _build_disagreements(
    iso_block: dict, bearing_block: dict, anomaly_block: dict
) -> list[dict]:
    """Name the indicators that disagree, and which one governs the action.

    A refused ISO block is not an indicator and cannot disagree with one — an
    absent verdict is silence, not a contradicting opinion.
    """
    if iso_block["status"] != ASSESSED:
        return []
    if iso_block["zone"] not in _ACCEPTABLE_ZONES:
        return []

    dissenting: list[str] = []
    if anomaly_block.get("overall_health") in ("Faulty", "Suspicious"):
        dissenting.append(
            f"the anomaly model ({anomaly_block['overall_health']}, "
            f"{anomaly_block['anomaly_ratio'] * 100:.0f}% of segments)"
        )
    strong_matches = [
        r
        for r in bearing_block.get("rows", [])
        if r["matched"] and r["evidence_strength"] in ("high", "moderate")
    ]
    if strong_matches:
        names = ", ".join(r["fault_type"] for r in strong_matches)
        dissenting.append(f"the characteristic-frequency match ({names})")

    if not dissenting:
        return []

    return [
        {
            "statement": (
                f"Indicators disagree. Zone {iso_block['zone']} describes "
                f"overall vibration energy as acceptable, while "
                f"{' and '.join(dissenting)} indicate a developing fault. "
                f"These are not contradictory: a localised defect can be "
                f"unambiguous in the envelope spectrum while the broadband "
                f"level it contributes is still low. The fault-pattern "
                f"evidence governs the recommended action; the ISO zone "
                f"describes how far the condition has progressed."
            ),
            "governing_indicator": "fault-pattern evidence",
            "deferring_indicator": f"ISO zone {iso_block['zone']}",
        }
    ]


# ---------------------------------------------------------------------------
# Recommendations (R7)
# ---------------------------------------------------------------------------


def _build_recommendations(
    iso_block: dict,
    bearing_block: dict,
    anomaly_block: dict,
    disagreements: list[dict],
) -> list[dict]:
    """Attach a motivation and its supporting evidence to each action."""
    recommendations: list[dict] = []

    if iso_block["status"] == REFUSED:
        recommendations.append(
            {
                "action": iso_block["remedy"],
                "urgency": "medium",
                "description": (
                    "The severity verdict is unavailable until this is "
                    "resolved."
                ),
                "motivation": iso_block["reason"],
                "evidence": [iso_block["statement"]],
            }
        )
        zone = None
    else:
        zone = iso_block["zone"]

    fault_types = [
        r["fault_canonical"]
        for r in bearing_block.get("rows", [])
        if r["matched"] and r["fault_canonical"]
    ]

    if zone is not None:
        base = generate_recommendations(severity_zone=zone, fault_types=fault_types)
        for entry in base:
            if entry["description"].startswith("Detected fault"):
                acronym = _acronym_for(
                    entry["description"].removeprefix("Detected fault: ").rstrip(".")
                )
                motivation = (
                    "The characteristic frequency of this bearing element "
                    "matched a peak in the envelope spectrum within tolerance"
                    + (f" ({acronym})." if acronym else ".")
                )
                evidence = [bearing_block["statement"]]
            else:
                severity = iso_block.get("severity_level")
                motivation = (
                    f"ISO Zone {zone} is classified as {severity.lower()}."
                    if severity
                    else f"ISO Zone {zone}."
                )
                evidence = [iso_block["statement"]]
            recommendations.append(
                {**entry, "motivation": motivation, "evidence": evidence}
            )

    if disagreements:
        recommendations.insert(
            0,
            {
                "action": (
                    "Treat this as actionable despite the acceptable ISO zone"
                ),
                "urgency": "medium",
                "description": (
                    "Fault-pattern evidence and the severity zone describe "
                    "different stages of the same condition."
                ),
                "motivation": disagreements[0]["statement"],
                "evidence": [disagreements[0]["statement"]],
            },
        )

    if anomaly_block["status"] == ABSENT:
        recommendations.append(
            {
                "action": anomaly_block["remedy"],
                "urgency": "low",
                "description": (
                    "Pattern-level corroboration is unavailable without a "
                    "trained model."
                ),
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
            "statement": (
                "No healthy baseline was supplied for this machine, so the "
                "readings below are absolute rather than relative. Whether "
                "this condition is new, stable, or worsening cannot be "
                "determined from a single acquisition."
            ),
            "remedy": (
                "Re-run the diagnosis with a baseline signal id from the same "
                "machine in a known-good state."
            ),
            "deltas": [],
        }

    incompatible = _baseline_incompatibility(diagnosis, baseline)
    if incompatible:
        return {
            "status": REFUSED,
            "statement": (
                f"Baseline comparison was refused: {incompatible} A delta "
                f"between measurements taken under different conditions "
                f"would look like a change in machine condition."
            ),
            "remedy": (
                "Supply a baseline acquired from the same measurement point "
                "under the same declared conditions."
            ),
            "deltas": [],
        }

    deltas = [
        delta
        for delta in (
            _rms_delta(diagnosis, baseline),
            _anomaly_delta(diagnosis, baseline),
            _envelope_delta(diagnosis, baseline),
        )
        if delta is not None
    ]

    if not deltas:
        return {
            "status": ABSENT,
            "statement": (
                f"A baseline was supplied ({baseline.get('signal_id')}), but "
                f"no indicator could be compared: the two diagnoses share no "
                f"assessed block. Whether this condition is new, stable, or "
                f"worsening cannot be determined."
            ),
            "remedy": (
                "Ensure both signals declare their unit so the severity and "
                "anomaly blocks are assessed rather than refused."
            ),
            "deltas": [],
        }

    moved = [d for d in deltas if d["direction"] != "unchanged"]
    if not moved:
        statement = (
            f"Compared with baseline {baseline.get('signal_id')}: no "
            f"measurable change in any compared indicator. The condition "
            f"described above is stable, not developing."
        )
    else:
        headline = max(moved, key=lambda d: abs(d["delta"]))
        statement = (
            f"Compared with baseline {baseline.get('signal_id')}: "
            f"{len(moved)} of {len(deltas)} indicators moved. "
            f"{headline['statement']}"
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
    if signal_iso.get("status") == "assessed" and baseline_iso.get("status") == "assessed":
        signal_unit = signal_iso.get("original_unit")
        baseline_unit = baseline_iso.get("original_unit")
        if signal_unit != baseline_unit:
            return (
                f"the signal declares its unit as '{signal_unit}' while the "
                f"baseline declares '{baseline_unit}'."
            )

    signal_bearing = diagnosis.get("bearing_id")
    baseline_bearing = baseline.get("bearing_id")
    if signal_bearing and baseline_bearing and signal_bearing != baseline_bearing:
        return (
            f"the signal was analysed against bearing '{signal_bearing}' and "
            f"the baseline against '{baseline_bearing}'."
        )

    return ""


def _direction(delta: float, tolerance: float) -> str:
    if abs(delta) < tolerance:
        return "unchanged"
    return "higher" if delta > 0 else "lower"


def _rms_delta(diagnosis: dict, baseline: dict) -> Optional[dict]:
    signal_iso = diagnosis.get("iso_severity", {})
    baseline_iso = baseline.get("iso_severity", {})
    if signal_iso.get("status") != "assessed" or baseline_iso.get("status") != "assessed":
        return None

    now = signal_iso["rms_velocity_mm_s"]
    then = baseline_iso["rms_velocity_mm_s"]
    delta = now - then
    direction = _direction(delta, 0.005)
    if direction == "unchanged":
        statement = (
            f"RMS velocity is unchanged against baseline "
            f"({now:.2f} mm/s, was {then:.2f} mm/s)."
        )
    else:
        statement = (
            f"RMS velocity is {abs(delta):.2f} mm/s {direction} than baseline "
            f"({now:.2f} mm/s, was {then:.2f} mm/s)."
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
        statement = (
            f"The share of anomalous segments is unchanged against baseline "
            f"({now:.0f}%, was {then:.0f}%) — a difference of under one "
            f"percentage point."
        )
    else:
        statement = (
            f"The share of anomalous segments is {abs(delta):.0f} percentage "
            f"points {direction} than baseline ({now:.0f}%, was {then:.0f}%)."
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


def _envelope_delta(diagnosis: dict, baseline: dict) -> Optional[dict]:
    """Compare envelope amplitude at the fault frequency that actually matched.

    This is the delta that separates "the machine is noisier" from "this
    specific defect grew": it looks only at the frequency the verdict rests on.
    """
    signal_faults = diagnosis.get("bearing_faults")
    baseline_faults = baseline.get("bearing_faults")
    if not signal_faults or not baseline_faults:
        return None

    matched = next(
        (
            c
            for c in signal_faults.get("fault_checks", [])
            if c.get("detected") and c.get("magnitude")
        ),
        None,
    )
    if matched is None:
        return None

    reference = next(
        (
            c
            for c in baseline_faults.get("fault_checks", [])
            if c["fault_type"] == matched["fault_type"] and c.get("magnitude")
        ),
        None,
    )
    if reference is None:
        return None

    now = matched["magnitude"]
    then = reference["magnitude"]
    delta_db = 20 * math.log10(now / then)
    direction = _direction(delta_db, 0.05)
    acronym = matched["fault_type"]
    if direction == "unchanged":
        statement = (
            f"Envelope amplitude at the {acronym} frequency is unchanged "
            f"against baseline."
        )
    else:
        statement = (
            f"Envelope amplitude at the {acronym} frequency is "
            f"{abs(delta_db):.1f} dB {direction} than baseline — the defect "
            f"signature itself, not overall machine noise."
        )
    return {
        "indicator": "envelope_magnitude",
        "unit": "dB",
        "value": now,
        "baseline_value": then,
        "delta": round(delta_db, 2),
        "direction": direction,
        "statement": statement,
    }


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

    iso_block = _build_iso_block(diagnosis["iso_severity"])
    bearing_block = _build_bearing_block(diagnosis.get("bearing_faults"))
    anomaly_block = _build_anomaly_block(diagnosis.get("anomaly_detection"))
    energy_block = _build_energy_block(diagnosis["stft_summary"])

    verdict = _build_verdict(bearing_block, iso_block, anomaly_block)
    evidence = _build_evidence(
        diagnosis, bearing_block, iso_block, anomaly_block, energy_block
    )
    disagreements = _build_disagreements(iso_block, bearing_block, anomaly_block)
    baseline_block = build_baseline_comparison(diagnosis, baseline_diagnosis)
    recommendations = _build_recommendations(
        iso_block, bearing_block, anomaly_block, disagreements
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


def collect_statements(advisory: dict) -> list[str]:
    """Return every authored statement in the payload, in document order.

    This is the surface the cross-rendering parity test asserts on: a
    statement present here and absent from a rendering means that rendering
    dropped something the server said.
    """
    statements: list[str] = []

    def add(value: Optional[str]) -> None:
        if value and value not in statements:
            statements.append(value)

    add(advisory["verdict"]["statement"])
    for block_key in ("iso", "bearing_match", "anomaly", "spectral_energy"):
        add(advisory[block_key].get("statement"))
        add(advisory[block_key].get("remedy"))
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

    return statements
