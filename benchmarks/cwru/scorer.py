"""Scorer: join runner outcomes with fault labels into the results artifact.

This module is the benchmark's SOLE label reader (origin acceptance
example AE3): downloader, importer, and runner parse only the ops view,
while the scorer joins the runner's outcomes with ``labels.json``
through :func:`benchmarks.cwru.records.label_view` and emits
``results.json``, the single source of truth every published number
must trace to (AE4).

Hit criterion, applied identically to the fault records and to the
normal baselines: a fault hits on a record when its check's ``state`` is
``"supported"``, a significant line of its own with at least one counted
corroborating line, shared with no other hypothesis. The check tests the
rolling-element line at twice the ball spin frequency itself, so no
fault has a harmonic leg.

What users see, per stratum, each as a count over its denominator:

- ``frequency_detection``: on a fault record, the labelled fault's
  tested line is significant (``detected``);
- ``label_supported``: on a fault record, the labelled fault meets the
  hit criterion;
- ``outcomes``: the check's outcome against the label (indicated with
  the labelled fault, indicated with another, several types with the
  labelled fault among the candidates, several types without it,
  inconclusive, no indication);
- ``indicated_precision``: indicated outcomes whose single candidate is
  the labelled fault, over every indicated outcome (a normal baseline's
  indicated outcome is never correct);
- ``strongest_candidate``: on a fault record, the labelled fault is the
  first candidate the check lists (it lists them by level over the
  noise floor, an order that does not rank which type is more likely);
- ``named_fault``: on a fault record, the fault type the check names
  (``most_likely_fault``, set only when its claim level is
  ``single_spectrum_evidence``) is the labelled one, another one, or
  none;
- ``sensitivity``: on a fault record, a bearing alert (outcome
  indicated or several types);
- ``wrong_type_support``: on a fault record, a fault other than the
  labelled one meets the hit criterion;
- ``confusion``: the named fault type against the labelled fault type;
- ``normals``: on the normal baselines, counts of no indication,
  inconclusive, bearing alert and named fault, and of each supported
  fault type. They are counts on the recordings measured, never a rate.

Secondary ranking, kept as ``classification`` (also reported as
``end_to_end``): among the faults meeting the hit criterion, rank by
evidence tier (``high`` before ``moderate``), then fundamental peak
``magnitude`` descending, then absolute ``deviation_pct`` ascending,
then the fixed BPFO, BPFI, BSF, FTF order. Classification is correct
when the ranked-first fault maps to the label (``inner_race`` -> BPFI,
``ball`` -> BSF, ``outer_race`` -> BPFO; the outer-race position does
not change the frequency); a normal record is correct when no fault
meets the criterion. The check itself names no type on this ranking.

Stratification: Smith & Randall 2015 grades Y1/Y2 (expected detectable,
the ONLY strata feeding the headline aggregate), P1/P2 (partial), N1/N2
(known-hard, reported separately, NEVER in the headline), plus
``"ungraded"`` for the normal baselines and any record whose grade is
not transcribed. Records with documented ``known_anomalies`` get their
own reported line, flagged, never excluded from their stratum.

Fail closed: empty outcomes, an empty label view, a missing or extra
opaque id, any non-``"ok"`` outcome, or an outcome measured under an
evidence or claim policy version other than the engine's current one
refuses with the offending ids named. A partial, degenerate or stale
input must never masquerade as a score. Counts are always published
alongside rates; a rate over zero records is ``None``, never a
fabricated percentage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from predictive_maintenance_mcp.diagnostics.bearing_claims import CLAIM_POLICY_VERSION
from predictive_maintenance_mcp.diagnostics.bearing_evidence import EVIDENCE_POLICY

from benchmarks.cwru import runner
from benchmarks.cwru.records import LabeledRecord, label_view

__all__ = [
    "DATASET_SUBSET",
    "DEFAULT_RESULTS_PATH",
    "FAULT_KEYS",
    "HEADLINE_STRATA",
    "HIT_STATE",
    "LABEL_TO_CHECK_KEY",
    "LABEL_TYPES",
    "METADATA_KEYS",
    "OUTCOME_CLASSES",
    "STRATA",
    "collect_metadata",
    "fault_hit",
    "read_outcomes",
    "score_results",
    "write_results",
]

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Default committed-results path (the single source of truth README and
#: methodology cite). Tests always redirect to tmp paths, never here.
DEFAULT_RESULTS_PATH: Path = _PACKAGE_DIR / "results" / "results.json"

#: Human-readable name of the measured dataset subset, recorded in the
#: results metadata.
DATASET_SUBSET: str = "CWRU 12 kHz drive-end fault group + normal baselines (v1)"

#: The four fault-check keys of every outcome, in the fixed order used
#: as the ranking's final deterministic tie-break.
FAULT_KEYS: tuple[str, ...] = ("BPFO", "BPFI", "BSF", "FTF")

#: The evidence state that makes a hit, on fault records and normal
#: baselines alike.
HIT_STATE: str = "supported"

#: Outcomes that raise a bearing alert.
_ALERT_OUTCOMES: frozenset[str] = frozenset({"indicated", "several_types"})

#: The check's outcome read against the label, in reporting order.
OUTCOME_CLASSES: tuple[str, ...] = (
    "indicated_correct",
    "indicated_wrong",
    "several_types_with_label",
    "several_types_without_label",
    "inconclusive",
    "no_indication",
)

#: Label fault_type -> expected fault-check key. ``normal`` is absent on
#: purpose: a normal record expects NO fault to meet the criterion. The
#: outer-race position (or_position) does not change the frequency.
LABEL_TO_CHECK_KEY: Mapping[str, str] = MappingProxyType(
    {"inner_race": "BPFI", "ball": "BSF", "outer_race": "BPFO"}
)

#: Rows of the confusion table: the labelled fault types and the normal
#: baselines. Every row is always present so the schema is stable.
LABEL_TYPES: tuple[str, ...] = (*LABEL_TO_CHECK_KEY, "normal")

#: Columns of the confusion table: the named fault type, or none.
_NAMED_COLUMNS: tuple[str, ...] = (*FAULT_KEYS, "none")

#: Every reporting stratum, in order: S&R 2015 grades plus the visible
#: bucket for the baselines and not-yet-transcribed grades. All are always
#: present in the results (zero counts, null rates) so the schema is
#: stable.
STRATA: tuple[str, ...] = ("Y1", "Y2", "P1", "P2", "N1", "N2", "ungraded")

#: Strata feeding the headline aggregate. N1/N2 (known-hard) and the
#: partial/ungraded strata are reported separately, never here.
HEADLINE_STRATA: tuple[str, ...] = ("Y1", "Y2")

#: Metadata keys of the results artifact; every one is overridable so
#: tests stay deterministic (date, git describe, platform vary by run).
METADATA_KEYS: frozenset[str] = frozenset({*runner.PROVENANCE_KEYS, "dataset_subset"})

#: Secondary-ranking tiers of a supported hypothesis's evidence grade.
_EVIDENCE_TIER: Mapping[str, int] = MappingProxyType({"high": 2, "moderate": 1})

#: The policy versions an outcome must carry: the engine's current ones.
_CURRENT_POLICY: Mapping[str, Any] = MappingProxyType(
    {
        "evidence_policy_version": EVIDENCE_POLICY["version"],
        "claim_policy_version": CLAIM_POLICY_VERSION,
    }
)

#: Summary fields every scored outcome must carry next to its checks.
_SUMMARY_KEYS: tuple[str, ...] = (
    "outcome",
    "candidates",
    "claim_level",
    "most_likely_fault",
)


def collect_metadata(overrides: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Collect the results-artifact metadata, with deterministic overrides.

    Every value that varies between runs (date, git describe, platform,
    versions) is produced lazily and can be overridden, so tests never
    depend on the machine or the clock. Overridden keys skip their
    producer entirely (no subprocess runs for an overridden
    ``git_describe``).

    Args:
        overrides: Values to use verbatim instead of collecting. Keys
            must be a subset of :data:`METADATA_KEYS`.

    Returns:
        The complete metadata mapping (every key in
        :data:`METADATA_KEYS` present).

    Raises:
        ValueError: On an unknown override key (fail closed — a typo'd
            override silently ignored would leave a varying value in an
            artifact a test believed pinned), or from
            :func:`_git_describe`.
    """
    resolved = dict(overrides or {})
    unknown = sorted(set(resolved) - METADATA_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown metadata override key(s) {unknown} — valid keys are "
            f"{sorted(METADATA_KEYS)}. Fix the caller."
        )
    provenance = runner.collect_provenance(
        {key: value for key, value in resolved.items() if key in runner.PROVENANCE_KEYS}
    )
    return {
        **provenance,
        "dataset_subset": resolved.get("dataset_subset", DATASET_SUBSET),
    }


def fault_hit(check: Mapping[str, Any]) -> bool:
    """Apply the symmetric hit criterion to one fault check.

    The SAME criterion counts hits on fault records and on normal
    baselines: the check's evidence state is supported.

    Args:
        check: The check's raw analyzer fields from the outcome.

    Returns:
        True when the fault hits on this record.
    """
    return bool(check.get("state") == HIT_STATE)


def _rank_sort_key(
    fault_key: str, check: Mapping[str, Any]
) -> tuple[int, float, float, int]:
    """Sort key of the secondary ranking (min = strongest).

    Args:
        fault_key: The check's key (must meet the hit criterion).
        check: The check's raw analyzer fields.

    Returns:
        ``(-tier, -magnitude, |deviation|, fixed-order index)`` so that
        ``min()`` picks the strongest fault per the module docstring.
    """
    tier = _EVIDENCE_TIER.get(str(check.get("evidence_strength")), 0)
    magnitude = check.get("magnitude")
    deviation = check.get("deviation_pct")
    magnitude_key = float(magnitude) if magnitude is not None else float("-inf")
    deviation_key = abs(float(deviation)) if deviation is not None else float("inf")
    return (-tier, -magnitude_key, deviation_key, FAULT_KEYS.index(fault_key))


def _bearing_of(opaque_id: str, outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract and structurally validate an outcome's bearing block.

    Args:
        opaque_id: The record the outcome belongs to (for errors).
        outcome: One ``status: "ok"`` runner outcome.

    Returns:
        The bearing block, guaranteed to carry all of :data:`FAULT_KEYS`
        in ``fault_checks`` and every summary field the scorer reads.

    Raises:
        ValueError: When the bearing block, a fault check or a summary
            field is missing or malformed — the outcome cannot be scored,
            so the refusal names the record and the re-run remedy.
    """
    bearing = outcome.get("bearing")
    if not isinstance(bearing, Mapping) or not isinstance(
        bearing.get("fault_checks"), Mapping
    ):
        raise ValueError(
            f"Outcome for record '{opaque_id}' carries no "
            f"bearing.fault_checks block — it cannot be scored. Re-run "
            f"the measurement (python -m benchmarks.cwru run) so every "
            f"outcome carries the full bearing subset."
        )
    checks = bearing["fault_checks"]
    missing = [key for key in FAULT_KEYS if not isinstance(checks.get(key), Mapping)]
    if missing:
        raise ValueError(
            f"Outcome for record '{opaque_id}' is missing fault check(s) "
            f"{missing} (expected exactly {list(FAULT_KEYS)}) — it cannot "
            f"be scored. Re-run the measurement (python -m benchmarks.cwru "
            f"run) so every outcome carries all four checks."
        )
    absent = [key for key in _SUMMARY_KEYS if key not in bearing]
    if absent:
        raise ValueError(
            f"Outcome for record '{opaque_id}' is missing the summary "
            f"field(s) {absent} — it was measured by an earlier runner and "
            f"cannot be scored. Re-run the measurement (python -m "
            f"benchmarks.cwru all) so every outcome carries them."
        )
    return bearing


def _assert_current_policy(outcomes: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse outcomes measured under another evidence or claim policy.

    Args:
        outcomes: Runner outcomes keyed by opaque id.

    Raises:
        ValueError: Naming every record whose recorded policy versions
            differ from the engine's current ones, grouped by the versions
            recorded, with the re-run remedy.
    """
    current = dict(_CURRENT_POLICY)
    stale: dict[tuple[tuple[str, Any], ...], list[str]] = {}
    for opaque_id in sorted(outcomes):
        bearing = outcomes[opaque_id].get("bearing")
        if not isinstance(bearing, Mapping):
            continue
        recorded = {key: bearing.get(key) for key in current}
        if recorded != current:
            stale.setdefault(tuple(recorded.items()), []).append(opaque_id)
    if stale:
        groups = "; ".join(
            f"{ids} recorded {dict(versions)}" for versions, ids in stale.items()
        )
        raise ValueError(
            f"Refusing to score outcomes measured under another policy than "
            f"the engine's current {current}: {groups}. Re-measure with "
            f"python -m benchmarks.cwru all, so every published number "
            f"describes what the engine reports today."
        )


def _outcome_class(
    outcome: str, candidates: Sequence[str], expected: Optional[str]
) -> str:
    """Read the check's outcome against the label (one of OUTCOME_CLASSES)."""
    if outcome == "indicated":
        correct = expected is not None and list(candidates) == [expected]
        return "indicated_correct" if correct else "indicated_wrong"
    if outcome == "several_types":
        with_label = expected is not None and expected in candidates
        return (
            "several_types_with_label" if with_label else "several_types_without_label"
        )
    if outcome in ("inconclusive", "no_indication"):
        return outcome
    raise ValueError(
        f"Unknown check outcome {outcome!r} — the scorer reads "
        f"indicated, several_types, inconclusive and no_indication. "
        f"Update the scorer together with the engine's outcome vocabulary."
    )


def _score_record(labeled: LabeledRecord, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Score one record: hits, what the user sees, and the verdicts.

    Args:
        labeled: The record's ops metadata joined with its label.
        outcome: The record's ``status: "ok"`` runner outcome.

    Returns:
        The per-record results row (stratum, flags, per-fault hits, the
        check's outcome and candidates, the fault it names, and hit/miss
        per metric family; fault-record fields are ``None`` on normals).

    Raises:
        ValueError: Propagated from structural validation.
    """
    bearing = _bearing_of(labeled.ops.opaque_id, outcome)
    checks = bearing["fault_checks"]
    hits = {key: fault_hit(checks[key]) for key in FAULT_KEYS}
    hitting = [key for key in FAULT_KEYS if hits[key]]
    ranked_first = (
        min(hitting, key=lambda key: _rank_sort_key(key, checks[key]))
        if hitting
        else None
    )
    candidates = [str(candidate) for candidate in bearing["candidates"]]
    strongest = candidates[0] if candidates else None
    named = bearing["most_likely_fault"]
    check_outcome = str(bearing["outcome"])

    expected = LABEL_TO_CHECK_KEY.get(labeled.label.fault_type)
    if expected is None:
        frequency_detection_hit: Optional[bool] = None
        label_supported: Optional[bool] = None
        false_positive: Optional[bool] = bool(hitting)
        classification_correct = not hitting
        strongest_correct: Optional[bool] = None
        named_correct: Optional[bool] = None
        wrong_type_supported: Optional[bool] = None
    else:
        frequency_detection_hit = bool(checks[expected].get("detected"))
        label_supported = hits[expected]
        false_positive = None
        classification_correct = ranked_first == expected
        strongest_correct = strongest == expected
        named_correct = named == expected if named is not None else None
        wrong_type_supported = any(hits[key] for key in FAULT_KEYS if key != expected)

    return {
        "stratum": labeled.grade_stratum,
        "known_anomalies": bool(labeled.label.known_anomalies),
        "label_type": labeled.label.fault_type,
        "expected_fault": expected,
        "fault_hits": hits,
        "ranked_first": ranked_first,
        "frequency_detection_hit": frequency_detection_hit,
        "label_supported": label_supported,
        "classification_correct": classification_correct,
        # End-to-end is the record-level verdict (fault present + type)
        # from the same fields — kept as its own reported key per plan.
        "end_to_end_correct": classification_correct,
        "false_positive": false_positive,
        "outcome": check_outcome,
        "outcome_class": _outcome_class(check_outcome, candidates, expected),
        "claim_level": bearing["claim_level"],
        "candidates": candidates,
        "strongest_candidate": strongest,
        "strongest_candidate_correct": strongest_correct,
        "named_fault": named,
        "named_correct": named_correct,
        "wrong_type_supported": wrong_type_supported,
        "bearing_alert": check_outcome in _ALERT_OUTCOMES,
    }


def _rate(count: int, total: int) -> Optional[float]:
    """A rate with an honest zero-denominator: ``None``, never 0% or 100%."""
    return count / total if total else None


def _fraction(name: str, count: int, total: int) -> dict[str, Any]:
    """A count with its denominator and rate, under the given count name."""
    return {name: count, "total": total, "rate": _rate(count, total)}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record rows into counts AND rates per metric family.

    Counts are always published alongside rates; an empty denominator
    yields a ``None`` rate.

    Args:
        rows: Per-record results rows (from :func:`_score_record`).

    Returns:
        The aggregate block for one stratum/group.
    """
    faulted = [row for row in rows if row["expected_fault"] is not None]
    normal = [row for row in rows if row["expected_fault"] is None]
    indicated = [row for row in rows if row["outcome"] == "indicated"]
    classification = sum(1 for row in rows if row["classification_correct"])
    end_to_end = sum(1 for row in rows if row["end_to_end_correct"])
    confusion = {
        label: {column: 0 for column in _NAMED_COLUMNS} for label in LABEL_TYPES
    }
    for row in rows:
        confusion[row["label_type"]][row["named_fault"] or "none"] += 1
    return {
        "n_records": len(rows),
        "n_faulted": len(faulted),
        "n_normal": len(normal),
        "frequency_detection": _fraction(
            "hits",
            sum(1 for row in faulted if row["frequency_detection_hit"]),
            len(faulted),
        ),
        "label_supported": _fraction(
            "count", sum(1 for row in faulted if row["label_supported"]), len(faulted)
        ),
        "outcomes": {
            **{
                name: sum(1 for row in rows if row["outcome_class"] == name)
                for name in OUTCOME_CLASSES
            },
            "total": len(rows),
        },
        "indicated_precision": _fraction(
            "correct",
            sum(1 for row in indicated if row["outcome_class"] == "indicated_correct"),
            len(indicated),
        ),
        "strongest_candidate": _fraction(
            "correct",
            sum(1 for row in faulted if row["strongest_candidate_correct"]),
            len(faulted),
        ),
        "named_fault": {
            "correct": sum(1 for row in faulted if row["named_correct"] is True),
            "wrong": sum(1 for row in faulted if row["named_correct"] is False),
            "none": sum(1 for row in faulted if row["named_fault"] is None),
            "total": len(faulted),
        },
        "sensitivity": _fraction(
            "alerts", sum(1 for row in faulted if row["bearing_alert"]), len(faulted)
        ),
        "wrong_type_support": _fraction(
            "records",
            sum(1 for row in faulted if row["wrong_type_supported"]),
            len(faulted),
        ),
        "classification": _fraction("correct", classification, len(rows)),
        "end_to_end": _fraction("correct", end_to_end, len(rows)),
        "normals": {
            "total": len(normal),
            "no_indication": sum(
                1 for row in normal if row["outcome"] == "no_indication"
            ),
            "inconclusive": sum(
                1 for row in normal if row["outcome"] == "inconclusive"
            ),
            "bearing_alert": sum(1 for row in normal if row["bearing_alert"]),
            "named_fault": sum(1 for row in normal if row["named_fault"] is not None),
            "supported_counts": {
                key: sum(1 for row in normal if row["fault_hits"][key])
                for key in FAULT_KEYS
            },
        },
        "confusion": confusion,
    }


def _criteria() -> dict[str, Any]:
    """The operational definitions, embedded in the artifact for audit."""
    return {
        "hit_criterion": (
            "A fault hits when its check's state is 'supported': a "
            "significant line of its own with at least one counted "
            "corroborating line, shared with no other hypothesis. The "
            "identical criterion is applied to the fault records and to the "
            "normal baselines. The check tests the rolling-element line at "
            "twice the ball spin frequency itself, so no fault has a "
            "harmonic leg."
        ),
        "hit_state": HIT_STATE,
        "frequency_detection": (
            "On a fault record, the labelled fault's tested line is "
            "significant (detected == true), whether or not the hypothesis "
            "is supported."
        ),
        "outcome_classes": {
            "indicated_correct": (
                "Outcome 'indicated' with the labelled fault as its single "
                "candidate."
            ),
            "indicated_wrong": (
                "Outcome 'indicated' with another fault as its single "
                "candidate; on a normal baseline, every indicated outcome."
            ),
            "several_types_with_label": (
                "Outcome 'several_types' with the labelled fault among the "
                "candidates."
            ),
            "several_types_without_label": (
                "Outcome 'several_types' without the labelled fault among the "
                "candidates; on a normal baseline, every several-types outcome."
            ),
            "inconclusive": "Outcome 'inconclusive'.",
            "no_indication": "Outcome 'no_indication'.",
        },
        "indicated_precision": (
            "Indicated outcomes whose single candidate is the labelled fault, "
            "over every indicated outcome of the group."
        ),
        "strongest_candidate": (
            "On a fault record, the labelled fault is the first of the "
            "check's candidates. The check lists its candidates by level over "
            "the noise floor; that order does not rank which type is more "
            "likely."
        ),
        "named_fault": (
            "The fault type the check names: most_likely_fault, which it sets "
            "only when its claim level is 'single_spectrum_evidence'. Counted "
            "on the fault records as the labelled type, another type, or none."
        ),
        "sensitivity": (
            "On a fault record, a bearing alert: outcome 'indicated' or "
            "'several_types'."
        ),
        "wrong_type_support": (
            "On a fault record, a fault other than the labelled one meets the "
            "hit criterion."
        ),
        "normals": (
            "On the normal baselines, counts of each outcome kind, of named "
            "faults and of each supported fault type: counts on the "
            "recordings measured, never a rate."
        ),
        "ranking_rule": (
            "Secondary metric (classification, also reported as end_to_end). "
            "Among faults meeting the hit criterion: evidence tier (high "
            "before moderate), then fundamental peak magnitude descending, "
            "then absolute deviation_pct ascending, then the fixed "
            "BPFO/BPFI/BSF/FTF order. Classification is correct when the "
            "ranked-first fault maps to the label; a normal record is correct "
            "when no fault meets the criterion. The check itself names no "
            "type on this ranking."
        ),
        "label_to_check_key": dict(LABEL_TO_CHECK_KEY),
        "policy_versions": {
            "evidence": _CURRENT_POLICY["evidence_policy_version"],
            "claim": _CURRENT_POLICY["claim_policy_version"],
        },
    }


def score_results(
    outcomes: Mapping[str, Mapping[str, Any]],
    labeled: Optional[Sequence[LabeledRecord]] = None,
    *,
    metadata_overrides: Optional[Mapping[str, str]] = None,
    measurement_provenance: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Join outcomes with labels and compute the stratified results.

    The sole label join of the benchmark: everything upstream is blind.
    Fails closed on empty, partial or stale input: outcome ids must match
    the labeled records exactly, every outcome must be ``"ok"``, and every
    outcome must carry the engine's current policy versions.

    Args:
        outcomes: Runner outcomes keyed by opaque id.
        labeled: Labeled records to score against (tests only);
            ``None`` reads the vendored tables via ``label_view()``.
        metadata_overrides: Verbatim metadata values (tests pin date,
            git describe, and platform for determinism).
        measurement_provenance: Provenance values coming from the outcomes.

    Returns:
        The results document (see this module's docstring and
        ``write_results`` for the artifact schema): ``metadata``,
        ``measurement_provenance``, ``criteria``, per-stratum aggregates
        in ``strata``, the Y1/Y2-only ``headline``, the
        ``known_anomalies_flagged`` line, and the per-record ``records``
        table.

    Raises:
        ValueError: On empty outcomes, an empty label view, duplicate
            labeled ids, id mismatch or non-ok outcomes (ids named),
            outcomes from another policy version (ids named), a
            structurally unscorable outcome, or a bad metadata override.
    """
    resolved = tuple(labeled) if labeled is not None else label_view()
    if not resolved:
        raise ValueError(
            "The label view is empty — refusing to score against zero "
            "labels (a degenerate join would masquerade as a measured "
            "result). Restore the vendored tables from git, or pass a "
            "non-empty labeled record set."
        )
    if not outcomes:
        raise ValueError(
            "The outcome set is empty — refusing to produce a score from "
            "zero measurements (never a 100%/0% report). Produce outcomes "
            "first: python -m benchmarks.cwru run."
        )
    ids = [record.ops.opaque_id for record in resolved]
    duplicates = sorted({oid for oid in ids if ids.count(oid) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate opaque id(s) {duplicates} in the labeled records — "
            f"a duplicate would silently overwrite a scored row. Fix the "
            f"record set (label_view() already enforces uniqueness)."
        )
    # Fail-closed join gate, reusing the runner's canonical check:
    # missing ids, extra ids, and non-ok outcomes each refuse, named.
    runner.assert_outcomes_complete(outcomes, [record.ops for record in resolved])
    _assert_current_policy(outcomes)

    rows = {
        record.ops.opaque_id: _score_record(record, outcomes[record.ops.opaque_id])
        for record in resolved
    }
    strata = {
        stratum: _aggregate([row for row in rows.values() if row["stratum"] == stratum])
        for stratum in STRATA
    }
    headline_rows = [row for row in rows.values() if row["stratum"] in HEADLINE_STRATA]
    flagged_ids = sorted(oid for oid, row in rows.items() if row["known_anomalies"])
    return {
        "metadata": collect_metadata(metadata_overrides),
        "measurement_provenance": (
            dict(measurement_provenance) if measurement_provenance is not None else None
        ),
        "criteria": _criteria(),
        "strata": strata,
        "headline": {
            "strata_included": list(HEADLINE_STRATA),
            **_aggregate(headline_rows),
        },
        "known_anomalies_flagged": {
            "record_ids": flagged_ids,
            **_aggregate([rows[oid] for oid in flagged_ids]),
        },
        "records": rows,
    }


def read_outcomes(
    path: Optional[Path] = None,
) -> tuple[dict[str, dict[str, Any]], Optional[dict[str, str]]]:
    """Read a runner outcomes artifact for scoring.

    Args:
        path: Artifact override; ``None`` uses the runner's committed
            default (``benchmarks/cwru/results/outcomes.json``).

    Returns:
         A ``(outcomes, provenance)`` pair, where ``provenance`` is the
        ``_provenance`` block when present and ``None`` for artifacts
        that predate provenance collection.
    Raises:
        ValueError: If the artifact is absent (remedy: run the
            measurement first), unparseable, or not a JSON object.
    """
    target = Path(path) if path is not None else runner.DEFAULT_OUTCOMES_PATH
    if not target.exists():
        raise ValueError(
            f"Outcomes artifact not found at {target} — produce it first "
            f"with 'python -m benchmarks.cwru run' (default output "
            f"{runner.DEFAULT_OUTCOMES_PATH}), then score."
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Outcomes artifact at {target} is not valid JSON ({exc}) — "
            f"re-run 'python -m benchmarks.cwru run' to regenerate it."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"Outcomes artifact at {target} must be a JSON object keyed by "
            f"opaque id, got {type(raw).__name__} — re-run 'python -m "
            f"benchmarks.cwru run' to regenerate it."
        )
    return runner.split_provenance(raw)


def write_results(results: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Write the results artifact via the runner's canonical serializer.

    Same guarantees as the outcomes artifact: canonicalized scalars,
    floats rounded to :data:`runner.FLOAT_DECIMALS` places, sorted keys,
    newline-terminated UTF-8, atomic ``.part`` + rename.

    Args:
        results: The document from :func:`score_results`.
        path: Destination override; ``None`` uses
            :data:`DEFAULT_RESULTS_PATH` (the committed artifact slot —
            tests must always pass a tmp path).

    Returns:
        The path written.

    Raises:
        ValueError: Propagated from canonicalization (non-finite float,
            unsupported type, non-string key).
    """
    target = Path(path) if path is not None else DEFAULT_RESULTS_PATH
    return runner.write_outcomes(results, target)
