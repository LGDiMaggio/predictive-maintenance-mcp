"""Scorer: join runner outcomes with fault labels into the results artifact.

This module is the benchmark's SOLE label reader (origin acceptance
example AE3): downloader, importer, and runner parse only the ops view,
while the scorer joins the runner's outcomes with ``labels.json``
through :func:`benchmarks.cwru.records.label_view` and emits
``results.json`` — the single source of truth every published number
must trace to (AE4).

Hit criterion (the methodology's operational core, applied IDENTICALLY
to detection on faulted records and to false-positive counting on the
normal baselines — the symmetry is load-bearing for credibility):

- A fault hits on a record when its check has ``detected == True`` AND
  ``evidence_strength`` in ``{"high", "moderate"}`` (the analyzer's
  vocabulary is high/moderate/low/none; a detected fundamental is never
  rated below moderate).
- BSF (ball) ONLY additionally hits when harmonic 2 is present in the
  BSF check's ``harmonics_detected`` — an entry whose ``"harmonic"``
  field equals 2. CWRU's published "rolling element" factor (4.7135) is
  2xBSF, and the analyzer caps harmonic-only findings at
  ``detected=False`` / evidence ``"low"``, so this leg must read the
  raw per-fault fields. No other fault gets a harmonic leg.

Classification ranking (the scorer computes its own record-level
ranking from the raw per-fault fields; it never consumes the analyzer's
``most_likely``, which structurally cannot name BSF on 2x-only
evidence): among the faults meeting the hit criterion, rank by

1. evidence tier — ``high`` (3) > ``moderate`` (2) > harmonic-only (1);
2. fundamental peak ``magnitude``, descending (a harmonic-only hit uses
   the 2x harmonic entry's magnitude; a missing magnitude sorts last);
3. absolute ``deviation_pct``, ascending (missing sorts last);
4. fixed BPFO, BPFI, BSF, FTF order as the final deterministic
   tie-break.

Classification is correct when the ranked-first fault maps to the label
(``inner_race`` -> BPFI, ``ball`` -> BSF, ``outer_race`` -> BPFO — the
outer-race position does not change the frequency); a normal record is
correct when NO fault meets the criterion. End-to-end is the
record-level verdict (fault present + type) from the same fields —
numerically the classification metric, reported under both names per
stratum, as the plan requires.

Stratification: Smith & Randall 2015 grades Y1/Y2 (expected detectable
— the ONLY strata feeding the headline aggregate), P1/P2 (partial),
N1/N2 (known-hard, reported separately, NEVER in headline accuracy),
plus ``"ungraded"`` for records whose grade is not yet transcribed.
Records with documented ``known_anomalies`` get their own reported
line, flagged, never excluded from their stratum.

Fail closed: empty outcomes, an empty label view, a missing or extra
opaque id, or any non-``"ok"`` outcome refuses with the offending ids
named — a partial or degenerate input must never masquerade as a score
(never a 100%/0% report). Counts are always published alongside rates;
a rate over zero records is ``None``, never a fabricated percentage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from benchmarks.cwru import runner
from benchmarks.cwru.records import LabeledRecord, label_view

__all__ = [
    "DATASET_SUBSET",
    "DEFAULT_RESULTS_PATH",
    "FAULT_KEYS",
    "HARMONIC_LEG_FAULT",
    "HARMONIC_LEG_ORDER",
    "HEADLINE_STRATA",
    "HIT_EVIDENCE_LEVELS",
    "LABEL_TO_CHECK_KEY",
    "METADATA_KEYS",
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

#: Evidence levels that satisfy the primary leg of the hit criterion
#: (the analyzer vocabulary is high/moderate/low/none).
HIT_EVIDENCE_LEVELS: frozenset[str] = frozenset({"high", "moderate"})

#: The ONLY fault with a harmonic leg: CWRU's published "rolling
#: element" factor is 2xBSF, so a 2x-harmonic-only finding is a hit for
#: BSF and for no other fault.
HARMONIC_LEG_FAULT: str = "BSF"

#: Harmonic order the harmonic leg tests for (the 2x entry).
HARMONIC_LEG_ORDER: int = 2

#: Label fault_type -> expected fault-check key. ``normal`` is absent on
#: purpose: a normal record expects NO fault to meet the criterion. The
#: outer-race position (or_position) does not change the frequency.
LABEL_TO_CHECK_KEY: Mapping[str, str] = MappingProxyType(
    {"inner_race": "BPFI", "ball": "BSF", "outer_race": "BPFO"}
)

#: Every reporting stratum, in order: S&R 2015 grades plus the visible
#: bucket for not-yet-transcribed grades. All are always present in the
#: results (zero counts, null rates) so the schema is stable.
STRATA: tuple[str, ...] = ("Y1", "Y2", "P1", "P2", "N1", "N2", "ungraded")

#: Strata feeding the headline aggregate. N1/N2 (known-hard) and the
#: partial/ungraded strata are reported separately, never here.
HEADLINE_STRATA: tuple[str, ...] = ("Y1", "Y2")

#: Metadata keys of the results artifact; every one is overridable so
#: tests stay deterministic (date, git describe, platform vary by run).
METADATA_KEYS: frozenset[str] = frozenset({*runner.PROVENANCE_KEYS, "dataset_subset"})
#: Ranking tiers for the primary (detected + evidence) leg.
_EVIDENCE_TIER: Mapping[str, int] = MappingProxyType({"high": 3, "moderate": 2})

#: Ranking tier for a harmonic-only BSF hit — below every primary-leg
#: tier: a detected fundamental always outranks a 2x-only finding.
_HARMONIC_ONLY_TIER: int = 1


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


def _harmonic_leg_entry(check: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the check's 2x-harmonic entry, or ``None`` when absent.

    Membership test for the harmonic leg, defined from the analyzer's
    ``harmonics_detected`` shape: a list of dicts, each carrying an
    integer ``"harmonic"`` order (2, 3, ...) plus expected/detected
    frequency and magnitude. The leg fires exactly when an entry's
    order equals :data:`HARMONIC_LEG_ORDER`.

    Args:
        check: One fault check's raw analyzer fields.

    Returns:
        The matching harmonic entry, or ``None``.
    """
    harmonics = check.get("harmonics_detected") or []
    for entry in harmonics:
        if isinstance(entry, Mapping) and entry.get("harmonic") == HARMONIC_LEG_ORDER:
            return entry
    return None


def _primary_leg_hits(check: Mapping[str, Any]) -> bool:
    """The primary leg: detected fundamental with evidence >= moderate."""
    return (
        bool(check.get("detected"))
        and check.get("evidence_strength") in HIT_EVIDENCE_LEVELS
    )


def fault_hit(fault_key: str, check: Mapping[str, Any]) -> bool:
    """Apply the symmetric hit criterion to one fault check.

    The SAME criterion counts detections on faulted records and false
    positives on normal baselines — see the module docstring for the
    full definition.

    Args:
        fault_key: The check's key (BPFO/BPFI/BSF/FTF).
        check: The check's raw analyzer fields from the outcome.

    Returns:
        True when the fault hits on this record.
    """
    if _primary_leg_hits(check):
        return True
    if fault_key == HARMONIC_LEG_FAULT:
        return _harmonic_leg_entry(check) is not None
    return False


def _rank_sort_key(
    fault_key: str, check: Mapping[str, Any]
) -> tuple[int, float, float, int]:
    """Sort key implementing the documented ranking rule (min = strongest).

    Args:
        fault_key: The check's key (must meet the hit criterion).
        check: The check's raw analyzer fields.

    Returns:
        ``(-tier, -magnitude, |deviation|, fixed-order index)`` so that
        ``min()`` picks the strongest fault per the module docstring.
    """
    if _primary_leg_hits(check):
        tier = _EVIDENCE_TIER[str(check.get("evidence_strength"))]
        magnitude = check.get("magnitude")
        deviation = check.get("deviation_pct")
    else:
        # Harmonic-only leg (BSF): rank by the 2x entry's magnitude.
        tier = _HARMONIC_ONLY_TIER
        entry = _harmonic_leg_entry(check)
        magnitude = entry.get("magnitude") if entry is not None else None
        deviation = None
    magnitude_key = float(magnitude) if magnitude is not None else float("-inf")
    deviation_key = abs(float(deviation)) if deviation is not None else float("inf")
    return (-tier, -magnitude_key, deviation_key, FAULT_KEYS.index(fault_key))


def _fault_checks_of(
    opaque_id: str, outcome: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    """Extract and structurally validate an outcome's fault checks.

    Args:
        opaque_id: The record the outcome belongs to (for errors).
        outcome: One ``status: "ok"`` runner outcome.

    Returns:
        The per-fault checks, guaranteed to carry all of
        :data:`FAULT_KEYS`.

    Raises:
        ValueError: When the bearing block or any fault check is
            missing/malformed — the outcome cannot be scored, so the
            refusal names the record and the re-run remedy.
    """
    bearing = outcome.get("bearing")
    checks = bearing.get("fault_checks") if isinstance(bearing, Mapping) else None
    if not isinstance(checks, Mapping):
        raise ValueError(
            f"Outcome for record '{opaque_id}' carries no "
            f"bearing.fault_checks block — it cannot be scored. Re-run "
            f"the measurement (python -m benchmarks.cwru run) so every "
            f"outcome carries the full bearing subset."
        )
    missing = [key for key in FAULT_KEYS if not isinstance(checks.get(key), Mapping)]
    if missing:
        raise ValueError(
            f"Outcome for record '{opaque_id}' is missing fault check(s) "
            f"{missing} (expected exactly {list(FAULT_KEYS)}) — it cannot "
            f"be scored. Re-run the measurement (python -m benchmarks.cwru "
            f"run) so every outcome carries all four checks."
        )
    return {key: checks[key] for key in FAULT_KEYS}


def _score_record(labeled: LabeledRecord, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Score one record: hits, ranking, and the per-metric verdicts.

    Args:
        labeled: The record's ops metadata joined with its label.
        outcome: The record's ``status: "ok"`` runner outcome.

    Returns:
        The per-record results row (stratum, flags, per-fault hits,
        ranked-first fault, and hit/miss per metric family).

    Raises:
        ValueError: Propagated from structural validation.
    """
    checks = _fault_checks_of(labeled.ops.opaque_id, outcome)
    hits = {key: fault_hit(key, checks[key]) for key in FAULT_KEYS}
    hitting = [key for key in FAULT_KEYS if hits[key]]
    ranked_first = (
        min(hitting, key=lambda key: _rank_sort_key(key, checks[key]))
        if hitting
        else None
    )

    expected = LABEL_TO_CHECK_KEY.get(labeled.label.fault_type)
    if expected is None:
        # Normal baseline: correct means NO fault meets the criterion;
        # any hit is a false positive under the same criterion.
        frequency_detection_hit: Optional[bool] = None
        false_positive: Optional[bool] = bool(hitting)
        classification_correct = not hitting
    else:
        frequency_detection_hit = hits[expected]
        false_positive = None
        classification_correct = ranked_first == expected

    return {
        "stratum": labeled.grade_stratum,
        "known_anomalies": bool(labeled.label.known_anomalies),
        "expected_fault": expected,
        "fault_hits": hits,
        "ranked_first": ranked_first,
        "frequency_detection_hit": frequency_detection_hit,
        "classification_correct": classification_correct,
        # End-to-end is the record-level verdict (fault present + type)
        # from the same fields — kept as its own reported key per plan.
        "end_to_end_correct": classification_correct,
        "false_positive": false_positive,
    }


def _rate(count: int, total: int) -> Optional[float]:
    """A rate with an honest zero-denominator: ``None``, never 0% or 100%."""
    return count / total if total else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record rows into counts AND rates per metric family.

    Counts are always published alongside rates (with n=4 baselines a
    rate alone invites statistical critique); an empty denominator
    yields a ``None`` rate.

    Args:
        rows: Per-record results rows (from :func:`_score_record`).

    Returns:
        The aggregate block for one stratum/group.
    """
    faulted = [row for row in rows if row["expected_fault"] is not None]
    normal = [row for row in rows if row["expected_fault"] is None]
    frequency_hits = sum(1 for row in faulted if row["frequency_detection_hit"])
    classification = sum(1 for row in rows if row["classification_correct"])
    end_to_end = sum(1 for row in rows if row["end_to_end_correct"])
    fp_records = sum(1 for row in normal if row["false_positive"])
    per_fault_counts = {
        key: sum(1 for row in normal if row["fault_hits"][key]) for key in FAULT_KEYS
    }
    return {
        "n_records": len(rows),
        "n_faulted": len(faulted),
        "n_normal": len(normal),
        "frequency_detection": {
            "hits": frequency_hits,
            "total": len(faulted),
            "rate": _rate(frequency_hits, len(faulted)),
        },
        "classification": {
            "correct": classification,
            "total": len(rows),
            "rate": _rate(classification, len(rows)),
        },
        "end_to_end": {
            "correct": end_to_end,
            "total": len(rows),
            "rate": _rate(end_to_end, len(rows)),
        },
        "false_positives": {
            "records_with_any": fp_records,
            "total_normal": len(normal),
            "rate": _rate(fp_records, len(normal)),
            "per_fault_counts": per_fault_counts,
        },
    }


def _criteria() -> dict[str, Any]:
    """The operational definitions, embedded in the artifact for audit."""
    return {
        "hit_criterion": (
            "A fault hits when its check has detected == true AND "
            "evidence_strength in {'high', 'moderate'}; BSF (ball) ONLY "
            "additionally hits when harmonic 2 is present in the BSF "
            "check's harmonics_detected (CWRU's published rolling-element "
            "factor is 2xBSF). The identical criterion counts detections "
            "on faulted records and false positives on normal baselines."
        ),
        "hit_evidence_levels": sorted(HIT_EVIDENCE_LEVELS),
        "harmonic_leg": {
            "fault": HARMONIC_LEG_FAULT,
            "harmonic": HARMONIC_LEG_ORDER,
        },
        "ranking_rule": (
            "Among faults meeting the hit criterion: evidence tier (high > "
            "moderate > harmonic-only) first, then fundamental peak "
            "magnitude descending (a harmonic-only hit uses the 2x entry's "
            "magnitude), then absolute deviation_pct ascending, then fixed "
            "BPFO/BPFI/BSF/FTF order as the final deterministic tie-break. "
            "Classification is correct when the ranked-first fault maps to "
            "the label; a normal record is correct when no fault meets the "
            "criterion."
        ),
        "label_to_check_key": dict(LABEL_TO_CHECK_KEY),
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
    Fails closed on empty or partial input — outcome ids must match the
    labeled records exactly and every outcome must be ``"ok"``.

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
            labeled ids, id mismatch or non-ok outcomes (ids named), a
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
