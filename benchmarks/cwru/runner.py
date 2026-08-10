"""Benchmark runner: the deterministic pipeline over the ops view.

Executes ``diagnose_vibration`` as a library over every imported record
with OPERATIONAL context only (signal from the repository under the
opaque id, ``fs``/``rpm`` from the ops table, the catalog bearing id,
the declared unit) and records a per-record outcome subset suitable for
the U5 scorer. No label, filename, or CWRU record number ever reaches
this module: it parses nothing but :class:`~benchmarks.cwru.records.OpsRecord`
instances (origin acceptance example AE3).

Load-bearing decisions, documented here:

- **Import-provenance tripwire first** (:func:`assert_import_provenance`):
  editable installs pin ONE checkout and worktrees silently inherit it,
  so a benchmark run could measure source the maintainer never edited.
  Before any record runs, the imported ``predictive_maintenance_mcp``
  package must resolve to THIS repo's ``src/__init__.py`` (repo root
  derived from this package's own location); otherwise the runner
  refuses with the worktree remedy.
- **Anomaly stage excluded** (:data:`EXCLUDED_ANOMALY_MARKER`): the
  pipeline's anomaly models (``models/*.pkl``) are not versioned, so a
  committed outcome that included their output would bind the
  reproducibility claim to unversioned local state. The block is
  replaced with the fixed marker REGARDLESS of whether a local model
  ran. For the same reason the pipeline-level synthesis fields
  (``overall_diagnosis``/``evidence_strength``/``recommendations``) are
  not carried: they aggregate the anomaly stage.
- **ISO severity as context, not metric**: the ISO block's status (and
  zone, when assessed) is carried for transparency only; the ``g`` unit
  declaration is a documented community-convention assumption, so the
  block is never scored. Likewise ``machine_group=2`` / ``rigid`` are
  the closest ISO 20816-3 categories for the CWRU bench (a small rigid
  motor test rig; its 2 hp motor is in truth below the group-2 15 kW
  floor) — informational context only.
- **Deterministic serialization** (:func:`serialize_outcomes`): a
  canonicalization pass converts NumPy scalars to Python scalars,
  refuses non-finite floats and unsupported types (fail closed), and
  rounds every float to :data:`FLOAT_DECIMALS` (9) decimal places;
  the result is ``json.dumps(..., sort_keys=True, indent=2)`` plus a
  trailing newline, encoded UTF-8. Python's shortest-roundtrip float
  repr makes the bytes a pure function of the rounded values.
- **Determinism claim scope** (:func:`check_determinism`): the
  double-run byte-identity check is scoped to the measured environment
  (one machine, one interpreter, one BLAS). Cross-platform re-runs are
  expected to reproduce metric-level results, not byte-identical
  artifacts — scipy/BLAS low-order float bits differ across builds.

Outcome schema (per opaque id):

- ``status`` — ``"ok"`` | ``"missing_signal"`` | ``"failed"``; non-ok
  entries carry ``error`` and no measurement fields, so a partial run
  is visibly partial (and :func:`assert_outcomes_complete` refuses it).
- ``context`` — operational echo (bearing id, fs, rpm, load, channel,
  unit, ISO group/support, sample count).
- ``bearing`` — the scorer's input: per-fault ``fault_checks`` keyed by
  acronym (BPFO/BPFI/BSF/FTF) with the raw analyzer fields
  (``detected``, ``evidence_strength``, ``detected_frequency_hz``,
  ``harmonics_detected``, ...), plus ``most_likely_fault`` (context —
  the scorer computes its own ranking) and the catalog citation.
- ``iso_severity`` — status/zone context (never scored).
- ``anomaly_detection`` — always :data:`EXCLUDED_ANOMALY_MARKER`.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np

from benchmarks.cwru.importer import SIGNAL_UNIT
from benchmarks.cwru.records import OpsRecord
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    diagnose_vibration,
)
from predictive_maintenance_mcp.signal_acquisition.repository import (
    SignalRepository,
    get_repository,
)

__all__ = [
    "BEARING_ID",
    "DEFAULT_OUTCOMES_PATH",
    "EXCLUDED_ANOMALY_MARKER",
    "EXPECTED_PACKAGE_INIT",
    "FLOAT_DECIMALS",
    "MACHINE_GROUP",
    "OUTCOME_STATUS_FAILED",
    "OUTCOME_STATUS_MISSING_SIGNAL",
    "OUTCOME_STATUS_OK",
    "REPO_ROOT",
    "SUPPORT_TYPE",
    "assert_import_provenance",
    "assert_outcomes_complete",
    "check_determinism",
    "run_records",
    "serialize_outcomes",
    "write_outcomes",
]

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Repo root derived from THIS package's location (benchmarks/cwru/ is
#: two levels below it) — the tree the benchmark claims to measure.
REPO_ROOT: Path = _PACKAGE_DIR.parent.parent

#: Where the measured package must resolve from: this checkout's src
#: tree (``pyproject`` maps ``predictive_maintenance_mcp`` onto src/).
EXPECTED_PACKAGE_INIT: Path = REPO_ROOT / "src" / "__init__.py"

#: Default committed-outcomes path. The results/ directory is created on
#: write; tests always redirect to tmp paths and never touch it.
DEFAULT_OUTCOMES_PATH: Path = _PACKAGE_DIR / "results" / "outcomes.json"

#: Drive-end bearing of the CWRU rig (SKF 6205-2RS JEM), present in the
#: repo's bearing catalog with CWRU-published factors. The v1 subset
#: reads the DE channel only, so one bearing id covers every record.
BEARING_ID: str = "6205"

#: ISO 20816-3 context passed to the pipeline: group 2 (medium
#: machines) on rigid support is the closest category for the CWRU
#: bench — informational only, never scored (see module docstring).
MACHINE_GROUP: int = 2

#: ISO 20816-3 support type for the CWRU rig (motor bolted to a rigid
#: base plate) — informational only, never scored.
SUPPORT_TYPE: str = "rigid"

#: Decimal places every float is rounded to before serialization.
FLOAT_DECIMALS: int = 9

#: Outcome status: the record ran end-to-end.
OUTCOME_STATUS_OK: str = "ok"

#: Outcome status: no imported signal under the record's opaque id.
OUTCOME_STATUS_MISSING_SIGNAL: str = "missing_signal"

#: Outcome status: the pipeline ran but produced no bearing analysis.
OUTCOME_STATUS_FAILED: str = "failed"

#: Fixed marker replacing the anomaly block in every serialized outcome,
#: regardless of local model availability (see module docstring).
#: Read-only so no caller can mutate the committed constant in place.
EXCLUDED_ANOMALY_MARKER: Mapping[str, str] = MappingProxyType(
    {"status": "excluded_unversioned_models"}
)

#: Per-fault fields copied verbatim from each ``check_all_bearing_faults``
#: fault check into the outcome (``fault_type`` itself becomes the key).
_CHECK_FIELDS: tuple[str, ...] = (
    "fault_type_canonical",
    "expected_frequency_hz",
    "detected",
    "detected_frequency_hz",
    "magnitude",
    "deviation_pct",
    "harmonics_detected",
    "evidence_strength",
)


def assert_import_provenance() -> Path:
    """Refuse to measure unless the package resolves from THIS checkout.

    ``pip install -e .`` pins one absolute checkout path into the venv,
    and every git worktree sharing that venv silently imports the
    PRIMARY checkout's source — a benchmark run there would measure a
    tree the maintainer is not editing. This tripwire runs before any
    record (it is the first statement of :func:`run_records`) and is
    exposed publicly so tests and ``__main__`` can call it directly.

    Returns:
        The resolved package ``__file__`` (equal to
        :data:`EXPECTED_PACKAGE_INIT`) on success.

    Raises:
        ValueError: If the imported ``predictive_maintenance_mcp``
            resolves anywhere other than this repo's ``src/__init__.py``
            — with the worktree remedy, before anything is measured.
    """
    import predictive_maintenance_mcp as pkg

    pkg_file = getattr(pkg, "__file__", None)
    resolved = Path(pkg_file).resolve() if pkg_file else None
    expected = EXPECTED_PACKAGE_INIT.resolve()
    if resolved != expected:
        raise ValueError(
            f"Import-provenance tripwire: predictive_maintenance_mcp "
            f"resolved to {resolved if resolved else '<no __file__>'} "
            f"instead of this checkout's {expected} — running now would "
            f"measure a DIFFERENT checkout (editable installs pin one "
            f"tree; worktrees silently inherit it). Run the benchmark "
            f"from the checkout you intend to measure, or give this "
            f"worktree its own environment "
            f"(python scripts/setup-worktree.py), then re-run. Nothing "
            f"was measured."
        )
    return resolved


def _extract_bearing_subset(bearing: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the scorer-relevant subset of a bearing-faults block.

    Args:
        bearing: The ``bearing_faults`` block returned by
            ``check_all_bearing_faults`` via ``diagnose_vibration``.

    Returns:
        Per-fault checks keyed by acronym plus the context fields
        (shaft/characteristic frequencies, ``most_likely`` summary,
        catalog citation).
    """
    fault_checks: dict[str, dict[str, Any]] = {}
    for check in bearing["fault_checks"]:
        fault_checks[str(check["fault_type"])] = {
            field: check[field] for field in _CHECK_FIELDS
        }
    return {
        "shaft_frequency_hz": bearing["shaft_frequency_hz"],
        "bearing_frequencies": bearing["bearing_frequencies"],
        "fault_checks": fault_checks,
        "most_likely_fault": bearing["most_likely_fault"],
        "most_likely_fault_canonical": bearing["most_likely_fault_canonical"],
        "source": bearing["source"],
    }


def _extract_iso_context(iso: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the informational subset of the ISO severity block.

    Carries the status always, plus zone/severity/RMS when assessed and
    the reason when refused — context for the methodology, never a
    scored metric.
    """
    context: dict[str, Any] = {"status": iso.get("status")}
    for key in ("zone", "severity_level", "rms_velocity_mm_s", "reason"):
        if key in iso:
            context[key] = iso[key]
    return context


def _run_one(record: OpsRecord, repository: SignalRepository) -> dict[str, Any]:
    """Run the pipeline for one record and build its outcome entry.

    Args:
        record: The ops record to measure (sole source of fs/rpm/id).
        repository: Repository holding the imported opaque signals.

    Returns:
        An outcome dict — ``status: "ok"`` with the measurement subset,
        or ``status: "missing_signal"``/``"failed"`` with an ``error``
        naming the remedy (the caller distinguishes via ``status``).
    """
    try:
        signal = np.asarray(repository.get_signal(record.opaque_id))
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        return {
            "signal_id": record.opaque_id,
            "status": OUTCOME_STATUS_MISSING_SIGNAL,
            "error": (
                f"No imported signal for record '{record.opaque_id}' in "
                f"the repository ({detail}) — the repository is "
                f"per-process: run the import stage in this process "
                f"(python -m benchmarks.cwru run does so itself), or "
                f"pass a repository that holds the imported signals."
            ),
        }

    result = diagnose_vibration(
        signal,
        fs=float(record.fs_hz),
        rpm=float(record.nominal_rpm),
        signal_id=record.opaque_id,
        bearing_id=BEARING_ID,
        machine_group=MACHINE_GROUP,
        support_type=SUPPORT_TYPE,
        signal_unit=SIGNAL_UNIT,
    )

    bearing = result.get("bearing_faults")
    if bearing is None:
        return {
            "signal_id": record.opaque_id,
            "status": OUTCOME_STATUS_FAILED,
            "error": (
                f"diagnose_vibration produced no bearing analysis for "
                f"record '{record.opaque_id}' (bearing_faults is null; "
                f"the pipeline degrades a failed bearing stage to null) "
                f"— check that bearing '{BEARING_ID}' is in the catalog "
                f"and the signal is analyzable, then re-run."
            ),
        }

    return {
        "signal_id": record.opaque_id,
        "status": OUTCOME_STATUS_OK,
        "context": {
            "bearing_id": BEARING_ID,
            "channel": record.channel,
            "fs_hz": record.fs_hz,
            "load_hp": record.load_hp,
            "machine_group": MACHINE_GROUP,
            "nominal_rpm": record.nominal_rpm,
            "num_samples": int(signal.size),
            "signal_unit": SIGNAL_UNIT,
            "support_type": SUPPORT_TYPE,
        },
        "bearing": _extract_bearing_subset(bearing),
        "iso_severity": _extract_iso_context(result["iso_severity"]),
        # Constant marker, NOT the pipeline's block: models/*.pkl are
        # unversioned, so committed outcomes must not bind to them.
        "anomaly_detection": dict(EXCLUDED_ANOMALY_MARKER),
    }


def run_records(
    records: Iterable[OpsRecord],
    *,
    repository: Optional[SignalRepository] = None,
) -> dict[str, dict[str, Any]]:
    """Run the deterministic pipeline over *records* (ops fields only).

    The import-provenance tripwire runs FIRST — before any repository
    access or measurement. Each record's signal is fetched from the
    repository by its opaque id and fed to ``diagnose_vibration`` with
    operational context only; a record with no imported signal is
    recorded as a ``"missing_signal"`` entry rather than aborting the
    batch, so one broken import does not hide the other measurements
    (the ``all`` gate, :func:`assert_outcomes_complete`, still refuses
    to score a partial set).

    Args:
        records: Ops records to measure, in order.
        repository: Repository override (tests only); defaults to the
            process singleton ``get_repository()``.

    Returns:
        Outcomes keyed by opaque id (schema in the module docstring).

    Raises:
        ValueError: If the provenance tripwire trips, or two records
            share an opaque id.
    """
    assert_import_provenance()
    record_list = tuple(records)
    repo = repository if repository is not None else get_repository()

    outcomes: dict[str, dict[str, Any]] = {}
    for record in record_list:
        if record.opaque_id in outcomes:
            raise ValueError(
                f"Duplicate opaque_id '{record.opaque_id}' in the "
                f"records passed to run_records — outcomes are keyed by "
                f"opaque id, so a duplicate would silently overwrite a "
                f"measurement. Fix the record list (ops_view() already "
                f"enforces uniqueness)."
            )
        outcomes[record.opaque_id] = _run_one(record, repo)
    return outcomes


def _canonicalize(value: object, path: str) -> object:
    """Recursively canonicalize *value* for deterministic serialization.

    NumPy scalars become Python scalars; every float is rounded to
    :data:`FLOAT_DECIMALS` decimal places; non-finite floats, non-string
    mapping keys, and unsupported types are refused fail-closed with the
    offending key path named.

    Args:
        value: The value to canonicalize.
        path: Dotted key path for error messages.

    Returns:
        A canonical structure of dict/list/str/int/float/bool/None.

    Raises:
        ValueError: On non-finite floats, non-string keys, or types
            with no deterministic JSON form.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(
                f"Outcome value at '{path}' is {as_float!r} — non-finite "
                f"floats have no deterministic JSON form and indicate a "
                f"broken measurement. Investigate the producing record "
                f"instead of serializing it."
            )
        return round(as_float, FLOAT_DECIMALS)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key in value:
            if not isinstance(key, str):
                raise ValueError(
                    f"Outcome key {key!r} at '{path}' is not a string — "
                    f"JSON object keys must be strings for sort_keys "
                    f"serialization to be deterministic. Fix the "
                    f"producing code."
                )
            canonical[key] = _canonicalize(value[key], f"{path}.{key}")
        return canonical
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise ValueError(
        f"Outcome value at '{path}' has unsupported type "
        f"{type(value).__name__} — only JSON-representable values are "
        f"serialized (fail closed rather than guessing an encoding). "
        f"Convert it in the producing code."
    )


def serialize_outcomes(outcomes: Mapping[str, Mapping[str, Any]]) -> bytes:
    """Serialize *outcomes* to canonical, newline-terminated JSON bytes.

    Deterministic by construction: the canonicalization pass (see
    :func:`_canonicalize`) rounds floats to :data:`FLOAT_DECIMALS`
    decimal places and normalizes scalar types, then ``json.dumps`` with
    ``sort_keys=True``/``indent=2``/``allow_nan=False`` renders them via
    Python's shortest-roundtrip float repr. Identical outcome values
    therefore always produce identical bytes; the byte-identity claim is
    scoped to the measured environment (see module docstring).

    Args:
        outcomes: Outcomes keyed by opaque id.

    Returns:
        UTF-8 JSON bytes ending in exactly one newline.

    Raises:
        ValueError: Propagated from canonicalization (non-finite float,
            unsupported type, non-string key).
    """
    canonical = _canonicalize(outcomes, path="outcomes")
    text = json.dumps(canonical, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_outcomes(
    outcomes: Mapping[str, Mapping[str, Any]],
    path: Optional[Path] = None,
) -> Path:
    """Serialize *outcomes* and write them atomically to *path*.

    Args:
        outcomes: Outcomes keyed by opaque id.
        path: Destination override; ``None`` uses
            :data:`DEFAULT_OUTCOMES_PATH` (the committed artifact slot —
            tests must always pass a tmp path). Parent directories are
            created; the write goes through a ``.part`` temporary and an
            atomic rename so a crash never leaves a torn artifact.

    Returns:
        The path written.
    """
    target = Path(path) if path is not None else DEFAULT_OUTCOMES_PATH
    payload = serialize_outcomes(outcomes)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(payload)
    os.replace(part, target)
    return target


def check_determinism(
    records: Iterable[OpsRecord],
    *,
    repository: Optional[SignalRepository] = None,
) -> dict[str, dict[str, Any]]:
    """Run the record set twice and require byte-identical serialization.

    The claim this verifies is scoped to the measured environment: one
    machine, one interpreter, one numpy/scipy/BLAS build. Cross-platform
    re-runs are expected to reproduce metric-level results, not
    byte-identical artifacts (low-order float bits differ across
    builds) — the methodology states this explicitly.

    Args:
        records: Ops records to measure (materialized once, run twice).
        repository: Repository override (tests only).

    Returns:
        The first run's outcomes, verified identical to the second.

    Raises:
        ValueError: If the two serializations differ — the environment
            is not producing deterministic outcomes and the artifact
            must not be committed.
    """
    record_list = tuple(records)
    first_outcomes = run_records(record_list, repository=repository)
    first = serialize_outcomes(first_outcomes)
    second = serialize_outcomes(run_records(record_list, repository=repository))
    if first != second:
        raise ValueError(
            "Determinism check failed: two consecutive runs over the "
            "same records serialized to different bytes — the pipeline "
            "is not deterministic in this environment, so the outcomes "
            "artifact must not be committed. Diff the two runs to find "
            "the unstable field (unseeded randomness or ambient state) "
            "and fix it before re-measuring."
        )
    return first_outcomes


def assert_outcomes_complete(
    outcomes: Mapping[str, Mapping[str, Any]],
    records: Iterable[OpsRecord],
) -> None:
    """Fail closed unless every record produced an ``"ok"`` outcome.

    This is the ``all`` dispatch's gate between measuring and scoring:
    a missing or failed record means the outcome set is partial, and a
    partial set scored silently would masquerade as a measured result.

    Args:
        outcomes: Outcomes keyed by opaque id.
        records: The ops records the outcomes must cover exactly.

    Raises:
        ValueError: Naming every absent, non-ok, and unexpected opaque
            id, with the re-run remedy.
    """
    expected = [record.opaque_id for record in records]
    expected_set = set(expected)
    absent = [oid for oid in expected if oid not in outcomes]
    not_ok = [
        oid
        for oid in expected
        if oid in outcomes and outcomes[oid].get("status") != OUTCOME_STATUS_OK
    ]
    unexpected = sorted(set(outcomes) - expected_set)
    if absent or not_ok or unexpected:
        raise ValueError(
            f"Refusing to proceed to scoring on a partial or "
            f"inconsistent outcome set — records with no outcome: "
            f"{absent or 'none'}; records with a non-ok outcome: "
            f"{not_ok or 'none'}; outcomes matching no record: "
            f"{unexpected or 'none'}. Re-run download/import/run for "
            f"the named records until every outcome is 'ok'; scoring a "
            f"partial set would masquerade as a measured result."
        )
