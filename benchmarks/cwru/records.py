"""Vendored CWRU record tables: pydantic models and the ops/label view split.

Two vendored data files form the benchmark's blind-protocol trust boundary
(origin acceptance example AE3):

- ``records_ops.json`` -- operational metadata ONLY (opaque id, download
  URL, sampling rate, nominal speed, load). It is the sole file the
  downloader, importer, and runner ever parse, via :func:`ops_view`.
- ``labels.json`` -- fault labels keyed by opaque id. Parsed only by the
  scorer, via :func:`label_view`.

Because the ops table carries no fault semantics, no stage upstream of the
scorer can leak a label into a filename, signal id, or pipeline argument.
:func:`ops_view` enforces this mechanically: an ops table carrying any
label-bearing key is refused at load, before model validation.

Record enumeration source (official CWRU Bearing Data Center pages,
retrieved 2026-08-10):

- https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
- https://engineering.case.edu/bearingdatacenter/normal-baseline-data

The v1 subset is the full 12 kHz drive-end fault group (60 records; the
official table lists no 0.014" outer-race records at 3:00/12:00 and no
0.028" outer-race records at all) plus the four normal baselines, which
were recorded at 48 kHz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from predictive_maintenance_mcp.path_safety import validate_name_component

__all__ = [
    "LABELS_PATH",
    "LABEL_BEARING_KEYS",
    "OPS_ALLOWED_KEYS",
    "RECORDS_OPS_PATH",
    "LabelRecord",
    "LabeledRecord",
    "OpsRecord",
    "label_view",
    "ops_view",
]

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Vendored operational table -- the only record file the downloader,
#: importer, and runner may parse.
RECORDS_OPS_PATH: Path = _PACKAGE_DIR / "records_ops.json"

#: Vendored label table -- parsed only by the scorer through
#: :func:`label_view`.
LABELS_PATH: Path = _PACKAGE_DIR / "labels.json"

#: Exact key set every ``records_ops.json`` entry must carry -- nothing
#: more, nothing less. The blindness guard tests import this.
OPS_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "opaque_id",
        "file_id",
        "url",
        "internal_mat_key",
        "channel",
        "fs_hz",
        "nominal_rpm",
        "load_hp",
        "cache_filename",
    }
)

#: Keys that carry fault semantics. They live exclusively in
#: ``labels.json``; their presence in the ops table is refused at load.
LABEL_BEARING_KEYS: frozenset[str] = frozenset(
    {
        "fault_type",
        "or_position",
        "fault_diameter_in",
        "sr2015_grade",
        "known_anomalies",
    }
)


class OpsRecord(BaseModel):
    """Operational metadata for one benchmark record (no fault semantics).

    Fields mirror one ``records_ops.json`` entry exactly. ``extra="forbid"``
    turns an unknown key into a validation error rather than a silent drop,
    so the model itself is a second line of defence behind the loader-level
    label-key refusal.

    Attributes:
        opaque_id: Sequential fault-blind identifier (``cwru_001``, ...)
            assigned in table order. Converted signals and their companions
            carry only this name.
        file_id: Numeric ``.mat`` download id on the CWRU site (e.g. 105).
        url: Direct download URL for the record's ``.mat`` file.
        internal_mat_key: MATLAB variable holding the declared channel
            (e.g. ``X105_DE_time``). ``None`` until pinned by the maintainer
            freeze step, when the real files are first inspected.
        channel: Accelerometer channel the benchmark reads.
        fs_hz: Sampling frequency in Hz, declared from the dataset
            documentation (12000 for the fault group, 48000 for the normal
            baselines).
        nominal_rpm: Nominal shaft speed for the record's motor load.
        load_hp: Motor load in horsepower (0-3).
        cache_filename: Filename the downloader stores the raw ``.mat``
            under (single path component, validated).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    opaque_id: str
    file_id: int = Field(gt=0)
    url: str
    internal_mat_key: Optional[str]
    channel: Literal["DE", "FE"]
    fs_hz: int = Field(gt=0)
    nominal_rpm: int = Field(gt=0)
    load_hp: int = Field(ge=0)
    cache_filename: str

    @field_validator("opaque_id")
    @classmethod
    def _opaque_id_is_safe(cls, value: str) -> str:
        """Reject path-hostile ids through the repo's single name guard."""
        # Typed assignment because `predictive_maintenance_mcp.*` resolves as
        # Any under mypy (package-dir mapping, see the overrides section in
        # pyproject.toml) — the real function returns str.
        validated: str = validate_name_component(value, kind="opaque_id")
        return validated

    @field_validator("cache_filename")
    @classmethod
    def _cache_filename_is_safe(cls, value: str) -> str:
        """Reject path-hostile cache filenames (single component only)."""
        validated: str = validate_name_component(value, kind="cache_filename")
        return validated


class LabelRecord(BaseModel):
    """Fault labels for one record. Read only by the scorer.

    Attributes:
        fault_type: Ground-truth fault class for the record.
        or_position: Outer-race fault position relative to the load zone
            (set exactly when ``fault_type`` is ``"outer_race"``).
        fault_diameter_in: Seeded fault diameter in inches (``None`` exactly
            for normal baselines).
        sr2015_grade: Smith & Randall 2015 per-record diagnosability grade.
            ``None`` means not yet transcribed; such records are reported in
            an ``"ungraded"`` stratum, never silently dropped.
        known_anomalies: Documented per-record anomalies (clipping,
            electrical noise, duplicated channels). Filled at freeze, when
            the real files are inspected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fault_type: Literal["normal", "inner_race", "ball", "outer_race"]
    or_position: Optional[Literal["centered_6", "orthogonal_3", "opposite_12"]]
    fault_diameter_in: Optional[float]
    sr2015_grade: Optional[Literal["Y1", "Y2", "P1", "P2", "N1", "N2"]]
    known_anomalies: list[str]

    @model_validator(mode="after")
    def _internally_consistent(self) -> "LabelRecord":
        """Cross-field sanity: position iff outer race, diameter iff faulted."""
        if (self.or_position is not None) != (self.fault_type == "outer_race"):
            raise ValueError(
                f"or_position must be set exactly for outer_race records — got "
                f"fault_type={self.fault_type!r} with "
                f"or_position={self.or_position!r}. Fix the label entry."
            )
        if (self.fault_diameter_in is None) != (self.fault_type == "normal"):
            raise ValueError(
                f"fault_diameter_in must be null exactly for normal records — "
                f"got fault_type={self.fault_type!r} with "
                f"fault_diameter_in={self.fault_diameter_in!r}. Fix the label "
                f"entry."
            )
        return self


class LabeledRecord(BaseModel):
    """One record's ops metadata joined with its fault label (scorer-only)."""

    model_config = ConfigDict(frozen=True)

    ops: OpsRecord
    label: LabelRecord

    @property
    def grade_stratum(self) -> str:
        """Scoring stratum: the S&R 2015 grade, or ``"ungraded"`` when null."""
        return self.label.sr2015_grade or "ungraded"


def _read_json(path: Path, *, description: str) -> object:
    """Read and parse *path*, failing with a problem — remedy message.

    Args:
        path: File to read.
        description: Human-readable table name used in error messages.

    Returns:
        The parsed JSON document.

    Raises:
        ValueError: If the file is missing or not valid JSON.
    """
    if not path.exists():
        raise ValueError(
            f"{description} not found at {path} — the vendored table ships "
            f"with the repository; restore it from git before running the "
            f"benchmark."
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{description} at {path} is not valid JSON ({exc}) — restore "
            f"the vendored table from git."
        ) from exc


def _refuse_label_bearing_keys(entries: list[dict[str, Any]], source: Path) -> None:
    """Refuse to load an ops table that carries any label-bearing key.

    This is the loader-level half of the blind protocol (AE3): even a
    hand-edited ``records_ops.json`` cannot smuggle fault semantics past
    the stages that parse it.

    Args:
        entries: Raw (pre-validation) ops entries.
        source: The file the entries came from, for the error message.

    Raises:
        ValueError: If any entry carries a key from
            :data:`LABEL_BEARING_KEYS`.
    """
    for index, entry in enumerate(entries):
        leaked = LABEL_BEARING_KEYS.intersection(entry)
        if leaked:
            raise ValueError(
                f"Ops table {source} entry {index} "
                f"('{entry.get('opaque_id', '<missing opaque_id>')}') carries "
                f"label-bearing key(s) {sorted(leaked)} — the ops table must "
                f"stay blind. Move fault semantics to labels.json (read only "
                f"by the scorer) and keep ops entries to exactly "
                f"{sorted(OPS_ALLOWED_KEYS)}."
            )


def ops_view(ops_path: Optional[Path] = None) -> tuple[OpsRecord, ...]:
    """Parse the operational record table — and nothing else.

    This is the only record accessor the downloader, importer, and runner
    may use: it never touches ``labels.json``, so the stages feeding the
    system under test structurally cannot see a fault label.

    Args:
        ops_path: Override for the vendored ``records_ops.json`` path
            (tests only).

    Returns:
        The validated records, in table order.

    Raises:
        ValueError: If the table is missing or malformed, carries a
            label-bearing key, fails model validation, or contains
            duplicate opaque ids.
    """
    source = ops_path if ops_path is not None else RECORDS_OPS_PATH
    raw = _read_json(source, description="CWRU ops table")
    if not isinstance(raw, list):
        raise ValueError(
            f"Ops table {source} must be a JSON array of record objects, "
            f"got {type(raw).__name__} — restore the vendored table from "
            f"git."
        )
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Ops table {source} entry {index} is "
                f"{type(entry).__name__}, not an object — restore the "
                f"vendored table from git."
            )
    _refuse_label_bearing_keys(raw, source)

    records: list[OpsRecord] = []
    for index, entry in enumerate(raw):
        try:
            records.append(OpsRecord.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(
                f"Ops table {source} entry {index} "
                f"('{entry.get('opaque_id', '<missing opaque_id>')}') failed "
                f"validation — fix the vendored table. Details: {exc}"
            ) from exc

    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.opaque_id in seen:
            raise ValueError(
                f"Ops table {source} entries {seen[record.opaque_id]} and "
                f"{index} share opaque_id '{record.opaque_id}' — opaque ids "
                f"must be unique; renumber the table."
            )
        seen[record.opaque_id] = index
    return tuple(records)


def label_view(
    ops_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
) -> tuple[LabeledRecord, ...]:
    """Join the ops table with the fault labels. Scorer-only.

    No component upstream of the scorer may call this: it parses
    ``labels.json``, which carries exactly the fault semantics the blind
    protocol keeps away from the system under test.

    Args:
        ops_path: Override for the vendored ``records_ops.json`` path
            (tests only).
        labels_path: Override for the vendored ``labels.json`` path
            (tests only).

    Returns:
        One joined record per ops entry, in ops-table order.

    Raises:
        ValueError: If either table is invalid, or the two tables do not
            cover exactly the same opaque ids.
    """
    ops_records = ops_view(ops_path)
    source = labels_path if labels_path is not None else LABELS_PATH
    raw = _read_json(source, description="CWRU label table")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Label table {source} must be a JSON object keyed by opaque_id, "
            f"got {type(raw).__name__} — restore the vendored table from "
            f"git."
        )

    ops_ids = {record.opaque_id for record in ops_records}
    label_ids = set(raw)
    missing = sorted(ops_ids - label_ids)
    orphaned = sorted(label_ids - ops_ids)
    if missing or orphaned:
        raise ValueError(
            f"Label table {source} does not match the ops table — every "
            f"record needs exactly one label. Ops records without a label: "
            f"{missing or 'none'}; labels without an ops record: "
            f"{orphaned or 'none'}. Fix the vendored tables together."
        )

    joined: list[LabeledRecord] = []
    for record in ops_records:
        try:
            label = LabelRecord.model_validate(raw[record.opaque_id])
        except ValidationError as exc:
            raise ValueError(
                f"Label table {source} entry '{record.opaque_id}' failed "
                f"validation — fix the vendored table. Details: {exc}"
            ) from exc
        joined.append(LabeledRecord(ops=record, label=label))
    return tuple(joined)
