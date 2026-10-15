"""The one result contract every audit check writes: ``pmm-audit-result/1``.

Why one contract
----------------
The weekly audit runs checks in separate jobs (stdio smoke, installed-package
smoke, upstream releases, vulnerability audit, test suites) and a final job
turns them into a single report. Parsing each job's log would bind the report
to wording nobody promised to keep, so every check writes this document and
the report builder reads only that, plus pytest JUnit XML and pip-audit JSON.

What a result holds
-------------------
``check`` and ``status`` come from closed vocabularies, so a typo cannot
create a status the report builder silently treats as neither pass nor fail.
``facts`` carry evidence (negotiated protocol versions, tool counts,
timings). ``findings`` are the individual problems or observations, each with
a ``fingerprint`` that identifies the same problem across runs.

Fingerprints
------------
``sha256(category | rule | normalized subject)``, first 12 hex characters.
The caller decides what goes into the subject and keeps versions, timestamps,
line numbers and the matrix variant out of it, so the same problem matches
across runs and across matrix cells. The normalization (strip, lowercase,
collapse whitespace) only absorbs formatting noise.

Local-processing invariant
--------------------------
Raw vibration waveforms never leave the user's machine without consent, and
audit output is uploaded as a workflow artifact. The writer therefore refuses
any numeric array longer than :data:`MAX_NUMERIC_ARRAY` items anywhere in a
document: a waveform or a full spectrum cannot reach a report by accident,
while a handful of peak frequencies still can.

Serialization
-------------
Modelled on ``benchmarks/cwru/runner.py``: a canonicalization pass refuses
non-finite floats and unsupported types with the offending key path named,
then ``json.dumps(sort_keys=True, indent=2, allow_nan=False)`` plus one
trailing newline, written with ``\\n`` line endings on every platform.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Optional, Union

__all__ = [
    "CATEGORIES",
    "CHECK_IDS",
    "CLASSIFICATIONS",
    "DEFAULT_ENVIRONMENT_PACKAGES",
    "EXCERPT_LIMIT",
    "FINGERPRINT_LENGTH",
    "MAX_NUMERIC_ARRAY",
    "REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "SEVERITIES",
    "STATUSES",
    "Finding",
    "Result",
    "canonicalize",
    "environment_snapshot",
    "exit_code_for",
    "fingerprint",
    "normalize_subject",
    "serialize",
    "truncate_evidence",
    "utc_now",
    "validate_result",
    "write_json",
]

RESULT_SCHEMA = "pmm-audit-result/1"
REPORT_SCHEMA = "pmm-audit-report/1"

#: Which check produced a result.
CHECK_IDS = ("smoke", "package-smoke", "upstream", "security", "suite")

#: What kind of problem a finding describes.
CATEGORIES = (
    "regression",
    "install",
    "mcp-compat",
    "tool-contract",
    "doc-drift",
    "diagnostic-correctness",
    "security",
    "dependency",
)

SEVERITIES = ("critical", "high", "medium", "low", "info")

#: Upstream-change classes (normative table in the audit plan); optional on a
#: finding. ``not-checked`` marks evidence that could not be collected.
CLASSIFICATIONS = (
    "harmless",
    "test-gated",
    "review",
    "breaking-candidate",
    "blocked-by-policy",
    "not-checked",
)

#: ``error`` means the check could not run, which is never a pass.
STATUSES = ("pass", "fail", "error")

_EXIT_CODES = {"pass": 0, "fail": 1, "error": 2}

FINGERPRINT_LENGTH = 12

#: Default cap, in characters, for an evidence excerpt (stderr tails, tool
#: output). Long enough for a traceback, short enough to read in a report.
EXCERPT_LIMIT = 4000

#: Longest numeric array a document may contain (see module docstring).
MAX_NUMERIC_ARRAY = 64

#: Distributions whose installed versions describe a check's environment.
DEFAULT_ENVIRONMENT_PACKAGES = (
    "predictive-maintenance-mcp",
    "mcp",
    "mcp-types",
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
    "plotly",
    "scikit-learn",
    "jsonschema",
)


def _require(value: Any, allowed: Iterable[str], name: str) -> None:
    allowed = tuple(allowed)
    if value not in allowed:
        raise ValueError(f"{name} {value!r} is not one of {list(allowed)}")


def normalize_subject(subject: str) -> str:
    """Strip, lowercase and collapse whitespace runs to single spaces."""
    return " ".join(subject.split()).lower()


def fingerprint(category: str, rule: str, subject: str) -> str:
    """Return the cross-run identity of a finding.

    Args:
        category: One of :data:`CATEGORIES`.
        rule: The check-specific rule identifier, e.g. ``tool-missing``.
        subject: What the finding is about. The caller keeps volatile parts
            (versions, timestamps, line numbers, matrix variant) out of it.

    Returns:
        The first :data:`FINGERPRINT_LENGTH` hex characters of
        ``sha256(category|rule|normalized subject)``.
    """
    key = f"{category}|{rule}|{normalize_subject(subject)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def truncate_evidence(
    text: str,
    limit: int = EXCERPT_LIMIT,
    keep: Literal["head", "tail"] = "head",
) -> str:
    """Cap ``text`` at ``limit`` characters with a visible truncation marker.

    Args:
        text: The evidence text.
        limit: Characters of the original text to keep.
        keep: ``head`` keeps the beginning (tool output), ``tail`` keeps the
            end (stderr, where the traceback that matters is last).

    Returns:
        ``text`` unchanged when it fits, otherwise the kept part plus a
        marker naming how many characters were dropped.
    """
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    marker = f"[... truncated {dropped} characters ...]"
    if keep == "tail":
        return f"{marker}\n{text[-limit:]}"
    return f"{text[:limit]}\n{marker}"


def utc_now() -> str:
    """Current time as ISO-8601 UTC with a ``Z`` suffix, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def exit_code_for(status: str) -> int:
    """Map a result status to the audit exit convention (0, 1, 2)."""
    _require(status, STATUSES, "status")
    return _EXIT_CODES[status]


def environment_snapshot(
    packages: Iterable[str] = DEFAULT_ENVIRONMENT_PACKAGES,
) -> dict[str, Any]:
    """Describe the interpreter running the check and its installed packages.

    Args:
        packages: Distribution names to report; a missing one maps to None.

    Returns:
        ``{"python", "implementation", "platform", "packages"}``.
    """
    versions: dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{sys.platform}-{platform.machine()}".lower(),
        "packages": versions,
    }


@dataclass(frozen=True)
class Finding:
    """One problem or observation produced by a check.

    The fingerprint is computed from ``category``, ``rule`` and ``subject``
    at construction; pass it explicitly only when re-reading a stored one.
    """

    rule: str
    category: str
    severity: str
    subject: str
    title: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    classification: Optional[str] = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        _require(self.category, CATEGORIES, "category")
        _require(self.severity, SEVERITIES, "severity")
        if self.classification is not None:
            _require(self.classification, CLASSIFICATIONS, "classification")
        for name in ("rule", "subject", "title"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"finding {name} must be a non-empty string")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("finding evidence must be a mapping")
        expected = fingerprint(self.category, self.rule, self.subject)
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        elif self.fingerprint != expected:
            # A stored fingerprint that no longer matches its own fields would
            # make de-duplication match the wrong issue.
            raise ValueError(
                f"fingerprint {self.fingerprint!r} does not match "
                f"category|rule|subject ({expected!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form; the evidence excerpt is capped (see writer)."""
        doc: dict[str, Any] = {
            "rule": self.rule,
            "category": self.category,
            "severity": self.severity,
            "subject": self.subject,
            "title": self.title,
            "evidence": dict(self.evidence),
            "fingerprint": self.fingerprint,
        }
        if self.classification is not None:
            doc["classification"] = self.classification
        return doc


@dataclass
class Result:
    """A ``pmm-audit-result/1`` document for one run of one check."""

    check: str
    variant: str
    status: str
    started_at: str
    finished_at: str
    environment: Mapping[str, Any] = field(default_factory=dict)
    facts: Mapping[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise ValueError(f"schema {self.schema!r} is not {RESULT_SCHEMA!r}")
        _require(self.check, CHECK_IDS, "check")
        _require(self.status, STATUSES, "status")
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("variant must be a non-empty string")
        for finding in self.findings:
            if not isinstance(finding, Finding):
                raise ValueError("findings must be Finding instances")

    @property
    def exit_code(self) -> int:
        """0 pass, 1 fail, 2 error."""
        return exit_code_for(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "check": self.check,
            "variant": self.variant,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "environment": dict(self.environment),
            "facts": dict(self.facts),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def validate_result(doc: Mapping[str, Any]) -> None:
    """Check a result document read back as plain JSON.

    Raises:
        ValueError: On a wrong schema tag, a missing field, or a value outside
            a closed vocabulary.
    """
    required = (
        "schema",
        "check",
        "variant",
        "status",
        "started_at",
        "finished_at",
        "environment",
        "facts",
        "findings",
    )
    missing = [name for name in required if name not in doc]
    if missing:
        raise ValueError(f"result is missing {missing}")
    if doc["schema"] != RESULT_SCHEMA:
        raise ValueError(f"schema {doc['schema']!r} is not {RESULT_SCHEMA!r}")
    _require(doc["check"], CHECK_IDS, "check")
    _require(doc["status"], STATUSES, "status")
    if not isinstance(doc["findings"], list):
        raise ValueError("findings must be a list")
    for index, item in enumerate(doc["findings"]):
        try:
            Finding(**item)
        except TypeError as exc:
            raise ValueError(
                f"findings[{index}] has unexpected fields: {exc}"
            ) from None
        except ValueError as exc:
            raise ValueError(f"findings[{index}]: {exc}") from None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_array(value: list[Any]) -> bool:
    """A non-empty list of numbers, or of numeric arrays (2-D spectra)."""
    return bool(value) and all(
        _is_number(item) or (isinstance(item, list) and _is_numeric_array(item))
        for item in value
    )


def canonicalize(value: Any, path: str = "document") -> Any:
    """Recursively reduce ``value`` to plain, finite, waveform-free JSON.

    Tuples become lists, :class:`Result`/:class:`Finding` become dicts, and
    any string under an ``excerpt`` key is capped with
    :func:`truncate_evidence`.

    Raises:
        ValueError: On a non-finite float, a non-string key, an unsupported
            type, or a numeric array longer than :data:`MAX_NUMERIC_ARRAY`
            (the offending key path is named).
    """
    if isinstance(value, (Result, Finding)):
        value = value.to_dict()
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"value at '{path}' is {value!r}: non-finite floats have no "
                f"JSON form and indicate a broken measurement"
            )
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"key {key!r} at '{path}' is not a string")
            child = canonicalize(item, f"{path}.{key}")
            if key == "excerpt" and isinstance(child, str):
                child = truncate_evidence(child)
            out[key] = child
        return out
    if isinstance(value, (list, tuple)):
        items = [canonicalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if len(items) > MAX_NUMERIC_ARRAY and _is_numeric_array(items):
            raise ValueError(
                f"numeric array at '{path}' has {len(items)} items (limit "
                f"{MAX_NUMERIC_ARRAY}): raw waveforms and full spectra never "
                f"go into audit output. Summarize it in the producing check."
            )
        return items
    raise ValueError(
        f"value at '{path}' has unsupported type {type(value).__name__}; "
        f"convert it to plain JSON in the producing check"
    )


def serialize(doc: Union[Result, Mapping[str, Any]]) -> str:
    """Render ``doc`` as canonical JSON text ending in one newline.

    A mapping tagged ``pmm-audit-result/1`` is validated as a result first.
    """
    canonical = canonicalize(doc)
    if canonical.get("schema") == RESULT_SCHEMA:
        validate_result(canonical)
    text = json.dumps(canonical, sort_keys=True, indent=2, allow_nan=False)
    return text + "\n"


def write_json(doc: Union[Result, Mapping[str, Any]], path: Union[str, Path]) -> Path:
    """Serialize ``doc`` and write it atomically to ``path``.

    Nothing is written when serialization refuses the document, so a torn or
    half-valid result never reaches the report builder.

    Returns:
        The path written.
    """
    target = Path(path)
    text = serialize(doc)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with open(part, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(part, target)
    return target
