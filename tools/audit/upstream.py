"""Compare the compatibility inventory with official upstream release data.

``python -m tools.audit upstream`` reads ``compat_inventory.toml`` (what a
maintainer has reviewed), fetches release data for each component, and
classifies every difference with the normative table of the audit plan
(KTD7). It writes one ``pmm-audit-result/1`` result with ``check="upstream"``
and never edits ``pyproject.toml`` or ``uv.lock``: a human reviews, then
bumps the inventory's ``reviewed`` field by hand.

Component kinds
---------------
``pypi``
    PyPI JSON API. Yanked releases are dropped and pre-releases kept apart.
    "Latest" is computed per interpreter from each release's files'
    ``requires_python``, because numpy 2.5 and scipy 1.18 already left
    Python 3.11 while the project still supports it. Declared ranges come
    from ``[project.dependencies]`` at run time, never from the inventory.
``sdk-protocol``
    The protocol revisions the *installed* MCP SDK knows
    (``mcp.types.version.KNOWN_PROTOCOL_VERSIONS``), so run this in the
    newest resolution to see what an upgrade would bring.
``github-release``
    Newest non-draft release through the GitHub REST API (``GITHUB_TOKEN``
    is sent when set, to api.github.com only).
``manual``
    Nothing to fetch; ``reviewed_on + review_interval_days`` is compared
    with today.

Classification (rules per ``rules`` value)
------------------------------------------
=====================  =======================  ===================  ============
step over reviewed     library                  mcp-sdk              build-tool
=====================  =======================  ===================  ============
new major              breaking-candidate       breaking-candidate   test-gated
new minor              test-gated               review               (harmless)
new patch              test-gated               test-gated           (harmless)
=====================  =======================  ===================  ============

* For a ``0.x`` version the minor is the breaking step (uv's policy and
  SemVer section 4), so 0.13 -> 0.14 counts as a new major.
* A release newer than reviewed but outside the declared range (mcp 3.x
  final or pre-release against ``<3.0.0``) is ``blocked-by-policy``. Older
  lines (mcp 1.x maintenance releases) are ignored.
* ``library`` and ``mcp-sdk``: when the newest in-range release's
  ``requires_python`` excludes an inventory interpreter, a ``review``
  "matrix split" finding names the interpreters left on an older line.
* A deprecation listed in the inventory whose due date is within
  :data:`DEPRECATION_WINDOW_DAYS` is ``review`` (low).
* ``test-gated`` findings are emitted unresolved: this check never sees
  suite results. The report builder resolves them from the legs that
  installed one of ``evidence.candidates`` of ``evidence.package``.
* Latest equal to reviewed is no finding; the component's facts record
  ``outcome: harmless``.

Subjects (KTD11): ``review``, ``breaking-candidate`` and
``blocked-by-policy`` findings carry the upstream ``major.minor`` (``mcp
2.4``), so each new minor is reviewed once; ``test-gated`` subjects are the
bare package name.

Failure handling
----------------
A fetch that raises, or a payload without the expected shape, becomes a
``not-checked`` finding (medium) for that component only; the others are
still classified and the result status is ``error`` (exit 2). The same holds
for an installed SDK whose protocol constants moved, and for a declared
runtime dependency with no inventory entry.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Union

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tools.audit import deps, results

__all__ = [
    "DEFAULT_INVENTORY",
    "DEPRECATION_WINDOW_DAYS",
    "INVENTORY_SCHEMA",
    "KINDS",
    "PROTOCOL_LOCATIONS",
    "ROLES",
    "RULESETS",
    "Component",
    "Outcome",
    "ProtocolProbe",
    "add_arguments",
    "check_deprecations",
    "check_upstream",
    "classify_github",
    "classify_manual",
    "classify_protocol",
    "classify_pypi",
    "github_releases_url",
    "http_fetcher",
    "inventory_parity",
    "load_inventory",
    "parse_inventory",
    "pypi_url",
    "request_headers",
    "run",
    "sdk_protocol_versions",
]

INVENTORY_SCHEMA = "pmm-compat-inventory/1"
DEFAULT_INVENTORY = Path(__file__).with_name("compat_inventory.toml")

KINDS = ("pypi", "sdk-protocol", "github-release", "manual")
#: Which classification column of the module table a pypi component uses.
RULESETS = ("library", "mcp-sdk", "build-tool")
#: ``runtime`` entries must match ``[project.dependencies]`` both ways.
ROLES = ("runtime", "test", "tooling")

DEPRECATION_WINDOW_DAYS = 120

#: Where the installed SDK keeps its protocol revisions, newest layout first:
#: mcp 2.x re-exports ``mcp_types.version`` as ``mcp.types.version``; 1.x
#: kept a shorter list in ``mcp.shared.version``.
PROTOCOL_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("mcp.types.version", "KNOWN_PROTOCOL_VERSIONS"),
    ("mcp_types.version", "KNOWN_PROTOCOL_VERSIONS"),
    ("mcp.shared.version", "SUPPORTED_PROTOCOL_VERSIONS"),
)

USER_AGENT = (
    "predictive-maintenance-mcp-audit/1 "
    "(+https://github.com/LGDiMaggio/predictive-maintenance-mcp)"
)
FETCH_TIMEOUT_SECONDS = 30
GITHUB_API = "https://api.github.com/"
GITHUB_RELEASES_PER_PAGE = 30

_RC_SUFFIX = re.compile(r"[-.]?rc\d*$", re.IGNORECASE)

_COMMON_KEYS = {
    "id",
    "kind",
    "source",
    "reviewed",
    "reviewed_on",
    "review_interval_days",
    "floating",
    "note",
    "category",
    "deprecations",
    "details",
}
_KIND_KEYS = {
    "pypi": {"package", "rules", "role", "interpreters", "lockstep"},
    "sdk-protocol": set(),
    "github-release": {"repo"},
    "manual": set(),
}
_DEFAULT_CATEGORY = {"sdk-protocol": "mcp-compat"}

Fetcher = Callable[[str], Any]


class PayloadError(ValueError):
    """Upstream data arrived but not in the shape the checker reads."""


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Deprecation:
    """An upstream deprecation with a due date the project must act on."""

    id: str
    due: date
    note: str = ""
    url: str = ""


@dataclass(frozen=True)
class Component:
    """One inventory entry, validated."""

    id: str
    kind: str
    source: str
    reviewed: Union[str, tuple[str, ...], None]
    reviewed_on: date
    note: str
    category: str
    review_interval_days: Optional[int] = None
    floating: bool = False
    package: Optional[str] = None
    rules: Optional[str] = None
    role: Optional[str] = None
    interpreters: tuple[str, ...] = ()
    lockstep: tuple[str, ...] = ()
    repo: Optional[str] = None
    deprecations: tuple[Deprecation, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


def _as_date(value: Any, where: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{where} must be a date (YYYY-MM-DD), got {value!r}")


def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} must be a list of strings")
    return tuple(value)


def _require_str(entry: Mapping[str, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return value


def _parse_component(
    entry: Any, index: int, interpreters: tuple[str, ...]
) -> Component:
    where = f"component[{index}]"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where} is not a table")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{where}: kind {kind!r} is not one of {list(KINDS)}")
    unknown = set(entry) - _COMMON_KEYS - _KIND_KEYS[kind]
    if unknown:
        # Closed key set: a typo must not silently disable a check, and
        # declared ranges are read from pyproject.toml, never copied here.
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} for kind {kind}")
    cid = _require_str(entry, "id", where)
    where = f"component {cid!r}"
    category = entry.get("category", _DEFAULT_CATEGORY.get(kind, "dependency"))
    if category not in results.CATEGORIES:
        raise ValueError(f"{where}: category {category!r} is not a result category")

    reviewed: Union[str, tuple[str, ...], None] = entry.get("reviewed")
    if kind == "sdk-protocol":
        reviewed = _str_tuple(reviewed, f"{where}.reviewed")
    elif kind in ("pypi", "github-release"):
        reviewed = _require_str(entry, "reviewed", where)
    elif reviewed is not None and not isinstance(reviewed, str):
        reviewed = _str_tuple(reviewed, f"{where}.reviewed")

    interval = entry.get("review_interval_days")
    if interval is not None and (
        not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0
    ):
        raise ValueError(f"{where}: review_interval_days must be a positive integer")
    floating = entry.get("floating", False)
    if not isinstance(floating, bool):
        raise ValueError(f"{where}: floating must be true or false")
    details = entry.get("details", {})
    if not isinstance(details, Mapping):
        raise ValueError(f"{where}: details must be a table")

    deprecations = []
    for i, item in enumerate(entry.get("deprecations", [])):
        dwhere = f"{where}.deprecations[{i}]"
        if not isinstance(item, Mapping) or set(item) - {"id", "due", "note", "url"}:
            raise ValueError(f"{dwhere} must be a table of id, due, note, url")
        deprecations.append(
            Deprecation(
                id=_require_str(item, "id", dwhere),
                due=_as_date(item.get("due"), f"{dwhere}.due"),
                note=str(item.get("note", "")),
                url=str(item.get("url", "")),
            )
        )

    fields: dict[str, Any] = {}
    if kind == "pypi":
        package = _require_str(entry, "package", where)
        try:
            Version(str(reviewed))
        except InvalidVersion:
            raise ValueError(
                f"{where}: reviewed {reviewed!r} is not a version"
            ) from None
        rules = entry.get("rules")
        if rules not in RULESETS:
            raise ValueError(f"{where}: rules {rules!r} is not one of {list(RULESETS)}")
        role = entry.get("role")
        if role not in ROLES:
            raise ValueError(f"{where}: role {role!r} is not one of {list(ROLES)}")
        own = entry.get("interpreters")
        fields.update(
            package=package,
            rules=rules,
            role=role,
            interpreters=(
                _str_tuple(own, f"{where}.interpreters")
                if own is not None
                else interpreters
            ),
            lockstep=_str_tuple(entry.get("lockstep", []), f"{where}.lockstep"),
        )
        if not fields["interpreters"]:
            raise ValueError(f"{where}: no interpreters to evaluate")
    if kind == "github-release":
        repo = _require_str(entry, "repo", where)
        if repo.count("/") != 1:
            raise ValueError(f"{where}: repo must be 'owner/name', got {repo!r}")
        fields["repo"] = repo

    return Component(
        id=cid,
        kind=kind,
        source=_require_str(entry, "source", where),
        reviewed=reviewed,
        reviewed_on=_as_date(entry.get("reviewed_on"), f"{where}.reviewed_on"),
        note=str(entry.get("note", "")),
        category=category,
        review_interval_days=interval,
        floating=floating,
        deprecations=tuple(deprecations),
        details=dict(details),
        **fields,
    )


def parse_inventory(doc: Mapping[str, Any]) -> list[Component]:
    """Validate a parsed inventory document.

    Raises:
        ValueError: On a wrong schema tag, an unknown kind or key, a missing
            field, or a duplicate id.
    """
    if doc.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"inventory schema must be {INVENTORY_SCHEMA!r}")
    defaults = doc.get("defaults", {})
    interpreters = _str_tuple(defaults.get("interpreters", []), "defaults.interpreters")
    entries = doc.get("component")
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory has no [[component]] entries")
    components = [
        _parse_component(entry, index, interpreters)
        for index, entry in enumerate(entries)
    ]
    ids = [component.id for component in components]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise ValueError(f"duplicate component ids: {duplicates}")
    return components


def load_inventory(path: Union[str, Path] = DEFAULT_INVENTORY) -> list[Component]:
    """Read and validate ``compat_inventory.toml``."""
    return parse_inventory(deps.load_toml(path))


def inventory_parity(
    components: Sequence[Component], requirements: Sequence[Requirement]
) -> tuple[list[str], list[str]]:
    """Compare runtime pypi entries with ``[project.dependencies]`` both ways.

    Returns:
        ``(missing, extra)``: declared names without a runtime entry, and
        runtime entries naming no declared dependency (PEP 503 names).
    """
    declared = {canonicalize_name(r.name) for r in requirements}
    runtime = {
        canonicalize_name(c.package)
        for c in components
        if c.kind == "pypi" and c.role == "runtime" and c.package
    }
    return sorted(declared - runtime), sorted(runtime - declared)


# --------------------------------------------------------------------------
# Classification helpers
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    """What one component contributed: findings plus facts."""

    findings: list[results.Finding] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Release:
    version: Version
    requires_python: tuple[Optional[str], ...]
    yanked: bool

    def admits(self, interpreter: str) -> bool:
        """Whether any file of this release installs on ``interpreter``."""
        for spec in self.requires_python:
            if not spec:
                return True
            try:
                if SpecifierSet(spec).contains(interpreter, prereleases=True):
                    return True
            except InvalidSpecifier:
                # pip ignores an invalid Requires-Python, so the file installs.
                return True
        return False

    @property
    def requires_text(self) -> str:
        specs = sorted({spec for spec in self.requires_python if spec})
        return " | ".join(specs) or "any"


def parse_pypi(payload: Any) -> list[Release]:
    """Releases of a PyPI JSON API document, invalid versions skipped.

    A release whose every file is yanked counts as yanked; one without
    files (old metadata-only entries) is dropped.

    Raises:
        PayloadError: When ``releases`` is missing or malformed.
    """
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("releases"), Mapping
    ):
        raise PayloadError("PyPI payload has no 'releases' mapping")
    out = []
    for text, files in payload["releases"].items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if not isinstance(files, list):
            raise PayloadError(f"PyPI release {text!r} files are not a list")
        if not files:
            continue
        if not all(isinstance(f, Mapping) for f in files):
            raise PayloadError(f"PyPI release {text!r} has a malformed file entry")
        live = [f for f in files if not f.get("yanked")]
        considered = live or files
        out.append(
            Release(
                version=version,
                requires_python=tuple(f.get("requires_python") for f in considered),
                yanked=not live,
            )
        )
    return out


def _line(version: Version) -> tuple[int, ...]:
    """The breaking-change unit: the major, or the minor under ``0.x``."""
    return (0, version.minor) if version.major == 0 else (version.major,)


def _step(reviewed: Version, candidate: Version) -> str:
    if _line(candidate) != _line(reviewed):
        return "major"
    if candidate.minor != reviewed.minor:
        return "minor"
    return "patch"


def _major_minor(version: Version) -> str:
    return f"{version.major}.{version.minor}"


#: (classification, severity, subject carries major.minor) per ruleset/step.
_RULE_TABLE: dict[str, dict[str, Optional[tuple[str, str, bool]]]] = {
    "library": {
        "major": ("breaking-candidate", "medium", True),
        "minor": ("test-gated", "low", False),
        "patch": ("test-gated", "low", False),
    },
    "mcp-sdk": {
        "major": ("breaking-candidate", "medium", True),
        "minor": ("review", "medium", True),
        "patch": ("test-gated", "low", False),
    },
    "build-tool": {
        "major": ("test-gated", "low", False),
        "minor": None,
        "patch": None,
    },
}

_STEP_TITLES = {
    "breaking-candidate": "{pkg} {versions} is a new major over reviewed "
    "{reviewed}{reachable}",
    "review": "{pkg} {versions} is a new minor over reviewed {reviewed}: "
    "review its behavior changes",
    "test-gated": "{pkg} {versions} is newer than reviewed {reviewed} "
    "({step}): the newest suite and package-smoke legs decide",
}


def _not_checked(component: Component, url: str, error: str) -> results.Finding:
    return results.Finding(
        rule="not-checked",
        category=component.category,
        severity="medium",
        subject=component.id,
        title=f"{component.id}: upstream release data could not be checked",
        evidence={"url": url, "error": results.truncate_evidence(error)},
        classification="not-checked",
    )


# --------------------------------------------------------------------------
# Per-kind classification
# --------------------------------------------------------------------------


def classify_pypi(
    component: Component,
    payload: Any,
    declared: Optional[SpecifierSet],
) -> Outcome:
    """Classify a pypi component against its PyPI JSON document.

    Args:
        component: A ``pypi`` inventory entry.
        payload: Parsed ``https://pypi.org/pypi/<package>/json``.
        declared: The package's specifier from ``[project.dependencies]``,
            or None for components that are not runtime dependencies.

    Raises:
        PayloadError: When the payload is malformed.
    """
    package = canonicalize_name(component.package or component.id)
    reviewed = Version(str(component.reviewed))
    releases = [r for r in parse_pypi(payload) if not r.yanked]
    finals = [r for r in releases if not r.version.is_prerelease]

    def in_range(version: Version) -> bool:
        return declared is None or declared.contains(version, prereleases=True)

    candidates = [r for r in finals if in_range(r.version)]
    latest: dict[str, Optional[Version]] = {}
    for interpreter in component.interpreters:
        admitted = [r.version for r in candidates if r.admits(interpreter)]
        latest[interpreter] = max(admitted) if admitted else None
    newest = max(candidates, key=lambda r: r.version, default=None)
    # Only a pre-release ahead of the newest final is news (pandas 3.1.0rc0).
    prereleases = [
        r.version
        for r in releases
        if r.version.is_prerelease and (newest is None or r.version > newest.version)
    ]
    declared_text = str(declared) if declared is not None else None

    findings: list[results.Finding] = []
    base_evidence: dict[str, Any] = {
        "package": package,
        "reviewed": str(reviewed),
        "declared": declared_text,
        "latest": {k: str(v) if v else None for k, v in latest.items()},
    }
    if component.lockstep:
        base_evidence["lockstep"] = list(component.lockstep)

    # Newer than reviewed, per (classification, subject); merged across
    # interpreters so one upstream line is one finding.
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    harmless_steps: list[str] = []
    for interpreter, version in latest.items():
        if version is None or version <= reviewed:
            continue
        step = _step(reviewed, version)
        rule = _RULE_TABLE[component.rules or "library"][step]
        if rule is None:
            harmless_steps.append(f"{version} ({step})")
            continue
        classification, severity, with_line = rule
        subject = f"{package} {_major_minor(version)}" if with_line else package
        group = groups.setdefault(
            (f"upstream-{step}", subject),
            {
                "classification": classification,
                "severity": severity,
                "step": step,
                "versions": set(),
                "interpreters": [],
            },
        )
        group["versions"].add(version)
        group["interpreters"].append(interpreter)
    for (rule_id, subject), group in sorted(groups.items()):
        versions = sorted(group["versions"])
        title = _STEP_TITLES[group["classification"]].format(
            pkg=package,
            reachable=(
                f", reachable within the declared range {declared_text}"
                if declared_text
                else ""
            ),
            reviewed=reviewed,
            versions=", ".join(str(v) for v in versions),
            step=group["step"],
        )
        findings.append(
            results.Finding(
                rule=rule_id,
                category=component.category,
                severity=group["severity"],
                subject=subject,
                title=title,
                evidence={
                    **base_evidence,
                    "candidates": [str(v) for v in versions],
                    "interpreters": group["interpreters"],
                },
                classification=group["classification"],
            )
        )

    # Published beyond the declared range: a cap is a policy decision.
    if declared is not None:
        ceiling = max(reviewed, newest.version) if newest else reviewed
        above = sorted(
            r.version
            for r in releases
            if not in_range(r.version) and r.version > ceiling
        )
        if above:
            top = above[-1]
            findings.append(
                results.Finding(
                    rule="above-declared-range",
                    category=component.category,
                    severity="medium",
                    subject=f"{package} {_major_minor(top)}",
                    title=(
                        f"{package} {top} is published outside the declared range "
                        f"{declared_text}: blocked by policy, read its migration "
                        f"guide before widening"
                    ),
                    evidence={
                        **base_evidence,
                        "candidates": [str(v) for v in above[-10:]],
                    },
                    classification="blocked-by-policy",
                )
            )

    # Matrix split: the newest release leaves a supported interpreter behind.
    if component.rules in ("library", "mcp-sdk") and newest is not None:
        excluded = [i for i in component.interpreters if not newest.admits(i)]
        if excluded and newest.version > reviewed:
            findings.append(
                results.Finding(
                    rule="matrix-split",
                    category=component.category,
                    severity="info",
                    subject=f"{package} {_major_minor(newest.version)}",
                    title=(
                        f"{package} {newest.version} requires Python "
                        f"{newest.requires_text}; on {', '.join(excluded)} the "
                        f"resolver stays on an older line"
                    ),
                    evidence={
                        **base_evidence,
                        "requires_python": newest.requires_text,
                        "excluded_interpreters": excluded,
                    },
                    classification="review",
                )
            )

    facts: dict[str, Any] = {
        "kind": component.kind,
        "package": package,
        "reviewed": str(reviewed),
        "declared": declared_text,
        "latest": base_evidence["latest"],
        "latest_prerelease": str(max(prereleases)) if prereleases else None,
        "outcome": "findings" if findings else "harmless",
    }
    if harmless_steps:
        facts["harmless"] = harmless_steps
    return Outcome(findings=findings, facts=facts)


def check_deprecations(component: Component, today: date) -> list[results.Finding]:
    """Deprecations whose due date is within the window (or past) -> review."""
    findings = []
    for item in component.deprecations:
        days_left = (item.due - today).days
        if days_left > DEPRECATION_WINDOW_DAYS:
            continue
        findings.append(
            results.Finding(
                rule="deprecation-due",
                category=component.category,
                severity="low",
                subject=f"{component.id} {item.id}",
                title=(
                    f"{component.id}: deprecation '{item.id}' is due "
                    f"{item.due.isoformat()} ({days_left} days)"
                ),
                evidence={
                    "due": item.due.isoformat(),
                    "days_left": days_left,
                    "note": item.note,
                    "url": item.url,
                },
                classification="review",
            )
        )
    return findings


@dataclass(frozen=True)
class ProtocolProbe:
    """What the installed SDK says about protocol revisions."""

    versions: Optional[tuple[str, ...]]
    source: str
    sdk_version: Optional[str]


def sdk_protocol_versions() -> ProtocolProbe:
    """Read the installed SDK's protocol revisions without ever raising."""
    try:
        sdk_version: Optional[str] = metadata.version("mcp")
    except metadata.PackageNotFoundError:
        sdk_version = None
    tried = []
    for module_name, attribute in PROTOCOL_LOCATIONS:
        location = f"{module_name}.{attribute}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # any import failure means "not here"
            tried.append(f"{location} ({type(exc).__name__})")
            continue
        value = getattr(module, attribute, None)
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, str) for v in value)
        ):
            return ProtocolProbe(tuple(value), location, sdk_version)
        tried.append(f"{location} (absent)")
    return ProtocolProbe(None, "tried " + ", ".join(tried), sdk_version)


def classify_protocol(component: Component, probe: ProtocolProbe) -> Outcome:
    """New protocol revisions in the installed SDK -> review (one each)."""
    reviewed = tuple(component.reviewed or ())
    if probe.versions is None:
        finding = results.Finding(
            rule="protocol-constants-missing",
            category=component.category,
            severity="medium",
            subject=component.id,
            title=(
                f"{component.id}: the installed MCP SDK exposes no known protocol "
                f"revisions where the checker looks"
            ),
            evidence={"sdk_version": probe.sdk_version, "tried": probe.source},
            classification="not-checked",
        )
        return Outcome(
            [finding],
            {
                "kind": component.kind,
                "sdk_version": probe.sdk_version,
                "outcome": "not-checked",
            },
        )
    new = [rev for rev in probe.versions if rev not in reviewed]
    dropped = [rev for rev in reviewed if rev not in probe.versions]
    findings = [
        results.Finding(
            rule="protocol-revision",
            category=component.category,
            severity="medium",
            subject=f"{component.id} {revision}",
            title=(
                f"MCP SDK {probe.sdk_version} knows protocol revision {revision}, "
                f"which the inventory has not reviewed"
            ),
            evidence={
                "sdk_version": probe.sdk_version,
                "source": probe.source,
                "known": list(probe.versions),
                "reviewed": list(reviewed),
            },
            classification="review",
        )
        for revision in new
    ]
    facts = {
        "kind": component.kind,
        "sdk_version": probe.sdk_version,
        "source": probe.source,
        "known": list(probe.versions),
        "new": new,
        "dropped": dropped,
        "outcome": "findings" if findings else "harmless",
    }
    return Outcome(findings, facts)


def _published(release: Mapping[str, Any]) -> str:
    return str(release.get("published_at") or release.get("created_at") or "")


def _is_candidate(release: Mapping[str, Any]) -> bool:
    return bool(release.get("prerelease")) or bool(
        _RC_SUFFIX.search(str(release.get("tag_name", "")))
    )


def _tag_line(tag: str) -> str:
    """``v1.8.1`` -> ``1.8``; ``2026-07-28-RC`` -> ``2026-07-28``."""
    stripped = _RC_SUFFIX.sub("", tag)
    try:
        return _major_minor(Version(stripped))
    except InvalidVersion:
        return stripped


def classify_github(component: Component, payload: Any) -> Outcome:
    """Classify a github-release component against its releases list.

    Raises:
        PayloadError: When the payload is not a list of release objects.
    """
    if not isinstance(payload, list) or not all(
        isinstance(r, Mapping) and isinstance(r.get("tag_name"), str) for r in payload
    ):
        raise PayloadError("GitHub payload is not a list of releases")
    published = sorted(
        (r for r in payload if not r.get("draft")), key=_published, reverse=True
    )
    reviewed = str(component.reviewed)
    newest = published[0] if published else None
    newest_final = next((r for r in published if not _is_candidate(r)), None)
    findings = []
    if newest is not None and _is_candidate(newest) and newest["tag_name"] != reviewed:
        tag = newest["tag_name"]
        findings.append(
            results.Finding(
                rule="release-candidate",
                category=component.category,
                severity="low",
                subject=f"{component.id} {_tag_line(tag)}",
                title=f"{component.repo} published release candidate {tag}",
                evidence={"repo": component.repo, "tag": tag, "reviewed": reviewed},
                classification="review",
            )
        )
    if newest_final is not None and newest_final["tag_name"] != reviewed:
        tag = newest_final["tag_name"]
        if component.floating:
            findings.append(
                results.Finding(
                    rule="floating-ref-moved",
                    category=component.category,
                    severity="info",
                    subject=component.id,
                    title=(
                        f"{component.repo} released {tag}; the floating ref picks "
                        f"it up (reviewed {reviewed})"
                    ),
                    evidence={"repo": component.repo, "tag": tag, "reviewed": reviewed},
                    classification="harmless",
                )
            )
        else:
            findings.append(
                results.Finding(
                    rule="release-newer",
                    category=component.category,
                    severity="medium",
                    subject=f"{component.id} {_tag_line(tag)}",
                    title=f"{component.repo} released {tag} (reviewed {reviewed})",
                    evidence={"repo": component.repo, "tag": tag, "reviewed": reviewed},
                    classification="review",
                )
            )
    facts = {
        "kind": component.kind,
        "repo": component.repo,
        "reviewed": reviewed,
        "newest": newest["tag_name"] if newest else None,
        "newest_final": newest_final["tag_name"] if newest_final else None,
        "floating": component.floating,
        "outcome": "findings" if findings else "harmless",
    }
    return Outcome(findings, facts)


def classify_manual(component: Component, today: date) -> Outcome:
    """A manual entry past ``reviewed_on + review_interval_days`` -> low review."""
    due = (
        component.reviewed_on + timedelta(days=component.review_interval_days)
        if component.review_interval_days
        else None
    )
    findings = []
    if due is not None and today > due:
        findings.append(
            results.Finding(
                rule="review-overdue",
                category=component.category,
                severity="low",
                subject=component.id,
                title=(
                    f"{component.id}: manual review was due {due.isoformat()} "
                    f"(last reviewed {component.reviewed_on.isoformat()})"
                ),
                evidence={
                    "reviewed_on": component.reviewed_on.isoformat(),
                    "review_interval_days": component.review_interval_days,
                    "source": component.source,
                },
                classification="review",
            )
        )
    facts = {
        "kind": component.kind,
        "reviewed_on": component.reviewed_on.isoformat(),
        "review_due": due.isoformat() if due else None,
        "outcome": "findings" if findings else "harmless",
    }
    return Outcome(findings, facts)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def pypi_url(package: str) -> str:
    return f"https://pypi.org/pypi/{canonicalize_name(package)}/json"


def github_releases_url(repo: str) -> str:
    return f"{GITHUB_API}repos/{repo}/releases?per_page={GITHUB_RELEASES_PER_PAGE}"


def request_headers(url: str) -> dict[str, str]:
    """Headers for ``url``; the GitHub token goes to api.github.com only."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if url.startswith(GITHUB_API):
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def http_fetcher(url: str) -> Any:
    """GET ``url`` and parse JSON (stdlib only, bounded by a timeout)."""
    request = urllib.request.Request(url, headers=request_headers(url))
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.load(response)


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def _fetch_and_classify(
    component: Component,
    url: str,
    fetcher: Fetcher,
    classify: Callable[[Any], Outcome],
) -> Outcome:
    failed = {"kind": component.kind, "url": url, "outcome": "not-checked"}
    try:
        payload = fetcher(url)
    except Exception as exc:  # any failure at the network boundary
        error = f"{type(exc).__name__}: {exc}"
        return Outcome([_not_checked(component, url, error)], failed)
    try:
        return classify(payload)
    except PayloadError as exc:
        error = f"unexpected payload: {exc}"
        return Outcome([_not_checked(component, url, error)], failed)


def _parity_findings(
    components: Sequence[Component], requirements: Sequence[Requirement]
) -> list[results.Finding]:
    missing, extra = inventory_parity(components, requirements)
    findings = [
        results.Finding(
            rule="inventory-parity",
            category="dependency",
            severity="medium",
            subject=name,
            title=f"{name} is a declared dependency with no inventory entry: unchecked",
            evidence={"direction": "missing-entry"},
            classification="not-checked",
        )
        for name in missing
    ]
    findings += [
        results.Finding(
            rule="inventory-parity",
            category="dependency",
            severity="low",
            subject=name,
            title=f"{name} has a runtime inventory entry but is not declared",
            evidence={"direction": "undeclared-entry"},
            classification="review",
        )
        for name in extra
    ]
    return findings


def check_upstream(
    components: Sequence[Component],
    *,
    requirements: Sequence[Requirement],
    fetcher: Fetcher,
    today: date,
    protocol_probe: Callable[[], ProtocolProbe] = sdk_protocol_versions,
    variant: str = "upstream",
) -> results.Result:
    """Classify every component and return one ``upstream`` result.

    Args:
        components: Parsed inventory.
        requirements: ``[project.dependencies]``, read live.
        fetcher: URL -> parsed JSON; injected so tests stay offline.
        today: Date for manual reviews and deprecation windows.
        protocol_probe: Reads the installed SDK's protocol revisions.
        variant: Free label for the run.
    """
    started_at = results.utc_now()
    declared = {canonicalize_name(r.name): r.specifier for r in requirements}
    findings = _parity_findings(components, requirements)
    component_facts: dict[str, Any] = {}
    for component in components:
        if component.kind == "pypi":
            spec = (
                declared.get(canonicalize_name(component.package or ""))
                if component.role == "runtime"
                else None
            )
            outcome = _fetch_and_classify(
                component,
                pypi_url(component.package or component.id),
                fetcher,
                lambda payload, c=component, s=spec: classify_pypi(c, payload, s),
            )
        elif component.kind == "github-release":
            outcome = _fetch_and_classify(
                component,
                github_releases_url(component.repo or ""),
                fetcher,
                lambda payload, c=component: classify_github(c, payload),
            )
        elif component.kind == "sdk-protocol":
            outcome = classify_protocol(component, protocol_probe())
        else:
            outcome = classify_manual(component, today)
        outcome.findings.extend(check_deprecations(component, today))
        if outcome.findings and outcome.facts.get("outcome") == "harmless":
            outcome.facts["outcome"] = "findings"
        findings.extend(outcome.findings)
        component_facts[component.id] = outcome.facts

    classifications: dict[str, int] = {}
    for finding in findings:
        key = finding.classification or "unclassified"
        classifications[key] = classifications.get(key, 0) + 1
    status = "error" if classifications.get("not-checked") else "pass"
    return results.Result(
        check="upstream",
        variant=variant,
        status=status,
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=results.environment_snapshot(),
        facts={
            "today": today.isoformat(),
            "components": component_facts,
            "classifications": dict(sorted(classifications.items())),
        },
        findings=findings,
    )


def _today() -> date:
    return datetime.now(timezone.utc).date()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``upstream`` subcommand's flags."""
    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_INVENTORY, help="inventory TOML"
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=deps.DEFAULT_PYPROJECT,
        help="pyproject.toml (declared ranges are read from it)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument("--variant", default="upstream", help="free label")


def run(args: argparse.Namespace) -> int:
    """``upstream`` subcommand: classify, optionally write JSON, summarize."""
    result = check_upstream(
        load_inventory(args.inventory),
        requirements=deps.read_declared(args.pyproject),
        fetcher=http_fetcher,
        today=_today(),
        protocol_probe=sdk_protocol_versions,
        variant=args.variant,
    )
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    counts = ", ".join(f"{k} {v}" for k, v in result.facts["classifications"].items())
    print(
        f"upstream {result.status}: {len(result.facts['components'])} components, "
        f"{len(result.findings)} findings" + (f" ({counts})" if counts else "")
    )
    for finding in result.findings:
        print(f"  [{finding.classification}] {finding.title}")
    return result.exit_code
