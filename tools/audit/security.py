"""Normalize pip-audit JSON into one ``security`` result.

The weekly security job runs pip-audit twice: over the ``uv.lock`` pins
(``lock-reqs`` output, ``--no-deps --disable-pip``) and over a fresh
resolution. pip-audit's exit code is not trusted, because a tool failure and
a found vulnerability share non-zero codes; its JSON is read instead::

    python -m tools.audit security --lock lock.json --fresh fresh.json \\
        --json-out .audit/results/security.json

Findings
--------
One finding per (package, advisory) across both audits, subject
``<package> <advisory id>``, so each advisory has its own
fingerprint and a later fix version or alias does not re-key it. Evidence
records the audited versions, fix versions, aliases, which audit saw it,
``fix_in_declared_range`` (a listed fix satisfies the declared range),
``fix_reachable`` and ``fixable_in_range``.

``fix_reachable`` reads the fresh resolution, which installs the newest
releases every declared range and every dependency's own constraints allow:

* true when the fresh resolution installs the package without this
  advisory, under its id or any alias (refreshing the lock reaches an
  unaffected version);
* false when the fresh resolution still has it (no release the current
  constraints admit fixes it, whatever ``fix_versions`` lists);
* None when the fresh resolution does not install the package (a
  platform-specific dependency on another platform) or was not read.

``fixable_in_range`` is true when a fix version exists, satisfies the
package's declared range in ``[project.dependencies]`` (a transitive package
has no declared range, so any fix version counts), and ``fix_reachable`` is
not false. A fixable advisory is ``breaking-candidate`` (high), anything else
``review``. The project itself is always skipped by pip-audit (it is not on
PyPI under the audited version), so it is left out; another package
pip-audit skipped is an ``audit-skipped`` finding, unless it is declared.

Fail closed
-----------
Status ``error`` (exit 2) when an input file is missing or unparseable, or
when an audit, after PEP 503 normalization, does not audit every declared
dependency: a declared dependency that is absent or that pip-audit skipped
was not checked, and an empty or truncated audit must never read as "no
vulnerabilities". The result is still written, with whatever the other file
showed, so the report has evidence. Otherwise the status is ``pass`` even
with advisories: the weekly verdict belongs to the report builder.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tools.audit import deps, results

__all__ = [
    "AUDIT_LABELS",
    "AuditInputError",
    "add_arguments",
    "build_result",
    "fixable_in_range",
    "load_audit",
    "run",
]

#: The two audits the weekly job runs, in report order.
AUDIT_LABELS = ("lock", "fresh")

DEFAULT_VARIANT = "pip-audit"


class AuditInputError(ValueError):
    """A pip-audit JSON file is missing or does not have the expected shape."""


@dataclass
class Vuln:
    id: str
    fix_versions: list[str]
    aliases: list[str]
    description: str


@dataclass
class Audited:
    """One ``dependencies[]`` entry of a pip-audit JSON document."""

    name: str
    version: Optional[str]
    vulns: list[Vuln] = field(default_factory=list)
    skip_reason: Optional[str] = None


def _str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise AuditInputError(f"{where} is not a list of strings")
    return list(value)


def _parse_vuln(item: Any, where: str) -> Vuln:
    if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
        raise AuditInputError(f"{where} has no string 'id'")
    description = item.get("description") or ""
    if not isinstance(description, str):
        raise AuditInputError(f"{where}.description is not a string")
    return Vuln(
        id=item["id"],
        fix_versions=_str_list(item.get("fix_versions"), f"{where}.fix_versions"),
        aliases=_str_list(item.get("aliases"), f"{where}.aliases"),
        description=description,
    )


def parse_audit(doc: Any) -> list[Audited]:
    """Validate and convert a parsed pip-audit JSON document.

    Raises:
        AuditInputError: On any shape the normalizer cannot trust.
    """
    if not isinstance(doc, Mapping) or not isinstance(doc.get("dependencies"), list):
        raise AuditInputError("expected an object with a 'dependencies' list")
    audited = []
    for index, item in enumerate(doc["dependencies"]):
        where = f"dependencies[{index}]"
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise AuditInputError(f"{where} has no string 'name'")
        version = item.get("version")
        if version is not None and not isinstance(version, str):
            raise AuditInputError(f"{where}.version is not a string")
        skip_reason = item.get("skip_reason")
        if skip_reason is not None and not isinstance(skip_reason, str):
            raise AuditInputError(f"{where}.skip_reason is not a string")
        vulns = item.get("vulns", [])
        if not isinstance(vulns, list):
            raise AuditInputError(f"{where}.vulns is not a list")
        audited.append(
            Audited(
                name=item["name"],
                version=version,
                vulns=[
                    _parse_vuln(v, f"{where}.vulns[{i}]") for i, v in enumerate(vulns)
                ],
                skip_reason=skip_reason,
            )
        )
    return audited


def load_audit(path: Union[str, Path]) -> list[Audited]:
    """Read and validate one pip-audit JSON file.

    Raises:
        AuditInputError: When the file is missing, not JSON, or malformed.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AuditInputError(f"{target} does not exist") from None
    except OSError as exc:
        raise AuditInputError(f"{target} cannot be read: {exc}") from None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"{target} is not JSON: {exc}") from None
    return parse_audit(doc)


def fixable_in_range(
    fix_versions: Sequence[str],
    declared: Optional[SpecifierSet],
    fix_reachable: Optional[bool] = None,
) -> bool:
    """Whether some fix version is installable within the declared range.

    Args:
        fix_versions: pip-audit's ``fix_versions`` for one advisory.
        declared: The package's specifier from ``[project.dependencies]``, or
            None for a transitive package (any fix is then in range).
        fix_reachable: What the fresh resolution showed (see the module
            docstring); False overrides the fix versions.
    """
    if fix_reachable is False:
        return False
    valid = []
    for text in fix_versions:
        try:
            valid.append(Version(text))
        except InvalidVersion:
            continue
    if not valid:
        return False
    if declared is None:
        return True
    return any(declared.contains(version) for version in valid)


def _input_error(label: str, path: Path, error: str) -> results.Finding:
    return results.Finding(
        rule="audit-input-invalid",
        category="security",
        severity="high",
        subject=f"pip-audit {label}",
        title=f"The {label} pip-audit output could not be read: nothing was audited",
        evidence={"path": str(path), "error": results.truncate_evidence(error)},
        classification="not-checked",
    )


def _coverage_error(
    label: str, missing: list[str], skipped: Mapping[str, str]
) -> results.Finding:
    """Declared dependencies the ``label`` audit did not check.

    Args:
        missing: Every declared name not audited, skipped ones included.
        skipped: pip-audit's skip reason for the declared names it skipped.
    """
    return results.Finding(
        rule="audit-coverage",
        category="security",
        severity="high",
        subject=f"pip-audit {label} coverage",
        title=(
            f"The {label} pip-audit output does not cover declared "
            f"dependencies: {', '.join(missing)}"
        ),
        evidence={
            "missing": missing,
            "skipped": {
                name: results.truncate_evidence(reason)
                for name, reason in sorted(skipped.items())
            },
        },
        classification="not-checked",
    )


def _normalize(
    audits: Mapping[str, list[Audited]],
    declared: Mapping[str, SpecifierSet],
) -> list[results.Finding]:
    advisories: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: dict[str, dict[str, Any]] = {}
    project = canonicalize_name(deps.PROJECT_NAME)
    for label, audited in audits.items():
        for dep in audited:
            name = canonicalize_name(dep.name)
            if dep.skip_reason is not None:
                if name == project:
                    continue
                entry = skipped.setdefault(
                    name, {"reasons": {}, "versions": {}, "seen_in": set()}
                )
                entry["reasons"][label] = dep.skip_reason
                if dep.version:
                    entry["versions"][label] = dep.version
                entry["seen_in"].add(label)
                continue
            for vuln in dep.vulns:
                entry = advisories.setdefault(
                    (name, vuln.id),
                    {
                        "versions": {},
                        "fix_versions": set(),
                        "aliases": set(),
                        "seen_in": set(),
                        "description": "",
                    },
                )
                if dep.version:
                    entry["versions"][label] = dep.version
                entry["fix_versions"].update(vuln.fix_versions)
                entry["aliases"].update(vuln.aliases)
                entry["seen_in"].add(label)
                entry["description"] = entry["description"] or vuln.description

    fresh = audits.get("fresh")
    fresh_installed: set[str] = set()
    # Advisory ids and aliases the fresh audit reported, per package: the two
    # audits may name one vulnerability differently (PYSEC in one, GHSA in
    # the other), so reachability matches on either.
    fresh_ids: dict[str, set[str]] = {}
    for dep in fresh or ():
        if dep.skip_reason is None:
            name = canonicalize_name(dep.name)
            fresh_installed.add(name)
            ids = fresh_ids.setdefault(name, set())
            for vuln in dep.vulns:
                ids.add(vuln.id)
                ids.update(vuln.aliases)
    findings = []
    for (name, advisory), entry in sorted(advisories.items()):
        spec = declared.get(name)
        fixes = sorted(entry["fix_versions"], key=_version_sort_key)
        known_as = {advisory, *entry["aliases"]}
        reachable = (
            not (known_as & fresh_ids.get(name, set()))
            if name in fresh_installed
            else None
        )
        fixable = fixable_in_range(fixes, spec, reachable)
        # Whether the declared range admits a listed fix at all: it decides
        # which explanation a non-fixable advisory gets.
        fix_in_range = fixable_in_range(fixes, spec)
        affected = ", ".join(sorted(set(entry["versions"].values()))) or "?"
        if fixable:
            title = (
                f"{name} {affected} is affected by {advisory}; a fix "
                f"({', '.join(fixes)}) is installable within the declared range"
            )
        elif fix_in_range:
            title = (
                f"{name} {affected} is affected by {advisory}; fixes exist "
                f"({', '.join(fixes)}), but the newest resolution the declared "
                f"ranges allow is still affected"
            )
        elif not fixes and reachable:
            title = (
                f"{name} {affected} is affected by {advisory}; no fix version is "
                f"listed, and the fresh resolution installs a release without it"
            )
        else:
            title = (
                f"{name} {affected} is affected by {advisory}; no fix is "
                f"installable within the declared range"
            )
        findings.append(
            results.Finding(
                rule="advisory",
                category="security",
                severity="high" if fixable else "medium",
                subject=f"{name} {advisory}",
                title=title,
                evidence={
                    "package": name,
                    "advisory": advisory,
                    "versions": dict(sorted(entry["versions"].items())),
                    "fix_versions": fixes,
                    "aliases": sorted(entry["aliases"]),
                    "seen_in": sorted(entry["seen_in"]),
                    "declared": str(spec) if spec is not None else None,
                    "fix_in_declared_range": fix_in_range,
                    "fix_reachable": reachable,
                    "fixable_in_range": fixable,
                    "excerpt": results.truncate_evidence(entry["description"]),
                },
                classification="breaking-candidate" if fixable else "review",
            )
        )
    for name, entry in sorted(skipped.items()):
        # A skipped declared dependency is a coverage error (build_result),
        # not a finding of its own.
        if name in declared:
            continue
        reasons = " | ".join(
            f"{label}: {reason}" for label, reason in sorted(entry["reasons"].items())
        )
        findings.append(
            results.Finding(
                rule="audit-skipped",
                category="security",
                severity="low",
                subject=name,
                title=f"pip-audit skipped {name}: its advisories were not checked",
                evidence={
                    "package": name,
                    "versions": dict(sorted(entry["versions"].items())),
                    "seen_in": sorted(entry["seen_in"]),
                    "excerpt": results.truncate_evidence(reasons),
                },
                classification="review",
            )
        )
    return findings


def _version_sort_key(text: str) -> tuple[int, Any]:
    try:
        return (0, Version(text))
    except InvalidVersion:
        return (1, text)


def build_result(
    lock: Union[str, Path],
    fresh: Union[str, Path],
    requirements: Sequence[Requirement],
    variant: str = DEFAULT_VARIANT,
) -> results.Result:
    """Read both pip-audit outputs and return one ``security`` result.

    Args:
        lock: pip-audit JSON for the uv.lock pins.
        fresh: pip-audit JSON for a fresh resolution.
        requirements: ``[project.dependencies]``, read live from pyproject.
        variant: Free label for the run.
    """
    started_at = results.utc_now()
    declared = {canonicalize_name(r.name): r.specifier for r in requirements}
    findings: list[results.Finding] = []
    audits: dict[str, list[Audited]] = {}
    inputs: dict[str, Any] = {}
    failed = False
    for label, path in (("lock", Path(lock)), ("fresh", Path(fresh))):
        try:
            audited = load_audit(path)
        except AuditInputError as exc:
            findings.append(_input_error(label, path, str(exc)))
            inputs[label] = {"path": str(path), "error": str(exc)}
            failed = True
            continue
        audits[label] = audited
        # A skipped dependency was not checked, so it does not count as covered.
        names = {
            canonicalize_name(dep.name) for dep in audited if dep.skip_reason is None
        }
        missing = sorted(set(declared) - names)
        if missing:
            skipped = {
                canonicalize_name(dep.name): dep.skip_reason
                for dep in audited
                if dep.skip_reason is not None
                and canonicalize_name(dep.name) in missing
            }
            findings.append(_coverage_error(label, missing, skipped))
            failed = True
        inputs[label] = {
            "path": str(path),
            "dependencies": len(audited),
            "vulnerable": sum(1 for dep in audited if dep.vulns),
            "skipped": sum(1 for dep in audited if dep.skip_reason is not None),
            "missing_declared": missing,
        }
    normalized = _normalize(audits, declared)
    findings.extend(normalized)
    reported = [f for f in normalized if f.rule == "advisory"]
    return results.Result(
        check="security",
        variant=variant,
        status="error" if failed else "pass",
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=results.environment_snapshot(),
        facts={
            "inputs": inputs,
            "declared": sorted(declared),
            "advisories": len(reported),
            "fixable_in_range": sum(
                1 for f in reported if f.evidence["fixable_in_range"]
            ),
        },
        findings=findings,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``security`` subcommand's flags."""
    parser.add_argument(
        "--lock", type=Path, required=True, help="pip-audit JSON for uv.lock pins"
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        required=True,
        help="pip-audit JSON for a fresh resolution",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=deps.DEFAULT_PYPROJECT,
        help="pyproject.toml whose [project.dependencies] the audits must cover",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="free label")


def run(args: argparse.Namespace) -> int:
    """``security`` subcommand: normalize, optionally write JSON, summarize."""
    requirements = deps.read_declared(args.pyproject)
    result = build_result(args.lock, args.fresh, requirements, variant=args.variant)
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    facts = result.facts
    print(
        f"security {result.status}: {facts['advisories']} advisories, "
        f"{facts['fixable_in_range']} fixable within the declared range"
    )
    for finding in result.findings:
        print(f"  [{finding.classification}] {finding.title}")
    return result.exit_code
