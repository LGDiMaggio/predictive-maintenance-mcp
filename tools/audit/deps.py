"""Dependency helpers for the weekly audit: ``floors`` and ``lock-reqs``.

Both read the project's own files and write requirement lines; neither ever
edits ``pyproject.toml`` or ``uv.lock`` (the audit reports, a human upgrades).

floors
------
The floors suite leg (KTD6) installs every runtime dependency at the lowest
version ``[project.dependencies]`` admits, through a constraints file::

    python -m tools.audit floors --output floors.txt
    pip install -c floors.txt -e .[dev]

Each ``>=`` bound becomes ``name==version``, with extras and markers
stripped (constraints files take neither). A dependency without a ``>=``
bound is an error naming it, never skipped: a skipped entry would let that
leg resolve the newest release while still being reported as "floors".

lock-reqs
---------
pip-audit cannot read ``uv.lock``, so the security job audits a requirements
file of its pins (KTD8, ``pip-audit -r ... --no-deps --disable-pip``). Only
registry-sourced packages are emitted: the project itself and any editable,
path, directory, git or virtual source have no PyPI advisory data to query.

uv.lock is a universal lock. When resolution forks by Python version, one
name appears once per fork (today ``sphinx`` and ``sphinx-autodoc-typehints``),
and each entry lists the forks it belongs to under ``resolution-markers``.
Those markers are joined with ``or`` into a PEP 508 marker on the pin, so on
any interpreter exactly one pin per name applies; pip-audit evaluates markers
before its duplicate-name check. An empty result is an error: an audit of
zero packages would pass trivially.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

__all__ = [
    "DEFAULT_LOCK",
    "DEFAULT_PYPROJECT",
    "PROJECT_NAME",
    "add_floors_arguments",
    "add_lock_reqs_arguments",
    "declared_dependencies",
    "floor_pins",
    "load_toml",
    "lock_pins",
    "read_declared",
    "run_floors",
    "run_lock_reqs",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_LOCK = REPO_ROOT / "uv.lock"

#: The distribution this repository builds; never audited as a dependency.
PROJECT_NAME = "predictive-maintenance-mcp"


def load_toml(path: Union[str, Path]) -> dict[str, Any]:
    """Parse a TOML file.

    Raises:
        ValueError: When the file is missing or is not valid TOML, with the
            path named (the CLI turns this into exit 2).
    """
    target = Path(path)
    try:
        with open(target, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ValueError(f"{target} does not exist") from None
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{target} is not valid TOML: {exc}") from None


def declared_dependencies(pyproject: Mapping[str, Any]) -> list[Requirement]:
    """Return ``[project.dependencies]`` of a parsed pyproject as requirements.

    Raises:
        ValueError: When the table is missing or an entry does not parse.
    """
    project = pyproject.get("project")
    entries = project.get("dependencies") if isinstance(project, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("pyproject.toml has no [project].dependencies list")
    requirements = []
    for entry in entries:
        try:
            requirements.append(Requirement(entry))
        except (InvalidRequirement, TypeError) as exc:
            raise ValueError(f"unparseable dependency {entry!r}: {exc}") from None
    return requirements


def read_declared(path: Union[str, Path] = DEFAULT_PYPROJECT) -> list[Requirement]:
    """Read the runtime dependencies declared in ``pyproject.toml`` at ``path``."""
    return declared_dependencies(load_toml(path))


def floor_pins(requirements: Iterable[Requirement]) -> list[str]:
    """Pin each requirement to its ``>=`` floor as ``name==version``.

    Args:
        requirements: Declared runtime dependencies.

    Returns:
        One line per requirement, in declaration order, without extras or
        markers.

    Raises:
        ValueError: Naming every requirement that has no ``>=`` bound.
    """
    pins: list[str] = []
    unbounded: list[str] = []
    for requirement in requirements:
        floors = [
            spec.version for spec in requirement.specifier if spec.operator == ">="
        ]
        if not floors:
            unbounded.append(str(requirement))
            continue
        # Two ">=" bounds are legal PEP 440; the higher one is the real floor.
        floor = max(floors, key=Version)
        pins.append(f"{requirement.name}=={floor}")
    if unbounded:
        raise ValueError(
            "runtime dependencies without a '>=' floor cannot be pinned for the "
            f"floors leg: {', '.join(unbounded)}"
        )
    return pins


def _registry_entry(package: Mapping[str, Any]) -> bool:
    source = package.get("source")
    return isinstance(source, Mapping) and "registry" in source


def _marker_for(package: Mapping[str, Any]) -> Optional[str]:
    markers = package.get("resolution-markers")
    if not markers:
        return None
    if not isinstance(markers, list) or not all(isinstance(m, str) for m in markers):
        raise ValueError(
            f"uv.lock entry {package.get('name')!r} has malformed resolution-markers"
        )
    return " or ".join(f"({marker})" for marker in markers)


def lock_pins(lock: Mapping[str, Any], project: str = PROJECT_NAME) -> list[str]:
    """Pinned requirements for every registry-sourced package in a parsed lock.

    Args:
        lock: Parsed ``uv.lock``.
        project: Distribution name to leave out (the project itself).

    Returns:
        ``name==version`` lines, a fork's lines carrying its
        ``resolution-markers`` as a PEP 508 marker, sorted by name and version.

    Raises:
        ValueError: When an entry is malformed or no package qualifies.
    """
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise ValueError("uv.lock [[package]] entries are malformed")
    skip = canonicalize_name(project)
    pins: list[tuple[str, str, str]] = []
    for package in packages:
        if not isinstance(package, Mapping):
            raise ValueError("uv.lock [[package]] entry is not a table")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError(f"uv.lock entry without name or version: {package!r}")
        if canonicalize_name(name) == skip or not _registry_entry(package):
            continue
        line = f"{name}=={version}"
        marker = _marker_for(package)
        if marker:
            line = f"{line}; {marker}"
        pins.append((canonicalize_name(name), version, line))
    if not pins:
        raise ValueError(
            "uv.lock yielded no registry-sourced packages: an audit of nothing "
            "would pass trivially"
        )
    pins.sort(key=lambda item: (item[0], Version(item[1])))
    return [line for _, _, line in pins]


def _emit(lines: list[str], output: Optional[Path], label: str) -> None:
    text = "\n".join(lines) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"{label}: {len(lines)} requirement lines written to {output}")


def add_floors_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``floors`` subcommand's flags."""
    parser.add_argument(
        "--pyproject", type=Path, default=DEFAULT_PYPROJECT, help="pyproject.toml"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="file to write (default: stdout)"
    )


def run_floors(args: argparse.Namespace) -> int:
    """``floors`` subcommand. A missing floor raises ValueError (exit 2)."""
    _emit(floor_pins(read_declared(args.pyproject)), args.output, "floors")
    return 0


def add_lock_reqs_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``lock-reqs`` subcommand's flags."""
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="uv.lock")
    parser.add_argument(
        "--output", type=Path, default=None, help="file to write (default: stdout)"
    )


def run_lock_reqs(args: argparse.Namespace) -> int:
    """``lock-reqs`` subcommand. An empty result raises ValueError (exit 2)."""
    _emit(lock_pins(load_toml(args.lock)), args.output, "lock-reqs")
    return 0
