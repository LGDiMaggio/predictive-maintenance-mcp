"""Subcommand dispatch for ``python -m tools.audit``.

Thin by design, like ``benchmarks/cwru/__main__.py``: each check lives in its
own module and this file only routes to it. Subcommands::

    smoke               stdio protocol smoke harness against a server command
    package-smoke       build artifact -> fresh venv -> smoke harness
    artifact-licensing  built wheel and sdist: MIT license only, no sample data
    upstream            compatibility inventory vs upstream release data
    floors              constraints pinning runtime deps to their declared floors
    lock-reqs           uv.lock -> pinned requirements for the vulnerability audit
    security            pip-audit JSON -> one normalized result, fail closed
    report              results -> audit report, issue candidates

Why the imports are lazy
------------------------
Checks run in different environments: the smoke harness inside a fresh wheel
venv that has ``mcp`` but no dev tools, the upstream checker where
``packaging`` is installed. Importing every module up front would let one
check's missing dependency break all the others. So :data:`DISPATCH` names
each module as a string, and only the selected subcommand's module is
imported, at parse time.

Module convention: ``add_arguments(parser)`` and ``run(args) -> int``. A
module serving two subcommands names its functions per subcommand (see the
``floors`` and ``lock-reqs`` entries). A subcommand whose module or functions
cannot be loaded fails cleanly: one ``error:`` line on stderr, exit 2.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["DISPATCH", "SUBCOMMANDS", "Entry", "build_parser", "main"]

PROG = "python -m tools.audit"


@dataclass(frozen=True)
class Entry:
    """Where a subcommand lives: module path and the two function names."""

    module: str
    add_arguments: str
    run: str
    help: str


class SubcommandUnavailable(Exception):
    """The selected subcommand's module or functions could not be loaded."""


DISPATCH: dict[str, Entry] = {
    "smoke": Entry(
        "tools.audit.mcp_smoke",
        "add_arguments",
        "run",
        "stdio protocol smoke harness against a server command",
    ),
    "package-smoke": Entry(
        "tools.audit.package_smoke",
        "add_arguments",
        "run",
        "build artifact, install into a fresh venv, run the smoke harness",
    ),
    "artifact-licensing": Entry(
        "tools.audit.artifact_licensing",
        "add_arguments",
        "run",
        "check the built wheel and sdist: MIT license only, no sample data",
    ),
    "upstream": Entry(
        "tools.audit.upstream",
        "add_arguments",
        "run",
        "compare the compatibility inventory with upstream release data",
    ),
    "floors": Entry(
        "tools.audit.deps",
        "add_floors_arguments",
        "run_floors",
        "write constraints pinning runtime dependencies to their floors",
    ),
    "lock-reqs": Entry(
        "tools.audit.deps",
        "add_lock_reqs_arguments",
        "run_lock_reqs",
        "write uv.lock pins as a requirements file",
    ),
    "security": Entry(
        "tools.audit.security",
        "add_arguments",
        "run",
        "normalize pip-audit JSON into one result, fail closed",
    ),
    "report": Entry(
        "tools.audit.report",
        "add_arguments",
        "run",
        "aggregate results into the audit report and issue candidates",
    ),
}

#: Subcommand names, in the order ``--help`` lists them.
SUBCOMMANDS = tuple(DISPATCH)


def _load(command: str) -> tuple[Callable[[Any], None], Callable[[Any], int]]:
    """Import ``command``'s module and return its (add_arguments, run) pair.

    Raises:
        SubcommandUnavailable: When the module, one of its imports, or one of
            the two functions is missing.
    """
    entry = DISPATCH[command]
    try:
        module = importlib.import_module(entry.module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if entry.module == missing or entry.module.startswith(missing + "."):
            raise SubcommandUnavailable(
                f"subcommand {command!r} is not available: module "
                f"{entry.module} does not exist in this tree"
            ) from None
        raise SubcommandUnavailable(
            f"subcommand {command!r} cannot run: {entry.module} needs "
            f"{exc.name!r}, which is not installed in this environment"
        ) from None
    functions = []
    for attribute in (entry.add_arguments, entry.run):
        function = getattr(module, attribute, None)
        if not callable(function):
            raise SubcommandUnavailable(
                f"subcommand {command!r} is not available: {entry.module} "
                f"defines no {attribute}()"
            )
        functions.append(function)
    return functions[0], functions[1]


def build_parser(command: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the parser; only ``command``'s module is imported.

    Args:
        command: The subcommand whose arguments to attach. ``None`` builds the
            top-level parser alone (enough for ``--help``).

    Raises:
        SubcommandUnavailable: When ``command``'s module cannot be loaded.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Quality and compatibility audit checks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in SUBCOMMANDS:
        entry = DISPATCH[name]
        sub = subparsers.add_parser(name, help=entry.help, description=entry.help)
        if name == command:
            add_arguments, run = _load(name)
            add_arguments(sub)
            sub.set_defaults(func=run)
    return parser


def _selected(argv: Sequence[str]) -> Optional[str]:
    """The first argument naming a subcommand (no global options exist)."""
    return next((arg for arg in argv if arg in SUBCOMMANDS), None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list override (tests); ``None`` uses ``sys.argv``.

    Returns:
        The subcommand's exit code, or 2 when it cannot be loaded or refuses
        its input with a ``ValueError``.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    command = _selected(args_list)
    try:
        parser = build_parser(command)
    except SubcommandUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args = parser.parse_args(args_list)
    try:
        result: int = args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
