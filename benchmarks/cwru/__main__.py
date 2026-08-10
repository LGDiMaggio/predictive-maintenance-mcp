"""Subcommand dispatch for ``python -m benchmarks.cwru``.

Maintainer tooling, thin by design: every stage lives in its module
(:mod:`benchmarks.cwru.download`, :mod:`benchmarks.cwru.importer`,
:mod:`benchmarks.cwru.runner`, and — from U5 — ``scorer``); this file
only parses arguments and chains the calls.

Subcommands::

    freeze     maintainer-only: download and pin checksums.json
    download   verify-mode download of every ops record into the cache
    import     cached .mat -> opaque signals in the repository
    run        provenance tripwire -> pipeline over all records -> outcomes
    score      NotImplementedError until U5 lands (scorer.py)
    all        download -> import -> run -> fail-closed gate -> score

Process boundary (documented decision): the signal repository is
in-memory and per-process, so ``run`` (and ``all``) re-import the
converted signals from the raw cache into the current process's
repository (``overwrite=True``) before measuring. ``import`` remains a
separately verifiable stage, but running it in another process does not
populate this one. ``all`` refuses to proceed to scoring unless every
record produced an ``"ok"`` outcome (a prior-stage failure raises and
aborts the chain on its own).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from benchmarks.cwru import runner
from benchmarks.cwru.download import ensure_cached, freeze_checksums
from benchmarks.cwru.importer import import_records
from benchmarks.cwru.records import OpsRecord, ops_view

__all__ = ["build_parser", "main"]


def _cmd_freeze(args: argparse.Namespace) -> int:
    """Maintainer-only: download every ops record and pin its checksum."""
    records = ops_view()
    pins = freeze_checksums(records, force=args.force)
    print(
        f"Froze checksums for {len(records)} record(s); "
        f"{len(pins)} pin(s) now in the table."
    )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Verify-mode download of every ops record into the cache."""
    records = ops_view()
    for record in records:
        ensure_cached(record)
    print(f"Cache verified for {len(records)} record(s).")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    """Import every cached record into the in-process repository."""
    results = import_records(ops_view(), overwrite=args.overwrite)
    print(f"Imported {len(results)} record(s) into the signal repository.")
    return 0


def _run_stage(
    records: Sequence[OpsRecord], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    """Shared measurement stage for ``run`` and ``all``.

    Tripwire first, then an in-process re-import (see the module
    docstring's process-boundary note), then the runner — doubled when
    ``--check-determinism`` is set — and the atomic outcomes write.

    Args:
        records: The ops records to measure.
        args: Parsed CLI namespace (``check_determinism``, ``output``).

    Returns:
        The outcomes produced (and written).
    """
    runner.assert_import_provenance()
    import_records(records, overwrite=True)
    if args.check_determinism:
        outcomes = runner.check_determinism(records)
    else:
        outcomes = runner.run_records(records)
    target = runner.write_outcomes(outcomes, args.output)
    ok = sum(
        1
        for outcome in outcomes.values()
        if outcome.get("status") == runner.OUTCOME_STATUS_OK
    )
    print(
        f"Wrote {len(outcomes)} outcome(s) ({ok} ok, "
        f"{len(outcomes) - ok} missing/failed) to {target}."
    )
    return outcomes


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the deterministic pipeline over all ops records."""
    _run_stage(ops_view(), args)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Score outcomes against labels — not implemented until U5."""
    raise NotImplementedError(
        "Scoring is not implemented yet — it is implemented in U5 "
        "(benchmarks/cwru/scorer.py, the sole label reader). Produce "
        "outcomes with 'python -m benchmarks.cwru run' now and score "
        "once U5 lands."
    )


def _cmd_all(args: argparse.Namespace) -> int:
    """Run download -> import -> run -> gate -> score, fail closed.

    Any stage failure raises and aborts the chain; the explicit gate
    additionally refuses to reach scoring when a record is missing or
    produced a non-ok outcome, so a partial run can never be scored.
    """
    records = ops_view()
    for record in records:
        ensure_cached(record)
    outcomes = _run_stage(records, args)
    runner.assert_outcomes_complete(outcomes, records)
    return _cmd_score(args)


def _add_measurement_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by ``run`` and ``all``."""
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help=(
            "run the record set twice and require byte-identical "
            "serialized outcomes (claim scoped to this environment)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "outcomes destination (default: " "benchmarks/cwru/results/outcomes.json)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m benchmarks.cwru`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.cwru",
        description=(
            "CWRU diagnostic-accuracy benchmark (maintainer tooling; "
            "CI never downloads)."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    freeze = subcommands.add_parser(
        "freeze", help="maintainer-only: download and pin checksums.json"
    )
    freeze.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting an existing, differing checksum pin",
    )
    freeze.set_defaults(func=_cmd_freeze)

    download = subcommands.add_parser(
        "download", help="verified download of every record into the cache"
    )
    download.set_defaults(func=_cmd_download)

    importer = subcommands.add_parser(
        "import", help="cached .mat -> opaque signals in the repository"
    )
    importer.add_argument(
        "--overwrite",
        action="store_true",
        help="replace already-imported signals instead of refusing",
    )
    importer.set_defaults(func=_cmd_import)

    run = subcommands.add_parser(
        "run", help="deterministic pipeline over all records -> outcomes"
    )
    _add_measurement_flags(run)
    run.set_defaults(func=_cmd_run)

    score = subcommands.add_parser(
        "score", help="score outcomes against labels (U5 — not implemented)"
    )
    score.set_defaults(func=_cmd_score)

    everything = subcommands.add_parser(
        "all", help="download -> import -> run -> gate -> score, fail closed"
    )
    _add_measurement_flags(everything)
    everything.set_defaults(func=_cmd_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    ``ValueError`` — the benchmark's "problem — remedy" failure mode —
    is rendered as a single stderr line with exit code 2;
    ``NotImplementedError`` (the U5 scoring placeholder) propagates,
    because it marks unimplemented surface rather than an operational
    failure.

    Args:
        argv: Argument list override (tests only); ``None`` uses
            ``sys.argv``.

    Returns:
        Process exit code (0 success, 2 refused).
    """
    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
