"""Subcommand dispatch for ``python -m benchmarks.cwru``.

Maintainer tooling, thin by design: every stage lives in its module
(:mod:`benchmarks.cwru.download`, :mod:`benchmarks.cwru.importer`,
:mod:`benchmarks.cwru.runner`, :mod:`benchmarks.cwru.scorer`); this file
only parses arguments and chains the calls.

Subcommands::

    freeze     maintainer-only: download and pin checksums.json
    download   verify-mode download of every ops record into the cache
    import     cached .mat -> opaque signals in the repository
    run        provenance tripwire -> pipeline over all records -> outcomes
    score      join outcomes.json with labels -> results.json (sole label
               reader; refuses when the outcomes artifact is absent)
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

from benchmarks.cwru import runner, scorer
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
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Shared measurement stage for ``run`` and ``all``.

    Tripwire first, then an in-process re-import (see the module
    docstring's process-boundary note), then the runner — doubled when
    ``--check-determinism`` is set — and the atomic outcomes write.

    Args:
        records: The ops records to measure.
        args: Parsed CLI namespace (``check_determinism``, ``output``).

    Returns:
        ``(outcomes, provenance)`` — the outcomes produced
        (and written) plus the provenance block that accompanied the write.

    """
    runner.assert_import_provenance()
    provenance = runner.collect_provenance()
    import_records(records, overwrite=True)
    if args.check_determinism:
        outcomes = runner.check_determinism(records)
    else:
        outcomes = runner.run_records(records)
    outcomes_w_provenance = runner.compose_outcomes_document(outcomes, provenance)
    target = runner.write_outcomes(outcomes_w_provenance, args.output)
    ok = sum(
        1
        for outcome in outcomes.values()
        if outcome.get("status") == runner.OUTCOME_STATUS_OK
    )
    print(
        f"Wrote {len(outcomes)} outcome(s) ({ok} ok, "
        f"{len(outcomes) - ok} missing/failed) to {target}."
    )
    return outcomes, provenance


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the deterministic pipeline over all ops records."""
    _run_stage(ops_view(), args)
    return 0


def _score_outcomes(
    outcomes: dict[str, dict[str, Any]],
    results_path: Optional[Path],
    measurement_provenance: Optional[Mapping[str, str]],
) -> Path:
    """Shared scoring stage: label join, metrics, atomic results write.

    Args:
        outcomes: Runner outcomes keyed by opaque id.
        results_path: Results destination; ``None`` uses the scorer's
            committed default.
        measurement_provenance: Provenance values coming from the outcomes.

    Returns:
        The results path written.
    """
    results = scorer.score_results(
        outcomes, measurement_provenance=measurement_provenance
    )
    target = scorer.write_results(results, results_path)
    print(f"Scored {len(results['records'])} record(s); wrote {target}.")
    return target


def _cmd_score(args: argparse.Namespace) -> int:
    """Score a previously written outcomes artifact against the labels."""
    outcomes, measurement_provenance = scorer.read_outcomes(args.outcomes)
    _score_outcomes(outcomes, args.output, measurement_provenance)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    """Run download -> import -> run -> gate -> score, fail closed.

    Any stage failure raises and aborts the chain; the explicit gate
    additionally refuses to reach scoring when a record is missing or
    produced a non-ok outcome, so a partial run can never be scored.
    """
    records = ops_view()
    for record in records:
        ensure_cached(record)
    outcomes, measurement_provenance = _run_stage(records, args)
    runner.assert_outcomes_complete(outcomes, records)
    _score_outcomes(outcomes, args.results_output, measurement_provenance)
    return 0


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
        help=("outcomes destination (default: benchmarks/cwru/results/outcomes.json)"),
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
        "score",
        help="join outcomes.json with labels -> results.json (sole label reader)",
    )
    score.add_argument(
        "--outcomes",
        type=Path,
        default=None,
        help=(
            "outcomes artifact to score (default: "
            "benchmarks/cwru/results/outcomes.json)"
        ),
    )
    score.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("results destination (default: benchmarks/cwru/results/results.json)"),
    )
    score.set_defaults(func=_cmd_score)

    everything = subcommands.add_parser(
        "all", help="download -> import -> run -> gate -> score, fail closed"
    )
    _add_measurement_flags(everything)
    everything.add_argument(
        "--results-output",
        type=Path,
        default=None,
        help=("results destination (default: benchmarks/cwru/results/results.json)"),
    )
    everything.set_defaults(func=_cmd_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    ``ValueError`` — the benchmark's "problem — remedy" failure mode —
    is rendered as a single stderr line with exit code 2.

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
