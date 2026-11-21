"""Docker build boundary probes: ``python -m tools.audit docker-probe``.

Why this exists
---------------
Two files decide what a Docker image built from this repository can contain:
``.dockerignore`` decides what enters the build context, and the runtime
``COPY`` lines of the ``Dockerfile`` decide what enters the image. Reading
either file says what its author meant; only Docker says what it does. So
each probe plants files that must never reach an image into a temporary
build context, runs a tiny build ``FROM scratch`` whose files Docker exports
to a local folder (``--output type=local``), and reads that folder. No base
image is pulled and nothing runs inside a container.

The probes
----------
``copy-lines``
    A context with a stand-in for every source the production runtime
    ``COPY`` lines name, plus the planted files of :data:`PLANTED_DATA`, and
    an empty ``.dockerignore``. The probe build runs those ``COPY`` lines,
    exactly as the ``Dockerfile`` writes them. No planted file may come out;
    every stand-in must, which proves the build ran.
``context``
    A copy of the repository's tracked files (those present on disk) with the
    production ``.dockerignore``, plus :data:`PLANTED_DATA` and
    :data:`PLANTED_WEIGHTS`. The probe build copies the whole context
    (``COPY . /ctx``). No planted data file and none of
    :data:`EXCLUDED_TRACKED` may come out; every file of
    :data:`POSITIVE_CONTROL` must.
``weights``
    Read from the same build: no planted weight file under ``src/`` may come
    out. Until ``.dockerignore`` carries recursive weight exclusions this
    control is missing; ``--expect-missing weights`` reports that as a known
    gap instead of a failure, and turns into an error the day the control is
    present, so the flag cannot outlive the gap.

Status: ``error`` when Docker is unavailable or a probe build fails (a probe
that did not run is never a pass), ``fail`` when a planted file comes out or
a control file does not, ``pass`` otherwise. Results use the
``artifact-licensing`` check id: the probes guard the same boundary.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from tools.audit import results
from tools.audit.artifact_licensing import CHECK, _finding, _tracked_files
from tools.audit.results import Finding

__all__ = [
    "EXCLUDED_TRACKED",
    "PLANTED_DATA",
    "PLANTED_WEIGHTS",
    "POSITIVE_CONTROL",
    "PROBES",
    "DockerProbeConfig",
    "ProbeOutcome",
    "add_arguments",
    "copy_sources",
    "dockerfile_instructions",
    "exported_files",
    "run",
    "run_docker_probe",
    "runtime_copy_lines",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files that must never reach an image: a trained model, recordings (one in
#: each third-party sample folder), an asset ledger, a report, an unknown
#: folder, a document index and an operator manual.
PLANTED_DATA = (
    "models/x_model.pkl",
    "data/cache/x.mat",
    "data/real_bearings/planted.mat",
    "data/signals/real_train/planted.mat",
    "data/signals/real_test/planted.mat",
    "data/ledger/P-101.jsonl",
    "reports/r.html",
    "newdir/x.txt",
    "resources/cache/rag_index_tfidf.pkl",
    "resources/machine_manuals/operator.pdf",
)

#: Model weight files planted only under ``src/``, where the allowlist lets
#: files in by default.
PLANTED_WEIGHTS = (
    "src/_probe/model.safetensors",
    "src/_probe/a/b/w.bin",
    "src/_probe/x.pt",
    "src/_probe/y.pth",
)

#: Tracked files the context must leave out: the example manual, its README
#: and the index extracted from it.
EXCLUDED_TRACKED = (
    "resources/machine_manuals/README.md",
    "resources/machine_manuals/test_pump_manual.pdf",
    "resources/machine_manuals/test_pump_manual.txt",
    "resources/cache/test_pump_manual_extraction.json",
)

#: Files the context must keep: without them the image cannot be built.
POSITIVE_CONTROL = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src/server.py",
    "resources/bearing_catalogs/common_bearings_catalog.json",
    "resources/bearing_catalogs/README.md",
)

#: The probe names, in the order they run and report.
PROBES = ("copy-lines", "context", "weights")

BUILD_TIMEOUT_SECONDS = 900

ProbeStatus = Literal["pass", "fail", "missing", "error"]

#: Setup failures that end a probe as ``error``: an unreadable Dockerfile line,
#: a context that cannot be written, a repository root git cannot list.
_SETUP_ERRORS = (ValueError, OSError, subprocess.CalledProcessError)

PLANTED_CONTENT = b"planted by tools.audit docker-probe; must not reach an image\n"


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------


def dockerfile_instructions(text: str) -> list[str]:
    """The instructions of a Dockerfile, continuation lines joined, comments
    and blank lines dropped."""
    instructions: list[str] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        instructions.append((pending + line).strip())
        pending = ""
    if pending.strip():
        instructions.append(pending.strip())
    return instructions


def _keyword(instruction: str) -> str:
    return instruction.split(None, 1)[0].upper() if instruction else ""


def runtime_copy_lines(text: str) -> list[str]:
    """The ``COPY`` and ``ADD`` instructions of the last build stage that copy
    from the build context (not ``--from`` another stage).

    Raises:
        ValueError: When the Dockerfile has no ``FROM``.
    """
    instructions = dockerfile_instructions(text)
    starts = [i for i, item in enumerate(instructions) if _keyword(item) == "FROM"]
    if not starts:
        raise ValueError("the Dockerfile has no FROM instruction")
    return [
        item
        for item in instructions[starts[-1] + 1 :]
        if _keyword(item) in ("COPY", "ADD")
        and not any(
            token.startswith("--from") for token in shlex.split(item, posix=True)
        )
    ]


def copy_sources(instruction: str) -> list[str]:
    """The source paths of one ``COPY``/``ADD`` instruction (flags and the
    destination dropped); the JSON array form is read too."""
    rest = instruction.split(None, 1)[1] if " " in instruction else ""
    rest = rest.strip()
    if rest.startswith("["):
        tokens = [str(item) for item in json.loads(rest)]
    else:
        tokens = shlex.split(rest, posix=True)
    args = [token for token in tokens if not token.startswith("--")]
    return args[:-1]


def exported_files(root: Path) -> set[str]:
    """Every file under ``root``, as POSIX paths relative to it."""
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def _present(files: set[str], wanted: str) -> bool:
    """Whether ``wanted`` (a context path) came out, at any destination."""
    return any(path == wanted or path.endswith("/" + wanted) for path in files)


def _plant(root: Path, paths: Iterable[str]) -> None:
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PLANTED_CONTENT)


def _stand_ins(sources: Sequence[str]) -> list[str]:
    """One file per ``COPY`` source: the file itself, or one inside a folder
    (the context root for ``.``)."""
    names = []
    for source in sources:
        if posixpath.normpath(source) == ".":
            names.append("__probe__.txt")
        elif source.endswith("/"):
            names.append(source.rstrip("/") + "/__probe__.txt")
        else:
            names.append(source)
    return names


@dataclass
class ProbeOutcome:
    """One probe's verdict; ``missing`` marks a control known to be absent."""

    probe: str
    status: ProbeStatus
    leaked: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    detail: Optional[str] = None
    seconds: Optional[float] = None


def copy_lines_verdict(files: set[str], stand_ins: Sequence[str]) -> ProbeOutcome:
    """The ``copy-lines`` verdict over the exported files."""
    leaked = sorted(name for name in PLANTED_DATA if _present(files, name))
    absent = sorted(name for name in stand_ins if not _present(files, name))
    status: ProbeStatus = "fail" if leaked or absent else "pass"
    return ProbeOutcome("copy-lines", status, leaked, absent)


def context_verdict(files: set[str]) -> ProbeOutcome:
    """The ``context`` verdict over the exported ``/ctx`` files."""
    leaked = sorted(
        name for name in (*PLANTED_DATA, *EXCLUDED_TRACKED) if name in files
    )
    absent = sorted(name for name in POSITIVE_CONTROL if name not in files)
    status: ProbeStatus = "fail" if leaked or absent else "pass"
    return ProbeOutcome("context", status, leaked, absent)


def weights_verdict(files: set[str], expect_missing: bool) -> ProbeOutcome:
    """The ``weights`` verdict over the exported ``/ctx`` files."""
    leaked = sorted(name for name in PLANTED_WEIGHTS if name in files)
    if not leaked:
        if expect_missing:
            return ProbeOutcome(
                "weights",
                "error",
                detail=(
                    "the weight exclusions are present: drop "
                    "--expect-missing weights"
                ),
            )
        return ProbeOutcome("weights", "pass")
    if expect_missing:
        return ProbeOutcome(
            "weights",
            "missing",
            leaked,
            detail="no recursive weight exclusion in .dockerignore (known gap)",
        )
    return ProbeOutcome("weights", "fail", leaked)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

#: How a probe build runs: ``(command, timeout) -> (exit code, output)``.
Runner = Callable[[Sequence[str], int], tuple[int, str]]


def _subprocess_runner(command: Sequence[str], timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _build(
    docker: str,
    context: Path,
    dockerfile_text: str,
    workdir: Path,
    runner: Runner,
) -> tuple[Optional[set[str]], str]:
    """Run one ``FROM scratch`` build and return the exported files, or None
    and the build output when it failed."""
    dockerfile = workdir / "probe.Dockerfile"
    dockerfile.write_text(dockerfile_text, encoding="utf-8")
    out = workdir / "out"
    command = [
        docker,
        "build",
        "--progress=plain",
        "--output",
        f"type=local,dest={out}",
        "-f",
        str(dockerfile),
        str(context),
    ]
    code, output = runner(command, BUILD_TIMEOUT_SECONDS)
    if code != 0 or not out.is_dir():
        return None, output[-2000:]
    return exported_files(out), output


def _copy_tracked(repo_root: Path, target: Path) -> int:
    """Copy the tracked files present on disk; return how many were copied."""
    copied = 0
    for name in _tracked_files(repo_root, ()):
        source = repo_root / name
        if not source.is_file():
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied += 1
    return copied


def probe_copy_lines(
    dockerfile_text: str, docker: str, workdir: Path, runner: Runner
) -> ProbeOutcome:
    """The ``copy-lines`` probe for the given Dockerfile text."""
    started = time.monotonic()
    lines = runtime_copy_lines(dockerfile_text)
    sources = [source for line in lines for source in copy_sources(line)]
    stand_ins = _stand_ins(sources)
    context = workdir / "context"
    _plant(context, [*stand_ins, *PLANTED_DATA])
    (context / ".dockerignore").write_text("", encoding="utf-8")
    probe = "FROM scratch\nWORKDIR /app\n" + "\n".join(lines) + "\n"
    files, output = _build(docker, context, probe, workdir, runner)
    if files is None:
        return ProbeOutcome("copy-lines", "error", detail=output)
    outcome = copy_lines_verdict(files, stand_ins)
    outcome.seconds = round(time.monotonic() - started, 1)
    return outcome


def probe_context(
    repo_root: Path,
    docker: str,
    workdir: Path,
    runner: Runner,
    expect_missing_weights: bool,
    dockerignore_text: Optional[str] = None,
) -> list[ProbeOutcome]:
    """The ``context`` and ``weights`` probes, from one build of a copy of the
    tracked files. ``dockerignore_text`` replaces the copied ``.dockerignore``
    (a removal proof runs the probe against a weakened copy)."""
    started = time.monotonic()
    context = workdir / "context"
    _copy_tracked(repo_root, context)
    if dockerignore_text is not None:
        (context / ".dockerignore").write_text(dockerignore_text, encoding="utf-8")
    _plant(context, [*PLANTED_DATA, *PLANTED_WEIGHTS])
    files, output = _build(
        docker, context, "FROM scratch\nCOPY . /ctx\n", workdir, runner
    )
    if files is None:
        return [
            ProbeOutcome("context", "error", detail=output),
            ProbeOutcome("weights", "error", detail="the context build failed"),
        ]
    seconds = round(time.monotonic() - started, 1)
    ctx_files = {name[len("ctx/") :] for name in files if name.startswith("ctx/")}
    outcomes = [
        context_verdict(ctx_files),
        weights_verdict(ctx_files, expect_missing_weights),
    ]
    for outcome in outcomes:
        outcome.seconds = seconds
    return outcomes


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class DockerProbeConfig:
    """Inputs of one run (the CLI maps its flags onto this)."""

    repo_root: Path = REPO_ROOT
    docker: str = "docker"
    expect_missing: Sequence[str] = ()
    variant: Optional[str] = None


def _outcome_finding(outcome: ProbeOutcome) -> Optional[Finding]:
    subject = f"docker-probe {outcome.probe}"
    if outcome.status == "error":
        return _finding(
            "docker-probe-error",
            subject,
            f"The {outcome.probe} probe did not run to a verdict",
            classification="not-checked",
            detail=outcome.detail,
        )
    if outcome.status == "fail":
        parts = []
        if outcome.leaked:
            parts.append("came out: " + ", ".join(outcome.leaked))
        if outcome.absent:
            parts.append("missing: " + ", ".join(outcome.absent))
        return _finding(
            f"docker-probe-{outcome.probe}",
            subject,
            f"The {outcome.probe} probe failed; " + "; ".join(parts),
            leaked=outcome.leaked or None,
            absent=outcome.absent or None,
        )
    return None


def _status(outcomes: Sequence[ProbeOutcome], errors: Sequence[Finding]) -> str:
    if errors or any(item.status == "error" for item in outcomes):
        return "error"
    if any(item.status == "fail" for item in outcomes):
        return "fail"
    return "pass"


def _run_probes(config: DockerProbeConfig, errors: list[Finding]) -> list[ProbeOutcome]:
    """Run the probe builds; a reason no probe could run goes to ``errors``."""
    if shutil.which(config.docker) is None:
        errors.append(
            _finding(
                "docker-unavailable",
                "docker",
                f"{config.docker!r} is not on PATH, so no probe ran; install "
                "Docker with BuildKit (docker buildx) and run the probes again",
                classification="not-checked",
            )
        )
        return []
    repo_root = Path(config.repo_root)
    try:
        dockerfile_text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(
            _finding(
                "dockerfile-unreadable",
                "Dockerfile",
                f"The Dockerfile cannot be read: {exc}",
                classification="not-checked",
            )
        )
        return []
    outcomes: list[ProbeOutcome] = []
    with tempfile.TemporaryDirectory(prefix="pmm-docker-probe-") as tmp:
        base = Path(tmp)
        (base / "copy").mkdir()
        (base / "ctx").mkdir()
        try:
            outcomes.append(
                probe_copy_lines(
                    dockerfile_text, config.docker, base / "copy", _subprocess_runner
                )
            )
        except _SETUP_ERRORS as exc:
            outcomes.append(ProbeOutcome("copy-lines", "error", detail=str(exc)))
        try:
            outcomes += probe_context(
                repo_root,
                config.docker,
                base / "ctx",
                _subprocess_runner,
                "weights" in config.expect_missing,
            )
        except _SETUP_ERRORS as exc:
            outcomes += [
                ProbeOutcome("context", "error", detail=str(exc)),
                ProbeOutcome("weights", "error", detail="the context setup failed"),
            ]
    return outcomes


def run_docker_probe(config: DockerProbeConfig) -> results.Result:
    """Run every probe and report; never a pass without a Docker verdict."""
    started_at = results.utc_now()
    facts: dict[str, Any] = {
        "probes": {},
        "expect_missing": sorted(config.expect_missing),
    }
    errors: list[Finding] = []
    outcomes = _run_probes(config, errors)

    findings: list[Finding] = list(errors)
    for outcome in outcomes:
        facts["probes"][outcome.probe] = {
            key: value
            for key, value in (
                ("status", outcome.status),
                ("leaked", outcome.leaked),
                ("absent", outcome.absent),
                ("detail", outcome.detail),
                ("seconds", outcome.seconds),
            )
            if value not in (None, [])
        }
        finding = _outcome_finding(outcome)
        if finding is not None:
            findings.append(finding)

    return results.Result(
        check=CHECK,
        variant=config.variant or "docker-probe",
        status=_status(outcomes, errors),
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=results.environment_snapshot(packages=()),
        facts=facts,
        findings=sorted(findings, key=lambda item: (item.rule, item.subject)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``docker-probe`` subcommand's flags."""
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository holding the Dockerfile and .dockerignore",
    )
    parser.add_argument("--docker", default="docker", help="docker executable")
    parser.add_argument(
        "--expect-missing",
        action="append",
        choices=["weights"],
        default=[],
        help=(
            "report this control as a known gap rather than a failure; an "
            "error once the control is present"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument("--variant", default=None, help="free label for the run")


def run(args: argparse.Namespace) -> int:
    """``docker-probe`` subcommand: run, optionally write JSON, summarize."""
    result = run_docker_probe(
        DockerProbeConfig(
            repo_root=args.repo_root,
            docker=args.docker,
            expect_missing=tuple(args.expect_missing),
            variant=args.variant,
        )
    )
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    summary = ", ".join(
        f"{name} {result.facts['probes'][name]['status']}"
        for name in PROBES
        if name in result.facts["probes"]
    )
    print(
        f"docker-probe {result.status} ({result.variant}): {summary or 'no probe ran'}"
    )
    for finding in result.findings:
        print(f"  [{finding.severity}] {finding.rule}: {finding.title}")
    return result.exit_code
