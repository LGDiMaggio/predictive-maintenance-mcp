"""Installed-package smoke: ``python -m tools.audit package-smoke``.

Why this exists
---------------
The stdio harness (``tools/audit/mcp_smoke.py``) proves the server a host
launches from a checkout. Users never run a checkout: they install the wheel
(or, rarely, the sdist) into an environment with no repository source on its
import path. Missing package data, an import that only resolves inside a
clone, or a console script that is not declared all pass the in-tree suite
and still break that user. This check installs the built artifact into a
fresh virtual environment and runs the harness against the installed console
script.

What one run does
-----------------
1. Finds exactly one artifact of the requested kind in ``--dist``. Zero, or
   two and more, is an error: a stale artifact left beside a new one would
   otherwise be tested in its place.
2. Creates a venv under the system temp directory with this interpreter,
   upgrades pip, installs the artifact, and records ``pip list`` (names and
   versions only).
3. Runs the harness with the VENV's interpreter, in ``--installed`` mode,
   with the venv as the expected install prefix. The harness process runs
   from the repository root so ``tools`` is importable; the server it launches
   runs in its own temporary project root. Every other step runs in the
   temporary work directory, so the checkout's own egg-info metadata (left by
   ``python -m build``) can never be read as the installed version.
4. Writes one ``package-smoke`` result embedding the harness's status, facts
   and findings plus the installed versions.

Status: ``error`` when any step before the harness fails or the harness left
no usable result (an exit code that contradicts its JSON counts as none),
otherwise the harness's own status. Nothing here imports ``mcp``: the job that
runs this has only ``build`` installed, and the harness's dependencies arrive
with the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from tools.audit import results
from tools.audit.results import Finding

__all__ = [
    "ARTIFACT_PATTERNS",
    "ArtifactError",
    "PackageSmokeConfig",
    "add_arguments",
    "console_script",
    "find_artifact",
    "harness_command",
    "install_command",
    "installed_versions",
    "pip_list_command",
    "pip_upgrade_command",
    "run",
    "run_package_smoke",
    "run_subprocess",
    "venv_command",
    "venv_python",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = "predictive-maintenance-mcp"

#: Which files count as an artifact of each kind in ``--dist``.
ARTIFACT_PATTERNS = {"wheel": "*.whl", "sdist": "*.tar.gz"}

#: Backstops so a hung step ends as an ``error`` result instead of holding a
#: CI runner until the job limit. The harness has its own per-phase timeouts.
STEP_TIMEOUT = 1800.0
HARNESS_TIMEOUT = 3600.0

#: Severities whose evidence excerpt the CLI prints under the summary. Every
#: install-step failure is ``high``, and these are the findings a reader of a
#: red CI log needs the output of; lower severities are observations.
ECHOED_EXCERPT_SEVERITIES = ("critical", "high")

PathLike = Union[str, Path]
Runner = Callable[[Sequence[str], Path, float], "subprocess.CompletedProcess[str]"]


class ArtifactError(ValueError):
    """``--dist`` does not hold exactly one artifact of the requested kind."""

    def __init__(self, message: str, rule: str) -> None:
        super().__init__(message)
        self.rule = rule


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------


def find_artifact(dist: Path, kind: str) -> Path:
    """Return the one ``kind`` artifact in ``dist``.

    Raises:
        ValueError: When ``kind`` is not a key of :data:`ARTIFACT_PATTERNS`.
        ArtifactError: When ``dist`` is missing, holds no match, or holds
            several (all of them are listed).
    """
    pattern = ARTIFACT_PATTERNS.get(kind)
    if pattern is None:
        raise ValueError(
            f"artifact kind {kind!r} is not one of {sorted(ARTIFACT_PATTERNS)}"
        )
    if not dist.is_dir():
        raise ArtifactError(
            f"dist directory {dist} does not exist", "artifact-not-found"
        )
    matches = sorted(path for path in dist.glob(pattern) if path.is_file())
    if not matches:
        raise ArtifactError(f"no {kind} ({pattern}) in {dist}", "artifact-not-found")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ArtifactError(
            f"{len(matches)} {kind} files ({pattern}) in {dist}, expected exactly "
            f"one: {names}",
            "artifact-ambiguous",
        )
    return matches[0]


def venv_python(venv: Path, windows: bool) -> Path:
    """The venv's interpreter: ``Scripts/python.exe`` or ``bin/python``."""
    return venv / "Scripts" / "python.exe" if windows else venv / "bin" / "python"


def console_script(venv: Path, windows: bool) -> Path:
    """The installed console script inside ``venv``."""
    if windows:
        return venv / "Scripts" / f"{SCRIPT_NAME}.exe"
    return venv / "bin" / SCRIPT_NAME


def venv_command(python: PathLike, venv: Path) -> list[str]:
    """Create ``venv`` with ``python``'s standard-library ``venv`` module."""
    return [str(python), "-m", "venv", str(venv)]


def _pip(python: PathLike, *args: str) -> list[str]:
    command, *rest = args
    return [str(python), "-m", "pip", command, "--disable-pip-version-check", *rest]


def pip_upgrade_command(python: PathLike) -> list[str]:
    """Upgrade pip in the venv (an old bundled pip may misread new metadata)."""
    return _pip(python, "install", "--upgrade", "pip")


def install_command(python: PathLike, artifact: Path) -> list[str]:
    """Install the artifact file, resolving its dependencies from the index."""
    return _pip(python, "install", str(artifact))


def pip_list_command(python: PathLike) -> list[str]:
    """List the venv's distributions as JSON."""
    return _pip(python, "list", "--format", "json")


def harness_command(
    python: PathLike,
    script: Path,
    venv: Path,
    json_out: Path,
    variant: str,
    timeout_startup: Optional[float] = None,
    timeout_request: Optional[float] = None,
) -> list[str]:
    """The harness run against the installed console script.

    Timeouts are forwarded only when given, so the harness's own defaults
    stay the single source for them.
    """
    command = [
        str(python),
        "-m",
        "tools.audit",
        "smoke",
        "--server-cmd",
        str(script),
        "--installed",
        "--expect-installed-under",
        str(venv),
        "--python",
        str(python),
        "--json-out",
        str(json_out),
        "--variant",
        variant,
    ]
    if timeout_startup is not None:
        command += ["--timeout-startup", str(timeout_startup)]
    if timeout_request is not None:
        command += ["--timeout-request", str(timeout_request)]
    return command


def installed_versions(pip_list_json: str) -> dict[str, str]:
    """Reduce ``pip list --format json`` output to ``{name: version}``.

    Raises:
        ValueError: On output that is not a non-empty list of entries that
            each carry a name and a version. An empty listing is refused
            rather than read as "nothing to report".
    """
    return results.parse_pip_list_json(pip_list_json)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_subprocess(
    command: Sequence[str], cwd: Path, timeout: float
) -> "subprocess.CompletedProcess[str]":
    """Default runner: captured, decoded as UTF-8 with replacement."""
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@dataclass
class PackageSmokeConfig:
    """Inputs of one run (the CLI maps its flags onto this).

    ``work_dir`` and ``windows`` exist for tests: by default the work
    directory is a fresh ``tempfile.mkdtemp`` removed afterwards, and the venv
    layout follows the running platform.
    """

    dist: Path
    artifact: str = "wheel"
    variant: Optional[str] = None
    timeout_startup: Optional[float] = None
    timeout_request: Optional[float] = None
    python: str = sys.executable
    work_dir: Optional[Path] = None
    windows: bool = os.name == "nt"

    def resolved_variant(self) -> str:
        if self.variant:
            return self.variant
        major, minor = sys.version_info[:2]
        return f"{self.artifact}-{sys.platform}-py{major}.{minor}"


def _tail(proc: "subprocess.CompletedProcess[str]") -> str:
    text = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
    return results.truncate_evidence(text, keep="tail")


def _install_finding(
    rule: str,
    subject: str,
    title: str,
    classification: Optional[str] = None,
    **evidence: Any,
) -> Finding:
    return Finding(
        rule=rule,
        category="install",
        severity="high",
        subject=subject,
        title=title,
        classification=classification,
        evidence={key: value for key, value in evidence.items() if value is not None},
    )


class _Steps:
    """Runs steps through the runner, timing them and recording failures."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.findings: list[Finding] = []
        self.timing: dict[str, float] = {}

    def call(
        self, name: str, command: list[str], cwd: Path, timeout: float
    ) -> tuple[Optional["subprocess.CompletedProcess[str]"], Optional[str]]:
        """Return ``(process, None)``, or ``(None, problem)`` if it never ran."""
        started = time.perf_counter()
        try:
            return self.runner(command, cwd, timeout), None
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            self.timing[name] = round(time.perf_counter() - started, 3)

    def required(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        rule: str,
        subject: str,
        title: str,
        classification: Optional[str] = None,
    ) -> Optional["subprocess.CompletedProcess[str]"]:
        """Run a step that must exit 0; on anything else record a finding."""
        proc, problem = self.call(name, command, cwd, STEP_TIMEOUT)
        if proc is not None and proc.returncode == 0:
            return proc
        self.findings.append(
            _install_finding(
                rule,
                subject,
                title,
                classification,
                command=command,
                excerpt=(problem if proc is None else _tail(proc)) or None,
                returncode=None if proc is None else proc.returncode,
            )
        )
        return None


def _read_harness(
    path: Path,
    proc: Optional["subprocess.CompletedProcess[str]"],
    problem: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The harness result, or why there is none to trust."""
    if proc is None:
        return None, f"the harness did not run to completion: {problem}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        results.validate_result(doc)
    except FileNotFoundError:
        return None, f"the harness exited {proc.returncode} without writing a result"
    except (OSError, ValueError) as exc:
        return None, f"the harness result is unreadable: {exc}"
    if doc["check"] != "smoke":
        return None, f"the harness wrote a {doc['check']!r} result, not 'smoke'"
    expected = results.exit_code_for(doc["status"])
    if proc.returncode != expected:
        return None, (
            f"the harness exited {proc.returncode} but its result says "
            f"{doc['status']!r} (exit {expected})"
        )
    return doc, None


def _environment(
    harness: Optional[Mapping[str, Any]], installed: Optional[Mapping[str, str]]
) -> dict[str, Any]:
    """The harness's environment, with package versions taken from pip list."""
    if harness is not None:
        environment = dict(harness.get("environment") or {})
    else:
        environment = results.environment_snapshot(packages=())
    if installed:
        lookup = {
            results.canonical_name(name): version for name, version in installed.items()
        }
        packages = dict(environment.get("packages") or {})
        for name in results.DEFAULT_ENVIRONMENT_PACKAGES:
            packages[name] = lookup.get(results.canonical_name(name))
        environment["packages"] = packages
    return environment


def run_package_smoke(
    config: PackageSmokeConfig, runner: Runner = run_subprocess
) -> results.Result:
    """Install the artifact into a fresh venv, run the harness, and report.

    Raises:
        ValueError: When ``config.artifact`` is not a known kind.
    """
    if config.artifact not in ARTIFACT_PATTERNS:
        raise ValueError(
            f"artifact kind {config.artifact!r} is not one of "
            f"{sorted(ARTIFACT_PATTERNS)}"
        )
    started_at = results.utc_now()
    started = time.perf_counter()
    variant = config.resolved_variant()
    kind = config.artifact
    steps = _Steps(runner)
    facts: dict[str, Any] = {"artifact_kind": kind, "artifact": None}
    installed: Optional[dict[str, str]] = None
    harness: Optional[dict[str, Any]] = None

    owned_work_dir = config.work_dir is None
    work = (
        Path(tempfile.mkdtemp(prefix="pmm-package-smoke-"))
        if config.work_dir is None
        else config.work_dir
    )
    try:
        work.mkdir(parents=True, exist_ok=True)
        harness, installed = _install_and_smoke(config, variant, work, steps, facts)
    finally:
        if owned_work_dir:
            shutil.rmtree(work, ignore_errors=True)

    findings = list(steps.findings)
    if harness is not None:
        findings += [Finding(**item) for item in harness["findings"]]
        facts["harness"] = {
            "status": harness["status"],
            "variant": harness["variant"],
            "facts": harness["facts"],
        }
    if installed is not None:
        facts["installed_versions"] = installed
    status = (
        harness["status"] if harness is not None and not steps.findings else "error"
    )
    facts["timing_s"] = {
        **steps.timing,
        "total": round(time.perf_counter() - started, 3),
    }
    return results.Result(
        check="package-smoke",
        variant=variant,
        status=status,
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=_environment(harness, installed),
        facts=facts,
        findings=sorted(findings, key=lambda item: (item.rule, item.subject)),
    )


def _install_and_smoke(
    config: PackageSmokeConfig,
    variant: str,
    work: Path,
    steps: _Steps,
    facts: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, str]]]:
    """Steps 1-3 of the module docstring; failures land in ``steps.findings``."""
    kind = config.artifact
    try:
        artifact = find_artifact(Path(config.dist), kind)
    except ArtifactError as exc:
        steps.findings.append(
            _install_finding(
                exc.rule,
                f"{kind} artifact",
                str(exc),
                classification="not-checked",
                paths=[str(config.dist)],
            )
        )
        return None, None
    facts["artifact"] = artifact.name

    venv = work / "venv"
    python = venv_python(venv, config.windows)
    if not steps.required(
        "venv",
        venv_command(config.python, venv),
        work,
        "venv-create-failed",
        "fresh virtual environment",
        "A fresh virtual environment could not be created",
        "not-checked",
    ):
        return None, None
    if not steps.required(
        "pip_upgrade",
        pip_upgrade_command(python),
        work,
        "pip-upgrade-failed",
        "pip in a fresh virtual environment",
        "pip could not be upgraded in the fresh virtual environment",
        "not-checked",
    ):
        return None, None
    if not steps.required(
        "install",
        install_command(python, artifact),
        work,
        "install-failed",
        f"{kind} install into a fresh virtual environment",
        f"The built {kind} does not install into a fresh virtual environment",
    ):
        return None, None

    versions_unavailable = (
        "installed-versions-unavailable",
        "installed versions",
        "pip list did not report the installed versions",
        "not-checked",
    )
    listing = steps.required(
        "pip_list", pip_list_command(python), work, *versions_unavailable
    )
    if listing is None:
        return None, None
    try:
        installed = installed_versions(listing.stdout)
    except ValueError as exc:
        steps.findings.append(_install_finding(*versions_unavailable, excerpt=str(exc)))
        return None, None

    script = console_script(venv, config.windows)
    if not script.is_file():
        steps.findings.append(
            _install_finding(
                "console-script-missing",
                f"{SCRIPT_NAME} console script",
                f"The install did not create the {SCRIPT_NAME} console script",
                paths=[str(script)],
            )
        )
        return None, installed

    out = work / "harness-result.json"
    command = harness_command(
        python,
        script,
        venv,
        out,
        variant,
        timeout_startup=config.timeout_startup,
        timeout_request=config.timeout_request,
    )
    proc, problem = steps.call("harness", command, REPO_ROOT, HARNESS_TIMEOUT)
    if proc is not None and proc.stdout:
        print(proc.stdout.rstrip("\n"))
    harness, why = _read_harness(out, proc, problem)
    if harness is None:
        detail = [why or "", _tail(proc) if proc is not None else ""]
        steps.findings.append(
            _install_finding(
                "harness-no-result",
                "smoke harness result",
                "The smoke harness produced no usable result",
                "not-checked",
                command=command,
                excerpt="\n".join(part for part in detail if part),
            )
        )
    return harness, installed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not seconds > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return seconds


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``package-smoke`` subcommand's flags."""
    parser.add_argument(
        "--dist",
        type=Path,
        required=True,
        help="directory holding exactly one artifact of the chosen kind",
    )
    parser.add_argument(
        "--artifact",
        choices=sorted(ARTIFACT_PATTERNS),
        default="wheel",
        help="which artifact to install (default: wheel)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument("--variant", default=None, help="free label for the run")
    parser.add_argument(
        "--timeout-startup",
        type=_positive_seconds,
        default=None,
        help="forwarded to the harness (default: the harness's own)",
    )
    parser.add_argument(
        "--timeout-request",
        type=_positive_seconds,
        default=None,
        help="forwarded to the harness (default: the harness's own)",
    )


def run(args: argparse.Namespace) -> int:
    """``package-smoke`` subcommand: run, optionally write JSON, summarize."""
    config = PackageSmokeConfig(
        dist=args.dist,
        artifact=args.artifact,
        variant=args.variant,
        timeout_startup=args.timeout_startup,
        timeout_request=args.timeout_request,
    )
    result = run_package_smoke(config)
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    harness = result.facts.get("harness") or {}
    print(
        f"package-smoke {result.status} ({result.variant}): "
        f"{result.facts.get('artifact')}, harness {harness.get('status', 'not run')}"
    )
    for finding in result.findings:
        print(f"  [{finding.severity}] {finding.rule}: {finding.title}")
        excerpt = finding.evidence.get("excerpt")
        if excerpt and finding.severity in ECHOED_EXCERPT_SEVERITIES:
            print("    " + "\n    ".join(str(excerpt).splitlines()))
    return result.exit_code
