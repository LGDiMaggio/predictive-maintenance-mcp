"""The CPU-only dependency file of the official Docker image.

``docker/requirements-cpu.txt`` lists what the official image installs when it
is built with ``--build-arg TORCH_VARIANT=cpu``. It is derived from ``uv.lock``,
never resolved afresh, so the image follows the same pins as every other
install of the ``full`` extra, with one difference: no NVIDIA, CUDA or Triton
component.

Three subcommands, standard library only (the image workflow runs ``check-freeze``
on a bare runner Python)::

    python -m tools.cpu_requirements generate      # rewrite the file from uv.lock
    python -m tools.cpu_requirements check         # the file still follows uv.lock
    python -m tools.cpu_requirements check-freeze --freeze freeze.txt

generate
    Runs ``uv export --frozen --no-dev --extra full --no-emit-project`` (every
    pin with its hashes), drops the NVIDIA, CUDA and Triton distributions and
    replaces ``torch==<lock version>`` with the PyTorch CPU wheel of the same
    version, as a direct URL pinned by its SHA-256, read from the PyTorch CPU
    index. The header records the ``uv.lock`` blob it was derived from.
    ``pyproject.toml`` and ``uv.lock`` are never edited.

check
    Every pin of the file is a ``uv.lock`` version with hashes the lock lists;
    no NVIDIA, CUDA or Triton distribution; torch is a ``+cpu`` wheel of the
    lock's torch version with a SHA-256.

check-freeze
    ``pip freeze`` of a built image installs exactly the file's pins (torch as
    ``<version>+cpu``) and nothing else, apart from the project itself.

The file targets ``linux/amd64`` with CPython 3.11 (``python:3.11-slim``). The
torch line has no environment marker, so an install for any other platform
fails at that line rather than silently leaving torch out.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.audit.cpu_image import is_gpu_distribution, normalize_name

__all__ = [
    "PYTORCH_CPU_INDEX",
    "REQUIREMENTS_RELPATH",
    "Entry",
    "generate",
    "is_gpu_distribution",
    "lock_closure",
    "lock_index",
    "main",
    "parse_freeze",
    "parse_requirements",
    "select_torch_wheel",
    "verify_freeze",
    "verify_requirements",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_RELPATH = "docker/requirements-cpu.txt"
LOCK_RELPATH = "uv.lock"
PROJECT_NAME = "predictive-maintenance-mcp"

#: The PyTorch CPU wheel index (a PEP 503 simple page for ``torch``).
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu/torch/"
TARGET_PYTHON = "cp311"
TARGET_PLATFORM = "manylinux_2_28_x86_64"

UV_EXPORT = (
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--extra",
    "full",
    "--no-emit-project",
    "--no-annotate",
    "--no-header",
)

_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
_URL = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)")
_HREF = re.compile(r'href="([^"]+)"')


#: Shared with the image rule, so the file and the image judge names alike.
normalize = normalize_name


def lock_closure(lock: Mapping[str, Any], extra: str = "full") -> set[str]:
    """Normalized names of everything the project and its ``extra`` need, per uv.lock.

    Every edge counts, whatever its environment marker (as ``uv export``
    keeps them, with the marker). The project itself is not in the set.

    Raises:
        ValueError: When uv.lock has no entry for the project or the extra.
    """
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no [[package]] list")
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for package in packages:
        if isinstance(package, Mapping) and isinstance(package.get("name"), str):
            by_name.setdefault(normalize(package["name"]), []).append(package)
    project = normalize(PROJECT_NAME)
    if project not in by_name:
        raise ValueError(f"uv.lock has no {PROJECT_NAME} entry")
    extras = by_name[project][0].get("optional-dependencies") or {}
    if extra not in extras:
        raise ValueError(f"uv.lock has no {extra!r} extra for {PROJECT_NAME}")
    pending: list[tuple[str, frozenset[str]]] = []

    def push(deps: Iterable[Any]) -> None:
        for dep in deps:
            if isinstance(dep, Mapping) and isinstance(dep.get("name"), str):
                pending.append(
                    (normalize(dep["name"]), frozenset(dep.get("extra") or ()))
                )

    push(by_name[project][0].get("dependencies") or ())
    push(extras[extra])
    seen: set[tuple[str, frozenset[str]]] = set()
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        name, wanted = item
        for package in by_name.get(name, []):
            push(package.get("dependencies") or ())
            optional = package.get("optional-dependencies") or {}
            for key in wanted:
                push(optional.get(key) or ())
    return {name for name, _ in seen} - {project}


@dataclass
class Entry:
    """One requirement of a requirements file: a pin or a direct URL."""

    name: str
    version: Optional[str] = None
    url: Optional[str] = None
    marker: Optional[str] = None
    hashes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return normalize(self.name)

    def render(self) -> str:
        head = (
            f"{self.name} @ {self.url}" if self.url else f"{self.name}=={self.version}"
        )
        if self.marker:
            head = f"{head} ; {self.marker}"
        lines = [head] + [f"    --hash=sha256:{value}" for value in self.hashes]
        return " \\\n".join(lines)


def _logical_lines(text: str) -> list[str]:
    joined: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        joined.append(current + line)
        current = ""
    if current.strip():
        joined.append(current)
    return joined


def parse_requirements(text: str) -> list[Entry]:
    """Parse a hash-pinned requirements file (as ``uv export`` writes it).

    Raises:
        ValueError: On a line that is neither a ``name==version`` pin nor a
            ``name @ url`` requirement.
    """
    entries: list[Entry] = []
    for line in _logical_lines(text):
        body = line.split("--hash=", 1)[0].strip()
        marker = None
        if ";" in body:
            body, marker_text = body.split(";", 1)
            body, marker = body.strip(), marker_text.strip() or None
        hashes = _HASH.findall(line)
        pin = _PIN.match(body)
        if pin and body == pin.group(0):
            entries.append(Entry(pin.group(1), pin.group(2), None, marker, hashes))
            continue
        url = _URL.match(body)
        if url and body == url.group(0):
            entries.append(Entry(url.group(1), None, url.group(2), marker, hashes))
            continue
        raise ValueError(f"unparseable requirement line: {line.strip()!r}")
    return entries


def lock_index(lock: Mapping[str, Any]) -> dict[str, dict[str, set[str]]]:
    """Map each normalized name in a parsed ``uv.lock`` to its versions and hashes.

    Raises:
        ValueError: When the ``[[package]]`` entries are malformed.
    """
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no [[package]] list")
    index: dict[str, dict[str, set[str]]] = {}
    for package in packages:
        if not isinstance(package, Mapping):
            raise ValueError("uv.lock [[package]] entry is not a table")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        files = [package.get("sdist")] + list(package.get("wheels") or [])
        hashes = {
            str(item["hash"]).removeprefix("sha256:")
            for item in files
            if isinstance(item, Mapping) and isinstance(item.get("hash"), str)
        }
        index.setdefault(normalize(name), {}).setdefault(version, set()).update(hashes)
    return index


def load_lock(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _torch_filename(version: str) -> str:
    return f"torch-{version}+cpu-{TARGET_PYTHON}-{TARGET_PYTHON}-{TARGET_PLATFORM}.whl"


def select_torch_wheel(
    index_html: str, version: str, index_url: str = PYTORCH_CPU_INDEX
) -> tuple[str, str]:
    """Pick the CPU torch wheel of ``version`` for the target from an index page.

    Returns:
        The wheel's absolute URL (without fragment) and its SHA-256 (hex).

    Raises:
        ValueError: Unless exactly one link names the wheel and carries a
            ``#sha256=`` fragment.
    """
    wanted = _torch_filename(version)
    found: list[tuple[str, str]] = []
    for href in _HREF.findall(index_html):
        url = urllib.parse.urljoin(index_url, href.replace("&amp;", "&"))
        base, _, fragment = url.partition("#")
        filename = urllib.parse.unquote(base.rsplit("/", 1)[-1])
        if filename != wanted:
            continue
        digest = fragment.removeprefix("sha256=")
        if not fragment.startswith("sha256=") or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError(f"{wanted} is listed without a sha256 fragment")
        found.append((base, digest))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one {wanted} in the index, found {len(found)}"
        )
    return found[0]


def _fetch(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "pmm-cpu-requirements"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _git_blob_id(repo_root: Path, relpath: str) -> str:
    result = subprocess.run(
        ["git", "hash-object", relpath],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _uv_export(repo_root: Path) -> str:
    result = subprocess.run(
        list(UV_EXPORT), cwd=repo_root, capture_output=True, text=True, check=True
    )
    return result.stdout


def _lock_version(index: Mapping[str, Mapping[str, set[str]]], name: str) -> str:
    versions = sorted(index.get(normalize(name), {}))
    if len(versions) != 1:
        raise ValueError(
            f"uv.lock has {len(versions)} versions of {name}, expected one"
        )
    return versions[0]


def generate(
    repo_root: Path = REPO_ROOT,
    export: Callable[[Path], str] = _uv_export,
    fetch: Callable[[str], str] = _fetch,
    blob_id: Callable[[Path, str], str] = _git_blob_id,
) -> str:
    """Build the text of ``docker/requirements-cpu.txt`` from ``uv.lock``.

    Raises:
        ValueError: When the export holds no torch pin, or the lock or index
            does not give exactly one matching torch.
    """
    entries = parse_requirements(export(repo_root))
    lock = lock_index(load_lock(repo_root / LOCK_RELPATH))
    removed = sorted(
        f"{entry.name}=={entry.version}"
        for entry in entries
        if is_gpu_distribution(entry.name)
    )
    kept = [entry for entry in entries if not is_gpu_distribution(entry.name)]
    torch = [entry for entry in kept if entry.key == "torch"]
    if len(torch) != 1 or torch[0].version is None:
        raise ValueError(
            "the uv export of the full extra must hold exactly one torch pin"
        )
    version = _lock_version(lock, "torch")
    if torch[0].version != version:
        raise ValueError(
            f"uv export pins torch {torch[0].version}, uv.lock has {version}"
        )
    url, digest = select_torch_wheel(fetch(PYTORCH_CPU_INDEX), version)
    cpu_torch = Entry("torch", None, url, None, [digest])
    body = [cpu_torch if entry.key == "torch" else entry for entry in kept]
    header = [
        "# The official Docker image's dependencies, CPU only: built with",
        "# --build-arg TORCH_VARIANT=cpu and installed with pip --require-hashes --no-deps.",
        f"# Generated by `python -m tools.cpu_requirements generate` from {LOCK_RELPATH}",
        f"# (git blob {blob_id(repo_root, LOCK_RELPATH)}): every pin of its `full` extra,",
        "# with its hashes, except the NVIDIA, CUDA and Triton distributions, and with",
        f"# torch {version} taken as the CPU wheel {_torch_filename(version)}",
        f"# from {PYTORCH_CPU_INDEX} (pinned by SHA-256).",
        f"# Target: linux/amd64, CPython 3.11. Left out ({len(removed)}): {', '.join(removed)}.",
        "# Do not edit by hand: regenerate whenever uv.lock changes.",
    ]
    return "\n".join(header) + "\n" + "\n".join(entry.render() for entry in body) + "\n"


def _torch_problems(entry: Entry, lock_versions: Iterable[str]) -> list[str]:
    problems: list[str] = []
    if entry.url is None:
        return ["torch is not a direct URL to a CPU wheel"]
    parts = urllib.parse.urlsplit(entry.url)
    filename = urllib.parse.unquote(parts.path.rsplit("/", 1)[-1])
    match = re.fullmatch(
        rf"torch-([^+-]+)\+cpu-{TARGET_PYTHON}-{TARGET_PYTHON}-{TARGET_PLATFORM}\.whl",
        filename,
    )
    host = parts.hostname or ""
    if parts.scheme != "https" or not (
        host == "pytorch.org" or host.endswith(".pytorch.org")
    ):
        problems.append(
            f"torch does not come from an https pytorch.org host: {entry.url}"
        )
    if "/whl/cpu/" not in parts.path or match is None:
        problems.append(f"torch is not a +cpu {TARGET_PLATFORM} wheel: {filename}")
    elif match.group(1) not in set(lock_versions):
        problems.append(f"torch {match.group(1)}+cpu is not the uv.lock torch version")
    if len(entry.hashes) != 1:
        problems.append("torch must carry exactly one --hash=sha256")
    return problems


def verify_requirements(text: str, lock: Mapping[str, Any]) -> list[str]:
    """Every rule the CPU dependency file must meet (empty list: it does)."""
    try:
        entries = parse_requirements(text)
    except ValueError as exc:
        return [str(exc)]
    index = lock_index(lock)
    problems: list[str] = []
    if not entries:
        problems.append("the file has no requirement")
    try:
        closure = lock_closure(lock)
    except ValueError as exc:
        return [str(exc)]
    names = {entry.key for entry in entries}
    problems += [
        f"{name} is in uv.lock's full extra but not in the file"
        for name in sorted(closure - names)
        if not is_gpu_distribution(name)
    ]
    problems += [
        f"{name} is in the file but not in uv.lock's full extra"
        for name in sorted(names - closure)
    ]
    torch = [entry for entry in entries if entry.key == "torch"]
    if len(torch) != 1:
        problems.append(f"expected exactly one torch requirement, found {len(torch)}")
    for entry in entries:
        if is_gpu_distribution(entry.name):
            problems.append(f"{entry.name} is an NVIDIA, CUDA or Triton distribution")
            continue
        if entry.key == "torch":
            problems += _torch_problems(entry, index.get("torch", {}))
            continue
        if entry.url is not None or entry.version is None:
            problems.append(f"{entry.name} is a direct URL; only torch may be one")
            continue
        if not entry.hashes:
            problems.append(f"{entry.name}=={entry.version} has no --hash")
        lock_hashes = index.get(entry.key, {}).get(entry.version)
        if lock_hashes is None:
            problems.append(f"{entry.name}=={entry.version} is not a uv.lock version")
        elif not set(entry.hashes) <= lock_hashes:
            problems.append(
                f"{entry.name}=={entry.version} has a hash uv.lock does not list"
            )
    return problems


def parse_freeze(text: str) -> dict[str, str]:
    """``pip freeze`` lines as normalized name -> version, or -> URL for a
    ``name @ url`` line (as pip writes a direct-URL install, fragment kept)."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name, url = line.split(" @ ", 1)
            pins[normalize(name.strip())] = url.strip()
        elif "==" in line:
            name, version = line.split("==", 1)
            pins[normalize(name.strip())] = version.strip()
        else:
            raise ValueError(f"unparseable pip freeze line: {line!r}")
    return pins


def _same_url(installed: str, entry: Entry) -> bool:
    """A freeze URL matches a file entry: same URL, and any ``#sha256=``
    fragment equal to the entry's hash."""
    base, _, fragment = installed.partition("#")
    if base != entry.url:
        return False
    if not fragment:
        return True
    return any(fragment == f"sha256={value}" for value in entry.hashes)


def verify_freeze(
    freeze_text: str, requirements_text: str, base_text: Optional[str] = None
) -> list[str]:
    """``pip freeze --all`` of a built image against the CPU dependency file.

    Args:
        freeze_text: ``pip freeze --all`` of the image.
        requirements_text: The CPU dependency file.
        base_text: ``pip freeze --all`` of the pinned base image. A package
            that is not in the file is accepted only when the base image
            holds it at the same version (``pip`` itself, for example).

    An entry without an environment marker must be installed at its version.
    Entries with markers (a fork by Python version, a Windows-only package)
    are not evaluated here: such a package may be absent, and when installed
    its version must be one the file lists for it. ``pip check`` in the build
    covers what the marker entries leave open.
    """
    allowed: dict[str, list[Entry]] = {}
    for entry in parse_requirements(requirements_text):
        allowed.setdefault(entry.key, []).append(entry)
    required = {
        name
        for name, entries in allowed.items()
        if any(e.marker is None for e in entries)
    }
    base = parse_freeze(base_text) if base_text is not None else {}
    installed = parse_freeze(freeze_text)
    installed.pop(normalize(PROJECT_NAME), None)
    problems = [
        f"{name} is installed but is neither in the file nor in the base image"
        for name in sorted(set(installed) - set(allowed))
        if base.get(name) != installed[name]
    ]
    problems += [
        f"{name} is in the file but is not installed"
        for name in sorted(required - set(installed))
    ]
    for name in sorted(set(installed) & set(allowed)):
        value, entries = installed[name], allowed[name]
        if name not in required and base.get(name) == value:
            continue  # listed only behind markers; the base image's own version
        if not any(
            _same_url(value, e) if e.url else value == e.version for e in entries
        ):
            listed = ", ".join(e.url or str(e.version) for e in entries)
            problems.append(f"{name} is installed as {value}, the file pins {listed}")
    problems += [
        f"{name} is an NVIDIA, CUDA or Triton distribution"
        for name in sorted(installed)
        if is_gpu_distribution(name)
    ]
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.cpu_requirements", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="rewrite the file from uv.lock")
    gen.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    gen.add_argument(
        "--output", type=Path, default=None, help=f"default: {REQUIREMENTS_RELPATH}"
    )
    chk = sub.add_parser("check", help="the file still follows uv.lock")
    chk.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    frz = sub.add_parser(
        "check-freeze", help="pip freeze of a built image against the file"
    )
    frz.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    frz.add_argument(
        "--freeze", type=Path, required=True, help="pip freeze --all of the image"
    )
    frz.add_argument(
        "--base", type=Path, default=None, help="pip freeze --all of the base image"
    )
    args = parser.parse_args(argv)
    root: Path = args.repo_root
    requirements = root / REQUIREMENTS_RELPATH
    try:
        if args.command == "generate":
            text = generate(root)
            output = args.output or requirements
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            print(f"cpu-requirements: wrote {output} (sha256 {_sha256(output)})")
            return 0
        text = requirements.read_text(encoding="utf-8")
        if args.command == "check":
            problems = verify_requirements(text, load_lock(root / LOCK_RELPATH))
        else:
            base = args.base.read_text(encoding="utf-8") if args.base else None
            problems = verify_freeze(
                args.freeze.read_text(encoding="utf-8"), text, base
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"cpu-requirements {args.command} error: {exc}", file=sys.stderr)
        return 2
    status = "pass" if not problems else "fail"
    print(f"cpu-requirements {args.command} {status}: {requirements.name}")
    for problem in problems:
        print(f"  {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
