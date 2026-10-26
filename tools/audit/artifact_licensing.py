"""Licensing check of the built artifacts: ``python -m tools.audit artifact-licensing``.

Why this exists
---------------
The project's code is MIT. The repository also holds third-party sample data
under CC BY-NC-SA 4.0, described in ``data/LICENSE``. That data must never
reach an artifact labelled MIT, and the ``LICENSE`` an artifact carries must be
the MIT License alone. Neither property can be read from ``MANIFEST.in`` or
``pyproject.toml``: setuptools decides what an sdist contains from the files
on disk, so this check opens the wheel and the sdist that were actually built.

What one run checks
-------------------
1. ``--dist`` holds exactly one wheel and exactly one sdist, both readable and
   non-empty. Every sdist member must sit under one ``<name>-<version>/`` root,
   which is stripped before any rule looks at a path.
2. No member lies under ``data/``, no wheel member uses the ``*.data/``
   scheme, and no member is a ``.mat`` file.
3. No member, under any name, is an exact copy of a sample file. The committed
   manifest (:data:`MANIFEST_RELPATH`) lists each sample file's SHA-256 and
   sizes; a member is hashed only when its size matches, as stored and with
   CRLF line endings converted to LF, so a copy made on Windows still matches.
4. The metadata declares MIT (``License-Expression: MIT``, or ``License: MIT``
   with no expression), names no ``License ::`` classifier other than the MIT
   one, and lists ``LICENSE`` as its only license file, and the wheel carries
   that one file in its dist-info.
5. Every ``LICENSE`` copy, the repository's included, is the MIT License text
   with one copyright line and nothing else, and the shipped copies equal the
   repository's (line endings aside).
6. The wheel, the sdist and ``pyproject.toml`` name the same version.

Status: ``error`` when any input is missing, ambiguous or unreadable (no
evidence is never a pass), ``fail`` on any rule violation, ``pass`` otherwise.

Archive mode
------------
``--archive`` (repeatable) checks source archives instead: the zip and the
tarball GitHub serves for a commit, a tag or a release, which a Zenodo deposit
stores unchanged, are ``git archive`` output filtered by the committed
``.gitattributes``. Each archive is checked on its own:

1. It is a readable, non-empty ``.zip`` or ``.tar.gz`` whose members sit under
   one root folder (``git archive --prefix``, or GitHub's
   ``<owner>-<repo>-<commit>/``), which is stripped.
2. Under ``data/`` it holds exactly :data:`ARCHIVE_DATA_FILES`: the notices and
   the placeholder that keeps ``data/signals/`` in an extracted tree. Rule 3
   above applies to every member, and no member is a ``.mat`` file.
3. Its root ``LICENSE`` is its only root license file and passes rule 5.
4. Its ``.gitattributes`` export-ignores every manifest root, and no member
   lies under a path it export-ignores, so the archive is proven to come from
   a revision that excludes the sample data and to have been filtered by it.

The manifest
------------
The manifest is the one place that names the third-party data: its roots, its
license, its notice file, and a hash per file. Regenerate it after the sample
files change, from a checkout that holds them::

    python -c "from tools.audit import artifact_licensing as a; a.write_manifest()"

Only the standard library is used: the jobs that run this install nothing
beyond ``build``.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import posixpath
import re
import stat
import subprocess
import tarfile
import tomllib
import zipfile
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

from tools.audit import results
from tools.audit.package_smoke import ArtifactError, find_artifact
from tools.audit.results import Finding

__all__ = [
    "ARCHIVE_DATA_FILES",
    "LICENSE_FILE",
    "MANIFEST_RELPATH",
    "MIT_CLASSIFIER",
    "MIT_LICENSE_LINES",
    "ArchiveLicensingConfig",
    "Artifact",
    "ArtifactLicensingConfig",
    "Manifest",
    "add_arguments",
    "build_manifest",
    "check_archive",
    "check_artifact",
    "content_hashes",
    "export_ignored_paths",
    "load_manifest",
    "manifest_entry",
    "mit_license_problem",
    "read_sdist",
    "read_source_archive",
    "read_wheel",
    "run",
    "run_archive_licensing",
    "run_artifact_licensing",
    "write_manifest",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK = "artifact-licensing"
CATEGORY = "licensing"

#: Where the third-party data manifest lives, relative to the repository root.
MANIFEST_RELPATH = "tools/audit/third_party_data.json"

#: The MIT License as the repository carries it; ``None`` marks the line that
#: holds the copyright notice.
MIT_LICENSE_LINES: tuple[Optional[str], ...] = (
    "MIT License",
    "",
    None,
    "",
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    'of this software and associated documentation files (the "Software"), to deal',
    "in the Software without restriction, including without limitation the rights",
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
    "copies of the Software, and to permit persons to whom the Software is",
    "furnished to do so, subject to the following conditions:",
    "",
    "The above copyright notice and this permission notice shall be included in all",
    "copies or substantial portions of the Software.",
    "",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,",
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE",
    "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER",
    "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,",
    "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE",
    "SOFTWARE.",
)

_COPYRIGHT_LINE = re.compile(r"Copyright \(c\) \d{4}(?:-\d{4})? \S.*")

#: The only license file an artifact may carry, and the repository's own.
LICENSE_FILE = "LICENSE"

#: The only ``License ::`` classifier an artifact may declare.
MIT_CLASSIFIER = "License :: OSI Approved :: MIT License"

#: File names setuptools ships as license files by default.
_LICENSE_NAME = r"(?:LICEN[CS]E|COPYING|NOTICE|AUTHORS)[^/]*"
_DIST_INFO = r"[^/]+\.dist-info"

#: What a source archive holds under data/: the two notices, and the
#: placeholder that keeps data/signals/ in an extracted tree, where the server
#: finds its project root (src/config.py) and compose mounts the folder.
ARCHIVE_DATA_FILES = ("data/LICENSE", "data/README.md", "data/signals/.gitkeep")

GITATTRIBUTES = ".gitattributes"

#: The only export-ignore pattern form the archive mode reads: a plain path
#: anchored at the repository root, no glob, quote or trailing slash.
_PLAIN_ANCHORED = re.compile(r"/[^*?\[\]\\!\"\s]*[^*?\[\]\\!\"\s/]")

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------


def _normalize_newlines(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n")


def _lines(text: str) -> list[str]:
    """``text`` as lines, ignoring line endings and trailing whitespace."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def manifest_entry(content: bytes) -> dict[str, Any]:
    """Hash and sizes of one sample file, as the manifest stores them.

    Text (no NUL byte) is hashed with LF line endings and recorded with both
    its LF and its CRLF size; binary content is hashed and sized as is.
    """
    if b"\x00" in content:
        return {"sha256": hashlib.sha256(content).hexdigest(), "sizes": [len(content)]}
    lf = _normalize_newlines(content)
    return {
        "sha256": hashlib.sha256(lf).hexdigest(),
        "sizes": [len(lf), len(lf) + lf.count(b"\n")],
    }


def content_hashes(content: bytes) -> set[str]:
    """The hashes a member could match: as stored, and with LF line endings."""
    hashes = {hashlib.sha256(content).hexdigest()}
    if b"\r\n" in content:
        hashes.add(hashlib.sha256(_normalize_newlines(content)).hexdigest())
    return hashes


def mit_license_problem(text: str) -> Optional[str]:
    """Why ``text`` is not the MIT License with one copyright line, or None.

    Line endings and trailing whitespace are ignored; any other difference,
    including text appended after the warranty disclaimer, is a problem.
    """
    lines = _lines(text)
    if len(lines) != len(MIT_LICENSE_LINES):
        extra = len(lines) - len(MIT_LICENSE_LINES)
        if extra > 0:
            return f"{extra} line(s) follow or interrupt the MIT License text"
        return f"{-extra} line(s) of the MIT License text are missing"
    for number, (line, expected) in enumerate(zip(lines, MIT_LICENSE_LINES), 1):
        if expected is None:
            if not _COPYRIGHT_LINE.fullmatch(line):
                return f"line {number} is not a copyright notice: {line!r}"
        elif line != expected:
            return f"line {number} differs from the MIT License text: {line!r}"
    return None


@dataclass(frozen=True)
class Manifest:
    """The third-party data the artifacts must not carry.

    ``by_hash`` maps each content hash to the first file listed with it:
    several sample files (the metadata of recordings taken under the same
    conditions) have identical content.
    """

    roots: tuple[str, ...]
    by_hash: Mapping[str, str]
    sizes: frozenset[int]
    file_count: int


def load_manifest(path: Path) -> Manifest:
    """Read the committed manifest.

    Raises:
        ValueError: When it is missing, not JSON, names no roots or no files,
            or holds a malformed entry.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} does not exist") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is unreadable: {exc}") from None
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    roots = doc.get("roots")
    files = doc.get("files")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{path} names no data roots")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"{path} lists no files")
    by_hash: dict[str, str] = {}
    sizes: set[int] = set()
    for name, entry in files.items():
        fields = entry if isinstance(entry, Mapping) else {}
        digest = fields.get("sha256")
        entry_sizes = fields.get("sizes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: entry {name!r} has no valid sha256")
        if not isinstance(entry_sizes, list) or not entry_sizes:
            raise ValueError(f"{path}: entry {name!r} has no sizes")
        if not all(isinstance(size, int) and size >= 0 for size in entry_sizes):
            raise ValueError(f"{path}: entry {name!r} has an invalid size")
        by_hash.setdefault(digest, name)
        sizes.update(entry_sizes)
    return Manifest(
        roots=tuple(roots),
        by_hash=by_hash,
        sizes=frozenset(sizes),
        file_count=len(files),
    )


def _tracked_files(repo_root: Path, roots: Sequence[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return sorted(name for name in proc.stdout.decode("utf-8").split("\0") if name)


def build_manifest(
    repo_root: Path, roots: Sequence[str], **fields: Any
) -> dict[str, Any]:
    """The manifest document for the tracked files under ``roots``.

    Raises:
        ValueError: When no tracked file lies under the roots.
    """
    names = _tracked_files(repo_root, roots)
    if not names:
        raise ValueError(f"no tracked file under {list(roots)} in {repo_root}")
    files = {name: manifest_entry((repo_root / name).read_bytes()) for name in names}
    return {**fields, "roots": list(roots), "files": files}


def write_manifest(repo_root: PathLike = REPO_ROOT) -> Path:
    """Rebuild the committed manifest from the tracked files under its roots.

    Its other top-level fields (license, notice) are kept as they are.
    """
    root = Path(repo_root)
    path = root / MANIFEST_RELPATH
    current = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        key: value for key, value in current.items() if key not in ("roots", "files")
    }
    return results.write_json(build_manifest(root, current["roots"], **fields), path)


@dataclass
class Artifact:
    """What the rules need from one built artifact.

    ``members`` maps each root-relative path that is not a directory to its
    size. ``candidates`` holds the content of the regular files whose size
    matches a manifest size, ``licenses`` the text of every license file.
    A source archive (``kind="archive"``) has empty metadata and keeps its
    root ``.gitattributes`` in ``attributes``.
    """

    kind: str
    name: str
    metadata: Message
    members: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, bytes] = field(default_factory=dict)
    licenses: dict[str, str] = field(default_factory=dict)
    attributes: Optional[str] = None


def _parse_metadata(content: bytes) -> Message:
    return email.parser.Parser().parsestr(content.decode("utf-8"))


def read_wheel(path: Path, sizes: Iterable[int] = ()) -> Artifact:
    """Read a wheel's members, metadata and license files.

    Raises:
        ValueError: When it is not a readable zip, holds no member, or has no
            single ``*.dist-info/METADATA``.
    """
    wanted = set(sizes)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not infos:
                raise ValueError(f"{path.name} holds no files")
            metadata_names = [
                info.filename
                for info in infos
                if re.fullmatch(rf"{_DIST_INFO}/METADATA", info.filename)
            ]
            if len(metadata_names) != 1:
                raise ValueError(
                    f"{path.name} has {len(metadata_names)} dist-info METADATA files"
                )
            dist_info = metadata_names[0].split("/", 1)[0]
            artifact = Artifact(
                kind="wheel",
                name=path.name,
                metadata=_parse_metadata(archive.read(metadata_names[0])),
            )
            for info in infos:
                artifact.members[info.filename] = info.file_size
                if info.file_size in wanted:
                    artifact.candidates[info.filename] = archive.read(info)
                if info.filename.startswith(f"{dist_info}/licenses/") or re.fullmatch(
                    rf"{re.escape(dist_info)}/{_LICENSE_NAME}", info.filename
                ):
                    artifact.licenses[info.filename] = archive.read(info).decode(
                        "utf-8", errors="replace"
                    )
    except (
        zipfile.BadZipFile,
        OSError,
        zlib.error,
        UnicodeDecodeError,
        NotImplementedError,  # an unsupported compression method
        RuntimeError,  # an encrypted member
    ) as exc:
        raise ValueError(f"{path.name} is not a readable wheel: {exc}") from None
    return artifact


def read_sdist(path: Path, sizes: Iterable[int] = ()) -> Artifact:
    """Read an sdist's members (root stripped), PKG-INFO and root license files.

    Raises:
        ValueError: When it is not a readable tar.gz, holds no file, its
            members do not share one root folder, or it has no PKG-INFO.
    """
    wanted = set(sizes)
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            files = [member for member in members if member.isfile()]
            if not files:
                raise ValueError(f"{path.name} holds no files")
            roots = {member.name.split("/", 1)[0] for member in members}
            if len(roots) != 1 or not all("/" in member.name for member in files):
                raise ValueError(
                    f"{path.name} does not keep its members under one root folder"
                )
            root = roots.pop()
            relative: dict[str, tarfile.TarInfo] = {}
            for member in members:
                if member.name == root:
                    continue
                relative[member.name[len(root) + 1 :]] = member
            pkg_info = relative.get("PKG-INFO")
            if pkg_info is None or not pkg_info.isfile():
                raise ValueError(f"{path.name} has no PKG-INFO at its root")
            artifact = Artifact(
                kind="sdist",
                name=path.name,
                metadata=_parse_metadata(_extract(archive, pkg_info)),
            )
            for name, member in relative.items():
                if member.isdir():
                    continue
                # Links are listed so the path rules see them; only regular
                # files have content to hash.
                artifact.members[name] = member.size
                if not member.isfile():
                    continue
                if member.size in wanted:
                    artifact.candidates[name] = _extract(archive, member)
                if re.fullmatch(_LICENSE_NAME, name):
                    artifact.licenses[name] = _extract(archive, member).decode(
                        "utf-8", errors="replace"
                    )
    except (tarfile.TarError, OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name} is not a readable sdist: {exc}") from None
    return artifact


def _extract(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"member {member.name} cannot be read")
    return handle.read()


def read_source_archive(path: Path, sizes: Iterable[int] = ()) -> Artifact:
    """Read a source archive's members (root stripped), root license files and
    ``.gitattributes``.

    Raises:
        ValueError: When it is not a ``.zip`` or ``.tar.gz``, cannot be read,
            holds no file, or its members do not share one root folder.
    """
    wanted = set(sizes)
    lowered = path.name.casefold()
    try:
        if lowered.endswith(".zip"):
            return _read_zip_archive(path, wanted)
        if lowered.endswith((".tar.gz", ".tgz")):
            return _read_tar_archive(path, wanted)
    except (
        zipfile.BadZipFile,
        tarfile.TarError,
        OSError,
        EOFError,
        zlib.error,
        UnicodeDecodeError,
        NotImplementedError,  # an unsupported compression method
        RuntimeError,  # an encrypted member
    ) as exc:
        raise ValueError(
            f"{path.name} is not a readable source archive: {exc}"
        ) from None
    raise ValueError(f"{path.name} is neither a .zip nor a .tar.gz archive")


def _archive_root(entries: Sequence[tuple[str, bool]], label: str) -> str:
    """The one folder every ``(name, is_dir)`` entry sits under."""
    files = [name for name, is_dir in entries if not is_dir]
    if not files:
        raise ValueError(f"{label} holds no files")
    roots = {name.split("/", 1)[0] for name, _ in entries}
    if len(roots) != 1 or not all("/" in name for name in files):
        raise ValueError(f"{label} does not keep its members under one root folder")
    return roots.pop()


def _keep_content(
    artifact: Artifact,
    relative: str,
    size: int,
    wanted: set[int],
    read: Callable[[], bytes],
) -> None:
    is_attributes = relative == GITATTRIBUTES
    is_license = re.fullmatch(_LICENSE_NAME, relative) is not None
    if size not in wanted and not (is_attributes or is_license):
        return
    content = read()
    if size in wanted:
        artifact.candidates[relative] = content
    if is_attributes:
        artifact.attributes = content.decode("utf-8")
    elif is_license:
        artifact.licenses[relative] = content.decode("utf-8", errors="replace")


def _read_zip_archive(path: Path, wanted: set[int]) -> Artifact:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        root = _archive_root(
            [(info.filename, info.is_dir()) for info in infos], path.name
        )
        artifact = Artifact(kind="archive", name=path.name, metadata=Message())
        for info in infos:
            relative = info.filename[len(root) + 1 :]
            if info.is_dir() or not relative:
                continue
            artifact.members[relative] = info.file_size
            # git archive stores a symlink as a member holding its target;
            # it is listed for the path rules, never hashed.
            if not stat.S_ISLNK(info.external_attr >> 16):
                _keep_content(
                    artifact,
                    relative,
                    info.file_size,
                    wanted,
                    partial(archive.read, info),
                )
    return artifact


def _read_tar_archive(path: Path, wanted: set[int]) -> Artifact:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        root = _archive_root(
            [(member.name, member.isdir()) for member in members], path.name
        )
        artifact = Artifact(kind="archive", name=path.name, metadata=Message())
        for member in members:
            relative = member.name[len(root) + 1 :]
            if member.isdir() or not relative:
                continue
            artifact.members[relative] = member.size
            if member.isfile():
                _keep_content(
                    artifact,
                    relative,
                    member.size,
                    wanted,
                    partial(_extract, archive, member),
                )
    return artifact


def _finding(
    rule: str,
    subject: str,
    title: str,
    severity: str = "high",
    classification: Optional[str] = None,
    **evidence: Any,
) -> Finding:
    return Finding(
        rule=rule,
        category=CATEGORY,
        severity=severity,
        subject=subject,
        title=title,
        classification=classification,
        evidence={key: value for key, value in evidence.items() if value is not None},
    )


def _data_path_reason(kind: str, path: str) -> Optional[str]:
    # Compared after normalizing (./data, data//x) and case-folding (Data/ on a
    # case-insensitive file system), the way an installer would resolve it.
    normalized = posixpath.normpath(path)
    first = normalized.split("/", 1)[0].casefold()
    if first == "data":
        if kind == "archive" and normalized in ARCHIVE_DATA_FILES:
            return None
        return "lies under data/"
    if kind == "wheel" and first.endswith(".data"):
        return "uses the wheel data scheme"
    if path.casefold().endswith(".mat"):
        return "is a .mat recording"
    return None


def _license_metadata_problem(metadata: Message) -> Optional[str]:
    problems = []
    expression = metadata.get("License-Expression")
    legacy = metadata.get("License")
    if expression is not None:
        if expression.strip() != "MIT":
            problems.append(f"License-Expression is {expression.strip()!r}, not 'MIT'")
    elif legacy is None:
        problems.append("neither License-Expression nor License is declared")
    elif legacy.strip() != "MIT":
        problems.append(f"License is {legacy.strip()!r}, not 'MIT'")
    problems += [
        f"the classifier {classifier!r} is not the MIT one"
        for classifier in metadata.get_all("Classifier") or []
        if classifier.startswith("License ::") and classifier != MIT_CLASSIFIER
    ]
    return "; ".join(problems) or None


def _source(artifact: Artifact) -> dict[str, str]:
    """The evidence key naming what was checked; report.py reproduces by it."""
    key = "archive" if artifact.kind == "archive" else "artifact"
    return {key: artifact.name}


def _label(artifact: Artifact) -> str:
    """How titles name the artifact: one run checks two source archives."""
    if artifact.kind == "archive":
        return f"archive {artifact.name}"
    return artifact.kind


def _data_findings(artifact: Artifact, manifest: Manifest) -> list[Finding]:
    """Members that are, or copy, third-party data (rules 2 and 3)."""
    kind = artifact.kind
    label = _label(artifact)
    findings: list[Finding] = []
    for path in sorted(artifact.members):
        reason = _data_path_reason(kind, path)
        if reason is not None:
            findings.append(
                _finding(
                    "third-party-data-path",
                    f"{label} member {path}",
                    f"The {label} contains {path}, which {reason}",
                    member=path,
                    **_source(artifact),
                )
            )
            continue
        content = artifact.candidates.get(path)
        if content is None:
            continue
        match = next(
            (
                manifest.by_hash[digest]
                for digest in sorted(content_hashes(content))
                if digest in manifest.by_hash
            ),
            None,
        )
        if match is not None:
            findings.append(
                _finding(
                    "third-party-data-content",
                    f"{label} member {path}",
                    f"The {label} member {path} is a copy of the sample file {match}",
                    member=path,
                    matches=match,
                    **_source(artifact),
                )
            )
    return findings


def _expected_license_file(artifact: Artifact) -> Optional[str]:
    """Where the artifact's LICENSE is, or None when it has none.

    A wheel keeps it under dist-info/licenses/ (Metadata 2.4) or, from an
    older setuptools, at the dist-info root; an sdist at its root.
    """
    if artifact.kind == "sdist":
        return LICENSE_FILE if LICENSE_FILE in artifact.licenses else None
    return next(
        (
            name
            for name in sorted(artifact.licenses)
            if name.endswith(f"/licenses/{LICENSE_FILE}")
            or re.fullmatch(rf"{_DIST_INFO}/{re.escape(LICENSE_FILE)}", name)
        ),
        None,
    )


def check_artifact(
    artifact: Artifact, manifest: Manifest, repo_license: str
) -> list[Finding]:
    """Every rule violation in one artifact (rules 2 to 5 of the docstring)."""
    kind = artifact.kind
    findings = _data_findings(artifact, manifest)

    problem = _license_metadata_problem(artifact.metadata)
    if problem is not None:
        findings.append(
            _finding(
                "license-metadata",
                f"{kind} license metadata",
                f"The {kind} metadata does not declare MIT: {problem}",
                artifact=artifact.name,
            )
        )

    declared = sorted(artifact.metadata.get_all("License-File") or [])
    shipped = sorted(artifact.licenses)
    expected_file = _expected_license_file(artifact)
    file_problems = []
    if declared != [LICENSE_FILE]:
        file_problems.append(
            f"License-File entries are {declared}, expected {[LICENSE_FILE]}"
        )
    if expected_file is None:
        file_problems.append("the LICENSE file is missing")
    others = [name for name in shipped if name != expected_file]
    if others:
        file_problems.append(f"other license files are present: {others}")
    if file_problems:
        findings.append(
            _finding(
                "license-files",
                f"{kind} license files",
                f"The {kind} does not carry LICENSE as its only license file",
                artifact=artifact.name,
                problems=file_problems,
            )
        )

    if expected_file is not None:
        text = artifact.licenses[expected_file]
        problem = mit_license_problem(text)
        if problem is None and _lines(text) != _lines(repo_license):
            problem = "it differs from the repository LICENSE"
        if problem is not None:
            findings.append(
                _finding(
                    "license-text",
                    f"{kind} LICENSE",
                    f"The {kind} LICENSE is not the repository's MIT License: {problem}",
                    artifact=artifact.name,
                    member=expected_file,
                )
            )
    return findings


def export_ignored_paths(text: str) -> tuple[set[str], list[str]]:
    """The paths a ``.gitattributes`` text export-ignores, and the lines that
    touch ``export-ignore`` in a form this check cannot read.

    Only a plain path anchored at the root (``/data/real_bearings``) is read;
    a glob, an unanchored, quoted or trailing-slash pattern, a value, or a
    line that unsets the attribute is returned as unreadable, never guessed.
    """
    paths: set[str] = set()
    unreadable: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *attributes = line.split()
        touching = [
            attribute
            for attribute in attributes
            if attribute.lstrip("-!").split("=", 1)[0] == "export-ignore"
        ]
        if not touching:
            continue
        if touching != ["export-ignore"] or not _PLAIN_ANCHORED.fullmatch(pattern):
            unreadable.append(line)
            continue
        paths.add(pattern[1:])
    return paths, unreadable


def _attribute_findings(artifact: Artifact, manifest: Manifest) -> list[Finding]:
    """The archive's own .gitattributes excludes the data, and was applied."""
    label = _label(artifact)
    source = _source(artifact)
    if artifact.attributes is None:
        return [
            _finding(
                "archive-attributes",
                f"{label} .gitattributes",
                f"The {label} has no .gitattributes, so no rule of its revision "
                "excludes the sample data",
                **source,
            )
        ]
    ignored, unreadable = export_ignored_paths(artifact.attributes)
    findings: list[Finding] = []
    if unreadable:
        findings.append(
            _finding(
                "archive-attributes",
                f"{label} .gitattributes",
                f"The {label} has export-ignore lines this check cannot read",
                lines=unreadable,
                **source,
            )
        )
    uncovered = [root for root in manifest.roots if root not in ignored]
    if uncovered:
        findings.append(
            _finding(
                "archive-attributes",
                f"{label} .gitattributes",
                f"The {label} .gitattributes does not export-ignore "
                + ", ".join(uncovered),
                roots=uncovered,
                **source,
            )
        )
    for path in sorted(ignored):
        leaked = sorted(
            member
            for member in artifact.members
            if member == path or member.startswith(f"{path}/")
        )
        if leaked:
            findings.append(
                _finding(
                    "archive-attributes",
                    f"{label} export-ignore {path}",
                    f"The {label} holds {len(leaked)} file(s) under {path}, which "
                    "its .gitattributes export-ignores",
                    members=leaked[:20],
                    **source,
                )
            )
    return findings


def check_archive(
    artifact: Artifact, manifest: Manifest, repo_license: str
) -> list[Finding]:
    """Every rule violation in one source archive (rules of the archive mode)."""
    label = _label(artifact)
    source = _source(artifact)
    findings = _data_findings(artifact, manifest)

    missing = [path for path in ARCHIVE_DATA_FILES if path not in artifact.members]
    if missing:
        findings.append(
            _finding(
                "archive-layout",
                f"{label} data/ files",
                f"The {label} lacks " + ", ".join(missing),
                missing=missing,
                **source,
            )
        )

    problems = []
    if LICENSE_FILE not in artifact.licenses:
        problems.append("the LICENSE file is missing")
    others = sorted(name for name in artifact.licenses if name != LICENSE_FILE)
    if others:
        problems.append(f"other license files are present at its root: {others}")
    if problems:
        findings.append(
            _finding(
                "license-files",
                f"{label} license files",
                f"The {label} does not carry LICENSE as its only root license file",
                problems=problems,
                **source,
            )
        )
    if LICENSE_FILE in artifact.licenses:
        text = artifact.licenses[LICENSE_FILE]
        problem = mit_license_problem(text)
        if problem is None and _lines(text) != _lines(repo_license):
            problem = "it differs from the repository LICENSE"
        if problem is not None:
            findings.append(
                _finding(
                    "license-text",
                    f"{label} LICENSE",
                    f"The {label} LICENSE is not the repository's MIT License: "
                    f"{problem}",
                    member=LICENSE_FILE,
                    **source,
                )
            )

    return findings + _attribute_findings(artifact, manifest)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class ArtifactLicensingConfig:
    """Inputs of one run (the CLI maps its flags onto this)."""

    dist: Path
    repo_root: Path = REPO_ROOT
    variant: Optional[str] = None


def _error(rule: str, subject: str, title: str) -> Finding:
    """An input that could not be checked: never a pass."""
    return _finding(rule, subject, title, classification="not-checked")


def _project_version(repo_root: Path) -> Optional[str]:
    """``[project] version`` from ``pyproject.toml``, or None when it has none.

    Raises:
        ValueError: When the file cannot be read or is not valid TOML.
    """
    path = repo_root / "pyproject.toml"
    try:
        with open(path, "rb") as handle:
            version = tomllib.load(handle).get("project", {}).get("version")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path} cannot be read: {exc}") from None
    return version if isinstance(version, str) else None


def _reference_inputs(
    repo_root: Path, errors: list[Finding], facts: dict[str, Any]
) -> tuple[Optional[Manifest], Optional[str]]:
    """The data manifest and the repository LICENSE text; each one that cannot
    be read adds an error instead."""
    manifest: Optional[Manifest] = None
    try:
        manifest = load_manifest(repo_root / MANIFEST_RELPATH)
        facts["manifest_files"] = manifest.file_count
        facts["manifest_roots"] = list(manifest.roots)
    except ValueError as exc:
        errors.append(
            _error("manifest-unusable", "third-party data manifest", str(exc))
        )

    repo_license: Optional[str] = None
    try:
        repo_license = (repo_root / LICENSE_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(
            _error(
                "repo-license-unreadable",
                "repository LICENSE",
                f"The repository LICENSE cannot be read: {exc}",
            )
        )
    return manifest, repo_license


def _repo_license_findings(repo_license: str) -> list[Finding]:
    problem = mit_license_problem(repo_license)
    if problem is None:
        return []
    return [
        _finding(
            "license-text",
            "repository LICENSE",
            f"The repository LICENSE is not the MIT License alone: {problem}",
        )
    ]


def _result(
    variant: Optional[str],
    started_at: str,
    facts: dict[str, Any],
    errors: list[Finding],
    findings: list[Finding],
) -> results.Result:
    status = "error" if errors else ("fail" if findings else "pass")
    return results.Result(
        check=CHECK,
        variant=variant or "local",
        status=status,
        started_at=started_at,
        finished_at=results.utc_now(),
        environment=results.environment_snapshot(packages=()),
        facts=facts,
        findings=sorted(findings, key=lambda item: (item.rule, item.subject)),
    )


def run_artifact_licensing(config: ArtifactLicensingConfig) -> results.Result:
    """Inspect the wheel and the sdist in ``config.dist`` and report."""
    started_at = results.utc_now()
    repo_root = Path(config.repo_root)
    errors: list[Finding] = []
    facts: dict[str, Any] = {"wheel": None, "sdist": None}
    manifest, repo_license = _reference_inputs(repo_root, errors, facts)

    project_version: Optional[str] = None
    try:
        project_version = _project_version(repo_root)
    except ValueError as exc:
        errors.append(_error("pyproject-unreadable", "pyproject.toml", str(exc)))

    artifacts: list[Artifact] = []
    sizes = manifest.sizes if manifest is not None else frozenset()
    for kind, reader in (("wheel", read_wheel), ("sdist", read_sdist)):
        try:
            path = find_artifact(Path(config.dist), kind)
        except ArtifactError as exc:
            errors.append(_error(exc.rule, f"{kind} artifact", str(exc)))
            continue
        facts[kind] = path.name
        try:
            artifacts.append(reader(path, sizes))
        except ValueError as exc:
            errors.append(_error("artifact-unreadable", f"{kind} artifact", str(exc)))

    findings: list[Finding] = list(errors)
    if not errors and manifest is not None and repo_license is not None:
        facts["members"] = {
            artifact.kind: len(artifact.members) for artifact in artifacts
        }
        facts["license_metadata"] = {
            artifact.kind: {
                "License-Expression": artifact.metadata.get("License-Expression"),
                "License": artifact.metadata.get("License"),
                "License-File": artifact.metadata.get_all("License-File") or [],
            }
            for artifact in artifacts
        }
        findings += _repo_license_findings(repo_license)
        for artifact in artifacts:
            findings += check_artifact(artifact, manifest, repo_license)
        versions = {
            artifact.kind: artifact.metadata.get("Version") for artifact in artifacts
        }
        versions["pyproject"] = project_version
        facts["versions"] = versions
        if None in versions.values() or len(set(versions.values())) != 1:
            findings.append(
                _finding(
                    "version-mismatch",
                    "artifact versions",
                    "The wheel, the sdist and pyproject.toml name different versions",
                    severity="medium",
                    versions=versions,
                )
            )

    return _result(config.variant, started_at, facts, errors, findings)


@dataclass
class ArchiveLicensingConfig:
    """Inputs of one archive-mode run (``--archive``, repeatable)."""

    archives: Sequence[Path]
    repo_root: Path = REPO_ROOT
    variant: Optional[str] = None


def run_archive_licensing(config: ArchiveLicensingConfig) -> results.Result:
    """Inspect every source archive in ``config.archives`` and report."""
    started_at = results.utc_now()
    archives = [Path(path) for path in config.archives]
    errors: list[Finding] = []
    facts: dict[str, Any] = {"archives": [path.name for path in archives]}
    manifest, repo_license = _reference_inputs(Path(config.repo_root), errors, facts)

    if not archives:
        errors.append(
            _error(
                "artifact-not-found", "source archive", "No source archive was given"
            )
        )
    artifacts: list[Artifact] = []
    sizes = manifest.sizes if manifest is not None else frozenset()
    for path in archives:
        if not path.is_file():
            errors.append(
                _error(
                    "artifact-not-found",
                    f"archive {path.name}",
                    f"{path} does not exist",
                )
            )
            continue
        try:
            artifacts.append(read_source_archive(path, sizes))
        except ValueError as exc:
            errors.append(
                _error("artifact-unreadable", f"archive {path.name}", str(exc))
            )

    findings: list[Finding] = list(errors)
    if not errors and manifest is not None and repo_license is not None:
        facts["members"] = {
            artifact.name: len(artifact.members) for artifact in artifacts
        }
        facts["export_ignore"] = {
            artifact.name: (
                sorted(export_ignored_paths(artifact.attributes)[0])
                if artifact.attributes is not None
                else None
            )
            for artifact in artifacts
        }
        findings += _repo_license_findings(repo_license)
        for artifact in artifacts:
            findings += check_archive(artifact, manifest, repo_license)

    return _result(config.variant, started_at, facts, errors, findings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``artifact-licensing`` subcommand's flags."""
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--dist",
        type=Path,
        help="directory holding exactly one wheel and exactly one sdist",
    )
    inputs.add_argument(
        "--archive",
        type=Path,
        action="append",
        help=(
            "a source archive (.zip or .tar.gz, one root folder) as git archive "
            "or GitHub writes it; repeatable"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository holding LICENSE, pyproject.toml and the data manifest",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="result path")
    parser.add_argument("--variant", default=None, help="free label for the run")


def run(args: argparse.Namespace) -> int:
    """``artifact-licensing`` subcommand: run, optionally write JSON, summarize."""
    if args.archive:
        result = run_archive_licensing(
            ArchiveLicensingConfig(
                archives=tuple(args.archive),
                repo_root=args.repo_root,
                variant=args.variant,
            )
        )
        checked = ", ".join(result.facts.get("archives") or [])
    else:
        result = run_artifact_licensing(
            ArtifactLicensingConfig(
                dist=args.dist, repo_root=args.repo_root, variant=args.variant
            )
        )
        checked = f"{result.facts.get('wheel')}, {result.facts.get('sdist')}"
    if args.json_out is not None:
        results.write_json(result, args.json_out)
    print(f"{CHECK} {result.status} ({result.variant}): {checked}")
    for finding in result.findings:
        print(f"  [{finding.severity}] {finding.rule}: {finding.title}")
    return result.exit_code
