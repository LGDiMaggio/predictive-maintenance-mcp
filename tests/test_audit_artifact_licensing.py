"""Guards for the artifact licensing check (``python -m tools.audit artifact-licensing``).

The project's code is MIT, but the repository also holds third-party sample
data under CC BY-NC-SA 4.0 (``data/LICENSE``). A packaging change that lets the
data into the wheel or the sdist, or a ``LICENSE`` that grows anything beyond
the MIT text, ships material the MIT label does not cover. The check reads the
built artifacts themselves, never ``MANIFEST.in``, because only the artifact
shows what a user receives.

The real run is proven by the CI step on freshly built artifacts. This module
pins everything that decides the verdict, on synthetic archives:

* every rule fires on the case it exists for: a path under ``data/`` (with the
  sdist's ``<name>-<version>/`` root stripped), a wheel ``*.data/`` member, a
  ``.mat`` file, a renamed copy of a sample file with either line ending, a
  license text with anything beyond the MIT text, a second license file,
  non-MIT metadata, and disagreeing versions;
* every way of producing no evidence (a missing or extra artifact, an
  unreadable or empty archive, a missing or empty data manifest, an unreadable
  repository ``LICENSE``) ends as ``error``, never ``pass``;
* the archive mode, on the source archives GitHub serves and Zenodo stores:
  only the notices and the ``data/signals/`` placeholder under ``data/``, the
  root ``LICENSE`` alone, and a ``.gitattributes`` that export-ignores every
  sample folder and was applied, in zip and tarball form, including the real
  output of ``git archive``;
* the image mode, on synthetic container filesystems: nothing in the folders
  an image keeps empty, only the allowed resource files, the MIT ``LICENSE``
  in the image and in the project's dist-info, no copy of a sample file, and
  no ``.mat`` file except a library's own recorded test file, each exemption
  condition proven necessary by dropping it;
* the committed manifest still describes the tracked sample files, and the
  repository ``LICENSE``, ``data/LICENSE`` and ``.gitattributes`` say what the
  check assumes.
"""

import base64
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from _research_data import require_recordings
from tools.audit import __main__ as audit_main
from tools.audit import artifact_licensing, report, results

REPO_ROOT = Path(__file__).resolve().parents[1]

NAME = "predictive_maintenance_mcp"
VERSION = "0.13.0"
WHEEL_NAME = f"{NAME}-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"{NAME}-{VERSION}.tar.gz"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
SDIST_ROOT = f"{NAME}-{VERSION}"

COPYRIGHT = "Copyright (c) 2025 Test Author"

MIT_BODY = """MIT License

{copyright}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DATA_NOTE = """
---

Note on Sample Data:

The sample bearing vibration data in the `data/` directory is licensed under
CC BY-NC-SA 4.0.
"""

REGENERATE = (
    "regenerate the manifest with: python -c "
    '"from tools.audit import artifact_licensing as a; a.write_manifest()"'
)

#: A sample "recording" in the manifest: text with LF line endings.
SAMPLE_CSV = b"0.125\n-0.250\n0.375\n-0.500\n"
SAMPLE_CSV_PATH = "data/signals/real_train/baseline_1.csv"
#: A binary sample, as the .mat files are.
SAMPLE_MAT = b"MATLAB 5.0 MAT-file\x00\x01\x02\x03"
SAMPLE_MAT_PATH = "data/real_bearings/train/baseline_1.mat"


def _mit(copyright_line: str = COPYRIGHT, newline: str = "\n") -> str:
    return MIT_BODY.format(copyright=copyright_line).replace("\n", newline)


def _metadata(
    license_field="MIT",
    expression=None,
    license_files=("LICENSE",),
    version=VERSION,
    classifiers=("License :: OSI Approved :: MIT License",),
) -> str:
    lines = [
        "Metadata-Version: 2.4",
        "Name: predictive-maintenance-mcp",
        f"Version: {version}",
    ]
    if expression is not None:
        lines.append(f"License-Expression: {expression}")
    if license_field is not None:
        lines.append(f"License: {license_field}")
    lines += [f"Classifier: {classifier}" for classifier in classifiers]
    lines += [f"License-File: {name}" for name in license_files]
    return "\n".join(lines) + "\n\n"


def _wheel(dist: Path, members=None, metadata=None, license_text=None) -> Path:
    """A wheel holding the package, its METADATA and ``licenses/LICENSE``."""
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / WHEEL_NAME
    files = {
        f"{NAME}/__init__.py": b"",
        f"{DIST_INFO}/METADATA": (metadata or _metadata()).encode(),
        f"{DIST_INFO}/licenses/LICENSE": (license_text or _mit()).encode(),
    }
    files.update(members or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            if content is not None:
                archive.writestr(name, content)
    return path


def _sdist(dist: Path, members=None, pkg_info=None, license_text=None, root=SDIST_ROOT):
    """An sdist with every member under ``root/``, as setuptools builds it."""
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / SDIST_NAME
    files = {
        "PKG-INFO": (pkg_info or _metadata()).encode(),
        "LICENSE": (license_text or _mit()).encode(),
        f"src/{NAME}/__init__.py": b"",
    }
    files.update(members or {})
    with tarfile.open(path, "w:gz") as archive:
        if root:
            directory = tarfile.TarInfo(root)
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        for name, content in files.items():
            if content is None:
                continue
            info = tarfile.TarInfo(f"{root}/{name}" if root else name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def _entry(content: bytes) -> dict:
    """A manifest entry computed independently of the module under test."""
    if b"\x00" in content:
        return {"sha256": hashlib.sha256(content).hexdigest(), "sizes": [len(content)]}
    lf = content.replace(b"\r\n", b"\n")
    return {
        "sha256": hashlib.sha256(lf).hexdigest(),
        "sizes": [len(lf), len(lf) + lf.count(b"\n")],
    }


def _repo(
    tmp_path: Path, license_text=None, version=VERSION, manifest="default"
) -> Path:
    """A repository root with ``LICENSE``, ``pyproject.toml`` and the manifest."""
    root = tmp_path / "repo"
    (root / "tools" / "audit").mkdir(parents=True)
    (root / "LICENSE").write_text(license_text or _mit(), encoding="utf-8", newline="")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "predictive-maintenance-mcp"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    if manifest == "default":
        manifest = {
            "license": "CC-BY-NC-SA-4.0",
            "notice": "data/LICENSE",
            "roots": ["data/real_bearings", "data/signals/real_train"],
            "files": {
                SAMPLE_CSV_PATH: _entry(SAMPLE_CSV),
                SAMPLE_MAT_PATH: _entry(SAMPLE_MAT),
            },
        }
    if manifest is not None:
        text = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (root / artifact_licensing.MANIFEST_RELPATH).write_text(text, encoding="utf-8")
    return root


def _run(tmp_path: Path, repo=None, dist=None) -> results.Result:
    config = artifact_licensing.ArtifactLicensingConfig(
        dist=dist or tmp_path / "dist",
        repo_root=repo or _repo(tmp_path),
    )
    return artifact_licensing.run_artifact_licensing(config)


def _rules(result: results.Result) -> set:
    return {finding.rule for finding in result.findings}


def _clean_dist(tmp_path: Path, wheel=None, sdist=None) -> Path:
    dist = tmp_path / "dist"
    _wheel(dist, **(wheel or {}))
    _sdist(dist, **(sdist or {}))
    return dist


# ---------------------------------------------------------------------------
# A clean pair passes
# ---------------------------------------------------------------------------


class TestCleanArtifacts:
    def test_mit_metadata_and_mit_license_pass(self, tmp_path):
        result = _run(tmp_path, dist=_clean_dist(tmp_path))
        assert (result.check, result.status) == ("artifact-licensing", "pass")
        assert result.findings == []
        assert result.facts["wheel"] == WHEEL_NAME
        assert result.facts["sdist"] == SDIST_NAME

    def test_a_license_expression_without_the_legacy_field_passes(self, tmp_path):
        metadata = _metadata(license_field=None, expression="MIT")
        dist = _clean_dist(
            tmp_path, wheel={"metadata": metadata}, sdist={"pkg_info": metadata}
        )
        assert _run(tmp_path, dist=dist).status == "pass"

    def test_crlf_line_endings_in_every_license_copy_pass(self, tmp_path):
        crlf = _mit(newline="\r\n")
        dist = _clean_dist(
            tmp_path, wheel={"license_text": crlf}, sdist={"license_text": crlf}
        )
        assert _run(
            tmp_path, repo=_repo(tmp_path, license_text=crlf), dist=dist
        ).status == ("pass")

    def test_project_code_and_tests_are_not_mistaken_for_data(self, tmp_path):
        dist = _clean_dist(
            tmp_path,
            sdist={
                "members": {
                    "tests/test_signal_loader.py": b"DATA = 'data/signals/real_train'\n",
                    "src/predictive_maintenance_mcp/data_paths.py": b"x = 1\n",
                }
            },
        )
        assert _run(tmp_path, dist=dist).status == "pass"


# ---------------------------------------------------------------------------
# Sample data in an artifact fails
# ---------------------------------------------------------------------------


class TestThirdPartyData:
    def test_an_sdist_data_member_fails_after_the_root_is_stripped(self, tmp_path):
        dist = _clean_dist(tmp_path, sdist={"members": {SAMPLE_CSV_PATH: SAMPLE_CSV}})
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        paths = [f.evidence["member"] for f in result.findings]
        assert SAMPLE_CSV_PATH in paths
        assert {f.rule for f in result.findings} == {"third-party-data-path"}

    def test_any_file_under_data_fails_not_only_the_sample_roots(self, tmp_path):
        dist = _clean_dist(
            tmp_path, sdist={"members": {"data/signals/cwru/97.npy": b"\x93NUMPY"}}
        )
        assert "third-party-data-path" in _rules(_run(tmp_path, dist=dist))

    def test_a_wheel_data_scheme_member_fails(self, tmp_path):
        dist = _clean_dist(
            tmp_path, wheel={"members": {f"{NAME}-{VERSION}.data/data/x.csv": b"1\n"}}
        )
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        assert "third-party-data-path" in _rules(result)

    def test_a_mat_file_anywhere_fails(self, tmp_path):
        dist = _clean_dist(
            tmp_path, sdist={"members": {"tests/fixtures/recording.mat": b"\x00mat"}}
        )
        assert "third-party-data-path" in _rules(_run(tmp_path, dist=dist))

    def test_a_renamed_crlf_copy_of_a_sample_fails_on_content(self, tmp_path):
        copy = SAMPLE_CSV.replace(b"\n", b"\r\n")
        dist = _clean_dist(
            tmp_path, sdist={"members": {"tests/fixtures/copy.csv": copy}}
        )
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        (finding,) = result.findings
        assert finding.rule == "third-party-data-content"
        assert finding.evidence["member"] == "tests/fixtures/copy.csv"
        assert finding.evidence["matches"] == SAMPLE_CSV_PATH

    def test_a_renamed_copy_of_a_binary_sample_fails_on_content(self, tmp_path):
        # Not ".bin": that suffix is refused as a model weight file (rule 7).
        dist = _clean_dist(
            tmp_path,
            wheel={"members": {f"{NAME}/resources/blob.dat": SAMPLE_MAT}},
        )
        result = _run(tmp_path, dist=dist)
        assert [f.rule for f in result.findings] == ["third-party-data-content"]

    def test_a_link_under_data_in_the_sdist_fails(self, tmp_path):
        dist = _clean_dist(tmp_path)
        sdist = dist / SDIST_NAME
        with tarfile.open(sdist, "r:gz") as archive:
            members = [(m, archive.extractfile(m)) for m in archive.getmembers()]
            payload = [(m, f.read() if f else None) for m, f in members]
        with tarfile.open(sdist, "w:gz") as archive:
            for member, content in payload:
                archive.addfile(member, io.BytesIO(content) if content else None)
            link = tarfile.TarInfo(f"{SDIST_ROOT}/{SAMPLE_CSV_PATH}")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../../README.md"
            archive.addfile(link)
        result = _run(tmp_path, dist=dist)
        assert [f.rule for f in result.findings] == ["third-party-data-path"]

    @pytest.mark.parametrize(
        "member", ["./data/x.csv", "data//x.csv", "Data/x.csv", "DATA/signals/y.json"]
    )
    def test_a_data_path_is_matched_after_normalizing_it(self, tmp_path, member):
        dist = _clean_dist(tmp_path, sdist={"members": {member: b"1\n"}})
        assert "third-party-data-path" in _rules(_run(tmp_path, dist=dist))

    def test_a_file_of_matching_size_but_other_content_passes(self, tmp_path):
        other = b"9" * len(SAMPLE_CSV)
        dist = _clean_dist(tmp_path, sdist={"members": {"tests/fixtures/x.csv": other}})
        assert _run(tmp_path, dist=dist).status == "pass"


# ---------------------------------------------------------------------------
# License text, license files and metadata
# ---------------------------------------------------------------------------


class TestLicense:
    def test_a_data_note_under_the_mit_text_fails_even_when_every_copy_agrees(
        self, tmp_path
    ):
        noted = _mit() + DATA_NOTE
        dist = _clean_dist(
            tmp_path, wheel={"license_text": noted}, sdist={"license_text": noted}
        )
        result = _run(tmp_path, repo=_repo(tmp_path, license_text=noted), dist=dist)
        assert result.status == "fail"
        subjects = {f.subject for f in result.findings if f.rule == "license-text"}
        assert subjects == {"repository LICENSE", "wheel LICENSE", "sdist LICENSE"}

    def test_a_shipped_license_that_differs_from_the_repository_fails(self, tmp_path):
        other = _mit(copyright_line="Copyright (c) 2024 Someone Else")
        dist = _clean_dist(tmp_path, wheel={"license_text": other})
        result = _run(tmp_path, dist=dist)
        assert [(f.rule, f.subject) for f in result.findings] == [
            ("license-text", "wheel LICENSE")
        ]

    def test_a_license_without_a_copyright_line_fails(self, tmp_path):
        bare = _mit(copyright_line="")
        dist = _clean_dist(
            tmp_path, wheel={"license_text": bare}, sdist={"license_text": bare}
        )
        result = _run(tmp_path, repo=_repo(tmp_path, license_text=bare), dist=dist)
        assert "license-text" in _rules(result)

    def test_a_second_license_file_in_the_metadata_fails(self, tmp_path):
        metadata = _metadata(license_files=("LICENSE", "LICENSE-DATA"))
        dist = _clean_dist(
            tmp_path, wheel={"metadata": metadata}, sdist={"pkg_info": metadata}
        )
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        assert {f.subject for f in result.findings} == {
            "wheel license files",
            "sdist license files",
        }

    def test_a_second_file_in_the_wheel_license_folder_fails(self, tmp_path):
        dist = _clean_dist(
            tmp_path,
            wheel={"members": {f"{DIST_INFO}/licenses/data/LICENSE": b"CC BY-NC-SA"}},
        )
        assert "license-files" in _rules(_run(tmp_path, dist=dist))

    def test_a_wheel_without_a_license_file_fails(self, tmp_path):
        dist = tmp_path / "dist"
        _sdist(dist)
        _wheel(dist, members={f"{DIST_INFO}/licenses/LICENSE": None})
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        assert "license-files" in _rules(result)

    @pytest.mark.parametrize(
        "license_field, expression",
        [("Other", None), (None, "MIT AND CC-BY-NC-SA-4.0"), (None, None)],
    )
    def test_metadata_that_does_not_say_mit_fails(
        self, tmp_path, license_field, expression
    ):
        metadata = _metadata(license_field=license_field, expression=expression)
        dist = _clean_dist(tmp_path, wheel={"metadata": metadata})
        result = _run(tmp_path, dist=dist)
        assert [(f.rule, f.subject) for f in result.findings] == [
            ("license-metadata", "wheel license metadata")
        ]

    @pytest.mark.parametrize(
        "edit, message",
        [
            (
                lambda text: text.replace(
                    "free of charge, to any person",
                    "free of charge, for non-commercial use, to any person",
                ),
                "differs from the MIT License text",
            ),
            (lambda text: "\n".join(text.split("\n")[:-5]) + "\n", "missing"),
        ],
        ids=["reworded-line", "truncated"],
    )
    def test_an_edit_inside_the_mit_text_fails_even_when_every_copy_agrees(
        self, tmp_path, edit, message
    ):
        edited = edit(_mit())
        assert message in artifact_licensing.mit_license_problem(edited)
        dist = _clean_dist(
            tmp_path, wheel={"license_text": edited}, sdist={"license_text": edited}
        )
        result = _run(tmp_path, repo=_repo(tmp_path, license_text=edited), dist=dist)
        assert result.status == "fail"
        subjects = {f.subject for f in result.findings if f.rule == "license-text"}
        assert subjects == {"repository LICENSE", "wheel LICENSE", "sdist LICENSE"}

    def test_a_non_mit_license_classifier_fails(self, tmp_path):
        metadata = _metadata(
            classifiers=(
                "License :: OSI Approved :: MIT License",
                "License :: Free for non-commercial use",
            )
        )
        dist = _clean_dist(tmp_path, sdist={"pkg_info": metadata})
        result = _run(tmp_path, dist=dist)
        assert [(f.rule, f.subject) for f in result.findings] == [
            ("license-metadata", "sdist license metadata")
        ]
        assert "Free for non-commercial use" in result.findings[0].title

    def test_no_license_classifier_at_all_passes(self, tmp_path):
        metadata = _metadata(classifiers=())
        dist = _clean_dist(
            tmp_path, wheel={"metadata": metadata}, sdist={"pkg_info": metadata}
        )
        assert _run(tmp_path, dist=dist).status == "pass"

    @pytest.mark.parametrize("name", ["NOTICE", "COPYING", "LICENSE-DATA", "AUTHORS"])
    def test_a_second_license_file_at_the_sdist_root_fails(self, tmp_path, name):
        dist = _clean_dist(tmp_path, sdist={"members": {name: b"other terms\n"}})
        result = _run(tmp_path, dist=dist)
        (finding,) = result.findings
        assert (finding.rule, finding.subject) == (
            "license-files",
            "sdist license files",
        )
        assert name in " ".join(finding.evidence["problems"])

    def test_a_wheel_with_the_license_at_the_dist_info_root_passes(self, tmp_path):
        """The layout setuptools wrote before Metadata 2.4."""
        dist = tmp_path / "dist"
        _sdist(dist)
        _wheel(
            dist,
            members={
                f"{DIST_INFO}/licenses/LICENSE": None,
                f"{DIST_INFO}/LICENSE": _mit().encode(),
            },
        )
        assert _run(tmp_path, dist=dist).status == "pass"

    def test_artifact_versions_must_agree_with_each_other(self, tmp_path):
        dist = _clean_dist(tmp_path, sdist={"pkg_info": _metadata(version="0.12.0")})
        result = _run(tmp_path, dist=dist)
        assert [f.rule for f in result.findings] == ["version-mismatch"]

    def test_artifacts_that_agree_with_each_other_must_agree_with_pyproject(
        self, tmp_path
    ):
        stale = _metadata(version="0.12.0")
        dist = _clean_dist(
            tmp_path, wheel={"metadata": stale}, sdist={"pkg_info": stale}
        )
        result = _run(tmp_path, dist=dist)
        assert [f.rule for f in result.findings] == ["version-mismatch"]
        assert result.facts["versions"]["pyproject"] == VERSION

    def test_versions_missing_everywhere_do_not_agree(self, tmp_path):
        metadata = _metadata().replace(f"Version: {VERSION}\n", "")
        dist = _clean_dist(
            tmp_path, wheel={"metadata": metadata}, sdist={"pkg_info": metadata}
        )
        repo = _repo(tmp_path)
        (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        result = _run(tmp_path, repo=repo, dist=dist)
        assert [f.rule for f in result.findings] == ["version-mismatch"]


# ---------------------------------------------------------------------------
# No evidence is never a pass
# ---------------------------------------------------------------------------


class TestNoEvidence:
    @pytest.mark.parametrize("present", ["wheel", "sdist", "neither"])
    def test_a_missing_artifact_is_an_error(self, tmp_path, present):
        dist = tmp_path / "dist"
        dist.mkdir()
        if present == "wheel":
            _wheel(dist)
        elif present == "sdist":
            _sdist(dist)
        result = _run(tmp_path, dist=dist)
        assert result.status == "error"
        assert "artifact-not-found" in _rules(result)

    def test_two_wheels_is_an_error(self, tmp_path):
        dist = _clean_dist(tmp_path)
        (dist / f"{NAME}-0.12.0-py3-none-any.whl").write_bytes(
            (dist / WHEEL_NAME).read_bytes()
        )
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-ambiguous"})

    def test_an_empty_zip_is_an_error(self, tmp_path):
        dist = tmp_path / "dist"
        _sdist(dist)
        with zipfile.ZipFile(dist / WHEEL_NAME, "w"):
            pass
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    @pytest.mark.parametrize("raised", [NotImplementedError, RuntimeError])
    def test_a_wheel_member_zipfile_cannot_read_is_an_error(
        self, tmp_path, monkeypatch, raised
    ):
        """Unsupported compression and encrypted members end as ``error``."""
        dist = _clean_dist(tmp_path)

        def refuse(self, name, pwd=None):
            raise raised("cannot read this member")

        monkeypatch.setattr(zipfile.ZipFile, "read", refuse)
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_a_truncated_sdist_is_an_error(self, tmp_path):
        dist = _clean_dist(tmp_path)
        sdist = dist / SDIST_NAME
        sdist.write_bytes(sdist.read_bytes()[:40])
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_sdist_without_one_common_root_is_an_error(self, tmp_path):
        dist = tmp_path / "dist"
        _wheel(dist)
        _sdist(dist, root="")
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    @pytest.mark.parametrize(
        "manifest",
        [
            None,
            "{not json",
            {"roots": [], "files": {}},
            {"roots": ["data/real_bearings"], "files": {}},
        ],
        ids=["missing", "invalid-json", "no-roots", "no-files"],
    )
    def test_an_unusable_manifest_is_an_error(self, tmp_path, manifest):
        repo = _repo(tmp_path, manifest=manifest)
        result = _run(tmp_path, repo=repo, dist=_clean_dist(tmp_path))
        assert (result.status, _rules(result)) == ("error", {"manifest-unusable"})

    def test_a_missing_repository_license_is_an_error(self, tmp_path):
        repo = _repo(tmp_path)
        (repo / "LICENSE").unlink()
        result = _run(tmp_path, repo=repo, dist=_clean_dist(tmp_path))
        assert (result.status, _rules(result)) == ("error", {"repo-license-unreadable"})

    @pytest.mark.parametrize("content", [None, "[project\nversion = "])
    def test_a_missing_or_invalid_pyproject_is_an_error(self, tmp_path, content):
        repo = _repo(tmp_path)
        if content is None:
            (repo / "pyproject.toml").unlink()
        else:
            (repo / "pyproject.toml").write_text(content, encoding="utf-8")
        result = _run(tmp_path, repo=repo, dist=_clean_dist(tmp_path))
        assert (result.status, _rules(result)) == ("error", {"pyproject-unreadable"})

    @pytest.mark.parametrize("count", [0, 2])
    def test_a_wheel_without_exactly_one_metadata_file_is_an_error(
        self, tmp_path, count
    ):
        members = {f"{DIST_INFO}/METADATA": None}
        if count == 2:
            members = {"other-1.0.dist-info/METADATA": _metadata().encode()}
        dist = tmp_path / "dist"
        _sdist(dist)
        _wheel(dist, members=members)
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_sdist_without_pkg_info_is_an_error(self, tmp_path):
        dist = tmp_path / "dist"
        _wheel(dist)
        _sdist(dist, members={"PKG-INFO": None})
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_sdist_with_no_regular_file_is_an_error(self, tmp_path):
        dist = tmp_path / "dist"
        _wheel(dist)
        dist.mkdir(exist_ok=True)
        with tarfile.open(dist / SDIST_NAME, "w:gz") as archive:
            directory = tarfile.TarInfo(SDIST_ROOT)
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        result = _run(tmp_path, dist=dist)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})


# ---------------------------------------------------------------------------
# Source archives (archive mode)
# ---------------------------------------------------------------------------

ARCHIVE_ROOT = "predictive-maintenance-mcp"

#: The export-ignore lines for the roots of the test manifest in _repo().
ATTRIBUTES = (
    "/data/real_bearings export-ignore\n/data/signals/real_train export-ignore\n"
)


def _archive_files(members=None, attributes=ATTRIBUTES, license_text=None) -> dict:
    files = {
        "LICENSE": (license_text or _mit()).encode(),
        ".gitattributes": attributes.encode() if attributes is not None else None,
        "README.md": b"# Predictive Maintenance MCP\n",
        f"src/{NAME}/__init__.py": b"",
        "data/LICENSE": b"Sample bearing vibration data\n",
        "data/README.md": b"# Real Bearing Vibration Dataset\n",
        "data/signals/.gitkeep": b"",
    }
    files.update(members or {})
    return {name: content for name, content in files.items() if content is not None}


def _source_zip(path: Path, root=ARCHIVE_ROOT, links=(), **files) -> Path:
    """A zip as git archive writes one: a root folder, then every member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{root}/" if root else ""
    with zipfile.ZipFile(path, "w") as archive:
        if root:
            archive.writestr(zipfile.ZipInfo(prefix), b"")
        for name, content in _archive_files(**files).items():
            archive.writestr(f"{prefix}{name}", content)
        for name, target in links:
            info = zipfile.ZipInfo(f"{prefix}{name}")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, target)
    return path


def _source_tar(path: Path, root=ARCHIVE_ROOT, **files) -> Path:
    """A tarball as git archive writes one: a root folder, then every member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{root}/" if root else ""
    with tarfile.open(path, "w:gz") as archive:
        if root:
            directory = tarfile.TarInfo(root)
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        for name, content in _archive_files(**files).items():
            info = tarfile.TarInfo(f"{prefix}{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


ARCHIVE_WRITERS = {
    "zip": (_source_zip, "source.zip"),
    "tar.gz": (_source_tar, "source.tar.gz"),
}


def _source_archive(tmp_path: Path, fmt: str, **files) -> Path:
    writer, name = ARCHIVE_WRITERS[fmt]
    return writer(tmp_path / "archives" / name, **files)


def _run_archives(tmp_path: Path, *archives, repo=None) -> results.Result:
    config = artifact_licensing.ArchiveLicensingConfig(
        archives=archives, repo_root=repo or _repo(tmp_path)
    )
    return artifact_licensing.run_archive_licensing(config)


@pytest.fixture(params=sorted(ARCHIVE_WRITERS))
def fmt(request):
    """Every archive rule holds for the zip and for the tarball."""
    return request.param


class TestSourceArchives:
    def test_a_clean_archive_passes(self, tmp_path, fmt):
        result = _run_archives(tmp_path, _source_archive(tmp_path, fmt))
        assert (result.status, result.findings) == ("pass", [])
        assert result.facts["export_ignore"] == {
            f"source.{fmt}": ["data/real_bearings", "data/signals/real_train"]
        }

    def test_both_formats_pass_together(self, tmp_path):
        result = _run_archives(
            tmp_path,
            _source_archive(tmp_path, "zip"),
            _source_archive(tmp_path, "tar.gz"),
        )
        assert result.status == "pass"
        assert sorted(result.facts["members"]) == ["source.tar.gz", "source.zip"]

    def test_a_sample_file_fails_and_names_the_archive(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, members={SAMPLE_CSV_PATH: SAMPLE_CSV})
        result = _run_archives(tmp_path, archive)
        assert result.status == "fail"
        paths = [f for f in result.findings if f.rule == "third-party-data-path"]
        assert [f.evidence["member"] for f in paths] == [SAMPLE_CSV_PATH]
        assert paths[0].evidence["archive"] == archive.name
        assert archive.name in paths[0].title
        # The export-ignore rule was not applied either.
        assert "archive-attributes" in _rules(result)

    def test_a_new_dataset_folder_fails_even_outside_the_manifest_roots(
        self, tmp_path, fmt
    ):
        archive = _source_archive(
            tmp_path, fmt, members={"data/other_dataset/x.csv": b"1\n"}
        )
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"third-party-data-path"})

    @pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
    def test_a_renamed_copy_of_a_sample_fails_on_content(self, tmp_path, fmt, newline):
        copy = SAMPLE_CSV.replace(b"\n", newline)
        archive = _source_archive(
            tmp_path, fmt, members={"tests/fixtures/copy.csv": copy}
        )
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"third-party-data-content"})
        assert result.findings[0].evidence["member"] == "tests/fixtures/copy.csv"

    def test_a_mat_file_outside_data_fails(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, members={"assets/x.mat": b"MAT"})
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"third-party-data-path"})

    def test_a_case_variant_of_an_allowed_data_file_fails(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, members={"Data/README.md": b"x"})
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"third-party-data-path"})

    def test_a_link_under_data_fails_without_being_read(self, tmp_path):
        archive = _source_zip(
            tmp_path / "archives" / "source.zip",
            links=[("data/signals/real_test", "../../README.md")],
        )
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"third-party-data-path"})

    @pytest.mark.parametrize("missing", artifact_licensing.ARCHIVE_DATA_FILES)
    def test_a_missing_notice_or_placeholder_fails(self, tmp_path, fmt, missing):
        archive = _source_archive(tmp_path, fmt, members={missing: None})
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"archive-layout"})
        assert result.findings[0].evidence["missing"] == [missing]

    def test_a_missing_root_license_fails(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, members={"LICENSE": None})
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"license-files"})

    @pytest.mark.parametrize("name", ["COPYING", "NOTICE", "LICENSE-DATA"])
    def test_a_second_root_license_file_fails(self, tmp_path, fmt, name):
        archive = _source_archive(tmp_path, fmt, members={name: b"CC BY-NC-SA"})
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"license-files"})

    def test_the_data_note_under_the_mit_text_fails(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, license_text=_mit() + DATA_NOTE)
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"license-text"})

    def test_an_mit_license_that_is_not_the_repositorys_fails(self, tmp_path, fmt):
        """A valid MIT text with another copyright line passes the template, so
        only the comparison with the repository LICENSE catches it."""
        other = _mit("Copyright (c) 2025 Someone Else")
        assert artifact_licensing.mit_license_problem(other) is None
        archive = _source_archive(tmp_path, fmt, license_text=other)
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"license-text"})

    def test_an_archive_without_gitattributes_fails(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt, attributes=None)
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"archive-attributes"})

    def test_attributes_that_leave_a_manifest_root_out_fail(self, tmp_path, fmt):
        archive = _source_archive(
            tmp_path, fmt, attributes="/data/real_bearings export-ignore\n"
        )
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"archive-attributes"})
        assert result.findings[0].evidence["roots"] == ["data/signals/real_train"]

    @pytest.mark.parametrize(
        "line",
        [
            "/data/real_bearings/** export-ignore",
            "data/real_bearings export-ignore",
            "/data/real_bearings/ export-ignore",
            "/data/real_bearings -export-ignore",
        ],
        ids=["glob", "unanchored", "trailing-slash", "unset"],
    )
    def test_an_export_ignore_line_it_cannot_read_fails(self, tmp_path, fmt, line):
        attributes = f"{ATTRIBUTES}{line}\n"
        archive = _source_archive(tmp_path, fmt, attributes=attributes)
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"archive-attributes"})
        assert result.findings[0].evidence["lines"] == [line]

    def test_files_under_an_export_ignored_path_fail(self, tmp_path, fmt):
        """An archive server that ignored .gitattributes is caught by the
        attributes rule even where the content rule has no hash to match."""
        attributes = f"{ATTRIBUTES}/assets/demo.gif export-ignore\n"
        archive = _source_archive(
            tmp_path, fmt, attributes=attributes, members={"assets/demo.gif": b"GIF89a"}
        )
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("fail", {"archive-attributes"})
        assert result.findings[0].evidence["members"] == ["assets/demo.gif"]


class TestExportIgnoredPaths:
    def test_reads_plain_anchored_paths_and_skips_other_attributes(self):
        text = (
            "# comment\n\n*.py text eol=lf\n"
            "/data/real_bearings export-ignore\n"
            "/assets/a.gif export-ignore binary\n"
        )
        paths, unreadable = artifact_licensing.export_ignored_paths(text)
        assert (paths, unreadable) == ({"data/real_bearings", "assets/a.gif"}, [])

    @pytest.mark.parametrize(
        "line",
        [
            '"/data/real bearings" export-ignore',
            "/data/*.mat export-ignore",
            "/data/x export-ignore=1",
            "/data/x !export-ignore",
        ],
        ids=["quoted", "wildcard", "value", "unspecified"],
    )
    def test_returns_lines_it_cannot_read(self, line):
        assert artifact_licensing.export_ignored_paths(line) == (set(), [line])


class TestSourceArchiveErrors:
    def test_no_archive_is_an_error(self, tmp_path):
        result = _run_archives(tmp_path)
        assert (result.status, _rules(result)) == ("error", {"artifact-not-found"})

    def test_a_missing_archive_is_an_error(self, tmp_path):
        result = _run_archives(tmp_path, tmp_path / "absent.zip")
        assert (result.status, _rules(result)) == ("error", {"artifact-not-found"})

    def test_an_unknown_format_is_an_error(self, tmp_path):
        path = tmp_path / "source.7z"
        path.write_bytes(b"7z")
        result = _run_archives(tmp_path, path)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_a_truncated_archive_is_an_error(self, tmp_path, fmt):
        archive = _source_archive(tmp_path, fmt)
        archive.write_bytes(archive.read_bytes()[:40])
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_archive_without_one_root_folder_is_an_error(self, tmp_path, fmt):
        writer, name = ARCHIVE_WRITERS[fmt]
        archive = writer(tmp_path / "archives" / name, root="")
        result = _run_archives(tmp_path, archive)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_archive_with_two_root_folders_is_an_error(self, tmp_path):
        path = _source_zip(tmp_path / "archives" / "source.zip")
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("second-root/README.md", b"x")
        result = _run_archives(tmp_path, path)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_archive_with_no_file_is_an_error(self, tmp_path):
        path = tmp_path / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(zipfile.ZipInfo(f"{ARCHIVE_ROOT}/"), b"")
        result = _run_archives(tmp_path, path)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})

    def test_an_unusable_manifest_is_an_error(self, tmp_path):
        repo = _repo(tmp_path, manifest=None)
        result = _run_archives(tmp_path, _source_archive(tmp_path, "zip"), repo=repo)
        assert (result.status, _rules(result)) == ("error", {"manifest-unusable"})

    def test_one_unreadable_archive_makes_the_run_an_error(self, tmp_path):
        good = _source_archive(tmp_path, "zip")
        bad = _source_archive(tmp_path, "tar.gz")
        bad.write_bytes(b"not a tarball")
        result = _run_archives(tmp_path, good, bad)
        assert (result.status, _rules(result)) == ("error", {"artifact-unreadable"})


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


class TestRealGitArchive:
    """The rules against what git itself writes, not a hand-built archive."""

    @pytest.fixture
    def worktree(self, tmp_path):
        work = tmp_path / "work"
        for name, content in _archive_files(
            members={SAMPLE_CSV_PATH: SAMPLE_CSV, SAMPLE_MAT_PATH: SAMPLE_MAT}
        ).items():
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        try:
            _git(work, "init", "--quiet")
            _git(work, "add", "--all")
            _git(work, "commit", "--quiet", "-m", "fixture")
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.fail(f"this test needs git: {exc}")
        return work

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_export_ignore_keeps_the_samples_out_of_both_formats(
        self, tmp_path, worktree, fmt
    ):
        out = tmp_path / f"source.{fmt}"
        _git(
            worktree,
            "archive",
            f"--format={fmt}",
            f"--prefix={ARCHIVE_ROOT}/",
            "-o",
            str(out),
            "HEAD",
        )
        result = _run_archives(tmp_path, out)
        assert (result.status, result.findings) == ("pass", [])

    def test_without_the_attributes_the_samples_are_caught(self, tmp_path, worktree):
        _git(worktree, "rm", "--quiet", ".gitattributes")
        _git(worktree, "commit", "--quiet", "-m", "drop the attributes")
        out = tmp_path / "source.zip"
        _git(
            worktree,
            "archive",
            "--format=zip",
            f"--prefix={ARCHIVE_ROOT}/",
            "-o",
            str(out),
            "HEAD",
        )
        result = _run_archives(tmp_path, out)
        assert result.status == "fail"
        assert _rules(result) == {"third-party-data-path", "archive-attributes"}
        members = {
            f.evidence["member"]
            for f in result.findings
            if f.rule == "third-party-data-path"
        }
        assert members == {SAMPLE_CSV_PATH, SAMPLE_MAT_PATH}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_dispatch_writes_the_result_and_exits_by_status(self, tmp_path, capsys):
        out = tmp_path / "artifact-licensing.json"
        code = audit_main.main(
            [
                "artifact-licensing",
                "--dist",
                str(_clean_dist(tmp_path)),
                "--repo-root",
                str(_repo(tmp_path)),
                "--json-out",
                str(out),
            ]
        )
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert (doc["check"], doc["status"]) == ("artifact-licensing", "pass")
        assert "artifact-licensing pass" in capsys.readouterr().out

    def test_a_failing_artifact_exits_1_and_names_the_member(self, tmp_path, capsys):
        dist = _clean_dist(tmp_path, sdist={"members": {SAMPLE_CSV_PATH: SAMPLE_CSV}})
        code = audit_main.main(
            [
                "artifact-licensing",
                "--dist",
                str(dist),
                "--repo-root",
                str(_repo(tmp_path)),
            ]
        )
        assert code == 1
        assert SAMPLE_CSV_PATH in capsys.readouterr().out

    def test_an_empty_dist_exits_2(self, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        code = audit_main.main(
            [
                "artifact-licensing",
                "--dist",
                str(dist),
                "--repo-root",
                str(_repo(tmp_path)),
            ]
        )
        assert code == 2

    def test_the_archive_mode_checks_every_archive_given(self, tmp_path, capsys):
        out = tmp_path / "archives.json"
        code = audit_main.main(
            [
                "artifact-licensing",
                "--archive",
                str(_source_archive(tmp_path, "zip")),
                "--archive",
                str(_source_archive(tmp_path, "tar.gz")),
                "--repo-root",
                str(_repo(tmp_path)),
                "--json-out",
                str(out),
            ]
        )
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert doc["facts"]["archives"] == ["source.zip", "source.tar.gz"]
        assert "pass (local): source.zip, source.tar.gz" in capsys.readouterr().out

    def test_a_failing_archive_exits_1_and_names_it(self, tmp_path, capsys):
        archive = _source_archive(
            tmp_path, "tar.gz", members={SAMPLE_CSV_PATH: SAMPLE_CSV}
        )
        code = audit_main.main(
            [
                "artifact-licensing",
                "--archive",
                str(archive),
                "--repo-root",
                str(_repo(tmp_path)),
            ]
        )
        assert code == 1
        assert (
            f"archive source.tar.gz contains {SAMPLE_CSV_PATH}"
            in capsys.readouterr().out
        )

    @pytest.mark.parametrize("both", [True, False], ids=["both", "neither"])
    def test_exactly_one_of_dist_and_archive_is_accepted(self, tmp_path, both):
        args = ["artifact-licensing", "--repo-root", str(_repo(tmp_path))]
        if both:
            args += ["--dist", str(tmp_path), "--archive", str(tmp_path / "a.zip")]
        with pytest.raises(SystemExit) as exc:
            audit_main.main(args)
        assert exc.value.code == 2


class TestReportReproduction:
    """An audit issue built from a licensing finding reproduces it the way the
    finding was produced: the archive mode for an archive, a build otherwise."""

    @staticmethod
    def _reproduction(evidence: dict) -> str:
        group = [
            {
                "rule": "third-party-data-path",
                "evidence": evidence,
                "checks": ["artifact-licensing"],
            }
        ]
        return report._reproduction(group)

    def test_an_archive_finding_is_reproduced_with_git_archive(self):
        text = self._reproduction({"archive": "source.zip", "member": SAMPLE_CSV_PATH})
        assert (
            "git archive --format=zip --prefix=predictive-maintenance-mcp/ "
            "-o source.zip HEAD"
        ) in text
        assert "python -m tools.audit artifact-licensing --archive source.zip" in text
        assert "--dist" not in text

    def test_a_wheel_or_sdist_finding_is_reproduced_with_a_build(self):
        text = self._reproduction({"artifact": SDIST_NAME, "member": SAMPLE_CSV_PATH})
        assert "python -m build" in text
        assert "python -m tools.audit artifact-licensing --dist dist" in text
        assert "--archive" not in text


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------


def _tracked(paths) -> list:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", *paths],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(
            "the manifest check needs a Git checkout that holds the sample data; "
            f"git ls-files failed here: {exc}"
        )
    return sorted(name for name in proc.stdout.decode("utf-8").split("\0") if name)


@pytest.fixture(scope="module")
def manifest():
    """The committed third-party data manifest."""
    return json.loads(
        (REPO_ROOT / artifact_licensing.MANIFEST_RELPATH).read_text(encoding="utf-8")
    )


class TestRepository:
    """What the check assumes about this checkout's own files."""

    def test_the_manifest_lists_every_tracked_file_under_its_roots(self, manifest):
        tracked = _tracked(manifest["roots"])
        assert tracked, (
            "no tracked file under the manifest roots: this check needs a Git "
            "checkout that holds the sample data"
        )
        assert sorted(manifest["files"]) == tracked, REGENERATE

    @pytest.mark.research_data
    def test_the_manifest_hashes_match_the_files(self, manifest):
        require_recordings(*(REPO_ROOT / path for path in manifest["files"]))
        for path, entry in manifest["files"].items():
            content = (REPO_ROOT / path).read_bytes()
            assert (
                _entry(content) == entry
            ), f"manifest entry for {path} is stale; {REGENERATE}"

    @pytest.mark.research_data
    def test_the_regeneration_path_reproduces_the_committed_manifest(self, manifest):
        require_recordings(*(REPO_ROOT / path for path in manifest["files"]))
        fields = {k: v for k, v in manifest.items() if k not in ("roots", "files")}
        rebuilt = artifact_licensing.build_manifest(
            REPO_ROOT, manifest["roots"], **fields
        )
        assert rebuilt == manifest

    def test_every_tracked_data_file_is_covered_by_the_manifest_or_is_a_notice(
        self, manifest
    ):
        """A new dataset under data/ must join the manifest roots and data/LICENSE."""
        roots = tuple(f"{root}/" for root in manifest["roots"])
        uncovered = [
            path
            for path in _tracked(["data"])
            if not path.startswith(roots)
            and path not in artifact_licensing.ARCHIVE_DATA_FILES
        ]
        assert uncovered == []

    def test_the_export_ignore_paths_are_exactly_the_manifest_roots(self, manifest):
        """The source archives leave out every sample folder, and only those."""
        text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        paths, unreadable = artifact_licensing.export_ignored_paths(text)
        assert unreadable == []
        assert paths == set(manifest["roots"])

    def test_git_reads_the_export_ignore_rules_as_the_check_does(self, manifest):
        """What git archive applies, asked of git itself: each root is
        export-ignored, and the notices and the placeholder are not."""
        paths = [*manifest["roots"], *artifact_licensing.ARCHIVE_DATA_FILES]
        try:
            proc = subprocess.run(
                ["git", "check-attr", "export-ignore", "--", *paths],
                cwd=REPO_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.fail(f"this check needs a Git checkout: {exc}")
        # Each line reads "<path>: export-ignore: <set|unspecified|...>".
        state = {}
        for line in proc.stdout.splitlines():
            path, _attribute, value = line.rsplit(": ", 2)
            state[path] = value
        assert state == {
            **{root: "set" for root in manifest["roots"]},
            **{path: "unspecified" for path in artifact_licensing.ARCHIVE_DATA_FILES},
        }

    def test_the_placeholder_that_keeps_data_signals_in_an_archive_exists(self):
        assert (REPO_ROOT / "data" / "signals" / ".gitkeep").is_file()

    def test_the_repository_license_is_the_mit_text_with_its_copyright_line(self):
        text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
        assert artifact_licensing.mit_license_problem(text) is None
        assert text.splitlines()[2] == "Copyright (c) 2025 Luigi Gianpio Di Maggio"

    def test_the_data_notice_covers_exactly_the_manifest_roots(self, manifest):
        notice = (REPO_ROOT / manifest["notice"]).read_text(encoding="utf-8")
        # The scope lines are bare paths; prose that starts with a path is not.
        named = {
            line.strip().rstrip("/")
            for line in notice.splitlines()
            if line.strip().startswith("data/") and " " not in line.strip()
        }
        assert named == set(manifest["roots"])
        assert "https://creativecommons.org/licenses/by-nc-sa/4.0/" in notice
        assert "Eric Bechhoefer" in notice

    def test_no_second_license_file_sits_at_the_repository_root(self):
        patterns = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")
        found = sorted(
            path.name for pattern in patterns for path in REPO_ROOT.glob(pattern)
        )
        assert found == ["LICENSE"]


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

SITE = artifact_licensing.IMAGE_SITE_PACKAGES
SCIPY_DIST = f"{SITE}/scipy-1.17.1.dist-info"
MAT_FOLDER = f"{SITE}/scipy/io/matlab/tests/data"
PROJECT_DIST = f"{SITE}/{DIST_INFO}"
FIXTURE_MAT = b"MATLAB 5.0 MAT-file, synthetic test fixture" + bytes(range(64))
CLEAN_IMAGE = {
    "app/LICENSE": _mit().encode(),
    f"{PROJECT_DIST}/licenses/LICENSE": _mit().encode(),
    f"{PROJECT_DIST}/METADATA": b"Metadata-Version: 2.4\n",
    "app/resources/bearing_catalogs/README.md": b"# Bearing catalogs\n",
    "app/resources/bearing_catalogs/common_bearings_catalog.json": b"{}\n",
    "app/src/server.py": b"print('server')\n",
    f"{SITE}/nolicense-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\n",
}


def _record_line(path: str, content: bytes) -> str:
    """A RECORD row computed independently of the module under test."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode('ascii')},{len(content)}"


def _image_tar(path: Path, files: dict, dirs=(), links=()) -> Path:
    """Write a filesystem tar. ``links`` holds ``(kind, name, target)`` with
    kind ``sym`` or ``hard``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name in dirs:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        for kind, name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE if kind == "sym" else tarfile.LNKTYPE
            info.linkname = target
            archive.addfile(info)
    return path


def _scipy(files: dict, record=None, dist=SCIPY_DIST) -> dict:
    """``files`` (relative to site-packages) plus a scipy RECORD listing them,
    or the given ``record`` rows."""
    rows = (
        record
        if record is not None
        else [_record_line(name, content) for name, content in files.items()]
    )
    members = {f"{SITE}/{name}": content for name, content in files.items()}
    members[f"{dist}/RECORD"] = ("\n".join(rows) + "\n").encode()
    return members


def _run_image(
    tmp_path: Path, files=None, repo=None, links=(), dirs=()
) -> results.Result:
    image = _image_tar(
        tmp_path / "image.tar", {**CLEAN_IMAGE, **(files or {})}, dirs=dirs, links=links
    )
    config = artifact_licensing.ImageLicensingConfig(
        image=str(image), repo_root=repo or _repo(tmp_path)
    )
    return artifact_licensing.run_image_licensing(config)


def _mat_findings(result: results.Result) -> dict:
    return {
        finding.evidence["member"]: finding.evidence["problems"]
        for finding in result.findings
        if finding.rule == "image-mat-file"
    }


EXEMPT_NAME = "scipy/io/matlab/tests/data/fixture.mat"
REASONS = {
    "location": "it is not directly in an exempt test-data folder of the site-packages",
    "ownership": "no single installed distribution lists it in its RECORD",
    "integrity": "its SHA-256 and size match no RECORD entry",
    "size": f"it is larger than {artifact_licensing.MAT_EXEMPT_MAX_BYTES} bytes",
    "regular": "it is a link or special file, not a regular file",
}


class TestImageMode:
    def test_a_clean_image_passes_and_lists_license_gaps(self, tmp_path):
        result = _run_image(tmp_path, dirs=("app/data", "app/models", "app/reports"))
        assert result.status == "pass", result.findings
        assert result.facts["license_file_gaps"] == ["nolicense-1.0.dist-info"]
        assert result.facts["mat_exempt"]["count"] == 0

    @pytest.mark.parametrize(
        "member",
        [
            "app/models/x_model.pkl",
            "app/data/signals/x.csv",
            "app/data/cache/x.npy",
            "app/reports/r.html",
        ],
    )
    def test_a_file_in_a_folder_the_image_keeps_empty_fails(self, tmp_path, member):
        result = _run_image(tmp_path, {member: b"x"})
        assert result.status == "fail"
        assert ("image-forbidden-path", member) in {
            (f.rule, f.evidence.get("member")) for f in result.findings
        }

    @pytest.mark.parametrize(
        "member",
        [
            "app/resources/machine_manuals/operator.pdf",
            "app/resources/machine_manuals/README.md",
            "app/resources/machine_manuals/test_pump_manual.pdf",
            "app/resources/cache/test_pump_manual_extraction.json",
            "app/resources/cache/rag_index_tfidf.pkl",
        ],
    )
    def test_a_resource_outside_the_allowlist_fails(self, tmp_path, member):
        result = _run_image(tmp_path, {member: b"x"})
        assert ("image-resource-not-allowed", member) in {
            (f.rule, f.evidence.get("member")) for f in result.findings
        }

    def test_a_renamed_copy_of_a_sample_file_fails(self, tmp_path):
        result = _run_image(tmp_path, {"usr/share/x.bin": SAMPLE_CSV})
        finding = next(
            f for f in result.findings if f.rule == "third-party-data-content"
        )
        assert finding.evidence["matches"] == SAMPLE_CSV_PATH

    def test_a_missing_image_license_fails(self, tmp_path):
        files = {k: v for k, v in CLEAN_IMAGE.items() if k != "app/LICENSE"}
        image = _image_tar(tmp_path / "image.tar", files)
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(str(image), _repo(tmp_path))
        )
        assert "image-license-missing" in _rules(result)

    def test_a_missing_project_license_fails(self, tmp_path):
        files = {
            k: v for k, v in CLEAN_IMAGE.items() if not k.endswith("licenses/LICENSE")
        }
        image = _image_tar(tmp_path / "image.tar", files)
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(str(image), _repo(tmp_path))
        )
        assert "image-license-missing" in _rules(result)

    def test_a_non_mit_image_license_fails(self, tmp_path):
        result = _run_image(tmp_path, {"app/LICENSE": b"Apache License 2.0\n"})
        assert "license-text" in _rules(result)

    @pytest.mark.parametrize(
        ("rule", "files"),
        [
            ("image-forbidden-path", {"app/models/x_model.pkl": b"x"}),
            ("image-resource-not-allowed", {"app/resources/x.pdf": b"x"}),
            ("third-party-data-content", {"usr/share/x.bin": SAMPLE_CSV}),
            ("license-text", {"app/LICENSE": b"Apache License 2.0\n"}),
            ("image-mat-file", {"app/x.mat": b"x"}),
        ],
    )
    def test_each_rule_alone_catches_its_fixture(
        self, tmp_path, monkeypatch, rule, files
    ):
        result = _run_image(tmp_path / "with", files)
        assert rule in _rules(result)
        owner = {
            "image-forbidden-path": artifact_licensing._image_folder_findings,
            "image-resource-not-allowed": artifact_licensing._image_resource_findings,
            "third-party-data-content": artifact_licensing._image_content_findings,
            "license-text": artifact_licensing._image_license_findings,
            "image-mat-file": artifact_licensing._image_mat_findings,
        }[rule]
        monkeypatch.setattr(
            artifact_licensing,
            "IMAGE_RULES",
            tuple(r for r in artifact_licensing.IMAGE_RULES if r is not owner),
        )
        assert rule not in _rules(_run_image(tmp_path / "without", files))


class TestImageModeErrors:
    def test_an_empty_tar_is_an_error(self, tmp_path):
        image = _image_tar(tmp_path / "image.tar", {})
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(str(image), _repo(tmp_path))
        )
        assert result.status == "error"
        assert "artifact-unreadable" in _rules(result)

    def test_a_truncated_stream_is_an_error(self, tmp_path):
        image = _image_tar(tmp_path / "image.tar", CLEAN_IMAGE)
        truncated = tmp_path / "truncated.tar"
        truncated.write_bytes(image.read_bytes()[:700])
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(str(truncated), _repo(tmp_path))
        )
        assert result.status == "error"

    @pytest.mark.parametrize(
        "cut",
        ["at-last-header", "inside-last-header", "no-end-marker", "trailing-junk"],
    )
    def test_an_export_that_does_not_end_cleanly_is_an_error(self, tmp_path, cut):
        # The forbidden member comes last, after both license files, so a cut
        # before it would otherwise read as a complete, passing image.
        image = _image_tar(
            tmp_path / "image.tar", {**CLEAN_IMAGE, "app/models/x_model.pkl": b"x"}
        )
        data = image.read_bytes()
        with tarfile.open(image) as archive:
            last = archive.getmembers()[-1].offset
            end = archive.offset
        damaged = {
            "at-last-header": data[:last],
            "inside-last-header": data[: last + 100],
            "no-end-marker": data[:end],
            "trailing-junk": data + b"\x01" * 512,
        }[cut]
        (tmp_path / "damaged.tar").write_bytes(damaged)
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(
                str(tmp_path / "damaged.tar"), _repo(tmp_path)
            )
        )
        assert result.status == "error"
        assert "artifact-unreadable" in _rules(result)

    def test_a_missing_file_is_an_error(self, tmp_path):
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(
                str(tmp_path / "absent.tar"), _repo(tmp_path)
            )
        )
        assert result.status == "error"
        assert "artifact-not-found" in _rules(result)

    def test_a_missing_manifest_is_an_error(self, tmp_path):
        image = _image_tar(tmp_path / "image.tar", CLEAN_IMAGE)
        result = artifact_licensing.run_image_licensing(
            artifact_licensing.ImageLicensingConfig(
                str(image), _repo(tmp_path, manifest=None)
            )
        )
        assert result.status == "error"
        assert "manifest-unusable" in _rules(result)


class TestImageMatExemption:
    def test_a_recorded_scipy_test_file_is_exempt_and_listed(self, tmp_path):
        result = _run_image(tmp_path, _scipy({EXEMPT_NAME: FIXTURE_MAT}))
        assert result.status == "pass", result.findings
        exempt = result.facts["mat_exempt"]
        assert exempt["count"] == 1
        assert exempt["bytes"] == len(FIXTURE_MAT)
        assert exempt["distributions"] == ["scipy-1.17.1.dist-info"]
        assert exempt["files"] == [
            {
                "path": f"{SITE}/{EXEMPT_NAME}",
                "size": len(FIXTURE_MAT),
                "sha256": hashlib.sha256(FIXTURE_MAT).hexdigest(),
            }
        ]

    def test_a_mat_file_in_app_data_fails_both_rules(self, tmp_path):
        result = _run_image(tmp_path, {"app/data/signals/x.mat": FIXTURE_MAT})
        assert "app/data/signals/x.mat" in _mat_findings(result)
        assert "image-forbidden-path" in _rules(result)

    @pytest.mark.parametrize(
        "member",
        [
            "app/x.mat",
            "tmp/x.mat",
            "root/x.mat",
            "home/u/x.mat",
            "opt/x.mat",
            "usr/share/x.mat",
            f"{SITE}/predictive_maintenance_mcp/x.mat",
            f"{SITE}/torch/x.mat",
            f"{SITE}/scipy/io/x.mat",
            f"{MAT_FOLDER}/sub/x.mat",
            "app/X.MAT",
            "tmp/x.Mat",
        ],
    )
    def test_a_mat_file_outside_the_exempt_folder_fails(self, tmp_path, member):
        problems = _mat_findings(_run_image(tmp_path, {member: FIXTURE_MAT}))
        assert REASONS["location"] in problems[member]

    def test_a_file_the_record_does_not_list_fails(self, tmp_path):
        files = _scipy({}, record=[])
        files[f"{MAT_FOLDER}/fixture.mat"] = FIXTURE_MAT
        problems = _mat_findings(_run_image(tmp_path, files))[
            f"{MAT_FOLDER}/fixture.mat"
        ]
        assert problems == [REASONS["ownership"], REASONS["integrity"]]

    @pytest.mark.parametrize(
        "record_content",
        [FIXTURE_MAT[:-1] + b"\x00", FIXTURE_MAT + b"\x00"],
        ids=["hash-differs", "size-differs"],
    )
    def test_a_file_whose_hash_or_size_differs_from_its_record_fails(
        self, tmp_path, record_content
    ):
        files = _scipy(
            {EXEMPT_NAME: FIXTURE_MAT},
            record=[_record_line(EXEMPT_NAME, record_content)],
        )
        problems = _mat_findings(_run_image(tmp_path, files))[f"{SITE}/{EXEMPT_NAME}"]
        assert problems == [REASONS["integrity"]]

    def test_a_recorded_file_over_the_size_limit_fails(self, tmp_path):
        big = b"M" * (artifact_licensing.MAT_EXEMPT_MAX_BYTES + 1)
        problems = _mat_findings(_run_image(tmp_path, _scipy({EXEMPT_NAME: big})))
        assert problems[f"{SITE}/{EXEMPT_NAME}"] == [REASONS["size"]]

    def test_a_copy_of_a_sample_file_in_the_exempt_folder_still_fails(self, tmp_path):
        # The exemption never suppresses the copy rule: SAMPLE_CSV is in the
        # test manifest, small and recorded, and lies in the exempt folder.
        result = _run_image(tmp_path, _scipy({EXEMPT_NAME: SAMPLE_CSV}))
        assert _mat_findings(result) == {}
        finding = next(
            f for f in result.findings if f.rule == "third-party-data-content"
        )
        assert finding.evidence["member"] == f"{SITE}/{EXEMPT_NAME}"

    @pytest.mark.parametrize("kind", ["sym", "hard"])
    def test_a_link_named_mat_in_the_exempt_folder_fails(self, tmp_path, kind):
        files = _scipy({}, record=[_record_line(EXEMPT_NAME, b"")])
        files["app/src/server.py"] = b"print('server')\n"
        result = _run_image(
            tmp_path,
            files,
            links=[(kind, f"{SITE}/{EXEMPT_NAME}", "app/src/server.py")],
        )
        assert _mat_findings(result)[f"{SITE}/{EXEMPT_NAME}"] == [REASONS["regular"]]

    def test_two_scipy_distributions_make_the_owner_ambiguous(self, tmp_path):
        files = {
            **_scipy({EXEMPT_NAME: FIXTURE_MAT}),
            **_scipy({EXEMPT_NAME: FIXTURE_MAT}, dist=f"{SITE}/scipy-1.16.0.dist-info"),
        }
        problems = _mat_findings(_run_image(tmp_path, files))[f"{SITE}/{EXEMPT_NAME}"]
        assert problems == [REASONS["ownership"]]

    def test_a_missing_record_trusts_nothing(self, tmp_path):
        files = {
            f"{SITE}/{EXEMPT_NAME}": FIXTURE_MAT,
            f"{SCIPY_DIST}/METADATA": b"Metadata-Version: 2.1\n",
        }
        problems = _mat_findings(_run_image(tmp_path, files))[f"{SITE}/{EXEMPT_NAME}"]
        assert problems == [REASONS["ownership"], REASONS["integrity"]]

    def test_a_record_that_is_a_link_trusts_nothing(self, tmp_path):
        files = {
            f"{SITE}/{EXEMPT_NAME}": FIXTURE_MAT,
            f"{SCIPY_DIST}/METADATA": b"Metadata-Version: 2.1\n",
            "tmp/record.txt": (_record_line(EXEMPT_NAME, FIXTURE_MAT) + "\n").encode(),
        }
        result = _run_image(
            tmp_path, files, links=[("sym", f"{SCIPY_DIST}/RECORD", "tmp/record.txt")]
        )
        problems = _mat_findings(result)[f"{SITE}/{EXEMPT_NAME}"]
        assert problems == [REASONS["ownership"], REASONS["integrity"]]

    def test_another_interpreter_folder_is_not_exempt(self, tmp_path):
        other = "usr/local/lib/python3.12/site-packages"
        files = {
            f"{other}/{EXEMPT_NAME}": FIXTURE_MAT,
            f"{other}/scipy-1.17.1.dist-info/RECORD": (
                _record_line(EXEMPT_NAME, FIXTURE_MAT) + "\n"
            ).encode(),
        }
        problems = _mat_findings(_run_image(tmp_path, files))[f"{other}/{EXEMPT_NAME}"]
        assert REASONS["location"] in problems


#: One fixture per exemption condition, each failing that condition alone.
_SINGLE_CONDITION_FIXTURES = {
    "_mat_location": lambda: (
        _scipy({"scipy/io/fixture.mat": FIXTURE_MAT}),
        (),
        f"{SITE}/scipy/io/fixture.mat",
    ),
    "_mat_ownership": lambda: (
        {
            **_scipy({EXEMPT_NAME: FIXTURE_MAT}),
            **_scipy({EXEMPT_NAME: FIXTURE_MAT}, dist=f"{SITE}/scipy-1.16.0.dist-info"),
        },
        (),
        f"{SITE}/{EXEMPT_NAME}",
    ),
    "_mat_integrity": lambda: (
        _scipy(
            {EXEMPT_NAME: FIXTURE_MAT},
            record=[_record_line(EXEMPT_NAME, FIXTURE_MAT + b"\x00")],
        ),
        (),
        f"{SITE}/{EXEMPT_NAME}",
    ),
    "_mat_size": lambda: (
        _scipy({EXEMPT_NAME: b"M" * (artifact_licensing.MAT_EXEMPT_MAX_BYTES + 1)}),
        (),
        f"{SITE}/{EXEMPT_NAME}",
    ),
    "_mat_regular": lambda: (
        {**_scipy({}, record=[_record_line(EXEMPT_NAME, b"")])},
        [("sym", f"{SITE}/{EXEMPT_NAME}", "app/src/server.py")],
        f"{SITE}/{EXEMPT_NAME}",
    ),
}


class TestImageMatExemptionRemovalProofs:
    """Each exemption condition is needed: its fixture fails with every
    condition in place and passes once that condition alone is dropped."""

    @pytest.mark.parametrize("condition", sorted(_SINGLE_CONDITION_FIXTURES))
    def test_dropping_one_condition_lets_its_fixture_through(
        self, tmp_path, monkeypatch, condition
    ):
        files, links, member = _SINGLE_CONDITION_FIXTURES[condition]()
        with_all = _mat_findings(_run_image(tmp_path / "all", files, links=links))
        assert len(with_all[member]) == 1
        target = getattr(artifact_licensing, condition)
        monkeypatch.setattr(
            artifact_licensing,
            "MAT_EXEMPTION_CONDITIONS",
            tuple(
                c
                for c in artifact_licensing.MAT_EXEMPTION_CONDITIONS
                if c is not target
            ),
        )
        without = _mat_findings(_run_image(tmp_path / "without", files, links=links))
        assert member not in without

    def test_an_empty_exemption_table_exempts_nothing(self, tmp_path, monkeypatch):
        files = _scipy({EXEMPT_NAME: FIXTURE_MAT})
        assert _mat_findings(_run_image(tmp_path / "table", files)) == {}
        monkeypatch.setattr(artifact_licensing, "MAT_EXEMPTIONS", ())
        problems = _mat_findings(_run_image(tmp_path / "empty", files))
        assert REASONS["location"] in problems[f"{SITE}/{EXEMPT_NAME}"]


class TestImageCli:
    def test_the_image_mode_writes_the_result_and_exits_0(self, tmp_path, capsys):
        image = _image_tar(tmp_path / "image.tar", CLEAN_IMAGE)
        out = tmp_path / "image.json"
        code = audit_main.main(
            [
                "artifact-licensing",
                "--image",
                str(image),
                "--repo-root",
                str(_repo(tmp_path)),
                "--json-out",
                str(out),
            ]
        )
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert doc["facts"]["image"] == "image.tar"
        assert (
            "artifact-licensing pass (local): image image.tar"
            in capsys.readouterr().out
        )

    def test_a_failing_image_exits_1_and_names_the_member(self, tmp_path, capsys):
        image = _image_tar(
            tmp_path / "image.tar", {**CLEAN_IMAGE, "app/models/x_model.pkl": b"x"}
        )
        code = audit_main.main(
            [
                "artifact-licensing",
                "--image",
                str(image),
                "--repo-root",
                str(_repo(tmp_path)),
            ]
        )
        assert code == 1
        assert "app/models/x_model.pkl" in capsys.readouterr().out

    def test_the_image_is_read_from_standard_input(self, tmp_path, monkeypatch, capsys):
        image = _image_tar(tmp_path / "image.tar", CLEAN_IMAGE)

        class _Stdin:
            buffer = io.BytesIO(image.read_bytes())

        monkeypatch.setattr(artifact_licensing.sys, "stdin", _Stdin())
        code = audit_main.main(
            ["artifact-licensing", "--image", "-", "--repo-root", str(_repo(tmp_path))]
        )
        assert code == 0
        assert "image stdin" in capsys.readouterr().out

    def test_image_and_dist_together_are_refused(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            audit_main.main(
                [
                    "artifact-licensing",
                    "--image",
                    "-",
                    "--dist",
                    str(tmp_path),
                    "--repo-root",
                    str(_repo(tmp_path)),
                ]
            )
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Model weight files in an artifact fail (rule 7)
# ---------------------------------------------------------------------------

#: One planted weight file per refused suffix, in folders a build could pick up,
#: plus a case variant: the rule reads the member's own name, case-folded.
WEIGHT_MEMBERS = (
    (f"{NAME}/models/model.safetensors", ".safetensors"),
    ("models/chronos-2/pytorch_model.bin", ".bin"),
    (f"{NAME}/x.pt", ".pt"),
    ("a/b/y.pth", ".pth"),
    (f"{NAME}/W.SafeTensors", ".safetensors"),
)


class TestModelWeightFiles:
    @pytest.mark.parametrize("member, suffix", WEIGHT_MEMBERS)
    @pytest.mark.parametrize("kind", ["wheel", "sdist"])
    def test_a_weight_file_in_a_built_artifact_fails(
        self, tmp_path, kind, member, suffix
    ):
        dist = _clean_dist(tmp_path, **{kind: {"members": {member: b"\x00weights"}}})
        result = _run(tmp_path, dist=dist)
        assert result.status == "fail"
        (finding,) = result.findings
        assert finding.rule == "model-weight-file"
        assert finding.title == (
            f"The {kind} contains {member}, which is a model weight file ({suffix})"
        )
        assert finding.evidence["member"] == member
        name = WHEEL_NAME if kind == "wheel" else SDIST_NAME
        assert finding.evidence["artifact"] == name

    @pytest.mark.parametrize("member, suffix", WEIGHT_MEMBERS)
    def test_a_weight_file_in_a_source_archive_fails(
        self, tmp_path, fmt, member, suffix
    ):
        archive = _source_archive(tmp_path, fmt, members={member: b"\x00weights"})
        result = _run_archives(tmp_path, archive)
        assert result.status == "fail"
        (finding,) = result.findings
        assert finding.rule == "model-weight-file"
        assert finding.title == (
            f"The archive {archive.name} contains {member}, which is a model weight "
            f"file ({suffix})"
        )
        assert finding.evidence == {"member": member, "archive": archive.name}

    @pytest.mark.parametrize(
        "member", ["notes.bin.txt", "binary/readme.md", "src/pt.py", "a.pthx"]
    )
    def test_a_name_that_only_contains_a_suffix_passes(self, tmp_path, member):
        dist = _clean_dist(tmp_path, sdist={"members": {member: b"x"}})
        assert _run(tmp_path, dist=dist).status == "pass"

    def test_the_suffixes_are_the_four_model_weight_formats(self):
        assert artifact_licensing.WEIGHT_SUFFIXES == (
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
        )
