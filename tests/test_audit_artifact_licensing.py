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
* the committed manifest still describes the tracked sample files, and the
  repository ``LICENSE`` and ``data/LICENSE`` say what the check assumes.
"""

import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.audit import __main__ as audit_main
from tools.audit import artifact_licensing, results

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
        dist = _clean_dist(
            tmp_path,
            wheel={"members": {f"{NAME}/resources/blob.bin": SAMPLE_MAT}},
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

    def test_the_manifest_hashes_match_the_files(self, manifest):
        for path, entry in manifest["files"].items():
            content = (REPO_ROOT / path).read_bytes()
            assert (
                _entry(content) == entry
            ), f"manifest entry for {path} is stale; {REGENERATE}"

    def test_the_regeneration_path_reproduces_the_committed_manifest(self, manifest):
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
            and path not in ("data/LICENSE", "data/README.md")
        ]
        assert uncovered == []

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
