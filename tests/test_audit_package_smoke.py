"""Guards for the installed-package smoke (``python -m tools.audit package-smoke``).

The stdio harness proves the server a desktop host launches from a checkout.
A user never runs a checkout: they get the wheel, installed into an
environment that has no repository source on its import path. Missing package
data, an import that only resolves inside a clone, or a console script that
was never declared all pass the in-tree suite and break that user. The
package smoke installs the built artifact into a fresh venv and runs the
harness against the installed console script.

The real run needs a network and minutes of installing, so it is proven by the
local end-to-end run and the CI job. What this module pins is everything that
decides the verdict without a subprocess:

* artifact discovery: exactly one artifact of the requested kind, or an error
  that says where it looked and what it found;
* console-script and interpreter paths for Windows-shaped and POSIX venvs;
* the argument lists of every step, including that the harness runs with the
  venv's interpreter in installed mode against the venv as install prefix;
* status propagation through an injected runner: a harness ``error`` stays an
  ``error`` (exit 2), and every way of producing no evidence (a failed install,
  a missing console script, a harness that wrote nothing, a harness whose exit
  code contradicts its JSON) ends as ``error``, never ``pass``.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.audit import __main__ as audit_main
from tools.audit import package_smoke, results

WHEEL_NAME = "predictive_maintenance_mcp-0.13.0-py3-none-any.whl"
SDIST_NAME = "predictive_maintenance_mcp-0.13.0.tar.gz"
PIP_LIST = [
    {"name": "mcp", "version": "2.3.0"},
    {"name": "numpy", "version": "2.3.3"},
    {"name": "predictive-maintenance-mcp", "version": "0.13.0"},
]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


class TestFindArtifact:
    def test_the_single_wheel_is_returned(self, tmp_path):
        wheel = _touch(tmp_path / WHEEL_NAME)
        _touch(tmp_path / SDIST_NAME)  # the other kind never counts
        assert package_smoke.find_artifact(tmp_path, "wheel") == wheel

    def test_the_single_sdist_is_returned(self, tmp_path):
        _touch(tmp_path / WHEEL_NAME)
        sdist = _touch(tmp_path / SDIST_NAME)
        assert package_smoke.find_artifact(tmp_path, "sdist") == sdist

    def test_zero_matches_is_an_error_naming_the_directory(self, tmp_path):
        _touch(tmp_path / SDIST_NAME)
        with pytest.raises(package_smoke.ArtifactError) as excinfo:
            package_smoke.find_artifact(tmp_path, "wheel")
        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert "*.whl" in message

    def test_two_wheels_is_an_error_listing_both(self, tmp_path):
        first = _touch(tmp_path / WHEEL_NAME)
        second = _touch(tmp_path / "predictive_maintenance_mcp-0.12.0-py3-none-any.whl")
        with pytest.raises(package_smoke.ArtifactError) as excinfo:
            package_smoke.find_artifact(tmp_path, "wheel")
        message = str(excinfo.value)
        assert first.name in message
        assert second.name in message

    def test_a_missing_directory_is_an_error_naming_it(self, tmp_path):
        missing = tmp_path / "no-dist-here"
        with pytest.raises(package_smoke.ArtifactError) as excinfo:
            package_smoke.find_artifact(missing, "wheel")
        assert str(missing) in str(excinfo.value)

    def test_an_unknown_kind_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            package_smoke.find_artifact(tmp_path, "egg")


# ---------------------------------------------------------------------------
# Paths inside the venv
# ---------------------------------------------------------------------------


class TestVenvPaths:
    def test_console_script_on_a_windows_shaped_venv(self, tmp_path):
        assert package_smoke.console_script(tmp_path, windows=True) == (
            tmp_path / "Scripts" / "predictive-maintenance-mcp.exe"
        )

    def test_console_script_on_a_posix_venv(self, tmp_path):
        assert package_smoke.console_script(tmp_path, windows=False) == (
            tmp_path / "bin" / "predictive-maintenance-mcp"
        )

    def test_interpreter_on_a_windows_shaped_venv(self, tmp_path):
        assert package_smoke.venv_python(tmp_path, windows=True) == (
            tmp_path / "Scripts" / "python.exe"
        )

    def test_interpreter_on_a_posix_venv(self, tmp_path):
        assert package_smoke.venv_python(tmp_path, windows=False) == (
            tmp_path / "bin" / "python"
        )


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


class TestCommands:
    PY = Path("/work/venv/bin/python")

    def test_venv_creation_uses_the_given_interpreter(self):
        venv = Path("/work/venv")
        assert package_smoke.venv_command("/usr/bin/python3.11", venv) == [
            "/usr/bin/python3.11",
            "-m",
            "venv",
            str(venv),
        ]

    def test_pip_upgrade(self):
        assert package_smoke.pip_upgrade_command(self.PY) == [
            str(self.PY),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "pip",
        ]

    def test_install_names_the_artifact_file(self):
        wheel = Path("/dist") / WHEEL_NAME
        assert package_smoke.install_command(self.PY, wheel) == [
            str(self.PY),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel),
        ]

    def test_pip_list_is_json(self):
        assert package_smoke.pip_list_command(self.PY) == [
            str(self.PY),
            "-m",
            "pip",
            "list",
            "--disable-pip-version-check",
            "--format",
            "json",
        ]

    def test_harness_runs_installed_mode_with_the_venv_interpreter(self):
        venv = Path("/work/venv")
        script = venv / "bin" / "predictive-maintenance-mcp"
        out = Path("/work/harness.json")
        assert package_smoke.harness_command(
            self.PY, script, venv, out, "pr-ubuntu-py3.11"
        ) == [
            str(self.PY),
            "-m",
            "tools.audit",
            "smoke",
            "--server-cmd",
            str(script),
            "--installed",
            "--expect-installed-under",
            str(venv),
            "--python",
            str(self.PY),
            "--json-out",
            str(out),
            "--variant",
            "pr-ubuntu-py3.11",
        ]

    def test_harness_forwards_timeouts_only_when_given(self):
        venv = Path("/work/venv")
        command = package_smoke.harness_command(
            self.PY,
            venv / "bin" / "predictive-maintenance-mcp",
            venv,
            Path("/work/harness.json"),
            "v",
            timeout_startup=300.0,
            timeout_request=90.0,
        )
        assert command[-4:] == [
            "--timeout-startup",
            "300.0",
            "--timeout-request",
            "90.0",
        ]


# ---------------------------------------------------------------------------
# pip list parsing
# ---------------------------------------------------------------------------


class TestInstalledVersions:
    def test_only_names_and_versions_are_kept(self):
        text = json.dumps(
            PIP_LIST
            + [
                {
                    "name": "local-thing",
                    "version": "1.0",
                    "editable_project_location": "/x",
                }
            ]
        )
        assert package_smoke.installed_versions(text) == {
            "mcp": "2.3.0",
            "numpy": "2.3.3",
            "predictive-maintenance-mcp": "0.13.0",
            "local-thing": "1.0",
        }

    @pytest.mark.parametrize(
        "text",
        ["", "not json", "{}", "[]", '[{"name": "mcp"}]'],
        ids=["empty", "garbage", "object", "no-packages", "no-version"],
    )
    def test_output_without_versions_is_refused(self, text):
        """An empty listing must not read as 'nothing installed, all fine'."""
        with pytest.raises(ValueError):
            package_smoke.installed_versions(text)


# ---------------------------------------------------------------------------
# The whole run, through an injected runner (no real subprocess)
# ---------------------------------------------------------------------------


def _harness_doc(status="pass", findings=None):
    finding = results.Finding(
        rule="catalog-bearing",
        category="install",
        severity="info",
        subject="bearing catalog in the installed package",
        title="Bearing 6205 is found in the installed catalog",
        classification="harmless",
    )
    doc = results.Result(
        check="smoke",
        variant="pr-ubuntu-py3.11",
        status=status,
        started_at="2026-10-15T10:00:00Z",
        finished_at="2026-10-15T10:01:00Z",
        environment={
            "python": "3.11.9",
            "packages": {"predictive-maintenance-mcp": "0.0.0-from-checkout"},
        },
        facts={"tool_count": {"legacy": 38, "auto": 38}, "installed_mode": True},
        findings=[finding] if findings is None else findings,
    )
    return doc.to_dict()


class FakeRunner:
    """Stands in for ``subprocess.run``; records every step it is asked for."""

    def __init__(
        self,
        windows,
        harness_doc=None,
        harness_returncode=None,
        fail_step=None,
        raise_step=None,
        make_script=True,
    ):
        self.windows = windows
        self.harness_doc = harness_doc
        self.harness_returncode = harness_returncode
        self.fail_step = fail_step
        self.raise_step = raise_step
        self.make_script = make_script
        self.calls = []

    @staticmethod
    def step_of(command):
        if command[1:3] == ["-m", "venv"]:
            return "venv"
        if command[1:3] == ["-m", "tools.audit"]:
            return "harness"
        if "list" in command:
            return "pip-list"
        if "--upgrade" in command:
            return "pip-upgrade"
        return "install"

    def __call__(self, command, cwd, timeout):
        command = [str(part) for part in command]
        step = self.step_of(command)
        self.calls.append((step, command, Path(cwd)))
        if step == self.raise_step:
            raise FileNotFoundError(command[0])
        if step == self.fail_step:
            return subprocess.CompletedProcess(command, 1, "", f"{step} exploded")
        if step == "venv":
            venv = Path(command[3])
            _touch(package_smoke.venv_python(venv, self.windows))
            if self.make_script:
                _touch(package_smoke.console_script(venv, self.windows))
        if step == "pip-list":
            return subprocess.CompletedProcess(command, 0, json.dumps(PIP_LIST), "")
        if step == "harness":
            if self.harness_doc is not None:
                out = Path(command[command.index("--json-out") + 1])
                results.write_json(self.harness_doc, out)
            code = self.harness_returncode
            if code is None:
                status = (self.harness_doc or {}).get("status", "error")
                code = results.exit_code_for(status)
            return subprocess.CompletedProcess(command, code, "smoke summary\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    def steps(self):
        return [step for step, _, _ in self.calls]


WINDOWS = os.name == "nt"


def _run(tmp_path, runner, kind="wheel", artifact=True):
    dist = tmp_path / "dist"
    dist.mkdir()
    if artifact:
        _touch(dist / (WHEEL_NAME if kind == "wheel" else SDIST_NAME))
    config = package_smoke.PackageSmokeConfig(
        dist=dist,
        artifact=kind,
        variant="pr-test",
        work_dir=tmp_path / "work",
        windows=WINDOWS,
    )
    return package_smoke.run_package_smoke(config, runner=runner)


def _rules(result):
    return {finding.rule for finding in result.findings}


class TestRunPackageSmoke:
    def test_a_passing_harness_passes_and_embeds_its_evidence(self, tmp_path):
        runner = FakeRunner(WINDOWS, harness_doc=_harness_doc("pass"))
        result = _run(tmp_path, runner)
        assert result.check == "package-smoke"
        assert result.status == "pass"
        assert result.exit_code == 0
        assert runner.steps() == [
            "venv",
            "pip-upgrade",
            "install",
            "pip-list",
            "harness",
        ]
        assert result.facts["harness"]["status"] == "pass"
        assert result.facts["harness"]["facts"]["tool_count"] == {
            "legacy": 38,
            "auto": 38,
        }
        assert result.facts["installed_versions"]["mcp"] == "2.3.0"
        assert result.facts["artifact"] == WHEEL_NAME
        assert "catalog-bearing" in _rules(result)
        # The installed version wins over whatever the harness process saw.
        packages = result.environment["packages"]
        assert packages["predictive-maintenance-mcp"] == "0.13.0"
        results.validate_result(json.loads(results.serialize(result)))

    def test_a_harness_error_propagates_as_error_exit_2(self, tmp_path):
        failing = results.Finding(
            rule="startup-failed",
            category="install",
            severity="critical",
            subject="server startup",
            title="The server did not start",
        )
        doc = _harness_doc("error", findings=[failing])
        result = _run(tmp_path, FakeRunner(WINDOWS, harness_doc=doc))
        assert result.status == "error"
        assert result.exit_code == 2
        assert "startup-failed" in _rules(result)
        assert result.facts["harness"]["status"] == "error"

    def test_a_harness_fail_propagates_as_fail(self, tmp_path):
        result = _run(tmp_path, FakeRunner(WINDOWS, harness_doc=_harness_doc("fail")))
        assert (result.status, result.exit_code) == ("fail", 1)

    def test_the_harness_runs_from_the_repo_root_and_nothing_else_does(self, tmp_path):
        runner = FakeRunner(WINDOWS, harness_doc=_harness_doc("pass"))
        _run(tmp_path, runner)
        for step, command, cwd in runner.calls:
            if step == "harness":
                assert cwd == package_smoke.REPO_ROOT
                assert command[0] == str(
                    package_smoke.venv_python(tmp_path / "work" / "venv", WINDOWS)
                )
            else:
                assert cwd != package_smoke.REPO_ROOT

    def test_a_relative_dist_still_names_the_artifact_from_the_install_cwd(
        self, tmp_path, monkeypatch
    ):
        # CI passes `--dist dist` from the checkout, while pip runs in the work
        # directory: a relative artifact path would point inside the work dir.
        _touch(tmp_path / "dist" / WHEEL_NAME)
        monkeypatch.chdir(tmp_path)
        runner = FakeRunner(WINDOWS, harness_doc=_harness_doc("pass"))
        config = package_smoke.PackageSmokeConfig(
            dist=Path("dist"),
            variant="pr-test",
            work_dir=tmp_path / "work",
            windows=WINDOWS,
        )
        result = package_smoke.run_package_smoke(config, runner=runner)
        (install,) = [
            (cmd, cwd) for step, cmd, cwd in runner.calls if step == "install"
        ]
        command, cwd = install
        assert cwd != tmp_path
        assert (cwd / command[-1]).is_file()
        assert result.status == "pass"

    def test_a_harness_that_wrote_nothing_is_an_error(self, tmp_path):
        result = _run(
            tmp_path, FakeRunner(WINDOWS, harness_doc=None, harness_returncode=0)
        )
        assert result.status == "error"
        assert "harness-no-result" in _rules(result)

    def test_an_exit_code_contradicting_the_json_is_an_error(self, tmp_path):
        runner = FakeRunner(
            WINDOWS, harness_doc=_harness_doc("pass"), harness_returncode=1
        )
        result = _run(tmp_path, runner)
        assert result.status == "error"
        assert "harness-no-result" in _rules(result)

    def test_a_failed_install_is_an_error_and_the_harness_never_runs(self, tmp_path):
        runner = FakeRunner(
            WINDOWS, harness_doc=_harness_doc("pass"), fail_step="install"
        )
        result = _run(tmp_path, runner)
        assert result.status == "error"
        assert "harness" not in runner.steps()
        (finding,) = [item for item in result.findings if item.rule == "install-failed"]
        assert "install exploded" in finding.evidence["excerpt"]

    def test_a_venv_that_cannot_be_created_is_an_error(self, tmp_path):
        runner = FakeRunner(WINDOWS, raise_step="venv")
        result = _run(tmp_path, runner)
        assert result.status == "error"
        assert "venv-create-failed" in _rules(result)
        assert runner.steps() == ["venv"]

    def test_a_missing_console_script_is_an_error(self, tmp_path):
        runner = FakeRunner(
            WINDOWS, harness_doc=_harness_doc("pass"), make_script=False
        )
        result = _run(tmp_path, runner)
        assert result.status == "error"
        assert "console-script-missing" in _rules(result)
        assert "harness" not in runner.steps()

    def test_a_missing_artifact_is_an_error_naming_the_directory(self, tmp_path):
        runner = FakeRunner(WINDOWS)
        result = _run(tmp_path, runner, artifact=False)
        assert result.status == "error"
        assert runner.calls == []
        (finding,) = result.findings
        assert finding.rule == "artifact-not-found"
        assert str(tmp_path / "dist") in finding.title


class TestCli:
    def test_dispatch_writes_an_error_result_for_an_empty_dist(self, tmp_path, capsys):
        dist = tmp_path / "dist"
        dist.mkdir()
        out = tmp_path / "package-smoke.json"
        code = audit_main.main(
            [
                "package-smoke",
                "--dist",
                str(dist),
                "--artifact",
                "wheel",
                "--json-out",
                str(out),
            ]
        )
        assert code == 2
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert (doc["check"], doc["status"]) == ("package-smoke", "error")
        assert "artifact-not-found" in capsys.readouterr().out

    def test_dist_is_required(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            audit_main.main(["package-smoke"])
        assert excinfo.value.code == 2
        assert "--dist" in capsys.readouterr().err
