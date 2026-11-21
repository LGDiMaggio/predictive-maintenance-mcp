"""Guards for the Docker build boundary probes (``python -m tools.audit docker-probe``).

The probes themselves need Docker; the local run and the manual
``docker-image.yml`` workflow prove them on a real build. This module pins
what decides their verdicts without Docker:

* how the production runtime ``COPY`` lines are read from the ``Dockerfile``
  (continuations, comments, ``--from`` stages, flags, the JSON form);
* each verdict, on synthetic exported folders: a planted file that came out
  fails, a control file that did not come out fails, and the weight control's
  known gap is reported as a gap, never as a pass, and becomes an error once
  the control exists;
* a run without Docker, or with a probe build that fails, ends as ``error``,
  never ``pass``;
* ``docker-image.yml`` is the only workflow that builds the image, started by
  hand only, with a read-only token, a timeout, and the bypass check that an
  image defeating the ``COPY`` boundary fails;
* the recursive weight exclusions are still missing from ``.dockerignore``
  (a strict expected failure: it starts passing, and so fails, the day they
  land, so the marker cannot outlive the gap).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from _workflows import load_workflows
from tools.audit import __main__ as audit_main
from tools.audit import docker_probe, results

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DOCKERFILE = """\
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install .

# --- runtime ---
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages \\
     /usr/local/lib/python3.11/site-packages
# a comment between instructions
COPY src/ ./src/
COPY --chown=1000:1000 LICENSE ./LICENSE
COPY ["resources/a.json", "resources/b.md", "./resources/"]
RUN mkdir -p /app/data
"""


TINY_REPO_FILES = {
    "pyproject.toml": "[project]\nname = 'probe-fixture'\n",
    "README.md": "readme\n",
    "LICENSE": "license\n",
    "src/server.py": "print('server')\n",
    "resources/bearing_catalogs/README.md": "catalog readme\n",
    "resources/bearing_catalogs/common_bearings_catalog.json": "{}\n",
}


@pytest.fixture
def tiny_repo(tmp_path):
    """A minimal git repository with the production Dockerfile and
    .dockerignore, so no probe test copies the real checkout."""
    repo = tmp_path / "repo"
    files = {
        **TINY_REPO_FILES,
        "Dockerfile": (REPO_ROOT / "Dockerfile").read_text("utf-8"),
        ".dockerignore": (REPO_ROOT / ".dockerignore").read_text("utf-8"),
    }
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


@pytest.fixture
def fake_docker(monkeypatch):
    """Replace the probe build with a fake that exports the context files the
    given predicate selects: under ``ctx/`` for the context build, under
    ``app/`` for the COPY-line build."""

    def install(select):
        def runner(command, timeout):
            out = Path(command[command.index("--output") + 1].split("dest=", 1)[1])
            dockerfile = Path(command[command.index("-f") + 1]).read_text("utf-8")
            prefix = "ctx" if "COPY . /ctx" in dockerfile else "app"
            context = Path(command[-1])
            for path in context.rglob("*"):
                relative = path.relative_to(context).as_posix()
                if path.is_file() and select(relative):
                    target = out / prefix / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            out.mkdir(parents=True, exist_ok=True)
            return 0, "fake build"

        monkeypatch.setattr(docker_probe, "_subprocess_runner", runner)
        monkeypatch.setattr(docker_probe.shutil, "which", lambda name: name)

    return install


class TestDockerfileReading:
    def test_continuations_are_joined_and_comments_dropped(self):
        instructions = docker_probe.dockerfile_instructions(SAMPLE_DOCKERFILE)
        assert (
            "COPY --from=builder /usr/local/lib/python3.11/site-packages "
            "/usr/local/lib/python3.11/site-packages"
        ) in instructions
        assert not any(item.startswith("#") for item in instructions)

    def test_only_the_last_stage_context_copies_are_runtime_lines(self):
        assert docker_probe.runtime_copy_lines(SAMPLE_DOCKERFILE) == [
            "COPY src/ ./src/",
            "COPY --chown=1000:1000 LICENSE ./LICENSE",
            'COPY ["resources/a.json", "resources/b.md", "./resources/"]',
        ]

    @pytest.mark.parametrize(
        ("line", "sources"),
        [
            ("COPY src/ ./src/", ["src/"]),
            ("COPY --chown=1000:1000 LICENSE ./LICENSE", ["LICENSE"]),
            ("ADD a b c ./dest/", ["a", "b", "c"]),
            ('COPY ["x y.txt", "z", "/dest/"]', ["x y.txt", "z"]),
        ],
    )
    def test_copy_sources_drop_flags_and_the_destination(self, line, sources):
        assert docker_probe.copy_sources(line) == sources

    def test_a_dockerfile_without_from_is_refused(self):
        with pytest.raises(ValueError):
            docker_probe.runtime_copy_lines("COPY . .\n")


def _files(*names: str) -> set:
    return set(names)


class TestVerdicts:
    STAND_INS = ["src/__probe__.txt", "LICENSE"]

    def test_copy_lines_pass_when_only_the_stand_ins_come_out(self):
        files = _files("app/src/__probe__.txt", "app/LICENSE")
        outcome = docker_probe.copy_lines_verdict(files, self.STAND_INS)
        assert outcome.status == "pass"

    def test_copy_lines_fail_on_a_planted_file(self):
        files = _files("app/src/__probe__.txt", "app/LICENSE", "app/models/x_model.pkl")
        outcome = docker_probe.copy_lines_verdict(files, self.STAND_INS)
        assert (outcome.status, outcome.leaked) == ("fail", ["models/x_model.pkl"])

    def test_copy_lines_fail_when_a_stand_in_is_missing(self):
        # A build that copied nothing must not read as a clean pass.
        outcome = docker_probe.copy_lines_verdict(set(), self.STAND_INS)
        assert outcome.status == "fail"
        assert outcome.absent == sorted(self.STAND_INS)

    def test_context_passes_with_the_control_and_nothing_planted(self):
        outcome = docker_probe.context_verdict(set(docker_probe.POSITIVE_CONTROL))
        assert outcome.status == "pass"

    @pytest.mark.parametrize(
        "leaked", [*docker_probe.PLANTED_DATA, *docker_probe.EXCLUDED_TRACKED]
    )
    def test_context_fails_on_each_planted_or_excluded_file(self, leaked):
        files = set(docker_probe.POSITIVE_CONTROL) | {leaked}
        outcome = docker_probe.context_verdict(files)
        assert (outcome.status, outcome.leaked) == ("fail", [leaked])

    @pytest.mark.parametrize("absent", docker_probe.POSITIVE_CONTROL)
    def test_context_fails_when_a_control_file_is_missing(self, absent):
        files = set(docker_probe.POSITIVE_CONTROL) - {absent}
        outcome = docker_probe.context_verdict(files)
        assert (outcome.status, outcome.absent) == ("fail", [absent])

    @pytest.mark.parametrize(
        ("leaked", "expect_missing", "status"),
        [
            (True, False, "fail"),
            (True, True, "missing"),
            (False, False, "pass"),
            (False, True, "error"),
        ],
    )
    def test_the_weight_verdict_never_reads_a_gap_as_a_pass(
        self, leaked, expect_missing, status
    ):
        files = set(docker_probe.PLANTED_WEIGHTS) if leaked else set()
        outcome = docker_probe.weights_verdict(files, expect_missing)
        assert outcome.status == status


class TestNoEvidence:
    def test_without_docker_the_run_is_an_error(self, tmp_path, capsys):
        out = tmp_path / "probe.json"
        code = audit_main.main(
            [
                "docker-probe",
                "--docker",
                "definitely-not-a-docker-binary",
                "--json-out",
                str(out),
            ]
        )
        assert code == 2
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert doc["status"] == "error"
        assert [f["rule"] for f in doc["findings"]] == ["docker-unavailable"]
        assert doc["facts"]["probes"] == {}
        assert "docker-probe error" in capsys.readouterr().out

    def test_a_failed_probe_build_is_an_error_for_both_context_probes(
        self, tmp_path, tiny_repo
    ):
        def failing_runner(command, timeout):
            return 1, "ERROR: failed to solve"

        outcomes = docker_probe.probe_context(
            tiny_repo, "docker", tmp_path / "work", failing_runner, True
        )
        assert [(o.probe, o.status) for o in outcomes] == [
            ("context", "error"),
            ("weights", "error"),
        ]

    def test_a_root_git_cannot_list_ends_as_an_error_not_a_traceback(
        self, tmp_path, monkeypatch, fake_docker
    ):
        root = tmp_path / "not-a-repo"
        root.mkdir()
        (root / "Dockerfile").write_text(
            (REPO_ROOT / "Dockerfile").read_text("utf-8"), encoding="utf-8"
        )
        fake_docker(lambda rel: True)
        out = tmp_path / "probe.json"
        code = audit_main.main(
            ["docker-probe", "--repo-root", str(root), "--json-out", str(out)]
        )
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert (code, doc["status"]) == (2, "error")
        assert doc["facts"]["probes"]["context"]["status"] == "error"

    def test_an_error_outcome_makes_the_whole_run_an_error(self):
        outcomes = [
            docker_probe.ProbeOutcome("copy-lines", "pass"),
            docker_probe.ProbeOutcome("context", "error", detail="x"),
        ]
        assert docker_probe._status(outcomes, []) == "error"

    def test_the_known_weight_gap_alone_does_not_fail_the_run(self):
        outcomes = [
            docker_probe.ProbeOutcome("copy-lines", "pass"),
            docker_probe.ProbeOutcome("context", "pass"),
            docker_probe.ProbeOutcome("weights", "missing"),
        ]
        assert docker_probe._status(outcomes, []) == "pass"


class TestRunWithFakeDocker:
    """The whole run, through the CLI, with a fake build that exports a chosen
    subset of each probe's context."""

    def _run(self, tmp_path, repo):
        out = tmp_path / "probe.json"
        code = audit_main.main(
            [
                "docker-probe",
                "--repo-root",
                str(repo),
                "--expect-missing",
                "weights",
                "--json-out",
                str(out),
            ]
        )
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        return code, doc

    def test_a_leaking_build_fails_and_names_the_leaked_files(
        self, tmp_path, tiny_repo, fake_docker
    ):
        fake_docker(lambda rel: True)
        code, doc = self._run(tmp_path, tiny_repo)
        assert (code, doc["status"]) == (1, "fail")
        rules = {f["rule"] for f in doc["findings"]}
        assert {"docker-probe-copy-lines", "docker-probe-context"} <= rules
        leaked = doc["facts"]["probes"]["context"]["leaked"]
        assert "data/signals/real_train/planted.mat" in leaked
        assert "models/x_model.pkl" in leaked

    def test_a_clean_build_passes_with_the_weight_gap_reported(
        self, tmp_path, tiny_repo, fake_docker
    ):
        planted = set(docker_probe.PLANTED_DATA)
        fake_docker(lambda rel: rel not in planted)
        code, doc = self._run(tmp_path, tiny_repo)
        assert (code, doc["status"]) == (0, "pass")
        probes = doc["facts"]["probes"]
        assert probes["copy-lines"]["status"] == "pass"
        assert probes["context"]["status"] == "pass"
        assert probes["weights"]["status"] == "missing"

    def test_a_copy_dot_line_is_probed_not_crashed_on(
        self, tmp_path, tiny_repo, fake_docker
    ):
        dockerfile = tiny_repo / "Dockerfile"
        dockerfile.write_text(
            dockerfile.read_text("utf-8") + "COPY . .\n", encoding="utf-8"
        )
        fake_docker(lambda rel: True)
        code, doc = self._run(tmp_path, tiny_repo)
        assert (code, doc["status"]) == (1, "fail")
        assert doc["facts"]["probes"]["copy-lines"]["status"] == "fail"

    def test_a_fail_outcome_makes_the_whole_run_fail(self):
        outcomes = [
            docker_probe.ProbeOutcome("copy-lines", "pass"),
            docker_probe.ProbeOutcome("context", "fail", leaked=["models/x.pkl"]),
        ]
        assert docker_probe._status(outcomes, []) == "fail"


class TestWeightExclusions:
    WEIGHT_PATTERNS = ("**/*.safetensors", "**/*.bin", "**/*.pt", "**/*.pth")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "known gap: .dockerignore has no recursive weight exclusions yet; the "
            "change that adds them removes this marker"
        ),
    )
    def test_dockerignore_excludes_weight_files_after_every_reinclude(self):
        lines = [
            line.strip()
            for line in (REPO_ROOT / ".dockerignore").read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        last_reinclude = max(i for i, line in enumerate(lines) if line.startswith("!"))
        after = set(lines[last_reinclude + 1 :])
        assert set(self.WEIGHT_PATTERNS) <= after


class TestImageWorkflow:
    """``docker-image.yml`` is the only workflow that builds the image, and it
    runs only when started by hand, with a read-only token."""

    @pytest.fixture(scope="class")
    def workflows(self):
        return load_workflows()

    def test_it_runs_only_on_manual_dispatch_without_inputs(self, workflows):
        workflow = workflows["docker-image.yml"]
        trigger = workflow.get("on", workflow.get(True))
        assert trigger == {"workflow_dispatch": None}

    def test_its_token_is_read_only_and_its_job_is_bounded(self, workflows):
        workflow = workflows["docker-image.yml"]
        assert workflow["permissions"] == {"contents": "read"}
        job = workflow["jobs"]["image"]
        assert job.get("timeout-minutes")
        # bash -eo pipefail, so a failing docker export fails its step
        assert job["defaults"]["run"]["shell"] == "bash"
        assert "continue-on-error" not in job
        assert not any("continue-on-error" in step for step in job["steps"])

    def test_it_runs_the_probes_the_image_check_and_the_bypass_check(self, workflows):
        runs = "\n".join(
            step.get("run", "")
            for step in workflows["docker-image.yml"]["jobs"]["image"]["steps"]
        )
        assert "python -m tools.audit docker-probe" in runs
        assert "artifact-licensing --image -" in runs
        assert "COPY planted/ /app/models/" in runs
        assert 'test "$code" -eq 1' in runs

    #: Every way a workflow step can build an image.
    BUILD = re.compile(
        r"\bdocker(?:\s+buildx)?\s+build\b|\bdocker(?:\s+|-)compose\s+build\b"
    )

    @classmethod
    def _builds_an_image(cls, step: dict) -> bool:
        return bool(cls.BUILD.search(str(step.get("run", "")))) or str(
            step.get("uses", "")
        ).startswith("docker/build-push-action")

    @pytest.mark.parametrize(
        "step",
        [
            {"run": "docker build -t x ."},
            {"run": "docker buildx build --push ."},
            {"run": "docker compose build"},
            {"run": "docker-compose build"},
            {"uses": "docker/build-push-action@v6"},
        ],
    )
    def test_every_build_form_is_recognised(self, step):
        assert self._builds_an_image(step)

    def test_no_other_workflow_builds_an_image(self, workflows):
        builders = {
            name
            for name, workflow in workflows.items()
            for job in (workflow.get("jobs") or {}).values()
            for step in job.get("steps", [])
            if self._builds_an_image(step)
        }
        assert builders == {"docker-image.yml"}


class TestCli:
    def test_the_subcommand_is_dispatched(self):
        assert audit_main.DISPATCH["docker-probe"].module == "tools.audit.docker_probe"

    def test_only_the_weight_control_can_be_expected_missing(self):
        with pytest.raises(SystemExit) as exc:
            audit_main.main(["docker-probe", "--expect-missing", "context"])
        assert exc.value.code == 2
