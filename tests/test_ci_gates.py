"""Every check in every workflow must be able to turn CI red.

``continue-on-error: true`` makes a check decorative: the step runs, the log
shows the failure in full, and the checkmark is green regardless. Nobody
reads a log under a green tick, so the check stops existing while still
looking like it exists.

This repo has now met that shape four times (three are written up in
``docs/solutions/architecture-patterns/global-config-you-cannot-win-and-claims-only-running-can-verify-2026-08-08.md``).
Two were literally this line: the mypy job, and the Black job that reported
green over a tree where 70 of 91 files would have been reformatted.

The guard started on ``tests.yml`` alone. It now sweeps every file under
``.github/workflows``: the weekly audit's report job carries its verdict in
its exit code, and a release smoke job that could not fail would publish an
untested wheel, so a decorative step anywhere is the same defect.

A guard against it is the same shape again. If :func:`non_blocking` quietly
stopped finding anything -- a key renamed, a workflow it can no longer
parse, a walk that skips steps or files -- it would report "everything
blocks" on a workflow made entirely of ``continue-on-error``. So its failing
paths are tested too, and none of that rests on the real files:

* :class:`TestTheExtractor` drives the pure functions against synthetic
  workflows that *do* carry the defect, at job level and at step level, in
  ``tests.yml`` and in any other workflow file;
* :func:`test_the_real_workflows_actually_parsed` establishes that every
  real file was read into jobs and steps, before any "found nothing" result
  from them is believed;
* :func:`test_the_allowlist_has_no_stale_entries` stops the allowlist from
  outliving the thing it excuses.

The workflows are read with a real YAML parser rather than by string
matching for the same reason: a text window that stopped matching would also
fail silently, in the passing direction.
"""

import pytest
import yaml

from _workflows import WORKFLOWS_DIR, load_workflows, step_label

WORKFLOW_PATH = WORKFLOWS_DIR / "tests.yml"

#: Workflow files this guard was written against; more are swept as added.
KNOWN_WORKFLOWS = {"tests.yml", "publish.yml", "publish-mcp.yml", "audit.yml"}

#: Marks a ``continue-on-error`` on the job itself rather than on one step.
JOB_LEVEL = "<job>"

# The only place a non-blocking check is legitimate here. Codecov's upload is
# a call to a third-party service, not a check of this repo's code: an outage
# or a rate limit there must not fail a PR. Everything else in these
# workflows is a claim about the tree, and a claim about the tree has to be
# able to be wrong. Keys are (workflow file, job, step).
ALLOWED_NON_BLOCKING = {("tests.yml", "test", "Upload coverage to Codecov")}


def non_blocking(workflow):
    """Locate every ``continue-on-error`` in one parsed workflow.

    Returns ``(job_id, where)`` pairs, where *where* is the step's name or
    :data:`JOB_LEVEL`. Only a literal ``false`` counts as blocking -- an
    expression such as ``${{ inputs.soft }}`` is reported, because whether it
    blocks is then decided somewhere this test cannot read.
    """
    found = set()
    for job_id, job in (workflow.get("jobs") or {}).items():
        if job.get("continue-on-error", False) is not False:
            found.add((job_id, JOB_LEVEL))
        for index, step in enumerate(job.get("steps") or []):
            if step.get("continue-on-error", False) is not False:
                found.add((job_id, step_label(step, index)))
    return found


def non_blocking_in(workflows):
    """:func:`non_blocking` over ``{file name: parsed workflow}``.

    Returns ``(file name, job_id, where)`` triples, so the same job and step
    names in two workflows stay distinct findings.
    """
    return {
        (name, job_id, where)
        for name, workflow in workflows.items()
        for job_id, where in non_blocking(workflow)
    }


@pytest.fixture(scope="module")
def workflows():
    return load_workflows()


@pytest.fixture(scope="module")
def workflow(workflows):
    return workflows[WORKFLOW_PATH.name]


def job_script(workflow, job_id):
    """Everything a job runs, concatenated -- for ``in`` assertions."""
    return "\n".join(step.get("run", "") for step in workflow["jobs"][job_id]["steps"])


AT_JOB_LEVEL = """\
jobs:
  format-check:
    continue-on-error: true
    steps:
      - run: black --check src tests
"""

AT_STEP_LEVEL = """\
jobs:
  format-check:
    steps:
      - name: Check code formatting
        run: black --check src tests
        continue-on-error: true
"""

UNNAMED_STEPS = """\
jobs:
  test:
    steps:
      - uses: codecov/codecov-action@v4
        continue-on-error: true
      - run: echo hi
        continue-on-error: true
"""

EXPLICITLY_FALSE = """\
jobs:
  lint:
    continue-on-error: false
    steps:
      - run: flake8 src
        continue-on-error: false
"""

AN_EXPRESSION = """\
jobs:
  lint:
    continue-on-error: ${{ github.event_name == 'push' }}
    steps:
      - run: flake8 src
"""

A_JOB_THIS_FILE_HAS_NEVER_HEARD_OF = """\
jobs:
  brand-new-job:
    steps:
      - name: Something nobody has written yet
        run: ./whatever
        continue-on-error: true
"""

ANOTHER_WORKFLOW = """\
on: workflow_dispatch
jobs:
  report:
    continue-on-error: true
    steps:
      - name: Build the report
        run: python -m tools.audit report
  security:
    steps:
      - name: Normalize the audits
        run: python -m tools.audit security
        continue-on-error: true
"""

# Same job and step name as the allowlisted Codecov step, in another file.
THE_ALLOWLISTED_STEP_ELSEWHERE = """\
jobs:
  test:
    steps:
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        continue-on-error: true
"""


class TestTheExtractor:
    """The guard's own failing paths. Without these it can only pass."""

    def test_a_non_blocking_job_is_found(self):
        assert non_blocking(yaml.safe_load(AT_JOB_LEVEL)) == {
            ("format-check", JOB_LEVEL)
        }

    def test_a_non_blocking_step_is_found(self):
        """The shape the Black job actually had: the flag under the step."""
        assert non_blocking(yaml.safe_load(AT_STEP_LEVEL)) == {
            ("format-check", "Check code formatting")
        }

    def test_an_unnamed_step_is_still_located(self):
        """A finding nobody can find in the file is barely a finding."""
        assert non_blocking(yaml.safe_load(UNNAMED_STEPS)) == {
            ("test", "codecov/codecov-action@v4"),
            ("test", "step 1"),
        }

    def test_an_explicit_false_is_blocking(self):
        """Spelling out the default must not force an allowlist entry."""
        assert non_blocking(yaml.safe_load(EXPLICITLY_FALSE)) == set()

    def test_an_expression_is_reported_rather_than_trusted(self):
        """Whether ``${{ ... }}`` blocks is decided outside this file."""
        assert non_blocking(yaml.safe_load(AN_EXPRESSION)) == {("lint", JOB_LEVEL)}

    def test_every_job_is_swept_not_just_the_known_ones(self):
        """A job added tomorrow is covered without editing this file."""
        assert non_blocking(yaml.safe_load(A_JOB_THIS_FILE_HAS_NEVER_HEARD_OF)) == {
            ("brand-new-job", "Something nobody has written yet")
        }

    def test_another_workflow_file_is_swept_at_job_and_step_level(self):
        """The flag outside tests.yml is the same defect."""
        found = non_blocking_in(
            {
                "tests.yml": yaml.safe_load(EXPLICITLY_FALSE),
                "audit.yml": yaml.safe_load(ANOTHER_WORKFLOW),
            }
        )
        assert found == {
            ("audit.yml", "report", JOB_LEVEL),
            ("audit.yml", "security", "Normalize the audits"),
        }

    def test_the_allowlist_excuses_one_file_not_a_step_name(self):
        """Copying the Codecov step's name into another workflow is not excused."""
        found = non_blocking_in(
            {"publish.yml": yaml.safe_load(THE_ALLOWLISTED_STEP_ELSEWHERE)}
        )
        assert found - ALLOWED_NON_BLOCKING == {
            ("publish.yml", "test", "Upload coverage to Codecov")
        }


def test_the_real_workflows_actually_parsed(workflows):
    """Anchor for the rest: "found nothing" has to mean something.

    An empty or reshaped parse would make every assertion below pass
    vacuously -- the exact failure mode this module exists to prevent.
    """
    missing_files = KNOWN_WORKFLOWS - set(workflows)
    assert not missing_files, (
        f"the workflow sweep no longer sees {sorted(missing_files)}; "
        f"it found {sorted(workflows)} in {WORKFLOWS_DIR}"
    )
    for name, parsed in workflows.items():
        jobs = (parsed or {}).get("jobs") or {}
        assert jobs, f"{name} parsed with no jobs"
        stepless = sorted(
            job_id for job_id, job in jobs.items() if not job.get("steps")
        )
        assert not stepless, (
            f"{name}: {stepless} parsed with no steps -- step-level flags "
            f"would be invisible"
        )
    jobs = workflows[WORKFLOW_PATH.name]["jobs"]
    missing = {"test", "lint", "type-check", "format-check"} - set(jobs)
    assert not missing, (
        f"tests.yml no longer defines the jobs this guard was written against: "
        f"{sorted(missing)} absent from {sorted(jobs)}"
    )


def test_no_check_is_non_blocking_outside_the_allowlist(workflows):
    """The guard itself. Covers every job of every workflow, including new ones.

    Re-adding one line to a workflow switches a check off with every other
    check still green, which is why this is asserted rather than trusted.
    """
    unexpected = non_blocking_in(workflows) - ALLOWED_NON_BLOCKING
    assert not unexpected, (
        f"these checks pass whatever they find, so they are decorative: "
        f"{sorted(unexpected)}. Remove the `continue-on-error`, or add it to "
        f"ALLOWED_NON_BLOCKING with a reason it is not a claim about this tree."
    )


def test_the_allowlist_has_no_stale_entries(workflows):
    """An excuse for a step that no longer exists excuses nothing."""
    gone = ALLOWED_NON_BLOCKING - non_blocking_in(workflows)
    assert not gone, f"ALLOWED_NON_BLOCKING outlived what it excused: {sorted(gone)}"


def test_the_type_check_job_still_runs_the_mypy_gate(workflow):
    """Blocking is only half of it -- the job must still run the gate."""
    assert "tools/check_mypy_baseline.py" in job_script(workflow, "type-check")


def test_the_format_check_job_still_runs_black(workflow):
    assert "black --check" in job_script(workflow, "format-check")


def test_the_pull_request_package_smoke_job_still_installs_the_wheel(workflow):
    assert "python -m tools.audit package-smoke" in job_script(
        workflow, "package-smoke"
    )


def test_the_pull_request_package_smoke_job_checks_both_artifacts_licensing(workflow):
    """The job builds the sdist too and reads both artifacts, not MANIFEST.in."""
    script = job_script(workflow, "package-smoke")
    assert "python -m build --wheel" not in script
    assert "python -m build" in script
    assert "python -m tools.audit artifact-licensing --dist dist" in script


def test_no_release_upload_runs_before_the_artifacts_licensing_is_checked(workflows):
    """The smoke job every upload waits for also checks the licensing of the
    exact sdist and wheel the build job produced."""
    publish = workflows["publish.yml"]
    assert "python -m tools.audit artifact-licensing --dist dist" in job_script(
        publish, "smoke"
    )


ARCHIVE_COMMANDS = (
    "git archive --format=zip --prefix=predictive-maintenance-mcp/ "
    '-o "$RUNNER_TEMP/source.zip" HEAD',
    "git archive --format=tar.gz --prefix=predictive-maintenance-mcp/ "
    '-o "$RUNNER_TEMP/source.tar.gz" HEAD',
    "python -m tools.audit artifact-licensing "
    '--archive "$RUNNER_TEMP/source.zip" --archive "$RUNNER_TEMP/source.tar.gz"',
)


def test_the_pull_request_package_smoke_job_checks_both_source_archive_formats(
    workflow,
):
    """The checked-out commit's ZIP and tarball, as GitHub and Zenodo get them."""
    script = job_script(workflow, "package-smoke")
    for command in ARCHIVE_COMMANDS:
        assert command in script


def test_no_release_upload_runs_before_the_source_archives_licensing_is_checked(
    workflows,
):
    """The smoke job every upload waits for archives the release commit it
    checked out, in both formats, and checks them."""
    publish = workflows["publish.yml"]
    smoke = publish["jobs"]["smoke"]
    assert any("actions/checkout" in step.get("uses", "") for step in smoke["steps"])
    script = job_script(publish, "smoke")
    for command in ARCHIVE_COMMANDS:
        assert command in script


def test_no_release_upload_runs_before_the_wheel_is_smoke_tested(workflows):
    """The TestPyPI upload waits for the smoke job, and PyPI waits for TestPyPI,
    so a wheel that does not install and start never reaches either index."""
    publish = workflows["publish.yml"]
    assert "python -m tools.audit package-smoke" in job_script(publish, "smoke")
    needs = publish["jobs"]["publish-testpypi"].get("needs") or []
    assert "smoke" in ([needs] if isinstance(needs, str) else needs)
    pypi_needs = publish["jobs"]["publish-pypi"].get("needs") or []
    assert "publish-testpypi" in (
        [pypi_needs] if isinstance(pypi_needs, str) else pypi_needs
    )
