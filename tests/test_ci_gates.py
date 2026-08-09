"""Every check in tests.yml must be able to turn CI red.

``continue-on-error: true`` makes a check decorative: the step runs, the log
shows the failure in full, and the checkmark is green regardless. Nobody
reads a log under a green tick, so the check stops existing while still
looking like it exists.

This repo has now met that shape four times (three are written up in
``docs/solutions/architecture-patterns/global-config-you-cannot-win-and-claims-only-running-can-verify-2026-08-08.md``).
Two were literally this line: the mypy job, and the Black job that reported
green over a tree where 70 of 91 files would have been reformatted.

A guard against it is the same shape again. If :func:`non_blocking` quietly
stopped finding anything -- a key renamed, a workflow it can no longer
parse, a walk that skips steps -- it would report "everything blocks" on a
workflow made entirely of ``continue-on-error``. So its failing paths are
tested too, and none of that rests on the real file:

* :class:`TestTheExtractor` drives the pure function against synthetic
  workflows that *do* carry the defect, at job level and at step level;
* :func:`test_the_real_workflow_actually_parsed` establishes that the real
  file was read into jobs and steps, before any "found nothing" result from
  it is believed;
* :func:`test_the_allowlist_has_no_stale_entries` stops the allowlist from
  outliving the thing it excuses.

The workflow is read with a real YAML parser rather than by string matching
for the same reason: a text window that stopped matching would also fail
silently, in the passing direction.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "tests.yml"

#: Marks a ``continue-on-error`` on the job itself rather than on one step.
JOB_LEVEL = "<job>"

# The only place a non-blocking check is legitimate here. Codecov's upload is
# a call to a third-party service, not a check of this repo's code: an outage
# or a rate limit there must not fail a PR. Everything else in this workflow
# is a claim about the tree, and a claim about the tree has to be able to be
# wrong.
ALLOWED_NON_BLOCKING = {("test", "Upload coverage to Codecov")}


def non_blocking(workflow):
    """Locate every ``continue-on-error`` in a parsed workflow.

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
                label = step.get("name") or step.get("uses") or f"step {index}"
                found.add((job_id, label))
    return found


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


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


def test_the_real_workflow_actually_parsed(workflow):
    """Anchor for the rest: "found nothing" has to mean something.

    An empty or reshaped parse would make every assertion below pass
    vacuously -- the exact failure mode this module exists to prevent.
    """
    jobs = workflow.get("jobs") or {}
    missing = {"test", "lint", "type-check", "format-check"} - set(jobs)
    assert not missing, (
        f"tests.yml no longer defines the jobs this guard was written against: "
        f"{sorted(missing)} absent from {sorted(jobs)}"
    )
    assert all(
        job.get("steps") for job in jobs.values()
    ), "a job parsed with no steps -- step-level flags would be invisible"


def test_no_check_is_non_blocking_outside_the_allowlist(workflow):
    """The guard itself. Covers every job, including ones added later.

    Re-adding one line to the workflow switches a check off with every other
    check still green, which is why this is asserted rather than trusted.
    """
    unexpected = non_blocking(workflow) - ALLOWED_NON_BLOCKING
    assert not unexpected, (
        f"these checks pass whatever they find, so they are decorative: "
        f"{sorted(unexpected)}. Remove the `continue-on-error`, or add it to "
        f"ALLOWED_NON_BLOCKING with a reason it is not a claim about this tree."
    )


def test_the_allowlist_has_no_stale_entries(workflow):
    """An excuse for a step that no longer exists excuses nothing."""
    gone = ALLOWED_NON_BLOCKING - non_blocking(workflow)
    assert not gone, f"ALLOWED_NON_BLOCKING outlived what it excused: {sorted(gone)}"


def test_the_type_check_job_still_runs_the_mypy_gate(workflow):
    """Blocking is only half of it -- the job must still run the gate."""
    assert "tools/check_mypy_baseline.py" in job_script(workflow, "type-check")


def test_the_format_check_job_still_runs_black(workflow):
    assert "black --check" in job_script(workflow, "format-check")
