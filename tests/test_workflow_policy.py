"""Workflow tokens stay minimal, and the weekly audit stays AI-free.

Every workflow here runs with a ``GITHUB_TOKEN`` whose scopes are whatever
the YAML grants. A grant is invisible in a green run: a job that could push
code or file issues looks exactly like one that can only read. The weekly
audit (``audit.yml``) runs on a schedule with nobody watching, which is where
a stray write scope or an AI provider key would do the most damage unnoticed.
So these rules are asserted, not trusted:

1. each workflow declares top-level ``permissions``, or every job does
   (without either, the token gets the repository default, which can be
   read-write on everything);
2. write scopes come only from :data:`ALLOWED_WRITE_SCOPES`, the trusted
   publishing grants (``id-token: write``) of the three publish jobs;
3. nothing uses ``write-all``;
4. ``audit.yml`` runs on ``schedule`` and ``workflow_dispatch`` and never
   holds ``issues``, ``contents`` or ``pull-requests`` write: issue
   candidates are for humans, and the audit never writes to the tracker;
5. ``audit.yml`` references no secret except the default token;
6. ``audit.yml`` names no environment variable or input after an AI
   provider (no AI step in any workflow);
7. every check the report builder expects
   (``tools.audit.results.EXPECTED_WEEKLY_CHECKS``) has an artifact upload
   in ``audit.yml``, so the two lists cannot drift apart;
8. ``dependabot.yml`` ignores semver-major updates of ``mcp`` and
   ``mcp-types`` in every Python ecosystem entry (never auto-upgrade to a
   new MCP major).

Contracts between ``audit.yml`` and the report builder are pinned too:

9. the ``gh issue list --limit`` equals ``KNOWN_ISSUES_LIMIT``, the size at
   which the builder treats the known-issue list as possibly truncated;
10. every known-issue fetch filters on a label and asks for ``labels``, one
    of them on the report's ``ACCEPTED_LABEL``, and no issue form applies
    that label, so an issue anyone can open never tracks a finding until a
    maintainer labels it;
11. the report step expects one result per matrix leg (``check:variant``),
    and exactly the legs the matrices declare, so a leg that uploaded nothing
    is reported missing;
12. every audit job sets ``timeout-minutes``.

Each rule is a pure function over parsed YAML with red-path tests against
synthetic documents that carry the defect, following ``tests/test_ci_gates.py``:
a rule that quietly stopped finding anything would otherwise report every real
file as clean. :func:`test_the_real_files_actually_parsed` establishes that
the real files were read into jobs before any "no violations" is believed.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Iterator, Optional

import pytest
import yaml

from _workflows import REPO_ROOT, load_workflows, step_label
from tools.audit import report, results

DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"
ISSUE_TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
AUDIT_WORKFLOW = "audit.yml"

#: Marks a permissions block on the workflow rather than on one job.
WORKFLOW_LEVEL = "<workflow>"

#: (workflow file, job, scope) triples allowed to write. Trusted publishing
#: needs ``id-token: write`` to mint the OIDC token PyPI and the MCP Registry
#: accept; nothing else in this repository writes with the workflow token.
ALLOWED_WRITE_SCOPES = {
    ("publish.yml", "publish-testpypi", "id-token"),
    ("publish.yml", "publish-pypi", "id-token"),
    ("publish-mcp.yml", "publish-mcp", "id-token"),
}

#: Scopes the weekly audit must never hold with write access.
AUDIT_FORBIDDEN_WRITES = ("issues", "contents", "pull-requests")

AI_PROVIDER_NAME = re.compile(r"ANTHROPIC|OPENAI|CLAUDE|CODEX|GEMINI", re.IGNORECASE)

#: Uploaded artifact names that carry each weekly check's evidence, as
#: ("prefix" | "exact", value). Matrix jobs upload one artifact per leg.
CHECK_ARTIFACTS = {
    "suite": ("prefix", "suite-"),
    "package-smoke": ("prefix", "package-smoke-"),
    "upstream": ("exact", "upstream"),
    "security": ("exact", "security"),
}

PYTHON_ECOSYSTEMS = ("pip", "uv")
MCP_DISTRIBUTIONS = ("mcp", "mcp-types")
SEMVER_MAJOR = "version-update:semver-major"
DEFAULT_TOKEN = "GITHUB_TOKEN"


# ---------------------------------------------------------------------------
# Reading workflows
# ---------------------------------------------------------------------------


def triggers(workflow: dict) -> dict:
    """The workflow's events as a mapping, whichever of the three forms it uses.

    PyYAML implements YAML 1.1, where a bare ``on`` key is the boolean
    ``True``, so the key is looked up under both spellings.
    """
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {event: None for event in raw}
    return dict(raw or {})


def _jobs(workflow: dict) -> dict:
    return workflow.get("jobs") or {}


def _steps(job: dict) -> Iterator[tuple[str, dict]]:
    for index, step in enumerate(job.get("steps") or []):
        yield step_label(step, index), step


def _permission_blocks(workflow: dict) -> Iterator[tuple[str, Any]]:
    """(job id or :data:`WORKFLOW_LEVEL`, block) for every declared block."""
    if "permissions" in workflow:
        yield WORKFLOW_LEVEL, workflow["permissions"]
    for job_id, job in _jobs(workflow).items():
        if "permissions" in job:
            yield job_id, job["permissions"]


# ---------------------------------------------------------------------------
# The rules: parsed YAML in, violations out
# ---------------------------------------------------------------------------


def undeclared_permissions(workflow: dict) -> list[str]:
    """Rule 1: jobs whose token falls back to the repository default.

    An empty ``permissions:`` key parses to None and declares nothing;
    ``permissions: {}`` (no scopes at all) is a declaration.
    """
    if workflow.get("permissions") is not None:
        return []
    jobs = _jobs(workflow)
    if not jobs:
        return [WORKFLOW_LEVEL]
    return sorted(
        job_id for job_id, job in jobs.items() if job.get("permissions") is None
    )


def write_grants(name: str, workflow: dict) -> set[tuple[str, str, str]]:
    """Rule 2: every (file, job, scope) the workflow's token can write with.

    Only ``read`` and ``none`` are known to be harmless; any other access
    level, a typo included, is reported rather than assumed read-only. A
    string block other than ``read-all`` is reported whole.
    """
    found = set()
    for where, block in _permission_blocks(workflow):
        if isinstance(block, str):
            if block != "read-all":
                found.add((name, where, block))
        elif isinstance(block, dict):
            for scope, access in block.items():
                if access not in ("read", "none"):
                    found.add((name, where, str(scope)))
        elif block is not None:
            found.add((name, where, repr(block)))
    return found


def write_all(workflow: dict) -> list[str]:
    """Rule 3: where ``write-all`` is granted."""
    return sorted(
        where for where, block in _permission_blocks(workflow) if block == "write-all"
    )


def audit_policy_violations(workflow: dict) -> list[str]:
    """Rule 4: missing triggers and tracker or repository write scopes."""
    on = triggers(workflow)
    problems = []
    if not on.get("schedule"):
        problems.append("no schedule trigger")
    if "workflow_dispatch" not in on:
        problems.append("no workflow_dispatch trigger")
    for where, block in _permission_blocks(workflow):
        if block == "write-all":
            scopes = list(AUDIT_FORBIDDEN_WRITES)
        elif isinstance(block, dict):
            scopes = [
                scope
                for scope in AUDIT_FORBIDDEN_WRITES
                if block.get(scope) not in (None, "read", "none")
            ]
        else:
            scopes = []
        problems += [f"{where}: {scope}: write" for scope in scopes]
    return problems


EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)

#: The secrets context in any spelling: ``secrets.NAME``, ``secrets['NAME']``
#: or the whole context (``toJSON(secrets)`` would print every secret).
SECRET_REFERENCE = re.compile(
    r"\bsecrets\b(?:\s*\.\s*([A-Za-z0-9_-]+)|\s*\[\s*['\"]([^'\"]*)['\"]\s*\])?"
)


def _strings(node: Any, key: Optional[str] = None) -> Iterator[tuple[Any, str]]:
    """(owning key, text) for every string value in a parsed document."""
    if isinstance(node, dict):
        for child_key, value in node.items():
            yield from _strings(value, child_key)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item, key)
    elif isinstance(node, str):
        yield key, node


def _expressions(workflow: dict) -> list[str]:
    """Every expression: ``${{ ... }}`` anywhere, and ``if:`` values whole.

    ``if:`` is evaluated as an expression with or without the braces, and
    only expressions can read the secrets context, so words in step names or
    scripts are not references.
    """
    found = []
    for key, text in _strings(workflow):
        if key == "if":
            found.append(text)
        else:
            found += EXPRESSION.findall(text)
    return found


def foreign_secrets(workflow: dict) -> list[str]:
    """Rule 5: references to the secrets context other than the default token."""
    found = []
    for expression in _expressions(workflow):
        for match in SECRET_REFERENCE.finditer(expression):
            name = match.group(1) or match.group(2)
            if name is None or name.upper() != DEFAULT_TOKEN:
                found.append(re.sub(r"\s+", "", match.group(0)))
    return sorted(found)


def ai_provider_names(workflow: dict) -> list[str]:
    """Rule 6: env, input and action-input names that point at an AI provider.

    Swept: workflow, job and step ``env``; the inputs and secrets a
    ``workflow_dispatch`` or ``workflow_call`` trigger declares; and the
    ``with:`` inputs passed to an action or a reusable workflow.
    """
    found = []

    def sweep(where: str, kind: str, mapping: Any) -> None:
        if isinstance(mapping, dict):
            found.extend(
                f"{where} {kind} {name}"
                for name in mapping
                if AI_PROVIDER_NAME.search(str(name))
            )

    sweep(WORKFLOW_LEVEL, "env", workflow.get("env"))
    for event, config in triggers(workflow).items():
        if isinstance(config, dict):
            for kind in ("inputs", "secrets"):
                sweep(f"on.{event}.{kind}", "", config.get(kind))
    for job_id, job in _jobs(workflow).items():
        sweep(job_id, "env", job.get("env"))
        sweep(job_id, "with", job.get("with"))
        for label, step in _steps(job):
            sweep(f"{job_id}/{label}", "env", step.get("env"))
            sweep(f"{job_id}/{label}", "with", step.get("with"))
    return sorted(" ".join(item.split()) for item in found)


def uploaded_artifacts(workflow: dict) -> set[str]:
    """Names of every ``actions/upload-artifact`` step's artifact."""
    names = set()
    for job in _jobs(workflow).values():
        for _, step in _steps(job):
            if str(step.get("uses", "")).startswith("actions/upload-artifact@"):
                name = (step.get("with") or {}).get("name")
                if name:
                    names.add(str(name))
    return names


def checks_without_artifacts(workflow: dict, expected) -> list[str]:
    """Rule 7: expected check ids with no matching artifact upload.

    An expected id missing from :data:`CHECK_ARTIFACTS` is reported too: a
    check added to the report builder's list must be wired here, not skipped.
    """
    uploaded = uploaded_artifacts(workflow)
    missing = []
    for check in expected:
        if check not in CHECK_ARTIFACTS:
            missing.append(f"{check} (no artifact mapping in CHECK_ARTIFACTS)")
            continue
        kind, value = CHECK_ARTIFACTS[check]
        matches = [
            name
            for name in uploaded
            if (name.startswith(value) if kind == "prefix" else name == value)
        ]
        if not matches:
            missing.append(check)
    return missing


GH_ISSUE_LIST_LIMIT = re.compile(r"--limit[\s=]+(\d+)")


def known_issue_fetch_limits(workflow: dict) -> list[Optional[int]]:
    """``--limit`` of every ``gh issue list`` call; None where it is absent.

    The report builder treats a list of exactly ``KNOWN_ISSUES_LIMIT`` issues
    as possibly truncated. A fetch with a smaller limit, or gh's default of
    30, would truncate below that check and drop de-duplication silently.
    """
    limits: list[Optional[int]] = []
    for job in _jobs(workflow).values():
        for _, step in _steps(job):
            # One shell command per line once backslash continuations are joined.
            script = re.sub(r"\\\r?\n", " ", str(step.get("run", "")))
            for line in script.splitlines():
                for command in line.split("gh issue list")[1:]:
                    match = GH_ISSUE_LIST_LIMIT.search(command)
                    limits.append(int(match.group(1)) if match else None)
    return limits


GH_ISSUE_LIST_LABEL = re.compile(r"--label[\s=]+['\"]?([^\s'\"]+)")
GH_ISSUE_LIST_JSON = re.compile(r"--json[\s=]+['\"]?([^\s'\"]+)")


def known_issue_trust_gaps(workflow: dict, label: str) -> list[str]:
    """Rule 10, workflow half: known-issue fetches the report cannot judge.

    The report decides acceptance from each issue's labels, so every
    ``gh issue list`` must ask for ``labels``. Every fetch must filter on a
    label (a plain list query: no unfiltered or search fetch), and one of them
    on ``label``, the acceptance label.
    """
    gaps = []
    accepted_fetches = 0
    for job_id, job in _jobs(workflow).items():
        for step_name, step in _steps(job):
            script = re.sub(r"\\\r?\n", " ", str(step.get("run", "")))
            for line in script.splitlines():
                for command in line.split("gh issue list")[1:]:
                    where = f"{job_id}/{step_name}"
                    labels = GH_ISSUE_LIST_LABEL.findall(command)
                    if not labels:
                        gaps.append(f"{where}: no --label filter")
                    accepted_fetches += label in labels
                    fields = GH_ISSUE_LIST_JSON.search(command)
                    if not fields or "labels" not in fields.group(1).split(","):
                        gaps.append(f"{where}: --json without labels")
    if not accepted_fetches:
        gaps.append(f"no fetch filters on --label {label}")
    return gaps


def template_labels(text: str, suffix: str) -> list[str]:
    """Labels an issue template applies: a form's ``labels`` key, or the
    front matter of a Markdown template (a list or a comma-separated string)."""
    if suffix == ".md":
        match = re.match(r"\A---\r?\n(.*?)\r?\n---", text, re.DOTALL)
        doc = yaml.safe_load(match.group(1)) if match else None
    else:
        doc = yaml.safe_load(text)
    labels = doc.get("labels") if isinstance(doc, dict) else None
    if isinstance(labels, str):
        labels = [part.strip() for part in labels.split(",")]
    return [str(item) for item in labels or [] if str(item).strip()]


def templates_applying(templates: dict[str, list[str]], label: str) -> list[str]:
    """Rule 10, template half: templates whose labels include ``label``."""
    return sorted(
        name
        for name, labels in templates.items()
        if any(item.casefold() == label.casefold() for item in labels)
    )


MATRIX_VALUE = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _render_matrix(template: str, leg: dict) -> str:
    return MATRIX_VALUE.sub(lambda m: str(leg.get(m.group(1), m.group(0))), template)


def expected_results_gaps(workflow: dict) -> list[str]:
    """Rule 11: differences between the report's ``--expect`` and the legs.

    Suite legs are named after their uploaded artifact (the directory the
    report reads); package-smoke legs after the ``--variant`` they record.
    """
    jobs = _jobs(workflow)
    wanted = {"upstream", "security"}
    suite = jobs.get("suite") or {}
    artifact = next(
        (
            str((step.get("with") or {}).get("name", ""))
            for _, step in _steps(suite)
            if str(step.get("uses", "")).startswith("actions/upload-artifact@")
        ),
        "",
    )
    for leg in (suite.get("strategy") or {}).get("matrix", {}).get("include") or []:
        wanted.add(f"suite:{_render_matrix(artifact, leg)}")
    smoke = jobs.get("package-smoke") or {}
    for leg in (smoke.get("strategy") or {}).get("matrix", {}).get("include") or []:
        wanted.add(f"package-smoke:{leg.get('variant')}")
    report_steps = [
        step
        for _, step in _steps(jobs.get("report") or {})
        if "tools.audit report" in str(step.get("run", ""))
    ]
    if len(report_steps) != 1:
        return [f"expected one report step, found {len(report_steps)}"]
    (step,) = report_steps
    if '--expect "$EXPECTED_RESULTS"' not in str(step.get("run", "")):
        return ['the report step does not pass --expect "$EXPECTED_RESULTS"']
    raw = str((step.get("env") or {}).get("EXPECTED_RESULTS", ""))
    given = {part.strip() for part in raw.split(",") if part.strip()}
    gaps = [f"not expected: {entry}" for entry in sorted(wanted - given)]
    gaps += [f"expected but no such leg: {entry}" for entry in sorted(given - wanted)]
    return gaps


def jobs_without_timeout(workflow: dict) -> list[str]:
    """Rule 12: jobs that would run for GitHub's six-hour default."""
    return sorted(
        job_id
        for job_id, job in _jobs(workflow).items()
        if not job.get("timeout-minutes")
    )


def _ignores_majors(rule: Any, distribution: str) -> bool:
    """True when one ``ignore`` entry stops major updates of ``distribution``.

    An entry with neither ``versions`` nor ``update-types`` ignores every
    update. A ``versions`` range alone is not accepted: whether it covers the
    next major cannot be read from the range here.
    """
    if not isinstance(rule, dict):
        return False
    pattern = str(rule.get("dependency-name", "")).lower()
    if not fnmatch.fnmatchcase(distribution, pattern):
        return False
    update_types = rule.get("update-types")
    if update_types is None and rule.get("versions") is None:
        return True
    return SEMVER_MAJOR in (update_types or [])


def dependabot_mcp_major_gaps(config: dict) -> list[str]:
    """Rule 8: Python ecosystem entries that would propose an mcp major.

    No Python entry at all is a gap as well: once security updates are
    enabled they run for the repository with no ignore rule applied.
    """
    entries = [
        entry
        for entry in config.get("updates") or []
        if entry.get("package-ecosystem") in PYTHON_ECOSYSTEMS
    ]
    if not entries:
        return ["no pip or uv entry"]
    gaps = []
    for entry in entries:
        where = (
            f"{entry.get('package-ecosystem')} "
            f"{entry.get('directory') or entry.get('directories')}"
        )
        for distribution in MCP_DISTRIBUTIONS:
            if not any(
                _ignores_majors(rule, distribution)
                for rule in entry.get("ignore") or []
            ):
                gaps.append(f"{where}: {distribution}")
    return gaps


# ---------------------------------------------------------------------------
# Real files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflows():
    return load_workflows()


@pytest.fixture(scope="module")
def audit(workflows):
    assert AUDIT_WORKFLOW in workflows, f"{AUDIT_WORKFLOW} is missing"
    return workflows[AUDIT_WORKFLOW]


@pytest.fixture(scope="module")
def dependabot():
    return yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))


def test_the_real_files_actually_parsed(workflows, audit, dependabot):
    """Anchor for the rest: "no violations" has to mean something."""
    assert len(workflows) >= 4, f"only {sorted(workflows)} parsed"
    for name, workflow in workflows.items():
        assert isinstance(workflow, dict) and _jobs(workflow), f"{name}: no jobs"
        assert triggers(workflow), f"{name}: no triggers read (the `on` key)"
    assert len(_jobs(audit)) >= 5, f"audit.yml has only {sorted(_jobs(audit))}"
    assert dependabot.get("updates"), "dependabot.yml parsed with no updates"


def test_every_workflow_declares_permissions(workflows):
    undeclared = {
        name: jobs
        for name, workflow in workflows.items()
        if (jobs := undeclared_permissions(workflow))
    }
    assert not undeclared, (
        f"these jobs get the repository's default token scopes: {undeclared}. "
        f"Declare `permissions:` at the top of the workflow or on each job."
    )


def _all_write_grants(workflows) -> set:
    found: set = set()
    for name, workflow in workflows.items():
        found |= write_grants(name, workflow)
    return found


def test_write_scopes_come_only_from_the_allowlist(workflows):
    unexpected = _all_write_grants(workflows) - ALLOWED_WRITE_SCOPES
    assert (
        not unexpected
    ), f"write scopes outside ALLOWED_WRITE_SCOPES: {sorted(unexpected)}"


def test_the_write_allowlist_has_no_stale_entries(workflows):
    """An excuse for a grant that no longer exists excuses nothing."""
    gone = ALLOWED_WRITE_SCOPES - _all_write_grants(workflows)
    assert not gone, f"ALLOWED_WRITE_SCOPES outlived what it excused: {sorted(gone)}"


def test_no_workflow_uses_write_all(workflows):
    found = {name: where for name, wf in workflows.items() if (where := write_all(wf))}
    assert not found, f"write-all granted: {found}"


def test_the_audit_runs_weekly_and_on_demand_without_write_scopes(audit):
    assert audit_policy_violations(audit) == []


def test_the_audit_references_only_the_default_token(audit):
    assert foreign_secrets(audit) == []


def test_the_audit_names_no_ai_provider(audit):
    assert ai_provider_names(audit) == []


def test_every_expected_weekly_check_uploads_an_artifact(audit):
    assert checks_without_artifacts(audit, results.EXPECTED_WEEKLY_CHECKS) == []


def test_the_known_issue_fetch_uses_the_report_builder_limit(audit):
    limits = known_issue_fetch_limits(audit)
    assert limits and set(limits) == {results.KNOWN_ISSUES_LIMIT}, limits


def test_the_known_issue_fetch_reads_only_accepted_issues(audit):
    assert known_issue_trust_gaps(audit, report.ACCEPTED_LABEL) == []


def test_no_issue_template_applies_the_acceptance_label():
    templates = {
        path.name: template_labels(path.read_text(encoding="utf-8"), path.suffix)
        for path in sorted(ISSUE_TEMPLATE_DIR.iterdir())
        if path.suffix in (".md", ".yml", ".yaml")
    }
    # Anchor: the audit form's own label is read, so "none apply it" is real.
    assert templates["audit_finding.yml"] == ["audit"]
    assert templates_applying(templates, report.ACCEPTED_LABEL) == []


def test_the_report_expects_every_matrix_leg(audit):
    assert expected_results_gaps(audit) == []


def test_every_audit_job_has_a_timeout(audit):
    assert jobs_without_timeout(audit) == []


def test_dependabot_never_proposes_an_mcp_major(dependabot):
    assert dependabot_mcp_major_gaps(dependabot) == []


def test_the_audit_suite_matrix_covers_floors_and_newest(audit):
    """The floors leg runs on 3.11 and newest legs run on 3.11, 3.12 and 3.14."""
    suite = _jobs(audit)["suite"]
    legs = {
        (str(leg.get("leg")), str(leg.get("python")))
        for leg in suite["strategy"]["matrix"]["include"]
    }
    required = {
        ("floors", "3.11"),
        ("newest", "3.11"),
        ("newest", "3.12"),
        ("newest", "3.14"),
    }
    assert required <= legs, f"missing suite legs: {sorted(required - legs)}"
    script = "\n".join(step.get("run", "") for _, step in _steps(suite))
    # A floors leg that lost its constraints would resolve the newest
    # releases and still be reported as "floors".
    assert "python -m tools.audit floors" in script
    assert "pip install -c" in script
    assert "--junitxml" in script
    # A leg that fails only on the coverage gate leaves a clean JUnit file;
    # the report reads pytest's recorded exit status to see it.
    assert "pytest-exit-code.txt" in script
    assert fnmatch.fnmatch("pytest-exit-code.txt", report.PYTEST_EXIT_PATTERN)


# ---------------------------------------------------------------------------
# Red paths, none of which rest on the real files
# ---------------------------------------------------------------------------


def _doc(text: str) -> dict:
    return yaml.safe_load(text)


class TestTriggers:
    def test_the_bare_on_key_is_read(self):
        """YAML 1.1 turns `on` into True; the events must still be found."""
        assert set(triggers(_doc("on: [push, pull_request]\njobs: {}\n"))) == {
            "push",
            "pull_request",
        }

    def test_a_single_event_string_is_read(self):
        assert set(triggers(_doc("on: push\njobs: {}\n"))) == {"push"}

    def test_a_bare_workflow_dispatch_is_present(self):
        """`workflow_dispatch:` with no inputs parses to None, not absent."""
        doc = _doc("on:\n  workflow_dispatch:\njobs: {}\n")
        assert "workflow_dispatch" in triggers(doc)


NO_PERMISSIONS = """\
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""

ONE_JOB_UNDECLARED = """\
on: push
jobs:
  declared:
    permissions:
      contents: read
    steps:
      - run: echo hi
  forgotten:
    steps:
      - run: echo hi
"""

EVERY_JOB_DECLARES = """\
on: push
jobs:
  publish:
    permissions:
      id-token: write
      contents: read
    steps:
      - run: echo hi
"""

NULL_PERMISSIONS = """\
on: push
permissions:
jobs:
  build:
    steps:
      - run: echo hi
"""


class TestUndeclaredPermissions:
    def test_a_workflow_without_any_permissions_is_reported(self):
        assert undeclared_permissions(_doc(NO_PERMISSIONS)) == ["build"]

    def test_one_job_without_permissions_is_reported(self):
        assert undeclared_permissions(_doc(ONE_JOB_UNDECLARED)) == ["forgotten"]

    def test_job_level_permissions_on_every_job_suffice(self):
        assert undeclared_permissions(_doc(EVERY_JOB_DECLARES)) == []

    def test_an_empty_permissions_key_is_not_a_declaration(self):
        assert undeclared_permissions(_doc(NULL_PERMISSIONS)) == ["build"]


CONTENTS_WRITE = """\
on: push
permissions:
  contents: read
jobs:
  test:
    permissions:
      contents: write
    steps:
      - run: echo hi
"""

PUBLISH_SHAPE = """\
on: push
permissions:
  contents: read
jobs:
  build:
    steps:
      - run: python -m build
  publish-pypi:
    permissions:
      id-token: write
    steps:
      - run: echo publish
"""

ID_TOKEN_MOVED_TO_BUILD = """\
on: push
permissions:
  contents: read
jobs:
  build:
    permissions:
      id-token: write
    steps:
      - run: python -m build
"""

ID_TOKEN_MOVED_TO_WORKFLOW = """\
on: push
permissions:
  id-token: write
jobs:
  publish-pypi:
    steps:
      - run: echo publish
"""

UNKNOWN_ACCESS_LEVEL = """\
on: push
permissions:
  contents: Write
  packages: admin
  actions: read
  checks: none
jobs:
  test:
    steps:
      - run: echo hi
"""


class TestWriteGrants:
    def test_a_contents_write_is_found(self):
        assert write_grants("tests.yml", _doc(CONTENTS_WRITE)) == {
            ("tests.yml", "test", "contents")
        }

    def test_the_allowlisted_publish_grant_is_found_and_excused(self):
        grants = write_grants("publish.yml", _doc(PUBLISH_SHAPE))
        assert grants == {("publish.yml", "publish-pypi", "id-token")}
        assert grants <= ALLOWED_WRITE_SCOPES

    def test_the_same_grant_on_another_job_is_not_excused(self):
        grants = write_grants("publish.yml", _doc(ID_TOKEN_MOVED_TO_BUILD))
        assert grants - ALLOWED_WRITE_SCOPES == {("publish.yml", "build", "id-token")}

    def test_the_same_grant_on_the_whole_workflow_is_not_excused(self):
        grants = write_grants("publish.yml", _doc(ID_TOKEN_MOVED_TO_WORKFLOW))
        assert grants - ALLOWED_WRITE_SCOPES == {
            ("publish.yml", WORKFLOW_LEVEL, "id-token")
        }

    def test_the_same_grant_in_another_file_is_not_excused(self):
        grants = write_grants("audit.yml", _doc(PUBLISH_SHAPE))
        assert grants - ALLOWED_WRITE_SCOPES == {
            ("audit.yml", "publish-pypi", "id-token")
        }

    def test_an_unrecognized_access_level_counts_as_write(self):
        """Only `read` and `none` are known to be harmless."""
        assert write_grants("x.yml", _doc(UNKNOWN_ACCESS_LEVEL)) == {
            ("x.yml", WORKFLOW_LEVEL, "contents"),
            ("x.yml", WORKFLOW_LEVEL, "packages"),
        }


WRITE_ALL = """\
on: push
permissions: write-all
jobs:
  test:
    permissions: write-all
    steps:
      - run: echo hi
  lint:
    permissions: read-all
    steps:
      - run: echo hi
"""


class TestWriteAll:
    def test_write_all_is_found_at_both_levels(self):
        assert write_all(_doc(WRITE_ALL)) == sorted([WORKFLOW_LEVEL, "test"])

    def test_write_all_is_also_a_write_grant(self):
        assert write_grants("x.yml", _doc(WRITE_ALL)) == {
            ("x.yml", WORKFLOW_LEVEL, "write-all"),
            ("x.yml", "test", "write-all"),
        }


AUDIT_ISSUES_WRITE = """\
on:
  schedule:
    - cron: "17 5 * * 1"
  workflow_dispatch:
permissions:
  contents: read
jobs:
  report:
    permissions:
      contents: read
      issues: write
    steps:
      - run: python -m tools.audit report
"""

AUDIT_NO_SCHEDULE = """\
on:
  workflow_dispatch:
  pull_request:
permissions:
  contents: read
jobs:
  report:
    steps:
      - run: python -m tools.audit report
"""

AUDIT_NO_DISPATCH = """\
on:
  schedule:
    - cron: "17 5 * * 1"
permissions:
  contents: read
jobs:
  report:
    steps:
      - run: python -m tools.audit report
"""

AUDIT_WRITE_ALL = """\
on:
  schedule:
    - cron: "17 5 * * 1"
  workflow_dispatch:
permissions: write-all
jobs:
  report:
    steps:
      - run: python -m tools.audit report
"""


class TestAuditPolicy:
    def test_issues_write_on_the_report_job_is_reported(self):
        assert audit_policy_violations(_doc(AUDIT_ISSUES_WRITE)) == [
            "report: issues: write"
        ]

    def test_issues_write_is_also_outside_the_write_allowlist(self):
        grants = write_grants(AUDIT_WORKFLOW, _doc(AUDIT_ISSUES_WRITE))
        assert grants - ALLOWED_WRITE_SCOPES == {(AUDIT_WORKFLOW, "report", "issues")}

    def test_a_missing_schedule_is_reported(self):
        assert audit_policy_violations(_doc(AUDIT_NO_SCHEDULE)) == [
            "no schedule trigger"
        ]

    def test_a_missing_manual_trigger_is_reported(self):
        assert audit_policy_violations(_doc(AUDIT_NO_DISPATCH)) == [
            "no workflow_dispatch trigger"
        ]

    def test_write_all_grants_every_forbidden_scope(self):
        assert audit_policy_violations(_doc(AUDIT_WRITE_ALL)) == [
            f"{WORKFLOW_LEVEL}: {scope}: write" for scope in AUDIT_FORBIDDEN_WRITES
        ]


PROVIDER_SECRET = """\
on: workflow_dispatch
jobs:
  report:
    steps:
      - name: Summarize
        run: ./summarize
        env:
          API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
"""

DEFAULT_TOKEN_ONLY = """\
on: workflow_dispatch
jobs:
  report:
    steps:
      - run: gh issue list
        env:
          GH_TOKEN: ${{ github.token }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          OTHER: ${{ secrets.github_token }}
"""

OTHER_SECRET_SPELLINGS = """\
on: workflow_dispatch
jobs:
  report:
    if: secrets.DEPLOY_KEY != ''
    steps:
      - run: echo "${{ secrets['OPENAI_API_KEY'] }}"
      - run: echo '${{ toJSON(secrets) }}'
"""


class TestForeignSecrets:
    def test_a_provider_secret_is_reported(self):
        assert foreign_secrets(_doc(PROVIDER_SECRET)) == ["secrets.ANTHROPIC_API_KEY"]

    def test_the_default_token_is_allowed_in_every_spelling(self):
        assert foreign_secrets(_doc(DEFAULT_TOKEN_ONLY)) == []

    def test_bracket_whole_context_and_bare_if_references_are_reported(self):
        assert foreign_secrets(_doc(OTHER_SECRET_SPELLINGS)) == sorted(
            ["secrets.DEPLOY_KEY", "secrets['OPENAI_API_KEY']", "secrets"]
        )

    def test_the_word_outside_an_expression_is_not_a_reference(self):
        doc = _doc("on: push\njobs:\n  a:\n    steps:\n      - name: no secrets here\n")
        assert foreign_secrets(doc) == []


LITERAL_PROVIDER_KEY = """\
on: workflow_dispatch
jobs:
  report:
    steps:
      - name: Summarize
        run: ./summarize
        env:
          OPENAI_API_KEY: sk-not-a-real-key
"""

PROVIDER_NAMES_ELSEWHERE = """\
on:
  workflow_dispatch:
    inputs:
      claude_model:
        type: string
env:
  GEMINI_KEY: x
jobs:
  report:
    env:
      CODEX_HOME: /tmp
    steps:
      - uses: some/action@v1
        with:
          anthropic_api_key: x
"""

CLEAN_NAMES = """\
on: workflow_dispatch
env:
  PYTHONUTF8: "1"
jobs:
  upstream:
    steps:
      - run: python -m tools.audit upstream
        env:
          GITHUB_TOKEN: ${{ github.token }}
          GH_TOKEN: ${{ github.token }}
"""


class TestAiProviderNames:
    def test_a_literal_provider_key_in_env_is_reported(self):
        assert ai_provider_names(_doc(LITERAL_PROVIDER_KEY)) == [
            "report/Summarize env OPENAI_API_KEY"
        ]

    def test_inputs_workflow_job_env_and_action_inputs_are_swept(self):
        assert ai_provider_names(_doc(PROVIDER_NAMES_ELSEWHERE)) == sorted(
            [
                "on.workflow_dispatch.inputs claude_model",
                f"{WORKFLOW_LEVEL} env GEMINI_KEY",
                "report env CODEX_HOME",
                "report/some/action@v1 with anthropic_api_key",
            ]
        )

    def test_ordinary_names_pass(self):
        assert ai_provider_names(_doc(CLEAN_NAMES)) == []


NO_SECURITY_UPLOAD = """\
on: workflow_dispatch
jobs:
  suite:
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: suite-${{ matrix.leg }}-py${{ matrix.python }}
          path: audit-results/
  package-smoke:
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: package-smoke-${{ matrix.variant }}
          path: audit-results/
  upstream:
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: upstream
          path: audit-results/
  security:
    steps:
      - run: python -m tools.audit security --lock a.json --fresh b.json
      - uses: actions/upload-artifact@v7
        with:
          name: security-results
          path: audit-results/
      - uses: actions/upload-artifact@v7
        with:
          path: audit-results/
"""


class TestChecksWithoutArtifacts:
    def test_a_dropped_security_upload_is_reported_against_the_constant(self):
        """Renamed and unnamed uploads do not count as the security artifact."""
        doc = _doc(NO_SECURITY_UPLOAD)
        assert checks_without_artifacts(doc, results.EXPECTED_WEEKLY_CHECKS) == [
            "security"
        ]

    def test_the_uploads_are_found(self):
        assert uploaded_artifacts(_doc(NO_SECURITY_UPLOAD)) == {
            "suite-${{ matrix.leg }}-py${{ matrix.python }}",
            "package-smoke-${{ matrix.variant }}",
            "upstream",
            "security-results",
        }

    def test_an_expected_check_this_test_cannot_map_is_reported(self):
        """A check added to the constant must be wired here, not skipped."""
        assert checks_without_artifacts(_doc(NO_SECURITY_UPLOAD), ("smoke",)) == [
            "smoke (no artifact mapping in CHECK_ARTIFACTS)"
        ]


ISSUE_FETCHES = """\
on: workflow_dispatch
jobs:
  report:
    steps:
      - run: |
          gh issue list --state all \\
            --json number,body > a.json
          gh issue list --limit 500 --json number > b.json
          echo --limit 1000
"""


class TestKnownIssueFetchLimits:
    def test_a_missing_or_smaller_limit_is_read_per_command(self):
        """The default (30) and a limit that drifted both differ from the constant."""
        assert known_issue_fetch_limits(_doc(ISSUE_FETCHES)) == [None, 500]


TRUSTED_FETCH = """\
on: workflow_dispatch
jobs:
  report:
    steps:
      - name: fetch
        run: |
          gh issue list --state all --label audit-accepted --limit 1000 \\
            --json number,title,state,stateReason,body,labels > a.json
"""


class TestKnownIssueTrust:
    def test_a_label_filtered_fetch_with_labels_passes(self):
        assert known_issue_trust_gaps(_doc(TRUSTED_FETCH), "audit-accepted") == []

    def test_an_unfiltered_fetch_and_one_without_labels_are_reported(self):
        """The fetches of ISSUE_FETCHES read every issue, without labels."""
        gaps = known_issue_trust_gaps(_doc(ISSUE_FETCHES), "audit-accepted")
        assert gaps == [
            "report/step 0: no --label filter",
            "report/step 0: --json without labels",
            "report/step 0: no --label filter",
            "report/step 0: --json without labels",
            "no fetch filters on --label audit-accepted",
        ]

    def test_a_search_fetch_is_reported(self):
        doc = _doc(
            TRUSTED_FETCH.replace("--label audit-accepted", '--search "pmm-audit"')
        )
        assert known_issue_trust_gaps(doc, "audit-accepted") == [
            "report/fetch: no --label filter",
            "no fetch filters on --label audit-accepted",
        ]

    def test_a_second_fetch_of_another_label_is_allowed_beside_the_accepted_one(self):
        second = (
            "          gh issue list --state all --label audit --limit 1000 \\\n"
            "            --json number,title,state,stateReason,body,labels > b.json\n"
        )
        doc = _doc(TRUSTED_FETCH + second)
        assert known_issue_trust_gaps(doc, "audit-accepted") == []
        only_form = _doc(TRUSTED_FETCH.replace("audit-accepted", "audit"))
        assert known_issue_trust_gaps(only_form, "audit-accepted") == [
            "no fetch filters on --label audit-accepted"
        ]

    def test_a_form_or_front_matter_applying_the_label_is_reported(self):
        form = template_labels('name: x\nlabels: ["audit", "Audit-Accepted"]\n', ".yml")
        markdown = template_labels(
            "---\nname: x\nlabels: bug, audit-accepted\n---\nbody\n", ".md"
        )
        plain = template_labels("---\nname: x\nlabels: bug\n---\n", ".md")
        templates = {"form.yml": form, "bug.md": markdown, "plain.md": plain}
        assert templates_applying(templates, "audit-accepted") == ["bug.md", "form.yml"]

    def test_a_markdown_template_without_front_matter_applies_nothing(self):
        assert template_labels("labels: audit-accepted\n", ".md") == []


EXPECT_WORKFLOW = """\
on: workflow_dispatch
jobs:
  suite:
    strategy:
      matrix:
        include:
          - {leg: floors, python: "3.11"}
          - {leg: newest, python: "3.12"}
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: suite-${{ matrix.leg }}-py${{ matrix.python }}
  package-smoke:
    strategy:
      matrix:
        include:
          - {variant: wheel-ubuntu-py3.11}
          - {variant: sdist-ubuntu-py3.12}
  report:
    steps:
      - env:
          EXPECTED_RESULTS: >-
            suite:suite-floors-py3.11,
            suite:suite-newest-py3.12,
            package-smoke:wheel-ubuntu-py3.11,
            package-smoke:sdist-ubuntu-py3.12,
            upstream,
            security
        run: python -m tools.audit report --expect "$EXPECTED_RESULTS"
"""


class TestExpectedResults:
    def test_every_leg_expected_passes(self):
        assert expected_results_gaps(_doc(EXPECT_WORKFLOW)) == []

    def test_a_leg_added_to_a_matrix_but_not_expected_is_reported(self):
        doc = _doc(
            EXPECT_WORKFLOW.replace(
                '- {leg: newest, python: "3.12"}',
                '- {leg: newest, python: "3.12"}\n          - {leg: newest, python: "3.14"}',
            )
        )
        assert expected_results_gaps(doc) == ["not expected: suite:suite-newest-py3.14"]

    def test_a_stale_expectation_is_reported(self):
        doc = _doc(EXPECT_WORKFLOW.replace("- {variant: sdist-ubuntu-py3.12}", ""))
        assert expected_results_gaps(doc) == [
            "expected but no such leg: package-smoke:sdist-ubuntu-py3.12"
        ]

    def test_a_report_step_without_the_expectation_is_reported(self):
        doc = _doc(EXPECT_WORKFLOW.replace(' --expect "$EXPECTED_RESULTS"', ""))
        assert expected_results_gaps(doc) == [
            'the report step does not pass --expect "$EXPECTED_RESULTS"'
        ]


class TestTimeouts:
    def test_a_job_without_a_timeout_is_reported(self):
        doc = _doc(
            "on: workflow_dispatch\njobs:\n  a:\n    timeout-minutes: 5\n"
            "    steps: [{run: x}]\n  b:\n    steps: [{run: x}]\n"
        )
        assert jobs_without_timeout(doc) == ["b"]


def _dependabot(*entries: str) -> dict:
    return _doc("version: 2\nupdates:\n" + "".join(entries))


UV_HEAD = """\
  - package-ecosystem: uv
    directory: "/"
    schedule:
      interval: weekly
"""

PIP_HEAD = UV_HEAD.replace("uv", "pip")

IGNORE_BOTH = """\
    ignore:
      - dependency-name: mcp
        update-types: ["version-update:semver-major"]
      - dependency-name: mcp-types
        update-types: ["version-update:semver-major"]
"""

IGNORE_ONLY_MCP_TYPES = """\
    ignore:
      - dependency-name: mcp-types
        update-types: ["version-update:semver-major"]
"""

IGNORE_ONLY_MCP = """\
    ignore:
      - dependency-name: mcp
        update-types: ["version-update:semver-major"]
"""

IGNORE_WRONG_SHAPES = """\
    ignore:
      - dependency-name: mcp
        update-types: ["version-update:semver-minor"]
      - dependency-name: mcp-types
        versions: ["2.4.x"]
"""

IGNORE_WILDCARD = """\
    ignore:
      - dependency-name: "mcp*"
        update-types: ["version-update:semver-major"]
"""

GITHUB_ACTIONS = """\
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
"""


class TestDependabotMcpMajor:
    def test_a_missing_mcp_ignore_is_reported(self):
        config = _dependabot(UV_HEAD + IGNORE_ONLY_MCP_TYPES)
        assert dependabot_mcp_major_gaps(config) == ["uv /: mcp"]

    def test_a_missing_mcp_types_ignore_alone_is_reported(self):
        config = _dependabot(UV_HEAD + IGNORE_ONLY_MCP)
        assert dependabot_mcp_major_gaps(config) == ["uv /: mcp-types"]

    def test_every_python_entry_is_checked(self):
        config = _dependabot(UV_HEAD + IGNORE_BOTH, PIP_HEAD)
        assert dependabot_mcp_major_gaps(config) == ["pip /: mcp", "pip /: mcp-types"]

    def test_a_minor_ignore_or_a_version_range_does_not_cover_majors(self):
        config = _dependabot(UV_HEAD + IGNORE_WRONG_SHAPES)
        assert dependabot_mcp_major_gaps(config) == ["uv /: mcp", "uv /: mcp-types"]

    def test_a_wildcard_covering_both_names_is_accepted(self):
        config = _dependabot(UV_HEAD + IGNORE_WILDCARD)
        assert dependabot_mcp_major_gaps(config) == []

    def test_an_ignore_of_every_update_is_accepted(self):
        config = _dependabot(
            UV_HEAD + "    ignore:\n      - dependency-name: mcp\n"
            "      - dependency-name: mcp-types\n"
        )
        assert dependabot_mcp_major_gaps(config) == []

    def test_no_python_entry_at_all_is_reported(self):
        """Security updates would then run with no mcp ignore rule."""
        assert dependabot_mcp_major_gaps(_dependabot(GITHUB_ACTIONS)) == [
            "no pip or uv entry"
        ]
