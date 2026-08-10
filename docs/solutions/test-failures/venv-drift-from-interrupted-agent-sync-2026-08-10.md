---
title: Venv drift from an interrupted dependency sync fails unrelated tests
date: 2026-08-10
category: test-failures
module: development-environment
problem_type: test_failure
component: development_workflow
symptoms:
  - "Content-assertion failures in a test module the branch never touched (PDF-rendering tests)"
  - "New mypy [import-untyped] error on a library that was previously clean"
  - "uv.lock unexpectedly modified (+235/-36) with no dependency work in the task"
  - "pip show <pkg> reports Version None; the module imports as a namespace package with __file__ = None"
root_cause: config_error
resolution_type: environment_setup
severity: medium
tags: [uv, venv, optional-extras, skipif, mypy-baseline, environment-drift]
---

# Venv drift from an interrupted dependency sync fails unrelated tests

## Problem

Mid-branch, three PDF-rendering tests started failing and mypy reported a new
`[import-untyped]` error — in modules the branch never touched. The cause was
not the branch: a subagent, killed mid-task, had run a dependency sync that
mutated the venv beyond the committed lockfile.

## Symptoms

- `tests/test_integrated_report.py::TestPdfRendering` failures although no
  report code changed on the branch.
- `uv.lock` modified without any dependency work in scope.
- `tools/check_mypy_baseline.py` flagging `pypdf` as `[import-untyped]`.
- `pip show pypdf` → `Version: None`; `pypdf.__file__` → `None` (half-installed
  package importing as a bare namespace).

## What Didn't Work

- Suspecting the branch's own changes: the failing area was entirely disjoint
  from the diff. Time was lost before comparing installed packages against the
  committed lock.

## Solution

1. `git checkout -- uv.lock` (revert the file — this does NOT restore the venv).
2. Uninstall the optional extra the sync had added (`pip uninstall weasyprint`)
   so `skipif(not HAS_PDF)` tests return to their skipped baseline.
3. Repair the half-installed package at the lock's pinned version:
   `pip install --ignore-installed --no-deps pypdf==<lock version>`
   (`--force-reinstall` fails when the broken dist-info can't be uninstalled;
   `--ignore-installed` is the recovery route).

## Why This Works

`skipif`-gated tests silently change population when optional extras
appear/disappear: a sync with extras makes previously-skipped tests RUN, and
they can fail on latent issues unrelated to the branch. A sync killed
mid-upgrade can leave a package with orphaned metadata — imports "succeed" as
a namespace package while every attribute access and type-stub lookup breaks.
Reverting the lock file alone never restores the environment; installed state
must be reconciled explicitly.

## Prevention

- Agents and subagents run tests through the project venv's interpreter
  directly (`.venv\Scripts\python.exe -m pytest ...`); they do not run
  `uv sync` / `uv run` unless dependency management IS the task.
- Treat an unexpectedly modified lockfile as a red flag, and compare the
  skipped-test COUNT between runs — a skip-count delta means the test
  population changed, not the code.
- Latent finding recorded: with the `pdf` extra installed, the PDF-rendering
  content assertions fail and need attention whenever that extra becomes part
  of the supported environment.

## Related Issues

- None on the tracker; the PDF-extra latent failures are noted in the raw
  ingestion PR's follow-ups.
