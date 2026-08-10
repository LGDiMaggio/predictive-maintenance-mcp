---
title: The public repo carries user-facing content only
date: 2026-08-10
category: conventions
module: repository-hygiene
problem_type: convention
component: development_workflow
severity: high
applies_when:
  - "Committing, pushing, or opening a PR on the public repository"
  - "Writing public-facing text (PR bodies, CHANGELOG entries, test names, code comments)"
tags: [git, gitignore, privacy, planning-docs, public-repo]
---

# The public repo carries user-facing content only

## Context

The development process produces internal planning artifacts — brainstorms,
implementation plans, audits, review-residual records. Their content (internal
reasoning, strategy, references to private interactions) is not meant for
repository visitors, and anything pushed becomes part of public history where
removal requires an expensive and imperfect history rewrite.

## Guidance

- The public repo carries what its users and contributors need: source, tests,
  user documentation, CHANGELOG, CI, packaging.
- Internal planning docs are **local-only**, enforced via `.gitignore` (same
  section as `CLAUDE.md`): `docs/brainstorms/`, `docs/plans/`,
  `docs/residual-review-findings/`, `docs/AUDIT-*.md`.
- Public-facing text (PR bodies, CHANGELOG, test/fixture names, comments)
  stays impersonal: no identities or stories of third parties, no links to
  internal planning docs. Motivation wording stays generic ("industrial DAQ
  hardware writes such files"), not anecdotal.
- Before any push: glance at the staged file list; an entry under an internal
  docs directory, or an unexpectedly modified lockfile, is a stop signal.

## Why This Matters

Un-pushing is far costlier than not pushing: scrubbing already-public content
required rewriting a branch's history (dropping commits, amending a CHANGELOG
entry, force-pushing) — and content merged to the default branch remains
visible in old commits without a full history rewrite.

## When to Apply

- Every commit and PR; every regeneration of docs, prompts, or plugin content
  that might quote internal material.

## Related

- `.gitignore` — "Project-internal docs (local only)" section is the
  enforcement point.
