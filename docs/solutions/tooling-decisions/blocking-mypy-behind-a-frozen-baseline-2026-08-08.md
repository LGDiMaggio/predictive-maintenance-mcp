---
title: "Making a decorative CI check blocking, behind a frozen baseline"
date: 2026-08-08
category: tooling-decisions
module: predictive-maintenance-mcp
problem_type: tooling_decision
component: ci
severity: medium
applies_when:
  - "A CI check runs with continue-on-error and has been reporting green regardless"
  - "Turning a check on would fail immediately on pre-existing debt"
  - "A guard script decides pass/fail and nothing tests the guard itself"
  - "Moving CLI flags into a config file, where they apply to local runs too"
  - "Adding a type-checker plugin whose defaults change what is checked"
tags: [mypy, ci-guards, ratchet, frozen-baseline, fail-closed, pydantic]
---

# Making a decorative CI check blocking, behind a frozen baseline

## Problem


`.github/workflows/tests.yml` runs the mypy job with `continue-on-error: true`. The job
reports green even when mypy fails, so the check is decorative.

That matters more here than in most repos. The test suite substitutes `ctx` with a bare
`AsyncMock` in 42 places across 11 files, so a regression in how the server talks to the
MCP client passes 890/890 tests. During the mcp 2.x migration, `await` was stripped from
116 `ctx.info(...)` call sites on a wrong assumption; the suite stayed green and mypy was
the only signal that caught it (`Value of type "Coroutine[Any, Any, None]" must be used`).

Removing `continue-on-error` alone turns CI red immediately on pre-existing debt.

A second, quieter hole: the job passes `--ignore-missing-imports`. If `mcp` ever fails to
resolve in that job, `Context` degrades to `Any` and every `ctx` call site type-checks
vacuously — the same hole reopens without anyone noticing.

## Findings that shaped the design

**The regression class has no surface area on main today.** Commit `f7e339d` (v0.12.0,
"narrate progress on the stdlib logger, not to the client") replaced every client-facing
`ctx.*` call with stdlib `logger.*`:

| | `ctx.` call sites in `src/` |
|---|---|
| before `f7e339d` | 119, across 6 files in `src/mcp_tools/` |
| HEAD | 0 |

`ctx` survives 66 times, but only as an uncalled parameter annotation (`ctx: Context`).
This rules out scoping the blocking check to `src/mcp_tools/**` on the rationale that it
is "where every `Context` call lives" — that is no longer true. It also fails on
mechanics: 28 of the 53 errors are already inside `src/mcp_tools/`.

**Both technical claims verified directly.** Synthetic probes against the installed `mcp`:

```
A. mcp resolves:      error: Value of type "Coroutine[Any, Any, None]" must be used  [unused-coroutine]
                      note: Are you missing an await?
B. mcp unresolvable
   + --ignore-missing-imports:   Success: no issues found in 1 source file
```

**`--ignore-missing-imports` cannot simply be dropped.** Without it, ~30 import errors
appear — but `mcp` is not among them. They are all stub-less packages (`scipy`, `pandas`,
`sklearn`, `plotly`) or genuinely-uninstalled optional extras (`docx`, `weasyprint`,
`faiss`, `sentence_transformers`, `pytesseract`, `pdf2image`).

**The debt is much smaller than 53.** One config line, `plugins = ["pydantic.mypy"]`,
takes it to 35 — mypy currently does not understand `Field(default)` and mis-reads `**dict`
splats into models:

| | 53 → 35 |
|---|---|
| `prognostics_tools.py` | 17 → 1 |
| `diagnostics_tools.py` | 5 → 3 (both "Missing named argument `status`" were false positives) |
| `src/mcp_tools/` total | 28 → 10 |

**A second detector already exists.** `pyproject.toml` sets
`filterwarnings = ["error:coroutine.*was never awaited:RuntimeWarning"]`. Unlike mypy, that
one is AsyncMock-proof — it fires at runtime regardless of mocking.

## Design

### 1. Consolidate mypy config into `pyproject.toml`

CI's CLI flags override the `[tool.mypy]` config, so local and CI runs disagree today. Move
them into config:

| Current CLI flag | Becomes |
|---|---|
| `--allow-untyped-defs` | dropped — already `disallow_untyped_defs = false` |
| `--no-strict-optional` | `strict_optional = false` |
| `--ignore-missing-imports` | per-module `[[tool.mypy.overrides]]` allowlist |
| — | `plugins = ["pydantic.mypy"]` (new) |

The allowlist names exactly the stub-less and optional packages. **`mcp` is deliberately
absent**, with a comment saying why. That single omission converts a vacuous pass into CI
red: breaking the `mcp` import now produces `import-not-found` rather than `Success`.

### 2. Baseline the remaining 35

`tools/mypy_baseline.txt` keyed on `file + error-code + count` — no line numbers, so it
survives ordinary code motion. `tools/check_mypy_baseline.py` runs mypy, normalizes, and
compares.

**Asymmetric comparison, deliberately.** CI (ubuntu, py3.11, `[dev]` only) resolves a
different module set than a local dev venv with extras installed. A strict two-way ratchet
would fail on environment drift rather than on real regressions. So:

- new key, or increased count → **fail**
- disappeared errors → **warn and list**, non-fatal, with `--update` to resync

This blocks regressions reliably without CI flapping. It can be tightened to a hard ratchet
later if the environment drift turns out to be a non-issue in practice.

### 3. Fix two real defects rather than baseline them

- `document_reader.py:259` — `List[Dict[str, any]]` uses builtins `any` (the function)
  where `typing.Any` was meant.
- `prognostics_tools.py:232` — dead `# type: ignore[arg-type]`, flagged by the existing
  `warn_unused_ignores = true`.

Both are defects the baseline would otherwise entomb.

### 4. Rewrite the stale dependency comment

`pyproject.toml` currently tells readers that "the ~116 await sites in `src/mcp_tools/`
still work and still need their await". There are none — it sends a future reader hunting
for removed code. The rule it states is still correct and worth keeping, so it becomes
conditional: *if* client narration returns, those calls must be awaited, and here are the
two detectors that enforce it.

### 5. CI

Drop `continue-on-error` from the mypy job and call the check script. The Black job's
`continue-on-error` is out of scope and left untouched.

## Scope note

Black is also `continue-on-error` and red across 63 files on main. That is separate,
larger, and purely cosmetic — explicitly not bundled here.

## Verification

- baseline check passes on clean HEAD
- fails on an injected new error
- fails when `mcp` is unresolvable
- `--update` round-trips
