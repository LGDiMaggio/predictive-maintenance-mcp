---
title: "Editable installs pin one checkout, and every worktree silently inherits it"
date: 2026-08-08
category: architecture-patterns
module: predictive-maintenance-mcp
problem_type: architecture_pattern
component: tests
severity: high
applies_when:
  - "A repository is worked on from more than one git worktree sharing a single virtualenv"
  - "A package is installed with `pip install -e .` and imported by its distribution name rather than a relative path"
  - "pyproject maps a package name onto a differently-named directory via [tool.setuptools.package-dir]"
  - "A test suite's correctness depends on WHICH copy of the source it imported, and nothing checks"
  - "A `sys.path.insert` exists in conftest and is assumed to control import resolution"
  - "Choosing between an environment-level fix and an in-process one under a filesystem-cost constraint"
tags: [editable-install, git-worktree, import-resolution, sys-meta-path, conftest, silent-failure, tripwire, onedrive]
---

# Editable installs pin one checkout, and every worktree silently inherits it

## Context

Claude Code creates a git worktree per agent session under `.claude/worktrees/`.
Every one of them shared the primary checkout's `.venv`. Running `pytest` in a
worktree therefore tested the **primary checkout's** source, not the code the
session had just written.

Nothing said so. The suite ran, went green, and reported numbers about a tree
nobody was editing.

It cost four separate sessions before anyone named the cause:

| Session | Symptom |
|---|---|
| mcp 2.x migration | Baseline run reported 34 registered tools; the worktree had 33. Time spent chasing a phantom regression before finding the cause. |
| Reliability review | Reported "package version 0.8.0" — a version present in no file in the worktree. Several dynamic findings had to be discarded. |
| Two later review agents | Same resolution confusion, rediscovered independently. |

## The mechanism

`pip install -e .` writes two files into site-packages: a `.pth` that runs at
interpreter start, and a finder module holding a hardcoded map.

```python
# __editable___predictive_maintenance_mcp_0_11_0_finder.py
MAPPING = {'predictive_maintenance_mcp': 'C:\\...\\predictive-maintenance-mcp\\src'}
```

One absolute path — the checkout that happened to run the install. Note the
`0_11_0` in the filename while `pyproject.toml` said `0.12.0`: the finder is not
just wrong-tree, it goes stale, which is what produced the phantom version
reports.

Observed from inside a worktree:

```
pkg file : ...\predictive-maintenance-mcp\src\__init__.py        <- primary, not here
meta_path: [..., PathFinder, _EditableFinder, _SixMetaPathImporter]
```

setuptools **appends** its finder to `sys.meta_path`, so it sits behind
`PathFinder`. Anything resolvable through `sys.path` wins. That detail is what
makes the fix cheap — and it is also what made the trap so convincing.

## Why the obvious fix was not a fix

`tests/conftest.py` opened with:

```python
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

That line looks exactly like the guard everyone assumed was already in place. It
did nothing, for two independent reasons:

1. All 44 test files import `predictive_maintenance_mcp.<something>`. Putting
   `src/` on `sys.path` exposes `server`, `models`, `rag` as **top-level** names.
   It never binds the package name — and there is no directory literally named
   `predictive_maintenance_mcp` anywhere in the tree, because pyproject maps the
   name onto `src/` via `[tool.setuptools.package-dir]`.
2. Every intra-`src` import is relative (`from ..models import`, `from .config
   import`), so nothing inside the package resolved through that entry either.

**Generalisation:** a `sys.path` entry only helps if some directory on it is
*named* like the package being imported. When a build backend renames a
directory into a package, `sys.path` manipulation cannot reproduce that rename.
A dead guard is worse than no guard: it stops people looking.

## The fix

Two layers, split by what each costs.

### Default: bind the name in-process (free)

`tests/conftest.py` installs a finder at position **0** of `sys.meta_path`,
ahead of both `PathFinder` and setuptools' appended finder:

```python
class _LocalTreeFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname != "predictive_maintenance_mcp":
            return None
        return importlib.util.spec_from_file_location(
            fullname, SRC_DIR / "__init__.py",
            submodule_search_locations=[str(SRC_DIR)],
        )
```

Sub-packages resolve through the parent's `__path__`, so only the top-level name
needs claiming. In CI and in the primary checkout the editable install already
points at the same files, so this is a no-op there; it diverges only where the
old behaviour was wrong.

### The tripwire (the part that mattered most)

The pin makes the bug go away. The tripwire makes it *impossible to have again
without noticing* — which is the actual lesson, because the original failure was
silent, not hard.

`conftest.py` raises at collection if `predictive_maintenance_mcp.__file__` is
not under the tree the conftest lives in, naming both paths and the fix.
`tests/test_import_provenance.py` re-asserts it as named tests so the guarantee
survives someone tidying the conftest, and adds the drift checks
(`__version__` vs `src/__init__.py` vs `pyproject.toml`) that catch a stale
install whose path happens to be right.

Both directions were verified rather than assumed:

```
$ pytest tests/test_import_provenance.py              -> 7 passed
$ PMM_TEST_ALLOW_INSTALLED=1 pytest ...               -> 2 failed
    package imported from ...\predictive-maintenance-mcp\src\__init__.py,
    outside this checkout (...\worktrees\vigilant-cartwright-85fd5e)
```

The second run is the point: the underlying environment is still broken. The pin
is doing the work, and the escape hatch proves it rather than hiding it.

### Opt-in: a real environment per worktree

`scripts/setup-worktree.py` — `uv sync --extra dev --frozen` when `uv` is on
PATH, `python -m venv` + `pip install -e ".[dev]"` otherwise, then it verifies
the resulting interpreter imports this checkout before declaring success.

It is **not** automatic, and that is a deliberate trade rather than laziness.
The environment measures 528 MB across 18,434 files, the repository lives on
OneDrive, and there were four live worktrees. Making it the default would have
meant ~2 GB and ~74k files for the sync client to churn through — to fix a
problem the free layer already fixes for the 95% case (running tests). A single
`du` of one venv took over two minutes, which is the cost being avoided.

So the split is: the free layer covers `pytest` everywhere, automatically; the
expensive layer is one documented command, for when you need to *run* the
checkout (server, REPL, `validate_server.py`) rather than test it.

## Rejected: a `.claude/` hook at worktree creation

Tempting, and wrong twice. `.claude/settings.local.json` is gitignored as
per-developer state, so a hook there is not shareable — the next contributor
rediscovers the bug regardless. And it fires at worktree *creation*, whereas the
failure only matters at the moment someone trusts a test result. A tripwire at
the point of use beats setup-time automation that can be skipped, forgotten, or
simply never installed.

## What to carry forward

1. **A test suite that does not verify which source it imported is not evidence.**
   The green run was the failure mode, not the absence of one.
2. **`sys.path` cannot undo a build backend's directory-to-package rename.** If
   you need a specific tree to win, claim the name on `sys.meta_path`.
3. **Prefer a tripwire at the point of use over setup automation.** Setup can be
   skipped; an assertion in the path everyone already runs cannot.
4. **When a guard is disabled by an escape hatch, check that the failure returns.**
   A guard whose absence changes nothing was never guarding anything.
5. **Silent-vs-loud is a property worth spending code on.** The pin saves minutes;
   the tripwire saves the four-sessions-of-confusion that the pin alone would
   have merely postponed to the next environment quirk.
