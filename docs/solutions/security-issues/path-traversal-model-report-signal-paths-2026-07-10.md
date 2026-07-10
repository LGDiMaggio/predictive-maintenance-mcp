---
title: Path traversal across MCP model, report, and signal file paths
date: 2026-07-10
category: docs/solutions/security-issues
module: mcp_tools file I/O (model / report / signal paths)
problem_type: security_issue
component: tooling
symptoms:
  - "train_anomaly_model(model_name='../../evil') builds its pickle output path from an unvalidated name and writes outside MODELS_DIR — arbitrary file write reachable by any MCP client"
  - "load_signal_data(signal_file='../../secret.csv') resolves and reads a file outside the data directory, returning its contents to the caller"
  - "read_report_metadata(file_name='../../x') opens an arbitrary path, acting as a file existence/size oracle"
  - "A prior containment fix using str.startswith was bypassable via a sibling directory (data/signals_evil passes startswith('data/signals'))"
root_cause: missing_validation
resolution_type: code_fix
severity: critical
related_components:
  - authentication
tags:
  - path-traversal
  - mcp
  - security
  - pickle
  - file-io
  - containment
  - is-relative-to
  - single-source-of-truth
---

# Path traversal across MCP model, report, and signal file paths

## Problem

Every user-controlled filesystem path in the MCP server was concatenated onto a
base directory (`MODELS_DIR`, `REPORTS_DIR`, `DATA_DIR`) without containment, so an
MCP client — including a prompt-injected LLM — could escape the base with `../`
sequences. The most severe was `train_anomaly_model`, which built its pickle
output path from an unvalidated `model_name` and then `pickle.dump`ed to it: an
arbitrary-file-write primitive, and (because the model is later `pickle.load`ed)
an RCE-adjacent one. The signal read path (`load_signal_data`) and report metadata
read (`read_report_metadata`) were equally uncontained, yielding arbitrary file
reads and existence oracles.

## Symptoms

- `train_anomaly_model(model_name="../../evil")` writes `../../evil_model.pkl` outside `MODELS_DIR`.
- `predict_anomalies` / the PCA report / the diagnosis pipeline load pickles from an attacker-chosen `model_name`.
- `load_signal_data(signal_file="../../secret.csv")` returns the first column of a file outside `DATA_DIR`.
- `read_report_metadata(file_name="../../../etc/passwd")` opens an out-of-bounds path.
- The vulnerability existed in **duplicate** — the modular server (`mcp_tools/`) and the deprecated monolith (`machinery_diagnostics_server.py`) each built the same paths independently.

## What Didn't Work

- **`str.startswith` containment (a prior fix, commit `61627b0`).** `resolved.startswith(str(base))` accepts a *sibling* directory: `/data/signals_evil` starts with `/data/signals`. The correct check is `Path.is_relative_to` (adopted in `d689886`), which compares path components, not string prefixes.
- **Patching one copy.** Fixing only the modular `predict_anomalies` read path left the *write* path (`train_anomaly_model`) and the monolith copies open. Because the path-building logic was duplicated across two servers and several tools, a per-call-site fix is fragile — the class of bug is duplicated-value drift, and it recurs every time a new call site is added.
- **`sanitize_filename` alone as a guard.** `sanitize_filename` silently *rewrites* a bad name (`../../evil` → `evil`), so a caller cannot tell rejection from acceptance, and a name like `..` survives it: `Path("..").name == ".."` on Windows, and a `..` prefix yields a *contained* `.._model.pkl`, so `..`/`.` must be rejected explicitly.
- **Validating outside the `try` block.** In the diagnosis pipeline's defense-in-depth site, calling the (raising) validator before the function's `try/except` turned an invalid name into a crash of the *entire* diagnosis instead of the documented graceful `None`.

## Solution

Introduce **one canonical containment module** and route every path through it.

`src/path_safety.py`:

```python
def safe_resolve(base_dir: Path, user_input: str) -> Path:
    candidate = (Path(base_dir) / user_input).resolve()
    allowed = Path(base_dir).resolve()
    if not candidate.is_relative_to(allowed):   # NOT str.startswith
        raise ValueError(f"Invalid path — escapes base directory: {user_input}")
    return candidate

def validate_name_component(name: str, *, kind: str = "name") -> str:
    """Reject (do not rewrite) anything that is not already a safe component."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid {kind}: must be a non-empty string.")
    if name in (".", ".."):                     # survive the charset check; reject explicitly
        raise ValueError(f"Invalid {kind} '{name}': reserved path name.")
    if sanitize_filename(name) != name:
        raise ValueError(f"Invalid {kind} '{name}'. Use only [A-Za-z0-9_.-].")
    return name

class ModelPaths(NamedTuple):               # typed, not a stringly-keyed dict
    model: Path; scaler: Path; pca: Path; metadata: Path

def resolve_model_paths(models_dir: Path, model_name: str) -> ModelPaths:
    safe = validate_name_component(model_name, kind="model_name")
    return ModelPaths(
        model=safe_resolve(models_dir, f"{safe}_model.pkl"),
        scaler=safe_resolve(models_dir, f"{safe}_scaler.pkl"),
        pca=safe_resolve(models_dir, f"{safe}_pca.pkl"),
        metadata=safe_resolve(models_dir, f"{safe}_metadata.json"),
    )
```

- Every model write/read site (`train_anomaly_model`, `predict_anomalies`, PCA report, diagnosis pipeline) — in **both** the modular server and the monolith — calls `resolve_model_paths`, which validates the name *and* contains all four derived paths in one call.
- `mcp_tools/_utils.py` re-exports the helpers so existing `from ._utils import ...` callers keep working, and a test asserts `_utils.safe_resolve is path_safety.safe_resolve` so a future edit can't silently reintroduce a divergent copy.
- `load_signal_data` resolves the filename with `safe_resolve` *inside* its existing `try` (a traversal → `ValueError` → the function's `except` → `None`, preserving its "return None on failure" contract — content is never read).
- Fail-fast: `validate_name_component(model_name)` runs at the top of `train_anomaly_model`, before any feature extraction or filesystem touch.
- Cross-platform test gotcha: a backslash is a path separator only on Windows. `..\..\x` traverses on Windows but is a harmless literal filename on POSIX, so that assertion must be `@pytest.mark.skipif(os.name != "nt")`, not part of the cross-platform set.

## Why This Works

`Path.is_relative_to` performs component-wise containment on the *resolved* path,
so `../` sequences, absolute paths, UNC paths, and the sibling-directory bypass
all fail the check — `.resolve()` collapses `..` before comparison, and a sibling
like `models_evil` is a different component than `models`. Housing the logic in
one module and routing all sites through `resolve_model_paths` converts a
scattered, drift-prone set of call sites into a single choke point: the
containment check cannot be present at one site and forgotten at another. An
adversarial review (UNC, drive-absolute, `...`, trailing-dot, and Windows device
names) could construct no escape against the model/report paths after the fix.

## Prevention

- **One choke point for every user-controlled path.** Never build `base / user_input` inline; always go through `safe_resolve` / `resolve_model_paths`. Guard it with a re-export identity test so the single source of truth can't fork.
- **Containment via `Path.is_relative_to`, never `str.startswith`** — the sibling-directory bypass is the canonical failure of prefix matching.
- **Validate, don't rewrite, security-relevant names.** A guard that silently sanitizes hides attacks; reject and raise. Explicitly reject `.` and `..` (they survive charset checks and collapse unexpectedly per-OS).
- **Prefer a typed `NamedTuple` over a stringly-keyed dict** for a fixed shape — a typo becomes a type error, not a runtime `KeyError`.
- **Keep a raising validator inside the caller's `try`** when the caller's contract is graceful degradation, so a rejected input degrades instead of crashing the surrounding flow.
- **Fix all copies at once.** When logic is duplicated (here: modular server + legacy monolith), a security fix must land in every copy; the duplicated-value-drift bug class is the repo's most recurrent.
- **Mind path-separator semantics in tests** — backslash traverses only on Windows; gate such cases with `skipif`.
- Regression suite: `tests/test_security_paths.py` (52 cases) covers helper containment, the sibling-directory bypass at both the helper and tool level, write/read/pipeline sites in both servers, and signal-read containment.

## Related Issues

- Prior fixes to the same vulnerability class: commit `61627b0` (introduced the bypassable `str.startswith` check) and `d689886` (switched to `Path.is_relative_to`).
- Residuals tracked for follow-up: pickle deserialization is unauthenticated (RCE if `MODELS_DIR` is writable by another actor — signing/allowlist deferred); broader signal-path hardening (companion `_metadata.json` resolution and per-tool `exists()` existence oracles) tracked for the signal-repository rework (plan unit U8).
- Plan: `docs/plans/2026-07-10-001-refactor-security-credibility-unified-api-plan.md` (unit U1), audit `docs/AUDIT-2026-07-10.md`.
