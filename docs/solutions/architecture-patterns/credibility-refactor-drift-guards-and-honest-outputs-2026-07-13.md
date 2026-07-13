---
title: "Hardening an analysis tool: single-source-of-truth drift guards and input-aligned outputs"
date: 2026-07-13
category: architecture-patterns
module: predictive-maintenance-mcp
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - "The same fact (a threshold, a count, a version, a geometry) is encoded in more than one place"
  - "An analysis tool infers a physical quantity (unit, sampling rate) it was not told"
  - "Consolidating or merging endpoints/functions that must preserve behavior"
  - "Removing a duplicated live copy of code (a monolith, a shim) ahead of a deprecation promise"
  - "A library default silently over-specifies a parameter that later validation hard-rejects"
tags: [drift-prevention, single-source-of-truth, ci-guards, unit-discipline, golden-characterization, monolith-removal]
---

# Hardening an analysis tool: single-source-of-truth drift guards and input-aligned outputs

## Context

The v0.9.0 refactor (units U2–U11) advanced the MCP server toward industrial-grade
rigor. A review identified a handful of recurring shapes worth engineering against:
the same value encoded in two places that had drifted apart; output values that were
not traceable to a declared input or cited source (a hardcoded BPFO reference, a
confidence score derived from severity); and verdicts issued on quantities the tool
had inferred rather than been told. This doc captures the five patterns that address
those shapes so the next contributor reaches for them by default. An adversarial
multi-agent review of the finished refactor confirmed the patterns held — and caught
one regression (pattern 5) that a naive fix introduced.

## Guidance

### 1. Value drift is this repo's #1 bug class — kill it with one source + a CI guard

The single most recurring defect here is the same fact living in two places: ISO zone
thresholds duplicated between the engine and the alert module (3.0 mm/s gave zone C on
one path and B on the other); tool counts stated stale in four files; version strings
misaligned across `pyproject.toml` / `__init__.py` / `server.json` / `CITATION.cff` /
README; bearing geometry copied between the catalog JSON and an in-memory fallback.

The fix is always the same two moves:

1. **Collapse to one authoritative source.** One `_ZONE_BOUNDARIES` table. One
   `FAULT_TYPE_CANONICAL` map. The catalog JSON read by both the catalog tool and the
   document-reader fallback. Downstream code *references* the source; it never restates
   the values.
2. **Add a CI guard that mechanically compares the copies that must exist.** Where a
   value legitimately appears in two artifacts (e.g. a version in a manifest and in a
   citation file, or a documented tool call and the real schema), a test introspects
   the authoritative source and asserts the copy matches. `tests/test_version_alignment.py`
   and `tests/test_documented_calls.py` are these guards.

The guard is only as good as its coverage: the review found `CITATION.cff`'s indented
`preferred-citation.version` slipped past a `^version:` regex that only matched the
top-level line. A guard that checks *one* copy of a two-copy fact gives false confidence.
When you write a drift guard, enumerate **every** occurrence it must see.

### 2. Never infer a physical unit or sampling rate — refuse, with a remedy

An ISO severity verdict on a signal whose unit was inferred from amplitude
(`RMS > 0.5 → "g"`) is worse than no verdict: it can be confidently incorrect. The
refactor made declared units mandatory. When the unit is undeclared, the tool returns a
**structured refusal at the schema level** (`status: "refused"` + `reason` + `remedy`
naming `load_signal(signal_unit=...)`), not prose in a log line an LLM can drop. The
same discipline applies to sampling rate: explicit > companion metadata > structured
error, never a silent default.

This is also a positioning invariant: the tool augments expert judgment, it never
replaces it (auto memory [claude]). An inferred unit is the tool overreaching — asserting
something only the operator can know.

The trap to avoid: a *degraded* tool must still refuse cleanly. `diagnose_vibration`
degrades (the ISO block refuses while spectral/bearing/anomaly blocks still run) rather
than either failing entirely or fabricating a unit to keep going. And a regenerated
prompt/skill must carry the same discipline — the review caught the `diagnose_bearing`
prompt hardcoding `signal_unit="g"` in its refusal-recovery step, silently reintroducing
the exact guess the engine had been hardened to reject.

### 3. Behavior-preserving merges need golden characterization, not just a parity test

Consolidating 54 endpoints to 36 meant four tools merging into `assess_severity`, three
into `check_bearing_faults`, etc. A structural parity test (every old name maps to a
declared destination; the destination exists) proves the *surface* is right but says
nothing about whether a merge silently changed a *code path* — which is exactly the drift
class from pattern 1, one level up.

The guard is **golden characterization**: before deleting the old tools, capture their
outputs on fixed deterministic fixtures to a committed snapshot
(`tests/fixtures/golden_merges.json`), then assert the merged tool reproduces them for
every behavior-preserving route. Paths that are *intentionally* corrected (envelope
detrend, mandatory units) get new expected-value tests instead, with a comment marking
the change as deliberate. The snapshot is frozen — it cannot be regenerated once the old
tools are gone, which is the point.

### 4. Removing a duplicated live copy can be a security decision, not just cleanup

The monolith (`machinery_diagnostics_server.py`) was promised for removal in v1.0.0. It
was removed in v0.9.0 instead. The reason is the security lesson from U1: a path-traversal
fix had originally landed in only one of the two live copies of the analysis code. As long
as the monolith survived as an unmaintained second copy of every path, it was a standing
liability — the next fix could just as easily miss half the surface, and the divergent copy
could rot into wrong behavior. Removing it ahead of the promise was justified explicitly in
the CHANGELOG (deviating from a public promise needs a written rationale). The lesson: a
second live copy of security-relevant code is a risk to retire, and "we promised to keep it
until X" is outweighed by "it is unpatched and divergent now."

### 5. An fs-aware default must not become a hard-fail when validation tightens

When you add strict validation (here: a band-pass band that raises instead of silently
clamping to Nyquist), audit every **default** that feeds it. The envelope band defaulted
to `(500, 5000)` Hz; the new validator hard-raised on any signal with Nyquist ≤ 5000 Hz
(fs ≤ 10 kHz — common in real bearing datasets), so a bearing check on an 8 kHz signal
failed entirely. Tightening validation is correct; letting a fixed default trip it for
legitimate inputs is a regression.

The fix distinguishes *default* from *explicit*: a `None` sentinel default resolves
fs-aware (`min(5000, Nyquist − 1)`), while an explicitly caller-supplied over-Nyquist band
still raises. Critically, the fs-aware clamp must be a **no-op at the fs where behavior was
already correct** (fs = 10 kHz here) so golden tests stay byte-identical — verify that
before shipping.

## Why This Matters

Each of these is a way the tool can be *confidently incorrect* — the failure mode expert
users notice fastest. A drifted threshold, a hardcoded reference frequency, an inferred
unit, a silently-changed merge, a default that rejects real data: none throw an error a
user would notice, and all produce output that looks authoritative. The guards turn these
silent failures into loud ones — a red CI test, a structured refusal, a frozen snapshot
mismatch — which is the only way they stay fixed as the code keeps changing.

## When to Apply

- Before adding a second place that states a value already stated elsewhere — collapse or guard it.
- When a tool is about to infer a physical quantity it wasn't given — refuse with a remedy instead.
- Before merging or deleting code that must preserve behavior — snapshot first.
- When tightening validation — re-audit every default that feeds the new check.
- When deviating from a public deprecation promise — write the rationale in the CHANGELOG.

## Examples

Drift guard that must see *every* copy (the review's near-miss):

```python
# WEAK — only matches the un-indented top-level version, misses preferred-citation:
re.search(r"^version:\s*(\S+)\s*$", citation_cff, re.M)

# CORRECT — captures both the top-level and the indented preferred-citation version:
re.findall(r"^\s*version:\s*(\S+)\s*$", citation_cff, re.M)  # assert all == package version
```

Schema-level refusal instead of a guessed unit:

```python
# Undeclared unit -> structured refusal the LLM cannot lose, naming the fix:
if unit is None:
    raise ValueError(
        "ISO severity refused: signal unit not declared — units are never "
        "guessed from amplitude. Re-load with "
        "load_signal(filepath=..., signal_unit='g'|'m/s2'|'mm/s'|'m/s', overwrite=True)."
    )
```

Default-vs-explicit so tightened validation doesn't reject real data:

```python
def compute_envelope_spectrum(signal, fs, frequency_range=None):
    nyquist = fs / 2
    if frequency_range is None:                      # default: fs-aware, never raises
        low, high = 500.0, min(5000.0, nyquist - 1.0)  # no-op at fs=10 kHz (golden stays green)
    else:                                            # explicit: honored and validated as given
        low, high = frequency_range
        validate_bandpass_band(low, high, nyquist)   # over-Nyquist explicit band still raises
```

## Related
- [Path traversal contained across every model/report/signal path](../security-issues/path-traversal-model-report-signal-paths-2026-07-10.md) — the U1 security fix whose "one of two live copies" lesson motivated pattern 4.
- Plan: `docs/plans/2026-07-10-001-refactor-security-credibility-unified-api-plan.md` (requirements R1–R12, units U1–U11).
- Origin audit: `docs/AUDIT-2026-07-10.md`.
