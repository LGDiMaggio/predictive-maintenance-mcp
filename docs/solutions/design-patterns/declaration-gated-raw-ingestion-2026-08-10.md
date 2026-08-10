---
title: Declaration-gated ingestion for headerless binary formats
date: 2026-08-10
category: design-patterns
module: signal_acquisition
problem_type: design_pattern
component: service_object
severity: medium
applies_when:
  - "Adding ingestion for an opaque/headerless format (raw binary, fixed-record dumps) that carries zero self-description"
  - "The tool surface is consumed by LLM agents that must recover from refusals without human help"
tags: [raw-binary, ingestion, declaration, validation, error-contract, no-inference]
---

# Declaration-gated ingestion for headerless binary formats

## Context

Supporting raw binary waveforms (`.bin`/`.raw`/`.dat`) tempts two bad designs:
vendor-specific format parsers in the core, or inference heuristics (guess the
dtype, sniff the endianness). Both violate the project's declared-never-guessed
policy and produce silent garbage when wrong.

## Guidance

Make the **user's declaration** the trusted input, and validate it against the
file's arithmetic instead of inferring anything:

1. **Closed vocabularies** for every enum-ish parameter (`sample_format`,
   `byte_order`), exported from ONE module and reused by the tool-layer
   `Literal`s, the companion-metadata validator, and the decoder — with a
   sync-guard test (`set(get_args(Literal)) == set(VALID_*)`).
2. **Required minimum** (here: `sample_format` + `sampling_rate`) refused in
   ONE accumulated message naming every missing parameter, the exact re-call,
   and the sidecar alternative (`<stem>_metadata.json`). Precedence: explicit
   parameter > companion field > documented default — never a silent guess.
3. **Validation of the declaration, not inference**: refuse when
   `(file_size − header_offset) % (dtype_size × n_channels) != 0` and SHOW the
   arithmetic — a size mismatch is the best detector of a wrong declaration.
   Refuse float payloads that decode to NaN/Inf (likely endianness mistake),
   with the non-finite count.
4. **The policy cuts both ways**: declaring decode parameters on a
   self-describing format (CSV, WAV, …) is refused as a contradiction of the
   file's own header; an unknown extension gets an honest "unsupported
   format", never a false "self-describing" claim.
5. **Typed raises over Optional-returns** for this path, even when the module
   convention is return-`None`: a refusal message that must reach the caller
   verbatim cannot survive a `None`. Document the deviation in the module
   docstring, and make sure no legacy try/except-`None` shell swallows it.
6. **Record the effective declaration** (post-merge, post-default) as
   provenance on the stored object so "how was this decoded" is answerable
   after the fact.
7. **Pre-read size guard**: `stat()` against an env-configurable cap before
   reading a byte; read the cap at call time (not import time) so tests can
   monkeypatch it.

## Why This Matters

Declaration-gating keeps the core vendor-neutral (adapters translate vendor
metadata into declarations), makes every failure mode honest and
LLM-recoverable (the refusal contains the fix), and prevents plausible-garbage
loads that would poison downstream analysis with confident wrong numbers.

## When to Apply

- Any new opaque-format ingestion path (planar layouts, framed records → keep
  those in adapters, not the core).
- Any parameter whose value the system could "probably guess" — don't; refuse
  with a remedy instead.

## Examples

Validation order that avoids raw exceptions from bad declarations:

```python
# vocabulary checks first (sample_format, byte_order)
# bounds BEFORE use as divisor: n_channels >= 1, channel_index in [0, n)
# stat()-based size cap BEFORE any read
# payload > 0 and payload % (dtype_size * n_channels) == 0 — message shows the math
# decode, then non-finite check on the SELECTED channel only (float formats)
```

## Related

- docs/solutions/architecture-patterns/credibility-refactor-drift-guards-and-honest-outputs-2026-07-13.md
  (single-source-of-truth vocabularies, structured refusals)
- docs/solutions/security-issues/path-traversal-model-report-signal-paths-2026-07-10.md
  (containment stays at the canonical choke point; the pure decoder takes an
  already-resolved path)
