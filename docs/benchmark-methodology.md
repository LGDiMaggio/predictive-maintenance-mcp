# CWRU Diagnostic Benchmark — Methodology

This document describes how the benchmark in [`benchmarks/cwru/`](../benchmarks/cwru/)
measures the diagnostic accuracy of this server's **deterministic signal-processing
pipeline** on the Case Western Reserve University (CWRU) Bearing Data Center
dataset, and how to reproduce every published number.

The benchmark measures the evidence the tool surfaces to an expert — detected
characteristic frequencies, their evidence strength, and a ranked fault
indication. It does not position the tool as an autonomous diagnostician, and
it does not measure LLM orchestration: only the deterministic pipeline
(`diagnose_vibration` called as a library function) is under test.

---

## What is measured

- **System under test**: envelope-based bearing analysis + characteristic-frequency
  matching (`check_all_bearing_faults` inside `diagnose_vibration`), called
  deterministically with operational context only.
- **Out of scope**: the anomaly-detection stage (its models are not versioned in
  the repository, so including it would break reproducibility — the runner
  replaces its output with a constant
  `{"status": "excluded_unversioned_models"}` marker regardless of local model
  availability), ISO 20816-3 severity (carried in the outcomes as context, never
  scored — the CWRU test rig is not a machine class the standard targets), and
  any LLM-in-the-loop behavior.

## Dataset and access

- **Source**: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter).
  The site publishes the data as direct `.mat` downloads and, at the time of
  writing, publishes no explicit license text. This repository therefore
  **redistributes no CWRU data** and makes no claim about licensing: files are
  downloaded on demand from the official site, cached locally (gitignored), and
  verified against vendored SHA-256 + byte-size pins
  ([`benchmarks/cwru/checksums.json`](../benchmarks/cwru/checksums.json),
  produced once by a maintainer `freeze` step and required thereafter —
  verification fails closed on any missing pin or mismatch).
- **v1 subset**: the 12 kHz drive-end (DE) fault group — 60 records covering
  inner-race, ball, and outer-race faults (0.007″–0.028″, motor loads 0–3 hp)
  — plus the 4 normal baselines (files 97–100, recorded at 48 kHz). 64 records
  total. The vendored tables assert the official per-configuration quirks
  (0.014″ OR only centred; 0.028″ has no OR records; file 157 does not exist).
- **Known data-quality issues**: per Smith & Randall's Table 3, the only
  documented anomalies inside this subset are clipped sections in the DE
  channel of files 236 and 237; both records are flagged (`known_anomalies`)
  and reported, never silently excluded.

## Blind protocol

Fault labels are never visible to the system under test:

- The vendored metadata is split into two files:
  [`records_ops.json`](../benchmarks/cwru/records_ops.json) (operational fields
  only — opaque id, download URL, `.mat` variable key, sampling rate, nominal
  rpm, load, cache filename) parsed by the downloader/importer/runner, and
  [`labels.json`](../benchmarks/cwru/labels.json) (fault type, size, position,
  diagnosability grade, known anomalies) parsed **only** by the scorer.
- Signals enter the ordinary signal repository under opaque ids
  (`cwru_001`…`cwru_064`) with a minimal companion (`sampling_rate`,
  `signal_unit`) — CWRU's own label-encoding filenames never reach the
  pipeline.
- The pipeline receives exactly: the waveform, sampling rate, nominal rpm,
  bearing designation (SKF 6205, drive end), and declared unit — the same
  operational context a technician would have.
- Blindness is enforced by executable guards, not convention
  (`tests/test_cwru_benchmark_guards.py`): ops/label key disjointness at file
  level, a static scan that no runner-side module references the label
  accessor, companion-content checks, and a call-boundary guard that intercepts
  the actual `diagnose_vibration` invocation and asserts no label-bearing value
  is passed. Each guard is mutation-tested (shown to go red when a leak is
  injected).

## Operational metric definitions

Analyzer parameters in effect (pinned here so a future default change cannot
silently change benchmark semantics): frequency tolerance **±5 %**
(`tolerance_pct=5.0`), up to **3 harmonics** checked per fault frequency
(`num_harmonics=3`), envelope demodulation band **500–5000 Hz** — the
fs-dependent default resolves to the same band for both the 12 kHz fault
records and the 48 kHz baselines, so detection and false-positive metrics run
in a comparable band (noise floors still differ with fs; treat cross-group
comparisons accordingly. Signal unit is taken as *g* per community convention;
CWRU does not state units explicitly — this assumption does not affect scored
metrics, which are frequency-based).

**Hit criterion** (one rule, applied identically everywhere): a fault *hits*
when its characteristic frequency is `detected` with evidence strength
*high* or *moderate*; **ball faults additionally** hit when the 2× harmonic of
the classical BSF appears among the detected harmonics. CWRU publishes the
ball "rolling element" factor as 2×BSF (a ball defect strikes both races per
revolution of the ball); the analyzer computes classical BSF from geometry and
caps harmonic-only findings below the detection threshold, so the 2× leg reads
the raw per-fault fields. The **same criterion** — harmonic leg included —
counts detections on faulted records and false positives on baselines: no
asymmetry between what earns a hit and what earns a false alarm.

- **Frequency detection**: on a faulted record, the labeled fault meets the hit
  criterion.
- **Classification**: the scorer ranks all faults meeting the criterion
  (evidence tier, then fundamental magnitude, then frequency deviation, with a
  fixed order as final tie-break) and scores correct when the top-ranked fault
  maps to the label. The pipeline's own `most_likely` summary is deliberately
  not consumed (it cannot express a 2×-harmonic-only ball detection). A normal
  record is correct when no fault meets the criterion.
- **False positives**: on the 4 normal baselines, any fault meeting the same
  hit criterion.
- The full criterion and ranking rule are embedded verbatim in the results
  artifact (`criteria` block) for audit.

## Stratification

Per-record scoring is stratified by the diagnosability grades of
**Smith & Randall (2015)** — the reference benchmark study that applied three
classical diagnostic methods to every CWRU record and graded each outcome
(Y1 clearly diagnosable/classic … N2 not diagnosable/noise-like). Grades are
transcribed from the paper's Appendix Table B2 (drive-end channel, best grade
across the paper's methods). Normal baselines are not fault-graded and appear
in a visible *ungraded* stratum.

- **Headline numbers cover Y1+Y2 only** — records the reference study found
  clearly diagnosable by classical methods.
- **N1/N2 records are reported separately, never silently excluded and never
  counted in headline accuracy**: the reference study itself could not
  diagnose them with any method, so scoring them as plain failures would
  misrepresent any pipeline — honest reporting shows them as their own line.

## Measured results

Produced by `python -m benchmarks.cwru all` on the v1 subset; artifact:
[`benchmarks/cwru/results/results.json`](../benchmarks/cwru/results/results.json)
(single source of truth — every number below is bound to a key in that file
and drift-guarded by CI).

<!-- cwru-benchmark:start -->

**Headline (Y1+Y2 strata, <!-- slot: headline.n_records -->44<!-- /slot --> records):**
frequency detection <!-- slot: headline.frequency_detection.hits -->44<!-- /slot -->/<!-- slot: headline.frequency_detection.total -->44<!-- /slot --> (<!-- slot: headline.frequency_detection.rate pct1 -->100.0<!-- /slot -->%),
fault classification <!-- slot: headline.classification.correct -->34<!-- /slot -->/<!-- slot: headline.classification.total -->44<!-- /slot --> (<!-- slot: headline.classification.rate pct1 -->77.3<!-- /slot -->%).

| Stratum (Smith & Randall 2015) | Records | Frequency detection | Classification |
|---|---|---|---|
| Y1 — clearly diagnosable, classic signature | <!-- slot: strata.Y1.n_records -->9<!-- /slot --> | <!-- slot: strata.Y1.frequency_detection.hits -->9<!-- /slot -->/<!-- slot: strata.Y1.frequency_detection.total -->9<!-- /slot --> | <!-- slot: strata.Y1.classification.correct -->9<!-- /slot -->/<!-- slot: strata.Y1.classification.total -->9<!-- /slot --> (<!-- slot: strata.Y1.classification.rate pct0 -->100<!-- /slot -->%) |
| Y2 — clearly diagnosable, non-classic | <!-- slot: strata.Y2.n_records -->35<!-- /slot --> | <!-- slot: strata.Y2.frequency_detection.hits -->35<!-- /slot -->/<!-- slot: strata.Y2.frequency_detection.total -->35<!-- /slot --> | <!-- slot: strata.Y2.classification.correct -->25<!-- /slot -->/<!-- slot: strata.Y2.classification.total -->35<!-- /slot --> (<!-- slot: strata.Y2.classification.rate pct1 -->71.4<!-- /slot -->%) |
| P1 — partially diagnosable | <!-- slot: strata.P1.n_records -->1<!-- /slot --> | <!-- slot: strata.P1.frequency_detection.hits -->1<!-- /slot -->/<!-- slot: strata.P1.frequency_detection.total -->1<!-- /slot --> | <!-- slot: strata.P1.classification.correct -->0<!-- /slot -->/<!-- slot: strata.P1.classification.total -->1<!-- /slot --> |
| P2 — partially diagnosable, smeared | <!-- slot: strata.P2.n_records -->5<!-- /slot --> | <!-- slot: strata.P2.frequency_detection.hits -->5<!-- /slot -->/<!-- slot: strata.P2.frequency_detection.total -->5<!-- /slot --> | <!-- slot: strata.P2.classification.correct -->0<!-- /slot -->/<!-- slot: strata.P2.classification.total -->5<!-- /slot --> |
| N1 — not diagnosable by the reference study | <!-- slot: strata.N1.n_records -->10<!-- /slot --> | <!-- slot: strata.N1.frequency_detection.hits -->10<!-- /slot -->/<!-- slot: strata.N1.frequency_detection.total -->10<!-- /slot --> | <!-- slot: strata.N1.classification.correct -->2<!-- /slot -->/<!-- slot: strata.N1.classification.total -->10<!-- /slot --> |

**Normal baselines** (<!-- slot: strata.ungraded.false_positives.total_normal -->4<!-- /slot --> records):
<!-- slot: strata.ungraded.false_positives.records_with_any -->2<!-- /slot --> raised at least one false indication
(<!-- slot: strata.ungraded.false_positives.per_fault_counts.BPFO -->1<!-- /slot --> outer-race, <!-- slot: strata.ungraded.false_positives.per_fault_counts.BSF -->1<!-- /slot --> ball) under the same hit criterion used for detection.

<!-- cwru-benchmark:end -->

Counts are always published alongside rates: with only 4 baseline records the
false-positive rate has 25-percentage-point granularity, and rates alone would
overstate the precision of the measurement.

Reading the strata: on records the reference study found clearly diagnosable
with a classic signature (Y1), the pipeline identifies the correct fault every
time. Frequency-level energy at the labeled fault frequency is detectable on
every fault record in the subset; naming the *correct* fault first is the
discriminating metric, and it degrades exactly where the reference study says
records get hard — most residual misclassifications sit in the ball-fault and
0.028″ records that Smith & Randall could not diagnose with any method.

## Honest-benchmarking notes

- **Whole-record scoring, no training, no splits.** Widely-cited 99–100 %
  accuracies on CWRU typically come from slicing recordings into overlapping
  segments and randomly splitting them between training and test — near-
  duplicate leakage that inflates results (Hendriks, Dumond & Knox 2022,
  MSSP 169:108732; see also Smith & Randall 2015). This benchmark has no
  trained component and scores each physical recording exactly once, so those
  numbers are not comparable with these.
- **The reference grading is itself per-method.** Using the best-across-methods
  grade makes the Y1/Y2 strata a *ceiling* claim about record diagnosability,
  which is the conservative direction for headline reporting.

## Determinism and reproducibility

- The runner performs a byte-identity check: the full record set is executed
  twice and the serialized outcomes must match byte-for-byte
  (`python -m benchmarks.cwru run --check-determinism`). This claim is scoped
  to the measured environment, which is recorded in the artifact metadata
  (platform, Python, NumPy and SciPy versions, git describe of the measured
  tree). Cross-platform re-runs are expected to reproduce **metric-level**
  results (per-record hits and rates), not byte-identical artifacts —
  low-order floating-point bits differ across BLAS builds and operating
  systems.
- An import-provenance tripwire refuses to run if the measured package does not
  resolve to the working tree being benchmarked (guards against measuring a
  different checkout, e.g. from a stale editable install).
- Reproduce end to end:

```bash
python -m benchmarks.cwru all
```

  (downloads ~186 MB from the CWRU site on first run, verifies pins, imports,
  runs the pipeline, scores, and rewrites the results artifact).

## References

- Smith, W.A., Randall, R.B. (2015). *Rolling element bearing diagnostics using
  the Case Western Reserve University data: a benchmark study.* Mechanical
  Systems and Signal Processing 64–65, 100–131.
  DOI [10.1016/j.ymssp.2015.04.021](https://doi.org/10.1016/j.ymssp.2015.04.021)
- Hendriks, J., Dumond, P., Knox, D.A. (2022). *Towards better benchmarking
  using the CWRU bearing fault dataset.* Mechanical Systems and Signal
  Processing 169, 108732.
  DOI [10.1016/j.ymssp.2021.108732](https://doi.org/10.1016/j.ymssp.2021.108732)
- CWRU Bearing Data Center: <https://engineering.case.edu/bearingdatacenter>
