# CWRU Diagnostic Benchmark — Methodology

This document describes how the benchmark in [`benchmarks/cwru/`](../benchmarks/cwru/)
measures what the bearing check of this server's **deterministic signal-processing
pipeline** reports on the Case Western Reserve University (CWRU) Bearing Data
Center dataset, and how to reproduce every published number.

The benchmark measures the evidence the tool surfaces to an expert: the lines
at the characteristic frequencies that stand above the local noise floor, the
outcome of the check, and the fault type its claim names, if any. It does not
position the tool as an autonomous diagnostician, and it does not measure LLM
orchestration: only the deterministic pipeline (`diagnose_vibration` called as
a library function) is under test.

---

## What is measured

- **System under test**: the envelope line test of `check_all_bearing_faults`
  inside `diagnose_vibration`, called deterministically with operational
  context only. Each characteristic line is tested against the local noise
  floor of one envelope spectrum, then the further lines its kinematics
  predict, and the check reports an outcome and a claim from that evidence.
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
silently change benchmark semantics): a fundamental search window of **±5 %**
of each line frequency (`tolerance_pct=5.0`), harmonics **2× to 4×** tested as
corroborating lines (`num_harmonics=3`), and the envelope demodulation band
**500 to 5000 Hz**. The fs-dependent default resolves to the same band for
both the 12 kHz fault records and the 48 kHz baselines, so the fault records
and the baselines are tested in a comparable band (noise floors still differ
with fs; treat cross-group comparisons accordingly). Signal unit is taken as
*g* per community convention; CWRU does not state units explicitly. This
assumption does not affect the scored metrics, which rest on line frequencies
and levels over the noise floor.

**Statistical policy.** The line test is an ordered-statistic CFAR test on the
envelope spectrum. Each complete check has a nominal family-wise false-alarm
budget of 1 % (`false_alarm_probability_per_check`), split evenly over the
hypotheses it can test and over the bins of each search window (Bonferroni).
The budget is nominal: it holds under the stated assumptions, chiefly
approximately exponential bin power under a stationary random envelope, and it
does not cover deterministic non-bearing lines (shaft harmonics, mains, gear
mesh), non-stationary records, or repeated checks of the same machine. It is
not the benchmark's false-alarm rate and not a specificity. The full
assumptions and exclusions are in the `evidence_policy` block of every check
result. The evidence and claim policy versions are recorded in every outcome
and in the results' `criteria.policy_versions`; the scorer refuses an outcomes
file measured under another version.

**Hit criterion** (one rule, applied identically everywhere): a fault type
*hits* when its check is *supported*: a line of its own stands above the local
noise floor and at least one further line its kinematics predict is counted,
neither of them shared with another fault type. CWRU publishes the ball
"rolling element" factor as 2×BSF (a ball defect strikes both races once per
spin revolution); the check tests the rolling-element line at twice the ball
spin frequency itself, so no fault type gets a separate harmonic leg. The
**same criterion** counts supported fault types on the fault records and on
the baselines: no asymmetry between what earns a hit and what earns a false
alarm.

- **Frequency detection**: on a fault record, the line tested for the labelled
  fault stands above the local noise floor, whether or not it is supported.
- **Label supported**: on a fault record, the labelled fault meets the hit
  criterion.
- **Outcome against the label**: the check's outcome (`indicated`,
  `several_types`, `inconclusive`, `no_indication`) read against the label:
  indicated with the labelled fault or with another one, several types with or
  without the labelled fault among the candidates. On a baseline, an indicated
  outcome counts as indicated with another fault, and a several-types outcome
  as without the label.
- **Precision of "indicated"**: indicated outcomes whose single candidate is
  the labelled fault, over every indicated outcome of the stratum. On a normal
  baseline no fault is labelled, so an indicated outcome there is never
  counted as correct. A stratum with no indicated outcome has no precision:
  it is published with both counts at zero, never as a percentage, and
  `results.json` records its rate as null.
- **Named fault**: the fault type the check names, `most_likely_fault`, which
  it sets only when its claim level is `single_spectrum_evidence`: the
  labelled one, another one, or none. No other claim level names a single
  fault type.
- **Strongest candidate**: on a fault record, the labelled fault is the first
  candidate the check lists. The check lists its candidates by level over the
  noise floor; that order is not a ranking of likelihood and names no fault.
- **Sensitivity**: on a fault record, the check lists at least one candidate
  (outcome `indicated` or `several_types`).
- **Wrong-type support**: on a fault record, a fault type other than the
  labelled one meets the hit criterion.
- **Normal baselines**: counts of no indication, inconclusive, candidates
  listed and a single fault type named, under the same criterion, and of each
  fault type supported. They are counts on these recordings, never a rate.
- **Classification (secondary)**: a secondary offline scoring analysis kept
  from the earlier scorer, not a ranking the server produces or reports. It
  ranks the supported fault types (evidence grade, then fundamental
  magnitude, then frequency deviation, then a fixed order) and counts a
  record as correct when the first-ranked type is the labelled one. It is
  computed on the fault records only and under the hit criterion above, so
  its values are not comparable with the classification figures published
  before this change. The check names no fault from it, and it is recorded
  only in `results.json`, not published in this document or the README.
- The full criterion and every definition are also recorded in the results
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
  counted in the headline**: the reference study itself could not
  diagnose them with any method, so scoring them as plain failures would
  misrepresent any pipeline — honest reporting shows them as their own line.

## Measured results

Produced by `python -m benchmarks.cwru all` on the v1 subset; artifact:
[`benchmarks/cwru/results/results.json`](../benchmarks/cwru/results/results.json)
(single source of truth — every number below is bound to a key in that file
and drift-guarded by CI).

<!-- cwru-benchmark:start -->

**Headline (Y1+Y2 strata, <!-- slot: headline.n_records -->44<!-- /slot --> records):**
line tested for the labelled fault above the noise floor <!-- slot: headline.frequency_detection.hits -->44<!-- /slot -->/<!-- slot: headline.frequency_detection.total -->44<!-- /slot -->,
labelled fault supported <!-- slot: headline.label_supported.count -->44<!-- /slot -->/<!-- slot: headline.label_supported.total -->44<!-- /slot -->,
another fault type supported as well <!-- slot: headline.wrong_type_support.records -->44<!-- /slot -->/<!-- slot: headline.wrong_type_support.total -->44<!-- /slot -->,
several types listed with the labelled fault among them <!-- slot: headline.outcomes.several_types_with_label -->44<!-- /slot -->/<!-- slot: headline.outcomes.total -->44<!-- /slot -->,
no single fault type named <!-- slot: headline.named_fault.none -->44<!-- /slot -->/<!-- slot: headline.named_fault.total -->44<!-- /slot -->,
labelled fault first among the candidates <!-- slot: headline.strongest_candidate.correct -->40<!-- /slot -->/<!-- slot: headline.strongest_candidate.total -->44<!-- /slot -->.

Evidence per stratum (fault records):

| Stratum (Smith & Randall 2015) | Records | Line above the floor | Label supported | Another type supported | Labelled fault first candidate |
|---|---|---|---|---|---|
| Y1, clearly diagnosable, classic signature | <!-- slot: strata.Y1.n_records -->9<!-- /slot --> | <!-- slot: strata.Y1.frequency_detection.hits -->9<!-- /slot -->/<!-- slot: strata.Y1.frequency_detection.total -->9<!-- /slot --> | <!-- slot: strata.Y1.label_supported.count -->9<!-- /slot -->/<!-- slot: strata.Y1.label_supported.total -->9<!-- /slot --> | <!-- slot: strata.Y1.wrong_type_support.records -->9<!-- /slot -->/<!-- slot: strata.Y1.wrong_type_support.total -->9<!-- /slot --> | <!-- slot: strata.Y1.strongest_candidate.correct -->9<!-- /slot -->/<!-- slot: strata.Y1.strongest_candidate.total -->9<!-- /slot --> |
| Y2, clearly diagnosable, non-classic | <!-- slot: strata.Y2.n_records -->35<!-- /slot --> | <!-- slot: strata.Y2.frequency_detection.hits -->35<!-- /slot -->/<!-- slot: strata.Y2.frequency_detection.total -->35<!-- /slot --> | <!-- slot: strata.Y2.label_supported.count -->35<!-- /slot -->/<!-- slot: strata.Y2.label_supported.total -->35<!-- /slot --> | <!-- slot: strata.Y2.wrong_type_support.records -->35<!-- /slot -->/<!-- slot: strata.Y2.wrong_type_support.total -->35<!-- /slot --> | <!-- slot: strata.Y2.strongest_candidate.correct -->31<!-- /slot -->/<!-- slot: strata.Y2.strongest_candidate.total -->35<!-- /slot --> |
| P1, partially diagnosable | <!-- slot: strata.P1.n_records -->1<!-- /slot --> | <!-- slot: strata.P1.frequency_detection.hits -->1<!-- /slot -->/<!-- slot: strata.P1.frequency_detection.total -->1<!-- /slot --> | <!-- slot: strata.P1.label_supported.count -->1<!-- /slot -->/<!-- slot: strata.P1.label_supported.total -->1<!-- /slot --> | <!-- slot: strata.P1.wrong_type_support.records -->1<!-- /slot -->/<!-- slot: strata.P1.wrong_type_support.total -->1<!-- /slot --> | <!-- slot: strata.P1.strongest_candidate.correct -->0<!-- /slot -->/<!-- slot: strata.P1.strongest_candidate.total -->1<!-- /slot --> |
| P2, partially diagnosable, smeared | <!-- slot: strata.P2.n_records -->5<!-- /slot --> | <!-- slot: strata.P2.frequency_detection.hits -->4<!-- /slot -->/<!-- slot: strata.P2.frequency_detection.total -->5<!-- /slot --> | <!-- slot: strata.P2.label_supported.count -->3<!-- /slot -->/<!-- slot: strata.P2.label_supported.total -->5<!-- /slot --> | <!-- slot: strata.P2.wrong_type_support.records -->4<!-- /slot -->/<!-- slot: strata.P2.wrong_type_support.total -->5<!-- /slot --> | <!-- slot: strata.P2.strongest_candidate.correct -->0<!-- /slot -->/<!-- slot: strata.P2.strongest_candidate.total -->5<!-- /slot --> |
| N1, not diagnosable by the reference study | <!-- slot: strata.N1.n_records -->10<!-- /slot --> | <!-- slot: strata.N1.frequency_detection.hits -->7<!-- /slot -->/<!-- slot: strata.N1.frequency_detection.total -->10<!-- /slot --> | <!-- slot: strata.N1.label_supported.count -->6<!-- /slot -->/<!-- slot: strata.N1.label_supported.total -->10<!-- /slot --> | <!-- slot: strata.N1.wrong_type_support.records -->10<!-- /slot -->/<!-- slot: strata.N1.wrong_type_support.total -->10<!-- /slot --> | <!-- slot: strata.N1.strongest_candidate.correct -->0<!-- /slot -->/<!-- slot: strata.N1.strongest_candidate.total -->10<!-- /slot --> |

Outcome against the label (fault records):

| Stratum | Indicated, labelled fault | Indicated, another fault | Several types with the label | Several types without it | Inconclusive | No indication |
|---|---|---|---|---|---|---|
| Y1 | <!-- slot: strata.Y1.outcomes.indicated_correct -->0<!-- /slot --> | <!-- slot: strata.Y1.outcomes.indicated_wrong -->0<!-- /slot --> | <!-- slot: strata.Y1.outcomes.several_types_with_label -->9<!-- /slot --> | <!-- slot: strata.Y1.outcomes.several_types_without_label -->0<!-- /slot --> | <!-- slot: strata.Y1.outcomes.inconclusive -->0<!-- /slot --> | <!-- slot: strata.Y1.outcomes.no_indication -->0<!-- /slot --> |
| Y2 | <!-- slot: strata.Y2.outcomes.indicated_correct -->0<!-- /slot --> | <!-- slot: strata.Y2.outcomes.indicated_wrong -->0<!-- /slot --> | <!-- slot: strata.Y2.outcomes.several_types_with_label -->35<!-- /slot --> | <!-- slot: strata.Y2.outcomes.several_types_without_label -->0<!-- /slot --> | <!-- slot: strata.Y2.outcomes.inconclusive -->0<!-- /slot --> | <!-- slot: strata.Y2.outcomes.no_indication -->0<!-- /slot --> |
| P1 | <!-- slot: strata.P1.outcomes.indicated_correct -->0<!-- /slot --> | <!-- slot: strata.P1.outcomes.indicated_wrong -->0<!-- /slot --> | <!-- slot: strata.P1.outcomes.several_types_with_label -->1<!-- /slot --> | <!-- slot: strata.P1.outcomes.several_types_without_label -->0<!-- /slot --> | <!-- slot: strata.P1.outcomes.inconclusive -->0<!-- /slot --> | <!-- slot: strata.P1.outcomes.no_indication -->0<!-- /slot --> |
| P2 | <!-- slot: strata.P2.outcomes.indicated_correct -->0<!-- /slot --> | <!-- slot: strata.P2.outcomes.indicated_wrong -->1<!-- /slot --> | <!-- slot: strata.P2.outcomes.several_types_with_label -->3<!-- /slot --> | <!-- slot: strata.P2.outcomes.several_types_without_label -->0<!-- /slot --> | <!-- slot: strata.P2.outcomes.inconclusive -->1<!-- /slot --> | <!-- slot: strata.P2.outcomes.no_indication -->0<!-- /slot --> |
| N1 | <!-- slot: strata.N1.outcomes.indicated_correct -->0<!-- /slot --> | <!-- slot: strata.N1.outcomes.indicated_wrong -->2<!-- /slot --> | <!-- slot: strata.N1.outcomes.several_types_with_label -->6<!-- /slot --> | <!-- slot: strata.N1.outcomes.several_types_without_label -->2<!-- /slot --> | <!-- slot: strata.N1.outcomes.inconclusive -->0<!-- /slot --> | <!-- slot: strata.N1.outcomes.no_indication -->0<!-- /slot --> |

Precision of an indicated outcome and sensitivity:

| Stratum | Indicated with the labelled fault, of all indicated | Candidates listed, of all fault records |
|---|---|---|
| Y1+Y2 (headline) | <!-- slot: headline.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: headline.indicated_precision.total -->0<!-- /slot --> | <!-- slot: headline.sensitivity.alerts -->44<!-- /slot -->/<!-- slot: headline.sensitivity.total -->44<!-- /slot --> |
| Y1 | <!-- slot: strata.Y1.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.Y1.indicated_precision.total -->0<!-- /slot --> | <!-- slot: strata.Y1.sensitivity.alerts -->9<!-- /slot -->/<!-- slot: strata.Y1.sensitivity.total -->9<!-- /slot --> |
| Y2 | <!-- slot: strata.Y2.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.Y2.indicated_precision.total -->0<!-- /slot --> | <!-- slot: strata.Y2.sensitivity.alerts -->35<!-- /slot -->/<!-- slot: strata.Y2.sensitivity.total -->35<!-- /slot --> |
| P1 | <!-- slot: strata.P1.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.P1.indicated_precision.total -->0<!-- /slot --> | <!-- slot: strata.P1.sensitivity.alerts -->1<!-- /slot -->/<!-- slot: strata.P1.sensitivity.total -->1<!-- /slot --> |
| P2 | <!-- slot: strata.P2.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.P2.indicated_precision.total -->1<!-- /slot --> | <!-- slot: strata.P2.sensitivity.alerts -->4<!-- /slot -->/<!-- slot: strata.P2.sensitivity.total -->5<!-- /slot --> |
| N1 | <!-- slot: strata.N1.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.N1.indicated_precision.total -->2<!-- /slot --> | <!-- slot: strata.N1.sensitivity.alerts -->10<!-- /slot -->/<!-- slot: strata.N1.sensitivity.total -->10<!-- /slot --> |
| Ungraded (normal baselines) | <!-- slot: strata.ungraded.indicated_precision.correct -->0<!-- /slot -->/<!-- slot: strata.ungraded.indicated_precision.total -->1<!-- /slot --> | not applicable |

**Normal baselines** (<!-- slot: strata.ungraded.normals.total -->4<!-- /slot --> records, counts only):
<!-- slot: strata.ungraded.normals.no_indication -->1<!-- /slot --> no indication,
<!-- slot: strata.ungraded.normals.inconclusive -->0<!-- /slot --> inconclusive,
<!-- slot: strata.ungraded.normals.several_types -->2<!-- /slot --> several types listed,
<!-- slot: strata.ungraded.normals.indicated -->1<!-- /slot --> indicated, with a single fault type named on <!-- slot: strata.ungraded.normals.named_fault -->1<!-- /slot -->;
supported on the baselines: outer race <!-- slot: strata.ungraded.normals.supported_counts.BPFO -->3<!-- /slot -->, inner race <!-- slot: strata.ungraded.normals.supported_counts.BPFI -->2<!-- /slot -->, ball <!-- slot: strata.ungraded.normals.supported_counts.BSF -->0<!-- /slot -->, cage <!-- slot: strata.ungraded.normals.supported_counts.FTF -->0<!-- /slot -->.

<!-- cwru-benchmark:end -->

Counts are always published with their denominators. The normal baselines are
reported as counts only: four recordings of one test rig cannot support a
false-alarm or specificity rate.

Reading the results: when more than one fault type is supported, the check
lists them as candidates and singles none out. Each of them has lines of its
own, since a line that falls where another detected fault type also predicts
one supports neither (on the 6205, the third outer-race harmonic and the
second inner-race harmonic lie within one percent of each other), and one
spectrum cannot tell which of them, if any, are present: the reference study
reports inner-race and outer-race components on many of the ball-fault
records. A single-record indication therefore asks for confirmation, and the
order in which the candidates are listed, by level over the noise floor, is
reported here as a measurement, not as a basis for naming a fault.

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
- **The baselines are counts, not a specificity.** Four recordings of one rig
  show how the check behaved on them; they do not estimate how often it would
  list candidates on healthy machines in general, and the nominal budget of
  the check is not such an estimate either. Only validation on further
  datasets can state a real-world specificity.

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
- **Measurement provenance is recorded with the outcomes.** Each outcomes
  artifact carries a `_provenance` block — date, git describe of the measured
  tree, platform, Python/NumPy/SciPy versions, and the measured pipeline
  version — collected once before the run. `results.json` echoes it as
  `measurement_provenance`, so every published number stays bound to the exact
  environment that measured it.
- An import-provenance tripwire refuses to run if the measured package does not
  resolve to the working tree being benchmarked (guards against measuring a
  different checkout, e.g. from a stale editable install).
- Reproduce end to end:

```bash
python -m benchmarks.cwru all
```

  (downloads ~186 MB from the CWRU site on first run, verifies pins, imports,
  runs the pipeline, scores, and rewrites the results artifact).

- **Imported signals persist locally.** `python -m benchmarks.cwru import`
  (and the `run`/`all` dispatches that invoke it) writes the converted signals
  to `data/signals/cwru/` — gitignored, never committed. These files remain on
  disk after the process exits, so in later sessions on the same machine they
  appear as `cwru_001`…`cwru_064` entries in the MCP server's `list_signals`
  disk scope. There is no built-in cleanup step: delete the
  `data/signals/cwru/` directory to remove them.

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
