---
description: >
  Prognostic assessment workflow: trend analysis, degradation onset detection,
  and Remaining Useful Life (RUL) estimation using the predictive-maintenance-mcp
  server. Use this skill when the user says "trend analysis", "degradation trend",
  "RUL", "remaining useful life", "prognosis", "prognostics", "failure prediction",
  "time to failure", "degradation onset", "predict failure", "vita utile residua",
  "previsione guasto", "quanto dura", "stima vita", or asks to predict when a
  component will fail based on vibration data.
---

# Prognostic Assessment (ISO 13374 Block 5)

Screen degradation trends within a recording and estimate Remaining Useful
Life from repeated measurements over time. Two distinct scopes — do not
confuse them:

1. **Within-recording screening** (`analyze_signal_trend`): is this single
   recording stationary, or does a feature drift inside it? This is a
   screening result, NOT a prognosis.
2. **Prognosis** (`estimate_rul`): extrapolation of a degradation trend
   across MULTIPLE measurements of the same machine taken days/weeks/months
   apart. RUL from a single recording is physically meaningless and the tool
   refuses it.

RUL estimates are planning input for the maintenance engineer — never a
guarantee, never a substitute for engineering judgment.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow A — Within-Recording Screening

### Step 1 — Signal Discovery

Call `list_signals(scope="memory")` for loaded signal_ids, or
`list_signals(scope="disk")` and `load_signal(filepath="<file>")` to load a
recording. This works best on long monitoring recordings or run-to-failure
data where degradation evolves within the file. For raw binary files
(`.bin`/`.raw`/`.dat`), also declare `sample_format` and `sampling_rate` (or
provide a companion `<stem>_metadata.json`) — see the signal-management skill.

### Step 2 — Trend + Onset Screening (one call)

Call `analyze_signal_trend(signal_id="<id>", feature_name="rms")`.

The tool segments the recording, extracts the chosen feature per segment,
tests trend significance (slope p-value, not bare R-squared), and detects the
first post-baseline segment exceeding baseline mean + N sigma — trend AND
onset in one result.

Parameters:
- `feature_name`: `"rms"` (default), `"kurtosis"`, `"crest_factor"`,
  `"peak_to_peak"`, ...
- `segment_duration` (default 0.1 s) and `overlap_ratio` (default 0.5)
- `onset_threshold_sigma`: baseline standard deviations that trigger onset
  detection (default 3.0; lower to 2.0 for more sensitivity, with more false
  positives)

Feature guidance:

| Indicator | Use when | Notes |
|-----------|----------|-------|
| rms | General degradation | Most common, ISO 20816 aligned |
| kurtosis | Bearing degradation | Peaks early, then may drop |
| crest_factor | Impulsive faults | Sensitive to early damage |
| peak_to_peak | Looseness, imbalance | Good for mechanical looseness |

Interpreting the result:
- `trend_direction` is decided by the slope p-value (< 0.05), not by eye
- `analysis_scope` is `"within_recording_screening"` — report it as such
- Onset fields: `onset_detected`, `onset_segment_index`, `onset_time_s`,
  `baseline_segments`. Onset inside the baseline window (first half) cannot
  be detected by construction — say so if the user asks
- The result includes the (truncated) per-segment `feature_series`: use it to
  reduce this recording to ONE measurement point for Workflow B

**Decision gate**: if the trend is stable or decreasing, report that no
degradation trend is detected within the recording. Do NOT proceed to RUL on
this basis alone.

## Workflow B — RUL from Repeated Measurements

### Step 1 — Assemble a Measurement Series

RUL needs at least 3 measurements of the same machine over time, as either:

- **Externally measured values** (e.g. RMS velocity trended by a data
  collector): pass them directly as `feature_values`.
- **One stored recording per measurement session**: load each session's
  file (batch load works:
  `load_signal(filepath=["session_day0.csv", "session_day7.csv", "session_day14.csv"], signal_unit="mm/s")`)
  and pass the ids as `signal_ids` — each recording is reduced to a single
  feature value.

Timestamps are required, one per measurement, strictly increasing, in the
unit you choose (`time_unit`, default "hours").

### Step 2 — Choose the Failure Threshold

Ask the user — do NOT guess. If the indicator is broadband velocity RMS in
mm/s, the ISO zone C/D boundary for the machine is the standard choice (the
same table assess_severity uses): 4.5 (group 2 rigid), 7.1 (group 1 rigid /
group 2 flexible), or 11.0 mm/s (group 1 flexible). A site maintenance policy
threshold also works.

### Step 3 — Estimate

Call `estimate_rul(feature_values=[2.1, 2.4, 2.9, 3.5], timestamps=[0, 168, 336, 504], failure_threshold=4.5, time_unit="hours", method="linear")`

or, from stored recordings:

Call `estimate_rul(signal_ids=["session_day0", "session_day7", "session_day14"], timestamps=[0, 7, 14], failure_threshold=4.5, feature_name="rms", time_unit="days")`

Method choice:
- `"linear"` — steady degradation (default)
- `"exponential"` — accelerating degradation curves
- `"kalman"` — needs approximately uniform measurement spacing; also returns
  a 95% RUL interval

### Step 4 — Interpret the Result Honestly

Possible statuses:
- `"estimated"`: RUL in `time_unit`, with `fit_r_squared` (goodness of fit —
  NOT a confidence) and `observation_horizon`. If RUL extends far beyond the
  observed horizon, the result says so — relay that caution verbatim
- `"no_degradation_trend"`: flat/insignificant series — no RUL number is
  produced. Recommend continuing to collect measurements
- `"threshold_already_exceeded"`: the last measurement is at/above the
  threshold — inspection is due now, there is no remaining life to estimate

Report: trend significance (p-value), fit quality, RUL with its time unit,
observation horizon, and urgency relative to the next maintenance window.

## Important Notes

- RUL estimates are **extrapolations** — actual failure time depends on
  operating conditions that may change
- Always use cautious language: "estimated RUL", "projected time to threshold"
- `fit_r_squared` measures curve fit, not probability of being right
- A single recording can NEVER produce an RUL — if the user asks, explain why
  and offer Workflow A screening plus a measurement-collection plan
- This skill supports expert maintenance scheduling decisions; it does not
  replace engineering judgment
- All processing happens locally — raw signal data never leaves the machine
