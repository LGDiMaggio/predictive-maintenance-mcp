---
name: prognostics
description: >
  Prognostic assessment workflow: trend analysis, degradation onset detection,
  and Remaining Useful Life (RUL) estimation using the predictive-maintenance-mcp
  server. Use this skill when the user says "trend analysis", "degradation trend",
  "RUL", "remaining useful life", "prognosis", "prognostics", "failure prediction",
  "time to failure", "degradation onset", "predict failure", "vita utile residua",
  "previsione guasto", "quanto dura", "stima vita", or asks to predict when a
  component will fail based on vibration data. Also use it when the user asks
  where a condition indicator is heading: "forecast this indicator", "project
  this trend", "indicator projection", "previsione andamento".
---

# Prognostic Assessment (ISO 13374 Block 5)

Screen degradation trends within a recording and estimate Remaining Useful
Life from repeated measurements over time. Three distinct scopes; do not
confuse them:

1. **Within-recording screening** (`analyze_signal_trend`): is this single
   recording stationary, or does a feature drift inside it? This is a
   screening result, NOT a prognosis.
2. **Prognosis** (`estimate_rul`): extrapolation of a degradation trend
   across MULTIPLE measurements of the same machine taken days/weeks/months
   apart. RUL from a single recording is physically meaningless and the tool
   refuses it.
3. **Indicator projection** (`forecast_asset_indicator`, not yet released):
   where one indicator's values are heading over the next hours or days. A
   projection, NOT a prognosis.

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
  `"range"` (peak-to-peak), ...
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
| range | Looseness, imbalance | Good for mechanical looseness |

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
  feature value. That value is computed over the whole recording in the
  recording's own declared unit, with no unit conversion and no frequency
  band. It is not the band-limited velocity RMS that the server's ISO 20816-3
  assessment evaluates (10-1000 Hz, or 2-1000 Hz when the operating speed
  supplied or declared for the assessment is below 600 rpm), even when the
  recording is declared in mm/s, so an ISO zone boundary is not a threshold
  for it (see Step 2).

Timestamps are required, one per measurement, strictly increasing, in the
unit you choose (`time_unit`, default "hours").

### Step 2 — Choose the Failure Threshold

Ask the user — do NOT guess. A threshold must be justified for the indicator
actually trended: its feature, how it was computed (frequency band, whole
recording or segment), its unit and the measurement conditions. An ISO
20816-3 zone C/D boundary (the same table assess_severity uses: 4.5 (group 2
rigid), 7.1 (group 1 rigid / group 2 flexible), or 11.0 mm/s (group 1
flexible)) is a threshold for the series only when its values are
demonstrably comparable with that assessment: velocity RMS over the server's
ISO 20816-3 evaluation band (10-1000 Hz, or 2-1000 Hz when the operating
speed supplied or declared for the assessment is below 600 rpm), measured
under comparable conditions, for the machine's actual group and support. A
mm/s unit alone does not establish that, and the
RMS of a whole stored recording is not band-limited. A site maintenance
policy threshold also works when it was set for the same indicator and
conditions.

### Step 3 — Estimate

Call `estimate_rul(feature_values=[2.1, 2.4, 2.9, 3.5], timestamps=[0, 168, 336, 504], failure_threshold=<threshold>, time_unit="hours", method="linear")`

or, from stored recordings:

Call `estimate_rul(signal_ids=["session_day0", "session_day7", "session_day14"], timestamps=[0, 7, 14], failure_threshold=<threshold>, feature_name="rms", time_unit="days")`

`<threshold>` is the value chosen in Step 2, in the unit of the series.

Method choice:
- `"linear"` — steady degradation (default)
- `"exponential"` — accelerating degradation curves
- `"kalman"` — needs approximately uniform measurement spacing; also returns
  a nominal 95% RUL interval (`rul_interval_95`) whose coverage is not
  validated

### Step 4 — Interpret the Result Honestly

Possible statuses (quote the result's `message` for each):
- `"estimated"`: RUL in `time_unit`, with `fit_r_squared` (goodness of fit —
  NOT a confidence) and `observation_horizon`. If RUL extends far beyond the
  observed horizon, the result says so — relay that caution verbatim
- `"no_degradation_trend"`: the slope test found no statistically significant
  increasing trend, or could not be computed: no time to the threshold is
  estimated, and nothing about the machine's health follows. The `message`
  says what the slope test supports
- `"model_threshold_crossed"`: the latest measurement is below the
  threshold, but the fitted model places the indicator at or above it at that
  time: a model estimate, not an observed exceedance or a confirmed failure,
  and no time to the threshold is estimated. Report `modelled_current_value`
  next to `current_value`, the measured value
- `"model_not_rising"`: the slope test finds a significant increasing trend,
  but the requested model's rate at the latest measurement is not positive,
  so the model projects no crossing and no time to the threshold is
  estimated. The trend test and the model disagree; the `message` names the
  follow-up
- `"threshold_already_exceeded"`: the latest measurement is at or above the
  threshold supplied in the call. The threshold is an indicator limit, not an
  observed component failure, and no time to it is estimated

For `kalman`, `model_threshold_crossed` and `model_not_rising` can change
with the time unit of the timestamps, and `rul_interval_95` and
`precision_heuristic` change with both the time unit and the value scale of
the series. `precision_heuristic` is a heuristic, not a probability of
correctness.

Report: trend significance (p-value), fit quality, RUL with its time unit,
observation horizon, and urgency relative to the next maintenance window.

## Workflow C: Indicator Projection (not a prognosis; unreleased)

Use `forecast_asset_indicator` only when the user asks where an indicator's
values are heading in the next hours or days. Questions about time to failure,
remaining life or when to replace a part go to Workflow B. If the connected
server does not list `forecast_asset_indicator`, tell the user it is not
available in this server version; do not substitute `estimate_rul` or
extrapolate by hand.

Call `forecast_asset_indicator(observed_at=["2026-10-20T00:00:00Z", "2026-10-20T01:00:00Z", ...], values=[2.1, 2.2, ...], indicator="velocity RMS", unit="mm/s", resolution="1h", horizon_steps=24)`

- Every instant needs `Z` or a UTC offset; an instant without one is refused.
  A missing observation is left out, never sent as null, zero or a filled value.
- The unit and, for `backend="seasonal_naive"`, `season_length_steps` come
  from what the user states, never from a guess: ask the user.
- `status="insufficient_history"` gives a reason, the available and required
  amounts and a remedy; follow the remedy, never pad the series.
- Quote the `interpretation`, `method_note` and `interval_note` fields. A flat
  naive median repeats the last bin by construction; it does not mean the
  indicator will stay stable.
- Never derive a time to a threshold, a crossing probability or an ISO zone
  from the projection, and never pass its values to `estimate_rul` or
  `assess_severity`.

## Important Notes

- RUL estimates are **extrapolations** — actual failure time depends on
  operating conditions that may change
- For RUL results (Workflow B), always use cautious language: "estimated
  RUL", "projected time to threshold"
- `fit_r_squared` measures curve fit, not probability of being right
- A single recording can NEVER produce an RUL — if the user asks, explain why
  and offer Workflow A screening plus a measurement-collection plan
- This skill supports expert maintenance scheduling decisions; it does not
  replace engineering judgment
- All processing happens locally — raw signal data never leaves the machine
