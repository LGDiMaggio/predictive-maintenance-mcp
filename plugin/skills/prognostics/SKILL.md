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

Estimate degradation trends and remaining useful life from vibration signals.
Orchestrate MCP tools in a precise prognostic sequence.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Workflow

### Step 1 — Signal Discovery

Call `list_stored_signals()` to check cached signals, or `list_signals()` to
browse the data/ directory. Identify the signal file to analyze.

The prognostics tools segment a single signal file internally, extracting a
feature series over time from consecutive segments. This works best on
run-to-failure recordings or long monitoring sessions where degradation
evolves within the signal.

### Step 2 — Trend Analysis

Call `analyze_signal_trend(signal_file=..., feature_name="rms")`.

The tool automatically segments the signal, extracts the chosen feature per
segment, and fits a linear trend.

Parameters:
- `feature_name`: degradation indicator — `"rms"` (default), `"kurtosis"`,
  `"crest_factor"`, `"peak_to_peak"`, etc.
- `sampling_rate`: auto-detected from metadata if not provided
- `segment_duration`: segment length in seconds (default: 0.1s)
- `overlap_ratio`: overlap between segments (default: 0.5)

Good feature candidates:

| Indicator | Use when | Notes |
|-----------|----------|-------|
| rms | General degradation | Most common, ISO 20816 aligned |
| kurtosis | Bearing degradation | Peaks early, then may drop |
| crest_factor | Impulsive faults | Sensitive to early damage |
| peak_to_peak | Looseness, imbalance | Good for mechanical looseness |

Ask the user which indicator to track, or default to **rms** if unsure.

Interpretation:

| R-squared | Trend direction | Meaning |
|-----------|----------------|---------|
| > 0.7 | Increasing | Strong degradation trend |
| 0.3 - 0.7 | Increasing | Moderate trend, monitor closely |
| < 0.3 | Any | No clear trend (or non-linear) |
| Any | Decreasing | Improving or post-maintenance |
| Any | Stable | Stationary condition |

**Decision gate**: If trend_direction is "stable" or "decreasing", report that
no degradation trend is detected. Ask if the user wants to proceed with RUL
estimation anyway.

### Step 3 — Degradation Onset Detection

Call `detect_signal_degradation_onset(signal_file=..., feature_name="rms", threshold_sigma=3.0)`.

- Default threshold: 3 sigma (99.7% confidence)
- Lower to 2.0 for more sensitive detection (may trigger false positives)

If onset is detected, report:
- At which segment the degradation began (`onset_segment_index`)
- How many total segments were analyzed

### Step 4 — RUL Estimation

Call `estimate_rul(signal_file=..., failure_threshold=..., method="linear")`.

**Choosing the failure threshold**:
- If ISO 20816 zone C/D boundary is known, use it (e.g., 4.5 mm/s RMS for
  Group 2 rigid machines)
- If the user has a maintenance policy threshold, use that
- If unknown, ask the user — do NOT guess

**Choosing sampling_interval**:
- Defines the time unit of the returned RUL
- Default 1.0 means RUL is expressed in "segments"
- Adjust based on what the user needs (e.g., set to actual time between
  segments for real time units)

**Choosing method**:
- `"linear"` — simpler, works for steady degradation (default)
- `"exponential"` — for accelerating degradation curves
- `"weibull"` — Weibull distribution fit (if available)
- `"kalman"` — Kalman filter estimation (if available)
- Compare linear and exponential if unsure — the one with higher confidence
  (R-squared) is typically more appropriate

If RUL is infinity or confidence is 0, the curve doesn't reach the threshold.
Explain why and suggest:
- The threshold may be too high for the current trend
- Try the other method
- The signal may not show sufficient degradation

### Step 5 — Interpretation and Recommendations

Combine all findings into a concise prognostic summary:

1. **Trend**: direction, strength (R-squared), slope, feature analyzed
2. **Onset**: which segment degradation started (if detected)
3. **RUL**: estimated time to failure with confidence
4. **Urgency**: based on RUL vs. next planned maintenance window

Recommended actions based on RUL:

| RUL | Urgency | Recommendation |
|-----|---------|----------------|
| < 1 interval | Critical | Immediate inspection/replacement |
| 1-3 intervals | High | Schedule maintenance soon |
| 3-10 intervals | Moderate | Monitor more frequently |
| > 10 intervals | Low | Continue routine monitoring |

## Important Notes

- RUL estimates are **extrapolations** — actual failure time depends on
  operating conditions that may change
- Always use cautious language: "estimated RUL", "projected failure time"
- Confidence (R-squared) below 0.5 means the estimate is unreliable
- This tool augments expert decision-making; it does not replace engineering
  judgment for maintenance scheduling
- All processing happens locally — raw signal data never leaves the machine
