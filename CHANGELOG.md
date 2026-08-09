# Changelog

All notable changes to the Predictive Maintenance MCP Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Lets the existing `load_signal` tool open headerless raw binary waveforms
(`.bin`, `.raw`, `.dat`) when — and only when — the caller declares how to
decode them. Vendor-neutral by design: no format parsers ship in the core and
nothing is inferred from file content or names; translating a vendor's
metadata into the declaration is the user's (or an external adapter's) job.

### Added
- **Raw binary ingestion via declaration parameters on `load_signal`.**
  Additive keyword parameters describe the layout: `sample_format`
  (`float32`/`float64`/`int16`/`int32`), `byte_order` (documented default
  `little`), `n_channels`/`channel_index` (interleaved extraction; a
  multi-channel derived id gains a `_ch<k>` suffix so channels never
  collide), `header_offset`, and an optional `scale_factor`. Required for a
  raw file: `sample_format` AND `sampling_rate` — explicit or from the
  companion `<stem>_metadata.json` (explicit wins) — because a headerless
  file has zero self-description; a load missing either is refused with ONE
  message naming everything missing and both remedies. The declaration is
  validated, never trusted blindly: payload-divisibility failures show the
  arithmetic (the best detector of a wrong dtype/channel count), and a float
  payload decoding to NaN/Inf is refused as a likely endianness/dtype
  mismatch. Declaring raw parameters on a self-describing format (CSV, WAV,
  ...) is refused as a contradiction — no-inference cuts both ways. Integer
  formats decode to raw ADC counts; `scale_factor` is the user's declared
  calibration multiplier into the physical unit — without it, declaring a
  unit on raw counts would have produced dangerously wrong ISO verdicts.
- **`raw_format` provenance on stored signals.** The six EFFECTIVE decode
  parameters (after the explicit > companion > default merge) are recorded
  on `StoredSignalInfo`, so `get_signal_info` can answer "how was this file
  decoded" after the fact instead of leaving the decode contract implicit.
- **`PMM_MAX_SIGNAL_SIZE` pre-read size cap** (bytes, default 500 MB) for
  raw loads, checked with `stat()` before a single byte is read — an
  explicit refusal with the env-var remedy instead of an OOM. Read from the
  environment at each call (the `PMM_SIGNAL_CACHE_GB` pattern), so it is
  overridable at runtime and testable. The 100 MB figure CLAUDE.md promised
  was never implemented and is too low for legitimate captures (1 h at
  25.6 kHz float32 ≈ 368 MB). Extending the cap to the other formats is
  follow-up work.

### Fixed
- **`.dat` was listed but unloadable.** The extension sat in
  `SUPPORTED_EXTENSIONS` — so `list_signals(scope="disk")` showed such
  files — but no loader branch existed, so every load failed. `.dat` is now
  raw-eligible on the same terms as `.bin`/`.raw`: it loads with a declared
  decode contract and is refused with the full remedy message without one.

### Security
- Path-containment tests extended to raw reads: relative traversal,
  Windows backslash traversal, and sibling-directory escapes with `.bin`
  paths are asserted rejected on the same closed-oracle terms as the
  existing formats (refusals reveal neither external file content nor
  directory listings).

## [0.12.0] - 2026-08-08

Moves progress narration off the MCP logging capability, which SEP-2577
deprecated on 2026-07-28 with no in-protocol replacement.

Breaking, released as a MINOR bump under the pre-1.0 rule (SemVer §4; see
CLAUDE.md Key Invariants #3). No deprecation cycle is possible — upstream
removed the capability with no replacement.

### Changed
- **The client no longer receives progress notifications.** All 119
  `ctx.info` / `ctx.warning` call sites now write to the module logger.
  This is a user-visible change for MCP clients that displayed those
  messages, and it is forced: SEP-2577 names stderr and OpenTelemetry as the
  alternatives, and `ctx.report_progress` (which survives) carries a numeric
  fraction, not narration.
- **Logging is now configured on the package logger, at import, with
  `propagate = False`** (`server.configure_logging`). The `logging.basicConfig`
  call this previously relied on was inert: `MCPServer(...)` claims the root
  logger while `server.py` is still being imported, and `basicConfig` does
  nothing when root already has handlers. Three consequences, all now fixed:
  the intended format was silently discarded (records rendered with no logger
  name, so nothing said which of the six tool modules emitted them); the
  destination belonged to whichever component configured root first, so a host
  calling `basicConfig(stream=sys.stdout)` before importing this package put
  every tool's narration onto the stdio transport's JSON-RPC channel; and
  because `__init__` exports `mcp` as a runnable object, an embedder calling
  `mcp.run()` never reached `main()` and so dropped all INFO narration
  entirely rather than relocating it.
- **`MCP_LOG_LEVEL`** now sets the package log level (default `INFO`). An
  unrecognised value falls back to `INFO` and says so, rather than failing an
  import.
- Log records are held to one physical line and bounded in length. A newline
  in a caller-supplied `signal_id` / `file_name` / `bearing_id` was cosmetic
  when the client re-rendered it; in a line-oriented operator log it forges an
  entry. An unbounded value was a large JSON payload the client had to drain;
  it is now an unbounded synchronous write to a pipe the client is not obliged
  to drain, which can block a coroutine that has no await points.
- stderr is reconfigured to UTF-8 with `backslashreplace`. A piped stderr on
  Windows decodes as the ANSI code page, where a non-ASCII identifier would
  raise inside `emit()` — dropping the line and printing "--- Logging error
  ---" in its place.
- WeasyPrint's INFO progress chatter is quieted to WARNING. It propagates to
  root, which this package does not control.
- Under mcp 2.x those methods still worked, but `MCPDeprecationWarning`
  subclasses `UserWarning` rather than `DeprecationWarning` precisely so it
  shows without any filter configured — so every one of those call sites
  had begun printing a warning on first execution.
- The 75 `if ctx:` guards that existed only to protect those calls are gone;
  the module logger needs no guard. Net -57 lines.
- `ctx` is still injected into every tool. `tests/fixtures/tool_inventory.json`
  pins `context_kwarg` per tool, so removing the parameter would have been a
  protocol-visible change. What changed is how progress is emitted, not
  whether tools accept a context.

### Added
- `tests/test_no_client_logging.py` — tripwires for both properties this
  release depends on. `ctx` is still injected, so `await ctx.info(...)` still
  type-checks and still appears to work; every other test mocks `ctx`, so
  nothing in the suite would notice a reintroduction. The guard walks the AST
  and keys on the `Context` annotation rather than the literal name `ctx`,
  which catches a renamed parameter, a local alias, and
  `ctx.session.log(...)` — the connection-level shape, deprecated by the same
  SEP. Two subprocess tests assert where the bytes actually land: that a
  record reaches fd 2 and never fd 1, and that a host owning the root logger
  cannot redirect them onto stdout. Nothing previously asserted this, which is
  why the inert `basicConfig` went unnoticed.
- `package_caplog` fixture (`tests/conftest.py`). With `propagate = False`,
  caplog's root handler no longer sees this package's records, so a test
  asserting on plain `caplog.records` is not lenient — it is vacuous.

### Fixed
- The `ctx:` line in 29 tool docstrings, which had drifted into four different
  wordings — including seven pointing at a "module note" nobody had written,
  and four still promising client-facing progress the code no longer sends.
  Docstrings are the tool descriptions MCP clients read, so these were
  user-facing. Each module now carries the note the references promised.
- Three `caplog` assertions in `tests/test_report_tools.py` asserted only that
  *some* record existed, which any logger in the process satisfies. In
  `test_envelope_report_with_ctx_bearing_matches` three unconditional records
  precede the branch under test, so it passed with that branch deleted.
- `_extract_features_from_ids` and `_extract_and_transform_validation_features`
  no longer take a dead `ctx` (removed from 8 call sites). They are private
  helpers, not registered tools, so the `tool_inventory.json` constraint that
  justifies keeping `ctx` on public tools never applied to them.

### Development

No runtime behaviour changes in this section. It is recorded because the
shipped source is affected: the whole tree was reformatted, and the checks
that decide whether a release is publishable are no longer decorative.

- **Every CI check now blocks.** The mypy and Black jobs both ran with
  `continue-on-error: true`, so they reported green whatever they found. mypy
  is blocking against a frozen baseline of 33 pre-existing errors
  (`tools/check_mypy_baseline.py`); Black is blocking outright, which required
  reformatting 70 files. `tests/test_ci_gates.py` asserts no job can go
  non-blocking again, with one documented exception (the Codecov upload, whose
  failure is not a claim about this tree).
- **The type checker sees model construction again.** Adding the `pydantic.mypy`
  plugin with its default `init_forbid_extra = false` had replaced the typed
  `__init__` with `**kwargs: Any`; combined with pydantic's default
  `extra='ignore'`, a misspelled field on any of the 18 result models was
  caught by nothing and silently fell back to its default. Now set to `true`.
- **Tests exercise the checkout they live in.** A shared virtualenv's editable
  install pins the package to one absolute path, so every git worktree ran the
  suite against the primary checkout's source. `tests/conftest.py` binds
  resolution to its own tree and fails loudly if that ever breaks;
  `scripts/setup-worktree.py` builds a per-worktree environment and refuses to
  run in the primary checkout, where it would prune the shared venv's optional
  extras and silently disable the PDF tests everywhere.
- mypy pinned to `>=2.3,<2.4`, since the baseline freezes an exact per-error
  tally and the job now blocks.
- Source formatting normalised repo-wide by Black. Verified semantics-preserving
  by AST comparison of every changed file.

## [0.11.0] - 2026-08-08

Adds the integrated diagnostic report. Additive: no existing tool
signature changes, and the endpoint surface grows from 36 to 37
(33 -> 34 tools).

### Added
- `generate_diagnostic_report` — one integrated diagnostic document covering
  signal overview, ISO severity, anomaly state, characteristic-frequency
  matching, spectral energy, an annotated envelope spectrum, and recommended
  actions. Additive: the surface grows from 36 to 37 endpoints (33 → 34
  tools), no existing tool signature changes.
- Server-authored advisory layer (`decision_support.advisory`). Every
  evaluative sentence a report shows is written by the server that computed
  the numbers and returned to the caller in a `statements` list. The previous
  arrangement let the caller author report content — see
  `generate_diagnostic_report_docx`'s `sections` argument — which is how a
  rendering ends up carrying faithful numbers under a standard name, machine
  class, or confidence grade this codebase does not produce.
- Indicator-disagreement reconciliation: when the ISO zone reads acceptable
  while the fault pattern does not, the report says so explicitly and names
  which indicator governs the recommended action.
- Baseline comparison: supplying a healthy reference signal turns absolute
  readings into deltas, including envelope amplitude at the characteristic
  frequency the verdict rests on — the figure that separates "the machine got
  noisier" from "this defect grew".
- Annotated envelope spectrum (`figures`), drawn as inline SVG with the
  characteristic-frequency bands at the tolerance the diagnosis actually
  used, so the match is visible rather than asserted. No CDN script and no
  hover-only labels: the report opens with no network access and the
  annotation survives a static export.
- Optional `[pdf]` extra. The PDF is a rendering of the same HTML document
  rather than a second layout, so both renderings carry identical statements
  by construction; a test asserts it.
- Report authorship policy in the server instructions: standard names,
  machine classes, zones, and confidence levels are not the caller's to coin,
  and `evidence_strength` is a count of corroborating findings that must
  never be rendered as a probability.

### Fixed
- The shared HTML report base template interpolated the report title — which
  carries a user-supplied signal identifier — without escaping, and embedded
  its metadata JSON in a way a `</script>` sequence in any value could close
  early. Both affected every report family.
- The envelope spectrum drawn in the integrated report is now computed by the
  same path the bearing matching consumes (`envelope_spectrum_arrays`), so a
  figure cannot show a peak the verdict never saw.

### Changed
- `tests/test_surface_parity.py` previously required every registered tool to
  be a migration destination of the v0.8.x surface, which forbade adding any
  new capability. It now requires every tool to be migrated **or** declared
  in `POST_U9_ADDITIONS`, preserving the "no orphan tools" property while
  allowing intentional additions.

## [0.10.0] - 2026-08-08

Minor release rather than a patch: the dependency floor change below is
breaking for installs, and this project's pre-1.0 semver treats that as a
MINOR bump.

### Changed
- **BREAKING (install):** migrated the server to the mcp 2.x API and raised
  the floor to `mcp[cli]>=2.0.0`. mcp 2.0.0 removed `mcp.server.fastmcp`,
  so `FastMCP` is now `MCPServer` from `mcp.server.mcpserver`. The two APIs
  cannot be supported from one source tree, so mcp 1.x is no longer
  installable with this package.
- Transport wiring no longer goes through `mcp.settings`: mcp 2.x dropped
  `Settings.host` / `Settings.port`, so `--host` / `--port` are passed to
  `mcp.run()` as per-transport kwargs. `stdio` is invoked without them
  (`run_stdio_async` accepts neither).
- **Public export shape.** `predictive_maintenance_mcp.mcp` is a declared
  export (`__all__` in `src/__init__.py`) and its runtime type changes from
  `mcp.server.fastmcp.FastMCP` to `mcp.server.mcpserver.MCPServer`. It no
  longer carries `.settings.host` / `.settings.port` — those moved to
  per-transport `run()` keyword arguments upstream. Anyone importing that
  singleton and reading the bind address off it must read it from the CLI /
  env instead.
- Environment-supplied configuration is now validated. `MCP_TRANSPORT` was
  only checked by argparse when passed as a flag, never as a default, so an
  unrecognised value from a compose file reached `run()` and crash-looped.
  A set-but-empty `MCP_HOST` resolved to the empty string, which binds to
  every interface — the opposite of what blanking it suggests. Both now
  resolve to the documented defaults or exit with a clear message.
- The pre-bind log line reads "Binding to" rather than "Listening on": it is
  emitted before `run()` is called, so on a port conflict the old wording
  asserted success directly above the traceback.
- `Context.debug/info/warning/error` are deprecated upstream under SEP-2577
  (still async, still functional, but they now emit a warning that is shown
  by default). This release does not migrate the ~116 call sites; that is
  tracked separately.

The registered surface is unchanged — 33 tools, 0 resources, 3 prompts,
with identical names and input schemas. `tests/fixtures/tool_inventory.json`
was **not** regenerated: it passes as-is against mcp 2.0.0, which is the
evidence that the migration is protocol-invisible.

## [0.9.1] - 2026-07-13

Patch release: two robustness fixes of the same class caught while
diagnosing crash reports from a pre-0.9.0 server.

### Fixed
- `generate_envelope_report` now resolves its default upper band edge
  fs-aware (`min(5000, Nyquist-1)`) instead of a fixed 5000 Hz. On any
  signal sampled below 10 kHz the fixed default exceeded Nyquist and the
  report failed with an invalid-band error; a default envelope report now
  succeeds on sub-10 kHz signals. An explicitly requested band above
  Nyquist is still rejected — never clamped silently.
- The ISO unit conversion (`_convert_to_velocity_mm_s` /
  `assess_severity_raw`) now refuses an undeclared (`None`) unit with an
  actionable message instead of crashing with `'NoneType' object has no
  attribute 'lower'`. The live `diagnose_vibration` / `assess_severity`
  paths already guarded this; the fix hardens the pure conversion for any
  direct caller — refusing, never guessing a unit (consistent with the
  declared-unit discipline).

## [0.9.0] - 2026-07-13

Consolidation release: one credible diagnostic engine behind one unified API.
The endpoint surface shrinks from 54 endpoints (46 tools + 4 resources + 4
prompts) to **36 endpoints (33 tools + 0 resources + 3 prompts)**, every
analysis flows through a single `signal_id` handle, and every number in every
output is either measured, computed, or absent — never invented. **This is a
breaking release** (pre-1.0 semver): see the migration table below.

### Added
- Prognostics MCP endpoints: `analyze_signal_trend` (within-recording
  screening with degradation-onset detection and a truncated feature series)
  and `estimate_rul` (Remaining Useful Life from a *multi-measurement* series —
  explicit `feature_values`+`timestamps` or multiple `signal_ids`). ISO 13374
  Block 5.
- Decision support endpoint `generate_maintenance_recommendations` with a
  closed, canonical `fault_types` vocabulary (`outer_race`, `inner_race`,
  `ball`, `cage`, ...) that raises on unknown values instead of silently
  dropping them. ISO 13374 Block 6.
- `load_signal` accepts a list of file paths for batch loading (fail-fast
  atomic: on the first invalid entry nothing is loaded), plus
  `signal_unit: "g" | "m/s2" | "mm/s" | "m/s"` and `overwrite` parameters.
- `generate_test_signal` writes companion metadata (`sampling_rate`,
  `signal_unit`), auto-registers the signal in the repository and returns a
  `StoredSignalInfo` — immediately analyzable and ISO-assessable.
- Bearing catalog entries carry a mandatory `source` citation (CWRU / XJTU-SY
  provenance), echoed in bearing-check outputs; a geometry-validation test
  suite guards every entry (bore < pitch < OD, ball fit, BPFO/fr + BPFI/fr ≈ Z).
- CI drift guards: every documented tool call in the plugin and MCP prompts is
  validated against the introspected server inventory
  (`tests/test_documented_calls.py`); version strings and endpoint counts are
  pinned across `pyproject.toml`, `src/__init__.py`, `server.json`,
  `CITATION.cff` and the READMEs (`tests/test_version_alignment.py`); a
  surface-parity test maps all 54 v0.8.x endpoints to their destinations.

### Changed
- **One severity engine.** `assess_severity` replaces `evaluate_iso_20816`,
  `assess_vibration_severity`, `check_vibration_alert` and
  `check_custom_vibration_alert`. Input is `signal_id` XOR `rms_velocity_mm_s`
  (portable-instrument route), with native ISO vocabulary
  (`machine_group: 1|2`, `support_type: "rigid"|"flexible"`), optional custom
  `thresholds` and optional `machine_power_kw` (declared power < 15 kW →
  explicit scope refusal). Zone boundaries come from a single table (values
  from ISO 10816-3:2009, 4-zone scheme; outputs note that ISO 20816-3:2022
  merges zones A/B). The invented `machine_class I-IV` mapping is gone.
- **ISO verdicts require a declared unit.** Severity is computed only when the
  signal unit is declared via `load_signal(signal_unit=...)` or companion
  `_metadata.json` — never guessed from amplitude. `diagnose_vibration`
  degrades honestly: the ISO block becomes a structured refusal
  (`status: "refused"` + `reason` + `remedy`) while spectral, bearing and
  anomaly blocks still run. Same discipline for `sampling_rate`: explicit >
  metadata > structured error (no more silent 1 kHz / 10 kHz defaults).
- **One envelope tool.** `analyze_envelope` absorbs
  `compute_envelope_spectrum_tool`: default band 500–5000 Hz, invalid band vs
  Nyquist raises (never a silent clamp), detrend + window before the envelope
  FFT (no more DC skirt over the FTF zone), band echoed in the output.
- **One bearing-fault tool.** `check_bearing_faults` absorbs
  `check_bearing_fault_peak_tool`, `check_bearing_faults_direct` and
  `lookup_bearing_and_compute_tool`: input `bearing_id` XOR
  `frequencies: {label: Hz}` XOR explicit geometry — the frequencies route
  covers gearbox GMF checks and out-of-catalog bearings. Outputs expose the
  canonical fault vocabulary (`fault_type_canonical`).
- **`signal_id` is the universal handle.** Every analysis, diagnostics,
  prognostics and report tool takes `signal_id`; filename parameters are gone.
  Default ids derive from the path relative to the data directory
  (`real_train/baseline_1.csv` → `real_train_baseline_1`), so same-named files
  in different folders no longer collide; reloading an existing id errors
  unless `overwrite=True`.
- Honest field names in prognostics and diagnosis: `fit_r_squared` (was
  `confidence` on RUL fits), `evidence_strength` (categorical, derived from
  corroborating evidence — a quiet machine can no longer score "high"
  confidence from severity alone), `precision_heuristic` (Kalman, explicitly
  labeled heuristic). No tool accepts a `confidence` input anymore.
- Analysis segments are deterministic by default; random sampling is opt-in
  via an explicit `random_seed` parameter.
- `predict_anomalies` returns bounded summaries (counts, score percentiles,
  worst segments) instead of per-segment arrays; its not-found error lists the
  models actually on disk.
- Report filenames are timestamped (consecutive runs no longer overwrite);
  `list_html_reports(file_name=...)` returns per-report metadata (absorbs
  `get_report_info`).
- Parameter naming unified: `rpm` (note: `generate_fft_report`'s old
  `rotation_freq` was in **Hz**; the new `rpm` parameter is in RPM),
  `file_name`, `bearing_id`, `sampling_rate`, `signal_id` — one name per
  concept across the whole surface.
- Error contract unified: misuse and failures raise (surfaced as MCP errors)
  with "problem — actionable remedy" messages; legitimate negative outcomes
  (bearing not in catalog, no degradation trend) are typed results. No more
  error-shaped dicts returned as success.
- All tools are module-level importable functions
  (`from predictive_maintenance_mcp.mcp_tools.analysis_tools import analyze_fft`).
- Kinematic bearing formulas now cite Randall & Antoni (2011) instead of the
  incorrect "ISO 15243" attribution.

### Removed
- **The legacy monolith `machinery_diagnostics_server.py` and the root import
  shims** (`bearing_analyzer`, `iso10816`, `spectral`, `diagnosis_pipeline`,
  `bearing_catalog`). The package ships only the modular server; the entry
  point (`predictive-maintenance-mcp` / `python -m predictive_maintenance_mcp`)
  is unchanged, so existing Claude Desktop configs keep working.
  *Why now instead of the promised v1.0.0*: after the 0.8.1 security patch the
  monolith remained a second, divergent copy of every analysis path — the same
  class of risk that let the path-traversal fix miss half the code in the
  first place. Keeping an unmaintained twin alive for one more minor version
  was a standing security and drift liability; pre-1.0, the deprecation
  promise is superseded by the safety argument.
- The 4 MCP resources (`signal://list`, `signal://read`, `manual://list`,
  `manual://read`) — duplicates of `list_signals`, `get_signal_info`,
  `list_machine_manuals`, `read_manual_excerpt`.
- The Weibull RUL estimator (physically unjustified on vibration features) and
  single-recording RUL extrapolation: `estimate_rul` now refuses anything less
  than 3 timestamped measurements and points to `analyze_signal_trend` for
  within-recording screening.
- The amplitude-based unit-guessing heuristic (RMS > 0.5 → "g"), the
  "HYPOTHESIS/PROCEEDING" flow and the "PLEASE CONFIRM" log walls.
- The hardcoded 81.13 Hz BPFO "example @ 1500 RPM" block that injected
  fictitious reference frequencies into envelope outputs.
- 19 bearing-catalog entries with fabricated internal geometry (the old 6205
  pitch diameter was contaminated from a different bearing); only
  source-verifiable entries remain (6205, 6203 from CWRU; UER204 from XJTU-SY).
- The ASCII-art ISO diagnostic prompt.

### Fixed
- Same reading → same zone: `check_alert_thresholds` and the severity engine
  shared drifted threshold tables (3.0 mm/s, group 2, rigid gave zone C on one
  path and B on the other). One table now feeds every path.
- ISO evaluation refuses when Nyquist < 1 kHz and reports the *real*
  integration band (e.g. "10-950 Hz" at fs = 2 kHz).
- Kalman RUL variance includes the previously missing covariance cross-term.
- Trend direction is gated on the computed p-value (not R² > 0.3); onset
  detection can no longer fire inside its own baseline window.
- Absolute paths outside the data directory load *that* file (previously a
  same-named file inside the data directory could silently win).
- Repository arrays are read-only views — tools can no longer corrupt the
  signal cache in place.
- Envelope band-pass validation (including `generate_envelope_report`, which
  previously crashed with a raw scipy error when the band hit Nyquist).

### Migration table (v0.8.x → v0.9.0)

| v0.8.x endpoint | v0.9.0 destination |
|---|---|
| `evaluate_iso_20816`, `assess_vibration_severity`, `check_vibration_alert`, `check_custom_vibration_alert` | `assess_severity` |
| `compute_envelope_spectrum_tool` | `analyze_envelope` |
| `check_bearing_fault_peak_tool`, `check_bearing_faults_direct`, `lookup_bearing_and_compute_tool` | `check_bearing_faults` |
| `detect_signal_degradation_onset` | `analyze_signal_trend` (onset fields in output) |
| `diagnose_vibration_tool` | `diagnose_vibration` (renamed) |
| `list_stored_signals` | `list_signals(scope="memory")` |
| `clear_signal`, `clear_all_signals` | `clear_signals(signal_id=None)` |
| `get_report_info` | `list_html_reports(file_name=...)` |
| `plot_spectrum` | `generate_fft_report` |
| `plot_envelope` | `generate_envelope_report` |
| `plot_iso_20816_chart` | `generate_iso_report` |
| `signal://list`, `signal://read` (resources) | `list_signals(scope="disk")`, `get_signal_info` |
| `manual://list`, `manual://read` (resources) | `list_machine_manuals`, `read_manual_excerpt` |
| `generate_iso_diagnostic_report` (prompt) | dropped |
| params `filename` / `signal_file` / `signal_path` | `signal_id` (via `load_signal`) |
| params `shaft_speed_rpm` / `operating_speed_rpm` / `rotation_freq` (Hz) | `rpm` |
| param `manual_filename` | `file_name` |
| param `bearing_designation` | `bearing_id` |
| output `confidence` | `fit_r_squared` / `evidence_strength` / `precision_heuristic` |

Scripts that assumed `signal_id == file stem` must switch to the relative-path
derivation (`folder/file.csv` → `folder_file`) or pass an explicit
`signal_id=` to `load_signal`.

## [0.8.1] - 2026-07-10

Security-only patch release. No new features or API changes.

### Security
- **Path traversal fixed across every model and report file path** (all sites, in
  both the modular server and the legacy monolith). `train_anomaly_model` built its
  pickle output path from an unvalidated `model_name` — an arbitrary-file-write
  primitive reachable from any MCP client — and the model-load and
  `read_report_metadata` read paths were likewise unvalidated. Every user-supplied
  filesystem path now flows through a single canonical `path_safety` helper that
  uses `Path.is_relative_to` for containment (closing the sibling-directory bypass
  a `str.startswith` check would miss) and validates model names before any I/O.
  Unsafe names are rejected with a clear error and no file is written or read.
- **Signal read path contained.** `load_signal_data` (the shared read sink behind
  `analyze_fft`, `predict_anomalies`, and every signal tool) now resolves the
  user-supplied filename inside the data directory, closing an arbitrary
  file-content read reachable from any MCP client. Broader signal-path hardening
  (companion-metadata resolution and per-tool existence checks) is tracked as a
  follow-up.

## [0.8.0] - 2026-03-29

### Added
- **Claude Code plugin** — distributable plugin with 7 domain skills (bearing diagnosis, gear analysis, quick screening, report generation, anomaly detection, signal management, documentation search), 2 autonomous agents (`diagnostic-pipeline`, `signal-explorer`), and 3 slash commands (`/pm-diagnose`, `/pm-screen`, `/pm-report`). Installable from the Claude Code marketplace.
- **Prognostics sub-package** — `src/prognostics/` with `RULEstimator` (Remaining Useful Life) and `TrendAnalyzer` with confidence intervals — ISO 13374 Block 5 implementation.
- **Decision support sub-package** — `src/decision_support/` with `AlertManager`, `MaintenanceRecommendations`, and evidence-based `DiagnosisPipeline` — ISO 13374 Block 6 advisory layer.
- **Phase 1 modular refactoring** — monolithic server split into `mcp_tools/` (acquisition, analysis, diagnostics, report, prompts), `signal_acquisition/`, `signal_processing/`, `diagnostics/`, `prognostics/`, `decision_support/` sub-packages following ISO 13374 six-block architecture.
- **Jupyter notebooks** — 3 interactive notebooks: getting started, bearing diagnostics, condition monitoring.
- **Guided workflow prompts** — 4 MCP prompt endpoints: `diagnose_bearing_prompt`, `diagnose_gear_prompt`, `quick_diagnostic_report_prompt`, `analyze_anomalies_prompt`.

### Changed
- Total MCP endpoints expanded to **48** (from 24).
- README strategically redesigned: pitch-first, 60% shorter, engineer/developer split quickstarts.
- GitHub Pages updated: Tools at a Glance section, ISO 13374 six-block architecture diagram, standards compliance strip, Prognostics & Decision Intelligence feature cards.
- Test coverage at **86%** across 20+ test files.

## [0.7.1] - 2025-07-15

### Added
- **SSE & Streamable-HTTP transport** — `main()` now accepts `--transport sse|streamable-http` (or env var `MCP_TRANSPORT`) via argparse CLI, enabling remote HTTPS deployment for Microsoft Copilot Studio and other networked MCP clients. `--host` / `--port` (or `MCP_HOST` / `MCP_PORT`) configure the listen address.
- **Docker Compose + Caddy** — New `docker-compose.yml` with `mcp-server` service (SSE by default) and commented-out Caddy reverse proxy for automatic Let's Encrypt HTTPS certificates. New `Caddyfile` template.
- **HTTPS Deployment guide** — New `docs/DEPLOYMENT.md` covering local SSE testing, Docker Compose + Caddy auto-TLS, nginx reverse proxy, Azure/cloud deployment, Copilot Studio connection, and CLI reference.

### Changed
- **Dockerfile** rebuilt for SSE default — installs `uvicorn`, sets `MCP_TRANSPORT=sse`, `MCP_HOST=0.0.0.0`, `EXPOSE 8000`
- README updated: enterprise-ready features, Copilot Studio mention, deployment docs table, roadmap progress
- Version bumped to 0.7.1

## [0.7.0] - 2025-07-14

### Added
- **FAISS vector search** — `search_documentation` now uses FAISS + sentence-transformers for semantic retrieval when installed (`pip install predictive-maintenance-mcp[vector-search]`). Falls back to TF-IDF keyword search when not installed. Dual-backend `DocumentIndex` in `src/rag.py`.
- **OCR for scanned PDFs** — `document_reader.extract_text_from_pdf()` automatically falls back to Tesseract OCR for pages with empty/minimal text. Requires optional `pytesseract` + `pdf2image` + Poppler.
- **DOCX diagnostic reports** — New `generate_diagnostic_report_docx` MCP tool and `save_diagnostic_report_docx()` in report generator. Creates structured Word documents with statistics tables, FFT/envelope peaks, bearing frequencies, ISO evaluation, and diagnostic summary. Requires optional `python-docx`.
- **New optional dependency groups** in `pyproject.toml`: `vector-search`, `ocr`, `docx`. The `full` extra now includes all of them.
- **Overlapping chunking** — New `chunk_text()` helper in RAG module for character-level overlapping chunks alongside paragraph-aware chunking.

### Changed
- **`search_documentation`** now reports active backend (`faiss` or `tfidf`) in response
- **27 MCP tools** (was 26) — added `generate_diagnostic_report_docx`
- Version bumped to 0.7.0

## [0.6.0] - 2025-07-08

### Added
- **RAG-based document search** — New `search_documentation` MCP tool using TF-IDF indexing over machine manuals and bearing catalogs (`src/rag.py`)
- **`SpectralPeak` model** — Structured representation for individual FFT peaks (frequency, magnitude, dB, annotation)

### Changed
- **Compact FFT output** — `analyze_fft` now returns top-20 peaks + RMS/stats instead of full frequency/magnitude arrays (~200 KB → ~2 KB per call), eliminating LLM context overflow
- **Compact signal resource** — `read_signal_file` returns metadata + statistics only (no raw samples), preventing large JSON payloads
- **Server instructions** updated with output-efficiency policy and RAG documentation guidance
- **`pypdf`** promoted from optional to required dependency

### Fixed
- LLM "output too long" errors caused by full-array serialisation in `FFTResult`

## [0.5.0] - 2025-02-16

### Added
- **Multi-format signal loading** — `load_signal_data()` now supports CSV, TXT, NPY, MAT (MATLAB), WAV, and Parquet formats
- **`__main__.py`** — Server can now be run as `python -m predictive_maintenance_mcp`
- **Ollama Guide** added to documentation table in README

### Changed
- **Unified signal loading** — All 16 `pd.read_csv()` call sites refactored to use `load_signal_data()`, enabling all tools to accept any supported format
- **ML code deduplication** — Extracted 4 helper functions (`_resolve_sampling_rate`, `_segment_and_extract_features`, `_extract_features_from_files`, `_extract_and_transform_validation_features`), reducing ~163 statements in `train_anomaly_model`
- **ISO metadata consolidation** — `evaluate_iso_20816` now reads metadata file once instead of twice
- **PyPDF2 → pypdf migration** — Replaced deprecated PyPDF2 with pypdf in `document_reader.py`
- **Pytest config consolidation** — Merged `pytest.ini` into `pyproject.toml` (`[tool.pytest.ini_options]`)
- **Logging to stderr** — Server logging now uses `stderr` to avoid polluting MCP stdio transport
- **Report filenames** — `report_generator.py` uses `Path.stem` for all filename sanitizations

### Fixed
- **Packaging** — Corrected `pyproject.toml` package-dir mapping (`src/` → `predictive_maintenance_mcp`)
- **Metadata paths** — `get_metadata_path()` now uses `Path.stem` to work with all signal extensions
- **Plot output directories** — Report generator creates output directories before writing files
- **Flaky ML test** — Fixed `test_predict_anomalies` instability with deterministic seed

## [0.4.1] - 2026-02-15

### Fixed
- Aligned `__version__` in `src/__init__.py` and `SERVER_VERSION` in `.env.example` to 0.4.x (were still 0.3.4 after merge)
- Shortened `server.json` description to ≤100 characters (MCP Registry validation requirement)
- Added `0.4.x` to supported versions table in `SECURITY.md`
- PyPI publish: v0.4.0 was uploaded with stale `__init__` version; this release corrects it

## [0.4.0] - 2026-02-15

### Added
- **Persona-Based Documentation System**
  - New `docs/QUICKSTART_ENGINEER.md` — Zero-code guide for maintenance and reliability engineers
  - New `docs/QUICKSTART_DEVELOPER.md` — Architecture guide for AI/software developers with tutorial on creating new MCP tools
  - "Choose Your Path" section in README with two clear entry points

- **"Our Mission" Section in README**
  - Project vision and purpose integrated directly into the repository (previously only on external blog post)
  - Explains the "why" of MCP for industrial diagnostics

- **Ecosystem Architecture Overview**
  - Visual diagram explaining the MCP flow: User → LLM → MCP Server → Data
  - Explanation of MCP as "USB port for AI" — plug-and-play tool integration
  - Clarifies the Resource vs Tool pattern

- **GitHub Issue Templates**
  - Bug Report template with environment details
  - Feature Request template with impact assessment
  - Good First Issue template with effort estimates and mentorship links
  - Domain Validation template for engineers to provide expert feedback (no code required)
  - Pull Request template with standardized checklist
  - Issue template config with contact links to Discussions and guides

- **Revamped CONTRIBUTING.md with Four Contribution Paths**
  - Path 1: Domain Expert (no code required — validate results, provide datasets, review diagnostics)
  - Path 2: Software Developer (add tools, improve architecture, build Docker support)
  - Path 3: Technical Writer (tutorials, translations, case studies)
  - Path 4: Tester / QA (edge cases, cross-platform, ground truth validation)

- **Actionable Roadmap**
  - Roadmap items now link to GitHub Issues/Discussions
  - Priority-based table with Get Involved column
  - Docker image for zero-install setup added as high-priority item

### Changed
- **README.md completely restructured** — Mission → Architecture → Choose Your Path → Content
  - Moved from purely technical README to narrative + technical hybrid
  - Added Documentation table linking all guides by audience
  - Consolidated support links (Issues, Discussions, Blog post) in dedicated section
- **CONTRIBUTING.md rewritten** — From generic PR guide to persona-based contribution manifesto
- Version bump from 0.3.2 to 0.4.0

## [0.2.0] - 2025-11-11

### Added
- **Professional HTML Report Generation System**
  - Interactive Plotly visualizations with modern, responsive design
  - `generate_fft_report()` - FFT spectrum analysis with peak detection
  - `generate_envelope_report()` - Bearing fault detection with frequency markers
  - `generate_iso_report()` - ISO 20816-3 compliance evaluation with zone charts
  - `list_html_reports()` - List all generated reports with metadata
  - `get_report_info()` - Extract metadata without loading full HTML

- **Real Bearing Vibration Dataset**
  - 20 production-quality signals from real machinery tests (train: 14, test: 6)
  - 3 healthy baselines, 7 inner race faults, 10 outer race faults
  - Sampling rates: 48.8-97.7 kHz, durations: 3-6 seconds (varies by signal)
  - Complete metadata with bearing frequencies (BPFO, BPFI, BSF, FTF)

- **Advanced Diagnostics**
  - Evidence-based bearing diagnostic workflow (`diagnose_bearing`)
  - Gear fault detection workflow (`diagnose_gear`)
  - ISO 20816-3 vibration severity assessment
  - Automatic acceleration→velocity conversion

- **Machine Learning Tools**
  - `extract_features_from_signal()` - 17+ statistical features
  - `train_anomaly_model()` - OneClassSVM/LocalOutlierFactor training
  - `predict_anomalies()` - Anomaly detection with confidence scores

- **Comprehensive Test Suite**
  - 80%+ test coverage
  - Real data validation tests
  - CI/CD pipeline with GitHub Actions
  - Automated code quality checks (pytest, flake8, mypy, black)

### Changed
- Migrated from inline HTML artifacts to file-based reports
- Optimized signal processing algorithms for accuracy and performance
- Enhanced documentation with step-by-step tutorials
- Improved diagnostic accuracy with evidence-based workflows

### Fixed
- Signal processing edge cases
- Peak detection accuracy
- ISO 20816-3 zone classification
- Metadata handling for various signal formats

## [0.1.0] - 2025-11-01

### Added
- Initial release of Predictive Maintenance MCP Server
- Core vibration analysis tools (FFT, envelope, statistics)
- Basic MCP server implementation with FastMCP
- Sample signal generation
- Initial documentation and examples

---

## Roadmap

### Planned for v0.7.0
- **📦 Docker image** for zero-install setup
- **📏 Customizable ISO report thresholds**
- Multi-signal comparison tools
- Advanced trending and monitoring
- Additional diagnostic workflows (pumps, motors, gearboxes)
- Extended dataset with more fault types

### Future Enhancements
- Real-time signal streaming support
- Cloud integration options
- Dashboard for multi-asset monitoring
- Mobile-friendly report viewing
- Integration with industrial IoT platforms
- **Multimodal diagnostics**: Combine vibration, temperature, acoustic data
