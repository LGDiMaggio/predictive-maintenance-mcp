"""The approved public texts of the chronos2 backend, typed out independently
of the code so that a test compares the code's text with the approved one.

Placeholders in braces are filled with ``str.format`` by the tests.
"""

METHOD_NOTE = (
    "Chronos-2 backend: P50 is the 0.5 quantile of the pinned Chronos-2 model "
    "(amazon/chronos-2, revision 29ec3766), run zero-shot on at most the most recent "
    "2,048 bins of the supplied history. It is a projection of the indicator's values "
    "from that history, not a prediction that the indicator will reach any limit."
)

INTERVAL_NOTE = (
    "P10 and P90 are quantiles produced by the model, not a band computed from the "
    "differences between the supplied values, and their coverage on this series is "
    "not validated. In the evaluation behind this backend, on two indicators sampled "
    "every 10 seconds and every 10 minutes, the band contained fewer outcomes than "
    "its nominal level on both, and fewer still in some conditions: at the first "
    "projected steps and at high operating load on one indicator, and late in a "
    "degradation record on the other. Other sampling cadences were not evaluated. "
    "The band is not a probability of crossing any limit and must not be compared "
    "with alarm or ISO zone boundaries."
)

REFERENCE_DESCRIPTION = (
    "Backend 'chronos2' only: the reference backend's projection from the full "
    "supplied history at the forecast's anchor steps, for comparing the model with "
    "a simple reference at those steps. Not a per-step comparison."
)
REFERENCE_BACKEND_DESCRIPTION = (
    "Reference backend used: 'seasonal_naive' when season_length_steps was given, "
    "otherwise 'naive'"
)
REFERENCE_ANCHORS_DESCRIPTION = (
    "The reference's P10, P50 and P90 at the forecast's anchor steps, with absolute "
    "bin times: four steps when the horizon is 5 steps or more, and every distinct "
    "anchor step for shorter horizons"
)
REFERENCE_NOTE = (
    "Reference anchors: the {reference_backend} reference's P10, P50 and P90 at the "
    "forecast's anchor steps, computed from the whole supplied history. They show "
    "where the model and a simple reference differ at those steps only. For every "
    "step, call again with backend '{reference_backend}'."
)

CONTEXT_TRUNCATED = (
    "Only the most recent 2,048 of the {grid_bins} bins in the supplied history were "
    "passed to the model; older values did not influence its projection. The "
    "reference anchors use the whole supplied history."
)
RECENT_GAP = (
    "{empty} of the last {window_tail} bins before the forecast origin are empty. "
    "With little recent information, the model's projection can depend more on "
    "older values than on the indicator's current behaviour."
)

SIGN = {
    "positive": (
        "Every observed value is positive, but P10 falls below zero: the model's "
        "quantiles do not respect the indicator's sign."
    ),
    "negative": (
        "Every observed value is negative, but P90 rises above zero: the model's "
        "quantiles do not respect the indicator's sign."
    ),
    "zero_or_positive": (
        "Every observed value is zero or positive, but P10 falls below zero: the "
        "model's quantiles do not respect the indicator's sign."
    ),
    "zero_or_negative": (
        "Every observed value is zero or negative, but P90 rises above zero: the "
        "model's quantiles do not respect the indicator's sign."
    ),
}

TOO_FEW_OBSERVED = (
    "backend 'chronos2' needs at least 64 observed bins within the most recent 2,048 "
    "bins of the grid; this history has {observed_in_window}. Supply a longer history "
    "at this resolution, or use backend 'naive', or 'seasonal_naive' with a declared "
    "season. These are operating limits of this backend: it was evaluated only on "
    "indicators sampled every 10 seconds and every 10 minutes, and other sampling "
    "cadences were not evaluated."
)
FLAT_WINDOW = (
    "backend 'chronos2' needs variation within the most recent 2,048 bins of the "
    "grid, but all {observed_in_window} observed bins there hold the same value, so "
    "the model's band would come from numerical rounding rather than from the data "
    "and could read as certainty. A flat stretch can come from a stuck sensor, a "
    "held or interpolated value, or an indicator that did not change at the recorded "
    "resolution; it is not evidence about the condition of the machine. Check the "
    "source of the values, or use backend 'naive' or 'seasonal_naive'."
)
HORIZON_ABOVE_144 = (
    "horizon_steps must be at most 144 with backend 'chronos2', got {horizon_steps}. "
    "Call again with horizon_steps of at most 144, or use backend 'naive' or "
    "'seasonal_naive' for a longer horizon. The 144-step limit is an operating limit "
    "of this backend: it was evaluated only on indicators sampled every 10 seconds "
    "and every 10 minutes, and other sampling cadences were not evaluated."
)
HORIZON_BEYOND_SPAN = (
    "The requested horizon of {required} steps ({duration}) is longer than the "
    "history allows: at most {max_supported} steps can be projected at resolution "
    "{resolution}, because with backend 'chronos2' the forecast may not span more "
    "than the observed history, 144 steps or 30 days."
)

EXTRA_NOT_INSTALLED = (
    "backend 'chronos2' is not available on this server: the optional "
    "'forecast-chronos2' dependencies are not installed. An operator can install "
    "them as described in the forecasting model guide; on Linux, install the "
    "CPU-only build of torch first to avoid the CUDA libraries. The extra cannot be "
    "installed on macOS with an Intel processor or on Python 3.15 or later, because "
    "no torch build in the supported range is published for them. Until then, use "
    "backend 'naive' or 'seasonal_naive'."
)
DIR_NOT_SET = (
    "backend 'chronos2' is not available: PMM_FORECAST_MODEL_DIR is not set. An "
    "operator must download amazon/chronos-2 at revision "
    "29ec3766d36d6f73f0696f85560a422f50e8498c into a local directory, check its "
    "SHA-256 values as described in the forecasting model guide, and set "
    "PMM_FORECAST_MODEL_DIR to that directory. The server never downloads model "
    "files itself. Until then, use backend 'naive' or 'seasonal_naive'."
)
NOT_A_MODEL_DIR = (
    "backend 'chronos2' is not available: PMM_FORECAST_MODEL_DIR is {path}, which is "
    "not a local directory holding config.json and model.safetensors. Download "
    "amazon/chronos-2 at revision 29ec3766d36d6f73f0696f85560a422f50e8498c into that "
    "directory as described in the forecasting model guide, or point "
    "PMM_FORECAST_MODEL_DIR to the directory that holds it. The server never "
    "downloads model files itself."
)
DIGEST_MISMATCH = (
    "backend 'chronos2' did not load the model: {file} in PMM_FORECAST_MODEL_DIR does "
    "not match the SHA-256 published for revision 29ec3766 of amazon/chronos-2, so "
    "the files cannot be verified as that revision. Download that exact revision "
    "again into an empty directory and check both SHA-256 values as described in the "
    "forecasting model guide. Unverified model files are never loaded."
)
VERSION_OUTSIDE_RANGE = (
    "backend 'chronos2' is not available: the installed {package} {version} is "
    "outside the range this backend supports ({range}). Reinstall the "
    "'forecast-chronos2' extra so that its declared versions are used, as described "
    "in the forecasting model guide. Until then, use backend 'naive' or "
    "'seasonal_naive'."
)
LOAD_FAILED = (
    "backend 'chronos2' could not load the verified model files. The server log "
    "records the cause. Check that the 'forecast-chronos2' dependencies are "
    "installed as described in the forecasting model guide, then restart the "
    "server. Until then, use backend 'naive' or 'seasonal_naive'."
)
STILL_LOADING = (
    "backend 'chronos2' is still loading its model in the background, and no "
    "projection was made from this call. The first load can take from several "
    "seconds to more than a minute, depending on the machine and the disk. Call "
    "again in about a minute. An operator can set PMM_FORECAST_MODEL_PRELOAD=1 so "
    "that the model loads when the server starts."
)
REFERENCE_NOT_COMPUTABLE = (
    "backend 'chronos2' returns a projection only together with its reference "
    "anchors, and the {reference_backend} reference cannot be computed for this "
    "history: {reference_reason} {reference_remedy} No model projection was made "
    "from this call."
)
REFERENCE_BAND_NOT_REPORTABLE = (
    "backend 'chronos2' returns a projection only together with its reference "
    "anchors, and the {reference_backend} reference's band cannot be reported: "
    "{reference_refusal}"
)

DIFFERENCE_LAG_DESCRIPTION = (
    "Lag in steps of the difference pairs the backend uses; for 'chronos2', the "
    "pairs its reference backend uses"
)

FAMILY_DESCRIPTION = (
    "Algorithm family: 'indicator_forecast' for the reference backends, "
    "'indicator_forecast_chronos2' for backend 'chronos2'"
)
ALGORITHM_VERSION_APPENDED = (
    "For 'indicator_forecast_chronos2', the same inputs give the same numbers only "
    "with the same model files, library versions, thread setting and platform; these "
    "are recorded in the model block, and other settings can change the last digits."
)
MODEL_DESCRIPTION = (
    "Backend 'chronos2' only: the verified model files and the libraries that "
    "produced the numbers. Version ranges are compatibility boundaries, not evidence "
    "that every combination within them was tested; the versions recorded here are "
    "the ones actually used."
)
MODEL_FIELDS = {
    "repo_id": "Hugging Face repository of the model: 'amazon/chronos-2'",
    "revision": "Revision whose files were verified by SHA-256 before loading",
    "weights_sha256": "SHA-256 of model.safetensors, verified before loading",
    "config_sha256": "SHA-256 of config.json, verified before loading",
    "weights_licence": (
        "Licence declared for the model weights in the model card metadata of "
        "amazon/chronos-2: Apache-2.0; the repository ships no LICENSE file"
    ),
    "chronos_forecasting_version": "Installed version used for this projection",
    "torch_version": "Installed version used for this projection",
    "transformers_version": "Installed version used for this projection",
    "accelerate_version": "Installed version used for this projection",
    "huggingface_hub_version": "Installed version used for this projection",
    "torch_threads": "PyTorch threads, as read inside the inference thread",
    "context_bins": "Grid bins passed to the model, at most 2,048",
}

DOCSTRING_OPENING = (
    "and projects it forward on a regular UTC grid with a statistical reference "
    "method or, where an operator has installed it, the Chronos-2 forecasting model, "
    "returning P10, P50 and P90 per step."
)
DOCSTRING_BACKEND = (
    "backend: 'naive' (default; repeats the last observed bin), 'seasonal_naive' "
    "(repeats the same phase of the declared season) or 'chronos2' (the optional "
    "Chronos-2 model; available only where an operator has installed and configured "
    "it, otherwise refused with a remedy). With 'chronos2' the history needs at least "
    "64 observed bins within its most recent 2,048 bins, horizon_steps is at most "
    "144, older bins are not used by the model, and the result carries the reference "
    "backend's P10, P50 and P90 at the forecast's anchor steps; season_length_steps, "
    "when given, selects the seasonal_naive reference. The first 'chronos2' call in a "
    "server process may be refused while the model loads; call again as the refusal "
    "says."
)
DOCSTRING_RETURNS = (
    "With 'chronos2', a call whose reference cannot be computed is refused rather "
    "than answered without its reference anchors."
)

INSTRUCTION_BULLETS = (
    "- forecast_asset_indicator projects one indicator series the user supplies (ISO "
    "8601 instants with Z or a UTC offset, values, a declared unit and resolution) "
    "with a statistical reference method (naive, seasonal_naive) or, where installed, "
    "the Chronos-2 model (chronos2), and returns P10, P50 and P90 per step or a typed "
    "insufficient-history result; it reads only the series passed in the call",
    "- Quote its interpretation, method_note and interval_note fields: the "
    "projection is not a fault detection, a failure probability or a "
    "remaining-useful-life estimate, and its band is nominal for the reference "
    "methods (naive, seasonal_naive); with chronos2 the band comes from the model, "
    "and its coverage on the series is not validated",
    "- With chronos2, report the reference anchors beside the model's values and say "
    "that they cover a few steps only, not every step; if a chronos2 call is refused "
    "because the model is still loading, tell the user and call again later rather "
    "than switching backend silently",
)

#: The operator guide, paragraph by paragraph and bullet by bullet, as approved
#: (docs/guides/forecast-model-backend.md).
GUIDE = (
    "Forecasting model guide (optional Chronos-2 backend)",
    "The chronos2 backend of forecast_asset_indicator projects one indicator series "
    "with the Chronos-2 forecasting model, run on this machine. It forecasts the "
    "indicator's values only: it is not a fault detection, a failure probability, a "
    "time to a limit or a remaining-useful-life estimate. It was evaluated only on "
    "indicators sampled every 10 seconds and every 10 minutes; other sampling "
    "cadences were not evaluated.",
    "Windows x64, Linux x86_64 or aarch64, or macOS 14 or later on Apple silicon. "
    "Tested on Windows with Python 3.11 and 3.12; Python 3.13 and 3.14 have matching "
    "torch builds but were not tested. Not available on macOS with an Intel "
    "processor.",
    "About 0.6 GB more disk on Windows with the extra.",
    "On Linux, about 2.9 GB of downloads with the default PyPI torch; the CPU-only "
    "torch build is about 0.2 GB instead of the CUDA build.",
    "478 MB of model files.",
    "About 0.7 GB more server memory once the model is used.",
    "The forecast-chronos2 extra pins chronos-forecasting 2.3.2 and accepts torch "
    "2.13 or 2.14 and transformers 5.14 to 5.20. These ranges are compatibility "
    "boundaries, not a statement that every version in them was tested; each "
    "result's provenance records the versions actually used. If a different version "
    "is installed, the chronos2 backend refuses with a remedy, and the naive and "
    "seasonal_naive backends keep working.",
    "In a lock file that resolves all extras together, such as this project's "
    "uv.lock, these bounds also hold the torch and transformers versions used with "
    "the full extra until the supported range is widened. A plain pip install of the "
    "full extra is not affected.",
    'pip install "torch>=2.13,<2.15" --index-url https://download.pytorch.org/whl/cpu',
    'pip install "predictive-maintenance-mcp[forecast-chronos2]"',
    "uv sync --extra forecast-chronos2 installs the versions in the project's lock "
    "file, which on Linux include the CUDA build of torch: use pip or uv pip install "
    "for the CPU-only route.",
    "The server never downloads the model files. An operator downloads them once, at "
    "the pinned revision; the hf command is installed with the forecast-chronos2 "
    "extra:",
    "hf download amazon/chronos-2 config.json model.safetensors --revision "
    "29ec3766d36d6f73f0696f85560a422f50e8498c --local-dir <DIR>",
    "Check both files against these SHA-256 values before use:",
    "config.json: ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031 "
    "(1,067 bytes)",
    "model.safetensors: "
    "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 "
    "(477,930,472 bytes)",
    "On Windows use Get-FileHash -Algorithm SHA256 <file>; on Linux and macOS use "
    "sha256sum <file>. Keep <DIR> outside cloud-synced folders. For a machine without "
    "internet access, copy the directory and check the same values. Then set "
    "PMM_FORECAST_MODEL_DIR to <DIR>. If a file is missing or does not match, the "
    "chronos2 backend refuses to load it.",
    "The model card metadata of amazon/chronos-2 on Hugging Face declares the "
    "Apache-2.0 licence for the weights; the repository ships no separate LICENSE "
    "file. PMM does not include or redistribute the weights. Licence metadata as "
    "shown on Hugging Face on {licence_check_date}.",
    "The first chronos2 call in a server process loads the model in the background. "
    "If the model is not ready within PMM_FORECAST_MODEL_LOAD_WAIT_S seconds "
    "(default 30), that call is refused with a request to call again, and the load "
    "continues. Set PMM_FORECAST_MODEL_PRELOAD=1 to start loading when the server "
    "starts; this may reduce first-call waiting if loading completes before the "
    "first chronos2 request. Set PMM_FORECAST_MODEL_LOAD_WAIT_S to 0 to refuse at "
    "once while the model is loading, or up to 120 if your client allows longer tool "
    "calls.",
    "On the laptop used for testing, warm calls without the output-size retry search "
    "took under one second; the retry search took longer.",
    "PMM_FORECAST_MODEL_DIR: the local directory holding config.json and "
    "model.safetensors of amazon/chronos-2 at revision "
    "29ec3766d36d6f73f0696f85560a422f50e8498c. Without it, the chronos2 backend is "
    "unavailable.",
    "PMM_FORECAST_MODEL_THREADS: the number of PyTorch threads used for each "
    "projection, from 1 to 64 (default 4).",
    "PMM_FORECAST_MODEL_PRELOAD: set to 1 to load the model when the server starts "
    "(default 0).",
    "PMM_FORECAST_MODEL_LOAD_WAIT_S: how many seconds a call waits for a load in "
    "progress, from 0 to 120 (default 30).",
    INTERVAL_NOTE,
    TOO_FEW_OBSERVED,
    HORIZON_ABOVE_144,
    CONTEXT_TRUNCATED,
    RECENT_GAP,
    "Each chronos2 result also carries the reference backend's P10, P50 and P90 at "
    "the forecast's anchor steps; to see the full reference, call "
    "forecast_asset_indicator again with backend 'naive', or 'seasonal_naive' when a "
    "season is declared.",
    "The torch and transformers ranges are maintained, not frozen. To support a "
    "newer release: (1) install the candidate versions in an isolated environment; "
    "(2) run the gated numerical-equivalence test and the forecast regression tests; "
    "(3) check security advisories, platform wheels and that the default and full "
    "installations resolve as before; (4) widen the ranges, and the recorded "
    "evidence, only if those checks pass; (5) record the resolved versions and the "
    "result. A dependency update does not require repeating the original evaluation "
    "study. If the ranges block a security-critical update of the full extra, the "
    "conflict is raised to the maintainers at once rather than postponed or bypassed "
    "with an untested upgrade.",
)

CHANGELOG_ENTRY = (
    "forecast_asset_indicator accepts an optional chronos2 backend that projects the "
    "indicator with the Chronos-2 forecasting model, run locally, when an operator "
    "installs the forecast-chronos2 extra and the model files. It is never the "
    "default. Each result carries the reference backend's values at the forecast's "
    "anchor steps, and a call whose reference cannot be computed is refused. History "
    "is limited to the most recent 2,048 bins (at least 64 observed) and horizons to "
    "144 steps; the model was evaluated only on indicators sampled every 10 seconds "
    "and every 10 minutes. The projection is not a fault detection, a failure "
    "probability or a remaining-useful-life estimate.",
    "The extra is not part of full or the Docker image. Model files are fetched by "
    "the operator at a pinned revision and verified by SHA-256 before loading; the "
    "server never downloads them.",
    "The extra pins chronos-forecasting 2.3.2 and bounds torch and transformers; in "
    "the project's lock file these bounds also apply to the versions shared with the "
    "full extra.",
)

CATALOG_ROW = (
    "| `forecast_asset_indicator` | Projection of one indicator series you supply "
    "(ISO 8601 instants with an offset, declared unit and resolution) with a naive or "
    "seasonal-naive reference method or, where an operator has installed it, the "
    "Chronos-2 forecasting model: P10, P50 and P90 per step. A projection only, never "
    "a fault finding, a failure probability or a remaining life. Unreleased: main "
    "branch only |"
)
