# Forecasting model guide (optional Chronos-2 backend)

## 1. What it is

The chronos2 backend of forecast_asset_indicator projects one indicator series with the Chronos-2 forecasting model, run on this machine. It forecasts the indicator's values only: it is not a fault detection, a failure probability, a time to a limit or a remaining-useful-life estimate. It was evaluated only on indicators sampled every 10 seconds and every 10 minutes; other sampling cadences were not evaluated.

## 2. Requirements

- Windows x64, Linux x86_64 or aarch64, or macOS 14 or later on Apple silicon. Tested on Windows with Python 3.11 and 3.12; Python 3.13 and 3.14 have matching torch builds but were not tested. Not available on macOS with an Intel processor.
- About 0.6 GB more disk on Windows with the extra.
- On Linux, about 2.9 GB of downloads with the default PyPI torch; the CPU-only torch build is about 0.2 GB instead of the CUDA build.
- 478 MB of model files.
- About 0.7 GB more server memory once the model is used.

The forecast-chronos2 extra pins chronos-forecasting 2.3.2 and accepts torch 2.13 or 2.14 and transformers 5.14 to 5.20. These ranges are compatibility boundaries, not a statement that every version in them was tested; each result's provenance records the versions actually used. If a different version is installed, the chronos2 backend refuses with a remedy, and the naive and seasonal_naive backends keep working.

In a lock file that resolves all extras together, such as this project's uv.lock, these bounds also hold the torch and transformers versions used with the full extra until the supported range is widened. A plain pip install of the full extra is not affected.

## 3. Install

```
pip install "torch>=2.13,<2.15" --index-url https://download.pytorch.org/whl/cpu
pip install "predictive-maintenance-mcp[forecast-chronos2]"
```

`uv sync --extra forecast-chronos2` installs the versions in the project's lock file, which on Linux include the CUDA build of torch: use `pip` or `uv pip install` for the CPU-only route.

## 4. Get the model files

The server never downloads the model files. An operator downloads them once, at the pinned revision; the hf command is installed with the forecast-chronos2 extra:

```
hf download amazon/chronos-2 config.json model.safetensors --revision 29ec3766d36d6f73f0696f85560a422f50e8498c --local-dir <DIR>
```

Check both files against these SHA-256 values before use:

- config.json: ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031 (1,067 bytes)
- model.safetensors: ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42 (477,930,472 bytes)

On Windows use `Get-FileHash -Algorithm SHA256 <file>`; on Linux and macOS use `sha256sum <file>`. Keep `<DIR>` outside cloud-synced folders. For a machine without internet access, copy the directory and check the same values. Then set PMM_FORECAST_MODEL_DIR to `<DIR>`. If a file is missing or does not match, the chronos2 backend refuses to load it.

## 5. Licence

The model card metadata of amazon/chronos-2 on Hugging Face declares the Apache-2.0 licence for the weights; the repository ships no separate LICENSE file. PMM does not include or redistribute the weights. Licence metadata as shown on Hugging Face on {licence_check_date}.

## 6. First call, performance and settings

The first chronos2 call in a server process loads the model in the background. If the model is not ready within PMM_FORECAST_MODEL_LOAD_WAIT_S seconds (default 30), that call is refused with a request to call again, and the load continues. Set PMM_FORECAST_MODEL_PRELOAD=1 to start loading when the server starts; this may reduce first-call waiting if loading completes before the first chronos2 request. Set PMM_FORECAST_MODEL_LOAD_WAIT_S to 0 to refuse at once while the model is loading, or up to 120 if your client allows longer tool calls.

On the laptop used for testing, warm calls without the output-size retry search took under one second; the retry search took longer.

- PMM_FORECAST_MODEL_DIR: the local directory holding config.json and model.safetensors of amazon/chronos-2 at revision 29ec3766d36d6f73f0696f85560a422f50e8498c. Without it, the chronos2 backend is unavailable.
- PMM_FORECAST_MODEL_THREADS: the number of PyTorch threads used for each projection, from 1 to 64 (default 4).
- PMM_FORECAST_MODEL_PRELOAD: set to 1 to load the model when the server starts (default 0).
- PMM_FORECAST_MODEL_LOAD_WAIT_S: how many seconds a call waits for a load in progress, from 0 to 120 (default 30).

## 7. Limits

> P10 and P90 are quantiles produced by the model, not a band computed from the differences between the supplied values, and their coverage on this series is not validated. In the evaluation behind this backend, on two indicators sampled every 10 seconds and every 10 minutes, the band contained fewer outcomes than its nominal level on both, and fewer still in some conditions: at the first projected steps and at high operating load on one indicator, and late in a degradation record on the other. Other sampling cadences were not evaluated. The band is not a probability of crossing any limit and must not be compared with alarm or ISO zone boundaries.

> backend 'chronos2' needs at least 64 observed bins within the most recent 2,048 bins of the grid; this history has {observed_in_window}. Supply a longer history at this resolution, or use backend 'naive', or 'seasonal_naive' with a declared season. These are operating limits of this backend: it was evaluated only on indicators sampled every 10 seconds and every 10 minutes, and other sampling cadences were not evaluated.

> horizon_steps must be at most 144 with backend 'chronos2', got {horizon_steps}. Call again with horizon_steps of at most 144, or use backend 'naive' or 'seasonal_naive' for a longer horizon. The 144-step limit is an operating limit of this backend: it was evaluated only on indicators sampled every 10 seconds and every 10 minutes, and other sampling cadences were not evaluated.

> Only the most recent 2,048 of the {grid_bins} bins in the supplied history were passed to the model; older values did not influence its projection. The reference anchors use the whole supplied history.

> {empty} of the last {window_tail} bins before the forecast origin are empty. With little recent information, the model's projection can depend more on older values than on the indicator's current behaviour.

Each chronos2 result also carries the reference backend's P10, P50 and P90 at the forecast's anchor steps; to see the full reference, call forecast_asset_indicator again with backend 'naive', or 'seasonal_naive' when a season is declared.

## 8. Updating the supported versions

The torch and transformers ranges are maintained, not frozen. To support a newer release: (1) install the candidate versions in an isolated environment; (2) run the gated numerical-equivalence test and the forecast regression tests; (3) check security advisories, platform wheels and that the default and full installations resolve as before; (4) widen the ranges, and the recorded evidence, only if those checks pass; (5) record the resolved versions and the result. A dependency update does not require repeating the original evaluation study. If the ranges block a security-critical update of the full extra, the conflict is raised to the maintainers at once rather than postponed or bypassed with an untested upgrade.
