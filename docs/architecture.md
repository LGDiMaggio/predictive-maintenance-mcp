# Architecture Guide

## Overview

predictive-maintenance-mcp is structured around the **ISO 13374 six-block architecture** for condition monitoring and diagnostics. Each block maps to a Python sub-package under `predictive_maintenance_mcp`, providing clear separation of concerns. MCP tools live in `mcp_tools/` (one module per block family) and are registered by the thin `server.py` orchestrator.

## ISO 13374 Block Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (FastMCP)                        │
│                 server.py  +  mcp_tools/*.py                    │
│                   MCP tools, resources, prompts                 │
└──────────┬──────────┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │          │
     ┌─────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐ ┌──▼──────┐
     │ Block 1  │ │Block │ │Block   │ │Block 5 │ │ Block 6 │
     │ Acquisi- │ │  2   │ │ 3 & 4  │ │Prognos-│ │Decision │
     │  tion    │ │Proc. │ │Diagno- │ │ tics   │ │Support  │
     │          │ │      │ │ stics  │ │        │ │         │
     │signal_   │ │signal│ │diagno- │ │progno- │ │decision │
     │acquisi-  │ │_proc-│ │stics/  │ │stics/  │ │_support/│
     │tion/     │ │ess-  │ │        │ │        │ │         │
     │          │ │ing/  │ │        │ │        │ │         │
     └──────────┘ └──────┘ └────────┘ └────────┘ └─────────┘
```

### Block 1: Signal Acquisition (`signal_acquisition/`)

Handles data ingestion from multiple file formats and in-memory caching.

| Module | Purpose |
|--------|---------|
| `loaders.py` | Multi-format signal loading (CSV, NPY, MAT, WAV, Parquet) |
| `repository.py` | LRU-cached signal store with signal_id reference pattern |

**Signal Repository Pattern**: Signals are loaded once and referenced by ID across tool calls. This avoids re-reading files on every analysis and enables efficient batch processing.

```
Client ──load_signal("bearing_001.csv")──▶ Repository ──signal_id──▶ All tools
                                              │
                                         LRU Cache
                                         (10 GB cap)
```

### Block 2: Signal Processing (`signal_processing/`)

Pure spectral analysis functions. No file I/O, no MCP dependency.

| Module | Purpose |
|--------|---------|
| `spectral.py` | PSD (Welch), STFT spectrogram, envelope spectrum (Hilbert) |

All functions take `(signal: np.ndarray, fs: float, ...)` and return plain dicts.

### Blocks 3-4: Diagnostics (`diagnostics/`)

State detection and health assessment.

| Module | Purpose |
|--------|---------|
| `bearing_analyzer.py` | Fault peak detection, evidence scoring, harmonic analysis |
| `bearing_catalog.py` | Verified bearing geometries with sources, characteristic frequencies |
| `iso20816.py` | ISO 20816-3 vibration severity zones A-D (boundary values from ISO 10816-3:2009) |

### Block 5: Prognostics (`prognostics/`) — Implemented

Remaining useful life estimation and trend analysis.

| Module | Purpose |
|--------|---------|
| `rul_estimator.py` | RUL estimation via linear extrapolation and exponential degradation models (multi-measurement series required) |
| `trend_analyzer.py` | Linear trend fitting with statistical significance and degradation onset detection |

### Block 6: Decision Support (`decision_support/`)

Integrated diagnosis combining all analysis results.

| Module | Purpose |
|--------|---------|
| `diagnosis_pipeline.py` | Full diagnostic: FFT + PSD + STFT + bearing + ISO + anomaly |

## Data Flow

```
Signal File (.csv/.mat/.wav)
    │
    ▼
Block 1: load_signal() ──▶ SignalRepository (in-memory cache)
    │
    ▼
Block 2: compute_psd(), compute_stft(), compute_envelope_spectrum()
    │
    ▼
Block 3: check_bearing_fault_peak(), check_all_bearing_faults()
    │
    ▼
Block 4: assess_vibration_severity() ──▶ ISO 20816-3 zones
    │
    ▼
Block 5: estimate_rul(), analyze_trend() ──▶ RUL & degradation forecasts
    │
    ▼
Block 6: diagnose_vibration() ──▶ Structured DiagnosisResult
    │
    ▼
MCP Server ──▶ LLM (Claude) ──▶ Human-readable recommendations
```

## Package Layout

```
src/
├── __init__.py                        # Package entry
├── __main__.py                        # CLI entry
├── server.py                          # FastMCP orchestrator (registers mcp_tools)
├── config.py                          # Path resolution (DATA_DIR, etc.)
├── models.py                          # Pydantic v2 response models
├── path_safety.py                     # Path containment helpers
│
├── mcp_tools/                         # MCP tool layer (one module per block family)
│   ├── __init__.py                    #   register_all(mcp)
│   ├── acquisition_tools.py           #   Block 1 tools + resources
│   ├── analysis_tools.py              #   Block 2 tools
│   ├── diagnostics_tools.py           #   Blocks 3-4 tools
│   ├── prognostics_tools.py           #   Block 5 tools
│   ├── report_tools.py                #   Block 6 report tools
│   ├── decision_support_tools.py      #   Block 6 decision tools
│   ├── prompts.py                     #   Workflow prompts
│   └── _utils.py                      #   Shared helpers (safe_resolve, ...)
│
├── signal_acquisition/                # ISO 13374 Block 1
│   ├── __init__.py                    #   re-exports loaders + repository
│   ├── loaders.py                     #   multi-format signal loading
│   └── repository.py                  #   LRU signal cache
│
├── signal_processing/                 # ISO 13374 Block 2
│   ├── __init__.py                    #   re-exports spectral functions
│   ├── spectral.py                    #   PSD, STFT, envelope spectrum
│   └── features.py                    #   time-domain feature extraction
│
├── diagnostics/                       # ISO 13374 Blocks 3-4
│   ├── __init__.py                    #   re-exports all diagnostics
│   ├── bearing_analyzer.py            #   fault peak detection
│   ├── bearing_catalog.py             #   bearing specs lookup
│   └── iso20816.py                    #   severity zone classification
│
├── decision_support/                  # ISO 13374 Block 6
│   ├── __init__.py                    #   re-exports diagnosis pipeline
│   ├── diagnosis_pipeline.py          #   integrated diagnosis
│   ├── alerts.py                      #   threshold alerting
│   └── recommendations.py             #   maintenance recommendations
│
├── prognostics/                       # ISO 13374 Block 5
│   ├── __init__.py                    #   re-exports RUL + trend functions
│   ├── rul_estimator.py               #   linear, exponential RUL
│   ├── kalman_rul.py                  #   Kalman-filter RUL
│   └── trend_analyzer.py              #   trend fitting, degradation onset
│
├── document_reader.py                 # PDF extraction, bearing spec parsing
├── report_generator.py                # HTML/DOCX report generation
├── html_templates.py                  # Report HTML templates
└── rag.py                             # Document search (FAISS/TF-IDF)
```

## Import Paths

Canonical sub-package paths only (the legacy flat shims were removed in v0.9.0):

```python
from predictive_maintenance_mcp.signal_processing.spectral import compute_psd
from predictive_maintenance_mcp.diagnostics.iso20816 import assess_vibration_severity
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import diagnose_vibration
```

## Extension Points

### Adding a new signal processor

1. Create `src/signal_processing/your_processor.py`
2. Add pure function: `def your_analysis(signal: np.ndarray, fs: float, ...) -> dict`
3. Re-export in `signal_processing/__init__.py`
4. Add a module-level MCP tool in `src/mcp_tools/analysis_tools.py` and register it in that module's `register()`

### Adding a new diagnostic module

1. Create `src/diagnostics/your_diagnostic.py`
2. Implement detection logic using spectral functions from Block 2
3. Re-export in `diagnostics/__init__.py`
4. Optionally integrate into `diagnosis_pipeline.py`

### Adding prognostic models

1. Create `src/prognostics/your_model.py` (existing models: `rul_estimator.py`, `trend_analyzer.py`)
2. Implement RUL estimation or trend analysis
3. Re-export in `prognostics/__init__.py`
4. Wire into decision support pipeline

## Data Privacy Architecture

Raw sensor data never leaves the user's machine:

```
User's Machine (Local Processing Only)
├── Signal Files (on disk)
├── SignalRepository (in-memory cache)
├── Analysis Results (computed locally)
└── Reports (generated locally)

↓ Network Boundary ↓
Only: Diagnostic reports, LLM queries (no raw waveforms)
```

## Standards Compliance

| Standard | Coverage | Modules |
|----------|----------|---------|
| ISO 13374 | Blocks 1-6 | All sub-packages |
| ISO 20816-3 (thresholds from ISO 10816-3:2009) | Severity zones A-D | `diagnostics/iso20816.py` |
| MIMOSA OSA-CBM | Signature analysis, detection | `diagnostics/`, `signal_processing/` |
