# Architecture Guide

## Overview

predictive-maintenance-mcp is structured around the **ISO 13374 six-block architecture** for condition monitoring and diagnostics. Each block maps to a Python sub-package under `predictive_maintenance_mcp`, providing clear separation of concerns while maintaining backward compatibility with the flat module layout.

## ISO 13374 Block Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (FastMCP)                        │
│              machinery_diagnostics_server.py                    │
│                   50+ MCP tools & resources                     │
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
| `bearing_analyzer.py` | Fault peak detection, confidence scoring, harmonic analysis |
| `bearing_catalog.py` | SKF/FAG/Timken/NSK bearing specs, characteristic frequencies |
| `iso10816.py` | ISO 10816 / ISO 20816-3 vibration severity zones (A/B/C/D) |

### Block 5: Prognostics (`prognostics/`)

Placeholder for Phase 2. Will contain:
- Remaining Useful Life (RUL) estimation
- Trend analysis with confidence intervals

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
Block 4: assess_vibration_severity() ──▶ ISO 10816 zones
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
├── machinery_diagnostics_server.py    # MCP server (all tool registrations)
├── config.py                          # Path resolution (DATA_DIR, etc.)
├── models.py                          # Pydantic v2 response models
│
├── signal_acquisition/                # ISO 13374 Block 1
│   ├── __init__.py                    #   re-exports loaders + repository
│   ├── loaders.py                     #   multi-format signal loading
│   └── repository.py                  #   LRU signal cache
│
├── signal_processing/                 # ISO 13374 Block 2
│   ├── __init__.py                    #   re-exports spectral functions
│   └── spectral.py                    #   PSD, STFT, envelope spectrum
│
├── diagnostics/                       # ISO 13374 Blocks 3-4
│   ├── __init__.py                    #   re-exports all diagnostics
│   ├── bearing_analyzer.py            #   fault peak detection
│   ├── bearing_catalog.py             #   bearing specs lookup
│   └── iso10816.py                    #   severity zone classification
│
├── decision_support/                  # ISO 13374 Block 6
│   ├── __init__.py                    #   re-exports diagnosis pipeline
│   └── diagnosis_pipeline.py          #   integrated diagnosis
│
├── prognostics/                       # ISO 13374 Block 5 (Phase 2)
│   └── __init__.py                    #   placeholder
│
├── signal_loader.py                   # Canonical loaders (+ compat shim)
├── signal_repository.py               # Canonical repository (+ compat shim)
├── spectral.py                        # Compat shim → signal_processing/
├── bearing_analyzer.py                # Compat shim → diagnostics/
├── bearing_catalog.py                 # Compat shim → diagnostics/
├── iso10816.py                        # Compat shim → diagnostics/
├── diagnosis_pipeline.py              # Compat shim → decision_support/
│
├── document_reader.py                 # PDF extraction, bearing spec parsing
├── report_generator.py                # HTML/DOCX report generation
├── html_templates.py                  # Report HTML templates
└── rag.py                             # Document search (FAISS/TF-IDF)
```

## Import Paths

Both old (flat) and new (sub-package) import paths work:

```python
# New (preferred)
from predictive_maintenance_mcp.signal_processing import compute_psd
from predictive_maintenance_mcp.diagnostics import assess_vibration_severity
from predictive_maintenance_mcp.decision_support import diagnose_vibration

# Old (backward compatible, will work indefinitely)
from predictive_maintenance_mcp.spectral import compute_psd
from predictive_maintenance_mcp.iso10816 import assess_vibration_severity
from predictive_maintenance_mcp.diagnosis_pipeline import diagnose_vibration
```

## Extension Points

### Adding a new signal processor

1. Create `src/signal_processing/your_processor.py`
2. Add pure function: `def your_analysis(signal: np.ndarray, fs: float, ...) -> dict`
3. Re-export in `signal_processing/__init__.py`
4. Register MCP tool in `machinery_diagnostics_server.py`

### Adding a new diagnostic module

1. Create `src/diagnostics/your_diagnostic.py`
2. Implement detection logic using spectral functions from Block 2
3. Re-export in `diagnostics/__init__.py`
4. Optionally integrate into `diagnosis_pipeline.py`

### Adding prognostic models (Phase 2)

1. Create `src/prognostics/your_model.py`
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
| ISO 13374 | Blocks 1-4, 6 | All sub-packages |
| ISO 10816 / 20816-3 | Severity zones A-D | `diagnostics/iso10816.py` |
| MIMOSA OSA-CBM | Signature analysis, detection | `diagnostics/`, `signal_processing/` |
