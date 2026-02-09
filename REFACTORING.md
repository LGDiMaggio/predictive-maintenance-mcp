# Refactoring Guide

## Current State (v0.3.2)

The `machinery_diagnostics_server.py` file (~4800 lines) has been partially refactored:
- **Pydantic models** → extracted to `src/models.py`
- **Report generation** → already in `src/report_generator.py`
- **Document reader** → already in `src/document_reader.py`
- **HTML templates** → already in `src/html_templates.py`

## Recommended Next Steps

### Phase 1: Extract Pure Logic (Low Risk)
Extract functions that don't depend on `@mcp` decorators:

1. **`src/signal_processing.py`** — Pure signal analysis functions
   - `extract_time_domain_features()` (line ~1556)
   - FFT computation core logic
   - Envelope computation core logic
   - Bandpass filter helpers

2. **`src/ml_engine.py`** — ML pipeline functions
   - Feature extraction pipeline
   - Model training logic
   - Prediction logic
   - PCA/Scaler management

### Phase 2: Extract MCP Tool Groups (Medium Risk)
Group related `@mcp.tool()` functions by passing the `mcp` instance:

3. **`src/tools/analysis_tools.py`** — FFT, envelope, statistics, ISO tools
4. **`src/tools/ml_tools.py`** — Feature extraction, training, prediction
5. **`src/tools/manual_tools.py`** — Document reader MCP wrappers
6. **`src/tools/visualization_tools.py`** — Plotting and report tools
7. **`src/tools/prompt_tools.py`** — All MCP prompt definitions

### Phase 3: Registration Pattern
Use a registration pattern in each module:

```python
# src/tools/analysis_tools.py
def register_analysis_tools(mcp):
    @mcp.tool()
    async def analyze_fft(...): ...
    
    @mcp.tool()
    async def analyze_envelope(...): ...
```

```python
# src/machinery_diagnostics_server.py (main)
from tools.analysis_tools import register_analysis_tools
from tools.ml_tools import register_ml_tools
# ...

register_analysis_tools(mcp)
register_ml_tools(mcp)
```

## Testing After Each Phase
Run after every extraction step:
```bash
pytest tests/ -v
uv run mcp dev src/machinery_diagnostics_server.py  # Quick MCP Inspector test
```

## Important Notes
- Always keep backward compatibility for direct script execution (`python src/machinery_diagnostics_server.py`)
- The try/except import pattern (relative vs absolute) must be maintained in each new module
- Test with both `pip install -e .` and direct execution
