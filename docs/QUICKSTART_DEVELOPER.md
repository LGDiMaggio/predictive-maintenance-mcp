# 💻 Quickstart for AI & Software Developers

> **Goal**: Understand how MCP works, run the server, and create your first custom diagnostic tool in under 20 minutes.

---

## What You'll Learn

By the end of this guide, you will:

1. ✅ Understand the Model Context Protocol (MCP) architecture
2. ✅ Run the server and inspect its tools interactively
3. ✅ Read and understand the server codebase
4. ✅ Know how to create a new diagnostic tool from scratch
5. ✅ Have a clear path to contributing

---

## Why This Project Matters (For Developers)

### The Problem

Industrial machines generate **terabytes of sensor data** — vibration, temperature, pressure, acoustics. Expert analysts can interpret this data to predict failures before they happen, saving thousands of euros per avoided downtime. But these experts are rare and expensive.

### The Insight

LLMs are excellent at **reasoning, planning, and communicating** — but they don't know signal processing. Signal processing libraries (NumPy, SciPy, scikit-learn) are excellent at **computation** — but they can't reason or communicate.

### The Solution: MCP

The **Model Context Protocol** is the bridge. It's an open standard (by Anthropic) that defines how an LLM can discover and invoke external tools. Think of it as a universal plugin system:

```
┌─────────────┐     MCP Protocol      ┌──────────────────┐
│             │ ◄──────────────────── │                  │
│    LLM      │   "What tools exist?" │   MCP Server     │
│  (Claude)   │ ────────────────────► │  (This project)  │
│             │   "Call analyze_fft"  │                  │
│             │ ◄──────────────────── │  • analyze_fft   │
│             │   {peaks: [...]}     │  • envelope      │
│             │                      │  • iso_20816     │
│             │                      │  • ml_anomaly    │
└─────────────┘                      └──────────────────┘
```

**The LLM has no idea what FFT is.** It just knows:
- *"There's a tool called `analyze_fft`"*
- *"It takes a file path and sampling rate"*
- *"It returns peaks and frequencies"*

This is **the same pattern** you can use for any domain: medical imaging, financial analysis, climate modeling, robotics. This project is a production-quality reference implementation for the **predictive maintenance** domain.

---

## Step 1: Set Up the Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
python -m venv .venv

# Activate
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS/Linux

# Install (includes pytest, black, mypy, flake8)
pip install -e ".[dev]"

# Verify everything works
python validate_server.py
pytest -v
```

---

## Step 2: Explore the Architecture

### Project Structure

```
predictive-maintenance-mcp/
├── src/
│   ├── machinery_diagnostics_server.py  ← THE SERVER (all MCP tools defined here)
│   ├── document_reader.py               ← PDF/manual processing module
│   ├── report_generator.py              ← HTML report generation (Plotly)
│   └── html_templates.py                ← Report HTML templates
├── data/
│   └── signals/                         ← Real vibration data (CSV + metadata)
│       ├── real_train/                  ← 14 signals for training
│       └── real_test/                   ← 6 signals for validation
├── resources/
│   ├── machine_manuals/                 ← Equipment manuals (PDF/TXT)
│   ├── bearing_catalogs/                ← Bearing geometry database
│   └── cache/                           ← Auto-cached manual extractions
├── models/                              ← Trained ML models (joblib)
├── reports/                             ← Generated HTML reports
└── tests/                               ← Comprehensive test suite
```

### The Core: `machinery_diagnostics_server.py`

This **single file** contains the entire MCP server. Open it and you'll see a very clean pattern:

```python
from mcp.server.fastmcp import FastMCP

# Create the server
mcp = FastMCP("Predictive Maintenance Server")

# Define a RESOURCE (direct data access for the LLM)
@mcp.resource("signal://list")
def list_signals() -> str:
    """List all available vibration signals."""
    # ... returns signal listing
    
# Define a TOOL (computation the LLM can invoke)
@mcp.tool()
def analyze_fft(file_path: str, sampling_rate: float = None, ...) -> dict:
    """FFT spectrum analysis with automatic peak detection."""
    # ... performs actual signal processing
    # ... returns structured results
```

**That's it.** A `@mcp.tool()` decorator + a Python function = an LLM-accessible tool.

### Resources vs Tools

| Concept | Purpose | Example | When to Use |
|---------|---------|---------|-------------|
| **Resource** | Read-only data access | `signal://list` | LLM needs to browse/read data |
| **Tool** | Computation & side effects | `analyze_fft()` | LLM needs to compute something |

Resources are like GET endpoints. Tools are like POST endpoints. The LLM uses Resources to gather context and Tools to take action.

---

## Step 3: Inspect the Server Interactively

### Using MCP Inspector

The best way to understand the server is to see it live:

```bash
npx @modelcontextprotocol/inspector npx predictive-maintenance-mcp
```

Or from source:

```bash
uv run mcp dev src/machinery_diagnostics_server.py
```

This opens a web UI where you can:
- Browse all registered Resources and Tools
- See each tool's parameters, types, and documentation
- Call tools with test data and see the results
- Understand exactly what the LLM sees

### Using Claude Desktop

Configure the server in Claude Desktop ([see INSTALL.md](../INSTALL.md)) and try:

```
What tools do you have available for vibration analysis?
```

Claude will describe every tool, its parameters, and what it does — because MCP exposes this metadata automatically.

---

## Step 4: Create Your First Tool

Let's add a new tool: **thermographic analysis** (simplified example). This demonstrates the pattern for any new diagnostic capability.

### 4.1 — Write the Tool Function

Add to `src/machinery_diagnostics_server.py`:

```python
@mcp.tool()
def analyze_temperature_trend(
    temperatures: list[float],
    timestamps: list[str],
    warning_threshold: float = 80.0,
    critical_threshold: float = 100.0
) -> dict:
    """
    Analyze temperature trend data for thermal anomalies.
    
    Detects overheating patterns, calculates rate of change,
    and classifies severity based on configurable thresholds.
    
    Args:
        temperatures: List of temperature readings in °C
        timestamps: List of ISO timestamps for each reading
        warning_threshold: Warning temperature in °C (default: 80.0)
        critical_threshold: Critical temperature in °C (default: 100.0)
    
    Returns:
        Dictionary with trend analysis, anomaly detection, and severity
    """
    import numpy as np
    
    temps = np.array(temperatures)
    
    # Calculate statistics
    current = temps[-1]
    rate_of_change = np.gradient(temps).mean()  # °C per sample
    
    # Classify severity
    if current >= critical_threshold:
        severity = "CRITICAL"
        recommendation = "Immediate shutdown recommended"
    elif current >= warning_threshold:
        severity = "WARNING"
        recommendation = "Schedule inspection within 24 hours"
    else:
        severity = "NORMAL"
        recommendation = "No action required"
    
    return {
        "current_temperature": float(current),
        "max_temperature": float(temps.max()),
        "min_temperature": float(temps.min()),
        "mean_temperature": float(temps.mean()),
        "rate_of_change": float(rate_of_change),
        "severity": severity,
        "recommendation": recommendation,
        "samples_analyzed": len(temps)
    }
```

### 4.2 — Write Tests

Create `tests/test_temperature.py`:

```python
import pytest

def test_analyze_temperature_normal():
    """Test normal temperature classification."""
    from machinery_diagnostics_server import analyze_temperature_trend
    
    result = analyze_temperature_trend(
        temperatures=[45.0, 46.0, 45.5, 46.2, 45.8],
        timestamps=["2025-01-01T00:00", "2025-01-01T01:00", 
                     "2025-01-01T02:00", "2025-01-01T03:00", 
                     "2025-01-01T04:00"]
    )
    
    assert result["severity"] == "NORMAL"
    assert result["current_temperature"] == pytest.approx(45.8)

def test_analyze_temperature_critical():
    """Test critical temperature detection."""
    from machinery_diagnostics_server import analyze_temperature_trend
    
    result = analyze_temperature_trend(
        temperatures=[60.0, 75.0, 90.0, 105.0, 110.0],
        timestamps=["2025-01-01T00:00", "2025-01-01T01:00",
                     "2025-01-01T02:00", "2025-01-01T03:00",
                     "2025-01-01T04:00"]
    )
    
    assert result["severity"] == "CRITICAL"
    assert "shutdown" in result["recommendation"].lower()
```

### 4.3 — Run Tests

```bash
pytest tests/test_temperature.py -v
```

### 4.4 — Test with Claude

Restart Claude Desktop and try:

```
I have these temperature readings from a pump motor: 
45, 52, 61, 73, 85, 92, 98, 103 °C 
taken hourly. Analyze the trend.
```

Claude will automatically discover and call your new tool.

**That's it.** You just extended an industrial AI system with a new diagnostic capability.

---

## Step 5: Understand the Codebase

### Key Patterns to Study

| Pattern | Where | What You'll Learn |
|---------|-------|-------------------|
| **Tool with auto-detection** | `analyze_fft()` | How to load metadata and infer parameters |
| **Report generation** | `generate_fft_report()` | How to create interactive Plotly HTML reports |
| **ML pipeline** | `train_anomaly_model()` | How to build scikit-learn pipelines as MCP tools |
| **Caching** | `extract_manual_specs()` | How to cache expensive operations (PDF parsing) |
| **Resource + Tool combo** | `signal://read` + `analyze_fft` | How Resources and Tools work together |

### Design Principles

1. **One file, one server** — The entire MCP interface is in `machinery_diagnostics_server.py`. This makes it easy to understand, test, and deploy.

2. **Tools are pure functions** — Each tool takes parameters, does computation, returns structured data. No side effects except file I/O (reports, models).

3. **Metadata-driven** — Sampling rates, bearing frequencies, signal units are auto-detected from metadata JSON files. No hardcoded assumptions.

4. **Fail gracefully** — Tools return meaningful error messages, not stack traces. The LLM can relay these to the user in natural language.

---

## Step 6: Where to Contribute

### Good First Issues (Start Here)

These are tasks specifically designed to be completable by someone new to the project:

| Task | Skills Needed | Impact |
|------|---------------|--------|
| Add Parquet file reading support | Python, pandas | Allows users with industrial data historians to connect directly |
| Make ISO report thresholds configurable | Python | Enables different machine classes (pumps vs turbines) |
| Add unit conversion tool (mil ↔ mm/s ↔ g) | Python, vibration basics | Helps users with data in different unit systems |
| Improve error messages for missing metadata | Python | Better UX for new users |
| Add more bearings to the catalog | JSON editing | Expands bearing frequency lookup without writing code |

Browse all open issues: [GitHub Issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) — filter by `good first issue` label.

### Architecture Contributions (Intermediate)

- **Docker containerization** — Package the whole server into a single `docker run` command
- **Vector search for manuals** — Integrate ChromaDB for semantic search over large PDFs
- **Streaming signal support** — Real-time vibration monitoring through Server-Sent Events

### Research Contributions (Advanced)

- **Deep learning models** — Replace OneClassSVM with autoencoders or transformers
- **Multi-modal fusion** — Combine vibration + temperature + current data
- **Transfer learning** — Pre-trained models that work across different machine types

---

## Key Resources

| Resource | URL |
|----------|-----|
| **MCP Specification** | [modelcontextprotocol.io](https://modelcontextprotocol.io/) |
| **FastMCP Framework** | [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp) |
| **MCP Inspector** | [github.com/modelcontextprotocol/inspector](https://github.com/modelcontextprotocol/inspector) |
| **Project Issues** | [GitHub Issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) |
| **Contributing Guide** | [CONTRIBUTING.md](../CONTRIBUTING.md) |

---

## What's Next?

1. **Read the full examples**: [EXAMPLES.md](../EXAMPLES.md) — 7 complete workflows showing every tool
2. **Browse the server code**: `src/machinery_diagnostics_server.py` is well-documented
3. **Pick an issue**: [Issues · good first issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
4. **Read the contributing guide**: [CONTRIBUTING.md](../CONTRIBUTING.md)
5. **Think bigger**: What domain could YOU build an MCP server for?

---

**This project is a template. The real power is what you build next.**
