# Predictive Maintenance MCP Server

<!-- mcp-name: io.github.LGDiMaggio/predictive-maintenance-mcp -->

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17611542.svg)](https://doi.org/10.5281/zenodo.17611542)
[![Tests](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/LGDiMaggio/predictive-maintenance-mcp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/LGDiMaggio/predictive-maintenance-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Give your AI assistant evidence-based vibration diagnostics — machinery fault detection, ISO-cited severity, and diagnostic reports built to support and accelerate expert decision-making.**

An open-source [MCP](https://modelcontextprotocol.io/) server that turns LLMs into condition monitoring assistants for reliability engineers. Its core design rule: **the server refuses to guess**. No diagnosis is ever inferred from filenames or statistical parameters alone — a fault indication requires matching spectral evidence. Every severity claim cites ISO 20816-3, and the evaluative wording in reports is authored by the server, not improvised by the model. The AI orchestrates the analysis and presents the evidence — detected fault frequencies, matched fault patterns, severity zones — while the final judgment stays with the engineer. Also available as a **Claude Code plugin** with 8 diagnostic skills.

---

## See It in Action

<p align="center">
  <img src="assets/claude_gif.gif" alt="Predictive Maintenance MCP — diagnostic workflow in Claude Desktop" width="720">
</p>

<p align="center"><em>Full diagnostic workflow: load signal → spectral analysis → fault detection → severity assessment → report generation</em></p>

---

## Choose Your Path

| You are | Start here |
|---------|------------|
| **Reliability / maintenance engineer** — diagnostics in plain language, no coding | [Engineer's Quickstart](docs/QUICKSTART_ENGINEER.md) |
| **AI / MCP developer** — run, integrate, and extend the server | [Developer's Quickstart](docs/QUICKSTART_DEVELOPER.md) · [Quick Start](#quick-start) below |
| **Researcher / evaluator** — how the numbers are measured | [Benchmark Methodology](docs/benchmark-methodology.md) · [Benchmark](#benchmark) below |

---

## Quick Start

**Get running in ~3 minutes.** On Windows, one script wires everything into Claude Desktop — it installs the venv, pre-compiles dependencies, and writes `claude_desktop_config.json` for you (OneDrive / cloud-sync paths included):

```powershell
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
.\setup_claude.ps1
```

Restart Claude Desktop, then try:

> *"Load real_train/OuterRaceFault_1.csv and check if the bearing is healthy."*

<details>
<summary><b>Manual config (macOS / Linux / other MCP clients)</b></summary>

Install the package:

```bash
pip install predictive-maintenance-mcp
```

Find the full path to `uvx` (`which uvx` on macOS/Linux, `where uvx` on Windows), then add to your client config — `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "/full/path/to/uvx",
      "args": ["predictive-maintenance-mcp"],
      "env": { "UV_LINK_MODE": "copy" }
    }
  }
}
```

> **Why the full path?** Claude Desktop launches servers with a minimal `PATH` that often omits user-local tool directories (e.g. `~/.local/bin`). Using the full path to `uvx` avoids a silent "command not found" failure. On Windows the typical path is `C:\Users\<you>\.local\bin\uvx.exe`.

</details>

> **More options**: [install from source](INSTALL.md) · [VS Code setup](INSTALL.md) · [Docker / HTTPS deployment](docs/DEPLOYMENT.md) · [use with local LLMs (Ollama)](docs/OLLAMA_GUIDE.md)

---

## Benchmark

A blind, reproducible diagnostic-accuracy benchmark on the public
[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
dataset (12 kHz drive-end subset: 60 fault records + 4 normal baselines).
Fault labels never reach the system under test — signals enter under opaque
ids, a separate scorer is the only label reader, and blindness, checksum
integrity, and determinism are enforced by CI-run guard tests, not prose.
Results are stratified by the per-record diagnosability grades of the
Smith & Randall (2015) reference study, so records that study found
undiagnosable by any classical method are reported separately instead of
inflating or deflating the headline.

<!-- cwru-benchmark:start -->
On records the reference study grades clearly diagnosable (Y1+Y2, <!-- slot: headline.n_records -->44<!-- /slot --> records):
characteristic fault frequency detected on <!-- slot: headline.frequency_detection.hits -->44<!-- /slot -->/<!-- slot: headline.frequency_detection.total -->44<!-- /slot -->,
correct fault ranked first on <!-- slot: headline.classification.correct -->34<!-- /slot -->/<!-- slot: headline.classification.total -->44<!-- /slot --> (<!-- slot: headline.classification.rate pct1 -->77.3<!-- /slot -->%),
and <!-- slot: strata.Y1.classification.correct -->9<!-- /slot -->/<!-- slot: strata.Y1.classification.total -->9<!-- /slot --> on the textbook-signature (Y1) stratum.
On the <!-- slot: strata.ungraded.false_positives.total_normal -->4<!-- /slot --> healthy baselines, <!-- slot: strata.ungraded.false_positives.records_with_any -->2<!-- /slot --> records raised a false indication under the same criterion.
<!-- cwru-benchmark:end -->

The numbers above are read from the committed, re-runnable artifact
([results.json](benchmarks/cwru/results/results.json)) and drift-guarded by CI:
every value is bound to its key in the artifact, and a mismatch fails the build.
Methodology, blind protocol, and honest-benchmarking notes:
[docs/benchmark-methodology.md](docs/benchmark-methodology.md). Reproduce with:

```bash
python -m benchmarks.cwru all
```

---

## What Can It Do?

**Point the AI at a vibration signal → get the evidence behind the fault — detected frequencies, matched fault patterns, ISO-cited severity — to support your call.**

| You say | The AI does |
|---------|-------------|
| *"Is this bearing healthy?"* | Loads the signal, runs spectral analysis, surfaces matching fault-frequency evidence, cites the ISO 20816-3 severity zone |
| *"Generate a full diagnostic report"* | Produces an interactive HTML report with charts, fault markers, and server-authored severity wording |
| *"Extract specs from test_pump_manual.pdf and diagnose the signal"* | Reads the equipment manual, looks up the bearing model, calculates expected fault frequencies, flags which ones the signal actually shows |
| *"Train an anomaly detector on my healthy baselines, then flag anomalies"* | Trains a model on your normal data, scores new signals, flags outliers for your review |

The AI doesn't guess — it calls **37 specialized MCP endpoints** (34 tools + 3 prompts) running locally on your machine. Every signal is referenced by a single `signal_id` handle from load to report. Your data never leaves your infrastructure.

Full endpoint reference, grouped by category: **[Tool Catalog](docs/TOOL_CATALOG.md)**.

---

## Claude Code Plugin

The project includes a **plugin for Claude Code** with domain-specific skills that activate automatically during conversation.

```shell
/plugin marketplace add LGDiMaggio/predictive-maintenance-mcp
/plugin install predictive-maintenance@predictive-maintenance-marketplace
```

<p align="center">
  <img src="assets/plugin.gif" alt="Claude Code Plugin — skills, agents, and slash commands in action" width="720">
</p>

The plugin adds **8 skills** that activate automatically based on context (bearing-diagnosis, gear-diagnosis, quick-screening, report-generation, anomaly-detection, signal-management, documentation-search, prognostics), **2 agents** that run multi-step diagnostic workflows end-to-end and hand you the evidence (diagnostic-pipeline, signal-explorer), and **3 commands** for quick entry points (`/pm-diagnose`, `/pm-screen`, `/pm-report`).

Full skill, agent, and command reference: [Plugin README](plugin/README.md).

---

## Reports

All analysis tools generate **interactive HTML reports** you can open in any browser — pan, zoom, hover for details. Also supports structured Word (.docx) exports.

<details>
<summary><b>Report examples</b></summary>

![Envelope Analysis Report](assets/envelope_analysis.png)

![ISO Severity Assessment](assets/iso.png)

| Report Type | What it shows |
|-------------|---------------|
| Frequency spectrum | Peak detection, harmonic markers |
| Envelope analysis | Bearing fault frequency matching |
| Severity assessment | Vibration health zones (ISO 20816-3) |
| Word document | Full diagnostic narrative with embedded charts |
| PCA visualization | Multi-signal anomaly clustering |
| Feature comparison | Side-by-side signal feature analysis |

</details>

---

## Sample Data Included

The project ships with **20 real bearing vibration signals** from production machinery tests — ready to use out of the box: a training set (2 healthy baselines + 12 fault signals, inner and outer race) and a test set (1 healthy baseline + 5 fault signals).

Try: *"Load real_train/OuterRaceFault_1.csv and diagnose the bearing fault."*

Full dataset documentation: [data/README.md](data/README.md)

---

## Architecture

```
          YOU (natural language)
               │
               v
     LLM (Claude, GPT, Ollama...)
     understands intent, selects tools
               │
               v  ── Model Context Protocol ──
    ┌──────────────────────────────┐
    │    Predictive Maintenance    │
    │         MCP Server           │
    │                              │
    │  Signal Analysis    Reports  │
    │  Fault Detection    ML       │
    │  Severity Rating    RAG Docs │
    └──────────────────────────────┘
               │
               v
       YOUR DATA (stays local)
    signals · manuals · models
```

The codebase follows a **modular architecture** organized around the ISO 13374 Six-Block Diagnostic standard — signal acquisition, processing, diagnostics, prognostics, and decision support as separate sub-packages. Standards implemented: ISO 13374, ISO 20816-3, MIMOSA OSA-CBM. Module-level detail: [Architecture guide](docs/architecture.md).

**Key design choices:**
- **Privacy-first** — raw vibration data never leaves your machine; only computed results flow to the LLM
- **LLM-agnostic** — works with Claude, ChatGPT, Microsoft Copilot Studio, or any MCP-compatible client. Use [Ollama](docs/OLLAMA_GUIDE.md) for fully air-gapped deployments
- **Modular** — use only the tools you need, extend with your own

---

## Documentation

| Guide | For |
|-------|-----|
| [Quickstart for Engineers](docs/QUICKSTART_ENGINEER.md) | Get results fast, no coding required |
| [Quickstart for Developers](docs/QUICKSTART_DEVELOPER.md) | Understand MCP, extend the server |
| [Tool Catalog](docs/TOOL_CATALOG.md) | Every MCP endpoint, grouped by category |
| [Adapter Guide](docs/ADAPTER_GUIDE.md) | Bring vendor/DAQ raw data in via explicit declarations |
| [Plugin README](plugin/README.md) | Claude Code plugin installation and usage |
| [HTTPS Deployment](docs/DEPLOYMENT.md) | Docker + HTTPS for enterprise environments |
| [Ollama Guide](docs/OLLAMA_GUIDE.md) | Use with local LLMs (fully air-gapped) |
| [Architecture](docs/architecture.md) | ISO 13374 block mapping and module design |
| [Benchmark Methodology](docs/benchmark-methodology.md) | How the CWRU diagnostic benchmark is measured |
| [Examples](EXAMPLES.md) | Complete diagnostic workflows |
| [Installation](INSTALL.md) | Detailed setup and troubleshooting |
| [Contributing](CONTRIBUTING.md) | How to contribute (all skill levels welcome) |
| [Changelog](CHANGELOG.md) | Version history |

---

## Testing

**85%+ test coverage**, enforced as a CI minimum, across Windows, macOS, and Linux (Python 3.11 & 3.12) — the current measured figure is on the codecov badge above.

```bash
pytest                                  # run all tests
pytest --cov=src --cov-report=html      # with coverage report
```

20+ test files covering signal analysis, fault detection, severity assessment, ML models, report generation, RAG search, and real bearing fault data validation.

---

## Roadmap

- [x] 37 MCP endpoints (34 tools, 3 prompts) with modular architecture and a single `signal_id` handle
- [x] Claude Code plugin (8 skills, 2 agents, 3 commands)
- [x] 85%+ test coverage enforced in CI, CI/CD on 3 platforms
- [x] Docker + SSE/HTTP transport for enterprise deployment
- [x] Semantic document search (FAISS + TF-IDF)
- [x] Blind, reproducible diagnostic benchmark on the CWRU dataset (extensible to Paderborn)
- [ ] Customizable severity thresholds
- [x] Remaining useful life (RUL) estimation from repeated measurements (linear, exponential, Kalman)
- [x] Trend analysis and degradation onset detection
- [ ] Multi-signal trending and historical comparison
- [ ] Real-time streaming (MQTT/Kafka)
- [ ] Fleet dashboard for multi-asset monitoring
- [ ] CMMS integration (SAP, Maximo, Infor)

> Ideas? [Open a discussion](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) or [create an issue](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues/new/choose).

---

## Are you using this?

I'd genuinely love to know. Whether you ran it on real machinery or just tried the sample data, drop a line in [Discussions](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions) — one sentence about your machine or use case is enough. Real-world feedback directly shapes what gets built next.

---

## Related

**[claude-stwinbox-diagnostics](https://github.com/LGDiMaggio/claude-stwinbox-diagnostics)** — Extends this project by connecting a physical edge sensor (STEVAL-STWINBX1) to Claude via MCP, with Claude Skills for guided condition monitoring. Same analysis engine, real hardware, operator-friendly reports.

---

## Contributing

Contributions welcome from **everyone** — not just programmers. Domain experts, technical writers, and testers are equally valued. See [CONTRIBUTING.md](CONTRIBUTING.md) for paths tailored to your background.

**Quick start**: browse [Issues](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues) for `good first issue` or `help wanted` labels.

---

## Citation

```bibtex
@software{dimaggio_predictive_maintenance_mcp_2025,
  title   = {Predictive Maintenance MCP Server},
  author  = {Di Maggio, Luigi Gianpio},
  year    = {2025},
  version = {0.12.0},
  url     = {https://github.com/LGDiMaggio/predictive-maintenance-mcp},
  doi     = {10.5281/zenodo.17611542}
}
```

## License

MIT — see [LICENSE](LICENSE). Sample data is CC BY-NC-SA 4.0 (non-commercial); for commercial use, replace with your own machinery data.

## Acknowledgments

[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (descended from [FastMCP](https://github.com/jlowin/fastmcp)) · [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic · Sample data from [MathWorks](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data) · Core development assisted by [Claude](https://claude.ai)

---

*An open-source predictive maintenance AI agent and condition monitoring copilot — built to support reliability engineers and the developer community.*
