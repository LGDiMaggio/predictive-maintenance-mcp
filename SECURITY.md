# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| 0.4.x   | :white_check_mark: |
| 0.3.x   | :x:                |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. **Email**: Send details to luigi.dimaggio@polito.it with subject "SECURITY: predictive-maintenance-mcp"
3. **Include**: Description of the vulnerability, steps to reproduce, potential impact, and suggested fix if available

### What to expect

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix timeline**: Critical issues within 2 weeks, others within 1 month
- **Credit**: Security reporters will be credited in the CHANGELOG (unless they prefer anonymity)

## Security Considerations

### Local-First Architecture

This MCP server is designed to run **locally** on your machine:

- All signal processing (FFT, envelope, statistics, ML) runs locally — no third-party analytics APIs
- Raw signal files (CSV), equipment manuals (PDF), and trained ML models never leave your filesystem
- HTML reports are generated and stored locally
- The MCP server itself makes **no network requests** during operation

> **Important**: While the server processes data locally, the **analysis results** (peak frequencies, RMS values, statistical summaries, diagnostic text, manual excerpts) are returned to the LLM client and transmitted to the LLM provider's API (e.g., Anthropic, OpenAI). This is inherent to any LLM-based workflow — the LLM needs tool outputs to generate responses. Raw signal arrays are never sent in full; only computed summaries and metrics flow through the LLM.
>
> **To maximize privacy**: Use a local LLM (e.g., Ollama, LM Studio) as your MCP client — this keeps the entire pipeline on your machine with zero data leaving your network.

### File System Access

The server accesses the local filesystem for:
- Reading vibration signal data from `data/signals/`
- Reading machine manuals from `resources/machine_manuals/`
- Writing HTML reports to `reports/`
- Saving trained ML models to `models/`
- Caching extracted manual specs in `resources/cache/`

**Mitigations**:
- File paths are validated and restricted to project subdirectories
- No arbitrary file system access outside the project root
- Report generation uses sanitized filenames

### Dependencies

- All dependencies are pinned with minimum versions in `pyproject.toml`
- Regular dependency audits are recommended: `pip audit`
- No dependencies with known critical vulnerabilities at time of release

### Data Privacy

- **No telemetry**: The server does not collect or transmit usage data
- **No external APIs**: The server itself makes no network calls — all analysis runs locally
- **Sample data**: Included dataset is from public research sources (MathWorks)
- **Raw data stays local**: Your proprietary vibration signals, manuals, and models remain on your filesystem
- **Analysis results flow to LLM**: Computed metrics (frequencies, RMS, kurtosis, diagnoses) are returned to the LLM provider as tool outputs — this is inherent to MCP-based workflows. For full air-gapped privacy, use a local LLM client
