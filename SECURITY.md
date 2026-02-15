# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

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

- All data processing happens locally (no cloud transmission)
- Signal data and reports remain on the local filesystem
- ML models are trained and stored locally
- No network requests are made during normal operation

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
- **No external APIs**: All analysis runs locally without internet
- **Sample data**: Included dataset is from public research sources (MathWorks)
- **Your data**: Any proprietary vibration data you add stays on your machine
