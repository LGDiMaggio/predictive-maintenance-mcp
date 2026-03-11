# Installation Guide

## Quick Start (Recommended)

### From PyPI

```bash
pip install predictive-maintenance-mcp
```

This installs the server and all dependencies. You can then run it directly:

```bash
predictive-maintenance-mcp
```

> **Note**: The PyPI package does not include sample data. Clone the repository (see below) if you need the bundled vibration signals.

---

## From Source

### Prerequisites
- Python 3.11 or 3.12
- pip (Python package manager)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Package
```bash
# Install production dependencies only
pip install -e .

# OR install with development tools (for contributors)
pip install -e .[dev]
```

### Alternative: Install with uv

```bash
uv sync            # production deps
uv sync --all-extras  # include dev deps
```

### 4. Verify Installation
```bash
python -c "import mcp; print('MCP installed successfully')"
python validate_server.py
```

### 5. Run the Server
```bash
# Default: stdio transport (Claude Desktop, VS Code)
predictive-maintenance-mcp
python -m predictive_maintenance_mcp

# SSE transport for remote/enterprise clients (Copilot Studio, networked)
predictive-maintenance-mcp --transport sse --host 0.0.0.0 --port 8080

# Docker (SSE mode by default)
docker compose up -d
```

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for HTTPS setup with Docker + Caddy auto-TLS.

---

## For Claude Desktop Users

### Automatic Setup (Windows)
Run the PowerShell script for automatic configuration:
```powershell
.\setup_claude.ps1
```

This will:
- Create virtual environment
- Install dependencies
- Configure Claude Desktop automatically
- Test the server

### Manual Setup
1. Complete steps 1-3 above
2. Edit your Claude Desktop config:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

3. Add this configuration:
```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "C:/path/to/predictive-maintenance-mcp/.venv/Scripts/python.exe",
      "args": [
        "C:/path/to/predictive-maintenance-mcp/src/machinery_diagnostics_server.py"
      ]
    }
  }
}
```

> **Important**: 
> - Replace `C:/path/to/predictive-maintenance-mcp` with your actual project path
> - Use **absolute paths** for both `command` and `args`
> - On macOS/Linux, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`

4. Restart Claude Desktop

---

## For Developers

### Install Development Dependencies
```bash
pip install -e .[dev]
```

This includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_real_data.py -v
```

### Code Quality Checks
```bash
# Format code
black src tests

# Check formatting
black --check src tests

# Lint code
flake8 src --max-line-length=120

# Type check
mypy src --ignore-missing-imports
```

---

## Troubleshooting

### ImportError: No module named 'mcp'
```bash
pip install --upgrade mcp[cli]
```

### ModuleNotFoundError: No module named 'numpy'
```bash
pip install -e .
```

### Claude Desktop doesn't see the server

1. **Use absolute paths**: Both `command` and `args` must use full absolute paths
   ```json
   {
     "mcpServers": {
       "predictive-maintenance": {
         "command": "C:/full/path/to/.venv/Scripts/python.exe",
         "args": ["C:/full/path/to/src/machinery_diagnostics_server.py"]
       }
     }
   }
   ```

2. **Don't use `cwd`**: Claude Desktop may ignore it, use absolute paths instead

3. **Avoid `python` command**: Use the full path to your virtual environment's Python:
   - ❌ `"command": "python"` (won't work if not in PATH)
   - ✅ `"command": "C:/path/.venv/Scripts/python.exe"` (Windows)
   - ✅ `"command": "/path/.venv/bin/python"` (macOS/Linux)

4. **Module import**: If installed via `pip install -e .`, you can also use:
   - ✅ `"args": ["-m", "predictive_maintenance_mcp"]`
   - ✅ `"args": ["C:/path/src/machinery_diagnostics_server.py"]`

5. **Restart Claude Desktop completely** after config changes

6. **Check logs**: 
   - Windows: `%LOCALAPPDATA%\AnthropicClaude\logs`
   - Look for errors like "spawn python ENOENT" or "No module named"

### Tests failing
```bash
# Ensure dev dependencies installed
pip install -e .[dev]

# Check Python version
python --version  # Should be 3.11 or 3.12

# Run validation
python validate_server.py
```

---

## System Requirements

### Minimum
- Python: 3.11+
- RAM: 4 GB
- Disk: 500 MB (including sample data)

### Recommended
- Python: 3.12
- RAM: 8 GB (for ML model training)
- Disk: 2 GB (for reports and models)

---

## Dependencies

### Core Dependencies
- `mcp[cli]>=1.16.0` - Model Context Protocol framework
- `numpy>=2.3.3` - Numerical computing
- `pandas>=2.3.3` - Data manipulation
- `scipy>=1.16.2` - Scientific computing (FFT, filters)
- `scikit-learn>=1.7.2` - Machine learning
- `plotly>=5.24.0` - Interactive visualizations
- `pydantic>=2.12.0` - Data validation
- `pypdf>=4.0` - PDF text extraction

### Optional Extras

Install any combination using pip extras:

```bash
# Semantic vector search (FAISS + sentence-transformers)
pip install predictive-maintenance-mcp[vector-search]

# OCR for scanned PDF manuals (Tesseract)
pip install predictive-maintenance-mcp[ocr]

# DOCX diagnostic report generation
pip install predictive-maintenance-mcp[docx]

# SSE transport for remote/enterprise deployment (Copilot Studio, networked clients)
pip install predictive-maintenance-mcp[sse]

# Everything (all optional features)
pip install predictive-maintenance-mcp[full]
```

| Extra | Packages | Purpose |
|-------|----------|---------|
| `vector-search` | `faiss-cpu`, `sentence-transformers` | Semantic document search (FAISS). Falls back to TF-IDF when not installed. |
| `ocr` | `pytesseract`, `Pillow`, `pdf2image` | OCR for scanned/image-based PDF manuals. Requires [Poppler](https://github.com/ossamamehmood/Poppler-windows/releases) on system PATH. |
| `docx` | `python-docx` | Generate structured Word (.docx) diagnostic reports. |
| `sse` | `uvicorn` | SSE/HTTP transport for remote MCP clients. Required for `--transport sse`. |
| `full` | All of the above | Install all optional features at once. |

> **Note**: `vector-search` pulls in PyTorch (~2 GB). For lightweight installs, skip it — TF-IDF keyword search works well for technical documentation.

### Development Dependencies
- `pytest>=8.0.0` - Testing
- `pytest-cov>=4.1.0` - Coverage
- `black>=24.0.0` - Formatting
- `flake8>=7.0.0` - Linting
- `mypy>=1.8.0` - Type checking

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

Sample vibration data: CC BY-NC-SA 4.0
