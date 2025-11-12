# Installation Guide

## Quick Start (Recommended)

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

### 4. Verify Installation
```bash
python -c "import mcp; print('MCP installed successfully')"
python validate_server.py
```

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

4. **Don't use `-m` module import**: The package isn't installed as a module:
   - ❌ `"args": ["-m", "machinery_diagnostics_server"]`
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
