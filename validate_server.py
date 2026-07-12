#!/usr/bin/env python3
"""
Quick validation of the Predictive Maintenance MCP server.
Run before testing with Claude Desktop.
"""

import sys
from pathlib import Path

print("Validating Predictive Maintenance MCP server...")
print()

# 1. Check Python version
print("1. Python version:")
print(f"   {sys.version}")
if sys.version_info < (3, 11):
    print("   WARNING: Python 3.11+ recommended")
else:
    print("   OK")
print()

# 2. Check project structure
print("2. Project structure:")
required_paths = [
    "src/server.py",
    "src/mcp_tools",
    "data/signals/real_train",
    "data/signals/real_test",
    "pyproject.toml",
]

all_exist = True
for path in required_paths:
    exists = Path(path).exists()
    status = "OK " if exists else "MISSING"
    print(f"   [{status}] {path}")
    all_exist = all_exist and exists

if not all_exist:
    print("\nMissing required files!")
    sys.exit(1)
print()

# 3. Check data files
print("3. Data files:")
train_files = list(Path("data/signals/real_train").glob("*.csv"))
test_files = list(Path("data/signals/real_test").glob("*.csv"))
print(f"   Training signals: {len(train_files)}")
print(f"   Test signals: {len(test_files)}")
if len(train_files) + len(test_files) < 20:
    print("   WARNING: Expected at least 20 signal files")
else:
    print("   OK")
print("   Supported formats: CSV, MAT, WAV, NPY, Parquet")
print()

# 4. Import check (critical imports only)
print("4. Critical imports:")
critical_imports = ["numpy", "pandas", "scipy.signal", "scipy.stats", "mcp"]

all_imports_ok = True
for module in critical_imports:
    try:
        __import__(module)
        print(f"   [OK ] {module}")
    except ImportError:
        print(f"   [MISSING] {module}")
        all_imports_ok = False

if not all_imports_ok:
    print("\nMissing required packages!")
    print("   Run: uv sync")
    sys.exit(1)
print()

# 5. Server import + registered endpoint inventory
print("5. MCP server:")
try:
    from predictive_maintenance_mcp.server import mcp
except Exception as e:  # noqa: BLE001 — report any import failure
    print(f"   [FAIL] Server import error: {e}")
    sys.exit(1)

tools = list(mcp._tool_manager._tools)
prompts = list(mcp._prompt_manager._prompts)
resources = list(mcp._resource_manager._resources) + list(
    mcp._resource_manager._templates
)
print("   [OK ] from predictive_maintenance_mcp.server import mcp")
print(f"   MCP Tools: {len(tools)}")
print(f"   MCP Prompts: {len(prompts)}")
print(f"   MCP Resources: {len(resources)}")
print()

# 6. Check for key MCP tools
print("6. Key MCP tools:")
key_tools = [
    "generate_fft_report",
    "generate_envelope_report",
    "generate_iso_report",
    "analyze_fft",
    "analyze_envelope",
    "assess_severity",
    "train_anomaly_model",
    "predict_anomalies",
]

missing = [t for t in key_tools if t not in tools]
for tool in key_tools:
    status = "OK " if tool in tools else "MISSING"
    print(f"   [{status}] {tool}()")

print()

# Summary
print("=" * 60)
if missing:
    print("VALIDATION FAILED - missing tools:", ", ".join(missing))
    sys.exit(1)
print("VALIDATION PASSED - Server ready for testing!")
print()
print("Next steps:")
print("1. Restart Claude Desktop")
print("2. Test with: 'List available signals in predictive maintenance'")
print("=" * 60)
