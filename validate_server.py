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
# The sample signals come with a clone of the repository; the source archives
# and the PyPI package leave them out. The server runs on the operator's own
# files without them, so step 3 warns instead of failing.
sample_paths = ["data/signals/real_train", "data/signals/real_test"]

all_exist = True
for path in required_paths:
    exists = Path(path).exists()
    if not exists and path in sample_paths:
        print(f"   [MISSING] {path} (sample signals: see step 3)")
        continue
    status = "OK " if exists else "MISSING"
    print(f"   [{status}] {path}")
    all_exist = all_exist and exists

if not all_exist:
    print("\nMissing required files!")
    sys.exit(1)
print()

# 3. Check data files
print("3. Data files:")
if all(Path(path).is_dir() for path in sample_paths):
    train_files = list(Path("data/signals/real_train").glob("*.csv"))
    test_files = list(Path("data/signals/real_test").glob("*.csv"))
    print(f"   Training signals: {len(train_files)}")
    print(f"   Test signals: {len(test_files)}")
    if len(train_files) + len(test_files) < 20:
        print("   WARNING: Expected at least 20 signal files")
    else:
        print("   OK")
else:
    print("   Sample signals: not found")
    print("   WARNING: the sample signals come with a clone of the repository:")
    print("   git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git")
    print("   Your own signal files go in data/signals/.")
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
# Frozen registered surface (tools, resources, prompts). The same numbers are
# pinned by tests/test_version_alignment.py::test_expected_final_surface;
# update both together when the registered surface changes.
EXPECTED_TOOLS, EXPECTED_RESOURCES, EXPECTED_PROMPTS = 39, 0, 3
EXPECTED_LABEL = f"{EXPECTED_TOOLS}/{EXPECTED_RESOURCES}/{EXPECTED_PROMPTS}"

print("   [OK ] from predictive_maintenance_mcp.server import mcp")
print(f"   MCP Tools: {len(tools)} (expected {EXPECTED_TOOLS})")
print(f"   MCP Prompts: {len(prompts)} (expected {EXPECTED_PROMPTS})")
print(f"   MCP Resources: {len(resources)} (expected {EXPECTED_RESOURCES})")

counts_ok = (len(tools), len(resources), len(prompts)) == (
    EXPECTED_TOOLS,
    EXPECTED_RESOURCES,
    EXPECTED_PROMPTS,
)
if not counts_ok:
    print(
        f"   [FAIL] Registered surface differs from the frozen {EXPECTED_LABEL} target"
    )
print()

# 6. Check for key MCP tools
print("6. Key MCP tools:")
key_tools = [
    "load_signal",
    "list_signals",
    "clear_signals",
    "generate_test_signal",
    "analyze_fft",
    "analyze_envelope",
    "assess_severity",
    "check_bearing_faults",
    "diagnose_vibration",
    "train_anomaly_model",
    "predict_anomalies",
    "generate_fft_report",
    "generate_envelope_report",
    "generate_iso_report",
    "list_html_reports",
    "analyze_signal_trend",
    "estimate_rul",
    "generate_maintenance_recommendations",
]

missing = [t for t in key_tools if t not in tools]
for tool in key_tools:
    status = "OK " if tool in tools else "MISSING"
    print(f"   [{status}] {tool}()")

print()

# Summary
print("=" * 60)
if missing or not counts_ok:
    if missing:
        print("VALIDATION FAILED - missing tools:", ", ".join(missing))
    if not counts_ok:
        print(
            "VALIDATION FAILED - endpoint counts "
            f"{len(tools)}/{len(resources)}/{len(prompts)} != {EXPECTED_LABEL}"
        )
    sys.exit(1)
print("VALIDATION PASSED - Server ready for testing!")
print()
print("Next steps:")
print("1. Restart Claude Desktop")
print("2. Test with: 'List available signals in predictive maintenance'")
print("=" * 60)
