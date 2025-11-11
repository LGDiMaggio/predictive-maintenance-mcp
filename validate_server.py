#!/usr/bin/env python3
"""
Quick syntax and import validation for machinery diagnostics server.
Run before testing with Claude Desktop.
"""

import sys
from pathlib import Path

print("🔍 Validating machinery diagnostics server...")
print()

# 1. Check Python version
print("1️⃣ Python version:")
print(f"   {sys.version}")
if sys.version_info < (3, 11):
    print("   ⚠️  Warning: Python 3.11+ recommended")
else:
    print("   ✅ OK")
print()

# 2. Check project structure
print("2️⃣ Project structure:")
required_paths = [
    "src/machinery_diagnostics_server.py",
    "data/signals/real_train",
    "data/signals/real_test",
    "pyproject.toml"
]

all_exist = True
for path in required_paths:
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {path}")
    all_exist = all_exist and exists

if not all_exist:
    print("\n❌ Missing required files!")
    sys.exit(1)
print()

# 3. Check data files
print("3️⃣ Data files:")
train_files = list(Path("data/signals/real_train").glob("*.csv"))
test_files = list(Path("data/signals/real_test").glob("*.csv"))
print(f"   Training signals: {len(train_files)}")
print(f"   Test signals: {len(test_files)}")
if len(train_files) + len(test_files) < 20:
    print("   ⚠️  Warning: Expected 20 CSV files")
else:
    print("   ✅ OK")
print()

# 4. Syntax check
print("4️⃣ Syntax validation:")
import py_compile
try:
    py_compile.compile("src/machinery_diagnostics_server.py", doraise=True)
    print("   ✅ No syntax errors")
except py_compile.PyCompileError as e:
    print(f"   ❌ Syntax error: {e}")
    sys.exit(1)
print()

# 5. Import check (critical imports only)
print("5️⃣ Critical imports:")
critical_imports = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("scipy.signal", None),
    ("scipy.stats", None),
]

all_imports_ok = True
for module, alias in critical_imports:
    try:
        if alias:
            exec(f"import {module} as {alias}")
        else:
            exec(f"import {module}")
        print(f"   ✅ {module}")
    except ImportError:
        print(f"   ❌ {module} - NOT INSTALLED")
        all_imports_ok = False

if not all_imports_ok:
    print("\n❌ Missing required packages!")
    print("   Run: uv sync")
    sys.exit(1)
print()

# 6. Server file stats
print("6️⃣ Server file stats:")
server_path = Path("src/machinery_diagnostics_server.py")
content = server_path.read_text(encoding='utf-8')
lines = content.split('\n')
print(f"   Total lines: {len(lines):,}")
print(f"   File size: {server_path.stat().st_size:,} bytes")

# Count tools
tool_count = content.count("@mcp.tool()")
prompt_count = content.count("@mcp.prompt()")
resource_count = content.count("@mcp.resource(")
print(f"   MCP Tools: {tool_count}")
print(f"   MCP Prompts: {prompt_count}")
print(f"   MCP Resources: {resource_count}")
print("   ✅ OK")
print()

# 7. Check for HTML artifact tools
print("7️⃣ HTML Artifact tools:")
html_tools = [
    "generate_iso_chart_html",
    "generate_fft_chart_html",
    "generate_envelope_html",
    "generate_signal_plot_html"
]

for tool in html_tools:
    found = f"def {tool}(" in content
    status = "✅" if found else "❌"
    print(f"   {status} {tool}()")

print()

# Summary
print("=" * 60)
print("✅ VALIDATION PASSED - Server ready for testing!")
print()
print("Next steps:")
print("1. Restart Claude Desktop")
print("2. Test with: 'List available signals in machinery diagnostics'")
print("3. Run HTML artifact test from START_HERE.md")
print("=" * 60)
