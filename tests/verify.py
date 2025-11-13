#!/usr/bin/env python3
"""
Final project verification script.
Performs all checks to ensure the server is ready to use.
"""

import sys
from pathlib import Path


def check_files():
    """Verify presence of essential files."""
    print("\n" + "="*70)
    print("📁 ESSENTIAL FILES VERIFICATION")
    print("="*70)
    
    required_files = [
        "src/machinery_diagnostics_server.py",
        "README.md",
        "EXAMPLES.md",
        "pyproject.toml"
    ]
    
    all_ok = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {file:<40} ({size:>8} bytes)")
        else:
            print(f"   ❌ {file:<40} MISSING!")
            all_ok = False
    
    return all_ok


def check_directories():
    """Verify presence of directories."""
    print("\n" + "="*70)
    print("📂 DIRECTORY VERIFICATION")
    print("="*70)
    
    required_dirs = [
        "src",
        "data",
        "data/signals",
        "tests",
        ".venv"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ MISSING!")
            all_ok = False
    
    return all_ok


def check_imports():
    """Verify Python imports."""
    print("\n" + "="*70)
    print("📦 DEPENDENCIES VERIFICATION")
    print("="*70)
    
    deps = [
        ("numpy", "np"),
        ("scipy", None),
        ("pandas", "pd"),
        ("plotly", None),
        ("pydantic", None),
        ("mcp", None),
    ]
    
    all_ok = True
    for module, alias in deps:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            
            # Get version if available
            try:
                if alias:
                    version = eval(f"{alias}.__version__")
                else:
                    version = eval(f"{module}.__version__")
                print(f"   ✅ {module:<20} v{version}")
            except:
                print(f"   ✅ {module:<20} (installed)")
        except ImportError:
            print(f"   ❌ {module:<20} NOT INSTALLED!")
            all_ok = False
    
    return all_ok


def check_server():
    """Verify server loading."""
    print("\n" + "="*70)
    print("🔧 MCP SERVER VERIFICATION")
    print("="*70)
    
    try:
        sys.path.insert(0, str(Path("src")))
        from machinery_diagnostics_server import mcp, DATA_DIR
        
        print(f"   ✅ Server loaded correctly")
        print(f"   ✅ Data directory: {DATA_DIR}")
        
        # Count components
        # Note: Can't easily count tools/resources/prompts without introspection
        print(f"   ✅ Server configured and ready")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Server loading error: {e}")
        return False


def check_data_files():
    """Verify presence of data files."""
    print("\n" + "="*70)
    print("📊 DATA FILES VERIFICATION")
    print("="*70)
    
    data_dir = Path("data/signals")
    if not data_dir.exists():
        print(f"   ⚠️  Directory {data_dir} does not exist")
        print(f"   ℹ️  Run: uv run python helper.py generate")
        return True
    
    files = list(data_dir.glob("*.csv"))
    
    if not files:
        print(f"   ⚠️  No signal files found")
        print(f"   ℹ️  Run: uv run python helper.py generate")
        return True
    
    print(f"   ✅ {len(files)} signal file(s) found:")
    for f in files:
        size = f.stat().st_size
        print(f"      - {f.name} ({size} bytes)")
    
    return True


def print_summary(results):
    """Print summary."""
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    all_ok = all(results.values())
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {check}")
    
    print("\n" + "="*70)
    
    if all_ok:
        print("🎉 ALL CHECKS PASSED!")
        print("="*70)
        print("\n✅ The server is READY TO USE!")
        print("\n🚀 Next steps:")
        print("   1. Start with: uv run mcp dev src/machinery_diagnostics_server.py")
        print("   2. Or configure Claude Desktop (see README.md)")
        print("\n📖 Documentation:")
        print("   - README.md       # Complete guide")
        print("   - EXAMPLES.md     # Usage examples and tutorials")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("="*70)
        print("\nResolve the issues indicated above.")
        print("For help, check README.md or EXAMPLES.md")
    
    print("\n" + "="*70 + "\n")
    
    return all_ok


def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("🔍 MACHINERY DIAGNOSTICS MCP SERVER - FINAL VERIFICATION")
    print("="*70)
    print("\nVerifying configuration and preparation for use...\n")
    
    results = {
        "Essential files": check_files(),
        "Directories": check_directories(),
        "Python dependencies": check_imports(),
        "MCP Server": check_server(),
        "Data files": check_data_files(),
    }
    
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
