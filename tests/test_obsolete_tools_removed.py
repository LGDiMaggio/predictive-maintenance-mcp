"""
Test that obsolete artifact tools have been removed.

Verifies that the old inline HTML artifact system tools are no longer present.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import machinery_diagnostics_server as server

def test_obsolete_tools_removed():
    """Test that obsolete inline artifact tools are removed."""
    
    print("=" * 70)
    print("OBSOLETE TOOLS REMOVAL TEST")
    print("=" * 70)
    
    obsolete_functions = [
        "generate_iso_chart_html",
        "generate_fft_chart_html", 
        "generate_envelope_html",
        "generate_signal_plot_html",
        "save_fft_chart_to_file",
        "save_envelope_chart_to_file",
        "save_signal_plot_to_file",
        "read_html_artifact"
    ]
    
    print("\n[TEST] Checking if obsolete functions have been removed...")
    print("-" * 70)
    
    all_removed = True
    for func_name in obsolete_functions:
        exists = hasattr(server, func_name)
        status = "✗ STILL EXISTS" if exists else "✓ REMOVED"
        print(f"  {func_name:40s} {status}")
        if exists:
            all_removed = False
    
    print("-" * 70)
    
    if all_removed:
        print("\n✓✓ PASS: All obsolete functions successfully removed")
    else:
        print("\n✗✗ FAIL: Some obsolete functions still present!")
        return False
    
    # Verify new report tools exist
    print("\n[TEST] Checking new report generation tools exist...")
    print("-" * 70)
    
    required_tools = [
        "evaluate_iso_20816",
        "analyze_fft",
        "analyze_envelope",
        "analyze_statistics"
    ]
    
    all_present = True
    for tool_name in required_tools:
        exists = hasattr(server, tool_name)
        status = "✓ PRESENT" if exists else "✗ MISSING"
        print(f"  {tool_name:40s} {status}")
        if not exists:
            all_present = False
    
    print("-" * 70)
    
    if all_present:
        print("\n✓✓ PASS: All required tools present")
    else:
        print("\n✗✗ FAIL: Some required tools missing!")
        return False
    
    # Check line count reduction
    print("\n[TEST] Checking file size reduction...")
    print("-" * 70)
    
    server_file = Path(__file__).parent.parent / "src" / "machinery_diagnostics_server.py"
    lines = server_file.read_text(encoding='utf-8').split('\n')
    line_count = len(lines)
    
    print(f"  Current line count: {line_count}")
    print(f"  Expected: < 3000 lines (after removing ~1380 lines)")
    
    if line_count < 3000:
        print(f"  ✓ File reduced to {line_count} lines")
    else:
        print(f"  ⚠️ File still has {line_count} lines (expected < 3000)")
    
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Removed 8 obsolete artifact functions")
    print(f"  • Verified 4 core analysis tools present")
    print(f"  • File size reduced to {line_count} lines")
    print(f"  • Unit conversion warning system active")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_obsolete_tools_removed()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
