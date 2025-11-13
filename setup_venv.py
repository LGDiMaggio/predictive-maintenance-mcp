#!/usr/bin/env python3
"""
Setup script for creating a clean virtual environment.
This ensures all users can set up the project correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Create clean virtual environment and install dependencies."""
    
    # Fix Windows Unicode output
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("Predictive Maintenance MCP - Clean Environment Setup")
    print("=" * 70)
    print()
    
    # Get project root
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv"
    
    # Check Python version
    print("1. Checking Python version...")
    py_version = sys.version_info
    print(f"   ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version < (3, 11):
        print(f"   ✗ ERROR: Python 3.11+ required (you have {py_version.major}.{py_version.minor})")
        sys.exit(1)
    
    # Remove old venv if exists
    if venv_path.exists():
        print("\n2. Checking for existing virtual environment...")
        print(f"   ⚠️  Virtual environment already exists at: {venv_path}")
        response = input("   Remove and recreate? [y/N]: ").strip().lower()
        
        if response in ('y', 'yes'):
            print("   Removing old virtual environment...")
            try:
                import shutil
                shutil.rmtree(venv_path)
                print("   ✓ Old .venv removed")
            except PermissionError:
                print("   ✗ ERROR: Cannot remove .venv (may be active in another process)")
                print("   Please deactivate any active virtual environments and try again.")
                print("\n   To deactivate:")
                if sys.platform == "win32":
                    print("   - Close terminals with activated .venv")
                    print("   - Or run: deactivate")
                else:
                    print("   - Run: deactivate")
                sys.exit(1)
        else:
            print("   ✓ Using existing virtual environment")
            print("   (skipping venv creation)")
            # Skip to pip upgrade
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
                venv_pip = venv_path / "Scripts" / "pip.exe"
            else:
                venv_python = venv_path / "bin" / "python"
                venv_pip = venv_path / "bin" / "pip"
            
            # Upgrade pip
            print("\n3. Upgrading pip...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            print("   ✓ pip upgraded")
            
            # Install package
            print("\n4. Installing predictive-maintenance-mcp...")
            subprocess.run([str(venv_pip), "install", "-e", "."], cwd=project_root, check=True)
            print("   ✓ Package installed")
            
            # Ask about dev dependencies
            print("\n5. Development dependencies:")
            response = input("   Install dev dependencies (pytest, black, flake8, mypy)? [y/N]: ").strip().lower()
            
            if response in ('y', 'yes'):
                print("   Installing dev dependencies...")
                subprocess.run([str(venv_pip), "install", "-e", ".[dev]"], cwd=project_root, check=True)
                print("   ✓ Dev dependencies installed")
            else:
                print("   ✓ Skipped dev dependencies")
            
            # Verify installation
            print("\n6. Verifying installation...")
            result = subprocess.run(
                [str(venv_python), "-c", "import mcp; import numpy; import pandas; import scipy; import sklearn; import plotly; print('All core packages imported successfully')"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("   ✓ All core packages working")
            else:
                print("   ✗ Import error:")
                print(result.stderr)
                sys.exit(1)
            
            # Final instructions
            print("\n" + "=" * 70)
            print("✅ SETUP COMPLETE!")
            print("=" * 70)
            print("\nExisting virtual environment updated successfully.")
            print("\nNext steps:")
            print("\n1. Virtual environment already active in some terminals")
            print("   No action needed if already activated")
            
            print("\n2. Validate server:")
            print("   python validate_server.py")
            
            print("\n3. Configure Claude Desktop:")
            if sys.platform == "win32":
                print("   .\\setup_claude.ps1")
            else:
                print("   See INSTALL.md for manual setup")
            
            print()
            return
    
    # Create new venv
    print("\n3. Creating new virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print("   ✓ Virtual environment created")
    
    # Determine venv python path
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("\n4. Upgrading pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("   ✓ pip upgraded")
    
    # Install package
    print("\n5. Installing predictive-maintenance-mcp...")
    subprocess.run([str(venv_pip), "install", "-e", "."], cwd=project_root, check=True)
    print("   ✓ Package installed")
    
    # Ask about dev dependencies
    print("\n6. Development dependencies:")
    response = input("   Install dev dependencies (pytest, black, flake8, mypy)? [y/N]: ").strip().lower()
    
    if response in ('y', 'yes'):
        print("   Installing dev dependencies...")
        subprocess.run([str(venv_pip), "install", "-e", ".[dev]"], cwd=project_root, check=True)
        print("   ✓ Dev dependencies installed")
    else:
        print("   ✓ Skipped dev dependencies")
    
    # Verify installation
    print("\n7. Verifying installation...")
    result = subprocess.run(
        [str(venv_python), "-c", "import mcp; import numpy; import pandas; import scipy; import sklearn; import plotly; print('All core packages imported successfully')"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("   ✓ All core packages working")
    else:
        print("   ✗ Import error:")
        print(result.stderr)
        sys.exit(1)
    
    # Final instructions
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("\n1. Activate virtual environment:")
    
    if sys.platform == "win32":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Validate server:")
    print("   python validate_server.py")
    
    print("\n3. Configure Claude Desktop:")
    if sys.platform == "win32":
        print("   .\\setup_claude.ps1")
    else:
        print("   See INSTALL.md for manual setup")
    
    print("\n4. Read documentation:")
    print("   - README.md - Project overview and quick examples")
    print("   - EXAMPLES.md - Complete usage examples and tutorials")
    print("   - INSTALL.md - Detailed installation guide")
    print()

if __name__ == "__main__":
    main()
