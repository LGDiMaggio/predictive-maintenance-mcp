#!/usr/bin/env python3
"""Give this checkout its own virtualenv, bound to its own source.

WHEN YOU NEED THIS
------------------
Only when you need to *run* this checkout outside pytest: start the MCP
server, open a REPL, run ``validate_server.py``, point a Claude Desktop
config at it. For running the test suite you do not need this at all --
``tests/conftest.py`` binds the package to its own tree, so ``pytest`` is
already correct in every worktree.

WHY IT EXISTS
-------------
``pip install -e .`` writes a finder whose MAPPING hardcodes one absolute
path: the checkout that ran the install. Sharing that venv across git
worktrees means every worktree imports the *primary* checkout's source. Under
pytest the conftest pin covers it; a bare ``python -c "import
predictive_maintenance_mcp"`` is still wrong, and silently so.

COST
----
A full environment for this project is roughly 500 MB across ~18k files. On a
OneDrive- or Dropbox-synced tree that is 18k files the sync client will chew
through, per worktree. That is why this is a deliberate command and not
something that happens automatically.

USAGE
-----
    python scripts/setup-worktree.py            # uv if available, else venv+pip
    python scripts/setup-worktree.py --no-uv    # force stdlib venv + pip
    python scripts/setup-worktree.py --force    # recreate an existing .venv

Non-interactive by design: agents and CI need to run it unattended. It never
touches any checkout other than the one it lives in.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
PKG = "predictive_maintenance_mcp"


def venv_python(venv: Path) -> Path:
    """Interpreter inside ``venv``, on either platform layout."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)


def install_with_uv() -> bool:
    """Sync from uv.lock. Returns False if uv declined, so we can fall back.

    ``--frozen`` keeps the lockfile out of the diff: setting up an
    environment should never show up as a repo change. If the lock genuinely
    needs regenerating, that is a decision for whoever changed the
    dependencies, not for a setup script.
    """
    result = run(["uv", "sync", "--extra", "dev", "--frozen"])
    if result.returncode == 0:
        return True

    print(
        "\n   uv sync --frozen failed. Usually this means uv.lock is behind\n"
        "   pyproject.toml. Regenerate it deliberately with:\n"
        "       uv sync --extra dev\n"
        "   ...or rerun this script with --no-uv to use venv + pip.\n"
    )
    return False


def install_with_pip() -> bool:
    """Create .venv with the stdlib and editable-install into it."""
    if not VENV_DIR.exists():
        if run([sys.executable, "-m", "venv", str(VENV_DIR)]).returncode != 0:
            return False

    python = venv_python(VENV_DIR)
    if not python.exists():
        print(f"   ERROR: no interpreter at {python}")
        return False

    steps = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        [str(python), "-m", "pip", "install", "-e", ".[dev]"],
    ]
    return all(run(step).returncode == 0 for step in steps)


def verify() -> bool:
    """Confirm the new environment imports THIS checkout's source.

    The whole point of the exercise, so it is checked rather than assumed --
    an editable install that silently resolved elsewhere is exactly the
    failure this script exists to prevent.
    """
    python = venv_python(VENV_DIR)
    if not python.exists():
        print(f"   ERROR: no interpreter at {python}")
        return False

    probe = f"import {PKG} as p; print(p.__file__); print(p.__version__)"
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"   ERROR: could not import {PKG}\n{result.stderr}")
        return False

    lines = result.stdout.strip().splitlines()
    resolved = Path(lines[0]).resolve()
    version = lines[1] if len(lines) > 1 else "?"

    print(f"   package  : {resolved}")
    print(f"   version  : {version}")

    if not resolved.is_relative_to(REPO_ROOT):
        print(
            f"\n   FAILED: still importing from outside this checkout.\n"
            f"   this checkout : {REPO_ROOT}\n"
            f"   imported from : {resolved}\n"
            f"   Try again with --force to rebuild .venv from scratch."
        )
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a virtualenv bound to this checkout's source."
    )
    parser.add_argument(
        "--no-uv",
        action="store_true",
        help="skip uv even if installed; use python -m venv + pip",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete an existing .venv first",
    )
    args = parser.parse_args()

    print(f"Setting up an environment for: {REPO_ROOT}")

    if VENV_DIR.exists():
        if args.force:
            print(f"\n1. Removing existing {VENV_DIR.name}/ ...")
            try:
                shutil.rmtree(VENV_DIR)
            except OSError as exc:
                print(
                    f"   ERROR: {exc}\n"
                    "   Close any shell that has this environment activated."
                )
                return 1
        else:
            print(f"\n1. Reusing existing {VENV_DIR.name}/ (--force to recreate)")
    else:
        print(f"\n1. No {VENV_DIR.name}/ yet -- creating one (~500 MB, ~18k files)")

    use_uv = not args.no_uv and shutil.which("uv") is not None
    print(f"\n2. Installing with {'uv' if use_uv else 'venv + pip'} ...")

    installed = False
    if use_uv:
        installed = install_with_uv()
        if not installed:
            print("\n2b. Falling back to venv + pip ...")
    if not installed:
        installed = install_with_pip()

    if not installed:
        print("\nFAILED: dependency installation did not complete.")
        return 1

    print("\n3. Verifying import provenance ...")
    if not verify():
        return 1

    activate = (
        ".venv\\Scripts\\activate"
        if sys.platform == "win32"
        else "source .venv/bin/activate"
    )
    print(
        "\nDone. This checkout now imports its own source.\n"
        f"\n  Activate : {activate}\n"
        "  Test     : pytest -v\n"
        "  Validate : python validate_server.py\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
