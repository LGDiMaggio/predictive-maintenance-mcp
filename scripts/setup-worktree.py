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

Non-interactive by design: agents and CI need to run it unattended. It refuses
to run outside a linked worktree unless you pass --primary, because in the
primary checkout ``.venv`` is the environment every worktree shares.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
SRC_DIR = REPO_ROOT / "src"
PKG = "predictive_maintenance_mcp"


def is_linked_worktree() -> bool | None:
    """Is this checkout a linked worktree rather than the primary one?

    ``--git-dir`` is per-worktree (``.git/worktrees/<name>``) while
    ``--git-common-dir`` is shared (``.git``); they differ exactly for a
    linked worktree. Returns None when git cannot answer, so the caller can
    tell "definitely primary" from "could not determine".
    """
    try:
        git_dir, common = (
            subprocess.run(
                ["git", "rev-parse", flag],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for flag in ("--git-dir", "--git-common-dir")
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return Path(git_dir).resolve() != Path(common).resolve()


def venv_python(venv: Path) -> Path:
    """Interpreter inside ``venv``, on either platform layout."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    # list2cmdline quotes what needs quoting, so the echoed line is
    # copy-pasteable even though this repo's path contains spaces. flush
    # because the output is normally piped (agents, CI): stdout is then
    # block-buffered while the child writes to the fd directly, which
    # otherwise prints every command *after* the output it was labelling.
    print(f"   $ {subprocess.list2cmdline(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)


def has_pip(python: Path) -> bool:
    """Does this interpreter have pip? uv-created venvs do not."""
    return (
        subprocess.run(
            [str(python), "-m", "pip", "--version"],
            capture_output=True,
        ).returncode
        == 0
    )


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
    """Create .venv with the stdlib and editable-install into it.

    Does not trust an existing .venv to be usable. When this runs as the
    fallback after ``uv sync`` failed, the directory it finds was created by
    uv moments earlier -- uv creates the venv before resolving, and does not
    seed pip. Adopting it means running ``python -m pip`` in an interpreter
    that has no pip, so the advertised fallback fails in exactly the
    situation it exists for.
    """
    python = venv_python(VENV_DIR)
    if VENV_DIR.exists() and not (python.exists() and has_pip(python)):
        print("   existing .venv/ has no usable pip -- recreating it")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

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

    # Against src/, not REPO_ROOT. Worktrees live at
    # <primary>/.claude/worktrees/<name> and .venv lives under the root, so
    # is_relative_to(REPO_ROOT) also accepts a sibling worktree's source and
    # a non-editable copy inside our own site-packages -- both of which mean
    # edits to src/ have no effect while this prints success.
    if not resolved.is_relative_to(SRC_DIR):
        print(
            f"\n   FAILED: still importing from outside this checkout's src/.\n"
            f"   this checkout : {SRC_DIR}\n"
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
    parser.add_argument(
        "--primary",
        action="store_true",
        help="allow running in the primary checkout (rebuilds the shared .venv)",
    )
    args = parser.parse_args()

    # Refuse in the primary checkout unless asked twice. There, .venv is not
    # "this checkout's environment" -- it is the one every worktree shares,
    # and `uv sync` prunes to base+dev by default. Running here silently
    # uninstalls the extras (weasyprint, pypdf2, ...), which does not fail
    # anything: it flips the HAS_PDF skipif gates, so the PDF path stops
    # being tested in every checkout at once, with no signal.
    linked = is_linked_worktree()
    if linked is False and not args.primary:
        print(
            f"Refusing to run: {REPO_ROOT} is the primary checkout, not a\n"
            "linked worktree. Its .venv/ is the environment every worktree\n"
            "shares, and rebuilding it here would prune the optional extras\n"
            "(weasyprint, pypdf2, ...) that the suite's skipif gates rely on\n"
            "-- silently turning PDF coverage off everywhere.\n\n"
            "If that is genuinely what you want, pass --primary."
        )
        return 1
    if linked is None:
        print("   (could not ask git whether this is a worktree; continuing)")

    print(f"Setting up an environment for: {REPO_ROOT}")

    if VENV_DIR.exists():
        if args.force:
            print(f"\n1. Removing existing {VENV_DIR.name}/ ...")
            # Rename first so the destructive step is atomic. rmtree deletes
            # depth-first and raises on the first locked file; on a synced or
            # AV-scanned tree of ~18k files that leaves a partial .venv that
            # the next run would silently adopt.
            doomed = VENV_DIR.with_name(f"{VENV_DIR.name}.old-{os.getpid()}")
            try:
                VENV_DIR.rename(doomed)
            except OSError as exc:
                print(
                    f"   ERROR: {exc}\n"
                    "   Close any shell that has this environment activated."
                )
                return 1
            shutil.rmtree(doomed, ignore_errors=True)
            if doomed.exists():
                print(f"   note: could not fully delete {doomed.name}; "
                      "remove it by hand when nothing holds it open")
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
        print(
            "\nFAILED: dependency installation did not complete.\n"
            "   Rerun with --force to rebuild .venv/ from scratch."
        )
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
