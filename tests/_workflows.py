"""Reading the GitHub Actions workflows, shared by the workflow guards.

``tests/test_ci_gates.py`` and ``tests/test_workflow_policy.py`` both sweep
every file under ``.github/workflows``. They read the files the same way and
name a step the same way, so a step reported by one guard is named
identically by the other.
"""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def workflow_files(directory: Path = WORKFLOWS_DIR) -> list[Path]:
    """Every workflow file; GitHub accepts both extensions."""
    return sorted(p for p in directory.iterdir() if p.suffix in (".yml", ".yaml"))


def load_workflows(directory: Path = WORKFLOWS_DIR) -> dict[str, Any]:
    """``{file name: parsed workflow}`` for every workflow file."""
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in workflow_files(directory)
    }


def step_label(step: dict, index: int) -> str:
    """A step's name, else the action it uses, else its position in the job."""
    return step.get("name") or step.get("uses") or f"step {index}"
