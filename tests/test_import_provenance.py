"""The suite must exercise the checkout it lives in.

A shared virtualenv's editable install pins `predictive_maintenance_mcp` to
one absolute path -- the checkout that ran `pip install -e .`. Every other
git worktree using that venv imports from there instead, and the suite goes
green against source nobody in this tree edited.

`tests/conftest.py` prevents that (a meta_path finder bound to this tree) and
raises at collection if it somehow fails. These tests exist so the guarantee
is a named, discoverable assertion rather than an implicit property of a
conftest block someone might tidy away.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_resolves_inside_this_checkout():
    """`import predictive_maintenance_mcp` must land in this tree's src/."""
    import predictive_maintenance_mcp as pkg

    resolved = Path(pkg.__file__).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"package imported from {resolved}, outside this checkout "
        f"({REPO_ROOT}) -- see tests/conftest.py"
    )
    assert resolved == REPO_ROOT / "src" / "__init__.py"


def test_submodules_resolve_inside_this_checkout():
    """Sub-packages inherit the parent's __path__, so they must follow it.

    Checked explicitly because the pin only names the top-level package: if
    sub-package resolution ever fell back to the editable finder, the top
    level would still look correct while the code under test came from
    elsewhere.
    """
    from predictive_maintenance_mcp import mcp_tools, server

    for module in (server, mcp_tools):
        resolved = Path(module.__file__).resolve()
        assert resolved.is_relative_to(REPO_ROOT / "src"), (
            f"{module.__name__} imported from {resolved}, outside "
            f"{REPO_ROOT / 'src'}"
        )


def test_declared_version_matches_this_checkout():
    """`__version__` must come from this tree, not a stale installed copy.

    This is the symptom that surfaced in review sessions -- a reported
    version matching no file in the worktree.
    """
    import predictive_maintenance_mcp as pkg

    source = (REPO_ROOT / "src" / "__init__.py").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in source.splitlines()
        if line.startswith("__version__")
    )
    assert pkg.__version__ == declared


def test_pyproject_version_matches_package_version():
    """Catches the other half of the stale-install symptom.

    An editable install's finder filename carries the version it was built
    at; when that drifts from pyproject, the environment is stale even if the
    path happens to be right.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in pyproject.splitlines()
        if line.startswith("version = ")
    )

    import predictive_maintenance_mcp as pkg

    assert (
        pkg.__version__ == declared
    ), f"src/__init__.py says {pkg.__version__}, pyproject.toml says {declared}"


@pytest.mark.parametrize("name", ["numpy", "scipy", "pandas"])
def test_third_party_imports_are_unaffected(name):
    """The pin must claim exactly one name and leave everything else alone."""
    module = __import__(name)
    resolved = Path(module.__file__).resolve()
    assert not resolved.is_relative_to(REPO_ROOT / "src")
