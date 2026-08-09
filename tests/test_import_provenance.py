"""The suite must exercise the checkout it lives in.

A shared virtualenv's editable install pins `predictive_maintenance_mcp` to
one absolute path -- the checkout that ran `pip install -e .`. Every other
git worktree using that venv imports from there instead, and the suite goes
green against source nobody in this tree edited.

`tests/conftest.py` prevents that (a meta_path finder bound to this tree) and
raises at collection if it somehow fails. These tests exist so the guarantee
is a named, discoverable assertion rather than an implicit property of a
conftest block someone might tidy away.

A caveat worth stating, because it decides what these tests are worth. The
observational tests below -- "the package resolved into this tree" -- are
only meaningful where a competing finder points somewhere else, i.e. in a
worktree sharing another checkout's venv. In CI, and in the primary checkout,
`pip install -e .` already resolves here, so they would pass with the pin
deleted. That is the whole environment CI runs in. So the mechanism itself is
tested directly, against a synthetic competing finder, by
`TestTheGuardItself` -- otherwise removing the one line that does the work
would stay green forever.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
PKG = "predictive_maintenance_mcp"

#: The pin is deliberately off under this flag, so the invariant these tests
#: assert does not hold. Skip rather than fail: conftest advertises the flag,
#: and a documented switch that reddens the suite is not a usable switch.
_PIN_DISABLED = os.environ.get("PMM_TEST_ALLOW_INSTALLED") == "1"
needs_pin = pytest.mark.skipif(
    _PIN_DISABLED, reason="PMM_TEST_ALLOW_INSTALLED=1 disables the pin"
)


@needs_pin
def test_package_resolves_inside_this_checkout():
    """`import predictive_maintenance_mcp` must land in this tree's src/."""
    import predictive_maintenance_mcp as pkg

    resolved = Path(pkg.__file__).resolve()
    assert resolved == (SRC_DIR / "__init__.py").resolve(), (
        f"package imported from {resolved}, not this checkout's src "
        f"({SRC_DIR}) -- see tests/conftest.py"
    )


@needs_pin
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
        assert resolved.is_relative_to(
            SRC_DIR
        ), f"{module.__name__} imported from {resolved}, outside {SRC_DIR}"


@needs_pin
def test_a_child_process_also_resolves_into_this_checkout():
    """Subprocesses load no conftest, so the pin has to travel explicitly.

    Tests that shell out do so because in-process observation could not
    answer their question (see tests/test_no_client_logging.py, which checks
    which file descriptor log records reach). A child that silently imported
    the primary checkout would answer about a different tree -- the same
    defect this module guards, one process boundary away.
    """
    import subprocess

    from conftest import SUBPROCESS_PIN

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            SUBPROCESS_PIN + f"import {PKG} as p; print(p.__file__)",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    resolved = Path(proc.stdout.strip()).resolve()
    assert (
        resolved == (SRC_DIR / "__init__.py").resolve()
    ), f"child process imported {resolved}, not this checkout"


class TestTheGuardItself:
    """Exercise the mechanism, not just its happy outcome.

    Everything above observes that resolution landed in the right place. In
    CI that is true with or without the pin, so those tests cannot notice its
    removal. These two drive the guard directly.
    """

    @needs_pin
    def test_the_pin_is_actually_installed_and_first(self):
        """The finder must exist and sit ahead of the editable install's."""
        from conftest import _LocalTreeFinder

        assert _LocalTreeFinder in sys.meta_path, (
            "the meta_path pin is gone -- resolution is back to whatever the "
            "editable install hardcoded"
        )
        names = [getattr(f, "__name__", type(f).__name__) for f in sys.meta_path]
        pin = names.index("_LocalTreeFinder")
        editable = next(
            (i for i, n in enumerate(names) if "editable" in n.lower()), None
        )
        if editable is not None:
            assert (
                pin < editable
            ), f"pin at {pin} sits behind the editable finder at {editable}"

    def test_the_tripwire_fires_when_the_package_resolves_elsewhere(
        self, tmp_path, monkeypatch
    ):
        """The backstop must actually raise, not just exist.

        This is the only defence left when something imports the package
        before conftest loads, so `sys.modules` wins and the pin cannot help.
        Driven by pointing the check at a decoy tree.

        Runs even under PMM_TEST_ALLOW_INSTALLED=1 -- the flag is neutralised
        locally rather than skipped, because the question here is whether the
        code raises, not whether it is currently switched on.
        """
        import conftest

        monkeypatch.setattr(conftest, "_ALLOW_INSTALLED", False)

        decoy = tmp_path / "elsewhere" / "src"
        decoy.mkdir(parents=True)
        (decoy / "__init__.py").write_text("__version__ = '0.0.0'\n")

        spec = importlib.util.spec_from_file_location(
            PKG, decoy / "__init__.py", submodule_search_locations=[str(decoy)]
        )
        impostor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(impostor)

        saved = sys.modules.get(PKG)
        sys.modules[PKG] = impostor
        try:
            with pytest.raises(RuntimeError, match="resolved OUTSIDE"):
                conftest._assert_import_provenance()
        finally:
            if saved is not None:
                sys.modules[PKG] = saved
            else:
                del sys.modules[PKG]

    def test_the_pin_claims_exactly_one_name(self):
        """An over-broad finder would hijack unrelated imports."""
        from conftest import _LocalTreeFinder

        for name in ("numpy", f"{PKG}_extra", f"{PKG}.mcp_tools", "json"):
            assert (
                _LocalTreeFinder.find_spec(name) is None
            ), f"the pin claimed {name!r}; it must only claim {PKG!r}"
        assert _LocalTreeFinder.find_spec(PKG) is not None


@needs_pin
def test_declared_version_matches_this_checkout():
    """`__version__` must come from this tree, not a stale installed copy.

    This is the symptom that surfaced in review sessions -- a reported
    version matching no file in the worktree.
    """
    import predictive_maintenance_mcp as pkg

    source = (SRC_DIR / "__init__.py").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in source.splitlines()
        if line.startswith("__version__")
    )
    assert pkg.__version__ == declared


def test_pyproject_version_matches_source_version():
    """Two declarations of one fact, in this tree, kept in lockstep.

    Named for what it does. It used to claim it caught a stale *install*,
    which it could not: both files it reads live in this checkout and move in
    the same commit, so it was green while the venv carried 0.11.0 metadata
    against 0.12.0 source. The installed distribution is checked separately
    below.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in pyproject.splitlines()
        if line.startswith("version = ")
    )
    source = (SRC_DIR / "__init__.py").read_text(encoding="utf-8")
    in_source = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in source.splitlines()
        if line.startswith("__version__")
    )
    assert (
        in_source == declared
    ), f"src/__init__.py says {in_source}, pyproject.toml says {declared}"


def test_installed_distribution_is_not_stale():
    """The one drift the pin deliberately hides.

    Binding imports to this tree makes the suite independent of the installed
    distribution -- which also means a `pip install -e .` that succeeded
    against an older version becomes invisible to every other test. That is
    not hypothetical: this venv carried 0.11.0 metadata while the source said
    0.12.0, and the whole provenance suite was green.

    A warning rather than a failure: a stale editable install does not affect
    what the tests exercise, and CI installs fresh every run.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("predictive-maintenance-mcp")
    except PackageNotFoundError:
        pytest.skip("package is not installed as a distribution")

    source = (SRC_DIR / "__init__.py").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip().strip("\"'")
        for line in source.splitlines()
        if line.startswith("__version__")
    )
    if installed != declared:
        pytest.skip(
            f"stale editable install: distribution metadata says {installed}, "
            f"source says {declared}. Harmless for the suite (imports are "
            f"pinned to this tree), but `python -m predictive_maintenance_mcp` "
            f"and anything reading importlib.metadata will disagree. "
            f"Reinstall with: pip install -e ."
        )


@pytest.mark.parametrize("name", ["numpy", "scipy", "pandas"])
def test_third_party_imports_are_unaffected(name):
    """The pin must claim exactly one name and leave everything else alone."""
    module = __import__(name)
    resolved = Path(module.__file__).resolve()
    assert not resolved.is_relative_to(SRC_DIR)
