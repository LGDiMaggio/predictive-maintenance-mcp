"""Guards for the dependency helpers (``floors`` and ``lock-reqs``).

Two subcommands feed the weekly audit's dependency legs:

* ``floors`` turns the ``>=`` bounds of ``[project.dependencies]`` into a
  constraints file, so one suite leg runs on exactly the versions the project
  claims to support. A dependency without a floor is an error, never skipped:
  a skipped entry would let that leg resolve the newest release and still be
  reported as "floors".
* ``lock-reqs`` turns ``uv.lock`` into pinned requirements for pip-audit.
  uv.lock is a universal lock, so one name can appear once per Python range;
  each pin carries its fork's markers so exactly one applies on any
  interpreter (pip-audit refuses duplicate names after marker evaluation).

The ``security`` normalizer that reads pip-audit's output is guarded in
``tests/test_audit_security.py``. Everything here is offline: fixtures live
in ``tests/fixtures/audit``.
"""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

from tools.audit import __main__ as audit_main
from tools.audit import deps

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audit"
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCK_SAMPLE = FIXTURES / "uv_lock_sample.toml"

INTERPRETERS = ("3.11.9", "3.12.4", "3.13.1", "3.14.0", "3.15.0")


def _write_pyproject(tmp_path, dependencies):
    body = "\n".join(f'    "{dep}",' for dep in dependencies)
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "demo"\nversion = "0"\ndependencies = [\n{body}\n]\n',
        encoding="utf-8",
    )
    return path


def _declared_floor(name):
    """``name==<floor>`` from the ``>=`` bound as written in pyproject.toml.

    Read with tomllib and a regex, independently of ``deps.floor_pins``, so
    a wrong pin cannot supply its own expectation.
    """
    with open(PYPROJECT, "rb") as handle:
        entries = tomllib.load(handle)["project"]["dependencies"]
    (entry,) = [item for item in entries if Requirement(item).name == name]
    (floor,) = re.findall(r">=\s*([^,;\s]+)", entry.split(";", 1)[0])
    return f"{name}=={floor}"


def _env(full_version):
    major, minor = full_version.split(".")[:2]
    return {"python_full_version": full_version, "python_version": f"{major}.{minor}"}


# --------------------------------------------------------------------------
# floors
# --------------------------------------------------------------------------


class TestFloors:
    def test_real_pyproject_pins_every_declared_floor(self):
        requirements = deps.read_declared(PYPROJECT)
        pins = deps.floor_pins(requirements)

        for name in ("mcp", "numpy", "scipy", "pydantic", "pypdf"):
            assert _declared_floor(name) in pins
        assert len(pins) == len(requirements)
        assert not any("[" in pin or ";" in pin for pin in pins), pins

    def test_every_pin_parses_as_an_exact_requirement(self):
        for pin in deps.floor_pins(deps.read_declared(PYPROJECT)):
            req = Requirement(pin)
            (spec,) = req.specifier
            assert spec.operator == "=="

    def test_extras_and_markers_are_stripped(self):
        reqs = [Requirement('mcp[cli]>=2.0.0,<3.0.0; python_version >= "3.10"')]
        assert deps.floor_pins(reqs) == ["mcp==2.0.0"]

    def test_dependency_without_a_floor_is_an_error_naming_it(self):
        reqs = [
            Requirement("numpy>=2.3.3"),
            Requirement("unbounded"),
            Requirement("pinned==1.0"),
            Requirement("compatible~=1.4"),
        ]
        with pytest.raises(ValueError) as excinfo:
            deps.floor_pins(reqs)
        message = str(excinfo.value)
        for name in ("unbounded", "pinned", "compatible"):
            assert name in message
        assert "numpy" not in message

    def test_cli_writes_constraints_file(self, tmp_path):
        out = tmp_path / "constraints.txt"
        code = audit_main.main(
            ["floors", "--pyproject", str(PYPROJECT), "--output", str(out)]
        )
        assert code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert _declared_floor("mcp") in lines and _declared_floor("numpy") in lines

    def test_cli_prints_to_stdout_without_output(self, capsys):
        code = audit_main.main(["floors", "--pyproject", str(PYPROJECT)])
        assert code == 0
        assert _declared_floor("mcp") in capsys.readouterr().out.splitlines()

    def test_cli_refuses_a_dependency_without_floor(self, tmp_path, capsys):
        path = _write_pyproject(tmp_path, ["numpy>=2.3.3", "no-floor-pkg<5"])
        code = audit_main.main(["floors", "--pyproject", str(path)])
        assert code != 0
        assert "no-floor-pkg" in capsys.readouterr().err


# --------------------------------------------------------------------------
# lock-reqs
# --------------------------------------------------------------------------


class TestLockReqs:
    def _pins(self):
        return deps.lock_pins(tomllib.loads(LOCK_SAMPLE.read_text(encoding="utf-8")))

    def test_excludes_project_and_non_registry_sources(self):
        names = [Requirement(pin).name for pin in self._pins()]
        assert "predictive-maintenance-mcp" not in names
        for excluded in ("local-helper", "path-dep", "git-dep", "workspace-root"):
            assert excluded not in names
        assert "alpha" in names and "beta-pkg" in names

    def test_unforked_package_is_a_plain_pin(self):
        assert "alpha==1.0.0" in self._pins()
        assert "beta-pkg==2.5.0" in self._pins()

    def test_forked_package_gives_one_applicable_pin_per_interpreter(self):
        sphinx = [Requirement(p) for p in self._pins() if p.startswith("sphinx==")]
        assert len(sphinx) == 2
        assert all(req.marker is not None for req in sphinx)
        for full_version in INTERPRETERS:
            applicable = [
                str(req.specifier)
                for req in sphinx
                if req.marker.evaluate(_env(full_version))
            ]
            assert len(applicable) == 1, (full_version, applicable)
        on_311 = [r for r in sphinx if r.marker.evaluate(_env("3.11.9"))]
        assert str(on_311[0].specifier) == "==9.0.4"

    def test_marker_text_is_valid_pep508(self):
        for pin in self._pins():
            if ";" in pin:
                Marker(pin.split(";", 1)[1])

    def test_lock_without_registry_packages_is_an_error(self):
        lock = {
            "version": 1,
            "package": [
                {
                    "name": "predictive-maintenance-mcp",
                    "version": "0.13.0",
                    "source": {"editable": "."},
                }
            ],
        }
        with pytest.raises(ValueError):
            deps.lock_pins(lock)
        with pytest.raises(ValueError):
            deps.lock_pins({"version": 1})

    def test_real_lock_yields_registry_pins(self):
        lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
        pins = deps.lock_pins(lock)
        names = [Requirement(p).name for p in pins]
        assert "predictive-maintenance-mcp" not in names
        assert "mcp" in names and "numpy" in names
        for full_version in ("3.11.9", "3.12.4", "3.14.0"):
            applicable = [
                Requirement(p).name
                for p in pins
                if Requirement(p).marker is None
                or Requirement(p).marker.evaluate(_env(full_version))
            ]
            assert len(applicable) == len(set(applicable)), full_version

    def test_cli_writes_requirements(self, tmp_path):
        out = tmp_path / "lock-reqs.txt"
        code = audit_main.main(
            ["lock-reqs", "--lock", str(LOCK_SAMPLE), "--output", str(out)]
        )
        assert code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "alpha==1.0.0" in lines

    def test_cli_empty_lock_exits_non_zero(self, tmp_path, capsys):
        empty = tmp_path / "uv.lock"
        empty.write_text("version = 1\n", encoding="utf-8")
        code = audit_main.main(["lock-reqs", "--lock", str(empty)])
        assert code != 0
        assert "error" in capsys.readouterr().err
