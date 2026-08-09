"""The mypy gate is a guard, so its value is entirely in failing.

``tools/check_mypy_baseline.py`` replaced a ``continue-on-error: true`` job
that reported green whatever happened. A gate that silently stopped catching
things would look exactly like a clean tree — the same shape as the defect it
was written to remove, and the same shape as two others this repo has already
paid for (see
``docs/solutions/architecture-patterns/global-config-you-cannot-win-and-claims-only-running-can-verify-2026-08-08.md``).

So the failing paths are tested, not the passing one. Everything here drives
the pure functions against in-memory fixtures; nothing invokes mypy, so the
suite stays fast and these tests cannot be broken by the tree's actual type
debt.

The other half -- that the CI job wiring still lets mypy fail the build --
moved to ``tests/test_ci_gates.py``, which now makes that assertion for
every job in the workflow rather than this one.
"""

import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "tools" / "check_mypy_baseline.py"


@pytest.fixture(scope="module")
def gate():
    """Import the gate by path — tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("check_mypy_baseline", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLE = """\
src/a.py:12: error: Returning Any from function declared to return "str"  [no-any-return]
src/a.py:40: error: Returning Any from function declared to return "int"  [no-any-return]
src/b.py:7: note: this is a note, not an error
src/b.py:7: error: Item "None" has no attribute "encode"  [union-attr]
Found 3 errors in 2 files (checked 40 source files)
"""


class TestParse:
    def test_counts_errors_and_ignores_notes_and_summary(self, gate):
        tally = gate.parse(SAMPLE)
        assert sum(tally.values()) == 3
        assert (
            "src/b.py",
            "union-attr",
            'Item "None" has no attribute "encode"',
        ) in tally

    def test_message_is_part_of_the_key(self, gate):
        """Two errors of one code in one file must not be fungible.

        Keyed on (file, code) alone, fixing one and introducing another nets
        to zero and the gate passes silently.
        """
        tally = gate.parse(SAMPLE)
        same_file_and_code = [k for k in tally if k[0] == "src/a.py"]
        assert len(same_file_and_code) == 2, (
            "both no-any-return errors collapsed into one key — a swap would "
            "be invisible"
        )

    def test_windows_paths_normalize_to_posix(self, gate):
        """The baseline is checked in, so it must read the same everywhere."""
        line = (
            r"src\mcp_tools\report_tools.py:838: error: Bad thing  [arg-type]"
            "\nFound 1 error in 1 file (checked 1 source file)\n"
        )
        assert list(gate.parse(line))[0][0] == "src/mcp_tools/report_tools.py"


class TestRunMypyFailsClosed:
    """The gate must never read "could not check" as "nothing to report"."""

    def _fake_mypy(self, gate, monkeypatch, stdout, returncode=1):
        monkeypatch.setattr(
            gate.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=returncode, stdout=stdout, stderr=""
            ),
        )

    def test_missing_mypy_is_not_a_clean_tree(self, gate, monkeypatch):
        """``python -m mypy`` exits 1 when mypy is absent — same as "errors".

        Without the summary check this printed "OK: 0 mypy error(s)" and
        exited 0, having type-checked nothing.
        """
        self._fake_mypy(gate, monkeypatch, "No module named mypy\n")
        with pytest.raises(SystemExit) as exc:
            gate.run_mypy()
        assert "did not run to completion" in str(exc.value)

    def test_unparseable_output_is_not_a_clean_tree(self, gate, monkeypatch):
        """A format change must fail loudly, not read as zero errors.

        mypy's own count is the cross-check: if the parser sees fewer lines
        than mypy says it emitted, the parser is wrong.
        """
        self._fake_mypy(
            gate,
            monkeypatch,
            "src/a.py:12:5: error: something in a shape we do not parse\n"
            "Found 1 error in 1 file (checked 1 source file)\n",
        )
        with pytest.raises(SystemExit) as exc:
            gate.run_mypy()
        assert "output format changed" in str(exc.value)

    def test_mypy_crash_fails_closed(self, gate, monkeypatch):
        self._fake_mypy(gate, monkeypatch, "internal error\n", returncode=2)
        with pytest.raises(SystemExit) as exc:
            gate.run_mypy()
        assert "failed to run" in str(exc.value)

    def test_a_genuinely_clean_tree_is_accepted(self, gate, monkeypatch):
        self._fake_mypy(
            gate,
            monkeypatch,
            "Success: no issues found in 40 source files\n",
            returncode=0,
        )
        assert gate.parse(gate.run_mypy()) == Counter()


class TestBaselineRoundTrip:
    def test_write_then_load_is_lossless(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "BASELINE_PATH", tmp_path / "baseline.txt")
        tally = Counter(
            {
                ("src/a.py", "no-any-return", 'Returning Any ... "str"'): 2,
                ("src/b.py", "union-attr", 'Item "None" has no attribute "x"'): 1,
            }
        )
        gate.write_baseline(tally)
        assert gate.load_baseline() == tally

    def test_a_malformed_entry_is_rejected_by_name(self, gate, tmp_path, monkeypatch):
        """Fail with a diagnosis, not a bare ValueError traceback."""
        path = tmp_path / "baseline.txt"
        path.write_text("# header\n2\tsrc/a.py\tno-any-return\n", encoding="utf-8")
        monkeypatch.setattr(gate, "BASELINE_PATH", path)
        with pytest.raises(SystemExit) as exc:
            gate.load_baseline()
        assert "malformed entry" in str(exc.value)

    def test_baseline_is_written_with_lf_on_every_platform(
        self, gate, tmp_path, monkeypatch
    ):
        """It is checked in; CRLF from a Windows --update is diff churn."""
        path = tmp_path / "baseline.txt"
        monkeypatch.setattr(gate, "BASELINE_PATH", path)
        gate.write_baseline(Counter({("src/a.py", "code", "msg"): 1}))
        assert b"\r\n" not in path.read_bytes()


class TestTheComparison:
    """The decision the gate actually makes."""

    def _run(self, gate, monkeypatch, current, baseline, capsys):
        monkeypatch.setattr(gate, "run_mypy", lambda: "")
        monkeypatch.setattr(gate, "parse", lambda _output: current)
        monkeypatch.setattr(gate, "load_baseline", lambda: baseline)
        monkeypatch.setattr(sys, "argv", ["check_mypy_baseline.py"])
        code = gate.main()
        return code, capsys.readouterr().out

    def test_a_brand_new_error_fails(self, gate, monkeypatch, capsys):
        code, out = self._run(
            gate,
            monkeypatch,
            Counter({("src/a.py", "no-any-return", "m1"): 1}),
            Counter(),
            capsys,
        )
        assert code == 1
        assert "FAIL" in out

    def test_more_of_an_existing_error_fails(self, gate, monkeypatch, capsys):
        key = ("src/a.py", "no-any-return", "m1")
        code, _ = self._run(
            gate, monkeypatch, Counter({key: 3}), Counter({key: 2}), capsys
        )
        assert code == 1

    def test_swapping_a_fixed_error_for_a_new_one_fails(
        self, gate, monkeypatch, capsys
    ):
        """The hole that (file, code) counting left open.

        One no-any-return in src/a.py fixed, a different one introduced. The
        totals match, so a count-only key saw nothing.
        """
        code, out = self._run(
            gate,
            monkeypatch,
            Counter({("src/a.py", "no-any-return", "NEW message"): 1}),
            Counter({("src/a.py", "no-any-return", "OLD message"): 1}),
            capsys,
        )
        assert code == 1, "a swapped error passed — the message is not in the key"
        assert "NEW message" in out

    def test_a_fixed_error_warns_but_does_not_fail(self, gate, monkeypatch, capsys):
        """Asymmetric on purpose: optional extras differ between CI and local."""
        code, out = self._run(
            gate,
            monkeypatch,
            Counter(),
            Counter({("src/a.py", "no-any-return", "m1"): 1}),
            capsys,
        )
        assert code == 0
        assert "improved" in out

    def test_an_unchanged_tree_passes(self, gate, monkeypatch, capsys):
        key = ("src/a.py", "no-any-return", "m1")
        code, out = self._run(
            gate, monkeypatch, Counter({key: 2}), Counter({key: 2}), capsys
        )
        assert code == 0
        assert "OK" in out
