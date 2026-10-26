"""``validate_server.py`` in a clone and in a tree extracted from a source archive.

INSTALL.md tells users to run the script. The sample signals come with a clone
of the repository; the source archives (``.gitattributes`` export-ignore) and
the PyPI package leave them out. In an archive tree the script must say where
the samples come from and still run its other checks, which keep failing on
what is really broken. In a clone its output does not change.

The script runs as a subprocess in a small synthetic tree: it reads paths
relative to its working directory and imports the server, pinned to this
checkout through ``conftest.SUBPROCESS_PIN``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SUBPROCESS_PIN

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "validate_server.py"
SETUP_SCRIPT = REPO_ROOT / "setup_claude.ps1"

REMEDY = [
    "   Sample signals: not found",
    "   WARNING: the sample signals come with a clone of the repository:",
    "   git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git",
    "   Your own signal files go in data/signals/.",
]


def _tree(tmp_path: Path, samples: bool, pyproject: bool = True) -> Path:
    """The paths the script checks; the sample folders only in a clone."""
    root = tmp_path / "tree"
    (root / "src" / "mcp_tools").mkdir(parents=True)
    (root / "src" / "server.py").write_text("", encoding="utf-8")
    if pyproject:
        (root / "pyproject.toml").write_text("", encoding="utf-8")
    signals = root / "data" / "signals"
    signals.mkdir(parents=True)
    (signals / ".gitkeep").write_text("", encoding="utf-8")
    if samples:
        for folder, count in (("real_train", 14), ("real_test", 6)):
            (signals / folder).mkdir()
            for index in range(count):
                (signals / folder / f"s{index}.csv").write_text(
                    "0.0\n", encoding="utf-8"
                )
    return root


def _validate(root: Path) -> subprocess.CompletedProcess:
    # The child imports the server; without the pin a worktree would import
    # the checkout the shared venv's editable install points at.
    run_script = f"import runpy; runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    return subprocess.run(
        [sys.executable, "-c", SUBPROCESS_PIN + run_script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PDM_PROJECT_DIR": str(root)},
    )


def _lines(proc: subprocess.CompletedProcess) -> list:
    return proc.stdout.splitlines()


class TestValidateServer:
    def test_an_archive_tree_warns_names_the_clone_and_runs_every_check(self, tmp_path):
        proc = _validate(_tree(tmp_path, samples=False))
        lines = _lines(proc)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (
            "   [MISSING] data/signals/real_train (sample signals: see step 3)" in lines
        )
        assert (
            "   [MISSING] data/signals/real_test (sample signals: see step 3)" in lines
        )
        start = lines.index(REMEDY[0])
        assert lines[start : start + len(REMEDY)] == REMEDY
        assert "5. MCP server:" in lines
        assert "VALIDATION PASSED - Server ready for testing!" in lines

    def test_a_clone_reports_the_samples_as_before(self, tmp_path):
        proc = _validate(_tree(tmp_path, samples=True))
        lines = _lines(proc)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "   [OK ] data/signals/real_train" in lines
        assert "   [OK ] data/signals/real_test" in lines
        start = lines.index("3. Data files:")
        assert lines[start + 1 : start + 4] == [
            "   Training signals: 14",
            "   Test signals: 6",
            "   OK",
        ]
        assert not any("sample signals" in line for line in lines)

    def test_a_missing_project_file_still_fails(self, tmp_path):
        proc = _validate(_tree(tmp_path, samples=False, pyproject=False))
        assert proc.returncode == 1
        assert "   [MISSING] pyproject.toml" in _lines(proc)
        assert "Missing required files!" in proc.stdout

    def test_a_missing_server_module_still_fails(self, tmp_path):
        root = _tree(tmp_path, samples=False)
        (root / "src" / "server.py").unlink()
        proc = _validate(root)
        assert proc.returncode == 1
        assert "   [MISSING] src/server.py" in _lines(proc)

    def test_one_sample_folder_alone_is_still_reported_as_missing_samples(
        self, tmp_path
    ):
        root = _tree(tmp_path, samples=True)
        for path in (root / "data" / "signals" / "real_test").iterdir():
            path.unlink()
        (root / "data" / "signals" / "real_test").rmdir()
        proc = _validate(root)
        lines = _lines(proc)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "   [OK ] data/signals/real_train" in lines
        assert (
            "   [MISSING] data/signals/real_test (sample signals: see step 3)" in lines
        )
        start = lines.index(REMEDY[0])
        assert lines[start : start + len(REMEDY)] == REMEDY


@pytest.fixture(scope="module")
def script():
    return SETUP_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")


class TestSetupScript:
    """Pinned by block, not by line: each condition with the branch it guards,
    so moving, swapping or reverting a branch fails a test."""

    def test_it_checks_the_sample_folders_not_data_signals(self, script):
        """An archive tree keeps data/signals/ (the placeholder), so finding the
        folder says nothing about the samples."""
        assert (
            '$samplesPresent = (Test-Path "$dataDir\\real_train") -and '
            '(Test-Path "$dataDir\\real_test")\n'
            "if ($samplesPresent) {\n"
            "    $csvCount = "
        ) in script
        assert "if (Test-Path $dataDir)" not in script

    def test_without_the_samples_it_names_the_clone(self, script):
        assert (
            "} else {\n"
            '    Write-Host "[!] The sample signals are not in this folder: they come '
            "with a clone of the repository "
            "(git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git). "
            "The server starts without them and reads your own files from "
            'data\\signals\\." -ForegroundColor Yellow\n'
            "}\n"
        ) in script

    def test_the_next_steps_follow_whether_the_samples_are_present(self, script):
        """Steps 3 and 4 both depend on the samples: with them, the README
        Quick Start prompt; without them, the clone or the operator's files."""
        assert (
            "if ($samplesPresent) {\n"
            "    Write-Host \"  3. In a new chat, ask: 'List the available vibration "
            "signals' (the 20 included samples should appear)\" -ForegroundColor White\n"
            '    Write-Host "  4. Then try the first prompt from the README Quick Start" '
            "-ForegroundColor White\n"
            "} else {\n"
            "    Write-Host \"  3. In a new chat, ask: 'List the available vibration "
            "signals' (your own files in data\\signals\\ should appear)\" "
            "-ForegroundColor White\n"
            '    Write-Host "  4. The example prompts in the README use the bundled '
            "sample signals. To try them, clone the Git repository. Alternatively, "
            'load your own vibration signals from data/signals/." -ForegroundColor White\n'
            "}\n"
        ) in script
        assert (
            script.count("Then try the first prompt from the README Quick Start") == 1
        )

    def test_it_stays_ascii_for_windows_powershell(self, script):
        assert script.isascii()
