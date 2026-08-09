"""U7 CI guard: the legacy monolith and root shims must never reappear.

Replaces ``test_obsolete_tools_removed.py``: instead of asserting which
functions the monolith no longer exposes, it asserts the monolith itself
(and every root-level compatibility shim) is gone — from the filesystem,
from imports, and from operational documentation.

The needle strings are built by concatenation so this guard never matches
itself in a repo-wide grep.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

#: The removed monolith's module name (concatenated to avoid self-match).
MONOLITH = "machinery_diagnostics" + "_server"

#: Root-level files removed in U7 that must never come back.
#: (The ISO filename is concatenated for the same self-match reason.)
REMOVED_SRC_FILES = [
    MONOLITH + ".py",
    "bearing_analyzer.py",
    "bearing_catalog.py",
    "diagnosis_pipeline.py",
    "iso" + "10816.py",
    "signal_loader.py",
    "signal_repository.py",
    "spectral.py",
]

#: Operational surfaces that must not reference the monolith. Historical
#: records are deliberately excluded (see _is_historical): the audit, plan
#: documents, solution learnings, and the CHANGELOG legitimately describe
#: the monolith's past existence and removal.
SCAN_DIRS = [SRC, ROOT / "tests", ROOT / "docs"]
SCAN_FILES = [
    ROOT / "Dockerfile",
    ROOT / "pyproject.toml",
    ROOT / "validate_server.py",
    ROOT / ".vscode" / "mcp.json",
    ROOT / "README.md",
    ROOT / "INSTALL.md",
    ROOT / "CONTRIBUTING.md",
]
SCAN_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ""}

#: Precise historical-record exceptions (path parts relative to ROOT).
HISTORICAL = (
    ("docs", "AUDIT-2026-07-10.md"),
    ("docs", "plans"),
    ("docs", "solutions"),
)


def _is_historical(path: Path) -> bool:
    rel = path.relative_to(ROOT).parts
    for exc in HISTORICAL:
        if rel[: len(exc)] == exc:
            return True
    return False


def _iter_scan_targets():
    for d in SCAN_DIRS:
        for p in d.rglob("*"):
            if (
                p.is_file()
                and p.suffix in SCAN_SUFFIXES
                and "__pycache__" not in p.parts
                and not _is_historical(p)
            ):
                yield p
    for p in SCAN_FILES:
        if p.exists():
            yield p


class TestMonolithRemoved:
    def test_monolith_file_does_not_exist(self):
        assert not (
            SRC / (MONOLITH + ".py")
        ).exists(), f"src/{MONOLITH}.py must stay removed (deleted in v0.9.0)"

    def test_root_shims_do_not_exist(self):
        present = [f for f in REMOVED_SRC_FILES if (SRC / f).exists()]
        assert present == [], (
            f"Removed root-level modules reappeared in src/: {present} — "
            f"the canonical homes are the ISO 13374 sub-packages "
            f"(signal_acquisition/, signal_processing/, diagnostics/, "
            f"decision_support/, prognostics/)."
        )

    def test_signal_acquisition_is_the_real_home(self):
        """The moved implementations live in signal_acquisition/ (not shims)."""
        loaders = (SRC / "signal_acquisition" / "loaders.py").read_text(
            encoding="utf-8"
        )
        repository = (SRC / "signal_acquisition" / "repository.py").read_text(
            encoding="utf-8"
        )
        assert "def load_signal_data(" in loaders
        assert "class SignalRepository" in repository
        for text, name in ((loaders, "loaders.py"), (repository, "repository.py")):
            assert "import *" not in text, (
                f"signal_acquisition/{name} looks like a shim again — "
                f"it must contain the real implementation"
            )

    def test_monolith_not_importable(self):
        import importlib.util

        spec = importlib.util.find_spec(f"predictive_maintenance_mcp.{MONOLITH}")
        assert spec is None, f"predictive_maintenance_mcp.{MONOLITH} is importable"

    def test_no_monolith_references_in_operational_surfaces(self):
        offenders = [
            str(p.relative_to(ROOT))
            for p in _iter_scan_targets()
            if MONOLITH in p.read_text(encoding="utf-8", errors="ignore")
        ]
        assert offenders == [], (
            f"References to the removed monolith found in: {offenders} — "
            f"point them at predictive_maintenance_mcp.server / mcp_tools/ "
            f"instead."
        )


class TestIsoCitationPrecision:
    """'ISO 10816' may appear ONLY as precise threshold provenance.

    Allowed forms: 'ISO 10816-3:2009' (prose/docstrings) and the identifier
    spelling 'iso_10816_3_2009' (test names). Anything else — bare
    'ISO 10816', 'iso10816' module names, other editions — is a citation
    regression.
    """

    _ANY = re.compile(r"iso[\s_]?10816", re.IGNORECASE)
    _ALLOWED = re.compile(r"(?i)iso[\s_]10816[-_]3[:_]2009")

    def test_only_precise_provenance_citations(self):
        offenders: list[str] = []
        for p in _iter_scan_targets():
            if p == Path(__file__):
                # This guard must spell the forbidden pattern to describe it.
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            stripped = self._ALLOWED.sub("", text)
            for m in self._ANY.finditer(stripped):
                snippet = stripped[max(0, m.start() - 30) : m.end() + 30]
                offenders.append(f"{p.relative_to(ROOT)}: ...{snippet!r}...")
        assert offenders == [], (
            "Imprecise 'ISO 10816' citations found (only the provenance "
            "form 'ISO 10816-3:2009' is allowed):\n" + "\n".join(offenders)
        )
