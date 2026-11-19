"""Tests for the integrated diagnostic report rendering.

The rendering layer places strings; it never composes them. These tests are
the enforcement of that: every evaluative sentence in the rendered document
must be findable, verbatim, in the advisory payload that produced it.
"""

import html
import importlib.abc
import importlib.machinery
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import predictive_maintenance_mcp.report_generator as report_generator

from predictive_maintenance_mcp.decision_support.advisory import (
    build_advisory,
    collect_statements,
)
from predictive_maintenance_mcp.figures import build_annotated_envelope_figure
from predictive_maintenance_mcp.integrated_report import (
    create_integrated_diagnostic_report,
)
from predictive_maintenance_mcp.report_generator import (
    HAS_PDF,
    REPORTS_DIR,
    save_integrated_diagnostic_report,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import THRESHOLD_PROVENANCE

# Sibling-module import, not `from tests.…`: pytest puts the tests directory
# on sys.path, but only `python -m pytest` also puts the repo root there. CI
# invokes the console script, where the packaged form would not resolve.
from test_advisory import (  # noqa: E402
    _anomaly,
    _bearing_faults,
    _diagnosis,
    _iso_assessed,
    _iso_refused,
)


def _case_advisory():
    return build_advisory(
        _diagnosis(
            iso=_iso_assessed(zone="B", rms=1.64),
            bearing_faults=_bearing_faults(),
            anomaly=_anomaly(health="Faulty", ratio=1.0),
        )
    )


@pytest.fixture
def advisory():
    return _case_advisory()


@pytest.fixture
def figure():
    freqs = np.linspace(0.0, 500.0, 1500)
    mags = 0.001 + 0.05 * np.exp(-(((freqs - 80.5) / 0.8) ** 2))
    return build_annotated_envelope_figure(freqs, mags, _bearing_faults())


class TestRenderedContent:
    def test_document_carries_the_verdict_and_the_iso_caveat(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert html.escape(advisory["verdict"]["statement"]) in rendered
        assert html.escape(THRESHOLD_PROVENANCE) in rendered

    def test_every_authored_statement_appears_verbatim(self, advisory, figure):
        """Covers AE4. The statements include the figure caption and the
        badges of the line-test table."""
        rendered = create_integrated_diagnostic_report(advisory, figure)
        missing = [
            statement
            for statement in collect_statements(advisory, figure)
            if html.escape(statement) not in rendered
        ]
        assert missing == [], f"rendering dropped authored statements: {missing}"

    def test_disagreement_statement_is_rendered(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert advisory["disagreements"], "fixture should produce a disagreement"
        assert html.escape(advisory["disagreements"][0]["statement"]) in rendered

    def test_every_recommendation_is_rendered_with_its_motivation(
        self, advisory, figure
    ):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        for rec in advisory["recommendations"]:
            assert html.escape(rec["action"]) in rendered
            assert html.escape(rec["motivation"]) in rendered

    def test_absence_statements_render_as_sections_not_omissions(self, figure):
        sparse = build_advisory(
            _diagnosis(iso=_iso_refused(), bearing_faults=None, anomaly=None)
        )
        rendered = create_integrated_diagnostic_report(sparse, None)
        for key in ("iso", "bearing_match", "anomaly", "baseline_comparison"):
            assert html.escape(sparse[key]["statement"]) in rendered

    def test_rendering_escapes_content_rather_than_interpolating_it_raw(self, figure):
        hostile = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        hostile["signal_id"] = "<script>alert(1)</script>"
        hostile["provenance"]["signal_id"] = "<script>alert(1)</script>"
        rendered = create_integrated_diagnostic_report(hostile, figure)
        assert "<script>alert(1)</script>" not in rendered
        assert "&lt;script&gt;" in rendered

    def test_metadata_block_cannot_be_closed_early_by_signal_content(self, figure):
        """The embedded JSON is tokenised as HTML before it is parsed as JSON."""
        hostile = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        hostile["provenance"]["signal_id"] = "</script><script>alert(1)</script>"
        rendered = create_integrated_diagnostic_report(hostile, figure)
        _, _, tail = rendered.partition(
            '<script type="application/json" id="report-metadata">'
        )
        json_block, closed, _ = tail.partition("</script>")

        # The block survives to its own closing tag rather than being cut
        # short by the payload, and what it holds is still valid JSON
        # carrying the hostile string inertly as data.
        assert closed, "metadata block was never closed"
        assert "</script>" not in json_block
        parsed = json.loads(json_block.replace("<\\/", "</"))
        assert parsed["provenance"]["signal_id"] == (
            "</script><script>alert(1)</script>"
        )

    def test_figure_caption_is_rendered(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert html.escape(figure["caption"]) in rendered

    def test_missing_figure_states_the_absence(self, advisory):
        rendered = create_integrated_diagnostic_report(advisory, None)
        assert "no envelope spectrum" in rendered.lower()


class TestProvenanceAndDeterminism:
    def test_provenance_names_signal_parameters_and_version(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        prov = advisory["provenance"]
        assert html.escape(str(prov["signal_id"])) in rendered
        assert str(prov["rpm"]) in rendered
        assert html.escape(str(prov["bearing_id"])) in rendered
        assert html.escape(prov["server_version"]) in rendered

    def test_same_payload_and_timestamp_render_identically(self, advisory, figure):
        """Covers AE5."""
        first = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        second = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        assert first == second

    def test_two_renders_differ_only_in_the_generation_timestamp(
        self, advisory, figure
    ):
        """Covers AE5.

        Determinism is a claim about content, not filenames — report names
        carry a timestamp and a sequence on purpose so a re-run never
        overwrites its predecessor.
        """
        first = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        second = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2027-01-01T23:59:59Z"
        )
        strip = lambda doc: re.sub(r"20\d\d-\d\d-\d\dT[\d:]+Z", "", doc)  # noqa: E731
        assert strip(first) == strip(second)


class TestSaving:
    def test_report_is_written_under_the_reports_directory(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure)
        written = Path(result["file_path"])
        assert written.exists()
        assert written.parent.resolve() == REPORTS_DIR.resolve()
        written.unlink()

    def test_traversal_shaped_signal_id_cannot_escape_the_reports_directory(
        self, advisory, figure
    ):
        advisory["signal_id"] = "../../../etc/passwd"
        advisory["provenance"]["signal_id"] = "../../../etc/passwd"
        result = save_integrated_diagnostic_report(advisory, figure)
        written = Path(result["file_path"])
        assert written.parent.resolve() == REPORTS_DIR.resolve()
        written.unlink()

    def test_consecutive_saves_never_overwrite(self, advisory, figure):
        first = save_integrated_diagnostic_report(advisory, figure)
        second = save_integrated_diagnostic_report(advisory, figure)
        assert first["file_name"] != second["file_name"]
        Path(first["file_path"]).unlink()
        Path(second["file_path"]).unlink()

    def test_result_carries_the_authored_statements(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure)
        assert result["statements"] == collect_statements(advisory, figure)
        assert result["report_type"] == "integrated_diagnostic"
        Path(result["file_path"]).unlink()


# ---------------------------------------------------------------------------
# PDF rendering and cross-rendering parity
#
# The parity guarantee is what makes "two renderings, one authored source"
# a property rather than an intention. Both files are produced from the same
# rendered HTML string, so this test would only fail if that stopped being
# true — which is exactly when someone needs to hear about it.
# ---------------------------------------------------------------------------


def _normalise(text):
    """Collapse whitespace so line breaks introduced by PDF layout do not
    make an identical sentence look different."""
    return " ".join(text.split())


def _pdf_text(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return _normalise(" ".join(page.extract_text() or "" for page in reader.pages))


@pytest.mark.skipif(not HAS_PDF, reason="PDF extra not installed")
class TestPdfRendering:
    def test_pdf_is_written_and_non_empty(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        written = Path(result["files"][0]["file_path"])
        assert written.exists()
        assert written.read_bytes()[:5] == b"%PDF-"
        assert result["files"][0]["file_size_kb"] > 0
        written.unlink()

    def test_both_renderings_are_produced_from_one_request(self, advisory, figure):
        result = save_integrated_diagnostic_report(
            advisory, figure, formats=["html", "pdf"]
        )
        assert [f["format"] for f in result["files"]] == ["html", "pdf"]
        for entry in result["files"]:
            path = Path(entry["file_path"])
            assert path.exists()
            path.unlink()

    def test_every_authored_statement_survives_into_the_pdf(self, advisory, figure):
        """Covers AE5 across renderings."""
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        path = Path(result["files"][0]["file_path"])
        try:
            extracted = _pdf_text(path)
            missing = [
                s
                for s in collect_statements(advisory, figure)
                if _normalise(s) not in extracted
            ]
            assert missing == [], f"PDF dropped authored statements: {missing}"
        finally:
            path.unlink()

    def test_the_iso_caveat_survives_into_the_pdf(self, advisory, figure):
        """The caveat is the single sentence most likely to be lost in a
        rendering that silently truncates."""
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        path = Path(result["files"][0]["file_path"])
        try:
            assert _normalise(THRESHOLD_PROVENANCE) in _pdf_text(path)
        finally:
            path.unlink()

    def test_pdf_and_html_agree_on_every_statement(self, advisory, figure):
        result = save_integrated_diagnostic_report(
            advisory, figure, formats=["html", "pdf"]
        )
        html_path = Path(result["files"][0]["file_path"])
        pdf_path = Path(result["files"][1]["file_path"])
        try:
            rendered_html = html_path.read_text(encoding="utf-8")
            extracted_pdf = _pdf_text(pdf_path)
            for statement in collect_statements(advisory, figure):
                in_html = html.escape(statement) in rendered_html
                in_pdf = _normalise(statement) in extracted_pdf
                assert (
                    in_html == in_pdf is True
                ), f"statement present in one rendering only: {statement}"
        finally:
            html_path.unlink()
            pdf_path.unlink()


class TestFormatValidation:
    def test_unknown_format_raises(self, advisory, figure):
        with pytest.raises(ValueError, match="Unknown report format"):
            save_integrated_diagnostic_report(advisory, figure, formats=["docx"])

    @pytest.mark.skipif(HAS_PDF, reason="PDF extra installed")
    def test_missing_pdf_dependency_raises_with_an_install_hint(self, advisory, figure):
        with pytest.raises(ValueError, match=r"\[pdf\]"):
            save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])

    def test_missing_pdf_dependency_message_names_the_extra(self, advisory, figure):
        """Exercised regardless of whether the extra is installed locally."""
        import predictive_maintenance_mcp.report_generator as rg

        original = rg.HAS_PDF
        rg.HAS_PDF = False
        try:
            with pytest.raises(ValueError) as excinfo:
                save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
            assert "[pdf]" in str(excinfo.value)
        finally:
            rg.HAS_PDF = original


# ---------------------------------------------------------------------------
# PDF availability when WeasyPrint cannot be imported
#
# WeasyPrint can be unavailable in two ways with different remedies: the
# Python package is absent (ImportError), or it is installed but cffi cannot
# load its native Pango libraries, which cffi reports as a plain OSError
# raised from its own code. Neither may stop the server from importing. A PDF
# request must then be refused with the message for that cause, without
# writing any file, and the HTML rendering must not change. Every other
# failure during the import must propagate unchanged, including an OSError
# subclass such as PermissionError and an OSError raised outside cffi.
#
# The native-library case is reproduced with cffi itself: the stand-in module
# asks cffi to load a library that does not exist, so the OSError, its message
# and its traceback are the ones cffi produces. The end-to-end checks run in a
# child interpreter because the import happens once per process, at module
# load: only a fresh interpreter shows what importing the server does when
# WeasyPrint fails to import.
# ---------------------------------------------------------------------------

PDF_MISSING_PACKAGE_MESSAGE = (
    "PDF rendering requires the optional dependency: "
    "pip install predictive-maintenance-mcp[pdf]. The HTML rendering "
    "needs nothing extra."
)

PDF_MISSING_NATIVE_MESSAGE = (
    "PDF rendering is unavailable: WeasyPrint is installed but cannot load "
    "its native libraries (Pango and the libraries it depends on). Install "
    "them following WeasyPrint's installation instructions for this "
    "platform, then restart the server. The HTML rendering needs nothing "
    "extra."
)

#: Libraries that do not exist, in both shapes of WeasyPrint's first-choice
#: names: with a dot ('libgobject-2.0-0') cffi tries the name itself and then
#: ctypes.util.find_library; without one ('libharfbuzz-0') only the latter.
MISSING_LIBRARIES = ("libpmm-d1-absent-2.0-0", "libpmm-d1-absent-0")

#: The OSError text cffi produced for WeasyPrint on python:3.11-slim without
#: Pango. Raised by anything other than cffi, it is not a native-load failure.
LOOKALIKE_ERROR = (
    "cannot load library 'libgobject-2.0-0': libgobject-2.0-0: cannot open "
    "shared object file: No such file or directory.  Additionally, "
    "ctypes.util.find_library() did not manage to locate a library called "
    "'libgobject-2.0-0'"
)

#: WeasyPrint prints a notice starting with this line before it raises.
WEASYPRINT_NOTICE = "WeasyPrint could not import some external libraries."

UNRELATED_ERROR = "simulated defect inside weasyprint"

#: Written by the child to stdout once the server has been imported.
STDOUT_SENTINEL = "stdout-after-server-import\n"

requires_cffi = pytest.mark.skipif(
    importlib.util.find_spec("cffi") is None, reason="cffi is not installed"
)


class _StandInWeasyPrint(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Make ``import weasyprint`` execute ``action`` instead of the package."""

    def __init__(self, action):
        self.action = action

    def find_spec(self, name, path=None, target=None):
        if name != "weasyprint":
            return None
        return importlib.machinery.ModuleSpec(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.action(module)


def _cffi_cannot_load(library):
    """Print WeasyPrint's notice, then have cffi fail to load ``library``."""

    def action(module):
        import cffi

        print(f"\n-----\n\n{WEASYPRINT_NOTICE}\n\n-----\n")
        cffi.FFI().dlopen(library)

    return action


def _raises(error):
    def action(module):
        raise error

    return action


def _raises_from_a_cffi_directory(tmp_path, error):
    """Raise ``error`` from a module file in a directory named ``cffi``."""
    package = tmp_path / "cffi"
    package.mkdir()
    source = package / "pmm_raise.py"
    source.write_text("def raise_(error):\n    raise error\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pmm_cffi_raise", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def action(module):
        helper.raise_(error)

    return action


class _StandInHTML:
    pass


def _provides_html(module):
    module.HTML = _StandInHTML


#: Child program: break `import weasyprint` the way MODE says, import the
#: server, write a sentinel to stdout, then exercise the report renderings.
#: Results go to a file, so stdout carries only the import's output and the
#: sentinel.
_PDF_PROBE = """
import importlib.abc, importlib.machinery, json, sys
from pathlib import Path

mode, src, workdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
cfg = json.loads((workdir / "probe.json").read_text(encoding="utf-8"))


class _StandInWeasyPrint(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name != "weasyprint":
            return None
        return importlib.machinery.ModuleSpec(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if mode == "native":
            import cffi

            print(cfg["notice"])
            cffi.FFI().dlopen(cfg["missing_library"])
        elif mode == "permission":
            raise PermissionError(13, "Permission denied", cfg["permission_path"])
        elif mode == "lookalike":
            raise OSError(cfg["lookalike_error"])
        raise RuntimeError(cfg["unrelated_error"])


if mode == "package":
    sys.modules["weasyprint"] = None
else:
    sys.meta_path.insert(0, _StandInWeasyPrint())

from predictive_maintenance_mcp.server import mcp  # noqa: F401
import predictive_maintenance_mcp.report_generator as rg

assert rg.__file__.startswith(src), "probe resolved to " + rg.__file__ + ", not " + src
sys.stdout.write(cfg["stdout_sentinel"])
sys.stdout.flush()
rg.REPORTS_DIR = workdir / "reports"
rg.REPORTS_DIR.mkdir()
advisory = json.loads((workdir / "advisory.json").read_text(encoding="utf-8"))

result = {"has_pdf": rg.HAS_PDF, "refusals": {}}
saved = rg.save_integrated_diagnostic_report(advisory, None, formats=["html"])
result["html_formats"] = [entry["format"] for entry in saved["files"]]
result["html_text"] = Path(saved["file_path"]).read_text(encoding="utf-8")
before = {p.name for p in rg.REPORTS_DIR.iterdir()}
for request in (["pdf"], ["html", "pdf"]):
    try:
        rg.save_integrated_diagnostic_report(advisory, None, formats=request)
    except ValueError as exc:
        refusal = {"message": str(exc), "cause": type(exc.__cause__).__name__}
    else:
        refusal = None
    result["refusals"]["+".join(request)] = refusal
result["written_by_refused_requests"] = sorted(
    {p.name for p in rg.REPORTS_DIR.iterdir()} - before
)
(workdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
"""


def _strip_timestamps(document):
    return re.sub(r"20\d\d-\d\d-\d\dT[\d:]+Z", "", document)


@pytest.fixture(scope="module")
def pdf_probe(tmp_path_factory):
    """Run the child program once per mode; return (process, result or None)."""
    from conftest import SRC_DIR, SUBPROCESS_PIN

    runs = {}

    def run(mode):
        if mode not in runs:
            workdir = tmp_path_factory.mktemp(f"pdf-probe-{mode}")
            (workdir / "advisory.json").write_text(
                json.dumps(_case_advisory()), encoding="utf-8"
            )
            (workdir / "probe.json").write_text(
                json.dumps(
                    {
                        "notice": f"\n-----\n\n{WEASYPRINT_NOTICE}\n\n-----\n",
                        "missing_library": MISSING_LIBRARIES[0],
                        "permission_path": "fonts.conf",
                        "lookalike_error": LOOKALIKE_ERROR,
                        "unrelated_error": UNRELATED_ERROR,
                        "stdout_sentinel": STDOUT_SENTINEL,
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    SUBPROCESS_PIN + _PDF_PROBE,
                    mode,
                    str(SRC_DIR),
                    str(workdir),
                ],
                capture_output=True,
                text=True,
                cwd=SRC_DIR.parent,
                timeout=300,
            )
            result_file = workdir / "result.json"
            result = (
                json.loads(result_file.read_text(encoding="utf-8"))
                if result_file.exists()
                else None
            )
            runs[mode] = (proc, result)
        return runs[mode]

    return run


class TestPdfUnavailable:
    @requires_cffi
    def test_server_imports_when_weasyprint_cannot_load_native_libraries(
        self, pdf_probe
    ):
        proc, result = pdf_probe("native")
        assert proc.returncode == 0, f"server import failed:\n{proc.stderr}"
        assert result is not None and result["has_pdf"] is False

    @requires_cffi
    def test_weasyprint_notice_is_kept_off_stdout(self, pdf_probe):
        """stdout carries the MCP protocol when the server runs over stdio.

        The sentinel, written after the import, shows that stdout is back in
        place and usable once the import is over.
        """
        proc, _ = pdf_probe("native")
        assert proc.stdout == STDOUT_SENTINEL
        assert WEASYPRINT_NOTICE in proc.stderr

    @requires_cffi
    def test_pdf_request_names_the_missing_native_libraries(self, pdf_probe):
        proc, result = pdf_probe("native")
        assert result is not None, f"server import failed:\n{proc.stderr}"
        assert result["refusals"]["pdf"] == {
            "message": PDF_MISSING_NATIVE_MESSAGE,
            "cause": "OSError",
        }

    def test_pdf_request_names_the_missing_python_package(self, pdf_probe):
        proc, result = pdf_probe("package")
        assert result is not None, f"server import failed:\n{proc.stderr}"
        assert proc.stdout == STDOUT_SENTINEL
        assert result["has_pdf"] is False
        assert result["refusals"]["pdf"] == {
            "message": PDF_MISSING_PACKAGE_MESSAGE,
            "cause": "ModuleNotFoundError",
        }

    @pytest.mark.parametrize(
        ("mode", "message"),
        [
            pytest.param("native", PDF_MISSING_NATIVE_MESSAGE, marks=requires_cffi),
            ("package", PDF_MISSING_PACKAGE_MESSAGE),
        ],
    )
    def test_request_with_pdf_is_refused_whole_and_writes_nothing(
        self, pdf_probe, mode, message
    ):
        """No HTML is written in place of the PDF, and nothing reports success."""
        proc, result = pdf_probe(mode)
        assert result is not None, f"server import failed:\n{proc.stderr}"
        assert result["refusals"]["html+pdf"]["message"] == message
        assert result["written_by_refused_requests"] == []

    @pytest.mark.parametrize(
        "mode", [pytest.param("native", marks=requires_cffi), "package"]
    )
    def test_html_rendering_is_unchanged(self, pdf_probe, mode, advisory):
        proc, result = pdf_probe(mode)
        assert result is not None, f"server import failed:\n{proc.stderr}"
        assert result["html_formats"] == ["html"]
        expected = create_integrated_diagnostic_report(
            advisory, None, generated_at="2026-01-01T00:00:00Z"
        )
        assert _strip_timestamps(result["html_text"]) == _strip_timestamps(expected)

    @pytest.mark.parametrize(
        ("mode", "last_line"),
        [
            (
                "permission",
                "PermissionError: [Errno 13] Permission denied: 'fonts.conf'",
            ),
            ("lookalike", f"OSError: {LOOKALIKE_ERROR}"),
            ("unrelated", f"RuntimeError: {UNRELATED_ERROR}"),
        ],
    )
    def test_other_failures_during_the_weasyprint_import_propagate(
        self, pdf_probe, mode, last_line
    ):
        """Not reported as missing native libraries, and not swallowed."""
        proc, result = pdf_probe(mode)
        assert proc.returncode != 0
        assert result is None
        assert proc.stderr.rstrip().splitlines()[-1] == last_line
        assert "report_generator.py" in proc.stderr


class TestWeasyPrintImportClassification:
    """The same classification in-process, on the module's loader function."""

    @staticmethod
    def _stand_in(monkeypatch, action):
        monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
        monkeypatch.setattr(
            sys, "meta_path", [_StandInWeasyPrint(action)] + sys.meta_path
        )

    @requires_cffi
    @pytest.mark.parametrize("library", MISSING_LIBRARIES)
    def test_cffi_failing_to_load_a_library_means_missing_native_libraries(
        self, monkeypatch, library
    ):
        self._stand_in(monkeypatch, _cffi_cannot_load(library))
        html_class, message, error = report_generator._load_weasyprint()
        assert (html_class, message) == (None, PDF_MISSING_NATIVE_MESSAGE)
        assert type(error) is OSError

    def test_plain_oserror_raised_from_cffi_means_missing_native_libraries(
        self, monkeypatch, tmp_path
    ):
        error = OSError("cannot load library 'libpmm-d1-absent-2.0-0'")
        self._stand_in(monkeypatch, _raises_from_a_cffi_directory(tmp_path, error))
        assert report_generator._load_weasyprint() == (
            None,
            PDF_MISSING_NATIVE_MESSAGE,
            error,
        )

    def test_missing_python_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "weasyprint", None)
        html_class, message, error = report_generator._load_weasyprint()
        assert (html_class, message) == (None, PDF_MISSING_PACKAGE_MESSAGE)
        assert type(error) is ModuleNotFoundError

    def test_successful_import_returns_the_html_class(self, monkeypatch):
        self._stand_in(monkeypatch, _provides_html)
        assert report_generator._load_weasyprint() == (_StandInHTML, None, None)

    @pytest.mark.parametrize(
        "error",
        [
            PermissionError(13, "Permission denied", "fonts.conf"),
            FileNotFoundError(2, "No such file or directory", "fonts.conf"),
            OSError(LOOKALIKE_ERROR),
            RuntimeError(UNRELATED_ERROR),
        ],
        ids=["permission-error", "file-not-found", "lookalike-oserror", "runtime"],
    )
    def test_other_failures_propagate_unchanged(self, monkeypatch, error):
        self._stand_in(monkeypatch, _raises(error))
        with pytest.raises(type(error)) as excinfo:
            report_generator._load_weasyprint()
        assert excinfo.value is error

    def test_oserror_subclass_raised_from_cffi_propagates(self, monkeypatch, tmp_path):
        error = PermissionError(13, "Permission denied", "libpmm-d1-absent-2.0-0")
        self._stand_in(monkeypatch, _raises_from_a_cffi_directory(tmp_path, error))
        with pytest.raises(PermissionError) as excinfo:
            report_generator._load_weasyprint()
        assert excinfo.value is error

    @requires_cffi
    def test_weasyprint_notice_goes_to_stderr(self, monkeypatch, capsys):
        self._stand_in(monkeypatch, _cffi_cannot_load(MISSING_LIBRARIES[0]))
        report_generator._load_weasyprint()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert WEASYPRINT_NOTICE in captured.err

    @pytest.mark.parametrize("outcome", ["loads", "missing-native", "propagates"])
    def test_stdout_is_restored_after_the_import(self, monkeypatch, tmp_path, outcome):
        """The redirect lasts only for the import, whatever its outcome."""
        actions = {
            "loads": _provides_html,
            "missing-native": _raises_from_a_cffi_directory(
                tmp_path, OSError("cannot load library 'libpmm-d1-absent-2.0-0'")
            ),
            "propagates": _raises(PermissionError(13, "Permission denied", "x")),
        }
        self._stand_in(monkeypatch, actions[outcome])
        stdout_before = sys.stdout
        if outcome == "propagates":
            with pytest.raises(PermissionError):
                report_generator._load_weasyprint()
        else:
            report_generator._load_weasyprint()
        assert sys.stdout is stdout_before

    def test_pdf_refusal_carries_the_recorded_cause(
        self, monkeypatch, advisory, figure
    ):
        error = OSError(LOOKALIKE_ERROR)
        monkeypatch.setattr(report_generator, "HAS_PDF", False)
        monkeypatch.setattr(
            report_generator, "_PDF_UNAVAILABLE_MESSAGE", PDF_MISSING_NATIVE_MESSAGE
        )
        monkeypatch.setattr(report_generator, "_PDF_IMPORT_ERROR", error)
        with pytest.raises(ValueError) as excinfo:
            save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        assert str(excinfo.value) == PDF_MISSING_NATIVE_MESSAGE
        assert excinfo.value.__cause__ is error
