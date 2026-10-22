"""Tests for MCP report generation tools (ISO 13374 Block 6).

Since U8 every report tool takes signal_id as its only signal handle:
signals are loaded once via the repository and referenced by id.
"""

import inspect
import json
import logging
import os
import pickle
from html import escape as html_escape

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock

from mcp.server.mcpserver import MCPServer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from _bearing_scenarios import (
    DEFECT_SEED,
    FAULT_FREE_SEED,
    FREQUENCIES_6205,
    RPM_6205,
    defect,
    white,
)
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import check_bearing_faults
from predictive_maintenance_mcp.mcp_tools.report_tools import register
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.mcp_tools import report_tools

SHAFT_6205_HZ = RPM_6205 / 60.0

#: The 6205 at 1797 rpm as the tool's docstring example declares it.
DECLARED_6205 = {"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.58, "FTF": 11.93}

#: The claim statement of an outer-race indication checked without a speed.
OUTER_RACE_WITHOUT_SPEED = (
    "The envelope evidence of this single spectrum is consistent with an "
    "outer-race defect, but no shaft speed was given, so the shaft-order and "
    "shaft-rate sideband tests did not run, and a shaft harmonic or a sideband "
    "of another fault type cannot be excluded. No single fault type is named "
    "from this spectrum."
)

#: White-noise records of the envelope-report label test.
ENVELOPE_NOISE_SEEDS = 20

#: Derived from the module so it cannot drift from the logger the
#: tools actually write to.
REPORT_LOGGER = report_tools.logger.name


def _logged(caplog, needle: str) -> bool:
    """Did *this* module's logger emit a record containing ``needle``?

    Both halves matter. ``caplog``'s handler collects records from every
    logger in the process — weasyprint and plotly both emit here — so a bare
    ``assert caplog.records`` passes on unrelated noise. And matching the
    message is what makes the assertion fail when the narration under test is
    removed, rather than when the process happens to be quiet.
    """
    return any(
        record.name == REPORT_LOGGER and needle in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp():
    server = MCPServer("test-reports")
    register(server)
    return server


@pytest.fixture
def tools(mcp):
    return {t.name: t.fn for t in mcp._tool_manager._tools.values()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 50 * t)
    pd.DataFrame(sig).to_csv(signals_dir / "report_test.csv", index=False, header=False)
    with open(signals_dir / "report_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir
    )
    return signals_dir


@pytest.fixture
def repo(data_dir):
    """Repository with the report test signal loaded; cleaned afterwards."""
    repo = get_repository()
    repo.clear_all()
    repo.load_signal("report_test.csv")  # metadata: fs=10000, unit 'g'
    yield repo
    repo.clear_all()


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    rd = tmp_path / "reports"
    rd.mkdir()
    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR", rd
    )
    monkeypatch.setattr("predictive_maintenance_mcp.report_generator.REPORTS_DIR", rd)
    # Also patch the REPORTS_DIR_PATH alias used in diagnostics import
    try:
        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR_PATH", rd
        )
    except AttributeError:
        pass
    return rd


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


def _load_extra_signal(
    repo,
    signals_dir,
    name,
    fs=10000,
    duration=1.0,
    freq=50.0,
    noise=0.0,
    with_meta=True,
):
    """Helper: create a CSV signal file and load it into the repository."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
    pd.DataFrame(sig).to_csv(signals_dir / name, index=False, header=False)
    if with_meta:
        meta_path = signals_dir / name.replace(".csv", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)
    return repo.load_signal(name)["signal_id"]


def _write_and_load(repo, signals_dir, name, samples, fs, measurement=None, **extra):
    """Write samples and a companion (rate, unit 'g', ``extra``, and the
    optional measurement identity) and load them; returns the signal_id."""
    pd.DataFrame(samples).to_csv(signals_dir / f"{name}.csv", index=False, header=False)
    companion = {"sampling_rate": fs, "signal_unit": "g", **extra}
    if measurement is not None:
        companion["measurement"] = measurement
    with open(signals_dir / f"{name}_metadata.json", "w") as f:
        json.dump(companion, f)
    return repo.load_signal(f"{name}.csv")["signal_id"]


def _outer_race_10s() -> np.ndarray:
    """An outer-race defect of a 6205 at 1797 rpm, 10 s at 12 kHz, -15 dB."""
    return defect(
        "outer",
        np.random.default_rng((DEFECT_SEED, 40, 0)),
        12000.0,
        10.0,
        FREQUENCIES_6205,
        SHAFT_6205_HZ,
        -15.0,
    )


def _outer_race_beside_a_weak_inner_race() -> np.ndarray:
    """10 s at 12 kHz: a 3 kHz carrier modulated by an impulse train at the
    declared BPFO, a weak component at the declared BPFI and white noise.

    With the 6205 speed (1797 rpm) the outer-race line is supported, the
    inner-race line stands above the floor without a counted line of its own,
    and BPFI minus twice the shaft rate (102.3 Hz) lies in the outer-race
    window (107.36 Hz +/- 5.37 Hz), so the check qualifies the outer race
    (inner_race_sideband_not_excluded).
    """
    t = np.arange(0, 10.0, 1 / 12000.0)
    carrier = np.sin(2 * np.pi * 3000.0 * t)
    outer = np.exp(-25.0 * ((t * DECLARED_6205["BPFO"]) % 1.0))
    inner = np.sin(2 * np.pi * DECLARED_6205["BPFI"] * t)
    noise = np.random.default_rng(0).standard_normal(t.size)
    return carrier * (0.3 + outer) + 0.09 * carrier * inner + 0.8 * noise


@pytest.fixture
def line_checks(monkeypatch):
    """Every line test the envelope report hands to save_envelope_report."""
    captured = []
    save = report_tools.save_envelope_report

    def spy(**kwargs):
        captured.append(kwargs["line_check"])
        return save(**kwargs)

    monkeypatch.setattr(report_tools, "save_envelope_report", spy)
    return captured


# ---------------------------------------------------------------------------
# Plot signal
# ---------------------------------------------------------------------------


class TestPlotSignal:
    """Tests for plot_signal tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_creates_html(self, tools, repo, reports_dir, mock_ctx):
        result = await tools["plot_signal"](signal_id="report_test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_plot_signal_with_time_range(self, tools, repo, reports_dir):
        """plot_signal with time_range should produce a zoomed plot."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            time_range=[0.1, 0.3],
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_no_statistics(self, tools, repo, reports_dir):
        """plot_signal with show_statistics=False omits reference lines."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            show_statistics=False,
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_custom_title(self, tools, repo, reports_dir):
        """plot_signal with custom title."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            title="Custom Title Test",
        )
        assert "Interactive plot saved" in result

    @pytest.mark.asyncio
    async def test_plot_signal_not_loaded(self, tools, repo, reports_dir):
        """Unknown signal_id → standard error naming load_signal."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["plot_signal"](signal_id="nonexistent")

    @pytest.mark.asyncio
    async def test_plot_signal_with_ctx(
        self, tools, repo, reports_dir, mock_ctx, package_caplog
    ):
        """plot_signal with ctx should log progress to the module logger, not to ctx."""
        result = await tools["plot_signal"](
            signal_id="report_test",
            ctx=mock_ctx,
        )
        assert _logged(package_caplog, "Generating time-domain plot")
        mock_ctx.info.assert_not_called()  # SEP-2577: not the client's channel
        assert "Interactive plot saved" in result


# ---------------------------------------------------------------------------
# Merged plot tools are gone (absorbed by the generate_* reports)
# ---------------------------------------------------------------------------


class TestPlotToolsMerged:
    """U9b clean cut: plot_spectrum/plot_envelope/plot_iso_20816_chart and
    get_report_info are no longer registered — their capability lives in
    generate_fft_report / generate_envelope_report / generate_iso_report /
    list_html_reports(file_name=...)."""

    def test_absorbed_tools_not_registered(self, tools):
        for old in (
            "plot_spectrum",
            "plot_envelope",
            "plot_iso_20816_chart",
            "get_report_info",
        ):
            assert old not in tools


# ---------------------------------------------------------------------------
# List reports
# ---------------------------------------------------------------------------


class TestListReports:
    """Tests for list_html_reports tool."""

    def test_empty_reports(self, tools, reports_dir):
        result = tools["list_html_reports"]()
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 0

    def test_finds_existing_report(self, tools, reports_dir):
        """Verify list_html_reports returns a list (report_generator uses its own REPORTS_DIR)."""
        result = tools["list_html_reports"]()
        # Just verify it returns a list without error
        assert isinstance(result, list)

    def test_lists_html_files_with_metadata(self, tools, reports_dir):
        """list_html_reports should find HTML reports that contain metadata."""
        # Create a fake HTML report with embedded metadata
        metadata = {"report_type": "fft_spectrum", "signal_file": "test.csv"}
        html_content = (
            "<html><head>"
            '<script type="application/json" id="report-metadata">'
            f"{json.dumps(metadata)}"
            "</script></head><body>report</body></html>"
        )
        (reports_dir / "fft_test_report.html").write_text(html_content)
        result = tools["list_html_reports"]()
        assert len(result) == 1
        assert result[0]["report_type"] == "fft_spectrum"
        assert result[0]["signal_file"] == "test.csv"

    def test_lists_ignores_html_without_metadata(self, tools, reports_dir):
        """HTML files without metadata block should be skipped."""
        (reports_dir / "plain.html").write_text("<html><body>no metadata</body></html>")
        result = tools["list_html_reports"]()
        # File has no metadata so list_reports skips it (error branch)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Single-report metadata route (merged get_report_info)
# ---------------------------------------------------------------------------


class TestListReportsFileNameRoute:
    """list_html_reports(file_name=...) — the absorbed get_report_info."""

    def test_report_not_found_raises(self, tools, reports_dir):
        """U6 error contract: a missing report is misuse — raise with an
        actionable message, never an error-shaped dict as success."""
        with pytest.raises(ValueError, match="list_html_reports"):
            tools["list_html_reports"](file_name="nonexistent.html")

    def test_report_found_with_metadata(self, tools, reports_dir):
        metadata = {
            "report_type": "envelope",
            "signal_file": "sig.csv",
            "num_peaks": 10,
        }
        html = (
            "<html><head>"
            '<script type="application/json" id="report-metadata">'
            f"{json.dumps(metadata)}"
            "</script></head><body></body></html>"
        )
        (reports_dir / "env_report.html").write_text(html)
        result = tools["list_html_reports"](file_name="env_report.html")
        assert "error" not in result
        assert result["metadata"]["report_type"] == "envelope"
        assert result["metadata"]["num_peaks"] == 10

    def test_report_without_metadata_block_raises(self, tools, reports_dir):
        """Report file exists but has no metadata JSON block → raise."""
        (reports_dir / "no_meta.html").write_text("<html><body>plain</body></html>")
        with pytest.raises(ValueError, match="Metadata not found"):
            tools["list_html_reports"](file_name="no_meta.html")

    def test_traversal_attempt_rejected(self, tools, reports_dir, tmp_path):
        """U1 security regression: the merged read path still contains the
        user-supplied name via safe_resolve — traversal is rejected and no
        file outside reports/ is read (no existence oracle)."""
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")
        # Forward-slash traversal and absolute paths escape reports/ on every
        # platform. A backslash is a path separator only on Windows; on POSIX
        # "..\\secret.txt" is a single (in-bounds, nonexistent) filename, so it
        # is only a traversal payload where the OS treats "\\" as a separator.
        hostile = ["../secret.txt", "../../etc/passwd", str(secret)]
        if os.sep == "\\" or os.altsep == "\\":
            hostile.append("..\\secret.txt")
        for name in hostile:
            with pytest.raises(ValueError, match="Invalid report filename"):
                tools["list_html_reports"](file_name=name)

    def test_sibling_directory_bypass_rejected(self, tools, reports_dir):
        """The d689886 bypass class: a name resolving into a SIBLING dir
        of reports/ (prefix match) must be rejected too."""
        evil = reports_dir.parent / (reports_dir.name + "_evil")
        evil.mkdir()
        (evil / "x.html").write_text("<html></html>")
        with pytest.raises(ValueError, match="Invalid report filename"):
            tools["list_html_reports"](file_name=f"../{reports_dir.name}_evil/x.html")


# ---------------------------------------------------------------------------
# Generate FFT report
# ---------------------------------------------------------------------------


class TestGenerateFFTReport:
    """Tests for generate_fft_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_fft_report(self, tools, repo, reports_dir):
        result = await tools["generate_fft_report"](signal_id="report_test")
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_fft_report_uses_stored_sampling_rate(self, tools, repo, reports_dir):
        """The rate comes from the stored signal (declared at load time)."""
        result = await tools["generate_fft_report"](signal_id="report_test")
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_fft_report_with_rpm(self, tools, repo, reports_dir):
        """rpm (user-facing) is converted to Hz for harmonic labeling."""
        result = await tools["generate_fft_report"](
            signal_id="report_test",
            rpm=3000.0,  # 50 Hz shaft — the test signal's tone
        )
        assert "file_path" in result
        assert result["metadata"]["rotation_freq"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_two_consecutive_runs_two_distinct_files(
        self, tools, repo, reports_dir
    ):
        """U9 loop closure: timestamped filenames — a re-run never
        overwrites, and list_html_reports lists both."""
        r1 = await tools["generate_fft_report"](signal_id="report_test")
        r2 = await tools["generate_fft_report"](signal_id="report_test")
        assert r1["file_name"] != r2["file_name"]
        assert Path(r1["file_path"]).exists()
        assert Path(r2["file_path"]).exists()
        listed = {r["file_name"] for r in tools["list_html_reports"]()}
        assert {r1["file_name"], r2["file_name"]} <= listed

    @pytest.mark.asyncio
    async def test_fft_report_signal_without_rate(
        self, tools, data_dir, repo, reports_dir
    ):
        """A stored signal without a sampling rate → structured error."""
        sig = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000, endpoint=False))
        pd.DataFrame(sig).to_csv(data_dir / "no_meta.csv", index=False, header=False)
        repo.load_signal("no_meta.csv")  # no metadata → no rate
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_fft_report"](signal_id="no_meta")

    @pytest.mark.asyncio
    async def test_fft_report_signal_not_loaded(self, tools, repo, reports_dir):
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_fft_report"](signal_id="does_not_exist")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpm", [0.0, float("nan")], ids=["zero", "nan"])
    async def test_fft_report_invalid_rpm_is_refused_before_any_report(
        self, tools, repo, reports_dir, rpm
    ):
        """Issue #83: a 0 or NaN rpm used to drop the harmonic labels without
        a word; it is refused, and no report file is written."""
        with pytest.raises(ValueError) as exc_info:
            await tools["generate_fft_report"](signal_id="report_test", rpm=rpm)

        msg = str(exc_info.value)
        assert "rpm must be positive and finite" in msg
        assert "omit rpm" in msg
        assert list(reports_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_fft_report_without_rpm_has_no_rotation_frequency(
        self, tools, repo, reports_dir
    ):
        result = await tools["generate_fft_report"](signal_id="report_test")
        assert result["metadata"]["rotation_freq"] is None

    @pytest.mark.asyncio
    async def test_fft_report_with_ctx(
        self, tools, repo, reports_dir, mock_ctx, package_caplog
    ):
        result = await tools["generate_fft_report"](
            signal_id="report_test",
            ctx=mock_ctx,
        )
        assert _logged(package_caplog, "Generating FFT report")
        mock_ctx.info.assert_not_called()  # SEP-2577: not the client's channel
        assert "file_path" in result


# ---------------------------------------------------------------------------
# Generate envelope report
# ---------------------------------------------------------------------------


class TestGenerateEnvelopeReport:
    """Tests for generate_envelope_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_envelope_report(self, tools, repo, reports_dir):
        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_default_band_is_fs_aware(self, tools, data_dir, repo, reports_dir):
        """The default filter_high must adapt to the signal's Nyquist. On a
        sub-10 kHz signal the former fixed 5000 Hz default exceeded Nyquist
        and the report raised; the default now resolves fs-aware and the
        report succeeds without the caller specifying a band."""
        sid = _load_extra_signal(repo, data_dir, "env_8k.csv", fs=8000)
        result = await tools["generate_envelope_report"](signal_id=sid)
        assert "file_path" in result
        # The echoed band is contained within Nyquist (4000 Hz here).
        assert result["metadata"]["filter_band"][1] <= 4000.0

    @pytest.mark.asyncio
    async def test_explicit_band_above_nyquist_still_raises(
        self, tools, data_dir, repo, reports_dir
    ):
        """An explicitly requested band above Nyquist is never silently
        clamped — only the default is fs-aware."""
        sid = _load_extra_signal(repo, data_dir, "env_8k2.csv", fs=8000)
        with pytest.raises(ValueError, match="Nyquist"):
            await tools["generate_envelope_report"](signal_id=sid, filter_high=5000.0)

    @pytest.mark.asyncio
    async def test_envelope_report_bearing_freqs_from_companion_metadata(
        self, tools, data_dir, repo, reports_dir
    ):
        """Optional bearing freqs are auto-read from the source file's
        companion metadata when not passed explicitly."""
        meta_path = data_dir / "report_test_metadata.json"
        meta = json.loads(meta_path.read_text())
        # 6205 (CWRU geometry) at 1797 RPM
        meta.update({"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.58, "FTF": 11.93})
        meta_path.write_text(json.dumps(meta))

        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_with_bearing_freqs(self, tools, repo, reports_dir):
        result = await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
            bearing_freqs={"BPFO": 81.0, "BPFI": 119.0, "BSF": 64.0, "FTF": 15.0},
        )
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_envelope_report_refuses_bad_bearing_freqs_before_any_report(
        self, tools, repo, reports_dir
    ):
        """Issue #83: a NaN, zero or negative reference frequency was skipped
        without a word (no marker for that label); it is refused, naming every
        bad label, and no report file is written."""
        with pytest.raises(ValueError) as exc_info:
            await tools["generate_envelope_report"](
                signal_id="report_test",
                filter_low=100.0,
                filter_high=4000.0,
                bearing_freqs={"BPFO": 107.4, "BPFI": float("nan"), "BSF": 0.0},
            )

        msg = str(exc_info.value)
        assert msg.startswith("bearing_freqs contains non-positive or non-finite")
        assert "BPFI" in msg
        assert "BSF" in msg
        assert list(reports_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_envelope_report_signal_without_rate(
        self, tools, data_dir, repo, reports_dir
    ):
        """Stored signal without a rate → structured error."""
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(
            data_dir / "no_meta_env.csv", index=False, header=False
        )
        repo.load_signal("no_meta_env.csv")
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_envelope_report"](signal_id="no_meta_env")

    @pytest.mark.asyncio
    async def test_envelope_report_with_ctx_bearing_line_test(
        self, tools, repo, reports_dir, mock_ctx, package_caplog
    ):
        """The bearing line test is reported on the module logger, not to ctx."""
        await tools["generate_envelope_report"](
            signal_id="report_test",
            filter_low=100.0,
            filter_high=4000.0,
            bearing_freqs={"BPFO": 50.0, "BPFI": 100.0},
            ctx=mock_ctx,
        )
        # The line-test record specifically, not "some record exists":
        # three unconditional logger.info calls run before this branch is
        # even reached, so a bare `assert caplog.records` stayed green with
        # the branch deleted — which is the whole subject of this test.
        assert _logged(package_caplog, "Bearing line test")
        mock_ctx.info.assert_not_called()  # SEP-2577: not the client's channel

    @pytest.mark.asyncio
    async def test_white_noise_carries_no_bearing_label(
        self, tools, data_dir, repo, reports_dir
    ):
        """T1 PR C: labels come from the central line test. White noise, 1 s
        at 12 kHz, 6205 frequencies at 1797 rpm: no seed carries a label (the
        former peak-to-frequency matcher labelled most of them)."""
        labelled = []
        for i in range(ENVELOPE_NOISE_SEEDS):
            rng = np.random.default_rng(FAULT_FREE_SEED + 1000 + i)
            sid = _write_and_load(
                repo,
                data_dir,
                f"env_noise_{i}",
                white(rng, 12000.0, 1.0, SHAFT_6205_HZ),
                12000.0,
                **FREQUENCIES_6205,
            )
            result = await tools["generate_envelope_report"](signal_id=sid)
            if result["bearing_matches"]:
                labelled.append((sid, result["bearing_matches"]))

        assert not labelled, labelled

    @pytest.mark.asyncio
    async def test_labels_without_a_declared_speed_say_the_tests_did_not_run(
        self, tools, data_dir, repo, reports_dir, line_checks
    ):
        """No rpm parameter, no measurement rpm and no point: the check runs
        without the sideband and shaft-order tests, every label and the report
        say so, and the outer-race indication is qualified (shaft_tests_not_run),
        never single-spectrum evidence."""
        sid = _write_and_load(
            repo, data_dir, "env_outer", _outer_race_10s(), 12000.0, **FREQUENCIES_6205
        )

        result = await tools["generate_envelope_report"](signal_id=sid)

        (check,) = line_checks
        assert check["rpm"] is None and check["shaft_frequency_hz"] is None
        assert check["outcome"] == "indicated"
        assert check["claim"]["level"] == "qualified"
        assert check["claim"]["conditions"] == ["shaft_tests_not_run"]
        assert check["most_likely_fault"] is None
        (label,) = result["bearing_matches"]
        assert label.startswith("BPFO line at ")
        assert label.endswith(
            "above the local noise floor and corroborated in this spectrum; no shaft "
            "speed is declared, so the sideband test and the shaft-order test were "
            "not run"
        )
        assert result["bearing_assessment"] == OUTER_RACE_WITHOUT_SPEED
        rendered = Path(result["file_path"]).read_text(encoding="utf-8")
        assert "No shaft speed is declared for this signal" in rendered
        assert html_escape(label) in rendered
        assert html_escape(result["bearing_assessment"]) in rendered

    @pytest.mark.asyncio
    async def test_a_declared_measurement_speed_runs_every_test(
        self, tools, data_dir, repo, reports_dir
    ):
        """The measurement's rpm, as diagnose_vibration resolves it, gives the
        check its shaft rate: the labels carry no note."""
        sid = _write_and_load(
            repo,
            data_dir,
            "env_outer_rpm",
            _outer_race_10s(),
            12000.0,
            measurement={
                "asset_id": "P-101",
                "measurement_point_id": "motor_de_h",
                "acquired_at": "2026-10-22T08:00:00+00:00",
                "rpm": RPM_6205,
            },
            **FREQUENCIES_6205,
        )

        result = await tools["generate_envelope_report"](signal_id=sid)

        (label,) = result["bearing_matches"]
        assert label.startswith("BPFO line at ")
        assert label.endswith(
            "above the local noise floor and corroborated in this spectrum"
        )
        rendered = Path(result["file_path"]).read_text(encoding="utf-8")
        assert "No shaft speed is declared" not in rendered

    @pytest.mark.asyncio
    async def test_without_a_speed_the_claim_is_no_stronger_than_the_bearing_check(
        self, tools, data_dir, repo, reports_dir, line_checks
    ):
        """Review finding: on a record where check_bearing_faults, given the
        speed, qualifies an outer-race indication (inner_race_sideband_not_
        excluded), the report without a speed used to name the outer race as
        single-spectrum evidence. Without rpm it is now qualified because the
        shaft tests did not run; with rpm=1797 it states the claim
        check_bearing_faults states."""
        sid = _write_and_load(
            repo,
            data_dir,
            "env_outer_weak_inner",
            _outer_race_beside_a_weak_inner_race(),
            12000.0,
            **DECLARED_6205,
        )
        faults = await check_bearing_faults(
            None, sid, rpm=RPM_6205, frequencies=dict(DECLARED_6205)
        )
        assert faults.outcome == "indicated"
        assert faults.claim.level == "qualified"
        assert faults.claim.conditions == ["inner_race_sideband_not_excluded"]

        without = await tools["generate_envelope_report"](signal_id=sid)
        with_speed = await tools["generate_envelope_report"](
            signal_id=sid, rpm=RPM_6205
        )

        no_speed_check, speed_check = line_checks
        assert no_speed_check["outcome"] == "indicated"
        assert no_speed_check["claim"]["level"] != "single_spectrum_evidence"
        assert no_speed_check["claim"]["level"] == "qualified"
        assert no_speed_check["claim"]["conditions"] == ["shaft_tests_not_run"]
        assert no_speed_check["most_likely_fault"] is None
        assert without["bearing_assessment"] == OUTER_RACE_WITHOUT_SPEED
        assert speed_check["claim"] == faults.claim.model_dump()
        assert speed_check["most_likely_fault"] is None
        assert with_speed["bearing_assessment"] == faults.overall_assessment

    @pytest.mark.asyncio
    async def test_an_explicit_rpm_runs_the_sideband_and_shaft_order_tests(
        self, tools, data_dir, repo, reports_dir, line_checks
    ):
        """A signal that declares no speed: the rpm parameter gives the check
        its shaft rate, the labels carry no note, and the claim is the one
        check_bearing_faults states with the same speed."""
        sid = _write_and_load(
            repo,
            data_dir,
            "env_outer_arg",
            _outer_race_10s(),
            12000.0,
            **FREQUENCIES_6205,
        )
        faults = await check_bearing_faults(
            None, sid, rpm=RPM_6205, frequencies=dict(FREQUENCIES_6205)
        )

        result = await tools["generate_envelope_report"](signal_id=sid, rpm=RPM_6205)

        (check,) = line_checks
        assert check["rpm"] == RPM_6205
        assert check["shaft_frequency_hz"] == pytest.approx(SHAFT_6205_HZ)
        assert check["claim"] == faults.claim.model_dump()
        assert check["claim"]["level"] == "single_spectrum_evidence"
        assert result["bearing_assessment"] == faults.overall_assessment
        (label,) = result["bearing_matches"]
        assert label.startswith("BPFO line at ")
        assert label.endswith(
            "above the local noise floor and corroborated in this spectrum"
        )
        rendered = Path(result["file_path"]).read_text(encoding="utf-8")
        assert "No shaft speed is declared" not in rendered

    @pytest.mark.asyncio
    async def test_an_explicit_rpm_takes_precedence_over_the_declared_one(
        self, tools, data_dir, repo, reports_dir, line_checks
    ):
        sid = _write_and_load(
            repo,
            data_dir,
            "env_outer_both",
            _outer_race_10s(),
            12000.0,
            measurement={
                "asset_id": "P-101",
                "measurement_point_id": "motor_de_h",
                "acquired_at": "2026-10-22T08:00:00+00:00",
                "rpm": 1500.0,
            },
            **FREQUENCIES_6205,
        )

        await tools["generate_envelope_report"](signal_id=sid, rpm=RPM_6205)
        await tools["generate_envelope_report"](signal_id=sid)

        explicit, declared = line_checks
        assert explicit["rpm"] == RPM_6205
        assert declared["rpm"] == 1500.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rpm",
        [0.0, -1797.0, float("nan"), float("inf")],
        ids=["zero", "negative", "nan", "inf"],
    )
    async def test_an_invalid_rpm_is_refused_before_any_report(
        self, tools, data_dir, repo, reports_dir, rpm
    ):
        sid = _write_and_load(
            repo,
            data_dir,
            "env_outer_bad",
            _outer_race_10s(),
            12000.0,
            **FREQUENCIES_6205,
        )

        with pytest.raises(ValueError) as exc_info:
            await tools["generate_envelope_report"](signal_id=sid, rpm=rpm)

        msg = str(exc_info.value)
        assert "rpm must be positive and finite" in msg
        assert "the rpm parameter" in msg
        assert "omit rpm" in msg
        assert list(reports_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_a_label_outside_the_vocabulary_is_not_capped_without_a_speed(
        self, tools, data_dir, repo, reports_dir, line_checks
    ):
        """A custom label never gets the shaft-order rule: without rpm its
        indication stays single-spectrum evidence, label-neutral."""
        sid = _write_and_load(repo, data_dir, "env_gmf", _outer_race_10s(), 12000.0)

        result = await tools["generate_envelope_report"](
            signal_id=sid, bearing_freqs={"GMF": FREQUENCIES_6205["BPFO"]}
        )

        (check,) = line_checks
        assert check["rpm"] is None
        assert check["outcome"] == "indicated"
        assert check["claim"]["level"] == "single_spectrum_evidence"
        assert check["claim"]["conditions"] == []
        assert result["bearing_assessment"].startswith(
            "The envelope evidence of this single spectrum is consistent with a "
            "periodic line at the declared GMF frequency: the GMF line at"
        )
        assert "defect" not in result["bearing_assessment"]
        (label,) = result["bearing_matches"]
        assert label.startswith("GMF line at ")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpm", [None, RPM_6205], ids=["no_rpm", "rpm"])
    async def test_without_bearing_frequencies_the_report_stays_descriptive(
        self, tools, data_dir, repo, reports_dir, line_checks, monkeypatch, rpm
    ):
        """No bearing_freqs and none in the companion: no line test runs, with
        or without rpm, and the report states no claim, marks no line and
        draws no bearing reference."""

        def no_line_test(*args, **kwargs):
            raise AssertionError("the line test ran without reference frequencies")

        monkeypatch.setattr(report_tools, "check_frequency_set", no_line_test)
        sid = _write_and_load(repo, data_dir, "env_plain", _outer_race_10s(), 12000.0)
        kwargs = {} if rpm is None else {"rpm": rpm}

        result = await tools["generate_envelope_report"](signal_id=sid, **kwargs)

        assert line_checks == [None]
        assert result["bearing_matches"] == []
        assert result["bearing_assessment"] is None
        assert result["metadata"]["bearing_frequencies"] is None
        assert result["peak_frequencies"]
        rendered = Path(result["file_path"]).read_text(encoding="utf-8")
        assert "Envelope Spectrum Peaks" in rendered
        for absent in (
            "Bearing line test",
            "Bearing Characteristic Frequencies",
            "consistent with",
            "single spectrum",
            "No single fault type",
            "inconclusive",
            "No eligible line",
            "No shaft speed is declared",
        ):
            assert absent not in rendered, absent

    @pytest.mark.asyncio
    async def test_existing_calls_keep_their_meaning(
        self, repo, reports_dir, line_checks
    ):
        """rpm is the last parameter, after ctx: every positional argument
        keeps its position, and a call without rpm runs as before."""
        assert list(
            inspect.signature(report_tools.generate_envelope_report).parameters
        ) == [
            "signal_id",
            "filter_low",
            "filter_high",
            "max_freq",
            "num_peaks",
            "bearing_freqs",
            "ctx",
            "rpm",
        ]
        freqs = {"BPFO": 81.0, "BPFI": 119.0, "BSF": 64.0, "FTF": 15.0}
        generate = report_tools.generate_envelope_report

        results = [
            await generate("report_test", 100.0, 4000.0, 500.0, 15, freqs),
            await generate("report_test", 100.0, 4000.0, 500.0, 15, freqs, None),
            await generate(
                signal_id="report_test",
                filter_low=100.0,
                filter_high=4000.0,
                max_freq=500.0,
                num_peaks=15,
                bearing_freqs=freqs,
            ),
        ]

        assert len(line_checks) == 3
        assert all(check["rpm"] is None for check in line_checks)
        assert line_checks[0] == line_checks[1] == line_checks[2]
        for result in results:
            assert result["metadata"]["filter_band"] == (100.0, 4000.0)
            assert result["metadata"]["bearing_frequencies"] == freqs

    def test_the_input_schema_gains_only_an_optional_rpm(self, mcp):
        (tool,) = [
            t
            for t in mcp._tool_manager._tools.values()
            if t.name == "generate_envelope_report"
        ]
        properties = tool.parameters["properties"]

        assert set(properties) == {
            "signal_id",
            "filter_low",
            "filter_high",
            "max_freq",
            "num_peaks",
            "bearing_freqs",
            "rpm",
        }
        assert tool.parameters["required"] == ["signal_id"]
        assert properties["rpm"] == {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "default": None,
            "title": "Rpm",
        }
        assert tool.context_kwarg == "ctx"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpm", [None, RPM_6205], ids=["no_rpm", "rpm"])
    async def test_noiseless_generated_signal_withholds_the_line_labels(
        self, tools, data_dir, repo, reports_dir, monkeypatch, line_checks, rpm
    ):
        """The report reads the claim's acquisition eligibility from the
        signal's own companion, as check_bearing_faults does: a
        generate_test_signal record with noise_level 0 gets no line label and
        the withheld statement, with and without rpm (with an unknown
        eligibility, the same record and frequencies get line labels)."""
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
            generate_test_signal,
        )

        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", data_dir
        )
        info = await generate_test_signal(
            signal_type="bearing_fault",
            duration=2.0,
            sampling_rate=10000.0,
            noise_level=0.0,
            random_seed=7,
        )
        kwargs = {} if rpm is None else {"rpm": rpm}

        result = await tools["generate_envelope_report"](
            signal_id=info.signal_id, bearing_freqs=dict(FREQUENCIES_6205), **kwargs
        )

        (check,) = line_checks
        assert check["rpm"] == rpm
        assert check["claim"]["level"] == "withheld"
        assert check["claim"]["conditions"] == ["generated_without_noise"]
        assert result["bearing_matches"] == []
        assert result["bearing_assessment"] == (
            "The signal's companion metadata declares a generated signal with an "
            "additive noise level of 0; the line test assumes a recording with "
            "measurement noise, so no bearing fault type is assessed."
        )

    @pytest.mark.asyncio
    async def test_the_report_plots_the_spectrum_the_line_test_used(
        self, tools, data_dir, repo, reports_dir, monkeypatch
    ):
        """The plotted envelope spectrum is the engine's, on the report's band
        (mean removed, Hann window), not a second computation."""
        import predictive_maintenance_mcp.report_generator as generator
        from predictive_maintenance_mcp.signal_processing.spectral import (
            envelope_spectrum_arrays,
        )

        plotted = {}
        render = generator.create_envelope_report

        def capture(**kwargs):
            plotted.update(kwargs)
            return render(**kwargs)

        monkeypatch.setattr(generator, "create_envelope_report", capture)
        sid = _write_and_load(
            repo, data_dir, "env_plot", _outer_race_10s(), 12000.0, **FREQUENCIES_6205
        )

        await tools["generate_envelope_report"](signal_id=sid)

        freqs, mags = envelope_spectrum_arrays(
            repo.get_signal(sid), 12000.0, (500.0, 5000.0)
        )
        shown = freqs <= 500.0
        expected_db = 20 * np.log10((mags[shown] + 1e-12) / np.max(mags[shown]))
        assert np.allclose(plotted["env_freq"], freqs[shown])
        assert np.allclose(plotted["env_mag_db"], expected_db)


# ---------------------------------------------------------------------------
# Generate ISO report
# ---------------------------------------------------------------------------


class TestGenerateISOReport:
    """Tests for generate_iso_report tool (signal_id handle)."""

    @pytest.mark.asyncio
    async def test_generates_iso_report(self, tools, repo, reports_dir, mock_ctx):
        result = await tools["generate_iso_report"](
            signal_id="report_test",
            machine_group=2,
            support_type="rigid",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    @pytest.mark.asyncio
    async def test_iso_report_signal_without_rate(
        self, tools, data_dir, repo, reports_dir
    ):
        sig = np.random.randn(5000)
        pd.DataFrame(sig).to_csv(
            data_dir / "no_meta_iso.csv", index=False, header=False
        )
        repo.load_signal("no_meta_iso.csv")
        with pytest.raises(ValueError, match="No sampling rate"):
            await tools["generate_iso_report"](signal_id="no_meta_iso")

    @pytest.mark.asyncio
    async def test_iso_report_undeclared_unit_refused(
        self, tools, data_dir, repo, reports_dir
    ):
        """Rate known but unit undeclared → structured error naming the fix."""
        sig = np.random.randn(20000)
        pd.DataFrame(sig).to_csv(
            data_dir / "no_unit_iso.csv", index=False, header=False
        )
        repo.load_signal("no_unit_iso.csv", sampling_rate=10000)
        with pytest.raises(ValueError, match="unit not declared"):
            await tools["generate_iso_report"](signal_id="no_unit_iso")

    @pytest.mark.asyncio
    async def test_iso_report_invalid_rpm_is_refused_through_assess_severity(
        self, tools, repo, reports_dir, mock_ctx
    ):
        """Issue #83: the ISO report delegates to assess_severity and inherits
        its rpm refusal; no report file is written."""
        with pytest.raises(ValueError, match="rpm must be positive and finite"):
            await tools["generate_iso_report"](
                signal_id="report_test", rpm=0.0, ctx=mock_ctx
            )
        assert list(reports_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Load → analyze → report round-trip (R7 golden path)
# ---------------------------------------------------------------------------


class TestSignalIdRoundTrip:
    """R7: load → analyze → report is walkable with ONE idiom (signal_id)."""

    @pytest.mark.asyncio
    async def test_load_analyze_report_single_idiom(
        self, tools, data_dir, reports_dir, mock_ctx
    ):
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
            load_signal,
        )
        from predictive_maintenance_mcp.mcp_tools.analysis_tools import (
            analyze_fft,
        )

        repo = get_repository()
        repo.clear_all()
        try:
            info = await load_signal(ctx=mock_ctx, filepath="report_test.csv")
            sid = info.signal_id
            assert sid == "report_test"

            fft_result = await analyze_fft(ctx=mock_ctx, signal_id=sid)
            assert abs(fft_result.peak_frequency - 50.0) < 2.0

            report = await tools["generate_fft_report"](signal_id=sid)
            assert Path(report["file_path"]).exists()
        finally:
            repo.clear_all()


# ---------------------------------------------------------------------------
# Generate PCA visualization report
# ---------------------------------------------------------------------------


class TestGeneratePCAReport:
    """Tests for generate_pca_visualization_report tool (signal_id handles)."""

    @pytest.fixture
    def models_dir(self, tmp_path, monkeypatch, data_dir):
        """Create a fake trained model with scaler, PCA, and metadata."""
        md = tmp_path / "models"
        md.mkdir()
        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.report_tools.MODELS_DIR", md
        )

        # Build a small model from synthetic features
        rng = np.random.RandomState(42)
        n_samples, n_features = 50, 17
        X_train = rng.randn(n_samples, n_features)

        scaler = StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        pca = PCA(n_components=2).fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_pca)

        model_name = "test_pca_model"
        with open(md / f"{model_name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(md / f"{model_name}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(md / f"{model_name}_pca.pkl", "wb") as f:
            pickle.dump(pca, f)
        with open(md / f"{model_name}_metadata.json", "w") as f:
            json.dump({"sampling_rate": 10000.0, "model_type": "OneClassSVM"}, f)

        return md

    @pytest.mark.asyncio
    async def test_pca_report_no_test_signals(
        self, tools, repo, reports_dir, models_dir, mock_ctx
    ):
        """PCA report with no test signals should still produce a report."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["summary"]["total_segments"] == 0

    @pytest.mark.asyncio
    async def test_pca_report_with_test_signals(
        self, tools, repo, reports_dir, models_dir, mock_ctx
    ):
        """PCA report with test signal_ids should segment, predict, and report."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_ids=["report_test"],
            segment_duration=0.1,
            overlap_ratio=0.5,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert result["summary"]["total_segments"] > 0

    @pytest.mark.asyncio
    async def test_pca_report_with_true_labels(
        self, tools, repo, reports_dir, models_dir, mock_ctx
    ):
        """PCA report with true_labels (keyed by signal_id) computes metrics."""
        result = await tools["generate_pca_visualization_report"](
            model_name="test_pca_model",
            test_signal_ids=["report_test"],
            true_labels={"report_test": "healthy"},
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert result["metadata"]["validation_mode"] is True
        assert "validation_metrics" in result["summary"]
        assert "overall_accuracy" in result["summary"]["validation_metrics"]

    @pytest.mark.asyncio
    async def test_pca_report_model_not_found(
        self, tools, repo, reports_dir, models_dir
    ):
        with pytest.raises(FileNotFoundError):
            await tools["generate_pca_visualization_report"](
                model_name="nonexistent_model",
            )

    @pytest.mark.asyncio
    async def test_pca_report_signal_not_loaded(
        self, tools, repo, reports_dir, models_dir
    ):
        """Unknown test signal_id → fail fast, no silent skipping."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_pca_visualization_report"](
                model_name="test_pca_model",
                test_signal_ids=["__ghost__"],
            )


# ---------------------------------------------------------------------------
# Generate feature comparison report
# ---------------------------------------------------------------------------


class TestGenerateFeatureComparisonReport:
    """Tests for generate_feature_comparison_report tool (signal_id handles)."""

    @pytest.fixture
    def multi_signal_ids(self, repo, data_dir):
        """Load two signal groups for comparison."""
        ids = {
            "healthy_1": _load_extra_signal(
                repo, data_dir, "healthy_1.csv", freq=50.0, noise=0.01
            ),
            "healthy_2": _load_extra_signal(
                repo, data_dir, "healthy_2.csv", freq=50.0, noise=0.02
            ),
            "faulty_1": _load_extra_signal(
                repo, data_dir, "faulty_1.csv", freq=50.0, noise=1.0
            ),
        }
        return ids

    @pytest.mark.asyncio
    async def test_feature_comparison_basic(
        self, tools, multi_signal_ids, reports_dir, mock_ctx
    ):
        result = await tools["generate_feature_comparison_report"](
            signal_groups={
                "Healthy": ["healthy_1", "healthy_2"],
                "Faulty": ["faulty_1"],
            },
            segment_duration=0.1,
            ctx=mock_ctx,
        )
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        assert result["metadata"]["report_type"] == "feature_comparison"
        assert "Healthy" in result["metadata"]["groups"]

    @pytest.mark.asyncio
    async def test_feature_comparison_subset_features(
        self, tools, multi_signal_ids, reports_dir
    ):
        """Only plot a subset of features."""
        result = await tools["generate_feature_comparison_report"](
            signal_groups={"A": ["healthy_1"], "B": ["faulty_1"]},
            features_to_plot=["rms", "kurtosis", "crest_factor"],
        )
        assert len(result["metadata"]["features_plotted"]) == 3

    @pytest.mark.asyncio
    async def test_feature_comparison_unknown_id_raises(
        self, tools, multi_signal_ids, reports_dir
    ):
        """Unknown signal_ids fail fast — no silent skipping (U8 contract)."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_feature_comparison_report"](
                signal_groups={
                    "Mixed": ["healthy_1", "__does_not_exist__"],
                },
            )


# ---------------------------------------------------------------------------
# DOCX report generation
# ---------------------------------------------------------------------------


def _docx_installed() -> bool:
    try:
        import docx  # noqa: F401

        return True
    except ImportError:
        return False


class TestDocxReport:
    """Tests for generate_diagnostic_report_docx tool (signal_id handle).

    Since U6 the tool RAISES ValueError when python-docx is missing (error
    contract: failures raise, never error dicts) — so the happy-path tests
    are skipped when the optional dependency is not installed.
    """

    @pytest.mark.asyncio
    async def test_generates_docx(self, tools, repo, reports_dir, mock_ctx):
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            ctx=mock_ctx,
            signal_id="report_test",
            sections={"diagnosis": "Test diagnosis summary"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_docx_with_all_sections(self, tools, repo, reports_dir, mock_ctx):
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            ctx=mock_ctx,
            signal_id="report_test",
            sections={
                "statistics": {"RMS": 0.707, "Kurtosis": 3.0, "Crest Factor": 1.414},
                "fft_peaks": [{"frequency": 50.0, "magnitude_db": 0.0, "note": "1x"}],
                "diagnosis": "All parameters within normal range.",
            },
            title="Custom DOCX Title",
        )
        assert result is not None
        assert "error" not in result
        assert "file_path" in result

    @pytest.mark.asyncio
    async def test_docx_no_ctx(self, tools, repo, reports_dir):
        """DOCX generation without ctx should still work."""
        if not _docx_installed():
            pytest.skip("python-docx not installed")
        result = await tools["generate_diagnostic_report_docx"](
            signal_id="report_test",
            sections={"diagnosis": "Summary"},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_docx_signal_not_loaded_raises(self, tools, repo, reports_dir):
        """Reports are only produced for loaded signals."""
        with pytest.raises(ValueError, match="load_signal"):
            await tools["generate_diagnostic_report_docx"](
                signal_id="__ghost__",
                sections={"diagnosis": "Summary"},
            )

    @pytest.mark.asyncio
    async def test_docx_missing_dependency_raises(
        self, tools, repo, reports_dir, mock_ctx, monkeypatch
    ):
        """Missing python-docx → raised ValueError, never an error dict."""
        monkeypatch.setattr(
            "predictive_maintenance_mcp.report_generator.HAS_DOCX", False
        )
        with pytest.raises(ValueError, match="python-docx"):
            await tools["generate_diagnostic_report_docx"](
                ctx=mock_ctx,
                signal_id="report_test",
                sections={"diagnosis": "Summary"},
            )


# ---------------------------------------------------------------------------
# Tool registration completeness
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify all expected tools are registered."""

    def test_all_expected_tools_registered(self, tools):
        expected = {
            "plot_signal",
            "generate_fft_report",
            "generate_envelope_report",
            "generate_iso_report",
            "list_html_reports",
            "generate_pca_visualization_report",
            "generate_feature_comparison_report",
            "generate_diagnostic_report_docx",
            "generate_diagnostic_report",
        }
        assert set(tools) == expected

    def test_no_report_tool_takes_the_eligibility_as_an_argument(self, tools):
        import inspect

        for name in ("generate_diagnostic_report", "generate_envelope_report"):
            parameters = inspect.signature(tools[name]).parameters
            assert "acquisition_eligibility" not in parameters, name


# ---------------------------------------------------------------------------
# generate_diagnostic_report — the integrated, server-authored report
#
# The point of this tool is that the caller supplies inputs, not wording.
# These tests hold that line.
# ---------------------------------------------------------------------------


class TestGenerateDiagnosticReport:
    @pytest.mark.asyncio
    async def test_returns_statements_and_one_file_per_format(
        self, tools, repo, reports_dir, mock_ctx
    ):
        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, ctx=mock_ctx
        )
        assert result["statements"], "no authored statements returned"
        assert [f["format"] for f in result["files"]] == ["html"]
        assert Path(result["files"][0]["file_path"]).exists()
        assert result["bearing_check"]["status"] == "not_requested"
        assert any(
            "generate_diagnostic_report with bearing_id" in s
            for s in result["statements"]
        )
        assert not any("companion metadata" in s for s in result["statements"])

    @pytest.mark.asyncio
    async def test_statements_match_what_the_document_renders(
        self, tools, repo, reports_dir, mock_ctx
    ):
        """Covers AE4."""
        import html as _html

        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, ctx=mock_ctx
        )
        rendered = Path(result["files"][0]["file_path"]).read_text(encoding="utf-8")
        missing = [s for s in result["statements"] if _html.escape(s) not in rendered]
        assert missing == [], f"returned but not rendered: {missing}"

    @pytest.mark.asyncio
    async def test_unknown_signal_raises_rather_than_returning_an_error_dict(
        self, tools, repo, reports_dir, mock_ctx
    ):
        with pytest.raises(ValueError):
            await tools["generate_diagnostic_report"](
                signal_id="nope", rpm=1500.0, ctx=mock_ctx
            )

    @pytest.mark.asyncio
    async def test_unsupported_format_raises_before_any_analysis(
        self, tools, repo, reports_dir, mock_ctx
    ):
        with pytest.raises(ValueError, match="Unsupported report format"):
            await tools["generate_diagnostic_report"](
                signal_id="report_test", rpm=1500.0, formats=["docx"], ctx=mock_ctx
            )

    @pytest.mark.asyncio
    async def test_invalid_rpm_is_refused_before_either_diagnosis(
        self, tools, repo, data_dir, reports_dir, mock_ctx
    ):
        """Issue #83: one rpm feeds both diagnoses, so it is refused up front
        and no report is written."""
        _load_extra_signal(repo, data_dir, "report_baseline.csv")
        with pytest.raises(
            ValueError, match=r"rpm must be positive and finite.*the rpm parameter"
        ):
            await tools["generate_diagnostic_report"](
                signal_id="report_test",
                rpm=-1.0,
                baseline_signal_id="report_baseline",
                ctx=mock_ctx,
            )
        assert list(reports_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_unsupported_format_is_still_reported_before_the_rpm(
        self, tools, repo, reports_dir, mock_ctx
    ):
        with pytest.raises(ValueError, match="Unsupported report format"):
            await tools["generate_diagnostic_report"](
                signal_id="report_test", rpm=-1.0, formats=["docx"], ctx=mock_ctx
            )

    @pytest.mark.asyncio
    async def test_tiny_rpm_with_a_catalog_bearing_writes_no_report(
        self, tools, repo, reports_dir, mock_ctx
    ):
        """0.5 rpm rounds the 6205's FTF to 0.0 Hz: refused instead of a
        report whose bearing section says no designation was supplied."""
        with pytest.raises(ValueError, match="expected FTF frequency must be positive"):
            await tools["generate_diagnostic_report"](
                signal_id="report_test", rpm=0.5, bearing_id="6205", ctx=mock_ctx
            )
        assert list(reports_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_unknown_bearing_id_report_names_the_catalog_miss_and_remedy(
        self, tools, repo, reports_dir, mock_ctx
    ):
        """Issue #84: the report said no designation was supplied and asked
        for a bearing_id although one was passed. Section, verdict,
        recommendations and figure caption now give the same reason."""
        import html as _html

        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, bearing_id="9999", ctx=mock_ctx
        )

        assert "not computed for bearing '9999'" in result["verdict"]
        assert result["bearing_check"]["status"] == "not_computed"
        assert result["bearing_check"]["reason_code"] == "not_in_catalog"
        statements = result["statements"]
        assert not any("no bearing designation" in s.lower() for s in statements)
        reason = next(
            s for s in statements if "'9999' is not in the bearing catalog" in s
        )
        assert reason.startswith("Bearing characteristic-frequency matching")
        actions = [r["action"] for r in result["recommendations"]]
        assert any(
            "check_bearing_faults(signal_id='report_test', rpm=1500," in a
            for a in actions
        )
        rendered = Path(result["files"][0]["file_path"]).read_text(encoding="utf-8")
        missing = [s for s in statements if _html.escape(s) not in rendered]
        assert missing == [], f"returned but not rendered: {missing}"
        assert "No bearing designation was supplied" not in rendered
        assert _html.escape("Envelope spectrum. Bearing '9999' is not in") in rendered

    @pytest.mark.asyncio
    async def test_no_confidence_field_reaches_the_caller(
        self, tools, repo, reports_dir, mock_ctx
    ):
        """Covers AE6 at the tool boundary — the surface an LLM actually reads."""
        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, ctx=mock_ctx
        )

        def walk(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert "confidence" not in key.lower(), f"{path}.{key}"
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")

        walk(result)
        assert "not a confidence" in result["evidence_strength_note"].lower()

    @pytest.mark.asyncio
    async def test_baseline_signal_produces_a_comparison(
        self, tools, repo, data_dir, reports_dir, mock_ctx
    ):
        _load_extra_signal(repo, data_dir, "report_baseline.csv")
        result = await tools["generate_diagnostic_report"](
            signal_id="report_test",
            rpm=1500.0,
            baseline_signal_id="report_baseline",
            ctx=mock_ctx,
        )
        joined = " ".join(result["statements"]).lower()
        assert "baseline" in joined

    @pytest.mark.asyncio
    async def test_absent_baseline_states_the_absence(
        self, tools, repo, reports_dir, mock_ctx
    ):
        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, ctx=mock_ctx
        )
        joined = " ".join(result["statements"]).lower()
        assert "no healthy baseline was supplied" in joined

    @pytest.mark.asyncio
    async def test_recommendations_carry_motivation(
        self, tools, repo, reports_dir, mock_ctx
    ):
        result = await tools["generate_diagnostic_report"](
            signal_id="report_test", rpm=1500.0, ctx=mock_ctx
        )
        assert result["recommendations"]
        for rec in result["recommendations"]:
            assert rec["motivation"]

    @pytest.mark.asyncio
    async def test_noiseless_generated_signal_withholds_the_claim(
        self, tools, data_dir, repo, reports_dir, mock_ctx, monkeypatch
    ):
        """The signal's own companion (generate_test_signal, noise_level 0)
        reaches the report's diagnosis: the claim is withheld there too."""
        from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
            generate_test_signal,
        )

        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.acquisition_tools.DATA_DIR", data_dir
        )
        info = await generate_test_signal(
            signal_type="bearing_fault",
            duration=2.0,
            sampling_rate=10000.0,
            noise_level=0.0,
            random_seed=7,
        )

        result = await tools["generate_diagnostic_report"](
            signal_id=info.signal_id, rpm=1800.0, bearing_id="6205", ctx=mock_ctx
        )

        assert result["verdict"] == (
            "The signal's companion metadata declares a generated signal with an "
            "additive noise level of 0; the line test assumes a recording with "
            "measurement noise, so no bearing fault type is assessed."
        )
        assert result["fault_canonical"] is None
        assert "not assessed" in result["statements"]
        assert not [
            r for r in result["recommendations"] if r["action"].startswith("Bearing")
        ]

    @pytest.mark.asyncio
    async def test_single_spectrum_evidence_report_asks_for_confirmation_only(
        self, tools, data_dir, repo, reports_dir, mock_ctx
    ):
        """End to end on an outer-race defect: the verdict names the type as
        single-spectrum evidence with its confirmation step, the figure
        emphasises its window, and no statement or action asks for
        replacement."""
        import html as _html

        sid = _write_and_load(repo, data_dir, "rep_outer", _outer_race_10s(), 12000.0)

        result = await tools["generate_diagnostic_report"](
            signal_id=sid, rpm=RPM_6205, bearing_id="6205", ctx=mock_ctx
        )

        assert result["fault_canonical"] == "outer_race"
        assert result["verdict"].startswith(
            "The envelope evidence of this single spectrum is consistent with an "
            "outer-race defect"
        )
        assert result["verdict"].endswith(
            "Bearing 6205: confirm the single-spectrum indication with a comparable "
            "measurement or an inspection of the outer race."
        )
        caption = next(
            s for s in result["statements"] if s.startswith("Envelope spectrum")
        )
        assert caption.endswith(
            "The emphasised window (BPFO) holds the candidate line of the bearing "
            "statement in this report."
        )
        texts = result["statements"] + [r["action"] for r in result["recommendations"]]
        assert not [t for t in texts if "replac" in t.lower()]
        rendered = Path(result["files"][0]["file_path"]).read_text(encoding="utf-8")
        missing = [s for s in result["statements"] if _html.escape(s) not in rendered]
        assert missing == [], f"returned but not rendered: {missing}"

    def test_docstring_states_the_caller_authorship_prohibition(self, tools):
        """The docstring is what a model reads before deciding how to render."""
        doc = tools["generate_diagnostic_report"].__doc__ or ""
        lowered = doc.lower()
        assert "authorship contract" in lowered
        assert "verbatim" in lowered
        assert "confidence" in lowered

    def test_server_instructions_carry_the_same_rule(self):
        from predictive_maintenance_mcp.server import mcp as server

        instructions = (server.instructions or "").lower()
        assert "report authorship policy" in instructions
        assert "no confidence score" in instructions
