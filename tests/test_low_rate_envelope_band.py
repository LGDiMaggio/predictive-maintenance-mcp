"""Signals sampled at 1002 Hz or less: the default envelope band is empty.

The default envelope band runs from 500 Hz to ``min(5000, fs/2 - 1)``, so it is
empty at fs <= 1002 Hz. The tools that use it were refusing such a signal with
a message asking the caller to re-specify band edges it had not passed. This
module pins what each surface says instead, at 800, 1000 and 1002 Hz, against
1003 Hz and 10 kHz, where the band exists:

- ``generate_diagnostic_report`` produces the report, with or without a
  bearing. The report carries no envelope spectrum figure and says why, in its
  statements and in the rendering, and a requested bearing check shows the
  reason and remedy ``diagnose_vibration`` gives. Its tool description says
  the annotated envelope spectrum is included only when the default band is
  usable.
- ``check_bearing_faults`` (the default band of the shared resolver) and
  ``generate_envelope_report`` without ``filter_high`` refuse with a message
  that names the sampling rate and the remedy. Bands passed explicitly keep
  their own validation messages.
- The asset ledger stores the new message for a new snapshot of such a
  measurement, under unchanged identifiers, and keeps the text of a snapshot
  recorded before.

Each sentence is compared with its full expected text. The controls at the end
mutate a produced output (substitute a word, negate the statement, move it)
and show that the checks reject each mutation. Synthetic, seeded noise only.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from conftest import patch_data_dir
from predictive_maintenance_mcp import report_generator
from predictive_maintenance_mcp.asset_ledger import snapshot as snapshot_module
from predictive_maintenance_mcp.asset_ledger.snapshot import (
    SnapshotPolicy,
    compute_health_snapshot,
)
from predictive_maintenance_mcp.asset_ledger.store import LedgerStore
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.mcp_tools import (
    acquisition_tools,
    analysis_tools,
    asset_tools,
    diagnostics_tools,
    register_all,
    report_tools,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.spectral import (
    resolve_envelope_band,
    validate_bandpass_band,
)

LOW_FS = (800.0, 1000.0, 1002.0)
HIGH_FS = (1003.0, 10000.0)
SEED = 20261120
RPM = 1797.0
BEARING = "6205"
FREQUENCIES = {"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.58, "FTF": 11.93}


# ---------------------------------------------------------------------------
# The expected sentences, written out here and not taken from the code
# ---------------------------------------------------------------------------


def absent_figure(fs: float) -> str:
    """The report's statement when the default envelope band is empty."""
    return (
        f"This report carries no envelope spectrum figure: at fs={fs:g} Hz the "
        f"default envelope band, from 500 Hz to just below the Nyquist frequency "
        f"({fs / 2:g} Hz), is empty."
    )


def default_band_refusal(fs: float) -> str:
    """The refusal of the default envelope band when it is empty."""
    return (
        f"At fs={fs:g} Hz the default envelope band, from 500 Hz to just below "
        f"the Nyquist frequency ({fs / 2:g} Hz), is empty, so no envelope spectrum "
        f"can be computed. Re-acquire the signal at a higher sampling rate, then "
        f"re-run the analysis. The default envelope band runs from 500 Hz to just "
        f"below half the sampling rate: it is empty at 1002 Hz or less, and it "
        f"reaches its full width, 500 to 5000 Hz, only above 10 kHz."
    )


def envelope_report_refusal(fs: float, filter_low: float = 500.0) -> str:
    """generate_envelope_report's refusal when its band, without filter_high, is empty."""
    return (
        f"At fs={fs:g} Hz the envelope band of this report, from {filter_low:g} Hz "
        f"to just below the Nyquist frequency ({fs / 2:g} Hz), is empty, so no "
        f"envelope spectrum can be computed. Re-acquire the signal at a higher "
        f"sampling rate, then re-run the report, or pass filter_low and "
        f"filter_high below the Nyquist frequency, with filter_low below "
        f"filter_high."
    )


#: The three sentences at fs = 1000 Hz, spelled out in full: they pin the
#: number formatting the templates above share with the code.
ABSENT_FIGURE_AT_1000 = (
    "This report carries no envelope spectrum figure: at fs=1000 Hz the default "
    "envelope band, from 500 Hz to just below the Nyquist frequency (500 Hz), is "
    "empty."
)
DEFAULT_BAND_REFUSAL_AT_1000 = (
    "At fs=1000 Hz the default envelope band, from 500 Hz to just below the "
    "Nyquist frequency (500 Hz), is empty, so no envelope spectrum can be "
    "computed. Re-acquire the signal at a higher sampling rate, then re-run the "
    "analysis. The default envelope band runs from 500 Hz to just below half the "
    "sampling rate: it is empty at 1002 Hz or less, and it reaches its full "
    "width, 500 to 5000 Hz, only above 10 kHz."
)
ENVELOPE_REPORT_REFUSAL_AT_1000 = (
    "At fs=1000 Hz the envelope band of this report, from 500 Hz to just below "
    "the Nyquist frequency (500 Hz), is empty, so no envelope spectrum can be "
    "computed. Re-acquire the signal at a higher sampling rate, then re-run the "
    "report, or pass filter_low and filter_high below the Nyquist frequency, "
    "with filter_low below filter_high."
)

#: Validation messages of bands passed explicitly, unchanged by the fix.
EXPLICIT_REVERSED_AT_1000 = (
    "Invalid bandpass band: filter_high=499 Hz must be greater than "
    "filter_low=500 Hz — re-specify the band edges in (low, high) order."
)
EXPLICIT_ABOVE_NYQUIST_AT_1000 = (
    "Invalid bandpass band: filter_high=5000 Hz exceeds the Nyquist frequency "
    "500 Hz of this signal (fs=1000 Hz) — choose filter_high <= 500 Hz or "
    "re-acquire at a higher sampling rate. The band is never clamped silently."
)

#: The sentence the renderer wrote when a report had no figure; false for an
#: empty band, where no line test ran.
FIXED_ABSENT_SENTENCE = "can be checked only against the values in the table above"

#: The report's embedded metadata, which repeats statements; a statement found
#: only there is not rendered.
_METADATA_BLOCK = re.compile(
    r'<script type="application/json" id="report-metadata">.*?</script>', re.DOTALL
)


def envelope_card(statement: str) -> str:
    """The rendered card that holds the absent-figure statement."""
    return (
        '<div class="card"><h2 class="card-title">Envelope spectrum</h2>'
        f"<p>{html.escape(statement)}</p></div>"
    )


# ---------------------------------------------------------------------------
# Checks, each exercised by a control at the end of the module
# ---------------------------------------------------------------------------


def assert_absent_figure_statement(rendered: str, statements: list, fs: float) -> None:
    """The report says, once, in its statements and its card, why it has no figure."""
    expected = absent_figure(fs)
    assert statements.count(expected) == 1, "absent-figure statement not returned once"
    assert rendered.count(envelope_card(expected)) == 1, "statement not in its card"
    assert rendered.count(html.escape(expected)) == 1, "statement rendered elsewhere"
    assert FIXED_ABSENT_SENTENCE not in rendered, "renderer's fixed sentence present"
    assert "<svg" not in rendered, "an envelope spectrum figure is drawn"


def assert_refused_with(error: BaseException, expected: str) -> None:
    """A ValueError (subclass allowed) whose message is exactly *expected*."""
    assert isinstance(error, ValueError), f"not a ValueError: {type(error)}"
    assert str(error) == expected, "refusal text differs"
    assert "re-specify the band edges" not in str(error), "band-edge message"


# ---------------------------------------------------------------------------
# Sandbox and calls
# ---------------------------------------------------------------------------


@dataclass
class Box:
    signals: Path
    reports: Path


@pytest.fixture
def box(tmp_path, monkeypatch):
    """Signals, an empty models folder and reports, all under ``tmp_path``."""
    sandbox = Box(tmp_path / "signals", tmp_path / "reports")
    models = tmp_path / "models"
    for path in (sandbox.signals, sandbox.reports, models):
        path.mkdir()
    patch_data_dir(monkeypatch, sandbox.signals)
    for module in (diagnosis_pipeline, diagnostics_tools, report_tools):
        monkeypatch.setattr(module, "MODELS_DIR", models)
    monkeypatch.setattr(report_tools, "REPORTS_DIR", sandbox.reports)
    monkeypatch.setattr(report_generator, "REPORTS_DIR", sandbox.reports)
    stored_before = _stored_ids()
    yield sandbox
    for signal_id in _stored_ids() - stored_before:
        get_repository().clear_signal(signal_id)


def _stored_ids() -> set:
    return {info["signal_id"] for info in get_repository().list_signals()}


def noise(fs: float, seed_offset: int = 0) -> np.ndarray:
    """Four seconds of seeded white noise at *fs*."""
    rng = np.random.default_rng((SEED, int(fs), seed_offset))
    return 0.1 * rng.standard_normal(int(4 * fs))


def write_signal(folder: Path, name: str, values: np.ndarray, companion: dict) -> None:
    with (folder / f"{name}.csv").open("w", encoding="ascii", newline="\n") as fh:
        np.savetxt(fh, values, fmt="%.8f")
    (folder / f"{name}_metadata.json").write_text(
        json.dumps(companion), encoding="utf-8"
    )


def stored(box: Box, fs: float) -> str:
    """Store the noise signal at *fs* (unit g) and return its signal id."""
    name = "f1_fs" + f"{fs:g}".replace(".", "p")
    write_signal(
        box.signals, name, noise(fs), {"sampling_rate": fs, "signal_unit": "g"}
    )
    loaded = get_repository().load_signal(f"{name}.csv")
    assert loaded["signal_id"] == name
    return name


def run(coroutine):
    return asyncio.run(coroutine)


def report(signal_id: str, **kwargs) -> tuple[dict, str]:
    """generate_diagnostic_report (HTML) and its rendering, metadata removed."""
    result = run(
        report_tools.generate_diagnostic_report(
            signal_id=signal_id, rpm=RPM, formats=["html"], **kwargs
        )
    )
    path = Path(result["files"][0]["file_path"])
    return result, _METADATA_BLOCK.sub("", path.read_text(encoding="utf-8"), count=1)


def diagnosis(signal_id: str, **kwargs) -> dict:
    result = run(
        diagnostics_tools.diagnose_vibration(AsyncMock(), signal_id, rpm=RPM, **kwargs)
    )
    return result.model_dump(mode="json")


def refusal(coroutine) -> BaseException:
    """The exception a tool call raises; a call that returns fails the test."""
    try:
        run(coroutine)
    except Exception as exc:  # the type and message are what is checked
        return exc
    pytest.fail("the call returned instead of refusing")


# ---------------------------------------------------------------------------
# generate_diagnostic_report
# ---------------------------------------------------------------------------


class TestDiagnosticReport:
    @pytest.mark.parametrize("fs", LOW_FS)
    def test_with_a_bearing_the_report_is_produced(self, box, fs):
        signal_id = stored(box, fs)
        result, rendered = report(signal_id, bearing_id=BEARING)

        assert [f["format"] for f in result["files"]] == ["html"]
        assert_absent_figure_statement(rendered, result["statements"], fs)

    @pytest.mark.parametrize("fs", LOW_FS)
    def test_without_a_bearing_the_report_is_produced(self, box, fs):
        signal_id = stored(box, fs)
        result, rendered = report(signal_id)

        assert [f["format"] for f in result["files"]] == ["html"]
        assert_absent_figure_statement(rendered, result["statements"], fs)
        assert result["bearing_check"] == diagnosis(signal_id)["bearing_check"]

    @pytest.mark.parametrize("fs", LOW_FS)
    def test_the_bearing_check_shows_the_reason_and_remedy_of_diagnose_vibration(
        self, box, fs
    ):
        signal_id = stored(box, fs)
        expected = diagnosis(signal_id, bearing_id=BEARING)["bearing_check"]
        result, rendered = report(signal_id, bearing_id=BEARING)

        assert expected["status"] == "not_computed"
        assert expected["reason_code"] == "envelope_band_unusable"
        assert result["bearing_check"] == expected
        assert any(expected["reason"] in s for s in result["statements"])
        assert expected["remedy"] in result["statements"]
        assert html.escape(expected["reason"]) in rendered
        assert html.escape(expected["remedy"]) in rendered

    def test_the_statement_at_1000_hz_reads_exactly(self, box):
        result, rendered = report(stored(box, 1000.0), bearing_id=BEARING)

        assert ABSENT_FIGURE_AT_1000 in result["statements"]
        assert envelope_card(ABSENT_FIGURE_AT_1000) in rendered

    @pytest.mark.parametrize("bearing_id", [BEARING, None], ids=["bearing", "none"])
    @pytest.mark.parametrize("fs", HIGH_FS)
    def test_where_the_band_exists_the_figure_is_drawn(self, box, fs, bearing_id):
        result, rendered = report(stored(box, fs), bearing_id=bearing_id)

        assert not [s for s in result["statements"] if "carries no envelope" in s]
        assert rendered.count('<div class="chart-container">') == 1
        assert "<svg" in rendered
        assert FIXED_ABSENT_SENTENCE not in rendered


#: The tool description's sentence on the figure, approved word for word.
FIGURE_DESCRIPTION = (
    "An annotated envelope spectrum is included when the default envelope band is "
    "usable; otherwise, the report explains why the envelope spectrum figure is "
    "absent."
)


def served_description(tool: str) -> str:
    """A tool's description as the registered server serves it, spaces collapsed."""
    mcp = MCPServer("low-rate-envelope-band")
    register_all(mcp)
    (description,) = [
        t.description or "" for t in mcp._tool_manager._tools.values() if t.name == tool
    ]
    return " ".join(description.split())


def test_the_report_description_says_when_the_figure_is_included():
    description = served_description("generate_diagnostic_report")

    assert description.count(FIGURE_DESCRIPTION) == 1
    assert "annotated envelope spectrum, and recommended actions" not in description


# ---------------------------------------------------------------------------
# check_bearing_faults and the shared resolver (default band)
# ---------------------------------------------------------------------------


class TestDefaultBandRefusal:
    @pytest.mark.parametrize("fs", LOW_FS)
    def test_check_bearing_faults_catalog_route(self, box, fs):
        signal_id = stored(box, fs)
        error = refusal(
            diagnostics_tools.check_bearing_faults(
                AsyncMock(), signal_id=signal_id, rpm=RPM, bearing_id=BEARING
            )
        )
        assert_refused_with(error, default_band_refusal(fs))

    @pytest.mark.parametrize("fs", LOW_FS)
    def test_check_bearing_faults_frequencies_route(self, box, fs):
        signal_id = stored(box, fs)
        error = refusal(
            diagnostics_tools.check_bearing_faults(
                AsyncMock(), signal_id=signal_id, rpm=RPM, frequencies=dict(FREQUENCIES)
            )
        )
        assert_refused_with(error, default_band_refusal(fs))

    def test_the_refusal_at_1000_hz_reads_exactly(self):
        with pytest.raises(ValueError) as caught:
            resolve_envelope_band(1000.0)
        assert str(caught.value) == DEFAULT_BAND_REFUSAL_AT_1000

    @pytest.mark.parametrize("fs", LOW_FS)
    def test_the_resolver_refuses_its_empty_default(self, fs):
        with pytest.raises(ValueError) as caught:
            resolve_envelope_band(fs, None)
        assert_refused_with(caught.value, default_band_refusal(fs))

    @pytest.mark.parametrize(
        "fs, band",
        [
            (1003.0, (500.0, 500.5)),
            (10000.0, (500.0, 4999.0)),
            (20000.0, (500.0, 5000.0)),
        ],
    )
    def test_the_default_band_where_it_exists(self, fs, band):
        assert resolve_envelope_band(fs) == band

    def test_an_explicit_reversed_band_keeps_its_message(self):
        with pytest.raises(ValueError) as caught:
            resolve_envelope_band(1000.0, (500.0, 499.0))
        assert str(caught.value) == EXPLICIT_REVERSED_AT_1000

    def test_an_explicit_band_above_nyquist_keeps_its_message(self):
        with pytest.raises(ValueError) as caught:
            resolve_envelope_band(1000.0, (500.0, 5000.0))
        assert str(caught.value) == EXPLICIT_ABOVE_NYQUIST_AT_1000

    def test_an_explicit_band_below_nyquist_is_honoured(self):
        assert resolve_envelope_band(1000.0, (100.0, 400.0)) == (100.0, 400.0)

    @pytest.mark.parametrize("fs", HIGH_FS)
    def test_check_bearing_faults_runs_where_the_band_exists(self, box, fs):
        summary = run(
            diagnostics_tools.check_bearing_faults(
                AsyncMock(), signal_id=stored(box, fs), rpm=RPM, bearing_id=BEARING
            )
        )
        assert summary.fault_checks

    def test_analyze_envelope_keeps_its_explicit_default_message(self, box):
        """analyze_envelope passes its band explicitly (500 to 5000 Hz)."""
        error = refusal(
            analysis_tools.analyze_envelope(ctx=None, signal_id=stored(box, 1000.0))
        )
        assert isinstance(error, ValueError)
        assert str(error) == EXPLICIT_ABOVE_NYQUIST_AT_1000


# ---------------------------------------------------------------------------
# generate_envelope_report
# ---------------------------------------------------------------------------


class TestEnvelopeReportRefusal:
    @pytest.mark.parametrize("fs", LOW_FS)
    def test_with_the_line_test(self, box, fs):
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, fs), bearing_freqs=dict(FREQUENCIES), rpm=RPM
            )
        )
        assert_refused_with(error, envelope_report_refusal(fs))
        assert not any(box.reports.iterdir())

    @pytest.mark.parametrize("fs", LOW_FS)
    def test_with_the_defaults(self, box, fs):
        error = refusal(
            report_tools.generate_envelope_report(signal_id=stored(box, fs))
        )
        assert_refused_with(error, envelope_report_refusal(fs))

    def test_the_refusal_at_1000_hz_reads_exactly(self, box):
        error = refusal(
            report_tools.generate_envelope_report(signal_id=stored(box, 1000.0))
        )
        assert str(error) == ENVELOPE_REPORT_REFUSAL_AT_1000

    @pytest.mark.parametrize(
        "fs, filter_low", [(900.0, 450.0), (10000.0, 4999.5)], ids=["900", "10000"]
    )
    def test_a_lower_edge_at_or_above_the_upper_default(self, box, fs, filter_low):
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, fs), filter_low=filter_low
            )
        )
        assert_refused_with(error, envelope_report_refusal(fs, filter_low))

    def test_a_lower_edge_from_5000_hz_keeps_its_message(self, box):
        """The refusal covers lower edges below 5000 Hz only."""
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, 1000.0), filter_low=5000.0
            )
        )
        assert str(error) == (
            "Invalid bandpass band: filter_high=499 Hz must be greater than "
            "filter_low=5000 Hz — re-specify the band edges in (low, high) order."
        )

    @pytest.mark.parametrize("filter_low", [0.0, -1.0], ids=["zero", "negative"])
    def test_a_lower_edge_not_above_zero_keeps_its_message(self, box, filter_low):
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, 1000.0), filter_low=filter_low
            )
        )
        assert str(error) == (
            f"Invalid bandpass band: filter_low={filter_low:g} Hz must be positive — "
            "pass a lower edge above 0 Hz (e.g. 500 Hz for bearing envelope analysis)."
        )

    def test_explicit_reversed_edges_keep_their_message(self, box):
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, 1000.0), filter_low=500.0, filter_high=499.0
            )
        )
        assert str(error) == EXPLICIT_REVERSED_AT_1000

    def test_explicit_edges_above_nyquist_keep_their_message(self, box):
        error = refusal(
            report_tools.generate_envelope_report(
                signal_id=stored(box, 1000.0), filter_low=500.0, filter_high=5000.0
            )
        )
        assert str(error) == EXPLICIT_ABOVE_NYQUIST_AT_1000

    def test_explicit_edges_below_nyquist_produce_the_report(self, box):
        result = run(
            report_tools.generate_envelope_report(
                signal_id=stored(box, 1000.0), filter_low=100.0, filter_high=400.0
            )
        )
        assert Path(result["file_path"]).is_file()

    @pytest.mark.parametrize("fs", HIGH_FS)
    def test_the_defaults_produce_the_report_where_the_band_exists(self, box, fs):
        result = run(report_tools.generate_envelope_report(signal_id=stored(box, fs)))
        assert Path(result["file_path"]).is_file()


# ---------------------------------------------------------------------------
# The asset ledger
# ---------------------------------------------------------------------------

ASSET, POINT = "F1-ASSET", "f1_point"
ACQUIRED_AT = "2026-11-20T09:00:00+00:00"

#: Identifiers of the scenario below, recorded before the fix: they do not
#: depend on the refusal text.
SNAPSHOT_ID = "3371aed8b1a4c972"
PROCESSING_ID = "health_snapshot/2+dcbb50a52f4e19d8"
CONTEXT_DIGEST = "f62744453d656057"
LEDGER_REMEDY = (
    "re-acquire at a higher sampling rate or set an explicit envelope_band_policy "
    "in the snapshot policy"
)


def _resolver_before_the_fix(fs, frequency_range=None):
    """The resolver as it was: the empty default band reached the band-edge check."""
    if frequency_range is None:
        band = (500.0, min(5000.0, fs / 2.0 - 1.0))
    else:
        band = (float(frequency_range[0]), float(frequency_range[1]))
    validate_bandpass_band(band[0], band[1], fs)
    return band


def _load_low_rate_measurement(signals: Path):
    """Declare the point, then load one 1000 Hz measurement bound to it."""
    values = 0.1 * np.random.default_rng((SEED, 1)).standard_normal(4000)
    write_signal(
        signals,
        "f1_low",
        values,
        {
            "sampling_rate": 1000.0,
            "signal_unit": "g",
            "measurement": {
                "asset_id": ASSET,
                "measurement_point_id": POINT,
                "acquired_at": ACQUIRED_AT,
                "direction": "horizontal",
                "rpm": RPM,
            },
        },
    )
    run(
        asset_tools.declare_measurement_point(
            AsyncMock(),
            asset_id=ASSET,
            measurement_point_id=POINT,
            bearing_id=BEARING,
            machine_group=2,
            support_type="rigid",
            nominal_rpm=RPM,
        )
    )
    return run(
        acquisition_tools.load_signal(
            AsyncMock(), filepath="f1_low.csv", overwrite=True
        )
    )


def _stored_snapshots() -> list[dict]:
    return [
        e["payload"]
        for e in LedgerStore(get_ledger_dir()).read(ASSET).events
        if e["event_type"] == "health_snapshot_computed"
    ]


class TestLedger:
    def test_a_new_snapshot_stores_the_refusal_under_unchanged_identifiers(self, box):
        loaded = _load_low_rate_measurement(box.signals)

        (snapshot,) = _stored_snapshots()
        assert snapshot["missing"]["bearing"] == {
            "reason": f"envelope band unusable at fs=1000 Hz: {DEFAULT_BAND_REFUSAL_AT_1000}",
            "remedy": LEDGER_REMEDY,
        }
        assert snapshot["snapshot_id"] == SNAPSHOT_ID
        assert snapshot["processing"]["processing_id"] == PROCESSING_ID
        assert snapshot["context_digest"] == CONTEXT_DIGEST
        assert loaded.measurement["snapshot_id"] == SNAPSHOT_ID

    def test_an_earlier_snapshot_keeps_its_text(self, box, monkeypatch):
        with monkeypatch.context() as before_the_fix:
            before_the_fix.setattr(
                snapshot_module, "resolve_envelope_band", _resolver_before_the_fix
            )
            first = _load_low_rate_measurement(box.signals)
        again = _load_low_rate_measurement(box.signals)

        (snapshot,) = _stored_snapshots()
        assert snapshot["missing"]["bearing"]["reason"] == (
            f"envelope band unusable at fs=1000 Hz: {EXPLICIT_REVERSED_AT_1000}"
        )
        assert snapshot["snapshot_id"] == SNAPSHOT_ID
        assert first.measurement["snapshot_id"] == again.measurement["snapshot_id"]

    def test_the_snapshot_engine_with_the_default_band(self):
        one = compute_health_snapshot(
            noise(1000.0), 1000.0, declaration=_DECLARATION, point=_POINT
        )
        assert one["missing"]["bearing"] == {
            "reason": f"envelope band unusable at fs=1000 Hz: {DEFAULT_BAND_REFUSAL_AT_1000}",
            "remedy": LEDGER_REMEDY,
        }
        assert one["processing"]["effective"]["envelope_band_hz"] is None

    def test_an_explicit_band_policy_keeps_its_message(self):
        one = compute_health_snapshot(
            noise(1000.0),
            1000.0,
            declaration=_DECLARATION,
            point=_POINT,
            policy=SnapshotPolicy(envelope_band_policy=(500.0, 5000.0)),
        )
        assert one["missing"]["bearing"]["reason"] == (
            f"envelope band unusable at fs=1000 Hz: {EXPLICIT_ABOVE_NYQUIST_AT_1000}"
        )


_DECLARATION = {
    "rpm": RPM,
    "direction": "horizontal",
    "signal_unit": "g",
    "sampling_rate": 1000.0,
    "raw_format": None,
}
_POINT = {"bearing_id": BEARING, "machine_group": 2, "support_type": "rigid"}


# ---------------------------------------------------------------------------
# Controls: each check rejects a mutated output
# ---------------------------------------------------------------------------


class TestControls:
    """Each mutation starts from an output the tools produced and changes one
    thing: a substituted word, the statement negated or dropped, or the
    statement moved. The check must reject it, for the reason named."""

    @pytest.fixture
    def produced(self, box):
        result, rendered = report(stored(box, 1000.0), bearing_id=BEARING)
        assert_absent_figure_statement(rendered, result["statements"], 1000.0)
        return result["statements"], rendered

    changed_statements = pytest.mark.parametrize(
        "replacement",
        [
            ABSENT_FIGURE_AT_1000.replace("fs=1000 Hz", "fs=1001 Hz"),
            ABSENT_FIGURE_AT_1000.replace("is empty", "is not empty"),
        ],
        ids=["substitute", "negate"],
    )

    @changed_statements
    def test_a_changed_statement(self, produced, replacement):
        statements, rendered = produced
        with pytest.raises(AssertionError, match="not returned once"):
            assert_absent_figure_statement(
                rendered.replace(ABSENT_FIGURE_AT_1000, replacement),
                [replacement if s == ABSENT_FIGURE_AT_1000 else s for s in statements],
                1000.0,
            )

    @changed_statements
    def test_a_statement_changed_in_the_rendering_only(self, produced, replacement):
        statements, rendered = produced
        with pytest.raises(AssertionError, match="not in its card"):
            assert_absent_figure_statement(
                rendered.replace(ABSENT_FIGURE_AT_1000, replacement), statements, 1000.0
            )

    def test_the_statement_left_out_of_the_returned_statements(self, produced):
        statements, rendered = produced
        with pytest.raises(AssertionError, match="not returned once"):
            assert_absent_figure_statement(
                rendered, [s for s in statements if s != ABSENT_FIGURE_AT_1000], 1000.0
            )

    def test_the_statement_moved_out_of_its_card(self, produced):
        statements, rendered = produced
        moved = rendered.replace(envelope_card(ABSENT_FIGURE_AT_1000), "").replace(
            "</body>", f"<p>{ABSENT_FIGURE_AT_1000}</p></body>"
        )
        with pytest.raises(AssertionError, match="not in its card"):
            assert_absent_figure_statement(moved, statements, 1000.0)

    def test_the_statement_rendered_a_second_time(self, produced):
        statements, rendered = produced
        doubled = rendered.replace("</body>", f"<p>{ABSENT_FIGURE_AT_1000}</p></body>")
        with pytest.raises(AssertionError, match="rendered elsewhere"):
            assert_absent_figure_statement(doubled, statements, 1000.0)

    def test_the_fixed_sentence_back_beside_the_statement(self, produced):
        statements, rendered = produced
        fixed = (
            "<p>This report carries no envelope spectrum figure, so the line tests "
            "can be checked only against the values in the table above.</p>"
        )
        with pytest.raises(AssertionError, match="renderer's fixed sentence"):
            assert_absent_figure_statement(
                rendered.replace("</body>", f"{fixed}</body>"), statements, 1000.0
            )

    def test_a_figure_drawn_beside_the_statement(self, produced):
        statements, rendered = produced
        with pytest.raises(AssertionError, match="figure is drawn"):
            assert_absent_figure_statement(
                rendered.replace("</body>", "<svg></svg></body>"), statements, 1000.0
            )

    @pytest.fixture
    def refusals(self, box):
        signal_id = stored(box, 1000.0)
        check = refusal(
            diagnostics_tools.check_bearing_faults(
                AsyncMock(), signal_id=signal_id, rpm=RPM, bearing_id=BEARING
            )
        )
        envelope = refusal(report_tools.generate_envelope_report(signal_id=signal_id))
        assert_refused_with(check, DEFAULT_BAND_REFUSAL_AT_1000)
        assert_refused_with(envelope, ENVELOPE_REPORT_REFUSAL_AT_1000)
        return {"check": check, "envelope": envelope}

    @pytest.mark.parametrize(
        "which, expected",
        [
            ("check", DEFAULT_BAND_REFUSAL_AT_1000),
            ("envelope", ENVELOPE_REPORT_REFUSAL_AT_1000),
        ],
        ids=["check", "envelope"],
    )
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda text: text.replace("fs=1000 Hz", "fs=1001 Hz"),
            lambda text: text.replace("is empty", "is not empty"),
            lambda text: " ".join(reversed(text.split(". "))),
        ],
        ids=["substitute", "negate", "relocate"],
    )
    def test_a_mutated_refusal(self, refusals, which, expected, mutate):
        mutated = type(refusals[which])(mutate(str(refusals[which])))
        with pytest.raises(AssertionError, match="refusal text differs"):
            assert_refused_with(mutated, expected)

    def test_a_refusal_on_the_wrong_surface(self, refusals):
        with pytest.raises(AssertionError, match="refusal text differs"):
            assert_refused_with(refusals["check"], ENVELOPE_REPORT_REFUSAL_AT_1000)

    def test_a_band_edge_message_is_rejected(self, refusals):
        with pytest.raises(AssertionError, match="band-edge message"):
            assert_refused_with(
                ValueError(EXPLICIT_REVERSED_AT_1000), EXPLICIT_REVERSED_AT_1000
            )
