"""Pin the README Quick Start chain to what the server returns.

README.md, "4. First diagnosis", documents one prompt on the bundled
``real_train/OuterRaceFault_1.csv`` at 1500 rpm and then describes in words
what comes back: two of the four declared fault frequencies graded as high
evidence, the outer race ranked most likely, the inner race graded high
because its second harmonic shares a peak with the third outer-race
harmonic, ball and cage unmatched, an ISO 20816-3 zone with the assumed
machine group and support type, and an HTML report under ``reports/``.

The chain behind that paragraph (``load_signal`` -> companion frequencies
-> ``check_bearing_faults`` -> ``assess_severity`` ->
``generate_envelope_report``) is deterministic, so this module runs it
against the checkout and asserts each claim of the paragraph. A change in
the envelope engine, the matching tolerance, the severity table or the
companion metadata that alters the outcome fails here instead of leaving
the README describing a result the server no longer returns. The last test
holds the paragraph itself to the same claims, so a rewrite of the prose
and a change of the engine meet in the same place.

Reports are redirected to a temporary directory; the checkout's
``reports/`` is never written.
"""

import asyncio
import re
from pathlib import Path

import pytest

from predictive_maintenance_mcp.mcp_tools.acquisition_tools import (
    get_signal_info,
    load_signal,
)
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import (
    assess_severity,
    check_bearing_faults,
)
from predictive_maintenance_mcp.mcp_tools.report_tools import (
    generate_envelope_report,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

#: The signal and shaft speed the README prompt names.
SAMPLE = "real_train/OuterRaceFault_1.csv"
RPM = 1500.0
DECLARED = ("BPFO", "BPFI", "BSF", "FTF")

#: What the README paragraph says comes back, as the server reports it.
EXPECTED_EVIDENCE = {"BPFO": "high", "BPFI": "high", "BSF": "low", "FTF": "none"}
EXPECTED_DETECTED = {"BPFO": True, "BPFI": True, "BSF": False, "FTF": False}


async def _run_chain() -> dict:
    info = await load_signal(None, SAMPLE, overwrite=True)
    meta = await get_signal_info(None, info.signal_id)
    companion = meta.source_metadata or {}
    frequencies = {k: float(companion[k]) for k in DECLARED if k in companion}
    faults = await check_bearing_faults(
        None, info.signal_id, rpm=RPM, frequencies=frequencies
    )
    severity = await assess_severity(None, signal_id=info.signal_id, rpm=RPM)
    report = await generate_envelope_report(info.signal_id)
    return {
        "frequencies": frequencies,
        "faults": faults,
        "severity": severity,
        "report": report,
    }


@pytest.fixture
def chain(tmp_path, monkeypatch) -> dict:
    """Run the documented chain once, with reports under ``tmp_path``."""
    sample = REPO_ROOT / "data" / "signals" / SAMPLE
    if not sample.exists():
        pytest.skip(f"Sample data not found: {sample}")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    for target in (
        "predictive_maintenance_mcp.report_generator.REPORTS_DIR",
        "predictive_maintenance_mcp.mcp_tools.report_tools.REPORTS_DIR",
    ):
        monkeypatch.setattr(target, reports_dir)
    repo = get_repository()
    repo.clear_all()
    try:
        result = asyncio.run(_run_chain())
        result["reports_dir"] = reports_dir
        yield result
    finally:
        repo.clear_all()


def _check(faults, label):
    return next(c for c in faults.fault_checks if c.fault_type == label)


def _harmonic_hz(check, order: float):
    return next(
        h["detected_hz"] for h in check.harmonics_detected if h["harmonic"] == order
    )


class TestBearingEvidence:
    def test_companion_declares_the_four_frequencies(self, chain):
        assert set(chain["frequencies"]) == set(DECLARED)

    def test_two_of_four_declared_frequencies_graded_high(self, chain):
        faults = chain["faults"]
        evidence = {c.fault_type: c.evidence_strength for c in faults.fault_checks}
        detected = {c.fault_type: c.detected for c in faults.fault_checks}
        assert evidence == EXPECTED_EVIDENCE
        assert detected == EXPECTED_DETECTED

    def test_outer_race_ranked_most_likely(self, chain):
        faults = chain["faults"]
        assert faults.most_likely_fault == "BPFO"
        assert faults.most_likely_fault_canonical == "outer_race"

    def test_inner_race_second_harmonic_shares_the_third_outer_race_peak(self, chain):
        faults = chain["faults"]
        bpfo_third = _harmonic_hz(_check(faults, "BPFO"), 3.0)
        bpfi_second = _harmonic_hz(_check(faults, "BPFI"), 2.0)
        assert bpfo_third == pytest.approx(bpfi_second, abs=0.01)


class TestSeverityAndReport:
    def test_iso_zone_with_the_assumed_group_and_support(self, chain):
        severity = chain["severity"]
        assert severity.status == "assessed"
        assert severity.zone == "B"
        assert severity.machine_group == 2
        assert severity.support_type == "rigid"
        assert severity.unit_conversion_performed is True
        assert severity.original_unit == "g"

    def test_envelope_report_saved_with_the_declared_frequencies_marked(self, chain):
        report = chain["report"]
        path = Path(report["file_path"])
        assert path.exists()
        assert path.suffix == ".html"
        assert path.parent.resolve() == chain["reports_dir"].resolve()
        matched = " ".join(report["bearing_matches"])
        assert "BPFO" in matched and "BPFI" in matched
        assert "BSF" not in matched and "FTF" not in matched


class TestReadmeParagraph:
    """The prose must keep claiming exactly what the chain above pins."""

    @pytest.fixture(scope="class")
    def quick_start(self) -> str:
        text = README.read_text(encoding="utf-8")
        start = text.index("**4. First diagnosis.**")
        end = text.index("**Use your own data.**", start)
        return text[start:end]

    def test_prompt_names_the_sample_and_the_shaft_speed(self, quick_start):
        prompt = re.search(r"^> \*\"(.+?)\"\*$", quick_start, flags=re.M)
        assert prompt, "the First diagnosis prompt must be a quoted line"
        assert SAMPLE in prompt.group(1)
        assert f"{RPM:.0f} rpm" in prompt.group(1)
        assert "envelope report" in prompt.group(1)

    @pytest.mark.parametrize(
        "claim",
        [
            "two of the four declared frequencies come back as high evidence",
            "the outer race ranked most likely",
            "the inner race graded high as well",
            "its second harmonic shares a peak with the third outer-race harmonic",
            "the ball and cage frequencies stay unmatched",
            "the ISO 20816-3 severity zone",
            "the machine group and support type the assessment assumed",
            "an HTML report in `reports/`",
        ],
    )
    def test_paragraph_states_each_pinned_claim(self, quick_start, claim):
        assert claim in quick_start
