"""Pin the README Quick Start chain to what the server returns.

README.md, "4. First diagnosis", documents one prompt on the bundled
``real_train/OuterRaceFault_1.csv`` at 1500 rpm and then describes in words
what comes back: the outer-race and inner-race lines both stand above the
local noise floor, each with a further line of its own, so the check reports
several types with both levels and singles out neither, the ball and cage
lines do not stand above the floor, an ISO 20816-3 zone with the assumed
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
EXPECTED_STATES = {
    "BPFO": "supported",
    "BPFI": "supported",
    "BSF": "absent",
    "FTF": "absent",
}
#: The two candidates and their levels over the local noise floor (dB).
EXPECTED_CANDIDATES = [("BPFO", 35.2), ("BPFI", 16.1)]


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


class TestBearingEvidence:
    def test_companion_declares_the_four_frequencies(self, chain):
        assert set(chain["frequencies"]) == set(DECLARED)

    def test_outer_and_inner_race_stand_above_the_floor_ball_and_cage_do_not(
        self, chain
    ):
        faults = chain["faults"]
        states = {c.fault_type: c.state for c in faults.fault_checks}
        assert states == EXPECTED_STATES

    def test_each_race_line_has_a_further_line_of_its_own(self, chain):
        for check in chain["faults"].fault_checks:
            if check.state == "supported":
                assert any(c.counted for c in check.corroboration), check.fault_type
                assert check.shared_with == [], check.fault_type

    def test_several_types_with_both_levels_and_none_singled_out(self, chain):
        faults = chain["faults"]
        assert faults.outcome == "several_types"
        assert [
            (c.fault_type, round(c.level_db, 1)) for c in faults.candidates
        ] == EXPECTED_CANDIDATES
        assert faults.claim.level == "qualified"
        assert faults.claim.conditions == ["several_types"]
        assert faults.most_likely_fault is None
        assert faults.most_likely_fault_canonical is None
        assert "most likely" not in faults.overall_assessment.lower()


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
        """Since T1 PR C the report marks the lines of the same line test
        check_bearing_faults runs, on the same band, and states its claim. The
        sample declares no measurement rpm, so the report's test runs without
        the sideband and shaft-order tests, every label says so, and its
        several-types statement is the bearing check's with one sentence added:
        that the shaft-order test did not run for its two lines."""
        report = chain["report"]
        path = Path(report["file_path"])
        assert path.exists()
        assert path.suffix == ".html"
        assert path.parent.resolve() == chain["reports_dir"].resolve()
        labels = report["bearing_matches"]
        assert [label.split(" ", 1)[0] for label in labels] == ["BPFO", "BPFI"]
        for label in labels:
            assert label.endswith(
                "shaft speed was not available to this check, so the shaft-rate "
                "sideband test and the shaft-order test did not run"
            )
        closing = " The line test assumes"
        assert report["bearing_assessment"] == chain[
            "faults"
        ].overall_assessment.replace(
            closing,
            " Shaft speed was not available to this check, so the shaft-order test "
            "did not run and an overlap of the outer-race line (BPFO) or the "
            "inner-race line (BPFI) with a shaft harmonic cannot be excluded."
            + closing,
        )
        assert chain["faults"].claim.conditions == ["several_types"]


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
            "the outer-race and inner-race lines both stand above the local noise "
            "floor",
            "each with a further line of its own",
            "the evidence of this single spectrum is consistent with more than one "
            "fault type",
            "lists both with their levels without singling either out",
            "neither the ball nor the cage line stands above the floor",
            "the ISO 20816-3 severity zone",
            "the machine group and support type the assessment assumed",
            "an HTML report in `reports/`",
        ],
    )
    def test_paragraph_states_each_pinned_claim(self, quick_start, claim):
        assert claim in quick_start

    def test_paragraph_names_no_fault_as_most_likely(self, quick_start):
        assert "most likely" not in quick_start.lower()
