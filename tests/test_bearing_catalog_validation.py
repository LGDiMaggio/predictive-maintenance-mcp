"""Validation tests for the verified bearing geometry catalog (audit item 2.5/2.6).

Guarantees that regressions cannot reintroduce the two failure modes found by
the 2026-07-10 audit:

1. Physically impossible geometries in the catalog (e.g. the old 6205 entry
   with pitch diameter 34.55 mm between a 25 mm bore and 52 mm OD — the real
   CWRU-documented value is 39.04 mm).
2. Fictitious reference frequencies injected into analysis outputs (the old
   hardcoded "BPFO ~81.13 Hz, example @ 1500 RPM" block in analyze_envelope).

Every catalog entry must be physically valid AND traceable to a public source
via its mandatory ``source`` field. Unverifiable entries are deleted, never
approximated.
"""

import json
import math
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance_mcp.document_reader import (
    calculate_bearing_frequencies,
    load_bearing_catalog,
    lookup_bearing_in_catalog,
)

CATALOG_PATH = (
    Path(__file__).parent.parent
    / "resources"
    / "bearing_catalogs"
    / "common_bearings_catalog.json"
)


def _catalog_entries() -> list[tuple[str, dict]]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    return sorted(catalog["bearings"].items())


ENTRIES = _catalog_entries()
ENTRY_IDS = [designation for designation, _ in ENTRIES]


# ---------------------------------------------------------------------------
# Geometry validity — runs against EVERY catalog entry
# ---------------------------------------------------------------------------


class TestCatalogGeometryValidity:
    """Physical validity checks for every entry in the catalog."""

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_bore_pitch_od_ordering(self, designation, entry):
        """bore < pitch diameter < outer diameter (strict)."""
        assert (
            entry["bore_mm"]
            < entry["pitch_diameter_mm"]
            < entry["outer_diameter_mm"]
        ), f"{designation}: pitch diameter must lie between bore and OD"

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_pitch_near_mean_of_bore_and_od(self, designation, entry):
        """Pitch diameter must be close to (bore + OD) / 2.

        Heuristic sanity check (8% tolerance) that catches the historical
        bogus 6205 entry: pitch 34.55 mm with bore 25 / OD 52 was 10.3% off
        the 38.5 mm midpoint; the real value 39.04 mm is 1.4% off.
        """
        midpoint = (entry["bore_mm"] + entry["outer_diameter_mm"]) / 2.0
        deviation = abs(entry["pitch_diameter_mm"] - midpoint) / midpoint
        assert deviation < 0.08, (
            f"{designation}: pitch {entry['pitch_diameter_mm']} mm deviates "
            f"{deviation:.1%} from the bore/OD midpoint {midpoint:.2f} mm"
        )

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_rolling_elements_fit_radially(self, designation, entry):
        """Balls must fit between bore and OD around the pitch circle."""
        pitch = entry["pitch_diameter_mm"]
        bd = entry["ball_diameter_mm"]
        assert pitch - bd > entry["bore_mm"], (
            f"{designation}: balls would intersect the bore"
        )
        assert pitch + bd < entry["outer_diameter_mm"], (
            f"{designation}: balls would intersect the outer diameter"
        )

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_rolling_elements_fit_circumferentially(self, designation, entry):
        """num_balls * ball diameter must fit on the pitch circumference."""
        circumference = math.pi * entry["pitch_diameter_mm"]
        total_ball_span = entry["num_balls"] * entry["ball_diameter_mm"]
        assert total_ball_span < circumference, (
            f"{designation}: {entry['num_balls']} balls of "
            f"{entry['ball_diameter_mm']} mm cannot fit on a "
            f"{entry['pitch_diameter_mm']} mm pitch circle"
        )

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_minimum_rolling_elements(self, designation, entry):
        """Rolling bearings have at least 6 rolling elements."""
        assert entry["num_balls"] >= 6

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_source_field_mandatory(self, designation, entry):
        """Every entry MUST carry a non-empty source citation."""
        assert isinstance(entry.get("source"), str)
        assert entry["source"].strip(), f"{designation}: empty source field"

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_kinematic_sanity_bpfo_plus_bpfi(self, designation, entry):
        """BPFO/fr + BPFI/fr must equal the number of rolling elements.

        This identity holds exactly for the standard kinematic formulas
        (Randall & Antoni 2011); a small tolerance absorbs output rounding.
        """
        rpm = 6000.0  # fr = 100 Hz, so rounding to 0.01 Hz is negligible
        fr = rpm / 60.0
        freqs = calculate_bearing_frequencies(
            num_balls=entry["num_balls"],
            ball_diameter_mm=entry["ball_diameter_mm"],
            pitch_diameter_mm=entry["pitch_diameter_mm"],
            contact_angle_deg=entry["contact_angle_deg"],
            shaft_speed_rpm=rpm,
        )
        factor_sum = (freqs["BPFO"] + freqs["BPFI"]) / fr
        assert factor_sum == pytest.approx(entry["num_balls"], abs=0.01)

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_matches_published_frequency_factors(self, designation, entry):
        """Where the source publishes defect frequency factors, the formula
        applied to the stored geometry must reproduce them (0.1% tolerance)."""
        reference = entry.get("reference_frequency_factors")
        if not reference:
            pytest.skip("no published frequency factors for this entry")
        rpm = 6000.0
        fr = rpm / 60.0
        freqs = calculate_bearing_frequencies(
            num_balls=entry["num_balls"],
            ball_diameter_mm=entry["ball_diameter_mm"],
            pitch_diameter_mm=entry["pitch_diameter_mm"],
            contact_angle_deg=entry["contact_angle_deg"],
            shaft_speed_rpm=rpm,
        )
        for name, published_factor in reference.items():
            computed_factor = freqs[name] / fr
            assert computed_factor == pytest.approx(published_factor, rel=0.001), (
                f"{designation}/{name}: computed {computed_factor:.4f} "
                f"vs published {published_factor}"
            )

    @pytest.mark.parametrize("designation,entry", ENTRIES, ids=ENTRY_IDS)
    def test_designation_matches_key(self, designation, entry):
        assert entry["designation"] == designation


# ---------------------------------------------------------------------------
# 6205 reference values (CWRU Bearing Data Center)
# ---------------------------------------------------------------------------


class TestCwru6205ReferenceValues:
    """The 6205 entry must reproduce the CWRU-published defect frequencies."""

    def test_bpfo_at_1797_rpm(self):
        entry = lookup_bearing_in_catalog("6205")
        assert entry is not None
        freqs = calculate_bearing_frequencies(
            num_balls=entry["num_balls"],
            ball_diameter_mm=entry["ball_diameter_mm"],
            pitch_diameter_mm=entry["pitch_diameter_mm"],
            contact_angle_deg=entry["contact_angle_deg"],
            shaft_speed_rpm=1797.0,
        )
        # CWRU: BPFO = 3.5848 x shaft speed -> 107.36 Hz at 1797 RPM
        assert freqs["BPFO"] == pytest.approx(107.4, rel=0.005)
        fr = 1797.0 / 60.0
        assert freqs["BPFO"] / fr == pytest.approx(3.585, abs=0.002)

    def test_bpfo_plus_bpfi_equals_num_balls(self):
        entry = lookup_bearing_in_catalog("6205")
        freqs = calculate_bearing_frequencies(
            num_balls=entry["num_balls"],
            ball_diameter_mm=entry["ball_diameter_mm"],
            pitch_diameter_mm=entry["pitch_diameter_mm"],
            contact_angle_deg=entry["contact_angle_deg"],
            shaft_speed_rpm=1797.0,
        )
        fr = 1797.0 / 60.0
        assert (freqs["BPFO"] + freqs["BPFI"]) / fr == pytest.approx(
            entry["num_balls"], abs=0.01
        )


# ---------------------------------------------------------------------------
# Single source of truth — no duplicated in-memory geometry
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """lookup_bearing_in_catalog must serve data from the JSON file only."""

    def test_lookup_returns_json_values(self):
        with open(CATALOG_PATH, encoding="utf-8") as f:
            expected = json.load(f)["bearings"]["6205"]
        result = lookup_bearing_in_catalog("6205")
        assert result is not None
        for key, value in expected.items():
            assert result[key] == value, f"drift on field {key!r}"

    def test_fallback_path_reads_same_json(self, monkeypatch, tmp_path):
        """With RESOURCES_DIR pointing at an empty directory, the package-
        relative fallback must still serve the SAME JSON file — never a
        duplicated in-memory copy."""
        import predictive_maintenance_mcp.document_reader as dr

        monkeypatch.setattr(dr, "RESOURCES_DIR", tmp_path)  # no catalog here
        result = dr.lookup_bearing_in_catalog("6205")
        assert result is not None
        with open(CATALOG_PATH, encoding="utf-8") as f:
            expected = json.load(f)["bearings"]["6205"]
        for key, value in expected.items():
            assert result[key] == value, f"fallback drift on field {key!r}"

    def test_load_bearing_catalog_structure(self):
        catalog = load_bearing_catalog()
        assert "bearings" in catalog
        assert len(catalog["bearings"]) >= 3
        assert "6205" in catalog["bearings"]


# ---------------------------------------------------------------------------
# Not-found path — structured payload, no invented geometry
# ---------------------------------------------------------------------------


@pytest.fixture
def diag_tools():
    from mcp.server.mcpserver import MCPServer

    from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register

    server = MCPServer("test-catalog-validation")
    register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


class TestBearingNotFound:
    """Unknown bearings produce a structured not-found result, never geometry."""

    def test_lookup_returns_none(self):
        assert lookup_bearing_in_catalog("ZZZ_NOT_A_BEARING_999") is None

    @pytest.mark.asyncio
    async def test_search_tool_returns_typed_miss(self, diag_tools):
        """U6 error contract: a catalog miss is a legitimate negative
        outcome — a TYPED result (status + suggestion), never a dict with
        an 'error' key returned as success."""
        from predictive_maintenance_mcp.models import BearingCatalogMiss

        result = await diag_tools["search_bearing_catalog"](
            bearing_id="99999",
            ctx=AsyncMock(),
        )
        assert isinstance(result, BearingCatalogMiss)
        assert result.status == "not_found"
        assert result.bearing_id == "99999"
        assert result.suggestion
        # No invented geometry in the payload
        dumped = result.model_dump()
        assert "error" not in dumped
        for key in ("num_balls", "ball_diameter_mm", "pitch_diameter_mm"):
            assert key not in dumped
        # The payload lists the real verified designations
        assert "6205" in result.catalog_contains


# ---------------------------------------------------------------------------
# Envelope analysis — no fictitious reference frequencies in output
# ---------------------------------------------------------------------------


@pytest.fixture
def analysis_tools_env(tmp_path, monkeypatch):
    """MCPServer with analysis tools registered + a synthetic signal on disk."""
    from mcp.server.mcpserver import MCPServer

    from predictive_maintenance_mcp.mcp_tools.analysis_tools import register

    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)

    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    noise = 0.05 * np.random.default_rng(0).standard_normal(fs)
    sig = np.sin(2 * np.pi * 50 * t) + noise
    pd.DataFrame(sig).to_csv(signals_dir / "env_test.csv", index=False, header=False)
    with open(signals_dir / "env_test_metadata.json", "w") as f:
        json.dump({"sampling_rate": fs, "signal_unit": "g"}, f)

    monkeypatch.setattr("predictive_maintenance_mcp.config.DATA_DIR", signals_dir)
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", signals_dir
    )
    monkeypatch.setattr(
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR", signals_dir
    )

    server = MCPServer("test-envelope-honesty")
    register(server)
    tools = {t.name: t.fn for t in server._tool_manager._tools.values()}

    from predictive_maintenance_mcp.signal_acquisition.repository import (
        get_repository,
    )

    repo = get_repository()
    repo.load_signal("env_test.csv", overwrite=True)  # metadata: fs + 'g'
    yield tools
    repo.clear_signal("env_test")


class TestNoFictitiousReferenceInEnvelopeOutput:
    """analyze_envelope must not inject the old hardcoded BPFO reference."""

    @pytest.mark.asyncio
    async def test_diagnosis_contains_no_hardcoded_reference(self, analysis_tools_env):
        ctx = AsyncMock()
        result = await analysis_tools_env["analyze_envelope"](
            ctx=ctx, signal_id="env_test"
        )
        assert result.diagnosis is not None
        assert "81.13" not in result.diagnosis
        assert "example @ 1500 RPM" not in result.diagnosis

    @pytest.mark.asyncio
    async def test_diagnosis_points_to_real_frequency_sources(self, analysis_tools_env):
        """Instead of invented numbers, the output must direct the user to
        compute frequencies for their actual bearing (post-U9 the unified
        check_bearing_faults covers catalog, explicit frequencies, and
        explicit geometry routes)."""
        ctx = AsyncMock()
        result = await analysis_tools_env["analyze_envelope"](
            ctx=ctx, signal_id="env_test"
        )
        assert "check_bearing_faults" in result.diagnosis
        assert "No reference bearing frequencies are assumed" in result.diagnosis
