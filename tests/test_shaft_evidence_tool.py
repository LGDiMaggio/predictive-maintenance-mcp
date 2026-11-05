"""The assess_shaft_fault_evidence MCP tool: resolution of the declared context.

Speed, directions, machine group and support type come from the call, else
from what the stored signals and their measurement point declare; never from
a default. Refusals are server sentences with a remedy.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_MEASUREMENT_POINT_DECLARED,
    LedgerStore,
    make_event,
)
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register
from predictive_maintenance_mcp.models import ShaftFaultEvidenceResult
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

FS = 10000
ASSET, POINT = "P-201", "fan_de"


def _write(data_dir: Path, name: str, signal, *, unit="mm/s", measurement=None, **meta):
    pd.DataFrame(signal).to_csv(data_dir / f"{name}.csv", index=False, header=False)
    companion = {"sampling_rate": FS, **meta}
    if unit is not None:
        companion["signal_unit"] = unit
    if measurement is not None:
        companion["measurement"] = measurement
    (data_dir / f"{name}_metadata.json").write_text(json.dumps(companion))


def _sine(amplitudes, f1=24.7, seconds=6.0, seed=0):
    t = np.arange(int(seconds * FS)) / FS
    x = sum(a * np.sin(2 * np.pi * k * f1 * t) for k, a in amplitudes)
    return x + 0.02 * np.random.default_rng(seed).standard_normal(t.size)


def _measurement(direction, rpm=None):
    m = {
        "asset_id": ASSET,
        "measurement_point_id": POINT,
        "acquired_at": "2026-11-05T10:00:00Z",
        "direction": direction,
    }
    if rpm is not None:
        m["rpm"] = rpm
    return m


@pytest.fixture
def tools():
    server = MCPServer("test-shaft-evidence")
    register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


@pytest.fixture
def ctx():
    return AsyncMock()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)
    for target in (
        "predictive_maintenance_mcp.config.DATA_DIR",
        "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR",
        "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR",
    ):
        monkeypatch.setattr(target, signals_dir)
    _write(
        signals_dir,
        "fan_h",
        _sine([(1, 12.0), (2, 0.5)], seed=1),
        measurement=_measurement("horizontal", 1500),
    )
    _write(
        signals_dir,
        "fan_v",
        _sine([(1, 10.0), (2, 0.4)], seed=2),
        measurement=_measurement("vertical", 1500),
    )
    _write(
        signals_dir,
        "fan_a",
        _sine([(1, 3.0), (2, 6.0)], seed=3),
        measurement=_measurement("axial", 1500),
    )
    _write(
        signals_dir,
        "other_speed",
        _sine([(1, 5.0)], seed=4),
        measurement=_measurement("horizontal", 1800),
    )
    _write(signals_dir, "anonymous", _sine([(1, 5.0)], seed=5))
    _write(signals_dir, "no_unit", _sine([(1, 5.0)], seed=6), unit=None)
    repo = get_repository()
    repo.clear_all()
    for name in ("fan_h", "fan_v", "fan_a", "other_speed", "anonymous", "no_unit"):
        repo.load_signal(f"{name}.csv")
    yield signals_dir
    repo.clear_all()


def _declare_point(**overrides):
    payload = {
        "measurement_point_id": POINT,
        "declaration_version": 1,
        "bearing_id": None,
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "mm/s",
        "expected_sensor_id": None,
        "expected_direction": None,
        "nominal_rpm": 1500.0,
        "declared_by": "test",
        "note": None,
        "changed": [],
    }
    payload.update(overrides)
    LedgerStore(get_ledger_dir()).append(
        ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload)
    )


def _family(result, family):
    return next(h for h in result.hypotheses if h.family == family)


@pytest.mark.asyncio
class TestResolution:
    async def test_declared_speed_and_directions_come_from_the_measurements(
        self, tools, ctx, data_dir
    ):
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h", "fan_v", "fan_a"]
        )
        assert isinstance(result, ShaftFaultEvidenceResult)
        assert result.rpm == 1500.0
        assert result.parameter_sources == {
            "rpm": "measurement",
            "machine_group": "none",
            "support_type": "none",
            "machine_power_kw": "none",
        }
        assert [c.direction for c in result.channels] == [
            "horizontal",
            "vertical",
            "axial",
        ]
        # No machine class declared anywhere: severity not evaluated, nothing consistent.
        assert result.level_context.status == "not_assessed"
        assert result.level_context.reason == "level_class_undeclared"
        assert all(h.status != "consistent" for h in result.hypotheses)
        assert result.detail == "summary"

    async def test_the_point_declaration_supplies_the_machine_class(
        self, tools, ctx, data_dir
    ):
        _declare_point()
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h", "fan_v", "fan_a"]
        )
        assert result.level_context.status == "assessed"
        assert (
            result.level_context.machine_group,
            result.level_context.support_type,
        ) == (2, "rigid")
        assert result.level_context.zone == "D"
        assert _family(result, "unbalance").status == "inconclusive"
        assert _family(result, "unbalance").reasons == ["attribution_not_validated"]
        assert _family(result, "unbalance").pattern_observed is True
        assert _family(result, "misalignment").status == "inconclusive"
        assert _family(result, "misalignment").reasons == [
            "single_record_insufficient",
            "attribution_not_validated",
        ]

    async def test_explicit_arguments_win(self, tools, ctx, data_dir):
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx,
            signal_ids=["anonymous"],
            rpm=1500.0,
            directions=["vertical"],
            machine_group=1,
            support_type="flexible",
            speed_tolerance_pct=2.0,
            max_order=6,
            line_frequency_hz=50.0,
            detail="full",
        )
        assert result.parameter_sources["rpm"] == "explicit"
        assert result.channels[0].direction_source == "explicit"
        assert result.channels[0].direction == "vertical"
        assert result.channels[0].speed.tolerance_pct == 2.0
        assert result.channels[0].lines[-1].order == 6.0
        assert result.channels[0].line_frequency_hz == 50.0
        assert result.level_context.machine_group == 1

    async def test_result_carries_policy_sources_and_the_experimental_flag(
        self, tools, ctx, data_dir
    ):
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h"]
        )
        assert result.experimental is True
        assert result.order_line_policy["method"] == "os_cfar_shaft_order_line_test"
        assert {s.key for s in result.sources} >= {"S1", "S3", "S5", "S6"}
        assert "No outcome is a diagnosis" in result.summary


@pytest.mark.asyncio
class TestRefusals:
    async def test_no_declared_speed_is_refused(self, tools, ctx, data_dir):
        with pytest.raises(ValueError, match="Cannot resolve the operating speed"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["anonymous"]
            )

    async def test_channels_declaring_different_speeds_are_refused(
        self, tools, ctx, data_dir
    ):
        with pytest.raises(ValueError, match="declare different speeds"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["fan_h", "other_speed"]
            )

    @pytest.mark.parametrize(
        ("ids", "match"),
        [
            ([], "1 to 6"),
            (["fan_h"] * 2, "different signal"),
            ([f"s{n}" for n in range(7)], "1 to 6"),
        ],
    )
    async def test_bad_signal_id_lists_are_refused(
        self, tools, ctx, data_dir, ids, match
    ):
        with pytest.raises(ValueError, match=match):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=ids, rpm=1500.0
            )

    async def test_directions_must_match_the_signals(self, tools, ctx, data_dir):
        with pytest.raises(ValueError, match="one direction"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["fan_h", "fan_v"], directions=["axial"]
            )

    async def test_an_invalid_explicit_rpm_is_refused_before_any_lookup(
        self, tools, ctx, data_dir
    ):
        with pytest.raises(ValueError, match="rpm must be positive"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["missing"], rpm=-1
            )

    async def test_an_undeclared_unit_is_not_guessed(self, tools, ctx, data_dir):
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["no_unit"], rpm=1500.0
        )
        assert result.channels[0].unit.signal_unit is None
        assert _family(result, "unbalance").reasons == ["unit_undeclared"]


@pytest.mark.asyncio
class TestProvenance:
    async def test_point_power_below_scope_is_used_and_reported(
        self, tools, ctx, data_dir
    ):
        _declare_point(machine_power_kw=7.5)
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h"]
        )
        assert result.parameter_sources["machine_power_kw"] == "point"
        assert result.level_context.reason == "level_outside_scope"

    async def test_points_declaring_different_classes_are_refused(
        self, tools, ctx, data_dir
    ):
        _declare_point(machine_group=2)
        other = dict(_measurement("vertical", 1500), measurement_point_id="fan_nde")
        _write(data_dir, "fan_nde_v", _sine([(1, 3.0)], seed=9), measurement=other)
        get_repository().load_signal("fan_nde_v.csv")
        payload_point = "fan_nde"
        LedgerStore(get_ledger_dir()).append(
            ASSET,
            make_event(
                EVENT_MEASUREMENT_POINT_DECLARED,
                ASSET,
                {
                    "measurement_point_id": payload_point,
                    "declaration_version": 1,
                    "bearing_id": None,
                    "fault_orders": None,
                    "machine_group": 1,
                    "support_type": "rigid",
                    "machine_power_kw": None,
                    "expected_signal_unit": "mm/s",
                    "expected_sensor_id": None,
                    "expected_direction": None,
                    "nominal_rpm": 1500.0,
                    "declared_by": "test",
                    "note": None,
                    "changed": [],
                },
            ),
        )
        with pytest.raises(ValueError, match="different machine groups"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["fan_h", "fan_nde_v"]
            )

    async def test_output_numbers_are_rounded_to_six_digits(self, tools, ctx, data_dir):
        result = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h"]
        )
        shaft = result.channels[0].speed.shaft_hz
        assert shaft == float(f"{shaft:.6g}")


@pytest.mark.asyncio
class TestDetail:
    async def test_summary_is_the_default_and_smaller(self, tools, ctx, data_dir):
        summary = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h", "fan_v", "fan_a"]
        )
        full = await tools["assess_shaft_fault_evidence"](
            ctx=ctx, signal_ids=["fan_h", "fan_v", "fan_a"], detail="full"
        )
        assert summary.detail == "summary" and full.detail == "full"
        size_summary, size_full = len(summary.model_dump_json()), len(
            full.model_dump_json()
        )
        assert size_summary < 0.75 * size_full and size_summary < 25_000
        assert [h.model_dump() for h in summary.hypotheses] == [
            h.model_dump() for h in full.hypotheses
        ]
        assert all(len(c.lines) == 20 for c in full.channels)
        assert all(all(line.significant for line in c.lines) for c in summary.channels)

    async def test_an_unknown_detail_is_refused_before_any_lookup(
        self, tools, ctx, data_dir
    ):
        with pytest.raises(ValueError, match="detail must be"):
            await tools["assess_shaft_fault_evidence"](
                ctx=ctx, signal_ids=["missing"], rpm=1500.0, detail="brief"
            )
