"""Tests for the asset health ledger MCP tools (U7).

The four tools are driven through a test ``MCPServer`` together with
``load_signal`` (the only way measurements enter the ledger), with
``ctx=AsyncMock()``, the shared ``sandbox_data_dir`` and the autouse ledger
directory, so every scenario of the plan's U7 unit runs end to end on real
files and a real ``LedgerStore``:

- point declaration versions, the diff, the stale-snapshot count and the
  identical re-declaration that appends nothing;
- the index and the history (spy: one ledger read), newest first,
  truncation, the index cap;
- AE9 (structure of the change assessment on the synthetic sequence) and
  AE7 (a declared baseline cites its declarer);
- every misuse rail (foreign ids, non-comparable members, empty declarer,
  free-text limits, a point without its asset, traversal and reserved
  ids, limits) and the typed misses;
- re-processing behind ``reprocess=True`` (nothing to do, then a bounded
  run with the next call named).
"""

import inspect
import json
import typing
import zlib
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from test_documented_calls import (
    build_endpoints,
    extract_calls,
    validate_documented_calls,
)
from _synthetic_sequence import (
    ASSET_ID as SEQUENCE_ASSET,
    BEARING_ID,
    NOMINAL_RPM,
    POINT_ID as SEQUENCE_POINT,
    SEQUENCE_LENGTH,
    build_measurement_sequence,
)
from predictive_maintenance_mcp.asset_ledger import service as ledger_service
from predictive_maintenance_mcp.asset_ledger import snapshot as snapshot_module
from predictive_maintenance_mcp.asset_ledger.assessment import (
    MIN_REFERENCE_MEASUREMENTS,
)
from predictive_maintenance_mcp.asset_ledger.service import MAX_REPROCESS_PER_CALL
from predictive_maintenance_mcp.asset_ledger.snapshot import processing_id
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_BASELINE_DECLARED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
)
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools import acquisition_tools, asset_tools
from predictive_maintenance_mcp.models import (
    AssetChangeAssessment,
    AssetHistoryResult,
    BaselineDeclarationResult,
    MeasurementPointDeclarationResult,
)
from predictive_maintenance_mcp.signal_acquisition import repository as repo_module
from predictive_maintenance_mcp.signal_acquisition.measurement import (
    MAX_FREE_TEXT_CHARS,
    VALID_DIRECTIONS,
)
from predictive_maintenance_mcp.signal_acquisition.repository import (
    VALID_SIGNAL_UNITS,
)

ASSET = "P-101"
OTHER_ASSET = "P-102"
POINT = "motor_de_h"
OTHER_POINT = "motor_nde_v"
FS = 10000
TOOL_NAMES = (
    "declare_measurement_point",
    "declare_healthy_baseline",
    "get_asset_history",
    "assess_asset_change",
)

#: Ids that must be refused by every tool with a closed oracle.
BAD_IDS = ["../evil", "NUL", "COM1.jsonl", "a/b", "P 101", "_index", ".hidden"]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tools():
    """Asset tools plus load_signal, mapped by name."""
    server = MCPServer("test-asset-tools")
    asset_tools.register(server)
    acquisition_tools.register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


@pytest.fixture
def ctx():
    return AsyncMock()


@pytest.fixture
def clean_repo():
    repo = repo_module.get_repository()
    repo.clear_all()
    yield repo
    repo.clear_all()


@pytest.fixture
def sandbox(sandbox_data_dir, ledger_dir, clean_repo):
    """Empty data directory, empty ledger directory, empty repository."""
    return sandbox_data_dir


def _noise(seed: int, n: int = FS) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(n)


def _acquired(index: int) -> str:
    """Weekly acquisitions from 2026-01-05 09:00 +01:00 (index is 1-based)."""
    day = 5 + 7 * (index - 1)
    month = 1
    while day > 28:
        day -= 28
        month += 1
    return f"2026-{month:02d}-{day:02d}T09:00:00+01:00"


def _companion(
    *,
    asset: str = ASSET,
    point: str = POINT,
    index: int = 1,
    rpm=1800,
    direction="horizontal",
    sensor_id="ACC01",
    signal_unit="g",
    sampling_rate=FS,
) -> dict:
    measurement = {
        "asset_id": asset,
        "measurement_point_id": point,
        "acquired_at": _acquired(index),
    }
    if rpm is not None:
        measurement["rpm"] = rpm
    if direction is not None:
        measurement["direction"] = direction
    if sensor_id is not None:
        measurement["sensor_id"] = sensor_id
    payload = {"measurement": measurement}
    if sampling_rate is not None:
        payload["sampling_rate"] = sampling_rate
    if signal_unit is not None:
        payload["signal_unit"] = signal_unit
    return payload


def _place(data_dir: Path, name: str, values: np.ndarray, companion: dict) -> str:
    """Write ``<name>`` (data_dir-relative) with its companion; return the name."""
    path = data_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        np.savetxt(fh, values, fmt="%.8f")
    (path.parent / f"{path.stem}_metadata.json").write_text(
        json.dumps(companion), encoding="utf-8"
    )
    return name


async def _load_measurements(
    tools, ctx, data_dir: Path, count: int, *, asset: str = ASSET, point: str = POINT
) -> list[str]:
    """Load *count* weekly noise measurements of a point; return their ids.

    The content is seeded by asset, point AND index: a measurement_id is
    the hash of the file bytes, so identical bytes under another asset or
    point would be the SAME measurement (a reattribution or a supersession,
    not a new measurement).
    """
    base = zlib.crc32(f"{asset}/{point}".encode()) % 100_000
    names = [
        _place(
            data_dir,
            f"{asset}_{point}_m{index:02d}.csv",
            _noise(base + index),
            _companion(asset=asset, point=point, index=index),
        )
        for index in range(1, count + 1)
    ]
    infos = await tools["load_signal"](ctx=ctx, filepath=names)
    assert [i.measurement["ledger_status"] for i in infos] == ["recorded"] * count
    return [i.measurement["measurement_id"] for i in infos]


async def _declare_point(tools, ctx, *, asset: str = ASSET, point: str = POINT, **kw):
    defaults = dict(
        bearing_id=BEARING_ID,
        nominal_rpm=NOMINAL_RPM,
        machine_group=2,
        support_type="rigid",
        expected_signal_unit="g",
        expected_direction="horizontal",
    )
    defaults.update(kw)
    return await tools["declare_measurement_point"](
        ctx=ctx, asset_id=asset, measurement_point_id=point, **defaults
    )


def _events(asset: str = ASSET) -> list[dict]:
    return LedgerStore(get_ledger_dir()).read(asset).events


def _event_types(asset: str = ASSET) -> list[str]:
    return [e["event_type"] for e in _events(asset)]


def _keys(tree) -> set:
    """Every dict key in a nested structure."""
    found: set = set()
    if isinstance(tree, dict):
        for key, value in tree.items():
            found.add(key)
            found |= _keys(value)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            found |= _keys(item)
    return found


def _no_error_key(model) -> None:
    dumped = model.model_dump()
    assert "error" not in _keys(dumped)
    json.dumps(dumped, allow_nan=False)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


class TestSurface:
    def test_four_module_level_async_tools_with_ctx_first(self, tools):
        for name in TOOL_NAMES:
            fn = tools[name]
            # Registered behind refusal_boundary (#73): compare the unwrapped function.
            assert getattr(asset_tools, name) is inspect.unwrap(fn)
            assert "<locals>" not in fn.__qualname__
            assert inspect.iscoroutinefunction(fn)
            assert list(inspect.signature(fn).parameters)[0] == "ctx"

    def test_assess_requires_only_asset_and_point(self):
        """AE9: no file, timestamp or id input is required beyond the asset
        and the point (every other parameter has a default)."""
        params = inspect.signature(asset_tools.assess_asset_change).parameters
        required = [
            name
            for name, p in params.items()
            if p.default is inspect.Parameter.empty and name != "ctx"
        ]
        assert required == ["asset_id", "measurement_point_id"]

    @staticmethod
    def _literal_values(annotation) -> set:
        literal = next(
            a
            for a in typing.get_args(annotation)
            if typing.get_origin(a) is typing.Literal
        )
        return set(typing.get_args(literal))

    def test_expected_direction_literal_matches_vocabulary(self):
        params = inspect.signature(asset_tools.declare_measurement_point).parameters
        annotation = params["expected_direction"].annotation
        assert self._literal_values(annotation) == set(VALID_DIRECTIONS)

    def test_expected_signal_unit_literal_matches_vocabulary(self):
        params = inspect.signature(asset_tools.declare_measurement_point).parameters
        annotation = params["expected_signal_unit"].annotation
        assert self._literal_values(annotation) == set(VALID_SIGNAL_UNITS)
        assert set(ledger_service.POINT_SIGNAL_UNITS) == set(VALID_SIGNAL_UNITS)

    def test_group_and_support_literals_match_the_service(self):
        params = inspect.signature(asset_tools.declare_measurement_point).parameters
        assert self._literal_values(params["machine_group"].annotation) == set(
            ledger_service.VALID_MACHINE_GROUPS
        )
        assert self._literal_values(params["support_type"].annotation) == set(
            ledger_service.VALID_SUPPORT_TYPES
        )


# ---------------------------------------------------------------------------
# declare_measurement_point
# ---------------------------------------------------------------------------


class TestDeclareMeasurementPoint:
    @pytest.mark.asyncio
    async def test_versions_diff_stale_count_and_identical_redeclaration(
        self, tools, ctx, sandbox
    ):
        """v1 appends; v2 (group changed after three loads) names the key and
        counts the three stale measurements; after re-processing, v3 (note
        only) leaves nothing stale; v4 identical to v3 appends nothing."""
        v1 = await _declare_point(tools, ctx, declared_by="test")
        assert isinstance(v1, MeasurementPointDeclarationResult)
        assert v1.declaration_version == 1 and v1.appended is True
        assert v1.previous_version is None
        assert v1.changed == sorted(
            [
                "bearing_id",
                "nominal_rpm",
                "machine_group",
                "support_type",
                "expected_signal_unit",
                "expected_direction",
                "declared_by",
            ]
        )
        assert v1.measurements_with_stale_context == 0 and v1.remedy is None
        assert v1.bearing_in_catalog is True
        assert v1.declaration["declaration_version"] == 1
        assert v1.declaration["nominal_rpm"] == 1800.0
        assert v1.event_id and len(v1.event_id) == 64
        _no_error_key(v1)
        assert _event_types() == [EVENT_MEASUREMENT_POINT_DECLARED]

        ids = await _load_measurements(tools, ctx, sandbox, 3)
        assert len(ids) == 3

        v2 = await _declare_point(tools, ctx, machine_group=1, declared_by="test")
        assert v2.declaration_version == 2 and v2.appended is True
        assert v2.previous_version == 1
        assert v2.changed == ["machine_group"]
        assert v2.measurements_with_stale_context == 3
        assert v2.remedy == (
            "assess_asset_change(asset_id='P-101', "
            "measurement_point_id='motor_de_h', reprocess=True)"
        )
        assert v2.remedy in v2.message
        assert v2.declaration["changed"] == ["machine_group"]

        reprocessed = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        assert reprocessed.reprocess["reprocessed"] == 3
        assert reprocessed.reprocess["remaining"] == 0

        v3 = await _declare_point(
            tools, ctx, machine_group=1, declared_by="test", note="rebuilt"
        )
        assert v3.declaration_version == 3 and v3.appended is True
        assert v3.changed == ["note"]
        assert v3.measurements_with_stale_context == 0 and v3.remedy is None

        before = len(_events())
        v4 = await _declare_point(
            tools, ctx, machine_group=1, declared_by="test", note="rebuilt"
        )
        assert v4.declaration_version == 3 and v4.appended is False
        assert v4.changed == [] and v4.previous_version == 3
        assert v4.event_id is None
        assert "nothing appended" in v4.message
        assert len(_events()) == before
        assert _event_types().count(EVENT_MEASUREMENT_POINT_DECLARED) == 3

    @pytest.mark.asyncio
    async def test_fault_orders_route_and_unknown_bearing(self, tools, ctx, sandbox):
        result = await tools["declare_measurement_point"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=POINT,
            fault_orders={"bpfo": 3.58, "BPFI": 5.42},
            nominal_rpm=1800,
        )
        assert result.declaration["fault_orders"] == {"BPFO": 3.58, "BPFI": 5.42}
        assert result.bearing_in_catalog is None

        other = await tools["declare_measurement_point"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=OTHER_POINT,
            bearing_id="ZZZ-NOT-A-BEARING",
        )
        assert other.bearing_in_catalog is False
        assert "not in the verified catalog" in other.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"fault_orders": {"XYZ": 3.5}}, "BPFO"),
            ({"fault_orders": {"BPFO": -1}}, "positive"),
            ({"nominal_rpm": 0}, "nominal_rpm"),
            ({"machine_power_kw": -5}, "machine_power_kw"),
            ({"machine_group": 3}, "machine_group"),
            ({"support_type": "floating"}, "support_type"),
            ({"expected_direction": "sideways"}, "direction"),
            ({"expected_signal_unit": "cm/s"}, "expected_signal_unit"),
        ],
    )
    async def test_vocabulary_and_number_refusals(
        self, tools, ctx, sandbox, kwargs, expected
    ):
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_measurement_point"](
                ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, **kwargs
            )
        assert expected in str(exc_info.value)
        assert not (get_ledger_dir() / f"{ASSET}.jsonl").exists()

    @pytest.mark.asyncio
    async def test_all_problems_in_one_message(self, tools, ctx, sandbox):
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_measurement_point"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                nominal_rpm=-1,
                expected_direction="sideways",
            )
        message = str(exc_info.value)
        assert "nominal_rpm" in message and "direction" in message

    @pytest.mark.asyncio
    async def test_free_text_limits_on_note_and_declared_by(self, tools, ctx, sandbox):
        with pytest.raises(ValueError) as exc_info:
            await _declare_point(tools, ctx, note="x" * 10_000)
        assert str(MAX_FREE_TEXT_CHARS) in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            await _declare_point(tools, ctx, declared_by="L.\nRossi")
        assert "control" in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            await _declare_point(tools, ctx, expected_sensor_id=" ACC")
        assert "control" in str(exc_info.value)
        assert _event_types() == []


# ---------------------------------------------------------------------------
# declare_healthy_baseline
# ---------------------------------------------------------------------------


class TestDeclareHealthyBaseline:
    @pytest.mark.asyncio
    async def test_ae7_assessment_cites_the_baseline_and_its_declarer(
        self, tools, ctx, sandbox
    ):
        await _declare_point(tools, ctx)
        ids = await _load_measurements(tools, ctx, sandbox, 6)
        members = [ids[2], ids[0], ids[3], ids[1]]  # any order in, acquisition out

        result = await tools["declare_healthy_baseline"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=members,
            declared_by="L. Rossi",
            note="after overhaul",
        )
        assert isinstance(result, BaselineDeclarationResult)
        assert len(result.baseline_id) == 16
        assert result.measurement_ids == ids[:4]
        assert [m["measurement_id"] for m in result.members] == ids[:4]
        assert all(m["declaration_version"] == 1 for m in result.members)
        assert all(m["point_declaration_version"] == 1 for m in result.members)
        assert result.declared_by == "L. Rossi" and result.note == "after overhaul"
        assert result.withdrawn is False and result.superseded_baseline_id is None
        assert result.declared_at.endswith("+00:00")
        assert "L. Rossi" in result.message
        _no_error_key(result)
        assert _event_types()[-1] == EVENT_BASELINE_DECLARED

        assessment = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT
        )
        assert assessment.status == "assessed"
        reference = assessment.reference
        assert reference["kind"] == "declared_baseline"
        assert reference["health_declared"] is True
        assert reference["measurement_ids"] == ids[:4]
        assert "declared healthy baseline" in reference["message"]
        assert "declared by L. Rossi" in reference["message"]
        assert "after overhaul" in reference["message"]
        assert reference["baseline"]["baseline_id"] == result.baseline_id
        assert "declared by L. Rossi" in assessment.message

        second = await tools["declare_healthy_baseline"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=ids[1:4],
            declared_by="L. Rossi",
        )
        assert second.superseded_baseline_id == result.baseline_id
        assert second.baseline_id != result.baseline_id

        withdrawal = await tools["declare_healthy_baseline"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=[],
            declared_by="L. Rossi",
            note="sensor remounted",
        )
        assert withdrawal.withdrawn is True
        assert withdrawal.measurement_ids == [] and withdrawal.members == []
        assert withdrawal.superseded_baseline_id == second.baseline_id
        after = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT
        )
        assert after.status == "assessed"
        assert after.reference["kind"] == "automatic_window"
        assert after.reference["health_declared"] is False
        # The withdrawal is itself a baseline_declared record (empty members);
        # the assessment cites that record's id, declarer and instant.
        assert after.reference["withdrawn_baseline"] == {
            "baseline_id": withdrawal.baseline_id,
            "declared_by": "L. Rossi",
            "declared_at": withdrawal.declared_at,
        }
        assert "withdrawn" in after.reference["message"]

    @pytest.mark.asyncio
    async def test_withdrawal_needs_a_note_and_an_active_baseline(
        self, tools, ctx, sandbox
    ):
        await _load_measurements(tools, ctx, sandbox, 3)
        with pytest.raises(ValueError, match="note"):
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=[],
                declared_by="L. Rossi",
            )
        with pytest.raises(ValueError, match="no active baseline"):
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=[],
                declared_by="L. Rossi",
                note="nothing to withdraw",
            )

    @pytest.mark.asyncio
    async def test_id_of_another_point_names_the_valid_ids(self, tools, ctx, sandbox):
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        other = await _load_measurements(tools, ctx, sandbox, 1, point=OTHER_POINT)
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids[:2] + other,
                declared_by="L. Rossi",
            )
        message = str(exc_info.value)
        assert other[0] in message
        assert f"belongs to point {OTHER_POINT!r}" in message
        for measurement_id in ids:
            assert measurement_id in message  # the valid ids of the point
        assert "get_asset_history" in message
        assert EVENT_BASELINE_DECLARED not in _event_types()

    @pytest.mark.asyncio
    async def test_unknown_id_and_too_few_members_are_refused(
        self, tools, ctx, sandbox
    ):
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        with pytest.raises(ValueError, match="not measurements of point"):
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids[:2] + ["0123456789abcdef"],
                declared_by="L. Rossi",
            )
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids[:2],
                declared_by="L. Rossi",
            )
        assert f"at least {MIN_REFERENCE_MEASUREMENTS}" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_comparable_member_is_refused_with_the_reason(
        self, tools, ctx, sandbox
    ):
        await _declare_point(tools, ctx)  # expects unit g
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        name = _place(
            sandbox,
            "velocity.csv",
            _noise(9),
            _companion(index=9, signal_unit="mm/s"),
        )
        info = await tools["load_signal"](ctx=ctx, filepath=name)
        assert info.measurement["comparability"]["grade"] == "non_comparable"
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids + [info.measurement["measurement_id"]],
                declared_by="L. Rossi",
            )
        message = str(exc_info.value)
        assert "unit_incompatible" in message
        assert info.measurement["measurement_id"] in message
        assert EVENT_BASELINE_DECLARED not in _event_types()

    @pytest.mark.asyncio
    async def test_declared_by_and_note_rules(self, tools, ctx, sandbox):
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        with pytest.raises(ValueError, match="declared_by"):
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids,
                declared_by="",
            )
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids,
                declared_by="L. Rossi",
                note="n" * 10_000,
            )
        assert str(MAX_FREE_TEXT_CHARS) in str(exc_info.value)
        with pytest.raises(ValueError, match="control"):
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                measurement_ids=ids,
                declared_by="L.\x00Rossi",
            )
        assert EVENT_BASELINE_DECLARED not in _event_types()

    @pytest.mark.asyncio
    async def test_unknown_asset_is_a_refusal_naming_known_assets(
        self, tools, ctx, sandbox
    ):
        await _load_measurements(tools, ctx, sandbox, 1)
        with pytest.raises(ValueError) as exc_info:
            await tools["declare_healthy_baseline"](
                ctx=ctx,
                asset_id="P-404",
                measurement_point_id=POINT,
                measurement_ids=["0123456789abcdef"] * 1,
                declared_by="L. Rossi",
            )
        assert "no ledger" in str(exc_info.value)
        assert ASSET in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_asset_history
# ---------------------------------------------------------------------------


class TestGetAssetHistory:
    @pytest.mark.asyncio
    async def test_index_after_two_assets(self, tools, ctx, sandbox):
        await _declare_point(tools, ctx)
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        await tools["declare_healthy_baseline"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=POINT,
            measurement_ids=ids,
            declared_by="L. Rossi",
        )
        await _load_measurements(tools, ctx, sandbox, 1, asset=OTHER_ASSET)

        result = await tools["get_asset_history"](ctx=ctx)
        assert isinstance(result, AssetHistoryResult)
        assert result.status == "index" and result.asset is None
        assert result.truncated is False
        assert result.known_assets == [ASSET, OTHER_ASSET]
        _no_error_key(result)

        first, second = result.assets
        assert first["asset_id"] == ASSET
        assert first["measurement_count"] == 3 and first["point_count"] == 1
        assert first["first_acquired_at"] == "2026-01-05T08:00:00+00:00"
        assert first["last_acquired_at"] == "2026-01-19T08:00:00+00:00"
        (point,) = first["points"]
        assert point["measurement_point_id"] == POINT
        assert point["measurement_count"] == 3
        assert point["last_acquired_at"] == "2026-01-19T08:00:00+00:00"
        assert point["latest_lineage"] == processing_id()
        assert point["baseline_declared"] is True
        assert point["declaration_version"] == 1
        assert first["event_count"] == 1 + 3 * 2 + 1 == len(_events(ASSET))
        assert (
            first["ledger_bytes"]
            == (get_ledger_dir() / f"{ASSET}.jsonl").stat().st_size
        )
        assert first["integrity"]["readable_records"] == 8
        assert first["integrity"]["unreadable_records"] == 0
        assert "issues" not in first["integrity"]

        assert second["asset_id"] == OTHER_ASSET
        assert second["measurement_count"] == 1
        assert second["points"][0]["baseline_declared"] is False
        assert second["points"][0]["declaration_version"] is None
        assert second["event_count"] == 2

    @pytest.mark.asyncio
    async def test_history_reads_only_that_ledger_newest_first(
        self, tools, ctx, sandbox, monkeypatch
    ):
        await _declare_point(tools, ctx)
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        await _load_measurements(tools, ctx, sandbox, 2, asset=OTHER_ASSET)

        touched: list[str] = []
        read_segment = LedgerStore._read_segment
        first_event = LedgerStore._first_readable_event

        def spy_read(self, path, offset):
            touched.append(Path(path).name)
            return read_segment(self, path, offset)

        def spy_first(path):
            touched.append(Path(path).name)
            return first_event(path)

        monkeypatch.setattr(LedgerStore, "_read_segment", spy_read)
        monkeypatch.setattr(
            LedgerStore, "_first_readable_event", staticmethod(spy_first)
        )

        result = await tools["get_asset_history"](ctx=ctx, asset_id=ASSET)
        assert result.status == "found" and result.assets == []
        assert set(touched) == {f"{ASSET}.jsonl"}
        _no_error_key(result)

        asset = result.asset
        assert asset["asset_id"] == ASSET
        assert [m["measurement_id"] for m in asset["measurements"]] == ids[::-1]
        assert asset["measurement_count"] == 3 and result.truncated is False
        newest = asset["measurements"][0]
        assert newest["acquired_at"] == "2026-01-19T08:00:00+00:00"
        assert newest["signal_id"] == f"{ASSET}_{POINT}_m03"
        assert newest["location"] == f"{ASSET}_{POINT}_m03.csv"
        assert newest["declaration_version"] == 1
        assert newest["lineages"] == [processing_id()]
        assert set(newest["indicators"]) == {
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
            "unit",
        }
        assert newest["indicators"]["unit"] == "g" and newest["indicators"]["rms"] > 0
        assert newest["comparability"]["grade"] == "comparable"
        assert asset["point_declarations"][POINT]["current"]["declaration_version"] == 1
        assert asset["baselines"] == {}
        assert asset["reattributed"] == []
        assert asset["integrity"]["readable_records"] == 7
        assert asset["event_count"] == 7
        assert (
            asset["ledger_bytes"]
            == (get_ledger_dir() / f"{ASSET}.jsonl").stat().st_size
        )
        assert result.known_points == [POINT]

    @pytest.mark.asyncio
    async def test_history_truncated_over_max_measurements(self, tools, ctx, sandbox):
        ids = await _load_measurements(tools, ctx, sandbox, 3)
        result = await tools["get_asset_history"](
            ctx=ctx, asset_id=ASSET, max_measurements=2
        )
        assert result.truncated is True
        assert [m["measurement_id"] for m in result.asset["measurements"]] == ids[:0:-1]
        assert result.asset["measurement_count"] == 3

    @pytest.mark.asyncio
    async def test_point_filter(self, tools, ctx, sandbox):
        ids = await _load_measurements(tools, ctx, sandbox, 2)
        other = await _load_measurements(tools, ctx, sandbox, 1, point=OTHER_POINT)
        result = await tools["get_asset_history"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=OTHER_POINT
        )
        assert result.status == "found"
        assert [m["measurement_id"] for m in result.asset["measurements"]] == other
        assert [
            p["measurement_point_id"] for p in result.asset["summary"]["points"]
        ] == [OTHER_POINT]
        assert result.asset["summary"]["measurement_count"] == 3
        assert result.known_points == [POINT, OTHER_POINT]
        assert ids[0] not in json.dumps(result.asset["measurements"])

        miss = await tools["get_asset_history"](
            ctx=ctx, asset_id=ASSET, measurement_point_id="pump_x"
        )
        assert miss.status == "not_found" and miss.asset is None
        assert miss.known_points == [POINT, OTHER_POINT]
        assert "declare_measurement_point" in miss.suggestion

    @pytest.mark.asyncio
    async def test_index_truncated_over_max_index_assets(
        self, tools, ctx, sandbox, monkeypatch
    ):
        await _load_measurements(tools, ctx, sandbox, 1)
        await _load_measurements(tools, ctx, sandbox, 1, asset=OTHER_ASSET)
        monkeypatch.setattr(asset_tools, "MAX_INDEX_ASSETS", 1)
        result = await tools["get_asset_history"](ctx=ctx)
        assert result.status == "index"
        assert [a["asset_id"] for a in result.assets] == [ASSET]
        assert result.known_assets == [ASSET]
        assert result.truncated is True
        assert "the first 1" in result.message

    @pytest.mark.asyncio
    async def test_point_without_asset_is_a_refusal(self, tools, ctx, sandbox):
        with pytest.raises(ValueError, match="asset_id"):
            await tools["get_asset_history"](ctx=ctx, measurement_point_id="x")

    @pytest.mark.asyncio
    async def test_max_measurements_bounds(self, tools, ctx, sandbox):
        for value in (0, 201):
            with pytest.raises(ValueError) as exc_info:
                await tools["get_asset_history"](
                    ctx=ctx, asset_id=ASSET, max_measurements=value
                )
            assert "between 1 and 200" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_asset_is_typed_not_found(self, tools, ctx, sandbox):
        await _load_measurements(tools, ctx, sandbox, 1)
        result = await tools["get_asset_history"](ctx=ctx, asset_id="P-404")
        assert result.status == "not_found" and result.asset is None
        assert result.known_assets == [ASSET]
        assert "P-404" in result.suggestion
        _no_error_key(result)

    @pytest.mark.asyncio
    async def test_empty_ledger_directory_is_an_empty_index(self, tools, ctx, sandbox):
        result = await tools["get_asset_history"](ctx=ctx)
        assert result.status == "index" and result.assets == []
        assert result.truncated is False


# ---------------------------------------------------------------------------
# assess_asset_change
# ---------------------------------------------------------------------------


class TestAssessAssetChange:
    @pytest.mark.asyncio
    async def test_ae9_structure_on_the_synthetic_sequence(self, tools, ctx, sandbox):
        """Declare the point, load M01..M30 in one batch, assess: the three
        blocks are present, the verification is one sentence or None, no
        key is named recommendations, and the ramp is a persistent change
        with BPFO evidence."""
        declared = await tools["declare_measurement_point"](
            ctx=ctx,
            asset_id=SEQUENCE_ASSET,
            measurement_point_id=SEQUENCE_POINT,
            bearing_id=BEARING_ID,
            nominal_rpm=NOMINAL_RPM,
            machine_group=2,
            support_type="rigid",
            expected_signal_unit="g",
            expected_direction="horizontal",
        )
        assert declared.declaration_version == 1
        paths = build_measurement_sequence(sandbox / "seq")
        assert len(paths) == SEQUENCE_LENGTH
        infos = await tools["load_signal"](
            ctx=ctx, filepath=[f"seq/{p.name}" for p in paths]
        )
        assert len(infos) == SEQUENCE_LENGTH
        assert {i.measurement["ledger_status"] for i in infos} == {"recorded"}
        assert {i.measurement["snapshot_status"] for i in infos} == {"complete"}

        result = await tools["assess_asset_change"](
            ctx=ctx, asset_id=SEQUENCE_ASSET, measurement_point_id=SEQUENCE_POINT
        )
        assert isinstance(result, AssetChangeAssessment)
        assert result.status == "assessed"
        assert result.reprocess is None
        _no_error_key(result)
        assert "recommendations" not in _keys(result.model_dump())

        assert result.reference["kind"] == "automatic_window"
        assert result.reference["health_declared"] is False
        assert result.reference["count"] == 10
        assert "not declared" in result.reference["message"]
        assert result.lineage["processing_id"] == processing_id()
        assert result.lineage["is_current"] is True

        observed = result.observed
        assert set(observed["reference_statistics"]) >= {"rms", "peak", "one_x"}
        assert "envelope_BPFO" in observed["reference_statistics"]
        assert observed["reference_statistics"]["rms"]["n"] == 10
        assert observed["acquisitions_assessed"] == 20
        assert (
            observed["latest_measurement_id"] == infos[-1].measurement["measurement_id"]
        )
        assert len(observed["measurement_ids_assessed"]) == 5
        assert observed["evidence_presence"]["BPFO"]["k"] == 5

        derived = result.derived
        assert derived["deltas"]["rms"]["latest"] > derived["deltas"]["rms"]["mean"]
        assert derived["per_indicator"]["rms"]["criterion"]
        assert derived["iso_change"]["machine_group"] == 2

        assessed = result.assessed
        assert assessed["classification"] == "persistent_change"
        assert assessed["direction"] == "increase"
        assert assessed["criterion"]
        assert assessed["evidence"]["BPFO"]["classification"] == "persistent_change"
        assert assessed["evidence"]["BPFO"]["present_in_last_k"] >= 3
        assert result.comparability["non_comparable"] == 0
        assert result.comparability["qualified"] == 10  # the rpm-less companions

        verification = result.suggested_verification
        assert verification is None or (
            isinstance(verification, str)
            and verification.endswith(".")
            and ". " not in verification
        )
        assert "persistent change" in result.message

    @pytest.mark.asyncio
    async def test_limits_are_refused_with_the_bounds(self, tools, ctx, sandbox):
        with pytest.raises(ValueError) as exc_info:
            await tools["assess_asset_change"](
                ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, last_k=0
            )
        assert "last_k" in str(exc_info.value) and "between 1 and 50" in str(
            exc_info.value
        )
        with pytest.raises(ValueError) as exc_info:
            await tools["assess_asset_change"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                reference_measurements=2,
            )
        assert "reference_measurements" in str(exc_info.value)
        assert "between 3 and 100" in str(exc_info.value)
        with pytest.raises(ValueError, match="ISO 8601"):
            await tools["assess_asset_change"](
                ctx=ctx,
                asset_id=ASSET,
                measurement_point_id=POINT,
                acquired_since="yesterday",
            )

    @pytest.mark.asyncio
    async def test_unknown_asset_and_point_are_typed_misses(self, tools, ctx, sandbox):
        await _load_measurements(tools, ctx, sandbox, 1)
        miss = await tools["assess_asset_change"](
            ctx=ctx, asset_id="P-404", measurement_point_id=POINT
        )
        assert miss.status == "not_found"
        assert miss.known_assets == [ASSET]
        assert miss.suggestion and miss.observed is None
        _no_error_key(miss)

        point_miss = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id="pump_x"
        )
        assert point_miss.status == "not_found"
        assert point_miss.known_points == [POINT]
        assert "declare_measurement_point" in point_miss.suggestion

    @pytest.mark.asyncio
    async def test_insufficient_history_is_typed(self, tools, ctx, sandbox):
        await _load_measurements(tools, ctx, sandbox, 2)
        result = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT
        )
        assert result.status == "insufficient_history"
        assert result.available == 2 and result.required == 4
        assert "declare_healthy_baseline" in result.remedy
        assert result.assessed is None

    @pytest.mark.asyncio
    async def test_reprocess_with_homogeneous_lineage_has_nothing_to_do(
        self, tools, ctx, sandbox
    ):
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 5)
        before = len(_events())
        result = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        assert result.status == "assessed"
        block = result.reprocess
        assert block["stale"] == 0 and block["reprocessed"] == 0
        assert block["up_to_date"] == 5 and block["remaining"] == 0
        assert block["next_call"] is None
        assert "nothing to reprocess" in block["message"]
        assert len(_events()) == before

    @pytest.mark.asyncio
    async def test_reprocess_is_bounded_and_names_the_next_call(
        self, tools, ctx, sandbox, monkeypatch
    ):
        """15 measurements on lineage 1; bump the algorithm version so the
        current lineage differs: the first call re-processes 10, reports 5
        remaining and the exact next call; the second finishes; a third has
        nothing to do. The old snapshots stay."""
        await _declare_point(tools, ctx)
        ids = await _load_measurements(tools, ctx, sandbox, 15)
        old = processing_id()
        monkeypatch.setattr(
            snapshot_module, "ALGORITHM_VERSION", snapshot_module.ALGORITHM_VERSION + 1
        )
        new = processing_id()
        assert new != old

        first = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        block = first.reprocess
        assert block["processing_id"] == new
        assert block["stale"] == 15
        assert block["reprocessed"] == MAX_REPROCESS_PER_CALL == 10
        assert block["remaining"] == 5
        assert block["next_call"] == (
            "assess_asset_change(asset_id='P-101', "
            "measurement_point_id='motor_de_h', reprocess=True)"
        )
        assert [r["outcome"] for r in block["results"]] == ["reprocessed"] * 10
        # Reference members first (the first ten by acquisition).
        assert [r["measurement_id"] for r in block["results"]] == ids[:10]
        assert first.status == "assessed"
        assert first.lineage["processing_id"] == old  # old lineage still covers all
        assert first.lineage["is_current"] is False
        assert first.lineage["candidates"] == {old: 15, new: 10}

        second = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        assert second.reprocess["reprocessed"] == 5
        assert second.reprocess["remaining"] == 0
        assert second.reprocess["next_call"] is None
        assert second.lineage["processing_id"] == new
        assert second.lineage["is_current"] is True

        third = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        assert third.reprocess["stale"] == 0
        assert "nothing to reprocess" in third.reprocess["message"]
        assert _event_types().count(EVENT_HEALTH_SNAPSHOT_COMPUTED) == 30
        assert _event_types().count(EVENT_MEASUREMENT_RECORDED) == 15

    @pytest.mark.asyncio
    async def test_revert_with_a_missing_file_keeps_both_blocks_consistent(
        self, tools, ctx, sandbox
    ):
        """Ten loads under A, re-processed under B; an eleventh
        loaded under B only, its file then removed; A declared again. The
        reverted slots are scored from A's snapshots, the eleventh stays
        stale, and the reprocess block in the same response finds it not
        reprocessable and ends the sequence: the message names no call and
        says the slot stays stale until its action is taken, so one response
        never contradicts itself."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 10)
        await _declare_point(tools, ctx, machine_group=1)
        first = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        assert first.reprocess["reprocessed"] == 10
        name = _place(sandbox, "late_m11.csv", _noise(777), _companion(index=11))
        late = await tools["load_signal"](ctx=ctx, filepath=name)
        late_id = late.measurement["measurement_id"]
        (sandbox / name).unlink()
        restored = await _declare_point(tools, ctx)
        assert restored.declaration_version == 3
        assert restored.measurements_with_stale_context == 1

        result = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
        )
        block = result.reprocess
        assert (block["stale"], block["reprocessed"]) == (1, 0)
        assert block["not_reprocessable"] == 1
        assert block["next_call"] is None
        assert block["sequence_state"] == "exhausted"
        assert block["results"][0]["measurement_id"] == late_id
        assert block["results"][0]["reason_code"] == "file_missing"
        assert "file not found" in block["results"][0]["reason"]

        assert result.status == "assessed"
        assert result.lineage["is_current"] is True
        assert result.lineage["stale_context"] == 1
        assert result.observed["latest_measurement_id"] == late_id
        assert result.message.startswith(
            "1 of the 11 evaluated acquisition slots was scored from snapshots "
            "computed under a previous declaration of the point or the "
            "measurement, so this verdict does not reflect the current "
            "declaration yet; 1 evaluated slot(s) lacking the "
            f"{processing_id()} snapshot of their current declarations stay "
            "stale until an action is taken: 1 needs the file restored or "
            "located (the reprocess block reports them). "
        )
        assert "reprocess=True" not in result.message
        assert "recomputes" not in result.message
        assert result.suggested_verification == (
            "Treat this classification as provisional: before relying on it, "
            "check the lineage assessed and whether the snapshots of the "
            "evaluated acquisitions were computed under their current "
            "declarations."
        )

    @pytest.mark.asyncio
    async def test_reverted_point_assessments_are_repeatable(self, tools, ctx, sandbox):
        """After A, B (re-processed), A: two calls give the same structured
        content, name no call and append nothing."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 11)
        await _declare_point(tools, ctx, machine_group=1)
        for _ in range(2):
            await tools["assess_asset_change"](
                ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, reprocess=True
            )
        await _declare_point(tools, ctx)
        before = len(_events())
        first = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT
        )
        second = await tools["assess_asset_change"](
            ctx=ctx, asset_id=ASSET, measurement_point_id=POINT
        )
        assert first.model_dump() == second.model_dump()
        assert first.lineage["stale_context"] == 0
        assert "reprocess=True" not in first.message
        assert len(_events()) == before


#: The parts of the re-processing stop rule, by the phrase every reference
#: surface states for each.
STOP_RULE_PARTS = {
    "a call without the token starts a new sequence": (
        "starts a new re-processing sequence"
    ),
    "a named call is an attempt": "an attempt, not a promise",
    "within a sequence, follow next_call exactly": "make exactly that call",
    "when it ends, read the state": "sequence_state",
    "and report what needs an action": "needs_action",
    "a new sequence attempts the costly failures again": (
        "attempts again the failures only an attempt finds"
    ),
}

#: The start conditions of a new sequence, which every surface that lists
#: them lists in full.
START_CONDITIONS = (
    "including the call a message, a remedy or a declaration names",
    "after the user took an action",
    "after a load or a declaration changed the point",
    "once after an environment fix",
    "asks to list the blocked measurements again",
)

#: Phrases no surface and no output uses: they promise an end a sequence
#: may never reach, or a computation that is only an attempt.
RETIRED_PHRASES = ("remaining 0", "repeat until remaining", "recomputes")

REPO = Path(__file__).resolve().parent.parent


def _plain(text: str) -> str:
    """*text* without Markdown code marks, on one line."""
    return " ".join(text.replace("`", "").split())


def assert_runtime_wording(*texts: Optional[str]) -> None:
    """Outputs built at runtime state only the part of the stop rule that
    applies, and never a retired phrase."""
    for text in texts:
        for phrase in RETIRED_PHRASES:
            assert phrase not in _plain(text or ""), (phrase, text)


def _measurement_file(sandbox: Path, index: int, point: str = POINT) -> Path:
    return sandbox / f"{ASSET}_{point}_m{index:02d}.csv"


def _same_size_change(path: Path) -> None:
    """Other bytes of the same length: the lines in reverse order."""
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(reversed(lines)))


def _named_calls(text: str) -> list[str]:
    return [
        f"{name}({args})"
        for name, args, _ in extract_calls(text)
        if name == "assess_asset_change"
    ]


async def _assess(tools, ctx, **kwargs) -> AssetChangeAssessment:
    return await tools["assess_asset_change"](
        ctx=ctx, asset_id=ASSET, measurement_point_id=POINT, **kwargs
    )


class TestReprocessStopRule:
    """Every surface that recommends re-processing states the same stop
    rule; none promises an end the sequence may not reach."""

    @staticmethod
    def reference_surfaces() -> dict[str, str]:
        from predictive_maintenance_mcp import server

        changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        assessment = AssetChangeAssessment.model_fields
        declaration = MeasurementPointDeclarationResult.model_fields
        return {
            "server instructions": server.mcp.instructions or "",
            "assess_asset_change docstring": asset_tools.assess_asset_change.__doc__,
            "declare_measurement_point docstring": (
                asset_tools.declare_measurement_point.__doc__
            ),
            "AssetChangeAssessment.reprocess": assessment["reprocess"].description,
            "AssetChangeAssessment.remedy": assessment["remedy"].description,
            "MeasurementPointDeclarationResult": (
                MeasurementPointDeclarationResult.__doc__
            ),
            "MeasurementPointDeclarationResult.remedy": (
                declaration["remedy"].description
            ),
            "EXAMPLES.md": (REPO / "EXAMPLES.md").read_text(encoding="utf-8"),
            "docs/TOOL_CATALOG.md": (REPO / "docs" / "TOOL_CATALOG.md").read_text(
                encoding="utf-8"
            ),
            "docs/ADAPTER_GUIDE.md": (REPO / "docs" / "ADAPTER_GUIDE.md").read_text(
                encoding="utf-8"
            ),
            "CHANGELOG.md [Unreleased]": unreleased,
        }

    def test_every_reference_surface_states_every_part(self):
        for surface, text in self.reference_surfaces().items():
            plain = _plain(text)
            for part, phrase in STOP_RULE_PARTS.items():
                assert phrase in plain, f"{surface} does not state: {part}"

    def test_no_reference_surface_uses_a_retired_phrase(self):
        for surface, text in self.reference_surfaces().items():
            for phrase in RETIRED_PHRASES:
                assert phrase not in _plain(text), f"{surface}: {phrase!r}"

    def test_every_listing_surface_lists_every_start_condition(self):
        """The surfaces that enumerate when a new sequence starts enumerate
        the same conditions (the shorter surfaces only say that a call
        without the token starts one)."""
        surfaces = self.reference_surfaces()
        listing = [
            "server instructions",
            "assess_asset_change docstring",
            "AssetChangeAssessment.reprocess",
            "AssetChangeAssessment.remedy",
            "MeasurementPointDeclarationResult.remedy",
            "EXAMPLES.md",
            "docs/TOOL_CATALOG.md",
        ]
        for surface in listing:
            plain = _plain(surfaces[surface])
            for condition in START_CONDITIONS:
                assert condition in plain, f"{surface} omits: {condition}"

    def test_the_stated_bounds_are_the_constants(self):
        """Every surface that states the per-call bounds states the values
        the service enforces."""
        bounds = (
            f"{ledger_service.MAX_REPROCESS_PER_CALL} attempts",
            f"{ledger_service.MAX_FILE_CHECKS_PER_CALL} file checks",
            f"{ledger_service.MAX_LISTED_ITEMS} failures",
        )
        surfaces = self.reference_surfaces()
        stating = [
            "server instructions",
            "assess_asset_change docstring",
            "AssetChangeAssessment.reprocess",
            "EXAMPLES.md",
            "docs/TOOL_CATALOG.md",
            "CHANGELOG.md [Unreleased]",
        ]
        for surface in stating:
            plain = _plain(surfaces[surface])
            for bound in bounds:
                assert bound in plain, f"{surface} does not state {bound}"


class TestReprocessSequenceThroughTheTool:
    """The tool's side of a re-processing sequence: the continuation token,
    the misuse rails, and one named call per response."""

    @pytest.mark.asyncio
    async def test_ae7_s1_ends_exhausted_and_names_no_call(self, tools, ctx, sandbox):
        """S1 in one reprocess=True call: the 8 later slots are re-processed,
        the 12 reference files are gone; the assessment still opens with the
        stale-evidence sentence and names no call."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 20)
        await _declare_point(tools, ctx, machine_group=1)
        for index in range(1, 13):
            _measurement_file(sandbox, index).unlink()
        result = await _assess(tools, ctx, reference_measurements=12, reprocess=True)
        block = result.reprocess
        assert (block["reprocessed"], block["not_reprocessable"]) == (8, 12)
        assert block["sequence_state"] == "exhausted"
        assert block["remaining"] == 0 and block["next_call"] is None
        assert result.status == "assessed"
        assert result.lineage["stale_context"] == 12
        assert result.message.startswith(
            "12 of the 20 evaluated acquisition slots were scored from snapshots "
            "computed under a previous declaration of the point or the "
            "measurement, so this verdict does not reflect the current "
            "declaration yet; 12 evaluated slot(s) lacking the "
            f"{processing_id()} snapshot of their current declarations stay "
            "stale until an action is taken: 12 need the file restored or "
            "located (the reprocess block reports them). "
        )
        assert "reprocess=True" not in result.message
        assert result.suggested_verification.startswith(
            "Treat this classification as provisional"
        )
        assert_runtime_wording(result.message, block["message"])

    @pytest.mark.asyncio
    async def test_mid_sequence_every_named_call_is_the_next_call(
        self, tools, ctx, sandbox
    ):
        """Ten changed files ahead of five eligible ones: the first response
        carries a token, and its message names exactly its next_call, which
        the registered surface accepts; the call it names ends the sequence."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 15)
        await _declare_point(tools, ctx, machine_group=1)
        for index in range(1, 11):
            _same_size_change(_measurement_file(sandbox, index))
        events = len(_events())
        first = await _assess(tools, ctx, last_k=4, reprocess=True)
        block = first.reprocess
        token = block["next_cursor"]
        assert token is not None and block["sequence_state"] == "continue"
        assert block["next_call"] == (
            "assess_asset_change(asset_id='P-101', measurement_point_id='motor_de_h', "
            f"last_k=4, reprocess=True, reprocess_cursor={token!r})"
        )
        assert _named_calls(first.message) == [block["next_call"]]
        assert block["next_call"] in block["message"]
        endpoints = build_endpoints()
        for text in (first.message, block["message"], block["next_call"]):
            assert validate_documented_calls(text, endpoints, "runtime") == []

        second = await _assess(
            tools, ctx, last_k=4, reprocess=True, reprocess_cursor=token
        )
        assert second.reprocess["restarted"] is None
        assert second.reprocess["cursor"] == token
        assert second.reprocess["sequence_state"] == "exhausted"
        assert second.reprocess["earlier_in_sequence"] == 10
        assert _named_calls(second.message) == []
        assert_runtime_wording(
            first.message, block["message"], second.message, second.reprocess["message"]
        )
        assert _event_types().count(EVENT_HEALTH_SNAPSHOT_COMPUTED) == 15 + 5
        assert len(_events()) == events + 5

    @pytest.mark.asyncio
    async def test_the_remedy_names_the_next_call_mid_sequence(
        self, tools, ctx, sandbox, monkeypatch
    ):
        """processing_not_homogeneous while a sequence runs: the remedy names
        the block's next_call, token included, and counts the slots found
        not reprocessable."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 20)
        monkeypatch.setattr(
            snapshot_module, "ALGORITHM_VERSION", snapshot_module.ALGORITHM_VERSION + 1
        )
        names = [
            _place(
                sandbox,
                f"late_m{index:02d}.csv",
                _noise(900 + index),
                _companion(index=index),
            )
            for index in range(21, 31)
        ]
        await tools["load_signal"](ctx=ctx, filepath=names)
        for index in range(1, 11):
            _same_size_change(_measurement_file(sandbox, index))
        result = await _assess(tools, ctx, reprocess=True)
        block = result.reprocess
        assert result.status == "processing_not_homogeneous"
        assert block["next_cursor"] is not None
        assert result.remedy.startswith(f"call {block['next_call']} to attempt")
        assert "10 of the 20 slots missing one stay without it" in result.remedy
        assert "need the user's decision on a changed file" in result.remedy
        assert (
            validate_documented_calls(result.remedy, build_endpoints(), "remedy") == []
        )
        assert_runtime_wording(result.remedy, result.message, block["message"])

    @pytest.mark.asyncio
    async def test_the_remedy_of_a_continuing_call_counts_the_earlier_failures(
        self, tools, ctx, sandbox, monkeypatch
    ):
        """Twenty changed files among 25 slots on the older lineage: the
        second call, given the token, names its own next_call and counts the
        10 failures the first call found with the 5 it found itself; the third
        call names no call."""
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 25)
        monkeypatch.setattr(
            snapshot_module, "ALGORITHM_VERSION", snapshot_module.ALGORITHM_VERSION + 1
        )
        names = [
            _place(
                sandbox,
                f"late_m{index:02d}.csv",
                _noise(900 + index),
                _companion(index=index),
            )
            for index in range(26, 31)
        ]
        await tools["load_signal"](ctx=ctx, filepath=names)
        for index in range(1, 21):
            _same_size_change(_measurement_file(sandbox, index))
        first = await _assess(tools, ctx, reprocess=True)
        assert first.reprocess["reprocessed"] == 0
        assert first.reprocess["not_reprocessable"] == 10
        token = first.reprocess["next_cursor"]
        assert token is not None

        second = await _assess(tools, ctx, reprocess=True, reprocess_cursor=token)
        block = second.reprocess
        assert block["restarted"] is None
        assert (block["reprocessed"], block["not_reprocessable"]) == (5, 5)
        assert block["earlier_in_sequence"] == 10
        assert block["next_cursor"] not in (None, token)
        assert second.status == "processing_not_homogeneous"
        assert second.remedy.startswith(f"call {block['next_call']} to attempt")
        assert _named_calls(second.remedy) == [block["next_call"]]
        assert (
            "; 15 of the 20 slots missing one stay without it until an action is "
            "taken: 15 need the user's decision on a changed file"
        ) in second.remedy
        assert (
            validate_documented_calls(second.remedy, build_endpoints(), "remedy") == []
        )

        third = await _assess(
            tools, ctx, reprocess=True, reprocess_cursor=block["next_cursor"]
        )
        assert third.reprocess["sequence_state"] == "exhausted"
        assert third.reprocess["earlier_in_sequence"] == 15
        assert third.remedy.startswith(
            "re-processing cannot compute the 20 missing "
            f"{third.current_processing_id} snapshot(s) of the evaluated slots until "
            "an action is taken: 20 need the user's decision on a changed file"
        )
        assert _named_calls(third.remedy) == []
        assert_runtime_wording(second.remedy, third.remedy, block["message"])

    @pytest.mark.asyncio
    async def test_ae9_token_misuse_is_refused_before_the_ledger_is_read(
        self, tools, ctx, sandbox, monkeypatch
    ):
        def unread(self, asset_id):
            raise AssertionError("the ledger was read")

        monkeypatch.setattr(LedgerStore, "read_view", unread)
        valid = "r1." + "0" * 16 + "." + "0" * 8 + ".f.f1"
        for kwargs, problem in (
            ({"reprocess_cursor": valid}, "needs reprocess=True"),
            ({"reprocess": True, "reprocess_cursor": valid[:-3]}, "shape"),
            ({"reprocess": True, "reprocess_cursor": valid + "é"}, "shape"),
            ({"reprocess": True, "reprocess_cursor": "r" * 201}, "more than the 200"),
        ):
            with pytest.raises(ValueError, match=problem) as refused:
                await _assess(tools, ctx, **kwargs)
            message = str(refused.value)
            assert "pass the token exactly as next_call gives it" in message

    @pytest.mark.asyncio
    async def test_ae9_a_token_of_another_point_restarts(self, tools, ctx, sandbox):
        await _declare_point(tools, ctx)
        await _load_measurements(tools, ctx, sandbox, 12)
        await _declare_point(tools, ctx, machine_group=1)
        for index in range(1, 11):
            _same_size_change(_measurement_file(sandbox, index))
        token = (await _assess(tools, ctx, reprocess=True)).reprocess["next_cursor"]
        assert token is not None
        await _declare_point(tools, ctx, point=OTHER_POINT)
        await _load_measurements(tools, ctx, sandbox, 4, point=OTHER_POINT)
        other = await tools["assess_asset_change"](
            ctx=ctx,
            asset_id=ASSET,
            measurement_point_id=OTHER_POINT,
            reprocess=True,
            reprocess_cursor=token,
        )
        assert other.reprocess["restarted"]["reason_code"] == "unknown_measurement"
        assert other.reprocess["message"].startswith(
            "The continuation token was not used (the measurement it continues "
            "after is no longer a measurement of this point)"
        )

    @pytest.mark.asyncio
    async def test_ae18_only_superseded_leftovers_suggest_nothing(
        self, tools, ctx, sandbox
    ):
        """A duplicate export, loaded first and collapsed into the slot of the
        later one, whose file is gone: the sequence ends complete, and
        neither the assessment nor the point declaration names a call or an
        action for it."""
        await _declare_point(tools, ctx)
        duplicate = _place(sandbox, "dup_m01.csv", _noise(9001), _companion(index=1))
        dup = await tools["load_signal"](ctx=ctx, filepath=duplicate)
        dup_id = dup.measurement["measurement_id"]
        await _load_measurements(tools, ctx, sandbox, 11)
        await _declare_point(tools, ctx, machine_group=1)
        (sandbox / duplicate).unlink()
        first = await _assess(tools, ctx, reprocess=True)
        assert first.reprocess["sequence_state"] == "continue"
        assert first.reprocess["next_cursor"] is None  # nothing failed yet
        final = await _assess(tools, ctx, reprocess=True)
        block = final.reprocess
        assert block["sequence_state"] == "complete"
        assert block["superseded"] == 1 and block["unresolved"] == 0
        assert block["groups"]["superseded"]["measurement_ids"] == [dup_id]
        assert block["needs_action"] == [] and block["next_call"] is None
        assert final.status == "assessed" and final.lineage["stale_context"] == 0
        assert "reprocess=True" not in final.message
        declared = await _declare_point(tools, ctx, machine_group=1)
        assert declared.appended is False
        assert declared.measurements_with_stale_context == 1
        assert declared.remedy is None
        assert "reprocess=True" not in declared.message
        assert "1 is a recording collapsed into another acquisition slot" in (
            declared.message
        )
        assert_runtime_wording(final.message, block["message"], declared.message)


# ---------------------------------------------------------------------------
# Closed-oracle refusals on every tool
# ---------------------------------------------------------------------------


class TestIdRefusals:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", BAD_IDS)
    @pytest.mark.parametrize("tool_name", TOOL_NAMES)
    async def test_traversal_and_reserved_asset_ids(
        self, tools, ctx, sandbox, tool_name, bad
    ):
        """A traversal or reserved asset id is refused by every tool before
        any ledger is read, with a closed oracle: the message names the
        offending input and the grammar, never another asset or file."""
        await _load_measurements(tools, ctx, sandbox, 1, asset="SECRET-ASSET")
        kwargs = {
            "declare_measurement_point": {"measurement_point_id": POINT},
            "declare_healthy_baseline": {
                "measurement_point_id": POINT,
                "measurement_ids": ["0123456789abcdef"],
                "declared_by": "L. Rossi",
            },
            "get_asset_history": {},
            "assess_asset_change": {"measurement_point_id": POINT},
        }[tool_name]
        with pytest.raises(ValueError) as exc_info:
            await tools[tool_name](ctx=ctx, asset_id=bad, **kwargs)
        message = str(exc_info.value)
        assert bad in message
        assert "SECRET-ASSET" not in message
        assert "available" not in message
        assert not (get_ledger_dir() / f"{bad}.jsonl").exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name", TOOL_NAMES)
    async def test_bad_point_ids(self, tools, ctx, sandbox, tool_name):
        kwargs = {
            "declare_measurement_point": {},
            "declare_healthy_baseline": {
                "measurement_ids": ["0123456789abcdef"],
                "declared_by": "L. Rossi",
            },
            "get_asset_history": {},
            "assess_asset_change": {},
        }[tool_name]
        with pytest.raises(ValueError, match="measurement_point_id"):
            await tools[tool_name](
                ctx=ctx, asset_id=ASSET, measurement_point_id="../x", **kwargs
            )
