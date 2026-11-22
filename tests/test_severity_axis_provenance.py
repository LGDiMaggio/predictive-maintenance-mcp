"""The measurement direction carried by an ISO severity result, and its source.

``assess_severity`` and the assessed ``iso_severity`` block of
``diagnose_vibration`` report in ``axis`` the direction declared by the
stored signal's companion ``measurement`` object, as the loader normalized
it, and say so in ``axis_source`` ("measurement"). When no direction is
declared, and on the ``rms_velocity_mm_s`` route, ``axis`` is
"not_declared" and ``axis_source`` is "none".

The direction is informational: these tests check that every other field of
both tools is the same whatever direction is declared, and that a refused
ISO block carries no direction at all.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from examples.adapters.stwinbox import stwinbox_to_measurement as adapter
from predictive_maintenance_mcp.diagnostics.iso20816 import assess_severity_with_axis
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register
from predictive_maintenance_mcp.models import (
    ISOSeverityRefusal,
    VibrationSeverityResult,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

AXIS_DESCRIPTION = (
    "Measurement direction declared by the stored signal's companion measurement "
    "object, as normalized ('horizontal', 'vertical', 'axial', 'x', 'y' or 'z'); "
    "'not_declared' when none is declared or on the rms_velocity_mm_s route. This "
    "field is informational, does not determine the ISO severity zone, and does "
    "not establish whether the zone thresholds apply to that measurement "
    "direction."
)
AXIS_SOURCE_DESCRIPTION = (
    "Where axis comes from: 'measurement' (the companion's measurement object) or "
    "'none' (no direction declared, or the rms_velocity_mm_s route)"
)

SIGNAL_ID = "axis_probe"
FS = 10000.0

#: Every declared form of a direction and the direction it is stored as.
DECLARED_DIRECTIONS = [
    ("horizontal", "horizontal"),
    ("vertical", "vertical"),
    ("axial", "axial"),
    ("x", "x"),
    ("y", "y"),
    ("z", "z"),
    ("h", "horizontal"),
    ("v", "vertical"),
    ("a", "axial"),
]

NOT_DECLARED = ("not_declared", "none")

#: A measurement object without a direction (the field is optional).
IDENTITY = {
    "asset_id": "P-201",
    "measurement_point_id": "pump_de",
    "acquired_at": "2026-11-21T10:00:00+01:00",
}


def _velocity_signal() -> np.ndarray:
    t = np.arange(int(FS)) / FS
    noise = 0.2 * np.random.default_rng(1121).standard_normal(t.size)
    return 4.0 * np.sin(2 * np.pi * 49.5 * t) + noise


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    signals_dir = tmp_path / "data" / "signals"
    signals_dir.mkdir(parents=True)
    for module in (
        "predictive_maintenance_mcp.config",
        "predictive_maintenance_mcp.signal_acquisition.loaders",
        "predictive_maintenance_mcp.signal_acquisition.repository",
    ):
        monkeypatch.setattr(f"{module}.DATA_DIR", signals_dir)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR",
        models_dir,
    )
    return signals_dir


@pytest.fixture
def repo(data_dir):
    repository = get_repository()
    repository.clear_all()
    yield repository
    repository.clear_all()


@pytest.fixture
def tools():
    server = MCPServer("test-severity-axis")
    register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


def _store(data_dir, repo, *, direction=None, identity=True, unit="mm/s") -> str:
    """Store the velocity signal under SIGNAL_ID, replacing any earlier copy.

    ``identity=False`` writes a companion without a measurement object;
    otherwise the object carries ``direction`` when one is given.
    """
    np.savetxt(data_dir / "probe.csv", _velocity_signal(), fmt="%.10g")
    companion: dict = {"sampling_rate": FS}
    if unit is not None:
        companion["signal_unit"] = unit
    if identity:
        companion["measurement"] = dict(IDENTITY)
        if direction is not None:
            companion["measurement"]["direction"] = direction
    (data_dir / "probe_metadata.json").write_text(json.dumps(companion))
    repo.clear_all()
    repo.load_signal("probe.csv", signal_id=SIGNAL_ID)
    return SIGNAL_ID


def _axis(result) -> tuple[str, str]:
    return result.axis, result.axis_source


class TestFieldContract:
    def test_descriptions_are_the_published_texts(self):
        fields = VibrationSeverityResult.model_fields
        assert fields["axis"].description == AXIS_DESCRIPTION
        assert fields["axis_source"].description == AXIS_SOURCE_DESCRIPTION

    def test_defaults_and_vocabulary(self):
        fields = VibrationSeverityResult.model_fields
        assert fields["axis"].default == "not_declared"
        assert fields["axis_source"].default == "none"
        schema = VibrationSeverityResult.model_json_schema()["properties"]
        assert schema["axis"]["type"] == "string"
        assert schema["axis_source"]["enum"] == ["measurement", "none"]

    def test_engine_states_no_direction_by_default(self):
        result = assess_severity_with_axis(
            _velocity_signal(), fs=FS, signal_unit="mm/s"
        )
        assert result["axis"] == "not_declared"


class TestAssessSeverity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("declared, stored", DECLARED_DIRECTIONS)
    async def test_declared_direction_is_reported(
        self, tools, data_dir, repo, declared, stored
    ):
        sid = _store(data_dir, repo, direction=declared)
        result = await tools["assess_severity"](ctx=None, signal_id=sid)
        assert _axis(result) == (stored, "measurement")

    @pytest.mark.asyncio
    async def test_signal_without_measurement_object(self, tools, data_dir, repo):
        sid = _store(data_dir, repo, identity=False)
        result = await tools["assess_severity"](ctx=None, signal_id=sid)
        assert _axis(result) == NOT_DECLARED

    @pytest.mark.asyncio
    async def test_measurement_object_without_direction(self, tools, data_dir, repo):
        sid = _store(data_dir, repo)
        result = await tools["assess_severity"](ctx=None, signal_id=sid)
        assert _axis(result) == NOT_DECLARED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reading",
        [
            {"rms_velocity_mm_s": 0.9},
            {"rms_velocity_mm_s": 6.0, "machine_group": 1, "support_type": "flexible"},
            {"rms_velocity_mm_s": 2.0, "machine_power_kw": 30.0, "rpm": 1500.0},
            {
                "rms_velocity_mm_s": 4.0,
                "thresholds": {"warning": 1.0, "alarm": 3.0, "danger": 6.0},
            },
        ],
    )
    async def test_direct_reading_declares_no_direction(
        self, tools, data_dir, repo, reading
    ):
        # A stored signal declaring a direction is present and must not leak in.
        _store(data_dir, repo, direction="x")
        result = await tools["assess_severity"](ctx=None, **reading)
        assert result.signal_id is None
        assert _axis(result) == NOT_DECLARED


class TestDiagnoseVibration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "store, expected",
        [
            *(
                ({"direction": declared}, (stored, "measurement"))
                for declared, stored in DECLARED_DIRECTIONS
            ),
            ({"identity": False}, NOT_DECLARED),
            ({}, NOT_DECLARED),
        ],
    )
    async def test_assessed_block_matches_assess_severity(
        self, tools, data_dir, repo, store, expected
    ):
        sid = _store(data_dir, repo, **store)
        severity = await tools["assess_severity"](ctx=None, signal_id=sid)
        diagnosis = await tools["diagnose_vibration"](
            ctx=None, signal_id=sid, rpm=1800.0
        )
        assert isinstance(diagnosis.iso_severity, VibrationSeverityResult)
        assert _axis(diagnosis.iso_severity) == expected
        assert _axis(severity) == expected

    @pytest.mark.asyncio
    async def test_refused_block_carries_no_direction(self, tools, data_dir, repo):
        refusals = []
        for store in ({"direction": "x"}, {"identity": False}):
            sid = _store(data_dir, repo, unit=None, **store)
            diagnosis = await tools["diagnose_vibration"](
                ctx=None, signal_id=sid, rpm=1800.0
            )
            assert isinstance(diagnosis.iso_severity, ISOSeverityRefusal)
            refusals.append(diagnosis.iso_severity.model_dump(mode="json"))
        assert set(refusals[0]) == {"status", "signal_id", "reason", "remedy"}
        assert refusals[0] == refusals[1]
        assert refusals[0]["reason"].startswith("Signal unit not declared")


class TestDirectionIsInformational:
    @pytest.mark.asyncio
    async def test_every_other_field_is_identical(self, tools, data_dir, repo):
        stores = [{"identity": False}, {}] + [
            {"direction": declared} for declared, _ in DECLARED_DIRECTIONS
        ]
        severities, diagnoses, seen = [], [], set()
        for store in stores:
            sid = _store(data_dir, repo, **store)
            severity = (
                await tools["assess_severity"](ctx=None, signal_id=sid)
            ).model_dump(mode="json")
            diagnosis = (
                await tools["diagnose_vibration"](ctx=None, signal_id=sid, rpm=1800.0)
            ).model_dump(mode="json")
            seen.add((severity.pop("axis"), severity.pop("axis_source")))
            iso = diagnosis["iso_severity"]
            seen.add((iso.pop("axis"), iso.pop("axis_source")))
            severities.append(severity)
            diagnoses.append(diagnosis)
        assert seen == {NOT_DECLARED} | {
            (stored, "measurement") for _, stored in DECLARED_DIRECTIONS
        }
        assert all(s == severities[0] for s in severities)
        assert all(d == diagnoses[0] for d in diagnoses)


class TestStwinboxAdapterCompanion:
    @pytest.mark.asyncio
    async def test_sensor_axis_is_reported(self, tools, data_dir, repo):
        """A companion built by the STWIN.box reference adapter declares a sensor
        axis; the severity result reports that axis, not a machine direction."""
        rate = 26585.318359375
        t = np.arange(int(rate)) / rate
        samples = (0.05 * np.sin(2 * np.pi * 50.0 * t)).astype("<f4")
        samples.tofile(data_dir / "stwin_x.bin")
        measurement = adapter.build_measurement(
            asset_id="P-201",
            measurement_point_id="motor_de_x",
            acquired_at="2026-06-01T10:00:00+00:00",
            direction="x",
            sensor_id="0054TEST000000000000001",
            rpm=None,
            declared_by="adapter:stwinbox",
        )
        companion = adapter.build_companion(
            sampling_rate=rate, measurement=measurement, adapter={"source": "test"}
        )
        (data_dir / "stwin_x_metadata.json").write_text(json.dumps(companion))
        repo.load_signal("stwin_x.bin", signal_id="stwin_x")
        result = await tools["assess_severity"](ctx=None, signal_id="stwin_x")
        assert _axis(result) == ("x", "measurement")
