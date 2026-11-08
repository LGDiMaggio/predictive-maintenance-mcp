"""Compatibility goldens of the Signal Import Engine (plan U1).

The import adds a new route in front of the signal store and the asset
ledger. It must leave what the existing route produces unchanged: the
``load_signal`` result, the ``measurement_id`` of a file, the declaration the
ledger fingerprints and the snapshot ``context_digest``. This module freezes
those values on three deterministic fixtures, before any shared file
changes, and every later unit of the import keeps them green:

- ``csv_companion``: a headerless single-column CSV whose companion carries
  ``sampling_rate``, ``signal_unit`` and a ``measurement`` object, recorded
  against a declared measurement point;
- ``raw_float32_companion``: a two-channel interleaved float32 file whose
  companion declares the raw format (channel 1, a scale factor) and a
  ``measurement`` object on an undeclared point;
- ``npy_plain``: an NPY file loaded with an explicit rate and unit and no
  companion (nothing is recorded).

Every sample value is an integer number of thousandths written with Python's
correctly rounded formatting, so the file bytes, and with them the content
digests and measurement ids, are the same on every platform and dependency
stack (no random generator stream, no vectorized ``sin``).

Comparison rules:

- every recorded value is compared exactly, type included;
- ``load_timestamp`` is not recorded and ``filepath`` is recorded relative
  to the data directory;
- the two lineage fields of the returned ``measurement`` block,
  ``processing_id`` and ``snapshot_id``, are NOT frozen: the golden holds
  :data:`LINEAGE_PLACEHOLDER` and the test recomputes them from the current
  :func:`snapshot.processing_id` and the frozen ``measurement_id`` and
  ``context_digest``, so a snapshot ``ALGORITHM_VERSION`` bump does not turn
  this golden red while every other value stays frozen;
- a key the golden does not hold is allowed only when
  :data:`ALLOWED_NEW_KEYS` names it with the only value it may take (the
  store-only import unit adds ``import_provenance: None`` there).

Fixture history:
- U1 (2026-11-02): initial capture on ``main`` 605036f.
- Store-only import (2026-11-07): not regenerated; ``import_provenance``
  (always None on the load route) added to :data:`ALLOWED_NEW_KEYS`.

Regenerate ON PURPOSE only (never to make a red test green), adding a line
to the history above. From the repo root, with the checkout's interpreter
(``import conftest`` pins the import to THIS tree, also inside a git
worktree):

    python -c "import sys; sys.path.insert(0, 'tests'); import conftest; \\
               from test_import_compat_golden import write_import_compat_golden; \\
               print(write_import_compat_golden())"
"""

import asyncio
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import numpy as np
import pytest

from conftest import patch_data_dir
from predictive_maintenance_mcp.asset_ledger import snapshot
from predictive_maintenance_mcp.asset_ledger.service import (
    build_declaration,
    declaration_fingerprint,
)
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    make_event,
)
from predictive_maintenance_mcp.config import get_ledger_dir
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

#: The golden fixture (see the module docstring for the recipe).
GOLDEN_FILE = Path(__file__).parent / "fixtures" / "import_compat_golden.json"

#: Name of the regeneration entry point, named in every failure message.
RECIPE = "write_import_compat_golden"

#: Value the golden holds for a lineage field that is recomputed at test time.
LINEAGE_PLACEHOLDER = "<recomputed at test time>"

#: Keys the golden does not hold that the comparison accepts, each mapped to
#: the only value it may take. Paths are dotted from the case's root.
ALLOWED_NEW_KEYS: dict[str, Any] = {"load_signal.import_provenance": None}

ASSET = "P-101"
CSV_POINT = "motor_de_h"
RAW_POINT = "pump_nde_v"
ACQUIRED_AT = "2026-08-20T13:42:00+02:00"
CSV_FS = 10000
RAW_FS = 25600
NPY_FS = 5000


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


def _thousandths(count: int, multiplier: int, offset: int = 0) -> list[int]:
    """A deterministic sawtooth of integers in [-1000, 1000]."""
    return [((n * multiplier + offset) % 2001) - 1000 for n in range(count)]


def _write_csv(path: Path, values: list[int]) -> None:
    """One value per line, ``k / 1000`` with three decimals, LF breaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        "".join(f"{value / 1000:.3f}\n" for value in values).encode("ascii")
    )


def _write_companion(path: Path, payload: dict[str, Any]) -> None:
    companion = path.parent / f"{path.stem}_metadata.json"
    companion.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _measurement(point: str, **extra: Any) -> dict[str, Any]:
    return {
        "asset_id": ASSET,
        "measurement_point_id": point,
        "acquired_at": ACQUIRED_AT,
        "direction": "horizontal",
        "sensor_id": "ACC01",
        "declared_by": "golden-fixture",
        **extra,
    }


def _declare_csv_point() -> None:
    """Declare the CSV fixture's point, so its snapshot has a full context."""
    payload = {
        "measurement_point_id": CSV_POINT,
        "declaration_version": 1,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": 1800.0,
        "declared_by": "golden-fixture",
        "note": None,
        "changed": [],
    }
    LedgerStore(get_ledger_dir()).append(
        ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload)
    )


def write_fixtures(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Write the three fixture files and return their ``load_signal`` kwargs."""
    csv_path = data_dir / "golden" / "m01.csv"
    _write_csv(csv_path, _thousandths(2 * CSV_FS, 7919))
    _write_companion(
        csv_path,
        {
            "sampling_rate": CSV_FS,
            "signal_unit": "g",
            "measurement": _measurement(CSV_POINT, rpm=1782, load=75),
        },
    )

    raw_path = data_dir / "golden" / "r01.bin"
    channel_0 = _thousandths(RAW_FS, 104729)
    channel_1 = _thousandths(RAW_FS, 1299709, offset=17)
    interleaved = np.empty(2 * RAW_FS, dtype="<f4")
    interleaved[0::2] = np.asarray(channel_0, dtype=np.float64) / 1000
    interleaved[1::2] = np.asarray(channel_1, dtype=np.float64) / 1000
    interleaved.tofile(raw_path)
    _write_companion(
        raw_path,
        {
            "sampling_rate": RAW_FS,
            "signal_unit": "m/s2",
            "sample_format": "float32",
            "byte_order": "little",
            "n_channels": 2,
            "channel_index": 1,
            "scale_factor": 0.5,
            "measurement": _measurement(RAW_POINT, operating_state="loaded"),
        },
    )

    npy_path = data_dir / "golden" / "n01.npy"
    np.save(
        npy_path,
        np.asarray(_thousandths(NPY_FS, 15485863), dtype=np.float64) / 1000,
    )

    return {
        "csv_companion": {"filepath": "golden/m01.csv"},
        "raw_float32_companion": {"filepath": "golden/r01.bin"},
        "npy_plain": {
            "filepath": "golden/n01.npy",
            "sampling_rate": float(NPY_FS),
            "signal_unit": "mm/s",
        },
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _ledger_payloads(event_type: str) -> list[dict[str, Any]]:
    store = LedgerStore(get_ledger_dir())
    return [
        event["payload"]
        for event in store.read(ASSET).events
        if event["event_type"] == event_type
    ]


def _relative_filepath(filepath: str, data_dir: Path) -> str:
    return Path(filepath).resolve().relative_to(data_dir.resolve()).as_posix()


async def capture_import_compat(data_dir: Path) -> dict[str, Any]:
    """Load every fixture through ``load_signal`` and capture what is frozen.

    *data_dir* must already be the data directory every ``DATA_DIR`` import
    site sees, and ``PMM_LEDGER_DIR`` an empty ledger directory.

    Returns:
        ``{"cases": {name: {"load_signal", "measurement_id", "declaration",
        "declaration_fingerprint", "context_digest"}}}``, JSON round-tripped,
        with the actual lineage values (the golden writer replaces them).
    """
    calls = write_fixtures(data_dir)
    _declare_csv_point()
    ctx = AsyncMock()
    repo = get_repository()
    cases: dict[str, Any] = {}
    try:
        for name, kwargs in calls.items():
            result = await load_signal(ctx=ctx, **kwargs)
            info = result.model_dump()
            info.pop("load_timestamp")
            info["filepath"] = _relative_filepath(info["filepath"], data_dir)
            block = info.get("measurement")
            case: dict[str, Any] = {
                "load_signal": info,
                "measurement_id": None,
                "declaration": None,
                "declaration_fingerprint": None,
                "context_digest": None,
            }
            if block is not None:
                measurement_id = block["measurement_id"]
                [recorded] = [
                    payload
                    for payload in _ledger_payloads(EVENT_MEASUREMENT_RECORDED)
                    if payload["measurement_id"] == measurement_id
                ]
                [computed] = [
                    payload
                    for payload in _ledger_payloads(EVENT_HEALTH_SNAPSHOT_COMPUTED)
                    if payload["measurement_id"] == measurement_id
                ]
                declaration = build_declaration(result.model_dump())
                assert recorded["declaration"] == declaration, (
                    "the ledger recorded another declaration than build_declaration "
                    "returns for the same load"
                )
                case.update(
                    measurement_id=measurement_id,
                    declaration=declaration,
                    declaration_fingerprint=declaration_fingerprint(
                        declaration, recorded["file"]["location"]
                    ),
                    context_digest=computed["context_digest"],
                )
            cases[name] = case
    finally:
        repo.clear_all()
    return json.loads(json.dumps({"cases": cases}, allow_nan=False))


def _with_placeholders(capture: dict[str, Any]) -> dict[str, Any]:
    golden = copy.deepcopy(capture)
    for case in golden["cases"].values():
        block = case["load_signal"].get("measurement")
        if block is not None:
            for key in ("processing_id", "snapshot_id"):
                block[key] = LINEAGE_PLACEHOLDER
    return golden


def build_import_compat_golden() -> dict[str, Any]:
    """Capture the golden in a throwaway sandbox (the regeneration path)."""
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        signals_dir = Path(tmp) / "data" / "signals"
        signals_dir.mkdir(parents=True)
        patch_data_dir(mp, signals_dir)
        mp.setenv("PMM_LEDGER_DIR", str(Path(tmp) / "ledger"))
        capture = asyncio.run(capture_import_compat(signals_dir))
    return _with_placeholders(capture)


def write_import_compat_golden() -> Path:
    """Regenerate the golden fixture ON PURPOSE (see the module docstring)."""
    GOLDEN_FILE.write_text(
        json.dumps(build_import_compat_golden(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return GOLDEN_FILE


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def resolve_lineage(golden: dict[str, Any]) -> dict[str, Any]:
    """Replace each lineage placeholder by its value recomputed NOW.

    ``processing_id`` comes from the current default policy and algorithm
    version; ``snapshot_id`` from it and the case's frozen ``measurement_id``
    and ``context_digest``.
    """
    resolved = copy.deepcopy(golden)
    current = snapshot.processing_id()
    for case in resolved["cases"].values():
        block = case["load_signal"].get("measurement")
        if block is None:
            continue
        if block.get("processing_id") == LINEAGE_PLACEHOLDER:
            block["processing_id"] = current
        if block.get("snapshot_id") == LINEAGE_PLACEHOLDER:
            block["snapshot_id"] = snapshot.snapshot_id(
                case["measurement_id"], current, case["context_digest"]
            )
    return resolved


def _same_leaf(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _compare(
    path: str,
    actual: Any,
    expected: Any,
    allowed: dict[str, Any],
    problems: list[str],
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            if key not in actual:
                problems.append(
                    f"{path}.{key}: missing (golden holds {expected[key]!r})"
                )
        for key in actual:
            child = f"{path}.{key}"
            if key not in expected:
                relative = child.split(".", 2)[-1]
                if relative not in allowed:
                    problems.append(f"{child}: new key, not in ALLOWED_NEW_KEYS")
                elif not _same_leaf(actual[key], allowed[relative]):
                    problems.append(
                        f"{child}: allowed new key holds {actual[key]!r}, "
                        f"only {allowed[relative]!r} is allowed"
                    )
            else:
                _compare(child, actual[key], expected[key], allowed, problems)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            problems.append(f"{path}: {len(actual)} items, golden {len(expected)}")
            return
        for index, (item, golden_item) in enumerate(zip(actual, expected)):
            _compare(f"{path}[{index}]", item, golden_item, allowed, problems)
        return
    if not _same_leaf(actual, expected):
        problems.append(f"{path}: {actual!r} != golden {expected!r}")


def golden_problems(
    capture: dict[str, Any],
    golden: dict[str, Any],
    allowed: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Every difference between a capture and the golden, as readable lines.

    Args:
        capture: Output of :func:`capture_import_compat`.
        golden: The golden fixture (with lineage placeholders).
        allowed: New keys accepted with their only value; defaults to
            :data:`ALLOWED_NEW_KEYS`. Paths are dotted from the case root,
            for example ``"load_signal.import_provenance"``.
    """
    problems: list[str] = []
    _compare(
        "cases",
        capture["cases"],
        resolve_lineage(golden)["cases"],
        ALLOWED_NEW_KEYS if allowed is None else allowed,
        problems,
    )
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def golden() -> dict[str, Any]:
    if not GOLDEN_FILE.exists():
        pytest.fail(f"{GOLDEN_FILE.name} is missing; regenerate it with {RECIPE}()")
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def capture(sandbox_data_dir, ledger_dir) -> dict[str, Any]:
    return asyncio.run(capture_import_compat(sandbox_data_dir))


class TestImportCompatGolden:
    """The existing route's outputs the import must not change."""

    def test_current_outputs_match_the_golden(self, capture, golden):
        problems = golden_problems(capture, golden)
        assert not problems, (
            "the existing load route no longer matches the frozen golden "
            f"(regenerate only ON PURPOSE with {RECIPE}()):\n" + "\n".join(problems)
        )

    def test_golden_covers_every_recorded_identity(self, golden):
        """The fixture is not vacuous: two recorded measurements with every
        frozen identity value, one plain load with none."""
        cases = golden["cases"]
        assert set(cases) == {"csv_companion", "raw_float32_companion", "npy_plain"}
        for name in ("csv_companion", "raw_float32_companion"):
            case = cases[name]
            assert len(case["measurement_id"]) == 16
            assert len(case["declaration_fingerprint"]) == 64
            assert len(case["context_digest"]) == 16
            block = case["load_signal"]["measurement"]
            assert block["ledger_status"] == "recorded"
            assert block["processing_id"] == LINEAGE_PLACEHOLDER
            assert block["snapshot_id"] == LINEAGE_PLACEHOLDER
        assert cases["raw_float32_companion"]["declaration"]["raw_format"] is not None
        assert cases["raw_float32_companion"]["declaration"]["channel_index"] == 1
        assert cases["npy_plain"]["load_signal"]["measurement"] is None
        assert cases["npy_plain"]["measurement_id"] is None

    @pytest.mark.parametrize(
        "path",
        [
            ("measurement_id",),
            ("declaration_fingerprint",),
            ("context_digest",),
            ("declaration", "sampling_rate"),
            ("load_signal", "num_samples"),
            ("load_signal", "measurement", "content_sha256"),
        ],
    )
    def test_a_mutated_golden_value_fails(self, capture, golden, path):
        mutated = copy.deepcopy(golden)
        target = mutated["cases"]["raw_float32_companion"]
        for key in path[:-1]:
            target = target[key]
        original = target[path[-1]]
        target[path[-1]] = (
            original + 1 if isinstance(original, (int, float)) else f"{original}0"
        )
        problems = golden_problems(capture, mutated)
        dotted = ".".join(path)
        assert any(dotted in problem for problem in problems), problems

    def test_an_int_float_swap_fails(self, capture, golden):
        """Exact comparison includes the type: 10000 is not 10000.0."""
        mutated = copy.deepcopy(golden)
        info = mutated["cases"]["csv_companion"]["load_signal"]
        info["num_samples"] = float(info["num_samples"])
        assert any("num_samples" in p for p in golden_problems(capture, mutated))

    def test_a_new_key_outside_the_allowlist_fails(self, capture, golden):
        extended = copy.deepcopy(capture)
        extended["cases"]["npy_plain"]["load_signal"]["unlisted_key"] = None
        problems = golden_problems(extended, golden)
        assert problems == [
            "cases.npy_plain.load_signal.unlisted_key: new key, not in "
            "ALLOWED_NEW_KEYS"
        ]

    def test_an_allowlisted_key_passes_only_with_its_value(self, capture, golden):
        allowed = {"load_signal.import_provenance": None}
        extended = copy.deepcopy(capture)
        for case in extended["cases"].values():
            case["load_signal"]["import_provenance"] = None
        assert golden_problems(extended, golden, allowed=allowed) == []

        extended["cases"]["csv_companion"]["load_signal"]["import_provenance"] = {}
        problems = golden_problems(extended, golden, allowed=allowed)
        assert len(problems) == 1
        assert "csv_companion.load_signal.import_provenance" in problems[0]

    def test_a_removed_key_fails(self, capture, golden):
        reduced = copy.deepcopy(capture)
        del reduced["cases"]["csv_companion"]["load_signal"]["raw_format"]
        problems = golden_problems(reduced, golden)
        assert any(
            "csv_companion.load_signal.raw_format: missing" in p for p in problems
        )

    @pytest.mark.parametrize("key", ["processing_id", "snapshot_id"])
    def test_a_wrong_lineage_value_fails(self, capture, golden, key):
        wrong = copy.deepcopy(capture)
        wrong["cases"]["csv_companion"]["load_signal"]["measurement"][key] = "0" * 16
        problems = golden_problems(wrong, golden)
        assert any(f"measurement.{key}" in problem for problem in problems), problems

    def test_an_algorithm_bump_changes_the_lineage_without_failing(
        self, sandbox_data_dir, ledger_dir, golden, monkeypatch
    ):
        """Plan U1 step 2a: a snapshot ``ALGORITHM_VERSION`` bump changes both
        lineage fields and the golden still holds."""
        before = resolve_lineage(golden)["cases"]["csv_companion"]["load_signal"]
        monkeypatch.setattr(
            snapshot, "ALGORITHM_VERSION", snapshot.ALGORITHM_VERSION + 100
        )

        bumped = asyncio.run(capture_import_compat(sandbox_data_dir))

        after = bumped["cases"]["csv_companion"]["load_signal"]["measurement"]
        assert after["processing_id"] != before["measurement"]["processing_id"]
        assert after["snapshot_id"] != before["measurement"]["snapshot_id"]
        assert golden_problems(bumped, golden) == []
