"""Tests for ``asset_ledger.snapshot`` (U3): the derived health snapshot.

ISO 13374 Block 3. A snapshot is composed from the pure engines of the
codebase (``extract_time_domain_features``; ``amplitude_spectrum`` plus
``amplitude_near_frequency`` on the leading segment for the 1x amplitude;
``check_all_bearing_faults`` / ``check_frequency_set`` plus the envelope
amplitude at every expected frequency; ``assess_severity_raw``) and is
PARTIAL whenever the context is: every absent block carries ``{reason,
remedy}`` under ``missing``. Two snapshots are comparable only with the same
``processing_id``.

Scenarios pinned here, from the plan's U3 list:

- happy path on ``golden_bearing`` (catalog 6205 at 1800 rpm, group 2
  rigid, horizontal): indicators as the feature extractor, 1x at 30 Hz,
  BPFO detected, ISO with the declared direction, ``missing`` empty;
- 1x amplitude on a pure sinusoid: value, target, the leading segment and
  agreement with ``analyze_fft`` on the same segment (same convention);
- partial composition: no rpm, no bearing, declared fault orders (through
  ``check_frequency_set``), no unit, no machine group / support;
- error paths that become ``missing`` entries, never exceptions: fs too
  low for ISO, a designation outside the catalog, an unusable envelope
  band, a machine below the ISO power scope;
- lineage (``processing_id``), context digest, provenance, determinism,
  ``snapshot_id``, module purity;
- the golden tripwire: three canonical snapshots with fixed provenance in
  ``tests/fixtures/asset_snapshot_golden.json``, keyed by
  ``ALGORITHM_VERSION``.

Golden fixture
--------------
The fixture pins the numbers of the three snapshots along the indicator
paths (``signal_processing/features.py``, ``signal_processing/spectral.py``,
``diagnostics/bearing_analyzer.py``, ``diagnostics/iso20816.py``). A
numeric change along those paths without a bump of ``ALGORITHM_VERSION``
goes red here; a bump without regeneration goes red too (no golden exists
for the new version); a changed default policy changes ``processing_id``
and goes red naming the recipe. Numbers are compared with ``pytest.approx``
(as ``tests/test_golden_merges.py`` does: FFT-derived floats may differ by
a few ulps across the CI matrix) and the dict SHAPE must match exactly (no
key added, removed or renamed silently).

Fixture history:
- U3 (2026-09-08): initial capture, ALGORITHM_VERSION 1, three cases
  (catalog bearing on a group 2 rigid point; fault orders on a group 1
  flexible point with the rpm from the point; partial snapshot without rpm
  and without a point declaration).

Regenerate ON PURPOSE only (never to make a red test green), adding a line
to the history above and, when the numbers changed, bumping
``ALGORITHM_VERSION`` first. From the repo root, with the checkout's
interpreter (``import conftest`` pins the import to THIS tree, also inside
a git worktree):

    python -c "import sys; sys.path.insert(0, 'tests'); import conftest; \\
               from test_asset_ledger_snapshot import write_asset_snapshot_golden; \\
               print(write_asset_snapshot_golden())"
"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from scipy.fft import fft, fftfreq

from predictive_maintenance_mcp.asset_ledger import snapshot as s
from predictive_maintenance_mcp.asset_ledger.snapshot import (
    BEARING_LABELS,
    SNAPSHOT_BLOCKS,
    SNAPSHOT_PROVENANCE_KEYS,
    SnapshotPolicy,
    collect_provenance,
    compute_health_snapshot,
    context_digest,
    expected_frequencies,
    policy_params,
    processing_id,
    resolve_context,
    snapshot_id,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
    check_frequency_set,
)
from predictive_maintenance_mcp.diagnostics.bearing_catalog import (
    compute_fault_frequencies,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import assess_severity_raw
from predictive_maintenance_mcp.mcp_tools.analysis_tools import analyze_fft
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_processing.features import (
    extract_time_domain_features,
)
from predictive_maintenance_mcp.signal_processing.spectral import (
    amplitude_near_frequency,
    envelope_spectrum_arrays,
)

from _golden_signals import BPFO_6205_1800, FS, golden_signals, load_golden_signals

#: The golden tripwire fixture (see the module docstring for the recipe).
GOLDEN_FILE = Path(__file__).parent / "fixtures" / "asset_snapshot_golden.json"

#: Name of the regeneration entry point; every golden failure message
#: names it so the reader finds the recipe instead of editing the fixture.
RECIPE = "write_asset_snapshot_golden"

#: Fixed provenance of the golden snapshots: nothing host-dependent.
PROVENANCE_OVERRIDES: dict[str, str] = {
    "platform": "golden-fixture",
    "python_version": "0.0.0",
    "numpy_version": "0.0.0",
    "scipy_version": "0.0.0",
    "pipeline_version": "0.0.0",
}

#: Shaft speed of the golden bearing fixture and its shaft frequency.
RPM = 1800.0
SHAFT_HZ = RPM / 60.0

#: The historical (500, 4999) envelope band the fs-aware default resolves
#: to at 10 kHz; used when calling the engines directly for comparison.
BAND_10K = (500.0, 4999.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def declaration(**overrides: Any) -> dict[str, Any]:
    """Effective declaration of a golden_bearing-like measurement.

    The identity block as the repository normalizes it (``rpm``,
    ``direction``) plus ``signal_unit``, ``sampling_rate`` and
    ``raw_format`` as stored on the signal. Override with ``None`` to mean
    "not declared".
    """
    base: dict[str, Any] = {
        "rpm": RPM,
        "direction": "horizontal",
        "signal_unit": "g",
        "sampling_rate": FS,
        "raw_format": None,
    }
    base.update(overrides)
    return base


def point(**overrides: Any) -> dict[str, Any]:
    """Latest declaration of a fully declared measurement point."""
    base: dict[str, Any] = {
        "bearing_id": "6205",
        "machine_group": 2,
        "support_type": "rigid",
    }
    base.update(overrides)
    return base


def orders_6205_at_1800() -> dict[str, float]:
    """Fault orders (multiples of the shaft frequency) of a catalog 6205.

    Derived from the catalog frequencies at 1800 rpm (Hz / shaft Hz), so
    the fault-orders route of the golden fixture targets the SAME
    frequencies as the catalog route, through the other engine.
    """
    freqs = compute_fault_frequencies("6205", RPM)
    assert freqs is not None
    shaft = float(freqs["shaft_freq_hz"])
    return {label: float(freqs[label]) / shaft for label in BEARING_LABELS}


def snap(
    signal: np.ndarray,
    decl: dict[str, Any],
    pt: Optional[dict[str, Any]],
    fs: float = FS,
    policy: Optional[SnapshotPolicy] = None,
    provenance: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """``compute_health_snapshot`` with the fixed provenance of the tests."""
    return compute_health_snapshot(
        signal,
        fs,
        declaration=decl,
        point=pt,
        policy=policy,
        provenance_overrides=PROVENANCE_OVERRIDES if provenance is None else provenance,
    )


def canonical(payload: Any) -> str:
    """The ledger's canonical JSON (sorted, compact, ASCII, no NaN)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def inline_amplitude_near(
    signal: np.ndarray, fs: float, target_hz: float, tolerance_pct: float
) -> tuple[float, float]:
    """Independent re-derivation of the 1x amplitude.

    The historical inline FFT of ``analyze_fft`` written out here (Hamming
    window, ``scipy.fft.fft``, positive frequencies, ``2|X|/N``) and the
    largest bin within ``target_hz`` +/- ``tolerance_pct`` percent, without
    calling anything in ``spectral.py``.
    """
    n = len(signal)
    values = fft(signal * np.hamming(n))
    freqs = fftfreq(n, 1 / fs)
    positive = freqs > 0
    freqs, mags = freqs[positive], 2.0 * np.abs(values[positive]) / n
    inside = np.abs(freqs - target_hz) <= target_hz * tolerance_pct / 100.0
    best = int(np.argmax(mags[inside]))
    return float(mags[inside][best]), float(freqs[inside][best])


def sine(freq_hz: float, amplitude: float, seconds: float = 1.0) -> np.ndarray:
    """A pure sinusoid at the golden sampling rate."""
    t = np.arange(int(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def walk_keys(tree: Any):
    """Yield every dict key of a JSON-like tree."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield key
            yield from walk_keys(value)
    elif isinstance(tree, list):
        for item in tree:
            yield from walk_keys(item)


# ---------------------------------------------------------------------------
# Golden fixture: cases, builder, writer, comparison
# ---------------------------------------------------------------------------


def golden_cases() -> dict[str, dict[str, Any]]:
    """Inputs of the three canonical snapshots of the tripwire."""
    return {
        "catalog_bearing_group2_rigid": {
            "signal_id": "golden_bearing",
            "declaration": declaration(),
            "point": point(),
        },
        "fault_orders_group1_flexible": {
            "signal_id": "golden_bearing",
            "declaration": declaration(rpm=None, direction="vertical"),
            "point": {
                "fault_orders": orders_6205_at_1800(),
                "nominal_rpm": RPM,
                "machine_group": 1,
                "support_type": "flexible",
            },
        },
        "partial_no_rpm_no_group": {
            "signal_id": "golden_iso",
            "declaration": declaration(rpm=None),
            "point": None,
        },
    }


def build_asset_snapshot_golden() -> dict[str, Any]:
    """Compute the golden snapshots for the CURRENT ``ALGORITHM_VERSION``.

    Returns the whole fixture payload: one top-level key, the algorithm
    version, holding the lineage key, the fixed provenance and the cases
    (inputs plus the snapshot each produced).
    """
    signals = golden_signals()
    cases: dict[str, Any] = {}
    for name, case in golden_cases().items():
        snapshot = compute_health_snapshot(
            signals[case["signal_id"]],
            FS,
            declaration=case["declaration"],
            point=case["point"],
            provenance_overrides=PROVENANCE_OVERRIDES,
        )
        cases[name] = {**case, "fs": FS, "snapshot": snapshot}
    return {
        str(s.ALGORITHM_VERSION): {
            "processing_id": processing_id(),
            "provenance_overrides": dict(PROVENANCE_OVERRIDES),
            "cases": cases,
        }
    }


def write_asset_snapshot_golden() -> Path:
    """Regenerate the golden fixture ON PURPOSE (see the module docstring)."""
    GOLDEN_FILE.write_text(
        json.dumps(
            build_asset_snapshot_golden(), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return GOLDEN_FILE


def golden_for_current_version(golden: dict[str, Any]) -> dict[str, Any]:
    """The fixture entry of the current ``ALGORITHM_VERSION``, or a failure
    naming the recipe: a bumped algorithm needs a regenerated golden."""
    key = str(s.ALGORITHM_VERSION)
    if key not in golden:
        pytest.fail(
            f"no golden snapshots for ALGORITHM_VERSION={s.ALGORITHM_VERSION} "
            f"(the fixture holds version(s) {sorted(golden)}): a bumped algorithm "
            f"needs a regenerated fixture — run {RECIPE} on purpose (recipe in the "
            f"module docstring of tests/test_asset_ledger_snapshot.py)"
        )
    return golden[key]


def assert_lineage_matches_golden(stored: dict[str, Any]) -> None:
    """The current default lineage key must be the one the fixture was
    computed with; otherwise the numbers are not comparable at all."""
    current = processing_id()
    assert stored["processing_id"] == current, (
        f"processing_id drifted from the golden lineage: {stored['processing_id']} "
        f"(fixture) != {current} (current). A changed default SnapshotPolicy or a "
        f"changed ALGORITHM_VERSION changes the lineage key; if that is intended, "
        f"bump ALGORITHM_VERSION when the numbers changed and regenerate the "
        f"fixture on purpose with {RECIPE} (recipe in the module docstring)"
    )


def assert_close_tree(
    new: Any, old: Any, path: str = "", rel: float = 1e-6, abs_tol: float = 1e-6
) -> None:
    """Recursive equality of a JSON tree, numbers via ``pytest.approx``.

    Same shape on both sides (every key, no extra key), same strings, bools
    and Nones, numbers within a tolerance far below any algorithmic drift
    and above the ulp differences of the CI matrix (magnitudes the engines
    round to six decimals may flip their last digit across platforms).
    """
    hint = f"— regenerate on purpose with {RECIPE} if the change is intended"
    if isinstance(old, dict):
        assert isinstance(new, dict), f"{path}: expected dict, got {type(new).__name__}"
        assert set(new) == set(old), (
            f"{path}: keys {sorted(set(new) ^ set(old))} differ from the golden "
            f"shape {hint}"
        )
        for key, old_val in old.items():
            assert_close_tree(new[key], old_val, f"{path}.{key}", rel, abs_tol)
    elif isinstance(old, list):
        assert isinstance(new, (list, tuple)), f"{path}: expected list"
        assert len(new) == len(
            old
        ), f"{path}: length {len(new)} != golden {len(old)} {hint}"
        for index, (new_item, old_item) in enumerate(zip(new, old)):
            assert_close_tree(new_item, old_item, f"{path}[{index}]", rel, abs_tol)
    elif isinstance(old, bool) or old is None or isinstance(old, str):
        assert new == old, f"{path}: {new!r} != golden {old!r} {hint}"
    else:
        assert new == pytest.approx(
            old, rel=rel, abs=abs_tol
        ), f"{path}: {new} != golden {old} {hint}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signals() -> dict[str, np.ndarray]:
    """The deterministic golden signals (fixed seeds)."""
    return golden_signals()


@pytest.fixture(scope="module")
def bearing_snapshot(signals) -> dict[str, Any]:
    """The happy-path snapshot: golden_bearing, catalog 6205, group 2 rigid."""
    return snap(signals["golden_bearing"], declaration(), point())


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    with open(GOLDEN_FILE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def golden_repo(tmp_path_factory):
    """Repository holding the golden signals, for the analyze_fft comparison."""
    repo = get_repository()
    repo.clear_all()
    load_golden_signals(repo, tmp_path_factory.mktemp("golden"))
    yield repo
    repo.clear_all()


# ---------------------------------------------------------------------------
# Policy and lineage
# ---------------------------------------------------------------------------


class TestPolicyAndLineage:
    def test_default_policy(self):
        assert SnapshotPolicy() == (5.0, 3, "default", 1.0, 5.0)
        assert SnapshotPolicy._fields == (
            "tolerance_pct",
            "num_harmonics",
            "envelope_band_policy",
            "fft_segment_s",
            "one_x_tolerance_pct",
        )

    def test_processing_id_is_family_version_and_policy_hash(self):
        """``health_snapshot/<ALGORITHM_VERSION>+<sha256(canonical params)[:16]>``,
        re-derived here without the module's own hashing helper."""
        pid = processing_id()
        assert re.fullmatch(r"health_snapshot/\d+\+[0-9a-f]{16}", pid)
        digest = hashlib.sha256(canonical(policy_params(SnapshotPolicy())).encode())
        assert pid == f"health_snapshot/{s.ALGORITHM_VERSION}+{digest.hexdigest()[:16]}"
        assert processing_id(None) == processing_id(SnapshotPolicy()) == pid

    def test_same_policy_different_fs_same_id_different_effective(self, signals):
        """The lineage key holds policy only: a file at another sampling rate
        joins the same lineage and reports its own effective values."""
        full = snap(signals["golden_bearing"], declaration(), point())
        half = snap(signals["golden_bearing"][:5000], declaration(), point(), fs=5000.0)
        assert (
            full["processing"]["processing_id"] == half["processing"]["processing_id"]
        )
        assert full["processing"]["effective"] == {
            "envelope_band_hz": [500.0, 4999.0],
            "fft_segment_samples": 10000,
            "num_samples": 10000,
            "sampling_rate": 10000.0,
        }
        assert half["processing"]["effective"] == {
            "envelope_band_hz": [500.0, 2499.0],
            "fft_segment_samples": 5000,
            "num_samples": 5000,
            "sampling_rate": 5000.0,
        }
        assert half["missing"] == {}

    def test_different_tolerance_different_id(self):
        assert processing_id(SnapshotPolicy(tolerance_pct=6.0)) != processing_id()

    @pytest.mark.parametrize(
        "change",
        [
            {"tolerance_pct": 4.0},
            {"num_harmonics": 4},
            {"envelope_band_policy": (500.0, 4999.0)},
            {"fft_segment_s": None},
            {"fft_segment_s": 2.0},
            {"one_x_tolerance_pct": 2.0},
        ],
    )
    def test_every_policy_field_is_part_of_the_key(self, change):
        assert processing_id(SnapshotPolicy(**change)) != processing_id()

    def test_explicit_band_equal_to_the_default_is_still_another_policy(self):
        """Policy, not effective values: pinning the band explicitly is a
        different algorithm definition even where it resolves to the same
        edges the default would."""
        params = policy_params(SnapshotPolicy(envelope_band_policy=(500.0, 4999.0)))
        assert params["envelope_band_policy"] == [500.0, 4999.0]
        assert policy_params(SnapshotPolicy())["envelope_band_policy"] == "default"

    def test_algorithm_version_bump_changes_the_id(self, monkeypatch):
        before = processing_id()
        monkeypatch.setattr(s, "ALGORITHM_VERSION", s.ALGORITHM_VERSION + 1)
        after = processing_id()
        assert after != before
        assert after.startswith(f"health_snapshot/{s.ALGORITHM_VERSION}+")
        assert after.split("+")[1] == before.split("+")[1]

    def test_package_version_override_keeps_the_id(self, signals):
        """Package and library versions are provenance, never key."""
        one = snap(signals["golden_iso"], declaration(), point())
        other = snap(
            signals["golden_iso"],
            declaration(),
            point(),
            provenance={**PROVENANCE_OVERRIDES, "pipeline_version": "9.9.9"},
        )
        assert (
            one["processing"]["processing_id"] == other["processing"]["processing_id"]
        )
        assert one["processing"]["provenance"]["pipeline_version"] == "0.0.0"
        assert other["processing"]["provenance"]["pipeline_version"] == "9.9.9"

    def test_policy_params_canonical_form(self):
        """Ints and floats naming the same policy hash the same."""
        as_ints = SnapshotPolicy(tolerance_pct=5, num_harmonics=3, fft_segment_s=1)
        assert policy_params(as_ints) == {
            "tolerance_pct": 5.0,
            "num_harmonics": 3,
            "envelope_band_policy": "default",
            "fft_segment_s": 1.0,
            "one_x_tolerance_pct": 5.0,
        }
        assert processing_id(as_ints) == processing_id()
        assert (
            policy_params(SnapshotPolicy(fft_segment_s=None))["fft_segment_s"] is None
        )

    @pytest.mark.parametrize(
        "policy, message",
        [
            (SnapshotPolicy(tolerance_pct=0.0), "tolerance_pct"),
            (SnapshotPolicy(tolerance_pct=float("nan")), "tolerance_pct"),
            (SnapshotPolicy(num_harmonics=-1), "num_harmonics"),
            (SnapshotPolicy(num_harmonics=True), "num_harmonics"),
            (SnapshotPolicy(envelope_band_policy="auto"), "envelope_band_policy"),
            (SnapshotPolicy(envelope_band_policy=(5000.0, 500.0)), "high edge"),
            (SnapshotPolicy(envelope_band_policy=(0.0, 500.0)), "low edge"),
            (SnapshotPolicy(envelope_band_policy=(500.0,)), "envelope_band_policy"),
            (SnapshotPolicy(fft_segment_s=0.0), "fft_segment_s"),
            (SnapshotPolicy(one_x_tolerance_pct=-1.0), "one_x_tolerance_pct"),
        ],
    )
    def test_policy_params_rejects_out_of_range_values(self, policy, message):
        with pytest.raises(ValueError, match=message):
            policy_params(policy)

    def test_policy_params_rejects_a_non_policy(self):
        with pytest.raises(ValueError, match="SnapshotPolicy"):
            policy_params({"tolerance_pct": 5.0})  # type: ignore[arg-type]

    def test_processing_block_records_params_and_version(self, bearing_snapshot):
        processing = bearing_snapshot["processing"]
        assert set(processing) == {
            "processing_id",
            "algorithm_version",
            "params",
            "effective",
            "provenance",
        }
        assert processing["algorithm_version"] == s.ALGORITHM_VERSION
        assert processing["params"] == policy_params(SnapshotPolicy())
        assert processing["processing_id"] == processing_id()

    def test_fft_segment_policy_drives_the_effective_samples(self, signals):
        """golden_iso is 2 s: the default takes the leading second, ``None``
        the whole signal, a longer request the whole signal too."""
        default = snap(signals["golden_iso"], declaration(), point())
        whole = snap(
            signals["golden_iso"],
            declaration(),
            point(),
            policy=SnapshotPolicy(fft_segment_s=None),
        )
        longer = snap(
            signals["golden_iso"],
            declaration(),
            point(),
            policy=SnapshotPolicy(fft_segment_s=5.0),
        )
        assert default["processing"]["effective"]["fft_segment_samples"] == 10000
        assert default["processing"]["effective"]["num_samples"] == 20000
        assert whole["processing"]["effective"]["fft_segment_samples"] == 20000
        assert longer["processing"]["effective"]["fft_segment_samples"] == 20000


# ---------------------------------------------------------------------------
# Context resolution and digest
# ---------------------------------------------------------------------------


class TestResolveContextAndDigest:
    def test_rpm_from_the_measurement_wins_over_the_point(self):
        ctx = resolve_context(declaration(rpm=1800.0), point(nominal_rpm=1500.0))
        assert (ctx["rpm"], ctx["rpm_source"]) == (1800.0, "measurement")

    def test_rpm_falls_back_to_the_nominal_rpm_of_the_point(self):
        ctx = resolve_context(declaration(rpm=None), point(nominal_rpm=1500))
        assert (ctx["rpm"], ctx["rpm_source"]) == (1500.0, "point")
        assert isinstance(ctx["rpm"], float)

    def test_no_rpm_anywhere(self):
        ctx = resolve_context(declaration(rpm=None), point())
        assert (ctx["rpm"], ctx["rpm_source"]) == (None, None)
        assert resolve_context(declaration(rpm=None), None)["rpm_source"] is None

    def test_unit_and_direction_are_declared_never_guessed(self):
        """The point's expectations are not the measurement's declaration."""
        ctx = resolve_context(
            declaration(signal_unit=None, direction=None),
            point(expected_signal_unit="g", expected_direction="vertical"),
        )
        assert ctx["signal_unit"] is None
        assert ctx["direction"] is None

    def test_point_context_fields(self):
        ctx = resolve_context(
            declaration(),
            point(fault_orders=None, machine_power_kw=55, machine_group=1),
        )
        assert ctx == {
            "rpm": 1800.0,
            "rpm_source": "measurement",
            "bearing_id": "6205",
            "fault_orders": None,
            "machine_group": 1,
            "support_type": "rigid",
            "machine_power_kw": 55.0,
            "signal_unit": "g",
            "direction": "horizontal",
        }

    def test_fault_orders_are_canonical_floats(self):
        ctx = resolve_context(
            declaration(), point(fault_orders={"BPFO": 3, "GMF": 12.5})
        )
        assert ctx["fault_orders"] == {"BPFO": 3.0, "GMF": 12.5}
        assert (
            resolve_context(declaration(), point(fault_orders={}))["fault_orders"]
            is None
        )

    def test_no_point_means_nothing_mechanical(self):
        ctx = resolve_context(declaration(), None)
        assert ctx["bearing_id"] is None
        assert ctx["fault_orders"] is None
        assert ctx["machine_group"] is None
        assert ctx["support_type"] is None
        assert ctx["machine_power_kw"] is None

    @pytest.mark.parametrize(
        "decl, pt, message",
        [
            (declaration(rpm=-5), point(), "rpm"),
            (declaration(rpm=0), point(), "rpm"),
            (declaration(rpm=None), point(nominal_rpm=-1), "nominal_rpm"),
            (declaration(direction="sideways"), point(), "direction"),
            (declaration(), point(bearing_id=""), "bearing_id"),
            (declaration(), point(bearing_id=6205), "bearing_id"),
            (declaration(), point(fault_orders={"BPFO": 0}), "fault_orders"),
            (declaration(), point(fault_orders={"": 3.0}), "label"),
            (declaration(), point(fault_orders="BPFO"), "fault_orders"),
        ],
    )
    def test_malformed_declared_values_raise(self, decl, pt, message):
        """Malformed declarations are caller bugs (validated at their own
        boundary), not missing blocks."""
        with pytest.raises(ValueError, match=message):
            resolve_context(decl, pt)

    def test_digest_is_16_hex(self):
        digest = context_digest(resolve_context(declaration(), point()), declaration())
        assert re.fullmatch(r"[0-9a-f]{16}", digest)

    def test_digest_differs_when_the_same_rpm_comes_from_the_point(self):
        """Same 1800 rpm, another origin: another resolved context."""
        from_measurement = resolve_context(declaration(rpm=1800.0), point())
        from_point = resolve_context(declaration(rpm=None), point(nominal_rpm=1800.0))
        assert from_measurement["rpm"] == from_point["rpm"] == 1800.0
        assert context_digest(from_measurement, declaration()) != context_digest(
            from_point, declaration(rpm=None)
        )

    def test_digest_ignores_a_note_only_change_of_the_point(self):
        before = resolve_context(declaration(), point(note="installed 2024"))
        after = resolve_context(declaration(), point(note="re-greased 2026"))
        assert before == after
        assert context_digest(before, declaration()) == context_digest(
            after, declaration()
        )

    @pytest.mark.parametrize(
        "decl_change, point_change",
        [
            ({"rpm": 1801.0}, {}),
            ({}, {"bearing_id": "6206"}),
            ({}, {"bearing_id": None, "fault_orders": {"BPFO": 3.58}}),
            ({}, {"machine_group": 1}),
            ({}, {"support_type": "flexible"}),
            ({}, {"machine_power_kw": 75.0}),
            ({"signal_unit": "m/s2"}, {}),
            ({"raw_format": "int16"}, {}),
            ({"sampling_rate": 25600.0}, {}),
            ({"direction": "vertical"}, {}),
            ({"direction": None}, {}),
        ],
    )
    def test_digest_covers_every_resolved_input(self, decl_change, point_change):
        base_decl, base_point = declaration(), point()
        base = context_digest(resolve_context(base_decl, base_point), base_decl)
        decl = declaration(**decl_change)
        changed = context_digest(resolve_context(decl, point(**point_change)), decl)
        assert changed != base

    def test_snapshot_digest_is_over_the_computed_sampling_rate(self, signals):
        """The digest records the fs the numbers were computed at, whatever
        the declaration carries."""
        decl = declaration(sampling_rate=None)
        one = snap(signals["golden_iso"], decl, point())
        assert one["context_digest"] == context_digest(
            resolve_context(decl, point()), {**decl, "sampling_rate": FS}
        )
        assert one["context"] == resolve_context(decl, point())


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_keys_are_exactly_the_provenance_keys_and_values_are_strings(self):
        live = collect_provenance()
        assert set(live) == SNAPSHOT_PROVENANCE_KEYS
        assert SNAPSHOT_PROVENANCE_KEYS == {
            "platform",
            "python_version",
            "numpy_version",
            "scipy_version",
            "pipeline_version",
        }
        assert all(isinstance(value, str) and value for value in live.values())

    def test_overrides_are_taken_verbatim_and_partially(self):
        pinned = collect_provenance(PROVENANCE_OVERRIDES)
        assert pinned == PROVENANCE_OVERRIDES
        partial = collect_provenance({"platform": "somewhere"})
        assert partial["platform"] == "somewhere"
        assert partial["numpy_version"] == str(np.__version__)

    def test_unknown_override_key_fails_closed(self):
        with pytest.raises(ValueError, match="date"):
            collect_provenance({"date": "2026-09-08"})
        with pytest.raises(ValueError, match="git_describe"):
            collect_provenance({**PROVENANCE_OVERRIDES, "git_describe": "v0"})

    def test_pipeline_version_is_the_package_version(self):
        from predictive_maintenance_mcp import __version__

        assert collect_provenance()["pipeline_version"] == __version__

    def test_snapshot_records_the_provenance_block(self, bearing_snapshot):
        assert bearing_snapshot["processing"]["provenance"] == PROVENANCE_OVERRIDES


# ---------------------------------------------------------------------------
# Happy path: golden_bearing, catalog 6205, group 2 rigid, horizontal
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_every_block_present_and_missing_empty(self, bearing_snapshot):
        assert set(bearing_snapshot) == {
            "processing",
            "context_digest",
            "context",
            "indicators",
            "one_x",
            "bearing",
            "iso",
            "missing",
        }
        assert bearing_snapshot["missing"] == {}
        for block in SNAPSHOT_BLOCKS:
            assert bearing_snapshot[block] is not None, block

    def test_indicators_are_the_feature_extractor_in_the_declared_unit(
        self, signals, bearing_snapshot
    ):
        features = extract_time_domain_features(signals["golden_bearing"])
        indicators = bearing_snapshot["indicators"]
        assert set(indicators) == {"rms", "peak", "crest_factor", "kurtosis", "unit"}
        assert indicators["rms"] == features["rms"]
        assert indicators["peak"] == max(abs(features["max"]), abs(features["min"]))
        assert indicators["crest_factor"] == features["crest_factor"]
        assert indicators["kurtosis"] == features["kurtosis"]
        assert indicators["unit"] == "g"
        assert all(
            isinstance(indicators[k], float) for k in ("rms", "peak", "kurtosis")
        )

    def test_one_x_at_30_hz(self, signals, bearing_snapshot):
        one_x = bearing_snapshot["one_x"]
        assert set(one_x) == {
            "target_hz",
            "amplitude",
            "frequency_hz",
            "tolerance_pct",
            "bins_searched",
            "unit",
        }
        assert one_x["target_hz"] == SHAFT_HZ == 30.0
        assert 28.5 <= one_x["frequency_hz"] <= 31.5
        assert one_x["unit"] == "g"
        assert one_x["bins_searched"] == 3  # 29, 30, 31 Hz at 1 Hz resolution
        expected_amplitude, expected_hz = inline_amplitude_near(
            signals["golden_bearing"], FS, SHAFT_HZ, 5.0
        )
        assert one_x["amplitude"] == pytest.approx(expected_amplitude, rel=1e-12)
        assert one_x["frequency_hz"] == expected_hz

    def test_bpfo_detected_with_evidence(self, bearing_snapshot):
        labels = bearing_snapshot["bearing"]["labels"]
        assert tuple(labels) == BEARING_LABELS
        bpfo = labels["BPFO"]
        assert set(bpfo) == {
            "expected_hz",
            "detected",
            "evidence_strength",
            "magnitude",
            "deviation_pct",
            "harmonics_detected",
            "envelope_amplitude",
            "envelope_peak_hz",
        }
        assert bpfo["expected_hz"] == pytest.approx(BPFO_6205_1800, abs=0.01)
        assert bpfo["detected"] is True
        assert bpfo["evidence_strength"] != "none"
        assert bpfo["magnitude"] > 0
        assert abs(bpfo["deviation_pct"]) <= 5.0
        assert bpfo["harmonics_detected"]
        # The analyzer's peak is one of the bins the amplitude search saw.
        assert bpfo["envelope_amplitude"] == pytest.approx(bpfo["magnitude"], abs=1e-6)
        assert (
            abs(bpfo["envelope_peak_hz"] - bpfo["expected_hz"])
            <= 0.05 * bpfo["expected_hz"]
        )

    def test_bearing_block_identity_and_the_single_rpm_to_hz_helper(
        self, bearing_snapshot
    ):
        bearing = bearing_snapshot["bearing"]
        assert set(bearing) == {
            "source",
            "bearing_id",
            "shaft_hz",
            "catalog_source",
            "labels",
        }
        assert bearing["source"] == "catalog"
        assert bearing["bearing_id"] == "6205"
        assert bearing["shaft_hz"] == bearing_snapshot["one_x"]["target_hz"] == SHAFT_HZ
        assert isinstance(bearing["catalog_source"], str) and bearing["catalog_source"]
        freqs = compute_fault_frequencies("6205", RPM)
        for label in BEARING_LABELS:
            assert bearing["labels"][label]["expected_hz"] == float(freqs[label])

    def test_iso_block_with_the_declared_direction(self, signals, bearing_snapshot):
        iso = bearing_snapshot["iso"]
        assert set(iso) == {
            "velocity_rms_mm_s",
            "zone",
            "machine_group",
            "support_type",
            "direction",
            "operating_speed_rpm",
            "machine_power_kw",
            "frequency_range",
            "boundaries",
        }
        raw = assess_severity_raw(
            signals["golden_bearing"],
            FS,
            machine_group=2,
            support_type="rigid",
            signal_unit="g",
            operating_speed_rpm=RPM,
        )
        assert iso["velocity_rms_mm_s"] == raw["rms_velocity_mm_s"]
        assert iso["zone"] == raw["zone"] and iso["zone"] in "ABCD"
        assert iso["machine_group"] == 2
        assert iso["support_type"] == "rigid"
        assert iso["direction"] == "horizontal"
        assert iso["operating_speed_rpm"] == RPM
        assert iso["machine_power_kw"] is None
        assert iso["boundaries"] == raw["boundaries"]
        assert iso["frequency_range"] == raw["frequency_range"]

    def test_bearing_verdicts_are_the_engines_own(self, signals, bearing_snapshot):
        """The snapshot never re-decides: per label the verdict equals
        ``check_all_bearing_faults`` called with the policy's parameters."""
        engine = check_all_bearing_faults(
            signals["golden_bearing"],
            FS,
            bearing_id="6205",
            rpm=RPM,
            tolerance_pct=5.0,
            envelope_freq_range=BAND_10K,
            num_harmonics=3,
        )
        labels = bearing_snapshot["bearing"]["labels"]
        for check in engine["fault_checks"]:
            label = labels[check["fault_type"]]
            assert label["detected"] == check["detected"]
            assert label["evidence_strength"] == check["evidence_strength"]
            assert label["magnitude"] == check["magnitude"]
            assert label["deviation_pct"] == check["deviation_pct"]
            assert label["harmonics_detected"] == check["harmonics_detected"]

    def test_context_block_and_digest(self, bearing_snapshot):
        assert bearing_snapshot["context"] == resolve_context(declaration(), point())
        assert bearing_snapshot["context_digest"] == context_digest(
            bearing_snapshot["context"], declaration()
        )


# ---------------------------------------------------------------------------
# 1x amplitude
# ---------------------------------------------------------------------------


class TestOneX:
    def test_sine_at_1500_rpm_reads_its_amplitude_in_the_analyze_fft_convention(self):
        """A 2.0-amplitude sinusoid at 25 Hz with rpm 1500 (shaft 25 Hz).

        The single-sided spectrum is Hamming-windowed with no coherent-gain
        correction, deliberately (``amplitude_spectrum``), so the 1x reads
        ``2.0 x mean(hamming)`` (about 1.08), not 2.0: the SAME number
        ``analyze_fft`` reports for this signal, which is the property the
        ledger needs (comparable with every FFT amplitude the server ever
        emitted). Within 5% of that expectation, and much tighter.
        """
        one = snap(sine(25.0, 2.0), declaration(rpm=1500.0), None)
        one_x = one["one_x"]
        expected = 2.0 * float(np.mean(np.hamming(10000)))
        assert one_x["target_hz"] == 25.0
        assert one_x["frequency_hz"] == 25.0
        assert abs(one_x["amplitude"] - expected) / expected < 0.05
        assert one_x["amplitude"] == pytest.approx(expected, rel=1e-6)
        assert 1.0 < one_x["amplitude"] < 1.2  # not 2.0: the window gain, uncorrected
        assert one_x["unit"] == "g"

    def test_sine_at_3000_rpm_reads_nothing_at_50_hz(self):
        one = snap(sine(25.0, 2.0), declaration(rpm=3000.0), None)
        one_x = one["one_x"]
        assert one_x["target_hz"] == 50.0
        assert 47.5 <= one_x["frequency_hz"] <= 52.5
        assert one_x["bins_searched"] == 5  # 48..52 Hz
        assert one_x["amplitude"] < 1e-3  # Hamming leakage 25 bins away

    @pytest.mark.asyncio
    async def test_one_x_equals_the_analyze_fft_peak_on_the_same_segment(
        self, signals, golden_repo
    ):
        """golden_iso (50 Hz sine, 2 s) at rpm 3000: the 1x of the snapshot
        IS the peak ``analyze_fft`` reports on its default leading second."""
        result = await analyze_fft(ctx=AsyncMock(), signal_id="golden_iso")
        one = snap(signals["golden_iso"], declaration(rpm=3000.0), None)
        assert (
            result.num_samples == one["processing"]["effective"]["fft_segment_samples"]
        )
        assert result.peak_frequency == one["one_x"]["frequency_hz"] == 50.0
        assert one["one_x"]["amplitude"] == pytest.approx(
            result.peak_magnitude, rel=1e-12
        )

    @pytest.mark.asyncio
    async def test_sine_one_x_equals_the_analyze_fft_peak(self, golden_repo, tmp_path):
        signal = sine(25.0, 2.0, seconds=3.0)
        path = tmp_path / "sine25.csv"
        pd.DataFrame(signal).to_csv(path, index=False, header=False)
        golden_repo.load_signal(
            str(path),
            signal_id="sine25",
            sampling_rate=FS,
            signal_unit="g",
            overwrite=True,
        )
        result = await analyze_fft(ctx=AsyncMock(), signal_id="sine25")
        one = snap(signal, declaration(rpm=1500.0), None)
        assert result.num_samples == 10000
        assert result.peak_frequency == one["one_x"]["frequency_hz"] == 25.0
        assert one["one_x"]["amplitude"] == pytest.approx(
            result.peak_magnitude, rel=1e-12
        )

    def test_one_x_is_computed_on_the_leading_segment(self):
        """First second at amplitude 2.0, second second at 0.5: the default
        policy sees only the leading second; ``fft_segment_s=None`` sees both."""
        signal = np.concatenate([sine(25.0, 2.0), sine(25.0, 0.5)])
        leading = snap(signal, declaration(rpm=1500.0), None)
        whole = snap(
            signal,
            declaration(rpm=1500.0),
            None,
            policy=SnapshotPolicy(fft_segment_s=None),
        )
        expected_leading = 2.0 * float(np.mean(np.hamming(10000)))
        assert leading["one_x"]["amplitude"] == pytest.approx(
            expected_leading, rel=1e-6
        )
        assert leading["processing"]["effective"]["fft_segment_samples"] == 10000
        assert whole["processing"]["effective"]["fft_segment_samples"] == 20000
        assert whole["one_x"]["amplitude"] < leading["one_x"]["amplitude"]

    def test_target_between_bins_is_reported_not_raised(self):
        """A 0.1 s segment has 10 Hz bins: 25 Hz +/- 5% holds no bin."""
        one = snap(
            sine(25.0, 2.0),
            declaration(rpm=1500.0),
            None,
            policy=SnapshotPolicy(fft_segment_s=0.1),
        )
        assert one["one_x"] == {
            "target_hz": 25.0,
            "amplitude": 0.0,
            "frequency_hz": None,
            "tolerance_pct": 5.0,
            "bins_searched": 0,
            "unit": "g",
        }
        assert "one_x" not in one["missing"]

    def test_one_x_tolerance_policy_widens_the_window(self):
        wide = snap(
            sine(25.0, 2.0),
            declaration(rpm=1500.0),
            None,
            policy=SnapshotPolicy(one_x_tolerance_pct=20.0),
        )
        assert wide["one_x"]["tolerance_pct"] == 20.0
        assert wide["one_x"]["bins_searched"] == 11  # 20..30 Hz


# ---------------------------------------------------------------------------
# Partial composition
# ---------------------------------------------------------------------------


class TestPartialComposition:
    def test_no_rpm_and_no_nominal_rpm(self, signals):
        one = snap(signals["golden_bearing"], declaration(rpm=None), point())
        assert one["one_x"] is None and one["bearing"] is None
        assert set(one["missing"]) == {"one_x", "bearing"}
        for block in ("one_x", "bearing"):
            entry = one["missing"][block]
            assert set(entry) == {"reason", "remedy"}
            assert "rpm not declared" in entry["reason"]
            assert "companion" in entry["remedy"]
            assert "declare_measurement_point" in entry["remedy"]
            assert "nominal_rpm" in entry["remedy"]
        # ISO needs no shaft speed: unit, group and support are declared.
        assert one["iso"] is not None
        assert one["iso"]["operating_speed_rpm"] is None
        assert one["iso"]["direction"] == "horizontal"
        assert one["context"]["rpm_source"] is None

    def test_no_bearing_and_no_orders(self, signals):
        one = snap(signals["golden_bearing"], declaration(), point(bearing_id=None))
        assert one["bearing"] is None
        assert set(one["missing"]) == {"bearing"}
        entry = one["missing"]["bearing"]
        assert "no bearing declared" in entry["reason"]
        assert "bearing_id" in entry["remedy"] and "fault_orders" in entry["remedy"]
        assert "declare_measurement_point" in entry["remedy"]
        assert one["one_x"] is not None and one["iso"] is not None

    def test_point_never_declared(self, signals):
        one = snap(signals["golden_bearing"], declaration(), None)
        assert set(one["missing"]) == {"bearing", "iso"}
        assert one["one_x"]["target_hz"] == SHAFT_HZ
        assert one["indicators"]["unit"] == "g"

    def test_fault_orders_route_goes_through_check_frequency_set(
        self, signals, monkeypatch
    ):
        orders = orders_6205_at_1800()
        calls: list[dict[str, Any]] = []
        catalog_calls: list[Any] = []
        real = s.check_frequency_set

        def spy(signal, fs, **kwargs):
            calls.append(kwargs)
            return real(signal, fs, **kwargs)

        monkeypatch.setattr(s, "check_frequency_set", spy)
        monkeypatch.setattr(
            s, "check_all_bearing_faults", lambda *a, **k: catalog_calls.append(k)
        )
        one = snap(
            signals["golden_bearing"],
            declaration(),
            point(
                bearing_id=None,
                fault_orders=orders,
                machine_group=1,
                support_type="flexible",
            ),
        )
        assert one["missing"] == {}
        assert catalog_calls == []
        assert len(calls) == 1
        assert calls[0]["frequencies"] == {
            label: order * SHAFT_HZ for label, order in orders.items()
        }
        assert calls[0]["rpm"] == RPM
        assert calls[0]["tolerance_pct"] == 5.0
        assert calls[0]["num_harmonics"] == 3
        assert calls[0]["envelope_freq_range"] == BAND_10K

        bearing = one["bearing"]
        assert bearing["source"] == "fault_orders"
        assert bearing["bearing_id"] is None
        assert bearing["catalog_source"] is None
        assert bearing["shaft_hz"] == SHAFT_HZ
        assert tuple(bearing["labels"]) == BEARING_LABELS
        for label, order in orders.items():
            assert bearing["labels"][label]["expected_hz"] == order * SHAFT_HZ
        assert bearing["labels"]["BPFO"]["detected"] is True
        assert one["iso"]["machine_group"] == 1
        assert one["iso"]["support_type"] == "flexible"

    def test_fault_orders_verdict_equals_the_engine(self, signals):
        orders = {"BPFO": 3.5, "GMF": 12.0}
        one = snap(
            signals["golden_bearing"],
            declaration(),
            point(bearing_id=None, fault_orders=orders),
        )
        engine = check_frequency_set(
            signals["golden_bearing"],
            FS,
            frequencies={label: order * SHAFT_HZ for label, order in orders.items()},
            rpm=RPM,
            tolerance_pct=5.0,
            envelope_freq_range=BAND_10K,
            num_harmonics=3,
        )
        for check in engine["fault_checks"]:
            label = one["bearing"]["labels"][check["fault_type"]]
            assert label["detected"] == check["detected"]
            assert label["evidence_strength"] == check["evidence_strength"]
            assert label["magnitude"] == check["magnitude"]

    def test_catalog_designation_takes_precedence_over_orders(self, signals):
        one = snap(
            signals["golden_bearing"], declaration(), point(fault_orders={"GMF": 12.0})
        )
        assert one["bearing"]["source"] == "catalog"
        assert tuple(one["bearing"]["labels"]) == BEARING_LABELS

    @pytest.mark.parametrize(
        "change",
        [
            {"machine_group": None},
            {"support_type": None},
            {"machine_group": None, "support_type": None},
        ],
    )
    def test_unit_declared_but_group_or_support_none(self, signals, change):
        one = snap(signals["golden_bearing"], declaration(), point(**change))
        assert one["iso"] is None
        assert set(one["missing"]) == {"iso"}
        entry = one["missing"]["iso"]
        assert "machine group / support" in entry["reason"]
        assert "declare_measurement_point" in entry["remedy"]
        assert one["one_x"] is not None and one["bearing"] is not None

    def test_unit_none(self, signals):
        one = snap(signals["golden_bearing"], declaration(signal_unit=None), point())
        assert one["iso"] is None
        assert set(one["missing"]) == {"iso"}
        assert "unit not declared" in one["missing"]["iso"]["reason"]
        assert "signal_unit" in one["missing"]["iso"]["remedy"]
        assert one["indicators"]["unit"] is None
        assert one["one_x"]["unit"] is None
        assert one["bearing"] is not None  # the envelope needs no unit

    def test_indicators_stay_in_the_declared_unit_without_conversion(self, signals):
        """golden_vel3 is a 3.0 mm/s RMS sine declared in mm/s: the rms is
        the raw 3.0, never converted to another unit here."""
        one = snap(signals["golden_vel3"], declaration(signal_unit="mm/s"), point())
        assert one["indicators"]["unit"] == "mm/s"
        assert one["indicators"]["rms"] == pytest.approx(3.0, rel=1e-6)
        assert one["one_x"]["unit"] == "mm/s"
        assert one["iso"]["velocity_rms_mm_s"] == pytest.approx(3.0, rel=0.02)

    @pytest.mark.parametrize("direction", [None, "axial", "x"])
    def test_iso_direction_is_the_declared_one_never_a_default(
        self, signals, direction
    ):
        one = snap(signals["golden_iso"], declaration(direction=direction), point())
        assert one["iso"]["direction"] == direction
        assert one["context"]["direction"] == direction

    def test_missing_entries_and_absent_blocks_are_consistent(self, signals):
        cases = [
            snap(signals["golden_bearing"], declaration(), point()),
            snap(signals["golden_bearing"], declaration(rpm=None), point()),
            snap(signals["golden_bearing"], declaration(signal_unit=None), None),
            snap(signals["golden_iso"], declaration(rpm=None, signal_unit=None), None),
            snap(signals["golden_bearing"][:2000], declaration(), point(), fs=2000.0),
        ]
        for one in cases:
            assert set(one["missing"]) <= set(SNAPSHOT_BLOCKS)
            for block in SNAPSHOT_BLOCKS:
                assert (one[block] is None) == (block in one["missing"]), block
            for entry in one["missing"].values():
                assert set(entry) == {"reason", "remedy"}
                assert entry["reason"] and entry["remedy"]
            assert one["indicators"]["rms"] > 0


# ---------------------------------------------------------------------------
# Error paths that become ``missing`` entries, never exceptions
# ---------------------------------------------------------------------------


class TestErrorPathsBecomeMissing:
    def test_fs_2000_iso_refused_by_the_engine_rest_complete(self, signals):
        one = snap(signals["golden_bearing"][:2000], declaration(), point(), fs=2000.0)
        assert one["iso"] is None
        assert set(one["missing"]) == {"iso"}
        reason = one["missing"]["iso"]["reason"]
        assert "fs=2000" in reason and "2106" in reason  # the engine's own words
        assert "re-acquire" in one["missing"]["iso"]["remedy"]
        assert one["one_x"]["target_hz"] == SHAFT_HZ
        assert one["bearing"]["source"] == "catalog"
        assert one["processing"]["effective"]["envelope_band_hz"] == [500.0, 999.0]

    def test_unknown_bearing_is_a_catalog_reason_not_an_exception(self, signals):
        one = snap(
            signals["golden_bearing"], declaration(), point(bearing_id="NOPE-999")
        )
        assert one["bearing"] is None
        assert set(one["missing"]) == {"bearing"}
        entry = one["missing"]["bearing"]
        assert (
            "NOPE-999" in entry["reason"]
            and "not found in the catalog" in entry["reason"]
        )
        assert "search_bearing_catalog" in entry["remedy"]
        assert "fault_orders" in entry["remedy"]
        # The shaft frequency is still known: 1x and ISO are computed.
        assert one["one_x"]["target_hz"] == SHAFT_HZ
        assert one["iso"]["zone"] in "ABCD"
        assert one["context"]["bearing_id"] == "NOPE-999"  # what was declared

    def test_unknown_bearing_does_not_fall_back_to_declared_orders(self, signals):
        """A designation that does not resolve is a declaration to fix."""
        one = snap(
            signals["golden_bearing"],
            declaration(),
            point(bearing_id="NOPE-999", fault_orders=orders_6205_at_1800()),
        )
        assert one["bearing"] is None
        assert "NOPE-999" in one["missing"]["bearing"]["reason"]

    def test_envelope_band_unusable_at_a_very_low_fs(self, signals):
        one = snap(signals["golden_bearing"][:1000], declaration(), point(), fs=1000.0)
        assert one["bearing"] is None
        assert "envelope band unusable" in one["missing"]["bearing"]["reason"]
        assert "fs=1000" in one["missing"]["bearing"]["reason"]
        assert one["processing"]["effective"]["envelope_band_hz"] is None
        assert one["iso"] is None  # fs too low for the ISO band as well
        assert one["one_x"] is not None and one["indicators"]["rms"] > 0

    def test_machine_power_below_the_iso_scope(self, signals):
        one = snap(
            signals["golden_bearing"], declaration(), point(machine_power_kw=5.0)
        )
        assert one["iso"] is None
        assert "15 kW" in one["missing"]["iso"]["reason"]
        assert one["context"]["machine_power_kw"] == 5.0
        assert one["bearing"] is not None

    def test_invalid_group_from_the_point_becomes_missing(self, signals):
        one = snap(signals["golden_bearing"], declaration(), point(machine_group=3))
        assert one["iso"] is None
        assert set(one["missing"]) == {"iso"}
        assert one["missing"]["iso"]["reason"]

    @pytest.mark.parametrize(
        "signal, fs, message",
        [
            (np.zeros((4, 4)), FS, "1-D"),
            (np.array([1.0, 2.0]), FS, "at least 3"),
            (np.array([1.0, np.nan, 2.0, 3.0]), FS, "finite"),
            (np.array([1.0, np.inf, 2.0, 3.0]), FS, "finite"),
            (np.arange(10.0), 0.0, "fs > 0"),
            (np.arange(10.0), -1.0, "fs > 0"),
        ],
    )
    def test_unusable_signal_or_fs_raises(self, signal, fs, message):
        with pytest.raises(ValueError, match=message):
            compute_health_snapshot(signal, fs, declaration=declaration(), point=None)

    def test_malformed_declaration_raises_not_missing(self, signals):
        with pytest.raises(ValueError, match="direction"):
            snap(signals["golden_iso"], declaration(direction="sideways"), None)
        with pytest.raises(ValueError, match="tolerance_pct"):
            snap(
                signals["golden_iso"],
                declaration(),
                None,
                policy=SnapshotPolicy(tolerance_pct=0),
            )


# ---------------------------------------------------------------------------
# Bearing band on a signal without modulation (golden_iso)
# ---------------------------------------------------------------------------


class TestGoldenIsoBearingBand:
    """golden_iso is a pure 50 Hz sine: the 500-4999 Hz envelope band holds
    nothing but filter residue. The indicator must still exist per label."""

    @pytest.fixture(scope="class")
    def iso_bearing(self, signals) -> dict[str, Any]:
        return snap(signals["golden_iso"], declaration(), point())

    def test_envelope_amplitude_present_and_finite_for_every_label(self, iso_bearing):
        labels = iso_bearing["bearing"]["labels"]
        assert tuple(labels) == BEARING_LABELS
        for label in BEARING_LABELS:
            amplitude = labels[label]["envelope_amplitude"]
            assert amplitude is not None
            assert isinstance(amplitude, float) and math.isfinite(amplitude)
            assert amplitude >= 0.0
            assert isinstance(labels[label]["envelope_peak_hz"], float)

    def test_envelope_amplitude_is_the_max_bin_of_the_same_envelope_spectrum(
        self, signals, iso_bearing
    ):
        env_freqs, env_mags = envelope_spectrum_arrays(
            signals["golden_iso"], FS, BAND_10K
        )
        for label, block in iso_bearing["bearing"]["labels"].items():
            hit = amplitude_near_frequency(
                env_freqs, env_mags, block["expected_hz"], 5.0
            )
            assert block["envelope_amplitude"] == hit["amplitude"], label
            assert block["envelope_peak_hz"] == hit["frequency_hz"], label

    def test_no_label_carries_a_peak_magnitude_on_the_empty_band(self, iso_bearing):
        """The analyzer rounds every peak of the residue away (``magnitude``
        None) while the envelope amplitude stays a number: the indicator
        exists where the verdict has nothing to show."""
        for label, block in iso_bearing["bearing"]["labels"].items():
            assert block["magnitude"] is None, label
            assert block["envelope_amplitude"] < 1e-3, label

    def test_verdicts_are_the_engines_own(self, signals, iso_bearing):
        """The snapshot reports the engine's verdict verbatim, also on a
        numerically empty envelope where the analyzer's relative peak picker
        (prominence over the spectrum maximum, no absolute floor) may still
        mark a fundamental within tolerance as ``detected``. Overriding it
        here would make the ledger disagree with ``check_bearing_faults`` on
        the same signal; the engine, not the snapshot, is where a floor
        belongs."""
        engine = check_all_bearing_faults(
            signals["golden_iso"],
            FS,
            bearing_id="6205",
            rpm=RPM,
            tolerance_pct=5.0,
            envelope_freq_range=BAND_10K,
            num_harmonics=3,
        )
        for check in engine["fault_checks"]:
            label = iso_bearing["bearing"]["labels"][check["fault_type"]]
            assert label["detected"] == check["detected"]
            assert label["evidence_strength"] == check["evidence_strength"]
            assert label["magnitude"] == check["magnitude"]

    def test_explicit_band_policy_is_honoured_verbatim(self, signals):
        one = snap(
            signals["golden_iso"],
            declaration(),
            point(),
            policy=SnapshotPolicy(envelope_band_policy=(1000.0, 3000.0)),
        )
        assert one["processing"]["effective"]["envelope_band_hz"] == [1000.0, 3000.0]
        assert one["processing"]["params"]["envelope_band_policy"] == [1000.0, 3000.0]
        assert one["bearing"] is not None


# ---------------------------------------------------------------------------
# Golden tripwire
# ---------------------------------------------------------------------------


class TestGolden:
    def test_fixture_holds_the_current_version_with_fixed_provenance(self, golden):
        stored = golden_for_current_version(golden)
        assert set(stored) == {"processing_id", "provenance_overrides", "cases"}
        assert stored["provenance_overrides"] == PROVENANCE_OVERRIDES
        assert set(stored["cases"]) == set(golden_cases())

    def test_cases_match_their_definitions(self, golden):
        stored = golden_for_current_version(golden)["cases"]
        for name, case in golden_cases().items():
            assert stored[name]["signal_id"] == case["signal_id"], name
            assert stored[name]["fs"] == FS, name
            assert stored[name]["declaration"] == case["declaration"], name
            assert stored[name]["point"] == case["point"], name

    def test_lineage_matches_the_golden(self, golden):
        assert_lineage_matches_golden(golden_for_current_version(golden))

    @pytest.mark.parametrize("name", sorted(golden_cases()))
    def test_snapshot_matches_the_golden(self, name, golden, signals):
        stored = golden_for_current_version(golden)
        assert_lineage_matches_golden(stored)
        case = stored["cases"][name]
        current = compute_health_snapshot(
            signals[case["signal_id"]],
            case["fs"],
            declaration=case["declaration"],
            point=case["point"],
            provenance_overrides=stored["provenance_overrides"],
        )
        assert_close_tree(current, case["snapshot"], name)
        assert canonical(current)  # JSON-canonicalizable, no NaN

    def test_the_three_cases_cover_the_three_routes(self, golden):
        cases = golden_for_current_version(golden)["cases"]
        catalog = cases["catalog_bearing_group2_rigid"]["snapshot"]
        orders = cases["fault_orders_group1_flexible"]["snapshot"]
        partial = cases["partial_no_rpm_no_group"]["snapshot"]
        assert catalog["missing"] == {} and catalog["bearing"]["source"] == "catalog"
        assert (
            catalog["iso"]["machine_group"] == 2
            and catalog["iso"]["support_type"] == "rigid"
        )
        assert orders["missing"] == {} and orders["bearing"]["source"] == "fault_orders"
        assert orders["context"]["rpm_source"] == "point"
        assert (
            orders["iso"]["machine_group"] == 1
            and orders["iso"]["support_type"] == "flexible"
        )
        assert orders["iso"]["direction"] == "vertical"
        assert set(partial["missing"]) == set(SNAPSHOT_BLOCKS)
        assert (
            partial["one_x"] is None
            and partial["bearing"] is None
            and partial["iso"] is None
        )
        assert partial["indicators"]["unit"] == "g"

    def test_fault_orders_case_targets_the_catalog_frequencies(self, golden):
        """Same expected frequencies through the other engine."""
        cases = golden_for_current_version(golden)["cases"]
        catalog = cases["catalog_bearing_group2_rigid"]["snapshot"]["bearing"]["labels"]
        orders = cases["fault_orders_group1_flexible"]["snapshot"]["bearing"]["labels"]
        for label in BEARING_LABELS:
            assert orders[label]["expected_hz"] == pytest.approx(
                catalog[label]["expected_hz"], rel=1e-12
            )
            assert orders[label]["envelope_amplitude"] == pytest.approx(
                catalog[label]["envelope_amplitude"], rel=1e-12
            )

    def test_default_tolerance_change_without_a_bump_goes_red(
        self, golden, monkeypatch
    ):
        """Tripwire: a changed default policy changes the lineage key; the
        failure names the regeneration recipe."""
        defaults = SnapshotPolicy.__new__.__defaults__
        monkeypatch.setattr(
            SnapshotPolicy.__new__, "__defaults__", (7.5,) + defaults[1:]
        )
        assert SnapshotPolicy().tolerance_pct == 7.5
        with pytest.raises(AssertionError, match=RECIPE):
            assert_lineage_matches_golden(golden_for_current_version(golden))

    def test_version_bump_without_regeneration_goes_red(self, golden, monkeypatch):
        monkeypatch.setattr(s, "ALGORITHM_VERSION", s.ALGORITHM_VERSION + 1)
        with pytest.raises(pytest.fail.Exception, match=RECIPE):
            golden_for_current_version(golden)

    def test_builder_reproduces_the_fixture(self, golden):
        """The recipe's builder yields the fixture (modulo float ulps)."""
        assert_close_tree(build_asset_snapshot_golden(), golden, "fixture")

    def test_a_drifted_number_is_reported_with_the_recipe(self, golden):
        stored = golden_for_current_version(golden)["cases"][
            "catalog_bearing_group2_rigid"
        ]
        drifted = json.loads(json.dumps(stored["snapshot"]))
        drifted["indicators"]["rms"] *= 1.01
        with pytest.raises(AssertionError, match=RECIPE):
            assert_close_tree(drifted, stored["snapshot"])
        renamed = json.loads(json.dumps(stored["snapshot"]))
        renamed["indicators"]["rms_value"] = renamed["indicators"].pop("rms")
        with pytest.raises(AssertionError, match="golden shape"):
            assert_close_tree(renamed, stored["snapshot"])


# ---------------------------------------------------------------------------
# Determinism and identifiers
# ---------------------------------------------------------------------------


class TestDeterminismAndIds:
    def test_two_calls_give_the_same_canonical_payload(self, signals):
        first = snap(signals["golden_bearing"], declaration(), point())
        second = snap(signals["golden_bearing"], declaration(), point())
        assert canonical(first) == canonical(second)
        assert first == second

    def test_payload_holds_only_json_types(self, signals):
        for one in (
            snap(signals["golden_bearing"], declaration(), point()),
            snap(signals["golden_iso"], declaration(rpm=None, signal_unit=None), None),
        ):
            assert json.loads(canonical(one)) == one
            assert "error" not in set(walk_keys(one))

    def test_snapshot_id_formula(self):
        sid = snapshot_id("m" * 16, processing_id(), "c" * 16)
        joined = f"{'m' * 16}:{processing_id()}:{'c' * 16}".encode()
        assert sid == hashlib.sha256(joined).hexdigest()[:16]
        assert re.fullmatch(r"[0-9a-f]{16}", sid)
        assert snapshot_id("m" * 16, processing_id(), "c" * 16) == sid

    def test_snapshot_id_changes_with_every_component(self):
        base = snapshot_id("m1", "p1", "c1")
        assert snapshot_id("m2", "p1", "c1") != base
        assert snapshot_id("m1", "p2", "c1") != base
        assert snapshot_id("m1", "p1", "c2") != base

    @pytest.mark.parametrize(
        "args, message",
        [
            (("", "p", "c"), "measurement_id"),
            (("m", "", "c"), "processing_id"),
            (("m", "p", ""), "context_digest"),
            (("m", None, "c"), "processing_id"),
        ],
    )
    def test_snapshot_id_rejects_empty_components(self, args, message):
        with pytest.raises(ValueError, match=message):
            snapshot_id(*args)

    def test_expected_frequencies_is_the_single_rpm_to_hz_helper(self):
        catalog = expected_frequencies(RPM, bearing_id="6205")
        assert catalog["shaft_hz"] == SHAFT_HZ and catalog["source"] == "catalog"
        assert set(catalog["frequencies"]) == set(BEARING_LABELS)
        orders = expected_frequencies(RPM, fault_orders={"GMF": 12.0})
        assert orders["frequencies"] == {"GMF": 12.0 * SHAFT_HZ}
        assert orders["source"] == "fault_orders" and orders["bearing_id"] is None
        nothing = expected_frequencies(RPM)
        assert nothing == {
            "shaft_hz": SHAFT_HZ,
            "source": None,
            "bearing_id": None,
            "frequencies": None,
            "catalog_source": None,
        }
        with pytest.raises(ValueError, match="NOPE"):
            expected_frequencies(RPM, bearing_id="NOPE")


# ---------------------------------------------------------------------------
# Module purity and package surface
# ---------------------------------------------------------------------------


class TestModulePurity:
    def test_imports_nothing_from_mcp_repository_or_models(self):
        tree = ast.parse(Path(s.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = {"repository", "models", "mcp", "mcp_tools", "fastmcp", "server"}
        for name in imported:
            assert not (set(name.split(".")) & forbidden), name
        for engine in (
            "diagnostics.bearing_analyzer",
            "diagnostics.bearing_catalog",
            "diagnostics.iso20816",
            "signal_processing.features",
            "signal_processing.spectral",
            "signal_acquisition.measurement",
        ):
            assert engine in imported, engine

    def test_package_reexports_every_public_name(self):
        from predictive_maintenance_mcp import asset_ledger

        assert s.ALGORITHM_VERSION == 1
        for name in s.__all__:
            assert getattr(asset_ledger, name) is getattr(s, name), name
