"""AE7 harness of the ledger's reference-relative bearing evidence (U7, PR E).

A deterministic tripwire on fixed seeds (Key Invariant 4), not a statistical
proof. Every acquisition goes through the real snapshot code
(``compute_health_snapshot``) into a ledger event, and ``assess_change``
runs after EVERY post-reference acquisition (automatic reference: the first
ten), as a user calling ``assess_asset_change`` after each load would see it.

- Fault-free: 20 sequences of 30 acquisitions per family, the stable recipe
  of ``tests/_synthetic_sequence.py`` with seeds other than the default and
  unit white noise. No assessment may be driven by bearing evidence. The
  amplitude rules' false alarms are counted with their denominators for the
  pre-release decision D-T1-12; they are measured, not bounded here.
- The acceptance ramp (the default sequence, BPFO modulation from M21):
  BPFO evidence reaches persistent change within three acquisitions of its
  onset, the onset included (by M23), and BPFI, BSF and FTF never do.
- A rolling-element ramp, fixed before the test ran: M01-M20 unit white
  noise, M21-M30 the AE3 rolling-element defect (impacts at 2 x BSF,
  ``tests/_bearing_scenarios.py``) over the same unit noise, its amplitude
  growing linearly from 0.1 to 1.0 of the -10 dB SNR amplitude (as the
  acceptance ramp's depth grows from 0.1 to 1.0). BSF reaches persistent
  change; no other label does.
- AE8 on real snapshots: 20 sequences of the stable recipe whose BPFO line
  is present at a constant depth from M01 on, so the reference shows it.
  The line is supported in the later acquisitions, never above the
  reference band, and no assessment is driven by bearing evidence: these
  are the sequences on which the reference band decides.

The D-T1-12 numbers and the two ramps' timelines print with:

    python -c "import sys; sys.path.insert(0, 'tests'); import conftest; \\
from test_asset_ledger_false_alarm import report; report()"
"""

import json
import math
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
import pytest

from _bearing_scenarios import (
    DEFECT_SEED,
    FAULT_FREE_SEED,
    FREQUENCIES_6205,
    RPM_6205,
    defect,
    white,
)
from _synthetic_sequence import (
    DEFAULT_SEED,
    DURATION_S,
    FS,
    INDICES_WITHOUT_RPM,
    NOMINAL_RPM,
    PROGRESSIVE_START,
    SEQUENCE_LENGTH,
    acquired_at,
    synthesize_measurement,
)
from predictive_maintenance_mcp.asset_ledger.assessment import (
    CLASSIFICATIONS,
    EVIDENCE_INDICATOR_PREFIX,
    assess_change,
)
from predictive_maintenance_mcp.asset_ledger.service import _snapshot_payload
from predictive_maintenance_mcp.asset_ledger.snapshot import (
    BEARING_LABELS,
    compute_health_snapshot,
    snapshot_id,
)
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    build_asset_view,
    make_event,
)

pytestmark = pytest.mark.slow

ASSET = "P-AE7"
POINT = "motor_de_h"

#: Size of the automatic reference window (``AssessmentParams`` default).
REFERENCE_WINDOW = 10

#: Fault-free sequences per family.
FAULT_FREE_SEQUENCES = 20

#: Seeds of the stable family: none is the acceptance sequence's default, and
#: the per-measurement seeds (seed + index) never overlap.
STABLE_SEEDS = tuple(
    DEFAULT_SEED + 1000 * k for k in range(1, FAULT_FREE_SEQUENCES + 1)
)

#: AE7: BPFO reaches persistent change within this many acquisitions of its
#: onset, the onset acquisition included.
ONSET_ACQUISITIONS = 3

#: AE8: the BPFO modulation depth of the sequences whose line the reference
#: already shows (the acceptance ramp's depth at M23).
REFERENCE_LINE_DEPTH = 0.3

#: The rolling-element ramp: the AE3 record (10 s at 12 kHz, a 6205 at its
#: CWRU speed) and the SNR of each ramp acquisition, M21 to M30.
RAMP_FS = 12000.0
RAMP_SECONDS = 10.0
RAMP_SNR_DB = tuple(-10.0 + 20.0 * math.log10(step / 10.0) for step in range(1, 11))


# ---------------------------------------------------------------------------
# Real snapshots as ledger events
# ---------------------------------------------------------------------------


def _point_event(nominal_rpm: float) -> dict[str, Any]:
    payload = {
        "measurement_point_id": POINT,
        "declaration_version": 1,
        "bearing_id": "6205",
        "fault_orders": None,
        "machine_group": 2,
        "support_type": "rigid",
        "machine_power_kw": None,
        "expected_signal_unit": "g",
        "expected_sensor_id": None,
        "expected_direction": "horizontal",
        "nominal_rpm": nominal_rpm,
        "declared_by": "false-alarm-harness",
        "note": None,
        "changed": [],
    }
    return make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload)


def _measurement_payload(index: int, fs: float, rpm: Optional[float]) -> dict[str, Any]:
    measurement_id = f"{index:016x}"
    location = f"seq_m{index:02d}.csv"
    return {
        "measurement_id": measurement_id,
        "measurement_point_id": POINT,
        "declaration_version": 1,
        "declaration": {
            "asset_id": ASSET,
            "measurement_point_id": POINT,
            "acquired_at": acquired_at(index),
            "timezone_declared": True,
            "timestamp_suspect": False,
            "rpm": rpm,
            "load": None,
            "operating_state": None,
            "sensor_id": "ACC01",
            "direction": "horizontal",
            "declared_by": "false-alarm-harness",
            "sampling_rate": fs,
            "signal_unit": "g",
            "raw_format": None,
            "channel_index": 0,
        },
        "file": {
            "location": location,
            "location_is_relative": True,
            "content_sha256": measurement_id * 4,
            "size_bytes": 10,
        },
        "signal_id": f"sig_{measurement_id}",
        "changed": [],
        "locations": [location],
    }


def ledger_events(
    signals: Sequence[np.ndarray],
    fs: float,
    rpms: Sequence[Optional[float]],
    nominal_rpm: float,
) -> list[dict[str, Any]]:
    """The point declaration, then one measurement and its real snapshot per
    signal, in acquisition order."""
    point = _point_event(nominal_rpm)
    events = [point]
    for index, (signal, rpm) in enumerate(zip(signals, rpms), start=1):
        measurement = _measurement_payload(index, fs, rpm)
        events.append(make_event(EVENT_MEASUREMENT_RECORDED, ASSET, measurement))
        snapshot = compute_health_snapshot(
            signal, fs, declaration=measurement["declaration"], point=point["payload"]
        )
        identity = snapshot_id(
            measurement["measurement_id"],
            str(snapshot["processing"]["processing_id"]),
            str(snapshot["context_digest"]),
        )
        payload = _snapshot_payload(
            identity, measurement["measurement_id"], POINT, snapshot, point["payload"]
        )
        events.append(make_event(EVENT_HEALTH_SNAPSHOT_COMPUTED, ASSET, payload))
    return events


def assess_after_each(events: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """``assess_change`` after every post-reference acquisition, by index."""
    count = (len(events) - 1) // 2
    return {
        n: assess_change(
            build_asset_view(ASSET, list(events[: 1 + 2 * n]), {}), ASSET, POINT
        )
        for n in range(REFERENCE_WINDOW + 1, count + 1)
    }


# ---------------------------------------------------------------------------
# The sequences
# ---------------------------------------------------------------------------


def stable_sequence(seed: int, depth: float = 0.0) -> list[np.ndarray]:
    return [
        synthesize_measurement(index, seed=seed, stationary_depth=depth)
        for index in range(1, SEQUENCE_LENGTH + 1)
    ]


def white_noise_sequence(sequence: int) -> list[np.ndarray]:
    return [
        white(
            np.random.default_rng((FAULT_FREE_SEED, 60, sequence, index)),
            FS,
            DURATION_S,
            NOMINAL_RPM / 60.0,
        )
        for index in range(1, SEQUENCE_LENGTH + 1)
    ]


def fault_free_runs() -> dict[str, list[dict[int, dict[str, Any]]]]:
    """Every assessment of the two fault-free families."""
    rpm = float(NOMINAL_RPM)
    families = {
        "stable": [stable_sequence(seed) for seed in STABLE_SEEDS],
        "white_noise": [
            white_noise_sequence(k) for k in range(1, FAULT_FREE_SEQUENCES + 1)
        ],
    }
    return {
        family: [
            assess_after_each(ledger_events(signals, FS, [rpm] * SEQUENCE_LENGTH, rpm))
            for signals in sequences
        ]
        for family, sequences in families.items()
    }


def reference_line_runs() -> list[dict[int, dict[str, Any]]]:
    """Every assessment of the sequences whose BPFO line is present at
    REFERENCE_LINE_DEPTH from the first acquisition on."""
    rpm = float(NOMINAL_RPM)
    return [
        assess_after_each(
            ledger_events(
                stable_sequence(seed, REFERENCE_LINE_DEPTH),
                FS,
                [rpm] * SEQUENCE_LENGTH,
                rpm,
            )
        )
        for seed in STABLE_SEEDS
    ]


def bpfo_ramp_events() -> list[dict[str, Any]]:
    """The acceptance sequence, declared as its companions declare it (ten
    measurements without rpm, resolved from the point's nominal speed)."""
    signals = [synthesize_measurement(i) for i in range(1, SEQUENCE_LENGTH + 1)]
    rpms = [
        None if i in INDICES_WITHOUT_RPM else float(NOMINAL_RPM)
        for i in range(1, SEQUENCE_LENGTH + 1)
    ]
    return ledger_events(signals, FS, rpms, float(NOMINAL_RPM))


def rolling_element_ramp_events() -> list[dict[str, Any]]:
    shaft_hz = RPM_6205 / 60.0
    signals = []
    for index in range(1, SEQUENCE_LENGTH + 1):
        rng = np.random.default_rng((DEFECT_SEED, 60, index))
        if index < PROGRESSIVE_START:
            signals.append(white(rng, RAMP_FS, RAMP_SECONDS, shaft_hz))
            continue
        snr_db = RAMP_SNR_DB[index - PROGRESSIVE_START]
        # defect() adds noise of standard deviation 10^(-snr/20) to impacts of
        # unit standard deviation; scaling by 10^(snr/20) keeps that SNR over
        # the unit noise floor of the fault-free acquisitions.
        signals.append(
            10.0 ** (snr_db / 20.0)
            * defect(
                "rolling_element",
                rng,
                RAMP_FS,
                RAMP_SECONDS,
                FREQUENCIES_6205,
                shaft_hz,
                snr_db,
            )
        )
    return ledger_events(signals, RAMP_FS, [RPM_6205] * SEQUENCE_LENGTH, RPM_6205)


# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------


def tally(runs: Sequence[dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Counts over every assessment of a family, with their denominators."""
    overall: Counter[str] = Counter()
    persistent_drivers: Counter[str] = Counter()
    label_classes: Counter[str] = Counter()
    sequences_persistent = 0
    evidence_driven: list[tuple[int, int, list[str]]] = []
    for sequence, run in enumerate(runs):
        persistent = False
        for n, result in run.items():
            assessed = result["assessed"]
            overall[assessed["classification"]] += 1
            driving = assessed["indicators_driving"]
            by_evidence = [
                d for d in driving if d.startswith(EVIDENCE_INDICATOR_PREFIX)
            ]
            if by_evidence:
                evidence_driven.append((sequence, n, by_evidence))
            if assessed["classification"] == "persistent_change":
                persistent = True
                persistent_drivers.update(driving)
            for label, evidence in assessed["evidence"].items():
                if evidence["classification"] not in (None, "no_change"):
                    label_classes[f"{label}:{evidence['classification']}"] += 1
        sequences_persistent += persistent
    return {
        "sequences": len(runs),
        "assessments": sum(overall.values()),
        "overall": {name: overall[name] for name in CLASSIFICATIONS},
        "sequences_with_persistent_change": sequences_persistent,
        "persistent_change_drivers": dict(sorted(persistent_drivers.items())),
        "evidence_classes_other_than_no_change": dict(sorted(label_classes.items())),
        "evidence_driven": evidence_driven,
    }


def timeline(run: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per assessment: the overall classification and each label's evidence."""
    return {
        n: {
            "overall": result["assessed"]["classification"],
            **{
                label: (
                    evidence["classification"],
                    evidence["present_in_last_k"],
                    evidence["relative_present_in_last_k"],
                )
                for label, evidence in result["assessed"]["evidence"].items()
            },
        }
        for n, result in run.items()
    }


def first_persistent(run: dict[int, dict[str, Any]], label: str) -> Optional[int]:
    """The first acquisition after which *label*'s evidence is persistent."""
    return next(
        (
            n
            for n, result in sorted(run.items())
            if result["assessed"]["evidence"][label]["classification"]
            == "persistent_change"
        ),
        None,
    )


def report() -> None:
    """Print the D-T1-12 tallies and the two ramps' timelines as JSON."""
    runs = fault_free_runs()
    print(
        json.dumps(
            {
                "fault_free": {family: tally(r) for family, r in runs.items()},
                "reference_line": tally(reference_line_runs()),
                "bpfo_ramp": timeline(assess_after_each(bpfo_ramp_events())),
                "rolling_element_ramp": timeline(
                    assess_after_each(rolling_element_ramp_events())
                ),
            },
            indent=1,
        )
    )


# ---------------------------------------------------------------------------
# AE7
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fault_free() -> dict[str, list[dict[int, dict[str, Any]]]]:
    return fault_free_runs()


@pytest.fixture(scope="module")
def bpfo_ramp() -> dict[int, dict[str, Any]]:
    return assess_after_each(bpfo_ramp_events())


@pytest.fixture(scope="module")
def rolling_element_ramp() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    events = rolling_element_ramp_events()
    return events, assess_after_each(events)


@pytest.mark.parametrize("family", ["stable", "white_noise"])
def test_ae7_no_fault_free_assessment_is_driven_by_bearing_evidence(
    fault_free, family, record_property
):
    runs = fault_free[family]
    stats = tally(runs)
    assert stats["sequences"] == FAULT_FREE_SEQUENCES
    assert stats["assessments"] == FAULT_FREE_SEQUENCES * (
        SEQUENCE_LENGTH - REFERENCE_WINDOW
    )
    # Every assessment classified the evidence against the reference. On these
    # families the matcher supports no line, so this checks its specificity
    # through the ledger; the reference-line family exercises the band.
    bases = {
        evidence["basis"]
        for run in runs
        for result in run.values()
        for evidence in result["assessed"]["evidence"].values()
    }
    assert bases == {"reference_relative"}
    for key in (
        "assessments",
        "overall",
        "sequences_with_persistent_change",
        "persistent_change_drivers",
    ):
        record_property(key, json.dumps(stats[key]))
    assert stats["evidence_driven"] == [], (
        f"{family}: assessments driven by bearing evidence (sequence, "
        f"acquisition, drivers): {stats['evidence_driven']}"
    )
    assert stats["evidence_classes_other_than_no_change"] == {}


@pytest.fixture(scope="module")
def reference_line() -> list[dict[int, dict[str, Any]]]:
    return reference_line_runs()


def test_ae8_a_line_the_reference_already_shows_is_not_counted(reference_line):
    """The BPFO line is supported in the reference and in the later
    acquisitions, at the same depth: it reaches the reference band, never
    rises above it, and no assessment is driven by bearing evidence."""
    stats = tally(reference_line)
    assert stats["assessments"] == FAULT_FREE_SEQUENCES * (
        SEQUENCE_LENGTH - REFERENCE_WINDOW
    )
    assert stats["evidence_driven"] == [], stats["evidence_driven"]
    assert stats["evidence_classes_other_than_no_change"] == {}
    for run in reference_line:
        for n, result in run.items():
            presence = result["observed"]["evidence_presence"]["BPFO"]
            assert presence["present_in_last_k"] > 0, n
            assert presence["reference_present"] > 0, n
            assert presence["relative_present_in_last_k"] == 0, n


def test_ae7_bpfo_persists_within_three_acquisitions_of_its_onset(bpfo_ramp):
    first = first_persistent(bpfo_ramp, "BPFO")
    assert first is not None and (
        PROGRESSIVE_START <= first < PROGRESSIVE_START + ONSET_ACQUISITIONS
    ), f"BPFO first persistent after M{first}; timeline {timeline(bpfo_ramp)}"


def test_ae7_no_other_label_persists_on_the_bpfo_ramp(bpfo_ramp):
    persisting = {
        label: first_persistent(bpfo_ramp, label)
        for label in BEARING_LABELS
        if label != "BPFO"
    }
    assert persisting == {label: None for label in persisting}, timeline(bpfo_ramp)


def test_ae7_rolling_element_ramp_persists_on_bsf_only(rolling_element_ramp):
    events, run = rolling_element_ramp
    # The rolling-element label is tested at twice the ball spin (R4).
    for event in events:
        if event["event_type"] == EVENT_HEALTH_SNAPSHOT_COMPUTED:
            block = event["payload"]["bearing"]["labels"]["BSF"]
            assert block["line_frequency_hz"] == pytest.approx(
                2.0 * block["expected_hz"], abs=1e-3
            )
    first = first_persistent(run, "BSF")
    assert (
        first is not None and first >= PROGRESSIVE_START
    ), f"BSF first persistent after M{first}; timeline {timeline(run)}"
    persisting = {
        label: first_persistent(run, label)
        for label in BEARING_LABELS
        if label != "BSF"
    }
    assert persisting == {label: None for label in persisting}, timeline(run)
