"""A synthetic matrix of today's outputs from both single-acquisition authors.

Shared by ``test_t2_characterization.py`` and ``test_t2_known_defect_tripwires.py``.
Two server functions author the interpretation of one acquisition today:
``diagnosis_pipeline._synthesize_diagnosis``, behind ``diagnose_vibration``,
and ``advisory.build_advisory``, behind ``generate_diagnostic_report``. Each
case runs one stored synthetic signal through both MCP tool functions in
memory, the report as HTML only, and keeps what each author wrote.

What this module records is CHARACTERIZATION of today's outputs, not a
requirement: a changed output means the goldens are regenerated and the change
is reviewed, never that the code is wrong. Which wording, order or
recommendation table is right is still open, and nothing here encodes a
choice.

Boundaries this module keeps:

- Synthetic signals only, from generators seeded with ``T2_SEED``; no
  recording is read.
- No anomaly model exists in any case. ``MODELS_DIR`` points, in the pipeline
  and both tool modules, at an empty sandbox folder, so the anomaly
  segmentation, hop and label code never runs. Everything the anomaly family
  authors (the advisory's anomaly block, its recommendation and any
  anomaly-model disagreement) is left out of the goldens and the parity
  predicates, on both surfaces: its meaning, labels and wording belong to the
  anomaly-model work, which can then change them without editing these files.
- The report is rendered as HTML only; no PDF is requested, so nothing here
  depends on whether the optional PDF dependencies are installed.
"""

from __future__ import annotations

import asyncio
import copy
import html
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from _bearing_scenarios import FREQUENCIES_6205, RPM_6205, defect
from conftest import patch_data_dir
from predictive_maintenance_mcp import report_generator
from predictive_maintenance_mcp.decision_support import diagnosis_pipeline
from predictive_maintenance_mcp.decision_support.advisory import build_advisory
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
)
from predictive_maintenance_mcp.mcp_tools import (
    asset_tools,
    diagnostics_tools,
    report_tools,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

#: Seed range of this matrix, used by no other test. Every random source is
#: ``np.random.default_rng((T2_SEED, n))`` with the ``n`` of its case.
T2_SEED = 2_118_000

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "t2"

#: Set to "1" to write the goldens from today's outputs instead of comparing.
WRITE_VARIABLE = "T2_CHARACTERIZATION_WRITE"

ASSET_ID = "T2-ASSET"
POINT_ID = "t2_point"
ACQUIRED_AT = "2026-11-18T09:00:00+00:00"

#: Shaft speed of the zone cases (rpm), 25 Hz.
RPM_ZONES = 1500.0

#: First words and last sentence of the shaft-order statement: it is located
#: between them in a diagnosis text, never re-typed here.
SHAFT_ORDER_FIRST_WORDS = "The strongest line of the spectrum is at "
SHAFT_ORDER_LAST_WORDS = "does not corroborate it."


def rng(n: int) -> np.random.Generator:
    return np.random.default_rng((T2_SEED, n))


def _time(fs: float, seconds: float) -> np.ndarray:
    return np.arange(int(round(fs * seconds))) / fs


def velocity_tone(
    n: Optional[int],
    *,
    freq_hz: float,
    rms_mm_s: float,
    fs: float,
    seconds: float = 2.0,
    noise_mm_s: float = 0.02,
) -> np.ndarray:
    """A velocity sine (mm/s) of the given RMS, plus seeded white noise unless *n* is None."""
    t = _time(fs, seconds)
    x = rms_mm_s * math.sqrt(2.0) * np.sin(2.0 * np.pi * freq_hz * t)
    if n is not None:
        x = x + noise_mm_s * rng(n).standard_normal(t.size)
    return x


def white(n: int, *, scale: float, fs: float, seconds: float = 2.0) -> np.ndarray:
    return scale * rng(n).standard_normal(int(round(fs * seconds)))


def shaft_tones(
    n: Optional[int], *, rpm: float, fs: float, seconds: float = 2.0
) -> np.ndarray:
    """Tones at 3x (0.5), 1x (0.3), 2x (0.15) and 50 Hz (0.1), plus 0.05 noise.

    With *n* None: only the 3x and 1x tones, no random source (a generated
    record without noise).
    """
    t = _time(fs, seconds)
    shaft = rpm / 60.0
    x = 0.5 * np.sin(2 * np.pi * 3 * shaft * t) + 0.3 * np.sin(2 * np.pi * shaft * t)
    if n is not None:
        x = (
            x
            + 0.15 * np.sin(2 * np.pi * 2 * shaft * t)
            + 0.1 * np.sin(2 * np.pi * 50.0 * t)
            + 0.05 * rng(n).standard_normal(t.size)
        )
    return x


def bearing_defect(n: int, kind: str) -> np.ndarray:
    """A seeded 6205 defect at 1797 rpm, 10 s at 12 kHz, SNR -15 dB."""
    return defect(kind, rng(n), 12000.0, 10.0, FREQUENCIES_6205, RPM_6205 / 60.0, -15.0)


#: Companion keys that declare a generated record without noise.
NOISELESS = {"generator": "t2_characterization", "noise_level": 0}


@dataclass(frozen=True)
class Case:
    """One stored signal and the parameters both tools receive."""

    case_id: str
    description: str
    samples: Callable[[], np.ndarray]
    fs: float
    rpm: float
    unit: Optional[str]
    bearing_id: Optional[str] = None
    companion: dict = field(default_factory=dict)
    #: Keyword arguments of declare_measurement_point; the companion then
    #: carries the measurement identity and diagnose_vibration gets no
    #: explicit parameter, while the report (which reads no point) gets rpm.
    point: Optional[dict] = None


def _zone(case_id: str, n: int, rms_mm_s: float, zone: str) -> Case:
    return Case(
        case_id,
        f"Velocity tone of {rms_mm_s:.2f} mm/s RMS at 157 Hz (ISO zone {zone}), "
        "no bearing",
        lambda: velocity_tone(n, freq_hz=157.0, rms_mm_s=rms_mm_s, fs=10000.0),
        10000.0,
        RPM_ZONES,
        "mm/s",
    )


CASES: dict[str, Case] = {
    c.case_id: c
    for c in (
        _zone("zone_a", 1, 0.70, "A"),
        _zone("zone_b", 2, 2.00, "B"),
        _zone("zone_c", 3, 3.60, "C"),
        _zone("zone_d", 4, 6.00, "D"),
        Case(
            "white_noise_bearing",
            "Unit-variance white noise in g, catalog bearing 6205",
            lambda: white(11, scale=1.0, fs=12000.0),
            12000.0,
            RPM_6205,
            "g",
            "6205",
        ),
        Case(
            "shaft_tones_bearing",
            "Shaft-order tones, a 50 Hz line and noise in g, catalog bearing 6205",
            lambda: shaft_tones(12, rpm=RPM_6205, fs=12000.0),
            12000.0,
            RPM_6205,
            "g",
            "6205",
        ),
        Case(
            "strongest_line_1x",
            "Strongest spectral line at 1x shaft speed, no bearing",
            lambda: velocity_tone(
                13, freq_hz=RPM_ZONES / 60.0, rms_mm_s=2.00, fs=10000.0
            ),
            10000.0,
            RPM_ZONES,
            "mm/s",
        ),
        Case(
            "strongest_line_2x_bearing",
            "Strongest spectral line at 2x shaft speed, catalog bearing 6205",
            lambda: velocity_tone(
                14, freq_hz=2.0 * RPM_6205 / 60.0, rms_mm_s=1.00, fs=12000.0
            ),
            12000.0,
            RPM_6205,
            "mm/s",
            "6205",
        ),
        Case(
            "outer_race_defect",
            "Synthetic outer-race defect of a 6205 at -15 dB SNR",
            lambda: bearing_defect(15, "outer"),
            12000.0,
            RPM_6205,
            "g",
            "6205",
        ),
        Case(
            "combined_defect",
            "Synthetic outer- and inner-race defects of a 6205 at -15 dB SNR",
            lambda: bearing_defect(16, "combined"),
            12000.0,
            RPM_6205,
            "g",
            "6205",
        ),
        Case(
            "undeclared_unit",
            "White noise with no declared unit, no bearing",
            lambda: white(17, scale=0.1, fs=10000.0),
            10000.0,
            RPM_ZONES,
            None,
        ),
        Case(
            "low_sampling_rate",
            "White noise at fs 1000 Hz, catalog bearing 6205",
            lambda: white(18, scale=0.1, fs=1000.0, seconds=4.0),
            1000.0,
            RPM_ZONES,
            "g",
            "6205",
        ),
        Case(
            "bearing_not_in_catalog",
            "White noise with a bearing designation the catalog lacks",
            lambda: white(19, scale=0.1, fs=10000.0),
            10000.0,
            RPM_ZONES,
            "g",
            "ZZ9999",
        ),
        Case(
            "point_fault_orders_only",
            "Measurement point declaring fault orders and no bearing",
            lambda: white(20, scale=0.1, fs=10000.0),
            10000.0,
            RPM_ZONES,
            "g",
            point={
                "fault_orders": {"BPFO": 3.585, "BPFI": 5.415},
                "nominal_rpm": RPM_ZONES,
                "machine_group": 2,
                "support_type": "rigid",
            },
        ),
        Case(
            "generated_without_noise",
            "Generated shaft-order tones without noise, catalog bearing 6205",
            lambda: shaft_tones(None, rpm=RPM_6205, fs=12000.0),
            12000.0,
            RPM_6205,
            "g",
            "6205",
            companion=NOISELESS,
        ),
        Case(
            "zone_d_withheld_claim",
            "Generated zone D velocity tone without noise, catalog bearing 6205",
            lambda: velocity_tone(None, freq_hz=157.0, rms_mm_s=6.00, fs=12000.0),
            12000.0,
            RPM_6205,
            "mm/s",
            "6205",
            companion=NOISELESS,
        ),
        Case(
            "point_declared_bearing",
            "Measurement point declaring bearing 6205, its speed and ISO group",
            lambda: white(21, scale=0.1, fs=12000.0),
            12000.0,
            RPM_6205,
            "g",
            point={
                "bearing_id": "6205",
                "nominal_rpm": RPM_6205,
                "machine_group": 2,
                "support_type": "rigid",
            },
        ),
    )
}

#: The primary signal and its baselines, through the report only (the
#: diagnosis takes no baseline).
BASELINE_PRIMARY = Case(
    "baseline_primary",
    "Velocity tone of 2.00 mm/s RMS at 157 Hz, no bearing",
    lambda: velocity_tone(22, freq_hz=157.0, rms_mm_s=2.00, fs=10000.0),
    10000.0,
    RPM_ZONES,
    "mm/s",
)
BASELINES: dict[str, Case] = {
    c.case_id: c
    for c in (
        Case(
            "baseline_other_unit",
            "against a baseline that declares g instead of mm/s",
            lambda: velocity_tone(23, freq_hz=157.0, rms_mm_s=2.00, fs=10000.0),
            10000.0,
            RPM_ZONES,
            "g",
        ),
        Case(
            "baseline_other_sampling_rate",
            "against a baseline sampled at 20 kHz instead of 10 kHz",
            lambda: velocity_tone(24, freq_hz=157.0, rms_mm_s=2.00, fs=20000.0),
            20000.0,
            RPM_ZONES,
            "mm/s",
        ),
    )
}

REPORTED_CASES = tuple(CASES)

#: Same-input cases kept out of the parity predicates until their tripwires
#: are reviewed for them. On the low-rate case the two tools recommend the
#: same actions and state different verdicts today, so adding it changes the
#: set of known-defect tripwires.
PARITY_NOT_YET_REVIEWED = frozenset({"low_sampling_rate"})

#: Cases whose two tools receive the same parameters and both return, less
#: those not yet reviewed: the parity predicates apply to these only (a point
#: declaration reaches the diagnosis, not the report).
SAME_INPUT_CASES = tuple(
    cid
    for cid in REPORTED_CASES
    if CASES[cid].point is None and cid not in PARITY_NOT_YET_REVIEWED
)


# ---------------------------------------------------------------------------
# Sandbox and runs
# ---------------------------------------------------------------------------


@dataclass
class Sandbox:
    signals: Path
    models: Path
    reports: Path


@pytest.fixture
def t2_sandbox(tmp_path, monkeypatch, ledger_dir) -> Sandbox:
    """Signals, an empty models folder and reports, all under ``tmp_path``.

    Every by-value import of the three folders the two tools read is patched;
    the ledger is ``conftest.ledger_dir``'s per-test folder.
    """
    box = Sandbox(tmp_path / "signals", tmp_path / "models", tmp_path / "reports")
    for path in (box.signals, box.models, box.reports):
        path.mkdir()
    patch_data_dir(monkeypatch, box.signals)
    for module in (diagnosis_pipeline, diagnostics_tools, report_tools):
        monkeypatch.setattr(module, "MODELS_DIR", box.models)
    monkeypatch.setattr(report_tools, "REPORTS_DIR", box.reports)
    monkeypatch.setattr(report_generator, "REPORTS_DIR", box.reports)
    return box


def require(condition: bool, message: str) -> None:
    """A precondition of a test, reported as a failure even inside a tripwire.

    ``pytest.fail`` raises ``Failed``, not ``AssertionError``, so a strict
    tripwire marked ``raises=AssertionError`` cannot absorb a broken setup as
    its expected failure.
    """
    if not condition:
        pytest.fail(message, pytrace=False)


def store(box: Sandbox, signal_id: str, case: Case) -> str:
    """Write the case's samples and companion into the sandbox and load them."""
    values = np.asarray(case.samples(), dtype=float)
    pd.DataFrame(values).to_csv(
        box.signals / f"{signal_id}.csv", index=False, header=False
    )
    companion: dict[str, Any] = {"sampling_rate": case.fs, **case.companion}
    if case.unit is not None:
        companion["signal_unit"] = case.unit
    if case.point is not None:
        companion["measurement"] = {
            "asset_id": ASSET_ID,
            "measurement_point_id": POINT_ID,
            "acquired_at": ACQUIRED_AT,
            "direction": "horizontal",
        }
    (box.signals / f"{signal_id}_metadata.json").write_text(
        json.dumps(companion), encoding="utf-8"
    )
    loaded = get_repository().load_signal(f"{signal_id}.csv")
    require(loaded["signal_id"] == signal_id, f"loaded as {loaded['signal_id']}")
    return signal_id


def _require_no_model(box: Sandbox) -> None:
    require(
        not any(box.models.iterdir()), "the models folder must stay empty (no model)"
    )
    for module in (diagnosis_pipeline, diagnostics_tools, report_tools):
        require(module.MODELS_DIR == box.models, f"{module.__name__} reads a model dir")


def _recording(real: Callable, sink: list) -> Callable:
    """*real*, unchanged, with a deep copy of each result appended to *sink*."""

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        sink.append(copy.deepcopy(result))
        return result

    return spy


#: The report's embedded metadata, which repeats the verdict; a statement
#: found only there is not rendered.
_METADATA_BLOCK = re.compile(
    r'<script type="application/json" id="report-metadata">.*?</script>', re.DOTALL
)


def _report(box: Sandbox, monkeypatch, signal_id: str, **kwargs) -> dict:
    """Run generate_diagnostic_report (HTML) and keep what it built inside.

    Two spies wrap the module's own functions and record their results
    unchanged: the advisory the report built, and the pipeline diagnoses (the
    first author, run again inside the report path).
    """
    advisories: list[dict] = []
    diagnoses: list[dict] = []
    with monkeypatch.context() as patched:
        for name, sink in (
            ("build_advisory", advisories),
            ("_diagnose_vibration", diagnoses),
        ):
            patched.setattr(
                report_tools, name, _recording(getattr(report_tools, name), sink)
            )
        result = asyncio.run(
            report_tools.generate_diagnostic_report(
                signal_id=signal_id, formats=["html"], **kwargs
            )
        )
    require(len(advisories) == 1, f"report built {len(advisories)} advisories")
    require(
        [f["format"] for f in result["files"]] == ["html"],
        f"report files: {result['files']}",
    )
    rendered = _METADATA_BLOCK.sub(
        "", Path(result["files"][0]["file_path"]).read_text(encoding="utf-8"), count=1
    )
    return {
        "report": {k: v for k, v in result.items() if k != "files"},
        "advisory": advisories[0],
        "report_diagnoses": diagnoses,
        "html_missing": [
            s for s in result["statements"] if html.escape(s) not in rendered
        ],
    }


def _refusal(call: Callable[[], Any]) -> dict:
    """The exception a call raises, as type and message (or Nones)."""
    try:
        call()
    except Exception as exc:  # the type and message ARE the characterization
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": None, "message": None}


_RUNS: dict[str, dict] = {}
_BASELINE_RUNS: dict[str, dict] = {}


def run_case(case_id: str, box: Sandbox, monkeypatch) -> dict:
    """Both tools' outputs for one case, computed once per process."""
    if case_id in _RUNS:
        return _RUNS[case_id]
    case = CASES[case_id]
    _require_no_model(box)
    signal_id = store(box, f"t2_{case_id}", case)
    try:
        if case.point is not None:
            asyncio.run(
                asset_tools.declare_measurement_point(
                    AsyncMock(),
                    asset_id=ASSET_ID,
                    measurement_point_id=POINT_ID,
                    **case.point,
                )
            )
            dv_args: dict[str, Any] = {}
        else:
            dv_args = {"rpm": case.rpm, "bearing_id": case.bearing_id}
        diagnosis = asyncio.run(
            diagnostics_tools.diagnose_vibration(AsyncMock(), signal_id, **dv_args)
        )
        runs: dict[str, Any] = {"dv": diagnosis.model_dump(mode="json")}
        report_args = {
            "rpm": case.rpm,
            "bearing_id": None if case.point is not None else case.bearing_id,
        }
        runs.update(_report(box, monkeypatch, signal_id, **report_args))
    finally:
        get_repository().clear_signal(signal_id)
    _require_no_model(box)
    _RUNS[case_id] = runs
    return runs


def run_baseline_case(case_id: str, box: Sandbox, monkeypatch) -> dict:
    """The report on the primary signal against one baseline."""
    if case_id in _BASELINE_RUNS:
        return _BASELINE_RUNS[case_id]
    _require_no_model(box)
    primary = store(box, f"t2_{BASELINE_PRIMARY.case_id}", BASELINE_PRIMARY)
    baseline = store(box, f"t2_{case_id}", BASELINES[case_id])
    try:
        runs = _report(
            box, monkeypatch, primary, rpm=RPM_ZONES, baseline_signal_id=baseline
        )
    finally:
        get_repository().clear_signal(primary)
        get_repository().clear_signal(baseline)
    _BASELINE_RUNS[case_id] = runs
    return runs


# ---------------------------------------------------------------------------
# The anomaly family, left out of goldens and predicates
# ---------------------------------------------------------------------------


def is_anomaly_recommendation(rec: dict, advisory: dict) -> bool:
    """Whether an advisory recommendation comes from the anomaly block."""
    block = advisory["anomaly"]
    return rec["action"] == block.get("remedy") or block.get("statement") in (
        rec.get("evidence") or []
    )


def anomaly_strings(advisory: dict) -> set[str]:
    """Every string the advisory's anomaly family authored for this case."""
    block = advisory["anomaly"]
    owned = {block.get("statement"), block.get("remedy")}
    for rec in advisory["recommendations"]:
        if is_anomaly_recommendation(rec, advisory):
            owned |= {rec["action"], rec.get("description"), rec.get("motivation")}
    for entry in advisory["disagreements"]:
        if entry.get("governing_indicator") == "anomaly model":
            owned.add(entry["statement"])
    owned.discard(None)
    return owned


def _without_anomaly(recommendations: list[dict], advisory: dict) -> list[dict]:
    return [r for r in recommendations if not is_anomaly_recommendation(r, advisory)]


def report_recommendations(runs: dict) -> list[dict]:
    """The report's recommendations without the anomaly family's."""
    return _without_anomaly(runs["report"]["recommendations"], runs["advisory"])


#: Stands in a diagnosis text for a string the anomaly family authored.
ANOMALY_MARK = "<anomaly family>"


def diagnosis_recommendations(runs: dict) -> list[str]:
    """The diagnosis's recommendations without any the anomaly family authored."""
    owned = anomaly_strings(runs["advisory"])
    return [r for r in runs["dv"]["recommendations"] if r not in owned]


def diagnosis_text(runs: dict) -> str:
    """The diagnosis text with each anomaly-family string replaced by a mark."""
    text = runs["dv"]["overall_diagnosis"]
    for owned in sorted(anomaly_strings(runs["advisory"]), key=len, reverse=True):
        text = text.replace(owned, ANOMALY_MARK)
    return text


# ---------------------------------------------------------------------------
# Projections (what the goldens hold)
# ---------------------------------------------------------------------------


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _claim(bearing_faults: Optional[dict]) -> Optional[dict]:
    if not bearing_faults:
        return None
    claim = bearing_faults["claim"]
    return {"level": claim["level"], "conditions": claim.get("conditions")}


def _baseline(advisory: dict) -> dict:
    block = advisory["baseline_comparison"]
    return {k: block.get(k) for k in ("status", "statement", "remedy")}


def report_projection(runs: dict) -> dict:
    """The report surface of one run, anomaly family and file paths left out."""
    report, advisory = runs["report"], runs["advisory"]
    owned = anomaly_strings(advisory)
    return {
        "generate_diagnostic_report": {
            "verdict": report["verdict"],
            "fault_canonical": report["fault_canonical"],
            "evidence_strength": report["evidence_strength"],
            "evidence_strength_note": report["evidence_strength_note"],
            "bearing_check": report["bearing_check"],
            "statements": [s for s in report["statements"] if s not in owned],
            "recommendations": report_recommendations(runs),
            "provenance": {
                k: v for k, v in report["provenance"].items() if k != "server_version"
            },
        },
        "report_path": {
            "pipeline_evidence_strength": [
                d["evidence_strength"] for d in runs["report_diagnoses"]
            ],
            "block_status": {
                key: advisory[key]["status"]
                for key in ("iso", "bearing_match", "spectral_energy")
            },
            "baseline_comparison": _baseline(advisory),
            "iso_standard_note": advisory["iso"].get("standard_note"),
            "disagreements": [
                d
                for d in advisory["disagreements"]
                if d.get("governing_indicator") != "anomaly model"
            ],
            "statements_missing_from_html": runs["html_missing"],
        },
    }


def characterization(runs: dict) -> dict:
    """Both surfaces of one run, as the goldens hold them."""
    dv = runs["dv"]
    iso = dv["iso_severity"]
    return _json(
        {
            "diagnose_vibration": {
                "overall_diagnosis": diagnosis_text(runs),
                "recommendations": diagnosis_recommendations(runs),
                "evidence_strength": dv["evidence_strength"],
                "bearing_check": dv["bearing_check"],
                "claim": _claim(dv["bearing_faults"]),
                "iso_status": iso.get("status"),
                "iso_zone": iso.get("zone"),
                "parameter_sources": dv["parameter_sources"],
            },
            **report_projection(runs),
        }
    )


# ---------------------------------------------------------------------------
# Python-route cases: refusals, the composer on inputs no tool hands it
# ---------------------------------------------------------------------------


def refusals(box: Sandbox) -> dict:
    """Corrupt inputs, each refused before any block."""
    zone_a = CASES["zone_a"]
    signal_id = store(box, "t2_rpm_zero", zone_a)
    samples = np.asarray(zone_a.samples(), dtype=float)
    with_nan = samples.copy()
    with_nan[100] = np.nan
    try:
        observed = {
            "rpm_zero": {
                "diagnose_vibration": _refusal(
                    lambda: asyncio.run(
                        diagnostics_tools.diagnose_vibration(
                            AsyncMock(), signal_id, rpm=0.0
                        )
                    )
                ),
                "generate_diagnostic_report": _refusal(
                    lambda: asyncio.run(
                        report_tools.generate_diagnostic_report(
                            signal_id=signal_id, rpm=0.0, formats=["html"]
                        )
                    )
                ),
            },
            "nan_sample_pipeline": _refusal(
                lambda: diagnosis_pipeline.diagnose_vibration(
                    with_nan, zone_a.fs, RPM_ZONES, signal_unit="mm/s"
                )
            ),
            "fs_zero_pipeline": _refusal(
                lambda: diagnosis_pipeline.diagnose_vibration(
                    samples, 0.0, RPM_ZONES, signal_unit="mm/s"
                )
            ),
        }
    finally:
        get_repository().clear_signal(signal_id)
    require(not any(box.reports.iterdir()), "a refused report wrote a file")
    return _json(observed)


def _advisory_projection(advisory: dict) -> dict:
    owned = anomaly_strings(advisory)
    return {
        "verdict": advisory["verdict"],
        "bearing_match": {
            k: advisory["bearing_match"].get(k)
            for k in ("status", "statement", "claim_level")
        },
        "recommendations": _without_anomaly(advisory["recommendations"], advisory),
        "baseline_comparison": _baseline(advisory),
        "evidence_statements": [
            s for s in advisory["evidence"]["statements"] if s not in owned
        ],
    }


def controlled_union_route(box: Sandbox) -> dict:
    """A controlled_union_v1 bearing result fed to the composer.

    No MCP route hands that result to either author, so the composer is fed
    directly: the pipeline's own diagnosis of the outer-race defect signal,
    with its bearing block replaced by the experimental rule's result. The
    diagnosis's ``evidence_strength`` was scored on the released rule's claim
    and the advisory copies it, so it is left out here rather than recorded as
    if it scored the experimental claim.
    """
    _require_no_model(box)
    samples = bearing_defect(15, "outer")
    diagnosis = diagnosis_pipeline.diagnose_vibration(
        samples,
        12000.0,
        RPM_6205,
        signal_id="t2_controlled_union",
        bearing_id="6205",
        signal_unit="g",
    )
    experimental = check_all_bearing_faults(
        samples,
        12000.0,
        "6205",
        RPM_6205,
        signal_id="t2_controlled_union",
        corroboration_rule="controlled_union_v1",
    )
    advisory = build_advisory({**diagnosis, "bearing_faults": experimental})
    projection = _advisory_projection(advisory)
    projection["claim"] = {
        k: experimental["claim"].get(k)
        for k in ("level", "conditions", "policy_version")
    }
    return _json(projection)


def different_bearing_baseline(box: Sandbox) -> dict:
    """A baseline analysed against another catalog bearing, fed to the composer.

    The report passes one bearing_id to both of its diagnoses, so only the
    composer can receive a baseline analysed against another bearing.
    """
    _require_no_model(box)
    samples = BASELINE_PRIMARY.samples()

    def diagnose(bearing_id: str, signal_id: str) -> dict:
        return diagnosis_pipeline.diagnose_vibration(
            samples,
            BASELINE_PRIMARY.fs,
            RPM_ZONES,
            signal_id=signal_id,
            bearing_id=bearing_id,
            signal_unit="mm/s",
        )

    advisory = build_advisory(
        diagnose("6205", "t2_baseline_primary"), diagnose("6203", "t2_baseline_6203")
    )
    return _json(_baseline(advisory))


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------

ABOUT = (
    "Characterization of today's outputs, not a requirement. A difference means "
    "the output changed: regenerate with T2_CHARACTERIZATION_WRITE=1 and review "
    "the change. The anomaly family's strings are left out."
)


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text(encoding="utf-8"))["observed"]


def check_golden(name: str, description: str, observed: dict) -> None:
    """Compare *observed* with the golden *name*, or write it when asked."""
    if os.environ.get(WRITE_VARIABLE) == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "about": ABOUT,
            "case": name,
            "description": description,
            "observed": observed,
        }
        golden_path(name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pytest.skip(f"golden {name} written")
    assert observed == load_golden(name), (
        f"the characterized output of {name} changed (not necessarily wrong): "
        f"review the difference, then regenerate with {WRITE_VARIABLE}=1"
    )


# ---------------------------------------------------------------------------
# Parity predicates, shared by the tripwires and their controls
# ---------------------------------------------------------------------------


def diagnosis_actions(runs: dict) -> set[str]:
    return set(diagnosis_recommendations(runs))


def report_actions(runs: dict) -> set[str]:
    return {r["action"] for r in report_recommendations(runs)}


def actions_agree(runs: dict) -> bool:
    """Both tools recommend the same actions for the same input.

    A set comparison: the order and the format of each surface are still open
    and are not compared.
    """
    return diagnosis_actions(runs) == report_actions(runs)


def verdict_stated_by_diagnosis(runs: dict) -> bool:
    """The verdict the report states appears, verbatim, in the diagnosis text.

    Containment only: where the diagnosis places it, and what else its text
    says, are still open and are not compared.
    """
    return runs["report"]["verdict"] in runs["dv"]["overall_diagnosis"]


def evidence_strength_agrees(runs: dict) -> bool:
    return runs["dv"]["evidence_strength"] == runs["report"]["evidence_strength"]


_SHAFT_ORDER = re.compile(
    re.escape(SHAFT_ORDER_FIRST_WORDS) + r".*?" + re.escape(SHAFT_ORDER_LAST_WORDS)
)


def shaft_order_sentence(text: str) -> Optional[str]:
    """The shaft-order statement a diagnosis text carries, or None."""
    found = _SHAFT_ORDER.search(text)
    return found.group(0) if found else None


def shaft_order_carried(runs: dict) -> bool:
    """The report states the shaft-order statement the diagnosis states."""
    sentence = shaft_order_sentence(runs["dv"]["overall_diagnosis"])
    require(sentence is not None, "the diagnosis states no shaft-order line")
    return any(sentence in s for s in runs["report"]["statements"])
