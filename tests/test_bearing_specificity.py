"""Acceptance scenarios for the specificity of the bearing check.

The bearing check (``check_all_bearing_faults``, ``check_frequency_set``) is
the engine behind every bearing verdict the server gives: the check tools,
``diagnose_vibration``, the diagnostic report's advisory, ledger snapshots and
the CWRU benchmark. This module states what it must get right on signals whose
content is known by construction (``tests/_bearing_scenarios.py``) and on the
bundled MFPT recordings:

- fault-free signals name no fault type and grade none as supported;
- a single defect is named, or at least supported, as its own type and no
  other;
- an observed line is credited to at most one fault type, and a line that two
  types' families share is credited to neither;
- the order of the frequency dict never changes the outcome;
- a non-specific match drives no fault evidence, disagreement or bearing
  maintenance action in the diagnosis and the advisory, and envelope evidence
  alone never asks for a bearing replacement.

Today's fields express these properties as follows: a check *names* a fault
when ``most_likely_fault`` is set, *supports* a type when that type's
``evidence_strength`` is "moderate" or "high", and *detects* a type when its
``detected`` is true. A bearing maintenance action is a recommendation to
replace, repair, lubricate or dismantle a bearing, or to inspect one of its
elements; a verification step, such as a repeat or reference measurement or a
check of the bearing's designation or geometry, is not one.

Rules this module follows:

- Every expectation comes from how the signal was built, never from what the
  engine returns, and every random source is seeded.
- Rates over seeds are bounded by a nominal 1 % false-alarm probability per
  check or per corroborating line, or by the 95 % sensitivity bar, never by a
  count copied from a run. A check is one bearing check of one record (one
  ``check_all_bearing_faults`` call), and the tests of its fault frequencies
  share that probability; each corroborating line is tested at it alone.
- Nothing under ``src/`` is changed here. A property the shipped engine
  violates is a ``pytest.mark.xfail(strict=True, raises=AssertionError)``
  tripwire whose reason starts with ``audit-known-defect:``, so the weekly
  audit lists it; when the defect is fixed the tripwire XPASSes, which fails
  the strict run and asks for the marker to be removed. The tests without a
  marker are sensitivity controls that pass today and must keep passing.
"""

import math
import re

import numpy as np
import pytest
from scipy.stats import binom

from _bearing_scenarios import (
    DEFECT_SEED,
    DEFECTS_PRESENT,
    FAULT_FREE,
    FAULT_FREE_SEED,
    FREQUENCIES_6205,
    RPM_6205,
    DefectKind,
    defect,
    load_mfpt,
    shared_line,
    white,
)
from _golden_signals import FS as GOLDEN_FS
from _golden_signals import golden_signals
from predictive_maintenance_mcp.decision_support.advisory import (
    build_advisory,
    collect_statements,
)
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    _bearing_check,
    _synthesize_diagnosis,
    diagnose_vibration,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
    check_frequency_set,
)
from tools.audit.report import KNOWN_DEFECT_PREFIX

#: Sampling rate of the synthetic 6205 scenarios (the CWRU drive-end rate).
FS = 12000.0
SHAFT_HZ = RPM_6205 / 60.0

#: Golden fixture signals are built for a 6205 at 1800 rpm.
GOLDEN_RPM = 1800.0

#: Evidence grades of a type the check supports.
SUPPORTED = ("moderate", "high")

#: Nominal false-alarm probability per check, shared by its frequency tests.
CHECK_FALSE_ALARM = 0.01

#: Probability of a false failure tolerated when a rate is bounded over seeds.
RATE_TEST_LEVEL = 0.001

#: Share of seeds on which a defect must be supported (sensitivity bar).
SENSITIVITY_BAR = 0.95

#: No anomaly model: these scenarios isolate the bearing evidence.
NO_MODEL = "nonexistent_model_bearing_specificity"

#: A recommendation to replace a bearing, in any of the usual words.
BEARING_REPLACEMENT = re.compile(
    r"\breplac\w*"
    r"|\b(?:renew|change|swap|exchange|overhaul)\w*\s+(?:the\s+|a\s+)?bearing",
    re.IGNORECASE,
)

_BEARING_ELEMENT = (
    r"(?:bearings?|races?|outer|inner|rolling[- ]elements?|balls?|cages?)"
)

#: A bearing maintenance action (see the module docstring).
BEARING_MAINTENANCE = re.compile(
    BEARING_REPLACEMENT.pattern + r"|\b(?:repair|dismantl|disassembl)\w*"
    r"|\b(?:re-?lubricate|lubricate|regrease|grease)\b"
    r"|\binspect\w*(?:\W+\w+){0,4}?\W+" + _BEARING_ELEMENT + r"\b"
    r"|\b" + _BEARING_ELEMENT + r"\W+inspection",
    re.IGNORECASE,
)

#: What a single-record envelope indication must never say.
REPLACEMENT_OR_CONFIRMATION = re.compile(
    r"replace bearing|plan replacement|\bconfirmed\b", re.IGNORECASE
)

#: A diagnosis sentence that lists detected fault types or names the likeliest.
NAMED_FAULT_CLAIM = re.compile(r"faults? detected:|\bmost likely\b", re.IGNORECASE)

#: The advisory's claim that harmonics rule out a noise peak.
HARMONICS_RULE_OUT_NOISE = "which a single noise peak would not produce"


def tripwire(short_id: str, text: str) -> pytest.MarkDecorator:
    """Mark a property the shipped engine violates (see the module docstring).

    The reason is ``audit-known-defect: <id>: <text>``, the form the audit
    report reads.
    """
    return pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=f"{KNOWN_DEFECT_PREFIX} {short_id}: {text}",
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _catalog_check(signal, fs=FS, rpm=RPM_6205) -> dict:
    return check_all_bearing_faults(signal=signal, fs=fs, bearing_id="6205", rpm=rpm)


def _check(result: dict, label: str) -> dict:
    return next(c for c in result["fault_checks"] if c["fault_type"] == label)


def _named(checks: list[dict]) -> list[str]:
    return [r["most_likely_fault"] for r in checks if r["most_likely_fault"]]


def _supported(result: dict) -> list[str]:
    return [
        c["fault_type"]
        for c in result["fault_checks"]
        if c["evidence_strength"] in SUPPORTED
    ]


def _detected(result: dict) -> list[str]:
    return [c["fault_type"] for c in result["fault_checks"] if c["detected"]]


def _actions_matching(pattern: re.Pattern, actions) -> list[str]:
    return [a for a in actions if pattern.search(a)]


def _bar(seeds: int) -> int:
    """Seeds out of ``seeds`` that meet the sensitivity bar."""
    return math.ceil(SENSITIVITY_BAR * seeds)


# ===========================================================================
# Fault-free signals
# ===========================================================================

#: Records per fault-free sweep.
FAULT_FREE_SEEDS = 30

#: White-noise sweeps: duration (s) -> seed offset.
WHITE_NOISE_SEED_OFFSETS = {1.0: 0, 10.0: 100}

#: The other families' 1 s sweeps: family -> seed offset.
FAMILY_SEED_OFFSETS = {
    "shaft_lines": 300,
    "shaft_modulated": 400,
    "resonant_coloured": 500,
    "random_impacts": 600,
    "gear_tones": 700,
}


@pytest.fixture(scope="module")
def white_noise_checks() -> dict[float, list[dict]]:
    """6205 at 1797 rpm on seeded white noise at 12 kHz, per duration."""
    return {
        seconds: [
            _catalog_check(
                white(_rng(FAULT_FREE_SEED + offset + i), FS, seconds, SHAFT_HZ)
            )
            for i in range(FAULT_FREE_SEEDS)
        ]
        for seconds, offset in WHITE_NOISE_SEED_OFFSETS.items()
    }


class TestFaultFreeSignals:
    """No bearing defect by construction: no fault type is a finding."""

    @tripwire(
        "bearing-detection-without-significance",
        "T1 M1: the bearing check names a fault type on white noise",
    )
    def test_white_noise_names_no_fault(self, white_noise_checks):
        named = {s: _named(checks) for s, checks in white_noise_checks.items()}
        detail = "; ".join(
            f"{s:g} s: {len(n)} of {FAULT_FREE_SEEDS} records {sorted(set(n))}"
            for s, n in named.items()
        )

        assert not any(named.values()), f"white noise named a fault: {detail}"

    @tripwire(
        "bearing-detection-without-significance",
        "T1 M1: the bearing check grades a fault type moderate or high on "
        "white noise",
    )
    def test_white_noise_supports_no_fault_type(self, white_noise_checks):
        supported = {
            s: sum(bool(_supported(r)) for r in checks)
            for s, checks in white_noise_checks.items()
        }
        detail = "; ".join(
            f"{s:g} s: {n} of {FAULT_FREE_SEEDS} records" for s, n in supported.items()
        )
        message = f"white noise graded a fault type moderate or high: {detail}"

        assert not any(supported.values()), message

    @tripwire(
        "bearing-detection-without-significance",
        "T1 M1: fault frequencies are detected on white noise far above a "
        "nominal 1 % false-alarm probability per check",
    )
    def test_white_noise_detections_stay_within_the_check_false_alarm_rate(
        self, white_noise_checks
    ):
        """Each record is one check, and the check's fault-frequency tests
        share its nominal false-alarm probability, so a record detects any
        fault frequency with at most that probability. Records are
        independent, so the number of records with a detection is bounded by
        the binomial count that probability exceeds with probability
        ``RATE_TEST_LEVEL``. The number of detected frequencies is reported
        only: detections within one check are not independent, so no tail
        bound on their total follows from the policy."""
        records = [r for checks in white_noise_checks.values() for r in checks]
        allowed = int(binom.ppf(1 - RATE_TEST_LEVEL, len(records), CHECK_FALSE_ALARM))
        checks_with_detection = sum(bool(_detected(r)) for r in records)
        detected_frequencies = sum(len(_detected(r)) for r in records)

        assert checks_with_detection <= allowed, (
            f"{checks_with_detection} of {len(records)} white-noise checks detect "
            f"a fault frequency ({detected_frequencies} frequencies in all); a "
            f"nominal {CHECK_FALSE_ALARM:.0%} false-alarm probability per check "
            f"allows at most {allowed} checks"
        )

    @pytest.mark.parametrize("family", list(FAMILY_SEED_OFFSETS))
    @tripwire(
        "bearing-detection-without-significance",
        "T1 M2-M3: the bearing check names a fault type on fault-free "
        "coloured, impulsive, shaft-related and gear-tone signals",
    )
    def test_fault_free_family_names_no_fault(self, family):
        offset = FAMILY_SEED_OFFSETS[family]
        checks = [
            _catalog_check(
                FAULT_FREE[family](
                    _rng(FAULT_FREE_SEED + offset + i), FS, 1.0, SHAFT_HZ
                )
            )
            for i in range(FAULT_FREE_SEEDS)
        ]
        named = _named(checks)

        assert not named, (
            f"{family}: a fault was named on {len(named)} of "
            f"{FAULT_FREE_SEEDS} records {sorted(set(named))}"
        )

    @tripwire(
        "bearing-detection-without-significance",
        "T1 M4: fault frequencies are detected on a pure 50 Hz sine, which "
        "has no signal content in the envelope band",
    )
    def test_empty_envelope_band_detects_nothing(self):
        """``golden_iso`` is a pure 50 Hz sine: none of the signal lies in the
        500 Hz to 5 kHz envelope band. What the band-pass filter leaves there
        is its own residue (stopband leakage and its response at the record
        edges), not vibration content, so a fault frequency detected from it
        is a false detection."""
        detected = _detected(
            _catalog_check(golden_signals()["golden_iso"], fs=GOLDEN_FS, rpm=GOLDEN_RPM)
        )

        assert not detected, f"pure 50 Hz sine detected {detected}"


# ===========================================================================
# One defect at a time
# ===========================================================================

DEFECT_SEEDS = 20
DEFECT_SECONDS = 10.0
DEFECT_SNR_DB = -15.0


def _defect_checks(kind: DefectKind, seed_offset: int) -> list[dict]:
    return [
        _catalog_check(
            defect(
                kind,
                _rng(DEFECT_SEED + seed_offset + i),
                FS,
                DEFECT_SECONDS,
                FREQUENCIES_6205,
                SHAFT_HZ,
                DEFECT_SNR_DB,
            )
        )
        for i in range(DEFECT_SEEDS)
    ]


@pytest.fixture(scope="module")
def outer_race_checks() -> list[dict]:
    return _defect_checks("outer", 0)


@pytest.fixture(scope="module")
def rolling_element_checks() -> list[dict]:
    return _defect_checks("rolling_element", 100)


def _supported_and_not_misnamed(checks: list[dict], label: str) -> int:
    """Records where ``label`` is supported and no other type is named."""
    return sum(
        _check(r, label)["evidence_strength"] in SUPPORTED
        and r["most_likely_fault"] in (None, label)
        for r in checks
    )


SHARED_LINE_SEEDS = 20

#: Nominal false-alarm probability of one corroborating line, tested alone.
CORROBORATION_FALSE_ALARM = 0.01

#: BPFI's candidate corroborations in the shared-line scenario besides the
#: shared line: its third and fourth harmonics and its two shaft-rate
#: sidebands, none of which holds a line by construction.
BPFI_OTHER_CANDIDATES = 4


def _shared_line_problems(result: dict, line_hz: float) -> list[str]:
    """What the shared-line scenario forbids in any one check."""
    credits: dict[float, list[str]] = {}
    for c in result["fault_checks"]:
        for h in c["harmonics_detected"]:
            credits.setdefault(h["detected_hz"], []).append(
                f"{c['fault_type']} {h['harmonic']}x"
            )
    problems = []
    for hz, who in credits.items():
        if abs(hz - line_hz) <= 0.01 * line_hz:
            problems.append(
                f"the shared line is credited as {', '.join(who)} ({hz} Hz)"
            )
        elif len({w.split()[0] for w in who}) > 1:
            problems.append(f"the peak at {hz} Hz is credited as {', '.join(who)}")
    bpfo = _check(result, "BPFO")
    second_hz = 2 * line_hz / 3
    own_second = any(
        h["harmonic"] == 2 and abs(h["detected_hz"] - second_hz) <= 0.01 * second_hz
        for h in bpfo["harmonics_detected"]
    )
    if bpfo["evidence_strength"] not in SUPPORTED or not own_second:
        problems.append(
            f"BPFO is graded {bpfo['evidence_strength']!r} without its own second "
            f"harmonic at {second_hz:.1f} Hz"
        )
    return problems


class TestSingleDefect:
    """Each defect is credited to its own type, and to no other."""

    def test_outer_race_defect_is_named(self, outer_race_checks):
        """The outer-race defect is named as such, not merely graded: with no
        other type supported, BPFO is the one fault the check names."""
        (label,) = DEFECTS_PRESENT["outer"]
        named = sum(r["most_likely_fault"] == label for r in outer_race_checks)

        assert named >= _bar(DEFECT_SEEDS), (
            f"outer-race defect at {DEFECT_SNR_DB:g} dB: {label} named on "
            f"{named} of {DEFECT_SEEDS} seeds; named "
            f"{[r['most_likely_fault'] for r in outer_race_checks]}"
        )

    @tripwire(
        "bearing-wrong-type-co-detected",
        "T1 M5: an outer-race defect also detects other fault types",
    )
    def test_outer_race_defect_detects_no_other_type(self, outer_race_checks):
        others = [
            sorted(set(_detected(r)) - DEFECTS_PRESENT["outer"])
            for r in outer_race_checks
        ]
        clean = sum(not o for o in others)

        assert clean >= _bar(DEFECT_SEEDS), (
            f"outer-race defect at {DEFECT_SNR_DB:g} dB: another type detected "
            f"on {DEFECT_SEEDS - clean} of {DEFECT_SEEDS} seeds "
            f"{sorted({t for o in others for t in o})}"
        )

    @tripwire(
        "bearing-rolling-element-line",
        "T1 M5: the rolling-element check tests 1xBSF, so a defect striking at "
        "2xBSF is not supported and another type is named",
    )
    def test_rolling_element_defect_is_supported_and_no_other_type_named(
        self, rolling_element_checks
    ):
        """BSF is supported, and no other type is ever the one fault named.
        A strong rolling-element defect can also leave cage-rate sidebands in
        other windows, so several supported types, with none singled out, is
        an accepted outcome; which of them ranks first is not observable in
        these fields."""
        (label,) = DEFECTS_PRESENT["rolling_element"]
        ok = _supported_and_not_misnamed(rolling_element_checks, label)

        assert ok >= _bar(DEFECT_SEEDS), (
            f"rolling-element defect at {DEFECT_SNR_DB:g} dB: {label} supported "
            f"without another type named on {ok} of {DEFECT_SEEDS} seeds; "
            f"named {[r['most_likely_fault'] for r in rolling_element_checks]}"
        )

    @tripwire(
        "bearing-shared-line-double-counted",
        "T1 M6: a line two fault types share is credited to both",
    )
    def test_shared_line_does_not_corroborate_a_second_type(self):
        """The shared-line scenario (``tests/_bearing_scenarios.py``) on the
        6205: the line at 3 x BPFO is also exactly twice the 1.5 x BPFO line
        in the BPFI window, so it lies inside any harmonic window anchored at
        either line, however narrow. On every seed no observed peak is
        credited to two types, the shared line is credited to neither, and
        BPFO keeps the support of its own second harmonic. BPFI's line is
        real but its only real corroboration is the shared line, so BPFI is
        supported only where noise corroborates it. At a nominal 1 % per
        corroborating line, a seed does so with at most the sum over its other
        candidates (a union bound, whatever their correlation), and seeds are
        independent, so the count is bounded binomially like the white-noise
        checks."""
        line_hz = 3 * FREQUENCIES_6205["BPFO"]
        failing, bpfi_supported = {}, 0
        for i in range(SHARED_LINE_SEEDS):
            result = _catalog_check(
                shared_line(
                    _rng(DEFECT_SEED + 300 + i), FS, DEFECT_SECONDS, FREQUENCIES_6205
                )
            )
            problems = _shared_line_problems(result, line_hz)
            if problems:
                failing[i] = problems
            bpfi_supported += "BPFI" in _supported(result)
        noise_support = BPFI_OTHER_CANDIDATES * CORROBORATION_FALSE_ALARM
        allowed = int(binom.ppf(1 - RATE_TEST_LEVEL, SHARED_LINE_SEEDS, noise_support))
        message = (
            f"shared line at 3 x BPFO ({line_hz:.1f} Hz): BPFI supported on "
            f"{bpfi_supported} of {SHARED_LINE_SEEDS} seeds (at most {allowed} "
            f"allowed); failing on {len(failing)} seeds, first three: "
            f"{dict(list(failing.items())[:3])}"
        )

        assert not failing, message
        assert bpfi_supported <= allowed, message


# ===========================================================================
# Outcome independent of the frequency dict's order
# ===========================================================================

ORDER_RECORDS = 5


class TestOrderIndependence:
    @tripwire(
        "bearing-ranking-by-list-order",
        "T1 M7: the most likely fault among equally graded types follows the "
        "order of the frequency dict",
    )
    def test_frequency_order_does_not_change_the_outcome(self):
        """Combined outer- and inner-race defects at -10 dB, both orders."""
        forward = {k: FREQUENCIES_6205[k] for k in ("BPFO", "BPFI")}
        backward = dict(reversed(list(forward.items())))
        outcome = ("most_likely_fault", "overall_assessment")
        differing = []
        for i in range(ORDER_RECORDS):
            signal = defect(
                "combined",
                _rng(DEFECT_SEED + 200 + i),
                FS,
                DEFECT_SECONDS,
                FREQUENCIES_6205,
                SHAFT_HZ,
                -10.0,
            )
            a = check_frequency_set(signal, FS, forward, RPM_6205)
            b = check_frequency_set(signal, FS, backward, RPM_6205)
            if [a[k] for k in outcome] != [b[k] for k in outcome]:
                differing.append((a["most_likely_fault"], b["most_likely_fault"]))

        assert not differing, (
            f"reversing the frequency dict changed the outcome on "
            f"{len(differing)} of {ORDER_RECORDS} records: most likely {differing}"
        )


# ===========================================================================
# Statements and actions
# ===========================================================================

PIPELINE_NOISE_RECORDS = 10


def _diagnosis_with_declared_frequencies(
    signal: np.ndarray, fs: float, frequencies: dict[str, float], rpm: float
) -> dict:
    """A ``diagnose_vibration`` result whose bearing block checks *frequencies*.

    ``diagnose_vibration`` checks catalog bearings only, so the engine's check
    of the declared frequencies is given to the synthesis the pipeline runs.
    """
    check = check_frequency_set(signal, fs, frequencies, rpm)
    blocks = diagnose_vibration(
        signal, fs, rpm, signal_unit="g", anomaly_model_name=NO_MODEL
    )
    bearing_check = _bearing_check("assessed")
    text, strength, recommendations = _synthesize_diagnosis(
        fft_summary=blocks["fft_summary"],
        psd_summary=blocks["psd_summary"],
        stft_summary=blocks["stft_summary"],
        bearing_faults=check,
        bearing_check=bearing_check,
        iso_severity=blocks["iso_severity"],
        anomaly=blocks["anomaly_detection"],
        rpm=rpm,
    )
    return {
        **blocks,
        "bearing_faults": check,
        "bearing_check": bearing_check,
        "overall_diagnosis": text,
        "evidence_strength": strength,
        "recommendations": recommendations,
    }


@pytest.fixture(scope="module")
def mfpt_baseline() -> dict:
    """The MFPT healthy baseline_3 with its declared frequencies at 1500 rpm.

    Its envelope carries a line at twice the 60 Hz mains frequency inside the
    BPFI window, with no harmonic or sideband a bearing defect would add.
    """
    signal, fs, frequencies, rpm = load_mfpt("real_test/baseline_3.csv")
    return _diagnosis_with_declared_frequencies(signal, fs, frequencies, rpm)


class TestStatementsAndActions:
    """A non-specific match is no finding; envelope evidence is no replacement."""

    @tripwire(
        "bearing-noise-match-drives-actions",
        "T1 M12: the diagnosis turns a bearing match on white noise into fault "
        "evidence and a bearing maintenance action",
    )
    def test_diagnosis_of_white_noise_reports_no_bearing_evidence(self):
        wrong = []
        for i in range(PIPELINE_NOISE_RECORDS):
            signal = 0.1 * white(_rng(FAULT_FREE_SEED + 900 + i), FS, 1.0, SHAFT_HZ)
            result = diagnose_vibration(
                signal,
                FS,
                RPM_6205,
                signal_id="white_noise",
                bearing_id="6205",
                signal_unit="g",
                anomaly_model_name=NO_MODEL,
            )
            actions = _actions_matching(BEARING_MAINTENANCE, result["recommendations"])
            if result["evidence_strength"] != "none" or actions:
                wrong.append((result["evidence_strength"], actions))

        assert not wrong, (
            f"white noise with bearing 6205 gave fault evidence or a bearing "
            f"maintenance action on {len(wrong)} of {PIPELINE_NOISE_RECORDS} "
            f"records: {wrong[:3]}"
        )

    @tripwire(
        "bearing-noise-match-drives-actions",
        "T1 M9: an uncorroborated line in a healthy recording becomes a named "
        "fault in the diagnosis",
    )
    def test_uncorroborated_line_is_no_finding_of_the_diagnosis(self, mfpt_baseline):
        text = mfpt_baseline["overall_diagnosis"]
        named = mfpt_baseline["bearing_faults"]["most_likely_fault"]

        assert not NAMED_FAULT_CLAIM.search(
            text
        ), f"diagnosis of a healthy recording: {text}"
        assert named is None, f"the bearing check names {named} on a healthy recording"
        assert mfpt_baseline["evidence_strength"] == "none"
        assert not _actions_matching(
            BEARING_MAINTENANCE, mfpt_baseline["recommendations"]
        )

    @tripwire(
        "bearing-noise-match-drives-actions",
        "T1 M9: the advisory names a fault, reports disagreeing indicators, "
        "recommends a bearing replacement and says harmonics rule out noise on "
        "a healthy recording",
    )
    def test_uncorroborated_line_is_no_finding_of_the_advisory(self, mfpt_baseline):
        advisory = build_advisory(mfpt_baseline)
        found = {
            "verdict names": advisory["verdict"]["fault_canonical"],
            "disagreements": [e["statement"] for e in advisory["disagreements"]],
            "maintenance actions": _actions_matching(
                BEARING_MAINTENANCE, (r["action"] for r in advisory["recommendations"])
            ),
            "harmonics rule out noise": [
                s
                for s in advisory["evidence"]["statements"]
                if HARMONICS_RULE_OUT_NOISE in s
            ],
        }

        assert not any(found.values()), {k: v for k, v in found.items() if v}

    @tripwire(
        "bearing-replacement-from-envelope-evidence",
        "T1 AE11: a single-record envelope indication produces 'Replace "
        "bearing' and 'plan replacement'",
    )
    def test_envelope_indication_asks_for_no_replacement(self):
        """``golden_bearing``: a true BPFO modulation."""
        diagnosis = diagnose_vibration(
            golden_signals()["golden_bearing"],
            GOLDEN_FS,
            GOLDEN_RPM,
            signal_id="golden_bearing",
            bearing_id="6205",
            signal_unit="g",
            anomaly_model_name=NO_MODEL,
        )
        advisory = build_advisory(diagnosis)
        sentences = [
            diagnosis["overall_diagnosis"],
            *diagnosis["recommendations"],
            *collect_statements(advisory),
        ]
        actions = [
            *diagnosis["recommendations"],
            *(r["action"] for r in advisory["recommendations"]),
        ]
        offending = [
            *(s for s in sentences if REPLACEMENT_OR_CONFIRMATION.search(s)),
            *_actions_matching(BEARING_REPLACEMENT, actions),
        ]

        assert not offending, offending


class TestActionVocabulary:
    """What the action checks above count as bearing maintenance."""

    @pytest.mark.parametrize(
        "action",
        [
            "Bearing 6205: plan replacement at next opportunity.",
            "Replace bearing, check alignment",
            "Change the bearing.",
            "Grease the bearing.",
            "Re-lubricate the bearing.",
            "Schedule a bearing inspection.",
            "Plan an inspection of the outer race.",
            "Inspect the outer and inner races.",
        ],
    )
    def test_maintenance_action(self, action):
        assert BEARING_MAINTENANCE.search(action)

    @pytest.mark.parametrize(
        "action",
        [
            "Bearing 6205: monitor closely, confirm with additional measurements.",
            "Repeat the measurement, or compare it with a reference measurement "
            "of the same machine.",
            "Check the bearing designation and geometry against the "
            "manufacturer's data.",
            "Take a reference measurement on the same bearing.",
            "A longer capture would test the cage line.",
            "Schedule inspection",
        ],
    )
    def test_not_a_maintenance_action(self, action):
        assert not BEARING_MAINTENANCE.search(action)

    def test_a_confirmation_step_is_not_a_replacement(self):
        assert not BEARING_REPLACEMENT.search(
            "Inspect the outer race and confirm with a comparable measurement."
        )


# ===========================================================================
# Sensitivity controls
# ===========================================================================


class TestSensitivityControls:
    def test_golden_bearing_outer_race_is_detected_and_most_likely(self):
        result = _catalog_check(
            golden_signals()["golden_bearing"], fs=GOLDEN_FS, rpm=GOLDEN_RPM
        )

        assert _check(result, "BPFO")["detected"]
        assert result["most_likely_fault"] == "BPFO"
