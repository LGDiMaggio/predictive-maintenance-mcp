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

The older fields express these properties as follows: a check *names* a fault
when ``most_likely_fault`` is set, *supports* a type when that type's
``evidence_strength`` is "moderate" or "high", and *detects* a type when its
``detected`` is true. The evidence fields express them directly: the check's
``outcome`` names one type only when it is "indicated", ``candidates`` lists
the supported types, and each check's ``state`` says whether its line was
tested, found, corroborated or shared. A bearing maintenance action is a
recommendation to replace, repair, lubricate or dismantle a bearing, or to
inspect one of its elements; a verification step, such as a repeat or
reference measurement or a check of the bearing's designation or geometry, is
not one.

Rules this module follows:

- Every expectation comes from how the signal was built, never from what the
  engine returns, and every random source is seeded.
- Rates over seeds are bounded by a nominal 1 % false-alarm probability per
  check or per corroborating line, by the quality target on checks that name
  a type (screened, not proven), or by the 95 % sensitivity bar, never by a
  count copied from a run. A check is one bearing check of one record (one
  ``check_all_bearing_faults`` call), and the tests of its fault frequencies
  share that probability; each corroborating line is tested at it alone.
- A property the older fields violate is a
  ``pytest.mark.xfail(strict=True, raises=AssertionError)`` tripwire whose
  reason starts with ``audit-known-defect:``, so the weekly audit lists it;
  when the defect is fixed the tripwire XPASSes, which fails the strict run
  and asks for the marker to be removed. The tests without a marker, the
  evidence-field tests of the last sections among them, must keep passing.
"""

import math
import re
from collections import Counter

import numpy as np
import pytest
from scipy.stats import binom

from _bearing_scenarios import (
    DEFECT_SEED,
    DEFECTS_PRESENT,
    FAULT_FREE,
    FAULT_FREE_SEED,
    FREQUENCIES_6205,
    RESONANCE_HZ,
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
    check_bearing_fault_peak,
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


# ===========================================================================
# The evidence fields: fault-free signals
# ===========================================================================
#
# The properties above, on the fields of the evidence layer. These tests carry
# no marker: the evidence layer is the fix the tripwires above wait for, and
# the older fields keep their values until they switch to it.

#: The research's acquisition configurations: (fs in Hz, duration in s).
EVIDENCE_CONFIGS = [(12000.0, 10.0), (12000.0, 1.0), (25600.0, 4.0), (48000.0, 2.0)]

#: Records per fault-free family and configuration.
EVIDENCE_FAULT_FREE_SEEDS = 60

#: The outcomes that name a fault type.
NAMING_OUTCOMES = ("indicated", "several_types")

#: Quality target for the share of fault-free checks that name a type: one in
#: a thousand, a tenth of the nominal per-check probability. It is an
#: empirical target, not a bound established for every configuration.
#: Measured on 132 000 seeded fault-free checks (six families, four
#: configurations): 45 named, 0.034 %, every family's 95 % upper limit below
#: the target; per configuration it is not shown for every cell (gear tones
#: at 25.6 kHz and 4 s: 4 of 4 000, upper limit 0.26 %). Corroboration given
#: a significant fundamental is about 3 % on white noise but 16 % for the
#: inner race on shaft-modulated noise, through the shaft-rate sidebands the
#: modulation copies: the two nominal probabilities do not bound the named
#: rate. The tests below screen for departures from the target; they do not
#: prove it.
NAMED_OUTCOME_CEILING = 0.001

#: Checks per family in the powered named-rate test.
NAMED_RATE_CHECKS = 3000

#: The deterministic envelope lines each fault-free family holds by
#: construction (Hz). All are shaft orders, so a type named through one of
#: them would mean the shaft-order exclusion failed: the check guards that
#: exclusion. It does not see the modulation mechanism (sideband copies of
#: chance lines), which the named-rate bound and its report cover.
STRUCTURE = {
    "white": [],
    "resonant_coloured": [],
    "random_impacts": [],
    "shaft_lines": [k * SHAFT_HZ for k in range(1, 6)] + [50.0, 100.0],
    "shaft_modulated": [k * SHAFT_HZ for k in range(1, 40)],
    "gear_tones": [k * 41 * SHAFT_HZ for k in range(1, 6)],
}


def _significant(result: dict) -> list[str]:
    return [
        c["fault_type"]
        for c in result["fault_checks"]
        if c["state"] not in ("absent", "not_testable")
    ]


def _names_a_type(result: dict) -> bool:
    return result["outcome"] in NAMING_OUTCOMES


def _named_allowed(n: int) -> int:
    return int(binom.ppf(1 - RATE_TEST_LEVEL, n, NAMED_OUTCOME_CEILING))


def _named_through(result: dict, structure: list[float], seconds: float) -> bool:
    """Whether a supported type rests on a line of the family's structure."""
    df = 1.0 / seconds
    lines = [
        hz
        for c in result["fault_checks"]
        if c["state"] == "supported"
        for hz in [c["line_peak_hz"]]
        + [m["peak_hz"] for m in c["corroboration"] if m["counted"]]
    ]
    return any(abs(hz - s) <= max(3 * df, 0.01 * s) for hz in lines for s in structure)


#: Inconclusive reasons a fault-free family allows by construction: a line no
#: further line corroborates, in any family; a line at a shaft order only where
#: a shaft order lies inside a fault window, which no 6205 window holds at
#: 1797 rpm (asserted below).
ALLOWED_INCONCLUSIVE = {"no_corroboration"}


def _states(result: dict) -> dict[str, str]:
    return {c["fault_type"]: c["state"] for c in result["fault_checks"]}


def _candidates(result: dict) -> list[str]:
    return [c["fault_type"] for c in result["candidates"]]


def test_no_6205_fault_window_holds_a_shaft_order_at_1797_rpm():
    """The premise of ``ALLOWED_INCONCLUSIVE``, from the closed form."""
    lines = {**FREQUENCIES_6205, "BSF": 2 * FREQUENCIES_6205["BSF"]}
    inside = [
        (label, order)
        for label, line in lines.items()
        for order in range(1, 10)
        if abs(order * SHAFT_HZ - line) <= 0.05 * line
    ]

    assert not inside


class TestEvidenceFaultFree:
    @tripwire(
        "bearing-empty-band-residue",
        "T1 evidence layer: the band-pass residue of a pure 50 Hz sine, with "
        "nothing in the envelope band, is named an outer-race fault at 1700 rpm",
    )
    def test_empty_envelope_band_names_no_type(self):
        """``golden_iso`` at 1700 rpm, where a residue line of the band-pass
        filter falls inside the outer-race window. At the fixture's 1800 rpm
        it falls on the window edge, so that speed proves nothing."""
        result = _catalog_check(
            golden_signals()["golden_iso"], fs=GOLDEN_FS, rpm=1700.0
        )

        assert result["outcome"] not in NAMING_OUTCOMES, result["outcome_reason"]

    @pytest.mark.slow
    @pytest.mark.parametrize("fs, seconds", EVIDENCE_CONFIGS)
    @pytest.mark.parametrize("family", list(FAULT_FREE))
    def test_fault_free_family_stays_within_its_bounds(self, family, fs, seconds):
        """AE1 screen, per family and configuration, never pooled:

        - checks with a significant fundamental: at most the binomial bound
          of the nominal 1 % per check;
        - checks that name a type: at most the binomial bound of the quality
          target ``NAMED_OUTCOME_CEILING`` (the rate is measured, not
          derived from the two nominal probabilities: corroboration given a
          significant fundamental depends on the spectrum);
        - none names a type through the family's own deterministic lines;
        - every inconclusive record carries a reason its family allows.

        A screening test, not a proof of the target. With 60 records a cell
        allows 2 named checks: it catches gross failures only (27 % power
        against a 3 % rate). ``test_named_outcome_rate_stays_below_its_ceiling``
        screens the target with more power, at one configuration.
        """
        results = [
            _catalog_check(
                FAULT_FREE[family](
                    _rng(
                        (
                            FAULT_FREE_SEED,
                            1,
                            list(FAULT_FREE).index(family),
                            int(fs),
                            int(seconds),
                            i,
                        )
                    ),
                    fs,
                    seconds,
                    SHAFT_HZ,
                ),
                fs=fs,
            )
            for i in range(EVIDENCE_FAULT_FREE_SEEDS)
        ]
        n = len(results)
        fired = sum(bool(_significant(r)) for r in results)
        named = {i: _candidates(r) for i, r in enumerate(results) if _names_a_type(r)}
        structural = [
            i for i in named if _named_through(results[i], STRUCTURE[family], seconds)
        ]
        reasons = Counter(
            f"{c['fault_type']}:{c['state_reason']}"
            for r in results
            if r["outcome"] == "inconclusive"
            for c in r["fault_checks"]
            if c["state"] in ("unconfirmed", "shared")
        )
        detail = (
            f"{family} at {fs:g} Hz, {seconds:g} s, {n} records: "
            f"{fired} with a significant fundamental, named {named}, "
            f"structural {structural}, inconclusive reasons {dict(reasons)}"
        )

        assert fired <= int(
            binom.ppf(1 - RATE_TEST_LEVEL, n, CHECK_FALSE_ALARM)
        ), detail
        assert len(named) <= _named_allowed(n), detail
        assert not structural, detail
        assert {r.split(":")[1] for r in reasons} <= ALLOWED_INCONCLUSIVE, detail

    @pytest.mark.slow
    @pytest.mark.parametrize("family", list(FAULT_FREE))
    def test_named_outcome_rate_stays_below_its_ceiling(self, family):
        """AE1 screen with power, at 12 kHz and 1 s only: 3 000 checks per
        family. A screening test, not a proof of the target: the bound fails
        a rate five times the target with about 88 % power, twice the target
        with about 4 %, and other configurations are screened only by the
        60-record cells. Types named through a shaft-rate sideband alone are
        reported."""
        n = NAMED_RATE_CHECKS
        named = structural = sideband_only = 0
        index = list(FAULT_FREE).index(family)
        for i in range(n):
            result = _catalog_check(
                FAULT_FREE[family](
                    _rng((FAULT_FREE_SEED, 8, index, i)), FS, 1.0, SHAFT_HZ
                )
            )
            if _names_a_type(result):
                named += 1
                structural += _named_through(result, STRUCTURE[family], 1.0)
                sideband_only += any(
                    {m["kind"] for m in c["corroboration"] if m["counted"]}
                    == {"sideband"}
                    for c in result["fault_checks"]
                    if c["state"] == "supported"
                )

        assert named <= _named_allowed(n), (
            f"{family}: {named} of {n} checks name a type ({sideband_only} through "
            f"a sideband alone); a target of {NAMED_OUTCOME_CEILING:.1%} per check "
            f"allows {_named_allowed(n)}"
        )
        assert structural == 0, f"{family}: {structural} named through its own lines"

    def test_one_second_record_cannot_test_the_cage_line_and_says_so(
        self, white_noise_checks
    ):
        """AE10: at 12 kHz a 1 s record resolves 1 Hz, and the cage line's
        window (11.93 Hz +/- 5 %) holds fewer than 3 bins. No sentence may read
        as if the cage line had been checked, and the step names a longer
        capture."""
        untested = "The cage line (FTF) at 11.93 Hz could not be tested"
        results = white_noise_checks[1.0]
        for result in results:
            ftf = _check(result, "FTF")
            assert (ftf["state"], ftf["state_reason"]) == (
                "not_testable",
                "search_window_too_narrow",
            )
            assert untested in result["outcome_reason"]
            assert "longer capture" in result["outcome_reason"]
            assert "Record a longer capture to test the cage line (FTF)." in (
                result["next_step"]
            )
            assert result["outcome"] in ("no_indication", "inconclusive")
        # A record leaves "no indication" only when a fundamental test fires,
        # which the per-check probability bounds over independent records.
        quiet = [r for r in results if r["outcome"] == "no_indication"]
        fired = int(binom.ppf(1 - RATE_TEST_LEVEL, len(results), CHECK_FALSE_ALARM))
        assert len(quiet) >= len(results) - fired, [r["outcome"] for r in results]
        for result in quiet:
            assert result["outcome_reason"].startswith(
                "No eligible peak (a local maximum inside a search window) at a "
                "tested frequency satisfies the detection criteria"
            )
            assert result["next_step"] == (
                "Record a longer capture to test the cage line (FTF)."
            )


# ===========================================================================
# The evidence fields: defects
# ===========================================================================

EVIDENCE_DEFECT_SEEDS = 20


def _evidence_defect_checks(kind: DefectKind, snr_db: float) -> list[dict]:
    return [
        _catalog_check(
            defect(
                kind,
                _rng((DEFECT_SEED, 1, i, int(snr_db) + 100)),
                FS,
                DEFECT_SECONDS,
                FREQUENCIES_6205,
                SHAFT_HZ,
                snr_db,
            )
        )
        for i in range(EVIDENCE_DEFECT_SEEDS)
    ]


@pytest.fixture(scope="module")
def evidence_defects() -> dict[tuple[str, float], list[dict]]:
    """Checks of every defect scenario the evidence tests read, built once."""
    cases = [
        (kind, snr)
        for kind in ("outer", "inner", "rolling_element")
        for snr in (-15.0, -10.0)
    ]
    cases += [("rolling_element", 0.0), ("cage", -15.0), ("combined", -10.0)]
    return {case: _evidence_defect_checks(*case) for case in cases}


class TestEvidenceDefects:
    @pytest.mark.parametrize("snr_db", [-15.0, -10.0])
    @pytest.mark.parametrize("kind", ["outer", "inner", "rolling_element"])
    def test_defect_is_supported_and_ordered_first(
        self, evidence_defects, kind, snr_db
    ):
        """AE3: the true type is supported and the strongest candidate on at
        least 95 % of seeds."""
        (label,) = DEFECTS_PRESENT[kind]
        results = evidence_defects[(kind, snr_db)]
        first = sum(_candidates(r)[:1] == [label] for r in results)

        assert first >= _bar(EVIDENCE_DEFECT_SEEDS), (
            f"{kind} at {snr_db:g} dB: {label} supported and first on {first} of "
            f"{EVIDENCE_DEFECT_SEEDS} seeds; candidates "
            f"{[_candidates(r) for r in results]}"
        )

    @pytest.mark.parametrize("kind", ["outer", "inner"])
    def test_race_defect_supports_no_other_type(self, evidence_defects, kind):
        """AE3: at -15 dB no other type is supported on at least 95 % of
        seeds."""
        results = evidence_defects[(kind, -15.0)]
        clean = sum(not set(_candidates(r)) - DEFECTS_PRESENT[kind] for r in results)

        assert clean >= _bar(EVIDENCE_DEFECT_SEEDS), (
            f"{kind} at -15 dB: another type supported on "
            f"{EVIDENCE_DEFECT_SEEDS - clean} of {EVIDENCE_DEFECT_SEEDS} seeds; "
            f"candidates {[_candidates(r) for r in results]}"
        )

    def test_strong_rolling_element_defect_is_ordered_first(self, evidence_defects):
        """At 0 dB strong cage-rate sidebands fall inside other windows, so
        several supported types is an accepted outcome; the rolling element
        is still the strongest candidate."""
        results = evidence_defects[("rolling_element", 0.0)]
        first = sum(_candidates(r)[:1] == ["BSF"] for r in results)

        assert first >= _bar(EVIDENCE_DEFECT_SEEDS), (
            f"rolling element at 0 dB first on {first} of {EVIDENCE_DEFECT_SEEDS} "
            f"seeds; candidates {[_candidates(r) for r in results]}"
        )

    def test_cage_defect_is_supported_and_ordered_first(self, evidence_defects):
        """A 10 s record resolves the cage line's window."""
        results = evidence_defects[("cage", -15.0)]
        first = sum(_candidates(r)[:1] == ["FTF"] for r in results)

        assert first >= _bar(EVIDENCE_DEFECT_SEEDS), (
            f"cage defect first on {first} of {EVIDENCE_DEFECT_SEEDS} seeds; "
            f"candidates {[_candidates(r) for r in results]}"
        )

    def test_rolling_element_defect_is_not_named_a_cage_defect(self):
        """A strong 6203 rolling-element defect: its 2 x BSF line (119.4 Hz)
        lies at the 4th shaft order, so the rolling element cannot be
        supported, while its cage-rate modulation supports the cage line. The
        cage is not singled out while the rolling-element line is
        significant."""
        frequencies = {"BPFO": 91.44, "BPFI": 148.16, "BSF": 59.72, "FTF": 11.43}
        results = [
            check_all_bearing_faults(
                defect(
                    "rolling_element",
                    _rng((DEFECT_SEED, 12, i)),
                    FS,
                    DEFECT_SECONDS,
                    frequencies,
                    SHAFT_HZ,
                    0.0,
                ),
                FS,
                "6203",
                RPM_6205,
            )
            for i in range(EVIDENCE_DEFECT_SEEDS)
        ]
        named_cage = [
            i
            for i, r in enumerate(results)
            if r["outcome"] == "indicated" and _candidates(r) == ["FTF"]
        ]
        held = [
            r
            for r in results
            if _check(r, "FTF")["state_reason"] == "rolling_element_line_present"
        ]

        assert not named_cage, f"named a cage defect on seeds {named_cage}"
        assert held, "no record held the cage back for the rolling-element line"
        assert all(
            "A rolling-element defect cannot be excluded either" in r["outcome_reason"]
            for r in held
        )

    def test_combined_defects_are_several_types_with_both_levels(
        self, evidence_defects
    ):
        """AE4: both races, each with its level, and neither singled out."""
        for result in evidence_defects[("combined", -10.0)]:
            assert result["outcome"] == "several_types", result["outcome_reason"]
            assert set(_candidates(result)) == {"BPFO", "BPFI"}
            assert all(c["level_db"] is not None for c in result["candidates"])
            levels = [c["level_db"] for c in result["candidates"]]
            assert levels == sorted(levels, reverse=True)
            assert "does not single out one of them" in result["outcome_reason"]
            step = result["next_step"]
            assert "the outer race" in step and "the inner race" in step
            text = f"{result['outcome_reason']} {step}".lower()
            assert "most likely" not in text and "replace" not in text

    def test_uncorroborated_line_is_inconclusive_with_its_reason(self):
        """AE5: the MFPT baseline's line at twice the mains frequency in the
        BPFI window is significant, but no line a bearing defect would add
        stands beside it."""
        signal, fs, frequencies, rpm = load_mfpt("real_test/baseline_3.csv")
        result = check_frequency_set(signal, fs, frequencies, rpm)
        bpfi = _check(result, "BPFI")

        assert result["outcome"] == "inconclusive"
        assert (bpfi["state"], bpfi["state_reason"]) == (
            "unconfirmed",
            "no_corroboration",
        )
        assert "but none of the further lines its kinematics predict" in (
            result["outcome_reason"]
        )
        assert result["next_step"].startswith("Repeat the measurement")

    def test_candidate_order_does_not_follow_the_frequency_dict(self):
        """AE6, on the records of ``TestOrderIndependence``."""
        forward = {k: FREQUENCIES_6205[k] for k in ("BPFO", "BPFI")}
        backward = dict(reversed(list(forward.items())))
        fields = ("outcome", "candidates", "outcome_reason", "next_step")
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

            assert len(a["candidates"]) == 2
            levels = [c["level_db"] for c in a["candidates"]]
            assert levels == sorted(levels, reverse=True)
            assert [a[k] for k in fields] == [b[k] for k in fields]

    def test_evidence_fields_are_deterministic(self):
        signal = defect(
            "inner",
            _rng((DEFECT_SEED, 1, 0, 85)),
            FS,
            DEFECT_SECONDS,
            FREQUENCIES_6205,
            SHAFT_HZ,
            -15.0,
        )

        assert _catalog_check(signal) == _catalog_check(signal)


# ===========================================================================
# The evidence fields: shared lines and shaft orders
# ===========================================================================


def _modulated_carrier(
    rng: np.random.Generator, lines_hz: list[float], seconds: float = DEFECT_SECONDS
) -> np.ndarray:
    """A carrier at the resonance amplitude-modulated at each line, plus noise.

    The modulation stays positive, so the envelope holds exactly these lines.
    """
    t = np.arange(int(FS * seconds)) / FS
    depth = 0.8 / len(lines_hz)
    modulation = 1.0 + sum(depth * np.sin(2 * np.pi * f * t) for f in lines_hz)
    return np.sin(2 * np.pi * RESONANCE_HZ * t) * modulation + 0.05 * (
        rng.standard_normal(t.size)
    )


class TestEvidenceSharedLinesAndShaftOrders:
    def test_shared_line_corroborates_neither_type(self):
        """The shared-line scenario of ``TestSingleDefect``: on every seed the
        line at 3 x BPFO, which is also twice the 1.5 x BPFO line in the BPFI
        window, is excluded from both families as shared, and BPFO is
        supported by its own second harmonic. BPFI is supported only where
        noise corroborates it, bounded as in the older-field test."""
        line_hz = 3 * FREQUENCIES_6205["BPFO"]
        failing, bpfi_supported = {}, 0
        for i in range(SHARED_LINE_SEEDS):
            result = _catalog_check(
                shared_line(
                    _rng(DEFECT_SEED + 300 + i), FS, DEFECT_SECONDS, FREQUENCIES_6205
                )
            )
            bpfo, bpfi = _check(result, "BPFO"), _check(result, "BPFI")
            third = next(m for m in bpfo["corroboration"] if m["harmonic"] == 3)
            second = next(m for m in bpfi["corroboration"] if m["harmonic"] == 2)
            own = next(m for m in bpfo["corroboration"] if m["harmonic"] == 2)
            problems = [
                f"{name} {line['harmonic']}x at {line['peak_hz']} Hz: "
                f"significant={line['significant']} excluded_by={line['excluded_by']}"
                for name, line in (("BPFO", third), ("BPFI", second))
                if not (line["significant"] and line["excluded_by"] == "shared")
                or abs(line["peak_hz"] - line_hz) > 0.01 * line_hz
            ]
            if bpfo["state"] != "supported" or not own["counted"]:
                problems.append(
                    f"BPFO {bpfo['state']}, own 2x counted={own['counted']}"
                )
            if problems:
                failing[i] = problems
            bpfi_supported += bpfi["state"] == "supported"
        noise_support = BPFI_OTHER_CANDIDATES * CORROBORATION_FALSE_ALARM
        allowed = int(binom.ppf(1 - RATE_TEST_LEVEL, SHARED_LINE_SEEDS, noise_support))
        message = (
            f"shared line at {line_hz:.1f} Hz: BPFI supported on {bpfi_supported} "
            f"of {SHARED_LINE_SEEDS} seeds (at most {allowed} allowed); failing on "
            f"{len(failing)} seeds, first three: {dict(list(failing.items())[:3])}"
        )

        assert not failing, message
        assert bpfi_supported <= allowed, message

    def test_corroborating_line_at_a_shaft_order_counts_for_nothing(self):
        """One signal, lines at 105 and 210 Hz, a declared BPFO of 105 Hz. At
        1800 rpm the 210 Hz harmonic is the 7th shaft order and corroborates
        nothing; at 1750 rpm it is no shaft order and corroborates."""
        signal = _modulated_carrier(_rng((DEFECT_SEED, 2, 0)), [105.0, 210.0])
        at_order = check_frequency_set(signal, FS, {"BPFO": 105.0}, 1800.0)
        off_order = check_frequency_set(signal, FS, {"BPFO": 105.0}, 1750.0)

        second = at_order["fault_checks"][0]["corroboration"][0]
        assert (second["harmonic"], second["significant"]) == (2, True)
        assert (second["excluded_by"], second["counted"]) == ("shaft_order", False)
        assert _states(at_order) == {"BPFO": "unconfirmed"}
        assert at_order["outcome"] == "inconclusive"
        assert (
            "those of the further lines its kinematics predict that also stand "
            "above the floor lie at a shaft order"
        ) in at_order["outcome_reason"]
        assert _states(off_order) == {"BPFO": "supported"}
        assert off_order["outcome"] == "indicated"

    def test_detected_line_at_a_shaft_order_is_unconfirmed(self):
        """Lines at 120 and 240 Hz, a declared BPFO of 120 Hz at 1800 rpm: the
        line itself is the 4th shaft order."""
        signal = _modulated_carrier(_rng((DEFECT_SEED, 2, 1)), [120.0, 240.0])
        result = check_frequency_set(signal, FS, {"BPFO": 120.0}, 1800.0)
        check = result["fault_checks"][0]

        assert (check["state"], check["state_reason"]) == (
            "unconfirmed",
            "line_at_shaft_order",
        )
        assert "integer multiple of the shaft rotation frequency" in (
            result["outcome_reason"]
        )

    def test_indicated_reason_names_another_line_that_is_shared(self):
        """Lines at 100, 200, 300 and 400 Hz; BPFO declared at 100 Hz and a
        label X at 200 Hz, BPFO's second harmonic, at 2220 rpm (no shaft
        order within 1 % of any line). X's line lies in BPFO's family, so X is
        shared; BPFO keeps its own third harmonic and is indicated. The reason
        must not hide X's line."""
        signal = _modulated_carrier(
            _rng((DEFECT_SEED, 2, 2)), [100.0, 200.0, 300.0, 400.0]
        )
        result = check_frequency_set(signal, FS, {"BPFO": 100.0, "X": 200.0}, 2220.0)

        assert _states(result) == {"BPFO": "supported", "X": "shared"}
        assert _check(result, "X")["shared_with"] == ["BPFO"]
        assert result["outcome"] == "indicated"
        assert "No other tested type is supported in this spectrum." in (
            result["outcome_reason"]
        )
        assert (
            "The X line at 200.00 Hz stands above the local noise floor, but it lies "
            "where the BPFO check also predicts a line, so it supports neither."
        ) in result["outcome_reason"]


# ===========================================================================
# The evidence fields: arbitrary labels and the single-frequency route
# ===========================================================================


class TestEvidenceRoutes:
    def test_arbitrary_label_gets_harmonics_only_and_neutral_wording(self):
        """Two in-band tones 12 shaft orders apart, as a gear mesh is: their
        envelope beats at the mesh frequency and its harmonics. A "GMF" label
        gets the line test and harmonic corroboration only (KTD10): no
        sideband test, no shaft-order discount, no bearing wording."""
        mesh_hz = 12 * SHAFT_HZ
        t = np.arange(int(FS * DEFECT_SECONDS)) / FS
        signal = (
            np.sin(2 * np.pi * 2000.0 * t)
            + np.sin(2 * np.pi * (2000.0 + mesh_hz) * t)
            + 0.05 * _rng((DEFECT_SEED, 3, 0)).standard_normal(t.size)
        )
        result = check_frequency_set(signal, FS, {"GMF": mesh_hz}, RPM_6205)
        check = result["fault_checks"][0]
        text = f"{result['outcome_reason']} {result['next_step']}".lower()

        assert result["outcome"] == "indicated"
        assert (check["state"], check["state_reason"]) == ("supported", "corroborated")
        assert {c["kind"] for c in check["corroboration"]} == {"harmonic"}
        assert not any(c["excluded_by"] for c in check["corroboration"])
        assert "gmf line" in text
        for word in ("bearing", "kinematics", "defect", "race"):
            assert word not in text, word

    def test_single_frequency_route_carries_the_evidence_fields(self):
        """``check_bearing_fault_peak`` has no shaft speed: the line test and
        harmonics only, and the inner race gets no sideband test."""
        signal = defect(
            "inner",
            _rng((DEFECT_SEED, 1, 0, 90)),
            FS,
            DEFECT_SECONDS,
            FREQUENCIES_6205,
            SHAFT_HZ,
            -10.0,
        )
        check = check_bearing_fault_peak(
            signal, FS, FREQUENCIES_6205["BPFI"], "BPFI", "6205"
        )

        assert check["state"] == "supported"
        assert check["line_test"]["false_alarm_probability"] == CHECK_FALSE_ALARM
        assert {c["kind"] for c in check["corroboration"]} == {"harmonic"}

    def test_rolling_element_sidebands_need_a_declared_cage_rate(self):
        """On the declared-frequency route the cage rate is the FTF entry: with
        it the rolling-element line gets its cage-rate sidebands tested;
        without it the reason says that test did not run."""
        signal = defect(
            "rolling_element",
            _rng((DEFECT_SEED, 1, 0, 91)),
            FS,
            DEFECT_SECONDS,
            FREQUENCIES_6205,
            SHAFT_HZ,
            -10.0,
        )
        both = {k: FREQUENCIES_6205[k] for k in ("BSF", "FTF")}
        with_cage = check_frequency_set(signal, FS, both, RPM_6205)
        without = check_frequency_set(
            signal, FS, {"BSF": FREQUENCIES_6205["BSF"]}, RPM_6205
        )

        kinds = {c["kind"] for c in _check(with_cage, "BSF")["corroboration"]}
        assert {"sideband", "subharmonic"} <= kinds
        kinds = {c["kind"] for c in _check(without, "BSF")["corroboration"]}
        assert "sideband" not in kinds
        assert "The cage rate was not given" in without["outcome_reason"]
        assert "The cage rate was not given" not in with_cage["outcome_reason"]


# ===========================================================================
# The evidence fields: narrow envelope bands and the frequency range
# ===========================================================================

#: Narrow-band configurations where the floor reference used to reach past
#: the envelope's content: (fs, seconds, explicit band or None, families).
NARROW_BAND_CASES = [
    (1400.0, 10.0, None, "white"),
    (1400.0, 10.0, None, "random_impacts"),
    (12000.0, 4.0, (2000.0, 2200.0), "white"),
    (12000.0, 4.0, (2000.0, 2200.0), "resonant_coloured"),
]


def _checks_for_power(probability: float) -> int:
    """Fewest checks whose bound fails twice ``probability`` with 90 % power."""
    n = 500
    while True:
        allowed = binom.ppf(1 - RATE_TEST_LEVEL, n, probability)
        if binom.sf(allowed, n, 2 * probability) >= 0.9:
            return n
        n += 100


class TestEvidenceNarrowBands:
    def test_lines_beyond_the_usable_band_are_not_testable_with_their_reason(self):
        """A 150 Hz band is used below 135 Hz (0.9 of its width): the search
        windows of the 6205's inner-race and rolling-element lines reach
        beyond, the outer-race window stays below."""
        signal = white(_rng((FAULT_FREE_SEED, 6, 0)), FS, 2.0, SHAFT_HZ)
        result = check_all_bearing_faults(
            signal, FS, "6205", RPM_6205, envelope_freq_range=(2000.0, 2150.0)
        )

        for label in ("BPFI", "BSF"):
            check = _check(result, label)
            assert (check["state"], check["state_reason"]) == (
                "not_testable",
                "outside_usable_envelope_band",
            )
        assert _check(result, "BPFO")["state"] != "not_testable"
        assert "reaches beyond the usable part of the envelope band" in (
            result["outcome_reason"]
        )
        assert "Use a wider envelope band" in result["next_step"]

    def test_a_frequency_above_the_spectrum_asks_for_a_higher_sampling_rate(self):
        signal = white(_rng((FAULT_FREE_SEED, 6, 1)), FS, 2.0, SHAFT_HZ)
        result = check_frequency_set(
            signal, FS, {"BPFO": FREQUENCIES_6205["BPFO"], "GMF": 7000.0}, RPM_6205
        )
        gmf = _check(result, "GMF")

        assert (gmf["state"], gmf["state_reason"]) == (
            "not_testable",
            "above_frequency_range",
        )
        assert (
            "Record at a higher sampling rate, with an envelope band wide enough to "
            "include it, to test the GMF line."
        ) in result["next_step"]
        assert "longer capture" not in result["next_step"]

    @pytest.mark.slow
    @pytest.mark.parametrize("fs, seconds, band, family", NARROW_BAND_CASES)
    def test_whole_check_false_alarm_rate_holds_in_narrow_bands(
        self, fs, seconds, band, family
    ):
        """Checks with a significant fundamental, on fault-free noise, are
        screened against the nominal 1 % per check, per family, with 90 %
        power against twice that rate. The screen does not detect the excess
        of about 1.2 to 1.5 times the nominal measured where a tested line
        lies at 0.4 to 0.8 of the envelope band width; narrow-band control is
        improved by the usable-band limit, not solved."""
        n = _checks_for_power(CHECK_FALSE_ALARM)
        allowed = int(binom.ppf(1 - RATE_TEST_LEVEL, n, CHECK_FALSE_ALARM))
        fired = 0
        for i in range(n):
            signal = FAULT_FREE[family](
                _rng((FAULT_FREE_SEED, 7, list(FAULT_FREE).index(family), int(fs), i)),
                fs,
                seconds,
                SHAFT_HZ,
            )
            result = check_all_bearing_faults(
                signal, fs, "6205", RPM_6205, envelope_freq_range=band
            )
            fired += any(
                c["state"] not in ("absent", "not_testable")
                for c in result["fault_checks"]
            )

        assert fired <= allowed, (
            f"{family} at {fs:g} Hz, band {band}: {fired} of {n} checks with a "
            f"significant fundamental; a nominal 1 % per check allows {allowed}"
        )

    def test_corroborating_lines_beyond_the_usable_band_are_named_with_a_step(self):
        """A 6203 outer-race defect exciting a 2.1 kHz resonance, seen through
        a 2000-2200 Hz band: the outer-race line is found, its harmonics
        (from 182.9 Hz) reach beyond the usable 180 Hz, and the reason and
        step say so instead of calling them absent."""
        frequencies = {"BPFO": 91.44, "BPFI": 148.16, "BSF": 59.72, "FTF": 11.43}
        signal = defect(
            "outer",
            _rng((DEFECT_SEED, 9, 0)),
            FS,
            4.0,
            frequencies,
            SHAFT_HZ,
            -10.0,
            resonance_hz=2100.0,
        )
        result = check_all_bearing_faults(
            signal, FS, "6203", RPM_6205, envelope_freq_range=(2000.0, 2200.0)
        )
        bpfo = _check(result, "BPFO")

        assert (bpfo["state"], bpfo["state_reason"]) == (
            "unconfirmed",
            "no_corroboration",
        )
        assert {m["untestable_reason"] for m in bpfo["corroboration"]} == {
            "outside_usable_envelope_band"
        }
        assert (
            "none of the further lines its kinematics predict could be tested: they "
            "reach beyond the usable part of the envelope band"
        ) in result["outcome_reason"]
        assert (
            "Use a wider envelope band, which may need a higher sampling rate, to "
            "test the further lines of the outer-race line (BPFO)."
        ) in result["next_step"]
