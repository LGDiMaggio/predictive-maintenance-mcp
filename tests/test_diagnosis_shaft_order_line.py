"""The strongest spectral line at 1x or 2x of the shaft speed in diagnose_vibration.

The synthesis reports, as a measurement, when the strongest bin of the raw
amplitude spectrum lies within 10 % of 1x or 5 % of 2x of the declared shaft
frequency. It names no fault, counts nothing in ``evidence_strength`` and adds
no recommendation: a shaft-rate line is expected on a rotating machine, and the
order alone does not separate normal running from the faults and non-fault
sources that share it.

``TestTriggerSet`` pins WHEN the line is reported: the same peak, the same open
windows (0.9 < ratio < 1.1, 1.9 < ratio < 2.1) as the released code. Its
assertions hold for the old and the new wording alike.
"""

import itertools
import re

import numpy as np
import pytest

from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    _bearing_check,
    _synthesize_diagnosis,
    diagnose_vibration,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
)

NO_MODEL = "nonexistent_model_shaft_order_line"
ROUTINE = "Continue routine monitoring per maintenance schedule."

#: The approved model-facing sentences, rendered (W1-R and W2-R of the owner
#: gate). Pinned in full: a reworded reading list fails here first.
DISCRIMINATION = (
    "Narrowing these down needs more than this check, such as a finer analysis "
    "of the line's frequency, a baseline from the same machine at the same "
    "condition, phase, a second radial or an axial channel, or several speeds. "
    "Establishing a cause needs a physical check of the machine."
)
REPEAT = (
    "Another analysis that reads this same line of the same signal repeats the "
    "observation and does not corroborate it."
)


def one_x_sentence(ratio: str) -> str:
    return (
        f"The strongest line of the spectrum is at {ratio} times the declared "
        "shaft frequency (within 10 % of 1x). Its dominance does not identify a "
        "fault, and it is not counted as a finding. This check cannot "
        "distinguish the 1x response of a machine in normal condition from "
        "unbalance, a bent or bowed rotor, eccentricity, misalignment, "
        "mechanical looseness or a soft foot, a worn or damaged coupling, a "
        "resonance near running speed, vibration transmitted from another "
        "machine, or a component at the electrical supply frequency, which lies "
        f"close to 1x on a two-pole motor. {DISCRIMINATION} {REPEAT}"
    )


def two_x_sentence(ratio: str) -> str:
    return (
        f"The strongest line of the spectrum is at {ratio} times the declared "
        "shaft frequency (within 5 % of 2x). Its dominance does not identify a "
        "fault, and it is not counted as a finding. This check cannot "
        "distinguish the 2x response of a machine in normal condition from "
        "misalignment, mechanical looseness, a rub, a shaft crack, a shaft or "
        "rotor with asymmetric stiffness, a resonance near twice running speed, "
        "the passing tone of a rotor with two blades or lobes, vibration "
        "transmitted from another machine, a component at the electrical supply "
        "frequency, which lies close to 2x on a four-pole motor, or, on a "
        "two-pole motor, the electromagnetic vibration at twice the supply "
        "frequency, which lies close to 2x (exactly at 2x on a synchronous "
        f"motor). {DISCRIMINATION} {REPEAT}"
    )


#: Phrases of the released fault reading that no output may carry again.
RETIRED = (
    "possible unbalance",
    "possible misalignment",
    "Check rotor balance.",
    "Check alignment and coupling condition.",
)

#: A fault ATTRIBUTED: unbalance, misalignment or looseness named as what the
#: record shows, as likely, possible or suspected, or as a cause. Naming one
#: as a reading the check cannot distinguish ("cannot distinguish ... from
#: unbalance") is not an attribution and stays allowed.
_FAULT = r"(?:an?\s+|the\s+)?(?:rotor\s+|shaft\s+)?(?:unbalance|imbalance|misalignment|looseness)\b"
ATTRIBUTION = re.compile(
    r"\b(?:possible|probable|likely|suspected|indicat\w*|suggest\w*|"
    r"consistent with|caused by|due to|points? to|evidence of|signs? of)\s+" + _FAULT,
    re.IGNORECASE,
)
#: A fault declared ABSENT, which a line elsewhere never establishes.
ABSENCE = re.compile(
    r"\b(?:no|without|rules? out|excludes?)\s+" + _FAULT, re.IGNORECASE
)
#: Fault-specific maintenance actions: the released line's two, and the
#: recommendation table's machine-level ones.
FAULT_SPECIFIC_ACTIONS = (
    "Check rotor balance.",
    "Check alignment and coupling condition.",
    "Balance rotor",
    "Realign coupling",
    "Tighten foundation bolts",
)


def assert_no_fault_attribution(texts, actions):
    """No text attributes a shaft-order fault and no action is fault-specific."""
    for text in texts:
        assert ATTRIBUTION.search(text) is None, text
    for action in actions:
        assert not any(a in action for a in FAULT_SPECIFIC_ACTIONS), action


class TestTheAttributionCheck:
    """The check catches the released reading and lets alternatives through."""

    @pytest.mark.parametrize(
        "text",
        [
            "Dominant peak at 1x shaft speed — possible unbalance.",
            "Dominant peak at 2x shaft speed — possible misalignment.",
            "The spectrum is consistent with misalignment.",
            "This indicates rotor unbalance.",
        ],
    )
    def test_attributions_are_caught(self, text):
        assert ATTRIBUTION.search(text)

    def test_readings_the_check_cannot_distinguish_are_not_attributions(self):
        assert ATTRIBUTION.search(one_x_sentence("1.00")) is None
        assert ATTRIBUTION.search(two_x_sentence("2.00")) is None
        assert ABSENCE.search(one_x_sentence("1.00")) is None
        assert ABSENCE.search(two_x_sentence("2.00")) is None


def _tone(freq_hz, seconds=1.0, fs=10000, amplitude=0.01, noise=0.0, seed=0):
    t = np.arange(int(seconds * fs)) / fs
    x = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(t.size)
    return x


SHAFT_HZ = 30.0  # rpm 1800
RPM = 1800.0

#: The refused ISO block of a diagnosis without a declared unit.
REFUSED_ISO = {
    "status": "refused",
    "signal_id": "s",
    "reason": "Signal unit not declared.",
    "remedy": "Declare the unit.",
}


def _synthesize(
    peak_hz,
    *,
    iso=None,
    bearing_faults=None,
    anomaly=None,
    rpm=RPM,
):
    """Synthesis with a stub FFT summary whose strongest line is *peak_hz*."""
    return _synthesize_diagnosis(
        fft_summary={"peak_frequency_hz": peak_hz},
        psd_summary={},
        stft_summary={},
        bearing_faults=bearing_faults,
        bearing_check=_bearing_check("assessed" if bearing_faults else "not_requested"),
        iso_severity=iso or REFUSED_ISO,
        anomaly=anomaly,
        rpm=rpm,
    )


def _order_sentences(text: str) -> list[str]:
    """Sentences of *text* that mention 1x or 2x (old or new wording)."""
    return [s for s in text.split(". ") if "1x" in s or "2x" in s]


class TestTriggerSet:
    """The windows are open intervals around 1x and 2x, as released."""

    @pytest.mark.parametrize("ratio", [0.9, 1.1, 1.9, 2.1])
    def test_window_edges_report_no_order_line(self, ratio):
        text, _, _ = _synthesize(round(ratio * SHAFT_HZ, 2))
        assert _order_sentences(text) == []

    @pytest.mark.parametrize("ratio", [1.5, 3.0, 0.5])
    def test_lines_outside_the_windows_report_no_order_line(self, ratio):
        text, _, _ = _synthesize(ratio * SHAFT_HZ)
        assert _order_sentences(text) == []

    @pytest.mark.parametrize(
        ("peak_hz", "order"),
        [
            (27.01, "1x"),  # ratio 0.9003
            (30.0, "1x"),
            (32.99, "1x"),  # ratio 1.0997
            (57.01, "2x"),  # ratio 1.9003
            (60.0, "2x"),
            (62.99, "2x"),  # ratio 2.0997
        ],
    )
    def test_lines_inside_the_windows_report_their_order(self, peak_hz, order):
        text, _, _ = _synthesize(peak_hz)
        sentences = _order_sentences(text)
        assert sentences, "no sentence names the order"
        assert all(order in s for s in sentences)


class TestMeasuredLine:
    """Dominant 1x or 2x without the fault the old sentences named."""

    def test_quiet_machine_with_a_1x_line(self):
        """A quiet 25 Hz tone at 1500 rpm, Zone A: no fault, no grade, the
        routine line only (the released code said 'possible unbalance',
        graded it 'weak' and recommended a balance check)."""
        result = diagnose_vibration(
            _tone(25.0, noise=0.001),
            10000,
            rpm=1500,
            signal_unit="g",
            anomaly_model_name=NO_MODEL,
        )
        assert result["iso_severity"]["zone"] == "A"
        text = result["overall_diagnosis"]
        assert one_x_sentence("1.00") in text
        # Unbalance is named, as one reading the check cannot distinguish,
        # and is not attributed.
        assert "from unbalance" in text
        assert_no_fault_attribution([text], result["recommendations"])
        assert result["evidence_strength"] == "none"
        assert result["recommendations"] == [ROUTINE]

    def test_twice_supply_frequency_on_a_two_pole_motor_reads_as_2x(self):
        """100 Hz at 2950 rpm (two-pole motor on 50 Hz): ratio 2.03, inside
        the 2x window, with no misalignment in the record."""
        result = diagnose_vibration(
            _tone(100.0), 10000, rpm=2950, anomaly_model_name=NO_MODEL
        )
        text = result["overall_diagnosis"]
        assert two_x_sentence("2.03") in text
        assert "electromagnetic vibration at twice the supply frequency" in text
        assert result["evidence_strength"] == "none"

    def test_supply_frequency_on_a_four_pole_motor_reads_as_2x(self):
        """50 Hz at 1475 rpm (four-pole motor on 50 Hz): ratio 2.03."""
        result = diagnose_vibration(
            _tone(50.0), 10000, rpm=1475, anomaly_model_name=NO_MODEL
        )
        assert two_x_sentence("2.03") in result["overall_diagnosis"]
        assert result["evidence_strength"] == "none"

    def test_supply_frequency_on_a_two_pole_motor_reads_as_1x(self):
        """50 Hz at 2950 rpm: ratio 1.02, a supply-frequency component."""
        result = diagnose_vibration(
            _tone(50.0), 10000, rpm=2950, anomaly_model_name=NO_MODEL
        )
        assert one_x_sentence("1.02") in result["overall_diagnosis"]
        assert result["evidence_strength"] == "none"

    def test_a_3x_line_reports_no_order_and_no_absence(self):
        """A strongest line at 3x: no order sentence, and nothing that reads
        as unbalance, misalignment or looseness being present or absent."""
        text, strength, recommendations = _synthesize(3.0 * SHAFT_HZ)
        assert _order_sentences(text) == []
        assert not any(phrase in text for phrase in RETIRED)
        assert ABSENCE.search(text) is None
        assert_no_fault_attribution([text], recommendations)
        assert strength == "none"


@pytest.fixture(scope="module")
def bearing_variants():
    """Bearing summaries for each claim level the synthesis distinguishes.

    The 3 kHz carrier AM-modulated at BPFO of a 6205 at 1800 rpm: 1 s gives a
    qualified claim (cage line untestable), 10 s single-spectrum evidence,
    and the 10 s record declared a generated signal without noise a withheld
    claim.
    """
    from test_diagnosis_pipeline import _bpfo_carrier

    def check(seconds, eligibility="unknown"):
        return check_all_bearing_faults(
            signal=_bpfo_carrier(seconds),
            fs=10000,
            bearing_id="6205",
            rpm=1800,
            signal_id="b",
            acquisition_eligibility=eligibility,
        )

    variants = {
        "none": None,
        "qualified": check(1.0),
        "single_spectrum_evidence": check(10.0),
        "withheld": check(10.0, "known_inapplicable"),
    }
    for level in ("qualified", "single_spectrum_evidence", "withheld"):
        assert variants[level]["claim"]["level"] == level
    return variants


def _iso(zone):
    if zone is None:
        return REFUSED_ISO
    return {
        "status": "assessed",
        "zone": zone,
        "rms_velocity_mm_s": {"A": 1.0, "B": 2.0, "C": 5.0, "D": 9.0}[zone],
        "severity_level": "x",
    }


ANOMALIES = {
    "none": None,
    "healthy": {"overall_health": "Healthy", "anomaly_ratio": 0.0},
    "suspicious": {"overall_health": "Suspicious", "anomaly_ratio": 0.2},
    "faulty": {"overall_health": "Faulty", "anomaly_ratio": 0.5},
}


class TestEvidenceDoesNotDependOnTheLine:
    """Same inputs, same grade and actions wherever the strongest line sits."""

    @pytest.mark.parametrize(
        ("zone", "bearing", "anomaly"),
        list(
            itertools.product(
                [None, "A", "B", "C", "D"],
                ["none", "qualified", "single_spectrum_evidence", "withheld"],
                list(ANOMALIES),
            )
        ),
    )
    def test_grade_and_actions_are_the_same_at_1x_2x_and_off_order(
        self, bearing_variants, zone, bearing, anomaly
    ):
        outcomes = set()
        for peak_hz in (SHAFT_HZ, 2 * SHAFT_HZ, 1.5 * SHAFT_HZ, 3 * SHAFT_HZ):
            _, strength, recommendations = _synthesize(
                peak_hz,
                iso=_iso(zone),
                bearing_faults=bearing_variants[bearing],
                anomaly=ANOMALIES[anomaly],
            )
            outcomes.add((strength, tuple(recommendations)))
        assert len(outcomes) == 1, outcomes


class TestGradeWithA1xLine:
    """The grade a 1x line no longer raises: only the other findings count."""

    @pytest.mark.parametrize(
        ("zone", "anomaly", "grade"),
        [
            (None, "none", "none"),  # was weak
            (None, "suspicious", "weak"),  # 0.5, was weak (1.5)
            ("C", "none", "weak"),  # 1.0, was moderate
            ("C", "suspicious", "weak"),  # 1.5, was moderate
            ("D", "none", "moderate"),  # 2.0, was strong
            ("D", "suspicious", "moderate"),  # 2.5, was strong
            ("D", "faulty", "strong"),  # 3.0, was strong
        ],
    )
    def test_grade(self, zone, anomaly, grade):
        _, strength, _ = _synthesize(
            SHAFT_HZ, iso=_iso(zone), anomaly=ANOMALIES[anomaly]
        )
        assert strength == grade


class TestNoActionFromTheLine:
    def test_zone_b_with_a_1x_line_recommends_routine_monitoring_only(self):
        _, _, recommendations = _synthesize(SHAFT_HZ, iso=_iso("B"))
        assert recommendations == [ROUTINE]

    def test_zone_d_with_a_1x_line_keeps_only_the_zone_action(self):
        _, _, recommendations = _synthesize(SHAFT_HZ, iso=_iso("D"))
        assert recommendations == ["IMMEDIATE ACTION: Stop machine and inspect."]

    def test_zone_c_with_a_2x_line_keeps_only_the_zone_action(self):
        _, _, recommendations = _synthesize(2 * SHAFT_HZ, iso=_iso("C"))
        assert recommendations == ["Schedule maintenance within next planned shutdown."]


class TestRetiredWording:
    """No server-authored output of the diagnosis or its report path carries
    the released fault reading of the line, or the report's old energy
    reading, whatever the ISO and bearing context."""

    @pytest.mark.parametrize(
        ("freq_hz", "rpm"), [(30.0, 1800), (60.0, 1800), (50.0, 2950)]
    )
    @pytest.mark.parametrize("unit", [None, "g"])
    @pytest.mark.parametrize("bearing_id", [None, "6205"])
    def test_the_line_attributes_no_fault_and_asks_for_no_fault_action(
        self, freq_hz, rpm, unit, bearing_id
    ):
        """The fault names may appear as readings the check cannot
        distinguish; no output attributes one, and no action is specific to
        unbalance, misalignment or looseness."""
        from predictive_maintenance_mcp.decision_support.advisory import (
            build_advisory,
            collect_statements,
        )

        diagnosis = diagnose_vibration(
            _tone(freq_hz, amplitude=1.0, noise=0.01),
            10000,
            rpm=rpm,
            signal_id="tripwire",
            bearing_id=bearing_id,
            signal_unit=unit,
            anomaly_model_name=NO_MODEL,
        )
        texts = [
            diagnosis["overall_diagnosis"],
            *diagnosis["recommendations"],
            *collect_statements(build_advisory(diagnosis)),
        ]
        joined = " ".join(texts)
        for phrase in (*RETIRED, "consistent with shaft-order sources"):
            assert phrase not in joined, phrase
        advisory_actions = [
            r["action"] for r in build_advisory(diagnosis)["recommendations"]
        ]
        assert_no_fault_attribution(
            texts, [*diagnosis["recommendations"], *advisory_actions]
        )


def _flat(text: str) -> str:
    return " ".join(text.split())


class TestSameLineIsOneObservation:
    """diagnose_vibration and assess_shaft_fault_evidence read the same
    spectrum of the same signal. Their readings of one line are one
    observation: neither result counts the other's, and both descriptions
    say so in the same terms."""

    def test_both_descriptions_state_that_one_line_read_twice_is_one_observation(
        self,
    ):
        import inspect

        from predictive_maintenance_mcp.mcp_tools import diagnostics_tools

        diagnosis = _flat(inspect.getdoc(diagnostics_tools.diagnose_vibration))
        shaft = _flat(inspect.getdoc(diagnostics_tools.assess_shaft_fault_evidence))
        assert (
            "Another result that reads the same spectral line of the same signal, "
            "from this tool or another, repeats that observation and does not "
            "corroborate it." in diagnosis
        )
        assert (
            "When diagnose_vibration reports a strongest line near 1x or 2x of a "
            "signal also passed here, it reads the same spectrum of that signal: a "
            "pattern here that rests on that line repeats the observation, and the "
            "two readings of that line do not corroborate each other." in shaft
        )
        assert "A strongest line elsewhere excludes none of these readings." in (
            diagnosis
        )
        # The shaft description names the sentence diagnose_vibration emits.
        assert one_x_sentence("1.00").startswith("The strongest line")
        assert two_x_sentence("2.00").startswith("The strongest line")
        # Neither description states a general rule about every result of a
        # signal, which would contradict the findings evidence_strength counts.
        for text in (diagnosis, shaft):
            assert "not independent evidence" not in text
            assert "confirms nothing" not in text

    def test_a_shared_1x_reading_adds_nothing_to_the_diagnosis(self):
        """One 10 s velocity record at 1500 rpm, 1x dominant, Zone A: the
        shaft assessment observes its unbalance pattern on that line, and
        diagnose_vibration reports the same line and counts nothing."""
        from predictive_maintenance_mcp.diagnostics.shaft_fault_evidence import (
            assess_shaft_fault_evidence,
        )

        fs = 10000.0
        t = np.arange(int(10 * fs)) / fs
        signal = np.sin(2 * np.pi * 24.7 * t) + 0.02 * np.random.default_rng(
            7
        ).standard_normal(t.size)

        shaft = assess_shaft_fault_evidence(
            [
                {
                    "signal": signal,
                    "fs": fs,
                    "signal_id": "same",
                    "signal_unit": "mm/s",
                    "direction": "horizontal",
                }
            ],
            1500,
            machine_group=2,
            support_type="rigid",
        )
        unbalance = next(h for h in shaft["hypotheses"] if h["family"] == "unbalance")
        assert unbalance["pattern_observed"] is True

        diagnosis = diagnose_vibration(
            signal,
            fs,
            rpm=1500,
            signal_id="same",
            signal_unit="mm/s",
            machine_group=2,
            support_type="rigid",
            anomaly_model_name=NO_MODEL,
        )
        assert diagnosis["iso_severity"]["zone"] == "A"
        assert "(within 10 % of 1x)" in diagnosis["overall_diagnosis"]
        assert diagnosis["evidence_strength"] == "none"
        assert diagnosis["recommendations"] == [ROUTINE]

    def test_the_diagnosis_does_not_read_the_shaft_assessment(self):
        """No module that grades a diagnosis imports the shaft-order
        assessment, so no code path can add its result to evidence_strength."""
        import ast
        import inspect

        from predictive_maintenance_mcp.decision_support import (
            advisory,
            diagnosis_pipeline,
        )

        for module in (diagnosis_pipeline, advisory):
            tree = ast.parse(inspect.getsource(module))
            imported = {
                (node.module or "") + "." + alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            assert not any(
                "shaft_fault_evidence" in name or "shaft_orders" in name
                for name in imported
            ), (module.__name__, sorted(imported))
