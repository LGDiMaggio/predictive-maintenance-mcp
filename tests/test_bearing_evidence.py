"""Unit tests of the bearing evidence layer.

``bearing_evidence`` decides, on one envelope spectrum, whether each fault
frequency carries a line significant against the local noise floor, whether
further lines its kinematics predict corroborate it, and which outcome the
check reaches. These tests pin the statistics and the rules on spectra whose
content is known by construction:

- the ordered-statistic threshold reproduces its closed-form false-alarm
  probability, the values published for it, and a simulation of the null;
- on seeded noise the measured false-alarm rate per search window stays
  within the nominal rate plus a one-sided binomial margin;
- a window or a reference too small to test is "not testable", never a
  quiet "absent";
- a counted line is a local maximum strictly inside its window;
- without a shaft speed no sideband or shaft-order test runs, and the
  outcome says so.

The scientific acceptance scenarios on whole signals live in
``tests/test_bearing_specificity.py``.
"""

import math
from collections import Counter

import numpy as np
import pytest
from scipy.stats import binom

from _bearing_scenarios import FAULT_FREE, FAULT_FREE_SEED
from predictive_maintenance_mcp.diagnostics import bearing_evidence as E
from predictive_maintenance_mcp.diagnostics.bearing_evidence import (
    CHECK_FALSE_ALARM_PROBABILITY,
    EVIDENCE_POLICY,
    Hypothesis,
    evaluate_hypotheses,
    evidence_policy,
    line_significance,
    os_cfar_false_alarm_probability,
    os_cfar_threshold,
)
from predictive_maintenance_mcp.models import BearingEvidencePolicy
from predictive_maintenance_mcp.signal_processing.spectral import (
    envelope_spectrum_arrays,
)

#: Probability of a false failure tolerated when a rate is bounded.
RATE_TEST_LEVEL = 0.001


def _noise_spectrum(seed: int, bins: int = 4000, df: float = 0.1):
    """Positive-frequency axis and magnitudes of exponential bin power."""
    rng = np.random.default_rng(seed)
    freqs = df * np.arange(1, bins + 1)
    return freqs, np.sqrt(rng.exponential(size=bins))


def _with_line(freqs, mags, line_hz, power):
    """Put a line of the given power on the bin nearest ``line_hz``."""
    mags = mags.copy()
    mags[int(np.argmin(np.abs(freqs - line_hz)))] = math.sqrt(power)
    return mags


class TestThreshold:
    """The ordered-statistic CFAR threshold (Rohling 1983)."""

    @pytest.mark.parametrize("reference_bins", [20, 40, 200])
    def test_threshold_reproduces_the_closed_form_probability(self, reference_bins):
        """Bonferroni over the search bins: bins x per-bin probability is the
        share the threshold was solved for."""
        rank = (reference_bins + 1) // 2
        search_bins, share = 25, CHECK_FALSE_ALARM_PROBABILITY / 4
        threshold = os_cfar_threshold(reference_bins, rank, search_bins, share)
        per_bin = math.prod(
            (reference_bins - i) / (reference_bins - i + threshold) for i in range(rank)
        )

        assert search_bins * per_bin == pytest.approx(share, rel=1e-9)

    def test_threshold_falls_to_the_known_floor_limit_as_the_reference_grows(self):
        """With a known floor (N -> infinity) the median of exponential power
        is ln 2 times its mean, the per-bin probability is 2**-T, and T is
        log2(M / alpha). A finite reference needs more: an estimated floor
        is noisy, which the asymptotic threshold ignores."""
        search_bins, probability = 50, 0.05
        limit = math.log2(search_bins / probability)
        thresholds = [
            os_cfar_threshold(n, (n + 1) // 2, search_bins, probability)
            for n in (20, 40, 80, 200, 20_000)
        ]

        assert thresholds == sorted(thresholds, reverse=True)
        assert thresholds[0] > 1.4 * limit
        assert thresholds[-1] == pytest.approx(limit, rel=0.01)

    def test_closed_form_matches_a_simulation_of_the_null(self):
        """Exponential reference and test bins, N = 20, k = 10."""
        rng = np.random.default_rng(FAULT_FREE_SEED + 4000)
        trials, reference_bins, rank, threshold = 400_000, 20, 10, 9.97
        reference = rng.exponential(size=(trials, reference_bins))
        floor = np.partition(reference, rank - 1, axis=1)[:, rank - 1]
        simulated = np.mean(rng.exponential(size=trials) > threshold * floor)
        expected = os_cfar_false_alarm_probability(threshold, reference_bins, rank)

        assert simulated == pytest.approx(expected, rel=0.05)


class TestLineSignificance:
    def test_strong_line_is_significant_and_reported(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4100)
        mags = _with_line(freqs, mags, 107.4, power=400.0)

        result = line_significance(freqs, mags, 107.4, 5.4, 0.0025)

        assert result["significant"] is True
        assert result["peak_hz"] == pytest.approx(107.4)
        assert result["level_db"] > result["threshold_db"]
        assert result["search_bins"] >= 3
        assert result["reference_bins"] >= 16
        assert result["reference_rank"] == (result["reference_bins"] + 1) // 2
        assert result["false_alarm_probability"] == 0.0025

    def test_window_with_fewer_than_three_bins_is_not_testable(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4101, df=1.0)

        result = line_significance(freqs, mags, 11.93, 0.6, 0.0025)

        assert result["testable"] is False
        assert result["untestable_reason"] == "search_window_too_narrow"

    def test_too_few_reference_bins_is_not_testable(self):
        """A 24-bin spectrum leaves 13 reference bins around a 5-bin window
        (no route of the analyzer produces one: its spectra are far longer)."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4102, bins=24, df=1.0)

        result = line_significance(freqs, mags, 15.0, 2.0, 0.0025)

        assert result["testable"] is False
        assert result["untestable_reason"] == "noise_floor_not_estimable"

    def test_skirt_of_a_line_outside_the_window_is_not_counted(self):
        """A strong line two bins beyond the window edge makes the window's
        strongest bin its edge bin: no local maximum strictly inside."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4103)
        centre, half_width = 107.4, 5.4
        outside = centre + half_width + 0.2
        mags = _with_line(freqs, mags, outside, power=1e6)
        edge = int(np.flatnonzero(np.abs(freqs - centre) <= half_width)[-1])
        mags[edge] = math.sqrt(1e4)

        result = line_significance(freqs, mags, centre, half_width, 0.0025)

        assert result["local_maximum"] is False
        assert result["significant"] is False


def _disjoint_centres(df: float, highest: float = 1500.0) -> list[float]:
    """Centres of 5 % search windows whose windows and references never
    overlap, also after a common jitter of up to 3 %: each reference reaches
    the larger of 30 % of its centre and 40 bins beyond a two-bin guard."""
    centres: list[float] = []
    centre = max(10.0, 40 * df)
    while centre < highest:
        centres.append(centre)
        top = 1.05 * 1.03 * centre + 3 * df + max(0.3 * 1.03 * centre, 40 * df)
        while 0.95 * 0.97 * centre - 3 * df - max(0.3 * 1.03 * centre, 40 * df) <= top:
            centre *= 1.01
    return centres


def _records_for_power(windows: int, share: float) -> int:
    """Fewest records whose window tests fail twice the nominal share with at
    least 90 % probability at the ``RATE_TEST_LEVEL`` bound."""
    records = 100
    while True:
        tests = records * windows
        allowed = binom.ppf(1 - RATE_TEST_LEVEL, tests, share)
        if binom.sf(allowed, tests, 2 * share) >= 0.9:
            return records
        records += 100


class TestCalibration:
    """AE2: the measured per-window false-alarm rate on seeded noise.

    Each record is one check of label-neutral hypotheses at disjoint windows,
    so the share of each window is the one ``evaluate_hypotheses`` gives it:
    the per-check probability split over the windows. Records alternate
    between white noise and the coloured and resonant family. The count of
    windows with a significant line is bounded by the binomial count the
    share exceeds with probability ``RATE_TEST_LEVEL``, pooled and per family,
    and the pooled test has at least 90 % power against twice the share.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fs, seconds", [(12000.0, 10.0), (12000.0, 1.0), (25600.0, 4.0), (48000.0, 2.0)]
    )
    def test_false_alarm_rate_stays_within_the_nominal_share(self, fs, seconds):
        centres = _disjoint_centres(fs / round(fs * seconds))
        share = CHECK_FALSE_ALARM_PROBABILITY / len(centres)
        records = _records_for_power(len(centres), share)

        families = ("white", "resonant_coloured")
        false_alarms, tests = Counter(), Counter()
        for i in range(records):
            family = families[i % 2]
            rng = np.random.default_rng((FAULT_FREE_SEED, 5, int(fs), int(seconds), i))
            signal = FAULT_FREE[family](rng, fs, seconds, 29.95)
            freqs, mags = envelope_spectrum_arrays(signal, fs)
            jitter = rng.uniform(0.97, 1.03)
            hypotheses = [
                Hypothesis(f"W{j}", centre * jitter, None)
                for j, centre in enumerate(centres)
            ]
            checks = evaluate_hypotheses(
                freqs, mags, hypotheses, shaft_hz=None, cage_hz=None
            )["checks"]
            assert all(
                c["line_test"]["false_alarm_probability"] == pytest.approx(share)
                for c in checks
            )
            tests[family] += len(checks)
            false_alarms[family] += sum(c["state"] != "absent" for c in checks)

        def allowed(n: int) -> int:
            return int(binom.ppf(1 - RATE_TEST_LEVEL, n, share))

        total, n_tests = sum(false_alarms.values()), sum(tests.values())
        detail = (
            f"{fs:g} Hz, {seconds:g} s: {total} false alarms in {n_tests} window "
            f"tests ({total / n_tests:.3%}; {dict(false_alarms)} of {dict(tests)}); "
            f"a nominal {share:.3%} share allows at most {allowed(n_tests)} pooled"
        )
        assert total <= allowed(n_tests), detail
        for family in families:
            assert false_alarms[family] <= allowed(tests[family]), detail


class TestProbabilitySplit:
    """D-T1-1: one check's probability is split over the lines it can test."""

    def test_four_testable_hypotheses_each_get_a_quarter(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4300, bins=12_000)
        hypotheses = [
            Hypothesis(f"W{j}", c, None) for j, c in enumerate((40, 90, 200, 440))
        ]

        checks = evaluate_hypotheses(
            freqs, mags, hypotheses, shaft_hz=None, cage_hz=None
        )["checks"]

        assert [c["line_test"]["false_alarm_probability"] for c in checks] == [
            pytest.approx(CHECK_FALSE_ALARM_PROBABILITY / 4)
        ] * 4

    def test_an_untestable_hypothesis_takes_no_share(self):
        """At 1 Hz resolution a window at 11.93 Hz +/- 5 % holds one bin."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4301, bins=3000, df=1.0)
        hypotheses = [
            Hypothesis("BPFO", 107.36, "outer_race"),
            Hypothesis("BPFI", 162.19, "inner_race"),
            Hypothesis("BSF", 70.58, "ball"),
            Hypothesis("FTF", 11.93, "cage"),
        ]

        checks = evaluate_hypotheses(
            freqs, mags, hypotheses, shaft_hz=29.95, cage_hz=11.93
        )["checks"]

        assert checks[3]["state"] == "not_testable"
        assert [c["line_test"]["false_alarm_probability"] for c in checks[:3]] == [
            pytest.approx(CHECK_FALSE_ALARM_PROBABILITY / 3)
        ] * 3

    def test_nothing_testable_is_inconclusive_and_asks_only_for_a_longer_capture(
        self,
    ):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4302, bins=3000, df=1.0)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("FTF", 11.93, "cage"), Hypothesis("X", 9.0, None)],
            shaft_hz=29.95,
            cage_hz=11.93,
        )

        assert result["outcome"] == "inconclusive"
        assert result["outcome_reason"].startswith(
            "None of the lines could be tested in this record."
        )
        assert result["next_step"] == (
            "Record a longer capture to test the cage line (FTF) and the X line."
        )


class TestWithoutShaftSpeed:
    def test_no_sideband_or_shaft_order_test_runs_and_the_outcome_says_so(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4200, bins=8000)
        for line_hz in (162.2, 324.4):
            mags = _with_line(freqs, mags, line_hz, power=400.0)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("BPFI", 162.2, "inner_race")],
            shaft_hz=None,
            cage_hz=None,
        )
        (check,) = result["checks"]

        assert {c["kind"] for c in check["corroboration"]} == {"harmonic"}
        assert "shaft speed was not given" in result["outcome_reason"]


class TestPolicy:
    def test_policy_states_a_nominal_probability_per_check_and_its_assumptions(self):
        policy = EVIDENCE_POLICY

        assert policy["false_alarm_probability_per_check"] == 0.01
        assert policy["false_alarm_probability_per_corroborating_line"] == 0.01
        meaning = policy["false_alarm_probability_meaning"].lower()
        assert "nominal" in meaning and "one complete check" in meaning
        assumptions = " ".join(policy["assumptions"]).lower()
        for term in ("exponential", "ordered-statistic", "bonferroni"):
            assert term in assumptions
        exclusions = " ".join(policy["exclusions"]).lower()
        for term in ("deterministic", "non-stationary", "repeated"):
            assert term in exclusions
        assert "specificity" in exclusions

    def test_policy_is_versioned(self):
        assert EVIDENCE_POLICY["method"]
        assert isinstance(EVIDENCE_POLICY["version"], int)

    def test_record_carries_every_tunable_and_matches_the_model(self):
        """A key the model lacks would be dropped from the tool output, and a
        tunable missing from the record would change results unannounced."""
        tunables = {
            "false_alarm_probability_per_check": E.CHECK_FALSE_ALARM_PROBABILITY,
            "false_alarm_probability_per_corroborating_line": (
                E.CORROBORATION_FALSE_ALARM_PROBABILITY
            ),
            "min_search_bins": E.MIN_SEARCH_BINS,
            "min_reference_bins": E.MIN_REFERENCE_BINS,
            "reference_width_fraction": E.REFERENCE_WIDTH_FRACTION,
            "min_reference_bins_per_side": E.MIN_REFERENCE_BINS_PER_SIDE,
            "guard_bins": E.GUARD_BINS,
            "lowest_reference_bin": E.LOWEST_REFERENCE_BIN,
            "usable_band_fraction": E.USABLE_BAND_FRACTION,
            "narrow_window_fraction": E.NARROW_WINDOW_FRACTION,
            "min_narrow_window_bins": E.MIN_NARROW_WINDOW_BINS,
        }
        constants = [
            value
            for name, value in vars(E).items()
            if name.isupper() and isinstance(value, (int, float))
        ]
        record = evidence_policy(5.0, 3)

        assert sorted(constants) == sorted(tunables.values())
        assert {k: record[k] for k in tunables} == tunables
        assert set(record) == set(BearingEvidencePolicy.model_fields)


class TestCageAndRollingElement:
    """A rolling-element defect can be carried through the load zone at the cage
    rate, so its cage-rate content can support the cage hypothesis. A cage
    supported on its own is not singled out while the rolling-element line
    (2 x BSF) is significant."""

    HYPOTHESES = [Hypothesis("FTF", 11.93, "cage"), Hypothesis("BSF", 70.58, "ball")]

    #: A 6203 at 1797 rpm, whose 2 x BSF line (119.44 Hz) lies at the fourth
    #: shaft order (119.80 Hz).
    HYPOTHESES_6203 = [
        Hypothesis("BPFO", 91.44, "outer_race"),
        Hypothesis("BPFI", 148.16, "inner_race"),
        Hypothesis("BSF", 59.72, "ball"),
        Hypothesis("FTF", 11.43, "cage"),
    ]

    def _check(self, line_hz_list, seed, hypotheses=None, cage_hz=11.93):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + seed)
        for line_hz in line_hz_list:
            mags = _with_line(freqs, mags, line_hz, power=400.0)
        return evaluate_hypotheses(
            freqs,
            mags,
            hypotheses or self.HYPOTHESES,
            shaft_hz=29.95,
            cage_hz=cage_hz,
        )

    @staticmethod
    def _states(result):
        return {
            c["fault_type"]: (c["state"], c["state_reason"]) for c in result["checks"]
        }

    def test_a_lone_cage_with_a_significant_rolling_element_line_is_not_singled_out(
        self,
    ):
        """Cage line and its 2x stand out, and so does the 2 x BSF line
        (141.16 Hz) without corroboration of its own."""
        result = self._check([11.93, 23.86, 141.16], seed=4600)
        cage, ball = result["checks"]

        assert (cage["state"], cage["state_reason"]) == (
            "unconfirmed",
            "rolling_element_line_present",
        )
        assert (ball["state"], ball["state_reason"]) == (
            "unconfirmed",
            "no_corroboration",
        )
        assert (result["outcome"], result["candidates"]) == ("inconclusive", [])
        assert (
            "The cage line (FTF) at 11.90 Hz stands above the local noise floor, but "
            "another significant line falls within the search window of the "
            "rolling-element line (2 x BSF). A rolling-element defect can also produce "
            "modulation at the cage rate, which could explain the observed cage lines; "
            "that line alone does not establish a rolling-element defect. Under this "
            "conservative ambiguity rule, the significance-tested evidence is "
            "inconclusive and does not support a unique fault type."
        ) in result["outcome_reason"]

    def test_a_lone_cage_without_a_rolling_element_line_is_still_indicated(self):
        result = self._check([11.93, 23.86], seed=4601)
        cage, ball = result["checks"]

        assert (cage["state"], cage["state_reason"]) == ("supported", "corroborated")
        assert ball["state"] == "absent"
        assert result["outcome"] == "indicated"
        assert [c["fault_type"] for c in result["candidates"]] == ["FTF"]

    def test_a_cage_supported_next_to_another_type_stays_several_types(self):
        """Only a cage supported on its own is held back. The same spectrum
        with a corroborated outer-race line (BPFO and its 2x) as well is
        several types, with the cage still supported."""
        hypotheses = self.HYPOTHESES + [Hypothesis("BPFO", 107.36, "outer_race")]
        lone = self._check([11.93, 23.86, 141.16], 4602, hypotheses)
        both = self._check([11.93, 23.86, 141.16, 107.36, 214.72], 4602, hypotheses)

        assert lone["outcome"] == "inconclusive"
        assert self._states(lone)["FTF"] == (
            "unconfirmed",
            "rolling_element_line_present",
        )
        assert both["outcome"] == "several_types"
        assert {c["fault_type"] for c in both["candidates"]} == {"FTF", "BPFO"}
        assert self._states(both)["FTF"] == ("supported", "corroborated")

    def test_a_shared_rolling_element_line_still_holds_a_lone_cage_back(self):
        """The hold-back follows the significance of the rolling-element line,
        not its state: a further declared frequency at 140 Hz also claims the
        2 x BSF line, so that line is shared, and the cage, supported on its
        own, is still not singled out."""
        hypotheses = self.HYPOTHESES + [Hypothesis("X", 140.0, None)]
        result = self._check([11.93, 23.86, 141.16], 4603, hypotheses)
        states = self._states(result)

        assert states["BSF"] == ("shared", "line_shared")
        assert states["FTF"] == ("unconfirmed", "rolling_element_line_present")
        assert (result["outcome"], result["candidates"]) == ("inconclusive", [])
        assert (
            "window of the rolling-element line (2 x BSF). A rolling-element defect "
            "can also produce modulation at the cage rate"
        ) in result["outcome_reason"]

    def test_another_type_at_a_shaft_order_keeps_the_ordinary_sentence(self):
        """Only the rolling-element line's own position decides the wording: an
        outer-race line at the third shaft order, next to an off-order 2 x BSF
        line, leaves the held-back cage with the ordinary sentence."""
        hypotheses = self.HYPOTHESES + [Hypothesis("BPFO", 89.85, "outer_race")]
        result = self._check([11.93, 23.86, 141.16, 89.85], 4605, hypotheses)
        states = self._states(result)

        assert states["BPFO"] == ("unconfirmed", "line_at_shaft_order")
        assert states["BSF"] == ("unconfirmed", "no_corroboration")
        assert states["FTF"] == ("unconfirmed", "rolling_element_line_present")
        assert "and lies at a shaft order" not in result["outcome_reason"]
        assert (
            "window of the rolling-element line (2 x BSF). A rolling-element defect "
            "can also produce modulation at the cage rate"
        ) in result["outcome_reason"]

    def test_a_6203_shaft_harmonic_in_the_rolling_element_window_holds_the_cage_back(
        self,
    ):
        """An ordinary 4x shaft line, with no rolling-element content, lies in
        the 6203's 2 x BSF window and holds back a genuine cage indication: a
        sensitivity cost the conservative rule accepts. The otherwise
        identical spectrum without it is indicated."""
        cage_lines = [11.43, 22.86]
        held = self._check(
            cage_lines + [4 * 29.95], 4604, self.HYPOTHESES_6203, cage_hz=11.43
        )
        control = self._check(cage_lines, 4604, self.HYPOTHESES_6203, cage_hz=11.43)

        assert self._states(held)["BSF"] == ("unconfirmed", "line_at_shaft_order")
        assert self._states(held)["FTF"] == (
            "unconfirmed",
            "rolling_element_line_present",
        )
        assert (held["outcome"], held["candidates"]) == ("inconclusive", [])
        assert (
            "The cage line (FTF) at 11.40 Hz stands above the local noise floor, but "
            "another significant line falls within the search window of the "
            "rolling-element line (2 x BSF) and lies at a shaft order. That line may "
            "be a shaft harmonic and does not independently establish a "
            "rolling-element defect. A rolling-element defect cannot be excluded "
            "either: it can also produce modulation at the cage rate, which could "
            "explain the observed cage lines. Under this conservative ambiguity rule, "
            "the significance-tested evidence is inconclusive and does not support a "
            "unique fault type."
        ) in held["outcome_reason"]
        assert control["outcome"] == "indicated"
        assert [c["fault_type"] for c in control["candidates"]] == ["FTF"]


class TestLabelsOutsideTheVocabulary:
    def test_a_gear_mesh_at_a_shaft_order_is_not_discounted_as_a_shaft_line(self):
        """KTD10: a label outside the bearing vocabulary gets the line test and
        harmonic corroboration only. A gear mesh is a shaft order by
        construction, here 12 x 29.95 Hz, with its harmonic at 24 x."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4400, bins=8000)
        for line_hz in (359.4, 718.8):
            mags = _with_line(freqs, mags, line_hz, power=400.0)

        result = evaluate_hypotheses(
            freqs, mags, [Hypothesis("GMF", 359.4, None)], shaft_hz=29.95, cage_hz=None
        )
        (check,) = result["checks"]

        assert (check["state"], check["state_reason"]) == ("supported", "corroborated")
        assert not any(c["excluded_by"] for c in check["corroboration"])
        assert result["outcome"] == "indicated"

    def test_several_labels_get_a_label_neutral_step(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4401, bins=8000)
        for line_hz in (100.0, 200.0, 330.0, 660.0):
            mags = _with_line(freqs, mags, line_hz, power=400.0)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("A", 100.0, None), Hypothesis("B", 330.0, None)],
            shaft_hz=29.95,
            cage_hz=None,
        )

        assert result["outcome"] == "several_types"
        assert result["next_step"] in (
            "Compare with an earlier measurement of the same machine under comparable "
            f"conditions before acting on the {labels} lines."
            for labels in ("A and B", "B and A")
        )

    def test_a_line_whose_harmonics_lie_beyond_the_spectrum_says_so(self):
        """A 400 Hz spectrum cannot hold the harmonics of a 300 Hz line."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4402, bins=4000)
        mags = _with_line(freqs, mags, 300.0, power=400.0)

        result = evaluate_hypotheses(
            freqs, mags, [Hypothesis("X", 300.0, None)], shaft_hz=29.95, cage_hz=None
        )

        assert result["checks"][0]["state_reason"] == "no_corroboration"
        assert (
            "The X line at 300.00 Hz stands above the local noise floor, but none of "
            "its harmonics could be tested: they lie above half the sampling rate."
        ) in result["outcome_reason"]
        assert (
            "Record at a higher sampling rate, with an envelope band wide enough to "
            "include them, to test the further lines of the X line."
        ) in result["next_step"]

    def test_further_lines_above_the_spectrum_need_a_higher_sampling_rate(self):
        """A 150 Hz line in a 400 Hz spectrum with a usable band of 250 Hz:
        its 2x lies beyond the band, its 3x and 4x above the spectrum. Only
        the sampling-rate step is given, which also asks for the band."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4403, bins=4000)
        mags = _with_line(freqs, mags, 150.0, power=400.0)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("X", 150.0, None)],
            shaft_hz=29.95,
            cage_hz=None,
            envelope_band_hz=250.0 / E.USABLE_BAND_FRACTION,
        )

        assert {
            m["untestable_reason"] for m in result["checks"][0]["corroboration"]
        } == {
            "outside_usable_envelope_band",
            "above_frequency_range",
        }
        assert result["next_step"] == (
            "Repeat the measurement, or compare it with an earlier measurement of the "
            "same machine under comparable conditions. Record at a higher sampling "
            "rate, with an envelope band wide enough to include them, to test the "
            "further lines of the X line."
        )


class TestUsableBandAndFrequencyRange:
    def test_a_window_beyond_the_usable_band_is_not_testable(self):
        """Band width 150 Hz: lines and references below 135 Hz only."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4500)
        result = evaluate_hypotheses(
            freqs,
            mags,
            [
                Hypothesis("BPFO", 107.36, "outer_race"),
                Hypothesis("BPFI", 162.19, "inner_race"),
            ],
            shaft_hz=29.95,
            cage_hz=None,
            envelope_band_hz=150.0,
        )
        bpfo, bpfi = result["checks"]

        assert bpfi["state_reason"] == "outside_usable_envelope_band"
        assert bpfo["state"] != "not_testable"
        assert (
            bpfo["line_test"]["false_alarm_probability"]
            == CHECK_FALSE_ALARM_PROBABILITY
        )

    def test_a_reference_cut_short_by_the_usable_band_names_the_band(self):
        """At 0.5 Hz resolution a 10 Hz line keeps 14 reference bins below
        it; the usable band (11 Hz) removes every bin above it."""
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4501, bins=2000, df=0.5)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("X", 10.0, None)],
            shaft_hz=None,
            cage_hz=None,
            envelope_band_hz=11.0 / E.USABLE_BAND_FRACTION,
        )

        assert result["checks"][0]["state_reason"] == "outside_usable_envelope_band"

    def test_a_window_above_the_spectrum_names_the_sampling_rate(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4502)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("GMF", 700.0, None)],
            shaft_hz=29.95,
            cage_hz=None,
            envelope_band_hz=1000.0,
        )

        assert result["checks"][0]["state_reason"] == "above_frequency_range"
        assert (
            result["next_step"]
            == "Record at a higher sampling rate, with an envelope band wide enough "
            "to include it, to test the GMF line."
        )

    def test_a_line_with_no_further_line_requested_says_so(self):
        freqs, mags = _noise_spectrum(FAULT_FREE_SEED + 4503)
        mags = _with_line(freqs, mags, 107.4, power=400.0)

        result = evaluate_hypotheses(
            freqs,
            mags,
            [Hypothesis("BPFO", 107.4, "outer_race")],
            shaft_hz=29.95,
            cage_hz=None,
            num_harmonics=0,
        )

        assert result["checks"][0]["corroboration"] == []
        assert "no further line is tested for it at these settings" in (
            result["outcome_reason"]
        )
