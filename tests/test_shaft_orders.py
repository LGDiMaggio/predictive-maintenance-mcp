"""Numerical correctness of the shaft-order line measurements.

These tests check the measurement against independent references: analytic
sinusoid amplitudes and frequencies, the integration of a sinusoid, the mean
square of a sum of sinusoids, and the server's own amplitude spectrum for the
historical convention. They say nothing about diagnostic specificity, which
tests/test_shaft_fault_evidence.py covers.
"""

import math

import numpy as np
import pytest
from scipy.stats import beta

from predictive_maintenance_mcp.diagnostics import shaft_orders
from predictive_maintenance_mcp.diagnostics.shaft_orders import (
    FAMILY_FALSE_ALARM_PROBABILITY,
    ORDER_LINE_POLICY,
    interpolated_peak_hz,
    measure_shaft_orders,
    velocity_amplitude_mm_s,
)
from predictive_maintenance_mcp.signal_processing.spectral import amplitude_spectrum
from predictive_maintenance_mcp.signal_validation import InvalidInputError

FS = 10000.0


def _t(duration_s: float, fs: float = FS) -> np.ndarray:
    return np.arange(int(round(duration_s * fs))) / fs


def _tones(t, components, noise=0.0, seed=0):
    """Sum of sinusoids [(amplitude, frequency_hz, phase)] plus white noise."""
    x = np.zeros_like(t)
    for amplitude, frequency, phase in components:
        x += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if noise:
        x += noise * np.random.default_rng(seed).standard_normal(t.size)
    return x


def _line(result, order):
    return next(line for line in result["lines"] if line["order"] == order)


class TestAmplitudeAndFrequency:
    def test_on_bin_sinusoid_amplitude_estimate_matches_the_true_amplitude(self):
        # 25 Hz on a 0.1 Hz grid sits exactly on a bin.
        t = _t(10.0)
        x = _tones(t, [(2.0, 25.0, 0.3)], noise=0.01, seed=1)
        r = measure_shaft_orders(x, FS, 1500)
        one = _line(r, 1.0)
        assert one["significant"]
        assert one["amplitude_estimate"] == pytest.approx(2.0, rel=5e-3)
        assert one["frequency_hz"] == pytest.approx(25.0, abs=0.01)

    def test_mid_bin_sinusoid_estimate_has_no_scalloping_loss(self):
        # 25.05 Hz is half a bin off the 0.1 Hz grid: the peak bin loses
        # about 1.75 dB (Hamming scalloping), the estimate does not.
        t = _t(10.0)
        x = _tones(t, [(1.0, 25.05, 0.0)], noise=0.005, seed=2)
        r = measure_shaft_orders(x, FS, 25.05 * 60)
        one = _line(r, 1.0)
        assert one["amplitude_estimate"] == pytest.approx(1.0, rel=5e-3)
        assert one["amplitude"] < 0.54 * 0.85  # the peak bin shows the loss
        assert one["frequency_hz"] == pytest.approx(25.05, abs=0.1 * 0.1)

    def test_peak_bin_amplitude_is_the_servers_fft_convention(self):
        t = _t(4.0)
        x = _tones(t, [(1.0, 30.0, 0.0), (0.4, 60.0, 0.0)], noise=0.01, seed=3)
        r = measure_shaft_orders(x, FS, 1800)
        freqs, mags = amplitude_spectrum(x, FS)
        for order in (1.0, 2.0):
            line = _line(r, order)
            k = int(np.argmin(np.abs(freqs - line["peak_hz"])))
            assert line["amplitude"] == mags[k]

    def test_interpolation_locates_off_bin_frequencies_within_two_hundredths_of_a_bin(
        self,
    ):
        t = _t(2.0)  # df = 0.5 Hz
        for offset in (-0.5, -0.31, -0.13, 0.0, 0.13, 0.27, 0.41, 0.5):
            f = 40.0 + offset * 0.5
            x = _tones(t, [(1.0, f, 0.7)])
            freqs, mags = amplitude_spectrum(x, FS)
            peak = int(np.argmax(mags))
            assert interpolated_peak_hz(freqs, mags, peak) == pytest.approx(
                f, abs=0.02 * 0.5
            )

    def test_harmonics_and_half_orders_are_measured_independently(self):
        t = _t(10.0)
        x = _tones(
            t,
            [(1.0, 24.7, 0.0), (0.3, 49.4, 0.4), (0.1, 74.1, 1.0), (0.05, 12.35, 0.0)],
            noise=0.01,
            seed=4,
        )
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["source"] == "located_line"
        for order, amplitude in ((0.5, 0.05), (1.0, 1.0), (2.0, 0.3), (3.0, 0.1)):
            assert _line(r, order)["amplitude_estimate"] == pytest.approx(
                amplitude, rel=0.01
            )
        assert not _line(r, 1.5)["significant"]
        assert r["significant_orders"] == [0.5, 1.0, 2.0, 3.0]


class TestShaftFrequency:
    def test_located_line_reports_its_deviation_from_the_declared_speed(self):
        t = _t(10.0)
        x = _tones(t, [(1.0, 24.7, 0.0)], noise=0.01)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["source"] == "located_line"
        assert r["speed"]["shaft_hz"] == pytest.approx(24.7, abs=0.01)
        assert r["speed"]["deviation_from_declared_pct"] == pytest.approx(
            -1.2, abs=0.05
        )
        assert r["speed"]["uncertainty_hz"] == pytest.approx(0.1)

    def test_a_speed_outside_the_declared_uncertainty_is_not_located(self):
        t = _t(10.0)
        x = _tones(t, [(1.0, 27.0, 0.0)], noise=0.01)  # 8 % above 25 Hz
        r = measure_shaft_orders(x, FS, 1500, speed_tolerance_pct=5.0)
        assert r["speed"]["search"]["status"] == "not_significant"
        assert r["speed"]["source"] == "declared"
        assert r["speed"]["uncertainty_hz"] == pytest.approx(1.25)
        wide = measure_shaft_orders(x, FS, 1500, speed_tolerance_pct=10.0)
        assert wide["speed"]["source"] == "located_line"
        assert wide["speed"]["shaft_hz"] == pytest.approx(27.0, abs=0.01)

    def test_two_lines_in_the_1x_window_are_ambiguous(self):
        # A two-pole motor at 49.5 Hz and a mains line at 50 Hz.
        t = _t(10.0)
        x = _tones(t, [(1.0, 49.5, 0.0), (0.6, 50.0, 0.0)], noise=0.01)
        r = measure_shaft_orders(x, FS, 2970)
        assert r["speed"]["search"]["status"] == "ambiguous"
        assert r["speed"]["search"]["competing_hz"] == [pytest.approx(50.0)]
        assert r["speed"]["source"] == "declared"

    @pytest.mark.parametrize("seed", range(8))
    def test_an_off_bin_line_is_not_ambiguous_with_its_own_sidelobes(self, seed):
        # Half a bin off the grid, 50 to 55 dB above the floor: the case where
        # the highest sidelobe is closest to the peak bin (40.8 dB down).
        t = _t(10.0)
        x = _tones(t, [(1.0, 24.75, 0.0)], noise=0.002, seed=seed)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["search"]["status"] == "located"

    def test_a_slightly_modulated_line_is_not_ambiguous(self):
        t = _t(10.0)
        carrier = np.sin(2 * np.pi * 24.7 * t)
        x = (1 + 0.02 * np.sin(2 * np.pi * 0.7 * t)) * carrier
        x += 0.002 * np.random.default_rng(3).standard_normal(t.size)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["search"]["status"] == "located"

    def test_a_strong_line_is_not_ambiguous_with_its_own_sidelobes(self):
        t = _t(10.0)
        x = _tones(t, [(100.0, 24.7, 0.0)], noise=1e-4)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["search"]["status"] == "located"

    def test_declared_speed_widens_the_windows_until_they_overlap(self):
        t = _t(10.0)
        x = 0.01 * np.random.default_rng(5).standard_normal(t.size)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["source"] == "declared"
        assert _line(r, 0.5)["testable"]
        assert _line(r, 10.0)["untestable_reason"] == "order_windows_overlap"
        # each window is order x 5 % of 25 Hz, at least 3 bins
        assert _line(r, 2.0)["half_width_hz"] == pytest.approx(2.5)


class TestConfounders:
    def test_declared_line_frequency_inside_a_window_is_reported(self):
        t = _t(10.0)
        x = 0.01 * np.random.default_rng(6).standard_normal(t.size)
        r = measure_shaft_orders(x, FS, 2976, line_frequency_hz=50.0)  # 49.6 Hz
        multiples = _line(r, 1.0)["line_frequency_multiples"]
        assert multiples == [
            {
                "line_frequency_hz": 50.0,
                "multiple": 1,
                "frequency_hz": 50.0,
                "declared": True,
            }
        ]
        assert _line(r, 2.0)["line_frequency_multiples"][0]["multiple"] == 2

    def test_undeclared_line_frequency_checks_50_and_60_hz(self):
        t = _t(10.0)
        x = 0.01 * np.random.default_rng(7).standard_normal(t.size)
        r = measure_shaft_orders(x, FS, 3000)
        found = _line(r, 1.0)["line_frequency_multiples"]
        assert [(m["frequency_hz"], m["declared"]) for m in found] == [(50.0, False)]

    def test_a_located_line_separates_2x_from_twice_line_frequency(self):
        t = _t(10.0)
        x = _tones(
            t, [(1.0, 49.4, 0.0), (0.3, 98.8, 0.0), (0.5, 100.0, 0.0)], noise=0.01
        )
        r = measure_shaft_orders(x, FS, 2964, line_frequency_hz=50.0)
        two = _line(r, 2.0)
        assert r["speed"]["source"] == "located_line"
        assert two["line_frequency_multiples"] == []
        assert two["amplitude_estimate"] == pytest.approx(0.3, rel=0.02)

    def test_speed_change_between_the_halves_is_flagged(self):
        t = _t(10.0)
        f = np.where(t < 5.0, 24.7, 25.2)
        phase = 2 * np.pi * np.cumsum(f) / FS
        x = np.sin(phase) + 0.01 * np.random.default_rng(8).standard_normal(t.size)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["stationarity"]["status"] == "assessed"
        assert r["stationarity"]["speed_changed"] is True
        steady = measure_shaft_orders(_tones(t, [(1.0, 24.7, 0.0)], 0.01), FS, 1500)
        assert steady["stationarity"]["speed_changed"] is False

    def test_impulsive_contamination_keeps_the_line_and_raises_kurtosis(self):
        t = _t(10.0)
        x = _tones(t, [(1.0, 24.7, 0.0)], noise=0.01)
        # Aperiodic transients: a periodic impulse train would be a line series.
        hits = np.random.default_rng(9).choice(t.size, size=40, replace=False)
        x[hits] += 20.0
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["source"] == "located_line"
        assert r["time_domain"]["kurtosis_excess"] > 3.0


class TestRecordLimits:
    def test_a_short_record_cannot_estimate_the_noise_floor(self):
        t = _t(0.2)  # 5 revolutions at 25 Hz, 5 Hz bins
        x = _tones(t, [(1.0, 25.0, 0.0)], noise=0.01)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["speed"]["search"]["status"] == "untestable"
        assert {line["untestable_reason"] for line in r["lines"]} <= {
            "noise_floor_not_estimable",
            "order_windows_overlap",
            "search_window_too_narrow",
        }
        assert not any(line["testable"] for line in r["lines"])

    def test_orders_above_nyquist_are_untestable(self):
        fs = 200.0
        t = _t(20.0, fs)
        x = _tones(t, [(1.0, 25.0, 0.0)], noise=0.01)
        r = measure_shaft_orders(x, fs, 1500)
        assert _line(r, 1.0)["significant"]
        assert _line(r, 4.0)["untestable_reason"] == "above_frequency_range"
        assert not _line(r, 3.0)["beyond_conventional_bandwidth"]  # 75 Hz < 78.1 Hz
        assert _line(r, 3.5)["beyond_conventional_bandwidth"]  # 87.5 Hz > 200 / 2.56

    def test_long_records_are_bounded_to_their_leading_samples(self, monkeypatch):
        monkeypatch.setattr(shaft_orders, "MAX_ANALYSIS_SAMPLES", 40000)
        t = _t(10.0)
        x = _tones(t, [(1.0, 25.0, 0.0)], noise=0.01)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["analysis"]["truncated"] is True
        assert r["analysis"]["samples"] == 40000
        assert r["analysis"]["total_samples"] == t.size
        assert r["analysis"]["duration_s"] == pytest.approx(4.0)


class TestUnits:
    @pytest.mark.parametrize(
        ("unit", "amplitude", "frequency", "expected"),
        [
            ("g", 1.0, 25.0, 9.80665 / (2 * math.pi * 25.0) * 1000.0),
            ("m/s2", 2.0, 50.0, 2.0 / (2 * math.pi * 50.0) * 1000.0),
            ("m/s", 0.004, 30.0, 4.0),
            ("mm/s", 3.5, 30.0, 3.5),
            (None, 1.0, 25.0, None),
        ],
    )
    def test_velocity_of_a_sinusoid(self, unit, amplitude, frequency, expected):
        value = velocity_amplitude_mm_s(amplitude, frequency, unit)
        assert value == (None if expected is None else pytest.approx(expected))

    def test_velocity_band_share_of_two_sinusoids_is_their_power_ratio(self):
        t = _t(10.0)
        x = _tones(t, [(3.0, 25.0, 0.0), (4.0, 100.0, 0.5)], noise=0.001)
        r = measure_shaft_orders(x, FS, 1500, signal_unit="mm/s")
        band = r["velocity_band"]
        assert band["status"] == "computed"
        assert band["velocity_rms_mm_s"] == pytest.approx(
            math.sqrt((9 + 16) / 2), rel=5e-3
        )
        assert band["one_x_share"] == pytest.approx(9 / 25, rel=5e-3)

    def test_acceleration_band_is_integrated_to_velocity(self):
        t = _t(10.0)
        x = _tones(t, [(1.0, 25.0, 0.0)], noise=1e-4)
        r = measure_shaft_orders(x, FS, 1500, signal_unit="g")
        v = 9.80665 / (2 * math.pi * 25.0) * 1000.0
        assert _line(r, 1.0)["velocity_amplitude_mm_s"] == pytest.approx(v, rel=5e-3)
        assert r["velocity_band"]["velocity_rms_mm_s"] == pytest.approx(
            v / math.sqrt(2), rel=5e-3
        )

    def test_undeclared_unit_computes_no_velocity(self):
        t = _t(4.0)
        r = measure_shaft_orders(_tones(t, [(1.0, 25.0, 0.0)], 0.01), FS, 1500)
        assert r["unit"] == {"signal_unit": None, "quantity": None}
        assert _line(r, 1.0)["velocity_amplitude_mm_s"] is None
        assert r["velocity_band"]["reason"] == "unit_undeclared"

    def test_the_1x_share_is_not_overstated_at_the_band_edge(self):
        t = _t(20.0)
        f1 = 601 / 60.0
        x = _tones(t, [(1.0, f1, 0.0), (1.0, 5 * f1, 0.3)], noise=1e-4)
        r = measure_shaft_orders(x, FS, 601, signal_unit="mm/s")
        assert r["velocity_band"]["low_hz"] == 2.0
        assert r["velocity_band"]["one_x_share"] == pytest.approx(0.5, rel=5e-3)

    def test_slow_machines_use_the_2_hz_band_edge(self):
        t = _t(20.0)
        r = measure_shaft_orders(
            _tones(t, [(1.0, 8.0, 0.0)], 0.01), FS, 480, signal_unit="mm/s"
        )
        assert r["velocity_band"]["low_hz"] == 2.0


class TestRefusals:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rpm": 0},
            {"rpm": float("nan")},
            {"fs": -1.0},
            {"signal_unit": "furlongs"},
            {"speed_tolerance_pct": 0},
            {"speed_tolerance_pct": 20},
            {"max_order": 1},
            {"max_order": 21},
            {"max_order": True},
            {"max_order": 4.0},
            {"line_frequency_hz": -50},
        ],
    )
    def test_invalid_parameters_are_refused(self, kwargs):
        args = {"signal": np.ones(1000), "fs": FS, "rpm": 1500, **kwargs}
        with pytest.raises(InvalidInputError):
            measure_shaft_orders(
                args.pop("signal"), args.pop("fs"), args.pop("rpm"), **args
            )

    def test_non_finite_samples_are_refused(self):
        x = np.ones(1000)
        x[10] = np.nan
        with pytest.raises(InvalidInputError, match="non-finite"):
            measure_shaft_orders(x, FS, 1500)

    def test_samples_too_large_to_square_are_refused(self):
        with pytest.raises(InvalidInputError, match="overflowing"):
            measure_shaft_orders(np.full(1000, 1e154), FS, 1500)

    def test_too_few_samples_are_refused(self):
        with pytest.raises(InvalidInputError, match="at least 3 samples"):
            measure_shaft_orders(np.ones(2), FS, 1500)


class TestCalibration:
    """Realized family-wise false-alarm rate on seeded noise (a screen).

    600 records per family; with the nominal 1 % budget the test fails only
    when the Clopper-Pearson 95 % lower bound of the measured rate exceeds
    2 %. Its power is about 0.6 against a true rate of 3.5 % and close to 1
    against 5 %; it says nothing finer. White and pink noise leave the 1x line
    unlocated (declared-speed windows); the third family has strong 1X and 2X
    lines, so the 1x line is located and every window is narrow, and counts
    chance lines at the other orders. Larger runs (3000 records per family)
    measured 0.37 %, 0.33 % and 1.33 % (95 % interval 0.95 to 1.81 %).
    """

    @staticmethod
    def _rate(family: int, make, allowed=frozenset()) -> tuple[int, int]:
        fs, n, hits, records = 2048.0, 8192, 0, 600
        for seed in range(records):
            x = make(np.random.default_rng((family, seed)), n)
            r = measure_shaft_orders(x, fs, 1500)
            hits += bool(set(r["significant_orders"]) - allowed)
        return hits, records

    @staticmethod
    def _lower(hits: int, records: int) -> float:
        return 0.0 if hits == 0 else float(beta.ppf(0.025, hits, records - hits + 1))

    def test_white_noise(self):
        hits, records = self._rate(1, lambda rng, n: rng.standard_normal(n))
        assert (
            self._lower(hits, records) < 2 * FAMILY_FALSE_ALARM_PROBABILITY
        ), f"{hits}/{records}"

    def test_pink_noise(self):
        def pink(rng, n):
            spectrum = np.fft.rfft(rng.standard_normal(n))
            f = np.arange(spectrum.size, dtype=float)
            f[0] = 1.0
            return np.fft.irfft(spectrum / np.sqrt(f), n)

        hits, records = self._rate(2, pink)
        assert (
            self._lower(hits, records) < 2 * FAMILY_FALSE_ALARM_PROBABILITY
        ), f"{hits}/{records}"

    def test_located_shaft_lines(self):
        t = np.arange(8192) / 2048.0

        def lines(rng, n):
            x = 5 * np.sin(2 * np.pi * 24.7 * t) + 2 * np.sin(2 * np.pi * 49.4 * t + 1)
            return x + rng.standard_normal(n)

        hits, records = self._rate(3, lines, frozenset({1.0, 2.0}))
        assert (
            self._lower(hits, records) < 2 * FAMILY_FALSE_ALARM_PROBABILITY
        ), f"{hits}/{records}"


def test_policy_states_that_significance_is_not_abnormality():
    assert ORDER_LINE_POLICY["false_alarm_probability_per_call"] == 0.01
    assert any("not that it is abnormal" in e for e in ORDER_LINE_POLICY["exclusions"])


class TestBoundsAndInputs:
    def test_the_competing_list_is_capped_and_counted(self):
        t = _t(10.0)
        comps = [(1.0, 24.0 + 0.4 * k, 0.1 * k) for k in range(6)]
        r = measure_shaft_orders(_tones(t, comps, noise=0.001), FS, 1500)
        search = r["speed"]["search"]
        assert search["status"] == "ambiguous"
        assert search["competing_count"] == 5
        assert len(search["competing_hz"]) <= 5

    def test_integer_samples_are_analyzed_as_floats(self):
        t = _t(4.0)
        x = (20000 * np.sin(2 * np.pi * 25.0 * t)).astype(np.int16)
        r = measure_shaft_orders(x, FS, 1500)
        assert r["time_domain"]["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)

    @pytest.mark.parametrize(("fs", "rpm"), [(0.5, 1500), (FS, 0.5), (FS, 1e-320)])
    def test_unrepresentable_rates_are_refused(self, fs, rpm):
        with pytest.raises(InvalidInputError):
            measure_shaft_orders(np.ones(1000), fs, rpm)

    @pytest.mark.parametrize("n", [2, 3, 1000, 1001, 40000])
    def test_hamming_power_gain_closed_form(self, n):
        from predictive_maintenance_mcp.diagnostics.shaft_orders import (
            hamming_power_gain,
        )

        assert hamming_power_gain(n) == pytest.approx(
            np.sum(np.hamming(n) ** 2) / n, rel=1e-12
        )

    def test_the_returned_policy_is_a_copy(self):
        t = _t(4.0)
        r = measure_shaft_orders(_tones(t, [(1.0, 25.0, 0.0)], 0.01), FS, 1500)
        r["policy"]["exclusions"].append("edited")
        assert "edited" not in ORDER_LINE_POLICY["exclusions"]

    def test_an_all_zero_record_has_no_undefined_statistics(self):
        r = measure_shaft_orders(np.zeros(20000), FS, 1500)
        assert r["time_domain"] == {
            "rms": 0.0,
            "kurtosis_excess": None,
            "crest_factor": None,
        }
