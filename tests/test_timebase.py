"""Tests for the time-base assessment (Signal Import Engine plan, U4, KTD9).

The first class is the plan's Appendix "time-base outcome table", one test
per row, written before the implementation. Synthetic columns run at
25.6 kHz with 6 decimals and 20,000 rows unless a test says otherwise. A
column is printed from exact rational stamps with correct rounding (half to
even), the way ``printf`` prints, so every expected tick count is exact.
"""

import random
import time
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from predictive_maintenance_mcp.file_text import TOKEN_CHARS
from predictive_maintenance_mcp.signal_acquisition import import_config as ic
from predictive_maintenance_mcp.signal_acquisition import text_decoder as td
from predictive_maintenance_mcp.signal_import import timebase as tb

FS = 25600
N = 20000
FIRST_LINE = 5  # four preamble and header lines above the data


# ---------------------------------------------------------------------------
# Synthetic columns
# ---------------------------------------------------------------------------


def _round_half_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2):
        quotient += 1
    return quotient


def _fixed(ticks: int, decimals: int) -> str:
    sign = "-" if ticks < 0 else ""
    digits = str(abs(ticks)).rjust(decimals + 1, "0")
    if not decimals:
        return sign + digits
    return f"{sign}{digits[:-decimals]}.{digits[-decimals:]}"


def printed(numerators, denominator: int, decimals: int = 6) -> list[str]:
    """Stamps ``numerator / denominator`` seconds, printed with *decimals*."""
    scale = 10**decimals
    return [
        _fixed(_round_half_even(int(n) * scale, denominator), decimals)
        for n in numerators
    ]


def uniform(decimals: int = 6, *, rows: int = N, fs: int = FS) -> list[str]:
    return printed(range(rows), fs, decimals)


def assess(tokens, *, unit="s", first_line=FIRST_LINE, decimal_mark=".", **kwargs):
    mantissa, exponent = tb.parse_time_tokens(
        tokens, decimal_mark=decimal_mark, first_line=first_line
    )
    return tb.assess_time_base(
        mantissa, exponent, time_unit=unit, first_line=first_line, **kwargs
    )


def codes(result: tb.TimeBaseAssessment) -> set[str]:
    return {finding.code for finding in result.findings}


def finding(result: tb.TimeBaseAssessment, code: str) -> tb.TimeFinding:
    (match,) = [f for f in result.findings if f.code == code]
    return match


def comparison(result, first: str, second: str) -> tb.RateComparison:
    (match,) = [
        c for c in result.rate_comparisons if (c.first, c.second) == (first, second)
    ]
    return match


def assert_accepted(result: tb.TimeBaseAssessment) -> None:
    assert result.outcome == "accepted", result.findings
    assert result.findings == ()


# ---------------------------------------------------------------------------
# The Appendix outcome table
# ---------------------------------------------------------------------------


class TestOutcomeTable:
    """One test per row of the plan's time-base outcome table."""

    @pytest.mark.parametrize("decimals", [9, 7, 6, 5])
    def test_uniform_at_9_7_6_or_5_decimals_is_accepted_with_full_evidence(
        self, decimals
    ):
        result = assess(uniform(decimals))
        assert_accepted(result)
        assert result.evidence == "full"
        assert result.regime == "fixed_decimals"
        assert result.resolution == _fixed(1, decimals)
        assert result.proposed_rate_hz is not None
        rate = Decimal(result.fitted_rate_hz)
        assert abs(rate - FS) <= Decimal(result.rate_uncertainty_hz)
        tb_interval = Decimal(10) ** decimals / FS
        assert result.modal_steps == (int(tb_interval), int(tb_interval) + 1)

    def test_uniform_at_6_decimals_steps_in_39_and_40_ticks(self):
        result = assess(uniform(6))
        assert result.modal_steps == (39, 40)
        assert result.step_range == (39, 40)
        assert result.reference_interval_ticks == pytest.approx(39.0625)

    def test_uniform_at_4_decimals_is_duration_only_and_needs_a_rate_basis(self):
        result = assess(uniform(4))
        assert result.evidence == "duration_only"
        assert result.outcome == "accepted"
        assert result.findings == ()
        assert result.fitted_rate_hz is None
        assert result.proposed_rate_hz is None
        assert result.needs_rate_basis

    def test_51200_over_3_hz_at_6_decimals_steps_in_58_and_59(self):
        result = assess(printed(range(0, 3 * N, 3), 51200))
        assert_accepted(result)
        assert result.modal_steps == (58, 59)
        assert Decimal(result.fitted_rate_hz) == pytest.approx(
            Decimal(51200) / 3, abs=Decimal(result.rate_uncertainty_hz)
        )

    def test_12_khz_at_5_decimals_steps_in_8_and_9(self):
        result = assess(printed(range(N), 12000, 5))
        assert_accepted(result)
        assert result.evidence == "full"
        assert result.modal_steps == (8, 9)

    @pytest.mark.parametrize("missing, intervals", [(1, 2), (7, 8)])
    def test_missing_samples_are_a_time_gap_at_the_line(self, missing, intervals):
        kept = [n for n in range(N) if not 10000 <= n < 10000 + missing]
        result = assess(printed(kept, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_gap"}
        gap = finding(result, "time_gap")
        assert gap.count == 1
        # The stamp after the gap is row 10000 of the column.
        assert gap.lines == (FIRST_LINE + 10000,)
        assert gap.measures[0] == pytest.approx(intervals, abs=0.03)

    def test_duplicate_row_is_a_time_reversal(self):
        numerators = list(range(N))
        numerators.insert(5001, 5000)
        result = assess(printed(numerators, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_reversal"}
        assert finding(result, "time_reversal").lines == (FIRST_LINE + 5001,)

    def test_adjacent_rows_swapped_is_a_time_reversal(self):
        numerators = list(range(N))
        numerators[5000], numerators[5001] = numerators[5001], numerators[5000]
        result = assess(printed(numerators, FS))
        assert result.outcome == "invalid"
        reversal = finding(result, "time_reversal")
        assert reversal.lines == (FIRST_LINE + 5001,)
        # Around the swap the steps also measure two intervals.
        assert finding(result, "time_gap").count == 2

    def test_restart_to_zero_is_a_time_reversal(self):
        result = assess(printed(list(range(N // 2)) * 2, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_reversal"}
        assert finding(result, "time_reversal").lines == (FIRST_LINE + N // 2,)

    def test_rate_halves_mid_file_is_a_time_gap_at_the_change(self):
        change = N // 2
        stamps = [n if n <= change else change + 2 * (n - change) for n in range(N)]
        result = assess(printed(stamps, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_gap"}
        gap = finding(result, "time_gap")
        assert gap.lines[0] == FIRST_LINE + change + 1
        assert gap.count == N - 1 - change

    def test_rate_halves_for_the_last_20_percent_is_a_time_gap_at_the_change(self):
        change = 4 * N // 5
        stamps = [n if n <= change else change + 2 * (n - change) for n in range(N)]
        result = assess(printed(stamps, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_gap"}
        assert finding(result, "time_gap").lines[0] == FIRST_LINE + change + 1

    def test_rate_halves_at_20_percent_is_invalid_against_the_modal_interval(self):
        """The halved rate covers 80% of the rows, so it sets the reference
        interval; the earlier steps measure half of it, and a step below half
        an interval is a duplicate or reversal by KTD9. The column is invalid,
        as the table says, but under ``time_reversal`` from its first step
        (a recorded deviation from the table's ``time_gap``)."""
        change = N // 5
        stamps = [n if n <= change else change + 2 * (n - change) for n in range(N)]
        result = assess(printed(stamps, FS))
        assert result.outcome == "invalid"
        assert result.reference_interval_ticks == pytest.approx(78.125, abs=0.01)
        assert codes(result) == {"time_reversal"}
        assert finding(result, "time_reversal").lines[0] == FIRST_LINE + 1

    def test_every_third_sample_missing_is_a_time_gap(self):
        kept = [n for n in range(3 * N // 2) if n % 3 != 2]
        result = assess(printed(kept, FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_gap"}
        assert 39 <= result.reference_interval_ticks <= 40

    def test_rate_change_by_40_percent_mid_file_is_a_timing_defect(self):
        change = N // 2
        # Rate times 1.4 from the change: the interval becomes 5/7 of itself.
        stamps = [
            7 * n if n <= change else 7 * change + 5 * (n - change) for n in range(N)
        ]
        result = assess(printed(stamps, 7 * FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_residual"}
        residual = finding(result, "time_residual")
        assert residual.count > 0
        assert all(measure > 0.5 for measure in residual.measures)

    def test_200_ppm_quadratic_drift_is_a_timing_defect(self):
        # t_n = (n + 200e-6 * n**2 / N) / fs: the clock has drifted by 200 ppm
        # of the duration at the last row.
        stamps = [n * (5000 * N + n) for n in range(N)]
        result = assess(printed(stamps, 5000 * N * FS))
        assert result.outcome == "invalid"
        assert codes(result) == {"time_residual"}
        assert result.step_range == (39, 40)
        assert result.max_residual_ticks == pytest.approx(26, abs=1)
        assert result.reference_interval_ticks / 2 == pytest.approx(19.53, abs=0.01)

    @pytest.mark.parametrize("sigma", [0.2, 0.5, 2.0, 0.05 * 39.0625])
    def test_jitter_needs_confirmation_and_is_never_a_gap(self, sigma):
        rng = np.random.default_rng(20261103)
        exact = np.arange(N) * 39.0625 + rng.normal(0.0, sigma, N)
        ticks = np.rint(exact).astype(np.int64)
        result = assess([_fixed(int(k), 6) for k in ticks])
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"time_off_pattern"}
        off = finding(result, "time_off_pattern")
        assert off.count > 0
        assert result.max_residual_ticks is not None
        assert result.rms_residual_ticks is not None
        assert "residual" in off.statement

    def test_gapped_column_is_a_time_gap_when_the_rate_comes_from_the_preamble(self):
        kept = [n for n in range(N) if n != 10000]
        result = assess(
            printed(kept, FS),
            declared=tb.DeclaredTimeBase(kind="rate", token="25600"),
        )
        assert result.outcome == "invalid"
        assert "time_gap" in codes(result)

    @pytest.mark.parametrize(
        "token",
        ["2026-11-03 10:00:00.000039", "2026-11-03T10:00:00", "10:00:00.000039"],
    )
    def test_date_time_text_column_blocks_until_the_override(self, token):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0.000000", token], first_line=FIRST_LINE)
        refused = caught.value
        assert refused.code == "time_grammar"
        assert refused.datetime_text
        assert refused.line == FIRST_LINE + 1
        assert "unassessed" in str(refused)

    def test_float32_accumulated_time_past_330_s_is_duration_only(self):
        # A float32 clock accumulated past 330 s: each step rounds to one
        # float32 unit (2**-15 s), so the printed steps look uniform at
        # 32768 Hz although the samples run at 25.6 kHz.
        start = np.float32(330.0)
        step = np.float32(1 / FS)
        values = np.cumsum(
            np.concatenate([[start], np.full(N - 1, step, dtype=np.float32)]),
            dtype=np.float32,
        )
        tokens = ["%.6f" % float(v) for v in values]
        steps = np.diff([int(t.replace(".", "")) for t in tokens])
        assert set(steps.tolist()) <= {30, 31}
        result = assess(tokens)
        assert result.evidence == "duration_only"
        assert result.float32_values
        assert Decimal(result.effective_resolution) > Decimal(result.resolution)
        assert result.outcome == "accepted"
        assert result.fitted_rate_hz is None
        assert result.needs_rate_basis

    def test_uniform_column_with_a_stated_rate_50_ppm_away_disagrees(self):
        stated = str(Decimal(FS) * Decimal("1.00005"))
        result = assess(uniform(6), stated_rate_hz=stated)
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"rate_sources_disagree"}
        disagreement = comparison(result, "time_column", "stated")
        assert not disagreement.consistent
        assert disagreement.difference_ppm == pytest.approx(50, abs=0.1)
        assert result.proposed_rate_hz is None

    def test_uniform_column_with_a_stated_rate_2_percent_away_disagrees(self):
        result = assess(uniform(6), stated_rate_hz=str(FS * 102 // 100))
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"rate_sources_disagree"}
        assert comparison(result, "time_column", "stated").difference_ppm == (
            pytest.approx(20000, abs=1)
        )

    def test_repr_of_n_over_fs_is_accepted_under_the_shortest_repr_regime(self):
        result = assess([repr(n / 12000) for n in range(N)])
        assert_accepted(result)
        assert result.regime == "shortest_repr"
        assert result.evidence == "full"
        assert Decimal(result.proposed_rate_hz) == pytest.approx(
            Decimal(12000), abs=Decimal(result.rate_uncertainty_hz)
        )

    def test_repr_of_a_cumulative_sum_is_accepted_under_the_shortest_repr_regime(
        self,
    ):
        tokens, t = [], 0.0
        for _ in range(N):
            tokens.append(repr(t))
            t += 1 / FS
        result = assess(tokens)
        assert_accepted(result)
        assert result.regime == "shortest_repr"
        assert result.proposed_rate_hz is not None

    def test_repr_of_n_over_25600_is_exact_on_a_fixed_grid(self):
        """n/25600 terminates within 10 decimals, so its repr is exact and the
        column reads as fixed decimals at 1e-10 s."""
        result = assess([repr(n / FS) for n in range(N)])
        assert_accepted(result)
        assert result.regime == "fixed_decimals"
        assert result.resolution == "0.0000000001"
        assert result.proposed_rate_hz == "25600"

    @pytest.mark.parametrize("style", ["%.6e", "%g", "%.9g"])
    def test_significant_digit_stamps_need_a_stated_basis(self, style):
        tokens = [style % (n / FS) for n in range(N)]
        result = assess(tokens)
        assert result.regime == "significant_digits"
        assert result.outcome == "accepted"
        assert result.proposed_rate_hz is None
        assert result.needs_rate_basis
        # Every stamp is checked against a stated rate: the true one agrees,
        # one 2% away does not.
        agreeing = assess(tokens, stated_rate_hz="25600")
        assert agreeing.outcome == "accepted"
        assert comparison(agreeing, "time_column", "stated").consistent
        off = assess(tokens, stated_rate_hz="26112")
        assert off.outcome == "needs_confirmation"
        assert not comparison(off, "time_column", "stated").consistent

    def test_preamble_delta_x_and_stated_25600_are_consistent(self):
        declared = tb.DeclaredTimeBase(kind="interval", token="0.000039")
        result = assess(uniform(6), stated_rate_hz="25600", declared=declared)
        assert_accepted(result)
        assert comparison(result, "file_declared", "stated").consistent
        assert comparison(result, "time_column", "file_declared").consistent
        (source,) = [s for s in result.rate_sources if s.source == "file_declared"]
        assert Decimal(source.low_hz) < FS < Decimal(source.high_hz)

    def test_preamble_delta_x_and_stated_25000_disagree(self):
        declared = tb.DeclaredTimeBase(kind="interval", token="0.000039")
        result = assess(uniform(6), stated_rate_hz="25000", declared=declared)
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"rate_sources_disagree"}
        assert not comparison(result, "file_declared", "stated").consistent

    @pytest.mark.parametrize(
        "token", ["NaN", "nan", "inf", "-Infinity", "", "   ", "0,000039"]
    )
    def test_non_finite_empty_and_decimal_comma_tokens_are_time_grammar(self, token):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0.000000", token, "0.000078"], first_line=FIRST_LINE)
        assert caught.value.code == "time_grammar"
        assert caught.value.line == FIRST_LINE + 1
        assert not caught.value.datetime_text

    def test_empty_column_is_refused(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            assess([])
        assert caught.value.code == "region_too_short"

    def test_single_row_is_refused(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            assess(["0.000000"])
        assert caught.value.code == "region_too_short"


# ---------------------------------------------------------------------------
# The plan's further scenarios
# ---------------------------------------------------------------------------


class TestGrammarCaps:
    @pytest.mark.parametrize("token", ["1e1000000", "1" * 10_000, "0." + "0" * 9_999])
    def test_extreme_tokens_are_refused_within_50_ms(self, token):
        start = time.perf_counter()
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0", token])
        assert time.perf_counter() - start < 0.05
        assert caught.value.code == "time_grammar"
        assert caught.value.token.truncated == (len(token) > TOKEN_CHARS)

    def test_a_dot_in_a_decimal_comma_column_is_refused(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0,000000", "0.000039"], decimal_mark=",")
        assert caught.value.code == "time_grammar"
        assert caught.value.line == 2

    def test_decimal_comma_column_parses_like_its_dot_twin(self):
        dot = uniform(6, rows=50)
        comma = [t.replace(".", ",") for t in dot]
        a = tb.parse_time_tokens(dot)
        b = tb.parse_time_tokens(comma, decimal_mark=",")
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_spaces_at_either_end_are_allowed(self):
        mantissa, exponent = tb.parse_time_tokens(["  0.5", "1.0  "])
        assert mantissa.tolist() == [5, 10] and exponent.tolist() == [-1, -1]

    def test_non_ascii_token_is_refused_and_escaped(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0.5", "1" + chr(0x2212) + "0"])
        assert caught.value.code == "time_grammar"
        assert chr(0x2212) not in str(caught.value)

    def test_parse_agrees_with_decimal_on_random_stamps(self):
        rng = np.random.default_rng(7)
        tokens = []
        for _ in range(2000):
            digits = "".join(rng.choice(list("0123456789"), rng.integers(1, 15)))
            point = rng.integers(0, len(digits) + 1)
            token = digits[:point] + "." + digits[point:]
            if rng.random() < 0.3:
                token += "e" + str(int(rng.integers(-30, 30)))
            if rng.random() < 0.2:
                token = "-" + token
            tokens.append(token)
        mantissa, exponent = tb.parse_time_tokens(tokens)
        for token, m, e in zip(tokens, mantissa.tolist(), exponent.tolist()):
            assert Decimal(m).scaleb(e) == Decimal(token), token

    def test_tokens_must_be_text(self):
        with pytest.raises(TypeError):
            tb.parse_time_tokens([b"0.5"])


class TestTimeUnit:
    def test_a_ms_time_unit_scales_the_derived_rate_by_1000_exactly(self):
        tokens = uniform(6)
        seconds = assess(tokens, unit="s")
        milliseconds = assess(tokens, unit="ms")
        assert (
            Decimal(milliseconds.fitted_rate_hz)
            == Decimal(seconds.fitted_rate_hz) * 1000
        )
        assert Decimal(milliseconds.rate_uncertainty_hz) == (
            Decimal(seconds.rate_uncertainty_hz) * 1000
        )
        assert milliseconds.resolution == seconds.resolution
        assert milliseconds.time_unit == "ms"

    @pytest.mark.parametrize("unit, factor", [("us", 10**6), ("ns", 10**9)])
    def test_finer_units_scale_by_powers_of_ten(self, unit, factor):
        tokens = uniform(6, rows=2000)
        base = Decimal(assess(tokens).fitted_rate_hz)
        assert Decimal(assess(tokens, unit=unit).fitted_rate_hz) == base * factor

    def test_unknown_time_unit_is_refused(self):
        with pytest.raises(ValueError, match="time unit"):
            assess(uniform(6, rows=10), unit="min")


# ---------------------------------------------------------------------------
# KTD9 rules one by one
# ---------------------------------------------------------------------------


class TestEvidenceLevels:
    def test_partial_evidence_when_the_interval_is_one_to_two_ticks(self):
        # 64 kHz at 5 decimals: 1.5625 ticks per sample interval.
        result = assess(uniform(5, fs=64000))
        assert result.evidence == "partial"
        assert result.outcome == "accepted"
        assert result.modal_steps == (1, 2)
        assert result.fitted_rate_hz is not None
        assert result.proposed_rate_hz is None
        assert result.needs_rate_basis

    def test_partial_still_refuses_duplicates(self):
        numerators = list(range(N))
        numerators.insert(100, 99)
        result = assess(printed(numerators, 64000, 5))
        assert result.evidence == "partial"
        assert "time_reversal" in codes(result)

    def test_partial_detects_a_long_gap(self):
        kept = [n for n in range(N) if not 100 <= n < 110]
        result = assess(printed(kept, 64000, 5))
        assert "time_gap" in codes(result)

    def test_partial_does_not_refuse_a_uniform_clock_near_one_tick(self):
        # 1.0526 ticks per interval: steps of 1 and occasionally 2.
        result = assess(uniform(5, fs=95000))
        assert result.evidence == "partial"
        assert result.outcome == "accepted", result.findings

    def test_duration_only_still_refuses_a_backward_step(self):
        stamps = list(range(N))
        stamps[1000] = 0
        result = assess(printed(stamps, FS, 4))
        assert result.evidence == "duration_only"
        assert result.outcome == "invalid"
        assert finding(result, "time_reversal").lines == (FIRST_LINE + 1000,)

    def test_equal_stamps_are_no_evidence_but_contradict_a_stated_rate(self):
        """Equal first and last stamps are legitimate when the column is short
        against its resolution, so they are not refused; only the total
        duration is checked against a rate."""
        result = assess(["1.0000"] * 100)
        assert result.outcome == "accepted" and result.evidence == "duration_only"
        assert result.duration == "0.0000"
        short = assess(["1.00"] * 30, stated_rate_hz="25600")
        assert short.outcome == "accepted", short.rate_comparisons
        long = assess(["1.0000"] * 100, stated_rate_hz="25600")
        assert long.outcome == "needs_confirmation"
        (disagreement,) = long.rate_comparisons
        assert disagreement.difference_ppm is None  # no duration to compare

    def test_duration_only_checks_the_total_duration_against_a_stated_rate(self):
        tokens = uniform(4)
        assert assess(tokens, stated_rate_hz="25600").outcome == "accepted"
        # 50 ppm moves the duration by 0.4 of a tick: not resolvable.
        assert assess(tokens, stated_rate_hz="25601.28").outcome == "accepted"
        far = assess(tokens, stated_rate_hz="26112")
        assert far.outcome == "needs_confirmation"
        assert not comparison(far, "time_column", "stated").consistent

    def test_float32_n_over_fs_past_330_s_is_duration_only_and_consistent(self):
        start = 330 * FS + 7
        values = np.arange(start, start + N, dtype=np.float32) / np.float32(FS)
        tokens = ["%.6f" % float(v) for v in values]
        result = assess(tokens, stated_rate_hz="25600")
        assert result.float32_values
        assert result.evidence == "duration_only"
        assert result.outcome == "accepted", result.rate_comparisons

    def test_float32_accumulated_clock_disagrees_with_the_stated_rate(self):
        values = np.cumsum(
            np.concatenate(
                [[np.float32(330.0)], np.full(N - 1, np.float32(1 / FS), np.float32)]
            ),
            dtype=np.float32,
        )
        result = assess(["%.6f" % float(v) for v in values], stated_rate_hz="25600")
        assert result.evidence == "duration_only"
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"rate_sources_disagree"}

    def test_whole_seconds_are_not_mistaken_for_float32_values(self):
        tokens = [_fixed((330 + n) * 10**6, 6) for n in range(200)]
        result = assess(tokens)
        assert not result.float32_values
        assert result.evidence == "full"
        assert result.proposed_rate_hz == "1"

    def test_decimal_clock_past_330_s_is_not_mistaken_for_float32(self):
        result = assess(printed(range(330 * FS, 330 * FS + N), FS))
        assert not result.float32_values
        assert_accepted(result)


class TestStepRules:
    def test_thresholds_are_exact_at_one_and_a_half_intervals(self):
        # Interval of exactly 40 ticks (25 kHz at 6 decimals): a step of 59
        # stays, a step of 60 is a gap; 20 stays, 19 is a reversal.
        base = [40 * n for n in range(1000)]
        for step, code in [
            (59, None),
            (60, "time_gap"),
            (21, None),
            (19, "time_reversal"),
        ]:
            ticks = base[:500] + [base[499] + step + 40 * i for i in range(500)]
            result = assess([_fixed(k, 6) for k in ticks])
            found = {f.code for f in result.findings} - {
                "time_off_pattern",
                "time_residual",
            }
            assert found == ({code} if code else set()), (step, result.findings)

    def test_at_most_ten_lines_are_listed_but_every_offence_is_counted(self):
        kept = [n for n in range(N) if n % 100 != 50]
        result = assess(printed(kept, FS))
        gap = finding(result, "time_gap")
        assert gap.count == N // 100
        assert len(gap.lines) == tb.MAX_LISTED_LINES == 10
        assert list(gap.lines) == sorted(gap.lines)
        assert len(gap.measures) == len(gap.lines)

    def test_the_modal_pair_is_the_reference_not_a_fitted_slope(self):
        # Every third sample missing fits a slope near 1.5 intervals, at which
        # every step rounds to one interval; the modal pair still finds gaps.
        kept = [n for n in range(3000) if n % 3 != 2]
        result = assess(printed(kept, FS))
        # 2000 rows: 1000 one-interval steps, 999 two-interval steps.
        assert finding(result, "time_gap").count == len(kept) // 2 - 1

    def test_the_residual_test_waits_for_the_step_tests(self):
        numerators = list(range(N))
        del numerators[100]
        result = assess(printed(numerators, FS))
        assert codes(result) == {"time_gap"}
        assert result.max_residual_ticks is None


class TestRateReport:
    @pytest.mark.parametrize("fs", [12000, 25600, 1])
    def test_fitted_rate_has_twelve_significant_digits(self, fs):
        result = assess(printed(range(2000), fs, 9))
        assert len(Decimal(result.fitted_rate_hz).as_tuple().digits) == 12
        assert abs(Decimal(result.fitted_rate_hz) - fs) <= Decimal(
            result.rate_uncertainty_hz
        )
        # The proposal is the same value in the configuration's canonical form.
        assert result.proposed_rate_hz == ic.normalize_decimal(
            result.fitted_rate_hz, field="rate"
        )

    def test_uncertainty_shrinks_with_the_row_count(self):
        short = assess(uniform(6, rows=1000))
        long = assess(uniform(6, rows=N))
        assert Decimal(long.rate_uncertainty_hz) < Decimal(short.rate_uncertainty_hz)

    def test_the_derived_rate_passes_validate_sampling_rate(self, monkeypatch):
        seen = []
        original = tb.validate_sampling_rate

        def spy(value, **kwargs):
            seen.append(value)
            return original(value, **kwargs)

        monkeypatch.setattr(tb, "validate_sampling_rate", spy)
        result = assess(uniform(6, rows=500))
        assert seen and seen[0] == float(result.fitted_rate_hz)

    def test_duration_is_reported_in_the_time_unit(self):
        result = assess(uniform(6, rows=1001))
        assert result.duration == "0.039062"
        assert result.row_count == 1001
        assert result.last_line == FIRST_LINE + 1000

    def test_rate_sources_are_listed(self):
        result = assess(
            uniform(6),
            stated_rate_hz="25600",
            declared=tb.DeclaredTimeBase(kind="rate", token="25600"),
        )
        assert [s.source for s in result.rate_sources] == [
            "time_column",
            "file_declared",
            "stated",
        ]
        declared = result.rate_sources[1]
        assert (declared.low_hz, declared.high_hz) == ("25599.5", "25600.5")
        assert (
            result.rate_sources[2].low_hz == result.rate_sources[2].high_hz == "25600"
        )

    def test_declared_interval_in_ms(self):
        declared = tb.DeclaredTimeBase(kind="interval", token="0.039", time_unit="ms")
        result = assess(uniform(6), declared=declared)
        assert comparison(result, "time_column", "file_declared").consistent

    @pytest.mark.parametrize("token", ["NaN", "0", "-1", "1e1000000", "abc", ""])
    def test_bad_declared_values_are_refused(self, token):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            assess(uniform(6, rows=10), declared=tb.DeclaredTimeBase("rate", token))
        assert caught.value.code == "time_grammar"

    @pytest.mark.parametrize("value", ["0", "-25600", "NaN", 25600])
    def test_bad_stated_rates_are_refused(self, value):
        with pytest.raises(ValueError):
            assess(uniform(6, rows=10), stated_rate_hz=value)

    def test_off_pattern_columns_skip_the_column_comparisons(self):
        rng = np.random.default_rng(3)
        ticks = np.rint(np.arange(N) * 39.0625 + rng.normal(0, 2, N)).astype(np.int64)
        result = assess([_fixed(int(k), 6) for k in ticks], stated_rate_hz="25600")
        assert codes(result) == {"time_off_pattern"}
        assert all(
            "time_column" not in (c.first, c.second) for c in result.rate_comparisons
        )


class TestInputs:
    def test_mismatched_arrays_are_refused(self):
        with pytest.raises(ValueError):
            tb.assess_time_base(
                np.zeros(3, np.int64),
                np.zeros(2, np.int16),
                time_unit="s",
                first_line=1,
            )

    def test_mixed_exponents_far_apart_align_without_overflow(self):
        mantissa = np.array([0, 999_999_999_999_999_999, 1], dtype=np.int64)
        exponent = np.array([0, 5, -30], dtype=np.int16)
        result = tb.assess_time_base(mantissa, exponent, time_unit="s", first_line=1)
        assert result.outcome == "invalid"
        assert "time_reversal" in codes(result)

    def test_the_grid_alignment_refuses_instead_of_overflowing(self):
        mantissa = np.array([1, 999_999_999_999_999_999], dtype=np.int64)
        exponent = np.array([0, 3], dtype=np.int16)
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb._on_grid(mantissa, exponent, 0, first_line=7)
        assert caught.value.code == "time_grammar"
        assert caught.value.line == 8

    def test_the_assessment_is_deterministic(self):
        tokens = ["%.9g" % (n / FS) for n in range(2000)]
        assert assess(tokens, stated_rate_hz="25600") == assess(
            tokens, stated_rate_hz="25600"
        )

    def test_finding_codes_are_issue_codes(self):
        assert set(tb.FINDING_CODES) <= set(ic.ISSUE_CODES)
        assert {"time_grammar", "region_too_short"} <= set(ic.ISSUE_CODES)

    def test_assess_time_column_reads_a_decoder_result(self, tmp_path):
        from predictive_maintenance_mcp.path_safety import open_import_file

        lines = ["time;x", *(f"{t};0.5" for t in uniform(6, rows=3000))]
        (tmp_path / "export.csv").write_bytes(("\n".join(lines) + "\n").encode())
        interpretation = {
            "schema": ic.INTERPRETATION_SCHEMA,
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ";",
                "decimal": ".",
                "header_line": 1,
            },
            "data_lines": {"first": 2, "last": None},
            "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
            "time_columns": [{"column": 0, "time_unit": "s"}],
            "channels": [
                {
                    "column": 1,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "scaling": None,
                }
            ],
        }
        with open_import_file(
            "export.csv", [tmp_path], base=tmp_path, max_size=1 << 40
        ) as file:
            decoded = td.decode_text(file, interpretation, mode="digest")
        (column,) = decoded.time_columns
        via_decoder = tb.assess_time_column(
            column, first_line=decoded.first_line, stated_rate_hz="25600"
        )
        via_tokens = assess(uniform(6, rows=3000), first_line=2, stated_rate_hz="25600")
        assert via_decoder == via_tokens
        assert_accepted(via_decoder)


class TestRandomizedColumns:
    """Property-style checks on seeded random columns."""

    def test_uniform_columns_are_accepted_and_one_gap_is_found_at_its_line(self):
        rng = np.random.default_rng(1103)
        for _ in range(40):
            fs = int(rng.integers(1000, 100_000))
            decimals = int(rng.integers(5, 10))
            rows = int(rng.integers(50, 3000))
            interval = Fraction(10**decimals, fs)
            if interval < 2:
                continue
            tokens = uniform(decimals, rows=rows, fs=fs)
            result = assess(tokens)
            assert result.outcome == "accepted", (fs, decimals, rows, result.findings)
            assert result.evidence == "full"
            assert abs(Decimal(result.fitted_rate_hz) - fs) <= Decimal(
                result.rate_uncertainty_hz
            )
            hole = int(rng.integers(1, rows - 1))
            gapped = assess(
                printed([n for n in range(rows) if n != hole], fs, decimals)
            )
            assert "time_gap" in codes(gapped)
            assert finding(gapped, "time_gap").lines == (FIRST_LINE + hole,)


# ---------------------------------------------------------------------------
# Cases the independent review found (each failed before its fix)
# ---------------------------------------------------------------------------


def float32_clock(start: float, rows: int = N, fs: int = FS) -> np.ndarray:
    """A clock accumulated in float32 from *start*."""
    return np.cumsum(
        np.concatenate(
            [[np.float32(start)], np.full(rows - 1, np.float32(1 / fs), np.float32)]
        ),
        dtype=np.float32,
    )


class TestFloat32Detection:
    @pytest.mark.parametrize("style", ["%.6f", "%.7f", "%.8f", "%.9f"])
    @pytest.mark.parametrize("start", [330.0, 1000.0])
    def test_a_float32_clock_is_duration_only_at_any_print(self, style, start):
        """Prints that put a float32 value exactly half a tick from its
        print (a tie) are recognized too: the test is exact."""
        result = assess([style % float(v) for v in float32_clock(start)])
        assert result.float32_values
        assert result.evidence == "duration_only"
        assert result.fitted_rate_hz is None and result.proposed_rate_hz is None
        assert result.outcome == "accepted"

    @pytest.mark.parametrize(
        "tokens_of",
        [
            lambda n: "%.6f" % (20 + n / 64),
            lambda n: "%.12f" % (n / 4096),
        ],
        ids=["%.6f n/64 from 20 s", "%.12f n/4096"],
    )
    def test_a_missing_sample_of_an_exact_float32_clock_is_a_gap(self, tokens_of):
        """Exact clocks at power-of-two rates are float32 values, so they carry
        no derived rate; their steps are still tested, whatever the rounding."""
        rows = 2000
        clean = assess([tokens_of(n) for n in range(rows)])
        assert clean.float32_values and clean.outcome == "accepted"
        gapped = assess([tokens_of(n) for n in range(rows) if n != 1000])
        assert gapped.outcome == "invalid"
        assert finding(gapped, "time_gap").lines == (FIRST_LINE + 1000,)

    def test_a_power_of_two_repr_clock_is_float32_and_still_tests_its_steps(self):
        """n/32768 is exactly a float32 value, so its repr cannot be told from
        float32 data: no derived rate, but its steps are still tested."""
        result = assess([repr(n / 32768) for n in range(N)])
        assert result.float32_values and result.evidence == "duration_only"
        assert result.outcome == "accepted" and result.proposed_rate_hz is None
        gapped = assess([repr(n / 32768) for n in range(N) if n != 10000])
        assert codes(gapped) == {"time_gap"}

    def test_a_short_significant_digit_print_is_not_mistaken_for_float32(self):
        """Six significant digits round-trip through float32 whatever they
        hold, so they are no float32 evidence."""
        result = assess(["%g" % (n / FS) for n in range(N)])
        assert not result.float32_values
        assert result.evidence == "full"

    def test_a_frozen_float32_clock_is_no_evidence(self):
        """Past 5000 s a float32 clock no longer advances at 25.6 kHz."""
        result = assess(["%.6f" % float(v) for v in float32_clock(5000.0, rows=500)])
        assert result.evidence == "duration_only" and result.outcome == "accepted"
        stated = assess(
            ["%.6f" % float(v) for v in float32_clock(5000.0, rows=500)],
            stated_rate_hz="25600",
        )
        assert stated.outcome == "needs_confirmation"


class TestPrintRegimes:
    @pytest.mark.parametrize(
        "tokens",
        [
            [repr(n / 12000) for n in range(12000, 30000)],
            [repr(0.1 + n / FS) for n in range(N)],
            [repr(n / 12000) for n in range(N)],
            ["%.17f" % (n / 12000) for n in range(N)],
        ],
        ids=["repr 1 to 2.5 s", "repr 0.1 + n/fs", "repr n/fs", "%.17f"],
    )
    def test_digits_beyond_float64_precision_are_shortest_repr(self, tokens):
        result = assess(tokens)
        assert result.regime == "shortest_repr"
        assert_accepted(result)

    def test_a_cumulative_sum_from_10_s_is_shortest_repr(self):
        tokens, t = [], 10.0
        for _ in range(N):
            tokens.append(repr(t))
            t += 1 / FS
        result = assess(tokens)
        assert result.regime == "shortest_repr"
        assert_accepted(result)

    @pytest.mark.parametrize(
        "fs, decimals, evidence", [(FS, 6, "full"), (1000, 3, "partial")]
    )
    def test_stripped_decimals_ending_on_a_round_stamp_keep_their_resolution(
        self, fs, decimals, evidence
    ):
        """A last stamp "10" after "9.999..." reads as a %g row (it lies in a
        decade above every widest row): the column keeps its grid and
        evidence and loses only its rate proposal. The other reading would
        refuse a %g column's legitimate repeated stamps."""
        rows = 10 * fs + 1  # the last stamp is exactly 10 s
        tokens = [t.rstrip("0").rstrip(".") for t in printed(range(rows), fs, decimals)]
        assert tokens[-1] == "10"
        result = assess(tokens)
        assert result.regime == "significant_digits"
        assert result.resolution == _fixed(1, decimals)
        assert result.evidence == evidence
        assert_accepted(result)
        assert result.proposed_rate_hz is None
        # Without the round last stamp it is plain fixed decimals.
        assert assess(tokens[:-1]).regime == "fixed_decimals"

    def test_a_g_print_across_a_decade_keeps_its_finer_resolution(self):
        stamps = range(int(9.2 * FS), int(10.2 * FS))
        tokens = ["%g" % (n / FS) for n in stamps]
        result = assess(tokens, stated_rate_hz="25600")
        assert result.regime == "significant_digits"
        assert result.resolution == "0.00001" and result.evidence == "full"
        assert_accepted(result)
        hole = int(9.5 * FS)
        gapped = assess(
            ["%g" % (n / FS) for n in stamps if not hole <= n < hole + 3],
            stated_rate_hz="25600",
        )
        assert gapped.outcome == "invalid" and "time_gap" in codes(gapped)


class TestDurationOnlySteps:
    def test_a_long_gap_at_4_decimals_is_refused(self):
        kept = [n for n in range(N) if not 5000 <= n < 6000]
        result = assess(printed(kept, FS, 4))
        assert result.evidence == "duration_only"
        assert result.outcome == "invalid"
        assert finding(result, "time_gap").lines == (FIRST_LINE + 5000,)

    def test_a_single_missing_sample_at_4_decimals_cannot_be_seen(self):
        kept = [n for n in range(N) if n != 5000]
        assert assess(printed(kept, FS, 4)).outcome == "accepted"

    @pytest.mark.parametrize("copies", [2, 3])
    def test_repeated_stamps_are_duplicates_not_a_coarse_print(self, copies):
        tokens = [t for t in uniform(6, rows=2000) for _ in range(copies)]
        result = assess(tokens)
        assert result.evidence == "full"
        assert result.reference_interval_ticks == pytest.approx(39.0625, abs=0.01)
        assert codes(result) == {"time_reversal"}


class TestUncertaintyBound:
    def test_the_true_rate_lies_within_the_reported_interval(self):
        rng = np.random.default_rng(11)
        for _ in range(300):
            fs = int(rng.integers(1000, 100_000))
            decimals = int(rng.integers(5, 10))
            rows = int(rng.integers(3, 40))
            result = assess(uniform(decimals, rows=rows, fs=fs))
            if result.fitted_rate_hz is None or result.rate_uncertainty_hz is None:
                continue
            (own,) = [s for s in result.rate_sources if s.source == "time_column"]
            assert Decimal(own.low_hz) <= fs <= Decimal(own.high_hz), (
                fs,
                decimals,
                rows,
            )
            assert abs(Decimal(result.fitted_rate_hz) - fs) <= Decimal(
                result.rate_uncertainty_hz
            )

    def test_the_bound_is_the_exact_worst_case_not_first_order(self):
        """5 stamps, half a tick each: the slope may be off by
        0.5 * 12 * floor(25/4) / (5 * 24) = 0.3 ticks, so the rate lies within
        1e5 / (slope +- 0.3), not within the first-order rate * 0.3 / slope."""
        result = assess(uniform(5, rows=5, fs=51200))
        (own,) = [s for s in result.rate_sources if s.source == "time_column"]
        slope = Decimal(10**5) / Decimal(result.fitted_rate_hz)
        worst = Decimal("0.3")
        assert Decimal(own.high_hz) == pytest.approx(
            Decimal(10**5) / (slope - worst), rel=Decimal("1e-9")
        )
        assert Decimal(own.low_hz) == pytest.approx(
            Decimal(10**5) / (slope + worst), rel=Decimal("1e-9")
        )
        first_order = Decimal(result.fitted_rate_hz) * worst / slope
        assert Decimal(result.rate_uncertainty_hz) > first_order

    def test_two_stamps_bound_no_rate(self):
        result = assess(uniform(5, rows=2, fs=95000))  # one step of 1 tick
        assert result.evidence == "partial"
        assert result.fitted_rate_hz is not None
        assert result.rate_uncertainty_hz is None
        assert [s.source for s in result.rate_sources] == []
        assert result.proposed_rate_hz is None


class TestChunkBoundaries:
    COLUMNS = {
        "uniform": lambda: uniform(6, rows=5000),
        "gap": lambda: printed([n for n in range(5000) if n != 2500], FS),
        "jitter": lambda: [
            _fixed(int(k), 6)
            for k in np.rint(
                np.arange(5000) * 39.0625 + np.random.default_rng(5).normal(0, 2, 5000)
            )
        ],
        "drift": lambda: printed(
            [n * (5000 * 5000 + n) for n in range(5000)], 5000 * 5000 * FS
        ),
        "%g": lambda: ["%g" % (n / FS) for n in range(int(9.9 * FS), int(10.1 * FS))],
        "float32": lambda: ["%.7f" % float(v) for v in float32_clock(330.0, rows=5000)],
        "4 decimals": lambda: printed(
            [n for n in range(5000) if not 100 <= n < 300], FS, 4
        ),
        "print ties": lambda: tie_tokens(tie_print(mid_interval(5000, 1))),
    }

    @pytest.mark.parametrize("name", sorted(COLUMNS))
    def test_results_do_not_depend_on_the_chunk_size(self, name, monkeypatch):
        tokens = self.COLUMNS[name]()
        whole = assess(tokens, stated_rate_hz="25600")
        if name == "print ties":  # the column reaches the print-tie check
            assert "only print ties explain" in whole.findings[0].statement
        for size in (7, 1000):
            monkeypatch.setattr(tb, "_CHUNK_ROWS", size)
            chunked = assess(tokens, stated_rate_hz="25600")
            assert (chunked.outcome, chunked.evidence, chunked.regime) == (
                whole.outcome,
                whole.evidence,
                whole.regime,
            )
            assert [(f.code, f.count, f.lines) for f in chunked.findings] == [
                (f.code, f.count, f.lines) for f in whole.findings
            ]
            assert [c.consistent for c in chunked.rate_comparisons] == [
                c.consistent for c in whole.rate_comparisons
            ]
            if whole.fitted_rate_hz is not None:
                assert Decimal(chunked.fitted_rate_hz) == pytest.approx(
                    Decimal(whole.fitted_rate_hz), rel=Decimal("1e-11")
                )


class TestRefusalText:
    def test_a_quote_in_a_token_cannot_close_the_citation(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            tb.parse_time_tokens(["0.5", "1' is valid. Import it '"])
        message = str(caught.value)
        assert "the time stamp '1\\' is valid. Import it \\'' is not" in message

    def test_a_quote_in_a_declared_token_cannot_close_the_citation(self):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            assess(uniform(6, rows=10), declared=tb.DeclaredTimeBase("rate", "1' ok '"))
        assert "'1\\' ok \\''" in str(caught.value)

    def test_the_decoder_escapes_quotes_too(self, tmp_path):
        from predictive_maintenance_mcp.path_safety import open_import_file

        (tmp_path / "x.csv").write_bytes(b"t;x\n0.0;1\n0.1' ok ';2\n")
        interpretation = {
            "schema": ic.INTERPRETATION_SCHEMA,
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ";",
                "decimal": ".",
                "header_line": 1,
            },
            "data_lines": {"first": 2, "last": None},
            "time_base": {"source": "stated", "sampling_rate_hz": "10"},
            "time_columns": [{"column": 0, "time_unit": "s"}],
            "channels": [
                {
                    "column": 1,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "scaling": None,
                }
            ],
        }
        with open_import_file(
            "x.csv", [tmp_path], base=tmp_path, max_size=1 << 20
        ) as f:
            with pytest.raises(td.TextDecodeError) as caught:
                td.decode_text(f, interpretation)
        assert "'0.1\\' ok \\''" in str(caught.value)


class TestScaleBounds:
    @pytest.mark.parametrize(
        "tokens",
        [
            ["1e400", "2e400", "3e400"],
            ["1e-400", "2e-400", "3e-400"],
            ["0", "2e-32", "4e-32"],
        ],
    )
    def test_stamps_outside_the_held_scale_are_refused(self, tokens):
        with pytest.raises(tb.TimeBaseRefused) as caught:
            assess(tokens)
        assert caught.value.code == "time_grammar"

    def test_nanoseconds_extend_the_scale_by_their_unit(self):
        with pytest.raises(tb.TimeBaseRefused):
            assess(["0", "2e-22", "4e-22"], unit="ns")

    @pytest.mark.parametrize(
        "mantissa, exponent",
        [
            (np.array([0, 2**63], dtype=np.uint64), np.zeros(2, np.int16)),
            (np.array([0, 10**18], dtype=np.int64), np.zeros(2, np.int16)),
            (np.array([0, 1], dtype=np.int64), np.array([0, 40000], dtype=np.int64)),
        ],
    )
    def test_arrays_beyond_what_the_decoder_produces_are_refused(
        self, mantissa, exponent
    ):
        with pytest.raises(ValueError, match="text_decoder"):
            tb.assess_time_base(mantissa, exponent, time_unit="s", first_line=1)


class TestSecondReviewRound:
    """Cases the second review round found (each failed before its fix)."""

    @pytest.mark.parametrize(
        "write",
        [
            lambda v: np.format_float_positional(v, trim="-"),
            lambda v: str(v),
        ],
        ids=["pandas float32 to_csv", "str(np.float32)"],
    )
    def test_shortest_float32_reprs_of_a_uniform_clock_are_not_refused(self, write):
        values = (330.0 + np.arange(N) / FS).astype(np.float32)
        tokens = [write(v) for v in values]
        assert tokens[1:3] == ["330.00003", "330.0001"]
        result = assess(tokens, stated_rate_hz="25600")
        assert result.float32_values and result.evidence == "duration_only"
        assert result.outcome == "accepted", result.findings

    @pytest.mark.parametrize("style", ["%.13f", "%.14f", "%.15f"])
    @pytest.mark.parametrize("start", [20.0, 330.0])
    def test_float32_clocks_printed_past_12_decimals_derive_no_rate(self, style, start):
        result = assess([style % float(v) for v in float32_clock(start)])
        assert result.float32_values
        assert result.proposed_rate_hz is None and result.fitted_rate_hz is None

    @pytest.mark.parametrize(
        "start, fs, rows", [(256.56, 65536, 5), (151.36, 12000, 2), (330.0, FS, 50)]
    )
    def test_short_float64_reprs_of_float32_values_are_not_refused(
        self, start, fs, rows
    ):
        values = np.float32(start) + np.arange(rows, dtype=np.float32) / np.float32(fs)
        tokens = [repr(float(v)) for v in values.astype(np.float32)]
        result = assess(tokens, stated_rate_hz=str(fs))
        assert result.outcome == "accepted", result.findings

    @pytest.mark.parametrize(
        "tokens, stated",
        [
            (["%.4g" % ((9948 + i) / 10000) for i in range(78)], None),
            (["%.5g" % ((51200 - 230 + i) / 51200) for i in range(232)], None),
            (["%.6g" % ((441000 - 122 + i) / 44100) for i in range(125)], None),
            (["%.4g" % ((200000 - 167 + i) / 20000) for i in range(201)], "20000"),
            (["%.7e" % ((256000 - 77 + i) / 25600) for i in range(118)], None),
        ],
        ids=[
            "%.4g across 1 s",
            "%.5g ending 1, 1",
            "%.6g across 10 s",
            "%.4g with a stated rate",
            "%.7e across 10 s",
        ],
    )
    def test_significant_digit_prints_across_a_decade_are_accepted(
        self, tokens, stated
    ):
        result = assess(tokens, stated_rate_hz=stated)
        assert result.regime == "significant_digits"
        assert_accepted(result)

    def test_a_short_16_digit_repr_cumulative_sum_is_shortest_repr(self):
        tokens, t = [], 64962.6
        for _ in range(10):
            tokens.append(repr(t))
            t += 1 / 44100
        result = assess(tokens, stated_rate_hz="44100")
        assert result.regime == "shortest_repr"
        assert_accepted(result)


# ---------------------------------------------------------------------------
# Print ties (owner decision of 2026-11-04)
# ---------------------------------------------------------------------------


def tie_print(instants, rounding="even"):
    """Integer ticks of exact instants (Fractions of a tick), each the correct
    rounding, a tie to even (half to even, as printf prints an exact tie), up,
    down, alternately, or by a seeded coin per row (ties decided row by row,
    as float noise decides them for a float64 instant next to a decimal
    tie)."""
    ticks = []
    for i, t in enumerate(instants):
        low = t.numerator // t.denominator
        frac = t - low
        if frac != Fraction(1, 2):
            ticks.append(low + (1 if frac > Fraction(1, 2) else 0))
        elif rounding == "even":
            ticks.append(low + (low % 2))
        elif rounding == "up":
            ticks.append(low + 1)
        elif rounding == "down":
            ticks.append(low)
        elif rounding == "alternate":
            ticks.append(low + (i % 2))
        else:  # noise
            ticks.append(low + (random.Random(i).random() < 0.5))
    return ticks


def tie_tokens(ticks, decimals=3):
    return [_fixed(k, decimals) for k in ticks]


def mid_interval(rows, interval, start=Fraction(1, 2)):
    return [start + i * Fraction(interval) for i in range(rows)]


def oracle_band(ticks, height=1):
    """The exact intervals ``beta`` (ticks per sample) for which some start
    ``c`` puts every stamp within ``height / 2`` of its instant ``c + i *
    beta``: the intersection, over every pair of rows, of the intervals that
    pair allows. None when empty."""
    low, high = None, None
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            gap = Fraction(ticks[j] - ticks[i])
            lo, hi = (gap - height) / (j - i), (gap + height) / (j - i)
            low = lo if low is None else max(low, lo)
            high = hi if high is None else min(high, hi)
    return None if low > high else (low, high)


def oracle_half_even(ticks):
    """Is the column the half-to-even print of one advancing uniform clock?
    Brute force, exact: an interval inside the band's admitted range leaves
    the start free, with no stamp on a tie; a single admitted interval fixes
    the start, and every stamp it puts on a tie must be an even tick."""
    band = oracle_band(ticks)
    if band is None or band[1] <= 0:
        return False
    low, high = band
    if low < high:
        return True
    starts_low = [k - Fraction(1, 2) - i * low for i, k in enumerate(ticks)]
    starts_high = [k + Fraction(1, 2) - i * low for i, k in enumerate(ticks)]
    c_low, c_high = max(starts_low), min(starts_high)
    if c_low < c_high:
        return True
    if c_low > c_high:
        return False
    tied = [i for i, s in enumerate(starts_low) if s == c_low] + [
        i for i, s in enumerate(starts_high) if s == c_high
    ]
    return all(ticks[i] % 2 == 0 for i in tied)


def tie_column(tokens):
    mantissa, exponent = tb.parse_time_tokens(tokens)
    return tb._column(mantissa, exponent, "s", FIRST_LINE)


class TestPrintTies:
    def test_mid_interval_stamps_at_one_tick_per_sample_need_confirmation(self):
        """The retired tripwire ``timebase-print-ties-misread``: 1 kHz stamps
        at the middle of each interval, printed with 3 decimals, half to
        even (0.000, 0.002, 0.002, 0.004, ...)."""
        ticks = tie_print(mid_interval(2000, 1))
        assert ticks[:4] == [0, 2, 2, 4]
        result = assess(tie_tokens(ticks))
        assert result.outcome == "needs_confirmation"
        assert codes(result) == {"time_off_pattern"}
        statement = finding(result, "time_off_pattern").statement
        assert "only print ties explain" in statement
        assert (
            "half-to-even print of one uniform clock of 1 tick per sample" in statement
        )
        assert "(1000 Hz)" in statement
        assert result.evidence == "partial"
        assert result.proposed_rate_hz is None and result.fitted_rate_hz is None
        assert result.rate_sources == ()

    @pytest.mark.parametrize("interval", [1, Fraction(3, 2), 2, Fraction(5, 2), 3])
    @pytest.mark.parametrize("rounding", ["even", "up", "down"])
    @pytest.mark.parametrize("start", [Fraction(1, 2), Fraction(0), Fraction(1, 4)])
    def test_a_uniform_clock_printed_by_one_rounding_rule_is_never_invalid(
        self, interval, rounding, start
    ):
        ticks = tie_print(mid_interval(400, interval, start), rounding)
        assert oracle_half_even(ticks) or rounding != "even"
        result = assess(tie_tokens(ticks))
        assert result.outcome != "invalid", result.findings
        if result.outcome == "needs_confirmation":
            assert result.proposed_rate_hz is None

    def test_ties_decided_row_by_row_are_not_excused(self, monkeypatch):
        """Ties decided by float noise follow no single rounding rule: the
        column is not the half-to-even print of any clock, so the check
        leaves the step tests' verdict as it was (a documented limit)."""
        ticks = tie_print(mid_interval(400, 1), "noise")
        assert oracle_band(ticks) is not None  # ties either way: a clock fits
        assert not oracle_half_even(ticks)
        tokens = tie_tokens(ticks)
        with monkeypatch.context() as patched:
            patched.setattr(tb, "_tie_band", lambda column: None)
            alone = assess(tokens)
        assert alone.outcome == "invalid"
        assert assess(tokens) == alone

    def test_the_ambiguity_replaces_exactly_the_refusals_of_tie_prints(
        self, monkeypatch
    ):
        """Where the step tests alone refuse a uniform clock printed half to
        even, the check turns the refusal into needs_confirmation; where
        they accept it, it stays accepted."""
        refused = 0
        for interval in (1, Fraction(3, 2), 2, Fraction(5, 2)):
            for start in (Fraction(1, 2), Fraction(0)):
                tokens = tie_tokens(tie_print(mid_interval(400, interval, start)))
                with monkeypatch.context() as patched:
                    patched.setattr(tb, "_tie_band", lambda column: None)
                    alone = assess(tokens)
                result = assess(tokens)
                if alone.outcome == "invalid":
                    refused += 1
                    assert result.outcome == "needs_confirmation"
                    statement = finding(result, "time_off_pattern").statement
                    assert "only print ties explain" in statement
                else:
                    assert result == alone
        assert refused >= 1  # at least the one-tick mid-interval clock

    def test_uniform_clocks_without_ties_are_unchanged(self):
        for interval in (Fraction(21, 20), Fraction(39, 25), 39):
            ticks = tie_print(mid_interval(2000, interval, Fraction(1, 7)))
            assert_accepted(assess(tie_tokens(ticks)))

    @pytest.mark.parametrize("interval", [1, Fraction(3, 2), 2])
    def test_a_genuine_missing_sample_stays_a_gap(self, interval):
        instants = mid_interval(400, interval)
        del instants[200]
        ticks = tie_print(instants)
        assert not oracle_half_even(ticks)
        result = assess(tie_tokens(ticks))
        assert result.outcome == "invalid"
        assert codes(result) & {"time_gap", "time_reversal"}

    @pytest.mark.parametrize("defect", ["duplicate", "swap", "restart", "backward"])
    def test_genuine_repeats_and_reversals_stay_refused(self, defect):
        ticks = tie_print(mid_interval(400, Fraction(3, 2)))
        if defect == "duplicate":
            ticks.insert(200, ticks[200])
        elif defect == "swap":
            ticks[200], ticks[201] = ticks[201], ticks[200]
        elif defect == "restart":
            ticks = ticks[:200] + ticks[:200]
        else:
            ticks[200] = ticks[199] - 1
        assert not oracle_half_even(ticks)
        assert assess(tie_tokens(ticks)).outcome == "invalid", defect

    @pytest.mark.parametrize("rows", [401, 200001])
    def test_drops_and_a_duplicate_in_a_clean_log_stay_refused(self, rows):
        """A clean one-tick log (k_i = i, no stamp on a tie) with a dropped
        row, a duplicated row and another dropped row: a clock with every
        stamp on a tie could print it only if its ties were rounded up and
        down at exactly those rows, which half to even does not do."""
        ticks = list(range(rows))
        del ticks[3 * rows // 4]
        ticks.insert(rows // 2, ticks[rows // 2])
        del ticks[rows // 4]
        if rows <= 401:
            assert oracle_band(ticks) is not None and not oracle_half_even(ticks)
        result = assess(tie_tokens(ticks))
        assert result.outcome == "invalid"
        assert "time_reversal" in codes(result)

    def test_a_real_rate_change_stays_refused(self):
        instants = mid_interval(200, 1) + [
            Fraction(1, 2) + 200 + i * Fraction(3, 2) for i in range(200)
        ]
        ticks = tie_print(instants)
        assert not oracle_half_even(ticks)
        assert assess(tie_tokens(ticks)).outcome == "invalid"

    def test_a_stated_rate_is_compared_but_never_validates(self):
        tokens = tie_tokens(tie_print(mid_interval(2000, 1)))
        for rate, consistent in (("1000", True), ("1000.0001", True), ("1010", False)):
            result = assess(tokens, stated_rate_hz=rate)
            assert result.outcome == "needs_confirmation"
            assert result.proposed_rate_hz is None
            assert comparison(result, "time_column", "stated").consistent is consistent
            expected = {"time_off_pattern"} | (
                set() if consistent else {"rate_sources_disagree"}
            )
            assert codes(result) == expected
        off = comparison(assess(tokens, stated_rate_hz="1010"), "time_column", "stated")
        assert off.difference_ppm == pytest.approx(10000)

    def test_a_declared_interval_is_compared_but_never_validates(self):
        tokens = tie_tokens(tie_print(mid_interval(2000, 1)))
        inside = tb.DeclaredTimeBase(kind="interval", token="0.001")
        result = assess(tokens, declared=inside)
        assert result.outcome == "needs_confirmation"
        assert comparison(result, "time_column", "file_declared").consistent
        outside = tb.DeclaredTimeBase(kind="interval", token="0.0011")
        assert not comparison(
            assess(tokens, declared=outside), "time_column", "file_declared"
        ).consistent

    def test_the_stated_rule_agrees_with_the_exact_oracle(self):
        """A stated rate fits an ambiguous column exactly when its clock
        predicts every stamp within one tick, KTD9's rule for any column."""
        ticks = tie_print(mid_interval(60, 1))
        tokens = tie_tokens(ticks)
        assert assess(tokens).outcome == "needs_confirmation"
        low, high = oracle_band(ticks, 2)
        verdicts = set()
        for rate in ("1000", "1017", "1017.3", "983", "982.9", "1100"):
            interval = Fraction(1000) / Fraction(rate)
            expected = low <= interval <= high
            verdicts.add(expected)
            stated = assess(tokens, stated_rate_hz=rate)
            assert comparison(stated, "time_column", "stated").consistent is expected
        assert verdicts == {True, False}

    @pytest.mark.parametrize(
        "ticks, outcome",
        [
            ([0, 2, 2], "needs_confirmation"),
            ([0, 2, 2, 4], "needs_confirmation"),
            ([0, 0, 2], "needs_confirmation"),
            ([1, 1, 3], "invalid"),
            ([0, 0, 3], "invalid"),
            ([0, 2, 0], "invalid"),
        ],
    )
    def test_short_sequences(self, ticks, outcome):
        result = assess(tie_tokens(ticks))
        assert result.outcome == outcome, result.findings
        if outcome == "needs_confirmation":
            assert oracle_half_even(ticks) and result.proposed_rate_hz is None

    def test_against_the_oracle_on_random_columns(self, monkeypatch):
        """Both directions: the check changes an outcome only where the
        column is the half-to-even print of one clock, and only from invalid
        to needs_confirmation without a rate; and every column the step
        tests alone refuse within the check's scope that the oracle admits
        is changed."""
        rng = random.Random(20261104)
        kinds = ["uniform", "gap", "duplicate", "swap", "rate", "jitter"]
        changed = admitted_refusals = 0
        for _ in range(1500):
            # Weighted towards mid-interval clocks, where print ties live.
            interval = rng.choice(
                [1, 1, 1, Fraction(3, 2), 2, Fraction(5, 2), Fraction(7, 5)]
            )
            start = rng.choice(
                [Fraction(1, 2), Fraction(1, 2), Fraction(0), Fraction(1, 3)]
            )
            rows = rng.randint(3, 40)
            instants = mid_interval(rows, interval, start)
            kind = rng.choice(kinds)
            k = rng.randint(1, rows - 2)
            if kind == "gap":
                del instants[k]
            elif kind == "duplicate":
                instants.insert(k, instants[k])
            elif kind == "swap":
                instants[k], instants[k + 1] = instants[k + 1], instants[k]
            elif kind == "rate":
                instants = instants[:k] + [
                    instants[k - 1] + (i + 1) * interval * Fraction(4, 3)
                    for i in range(rows - k)
                ]
            elif kind == "jitter":
                instants = [t + Fraction(rng.randint(-3, 3), 8) for t in instants]
            ticks = tie_print(
                instants,
                rng.choice(["even", "even", "up", "down", "alternate", "noise"]),
            )
            tokens = tie_tokens(ticks)
            with_ties = assess(tokens)
            with monkeypatch.context() as patched:
                patched.setattr(tb, "_tie_band", lambda column: None)
                without = assess(tokens)
            steps = np.diff(ticks)
            in_scope = (
                without.outcome == "invalid"
                and codes(without) <= {"time_gap", "time_reversal"}
                and steps.min() >= 0
                and steps.max() - steps.min() <= 2
            )
            if in_scope and oracle_half_even(ticks):
                admitted_refusals += 1
                assert with_ties.outcome == "needs_confirmation", ticks
                assert with_ties.proposed_rate_hz is None
            if with_ties != without:
                changed += 1
                assert in_scope and oracle_half_even(ticks), (kind, ticks)
        assert changed == admitted_refusals > 30

    def test_the_band_agrees_with_the_exact_oracle(self):
        """`_tie_band` and `_Band.level` against the pairwise oracle on
        random columns in its scope, ties included, with exact endpoints."""
        rng = random.Random(4)
        checked = 0
        for _ in range(3000):
            rows = rng.randint(2, 30)
            steps = [
                rng.choice([0, 1, 2]) if rng.random() < 0.7 else 1
                for _ in range(rows - 1)
            ]
            base = rng.randint(0, 2)
            ticks = [0]
            for step in steps:
                ticks.append(ticks[-1] + base + step)
            column = tie_column(tie_tokens(ticks))
            values = np.asarray(ticks, dtype=np.int64) - ticks[0]
            values -= np.arange(rows, dtype=np.int64) * int(column.steps.min())
            band = tb._Band(values)
            for height in (1, 2):
                level = band.level(height)
                expected = oracle_band(ticks, height)
                m0 = int(column.steps.min())
                got = None if level is None else (m0 + level[0], m0 + level[1])
                assert got == expected, (ticks, height)
            ties = tb._tie_band(column)
            assert (ties is not None) == oracle_half_even(ticks), ticks
            if ties is not None:
                low, high = oracle_band(ticks)
                assert (ties.low, ties.high) == (max(low, Fraction(0)), high)
                checked += 1
        assert checked > 300

    def test_a_constant_column_clamps_its_intervals_at_zero(self):
        ties = tb._tie_band(tie_column(["1.000"] * 5))
        assert ties is not None
        assert ties.low == 0 and ties.high == Fraction(1, 4)
        assert ties.predict_low == 0 and ties.predict_high == Fraction(1, 2)

    def test_the_statement_names_a_range_of_clocks(self):
        column = tie_column(tie_tokens(tie_print(mid_interval(200, 1))))
        findings = [tb.TimeFinding("time_reversal", 3, (7, 9, 11), (), "")]
        ties = tb._Ties(Fraction(1), Fraction(5, 4), Fraction(1), Fraction(2))
        statement = tb._tie_finding(column, findings, ties).statement
        assert "1 tick to 1.25 ticks per sample (800 to 1000 Hz)" in statement


class TestPrintTieScope:
    """The check runs only on a refused fixed-decimal column, no float32
    values, no extra rounding, no backward step, steps within two ticks."""

    @pytest.fixture
    def refused(self):
        tokens = tie_tokens(tie_print(mid_interval(400, 1)))
        column = tie_column(tokens)
        findings = tb._step_findings(column)
        assert findings and tb._tie_candidate(column, findings)
        return column, findings

    @pytest.mark.parametrize(
        "change",
        [
            {"printed_regime": "significant_digits"},
            {"printed_regime": "shortest_repr"},
            {"spacing": 2.0**-15},
            {"largest_extra": 0.5},
        ],
    )
    def test_each_column_condition_excludes(self, refused, change):
        import dataclasses

        column, findings = refused
        if "printed_regime" in change:
            printed = dataclasses.replace(
                column.printed, regime=change["printed_regime"]
            )
            column = dataclasses.replace(column, printed=printed)
        else:
            column = dataclasses.replace(column, **change)
        assert not tb._tie_candidate(column, findings)

    def test_backward_steps_wide_steps_and_other_findings_exclude(self, refused):
        column, findings = refused
        residual = tb.TimeFinding("time_residual", 1, (7,), (), "")
        assert not tb._tie_candidate(column, [*findings, residual])
        assert not tb._tie_candidate(column, [])
        for ticks in ([0, 2, 2, 1, 3], [0, 0, 3, 3, 6]):
            other = tie_column(tie_tokens(ticks))
            assert not tb._tie_candidate(other, tb._step_findings(other) or findings)

    def test_a_significant_digit_tie_column_never_reaches_the_check(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tb, "_tie_band", lambda column: calls.append(1))
        tokens = ["%.4g" % ((n + 0.5) / 1000) for n in range(1, 3000)]
        result = assess(tokens)
        assert result.regime == "significant_digits"
        assert result.outcome == "invalid"
        assert calls == []
