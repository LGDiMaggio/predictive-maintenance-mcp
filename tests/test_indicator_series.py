"""Tests for the indicator-series checks and the regular grid (forecast PR 1, U1).

The module turns a caller's ISO 8601 instants and values into a regular UTC
grid without interpolation, and refuses every call error in the caller's
vocabulary before any data-dependent check runs.
"""

import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics import indicator_series as s
from predictive_maintenance_mcp.signal_validation import InvalidInputError

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
T0 = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def _instants(n, step=timedelta(hours=1), start=T0, fmt="Z"):
    out = []
    for i in range(n):
        t = start + i * step
        if fmt == "Z":
            out.append(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            out.append(t.astimezone(timezone(timedelta(hours=2))).isoformat())
    return out


def _prepare(observed_at, values, **overrides):
    kwargs = dict(
        indicator="velocity RMS",
        unit="mm/s",
        resolution="1h",
        horizon_steps=6,
        backend="naive",
        aggregation="mean",
        season_length_steps=None,
        now=NOW,
    )
    kwargs.update(overrides)
    return s.prepare_series(observed_at=observed_at, values=values, **kwargs)


class TestHappyPath:
    def test_hourly_series_with_offsets_gives_one_bin_per_point(self):
        """Covers AE1: 300 hourly instants give 300 observed bins."""
        obs = _instants(300)
        vals = list(np.linspace(1.0, 2.0, 300))
        decl, grid = _prepare(obs, vals, horizon_steps=24)
        assert grid.bins == 300
        report = s.history_report(grid, lag=1)
        assert report["observed_bins"] == 300
        assert report["empty_bins"] == 0
        assert report["first_observed_at"] == "2026-10-01T00:00:00Z"
        assert report["last_observed_at"] == "2026-10-13T11:00:00Z"
        assert report["last_value"] == pytest.approx(2.0)
        assert report["raw_points"] == 300
        assert report["bin_alignment"] == "utc_epoch"
        assert decl.step_seconds == 3600

    def test_offset_instants_are_normalised_to_utc(self):
        obs = _instants(20, fmt="+02:00")
        decl, grid = _prepare(obs, [1.0] * 19 + [2.0])
        report = s.history_report(grid, lag=1)
        assert report["first_observed_at"] == "2026-10-01T00:00:00Z"

    @pytest.mark.parametrize(
        "aggregation, expected",
        [("mean", 4.0), ("median", 3.5), ("max", 9.0), ("last", 3.0)],
    )
    def test_aggregation_per_bin(self, aggregation, expected):
        """Six 10-minute points per 1h bin, unsorted, so the four methods differ."""
        obs = _instants(24, step=timedelta(minutes=10))
        vals = [1.0, 2.0, 9.0, 4.0, 5.0, 3.0] * 4
        _, grid = _prepare(obs, vals, aggregation=aggregation)
        assert grid.bins == 4
        assert grid.values[0] == pytest.approx(expected)
        assert s.history_report(grid, lag=1)["bins_with_several_points"] == 4

    def test_median_of_shared_and_single_point_bins(self):
        """Bins of three, two and one point: 20-minute instants on a 1h grid."""
        obs = _instants(7, step=timedelta(minutes=20))
        del obs[5]
        _, grid = _prepare(obs, [5.0, 1.0, 2.0, 2.0, 8.0, 9.0], aggregation="median")
        assert list(grid.values) == [2.0, 5.0, 9.0]
        assert list(grid.counts) == [3, 2, 1]

    def test_instant_on_a_boundary_lands_in_the_later_bin(self):
        obs = ["2026-10-01T00:59:59.999999Z", "2026-10-01T01:00:00Z"]
        _, grid = _prepare(obs, [1.0, 2.0])
        assert grid.bins == 2
        assert list(grid.counts) == [1, 1]

    def test_empty_bins_stay_empty(self):
        obs = _instants(10)
        del obs[4:7]
        vals = [float(i) for i in range(len(obs))]
        _, grid = _prepare(obs, vals)
        assert grid.bins == 10
        assert np.isnan(grid.values[4:7]).all()
        report = s.history_report(grid, lag=1)
        assert report["empty_bins"] == 3
        assert report["observed_bins"] == 7

    def test_history_counts_pairs_at_the_backend_lag(self):
        obs = _instants(72)
        vals = [float(i % 24) + 0.1 * i for i in range(72)]
        _, grid = _prepare(obs, vals, backend="seasonal_naive", season_length_steps=24)
        assert s.history_report(grid, lag=24)["difference_pairs"] == 48
        assert s.history_report(grid, lag=24)["difference_lag"] == 24
        assert s.history_report(grid, lag=1)["difference_pairs"] == 71

    def test_last_bin_and_isolated_origin_counts(self):
        obs = _instants(10, step=timedelta(minutes=30))
        obs = obs[:6] + [obs[-1]]  # three bins of two points, a gap, then one point
        _, grid = _prepare(obs, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        report = s.history_report(grid, lag=1)
        assert report["last_bin_points"] == 1
        assert report["median_points_per_observed_bin"] == pytest.approx(2.0)
        assert report["empty_bins_before_last_observed"] == 1

    @pytest.mark.parametrize(
        "text, seconds",
        [
            ("1min", 60),
            ("15min", 900),
            ("1h", 3600),
            ("1d", 86400),
            ("1w", 604800),
            ("30s", 30),
        ],
    )
    def test_resolution_grammar_accepts(self, text, seconds):
        assert s.parse_resolution(text) == seconds


class TestRefusals:
    def test_empty_lists(self):
        with pytest.raises(InvalidInputError, match="at least one entry"):
            _prepare([], [])

    def test_length_mismatch(self):
        with pytest.raises(
            InvalidInputError, match="observed_at has 3 entries but values has 2"
        ):
            _prepare(_instants(3), [1.0, 2.0])

    def test_raw_point_cap_before_parsing(self):
        n = s.MAX_INPUT_POINTS + 1
        with pytest.raises(
            InvalidInputError,
            match=f"{n} points exceed the limit of {s.MAX_INPUT_POINTS}",
        ):
            _prepare(["not parsed"] * n, [1.0] * n)

    def test_null_value_names_its_index_and_the_gap_rule(self):
        """Covers AE2."""
        vals = [1.0] * 20
        vals[7] = None
        with pytest.raises(InvalidInputError) as exc:
            _prepare(_instants(20), vals)
        msg = str(exc.value)
        assert "values[7]" in msg
        assert "omitting its instant" in msg
        assert "interpolat" not in msg.split("never")[0]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_value(self, bad):
        vals = [1.0] * 20
        vals[3] = bad
        with pytest.raises(InvalidInputError, match="sample index 3"):
            _prepare(_instants(20), vals)

    @pytest.mark.parametrize(
        "field, value, fragment",
        [
            ("unit", "", "unit must be declared"),
            ("unit", "   ", "unit must be declared"),
            ("indicator", "", "indicator must name"),
            ("horizon_steps", 0, "horizon_steps must be an integer between 1 and 256"),
            (
                "horizon_steps",
                True,
                "horizon_steps must be an integer between 1 and 256",
            ),
            (
                "horizon_steps",
                257,
                "horizon_steps must be an integer between 1 and 256",
            ),
            ("season_length_steps", 24, "applies only to the seasonal_naive backend"),
        ],
    )
    def test_declaration_refusals(self, field, value, fragment):
        with pytest.raises(InvalidInputError, match=fragment):
            _prepare(_instants(20), [1.0] * 20, **{field: value})

    def test_seasonal_needs_a_declared_season(self):
        with pytest.raises(InvalidInputError, match="season_length_steps is required"):
            _prepare(_instants(20), [1.0] * 20, backend="seasonal_naive")

    @pytest.mark.parametrize("season", [1, 0, True])
    def test_season_below_two(self, season):
        with pytest.raises(InvalidInputError, match="at least 2"):
            _prepare(
                _instants(20),
                [1.0] * 20,
                backend="seasonal_naive",
                season_length_steps=season,
            )

    @pytest.mark.parametrize(
        "text", ["10m", "M", "1.5h", "0min", "-1h", "1 h", "", "1month"]
    )
    def test_resolution_grammar_refuses(self, text):
        with pytest.raises(InvalidInputError, match="'10min' or '1h'"):
            _prepare(_instants(20), [1.0] * 20, resolution=text)

    def test_horizon_duration_cap(self):
        with pytest.raises(
            InvalidInputError,
            match="over the 30-day limit; at this resolution pass at most 30 steps",
        ):
            _prepare(
                _instants(40, step=timedelta(days=1)),
                [1.0] * 40,
                resolution="1d",
                horizon_steps=31,
            )

    def test_independent_problems_are_reported_together(self):
        vals = [1.0] * 20
        vals[2] = None
        with pytest.raises(InvalidInputError) as exc:
            _prepare(_instants(20), vals, unit="", resolution="10m")
        msg = str(exc.value)
        assert "3 problems" in msg
        assert "unit must be declared" in msg
        assert "'10min' or '1h'" in msg
        assert "values[2]" in msg

    def test_unparseable_instant(self):
        obs = _instants(5)
        obs[2] = "yesterday"
        with pytest.raises(
            InvalidInputError,
            match=r"observed_at\[2\] = 'yesterday' is not an ISO 8601 instant",
        ):
            _prepare(obs, [1.0] * 5)

    def test_instant_without_offset_is_refused(self):
        """Covers AE6."""
        obs = _instants(5)
        obs[3] = "2026-10-01T03:00:00"
        with pytest.raises(InvalidInputError) as exc:
            _prepare(obs, [1.0] * 5)
        msg = str(exc.value)
        assert "observed_at[3]" in msg
        assert "has no timezone" in msg
        assert "'Z'" in msg

    @pytest.mark.parametrize(
        "instant", ["1970-01-01T00:00:10Z", "1899-12-30T00:00:00Z"]
    )
    def test_instant_before_1971(self, instant):
        obs = _instants(5)
        obs[0] = instant
        with pytest.raises(InvalidInputError) as exc:
            _prepare(obs, [1.0] * 5)
        msg = str(exc.value)
        assert "observed_at[0]" in msg and "lies before 1971" in msg
        assert "server clock (2026-10-24T12:00:00Z)" in msg

    @pytest.mark.parametrize(
        "instant", ["9999-12-31T23:00:00-05:00", "0001-01-01T00:30:00+01:00"]
    )
    def test_instant_at_the_edge_of_the_datetime_range(self, instant):
        obs = _instants(5)
        obs[2] = instant
        with pytest.raises(InvalidInputError, match=r"observed_at\[2\].*1971"):
            _prepare(obs, [1.0] * 5)

    def test_newest_first_series_is_told_to_reverse_both_lists(self):
        obs = list(reversed(_instants(5)))
        with pytest.raises(InvalidInputError) as exc:
            _prepare(obs, [1.0] * 5)
        msg = str(exc.value)
        assert "newest first" in msg and "Reverse both lists together" in msg

    @pytest.mark.parametrize("resolution", ["31d", "5w", "1000h"])
    def test_resolution_longer_than_the_duration_cap(self, resolution):
        with pytest.raises(InvalidInputError) as exc:
            _prepare(
                _instants(20, step=timedelta(days=40), start=T0 - timedelta(days=800)),
                [1.0] * 20,
                resolution=resolution,
                horizon_steps=1,
            )
        msg = str(exc.value)
        assert f"resolution {resolution} is longer than the 30-day" in msg
        assert "at most 0 steps" not in msg

    def test_the_longest_resolution_inside_the_cap_allows_one_step(self):
        decl, _ = _prepare(
            _instants(20, step=timedelta(days=30), start=T0 - timedelta(days=600)),
            [float(i) for i in range(20)],
            resolution="30d",
            horizon_steps=1,
        )
        assert decl.step_seconds == 30 * 86400

    def test_instant_two_days_in_the_future(self):
        obs = _instants(5, start=NOW + timedelta(days=2))
        with pytest.raises(
            InvalidInputError, match="more than one day after the server clock"
        ):
            _prepare(obs, [1.0] * 5)

    def test_duplicate_instant(self):
        obs = _instants(5)
        obs[3] = obs[2]
        with pytest.raises(
            InvalidInputError, match=r"observed_at\[3\].*is not after observed_at\[2\]"
        ):
            _prepare(obs, [1.0] * 5)

    def test_out_of_order_instant(self):
        obs = _instants(5)
        obs[3] = "2026-10-01T01:59:59Z"
        with pytest.raises(InvalidInputError, match="strictly increase"):
            _prepare(obs, [1.0] * 5)

    def test_same_instant_with_different_offsets_is_a_duplicate(self):
        obs = ["2026-10-01T00:00:00Z", "2026-10-01T02:00:00+02:00"]
        with pytest.raises(InvalidInputError, match="is not after"):
            _prepare(obs, [1.0, 2.0])

    def test_grid_bin_cap_names_the_largest_gap(self):
        obs = _instants(5, step=timedelta(minutes=1), start=T0 - timedelta(days=400))
        obs.append("2026-10-01T00:00:00Z")
        with pytest.raises(InvalidInputError) as exc:
            _prepare(obs, [1.0] * 6, resolution="1min")
        msg = str(exc.value)
        assert f"over the limit of {s.MAX_GRID_BINS}" in msg
        assert "observed_at[4] and observed_at[5]" in msg

    @pytest.mark.parametrize("aggregation", ["mean", "median"])
    def test_overflowing_aggregation_is_refused_without_a_warning(self, aggregation):
        obs = _instants(2, step=timedelta(minutes=10))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(InvalidInputError, match="too large in magnitude"):
                _prepare(obs, [1.7e308, 1.7e308], aggregation=aggregation)

    def test_refusals_are_value_errors(self):
        assert issubclass(InvalidInputError, ValueError)
