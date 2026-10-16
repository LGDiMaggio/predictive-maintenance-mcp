"""Shared finiteness checks for vibration samples and sampling rates.

``signal_validation`` is the one implementation the signal store, the asset
ledger, the raw decoder and the public diagnostic entry points call (issue
#82). These tests pin what every caller relies on: which inputs are refused,
what the refusal says (count, total, first index, the caller's source and
remedy), and that the refusal is a plain ``ValueError`` raised from project
code, so the refusal boundary forwards it verbatim.
"""

import math
import traceback
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.signal_validation import (
    BAD_SAMPLE_REMEDY,
    validate_finite_samples,
    validate_sampling_rate,
)

SRC_DIR = (Path(__file__).resolve().parent.parent / "src").resolve()


def _raised_in_src(exc: BaseException) -> bool:
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and Path(frames[-1].filename).resolve().is_relative_to(SRC_DIR)


# ---------------------------------------------------------------------------
# Sampling rate
# ---------------------------------------------------------------------------


class TestValidateSamplingRate:
    @pytest.mark.parametrize("value", [10000, 25600.0, np.float64(12800)])
    def test_positive_finite_rates_are_returned_as_float(self, value):
        result = validate_sampling_rate(value)

        assert result == float(value)
        assert type(result) is float

    @pytest.mark.parametrize("value", [0, -5, math.nan, math.inf, -math.inf])
    def test_non_positive_or_non_finite_rates_are_refused(self, value):
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            validate_sampling_rate(value)

    @pytest.mark.parametrize("value", [True, np.bool_(True), "10000", [10000]])
    def test_values_that_are_not_real_numbers_are_refused(self, value):
        with pytest.raises(ValueError, match="not a number"):
            validate_sampling_rate(value)

    def test_integer_too_large_for_a_float_is_refused_as_a_value_error(self):
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            validate_sampling_rate(10**400)

    def test_name_shapes_the_message(self):
        with pytest.raises(ValueError, match=r"fs > 0"):
            validate_sampling_rate(math.inf, name="fs")

    def test_source_and_remedy_appear_in_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            validate_sampling_rate(
                math.nan,
                source="pump_metadata.json",
                remedy="fix the companion's sampling_rate",
            )

        message = str(excinfo.value)
        assert "pump_metadata.json" in message
        assert "fix the companion's sampling_rate" in message

    def test_refusal_is_raised_from_project_code(self):
        with pytest.raises(ValueError) as excinfo:
            validate_sampling_rate(math.nan)

        assert _raised_in_src(excinfo.value)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


class TestValidateFiniteSamples:
    def test_finite_float_array_passes_without_a_copy(self):
        data = np.sin(np.linspace(0.0, 10.0, 10000))

        result = validate_finite_samples(data)

        assert result is data

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.bool_])
    def test_integer_and_boolean_arrays_pass(self, dtype):
        data = np.ones(64, dtype=dtype)

        assert validate_finite_samples(data) is data

    def test_one_nan_is_refused_with_count_total_and_index(self):
        data = np.zeros(10000)
        data[100] = np.nan

        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        message = str(excinfo.value)
        assert "1 non-finite" in message
        assert "10000" in message
        assert "index 100" in message

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_infinities_are_refused(self, bad):
        data = np.zeros(50)
        data[7] = bad

        with pytest.raises(ValueError, match=r"1 non-finite.*index 7"):
            validate_finite_samples(data)

    def test_all_nan_reports_full_count_and_index_zero(self):
        with pytest.raises(ValueError, match=r"5 non-finite.*out of 5.*index 0"):
            validate_finite_samples(np.full(5, np.nan))

    def test_masked_nan_is_judged_by_its_data(self):
        data = np.ma.masked_array([1.0, np.nan, 3.0], mask=[False, True, False])

        with pytest.raises(ValueError, match="1 non-finite"):
            validate_finite_samples(data)

    def test_two_dimensional_array_reports_flat_index_and_shape(self):
        data = np.zeros((3, 4))
        data[1, 2] = np.nan

        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        message = str(excinfo.value)
        assert "index 6" in message
        assert "(3, 4)" in message

    def test_complex_nan_is_refused(self):
        data = np.array([1 + 1j, complex(np.nan, 0.0), 2 + 0j])

        with pytest.raises(ValueError, match="1 non-finite"):
            validate_finite_samples(data)

    def test_plain_list_with_nan_is_refused(self):
        with pytest.raises(ValueError, match="1 non-finite"):
            validate_finite_samples([1.0, float("nan"), 2.0])

    def test_header_row_column_reports_only_the_header(self):
        """A header row makes the CSV decoder return every value as text."""
        data = np.array(["accel", "1.0", "2.0"], dtype=object)

        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        message = str(excinfo.value)
        assert "1 non-numeric" in message
        assert "out of 3" in message
        assert "index 0" in message
        assert "'accel'" in message

    def test_marker_token_is_located_among_text_values(self):
        data = np.array(["1.0", "2.0", "ERR", "3.5"], dtype=object)

        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        message = str(excinfo.value)
        assert "1 non-numeric" in message
        assert "out of 4" in message
        assert "index 2" in message
        assert "'ERR'" in message

    def test_text_values_that_all_parse_are_returned_as_float(self):
        result = validate_finite_samples(np.array(["1.0", "2.0"], dtype=object))

        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_number_too_large_for_a_float_is_refused_as_non_finite(self):
        with pytest.raises(ValueError, match=r"1 non-finite.*index 1"):
            validate_finite_samples(np.array([1.0, 10**400], dtype=object))

    def test_text_nan_token_is_refused_as_non_finite(self):
        with pytest.raises(ValueError, match="1 non-finite"):
            validate_finite_samples(np.array(["1.0", "nan"], dtype=object))

    def test_default_remedy_rules_out_filling_samples(self):
        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples([np.nan])

        message = str(excinfo.value)
        assert BAD_SAMPLE_REMEDY in message
        assert "re-export" in BAD_SAMPLE_REMEDY
        assert "interpolating" in BAD_SAMPLE_REMEDY

    def test_source_and_remedy_appear_in_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(
                [np.nan], source="File 'pump.csv'", remedy="re-check the export"
            )

        message = str(excinfo.value)
        assert message.startswith("File 'pump.csv'")
        assert "re-check the export" in message

    @pytest.mark.parametrize(
        "data",
        [[np.nan], np.array(["x"], dtype=object)],
        ids=["non-finite", "non-numeric"],
    )
    def test_refusal_is_raised_from_project_code(self, data):
        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        assert _raised_in_src(excinfo.value)
