"""Shared finiteness checks for samples, sampling rates, shaft speeds and
expected fault frequencies.

``signal_validation`` is the one implementation the signal store, the asset
ledger, the raw decoder and the public diagnostic entry points call (issues
#82 and #83). These tests pin what every caller relies on: which inputs are
refused, what the refusal says (count, total, first index, the caller's source
and remedy), and that the refusal is an ``InvalidInputError`` (a ``ValueError``)
raised from project code, so the refusal boundary forwards it verbatim.
"""

import math
import traceback
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.signal_validation import (
    BAD_SAMPLE_REMEDY,
    InvalidInputError,
    validate_expected_frequencies,
    validate_expected_frequency,
    validate_finite_samples,
    validate_rpm,
    validate_sampling_rate,
    validate_signal_and_fs,
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

    def test_several_text_tokens_report_the_count_and_the_first(self):
        data = np.array(["a", "1", "b"], dtype=object)

        with pytest.raises(ValueError) as excinfo:
            validate_finite_samples(data)

        message = str(excinfo.value)
        assert "2 non-numeric" in message
        assert "out of 3" in message
        assert "index 0" in message
        assert "'a'" in message

    def test_dates_are_not_samples(self):
        data = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match="holds datetime64"):
            validate_finite_samples(data)

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


# ---------------------------------------------------------------------------
# Shaft speed
# ---------------------------------------------------------------------------


class TestValidateRpm:
    @pytest.mark.parametrize(
        "value", [1797, 1797.0, np.float64(1797), np.int64(1797), 0.5]
    )
    def test_positive_finite_speeds_are_returned_as_float(self, value):
        result = validate_rpm(value)

        assert result == float(value)
        assert type(result) is float

    @pytest.mark.parametrize(
        "value", [0.0, -1.0, -1797, math.nan, math.inf, -math.inf, 10**400]
    )
    def test_non_positive_or_non_finite_speeds_are_refused(self, value):
        with pytest.raises(ValueError, match=r"rpm must be positive and finite"):
            validate_rpm(value)

    @pytest.mark.parametrize("value", [True, np.bool_(True), "1500", None])
    def test_values_that_are_not_numbers_are_refused(self, value):
        with pytest.raises(ValueError, match="not a number"):
            validate_rpm(value)

    def test_required_remedy_asks_for_the_measured_speed(self):
        with pytest.raises(ValueError) as excinfo:
            validate_rpm(0.0)

        message = str(excinfo.value)
        assert "measured shaft speed" in message
        assert "omit" not in message

    def test_optional_remedy_says_the_speed_can_be_omitted(self):
        with pytest.raises(ValueError) as excinfo:
            validate_rpm(0.0, optional=True)

        message = str(excinfo.value)
        assert "measured shaft speed" in message
        assert "omit rpm" in message

    def test_source_appears_in_the_message(self):
        with pytest.raises(ValueError, match=r"from the rpm parameter"):
            validate_rpm(math.nan, source="the rpm parameter")

    def test_refusal_is_raised_from_project_code(self):
        with pytest.raises(ValueError) as excinfo:
            validate_rpm(math.inf)

        assert _raised_in_src(excinfo.value)


# ---------------------------------------------------------------------------
# Expected fault frequency
# ---------------------------------------------------------------------------


class TestValidateExpectedFrequency:
    @pytest.mark.parametrize("value", [107.36, 162, np.float64(70.58)])
    def test_positive_finite_frequencies_are_returned_as_float(self, value):
        result = validate_expected_frequency(value, label="BPFO")

        assert result == float(value)
        assert type(result) is float

    @pytest.mark.parametrize("value", [0.0, -5, math.nan, math.inf, True, "107"])
    def test_unusable_frequencies_are_refused_naming_the_label(self, value):
        with pytest.raises(ValueError) as excinfo:
            validate_expected_frequency(value, label="BPFI")

        message = str(excinfo.value)
        assert "expected BPFI frequency must be positive and finite" in message
        assert "Hz" in message

    def test_refusal_is_raised_from_project_code(self):
        with pytest.raises(ValueError) as excinfo:
            validate_expected_frequency(math.nan, label="FTF")

        assert _raised_in_src(excinfo.value)


class TestValidateExpectedFrequencies:
    def test_valid_mapping_passes(self):
        validate_expected_frequencies({"BPFO": 107.36, "GMF": 800})

    def test_every_bad_label_is_listed_in_one_message(self):
        with pytest.raises(ValueError) as excinfo:
            validate_expected_frequencies(
                {"BPFO": 107.4, "BPFI": math.nan, "BSF": -1.0}, name="bearing_freqs"
            )

        message = str(excinfo.value)
        assert message.startswith("bearing_freqs contains non-positive or non-finite")
        listed = message.split(":", 1)[1].split("—")[0]
        assert "BPFI" in listed
        assert "BSF" in listed
        assert "BPFO" not in listed

    @pytest.mark.parametrize("value", [0.0, math.inf, True, "107.4", None])
    def test_each_unusable_value_is_refused(self, value):
        with pytest.raises(InvalidInputError, match="non-positive"):
            validate_expected_frequencies({"X": value})


# ---------------------------------------------------------------------------
# Refusal type
# ---------------------------------------------------------------------------


class TestRefusalType:
    """Every refusal of the module is an InvalidInputError, which is a
    ValueError: existing handlers and the refusal boundary see a ValueError,
    and a pipeline can tell a validator refusal from other ValueErrors."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: validate_finite_samples([np.nan]),
            lambda: validate_finite_samples(np.array(["x"], dtype=object)),
            lambda: validate_finite_samples(np.array(["2026-01-01"], dtype="M8[ns]")),
            lambda: validate_sampling_rate(0),
            lambda: validate_signal_and_fs([1.0, 2.0], math.nan),
            lambda: validate_rpm(-1.0),
            lambda: validate_expected_frequency(0.0, label="BPFO"),
        ],
        ids=[
            "non-finite sample",
            "non-numeric sample",
            "non-sample dtype",
            "sampling rate",
            "fs",
            "rpm",
            "expected frequency",
        ],
    )
    def test_every_refusal_is_an_invalid_input_error(self, call):
        with pytest.raises(InvalidInputError) as excinfo:
            call()

        assert isinstance(excinfo.value, ValueError)
