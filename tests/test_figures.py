"""Tests for the explanatory figure builder.

The annotated envelope spectrum is not decoration. It is the place a reader
*sees* that the dominant peak falls inside one characteristic-frequency band
and outside the other three — the same tolerance the verdict used, drawn.

A figure that only carries its argument when you can hover it is a figure that
loses its argument in the PDF rendering, so these tests assert the annotation
is in the figure data itself.
"""

import numpy as np
import pytest

from predictive_maintenance_mcp.figures import (
    ANNOTATED_ENVELOPE,
    build_annotated_envelope_figure,
)

BEARING_FREQS = {
    "BPFO": 81.125,
    "BPFI": 118.875,
    "BSF": 63.91,
    "FTF": 14.84,
    "shaft_freq_hz": 25.0,
}


def _spectrum(peak_hz=80.5, span_hz=500.0, n=2000):
    """A synthetic envelope spectrum with one dominant peak."""
    freqs = np.linspace(0.0, span_hz, n)
    mags = 0.001 + 0.05 * np.exp(-(((freqs - peak_hz) / 0.8) ** 2))
    return freqs, mags


class TestCharacteristicFrequencyBands:
    def test_one_band_per_characteristic_frequency(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        labels = [band["label"] for band in figure["bands"]]
        assert labels == ["FTF", "BSF", "BPFO", "BPFI"]

    def test_shaft_frequency_is_not_drawn_as_a_fault_band(self):
        """shaft_freq_hz rides along in the same dict but is not a defect."""
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        assert all(band["label"] != "shaft_freq_hz" for band in figure["bands"])

    def test_band_width_matches_the_matching_tolerance(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, BEARING_FREQS, tolerance_pct=5.0
        )
        bpfo = next(b for b in figure["bands"] if b["label"] == "BPFO")
        expected_width = 81.125 * 2 * 0.05
        assert bpfo["high_hz"] - bpfo["low_hz"] == pytest.approx(expected_width)
        assert bpfo["center_hz"] == pytest.approx(81.125)

    def test_a_different_tolerance_produces_a_different_band_width(self):
        freqs, mags = _spectrum()
        narrow = build_annotated_envelope_figure(
            freqs, mags, BEARING_FREQS, tolerance_pct=2.0
        )
        band = next(b for b in narrow["bands"] if b["label"] == "BPFO")
        assert band["high_hz"] - band["low_hz"] == pytest.approx(81.125 * 2 * 0.02)

    def test_characteristic_frequency_beyond_the_axis_is_omitted(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, BEARING_FREQS, max_freq=100.0
        )
        labels = [band["label"] for band in figure["bands"]]
        assert "BPFI" not in labels, "118.875 Hz cannot be drawn on a 100 Hz axis"
        assert "BPFO" in labels
        assert figure["omitted_labels"] == ["BPFI"]


class TestMatchAnnotation:
    def test_matched_annotation_carries_calculated_and_measured_values(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs,
            mags,
            BEARING_FREQS,
            matched={"fault_type": "BPFO", "measured_hz": 80.5, "deviation_pct": 0.77},
        )
        annotation = next(a for a in figure["annotations"] if a["matched"])
        assert "81.12" in annotation["text"] or "81.13" in annotation["text"]
        assert "80.5" in annotation["text"]
        assert "BPFO" in annotation["text"]

    def test_only_the_matched_band_is_flagged_as_matched(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, BEARING_FREQS, matched={"fault_type": "BPFO", "measured_hz": 80.5}
        )
        matched = [b for b in figure["bands"] if b["matched"]]
        assert [b["label"] for b in matched] == ["BPFO"]

    def test_no_match_leaves_every_band_unmatched(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        assert all(not band["matched"] for band in figure["bands"])
        assert all(not ann["matched"] for ann in figure["annotations"])


class TestDegradation:
    def test_absent_bearing_metadata_still_produces_a_figure(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, None)
        assert figure["kind"] == ANNOTATED_ENVELOPE
        assert figure["bands"] == []
        assert figure["annotations"] == []
        assert len(figure["x"]) > 0

    def test_caption_states_what_the_figure_shows_in_both_cases(self):
        freqs, mags = _spectrum()
        with_meta = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        without = build_annotated_envelope_figure(freqs, mags, None)
        assert "tolerance" in with_meta["caption"].lower()
        assert without["caption"]
        assert "no bearing" in without["caption"].lower()

    def test_empty_spectrum_raises_rather_than_drawing_nothing(self):
        with pytest.raises(ValueError):
            build_annotated_envelope_figure(np.array([]), np.array([]), None)

    def test_mismatched_array_lengths_raise(self):
        with pytest.raises(ValueError):
            build_annotated_envelope_figure(
                np.linspace(0, 100, 50), np.zeros(20), None
            )


class TestAxesAndScaling:
    def test_magnitudes_are_normalised_to_decibels_below_the_peak(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        assert max(figure["y"]) == pytest.approx(0.0, abs=1e-9)
        assert min(figure["y"]) < 0.0
        assert "dB" in figure["y_label"]

    def test_axis_extends_far_enough_to_show_every_characteristic_band(self):
        freqs, mags = _spectrum(span_hz=1000.0)
        figure = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        assert max(figure["x"]) >= 118.875 * 1.05

    def test_figure_data_is_json_serialisable(self):
        """Both renderings serialise this; numpy scalars would break one."""
        import json

        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, BEARING_FREQS, matched={"fault_type": "BPFO", "measured_hz": 80.5}
        )
        json.dumps(figure)

    def test_figure_is_deterministic_for_the_same_input(self):
        freqs, mags = _spectrum()
        first = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        second = build_annotated_envelope_figure(freqs, mags, BEARING_FREQS)
        assert first == second
