"""Tests for the explanatory figure builder.

The annotated envelope spectrum is where a reader sees the search windows
the bearing check tested: one window around the line each hypothesis is
tested at (twice the ball spin frequency for the rolling element), drawn at
the tolerance the check used. Since T1 PR C a window is emphasised only for a
candidate line of the check's claim; a peak inside a window is not a finding
by itself, and the caption says so.

A figure that only carries its argument when you can hover it is a figure that
loses its argument in the PDF rendering, so these tests assert the annotation
is in the figure data itself.
"""

import numpy as np
import pytest

from _bearing_scenarios import DEFECT_SEED, FREQUENCIES_6205, RPM_6205, defect
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
)
from predictive_maintenance_mcp.figures import (
    ANNOTATED_ENVELOPE,
    build_annotated_envelope_figure,
)
from predictive_maintenance_mcp.signal_processing.spectral import (
    envelope_spectrum_arrays,
    resolve_envelope_band,
)

FREQUENCIES = {"BPFO": 81.125, "BPFI": 118.875, "BSF": 63.91, "FTF": 14.84}
CANONICAL = {"BPFO": "outer_race", "BPFI": "inner_race", "BSF": "ball", "FTF": "cage"}


def _summary(supported=(), level=None, conditions=None) -> dict:
    """The parts of a check summary the figure reads.

    Each check's line is tested at its declared frequency, twice it for the
    rolling element; a supported line peaks 0.5 % below its line. The claim
    level follows the candidate count unless given.
    """
    checks = []
    for label, declared in FREQUENCIES.items():
        line_hz = 2.0 * declared if label == "BSF" else declared
        state = "supported" if label in supported else "absent"
        checks.append(
            {
                "fault_type": label,
                "fault_type_canonical": CANONICAL[label],
                "state": state,
                "line_frequency_hz": line_hz,
                "line_peak_hz": round(0.995 * line_hz, 3),
                "level_db": 21.5 if state == "supported" else 4.0,
            }
        )
    candidates = [
        {
            "fault_type": c["fault_type"],
            "fault_type_canonical": c["fault_type_canonical"],
            "line_frequency_hz": c["line_frequency_hz"],
            "line_peak_hz": c["line_peak_hz"],
            "level_db": c["level_db"],
        }
        for c in checks
        if c["state"] == "supported"
    ]
    if level is None:
        level = (
            "single_spectrum_evidence"
            if len(candidates) == 1
            else "qualified" if candidates else "withheld"
        )
    if conditions is None:
        conditions = (
            ["several_types"]
            if len(candidates) > 1
            else [] if candidates else ["no_indication"]
        )
    return {
        "fault_checks": checks,
        "candidates": candidates,
        "claim": {"level": level, "conditions": conditions, "limitations": []},
    }


def _spectrum(peak_hz=80.5, span_hz=500.0, n=2000):
    """A synthetic envelope spectrum with one dominant peak."""
    freqs = np.linspace(0.0, span_hz, n)
    mags = 0.001 + 0.05 * np.exp(-(((freqs - peak_hz) / 0.8) ** 2))
    return freqs, mags


class TestSearchWindows:
    def test_one_window_per_tested_line_in_frequency_order(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        labels = [band["label"] for band in figure["bands"]]
        # The rolling element's window sits at 2 x BSF = 127.82 Hz.
        assert labels == ["FTF", "BPFO", "BPFI", "BSF"]

    def test_rolling_element_window_and_annotation_sit_at_twice_bsf(self):
        freqs, mags = _spectrum(span_hz=1000.0)
        figure = build_annotated_envelope_figure(freqs, mags, _summary(("BSF",)))
        band = next(b for b in figure["bands"] if b["label"] == "BSF")
        annotation = next(a for a in figure["annotations"] if a["label"] == "BSF")

        assert band["center_hz"] == pytest.approx(2 * 63.91)
        assert annotation["x"] == pytest.approx(2 * 63.91)
        assert annotation["text"].startswith("2 x BSF 127.82 Hz")
        assert "(2 x BSF for the rolling element)" in figure["caption"]

    def test_window_width_is_the_search_tolerance_of_the_tested_line(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, _summary(), tolerance_pct=5.0
        )
        bpfo = next(b for b in figure["bands"] if b["label"] == "BPFO")
        assert bpfo["high_hz"] - bpfo["low_hz"] == pytest.approx(81.125 * 2 * 0.05)
        assert bpfo["center_hz"] == pytest.approx(81.125)

    def test_a_different_tolerance_produces_a_different_window_width(self):
        freqs, mags = _spectrum()
        narrow = build_annotated_envelope_figure(
            freqs, mags, _summary(), tolerance_pct=2.0
        )
        band = next(b for b in narrow["bands"] if b["label"] == "BPFO")
        assert band["high_hz"] - band["low_hz"] == pytest.approx(81.125 * 2 * 0.02)

    def test_line_beyond_the_axis_is_omitted(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, _summary(), max_freq=100.0
        )
        labels = [band["label"] for band in figure["bands"]]
        assert "BPFI" not in labels, "118.875 Hz cannot be drawn on a 100 Hz axis"
        assert "BPFO" in labels
        assert figure["omitted_labels"] == ["BPFI", "BSF"]

    def test_rolling_element_window_of_an_engine_check_sits_at_twice_bsf(self):
        """A rolling-element defect, 10 s at 12 kHz and -10 dB: the window the
        figure emphasises is the line the engine tested, 2 x BSF."""
        fs = 12000.0
        signal = defect(
            "rolling_element",
            np.random.default_rng((DEFECT_SEED, 43, 0)),
            fs,
            10.0,
            FREQUENCIES_6205,
            RPM_6205 / 60.0,
            -10.0,
        )
        check = check_all_bearing_faults(signal, fs, "6205", RPM_6205)
        freqs, mags = envelope_spectrum_arrays(
            signal, fs, resolve_envelope_band(fs, None)
        )
        figure = build_annotated_envelope_figure(freqs, mags, check)
        bsf = next(c for c in check["fault_checks"] if c["fault_type"] == "BSF")
        band = next(b for b in figure["bands"] if b["label"] == "BSF")
        annotation = next(a for a in figure["annotations"] if a["label"] == "BSF")

        assert [c["fault_type"] for c in check["candidates"]] == ["BSF"]
        assert band["center_hz"] == pytest.approx(bsf["line_frequency_hz"])
        assert bsf["line_frequency_hz"] == pytest.approx(
            2 * bsf["expected_frequency_hz"], abs=0.02
        )
        assert band["emphasised"] and annotation["emphasised"]
        assert annotation["x"] == pytest.approx(bsf["line_frequency_hz"])
        assert f"strongest bin {bsf['line_peak_hz']:.1f} Hz" in annotation["text"]


class TestEmphasis:
    def test_single_spectrum_evidence_emphasises_its_candidate(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, _summary(("BPFO",)))
        annotation = next(a for a in figure["annotations"] if a["emphasised"])

        assert [b["label"] for b in figure["bands"] if b["emphasised"]] == ["BPFO"]
        assert annotation["text"] == (
            "BPFO 81.12 Hz: strongest bin 80.7 Hz, 21.5 dB above the floor"
        )

    def test_several_types_emphasise_every_candidate_alike(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(
            freqs, mags, _summary(("BPFO", "BPFI"))
        )
        emphasised = [b["label"] for b in figure["bands"] if b["emphasised"]]
        annotated = {
            a["label"]: a["text"] for a in figure["annotations"] if a["emphasised"]
        }

        assert emphasised == ["BPFO", "BPFI"]
        assert set(annotated) == {"BPFO", "BPFI"}
        assert all("strongest bin" in text for text in annotated.values())
        assert figure["caption"].endswith(
            "The emphasised windows (BPFO and BPFI) hold the candidate lines of the "
            "bearing statement in this report, emphasised alike."
        )

    def test_withheld_claim_emphasises_nothing_even_with_candidates(self):
        """A generated signal without noise keeps its candidates in the
        evidence, but the claim is withheld, so no window is emphasised."""
        freqs, mags = _spectrum()
        summary = _summary(
            ("BPFO",), level="withheld", conditions=["generated_without_noise"]
        )
        figure = build_annotated_envelope_figure(freqs, mags, summary)

        assert not any(b["emphasised"] for b in figure["bands"])
        assert not any(a["emphasised"] for a in figure["annotations"])
        assert figure["caption"].endswith(
            "No window is emphasised: the bearing statement in this report names "
            "no candidate line."
        )

    def test_no_candidate_leaves_every_window_plain(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        assert all(not band["emphasised"] for band in figure["bands"])
        assert all(not ann["emphasised"] for ann in figure["annotations"])


class TestCaption:
    def test_caption_describes_search_windows_not_matches(self):
        freqs, mags = _spectrum()
        caption = build_annotated_envelope_figure(freqs, mags, _summary(("BPFO",)))[
            "caption"
        ]

        assert caption == (
            "Envelope spectrum with the FTF, BPFO, BPFI and BSF search windows "
            "overlaid. Each shaded band is the ±5% window around the line its "
            "check tests (2 x BSF for the rolling element). A peak inside a window "
            "is not a finding by itself: the check counts a line only when it "
            "stands above the local noise floor and a further line it predicts "
            "does too. The emphasised window (BPFO) holds the candidate line of "
            "the bearing statement in this report."
        )
        assert "is a match" not in caption


class TestDegradation:
    def test_absent_bearing_check_still_produces_a_figure(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, None)
        assert figure["kind"] == ANNOTATED_ENVELOPE
        assert figure["bands"] == []
        assert figure["annotations"] == []
        assert len(figure["x"]) > 0

    def test_caption_states_what_the_figure_shows_in_both_cases(self):
        freqs, mags = _spectrum()
        with_check = build_annotated_envelope_figure(freqs, mags, _summary())
        without = build_annotated_envelope_figure(freqs, mags, None)
        assert "search windows" in with_check["caption"]
        assert without["caption"]
        assert "no bearing" in without["caption"].lower()

    def test_caption_gives_the_reason_when_a_bearing_was_named(self):
        """Issue #84: a designation was supplied but is not in the catalog;
        the caption must not say that none was supplied."""
        freqs, mags = _spectrum()
        reason = "Bearing '9999' is not in the bearing catalog, so ..."
        figure = build_annotated_envelope_figure(
            freqs, mags, None, no_bands_reason=reason
        )
        assert figure["bands"] == []
        assert figure["caption"].startswith(f"Envelope spectrum. {reason} ")
        assert "no bearing designation" not in figure["caption"].lower()

    def test_empty_spectrum_raises_rather_than_drawing_nothing(self):
        with pytest.raises(ValueError):
            build_annotated_envelope_figure(np.array([]), np.array([]), None)

    def test_mismatched_array_lengths_raise(self):
        with pytest.raises(ValueError):
            build_annotated_envelope_figure(np.linspace(0, 100, 50), np.zeros(20), None)


class TestAxesAndScaling:
    def test_magnitudes_are_normalised_to_decibels_below_the_peak(self):
        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        assert max(figure["y"]) == pytest.approx(0.0, abs=1e-9)
        assert min(figure["y"]) < 0.0
        assert "dB" in figure["y_label"]

    def test_axis_extends_far_enough_to_show_every_window(self):
        freqs, mags = _spectrum(span_hz=1000.0)
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        assert max(figure["x"]) >= 2 * 63.91 * 1.05

    def test_figure_data_is_json_serialisable(self):
        """Both renderings serialise this; numpy scalars would break one."""
        import json

        freqs, mags = _spectrum()
        figure = build_annotated_envelope_figure(freqs, mags, _summary(("BPFO",)))
        json.dumps(figure)

    def test_figure_is_deterministic_for_the_same_input(self):
        freqs, mags = _spectrum()
        first = build_annotated_envelope_figure(freqs, mags, _summary(("BPFO",)))
        second = build_annotated_envelope_figure(freqs, mags, _summary(("BPFO",)))
        assert first == second


class TestTraceDecimation:
    def test_long_spectrum_is_capped(self):
        freqs, mags = _spectrum(n=200_000)
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        assert len(figure["x"]) <= 2400
        assert len(figure["x"]) == len(figure["y"])

    def test_decimation_keeps_the_peak_rather_than_dropping_it(self):
        """Plain decimation drops exactly the narrow peaks the figure exists
        to show. The dominant peak must survive at its own frequency."""
        freqs, mags = _spectrum(peak_hz=80.5, n=200_000)
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        peak_index = max(range(len(figure["y"])), key=lambda i: figure["y"][i])
        assert figure["x"][peak_index] == pytest.approx(80.5, abs=0.5)
        assert max(figure["y"]) == pytest.approx(0.0, abs=1e-9)

    def test_short_spectrum_is_untouched(self):
        freqs, mags = _spectrum(n=500)
        figure = build_annotated_envelope_figure(freqs, mags, _summary())
        capped = build_annotated_envelope_figure(
            freqs, mags, _summary(), max_points=10_000
        )
        assert figure["x"] == capped["x"]

    def test_decimation_is_deterministic(self):
        freqs, mags = _spectrum(n=200_000)
        first = build_annotated_envelope_figure(freqs, mags, _summary())
        second = build_annotated_envelope_figure(freqs, mags, _summary())
        assert first == second
