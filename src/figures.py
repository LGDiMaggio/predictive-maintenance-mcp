"""Explanatory figures for diagnostic reports.

A figure earns its place in a report only if removing it would weaken a
conclusion. The annotated envelope spectrum earns its place: it is where a
reader sees the search window of each line the bearing check tested, at the
tolerance the check used, and which windows hold the candidate lines of the
check's claim.

The windows sit around each hypothesis's tested line (``line_frequency_hz``:
twice the ball spin frequency for the rolling element), and emphasis follows
the claim (:mod:`.diagnostics.bearing_claims`): every candidate alike at
single-spectrum evidence and at a qualified claim, none when the claim is
withheld. A peak inside a window is not a finding by itself, and the caption,
written from the same closed list of sentences as every bearing statement,
says so.

Figures are built as plain data — lists, floats, strings — rather than as
rendered output, for two reasons. Both the HTML and the PDF rendering consume
the same structure, so neither can quietly show something the other does not;
and the annotation lives in the figure data rather than in a hover interaction,
so the argument survives a static export.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .diagnostics.bearing_claims import (
    CLAIM_SENTENCES,
    emphasised_candidates,
    line_display_name,
)

#: Figure kind discriminator, so a renderer can dispatch without guessing.
ANNOTATED_ENVELOPE = "annotated_envelope_spectrum"

#: Default upper edge of the frequency axis. Bearing defect frequencies for
#: industrial rolling-element bearings sit well below this; the axis extends
#: automatically when a tested line would otherwise fall outside.
_DEFAULT_MAX_FREQ_HZ = 500.0

_FLOOR_DB = 1e-12

#: Decimal places kept on band edges. Coarser rounding makes the drawn band
#: width diverge from the tolerance it claims to represent.
_BAND_PRECISION = 6

#: Cap on plotted trace points, so a long high-rate acquisition does not
#: produce a report too heavy to send.
_MAX_TRACE_POINTS = 2400


def _decimate_preserving_peaks(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the trace to at most ``max_points`` without losing peaks.

    Plain decimation would drop samples, and in a spectrum the samples it
    drops are exactly the narrow peaks the whole figure exists to show. This
    keeps the maximum within each bucket instead, so a peak survives at the
    frequency where it occurred.
    """
    if max_points <= 0 or x.size <= max_points:
        return x, y

    bucket = int(np.ceil(x.size / max_points))
    usable = (x.size // bucket) * bucket
    head_y = y[:usable].reshape(-1, bucket)
    head_x = x[:usable].reshape(-1, bucket)
    peak_offsets = np.argmax(head_y, axis=1)
    rows = np.arange(head_y.shape[0])

    kept_x = head_x[rows, peak_offsets]
    kept_y = head_y[rows, peak_offsets]

    if usable < x.size:
        tail_index = usable + int(np.argmax(y[usable:]))
        kept_x = np.append(kept_x, x[tail_index])
        kept_y = np.append(kept_y, y[tail_index])

    return kept_x, kept_y


def build_annotated_envelope_figure(
    env_frequencies: Sequence[float] | np.ndarray,
    env_magnitudes: Sequence[float] | np.ndarray,
    bearing_check: Optional[Mapping[str, Any]] = None,
    tolerance_pct: float = 5.0,
    max_freq: Optional[float] = None,
    max_points: int = _MAX_TRACE_POINTS,
    no_bands_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Build the annotated envelope spectrum as renderer-agnostic data.

    Args:
        env_frequencies: Envelope spectrum frequency axis (Hz).
        env_magnitudes: Envelope spectrum magnitudes, linear scale.
        bearing_check: The bearing check summary (``fault_checks`` with each
            line's ``line_frequency_hz``, ``candidates`` and ``claim``). One
            search window is drawn per tested line; the windows of the
            candidates the claim lets a report mark are emphasised alike.
            ``None`` produces a plain spectrum with a caption saying why it
            carries no windows.
        tolerance_pct: The check's search tolerance, drawn as the window
            half-width in percent of the tested line. It must be the
            tolerance the check used, or the figure would show windows the
            check did not search.
        max_freq: Upper edge of the frequency axis. ``None`` extends the axis
            to cover every tested line.
        max_points: Cap on trace points. A long acquisition at a high
            sampling rate resolves the plotted band far more finely than any
            screen or page can show, and every extra point is bytes in a
            document meant to be emailed.
        no_bands_reason: The sentence saying why no windows are drawn when a
            bearing was named but its frequencies could not be computed.
            ``None`` keeps the "no bearing designation was supplied"
            caption, which would be false for a named bearing.

    Returns:
        A JSON-serialisable figure description: axis data, search windows
        (``bands``), annotations, the labels that fell outside the axis, and
        a caption.

    Raises:
        ValueError: If the spectrum is empty or the two arrays disagree in
            length — a figure drawn from a malformed spectrum would look
            exactly like a figure drawn from a real one.
    """
    freqs = np.asarray(env_frequencies, dtype=float)
    mags = np.asarray(env_magnitudes, dtype=float)

    if freqs.size == 0 or mags.size == 0:
        raise ValueError(
            "Cannot build an envelope figure from an empty spectrum — pass "
            "the frequency and magnitude arrays from compute_envelope_spectrum."
        )
    if freqs.size != mags.size:
        raise ValueError(
            f"Envelope frequency and magnitude arrays disagree in length "
            f"({freqs.size} vs {mags.size}); they must come from the same "
            f"spectrum computation."
        )

    lines = _tested_lines(bearing_check)
    axis_max = _resolve_axis_max(
        freqs, [line["line_frequency_hz"] for line in lines], max_freq
    )

    mask = freqs <= axis_max
    x, y_linear = _decimate_preserving_peaks(freqs[mask], mags[mask], max_points)
    peak = float(np.max(y_linear))
    if peak <= 0.0:
        y_db = np.zeros_like(y_linear)
    else:
        y_db = 20.0 * np.log10((y_linear + _FLOOR_DB) / peak)

    marked = (
        [c["fault_type"] for c in emphasised_candidates(bearing_check)]
        if bearing_check
        else []
    )
    bands: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    omitted: list[str] = []
    rolling_element = False

    for check in lines:
        label = check["fault_type"]
        center = float(check["line_frequency_hz"])
        if center > axis_max:
            omitted.append(label)
            continue
        canonical = check.get("fault_type_canonical")
        name = line_display_name(label, canonical)
        # The rolling element's window sits at 2 x BSF; the caption says so.
        rolling_element = rolling_element or name != label
        half = center * tolerance_pct / 100.0
        emphasised = label in marked
        bands.append(
            {
                "label": label,
                "name": name,
                "center_hz": round(center, _BAND_PRECISION),
                # Rounded at a precision fine enough that the band edges still
                # span exactly 2x the tolerance — rounding the two edges more
                # coarsely distorts the width the reader is asked to trust.
                "low_hz": round(center - half, _BAND_PRECISION),
                "high_hz": round(center + half, _BAND_PRECISION),
                "emphasised": emphasised,
            }
        )
        annotations.append(
            {
                "label": label,
                "x": round(center, _BAND_PRECISION),
                "y": _local_peak_db(x, y_db, center - half, center + half),
                "text": _annotation_text(name, center, check if emphasised else None),
                "emphasised": emphasised,
            }
        )

    drawn = {band["label"]: band["name"] for band in bands}
    return {
        "kind": ANNOTATED_ENVELOPE,
        "x": [round(float(v), 4) for v in x],
        "y": [round(float(v), 4) for v in y_db],
        "x_label": "Frequency (Hz)",
        "y_label": "Envelope amplitude (dB relative to peak)",
        "tolerance_pct": tolerance_pct,
        "bands": bands,
        "annotations": annotations,
        "omitted_labels": omitted,
        "caption": _caption(
            bands,
            omitted,
            tolerance_pct,
            [drawn.get(label, label) for label in marked],
            rolling_element,
            no_bands_reason,
        ),
    }


def figure_to_svg(figure: dict[str, Any], width: int = 900, height: int = 420) -> str:
    """Render a figure description as a standalone inline SVG.

    SVG rather than a charting library because the report must open with no
    network access and must survive export to a static format. A chart that
    needs a CDN script is not a self-contained document, and a chart whose
    labels only appear on hover loses its argument the moment it is printed.

    Args:
        figure: A description from :func:`build_annotated_envelope_figure`.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        An ``<svg>`` element with no external references.

    Raises:
        ValueError: If the figure is not a kind this renderer knows.
    """
    if figure.get("kind") != ANNOTATED_ENVELOPE:
        raise ValueError(
            f"Unknown figure kind {figure.get('kind')!r} — this renderer "
            f"draws {ANNOTATED_ENVELOPE} only."
        )

    left, right, top, bottom = 70, 20, 34, 56
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = figure["x"]
    ys = figure["y"]
    x_max = max(xs) if xs else 1.0
    y_min = min(min(ys), -6.0) if ys else -60.0

    def px(value: float) -> float:
        return round(left + (value / x_max) * plot_w, 2) if x_max else left

    def py(value: float) -> float:
        span = -y_min or 1.0
        return round(top + ((0.0 - value) / span) * plot_h, 2)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="{_esc(figure["caption"])}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    # Horizontal gridlines every 10 dB, so amplitude differences are readable
    # off the page rather than only by hovering.
    tick = -10.0
    while tick > y_min:
        y_pos = py(tick)
        parts.append(
            f'<line x1="{left}" y1="{y_pos}" x2="{left + plot_w}" y2="{y_pos}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y_pos + 4}" font-size="11" fill="#7f8c8d" '
            f'text-anchor="end">{tick:.0f}</text>'
        )
        tick -= 10.0

    # Search windows behind the trace; the candidates' windows are emphasised
    # alike.
    for band in figure["bands"]:
        x_low, x_high = px(band["low_hz"]), px(band["high_hz"])
        band_w = max(x_high - x_low, 1.5)
        fill = "#e74c3c" if band["emphasised"] else "#95a5a6"
        opacity = "0.28" if band["emphasised"] else "0.14"
        parts.append(
            f'<rect x="{x_low}" y="{top}" width="{round(band_w, 2)}" '
            f'height="{plot_h}" fill="{fill}" fill-opacity="{opacity}"/>'
        )
        centre = px(band["center_hz"])
        parts.append(
            f'<line x1="{centre}" y1="{top}" x2="{centre}" y2="{top + plot_h}" '
            f'stroke="{fill}" stroke-width="1" stroke-dasharray="4 3"/>'
        )

    points = " ".join(f"{px(x)},{py(y)}" for x, y in zip(xs, ys))
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="#2c3e50" '
        f'stroke-width="1.2"/>'
    )

    # Axes
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#2c3e50" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#2c3e50" stroke-width="1.5"/>'
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = x_max * fraction
        x_pos = px(value)
        parts.append(
            f'<line x1="{x_pos}" y1="{top + plot_h}" x2="{x_pos}" '
            f'y2="{top + plot_h + 5}" stroke="#2c3e50" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos}" y="{top + plot_h + 19}" font-size="11" '
            f'fill="#7f8c8d" text-anchor="middle">{value:.0f}</text>'
        )

    # Annotations, staggered vertically so adjacent bands do not collide.
    for index, annotation in enumerate(figure["annotations"]):
        x_pos = px(annotation["x"])
        y_pos = max(py(annotation["y"]) - 10, top + 12 + (index % 2) * 14)
        weight = "700" if annotation["emphasised"] else "500"
        colour = "#c0392b" if annotation["emphasised"] else "#5d6d7e"
        anchor = "start" if x_pos < left + plot_w * 0.75 else "end"
        parts.append(
            f'<text x="{round(x_pos + (4 if anchor == "start" else -4), 2)}" '
            f'y="{y_pos}" font-size="11" font-weight="{weight}" fill="{colour}" '
            f'text-anchor="{anchor}">{_esc(annotation["text"])}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 8}" font-size="12" '
        f'fill="#2c3e50" text-anchor="middle">{_esc(figure["x_label"])}</text>'
    )
    parts.append(
        f'<text x="14" y="{top + plot_h / 2}" font-size="12" fill="#2c3e50" '
        f'text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2})">'
        f'{_esc(figure["y_label"])}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _esc(text: str) -> str:
    """Escape text for embedding in SVG markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tested_lines(
    bearing_check: Optional[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The checks of a bearing check summary, by tested line, ascending."""
    if not bearing_check:
        return []
    return sorted(
        bearing_check["fault_checks"], key=lambda check: check["line_frequency_hz"]
    )


def _resolve_axis_max(
    freqs: np.ndarray,
    line_frequencies: list[float],
    max_freq: Optional[float],
) -> float:
    """Choose the frequency axis upper edge."""
    spectrum_max = float(np.max(freqs))
    if max_freq is not None:
        return min(float(max_freq), spectrum_max)

    wanted = _DEFAULT_MAX_FREQ_HZ
    if line_frequencies:
        # Leave headroom past the highest window so it is not clipped at the
        # axis edge, where a reader cannot see which side the peak is on.
        wanted = max(wanted, max(line_frequencies) * 1.25)
    return min(wanted, spectrum_max)


def _local_peak_db(x: np.ndarray, y_db: np.ndarray, low: float, high: float) -> float:
    """Height at which to anchor a band's annotation."""
    window = (x >= low) & (x <= high)
    if not window.any():
        return round(float(np.min(y_db)), 4)
    return round(float(np.max(y_db[window])), 4)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _annotation_text(
    name: str, center: float, candidate: Optional[Mapping[str, Any]]
) -> str:
    """Annotation copy: carries the numbers, so a static export still argues.

    A candidate's annotation adds the strongest bin of its window and its
    level above the local noise floor.
    """
    if not candidate:
        return CLAIM_SENTENCES["figure_annotation"].format(name=name, line_hz=center)
    return CLAIM_SENTENCES["figure_annotation_candidate"].format(
        name=name,
        line_hz=center,
        peak_hz=candidate["line_peak_hz"],
        level_db=candidate["level_db"],
    )


def _caption(
    bands: list[dict[str, Any]],
    omitted: list[str],
    tolerance_pct: float,
    emphasised: list[str],
    rolling_element: bool,
    no_bands_reason: Optional[str] = None,
) -> str:
    """Author the sentence that says what the figure is evidence of."""
    if not bands and no_bands_reason:
        return (
            f"Envelope spectrum. {no_bands_reason} No characteristic-frequency "
            f"bands are drawn, so a peak cannot be attributed to a specific "
            f"bearing element from this figure."
        )
    if not bands:
        return (
            "Envelope spectrum. No bearing designation was supplied, so no "
            "characteristic-frequency bands are drawn and a peak cannot be "
            "attributed to a specific bearing element from this figure."
        )

    caption = CLAIM_SENTENCES["figure_caption"].format(
        labels=_join([band["label"] for band in bands]),
        tolerance_pct=tolerance_pct,
        rolling_element=(
            CLAIM_SENTENCES["figure_rolling_element"] if rolling_element else ""
        ),
    )
    if not emphasised:
        caption += " " + CLAIM_SENTENCES["figure_emphasis_none"]
    else:
        key = "figure_emphasis_one" if len(emphasised) == 1 else "figure_emphasis_many"
        caption += " " + CLAIM_SENTENCES[key].format(labels=_join(emphasised))
    if omitted:
        caption += (
            f" {', '.join(omitted)} lies beyond the plotted range and is not " f"shown."
        )
    return caption
