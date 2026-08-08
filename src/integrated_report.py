"""Integrated diagnostic report template.

This module places strings; it never composes them. Every evaluative sentence
it renders arrives already written from
:mod:`.decision_support.advisory`, and the template's only job is layout.

The analogy that shaped it is the radiology report: an annotated image, the
findings, the impression, the recommendation — and nobody expects the imaging
software to write the impression. What this template contributes is the
annotation and the arrangement, not the judgement.

It lives outside :mod:`.html_templates` for two reasons: that module is
already long and is excluded from the coverage gate, and this one is the
report whose contract most needs test coverage.
"""

from __future__ import annotations

import html
from typing import Any, Optional

from .figures import figure_to_svg
from .html_templates import get_base_template

_URGENCY_BADGE = {
    "low": "badge-success",
    "medium": "badge-warning",
    "high": "badge-warning",
    "critical": "badge-danger",
}

_STATUS_BADGE = {
    "assessed": "badge-info",
    "refused": "badge-warning",
    "absent": "badge-warning",
}

_MUTED = "color:var(--text-secondary);"


def _esc(value: Any) -> str:
    """Escape a value for HTML. Signal ids reach this template from user input."""
    return html.escape(str(value if value is not None else ""))


def _remedy_paragraph(remedy: Optional[str]) -> str:
    if not remedy:
        return ""
    return (
        f'<p style="margin-top:0.75rem;{_MUTED}">'
        f"<strong>To resolve:</strong> {_esc(remedy)}</p>"
    )


def _status_badge(status: Optional[str]) -> str:
    css = _STATUS_BADGE.get(status or "", "badge-info")
    return (
        f'<span class="badge {css}" style="margin-left:0.5rem;">'
        f"{_esc(status)}</span>"
    )


def _block_card(title: str, block: dict[str, Any]) -> str:
    """Render one indicator block, including its authored absence statement.

    A block whose input was missing still renders. A silently omitted section
    reads as "nothing to report", which is a different claim from "this could
    not be determined, and here is what that costs you".
    """
    return (
        '<div class="card">'
        f'<h2 class="card-title">{_esc(title)}'
        f'{_status_badge(block.get("status"))}</h2>'
        f'<p>{_esc(block.get("statement"))}</p>'
        f'{_remedy_paragraph(block.get("remedy"))}'
        "</div>"
    )


def _bearing_table(block: dict[str, Any]) -> str:
    """Calculated against measured, with a verdict per bearing element.

    Putting the two numbers side by side is what lets a reader check the
    match rather than accept it.
    """
    rows = block.get("rows") or []
    if not rows:
        return ""

    cells = []
    for row in rows:
        measured = (
            f"{row['measured_hz']:.2f} Hz" if row.get("measured_hz") else "&mdash;"
        )
        deviation = (
            f"{row['deviation_pct']:.2f}%"
            if row.get("deviation_pct") is not None
            else "&mdash;"
        )
        verdict_css = "badge-danger" if row["matched"] else "badge-success"
        verdict_text = "match" if row["matched"] else "no match"
        cells.append(
            "<tr>"
            f"<td>{_esc(row['fault_type'])}</td>"
            f"<td>{row['expected_hz']:.2f} Hz</td>"
            f"<td>{measured}</td>"
            f"<td>{deviation}</td>"
            f'<td><span class="badge {verdict_css}">{verdict_text}</span></td>'
            "</tr>"
        )

    return (
        '<div class="card">'
        '<h2 class="card-title">Calculated against measured</h2>'
        '<table style="width:100%;border-collapse:collapse;">'
        '<thead><tr style="text-align:left;border-bottom:2px solid var(--border-color);">'
        "<th>Element</th><th>Calculated</th><th>Measured</th>"
        "<th>Deviation</th><th>Verdict</th></tr></thead>"
        f'<tbody>{"".join(cells)}</tbody>'
        "</table></div>"
    )


def _figure_card(figure: Optional[dict[str, Any]]) -> str:
    """The figure that closes the matching argument, or a note that it is absent."""
    if not figure:
        return (
            '<div class="card">'
            '<h2 class="card-title">Envelope spectrum</h2>'
            "<p>This report carries no envelope spectrum figure, so the "
            "frequency match cannot be checked visually here &mdash; only "
            "against the values in the table above.</p>"
            "</div>"
        )

    return (
        '<div class="chart-container">'
        '<h2 class="card-title">Envelope spectrum</h2>'
        f"{figure_to_svg(figure)}"
        f'<p style="margin-top:1rem;{_MUTED}font-size:0.9rem;">'
        f'{_esc(figure["caption"])}</p>'
        "</div>"
    )


def _recommendations_card(recommendations: list[dict[str, Any]]) -> str:
    """Each action with the reasoning and the evidence it rests on."""
    if not recommendations:
        return ""

    items = []
    for rec in recommendations:
        urgency = rec.get("urgency", "medium")
        badge = _URGENCY_BADGE.get(urgency, "badge-info")
        evidence = "".join(
            f'<p style="margin-top:0.25rem;{_MUTED}font-size:0.9rem;">'
            f"Evidence: {_esc(item)}</p>"
            for item in rec.get("evidence", [])
        )
        items.append(
            '<div style="padding:1rem 0;border-bottom:1px solid var(--border-color);">'
            '<p style="font-weight:600;font-size:1.05rem;">'
            f'<span class="badge {badge}" style="margin-right:0.5rem;">'
            f"{_esc(urgency)}</span>{_esc(rec['action'])}</p>"
            f'<p style="margin-top:0.4rem;">{_esc(rec.get("description"))}</p>'
            f'<p style="margin-top:0.4rem;{_MUTED}">'
            f"<strong>Why:</strong> {_esc(rec.get('motivation'))}</p>"
            f"{evidence}</div>"
        )

    return (
        '<div class="card">'
        '<h2 class="card-title">Recommended actions</h2>'
        f'{"".join(items)}</div>'
    )


def _baseline_card(baseline: dict[str, Any]) -> str:
    deltas = "".join(
        f'<li style="margin-bottom:0.4rem;">{_esc(delta["statement"])}</li>'
        for delta in baseline.get("deltas", [])
    )
    delta_list = (
        f'<ul style="margin-top:0.75rem;padding-left:1.25rem;">{deltas}</ul>'
        if deltas
        else ""
    )
    return (
        '<div class="card">'
        '<h2 class="card-title">Comparison with baseline'
        f'{_status_badge(baseline.get("status"))}</h2>'
        f'<p>{_esc(baseline.get("statement"))}</p>'
        f"{delta_list}"
        f'{_remedy_paragraph(baseline.get("remedy"))}'
        "</div>"
    )


def _iso_card(iso: dict[str, Any]) -> str:
    note = (
        f'<p style="margin-top:0.75rem;font-size:0.875rem;{_MUTED}">'
        f'{_esc(iso["standard_note"])}</p>'
        if iso.get("standard_note")
        else ""
    )
    return (
        '<div class="card">'
        '<h2 class="card-title">ISO severity'
        f'{_status_badge(iso.get("status"))}</h2>'
        f'<p>{_esc(iso.get("statement"))}</p>'
        f"{note}"
        f'{_remedy_paragraph(iso.get("remedy"))}'
        "</div>"
    )


def _evidence_card(evidence: dict[str, Any]) -> str:
    items = "".join(
        f'<li style="margin-bottom:0.5rem;">{_esc(item)}</li>'
        for item in evidence["statements"]
    )
    return (
        '<div class="card">'
        '<h2 class="card-title">Evidence</h2>'
        f'<p><strong>Evidence strength: {_esc(evidence["strength"])}</strong></p>'
        f'<p style="{_MUTED}font-size:0.9rem;margin-top:0.25rem;">'
        f'{_esc(evidence["strength_explanation"])}</p>'
        f'<ul style="margin-top:0.9rem;padding-left:1.25rem;">{items}</ul>'
        "</div>"
    )


def _provenance_card(provenance: dict[str, Any], generated_at: str) -> str:
    fields = [
        ("Signal", provenance.get("signal_id")),
        ("Shaft speed (RPM)", provenance.get("rpm")),
        ("Bearing", provenance.get("bearing_id")),
        (
            "ISO machine group / support",
            f"{provenance.get('machine_group')} / {provenance.get('support_type')}",
        ),
        ("Server version", provenance.get("server_version")),
        ("Generated", generated_at),
    ]
    cells = "".join(
        '<div class="info-item">'
        f'<div class="info-label">{_esc(label)}</div>'
        f'<div class="info-value" style="font-size:1rem;">{_esc(value)}</div>'
        "</div>"
        for label, value in fields
    )
    return (
        '<div class="card">'
        '<h2 class="card-title">Provenance</h2>'
        f'<div class="info-grid">{cells}</div>'
        "</div>"
    )


def create_integrated_diagnostic_report(
    advisory: dict[str, Any],
    figure: Optional[dict[str, Any]] = None,
    generated_at: str = "",
) -> str:
    """Render the server-authored advisory as one integrated document.

    Args:
        advisory: Payload from
            :func:`.decision_support.advisory.build_advisory`. Every
            evaluative string rendered here comes from it.
        figure: Optional description from
            :func:`.figures.build_annotated_envelope_figure`.
        generated_at: Generation timestamp, passed in rather than read from
            the clock. Reproducibility is a claim about content, and a
            template that reads the clock cannot make it.

    Returns:
        A complete, self-contained HTML document with no external references.
    """
    verdict = advisory["verdict"]

    disagreements = "".join(
        '<div class="card" style="border-left:4px solid var(--warning-color);">'
        '<h2 class="card-title">Indicators disagree</h2>'
        f'<p>{_esc(entry["statement"])}</p></div>'
        for entry in advisory["disagreements"]
    )

    content = (
        '<div class="header"><div class="header-content">'
        f'<h1>Diagnostic report &mdash; {_esc(advisory["signal_id"])}</h1>'
        f'<p class="subtitle">{_esc(verdict["statement"])}</p>'
        "</div></div>"
        '<div class="container">'
        f"{_evidence_card(advisory['evidence'])}"
        f"{_iso_card(advisory['iso'])}"
        f"{disagreements}"
        f"{_block_card('Bearing frequency matching', advisory['bearing_match'])}"
        f"{_bearing_table(advisory['bearing_match'])}"
        f"{_figure_card(figure)}"
        f"{_block_card('Anomaly detection', advisory['anomaly'])}"
        f"{_block_card('Spectral energy distribution', advisory['spectral_energy'])}"
        f"{_baseline_card(advisory['baseline_comparison'])}"
        f"{_recommendations_card(advisory['recommendations'])}"
        f"{_provenance_card(advisory['provenance'], generated_at)}"
        "</div>"
    )

    return get_base_template(
        title=f"Diagnostic report - {advisory['signal_id']}",
        content=content,
        metadata={
            "report_type": "integrated_diagnostic",
            "provenance": advisory["provenance"],
            "verdict": verdict,
            "evidence_strength": advisory["evidence"]["strength"],
        },
        include_plotly=False,
    )
