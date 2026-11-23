"""The chronos2 backend's public texts: placed as approved, and inside the
scientific boundary.

- Placement: the tool docstring, the server instructions, the tool catalog
  row, the operator guide and the CHANGELOG entry carry the approved words
  (line wrapping, Markdown quoting and list markers aside).
- Boundary: no new text names a threshold, a crossing, a failure probability
  or a remaining useful life except to say the result is not one; no new
  field or warning code names one; no model-accuracy figure; no dash that the
  wording rules exclude.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import _chronos2_texts as T
from predictive_maintenance_mcp import models as m
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
    forecast_asset_indicator,
)
from predictive_maintenance_mcp.prognostics import chronos2_backend as cb
from predictive_maintenance_mcp.prognostics import indicator_forecast as f

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "guides" / "forecast-model-backend.md"


def _flat(text: str) -> str:
    """Words as a reader sees them: no Markdown quoting, list markers or code
    ticks, and every run of whitespace one space."""
    text = re.sub(r"(?m)^\s*(?:>\s?|[-*]\s|#+\s)", "", text)
    return " ".join(text.replace("`", "").split())


def _instructions() -> str:
    from predictive_maintenance_mcp.server import mcp

    return _flat(mcp.instructions or "")


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", [T.DOCSTRING_OPENING, T.DOCSTRING_BACKEND, T.DOCSTRING_RETURNS]
)
def test_the_tool_docstring_carries_the_approved_text(text):
    assert _flat(text) in _flat(forecast_asset_indicator.__doc__ or "")


@pytest.mark.parametrize("bullet", T.INSTRUCTION_BULLETS)
def test_the_server_instructions_carry_the_approved_bullets(bullet):
    assert _flat(bullet) in _instructions()


def test_the_replaced_instruction_sentences_are_gone():
    text = _instructions()
    assert "with a statistical reference method, and returns" not in text
    assert "remaining-useful-life estimate, and its band is nominal Never" not in text


def test_the_tool_catalog_row_is_the_approved_one():
    lines = (REPO_ROOT / "docs" / "TOOL_CATALOG.md").read_text("utf-8").splitlines()
    assert T.CATALOG_ROW in lines


@pytest.mark.parametrize("paragraph", T.GUIDE, ids=range(len(T.GUIDE)))
def test_the_operator_guide_carries_the_approved_text(paragraph):
    assert _flat(paragraph) in _flat(GUIDE.read_text("utf-8"))


@pytest.mark.parametrize(
    "paragraph", T.CHANGELOG_ENTRY, ids=range(len(T.CHANGELOG_ENTRY))
)
def test_the_changelog_entry_is_the_approved_one(paragraph):
    text = (REPO_ROOT / "CHANGELOG.md").read_text("utf-8")
    unreleased = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    assert _flat(paragraph) in _flat(unreleased)


def test_the_guide_points_to_the_refusals_name_for_it():
    # Every refusal sends the operator to "the forecasting model guide".
    assert GUIDE.read_text("utf-8").startswith("# Forecasting model guide")


# ---------------------------------------------------------------------------
# The scientific boundary
# ---------------------------------------------------------------------------


def _new_texts() -> dict[str, str]:
    texts = {
        "method_note": f.CHRONOS2_METHOD_NOTE,
        "interval_note": f.CHRONOS2_INTERVAL_NOTE,
        "reference_note": f.REFERENCE_NOTE,
        "context_truncated": f.CONTEXT_TRUNCATED,
        "recent_gap": f.RECENT_GAP,
        "too_few_observed": f.TOO_FEW_OBSERVED_IN_WINDOW,
        "flat_window": f.FLAT_WINDOW,
        "reference_not_computable": f.REFERENCE_NOT_COMPUTABLE,
        "reference_band": f.REFERENCE_BAND_NOT_REPORTABLE,
        "horizon_above_144": T.HORIZON_ABOVE_144,
        "horizon_beyond_span": T.HORIZON_BEYOND_SPAN,
        "extra_not_installed": cb.EXTRA_NOT_INSTALLED,
        "dir_not_set": cb.DIR_NOT_SET,
        "not_a_model_dir": cb.NOT_A_MODEL_DIR,
        "digest_mismatch": cb.DIGEST_MISMATCH,
        "version_outside_range": cb.VERSION_OUTSIDE_RANGE,
        "load_failed": cb.LOAD_FAILED,
        "still_loading": cb.STILL_LOADING,
        "docstring": " ".join(
            [T.DOCSTRING_OPENING, T.DOCSTRING_BACKEND, T.DOCSTRING_RETURNS]
        ),
        "instructions": " ".join(T.INSTRUCTION_BULLETS),
        "guide": GUIDE.read_text("utf-8"),
        "changelog": " ".join(T.CHANGELOG_ENTRY),
        "catalog": T.CATALOG_ROW,
    }
    texts.update({f"sign_{k}": v for k, v in f.MODEL_SIGN_WARNING_DETAILS.items()})
    for model in (m.IndicatorForecastModel, m.IndicatorForecastReference):
        for name, field in model.model_fields.items():
            texts[f"{model.__name__}.{name}"] = field.description or ""
    for model, name in (
        (m.IndicatorForecastResult, "reference"),
        (m.IndicatorForecastHistory, "difference_lag"),
        (m.IndicatorForecastProvenance, "model"),
        (m.IndicatorForecastProvenance, "family"),
        (m.IndicatorForecastProvenance, "algorithm_version"),
    ):
        texts[f"{model.__name__}.{name}"] = model.model_fields[name].description or ""
    return texts


#: Concepts the backend's texts may name only to say the result is not one.
BOUNDARY = re.compile(
    r"threshold|crossing|remaining.useful.life|\bRUL\b|failure probability|"
    r"probability|time to a limit|reach any limit",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\bnot\b|\bnever\b|\bno\b", re.IGNORECASE)


@pytest.mark.parametrize("name, text", sorted(_new_texts().items()))
def test_a_boundary_concept_appears_only_negated(name, text):
    sentences = re.split(r"(?<=[.;:])\s+", " ".join(text.split()))
    for sentence in sentences:
        if BOUNDARY.search(sentence):
            assert NEGATION.search(sentence), (name, sentence)


def test_no_new_field_or_code_names_a_boundary_concept():
    names = list(m.IndicatorForecastModel.model_fields) + list(
        m.IndicatorForecastReference.model_fields
    )
    names += [
        "reference",
        "model",
        "model_context_truncated",
        "recent_gap_before_origin",
    ]
    for name in names:
        assert not re.search(r"threshold|crossing|rul|failure|probab|life", name), name


def test_the_interval_note_carries_no_accuracy_figure():
    # Standalone numbers only: P10 and P90 name the quantiles.
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", f.CHRONOS2_INTERVAL_NOTE))
    assert numbers == {"10"}  # the two evaluated cadences only
    assert all("%" not in text for text in _new_texts().values())


@pytest.mark.parametrize("name, text", sorted(_new_texts().items()))
def test_no_excluded_dash(name, text):
    assert chr(0x2014) not in text and chr(0x2013) not in text, name
