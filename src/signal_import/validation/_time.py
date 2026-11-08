"""
Time-base checks of an import validation.

Every time column is assessed over the whole data region; the rate's
basis is checked against the evidence it cites, and the rate against
every rate or interval the file prints.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

from fractions import Fraction
from typing import Mapping, Optional

from ...signal_acquisition.import_config import normalize_decimal
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..contracts import IssueEffect, TimeColumnCheck
from ..timebase import (
    DeclaredTimeBase,
    TimeBaseAssessment,
    TimeBaseRefused,
    assess_time_column,
)
from ._common import _ASK, _Run, _decimal, _printed
from ._evidence import _Evidence, _Preamble, _time_like
from ._history import _Prior


def _interval_rates(printed: str) -> Optional[tuple[Fraction, Fraction]]:
    """The rates a printed sample interval supports: its half-unit interval,
    inverted into rates. A value the decimal contract refuses (an exponent
    beyond 1e30 either way, too many digits) supports none, before any exact
    arithmetic on it."""
    try:
        normalize_decimal(printed, field="the printed sample interval")
    except ValueError:
        return None
    value = _decimal(printed)
    if value is None or value <= 0:
        return None
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        return None
    half = Fraction(1, 2) * Fraction(10) ** exponent
    center = Fraction(value)
    low, high = center - half, center + half
    if low <= 0:
        return Fraction(0), Fraction(10**30)
    return 1 / high, 1 / low


def _declared(evidence: _Evidence, cited_line: Optional[int]) -> Optional[_Preamble]:
    """The preamble rate or interval: the cited one, else the first rate,
    else the first interval."""
    rates = [
        p
        for p in evidence.preamble
        if p.field in ("sampling_rate_hz", "sample_interval") and p.printed
    ]
    if cited_line is not None:
        for value in rates:
            if value.line == cited_line:
                return value
        return None
    for wanted in ("sampling_rate_hz", "sample_interval"):
        for value in rates:
            if value.field == wanted:
                return value
    return None


def _as_declared(value: Optional[_Preamble]) -> Optional[DeclaredTimeBase]:
    if value is None or value.printed is None:
        return None
    if value.field == "sampling_rate_hz":
        return DeclaredTimeBase(kind="rate", token=value.printed)
    return DeclaredTimeBase(kind="interval", token=value.printed, time_unit="s")


def _time_columns(
    run: _Run,
    decode: DecodeResult,
    declared: Optional[DeclaredTimeBase],
) -> tuple[list[TimeColumnCheck], dict[int, TimeBaseAssessment]]:
    """Assess every decoded time column; report each finding."""
    time_base = run.interpretation["time_base"]
    stated = time_base["sampling_rate_hz"] if time_base["source"] == "stated" else None
    checks: list[TimeColumnCheck] = []
    assessments: dict[int, TimeBaseAssessment] = {}
    choice = run.override("time_base")
    for column in decode.time_columns:
        where = f"The time column at {st.channel_words(column.column)}"
        try:
            assessment = assess_time_column(
                column,
                first_line=decode.first_line,
                stated_rate_hz=stated,
                declared=declared,
            )
        except TimeBaseRefused as refused:
            remedy = (
                " Remove it from time_columns and name an override with a reason "
                "to leave it unassessed."
                if refused.datetime_text
                else ""
            )
            run.issue(
                refused.code,
                "invalid",
                f"{where}: {refused}{remedy}",
                field_path=f"time_columns[{column.column}]",
                line=refused.line,
                column=column.column,
                token=refused.token,
            )
            continue
        assessments[column.column] = assessment
        checks.append(
            TimeColumnCheck(
                column=column.column,
                time_unit=assessment.time_unit,
                outcome=assessment.outcome,
                evidence=assessment.evidence,
                regime=assessment.regime,
                resolution=assessment.resolution,
                fitted_rate_hz=assessment.fitted_rate_hz,
                proposed_rate_hz=assessment.proposed_rate_hz,
                finding_codes=[f.code for f in assessment.findings],
                statements=[f.statement for f in assessment.findings],
            )
        )
        for finding in assessment.findings:
            effect: IssueEffect
            if finding.code in ("time_gap", "time_reversal", "time_residual"):
                effect = "invalid"
                measured = ""
                if finding.measures and finding.code != "time_residual":
                    measured = (
                        f" The first measures {finding.measures[0]:.6g} sample "
                        f"intervals."
                    )
                remedy = (
                    f"{measured} The samples are never resampled, filled or "
                    f"repaired: correct the export at its source."
                )
            elif choice is not None:
                effect, remedy = "notice", " The time-base override is recorded."
            else:
                effect = "needs_confirmation"
                remedy = (
                    f" {_ASK.capitalize()} to choose the time-base source and name "
                    f"an override with a reason for 'time_base'."
                )
            statement = finding.statement.rstrip()
            if not statement.endswith("."):
                statement += "."
            run.issue(
                finding.code,
                effect,
                f"{where}: {statement}{remedy}",
                field_path="time_base",
                line=finding.lines[0] if finding.lines else None,
                column=column.column,
            )
    return checks, assessments


def _check_rate(
    run: _Run,
    evidence: _Evidence,
    assessments: Mapping[int, TimeBaseAssessment],
    prior: "_Prior",
    assessed: Optional[_Preamble],
) -> None:
    """The time base's basis against the evidence it cites, and the rate
    against every rate or interval the file prints (*assessed* is the one
    the time-column assessment compared)."""
    time_base = run.interpretation["time_base"]
    rate, source = time_base["sampling_rate_hz"], time_base["source"]
    entry = run.basis("time_base")
    if entry is None:
        run.missing_basis("time_base", f"The sampling rate {rate} Hz")
    if source == "time_column":
        column = time_base["time_column"]
        assessment = assessments.get(column)
        if assessment is not None and assessment.proposed_rate_hz is None:
            run.issue(
                "time_resolution_insufficient",
                "needs_confirmation",
                f"The time column at {st.channel_words(column)} cannot prove "
                f"uniform sampling ({assessment.evidence} evidence at a resolution "
                f"of {assessment.resolution} {assessment.time_unit}), so it cannot "
                f"set the sampling rate: {_ASK} to state the rate, or to accept "
                f"one the file prints, as the time-base source.",
                field_path="time_base",
                column=column,
            )
        elif entry is not None and assessment is not None:
            if (
                entry["basis"] != "user_accepted_derived_value"
                or entry["column"] != column
            ):
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"A rate derived from the time column at "
                    f"{st.channel_words(column)} is accepted by citing that column "
                    f"with the basis 'user_accepted_derived_value'.",
                    field_path="time_base",
                )
            elif _decimal(rate) != _decimal(assessment.proposed_rate_hz):
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"The configured rate {rate} Hz is not the rate "
                    f"{assessment.proposed_rate_hz} Hz the time column at "
                    f"{st.channel_words(column)} derives.",
                    field_path="time_base",
                    column=column,
                )
    elif source == "file_declared" and entry is not None:
        cited = _declared(evidence, entry["line"]) if entry["line"] else None
        if entry["basis"] != "user_accepted_file_value" or cited is None:
            run.issue(
                "basis_mismatch",
                "invalid",
                "A file-declared rate is accepted by citing the preamble line "
                "that prints it (basis 'user_accepted_file_value'); no sampling "
                "rate or sample interval is printed on the cited line.",
                field_path="time_base",
                line=entry["line"],
            )
        elif not _rate_matches(rate, cited):
            run.issue(
                "basis_mismatch",
                "invalid",
                f"The configured rate {rate} Hz is not the value line "
                f"{cited.line} prints.",
                field_path="time_base",
                line=cited.line,
            )
    elif source == "stated" and entry is not None:
        if entry["basis"] == "user_accepted_prior_value":
            prior_rate = prior.time_base_rate(entry["import_id"])
            if prior_rate is None and prior.unread(entry["import_id"]):
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"The cited earlier import was not read in this validation, so "
                    f"the rate {rate} Hz cannot be compared with it.",
                    field_path="time_base",
                )
            elif prior_rate is None or _decimal(prior_rate) != _decimal(rate):
                run.issue(
                    "basis_mismatch",
                    "invalid",
                    f"The configured rate {rate} Hz is not the rate the cited "
                    f"earlier import of these bytes used.",
                    field_path="time_base",
                )
        elif entry["basis"] != "user_stated":
            run.issue(
                "basis_mismatch",
                "invalid",
                "A stated rate takes the basis 'user_stated' (or an earlier "
                "import's rate, cited by its import_id); a rate the file prints "
                "has the source 'file_declared', one a time column derives the "
                "source 'time_column'.",
                field_path="time_base",
            )
    # Every printed rate or interval the configured rate must fit (R15),
    # except the cited line (checked above) and the value the time-column
    # assessment already compared.
    compared: set[int] = set()
    if source == "file_declared" and entry is not None and entry["line"]:
        compared.add(entry["line"])
    if assessments and assessed is not None:
        compared.add(assessed.line)
    choice = run.override("time_base")
    for printed in evidence.preamble:
        if (
            printed.field in ("sampling_rate_hz", "sample_interval")
            and not printed.printed
            and printed.line not in compared
        ):
            # Recognized but not readable as the field's number: never taken
            # as agreeing with the configured rate.
            what = (
                "a sampling rate that this version cannot read as a number in Hz"
                if printed.field == "sampling_rate_hz"
                else "a sample interval that this version cannot read as a number "
                "in s"
            )
            run.issue(
                "printed_value_unverified",
                "notice" if choice is not None else "needs_confirmation",
                f"Line {printed.line} prints {what} in the file's decimal mark (its "
                f"text as printed is in this issue's token), so the configured rate "
                f"{rate} Hz cannot be checked against it"
                + (
                    "; the time-base override is recorded."
                    if choice is not None
                    else f": {_ASK} to choose the time-base source, and name an "
                    f"override with a reason for 'time_base'."
                ),
                field_path="time_base",
                line=printed.line,
                token=_printed(printed.raw),
            )
            continue
        if (
            printed.field not in ("sampling_rate_hz", "sample_interval")
            or not printed.printed
            or printed.line in compared
            or _rate_matches(rate, printed)
        ):
            continue
        run.issue(
            "rate_sources_disagree",
            "notice" if choice is not None else "needs_confirmation",
            f"The configured rate {rate} Hz lies outside what line "
            f"{printed.line} of the file prints, within its printed precision. "
            + (
                "The time-base override is recorded."
                if choice is not None
                else f"{_ASK.capitalize()} to choose the time-base source and "
                f"name an override with a reason for 'time_base'."
            ),
            field_path="time_base",
            line=printed.line,
        )


def _rate_matches(rate: str, declared: _Preamble) -> bool:
    """A configured rate equals a printed rate, or lies inside what a printed
    interval supports."""
    if declared.field == "sampling_rate_hz":
        return _decimal(rate) == _decimal(declared.normalized)
    bounds = _interval_rates(declared.printed or "")
    value = _decimal(rate)
    return (
        bounds is not None
        and value is not None
        and bounds[0] <= Fraction(value) <= bounds[1]
    )


def _time_candidates(run: _Run, evidence: _Evidence, field_count: int) -> None:
    """A column that looks like time and is neither listed nor imported
    needs an override to stay unassessed."""
    interpretation = run.interpretation
    taken = {c["column"] for c in interpretation["time_columns"]} | {
        c["column"] for c in interpretation["channels"]
    }
    decimal_mark = interpretation["dialect"]["decimal"]
    for column in range(field_count):
        if column in taken or not _time_like(evidence, column, decimal_mark):
            continue
        path = f"time_columns[{column}]"
        if run.override(path) is not None:
            run.issue(
                "time_column_unassessed",
                "notice",
                f"The column at {st.channel_words(column)} looks like time and is "
                f"left unassessed; the override with its reason is recorded.",
                field_path=path,
                column=column,
            )
            continue
        run.issue(
            "time_column_unassessed",
            "needs_confirmation",
            f"The column at {st.channel_words(column)} looks like time and is "
            f"neither listed in time_columns nor imported: {_ASK} whether it is "
            f"a time column (list it with its time_unit, so it is assessed over "
            f"the whole file) or why it should stay unassessed (an override with "
            f"a reason for {path!r}).",
            field_path=path,
            column=column,
        )
