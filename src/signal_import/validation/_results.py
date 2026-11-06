"""
Per-channel values and confirmation statements of an import validation.

Statements are in the server's words; file text travels only in their
file-text fields.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

import math
from typing import Any, Mapping, Optional

from ...file_text import LABEL_CHARS, TOKEN_CHARS, FileText
from ...signal_acquisition.import_config import scaling_divisor
from ...signal_acquisition.text_decoder import DecodeResult
from .. import statements as st
from ..contracts import ValidatedChannel
from ..inspection import HEAD_TOKENS
from ._common import _Run, _citation, _origin
from ._evidence import _Evidence, _Preamble, _unit_token


def _label(evidence: _Evidence, column: int, field_count: int) -> Optional[FileText]:
    header = evidence.header
    if header is None or len(header) != field_count or not header[column]:
        return None
    return FileText.of(header[column], cap=LABEL_CHARS)


def _head_tokens(evidence: _Evidence, column: int) -> list[FileText]:
    tokens = [row[column] for row in evidence.head_rows if column < len(row)]
    return [FileText.of(token, cap=TOKEN_CHARS) for token in tokens[:HEAD_TOKENS]]


def _structure_statements(
    run: _Run, decode: DecodeResult, proofs: Mapping[str, bool]
) -> None:
    dialect = run.interpretation["dialect"]
    delimiter = {"tab": "a tab", "whitespace": "runs of spaces"}.get(
        dialect["delimiter"], f"'{dialect['delimiter']}'"
    )
    for path, text in (
        (
            "dialect",
            f"The file is read as {dialect['encoding']} text with {delimiter} "
            f"between fields and '{dialect['decimal']}' as the decimal mark",
        ),
        (
            "data_lines",
            f"The data rows are lines {decode.first_line} to {decode.last_line} "
            f"({decode.row_count} samples per channel)",
        ),
    ):
        entry = run.basis(path)
        run.statements.append(
            st.structure_statement(
                field=path,
                text=text,
                basis=entry["basis"] if entry else None,
                origin=_origin(entry),
                caller_asserted=run.caller,
                proven=proofs[path],
            )
        )


def _sample_stats(
    decode: DecodeResult, interpretation: Mapping[str, Any]
) -> dict[int, tuple[Optional[float], Optional[float]]]:
    """The mean and mean-to-standard-deviation ratio of each electrical
    channel's stored samples, from the moments the decoder streamed."""
    scaled = {
        c["column"]
        for c in interpretation["channels"]
        if c["scaling"] is not None and c["scaling"]["source_unit"]
    }
    stats: dict[int, tuple[Optional[float], Optional[float]]] = {}
    for decoded in decode.channels:
        if decoded.column not in scaled or decoded.mean is None:
            continue
        std = decoded.std
        ratio = (
            decoded.mean / std
            if std is not None and std > 0 and math.isfinite(std)
            else None
        )
        stats[decoded.column] = (decoded.mean, ratio)
    return stats


def _channel_results(
    run: _Run,
    decode: DecodeResult,
    evidence: _Evidence,
    identities: Mapping[int, Mapping[str, Any]],
) -> list[ValidatedChannel]:
    channels = {c["column"]: c for c in run.interpretation["channels"]}
    results: list[ValidatedChannel] = []
    for decoded in decode.channels:
        channel = channels[decoded.column]
        scaling = channel["scaling"]
        divisor = None
        if (
            scaling is not None
            and scaling["sensitivity"] is not None
            and scaling["source_unit"]
        ):
            try:
                divisor = format(scaling_divisor(scaling).normalize(), "f")
            except ValueError:
                divisor = None
        mean, ratio = run.stats.get(decoded.column, (None, None))
        token = _unit_token(evidence, decoded.column)
        results.append(
            ValidatedChannel(
                column=decoded.column,
                position=st.position_words(decoded.column),
                role=channel["role"],
                quantity=channel["quantity"],
                unit=channel["unit"],
                scaling_divisor=divisor,
                sample_count=decoded.sample_count,
                samples_digest=decoded.samples_digest,
                label=_label(evidence, decoded.column, decode.field_count),
                unit_token=FileText.of(token, cap=TOKEN_CHARS) if token else None,
                head_tokens=_head_tokens(evidence, decoded.column),
                mean=mean,
                mean_to_std=ratio,
                measurement_id=(identities.get(decoded.column) or {}).get(
                    "measurement_id"
                ),
            )
        )
    return results


def _statements(
    run: _Run,
    channels_out: list[ValidatedChannel],
    identities: Mapping[int, Mapping[str, Any]],
    shift: Optional[Mapping[str, Any]],
    declared: Optional[_Preamble],
) -> None:
    interpretation = run.interpretation
    time_base = interpretation["time_base"]
    entry = run.basis("time_base")
    discrepancy = ""
    if run.override("time_base") is not None:
        discrepancy = (
            "The rate sources disagree or a time column is off pattern; the "
            "choice of time-base source is recorded with its reason."
        )
    run.statements.insert(
        0,
        st.time_base_statement(
            rate_hz=time_base["sampling_rate_hz"],
            source=time_base["source"],
            time_column=time_base["time_column"],
            basis=entry["basis"] if entry else None,
            origin=_origin(entry),
            caller_asserted=run.caller,
            citation=_citation(entry),
            discrepancy=discrepancy,
            override=run.override("time_base"),
        ),
    )
    decoded = {c.column: c for c in channels_out}
    for channel in interpretation["channels"]:
        column = channel["column"]
        result = decoded.get(column)
        entry = run.basis(f"channels[{column}].unit")
        run.statements.append(
            st.channel_unit_statement(
                column=column,
                quantity=channel["quantity"],
                unit=channel["unit"],
                scaling=channel["scaling"],
                label=result.label if result else None,
                head_tokens=result.head_tokens if result else [],
                basis=entry["basis"] if entry else None,
                origin=_origin(entry),
                caller_asserted=run.caller,
                citation=_citation(entry),
                override=run.override(f"channels[{column}].unit"),
            )
        )
        scaling = channel["scaling"]
        if scaling is not None:
            entry = run.basis(f"channels[{column}].scaling")
            run.statements.append(
                st.scaling_statement(
                    column=column,
                    unit=channel["unit"],
                    scaling=scaling,
                    divisor=result.scaling_divisor if result else None,
                    basis=entry["basis"] if entry else None,
                    origin=_origin(entry),
                    caller_asserted=run.caller,
                    citation=_citation(entry),
                )
            )
        if (
            scaling is not None
            and scaling["source_unit"]
            and scaling["dc_policy"]
            and column not in run.pending_dc
        ):
            entry = run.basis(f"channels[{column}].dc_policy")
            run.statements.append(
                st.dc_policy_statement(
                    column=column,
                    unit=channel["unit"],
                    mean=result.mean if result else None,
                    mean_to_std=result.mean_to_std if result else None,
                    head_tokens=result.head_tokens if result else [],
                    basis=entry["basis"] if entry else None,
                    origin=_origin(entry),
                    caller_asserted=run.caller,
                )
            )
        identity = identities.get(column)
        if identity is not None:
            entry = run.basis(f"channels[{column}].binding")
            run.statements.append(
                st.binding_statement(
                    column=column,
                    identity=identity,
                    basis=entry["basis"] if entry else None,
                    origin=_origin(entry),
                    caller_asserted=run.caller,
                    citation=_citation(entry),
                )
            )
    for name, value in interpretation["defaults"].items():
        if value is None or name == "asset_id":
            continue
        entry = run.basis(f"defaults.{name}")
        run.statements.append(
            st.default_statement(
                field=name,
                value=str(value),
                basis=entry["basis"] if entry else None,
                origin=_origin(entry),
                caller_asserted=run.caller,
                citation=_citation(entry),
                shift=shift if name == "acquired_at" else None,
                override=run.override(f"defaults.{name}"),
            )
        )
    if run.basis("companion") is not None or any(
        issue.code == "companion_present" for issue in run.issues
    ):
        entry = run.basis("companion")
        run.statements.append(
            st.companion_statement(
                basis=entry["basis"] if entry else None, caller_asserted=run.caller
            )
        )
