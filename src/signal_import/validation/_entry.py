"""
Entry point of an import validation.

Normalizes the configuration (a pending DC policy included), gates and
decodes the file, runs every check and returns the typed result.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

from ... import config
from ...asset_ledger.store import LedgerStore
from ...file_text import LABEL_CHARS, FileText
from ...path_safety import PathRefused, open_import_file
from ...signal_acquisition.import_config import (
    build_import_format,
    import_id as compute_import_id,
)
from ...signal_acquisition.text_decoder import (
    DecodeMode,
    DecodeResult,
    TextDecodeError,
    decode_text,
)
from ...signal_acquisition.text_scan import scan_text
from ..attestation import (
    ConfigurationError,
    check_channel,
    check_transport,
    import_digest,
    is_caller_asserted,
    normalize_configuration,
)
from ..contracts import (
    ChannelPrecheck,
    TimeColumnCheck,
    ValidatedChannel,
    ValidationOutcome,
    ValidationResult,
)
from ..inspection import _LABEL_UNIT, MIN_WAVEFORM_ROWS, _bracketed
from ..records import ImportRecordStore
from ._checks import (
    _check_stated_defaults,
    _check_time_like_channel,
    _check_units,
    _check_value_bases,
    _effective_defaults,
    _identities,
    _required_bases,
    _structure_proofs,
)
from ._common import _Run
from ._evidence import _aligned, _Evidence, _evidence
from ._history import (
    _Budget,
    _Prior,
    _Views,
    _candidate_ids,
    _check_history,
    _precheck,
    _prior_differs,
)
from ._printed import _check_printed
from ._results import (
    _channel_results,
    _sample_stats,
    _statements,
    _structure_statements,
)
from ._time import _as_declared, _check_rate, _declared, _time_candidates, _time_columns

_DC_POLICY_PATH = re.compile(r"interpretation\.channels\[(\d+)\]\.scaling\.dc_policy")


def _pending_dc_policies(
    configuration: Mapping[str, Any], refused: ConfigurationError
) -> tuple[Optional[dict[str, Any]], set[int]]:
    """When the only problems are voltage channels without a DC policy, the
    configuration with that policy pending (``keep`` for the parse, which
    subtracts nothing) and the columns concerned; else ``(None, set())``.

    The choice is the user's, after the statement shows the channel's mean,
    so validation runs with the policy pending instead of refusing outright.
    """
    indices = []
    for problem in refused.problems:
        match = _DC_POLICY_PATH.fullmatch(problem.path)
        if problem.kind != "missing" or match is None:
            return None, set()
        indices.append(int(match.group(1)))
    patched = copy.deepcopy(dict(configuration))
    columns: set[int] = set()
    try:
        entries = patched["interpretation"]["channels"]
        for index in indices:
            entry = entries[index]
            entry["scaling"]["dc_policy"] = "keep"
            if isinstance(entry["column"], bool) or not isinstance(
                entry["column"], int
            ):
                return None, set()
            columns.add(entry["column"])
    except (KeyError, IndexError, TypeError):
        return None, set()
    return patched, columns


def _match_rows(run: _Run, evidence: _Evidence, field_count: int) -> None:
    """Match the header and units rows to the data's fields. A row that does
    not match is no longer read for labels or tokens; a row whose unit
    tokens are lost that way is recorded, so the units cannot pass as
    checked against it."""
    evidence.header = _aligned(evidence.header, evidence, field_count)
    evidence.units = _aligned(evidence.units, evidence, field_count)
    evidence.extra_units = _aligned(evidence.extra_units, evidence, field_count)
    if evidence.header is not None and len(evidence.header) != field_count:
        run.issue(
            "field_count_mismatch",
            "notice",
            f"The header has {len(evidence.header)} fields but the data rows "
            f"have {field_count}, so labels and unit tokens are not "
            f"matched to columns: every channel is mapped by its index, and "
            f"the statements show no label.",
            line=evidence.header_line,
        )
        if evidence.header_line is not None and any(
            _bracketed(label, _LABEL_UNIT) for label in evidence.header
        ):
            evidence.lost.append(
                ("header", evidence.header_line, len(evidence.header), field_count)
            )
        evidence.header = None
    if evidence.units is not None and len(evidence.units) != field_count:
        run.issue(
            "field_count_mismatch",
            "notice",
            f"The units row has {len(evidence.units)} fields but the data rows "
            f"have {field_count}, so its unit tokens are not matched to "
            f"columns.",
            line=evidence.units_line,
        )
        if evidence.units_line is not None and any(evidence.units):
            evidence.lost.append(
                ("units", evidence.units_line, len(evidence.units), field_count)
            )
        evidence.units = None
    extra = evidence.extra_units
    if extra is not None and len(extra) != field_count:
        if evidence.extra_units_line is not None and any(extra):
            evidence.lost.append(
                ("units", evidence.extra_units_line, len(extra), field_count)
            )
        evidence.extra_units = None
    evidence.extra_header = _aligned(evidence.extra_header, evidence, field_count)
    header = evidence.extra_header
    if header is not None and len(header) != field_count:
        if evidence.extra_header_line is not None and any(
            _bracketed(label, _LABEL_UNIT) for label in header
        ):
            evidence.lost.append(
                ("header", evidence.extra_header_line, len(header), field_count)
            )
        evidence.extra_header = None


class Validated(NamedTuple):
    """A validation together with what an import of it needs.

    Attributes:
        result: The :class:`ValidationResult` a caller sees.
        decode: The decode of the whole file (samples kept only when the
            validation ran in ``full`` mode); None when the parse stopped.
        path: The resolved path of the file that was read.
        interpretation: The canonical interpretation.
        attestation: The canonical attestation.
        identities: The identity block of each bound channel, by column.
        formats: The ``import_format`` of each decoded channel, by column.
        holders: For each bound measurement, the assets the bounded history
            read shows holding it as a current measurement.
    """

    result: ValidationResult
    decode: Optional[DecodeResult]
    path: Path
    interpretation: dict[str, Any]
    attestation: dict[str, Any]
    identities: dict[int, dict[str, Any]]
    formats: dict[int, dict[str, Any]]
    holders: dict[str, list[str]]


def validate_import(
    path: str,
    configuration: Mapping[str, Any],
    *,
    channel: str,
    transport: Optional[str] = None,
) -> ValidationResult:
    """Validate an import configuration against the whole file.

    Args:
        path: The file, absolute or relative to the data directory, under the
            data directory or a folder listed in ``PMM_IMPORT_ROOTS``.
        configuration: The configuration (``attestation.CONFIGURATION_SCHEMA``).
        channel: The confirmation channel, set by the calling adapter
            (``python_api``, ``mcp_tool_argument`` or ``playground_ui``);
            keyword-only with no default, since no request field may set it.
        transport: The MCP transport, when the channel is an MCP one.

    Returns:
        The :class:`ValidationResult`. Nothing is written anywhere.

    Raises:
        ConfigurationError: The configuration is malformed (one message,
            every problem typed).
        ValueError: An unknown channel or transport, a malformed
            ``PMM_IMPORT_ROOTS``, or a bound asset id that differs from an
            existing ledger only by letter case (the ledger store refuses it).
        PathRefused: The path is unsafe, outside the import roots, not a
            regular ``.csv`` or ``.txt`` file, too large, or unreadable.
        TextScanRefused: Binary, compressed, UTF-16/UTF-32 or undecodable
            content, a line over the line cap, or rows too wide under every
            delimiter.
        ImportRecordError: An earlier import record, or the record
            directory, cannot be listed, sized or read (an operating-system
            error); it is never taken as no earlier record.
        LedgerError: The ledger directory or a ledger cannot be listed, sized
            or read (an operating-system error). Both are ``ValueError``
            subclasses: the tool boundary returns them as refusals with their
            remedy, and no result (so never ``ready``) is produced.
    """
    return validate_for_import(
        path, configuration, channel=channel, transport=transport, mode="digest"
    ).result


def validate_for_import(
    path: str,
    configuration: Mapping[str, Any],
    *,
    channel: str,
    transport: Optional[str],
    mode: DecodeMode,
) -> Validated:
    """Run :func:`validate_import` and keep what an import of it needs.

    In ``full`` mode the decode keeps every channel's samples, so an import
    stores exactly the bytes this validation judged, read once through one
    verified descriptor. Nothing is written anywhere.

    Raises:
        The exceptions of :func:`validate_import`.
    """
    channel = check_channel(channel)
    transport = check_transport(transport)
    pending: set[int] = set()
    try:
        normalized = normalize_configuration(configuration)
    except ConfigurationError as refused:
        patched, pending = _pending_dc_policies(configuration, refused)
        if patched is None:
            raise
        normalized = normalize_configuration(patched)
        attested = normalized["attestation"]
        if any(
            f"channels[{column}].dc_policy" in attested[part]
            for column in pending
            for part in ("bases", "overrides")
        ):
            raise refused from None
    interpretation = normalized["interpretation"]
    run = _Run(
        caller=is_caller_asserted(channel),
        interpretation=interpretation,
        attestation=normalized["attestation"],
        pending_dc=pending,
    )
    roots = config.get_import_roots()
    decode: Optional[DecodeResult] = None
    with open_import_file(
        path, roots, base=config.DATA_DIR, max_size=config.get_max_signal_size()
    ) as file:
        name = FileText.of(file.path.name, cap=LABEL_CHARS)
        resolved = file.path
        size = file.size
        scan = scan_text(file)
        companion = os.path.exists(
            file.path.with_name(f"{file.path.stem}_metadata.json")
        )
        evidence = _evidence(scan, interpretation)
        try:
            # Digest mode keeps no samples; the decoder streams each channel's
            # digest and moments (R19's mean and ratio) block by block.
            decode = decode_text(file, interpretation, mode=mode)
        except TextDecodeError as refused:
            run.issue(
                refused.code,
                "invalid",
                str(refused),
                line=refused.line,
                column=refused.field,
                token=refused.token,
            )
        except PathRefused as refused:
            run.issue(refused.code, "invalid", str(refused))
    if decode is not None:
        run.stats = _sample_stats(decode, interpretation)

    time_base = interpretation["time_base"]
    entry = run.basis("time_base")
    declared_value = _declared(
        evidence,
        (
            entry["line"]
            if entry is not None and entry["basis"] == "user_accepted_file_value"
            else None
        ),
    )
    if declared_value is None:
        declared_value = _declared(evidence, None)
    budget = _Budget()
    prior = _Prior(
        store=ImportRecordStore(config.get_import_dir()),
        sha256=decode.source_sha256 if decode is not None else None,
        budget=budget,
    )
    checks: list[TimeColumnCheck] = []
    channels_out: list[ValidatedChannel] = []
    prechecks: list[ChannelPrecheck] = []
    identities: dict[int, dict[str, Any]] = {}
    points: dict[int, Optional[dict[str, Any]]] = {}
    shift: Optional[dict[str, Any]] = None
    the_import_id: Optional[str] = None
    formats: dict[int, dict[str, Any]] = {}
    holders: dict[str, list[str]] = {}

    if decode is not None:
        if decode.row_count < MIN_WAVEFORM_ROWS:
            run.issue(
                "region_too_short",
                "invalid",
                f"The data region holds {decode.row_count} rows, fewer than the "
                f"{MIN_WAVEFORM_ROWS}-sample default segment of the spectral tools, "
                f"which this version requires of an imported signal.",
                line=decode.first_line,
            )
        _match_rows(run, evidence, decode.field_count)
        checks, assessments = _time_columns(run, decode, _as_declared(declared_value))
        _check_rate(run, evidence, assessments, prior, declared_value)
        _time_candidates(run, evidence, decode.field_count)
        for channel_entry in interpretation["channels"]:
            _check_units(run, evidence, channel_entry)
            _check_time_like_channel(run, evidence, channel_entry)
        _check_stated_defaults(run, evidence)
        _check_printed(run, evidence, decode.field_count)
        the_import_id = compute_import_id(
            decode.source_sha256, decode.size_bytes, interpretation
        )
        defaults, shift = _effective_defaults(run, evidence)
        identities = _identities(run, decode, defaults)
        formats = {
            result.column: build_import_format(
                interpretation,
                result.column,
                samples_digest=result.samples_digest,
                sample_count=result.sample_count,
            )
            for result in decode.channels
        }
        candidates = _candidate_ids(decode)
        views = _Views(
            LedgerStore(config.get_ledger_dir()),
            candidates,
            decode.source_sha256,
            budget=budget,
        )
        prechecks, points = _precheck(run, decode, identities, formats, views)
        _prior_differs(
            run,
            decode,
            views,
            prior,
            the_import_id,
            {identity["measurement_id"] for identity in identities.values()},
        )
        _check_history(run, views, prior, candidates)
        holders = {
            identity["measurement_id"]: views.holders(identity["measurement_id"])
            for identity in identities.values()
        }
        channels_out = _channel_results(run, decode, evidence, identities)
    proofs = _structure_proofs(interpretation, evidence, decode)
    if decode is not None:
        _structure_statements(run, decode, proofs)
    _check_value_bases(run, evidence, prior, points)
    _required_bases(run, evidence, proofs, companion)
    _statements(run, channels_out, identities, shift, declared_value)

    effects = {issue.effect for issue in run.issues}
    outcome: ValidationOutcome = (
        "invalid"
        if "invalid" in effects
        else "needs_confirmation" if "needs_confirmation" in effects else "ready"
    )
    if run.pending_dc:
        # Computed with the policy pending: not the identity of any import.
        the_import_id = None
    digest = None
    if outcome == "ready" and the_import_id is not None:
        digest = import_digest(
            the_import_id, run.attestation, channel=channel, transport=transport
        )
    result = ValidationResult(
        outcome=outcome,
        file=name,
        source_sha256=decode.source_sha256 if decode is not None else None,
        size_bytes=size,
        first_line=decode.first_line if decode is not None else None,
        last_line=decode.last_line if decode is not None else None,
        row_count=decode.row_count if decode is not None else None,
        sampling_rate_hz=time_base["sampling_rate_hz"],
        time_base_source=time_base["source"],
        time_columns=checks,
        channels=channels_out,
        statements=run.statements,
        issues=run.issues,
        required_confirmations=[
            issue.message
            for issue in run.issues
            if issue.effect == "needs_confirmation"
        ],
        ledger_precheck=prechecks,
        confirmation_channel=channel,
        caller_asserted=run.caller,
        import_id=the_import_id,
        import_digest=digest,
    )
    return Validated(
        result=result,
        decode=decode,
        path=resolved,
        interpretation=interpretation,
        attestation=run.attestation,
        identities=identities,
        formats=formats,
        holders=holders,
    )
