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
from typing import Any, Mapping, Optional

from ... import config
from ...asset_ledger.store import LedgerStore
from ...file_text import LABEL_CHARS, FileText
from ...path_safety import PathRefused, open_import_file
from ...signal_acquisition.import_config import (
    build_import_format,
    import_id as compute_import_id,
)
from ...signal_acquisition.text_decoder import (
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
from ..inspection import MIN_WAVEFORM_ROWS
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
from ._evidence import _aligned, _evidence
from ._history import (
    _Prior,
    _Views,
    _candidate_ids,
    _check_history,
    _precheck,
    _prior_differs,
)
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
        ValueError: An unknown channel or transport, or a malformed
            ``PMM_IMPORT_ROOTS``.
        PathRefused: The path is unsafe, outside the import roots, not a
            regular ``.csv`` or ``.txt`` file, too large, or unreadable.
        TextScanRefused: Binary, compressed, UTF-16/UTF-32 or undecodable
            content, a line over the line cap, or rows too wide under every
            delimiter.
        ImportRecordError: An earlier import record cannot be read; it is
            never taken as no earlier record.
        LedgerError: The asset ledger cannot be read.
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
        size = file.size
        scan = scan_text(file)
        companion = os.path.exists(
            file.path.with_name(f"{file.path.stem}_metadata.json")
        )
        evidence = _evidence(scan, interpretation)
        try:
            # Digest mode keeps no samples; the decoder streams each channel's
            # digest and moments (R19's mean and ratio) block by block.
            decode = decode_text(file, interpretation, mode="digest")
        except TextDecodeError as refused:
            run.issue(
                refused.code,
                "invalid",
                str(refused),
                line=refused.line,
                column=refused.field,
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
    prior = _Prior(
        store=ImportRecordStore(config.get_import_dir()),
        sha256=decode.source_sha256 if decode is not None else None,
    )
    checks: list[TimeColumnCheck] = []
    channels_out: list[ValidatedChannel] = []
    prechecks: list[ChannelPrecheck] = []
    identities: dict[int, dict[str, Any]] = {}
    points: dict[int, Optional[dict[str, Any]]] = {}
    shift: Optional[dict[str, Any]] = None
    the_import_id: Optional[str] = None

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
        evidence.header = _aligned(evidence.header, evidence, decode.field_count)
        evidence.units = _aligned(evidence.units, evidence, decode.field_count)
        if evidence.header is not None and len(evidence.header) != decode.field_count:
            run.issue(
                "field_count_mismatch",
                "notice",
                f"The header has {len(evidence.header)} fields but the data rows "
                f"have {decode.field_count}, so labels and unit tokens are not "
                f"matched to columns: every channel is mapped by its index, and "
                f"the statements show no label.",
                line=evidence.header_line,
            )
            evidence.header = None
        if evidence.units is not None and len(evidence.units) != decode.field_count:
            run.issue(
                "field_count_mismatch",
                "notice",
                f"The units row has {len(evidence.units)} fields but the data rows "
                f"have {decode.field_count}, so its unit tokens are not matched to "
                f"columns.",
                line=evidence.units_line,
            )
            evidence.units = None
        checks, assessments = _time_columns(run, decode, _as_declared(declared_value))
        _check_rate(run, evidence, assessments, prior, declared_value)
        _time_candidates(run, evidence, decode.field_count)
        for channel_entry in interpretation["channels"]:
            _check_units(run, evidence, channel_entry)
            _check_time_like_channel(run, evidence, channel_entry)
        _check_stated_defaults(run, evidence)
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
        views = _Views(LedgerStore(config.get_ledger_dir()), candidates)
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
    return ValidationResult(
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
