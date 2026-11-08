"""
The signal import service: import, re-materialization and the import index.

ISO 13374 Block 1 (data acquisition). The operations a caller strings
together are inspect (:mod:`.inspection`), configure (:mod:`.configure`),
validate (:mod:`.validation`) and, here, import. Every adapter (the MCP
tools, the Python API, later the playground) goes through
:class:`SignalImportService` and passes its own confirmation channel and
transport; no request field can set them.

An import runs in this order:

1. validate the configuration against the whole file again, keeping the
   samples, through one verified descriptor, and compare the import digest
   the caller presents with the one this validation computes. A byte of the
   file, a value, a basis, an override, the channel or the transport that
   changed since the caller validated is refused with "validate again";
2. check the signal ids in the store;
3. write the import record's interpretation line, then the attestation,
   location and runtime lines;
4. insert every channel into the signal store in one batch, or none;
5. append the outcome (signal ids, materialized flags) to the record;
6. record each bound channel in the asset ledger, computing its snapshot
   from the array this import decoded, never from a store lookup that a
   concurrent load could have replaced.

A refusal before step 3 leaves the store, the record store and the ledger
unchanged. A store collision after step 3 (a concurrent load) leaves the
record, which is harmless because it is content-addressed, and reports the
channels as not materialized. A ledger failure never undoes the store
insert; it is reported per channel, and repeating the import records the
channels not yet recorded. Repeating an import of a recorded interpretation
returns ``already_imported`` with the same signal ids and re-materializes
the channels the store evicted.

Nothing here leaves the machine, and file text reaches a result only as
:class:`~predictive_maintenance_mcp.file_text.FileText`.
"""

import hashlib
import logging
import platform
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any, Literal, Optional, Union

import numpy as np

from .. import config
from ..asset_ledger.service import IMPORT_DECLARED_BY, record_measurements
from ..asset_ledger.store import LedgerError, LedgerStore
from ..file_text import LABEL_CHARS, FileText
from ..path_safety import (
    PathRefused,
    gate_import_path,
    open_import_file,
    validate_name_component,
)
from ..signal_acquisition.import_config import build_import_format, is_import_id
from ..signal_acquisition.repository import (
    DecodedChannel,
    SignalIdTaken,
    SignalRepository,
)
from ..signal_acquisition.text_decoder import (
    DECODER_CONTRACT,
    TextDecodeError,
    get_decoder,
)
from .attestation import check_channel, check_transport, is_caller_asserted
from .contracts import ImportedChannel, ImportResult, ImportStatus
from .records import (
    ImportRecord,
    ImportRecordIndexEntry,
    ImportRecordMissing,
    ImportRecordStore,
)
from .statements import position_words
from .validation import Validated, validate_for_import

__all__ = [
    "ImportRefused",
    "Presence",
    "SignalImportService",
    "record_locations",
]

logger = logging.getLogger(__name__)

#: Whether a channel of an import is stored under its id: this import's
#: channel (``same``), nothing (``absent``) or another signal (``other``).
Presence = Literal["same", "absent", "other"]

#: Remedy of an import whose digest no longer matches.
_VALIDATE_AGAIN = (
    "validate the configuration again and import with the import_digest that "
    "validation returns; if the file is still being written (an export in "
    "progress), wait until it is complete first"
)

#: Most locations a refusal lists.
_MAX_LISTED_LOCATIONS = 3

#: Bytes hashed per read when a recorded file is verified before decoding.
_HASH_CHUNK = 4 * 1024 * 1024


class ImportRefused(ValueError):
    """An import the service refuses.

    Attributes:
        code: What was refused (``validate_again``, ``import_digest_malformed``,
            ``import_id_malformed``, ``signal_id_invalid``,
            ``signal_id_taken``, ``source_unavailable``,
            ``determinism_failure``, ``record_incomplete``; the MCP adapter
            adds ``remote_import_refused``, ``mixed_modes`` and
            ``incomplete_request``).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _shown(text: str) -> str:
    """File text quoted inside a refusal: escaped and capped."""
    shown = FileText.of(text, cap=LABEL_CHARS)
    return f"'{shown.file_text}{'...' if shown.truncated else ''}'"


def _named(location: str) -> str:
    """A recorded location in a refusal: its file name, as file text."""
    return _shown(PurePath(location).name)


def _package_version() -> str:
    try:
        return metadata.version("predictive-maintenance-mcp")
    except metadata.PackageNotFoundError:
        return "unknown"


def _validate_again_message(validated: Validated) -> str:
    result = validated.result
    if result.outcome != "ready":
        return (
            f"This configuration is not ready to import: validation now returns "
            f"'{result.outcome}'. Nothing was written. Resolve what "
            f"validate_signal_import reports, then {_VALIDATE_AGAIN}."
        )
    return (
        "The import_digest does not match what this configuration and file "
        "validate to now: the file, a value, a basis, an override, the "
        "confirmation channel or the transport changed since it was validated. "
        f"Nothing was written; {_VALIDATE_AGAIN}."
    )


def _under(raw: str, root: Union[str, Path]) -> Optional[str]:
    """*raw* relative to *root* when the lexical gate places it there."""
    try:
        return gate_import_path(raw, [root], base=config.DATA_DIR).relative
    except PathRefused:
        return None


def _default_base(raw: str) -> str:
    """The path-derived id base: the path relative to the data directory, or
    relative to the import root with the root folder's name first, its parts
    joined by ``_`` and the suffix dropped."""
    relative = _under(raw, config.DATA_DIR)
    parts: list[str]
    if relative is not None:
        parts = list(PurePath(relative).with_suffix("").parts)
    else:
        gated = gate_import_path(raw, config.get_import_roots(), base=config.DATA_DIR)
        parts = [
            PurePath(gated.root).name,
            *PurePath(gated.relative).with_suffix("").parts,
        ]
    return "_".join(part for part in parts if part)


def _signal_ids(raw: str, columns: list[int], prefix: Optional[str]) -> dict[int, str]:
    if prefix is None:
        base = _default_base(raw)
        try:
            validate_name_component(base, kind="signal_id")
        except ValueError:
            raise ImportRefused(
                "signal_id_invalid",
                "The file's path holds characters a signal id cannot hold, so "
                "no default signal id can be formed from it. Pass "
                "signal_id_prefix=... made of letters, digits, '_', '-' or "
                "'.'; each channel is stored as <prefix>_c<index>. Nothing was "
                "written.",
            ) from None
    else:
        base = prefix
    return {column: f"{base}_c{column}" for column in columns}


def _check_prefix(prefix: Optional[str]) -> Optional[str]:
    if prefix is None:
        return None
    try:
        return validate_name_component(prefix, kind="signal_id_prefix")
    except ValueError:
        raise ImportRefused(
            "signal_id_invalid",
            "signal_id_prefix must be a non-empty name made of letters, digits, "
            "'_', '-' or '.' (no path separators, not '.' or '..'). Nothing was "
            "written.",
        ) from None


def _latest(record: ImportRecord, kind: str) -> Optional[dict[str, Any]]:
    for event in reversed(record.events):
        if event.kind == kind:
            return event.payload
    return None


def _check_import_id(import_id: str) -> None:
    if not is_import_id(import_id):
        raise ImportRefused(
            "import_id_malformed",
            "An import_id is exactly 64 lowercase hexadecimal characters; "
            "get_signal_imports lists them.",
        )


def _identity_field(item: DecodedChannel, key: str) -> Optional[str]:
    """A field of the channel's identity block (None without a binding)."""
    if item.measurement is None:
        return None
    value = item.measurement.get(key)
    return None if value is None else str(value)


def record_locations(record: ImportRecord) -> list[str]:
    """Every recorded location, most recent first, each once."""
    seen: list[str] = []
    for event in reversed(record.events):
        location = event.payload.get("location") if event.kind == "location" else None
        if isinstance(location, str) and location and location not in seen:
            seen.append(location)
    return seen


def _recorded_channels(record: Optional[ImportRecord]) -> dict[int, dict[str, Any]]:
    """The channels of the latest outcome that stored their samples, by
    column (empty when no outcome stored any)."""
    if record is None:
        return {}
    for event in reversed(record.events):
        if event.kind != "outcome":
            continue
        channels = event.payload.get("channels")
        stored = {
            item["column"]: item
            for item in channels or []
            if isinstance(item, dict)
            and type(item.get("column")) is int
            and isinstance(item.get("signal_id"), str)
            and item.get("materialized") is True
        }
        if stored:
            return stored
    return {}


def _bases_for(attestation: Mapping[str, Any], column: int) -> dict[str, str]:
    """The basis of every file-level value and of this channel's values."""
    prefix = f"channels[{column}]"
    return {
        field: entry["basis"]
        for field, entry in sorted(attestation.get("bases", {}).items())
        if not field.startswith(("channels[", "time_columns["))
        or field.startswith(prefix + ".")
        or field == prefix
    }


def _overrides_for(attestation: Mapping[str, Any], column: int) -> list[str]:
    prefix = f"channels[{column}]"
    return sorted(
        field
        for field in attestation.get("overrides", {})
        if not field.startswith("channels[") or field.startswith(prefix + ".")
    )


def _sha256_of(file: Any) -> str:
    hasher = hashlib.sha256()
    for chunk in file.iter_chunks(_HASH_CHUNK):
        hasher.update(chunk)
    return hasher.hexdigest()


class SignalImportService:
    """Import, re-materialize and list signal imports.

    Args:
        repository: The signal store the channels go into.
        records: The import record store; None reads ``PMM_IMPORT_DIR`` at
            each call.
        ledger: The asset ledger store; None reads ``PMM_LEDGER_DIR`` at each
            call.
    """

    def __init__(
        self,
        repository: SignalRepository,
        *,
        records: Optional[ImportRecordStore] = None,
        ledger: Optional[LedgerStore] = None,
    ) -> None:
        self.repository = repository
        self._records = records
        self._ledger = ledger

    @property
    def records(self) -> ImportRecordStore:
        if self._records is not None:
            return self._records
        return ImportRecordStore(config.get_import_dir())

    @property
    def ledger(self) -> LedgerStore:
        if self._ledger is not None:
            return self._ledger
        return LedgerStore(config.get_ledger_dir())

    # -- import --------------------------------------------------------------

    def import_(
        self,
        path: str,
        configuration: Mapping[str, Any],
        import_digest: str,
        *,
        channel: str,
        transport: Optional[str],
        overwrite: bool = False,
        signal_id_prefix: Optional[str] = None,
    ) -> ImportResult:
        """Import a validated configuration.

        Args:
            path: The file, as it was validated.
            configuration: The configuration, as it was validated.
            import_digest: The digest validation returned with ``ready``.
            channel: The confirmation channel, set by the calling adapter.
            transport: The MCP transport, when the channel is an MCP one.
            overwrite: Replace stored signals with the same ids.
            signal_id_prefix: Store each channel as ``<prefix>_c<index>``
                instead of the path-derived id. A repeat of a recorded import
                keeps the ids it stored first.

        Returns:
            The :class:`ImportResult`.

        Raises:
            ImportRefused: The digest is malformed or no longer matches (also
                when the outcome is not ``ready``), no valid signal id can be
                formed, or an id is taken by another signal.
            ConfigurationError, PathRefused, TextScanRefused, LedgerError,
            ImportRecordError: As for ``validate_import``, and from the
                record store's writes (``LedgerLockTimeout``,
                ``LedgerWriteError``).
        """
        channel = check_channel(channel)
        transport = check_transport(transport)
        if not is_import_id(import_digest):
            raise ImportRefused(
                "import_digest_malformed",
                "import_digest must be the 64 lowercase hexadecimal characters "
                "validate_signal_import returns with 'ready'. Nothing was written.",
            )
        prefix = _check_prefix(signal_id_prefix)
        validated = validate_for_import(
            path, configuration, channel=channel, transport=transport, mode="full"
        )
        result = validated.result
        decode = validated.decode
        if (
            result.outcome != "ready"
            or result.import_digest != import_digest
            or decode is None
            or result.import_id is None
        ):
            raise ImportRefused("validate_again", _validate_again_message(validated))
        import_id = result.import_id
        location = gate_import_path(
            path, config.get_import_roots(), base=config.DATA_DIR
        ).path

        records = self.records
        try:
            existing: Optional[ImportRecord] = records.read(import_id)
        except ImportRecordMissing:
            existing = None
        stored_before = _recorded_channels(existing)
        columns = [entry.column for entry in decode.channels]
        notices: list[str] = []
        if stored_before and set(stored_before) == set(columns):
            ids = {column: stored_before[column]["signal_id"] for column in columns}
            if prefix is not None and any(
                ids[column] != f"{prefix}_c{column}" for column in columns
            ):
                notices.append(
                    "This interpretation was imported before, so its channels "
                    "keep the signal ids they were stored under first; "
                    "signal_id_prefix was not used."
                )
        else:
            ids = _signal_ids(path, columns, prefix)
            stored_before = {}

        rate = float(validated.interpretation["time_base"]["sampling_rate_hz"])
        units = {
            entry["column"]: entry["unit"]
            for entry in validated.interpretation["channels"]
        }
        labels = {entry.column: entry.label for entry in result.channels}
        caller = is_caller_asserted(channel)
        decoded: list[DecodedChannel] = []
        for entry in decode.channels:
            column = entry.column
            identity = validated.identities.get(column)
            provenance = {
                "import_id": import_id,
                "column": column,
                "position": position_words(column),
                "label": (
                    labels[column].as_dict() if labels.get(column) is not None else None
                ),
                "time_base_source": validated.interpretation["time_base"]["source"],
                "sampling_rate_hz": validated.interpretation["time_base"][
                    "sampling_rate_hz"
                ],
                "quantity": validated.formats[column]["quantity"],
                "unit": units[column],
                "scaling": validated.formats[column]["scaling"],
                "samples_digest": entry.samples_digest,
                "sample_count": entry.sample_count,
                "decoder_contract": decode.contract,
                "confirmation_channel": channel,
                "transport": transport,
                "caller_asserted": caller,
                "bases": _bases_for(validated.attestation, column),
                "overrides": _overrides_for(validated.attestation, column),
            }
            decoded.append(
                DecodedChannel(
                    signal_id=ids[column],
                    filepath=Path(location),
                    samples=np.asarray(entry.samples, dtype=np.float64),
                    sampling_rate=rate,
                    signal_unit=units[column],
                    measurement=None if identity is None else dict(identity),
                    import_provenance=provenance,
                )
            )

        presence = {item.signal_id: self._presence(item) for item in decoded}
        taken = [sid for sid, state in presence.items() if state == "other"]
        if taken and not overwrite and stored_before:
            raise ImportRefused(
                "signal_id_taken",
                f"Signal id(s) {taken}, where this import's channels were stored "
                f"first, now hold other signals. A repeated import keeps those "
                f"ids: pass overwrite=True to replace the other signals. Nothing "
                f"was written.",
            )
        if taken and not overwrite:
            raise ImportRefused(
                "signal_id_taken",
                f"Signal id(s) {taken} already hold other signals. Pass "
                f"overwrite=True to replace them, or signal_id_prefix=... to "
                f"store this import under other ids. Nothing was written.",
            )
        formats = {
            item.signal_id: validated.formats[item.import_provenance["column"]]
            for item in decoded
        }

        # Step 3: the record, before any signal becomes visible.
        write = records.write_interpretation(
            decode.source_sha256, decode.size_bytes, validated.interpretation
        )
        record = existing
        attested = {
            event.payload.get("import_digest")
            for event in (record.events if record is not None else ())
            if event.kind == "attestation"
        }
        if import_digest not in attested:
            records.append_event(
                import_id,
                "attestation",
                {
                    "attestation": validated.attestation,
                    "confirmation_channel": channel,
                    "transport": transport,
                    "caller_asserted": caller,
                    "import_digest": import_digest,
                },
            )
        recorded_locations = record_locations(record) if record is not None else []
        if not recorded_locations or recorded_locations[0] != location:
            records.append_event(import_id, "location", {"location": location})
        if write.created or record is None or _latest(record, "runtime") is None:
            records.append_event(
                import_id,
                "runtime",
                {
                    "package_version": _package_version(),
                    "decoder_contract": decode.contract,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
            )

        # Steps 4 and 5: the store, then the outcome.
        materialized = self._materialize(decoded, presence, overwrite, notices)
        status: ImportStatus = "already_imported" if stored_before else "imported"
        inserted = any(presence[item.signal_id] != "same" for item in decoded)
        if inserted or not stored_before:
            self._append_outcome(records, import_id, status, decoded, materialized)

        # Step 6: the ledger, per bound channel that is stored.
        ledger_outcomes = self._record_in_ledger(
            decoded, materialized, formats, validated.holders, decode.contract
        )
        channels = []
        for item in decoded:
            outcome = ledger_outcomes.get(item.signal_id)
            reason = None
            if not materialized[item.signal_id]:
                reason = (
                    "not stored: another signal took its id during this import; "
                    "repeat the import to store and record it"
                )
            elif outcome is not None:
                reason = outcome["reason"]
            channels.append(
                ImportedChannel(
                    column=item.import_provenance["column"],
                    position=item.import_provenance["position"],
                    signal_id=item.signal_id if materialized[item.signal_id] else None,
                    materialized=materialized[item.signal_id],
                    samples_digest=item.import_provenance["samples_digest"],
                    sample_count=item.import_provenance["sample_count"],
                    measurement_id=_identity_field(item, "measurement_id"),
                    asset_id=_identity_field(item, "asset_id"),
                    ledger_status=None if outcome is None else outcome["ledger_status"],
                    snapshot_status=(
                        None if outcome is None else outcome["snapshot_status"]
                    ),
                    changed=[] if outcome is None else list(outcome["changed"]),
                    reattributed_from=(
                        None if outcome is None else outcome["reattributed_from"]
                    ),
                    reason=reason,
                )
            )
        logger.info(
            "Import %s: %s, %d channel(s), record %s",
            import_id[:12],
            status,
            len(decoded),
            "created" if write.created else "reused",
        )
        return ImportResult(
            status=status,
            import_id=import_id,
            file=result.file,
            record_created=write.created,
            caller_asserted=caller,
            channels=channels,
            notices=notices,
        )

    # -- re-materialization ---------------------------------------------------

    def rematerialize(self, import_id: str, *, overwrite: bool = False) -> ImportResult:
        """Store the channels of a recorded import again, from its record and
        its original file, with no new confirmation.

        Each recorded location passes the path gate under the current import
        roots before any file is touched; the file must have the recorded
        size and SHA-256 before it is decoded, and every channel must
        reproduce its recorded samples digest.

        Args:
            import_id: The import's identity (64 lowercase hex characters).
            overwrite: Replace stored signals with the same ids.

        Returns:
            The :class:`ImportResult` with status ``rematerialized``.

        Raises:
            ImportRefused: A malformed id, a record without a stored outcome,
                no recorded location that still holds the recorded bytes, a
                samples digest that differs, or an id taken by another signal.
            ImportRecordMissing, ImportRecordDamaged, ImportRecordError: The
                record cannot be found or read.
        """
        _check_import_id(import_id)
        records = self.records
        record = records.read(import_id)
        stored = _recorded_channels(record)
        if not stored:
            raise ImportRefused(
                "record_incomplete",
                f"Import {import_id[:12]} has no recorded outcome that stored its "
                f"channels, so there is nothing to restore from it alone. Import "
                f"the file again with validate_signal_import and "
                f"import_signal_file.",
            )
        locations = record_locations(record)
        runtime = _latest(record, "runtime") or {}
        contract = runtime.get("decoder_contract", DECODER_CONTRACT)
        decoder = get_decoder(contract)
        roots = config.get_import_roots()
        failures: list[str] = []
        decode = None
        source_path: Optional[Path] = None
        for location in locations:
            try:
                with open_import_file(
                    location,
                    roots,
                    base=config.DATA_DIR,
                    max_size=config.get_max_signal_size(),
                ) as file:
                    if file.size != record.size_bytes:
                        failures.append(f"{_named(location)} has another size")
                        continue
                    if _sha256_of(file) != record.source_sha256:
                        failures.append(f"{_named(location)} holds other bytes")
                        continue
                    decode = decoder(file, record.interpretation, mode="full")
                    source_path = file.path
            except (PathRefused, TextDecodeError) as refused:
                failures.append(f"{_named(location)}: {refused.code}")
                continue
            break
        if decode is None or source_path is None:
            listed = "; ".join(failures[:_MAX_LISTED_LOCATIONS]) or "none recorded"
            raise ImportRefused(
                "source_unavailable",
                f"Import {import_id[:12]} cannot be restored: no recorded location "
                f"still holds the recorded file ({listed}). Put the original, "
                f"unchanged file back at a recorded location under the data "
                f"directory or a folder listed in PMM_IMPORT_ROOTS, or import it "
                f"again from where it is now.",
            )
        by_column = {entry.column: entry for entry in decode.channels}
        decoded: list[DecodedChannel] = []
        for column, item in sorted(stored.items()):
            entry = by_column.get(column)
            if entry is None or entry.samples_digest != item.get("samples_digest"):
                raise ImportRefused(
                    "determinism_failure",
                    f"Import {import_id[:12]}: channel {column} no longer decodes to "
                    f"its recorded samples digest, although the file's bytes are "
                    f"the recorded ones. Nothing was stored. This build does not "
                    f"reproduce the recorded decode; import the file again.",
                )
            provenance = item.get("provenance")
            decoded.append(
                DecodedChannel(
                    signal_id=item["signal_id"],
                    filepath=source_path,
                    samples=np.asarray(entry.samples, dtype=np.float64),
                    sampling_rate=float(
                        record.interpretation["time_base"]["sampling_rate_hz"]
                    ),
                    signal_unit=next(
                        channel["unit"]
                        for channel in record.interpretation["channels"]
                        if channel["column"] == column
                    ),
                    measurement=item.get("measurement"),
                    import_provenance=dict(provenance) if provenance else {},
                )
            )
        presence = {item.signal_id: self._presence(item) for item in decoded}
        taken = [sid for sid, state in presence.items() if state == "other"]
        if taken and not overwrite:
            raise ImportRefused(
                "signal_id_taken",
                f"Signal id(s) {taken} already hold other signals. Pass "
                f"overwrite=True to replace them. Nothing was stored.",
            )
        notices: list[str] = []
        materialized = self._materialize(decoded, presence, overwrite, notices)
        if any(presence[item.signal_id] != "same" for item in decoded):
            self._append_outcome(
                records, import_id, "rematerialized", decoded, materialized
            )
        notices.extend(self._ledger_disagreements(record, decoded, by_column))
        return ImportResult(
            status="rematerialized",
            import_id=import_id,
            file=FileText.of(source_path.name, cap=LABEL_CHARS),
            record_created=False,
            caller_asserted=bool(decoded[0].import_provenance.get("caller_asserted")),
            channels=[
                ImportedChannel(
                    column=item.import_provenance.get("column", column),
                    position=position_words(column),
                    signal_id=item.signal_id if materialized[item.signal_id] else None,
                    materialized=materialized[item.signal_id],
                    samples_digest=by_column[column].samples_digest,
                    sample_count=by_column[column].sample_count,
                    measurement_id=_identity_field(item, "measurement_id"),
                    asset_id=_identity_field(item, "asset_id"),
                )
                for column, item in zip(sorted(stored), decoded)
            ],
            notices=notices,
        )

    # -- index ----------------------------------------------------------------

    def list_imports(
        self,
        *,
        file_sha256: Optional[str] = None,
        file_name: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> tuple[list[ImportRecordIndexEntry], tuple[str, ...]]:
        """The import index, newest first (see ``ImportRecordStore.list_records``)."""
        return self.records.list_records(
            file_sha256=file_sha256, file_name=file_name, signal_id=signal_id
        )

    def get_import(self, import_id: str) -> ImportRecord:
        """One import record (see ``ImportRecordStore.read``).

        Raises:
            ImportRefused: A malformed id.
            ImportRecordMissing, ImportRecordDamaged, ImportRecordError: As
                the record store raises them.
        """
        _check_import_id(import_id)
        return self.records.read(import_id)

    # -- helpers --------------------------------------------------------------

    def _record_in_ledger(
        self,
        decoded: list[DecodedChannel],
        materialized: dict[str, bool],
        formats: dict[str, dict[str, Any]],
        holders: Mapping[str, list[str]],
        contract: str,
    ) -> dict[str, dict[str, Any]]:
        """Record every stored, bound channel in the asset ledger, each
        snapshot computed from the samples this import decoded. A ledger
        failure is reported per channel and never undoes the store insert;
        repeating the import records the channels not yet recorded."""
        bound = [
            item
            for item in decoded
            if item.measurement is not None and materialized[item.signal_id]
        ]
        if not bound:
            return {}
        infos = [
            {
                "signal_id": item.signal_id,
                "filepath": str(item.filepath),
                "sampling_rate": item.sampling_rate,
                "signal_unit": item.signal_unit,
                "measurement": {**item.measurement, "declared_by": IMPORT_DECLARED_BY},
                "import_format": formats[item.signal_id],
            }
            for item in bound
        ]
        outcomes = record_measurements(
            infos,
            {item.signal_id: item.samples for item in bound},
            store=self.ledger,
            data_dir=config.DATA_DIR,
            import_refs={
                item.signal_id: {
                    "import_id": item.import_provenance["import_id"],
                    "decoder_contract": contract,
                }
                for item in bound
            },
            holders=holders,
        )
        return {outcome["signal_id"]: outcome for outcome in outcomes}

    def _ledger_disagreements(
        self,
        record: ImportRecord,
        decoded: list[DecodedChannel],
        by_column: Mapping[int, Any],
    ) -> list[str]:
        """Where the ledger now records another interpretation of a channel
        than this import: the ledger's declaration stands, and the caller is
        told which channels differ."""
        differing: list[int] = []
        unreadable = False
        views: dict[str, Any] = {}
        for item in decoded:
            identity = item.measurement
            if identity is None:
                continue
            column = int(identity["channel_index"])
            expected = build_import_format(
                record.interpretation,
                column,
                samples_digest=by_column[column].samples_digest,
                sample_count=by_column[column].sample_count,
            )
            asset = identity["asset_id"]
            try:
                if asset not in views:
                    views[asset] = self.ledger.read_view(asset)
            except (LedgerError, OSError):
                unreadable = True
                continue
            slot = views[asset]["measurements"].get(identity["measurement_id"]) or {}
            current = slot.get("current") or {}
            recorded = (current.get("declaration") or {}).get("import_format")
            if recorded is not None and recorded != expected:
                differing.append(column)
        notices = []
        if differing:
            notices.append(
                f"The asset ledger now records another interpretation of the "
                f"channel(s) at indices {differing} than this import, so the "
                f"ledger's declaration stands; the signals restored here carry "
                f"this import's interpretation."
            )
        if unreadable:
            notices.append(
                "The asset ledger could not be read, so the restored channels "
                "were not compared with its declarations."
            )
        return notices

    def _presence(self, item: DecodedChannel) -> Presence:
        """``same`` (this import's channel is stored under its id),
        ``absent`` or ``other`` (another signal holds the id)."""
        try:
            info = self.repository.get_signal_info(item.signal_id)
        except KeyError:
            return "absent"
        stored = info.get("import_provenance") or {}
        if (
            stored.get("import_id") == item.import_provenance.get("import_id")
            and stored.get("samples_digest")
            == item.import_provenance.get("samples_digest")
            and stored.get("column") == item.import_provenance.get("column")
        ):
            return "same"
        return "other"

    def _materialize(
        self,
        decoded: list[DecodedChannel],
        presence: dict[str, Presence],
        overwrite: bool,
        notices: list[str],
    ) -> dict[str, bool]:
        """Insert every channel not already stored, all or none, and say which
        channels are stored afterwards."""
        pending = [item for item in decoded if presence[item.signal_id] != "same"]
        stored = {
            item.signal_id: presence[item.signal_id] == "same" for item in decoded
        }
        for _attempt in range(2):
            if not pending:
                break
            try:
                self.repository.insert_decoded(pending, overwrite=overwrite)
            except SignalIdTaken:
                # A concurrent load took an id after the check: nothing was
                # inserted. Ids now holding this same channel (a concurrent
                # repeat of this import) count as stored; the rest is tried
                # once more only when no other signal holds any of them.
                states = {item.signal_id: self._presence(item) for item in pending}
                for sid, state in states.items():
                    if state == "same":
                        stored[sid] = True
                if any(state == "other" for state in states.values()):
                    notices.append(
                        "Another signal took one of this import's ids while it "
                        "ran, so no channel of it was stored (the store holds "
                        "every channel of an import or none). Repeat the import, "
                        "with overwrite=True or signal_id_prefix=..., to store "
                        "it."
                    )
                    break
                pending = [item for item in pending if states[item.signal_id] != "same"]
                continue
            for item in pending:
                stored[item.signal_id] = True
            break
        return stored

    def _append_outcome(
        self,
        records: ImportRecordStore,
        import_id: str,
        status: ImportStatus,
        decoded: list[DecodedChannel],
        materialized: dict[str, bool],
    ) -> None:
        records.append_event(
            import_id,
            "outcome",
            {
                "status": status,
                "channels": [
                    {
                        "column": item.import_provenance.get("column"),
                        "signal_id": (
                            item.signal_id if materialized[item.signal_id] else None
                        ),
                        "materialized": materialized[item.signal_id],
                        "samples_digest": item.import_provenance.get("samples_digest"),
                        "sample_count": item.import_provenance.get("sample_count"),
                        "measurement": item.measurement,
                        "provenance": item.import_provenance,
                    }
                    for item in decoded
                ],
            },
        )
