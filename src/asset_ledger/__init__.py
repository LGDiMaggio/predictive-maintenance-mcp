"""
ISO 13374 Block 3 — State Detection (asset ledger).

Append-only, per-asset ledger of declared measurements, measurement-point
declarations, healthy baselines and derived health snapshots, persisted
locally as one JSONL file per asset under ``config.get_ledger_dir()``.
Nothing here leaves the machine.

Dependency direction: ``mcp_tools -> asset_ledger -> signal_acquisition ->
{config, path_safety}``. No module of this package imports
``signal_acquisition.repository``, ``models`` or anything MCP; the
measurement contract it builds on lives in
``signal_acquisition.measurement``. Later units add the store, snapshot,
comparability, assessment and service modules and re-export their public
functions from here.
"""

from .store import (  # noqa: F401
    SCHEMA_VERSION,
    PRODUCER_NAME,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    EVENT_HEALTH_SNAPSHOT_COMPUTED,
    EVENT_BASELINE_DECLARED,
    EVENT_TYPES,
    MEASUREMENT_INDEX_NAME,
    LedgerError,
    LedgerLockTimeout,
    LedgerWriteError,
    AppendResult,
    LedgerReadResult,
    canonical_json,
    content_hash,
    short_id,
    compute_event_id,
    make_event,
    LedgerStore,
    build_asset_view,
)

from .snapshot import (  # noqa: F401
    ALGORITHM_VERSION,
    BEARING_LABELS,
    ENVELOPE_BAND_DEFAULT,
    PROCESSING_FAMILY,
    SNAPSHOT_BLOCKS,
    SNAPSHOT_PROVENANCE_KEYS,
    SnapshotPolicy,
    collect_provenance,
    compute_health_snapshot,
    context_digest,
    expected_frequencies,
    policy_params,
    processing_id,
    resolve_context,
    snapshot_id,
)

from .comparability import (  # noqa: F401
    COMPARABILITY_GRADES,
    DEFAULT_THRESHOLDS,
    EXCLUDING_CODES,
    INFORMATIONAL_CODES,
    QUALIFICATION_CODES,
    ComparabilityThresholds,
    assess_measurement_comparability,
    build_reference_context,
    grade_of,
    summarize_comparability,
    unit_conversion_factor,
)
