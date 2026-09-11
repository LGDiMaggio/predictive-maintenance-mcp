"""MCP tools for the asset health ledger (ISO 13374 Block 3, change over time).

Four tools over the local append-only asset ledger: declare the context of
a measurement point, declare which measurements are its healthy baseline,
read the history of an asset (or the index of the assets) and assess the
change of a point against its reference. Every tool resolves the ledger
directory at call time (``LedgerStore(get_ledger_dir())``) and the data
directory as ``config.DATA_DIR``, and hands both to the service by
argument; nothing here computes a snapshot or reads a signal file.

Misuse (invalid ids, out-of-range limits, empty declarer, an id of another
point) raises ``ValueError`` naming the problem and the remedy; legitimate
negative outcomes (unknown asset or point, insufficient history, no
homogeneous lineage) are typed results discriminated by ``status``.

Logging note
------------
Every tool here takes a ``ctx`` parameter it never uses. That is deliberate,
not leftover: ``tests/fixtures/tool_inventory.json`` pins ``context_kwarg``
per tool, so dropping the parameter would be a protocol-visible change to
the tool surface.

What changed in 0.12.0 is *how* progress is emitted, not whether tools
accept a context. SEP-2577 deprecated the MCP logging capability with no
in-protocol replacement, so narration goes to this module's logger, which
``server.configure_logging`` binds to stderr (stdout is the stdio
transport's JSON-RPC channel). Clients no longer receive progress
notifications; any fact a caller needs is carried by the return value.
"""

import logging
from typing import Any, Literal, Optional

from mcp.server.mcpserver import MCPServer, Context

from .. import config
from ..asset_ledger import service as ledger_service
from ..asset_ledger.assessment import (
    MIN_REFERENCE_MEASUREMENTS,
    AssessmentParams,
    assess_change,
    validate_params,
)
from ..asset_ledger.store import LedgerStore
from ..config import get_ledger_dir
from ..models import (
    AssetChangeAssessment,
    AssetHistoryResult,
    BaselineDeclarationResult,
    MeasurementPointDeclarationResult,
)
from ..signal_acquisition.measurement import validate_ledger_id

logger = logging.getLogger(__name__)

#: Ledgers read (and assets listed) at most by the index form of
#: ``get_asset_history``; the ids beyond it are counted, never read.
MAX_INDEX_ASSETS = 50

#: Upper bound of ``max_measurements`` in ``get_asset_history``.
MAX_HISTORY_MEASUREMENTS = 200

#: Upper bound of ``last_k`` in ``assess_asset_change`` (the slots listed
#: for drill-down and scanned for bearing evidence).
MAX_LAST_K = 50

#: Upper bound of ``reference_measurements`` in ``assess_asset_change``.
MAX_REFERENCE_MEASUREMENTS = 100


def _check_range(name: str, value: object, low: int, high: int) -> int:
    """Refuse a limit outside ``[low, high]`` naming both bounds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(
            f"{name} must be an integer between {low} and {high}, got {value!r}; "
            f"pass a value in that range."
        )
    return value


# ------------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------------


async def declare_measurement_point(
    ctx: Context,
    asset_id: str,
    measurement_point_id: str,
    bearing_id: Optional[str] = None,
    fault_orders: Optional[dict[str, float]] = None,
    machine_group: Optional[Literal[1, 2]] = None,
    support_type: Optional[Literal["rigid", "flexible"]] = None,
    machine_power_kw: Optional[float] = None,
    expected_signal_unit: Optional[Literal["g", "m/s2", "mm/s", "m/s"]] = None,
    expected_sensor_id: Optional[str] = None,
    expected_direction: Optional[
        Literal["horizontal", "vertical", "axial", "x", "y", "z"]
    ] = None,
    nominal_rpm: Optional[float] = None,
    declared_by: Optional[str] = None,
    note: Optional[str] = None,
) -> MeasurementPointDeclarationResult:
    """Declare the context of a measurement point in the local asset ledger.

    The declaration is what the health snapshots and diagnose_vibration
    default to for every measurement of the point: the bearing at the
    point (bearing_id from the verified catalog, or fault_orders as
    multiples of the shaft frequency: BPFO, BPFI, BSF, FTF), the design
    speed nominal_rpm (distinct from the rpm a measurement declares, which
    wins when present), the ISO 20816-3 machine_group and support_type
    (undeclared means no ISO block, never a silent default), the rated
    power, and what the point expects of its measurements: signal unit,
    sensor and direction (a measurement contradicting them is excluded
    from the trend, one omitting them is qualified).

    Versioned per point: a re-declaration that changes nothing appends
    nothing and returns the current version; one that changes a value
    appends the next version and reports the changed keys. Measurements
    whose snapshot was computed with a previous context are counted in
    measurements_with_stale_context together with the exact
    assess_asset_change(..., reprocess=True) call that recomputes them (a
    note-only change makes nothing stale).

    Args:
        ctx: MCP context. Unused, see this module's docstring on logging.
        asset_id: The asset (letters, digits, '_', '-', '.', starting with
            a letter or digit; case-sensitive).
        measurement_point_id: The point on the asset (same grammar).
        bearing_id: Catalog designation of the bearing at the point.
        fault_orders: {label: order} with labels BPFO, BPFI, BSF, FTF and
            orders in multiples of the shaft frequency (e.g. BPFO 3.58).
        machine_group: ISO 20816-3 group, 1 (large) or 2 (medium).
        support_type: 'rigid' or 'flexible'.
        machine_power_kw: Rated power in kW (positive).
        expected_signal_unit: Unit the point's measurements are expected in.
        expected_sensor_id: Sensor expected at the point (free text, at
            most 200 characters, one line).
        expected_direction: Expected measurement direction.
        nominal_rpm: Design speed of the point in rev/min (positive).
        declared_by: Who declares (free text, at most 200 characters).
        note: Free-text note (at most 200 characters, one line).

    Returns:
        MeasurementPointDeclarationResult with the version, whether an
        event was appended, the changed keys, the stale-snapshot count and
        remedy, the recorded declaration and a summary message.

    Raises:
        ValueError: Invalid ids, a value outside its vocabulary, a
            non-positive number, free text over the limit or with control
            characters (one message names every problem), or a ledger
            that cannot be read or written.
    """
    result = ledger_service.declare_measurement_point(
        store=LedgerStore(get_ledger_dir()),
        asset_id=asset_id,
        measurement_point_id=measurement_point_id,
        bearing_id=bearing_id,
        fault_orders=fault_orders,
        machine_group=machine_group,
        support_type=support_type,
        machine_power_kw=machine_power_kw,
        expected_signal_unit=expected_signal_unit,
        expected_sensor_id=expected_sensor_id,
        expected_direction=expected_direction,
        nominal_rpm=nominal_rpm,
        declared_by=declared_by,
        note=note,
    )
    logger.info(result["message"])
    return MeasurementPointDeclarationResult(
        asset_id=result["asset_id"],
        measurement_point_id=result["measurement_point_id"],
        declaration_version=result["declaration_version"],
        appended=result["appended"],
        changed=result["changed"],
        previous_version=result["previous_version"],
        measurements_with_stale_context=result["measurements_with_stale_context"],
        remedy=result["remedy"],
        declaration=result["declaration"],
        event_id=result["event_id"],
        bearing_in_catalog=result["bearing_in_catalog"],
        message=result["message"],
    )


async def declare_healthy_baseline(
    ctx: Context,
    asset_id: str,
    measurement_point_id: str,
    measurement_ids: list[str],
    declared_by: str,
    note: Optional[str] = None,
) -> BaselineDeclarationResult:
    """Declare which recorded measurements are the healthy reference of a point.

    Until a baseline is declared, assess_asset_change compares against the
    first comparable acquisitions of the point and says so
    (health_declared False); a declared baseline is the only reference
    reported as declared healthy, and every assessment cites declared_by,
    the date and the note verbatim. The ids come from load_signal or
    get_asset_history (measurement_id); every member must be a recorded
    measurement of THIS point, comparable or qualified against the point's
    current declaration and against the other members (a contradicting
    unit, direction or speed is refused with the reason), and no two may
    share an acquisition slot. At least 3 members. An empty
    measurement_ids withdraws the active baseline and requires a note;
    later assessments fall back to the automatic window and name the
    withdrawal. Each member records the declaration versions it was
    validated against, so a later re-declaration excludes it with a
    qualification instead of silently changing the reference.

    Args:
        ctx: MCP context. Unused, see this module's docstring on logging.
        asset_id: The asset (ledger id).
        measurement_point_id: The point (ledger id).
        measurement_ids: measurement_id values of the members (at least 3,
            at most 100), or an empty list to withdraw.
        declared_by: Who declares (required, free text, at most 200
            characters, one line); quoted verbatim in later assessments.
        note: Free-text note (at most 200 characters); required for a
            withdrawal.

    Returns:
        BaselineDeclarationResult with the baseline_id, the members with
        their recorded versions, the superseded baseline and a summary.

    Raises:
        ValueError: Invalid ids or free text, an empty declared_by, an
            asset without a ledger (naming the known assets), an id that is
            not a measurement of the point (naming the valid ids), fewer
            than 3 members, two members in one acquisition slot, a member
            that is not comparable (naming the reason), a withdrawal
            without a note or without an active baseline, or a ledger that
            cannot be read or written.
    """
    result = ledger_service.declare_healthy_baseline(
        store=LedgerStore(get_ledger_dir()),
        asset_id=asset_id,
        measurement_point_id=measurement_point_id,
        measurement_ids=measurement_ids,
        declared_by=declared_by,
        note=note,
    )
    logger.info(result["message"])
    return BaselineDeclarationResult(
        asset_id=result["asset_id"],
        measurement_point_id=result["measurement_point_id"],
        baseline_id=result["baseline_id"],
        measurement_ids=result["measurement_ids"],
        members=result["members"],
        declared_by=result["declared_by"],
        note=result["note"],
        declared_at=result["declared_at"],
        superseded_baseline_id=result["superseded_baseline_id"],
        withdrawn=result["withdrawn"],
        event_id=result["event_id"],
        message=result["message"],
    )


async def get_asset_history(
    ctx: Context,
    asset_id: Optional[str] = None,
    measurement_point_id: Optional[str] = None,
    max_measurements: int = 20,
) -> AssetHistoryResult:
    """Read the asset ledger: the index of the assets, or one asset's history.

    Without asset_id (the index): every asset the ledger directory lists,
    at most 50, each with its points (measurement counts, first and last
    acquired_at, latest processing lineage, whether a baseline is
    declared, declaration version), event count, ledger size and
    integrity counters; truncated says whether more assets exist. With
    asset_id: that asset's history read from its ledger alone, the last
    max_measurements measurements newest first (identity, acquired_at,
    signal_id, file location, declaration version, lineages, a preview of
    the indicators of the latest snapshot, comparability grade and codes
    against the point), the point declarations and baselines with their
    history, the measurements re-attributed to another asset, and the
    integrity block. measurement_point_id restricts the history to one
    point and needs asset_id. An unknown asset or point is a typed
    'not_found' naming the known ids, not an exception.

    Args:
        ctx: MCP context. Unused, see this module's docstring on logging.
        asset_id: The asset (ledger id), or None for the index.
        measurement_point_id: Restrict to one point of the asset, or None.
        max_measurements: Measurements listed at most (1 to 200), newest
            first.

    Returns:
        AssetHistoryResult with status 'index', 'found' or 'not_found'.

    Raises:
        ValueError: measurement_point_id without asset_id, an invalid id,
            max_measurements outside 1..200, or a ledger that cannot be
            read.
    """
    if asset_id is None and measurement_point_id is not None:
        raise ValueError(
            "measurement_point_id needs an asset_id: a point is identified within "
            "its asset. Pass asset_id (call get_asset_history() with no arguments "
            "to list the assets and their points)."
        )
    limit = _check_range(
        "max_measurements", max_measurements, 1, MAX_HISTORY_MEASUREMENTS
    )
    store = LedgerStore(get_ledger_dir())

    if asset_id is None:
        index = ledger_service.asset_index(store, max_assets=MAX_INDEX_ASSETS)
        listed = [str(entry["asset_id"]) for entry in index["assets"]]
        message = (
            f"{index['asset_count']} asset(s) in the ledger, {len(listed)} listed"
            + (
                f" (the first {MAX_INDEX_ASSETS}; pass asset_id to read one beyond)"
                if index["truncated"]
                else ""
            )
            + "."
        )
        logger.info(message)
        return AssetHistoryResult(
            status="index",
            assets=index["assets"],
            asset=None,
            known_assets=listed,
            known_points=[],
            suggestion=None,
            truncated=bool(index["truncated"]),
            message=message,
        )

    validate_ledger_id(asset_id, kind="asset_id")
    if measurement_point_id is not None:
        validate_ledger_id(measurement_point_id, kind="measurement_point_id")
    history = ledger_service.asset_history(
        store,
        asset_id,
        measurement_point_id=measurement_point_id,
        max_measurements=limit,
    )
    if history is None:
        known = store.list_assets()
        message = f"Asset {asset_id!r} has no ledger; known assets: {known or 'none'}."
        logger.info(message)
        return AssetHistoryResult(
            status="not_found",
            assets=[],
            asset=None,
            known_assets=known,
            known_points=[],
            suggestion=(
                "Use one of the known assets, or load a measurement whose companion "
                f'declares "asset_id": "{asset_id}".'
            ),
            truncated=False,
            message=message,
        )
    known_points = [str(point) for point in history["known_points"]]
    if not history["point_found"]:
        message = (
            f"Measurement point {measurement_point_id!r} is not declared for asset "
            f"{asset_id!r} and no measurement names it; known points: "
            f"{known_points or 'none'}."
        )
        logger.info(message)
        return AssetHistoryResult(
            status="not_found",
            assets=[],
            asset=None,
            known_assets=[],
            known_points=known_points,
            suggestion=(
                "Use one of the known points, declare the point via "
                "declare_measurement_point, or load a measurement whose companion "
                f'declares "measurement_point_id": "{measurement_point_id}".'
            ),
            truncated=False,
            message=message,
        )
    listed_count = len(history["measurements"])
    message = (
        f"Asset {asset_id!r}: {history['event_count']} ledger event(s), "
        f"{history['summary']['measurement_count']} recorded measurement(s) over "
        f"{history['summary']['point_count']} point(s); {listed_count} of "
        f"{history['measurement_count']} listed newest first"
        + (f" for point {measurement_point_id!r}" if measurement_point_id else "")
        + "."
    )
    logger.info(message)
    return AssetHistoryResult(
        status="found",
        assets=[],
        asset=history,
        known_assets=[],
        known_points=known_points,
        suggestion=None,
        truncated=bool(history["truncated"]),
        message=message,
    )


async def assess_asset_change(
    ctx: Context,
    asset_id: str,
    measurement_point_id: str,
    acquired_since: Optional[str] = None,
    acquired_until: Optional[str] = None,
    last_k: int = 5,
    reference_measurements: int = 10,
    reprocess: bool = False,
) -> AssetChangeAssessment:
    """Assess the change of one measurement point against its reference.

    Needs nothing but the asset and the point: the measurements, their
    snapshots and the declarations are read from the ledger. The reference
    is the active declared baseline when one exists (health_declared True,
    declarer cited), else the first reference_measurements comparable
    acquisitions of the point, reported as a relative comparison whose
    health is not declared. For every amplitude indicator (rms, peak, 1x,
    ISO velocity, envelope amplitude at each bearing fault frequency) the
    reference band is mean plus or minus max(3 sigma, 25 percent) and the
    post-reference acquisitions are classified as no_change,
    isolated_episode, unconfirmed_single_acquisition or persistent_change
    with the criterion spelled out; bearing evidence is counted over the
    last_k acquisitions; the overall verdict is the most severe. Every
    comparability qualification is reported and non-comparable
    acquisitions are listed, never silently dropped. The result carries at
    most one suggested_verification, no list of recommendations.

    Snapshots are comparable only on one processing lineage. When no
    lineage covers every evaluated acquisition the status is
    processing_not_homogeneous and the remedy is this call with
    reprocess=True, which first recomputes the stale snapshots with the
    current lineage: idempotent (a repeated call recomputes nothing that
    is already current), bounded to 10 measurements per call (reference
    members first, then the most recent), verified against the recorded
    file hash, and the reprocess block names the exact next call while
    measurements remain. Old snapshots are never deleted.

    Args:
        ctx: MCP context. Unused, see this module's docstring on logging.
        asset_id: The asset (ledger id).
        measurement_point_id: The point (ledger id).
        acquired_since: ISO 8601 lower bound of the assessed post-reference
            acquisitions (the reference is always used), or None.
        acquired_until: ISO 8601 upper bound, or None.
        last_k: Acquisitions listed for drill-down and scanned for bearing
            evidence (1 to 50).
        reference_measurements: Size of the automatic reference window
            when no baseline is declared (3 to 100).
        reprocess: Recompute up to 10 stale snapshots with the current
            lineage before assessing.

    Returns:
        AssetChangeAssessment with status 'assessed', 'not_found',
        'insufficient_history' or 'processing_not_homogeneous'.

    Raises:
        ValueError: Invalid ids, last_k or reference_measurements outside
            their range, a bound that is not ISO 8601, or a ledger that
            cannot be read.
    """
    validate_ledger_id(asset_id, kind="asset_id")
    validate_ledger_id(measurement_point_id, kind="measurement_point_id")
    _check_range("last_k", last_k, 1, MAX_LAST_K)
    _check_range(
        "reference_measurements",
        reference_measurements,
        MIN_REFERENCE_MEASUREMENTS,
        MAX_REFERENCE_MEASUREMENTS,
    )
    params = AssessmentParams(
        reference_measurements=reference_measurements,
        last_k=last_k,
        acquired_since=acquired_since,
        acquired_until=acquired_until,
    )
    validate_params(params)
    store = LedgerStore(get_ledger_dir())

    reprocess_block: Optional[dict[str, Any]] = None
    if reprocess:
        reprocess_block = ledger_service.reprocess_stale_snapshots(
            asset_id,
            measurement_point_id,
            store=store,
            data_dir=config.DATA_DIR,
            params=params,
        )
        logger.info(reprocess_block["message"])

    verdict = assess_change(
        store.read_view(asset_id),
        asset_id,
        measurement_point_id,
        params=params,
        known_assets=store.list_assets(),
    )
    logger.info(
        "Change assessment of %s/%s: %s",
        asset_id,
        measurement_point_id,
        verdict["status"],
    )
    return AssetChangeAssessment(
        status=verdict["status"],
        asset_id=asset_id,
        measurement_point_id=measurement_point_id,
        reference=verdict.get("reference"),
        lineage=verdict.get("lineage"),
        observed=verdict.get("observed"),
        derived=verdict.get("derived"),
        assessed=verdict.get("assessed"),
        comparability=verdict.get("comparability"),
        suggested_verification=verdict.get("suggested_verification"),
        remedy=verdict.get("remedy"),
        suggestion=verdict.get("suggestion"),
        known_assets=list(verdict.get("known_assets") or []),
        known_points=list(verdict.get("known_points") or []),
        available=verdict.get("available"),
        required=verdict.get("required"),
        lineages=verdict.get("lineages"),
        evaluated_slots=verdict.get("evaluated_slots"),
        missing_for_current=verdict.get("missing_for_current"),
        current_processing_id=verdict.get("current_processing_id"),
        reprocess=reprocess_block,
        message=str(verdict["message"]),
    )


def register(mcp: MCPServer) -> None:
    """Register the asset health ledger tools on *mcp*."""
    mcp.tool()(declare_measurement_point)
    mcp.tool()(declare_healthy_baseline)
    mcp.tool()(get_asset_history)
    mcp.tool()(assess_asset_change)
