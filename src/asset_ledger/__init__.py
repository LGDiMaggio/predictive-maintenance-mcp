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
