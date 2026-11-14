"""Adversarial corpus and end-to-end acceptance of the signal import.

Every fixture of the corpus (``tests/import_corpus.py``, invented layouts,
fixed formulas) is observed through the Python service and through the MCP
tools, and compared with ``tests/fixtures/import_corpus_expected.json``, which
freezes, per fixture, the bytes' SHA-256, the inspection summary, the outcome
of the inspection's draft and, for the fixtures with a configuration, the
validation outcome and the samples digest of every imported channel
(freeze, then verify: the file is written by ``freeze_expected`` and only read
by the tests).

The tests also hold the import's acceptance rules end to end:

- a configuration with any engineering value left without a basis (the
  inspection's draft, which carries none) is never ``ready``;
- a configuration in which the caller supplies every basis reaches ``ready``,
  and its validation, record and ``import_provenance`` mark the bases as
  caller-asserted;
- every imported channel replays, through the asset ledger's own decoder and
  from the original file, to its pinned digest; a 17-digit export decodes bit
  for bit as ``float()`` reads it;
- a marker in every file position and in the file and folder names appears
  only inside ``file_text`` values of every tool result and refusal, and in no
  log line;
- the guided flow on the four-channel European export ends with every channel
  recorded in the ledger and the file byte-identical.

Synthetic data only. Expected outcomes are the code's, frozen; the fixtures'
intended outcomes are the plan's catalog, compared in review, not here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest
from mcp import Client

from import_corpus import (
    CATALOG,
    CORPUS_ACQUIRED,
    CORPUS_ROWS,
    CORPUS_RPM,
    CorpusFixture,
    seventeen_digits,
)
from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.asset_ledger.service import _decode_import
from predictive_maintenance_mcp.asset_ledger.store import LedgerStore
from predictive_maintenance_mcp.path_safety import PathRefused, open_import_file
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from predictive_maintenance_mcp.signal_acquisition.text_decoder import (
    DECODER_CONTRACT,
    get_decoder,
)
from predictive_maintenance_mcp.signal_acquisition.text_scan import TextScanRefused
from predictive_maintenance_mcp.signal_import import (
    ConfigurationError,
    SignalImportService,
    configure_import,
    inspect_signal_file,
    validate_import,
)

EXPECTED_FILE = (
    Path(__file__).resolve().parent / "fixtures" / "import_corpus_expected.json"
)
SERVICE_CHANNEL = "python_api"
STATED = {"basis": "user_stated"}
ASSET = "P-200"


# ---------------------------------------------------------------------------
# Configurations of the configured fixtures (every basis named explicitly)
# ---------------------------------------------------------------------------


def _file_value(line: int) -> dict[str, Any]:
    return {"basis": "user_accepted_file_value", "line": line}


def configuration(
    *,
    dialect: dict[str, Any],
    first: int,
    time_base: dict[str, Any],
    channels: list[dict[str, Any]],
    time_columns: tuple[dict[str, Any], ...] = (),
    defaults: Optional[dict[str, Any]] = None,
    bases: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """A configuration whose every value carries a basis: ``user_stated``
    unless *bases* names another."""
    every = {"time_base": STATED, "dialect": STATED, "data_lines": STATED}
    for channel in channels:
        column = channel["column"]
        every[f"channels[{column}].unit"] = STATED
        if channel.get("scaling"):
            every[f"channels[{column}].scaling"] = STATED
            every[f"channels[{column}].dc_policy"] = STATED
        if channel.get("binding"):
            every[f"channels[{column}].binding"] = STATED
    for key, value in (defaults or {}).items():
        if key != "asset_id" and value is not None:
            every[f"defaults.{key}"] = STATED
    every.update(bases or {})
    return {
        "schema": "pmm-text-import-config/1",
        "interpretation": {
            "schema": "pmm-text-import/1",
            "dialect": dialect,
            "data_lines": {"first": first},
            "time_base": time_base,
            "time_columns": list(time_columns),
            "defaults": defaults or {},
            "channels": channels,
        },
        "attestation": {"bases": every, "overrides": {}},
    }


def _dialect(
    delimiter: str,
    decimal: str,
    header: Optional[int],
    units: Optional[int],
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "decimal": decimal,
        "header_line": header,
        "units_line": units,
    }


def _stated_rate(rate: str = "25600") -> dict[str, Any]:
    return {"source": "stated", "sampling_rate_hz": rate}


def _channel(column: int, unit: str = "g", **extra: Any) -> dict[str, Any]:
    entry = {
        "column": column,
        "role": "vibration",
        "quantity": "acceleration",
        "unit": unit,
    }
    entry.update(extra)
    return entry


def _bound(column: int, point: str, direction: str, **extra: Any) -> dict[str, Any]:
    return _channel(
        column, binding={"measurement_point_id": point, "direction": direction}, **extra
    )


_SECONDS = {"column": 0, "time_unit": "s"}
_MV_SCALING = {
    "source_unit": "mV",
    "sensitivity": {"value": "100", "unit": "mV/g", "source": "datasheet_nominal"},
    "dc_policy": "keep",
}


def four_channel_configuration() -> dict[str, Any]:
    """F05: the file's rate, sensitivity, instant and speed accepted from the
    lines that print them; four bound channels."""
    places = [
        ("motor_de", "horizontal"),
        ("motor_de", "vertical"),
        ("motor_de", "axial"),
        ("motor_nde", "horizontal"),
    ]
    channels = [
        _bound(c, point, direction, unit="g", scaling=copy.deepcopy(_MV_SCALING))
        for c, (point, direction) in zip((1, 2, 3, 4), places)
    ]
    bases: dict[str, Any] = {
        "time_base": _file_value(2),
        "defaults.acquired_at": _file_value(4),
        "defaults.rpm": _file_value(5),
    }
    for c in (1, 2, 3, 4):
        bases[f"channels[{c}].scaling"] = _file_value(3)
    return configuration(
        dialect=_dialect(";", ",", 6, 7),
        first=8,
        time_base={"source": "file_declared", "sampling_rate_hz": "25600"},
        time_columns=(_SECONDS,),
        defaults={"asset_id": ASSET, "acquired_at": CORPUS_ACQUIRED, "rpm": CORPUS_RPM},
        channels=channels,
        bases=bases,
    )


def two_column_configuration() -> dict[str, Any]:
    """F09: every value stated by the caller, two bound channels."""
    return configuration(
        dialect=_dialect(",", ".", 1, None),
        first=2,
        time_base=_stated_rate(),
        defaults={"asset_id": ASSET, "acquired_at": CORPUS_ACQUIRED},
        channels=[_bound(0, "pump_de", "horizontal"), _bound(1, "pump_de", "vertical")],
    )


def _plain_time_configuration(
    dialect: dict[str, Any],
    first: int,
    columns: tuple[int, ...],
    time_column: int = 0,
    **channel_extra: Any,
) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        return configuration(
            dialect=dialect,
            first=first,
            time_base=_stated_rate(),
            time_columns=({"column": time_column, "time_unit": "s"},),
            channels=[_channel(c, **copy.deepcopy(channel_extra)) for c in columns],
        )

    return build


#: The fixtures validated (and, when ready, imported) with a full configuration.
CONFIGURATIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "F05": four_channel_configuration,
    "F06": _plain_time_configuration(_dialect("tab", ".", 1, 2), 3, (1, 2)),
    "F09": two_column_configuration,
    "F11": _plain_time_configuration(_dialect(";", ",", 4, 5, "cp1252"), 6, (1,)),
    "F12": _plain_time_configuration(_dialect(";", ",", 4, 5, "utf-8-sig"), 6, (1,)),
    "F14": _plain_time_configuration(_dialect(";", ",", 5, 6), 7, (1,)),
    "F15": _plain_time_configuration(
        _dialect(";", ",", 1, None), 2, (2, 3), time_column=1
    ),
    "F16": _plain_time_configuration(_dialect(";", ",", 1, None), 2, (1,)),
    "H04b": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H11": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H13": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H17": _plain_time_configuration(
        _dialect(";", ".", 1, None), 2, (1,), scaling=_MV_SCALING
    ),
    "H19": _plain_time_configuration(_dialect(";", ".", 1, 2), 3, (1, 2)),
    "F07": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1, 2)),
    "H03": _plain_time_configuration(_dialect(";", ",", 4, None), 5, (1,)),
    "H04a": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H05a": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H05b": _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,)),
    "H07a": _plain_time_configuration(_dialect(";", ".", 1, None), 2, (1,)),
    "H07b": _plain_time_configuration(_dialect(";", ",", 1, None), 2, (1,)),
    "H12": _plain_time_configuration(_dialect(",", ".", 3, None), 4, (1,)),
    "H15": _plain_time_configuration(
        _dialect(";", ".", 1, 2), 3, (1,), quantity="velocity", unit="mm/s"
    ),
}
CONFIGURATIONS.update(
    {
        fixture_id: _plain_time_configuration(_dialect(",", ".", 1, None), 2, (1,))
        for fixture_id in CATALOG
        if fixture_id.startswith("H06-")
    }
)


def _untimed_configuration(
    dialect: dict[str, Any], first: int, columns: tuple[int, ...]
) -> Callable[[], dict[str, Any]]:
    def build() -> dict[str, Any]:
        return configuration(
            dialect=dialect,
            first=first,
            time_base=_stated_rate(),
            channels=[_channel(c) for c in columns],
        )

    return build


CONFIGURATIONS.update(
    {
        "F08": _untimed_configuration(_dialect("tab", ".", None, None), 1, (0, 1)),
        "H09": _untimed_configuration(_dialect(";", ",", None, None), 1, (0, 1)),
        "H16": _untimed_configuration(_dialect(";", ",", 1, None), 2, (0, 1)),
    }
)


# ---------------------------------------------------------------------------
# Observation: one summary per fixture, from JSON results (service or tool)
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prefix_of(fixture: CorpusFixture) -> str:
    return fixture.fixture_id.lower().replace("-", "_")


def inspection_summary(result: dict[str, Any]) -> dict[str, Any]:
    proposal = result["dialect"]["proposal"] or {}
    return {
        "status": result["status"],
        "encoding": [
            result["file"]["encoding"]["value"],
            result["file"]["encoding"]["status"],
        ],
        "dialect": result["dialect"]["status"],
        "delimiter": proposal.get("delimiter"),
        "decimal": proposal.get("decimal"),
        "data_first_line": proposal.get("data_first_line"),
        "columns_total": result["columns_total"],
        "regions": [region["kind"] for region in result["regions"]],
        "issues": sorted({issue["code"] for issue in result["issues"]}),
    }


def validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": result["outcome"],
        "issues": sorted(
            {f"{issue['code']}:{issue['effect']}" for issue in result["issues"]}
        ),
        "row_count": result["row_count"],
    }


#: The summary of a configuration the service refuses before validating it
#: (a required field missing, an unknown key): never ready either.
CONFIGURATION_REFUSED = {"refused": "configuration"}


def validate_or_refuse(path: Path, conf: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    try:
        result = validate_import(str(path), conf, channel=SERVICE_CHANNEL)
    except ConfigurationError:
        return dict(CONFIGURATION_REFUSED), None
    return validation_summary(result.model_dump(mode="json")), result


def observe(fixture: CorpusFixture, folder: Path) -> dict[str, Any]:
    """Write the fixture under *folder* and observe it through the service."""
    data = fixture.build()
    path = folder / fixture.file_name
    path.write_bytes(data)
    observed: dict[str, Any] = {"sha256": sha256(data), "size": len(data)}
    try:
        inspected = inspect_signal_file(str(path))
    except (PathRefused, TextScanRefused) as refused:
        observed["inspection"] = {
            "refused": getattr(refused, "code", type(refused).__name__)
        }
        return observed
    observed["inspection"] = inspection_summary(inspected.model_dump(mode="json"))
    if inspected.status == "importable":
        draft = configure_import(inspected)
        observed["draft"], _ = validate_or_refuse(path, draft.configuration)
    build = CONFIGURATIONS.get(fixture.fixture_id)
    if build is not None:
        configured, result = validate_or_refuse(path, build())
        if result is not None and result.outcome == "ready":
            imported = SignalImportService(get_repository()).import_(
                str(path),
                build(),
                result.import_digest,
                channel=SERVICE_CHANNEL,
                transport=None,
                signal_id_prefix=prefix_of(fixture),
            )
            configured["import_id"] = imported.import_id
            configured["digests"] = {
                str(channel.column): channel.samples_digest
                for channel in imported.channels
            }
        observed["configured"] = configured
    return observed


def freeze_expected(folder: Path) -> dict[str, Any]:
    """The expected file's content, observed now (used only to write it)."""
    get_repository().clear_all()
    return {
        fixture_id: observe(fixture, folder)
        for fixture_id, fixture in sorted(CATALOG.items())
    }


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    return json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def corpus_root(tmp_path, monkeypatch, sandbox_data_dir) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
    return root


@pytest.fixture
def clean_store():
    repo = get_repository()
    repo.clear_all()
    yield repo
    repo.clear_all()


def server():
    from predictive_maintenance_mcp.server import mcp

    return mcp


async def tool(client: Client, name: str, arguments: dict[str, Any]):
    return await client.call_tool(name, arguments, read_timeout_seconds=300)


async def ok(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await tool(client, name, arguments)
    assert not result.is_error, result.content
    return result.structured_content


# ---------------------------------------------------------------------------
# The frozen corpus
# ---------------------------------------------------------------------------


class TestCorpusIsFrozen:
    def test_every_catalog_fixture_has_an_expected_entry(self, expected):
        assert sorted(expected) == sorted(CATALOG)

    def test_the_generators_are_deterministic(self, expected):
        for fixture_id, fixture in CATALOG.items():
            first, second = fixture.build(), fixture.build()
            assert first == second, fixture_id
            assert sha256(first) == expected[fixture_id]["sha256"], fixture_id

    def test_the_plan_catalog_is_covered(self):
        ids = {fid.split("-")[0].rstrip("ab") for fid in CATALOG}
        catalog = {
            "F01",
            "F02",
            "F05",
            "F06",
            "F07",
            "F08",
            "F09",
            "F11",
            "F12",
            "F13",
            "F14",
            "F15",
            "F16",
        }
        catalog |= {f"H{n:02d}" for n in range(1, 20) if n != 14}
        assert catalog <= ids, sorted(catalog - ids)


@pytest.mark.parametrize("fixture_id", sorted(CATALOG))
def test_each_fixture_reaches_its_frozen_outcome_through_the_service(
    fixture_id, expected, corpus_root, clean_store
):
    assert observe(CATALOG[fixture_id], corpus_root) == expected[fixture_id]


@pytest.mark.anyio
@pytest.mark.parametrize("fixture_id", sorted(CATALOG))
async def test_each_fixture_reaches_its_frozen_outcome_through_mcp(
    fixture_id, expected, corpus_root, clean_store, package_caplog
):
    """The tools return what the service returns, at the same caps."""
    fixture = CATALOG[fixture_id]
    want = expected[fixture_id]
    path = corpus_root / fixture.file_name
    path.write_bytes(fixture.build())
    async with Client(server(), mode="legacy") as client:
        inspected = await tool(client, "inspect_signal_file", {"filepath": str(path)})
        if "refused" in want["inspection"]:
            assert inspected.is_error, inspected.structured_content
            return
        assert not inspected.is_error, inspected.content
        body = inspected.structured_content
        assert inspection_summary(body["inspection"]) == want["inspection"]
        if "draft" in want:
            draft = await tool(
                client,
                "validate_signal_import",
                {"filepath": str(path), "config": body["draft"]["configuration"]},
            )
            if want["draft"] == CONFIGURATION_REFUSED:
                assert draft.is_error, draft.structured_content
            else:
                assert not draft.is_error, draft.content
                assert validation_summary(draft.structured_content) == want["draft"]
        if "configured" in want:
            conf = CONFIGURATIONS[fixture_id]()
            checked = await tool(
                client,
                "validate_signal_import",
                {"filepath": str(path), "config": conf},
            )
            if want["configured"] == CONFIGURATION_REFUSED:
                assert checked.is_error, checked.structured_content
                return
            assert not checked.is_error, checked.content
            validated = checked.structured_content
            summary = validation_summary(validated)
            assert summary == {k: want["configured"][k] for k in summary}
            if validated["outcome"] == "ready":
                assert validated["caller_asserted"] is True
                imported = await ok(
                    client,
                    "import_signal_file",
                    {
                        "filepath": str(path),
                        "config": conf,
                        "import_digest": validated["import_digest"],
                        "signal_id_prefix": prefix_of(fixture),
                    },
                )
                assert imported["import_id"] == want["configured"]["import_id"]
                assert {
                    str(c["column"]): c["samples_digest"] for c in imported["channels"]
                } == want["configured"]["digests"]
    assert all("Traceback" not in r.getMessage() for r in package_caplog.records)


# ---------------------------------------------------------------------------
# Acceptance rules over the corpus
# ---------------------------------------------------------------------------


def test_no_value_without_a_basis_is_ever_ready(expected):
    """The inspection's draft carries no basis: over the whole corpus its
    validation is never ready."""
    drafts = {
        fid: entry["draft"].get("outcome", "refused")
        for fid, entry in expected.items()
        if "draft" in entry
    }
    assert drafts, "no importable fixture in the corpus"
    assert "ready" not in drafts.values(), drafts


def test_the_corpus_reaches_every_outcome(expected):
    configured = {
        entry["configured"].get("outcome", "refused")
        for entry in expected.values()
        if "configured" in entry
    }
    assert {"ready", "needs_confirmation", "invalid"} <= configured, configured
    assert any("refused" in entry["inspection"] for entry in expected.values())
    assert any(
        entry["inspection"].get("status") == "not_importable"
        for entry in expected.values()
    )


#: Every basis of the complete, caller-stated configuration of F09.
F09_BASES = sorted(two_column_configuration()["attestation"]["bases"])
#: Values the whole-file parse itself proves for F09 (one reading, the
#: configured one, an unambiguous encoding, the header row where inspection
#: finds it, no stated last line): they need no basis while proven, and do
#: once the proof fails (the test after the sweep).
PROVEN_BY_PARSE = ("data_lines", "dialect")
ENGINEERING_BASES = [key for key in F09_BASES if key not in PROVEN_BY_PARSE]


@pytest.mark.parametrize("missing", ENGINEERING_BASES)
def test_removing_any_one_basis_from_a_ready_configuration_is_never_ready(
    missing, corpus_root
):
    """Live, not frozen: the complete configuration is ready, and without any
    one of its engineering bases (time base, unit, binding, acquisition
    instant) it is refused or not ready."""
    fixture = CATALOG["F09"]
    path = corpus_root / fixture.file_name
    path.write_bytes(fixture.build())
    complete = two_column_configuration()
    full, _ = validate_or_refuse(path, complete)
    assert full["outcome"] == "ready", full
    reduced = copy.deepcopy(complete)
    del reduced["attestation"]["bases"][missing]
    summary, _ = validate_or_refuse(path, reduced)
    assert summary.get("outcome") != "ready", (missing, summary)


def test_the_basis_removal_covers_every_value_kind():
    """The removal sweep reaches the time base, a default, and each channel's
    unit and binding; only the parse-proven values are left to the next test."""
    assert {"time_base", "defaults.acquired_at"} <= set(ENGINEERING_BASES)
    for column in (0, 1):
        assert {f"channels[{column}].unit", f"channels[{column}].binding"} <= set(
            ENGINEERING_BASES
        )
    assert set(F09_BASES) == set(ENGINEERING_BASES) | set(PROVEN_BY_PARSE)


def _state_last_line(conf: dict[str, Any]) -> None:
    conf["interpretation"]["data_lines"]["last"] = 1 + CORPUS_ROWS


def _drop_header_line(conf: dict[str, Any]) -> None:
    conf["interpretation"]["dialect"]["header_line"] = None


@pytest.mark.parametrize(
    "path, unprove",
    [("data_lines", _state_last_line), ("dialect", _drop_header_line)],
    ids=["data_lines", "dialect"],
)
def test_a_parse_proven_value_needs_a_basis_once_the_proof_fails(
    path, unprove, corpus_root
):
    fixture = CATALOG["F09"]
    file_path = corpus_root / fixture.file_name
    file_path.write_bytes(fixture.build())
    proven = two_column_configuration()
    del proven["attestation"]["bases"][path]
    summary, _ = validate_or_refuse(file_path, proven)
    assert summary["outcome"] == "ready", summary
    unproven = copy.deepcopy(proven)
    unprove(unproven)
    summary, _ = validate_or_refuse(file_path, unproven)
    assert summary.get("outcome") != "ready", summary
    restored = copy.deepcopy(unproven)
    restored["attestation"]["bases"][path] = dict(STATED)
    summary, _ = validate_or_refuse(file_path, restored)
    assert summary["outcome"] == "ready", summary


#: Fixtures the frozen file records as imported (their channels have digests).
IMPORTED = sorted(
    fixture_id
    for fixture_id, entry in json.loads(
        EXPECTED_FILE.read_text(encoding="utf-8")
    ).items()
    if "digests" in entry.get("configured", {})
)


def test_the_imported_fixtures_are_the_ready_ones(expected):
    ready = sorted(
        fid
        for fid, entry in expected.items()
        if entry.get("configured", {}).get("outcome") == "ready"
    )
    assert IMPORTED == ready and len(IMPORTED) >= 10, IMPORTED


@pytest.mark.parametrize("fixture_id", IMPORTED)
def test_every_imported_channel_replays_through_the_ledger_decoder(
    fixture_id, expected, corpus_root, clean_store
):
    """Every channel replays from the import record's interpretation through
    the decoder of its contract; a bound channel also replays from its ledger
    declaration through the ledger's own replay."""
    want = expected[fixture_id]["configured"]
    fixture = CATALOG[fixture_id]
    observed = observe(fixture, corpus_root)
    assert observed["configured"]["digests"] == want["digests"]
    path = corpus_root / fixture.file_name
    service = SignalImportService(get_repository())
    record = service.get_import(want["import_id"])
    bound = {
        channel["column"]: channel
        for channel in record.interpretation["channels"]
        if channel.get("binding")
    }
    with open_import_file(
        str(path),
        [corpus_root],
        base=path.parent,
        max_size=config.get_max_signal_size(),
    ) as file:
        replayed = get_decoder(DECODER_CONTRACT)(
            file, record.interpretation, mode="full"
        )
    assert {
        str(channel.column): channel.samples_digest for channel in replayed.channels
    } == want["digests"]
    store = LedgerStore(config.get_ledger_dir())
    for column, digest in want["digests"].items():
        column = int(column)
        if column not in bound:
            continue
        view = store.read_view(ASSET)
        current = next(
            m["current"]
            for m in view["measurements"].values()
            if m["current"]["declaration"].get("channel_index") == column
            and m["current"]["declaration"]["import_format"]["samples_digest"] == digest
        )
        samples = _decode_import(
            path,
            current["declaration"],
            import_ref=current["declaration"].get("import_ref")
            or current.get("import_ref"),
            roots=[corpus_root],
        )
        assert sha256(np.asarray(samples, dtype="<f8").tobytes()) == digest


def test_seventeen_digit_values_decode_as_float_reads_them(expected):
    want = expected["H11"]["configured"]
    assert want["outcome"] == "ready"
    rows = seventeen_digits().decode("ascii").splitlines()[1:]
    values = np.array([float(row.split(",")[1]) for row in rows], dtype="<f8")
    assert len(values) == CORPUS_ROWS
    assert sha256(values.tobytes()) == want["digests"]["1"]


# ---------------------------------------------------------------------------
# Every basis supplied by the caller: ready, and marked caller-asserted
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_every_basis_from_the_caller_is_ready_and_caller_asserted(
    corpus_root, clean_store
):
    fixture = CATALOG["F09"]
    path = corpus_root / fixture.file_name
    path.write_bytes(fixture.build())
    conf = two_column_configuration()
    assert {b["basis"] for b in conf["attestation"]["bases"].values()} == {
        "user_stated"
    }
    async with Client(server(), mode="legacy") as client:
        validated = await ok(
            client, "validate_signal_import", {"filepath": str(path), "config": conf}
        )
        assert validated["outcome"] == "ready"
        assert validated["caller_asserted"] is True
        assert all(
            s["caller_asserted"] for s in validated["statements"] if s.get("basis")
        )
        imported = await ok(
            client,
            "import_signal_file",
            {
                "filepath": str(path),
                "config": conf,
                "import_digest": validated["import_digest"],
                "signal_id_prefix": "f09",
            },
        )
        assert imported["caller_asserted"] is True
        record = await ok(
            client, "get_signal_imports", {"import_id": imported["import_id"]}
        )
        attestations = record["record"]["attestations"]
        assert attestations and all(a["caller_asserted"] is True for a in attestations)
        assert all(
            a["confirmation_channel"] == "mcp_tool_argument" for a in attestations
        )
        for channel in imported["channels"]:
            info = await ok(
                client, "get_signal_info", {"signal_id": channel["signal_id"]}
            )
            provenance = info["import_provenance"]
            assert provenance["caller_asserted"] is True, provenance
            assert set(provenance["bases"].values()) == {"user_stated"}, provenance


# ---------------------------------------------------------------------------
# AE17: a marker in every file position and in the file and folder names
# ---------------------------------------------------------------------------

MARKER = "QX MARK QX"


def outside_file_text(value: Any) -> list[str]:
    """Strings holding the marker anywhere but inside a ``file_text`` value."""
    found: list[str] = []
    if isinstance(value, dict):
        if set(value) == {"file_text", "truncated"}:
            return found
        for item in value.values():
            found += outside_file_text(item)
    elif isinstance(value, list):
        for item in value:
            found += outside_file_text(item)
    elif isinstance(value, str) and "QX" in value:
        found.append(value)
    return found


def marked_export(marker_in: set[str]) -> bytes:
    """The four-channel European layout with the marker in the positions
    named: ``preamble``, ``label``, ``unit`` (an unselected column),
    ``trailer``, ``data`` (a field of channel 1), ``time`` (a stamp)."""
    lines = european_lines()
    if "preamble" in marker_in:
        lines.insert(1, f"Note;{MARKER}")
    names, units = (6, 7) if "preamble" in marker_in else (5, 6)
    lines[names] += f";extra {MARKER}" if "label" in marker_in else ";extra"
    lines[units] += f";{MARKER}" if "unit" in marker_in else ";g"
    for index in range(units + 1, len(lines)):
        lines[index] += ";0"
    if "data" in marker_in:
        fields = lines[units + 50].split(";")
        fields[1] = MARKER
        lines[units + 50] = ";".join(fields)
    if "time" in marker_in:
        fields = lines[units + 60].split(";")
        fields[0] = MARKER
        lines[units + 60] = ";".join(fields)
    if "trailer" in marker_in:
        lines.append(f"end;{MARKER}")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def european_lines() -> list[str]:
    from import_corpus import european_mv_export

    return european_mv_export(rows=1024, channels=3).decode("utf-8").split("\r\n")[:-1]


@pytest.mark.anyio
async def test_ae17_the_marker_appears_only_in_file_text_and_never_in_logs(
    tmp_path, monkeypatch, sandbox_data_dir, clean_store, package_caplog
):
    root = tmp_path / f"exports {MARKER}"
    root.mkdir()
    monkeypatch.setenv("PMM_IMPORT_ROOTS", str(root))
    clean = root / f"run {MARKER}.csv"
    clean.write_bytes(marked_export({"preamble", "label", "trailer"}))
    # An unrecognized unit token keeps the units row from being read as one,
    # so the unit position has a file of its own (inspected and validated).
    in_unit = root / f"unit {MARKER}.csv"
    in_unit.write_bytes(marked_export({"preamble", "unit"}))
    in_data = root / f"data {MARKER}.csv"
    in_data.write_bytes(marked_export({"preamble", "data"}))
    in_time = root / f"time {MARKER}.csv"
    in_time.write_bytes(marked_export({"preamble", "time"}))

    conf = four_channel_configuration()
    interpretation = conf["interpretation"]
    interpretation["channels"] = interpretation["channels"][:3]
    for key in [k for k in conf["attestation"]["bases"] if k.startswith("channels[4]")]:
        del conf["attestation"]["bases"][key]
    interpretation["dialect"].update(header_line=7, units_line=8)
    interpretation["data_lines"] = {"first": 9, "last": 8 + 1024}
    for key, line in (
        ("time_base", 3),
        ("defaults.acquired_at", 5),
        ("defaults.rpm", 6),
    ):
        conf["attestation"]["bases"][key] = _file_value(line)
    for c in (1, 2, 3):
        conf["attestation"]["bases"][f"channels[{c}].scaling"] = _file_value(4)

    responses: list[Any] = []
    refusals: list[str] = []
    async with Client(server(), mode="legacy") as client:
        for path in (clean, in_unit, in_data, in_time):
            responses.append(
                await ok(client, "inspect_signal_file", {"filepath": str(path)})
            )
        unit_result = await ok(
            client, "validate_signal_import", {"filepath": str(in_unit), "config": conf}
        )
        assert unit_result["outcome"] != "ready"
        responses.append(unit_result)
        validated = await ok(
            client, "validate_signal_import", {"filepath": str(clean), "config": conf}
        )
        responses.append(validated)
        assert validated["outcome"] == "ready", [
            (i["code"], i["message"])
            for i in validated["issues"]
            if i["effect"] != "notice"
        ]
        imported = await ok(
            client,
            "import_signal_file",
            {
                "filepath": str(clean),
                "config": conf,
                "import_digest": validated["import_digest"],
                "signal_id_prefix": "run",
            },
        )
        responses.append(imported)
        responses.append(await ok(client, "get_signal_imports", {}))
        responses.append(
            await ok(client, "get_signal_imports", {"import_id": imported["import_id"]})
        )
        responses.append(await ok(client, "get_signal_info", {"signal_id": "run_c1"}))
        responses.append(await ok(client, "list_signals", {"scope": "memory"}))
        for path in (in_data, in_time):
            result = await ok(
                client,
                "validate_signal_import",
                {"filepath": str(path), "config": conf},
            )
            assert result["outcome"] == "invalid", result["outcome"]
            assert any(
                issue["token"] for issue in result["issues"]
            ), "the token is not shown as file text"
            responses.append(result)
        for arguments in (
            {"filepath": str(in_data), "config": conf, "import_digest": "0" * 64},
            {"filepath": str(clean), "config": conf, "import_digest": "0" * 64},
        ):
            refusal = await tool(client, "import_signal_file", arguments)
            assert refusal.is_error
            refusals.append(refusal.content[0].text)
    # Positive control: the marker did reach the results, inside file_text.
    assert (
        sum("QX" in json.dumps(response) for response in responses)
        >= len(responses) // 2
    )
    for response in responses:
        assert outside_file_text(response) == []
    assert refusals and all("QX" not in text for text in refusals), refusals
    assert package_caplog.records, "no log line was captured"
    assert all("QX" not in record.getMessage() for record in package_caplog.records)


def test_the_marker_check_finds_a_marker_outside_file_text():
    """Negative control of the AE17 check: a marker in the server's own words
    is found; the same marker inside a file_text value is not."""
    inside = {"label": {"file_text": f"acc {MARKER}", "truncated": False}}
    outside = {"issues": [{"message": f"column {MARKER} refused"}]}
    assert outside_file_text(inside) == []
    assert outside_file_text(outside) == [f"column {MARKER} refused"]
    assert outside_file_text({"nested": [inside, outside]}) == [
        f"column {MARKER} refused"
    ]


# ---------------------------------------------------------------------------
# The guided flow on the four-channel European export
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_the_four_channel_guided_flow_records_every_channel_and_edits_nothing(
    corpus_root, clean_store
):
    fixture = CATALOG["F05"]
    path = corpus_root / fixture.file_name
    path.write_bytes(fixture.build())
    before = sha256(path.read_bytes())
    async with Client(server(), mode="legacy") as client:
        inspected = await ok(client, "inspect_signal_file", {"filepath": str(path)})
        assert inspected["inspection"]["status"] == "importable"
        assert inspected["inspection"]["required_confirmations"]
        # The draft carries no basis: refused as incomplete, or not ready.
        draft = await tool(
            client,
            "validate_signal_import",
            {"filepath": str(path), "config": inspected["draft"]["configuration"]},
        )
        assert draft.is_error or draft.structured_content["outcome"] != "ready"
        conf = four_channel_configuration()
        validated = await ok(
            client, "validate_signal_import", {"filepath": str(path), "config": conf}
        )
        assert validated["outcome"] == "ready"
        imported = await ok(
            client,
            "import_signal_file",
            {
                "filepath": str(path),
                "config": conf,
                "import_digest": validated["import_digest"],
            },
        )
        assert [c["column"] for c in imported["channels"]] == [1, 2, 3, 4]
        assert all(c["materialized"] for c in imported["channels"])
        assert all(c["ledger_status"] == "recorded" for c in imported["channels"])
        history = await ok(client, "get_asset_history", {"asset_id": ASSET})
        recorded = {c["measurement_id"] for c in imported["channels"]}
        listed = json.dumps(history)
        assert all(mid in listed for mid in recorded)
    assert sha256(path.read_bytes()) == before
    assert sorted(p.name for p in corpus_root.iterdir()) == [fixture.file_name]
