"""Tests for the import configuration, attestation and validation.

Files are synthesized into a sandboxed data directory and every claim runs
against real bytes through the real path gate, the strict decoder and the
time-base assessment. Ledger pre-check scenarios write their events directly
through ``LedgerStore`` in the ``import_format`` layout, because the import
itself records nothing yet.
"""

import copy
import os
import re
from pathlib import Path
from typing import Any, Optional

import import_corpus as corpus
import pytest

from predictive_maintenance_mcp import config
from predictive_maintenance_mcp.asset_ledger.store import (
    EVENT_BASELINE_DECLARED,
    EVENT_MEASUREMENT_POINT_DECLARED,
    EVENT_MEASUREMENT_RECORDED,
    LedgerStore,
    make_event,
    short_id,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    build_import_format,
    import_raw_marker,
)
from predictive_maintenance_mcp.signal_acquisition.import_config import (
    import_id as compute_import_id,
)
from predictive_maintenance_mcp.signal_acquisition.measurement import (
    measurement_id_from_digest,
)
from predictive_maintenance_mcp.signal_import import (
    ConfigurationError,
    configure_import,
    inspect_signal_file,
    normalize_configuration,
    validate_import,
)
from predictive_maintenance_mcp.signal_import import validation as validation_module
from predictive_maintenance_mcp.signal_import.validation import (
    _history as history_module,
)
from predictive_maintenance_mcp.signal_import.attestation import (
    check_transport,
    import_digest,
)
from predictive_maintenance_mcp.signal_import.records import (
    ImportRecordError,
    ImportRecordStore,
)

RATE = 25600
ASSET = "P-101"
POINT = "motor_de_h"
ACQUIRED = "2026-08-20T11:42:00+00:00"


@pytest.fixture
def write(sandbox_data_dir):
    def _write(name: str, data: bytes) -> Path:
        path = sandbox_data_dir / name
        path.write_bytes(data)
        return path

    return _write


def rows(
    count: int,
    *,
    decimals: int = 6,
    channels: int = 1,
    gap_at: Optional[int] = None,
    sep: str = ",",
) -> list[str]:
    """Time column plus *channels* waveform columns; *gap_at* drops the
    sample of that row index, so the next stamp is two intervals on."""
    out = []
    for row in range(count):
        tick = row + (1 if gap_at is not None and row >= gap_at else 0)
        values = [f"{tick / RATE:.{decimals}f}"] + [
            f"{corpus.sample(row, c):.6f}" for c in range(channels)
        ]
        out.append(sep.join(values))
    return out


def interpretation(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "pmm-text-import/1",
        "dialect": {
            "encoding": "utf-8",
            "delimiter": ",",
            "decimal": ".",
            "header_line": 1,
        },
        "data_lines": {"first": 2},
        "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
        "time_columns": [{"column": 0, "time_unit": "s"}],
        "channels": [
            {"column": 1, "role": "vibration", "quantity": "acceleration", "unit": "g"}
        ],
    }
    for key, value in changes.items():
        base[key] = value
    return base


def configuration(
    interp: dict[str, Any],
    bases: Optional[dict[str, Any]] = None,
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return {
        "schema": "pmm-text-import-config/1",
        "interpretation": interp,
        "attestation": {"bases": bases or {}, "overrides": overrides or {}},
    }


def standard_bases(**extra: Any) -> dict[str, Any]:
    bases: dict[str, Any] = {
        "time_base": {"basis": "user_stated"},
        "channels[1].unit": {"basis": "user_accepted_file_value", "line": 1},
    }
    bases.update(extra)
    return bases


def standard_file(
    write, name: str = "run.csv", count: int = 400, header: str = "time [s],acc [g]"
) -> Path:
    return write(name, corpus.lines_to_bytes([header] + rows(count)))


def validate(path: Path, conf: dict[str, Any], channel: str = "python_api", **kwargs):
    return validate_import(str(path), conf, channel=channel, **kwargs)


def codes(result, effect: Optional[str] = None) -> list[str]:
    return [i.code for i in result.issues if effect is None or i.effect == effect]


def tree(*roots: Path) -> dict[str, tuple[bytes, int]]:
    snapshot = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                snapshot[str(path)] = (path.read_bytes(), path.stat().st_mtime_ns)
    return snapshot


# ---------------------------------------------------------------------------
# Acceptance examples
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_complete_configuration_is_ready_with_one_statement_per_value(self, write):
        path = standard_file(write)
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "ready", result.issues
        assert [s.field for s in result.statements] == [
            "time_base",
            "dialect",
            "data_lines",
            "channels[1].unit",
        ]
        canonical = normalize_configuration(
            configuration(interpretation(), standard_bases())
        )
        expected_id = compute_import_id(
            result.source_sha256, result.size_bytes, canonical["interpretation"]
        )
        assert result.import_id == expected_id
        assert result.import_digest == import_digest(
            expected_id, canonical["attestation"], channel="python_api", transport=None
        )
        assert result.row_count == 400 and result.first_line == 2
        assert result.time_columns[0].outcome == "accepted"

    def test_derived_rate_without_a_basis_asks_the_user(
        self, write, ledger_dir, import_dir
    ):
        path = standard_file(write)
        before = tree(path.parent, ledger_dir, import_dir)
        bases = standard_bases()
        del bases["time_base"]
        result = validate(
            path, configuration(interpretation(), bases), channel="mcp_tool_argument"
        )
        assert result.outcome == "needs_confirmation"
        missing = [i for i in result.issues if i.code == "missing_basis"]
        assert [i.field for i in missing] == ["time_base"]
        assert (
            "sampling rate" in missing[0].message
            and "ask the user" in missing[0].message
        )
        assert result.import_digest is None
        assert tree(path.parent, ledger_dir, import_dir) == before

    def test_four_decimal_time_column_cannot_set_the_rate(self, write):
        data = corpus.lines_to_bytes(["time [s],acc [g]"] + rows(400, decimals=4))
        path = write("coarse.csv", data)
        interp = interpretation(
            time_base={
                "source": "time_column",
                "time_column": 0,
                "sampling_rate_hz": "25600",
            }
        )
        bases = standard_bases(
            time_base={"basis": "user_accepted_derived_value", "column": 0}
        )
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "needs_confirmation"
        found = [i for i in result.issues if i.code == "time_resolution_insufficient"]
        assert len(found) == 1 and "cannot prove uniform sampling" in found[0].message
        assert result.time_columns[0].proposed_rate_hz is None

    def test_missing_sample_refuses_with_its_line_and_step(self, write):
        data = corpus.lines_to_bytes(["time [s],acc [g]"] + rows(120100, gap_at=120002))
        path = write("gap.csv", data)
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "invalid"
        gap = [i for i in result.issues if i.code == "time_gap"]
        assert len(gap) == 1 and gap[0].line == 120004
        # One missing sample is a step of two intervals; the 6-decimal stamps
        # measure it as 1.9968, and the message reports what was measured.
        step = re.search(
            r"The first measures ([0-9.]+) sample intervals\.", gap[0].message
        )
        assert step is not None and abs(float(step.group(1)) - 2) < 0.01
        assert "first at line 120004." in gap[0].message
        assert (
            "interpolat" not in gap[0].message and "fill the gap" not in gap[0].message
        )

    def test_gap_in_a_time_column_not_chosen_as_time_base(self, write):
        data = corpus.lines_to_bytes(
            ["Sampling rate: 25600", "time [s],acc [g]"] + rows(300100, gap_at=300001)
        )
        path = write("gap2.csv", data)
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
            time_base={"source": "file_declared", "sampling_rate_hz": "25600"},
        )
        bases = {
            "time_base": {"basis": "user_accepted_file_value", "line": 1},
            "channels[1].unit": {"basis": "user_accepted_file_value", "line": 2},
        }
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "invalid"
        gap = [i for i in result.issues if i.code == "time_gap"]
        assert [(i.line, i.column) for i in gap] == [(300004, 0)]

    def test_millivolt_column_stored_in_g_without_sensitivity(self, write):
        path = standard_file(write, header="time [s],Acc [mV]")
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "invalid"
        assert "contradicts_file_declaration" in codes(result, "invalid")

    def test_sensitivity_on_a_physical_unit_is_double_scaling(self, write):
        path = standard_file(write)
        interp = interpretation(
            channels=[
                {
                    "column": 1,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "scaling": {
                        "source_unit": "mV",
                        "sensitivity": {
                            "value": "100",
                            "unit": "mV/g",
                            "source": "datasheet_nominal",
                        },
                        "dc_policy": "keep",
                    },
                }
            ]
        )
        result = validate(path, configuration(interp, standard_bases()))
        assert result.outcome == "invalid"
        assert "double_scaling" in codes(result, "invalid")

    def test_tachometer_column_is_never_imported_as_vibration(self, write):
        path = standard_file(write, header="time [s],speed [rpm]")
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "invalid"
        assert "unit_token_preview_only" in codes(result, "invalid")
        inspected = inspect_signal_file(str(path))
        assert [c.role_candidate for c in inspected.columns if c.index == 1] == [
            "tachometer"
        ]

    def test_unrecognized_unit_token_needs_an_override(self, write):
        path = standard_file(write, header="time [s],Acc [mg]")
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        refused = validate(path, configuration(interpretation(), bases))
        assert refused.outcome == "invalid"
        assert "contradicts_file_declaration" in codes(refused, "invalid")
        overridden = validate(
            path,
            configuration(
                interpretation(),
                bases,
                overrides={"channels[1].unit": "the logger writes mg for g"},
            ),
        )
        assert overridden.outcome == "ready", overridden.issues
        assert "contradicts_file_declaration" in codes(overridden, "notice")
        statement = next(
            s for s in overridden.statements if s.field == "channels[1].unit"
        )
        assert "override with a reason is recorded" in statement.text
        canonical = normalize_configuration(
            configuration(
                interpretation(),
                bases,
                {"channels[1].unit": "the logger writes mg for g"},
            )
        )
        assert canonical["attestation"]["overrides"] == {
            "channels[1].unit": "the logger writes mg for g"
        }


# ---------------------------------------------------------------------------
# Plan scenarios beyond the acceptance examples
# ---------------------------------------------------------------------------


class TestBases:
    def test_rpm_without_a_basis_and_with_a_cited_preamble_speed(self, write):
        data = corpus.lines_to_bytes(["Speed: 1780", "time [s],acc [g]"] + rows(400))
        path = write("speed.csv", data)
        dialect = {
            "encoding": "utf-8",
            "delimiter": ",",
            "decimal": ".",
            "header_line": 2,
        }
        bases = standard_bases(
            **{"channels[1].unit": {"basis": "user_accepted_file_value", "line": 2}}
        )

        def interp(rpm: str) -> dict[str, Any]:
            return interpretation(
                dialect=dialect, data_lines={"first": 3}, defaults={"rpm": rpm}
            )

        without = validate(path, configuration(interp("1780"), bases))
        assert without.outcome == "needs_confirmation"
        assert [i.field for i in without.issues if i.code == "missing_basis"] == [
            "defaults.rpm"
        ]
        cited = {"defaults.rpm": {"basis": "user_accepted_file_value", "line": 1}}
        matching = validate(path, configuration(interp("1780"), {**bases, **cited}))
        assert matching.outcome == "ready", matching.issues
        wrong = validate(path, configuration(interp("1800"), {**bases, **cited}))
        assert wrong.outcome == "invalid"
        assert [i.field for i in wrong.issues if i.code == "basis_mismatch"] == [
            "defaults.rpm"
        ]

    def test_sensitivity_and_instant_accepted_from_the_preamble(self, write):
        data = corpus.lines_to_bytes(
            ["Sensitivity: 100 mV/g", f"Start time: {ACQUIRED}", "time [s],acc [mV]"]
            + rows(400)
        )
        path = write("scaled.csv", data)

        def interp(value: str, acquired: str) -> dict[str, Any]:
            return interpretation(
                dialect={
                    "encoding": "utf-8",
                    "delimiter": ",",
                    "decimal": ".",
                    "header_line": 3,
                },
                data_lines={"first": 4},
                defaults={"acquired_at": acquired},
                channels=[
                    {
                        "column": 1,
                        "role": "vibration",
                        "quantity": "acceleration",
                        "unit": "g",
                        "scaling": {
                            "source_unit": "mV",
                            "sensitivity": {
                                "value": value,
                                "unit": "mV/g",
                                "source": "datasheet_nominal",
                            },
                            "dc_policy": "keep",
                        },
                    }
                ],
            )

        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[1].unit": {"basis": "user_stated"},
            "channels[1].scaling": {"basis": "user_accepted_file_value", "line": 1},
            "channels[1].dc_policy": {"basis": "user_stated"},
            "defaults.acquired_at": {"basis": "user_accepted_file_value", "line": 2},
        }
        accepted = validate(path, configuration(interp("100", ACQUIRED), bases))
        assert accepted.outcome == "ready", accepted.issues
        unit = next(s for s in accepted.statements if s.field == "channels[1].unit")
        assert "converted from millivolts by its sensitivity" in unit.text
        scaling = next(
            s for s in accepted.statements if s.field == "channels[1].scaling"
        )
        # The basis (accepted from line 1, which validation checked) follows
        # the sensitivity it covers; the declared source and the configured
        # source unit are said as such, never under the file's basis.
        assert (
            "has the sensitivity 100 mV/g, accepted from the file (line 1); its "
            "declared source is a datasheet nominal value." in scaling.text
        )
        assert (
            "Its samples, in millivolts as configured, are divided by 100 mV per g."
            in scaling.text
        )
        assert (scaling.basis, scaling.origin) == (
            "user_accepted_file_value",
            "file_declared",
        )
        assert (unit.basis, unit.origin) == ("user_stated", "stated")
        naive = ACQUIRED.replace("+00:00", "")
        refused = validate(path, configuration(interp("98", naive), bases))
        assert sorted(
            i.field for i in refused.issues if i.code == "basis_mismatch"
        ) == [
            "channels[1].scaling",
            "defaults.acquired_at",
        ]
        wrong_line = {
            **bases,
            "channels[1].scaling": {"basis": "user_accepted_file_value", "line": 2},
            "channels[1].dc_policy": {"basis": "user_accepted_file_value", "line": 1},
            "defaults.acquired_at": {"basis": "user_accepted_file_value", "line": 1},
        }
        nowhere = validate(path, configuration(interp("100", ACQUIRED), wrong_line))
        messages = {
            i.field: i.message for i in nowhere.issues if i.code == "basis_mismatch"
        }
        assert "prints no sensitivity" in messages["channels[1].scaling"]
        assert "the file states no dc policy" in messages["channels[1].dc_policy"]
        assert "prints no acquired at" in messages["defaults.acquired_at"]

    @pytest.mark.parametrize(
        ("time_base", "basis", "says"),
        [
            (
                {
                    "source": "time_column",
                    "time_column": 0,
                    "sampling_rate_hz": "25600",
                },
                {"basis": "user_stated"},
                "citing that column",
            ),
            (
                {"source": "file_declared", "sampling_rate_hz": "25600"},
                {"basis": "user_stated"},
                "no sampling rate or sample interval is printed",
            ),
            (
                {"source": "stated", "sampling_rate_hz": "25600"},
                {"basis": "user_accepted_derived_value", "column": 0},
                "A stated rate takes the basis 'user_stated'",
            ),
            (
                {"source": "stated", "sampling_rate_hz": "25600"},
                {"basis": "user_accepted_prior_value", "import_id": "d" * 64},
                "the cited earlier import of these bytes used",
            ),
        ],
    )
    def test_time_base_basis_must_fit_its_source(self, write, time_base, basis, says):
        path = standard_file(write)
        result = validate(
            path,
            configuration(
                interpretation(time_base=time_base), standard_bases(time_base=basis)
            ),
        )
        assert result.outcome == "invalid"
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["time_base"] and says in mismatch[
            0
        ].message

    def test_three_missing_bases_are_named_in_one_result(self, write):
        data = corpus.lines_to_bytes(["Speed: 1780", "time [s],acc [g]"] + rows(400))
        path = write("three.csv", data)
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
            defaults={"rpm": "1780"},
        )
        result = validate(path, configuration(interp, {}))
        assert result.outcome == "needs_confirmation"
        missing = [i for i in result.issues if i.code == "missing_basis"]
        assert sorted(i.field for i in missing) == [
            "channels[1].unit",
            "defaults.rpm",
            "time_base",
        ]
        assert all("ask the user" in i.message for i in missing)
        assert len(result.required_confirmations) == 3

    def test_accepted_file_value_must_equal_the_cited_line(self, write):
        data = corpus.lines_to_bytes(
            ["Sampling rate: 25600", "time [s],acc [g]"] + rows(400)
        )
        path = write("cited.csv", data)
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
            time_base={"source": "file_declared", "sampling_rate_hz": "25000"},
            time_columns=[],
        )
        bases = {
            "time_base": {"basis": "user_accepted_file_value", "line": 1},
            "channels[1].unit": {"basis": "user_accepted_file_value", "line": 2},
        }
        result = validate(
            path,
            configuration(
                interp,
                bases,
                overrides={"time_columns[0]": "stamps from a software clock"},
            ),
        )
        assert result.outcome == "invalid"
        assert [i.field for i in result.issues if i.code == "basis_mismatch"] == [
            "time_base"
        ]

    def test_interval_preamble_and_stated_rate_outside_it_disagree(self, write):
        # 0.0000390 s carries a half-unit of 0.5e-7 s: rates of 25608 to 25674 Hz.
        data = corpus.lines_to_bytes(
            ["Delta_X: 0.0000390", "acc [g]"]
            + [f"{corpus.sample(r, 0):.6f}" for r in range(400)]
        )
        path = write("delta.csv", data)
        interp = {
            "schema": "pmm-text-import/1",
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            "data_lines": {"first": 3},
            "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
            "channels": [
                {
                    "column": 0,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                }
            ],
        }
        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[0].unit": {"basis": "user_accepted_file_value", "line": 2},
        }
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "needs_confirmation"
        assert "rate_sources_disagree" in codes(result, "needs_confirmation")
        chosen = validate(
            path,
            configuration(
                interp,
                bases,
                overrides={"time_base": "the logger's interval is rounded"},
            ),
        )
        assert chosen.outcome == "ready", chosen.issues
        assert "rate_sources_disagree" in codes(chosen, "notice")

    def test_stated_g_on_a_millivolt_column_with_an_override(self, write):
        path = standard_file(write, header="time [s],Acc [mV]")
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        overrides = {
            "channels[1].unit": "the channel is already converted by the logger"
        }
        result = validate(path, configuration(interpretation(), bases, overrides))
        assert result.outcome == "ready", result.issues
        assert "contradicts_file_declaration" in codes(result, "notice")
        assert (
            normalize_configuration(configuration(interpretation(), bases, overrides))[
                "attestation"
            ]["overrides"]
            == overrides
        )

    def test_caller_asserted_bases_are_marked(self, write):
        path = standard_file(write)
        result = validate(
            path,
            configuration(interpretation(), standard_bases()),
            channel="mcp_tool_argument",
            transport="stdio",
        )
        assert result.outcome == "ready"
        assert result.caller_asserted
        based = [s for s in result.statements if s.basis is not None]
        assert based and all(s.caller_asserted for s in based)
        assert all("as the caller asserted" in s.text for s in based)
        python = validate(path, configuration(interpretation(), standard_bases()))
        assert python.import_digest != result.import_digest  # the channel is bound


class TestStatements:
    def test_statements_name_index_and_position_and_never_the_label(self, write):
        path = standard_file(write, header="when [s],Accel X [g]")
        result = validate(path, configuration(interpretation(), standard_bases()))
        statement = next(s for s in result.statements if s.field == "channels[1].unit")
        assert "Index 1, the 2nd field from the left" in statement.text
        assert (
            statement.label is not None and statement.label.file_text == "Accel X [g]"
        )
        assert [t.file_text for t in statement.head_tokens] == [
            f"{corpus.sample(r, 0):.6f}" for r in range(3)
        ]
        for every in result.statements:
            assert "Accel" not in every.text and "when" not in every.text

    def test_mismatched_header_shows_no_label(self, write):
        data = corpus.lines_to_bytes(["acc [g]"] + rows(400))
        path = write("short.csv", data)
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        result = validate(path, configuration(interpretation(), bases))
        statement = next(s for s in result.statements if s.field == "channels[1].unit")
        assert statement.label is None
        assert statement.head_tokens  # the first tokens still travel as file text
        assert "field_count_mismatch" in codes(result, "notice")


class TestChannels:
    def test_two_channels_bound_to_one_measurement_collide(self, write):
        data = corpus.lines_to_bytes(["time [s],x [g],y [g]"] + rows(400, channels=2))
        path = write("two.csv", data)
        binding = {"measurement_point_id": POINT, "direction": "horizontal"}
        interp = interpretation(
            defaults={"asset_id": ASSET, "acquired_at": ACQUIRED},
            channels=[
                {
                    "column": c,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "binding": binding,
                }
                for c in (1, 2)
            ],
        )
        bases = {
            "time_base": {"basis": "user_stated"},
            "defaults.acquired_at": {"basis": "user_stated"},
            **{
                f"channels[{c}].unit": {"basis": "user_accepted_file_value", "line": 1}
                for c in (1, 2)
            },
            **{f"channels[{c}].binding": {"basis": "user_stated"} for c in (1, 2)},
        }
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "invalid"
        assert "binding_collision" in codes(result, "invalid")

    def test_voltage_channel_needs_a_dc_policy(self, write):
        path = standard_file(write, header="time [s],Acc [mV]")

        def interp(dc_policy: Optional[str]) -> dict[str, Any]:
            return interpretation(
                channels=[
                    {
                        "column": 1,
                        "role": "vibration",
                        "quantity": "acceleration",
                        "unit": "g",
                        "scaling": {
                            "source_unit": "mV",
                            "sensitivity": {
                                "value": "100",
                                "unit": "mV/g",
                                "source": "calibration_certificate",
                            },
                            "dc_policy": dc_policy,
                        },
                    }
                ]
            )

        bases = standard_bases(
            **{
                "channels[1].unit": {"basis": "user_stated"},
                "channels[1].scaling": {"basis": "user_stated"},
            }
        )
        without = validate(path, configuration(interp(None), bases))
        assert without.outcome == "needs_confirmation"
        assert [
            i.field for i in without.issues if i.effect == "needs_confirmation"
        ] == ["channels[1].dc_policy"]
        assert without.channels[0].mean is not None
        pending = next(i for i in without.issues if i.field == "channels[1].dc_policy")
        assert f"{without.channels[0].mean:.6g}" in pending.message
        assert without.import_id is None and without.import_digest is None
        assert not [s for s in without.statements if s.field == "channels[1].dc_policy"]
        with pytest.raises(ConfigurationError):
            # A basis cannot confirm a policy the interpretation does not state.
            validate(
                path,
                configuration(
                    interp(None),
                    {**bases, "channels[1].dc_policy": {"basis": "user_stated"}},
                ),
            )
        kept = validate(
            path,
            configuration(
                interp("keep"),
                {**bases, "channels[1].dc_policy": {"basis": "user_stated"}},
            ),
        )
        assert kept.outcome == "ready", kept.issues
        assert kept.channels[0].scaling_divisor == "100"
        dc = next(s for s in kept.statements if s.field == "channels[1].dc_policy")
        assert "no offset is subtracted" in dc.text and "mean" in dc.text

    def test_companion_blocks_ready_until_acknowledged(self, write):
        path = standard_file(write)
        path.with_name("run_metadata.json").write_text("{}", encoding="utf-8")
        blocked = validate(path, configuration(interpretation(), standard_bases()))
        assert blocked.outcome == "needs_confirmation"
        assert "companion_present" in codes(blocked, "needs_confirmation")
        acknowledged = validate(
            path,
            configuration(
                interpretation(), standard_bases(companion={"basis": "user_stated"})
            ),
        )
        assert acknowledged.outcome == "ready", acknowledged.issues
        statement = next(s for s in acknowledged.statements if s.field == "companion")
        assert "never reads it" in statement.text

    def test_unlisted_time_column_needs_an_override(self, write):
        path = standard_file(write)
        interp = interpretation(time_columns=[])
        unlisted = validate(path, configuration(interp, standard_bases()))
        assert unlisted.outcome == "needs_confirmation"
        assert "time_column_unassessed" in codes(unlisted, "needs_confirmation")
        overridden = validate(
            path,
            configuration(
                interp,
                standard_bases(),
                overrides={"time_columns[0]": "a software clock"},
            ),
        )
        assert overridden.outcome == "ready", overridden.issues

    def test_too_short_a_region_and_a_bad_token_are_invalid(self, write):
        short = validate(
            standard_file(write, "short.csv", count=100),
            configuration(interpretation(), standard_bases()),
        )
        assert "region_too_short" in codes(short, "invalid")
        lines = ["time [s],acc [g]"] + rows(400)
        lines[200] = lines[200].split(",")[0] + ",NaN"
        bad = validate(
            write("bad.csv", corpus.lines_to_bytes(lines)),
            configuration(interpretation(), standard_bases()),
        )
        assert bad.outcome == "invalid"
        assert [i.line for i in bad.issues if i.effect == "invalid"] == [201]
        assert bad.import_id is None and bad.import_digest is None


# ---------------------------------------------------------------------------
# Ledger pre-check
# ---------------------------------------------------------------------------


def _bound(
    columns: tuple[int, ...] = (1,), *, rate: str = "25600", first: int = 2
) -> dict[str, Any]:
    return interpretation(
        data_lines={"first": first},
        time_base={"source": "stated", "sampling_rate_hz": rate},
        defaults={"asset_id": ASSET, "acquired_at": ACQUIRED},
        channels=[
            {
                "column": c,
                "role": "vibration",
                "quantity": "acceleration",
                "unit": "g",
                "binding": {"measurement_point_id": POINT, "direction": "horizontal"},
            }
            for c in columns
        ],
    )


def _bound_bases(columns: tuple[int, ...] = (1,)) -> dict[str, Any]:
    return {
        "time_base": {"basis": "user_stated"},
        "defaults.acquired_at": {"basis": "user_stated"},
        **{f"channels[{c}].unit": {"basis": "user_stated"} for c in columns},
        **{f"channels[{c}].binding": {"basis": "user_stated"} for c in columns},
    }


def _record(
    store: LedgerStore,
    result,
    interp: dict[str, Any],
    column: int,
    *,
    into: str = ASSET,
    indexed: bool = True,
    file_block: Any = None,
    **declaration: Any,
) -> str:
    """A measurement_recorded event of *column* as an import would write it,
    in the ledger of *into*, and in the measurement index unless not
    *indexed* (as after a failed index append)."""
    channel = next(c for c in result.channels if c.column == column)
    mid = measurement_id_from_digest(result.source_sha256, column)
    payload_declaration = {
        "asset_id": into,
        "measurement_point_id": POINT,
        "acquired_at": ACQUIRED,
        "timezone_declared": True,
        "timestamp_suspect": False,
        "rpm": None,
        "load": None,
        "operating_state": None,
        "sensor_id": None,
        "direction": "horizontal",
        "declared_by": "import",
        "sampling_rate": float(interp["time_base"]["sampling_rate_hz"]),
        "signal_unit": "g",
        "raw_format": import_raw_marker(),
        "channel_index": column,
        "import_format": build_import_format(
            interp,
            column,
            samples_digest=channel.samples_digest,
            sample_count=channel.sample_count,
        ),
    }
    payload_declaration.update(declaration)
    payload = {
        "measurement_id": mid,
        "measurement_point_id": payload_declaration["measurement_point_id"],
        "declaration_version": 1,
        "declaration": payload_declaration,
        "file": (
            file_block
            if file_block is not None
            else {
                "location": "run.csv",
                "location_is_relative": True,
                "content_sha256": result.source_sha256,
                "size_bytes": result.size_bytes,
            }
        ),
        "signal_id": f"run_c{column}",
        "changed": [],
        "locations": ["run.csv"],
    }
    store.append(into, make_event(EVENT_MEASUREMENT_RECORDED, into, payload))
    if indexed:
        store.append_index_entry(mid, into, "e" * 64, ACQUIRED)
    return mid


def _baseline(store: LedgerStore, *measurement_ids: str) -> None:
    payload = {
        "baseline_id": short_id(POINT, *sorted(measurement_ids), ACQUIRED),
        "measurement_point_id": POINT,
        "measurement_ids": list(measurement_ids),
        "members": [
            {
                "measurement_id": m,
                "declaration_version": 1,
                "point_declaration_version": 1,
            }
            for m in measurement_ids
        ],
        "declared_by": "test",
        "note": None,
        "declared_at": ACQUIRED,
    }
    store.append(ASSET, make_event(EVENT_BASELINE_DECLARED, ASSET, payload))


class TestLedgerPrecheck:
    @pytest.fixture
    def store(self, ledger_dir) -> LedgerStore:
        return LedgerStore(config.get_ledger_dir())

    def test_new_and_known_asset(self, write, store):
        path = standard_file(write)
        new = validate(path, configuration(_bound(), _bound_bases()))
        assert new.outcome == "ready", new.issues
        assert [
            (p.asset_status, p.measurement_status) for p in new.ledger_precheck
        ] == [("new", "new")]
        point = {
            "measurement_point_id": POINT,
            "declaration_version": 1,
            "expected_signal_unit": "g",
            "changed": [],
        }
        store.append(ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, point))
        known = validate(path, configuration(_bound(), _bound_bases()))
        assert [
            (p.asset_status, p.point_status, p.point_declaration_version)
            for p in known.ledger_precheck
        ] == [("known", "declared", 1)]

    def test_reimport_of_a_baseline_member_is_reported(self, write, store):
        path = standard_file(write)
        first = validate(path, configuration(_bound(), _bound_bases()))
        mid = _record(
            store,
            first,
            normalize_configuration(configuration(_bound(), _bound_bases()))[
                "interpretation"
            ],
            1,
        )
        _baseline(store, mid)
        again = validate(path, configuration(_bound(rate="25601"), _bound_bases()))
        check = again.ledger_precheck[0]
        assert check.measurement_status == "supersedes" and check.baseline_member
        assert "sampling_rate" in check.changed
        assert {"prior_import_differs", "supersedes_baseline_member"} <= set(
            codes(again, "notice")
        )
        same = validate(path, configuration(_bound(), _bound_bases()))
        assert same.ledger_precheck[0].measurement_status == "already_recorded"

    def test_column_swap_onto_a_baseline_member_is_refused(self, write, store):
        data = corpus.lines_to_bytes(
            ["time [s],a [g],b [g],c [g],d [g]"] + rows(400, channels=4)
        )
        path = write("swap.csv", data)
        wrong = _bound((3,))
        first = validate(path, configuration(wrong, _bound_bases((3,))))
        assert first.outcome == "ready", first.issues
        mid = _record(
            store,
            first,
            normalize_configuration(configuration(wrong, _bound_bases((3,))))[
                "interpretation"
            ],
            3,
        )
        _baseline(store, mid)
        right = validate(path, configuration(_bound((4,)), _bound_bases((4,))))
        assert right.outcome == "invalid"
        swap = [i for i in right.issues if i.code == "binding_collision"]
        assert len(swap) == 1
        assert mid in swap[0].message and "index 3" in swap[0].message
        assert "active baseline" in swap[0].message
        # The limitation is stated; no workaround (such as importing twice)
        # is recommended.
        assert "does not swap or rebind channels within one import" in swap[0].message
        # The conflict is the operator's to correct outside this import: no
        # user confirmation is invited and no workaround is recommended.
        assert (
            "The existing binding conflict must be resolved outside this import, "
            "through an operator-controlled correction, before validation can "
            "succeed." in swap[0].message
        )
        assert "ask the user" not in swap[0].message.lower()
        assert "import it" not in swap[0].message
        assert swap[0].effect == "invalid"
        assert right.ledger_precheck[0].column_swap == mid

    def test_prior_import_differs_from_the_ledger_and_from_a_record(
        self, write, store, import_dir
    ):
        path = standard_file(write)
        recorded = _bound()
        first = validate(path, configuration(recorded, _bound_bases()))
        _record(
            store,
            first,
            normalize_configuration(configuration(recorded, _bound_bases()))[
                "interpretation"
            ],
            1,
        )
        unbound = interpretation(
            time_base={"source": "stated", "sampling_rate_hz": "25000"}, time_columns=[]
        )
        unbound_conf = configuration(
            unbound,
            standard_bases(),
            overrides={"time_columns[0]": "not assessed here"},
        )
        from_ledger = validate(path, unbound_conf)
        assert "prior_import_differs" in codes(from_ledger, "notice")

        other = write("copy.csv", path.read_bytes())
        for asset in os.listdir(config.get_ledger_dir()):
            os.remove(Path(config.get_ledger_dir()) / asset)
        records = ImportRecordStore(config.get_import_dir())
        records.write_interpretation(
            first.source_sha256,
            first.size_bytes,
            normalize_configuration(configuration(interpretation(), standard_bases()))[
                "interpretation"
            ],
        )
        from_record = validate(other, unbound_conf)
        assert "prior_import_differs" in codes(from_record, "notice")

    def test_restoring_a_superseded_interpretation_is_a_notice(self, write, store):
        path = standard_file(write)
        version_a = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        result_a = validate(path, configuration(_bound(), _bound_bases()))
        mid = _record(store, result_a, version_a, 1)
        version_b = normalize_configuration(
            configuration(_bound(rate="25601"), _bound_bases())
        )["interpretation"]
        result_b = validate(path, configuration(_bound(rate="25601"), _bound_bases()))
        _record(store, result_b, version_b, 1, sampling_rate=25601.0)
        restored = validate(path, configuration(_bound(), _bound_bases()))
        assert restored.outcome == "ready", restored.issues
        assert restored.ledger_precheck[0].restores_superseded
        assert "restores_superseded_interpretation" in codes(restored, "notice")
        assert mid == restored.ledger_precheck[0].measurement_id

    def test_non_overlapping_range_is_stated(self, write, store):
        path = standard_file(write, count=1000)
        early = _bound(first=2)
        early["data_lines"] = {"first": 2, "last": 401}
        early_bases = {**_bound_bases(), "data_lines": {"basis": "user_stated"}}
        result = validate(path, configuration(early, early_bases))
        assert result.outcome == "ready", result.issues
        _record(
            store,
            result,
            normalize_configuration(configuration(early, early_bases))[
                "interpretation"
            ],
            1,
        )
        late = _bound(first=600)
        late_bases = {**_bound_bases(), "data_lines": {"basis": "user_stated"}}
        moved = validate(path, configuration(late, late_bases))
        assert moved.ledger_precheck[0].data_range_overlaps is False
        assert "data_range_not_overlapping" in codes(moved, "notice")

    def test_point_values_must_equal_the_current_declaration(self, write, store):
        path = standard_file(write)
        point = {
            "measurement_point_id": POINT,
            "declaration_version": 1,
            "expected_signal_unit": "g",
            "expected_direction": "horizontal",
            "nominal_rpm": 1490.0,
            "changed": [],
        }
        store.append(ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, point))
        interp = _bound()
        interp["defaults"]["rpm"] = "1490"
        cited = {"basis": "user_accepted_point_value", "declaration_version": 1}
        bases = {
            **_bound_bases(),
            "channels[1].unit": cited,
            "channels[1].binding": cited,
            "defaults.rpm": cited,
        }
        accepted = validate(path, configuration(interp, bases))
        assert accepted.outcome == "ready", accepted.issues
        point_statements = [
            s for s in accepted.statements if s.origin == "point_declaration"
        ]
        assert {s.field for s in point_statements} == {
            "channels[1].unit",
            "channels[1].binding",
            "defaults.rpm",
        }
        assert all("point declaration version 1" in s.text for s in point_statements)

        stale = {"basis": "user_accepted_point_value", "declaration_version": 2}
        old = validate(
            path, configuration(interp, {**bases, "channels[1].unit": stale})
        )
        assert [i.field for i in old.issues if i.code == "basis_mismatch"] == [
            "channels[1].unit"
        ]

        moved = _bound()
        moved["defaults"]["rpm"] = "1500"
        moved["channels"][0]["binding"]["direction"] = "axial"
        moved["channels"][0]["unit"] = "m/s2"
        wrong = validate(path, configuration(moved, bases))
        assert sorted(i.field for i in wrong.issues if i.code == "basis_mismatch") == [
            "channels[1].binding",
            "channels[1].unit",
            "defaults.rpm",
        ]
        scaled = validate(
            path,
            configuration(interp, {**bases, "dialect": cited}),
        )
        assert "dialect" in [
            i.field for i in scaled.issues if i.code == "basis_mismatch"
        ]

    def test_point_value_without_a_declared_point_is_refused(self, write):
        path = standard_file(write)
        cited = {"basis": "user_accepted_point_value", "declaration_version": 1}
        result = validate(
            path,
            configuration(_bound(), {**_bound_bases(), "channels[1].binding": cited}),
        )
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["channels[1].binding"]
        assert "no declared measurement point" in mismatch[0].message

    def test_prior_values_must_equal_the_cited_record(self, write, import_dir):
        path = standard_file(write)
        recorded = normalize_configuration(
            configuration(interpretation(), standard_bases())
        )["interpretation"]
        first = validate(path, configuration(interpretation(), standard_bases()))
        written = ImportRecordStore(config.get_import_dir()).write_interpretation(
            first.source_sha256, first.size_bytes, recorded
        )
        cited = {"basis": "user_accepted_prior_value", "import_id": written.import_id}
        bases = {
            "time_base": cited,
            "channels[1].unit": cited,
            "dialect": cited,
            "data_lines": cited,
        }
        accepted = validate(path, configuration(interpretation(), bases))
        assert accepted.outcome == "ready", accepted.issues
        assert all(
            s.origin == "prior_import"
            for s in accepted.statements
            if s.basis is not None
        )
        assert f"import {written.import_id[:12]}" in accepted.statements[0].text

        changed = interpretation(
            time_base={"source": "stated", "sampling_rate_hz": "25000"},
            channels=[
                {
                    "column": 1,
                    "role": "vibration",
                    "quantity": "velocity",
                    "unit": "mm/s",
                }
            ],
            data_lines={"first": 3},
        )
        refused = validate(path, configuration(changed, bases))
        assert refused.outcome == "invalid"
        assert sorted(
            i.field for i in refused.issues if i.code == "basis_mismatch"
        ) == [
            "channels[1].unit",
            "data_lines",
            "time_base",
        ]
        unknown = {"basis": "user_accepted_prior_value", "import_id": "c" * 64}
        missing = validate(
            path,
            configuration(interpretation(), {**bases, "channels[1].unit": unknown}),
        )
        assert "reads no such column" in " ".join(
            i.message for i in missing.issues if i.code == "basis_mismatch"
        )

    def test_an_unreadable_record_is_never_taken_as_no_record(self, write, monkeypatch):
        path = standard_file(write)

        def unreadable(self, **kwargs):
            raise ImportRecordError("Cannot read import record 0123: denied.")

        monkeypatch.setattr(ImportRecordStore, "list_records", unreadable)
        with pytest.raises(ImportRecordError):
            validate(path, configuration(interpretation(), standard_bases()))


# ---------------------------------------------------------------------------
# Read-only, configuration contract, draft
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_validation_writes_nothing(
        self, write, ledger_dir, import_dir, monkeypatch
    ):
        path = standard_file(write)
        path.with_name("run_metadata.json").write_text("{}", encoding="utf-8")
        before = tree(path.parent, ledger_dir, import_dir)
        validate(path, configuration(_bound(), _bound_bases()))
        validate(path, configuration(interpretation(), {}))
        assert tree(path.parent, ledger_dir, import_dir) == before
        assert not Path(ledger_dir).exists() or not any(Path(ledger_dir).iterdir())
        assert not Path(import_dir).exists()


class TestConfigurationContract:
    def test_problems_are_typed_and_accumulated(self):
        bad = configuration(
            interpretation(),
            bases={
                "time_base": {"basis": "user_accepted_profile_value"},
                "channels[1].unit": {"basis": "user_accepted_file_value"},
                "channels[9].unit": {"basis": "user_stated"},
                "nonsense": {"basis": "user_stated"},
            },
            overrides={"dialect": "no"},
        )
        bad["extra"] = 1
        with pytest.raises(ConfigurationError) as caught:
            normalize_configuration(bad)
        kinds = {p.kind for p in caught.value.problems}
        assert kinds == {"unknown_key", "invalid", "missing"}
        text = str(caught.value)
        assert "profile" in text and "column 9" in text and "takes no override" in text

    @pytest.mark.parametrize(
        ("attestation", "path", "says"),
        [
            ("nope", "attestation", "pass an object"),
            ({"schema": "x/1"}, "attestation.schema", "the only schema"),
            ({"bases": []}, "attestation.bases", "pass an object"),
            ({"overrides": []}, "attestation.overrides", "pass an object"),
            (
                {"bases": {"time_base": 3}},
                "attestation.bases['time_base']",
                "pass an object",
            ),
            (
                {"bases": {"time_base": {}}},
                "attestation.bases['time_base'].basis",
                "is required",
            ),
            (
                {"bases": {"time_base": "guess"}},
                "attestation.bases['time_base'].basis",
                "is not one of",
            ),
            (
                {"bases": {"time_base": {"basis": "user_stated", "why": 1}}},
                "attestation.bases['time_base']",
                "unknown key",
            ),
            (
                {"bases": {"time_base": {"basis": "user_stated", "line": 2}}},
                "attestation.bases['time_base'].line",
                "cites no line",
            ),
            (
                {
                    "bases": {
                        "time_base": {
                            "basis": "user_accepted_prior_value",
                            "import_id": "ab",
                        }
                    }
                },
                "attestation.bases['time_base'].import_id",
                "not an import_id",
            ),
            (
                {
                    "bases": {
                        "time_base": {
                            "basis": "user_accepted_derived_value",
                            "column": "0",
                        }
                    }
                },
                "attestation.bases['time_base'].column",
                "pass an integer",
            ),
            (
                {
                    "bases": {
                        "time_base": {
                            "basis": "user_accepted_derived_value",
                            "column": 4096,
                        }
                    }
                },
                "attestation.bases['time_base'].column",
                "out of range",
            ),
            (
                {
                    "bases": {
                        "time_base": {"basis": "user_accepted_file_value", "line": 0}
                    }
                },
                "attestation.bases['time_base'].line",
                "out of range",
            ),
            (
                {
                    "bases": {
                        "time_base": {"basis": "user_accepted_file_value", "line": True}
                    }
                },
                "attestation.bases['time_base'].line",
                "pass a positive integer",
            ),
            (
                {"bases": {"channels[1].scaling": "user_stated"}},
                "channels[1].scaling",
                "which has none",
            ),
            (
                {"bases": {"channels[1].binding": "user_stated"}},
                "channels[1].binding",
                "which has none",
            ),
            (
                {"bases": {"defaults.load": "user_stated"}},
                "defaults.load",
                "leaves empty",
            ),
            (
                {"bases": {"time_columns[3]": "user_stated"}},
                "time_columns[3]",
                "not a basis",
            ),
            (
                {"overrides": {"time_columns[0]": "x"}},
                "time_columns[0]",
                "already reads",
            ),
            (
                {"overrides": {"nonsense": "x"}},
                "attestation.overrides['nonsense']",
                "not a field",
            ),
            (
                {"overrides": {"time_base": None}},
                "attestation.overrides['time_base']",
                "needs a reason",
            ),
            (
                {"overrides": {"time_base": "x‮y"}},
                "attestation.overrides['time_base']",
                "is refused",
            ),
            ({"declared_by": "\u0000"}, "attestation.declared_by", "is refused"),
            ({"bases": {"x" * 300: "user_stated"}}, "attestation.bases", "not a field"),
        ],
    )
    def test_attestation_refusals_name_their_path(self, attestation, path, says):
        conf = {
            "schema": "pmm-text-import-config/1",
            "interpretation": interpretation(),
            "attestation": attestation,
        }
        with pytest.raises(ConfigurationError) as caught:
            normalize_configuration(conf)
        assert any(
            p.path.startswith(path) and says in p.text for p in caught.value.problems
        ), caught.value.problems

    @pytest.mark.parametrize(
        ("conf", "path"),
        [
            ("nope", "configuration"),
            ({"interpretation": {}}, "schema"),
            ({"schema": "x/1", "interpretation": {}}, "schema"),
            ({"schema": "pmm-text-import-config/1"}, "interpretation"),
        ],
    )
    def test_configuration_shape_refusals(self, conf, path):
        with pytest.raises(ConfigurationError) as caught:
            normalize_configuration(conf)
        assert any(p.path == path for p in caught.value.problems)

    def test_canonical_form_is_idempotent_and_accepts_shorthand(self):
        conf = configuration(
            interpretation(),
            {"time_base": "user_stated", "channels[1].unit": "user_stated"},
        )
        conf["attestation"]["declared_by"] = "line engineer"
        canonical = normalize_configuration(conf)
        assert canonical["attestation"]["bases"]["time_base"]["basis"] == "user_stated"
        assert normalize_configuration(canonical) == canonical

    def test_transport_and_import_id_are_checked(self):
        with pytest.raises(ValueError):
            check_transport("carrier-pigeon")
        with pytest.raises(ValueError):
            import_digest("A" * 64, {}, channel="python_api", transport=None)
        assert check_transport(None) is None

    def test_digest_binds_attestation_channel_and_transport(self):
        canonical = normalize_configuration(
            configuration(interpretation(), standard_bases())
        )
        att = canonical["attestation"]
        other = copy.deepcopy(att)
        other["declared_by"] = "someone"
        digests = {
            import_digest("a" * 64, att, channel="python_api", transport=None),
            import_digest("a" * 64, other, channel="python_api", transport=None),
            import_digest(
                "a" * 64, att, channel="mcp_tool_argument", transport="stdio"
            ),
        }
        assert len(digests) == 3
        with pytest.raises(ValueError):
            import_digest("a" * 64, att, channel="mcp_elicitation", transport=None)

    def test_channel_comes_from_the_adapter(self, write):
        path = standard_file(write)
        with pytest.raises(TypeError):
            validate_import(str(path), configuration(interpretation(), standard_bases()))  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            validate(
                path, configuration(interpretation(), standard_bases()), channel="ui"
            )


class TestConfigure:
    def test_draft_prefills_with_origins_and_no_basis(self, write):
        path = write("eu.csv", corpus.european_export())
        draft = configure_import(inspect_signal_file(str(path)))
        conf = draft.configuration
        assert conf["attestation"]["bases"] == {}
        interp = conf["interpretation"]
        assert (interp["dialect"]["delimiter"], interp["dialect"]["decimal"]) == (
            ";",
            ",",
        )
        assert interp["data_lines"]["first"] == 9
        origins = {p.field: p.origin for p in draft.prefills}
        assert origins["channels[1].unit"] == "file_declared"
        assert origins["time_base"] in ("derived", "file_declared")
        assert any("sampling rate" in r for r in draft.remaining_confirmations)
        assert all("ask the user" in r for r in draft.remaining_confirmations)

    def test_edits_win_and_prior_and_point_prefill(self, write):
        path = write("eu.csv", corpus.european_export())
        inspection = inspect_signal_file(str(path))
        prior_interp = normalize_configuration(
            configuration(
                interpretation(
                    dialect={
                        "encoding": "utf-8",
                        "delimiter": ";",
                        "decimal": ",",
                        "header_line": 7,
                    },
                    data_lines={"first": 9},
                    channels=[
                        {
                            "column": 1,
                            "role": "vibration",
                            "quantity": "acceleration",
                            "unit": "g",
                            "binding": {"measurement_point_id": POINT},
                        }
                    ],
                ),
                {},
            )
        )["interpretation"]
        draft = configure_import(
            inspection,
            edits={"interpretation": {"defaults": {"asset_id": ASSET}}},
            prior={"import_id": "b" * 64, "interpretation": prior_interp},
            point={
                "measurement_point_id": POINT,
                "declaration_version": 2,
                "expected_direction": "axial",
                "nominal_rpm": 1490.0,
            },
        )
        interp = draft.configuration["interpretation"]
        assert interp["defaults"]["asset_id"] == ASSET
        channel = next(c for c in interp["channels"] if c["column"] == 1)
        assert channel["binding"]["measurement_point_id"] == POINT
        assert channel["binding"]["direction"] == "axial"
        origins = {(p.field, p.origin) for p in draft.prefills}
        assert ("channels[1].binding", "prior_import") in origins
        assert ("channels[1].binding", "point_declaration") in origins

    def test_printed_rate_speed_and_voltage_column(self, write):
        data = corpus.lines_to_bytes(
            ["Sampling rate: 25600", "Speed: 1490 rpm", "acc [g],probe [mV]"]
            + [
                f"{corpus.sample(r, 0):.6f},{corpus.sample(r, 1):.6f}"
                for r in range(400)
            ]
        )
        path = write("printed.csv", data)
        draft = configure_import(inspect_signal_file(str(path)))
        interp = draft.configuration["interpretation"]
        assert interp["time_base"] == {
            "source": "file_declared",
            "sampling_rate_hz": "25600",
            "time_column": None,
        }
        assert interp["defaults"]["rpm"] == "1490"
        assert [c["column"] for c in interp["channels"]] == [0]
        origins = {p.field: (p.origin, p.line) for p in draft.prefills}
        assert origins["time_base"] == ("file_declared", 1)
        assert origins["defaults.rpm"] == ("file_declared", 2)
        remaining = " ".join(draft.remaining_confirmations)
        assert (
            "Index 1, the 2nd field from the left carries an electrical unit"
            in remaining
        )
        assert "DC policy" in remaining
        assert "citing line 1" in remaining

    def test_prior_channels_fill_a_draft_without_importable_columns(self, write):
        data = corpus.lines_to_bytes(
            ["a,b"] + [f"{r},{corpus.sample(r, 0):.6f}" for r in range(400)]
        )
        path = write("bare.csv", data)
        prior_interp = normalize_configuration(
            configuration(interpretation(time_columns=[]), {})
        )["interpretation"]
        draft = configure_import(
            inspect_signal_file(str(path)),
            prior={"import_id": "b" * 64, "interpretation": prior_interp},
        )
        interp = draft.configuration["interpretation"]
        assert interp["time_base"]["source"] == "stated"
        assert [c["column"] for c in interp["channels"]] == [1]
        origins = {(p.field, p.origin) for p in draft.prefills}
        assert ("time_base", "prior_import") in origins
        assert ("channels[1].unit", "prior_import") in origins
        assert any(
            "citing import " + "b" * 64 in r for r in draft.remaining_confirmations
        )

    def test_draft_validates_once_confirmed(self, write):
        path = standard_file(write)
        draft = configure_import(inspect_signal_file(str(path)))
        assert any(
            "validation derives it from the whole column" in r
            for r in draft.remaining_confirmations
        )
        conf = copy.deepcopy(draft.configuration)
        time_base = conf["interpretation"]["time_base"]
        assert time_base["source"] == "time_column"
        conf["attestation"]["bases"]["time_base"] = {
            "basis": "user_accepted_derived_value",
            "column": time_base["time_column"],
        }
        conf["attestation"]["bases"]["channels[1].unit"] = {
            "basis": "user_accepted_file_value",
            "line": 1,
        }
        first = validate(path, conf)
        # The draft carries the head-derived rate; validation states the
        # whole-column rate, and accepting that value makes the draft ready.
        whole = first.time_columns[0].proposed_rate_hz
        assert whole is not None
        if whole != time_base["sampling_rate_hz"]:
            assert first.outcome == "invalid"
            assert [i.field for i in first.issues if i.code == "basis_mismatch"] == [
                "time_base"
            ]
        time_base["sampling_rate_hz"] = whole
        result = validate(path, conf)
        assert result.outcome == "ready", result.issues
        assert result.import_digest is not None

    def test_inputs_are_never_mutated_and_edits_win(self, write):
        path = write("eu.csv", corpus.european_export())
        inspection = inspect_signal_file(str(path))
        edits = {
            "interpretation": {
                "time_base": {"sampling_rate_hz": "25000"},
                "channels": [
                    {
                        "column": 1,
                        "role": "vibration",
                        "quantity": "acceleration",
                        "unit": "g",
                        "binding": {"measurement_point_id": POINT},
                    }
                ],
            }
        }
        before = copy.deepcopy(edits)
        point = {
            "measurement_point_id": POINT,
            "declaration_version": 1,
            "expected_direction": "axial",
            "expected_sensor_id": "S-1",
        }
        first = configure_import(inspection, edits=edits, point=point)
        assert edits == before
        assert (
            first.configuration["interpretation"]["time_base"]["sampling_rate_hz"]
            == "25000"
        )
        second = configure_import(inspection, edits=edits, point=point)
        origins = {(p.field, p.origin) for p in second.prefills}
        assert ("channels[1].binding", "point_declaration") in origins


# ---------------------------------------------------------------------------
# Evidence the configuration cannot hide, and contradictions it cannot state
# ---------------------------------------------------------------------------


def _scaled(sensitivity_unit: str, unit: str) -> dict[str, Any]:
    return {
        "column": 1,
        "role": "vibration",
        "quantity": "acceleration",
        "unit": unit,
        "scaling": {
            "source_unit": "mV",
            "sensitivity": {
                "value": "100",
                "unit": sensitivity_unit,
                "source": "datasheet_nominal",
            },
            "dc_policy": "keep",
        },
    }


class TestEvidenceTheConfigurationCannotHide:
    def test_an_omitted_units_line_still_contradicts_and_is_not_proven(self, write):
        path = write(
            "units.csv", corpus.lines_to_bytes(["time,acc", "s,mV"] + rows(400))
        )
        dialect = {"encoding": "utf-8", "delimiter": ",", "decimal": "."}
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        omitted = validate(
            path,
            configuration(
                interpretation(
                    dialect={**dialect, "header_line": 1}, data_lines={"first": 3}
                ),
                bases,
            ),
        )
        assert omitted.outcome == "invalid"
        assert "contradicts_file_declaration" in codes(omitted, "invalid")
        assert [i.field for i in omitted.issues if i.code == "missing_basis"] == [
            "dialect"
        ]
        dialect_statement = next(s for s in omitted.statements if s.field == "dialect")
        assert "parses this way" not in dialect_statement.text
        assert "no basis is given yet" in dialect_statement.text
        named = validate(
            path,
            configuration(
                interpretation(
                    dialect={**dialect, "header_line": 1, "units_line": 2},
                    data_lines={"first": 3},
                ),
                bases,
            ),
        )
        assert "contradicts_file_declaration" in codes(named, "invalid")
        assert "missing_basis" not in codes(named)

    def test_cp1252_needs_an_encoding_confirmation(self, write):
        header = "temps [s],accél [g]".encode("cp1252")
        body = corpus.lines_to_bytes(rows(400))
        path = write("latin.csv", header + b"\n" + body)
        interp = interpretation(
            dialect={
                "encoding": "cp1252",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 1,
            }
        )
        unconfirmed = validate(path, configuration(interp, standard_bases()))
        assert unconfirmed.outcome == "needs_confirmation"
        assert "encoding_needs_confirmation" in codes(unconfirmed, "needs_confirmation")
        statement = next(s for s in unconfirmed.statements if s.field == "dialect")
        assert "no basis is given yet" in statement.text
        confirmed = validate(
            path,
            configuration(interp, standard_bases(dialect={"basis": "user_stated"})),
        )
        assert confirmed.outcome == "ready", confirmed.issues

    def test_a_trailing_delimiter_keeps_the_units_row_matched(self, write):
        data = ["time,acc", "s,mm/s"] + [f"{line}," for line in rows(400)]
        path = write("trailing.csv", corpus.lines_to_bytes(data))
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 1,
                "units_line": 2,
            },
            data_lines={"first": 3},
        )
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "invalid"
        assert "contradicts_file_declaration" in codes(result, "invalid")
        assert "field_count_mismatch" not in codes(result)

    def test_a_huge_printed_exponent_is_refused_promptly(self, write):
        data = ["dt: 1e-999999999", "acc [g]"] + [
            f"{corpus.sample(r, 0):.6f}" for r in range(400)
        ]
        path = write("exponent.csv", corpus.lines_to_bytes(data))
        interp = {
            "schema": "pmm-text-import/1",
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            "data_lines": {"first": 3},
            "time_base": {"source": "stated", "sampling_rate_hz": "25600"},
            "channels": [
                {
                    "column": 0,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                }
            ],
        }
        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[0].unit": {"basis": "user_accepted_file_value", "line": 2},
        }
        import time

        started = time.perf_counter()
        result = validate(path, configuration(interp, bases))
        assert time.perf_counter() - started < 5
        assert result.outcome == "needs_confirmation"
        assert "rate_sources_disagree" in codes(result, "needs_confirmation")

    def test_a_time_like_column_selected_as_vibration_is_refused(self, write):
        path = write("timecol.csv", corpus.lines_to_bytes(["time,acc"] + rows(400)))
        interp = interpretation(
            time_columns=[],
            channels=[
                {
                    "column": 0,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                }
            ],
        )
        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[0].unit": {"basis": "user_stated"},
        }
        refused = validate(path, configuration(interp, bases))
        assert refused.outcome == "invalid"
        issue = next(
            i for i in refused.issues if i.code == "contradicts_file_declaration"
        )
        assert (
            issue.field == "channels[0].unit"
            and "look like a time column" in issue.message
        )
        chosen = validate(
            path,
            configuration(
                interp,
                bases,
                overrides={"channels[0].unit": "a logged ramp test signal"},
            ),
        )
        assert "contradicts_file_declaration" in codes(chosen, "notice")
        assert "contradicts_file_declaration" not in codes(chosen, "invalid")


class TestValuesCheckedAgainstEveryPrintedOne:
    def test_sensitivity_unit_must_be_the_printed_one(self, write):
        data = ["Sensitivity: 100 mV/g", "time [s],acc [mV]"] + rows(400)
        path = write("sens.csv", corpus.lines_to_bytes(data))
        dialect = {
            "encoding": "utf-8",
            "delimiter": ",",
            "decimal": ".",
            "header_line": 2,
        }

        def run(sensitivity_unit: str, unit: str):
            return validate(
                path,
                configuration(
                    interpretation(
                        dialect=dialect,
                        data_lines={"first": 3},
                        channels=[_scaled(sensitivity_unit, unit)],
                    ),
                    {
                        "time_base": {"basis": "user_stated"},
                        "channels[1].unit": {"basis": "user_stated"},
                        "channels[1].scaling": {
                            "basis": "user_accepted_file_value",
                            "line": 1,
                        },
                        "channels[1].dc_policy": {"basis": "user_stated"},
                    },
                ),
            )

        wrong = run("mV/(m/s2)", "m/s2")
        mismatch = [i for i in wrong.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["channels[1].scaling"]
        assert "another unit" in mismatch[0].message
        assert run("mV/g", "g").outcome == "ready"

    def test_a_bracketed_key_unit_is_the_printed_unit(self, write):
        data = ["Sensitivity [mV/g]: 100", "time [s],acc [mV]"] + rows(400)
        path = write("bracket.csv", corpus.lines_to_bytes(data))
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
            channels=[_scaled("mV/g", "g")],
        )
        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[1].unit": {"basis": "user_stated"},
            "channels[1].scaling": {"basis": "user_accepted_file_value", "line": 1},
            "channels[1].dc_policy": {"basis": "user_stated"},
        }
        assert validate(path, configuration(interp, bases)).outcome == "ready"

    def test_a_second_printed_rate_disagreeing_needs_a_choice(self, write):
        data = ["Sampling rate: 25600", "dt: 0.0000500", "acc [g]"] + [
            f"{corpus.sample(r, 0):.6f}" for r in range(400)
        ]
        path = write("tworates.csv", corpus.lines_to_bytes(data))
        interp = {
            "schema": "pmm-text-import/1",
            "dialect": {
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 3,
            },
            "data_lines": {"first": 4},
            "time_base": {"source": "file_declared", "sampling_rate_hz": "25600"},
            "channels": [
                {
                    "column": 0,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                }
            ],
        }
        bases = {
            "time_base": {"basis": "user_accepted_file_value", "line": 1},
            "channels[0].unit": {"basis": "user_accepted_file_value", "line": 3},
        }
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "needs_confirmation"
        disagree = [i for i in result.issues if i.code == "rate_sources_disagree"]
        assert [i.line for i in disagree] == [2]
        assert "basis_mismatch" not in codes(result)
        chosen = validate(
            path,
            configuration(
                interp, bases, overrides={"time_base": "the interval is rounded"}
            ),
        )
        assert chosen.outcome == "ready", chosen.issues

    def test_a_stated_speed_the_preamble_contradicts(self, write):
        data = ["Speed: 1780", "time [s],acc [g]"] + rows(400)
        path = write("speed2.csv", corpus.lines_to_bytes(data))
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
            defaults={"rpm": "1800"},
        )
        bases = standard_bases(
            **{
                "channels[1].unit": {"basis": "user_accepted_file_value", "line": 2},
                "defaults.rpm": {"basis": "user_stated"},
            }
        )
        refused = validate(path, configuration(interp, bases))
        issue = next(
            i for i in refused.issues if i.code == "contradicts_file_declaration"
        )
        assert (issue.field, issue.effect, issue.line) == ("defaults.rpm", "invalid", 1)
        chosen = validate(
            path,
            configuration(
                interp, bases, overrides={"defaults.rpm": "the speed changed"}
            ),
        )
        assert chosen.outcome == "ready", chosen.issues
        assert "contradicts_file_declaration" in codes(chosen, "notice")

    def test_python_api_bases_are_not_marked_caller_asserted(self, write):
        path = standard_file(write)
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.caller_asserted is False
        assert not any(s.caller_asserted for s in result.statements)
        assert not any("as the caller asserted" in s.text for s in result.statements)


class TestPointAndLedgerEdges:
    @pytest.fixture
    def store(self, ledger_dir) -> LedgerStore:
        return LedgerStore(config.get_ledger_dir())

    def _declare(self, store: LedgerStore, point_id: str, **fields: Any) -> None:
        payload = {
            "measurement_point_id": point_id,
            "declaration_version": 1,
            "changed": [],
        }
        payload.update(fields)
        store.append(
            ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload)
        )

    def test_a_point_value_cannot_confirm_a_direction_the_point_never_declares(
        self, write, store
    ):
        path = standard_file(write)
        self._declare(store, POINT, expected_signal_unit="g")
        cited = {"basis": "user_accepted_point_value", "declaration_version": 1}
        result = validate(
            path,
            configuration(_bound(), {**_bound_bases(), "channels[1].binding": cited}),
        )
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["channels[1].binding"]
        assert "holds no direction" in mismatch[0].message

    def test_a_point_rpm_must_hold_for_every_bound_point(self, write, store):
        data = corpus.lines_to_bytes(["time [s],x [g],y [g]"] + rows(400, channels=2))
        path = write("two_points.csv", data)
        self._declare(store, POINT, nominal_rpm=1490.0)
        self._declare(store, "motor_de_v", nominal_rpm=1500.0)
        interp = interpretation(
            defaults={"asset_id": ASSET, "acquired_at": ACQUIRED, "rpm": "1490"},
            channels=[
                {
                    "column": 1,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "binding": {
                        "measurement_point_id": POINT,
                        "direction": "horizontal",
                    },
                },
                {
                    "column": 2,
                    "role": "vibration",
                    "quantity": "acceleration",
                    "unit": "g",
                    "binding": {
                        "measurement_point_id": "motor_de_v",
                        "direction": "vertical",
                    },
                },
            ],
        )
        bases = {
            "time_base": {"basis": "user_stated"},
            "defaults.acquired_at": {"basis": "user_stated"},
            "defaults.rpm": {
                "basis": "user_accepted_point_value",
                "declaration_version": 1,
            },
            **{f"channels[{c}].unit": {"basis": "user_stated"} for c in (1, 2)},
            **{f"channels[{c}].binding": {"basis": "user_stated"} for c in (1, 2)},
        }
        result = validate(path, configuration(interp, bases))
        assert [i.field for i in result.issues if i.code == "basis_mismatch"] == [
            "defaults.rpm"
        ]

    def test_a_timezone_correction_supersedes(self, write, store):
        path = standard_file(write)
        interp = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        first = validate(path, configuration(_bound(), _bound_bases()))
        _record(store, first, interp, 1, timezone_declared=False)
        again = validate(path, configuration(_bound(), _bound_bases()))
        check = again.ledger_precheck[0]
        assert check.measurement_status == "supersedes"
        assert "timezone_declared" in check.changed

    def test_an_unreadable_recorded_rate_reads_as_changed_not_as_a_crash(
        self, write, store
    ):
        path = standard_file(write)
        interp = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        first = validate(path, configuration(_bound(), _bound_bases()))
        mid = _record(store, first, interp, 1, sampling_rate="fast", rpm="slow")
        result = validate(path, configuration(_bound(), _bound_bases()))
        check = result.ledger_precheck[0]
        assert check.measurement_id == mid
        assert check.measurement_status == "supersedes"
        assert "sampling_rate" in check.changed


class TestUnitTokenKinds:
    def test_another_electrical_unit_contradicts_the_scaling(self, write):
        path = standard_file(write, header="time [s],acc [mV]")
        channel = _scaled("V/g", "g")
        channel["scaling"]["source_unit"] = "V"
        bases = {
            "time_base": {"basis": "user_stated"},
            "channels[1].unit": {"basis": "user_stated"},
            "channels[1].scaling": {"basis": "user_stated"},
            "channels[1].dc_policy": {"basis": "user_stated"},
        }
        result = validate(
            path, configuration(interpretation(channels=[channel]), bases)
        )
        issue = next(
            i for i in result.issues if i.code == "contradicts_file_declaration"
        )
        assert issue.effect == "invalid" and "another electrical unit" in issue.message

    def test_a_time_unit_token_on_a_channel_contradicts_once(self, write):
        path = standard_file(write, header="time [s],acc [s]")
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        result = validate(path, configuration(interpretation(), bases))
        contradictions = [
            i for i in result.issues if i.code == "contradicts_file_declaration"
        ]
        assert (
            len(contradictions) == 1 and "is a time unit" in contradictions[0].message
        )

    def test_a_preview_token_with_an_override_is_a_notice(self, write):
        path = standard_file(write, header="time [s],speed [rpm]")
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        result = validate(
            path,
            configuration(
                interpretation(),
                bases,
                overrides={"channels[1].unit": "mislabelled column"},
            ),
        )
        assert "unit_token_preview_only" in codes(result, "notice")
        assert "unit_token_preview_only" not in codes(result, "invalid")

    def test_a_units_row_of_another_width_is_a_notice(self, write):
        path = write(
            "wide.csv", corpus.lines_to_bytes(["time,acc", "s,g,g"] + rows(400))
        )
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 1,
                "units_line": 2,
            },
            data_lines={"first": 3},
        )
        bases = standard_bases(**{"channels[1].unit": {"basis": "user_stated"}})
        result = validate(path, configuration(interp, bases))
        mismatch = [i for i in result.issues if i.code == "field_count_mismatch"]
        assert [(i.effect, i.line) for i in mismatch] == [("notice", 2)]


class TestBasisKinds:
    def test_a_point_value_cannot_confirm_a_scaling(self, write, ledger_dir):
        path = standard_file(write, header="time [s],acc [mV]")
        store = LedgerStore(config.get_ledger_dir())
        payload = {
            "measurement_point_id": POINT,
            "declaration_version": 1,
            "changed": [],
        }
        store.append(
            ASSET, make_event(EVENT_MEASUREMENT_POINT_DECLARED, ASSET, payload)
        )
        channel = {**_scaled("mV/g", "g"), "binding": {"measurement_point_id": POINT}}
        interp = interpretation(
            defaults={"asset_id": ASSET, "acquired_at": ACQUIRED}, channels=[channel]
        )
        cited = {"basis": "user_accepted_point_value", "declaration_version": 1}
        bases = {
            "time_base": {"basis": "user_stated"},
            "defaults.acquired_at": {"basis": "user_stated"},
            "channels[1].unit": {"basis": "user_stated"},
            "channels[1].scaling": cited,
            "channels[1].dc_policy": {"basis": "user_stated"},
            "channels[1].binding": {"basis": "user_stated"},
        }
        result = validate(path, configuration(interp, bases))
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["channels[1].scaling"]
        assert "holds no scaling" in mismatch[0].message

    @pytest.mark.parametrize(
        ("field", "entry", "says"),
        [
            (
                "channels[1].unit",
                {"basis": "user_accepted_derived_value", "column": 0},
                "cannot confirm a channel's unit",
            ),
            (
                "channels[1].unit",
                {"basis": "user_accepted_file_value", "line": 5},
                "holds no unit token",
            ),
        ],
    )
    def test_a_channel_basis_must_fit_its_value(self, write, field, entry, says):
        path = standard_file(write)
        result = validate(
            path, configuration(interpretation(), standard_bases(**{field: entry}))
        )
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == [field] and says in mismatch[0].message

    def test_a_derived_value_cannot_confirm_a_default(self, write):
        path = standard_file(write)
        interp = interpretation(defaults={"acquired_at": ACQUIRED})
        bases = standard_bases(
            **{
                "defaults.acquired_at": {
                    "basis": "user_accepted_derived_value",
                    "column": 0,
                }
            }
        )
        result = validate(path, configuration(interp, bases))
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["defaults.acquired_at"]
        assert "cannot confirm the acquired at" in mismatch[0].message

    def test_a_prior_speed_must_equal_the_cited_record(self, write, import_dir):
        path = standard_file(write)
        recorded = normalize_configuration(
            configuration(interpretation(defaults={"rpm": "1490"}), {})
        )["interpretation"]
        first = validate(path, configuration(interpretation(), standard_bases()))
        written = ImportRecordStore(config.get_import_dir()).write_interpretation(
            first.source_sha256, first.size_bytes, recorded
        )
        cited = {"basis": "user_accepted_prior_value", "import_id": written.import_id}

        def run(rpm: str):
            return validate(
                path,
                configuration(
                    interpretation(defaults={"rpm": rpm}),
                    standard_bases(**{"defaults.rpm": cited}),
                ),
            )

        assert "basis_mismatch" not in codes(run("1490"))
        assert [i.field for i in run("1500").issues if i.code == "basis_mismatch"] == [
            "defaults.rpm"
        ]

    def test_an_instant_the_preamble_prints_otherwise(self, write):
        def run(printed: str, basis: dict[str, Any]):
            data = [f"Start time: {printed}", "time [s],acc [g]"] + rows(400)
            path = write("instant.csv", corpus.lines_to_bytes(data))
            interp = interpretation(
                dialect={
                    "encoding": "utf-8",
                    "delimiter": ",",
                    "decimal": ".",
                    "header_line": 2,
                },
                data_lines={"first": 3},
                defaults={"acquired_at": ACQUIRED},
            )
            bases = standard_bases(
                **{
                    "channels[1].unit": {
                        "basis": "user_accepted_file_value",
                        "line": 2,
                    },
                    "defaults.acquired_at": basis,
                }
            )
            return validate(path, configuration(interp, bases))

        stated = {"basis": "user_stated"}
        earlier = run("2026-08-20T10:00:00+00:00", stated)
        issue = next(
            i for i in earlier.issues if i.code == "contradicts_file_declaration"
        )
        assert (issue.field, issue.line) == ("defaults.acquired_at", 1)
        assert "contradicts_file_declaration" not in codes(
            run("2026-08-20T11:42:00", stated)
        )
        assert "contradicts_file_declaration" not in codes(
            run("20/08/2026 11:42", stated)
        )
        cited = {"basis": "user_accepted_file_value", "line": 1}
        unreadable = run("20/08/2026 11:42", cited)
        assert [i.field for i in unreadable.issues if i.code == "basis_mismatch"] == [
            "defaults.acquired_at"
        ]


class TestStructureAndBindingIssues:
    def test_ambiguous_rows_need_a_dialect_choice(self, write):
        data = [f"{r},{r % 7}" for r in range(1, 401)]
        path = write("ambiguous.csv", corpus.lines_to_bytes(data))
        interp = interpretation(
            dialect={"encoding": "utf-8", "delimiter": ",", "decimal": "."},
            data_lines={"first": 1},
            time_columns=[],
        )
        result = validate(
            path, configuration(interp, {"time_base": {"basis": "user_stated"}})
        )
        assert "ambiguous_dialect" in codes(result, "needs_confirmation")
        assert result.outcome != "ready"

    def test_a_binding_without_a_basis_or_an_asset(self, write):
        path = standard_file(write)
        interp = _bound()
        interp["defaults"] = {"acquired_at": ACQUIRED}
        bases = {k: v for k, v in _bound_bases().items() if not k.endswith(".binding")}
        result = validate(path, configuration(interp, bases))
        assert "channels[1].binding" in [
            i.field for i in result.issues if i.code == "missing_basis"
        ]
        invalid_binding = [i for i in result.issues if i.code == "binding_invalid"]
        assert [i.effect for i in invalid_binding] == ["needs_confirmation"]
        assert "ask the user" in invalid_binding[0].message.lower()


class TestHandWrittenLedgerShapes:
    @pytest.fixture
    def store(self, ledger_dir) -> LedgerStore:
        return LedgerStore(config.get_ledger_dir())

    def test_missing_or_partial_import_formats_read_as_changes(self, write, store):
        path = standard_file(write)
        interp = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        first = validate(path, configuration(_bound(), _bound_bases()))
        _record(
            store, first, interp, 1, import_format=None, acquired_at="not an instant"
        )
        missing = validate(path, configuration(_bound(), _bound_bases()))
        assert "import_format" in missing.ledger_precheck[0].changed
        _record(store, first, interp, 1, import_format={"dialect": {}})
        partial = validate(path, configuration(_bound(), _bound_bases()))
        check = partial.ledger_precheck[0]
        assert check.measurement_status == "supersedes"
        assert check.data_range_overlaps is None

    def test_measurements_of_other_bytes_are_not_compared(self, write, store):
        other = write(
            "other.csv", corpus.lines_to_bytes(["time [s],acc [g]"] + rows(500))
        )
        recorded = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        elsewhere = validate(other, configuration(_bound(), _bound_bases()))
        _record(store, elsewhere, recorded, 1, direction="vertical")
        path = standard_file(write)
        result = validate(path, configuration(_bound(), _bound_bases()))
        assert result.ledger_precheck[0].measurement_status == "new"
        assert result.ledger_precheck[0].column_swap is None
        assert "prior_import_differs" not in codes(result)


# ---------------------------------------------------------------------------
# History that cannot be read, or is not read in full, fails closed
# ---------------------------------------------------------------------------


def _write_record(path: Path, interp: dict[str, Any]) -> str:
    result = validate(path, configuration(interpretation(), standard_bases()))
    written = ImportRecordStore(config.get_import_dir()).write_interpretation(
        result.source_sha256,
        result.size_bytes,
        normalize_configuration(configuration(interp, {}))["interpretation"],
    )
    return written.import_id


def _record_file(import_id: str) -> Path:
    return Path(config.get_import_dir()) / "records" / f"{import_id}.jsonl"


class TestHistoryFailsClosed:
    """Unreadable or partly read history is never taken as complete: the
    outcome is ``invalid`` (no confirmation can stand in for it), except
    where its irrelevance is established without the damaged data."""

    @pytest.fixture
    def store(self, ledger_dir) -> LedgerStore:
        return LedgerStore(config.get_ledger_dir())

    def _ready(self, path: Path):
        return validate(path, configuration(_bound(), _bound_bases()))

    def _bound_record(self, path: Path, store: LedgerStore, **declaration: Any) -> str:
        interp = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        return _record(store, self._ready(path), interp, 1, **declaration)

    def _history_issues(self, result) -> list:
        return [i for i in result.issues if i.code.startswith("history_")]

    def test_a_damaged_import_record_refuses_ready(self, write, import_dir):
        path = standard_file(write)
        import_id = _write_record(path, interpretation())
        _record_file(import_id).write_bytes(b'{"damaged": true}\n')
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "invalid" and result.import_digest is None
        [issue] = self._history_issues(result)
        assert (issue.code, issue.effect) == ("history_unreadable", "invalid")
        assert import_id[:12] in issue.message
        assert "cannot resolve" in issue.message
        # Investigation and a documented recovery that keeps the original
        # bytes and an audit trail; never deletion as a way past the check.
        assert (
            "An operator must investigate each one and, where appropriate, repair "
            "or quarantine it under a documented recovery procedure that preserves "
            "its original bytes and an audit trail." in issue.message
        )
        assert (
            "Deleting a damaged record is not an acceptable way to pass this check."
            in issue.message
        )
        assert "move it out" not in issue.message

    def test_a_damaged_record_is_not_resolved_by_any_confirmation(
        self, write, import_dir
    ):
        path = standard_file(write)
        import_id = _write_record(path, interpretation())
        _record_file(import_id).write_bytes(b"\x00not a record\n")
        everything = standard_bases(
            dialect={"basis": "user_stated"}, data_lines={"basis": "user_stated"}
        )
        result = validate(
            path,
            configuration(interpretation(), everything),
            channel="mcp_tool_argument",
            transport="stdio",
        )
        assert result.outcome == "invalid"
        assert "history_unreadable" in codes(result, "invalid")

    def test_later_lines_that_cannot_be_read_do_not_block(self, write, import_dir):
        # The comparison reads line 1 only, so a torn event line is irrelevant
        # by construction, without relying on its content.
        path = standard_file(write)
        import_id = _write_record(path, interpretation())
        with _record_file(import_id).open("ab") as handle:
            handle.write(b'{"torn":')
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "ready", result.issues
        assert not self._history_issues(result)

    def test_a_store_over_the_limit_is_incomplete_not_empty(
        self, write, import_dir, monkeypatch
    ):
        monkeypatch.setattr(history_module, "HISTORY_RECORD_LIMIT", 1)
        path = standard_file(write)
        other = write(
            "other.csv", corpus.lines_to_bytes(["time [s],acc [g]"] + rows(450))
        )
        _write_record(path, interpretation())
        _write_record(other, interpretation())
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert result.outcome == "invalid"
        [issue] = self._history_issues(result)
        assert issue.code == "history_incomplete"
        assert "holds 2 records, more than the 1" in issue.message

    def test_a_store_at_the_limit_is_compared_in_full(
        self, write, import_dir, monkeypatch
    ):
        monkeypatch.setattr(history_module, "HISTORY_RECORD_LIMIT", 1)
        path = standard_file(write)
        _write_record(
            path,
            interpretation(time_base={"source": "stated", "sampling_rate_hz": "25000"}),
        )
        result = validate(path, configuration(interpretation(), standard_bases()))
        assert not self._history_issues(result)
        assert "prior_import_differs" in codes(result, "notice")

    def test_a_damaged_line_in_a_bound_asset_refuses_ready(self, write, store):
        path = standard_file(write)
        self._bound_record(path, store)
        with store._paths(ASSET)[0].open("ab") as handle:
            handle.write(b"{not json\n")
        result = self._ready(path)
        assert result.outcome == "invalid"
        [issue] = self._history_issues(result)
        assert issue.code == "history_unreadable" and ASSET in issue.message

    def test_a_measurement_the_index_does_not_name_is_still_found(self, write, store):
        # The index is appended after the ledger and separately: a failed or
        # lost index entry must not hide a measurement of these bytes.
        path = standard_file(write)
        self._bound_record(path, store, into="P-202", indexed=False)
        index = Path(config.get_ledger_dir()) / "_measurements.jsonl"
        index.unlink(missing_ok=True)
        result = self._ready(path)
        check = result.ledger_precheck[0]
        assert (check.recorded_asset_id, check.measurement_status) == (
            "P-202",
            "supersedes",
        )
        assert "asset_id" in check.changed

    def test_a_damaged_index_is_not_relied_on(self, write, store):
        path = standard_file(write)
        self._bound_record(path, store)
        index = Path(config.get_ledger_dir()) / "_measurements.jsonl"
        with index.open("ab") as handle:
            handle.write(b'{"measurement_id": "torn')
        result = self._ready(path)
        assert not self._history_issues(result)
        assert result.ledger_precheck[0].measurement_status == "already_recorded"

    def test_damage_in_any_asset_refuses_ready(self, write, store):
        # Whether a damaged line concerns these bytes cannot be told without
        # it, whichever asset holds it.
        path = standard_file(write)
        payload = {
            "measurement_point_id": "fan_de",
            "declaration_version": 1,
            "changed": [],
        }
        store.append(
            "P-303", make_event(EVENT_MEASUREMENT_POINT_DECLARED, "P-303", payload)
        )
        with store._paths("P-303")[0].open("ab") as handle:
            handle.write(b"{not json\n")
        result = self._ready(path)
        assert result.outcome == "invalid"
        [issue] = self._history_issues(result)
        assert issue.code == "history_unreadable" and "P-303" in issue.message

    def test_a_malformed_recorded_measurement_refuses_ready(self, write, store):
        # Adversarial: the AE15 setup with column 3's file block garbled; the
        # ledger's own reader keeps it, so validation must count it.
        data = corpus.lines_to_bytes(
            ["time [s],a [g],b [g],c [g],d [g]"] + rows(400, channels=4)
        )
        path = write("swap.csv", data)
        wrong = _bound((3,))
        first = validate(path, configuration(wrong, _bound_bases((3,))))
        recorded = normalize_configuration(configuration(wrong, _bound_bases((3,))))[
            "interpretation"
        ]
        mid = _record(store, first, recorded, 3, file_block="garbled")
        _baseline(store, mid)
        right = validate(path, configuration(_bound((4,)), _bound_bases((4,))))
        assert right.outcome == "invalid"
        assert any(
            i.code == "history_unreadable" and ASSET in i.message for i in right.issues
        )

    def test_a_malformed_measurement_of_other_bytes_does_not_block(self, write, store):
        # Its measurement id (readable) is derived from other bytes, so it is
        # irrelevant without reading its damaged parts.
        other = write(
            "other.csv", corpus.lines_to_bytes(["time [s],acc [g]"] + rows(450))
        )
        recorded = normalize_configuration(configuration(_bound(), _bound_bases()))[
            "interpretation"
        ]
        elsewhere = validate(other, configuration(_bound(), _bound_bases()))
        _record(store, elsewhere, recorded, 1, file_block="garbled")
        path = standard_file(write)
        result = self._ready(path)
        assert result.outcome == "ready", result.issues
        assert not self._history_issues(result)

    def test_a_moved_measurement_of_these_bytes_is_checked_too(self, write, store):
        # Moved away from P-101 (its current declaration names P-202), a
        # garbled earlier entry of these bytes still counts.
        path = standard_file(write)
        self._bound_record(path, store, file_block="garbled")
        self._bound_record(path, store, asset_id="P-202")
        self._bound_record(path, store, into="P-202")
        result = self._ready(path)
        assert result.outcome == "invalid"
        assert any(
            ASSET in i.message and "cannot be read in full" in i.message
            for i in self._history_issues(result)
        )

    def test_an_empty_ledger_file_holds_nothing(self, write, store):
        # A first append that failed after creating the file leaves 0 bytes.
        path = standard_file(write)
        Path(config.get_ledger_dir()).mkdir(parents=True, exist_ok=True)
        (Path(config.get_ledger_dir()) / "P-707.jsonl").write_bytes(b"")
        result = self._ready(path)
        assert result.outcome == "ready", result.issues

    def test_an_unlisted_ledger_file_refuses_ready(self, write, store):
        path = standard_file(write)
        Path(config.get_ledger_dir()).mkdir(parents=True, exist_ok=True)
        (Path(config.get_ledger_dir()) / "P-505.jsonl").write_bytes(b"{garbled\n")
        result = self._ready(path)
        assert result.outcome == "invalid"
        assert any("P-505.jsonl" in i.message for i in self._history_issues(result))

    def test_a_file_that_is_not_a_ledger_name_is_ignored(self, write, store):
        path = standard_file(write)
        Path(config.get_ledger_dir()).mkdir(parents=True, exist_ok=True)
        (Path(config.get_ledger_dir()) / "notes!.jsonl").write_bytes(b"anything\n")
        result = self._ready(path)
        assert result.outcome == "ready", result.issues

    def test_a_measurement_current_under_two_assets_refuses_ready(self, write, store):
        path = standard_file(write)
        mid = self._bound_record(path, store)
        self._bound_record(path, store, into="P-202")
        result = self._ready(path)
        assert result.outcome == "invalid" and result.import_digest is None
        [issue] = [i for i in self._history_issues(result) if mid in i.message]
        assert issue.effect == "invalid"
        assert (
            f"Measurement {mid} of these bytes is recorded as current under assets "
            f"['P-101', 'P-202'], so its authoritative asset cannot be determined. "
            f"An operator must investigate and reconcile the conflicting records "
            f"before validation can succeed." in issue.message
        )
        assert "cannot resolve this" in issue.message
        assert "did not complete" not in issue.message

    def test_binding_to_one_of_two_holders_stays_invalid(self, write, store):
        # The requested asset being one of the holders does not downgrade it.
        path = standard_file(write)
        self._bound_record(path, store)
        self._bound_record(path, store, into="P-202")
        bound_to_holder = self._ready(path)
        assert bound_to_holder.ledger_precheck[0].asset_id == ASSET
        assert bound_to_holder.outcome == "invalid"
        assert "history_unreadable" in codes(bound_to_holder, "invalid")
        assert "history_unreadable" not in codes(bound_to_holder, "needs_confirmation")

    def test_a_ledger_directory_over_the_limit_is_incomplete(
        self, write, store, monkeypatch
    ):
        monkeypatch.setattr(history_module, "HISTORY_ASSET_LIMIT", 1)
        path = standard_file(write)
        payload = {
            "measurement_point_id": "fan_de",
            "declaration_version": 1,
            "changed": [],
        }
        for asset in ("P-601", "P-602"):
            store.append(
                asset, make_event(EVENT_MEASUREMENT_POINT_DECLARED, asset, payload)
            )
        result = self._ready(path)
        assert result.outcome == "invalid"
        issues = self._history_issues(result)
        assert [i.code for i in issues] == ["history_incomplete"]
        assert "holds 2 asset ledgers, more than the 1" in issues[0].message

    def test_a_ledger_directory_at_the_limit_is_compared_in_full(
        self, write, store, monkeypatch
    ):
        monkeypatch.setattr(history_module, "HISTORY_ASSET_LIMIT", 1)
        path = standard_file(write)
        self._bound_record(path, store)
        result = self._ready(path)
        assert result.outcome == "ready", result.issues
        assert result.ledger_precheck[0].measurement_status == "already_recorded"

    def test_a_cited_record_that_was_not_read_is_not_called_a_mismatch(
        self, write, import_dir, monkeypatch
    ):
        monkeypatch.setattr(history_module, "HISTORY_RECORD_LIMIT", 1)
        path = standard_file(write)
        other = write(
            "other.csv", corpus.lines_to_bytes(["time [s],acc [g]"] + rows(450))
        )
        cited = _write_record(path, interpretation())
        _write_record(other, interpretation())
        bases = standard_bases(
            time_base={"basis": "user_accepted_prior_value", "import_id": cited}
        )
        result = validate(path, configuration(interpretation(), bases))
        assert result.outcome == "invalid"
        mismatch = [i for i in result.issues if i.code == "basis_mismatch"]
        assert [i.field for i in mismatch] == ["time_base"]
        assert "was not read in this validation" in mismatch[0].message
        assert "history_incomplete" in codes(result, "invalid")


class TestReusedInspectionRules:
    """Validation reuses inspection's internal rules for unit tokens and
    time-likeness (kept internal on purpose): on the same bytes, both must
    name the same unit token per column and the same time-like columns, so a
    change to one that the other does not follow fails here."""

    FIXTURES = {
        "european.csv": (corpus.european_export(), ";", ",", 7, 8, 9),
        "standard.csv": (
            corpus.lines_to_bytes(
                ["time [s],acc [g],vel [mm/s]"] + rows(400, channels=2)
            ),
            ",",
            ".",
            1,
            None,
            2,
        ),
        "units_row.csv": (
            corpus.lines_to_bytes(["t,x,y", "s,g,mV"] + rows(400, channels=2)),
            ",",
            ".",
            1,
            2,
            3,
        ),
    }

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_unit_tokens_and_time_columns_agree(self, write, name):
        from predictive_maintenance_mcp.signal_acquisition.text_scan import scan_text
        from predictive_maintenance_mcp.path_safety import open_import_file
        from predictive_maintenance_mcp.signal_import.validation._evidence import (
            _evidence,
            _time_like,
            _unit_token,
        )

        data, delimiter, decimal, header, units, first = self.FIXTURES[name]
        path = write(name, data)
        inspection = inspect_signal_file(str(path))
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": delimiter,
                "decimal": decimal,
                "header_line": header,
                "units_line": units,
            },
            data_lines={"first": first},
        )
        with open_import_file(
            str(path), config.get_import_roots(), base=config.DATA_DIR, max_size=1 << 30
        ) as file:
            evidence = _evidence(
                scan_text(file),
                normalize_configuration(configuration(interp, {}))["interpretation"],
            )
        for column in inspection.columns:
            inspected = column.unit_token.file_text if column.unit_token else None
            assert _unit_token(evidence, column.index) == inspected, column.index
            assert _time_like(evidence, column.index, decimal) == (
                column.role_candidate == "time"
            ), column.index


class TestElectricalChannelsStayDigestOnly:
    """R19's mean and ratio come from moments the decoder streams in digest
    mode: no sample array is kept, so memory does not grow with the file."""

    def _voltage_file(self, write, count: int, name: str = "volts.csv") -> Path:
        body = [
            f"{1000 + 50 * corpus.sample(r, 0):.4f},{2000 + 80 * corpus.sample(r, 1):.4f}"
            for r in range(count)
        ]
        return write(name, corpus.lines_to_bytes(["a [mV],b [mV]"] + body))

    def _interp(self) -> dict[str, Any]:
        def channel(column: int) -> dict[str, Any]:
            return {**_scaled("mV/g", "g"), "column": column}

        return interpretation(time_columns=[], channels=[channel(0), channel(1)])

    def _bases(self) -> dict[str, Any]:
        bases: dict[str, Any] = {"time_base": {"basis": "user_stated"}}
        for column in (0, 1):
            for part in ("unit", "scaling", "dc_policy"):
                bases[f"channels[{column}].{part}"] = {"basis": "user_stated"}
        return bases

    def test_validation_decodes_in_digest_mode_only(self, write, monkeypatch):
        from predictive_maintenance_mcp.signal_import.validation import _entry

        modes: list[str] = []
        real = _entry.decode_text

        def spy(file, interp, *, mode="full"):
            modes.append(mode)
            return real(file, interp, mode=mode)

        monkeypatch.setattr(_entry, "decode_text", spy)
        path = self._voltage_file(write, 600)
        result = validate(path, configuration(self._interp(), self._bases()))
        assert modes == ["digest"]
        assert result.outcome == "ready", result.issues
        means = {c.column: c.mean for c in result.channels}
        assert means[0] == pytest.approx(10.0, abs=0.5) and means[1] == pytest.approx(
            20.0, abs=1
        )
        dc = next(s for s in result.statements if s.field == "channels[0].dc_policy")
        assert "mean" in dc.text and "no offset is subtracted" in dc.text

    def test_memory_does_not_grow_with_the_file(self, write, monkeypatch):
        import tracemalloc

        from predictive_maintenance_mcp.signal_acquisition import text_decoder

        monkeypatch.setattr(text_decoder, "BLOCK_BYTES", 64 * 1024)
        peaks = {}
        for count in (50_000, 200_000):
            path = self._voltage_file(write, count, name=f"volts_{count}.csv")
            tracemalloc.start()
            result = validate(path, configuration(self._interp(), self._bases()))
            _, peaks[count] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert result.outcome == "ready", result.issues
        kept_samples = (200_000 - 50_000) * 2 * 8  # what full mode would add
        assert peaks[200_000] - peaks[50_000] < kept_samples / 4


class TestRefusedTimeColumn:
    def test_a_refused_declared_interval_refuses_the_time_column(self, write):
        data = ["dt: 1e-999999999", "time [s],acc [g]"] + rows(400)
        path = write("refused.csv", corpus.lines_to_bytes(data))
        interp = interpretation(
            dialect={
                "encoding": "utf-8",
                "delimiter": ",",
                "decimal": ".",
                "header_line": 2,
            },
            data_lines={"first": 3},
        )
        bases = standard_bases(
            **{"channels[1].unit": {"basis": "user_accepted_file_value", "line": 2}}
        )
        result = validate(path, configuration(interp, bases))
        assert result.outcome == "invalid"
        refused = [i for i in result.issues if i.field == "time_columns[0]"]
        assert refused and refused[0].effect == "invalid"
        assert refused[0].code == "time_grammar"


class TestConfigureRemaining:
    def test_ambiguous_rows_and_no_rate_or_channels(self, write):
        data = [f"{r},{r % 7}" for r in range(1, 401)]
        path = write("ambiguous.csv", corpus.lines_to_bytes(data))
        draft = configure_import(inspect_signal_file(str(path)))
        remaining = " ".join(draft.remaining_confirmations)
        assert "The dialect" in remaining
        assert "no time column or preamble value" in remaining
        assert "The channels to import" in remaining

    def test_a_cp1252_draft_asks_for_the_encoding(self, write):
        header = "temps [s],accél [g]".encode("cp1252")
        path = write("latin.csv", header + b"\n" + corpus.lines_to_bytes(rows(400)))
        draft = configure_import(inspect_signal_file(str(path)))
        remaining = " ".join(draft.remaining_confirmations)
        assert "The text encoding" in remaining
        assert draft.configuration["interpretation"]["dialect"]["encoding"] == "cp1252"


def test_module_writes_nothing():
    """Validation reads the ledger and the records and never calls a writer,
    in any module of its package."""
    import importlib
    import inspect as introspect
    import pkgutil

    modules = [validation_module] + [
        importlib.import_module(f"{validation_module.__name__}.{info.name}")
        for info in pkgutil.iter_modules(validation_module.__path__)
    ]
    assert len(modules) == 8
    source = "\n".join(introspect.getsource(module) for module in modules)
    writers = (
        "store.append(",
        "ledger.append(",
        "append_index_entry",
        "append_versioned",
        "write_interpretation",
        "append_event",
        "write_bytes",
        "write_text",
        ".mkdir(",
        "os.remove",
        "os.replace",
        "unlink(",
        "os.open(",
    )
    for writer in writers:
        assert writer not in source, writer
    # Files are opened for reading only: every open call names mode "rb".
    opens = re.findall(r"\bopen\(([^)]*)\)", source)
    assert all(args.strip().strip("'\"") == "rb" for args in opens), opens
