"""Controls for the semantic consistency guards.

A guard that never fires proves nothing. Every guard in ``_semantic_guards``
is shown here to report the defect it names on a synthetic input, to stay
quiet on the corrected input, and to flip when a real repository file is
copied and mutated to add or remove the defect. None of these is an expected
failure: they must all pass.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

import _semantic_guards as guards
from _semantic_guards import (
    PLUGIN_DIR,
    REPO_ROOT,
    VERDICT_LABELS,
    Outcome,
    capture,
    capture_sync,
    coverage_list,
    feature_names,
    flat,
    is_placeholder,
    is_project_refusal,
    list_differences,
    literal_arguments,
    missing_example_files,
    missing_pinned_text,
    refused_or_accepted,
    require,
    renderer_section_keys,
    sentences,
    shipped_sample_names,
    unbound_endpoint_mentions,
    unhandled_values,
    unknown_features,
    unmarked_mentions,
    unread_docx_keys,
    verdict_vocabulary,
)


def _copy(relative: str, tmp_path: Path) -> Path:
    target = tmp_path / Path(relative).name
    shutil.copyfile(PLUGIN_DIR / relative, target)
    return target


class TestPlaceholdersAndLiterals:
    @pytest.mark.parametrize(
        "value", ["<rpm declared for the signal>", '"<id>"', "'<bearing>'", "...", "…"]
    )
    def test_placeholders(self, value):
        assert is_placeholder(value)

    @pytest.mark.parametrize(
        "value", ["1797", '"6205"', "2", '{"GMF": 800.0}', "<a> <b>"]
    )
    def test_literals(self, value):
        assert not is_placeholder(value)

    def test_literal_rpm_is_reported_and_a_placeholder_is_not(self):
        text = 'check_bearing_faults(signal_id="<id>", rpm=1797)\n'
        assert literal_arguments(text, "s.md", {"rpm"}) == [
            "s.md:1: check_bearing_faults(rpm=1797)"
        ]
        fixed = 'check_bearing_faults(signal_id="<id>", rpm=<rpm declared for the signal>)\n'
        assert literal_arguments(fixed, "s.md", {"rpm"}) == []

    def test_an_illustrative_line_is_skipped(self):
        text = (
            "Illustrative values, not your machine's:\ncheck_bearing_faults(rpm=1797)\n"
        )
        assert literal_arguments(text, "s.md", {"rpm"}) == []

    def test_the_tool_filter_limits_the_check(self):
        text = "diagnose_vibration(machine_group=2)\nassess_severity(machine_group=2)\n"
        found = literal_arguments(
            text, "s.md", {"machine_group"}, tools={"diagnose_vibration"}
        )
        assert found == ["s.md:1: diagnose_vibration(machine_group=2)"]

    def test_mutation_on_the_bearing_skill(self, tmp_path):
        """The corrected skill passes no literal speed; putting the audit's
        rpm=1797 back into its calls is reported (audit A-F1)."""
        path = _copy("skills/bearing-diagnosis/SKILL.md", tmp_path)
        text = path.read_text(encoding="utf-8")
        require(
            "rpm=<rpm>" in text,
            "the real skill no longer uses the rpm=<rpm> placeholder; update "
            "this control",
        )
        assert literal_arguments(text, "skill", {"rpm"}) == []
        mutated = text.replace("rpm=<rpm>", "rpm=1797")
        after = literal_arguments(mutated, "skill", {"rpm"})
        assert len(after) >= 3, after
        assert all(v.endswith("(rpm=1797)") for v in after), after


class TestValueDomains:
    def test_unhandled_values(self):
        text = "Statuses: `estimated`, `no_degradation_trend`."
        assert unhandled_values(
            text, ["estimated", "no_degradation_trend", "model_not_rising"]
        ) == ["model_not_rising"]

    def test_a_value_inside_a_longer_token_does_not_count(self):
        assert unhandled_values("`estimated_rate`", ["estimated"]) == ["estimated"]

    def test_feature_names_from_kwargs_lists_and_tables(self):
        text = (
            'analyze_signal_trend(signal_id="<id>", feature_name="rms")\n'
            '- `feature_name`: `"kurtosis"`,\n'
            '  `"bogus"`, ...\n'
            "\n"
            "| Indicator | Use when |\n"
            "|-----------|----------|\n"
            "| crest_factor | impulsive |\n"
        )
        assert [name for _, name in feature_names(text)] == [
            "rms",
            "kurtosis",
            "bogus",
            "crest_factor",
        ]
        assert unknown_features(text, "s.md", {"rms", "kurtosis", "crest_factor"}) == [
            "s.md:3: bogus"
        ]

    def test_mutation_on_the_prognostics_skill(self, tmp_path):
        valid = {"rms", "kurtosis", "crest_factor", "range"}
        text = _copy("skills/prognostics/SKILL.md", tmp_path).read_text(
            encoding="utf-8"
        )
        require(
            '"range"' in text and "| range |" in text,
            "the real skill no longer offers range in its list and table; "
            "update this control",
        )
        assert not any(
            "peak_to_peak" in v for v in unknown_features(text, "skill", valid)
        )
        # Put the refused name back where the audit found it (audit A-F8).
        mutated = text.replace('"range"', '"peak_to_peak"').replace(
            "| range |", "| peak_to_peak |"
        )
        hits = [
            v for v in unknown_features(mutated, "skill", valid) if "peak_to_peak" in v
        ]
        assert len(hits) == 2, hits


class TestDocxKeys:
    def test_renderer_keys_and_unread_keys(self):
        renderer = 'a = sections.get("statistics")\nb = sections.get( "iso" )\n'
        assert renderer_section_keys(renderer) == {"statistics", "iso"}
        text = 'generate_diagnostic_report_docx(signal_id="<id>", sections={"iso": "x", "Summary": "y"})'
        assert unread_docx_keys(text, "s.md", {"statistics", "iso"}) == [
            "s.md:1: sections key 'Summary'"
        ]

    def test_the_real_renderer_reads_section_keys(self):
        source = (REPO_ROOT / "src" / "report_generator.py").read_text(encoding="utf-8")
        assert renderer_section_keys(source)


class TestAvailability:
    PATTERNS = {"`measurement`": r"`measurement`"}

    def test_a_section_without_the_sentence_is_reported(self):
        text = "# A\nRead `measurement`.\n# B\nIf the connected server does not list it, say so.\n"
        assert unmarked_mentions(text, "s.md", self.PATTERNS) == [
            "s.md:2: `measurement`"
        ]

    def test_the_sentence_in_the_same_section_covers_it(self):
        text = "# A\nRead `measurement` (unreleased).\n"
        assert unmarked_mentions(text, "s.md", self.PATTERNS) == []

    def test_file_scope_accepts_a_sentence_anywhere(self):
        text = "# A\nRead `measurement`.\n# B\nNot yet in a published release.\n"
        assert unmarked_mentions(text, "s.md", self.PATTERNS, per_section=False) == []

    def test_a_wrapped_sentence_still_counts(self):
        text = "# A\nRead `measurement`. If the server does\nnot list it, say so.\n"
        assert unmarked_mentions(text, "s.md", self.PATTERNS) == []

    def test_mutation_on_the_signal_management_skill(self, tmp_path):
        text = _copy("skills/signal-management/SKILL.md", tmp_path).read_text(
            encoding="utf-8"
        )
        assert unmarked_mentions(text, "skill", self.PATTERNS, per_section=False) == []
        # Remove the availability bullet and the signal import's availability
        # sentence: the mentions are unmarked again (audit A-F11). The check is
        # file-scoped, so any availability sentence in the file covers them and
        # both must go.
        unmarked, n = re.subn(
            r"- \*\*Availability\*\*:.*?not available in this server version\n",
            "",
            text.replace("\r\n", "\n"),
            flags=re.S,
        )
        require(n == 1, "the availability bullet changed; update this control")
        assert (
            unmarked_mentions(unmarked, "skill", self.PATTERNS, per_section=False) == []
        )
        unmarked, n = re.subn(
            r"`inspect_signal_file`,\s+`validate_signal_import`,.*?"
            r"make\s+`load_signal` read it\.",
            "",
            unmarked,
            flags=re.S,
        )
        require(n == 1, "the import availability sentence changed; update this control")
        assert unmarked_mentions(unmarked, "skill", self.PATTERNS, per_section=False)


class TestEndpointAvailability:
    NAMES = {"forecast_asset_indicator", "assess_shaft_fault_evidence"}

    def test_sentences_split_list_items_and_full_stops(self):
        text = "Intro one. Intro\ntwo.\n- item a\n- item b\n\n# Head\n| x | y |"
        assert sentences(text) == [
            "Intro one.",
            "Intro two.",
            "- item a",
            "- item b",
            "# Head",
            "| x | y |",
        ]

    def test_an_unmarked_endpoint_is_reported(self):
        text = "Call `forecast_asset_indicator(values=[1])`.\n"
        assert unbound_endpoint_mentions(text, "s.md", self.NAMES) == [
            "s.md:1: forecast_asset_indicator"
        ]

    def test_a_sentence_naming_it_covers_it(self):
        text = (
            "Call `forecast_asset_indicator(values=[1])`.\n\n"
            "If the server does not list `forecast_asset_indicator`, say so.\n"
            "- assess_shaft_fault_evidence (unreleased, experimental)\n"
        )
        assert unbound_endpoint_mentions(text, "s.md", self.NAMES) == []

    def test_a_phrase_about_something_else_does_not_cover_it(self):
        text = (
            "Call `assess_shaft_fault_evidence(signal_ids=[])`.\n\n"
            "If the tool does not list `rpm`, omit it.\n"
        )
        assert unbound_endpoint_mentions(text, "s.md", self.NAMES) == [
            "s.md:1: assess_shaft_fault_evidence"
        ]

    def test_mutation_on_the_pipeline_agent(self, tmp_path):
        """The agent names the unreleased shaft-order tool; without its
        availability bullet it is reported, although the file still has an
        availability phrase about rpm (audit A-P-5)."""
        text = _copy("agents/diagnostic-pipeline.md", tmp_path).read_text(
            encoding="utf-8"
        )
        names = {"assess_shaft_fault_evidence"}
        assert unbound_endpoint_mentions(text, "agent", names) == []
        mutated, n = re.subn(
            r"\n- `assess_shaft_fault_evidence` is not yet in a published release",
            "",
            text,
        )
        require(n == 1, "the agent's availability bullet changed; update this control")
        require(
            "does not list" in " ".join(mutated.split()),
            "the agent lost its rpm sentence",
        )
        assert [
            v.split(": ", 1)[1]
            for v in unbound_endpoint_mentions(mutated, "agent", names)
        ] == ["assess_shaft_fault_evidence"]


class TestVerdictVocabulary:
    """S6 (FU-RC0-S6): role-aware, bound to no tripwire; never run against the
    real plugin text as a requirement."""

    @pytest.mark.parametrize(
        "text, role",
        [
            ("- ISO Zone A + Kurtosis < 1 -> Healthy", "arrow-verdict"),
            ("- Kurtosis not above 3 -> Healthy", "arrow-verdict"),
            ("Zone D or kurtosis above 6 → **Critical**", "arrow-verdict"),
            ("Overall: Healthy / Monitor / Suspicious / Critical", "overall-verdict"),
            (
                "| Anomaly ratio | Classification | Action |\n|---|---|---|\n"
                "| Near 0 | Normal | No action needed |",
                "table-verdict",
            ),
            (
                "| Anomaly ratio | Classification | Action |\n|---|---|---|\n"
                "| Near 0 | Normal | No action needed |",
                "plugin-action",
            ),
            ("Schedule follow-up in 2-4 weeks.", "plugin-action"),
            ("Schedule follow-up in 2-4 weeks.", "plugin-interval"),
            ("Re-check in 2" + chr(0x2013) + "4 weeks.", "plugin-interval"),
            ("No impulsive content means the bearing is healthy.", "absence-as-health"),
            (
                "A ratio without flags indicates that the machine is normal.",
                "absence-as-health",
            ),
        ],
    )
    def test_each_verdict_role_is_reported(self, text, role):
        roles = [v.split(": ")[1] for v in verdict_vocabulary(text, "s.md")]
        assert role in roles, roles

    @pytest.mark.parametrize(
        "text",
        [
            "Relay `overall_health` as returned (`Healthy`, `Suspicious` or "
            "`Faulty`) with its `anomaly_ratio`.",
            "`no_indication` is not evidence that the bearing is free of defects.",
            'Never report `no_indication` as "healthy" or "no fault".',
            "Healthy segments cluster together; anomalies appear as outliers.",
            'Never write "No action needed" for a refused assessment.',
            "A ratio without flags does not show that the machine is healthy.",
            "| Report Type | Tool | When to Use |\n|---|---|---|\n"
            "| FFT Spectrum | `generate_fft_report()` | Normal |",
            "Report the zone with its `zone_description` -> Unacceptable as returned",
        ],
    )
    def test_projections_mentions_and_negations_are_not_reported(self, text):
        assert verdict_vocabulary(text, "s.md") == []

    def test_a_server_enum_value_is_still_a_verdict_in_a_verdict_role(self):
        """The allow-list is not textual: "Healthy" and "Suspicious" are
        overall_health values, and a plugin rule assigning them is reported
        even when the server phrases contain them (RC-0 review finding 18)."""
        server = ["Healthy", "Suspicious", "Faulty"]
        text = "- Zone A -> Healthy\n- Zone C -> Suspicious"
        assert len(verdict_vocabulary(text, "s.md", server_phrases=server)) == 2

    def test_a_server_phrase_allows_an_action_it_contains(self):
        text = "Monitor more frequently."
        assert verdict_vocabulary(text, "s.md")
        server = ["Monitor more frequently and plan an inspection."]
        assert verdict_vocabulary(text, "s.md", server_phrases=server) == []

    def test_role_matching_is_what_keeps_a_mention_quiet(self):
        """A text-only matcher would report this sentence: the label is there,
        the role is not."""
        text = "Healthy segments cluster together; anomalies appear as outliers."
        assert any(re.search(rf"\b{label}\b", text) for label in VERDICT_LABELS)
        assert verdict_vocabulary(text, "s.md") == []

    def test_mutation_on_the_screening_command(self, tmp_path):
        """On a copy of the real /pm-screen command the guard reports the four
        arrow verdicts of its decision logic (audit A-F5); replacing them with
        projections of server fields removes every report. Not a tripwire:
        the real file is not required to be clean."""
        text = _copy("commands/pm-screen.md", tmp_path).read_text(encoding="utf-8")
        arrows = [v for v in verdict_vocabulary(text, "cmd") if "arrow-verdict" in v]
        require(
            len(arrows) == 4,
            f"the command's decision logic changed; update this control: {arrows}",
        )
        mutated = re.sub(
            r"(?m)^- ISO Zone .*$",
            "- Relay the zone with its `zone_description` as returned",
            text,
        )
        assert [
            v for v in verdict_vocabulary(mutated, "cmd") if "arrow-verdict" in v
        ] == []


class TestExampleFilesAndLists:
    def test_paths_into_sample_folders_are_checked(self, tmp_path):
        """The sample names come from the manifest; no sample folder exists."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "files": {
                        "data/signals/real_train/a.csv": {},
                        "data/real_bearings/train/b.mat": {},
                    }
                }
            ),
            encoding="utf-8",
        )
        text = (
            'load_signal(filepath="real_train/a.csv")\n'
            'load_signal(filepath="real_train/missing.csv")\n'
            'load_signal(filepath="motor.bin")\n'
            'load_signal(filepath=["real_train/a.csv", "real_train/gone.csv"])\n'
        )
        assert missing_example_files(text, "s.md", manifest) == [
            "s.md:2: load_signal(filepath='real_train/missing.csv')",
            "s.md:4: load_signal(filepath='real_train/gone.csv')",
        ]

    def test_the_committed_manifest_names_the_sample_folders(self):
        """Anchor for the strict A-F14 tripwire: the names are really read."""
        names = shipped_sample_names()
        assert "real_train/OuterRaceFault_1.csv" in names
        assert {name.split("/", 1)[0] for name in names} == {"real_test", "real_train"}

    def test_coverage_list_and_closure_differences(self):
        readme = "## MCP Tool Coverage\n- **A**: load_signal, analyze_fft\n## Next\n- **B**: other_tool\n"
        assert coverage_list(readme) == {"load_signal", "analyze_fft"}
        registered = {"load_signal", "analyze_fft", "fake_new_tool"}
        assert list_differences(coverage_list(readme), registered) == {
            "missing": ["fake_new_tool"],
            "unregistered": [],
        }

    def test_pinned_paragraph_check(self):
        pinned = "If a `check_bearing_faults` result has no `claim` field, say so."
        text = 'Call check_bearing_faults(signal_id="<id>").\n'
        assert missing_pinned_text(text, "s.md", {"check_bearing_faults"}, pinned)
        rewrapped = (
            text + "If a check_bearing_faults result has\nno claim field, say so.\n"
        )
        assert (
            missing_pinned_text(rewrapped, "s.md", {"check_bearing_faults"}, pinned)
            == []
        )
        assert (
            missing_pinned_text(
                "No call here.", "s.md", {"check_bearing_faults"}, pinned
            )
            == []
        )
        assert flat("a `b`\n c 'd'") == 'a b c "d"'


def _load_fake_src(tmp_path: Path):
    """A module living under a fake ``src`` directory, imported from its file."""
    src = tmp_path / "src"
    src.mkdir()
    module_file = src / "fake_tool.py"
    module_file.write_text(
        "def refuse():\n"
        "    raise ValueError(\n"
        "        'refused: give a finite number'\n"
        "    )\n"
        "\n"
        "def builtin_error():\n"
        "    for _ in range(0, 1, 0):\n"
        "        pass\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("fake_tool", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return src.resolve(), module


class TestProjectRefusal:
    def test_a_raise_statement_inside_src_is_a_project_refusal(self, tmp_path):
        src, module = _load_fake_src(tmp_path)
        outcome = capture_sync(module.refuse)
        assert is_project_refusal(outcome.error, src)

    def test_a_builtin_error_on_a_src_line_is_not(self, tmp_path):
        src, module = _load_fake_src(tmp_path)
        outcome = capture_sync(module.builtin_error)
        assert isinstance(outcome.error, ValueError)
        assert not is_project_refusal(outcome.error, src)

    def test_an_error_raised_outside_src_is_not(self, tmp_path):
        src, _ = _load_fake_src(tmp_path)
        outcome = capture_sync(int, "not a number")
        assert not is_project_refusal(outcome.error, src)
        assert not is_project_refusal(None, src)

    def test_refused_or_accepted(self, tmp_path, monkeypatch):
        src, module = _load_fake_src(tmp_path)
        monkeypatch.setattr(guards, "SRC_DIR", src)
        refusal = capture_sync(module.refuse)
        assert refused_or_accepted(refusal, lambda r: False) is is_project_refusal(
            refusal.error, src
        )
        assert refused_or_accepted(Outcome(result={"ok": True}), lambda r: r["ok"])
        assert not refused_or_accepted(Outcome(result={"ok": False}), lambda r: r["ok"])
        assert not refused_or_accepted(capture_sync(int, "x"), lambda r: True)

    def test_a_broken_precondition_is_not_an_assertion_error(self):
        assert not issubclass(pytest.fail.Exception, AssertionError)
        with pytest.raises(pytest.fail.Exception):
            require(False, "broken")
        require(True, "holds")

    def test_capture_keeps_results_and_errors(self):
        async def ok():
            return 1

        async def fail():
            raise RuntimeError("boom")

        assert asyncio.run(capture(ok())).result == 1
        assert isinstance(asyncio.run(capture(fail())).error, RuntimeError)
