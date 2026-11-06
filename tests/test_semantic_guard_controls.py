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
import shutil
from pathlib import Path

import pytest

import _semantic_guards as guards
from _semantic_guards import (
    PLUGIN_DIR,
    REPO_ROOT,
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
    unhandled_values,
    unknown_features,
    unmarked_mentions,
    unread_docx_keys,
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
        path = _copy("skills/bearing-diagnosis/SKILL.md", tmp_path)
        text = path.read_text(encoding="utf-8")
        before = [
            v for v in literal_arguments(text, "skill", {"rpm"}) if "rpm=1797" in v
        ]
        assert before, "the real skill no longer carries rpm=1797; update this control"
        mutated = text.replace("rpm=1797", "rpm=<rpm declared for the signal>")
        after = [v for v in literal_arguments(mutated, "skill", {"rpm"}) if "1797" in v]
        assert after == []


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
        assert any("peak_to_peak" in v for v in unknown_features(text, "skill", valid))
        mutated = text.replace("peak_to_peak", "range")
        assert not any(
            "peak_to_peak" in v for v in unknown_features(mutated, "skill", valid)
        )


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
        assert unmarked_mentions(text, "skill", self.PATTERNS, per_section=False)
        marked = text + "\nThe measurement object is not yet in a published release.\n"
        assert (
            unmarked_mentions(marked, "skill", self.PATTERNS, per_section=False) == []
        )


class TestExampleFilesAndLists:
    def test_paths_into_sample_folders_are_checked(self, tmp_path):
        (tmp_path / "real_train").mkdir()
        (tmp_path / "real_train" / "a.csv").write_text("1\n", encoding="utf-8")
        text = (
            'load_signal(filepath="real_train/a.csv")\n'
            'load_signal(filepath="real_train/missing.csv")\n'
            'load_signal(filepath="motor.bin")\n'
            'load_signal(filepath=["real_train/a.csv", "real_train/gone.csv"])\n'
        )
        assert missing_example_files(text, "s.md", tmp_path) == [
            "s.md:2: load_signal(filepath='real_train/missing.csv')",
            "s.md:4: load_signal(filepath='real_train/gone.csv')",
        ]

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
