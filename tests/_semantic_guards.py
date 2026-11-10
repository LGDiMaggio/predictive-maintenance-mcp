"""Pure guard functions for the semantic consistency tests.

Each guard takes text, file contents or a captured tool outcome and returns
what it found; the ``test_semantic_*`` modules bind the guards to the real
plugin, documentation and server surfaces. Keeping the guards pure lets
``tests/test_semantic_guard_controls.py`` prove, on synthetic inputs and on
mutated copies of real files, that every guard detects the defect it names
and stays quiet when the defect is absent.

Nothing here encodes a diagnostic threshold, label or classification. The
guards compare documentation with the registered server (names, parameters,
output enums), with the file system, or with the code's own declared
contracts.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pytest

from test_documented_calls import KWARG_RE, extract_calls, split_top_level_args

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = (REPO_ROOT / "src").resolve()
PLUGIN_DIR = REPO_ROOT / "plugin"
THIRD_PARTY_MANIFEST = REPO_ROOT / "tools" / "audit" / "third_party_data.json"
SIGNALS_PREFIX = "data/signals/"


# ---------------------------------------------------------------------------
# Plugin text
# ---------------------------------------------------------------------------


def plugin_markdown(plugin_dir: Path = PLUGIN_DIR) -> dict[str, str]:
    """Every markdown file under *plugin_dir* as {posix relative path: text}."""
    return {
        path.relative_to(plugin_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(plugin_dir.rglob("*.md"))
    }


def kwarg_values(args: str) -> dict[str, str]:
    """Keyword arguments of one documented call as {name: raw value text}."""
    values: dict[str, str] = {}
    for segment in split_top_level_args(args):
        match = KWARG_RE.match(segment)
        if match:
            values[match.group(1)] = segment[match.end() :].strip()
    return values


_PLACEHOLDER_RE = re.compile(r"""^["']?<[^<>]+>["']?$""")
_ILLUSTRATIVE_RE = re.compile(r"illustrative", re.IGNORECASE)


def is_placeholder(value: str) -> bool:
    """True when a documented argument value is a placeholder, not a literal.

    Placeholders are written ``<what to pass>``, bare or quoted, or as an
    ellipsis.
    """
    value = value.strip()
    return value in {"...", "…"} or bool(_PLACEHOLDER_RE.match(value))


def _marked_illustrative(lines: list[str], line_number: int) -> bool:
    """True when the call's line, or the previous non-blank line, says the
    values are illustrative."""
    index = line_number - 1
    if _ILLUSTRATIVE_RE.search(lines[index]):
        return True
    for previous in reversed(lines[:index]):
        if previous.strip():
            return bool(_ILLUSTRATIVE_RE.search(previous))
    return False


def literal_arguments(
    text: str,
    source: str,
    params: Iterable[str],
    tools: Iterable[str] | None = None,
) -> list[str]:
    """Documented calls that pass a literal value for any of *params*.

    *tools* limits the check to calls of those endpoints; ``None`` checks
    every documented call. A call on a line marked as illustrative, or whose
    previous non-blank line is, is skipped.
    """
    wanted = set(params)
    only = set(tools) if tools is not None else None
    lines = text.splitlines()
    found = []
    for name, args, line in extract_calls(text):
        if only is not None and name not in only:
            continue
        for key, value in kwarg_values(args).items():
            if key in wanted and not is_placeholder(value):
                if _marked_illustrative(lines, line):
                    continue
                found.append(f"{source}:{line}: {name}({key}={value})")
    return found


def unhandled_values(text: str, values: Iterable[str]) -> list[str]:
    """The *values* (enum members) the text never names as a token."""
    return sorted(
        value
        for value in values
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", text)
    )


_FEATURE_KWARG_RE = re.compile(r"""feature_name\s*=\s*["']([A-Za-z0-9_]+)["']""")
_QUOTED_TOKEN_RE = re.compile(r"""["'`]([a-z][a-z0-9_]*)["'`]""")
_TABLE_ROW_RE = re.compile(r"^\|\s*`?([a-z][a-z0-9_]*)`?\s*\|")


def feature_names(text: str) -> list[tuple[int, str]]:
    """Feature names the text offers, as (line number, name).

    Collected from ``feature_name="..."`` arguments, from quoted names on a
    line that mentions ``feature_name`` (and on its continuation lines), and
    from the first cell of a markdown table whose header names an indicator
    or feature.
    """
    found: list[tuple[int, str]] = []
    in_feature_table = False
    in_feature_list = False
    for number, line in enumerate(text.splitlines(), start=1):
        found.extend((number, m.group(1)) for m in _FEATURE_KWARG_RE.finditer(line))
        stripped = line.strip()
        if "feature_name" in line and not _FEATURE_KWARG_RE.search(line):
            in_feature_list = True
        elif not line.startswith((" ", "\t")) or not stripped:
            in_feature_list = False
        if in_feature_list:
            found.extend(
                (number, m.group(1))
                for m in _QUOTED_TOKEN_RE.finditer(line)
                if m.group(1) != "feature_name"
            )
        if stripped.startswith("|"):
            header = stripped.lower()
            if not in_feature_table and ("indicator" in header or "feature" in header):
                in_feature_table = True
                continue
            if in_feature_table:
                row = _TABLE_ROW_RE.match(stripped)
                if row and not set(row.group(1)) <= set("-"):
                    found.append((number, row.group(1)))
        else:
            in_feature_table = False
    return found


def unknown_features(text: str, source: str, valid: Iterable[str]) -> list[str]:
    """Offered feature names that are not keys of the feature extractor."""
    known = set(valid)
    return [
        f"{source}:{number}: {name}"
        for number, name in feature_names(text)
        if name not in known
    ]


# ---------------------------------------------------------------------------
# DOCX keys, availability, example paths, lists
# ---------------------------------------------------------------------------


_SECTIONS_GET_RE = re.compile(r"""sections\.get\(\s*["']([A-Za-z0-9_]+)["']""")


def renderer_section_keys(renderer_source: str) -> set[str]:
    """The DOCX section keys a renderer reads with ``sections.get("...")``."""
    return set(_SECTIONS_GET_RE.findall(renderer_source))


_DICT_KEY_RE = re.compile(r"""["']([^"']+)["']\s*:""")


def unread_docx_keys(text: str, source: str, renderer_keys: Iterable[str]) -> list[str]:
    """DOCX ``sections`` keys in documented calls that the renderer never reads."""
    readable = set(renderer_keys)
    found = []
    for name, args, line in extract_calls(text):
        if name != "generate_diagnostic_report_docx":
            continue
        sections = kwarg_values(args).get("sections", "")
        found.extend(
            f"{source}:{line}: sections key {key!r}"
            for key in _DICT_KEY_RE.findall(sections)
            if key not in readable
        )
    return found


#: Sentences the plugin uses to say a capability may be missing from the
#: connected server: the prognostics skill's "does not list ... not available
#: in this server version", the "(unreleased)" marker, and the shaft-order
#: skill's "not yet in a published release".
AVAILABILITY_RE = re.compile(
    r"does not list|not available in this server version|\(unreleased\)"
    r"|not yet in a published release",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


def sections_of(text: str) -> list[tuple[int, str]]:
    """The markdown sections of *text* as (first line number, section text)."""
    starts = [0] + [m.start() for m in _HEADING_RE.finditer(text) if m.start()]
    bounds = list(zip(starts, starts[1:] + [len(text)]))
    return [(text.count("\n", 0, a) + 1, text[a:b]) for a, b in bounds]


def unmarked_mentions(
    text: str,
    source: str,
    patterns: Mapping[str, str],
    availability: re.Pattern[str] = AVAILABILITY_RE,
    per_section: bool = True,
) -> list[str]:
    """Mentions of main-only names that no availability sentence covers.

    *patterns* maps a label to a regular expression that finds the mention.
    With *per_section* the sentence must sit in the mention's own markdown
    section; otherwise one sentence anywhere in the file covers it.
    """
    found = []
    scopes = sections_of(text) if per_section else [(1, text)]
    for first_line, section in scopes:
        if availability.search(" ".join(section.split())):
            continue
        for label, pattern in patterns.items():
            for m in re.finditer(pattern, section):
                line = first_line + section.count("\n", 0, m.start())
                found.append(f"{source}:{line}: {label}")
    return found


_QUOTED_PATH_RE = re.compile(r"""["']([^"']+)["']""")


def flat(text: str) -> str:
    """Backticks removed, whitespace collapsed, single quotes made double: the
    normalization the pinned-guidance tests apply, so re-wrapped markdown
    still matches."""
    return " ".join(text.replace("`", "").replace("'", '"').split())


def missing_pinned_text(
    text: str, source: str, calls: Iterable[str], pinned: str
) -> list[str]:
    """A surface that documents any of *calls* but lacks the *pinned* text."""
    names = set(calls)
    called = sorted({name for name, _, _ in extract_calls(text) if name in names})
    if called and flat(pinned) not in flat(text):
        return [f"{source}: calls {called} without the pinned paragraph"]
    return []


def shipped_sample_names(manifest_path: Path = THIRD_PARTY_MANIFEST) -> set[str]:
    """The files *manifest_path* lists under ``data/signals``, relative to it."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        name[len(SIGNALS_PREFIX) :]
        for name in manifest["files"]
        if name.startswith(SIGNALS_PREFIX)
    }


def missing_example_files(
    text: str, source: str, manifest_path: Path = THIRD_PARTY_MANIFEST
) -> list[str]:
    """Literal ``filepath`` arguments that point into a shipped sample folder
    (for example ``real_test/...``) but name no file the manifest lists there.

    The names come from the manifest, so the result is the same whether or
    not the files are in the checkout. Generic names such as ``motor.bin``
    stand for the user's own files and are not checked; only paths whose first
    folder is a sample folder of the manifest claim to be shipped examples.
    """
    shipped = shipped_sample_names(manifest_path)
    folders = {name.split("/", 1)[0] for name in shipped}
    found = []
    for name, args, line in extract_calls(text):
        value = kwarg_values(args).get("filepath")
        if value is None or is_placeholder(value):
            continue
        for relative in _QUOTED_PATH_RE.findall(value) or [value.strip()]:
            parts = Path(relative).parts
            if len(parts) < 2 or parts[0] not in folders:
                continue
            if "/".join(parts) not in shipped:
                found.append(f"{source}:{line}: {name}(filepath={relative!r})")
    return found


_SNAKE_NAME_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def coverage_list(readme: str, heading: str = "## MCP Tool Coverage") -> set[str]:
    """Endpoint names listed in the bullet lines of one README section."""
    if heading not in readme:
        return set()
    section = readme.split(heading, 1)[1].split("\n## ", 1)[0]
    bullets = "\n".join(
        line for line in section.splitlines() if line.startswith("- **")
    )
    return set(_SNAKE_NAME_RE.findall(bullets))


def list_differences(
    listed: Iterable[str], registered: Iterable[str]
) -> dict[str, list[str]]:
    """Names missing from a published list, and names it lists that are not registered."""
    listed_set, registered_set = set(listed), set(registered)
    return {
        "missing": sorted(registered_set - listed_set),
        "unregistered": sorted(listed_set - registered_set),
    }


# ---------------------------------------------------------------------------
# Server outcomes
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What one call produced: a result, or the exception it raised."""

    result: Any = None
    error: BaseException | None = None


async def capture(awaitable) -> Outcome:
    """Await *awaitable* and keep either its result or its exception."""
    try:
        return Outcome(result=await awaitable)
    except Exception as exc:  # noqa: BLE001 - the tests judge every exception
        return Outcome(error=exc)


def capture_sync(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Outcome:
    """Call *function* and keep either its result or its exception."""
    try:
        return Outcome(result=function(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - the tests judge every exception
        return Outcome(error=exc)


def is_project_refusal(exc: BaseException | None, src_dir: Path | None = None) -> bool:
    """True when *exc* was raised by a ``raise`` statement inside ``src/``.

    The innermost frame must lie in ``src/`` and its line must be a ``raise``
    statement: an error a library or a builtin raises while ``src/`` code
    calls it has its innermost Python frame on the calling line, which is
    not a ``raise``, so it does not count as a message the project wrote.
    """
    if exc is None:
        return False
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    last = frames[-1]
    try:
        inside = Path(last.filename).resolve().is_relative_to(src_dir or SRC_DIR)
    except (OSError, ValueError):
        return False
    return inside and (last.line or "").lstrip().startswith("raise")


def require(condition: object, message: str) -> None:
    """Fail the test when a precondition of a tripwire does not hold.

    ``pytest.fail`` raises ``Failed``, which is not an ``AssertionError``, so a
    strict ``xfail(raises=AssertionError)`` reports a broken precondition as
    a real failure instead of counting it as the known defect.
    """
    if not condition:
        pytest.fail(f"precondition: {message}", pytrace=False)


def no_foreign_error(outcome: Outcome, what: str) -> None:
    """A call may refuse with a project-authored message or succeed; any
    other exception is a broken precondition, not the defect."""
    require(
        outcome.error is None or is_project_refusal(outcome.error),
        f"{what} raised {outcome.error!r}",
    )


def refused_or_accepted(outcome: Outcome, accept: Callable[[Any], bool]) -> bool:
    """True when the call refused with a project-authored message, or its
    result satisfies *accept*. Any other exception counts as neither."""
    if outcome.error is not None:
        return is_project_refusal(outcome.error)
    return bool(accept(outcome.result))
