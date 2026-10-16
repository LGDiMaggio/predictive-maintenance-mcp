"""Install and setup commands in the user docs must run against this tree.

A reader copies a command out of README.md or INSTALL.md and pastes it into a
terminal. Nothing in CI executes those lines, so they drift without a sound:
an extra that was renamed, a flag that never existed, a ``python -m`` module
that moved, a script that was deleted, an environment variable no code reads.
The last shape is the worst one, because it fails toward success: the server
starts, ignores the setting, and the operator believes it took effect.

This module parses every fenced shell block of :data:`INSTALL_DOCS` and checks
each command against the real source of truth, never against a copy of it:

* pip, ``uv pip`` and ``uvx`` targets that name the project use the
  distribution name in ``pyproject.toml``, and only extras it declares;
* ``predictive-maintenance-mcp`` invocations (and ``python -m
  predictive_maintenance_mcp``, which runs the same ``main``) use only flags
  and transport values the real :func:`build_parser` declares, read off the
  parser object itself;
* ``python -m`` project modules resolve and are runnable, and a
  ``python -m tools.audit`` subcommand is one the CLI has;
* script paths exist in the checkout;
* every environment variable that the docs, ``smithery.yaml``,
  ``.env.example`` or ``docker-compose.yml`` set is read somewhere under
  ``src/``. "Read" is decided by walking the AST, including reads through a
  function that forwards its parameter to ``os.environ.get`` (``server._env``
  reads MCP_TRANSPORT, MCP_HOST and MCP_PORT that way), so it cannot be
  satisfied by a name merely appearing in a comment or a docstring.

Today's violations are listed in :data:`ALLOWLIST`, one entry per (file,
rule, token) with a reason. They are known drift, kept visible rather than
fixed here, and each one is an issue candidate. The comparison runs both
ways: a new violation fails :func:`test_no_violation_outside_the_allowlist`,
and an entry that no longer matches anything fails
:func:`test_the_allowlist_has_no_stale_entries`.

A guard like this can stop finding things and still look exactly like a clean
tree, so its failing paths are tested on small synthetic fixtures, none of
which rests on the real docs (the ``Test*`` classes), and
:func:`test_the_install_docs_actually_parsed` establishes that the real files
yielded commands before any "found nothing" from them is believed.
"""

import ast
import importlib
import importlib.util
import re
import shlex
import tomllib
from pathlib import Path
from typing import Callable, Collection, Iterable, Mapping, NamedTuple, Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The user-facing docs whose shell blocks are checked. One list, so a new
#: setup document joins every rule at once by being added here.
INSTALL_DOCS = (
    "README.md",
    "INSTALL.md",
    "CONTRIBUTING.md",
    "docs/QUICKSTART_DEVELOPER.md",
    "docs/QUICKSTART_ENGINEER.md",
    "docs/DEPLOYMENT.md",
    "docs/OLLAMA_GUIDE.md",
    "docs/QUALITY_AUDITS.md",
)

#: Configuration files that set environment variables for the server.
SMITHERY = "smithery.yaml"
DOTENV_EXAMPLE = ".env.example"
COMPOSE = "docker-compose.yml"

#: Fence info strings read as shell. An empty string is an unlabeled fence:
#: docs often leave a command block unlabeled, and skipping those would let a
#: command escape every rule by losing its label.
SHELL_LANGUAGES = frozenset({"bash", "sh", "shell", "console", "powershell", "ps1", ""})

#: Top-level packages whose ``python -m`` targets must resolve in this tree.
PROJECT_MODULE_ROOTS = ("predictive_maintenance_mcp", "benchmarks", "tools")

#: The audit CLI. Its subcommand vocabulary is owned by its own module.
AUDIT_MODULE = "tools.audit"

#: Known violations at the time this guard landed: (file, rule, token) ->
#: reason. Each is existing drift between a document and the code, recorded
#: here instead of fixed so that new drift stays blocked while old drift
#: stays visible. Remove an entry when the document is corrected; the stale
#: entry test will insist on it.
ALLOWLIST: dict[tuple[str, str, str], str] = {
    (SMITHERY, "env-unread", "DATA_DIR"): (
        "The Smithery launch sets DATA_DIR for the dataDir option; no code "
        "reads it, so the option has no effect (the data root comes from "
        "PDM_PROJECT_DIR)."
    ),
    (DOTENV_EXAMPLE, "env-unread", "DATA_DIR"): (
        "Not read by any code; the data root comes from PDM_PROJECT_DIR."
    ),
    (DOTENV_EXAMPLE, "env-unread", "LOG_LEVEL"): (
        "Not read by any code; the server reads MCP_LOG_LEVEL."
    ),
    (DOTENV_EXAMPLE, "env-unread", "DEFAULT_SAMPLING_RATE"): (
        "Not read by any code; a default sampling rate would also contradict "
        "the declared-sampling-rate policy."
    ),
    (DOTENV_EXAMPLE, "env-unread", "FFT_MAX_SAMPLES"): (
        "Not read by any code; no such FFT limit is configurable."
    ),
    (DOTENV_EXAMPLE, "env-unread", "ENVELOPE_FILTER_LOW"): (
        "Not read by any code; envelope bands are tool parameters."
    ),
    (DOTENV_EXAMPLE, "env-unread", "ENVELOPE_FILTER_HIGH"): (
        "Not read by any code; envelope bands are tool parameters."
    ),
    (DOTENV_EXAMPLE, "env-unread", "ENVELOPE_FILTER_ORDER"): (
        "Not read by any code; the filter order is not configurable."
    ),
    (DOTENV_EXAMPLE, "env-unread", "STATISTICS_WINDOW_SIZE"): (
        "Not read by any code; the file itself marks it as a future feature."
    ),
    (DOTENV_EXAMPLE, "env-unread", "SERVER_NAME"): (
        "Not read by any code; the server name is fixed in src/server.py."
    ),
    (DOTENV_EXAMPLE, "env-unread", "SERVER_VERSION"): (
        "Not read by any code; the version comes from package metadata."
    ),
}


class Violation(NamedTuple):
    """One finding. ``(file, rule, token)`` is its identity in the allowlist;
    the line and detail are for the reader of a failure."""

    file: str
    rule: str
    token: str
    line: int
    detail: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.file, self.rule, self.token)


class Command(NamedTuple):
    """One simple command: ``argv`` after quote removal, located by the
    first physical line it started on."""

    file: str
    line: int
    argv: tuple[str, ...]


class CliOption(NamedTuple):
    takes_value: bool
    choices: Optional[frozenset[str]]


class EnvSetting(NamedTuple):
    file: str
    line: int
    name: str


# ---------------------------------------------------------------------------
# Parsing: markdown fences -> logical lines -> simple commands
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)$")

#: A shell prompt in front of a command (``$ `` or ``PS C:\x> ``).
_PROMPT_RE = re.compile(r"^(?:\$[ \t]+|PS(?:[ \t][^>]*)?>[ \t]*)")

#: Separators between simple commands, as tokens after shlex.
_SEPARATORS = frozenset({"&&", "||", "|", ";", "&", ";;"})

#: A launcher whose remaining arguments are a whole command of their own.
_INSPECTOR = "@modelcontextprotocol/inspector"

_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
_PS_ENV_RE = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)(=?)", re.IGNORECASE)
_PYTHON_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?$", re.IGNORECASE)
_PIP_RE = re.compile(r"^pip(?:\d+(?:\.\d+)*)?(?:\.exe)?$", re.IGNORECASE)


def fenced_blocks(text: str) -> list[tuple[str, int, list[str]]]:
    """``(language, first content line number, lines)`` for every fence.

    The language is the first word of the info string, lowercased. A fence
    closes on a line of the same character, at least as long; an unclosed
    fence runs to the end of the text, as CommonMark has it.
    """
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        opened = _FENCE_OPEN_RE.match(lines[i])
        if not opened:
            i += 1
            continue
        fence = opened.group("fence")
        info = opened.group("info").strip()
        language = info.split()[0].lower() if info else ""
        close = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        blocks.append((language, i + 2, lines[i + 1 : j]))
        i = j + 1
    return blocks


def _strip_comment(line: str) -> str:
    """Drop a ``#`` comment: one at the start of a word, outside quotes."""
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _logical_lines(lines: list[str], first: int) -> list[tuple[int, str]]:
    """Join continuation lines (``\\`` for POSIX shells, a backtick for
    PowerShell, ``^`` for cmd) into ``(line number, text)`` pairs."""
    joined: list[tuple[int, str]] = []
    pending: Optional[tuple[int, str]] = None
    for offset, raw in enumerate(lines):
        number = first + offset
        text = _strip_comment(_PROMPT_RE.sub("", raw.strip(), count=1)).rstrip()
        if pending is not None:
            number, text = pending[0], pending[1] + " " + text.strip()
            pending = None
        if text.endswith(("\\", "`", "^")) and not text.endswith("\\\\"):
            pending = (number, text[:-1].rstrip())
            continue
        if text.strip():
            joined.append((number, text.strip()))
    if pending is not None and pending[1].strip():
        joined.append((pending[0], pending[1].strip()))
    return joined


def _words(text: str) -> list[str]:
    """Shell words with quotes removed and backslashes kept.

    Backslashes are kept because these docs are read on Windows as often as
    on POSIX: ``scripts\\setup-worktree.py`` is a path, not an escape. A line
    with an unbalanced quote is prose in an unlabeled block; plain
    whitespace splitting is the honest reading of it.
    """
    lexer = shlex.shlex(text, posix=True, punctuation_chars="&|;")
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return text.split()
    merged: list[str] = []
    for token in tokens:
        # ``2>&1`` lexes as ``2>``, ``&``, ``1``: a redirect, not a separator.
        if token == "&" and merged and merged[-1].endswith((">", "<")):
            merged[-1] += token
        elif merged and merged[-1].endswith(">&"):
            merged[-1] += token
        else:
            merged.append(token)
    return merged


def _simple_commands(tokens: list[str]) -> list[tuple[str, ...]]:
    """Split a token list at separators, and at the MCP inspector, whose
    remaining arguments are the command it launches."""
    commands: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                commands.append(tuple(current))
            current = []
        elif token.endswith(_INSPECTOR):
            current.append(token)
            commands.append(tuple(current))
            current = []
        else:
            current.append(token)
    if current:
        commands.append(tuple(current))
    return commands


def shell_commands(text: str, source: str) -> list[Command]:
    """Every simple command in the shell-language fences of *text*.

    Only fenced blocks are read: a command quoted in a sentence is prose,
    and the anti-rot count on the real docs catches a parser that stopped
    reading blocks altogether.
    """
    commands = []
    for language, first, lines in fenced_blocks(text):
        if language not in SHELL_LANGUAGES:
            continue
        for number, line in _logical_lines(lines, first):
            for argv in _simple_commands(_words(line)):
                commands.append(Command(source, number, argv))
    return commands


def _basename(token: str) -> str:
    return re.split(r"[\\/]", token)[-1]


def _env_prefix(argv: tuple[str, ...]) -> tuple[list[str], tuple[str, ...]]:
    """Split leading ``NAME=value`` assignments off a command."""
    names = []
    index = 0
    while index < len(argv) and (assignment := _ASSIGNMENT_RE.match(argv[index])):
        names.append(assignment.group(1))
        index += 1
    return names, argv[index:]


# ---------------------------------------------------------------------------
# Rule 1: install targets name the real distribution and real extras
# ---------------------------------------------------------------------------

#: Installer options whose next word is a value, not a target.
_VALUE_OPTIONS = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-t",
        "--target",
        "--prefix",
        "--root",
        "--python",
        "-p",
        "--python-version",
        "--platform",
        "--with",
        "--with-editable",
        "--with-requirements",
    }
)

_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[(?P<extras>[^\]]*)\])?(?P<rest>.*)$"
)
_PATH_EXTRAS_RE = re.compile(r"^(?P<path>.*?)\[(?P<extras>[^\]]*)\]$")


def canonical_name(name: str) -> str:
    """PEP 503 normalization, which is also how PEP 685 compares extras."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_local_path(target: str) -> bool:
    return (
        target.startswith((".", "/", "\\", "~", "$"))
        or re.match(r"^[A-Za-z]:[\\/]", target) is not None
        or "/" in target
        or "\\" in target
    )


def _uvx_target(args: tuple[str, ...]) -> tuple[Optional[str], Optional[str], int]:
    """``(--from value, command target, index after the target)`` of uvx args."""
    source = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--from" and index + 1 < len(args):
            source = args[index + 1]
            index += 2
        elif arg.startswith("--from="):
            source = arg.split("=", 1)[1]
            index += 1
        elif arg in _VALUE_OPTIONS:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            return source, arg, index + 1
    return source, None, index


def _install_targets(argv: tuple[str, ...]) -> list[str]:
    """The requirement or path targets of a pip / uv pip / uvx command."""
    _, argv = _env_prefix(argv)
    if not argv:
        return []
    program = _basename(argv[0]).lower()
    if _PIP_RE.match(program) and argv[1:2] == ("install",):
        args = argv[2:]
    elif _PYTHON_RE.match(program) and argv[1:4] == ("-m", "pip", "install"):
        args = argv[4:]
    elif program in ("uv", "uv.exe") and argv[1:3] == ("pip", "install"):
        args = argv[3:]
    elif program in ("uvx", "uvx.exe") or (
        program in ("uv", "uv.exe") and argv[1:3] == ("tool", "run")
    ):
        source, target, _ = _uvx_target(argv[1:] if "uvx" in program else argv[3:])
        if source is not None:
            return [source]
        return [target] if target is not None else []
    else:
        return []
    targets = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-e", "--editable") and index + 1 < len(args):
            targets.append(args[index + 1])
            index += 2
        elif arg.startswith("--editable="):
            targets.append(arg.split("=", 1)[1])
            index += 1
        elif arg in _VALUE_OPTIONS:
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            targets.append(arg)
            index += 1
    return targets


def install_target_violations(
    commands: Iterable[Command], dist_name: str, extras: Collection[str]
) -> list[Violation]:
    """Install targets that misname the project or ask for an unknown extra.

    A target names the project when its normalized name shares the first two
    words of the distribution name (``predictive-maintenance``): that is how
    a misspelling is told apart from an unrelated package. A local path is
    this checkout (docs install it from the clone), so its extras are checked
    too. URLs are left alone.
    """
    wanted = canonical_name(dist_name)
    stem = "-".join(wanted.split("-")[:2])
    known = {canonical_name(e) for e in extras}
    violations = []
    for command in commands:
        for target in _install_targets(command.argv):
            if "://" in target:
                continue
            if _is_local_path(target):
                matched = _PATH_EXTRAS_RE.match(target)
                requested = matched.group("extras") if matched else ""
            else:
                matched = _REQUIREMENT_RE.match(target)
                if not matched:
                    continue
                name = canonical_name(matched.group("name"))
                if not name.startswith(stem):
                    continue
                if name != wanted:
                    violations.append(
                        Violation(
                            command.file,
                            "dist-name",
                            matched.group("name"),
                            command.line,
                            f"installs '{matched.group('name')}', but the "
                            f"distribution is '{dist_name}'",
                        )
                    )
                    continue
                requested = matched.group("extras") or ""
            for extra in (e.strip() for e in requested.split(",")):
                if extra and canonical_name(extra) not in known:
                    violations.append(
                        Violation(
                            command.file,
                            "extra",
                            extra,
                            command.line,
                            f"'{target}' asks for extra '{extra}', not in "
                            f"[project.optional-dependencies] {sorted(known)}",
                        )
                    )
    return violations


# ---------------------------------------------------------------------------
# Rule 2: server invocations use real flags and real transport values
# ---------------------------------------------------------------------------


def cli_surface(parser) -> dict[str, CliOption]:
    """``{option string: CliOption}`` read off an argparse parser.

    ``_actions`` is argparse's internal list, but it is the only place the
    declared options live; reading it keeps this guard bound to the parser
    the console script runs instead of to a copy of its help text.
    """
    surface = {}
    for action in parser._actions:
        choices = frozenset(map(str, action.choices)) if action.choices else None
        for option in action.option_strings:
            surface[option] = CliOption(action.nargs != 0, choices)
    return surface


def _server_arguments(
    argv: tuple[str, ...], script: str, package: str
) -> Optional[tuple[str, ...]]:
    """The arguments passed to the server, or None if *argv* does not run it.

    The server runs as the console script in command position (after any
    ``NAME=value`` prefixes), as ``python -m <package>``, or as a ``uvx``
    target. The script name merely appearing as an argument, for example as
    a pip target, is not an invocation.
    """
    _, argv = _env_prefix(argv)
    if not argv:
        return None
    program = _basename(argv[0])
    if re.sub(r"\.exe$", "", program, flags=re.IGNORECASE) == script:
        return argv[1:]
    for index, token in enumerate(argv):
        if _PYTHON_RE.match(_basename(token)):
            rest = argv[index + 1 :]
            if rest[:2] == ("-m", package):
                return rest[2:]
    if program.lower() in ("uvx", "uvx.exe"):
        _, target, after = _uvx_target(argv[1:])
        if target is not None:
            matched = _REQUIREMENT_RE.match(target)
            if matched and canonical_name(matched.group("name")) == canonical_name(
                script
            ):
                return argv[1 + after :]
    return None


def cli_flag_violations(
    commands: Iterable[Command],
    script: str,
    package: str,
    options: Mapping[str, CliOption],
) -> list[Violation]:
    """Flags the parser does not declare, and values outside its choices.

    Abbreviations are reported although argparse accepts them: a documented
    prefix silently breaks the day another flag shares it. Positional words
    are not judged (a usage line such as ``[OPTIONS]`` is not a command).
    """
    violations = []
    for command in commands:
        arguments = _server_arguments(command.argv, script, package)
        if arguments is None:
            continue
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            index += 1
            if argument == "--":
                break
            if not argument.startswith("-") or argument == "-":
                continue
            flag, attached = argument, None
            if argument.startswith("--") and "=" in argument:
                flag, attached = argument.split("=", 1)
            elif not argument.startswith("--") and len(argument) > 2:
                flag, attached = argument[:2], argument[2:]
            option = options.get(flag)
            if option is None:
                violations.append(
                    Violation(
                        command.file,
                        "cli-flag",
                        flag,
                        command.line,
                        f"'{flag}' is not an option of {script} "
                        f"(known: {sorted(options)})",
                    )
                )
                continue
            if not option.takes_value:
                continue
            value = attached
            if value is None and index < len(arguments):
                value = arguments[index]
                index += 1
            if option.choices is not None and value not in option.choices:
                violations.append(
                    Violation(
                        command.file,
                        "cli-choice",
                        f"{flag}={value}",
                        command.line,
                        f"{flag} {value!r} is not one of {sorted(option.choices)}",
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# Rule 3: python -m project modules resolve; audit subcommands exist
# ---------------------------------------------------------------------------


def _module_runs(argv: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """``(module, its arguments)`` for every ``python ... -m module`` in argv."""
    runs = []
    for index, token in enumerate(argv):
        if not _PYTHON_RE.match(_basename(token)):
            continue
        rest = argv[index + 1 :]
        position = 0
        while position < len(rest) and rest[position].startswith("-"):
            if rest[position] == "-m" and position + 1 < len(rest):
                runs.append((rest[position + 1], rest[position + 2 :]))
                break
            if rest[position] == "-c":
                break
            position += 1
    return runs


def module_violations(
    commands: Iterable[Command],
    is_runnable: Callable[[str], bool],
    audit_subcommands: Callable[[], Collection[str]],
) -> list[Violation]:
    """``python -m`` project modules that do not run, and audit subcommands
    the audit CLI does not have.

    *audit_subcommands* is called only when a ``python -m tools.audit
    <sub>`` command is present, so the vocabulary is loaded from its owner
    lazily and never copied here.
    """
    commands = list(commands)
    violations = []
    vocabulary: Optional[frozenset[str]] = None
    for command in commands:
        for module, arguments in _module_runs(command.argv):
            if module.split(".")[0] not in PROJECT_MODULE_ROOTS:
                continue
            if not is_runnable(module):
                violations.append(
                    Violation(
                        command.file,
                        "module",
                        module,
                        command.line,
                        f"'python -m {module}' does not resolve to a runnable "
                        f"module in this checkout",
                    )
                )
                continue
            if module != AUDIT_MODULE or not arguments:
                continue
            subcommand = arguments[0]
            if subcommand.startswith("-"):
                continue
            if vocabulary is None:
                vocabulary = frozenset(audit_subcommands())
            if subcommand not in vocabulary:
                violations.append(
                    Violation(
                        command.file,
                        "audit-subcommand",
                        subcommand,
                        command.line,
                        f"'python -m {AUDIT_MODULE} {subcommand}': not one of "
                        f"{sorted(vocabulary)}",
                    )
                )
    return violations


def module_is_runnable(name: str) -> bool:
    """Whether ``python -m name`` would find something to run.

    A package runs its ``__main__``, so resolving the package alone is not
    enough. ``find_spec`` imports the parent packages but never runs the
    module itself.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if spec.submodule_search_locations is not None:
        return module_is_runnable(f"{name}.__main__")
    return True


def load_audit_subcommands() -> frozenset[str]:
    """The audit CLI's subcommands, from the module that dispatches them."""
    main = importlib.import_module(f"{AUDIT_MODULE}.__main__")
    return frozenset(main.SUBCOMMANDS)


# ---------------------------------------------------------------------------
# Rule 4: scripts named in commands exist
# ---------------------------------------------------------------------------


def _script_paths(argv: tuple[str, ...]) -> list[str]:
    """``python X.py`` and PowerShell ``-File X.ps1`` paths in *argv*."""
    paths = []
    for index, token in enumerate(argv):
        rest = argv[index + 1 :]
        if token.lower() == "-file" and rest:
            paths.append(rest[0])
        elif _PYTHON_RE.match(_basename(token)):
            for argument in rest:
                if argument in ("-m", "-c"):
                    break
                if argument.startswith("-"):
                    continue
                if argument.lower().endswith(".py"):
                    paths.append(argument)
                break
    return paths


def script_violations(
    commands: Iterable[Command], exists: Callable[[str], bool]
) -> list[Violation]:
    """Relative script paths that do not exist. An absolute path points
    outside the checkout, so there is nothing here to check it against."""
    violations = []
    for command in commands:
        for path in _script_paths(command.argv):
            normalized = path.replace("\\", "/")
            if normalized.startswith(("/", "~", "$")) or re.match(
                r"^[A-Za-z]:/", normalized
            ):
                continue
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if not exists(normalized):
                violations.append(
                    Violation(
                        command.file,
                        "script",
                        normalized,
                        command.line,
                        f"'{path}' does not exist in the checkout",
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# Rule 5: environment variables that are set are also read
# ---------------------------------------------------------------------------


def env_settings_from_commands(commands: Iterable[Command]) -> list[EnvSetting]:
    """Variables a command sets: ``NAME=value cmd`` prefixes, ``export``,
    ``env``, cmd's ``set``, PowerShell's ``$env:NAME =``, and the container
    forms ``docker run -e`` and ``az ... --environment-variables``."""
    settings = []
    for command in commands:
        prefix, rest = _env_prefix(command.argv)
        names = list(prefix)
        program = rest[0].lower() if rest else ""
        if program in ("export", "set", "env"):
            names += [
                m.group(1) for token in rest[1:] if (m := _ASSIGNMENT_RE.match(token))
            ]
        index = 0
        while index < len(rest):
            token = rest[index]
            if token in ("-e", "--env") and index + 1 < len(rest):
                if m := _ASSIGNMENT_RE.match(rest[index + 1]):
                    names.append(m.group(1))
            elif token.startswith("--env="):
                if m := _ASSIGNMENT_RE.match(token.split("=", 1)[1]):
                    names.append(m.group(1))
            elif token == "--environment-variables":
                following = index + 1
                while following < len(rest) and (
                    m := _ASSIGNMENT_RE.match(rest[following])
                ):
                    names.append(m.group(1))
                    following += 1
            elif m := _PS_ENV_RE.match(token):
                assigned = m.group(2) == "=" or (
                    index + 1 < len(rest) and rest[index + 1].startswith("=")
                )
                if assigned and not token[m.end() :].startswith("="):
                    names.append(m.group(1))
            index += 1
        settings += [EnvSetting(command.file, command.line, name) for name in names]
    return settings


def _line_of(text: str, needle: str) -> int:
    position = text.find(needle)
    return text.count("\n", 0, position) + 1 if position >= 0 else 0


def _object_keys_after(expression: str, anchor: re.Pattern) -> list[str]:
    """Keys of the JS object literals in the value that follows *anchor*.

    Walks the value with a stack of open brackets and stops at the comma or
    bracket that ends it, so ``env: cond ? { A: x } : { B: y }`` yields both
    branches. A key is an identifier or quoted string directly after ``{``
    or ``,`` inside a brace, followed by ``:``: a ternary's ``:`` follows a
    value, never one of those, so it is not mistaken for a key.
    """
    keys = []
    for found in anchor.finditer(expression):
        index = found.end()
        # One entry per open bracket, True for a brace.
        braces: list[bool] = []
        expect_key = False
        while index < len(expression):
            char = expression[index]
            if char in "([{":
                braces.append(char == "{")
                expect_key = char == "{"
            elif char in ")]}":
                if not braces:
                    break
                braces.pop()
                expect_key = False
            elif char == ",":
                if not braces:
                    break
                expect_key = braces[-1]
            elif expect_key and not char.isspace():
                key = re.match(
                    r"""(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][\w$]*))\s*:""",
                    expression[index:],
                )
                expect_key = False
                if key:
                    keys.append(next(g for g in key.groups() if g))
                    index += key.end()
                    continue
            index += 1
    return keys


def env_settings_from_smithery(text: str, source: str = SMITHERY) -> list[EnvSetting]:
    """Variables the Smithery launcher passes, read from the ``env`` object
    of its ``commandFunction`` (a JavaScript function in the YAML)."""
    config = yaml.safe_load(text) or {}
    function = (config.get("startCommand") or {}).get("commandFunction") or ""
    names = _object_keys_after(function, re.compile(r"\benv\s*:"))
    return [EnvSetting(source, _line_of(text, name), name) for name in names]


def env_settings_from_dotenv(
    text: str, source: str = DOTENV_EXAMPLE
) -> list[EnvSetting]:
    settings = []
    for number, line in enumerate(text.splitlines(), start=1):
        matched = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if matched:
            settings.append(EnvSetting(source, number, matched.group(1)))
    return settings


def env_settings_from_compose(text: str, source: str = COMPOSE) -> list[EnvSetting]:
    """``environment`` of every compose service, in list or mapping form."""
    config = yaml.safe_load(text) or {}
    settings = []
    for service in (config.get("services") or {}).values():
        environment = (service or {}).get("environment") or []
        if isinstance(environment, dict):
            names = [str(name) for name in environment]
        else:
            names = [str(entry).split("=", 1)[0] for entry in environment]
        settings += [EnvSetting(source, _line_of(text, name), name) for name in names]
    return settings


def _is_environ(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "environ" and isinstance(node.value, ast.Name)
    return isinstance(node, ast.Name) and node.id == "environ"


def _first_argument(call: ast.Call, keyword: str) -> Optional[ast.expr]:
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _env_read_key(node: ast.AST) -> Optional[ast.expr]:
    """The key expression of ``os.environ.get(k)``, ``os.getenv(k)`` or
    ``os.environ[k]``, or None when *node* is not an environment read."""
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        return node.slice
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if isinstance(function, ast.Attribute):
        if function.attr == "get" and _is_environ(function.value):
            return _first_argument(node, "key")
        if function.attr == "getenv" and isinstance(function.value, ast.Name):
            return _first_argument(node, "key")
    if isinstance(function, ast.Name) and function.id == "getenv":
        return _first_argument(node, "key")
    return None


def _called_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _wrapper_key(call: ast.Call, wrapper: tuple[int, str]) -> Optional[ast.expr]:
    position, parameter = wrapper
    if position < len(call.args):
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == parameter), None)


def _key_via_wrappers(
    node: ast.AST, wrappers: Mapping[str, tuple[int, str]]
) -> Optional[ast.expr]:
    """The key of an environment read at *node*, direct or through one of
    *wrappers*; None when *node* reads nothing."""
    key = _env_read_key(node)
    if key is None and isinstance(node, ast.Call):
        name = _called_name(node)
        if name in wrappers:
            key = _wrapper_key(node, wrappers[name])
    return key


def _parse_all(sources: Mapping[str, str]) -> list[ast.Module]:
    """``{path: source}`` parsed, one module per file."""
    return [ast.parse(source, filename=path) for path, source in sources.items()]


def env_wrappers(
    sources: Mapping[str, str], *, trees: Optional[list[ast.Module]] = None
) -> dict[str, tuple[int, str]]:
    """Functions that forward a parameter to an environment read.

    ``{function name: (argument position, parameter name)}``. Found to a
    fixed point, so a wrapper of a wrapper counts too. Keyed by bare name: a
    second function sharing a wrapper's name elsewhere would be trusted as
    one, which is the cost of not resolving imports here.

    *trees*, when given, is :func:`_parse_all` of *sources*, already
    parsed by the caller.
    """
    if trees is None:
        trees = _parse_all(sources)
    functions = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    wrappers: dict[str, tuple[int, str]] = {}
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name in wrappers:
                continue
            parameters = [a.arg for a in function.args.posonlyargs + function.args.args]
            if parameters[:1] in (["self"], ["cls"]):
                parameters = parameters[1:]
            for node in ast.walk(function):
                key = _key_via_wrappers(node, wrappers)
                if isinstance(key, ast.Name) and key.id in parameters:
                    wrappers[function.name] = (parameters.index(key.id), key.id)
                    changed = True
                    break
    return wrappers


def env_reads(
    sources: Mapping[str, str],
    *,
    trees: Optional[list[ast.Module]] = None,
    wrappers: Optional[Mapping[str, tuple[int, str]]] = None,
) -> set[str]:
    """Every variable name read as a string literal, directly or through a
    wrapper from :func:`env_wrappers`.

    *trees* and *wrappers*, when given, are :func:`_parse_all` and
    :func:`env_wrappers` of *sources*, already computed by the caller.
    """
    if trees is None:
        trees = _parse_all(sources)
    if wrappers is None:
        wrappers = env_wrappers(sources, trees=trees)
    names = set()
    for tree in trees:
        for node in ast.walk(tree):
            key = _key_via_wrappers(node, wrappers)
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    return names


def unread_env_violations(
    settings: Iterable[EnvSetting], reads: Collection[str]
) -> list[Violation]:
    return [
        Violation(
            setting.file,
            "env-unread",
            setting.name,
            setting.line,
            f"sets {setting.name}, which no code under src/ reads",
        )
        for setting in settings
        if setting.name not in reads
    ]


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(_read("pyproject.toml"))


@pytest.fixture(scope="module")
def doc_commands() -> list[Command]:
    commands = []
    for relative in INSTALL_DOCS:
        commands += shell_commands(_read(relative), relative)
    return commands


@pytest.fixture(scope="module")
def cli_options() -> dict[str, CliOption]:
    """The real parser's surface. The env vars it reads are cleared first:
    an invalid MCP_TRANSPORT in the developer's shell would make
    ``build_parser`` exit, which is its job, not this guard's."""
    from predictive_maintenance_mcp.server import build_parser

    with pytest.MonkeyPatch.context() as patch:
        for name in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT"):
            patch.delenv(name, raising=False)
        return cli_surface(build_parser())


@pytest.fixture(scope="module")
def src_sources() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "src").rglob("*.py"))
        if "__pycache__" not in path.parts
    }


@pytest.fixture(scope="module")
def src_trees(src_sources) -> list[ast.Module]:
    """``src/`` parsed once for every environment-read check below."""
    return _parse_all(src_sources)


@pytest.fixture(scope="module")
def src_env_wrappers(src_sources, src_trees) -> dict[str, tuple[int, str]]:
    return env_wrappers(src_sources, trees=src_trees)


@pytest.fixture(scope="module")
def src_env_reads(src_sources, src_trees, src_env_wrappers) -> set[str]:
    return env_reads(src_sources, trees=src_trees, wrappers=src_env_wrappers)


@pytest.fixture(scope="module")
def all_violations(
    doc_commands, pyproject, cli_options, src_env_reads
) -> list[Violation]:
    project = pyproject["project"]
    (script,) = project["scripts"]
    settings = (
        env_settings_from_commands(doc_commands)
        + env_settings_from_smithery(_read(SMITHERY))
        + env_settings_from_dotenv(_read(DOTENV_EXAMPLE))
        + env_settings_from_compose(_read(COMPOSE))
    )
    return (
        install_target_violations(
            doc_commands, project["name"], project["optional-dependencies"]
        )
        + cli_flag_violations(
            doc_commands, script, "predictive_maintenance_mcp", cli_options
        )
        + module_violations(doc_commands, module_is_runnable, load_audit_subcommands)
        + script_violations(doc_commands, lambda p: (REPO_ROOT / p).is_file())
        + unread_env_violations(settings, src_env_reads)
    )


def test_the_install_docs_actually_parsed(doc_commands):
    """Anchor for the rest: "found nothing" has to mean something."""
    shell_labeled = [
        command
        for relative in INSTALL_DOCS
        for language, first, lines in fenced_blocks(_read(relative))
        if language
        for command in shell_commands(f"```{language}\n" + "\n".join(lines), relative)
    ]
    assert len(shell_labeled) >= 15, (
        f"only {len(shell_labeled)} commands parsed from labeled shell blocks "
        f"of {INSTALL_DOCS}: the fence parser stopped reading them"
    )
    programs = {_basename(c.argv[0]) for c in doc_commands}
    assert {"pip", "predictive-maintenance-mcp", "python"} <= programs, programs
    files = {c.file for c in doc_commands}
    assert files >= {"README.md", "INSTALL.md", "docs/DEPLOYMENT.md"}, files


def test_the_real_parser_and_config_actually_parsed(
    cli_options, src_env_reads, src_env_wrappers, pyproject
):
    assert {"--transport", "-t", "--host", "--port"} <= set(cli_options)
    assert cli_options["--transport"].choices, "the transport lost its choices"
    assert "sse" in pyproject["project"]["optional-dependencies"]
    expected_reads = {"PDM_PROJECT_DIR", "MCP_TRANSPORT", "MCP_HOST", "MCP_PORT"}
    assert expected_reads <= src_env_reads, src_env_reads
    assert "_env" in src_env_wrappers
    assert env_settings_from_smithery(_read(SMITHERY)), "smithery env not parsed"
    assert len(env_settings_from_dotenv(_read(DOTENV_EXAMPLE))) >= 5
    assert env_settings_from_compose(_read(COMPOSE)), "compose env not parsed"


def test_no_violation_outside_the_allowlist(all_violations):
    """The guard itself: new drift in any install doc fails here."""
    unexpected = [v for v in all_violations if v.key not in ALLOWLIST]
    assert not unexpected, "\n".join(
        f"{v.file}:{v.line}: [{v.rule}] {v.detail}" for v in unexpected
    ) + (
        "\nFix the document. Only if the drift must stay for now, add "
        "(file, rule, token) to ALLOWLIST with a reason."
    )


def test_the_allowlist_has_no_stale_entries(all_violations):
    """An excuse for drift that is gone excuses nothing."""
    gone = sorted(set(ALLOWLIST) - {v.key for v in all_violations})
    assert not gone, f"ALLOWLIST outlived what it excused: {gone}"


def test_every_allowlist_entry_has_a_reason():
    assert all(reason.strip() for reason in ALLOWLIST.values())


# ---------------------------------------------------------------------------
# Synthetic fixtures: each check's failing paths, none of it on the real docs
# ---------------------------------------------------------------------------

EXTRAS = frozenset({"sse", "dev", "docs", "full"})
DIST = "predictive-maintenance-mcp"
SCRIPT = "predictive-maintenance-mcp"
PACKAGE = "predictive_maintenance_mcp"
OPTIONS = {
    "-h": CliOption(False, None),
    "--help": CliOption(False, None),
    "--transport": CliOption(True, frozenset({"stdio", "sse", "streamable-http"})),
    "-t": CliOption(True, frozenset({"stdio", "sse", "streamable-http"})),
    "--host": CliOption(True, None),
    "--port": CliOption(True, None),
    "-p": CliOption(True, None),
}


def doc(*lines, lang="bash"):
    """A markdown document holding *lines* in one fenced block."""
    return "Intro prose.\n\n```" + lang + "\n" + "\n".join(lines) + "\n```\n"


def cmds(*lines, lang="bash"):
    return shell_commands(doc(*lines, lang=lang), "fixture.md")


def keys(violations):
    return sorted((v.rule, v.token) for v in violations)


class TestShellCommandExtraction:
    def test_a_fenced_command_is_parsed(self):
        commands = cmds("pip install predictive-maintenance-mcp")
        assert [c.argv for c in commands] == [
            ("pip", "install", "predictive-maintenance-mcp")
        ]
        assert commands[0].line == 4

    def test_a_command_in_prose_is_not_a_command(self):
        """The same command outside a fence yields nothing."""
        text = "Run pip install predictive-maintenance-mcp[nope] to begin.\n"
        assert shell_commands(text, "fixture.md") == []

    def test_a_non_shell_block_is_not_read(self):
        assert cmds("pip install x", lang="python") == []

    @pytest.mark.parametrize(
        "lang", ["bash", "sh", "shell", "console", "powershell", "ps1", ""]
    )
    def test_every_shell_label_is_read(self, lang):
        assert cmds("pip install x", lang=lang) != []

    def test_chained_commands_split_into_separate_commands(self):
        commands = cmds('pip install -e ".[dev]" && pytest -v; black --check src')
        assert [c.argv[0] for c in commands] == ["pip", "pytest", "black"]

    def test_comments_and_prompts_are_dropped(self):
        commands = cmds("# a comment", "$ pip install x  # trailing", lang="console")
        assert [c.argv for c in commands] == [("pip", "install", "x")]

    def test_continuation_lines_join_into_one_command(self):
        commands = cmds("az container create \\", "  --name x \\", "  --ports 8000")
        assert len(commands) == 1
        assert commands[0].argv[-1] == "8000"

    def test_a_redirect_is_not_a_command_separator(self):
        commands = cmds("python -m benchmarks.cwru all 2>&1")
        assert len(commands) == 1

    def test_windows_paths_keep_their_backslashes(self):
        commands = cmds(r"python scripts\setup-worktree.py", lang="powershell")
        assert commands[0].argv == ("python", r"scripts\setup-worktree.py")

    def test_an_unclosed_fence_still_yields_its_commands(self):
        text = "```bash\npip install x\n"
        assert shell_commands(text, "fixture.md") != []

    def test_empty_input_yields_nothing(self):
        assert shell_commands("", "fixture.md") == []


class TestInstallTargets:
    def check(self, *lines, lang="bash"):
        return install_target_violations(cmds(*lines, lang=lang), DIST, EXTRAS)

    def test_a_declared_extra_passes(self):
        assert self.check("pip install predictive-maintenance-mcp[sse]") == []

    def test_an_unknown_extra_is_reported(self):
        assert keys(self.check("pip install predictive-maintenance-mcp[nope]")) == [
            ("extra", "nope")
        ]

    def test_an_unknown_extra_next_to_valid_ones_is_reported(self):
        """A phantom extra among declared ones is reported."""
        violations = self.check('pip install "predictive-maintenance-mcp[sse,nope]"')
        assert keys(violations) == [("extra", "nope")]

    def test_two_identical_lines_report_twice(self):
        """Each occurrence is its own finding, at its own line."""
        violations = self.check(
            "pip install predictive-maintenance-mcp[nope]",
            "pip install predictive-maintenance-mcp[nope]",
        )
        assert [v.line for v in violations] == [4, 5]

    def test_extras_on_the_local_checkout_are_checked(self):
        assert keys(self.check('pip install -e ".[nope]"')) == [("extra", "nope")]
        assert self.check('pip install -e ".[dev]"') == []

    def test_uv_pip_and_uvx_targets_are_checked(self):
        assert keys(
            self.check(
                "uv pip install --python .venv/bin/python predictive-maintenance-mcp[x1]",
                "uvx predictive-maintenance-mcp[x2]",
                "uvx --from predictive-maintenance-mcp[x3] predictive-maintenance-mcp",
                "python -m pip install predictive-maintenance-mcp[x4]",
            )
        ) == [("extra", "x1"), ("extra", "x2"), ("extra", "x3"), ("extra", "x4")]

    def test_a_misspelled_distribution_name_is_reported(self):
        violations = self.check("pip install predictive-maintenance-mpc")
        assert keys(violations) == [("dist-name", "predictive-maintenance-mpc")]

    def test_a_normalized_spelling_of_the_name_passes(self):
        """pip treats ``_`` and ``-`` alike (PEP 503), so this installs fine."""
        assert self.check("pip install predictive_maintenance_mcp==0.13.0") == []

    def test_another_package_is_not_the_project(self):
        """An unknown extra on another package is not this project's to judge."""
        assert self.check("pip install --upgrade mcp[nope]", "pip install x") == []

    def test_a_non_install_pip_command_is_ignored(self):
        assert self.check("pip uninstall predictive-maintenance-mcp[nope]") == []


class TestCliFlags:
    def check(self, *lines, lang="bash"):
        return cli_flag_violations(cmds(*lines, lang=lang), SCRIPT, PACKAGE, OPTIONS)

    def test_known_flags_and_choices_pass(self):
        assert (
            self.check(
                "predictive-maintenance-mcp --transport sse --host 0.0.0.0 --port 8080",
                "predictive-maintenance-mcp -t streamable-http -p 8000",
                "predictive-maintenance-mcp --transport=stdio",
                "predictive-maintenance-mcp",
            )
            == []
        )

    def test_an_unknown_transport_is_reported(self):
        assert keys(self.check("predictive-maintenance-mcp --transport websocket")) == [
            ("cli-choice", "--transport=websocket")
        ]

    def test_an_unknown_transport_after_the_short_flag_is_reported(self):
        assert keys(self.check("predictive-maintenance-mcp -t websocket")) == [
            ("cli-choice", "-t=websocket")
        ]

    def test_an_unknown_transport_in_equals_form_is_reported(self):
        assert keys(self.check("predictive-maintenance-mcp --transport=ws")) == [
            ("cli-choice", "--transport=ws")
        ]

    def test_an_unknown_flag_is_reported(self):
        assert keys(self.check("predictive-maintenance-mcp --data-dir x")) == [
            ("cli-flag", "--data-dir")
        ]

    def test_every_entry_point_spelling_is_checked(self):
        violations = self.check(
            "MCP_TRANSPORT=sse predictive-maintenance-mcp --a1",
            r"C:\venv\Scripts\predictive-maintenance-mcp.exe --a2",
            "python -m predictive_maintenance_mcp --a3",
            "uvx predictive-maintenance-mcp --a4",
            "npx @modelcontextprotocol/inspector uvx predictive-maintenance-mcp --a5",
        )
        assert keys(violations) == [("cli-flag", f"--a{i}") for i in range(1, 6)]

    def test_the_name_as_an_install_target_is_not_an_invocation(self):
        """A flag after the package name in a pip install belongs to pip."""
        assert self.check("pip install predictive-maintenance-mcp --upgrade") == []


class TestModules:
    RUNNABLE = {"predictive_maintenance_mcp", "benchmarks.cwru", "tools.audit"}

    def check(self, *lines, subcommands=frozenset({"smoke", "report"})):
        return module_violations(
            cmds(*lines), lambda m: m in self.RUNNABLE, lambda: subcommands
        )

    def test_runnable_project_modules_pass(self):
        assert (
            self.check(
                "python -m predictive_maintenance_mcp",
                "python -m benchmarks.cwru all",
                "python -m tools.audit smoke --out x",
                "/path/to/.venv/bin/python -m predictive_maintenance_mcp",
            )
            == []
        )

    def test_a_missing_project_module_is_reported(self):
        assert keys(self.check("python -m predictive_maintenance_mcp.nope")) == [
            ("module", "predictive_maintenance_mcp.nope")
        ]

    def test_other_modules_are_not_ours_to_resolve(self):
        assert self.check("python -m pip install x", "python -m venv .venv") == []

    def test_an_unknown_audit_subcommand_is_reported(self):
        assert keys(self.check("python -m tools.audit bogus")) == [
            ("audit-subcommand", "bogus")
        ]

    def test_audit_subcommands_are_only_loaded_when_needed(self):
        """The subcommand set is loaded only when a scanned doc contains a
        ``python -m tools.audit`` command."""

        def explode():
            raise AssertionError("loaded without a tools.audit command")

        violations = module_violations(
            cmds("python -m predictive_maintenance_mcp"),
            lambda m: True,
            explode,
        )
        assert violations == []


def test_a_package_without_main_is_not_runnable():
    """``python -m`` on a package needs ``__main__``: resolving is not enough."""
    assert module_is_runnable("predictive_maintenance_mcp")
    assert module_is_runnable("predictive_maintenance_mcp.server")
    assert not module_is_runnable("predictive_maintenance_mcp.mcp_tools")
    assert not module_is_runnable("predictive_maintenance_mcp.nope")
    assert not module_is_runnable("no_such_top_level_package_xyz.sub")


class TestScripts:
    EXISTING = {"validate_server.py", "setup_claude.ps1", "scripts/setup-worktree.py"}

    def check(self, *lines, lang="bash"):
        return script_violations(cmds(*lines, lang=lang), lambda p: p in self.EXISTING)

    def test_existing_scripts_pass(self):
        assert (
            self.check(
                "python validate_server.py",
                r"python scripts\setup-worktree.py",
                "powershell -ExecutionPolicy Bypass -File setup_claude.ps1",
            )
            == []
        )

    def test_missing_scripts_are_reported(self):
        assert keys(
            self.check(
                "python nope.py", "powershell -ExecutionPolicy Bypass -File gone.ps1"
            )
        ) == [("script", "gone.ps1"), ("script", "nope.py")]

    def test_an_absolute_path_cannot_be_judged_from_the_checkout(self):
        assert self.check("python /opt/elsewhere/run.py") == []

    def test_python_dash_c_is_not_a_script(self):
        assert self.check("python -c \"import mcp; print('x.py')\"") == []


class TestEnvironmentVariables:
    SOURCES = {
        "config.py": (
            "import os\n"
            "def project():\n"
            "    return os.environ.get('PDM_PROJECT_DIR')\n"
            "SIZE = os.getenv('PMM_MAX_SIGNAL_SIZE', '1')\n"
            "LEDGER = os.environ['PMM_LEDGER_DIR']\n"
        ),
        "server.py": (
            "import os\n"
            "def _env(name, fallback):\n"
            "    return (os.environ.get(name) or '').strip() or fallback\n"
            "def _env2(key):\n"
            "    return _env(key, '')\n"
            "T = _env('MCP_TRANSPORT', 'stdio')\n"
            "H = _env(name='MCP_HOST', fallback='x')\n"
            "P = _env2('MCP_PORT')\n"
            "for var in ('OneDrive',):\n"
            "    _env(var, '')\n"
        ),
    }

    def reads(self):
        return env_reads(self.SOURCES)

    def test_direct_reads_are_found(self):
        assert {"PDM_PROJECT_DIR", "PMM_MAX_SIGNAL_SIZE", "PMM_LEDGER_DIR"} <= (
            self.reads()
        )

    def test_reads_through_a_forwarding_wrapper_are_found(self):
        """``server._env`` is how MCP_TRANSPORT, MCP_HOST and MCP_PORT are read."""
        assert {"MCP_TRANSPORT", "MCP_HOST", "MCP_PORT"} <= self.reads()

    def test_wrappers_are_derived_from_the_ast(self):
        assert set(env_wrappers(self.SOURCES)) == {"_env", "_env2"}

    def test_a_variable_name_is_not_a_literal_read(self):
        assert "var" not in self.reads() and "OneDrive" not in self.reads()

    def test_no_sources_means_nothing_is_read(self):
        """Empty input fails toward red: every setting would be unread."""
        assert env_reads({}) == set()

    def test_every_doc_spelling_of_a_setting_is_found(self):
        commands = cmds(
            "export PMM_A=1",
            "set PMM_B=1",
            "PMM_C=1 predictive-maintenance-mcp",
            "docker run -e PMM_D=1 image",
            "az container create --environment-variables PMM_E=1 PMM_F=2",
            "env PMM_G=1 predictive-maintenance-mcp",
        )
        commands += cmds('$env:PMM_H = "1"', '$env:PMM_I="1"', lang="powershell")
        names = [s.name for s in env_settings_from_commands(commands)]
        assert names == [f"PMM_{c}" for c in "ABCDEFGHI"]

    def test_reading_a_variable_is_not_setting_it(self):
        commands = cmds(
            r"uv venv $env:LOCALAPPDATA\venvs\x --python 3.11",
            "set -e",
            "echo $PMM_X",
            lang="powershell",
        )
        assert env_settings_from_commands(commands) == []

    def test_an_unread_variable_set_in_a_doc_is_reported(self):
        settings = env_settings_from_commands(
            cmds(
                "export PMM_UNKNOWN_VAR=1",
                "export PDM_PROJECT_DIR=/srv/pmm",
                "MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8080 predictive-maintenance-mcp",
            )
        )
        violations = unread_env_violations(settings, self.reads())
        assert keys(violations) == [("env-unread", "PMM_UNKNOWN_VAR")]

    def test_smithery_env_keys_are_found(self):
        text = (
            "startCommand:\n"
            "  type: stdio\n"
            "  commandFunction: |-\n"
            "    (config) => ({\n"
            '      command: "predictive-maintenance-mcp",\n'
            "      args: [],\n"
            "      env: config.dataDir ? { DATA_DIR: config.dataDir, 'PMM_Q': x ? a : b }"
            " : { PDM_PROJECT_DIR: config.root }\n"
            "    })\n"
        )
        names = [s.name for s in env_settings_from_smithery(text)]
        assert names == ["DATA_DIR", "PMM_Q", "PDM_PROJECT_DIR"]

    def test_smithery_without_env_sets_nothing(self):
        text = "startCommand:\n  commandFunction: |-\n    () => ({ command: 'x' })\n"
        assert env_settings_from_smithery(text) == []

    def test_dotenv_keys_are_found(self):
        text = "# comment\nDATA_DIR=./data  # trailing\n\nexport LOG_LEVEL=INFO\n"
        assert [s.name for s in env_settings_from_dotenv(text)] == [
            "DATA_DIR",
            "LOG_LEVEL",
        ]

    def test_compose_environment_in_both_yaml_shapes_is_found(self):
        text = (
            "services:\n"
            "  a:\n"
            "    environment:\n"
            "      - MCP_TRANSPORT=sse\n"
            "      - PMM_BARE\n"
            "  b:\n"
            "    environment:\n"
            "      PMM_MAPPED: '1'\n"
            "  c:\n"
            "    image: x\n"
        )
        assert [s.name for s in env_settings_from_compose(text)] == [
            "MCP_TRANSPORT",
            "PMM_BARE",
            "PMM_MAPPED",
        ]
