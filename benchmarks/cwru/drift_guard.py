"""Publication drift guard: published numbers bound slot-by-slot to results.json.

Every quantitative claim the README or the methodology document publishes
about the CWRU benchmark must trace to ``benchmarks/cwru/results/results.json``
(origin acceptance example AE4). This module makes that binding executable:
each published value sits in a *slot* naming the exact key path it came
from, and :func:`check_document` compares every slot against the artifact,
failing on any single mismatch. Set-membership checking was rejected in the
plan: with small integers, a stale value colliding with another stratum's
count would stay green.

Slot-marker convention (U7 writes the real sections using EXACTLY this;
everything is invisible in rendered markdown because it is HTML comments):

- The benchmark section of a document is delimited by two exact marker
  lines: :data:`SECTION_START` and :data:`SECTION_END`. At most one
  section per document; a start without an end (or vice versa) is refused.
- Inside the section, every published value is wrapped in a slot::

      <!-- slot: DOTTED.KEY.PATH [FORMAT] -->VALUE<!-- /slot -->

  on a single line, where:

  - ``DOTTED.KEY.PATH`` is a dot-joined path into the parsed
    ``results.json`` (segments match ``[A-Za-z0-9_]+``; an all-digits
    segment indexes a JSON array), e.g.
    ``headline.frequency_detection.rate`` or ``headline.strata_included.0``.
  - ``FORMAT`` is optional and space-separated from the path: ``raw``
    (the default) or ``pct0`` .. ``pct9``.
  - ``VALUE`` is the visible published text, compared VERBATIM (no
    whitespace trimming) against the canonical rendering of the value at
    the key path.

- Canonical ``raw`` rendering of a JSON scalar: ``null`` -> ``null``,
  booleans -> ``true``/``false``, integers -> ``str(int)``, floats ->
  Python ``repr`` (shortest round-trip -- identical to how the artifact
  serializer wrote them, after its 9-decimal rounding), strings verbatim.
  Arrays and objects are refused: a slot binds one scalar.
- ``pctN`` rendering: the value must be a number (a rate in ``[0, 1]``);
  the canonical text is ``value * 100`` formatted with N decimal places
  (``format(value * 100, ".Nf")``). The ``%`` sign itself stays OUTSIDE
  the slot. Example: rate ``0.983333333`` with ``pct1`` -> ``98.3``.

Worked example (README line and what the guard checks)::

    Frequency detection: <!-- slot: headline.frequency_detection.hits -->59<!-- /slot -->/<!-- slot: headline.frequency_detection.total -->60<!-- /slot --> (<!-- slot: headline.frequency_detection.rate pct1 -->98.3<!-- /slot -->%)

State machine (the U6 test matrix):

- no section in the document -> green, 0 slots (pre-publication: an
  artifact without a section is fine);
- section present but ``results.json`` absent -> refused (a published
  section must have the artifact behind it);
- section present with zero slots -> refused (numbers written as plain
  prose would bypass the guard entirely);
- any slot naming a missing key path, binding a non-scalar, or showing a
  value different from the canonical rendering -> refused, naming the
  slot, the expected text, and the found text.

All refusals are ``ValueError`` in the repo's "problem -- remedy" style.
This module is deliberately stdlib-only so the guard can run without the
scientific stack.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_DOCUMENT_PATHS",
    "DEFAULT_RESULTS_PATH",
    "REPO_ROOT",
    "SECTION_END",
    "SECTION_START",
    "Slot",
    "check_document",
    "check_publication",
    "extract_section",
    "extract_slots",
    "load_results",
    "render_value",
    "resolve_key_path",
]

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Repo root derived from this package's location (benchmarks/cwru/ is two
#: levels below it). A test cross-checks it against ``runner.REPO_ROOT`` so
#: the two derivations cannot drift apart silently.
REPO_ROOT: Path = _PACKAGE_DIR.parent.parent

#: The committed results artifact -- the single source of truth every slot
#: is compared against. Kept as a local constant so this module stays
#: stdlib-only; a test cross-checks it against ``scorer.DEFAULT_RESULTS_PATH``.
DEFAULT_RESULTS_PATH: Path = _PACKAGE_DIR / "results" / "results.json"

#: Documents the publication guard sweeps: the README section and the
#: methodology document. A document that does not exist, or exists without
#: a benchmark section, is simply unpublished (green).
DEFAULT_DOCUMENT_PATHS: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "benchmark-methodology.md",
)

#: Exact section-start marker. U7 must emit it verbatim.
SECTION_START: str = "<!-- cwru-benchmark:start -->"

#: Exact section-end marker. U7 must emit it verbatim.
SECTION_END: str = "<!-- cwru-benchmark:end -->"

#: One complete slot: path, optional format, verbatim single-line value.
_SLOT_RE = re.compile(
    r"<!--\s*slot:\s*(?P<path>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
    r"(?:\s+(?P<fmt>raw|pct[0-9]))?\s*-->"
    r"(?P<value>[^\n]*?)"
    r"<!--\s*/slot\s*-->"
)

#: Anything that even looks like a slot opening. If this count diverges
#: from the full-match count, a slot is malformed (missing close marker,
#: missing colon, value spanning lines) and the guard refuses rather than
#: silently skipping the broken binding.
_SLOT_OPEN_RE = re.compile(r"<!--\s*slot\b")


@dataclass(frozen=True)
class Slot:
    """One extracted slot binding.

    Attributes:
        key_path: Dotted path into the parsed ``results.json``.
        fmt: Rendering format -- ``"raw"`` or ``"pct0"`` .. ``"pct9"``.
        value_text: The published text, exactly as it appears.
    """

    key_path: str
    fmt: str
    value_text: str


def extract_section(text: str, document_name: str) -> Optional[str]:
    """Locate the marker-delimited benchmark section of a document.

    Args:
        text: The document's full text.
        document_name: Human-readable name for error messages.

    Returns:
        The text between the markers, or ``None`` when the document
        carries no benchmark section (pre-publication state).

    Raises:
        ValueError: If the markers are unbalanced, duplicated, or in the
            wrong order -- a malformed section must never pass as "no
            section".
    """
    starts = text.count(SECTION_START)
    ends = text.count(SECTION_END)
    if starts == 0 and ends == 0:
        return None
    if starts != 1 or ends != 1:
        raise ValueError(
            f"{document_name} carries {starts} '{SECTION_START}' and "
            f"{ends} '{SECTION_END}' marker(s) -- a benchmark section "
            f"needs exactly one of each. Fix the markers; a malformed "
            f"section must never pass as unpublished."
        )
    begin = text.index(SECTION_START) + len(SECTION_START)
    end = text.index(SECTION_END)
    if end < begin:
        raise ValueError(
            f"{document_name} has '{SECTION_END}' before "
            f"'{SECTION_START}' -- the benchmark section markers are in "
            f"the wrong order. Fix the markers."
        )
    return text[begin:end]


def extract_slots(section: str, document_name: str) -> tuple[Slot, ...]:
    """Extract every slot binding from a benchmark section.

    Args:
        section: The section text (between the markers).
        document_name: Human-readable name for error messages.

    Returns:
        The slots, in document order.

    Raises:
        ValueError: If anything slot-shaped fails to parse as a complete
            slot -- a broken binding silently skipped would unbind the
            value it was supposed to guard.
    """
    matches = list(_SLOT_RE.finditer(section))
    lookalikes = len(_SLOT_OPEN_RE.findall(section))
    if lookalikes != len(matches):
        raise ValueError(
            f"{document_name} benchmark section contains {lookalikes} "
            f"slot opening(s) but only {len(matches)} complete slot(s) -- "
            f"a slot is malformed (missing '<!-- /slot -->' on the same "
            f"line, missing colon, or a value spanning lines). Fix it to "
            f"the documented form: "
            f"<!-- slot: dotted.key.path [raw|pctN] -->VALUE<!-- /slot -->."
        )
    return tuple(
        Slot(
            key_path=match.group("path"),
            fmt=match.group("fmt") or "raw",
            value_text=match.group("value"),
        )
        for match in matches
    )


def resolve_key_path(results: object, key_path: str, document_name: str) -> object:
    """Walk a dotted key path through the parsed results artifact.

    Args:
        results: The parsed ``results.json`` document.
        key_path: Dot-joined path; an all-digits segment indexes an array.
        document_name: Human-readable name for error messages.

    Returns:
        The value at the path.

    Raises:
        ValueError: If any segment is absent, indexes out of range, or
            descends into a scalar -- named per segment so the failing
            slot is directly fixable.
    """
    current: object = results
    walked: list[str] = []
    for segment in key_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise ValueError(
                    f"{document_name} slot '{key_path}': key '{segment}' "
                    f"does not exist at '{'.'.join(walked) or '<root>'}' "
                    f"in results.json -- fix the slot's key path, or "
                    f"re-run the benchmark if the artifact schema moved."
                )
            current = current[segment]
        elif isinstance(current, list):
            if not segment.isdigit() or int(segment) >= len(current):
                raise ValueError(
                    f"{document_name} slot '{key_path}': segment "
                    f"'{segment}' is not a valid index into the "
                    f"{len(current)}-element array at "
                    f"'{'.'.join(walked)}' in results.json -- fix the "
                    f"slot's key path."
                )
            current = current[int(segment)]
        else:
            raise ValueError(
                f"{document_name} slot '{key_path}': "
                f"'{'.'.join(walked)}' is a scalar, so segment "
                f"'{segment}' cannot descend into it -- fix the slot's "
                f"key path."
            )
        walked.append(segment)
    return current


def render_value(value: object, fmt: str, key_path: str, document_name: str) -> str:
    """Render an artifact value to its canonical published text.

    The canonical forms are documented in the module docstring; they are
    the contract U7's sections are written against.

    Args:
        value: The value resolved from ``results.json``.
        fmt: ``"raw"`` or ``"pct0"`` .. ``"pct9"``.
        key_path: The slot's key path, for error messages.
        document_name: Human-readable name for error messages.

    Returns:
        The canonical text the document must show verbatim.

    Raises:
        ValueError: If the value is not a scalar, or ``pctN`` is applied
            to a non-numeric value.
    """
    if fmt.startswith("pct"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{document_name} slot '{key_path}': format '{fmt}' "
                f"needs a numeric rate but results.json holds "
                f"{type(value).__name__} -- use 'raw', or point the slot "
                f"at a rate."
            )
        return format(value * 100, f".{int(fmt[3])}f")
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value
    raise ValueError(
        f"{document_name} slot '{key_path}': the value at that path is "
        f"a {type(value).__name__}, not a scalar -- a slot binds one "
        f"published number or string. Point the slot at a scalar leaf."
    )


def load_results(results_path: Path) -> dict[str, object]:
    """Load and structurally validate the results artifact.

    Args:
        results_path: Path to ``results.json``.

    Returns:
        The parsed artifact.

    Raises:
        ValueError: If the file is missing, not valid JSON, or not a
            JSON object.
    """
    if not results_path.exists():
        raise ValueError(
            f"Results artifact not found at {results_path} -- a "
            f"published benchmark section must trace to the committed "
            f"artifact. Run the benchmark (python -m benchmarks.cwru "
            f"all) and commit results.json, or remove the section until "
            f"measured numbers exist."
        )
    try:
        parsed: object = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Results artifact at {results_path} is not valid JSON "
            f"({exc}) -- restore it from git, or re-run the benchmark "
            f"(python -m benchmarks.cwru all)."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Results artifact at {results_path} must be a JSON object, "
            f"got {type(parsed).__name__} -- restore it from git, or "
            f"re-run the benchmark (python -m benchmarks.cwru all)."
        )
    return parsed


def check_document(document_path: Path, results_path: Optional[Path] = None) -> int:
    """Check one document's benchmark section against the artifact.

    Args:
        document_path: The markdown document to check. A document that
            does not exist, or has no benchmark section, is unpublished
            and passes with 0 slots.
        results_path: Artifact override (tests only); ``None`` uses
            :data:`DEFAULT_RESULTS_PATH`.

    Returns:
        The number of slots verified (0 means nothing is published in
        this document).

    Raises:
        ValueError: If the section exists without the artifact, carries
            no slots, or any slot is malformed, unresolvable, non-scalar,
            or shows a value different from the canonical rendering. All
            drifted slots are named together, each with expected and
            found text.
    """
    if not document_path.exists():
        return 0
    document_name = document_path.name
    section = extract_section(document_path.read_text(encoding="utf-8"), document_name)
    if section is None:
        return 0

    artifact = results_path if results_path is not None else DEFAULT_RESULTS_PATH
    results = load_results(artifact)
    slots = extract_slots(section, document_name)
    if not slots:
        raise ValueError(
            f"{document_name} carries a benchmark section with no slots "
            f"-- numbers written as plain prose bypass the drift guard "
            f"entirely. Bind every published value with "
            f"<!-- slot: dotted.key.path [raw|pctN] -->VALUE<!-- /slot -->, "
            f"or remove the section."
        )

    problems: list[str] = []
    for slot in slots:
        try:
            value = resolve_key_path(results, slot.key_path, document_name)
            expected = render_value(value, slot.fmt, slot.key_path, document_name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if slot.value_text != expected:
            problems.append(
                f"{document_name} slot '{slot.key_path}' (format "
                f"'{slot.fmt}'): the document shows {slot.value_text!r} "
                f"but results.json at {artifact} holds {expected!r}."
            )
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(
            f"Publication drift: {len(problems)} slot problem(s) in "
            f"{document_name}:\n{details}\nEvery published value must "
            f"equal the committed artifact (the single source of truth) "
            f"-- regenerate the section from results.json, or re-run the "
            f"benchmark and commit the new artifact."
        )
    return len(slots)


def check_publication(
    document_paths: Optional[Sequence[Path]] = None,
    results_path: Optional[Path] = None,
) -> dict[str, int]:
    """Check every publication document against the artifact.

    Args:
        document_paths: Documents to sweep (tests only); ``None`` uses
            :data:`DEFAULT_DOCUMENT_PATHS`.
        results_path: Artifact override (tests only).

    Returns:
        Verified slot count per document path (0 = unpublished there).

    Raises:
        ValueError: Propagated from :func:`check_document` for the first
            document with a problem.
    """
    documents = (
        tuple(document_paths) if document_paths is not None else DEFAULT_DOCUMENT_PATHS
    )
    return {
        str(document): check_document(document, results_path) for document in documents
    }
