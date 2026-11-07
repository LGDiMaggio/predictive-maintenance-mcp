"""
State shared by one import validation.

The run that accumulates issues and statements, and the basis, citation
and decimal helpers every check uses.

ISO 13374 Block 1 (data acquisition). Internal to
:mod:`predictive_maintenance_mcp.signal_import.validation`; nothing here
writes anywhere.
"""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional

from ...file_text import PREAMBLE_LINE_CHARS, FileText
from .. import statements as st
from ..attestation import BASIS_ORIGIN
from ..contracts import IssueEffect, Statement, ValidationIssue

_ASK = st.ASK


@dataclass
class _Run:
    """Everything a validation accumulates before its result."""

    caller: bool
    interpretation: dict[str, Any]
    attestation: dict[str, Any]
    issues: list[ValidationIssue] = field(default_factory=list)
    #: Voltage channels whose DC policy the user has yet to choose: validated
    #: with the policy pending, so the choice can see the channel's mean.
    pending_dc: set[int] = field(default_factory=set)
    stats: dict[int, tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )
    statements: list[Statement] = field(default_factory=list)

    def issue(
        self,
        code: str,
        effect: IssueEffect,
        message: str,
        *,
        field_path: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        token: Optional[FileText] = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                effect=effect,
                message=message,
                field=field_path,
                line=line,
                column=column,
                token=token,
            )
        )

    def basis(self, path: str) -> Optional[dict[str, Any]]:
        bases: dict[str, dict[str, Any]] = self.attestation["bases"]
        return bases.get(path)

    def override(self, path: str) -> Optional[str]:
        overrides: dict[str, str] = self.attestation["overrides"]
        return overrides.get(path)

    def missing_basis(self, path: str, what: str) -> None:
        self.issue(
            "missing_basis",
            "needs_confirmation",
            f"{what} has no basis: {_ASK} to state it, or to accept the value "
            f"from the evidence it comes from, and name that basis for "
            f"{path!r}.",
            field_path=path,
        )


def _citation(entry: Optional[Mapping[str, Any]]) -> str:
    if entry is None:
        return ""
    if entry.get("line") is not None:
        return f"line {entry['line']}"
    if entry.get("column") is not None:
        return f"time column at {st.channel_words(entry['column'])}"
    if entry.get("import_id") is not None:
        return f"import {entry['import_id'][:12]}"
    if entry.get("declaration_version") is not None:
        return f"point declaration version {entry['declaration_version']}"
    return ""


def _origin(entry: Optional[Mapping[str, Any]]) -> Optional[str]:
    return None if entry is None else BASIS_ORIGIN[entry["basis"]]


def _printed(raw: str) -> FileText:
    """File text an issue names, escaped and capped like a preamble line."""
    return FileText.of(raw.strip(), cap=PREAMBLE_LINE_CHARS)


def _decimal(text: Optional[str]) -> Optional[Decimal]:
    """A decimal string as a finite Decimal; None when it is absent or not a
    finite number (a hand-written ledger may hold ``NaN``), so it never
    equals anything."""
    if text is None:
        return None
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None
