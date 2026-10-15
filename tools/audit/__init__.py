"""Quality and compatibility audit checks, run as ``python -m tools.audit``.

Gates live in pytest; this package holds the operations around them: checks
that need a subprocess, a built artifact or the network, and the steps that
turn their outcomes into one report. Every check writes the same
``pmm-audit-result/1`` JSON document (see :mod:`tools.audit.results`), so the
report builder reads one contract instead of parsing job logs.

The package uses only the standard library plus what is already present
where each check runs: the stdio smoke harness needs ``mcp`` and
``jsonschema``, both of which a wheel install of the server provides.
Subcommand modules are imported lazily by :mod:`tools.audit.__main__`, so a
check whose dependencies are absent cannot break the others.
"""
