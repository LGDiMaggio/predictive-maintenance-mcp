"""Backward compatibility shim — use predictive_maintenance_mcp.diagnostics.iso10816."""
from .diagnostics.iso10816 import *  # noqa: F401,F403
from .diagnostics.iso10816 import assess_severity_raw  # noqa: F401
