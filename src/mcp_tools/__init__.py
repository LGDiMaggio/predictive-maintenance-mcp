"""MCP tool modules organized by ISO 13374 blocks."""

from . import acquisition_tools  # noqa: F401
from . import analysis_tools  # noqa: F401
from . import diagnostics_tools  # noqa: F401
from . import report_tools  # noqa: F401
from . import prompts  # noqa: F401


def register_all(mcp):
    """Register all MCP tools, resources, and prompts on the server."""
    acquisition_tools.register(mcp)
    analysis_tools.register(mcp)
    diagnostics_tools.register(mcp)
    report_tools.register(mcp)
    prompts.register(mcp)
