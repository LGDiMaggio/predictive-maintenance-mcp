"""MCP tools for decision support (ISO 13374 Block 6).

Exposes maintenance recommendation generation as an MCP tool. Alert
classification lives in the unified ``assess_severity`` diagnostics tool
(U9), which accepts a direct rms_velocity_mm_s reading and optional
custom thresholds.
"""

import logging
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer, Context

from ..decision_support import generate_recommendations

logger = logging.getLogger(__name__)

#: Closed fault vocabulary: canonical bearing faults (FAULT_TYPE_CANONICAL
#: values) + machine-level faults. Kept in sync with
#: decision_support.recommendations.VALID_FAULT_TYPES (asserted in tests).
FaultType = Literal[
    "ball",
    "cage",
    "inner_race",
    "looseness",
    "misalignment",
    "outer_race",
    "unbalance",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def generate_maintenance_recommendations(
    ctx: Context,
    severity_zone: Literal["A", "B", "C", "D"],
    fault_types: Optional[list[FaultType]] = None,
) -> str:
    """Generate maintenance recommendations based on severity and detected faults.

        Combines ISO zone-based urgency with fault-specific maintenance
        actions. This tool intentionally does NOT accept a confidence
        value: any number supplied by the caller would be echoed into
        advisory output without evidential basis.

        Args:
            ctx: MCP context for user communication.
            severity_zone: ISO zone letter — "A", "B", "C", or "D".
            fault_types: Detected fault types from the closed canonical
                vocabulary — outer_race/inner_race/ball/cage for bearings
                (NOT the BPFO/BPFI/BSF/FTF acronyms) plus misalignment/
                unbalance/looseness. None for zone-only advice.

        Returns:
            Formatted string listing all maintenance recommendations.

        Raises:
            ValueError: If any fault type is outside the canonical
                vocabulary (the message lists the allowed values —
                unknown values are never dropped silently).
        """
    fault_list = list(fault_types) if fault_types else None

    await ctx.info(
        f"Generating recommendations for zone {severity_zone}"
        + (f" with faults: {fault_list}" if fault_list else "")
        + " ..."
    )

    recs = generate_recommendations(severity_zone, fault_list)

    lines: list[str] = []
    for i, rec in enumerate(recs, 1):
        lines.append(
            f"{i}. [{rec['urgency'].upper()}] {rec['action']}\n"
                f"   {rec['description']}"
        )

    return "\n\n".join(lines)


def register(mcp: MCPServer) -> None:
    """Register decision-support MCP tools on *mcp*."""
    mcp.tool()(generate_maintenance_recommendations)
