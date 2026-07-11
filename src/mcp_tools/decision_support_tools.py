"""MCP tools for decision support (ISO 13374 Block 6).

Exposes vibration alert classification and maintenance recommendation
generation as MCP tools.
"""

import logging

from mcp.server.fastmcp import FastMCP, Context

from ..decision_support import (
    check_alert_thresholds,
    check_custom_alert,
    define_custom_thresholds,
    generate_recommendations,
)
from ..models import AlertResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(mcp: FastMCP) -> None:
    """Register decision-support MCP tools on *mcp*."""

    @mcp.tool()
    async def check_vibration_alert(
        ctx: Context,
        rms_velocity: float,
        machine_group: int = 2,
        support_type: str = "rigid",
    ) -> AlertResult:
        """Check an RMS velocity value against ISO 20816-3 alert thresholds.

        Classifies the reading into ISO zones (A/B/C/D) and returns the
        corresponding alert level (none, warning, alarm, danger). Zone
        boundary values come from ISO 10816-3:2009 (the four-zone scheme;
        ISO 20816-3:2022 merges zones A/B), via the single severity engine.

        Args:
            ctx: MCP context for user communication.
            rms_velocity: RMS velocity in mm/s.
            machine_group: ISO 20816-3 machine group — 1 (large, >300 kW)
                or 2 (medium, 15-300 kW). Default: 2.
            support_type: Support type — "rigid" or "flexible" (default: "rigid").

        Returns:
            AlertResult with zone classification and alert level.
        """
        await ctx.info(
            f"Checking alert thresholds for {rms_velocity:.2f} mm/s "
            f"(group {machine_group}, {support_type}) ..."
        )

        result = check_alert_thresholds(rms_velocity, machine_group, support_type)

        return AlertResult(
            alert_level=result["alert_level"],
            zone=result["zone"],
            rms_velocity=rms_velocity,
            exceeded_threshold=result.get("exceeded_threshold"),
            message=result["message"],
        )

    @mcp.tool()
    async def check_custom_vibration_alert(
        ctx: Context,
        rms_velocity: float,
        warning_threshold: float,
        alarm_threshold: float,
        danger_threshold: float,
    ) -> AlertResult:
        """Check an RMS velocity value against user-defined thresholds.

        Allows custom alert boundaries instead of the ISO 20816-3 defaults.

        Args:
            ctx: MCP context for user communication.
            rms_velocity: RMS velocity in mm/s.
            warning_threshold: Upper boundary of normal zone (mm/s).
            alarm_threshold: Upper boundary of warning zone (mm/s).
            danger_threshold: Upper boundary of alarm zone (mm/s).

        Returns:
            AlertResult with zone classification and alert level.
        """
        await ctx.info(
            f"Checking custom thresholds for {rms_velocity:.2f} mm/s "
            f"(warn={warning_threshold}, alarm={alarm_threshold}, danger={danger_threshold}) ..."
        )

        thresholds = define_custom_thresholds(
            warning_threshold, alarm_threshold, danger_threshold,
        )
        result = check_custom_alert(rms_velocity, thresholds)

        return AlertResult(
            alert_level=result["alert_level"],
            zone=result["zone"],
            rms_velocity=rms_velocity,
            exceeded_threshold=result.get("exceeded_threshold"),
            message=result["message"],
        )

    @mcp.tool()
    async def generate_maintenance_recommendations(
        ctx: Context,
        severity_zone: str,
        fault_types: str = "",
    ) -> str:
        """Generate maintenance recommendations based on severity and detected faults.

        Combines ISO zone-based urgency with fault-specific maintenance
        actions. This tool intentionally does NOT accept a confidence
        value: any number supplied by the caller would be echoed into
        advisory output without evidential basis.

        Args:
            ctx: MCP context for user communication.
            severity_zone: ISO zone letter — "A", "B", "C", or "D".
            fault_types: Comma-separated fault type keywords, e.g.
                "outer_race,misalignment". Leave empty for zone-only advice.

        Returns:
            Formatted string listing all maintenance recommendations.
        """
        fault_list = [
            ft.strip() for ft in fault_types.split(",") if ft.strip()
        ] if fault_types else None

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
