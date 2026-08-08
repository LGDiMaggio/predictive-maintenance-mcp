"""
Predictive Maintenance MCP Server — Orchestrator.

Thin entry point that creates the MCPServer instance and delegates tool
registration to the mcp_tools sub-package (one module per ISO 13374 block).
"""

import argparse
import logging
import os
import sys

from mcp.server.mcpserver import MCPServer

from .config import DATA_DIR, MODELS_DIR, RESOURCES_DIR, REPORTS_DIR, CACHE_DIR
from .mcp_tools import register_all

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server initialization
# ---------------------------------------------------------------------------
mcp = MCPServer(
    "Predictive Maintenance",
    instructions="""
    MCP server for predictive maintenance and industrial machinery diagnostics.

    Capabilities:
    - Reading and managing vibration signals
    - Spectral analysis (FFT with dB normalization)
    - Envelope analysis for bearing fault detection
    - Statistical analysis (RMS, Kurtosis, Crest Factor)
    - ISO 20816-3 vibration severity evaluation
    - Professional HTML report generation (saved to reports/ directory)
    - Automatic peak detection and harmonic identification
    - Guided diagnostic workflows (prompts)
    - Document search (RAG) across machine manuals and bearing catalogs

    Output efficiency:
    - All spectral tools return COMPACT summaries (top peaks + stats), NOT full arrays
    - predict_anomalies returns counts/percentiles/worst segments, NOT per-segment arrays
    - Reports are saved as HTML files; only path + summary returned to chat
    - Use generate_*_report() or plot_signal() for full visual inspection
    - NEVER attempt to display or return full-length arrays in chat

    Report Generation System:
    - All visualizations are generated as professional HTML files
    - Reports are saved in reports/ directory with timestamped filenames
      (consecutive runs never overwrite) and embedded metadata
    - LLM should inform user about report location and NOT display HTML content
    - Use list_html_reports() to see available reports, and
      list_html_reports(file_name=...) to read one report's metadata
      without consuming tokens

    Documentation Search (RAG):
    - Use search_documentation() to find relevant passages in manuals and catalogs
    - Prefer search_documentation() over read_manual_excerpt() for targeted queries
    - Use read_manual_excerpt() only when you need to read consecutive pages

    Evidence-based inference policy (hard rules):
    1) Do NOT infer fault type from filenames, paths, or user-provided labels. Treat filenames as opaque identifiers.
    2) Do NOT make diagnostic claims based solely on statistical parameters (RMS/CF/Kurtosis). Use them for screening only.
    3) Bearing fault identification (inner/outer/ball/cage) must be supported by frequency-domain evidence (envelope peaks at characteristic frequencies) and at least one additional indicator (e.g., high kurtosis or distinct harmonics). If this corroboration is missing, mark the result as "inconclusive" and recommend further analysis.
    4) Use cautious language: say "possible" or "consistent with" when evidence is partial; say "confirmed" only if multiple independent analyses agree.
    5) Always cite which analyses and thresholds support each conclusion. If data or parameters are missing, ask for them instead of guessing.
    6) NEVER suggest parameters, thresholds, or recommendations not explicitly provided in tool outputs or prompt workflows. Do NOT invent frequency ranges, filter settings, or maintenance actions. Only use guidance from STEP 6 of diagnostic prompts.

    Signal unit policy (CRITICAL - declared units only, never guessed):
    - ISO 20816-3 severity verdicts on stored signals require a DECLARED unit:
      1. Explicit parameter: load_signal(signal_unit='g'|'m/s2'|'mm/s'|'m/s')
      2. Companion metadata: 'signal_unit' field in <name>_metadata.json
      (explicit parameter takes precedence over metadata; the direct
      rms_velocity_mm_s route of assess_severity needs no declaration —
      the value is by definition mm/s)
    - Units are NEVER inferred from signal amplitude — there is no
      amplitude-based guessing flow and no default assumption
    - Without a declared unit: severity tools raise a structured error, and
      diagnose_vibration returns an iso_severity block with status='refused'
      plus reason and remedy (the other diagnosis blocks still run)
    - If the unit is unknown, ask the user for it — do not guess
    - Wrong unit declaration (g vs mm/s) completely invalidates ISO 20816-3 results!

    Output formatting rules:
    - Keep responses brief (<=300 words, bullet points)
    - Inform user about generated HTML reports with file path
    - DO NOT display HTML content in chat (wastes tokens)
    - NEVER print large data directly
    - Reports are professional, self-contained HTML files

    Signal handle policy (signal_id is THE handle):
    - list_signals(scope="disk") shows loadable files; load_signal() loads
      one (or a batch) and returns the signal_id
    - list_signals(scope="memory") shows the loaded signal_ids;
      get_signal_info() exposes metadata including the companion
      source_metadata (rpm, reference frequencies, ...)
    - Every analysis/diagnosis/report/prognostics tool takes signal_id
    - Do NOT auto-correct or guess file names or ids; if ambiguous, ask

    Prognostics (ISO 13374 Block 5):
    - Remaining Useful Life estimation (linear, exponential, kalman) —
      requires a series of measurements taken over time (values or
      signal_ids + timestamps); a single recording is refused
    - Within-recording trend + degradation-onset screening
      (analyze_signal_trend: p-value based direction, post-baseline onset)

    Severity & Decision Support (ISO 13374 Blocks 4/6):
    - assess_severity: unified ISO 20816-3 severity + alert classification
      (zones A-D; boundary values from ISO 10816-3:2009). Accepts a stored
      signal_id OR a direct rms_velocity_mm_s reading, plus optional
      custom thresholds {'warning','alarm','danger'}
    - Maintenance recommendation generation (severity + fault-specific)

    Workflow Prompts (use these for guided analysis):
    - diagnose_bearing() - Complete bearing diagnostic workflow with evidence-based decision tree
    - diagnose_gear() - Gear fault detection workflow
    - quick_diagnostic_report() - Fast screening analysis (non-definitive)
    """
)

# ---------------------------------------------------------------------------
# Register all tools, resources, and prompts from ISO 13374 modules
# ---------------------------------------------------------------------------
register_all(mcp)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def _setup_environment() -> None:
    """Configure logging and create required directories."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    (RESOURCES_DIR / "machine_manuals").mkdir(parents=True, exist_ok=True)
    (RESOURCES_DIR / "bearing_catalogs").mkdir(parents=True, exist_ok=True)
    (RESOURCES_DIR / "datasheets").mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run the MCP server.

    CLI usage::

        # Default: stdio transport (Claude Desktop, VS Code)
        predictive-maintenance-mcp

        # SSE transport for remote/enterprise clients
        predictive-maintenance-mcp --transport sse --host 0.0.0.0 --port 8080

        # Streamable-HTTP transport (MCP 2025-03-26 spec)
        predictive-maintenance-mcp --transport streamable-http --port 8080

    Environment variable overrides (useful in Docker)::

        MCP_TRANSPORT=sse  MCP_HOST=0.0.0.0  MCP_PORT=8080
    """
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance MCP Server",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="Transport protocol (default: stdio, env: MCP_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_HOST", "127.0.0.1"),
        help="Bind address for SSE/HTTP (default: 127.0.0.1, env: MCP_HOST)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("MCP_PORT", "8000")),
        help="Port for SSE/HTTP transport (default: 8000, env: MCP_PORT)",
    )
    args = parser.parse_args()

    _setup_environment()

    logger.info("Starting Predictive Maintenance MCP Server...")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Data directory: {DATA_DIR}")
    if args.transport != "stdio":
        logger.info(f"Listening on {args.host}:{args.port}")

    # mcp 2.x dropped Settings.host/port: the bind address is a per-transport
    # run() kwarg now. stdio accepts neither — passing them would raise.
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
