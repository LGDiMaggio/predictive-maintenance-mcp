"""
Predictive Maintenance MCP Server — Orchestrator.

Thin entry point that creates the FastMCP instance and delegates tool
registration to the mcp_tools sub-package (one module per ISO 13374 block).
"""

import argparse
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from .config import DATA_DIR, MODELS_DIR, RESOURCES_DIR, REPORTS_DIR, CACHE_DIR
from .mcp_tools import register_all

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server initialization
# ---------------------------------------------------------------------------
mcp = FastMCP(
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
    - Signal resources return metadata + statistics only, NOT raw samples
    - Reports are saved as HTML files; only path + summary returned to chat
    - Use generate_*_report() or plot_*() for full visual inspection
    - NEVER attempt to display or return full-length arrays in chat

    Report Generation System:
    - All visualizations are generated as professional HTML files
    - Reports are saved in reports/ directory with metadata
    - LLM should inform user about report location and NOT display HTML content
    - Use list_html_reports() to see available reports
    - Use get_report_info() to get report info without consuming tokens

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

    Signal unit confirmation policy (CRITICAL - Hypothesis-based flow):
    - When calling evaluate_iso_20816(), tool follows hypothesis-based unit detection:
      1. Check metadata file for 'signal_unit' field (best practice)
      2. Use user-provided signal_unit parameter if explicitly given
      3. If neither exists: Tool generates RMS-based hypothesis and ASKS USER to confirm
      4. Default assumption: 'g' (acceleration) if user doesn't know
    - Tool will display hypothesis reasoning and request user confirmation
    - ALWAYS inform user about the hypothesis and ask for confirmation
    - If user confirms hypothesis is wrong, they must provide correct signal_unit parameter
    - Wrong unit conversion (g vs mm/s) completely invalidates ISO 20816-3 results!
    - Best practice: Recommend user adds 'signal_unit' field to metadata JSON files

    Output formatting rules:
    - Keep responses brief (<=300 words, bullet points)
    - Inform user about generated HTML reports with file path
    - DO NOT display HTML content in chat (wastes tokens)
    - NEVER print large data directly
    - Reports are professional, self-contained HTML files

    Filename resolution policy:
    - FIRST call list_available_signals() to verify exact filename
    - Do NOT auto-correct or guess filenames
    - If ambiguous, ask user to clarify

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

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    logger.info("Starting Predictive Maintenance MCP Server...")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Data directory: {DATA_DIR}")
    if args.transport != "stdio":
        logger.info(f"Listening on {args.host}:{args.port}")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
