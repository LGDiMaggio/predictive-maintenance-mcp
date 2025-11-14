"""
Predictive Maintenance MCP Server

A Model Context Protocol server for industrial machinery diagnostics,
vibration analysis, and predictive maintenance.
"""

__version__ = "0.2.0"
__author__ = "Luigi Gianpio Di Maggio"
__license__ = "MIT"

from .machinery_diagnostics_server import mcp, main

__all__ = ["mcp", "main"]
