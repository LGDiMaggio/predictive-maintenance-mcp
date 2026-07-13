"""
Predictive Maintenance MCP Server

A Model Context Protocol server for industrial machinery diagnostics,
vibration analysis, and predictive maintenance.

Installable as 'predictive-maintenance-mcp' from PyPI.
Package name: predictive_maintenance_mcp (mapped from src/ directory).
"""

__version__ = "0.9.0"
__author__ = "Luigi Gianpio Di Maggio"
__license__ = "MIT"

from .server import mcp, main

__all__ = ["mcp", "main", "__version__"]
