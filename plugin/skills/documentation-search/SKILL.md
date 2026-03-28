---
description: >
  Search machine manuals, bearing catalogs, and technical documentation using
  RAG-based retrieval via the predictive-maintenance-mcp server. Use this skill
  when the user says "search documentation", "find in manual", "bearing catalog",
  "look up bearing", "machine manual", "extract specs", "find bearing specs",
  "SKF bearing", "technical documentation", "datasheet", "manual search",
  "what bearing", or needs to find technical information from stored documents.
---

# Documentation Search

Search and extract information from machine manuals, bearing catalogs, and
technical datasheets using RAG-based retrieval (FAISS + TF-IDF) and structured
document parsing.

**Prerequisite**: The `predictive-maintenance-mcp` MCP server must be connected.

## Core Operations

### List Available Documentation

Call `list_machine_manuals()` to see all available manuals and catalogs in the
resources/ directory (PDF, TXT formats).

### Search Across All Documents (RAG)

Call `search_documentation(query=..., top_k=5)` to perform semantic search
across all indexed documents.

- **query**: natural language search query (e.g., "bearing replacement procedure
  for pump XYZ", "maximum vibration limits")
- **top_k**: number of results to return (default 5)

The RAG engine uses FAISS vector similarity + TF-IDF keyword matching to find
the most relevant document passages.

### Read Manual Excerpts

Call `read_manual_excerpt(manual_name=..., start_page=..., end_page=...)` to
read specific pages from a manual.

Useful when you know which manual contains the information but need to read a
specific section.

### Extract Machine Specifications

Call `extract_manual_specs(manual_name=...)` to automatically extract structured
data from a manual:
- Bearing designations and types
- Power ratings
- RPM ranges
- Vibration limits
- Maintenance intervals

### Look Up Bearing in Catalog

Call `search_bearing_catalog(query=...)` to search the bearing database.

- Search by designation: "6205", "SKF 6205", "22210"
- Search by type: "deep groove ball bearing"
- Returns: geometry (n_balls, d_ball, d_pitch, contact_angle), dimensions,
  load ratings

### Combined Lookup + Frequency Calculation

Call `lookup_bearing_and_compute_tool(bearing_query=..., shaft_rpm=...)` to
search the catalog AND compute characteristic fault frequencies (BPFO, BPFI,
BSF, FTF) in one step.

## Typical Workflows

### "What bearing is installed in this machine?"

1. Call `list_machine_manuals()` to find the relevant manual
2. Call `extract_manual_specs(manual_name=...)` to pull bearing designations
3. Call `search_bearing_catalog(query=...)` with the found designation

### "Find the vibration limits for this equipment"

1. Call `search_documentation(query="vibration limits {machine_name}")`
2. If insufficient, call `read_manual_excerpt(...)` on the specific manual

### "Look up bearing 6205 and compute fault frequencies"

1. Call `lookup_bearing_and_compute_tool(bearing_query="6205", shaft_rpm=1800)`
2. Returns bearing geometry + BPFO, BPFI, BSF, FTF frequencies

## Important Notes

- RAG index is built on first search and cached for subsequent queries
- PDF extraction quality depends on document formatting
- Bearing catalog covers common SKF designations; less common bearings may need
  manual geometry input
- All documents are processed locally — no data leaves the machine
