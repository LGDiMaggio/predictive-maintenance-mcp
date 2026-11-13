---
name: documentation-search
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

Call `list_machine_manuals()` to see all manuals and catalogs available in
the resources/ directory (PDF, TXT formats), with their file names.

### Search Across All Documents (RAG)

Call `search_documentation(query="bearing replacement procedure", top_k=5)`
for semantic search across all indexed documents.

- **query**: natural-language search (e.g. "maximum vibration limits")
- **top_k**: number of passages to return (default 5)
- **force_reindex**: set True only if documents changed since the last search

### Read Manual Excerpts

Call `read_manual_excerpt(file_name="test_pump_manual.pdf", max_pages=10)` to
read the first pages of a manual (up to `max_pages`). Use the `file_name`
exactly as returned by list_machine_manuals.

### Extract Machine Specifications

Call `extract_manual_specs(file_name="test_pump_manual.pdf")` to pull
structured data from a manual:
- Bearing designations and types
- Power ratings and RPM ranges
- Vibration limits
- Maintenance intervals

### Look Up a Bearing in the Catalog

Call `search_bearing_catalog(bearing_id="<bearing designation>")`.

- Accepts designations like "6205" or "SKF 6205-2RS" (prefixes resolved)
- Returns verified geometry (number of balls, ball diameter, pitch diameter,
  contact angle) WITH its source citation, plus fault-frequency multipliers
  (BPFO/BPFI/BSF/FTF per unit of shaft speed)
- The catalog contains only entries with physically valid, source-traced
  geometry. A miss returns a structured "not found" result with a
  suggestion — never invented geometry. In that case ask the user for the
  geometry and use `calculate_bearing_characteristic_frequencies(num_balls=<n>, ball_diameter_mm=<mm>, pitch_diameter_mm=<mm>, contact_angle_deg=<deg>, rpm=<rpm>)`.

Values in angle brackets are placeholders, not your machine's values: take
the designation and geometry from the user or the documentation, and the
speed from what `get_signal_info` reports for the signal or from the user
(`rpm` in revolutions per minute); ask when none gives them.

## Typical Workflows

### "What bearing is installed in this machine?"

1. `list_machine_manuals()` to find the relevant manual
2. `extract_manual_specs(file_name="<manual>.pdf")` to pull bearing designations
3. `search_bearing_catalog(bearing_id="<designation>")` for verified geometry

### "Find the vibration limits for this equipment"

1. `search_documentation(query="vibration limits <machine name>")`
2. If insufficient, `read_manual_excerpt(file_name="<manual>.pdf", max_pages=10)`

### "Look up this bearing and check the signal against it"

1. `search_bearing_catalog(bearing_id="<bearing designation>")` — geometry + frequency multipliers
2. `check_bearing_faults(signal_id="<id>", rpm=<rpm>, bearing_id="<bearing designation>")` —
   computes BPFO/BPFI/BSF/FTF at the given speed and tests the line at each
   against the local noise floor of the signal's envelope spectrum in one
   call. Its `claim`, never your own matching of envelope peaks, decides what
   may be said about a fault type: report it as the bearing-diagnosis skill
   describes.

If a `check_bearing_faults` result, or a non-null `bearing_faults` block of a
`diagnose_vibration` result, has no `claim` field, the server predates the
bearing claim: infer or relay no bearing fault type or bearing assessment
from it (its `overall_assessment`, `most_likely_fault`, `fault_checks`,
bearing sentence or Bearing recommendations), and tell the user that this
bearing guidance needs a server whose `check_bearing_faults` returns
`claim`. The `evidence_strength` of such a `diagnose_vibration` result, and
its "Fault evidence strength" sentence, also count the bearing finding, so
neither is relayed; results that do not rest on the bearing check, such as
the ISO 20816-3 zone and its action, are relayed as usual.

## Important Notes

- The RAG index is built on first search and cached for later queries
- PDF extraction quality depends on document formatting
- The bearing catalog is deliberately small and verified — every entry has a
  `source` field. Uncommon bearings need user-provided geometry; the tools
  will say so rather than guess
- Documentation findings inform the engineer's decision — verify critical
  specs against the physical nameplate when possible
- All documents are processed locally — no data leaves the machine
