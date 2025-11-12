# Machine Manuals Directory

This directory contains equipment manuals and technical documentation for analysis.

## Test Example Available

A **test pump manual** is included for demonstration:
- **File**: `test_pump_manual.pdf` (also available as `.txt`)
- **Content**: Complete pump specifications with bearings, RPM, power, seals, maintenance
- **Purpose**: Test and learn the documentation reader features
- **Usage**: Try asking Claude about this manual to see how the system works

Example prompts to try:
```
"What bearings are used in the test pump manual?"
"Calculate bearing frequencies for the bearings in test_pump_manual.pdf"
"What is the maintenance schedule in the test manual?"
```

## How to Use

1. **Upload PDF manuals** to this directory
2. **Use MCP Resources** to let Claude read the full manual:
   ```
   "Read the pump manual and tell me what bearings are specified"
   ```
3. **Use extraction tools** for structured data:
   ```
   extract_manual_specs("pump_manual.pdf")
   ```

## Example Questions Claude Can Answer

With a manual uploaded, Claude can answer:

- ❓ "What bearings are mounted in this machine?"
- ❓ "What are the operating speeds?"
- ❓ "What type of mechanical seal is used?"
- ❓ "How many impeller vanes?"
- ❓ "What are the characteristic bearing frequencies?"
- ❓ "What's the recommended maintenance interval?"
- ❓ "Are there any special lubrication requirements?"

## Hybrid Approach

The system uses **3 methods** to answer questions:

### 1. MCP Resources (Primary)
Claude reads the **full PDF text** directly:
- Can answer ANY question about manual content
- No pre-defined extraction patterns needed
- Understands context (e.g., "nominal" vs "maximum" RPM)

### 2. Structured Extraction (Hints)
Regex-based extraction provides quick hints:
- Bearing designations (e.g., SKF 6205-2RS)
- RPM values (e.g., 1500 RPM, 3000 RPM max)
- Power ratings (e.g., 15 kW, 20 HP)

### 3. Catalog Lookup (Fallback)
If bearing geometry not in manual:
- Local cache (common bearings: 6205, 6206, etc.)
- Future: Web search (SKF/FAG APIs)
- Auto-calculates BPFO/BPFI/BSF/FTF

## Example Workflow

**Scenario**: Diagnose vibration fault on pump

1. **Upload manual**: `pump_XYZ_manual.pdf`

2. **Claude extracts bearings**:
   ```
   "What bearings are used in pump XYZ?"
   → Drive end: SKF 6205-2RS
   → Non-drive end: NSK 6206
   ```

3. **Claude looks up geometry**:
   ```
   lookup_bearing_in_catalog("6205")
   → 9 balls, Bd=7.94mm, Pd=34.55mm
   ```

4. **Claude calculates frequencies**:
   ```
   calculate_bearing_frequencies(
       num_balls=9,
       ball_diameter_mm=7.94,
       pitch_diameter_mm=34.55,
       shaft_speed_rpm=1475
   )
   → BPFO: 85.20 Hz (outer race fault)
   → BPFI: 136.05 Hz (inner race fault)
   ```

5. **Claude performs diagnosis**:
   ```
   generate_envelope_report("pump_vibration.csv")
   → Peak at 85 Hz detected → Outer race fault confirmed!
   ```

## Test Manual (Example)

To test the system, create a sample manual with:

```
PUMP SPECIFICATIONS - MODEL ABC-123
===================================

Bearings:
- Drive end: SKF 6205-2RS (deep groove ball bearing)
- Non-drive end: NSK 6206 (deep groove ball bearing)

Operating conditions:
- Nominal speed: 1475 RPM
- Maximum speed: 3000 RPM
- Motor power: 15 kW (400V, 3-phase)

Mechanical seal: Type 21, carbon/ceramic faces
Impeller: Bronze, closed type, 5 vanes
Shaft: Stainless steel 316, 25mm diameter

Maintenance:
- Bearing lubrication: Every 6 months (lithium grease)
- Mechanical seal inspection: Every 12 months
- Impeller inspection: Annually
```

Save as PDF and upload to this directory.

## Supported File Formats

- **PDF** (preferred) - Full text extraction with PyPDF2
- **Future**: DOCX, TXT, images with OCR

## Privacy Note

Manuals are stored **locally only**. No data is sent to external services
unless you explicitly use web search features (future).

## Tips for Best Results

✅ **Good manual quality**:
- Clear text (not scanned images)
- Well-structured sections
- Technical specifications in tables

✅ **Multiple RPM values are OK**:
- Claude understands "rated speed" vs "max speed"
- Can interpret context from manual

✅ **Missing bearing geometry**:
- System will auto-lookup in catalog
- If not found, suggest uploading bearing datasheet

❌ **Poor results from**:
- Low-quality scans (OCR needed)
- Handwritten notes
- Non-technical marketing brochures
