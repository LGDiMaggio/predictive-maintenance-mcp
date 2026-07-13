# Bearing Catalogs Directory

This directory contains bearing specification catalogs for automatic geometry lookup.

## Purpose

When a machine manual specifies a bearing designation (e.g., "SKF 6205-2RS") but doesn't include the bearing geometry (number of balls, ball diameter, pitch diameter), the system searches this directory to find the specifications needed for calculating characteristic frequencies.

## Search Priority

The LLM follows this workflow:

```
1. Check MACHINE MANUAL for bearing geometry
   ↓ Not found?
   
2. Search BEARING CATALOGS (this directory)
   ↓ Not found?
   
3. ASK USER for specifications
```

## Files in This Directory

### `common_bearings_catalog.json`
- **Content**: a SMALL set of bearings whose internal geometry is traceable to a
  public source. Small by design: unverifiable entries were removed, not
  approximated ("honest and small beats rich and fake").
- **Current entries**:
  - `6205` — SKF 6205-2RS JEM, geometry published by the CWRU Bearing Data Center
  - `6203` — SKF 6203-2RS JEM, geometry published by the CWRU Bearing Data Center
  - `UER204` — LDK UER204, geometry published with the XJTU-SY run-to-failure dataset
- **Data**: num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg, bore_mm, outer_diameter_mm, width_mm, **source (mandatory citation)**
- **Validation**: `tests/test_bearing_catalog_validation.py` checks every entry for
  physical validity (bore < pitch < OD, ball fit, kinematic identities) and a
  non-empty source field. New entries that fail these checks are rejected in CI.

### Future Additions (User Can Add)

You can add manufacturer catalogs as PDF files:
- `SKF_deep_groove_catalog.pdf`
- `FAG_ball_bearings.pdf`
- `NSK_bearing_catalog.pdf`

The system will:
1. Search JSON catalog first (fast)
2. Fall back to PDF search if needed (slower but comprehensive)

## How to Add More Bearings

### Option 1: Add to JSON (Recommended for Small Sets)

Entries are accepted ONLY with a verifiable public source (manufacturer
datasheet, dataset documentation, or peer-reviewed paper) recorded in the
mandatory `source` field. Edit `common_bearings_catalog.json` and add entries
like:

```json
"6205": {
  "designation": "6205",
  "type": "Deep Groove Ball Bearing",
  "series": "62xx",
  "num_balls": 9,
  "ball_diameter_mm": 7.94,
  "pitch_diameter_mm": 39.04,
  "contact_angle_deg": 0.0,
  "bore_mm": 25,
  "outer_diameter_mm": 52,
  "width_mm": 15,
  "source": "<URL or citation for the geometry data>"
}
```

Run `pytest tests/test_bearing_catalog_validation.py` afterwards: it validates
every entry for physical consistency and a non-empty source.

### Option 2: Upload Manufacturer PDF Catalog

1. Download PDF catalog from manufacturer website:
   - SKF: https://www.skf.com/us/products/rolling-bearings
   - FAG: https://www.schaeffler.com/en/products-and-solutions/industrial/product-finder/rolling-bearings/
   - NSK: https://www.nskamericas.com/en/products/bearing-product-index.html

2. Place PDF in this directory

3. System will extract specifications automatically (future feature)

## Usage Examples

### Via Claude Desktop

```
"What are the specifications for bearing 6205?"
→ System searches catalog → Returns geometry with its source citation

"Calculate bearing frequencies for SKF 6205 at 1500 RPM"
→ Looks up 6205 geometry → Calculates BPFO, BPFI, BSF, FTF
```

### Via MCP Tool

```python
specs = await search_bearing_catalog("6205")
# Returns:
{
  "designation": "6205",
  "type": "Deep Groove Ball Bearing",
  "num_balls": 9,
  "ball_diameter_mm": 7.94,
  "pitch_diameter_mm": 39.04,
  "contact_angle_deg": 0.0,
  "bore_mm": 25,
  "outer_diameter_mm": 52,
  "width_mm": 15,
  "source": "CWRU Bearing Data Center, drive-end bearing SKF 6205-2RS JEM (...)"
}
```

## Important Notes

1. **Catalog is Fallback Only**: Always check machine manual first
2. **Source-Traceable Only**: every entry carries a mandatory `source` citation;
   manufacturer-specific variants may still differ — verify against your bearing
3. **User Responsibility**: If bearing not in catalog, user must provide specifications
4. **No Web Search**: System does NOT search online for privacy/reliability reasons
5. **Extensible**: You can expand the JSON (with sources) or add PDF catalogs as needed

## LLM Behavior

When bearing geometry is needed:

✅ **Correct Workflow**:
```
1. "Checking machine manual for bearing geometry..."
2. "Not found in manual. Searching bearing catalog..."
3. "Found 6205 in catalog: 9 balls, 7.94mm diameter"
4. "Calculating frequencies with catalog specifications..."
```

❌ **Incorrect Workflow**:
```
1. "I'll estimate typical 6205 specifications..." (NO!)
2. "Searching online for bearing data..." (NO!)
3. "Using standard values for similar bearings..." (NO!)
```

The LLM should **NEVER**:
- Guess or estimate bearing geometry
- Use "typical" values without explicit confirmation
- Search online (not implemented, privacy concerns)
- Assume specifications from bearing series alone

## Technical Details

### JSON Schema

```json
{
  "catalog_info": {
    "name": "string",
    "policy": "string",
    "version": "string",
    "date": "YYYY-MM-DD"
  },
  "bearings": {
    "designation": {
      "designation": "string",
      "type": "string",
      "series": "string",
      "num_balls": integer,
      "ball_diameter_mm": float,
      "pitch_diameter_mm": float,
      "contact_angle_deg": float,
      "bore_mm": float,
      "outer_diameter_mm": float,
      "width_mm": float,
      "source": "string (MANDATORY: URL or citation for the geometry data)"
    }
  }
}
```

### Cleaning Algorithm

The system automatically cleans bearing designations:
- `"SKF 6205-2RS"` → `"6205"`
- `"FAG 6206 ZZ"` → `"6206"`
- `"NSK 6207"` → `"6207"`

Removes: manufacturer prefixes (SKF, FAG, NSK, NTN, TIMKEN, KOYO, INA)
Removes: suffixes (-2RS, -ZZ, -2Z, etc.)

---

**Need more bearings?** Edit the JSON file or upload manufacturer PDF catalogs!
