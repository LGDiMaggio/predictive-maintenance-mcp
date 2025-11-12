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
- **Content**: 20 common ISO deep groove ball bearings
- **Series**: 
  - 6200 series: 6200, 6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208, 6209, 6210
  - 6300 series: 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310
- **Data**: num_balls, ball_diameter_mm, pitch_diameter_mm, contact_angle_deg, bore_mm, outer_diameter_mm, width_mm
- **Source**: ISO 15:2017 standard dimensions

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

Edit `common_bearings_catalog.json` and add entries like:

```json
"6211": {
  "designation": "6211",
  "type": "Deep Groove Ball Bearing",
  "series": "62xx",
  "num_balls": 9,
  "ball_diameter_mm": 17.462,
  "pitch_diameter_mm": 77.5,
  "contact_angle_deg": 0.0,
  "bore_mm": 55,
  "outer_diameter_mm": 100,
  "width_mm": 21
}
```

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
→ System searches catalog → Returns geometry

"Calculate bearing frequencies for SKF 6207 at 1500 RPM"
→ Looks up 6207 geometry → Calculates BPFO, BPFI, BSF, FTF
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
  "pitch_diameter_mm": 34.55,
  "contact_angle_deg": 0.0,
  "bore_mm": 25,
  "outer_diameter_mm": 52,
  "width_mm": 15,
  "source": "catalog_json"
}
```

## Important Notes

1. **Catalog is Fallback Only**: Always check machine manual first
2. **Generic ISO Specifications**: Manufacturer-specific bearings may have slight variations
3. **User Responsibility**: If bearing not in catalog, user must provide specifications
4. **No Web Search**: System does NOT search online for privacy/reliability reasons
5. **Extensible**: You can expand the JSON or add PDF catalogs as needed

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
    "source": "string",
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
      "width_mm": float
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
