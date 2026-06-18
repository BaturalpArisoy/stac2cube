# 1) Build Data Cube

Builds the initial spectral-temporal data cube from a STAC archive.

---

## Recommended: build the JSON with the GUI

The easiest and safest way to produce a valid config is the **Data Cube Builder**:

1. Open `interactive/0_User_Interface_Tools.ipynb`.
2. Run **Data Cube Builder** and set your parameters interactively.
3. Click the **Copy JSON** button.
4. Paste the result into `get_stac_layers.json` (replacing the content) and save.

> The GUI also documents what each parameter means, so it doubles as a reference.

---

## Example config

```json
{
  "parameters": {
    "mission": "s2",
    "source": "terrabyte",
    "polygon": "/dss/.../test.gpkg",
    "resolution": 10,
    "daterange": ["2024-04-01", "2024-04-30"],
    "bands": ["blue", "green", "red", "nir"],
    "indices": ["ndvi", "ndwi"],
    "max_cc": 100,
    "cloud_masking": true,
    "output": "/dss/.../test.nc",
    "clip_raster": false,
    "aggregator": null,
    "stats": null
  }
}
```

## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))