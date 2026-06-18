# 4) Super-resolution

Super-resolves a data cube (Sentinel-2) using the SEN2SRLite models.

- **Runner:** `slurm_run.py` -> `stac2cube.super_resolution.super_resolve_cube(**parameters)`
- **Config:** `get_superres_layers.json`
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))

> This is a **preliminary** README - detailed per-parameter docs and worked
> examples will be added later.

## `model_type`

- `null` (default): auto-detect from the cube's bands.
  - bands only within `blue, green, red, nir` -> `rgbn`
  - otherwise -> `full_spectral`
- `"rgbn"`: SEN2SRLite-RGBN.
- `"full_spectral"`: SEN2SRLite (requires all 10 bands:
  `blue, green, red, nir, rededge1, rededge2, rededge3, nir08, swir16, swir22`).

## Example config

```json
{
  "parameters": {
    "input_path": "./results/test.nc",
    "output_path": "./results/test_superres.nc",
    "var_name": "Spectral_Temporal_Stack",
    "nan_pixel_buffer": 8,
    "model_type": null
  }
}
```

The values above are the function's defaults; only `input_path` is strictly
required. Paths are relative to this feature folder (where the job runs).
