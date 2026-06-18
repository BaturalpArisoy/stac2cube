# 3) Coregistration

Coregisters the scenes of a data cube to a common reference geometry.

- **Runner:** `slurm_run.py` -> `stac2cube.coregistration.coregister_cube(**parameters)`
- **Config:** `get_coreg_layers.json`
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))

> This is a **preliminary** README - detailed per-parameter docs and worked
> examples will be added later.

## Example config

```json
{
  "parameters": {
    "input_path": "./results/test.nc",
    "output_path": "./results/test_coreg.nc",
    "stack_name": "Spectral_Temporal_Stack",
    "first_scene_mode": "composite",
    "composite_window_days": 30,
    "grid_size": 3,
    "min_reliability_keep": 10.0,
    "min_reliability_update_ref": 50.0,
    "max_cloud_update_ref": 20.0,
    "max_cc": null,
    "time_period": null,
    "iteration": 1
  }
}
```

The values above are the function's defaults; only `input_path` is strictly
required. Paths are relative to this feature folder (where the job runs).
