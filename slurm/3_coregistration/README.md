# 3) Coregistration

Coregisters the scenes of a data cube to a common reference geometry.

## Recommended: build the JSON with the GUI

The easiest and safest way to produce a valid config is the
**Co-register Data Cube** tool:

1. Open `interactive/User_Interface_Tools.ipynb`.
2. Run **Analysis Ready Data Cube Tools** and open **2) Co-register Data Cube**.
3. Set your parameters interactively.
4. Click the **Copy JSON** button.
5. Paste the result into `coregistration.json` (replacing the content) and save.

> The GUI also documents what each parameter means, so it doubles as a reference.



## Example config

The example below uses the GUI's default parameters:

```json
{
  "parameters": {
    "input_path": "/dss/.../test.nc",
    "output_path": "/dss/.../test_coreg.nc",
    "max_cc": 100,
    "time_period": null,
    "grid_size": 7,
    "iteration": 1,
    "match_band": "auto",
    "min_inliers_keep": 3,
    "min_inliers_update_ref": 8,
    "max_cloud_update_ref": 20.0,
    "first_scene_mode": "composite",
    "composite_window_days": 30
  }
}
```





## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))
