# 2) Cloud Masking

Computes Sentinel-2 cloud **probability maps**, optional **mask layers**, and can
**mask** an existing data cube - via `s2cloudless`.

> **Where the heavy compute is:** generating the **probability maps** (Example 1) is the
> expensive step (runs `s2cloudless` per scene - this is why it goes to SLURM).
> Creating mask layers from a probability map, and masking an existing cube, are
> lightweight - you can do those interactively in the GUI
> (`interactive/0_User_Interface_Tools.ipynb` -> `ard_cube_tools` -> **1) Cloud Masking Data Cube**).

---

## Example 1 - probability maps only

Builds only the probability layer (`cloud_prob`) from an existing data cube. This
is the compute-heavy step you typically run on SLURM. Afterwards you can build
masks / mask the cube cheaply in the GUI.

```json
{
  "parameters": {
    "input_cube": "/dss/.../test.nc",
    "output_clouds": "/dss/.../test_cloud.nc"
  }
}
```

---

## Example 2 - probability maps + desired masks

Builds the probability layer **and** binary mask layers at one or more thresholds
(percent, 0-100). A list produces one `cloud_mask_<t>` band per threshold. No
masking of the data cube happens here.

```json
{
  "parameters": {
    "input_cube": "/dss/.../test.nc",
    "output_clouds": "/dss/.../test_cloud.nc",
    "threshold": [50, 70, 90]
  }
}
```

> The mask-generation part of this can also be done in the GUI:
> **b) Manually Build Cloud Masking Data Cube -> ii) Generate Masks from Probability Map**.

---

## Example 3 - fully automated workflow

Computes probability maps for the loaded cube, applies a **single** threshold to
build a mask, and uses it to **mask out the original data cube** in one run.
`masking` points at the data cube to be masked; polygon/daterange and the exact
timestamps are taken from it automatically.

```json
{
  "parameters": {
    "masking": "/dss/.../test.nc",
    "threshold": 70,
    "output_masked": "/dss/.../test_masked.nc",
    "output_clouds": "/dss/.../test_cloud.nc"
  }
}
```

- `threshold` here must be a **single** integer (not a list).
- `output_masked` is the masked data cube; `output_clouds` (optional) also saves
  the probability + mask stack.
- This is the same as the GUI's **a) Fully Automated Workflow**.

---

## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))
