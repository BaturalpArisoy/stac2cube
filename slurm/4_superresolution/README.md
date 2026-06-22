# 4) Super-resolution

Super-resolves a data cube (Sentinel-2) to 2.5 m using the SEN2SRLite models.


> **Recommended: run on a CUDA-capable GPU cluster.** The SEN2SRLite models run
> the inference far faster on a GPU. To send this job to a GPU node, copy the
> GPU `#SBATCH` lines from [`gpu_example.cmd`](gpu_example.cmd) into the shared
> `../slurm_setup.cmd` (replacing the matching CPU lines). Swap them back when
> running the CPU-only features.

## `model_type`

Leave it as `null` to auto-detect from the cube's bands or set it explicitly:

- `null`: auto-detect
  - `rgbn` if only `blue, green, red, nir` are present (**10 m -> 2.5 m**)
  - `full_spectral` otherwise (**10 m + 20 m -> 2.5 m**)
- `"rgbn"`: force 10 m only -> needs `blue, green, red, nir`.
- `"full_spectral"`: force 10 m + 20 m -> needs all 10:
  `blue, green, red, nir, rededge1, rededge2, rededge3, nir08, swir16, swir22`.

> The input cube must already contain the bands the chosen model requires,
> otherwise the run fails with a missing-bands error.

## `model_dir` (required on HPC)

The SEN2SRLite model files live in `interactive/model/` (`SEN2SRLite` and
`SEN2SRLite_RGBN`). The notebook finds them because it runs from `interactive/`,
but a SLURM job does not - so you must tell it where the model is via `model_dir`.

Set `model_dir` to the **`model` folder itself** (the one containing the
`SEN2SRLite` and `SEN2SRLite_RGBN` subfolders), e.g. `.../interactive/model`.
Pointing it at the parent `interactive` folder also works.

> If `model_dir` is wrong/missing, the run fails with a clear
> `Could not find the SEN2SRLite model ...` error.

---

## Example config

```json
{
  "parameters": {
    "input_path": "/dss/.../test.nc",
    "output_path": "/dss/.../test_superres.nc",
    "model_type": null,
    "model_dir": "/dss/.../stac2cube/interactive/model"
  }
}
```

---

## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))
