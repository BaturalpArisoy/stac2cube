# 4) Super-resolution

Super-resolves a data cube (Sentinel-2) to 2.5 m using the SEN2SRLite models.


> **Runs on GPU automatically.** The SEN2SRLite model is much faster on a GPU,
> so this feature's `submit.sh` already uses the GPU setup (`../config_gpu.cmd`,
> partition `hpda2_compute_gpu`) - no CPU/GPU swapping needed. Just make sure your
> email and account are filled in `config_gpu.cmd` (see the common guide).

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

## `model_dir` (ONE-TIME CHANGE)

The SEN2SRLite model files live in `interactive/model/` (`SEN2SRLite` and
`SEN2SRLite_RGBN`). The notebook finds them because it runs from `interactive/`,
but a SLURM job does not - so you must tell it where the model is via `model_dir`.

Set `model_dir` to the **`model` folder itself** (the one containing the
`SEN2SRLite` and `SEN2SRLite_RGBN` subfolders), e.g. `.../interactive/model`.
Pointing it at the parent `interactive` folder also works.

> If `model_dir` is wrong/missing, the run fails with a clear
> `Could not find the SEN2SRLite model ...` error.

## `compress`

Lossless zlib compression of the output NetCDF. Leave it `false` (default).

> **Warning:** compression shrinks the output file a further ~20-40%
> (scene-dependent), but the export step takes roughly **10x longer**. Enable
> it only for archiving, when disk space matters more than compute time.

---

## Example config

```json
{
  "parameters": {
    "input_path": "/dss/.../test.nc",
    "output_path": "/dss/.../test_superres.nc",
    "model_type": null,
    "model_dir": "/dss/.../stac2cube/interactive/model",
    "compress": false
  }
}
```

---

## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))
