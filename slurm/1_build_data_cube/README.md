# 1) Build Data Cube

Builds the initial spectral-temporal data cube from a STAC archive.

---

## Fastest: let the GUI submit for you

If you are running the notebook **on terrabyte**, the **Data Cube Builder** can do
steps 2 and 3 of the common guide by itself:

1. Open `interactive/User_Interface_Tools.ipynb`.
2. Run **Data Cube Builder** and set your parameters interactively.
3. Click **Submit SLURM**.

That writes the current settings into `build_data_cube.json` and runs
`sbatch ../config_cpu.cmd` for you - no editing, no terminal. The Status panel
reports the job id and where its logs land.

> The **one-time setup** (section 1 of [`../README.md`](../README.md): your email
> and account in `config_cpu.cmd`, and the micromamba env line) is still yours to
> do. The button checks it and refuses to submit while the `<e-mail>`,
> `<account>` or `/dss/.../envs` placeholders are still there, rather than
> queuing a job that would fail.
>
> The button also needs `sbatch` to exist, so it does nothing on a laptop - there,
> use Copy Settings below.

---

## Recommended elsewhere: build the JSON with the GUI

The easiest and safest way to produce a valid config by hand is the **Data Cube
Builder**:

1. Open `interactive/User_Interface_Tools.ipynb`.
2. Run **Data Cube Builder** and set your parameters interactively.
3. Click the **Copy Settings** button.
4. Paste the result into `build_data_cube.json` (replacing the content) and save.

> The GUI also documents what each parameter means, so it doubles as a reference.

## Going the other way: load a config back into the GUI

Click **Paste Settings** (next to Copy Settings) and paste the content of a
`build_data_cube.json` into the box that appears - the form fills itself with
those settings, so an HPC config can be inspected, tweaked and previewed
interactively.

Note that the config is a `get_stac_layers` call, so a few GUI-only choices are
not part of it and are reported as "not restored": the COG export mode (it
writes `"output": null`), the animation settings and the Result-panel date
ticks.

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
    "scene_cloud_coverage": 30,
    "cloud_masking": true,
    "output": "/dss/.../test.nc",
    "clip_raster": false,
    "resampling_method": "nearest",
    "aggregator": null,
    "stats": null
  }
}
```

## Final Step
- **Submit:** `submit.sh` (see the common guide in [`../README.md`](../README.md))