# Submitting **stac2cube** Features to HPC with **SLURM**
Run stac2cube workflows on LRZ **terrabyte** (HPC) using SLURM job submission.

This is the **common** guide shared by all features. Each feature folder has its
own short README with feature-specific details.

## Folder layout

```
slurm/
├── README.md              <- this common guide
├── slurm_setup.cmd        <- shared SLURM setup (edit once, used by all features)
├── log/                   <- job logs (.out / .err)
├── 1_build_data_cube/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── get_stac_layers.json
├── 2_cloud_masking/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── get_cloud_layers.json
├── 3_coregistration/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── get_coreg_layers.json
└── 4_superresolution/
    ├── README.md          <- feature guide + JSON examples
    ├── submit.sh
    ├── slurm_run.py
    └── get_superres_layers.json
```

The `slurm_setup.cmd` lives **once** at the `slurm/` level and is shared by all
features. Each `submit.sh` runs it via `sbatch ../slurm_setup.cmd`, so the job's
working directory becomes that feature's folder and the right `slurm_run.py` and
`get_*.json` are picked up automatically.

---

## Quick overview
1. **Update the shared SLURM setup** (only once)
2. **Configure the job** (edit the feature's JSON config)
3. **Submit the job**
4. **Track progress** via logs + portal

---

## 1) Update the shared SLURM setup (only once)

### 1.0 Prerequisite: set up micromamba (one-time)
Set up micromamba on your terrabyte account **once**, via the modules system:

- https://docs.terrabyte.lrz.de/software/environments/micromamba/ -> **"Loading micromamba via modules system"**

Then verify it is recognized by simply typing:

```bash
micromamba
```

If it prints its usage/help, you are good.

### 1.1 Edit `slurm_setup.cmd`
Open the shared file:

- `./slurm_setup.cmd`

Then update:

- Replace: `#SBATCH --mail-user=<e-mail>`
- With: `#SBATCH --mail-user=your.email@domain.com` *(remove the `<` and `>`)*
- Replace: `#SBATCH --account=<reach your contact person>`
- With your real terrabyte account/project string *(remove the `<` and `>`)*

> If you don't know your account string, reach your contact person or check an
> existing job here:
> https://portal.terrabyte.lrz.de/pun/sys/dashboard/activejobs -> any active job -> **Account**.
>
> A leftover `<...>` placeholder here makes sbatch fail with
> `Invalid directive found in batch script`.

### 1.2 Choose your partition
Change your cluster partition as desired:

- https://docs.terrabyte.lrz.de/services/terrabyte-hpc/introduction/

### 1.3 Choose how you run `slurm_run.py` (local vs shared env)

#### Option A - Local micromamba env (`-n stac2cube`)
Use this if **stac2cube is installed locally** in your micromamba:

```bash
# Local package
micromamba run -n stac2cube python slurm_run.py

# Shared DSS package (example)
# micromamba run -p /dss/.../envs/stac2cube python slurm_run.py
```

#### Option B - Shared DSS env (`-p /dss/.../envs/stac2cube`)
Use this if you want to use a **pre-installed env on DSS**:

```bash
# Local package
# micromamba run -n stac2cube python slurm_run.py

# Shared DSS package
micromamba run -p /dss/.../envs/stac2cube python slurm_run.py
```

> Pick **exactly one** run mode and keep the other commented.

---

## 2) Configure the job

Each feature has its **own** JSON config in its folder:

| Feature | Config file |
| --- | --- |
| `1_build_data_cube` | `get_stac_layers.json` |
| `2_cloud_masking` | `get_cloud_layers.json` |
| `3_coregistration` | `get_coreg_layers.json` |
| `4_superresolution` | `get_superres_layers.json` |

Edit the JSON of the feature you want to run and save it.

> Each feature folder has its own `README.md` with **ready-to-copy JSON examples**
> for the common use cases. Open the feature's README before editing its JSON.

---

## 3) Submit SLURM job

### 3.1 Make all `submit.sh` executable (only once)
From inside the `slurm/` folder, run:

```bash
chmod +x ./*/submit.sh
```

### 3.2 Submit
Submit the feature you want by running its `submit.sh`.

#### Recommended (MobaXTerm)
- Middle-click the feature's `submit.sh`, e.g. `./1_build_data_cube/submit.sh`
- If using a laptop touchpad:
  - Right-click -> copy file path to terminal -> **Enter**

#### Alternatives (no execute bit needed)
If not using MobaXTerm, or if middle-click is still denied:

```bash
bash submit.sh
```

or:

```bash
bash path/to/submit.sh
```

or submit the shared batch script directly **from inside the feature folder**:

```bash
sbatch ../slurm_setup.cmd
```

---

🎉 Congratulations
Your job is submitted to HPC! <br><br> 
Good job!  




## 4) Following the progress

### 4.1 Check log files
- Progress of xarray computing:
  - `./log/*.out`
- Errors or warnings:
  - `./log/*.err`

**Tip:** Don't forget to sort by time for the latest log files.  
On MobaXTerm you may need to close the editor and reopen it to track progress.

### 4.2 Recommended: terrabyte portal
My recommendation is to check the log files on:

- https://portal.terrabyte.lrz.de

You can simply **F5** the page to track the progress.

---

## Notes & troubleshooting

### Note - No new `.out` / `.err` files?
If you don't see any newly generated `.out` and `.err` files, most likely it means that your SLURM job is **waiting in the queue**.

The time spent in the queue depends on both server status and access privileges.

To check if your job is sitting in the queue:
- https://portal.terrabyte.lrz.de -> **Jobs** -> **Active Jobs**  
- Check the job **status**.
