# Submitting **stac2cube** Features to HPC with **SLURM**
Run stac2cube workflows on DLR&LRZ **terrabyte** (HPC) using SLURM job submission.


---
> **Good news:** Section **1** is a **ONE-TIME setup!** Do it once, never again.
> After that, running any feature is just **2) Configure the job** and **3) Submit the job**.
---


## Quick overview
1. **ONE-TIME SETUP** 
2. **Configure the job** - edit the feature's JSON config
3. **Submit the job**
4. **Extras**
    - Track the progress
    - Notes & troubleshooting
    - Switching a feature between CPU and GPU
    - Folder layout

---

## 1) One-time setup

### 1.1 Micromamba Setup
Set up micromamba on your terrabyte account **once**, via the modules system:

- https://docs.terrabyte.lrz.de/software/environments/micromamba/ -> **"Loading micromamba via modules system"**

Then verify it is recognized by simply typing:

```bash
micromamba
```

If it prints its usage/help, you are good.

### 1.2 Edit the config files
Both shared setup files need your email and account:

- `./config_cpu.cmd` (CPU -> Build Data Cube, Cloud Masking, Co-registration)
- `./config_gpu.cmd` (GPU -> Super-resolution)

In **each** file, update:

- Replace: `#SBATCH --mail-user=<e-mail>`
- With: `#SBATCH --mail-user=your.email@domain.com` *(remove the `<` and `>`)*
- Replace: `#SBATCH --account=<account>`
- With your real terrabyte account/project string *(remove the `<` and `>`)*

> If you don't know your account string, reach your contact person or check an
> existing job here:
> https://portal.terrabyte.lrz.de/pun/sys/dashboard/activejobs -> any active job -> **Account**.
>
> A leftover `<...>` placeholder here makes sbatch fail with
> `Invalid directive found in batch script`.

<br>

**FINALLY:** <br>Protecting these configurations from feature repository pulls is recommended, otherwise the changes will be lost!
```bash
git update-index --skip-worktree slurm/config_cpu.cmd slurm/config_gpu.cmd
```

### 1.3 Choose your python environment (local vs shared env)

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
| `1_build_data_cube` | `build_data_cube.json` |
| `2_cloud_masking` | `cloud_masking.json` |
| `3_coregistration` | `coregistration.json` |
| `4_superresolution` | `superresolution.json` |

Edit the JSON of the feature you want to run and save it.

> Each feature folder has its own `README.md` with **ready-to-copy JSON examples**
> for the common use cases. Open the feature's README before editing its JSON.

---

## 3) Submit SLURM job

Submit the feature you want by running its `submit.sh`.

#### Recommended (MobaXTerm)
- Middle-click the feature's `submit.sh`, e.g. `./1_build_data_cube/submit.sh`
- If using a laptop touchpad:
  - Right-click -> copy file path to terminal -> **Enter**

#### Alternatives (if middle-click is denied)
If not using MobaXTerm, or if middle-click is denied:

```bash
bash submit.sh
```

or:

```bash
bash path/to/submit.sh
```

or submit the right batch script directly **from inside the feature folder**
(`config_cpu.cmd` for features 1-3, `config_gpu.cmd` for super-resolution):

```bash
sbatch ../config_cpu.cmd
```

---

🎉 Congratulations
Your job is submitted to HPC! <br><br> 
Good job! Now you can track the progress, explained below.

<br>
<br>


# Extras


## Track the progress

**Check the log files** (in `./log/`):
- xarray computing progress -> `./log/*.out`
- errors / warnings -> `./log/*.err`

**Tip:** sort by time for the latest log files. On MobaXTerm you may need to close
and reopen the editor to refresh.

**terrabyte portal (recommended):** open https://portal.terrabyte.lrz.de and press
**F5** to refresh.

---

## Notes & troubleshooting

**No new `.out` / `.err` files?** <br><br> Your job is most likely still **waiting in the
queue** (the wait depends on server load and your access privileges). <br> Confirm at
https://portal.terrabyte.lrz.de -> **Jobs** -> **Active Jobs** and check the
**status**.

---

## Switching a feature between CPU and GPU

Each feature's `submit.sh` chooses the setup by the file it submits:

- `sbatch ../config_cpu.cmd` -> runs on **CPU**
- `sbatch ../config_gpu.cmd` -> runs on **GPU**

By default, features 1-3 use `config_cpu.cmd` and super-resolution uses
`config_gpu.cmd`. To switch a feature, open its `submit.sh` and change that one
line.

**Example - run Cloud Masking on GPU:** open `2_cloud_masking/submit.sh` and
change `sbatch ../config_cpu.cmd` to:

```bash
sbatch ../config_gpu.cmd
```

> Reminder: the config you switch to must have your email/account filled in
> (see **1.2**) - otherwise the job will fail.

---

## Choose your partition

**The default setup is good enough** for most use cases. Only change your cluster
partition in `config.cmd` files if you specifically need to:

- https://docs.terrabyte.lrz.de/services/terrabyte-hpc/introduction/

---

## Folder layout

```
slurm/
├── README.md              <- this common guide
├── config_cpu.cmd         <- CPU SLURM setup (features 1-3)
├── config_gpu.cmd         <- GPU SLURM setup (feature 4, super-resolution)
├── log/                   <- job logs (.out / .err)
├── 1_build_data_cube/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── build_data_cube.json
├── 2_cloud_masking/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── cloud_masking.json
├── 3_coregistration/
│   ├── README.md          <- feature guide + JSON examples
│   ├── submit.sh
│   ├── slurm_run.py
│   └── coregistration.json
└── 4_superresolution/
    ├── README.md          <- feature guide + JSON examples
    ├── submit.sh
    ├── slurm_run.py
    └── superresolution.json
```

There are two shared setup files at the `slurm/` level: `config_cpu.cmd` (used by
features 1-3) and `config_gpu.cmd` (used by feature 4, super-resolution, whose
SEN2SRLite model runs **much faster on a GPU**). Each `submit.sh` already points
at the right one, so the job's working directory becomes that feature's folder and
the correct `slurm_run.py` and `.json` are picked up automatically.

