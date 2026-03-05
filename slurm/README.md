# Submitting **stac2cube** Features to HPC with **SLURM**
Run stac2cube workflows on LRZ **terrabyte** (HPC) using SLURM job submission.

> This guide is written for **`1_initial_data_cube`**, and can be repeated for  
> **`2_cloud_mask_data_cube`** and other modules added in the future.

---

## Quick overview
1. **Update SLURM setup** (only once)
2. **Configure the job** (edit JSON config)
3. **Submit the job**
4. **Track progress** via logs + portal

---

## 1) Update SLURM setup (only once)

### 1.1 Edit `slurm_setup.json`
Open:

- `.1_initial_data_cube/slurm_setup.json`

Then update:

- Replace: `#SBATCH --mail-user=<e-mail>`
- With: `#SBATCH --mail-user=your.email@domain.com` *(remove the `<` and `>`)*

### 1.2 Choose your partition
Change your cluster partition as desired:

- https://docs.terrabyte.lrz.de/services/terrabyte-hpc/introduction/

### 1.3 Choose how you run `slurm_run.py` (local vs shared env)

#### Option A — Local micromamba env (`-n stac2cube`)
Use this if **stac2cube is installed locally** in your micromamba:

```bash
# Local package
micromamba run -n stac2cube python slurm_run.py

# Shared DSS package (example)
# micromamba run -p /dss/.../envs/stac2cube python slurm_run.py
```

#### Option B — Shared DSS env (`-p /dss/.../envs/stac2cube`)
Use this if you want to use a **pre-installed env on DSS**:

```bash
# Local package
# micromamba run -n stac2cube python slurm_run.py

# Shared DSS package
micromamba run -p /dss/.../envs/stac2cube python slurm_run.py
```

> Pick **exactly one** run mode and keep the other commented.

---

## 2) Update job configuration

Edit and save:

- `.1_initial_data_cube/get_stac_layers.json`

### Notes
- JSON annotations/syntax are **different from Python** → it’s easy to make typos.
- Therefore, take a look at examples:
  - `./1_initial_data_cube/examples.txt`
- You can also use `1. Data Cube Builder` from `User_Interface_Tools.ipynb` in [interactive folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive), set your desired setup and hit `Copy JSON` button, finally paste it to `.1_initial_data_cube/get_stac_layers.json`.

---

## 3) Submit SLURM job

### Recommended (MobaXTerm)
- Middle-click: `.1_initial_data_cube/submit.sh`
- If using a laptop touchpad:
  - Right-click → copy file path to terminal → **Enter**

> This method is sometimes denied. If so, follow alternatives below.

### Alternatives
If not using MobaXTerm but on terminal or Windows PowerShell:

```bash
bash submit.sh
```

or:

```bash
bash path/to/submit.sh
```

or:

```bash
sbatch slurm_setup.cmd
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

**Tip:** Don’t forget to sort by time for the latest log files.  
On MobaXTerm you may need to close the editor and reopen it to track progress.

### 4.2 Recommended: terrabyte portal
My recommendation is to check the log files on:

- https://portal.terrabyte.lrz.de

You can simply **F5** the page to track the progress.

---

## Notes & troubleshooting

### Note 1 — First-time SLURM message
If you did a fresh installation, the first time a SLURM job is created you may receive a first-time message from WhiteToolBox on both log files.

This is normal and only occurs the first time.

> **Update:** WhiteToolBox is deactivated at the moment.

### Note 2 — No new `.out` / `.err` files?
If you don't see any newly generated `.out` and `.err` files, most likely it means that your SLURM job is **waiting in the queue**.

The time spent in the queue depends on both server status and access privileges.

To check if your job is sitting in the queue:
- https://portal.terrabyte.lrz.de → **Jobs** → **Active Jobs**  
- Check the job **status**.
