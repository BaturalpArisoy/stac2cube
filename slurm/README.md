# Submitting **stac2cube** Features to HPC with **SLURM**
Run stac2cube workflows on LRZ **terrabyte** (HPC) using SLURM job submission.

> This guide is written for **`1_build_data_cube`**.

---

## Quick overview
1. **Update SLURM setup** (only once)
2. **Configure the job** (edit JSON config)
3. **Submit the job**
4. **Track progress** via logs + portal

---

## 1) Update SLURM setup (only once)

### 1.1 Edit `slurm_setup.cmd`
Open:

- `./1_build_data_cube/slurm_setup.cmd`

Then update:

- Replace: `#SBATCH --mail-user=<e-mail>`
- With: `#SBATCH --mail-user=your.email@domain.com` *(remove the `<` and `>`)*
- Replace: `#SBATCH --account=<reach your contact person>`
- With your real terrabyte account/project string *(remove the `<` and `>`)*

> If you don't know your account string, reach your contact person or check
> an existing job here:
> https://portal.terrabyte.lrz.de/pun/sys/dashboard/activejobs -> any active job -> **Account**.
>
> A leftover `<...>` placeholder here makes sbatch fail with
> `Invalid directive found in batch script`.

### 1.2 Choose your partition
Change your cluster partition as desired:

- https://docs.terrabyte.lrz.de/services/terrabyte-hpc/introduction/

### 1.3 Choose how you run `slurm_run.py` (local or shared env)

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

## 2) Update job configuration

Edit and save:

- `./1_build_data_cube/get_stac_layers.json`

### Recommended: build the JSON with the GUI
JSON syntax differs from Python and is easy to get wrong by hand. The easiest
and safest way to produce a valid config is the **Data Cube Builder** GUI:

1. Open `interactive/0_User_Interface_Tools.ipynb`.
2. Run **Data Cube Builder** and set your desired parameters interactively.
3. Click the **Copy JSON** button.
4. Paste the result into `./1_build_data_cube/get_stac_layers.json` (replacing
   the existing content) and save.

> The GUI also documents what each parameter means, so it doubles as a reference
> if you are unsure about a field.

---

## 3) Submit SLURM job

### 3.1 Make `submit.sh` executable (only once)
Run this once in your terminal:

```bash
chmod +x ./1_build_data_cube/submit.sh
```

### 3.2 Submit

#### Recommended (MobaXTerm)
- Middle-click: `./1_build_data_cube/submit.sh`
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

or submit the batch script directly:

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

**Tip:** Don't forget to sort by time for the latest log files.  
On MobaXTerm you may need to close the editor and reopen it to track progress.

### 4.2 Recommended: terrabyte portal
My recommendation is to check the log files on:

- https://portal.terrabyte.lrz.de

You can simply **F5** the page to track the progress.

---

## Troubleshooting

### 1 - No new `.out` / `.err` files?
If you don't see any newly generated `.out` and `.err` files, most likely it means that your SLURM job is **waiting in the queue**.

The time spent in the queue depends on both server status and access privileges.

To check if your job is sitting in the queue:
- https://portal.terrabyte.lrz.de -> **Jobs** -> **Active Jobs**  
- Check the job **status**.
