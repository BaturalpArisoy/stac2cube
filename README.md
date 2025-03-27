# stac2cube 🛰️🧊<br> Spatio-Temporal Asset Catalogs To Analysis-Ready Data Cubes
**stac2cube** converts STAC catalogs into Analysis-Ready Data Cubes for efficient Earth Observation (EO) processing. This tool is designed to work both on any local-machine and HPC system by _terrabyte_. We recommend using *MobaXTerm* for accessing the _terrabyte_ login node and editing files.

## Feature Overview
- **stac2cube.get_stac_layers**
    - Collects images from STAC catalogs for the selected mission based on users parameters.
    - Automatically preprocess spectral/radar values based on specifications of the selected mission.
    - Generates multi-dimensional data cubes, suitable for time-series.
    - The data cubes can be updated anytime without generating them from the scratch.
    - Available missions: **Sentinel-2 L2A, Sentinel-2 L1C, Sentinel-1 RTC, Landsat OT C2 L2, COP DEM Glo-30 (single time)**
- **stac2cube.get_cloud_layers**
    - Collects images from Sentinel-2 L1C to automatically apply s2cloudless cloud probability algorithm on data cube structure.
    - The result contains cloud probability maps and user defined binary cloud mask layers.
    - When selected, clouds from the generated data cube are automatically masked out.
- **stac2cube.coregister_scenes**
    - Applies coregistration algorithm on Sentinel-2 data cubes.
    - AROSICS package provides the coregistration algorithm<br>
    Daniel Scheffler. (2017, July 3). AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data (Version 0.12.1). Zenodo. https://doi.org/10.5281/zenodo.3742909
    - Fix the global X/Y shift between consecutive Sentinel-2 items.


## Installation:

There are two options for using this package, depending on your access privileges.

---

### 1. Members of EORC Uni Wuerzburg
You should have access to EORC DSS container.

#### a) Verify Access
    $ dssusrinfo all
`pr94no-dss-0001` should be in the list.

#### b) Initialize Micromamba
If EORC DSS listed and micromamba hasn't been activated yet,
micromamba should be initialized on the _terrabyte_ login-node:
```sh
$ module use /dss/dsstbyfs01/pn56su/pn56su-dss-0020/usr/share/modules/files/
$ module load micromamba
$ micromamba shell init --shell bash --root-prefix=~/micromamba
$ source ~/.bashrc
```
#### c) Optional: Check Configuration Slurm Setup (Default configuration)
- Edit `<path/to/stac2cube>/slurm/get_stac_layers/slurm_setup.cmd>`
- Comment (disable) local package micromamba `micromamba run -n stac2cube python slurm_run.py`
- Uncomment (enable) shared DSS package micromamba `micromamba run -p /dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/envs/stac2cube python slurm_run.py`
- Do the same for `slurm/get_cloud_layers`



### 2. Users who are not member of EORC Uni Wuerzburg or do not have access to EORC DSS container

#### a) Set Up Micromamba Environment
    $ module use /dss/dsstbyfs01/pn56su/pn56su-dss-0020/usr/share/modules/files/
    $ module load micromamba
    $ micromamba create -n stac2cube
    $ micromamba shell init --shell bash --root-prefix=~/micromamba
    $ source ~/.bashrc
    $ micromamba activate stac2cube

#### b) Install stac2cube via PIP or UV
Ensure you are in the directory containing setup.py and stac2cube env is activated, then choose your installation mode:

    $ micromamba install pip
You should be on the folder where `pyproject.toml` is located.

    $ cd <path_to_stac2cube_parent_folder>

Stable mode:<br>

    $ pip install .

Editable mode (for development):<br>

    $ pip install -e .


#### c) Configure Slurm Setup
- Edit `<path/to/stac2cube>/slurm/get_stac_layers/slurm_setup.cmd>`
- Comment (disable) shared DSS package micromamba `micromamba run -p /dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/envs/stac2cube python slurm_run.py`
- Uncomment (enable) local package micromamba `micromamba run -n stac2cube python slurm_run.py`
- Do the same for `slurm/get_cloud_layers`

---

## Examples:
Jupyter notebooks on how to use stac2cube features and how to process data cube structure can be found in the [interactive folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive).

## How to run on HPC (terrabyte users only)
A documentation file on how to use stac2cube features on terrabyte's HPC for compute-intensive processes and for faster processing time can be found in the [slurm folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/slurm).

## What's upcoming?:
- [ ] Sentinel-2 co-registration (Under development)
- [x] Caching mechanism to automatically update the missing scenes
- [x] More advanced interactive tools for better experience
- [x] Get table of available missions with details and what function instances they can receive
- [ ] Import bbox list with projected coords: proj to geographic transformation (Under development)
- [ ] Improvements for get_cloud_layers function for simpler use
- [ ] Cloud shadow detection and masking
- [ ] Cleaner way to work with CRS information of data arrays (Under development)
- [ ] Native cloud masking for Landsat and Sentinel-2 scenes (Under development)
- [ ] Orbit mode selection for Sentinel-1: Ascending/Descending
- [x] Switch python package setup from setup.py to pyproject.toml: enables uv install besides pip 
- [ ] Add new spectral indices: EVI, Built-up Index (More upon request!)
- [ ] Quite mode
- [ ] Verbose mode

## How to cite:
paper to be published <3

<br><br>
Contact: https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/