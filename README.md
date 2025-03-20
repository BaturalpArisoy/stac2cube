# stac2ardcube 🛰️🧊
Transform STAC catalogs into Analysis-Ready Data Cubes (ARDCube) for Earth Observation (EO) applications

## Overview
**stac2ardcube** converts STAC catalogs into Analysis-Ready Data Cubes for efficient Earth Observation (EO) processing. This tool is designed to work on HPC systems and is optimized for users working on _terrabyte_. We recommend using *MobaXTerm* for accessing the _terrabyte_ login node and editing files.


## Installation:

There are two options for using this repository, depending on your access privileges.

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
#### c) Optional: Configure Slurm Setup (Default configuration)
- Edit `<path/to/stac2ardcube>/slurm/get_stac_layers/slurm_setup.cmd`
- Comment (disable) local package micromamba `micromamba run -n stac2ardcube python slurm_run.py`
- Uncomment (enable) shared DSS package micromamba `micromamba run -p /dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/envs/stac2ardcube python slurm_run.py`
- Do the same for `slurm/get_cloud_layers`

#### d) Update Job Configuration and Submit
- Edit `slurm/get_stac_layers.json`
- Edit variables for data collection (take a look at `examples.txt`) and save
- On MobaXTerm, middle click `submit.sh` and enter
- If laptop pad, right click, copy file path to terminal and enter<br>

Congratulations, your job is submitted to HPC! You can check any error or the progress of xarray computing on `slurm/log/ .err & .out`<br>
Dont forget to sort by time for the latest log files!


### 2. Users who are not member of EORC Uni Wuerzburg or do not have access to EORC DSS container

#### a) Set Up Micromamba Environment
    $ module use /dss/dsstbyfs01/pn56su/pn56su-dss-0020/usr/share/modules/files/
    $ module load micromamba
    $ micromamba create -n stac2ardcube
    $ micromamba shell init --shell bash --root-prefix=~/micromamba
    $ source ~/.bashrc
    $ micromamba activate stac2ardcube

#### b) Install stac2ardcube via PIP
Ensure you are in the directory containing setup.py and stac2ardcube env is activated, then choose your installation mode:

    $ micromamba install pip
You should be on the folder where `setup.py` is located.

    $ cd <path_to_stac2ardcube_parent_folder>

Stable mode:<br>

    $ pip install .

Editable mode (for development):<br>

    $ pip install -e .


#### c) Configure Slurm Setup
- Edit `<path/to/stac2ardcube>/slurm/get_stac_layers/slurm_setup.cmd`
- Comment (disable) shared DSS package micromamba `micromamba run -p /dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/envs/stac2ardcube python slurm_run.py`
- Uncomment (enable) local package micromamba `micromamba run -n stac2ardcube python slurm_run.py`
- Do the same for `slurm/get_cloud_layers`

#### d) Update Job Configuration and Submit
- Edit `slurm/get_stac_layers.json`
- Edit variables for data collection (`examples.txt` will be updated with further explanations!) and save
- On MobaXTerm, middle click `submit.sh` and enter
- If laptop pad, right click, copy file path to terminal and enter<br><br>


Congratulations, your job is submitted to HPC! You can check any error or the progress of xarray computing on `slurm/log/ .err & .out`<br>
Note: The first time a Slurm job is created, you will receive a warning message about WhiteToolBox on both log files. This is normal and only occurs the first time.<br>
Dont forget to sort by time for the latest log files!<br>

---
<br><br>

## What's upcoming?:
- Sentinel 2 co-registration (Under development)
- Caching mechanism to automatically update the missing scenes
- More advanced interactive tools for better experience (Under development)
- Improvements of get_cloud_layers function for easier use
- Cloud shadow detection and masking
- Cleaner way to work with CRS information of data arrays (Under development)
- Native cloud masking for Landsat and Sentinel (Under development)
- Orbit mode selection for Sentinel-1: Ascending/Descending
- Sentinel tile information extraction

## How to cite:
paper to be published <3

<br><br>
Contact: https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/