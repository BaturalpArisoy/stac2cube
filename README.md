# stac2ardcube 🛰️🧊
Transform STAC catalogs into Analysis-Ready Data Cubes (ARDCube) for Earth Observation (EO) applications

## Overview
**stac2ardcube** converts STAC catalogs into analysis-ready data cubes for efficient Earth Observation (EO) processing. This tool is designed to work on HPC systems and is optimized for users working on _terrabyte_. We recommend using *MobaXTerm* for accessing the _terrabyte_ login node and editing files.


## Installation:

There are two options for using this repository, depending on your access privileges.

---

### 1. Members of EORC Uni Wuerzburg
You should have access to EORC DSS container. This setup is set as default. 

#### a) Verify Access
$ dssusrinfo all
find if _pr94no-dss-0001_ is listed

#### b) Initiliaze Micromamba
if EORC DSS listed and micromamba haven't been activated yet,
micromamba should be initiliazed on the _terrabyte_ login-node:
$ module use /dss/dsstbyfs01/pn56su/pn56su-dss-0020/usr/share/modules/files/
$ module load micromamba
$ micromamba shell init --shell bash --root-prefix=~/micromamba
$ source ~/.bashrc

c) Configure Slurm Setup
Edit <path/to/stac2ardcube>/slurm/get_stac_layers/slurm_setup.cmd
Comment local package micromamba
Uncomment shared DSS package micromamba
Do the same for slurm/get_cloud_layers

d) Update Job Configuration and Submit
Edit slurm/get_stac_layers.json
Edit variables for data collection (examples.txt will be updated with further explanations!) and save
On MobaXTerm, middle click _submit.sh_ and enter
If laptop pad, right click, copy file path to terminal and enter

Congrulations, your job is submitted to HPC! You can check any error or the progress of xarray computing on slurm/log/ .err & .out
Dont forget to sort by time for the latest log files!


2. Users who are not member of EORC Uni Wuerzburg or do not have access to EORC DSS container

a) Set Up Micromamba Environment
$ module use /dss/dsstbyfs01/pn56su/pn56su-dss-0020/usr/share/modules/files/
$ module load micromamba
$ micromamba create -n stac2ardcube
$ micromamba shell init --shell bash --root-prefix=~/micromamba
$ source ~/.bashrc
$ micromamba activate stac2ardcube

b) Install stac2ardcube via PIP
Ensure you are in the directory containing setup.py and stac2ardcube env is activated, then choose your installation mode:
$ micromamba install pip
You should be on the folder where setup.py is located.
$ cd <path_to_stac2ardcube_parent_folder>

Stable mode:
pip install .
Editable mode (for development):
pip install -e .

c) Configure Slurm Setup
Edit <path/to/stac2ardcube>/slurm/get_stac_layers/slurm_setup.cmd
Comment shared DSS package micromamba
Uncomment local package micromamba
Do the same for slurm/get_cloud_layers

d) Update Job Configuration and Submit
Edit slurm/get_stac_layers.json
Edit variables for data collection (examples.txt will be updated with further explanations!) and save
On MobaXTerm, middle click _submit.sh_ and enter
If laptop pad, right click, copy file path to terminal and enter

Congrulations, your job is submitted to HPC! You can check any error or the progress of xarray computing on slurm/log/ .err & .out
Dont forget to sort by time for the latest log files!




Contact: https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/#c1122000