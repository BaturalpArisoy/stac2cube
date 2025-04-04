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
- **stac2cube.coregister_scenes (still under-development)**
    - Applies coregistration algorithm on Sentinel-2 data cubes.
    - AROSICS package provides the coregistration algorithm<br>
    Daniel Scheffler. (2017, July 3). AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data (Version 0.12.1). Zenodo. https://doi.org/10.5281/zenodo.3742909
    - Fix the global X/Y shift between consecutive Sentinel-2 items.


## Installation

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
Ensure that stac2cube env is activated.

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

## Examples
Jupyter notebooks on how to use stac2cube features and how to process data cube structure can be found in the [interactive folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive).

## How to run on HPC (terrabyte users only)
A documentation file on how to use stac2cube features on terrabyte's HPC for compute-intensive processes and for faster processing time can be found in the [slurm folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/slurm).

## What's upcoming?
- [ ] Sentinel-2 co-registration (Under development-almost done)
- [ ] Merge Landsat TM and OLI missions.
- [x] Caching mechanism to automatically update the missing scenes
- [x] More advanced interactive tools for better experience
- [x] Sentinel-1: Orbit-state selection: Ascending/Descending
- [ ] Sentinel-1: Automatic preprocessing, e.g. SNAP tools.
- [ ] Add new spectral indices: EVI, Built-up Index (More upon request!)
- [ ] Add SLURM job array to submit multiple json files at once
- [ ] Silent parameter for get_stac_layers that will automatically switch to terrabyte STAC catalogs when run on HPC
- [ ] Import bbox list with projected coords: proj to geographic transformation (Under development)
- [ ] Improvements for get_cloud_layers function: mask calculation function, mask l2a data directly
- [ ] Cloud shadow detection and masking for Sentinel-2
- [x] Native cloud masking for Landsat and Sentinel-2 scenes (Under development)
- [x] Switch python package setup from setup.py to pyproject.toml: enables uv install besides pip 
- [ ] Quite mode
- [ ] Verbose mode

## Access and Licensing Details for STAC Catalogs

### Access to STAC Catalogs

- **Important**: _terrabyte_ STAC catalogs can be only computed when working on a _terrabyte_ environment.<br>
- However, stac2cube package is designed to work on both local-machine without _terrabyte_ connection and within _terrabyte_ HPC environment.<br>
- Therefore, a silent parameter will enable _terrabyte_ STAC catalogs when a SLURM job is activated.<br>
- The default set-up (_terrabyte_ disabled) will feature STAC catalogs that provide "open-access data" (**not** open-source).<br>
- Thus, note that stac2cube package **can not** guarantee unlimited access to these open-access data catalogs in the future!

### STAC Catalog Licenses

| Provider   | Service           | STAC API                                        | License                                                                                      | Open-Access | Open-Source |
|------------|-------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------|-------------|-------------|
| DLR        | terrabyte         | https://stac.terrabyte.lrz.de/public/api/       | MIT License Copyright (c) 2024 Deutsches Zentrum für Luft- und Raumfahrt e.V.                | No          | No          |
| Element 84 | Earth Search      | https://earth-search.aws.element84.com/v1/      | Apache License 2.0                                                                           | Yes         | Yes         |
| Microsoft  | Planetary Computer| https://planetarycomputer.microsoft.com/api/    | MIT License Copyright (c) Microsoft Corporation.                                             | Yes         | No          |

### Why use terrabyte then?

Why do _terraybte_ users collect data from _terrabyte_ STAC catalog instead of  open-source Earth Search?

- The data by Element 84 is stored in AWS S3 services. 
- The data by DLR is stored in the servers of The Leibniz Supercomputing Centre (LRZ) in Garching/Munich.
- When working on a _terrabyte_ environment, the data query is returned from same server instead of connecting to AWS. <br><br>

#### **Example**: Query for Sentinel-2 L2A: 
- daterange: ["2017-01-01", "2025-03-28"]
- polygon: Nord Hubland/Würzburg/Germany<br>

| Service          | Returned Date  | Processing Time (s)|
|------------------|----------------|--------------------|
|terrabyte         | 1134           | 24.0               |
|Earth Search      | 1038           | 140.5              |
|Planetary Computer| 1133           | 12.2               |

- Indicates* that queries are faster when working on a _terrabyte_ environment.
- Most importantly, this indicates that Earth Search archive has some missing scenes.
- Also Earth Search STAC definitions are sometimes faulty (especially Sentinel-2 L1C) and as a developer of this package, I prefer working with _terrabyte_ API.

\* Queries are iterated 10 times per each service and the average time per run is calculated (timeit module).

## How to cite:
paper to be published <3

<br><br>
Contact: https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/