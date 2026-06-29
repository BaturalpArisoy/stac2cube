<img src="assets/stac2cube_logo.png" alt="stac2cube logo" width="300">

# stac2cube <br> STACs to Analysis-Ready Data Cubes

[![Preprint DOI](https://img.shields.io/badge/DOI-10.5194%2Fegusphere--2026--619-blue)](https://doi.org/10.5194/egusphere-2026-619)
[![DOI](https://zenodo.org/badge/940592574.svg)](https://zenodo.org/records/18495808)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)


-   If you use **stac2cube** in your research, you are kindly asked to cite it. Thank you! <br> See: [Citation](#citation)
-   Free software: Apache 2.0
-   This software is designed to function on any local-machine and also HPC system using SLURM jobs.

## Table of Contents
- [Feature Overview](#feature-overview)
- [Installation](#installation)
- [How to run](#how-to-run)
- [How to run on HPC](#how-to-run-on-hpc)
- [Access and Licensing Details for STAC Catalogs](#access-and-licensing-details-for-stac-catalogs)
- [Method References](#method-references)
- [Citation](#citation)



## Feature Overview
**stac2cube** converts SpatioTemporal Asset Catalogs (STAC) into Analysis-Ready Data (ARD) cubes for efficient Earth Observation (EO) processing.

For Sentinel-2, the ARD cubes are built with three main components:

-   **Cloud masking** based on user-defined thresholds. This lets users control how strict cloud detection should be and export multiple cloud-masked cubes. Traditional options like filtering by max_cc (STAC metadata) and masking with the Scene Classification Layer (SCL) are also supported for faster processing.

-   **Co-registration** to reduce scene-to-scene X/Y misalignment (often around 1-2 pixels). Small sub-pixel shifts (below 10 m) can still remain.

-   **Super-resolution** of both 10-meters and 20-meters bands to 2.5 m.

The result is a data cube that is cloud-masked with customizable thresholds, spatially aligned across time, and available at higher spatial resolution. Details about the underlying algorithms and how to cite the used third-party tools can be found in the [Examples](#examples) section.

### Below is an example of 2 animations showing before and after ARD cube generation.

<div align="center">
  <h2>Before (Initial Data Cube)</h2>
  <a href="https://github.com/user-attachments/assets/d6458ba1-6112-4127-899e-9fa06ce58772">
    <img src="https://github.com/user-attachments/assets/d6458ba1-6112-4127-899e-9fa06ce58772" alt="Initial Data Cube">
  </a>
</div>

<br>

<div align="center">
  <h2>After (Co-registered and Super Resolved Data Cube)</h2>
  <a href="https://github.com/user-attachments/assets/529402c7-4ecc-4344-b63b-409aee94e3c9">
    <img src="https://github.com/user-attachments/assets/529402c7-4ecc-4344-b63b-409aee94e3c9" alt="Co-registered and Super-resolved Data Cube">
  </a>
</div>

<br><br>



## Installation
Installation is possible with package managers like [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) & [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install).<br>

Choose one of the two methods below:

---

### Option A - Manual Installation (recommended)

#### Step 1: Clone the repository to your current working directory

    $ git clone https://github.com/BaturalpArisoy/stac2cube.git

If git is not available for you, download and unzip the file: https://github.com/BaturalpArisoy/stac2cube/archive/refs/heads/main.zip

#### Step 2: Change directory to cloned stac2cube folder

    $ cd "path/to/stac2cube/"

*environment.yml* file should be present in this path, please double check.

#### Step 3: Install stac2cube via Micromamba or Anaconda Prompt (this might take a while!)

##### a) LINUX

    $ micromamba env create -n stac2cube -f environment.yml

##### b1) WINDOWS Micromamba

    $ micromamba env create -n stac2cube -f environment.yml; micromamba install -n stac2cube -c conda-forge vs2015_runtime

##### b2) WINDOWS Anaconda Prompt

    $ conda env create -n stac2cube -f environment.yml && conda activate stac2cube && conda install -c conda-forge vs2015_runtime

---

### Option B - Automated Installation via Claude Code

**Important**: AI agents can easily make mistakes and should not be trusted fully. This prompt is for installation only and does not produce any data or another result.<br><br>
If you have [Claude Code](https://claude.ai/code) available, the entire installation can be handled automatically, no manual commands needed.

1. Open the `stac2cube` folder in VS Code
2. Open Claude Code in the terminal (or any Claude agent with terminal access)
3. Open [INSTALL_PROMPT.md](INSTALL_PROMPT.md), copy the full prompt inside the code block, and paste it into Claude Code
4. Claude will detect your OS and package manager, then ask you **at most two questions**:
   - Which package manager to use (only if both Micromamba and conda are found)
   - Whether to keep or replace an existing `stac2cube` environment (only if one already exists)
5. Everything else runs automatically. Claude verifies the install before reporting success.

---

## How to run
### Friendly Interactive User Interface on Jupyter Notebook (recommended):
The **recommended** way to run stac2cube is through the interactive GUI tools in the [User Interface Tools](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive/User_Interface_Tools.ipynb) notebook. It bundles the full workflow in one place, requires no manual coding. <br> Just set the parameters and enjoy your coffee while your data cube is being built :) <br>

1) Data Cube Builder (see example below)
2) Data Cube Editor
3) Analysis Ready Data Cube Tools (Probabilistic Cloud Masking, Co-registration and Super-resolution)<br><br>

<img src="assets/data_cube_builder_GUI.png" alt="gui_editor">

### Step-by-step Interactive Notebooks
For a more detailed walkthrough of stac2cube features, including background, processing steps, and storage, see the well-documented notebooks in the [tutorials folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive/tutorials).

Each step is documented by the numbers and the general explanation is given below:

1. **Initial Data Cube**
    - Collects images from STAC catalogs for the selected mission based on users parameters.
    - Generates multi-dimensional data cubes, suitable for time-series.
    - The data cubes can be **updated** anytime without generating them from the scratch.
    - Available missions: **Sentinel-2 L2A, Sentinel-2 L1C, Sentinel-1 RTC, Landsat C2 L2, COP DEM Glo-30 (single time)**
2. **Cloud Mask Data Cube**
    - The result contains cloud probability maps and user defined binary cloud mask layers of time-series.
    - When selected, clouds from the initial data cube are automatically masked out.
    - Can be updated anytime.
3. **Co-register Data Cube**
    - Fix the global X/Y shift between consecutive Sentinel-2 items.
    - IMPORTANT: Please read notes in the notebook for better quality results.
4. **Super-resolve Data Cube**
    - Super resolves both 10-meters and 20-meters bands to 2.5-meters. ["blue", "green", "red", "nir", "nir08", "rededge1", "rededge2", "rededge3", "swir16", "swir22"] for the entire Sentinel-2 data cube time-series.


## How to run on HPC
A documentation file on how to use stac2cube features on terrabyte's HPC for compute-intensive processes and for faster processing time can be found in the [slurm folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/slurm). Reading step by step is actually pretty simple.

## Access and Licensing Details for STAC Catalogs

### Access to STAC Catalogs

- **Important**: _terrabyte_ STAC catalogs can be only computed when working on a _terrabyte_ environment.<br>
- However, stac2cube package is designed to work on both local-machine without _terrabyte_ connection and within _terrabyte_ HPC environment.<br>
- The user can select the desired STAC source (also in user interface).<br>
- Note that stac2cube package **can not** guarantee unlimited access to these open-access data catalogs in the future!

### STAC Catalog Licenses

| Provider   | Service           | STAC API                                        | License                                                                                      | Open-Access | Requires Credentials |
|------------|-------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------|-------------|----------------------|
| DLR        | terrabyte         | https://stac.terrabyte.lrz.de/public/api/       | MIT License Copyright (c) 2024 Deutsches Zentrum für Luft- und Raumfahrt e.V.                | No          | Yes                  |
| Element 84 | Earth Search      | https://earth-search.aws.element84.com/v1/      | Apache License 2.0                                                                           | Yes         | No                   |
| Microsoft  | Planetary Computer| https://planetarycomputer.microsoft.com/api/stac/v1    | MIT License Copyright (c) Microsoft Corporation.                                             | Yes         | No                   |
| ESA        | Copernicus Data Space Ecosystem | https://stac.dataspace.copernicus.eu/v1               | Copernicus data - free, full and open access (Legal notice on the use of Copernicus data)   | Yes         | Yes                  |


## Method References

1. **Cloud Mask Data Cube** applies **[s2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector)** by Sentinel Hub - CC-BY-SA-4.0 license.
2. **Co-register Data Cube** applies **[AROSICS](https://github.com/GFZ/arosics)** by Daniel Scheffler - Apache-2.0 license.
    
    Daniel Scheffler. (2017, July 3). AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data (Version 0.12.1). Zenodo. https://doi.org/10.5281/zenodo.3742909
    
3. **Super-resolve Data Cube** applies **[SEN2SR](https://github.com/ESAOpenSR/SEN2SR)** by Aybar et al. - CC0-1.0 license.

    Aybar, C., Contreras, J., Donike, S., Portalés-Julià, E., Mateo-García, G., & Gómez-Chova, L. (2026). A radiometrically and spatially consistent super-resolution framework for Sentinel-2. Remote Sensing of Environment, 334, 115222. https://doi.org/10.1016/j.rse.2025.115222
    

## Citation

### Method paper

    Arisoy, B., Betz, F., Stauch, G., Klein, D., Dech, S., and Ullmann, T.: Scalable Earth Observation Data Cubes for Advanced Analytics of Dynamic Earth Surface Processes: An Open-Source Package for Customized Processing of Sentinel-2 Data on HPCs and Beyond, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-619, 2026.

### Software
**Please include the exact version**

    Arisoy, B., Betz, F., Stauch, G., Klein, D., Dech, S., & Ullmann, T. (2025). stac2cube (Version 1.4.0). Zenodo. https://zenodo.org/records/18495808


## Contact 
https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/