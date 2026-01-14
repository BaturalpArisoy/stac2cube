<img src="assets/stac2cube_logo.png" alt="stac2cube logo" width="300">

# stac2cube <br> Spatio-Temporal Asset Catalogs To Analysis-Ready Data Cubes
**stac2cube** converts STAC catalogs into Analysis-Ready Data Cubes for efficient Earth Observation (EO) processing. This tool is designed to work both on any local-machine and HPC system by _terrabyte_. We recommend using *MobaXTerm* for accessing the _terrabyte_ login node and editing files.

## Feature Overview
- **stac2cube.get_stac_layers**
    - Collects images from STAC catalogs for the selected mission based on users parameters.
    - Automatically preprocess spectral/radar values based on specifications of the selected mission.
    - Generates multi-dimensional data cubes, suitable for time-series.
    - The data cubes can be updated anytime without generating them from the scratch.
    - Available missions: **Sentinel-2 L2A, Sentinel-2 L1C, Sentinel-1 RTC, Landsat C2 L2, COP DEM Glo-30 (single time)**
- **stac2cube.get_cloud_layers**
    - Collects images from Sentinel-2 L1C to automatically apply s2cloudless cloud probability algorithm on data cube structure.
    - The result contains cloud probability maps and user defined binary cloud mask layers.
    - When selected, clouds from the generated data cube are automatically masked out.
    - Can be updated anytime.
- **stac2cube.coregister_cube**
    - Applies coregistration algorithm on Sentinel-2 data cubes.
    - AROSICS package provides the coregistration algorithm<br>
    Daniel Scheffler. (2017, July 3). AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data (Version 0.12.1). Zenodo. https://doi.org/10.5281/zenodo.3742909
    - Fix the global X/Y shift between consecutive Sentinel-2 items.
- **stac2cube.super_resolve_cube**
    - Applies super-resolution algorithm on Sentinel-2 data cubes.
    - SEN2SR package provides DNN based super-resolution algorithm<br>
    Aybar, Cesar and Contreras, Julio and Donike, Simon and Portalés-Julià, Enrique and Mateo-García, Gonzalo and Gómez-Chova, Luis, A Radiometrically and Spatially Consistent Super-Resolution Framework for Sentinel-2. Available at SSRN: https://ssrn.com/abstract=5247739 or https://dx.doi.org/10.2139/ssrn.5247739
    - Currently super resolve 10-meters RGBN bands to 2.5-meters (soon 20-meters bands will be also super-resolved to 2.5-meters).


## Installation
Installation is possible with a package manager like Micromamba & Anaconda.<br><br>
**IMPORTANT NOTE**: The current installation structure works on both Linux and Windows environement for *get_stac_layers*, *get_cloud_layers* and *coregister_scenes*, however *superresolve_cube* might potentially not work on Windows environement. Soon will be a major installation adjustment with updated files and the super resolution issue on Windows should be resolved. Until then, please work on a Linux setup to work with super resolution tasks. <br>
Thank you!

### 1) Change directory to where environment.yml file is located 
    $ cd "path/to/stac2cube/"

### 2) Install stac2cube via e.g. Micromamba
#### a) LINUX
    $ micromamba env create -n stac2cube2 -f environment.yml
#### b) WINDOWS
    $ micromamba env create -n stac2cube -f environment.yml; micromamba install -n stac2cube -c conda-forge vs2015_runtime

---

## Examples
Jupyter notebooks on how to use stac2cube features and how to process data cube structure can be found in the [interactive folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/interactive).

## How to run on HPC (terrabyte users only)
A documentation file on how to use stac2cube features on terrabyte's HPC for compute-intensive processes and for faster processing time can be found in the [slurm folder](https://github.com/BaturalpArisoy/stac2cube/tree/main/slurm).

## What's upcoming?
- [x] Sentinel-2 co-registration
- [ ] Merge Landsat TM and OLI missions for terrabyte catalogues.
- [ ] Batch processing tools for all the steps (under development).
- [x] Caching mechanism to automatically update the missing scenes: get_stac_layers, get_cloud_layers
- [x] More advanced interactive tools for better experience
- [x] Sentinel-1: Orbit-state selection: Ascending/Descending
- [ ] Sentinel-1: Automatic preprocessing, e.g. SNAP tools. OR replace with NRB
- [ ] Add new spectral indices: EVI, Built-up Index (More upon request!)
- [ ] Add SLURM job array to submit multiple json files at once (good for enourmous areas; "divide and conquer")
- [ ] Silent parameter for get_stac_layers that will automatically switch to terrabyte STAC catalogs when run on HPC
- [ ] Import bbox list with projected coords: proj to geographic transformation (Under development)
- [x] Improvements for get_cloud_layers function: mask calculation function, mask l2a data directly
- [ ] Cloud shadow detection and masking for Sentinel-2
- [x] Native cloud masking for Landsat and Sentinel-2 scenes (Under development)
- [x] Switch python package setup from setup.py to pyproject.toml: enables uv install besides pip 
- [x] Quite mode
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
| Microsoft  | Planetary Computer| https://planetarycomputer.microsoft.com/api/stac/v1    | MIT License Copyright (c) Microsoft Corporation.                                             | Yes         | No          |

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
to be announced

<br><br>
Contact: https://www.geographie.uni-wuerzburg.de/en/earthobservation/staff/baturalp-arisoy/