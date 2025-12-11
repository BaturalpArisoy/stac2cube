#!/bin/bash
#SBATCH --job-name=stac_data
#SBATCH --output=../log/stac_xarray_output.%j.out
#SBATCH --error=../log/stac_xarray_error.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=baturalp.arisoy@uni-wuerzburg.de
#SBATCH --account=pr94no-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=512GB
#SBATCH --time=0-10:00:00 
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute

module load micromamba

# Local package
#micromamba run -n stac2cube python slurm_run.py

# Shared DSS package
micromamba run -p /dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/envs/stac2cube python slurm_run.py