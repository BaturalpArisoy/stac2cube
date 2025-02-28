#!/bin/bash
#SBATCH --job-name=stac_data
#SBATCH --output=../log/cloud_masking_output.%j.out
#SBATCH --error=../log/cloud_masking_output.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=baturalp.arisoy@uni-wuerzburg.de
#SBATCH --account=pr94no-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=512GB
#SBATCH --time=0-08:00:00 
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute

module load micromamba
micromamba run -n stac2ardcube python slurm_run.py