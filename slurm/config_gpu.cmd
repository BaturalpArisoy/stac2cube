#!/bin/bash
#SBATCH --job-name=stac2cube
#SBATCH --output=../log/stac2cube_output.%j.out
#SBATCH --error=../log/stac2cube_error.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=<e-mail>
#SBATCH --account=<account>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=0-10:00:00
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu

module load micromamba

# Local package
#micromamba run -n stac2cube python slurm_run.py

# Shared DSS package
micromamba run -p /dss/.../envs/stac2cube python slurm_run.py
