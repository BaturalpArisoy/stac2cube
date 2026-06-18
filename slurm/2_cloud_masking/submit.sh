#!/bin/bash
# Change to the directory where this script is located
cd "$(dirname "$0")" || exit
# Submit the SLURM job using the shared setup one level up
sbatch ../slurm_setup.cmd
