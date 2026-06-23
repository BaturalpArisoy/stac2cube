#!/bin/bash
# Change to the directory where this script is located
cd "$(dirname "$0")" || exit
# Submit the SLURM job using the shared GPU setup one level up
sbatch ../config_gpu.cmd
