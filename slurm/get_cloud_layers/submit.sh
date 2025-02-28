#!/bin/bash
# Change to the directory where this script is located
cd "$(dirname "$0")" || exit
# Submit the SLURM job
sbatch get_cloud_layers.cmd