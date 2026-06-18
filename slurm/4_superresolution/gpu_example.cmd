# ============================================================
# GPU example - for super-resolution
# ============================================================
# This is NOT a runnable script. Super-resolution (SEN2SRLite)
# runs much faster on a GPU. To send the job to a GPU node,
# copy the lines below into ../slurm_setup.cmd, REPLACING the
# matching CPU lines. Leave everything else (job-name, mail,
# account, output, module load, micromamba run) unchanged.
#
# terrabyte GPU partition (cluster stays the same: hpda2):
#   NVIDIA A100 80GB - 4 GPUs / 48 cores / 1024 GB per node
#   https://docs.terrabyte.lrz.de/services/terrabyte-hpc/introduction/

# --- replace the CPU partition line ---
# from:  #SBATCH --partition=hpda2_compute
# to:
#SBATCH --partition=hpda2_compute_gpu

# --- add this line to request a GPU ---
#SBATCH --gres=gpu:1

# --- (optional) right-size to ~one GPU's share of a node ---
# from:  #SBATCH --cpus-per-task=40
# from:  #SBATCH --mem=512GB
# to:
#SBATCH --cpus-per-task=12
#SBATCH --mem=256GB

# Note: --clusters=hpda2 stays the same (the GPU partition is on
# the same cluster), so that line does not need to change.
