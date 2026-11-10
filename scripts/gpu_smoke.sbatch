#!/bin/bash
#SBATCH --job-name=cl-sft-smoke
#SBATCH --partition=acd_u
#SBATCH --account=msun411
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/msun411/cl/Skill_Forgetting/runs/preflight/smoke-%j.log
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH=/data/user/msun411/cl/Skill_Forgetting/src
cd /data/user/msun411/cl/Skill_Forgetting
/data/user/msun411/cl-artifacts/envs/sft/bin/python -m skill_forgetting.experiments.smoke \
  --config configs/replay_shapley_pilot.json
