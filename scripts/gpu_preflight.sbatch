#!/bin/bash
#SBATCH --job-name=cl-gpu-check
#SBATCH --partition=acd_u
#SBATCH --account=msun411
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/data/user/msun411/cl/Skill_Forgetting/runs/preflight/gpu-%j.log
set -euo pipefail
/share/anaconda3/envs/pytorch271/bin/python - <<'PY'
import torch
print('torch:', torch.__version__, flush=True)
print('cuda available:', torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('device count:', torch.cuda.device_count(), flush=True)
assert torch.cuda.device_count() == 1
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print('bf16 matmul:', y.shape, float(y.mean()), flush=True)
print('GPU_PREFLIGHT_PASSED', flush=True)
PY
