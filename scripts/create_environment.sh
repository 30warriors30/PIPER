#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="dual-watermark-blackwell"

cd "$PROJECT_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists. Remove it first for a clean rebuild:"
  echo "  conda env remove -n $ENV_NAME -y"
  exit 2
fi

conda env create -f environment.yml
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1
python -m pip check

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Compiled architectures:", torch.cuda.get_arch_list())
if torch.cuda.get_device_capability(0) == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError("PyTorch does not contain sm_120 kernels")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
z = x @ y
if not torch.isfinite(z).all():
    raise RuntimeError("CUDA matrix multiplication produced non-finite values")
print("Blackwell CUDA verification passed")
PY
