#!/usr/bin/env bash
# Create the Python environment on the GPU node.
#
#   bash scripts/setup_env.sh            # venv in .venv (default)
#   ENV_KIND=conda bash scripts/setup_env.sh
#   CUDA=cu121 bash scripts/setup_env.sh # pick the CUDA wheel tag (cu118, cu121, cu124, cpu)
#
# Afterwards:  source scripts/env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_KIND="${ENV_KIND:-venv}"
ENV_NAME="${ENV_NAME:-credit-graph-bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA="${CUDA:-cu121}"          # match `nvidia-smi` driver support on the node
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"

echo "[setup] kind=$ENV_KIND cuda=$CUDA torch=$TORCH_VERSION"

if [ "$ENV_KIND" = "conda" ]; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
  conda activate "$ENV_NAME"
else
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
echo "$ENV_KIND" > .env_kind

python -m pip install --upgrade pip wheel

# PyTorch first, pinned to the CUDA tag
if [ "$CUDA" = "cpu" ]; then
  pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
else
  pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA"
fi

# PyTorch Geometric and its sampling backend (pyg-lib / torch-sparse are optional; NeighborLoader
# works with torch-only but is faster with pyg-lib).
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA}.html" || \
  echo "[setup] optional pyg extensions not installed; NeighborLoader falls back to torch sampling"

pip install -r requirements.txt

# Register a kernel so papermill can find it by name
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# Free disk: caches are the largest avoidable consumer on a quota-limited home directory
pip cache purge || true
if [ "$ENV_KIND" = "conda" ]; then conda clean -y --all || true; fi

python - <<'EOF'
from src import device
print("[setup] device summary:", device.summary())
EOF
echo "[setup] done. Next: source scripts/env.sh && bash scripts/fetch_data.sh"
