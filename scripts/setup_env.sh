#!/bin/bash
# Create autogaze conda env and install deps following README instructions.
# Run once on the cluster node; /home is mounted so the env persists.
set -euo pipefail

CONDA_BASE=${CONDA_BASE:-/home/scratch.thannan_wwfo/anaconda3}
REPO_DIR=${REPO_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}

echo "[setup] CONDA_BASE: $CONDA_BASE"
echo "[setup] REPO_DIR:   $REPO_DIR"

# Create env (skip if already exists)
if "$CONDA_BASE/bin/conda" env list | grep -q "^autogaze "; then
    echo "[setup] conda env 'autogaze' already exists, skipping create."
else
    "$CONDA_BASE/bin/conda" create -n autogaze python=3.11 -y
    echo "[setup] Created conda env 'autogaze'."
fi

CONDA_PYTHON="$CONDA_BASE/envs/autogaze/bin/python"
CONDA_PIP="$CONDA_BASE/envs/autogaze/bin/pip"

# Install CUDA toolkit
echo "[setup] Installing cuda-toolkit=12.8 ..."
"$CONDA_BASE/bin/conda" install -n autogaze -c nvidia cuda-toolkit=12.8 -y

# Install uv and AutoGaze deps
echo "[setup] Installing uv ..."
"$CONDA_PIP" install uv

echo "[setup] Installing AutoGaze and dependencies ..."
"$CONDA_BASE/envs/autogaze/bin/uv" pip install -e "$REPO_DIR"

echo "[setup] Done. Python: $CONDA_PYTHON"
"$CONDA_PYTHON" -c "import torch; print('[setup] torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
