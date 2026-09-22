#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod. Expects GITHUB_TOKEN (fine-grained,
# read-only on hal-rock/tournament) and WANDB_API_KEY in the environment, set
# via RunPod Secrets. Run with:  bash scripts/pod_setup.sh
# (or, before the repo exists on the pod, paste it from GitHub).
set -euo pipefail

: "${GITHUB_TOKEN:?set GITHUB_TOKEN (RunPod secret)}"
: "${WANDB_API_KEY:?set WANDB_API_KEY (RunPod secret)}"

# Keep everything on the persistent volume so it survives pod restarts.
export HF_HOME=/workspace/hf_cache
export UV_CACHE_DIR=/workspace/uv_cache
grep -q HF_HOME ~/.bashrc || cat >> ~/.bashrc <<'RC'
export HF_HOME=/workspace/hf_cache
export UV_CACHE_DIR=/workspace/uv_cache
export PATH="$HOME/.local/bin:$PATH"
RC

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Let git read the token from the environment on each clone/pull, so it is
# never written to disk.
git config --global credential.https://github.com.helper \
    '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'

cd /workspace
[ -d tournament ] || git clone https://github.com/hal-rock/tournament.git
cd tournament
git pull --ff-only
uv sync

uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Setup done. Next: cd /workspace/tournament && uv run python tournament.py configs/olmo3-32b.yaml"
