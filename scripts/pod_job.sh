#!/usr/bin/env bash
# Runs tournament configs on a RunPod pod, one after another, as in ../coinflip.
# The pod (runpod/pytorch image, startSsh, 22/tcp, a host driver with CUDA >= 13.0,
# env WANDB_API_KEY, HF_TOKEN and RUNPOD_API_KEY from RunPod secrets, and
# HF_HOME=/root/hf_cache on the container disk) gets this repo by rsync over SSH, so
# no GitHub token is needed. Then, on the pod:
#   cd /root/tournament && nohup bash scripts/pod_job.sh configs/models/a.yaml ... > job.log 2>&1 &
# Each config logs to logs/<config name>.log; a failed config is logged and skipped.
# A watchdog stops the pod after WATCHDOG_HOURS (default 10) regardless; terminate
# it when ALL_DONE appears.
set -uo pipefail
cd "$(dirname "$0")/.."
api="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
(sleep $(( ${WATCHDOG_HOURS:-10} * 3600 )); curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "$api/stop") &
echo "SELF_STOP_CHECK HTTP $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $RUNPOD_API_KEY" "$api")"
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group gpu || exit 1  # gpu: flash-linear-attention (Qwen 3.5), kernels (DeepSeek V4)
uv run python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('No usable GPU: the pod driver is probably too old for torch ' + torch.__version__ + ' (CUDA ' + str(torch.version.cuda) + ')')
print('CUDA OK:', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0), '| torch', torch.__version__)
" || exit 1
mkdir -p logs
for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    echo "[$(date -u +%FT%TZ)] JOB $cfg"
    uv run python tournament.py "$cfg" > "logs/$name.log" 2>&1
    status=$?
    echo "[$(date -u +%FT%TZ)] JOB_EXIT $cfg $status"
    rm -rf "$HF_HOME"/hub  # free disk between configs (blobs live in hub/blobs)
done
echo "[$(date -u +%FT%TZ)] ALL_DONE"
