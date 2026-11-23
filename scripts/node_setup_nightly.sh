#!/usr/bin/env bash
# node_setup_nightly.sh: the vLLM nightly venv for Kimi K3.1 Pro (PLAN_replication2.md), on the node, from the repo root.
# Called by node_setup.sh; on its own: bash scripts/node_setup_nightly.sh. It needs vLLM >= 0.35, a nightly on
# 2026-11-23, in its own venv so the other models keep the engine they were scored with in round 1 (0.33).
# transformers isn't pinned here (the nightly sets its own range): Kimi's tokenizer is tiktoken remote code, and
# tests/test_replication_new_models.py, run in this venv at the end, checks that it renders and tokenizes the prompts as
# in the project's env.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
NIGHTLY_VENV=${VLLM_NIGHTLY_ENV:-/root/vllm-nightly}
[ -x "$NIGHTLY_VENV/bin/python" ] || uv venv --python 3.12 "$NIGHTLY_VENV"
uv pip install --python "$NIGHTLY_VENV/bin/python" --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match
uv pip install --python "$NIGHTLY_VENV/bin/python" "huggingface_hub[hf_xet]" tiktoken sentencepiece "wandb==0.30.0" \
    pyyaml pytest -e ../mats-kit
uv pip install --python "$NIGHTLY_VENV/bin/python" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*"
CU13N=$("$NIGHTLY_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
[ -e "$CU13N/lib64" ] || ln -s lib "$CU13N/lib64"
[ -e "$CU13N/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CU13N/lib/libcudart.so"
"$NIGHTLY_VENV/bin/python" - <<'PY'
import torch, transformers, vllm
print(f"VLLM_NIGHTLY ok: vllm {vllm.__version__} torch {torch.__version__} (cuda {torch.version.cuda}) "
  f"transformers {transformers.__version__}")
major, minor = (int(x) for x in vllm.__version__.split(".")[:2])
assert (major, minor) >= (0, 35), f"Kimi K3.1 Pro needs vLLM >= 0.35, got {vllm.__version__}"
PY
"$NIGHTLY_VENV/bin/python" -m pytest -q tests/test_replication_new_models.py -k "kimi" -p no:cacheprovider
