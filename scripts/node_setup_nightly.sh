#!/usr/bin/env bash
# node_setup_nightly.sh: the vLLM nightly venv for Kimi K3.1 Pro (PLAN_replication2.md), on the node, from the repo root.
# Called by node_setup.sh; on its own: bash scripts/node_setup_nightly.sh. It needs a vLLM nightly after 0.34 (the
# recipe's "0.35.0+ nightly"), in its own venv so the other models keep the engine they were scored with in round 1 (0.33).
# transformers isn't pinned here (the nightly sets its own range): Kimi's tokenizer is tiktoken remote code, and
# tests/test_replication_new_models.py, run in this venv at the end, checks that it renders and tokenizes the prompts as
# in the project's env.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
NIGHTLY_VENV=${VLLM_NIGHTLY_ENV:-/root/vllm-nightly}
[ -x "$NIGHTLY_VENV/bin/python" ] || uv venv --python 3.12 "$NIGHTLY_VENV"
# The nightly pinned by commit. Main after the 0.34 branch cut is versioned 0.34.0rc1.devN, which PEP 440 ranks below the
# 0.34.0 release, so an unpinned `--pre vllm` from the nightly index installs 0.34.0 instead (node, 2026-11-23). 0.34.0
# registers KimiK31ForConditionalGeneration, but vLLM's recipe asks for a later build (new Kimi K3 kernels since).
VLLM_NIGHTLY_COMMIT=${VLLM_NIGHTLY_COMMIT:-4623da99ff85b0f52d9493cfafe2d1f58ede67e4}
VLLM_NIGHTLY_WHEEL=${VLLM_NIGHTLY_WHEEL:-vllm-0.34.0rc1.dev1196%2Bg4623da99f-cp38-abi3-manylinux_2_28_x86_64.whl}
uv pip install --python "$NIGHTLY_VENV/bin/python" \
    "vllm @ https://wheels.vllm.ai/$VLLM_NIGHTLY_COMMIT/$VLLM_NIGHTLY_WHEEL"
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
assert "+g4623da99f" in vllm.__version__ or "dev" in vllm.__version__, f"not the pinned nightly: {vllm.__version__}"
from vllm.model_executor.models.registry import ModelRegistry
assert "KimiK31ForConditionalGeneration" in ModelRegistry.get_supported_archs()
PY
"$NIGHTLY_VENV/bin/python" -m pytest -q tests/test_replication_new_models.py -k "kimi" -p no:cacheprovider
