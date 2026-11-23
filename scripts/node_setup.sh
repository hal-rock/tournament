#!/usr/bin/env bash
# node_setup.sh: the vLLM environments for the replication (PLAN_replication.md, PLAN_replication2.md), on the node,
# from the repo root.
#   bash scripts/node_setup.sh
# vLLM gets its own venv (/root/vllm-env) rather than a dependency group of this project: vLLM 0.33 caps numpy below the
# project's locked 2.5.3, and one lock means one numpy for everything. transformers is pinned to the project's version
# (5.17.0, inside vLLM's range), so prompts tokenize exactly as in the tests and the HF runs. score_replication.py needs only
# prompts.py / replication.py from the repo plus these packages; the project's own env (uv sync) runs everything else.
set -euo pipefail
VENV=${VLLM_ENV:-/root/vllm-env}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "vllm==0.33.0" "transformers==5.17.0" "huggingface_hub[hf_xet]" \
    tiktoken sentencepiece "wandb==0.30.0" pyyaml -e ../mats-kit
# FP8 models: vLLM JIT-compiles DeepGEMM with an nvcc >= 12.9 (score_replication.py points CUDA_HOME here), and
# nvcc/crt/nvvm/cccl must match the 13.0 runtime headers torch pulls in; unpinned they resolve to newer ones (user-models).
uv pip install --python "$VENV/bin/python" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" \
    "nvidia-cuda-cccl==13.0.*"
# FlashInfer's JIT links with -L$CUDA_HOME/lib64 -lcudart, but the pip CUDA tree has lib/ with only libcudart.so.13:
# "ld: cannot find -lcudart" building fp4_quantization for B200 (DeepSeek V4 Flash, replication node 2026-11-11).
CU13=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
[ -e "$CU13/lib64" ] || ln -s lib "$CU13/lib64"
[ -e "$CU13/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CU13/lib/libcudart.so"
"$VENV/bin/python" - <<'EOF'
import torch, transformers, vllm
print(f"VLLM_ENV ok: vllm {vllm.__version__} torch {torch.__version__} (cuda {torch.version.cuda}) "
      f"transformers {transformers.__version__} gpus {torch.cuda.device_count()}")
assert transformers.__version__ == "5.17.0", transformers.__version__
EOF
# Round 2 (PLAN_replication2.md): Kimi K3.1 Pro needs vLLM >= 0.35, a nightly on 2026-11-23, in its own venv so the other
# models keep the engine they were scored with in round 1 (0.33). NIGHTLY=0 skips it. transformers isn't pinned here
# (the nightly sets its own range): Kimi's tokenizer is tiktoken remote code, and tests/test_replication_new_models.py,
# run in this venv below, checks that it renders and tokenizes the prompts as in the project's env.
NIGHTLY_VENV=${VLLM_NIGHTLY_ENV:-/root/vllm-nightly}
if [ "${NIGHTLY:-1}" = "1" ]; then
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
fi
nvidia-smi topo -m | head -12
df -h "${HF_HOME:-/root/hf_cache}" 2>/dev/null || df -h /root
