#!/usr/bin/env bash
# node_setup.sh: the vLLM environment for the replication (PLAN_replication.md), on the node, from the repo root.
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
    tiktoken sentencepiece wandb pyyaml -e ../mats-kit
"$VENV/bin/python" - <<'EOF'
import torch, transformers, vllm
print(f"VLLM_ENV ok: vllm {vllm.__version__} torch {torch.__version__} (cuda {torch.version.cuda}) "
      f"transformers {transformers.__version__} gpus {torch.cuda.device_count()}")
assert transformers.__version__ == "5.17.0", transformers.__version__
EOF
nvidia-smi topo -m | head -12
df -h "${HF_HOME:-/root/hf_cache}" 2>/dev/null || df -h /root
