#!/usr/bin/env bash

set -euo pipefail

LLAVA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${LLAVA_ROOT}/.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${WORKSPACE_ROOT}/lmms-eval}"
VENV_PATH="${VENV_PATH:-${LLAVA_ROOT}/.venv_dynamic_prune_local}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi

if [[ -z "${UV_BIN}" ]]; then
  echo "uv was not found in PATH." >&2
  exit 1
fi
if [[ ! -f "${LMMS_EVAL_ROOT}/pyproject.toml" ]]; then
  echo "lmms-eval repository was not found: ${LMMS_EVAL_ROOT}" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/uv}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
mkdir -p "${UV_CACHE_DIR}"

"${UV_BIN}" venv --clear --python "${PYTHON_VERSION}" --seed "${VENV_PATH}"
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  -r "${LLAVA_ROOT}/requirements_dynamic_prune.txt"
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  --no-deps -e "${LLAVA_ROOT}"
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  --constraint "${LLAVA_ROOT}/requirements_dynamic_prune.txt" \
  -e "${LMMS_EVAL_ROOT}"

"${VENV_PATH}/bin/python" - <<'PY'
import numpy
import torch
import transformers
import loguru
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from lmms_eval.cli.dispatch import main as lmms_eval_main

print("dynamic-prune environment is ready")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loguru", loguru.__version__)
print("LLaVA model", LlavaLlamaForCausalLM.__name__)
print("lmms-eval CLI", lmms_eval_main.__name__)
PY

echo "Run: qsub ${LLAVA_ROOT}/scripts/train_dynamic_prune_for_abci.sh"
