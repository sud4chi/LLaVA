# SCoRe Core--Frontier Dynamic Pruning

This project should run dynamic-pruner training with a dedicated Python 3.10 virtualenv. The failed job `o.8079761` used `/usr/lib64/python3.9`, so it did not see the expected LLaVA dependencies. The existing `.venv_llava` also had `torch` but was missing packages such as `transformers`, and it contained NumPy 2.x, which is not appropriate for `torch==2.1.2`.

## Build the environment

Run these commands from the LLaVA root:

```bash
cd /gs/bs/hp190122/yasuda/vision_token/LLaVA

# Replace the incomplete old environment.
rm -rf .venv_llava

/home/2/ut05192/hp_bs/.pyenv/versions/3.10.16/bin/python -m venv .venv_llava
. .venv_llava/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_dynamic_prune.txt
python -m pip install -e . --no-deps
```

If the CUDA 12.1 PyTorch wheels are not suitable for the allocated node, change the first three lines in `requirements_dynamic_prune.txt` to the wheel set that matches the node CUDA stack.

## Verify

```bash
cd /gs/bs/hp190122/yasuda/vision_token/LLaVA
. .venv_llava/bin/activate

export PYTHONNOUSERSITE=1
export HF_HOME=/gs/bs/hp190122/yasuda/vision_token/LLaVA/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_DISABLE_XET=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys
import numpy
import torch
import transformers
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

print("python", sys.executable)
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("llava import", LlavaLlamaForCausalLM.__name__)
PY
```

Expected versions:

```text
numpy 1.26.4
torch 2.1.2+cu121
transformers 4.37.2
llava import LlavaLlamaForCausalLM
```

If an existing `.venv_llava` was created before `protobuf` was added, update it with:

```bash
cd /gs/bs/hp190122/yasuda/vision_token/LLaVA
. .venv_llava/bin/activate
python -m pip install protobuf==4.25.3
```

## Submit training

The training script now activates `.venv_llava` by default:

```bash
qsub /gs/bs/hp190122/yasuda/vision_token/LLaVA/scripts/train_dynamic_prune.sh
```

To use another environment, pass `VENV_PATH`:

```bash
qsub -v VENV_PATH=/path/to/venv /gs/bs/hp190122/yasuda/vision_token/LLaVA/scripts/train_dynamic_prune.sh
```

The default token budget is `min_tokens=40`, `max_tokens=80`, and
`target_avg_tokens=64`. Override it through environment variables passed to
the job script:

```bash
qsub -v MIN_TOKENS=40,MAX_TOKENS=80,TARGET_AVG_TOKENS=64 \
  /gs/bs/hp190122/yasuda/vision_token/LLaVA/scripts/train_dynamic_prune.sh
```

By default, a deterministic 5% split of `DATA_PATH` is reserved for threshold
calibration. To use a separate validation annotation file, invoke the Python
entry point directly with `--validation_data_path /path/to/validation.json`.
After utility-predictor training, the script selects one global strict-`>`
threshold whose dataset-wide average retained-token count is closest to
`target_avg_tokens`. The threshold is stored in
`dynamic_pruner_config.json` alongside `dynamic_pruner.bin`.

Each training sample produces one core-only sequence and one sequence for each
frontier candidate. These sequences are packed into a single LLM forward; the
vision encoder, projector, SCoRe ranking, and LLM are therefore each invoked
only once per sample/batch. This expands the effective LLM batch by
`1 + max_tokens - min_tokens`, so the job scripts default to a physical batch
size of one.

For inference, use the predictor checkpoint as `model_path` and the original
LLaVA checkpoint as `model_base` with `load_pretrained_model`. The loader
automatically attaches the utility predictor and restores its calibrated
threshold.

## Notes

- `PYTHONNOUSERSITE=1` prevents packages from `~/.local/lib/python*` leaking into the job.
- Hugging Face caches are placed under `/gs/bs/hp190122/yasuda/vision_token/LLaVA/.cache/huggingface`; `/home` is too small for LLaVA model shards.
- BLAS/tokenizer thread counts are capped to avoid per-user process/thread limits on login and compute nodes.
- The dynamic-prune script does not use DeepSpeed by default, so `deepspeed` is not required for this run.
- `flash-attn` is not required by `train_dynamic_prune.sh` as currently written.
