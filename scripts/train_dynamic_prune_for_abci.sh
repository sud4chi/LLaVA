#!/usr/bin/env bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -P gah51624
#PBS -N dynamic_prune
#PBS -j oe
#PBS -k oe

set -euo pipefail

DEFAULT_VISION_TOKEN_ROOT="/groups/gah51624/yasuda/vision_token"
VISION_TOKEN_ROOT="${VISION_TOKEN_ROOT:-${DEFAULT_VISION_TOKEN_ROOT}}"

# PBS runs a spool copy of this script, so BASH_SOURCE does not necessarily point
# into the repository. Prefer the ABCI workspace path, then PBS_O_WORKDIR when
# running as a batch job.
if [[ -z "${LLAVA_ROOT:-}" ]]; then
  for root_candidate in \
    "${VISION_TOKEN_ROOT}/LLaVA" \
    "${PBS_O_WORKDIR:-}" \
    "${PBS_O_WORKDIR:-}/LLaVA" \
    "${PBS_O_WORKDIR:-}/.." \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
    if [[ -n "${root_candidate}" && -f "${root_candidate}/llava/train/train_dynamic_prune.py" ]]; then
      LLAVA_ROOT="$(cd "${root_candidate}" && pwd)"
      break
    fi
  done
fi

if [[ -z "${LLAVA_ROOT:-}" || ! -f "${LLAVA_ROOT}/llava/train/train_dynamic_prune.py" ]]; then
  echo "Could not locate the LLaVA repository." >&2
  echo "Submit from the workspace/LLaVA directory, or use: qsub -v LLAVA_ROOT=/path/to/LLaVA $0" >&2
  exit 1
fi

WORKSPACE_ROOT="$(cd "${VISION_TOKEN_ROOT}" 2>/dev/null || cd "${LLAVA_ROOT}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${LLAVA_ROOT}/.venv_dynamic_prune_local}"

# ABCI's -k oe streams the PBS spool log to dynamic_prune.o<job-id> in the
# submission directory. Keep a second, predictably named log in the repository.
LOG_DIR="${LOG_DIR:-${LLAVA_ROOT}/logs/abci}"
mkdir -p "${LOG_DIR}"
JOB_ID="${PBS_JOBID:-local}"
JOB_ID="${JOB_ID//\//_}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${PBS_JOBNAME:-dynamic_prune}.${JOB_ID}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

START_SECONDS="${SECONDS}"
on_exit() {
  local status="$?"
  echo "[$(date --iso-8601=seconds)] job finished: status=${status} elapsed=$((SECONDS - START_SECONDS))s log=${LOG_FILE}"
  return "${status}"
}
trap on_exit EXIT

echo "[$(date --iso-8601=seconds)] job started"
echo "job_id=${PBS_JOBID:-local} job_name=${PBS_JOBNAME:-dynamic_prune} host=$(hostname)"
echo "pbs_workdir=${PBS_O_WORKDIR:-unset} llava_root=${LLAVA_ROOT}"
echo "log=${LOG_FILE}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "Virtual environment not found: ${VENV_PATH}" >&2
  echo "Run ${LLAVA_ROOT}/scripts/setup_dynamic_prune_env_for_abci.sh first." >&2
  exit 1
fi

export PATH="${VENV_PATH}/bin:${PATH}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # ABCI User Guide initializes Environment Modules from this file.
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
elif [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
  # shellcheck disable=SC1090
  source "${MODULESHOME}/init/bash"
fi
if type module >/dev/null 2>&1; then
  module load "${CUDA_MODULE:-cuda/12.1/12.1.1}"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${LLAVA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-liuhaotian/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-${WORKSPACE_ROOT}/D-prune_data/processed/llava/annotations/llava_instruct_150k_random10k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/D-prune_data/raw/llava/coco/train2017}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAVA_ROOT}/checkpoints/score_core_frontier_10k}"

[[ -f "${DATA_PATH}" ]] || { echo "DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -d "${IMAGE_FOLDER}" ]] || { echo "IMAGE_FOLDER not found: ${IMAGE_FOLDER}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

MIN_TOKENS="${MIN_TOKENS:-40}"
MAX_TOKENS="${MAX_TOKENS:-80}"
TARGET_AVG_TOKENS="${TARGET_AVG_TOKENS:-64}"
VALIDATION_SPLIT_RATIO="${VALIDATION_SPLIT_RATIO:-0.05}"
SCORE_ALPHA="${SCORE_ALPHA:-0.8}"
UTILITY_HIDDEN_SIZE="${UTILITY_HIDDEN_SIZE:-128}"

TRAIN_CMD=("${VENV_PATH}/bin/python" -m llava.train.train_dynamic_prune)
if (( NPROC_PER_NODE > 1 )); then
  TRAIN_CMD=("${VENV_PATH}/bin/torchrun" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT:-29501}" -m llava.train.train_dynamic_prune)
fi

cd "${LLAVA_ROOT}"
echo "python=${VENV_PATH}/bin/python"
echo "data=${DATA_PATH} images=${IMAGE_FOLDER} output=${OUTPUT_DIR}"
echo "nproc_per_node=${NPROC_PER_NODE} cuda_module=${CUDA_MODULE:-cuda/12.1/12.1.1}"
nvidia-smi || true

"${TRAIN_CMD[@]}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version v1 \
  --data_path "${DATA_PATH}" \
  --validation_split_ratio "${VALIDATION_SPLIT_RATIO}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_select_layer -2 \
  --mm_patch_merge_type flat \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --bits 16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_first_step True \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lazy_preprocess True \
  --report_to none \
  --score_method attention \
  --score_alpha "${SCORE_ALPHA}" \
  --utility_hidden_size "${UTILITY_HIDDEN_SIZE}" \
  --min_tokens "${MIN_TOKENS}" \
  --max_tokens "${MAX_TOKENS}" \
  --target_avg_tokens "${TARGET_AVG_TOKENS}"
