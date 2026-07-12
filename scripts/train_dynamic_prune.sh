#!/bin/sh
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=12:00:00
#$ -p -5
#$ -j y
#$ -o /gs/bs/hp190122/yasuda/vision_token/LLaVA/tsubame_logs/o.$JOB_ID
#$ -N dynamic_dprune

set -eu

LLAVA_ROOT="${LLAVA_ROOT:-/gs/bs/hp190122/yasuda/vision_token/LLaVA}"

cd "${LLAVA_ROOT}"
mkdir -p tsubame_logs

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
VENV_PATH="${VENV_PATH:-${LLAVA_ROOT}/.venv_llava}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "${VENV_PATH}/bin/activate"
else
  echo "VENV_PATH does not contain bin/activate: ${VENV_PATH}" >&2
  echo "Build the environment with docs/DynamicPruneEnv.md first." >&2
  exit 1
fi

export PYTHONPATH="${LLAVA_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${LLAVA_ROOT}/.cache/huggingface}"
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
DATA_PATH="${DATA_PATH:-/home/2/ut05192/hp_bs/datasets/D-prune_data/processed/llava/annotations/llava_instruct_150k_random10k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/2/ut05192/hp_bs/datasets/D-prune_data/raw/llava/coco/train2017}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAVA_ROOT}/checkpoints/dynamic_dprune_attention_10k}"
mkdir -p "${OUTPUT_DIR}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

DYNAMIC_PRUNE_TARGET_KEEP_RATIO="${DYNAMIC_PRUNE_TARGET_KEEP_RATIO:-0.25}"
DYNAMIC_PRUNE_MIN_KEEP="${DYNAMIC_PRUNE_MIN_KEEP:-64}"
DYNAMIC_PRUNE_BUDGET_LOSS_WEIGHT="${DYNAMIC_PRUNE_BUDGET_LOSS_WEIGHT:-0.01}"
DYNAMIC_PRUNE_TEMPERATURE="${DYNAMIC_PRUNE_TEMPERATURE:-0.1}"

TRAIN_CMD="python -m llava.train.train_dynamic_prune"
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  TRAIN_CMD="torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT:-29501} -m llava.train.train_dynamic_prune"
fi

${TRAIN_CMD} \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --version v1 \
  --data_path "${DATA_PATH}" \
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
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --dynamic_prune_input_type scores \
  --dynamic_prune_score_method attention \
  --dynamic_prune_target_keep_ratio "${DYNAMIC_PRUNE_TARGET_KEEP_RATIO}" \
  --dynamic_prune_min_keep "${DYNAMIC_PRUNE_MIN_KEEP}" \
  --dynamic_prune_budget_loss_weight "${DYNAMIC_PRUNE_BUDGET_LOSS_WEIGHT}" \
  --dynamic_prune_temperature "${DYNAMIC_PRUNE_TEMPERATURE}"
