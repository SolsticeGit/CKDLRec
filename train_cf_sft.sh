#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

DATASET=${DATASET:-ml1m}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-320}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
TAU=${TAU:-0.3}

FULL_FINETUNE=${FULL_FINETUNE:-0}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-0}

EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-20}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}

CF_DIR=${CF_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-}

CF_TAG="tau_${TAU}"

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false

CF_DEFAULT="outputs/cf_data/${DATASET}/${CF_TAG}"
case "${DATASET}" in
  ml1m|cds|toys|movies) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

CF_CHECK="${CF_DIR:-$CF_DEFAULT}/cf_train.jsonl"
if [[ ! -f "${CF_CHECK}" ]]; then
  echo "not found: ${CF_CHECK}" >&2
  echo "Run: ${PYTHON} build_item_emb.py --dataset ${DATASET} && ${PYTHON} build_cf_data.py --dataset ${DATASET}" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --tau "${TAU}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --preview_samples "${PREVIEW_SAMPLES}"
)
TUNE_MODE="LoRA finetune"
if [[ "${FULL_FINETUNE}" == "1" ]]; then
  ARGS+=(--full_finetune)
  TUNE_MODE="full finetune"
fi
if [[ "${GRAD_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--gradient_checkpointing)
fi
if [[ "${MAX_TRAIN_SAMPLES}" != "0" ]]; then
  ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${CF_DIR}" ]]; then
  ARGS+=(--cf_dir "${CF_DIR}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

echo ">>> dataset=${DATASET} gpu=${GPU} tau=${TAU} effective_batch=$((BATCH_SIZE * GRAD_ACCUM))"
echo ">>> ${TUNE_MODE} on history_cf -> checkpoints/cf_sft/${DATASET}/${CF_TAG}"
echo ">>> ${PYTHON} train_cf_sft.py ${ARGS[*]}"
exec ${PYTHON} train_cf_sft.py "${ARGS[@]}"
