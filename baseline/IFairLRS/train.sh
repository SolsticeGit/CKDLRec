#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-cds}
GROUP=${GROUP:-pop}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-320}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

FULL_FINETUNE=${FULL_FINETUNE:-0}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-0}

EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-20}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}

OUTPUT_DIR=${OUTPUT_DIR:-}

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false

case "${DATASET}" in
  ml1m)  DATA_DIR="data/MovieLens1M/processed" ;;
  cds)   DATA_DIR="data/CDs_and_Vinyl/processed" ;;
  toys)  DATA_DIR="data/Toys_and_Games/processed" ;;
  movies) DATA_DIR="data/Movies_and_TV/processed" ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

case "${GROUP}" in
  pop|category) ;;
  *) echo "unknown group: ${GROUP} (pop | category)" >&2; exit 1 ;;
esac

if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
  echo "not found: ${DATA_DIR}/train.jsonl" >&2
  echo "Run: ${PYTHON} preprocess.py --dataset ${DATASET}" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --group "${GROUP}"
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
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

echo ">>> dataset=${DATASET} group=${GROUP} gpu=${GPU} epochs=${EPOCHS} effective_batch=$((BATCH_SIZE * GRAD_ACCUM))"
echo ">>> ${TUNE_MODE} IFairLRS IPW -> baseline/IFairLRS/output/checkpoints/${DATASET}/${GROUP}"
echo ">>> ${PYTHON} baseline/IFairLRS/train.py ${ARGS[*]}"
exec ${PYTHON} baseline/IFairLRS/train.py "${ARGS[@]}"
