#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-ml1m}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

ITERS=${ITERS:-3}
BETA=${BETA:-0.1}
SFT_EPOCHS=${SFT_EPOCHS:-3}
DPO_EPOCHS=${DPO_EPOCHS:-1}
LR=${LR:-1e-4}
DPO_LR=${DPO_LR:-2e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-320}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}

INIT_SFT=${INIT_SFT:-auto}
SFT_CKPT=${SFT_CKPT:-}
PLAY_SAMPLES=${PLAY_SAMPLES:-4096}
GEN_BEAMS=${GEN_BEAMS:-4}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-48}

GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-0}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-20}
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

if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
  echo "not found: ${DATA_DIR}/train.jsonl" >&2
  echo "Run: ${PYTHON} preprocess.py --dataset ${DATASET}" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --iters "${ITERS}"
  --beta "${BETA}"
  --sft_epochs "${SFT_EPOCHS}"
  --dpo_epochs "${DPO_EPOCHS}"
  --lr "${LR}"
  --dpo_lr "${DPO_LR}"
  --batch_size "${BATCH_SIZE}"
  --dpo_batch_size "${DPO_BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --init_sft "${INIT_SFT}"
  --play_samples "${PLAY_SAMPLES}"
  --gen_beams "${GEN_BEAMS}"
  --gen_batch_size "${GEN_BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
)
if [[ "${GRAD_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--gradient_checkpointing)
fi
if [[ "${MAX_TRAIN_SAMPLES}" != "0" ]]; then
  ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${SFT_CKPT}" ]]; then
  ARGS+=(--sft_ckpt "${SFT_CKPT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

echo ">>> dataset=${DATASET} gpu=${GPU} iters=${ITERS} beta=${BETA} init_sft=${INIT_SFT}"
echo ">>> SPRec -> baseline/SPRec/output/checkpoints/${DATASET}"
echo ">>> ${PYTHON} baseline/SPRec/train.py ${ARGS[*]}"
exec ${PYTHON} baseline/SPRec/train.py "${ARGS[@]}"
