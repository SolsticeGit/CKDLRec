#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-ml1m}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-320}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-48}
NUM_BEAMS=${NUM_BEAMS:-10}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
MAX_SAMPLES=${MAX_SAMPLES:-0}

BETA=${BETA:-0.2}

CKPT=${CKPT:-}
SASREC=${SASREC:-}
OUTPUT=${OUTPUT:-}

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false

case "${DATASET}" in
  ml1m)  DATA_DIR="data/MovieLens1M/processed" ;;
  cds)   DATA_DIR="data/CDs_and_Vinyl/processed" ;;
  toys)  DATA_DIR="data/Toys_and_Games/processed" ;;
  movies) DATA_DIR="data/Movies_and_TV/processed" ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

if [[ ! -f "${DATA_DIR}/test.jsonl" ]]; then
  echo "not found: ${DATA_DIR}/test.jsonl" >&2
  echo "Run: ${PYTHON} preprocess.py --dataset ${DATASET}" >&2
  exit 1
fi

CKPT_CHECK="${CKPT:-baseline/D2LR/output/checkpoints/${DATASET}/best}"
if [[ ! -d "${CKPT_CHECK}" ]]; then
  echo "not found: ${CKPT_CHECK}" >&2
  echo "Run: bash baseline/D2LR/train.sh" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --batch_size "${BATCH_SIZE}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --num_beams "${NUM_BEAMS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --beta "${BETA}"
)
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${CKPT}" ]]; then
  ARGS+=(--ckpt "${CKPT}")
fi
if [[ -n "${SASREC}" ]]; then
  ARGS+=(--sasrec "${SASREC}")
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

echo ">>> dataset=${DATASET} gpu=${GPU} beams=${NUM_BEAMS} beta=${BETA}"
echo ">>> ckpt=${CKPT_CHECK}"
echo ">>> ${PYTHON} baseline/D2LR/inference.py ${ARGS[*]}"
exec ${PYTHON} baseline/D2LR/inference.py "${ARGS[@]}"
