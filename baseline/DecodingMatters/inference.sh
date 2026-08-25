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

ALPHA=${ALPHA:-0.5}

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

CKPT_CHECK="${CKPT:-baseline/sft/output/checkpoints/${DATASET}/best}"
if [[ ! -f "${CKPT_CHECK}/adapter_config.json" && ! -f "${CKPT_CHECK}/config.json" ]]; then
  echo "not found: ${CKPT_CHECK}" >&2
  echo "DecodingMatters reuses SFT. Run: DATASET=${DATASET} bash baseline/sft/train.sh" >&2
  exit 1
fi

SASREC_CHECK="${SASREC:-}"
if [[ -z "${SASREC_CHECK}" ]]; then
  if [[ -f "baseline/Flower/output/checkpoints/${DATASET}/sasrec.pt" ]]; then
    SASREC_CHECK="baseline/Flower/output/checkpoints/${DATASET}/sasrec.pt"
  else
    SASREC_CHECK="baseline/DecodingMatters/output/checkpoints/${DATASET}/sasrec.pt"
  fi
fi
if [[ ! -f "${SASREC_CHECK}" ]]; then
  echo "SASRec not found (Flower or local DecodingMatters)" >&2
  echo "Run: DATASET=${DATASET} bash baseline/Flower/train.sh" >&2
  echo "or:   DATASET=${DATASET} bash baseline/DecodingMatters/train.sh" >&2
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
  --alpha "${ALPHA}"
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

echo ">>> dataset=${DATASET} gpu=${GPU} beams=${NUM_BEAMS} alpha=${ALPHA}"
echo ">>> ckpt=${CKPT_CHECK} sasrec=${SASREC_CHECK}"
echo ">>> ${PYTHON} baseline/DecodingMatters/inference.py ${ARGS[*]}"
exec ${PYTHON} baseline/DecodingMatters/inference.py "${ARGS[@]}"
