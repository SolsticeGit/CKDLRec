#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-ml1m}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

SEED=${SEED:-42}
SFT_CKPT=${SFT_CKPT:-}
OUTPUT_DIR=${OUTPUT_DIR:-}

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false

SFT_CHECK="${SFT_CKPT:-baseline/sft/output/checkpoints/${DATASET}/best}"
if [[ ! -f "${SFT_CHECK}/adapter_config.json" && ! -f "${SFT_CHECK}/config.json" ]]; then
  echo "not found: ${SFT_CHECK}" >&2
  echo "DecodingMatters reuses SFT. Run: DATASET=${DATASET} bash baseline/sft/train.sh" >&2
  exit 1
fi

FLOWER_SASREC="baseline/Flower/output/checkpoints/${DATASET}/sasrec.pt"

ARGS=(
  --dataset "${DATASET}"
  --seed "${SEED}"
)
if [[ -n "${SFT_CKPT}" ]]; then
  ARGS+=(--sft_ckpt "${SFT_CKPT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ -f "${FLOWER_SASREC}" ]]; then
  echo ">>> dataset=${DATASET} reuse Flower SASRec ${FLOWER_SASREC}"
else
  echo ">>> dataset=${DATASET} gpu=${GPU} Flower SASRec missing; will train local SASRec"
fi
echo ">>> ${PYTHON} baseline/DecodingMatters/train.py ${ARGS[*]}"
exec ${PYTHON} baseline/DecodingMatters/train.py "${ARGS[@]}"
