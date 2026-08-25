#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

DATASET=${DATASET:-ml1m}
PYTHON=${PYTHON:-python}

TAU=${TAU:-0.4}
ALPHA=${ALPHA:-0.4}
BETA=${BETA:-0.3}

PREDS=${PREDS:-}
OUTPUT=${OUTPUT:-}

RUN_TAG="tau_${TAU}_sft_${ALPHA}_adv_${BETA}"

case "${DATASET}" in
  ml1m|cds|toys|movies) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

PREDS_CHECK="${PREDS:-outputs/results/${DATASET}/${RUN_TAG}/preds.jsonl}"
if [[ ! -f "${PREDS_CHECK}" ]]; then
  echo "not found: ${PREDS_CHECK}" >&2
  echo "Run: bash inference.sh" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --tau "${TAU}"
  --alpha "${ALPHA}"
  --beta "${BETA}"
)
if [[ -n "${PREDS}" ]]; then
  ARGS+=(--preds "${PREDS}")
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

echo ">>> dataset=${DATASET} ${RUN_TAG} preds=${PREDS_CHECK}"
echo ">>> ${PYTHON} evaluate.py ${ARGS[*]}"
exec ${PYTHON} evaluate.py "${ARGS[@]}"
