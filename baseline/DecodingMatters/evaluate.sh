#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-ml1m}
PYTHON=${PYTHON:-python}

PREDS=${PREDS:-}
OUTPUT=${OUTPUT:-}

case "${DATASET}" in
  ml1m|cds|toys|movies) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

PREDS_CHECK="${PREDS:-baseline/DecodingMatters/output/results/${DATASET}/preds.jsonl}"
if [[ ! -f "${PREDS_CHECK}" ]]; then
  echo "not found: ${PREDS_CHECK}" >&2
  echo "Run: bash baseline/DecodingMatters/inference.sh" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --preds "${PREDS_CHECK}"
)
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

echo ">>> dataset=${DATASET} preds=${PREDS_CHECK}"
echo ">>> ${PYTHON} evaluate.py ${ARGS[*]}"
exec ${PYTHON} evaluate.py "${ARGS[@]}"
