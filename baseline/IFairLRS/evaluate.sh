#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

DATASET=${DATASET:-ml1m}
GROUP=${GROUP:-pop}
PYTHON=${PYTHON:-python}

PREDS=${PREDS:-}
OUTPUT=${OUTPUT:-}

case "${DATASET}" in
  ml1m|cds|toys|movies) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

case "${GROUP}" in
  pop|category) ;;
  *) echo "unknown group: ${GROUP} (pop | category)" >&2; exit 1 ;;
esac

PREDS_CHECK="${PREDS:-baseline/IFairLRS/output/results/${DATASET}/${GROUP}/preds.jsonl}"
if [[ ! -f "${PREDS_CHECK}" ]]; then
  echo "not found: ${PREDS_CHECK}" >&2
  echo "Run: bash baseline/IFairLRS/inference.sh" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --preds "${PREDS_CHECK}"
)
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

echo ">>> dataset=${DATASET} group=${GROUP} preds=${PREDS_CHECK}"
echo ">>> ${PYTHON} evaluate.py ${ARGS[*]}"
exec ${PYTHON} evaluate.py "${ARGS[@]}"
