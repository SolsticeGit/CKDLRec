#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PYTHON=${PYTHON:-python}
TAU=${TAU:-0.4}
ALPHA=${ALPHA:-0.4}
BETA=${BETA:-0.2}
SKIP=${SKIP:-0}
INCLUDE_FULL=${INCLUDE_FULL:-1}
OUT_ROOT="ablation/output"
RUN_TAG="tau_${TAU}_sft_${ALPHA}_adv_${BETA}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS}"
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS=("${DATASET}")
else
  DATASETS=(cds toys movies)
fi

KNOWN=(teacher no_kd no_sft no_adv pop_only rand_ret)

discover_names() {
  local ds="$1" p name
  local dir="${OUT_ROOT}/results/${ds}"
  [[ -d "${dir}" ]] || return 0
  for p in "${dir}"/*; do
    [[ -d "${p}" ]] || continue
    name="$(basename "${p}")"
    case "${name}" in
      pop_only_teacher|rand_ret_teacher) continue ;;
    esac
    echo "${name}"
  done | sort
}

if [[ -n "${ABLATIONS:-}" ]]; then
  read -r -a ABLATIONS <<< "${ABLATIONS}"
else
  ABLATIONS=()
fi

skip_file() {
  [[ "${SKIP}" == "1" && -f "$1" ]]
}

n_ok=0
n_skip=0
n_miss=0

eval_preds() {
  local ds="$1" name="$2" preds="$3" metrics="$4"
  if [[ ! -f "${preds}" ]]; then
    echo ">>> miss  ${ds}/${name}  (${preds})"
    n_miss=$((n_miss + 1))
    return 0
  fi
  if skip_file "${metrics}"; then
    echo ">>> skip  ${ds}/${name}  (${metrics})"
    n_skip=$((n_skip + 1))
    return 0
  fi
  echo ">>> evaluate  ${ds}/${name}"
  mkdir -p "$(dirname "${metrics}")"
  DATASET="${ds}" PYTHON="${PYTHON}" \
    TAU="${TAU}" ALPHA="${ALPHA}" BETA="${BETA}" \
    PREDS="${preds}" OUTPUT="${metrics}" \
    bash evaluate.sh
  n_ok=$((n_ok + 1))
}

echo ">>> datasets=${DATASETS[*]}  skip=${SKIP}  include_full=${INCLUDE_FULL}"
echo ">>> tau=${TAU} α=${ALPHA} β=${BETA}"
echo ">>> out ${OUT_ROOT}/results/"

for ds in "${DATASETS[@]}"; do
  echo
  echo "########## dataset=${ds} ##########"
  case "${ds}" in
    ml1m|cds|toys|movies) ;;
    *) echo "unknown dataset: ${ds}" >&2; exit 1 ;;
  esac

  if [[ ${#ABLATIONS[@]} -gt 0 ]]; then
    names=("${ABLATIONS[@]}")
  else
    mapfile -t names < <(discover_names "${ds}")
    if [[ ${#names[@]} -eq 0 ]]; then
      names=("${KNOWN[@]}")
    fi
  fi

  if [[ "${INCLUDE_FULL}" == "1" ]]; then
    eval_preds "${ds}" full \
      "outputs/results/${ds}/${RUN_TAG}/preds.jsonl" \
      "outputs/results/${ds}/${RUN_TAG}/metrics.json"
  fi

  local_name=""
  for local_name in "${names[@]}"; do
    [[ -z "${local_name}" ]] && continue
    case "${local_name}" in
      pop_only_teacher|rand_ret_teacher) continue ;;
    esac
    eval_preds "${ds}" "${local_name}" \
      "${OUT_ROOT}/results/${ds}/${local_name}/preds.jsonl" \
      "${OUT_ROOT}/results/${ds}/${local_name}/metrics.json"
  done
done

SUM_ARGS=(
  --results_root "${OUT_ROOT}/results"
  --output "${OUT_ROOT}/ablation_summary.json"
  --txt "${OUT_ROOT}/ablation_summary.txt"
  --datasets "${DATASETS[@]}"
)
if [[ "${INCLUDE_FULL}" == "1" ]]; then
  SUM_ARGS+=(--full_root outputs/results --run_tag "${RUN_TAG}")
fi
if [[ -n "${K:-}" ]]; then
  SUM_ARGS+=(--k ${K})
fi

echo
echo ">>> summarize"
${PYTHON} ablation/summarize.py "${SUM_ARGS[@]}"

echo
echo ">>> ablation eval done  ok=${n_ok} skip=${n_skip} miss=${n_miss}"
if [[ "${n_ok}" -eq 0 && "${n_skip}" -eq 0 ]]; then
  echo "no preds.jsonl to evaluate. Run: bash ablation/run.sh" >&2
  exit 1
fi
