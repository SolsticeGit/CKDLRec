#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
PYTHON=${PYTHON:-python}
TAU=${TAU:-0.4}
ALPHA=${ALPHA:-0.4}
BETA=${BETA:-0.2}
GRL_LMAX=${GRL_LMAX:-0.6}
SKIP=${SKIP:-1}

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS}"
elif [[ -n "${DATASET:-}" ]]; then
  DATASETS=("${DATASET}")
else
  DATASETS=(cds toys movies)
fi
read -r -a ABLATIONS <<< "${ABLATIONS:-teacher no_kd no_adv pop_only rand_ret}"

OUT_ROOT="ablation/output"

stu_dir() {
  echo "${OUT_ROOT}/checkpoints/ckdlrec/${1}/${2}"
}

res_dir() {
  echo "${OUT_ROOT}/results/${1}/${2}"
}

cf_abl_dir() {
  echo "${OUT_ROOT}/cf_data/${1}/tau_${2}_${3}"
}

tea_abl_dir() {
  echo "${OUT_ROOT}/checkpoints/cf_sft/${1}/tau_${2}_${3}"
}

main_teacher() {
  echo "outputs/checkpoints/cf_sft/${1}/tau_${2}/best"
}

skip_file() {
  [[ "${SKIP}" == "1" && -f "$1" ]]
}

student_train_done() {
  local root="$1"
  [[ -f "${root}/train_done.json" ]] && return 0
  [[ -f "${root}/train_log.txt" ]] && grep -q 'done | best valid distill' "${root}/train_log.txt"
}

run_infer_eval() {
  local ds="$1" name="$2" ckpt="$3"
  local preds metrics
  preds="$(res_dir "${ds}" "${name}")/preds.jsonl"
  metrics="$(res_dir "${ds}" "${name}")/metrics.json"

  if [[ ! -d "${ckpt}" ]]; then
    echo "not found: ${ckpt}, cannot run inference for ${name}" >&2
    exit 1
  fi

  if skip_file "${preds}"; then
    echo ">>> skip inference  (${preds})"
  else
    echo ">>> inference  ${name}  ckpt=${ckpt}"
    mkdir -p "$(dirname "${preds}")"
    DATASET="${ds}" GPU="${GPU}" PYTHON="${PYTHON}" \
      TAU="${TAU}" ALPHA="${ALPHA}" BETA="${BETA}" GRL_LMAX="${GRL_LMAX}" \
      CKPT="${ckpt}" OUTPUT="${preds}" \
      bash inference.sh
  fi

  if skip_file "${metrics}"; then
    echo ">>> skip evaluate  (${metrics})"
  else
    echo ">>> evaluate  ${name}"
    DATASET="${ds}" PYTHON="${PYTHON}" \
      TAU="${TAU}" ALPHA="${ALPHA}" BETA="${BETA}" \
      PREDS="${preds}" OUTPUT="${metrics}" \
      bash evaluate.sh
  fi
}

ensure_main_teacher() {
  local ds="$1"
  local ckpt
  ckpt="$(main_teacher "${ds}" "${TAU}")"
  if skip_file "${ckpt}/adapter_config.json"; then
    echo ">>> skip train_cf_sft  (${ckpt})"
    return
  fi
  echo ">>> train_cf_sft  (main teacher tau=${TAU})"
  DATASET="${ds}" GPU="${GPU}" PYTHON="${PYTHON}" TAU="${TAU}" \
    bash train_cf_sft.sh
}

run_student() {
  local ds="$1" name="$2" kd="$3" alpha="$4" beta="$5"
  local cf_dir="${6:-}"
  local cf_sft="${7:-}"
  local ckpt
  ckpt="$(stu_dir "${ds}" "${name}")"

  echo
  echo "======== ablation=${name}  dataset=${ds}  γ=${kd} α=${alpha} β=${beta} ========"

  if student_train_done "${ckpt}" && [[ "${RESUME:-0}" != "1" ]]; then
    echo ">>> skip train_ckdlrec  (${ckpt}/best)"
  else
    local resume="${RESUME:-0}"
    if [[ "${resume}" != "1" ]] && { [[ -f "${ckpt}/best/adapter_config.json" ]] || [[ -f "${ckpt}/last/adapter_config.json" ]]; }; then
      resume=1
      echo ">>> resume train_ckdlrec  (${ckpt})"
    else
      echo ">>> train_ckdlrec"
    fi
    DATASET="${ds}" GPU="${GPU}" PYTHON="${PYTHON}" \
      TAU="${TAU}" ALPHA="${alpha}" BETA="${beta}" KD_WEIGHT="${kd}" \
      GRL_LMAX="${GRL_LMAX}" RESUME="${resume}" \
      CF_DIR="${cf_dir}" CF_SFT_PATH="${cf_sft}" OUTPUT_DIR="${ckpt}" \
      bash train_ckdlrec.sh
  fi

  run_infer_eval "${ds}" "${name}" "${ckpt}/best"
}

run_cf_variant() {
  local ds="$1" name="$2" flag="$3"
  local cf_dir tea_dir
  cf_dir="$(cf_abl_dir "${ds}" "${TAU}" "${name}")"
  tea_dir="$(tea_abl_dir "${ds}" "${TAU}" "${name}")"

  echo
  echo "======== ablation=${name}  dataset=${ds}  rebuild cf_data + teacher ========"

  if skip_file "${cf_dir}/cf_train.jsonl"; then
    echo ">>> skip build_cf_data  (${cf_dir})"
  else
    echo ">>> build_cf_data ${flag}"
    ${PYTHON} build_cf_data.py --dataset "${ds}" --tau "${TAU}" \
      --out_dir "${cf_dir}" ${flag}
  fi

  if skip_file "${tea_dir}/best/adapter_config.json"; then
    echo ">>> skip train_cf_sft  (${tea_dir}/best)"
  else
    echo ">>> train_cf_sft"
    DATASET="${ds}" GPU="${GPU}" PYTHON="${PYTHON}" TAU="${TAU}" \
      CF_DIR="${cf_dir}" OUTPUT_DIR="${tea_dir}" \
      bash train_cf_sft.sh
  fi

  run_student "${ds}" "${name}" "1" "${ALPHA}" "${BETA}" \
    "${cf_dir}" "${tea_dir}/best"
}

run_teacher() {
  local ds="$1"
  local ckpt
  ckpt="$(main_teacher "${ds}" "${TAU}")"
  echo
  echo "======== ablation=teacher  dataset=${ds}  infer frozen θ_cf ========"
  ensure_main_teacher "${ds}"
  run_infer_eval "${ds}" teacher "${ckpt}"
}

run_one_ablation() {
  local ds="$1" name="$2"
  case "${name}" in
    teacher)
      run_teacher "${ds}"
      ;;
    no_kd)
      run_student "${ds}" no_kd 0 1 "${BETA}"
      ;;
    no_sft)
      run_student "${ds}" no_sft 1 0 "${BETA}"
      ;;
    no_adv)
      run_student "${ds}" no_adv 1 "${ALPHA}" 0
      ;;
    pop_only)
      run_cf_variant "${ds}" pop_only --score_pop_only
      ;;
    rand_ret)
      run_cf_variant "${ds}" rand_ret --random_retrieve
      ;;
    *)
      echo "unknown ablation: ${name} (teacher no_kd no_sft no_adv pop_only rand_ret)" >&2
      exit 1
      ;;
  esac
}

echo ">>> GPU=${GPU} datasets=${DATASETS[*]} ablations=${ABLATIONS[*]}"
echo ">>> tau=${TAU} α=${ALPHA} β=${BETA} λ_max=${GRL_LMAX} skip=${SKIP}"
echo ">>> out ${OUT_ROOT}/"

for ds in "${DATASETS[@]}"; do
  echo
  echo "########## dataset=${ds} ##########"
  for abl in "${ABLATIONS[@]}"; do
    run_one_ablation "${ds}" "${abl}"
  done
done

echo
echo ">>> ablation done"
