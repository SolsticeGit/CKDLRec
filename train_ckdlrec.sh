#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

DATASET=${DATASET:-ml1m}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}

EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-320}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-0}

TAU=${TAU:-0.3}
ALPHA=${ALPHA:-0.4}
BETA=${BETA:-0.2}
KD_WEIGHT=${KD_WEIGHT:-1.0}
TAU_DISTILL=${TAU_DISTILL:-2.0}
GRL_LMAX=${GRL_LMAX:-0.6}

EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-20}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}

CF_SFT_PATH=${CF_SFT_PATH:-}
CF_DIR=${CF_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
RESUME=${RESUME:-0}

CF_TAG="tau_${TAU}"
RUN_TAG="tau_${TAU}_sft_${ALPHA}_adv_${BETA}"

export CUDA_VISIBLE_DEVICES=${GPU}
export TOKENIZERS_PARALLELISM=false

case "${DATASET}" in
  ml1m)  DATA_DIR="data/MovieLens1M/processed" ;;
  cds)   DATA_DIR="data/CDs_and_Vinyl/processed" ;;
  toys)  DATA_DIR="data/Toys_and_Games/processed" ;;
  movies) DATA_DIR="data/Movies_and_TV/processed" ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

CF_DATA="${CF_DIR:-outputs/cf_data/${DATASET}/${CF_TAG}}/cf_train.jsonl"
if [[ ! -f "${CF_DATA}" ]]; then
  echo "not found: ${CF_DATA}" >&2
  echo "Run: ${PYTHON} build_cf_data.py --dataset ${DATASET} --tau ${TAU}" >&2
  exit 1
fi

CF_SFT_CHECK="${CF_SFT_PATH:-outputs/checkpoints/cf_sft/${DATASET}/${CF_TAG}/best}"
if [[ "${KD_WEIGHT}" != "0" && "${KD_WEIGHT}" != "0.0" ]] && [[ ! -d "${CF_SFT_CHECK}" ]]; then
  echo "not found: ${CF_SFT_CHECK}" >&2
  echo "Run: bash train_cf_sft.sh" >&2
  exit 1
fi

ARGS=(
  --dataset "${DATASET}"
  --tau "${TAU}"
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
  --alpha "${ALPHA}"
  --beta "${BETA}"
  --kd_weight "${KD_WEIGHT}"
  --tau_distill "${TAU_DISTILL}"
  --grl_lambda_max "${GRL_LMAX}"
)
if [[ "${GRAD_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--gradient_checkpointing)
fi
if [[ "${MAX_TRAIN_SAMPLES}" != "0" ]]; then
  ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${CF_DIR}" ]]; then
  ARGS+=(--cf_dir "${CF_DIR}")
fi
if [[ -n "${CF_SFT_PATH}" ]]; then
  ARGS+=(--cf_sft_path "${CF_SFT_PATH}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

echo ">>> dataset=${DATASET} gpu=${GPU} ${RUN_TAG} effective_batch=$((BATCH_SIZE * GRAD_ACCUM))"
echo ">>> student = new LoRA + GRL | teacher = frozen ${CF_SFT_CHECK}"
echo ">>> out checkpoints/ckdlrec/${DATASET}/${RUN_TAG}"
echo ">>> ${PYTHON} train_ckdlrec.py ${ARGS[*]}"
exec ${PYTHON} train_ckdlrec.py "${ARGS[@]}"
