#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/prem/bin/python}"
MODEL_PATH="${MODEL_PATH:-ckpt/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/prem_attention_kv}"
DATA_FILE="${DATA_FILE:-data/llava-video-178k/trainset_9k.jsonl}"
DATASETS="${DATASETS:-egoschema,mlvu,longvideobench,videomme,mvbench,lvbench}"
VISUAL_BUDGETS="${VISUAL_BUDGETS:-16 32 64 90}"
TRAIN_ALPHAS="${TRAIN_ALPHAS:-0.25 0.5}"
EVAL_ALPHAS="${EVAL_ALPHAS:-0.0 0.25 0.5}"
MAX_FRAMES="${MAX_FRAMES:-64}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
PRED_WEIGHT="${PRED_WEIGHT:-0.2}"
PRED_TOKENS="${PRED_TOKENS:-4}"
ROUTER_GAMMA="${ROUTER_GAMMA:-0.10}"
NUM_SLOTS="${NUM_SLOTS:-1}"
PREM_LAYER_GROUPS="${PREM_LAYER_GROUPS:-4}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
export PREM_MODULATION="attention_kv"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

read -r -a budgets <<< "${VISUAL_BUDGETS}"
read -r -a train_alphas <<< "${TRAIN_ALPHAS}"
read -r -a eval_alphas <<< "${EVAL_ALPHAS}"

dest_root() {
  local budget="$1"
  local train_alpha="$2"
  local cfg_tag="ns${NUM_SLOTS}_rg${ROUTER_GAMMA//./p}_lg${PREM_LAYER_GROUPS}_pw${PRED_WEIGHT//./p}_pt${PRED_TOKENS}_ta${train_alpha//./p}_mf${MAX_FRAMES}"
  echo "${OUTPUT_ROOT}/qwen25_3b_b${budget}_${cfg_tag}"
}

echo "=== Qwen2.5-VL-3B EgoSchema search ==="
echo "    B=${VISUAL_BUDGETS}  T=${MAX_FRAMES}  train_alpha=${TRAIN_ALPHAS}  eval_alpha=${EVAL_ALPHAS}"
echo "    datasets=${DATASETS}"
echo "    frozen: lr=${LR} epochs=${EPOCHS} pred_w=${PRED_WEIGHT} pred_tok=${PRED_TOKENS} rg=${ROUTER_GAMMA} ns=${NUM_SLOTS} lg=${PREM_LAYER_GROUPS}"
for train_alpha in "${train_alphas[@]}"; do
  for budget in "${budgets[@]}"; do
    echo "[dest] B=${budget} T=${MAX_FRAMES} train_alpha=${train_alpha} -> $(dest_root "${budget}" "${train_alpha}")"
  done
done

run_budget() {
  local budget="$1"
  local train_alpha="$2"
  local max_frames="${MAX_FRAMES}"
  local root ckpt alpha alpha_tag eval_root

  root="$(dest_root "${budget}" "${train_alpha}")"
  ckpt="${root}/prem.pt"

  echo "=== Qwen25 B=${budget} T=${max_frames} train_alpha=${train_alpha} -> ${root} ==="

  if [[ "${SKIP_TRAIN}" != "1" && ( "${FORCE_TRAIN}" == "1" || ! -s "${ckpt}" ) ]]; then
    MODEL_TYPE=qwen2_5vl MODEL_PATH="${MODEL_PATH}" DATA_FILE="${DATA_FILE}" \
    LR="${LR}" EPOCHS="${EPOCHS}" MAX_FRAMES="${max_frames}" \
    VISUAL_BUFFER_FRAMES="${budget}" PREM_MODULATION="${PREM_MODULATION}" \
    ALPHA="${train_alpha}" ROUTER_GAMMA="${ROUTER_GAMMA}" NUM_SLOTS="${NUM_SLOTS}" \
    PREM_LAYER_GROUPS="${PREM_LAYER_GROUPS}" \
    PRED_WEIGHT="${PRED_WEIGHT}" PRED_TOKENS="${PRED_TOKENS}" \
    OUT_DIR="${root}" OUT_CKPT="${ckpt}" PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/train/run_train.sh
  elif [[ "${SKIP_TRAIN}" != "1" ]]; then
    echo "[skip] Existing checkpoint: ${ckpt} (set FORCE_TRAIN=1 to retrain)"
  fi

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    return
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi

  for alpha in "${eval_alphas[@]}"; do
    alpha_tag="alpha_${alpha//./p}"
    eval_root="${root}"
    if (( ${#eval_alphas[@]} > 1 )); then
      eval_root="${root}/${alpha_tag}"
    fi

    MAX_FRAMES="${max_frames}" PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" \
    PREM_MODULATION="${PREM_MODULATION}" PREM_ALPHA="${alpha}" OVERRIDE_PREM_ALPHA=1 \
    DATASETS="${DATASETS}" PREM_CKPT="${ckpt}" OUTPUT_DIR="${eval_root}/offline" \
    EVAL_MODE=offline MODEL_FAMILY=qwen25 OFFLINE_MODEL=prem \
    bash scripts/eval/run_eval.sh

    MAX_FRAMES="${max_frames}" PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" \
    PREM_MODULATION="${PREM_MODULATION}" PREM_ALPHA="${alpha}" OVERRIDE_PREM_ALPHA=1 \
    DATASETS="${DATASETS}" PREM_CKPT="${ckpt}" STREAM_OUTPUT_DIR="${eval_root}/online" \
    EVAL_MODE=online MODEL_FAMILY=qwen25 ONLINE_MODEL=prem \
    bash scripts/eval/run_eval.sh
  done
}

for train_alpha in "${train_alphas[@]}"; do
  for budget in "${budgets[@]}"; do
    run_budget "${budget}" "${train_alpha}"
  done
done
