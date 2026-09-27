#!/usr/bin/env bash

# Shared defaults for the temporal-mean, multi-slot PReM experiments.

PREM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PREM_REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/prem/bin/python}"
MAIN_OUTPUT_ROOT="${MAIN_OUTPUT_ROOT:-outputs/prem_attention_kv}"
ANALYSIS_OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:-outputs/model_analysis}"
DATASETS="${DATASETS:-egoschema,mlvu,longvideobench,videomme,mvbench,lvbench}"
VIDEO_ROOT="${VIDEO_ROOT:-data/llava-video-178k/frames}"
# Writer temporal cap is independent of decoder budget B. Train and eval share it.
TRAIN_MAX_FRAMES="${TRAIN_MAX_FRAMES:-240}"
EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-${TRAIN_MAX_FRAMES}}"
VISUAL_BUFFER_FRAMES="${VISUAL_BUFFER_FRAMES:-16}"
EPOCHS="${EPOCHS:-1}"
FORCE="${FORCE:-0}"
PREM_DRY_RUN="${PREM_DRY_RUN:-0}"

prem_pin_table2_recipe() {
  # Lockstep with scripts/run_prem_kv_qwen25_3b.sh. Call after sourcing this file.
  TRAIN_MAX_FRAMES="${TRAIN_MAX_FRAMES:-240}"
  EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-${TRAIN_MAX_FRAMES}}"
  EPOCHS="${EPOCHS:-1}"
  PREM_TRAIN_ALPHA="${PREM_TRAIN_ALPHA:-0.75}"
  PREM_PRED_WEIGHT="${PREM_PRED_WEIGHT:-0.2}"
  PREM_PRED_TOKENS="${PREM_PRED_TOKENS:-4}"
  PREM_ROUTER_GAMMA="${PREM_ROUTER_GAMMA:-0.10}"
  PREM_LAYER_GROUPS="${PREM_LAYER_GROUPS:-4}"
  QWEN25_NUM_SLOTS="${QWEN25_NUM_SLOTS:-1}"
  QWEN25_LR="${QWEN25_LR:-2e-4}"
  QWEN25_EVAL_ALPHA="${QWEN25_EVAL_ALPHA:-0.5}"
  QWEN25_MODEL_PATH="${QWEN25_MODEL_PATH:-ckpt/Qwen2.5-VL-3B-Instruct}"
  QWEN25_DATA_FILE="${QWEN25_DATA_FILE:-data/llava-video-178k/trainset_9k.jsonl}"
  DATASETS="${DATASETS:-egoschema,mlvu,longvideobench,videomme,mvbench,lvbench}"
  export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
}

prem_print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

prem_run() {
  if [[ "${PREM_DRY_RUN}" == "1" ]]; then
    prem_print_command "$@"
  else
    "$@"
  fi
}

prem_value_tag() {
  local value="$1"
  value="${value//-/m}"
  printf '%s\n' "${value//./p}"
}

prem_configure_backbone() {
  local backbone="$1" budget="${2:-16}" ckpt_stem
  # Match scripts/run_prem_kv_*.sh: T=240, tuned recipe, tagged checkpoint dirs.
  PREM_TRAIN_ALPHA="${PREM_TRAIN_ALPHA:-0.75}"
  PREM_ROUTER_GAMMA="${PREM_ROUTER_GAMMA:-0.10}"
  PREM_PRED_WEIGHT="${PREM_PRED_WEIGHT:-0.2}"
  PREM_PRED_TOKENS="${PREM_PRED_TOKENS:-4}"
  PREM_LAYER_GROUPS="${PREM_LAYER_GROUPS:-4}"
  case "${backbone}" in
    qwen25)
      PREM_MODEL_TYPE=qwen2_5vl
      PREM_MODEL_FAMILY=qwen25
      PREM_ONLINE_MODEL=qwen25vl
      PREM_MODEL_PATH="${QWEN25_MODEL_PATH:-ckpt/Qwen2.5-VL-3B-Instruct}"
      PREM_DATA_FILE="${QWEN25_DATA_FILE:-data/llava-video-178k/trainset_9k.jsonl}"
      PREM_TRAIN_LR="${QWEN25_LR:-2e-4}"
      PREM_EVAL_ALPHA="${QWEN25_EVAL_ALPHA:-0.5}"
      PREM_NUM_SLOTS="${QWEN25_NUM_SLOTS:-1}"
      ckpt_stem="qwen25_3b_b${budget}"
      ;;
    qwen3)
      PREM_MODEL_TYPE=qwen3vl
      PREM_MODEL_FAMILY=qwen3
      PREM_ONLINE_MODEL=qwen3vl
      PREM_MODEL_PATH="${QWEN3_MODEL_PATH:-ckpt/Qwen3-VL-8B-Instruct}"
      PREM_DATA_FILE="${QWEN3_DATA_FILE:-data/llava-video-178k/trainset_9k.jsonl}"
      PREM_TRAIN_LR="${QWEN3_LR:-1e-4}"
      PREM_EVAL_ALPHA="${QWEN3_EVAL_ALPHA:-0.5}"
      PREM_NUM_SLOTS="${QWEN3_NUM_SLOTS:-4}"
      ckpt_stem="qwen3_8b_b${budget}"
      ;;
    *)
      echo "Unknown backbone: ${backbone}; expected qwen25 or qwen3" >&2
      return 2
      ;;
  esac
  PREM_BACKBONE="${backbone}"
  PREM_CFG_TAG="ns${PREM_NUM_SLOTS}_rg${PREM_ROUTER_GAMMA//./p}_lg${PREM_LAYER_GROUPS}_pw${PREM_PRED_WEIGHT//./p}_pt${PREM_PRED_TOKENS}_ta${PREM_TRAIN_ALPHA//./p}"
  if [[ "${backbone}" == "qwen25" ]]; then
    PREM_MAIN_CKPT="${QWEN25_MAIN_CKPT:-${MAIN_OUTPUT_ROOT}/${ckpt_stem}_${PREM_CFG_TAG}/prem.pt}"
  else
    PREM_MAIN_CKPT="${QWEN3_MAIN_CKPT:-${MAIN_OUTPUT_ROOT}/${ckpt_stem}_${PREM_CFG_TAG}/prem.pt}"
  fi
}

prem_checkpoint_complete() {
  local checkpoint="$1"
  [[ -f "${checkpoint}" ]] || return 1
  "${PYTHON_BIN}" - "${checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
valid = (
    checkpoint.get("complete") is True
    and checkpoint.get("writer_mode") == "temporal_mean_per_step"
)
raise SystemExit(0 if valid else 1)
PY
}

prem_require_checkpoint() {
  local checkpoint="$1"
  if [[ "${PREM_DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    return 1
  fi
}

prem_train_current() {
  local out_dir="$1" budget="$2" modulation="$3"
  shift 3
  local checkpoint="${out_dir}/prem.pt"

  if [[ "${PREM_DRY_RUN}" != "1" ]] && prem_checkpoint_complete "${checkpoint}"; then
    echo "[skip] completed checkpoint: ${checkpoint}"
    return
  fi

  prem_run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_TYPE="${PREM_MODEL_TYPE}" \
    MODEL_PATH="${PREM_MODEL_PATH}" \
    DATA_FILE="${PREM_DATA_FILE}" \
    VIDEO_ROOT="${VIDEO_ROOT}" \
    OUT_DIR="${out_dir}" \
    OUT_CKPT="${checkpoint}" \
    RESUME="${RESUME:-1}" \
    LR="${PREM_TRAIN_LR}" \
    EPOCHS="${EPOCHS}" \
    MAX_FRAMES="${TRAIN_MAX_FRAMES}" \
    VISUAL_BUFFER_FRAMES="${budget}" \
    PREM_MODULATION="${modulation}" \
    ALPHA="${PREM_TRAIN_ALPHA}" \
    ROUTER_GAMMA="${PREM_ROUTER_GAMMA}" \
    PRED_WEIGHT="${PREM_PRED_WEIGHT}" \
    PRED_TOKENS="${PREM_PRED_TOKENS}" \
    NUM_SLOTS="${PREM_NUM_SLOTS}" \
    MEM_DIM=128 \
    MAX_MEMORY_TOKENS=128 \
    PREM_LAYER_GROUPS="${PREM_LAYER_GROUPS}" \
    "$@" \
    bash scripts/train/run_train.sh
}

prem_eval_offline() {
  local checkpoint="$1" output_dir="$2" alpha="$3" modulation="$4"
  shift 4
  prem_require_checkpoint "${checkpoint}"
  prem_run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EVAL_MODE=offline \
    MODEL_FAMILY="${PREM_MODEL_FAMILY}" \
    MODEL_PATH="${PREM_MODEL_PATH}" \
    OFFLINE_MODEL=prem \
    PREM_CKPT="${checkpoint}" \
    PREM_ALPHA="${alpha}" \
    OVERRIDE_PREM_ALPHA=1 \
    PREM_MODULATION="${modulation}" \
    DATASETS="${DATASETS}" \
    OUTPUT_DIR="${output_dir}" \
    MAX_FRAMES="${EVAL_MAX_FRAMES}" \
    FORCE="${FORCE}" \
    "$@" \
    bash scripts/eval/run_eval.sh
}

prem_eval_online() {
  local checkpoint="$1" output_dir="$2" alpha="$3" modulation="$4"
  shift 4
  prem_require_checkpoint "${checkpoint}"
  prem_run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EVAL_MODE=online \
    MODEL_FAMILY="${PREM_MODEL_FAMILY}" \
    MODEL_PATH="${PREM_MODEL_PATH}" \
    ONLINE_MODEL=prem \
    PREM_CKPT="${checkpoint}" \
    PREM_ALPHA="${alpha}" \
    OVERRIDE_PREM_ALPHA=1 \
    PREM_MODULATION="${modulation}" \
    DATASETS="${DATASETS}" \
    STREAM_OUTPUT_DIR="${output_dir}" \
    MAX_FRAMES="${EVAL_MAX_FRAMES}" \
    FORCE="${FORCE}" \
    "$@" \
    bash scripts/eval/run_eval.sh
}

prem_eval_text_only() {
  local checkpoint="$1" output_dir="$2"
  prem_require_checkpoint "${checkpoint}"
  prem_run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EVAL_MODE=offline \
    MODEL_FAMILY="${PREM_MODEL_FAMILY}" \
    MODEL_PATH="${PREM_MODEL_PATH}" \
    OFFLINE_MODEL=qwen \
    PREM_CKPT="${checkpoint}" \
    TEXT_ONLY=1 \
    DATASETS="${DATASETS}" \
    OUTPUT_DIR="${output_dir}" \
    MAX_FRAMES="${EVAL_MAX_FRAMES}" \
    FORCE="${FORCE}" \
    bash scripts/eval/run_eval.sh
}
