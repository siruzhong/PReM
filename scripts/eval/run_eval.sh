#!/usr/bin/env bash
set -euo pipefail

# Unified evaluation entry point. Override the variables below for experiments.
EVAL_MODE="${EVAL_MODE:-offline}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen2}"
ONLINE_MODEL="${ONLINE_MODEL:-prem}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-egoschema,mlvu,longvideobench,videomme}"
FPS="${FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-240}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MCQ_MAX_NEW_TOKENS="${MCQ_MAX_NEW_TOKENS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PREM_ALPHA="${PREM_ALPHA:-1.0}"
OVERRIDE_PREM_ALPHA="${OVERRIDE_PREM_ALPHA:-0}"
PREM_MODULATION="${PREM_MODULATION:-attention}"
PREM_DISABLE_ANTI_DISTRACTOR="${PREM_DISABLE_ANTI_DISTRACTOR:-0}"
PREM_DISABLE_NOVELTY="${PREM_DISABLE_NOVELTY:-0}"
PREM_DISABLE_STABILITY="${PREM_DISABLE_STABILITY:-0}"
PREM_DISABLE_EVIDENCE_GATE_WRITE="${PREM_DISABLE_EVIDENCE_GATE_WRITE:-0}"
PREM_UNIFORM_WRITE_ROUTE="${PREM_UNIFORM_WRITE_ROUTE:-0}"
TEXT_ONLY="${TEXT_ONLY:-0}"
PROFILE_EFFICIENCY="${PROFILE_EFFICIENCY:-0}"
FORCE="${FORCE:-0}"
# Decord times out seeking the last frames of some DASH/YouTube MP4s on shared GPFS.
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

check_done() {
  local dir="$1"
  local mode="$2"
  if [[ "${FORCE}" == "1" ]]; then
    return 1
  fi
  local dataset output_dataset
  local -a selected_datasets
  IFS=',' read -r -a selected_datasets <<< "${DATASETS}"
  for dataset in "${selected_datasets[@]}"; do
    output_dataset="${dataset}"
    if [[ "${mode}" == "offline" && "${dataset}" == "videomme" ]]; then
      output_dataset="videommewo"
    fi
    if [[ ! -f "${dir}/${output_dataset}/result.json" ]] || \
       ! grep -q '"eval_signature"' "${dir}/${output_dataset}/result.json"; then
      return 1
    fi
  done
  echo "[skip] ${dir} has complete results for ${DATASETS}; use FORCE=1 to re-run." >&2
  return 0
}

case "${MODEL_FAMILY}" in
  qwen2)
    MODEL_PATH="${MODEL_PATH:-ckpt/Qwen2-VL-7B-Instruct}"
    PREM_CKPT="${PREM_CKPT:-outputs/prem_attention/qwen2/prem.pt}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/prem_attention/qwen2/offline}"
    STREAM_OUTPUT_DIR="${STREAM_OUTPUT_DIR:-outputs/prem_attention/qwen2/online}"
    ;;
  qwen25)
    MODEL_PATH="${MODEL_PATH:-ckpt/Qwen2.5-VL-3B-Instruct}"
    PREM_CKPT="${PREM_CKPT:-outputs/prem_attention/qwen25/prem.pt}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/prem_attention/qwen25/offline}"
    STREAM_OUTPUT_DIR="${STREAM_OUTPUT_DIR:-outputs/prem_attention/qwen25/online}"
    ;;
  qwen25_7b)
    MODEL_PATH="${MODEL_PATH:-ckpt/Qwen2.5-VL-7B-Instruct}"
    PREM_CKPT="${PREM_CKPT:-outputs/prem_attention/qwen25_7b/prem.pt}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/prem_attention/qwen25_7b/offline}"
    STREAM_OUTPUT_DIR="${STREAM_OUTPUT_DIR:-outputs/prem_attention/qwen25_7b/online}"
    ;;
  qwen3)
    MODEL_PATH="${MODEL_PATH:-ckpt/Qwen3-VL-8B-Instruct}"
    PREM_CKPT="${PREM_CKPT:-outputs/prem_attention/qwen3_8b/prem.pt}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/prem_attention/qwen3_8b/offline}"
    STREAM_OUTPUT_DIR="${STREAM_OUTPUT_DIR:-outputs/prem_attention/qwen3_8b/online}"
    ;;
  *)
    echo "MODEL_FAMILY must be qwen2, qwen25, qwen25_7b, or qwen3" >&2
    exit 2
    ;;
esac

FLASH_MODEL="${FLASH_MODEL:-ckpt/Flash-VStream-Qwen-7b}"

offline_result_name() {
  case "$1" in
    qwen)
      case "${MODEL_FAMILY}" in
        qwen2) echo "qwen2vl_7b" ;;
        qwen25) echo "qwen2_5vl_3b" ;;
        qwen25_7b) echo "qwen2_5vl_7b" ;;
        qwen3) echo "qwen3vl_8b" ;;
      esac
      ;;
    prem) echo "prem_attention" ;;
    flash) echo "flash_vstream" ;;
  esac
}

run_offline() {
  local selected_method="${OFFLINE_MODEL:-all}"
  local -a args=(
    "${PYTHON_BIN}" scripts/eval/run_eval_offline.py
    --qwen_model "${MODEL_PATH}"
    --flash_model "${FLASH_MODEL}"
    --prem_ckpt "${PREM_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --datasets "${DATASETS}"
    --fps "${FPS}" --max_frames "${MAX_FRAMES}" --max_pixels "${MAX_PIXELS}"
    --mcq_max_new_tokens "${MCQ_MAX_NEW_TOKENS}" --max_new_tokens "${MAX_NEW_TOKENS}"
    --prem_alpha "${PREM_ALPHA}" --prem_modulation "${PREM_MODULATION}"
  )
  if [[ "${selected_method}" != "all" ]]; then
    args+=(--only "${selected_method}")
  fi
  if [[ "${PROFILE_EFFICIENCY}" == "1" ]]; then
    args+=(--profile_efficiency)
  fi
  if [[ "${TEXT_ONLY}" == "1" ]]; then
    args+=(--text_only)
  fi
  if [[ "${OVERRIDE_PREM_ALPHA}" == "1" ]]; then
    args+=(--prem_override_alpha)
  fi
  if [[ "${PREM_DISABLE_ANTI_DISTRACTOR}" == "1" ]]; then
    args+=(--prem_disable_anti_distractor)
  fi
  if [[ "${PREM_DISABLE_NOVELTY}" == "1" ]]; then
    args+=(--prem_disable_novelty)
  fi
  if [[ "${PREM_DISABLE_STABILITY}" == "1" ]]; then
    args+=(--prem_disable_stability)
  fi
  if [[ "${PREM_DISABLE_EVIDENCE_GATE_WRITE}" == "1" ]]; then
    args+=(--prem_disable_evidence_gate_write)
  fi
  if [[ "${PREM_UNIFORM_WRITE_ROUTE}" == "1" ]]; then
    args+=(--prem_uniform_write_route)
  fi
  if [[ "${FORCE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ -n "${CUDA_DEVICES:-}" ]]; then
    args+=(--cuda_devices "${CUDA_DEVICES}")
  fi
  if [[ "${selected_method}" == "all" ]]; then
    local all_done=1 method result_name
    for method in qwen prem flash; do
      result_name="$(offline_result_name "${method}")"
      if ! check_done "${OUTPUT_DIR}/${result_name}" offline; then
        all_done=0
      fi
    done
    if [[ "${all_done}" == "1" ]]; then
      return
    fi
  else
    local selected_result_name
    selected_result_name="$(offline_result_name "${selected_method}")"
    if check_done "${OUTPUT_DIR}/${selected_result_name}" offline; then
      return
    fi
  fi
  local log_file="${OUTPUT_DIR}/offline_${selected_method}.log"
  mkdir -p "$(dirname "${log_file}")"
  "${args[@]}" 2>&1 | tee -a "${log_file}"
}

run_online() {
  local scenario="${1}"
  local output_dir="${STREAM_OUTPUT_DIR}"
  local evaluation_name="${ONLINE_MODEL}_${MODEL_FAMILY}_${scenario}"
  local -a args=(
    "${PYTHON_BIN}" scripts/eval/run_eval_online.py
    --mode "${ONLINE_MODEL}" --scenario "${scenario}"
    --model_path "${MODEL_PATH}"
    --output_dir "${output_dir}"
    --evaluation_name "${evaluation_name}"
    --datasets "${DATASETS}"
    --fps "${FPS}" --max_frames "${MAX_FRAMES}" --max_pixels "${MAX_PIXELS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
  )
  if check_done "${output_dir}/${evaluation_name}" online; then
    return
  fi
  if [[ -n "${CUDA_DEVICES:-}" ]]; then
    args+=(--cuda_devices "${CUDA_DEVICES}")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${PROFILE_EFFICIENCY}" == "1" ]]; then
    args+=(--profile_efficiency)
  fi
  case "${ONLINE_MODEL}" in
    prem)
      args+=(--prem_ckpt "${PREM_CKPT}" --prem_alpha "${PREM_ALPHA}" --prem_modulation "${PREM_MODULATION}"
        --chunk_frames "${CHUNK_FRAMES:-2}"
        --initial_chunk_frames "${INITIAL_CHUNK_FRAMES:-0}"
        --stream_chunk_tokens "${STREAM_CHUNK_TOKENS:-32}")
      if [[ "${OVERRIDE_PREM_ALPHA}" == "1" ]]; then
        args+=(--prem_override_alpha)
      fi
      if [[ "${PREM_DISABLE_ANTI_DISTRACTOR}" == "1" ]]; then
        args+=(--prem_disable_anti_distractor)
      fi
      if [[ "${PREM_DISABLE_NOVELTY}" == "1" ]]; then
        args+=(--prem_disable_novelty)
      fi
      if [[ "${PREM_DISABLE_STABILITY}" == "1" ]]; then
        args+=(--prem_disable_stability)
      fi
      if [[ "${PREM_DISABLE_EVIDENCE_GATE_WRITE}" == "1" ]]; then
        args+=(--prem_disable_evidence_gate_write)
      fi
      if [[ "${PREM_UNIFORM_WRITE_ROUTE}" == "1" ]]; then
        args+=(--prem_uniform_write_route)
      fi
      ;;
    flash)
      args+=(--model_path "${FLASH_MODEL}" --qwen_model "${MODEL_PATH}"
        --flash_initial_chunk_frames "${FLASH_INITIAL_CHUNK_FRAMES:-120}"
        --flash_chunk_frames "${FLASH_CHUNK_FRAMES:-1}")
      ;;
    qwen25vl|qwen3vl)
      args+=(--buffer_frames "${BUFFER_FRAMES:-16}" --max_video_tokens "${MAX_VIDEO_TOKENS:-11520}")
      if [[ "${TEXT_ONLY}" == "1" ]]; then
        args+=(--text_only)
      fi
      ;;
    qwen2vl)
      args+=(--buffer_frames "${BUFFER_FRAMES:-16}" --max_video_tokens "${MAX_VIDEO_TOKENS:-11520}")
      if [[ "${TEXT_ONLY}" == "1" ]]; then
        args+=(--text_only)
      fi
      ;;
    *)
      echo "ONLINE_MODEL must be prem, flash, qwen2vl, qwen25vl, or qwen3vl" >&2
      exit 2
      ;;
  esac
  local log_file="${output_dir}/${evaluation_name}.log"
  mkdir -p "$(dirname "${log_file}")"
  "${args[@]}" 2>&1 | tee -a "${log_file}"
}

case "${EVAL_MODE}" in
  offline) run_offline ;;
  online) run_online public ;;
  *)
    echo "EVAL_MODE must be offline or online" >&2
    exit 2
    ;;
esac
