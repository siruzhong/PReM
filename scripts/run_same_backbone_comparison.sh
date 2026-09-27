#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON="python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
MODEL_PATH="${MODEL_PATH:-ckpt/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/same_backbone_qwen25_b16}"
TRAIN_DATA="${TRAIN_DATA:-data/llava-video-178k/trainset_9k.jsonl}"
TRAIN_VIDEO_ROOT="${TRAIN_VIDEO_ROOT:-data/llava-video-178k/frames}"
PREM_CKPT="${PREM_CKPT:-outputs/table4_efficiency/train/qwen25_3b_b16_ns1_rg0p10_lg4_pw0p2_pt4_ta0p75/prem.pt}"
PREM_RESULTS_ROOT="${PREM_RESULTS_ROOT:-outputs/table4_efficiency/prem_B16/prem_qwen25_public}"
LORA_CKPT="${LORA_CKPT:-${OUTPUT_ROOT}/train/lora.pt}"
TOKEN_CKPT="${TOKEN_CKPT:-${OUTPUT_ROOT}/train/token_readout.pt}"
DATASETS="${DATASETS:-longvideobench,mlvu,videomme,egoschema,mvbench,lvbench}"
METHODS="${METHODS:-base,lora,token_readout,infinipot_v,prem}"
H2O_HEAVY_TOKENS="${H2O_HEAVY_TOKENS:-1024}"
H2O_RECENT_TOKENS="${H2O_RECENT_TOKENS:-1024}"
INFINIPOT_BLOCK_UNITS="${INFINIPOT_BLOCK_UNITS:-32}"
INFINIPOT_KEEP_UNITS="${INFINIPOT_KEEP_UNITS:-24}"
INFINIPOT_TAR_RATIO="${INFINIPOT_TAR_RATIO:-0.5}"
INFINIPOT_QUERY_RATIO="${INFINIPOT_QUERY_RATIO:-0.25}"

DRY_RUN=0
DO_TRAIN=1
DO_EVAL=1
DO_SUMMARY=1
OVERWRITE=0
REUSE_PREM="${REUSE_PREM:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/run_same_backbone_comparison.sh [options]

Options:
  --dry-run          Print commands without training or evaluation.
  --train-only       Train LoRA and Token-Readout checkpoints only.
  --eval-only        Evaluate selected methods only.
  --summarize-only   Validate existing results and generate summary files.
  --methods LIST     Comma-separated subset of the five formal rows; h2o is optional.
  --datasets LIST    Comma-separated public dataset subset (summary needs all six).
  --overwrite        Replace predictions/checkpoints selected by this entry point.
  --reuse-prem       Import the completed PReM run (default).
  --rerun-prem       Evaluate PReM again instead of importing the completed run.
  -h, --help         Show this help.

Environment overrides:
  PYTHON_BIN, MODEL_PATH, OUTPUT_ROOT, PREM_CKPT, PREM_RESULTS_ROOT, LORA_CKPT, TOKEN_CKPT,
  TRAIN_DATA, TRAIN_VIDEO_ROOT, CUDA_DEVICES, TRAIN_CUDA_DEVICES,
  WORKERS_PER_GPU, INFINIPOT_BLOCK_UNITS, INFINIPOT_KEEP_UNITS,
  INFINIPOT_TAR_RATIO, INFINIPOT_QUERY_RATIO, H2O_HEAVY_TOKENS,
  H2O_RECENT_TOKENS.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --train-only)
      DO_TRAIN=1
      DO_EVAL=0
      DO_SUMMARY=0
      shift
      ;;
    --eval-only)
      DO_TRAIN=0
      DO_EVAL=1
      DO_SUMMARY=0
      shift
      ;;
    --summarize-only)
      DO_TRAIN=0
      DO_EVAL=0
      DO_SUMMARY=1
      shift
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --reuse-prem)
      REUSE_PREM=1
      shift
      ;;
    --rerun-prem)
      REUSE_PREM=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_command() {
  printf '[exec]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

contains_method() {
  [[ ",${METHODS}," == *",$1,"* ]]
}

for method in ${METHODS//,/ }; do
  case "${method}" in
    base|lora|token_readout|infinipot_v|h2o|prem) ;;
    *)
      echo "Unknown method in METHODS: ${method}" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" && "${DRY_RUN}" -eq 0 ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${CUDA_DEVICES:-}" ]]; then
  CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {gsub(/[[:space:]]/, "", $0); printf "%s%s", sep, $0; sep=","}')"
fi
if [[ -z "${CUDA_DEVICES}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    CUDA_DEVICES="0"
  else
    echo "No CUDA devices detected; set CUDA_DEVICES explicitly" >&2
    exit 1
  fi
fi
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-${CUDA_DEVICES}}"
TRAIN_WORLD_SIZE="$(awk -F, '{print NF}' <<<"${TRAIN_CUDA_DEVICES}")"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/eval"
fi

if [[ "${DO_TRAIN}" -eq 1 ]]; then
  TRAIN_COMMON=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${TRAIN_WORLD_SIZE}"
    scripts/train/train_same_backbone.py
    --model_path "${MODEL_PATH}"
    --llava_data_file "${TRAIN_DATA}"
    --llava_video_root "${TRAIN_VIDEO_ROOT}"
    --max_frames 240
    --visual_buffer_frames 16
    --max_pixels 200704
    --fps 1
    --epochs 1
    --lr 2e-4
    --warmup_ratio 0.03
    --global_batch_size 8
    --stream_jsonl
  )
  TRAIN_FLAGS=()
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    TRAIN_FLAGS+=(--overwrite)
  else
    TRAIN_FLAGS+=(--resume)
  fi
  if contains_method lora; then
    run_command env "CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES}" \
      "${TRAIN_COMMON[@]}" --method lora --out_ckpt "${LORA_CKPT}" \
      --lora_rank 20 --lora_alpha 40 "${TRAIN_FLAGS[@]}"
  fi
  if contains_method token_readout; then
    run_command env "CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES}" \
      "${TRAIN_COMMON[@]}" --method token_readout --out_ckpt "${TOKEN_CKPT}" \
      --num_slots 1 --mem_dim 128 --layer_groups 4 --alpha 0.75 \
      --max_memory_tokens 128 --pred_weight 0.2 --pred_tokens 4 "${TRAIN_FLAGS[@]}"
  fi
fi

if [[ "${DO_EVAL}" -eq 1 ]]; then
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    for required in "${MODEL_PATH}"; do
      [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
    done
    if contains_method lora && [[ ! -f "${LORA_CKPT}" ]]; then
      echo "Missing LoRA checkpoint: ${LORA_CKPT}" >&2
      exit 1
    fi
    if contains_method token_readout && [[ ! -f "${TOKEN_CKPT}" ]]; then
      echo "Missing Token-Readout checkpoint: ${TOKEN_CKPT}" >&2
      exit 1
    fi
    if contains_method prem && [[ "${REUSE_PREM}" -eq 0 && ! -f "${PREM_CKPT}" ]]; then
      echo "Missing PReM checkpoint: ${PREM_CKPT}" >&2
      exit 1
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && [[ "${DRY_RUN}" -eq 0 ]]; then
    nvidia-smi \
      --query-gpu=index,name,uuid,driver_version,memory.total \
      --format=csv,noheader > "${OUTPUT_ROOT}/hardware.csv"
  fi

  EVAL_COMMON=(
    "${PYTHON_BIN}" scripts/eval/run_same_backbone_online.py
    --model_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_ROOT}/eval"
    --datasets "${DATASETS}"
    --cuda_devices "${CUDA_DEVICES}"
    --max_frames 240
    --buffer_frames 16
    --max_pixels 200704
    --fps 1
    --chunk_frames 2
    --stream_chunk_tokens 32
    --h2o_heavy_tokens "${H2O_HEAVY_TOKENS}"
    --h2o_recent_tokens "${H2O_RECENT_TOKENS}"
    --infinipot_block_units "${INFINIPOT_BLOCK_UNITS}"
    --infinipot_keep_units "${INFINIPOT_KEEP_UNITS}"
    --infinipot_tar_ratio "${INFINIPOT_TAR_RATIO}"
    --infinipot_query_ratio "${INFINIPOT_QUERY_RATIO}"
    --prem_modulation attention_kv
    --profile_efficiency
  )
  if [[ -n "${WORKERS_PER_GPU:-}" ]]; then
    EVAL_COMMON+=(--workers_per_gpu "${WORKERS_PER_GPU}")
  fi
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    EVAL_COMMON+=(--overwrite)
  fi
  contains_method base && run_command "${EVAL_COMMON[@]}" --method base --evaluation_name base
  contains_method lora && run_command "${EVAL_COMMON[@]}" --method lora --evaluation_name lora --baseline_ckpt "${LORA_CKPT}"
  contains_method token_readout && run_command "${EVAL_COMMON[@]}" --method token_readout --evaluation_name token_readout --baseline_ckpt "${TOKEN_CKPT}"
  contains_method infinipot_v && run_command "${EVAL_COMMON[@]}" --method infinipot_v --evaluation_name infinipot_v
  contains_method h2o && run_command "${EVAL_COMMON[@]}" --method h2o --evaluation_name h2o
  if contains_method prem; then
    if [[ "${REUSE_PREM}" -eq 1 ]]; then
      run_command "${PYTHON_BIN}" scripts/eval/import_same_backbone_prem.py \
        --source_root "${PREM_RESULTS_ROOT}" --output_root "${OUTPUT_ROOT}"
    else
      run_command "${EVAL_COMMON[@]}" --method prem --evaluation_name prem --baseline_ckpt "${PREM_CKPT}"
    fi
  fi
fi

if [[ "${DO_SUMMARY}" -eq 1 ]]; then
  run_command "${PYTHON_BIN}" scripts/eval/summarize_same_backbone.py \
    --output_root "${OUTPUT_ROOT}"
fi

echo "[done] output_root=${OUTPUT_ROOT}"
