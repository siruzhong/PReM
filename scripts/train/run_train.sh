#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${CUDA_DEVICES}" ]]; then
  CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF { gsub(/[[:space:]]/, "", $0); printf "%s%s", sep, $0; sep = "," }')"
fi
CUDA_DEVICES="${CUDA_DEVICES:-0}"
IFS=',' read -r -a _cuda_devices <<< "${CUDA_DEVICES}"
NGPUS="${NGPUS:-${MLP_WORKER_GPU:-${#_cuda_devices[@]}}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6013}}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/train.py}"
MODEL_TYPE="${MODEL_TYPE:-qwen2vl}"
case "${MODEL_TYPE}" in
  qwen2vl)
    DEFAULT_MODEL_PATH="ckpt/Qwen2-VL-7B-Instruct"
    DEFAULT_OUT_DIR="outputs/prem_attention/qwen2_7b"
    ;;
  qwen2_5vl)
    DEFAULT_MODEL_PATH="ckpt/Qwen2.5-VL-7B-Instruct"
    DEFAULT_OUT_DIR="outputs/prem_attention/qwen25_7b"
    ;;
  qwen3vl)
    DEFAULT_MODEL_PATH="ckpt/Qwen3-VL-8B-Instruct"
    DEFAULT_OUT_DIR="outputs/prem_attention/qwen3_8b"
    ;;
  llava_video)
    DEFAULT_MODEL_PATH="ckpt/LLaVA-Video-7B-Qwen2"
    DEFAULT_OUT_DIR="outputs/prem_attention/llava_video_7b"
    ;;
  *)
    echo "MODEL_TYPE must be qwen2vl, qwen2_5vl, qwen3vl, or llava_video" >&2
    exit 2
    ;;
esac
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATA_FILE="${DATA_FILE:-data/llava-video-178k/trainset_9k.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-data/llava-video-178k/frames}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
OUT_CKPT="${OUT_CKPT:-${OUT_DIR}/prem.pt}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/train.log}"
RESUME="${RESUME:-0}"

mkdir -p "${OUT_DIR}"

EXTRA_ARGS=(--no_skip_oom --skip_bad_samples)
if [[ "${DATA_FILE}" == *.jsonl ]]; then
  EXTRA_ARGS+=(--stream_jsonl)
fi
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
EXTRA_ARGS+=(--warmup_ratio "${WARMUP_RATIO}")
if [[ "${DISABLE_ANTI_DISTRACTOR:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_anti_distractor)
fi
if [[ "${DISABLE_NOVELTY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_novelty)
fi
if [[ "${DISABLE_STABILITY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_stability)
fi
if [[ "${DISABLE_EVIDENCE_GATE_WRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_evidence_gate_write)
fi
if [[ "${UNIFORM_WRITE_ROUTE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--uniform_write_route)
fi
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
  if [[ -n "${RESUME_CKPT:-}" ]]; then
    EXTRA_ARGS+=(--resume_ckpt "${RESUME_CKPT}")
  fi
elif [[ -e "${OUT_CKPT}" || -e "${OUT_CKPT}.latest" ]]; then
  echo "Refusing a fresh run over an existing checkpoint. Set OUT_CKPT to a new path or RESUME=1." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if command -v rtk >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  RUN_PREFIX=(rtk conda run --no-capture-output -n prem)
elif command -v conda >/dev/null 2>&1; then
  RUN_PREFIX=(conda run --no-capture-output -n prem)
else
  RUN_PREFIX=()
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUN_PREFIX=("${PYTHON_BIN}" -m torch.distributed.run)
  TORCHRUN_ARGS=()
else
  TORCHRUN_ARGS=(torchrun)
fi

# Full-video writer M + fixed-budget visual buffer B + one QA decoder path.
# Select the backbone API explicitly through MODEL_TYPE.
# Evidence prediction enabled via pred_weight=0.1, pred_tokens=8.
"${RUN_PREFIX[@]}" "${TORCHRUN_ARGS[@]}" \
  --nproc_per_node="${NGPUS}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --llava_data_file "${DATA_FILE}" \
  --llava_video_root "${VIDEO_ROOT}" \
  --out_ckpt "${OUT_CKPT}" \
  --fps "${FPS:-1.0}" \
  --dev_ratio "${DEV_RATIO:-0}" \
  --epochs "${EPOCHS:-1}" \
  --max_frames "${MAX_FRAMES:-64}" \
  --visual_buffer_frames "${VISUAL_BUFFER_FRAMES:-16}" \
  --max_pixels "${MAX_PIXELS:-200704}" \
  --max_memory_tokens "${MAX_MEMORY_TOKENS:-128}" \
  --router_gamma "${ROUTER_GAMMA:-0.05}" \
  --pred_weight "${PRED_WEIGHT:-0.1}" \
  --pred_tokens "${PRED_TOKENS:-8}" \
  --num_slots "${NUM_SLOTS:-4}" \
  --mem_dim "${MEM_DIM:-128}" \
  --prem_layer_groups "${PREM_LAYER_GROUPS:-1}" \
  --lr "${LR:-2e-4}" \
  --alpha "${ALPHA:-1.0}" \
  --seed "${SEED:-13}" \
  --prem_modulation "${PREM_MODULATION:-attention}" \
  --log_every "${LOG_EVERY:-20}" \
  --save_every "${SAVE_EVERY:-100}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
