#!/usr/bin/env bash
# Table 3B offline ablation. Recipe matches scripts/run_prem_kv_qwen25_3b.sh.
# Full-model row is copied from Table 2, not re-evaluated here.
set -euo pipefail

PREM_DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) PREM_DRY_RUN=1 ;;
  *) echo "Usage: bash scripts/run_ablation.sh [--dry-run]" >&2; exit 2 ;;
esac
export PREM_DRY_RUN
FORCE="${FORCE:-1}"
export FORCE
source "$(dirname "${BASH_SOURCE[0]}")/lib/prem_experiment.sh"
prem_pin_table2_recipe

BACKBONES="${BACKBONES:-qwen25}"
VARIANTS="${VARIANTS:-qo k_only v_only no_salience no_novelty no_stability no_anti_overwrite b0_memory b0_no_memory}"
VARIANTS="${VARIANTS//,/ }"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ANALYSIS_OUTPUT_ROOT}/ablation}"

read -r -a backbones <<< "${BACKBONES}"
read -r -a variants <<< "${VARIANTS}"

ablation_root() {
  printf '%s/%s/%s/%s\n' "${OUTPUT_ROOT}" "${PREM_BACKBONE}" "${PREM_CFG_TAG}" "$1"
}

run_train_ablation() {
  local variant="$1" modulation="$2" budget="$3"
  shift 3
  local root
  root="$(ablation_root "${variant}")"
  prem_train_current "${root}" "${budget}" "${modulation}" "$@"
  prem_eval_offline "${root}/prem.pt" "${root}/offline" \
    "${PREM_EVAL_ALPHA}" "${modulation}" "$@"
}

for backbone in "${backbones[@]}"; do
  prem_configure_backbone "${backbone}" 16
  for variant in "${variants[@]}"; do
    echo "=== Ablation: ${backbone} ${variant} ==="
    case "${variant}" in
      qo) run_train_ablation qo attention 16 ;;
      k_only) run_train_ablation k_only attention_k 16 ;;
      v_only) run_train_ablation v_only attention_v 16 ;;
      no_salience)
        run_train_ablation no_salience attention_kv 16 \
          DISABLE_EVIDENCE_GATE_WRITE=1 PREM_DISABLE_EVIDENCE_GATE_WRITE=1
        ;;
      no_novelty)
        run_train_ablation no_novelty attention_kv 16 \
          DISABLE_NOVELTY=1 PREM_DISABLE_NOVELTY=1
        ;;
      no_stability)
        run_train_ablation no_stability attention_kv 16 \
          DISABLE_STABILITY=1 PREM_DISABLE_STABILITY=1
        ;;
      no_anti_overwrite)
        run_train_ablation no_anti_overwrite attention_kv 16 \
          DISABLE_ANTI_DISTRACTOR=1 PREM_DISABLE_ANTI_DISTRACTOR=1
        ;;
      b0_memory) run_train_ablation b0_memory attention_kv 0 ;;
      b0_no_memory)
        prem_eval_text_only "${PREM_MAIN_CKPT}" \
          "$(ablation_root b0_no_memory)/offline"
        ;;
      uniform_write_route)
        echo "uniform_write_route is a no-op with num_slots=1 and is not a current ablation." >&2
        exit 2
        ;;
      *) echo "Unknown VARIANTS value: ${variant}" >&2; exit 2 ;;
    esac
  done
done
