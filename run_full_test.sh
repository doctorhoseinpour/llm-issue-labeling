#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_full_test.sh — Full RAGTAG+ experiment: all models × all datasets
# ============================================================================
# Usage:
#   # On NRP server (download models to tmp, delete after):
#   ./run_full_test.sh --nrp
#
#   # On local machine (use default HF cache):
#   ./run_full_test.sh
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- Configuration ---
DATASETS=("issues3k.csv" "issues30k.csv")
TOP_KS="1,3,9,15,21"
MAX_SEQ_LENGTH=16384
EMBED_MODEL="BAAI/bge-base-en-v1.5"

MODELS=(
  "unsloth/Llama-3.2-3B-Instruct:50"
  "unsloth/llama-3-8b-Instruct:50"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit:50"
  "unsloth/Llama-3.3-70B-Instruct-bnb-4bit:50"
)

# --- Parse args ---
NRP_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nrp) NRP_MODE=1; shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

CACHE_ARGS=()
if [[ "$NRP_MODE" -eq 1 ]]; then
  CACHE_ARGS=(--model_cache_dir "/tmp/ragtag_models_$$")
  echo ">>> NRP mode: models will download to /tmp (local disk)"
else
  echo ">>> Local mode: using default HF cache"
fi

# --- Run ---
for dataset in "${DATASETS[@]}"; do
  dataset_tag="${dataset%.csv}"

  if [[ ! -f "$SCRIPT_DIR/$dataset" ]]; then
    echo "WARNING: $dataset not found in $SCRIPT_DIR, skipping."
    continue
  fi

  # Build comma-separated model spec
  MODEL_SPEC=$(IFS=,; echo "${MODELS[*]}")

  RUN_DIR="$SCRIPT_DIR/results/${dataset_tag}"

  echo ""
  echo "============================================================"
  echo "  Dataset: $dataset"
  echo "  Models:  ${MODELS[*]}"
  echo "  K vals:  $TOP_KS"
  echo "  Output:  $RUN_DIR"
  echo "============================================================"

  "$SCRIPT_DIR/run_pipeline.sh" \
    --dataset "$SCRIPT_DIR/$dataset" \
    --top_ks "$TOP_KS" \
    --embedding_model "$EMBED_MODEL" \
    --llm_models "$MODEL_SPEC" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --results_dir "$RUN_DIR" \
    "${CACHE_ARGS[@]}"

done

echo ""
echo "All experiments complete."
