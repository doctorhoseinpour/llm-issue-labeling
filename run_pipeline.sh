#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_pipeline.sh — RAGTAG+ Pipeline Orchestrator (Unsloth version)
# ============================================================================
# 1. Build FAISS index ONCE, query all k values in one pass
# 2. Run LLM labeling via Unsloth (direct GPU inference, no Ollama)
# 3. Evaluate after each k: per-label + overall P/R/F1, invalid rate
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python3 or python is required." >&2
    exit 1
  fi
fi

usage() {
  cat <<'USAGE'
Usage: run_pipeline.sh [options]

Required:
  --dataset PATH              Full dataset CSV (retrieval pool = full dataset;
                              test set = first 50% of each label, no shuffle)

Options:
  --top_ks "1,3,9,15"         Comma-separated k values (default: 1,3,9,15)
  --embedding_model NAME      HuggingFace embedding model
                              (default: sentence-transformers/all-MiniLM-L6-v2)
  --llm_models SPEC           Comma-separated HuggingFace model specs.
                              Format: model_id[:max_new_tokens]
                              (default max_new_tokens from --max_new_tokens flag)
                              Examples:
                                "meta-llama/Llama-3.2-3B-Instruct"
                                "unsloth/DeepSeek-R1-Distill-Llama-8B:512"
                                "meta-llama/Llama-3.2-3B-Instruct:10,unsloth/DeepSeek-R1-Distill-Llama-8B:512"
  --max_seq_length N          Context window for LLM (default: 16384)
  --max_new_tokens N          Max tokens for LLM output (default: 20, use 512 for thinking models)
  --results_dir PATH          Output directory (default: results/run_<timestamp>)
  --model_cache_dir PATH      Directory to download HF models (default: HF default cache)
  --skip_retrieval            Skip index building (reuse existing neighbors)
  --skip_llm                  Skip LLM labeling (retrieval only)
  -h, --help                  Show this message

Example:
  ./run_pipeline.sh \
    --dataset issues3k.csv \
    --top_ks "1,3,9,15" \
    --llm_models "meta-llama/Llama-3.2-3B-Instruct"
USAGE
}

# Defaults
DATASET=""
TOP_KS="1,3,9,15"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODELS_SPEC=""
MAX_SEQ_LENGTH=16384
MAX_NEW_TOKENS=20
RESULTS_DIR=""
MODEL_CACHE_DIR=""
SKIP_RETRIEVAL=0
SKIP_LLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)         DATASET="$2";         shift 2 ;;
    --top_ks)          TOP_KS="$2";          shift 2 ;;
    --embedding_model) EMBED_MODEL="$2";     shift 2 ;;
    --llm_models)      LLM_MODELS_SPEC="$2"; shift 2 ;;
    --max_seq_length)  MAX_SEQ_LENGTH="$2";  shift 2 ;;
    --max_new_tokens)  MAX_NEW_TOKENS="$2";  shift 2 ;;
    --results_dir)     RESULTS_DIR="$2";     shift 2 ;;
    --model_cache_dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --skip_retrieval)  SKIP_RETRIEVAL=1;     shift ;;
    --skip_llm)        SKIP_LLM=1;           shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

# Validate dataset
if [[ -z "$DATASET" ]]; then
  echo "Error: provide --dataset" >&2
  exit 1
fi
[[ -f "$DATASET" ]] || { echo "Error: file not found: $DATASET" >&2; exit 1; }

# Resolve results dir
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$SCRIPT_DIR/results/run_${RUN_STAMP}"
fi

NEIGHBORS_DIR="$RESULTS_DIR/neighbors"
PREDS_DIR="$RESULTS_DIR/predictions"
EVAL_DIR="$RESULTS_DIR/evaluations"
LOGS_DIR="$RESULTS_DIR/logs"
TIMING_FILE="$RESULTS_DIR/timing.csv"

mkdir -p "$NEIGHBORS_DIR" "$PREDS_DIR" "$EVAL_DIR" "$LOGS_DIR"

# Parse k values
IFS=',' read -ra K_ARRAY <<< "$TOP_KS"

# Parse LLM model specs (format: model_id[:max_new_tokens])
if [[ -z "$LLM_MODELS_SPEC" ]]; then
  LLM_MODELS_SPEC="meta-llama/Llama-3.2-3B-Instruct"
fi
IFS=',' read -ra MODEL_IDS <<< "$LLM_MODELS_SPEC"

# Create safe tags and per-model max_new_tokens
declare -a MODEL_NAMES MODEL_TAGS MODEL_TOKENS
for spec in "${MODEL_IDS[@]}"; do
  # Split on last colon that's followed by only digits (to avoid splitting HF paths)
  if [[ "$spec" =~ ^(.+):([0-9]+)$ ]]; then
    m="${BASH_REMATCH[1]}"
    t="${BASH_REMATCH[2]}"
  else
    m="$spec"
    t="$MAX_NEW_TOKENS"
  fi
  tag="$(echo "$m" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"
  MODEL_NAMES+=("$m")
  MODEL_TAGS+=("$tag")
  MODEL_TOKENS+=("$t")
done

# Write timing header
echo "stage,model,top_k,seconds" > "$TIMING_FILE"

log_time() {
  echo "$1,$2,$3,$4" >> "$TIMING_FILE"
}

echo "=========================================================="
echo "  RAGTAG+ Pipeline (Unsloth)"
echo "=========================================================="
echo "  Dataset:         $DATASET"
echo "  K values:        ${K_ARRAY[*]}"
echo "  Embedding:       $EMBED_MODEL"
echo "  max_seq_length:  $MAX_SEQ_LENGTH"
echo "  max_new_tokens:  $MAX_NEW_TOKENS"
echo "  Results:         $RESULTS_DIR"
if [[ -n "$MODEL_CACHE_DIR" ]]; then
  echo "  Model cache:     $MODEL_CACHE_DIR"
fi
echo "  Models:"
for i in "${!MODEL_NAMES[@]}"; do
  echo "    - ${MODEL_TAGS[$i]} (${MODEL_NAMES[$i]}, max_new_tokens=${MODEL_TOKENS[$i]})"
done
echo "=========================================================="

PIPELINE_START=$(date +%s)

# ===================== Stage 1: Build Index & Retrieve =====================
if [[ "$SKIP_RETRIEVAL" -eq 0 ]]; then
  echo ""
  echo ">>> Stage 1: Building FAISS index and retrieving neighbors"
  echo "    (full dataset = retrieval pool; test = first 50% per label)"
  STAGE_START=$(date +%s)

  RETRIEVAL_EXTRA_ARGS=()
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    RETRIEVAL_EXTRA_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
  fi

  "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
    --dataset "$DATASET" \
    --top_ks "$TOP_KS" \
    --embedding_model "$EMBED_MODEL" \
    --output_dir "$NEIGHBORS_DIR" \
    "${RETRIEVAL_EXTRA_ARGS[@]}"

  STAGE_END=$(date +%s)
  log_time "retrieval" "-" "all" "$((STAGE_END - STAGE_START))"
  echo "  Stage 1 done in $((STAGE_END - STAGE_START))s"
else
  echo ""
  echo ">>> Stage 1: SKIPPED (--skip_retrieval)"
  for k in "${K_ARRAY[@]}"; do
    f="$NEIGHBORS_DIR/neighbors_k${k}.csv"
    [[ -f "$f" ]] || { echo "Error: neighbor file missing: $f" >&2; exit 1; }
  done
fi

# ===================== Stage 2: LLM Labeling + Evaluation =================
if [[ "$SKIP_LLM" -eq 0 ]]; then
  echo ""
  echo ">>> Stage 2: LLM labeling + evaluation (per model, all K values in one load)"
  for i in "${!MODEL_NAMES[@]}"; do
    model="${MODEL_NAMES[$i]}"
    tag="${MODEL_TAGS[$i]}"
    model_max_tokens="${MODEL_TOKENS[$i]}"

    model_pred_dir="$PREDS_DIR/$tag"
    model_eval_dir="$EVAL_DIR/$tag"
    model_log_dir="$LOGS_DIR/$tag"
    mkdir -p "$model_pred_dir" "$model_eval_dir" "$model_log_dir"

    # Build K list: 0 (zero-shot) + all K values
    ALL_KS="0,${TOP_KS}"

    echo ""
    echo "  -- ${tag}: loading model ONCE, running K=${ALL_KS} (max_new_tokens=${model_max_tokens})"
    STAGE_START=$(date +%s)

    LLM_EXTRA_ARGS=()
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
      LLM_EXTRA_ARGS+=(--cache_dir "$MODEL_CACHE_DIR")
    fi

    "$PYTHON_BIN" "$SCRIPT_DIR/llm_labeler.py" \
      --model "$model" \
      --neighbors_dir "$NEIGHBORS_DIR" \
      --top_ks "$ALL_KS" \
      --output_dir "$model_pred_dir" \
      --log_dir "$model_log_dir" \
      --eval_dir "$model_eval_dir" \
      --model_name_for_eval "$model" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --max_new_tokens "$model_max_tokens" \
      "${LLM_EXTRA_ARGS[@]}"

    STAGE_END=$(date +%s)
    log_time "llm_labeling" "$tag" "all" "$((STAGE_END - STAGE_START))"

  done
else
  echo ""
  echo ">>> Stage 2: SKIPPED (--skip_llm)"
fi

# ===================== Aggregate all evaluations ==========================
echo ""
echo ">>> Aggregating results"
"$PYTHON_BIN" - "$EVAL_DIR" "$RESULTS_DIR/all_results.csv" <<'PYAGG'
import sys, os, pandas as pd
eval_dir, out_path = sys.argv[1], sys.argv[2]
dfs = []
for root, dirs, files in os.walk(eval_dir):
    for f in files:
        if f.startswith("eval_") and f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(root, f)))
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  Wrote aggregated results: {out_path}")
    print(combined.to_string(index=False))
else:
    print("  No evaluation results found.")
PYAGG

PIPELINE_END=$(date +%s)
TOTAL=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "=========================================================="
echo "  Pipeline complete in ${TOTAL}s"
echo "=========================================================="
echo "  Neighbors:     $NEIGHBORS_DIR"
echo "  Predictions:   $PREDS_DIR"
echo "  Evaluations:   $EVAL_DIR"
echo "  Timing:        $TIMING_FILE"
echo "  All results:   $RESULTS_DIR/all_results.csv"
echo "=========================================================="