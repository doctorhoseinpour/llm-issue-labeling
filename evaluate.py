#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate LLM predictions against ground truth.
Outputs:
  - Per-label precision, recall, F1
  - Overall (macro/weighted) precision, recall, F1
  - Invalid prediction count and rate
  - Timing and resource info (if provided)

Usage:
  python evaluate.py \
    --preds_csv results/predictions/preds_k9_deepseek.csv \
    --output_csv results/evaluations/eval_k9_deepseek.csv \
    --model_name deepseek-r1:8b \
    --top_k 9

  # Or evaluate all k values for one model at once:
  python evaluate.py \
    --preds_dir results/predictions/ \
    --preds_pattern "preds_k{K}_deepseek.csv" \
    --top_ks 1,3,9,15 \
    --output_csv results/evaluations/eval_deepseek_all_ks.csv \
    --model_name deepseek-r1:8b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

VALID_LABELS = ["bug", "feature", "question"]


def evaluate_predictions(
    df: pd.DataFrame,
    model_name: str = "",
    top_k: int = 0,
) -> dict:
    """
    Evaluate a single predictions DataFrame.
    Returns a dict with all metrics.
    """
    if "ground_truth" not in df.columns or "predicted_label" not in df.columns:
        raise ValueError(f"Need 'ground_truth' and 'predicted_label' columns. Got: {list(df.columns)}")

    y_true = df["ground_truth"].astype(str).str.lower().str.strip().tolist()
    y_pred = df["predicted_label"].astype(str).str.lower().str.strip().tolist()
    total = len(y_true)

    # Count invalids
    n_invalid = sum(1 for p in y_pred if p == "invalid")
    invalid_rate = n_invalid / total if total > 0 else 0.0

    # For sklearn metrics, treat "invalid" as a wrong prediction.
    # It won't match any of the three valid labels, so precision/recall
    # naturally penalise the model for invalid outputs.
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, zero_division=0
    )

    # Macro averages (equal weight per class — appropriate for balanced labels)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, average="macro", zero_division=0
    )

    # Weighted averages
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, average="weighted", zero_division=0
    )

    acc = accuracy_score(
        y_true, y_pred,
    )

    result = {
        "model": model_name,
        "top_k": top_k,
        "total_issues": total,
        "invalid_count": n_invalid,
        "invalid_rate": round(invalid_rate, 4),
        "accuracy": round(acc, 4),
    }

    # Per-label metrics
    for i, label in enumerate(VALID_LABELS):
        result[f"precision_{label}"] = round(precision_per[i], 4)
        result[f"recall_{label}"] = round(recall_per[i], 4)
        result[f"f1_{label}"] = round(f1_per[i], 4)
        result[f"support_{label}"] = int(support_per[i])

    # Aggregate metrics
    result["precision_macro"] = round(macro_p, 4)
    result["recall_macro"] = round(macro_r, 4)
    result["f1_macro"] = round(macro_f1, 4)
    result["precision_weighted"] = round(weighted_p, 4)
    result["recall_weighted"] = round(weighted_r, 4)
    result["f1_weighted"] = round(weighted_f1, 4)

    return result


def print_report(df: pd.DataFrame, model_name: str, top_k: int):
    """Print a readable classification report to stdout."""
    y_true = df["ground_truth"].astype(str).str.lower().str.strip().tolist()
    y_pred = df["predicted_label"].astype(str).str.lower().str.strip().tolist()
    n_invalid = sum(1 for p in y_pred if p == "invalid")

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  top_k: {top_k}")
    print(f"  Total issues: {len(y_true)}  |  Invalid outputs: {n_invalid} ({100*n_invalid/len(y_true):.1f}%)")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred, labels=VALID_LABELS, target_names=VALID_LABELS, zero_division=0
    ))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions")

    # Single file mode
    parser.add_argument("--preds_csv", default=None, help="Single predictions CSV to evaluate")
    parser.add_argument("--top_k", type=int, default=0, help="K value for the single CSV (for labeling)")

    # Multi-k mode
    parser.add_argument("--preds_dir", default=None, help="Directory containing prediction CSVs")
    parser.add_argument("--preds_pattern", default="preds_k{K}.csv",
                        help="Filename pattern with {K} placeholder")
    parser.add_argument("--top_ks", default=None, help="Comma-separated k values for multi-k mode")

    # Output
    parser.add_argument("--output_csv", required=True, help="Output evaluation summary CSV")
    parser.add_argument("--model_name", default="", help="Model name label for the output")

    args = parser.parse_args()

    all_results = []

    if args.preds_csv:
        # Single file mode
        df = pd.read_csv(args.preds_csv)
        print_report(df, args.model_name, args.top_k)
        result = evaluate_predictions(df, args.model_name, args.top_k)
        all_results.append(result)

    elif args.preds_dir and args.top_ks:
        # Multi-k mode
        ks = [int(x) for x in args.top_ks.split(",")]
        for k in ks:
            filename = args.preds_pattern.replace("{K}", str(k))
            filepath = os.path.join(args.preds_dir, filename)
            if not os.path.exists(filepath):
                print(f"  [WARN] Missing: {filepath}")
                continue
            df = pd.read_csv(filepath)
            print_report(df, args.model_name, k)
            result = evaluate_predictions(df, args.model_name, k)
            all_results.append(result)
    else:
        parser.error("Provide either --preds_csv or both --preds_dir and --top_ks")

    # Write summary
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_path, index=False)
    print(f"\nEvaluation summary written to: {out_path}")


if __name__ == "__main__":
    main()
