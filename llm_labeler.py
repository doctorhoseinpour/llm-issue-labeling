#!/usr/bin/env python3
"""
llm_labeler.py
==============
RAG few-shot LLM labeler — loads model ONCE, runs all K values in a single process.

Architecture:
  1. CHAT TEMPLATE — native chat format via apply_chat_template()
  2. FEW-SHOT FORMAT — neighbors become user/assistant turns with <label>X</label>
  3. ASSISTANT PREFILL — for instruct models, prefill "<label>" so model only
     completes the label word + closing tag
  4. XML PARSING + REGEX FALLBACK

Usage:
  # Single K:
  python llm_labeler.py --model unsloth/Llama-3.2-3B-Instruct \
    --neighbors_dir results/neighbors --top_ks "1,3,9,15" \
    --output_dir results/predictions/llama3b --max_seq_length 16384

  # With zero-shot included:
  python llm_labeler.py --model unsloth/Llama-3.2-3B-Instruct \
    --neighbors_dir results/neighbors --top_ks "0,1,3,9,15" \
    --output_dir results/predictions/llama3b --max_seq_length 16384
    (k=0 means zero-shot)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_LABELS = ["bug", "feature", "question"]
VALID_LABELS_SET = set(VALID_LABELS)

_CANON_MAP = {
    "enhancement": "feature", "feature-request": "feature",
    "feature_request": "feature", "feat": "feature", "request": "feature",
    "bugfix": "bug", "defect": "bug", "issue": "bug", "fix": "bug",
    "support": "question", "howto": "question", "help": "question",
}

SYSTEM_PROMPT = """Classify the GitHub issue into exactly one category.

Rules:
1. Read the issue title and body.
2. Choose one label: bug, feature, or question.
3. Respond with ONLY the label wrapped in XML tags.
4. Do NOT write anything else. No explanation. No reasoning. No extra text.

Correct response format examples:
<label>bug</label>
<label>feature</label>
<label>question</label>"""

SYSTEM_PROMPT_THINKING = """Classify the GitHub issue into exactly one category.

Rules:
1. Read the issue title and body.
2. Choose one label: bug, feature, or question.
3. You may reason about your choice first.
4. You MUST end your response with the label in XML tags.

Correct format for your final answer:
<label>bug</label>"""


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _strip_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s


def _is_label_list(text: str) -> bool:
    start = text[:80].lower().strip()
    return bool(re.search(r'bug[,\s]+feature[,\s]+(or\s+)?question', start) or
                re.search(r'feature[,\s]+bug[,\s]+(or\s+)?question', start) or
                re.search(r'bug[,\s]+question[,\s]+(or\s+)?feature', start))


def parse_label(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "invalid"

    # Layer 1: XML tag — take the LAST <label>...</label>
    label_matches = re.findall(r"<label>\s*(.*?)\s*</label>", raw, re.IGNORECASE | re.DOTALL)
    if label_matches:
        candidate = label_matches[-1].strip().lower()
        if candidate in VALID_LABELS_SET:
            return candidate
        if candidate in _CANON_MAP:
            return _CANON_MAP[candidate]
        squash = re.sub(r"[^a-z]", "", candidate)
        for v in VALID_LABELS:
            if v == squash:
                return v

    # Layer 2: Strip think blocks + regex
    s = _strip_think(raw).strip()
    if not s:
        return "invalid"

    if _is_label_list(s):
        after = re.search(r'bug[,\s]+feature[,\s]+(?:or\s+)?question[.\s]*\n*(.*)',
                          s, re.DOTALL | re.IGNORECASE)
        if after:
            remainder = after.group(1).strip()
            tag_match = re.search(r"<label>\s*(.*?)\s*</label>", remainder, re.IGNORECASE)
            if tag_match:
                t = tag_match.group(1).strip().lower()
                if t in VALID_LABELS_SET:
                    return t
            for tok in re.findall(r"[A-Za-z_\-]+", remainder):
                t = tok.lower().strip("-_ ")
                if t in VALID_LABELS_SET:
                    return t
                if t in _CANON_MAP:
                    return _CANON_MAP[t]
        return "invalid"

    # Layer 3: First valid word
    tokens = re.findall(r"[A-Za-z_\-]+", s)
    if not tokens:
        return "invalid"
    tok = tokens[0].lower().strip("-_ ")
    if tok in VALID_LABELS_SET:
        return tok
    if tok in _CANON_MAP:
        return _CANON_MAP[tok]
    squash = re.sub(r"[^a-z]", "", tok)
    for v in VALID_LABELS:
        if v == squash:
            return v
    for k, v in _CANON_MAP.items():
        if re.sub(r"[^a-z]", "", k) == squash:
            return v
    return "invalid"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def _truncate_text_by_tokens(text, max_tokens, tokenizer):
    if not text:
        return "", 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, 0
    removed = len(token_ids) - max_tokens
    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).rstrip() + "...", removed


@dataclass
class TruncationInfo:
    truncated: bool = False
    neighbors_truncated: bool = False
    query_truncated: bool = False
    original_tokens: int = 0
    final_tokens: int = 0
    tokens_removed: int = 0


def _count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Chat message building
# ---------------------------------------------------------------------------

def build_chat_messages(test_title, test_body, neighbors, k, is_thinking_model,
                        max_prompt_tokens, tokenizer):
    trunc = TruncationInfo()
    system = SYSTEM_PROMPT_THINKING if is_thinking_model else SYSTEM_PROMPT

    def format_issue(title, body):
        return f"Title: {title}\nBody: {body}"

    def format_label(label):
        return f"<label>{label}</label>"

    sys_tokens = _count_tokens(system, tokenizer)

    # CHANGED: We now only have ONE system turn and ONE user turn.
    # We no longer pay the overhead tax for 2*K extra chat turns.
    total_overhead = sys_tokens + 10

    neighbor_data = []
    for nb in neighbors:
        t = str(nb.get("title", ""))
        b = str(nb.get("body", ""))
        lab = str(nb.get("label", "")).strip().lower()
        if lab in _CANON_MAP:
            lab = _CANON_MAP[lab]
        if lab not in VALID_LABELS_SET:
            lab = "bug"
        issue_text = format_issue(t, b)
        neighbor_data.append({
            "title": t, "body": b, "label": lab,
            "issue_text": issue_text,
            "issue_tokens": _count_tokens(issue_text, tokenizer),
            "body_tokens": _count_tokens(b, tokenizer),
            "title_tokens": _count_tokens(t, tokenizer),
        })

    test_issue_text = format_issue(test_title, test_body)
    test_tokens = _count_tokens(test_issue_text, tokenizer)

    total_content_tokens = test_tokens + sum(nd["issue_tokens"] for nd in neighbor_data)
    trunc.original_tokens = total_content_tokens
    budget = max(100, max_prompt_tokens - total_overhead)

    # --- Truncation Logic (Unchanged) ---
    if total_content_tokens > budget:
        trunc.truncated = True
        query_reserve = int(budget * 0.3)
        neighbor_budget = budget - query_reserve
        total_nb_title = sum(nd["title_tokens"] for nd in neighbor_data)
        nb_body_budget = neighbor_budget - total_nb_title
        total_nb_body = sum(nd["body_tokens"] for nd in neighbor_data)

        if nb_body_budget > 0 and total_nb_body > nb_body_budget:
            trunc.neighbors_truncated = True
            for nd in neighbor_data:
                ratio = nd["body_tokens"] / total_nb_body if total_nb_body > 0 else 1.0 / max(1, len(neighbor_data))
                max_b = max(5, int(nb_body_budget * ratio))
                if nd["body_tokens"] > max_b:
                    nd["body"], _ = _truncate_text_by_tokens(nd["body"], max_b, tokenizer)
                    nd["body_tokens"] = max_b
                nd["issue_text"] = format_issue(nd["title"], nd["body"])
                nd["issue_tokens"] = nd["title_tokens"] + nd["body_tokens"]
        elif nb_body_budget <= 0:
            trunc.neighbors_truncated = True
            for nd in neighbor_data:
                nd["body"], _ = _truncate_text_by_tokens(nd["body"], 5, tokenizer)
                nd["body_tokens"] = min(5, nd["body_tokens"])
                nd["issue_text"] = format_issue(nd["title"], nd["body"])
                nd["issue_tokens"] = nd["title_tokens"] + nd["body_tokens"]

        used = sum(nd["issue_tokens"] for nd in neighbor_data)
        q_budget = budget - used
        if test_tokens > q_budget and q_budget > 0:
            trunc.query_truncated = True
            tt_tokens = _count_tokens(test_title, tokenizer)
            bb = q_budget - tt_tokens
            if bb > 10:
                test_body, _ = _truncate_text_by_tokens(test_body, bb, tokenizer)
            else:
                tb = max(5, int(q_budget * 0.4))
                bb = max(5, q_budget - tb)
                test_title, _ = _truncate_text_by_tokens(test_title, tb, tokenizer)
                test_body, _ = _truncate_text_by_tokens(test_body, bb, tokenizer)
            test_issue_text = format_issue(test_title, test_body)
            test_tokens = _count_tokens(test_issue_text, tokenizer)

    trunc.final_tokens = test_tokens + sum(nd["issue_tokens"] for nd in neighbor_data)
    trunc.tokens_removed = trunc.original_tokens - trunc.final_tokens

    # --- CHANGED: Building the Messages Array ---
    messages = [{"role": "system", "content": system}]

    # We build a single string for the user message
    user_content = ""

    if neighbor_data:
        user_content += "Here are some examples of correctly classified issues:\n\n"
        for i, nd in enumerate(neighbor_data, 1):
            user_content += f"--- Example {i} ---\n{nd['issue_text']}\nAnswer: {format_label(nd['label'])}\n\n"
        user_content += "Now, classify the following target issue:\n\n"

    user_content += test_issue_text

    messages.append({"role": "user", "content": user_content})

    return messages, trunc

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class TestIssue:
    idx: int
    title: str
    body: str
    label: str
    created_at: str
    neighbors: List[Dict[str, str]] = field(default_factory=list)


def load_test_issues(csv_path: str, k: int) -> List[TestIssue]:
    df = pd.read_csv(csv_path)
    issues: Dict[int, TestIssue] = {}
    for _, row in df.iterrows():
        ti = int(row["test_idx"])
        if ti not in issues:
            issues[ti] = TestIssue(
                idx=ti,
                title=str(row.get("test_title", "")),
                body=str(row.get("test_body", "")),
                label=str(row.get("test_label", "")),
                created_at=str(row.get("test_created_at", "")),
            )
        if row.get("neighbor_rank") is not None and int(row["neighbor_rank"]) < k:
            issues[ti].neighbors.append({
                "title": str(row.get("neighbor_title", "")),
                "body": str(row.get("neighbor_body", "")),
                "label": str(row.get("neighbor_label", "")),
            })
    return [issues[k_] for k_ in sorted(issues.keys())]


def print_gpu_stats(stage):
    if torch.cuda.is_available():
        print(f"  GPU [{stage}]: peak={torch.cuda.max_memory_allocated()/(1024**3):.2f}GB")


# ---------------------------------------------------------------------------
# Run inference for one K value
# ---------------------------------------------------------------------------

def run_one_k(
    test_issues: List[TestIssue],
    k: int,
    is_zero_shot: bool,
    model,
    tokenizer,
    is_thinking_model: bool,
    max_new_tokens: int,
    max_prompt_tokens: int,
    output_csv: str,
    log_file: Optional[str],
):
    """Run inference for a single K value (or zero-shot). Model is already loaded."""

    mode_label = "zero-shot" if is_zero_shot else f"k={k}"

    log_fh = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "w", encoding="utf-8")

    results = []
    t0 = time.time()
    n_truncated = n_nb_truncated = n_q_truncated = 0
    n_xml_parsed = n_regex_parsed = n_invalid = 0

    print(f"\n  [{mode_label}] Starting inference: {len(test_issues)} issues")

    for issue in tqdm(test_issues, desc=f"  {mode_label}", unit="issue"):
        neighbors_for_prompt = issue.neighbors[:k] if not is_zero_shot else []

        messages, trunc = build_chat_messages(
            test_title=issue.title,
            test_body=issue.body,
            neighbors=neighbors_for_prompt,
            k=k,
            is_thinking_model=is_thinking_model,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )

        if trunc.truncated:
            n_truncated += 1
        if trunc.neighbors_truncated:
            n_nb_truncated += 1
        if trunc.query_truncated:
            n_q_truncated += 1

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Prefill for instruct models
        prefilled = False
        if not is_thinking_model:
            prompt = prompt + "<label>"
            prefilled = True

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        try:
            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if is_thinking_model:
                    gen_kwargs.update({"temperature": 0.6, "top_p": 0.95, "top_k": 50})
                else:
                    gen_kwargs.update({"temperature": 0.1, "top_p": 0.9, "top_k": 50})

                outputs = model.generate(**gen_kwargs)

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if prefilled:
                raw_output = "<label>" + raw_output
        except Exception as e:
            raw_output = f"ERROR: {e}"

        pred = parse_label(raw_output)

        has_xml = bool(re.search(r"<label>.*?</label>", raw_output, re.IGNORECASE))
        if pred == "invalid":
            n_invalid += 1
        elif has_xml:
            n_xml_parsed += 1
        else:
            n_regex_parsed += 1

        results.append({
            "test_idx": issue.idx,
            "title": issue.title,
            "body": issue.body,
            "ground_truth": issue.label,
            "predicted_label": pred,
            "raw_output": raw_output[:300],
            "truncated": trunc.truncated,
            "neighbors_truncated": trunc.neighbors_truncated,
            "query_truncated": trunc.query_truncated,
            "tokens_removed": trunc.tokens_removed,
            "parsed_via": "xml" if (has_xml and pred != "invalid") else ("regex" if pred != "invalid" else "failed"),
        })

        if log_fh:
            log_fh.write(json.dumps({
                "test_idx": issue.idx, "raw_output": raw_output[:500],
                "parsed_label": pred, "ground_truth": issue.label,
                "truncated": trunc.truncated, "tokens_removed": trunc.tokens_removed,
            }) + "\n")

    if log_fh:
        log_fh.close()

    elapsed = time.time() - t0

    # Write output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["test_idx", "title", "body", "ground_truth", "predicted_label",
            "raw_output", "truncated", "neighbors_truncated", "query_truncated",
            "tokens_removed", "parsed_via"]
    df = pd.DataFrame(results)
    df[cols].to_csv(out_path, index=False)

    total = len(df)
    print(f"  [{mode_label}] Done: {total} predictions -> {out_path}")
    print(f"    XML parsed: {n_xml_parsed} ({100*n_xml_parsed/total:.1f}%)  "
          f"Regex: {n_regex_parsed} ({100*n_regex_parsed/total:.1f}%)  "
          f"Invalid: {n_invalid} ({100*n_invalid/total:.1f}%)")
    print(f"    Truncated: {n_truncated} ({100*n_truncated/total:.1f}%)")
    print(f"    Time: {elapsed:.1f}s ({total/elapsed:.1f} issues/s)")

    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG few-shot LLM labeler — loads model ONCE, runs all K values"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--neighbors_dir", required=True,
                        help="Directory containing neighbors_k{K}.csv files")
    parser.add_argument("--top_ks", required=True,
                        help="Comma-separated K values (use 0 for zero-shot)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for prediction CSVs")
    parser.add_argument("--log_dir", default=None,
                        help="Directory for JSONL log files")
    parser.add_argument("--eval_dir", default=None,
                        help="Directory for evaluation CSVs (runs evaluate.py after each K)")
    parser.add_argument("--model_name_for_eval", default=None,
                        help="Model name label for evaluation output")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--thinking_model", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to download/cache HuggingFace models (default: HF default cache)")
    args = parser.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False

    ks = [int(x) for x in args.top_ks.split(",")]

    # Auto-detect thinking model
    model_lower = args.model.lower()
    if any(x in model_lower for x in ["deepseek-r1", "deepseek_r1"]):
        if not args.thinking_model:
            print("  Auto-detected thinking model (DeepSeek R1)")
            args.thinking_model = True
        if args.max_new_tokens <= 20:
            print("  Auto-setting max_new_tokens=512 for thinking model")
            args.max_new_tokens = 512

    print(f"{'='*60}")
    print(f"  LLM Labeler (single model load, multi-K)")
    print(f"{'='*60}")
    print(f"  Model:           {args.model}")
    print(f"  K values:        {ks}")
    print(f"  max_seq_length:  {args.max_seq_length}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  thinking_model:  {args.thinking_model}")
    print(f"  load_in_4bit:    {args.load_in_4bit}")
    print(f"{'='*60}")

    # --- Load model ONCE ---
    print(f"\nLoading model: {args.model}")
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"  Model cache dir: {args.cache_dir}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_gpu_stats("model loaded")

    max_prompt_tokens = args.max_seq_length - args.max_new_tokens - 20
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Find the max K neighbor file (for loading test issues) ---
    real_ks = [k for k in ks if k > 0]
    max_k = max(real_ks) if real_ks else 1

    # Load test issues with max_k neighbors (we'll slice per-k)
    max_k_file = os.path.join(args.neighbors_dir, f"neighbors_k{max_k}.csv")
    if not os.path.exists(max_k_file):
        # Try to find any neighbor file
        for k in sorted(real_ks, reverse=True):
            f = os.path.join(args.neighbors_dir, f"neighbors_k{k}.csv")
            if os.path.exists(f):
                max_k_file = f
                max_k = k
                break

    print(f"\nLoading test issues from {max_k_file} (max_k={max_k})...")
    test_issues = load_test_issues(max_k_file, max_k)
    print(f"  Loaded {len(test_issues)} test issues with up to {max_k} neighbors each")

    # --- Run each K value ---
    total_time = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")

    for k in ks:
        is_zero_shot = (k == 0)
        k_label = "zero_shot" if is_zero_shot else f"k{k}"

        output_csv = os.path.join(args.output_dir, f"preds_{k_label}.csv")
        log_file = os.path.join(args.log_dir, f"{k_label}.jsonl") if args.log_dir else None

        elapsed = run_one_k(
            test_issues=test_issues,
            k=k,
            is_zero_shot=is_zero_shot,
            model=model,
            tokenizer=tokenizer,
            is_thinking_model=args.thinking_model,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            output_csv=output_csv,
            log_file=log_file,
        )
        total_time += elapsed

        # Evaluate immediately after each K
        if args.eval_dir and os.path.exists(eval_script) and os.path.exists(output_csv):
            import subprocess
            eval_csv = os.path.join(args.eval_dir, f"eval_{k_label}.csv")
            eval_model_name = args.model_name_for_eval or args.model
            print(f"  [{k_label}] Evaluating...")
            subprocess.run([
                sys.executable, eval_script,
                "--preds_csv", output_csv,
                "--top_k", str(k),
                "--output_csv", eval_csv,
                "--model_name", eval_model_name,
            ], check=False)

    print_gpu_stats("all done")
    print(f"\nAll K values complete. Total inference time: {total_time:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")


if __name__ == "__main__":
    main()