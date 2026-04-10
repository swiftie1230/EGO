#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two JSONL generations (e.g., GPT vs Qwen-72B) for GGB / SUPER_SCOTUS,
then select the better one using a Hugging Face judge model:
    Qwen/Qwen2.5-7B-Instruct

Supported datasets:
- GGB:
    compare base.vignette + bundle.items
- SUPER_SCOTUS:
    compare base.scenario + bundle.items

Input examples:
- ggb_skeleton_gpt-4.1-mini.jsonl
- ggb_skeleton_qwen2.5-72b-inst.jsonl
- SCOTUS_skeleton_gpt-4.1-mini.jsonl
- SCOTUS_skeleton_qwen2.5-72b-inst.jsonl

Output:
- selected JSONL
- optional audit JSONL with judge decision / score / reasons

Usage example:
python select_best_candidates.py \
  --dataset GGB \
  --file_a ggb_skeleton_gpt-4.1-mini.jsonl \
  --file_b ggb_skeleton_qwen2.5-72b-inst.jsonl \
  --output_jsonl ggb_selected_qwen7b_judge.jsonl \
  --audit_jsonl ggb_selected_qwen7b_judge.audit.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --max_new_tokens 32 \
  --device cuda

python select_best_candidates.py \
  --dataset SUPER_SCOTUS \
  --file_a SCOTUS_skeleton_gpt-4.1-mini.jsonl \
  --file_b SCOTUS_skeleton_qwen2.5-72b-inst.jsonl \
  --output_jsonl scotus_selected_qwen7b_judge.jsonl \
  --audit_jsonl scotus_selected_qwen7b_judge.audit.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --max_new_tokens 32 \
  --device cuda
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# IO utils
# =========================================================

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"[JSONL parse error] {path}:{line_no} -> {e}") from e


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index(path: str):
    out = {}
    for row in iter_jsonl(path):
        rid = row.get("id")
        option_order = row.get("option_order", "original")

        if not rid:
            continue

        key = (rid, option_order)

        if key in out:
            raise ValueError(f"Duplicate key in {path}: {key}")

        out[key] = row
    return out


# =========================================================
# Basic accessors
# =========================================================

def get_option_text_map(row: Dict[str, Any]) -> Dict[str, str]:
    items = row.get("bundle", {}).get("items", [])
    out = {}
    for item in items:
        item_id = item.get("item_id")
        text = item.get("text", "")
        if item_id:
            out[item_id] = text
    return out


def extract_candidate_fields(row: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    dataset = dataset.upper()

    if dataset == "GGB":
        main_text = row.get("base", {}).get("vignette", "")
    elif dataset == "SUPER_SCOTUS":
        main_text = row.get("base", {}).get("scenario", "")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    option_map = get_option_text_map(row)
    return {
        "main_text": main_text,
        "option_A": option_map.get("A", ""),
        "option_B": option_map.get("B", ""),
    }


def extract_original_context(row: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    dataset = dataset.upper()

    if dataset == "GGB":
        base = row.get("base", {})
        return {
            "type": base.get("type", ""),
            "statement": base.get("statement", ""),
            "core_phrases_preserved": base.get("core_phrases_preserved", []),
            "decision_question": row.get("decision_question", ""),
        }

    if dataset == "SUPER_SCOTUS":
        base = row.get("base", {})
        meta = row.get("meta", {})
        return {
            "case_title": base.get("case_title", "") or row.get("base", {}).get("case_title", ""),
            "scenario_seed": base.get("scenario", ""),
            "legal_issue": base.get("legal_issue", ""),
            "core_phrases_preserved": base.get("core_phrases_preserved", []),
            "decision_question": row.get("decision_question", ""),
            "petitioner": meta.get("petitioner", ""),
            "respondent": meta.get("respondent", ""),
            "year": meta.get("year", ""),
            "citation": meta.get("citation", ""),
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


# =========================================================
# Heuristics
# =========================================================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def word_count(text: str) -> int:
    return len(normalize_ws(text).split())


def char_count(text: str) -> int:
    return len(normalize_ws(text))


def is_nonempty_candidate(cand: Dict[str, Any]) -> bool:
    return all([
        bool(normalize_ws(cand.get("main_text", ""))),
        bool(normalize_ws(cand.get("option_A", ""))),
        bool(normalize_ws(cand.get("option_B", ""))),
    ])


def option_balance_score(option_a: str, option_b: str) -> float:
    """
    Smaller length gap => better.
    Returns a penalty score in [0, +inf), lower is better.
    """
    la = max(word_count(option_a), 1)
    lb = max(word_count(option_b), 1)
    return abs(la - lb) / max(la, lb)


def core_phrase_coverage_score(main_text: str, core_phrases: List[str]) -> float:
    """
    Fraction of core phrases appearing verbatim or near-verbatim (simple lowercase containment).
    """
    if not core_phrases:
        return 1.0
    text_l = normalize_ws(main_text).lower()
    hit = 0
    for p in core_phrases:
        p_l = normalize_ws(str(p)).lower()
        if p_l and p_l in text_l:
            hit += 1
    return hit / max(len(core_phrases), 1)


def cheap_compare(
    original_ctx: Dict[str, Any],
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
) -> Optional[str]:
    """
    Quick deterministic filter:
    - if one is empty and the other is not -> pick non-empty
    - if one has much better core phrase coverage and the other does not -> pick it
    - if one has clearly better option balance with otherwise similar coverage -> pick it
    Otherwise return None to defer to LLM judge.
    """
    a_ok = is_nonempty_candidate(cand_a)
    b_ok = is_nonempty_candidate(cand_b)

    if a_ok and not b_ok:
        return "A"
    if b_ok and not a_ok:
        return "B"
    if not a_ok and not b_ok:
        return "A"  # fallback

    core_phrases = original_ctx.get("core_phrases_preserved", [])
    a_cov = core_phrase_coverage_score(cand_a["main_text"], core_phrases)
    b_cov = core_phrase_coverage_score(cand_b["main_text"], core_phrases)

    if a_cov - b_cov >= 0.5:
        return "A"
    if b_cov - a_cov >= 0.5:
        return "B"

    a_bal = option_balance_score(cand_a["option_A"], cand_a["option_B"])
    b_bal = option_balance_score(cand_b["option_A"], cand_b["option_B"])

    # if coverage similar, prefer better-balanced options
    if abs(a_cov - b_cov) <= 0.15:
        if a_bal + 0.25 < b_bal:
            return "A"
        if b_bal + 0.25 < a_bal:
            return "B"

    return None


# =========================================================
# Prompt builders
# =========================================================

def build_prompt(dataset: str, original_ctx: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> str:
    dataset = dataset.upper()

    if dataset == "GGB":
        return f"""You are an expert dataset curator for moral dilemma decision-making tasks.

Your job is to compare two candidate rewrites and choose the BETTER one for a benchmark dataset.

You must prioritize:
1. Faithfulness to the original moral statement
2. Preservation of key facts and conditions
3. Clear and coherent vignette writing
4. Balanced and decision-appropriate options
5. No unnecessary embellishment or distortion

Original information:
- Type: {original_ctx.get("type", "")}
- Statement: {original_ctx.get("statement", "")}
- Core phrases to preserve: {original_ctx.get("core_phrases_preserved", [])}
- Decision question: {original_ctx.get("decision_question", "")}

Candidate A
[VIGNETTE]
{cand_a["main_text"]}

[OPTION A]
{cand_a["option_A"]}

[OPTION B]
{cand_a["option_B"]}

Candidate B
[VIGNETTE]
{cand_b["main_text"]}

[OPTION A]
{cand_b["option_A"]}

[OPTION B]
{cand_b["option_B"]}

Select the better candidate overall.

Output format:
Choice: A
Reason: <one short sentence>

or

Choice: B
Reason: <one short sentence>
"""

    if dataset == "SUPER_SCOTUS":
        return f"""You are an expert dataset curator for legal decision-making tasks.

Your job is to compare two candidate rewrites and choose the BETTER one for a benchmark dataset.

You must prioritize:
1. Faithfulness to the original legal scenario
2. Preservation of legal facts and procedural conditions
3. Clear and neutral case description
4. Balanced decision options
5. No added legal claims not supported by the original

Original information:
- Case title: {original_ctx.get("case_title", "")}
- Scenario seed: {original_ctx.get("scenario_seed", "")}
- Legal issue: {original_ctx.get("legal_issue", "")}
- Core phrases to preserve: {original_ctx.get("core_phrases_preserved", [])}
- Petitioner: {original_ctx.get("petitioner", "")}
- Respondent: {original_ctx.get("respondent", "")}
- Year: {original_ctx.get("year", "")}
- Citation: {original_ctx.get("citation", "")}
- Decision question: {original_ctx.get("decision_question", "")}

Candidate A
[SCENARIO]
{cand_a["main_text"]}

[OPTION A]
{cand_a["option_A"]}

[OPTION B]
{cand_a["option_B"]}

Candidate B
[SCENARIO]
{cand_b["main_text"]}

[OPTION A]
{cand_b["option_A"]}

[OPTION B]
{cand_b["option_B"]}

Select the better candidate overall.

Output format:
Choice: A
Reason: <one short sentence>

or

Choice: B
Reason: <one short sentence>
"""

    raise ValueError(f"Unsupported dataset: {dataset}")


# =========================================================
# Judge model
# =========================================================

class HFJudge:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 32,
        attn_implementation: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            ).to(device)

        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise and strict dataset judge."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()

    def judge(self, prompt: str) -> Tuple[str, str, str]:
        raw = self.generate(prompt)
        choice, reason = parse_choice_and_reason(raw)
        return choice, reason, raw


def parse_choice_and_reason(text: str) -> Tuple[str, str]:
    """
    Robust parsing:
    - Choice: A / B
    - fallback to first standalone A/B
    """
    text = text.strip()

    m = re.search(r"Choice\s*:\s*([AB])\b", text, flags=re.I)
    if m:
        choice = m.group(1).upper()
        m2 = re.search(r"Reason\s*:\s*(.+)", text, flags=re.I | re.S)
        reason = m2.group(1).strip() if m2 else ""
        return choice, reason

    m = re.search(r"\b([AB])\b", text)
    if m:
        return m.group(1).upper(), ""

    return "A", "parse_fallback"


# =========================================================
# Selection logic
# =========================================================

def select_one(
    dataset: str,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    judge: HFJudge,
    use_heuristic_first: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
    - selected row
    - audit info
    """
    rid = row_a["id"]

    original_ctx = extract_original_context(row_a, dataset)
    cand_a = extract_candidate_fields(row_a, dataset)
    cand_b = extract_candidate_fields(row_b, dataset)

    heuristic_choice = None
    if use_heuristic_first:
        heuristic_choice = cheap_compare(original_ctx, cand_a, cand_b)

    if heuristic_choice in {"A", "B"}:
        selected = row_a if heuristic_choice == "A" else row_b
        audit = {
            "id": rid,
            "dataset": dataset,
            "selection": heuristic_choice,
            "method": "heuristic",
            "reason": "deterministic prefilter",
        }
        return selected, audit

    prompt = build_prompt(dataset, original_ctx, cand_a, cand_b)
    choice, reason, raw = judge.judge(prompt)

    selected = row_a if choice == "A" else row_b

    audit = {
        "id": row_a["id"],
        "option_order": row_a.get("option_order", "original"),
        "dataset": dataset,
        "selection": choice,
        "method": "llm_judge",
        "reason": reason,
        "raw_judge_output": raw,
    }
    return selected, audit


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["GGB", "SUPER_SCOTUS"])
    parser.add_argument("--file_a", type=str, required=True, help="JSONL from model A")
    parser.add_argument("--file_b", type=str, required=True, help="JSONL from model B")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--audit_jsonl", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument("--disable_heuristic_first", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="For debugging only")
    args = parser.parse_args()

    idx_a = load_index(args.file_a)
    idx_b = load_index(args.file_b)

    keys_a = set(idx_a.keys())
    keys_b = set(idx_b.keys())

    common_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print("=" * 60)
    print("Selection Job")
    print("-" * 60)
    print(f"Dataset      : {args.dataset}")
    print(f"File A       : {args.file_a}")
    print(f"File B       : {args.file_b}")
    print(f"Output       : {args.output_jsonl}")
    print(f"Audit        : {args.audit_jsonl}")
    print(f"Judge model  : {args.model_name}")
    print(f"Common IDs   : {len(common_keys)}")
    print(f"Only A IDs   : {len(only_a)}")
    print(f"Only B IDs   : {len(only_b)}")
    print("=" * 60)

    if not common_keys:
        raise ValueError("No overlapping IDs found between the two files.")

    if args.limit is not None:
        common_ids = common_ids[:args.limit]
        print(f"[Debug] limited to first {len(common_ids)} examples")

    judge = HFJudge(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    )

    selected_rows: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []

    #count_a = 0
    #count_b = 0
    #count_heur = 0
    #count_llm = 0

    for i, key in enumerate(common_keys, start=1):
        row_a = idx_a[key]
        row_b = idx_b[key]

        selected, audit = select_one(
            dataset=args.dataset,
            row_a=row_a,
            row_b=row_b,
            judge=judge,
            use_heuristic_first=not args.disable_heuristic_first,
        )

        selected_rows.append(selected)
        audits.append(audit)

    # include leftovers if wanted:
    # by default, we only save common ids because these are comparable pairs
    write_jsonl(args.output_jsonl, selected_rows)

    if args.audit_jsonl:
        write_jsonl(args.audit_jsonl, audits)

    print("\nDone.")
    print(f"Selected rows : {len(selected_rows)}")
    #print(f"Chose A       : {count_a}")
    #print(f"Chose B       : {count_b}")
    #print(f"Heuristic     : {count_heur}")
    #print(f"LLM judge     : {count_llm}")
    print(f"Saved to      : {args.output_jsonl}")
    if args.audit_jsonl:
        print(f"Audit saved   : {args.audit_jsonl}")


if __name__ == "__main__":
    main()
