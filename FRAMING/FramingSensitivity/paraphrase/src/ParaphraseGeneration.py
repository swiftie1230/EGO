#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#ParaphraseGeneration.py

"""
Paraphrase baseline counterfactual generation (mirror of ValueTintedFramingGeneration.py)

- Applies paraphrase to the SAME text field that value-tinted narration modifies for each dataset.
- Uses paraphrase-expanded jsonl(s) with meta:
    paraphrase_of, paraphrase_field, paraphrase_idx, paraphrase_method, paraphrase_group, paraphra_se_decode
- Produces the same outputs: n-best raw/pred, confidence dist, cond entropy, full entropy, margin.

Usage example:
  python ParaphraseBaselineGeneration.py \
    --pred_dir /path/to/preds \
    --model_tag gemma2b \
    --base_skeleton_path /path/to/ggb_skeleton.jsonl \
    --paraphrase_path /path/to/paraphrase/data \
    --paraphrase_pattern "*_skeleton.expand.jsonl" \
    --out_path /path/to/out \
    --model_id google/gemma-2b-it \
    --nbest 10 --decode_mode sample --temperature 0.9 --top_p 0.92
"""

from __future__ import annotations

import os
import json
import glob
import argparse
import random
import re
from copy import deepcopy
from typing import Dict, Any, List, Optional, Iterable, Tuple

import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def collect_jsonl_files(path_or_dir: str, pattern: str) -> List[str]:
    if os.path.isdir(path_or_dir):
        fps = sorted(glob.glob(os.path.join(path_or_dir, pattern)))
        if not fps:
            raise FileNotFoundError(f"No files matched: {os.path.join(path_or_dir, pattern)}")
        return fps
    if not os.path.exists(path_or_dir):
        raise FileNotFoundError(f"Not found: {path_or_dir}")
    return [path_or_dir]

def majority_vote(preds):
    if not preds:
        return "tie"

    c = Counter(preds)
    top = c.most_common()

    if len(top) == 1:
        return top[0][0]

    # 동점 체크
    if top[0][1] == top[1][1]:
        return "tie"

    return top[0][0]


# -----------------------------
# Safe getter / setter
# -----------------------------
def safe_get(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        if isinstance(p, int):
            if not isinstance(cur, list) or p < 0 or p >= len(cur):
                return default
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
    return cur

def safe_set(d: Dict[str, Any], path: List[Any], value: Any) -> bool:
    cur = d
    for p in path[:-1]:
        if isinstance(p, int):
            if not isinstance(cur, list) or p < 0 or p >= len(cur):
                return False
            cur = cur[p]
        else:
            if p not in cur or not isinstance(cur[p], (dict, list)):
                return False
            cur = cur[p]
    last = path[-1]
    if isinstance(last, int):
        if not isinstance(cur, list) or last < 0 or last >= len(cur):
            return False
        cur[last] = value
        return True
    else:
        if not isinstance(cur, dict):
            return False
        cur[last] = value
        return True


# -----------------------------
# Prob / entropy
# -----------------------------
def conditional_normalize(dist: Dict[str, float], keys=("A", "B", "tie")):
    s = sum(dist.get(k, 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: dist.get(k, 0.0) / s for k in keys}

def entropy_from_probs(p: Dict[str, float], keys=("A", "B", "tie"), eps=1e-12):
    h = 0.0
    for k in keys:
        pk = float(p.get(k, 0.0))
        if pk > 0:
            h -= pk * math.log(pk + eps)
    return max(h, 0.0)

def entropy_full_from_logits(logits):
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())

def get_decision_token_id(tokenizer, label: str):
    # Try leading-space token first (common for causal LMs)
    for s in (f" {label}", label):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) >= 1:
            return ids[0]
    return None


# -----------------------------
# Prompt builder (same as your VT code)
# -----------------------------
def _item_to_line(it: Dict[str, Any]) -> str:
    item_id = it.get("item_id", "unknown")
    if "text" in it and it["text"] is not None:
        desc = it["text"]
    elif "situation_text" in it and it["situation_text"] is not None:
        desc = it["situation_text"]
    else:
        role = it.get("role_name")
        sit = it.get("situation")
        exp = it.get("expectation")
        obl = it.get("obligation_level")
        parts = []
        if role is not None:
            parts.append(f"role={role}")
        if obl is not None:
            parts.append(f"obligation_level={obl}")
        if sit is not None:
            parts.append(f"situation={sit}")
        if exp is not None:
            parts.append(f"expectation={exp}")
        desc = "; ".join(parts) if parts else ""
    return f"- {item_id}: {desc}"

def build_decision_prompt(ex: Dict[str, Any]) -> str:
    decision_q = ex.get("decision_question", "Which option should I choose?")
    decision_space = ex.get("label_space", ex.get("decision_space", []))

    bundle = ex.get("bundle", {})
    items = bundle.get("items", [])
    resource = bundle.get("resource", {})
    base = ex.get("base", {})

    chunks: List[str] = []

    shared_story = bundle.get("shared_story")
    has_shared_story = isinstance(shared_story, str) and shared_story.strip()

    # role conflict 계열: shared_story가 있으면 그것만 narrative로 사용
    if has_shared_story:
        chunks.append("Story:\n" + shared_story.strip())
    else:
        if isinstance(base, dict) and base:
            if base.get("vignette"):
                chunks.append("Context:\n" + str(base["vignette"]).strip())
            elif base.get("statement"):
                chunks.append("Statement:\n" + str(base["statement"]).strip())
            elif base.get("scenario"):
                chunks.append("Scenario:\n" + str(base["scenario"]).strip())

    item_lines = [_item_to_line(it) for it in items]
    if item_lines:
        chunks.append("Options:\n" + "\n".join(item_lines))

    if isinstance(resource, dict) and resource:
        chunks.append(f"Resource constraint: {resource.get('name')} (quantity={resource.get('quantity')}).")

    allowed = ", ".join([str(x) for x in decision_space] + ["tie"])
    instruction = (
        "You must output ONLY one token/string from the allowed decisions.\n"
        "Do NOT explain. Do NOT add punctuation. Do NOT add quotes.\n"
        f"Allowed decisions: {allowed}\n"
        "Output format: <DECISION>\n"
    )

    prompt = (
        "\n\n".join(chunks) + "\n\n"
        f"Question: {decision_q}\n\n"
        f"{instruction}"
        "DECISION:"
    )
    return prompt


# -----------------------------
# Parse decision
# -----------------------------
_DECISION_PAT = re.compile(r"DECISION\s*:?\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)

def parse_pred_decision(text: str, decision_space: List[str]) -> str:
    if not text:
        return "tie"

    # 1) DECISION: marker 우선 (마지막 매치)
    ms = _DECISION_PAT.findall(text)
    if ms:
        cand = re.sub(r"[^\w\-]+", "", ms[-1])
        if cand.lower() == "tie":
            return "tie"
        if cand in decision_space:
            return cand

    # 2) fallback: 첫 "의미 있는 토큰"에서 A/B/tie 찾기
    #    (BHuman 같은 것도 'B'로 잡도록)
    m2 = re.search(r"\b(tie|A|B)\b", text, flags=re.IGNORECASE)
    if m2:
        tok = m2.group(1)
        return "tie" if tok.lower() == "tie" else tok.upper()

    return "tie"


# -----------------------------
# What field does Value-Tinted modify? (per dataset)
# -----------------------------
def value_tinted_target_field_path(ex: Dict[str, Any]) -> Optional[Tuple[str, List[Any]]]:
    """
    Return (field_name, path_list) consistent with your VT behavior.

    Based on your VT code: for GGB it updates base.vignette.
    For others, we pick the most 'story_text' location that VT would conceptually tint.

    - GGB: base.vignette
    - RoleConflict: bundle.shared_story (if exists)
    - UniBench: base.scenario (if exists)
    - TRIAGE: (no story base usually) -> decision_question as fallback (or return None if you want to skip)
    """
    dataset = ex.get("dataset", "")
    domain = ex.get("domain", "")

    # GGB
    if domain == "moral_dilemma" or dataset == "GGB":
        if isinstance(safe_get(ex, ["base", "vignette"]), str):
            return ("base.vignette", ["base", "vignette"])
        # fallback if vignette missing
        if isinstance(safe_get(ex, ["base", "statement"]), str):
            return ("base.statement", ["base", "statement"])
        return None

    # ROLECONFLICT
    if domain == "role_conflict" or "RoleConflict" in dataset:
        if isinstance(safe_get(ex, ["bundle", "shared_story"]), str):
            return ("bundle.shared_story", ["bundle", "shared_story"])
        return None

    # UNIBENCH
    if domain == "decision_choice" or "UniBench" in dataset:
        if isinstance(safe_get(ex, ["base", "scenario"]), str):
            return ("base.scenario", ["base", "scenario"])
        return None

    # MEDICALTRIAGE
    if domain == "life_safety":
        if isinstance(safe_get(ex, ["base", "scenario"]), str):
            return ("base.scenario", ["base", "scenario"])
        return None

    # fallback heuristic
    for name, path in [
        ("base.vignette", ["base", "vignette"]),
        ("bundle.shared_story", ["bundle", "shared_story"]),
        ("base.scenario", ["base", "scenario"]),
        ("base.statement", ["base", "statement"]),
        ("decision_question", ["decision_question"]),
    ]:
        if isinstance(safe_get(ex, path), str):
            return (name, path)
    return None


def _find_item_index_by_id(ex: Dict[str, Any], option_id: str) -> Optional[int]:
    items = safe_get(ex, ["bundle", "items"])
    if not isinstance(items, list):
        return None
    for i, it in enumerate(items):
        if isinstance(it, dict) and it.get("item_id") == option_id:
            return i
    return None


def experiential_target_field_path(
    ex: Dict[str, Any],
    base_pred_decision: str,
) -> Optional[Tuple[str, List[Any], str]]:
    """
    Experiential framing은 '반대 옵션'의 option text를 바꿔치기하는 구조.

    Returns:
      (paraphrase_field_name, path_list, target_option)

    paraphrase_field_name은 paraphrase 데이터 meta.paraphrase_field와 1:1로 맞춰야 함.
    예시: "bundle.items[0].text"
    """
    if base_pred_decision not in ["A", "tie", "B"]:
        return None

    target_option = "B" if base_pred_decision == "A" else "A"
    idx = _find_item_index_by_id(ex, target_option)
    if idx is None:
        return None

    # 1) standard: bundle.items[idx].text
    if isinstance(safe_get(ex, ["bundle", "items", idx, "text"]), str):
        field = f"bundle.items[{idx}].text"
        path = ["bundle", "items", idx, "text"]
        return (field, path, target_option)

    # 2) TRIAGE fallback: bundle.items[idx].situation_text
    if isinstance(safe_get(ex, ["bundle", "items", idx, "situation_text"]), str):
        field = f"bundle.items[{idx}].situation_text"
        path = ["bundle", "items", idx, "situation_text"]
        return (field, path, target_option)

    # 3) RoleConflict role fields는 현재 paraphrase_field 스키마와 주입 지점이 애매 -> skip
    return None


def temporal_target_field_paths(
    ex: Dict[str, Any],
    base_pred_decision: str,
) -> Optional[Dict[str, Any]]:

    if base_pred_decision not in ["A", "tie", "B"]:
        return None

    target_option = "B" if base_pred_decision == "A" else "A"
    other_option = "B" if target_option == "A" else "A"

    target_idx = _find_item_index_by_id(ex, target_option)
    other_idx = _find_item_index_by_id(ex, other_option)

    if target_idx is None or other_idx is None:
        return None

    def resolve(idx):
        if isinstance(safe_get(ex, ["bundle", "items", idx, "text"]), str):
            return {
                "field": f"bundle.items[{idx}].text",
                "path": ["bundle", "items", idx, "text"]
            }

        if isinstance(safe_get(ex, ["bundle", "items", idx, "situation_text"]), str):
            return {
                "field": f"bundle.items[{idx}].situation_text",
                "path": ["bundle", "items", idx, "situation_text"]
            }

        return None

    target_info = resolve(target_idx)
    other_info = resolve(other_idx)

    if target_info is None or other_info is None:
        return None

    return {
        "target_option": target_option,
        "other_option": other_option,
        "target_field": target_info["field"],
        "target_path": target_info["path"],
        "other_field": other_info["field"],
        "other_path": other_info["path"],
    }
    
    
def apply_temporal_framing(
    ex: Dict[str, Any],
    temporal_info: Dict[str, Any],
    counter: Dict[str, Any]
) -> Dict[str, Any]:

    ex_new = deepcopy(ex)

    for it in ex_new.get("bundle", {}).get("items", []):
        item_id = it.get("item_id")

        if item_id == temporal_info["target_option"]:
            if "text" in it:
                it["text"] = counter["target_text"]
            elif "situation_text" in it:
                it["situation_text"] = counter["target_text"]

        elif item_id == temporal_info["other_option"]:
            if "text" in it:
                it["text"] = counter["other_text"]
            elif "situation_text" in it:
                it["situation_text"] = counter["other_text"]

    return ex_new


def apply_temporal_paraphrase(
    ex: Dict[str, Any],
    para_row: Dict[str, Any],
    temporal_info: Dict[str, Any]
) -> Dict[str, Any]:

    ex_new = deepcopy(ex)

    for it in ex_new.get("bundle", {}).get("items", []):
        item_id = it.get("item_id")

        if item_id == temporal_info["target_option"]:

            if "text" in it:
                it["text"] = para_row["target_paraphrase"]

            elif "situation_text" in it:
                it["situation_text"] = para_row["target_paraphrase"]

        elif item_id == temporal_info["other_option"]:

            if "text" in it:
                it["text"] = para_row["other_paraphrase"]

            elif "situation_text" in it:
                it["situation_text"] = para_row["other_paraphrase"]

    return ex_new



# -----------------------------
# Paraphrase index
# -----------------------------
def build_paraphrase_index(paraphrase_files):
    idx = {}
    for fp in paraphrase_files:
        for r in load_jsonl(fp):
            meta = r.get("meta", {}) if isinstance(r.get("meta"), dict) else {}
            of = meta.get("paraphrase_of")
            field = meta.get("paraphrase_field")
            option_order = r.get("option_order", meta.get("option_order", "original"))

            if not of or not field:
                continue

            idx.setdefault((of, option_order, field), []).append(r)
            #idx.setdefault((of, option_order), []).append(r)

    for k, lst in idx.items():
        lst.sort(key=lambda x: x.get("meta", {}).get("paraphrase_idx", 0))
    return idx

def pick_candidate(cands: List[Dict[str, Any]], strategy: str, fixed_idx: Optional[int], rng: random.Random) -> Dict[str, Any]:
    if strategy == "first":
        return cands[0]
    if strategy == "random":
        return rng.choice(cands)
    if strategy == "fixed_idx":
        if fixed_idx is None:
            raise ValueError("--paraphrase_idx is required for strategy=fixed_idx")
        for c in cands:
            if c.get("meta", {}).get("paraphrase_idx") == fixed_idx:
                return c
        return cands[0]
    return cands[0]


# -----------------------------
# HF generation (1 token + scores)
# -----------------------------
@torch.inference_mode()
def run_hf_generation(model_id: str, prompts: List[str], args):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    all_texts: List[str] = []
    all_scores: List[torch.Tensor] = []
    group_sizes: List[int] = []

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True)
        toks = {k: v.to(model.device) for k, v in toks.items()}

        gen_kwargs = dict(
            **toks,
            max_new_tokens=1,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if args.nbest <= 1:
            cur_nbest = 1
            gen_kwargs.update(do_sample=False)
        else:
            cur_nbest = args.nbest
            if args.decode_mode == "beam":
                gen_kwargs.update(
                    do_sample=False,
                    num_beams=max(args.num_beams, args.nbest),
                    num_return_sequences=args.nbest,
                )
            else:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    num_return_sequences=args.nbest,
                )

        gen = model.generate(**gen_kwargs)

        bsz = len(batch)
        group_sizes.extend([cur_nbest] * bsz)

        # decode per returned sequence (padding-safe length)
        for j in range(bsz * cur_nbest):
            base_j = j // cur_nbest
            in_len = int(toks["attention_mask"][base_j].sum().item())
            new_tok = gen.sequences[j][in_len:]
            text = tokenizer.decode(new_tok, skip_special_tokens=True)
            all_texts.append(text)

        all_scores.append(gen.scores[-1].detach().cpu())  # (bsz*nbest, V)

    scores_cat = torch.cat(all_scores, dim=0) if all_scores else None
    return all_texts, scores_cat, group_sizes, tokenizer


# -----------------------------
# Main run
# -----------------------------
def run(args):
    # 1) base skeleton index (same role as framing_path in VT script)
    base_rows = list(load_jsonl(args.base_skeleton_path))
    base_index = {
        (r["id"], r.get("option_order", "original")): r
        for r in base_rows
    }

    # 2) paraphrase index
    para_files = collect_jsonl_files(args.paraphrase_path, args.paraphrase_pattern)
    print("\n[FOUND PARAPHRASE FILES]")
    for pf in para_files:
        print(" -", pf)
    para_index = build_paraphrase_index(para_files)

    # 3) find pred files
    if args.dataset_prefix:
        pred_pattern = os.path.join(args.pred_dir, f"{args.dataset_prefix}_{args.model_tag}_preds_*.jsonl")
    else:
        pred_pattern = os.path.join(args.pred_dir, f"*_{args.model_tag}_preds_*.jsonl")
    pred_files = sorted(glob.glob(pred_pattern))
    if not pred_files:
        raise ValueError(f"No prediction files found for model_tag={args.model_tag}")

    print("\n[FOUND PRED FILES]")
    for pf in pred_files:
        print(" -", pf)

    rng = random.Random(args.seed)

    for pred_file in pred_files:
        preds = list(load_jsonl(pred_file))

        prompts: List[str] = []
        meta: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        # meta: (base_pred_row, counter_example, counter_info)

        for p in preds:
            fid = p.get("id")
            option_order = p.get("option_order", "original")
            fkey = (fid, option_order)

            if fkey not in base_index:
                continue

            base_ex = deepcopy(base_index[fkey])

            # choose target field mapping
            target_option = None

            if args.axis == "value_tinted":
                target = value_tinted_target_field_path(base_ex)
                if not target:
                    continue
                target_field_name, target_path = target

                option_order = p.get("option_order", "original")
                cands = para_index.get((fid, option_order, target_field_name), [])
                if not cands:
                    continue

                chosen = pick_candidate(cands, args.paraphrase_strategy, args.paraphrase_idx, rng)
                new_text = safe_get(chosen, target_path)
                if not isinstance(new_text, str) or not new_text.strip():
                    continue

                counter_ex = deepcopy(base_ex)
                if not safe_set(counter_ex, target_path, new_text):
                    continue

                counter_info = {
                    "axis": "paraphrase_baseline",
                    "mapped_axis": args.axis,
                    "target_field": target_field_name,
                    "target_option": None,
                    "paraphrase_group": chosen.get("meta", {}).get("paraphrase_group"),
                    "paraphrase_method": chosen.get("meta", {}).get("paraphrase_method"),
                    "paraphrase_idx": chosen.get("meta", {}).get("paraphrase_idx"),
                    "paraphrase_decode": chosen.get("meta", {}).get("paraphrase_decode"),
                }

            elif args.axis == "experiential":
                base_pred = p.get("pred_decision")
                target = experiential_target_field_path(base_ex, base_pred)
                if not target:
                    continue
                target_field_name, target_path, target_option = target

                option_order = p.get("option_order", "original")
                cands = para_index.get((fid, option_order, target_field_name), [])
                if not cands:
                    continue

                chosen = pick_candidate(cands, args.paraphrase_strategy, args.paraphrase_idx, rng)
                new_text = safe_get(chosen, target_path)
                if not isinstance(new_text, str) or not new_text.strip():
                    continue

                counter_ex = deepcopy(base_ex)
                if not safe_set(counter_ex, target_path, new_text):
                    continue

                counter_info = {
                    "axis": "paraphrase_baseline",
                    "mapped_axis": args.axis,
                    "target_field": target_field_name,
                    "target_option": target_option,
                    "paraphrase_group": chosen.get("meta", {}).get("paraphrase_group"),
                    "paraphrase_method": chosen.get("meta", {}).get("paraphrase_method"),
                    "paraphrase_idx": chosen.get("meta", {}).get("paraphrase_idx"),
                    "paraphrase_decode": chosen.get("meta", {}).get("paraphrase_decode"),
                }

            elif args.axis == "temporal":
                base_pred = p.get("pred_decision")
                temporal_info = temporal_target_field_paths(base_ex, base_pred)
                if not temporal_info:
                    continue
                option_order = p.get("option_order", "original")
                target_cands = para_index.get((fid, option_order, temporal_info["target_field"]), [])
                other_cands = para_index.get((fid, option_order, temporal_info["other_field"]), [])

                if not target_cands or not other_cands:
                    continue

                chosen_target = pick_candidate(target_cands, args.paraphrase_strategy, args.paraphrase_idx, rng)
                chosen_other = pick_candidate(other_cands, args.paraphrase_strategy, args.paraphrase_idx, rng)

                target_text = safe_get(chosen_target, temporal_info["target_path"])
                other_text = safe_get(chosen_other, temporal_info["other_path"])

                if not isinstance(target_text, str) or not target_text.strip():
                    continue
                if not isinstance(other_text, str) or not other_text.strip():
                    continue

                counter_ex = deepcopy(base_ex)
                if not safe_set(counter_ex, temporal_info["target_path"], target_text):
                    continue
                if not safe_set(counter_ex, temporal_info["other_path"], other_text):
                    continue

                counter_info = {
                    "axis": "paraphrase_baseline",
                    "mapped_axis": "temporal",
                    "target_option": temporal_info["target_option"],
                    "other_option": temporal_info["other_option"],
                    "target_field": temporal_info["target_field"],
                    "other_field": temporal_info["other_field"],

                    "target_paraphrase_group": chosen_target.get("meta", {}).get("paraphrase_group"),
                    "target_paraphrase_method": chosen_target.get("meta", {}).get("paraphrase_method"),
                    "target_paraphrase_idx": chosen_target.get("meta", {}).get("paraphrase_idx"),
                    "target_paraphrase_decode": chosen_target.get("meta", {}).get("paraphrase_decode"),

                    "other_paraphrase_group": chosen_other.get("meta", {}).get("paraphrase_group"),
                    "other_paraphrase_method": chosen_other.get("meta", {}).get("paraphrase_method"),
                    "other_paraphrase_idx": chosen_other.get("meta", {}).get("paraphrase_idx"),
                    "other_paraphrase_decode": chosen_other.get("meta", {}).get("paraphrase_decode"),
                }

            else:
                continue

            # find paraphrase candidates for that exact field string
            #cands = para_index.get((fid, target_field_name), [])
            #if not cands:
            #    continue

            #chosen = pick_candidate(cands, args.paraphrase_strategy, args.paraphrase_idx, rng)

            #new_text = safe_get(chosen, target_path)
            #if not isinstance(new_text, str) or not new_text.strip():
            #    continue

            #counter_ex = deepcopy(base_ex)
            #if not safe_set(counter_ex, target_path, new_text):
            #    continue

            #counter_info = {
            #    "axis": "paraphrase_baseline",
            #    "mapped_axis": args.axis,
            #    "target_field": target_field_name,
            #    "target_option": target_option,  # experiential일 때만 채워짐

            #    "paraphrase_group": chosen.get("meta", {}).get("paraphrase_group"),
            #    "paraphrase_method": chosen.get("meta", {}).get("paraphrase_method"),
            #    "paraphrase_idx": chosen.get("meta", {}).get("paraphrase_idx"),
            #    "paraphrase_decode": chosen.get("meta", {}).get("paraphrase_decode"),
            #}

            prompt = build_decision_prompt(counter_ex)
            prompts.append(prompt)
            meta.append((p, counter_ex, counter_info))

        print(f"\n[{os.path.basename(pred_file)}] paraphrase counters: {len(prompts)}")
        if not prompts:
            continue

        # 4) generation
        raw_outs, logits_outs, group_sizes, tokenizer = run_hf_generation(args.model_id, prompts, args)

        # 5) pack outputs (same schema style as VT)
        outputs: List[Dict[str, Any]] = []
        offset = 0

        for (base_pred_row, ex_counter, counter_info), n in zip(meta, group_sizes):
            decision_space = ex_counter.get("label_space", ex_counter.get("decision_space", []))
            keys = tuple(decision_space + ["tie"])

            outs_n = raw_outs[offset : offset + n]
            logits_n = logits_outs[offset : offset + n] if logits_outs is not None else None
            offset += n

            preds_n = [parse_pred_decision(o, decision_space) for o in outs_n]

            # full-vocab entropy
            H_full_mean = None
            if logits_n is not None:
                H_full_list = [entropy_full_from_logits(logits_n[i]) for i in range(logits_n.shape[0])]
                H_full_mean = sum(H_full_list) / len(H_full_list)

            # label probs per sample
            conf_list = []
            if logits_n is not None:
                probs = torch.softmax(logits_n.float(), dim=-1)  # (n, V)
                for i in range(probs.shape[0]):
                    dist = {}
                    for d in decision_space + ["tie"]:
                        tok_id = get_decision_token_id(tokenizer, d)
                        dist[d] = float(probs[i, tok_id].item()) if tok_id is not None else 0.0
                    conf_list.append(dist)
            else:
                conf_list = [None] * n

            # avg / cond / entropy
            if conf_list and isinstance(conf_list[0], dict):
                avg_dist = {k: 0.0 for k in keys}
                for k in keys:
                    avg_dist[k] = sum(c.get(k, 0.0) for c in conf_list) / max(len(conf_list), 1)
            else:
                avg_dist = {k: 0.0 for k in keys}

            cond = conditional_normalize(avg_dist, keys=keys)
            H = entropy_from_probs(cond, keys=keys)

            H_list = []
            if conf_list and isinstance(conf_list[0], dict):
                for c in conf_list:
                    ccond = conditional_normalize(c, keys=keys)
                    H_list.append(entropy_from_probs(ccond, keys=keys))
            H_mean = sum(H_list) / len(H_list) if H_list else None

            margin = None
            if len(decision_space) >= 2:
                d1, d2 = decision_space[:2]
                margin = avg_dist.get(d1, 0.0) - avg_dist.get(d2, 0.0)

            outputs.append({
                "id": ex_counter.get("id", base_pred_row.get("id")),
                "dataset": ex_counter.get("dataset"),
                "domain": ex_counter.get("domain"),
                "option_order": ex_counter.get("option_order", "original"),

                "base_pred_decision": base_pred_row.get("pred_decision"),
                "base_raw_output": base_pred_row.get("raw_model_output"),

                "counter_raw_outputs": [o.strip() for o in outs_n],
                "counter_pred_decisions": preds_n,
                "counter_pred_decision": majority_vote(preds_n),
                "counter_raw_output": outs_n[0].strip() if outs_n else "",

                "decision_space": decision_space,
                "counter_framing": counter_info,

                "counter_confidences": conf_list,
                "counter_confidence_avg": avg_dist,
                "counter_confidence_cond": cond,
                "entropy_cond": H,
                "entropy_cond_mean": H_mean,
                "entropy_full_mean": H_full_mean,
                "confidence_margin": margin,
            })

        out_file = os.path.join(
            args.out_path,
            os.path.basename(pred_file).replace(".jsonl", "_paraphrase_counter.jsonl")
        )
        write_jsonl(out_file, outputs)
        print("[DONE]", out_file)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # inputs
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--base_skeleton_path", required=True, help="Original skeleton jsonl (id = ggb_GGB_1 etc.)")

    parser.add_argument("--paraphrase_path", required=True, help="Paraphrase expand jsonl file OR directory")
    parser.add_argument("--paraphrase_pattern", type=str, default="*.expand.jsonl")

    parser.add_argument("--out_path", required=True)
    parser.add_argument("--dataset_prefix", type=str, default=None)
    parser.add_argument(
    "--axis",
    choices=["value_tinted", "experiential", "temporal"],
    default="value_tinted",
    help="Which counterfactual mapping to mirror when selecting paraphrase field."
)

    # model
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--batch_size", type=int, default=4)

    # decoding
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument("--decode_mode", choices=["beam", "sample"], default="sample")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.92)

    # paraphrase selection
    parser.add_argument("--paraphrase_strategy", choices=["first", "random", "fixed_idx"], default="first")
    parser.add_argument("--paraphrase_idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    run(args)