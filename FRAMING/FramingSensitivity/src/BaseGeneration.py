#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate LLM decisions for multiple skeleton datasets (ROLECONFLICT / TRIAGE / GGB / UNIBENCH).

Input skeleton JSONL/JSON examples should contain:
- id
- dataset
- domain
- bundle.items[*].item_id (+ text/situation fields)
- decision_question
- label_space  (decision options)
- framings (optional)

Output JSONL schema (one per input example):
{
  "id": ...,
  "dataset": ...,
  "domain": ...,
  "pred_decision": "...",          # one of decision_space or "tie"
  "raw_model_output": "...",
  "decision_space": [...],         # from label_space
  "framings": {...}                # copied for later framing eval
}

Example:
  python generate_decisions_hf.py \
    --input_file /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
    --model_id google/gemma-2b-it \
    --out_file /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/triage_preds.jsonl

Batch:
  python generate_decisions_hf.py \
    --input_dir /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data \
    --pattern "*_skeleton.jsonl" \
    --model_id google/gemma-2b-it \
    --out_file /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/all_preds.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import requests
import re
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math
from collections import Counter



# -----------------------------
# IO
# -----------------------------
def iter_examples_from_file(path: str) -> Iterable[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            yield obj
        else:
            raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")
    else:
        raise ValueError(f"Unsupported extension: {path}")


def iter_examples(input_file: Optional[str], input_dir: Optional[str], pattern: str) -> Iterable[Dict[str, Any]]:
    if input_file:
        yield from iter_examples_from_file(input_file)
        return
    if not input_dir:
        raise ValueError("Provide --input_file or --input_dir")
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(input_dir, pattern)}")
    for fp in files:
        yield from iter_examples_from_file(fp)


# -----------------------------
# Prompt builder (dataset-agnostic)
# -----------------------------
def _item_to_line(it: Dict[str, Any]) -> str:
    """
    Robust single-line rendering across datasets.
    Priority: text > situation_text > role fields.
    """
    item_id = it.get("item_id", "unknown")
    if "text" in it and it["text"] is not None:
        desc = it["text"]
    elif "situation_text" in it and it["situation_text"] is not None:
        desc = it["situation_text"]
    else:
        # role conflict specific fields
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
    """
    Forces model to output exactly one option from decision_space or "tie".
    Uses generic schema so it works for all 4 datasets.
    """
    decision_q = ex.get("decision_question", "Which option should I choose?")
    decision_space = ex.get("label_space", ex.get("decision_space", []))

    bundle = ex.get("bundle", {})
    items = bundle.get("items", [])
    resource = bundle.get("resource", {})
    base = ex.get("base", {})

    # Context construction:
    chunks: List[str] = []

    # If role_conflict has shared_story, include it
    shared_story = bundle.get("shared_story")
    if isinstance(shared_story, str) and shared_story.strip():
        chunks.append("Story:\n" + shared_story.strip())
    else:
        if isinstance(base, dict) and base:
            if base.get("vignette"):
                chunks.append("Context:\n" + str(base["vignette"]).strip())
            elif base.get("statement"):
                chunks.append("Statement:\n" + str(base["statement"]).strip())
            elif base.get("scenario"):
                chunks.append("Scenario:\n" + str(base["scenario"]).strip())

    # Items/options
    item_lines = [_item_to_line(it) for it in items]
    if item_lines:
        chunks.append("Options:\n" + "\n".join(item_lines))

    # Resource (optional)
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
        "DECISION: "
    )
    print(prompt)
    return prompt


# -----------------------------
# Parsing
# -----------------------------
import re
from typing import List

_DECISION_PAT = re.compile(r"DECISION\s*:?\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)

def parse_pred_decision(text: str, decision_space: List[str]) -> str:
    if not text:
        return "tie"

    ms = _DECISION_PAT.findall(text)
    if ms:
        cand = re.sub(r"[^\w\-]+", "", ms[-1])
        if cand.lower() == "tie":
            return "tie"
        if cand in decision_space:
            return cand

    candidates = list(decision_space) + ["tie"]
    pattern = r"\b(" + "|".join(re.escape(x) for x in candidates) + r")\b"
    m2 = re.search(pattern, text, flags=re.IGNORECASE)
    if m2:
        tok = m2.group(1)
        if tok.lower() == "tie":
            return "tie"
        for d in decision_space:
            if tok.lower() == str(d).lower():
                return d

    return "tie"


def get_decision_token_id(tokenizer, label: str):
    for s in (f" {label}", label):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) >= 1:
            return ids[0]
    return None


def conditional_normalize(dist: Dict[str, float], keys):
    s = sum(dist.get(k, 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: dist.get(k, 0.0) / s for k in keys}


def entropy_from_probs(p: Dict[str, float], keys, eps=1e-12):
    h = 0.0
    for k in keys:
        pk = float(p.get(k, 0.0))
        if pk > 0:
            h -= pk * math.log(pk + eps)
    return max(h, 0.0)


def entropy_full_from_logits(logits) -> float:
    #x = logits_1d.float()
    #logp = F.log_softmax(x, dim=-1)
    #p = torch.exp(logp)
    #H = -(p * logp).sum()
    #return float(max(H.item(), 0.0))
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())

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
# Openrouter generation
# -----------------------------
def openrouter_generate_one(prompt: str, api_key: str, model: str, max_new_tokens: int, temperature: float,
                            site: str, app: str, retries: int = 5) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional metadata for OpenRouter leaderboards/ranking
        "HTTP-Referer": site,
        "X-Title": app,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    backoff = 1.0
    for attempt in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=300)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        # rate limit / transient
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        # hard error
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

    raise RuntimeError(f"OpenRouter failed after {retries} retries. Last: {r.status_code} {r.text}")


# -----------------------------
# HF generation
# -----------------------------
@torch.inference_mode()
def run_hf_generation(
    model_id: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: str,
    nbest: int,
    decode_mode: str,
    num_beams: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype 결정
    if device.startswith("cuda"):
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    # model load
    if device == "cuda" or device == "cpu":
        # 자동 배치 허용 (단, 특정 GPU 고정은 아님)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to("cpu")
    else:
        # 예: cuda:0, cuda:1 처럼 특정 장치 고정
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device)

    model.eval()

    all_texts: List[str] = []
    all_last_logits: List[torch.Tensor] = []
    group_sizes: List[int] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(model.device) for k, v in toks.items()}

        gen_kwargs = dict(
            **toks,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if nbest <= 1:
            gen_kwargs.update(do_sample=False)
            cur_nbest = 1
        else:
            cur_nbest = nbest
            if decode_mode == "beam":
                gen_kwargs.update(
                    do_sample=False,
                    num_beams=max(num_beams, nbest),
                    num_return_sequences=nbest,
                )
            else:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=nbest,
                )

        gen = model.generate(**gen_kwargs)

        # (B*nbest, V)
        #last_logits = gen.scores[0].detach().cpu()
        #all_last_logits.append(last_logits)

        # decode (B*nbest)
        seqs = gen.sequences
        bsz = len(batch)
        group_sizes.extend([cur_nbest] * bsz)
        
        # --- inside run_hf_generation loop ---
        prompt_len = toks["input_ids"].shape[1]

        for j in range(bsz * cur_nbest):
            new_tokens = seqs[j][prompt_len:]
        #for j in range(bsz * cur_nbest):
            #base_j = j // cur_nbest
            #prompt_len = toks["input_ids"][base_j].shape[0]
            #prompt_len = int(toks["attention_mask"][base_j].sum().item())
            #new_tokens = seqs[j][prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_texts.append(text)

        step0_scores = gen.scores[0].detach().cpu()
        all_last_logits.append(step0_scores)

        #for j in range(bsz * cur_nbest):
            # attention_mask로 실제 prompt 길이 계산 (padding 안전)
        #    base_j = j // cur_nbest
            #prompt_len = int(toks["attention_mask"][base_j].sum().item())
        #    prompt_len = toks["input_ids"].shape[1]  # padding 포함 고정 길이 (left padding 정답)
        #    new_tokens = seqs[j][prompt_len:]
        #    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        #    all_texts.append(text)

    last_logits_cat = torch.cat(all_last_logits, dim=0) if len(all_last_logits) else None
    return all_texts, last_logits_cat, group_sizes, tokenizer

# -----------------------------
# Final generation
# -----------------------------
def generate_outputs(prompts: list[str], args):
    if args.backend == "hf":
        return run_hf_generation(
            model_id=args.model_id,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            device=args.device,
            nbest=args.nbest,
            decode_mode=args.decode_mode,
            num_beams=args.num_beams,
        )

    # openrouter는 confidence/logits 없음
    outs = []
    for p in prompts:
        txt = openrouter_generate_one(
            prompt=p,
            api_key=args.or_api_key,
            model=args.or_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            site=args.or_site,
            app=args.or_app,
        )
        outs.append(txt)

    # openrouter는 nbest를 여기서 구현하려면 반복 호출 필요(일단 제외)
    group_sizes = [1] * len(outs)
    return outs, None, group_sizes, None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, default=None)
    ap.add_argument("--input_dir", type=str, default=None)
    ap.add_argument("--pattern", type=str, default="*.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    
    ap.add_argument("--backend", type=str, default="hf", choices=["hf", "openrouter"])
    ap.add_argument("--or_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--or_model", type=str, default=None, help="e.g., meta-llama/llama-3.1-70b-instruct")
    ap.add_argument("--or_site", type=str, default="http://localhost", help="Optional OpenRouter ranking metadata")
    ap.add_argument("--or_app", type=str, default="FramingSensitivity", help="Optional OpenRouter ranking metadata")

    ap.add_argument("--model_id", type=str, required=False)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_file", type=str, required=True)
    ap.add_argument("--nbest", type=int, default=1)
    ap.add_argument("--decode_mode", choices=["beam", "sample"], default="beam")
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()
    
    # backend-specific argument validation
    if args.backend == "hf":
        if not args.model_id:
            ap.error("--model_id is required when --backend=hf")

    if args.backend == "openrouter":
        if not args.or_model:
            ap.error("--or_model is required when --backend=openrouter")

    examples = list(iter_examples(args.input_file, args.input_dir, args.pattern))
    
    if not examples:
        raise ValueError("No examples loaded.")
        
    if args.limit is not None:
        examples = examples[:args.limit]

    prompts = [build_decision_prompt(ex) for ex in examples]
    
    raw_outs, last_logits, group_sizes, tokenizer = generate_outputs(prompts, args)
    
    #tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True) if args.backend == "hf" else None

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(args.out_file, "w", encoding="utf-8") as f:

        offset = 0
        for ex, n in zip(examples, group_sizes):
            decision_space = ex.get("label_space", ex.get("decision_space", []))
            keys = tuple(decision_space + ["tie"])

            outs_n = raw_outs[offset : offset + n]
            logits_n = last_logits[offset : offset + n] if last_logits is not None else None
            offset += n

            preds_n = [parse_pred_decision(o, decision_space) for o in outs_n]
            pred = majority_vote(preds_n)

            # ---------- confidence per sample (A/B/tie) ----------
            conf_list = []
            if logits_n is not None and tokenizer is not None:
                probs = torch.softmax(logits_n.float(), dim=-1)  # (n, V)

                for i in range(probs.shape[0]):
                    dist = {}
                    for d in keys:
                        tok_id = get_decision_token_id(tokenizer, d)
                        dist[d] = float(probs[i, tok_id].item()) if tok_id is not None else 0.0
                    conf_list.append(dist)
            else:
                conf_list = [None] * n

            # ---------- avg / cond ----------
            avg_dist = {k: 0.0 for k in keys}
            if conf_list and isinstance(conf_list[0], dict):
                for k in keys:
                    avg_dist[k] = sum(c.get(k, 0.0) for c in conf_list) / max(len(conf_list), 1)

            cond = conditional_normalize(avg_dist, keys=keys)

            # ---------- entropy (cond) ----------
            H = entropy_from_probs(cond, keys=keys)

            H_list = []
            if conf_list and isinstance(conf_list[0], dict):
                for c in conf_list:
                    ccond = conditional_normalize(c, keys=keys)
                    H_list.append(entropy_from_probs(ccond, keys=keys))
            H_mean = (sum(H_list) / len(H_list)) if H_list else None

            # ---------- entropy (full vocab) ----------
            H_full_mean = None
            if logits_n is not None:
                H_full_list = [entropy_full_from_logits(logits_n[i]) for i in range(logits_n.shape[0])]
                H_full_mean = sum(H_full_list) / len(H_full_list)

            rec = {
                "id": ex.get("id"),
                "dataset": ex.get("dataset"),
                "domain": ex.get("domain"),
                "option_order": ex.get("option_order", "original"),

                # 기존 호환
                "pred_decision": majority_vote(preds_n),
                "raw_model_output": outs_n[0].strip() if outs_n else "",
                "decision_space": decision_space,
                "framings": ex.get("framings", {}),

                # n-best 저장
                "raw_outputs": [o.strip() for o in outs_n],
                "pred_decisions": preds_n,

                # 네가 원하는 동일 필드명
                "confidences": conf_list,
                "confidence_avg": avg_dist,
                "confidence_cond": cond,
                "entropy_cond": H,
                "entropy_cond_mean": H_mean,
                "entropy_full_mean": H_full_mean,
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote predictions: {args.out_file}")


if __name__ == "__main__":
    main()
