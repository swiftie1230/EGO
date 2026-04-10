# ValueTintedFramingGeneration.py

from __future__ import annotations
import os
import json
import argparse
import glob
import re
from copy import deepcopy
from typing import Dict, Any, List, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F

import math
from collections import Counter

def conditional_normalize(dist: Dict[str, float], keys=("A","B","tie")):
    s = sum(dist.get(k, 0.0) for k in keys)
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: dist.get(k, 0.0) / s for k in keys}

def entropy_from_probs(p: Dict[str, float], keys=("A","B","tie"), eps=1e-12):
    h = 0.0
    for k in keys:
        pk = float(p.get(k, 0.0))
        if pk > 0:
            h -= pk * math.log(pk + eps)
    return h

def get_decision_token_id(tokenizer, label: str):
    """
    DECISION: 뒤에 바로 생성되는 1st token id를 반환.
    대부분의 CausalLM은 leading space 포함 토큰을 생성하므로
    ' A' → 'A' 순서로 시도.
    """
    candidates = [f" {label}", label]

    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) >= 1:
            return ids[0]

    return None

def entropy_full_from_logits(logits):
    # logits: (V,) tensor
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


# --------------------------------
# IO
# --------------------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_index(rows):
    return {
        (r["id"], r.get("option_order", "original")): r
        for r in rows
    }


# --------------------------------
# VALUE TINTED NARRATION
# --------------------------------
def apply_value_tinted_narration(pred_row, frame_row):
    pred = pred_row.get("pred_decision")
    if pred not in ["A", "tie", "B"]:
        return None

    target_option = "B" if pred == "A" else "A"

    try:
        option_block = frame_row["framings"]["value_tinted_narration"][target_option]
        value_key = list(option_block.keys())[0]
        new_text = option_block[value_key]["text"]
    except Exception:
        return None

    ex_counter = deepcopy(frame_row)

    if "bundle" not in ex_counter or not isinstance(ex_counter["bundle"], dict):
        ex_counter["bundle"] = {}
    if "base" not in ex_counter or not isinstance(ex_counter["base"], dict):
        ex_counter["base"] = {}

    # shared_story가 있으면 거기를 우선 교체
    if ex_counter["bundle"].get("shared_story") is not None:
        ex_counter["bundle"]["shared_story"] = new_text
    elif ex_counter["base"].get("vignette") is not None:
        ex_counter["base"]["vignette"] = new_text
    elif ex_counter["base"].get("statement") is not None:
        ex_counter["base"]["statement"] = new_text
    else:
        ex_counter["base"]["scenario"] = new_text

    return ex_counter, {
        "axis": "value_tinted_narration",
        "target_option": target_option,
        "value_type": value_key
    }


# --------------------------------
# Prompt builder
# --------------------------------
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
    decision_q = ex.get("decision_question", "Which option should I choose?")
    decision_space = ex.get("label_space", ex.get("decision_space", []))

    bundle = ex.get("bundle", {})
    items = bundle.get("items", [])
    resource = bundle.get("resource", {})
    base = ex.get("base", {})
    dataset = ex.get("dataset", "")

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
        "DECISION: "
    )
    return prompt


# --------------------------------
# HF generation
# --------------------------------
@torch.inference_mode()
def run_hf_generation(model_id, prompts, args):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device.startswith("cuda"):
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    elif args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(args.device)

    model.eval()

    all_texts = []
    all_scores = []
    all_group_sizes = []

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        if args.device == "cuda":
            toks = {k: v.to(model.device) for k, v in toks.items()}
        else:
            toks = {k: v.to(args.device) for k, v in toks.items()}

        gen_kwargs = dict(
            **toks,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
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
                    num_return_sequences=args.nbest
                )
            else:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    num_return_sequences=args.nbest
                )

        gen = model.generate(**gen_kwargs)

        bsz = len(batch)
        all_group_sizes.extend([cur_nbest] * bsz)

        prompt_len = toks["input_ids"].shape[1]
        for j in range(bsz * cur_nbest):
            new_tok = gen.sequences[j][prompt_len:]
            text = tokenizer.decode(new_tok, skip_special_tokens=True)
            all_texts.append(text)

        all_scores.append(gen.scores[0].detach().cpu())

    all_scores = torch.cat(all_scores, dim=0) if len(all_scores) else None
    return all_texts, all_scores, all_group_sizes, tokenizer


# --------------------------------
# OpenRouter
# --------------------------------
import requests

def openrouter_generate(prompt, args):

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": args.or_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }

    headers = {
        "Authorization": f"Bearer {args.or_api_key}",
        "Content-Type": "application/json"
    }

    r = requests.post(url, headers=headers, json=payload, timeout=300)

    if r.status_code != 200:
        raise RuntimeError(r.text)

    return r.json()["choices"][0]["message"]["content"]


# -----------------------------
# Parsing
# -----------------------------
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


# --------------------------------
# RUN
# --------------------------------
def run(args):

    if args.dataset_prefix:
        pattern = os.path.join(
            args.pred_dir,
            f"{args.dataset_prefix}_{args.model_tag}_preds_*.jsonl"
        )
    else:
        pattern = os.path.join(
            args.pred_dir,
            f"*_{args.model_tag}_preds_*.jsonl"
        )

    pred_files = sorted(glob.glob(pattern))

    if not pred_files:
        raise ValueError(f"No prediction files found for model_tag={args.model_tag}")

    print("\n[FOUND FILES]")
    for pf in pred_files:
        print(" -", pf)

    frames = list(load_jsonl(args.framing_path))
    frame_index = build_index(frames)

    for pred_file in pred_files:
        preds = list(load_jsonl(pred_file))

        prompts = []
        meta = []

        for p in preds:
            fkey = (p["id"], p.get("option_order", "original"))
            if fkey not in frame_index:
                continue

            vt = apply_value_tinted_narration(p, frame_index[fkey])
            if not vt:
                continue

            ex_counter, counter = vt
            prompt = build_decision_prompt(ex_counter)

            prompts.append(prompt)
            meta.append((p, frame_index[fkey], counter))
        
        print(prompts)
        
        # -------------------------
        # generation
        # -------------------------
        if args.backend == "hf":
            raw_outs, conf_outs, group_sizes, tokenizer = run_hf_generation(args.model_id, prompts, args)
        else:
            raw_outs = [openrouter_generate(p, args) for p in prompts]
            conf_outs = None
            group_sizes = [1] * len(raw_outs)


        outputs = []

        #tokenizer = AutoTokenizer.from_pretrained(args.model_id)

        offset = 0
        for (p, ex, counter), n in zip(meta, group_sizes):

            decision_space = ex.get("label_space", [])
            keys = tuple(decision_space + ["tie"])

            # nbest raw outputs
            outs_n = raw_outs[offset: offset + n]
            logits_n = conf_outs[offset: offset + n] if conf_outs is not None else None
            offset += n
            
            # ------------------------
            # Full vocabulary entropy
            # ------------------------
            H_full_mean = None

            if logits_n is not None:
                H_full_list = [
                    entropy_full_from_logits(logits_n[i])
                    for i in range(logits_n.shape[0])
                ]
                H_full_mean = sum(H_full_list) / len(H_full_list)

            preds_n = [parse_pred_decision(o, decision_space) for o in outs_n]

            conf_list = []
            if logits_n is not None and tokenizer is not None:
                probs = F.softmax(logits_n.float(), dim=-1)  # (n, vocab)

                for i in range(probs.shape[0]):
                    conf_dict = {}
                    for d in decision_space + ["tie"]:
                        tok_id = get_decision_token_id(tokenizer, d)

                        if tok_id is not None and tok_id < probs.shape[-1]:
                            conf_dict[d] = float(probs[i, tok_id].item())
                        else:
                            conf_dict[d] = 0.0
                    conf_list.append(conf_dict)
            else:
                conf_list = [None] * n

            # ---- aggregate: 평균 분포 → 조건부 정규화 → entropy
            # (너가 말한 “flip rate + 조건부 정규화 prob” 흐름과 궁합 좋음)
            avg_dist = {}
            if conf_list and isinstance(conf_list[0], dict):
                for d in keys:
                    avg_dist[d] = sum(c.get(d, 0.0) for c in conf_list) / max(len(conf_list), 1)
            else:
                avg_dist = {d: 0.0 for d in keys}

            cond = conditional_normalize(avg_dist, keys=keys)
            H = entropy_from_probs(cond, keys=keys)

            # (선택) nbest별 entropy 평균
            H_list = []
            if conf_list and isinstance(conf_list[0], dict):
                for c in conf_list:
                    ccond = conditional_normalize(c, keys=keys)
                    H_list.append(entropy_from_probs(ccond, keys=keys))
            H_mean = sum(H_list)/len(H_list) if H_list else None

            # 기존 margin도 “avg_dist 기반”으로 만들면 일관됨
            margin = None
            if len(decision_space) >= 2:
                d1, d2 = decision_space[:2]
                margin = avg_dist.get(d1, 0.0) - avg_dist.get(d2, 0.0)

            outputs.append({
                "id": ex["id"],
                "dataset": ex.get("dataset"),
                "domain": ex.get("domain"),
                "option_order": ex.get("option_order", "original"),

                "base_pred_decision": p.get("pred_decision"),
                "base_raw_output": p.get("raw_model_output"),

                # n-best raw
                "counter_raw_outputs": [o.strip() for o in outs_n],
                "counter_pred_decisions": preds_n,

                # 기존 필드 유지용: 대표값(첫번째) 하나도 남겨두기
                "counter_pred_decision": majority_vote(preds_n),
                "counter_raw_output": outs_n[0].strip() if outs_n else "",

                "decision_space": decision_space,
                "counter_framing": counter,

                # confidence는 대표 1개 대신 nbest list + avg + cond + entropy를 추가
                "counter_confidences": conf_list,     # n개
                "counter_confidence_avg": avg_dist,   # 평균 분포
                "counter_confidence_cond": cond,      # 조건부 정규화 분포 (너의 메인 지표)
                "entropy_cond": H,                    # 조건부 엔트로피
                "entropy_cond_mean": H_mean,          # (옵션) nbest별 평균 엔트로피
                "entropy_full_mean": H_full_mean,     # 조건부 아닌 전체 토큰에 대한 엔트로피

                "confidence_margin": margin,
                
            })

        os.makedirs(args.out_path, exist_ok=True)

        out_file = os.path.join(
            args.out_path,
            os.path.basename(pred_file).replace(".jsonl","_value_tinted_counter.jsonl")
        )

        write_jsonl(out_file, outputs)
        print("[DONE]", out_file)


# --------------------------------
# CLI
# --------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--framing_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--dataset_prefix", type=str, default=None)
    
    parser.add_argument("--backend", default="hf", choices=["hf","openrouter"])
    parser.add_argument("--or_api_key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--or_model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)


    parser.add_argument("--model_id", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode_mode", choices=["beam", "sample"], default="beam")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    run(args)

