#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
from typing import Dict, Any, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCHWARTZ_VALUES = [
    "Self-direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
]


# -----------------------------
# Helpers
# -----------------------------
def get_instance_id(instance: Dict[str, Any]) -> str:
    return instance.get("id", "") or instance.get("instance_id", "")


def extract_option_text(item: Dict[str, Any]) -> str:
    """
    Robust option text extraction across skeleton styles.
    """
    if not isinstance(item, dict):
        return str(item)

    for key in ["text", "situation_text", "option_text", "statement", "description"]:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return json.dumps(item, ensure_ascii=False)


def build_option_map(instance: Dict[str, Any]) -> Dict[str, str]:
    """
    Map label -> human-readable option text using bundle.items.
    Works for legal skeletons, medical triage skeletons, etc.
    """
    label_space = instance.get("label_space", [])
    items = instance.get("bundle", {}).get("items", [])

    option_map = {}

    # 1) prefer item_id matching
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            item_id = it.get("item_id")
            if item_id in label_space:
                option_map[item_id] = extract_option_text(it)

    # 2) fallback: positional match if item_id missing
    if len(option_map) < len(label_space) and isinstance(items, list):
        for idx, label in enumerate(label_space):
            if label not in option_map and idx < len(items):
                option_map[label] = extract_option_text(items[idx])

    # 3) final fallback
    for label in label_space:
        option_map.setdefault(label, f"Option {label}")

    return option_map


def build_compact_context(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact, model-friendly context extracted from heterogeneous skeleton formats.
    """
    base = instance.get("base", {}) if isinstance(instance.get("base"), dict) else {}
    meta = instance.get("meta", {}) if isinstance(instance.get("meta"), dict) else {}

    label_space = instance.get("label_space", [])
    option_map = build_option_map(instance)

    compact = {
        "id": get_instance_id(instance),
        "dataset": instance.get("dataset"),
        "domain": instance.get("domain"),
        "decision_question": instance.get("decision_question"),
        "option_order": instance.get("option_order"),
        "label_space": label_space,
        "options": {label: option_map.get(label, f"Option {label}") for label in label_space},
        "scenario": base.get("scenario"),
    }

    if base.get("legal_issue"):
        compact["legal_issue"] = base.get("legal_issue")
    if base.get("value_dimension"):
        compact["value_dimension"] = base.get("value_dimension")
    if base.get("state") is not None:
        compact["state"] = base.get("state")
    if meta.get("task_type"):
        compact["task_type"] = meta.get("task_type")
    if meta.get("case_title"):
        compact["case_title"] = meta.get("case_title")
    elif base.get("case_title"):
        compact["case_title"] = base.get("case_title")
    if meta.get("citation"):
        compact["citation"] = meta.get("citation")
    if meta.get("year"):
        compact["year"] = meta.get("year")
    if base.get("core_phrases_preserved"):
        compact["core_phrases_preserved"] = base.get("core_phrases_preserved")

    return compact


# -----------------------------
# Value mining prompt
# -----------------------------
def build_value_prompt(instance: Dict[str, Any]) -> str:
    instance_id = get_instance_id(instance)
    label_space = instance.get("label_space", [])

    if not isinstance(label_space, list) or len(label_space) != 2:
        raise ValueError(f"label_space must be binary, got: {label_space}")

    option_map = build_option_map(instance)
    compact_context = build_compact_context(instance)

    a_label, b_label = label_space[0], label_space[1]
    a_text = option_map[a_label]
    b_text = option_map[b_label]

    return f"""
You are performing INSTANCE-LEVEL VALUE MINING for decision analysis.

Goal:
For each option, identify 2-3 interpretive perspectives under which that option
appears reasonable, compelling, or defensible.
Then, for EACH perspective, assign exactly ONE primary Schwartz value.

Schwartz values:
{", ".join(SCHWARTZ_VALUES)}

Global rules:
- Do NOT introduce new facts, outcomes, evidence, or hidden intentions.
- Do NOT give recommendations or predict the correct answer.
- Do NOT create fictional personas, roles, or identities not grounded in the input.
- Do NOT merely restate the option text; identify the underlying value-oriented interpretation that makes it appealing.
- Focus on interpretive lenses, value prioritization, and decision principles.
- Each perspective MUST support exactly one option.
- Each perspective MUST map to exactly ONE Schwartz value.
- Keep perspectives distinct and non-redundant.
- Keep the output grounded in the provided scenario and option texts only.
- Use concise but meaningful descriptions.

Return JSON ONLY using the schema below.

{{
  "id": {json.dumps(instance.get("id"), ensure_ascii=False)},
  "instance_id": "{instance_id}",
  "option_order": {json.dumps(instance.get("option_order"), ensure_ascii=False)},
  "dataset": {json.dumps(instance.get("dataset"), ensure_ascii=False)},
  "domain": {json.dumps(instance.get("domain"), ensure_ascii=False)},
  "label_space": {json.dumps(label_space, ensure_ascii=False)},
  "options": {{
    "{a_label}": {json.dumps(a_text, ensure_ascii=False)},
    "{b_label}": {json.dumps(b_text, ensure_ascii=False)}
  }},

  "{a_label}_values": [
    {{
      "perspective_id": "short_snake_case",
      "supports_option": "{a_label}",
      "option_text": {json.dumps(a_text, ensure_ascii=False)},
      "perspective_description": "interpretive lens that makes this option plausible",
      "instantiated_value": "ONE Schwartz value from the list",
      "value_rationale": "why this perspective primarily instantiates that value",
      "decision_principle": "trade-off rule or decision logic implied by the value",
      "attention_focus": ["salient aspects emphasized"]
    }}
  ],

  "{b_label}_values": [
    {{
      "perspective_id": "short_snake_case",
      "supports_option": "{b_label}",
      "option_text": {json.dumps(b_text, ensure_ascii=False)},
      "perspective_description": "interpretive lens that makes this option plausible",
      "instantiated_value": "ONE Schwartz value from the list",
      "value_rationale": "why this perspective primarily instantiates that value",
      "decision_principle": "trade-off rule or decision logic implied by the value",
      "attention_focus": ["salient aspects emphasized"]
    }}
  ]
}}

Compact instance context:
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

Full input instance:
{json.dumps(instance, ensure_ascii=False, indent=2)}
""".strip()


# -----------------------------
# Safe JSON parsing
# -----------------------------
def safe_load(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:i+1]
                    return json.loads(snippet)

    raise ValueError("Unbalanced JSON braces (likely truncated output or braces inside strings)")


# -----------------------------
# Validation
# -----------------------------
def validate_result(result: Optional[Dict[str, Any]], label_space) -> bool:
    if result is None:
        return False
    if not isinstance(result, dict):
        return False

    for label in label_space:
        key = f"{label}_values"
        if key not in result or not isinstance(result[key], list):
            return False

    return True


# -----------------------------
# OpenRouter call
# -----------------------------
def call_openrouter(prompt: str, api_key: str, model: str,
                    temperature: float, max_tokens: int,
                    timeout: int = 90, retries: int = 3) -> Optional[Dict[str, Any]]:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a careful research assistant. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    last_err = None
    for t in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return safe_load(content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (t + 1))

    print(f"[WARN] OpenRouter failed: {last_err}")
    return None


# -----------------------------
# Hugging Face call
# -----------------------------
def load_hf_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_map[dtype],
        device_map="auto" if device == "auto" else {"": device},
    )
    model.eval()
    return tok, model


def call_hf(prompt: str, tokenizer, model,
            temperature: float, max_new_tokens: int) -> Optional[Dict[str, Any]]:
    import torch

    messages = [
        {"role": "system", "content": "You are a careful research assistant. Output JSON only."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    try:
        return safe_load(decoded)
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}")
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--backend", choices=["openrouter", "hf"], default="openrouter")

    # OpenRouter
    parser.add_argument("--model", default="openai/gpt-4.0-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1400)

    # HuggingFace
    parser.add_argument("--hf_model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf_device", default="auto")
    parser.add_argument("--hf_dtype", default="bfloat16")
    parser.add_argument("--hf_max_new_tokens", type=int, default=1400)

    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Optional tag appended to output filename"
    )
    parser.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Comma-separated list of allowed domains"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to process per file"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (0-based) within each file"
    )

    args = parser.parse_args()

    if args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
    else:
        tokenizer, model = load_hf_model(
            args.hf_model_id, args.hf_device, args.hf_dtype
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.domain_filter is not None:
        allowed_domains = set(
            d.strip() for d in args.domain_filter.split(",") if d.strip()
        )
    else:
        allowed_domains = None

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".jsonl"):
            continue

        in_path = os.path.join(args.input_dir, fname)

        suffix = ""
        if args.model_tag is not None:
            suffix += f"_{args.model_tag}"
        if args.start_idx != 0:
            suffix += f"_start{args.start_idx}"
        if args.limit is not None:
            suffix += f"_limit{args.limit}"

        out_path = os.path.join(
            args.output_dir,
            fname.replace(".jsonl", f"_values{suffix}.jsonl")
        )

        print(f"[RUN] {fname}")
        print(f"      start_idx={args.start_idx}, limit={args.limit}")

        processed_in_file = 0
        seen_valid_lines = 0

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                try:
                    instance = json.loads(line)
                except Exception:
                    print(f"[WARN] invalid JSON at raw line {i} in {fname}")
                    continue

                # count valid JSON instances for slicing
                current_idx = seen_valid_lines
                seen_valid_lines += 1

                if current_idx < args.start_idx:
                    continue

                if args.limit is not None and processed_in_file >= args.limit:
                    break

                if allowed_domains is not None:
                    if instance.get("domain") not in allowed_domains:
                        print(f"  [SKIP domain] idx={current_idx} domain={instance.get('domain')}")
                        continue

                label_space = instance.get("label_space", [])
                if not isinstance(label_space, list) or len(label_space) != 2:
                    print(f"[SKIP non-binary] idx={current_idx} id={instance.get('id')} label_space={label_space}")
                    continue

                try:
                    prompt = build_value_prompt(instance)
                except Exception as e:
                    print(f"[WARN] prompt build failed for idx={current_idx} id={instance.get('id')}: {e}")
                    fout.write("null\n")
                    processed_in_file += 1
                    continue

                if args.backend == "openrouter":
                    result = call_openrouter(
                        prompt, api_key, args.model,
                        args.temperature, args.max_tokens
                    )
                else:
                    result = call_hf(
                        prompt, tokenizer, model,
                        args.temperature, args.hf_max_new_tokens
                    )

                if not validate_result(result, label_space):
                    print(f"[WARN] invalid result schema at idx={current_idx} id={instance.get('id')}")
                    result = None

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_in_file += 1

                if processed_in_file % 10 == 0:
                    print(f"  processed={processed_in_file}")

        print(f"[OK] saved → {out_path} (processed {processed_in_file} instances)")
        

if __name__ == "__main__":
    main()