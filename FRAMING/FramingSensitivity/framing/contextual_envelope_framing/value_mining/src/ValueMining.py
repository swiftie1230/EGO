#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
from typing import Dict, Any

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
# Value mining prompt
# -----------------------------
def build_value_prompt(instance: Dict[str, Any]) -> str:
    instance_id = instance.get("id", "") or instance.get("instance_id", "")
    label_space = instance.get("label_space", [])

    if not isinstance(label_space, list) or len(label_space) != 2:
        raise ValueError(f"label_space must be binary, got: {label_space}")

    return f"""
You are performing INSTANCE-LEVEL VALUE MINING for decision analysis.

Your task:
For each option, first identify interpretive perspectives
under which that option appears reasonable or compelling.
Then, for EACH perspective, determine which ONE of the following
Schwartz values it primarily instantiates.

Schwartz values:
{", ".join(SCHWARTZ_VALUES)}

Rules:
- Do NOT introduce new facts or outcomes.
- Do NOT give recommendations.
- Do NOT describe personas, roles, or identities.
- Focus on interpretation, value prioritization, and decision principles.
- Identify 2-3 perspectives per option.
- Each perspective MUST map to exactly ONE Schwartz value.

Return JSON ONLY using the schema below.

{{
  "instance_id": "{instance_id}",
  "label_space": {json.dumps(label_space, ensure_ascii=False)},

  "{label_space[0]}_values": [
    {{
      "perspective_id": "short_snake_case",
      "supports_option": "{label_space[0]}",
      "perspective_description": "interpretive lens that makes the option plausible",
      "instantiated_value": "ONE Schwartz value from the list",
      "value_rationale": "why this perspective instantiates that value",
      "decision_principle": "trade-off rule implied by the value",
      "attention_focus": ["salient aspects emphasized"]
    }}
  ],

  "{label_space[1]}_values": [
    {{
      "perspective_id": "short_snake_case",
      "supports_option": "{label_space[1]}",
      "perspective_description": "interpretive lens that makes the option plausible",
      "instantiated_value": "ONE Schwartz value from the list",
      "value_rationale": "why this perspective instantiates that value",
      "decision_principle": "trade-off rule implied by the value",
      "attention_focus": ["salient aspects emphasized"]
    }}
  ]
}}

Input instance:
{json.dumps(instance, ensure_ascii=False, indent=2)}
""".strip()


# -----------------------------
# Safe JSON parsing
# -----------------------------
import json

def safe_load(text: str):
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) find first JSON object and parse it with brace tracking,
    # ignoring braces inside strings.
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

    # 여기까지 왔으면 진짜로 잘렸거나(끝에 } 없음), JSON이 아닌 케이스
    raise ValueError("Unbalanced JSON braces (likely truncated output or braces inside strings)")


# -----------------------------
# OpenRouter call
# -----------------------------
def call_openrouter(prompt: str, api_key: str, model: str,
                    temperature: float, max_tokens: int,
                    timeout: int = 90, retries: int = 3) -> Dict[str, Any]:

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
    raise last_err


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
            temperature: float, max_new_tokens: int) -> Dict[str, Any]:
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
    parser.add_argument("--max_tokens", type=int, default=1200)

    # HuggingFace
    parser.add_argument("--hf_model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf_device", default="auto")
    parser.add_argument("--hf_dtype", default="bfloat16")
    parser.add_argument("--hf_max_new_tokens", type=int, default=1200)
    
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


    for fname in os.listdir(args.input_dir):
        if not fname.endswith(".jsonl"):
            continue

        in_path = os.path.join(args.input_dir, fname)

        suffix = ""
        if args.model_tag is not None:
            suffix += f"_{args.model_tag}"

        out_path = os.path.join(
            args.output_dir,
            fname.replace(".jsonl", f"_values{suffix}.jsonl")
        )

        print(f"[RUN] {fname}")

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                try:
                    instance = json.loads(line)
                except Exception as e:
                    print(f"[WARN] invalid JSON at line {i} in {fname}")
                    continue

                if allowed_domains is not None:
                    if instance.get("domain") not in allowed_domains:
                        print(f"  [SKIP] domain={instance.get('domain')}")
                        continue

                label_space = instance.get("label_space", [])

                if not isinstance(label_space, list) or len(label_space) != 2:
                    print(f"[SKIP non-binary] {instance.get('id')} label_space={label_space}")
                    continue
                
                prompt = build_value_prompt(instance)

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

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"[OK] saved → {out_path}")



if __name__ == "__main__":
    main()

