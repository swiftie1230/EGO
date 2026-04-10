#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
import re
from typing import Dict, Any, List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Persona mining prompt
# -----------------------------
def build_persona_prompt(instance: Dict[str, Any]) -> str:
    instance_id = instance.get("id", "") or instance.get("instance_id", "")
    label_space = instance.get("label_space", [])

    # ---- minimal repair for corrupted 3-choice: ['A','B','C'] where B is a ghost option ----
    # We keep A and C, and remap C -> B to preserve binary A/B downstream.
    if isinstance(label_space, list) and label_space == ["A", "B", "C"]:
        # Make a shallow copy so we don't mutate caller unexpectedly
        instance = dict(instance)

        # Fix label_space to canonical binary
        label_space = ["A", "B"]
        instance["label_space"] = label_space

        # If bundle.items exists, keep only A and C, and rename C -> B
        bundle = instance.get("bundle")
        if isinstance(bundle, dict):
            bundle = dict(bundle)
            items = bundle.get("items", [])
            if isinstance(items, list):
                new_items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    iid = it.get("item_id")
                    if iid == "A":
                        new_items.append(it)
                    elif iid == "C":
                        it2 = dict(it)
                        it2["item_id"] = "B"  # remap C -> B
                        new_items.append(it2)
                bundle["items"] = new_items
            instance["bundle"] = bundle

        # Optional: remap gold_label if present
        if instance.get("gold_label") == "C":
            instance["gold_label"] = "B"
        elif instance.get("gold_label") == "B":
            instance["gold_label"] = None

    # ---- enforce binary after repair ----
    if not isinstance(label_space, list) or len(label_space) != 2:
        raise ValueError(f"label_space must be a list of length 2, got: {label_space}")

    return f"""
You are performing INSTANCE-LEVEL PERSONA MINING for decision analysis.

A persona is a latent DECISION PRIOR — not a role or demographic label.

For each option, identify perspectives that would naturally
make that option feel justified.

Rules:
- No new facts or events.
- No recommendations.
- No generic roles.
- 3–5 personas per option.

Return JSON only using the following structure:

{{
  "instance_id": "{instance_id}",
  "label_space": {json.dumps(label_space, ensure_ascii=False)},

  "{label_space[0]}_personas": [
    {{
      "persona_id": "short_snake_case",
      "supports_option": "{label_space[0]}",
      "persona_label": "prior-based name",
      "core_prior": "belief or assumption",
      "decision_heuristic": "trade-off rule",
      "attention_focus": ["..."],
      "value_priority": ["..."]
    }}
  ],

  "{label_space[1]}_personas": [
    {{
      "persona_id": "short_snake_case",
      "supports_option": "{label_space[1]}",
      "persona_label": "prior-based name",
      "core_prior": "belief or assumption",
      "decision_heuristic": "trade-off rule",
      "attention_focus": ["..."],
      "value_priority": ["..."]
    }}
  ]
}}

Input instance:
{json.dumps(instance, ensure_ascii=False, indent=2)}
""".strip()



# -----------------------------
# Safe JSON parsing
# -----------------------------
#_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
#_JSON_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.S)

def extract_first_json(text: str):
    """
    Extract the first valid JSON object from text
    using brace counting (Python-safe).
    """
    text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)

    raise ValueError("Unbalanced JSON braces")



#def safe_load(text: str):
#    text = text.strip()

    # 1. direct parse
#    try:
#        return json.loads(text)
#    except Exception:
#        pass

    # 2. brace-balanced extraction
#    return extract_first_json(text)


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
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])

    raise ValueError("Unbalanced JSON braces")



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
        # If your provider/model supports it, this helps a lot:
        # "response_format": {"type": "json_object"},
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
def _ensure_transformers():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Hugging Face backend requires `transformers` and `torch`.\n"
            "Install: pip install -U transformers accelerate torch"
        ) from e


def load_hf_model(model_id: str, device: str, dtype: str):
    _ensure_transformers()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # dtype parsing
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"--hf_dtype must be one of {list(dtype_map.keys())}, got {dtype}")

    torch_dtype = dtype_map[dtype]

    # device_map: "auto" (accelerate) or explicit
    if device == "auto":
        device_map = "auto"
    else:
        # e.g., "cuda:0" or "cpu" or "mps"
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return tok, model


def call_hf(prompt: str, tokenizer, model,
            temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    _ensure_transformers()
    import torch

    system_msg = "You are a careful research assistant. Output JSON only."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    # Prefer chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
    else:
        # Fallback: simple concatenation
        text = f"{prompt}\n"
        inputs = tokenizer(text, return_tensors="pt")

    # Move to the model device
    for k in inputs:
        if hasattr(model, "device"):
            inputs[k] = inputs[k].to(model.device)

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # If chat template was used, decoded includes the whole transcript sometimes.
    # Extract last JSON object.
    return safe_load(decoded)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--backend", choices=["openrouter", "hf"], default="openrouter")

    # OpenRouter args
    parser.add_argument("--model", default="openai/gpt-4.0-mini")
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="String tag appended to output filename (e.g., Qwen_7B)"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1200)

    # HF args
    parser.add_argument("--hf_model_id", type=str, default=None,
                        help="e.g., Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf_device", type=str, default="auto",
                        help='e.g., "auto", "cuda:0", "cpu", "mps"')
    parser.add_argument("--hf_dtype", type=str, default="auto",
                        help='auto|float16|bfloat16|float32')
    parser.add_argument("--hf_max_new_tokens", type=int, default=1200)

    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Process only first N instances per file (for debugging)"
    )
    parser.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Only process instances with this domain (e.g., life_safety). Comma-separated allowed."
    )
    args = parser.parse_args()

    if args.domain_filter is not None:
        allowed_domains = set(d.strip() for d in args.domain_filter.split(",") if d.strip())
    else:
        allowed_domains = None

    # Backend setup
    api_key = None
    hf_tokenizer = None
    hf_model = None

    if args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
    else:
        if not args.hf_model_id:
            raise RuntimeError("--hf_model_id is required when --backend=hf")
        hf_tokenizer, hf_model = load_hf_model(args.hf_model_id, args.hf_device, args.hf_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".json") or f.endswith(".jsonl")]

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        suffix = ""

        if args.model_tag is not None:
            suffix += f"_{args.model_tag}"

        if args.max_instances is not None:
            suffix += f"_{args.max_instances}"

        if fname.endswith(".jsonl"):
            out_name = fname[:-6] + f"_personas{suffix}.jsonl"
        else:
            out_name = fname[:-5] + f"_personas{suffix}.json"

        out_path = os.path.join(args.output_dir, out_name)

        print(f"[RUN] {fname}")

        def run_one(instance: Dict[str, Any]) -> Dict[str, Any]:
            prompt = build_persona_prompt(instance)
            if args.backend == "openrouter":
                return call_openrouter(prompt, api_key, args.model, args.temperature, args.max_tokens)
            else:
                return call_hf(prompt, hf_tokenizer, hf_model, args.temperature, args.hf_max_new_tokens)

        if fname.endswith(".json"):
            instance = json.load(open(in_path))

            if allowed_domains is not None and instance.get("domain") not in allowed_domains:
                print(f"  [SKIP] domain={instance.get('domain')}")
                continue

            result = run_one(instance)
            json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)

        else:  # jsonl
            rows = []
            with open(in_path) as f:
                for i, line in enumerate(f):
                    if args.max_instances is not None and i >= args.max_instances:
                        break

                    instance = json.loads(line)

                    if allowed_domains is not None and instance.get("domain") not in allowed_domains:
                        print(f"  [SKIP] domain={instance.get('domain')}")
                        continue

                    rows.append(run_one(instance))

            if len(rows) == 0:
                print("  [SKIP] no instances matched domain filter")
                continue

            with open(out_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[OK ] saved → {out_path}")

    print("DONE.")


if __name__ == "__main__":
    main()

