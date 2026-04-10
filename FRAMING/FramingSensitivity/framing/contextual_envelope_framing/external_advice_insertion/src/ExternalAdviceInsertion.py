# ValueExternalAdviceInsertion.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

from llm_client import OpenRouterClient, HuggingFaceClient


# ----------------------------
# IO utils
# ----------------------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# Base text extraction
# ----------------------------
def extract_base_text(skel: Dict[str, Any]) -> Tuple[str, str]:
    shared_story = safe_get(skel, ["bundle", "shared_story"])
    if isinstance(shared_story, str) and shared_story.strip():
        return shared_story.strip(), "bundle.shared_story"

    ggb_vignette = safe_get(skel, ["base", "vignette"])
    if isinstance(ggb_vignette, str) and ggb_vignette.strip():
        return ggb_vignette.strip(), "base.vignette"

    unibench_scenario = safe_get(skel, ["base", "scenario"])
    if isinstance(unibench_scenario, str) and unibench_scenario.strip():
        return unibench_scenario.strip(), "base.scenario"

    items = safe_get(skel, ["bundle", "items"], [])
    if isinstance(items, list) and items:
        lines = []
        for it in items:
            st = it.get("situation_text")
            if isinstance(st, str) and st.strip():
                lines.append(st.strip())
        if lines:
            return " ".join(lines), "bundle.items[*].situation_text"

    raise ValueError(f"Cannot extract base narrative text from skeleton id={skel.get('id') or skel.get('instance_id')}")


# ----------------------------
# Value parsing
# ----------------------------
def extract_option_values(value_obj: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Supports:
    - A_values / B_values
    - patient_1_values / patient_2_values
    """
    option_map: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in value_obj.items():
        if not k.endswith("_values"):
            continue
        if not isinstance(v, list):
            continue
        option_id = k.replace("_values", "")
        option_map[option_id] = v
    return option_map


def get_instance_key(obj: Dict[str, Any]) -> str:
    return str(obj.get("id") or obj.get("instance_id") or obj.get("example_id"))


def index_value_file(value_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(value_jsonl_path):
        key = get_instance_key(row)
        if key:
            idx[key] = row
    return idx


# ----------------------------
# Prompt (Value-based External Advice)
# ----------------------------
EXTERNAL_ADVICE_PROMPT = """You are a controlled framing generation assistant.

Your task is to generate ONE short perspective statement
that could be naturally appended to the scenario.

IMPORTANT CONSTRAINTS:
- Do NOT add new facts, risks, outcomes, entities, or details.
- Do NOT modify the original scenario text.
- Do NOT introduce hypothetical events or imagined consequences.
- Base the statement strictly on the provided information.
- Do NOT reference value theory or terminology explicitly.
- Do NOT present a recommendation or conclusion.

STYLE AND LENGTH:
- Neutral, descriptive tone.
- NOT advisory or imperative.
- ONE concise sentence only.

OUTPUT RULE:
- Output ONLY the perspective statement.
"""


def build_external_advice_prompt(
    base_text: str,
    decision_question: str,
    options: List[Dict[str, Any]],
    value_frame: Dict[str, Any],
) -> str:
    perspective_desc = value_frame.get("perspective_description", "")
    decision_principle = value_frame.get("decision_principle", "")
    attention_focus = value_frame.get("attention_focus", [])
    supports_option = value_frame.get("supports_option", "")

    opt_lines = []
    for it in options:
        item_id = it.get("item_id")
        text = it.get("text") or it.get("situation_text") or ""
        if item_id is not None:
            opt_lines.append(f"{item_id}: {str(text).strip()}")

    af = ", ".join(attention_focus) if isinstance(attention_focus, list) else str(attention_focus)

    prompt = f"""{EXTERNAL_ADVICE_PROMPT}

SCENARIO:
{base_text}

DECISION QUESTION:
{decision_question}

OPTIONS:
{chr(10).join(opt_lines)}

VALUE-BASED INTERPRETIVE LENS (implicit):
- Interpretive perspective: {perspective_desc}
- Decision principle: {decision_principle}
- Attention focus: {af}
- (This perspective tends to support option: {supports_option})

INSTRUCTION:
Write ONE sentence that reflects what stands out or weighs most heavily
under this value-based perspective.

Do NOT recommend any option.
Do NOT add new facts.
OUTPUT ONLY the sentence.
"""
    return prompt


# ----------------------------
# Generation driver
# ----------------------------
def make_client(args):
    if args.backend == "openrouter":
        return OpenRouterClient(
            api_key=args.or_api_key,
            site=args.or_site,
            app=args.or_app,
            retries=args.retries,
        )
    return HuggingFaceClient(
        model_id=args.model_id,
        device=args.hf_device,
        dtype=args.hf_dtype,
    )


def generate_one(client, prompt: str, args) -> str:
    if args.backend == "openrouter":
        return client.generate(
            prompt,
            model=args.or_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            retries=args.retries,
        )
    return client.generate(
        prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )


def ensure_framings(skel: Dict[str, Any]) -> Dict[str, Any]:
    skel.setdefault("framings", {})
    if not isinstance(skel["framings"], dict):
        skel["framings"] = {}
    return skel["framings"]


# ----------------------------
# Main run
# ----------------------------
def run(args):
    value_index = index_value_file(args.value_jsonl)
    client = make_client(args)

    out_rows: List[Dict[str, Any]] = []

    for skel in iter_jsonl(args.skeleton_jsonl):
        skel_id = get_instance_key(skel)

        if skel_id not in value_index:
            out_rows.append(skel)
            continue

        value_obj = value_index[skel_id]
        option_to_values = extract_option_values(value_obj)

        base_text, base_src = extract_base_text(skel)
        decision_question = skel.get("decision_question", "")
        options = safe_get(skel, ["bundle", "items"], []) or []

        framings = ensure_framings(skel)
        ea_root = framings.setdefault("external_advice_value", {})

        for option_id, values in option_to_values.items():
            option_field = ea_root.setdefault(option_id, {})

            for v in values:
                vid = v.get("perspective_id") or "value_frame"

                if (
                    not args.overwrite
                    and vid in option_field
                    and option_field[vid].get("text")
                ):
                    continue

                prompt = build_external_advice_prompt(
                    base_text=base_text,
                    decision_question=decision_question,
                    options=options,
                    value_frame=v,
                )

                text = generate_one(client, prompt, args).strip()

                option_field[vid] = {
                    "text": text,
                    "source_field": base_src,
                    "supports_option": option_id,
                    "instantiated_value": v.get("instantiated_value"),
                    "decision_principle": v.get("decision_principle"),
                    "attention_focus": v.get("attention_focus"),
                }

        out_rows.append(skel)

    write_jsonl(args.output_jsonl, out_rows)
    print(f"[DONE] wrote: {args.output_jsonl}")


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("Value-Based External Advice Generation")

    p.add_argument("--skeleton_jsonl", type=str, required=True)
    p.add_argument("--value_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--backend", type=str, choices=["hf", "openrouter"], default="hf")

    # HF
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--hf_device", type=str, default="auto")
    p.add_argument("--hf_dtype", type=str, default="auto")

    # OpenRouter
    p.add_argument("--or_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--or_model", type=str, default="openai/gpt-4.1-mini")
    p.add_argument("--or_site", type=str, default="")
    p.add_argument("--or_app", type=str, default="")

    # sampling
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--retries", type=int, default=5)

    p.add_argument("--overwrite", action="store_true")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)

