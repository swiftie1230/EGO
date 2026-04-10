# ValueTintedNarration.py
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


def stringify_option(it: Dict[str, Any]) -> str:
    # 1) 일반적인 option format
    text = it.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    situation_text = it.get("situation_text")
    if isinstance(situation_text, str) and situation_text.strip():
        return situation_text.strip()

    # 2) RoleConflict format
    parts = []

    role_name = it.get("role_name")
    obligation_level = it.get("obligation_level")
    situation = it.get("situation")
    expectation = it.get("expectation")

    if role_name:
        parts.append(f"Role: {role_name}")
    if obligation_level is not None:
        parts.append(f"Obligation level: {obligation_level}")
    if situation:
        parts.append(f"Situation: {situation}")
    if expectation:
        parts.append(f"Expectation: {expectation}")

    if parts:
        return " | ".join(parts)

    # 3) fallback
    return json.dumps(it, ensure_ascii=False, sort_keys=True)


def extract_current_option_map(options: List[Dict[str, Any]]) -> Dict[str, str]:
    out = {}
    for it in options:
        item_id = it.get("item_id")
        if item_id is not None:
            out[str(item_id)] = stringify_option(it)
    return out


# ----------------------------
# Base text extraction
# ----------------------------
def extract_base_text(skel: Dict[str, Any]) -> Tuple[str, str]:
    candidates = [
        (["bundle", "shared_story"], "bundle.shared_story"),
        (["base", "vignette"], "base.vignette"),
        (["base", "scenario"], "base.scenario"),
        (["scenario"], "scenario"),
    ]

    for keys, src in candidates:
        text = safe_get(skel, keys)
        if isinstance(text, str) and text.strip():
            return text.strip(), src

    raise ValueError(f"Cannot extract base text for id={skel.get('id') or skel.get('instance_id')}")


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
    if not isinstance(obj, dict):
        return ""

    base_id = obj.get("id") or obj.get("instance_id") or obj.get("example_id")
    option_order = obj.get("option_order")

    if base_id is None:
        return ""

    if option_order is not None:
        return f"{base_id}::{option_order}"
    return str(base_id)


def index_value_file(value_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl(value_jsonl_path):
        if not isinstance(row, dict):
            continue

        key = get_instance_key(row)
        if not key:
            continue

        idx[key] = row

    return idx


# ----------------------------
# Prompt
# ----------------------------
BASE_PROMPT = """You are a controlled narrative generation assistant.

Your task is to rewrite the given scenario so that it reflects
a specific VALUE-BASED INTERPRETIVE PERSPECTIVE.

IMPORTANT CONSTRAINTS:
- Do NOT add new facts, risks, outcomes, entities, or details.
- Do NOT remove or alter any existing factual information.
- Do NOT change the decision structure or available options.
- Do NOT give advice or state which option should be chosen.
- Preserve the original causal structure.
- Keep the rewritten scenario approximately similar in length to the original.
- Do not significantly expand or compress the scenario.

You may ONLY modify:
- narrative focus and salience
- what feels urgent or morally salient
- attention allocation and framing
- sentence structure and narrative style

Do not explicitly mention values or theory names.
Do not explain the perspective.
Do not output headings or analysis.

OUTPUT RULE:
- Output ONLY the rewritten scenario text.
"""


def build_value_tinted_prompt(
    base_text: str,
    decision_question: str,
    options: List[Dict[str, Any]],
    value_frame: Dict[str, Any],
) -> str:
    perspective_desc = value_frame.get("perspective_description", "")
    decision_principle = value_frame.get("decision_principle", "")
    attention_focus = value_frame.get("attention_focus", [])
    instantiated_value = value_frame.get("instantiated_value", "")
    supports_option = value_frame.get("supports_option", "")

    opt_lines = []
    for it in options:
        item_id = it.get("item_id")
        if item_id is not None:
            opt_lines.append(f"{item_id}: {stringify_option(it)}")

    af = ", ".join(attention_focus) if isinstance(attention_focus, list) else str(attention_focus)

    prompt = f"""{BASE_PROMPT}

SCENARIO (rewrite this, facts unchanged):
{base_text}

DECISION QUESTION (do not change):
{decision_question}

OPTIONS (do not change):
{chr(10).join(opt_lines)}

VALUE-BASED INTERPRETIVE LENS (implicit):
- Interpretive perspective: {perspective_desc}
- Decision principle: {decision_principle}
- Attention focus: {af}
- (This perspective tends to support option: {supports_option})

INSTRUCTION:
Rewrite the scenario so that this value-based perspective subtly shapes:
- which aspects feel central or urgent,
- how consequences are perceived,
- where attention naturally gravitates.

Again: do NOT recommend any option.
OUTPUT ONLY the rewritten scenario text.
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
            #retries=args.retries,
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
            #retries=args.retries,
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


def get_base_id(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("id") or obj.get("instance_id") or obj.get("example_id") or "")

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())

def normalize_option_repr(x: Any) -> str:
    if isinstance(x, dict):
        return normalize_text(stringify_option(x))

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""

        # JSON stringified dict 인 경우 복원
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return normalize_text(stringify_option(parsed))
        except Exception:
            pass

        return normalize_text(s)

    return normalize_text(str(x))


def build_label_remap_from_original_to_current(
    original_options: List[Dict[str, Any]],
    current_options: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Returns:
        original_label -> current_label
    by matching option content string.
    """
    orig_map = extract_current_option_map(original_options)
    curr_map = extract_current_option_map(current_options)

    curr_text_to_label = {
        normalize_text(v): k for k, v in curr_map.items()
    }

    remap = {}
    for orig_label, orig_text in orig_map.items():
        key = normalize_text(orig_text)
        if key in curr_text_to_label:
            remap[orig_label] = curr_text_to_label[key]

    return remap


def copy_value_tinted_from_original(
    original_skel: Dict[str, Any],
    current_skel: Dict[str, Any],
) -> None:
    """
    Copy value_tinted_narration from original row into current row
    by remapping option labels based on option text identity.
    """
    original_vt = safe_get(original_skel, ["framings", "value_tinted_narration"], {})
    if not isinstance(original_vt, dict) or not original_vt:
        return

    original_options = safe_get(original_skel, ["bundle", "items"], []) or []
    current_options = safe_get(current_skel, ["bundle", "items"], []) or []

    remap = build_label_remap_from_original_to_current(
        original_options=original_options,
        current_options=current_options,
    )

    framings = ensure_framings(current_skel)
    vt_root = framings.setdefault("value_tinted_narration", {})

    for orig_option_id, per_value_dict in original_vt.items():
        if orig_option_id not in remap:
            continue

        new_option_id = remap[orig_option_id]
        dst = vt_root.setdefault(new_option_id, {})

        if not isinstance(per_value_dict, dict):
            continue

        for vid, payload in per_value_dict.items():
            if not isinstance(payload, dict):
                continue

            copied = dict(payload)
            copied["supports_option"] = new_option_id
            copied["copied_from"] = "original"
            copied["original_supports_option"] = payload.get("supports_option", orig_option_id)
            dst[vid] = copied


# ----------------------------
# Main run
# ----------------------------
def run(args):
    value_index = index_value_file(args.value_jsonl)
    client = make_client(args)

    out_rows: List[Dict[str, Any]] = []

    # original row cache
    original_cache: Dict[str, Dict[str, Any]] = {}

    processed = 0
    seen = 0

    for skel in iter_jsonl(args.skeleton_jsonl):
        current_idx = seen
        seen += 1

        if current_idx < args.start_idx:
            continue

        if args.limit is not None and processed >= args.limit:
            break

        skel_id = get_instance_key(skel)
        base_id = get_base_id(skel)
        option_order = skel.get("option_order", "original")

        if skel_id not in value_index:
            out_rows.append(skel)
            processed += 1
            continue

        # ----------------------------
        # swapped: do NOT generate
        # just copy from original
        # ----------------------------
        if option_order == "swapped":
            original_skel = original_cache.get(base_id)
            if original_skel is None:
                print(f"[WARN] swapped row encountered before original for {base_id}")
            else:
                copy_value_tinted_from_original(
                    original_skel=original_skel,
                    current_skel=skel,
                )

            out_rows.append(skel)
            processed += 1

            if processed % 20 == 0:
                print(f"[progress] processed={processed}")
            continue

        # ----------------------------
        # original: normal generation
        # ----------------------------
        value_obj = value_index[skel_id]
        option_to_values = extract_option_values(value_obj)

        try:
            base_text, base_src = extract_base_text(skel)
        except Exception as e:
            print(f"[WARN] base text extraction failed for {skel_id}: {e}")
            out_rows.append(skel)
            processed += 1
            continue

        decision_question = skel.get("decision_question", "")
        options = safe_get(skel, ["bundle", "items"], []) or []

        current_option_map = extract_current_option_map(options)
        value_option_map = value_obj.get("options", {})

        if isinstance(value_option_map, dict) and value_option_map:
            cleaned_current_map = {k: normalize_option_repr(v) for k, v in current_option_map.items()}
            cleaned_value_map = {k: normalize_option_repr(v) for k, v in value_option_map.items()}

            if cleaned_current_map != cleaned_value_map:
                print(f"[WARN] option mismatch for {skel_id}")
                print(f"  skeleton options: {cleaned_current_map}")
                print(f"  value options   : {cleaned_value_map}")

        framings = ensure_framings(skel)
        vt_root = framings.setdefault("value_tinted_narration", {})

        for option_id, values in option_to_values.items():
            option_field = vt_root.setdefault(option_id, {})

            for v in values:
                vid = v.get("perspective_id") or "value_frame"

                if (
                    not args.overwrite
                    and vid in option_field
                    and option_field[vid].get("text")
                ):
                    continue

                prompt = build_value_tinted_prompt(
                    base_text=base_text,
                    decision_question=decision_question,
                    options=options,
                    value_frame=v,
                )

                print(prompt)

                try:
                    text = generate_one(client, prompt, args).strip()
                except Exception as e:
                    print(f"[WARN] generation failed for {skel_id} / {option_id} / {vid}: {e}")
                    continue

                option_field[vid] = {
                    "text": text,
                    "source_field": base_src,
                    "supports_option": option_id,
                    "instantiated_value": v.get("instantiated_value"),
                    "decision_principle": v.get("decision_principle"),
                    "attention_focus": v.get("attention_focus"),
                }

        # cache the fully generated original row
        original_cache[base_id] = json.loads(json.dumps(skel, ensure_ascii=False))

        out_rows.append(skel)
        processed += 1

        if processed % 20 == 0:
            print(f"[progress] processed={processed}")

    write_jsonl(args.output_jsonl, out_rows)
    print(f"[DONE] wrote {args.output_jsonl} (processed={processed}, seen={seen})")

# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("Value-Tinted Narration Generation")

    p.add_argument("--skeleton_jsonl", type=str, required=True)
    p.add_argument("--value_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--backend", choices=["hf", "openrouter"], default="hf")

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
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--retries", type=int, default=5)
    
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to process"
    )

    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (0-based)"
    )

    p.add_argument("--overwrite", action="store_true")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)

