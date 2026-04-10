# TemporalFraming.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

from llm_client import OpenRouterClient, HuggingFaceClient
from tqdm import tqdm


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


def write_jsonl_incremental(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


# ----------------------------
# Option utils
# ----------------------------
def stringify_option(it: Dict[str, Any]) -> str:
    text = it.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    situation_text = it.get("situation_text")
    if isinstance(situation_text, str) and situation_text.strip():
        return situation_text.strip()

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

    return ""


def extract_options(skel: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = safe_get(skel, ["bundle", "items"], [])
    if not isinstance(items, list):
        return []

    options = []
    for it in items:
        if not isinstance(it, dict):
            continue

        item_id = it.get("item_id")
        if item_id is None:
            continue

        text = stringify_option(it)
        if not text:
            continue

        source_field = None
        if isinstance(it.get("text"), str) and it.get("text", "").strip():
            source_field = "bundle.items.text"
        elif isinstance(it.get("situation_text"), str) and it.get("situation_text", "").strip():
            source_field = "bundle.items.situation_text"
        elif any(k in it for k in ["role_name", "obligation_level", "situation", "expectation"]):
            source_field = "bundle.items.role_struct"

        options.append({
            "option_id": str(item_id),
            "text": text,
            "source_field": source_field,
        })

    return options


# ----------------------------
# Base text extraction
# ----------------------------
def extract_base_text(skel: Dict[str, Any]) -> Tuple[str, str]:
    candidates = [
        (["bundle", "shared_story"], "bundle.shared_story"),  # RoleConflict
        (["base", "vignette"], "base.vignette"),              # GGB
        (["base", "scenario"], "base.scenario"),              # UniBench / SCOTUS / MedTriageAlignment
        (["scenario"], "scenario"),
    ]

    for keys, src in candidates:
        text = safe_get(skel, keys)
        if isinstance(text, str) and text.strip():
            return text.strip(), src

    items = safe_get(skel, ["bundle", "items"], [])
    if isinstance(items, list) and items:
        lines = []
        for it in items:
            txt = stringify_option(it)
            if txt:
                lines.append(txt)
        if lines:
            return " ".join(lines), "bundle.items[*]"

    raise ValueError(f"Cannot extract base narrative text from skeleton id={skel.get('id')}")


# ----------------------------
# Prompt (Temporal Framing)
# ----------------------------
BASE_PROMPT = """You are a controlled data generation assistant for decision-making benchmarks.

Your task is to rewrite a single decision option
by applying a specific temporal framing.

IMPORTANT CONSTRAINTS:
- Do NOT add new facts, entities, or details beyond the given scenario.
- Do NOT modify the original scenario text.
- Do NOT change the core decision or its fundamental consequences.
- Base all reasoning strictly on the provided information.

You may ONLY modify:
- temporal focus (short-term vs long-term consequences)
- time-related phrasing and wording

You must focus on either immediate/near-term outcomes OR future/long-term outcomes,
while keeping the consequence types logically consistent with the original option.

OUTPUT RULE:
- Output ONLY the rewritten option text.
- Do not include bullets, headings, or explanations.
"""


SHORT_TERM_FRAMING = """FRAMING TYPE: Temporal Focus – Short-term Consequences

Rewrite the decision option to emphasize immediate and near-term outcomes.

Guidelines:
- Focus on consequences that occur within days to weeks.
- Use temporal markers like "immediately", "right now", "this week", "in the coming days".
- Emphasize urgent, proximate effects.
- Highlight instant reactions, immediate costs/benefits, or near-term impacts.
- Do NOT introduce new types of consequences - only shift temporal focus.
- Keep the same stakeholders and consequence domains.
- Describe the same consequence as something that is unfolding right now or in the immediate future.
- Do not describe additional effects; make the timing feel closer instead.

The result should make consequences feel imminent and urgent,
while preserving the original decision's core meaning.
"""


LONG_TERM_FRAMING = """FRAMING TYPE: Temporal Focus – Long-term Consequences

Rewrite the decision option to emphasize future and long-term outcomes.

Guidelines:
- Focus on consequences that occur over months to years.
- Use temporal markers like "in the long run", "eventually", "over time", "in the future", "months/years from now".
- Emphasize cumulative, delayed, or sustained effects.
- Highlight future developments, long-term costs/benefits, or extended impacts.
- Do NOT introduce new types of consequences - only shift temporal focus.
- Keep the same stakeholders and consequence domains.
- Describe the same consequence as something that will matter over an extended period.
- Do not shift to broader, societal, or abstract future effects; keep the focus on the original consequence.

The result should make consequences feel distant but significant,
while preserving the original decision's core meaning.
"""


def build_temporal_prompt(option_text: str, temporal_focus: str) -> str:
    if temporal_focus == "short":
        framing = SHORT_TERM_FRAMING
    elif temporal_focus == "long":
        framing = LONG_TERM_FRAMING
    else:
        raise ValueError(f"Unknown temporal focus: {temporal_focus}")

    return (
        f"{BASE_PROMPT}\n"
        f"{framing}\n"
        f"Decision option text:\n"
        f"\"\"\"\n{option_text}\n\"\"\"\n"
        f"Rewrite the option accordingly."
    )


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


def get_base_id(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("id") or obj.get("instance_id") or obj.get("example_id") or "")


# ----------------------------
# Copy helpers for swapped
# ----------------------------
def build_option_map(options: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(o["option_id"]): normalize_text(o["text"]) for o in options}


def build_label_remap(
    original_options: List[Dict[str, Any]],
    current_options: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    original_label -> current_label
    """
    orig_map = build_option_map(original_options)
    curr_map = build_option_map(current_options)

    curr_text_to_label = {v: k for k, v in curr_map.items()}

    remap: Dict[str, str] = {}
    for orig_label, orig_text in orig_map.items():
        if orig_text in curr_text_to_label:
            remap[orig_label] = curr_text_to_label[orig_text]
    return remap


def copy_temporal_from_original(original_skel: Dict[str, Any], current_skel: Dict[str, Any]) -> None:
    original_ts = safe_get(original_skel, ["framings", "temporal_slice", "option_level"], {})
    if not isinstance(original_ts, dict) or not original_ts:
        return

    original_options = extract_options(original_skel)
    current_options = extract_options(current_skel)

    remap = build_label_remap(original_options, current_options)

    framings = ensure_framings(current_skel)
    framings.setdefault("temporal_slice", {})
    copied_option_level: Dict[str, Any] = {}

    for orig_option_id, payload in original_ts.items():
        if orig_option_id not in remap:
            continue
        new_option_id = remap[orig_option_id]
        if not isinstance(payload, dict):
            continue

        copied = dict(payload)
        copied["copied_from"] = "original"
        copied["original_option_id"] = orig_option_id
        copied_option_level[new_option_id] = copied

    framings["temporal_slice"]["option_level"] = copied_option_level


# ----------------------------
# Main run
# ----------------------------
def run(args):
    client = make_client(args)

    if os.path.exists(args.output_jsonl) and not args.resume:
        os.remove(args.output_jsonl)
        print(f"[INFO] Removed existing output file: {args.output_jsonl}")

    processed_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        for row in iter_jsonl(args.output_jsonl):
            key = get_instance_key(row)
            if key:
                processed_ids.add(key)
        print(f"[RESUME] Found {len(processed_ids)} already processed examples")

    original_cache: Dict[str, Dict[str, Any]] = {}

    n_seen = 0
    n_processed = 0
    n_updated = 0
    n_skipped = 0
    n_passthrough = 0

    for skel in tqdm(iter_jsonl(args.skeleton_jsonl), desc="examples"):
        current_idx = n_seen
        n_seen += 1

        if current_idx < args.start_idx:
            continue

        if args.limit is not None and n_processed >= args.limit:
            break

        skel_id = get_instance_key(skel)
        base_id = get_base_id(skel)
        option_order = skel.get("option_order", "original")

        if args.resume and skel_id in processed_ids:
            n_skipped += 1
            continue

        try:
            options = extract_options(skel)

            if not options:
                write_jsonl_incremental(args.output_jsonl, skel)
                n_passthrough += 1
                n_processed += 1
                continue

            framings = ensure_framings(skel)
            framings.setdefault("temporal_slice", {})

            # swapped: do not generate, copy from original
            if option_order == "swapped":
                original_skel = original_cache.get(base_id)
                if original_skel is None:
                    print(f"[WARN] swapped row encountered before original for {base_id}")
                else:
                    copy_temporal_from_original(original_skel, skel)

                write_jsonl_incremental(args.output_jsonl, skel)
                n_updated += 1
                n_processed += 1

                if n_processed % 10 == 0:
                    print(
                        f"[PROGRESS] seen={n_seen}, processed={n_processed}, "
                        f"updated={n_updated}, skipped={n_skipped}, passthrough={n_passthrough}"
                    )
                continue

            # original: generate
            _base_text, _base_src = extract_base_text(skel)  # validation only

            option_framings = {}

            for opt in options:
                opt_id = opt["option_id"]
                opt_text = opt["text"]

                short_prompt = build_temporal_prompt(opt_text, "short")
                long_prompt = build_temporal_prompt(opt_text, "long")
                
                print(short_prompt)
                print("------------")
                print(long_prompt)
                print("=============")

                short_text = generate_one(client, short_prompt, args).strip()
                long_text = generate_one(client, long_prompt, args).strip()

                option_framings[opt_id] = {
                    "source_text": opt_text,
                    "source_field": opt.get("source_field"),
                    "short_term": short_text,
                    "long_term": long_text,
                }

            framings["temporal_slice"]["option_level"] = option_framings

            original_cache[base_id] = json.loads(json.dumps(skel, ensure_ascii=False))

            write_jsonl_incremental(args.output_jsonl, skel)
            n_updated += 1
            n_processed += 1

            if n_processed % 10 == 0:
                print(
                    f"[PROGRESS] seen={n_seen}, processed={n_processed}, "
                    f"updated={n_updated}, skipped={n_skipped}, passthrough={n_passthrough}"
                )

        except Exception as e:
            print(f"[ERROR] Failed to process {skel_id}: {e}")
            write_jsonl_incremental(args.output_jsonl, skel)
            n_passthrough += 1
            n_processed += 1

    print(f"\n[DONE] Wrote: {args.output_jsonl}")
    print(
        f"Seen: {n_seen}, Processed: {n_processed}, "
        f"Updated: {n_updated}, Skipped: {n_skipped}, Passthrough: {n_passthrough}"
    )


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("Temporal Framing Generation")

    p.add_argument("--skeleton_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--backend", type=str, choices=["hf", "openrouter"], default="hf")

    # HF
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--hf_device", type=str, default="auto")
    p.add_argument("--hf_dtype", type=str, default="auto")

    # OpenRouter
    p.add_argument("--or_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--or_model", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--or_site", type=str, default="")
    p.add_argument("--or_app", type=str, default="")

    # sampling
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=700)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--retries", type=int, default=5)

    # resume support
    p.add_argument("--resume", action="store_true", help="Resume from existing output file")

    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (0-based)"
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)