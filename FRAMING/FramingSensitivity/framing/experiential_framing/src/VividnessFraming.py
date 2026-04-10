# VividnessFraming.py
from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict, List


from llm_client import OpenRouterClient, HuggingFaceClient


BASE_PROMPT = """You are a controlled data generation assistant for decision-making benchmarks.

Your task is to rewrite ONE decision option using a specified narrative framing style.

CRITICAL REQUIREMENT:
The rewritten text MUST preserve the exact meaning of the original decision
while making the narrative style clearly different.

IMPORTANT CONSTRAINTS:
- Do NOT add new facts, risks, outcomes, entities, or details beyond the given scenario.
- Do NOT modify the scenario text.
- Do NOT introduce hypothetical events or imagined consequences.
- Do NOT introduce new causal outcomes or reasoning.

You may ONLY modify:
- narrative style
- sentence structure
- level of vividness or abstraction
- grammatical perspective (e.g., action vs description)

OUTPUT RULE:
- Output ONLY the rewritten option text.
- Do not include explanations.
"""


LOW_VIVIDNESS = """FRAMING TYPE: Narrative Distance – LOW VIVIDNESS

Rewrite the decision option in an abstract, policy-like form.

Style requirements:
- Use detached, analytical language.
- Avoid any sense of action occurring in time.
- Avoid verbs that describe physical actions (e.g., go, move, take, help).
- Prefer static, declarative phrasing.
- Avoid sensory wording, imagery, immediacy, or scene-like phrasing.
- Rewrite the decision option as a neutral description of the choice. Avoid turning it into a philosophical principle or policy statement.

Semantic preservation rule:
- Preserve the core entities and actions from the original option.
- Do not replace key elements with generic abstractions.
- For example, if the original mentions "child" or "sacrifice", these terms should remain identifiable.

Prohibited:
- expanding or justifying the decision.

The result should resemble a brief description of a choice,
rather than a lived or observed event.
"""


HIGH_VIVIDNESS = """FRAMING TYPE: Narrative Distance – HIGH VIVIDNESS

Rewrite the decision option as an immediate action scene.

Style requirements:
- Use dynamic, action-oriented verbs.
- Write as if the action is happening right now.
- Prefer active verbs over decision language.
- Emphasize immediacy, movement, or physical engagement through wording.
- Avoid abstract decision language such as "choose", "decide", or "the option is".
- Do not introduce specific methods or mechanisms that were not implied in the original text. For example, do not specify how the sacrifice occurs if the original text does not describe it.

Allowed stylistic elements:
- moment-focused phrasing
- concrete action wording
- present or present-progressive structure

Semantic preservation rule:
- Preserve the core entities and actions from the original option.
- Do not replace key elements with generic abstractions.
- For example, if the original mentions "child" or "sacrifice", these terms should remain identifiable.

Prohibited:
- adding new events
- adding new consequences
- adding reasoning or persuasion

The result should feel like witnessing the action being carried out, like a lived action
rather than an abstract decision statement.
"""


def build_option_prompt(option_text: str, level: str) -> str:
    if level == "low":
        framing = LOW_VIVIDNESS
    elif level == "high":
        framing = HIGH_VIVIDNESS
    else:
        raise ValueError(f"Unknown vividness level: {level}")

    return (
        f"{BASE_PROMPT}\n"
        f"{framing}\n"
        f"Decision option text:\n"
        f"\"\"\"\n{option_text}\n\"\"\"\n"
        f"Rewrite the option accordingly."
    )


def iter_json_files(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.endswith(".json") or fn.endswith(".jsonl"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_domain_filter(s: str | None):
    if s is None or not str(s).strip():
        return None
    return set(x.strip() for x in s.split(",") if x.strip())


def allow_example(example: dict, domain_filter: set[str] | None) -> bool:
    if domain_filter is None:
        return True
    dom = example.get("domain", None)
    return dom in domain_filter


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def stringify_option(it: Dict[str, Any]) -> str:
    # Standard option text
    text = it.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Patient / triage style
    situation_text = it.get("situation_text")
    if isinstance(situation_text, str) and situation_text.strip():
        return situation_text.strip()

    # RoleConflict style
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


def extract_options(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = (((example.get("bundle") or {}).get("items")) or [])
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue

        option_id = it.get("item_id")
        if option_id is None:
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

        out.append({
            "option_id": str(option_id),
            "text": text,
            "source_field": source_field,
        })

    return out


def validate_options(options: List[Dict[str, Any]]) -> bool:
    if not isinstance(options, list) or len(options) == 0:
        return False

    for opt in options:
        if not isinstance(opt, dict):
            return False
        if "option_id" not in opt:
            return False
        if "text" not in opt:
            return False
        if not isinstance(opt["text"], str) or not opt["text"].strip():
            return False

    return True


def get_base_id(ex: Dict[str, Any]) -> str:
    return str(ex.get("id") or ex.get("instance_id") or ex.get("example_id") or "")


def build_option_map(options: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(o["option_id"]): normalize_text(o["text"]) for o in options}


def build_label_remap(original_options: List[Dict[str, Any]], current_options: List[Dict[str, Any]]) -> Dict[str, str]:
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


def copy_narrative_distance_from_original(original_ex: Dict[str, Any], current_ex: Dict[str, Any]) -> None:
    original_nd = ((((original_ex.get("framings") or {}).get("narrative_distance")) or {}).get("option_level")) or {}
    if not isinstance(original_nd, dict) or not original_nd:
        return

    original_options = extract_options(original_ex)
    current_options = extract_options(current_ex)

    remap = build_label_remap(original_options, current_options)

    current_ex.setdefault("framings", {})
    if not isinstance(current_ex["framings"], dict):
        current_ex["framings"] = {}
    current_ex["framings"].setdefault("narrative_distance", {})

    copied_option_level: Dict[str, Any] = {}

    for orig_option_id, payload in original_nd.items():
        if orig_option_id not in remap:
            continue

        new_option_id = remap[orig_option_id]
        if not isinstance(payload, dict):
            continue

        copied = dict(payload)
        copied["copied_from"] = "original"
        copied["original_option_id"] = orig_option_id
        copied_option_level[new_option_id] = copied

    current_ex["framings"]["narrative_distance"]["option_level"] = copied_option_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_tag", type=str, default=None)

    # backend
    ap.add_argument("--backend", choices=["openrouter", "hf"], default="openrouter")

    # OpenRouter
    ap.add_argument("--model", default="", help="OpenRouter model id")

    # HuggingFace
    ap.add_argument("--hf_model_id", type=str, default=None)
    ap.add_argument("--hf_device", type=str, default="auto")
    ap.add_argument("--hf_dtype", type=str, default="auto")

    # generation
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--retries", type=int, default=4)

    ap.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Comma-separated domains"
    )

    ap.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (0-based) within each file"
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to process per file"
    )

    args = ap.parse_args()

    if args.backend == "openrouter":
        if not args.model:
            raise ValueError("--model required for openrouter backend")
        client = OpenRouterClient()
    else:
        if not args.hf_model_id:
            raise ValueError("--hf_model_id required for hf backend")
        client = HuggingFaceClient(
            model_id=args.hf_model_id,
            device=args.hf_device,
            dtype=args.hf_dtype,
        )

    in_files = iter_json_files(args.input_dir)
    ensure_dir(args.output_dir)

    domain_filter = parse_domain_filter(args.domain_filter)

    for path in in_files:
        rel = os.path.relpath(path, args.input_dir)
        rel_dir = os.path.dirname(rel)
        base = os.path.basename(rel)

        name, ext = os.path.splitext(base)
        if ext != ".jsonl":
            continue

        if args.model_tag:
            name = f"{name}__{args.model_tag}"
        if args.start_idx != 0:
            name = f"{name}__start{args.start_idx}"
        if args.limit is not None:
            name = f"{name}__limit{args.limit}"

        out_path = os.path.join(args.output_dir, rel_dir, name + ext)
        ensure_dir(os.path.dirname(out_path))

        n_total = 0
        n_updated = 0
        n_passthrough = 0
        seen_valid = 0
        processed = 0

        original_cache: Dict[str, Dict[str, Any]] = {}

        with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for raw_line_idx, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                try:
                    ex: Dict[str, Any] = json.loads(line)
                except Exception:
                    print(f"[WARN] invalid JSON at raw line {raw_line_idx} in {path}")
                    continue

                current_idx = seen_valid
                seen_valid += 1

                if current_idx < args.start_idx:
                    continue

                if args.limit is not None and processed >= args.limit:
                    break

                n_total += 1

                if not allow_example(ex, domain_filter):
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_passthrough += 1
                    processed += 1
                    continue

                base_id = get_base_id(ex)
                option_order = ex.get("option_order", "original")

                try:
                    options = extract_options(ex)
                except Exception as e:
                    print(f"[WARN] extract_options failed for id={ex.get('id')} idx={current_idx}: {e}")
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_passthrough += 1
                    processed += 1
                    continue

                if not validate_options(options):
                    print(f"[WARN] invalid options for id={ex.get('id')} idx={current_idx}")
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_passthrough += 1
                    processed += 1
                    continue

                ex.setdefault("framings", {})
                if not isinstance(ex["framings"], dict):
                    ex["framings"] = {}
                ex["framings"].setdefault("narrative_distance", {})

                # swapped: copy only
                if option_order == "swapped":
                    original_ex = original_cache.get(base_id)
                    if original_ex is None:
                        print(f"[WARN] swapped row encountered before original for id={base_id}")
                    else:
                        copy_narrative_distance_from_original(original_ex, ex)

                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_updated += 1
                    processed += 1

                    if processed % 10 == 0:
                        print(f"[progress] {os.path.basename(path)} processed={processed}")
                    continue

                # original: generate
                print(ex.get("id"), [o["option_id"] for o in options])

                option_framings = {}

                for opt in options:
                    opt_id = opt["option_id"]
                    opt_text = opt["text"]

                    low_prompt = build_option_prompt(opt_text, "low")
                    high_prompt = build_option_prompt(opt_text, "high")
                    
                    print(low_prompt)
                    print("-----------")
                    print(high_prompt)
                    print("=========================")

                    try:
                        low_text = client.generate(
                            low_prompt,
                            model=args.model if args.backend == "openrouter" else None,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            top_p=args.top_p,
                            retries=args.retries,
                        )
                    except Exception as e:
                        print(f"[WARN] low vividness generation failed for id={ex.get('id')} opt={opt_id}: {e}")
                        low_text = None

                    try:
                        high_text = client.generate(
                            high_prompt,
                            model=args.model if args.backend == "openrouter" else None,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            top_p=args.top_p,
                            retries=args.retries,
                        )
                    except Exception as e:
                        print(f"[WARN] high vividness generation failed for id={ex.get('id')} opt={opt_id}: {e}")
                        high_text = None

                    option_framings[opt_id] = {
                        "source_text": opt_text,
                        "source_field": opt.get("source_field"),
                        "low_vividness": low_text,
                        "high_vividness": high_text,
                    }

                ex["framings"]["narrative_distance"]["option_level"] = option_framings
                original_cache[base_id] = json.loads(json.dumps(ex, ensure_ascii=False))

                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_updated += 1
                processed += 1

                if processed % 10 == 0:
                    print(f"[progress] {os.path.basename(path)} processed={processed}")

        print(
            f"[vividness] {os.path.basename(path)} -> {out_path} "
            f"(total={n_total}, updated={n_updated}, passthrough={n_passthrough}, seen_valid={seen_valid}, processed={processed})"
        )


if __name__ == "__main__":
    main()
