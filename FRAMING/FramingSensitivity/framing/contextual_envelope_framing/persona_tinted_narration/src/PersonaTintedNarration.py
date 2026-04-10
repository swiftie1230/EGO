# PersonaTintedNarration.py
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

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
# Base text extraction from skeleton
# ----------------------------
def extract_base_text(skel: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (base_text, source_field)
    """
    # RoleConflictBench-style
    shared_story = safe_get(skel, ["bundle", "shared_story"])
    if isinstance(shared_story, str) and shared_story.strip():
        return shared_story.strip(), "bundle.shared_story"

    # GGB-style
    ggb_vignette = safe_get(skel, ["base", "vignette"])
    if isinstance(ggb_vignette, str) and ggb_vignette.strip():
        return ggb_vignette.strip(), "base.vignette"

    # UniBench-style
    unibench_scenario = safe_get(skel, ["base", "scenario"])
    if isinstance(unibench_scenario, str) and unibench_scenario.strip():
        return unibench_scenario.strip(), "base.scenario"

    # TRIAGE-style (no narrative; compose minimal neutral scene from situation_text)
    items = safe_get(skel, ["bundle", "items"], [])
    if isinstance(items, list) and items:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            item_id = it.get("item_id", "unknown")
            st = it.get("situation_text")
            if isinstance(st, str) and st.strip():
                lines.append(f"{item_id}: {st.strip()}")
        if lines:
            # Keep it as a neutral report-like narrative seed
            return " ".join(lines), "bundle.items[*].situation_text"

    raise ValueError(f"Cannot extract base narrative text from skeleton id={skel.get('id') or skel.get('instance_id')}")


# ----------------------------
# Persona parsing (robust to multiple schemas)
# ----------------------------
def flatten_personas(persona_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports formats like:
    - TRIAGE: patient_1_personas / patient_2_personas
    - UniBench/GGB: A_personas / B_personas
    - Or generic: personas: [...]
    """
    personas: List[Dict[str, Any]] = []

    # 1) Generic list
    if isinstance(persona_obj.get("personas"), list):
        for p in persona_obj["personas"]:
            if isinstance(p, dict):
                personas.append(p)

    # 2) Keyed by option
    for k, v in persona_obj.items():
        if not k.endswith("_personas"):
            continue
        if not isinstance(v, list):
            continue
        for p in v:
            if isinstance(p, dict):
                # attach implied supports_option if not present
                # e.g., "patient_1_personas" -> supports_option="patient_1"
                if "supports_option" not in p:
                    implied = k.replace("_personas", "")
                    p = dict(p)
                    p["supports_option"] = implied
                personas.append(p)

    # Deduplicate by persona_id if repeated
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for p in personas:
        pid = p.get("persona_id")
        key = pid if isinstance(pid, str) else json.dumps(p, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return uniq

def extract_option_personas(persona_obj: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
    {
        "A": [persona, persona, ...],
        "B": [...],
        "patient_1": [...],
        ...
    }
    """
    option_map: Dict[str, List[Dict[str, Any]]] = {}

    for k, v in persona_obj.items():
        if not k.endswith("_personas"):
            continue
        if not isinstance(v, list):
            continue

        option_id = k.replace("_personas", "")
        option_map[option_id] = v

    return option_map



def get_instance_key(obj: Dict[str, Any]) -> str:
    # persona file sometimes uses instance_id, sometimes id
    return str(obj.get("id") or obj.get("instance_id") or obj.get("example_id"))


def index_persona_file(persona_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(persona_jsonl_path):
        key = get_instance_key(row)
        if key:
            idx[key] = row
    return idx


# ----------------------------
# Prompt (Persona-Tinted Narration)
# ----------------------------
BASE_PROMPT = """You are a controlled narrative generation assistant.

Your task is to rewrite the given scenario so that it reflects the perceptual viewpoint of a specific persona.

IMPORTANT CONSTRAINTS:
- Do NOT add new facts, risks, outcomes, entities, or details.
- Do NOT remove or alter any existing factual information.
- Do NOT change the decision structure, decision question, or available options.
- Do NOT provide advice, justification, or moral judgment.
- Do NOT state which option should be chosen.
- Preserve the original causal structure and the same set of referenced facts.

You may ONLY modify:
- narrative focus and salience
- sentence structure and narrative style
- emotional and cognitive framing (what feels urgent/salient)
- experiential distance (without adding details)

Do not explicitly mention the persona profile.
Do not list the persona fields.
Do not output headings, bullets, or explanations.

OUTPUT RULE:
- Output ONLY the rewritten scenario text.
"""


def build_persona_tinted_prompt(
    base_text: str,
    decision_question: str,
    options: List[Dict[str, Any]],
    persona: Dict[str, Any],
) -> str:
    persona_label = persona.get("persona_label", "")
    core_prior = persona.get("core_prior", "")
    decision_heuristic = persona.get("decision_heuristic", "")
    attention_focus = persona.get("attention_focus", [])
    value_priority = persona.get("value_priority", [])
    supports_option = persona.get("supports_option", "")

    # options text (for structure lock; do NOT ask model to pick)
    opt_lines = []
    for it in options:
        item_id = it.get("item_id")
        text = it.get("text") or it.get("situation_text") or ""
        if item_id is None:
            continue
        opt_lines.append(f"{item_id}: {str(text).strip()}")
    opt_block = "\n".join(opt_lines)

    af = ", ".join(attention_focus) if isinstance(attention_focus, list) else str(attention_focus)
    vp = ", ".join(value_priority) if isinstance(value_priority, list) else str(value_priority)

    prompt = f"""{BASE_PROMPT}

SCENARIO (rewrite this, facts unchanged):
{base_text}

DECISION QUESTION (do not change):
{decision_question}

OPTIONS (do not change):
{opt_block}

PERSONA LENS (implicit; do not mention explicitly):
- Persona label: {persona_label}
- Core prior: {core_prior}
- Decision heuristic: {decision_heuristic}
- Attention focus: {af}
- Value priorities: {vp}
- (This persona tends to support option: {supports_option})

INSTRUCTION:
Rewrite the scenario so that the persona lens subtly shapes:
- what details are foregrounded vs backgrounded,
- what feels urgent or morally salient,
- the perceived stakes and attention allocation.

Again: do NOT recommend any option, and do NOT add any new facts.
OUTPUT ONLY the rewritten scenario text.
"""
    return prompt


# ----------------------------
# Generation driver
# ----------------------------
def make_client(args):
    if args.backend == "openrouter":
        if not args.or_api_key:
            raise ValueError("--or_api_key is required for openrouter backend")
        return OpenRouterClient(
            api_key=args.or_api_key,
            site=args.or_site,
            app=args.or_app,
            retries=args.retries,
        )
    elif args.backend == "hf":
        return HuggingFaceClient(
            model_id=args.model_id,
            device=args.hf_device,
            dtype=args.hf_dtype,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


def generate_one(client, prompt: str, args) -> str:
    # Match your existing usage pattern: for openrouter pass model name; for hf not
    if args.backend == "openrouter":
        return client.generate(
            prompt,
            model=args.or_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            retries=args.retries,
        )
    else:
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


def run(args):
    persona_index = index_persona_file(args.persona_jsonl)
    client = make_client(args)

    out_rows: List[Dict[str, Any]] = []
    n_total = 0
    n_skipped_no_persona = 0
    n_done = 0

    for skel in iter_jsonl(args.skeleton_jsonl):
        n_total += 1
        skel_id = str(skel.get("id") or skel.get("instance_id"))

        if skel_id not in persona_index:
            n_skipped_no_persona += 1
            out_rows.append(skel)
            continue

        persona_obj = persona_index[skel_id]
        option_to_personas = extract_option_personas(persona_obj)

        if not option_to_personas:
            n_skipped_no_persona += 1
            out_rows.append(skel)
            continue

        base_text, base_src = extract_base_text(skel)
        decision_question = skel.get("decision_question", "")
        options = safe_get(skel, ["bundle", "items"], [])
        if not isinstance(options, list):
            options = []

        framings = ensure_framings(skel)
        pt_root = framings.setdefault("persona_tinted_narration", {})

        for option_id, personas in option_to_personas.items():
            if not personas:
                continue

            option_field = pt_root.setdefault(option_id, {})

            for p in personas:
                pid = p.get("persona_id") or p.get("persona_label") or "persona"
                pid = str(pid)

                if (
                    not args.overwrite
                    and pid in option_field
                    and isinstance(option_field[pid], dict)
                    and option_field[pid].get("text")
                ):
                    continue

                prompt = build_persona_tinted_prompt(
                    base_text=base_text,
                    decision_question=decision_question,
                    options=options,
                    persona=p,
                )

                text = generate_one(client, prompt, args).strip()

                option_field[pid] = {
                    "text": text,
                    "source_field": base_src,
                    "supports_option": option_id,
                    "persona_label": p.get("persona_label"),
                    "core_prior": p.get("core_prior"),
                    "decision_heuristic": p.get("decision_heuristic"),
                    "attention_focus": p.get("attention_focus"),
                    "value_priority": p.get("value_priority"),
                }

                n_done += 1

        out_rows.append(skel)

    write_jsonl(args.output_jsonl, out_rows)

    print(f"[DONE] wrote: {args.output_jsonl}")
    print(f"  total_instances={n_total}")
    print(f"  instances_missing_persona={n_skipped_no_persona}")
    print(f"  persona_narrations_generated={n_done}")


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("Persona-Tinted Narration Generation")

    p.add_argument("--skeleton_jsonl", type=str, required=True)
    p.add_argument("--persona_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--backend", type=str, choices=["hf", "openrouter"], default="hf")

    # HF
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--hf_device", type=str, default="auto")
    p.add_argument("--hf_dtype", type=str, default="auto")

    # OpenRouter
    p.add_argument("--or_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--or_model", type=str, default="openai/gpt-4.1-mini")
    p.add_argument("--or_site", type=str, default="")   # optional metadata
    p.add_argument("--or_app", type=str, default="")    # optional metadata

    # sampling
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--retries", type=int, default=5)

    p.add_argument("--overwrite", action="store_true")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
