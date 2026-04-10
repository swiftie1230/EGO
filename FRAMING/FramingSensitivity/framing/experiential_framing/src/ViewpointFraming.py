# ViewpointFraming.py
from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict, List

from scenario_extract import extract_scenario_text
from option_extract import extract_options

from llm_client import OpenRouterClient, HuggingFaceClient

BASE_PROMPT = """You are a controlled data generation assistant for decision-making benchmarks.

Your task is to rewrite a single decision option
by changing only the observer viewpoint.

IMPORTANT CONSTRAINTS:
- Do NOT change the meaning of the decision option.
- Do NOT add new outcomes, consequences, or justifications.
- Do NOT introduce new entities, risks, or facts.
- Do NOT imply moral evaluation or preference.
- Preserve the original decision content exactly.

You may ONLY modify:
- grammatical viewpoint (first / second / third person)
- sentence structure required for viewpoint shift

OUTPUT RULE:
- Output ONLY the rewritten option text.
- Do not include bullets, headings, or explanations.
"""


THIRD_PERSON = """FRAMING TYPE: Observer Viewpoint – Third Person (Option Level)

Rewrite the decision option from a detached third-person perspective.

Guidelines:
- Do NOT use "you".
- Refer to the actor impersonally (e.g., "the individual", "the decision-maker").
- Do NOT introduce narration outside the option itself.
- Avoid internal thoughts or emotions.

Only change grammatical viewpoint; preserve semantics exactly.
"""


SECOND_PERSON = """FRAMING TYPE: Observer Viewpoint – Second Person (Option Level)

Rewrite the decision option from a second-person perspective.

Guidelines:
- Use "you" as the grammatical subject.
- Do NOT add moral pressure or obligation language.
- Do NOT use phrases like "you should", "you must", or "you feel".
- Do NOT introduce new actions or outcomes.

Only shift grammatical viewpoint, not moral framing.
"""


def build_prompt(option_text: str, perspective: str) -> str:
    if perspective == "third":
        framing = THIRD_PERSON
    elif perspective == "second":
        framing = SECOND_PERSON
    else:
        raise ValueError(f"Unknown perspective: {perspective}")

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
    return set([x.strip() for x in s.split(",") if x.strip()])


def allow_example(example: dict, domain_filter: set[str] | None) -> bool:
    if domain_filter is None:
        return True
    dom = example.get("domain", None)
    return (dom in domain_filter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_tag", type=str, default=None)

    # backend switch
    ap.add_argument("--backend", choices=["openrouter", "hf"], default="openrouter")

    # OpenRouter args
    ap.add_argument("--model", required=False, default="", help="OpenRouter model id (e.g., openai/gpt-4o-mini)")

    # HF args
    ap.add_argument("--hf_model_id", type=str, default=None, help="HF model id (e.g., Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--hf_device", type=str, default="auto", help='auto|cuda:0|cpu|mps')
    ap.add_argument("--hf_dtype", type=str, default="auto", help='auto|float16|bfloat16|float32')

    # shared gen params (mapped into client.generate)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--retries", type=int, default=4)

    ap.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Comma-separated domains to include, e.g., life_safety,moral_dilemma,decision_choice"
    )

    args = ap.parse_args()

    # Build client
    if args.backend == "openrouter":
        if not args.model:
            raise ValueError("--model is required when --backend=openrouter")
        client = OpenRouterClient()
        model_name_for_metadata = args.model
    else:
        if not args.hf_model_id:
            raise ValueError("--hf_model_id is required when --backend=hf")
        client = HuggingFaceClient(
            model_id=args.hf_model_id,
            device=args.hf_device,
            dtype=args.hf_dtype,
        )
        model_name_for_metadata = args.hf_model_id

    in_files = iter_json_files(args.input_dir)
    ensure_dir(args.output_dir)

    domain_filter = parse_domain_filter(args.domain_filter)

    for path in in_files:
        # mirror directory structure under output_dir + append model_tag to filename
        rel = os.path.relpath(path, args.input_dir)
        rel_dir = os.path.dirname(rel)
        base = os.path.basename(rel)

        name, ext = os.path.splitext(base)
        if ext != ".jsonl":
            continue

        if args.model_tag:
            name = f"{name}__{args.model_tag}"

        out_path = os.path.join(args.output_dir, rel_dir, name + ext)
        ensure_dir(os.path.dirname(out_path))

        n_total = 0
        n_written = 0
        n_skipped = 0

        with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                n_total += 1
                ex: Dict[str, Any] = json.loads(line)

                if not allow_example(ex, domain_filter):
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_skipped += 1
                    continue

                #scenario_text, scenario_src = extract_scenario_text(ex)
                options = extract_options(ex)

                ex.setdefault("framings", {})
                ex["framings"].setdefault("observer_viewpoint", {})
                option_viewpoints = {}

                for opt in options:
                    opt_id = opt["option_id"]
                    opt_text = opt["text"]

                    third_prompt = build_prompt(opt_text, "third")
                    second_prompt = build_prompt(opt_text, "second")

                    third_text = client.generate(
                        third_prompt,
                        model=args.model if args.backend == "openrouter" else None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        retries=args.retries,
                    )

                    second_text = client.generate(
                        second_prompt,
                        model=args.model if args.backend == "openrouter" else None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        retries=args.retries,
                    )

                    option_viewpoints[opt_id] = {
                        "source_text": opt_text,
                        "third_person": third_text,
                        "second_person": second_text,
                    }

                ex["framings"]["observer_viewpoint"]["option_level"] = option_viewpoints
                
                # ✅ WRITE OUTPUT (missing before)
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1


        print(f"[viewpoint] {os.path.basename(path)} -> {out_path} (total={n_total}, updated={n_written}, passthrough={n_skipped})")



if __name__ == "__main__":
    main()
