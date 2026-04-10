#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# =========================================================
# Basic utils
# =========================================================
def clean(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace('"""', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_json_load(text: str) -> Dict[str, Any]:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def clamp_1_5(x: Any, default: int = 1) -> int:
    try:
        x = int(x)
        return max(1, min(5, x))
    except Exception:
        return default


def get_item_text_map(row: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    items = row.get("bundle", {}).get("items", [])
    for item in items:
        item_id = clean(item.get("item_id"))
        text = clean(item.get("text"))
        if item_id:
            out[item_id] = text
    return out


def get_base_vignette(row: Dict[str, Any]) -> str:
    return clean(row.get("base", {}).get("vignette", ""))


# =========================================================
# OpenRouter
# =========================================================
def call_openrouter(
    prompt: str,
    model: str = "openai/gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
    max_retries: int = 3,
) -> Dict[str, Any]:
    api_key = "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"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "Framing Benchmark LLM Judge",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()

            text = data["choices"][0]["message"]["content"]
            if not text or not text.strip():
                raise ValueError("Empty LLM response")

            return safe_json_load(text)

        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise last_err


# =========================================================
# Prompt
# =========================================================
def build_judge_prompt(row: Dict[str, Any]) -> str:
    framing_type = clean(row["framing_type"])
    variant_label = clean(row.get("variant_label", ""))
    base_text = clean(row["base_text"])
    framed_text = clean(row["framed_text"])
    source_text = clean(row.get("source_text", ""))
    option_id = clean(row.get("option_id", ""))
    option_order = clean(row.get("option_order", ""))

    framing_specific_guide = ""

    if framing_type == "value_tinted":
        framing_specific_guide = """
Framing type details:
- This is VALUE-TINTED NARRATION.
- The rewrite should make a value lens more salient, but should not change the core situation or introduce unsupported facts.
- Framing Salience should reflect how clearly the value emphasis is perceptible.
- Framing Purity should reflect whether the rewrite becomes overt persuasion for that option rather than a framing lens.
"""
    elif framing_type == "experiential_vividness":
        framing_specific_guide = f"""
Framing type details:
- This is EXPERIENTIAL FRAMING (VIVIDNESS): {variant_label}.
- The rewrite should differ mainly in vividness / concreteness / imagery.
- It should preserve the same decision meaning as the source option text.
- Framing Salience should reflect how clearly the {variant_label} style is perceptible.
- Framing Purity should reflect whether the text is merely more/less vivid versus becoming a persuasive push.
"""
    elif framing_type == "temporal":
        framing_specific_guide = f"""
Framing type details:
- This is TEMPORAL FRAMING: {variant_label}.
- The rewrite should shift temporal focus (e.g., short-term vs long-term), while preserving the underlying option meaning.
- Framing Salience should reflect how clearly the temporal emphasis is perceptible.
- Framing Purity should reflect whether the rewrite only shifts time focus versus injecting new argumentative content.
"""

    return f"""
You are evaluating a framing transformation for a benchmark quality check.

Evaluate the FRAMED text against the ORIGINAL text.

Metadata:
- framing_type: {framing_type}
- variant_label: {variant_label}
- option_id: {option_id}
- option_order: {option_order}

Use the following scoring rubric.
All scores must be integers from 1 to 5.

------------------------------------------------
1. Structural Integrity
Does the framed text preserve the original meaning and objective content without introducing unsupported new facts or removing key facts?

5 = Meaning/content preserved almost perfectly
4 = Minor wording shifts, but meaning preserved
3 = Noticeable emphasis change or slight semantic drift
2 = Important semantic distortion or added implications
1 = Major distortion, new facts, or removed key content

------------------------------------------------
2. Framing Salience
How clearly is the intended framing transformation perceptible?

5 = Immediately and strongly clear
4 = Clearly noticeable
3 = Moderately noticeable
2 = Weak
1 = Barely noticeable

------------------------------------------------
3. Framing Purity
Does the framed text implement the framing cleanly without excessive extra changes, overt persuasion, or unnecessary new consequences?

5 = Clean framing only
4 = Slight extra directional tone
3 = Moderate extra bias or added interpretation
2 = Strong bias / extra consequences / argumentative push
1 = Explicit persuasion or heavily contaminated rewrite

------------------------------------------------
{framing_specific_guide}

Return JSON only:

{{
  "structural_integrity": 1,
  "framing_salience": 1,
  "framing_purity": 1,
  "reason": "short explanation"
}}

ORIGINAL:
{base_text}

SOURCE_TEXT:
{source_text}

FRAMED:
{framed_text}
"""


def normalize_judge_output(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "structural_integrity": clamp_1_5(out.get("structural_integrity", 1)),
        "framing_salience": clamp_1_5(out.get("framing_salience", 1)),
        "framing_purity": clamp_1_5(out.get("framing_purity", 1)),
        "reason": clean(out.get("reason", "")),
    }


# =========================================================
# Unit builders
# =========================================================
def build_value_tinted_units(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    units = []
    vt = row.get("framings", {}).get("value_tinted_narration", {})
    base_vignette = get_base_vignette(row)
    dataset = clean(row.get("dataset"))
    domain = clean(row.get("domain"))
    rid = clean(row.get("id"))
    option_order = clean(row.get("option_order"))

    for option_id, value_dict in vt.items():
        if not isinstance(value_dict, dict):
            continue

        for value_name, payload in value_dict.items():
            if not isinstance(payload, dict):
                continue

            framed_text = clean(payload.get("text"))
            if not framed_text:
                continue

            units.append({
                "id": rid,
                "dataset": dataset,
                "domain": domain,
                "option_order": option_order,
                "framing_type": "value_tinted",
                "variant_label": clean(value_name),
                "option_id": clean(option_id),
                "base_text": base_vignette,
                "source_text": base_vignette,
                "source_field": clean(payload.get("source_field", "base.vignette")),
                "framed_text": framed_text,
                "supports_option": clean(payload.get("supports_option")),
                "instantiated_value": clean(payload.get("instantiated_value")),
                "decision_principle": clean(payload.get("decision_principle")),
                "attention_focus": payload.get("attention_focus", []),
            })

    return units


def build_experiential_units(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    units = []
    nd = row.get("framings", {}).get("narrative_distance", {}).get("option_level", {})
    dataset = clean(row.get("dataset"))
    domain = clean(row.get("domain"))
    rid = clean(row.get("id"))
    option_order = clean(row.get("option_order"))
    item_text_map = get_item_text_map(row)

    for option_id, payload in nd.items():
        if not isinstance(payload, dict):
            continue

        source_text = clean(payload.get("source_text")) or clean(item_text_map.get(option_id, ""))
        source_field = clean(payload.get("source_field", "bundle.items.text"))

        low_v = clean(payload.get("low_vividness"))
        high_v = clean(payload.get("high_vividness"))

        common_meta = {
            "id": rid,
            "dataset": dataset,
            "domain": domain,
            "option_order": option_order,
            "framing_type": "experiential_vividness",
            "option_id": clean(option_id),
            "base_text": source_text,
            "source_text": source_text,
            "source_field": source_field,
            "copied_from": clean(payload.get("copied_from")),
            "original_option_id": clean(payload.get("original_option_id")),
        }

        if low_v:
            units.append({
                **common_meta,
                "variant_label": "low_vividness",
                "framed_text": low_v,
            })

        if high_v:
            units.append({
                **common_meta,
                "variant_label": "high_vividness",
                "framed_text": high_v,
            })

    return units


def build_temporal_units(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    units = []
    ts = row.get("framings", {}).get("temporal_slice", {}).get("option_level", {})
    dataset = clean(row.get("dataset"))
    domain = clean(row.get("domain"))
    rid = clean(row.get("id"))
    option_order = clean(row.get("option_order"))
    item_text_map = get_item_text_map(row)

    for option_id, payload in ts.items():
        if not isinstance(payload, dict):
            continue

        source_text = clean(payload.get("source_text")) or clean(item_text_map.get(option_id, ""))
        source_field = clean(payload.get("source_field", "bundle.items.text"))

        short_term = clean(payload.get("short_term"))
        long_term = clean(payload.get("long_term"))

        common_meta = {
            "id": rid,
            "dataset": dataset,
            "domain": domain,
            "option_order": option_order,
            "framing_type": "temporal",
            "option_id": clean(option_id),
            "base_text": source_text,
            "source_text": source_text,
            "source_field": source_field,
            "copied_from": clean(payload.get("copied_from")),
            "original_option_id": clean(payload.get("original_option_id")),
        }

        if short_term:
            units.append({
                **common_meta,
                "variant_label": "short_term",
                "framed_text": short_term,
            })

        if long_term:
            units.append({
                **common_meta,
                "variant_label": "long_term",
                "framed_text": long_term,
            })

    return units


# =========================================================
# Dataset loading
# =========================================================
def collect_jsonl_files(path_str: Optional[str]) -> List[Path]:
    if not path_str:
        return []

    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    raise FileNotFoundError(f"Not found: {path_str}")


def load_eval_units(
    value_tinted_path: Optional[str],
    experiential_path: Optional[str],
    temporal_path: Optional[str],
) -> List[Dict[str, Any]]:
    rows = []

    for fp in collect_jsonl_files(value_tinted_path):
        for row in iter_jsonl(fp):
            rows.extend(build_value_tinted_units(row))

    for fp in collect_jsonl_files(experiential_path):
        for row in iter_jsonl(fp):
            rows.extend(build_experiential_units(row))

    for fp in collect_jsonl_files(temporal_path):
        for row in iter_jsonl(fp):
            rows.extend(build_temporal_units(row))

    return rows


# =========================================================
# Judge
# =========================================================
def judge_one(row: Dict[str, Any], model: str) -> Dict[str, Any]:
    prompt = build_judge_prompt(row)
    raw = call_openrouter(prompt, model=model)
    parsed = normalize_judge_output(raw)

    return {
        **row,
        "llm_judge": parsed,
    }


# =========================================================
# Summary
# =========================================================
def summarize_by_framing_type(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def update_bucket(bucket: Dict[str, Any], judge: Dict[str, Any]):
        bucket["n"] += 1
        bucket["structural_integrity_sum"] += judge["structural_integrity"]
        bucket["framing_salience_sum"] += judge["framing_salience"]
        bucket["framing_purity_sum"] += judge["framing_purity"]

    def finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
        n = bucket["n"]
        if n == 0:
            return {
                "n": 0,
                "avg_structural_integrity": 0.0,
                "avg_framing_salience": 0.0,
                "avg_framing_purity": 0.0,
            }
        return {
            "n": n,
            "avg_structural_integrity": round(bucket["structural_integrity_sum"] / n, 4),
            "avg_framing_salience": round(bucket["framing_salience_sum"] / n, 4),
            "avg_framing_purity": round(bucket["framing_purity_sum"] / n, 4),
        }

    overall_by_framing = {}
    by_framing_dataset = {}
    by_framing_dataset_variant = {}

    for row in rows:
        judge = row["llm_judge"]

        framing_type = clean(row.get("framing_type", "unknown"))
        dataset = clean(row.get("dataset", "unknown"))
        variant = clean(row.get("variant_label", ""))

        key_ft = framing_type
        key_ft_ds = f"{framing_type}::{dataset}"
        key_ft_ds_var = f"{framing_type}::{dataset}::{variant}" if variant else f"{framing_type}::{dataset}"

        if key_ft not in overall_by_framing:
            overall_by_framing[key_ft] = {
                "n": 0,
                "structural_integrity_sum": 0,
                "framing_salience_sum": 0,
                "framing_purity_sum": 0,
            }
        update_bucket(overall_by_framing[key_ft], judge)

        if key_ft_ds not in by_framing_dataset:
            by_framing_dataset[key_ft_ds] = {
                "n": 0,
                "structural_integrity_sum": 0,
                "framing_salience_sum": 0,
                "framing_purity_sum": 0,
            }
        update_bucket(by_framing_dataset[key_ft_ds], judge)

        if key_ft_ds_var not in by_framing_dataset_variant:
            by_framing_dataset_variant[key_ft_ds_var] = {
                "n": 0,
                "structural_integrity_sum": 0,
                "framing_salience_sum": 0,
                "framing_purity_sum": 0,
            }
        update_bucket(by_framing_dataset_variant[key_ft_ds_var], judge)

    return {
        "by_framing_type": {
            k: finalize_bucket(v)
            for k, v in sorted(overall_by_framing.items())
        },
        "by_framing_type_dataset": {
            k: finalize_bucket(v)
            for k, v in sorted(by_framing_dataset.items())
        },
        "by_framing_type_dataset_variant": {
            k: finalize_bucket(v)
            for k, v in sorted(by_framing_dataset_variant.items())
        },
    }


def sample_for_human_eval(rows: List[Dict[str, Any]], ratio: float = 0.1, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    if not rows:
        return []

    n = max(1, int(len(rows) * ratio))
    sampled = random.sample(rows, min(n, len(rows)))

    out = []
    for row in sampled:
        out.append({
            "id": row["id"],
            "dataset": row["dataset"],
            "domain": row["domain"],
            "framing_type": row["framing_type"],
            "variant_label": row.get("variant_label", ""),
            "option_id": row.get("option_id", ""),
            "option_order": row.get("option_order", ""),
            "base_text": row["base_text"],
            "source_text": row.get("source_text", ""),
            "framed_text": row["framed_text"],
            "llm_structural_integrity": row["llm_judge"]["structural_integrity"],
            "llm_framing_salience": row["llm_judge"]["framing_salience"],
            "llm_framing_purity": row["llm_judge"]["framing_purity"],
            "llm_reason": row["llm_judge"]["reason"],
            "human_structural_integrity": "",
            "human_framing_salience": "",
            "human_framing_purity": "",
            "human_notes": "",
        })
    return out


def apply_limit_per_framing_dataset(
    rows: List[Dict[str, Any]],
    limit_per_dataset: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if limit_per_dataset <= 0:
        return rows

    rng = random.Random(seed)

    grouped = {}
    for row in rows:
        ft = clean(row.get("framing_type", ""))
        ds = clean(row.get("dataset", "unknown"))
        key = (ft, ds)
        grouped.setdefault(key, []).append(row)

    limited = []
    for key in sorted(grouped.keys()):
        bucket = grouped[key][:]
        rng.shuffle(bucket)
        limited.extend(bucket[:limit_per_dataset])

    return limited


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--value-tinted-path", type=str, default=None,
                        help="JSONL file or directory for value-tinted narration datasets")
    parser.add_argument("--experiential-path", type=str, default=None,
                        help="JSONL file or directory for experiential/vividness datasets")
    parser.add_argument("--temporal-path", type=str, default=None,
                        help="JSONL file or directory for temporal framing datasets")

    parser.add_argument("--units-output", type=str, required=True,
                        help="Extracted eval units JSONL path")
    parser.add_argument("--judged-output", type=str, required=True,
                        help="Judged JSONL output path")
    parser.add_argument("--summary-output", type=str, required=True,
                        help="Summary JSON output path")
    parser.add_argument("--human-sample-output", type=str, required=True,
                        help="Human eval sample JSONL path")

    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini")
    parser.add_argument("--human-sample-ratio", type=float, default=0.1)
    parser.add_argument("--limit-per-framing", type=int, default=0,
                    help="If > 0, keep up to N eval units for each (framing_type, dataset)")

    args = parser.parse_args()

    units_out = Path(args.units_output)
    judged_out = Path(args.judged_output)
    summary_out = Path(args.summary_output)
    human_out = Path(args.human_sample_output)

    eval_units = load_eval_units(
        value_tinted_path=args.value_tinted_path,
        experiential_path=args.experiential_path,
        temporal_path=args.temporal_path,
    )
    
    eval_units = apply_limit_per_framing_dataset(
        eval_units,
        args.limit_per_framing,
        seed=42,
    )
    print(f"[INFO] Final eval units after per-framing-per-dataset limit: {len(eval_units)}")

    write_jsonl(units_out, eval_units)
    print(f"Extracted {len(eval_units)} eval units")
    print(f"Saved eval units to: {units_out}")

    judged_rows = []
    for i, row in enumerate(eval_units, start=1):
        try:
            judged = judge_one(row, model=args.model)
            judged_rows.append(judged)
        except Exception as e:
            judged_rows.append({
                **row,
                "llm_judge_error": str(e),
            })

        if i % 20 == 0:
            print(f"[{i}/{len(eval_units)}] processed")

    write_jsonl(judged_out, judged_rows)

    valid_rows = [r for r in judged_rows if "llm_judge" in r]
    summary = summarize_by_framing_type(valid_rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    human_rows = sample_for_human_eval(
        valid_rows,
        ratio=args.human_sample_ratio,
        seed=42,
    )
    write_jsonl(human_out, human_rows)

    print(f"Saved judged results to: {judged_out}")
    print(f"Saved summary to: {summary_out}")
    print(f"Saved human sample to: {human_out}")


if __name__ == "__main__":
    main()