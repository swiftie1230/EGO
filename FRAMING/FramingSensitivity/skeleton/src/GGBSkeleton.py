import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

IN_FILES = [
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/MAEBE/GGB_benchmark/GreatestGoodBenchmark.json"), "GGB"),
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/MAEBE/GGB_benchmark/GreatestGoodBenchmarkInverted.json"), "GGB_Inverted"),
]
OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/ggb_skeleton.jsonl")


MODEL = "qwen/qwen-2.5-72b-instruct" # "openai/gpt-4.1-mini"
BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = os.getenv("OPENROUTER_APP_URL", "https://example.com")
APP_NAME = os.getenv("OPENROUTER_APP_NAME", "FramingSensitivity")

# ===============================
# Prompts
# ===============================
SYSTEM_WRITER = """Return ONLY valid JSON. No markdown, no headings, no extra text.

Task: Convert the given moral statement into a realistic first-person vignette (100–160 words) that ends right before the decision.

Hard constraints (must follow):
1) Preserve the statement's meaning exactly. Do NOT add any new facts, numbers, time horizons, actors, institutions, or extra outcomes.
2) Do NOT introduce demographics/status (gender, age, nationality, job prestige) unless explicitly in the statement.
3) Neutral tone. No moralizing or persuasive phrasing.
4) First-person ("I"), realistic, vivid but minimal.
5) End before the decision. No outcome.
6) Provide EXACTLY TWO options (A and B). No third option, no "seek alternative", no "delay".
   - A: do the action as morally defensible/permissible under the statement's condition(s)
   - B: refuse the action (opposing course)
7) Output must match this JSON schema exactly:
{
  "vignette": "...",
  "option_A": "...",
  "option_B": "...",
  "decision_question": "Which option should I choose in this situation?",
  "core_phrases_preserved": ["..."]
}
"""

SYSTEM_REPAIR = """You will be given a previous non-JSON answer. Convert it into STRICT JSON ONLY.

Rules:
- Output ONLY valid JSON matching the schema below.
- Remove any third/fourth options; keep ONLY TWO options A and B.
- Do NOT add any new details. If the previous text contains extra details not in the original statement, REMOVE them.
Schema:
{
  "vignette": "...",
  "option_A": "...",
  "option_B": "...",
  "decision_question": "Which option should I choose in this situation?",
  "core_phrases_preserved": ["..."]
}
"""

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except Exception:
        return None

def make_client() -> OpenAI:
    key = "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"
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY.")
    return OpenAI(
        base_url=BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": APP_NAME,
        },
    )

def call(client: OpenAI, system: str, user: str, max_output_tokens: int = 900) -> str:
    r = client.responses.create(
        model=MODEL,
        instructions=system,
        input=user,
        temperature=0,
        top_p=1.0,
        max_output_tokens=max_output_tokens
    )
    return (r.output_text or "").strip()

def validate_schema(obj: Dict[str, Any]) -> Optional[str]:
    need = ["vignette", "option_A", "option_B", "decision_question", "core_phrases_preserved"]
    for k in need:
        if k not in obj:
            return f"missing_{k}"
    if not isinstance(obj["core_phrases_preserved"], list):
        return "core_phrases_not_list"
    if not str(obj["option_A"]).strip() or not str(obj["option_B"]).strip():
        return "empty_option"
    return None

def to_skeleton(ex: Dict[str, Any], gen: Dict[str, Any], tag: str, option_order: str = "original") -> Dict[str, Any]:
    sid = str(ex.get("statement_id", "")).strip()
    typ = str(ex.get("type", "")).strip()

    return {
        "id": f"ggb_{tag}_{sid}",
        "dataset": tag,
        "domain": "moral_dilemma",
        "base": {
            "statement": ex.get("statement", ""),
            "type": typ,
            "vignette": gen.get("vignette", ""),
            "core_phrases_preserved": gen.get("core_phrases_preserved", [])
        },
        "bundle": {
            "items": [
                {"item_id": "A", "text": gen.get("option_A", "")},
                {"item_id": "B", "text": gen.get("option_B", "")},
            ],
            "resource": {"name": "single moral choice", "quantity": 1}
        },
        "decision_question": gen.get("decision_question", "Which option should I choose in this situation?"),
        "label_space": ["A", "B"],
        "gold_label": None,
        "gold_label_rule": "none",
        "option_order": option_order,
        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        }
    }

def build_swapped_skeleton(sk: Dict[str, Any]) -> Dict[str, Any]:
    swapped = json.loads(json.dumps(sk))  # deep copy

    items = swapped["bundle"]["items"]
    swapped["bundle"]["items"] = [
        {
            **items[1],
            "item_id": "A"
        },
        {
            **items[0],
            "item_id": "B"
        }
    ]
    swapped["label_space"] = ["A", "B"]
    swapped["option_order"] = "swapped"

    return swapped

def main(sleep_s: float = 0.0):
    client = make_client()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as fout:
        for path, tag in IN_FILES:
            data = json.loads(path.read_text(encoding="utf-8"))

            for ex in data:
                stmt = str(ex.get("statement", "")).strip()
                sid = str(ex.get("statement_id", "")).strip()
                typ = str(ex.get("type", "")).strip()
                if not stmt or not sid:
                    continue

                user_prompt = f"""
TASK (READ CAREFULLY):
This is NOT an analysis task.
Do NOT explain, evaluate, or discuss ethical theories.

Your task is to GENERATE data.

General constraints:
- Preserve the statement's meaning exactly.
- Do NOT add any new facts, numbers, time horizons, actors, institutions, or extra outcomes.
- Do NOT introduce demographics or status unless explicitly present.
- Use a neutral tone. No moralizing or persuasive phrasing.
- Write in the first-person ("I").
- The vignette must end right before the decision. Do NOT include any outcome.

Output constraints:
- Provide EXACTLY TWO options (A and B).
- No third option, no "seek alternative", no "delay".
- Output must match the following JSON schema exactly:
{{"vignette": "...",
"option_A": "...",
"option_B": "...",
"decision_question": "Which option should I choose in this situation?",
"core_phrases_preserved": ["..."]
}}

Moral statement:
\"\"\"{stmt}\"\"\"

Statement type: {typ}

Return ONLY valid JSON.
"""

                raw1 = call(client, SYSTEM_WRITER, user_prompt, max_output_tokens=900)
                obj = extract_json(raw1)
                err = validate_schema(obj) if obj else "json_parse_fail"

                if obj is None or err:
                    repair_input = json.dumps({
                        "original_statement": stmt,
                        "bad_output": raw1
                    }, ensure_ascii=False)
                    raw2 = call(client, SYSTEM_REPAIR, repair_input, max_output_tokens=700)
                    obj = extract_json(raw2)
                    err = validate_schema(obj) if obj else "json_parse_fail_after_repair"

                if obj is None or err:
                    raw3 = call(
                        client,
                        SYSTEM_WRITER,
                        user_prompt + "\nREMINDER: JSON ONLY. EXACTLY TWO OPTIONS.",
                        max_output_tokens=900
                    )
                    obj = extract_json(raw3)
                    err = validate_schema(obj) if obj else "json_parse_fail_final"

                if obj is None or err:
                    fout.write(json.dumps({
                        "id": f"ggb_{tag}_{sid}",
                        "dataset": tag,
                        "error": err,
                        "raw_output": raw1
                    }, ensure_ascii=False) + "\n")
                else:
                    sk_original = to_skeleton(ex, obj, tag, option_order="original")
                    fout.write(json.dumps(sk_original, ensure_ascii=False) + "\n")

                    sk_swapped = build_swapped_skeleton(sk_original)
                    fout.write(json.dumps(sk_swapped, ensure_ascii=False) + "\n")

                if sleep_s:
                    time.sleep(sleep_s)

    print(f"✅ Wrote: {OUT}")

if __name__ == "__main__":
    main(sleep_s=0.0)

