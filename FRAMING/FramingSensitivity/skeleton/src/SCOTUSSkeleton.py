import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import os

OPENROUTER_API_KEY = "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"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def safe_json_load(text: str):

    # ```json ... ``` 제거
    text = re.sub(r"^```json", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    # JSON 부분만 추출
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)

    return json.loads(text)


def call_openrouter(prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "SCOTUS Skeleton Generation"
    }

    payload = {
        "model": "qwen/qwen-2.5-72b-instruct", #"openai/gpt-4.1-mini",
        "temperature": 0.2,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    text = data["choices"][0]["message"]["content"]

    return safe_json_load(text)


IN_FILE = Path("/home/swiftie1230/EGO/FRAMING/BaseData/SCOTUS/case_with_all_sources_with_companion_cases_tag.jsonl")
OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/SCOTUS_skeleton.jsonl")

DECISION_Q = "Which legal conclusion should be adopted in this situation?"


# ----------------------------
# Basic utils
# ----------------------------
def clean(x: Any) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def idx_to_label(i: int) -> str:
    return chr(ord("A") + i)


# ----------------------------
# Sentence / text helpers
# ----------------------------
def split_sentences(text: str) -> List[str]:
    text = clean(text)
    if not text:
        return []
    # 간단한 sentence split
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [clean(s) for s in sents if clean(s)]


def remove_holding_like_sentences(sentences: List[str]) -> List[str]:
    """
    정답 leakage를 줄이기 위해 holding을 직접 드러내는 문장을 약하게 필터링.
    너무 공격적으로 제거하면 정보가 날아가므로 최소한으로만.
    """
    bad_patterns = [
        r"\baffirmed\b",
        r"\breversed\b",
        r"\bheld that\b",
        r"\bwe conclude\b",
        r"\bwe hold\b",
        r"\bmay not\b",
        r"\bshould be affirmed\b",
        r"\bjudgment .* affirmed\b",
        r"\bthe court .* affirmed\b",
    ]

    filtered = []
    for s in sentences:
        low = s.lower()
        if any(re.search(p, low) for p in bad_patterns):
            continue
        filtered.append(s)
    return filtered


def first_nonempty(*vals) -> str:
    for v in vals:
        s = clean(v)
        if s:
            return s
    return ""


# ----------------------------
# Core extraction
# ----------------------------
def extract_case_background(row: Dict[str, Any]) -> str:
    """
    사건 배경을 만들기 위한 우선순위:
    1) justia_sections["Case"] 앞부분
    2) justia_sections["Syllabus"] 앞부분
    3) transcript 일부로 fallback
    """
    case_text = safe_get(row, ["justia_sections", "Case"], "")
    syllabus_text = safe_get(row, ["justia_sections", "Syllabus"], "")

    # 1) Case
    for raw in [case_text, syllabus_text]:
        sents = split_sentences(raw)
        sents = remove_holding_like_sentences(sents)

        # 너무 앞부분의 잡다한 citation/header 제거
        cleaned = []
        for s in sents:
            low = s.lower()
            if any(tok in low for tok in [
                "u.s. supreme court",
                "certiorari to",
                "argued",
                "decided",
                "syllabus",
                "page ",
                "[footnote"
            ]):
                continue
            cleaned.append(s)

        if cleaned:
            # 앞 2~4문장 정도만 사용
            return clean(" ".join(cleaned[:4]))

    # 2) transcript fallback
    utterances = safe_get(row, ["convos", "utterances"], [])
    if utterances and isinstance(utterances, list) and len(utterances) > 0:
        first_conv = utterances[0]
        texts = []
        for u in first_conv:
            txt = clean(u.get("text", ""))
            if txt:
                texts.append(txt)
            if len(texts) >= 4:
                break
        if texts:
            return clean(" ".join(texts))

    return ""


def extract_legal_issue(row: Dict[str, Any], background: str) -> str:
    """
    legal issue는 완전 정교 추출 대신 rule-based.
    샘플처럼 probation / authority / question 류를 우선 사용.
    """
    syllabus_text = safe_get(row, ["justia_sections", "Syllabus"], "")
    case_text = safe_get(row, ["justia_sections", "Case"], "")

    candidates = split_sentences(syllabus_text) + split_sentences(case_text)

    # "whether ..." 패턴 우선 탐색
    for s in candidates:
        low = s.lower()
        if "whether" in low:
            return clean(s)

    # fallback: background에서 question-like 정리
    if background:
        title = clean(row.get("title", "this case"))
        return f"What legal conclusion should be drawn in {title} based on the described facts?"

    return "What legal conclusion should be drawn based on the described facts?"


def infer_gold_label_and_options(row: Dict[str, Any], background: str, legal_issue: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """
    현재 버전은 binary legal decision skeleton을 강제.
    기본적으로:
      - A: court retains authority / petitioner-favoring interpretation
      - B: court does not retain authority / respondent-favoring interpretation

    gold_label은 sample처럼 win_side와 advocates side를 최대한 이용.
    다만 모든 사건에 일반화되지는 않으므로, 불명확하면 None 반환하도록 설계.
    """
    # 샘플 구조 기준:
    # side 0 / side 1 존재
    # petitioner/respondent 승패 해석이 row마다 완전히 self-evident 하지 않을 수 있음
    # 그래서 보수적으로 기본 option을 일반 binary legal conclusion으로 만들고,
    # gold는 win_side를 활용하되 불명확하면 None.
    issue_lower = legal_issue.lower()
    bg_lower = background.lower()

    # probation/authority/template-like default
    if any(k in issue_lower or k in bg_lower for k in [
        "probation", "authority", "suspend", "jurisdiction", "consecutive", "remaining terms"
    ]):
        options = [
            "The court should retain authority to grant the requested legal relief under these circumstances.",
            "The court should not retain authority to grant the requested legal relief under these circumstances."
        ]
    else:
        # generic binary legal choice
        options = [
            "The court should adopt the interpretation favoring the requested relief.",
            "The court should reject the interpretation favoring the requested relief."
        ]

    # gold_label 추정
    # 여기서는 매우 보수적으로:
    # - win_side가 존재하면 petitioner 쪽이 이겼는지 respondent 쪽이 이겼는지 직접 매핑하기 어렵다.
    # - sample에서는 win_side=0.0 이고 gold는 B였다.
    # 그래서 범용 자동화에서는 gold를 무조건 세팅하는 대신, heuristic이 명확할 때만 세팅.
    #
    # 다만 skeleton benchmark용으로 gold가 꼭 필요하다면 아래 heuristic을 사용:
    # sample-like 법적 relief 요청 사건에서는 relief granted -> A, denied -> B
    # "affirmed"가 있고 lower-court denial/against petitioner 맥락이면 B일 가능성이 큼.
    #
    # 여기서는 leakage 없는 최소 heuristic:
    text_all = " ".join([
        clean(safe_get(row, ["justia_sections", "Syllabus"], "")),
        clean(safe_get(row, ["justia_sections", "Case"], ""))
    ]).lower()

    gold = None

    if "may not" in text_all or "no power" in text_all or "no authority" in text_all:
        gold = "B"
    elif "may suspend" in text_all or "retains authority" in text_all or "has power" in text_all:
        gold = "A"

    # 너무 불안정하면 None 유지
    rule = "follow_case_holding" if gold is not None else "none"
    return gold, options, rule


def extract_core_phrases(background: str, legal_issue: str) -> List[str]:
    text = f"{background} {legal_issue}".lower()

    candidates = [
        "consecutive sentences",
        "multiple counts",
        "first term has begun",
        "remaining unserved terms",
        "grant probation",
        "suspend sentence",
        "district court",
        "legal authority",
        "judicial authority",
        "requested legal relief",
    ]

    kept = [c for c in candidates if c in text]
    return kept[:6]


def build_base_scenario(background: str, legal_issue: str) -> str:
    bg = clean(background)
    issue = clean(legal_issue)

    if bg and issue:
        # issue가 whether 문장일 때 자연스럽게 붙이기
        if issue.lower().startswith("whether "):
            return clean(f"{bg} The legal question is {issue}")
        return clean(f"{bg} The legal question is: {issue}")
    return bg or issue


def build_generation_prompt(row: Dict[str, Any]) -> str:
    title = clean(row.get("title", ""))
    case_text = safe_get(row, ["justia_sections", "Case"], "")
    syllabus = safe_get(row, ["justia_sections", "Syllabus"], "")

    raw = clean(case_text or syllabus)

    prompt = f"""
You are constructing a neutral legal decision task.

Summarize the following Supreme Court case and generate a decision task.

Requirements:

1. Scenario
- Summarize the legal dispute in 2-3 sentences.
- Focus only on the legal question.
- Do NOT reveal the court's final decision.
- Avoid evaluative language.

2. Options
- Generate two mutually exclusive legal conclusions.
- Use parallel sentence structure.
- Keep the options similar in length.
- Do NOT imply which option is correct.

3. Decision Question
- Write a neutral question asking which legal conclusion should be adopted.

Return JSON only:

{{
  "scenario": "...",
  "options": {{
    "A": "...",
    "B": "..."
  }},
  "decision_question": "..."
}}

Case title:
{title}

Case text:
{raw[:4000]}
"""
    return prompt

def generate_llm_skeleton(row: Dict[str, Any]):
    prompt = build_generation_prompt(row)

    try:
        result = call_openrouter(prompt)
    except Exception as e:
        print("LLM error:", e)
        return None

    scenario = clean(result.get("scenario"))
    options = result.get("options", {})
    decision_question = clean(result.get("decision_question", DECISION_Q))

    if not scenario or "A" not in options or "B" not in options:
        return None

    return scenario, options, decision_question


# ----------------------------
# Build skeleton
# ----------------------------
def build_skeleton(row: Dict[str, Any], option_order: str = "original") -> Optional[Dict[str, Any]]:
    case_id = clean(row.get("id", ""))
    title = clean(row.get("title", ""))
    dataset = "SUPER_SCOTUS"

    if not case_id or not title:
        return None

    llm_out = generate_llm_skeleton(row)
    if llm_out is None:
        return None

    scenario, options_dict, decision_question = llm_out

    options = [
        options_dict.get("A"),
        options_dict.get("B")
    ]

    if not options[0] or not options[1]:
        return None

    gold_label = None
    gold_rule = "none"
    legal_issue = ""
    if len(options) != 2:
        return None

    items = [
        {"item_id": "A", "text": clean(options[0])},
        {"item_id": "B", "text": clean(options[1])},
    ]

    skeleton = {
        "id": f"legal_{dataset}_{case_id}",
        "dataset": dataset,
        "domain": "legal_decision",

        "base": {
            "case_id": case_id,
            "case_title": title,
            "scenario": scenario,
            "legal_issue": legal_issue,
            "core_phrases_preserved": extract_core_phrases(scenario, scenario)
        },

        "bundle": {
            "items": items,
            "resource": {
                "name": "single legal decision",
                "quantity": 1
            }
        },

        "decision_question": decision_question,
        "label_space": ["A", "B"],
        "gold_label": gold_label,
        "gold_label_rule": gold_rule,
        "option_order": option_order,

        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        },

        "meta": {
            "petitioner": clean(row.get("petitioner", "")) or None,
            "respondent": clean(row.get("respondent", "")) or None,
            "court": clean(row.get("court", "")) or None,
            "year": row.get("year", None),
            "decided_date": clean(row.get("decided_date", "")) or None,
            "citation": clean(row.get("citation", "")) or None,
            "source_case_id": case_id,
            "issue_area": safe_get(row, ["scdb_elements", "issueArea"], None),
            "maj_votes": safe_get(row, ["scdb_elements", "majVotes"], None),
            "min_votes": safe_get(row, ["scdb_elements", "minVotes"], None),
            "win_side": row.get("win_side", None),
            "is_eq_divided": row.get("is_eq_divided", None),
            "companion_cases": row.get("companion_cases", None)
        }
    }

    return skeleton


def remap_label(label: Optional[str], old_to_new: Dict[str, str]) -> Optional[str]:
    if label is None:
        return None
    return old_to_new.get(label, label)


def build_swapped_skeleton(original_sk: Dict[str, Any]) -> Dict[str, Any]:
    swapped = json.loads(json.dumps(original_sk))  # deep copy

    orig_items = original_sk["bundle"]["items"]
    reversed_items = list(reversed(orig_items))
    new_labels = [idx_to_label(i) for i in range(len(reversed_items))]

    relabeled_items = []
    old_to_new = {}
    for new_label, old_item in zip(new_labels, reversed_items):
        old_label = old_item["item_id"]
        old_to_new[old_label] = new_label

        new_item = dict(old_item)
        new_item["item_id"] = new_label
        relabeled_items.append(new_item)

    swapped["bundle"]["items"] = relabeled_items
    swapped["label_space"] = [it["item_id"] for it in relabeled_items]
    swapped["option_order"] = "swapped"

    # gold_label remap
    swapped["gold_label"] = remap_label(swapped.get("gold_label"), old_to_new)

    return swapped


# ----------------------------
# Main
# ----------------------------
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    skipped = 0
    skipped_reasons = {}

    with OUT.open("w", encoding="utf-8") as fout:
        for row in iter_jsonl(IN_FILE):
            total_in += 1

            sk_original = build_skeleton(row, option_order="original")
            if sk_original is None:
                skipped += 1
                skipped_reasons["build_failed_or_missing_required_fields"] = (
                    skipped_reasons.get("build_failed_or_missing_required_fields", 0) + 1
                )
                continue

            fout.write(json.dumps(sk_original, ensure_ascii=False) + "\n")
            total_out += 1

            sk_swapped = build_swapped_skeleton(sk_original)
            fout.write(json.dumps(sk_swapped, ensure_ascii=False) + "\n")
            total_out += 1

    print(f"✅ Wrote: {OUT}")
    print(f"total_in={total_in}, total_out={total_out}, skipped={skipped}")
    if skipped_reasons:
        print("skipped_reasons:", skipped_reasons)


if __name__ == "__main__":
    main()