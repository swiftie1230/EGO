import json
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

IN_FILES = [
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/UniMoral/data/English_short_formatted.csv"), "UniBench_English_short"),
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/UniMoral/data/English_long_formatted.csv"), "UniBench_English_long"),
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/UniMoral/data/English_short_withDemo.csv"), "UniBench_English_short_withDemo"),
    (Path("/home/swiftie1230/EGO/FRAMING/BaseData/UniMoral/data/English_long_withDemo.csv"), "UniBench_English_long_withDemo"),
]

OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/unibench_skeleton.jsonl")

DECISION_Q = "Which action should I choose in this situation?"

def clean(x: Any) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_possible_actions(raw: Any) -> Tuple[List[str], Optional[str]]:
    """
    Returns (actions, error_reason)
    - actions: list[str] (cleaned)
    - error_reason: None if ok, else string
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return [], "missing"

    # Already a list?
    if isinstance(raw, list):
        acts = [clean(a) for a in raw if clean(a)]
        return acts, None if len(acts) >= 2 else "too_few_actions"

    s = clean(raw)
    if not s:
        return [], "empty"

    # 1) JSON loads
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            acts = [clean(a) for a in obj if clean(a)]
            return acts, None if len(acts) >= 2 else "too_few_actions"
    except Exception:
        pass

    # 2) Python literal eval (UniBench는 이 형태가 흔함: "['a','b']" or "[\"a\", \"b\"]")
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            acts = [clean(a) for a in obj if clean(a)]
            return acts, None if len(acts) >= 2 else "too_few_actions"
    except Exception:
        pass

    # 3) Fallback: bracket 안의 quote 문자열을 최대한 추출
    # e.g. ['Promote ...', 'Do nothing ...'] / ["Promote ...", "Do nothing ..."]
    candidates = re.findall(r"""['"]([^'"]+)['"]""", s)
    acts = [clean(a) for a in candidates if clean(a)]
    if len(acts) >= 2:
        return acts, None

    return [], "parse_fail"

def idx_to_label(i: int) -> str:
    # 0->A, 1->B, ...
    return chr(ord("A") + i)

def normalize_selected_action(sel: Any, n_actions: int) -> Optional[str]:
    """
    UniBench의 Selected_action은 gold가 아니라 annotator choice로 보관.
    보통 1-based index (1..n) 형태. 안전하게 처리.
    """
    if sel is None or (isinstance(sel, float) and pd.isna(sel)):
        return None
    try:
        k = int(sel)
        # 1-based
        if 1 <= k <= n_actions:
            return idx_to_label(k - 1)
        # 0-based가 섞였을 가능성도 방어
        if 0 <= k < n_actions:
            return idx_to_label(k)
    except Exception:
        pass
    return None

def remap_label(label: Optional[str], old_to_new: Dict[str, str]) -> Optional[str]:
    if label is None:
        return None
    return old_to_new.get(label, label)

def build_skeleton_for_group(
    dataset_tag: str,
    scenario_id: str,
    scenario: str,
    actions: List[str],
    group_rows: List[Dict[str, Any]],
    option_order: str = "original",
) -> Dict[str, Any]:
    label_space = [idx_to_label(i) for i in range(len(actions))]
    items = [{"item_id": label_space[i], "text": actions[i]} for i in range(len(actions))]

    # annotator별 선택/메타 모으기
    annotations = []
    for r in group_rows:
        ann = {
            "annotator_id": clean(r.get("Annotator_id", "")) or None,
            "selected_action": normalize_selected_action(r.get("Selected_action", None), len(actions)),
            "moral_values": r.get("Moral_values", None),
            "cultural_values": r.get("Cultural_values", None),
            "self_description": r.get("Annotator_self_description", None),
        }
        if "annotator_demographics" in r:
            ann["annotator_demographics"] = r.get("annotator_demographics", None)

        # long에만 존재하는 필드들(있으면 보관)
        for k in ["Reason", "Consequence", "Action_criteria", "Contributing_factors", "Contributing_emotion"]:
            if k in r and pd.notna(r[k]):
                ann[k] = r[k]
        annotations.append(ann)

    # 선택 분포 요약(나중에 분석에 유용)
    dist = {}
    for a in annotations:
        sa = a.get("selected_action")
        if sa:
            dist[sa] = dist.get(sa, 0) + 1

    return {
        "id": f"unibench_{dataset_tag}_{scenario_id}",
        "dataset": dataset_tag,
        "domain": "decision_choice",

        "base": {
            "scenario_id": scenario_id,
            "scenario": scenario
        },

        "bundle": {
            "items": items,
            "resource": {"name": "single decision", "quantity": 1}
        },

        "decision_question": DECISION_Q,
        "label_space": label_space,

        # gold 없음
        "gold_label": None,
        "gold_label_rule": "none",
        "option_order": option_order,

        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        },

        "meta": {
            "n_annotators": len(annotations),
            "selection_distribution": dist,
            "annotations": annotations
        }
    }

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

    # annotation의 selected_action 재매핑
    new_annotations = []
    for ann in swapped["meta"]["annotations"]:
        new_ann = dict(ann)
        new_ann["selected_action"] = remap_label(new_ann.get("selected_action"), old_to_new)
        new_annotations.append(new_ann)
    swapped["meta"]["annotations"] = new_annotations

    # selection_distribution 재계산
    new_dist = {}
    for ann in new_annotations:
        sa = ann.get("selected_action")
        if sa:
            new_dist[sa] = new_dist.get(sa, 0) + 1
    swapped["meta"]["selection_distribution"] = new_dist

    return swapped

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    total_out = 0
    skipped = 0
    skipped_reasons = {}

    with OUT.open("w", encoding="utf-8") as fout:
        for path, tag in IN_FILES:
            df = pd.read_csv(path)

            # row들을 dict로 변환
            rows = [r._asdict() if hasattr(r, "_asdict") else r for r in df.to_dict(orient="records")]

            # scenario_id 기준으로 그룹화
            by_sid: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                sid = clean(r.get("Scenario_id", ""))
                if not sid:
                    skipped += 1
                    skipped_reasons["missing_scenario_id"] = skipped_reasons.get("missing_scenario_id", 0) + 1
                    continue
                by_sid.setdefault(sid, []).append(r)

            for sid, grp in by_sid.items():
                # 대표 scenario: 첫 row의 Scenario (대부분 동일)
                scenario = clean(grp[0].get("Scenario", ""))
                if not scenario:
                    skipped += 1
                    skipped_reasons["missing_scenario"] = skipped_reasons.get("missing_scenario", 0) + 1
                    continue

                # 대표 actions: 첫 row의 Possible_actions로 파싱 (대부분 동일)
                actions, err = parse_possible_actions(grp[0].get("Possible_actions", None))
                if err is not None:
                    # 혹시 첫 row만 이상할 수 있으니, 그룹 내에서 파싱 가능한 걸 찾기
                    recovered = False
                    for r in grp[1:]:
                        actions, err2 = parse_possible_actions(r.get("Possible_actions", None))
                        if err2 is None:
                            recovered = True
                            err = None
                            break
                    if not recovered:
                        skipped += 1
                        skipped_reasons[err] = skipped_reasons.get(err, 0) + 1
                        continue

                # original
                sk_original = build_skeleton_for_group(
                    dataset_tag=tag,
                    scenario_id=sid,
                    scenario=scenario,
                    actions=actions,
                    group_rows=grp,
                    option_order="original",
                )
                fout.write(json.dumps(sk_original, ensure_ascii=False) + "\n")
                total_out += 1

                # swapped
                sk_swapped = build_swapped_skeleton(sk_original)
                fout.write(json.dumps(sk_swapped, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"✅ Wrote: {OUT}")
    print(f"total_out={total_out}, skipped={skipped}")
    if skipped_reasons:
        print("skipped_reasons:", skipped_reasons)

if __name__ == "__main__":
    main()