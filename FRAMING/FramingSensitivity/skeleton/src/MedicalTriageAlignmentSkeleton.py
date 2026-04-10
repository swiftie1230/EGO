import json
from pathlib import Path
from typing import Any, Dict, List, Optional

IN_FILE = Path("/home/swiftie1230/EGO/FRAMING/BaseData/MedicalTriageAlignment/paper-dataset-1-12.json")
OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/medical_triage_alignment_skeleton.jsonl")


# ----------------------------
# Utils
# ----------------------------
def clean(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).split()).strip()


def idx_to_label(i: int) -> str:
    return chr(ord("A") + i)


def remap_label(label: Optional[str], old_to_new: Dict[str, str]) -> Optional[str]:
    if label is None:
        return None
    return old_to_new.get(label, label)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Core helpers
# ----------------------------
def infer_item_type(probe: str) -> str:
    probe_low = clean(probe).lower()
    if probe_low.startswith("who "):
        return "patient_choice"
    if probe_low.startswith("why"):
        return "justification"
    return "choice"


def infer_task_type(probe: str) -> str:
    probe_low = clean(probe).lower()
    if probe_low.startswith("who "):
        return "patient_prioritization"
    if probe_low.startswith("why"):
        return "justification_selection"
    return "choice_selection"


def extract_value_dimension(score_list: List[Dict[str, Any]]) -> Optional[str]:
    """
    score_list 예:
      [{"fairness": 10.0}, {"fairness": 0.0}]
    또는
      [{"utilitarianism": 10.0}, ...]
    """
    if not score_list:
        return None
    first = score_list[0]
    if not isinstance(first, dict) or len(first) != 1:
        return None
    return next(iter(first.keys()))


def extract_scores(score_list: List[Dict[str, Any]], value_dim: str) -> List[Optional[float]]:
    scores = []
    for d in score_list:
        if not isinstance(d, dict):
            scores.append(None)
            continue
        val = d.get(value_dim, None)
        try:
            scores.append(float(val))
        except Exception:
            scores.append(None)
    return scores


def choose_gold_label(scores: List[Optional[float]]) -> (Optional[str], bool):
    """
    max score 기준.
    tie면 gold_label=None, tie=True
    """
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None, False

    max_score = max(s for _, s in valid)
    max_idxs = [i for i, s in valid if s == max_score]

    if len(max_idxs) != 1:
        return None, True

    return idx_to_label(max_idxs[0]), False


# ----------------------------
# Skeleton build
# ----------------------------
def build_skeleton(entry, option_order="original") -> Optional[Dict[str, Any]]:
    """
    entry format:
    [
      {...meta/probe...},
      [{...score...}, {...score...}, ...]
    ]
    """
    if not isinstance(entry, list) or len(entry) != 2:
        return None

    info, score_list = entry
    if not isinstance(info, dict) or not isinstance(score_list, list):
        return None

    scenario_id = clean(info.get("scenario_id"))
    probe_id = clean(info.get("probe_id"))
    scenario = clean(info.get("scenario"))
    state = info.get("state", None)
    probe = clean(info.get("probe"))
    choices = info.get("choices", [])

    # ----------------------------
    # decision task만 사용
    # ----------------------------
    if state is not None:
        return None

    if not scenario_id or not probe_id or not scenario or not probe or not isinstance(choices, list) or len(choices) < 2:
        return None

    value_dim = extract_value_dimension(score_list)
    if value_dim is None:
        return None

    scores = extract_scores(score_list, value_dim)
    if len(scores) != len(choices):
        return None

    gold_label, tie = choose_gold_label(scores)

    item_type = infer_item_type(probe)
    task_type = infer_task_type(probe)

    items = []
    label_space = []
    for i, (choice_text, score) in enumerate(zip(choices, scores)):
        label = idx_to_label(i)
        label_space.append(label)

        items.append({
            "item_id": label,
            "type": item_type,
            "priority_score": score,
            "situation_text": clean(choice_text)
        })

    sk = {
        "id": f"medical_triage_alignment_{value_dim}_{probe_id}",
        "dataset": "MedicalTriageAlignment",
        "domain": "life_safety",

        "base": {
            "scenario_id": scenario_id,
            "probe_id": probe_id,
            "scenario": scenario,
            "state": state,
            "value_dimension": value_dim
        },

        "bundle": {
            "items": items,
            "resource": {
                "name": "single medical resource",
                "quantity": 1
            }
        },

        "decision_question": probe,
        "label_space": label_space,
        "gold_label_rule": f"choose_max_{value_dim}",
        "gold_label": gold_label,
        "tie": tie,
        "option_order": option_order,

        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        },

        "meta": {
            "task_type": task_type
        }
    }

    return sk


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
    swapped["gold_label"] = remap_label(swapped.get("gold_label"), old_to_new)

    return swapped


# ----------------------------
# Main
# ----------------------------
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(IN_FILE)

    total_in = 0
    total_out = 0
    skipped = 0
    skipped_reasons = {}

    with OUT.open("w", encoding="utf-8") as fout:
        for entry in data:
            total_in += 1

            sk_original = build_skeleton(entry, option_order="original")
            if sk_original is None:
                skipped += 1
                skipped_reasons["build_failed"] = skipped_reasons.get("build_failed", 0) + 1
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
