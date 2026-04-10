import pandas as pd
import json
import re
from pathlib import Path

ROLE_IN = Path("/home/swiftie1230/EGO/FRAMING/BaseData/RoleConflictBench/benchmark/question.csv")
ROLE_OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/roleconflict_allocation.jsonl")

def clean(x):
    if x is None: return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = pd.read_csv(ROLE_IN)

records = []
for i, row in df.iterrows():
    role1 = clean(row.get("Role1"))
    role2 = clean(row.get("Role2"))
    story = clean(row.get("Story"))
    sit1  = clean(row.get("Situation1"))
    sit2  = clean(row.get("Situation2"))
    exp1  = clean(row.get("Expectation1"))
    exp2  = clean(row.get("Expectation2"))

    # urgency / priority signal
    ob1 = row.get("Obligation1")
    ob2 = row.get("Obligation2")

    # gold by urgency (higher obligation = higher urgency)
    gold = None
    if pd.notna(ob1) and pd.notna(ob2):
        ob1i, ob2i = int(ob1), int(ob2)
        if ob1i > ob2i:
            gold = "A"
        elif ob2i > ob1i:
            gold = "B"
        else:
            gold = "tie"

    items = [
        {
            "item_id": "A",
            "type": "role",
            "role_name": role1,
            "obligation_level": None if pd.isna(ob1) else int(ob1),
            "situation": sit1,
            "expectation": exp1
        },
        {
            "item_id": "B",
            "type": "role",
            "role_name": role2,
            "obligation_level": None if pd.isna(ob2) else int(ob2),
            "situation": sit2,
            "expectation": exp2
        }
    ]

    rec = {
        "id": f"roleconflict_alloc_{i:05d}",
        "dataset": "RoleConflictBench_csv",
        "domain": "role_conflict",

        "bundle": {
            "items": items,
            "resource": {"name": "single action slot", "quantity": 1},
            "shared_story": story
        },

        "decision_question": "Which role should I prioritize more in this situation?",
        "label_space": ["A", "B"],
        "gold_label_rule": "choose_higher_obligation_level",
        "gold_label": gold,
        "option_order": "original",

        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        }
    }
    records.append(rec)

    # swapped
    swapped_items = [
        {
            **items[1],
            "item_id": "A"
        },
        {
            **items[0],
            "item_id": "B"
        }
    ]

    if gold == "A":
        swapped_gold = "B"
    elif gold == "B":
        swapped_gold = "A"
    else:
        swapped_gold = gold

    rec_swapped = {
        "id": f"roleconflict_alloc_{i:05d}",
        "dataset": "RoleConflictBench_csv",
        "domain": "role_conflict",

        "bundle": {
            "items": swapped_items,
            "resource": {"name": "single action slot", "quantity": 1},
            "shared_story": story
        },

        "decision_question": "Which role should I prioritize more in this situation?",
        "label_space": ["A", "B"],
        "gold_label_rule": "choose_higher_obligation_level",
        "gold_label": swapped_gold,
        "option_order": "swapped",

        "framings": {
            "bluffing_contextual_envelope": {},
            "narrative_distance": {},
            "observer_viewpoint": {},
            "temporal_slice": {}
        }
    }
    records.append(rec_swapped)

with open(ROLE_OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote:", ROLE_OUT, "rows:", len(records),
      "ties:", sum(1 for r in records if r["gold_label"] == "tie"))