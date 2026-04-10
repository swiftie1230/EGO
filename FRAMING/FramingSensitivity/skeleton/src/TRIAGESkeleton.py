import pandas as pd
import json
import random
import re
from pathlib import Path

TRIAGE_IN = Path("/home/swiftie1230/EGO/FRAMING/BaseData/Triage/datasets/triage_questions.csv")
TRIAGE_OUT = Path("/home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl")

def clean(x):
    if x is None: return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ZONE_TO_URGENCY = {"Red": 3, "Yellow": 2, "Green": 1, "Black": 0}
ZONE_TO_LABEL   = {"Red": "Immediate", "Yellow": "Delayed", "Green": "Minor", "Black": "Expectant"}

df = pd.read_csv(TRIAGE_IN)

rows = []
for i, row in df.iterrows():
    q  = clean(row.get("question"))
    z  = clean(row.get("triage_zone"))
    if not q or z not in ZONE_TO_URGENCY:
        continue
    rows.append({
        "idx": i,
        "question": q,
        "zone": z,
        "urgency": ZONE_TO_URGENCY[z],
        "gold_class": ZONE_TO_LABEL[z],
        "class_definition": clean(row.get("class")),
        "action": clean(row.get("action"))
    })

# group by zone for controlled sampling
by_zone = {}
for r in rows:
    by_zone.setdefault(r["zone"], []).append(r)

random.seed(7)

def sample_bundle(k=2):
    """
    Sample a bundle of k patients with mixed zones when possible.
    """
    # prefer mixing: pick distinct zones first
    zones = list(by_zone.keys())
    random.shuffle(zones)
    picked = []

    # 1) try distinct zones
    for z in zones:
        if len(picked) >= k: break
        if by_zone[z]:
            picked.append(random.choice(by_zone[z]))

    # 2) if not enough, fill randomly
    while len(picked) < k:
        picked.append(random.choice(rows))

    # determine gold: max urgency (tie => pick first max; you can mark tie too)
    max_u = max(p["urgency"] for p in picked)
    max_items = [p for p in picked if p["urgency"] == max_u]
    if len(max_items) == 1:
        gold_item = max_items[0]
        gold = gold_item["idx"]
        tie = False
    else:
        # tie handling
        gold = "tie"
        tie = True

    return picked, gold, tie

records = []
N_BUNDLES_2 = 400   # 원하는 만큼
N_BUNDLES_3 = 0

bundle_id = 0
for k, n in [(2, N_BUNDLES_2), (3, N_BUNDLES_3)]:
    for _ in range(n):
        picked, gold, tie = sample_bundle(k=k)

        items = []
        idx_to_label = {}

        for j, p in enumerate(picked):
            item_label = chr(ord("A") + j)   # A, B, C ...
            idx_to_label[p["idx"]] = item_label

            items.append({
                "item_id": item_label,
                "type": "patient",
                "triage_zone": p["zone"],
                "urgency": p["urgency"],
                "situation_text": p["question"],
                "source_idx": p["idx"]
            })

        gold_label = idx_to_label[gold] if gold != "tie" else "tie"

        # 1) original
        rec = {
            "id": f"triage_alloc_{bundle_id:05d}",
            "dataset": "TRIAGE",
            "domain": "life_safety",

            "bundle": {
                "items": items,
                "resource": {"name": "single medical resource", "quantity": 1}
            },

            "decision_question": "Given one available resource right now, who should receive it first?",
            "label_space": [it["item_id"] for it in items],
            "gold_label_rule": "choose_max_urgency",
            "gold_label": gold_label,
            "tie": tie,
            "option_order": "original",

            "framings": {
                "bluffing_contextual_envelope": {},
                "narrative_distance": {},
                "observer_viewpoint": {},
                "temporal_slice": {}
            }
        }
        records.append(rec)
        #bundle_id += 1

        # 2) swapped
        swapped_items = list(reversed(items))

        # item_id를 다시 A, B, C로 재부여
        relabeled_items = []
        old_to_new = {}
        for j, item in enumerate(swapped_items):
            new_label = chr(ord("A") + j)
            old_to_new[item["item_id"]] = new_label

            new_item = dict(item)
            new_item["item_id"] = new_label
            relabeled_items.append(new_item)

        swapped_gold_label = old_to_new[gold_label] if gold_label != "tie" else "tie"

        rec_swapped = {
            "id": f"triage_alloc_{bundle_id:05d}",
            "dataset": "TRIAGE",
            "domain": "life_safety",

            "bundle": {
                "items": relabeled_items,
                "resource": {"name": "single medical resource", "quantity": 1}
            },

            "decision_question": "Given one available resource right now, who should receive it first?",
            "label_space": [it["item_id"] for it in relabeled_items],
            "gold_label_rule": "choose_max_urgency",
            "gold_label": swapped_gold_label,
            "tie": tie,
            "option_order": "swapped",

            "framings": {
                "bluffing_contextual_envelope": {},
                "narrative_distance": {},
                "observer_viewpoint": {},
                "temporal_slice": {}
            }
        }
        records.append(rec_swapped)
        bundle_id += 1

with open(TRIAGE_OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote:", TRIAGE_OUT, "bundles:", len(records),
      "ties:", sum(1 for r in records if r["tie"]))
