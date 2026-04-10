#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Urgency-based evaluation (urgencyEval):
- Evaluate only TRIAGE + ROLECONFLICT
- TRIAGE gold: choose_max_urgency (gold_label may be item_id OR source_idx int)
- ROLECONFLICT gold: choose_higher_obligation_level (gold_label may be "tie" or item_id)

Inputs:
- skeleton_file (original data) to read gold and item metadata
- pred_file (generated decisions JSONL) to read pred_decision

Outputs:
- accuracy overall + per-domain (life_safety, role_conflict)

Example:
  python urgencyEval.py \
    --skeleton_file /home/swiftie1230/EGO/FRAMING/FramingSensitivity/skeleton/data/triage_allocation.jsonl \
    --pred_file /home/swiftie1230/EGO/FRAMING/FramingSensitivity/outputs/triage_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


# -----------------------------
# IO
# -----------------------------
def iter_examples(path: str) -> Iterable[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            yield obj
        else:
            raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")
    else:
        raise ValueError(f"Unsupported extension: {path}")


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# -----------------------------
# Gold normalization for urgency eval
# -----------------------------
def normalize_gold_decision(ex: Dict[str, Any]) -> Optional[str]:
    """
    Return gold decision as item_id or "tie".
    TRIAGE: gold_label may be int (source_idx) -> map to item_id
    ROLECONFLICT: gold_label often "tie" or item_id
    """
    gold = ex.get("gold_label", ex.get("gold_decision", None))
    if gold is None:
        return None

    if isinstance(gold, str):
        return gold

    # TRIAGE: gold_label might be source_idx (int)
    if isinstance(gold, (int, float)):
        bundle = ex.get("bundle", {})
        items = bundle.get("items", [])
        gold_int = int(gold)
        for it in items:
            if it.get("source_idx", None) == gold_int:
                return it.get("item_id")
        return None

    return None


# -----------------------------
# Stats
# -----------------------------
@dataclass
class EvalStats:
    n: int = 0
    correct: int = 0
    n_tie_gold: int = 0
    n_tie_pred: int = 0
    skipped: int = 0

    def acc(self) -> float:
        return 0.0 if self.n == 0 else self.correct / self.n


def update(stats: EvalStats, pred: str, gold: str) -> None:
    stats.n += 1
    if gold == "tie":
        stats.n_tie_gold += 1
    if pred == "tie":
        stats.n_tie_pred += 1
    if pred == gold:
        stats.correct += 1


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_file", type=str, required=True)
    ap.add_argument("--pred_file", type=str, required=True)
    args = ap.parse_args()

    # skeleton id -> (domain, gold_decision)
    gold_map: Dict[str, Tuple[str, Optional[str]]] = {}
    for ex in iter_examples(args.skeleton_file):
        ex_id = ex.get("id")
        if not ex_id:
            continue
        domain = ex.get("domain", "unknown")
        # only TRIAGE + ROLECONFLICT for this eval
        if domain not in ("life_safety", "role_conflict"):
            continue
        gold = normalize_gold_decision(ex)
        gold_map[ex_id] = (domain, gold)

    overall = EvalStats()
    by_domain: Dict[str, EvalStats] = {
        "life_safety": EvalStats(),
        "role_conflict": EvalStats(),
    }

    for rec in iter_jsonl(args.pred_file):
        ex_id = rec.get("id")
        if not ex_id or ex_id not in gold_map:
            overall.skipped += 1
            continue

        domain, gold = gold_map[ex_id]
        pred = rec.get("pred_decision", rec.get("pred_label", "tie"))

        if gold is None:
            overall.skipped += 1
            continue

        update(overall, pred, gold)
        update(by_domain[domain], pred, gold)

    print("[urgencyEval: overall]")
    print(f"  n={overall.n} correct={overall.correct} acc={overall.acc():.4f}")
    print(f"  tie_gold={overall.n_tie_gold} tie_pred={overall.n_tie_pred} skipped={overall.skipped}")

    for d, st in by_domain.items():
        print(f"\n[urgencyEval: {d}]")
        print(f"  n={st.n} correct={st.correct} acc={st.acc():.4f}")
        print(f"  tie_gold={st.n_tie_gold} tie_pred={st.n_tie_pred} skipped={st.skipped}")


if __name__ == "__main__":
    main()
