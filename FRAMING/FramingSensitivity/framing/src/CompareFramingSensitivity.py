#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (row.get("id"), row.get("option_order", "original"))


def load_index(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out = {}
    for r in iter_jsonl(path):
        if "id" not in r:
            continue
        out[make_key(r)] = r
    return out


def extract_prob(row: Dict[str, Any], cand_fields: List[str]) -> Optional[Dict[str, float]]:
    for f in cand_fields:
        d = row.get(f)
        if isinstance(d, dict) and d:
            return {k: float(v) for k, v in d.items()}
    return None


def normalize_keys(decision_space: List[str], p: Dict[str, float], q: Dict[str, float]) -> List[str]:
    keys = list(decision_space)
    if ("tie" in p) or ("tie" in q):
        keys.append("tie")
    for k in list(p.keys()) + list(q.keys()):
        if k not in keys:
            keys.append(k)
    return keys


def l1_distance(p: Dict[str, float], q: Dict[str, float], keys: List[str]) -> float:
    return sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)


def compute_counter_metrics(
    base_idx: Dict[Tuple[str, str], Dict[str, Any]],
    counter_idx: Dict[Tuple[str, str], Dict[str, Any]],
    ids: List[Tuple[str, str]],
    thresh: float,
):
    """
    Metrics on a fixed key subset:
      flip rate and 4-quadrants based on:
        flip 여부 = base_pred_decision vs counter_pred_decision
        prob-change = L1( base.confidence_cond vs counter.counter_confidence_cond )
    """
    c = Counter()
    for rid in ids:
        b = base_idx.get(rid)
        r = counter_idx.get(rid)
        if b is None or r is None:
            continue

        decision_space = r.get("decision_space") or b.get("decision_space") or b.get("label_space") or []
        if not isinstance(decision_space, list):
            decision_space = []

        base_dec = b.get("pred_decision")
        counter_dec = r.get("counter_pred_decision")

        p_base = extract_prob(b, ["confidence_cond", "confidence_avg"]) or {}
        p_cnt = extract_prob(r, ["counter_confidence_cond", "counter_confidence_avg"]) or {}

        keys = normalize_keys(decision_space, p_base, p_cnt)
        d = l1_distance(p_base, p_cnt, keys)

        flip = (base_dec is not None) and (counter_dec is not None) and (base_dec != counter_dec)
        high = d >= thresh

        if flip and high:
            quad = "flip_high"
        elif flip and (not high):
            quad = "flip_low"
        elif (not flip) and high:
            quad = "noflip_high"
        else:
            quad = "noflip_low"

        c["N"] += 1
        c["flip"] += int(flip)
        c[quad] += 1
        c["dist_sum"] += d

    return c


def print_metrics(name: str, c: Counter, thresh: float):
    n = c.get("N", 0)
    if n == 0:
        print(f"\n[{name}] N=0")
        return
    flip = c.get("flip", 0)
    avg_d = c.get("dist_sum", 0.0) / max(n, 1)

    print(f"\n[{name}] (anchor-filtered)")
    print(f"- N: {n}")
    print(f"- flip rate: {flip/n:.4f} ({flip}/{n})")
    print(f"- threshold(high): {thresh}")
    print(f"- avg L1 distance: {avg_d:.6f}")
    for k in ["flip_high", "flip_low", "noflip_high", "noflip_low"]:
        v = c.get(k, 0)
        print(f"  - {k}: {v} ({v/n:.4f})")


def split_keys_by_option_order(keys: List[Tuple[str, str]]):
    all_keys = sorted(keys)
    original_keys = [k for k in all_keys if k[1] == "original"]
    swapped_keys = [k for k in all_keys if k[1] == "swapped"]
    return all_keys, original_keys, swapped_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", required=True, help="base preds jsonl (pred_decision + confidence_cond/avg)")
    ap.add_argument("--anchor_counter", required=True, help="value-tinted counter jsonl (ID anchor, e.g., only 51 rows)")
    ap.add_argument("--other_counters", nargs="+", required=True, help="other counter jsonls to compare (e.g., paraphrase)")
    ap.add_argument("--names", nargs="*", default=None, help="names aligned with other_counters; optional")
    ap.add_argument("--thresh", type=float, default=0.3, help="high/low threshold on L1 distance")
    args = ap.parse_args()

    base_idx = load_index(args.base_jsonl)

    anchor_idx = load_index(args.anchor_counter)
    anchor_keys = sorted(anchor_idx.keys())

    anchor_all, anchor_original, anchor_swapped = split_keys_by_option_order(anchor_keys)

    print(f"[ANCHOR] rows={len(anchor_all)} from: {args.anchor_counter}")
    print(f"         original={len(anchor_original)}, swapped={len(anchor_swapped)}")

    vt_metrics_all = compute_counter_metrics(base_idx, anchor_idx, anchor_all, args.thresh)
    print_metrics("framing (all)", vt_metrics_all, args.thresh)

    vt_metrics_orig = compute_counter_metrics(base_idx, anchor_idx, anchor_original, args.thresh)
    print_metrics("framing (original)", vt_metrics_orig, args.thresh)

    vt_metrics_swap = compute_counter_metrics(base_idx, anchor_idx, anchor_swapped, args.thresh)
    print_metrics("framing (swapped)", vt_metrics_swap, args.thresh)

    if args.names is not None and len(args.names) not in (0, len(args.other_counters)):
        raise ValueError("--names must match --other_counters length (or omit)")

    names = args.names if args.names and len(args.names) == len(args.other_counters) else [
        f"cond_{i}" for i in range(len(args.other_counters))
    ]

    for name, path in zip(names, args.other_counters):
        idx = load_index(path)

        eval_all = [rid for rid in anchor_all if (rid in base_idx) and (rid in idx)]
        eval_original = [rid for rid in anchor_original if (rid in base_idx) and (rid in idx)]
        eval_swapped = [rid for rid in anchor_swapped if (rid in base_idx) and (rid in idx)]

        print(f"\n[COVERAGE] {name}: "
              f"all={len(eval_all)}/{len(anchor_all)}, "
              f"original={len(eval_original)}/{len(anchor_original)}, "
              f"swapped={len(eval_swapped)}/{len(anchor_swapped)}")

        framing_met_all = compute_counter_metrics(base_idx, anchor_idx, eval_all, args.thresh)
        print_metrics(f"framing@{name} (all)", framing_met_all, args.thresh)

        framing_met_orig = compute_counter_metrics(base_idx, anchor_idx, eval_original, args.thresh)
        print_metrics(f"framing@{name} (original)", framing_met_orig, args.thresh)

        framing_met_swap = compute_counter_metrics(base_idx, anchor_idx, eval_swapped, args.thresh)
        print_metrics(f"framing@{name} (swapped)", framing_met_swap, args.thresh)

        met_all = compute_counter_metrics(base_idx, idx, eval_all, args.thresh)
        print_metrics(f"{name} (all)", met_all, args.thresh)

        met_orig = compute_counter_metrics(base_idx, idx, eval_original, args.thresh)
        print_metrics(f"{name} (original)", met_orig, args.thresh)

        met_swap = compute_counter_metrics(base_idx, idx, eval_swapped, args.thresh)
        print_metrics(f"{name} (swapped)", met_swap, args.thresh)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()