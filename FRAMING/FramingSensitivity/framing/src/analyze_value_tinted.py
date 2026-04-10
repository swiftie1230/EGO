# analyze_value_tinted.py

import os
import json
from glob import glob
import statistics

# ===============================
# CONFIG
# ===============================
ROOT = "/home/swiftie1230/EGO/FRAMING/FramingSensitivity/framing/outputs/contextual_envelope_framing/value_tinted_narration"


# ===============================
# FIND ALL JSONL FILES
# ===============================
files = glob(os.path.join(ROOT, "**/*.jsonl"), recursive=True)

print("\n[FOUND FILES]")
for f in files:
    print(" -", f)

print("\n===== Framing Sensitivity (Confidence-aware) =====\n")


# ===============================
# ANALYSIS
# ===============================
for fp in files:

    total = 0
    flip = 0
    flip_A_to_B = 0
    flip_B_to_A = 0

    margins = []
    abs_margins = []
    chosen_confidences = []
    ratios = []

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            base = row.get("base_pred_decision")
            counter = row.get("counter_pred_decision")
            conf = row.get("counter_confidence")
            margin = row.get("confidence_margin")

            # only binary A/B analysis
            if base not in ["A","B"] or counter not in ["A","B"]:
                continue

            total += 1

            # ----------------------
            # flip
            # ----------------------
            if base != counter:
                flip += 1

                if base == "A" and counter == "B":
                    flip_A_to_B += 1
                elif base == "B" and counter == "A":
                    flip_B_to_A += 1

            # ----------------------
            # margin stats
            # ----------------------
            if isinstance(margin, (int,float)):
                margins.append(margin)
                abs_margins.append(abs(margin))

            # ----------------------
            # confidence stats
            # ----------------------
            if isinstance(conf, dict):

                chosen_conf = conf.get(counter)
                if isinstance(chosen_conf, (int,float)):
                    chosen_confidences.append(chosen_conf)

                a = conf.get("A", 0.0)
                b = conf.get("B", 0.0)

                if a > 0 and b > 0:
                    ratios.append(max(a,b) / min(a,b))

    # ===============================
    # AGGREGATION
    # ===============================
    flip_rate = flip / total if total > 0 else 0

    avg_margin = statistics.mean(margins) if margins else 0
    avg_abs_margin = statistics.mean(abs_margins) if abs_margins else 0
    avg_conf = statistics.mean(chosen_confidences) if chosen_confidences else 0
    avg_ratio = statistics.mean(ratios) if ratios else 0

    # domain 자동 추출
    parts = fp.split(os.sep)
    domain = parts[-3] if len(parts) >= 3 else "unknown"

    result = {
        "file": os.path.basename(fp),
        "domain": domain,
        "total": total,
        "flip": flip,
        "flip_rate": round(flip_rate,4),
        "A_to_B": flip_A_to_B,
        "B_to_A": flip_B_to_A,
        "avg_margin": round(avg_margin,8),
        "avg_abs_margin": round(avg_abs_margin,8),
        "avg_chosen_confidence": round(avg_conf,8),
        "avg_conf_ratio": round(avg_ratio,2),
    }

    print(result)

print("\n===== Sensitivity DONE =====")
