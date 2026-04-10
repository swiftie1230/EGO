# paraphrase_skeleton_bt.py
from __future__ import annotations

import os
import json
import argparse
from copy import deepcopy
from typing import Any, Dict, List, Tuple

# -----------------------------
# IO utils
# -----------------------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_get(d: Dict[str, Any], path: List[Any], default=None):
    cur: Any = d
    for k in path:
        if isinstance(k, int):
            if not isinstance(cur, list) or k >= len(cur):
                return default
            cur = cur[k]
        else:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
    return cur

def safe_set(d: Dict[str, Any], path: List[Any], value: Any) -> bool:
    cur: Any = d
    for i, k in enumerate(path):
        last = (i == len(path) - 1)
        if last:
            if isinstance(k, int):
                if not isinstance(cur, list) or k >= len(cur):
                    return False
                cur[k] = value
                return True
            else:
                if not isinstance(cur, dict):
                    return False
                cur[k] = value
                return True

        # walk
        nxt = path[i + 1]
        if isinstance(k, int):
            if not isinstance(cur, list) or k >= len(cur):
                return False
            cur = cur[k]
        else:
            if not isinstance(cur, dict):
                return False
            if k not in cur:
                # create container
                cur[k] = [] if isinstance(nxt, int) else {}
            cur = cur[k]
    return False

# -----------------------------
# Back-translation paraphraser (HF MarianMT)
# -----------------------------
class BackTranslator:
    """
    English paraphrase baseline via back-translation:
      en -> de -> en  (default)
    """
    def __init__(self, device: str = "cpu",
                 src2pivot_name: str = "Helsinki-NLP/opus-mt-en-de",
                 pivot2src_name: str = "Helsinki-NLP/opus-mt-de-en"):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.torch = torch
        self.device = torch.device(device)

        self.src2pivot_tok = MarianTokenizer.from_pretrained(src2pivot_name)
        self.src2pivot = MarianMTModel.from_pretrained(src2pivot_name).to(self.device)

        self.pivot2src_tok = MarianTokenizer.from_pretrained(pivot2src_name)
        self.pivot2src = MarianMTModel.from_pretrained(pivot2src_name).to(self.device)

    @staticmethod
    def _chunk(lst: List[str], n: int):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    def _generate(self, texts: List[str], tok, model,
                  max_new_tokens: int,
                  do_sample: bool,
                  top_p: float,
                  temperature: float,
                  num_return_sequences: int,
                  batch_size: int = 16) -> List[List[str]]:
        """
        Returns: list per input, each is list[str] of size num_return_sequences
        """
        out_all: List[List[str]] = []
        for batch in self._chunk(texts, batch_size):
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                # if do_sample=False, you may want num_beams; but we use sampling for diversity baseline
            )
            dec = tok.batch_decode(gen, skip_special_tokens=True)

            # group by input
            # HF returns sequences in order: for each input, num_return_sequences outputs (usually)
            grouped: List[List[str]] = [[] for _ in batch]
            for i, s in enumerate(dec):
                grouped[i // num_return_sequences].append(s)

            out_all.extend(grouped)
        return out_all

    def paraphrase(self,
                   texts: List[str],
                   n: int = 3,
                   max_new_tokens: int = 256,
                   top_p: float = 0.92,
                   temperature: float = 0.9,
                   batch_size: int = 16) -> List[List[str]]:
        """
        For each input text, return n paraphrases via BT sampling.
        """
        # step1: en -> de (generate n candidates)
        pivots = self._generate(
            texts, self.src2pivot_tok, self.src2pivot,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=n,
            batch_size=batch_size
        )

        # step2: de -> en (back translate each pivot)
        # flatten
        flat_pivot: List[str] = []
        owners: List[int] = []
        for i, cand_list in enumerate(pivots):
            for c in cand_list:
                flat_pivot.append(c)
                owners.append(i)

        back = self._generate(
            flat_pivot, self.pivot2src_tok, self.pivot2src,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            batch_size=batch_size
        )
        # back is list[list[str]] size = len(flat_pivot), each inner size 1
        flat_back = [x[0] for x in back]

        # regroup to per original text
        results: List[List[str]] = [[] for _ in texts]
        for ob, s in zip(owners, flat_back):
            results[ob].append(s)

        return results


# -----------------------------
# Field specs per dataset schema
# -----------------------------
def list_text_paths(row: Dict[str, Any]) -> List[Tuple[str, str, List[Any]]]:
    """
    Return (group, field_name, path_list) for text fields to paraphrase.
    groups: item_text | story_text | decision_text
    """
    dataset = row.get("dataset", "")
    domain = row.get("domain", "")

    paths: List[Tuple[str, str, List[Any]]] = []

    # ROLECONFLICT
    if domain == "role_conflict" or "RoleConflict" in dataset:
        # story_text
        paths.append(("story_text", "bundle.shared_story", ["bundle", "shared_story"]))
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text: situation only (exclude expectation!)
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].situation", ["bundle", "items", i, "situation"]))
        # NOTE: expectation intentionally excluded

    # SUPER_SCOTUS
    elif domain == "legal_decision" or dataset == "SUPER_SCOTUS":
        # story_text
        paths.append(("story_text", "base.scenario", ["base", "scenario"]))
        # legal_issue is optional / sometimes empty
        #paths.append(("story_text", "base.legal_issue", ["base", "legal_issue"]))
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].text", ["bundle", "items", i, "text"]))

    # MedicalTriageAlignment
    elif dataset == "MedicalTriageAlignment":
        # story_text
        paths.append(("story_text", "base.scenario", ["base", "scenario"]))
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].situation_text", ["bundle", "items", i, "situation_text"]))

    # TRIAGE
    elif domain == "life_safety" or dataset == "TRIAGE":
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].situation_text", ["bundle", "items", i, "situation_text"]))

    # MAEBE_GGB
    elif domain == "moral_dilemma" or dataset == "GGB":
        # story_text (base texts)
        paths.append(("story_text", "base.statement", ["base", "statement"]))
        paths.append(("story_text", "base.vignette", ["base", "vignette"]))
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].text", ["bundle", "items", i, "text"]))

    # UNIBENCH
    elif domain == "decision_choice" or "UniBench" in dataset:
        # story_text
        paths.append(("story_text", "base.scenario", ["base", "scenario"]))
        # decision_text
        paths.append(("decision_text", "decision_question", ["decision_question"]))
        # item_text
        items = safe_get(row, ["bundle", "items"], default=[])
        if isinstance(items, list):
            for i in range(len(items)):
                paths.append(("item_text", f"bundle.items[{i}].text", ["bundle", "items", i, "text"]))

    else:
        # fallback (best-effort)
        for group, name, p in [
            ("decision_text", "decision_question", ["decision_question"]),
            ("story_text", "base.scenario", ["base", "scenario"]),
            ("story_text", "bundle.shared_story", ["bundle", "shared_story"]),
        ]:
            if isinstance(safe_get(row, p), str):
                paths.append((group, name, p))

    # Keep only existing str fields + dedup
    filtered: List[Tuple[str, str, List[Any]]] = []
    seen = set()
    for group, name, p in paths:
        key = (group, name, tuple(p))
        if key in seen:
            continue
        seen.add(key)
        val = safe_get(row, p)
        if isinstance(val, str) and val.strip():
            filtered.append((group, name, p))
    return filtered


def attach_mode(row: Dict[str, Any],
                bt: BackTranslator,
                n: int,
                max_new_tokens: int,
                top_p: float,
                temperature: float) -> Dict[str, Any]:
    """
    Keep original row, add row["paraphrases_bt"] = { field_name: [p1..pn], ... }
    """
    out = deepcopy(row)
    field_paths = list_text_paths(row)
    texts = [safe_get(row, p) for _, _, p in field_paths]

    paras = bt.paraphrase(
        texts, n=n, max_new_tokens=max_new_tokens,
        top_p=top_p, temperature=temperature
    )

    out["paraphrases_bt"] = {
        "method": "back_translation",
        "pivot": "de",
        "n": n,
        "decode": {
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        },
        "fields": {
            field_name: paras[i]
            for i, (_, field_name, _) in enumerate(field_paths)
        }
    }
    return out


def _sanitize_field_for_id(s: str) -> str:
    return (
        s.replace(".", "_")
         .replace("[", "")
         .replace("]", "")
         .replace("/", "_")
         .replace(" ", "_")
    )

def expand_mode(row: Dict[str, Any],
                bt: BackTranslator,
                n: int,
                max_new_tokens: int,
                top_p: float,
                temperature: float) -> List[Dict[str, Any]]:
    """
    Fieldwise expand:
      - item_text only variants (one item field changed at a time)
      - story_text only variants (one story/base field changed at a time)
      - decision_text only variants (decision_question changed)
    Adds meta.paraphrase_group + meta.paraphrase_field so you can filter later.
    """
    field_specs = list_text_paths(row)  # (group, field_name, path)
    variants: List[Dict[str, Any]] = []
    base_id = row.get("id", "unknown_id")

    for group, field_name, field_path in field_specs:
        src_text = safe_get(row, field_path)
        if not isinstance(src_text, str) or not src_text.strip():
            continue

        # paraphrase ONLY this field
        paras = bt.paraphrase(
            [src_text],
            n=n,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature
        )[0]  # List[str] length n

        field_tag = _sanitize_field_for_id(field_name)

        for k in range(n):
            r = deepcopy(row)
            safe_set(r, field_path, paras[k])

            r["meta"] = dict(r.get("meta", {}))
            r["meta"].update({
                "paraphrase_of": base_id,
                "paraphrase_group": group,          # item_text | story_text | decision_text
                "paraphrase_field": field_name,     # exact field name
                "paraphrase_idx": k,
                "paraphrase_method": "back_translation_en_de_en",
                "paraphrase_decode": {
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens
                },
            })

            r["id"] = f"{base_id}_para_bt_{group}_{field_tag}_{k:03d}"
            variants.append(r)

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str,
                    default="/EGO/FRAMING/FramingSensitivity/skeleton/data",
                    help="Directory containing skeleton jsonl files (see screenshot path).")
    ap.add_argument("--output_dir", type=str,
                    default="/EGO/FRAMING/FramingSensitivity/skeleton/data_paraphrased",
                    help="Where to write outputs.")
    ap.add_argument("--files", type=str, nargs="*",
                    default=[
                        "roleconflict_allocation.jsonl",
                        "triage_allocation.jsonl",
                        "ggb_skeleton.jsonl",
                        "unibench_skeleton.jsonl",
                        "SCOTUS_skeleton.jsonl",
                        "medical_triage_alignment_skeleton.jsonl",
                    ])
    ap.add_argument("--mode", type=str, choices=["attach", "expand"], default="expand")
    ap.add_argument("--n", type=int, default=3, help="Number of paraphrases per example.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0,
                help="If >0, process only first N rows per file.")

    args = ap.parse_args()

    bt = BackTranslator(device=args.device)

    for fn in args.files:
        in_path = os.path.join(args.input_dir, fn)
        if not os.path.exists(in_path):
            print(f"[skip] not found: {in_path}")
            continue

        out_rows: List[Dict[str, Any]] = []
        seen_in_file = 0

        for row in iter_jsonl(in_path):

            if args.limit > 0 and seen_in_file >= args.limit:
                break

            if args.mode == "attach":
                out_rows.append(
                    attach_mode(row, bt, args.n, args.max_new_tokens, args.top_p, args.temperature)
                )
            else:
                out_rows.extend(
                    expand_mode(row, bt, args.n, args.max_new_tokens, args.top_p, args.temperature)
                )

            seen_in_file += 1

        out_path = os.path.join(args.output_dir, fn.replace(".jsonl", f".{args.mode}.jsonl"))
        write_jsonl(out_path, out_rows)
        print(f"[ok] wrote: {out_path}  (rows={len(out_rows)})")


if __name__ == "__main__":
    main()