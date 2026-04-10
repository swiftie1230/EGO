from __future__ import annotations
from typing import Dict, Any, List


def extract_options(example: Dict[str, Any]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []

    items = example.get("bundle", {}).get("items", None)
    if not isinstance(items, list):
        return options

    for it in items:
        if not isinstance(it, dict):
            continue

        raw_option_id = it.get("item_id")
        if raw_option_id is None:
            continue

        option_id = str(raw_option_id).strip()
        if not option_id:
            continue

        # ------------------------
        # 1) Standard text field
        # ------------------------
        text = it.get("text")
        if isinstance(text, str) and text.strip():
            options.append({
                "option_id": option_id,
                "text": text.strip(),
                "source_field": "bundle.items[*].text",
            })
            continue

        # ------------------------
        # 2) Medical / triage style
        # ------------------------
        situation_text = it.get("situation_text")
        if isinstance(situation_text, str) and situation_text.strip():
            options.append({
                "option_id": option_id,
                "text": situation_text.strip(),
                "source_field": "bundle.items[*].situation_text",
            })
            continue

        # ------------------------
        # 3) More generic fallbacks
        # ------------------------
        for field_name in ["option_text", "description", "statement"]:
            field_val = it.get(field_name)
            if isinstance(field_val, str) and field_val.strip():
                options.append({
                    "option_id": option_id,
                    "text": field_val.strip(),
                    "source_field": f"bundle.items[*].{field_name}",
                })
                break
        else:
            # ------------------------
            # 4) RoleConflict fallback
            # ------------------------
            role = it.get("role_name")
            situation = it.get("situation")
            expectation = it.get("expectation")

            if role and (situation or expectation):
                parts = []
                if situation:
                    parts.append(str(situation).strip())
                if expectation:
                    parts.append(str(expectation).strip())

                merged = " ".join(p for p in parts if p).strip()
                if merged:
                    options.append({
                        "option_id": option_id,
                        "text": merged,
                        "source_field": "bundle.items[*].(role fields)",
                    })

    return options
