# scenario_extract.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def extract_scenario_text(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (scenario_text, source_field_name).
    Chooses the most appropriate scenario text field depending on dataset structure.
    """
    # 1) RoleConflict: bundle.shared_story
    shared_story = _get(example, "bundle.shared_story", None)
    if isinstance(shared_story, str) and shared_story.strip():
        return shared_story.strip(), "bundle.shared_story"

    # 2) UniBench: base.scenario
    base_scenario = _get(example, "base.scenario", None)
    if isinstance(base_scenario, str) and base_scenario.strip():
        return base_scenario.strip(), "base.scenario"

    # 3) GGB: base.vignette (fallback: base.statement)
    vignette = _get(example, "base.vignette", None)
    if isinstance(vignette, str) and vignette.strip():
        return vignette.strip(), "base.vignette"
    statement = _get(example, "base.statement", None)
    if isinstance(statement, str) and statement.strip():
        return statement.strip(), "base.statement"

    # 4) TRIAGE-like: bundle.items[*].situation_text
    items = _get(example, "bundle.items", None)
    if isinstance(items, list) and items:
        # If items have situation_text, stitch them into a single scenario
        parts: List[str] = []
        has_situation = False
        for it in items:
            if not isinstance(it, dict):
                continue
            st = it.get("situation_text")
            if isinstance(st, str) and st.strip():
                has_situation = True
                item_id = it.get("item_id", "item")
                parts.append(f"{item_id}: {st.strip()}")
        if has_situation:
            # Include resource info minimally (no new facts; just restating existing fields)
            rname = _get(example, "bundle.resource.name", None)
            rqty = _get(example, "bundle.resource.quantity", None)
            if isinstance(rname, str) and rname.strip() and isinstance(rqty, (int, float)):
                parts.append(f"Resource: {rname.strip()} (quantity={rqty})")
            return "\n".join(parts).strip(), "bundle.items[*].situation_text(+resource)"

    # Fallback: try to build from bundle.items[*].text if exists (e.g., some choice datasets)
    if isinstance(items, list) and items:
        parts = []
        for it in items:
            if not isinstance(it, dict):
                continue
            txt = it.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(f"{it.get('item_id','item')}: {txt.strip()}")
        if parts:
            return "\n".join(parts).strip(), "bundle.items[*].text"

    raise ValueError("Could not find a suitable scenario text field in this example.")
