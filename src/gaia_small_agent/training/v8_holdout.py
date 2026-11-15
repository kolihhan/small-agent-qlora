"""Frozen source-separated v8 diagnostic: last tool call + final per held-out parent.

Teacher-forced next actions are NOT end-to-end agent task success.
Only 35 whole parents excluded from the frozen 304 QLoRA training parents.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

_ALLOWED_SOURCES = frozenset({
    "musique-train", "hotpotqa-distractor-train", "2wikimultihopqa-train",
})


def select_v8_targets(rows: list[dict]) -> dict[str, dict[str, str]]:
    """Pre-register two oracle-prefix actions per held-out parent, no outcomes."""
    if len(rows) != 35:
        raise ValueError("v8 evaluation requires exactly 35 frozen held-out parents")
    selected: dict[str, dict[str, str]] = {}
    parents: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        source = str(row.get("source") or "")
        messages = row.get("messages")
        indices = row.get("training_action_indices")
        if not task_id or task_id in parents or source not in _ALLOWED_SOURCES:
            raise ValueError("duplicate, missing or forbidden whole-parent task")
        parents.add(task_id)
        if (not isinstance(messages, list) or not isinstance(indices, list)
                or len(indices) < 2 or indices != sorted(set(indices))):
            raise ValueError("invalid frozen oracle action indices")
        if any(type(i) is not int or i < 1 or i >= len(messages)
               or not isinstance(messages[i], dict)
               or messages[i].get("role") != "assistant" for i in indices):
            raise ValueError("invalid assistant action")
        final_index = indices[-1]
        final_message = messages[final_index]
        if final_message.get("tool_calls"):
            raise ValueError("frozen parent has no final answer")
        tool_candidates = [i for i in indices[:-1] if messages[i].get("tool_calls")]
        if not tool_candidates or len(tool_candidates) != len(indices)-1:
            raise ValueError("unsupported tool/final sequence")
        last_tool = tool_candidates[-1]
        for index, action_type in ((last_tool, "tool_call"), (final_index, "final")):
            ident = f"{task_id}:{index}"
            if ident in selected:
                raise ValueError("duplicate evaluation example")
            selected[ident] = {
                "task_id": task_id, "source": source, "action_type": action_type,
                "capability": str(row.get("capability") or "unspecified"),
            }
    if len(selected) != 70:
        raise ValueError("v8 holdout must have exactly 70 next-action targets")
    return selected


def _case_index(outcome: dict, selected: dict[str, dict[str, str]]) -> dict[str, bool]:
    items = outcome.get("examples")
    if not isinstance(items, list):
        raise ValueError("paired action report lacks examples")
    result: dict[str, bool] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid scored row")
        ident = str(item.get("example_id") or "")
        if ident in result:
            raise ValueError("duplicate scored example_id")
        if type(item.get("correct")) is not bool:
            raise ValueError("non-boolean strict next-action correctness")
        result[ident] = item["correct"]
    if result.keys() != selected.keys():
        raise ValueError("paired score coverage mismatch against frozen 70")
    return result


def paired_v8_actions(selected: dict[str, dict[str, str]], base: dict, adapter: dict) -> dict[str, Any]:
    """Count matched next-action transitions; never automatically promote."""
    if len(selected) != 70:
        raise ValueError("incomplete held-out target set")
    base_index = _case_index(base, selected)
    adapted = _case_index(adapter, selected)
    overall = Counter()
    by_action: dict[str, Counter] = {}
    by_source: dict[str, Counter] = {}
    cases: list[dict] = []

    for ident, meta in selected.items():
        old, new = base_index[ident], adapted[ident]
        transition = {
            "total": 1, "base_correct": int(old), "adapter_correct": int(new),
            "rescues": int(not old and new), "regressions": int(old and not new),
        }
        overall.update(transition)
        by_action.setdefault(meta["action_type"], Counter()).update(transition)
        by_source.setdefault(meta["source"], Counter()).update(transition)
        cases.append({
            "example_id": ident, **meta, "base_correct": old,
            "adapter_correct": new,
        })
    if overall["adapter_correct"] - overall["base_correct"] != overall["rescues"] - overall["regressions"]:
        raise AssertionError("paired transition score drift")
    return {
        **dict(overall),
        "per_action_type": {k: dict(v) for k, v in sorted(by_action.items())},
        "per_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "metric": "teacher-forced exact tool name+args / GAIA-style final answer only",
        "promotion_allowed": False,
        "caveat": "Not end-to-end, not GAIA, no evidence of autonomous multi-hop gains",
        "cases": cases,
    }
