from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


# 54 complete source trajectories -> 128 assistant decisions.
# The mix emphasizes the failures observed in the frozen v3B held-out run while
# keeping the overall tool/final balance close to the 3,600-turn source corpus.
V3C_TRAJECTORY_QUOTAS: dict[str, int] = {
    "evidence_to_final": 10,
    "multi_step": 8,
    "failure_recovery": 12,
    "strategy_switch": 6,
    "argument_grounding": 6,
    "tool_selection": 4,
    "duplicate_avoidance": 2,
    "no_tool_stop": 6,
}


def _stable_task_key(seed: str, capability: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{capability}\0{task_id}".encode("utf-8")).hexdigest()


def select_v3c_trajectories(
    rows: list[dict[str, Any]],
    *,
    seed: str = "qlora-v3c-full-trajectory",
) -> list[dict[str, Any]]:
    """Select complete verified trajectories for the v3C exposure treatment.

    Selection happens at task/trajectory level. Every assistant decision from a
    selected task is kept, preserving failure -> recovery -> evidence -> final
    state transitions that v3B's one-decision-per-task probe intentionally broke.
    """
    by_capability: dict[str, list[dict[str, Any]]] = {
        capability: [] for capability in V3C_TRAJECTORY_QUOTAS
    }
    seen_task_ids: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        capability = str(row.get("capability") or "")
        messages = row.get("messages")
        if not task_id or not capability or not isinstance(messages, list):
            raise ValueError("v3C source row has incomplete task metadata")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate source task_id: {task_id}")
        seen_task_ids.add(task_id)
        if capability in by_capability:
            by_capability[capability].append(row)

    selected: list[dict[str, Any]] = []
    for capability, quota in V3C_TRAJECTORY_QUOTAS.items():
        candidates = sorted(
            by_capability[capability],
            key=lambda row: _stable_task_key(seed, capability, str(row["task_id"])),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"not enough {capability} trajectories for v3C quota {quota}: found {len(candidates)}"
            )
        selected.extend(candidates[:quota])

    selected.sort(
        key=lambda row: _stable_task_key(seed, "training-order", str(row["task_id"]))
    )
    return selected


def summarize_v3c_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids: list[str] = []
    capability_task_counts: Counter[str] = Counter()
    capability_example_counts: Counter[str] = Counter()
    action_type_counts: Counter[str] = Counter()

    for row in rows:
        task_id = str(row.get("task_id") or "")
        capability = str(row.get("capability") or "")
        messages = row.get("messages")
        if not task_id or not capability or not isinstance(messages, list):
            raise ValueError("v3C selected row has incomplete task metadata")
        task_ids.append(task_id)
        capability_task_counts[capability] += 1
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action_type = "tool_call" if message.get("tool_calls") else "final"
            action_type_counts[action_type] += 1
            capability_example_counts[capability] += 1

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("v3C selection contains duplicate task IDs")

    selection_bytes = json.dumps(task_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "small-agent-v3c-selection/v1",
        "selected_task_count": len(task_ids),
        "selected_example_count": sum(action_type_counts.values()),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "task_ids": task_ids,
        "capability_task_counts": dict(sorted(capability_task_counts.items())),
        "capability_example_counts": dict(sorted(capability_example_counts.items())),
        "action_type_counts": dict(sorted(action_type_counts.items())),
    }


__all__ = [
    "V3C_TRAJECTORY_QUOTAS",
    "select_v3c_trajectories",
    "summarize_v3c_selection",
]
