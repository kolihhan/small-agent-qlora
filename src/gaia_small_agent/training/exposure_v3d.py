from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


# Small corrective continuation from v3B.  Unlike v3B, the selector keys on
# trajectory state, not only capability + action type.  In particular it never
# supervises the deliberately failing index-2 search as a recovery/switch action.
# The 24-turn mix stays exactly balanced between tool calls and finals so the
# corrective run cannot repeat v3C's tool-heavy 74/54 exposure shift.
V3D_STATE_QUOTAS: dict[tuple[str, str, int], int] = {
    ("evidence_to_final", "final", 4): 4,
    ("multi_step", "tool_call", 4): 2,
    ("multi_step", "final", 6): 3,
    ("failure_recovery", "tool_call", 4): 3,
    ("failure_recovery", "final", 6): 3,
    ("strategy_switch", "tool_call", 4): 2,
    ("strategy_switch", "final", 6): 1,
    ("argument_grounding", "tool_call", 2): 3,
    ("tool_selection", "tool_call", 2): 1,
    ("duplicate_avoidance", "tool_call", 2): 1,
    ("no_tool_stop", "final", 2): 1,
}


def _stable_key(seed: str, namespace: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{example_id}".encode("utf-8")).hexdigest()


def _metadata(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("v3D training example is missing trace metadata")
    task_id = str(metadata.get("task_id") or "")
    example_id = str(metadata.get("example_id") or "")
    capability = str(metadata.get("capability") or "")
    action_type = str(metadata.get("action_type") or "")
    message_index = metadata.get("assistant_message_index")
    if (
        not task_id
        or not example_id
        or not capability
        or action_type not in {"tool_call", "final"}
        or not isinstance(message_index, int)
    ):
        raise ValueError("v3D training example has incomplete trace metadata")
    return metadata


def select_v3d_corrective_examples(
    examples: list[dict[str, Any]],
    *,
    seed: str = "qlora-v3d-corrective",
    exclude_task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select state-aware corrective decisions for v3B continuation.

    Selection remains task-unique: no complete trajectory is replayed and no
    task contributes more than one optimizer target.  The state quotas encode
    the held-out diagnosis directly: post-error recovery/switch actions use
    assistant index 4, deep finalization uses index 6, and initial index-2
    failure actions are never treated as recovery supervision.
    """
    excluded = {str(task_id) for task_id in (exclude_task_ids or set())}
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {
        state: [] for state in V3D_STATE_QUOTAS
    }
    for example in examples:
        metadata = _metadata(example)
        state = (
            str(metadata["capability"]),
            str(metadata["action_type"]),
            int(metadata["assistant_message_index"]),
        )
        if state in buckets and str(metadata["task_id"]) not in excluded:
            buckets[state].append(example)

    selected: list[dict[str, Any]] = []
    selected_tasks: set[str] = set()
    for (capability, action_type, message_index), quota in V3D_STATE_QUOTAS.items():
        namespace = f"{capability}:{action_type}:index{message_index}"
        candidates = sorted(
            buckets[(capability, action_type, message_index)],
            key=lambda item: _stable_key(
                seed,
                namespace,
                str(_metadata(item)["example_id"]),
            ),
        )
        chosen = 0
        for item in candidates:
            task_id = str(_metadata(item)["task_id"])
            if task_id in selected_tasks:
                continue
            selected.append(item)
            selected_tasks.add(task_id)
            chosen += 1
            if chosen == quota:
                break
        if chosen != quota:
            raise ValueError(
                f"not enough unique {capability}/{action_type}/index{message_index} "
                f"examples for v3D quota {quota}: found {chosen}"
            )

    expected = sum(V3D_STATE_QUOTAS.values())
    if len(selected) != expected:
        raise RuntimeError(f"v3D selected {len(selected)} examples, expected {expected}")
    selected.sort(
        key=lambda item: _stable_key(seed, "training-order", str(_metadata(item)["example_id"]))
    )
    return selected


def plan_v3d_batches(
    selected: list[dict[str, Any]],
    *,
    batch_size: int = 6,
) -> list[list[dict[str, Any]]]:
    """Slice the frozen v3D training order into resumable sequential batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [selected[start : start + batch_size] for start in range(0, len(selected), batch_size)]


def summarize_v3d_selection(selected: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [dict(_metadata(item)) for item in selected]
    example_ids = [str(item["example_id"]) for item in metadata]
    task_ids = [str(item["task_id"]) for item in metadata]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("v3D selection contains duplicate task IDs")

    action_counts = Counter(str(item["action_type"]) for item in metadata)
    capability_counts = Counter(str(item["capability"]) for item in metadata)
    state_counts = Counter(
        (
            str(item["capability"]),
            str(item["action_type"]),
            int(item["assistant_message_index"]),
        )
        for item in metadata
    )
    state_report = {
        f"{capability}/{action_type}/index{message_index}": count
        for (capability, action_type, message_index), count in sorted(state_counts.items())
    }
    payload = json.dumps(example_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "small-agent-v3d-corrective-selection/v1",
        "selected_example_count": len(metadata),
        "selection_sha256": hashlib.sha256(payload).hexdigest(),
        "action_type_counts": dict(sorted(action_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "state_counts": state_report,
        "examples": metadata,
    }


__all__ = [
    "V3D_STATE_QUOTAS",
    "plan_v3d_batches",
    "select_v3d_corrective_examples",
    "summarize_v3d_selection",
]
