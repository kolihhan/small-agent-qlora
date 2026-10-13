from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable

from ..agent.types import AssistantTurn
from ..benchmark.scoring import gaia_score


def _canonical_arguments(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expected_tool_calls(message: dict) -> list[tuple[str, str]]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"__raw__": arguments}
        if not isinstance(arguments, dict):
            arguments = {"__raw__": arguments}
        calls.append((name, _canonical_arguments(arguments)))
    return calls


def _predicted_tool_calls(turn: AssistantTurn) -> list[tuple[str, str]]:
    return [(call.name, _canonical_arguments(call.arguments)) for call in turn.tool_calls]


def _score_turn(expected: dict, predicted: AssistantTurn) -> tuple[bool, str | None]:
    expected_calls = _expected_tool_calls(expected)
    predicted_calls = _predicted_tool_calls(predicted)
    expected_is_tool = bool(expected_calls)
    predicted_is_tool = bool(predicted_calls)
    if expected_is_tool != predicted_is_tool:
        return False, "wrong_action_type"
    if expected_is_tool:
        if [name for name, _ in expected_calls] != [name for name, _ in predicted_calls]:
            return False, "wrong_tool"
        if expected_calls != predicted_calls:
            return False, "wrong_arguments"
        return True, None
    if gaia_score(predicted.content, str(expected.get("content") or "")):
        return True, None
    return False, "wrong_final"


def evaluate_next_actions(
    rows_path: str | Path,
    model_factory: Callable[[], object],
    *,
    example_ids: set[str] | None = None,
) -> dict:
    """Teacher-force held-out oracle prefixes and score each immediate next action."""
    rows = [json.loads(line) for line in Path(rows_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No next-action evaluation rows")

    selected_ids = set(example_ids) if example_ids is not None else None
    total = correct = 0
    by_capability: dict[str, dict[str, int]] = {}
    by_action_type: dict[str, dict[str, int]] = {}
    error_counts: Counter[str] = Counter()
    examples: list[dict] = []

    for row_index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"line {row_index}: missing messages")
        tools = row.get("tools") or []
        capability = str(row.get("capability") or "unspecified")
        task_id = str(row.get("task_id") or f"row-{row_index}")
        assistant_targets = 0

        for message_index, expected in enumerate(messages):
            if not isinstance(expected, dict) or expected.get("role") != "assistant":
                continue
            prefix = messages[:message_index]
            if not prefix:
                continue
            assistant_targets += 1
            example_id = f"{task_id}:{message_index}"
            if selected_ids is not None and example_id not in selected_ids:
                continue

            predicted = model_factory().complete(prefix, tools)
            is_correct, error = _score_turn(expected, predicted)
            action_type = "tool_call" if expected.get("tool_calls") else "final"
            total += 1
            correct += int(is_correct)
            if error:
                error_counts[error] += 1
            for buckets, key in ((by_capability, capability), (by_action_type, action_type)):
                bucket = buckets.setdefault(key, {"total": 0, "correct": 0})
                bucket["total"] += 1
                bucket["correct"] += int(is_correct)
            examples.append({
                "example_id": example_id,
                "task_id": task_id,
                "capability": capability,
                "action_type": action_type,
                "correct": is_correct,
                "error": error,
                "expected_tool_calls": [name for name, _ in _expected_tool_calls(expected)],
                "predicted_tool_calls": [call.name for call in predicted.tool_calls],
            })

        if assistant_targets == 0:
            raise ValueError(f"line {row_index}: no assistant target")

    if selected_ids is not None:
        observed = {item["example_id"] for item in examples}
        missing = selected_ids - observed
        if missing:
            raise ValueError(f"selected example_ids not found: {sorted(missing)}")
    if total == 0:
        raise ValueError("No next-action examples selected")

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct * 100.0 / total,
        "error_counts": dict(error_counts),
        "by_capability": by_capability,
        "by_action_type": by_action_type,
        "examples": examples,
    }


__all__ = ["evaluate_next_actions"]
