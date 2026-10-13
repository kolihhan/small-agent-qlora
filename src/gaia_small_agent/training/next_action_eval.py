from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from ..agent.types import AssistantTurn
from ..benchmark.scoring import gaia_score


def _decode_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("tool-call arguments must be an object")


def _expected_tool_calls(message: dict) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("assistant tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("assistant tool call must be an object")
        function = call.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError("assistant tool call function must be an object")
        name = str(function.get("name") or "")
        if not name:
            raise ValueError("assistant tool call must name a function")
        normalized.append(
            {
                "name": name,
                "arguments": _decode_arguments(function.get("arguments") or {}),
            }
        )
    return normalized


def _predicted_tool_calls(turn: AssistantTurn) -> list[dict[str, Any]]:
    return [
        {"name": str(call.name), "arguments": dict(call.arguments)}
        for call in turn.tool_calls
    ]


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _summary(decisions: list[dict]) -> dict:
    total = len(decisions)
    exact = sum(bool(row["exact"]) for row in decisions)
    action_type_correct = sum(bool(row["action_type_correct"]) for row in decisions)
    tool_rows = [row for row in decisions if row["expected_action_type"] == "tool_call"]
    final_rows = [row for row in decisions if row["expected_action_type"] == "final"]
    tool_name_correct = sum(bool(row["tool_name_correct"]) for row in tool_rows)
    tool_arguments_correct = sum(bool(row["tool_arguments_correct"]) for row in tool_rows)
    final_correct = sum(bool(row["final_correct"]) for row in final_rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in decisions:
        grouped[str(row["capability"])].append(row)

    by_capability = {}
    for capability, rows in sorted(grouped.items()):
        cap_total = len(rows)
        cap_exact = sum(bool(row["exact"]) for row in rows)
        cap_action = sum(bool(row["action_type_correct"]) for row in rows)
        by_capability[capability] = {
            "total_decisions": cap_total,
            "exact_decisions": cap_exact,
            "exact_accuracy": _safe_rate(cap_exact, cap_total),
            "action_type_correct": cap_action,
            "action_type_accuracy": _safe_rate(cap_action, cap_total),
        }

    return {
        "schema_version": "small-agent-next-action-summary/v1",
        "total_decisions": total,
        "exact_decisions": exact,
        "exact_accuracy": _safe_rate(exact, total),
        "action_type_correct": action_type_correct,
        "action_type_accuracy": _safe_rate(action_type_correct, total),
        "tool_decisions": len(tool_rows),
        "tool_name_correct": tool_name_correct,
        "tool_name_accuracy": _safe_rate(tool_name_correct, len(tool_rows)),
        "tool_arguments_correct": tool_arguments_correct,
        "tool_arguments_accuracy": _safe_rate(tool_arguments_correct, len(tool_rows)),
        "final_decisions": len(final_rows),
        "final_correct": final_correct,
        "final_accuracy": _safe_rate(final_correct, len(final_rows)),
        "by_capability": by_capability,
    }


def evaluate_next_action_tasks(tasks_path: str | Path, model) -> dict:
    """Teacher-force held-out oracle states and score only the next assistant action.

    The evaluator never executes tools. Each assistant turn in a verified oracle
    trajectory becomes one decision: the model receives exactly the messages before
    that turn plus the row's tool schemas. Tool-call IDs and hidden reasoning are
    ignored; tool names/arguments and final answers are scored directly.
    """
    tasks_path = Path(tasks_path)
    rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No next-action evaluation tasks")

    decisions: list[dict] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows, 1):
        task_id = str(row.get("task_id") or row.get("id") or f"task-{row_index}")
        capability = str(row.get("capability") or (row.get("eval_task") or {}).get("capability") or "unknown")
        tools = row.get("tools") or []
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{task_id}: missing messages")
        if not isinstance(tools, list):
            raise ValueError(f"{task_id}: tools must be a list")

        for message_index, expected_message in enumerate(messages):
            if not isinstance(expected_message, dict) or expected_message.get("role") != "assistant":
                continue
            decision_id = f"{task_id}:{message_index}"
            if decision_id in seen_ids:
                raise ValueError(f"duplicate decision_id: {decision_id}")
            seen_ids.add(decision_id)

            expected_calls = _expected_tool_calls(expected_message)
            expected_action_type = "tool_call" if expected_calls else "final"
            prefix = messages[:message_index]
            turn = model.complete(prefix, tools)
            predicted_calls = _predicted_tool_calls(turn)
            predicted_action_type = "tool_call" if predicted_calls else "final"
            action_type_correct = predicted_action_type == expected_action_type

            tool_name_correct: bool | None = None
            tool_arguments_correct: bool | None = None
            final_correct: bool | None = None
            if expected_action_type == "tool_call":
                expected_names = [call["name"] for call in expected_calls]
                predicted_names = [call["name"] for call in predicted_calls]
                expected_arguments = [call["arguments"] for call in expected_calls]
                predicted_arguments = [call["arguments"] for call in predicted_calls]
                tool_name_correct = predicted_names == expected_names
                tool_arguments_correct = predicted_arguments == expected_arguments
                exact = bool(tool_name_correct and tool_arguments_correct)
            else:
                expected_names = []
                predicted_names = [call["name"] for call in predicted_calls]
                final_correct = bool(
                    not predicted_calls
                    and gaia_score(str(turn.content or ""), str(expected_message.get("content") or ""))
                )
                exact = final_correct

            decisions.append(
                {
                    "decision_id": decision_id,
                    "task_id": task_id,
                    "assistant_message_index": message_index,
                    "capability": capability,
                    "expected_action_type": expected_action_type,
                    "predicted_action_type": predicted_action_type,
                    "action_type_correct": action_type_correct,
                    "expected_tool_names": expected_names,
                    "predicted_tool_names": predicted_names,
                    "tool_name_correct": tool_name_correct,
                    "tool_arguments_correct": tool_arguments_correct,
                    "final_correct": final_correct,
                    "exact": exact,
                }
            )

    if not decisions:
        raise ValueError("No assistant decisions in next-action evaluation tasks")
    return {
        "schema_version": "small-agent-next-action-eval/v1",
        "summary": _summary(decisions),
        "decisions": decisions,
    }


def _mcnemar_exact_p(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    lower = min(base_only, candidate_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _comparison_rows(base_rows: dict[str, dict], candidate_rows: dict[str, dict], ids: list[str]) -> dict:
    both_correct = sum(bool(base_rows[k]["exact"]) and bool(candidate_rows[k]["exact"]) for k in ids)
    both_wrong = sum((not bool(base_rows[k]["exact"])) and (not bool(candidate_rows[k]["exact"])) for k in ids)
    base_only = sum(bool(base_rows[k]["exact"]) and (not bool(candidate_rows[k]["exact"])) for k in ids)
    candidate_only = sum((not bool(base_rows[k]["exact"])) and bool(candidate_rows[k]["exact"]) for k in ids)
    return {
        "total_decisions": len(ids),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_only": base_only,
        "candidate_only": candidate_only,
        "net_exact_change": candidate_only - base_only,
        "mcnemar_exact_p": _mcnemar_exact_p(base_only, candidate_only),
    }


def compare_next_action_results(base: dict, candidate: dict) -> dict:
    """Compare two next-action eval artifacts on exactly the same decision IDs."""
    base_list = base.get("decisions") or []
    candidate_list = candidate.get("decisions") or []
    base_rows = {str(row["decision_id"]): row for row in base_list}
    candidate_rows = {str(row["decision_id"]): row for row in candidate_list}
    if len(base_rows) != len(base_list) or len(candidate_rows) != len(candidate_list):
        raise ValueError("next-action results contain duplicate decision IDs")
    if set(base_rows) != set(candidate_rows):
        raise ValueError("next-action result decision IDs differ")

    ids = sorted(base_rows)
    result = _comparison_rows(base_rows, candidate_rows, ids)
    capabilities = sorted({str(base_rows[k].get("capability") or "unknown") for k in ids})
    result["schema_version"] = "small-agent-next-action-comparison/v1"
    result["by_capability"] = {
        capability: _comparison_rows(
            base_rows,
            candidate_rows,
            [k for k in ids if str(base_rows[k].get("capability") or "unknown") == capability],
        )
        for capability in capabilities
    }
    return result


__all__ = ["evaluate_next_action_tasks", "compare_next_action_results"]
