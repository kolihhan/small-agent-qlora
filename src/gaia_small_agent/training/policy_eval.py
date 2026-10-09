from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from ..benchmark.scoring import gaia_score
from ..tools.base import ToolResult
from ..tools.search import SearchTool
from .trajectories import _materialize_files


class _FixtureSearchTool(SearchTool):
    """Deterministic search replacement for synthetic held-out tasks."""

    def __init__(self, fixtures: dict[str, str]):
        self.fixtures = dict(fixtures)

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "'query' must be a non-empty string", "BAD_ARGUMENTS")
        folded = query.casefold()
        for needle, content in self.fixtures.items():
            if str(needle).casefold() in folded:
                return ToolResult(True, str(content))
        return ToolResult(False, "No deterministic search fixture matched query", "SEARCH_FIXTURE_MISS")


def _used_tools(result) -> set[str]:
    return {
        str(event.data.get("name"))
        for event in result.trace
        if event.kind == "tool_call" and event.data.get("name")
    }


def evaluate_policy_tasks(
    tasks_path: str | Path,
    runtime_factory: Callable[[], object],
    work_root: str | Path,
) -> dict:
    """Evaluate a runtime on deterministic non-GAIA tool-policy tasks."""
    tasks_path = Path(tasks_path)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No policy evaluation tasks")

    total = correct = completed = required_ok_count = 0
    tool_calls = tool_errors = duplicate_calls_blocked = 0
    stop_reasons: Counter[str] = Counter()
    by_capability: dict[str, dict[str, int]] = {}

    for index, row in enumerate(rows, 1):
        task = row.get("eval_task")
        if not isinstance(task, dict):
            raise ValueError(f"line {index}: missing eval_task")
        task_id = str(row.get("task_id") or f"task-{index}")
        question = str(task["question"])
        expected = str(task["expected_answer"])
        required_tools = task.get("required_tools") or []
        if not isinstance(required_tools, list):
            raise ValueError(f"line {index}: required_tools must be a list")
        capability = str(task.get("capability") or "unknown")

        workspace = work_root / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        _materialize_files(task, workspace)

        runtime = runtime_factory()
        fixtures = task.get("search_fixtures") or {}
        original_search = None
        replaced_search = False
        if fixtures:
            tools = getattr(runtime, "tools", None)
            if not isinstance(tools, dict):
                raise ValueError("runtime must expose a mutable tools mapping for search fixtures")
            original_search = tools.get("search")
            tools["search"] = _FixtureSearchTool(fixtures)
            replaced_search = True

        try:
            result = runtime.run(question, workspace)
        finally:
            if replaced_search:
                if original_search is None:
                    runtime.tools.pop("search", None)
                else:
                    runtime.tools["search"] = original_search

        is_completed = bool(result.completed)
        is_correct = bool(is_completed and gaia_score(result.answer, expected))
        required_ok = set(map(str, required_tools)).issubset(_used_tools(result))

        total += 1
        correct += int(is_correct)
        completed += int(is_completed)
        required_ok_count += int(required_ok)
        tool_calls += int(result.metrics.tool_calls)
        tool_errors += int(result.metrics.tool_errors)
        duplicate_calls_blocked += int(result.metrics.duplicate_calls_blocked)
        stop_reasons[str(result.stop_reason)] += 1

        bucket = by_capability.setdefault(
            capability,
            {"total": 0, "correct": 0, "completed": 0, "required_tools_satisfied": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(is_correct)
        bucket["completed"] += int(is_completed)
        bucket["required_tools_satisfied"] += int(required_ok)

    return {
        "total": total,
        "correct": correct,
        "completed": completed,
        "exact_success_rate": (correct * 100.0 / total),
        "completion_rate": (completed * 100.0 / total),
        "required_tools_satisfied": required_ok_count,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "duplicate_calls_blocked": duplicate_calls_blocked,
        "stop_reasons": dict(stop_reasons),
        "by_capability": by_capability,
    }


__all__ = ["evaluate_policy_tasks"]
