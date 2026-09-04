from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_results(root: Path) -> dict[str, dict]:
    rows = {}
    for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row["task_id"])
        if task_id in rows:
            raise ValueError(f"duplicate task_id in {root}: {task_id}")
        rows[task_id] = row
    return rows


def _summary(rows: dict[str, dict]) -> dict:
    values = list(rows.values())
    n = len(values)
    return {
        "correct": sum(bool(row["correct"]) for row in values),
        "total": n,
        "accuracy": (sum(bool(row["correct"]) for row in values) / n) if n else 0.0,
        "average_steps": (sum(int(row.get("steps", 0)) for row in values) / n) if n else 0.0,
        "average_tool_calls": (sum(int(row.get("tool_calls", 0)) for row in values) / n) if n else 0.0,
        "average_tool_errors": (sum(int(row.get("tool_errors", 0)) for row in values) / n) if n else 0.0,
    }


def _transition_summary(base_rows: dict[str, dict], tuned_rows: dict[str, dict], task_ids: list[str]) -> dict:
    wrong_to_correct = sum((not bool(base_rows[k]["correct"])) and bool(tuned_rows[k]["correct"]) for k in task_ids)
    correct_to_wrong = sum(bool(base_rows[k]["correct"]) and (not bool(tuned_rows[k]["correct"])) for k in task_ids)
    return {
        "task_count": len(task_ids),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_correct_change": wrong_to_correct - correct_to_wrong,
    }


def compare_runs(base_root: str | Path, adapter_root: str | Path) -> dict:
    base_root = Path(base_root)
    adapter_root = Path(adapter_root)
    base_config = _load_json(base_root / "run_config.json")
    adapter_config = _load_json(adapter_root / "run_config.json")
    base_without_adapter = dict(base_config)
    tuned_without_adapter = dict(adapter_config)
    base_adapter = base_without_adapter.pop("adapter", None)
    tuned_adapter = tuned_without_adapter.pop("adapter", None)
    if base_adapter is not None or not tuned_adapter:
        raise ValueError("comparison requires Base without adapter and E1 with adapter")
    if base_without_adapter != tuned_without_adapter:
        raise ValueError("run configuration mismatch: only adapter may differ")

    base_rows = _load_results(base_root)
    tuned_rows = _load_results(adapter_root)
    if set(base_rows) != set(tuned_rows):
        raise ValueError("run task IDs differ")

    all_ids = list(base_rows)
    overall = _transition_summary(base_rows, tuned_rows, all_ids)
    supported_ids = [
        task_id for task_id in all_ids
        if not base_rows[task_id].get("capability_gap") and not tuned_rows[task_id].get("capability_gap")
    ]
    supported = _transition_summary(base_rows, tuned_rows, supported_ids)
    return {
        "treatment": "LoRA adapter only",
        **overall,
        "supported": supported,
        "base": _summary(base_rows),
        "adapter": _summary(tuned_rows),
    }
