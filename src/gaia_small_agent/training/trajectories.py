from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..agent.loop import AgentRuntime, SYSTEM_PROMPT
from ..agent.types import AgentResult
from ..benchmark.scoring import gaia_score
from .protection import load_protected_question_hashes, question_hash

_REQUIRED_PROVENANCE = ("license", "generator", "generator_version", "oracle_type", "oracle_version")


def _verification(result: AgentResult, expected_answer: str, required_tools: list[str], provenance: dict[str, str]) -> dict[str, bool]:
    used_tools = {
        str(event.data.get("name"))
        for event in result.trace
        if event.kind == "tool_call" and event.data.get("name")
    }
    checks = {
        "completed": bool(result.completed),
        "final_answer_correct": bool(result.completed and gaia_score(result.answer, expected_answer)),
        "zero_tool_errors": result.metrics.tool_errors == 0,
        "zero_duplicate_blocks": result.metrics.duplicate_calls_blocked == 0,
        "required_tools_satisfied": set(required_tools).issubset(used_tools),
        "provenance_complete": all(str(provenance.get(key) or "") for key in _REQUIRED_PROVENANCE),
    }
    return checks


def trajectory_from_result(
    *,
    task_id: str,
    source: str,
    question: str,
    expected_answer: str,
    result: AgentResult,
    tools: list[dict],
    required_tools: list[str] | None = None,
    provenance: dict[str, str] | None = None,
) -> dict:
    """Convert a visible trace into a policy-SFT trajectory.

    Hidden reasoning is never stored. A row is training-eligible only when the
    final answer is correct *and* the visible policy trace is clean.
    """
    required_tools = list(required_tools or [])
    provenance = dict(provenance or {})
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = sorted({event.step for event in result.trace})
    for step in steps:
        events = [event for event in result.trace if event.step == step]
        calls = [event for event in events if event.kind == "tool_call"]
        if calls:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": str(event.data["id"]),
                        "type": "function",
                        "function": {
                            "name": str(event.data["name"]),
                            "arguments": event.data.get("arguments") or {},
                        },
                    }
                    for event in calls
                ],
            })
            for event in events:
                if event.kind != "tool_result":
                    continue
                content = str(event.data.get("content", ""))
                if not event.data.get("ok", False):
                    code = event.data.get("error_code") or "TOOL_ERROR"
                    content = f"ToolError[{code}]: {content}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(event.data["id"]),
                    "name": str(event.data["name"]),
                    "content": content,
                })
        final = next((event for event in events if event.kind == "final"), None)
        if final is not None:
            messages.append({"role": "assistant", "content": str(final.data.get("answer", result.answer))})

    checks = _verification(result, expected_answer, required_tools, provenance)
    return {
        "task_id": task_id,
        "source": source,
        "verified": all(checks.values()),
        "verification": checks,
        "provenance": provenance,
        "required_tools": required_tools,
        "expected_answer": expected_answer,
        "tools": tools,
        "messages": messages,
        "metrics": {
            "steps": result.metrics.steps,
            "tool_calls": result.metrics.tool_calls,
            "tool_successes": result.metrics.tool_successes,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
        },
    }


def _materialize_files(task: dict, workspace: Path) -> None:
    files = task.get("files") or {}
    if not isinstance(files, dict):
        raise ValueError("task files must be an object")
    root = workspace.resolve()
    for raw_name, raw_content in files.items():
        if not isinstance(raw_name, str) or not isinstance(raw_content, str):
            raise ValueError("task files must map string paths to string content")
        target = (root / raw_name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("task file path escapes workspace") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = raw_content.encode("utf-8")
        if target.exists() and target.read_bytes() != encoded:
            raise ValueError(f"task workspace file drift: {raw_name}")
        target.write_bytes(encoded)


def _load_progress(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            task_id = str(row["task_id"])
            status = str(row["status"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trajectory progress line {line_no}") from exc
        if task_id in rows:
            raise ValueError(f"duplicate trajectory progress task_id: {task_id}")
        rows[task_id] = status
    return rows


def _existing_verified_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            task_id = str(row["task_id"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid verified trajectory line {line_no}") from exc
        if task_id in ids:
            raise ValueError(f"duplicate verified trajectory task_id: {task_id}")
        ids.add(task_id)
    return ids


def collect_verified_trajectories(
    tasks_path: str | Path,
    output_path: str | Path,
    runtime_factory: Callable[[], AgentRuntime],
    work_root: str | Path,
    protected_questions_path: str | Path | None = None,
) -> dict[str, int]:
    """Persist only clean answer-correct trajectories, resumably by task id."""
    tasks_path = Path(tasks_path)
    output_path = Path(output_path)
    work_root = Path(work_root)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.jsonl")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    progress_path.touch(exist_ok=True)

    protected_hashes = load_protected_question_hashes(protected_questions_path)
    progress = _load_progress(progress_path)
    verified_ids = _existing_verified_ids(output_path)
    for task_id in verified_ids:
        progress.setdefault(task_id, "verified")

    counts = {"total": 0, "verified": 0, "failed": 0, "rejected_gaia": 0}
    if protected_questions_path is not None:
        counts["rejected_protected"] = 0

    tasks: list[tuple[int, dict]] = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                tasks.append((line_number, json.loads(line)))
    counts["total"] = len(tasks)

    def record_progress(task_id: str, status: str) -> None:
        if task_id in progress:
            return
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id, "status": status}, sort_keys=True) + "\n")
            handle.flush()
        progress[task_id] = status

    runtime: AgentRuntime | None = None
    tool_schemas: list[dict] | None = None
    for line_number, task in tasks:
        source = str(task.get("source") or "")
        task_id = str(task.get("id") or f"task-{line_number}")
        if task_id in progress:
            continue
        if "gaia" in source.casefold():
            record_progress(task_id, "rejected_gaia")
            continue
        provenance = {key: str(task.get(key) or "") for key in _REQUIRED_PROVENANCE}
        if any(not provenance[key] for key in _REQUIRED_PROVENANCE):
            raise ValueError(f"line {line_number}: incomplete task provenance")
        required_tools = task.get("required_tools")
        if not isinstance(required_tools, list) or any(not isinstance(item, str) or not item for item in required_tools):
            raise ValueError(f"line {line_number}: required_tools must be a list of tool names")
        question = str(task["question"])
        if question_hash(question) in protected_hashes:
            record_progress(task_id, "rejected_protected")
            continue
        expected = str(task["expected_answer"])
        if runtime is None:
            runtime = runtime_factory()
            tool_schemas = [tool.definition() for tool in runtime.tools.values()]
        workspace = work_root / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        _materialize_files(task, workspace)
        result = runtime.run(question, workspace)
        row = trajectory_from_result(
            task_id=task_id,
            source=source,
            question=question,
            expected_answer=expected,
            result=result,
            tools=tool_schemas or [],
            required_tools=required_tools,
            provenance=provenance,
        )
        if row["verified"]:
            if task_id not in verified_ids:
                with output_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                verified_ids.add(task_id)
            record_progress(task_id, "verified")
        else:
            record_progress(task_id, "failed")

    for status in progress.values():
        if status in counts:
            counts[status] += 1
    summary_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts
