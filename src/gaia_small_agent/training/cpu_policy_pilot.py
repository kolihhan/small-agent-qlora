from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

from ..agent.loop import AgentRuntime
from ..agent.types import AgentResult, RunMetrics, TraceEvent
from ..benchmark.scoring import gaia_score
from ..tools.inspect import InspectTool
from ..tools.python_tool import PythonTool
from ..tools.read import ReadTool
from .policy_tasks import generate_policy_tasks
from .trajectories import _materialize_files, trajectory_from_result

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TRAIN_COUNT = 16
EVAL_COUNT = 12
TRAIN_SEED = "cpu-policy-pilot-train-v1"
EVAL_SEED = "cpu-policy-pilot-eval-v1"
MAX_STEPS = 4
MAX_NEW_TOKENS = 96
TRAIN_MAX_LENGTH = 768
TRAIN_EPOCHS = 1.0
LEARNING_RATE = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def _local_tools():
    return [ReadTool(), InspectTool(), PythonTool()]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different frozen pilot file: {path}")
    path.write_text(text, encoding="utf-8")


def build_frozen_task_sets(root: str | Path) -> tuple[list[dict], list[dict]]:
    """Create the pre-registered train/eval split used by the CPU pilot.

    Train uses policy-001..016 from one seed. Evaluation uses policy-017..028
    from a different seed so filenames, keys and arithmetic prompts are also
    distinct. Exact question overlap is rejected before any model is loaded.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_path = generate_policy_tasks(root / "train-tasks.jsonl", count=TRAIN_COUNT, seed=TRAIN_SEED)
    eval_pool_path = generate_policy_tasks(root / "eval-pool.jsonl", count=TRAIN_COUNT + EVAL_COUNT, seed=EVAL_SEED)
    train = _load_jsonl(train_path)
    evaluation = _load_jsonl(eval_pool_path)[TRAIN_COUNT:]

    if len(train) != TRAIN_COUNT or len(evaluation) != EVAL_COUNT:
        raise RuntimeError("frozen pilot split size drift")
    expected_train_ids = [f"policy-{i:03d}" for i in range(1, TRAIN_COUNT + 1)]
    expected_eval_ids = [f"policy-{i:03d}" for i in range(TRAIN_COUNT + 1, TRAIN_COUNT + EVAL_COUNT + 1)]
    if [row["id"] for row in train] != expected_train_ids:
        raise RuntimeError("frozen pilot training IDs drifted")
    if [row["id"] for row in evaluation] != expected_eval_ids:
        raise RuntimeError("frozen pilot evaluation IDs drifted")

    train_questions = {str(row["question"]) for row in train}
    eval_questions = {str(row["question"]) for row in evaluation}
    overlap = train_questions & eval_questions
    if overlap:
        raise RuntimeError(f"frozen pilot train/eval question overlap: {sorted(overlap)}")

    _write_jsonl(root / "eval-tasks.jsonl", evaluation)
    return train, evaluation


def _oracle_tool_call(task: dict) -> tuple[str, dict[str, Any]]:
    required = list(task.get("required_tools") or [])
    if len(required) != 1:
        raise ValueError(f"pilot oracle expects exactly one required tool: {task.get('id')}")
    tool_name = required[0]
    files = task.get("files") or {}
    if tool_name in {"read", "inspect"}:
        if len(files) != 1:
            raise ValueError(f"pilot file task must contain exactly one file: {task.get('id')}")
        filename = next(iter(files))
        return tool_name, {"source": filename} if tool_name == "read" else {"path": filename}
    if tool_name == "python":
        match = re.search(r"\bcompute\s+(.+?)\.\s", str(task["question"]))
        if match is None:
            raise ValueError(f"cannot derive deterministic python oracle: {task.get('id')}")
        return tool_name, {"code": f"print({match.group(1)})"}
    raise ValueError(f"unsupported pilot oracle tool: {tool_name}")


def oracle_trajectory(task: dict, workspace: str | Path) -> dict:
    """Create one verified tool-policy trajectory from the task's exact oracle."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    _materialize_files(task, workspace)
    tools = _local_tools()
    tool_by_name = {tool.name: tool for tool in tools}
    tool_name, arguments = _oracle_tool_call(task)
    result = tool_by_name[tool_name].run(arguments, workspace)
    if not result.ok:
        raise RuntimeError(f"oracle tool failed for {task['id']}: {result.error_code}: {result.content}")

    call_id = "oracle-1"
    answer = str(task["expected_answer"])
    agent_result = AgentResult(
        answer=answer,
        completed=True,
        stop_reason="final",
        trace=[
            TraceEvent("tool_call", 1, {"id": call_id, "name": tool_name, "arguments": arguments}),
            TraceEvent(
                "tool_result",
                1,
                {"id": call_id, "name": tool_name, "ok": True, "error_code": None, "content": result.content},
            ),
            TraceEvent("final", 2, {"answer": answer}),
        ],
        metrics=RunMetrics(steps=2, tool_calls=1, tool_successes=1),
    )
    provenance = {
        key: str(task[key])
        for key in ("license", "generator", "generator_version", "oracle_type", "oracle_version")
    }
    row = trajectory_from_result(
        task_id=str(task["id"]),
        source=str(task["source"]),
        question=str(task["question"]),
        expected_answer=answer,
        result=agent_result,
        tools=[tool.definition() for tool in tools],
        required_tools=list(task["required_tools"]),
        provenance=provenance,
    )
    if row["verified"] is not True:
        raise RuntimeError(f"oracle trajectory did not verify: {task['id']}")
    return row


def build_oracle_training_data(tasks: list[dict], output_path: str | Path, work_root: str | Path) -> Path:
    output_path = Path(output_path)
    work_root = Path(work_root)
    rows = [oracle_trajectory(task, work_root / str(task["id"])) for task in tasks]
    if len(rows) != TRAIN_COUNT or not all(row.get("verified") is True for row in rows):
        raise RuntimeError("pilot training trajectory gate failed")
    _write_jsonl(output_path, rows)
    return output_path


def evaluate_runtime(runtime: AgentRuntime, tasks: list[dict], work_root: str | Path) -> list[dict]:
    work_root = Path(work_root)
    rows: list[dict] = []
    for task in tasks:
        workspace = work_root / str(task["id"])
        workspace.mkdir(parents=True, exist_ok=True)
        _materialize_files(task, workspace)
        result = runtime.run(str(task["question"]), workspace)
        called_tools = [
            str(event.data.get("name"))
            for event in result.trace
            if event.kind == "tool_call" and event.data.get("name")
        ]
        required = set(str(name) for name in task.get("required_tools") or [])
        row = {
            "task_id": str(task["id"]),
            "category": str(task["category"]),
            "question": str(task["question"]),
            "expected_answer": str(task["expected_answer"]),
            "answer": result.answer,
            "completed": bool(result.completed),
            "stop_reason": result.stop_reason,
            "correct": bool(result.completed and gaia_score(result.answer, str(task["expected_answer"]))),
            "required_tool_used": required.issubset(set(called_tools)),
            "called_tools": called_tools,
            "metrics": {
                "steps": result.metrics.steps,
                "tool_calls": result.metrics.tool_calls,
                "tool_successes": result.metrics.tool_successes,
                "tool_errors": result.metrics.tool_errors,
                "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
            },
            "trace": [event.to_dict() for event in result.trace],
        }
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    exact = sum(bool(row["correct"]) for row in rows)
    completed = sum(bool(row["completed"]) for row in rows)
    tool_used = sum(bool(row["required_tool_used"]) for row in rows)
    return {
        "total": total,
        "exact_correct": exact,
        "exact_accuracy": exact / total if total else 0.0,
        "completed": completed,
        "completion_rate": completed / total if total else 0.0,
        "required_tool_used": tool_used,
        "required_tool_use_rate": tool_used / total if total else 0.0,
    }


def summarize_pair(base_rows: list[dict], lora_rows: list[dict]) -> dict:
    base_by_id = {row["task_id"]: row for row in base_rows}
    lora_by_id = {row["task_id"]: row for row in lora_rows}
    if list(base_by_id) != list(lora_by_id) or len(base_by_id) != len(base_rows) or len(lora_by_id) != len(lora_rows):
        raise ValueError("paired pilot task IDs differ or contain duplicates")

    transitions = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for task_id in base_by_id:
        before = bool(base_by_id[task_id]["correct"])
        after = bool(lora_by_id[task_id]["correct"])
        if not before and after:
            transitions["wrong_to_correct"] += 1
        elif before and not after:
            transitions["correct_to_wrong"] += 1
        elif before and after:
            transitions["correct_to_correct"] += 1
        else:
            transitions["wrong_to_wrong"] += 1

    base = _aggregate(base_rows)
    lora = _aggregate(lora_rows)
    if lora["exact_correct"] > base["exact_correct"]:
        verdict = "PILOT_IMPROVED"
    elif lora["exact_correct"] < base["exact_correct"]:
        verdict = "PILOT_REGRESSED"
    else:
        verdict = "PILOT_NO_GAIN"
    return {"base": base, "lora": lora, "transitions": transitions, "verdict": verdict}


def _question_hashes(rows: list[dict]) -> list[str]:
    return [hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest() for row in rows]


def _versions() -> dict[str, str | None]:
    packages = ("torch", "transformers", "bitsandbytes", "peft", "trl", "accelerate")
    out: dict[str, str | None] = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def _run_model_arm(tasks: list[dict], work_root: Path, *, adapter: str | None) -> list[dict]:
    from ..model.transformers_qwen import TransformersQwenModel

    model = TransformersQwenModel(
        model_name=MODEL_NAME,
        quantize_4bit=True,
        adapter=adapter,
        enable_thinking=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    runtime = AgentRuntime(model, _local_tools(), max_steps=MAX_STEPS)
    try:
        return evaluate_runtime(runtime, tasks, work_root)
    finally:
        del runtime
        del model
        gc.collect()


def run_cpu_policy_pilot(output_root: str | Path = "runs/cpu-policy-pilot-v1") -> dict:
    """Run the frozen resource-bounded Base-vs-LoRA synthetic pilot."""
    from .qlora import train_qlora

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    train_tasks, eval_tasks = build_frozen_task_sets(root / "protocol")
    trajectories_path = build_oracle_training_data(
        train_tasks,
        root / "train-trajectories.jsonl",
        root / "oracle-work",
    )

    base_rows = _run_model_arm(eval_tasks, root / "base-work", adapter=None)
    (root / "base-results.json").write_text(json.dumps(base_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gc.collect()

    adapter_dir = root / "adapter"
    train_qlora(
        trajectories_path,
        adapter_dir,
        model_name=MODEL_NAME,
        max_length=TRAIN_MAX_LENGTH,
        epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    gc.collect()

    lora_rows = _run_model_arm(eval_tasks, root / "lora-work", adapter=str(adapter_dir))
    (root / "lora-results.json").write_text(json.dumps(lora_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pair = summarize_pair(base_rows, lora_rows)

    result = {
        "schema_version": "cpu-policy-pilot/v1",
        "scope": "resource-bounded synthetic tool-policy pilot",
        "not_gaia": True,
        "not_4b": True,
        "model": MODEL_NAME,
        "protocol": {
            "train_count": TRAIN_COUNT,
            "eval_count": EVAL_COUNT,
            "train_seed": TRAIN_SEED,
            "eval_seed": EVAL_SEED,
            "train_question_hashes": _question_hashes(train_tasks),
            "eval_question_hashes": _question_hashes(eval_tasks),
            "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "tools": [tool.name for tool in _local_tools()],
            "training": {
                "oracle_trajectories": TRAIN_COUNT,
                "max_length": TRAIN_MAX_LENGTH,
                "epochs": TRAIN_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
            },
        },
        "environment": {
            "github_sha": os.environ.get("GITHUB_SHA"),
            "versions": _versions(),
        },
        **pair,
        "tasks": [
            {
                "task_id": base["task_id"],
                "category": base["category"],
                "base": base,
                "lora": lora,
            }
            for base, lora in zip(base_rows, lora_rows, strict=True)
        ],
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CPU_POLICY_PILOT_RESULT_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return result


def main() -> int:
    run_cpu_policy_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
