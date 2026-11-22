"""Pre-registered non-GAIA closed-loop transfer: Qwen3.5-9B Step100 stop policy on frozen MuSiQue24.

No teacher actions, GAIA rows, original raw question/answer, or task paths leave the runner.
The 24 MuSiQue TRAIN-origin parents were held out of v8 adapter training; this
is a mechanism transfer probe, not proof on unseen source families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

FROZEN_TASKS_SHA256 = "e6253660ac6b834041f4ac48dd5c930a3b37ce71654d0a1261bf182c756b6b0f"
FROZEN_TASK_MANIFEST_SHA256 = "2767ec507540cabe91af5feb58f76c15669b10ac8d64030033347ae3733018fe"
FROZEN_VALIDATION_PARENT_SHA256 = "cf4560d3a88059aa0d0fb00467d86b36d696dcc6e6c3cc4208637c13bcdbe8b7"
PROTOCOL = "V8-STEP100-STOP-MUSIQUE24-PREREG-V1"
TASKS = 24
SHARDS = 4
PER_SHARD = 6


def frozen_tasks(tasks_path: Path, report_path: Path) -> list[dict]:
    raw = tasks_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_TASKS_SHA256:
        raise ValueError("Frozen MuSiQue24 physical source SHA mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (len(rows) != TASKS
        or report.get("protocol") != "v8-musique-real-20passage-autonomous-probe-v1"
        or report.get("eligible_mu_parents") != TASKS
        or report.get("input_validation_sha256") != FROZEN_VALIDATION_PARENT_SHA256
        or report.get("task_manifest_sha256") != FROZEN_TASK_MANIFEST_SHA256
        or report.get("train_holdout_task_overlap") != 0
        or report.get("not_GAIA") is not True
        or report.get("not_model_evaluation") is not True):
        raise ValueError("Frozen MuSiQue24 source and training-holdout isolation failed")
    rows.sort(key=lambda row: row["task_id"])
    if (len({r["task_id"] for r in rows}) != TASKS
        or hashlib.sha256("\n".join(r["task_id"] for r in rows).encode()).hexdigest()
            != FROZEN_TASK_MANIFEST_SHA256):
        raise ValueError("Frozen MuSiQue task identities drifted")
    for row in rows:
        task = row["eval_task"]
        if (row["source"] != "musique-train-locked-holdout"
            or int(row["hop_count"]) not in (3, 4)
            or set(task["files"]) != {"passages/index.txt"} |
                {k for k in task["files"] if k.startswith("passages/p") and k.endswith(".txt")}
            or len(task["files"]) != 21
            or task.get("required_tools") != ["read"]
            or not task.get("question")
            or not task.get("expected_answer")):
            raise ValueError("Frozen source has missing passages or invalid task contract")
    return rows


def evaluate(tasks: list[dict], *, shard: int, arm: str, base_url: str, work_root: Path) -> dict:
    if shard not in range(SHARDS) or arm not in {"control", "stop"}:
        raise ValueError("Invalid frozen transfer shard/arm")
    if len(tasks) != TASKS:
        raise ValueError("Must evaluate the full frozen MuSiQue24 selection")
    selected = tasks[shard * PER_SHARD:(shard + 1) * PER_SHARD]
    if len(selected) != PER_SHARD:
        raise AssertionError("Incomplete frozen transfer shard")
    model = OpenAICompatModel(model="qwen3.5-9b", base_url=base_url, max_new_tokens=512)
    safe = []
    for row in selected:
        task = row["eval_task"]
        ident = row["task_id"]
        workspace = work_root / ident
        _materialize_files(task, workspace)
        agent = AgentRuntime(
            model, default_tools(), max_steps=12,
            evidence_stop_contract=(arm == "stop"),
        )
        result = agent.run(task["question"], workspace)
        safe.append({
            "id_sha256": hashlib.sha256(("musique24-transfer-v1:" + ident).encode()).hexdigest(),
            "hop_count": int(row["hop_count"]),
            "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
            "completed": bool(result.completed),
            "stop_reason": str(result.stop_reason),
            "steps": result.metrics.steps,
            "tool_calls": result.metrics.tool_calls,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
        })
    return {
        "protocol": PROTOCOL, "shard": shard, "arm": arm, "n_tasks": PER_SHARD,
        "frozen_tasks_sha256": FROZEN_TASKS_SHA256,
        "v8_training_parent_overlap": 0,
        "tool_names": ["search", "read", "inspect", "python"],
        "max_steps": 12, "max_new_tokens": 512,
        "only_variable": "evidence_stop_contract",
        "no_task_content_or_answers_exported": True,
        "cases": safe,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--arm", choices=["control", "stop"], required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite frozen experiment evidence")
    rows = frozen_tasks(args.tasks, args.source_report)
    summary = evaluate(rows, shard=args.shard, arm=args.arm,
                       base_url=args.base_url, work_root=args.work_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print("MUSIQUE24_TRANSFER_SHARD", args.shard, args.arm,
          "correct", sum(row["correct"] for row in summary["cases"]),
          "of", PER_SHARD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
