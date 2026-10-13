from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import gaia_small_agent.agent.loop as agent_loop
import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import check_default_tools, default_tools


TASK_IDS = (
    "f918266a-b3e0-4914-865d-4faa564f1aef",
    "e4e91f1c-1dcd-439e-9fdd-cb976f5293fd",
    "8131e2c0-0083-4265-9ce7-78c2d568425d",
)

OLD_PROMPT = """You are Small Agent, a compact autonomous tool-using assistant.
Choose the next action from the available tools based only on the user task and observations.
There is no fixed workflow. Recover from tool errors by changing strategy when useful.
Use tools only when needed. Never invent tool results.
When you have enough evidence, return the final answer concisely.
Do not reveal hidden chain-of-thought; tool calls and short factual status are sufficient.
"""

NEW_PROMPT = agent_loop.SYSTEM_PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _preflight_tools() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="small-agent-prompt-diagnostic-") as workspace:
        report = check_default_tools(workspace, live_search=False)
    if not report["ready"]:
        failed = [
            f"{name}: {check.get('error_code') or check.get('detail') or 'failed'}"
            for name, check in report["checks"].items()
            if not check.get("ok")
        ]
        raise RuntimeError("evaluation tool preflight failed: " + "; ".join(failed))
    return list(report["tool_names"])


def _read_results(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["task_id"]): row for row in rows}


def _compact(row: dict) -> dict:
    return {
        "correct": bool(row["correct"]),
        "completed": bool(row["completed"]),
        "stop_reason": row["stop_reason"],
        "model_answer": row["model_answer"],
        "steps": int(row["steps"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_errors": int(row["tool_errors"]),
        "duplicate_calls_blocked": int(row["duplicate_calls_blocked"]),
        "failure_label": row["failure_label"],
    }


def main() -> int:
    args = build_parser().parse_args()
    tool_names = _preflight_tools()
    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    evaluation = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    selected = [row for row in evaluation if str(row["task_id"]) in TASK_IDS]
    found = {str(row["task_id"]) for row in selected}
    missing = set(TASK_IDS).difference(found)
    if missing:
        raise SystemExit(f"diagnostic task ids missing from frozen Evaluation100: {sorted(missing)}")
    if len(selected) != len(TASK_IDS):
        raise SystemExit(f"expected {len(TASK_IDS)} diagnostic tasks, got {len(selected)}")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    original_selector = gaia_runner.select_gaia_partition
    original_prompt = agent_loop.SYSTEM_PROMPT
    summaries: dict[str, dict] = {}
    arm_rows: dict[str, dict[str, dict]] = {}

    try:
        gaia_runner.select_gaia_partition = lambda rows, partition, seed: list(selected)
        for arm, prompt in (("old", OLD_PROMPT), ("goal_tracking", NEW_PROMPT)):
            agent_loop.SYSTEM_PROMPT = prompt
            arm_root = work_root / arm

            def factory() -> AgentRuntime:
                model = OpenAICompatModel(
                    model=args.model,
                    base_url=args.openai_base_url,
                    max_new_tokens=512,
                )
                return AgentRuntime(model, default_tools(), max_steps=12)

            run_config = {
                "experiment": "goal-tracking-prompt-diagnostic/v1",
                "arm": arm,
                "model": args.model,
                "max_steps": 12,
                "max_new_tokens": 512,
                "seed": LOCAL_PROTOCOL_SEED,
                "dataset_revision": GAIA_REVISION,
                "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
                "tool_names": tool_names,
                "task_ids": list(TASK_IDS),
            }
            summaries[arm] = gaia_runner.run_gaia100(
                dataset,
                factory,
                arm_root,
                seed=LOCAL_PROTOCOL_SEED,
                manifest_path=None,
                dataset_revision=GAIA_REVISION,
                run_config=run_config,
                partition="evaluation",
            )
            arm_rows[arm] = _read_results(arm_root / "results.jsonl")
    finally:
        agent_loop.SYSTEM_PROMPT = original_prompt
        gaia_runner.select_gaia_partition = original_selector

    paired = []
    for task_id in TASK_IDS:
        old = _compact(arm_rows["old"][task_id])
        new = _compact(arm_rows["goal_tracking"][task_id])
        paired.append(
            {
                "task_id": task_id,
                "old": old,
                "goal_tracking": new,
                "delta": {
                    "correct": int(new["correct"]) - int(old["correct"]),
                    "steps": new["steps"] - old["steps"],
                    "tool_calls": new["tool_calls"] - old["tool_calls"],
                    "tool_errors": new["tool_errors"] - old["tool_errors"],
                    "duplicate_calls_blocked": new["duplicate_calls_blocked"] - old["duplicate_calls_blocked"],
                },
            }
        )

    payload = {
        "schema_version": "goal-tracking-prompt-diagnostic/v1",
        "protocol": {
            "benchmark": "GAIA 2023 validation / frozen Evaluation100 diagnostic subset",
            "task_ids": list(TASK_IDS),
            "model": args.model,
            "temperature": 0,
            "reasoning_budget": 0,
            "context_tokens": 16384,
            "max_new_tokens": 512,
            "agent_max_steps": 12,
            "only_behavioral_variable": "SYSTEM_PROMPT",
        },
        "summary": summaries,
        "paired": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
