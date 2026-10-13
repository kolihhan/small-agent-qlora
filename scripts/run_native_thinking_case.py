from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import check_default_tools, default_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--reasoning-budget", type=int, required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.reasoning_budget < 0:
        raise SystemExit("reasoning budget must be bounded and >= 0")

    with tempfile.TemporaryDirectory(prefix="native-thinking-preflight-") as workspace:
        preflight = check_default_tools(workspace, live_search=False)
    if not preflight["ready"]:
        raise SystemExit("default tool preflight failed")

    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    evaluation = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    selected = [row for row in evaluation if str(row["task_id"]) == args.task_id]
    if len(selected) != 1:
        raise SystemExit(f"expected exactly one frozen Evaluation100 task {args.task_id}, got {len(selected)}")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    def factory() -> AgentRuntime:
        return AgentRuntime(
            OpenAICompatModel(
                model=args.model,
                base_url=args.openai_base_url,
                max_new_tokens=512,
            ),
            default_tools(),
            max_steps=12,
        )

    run_config = {
        "experiment": "native-thinking-budget-spike/v1",
        "model": args.model,
        "task_id": args.task_id,
        "reasoning_budget": args.reasoning_budget,
        "max_steps": 12,
        "max_new_tokens": 512,
        "context_tokens": 16384,
        "temperature": 0,
        "dataset_revision": GAIA_REVISION,
        "seed": LOCAL_PROTOCOL_SEED,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
        "tool_names": list(preflight["tool_names"]),
    }

    original_selector = gaia_runner.select_gaia_partition
    gaia_runner.select_gaia_partition = lambda rows, partition, seed: list(selected)
    try:
        summary = gaia_runner.run_gaia100(
            dataset,
            factory,
            work_root,
            seed=LOCAL_PROTOCOL_SEED,
            manifest_path=None,
            dataset_revision=GAIA_REVISION,
            run_config=run_config,
            partition="evaluation",
        )
    finally:
        gaia_runner.select_gaia_partition = original_selector

    result_path = work_root / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise SystemExit(f"expected one result row, got {len(rows)}")
    row = rows[0]
    tool_names = [
        str(event.get("data", {}).get("name") or "")
        for event in row.get("trace", [])
        if event.get("kind") == "tool_call"
    ]
    payload = {
        "schema_version": "native-thinking-budget-spike/v1",
        "reasoning_budget": args.reasoning_budget,
        "task_id": args.task_id,
        "correct": bool(row["correct"]),
        "completed": bool(row["completed"]),
        "stop_reason": row["stop_reason"],
        "model_answer": row["model_answer"],
        "steps": int(row["steps"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_errors": int(row["tool_errors"]),
        "duplicate_calls_blocked": int(row["duplicate_calls_blocked"]),
        "latency_ms": float(row["latency_ms"]),
        "used_tools": tool_names,
        "used_live_search": "search" in tool_names,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
