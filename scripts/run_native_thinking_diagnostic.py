from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS, SYSTEM_PROMPT
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import check_default_tools, default_tools
from gaia_small_agent.tools.base import Tool


TASK_ID = "f918266a-b3e0-4914-865d-4faa564f1aef"
BUDGETS = (0, 256)


class EchoSchemaTool(Tool):
    name = "echo"
    description = "Return the supplied text unchanged."
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, arguments, workspace):  # pragma: no cover - smoke calls model only
        raise RuntimeError("smoke tool should not execute")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _preflight_tools() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="small-agent-native-thinking-") as workspace:
        report = check_default_tools(workspace, live_search=False)
    if not report["ready"]:
        failed = [
            f"{name}: {check.get('error_code') or check.get('detail') or 'failed'}"
            for name, check in report["checks"].items()
            if not check.get("ok")
        ]
        raise RuntimeError("evaluation tool preflight failed: " + "; ".join(failed))
    return list(report["tool_names"])


def _smoke(model_name: str, base_url: str, budget: int) -> dict:
    model = OpenAICompatModel(
        model=model_name,
        base_url=base_url,
        max_new_tokens=512,
        thinking_budget_tokens=budget,
    )
    turn = model.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Use the echo tool exactly once with text ping."},
        ],
        [EchoSchemaTool().definition()],
    )
    return {
        "budget": budget,
        "reasoning_present": bool(turn.reasoning_content),
        "reasoning_chars": len(turn.reasoning_content),
        "tool_call_count": len(turn.tool_calls),
        "tool_names": [call.name for call in turn.tool_calls],
        "content_present": bool(turn.content.strip()),
    }


def _read_single_result(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row, got {len(rows)}")
    return rows[0]


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

    smokes = {
        str(budget): _smoke(args.model, args.openai_base_url, budget)
        for budget in BUDGETS
    }
    if smokes["256"]["tool_call_count"] < 1:
        raise RuntimeError("bounded-thinking smoke did not produce a native tool call")

    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    evaluation = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    selected = [row for row in evaluation if str(row["task_id"]) == TASK_ID]
    if len(selected) != 1:
        raise SystemExit(f"expected diagnostic task {TASK_ID}, got {len(selected)} matches")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    original_selector = gaia_runner.select_gaia_partition
    summaries: dict[str, dict] = {}
    rows_by_arm: dict[str, dict] = {}

    try:
        gaia_runner.select_gaia_partition = lambda rows, partition, seed: list(selected)
        for budget in BUDGETS:
            arm = f"thinking_{budget}"
            arm_root = work_root / arm

            def factory(budget_tokens: int = budget) -> AgentRuntime:
                model = OpenAICompatModel(
                    model=args.model,
                    base_url=args.openai_base_url,
                    max_new_tokens=512,
                    thinking_budget_tokens=budget_tokens,
                )
                return AgentRuntime(model, default_tools(), max_steps=12)

            run_config = {
                "experiment": "native-thinking-diagnostic/v1",
                "arm": arm,
                "model": args.model,
                "thinking_budget_tokens": budget,
                "max_steps": 12,
                "max_new_tokens": 512,
                "seed": LOCAL_PROTOCOL_SEED,
                "dataset_revision": GAIA_REVISION,
                "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
                "tool_names": tool_names,
                "task_ids": [TASK_ID],
                "system_prompt_sha256": __import__("hashlib").sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
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
            rows_by_arm[arm] = _compact(_read_single_result(arm_root / "results.jsonl"))
    finally:
        gaia_runner.select_gaia_partition = original_selector

    off = rows_by_arm["thinking_0"]
    on = rows_by_arm["thinking_256"]
    payload = {
        "schema_version": "native-thinking-diagnostic/v1",
        "protocol": {
            "benchmark": "GAIA 2023 validation / frozen Evaluation100 deterministic diagnostic",
            "task_id": TASK_ID,
            "model": args.model,
            "temperature": 0,
            "thinking_budgets": list(BUDGETS),
            "context_tokens": 16384,
            "max_new_tokens": 512,
            "agent_max_steps": 12,
            "system_prompt_changed": False,
            "adapter": None,
            "intended_behavioral_variable": "thinking_budget_tokens",
        },
        "smoke": smokes,
        "summary": summaries,
        "paired": {
            "thinking_0": off,
            "thinking_256": on,
            "delta_256_minus_0": {
                "correct": int(on["correct"]) - int(off["correct"]),
                "steps": on["steps"] - off["steps"],
                "tool_calls": on["tool_calls"] - off["tool_calls"],
                "tool_errors": on["tool_errors"] - off["tool_errors"],
                "duplicate_calls_blocked": on["duplicate_calls_blocked"] - off["duplicate_calls_blocked"],
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
