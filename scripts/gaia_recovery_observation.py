"""Compare the v7 curriculum's error serialization against the runtime's.

These are synthetic, deterministic held-out episodes. Only the serialization of
the same search failure changes. The model chooses every action in both arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from gaia_small_agent.training.policy_eval import _FixtureSearchTool
from gaia_small_agent.training.trajectories import _materialize_files

_MISS = "ToolError[SEARCH_FIXTURE_MISS]: "


def oracle_error_messages(messages: list[dict]) -> list[dict]:
    return [
        {**m, "content": "ERROR[SEARCH_FIXTURE_MISS]: " + m["content"][len(_MISS):]}
        if m["role"] == "tool" and m["content"].startswith(_MISS) else m
        for m in messages
    ]


class Model(OpenAICompatModel):
    def __init__(self, *, oracle_serialization: bool):
        super().__init__("qwen3.5-9b", max_new_tokens=512)
        self.oracle_serialization = oracle_serialization

    def complete(self, messages, tools):
        if self.oracle_serialization:
            messages = oracle_error_messages(messages)
        return super().complete(messages, tools)

    def complete_final(self, messages, tools):
        if self.oracle_serialization:
            messages = oracle_error_messages(messages)
        return super().complete_final(messages, tools)


def evaluate(shard: int, work: Path) -> list[dict]:
    source = generate_oracle_policy_v3_dataset(work / "oracle")["test"]
    rows = [r for line in source.read_text().splitlines() if (r := json.loads(line))
            and r["capability"] in {"failure_recovery", "strategy_switch"}]
    if len(rows) != 50 or shard not in range(10):
        raise ValueError("expected 50 held-out recovery cases and 10 shards")
    records = []
    for row in rows[shard::10]:
        task = row["eval_task"]
        task_id = row["task_id"]
        record = {"id": task_id, "capability": row["capability"]}
        for condition in ("runtime", "oracle"):
            workspace = work / condition / task_id
            _materialize_files(task, workspace)
            runtime = AgentRuntime(Model(oracle_serialization=condition == "oracle"),
                                   default_tools(), max_steps=12)
            runtime.tools["search"] = _FixtureSearchTool(task["search_fixtures"])
            start = perf_counter()
            result = runtime.run(task["question"], workspace)
            errors = [e.step for e in result.trace if e.kind == "tool_result"
                      and e.data["error_code"] == "SEARCH_FIXTURE_MISS"]
            valid_after_miss = any(e.kind == "tool_result" and e.data["ok"]
                                   and errors and e.step > errors[0] for e in result.trace)
            record[condition] = {
                "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
                "completed": result.completed, "stop": result.stop_reason,
                "tool_calls": result.metrics.tool_calls, "tool_errors": result.metrics.tool_errors,
                "duplicates": result.metrics.duplicate_calls_blocked,
                "miss_observed": bool(errors), "successful_tool_after_miss": valid_after_miss,
                "latency_s": round(perf_counter() - start, 3),
            }
        records.append(record)
    return records


def aggregate(files: list[Path]) -> dict:
    cases = [r for f in files for r in json.loads(f.read_text())]
    if len(files) != 10 or len({r["id"] for r in cases}) != 50 or len(cases) != 50:
        raise ValueError("incomplete paired recovery evidence")
    result = {}
    for capability in ("failure_recovery", "strategy_switch"):
        cohort = [r for r in cases if r["capability"] == capability]
        if len(cohort) != 25:
            raise ValueError("need 25 paired cases per capability")
        summary = {}
        for arm in ("runtime", "oracle"):
            arm_rows = [r[arm] for r in cohort]
            summary[arm] = {
                **{k: sum(r[k] for r in arm_rows) for k in (
                    "correct", "completed", "tool_calls", "tool_errors", "duplicates",
                    "miss_observed", "successful_tool_after_miss",
                )},
                "stops": dict(Counter(r["stop"] for r in arm_rows)),
                "mean_latency_s": round(sum(r["latency_s"] for r in arm_rows) / 25, 3),
            }
        summary["rescues"] = sum(not r["runtime"]["correct"] and r["oracle"]["correct"] for r in cohort)
        summary["regressions"] = sum(r["runtime"]["correct"] and not r["oracle"]["correct"] for r in cohort)
        result[capability] = summary
    return {"count": 50, "by_capability": result, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate(args.inputs)
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    else:
        result = evaluate(args.shard, Path("runs/recovery-observation"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
