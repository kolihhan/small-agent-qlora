"""Paired generic answer-format instruction on heldout v7 closed-loop tasks.

Does a generic instruction-compliance reminder fix exact answer formatting
without breaking task decisions? No training and no new inference components.
The first 3 analyzed cases of the two failure cohorts are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from gaia_small_agent.training.policy_eval import _FixtureSearchTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.gaia_recovery_observation import Model

FORMAT_REMINDER = (
    "\nFollow the user's requested output format exactly. If they request only "
    "a code, number, value, or token, output that value alone without a sentence, "
    "Markdown, or explanation. Do not skip necessary tools or invent evidence.\n"
)
COHORT = {
    "failure_recovery": (3, 9),
    "strategy_switch": (3, 9),
    "tool_selection": (3, 6),
    "multi_step": (3, 6),
    "evidence_to_final": (3, 6),
    "no_tool_stop": (3, 6),
}
N_SHARDS = 6


def select_cases(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = []
    for capability, (start, stop) in COHORT.items():
        cohort = [r for r in records if r["capability"] == capability]
        if len(cohort) != 25:
            raise ValueError("not the frozen v3 policy test split")
        selected += cohort[start:stop]
    if len(selected) != 24 or len({r["task_id"] for r in selected}) != 24:
        raise ValueError("incomplete format diagnostic cohort")
    return selected


class FormatModel(Model):
    def __init__(self, *, reminder: bool):
        super().__init__(oracle_serialization=False)
        self.reminder = reminder

    def _prepare(self, messages: list[dict]) -> list[dict]:
        if not self.reminder:
            return messages
        if not messages or messages[0].get("role") != "system":
            raise ValueError("unexpected transcript prefix")
        return [{**messages[0], "content": messages[0]["content"] + FORMAT_REMINDER}, *messages[1:]]

    def complete(self, messages, tools):
        return super().complete(self._prepare(messages), tools)

    def complete_final(self, messages, tools):
        return super().complete_final(self._prepare(messages), tools)


def evaluate_shard(shard: int, work: Path) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid format shard")
    selected = select_cases(generate_oracle_policy_v3_dataset(work / "oracle")["test"])
    results = []
    for row in selected[shard::N_SHARDS]:
        task = row["eval_task"]
        record = {"id": row["task_id"], "capability": row["capability"]}
        for arm in ("original", "format_reminder"):
            workspace = work / arm / row["task_id"]
            _materialize_files(task, workspace)
            runtime = AgentRuntime(FormatModel(reminder=arm == "format_reminder"),
                                   default_tools(), max_steps=12)
            runtime.tools["search"] = _FixtureSearchTool(task["search_fixtures"])
            start = perf_counter()
            result = runtime.run(task["question"], workspace)
            observed = [str(event.data.get("content", ""))
                        for event in result.trace if event.kind == "tool_result"
                        and event.data.get("ok")]
            record[arm] = {
                "strict_correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
                "expected_value_present": bool(task["expected_answer"] in result.answer),
                "value_found_in_successful_observation": any(task["expected_answer"] in text for text in observed),
                "completed": bool(result.completed),
                "stop_reason": result.stop_reason,
                "tool_calls": result.metrics.tool_calls,
                "tool_errors": result.metrics.tool_errors,
                "duplicates_blocked": result.metrics.duplicate_calls_blocked,
                "latency_s": round(perf_counter() - start, 2),
                "answer": result.answer,
            }
        results.append(record)
        print("FORMAT_CASE", json.dumps(record), flush=True)
    return {"shard": shard, "count": len(results), "cases": results}


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(x["shard"] for x in shards) != list(range(N_SHARDS)):
        raise ValueError("missing/duplicate format diagnostic shard")
    rows = [r for x in shards for r in x["cases"]]
    if len(rows) != 24 or len({r["id"] for r in rows}) != 24:
        raise ValueError("incomplete paired format cohort")
    if Counter(r["capability"] for r in rows) != {
        "failure_recovery": 6, "strategy_switch": 6, "tool_selection": 3,
        "multi_step": 3, "evidence_to_final": 3, "no_tool_stop": 3,
    }:
        raise ValueError("format cohort drift")
    metrics = ("strict_correct", "expected_value_present",
               "value_found_in_successful_observation", "completed",
               "tool_calls", "tool_errors", "duplicates_blocked")
    stats = {}
    for arm in ("original", "format_reminder"):
        stats[arm] = {name: sum(r[arm][name] for r in rows) for name in metrics}
        stats[arm]["mean_latency_s"] = round(
            sum(r[arm]["latency_s"] for r in rows) / len(rows), 2)
        stats[arm]["stop_reasons"] = dict(Counter(r[arm]["stop_reason"] for r in rows))
    return {
        "count": len(rows), "arms": stats,
        "rescues": sum(not r["original"]["strict_correct"] and
                       r["format_reminder"]["strict_correct"] for r in rows),
        "regressions": sum(r["original"]["strict_correct"] and
                           not r["format_reminder"]["strict_correct"] for r in rows),
        "by_capability": {
            capability: {
                arm: {"strict": sum(r[arm]["strict_correct"] for r in rows
                                    if r["capability"] == capability),
                      "value_present": sum(r[arm]["expected_value_present"] for r in rows
                                           if r["capability"] == capability)}
                for arm in ("original", "format_reminder")
            } for capability in COHORT
        },
        "limitations": "synthetic 24-case diagnostic; not GAIA or a new checkpoint",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("FORMAT_AGGREGATE", json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    else:
        if args.shard is None:
            parser.error("--shard is required")
        result = evaluate_shard(args.shard, Path("runs/v7-format-ablation"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
