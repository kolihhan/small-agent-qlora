"""Diagnostic-only v7 full traces on six frozen synthetic held-out policy tasks.

No new training, no modified tool execution, no search-format intervention.
This makes the failure cause observable rather than rerunning a 50-task score.
"""
from __future__ import annotations

import argparse
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

CAPABILITIES = ("failure_recovery", "strategy_switch")


def select_probe_rows(source: Path) -> list[dict]:
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    chosen = []
    for capability in CAPABILITIES:
        rows = [r for r in records if r["capability"] == capability]
        if len(rows) != 25:
            raise ValueError("not the frozen 25-case per-capability test split")
        chosen.extend(rows[:3])
    if len({r["task_id"] for r in chosen}) != 6:
        raise ValueError("duplicated probe IDs")
    return chosen


def probe(work: Path) -> dict:
    source = generate_oracle_policy_v3_dataset(work / "oracle")["test"]
    rows = select_probe_rows(source)
    output = []
    for item in rows:
        task = item["eval_task"]
        workspace = work / "runs" / item["task_id"]
        _materialize_files(task, workspace)
        runtime = AgentRuntime(Model(oracle_serialization=False), default_tools(), max_steps=12)
        runtime.tools["search"] = _FixtureSearchTool(task["search_fixtures"])
        begin = perf_counter()
        result = runtime.run(task["question"], workspace)
        trace = [{
            "step": event.step,
            "kind": event.kind,
            "name": event.data.get("name"),
            "arguments": event.data.get("arguments"),
            "ok": event.data.get("ok"),
            "error_code": event.data.get("error_code"),
            "content": str(event.data.get("content", ""))[:2000],
        } for event in result.trace]
        record = {
            "id": item["task_id"], "capability": item["capability"],
            "question": task["question"], "expected": task["expected_answer"],
            "answer": result.answer,
            "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
            "completed": result.completed, "stop_reason": result.stop_reason,
            "tool_calls": result.metrics.tool_calls,
            "tool_errors": result.metrics.tool_errors,
            "latency_s": round(perf_counter() - begin, 2),
            "trace": trace,
        }
        output.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return {"count": len(output), "cohort": "frozen synthetic policy200 test",
            "model": "v7 GGUF adapter checksum-pinned in workflow",
            "cases": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = probe(Path("runs/v7-recovery-trace-probe"))
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
