"""Strict complete GAIA100 matched Base-v8 summary; no partial score claims."""
from __future__ import annotations

import json
import os
from pathlib import Path

from gaia_small_agent.benchmark.gaia100 import summarize_results


def aggregate(root: Path, *, shard_count: int = 20, cases: int = 100) -> dict:
    rows_by_arm = {}
    for arm in ("base", "qlora-v8"):
        selected = {}
        paths = sorted(root.glob("gaia-v8-paired-shard-*/" + arm + "/shard-*/results.jsonl"))
        if len(paths) != shard_count:
            raise ValueError(f"{arm}: expected {shard_count} shards, got {len(paths)}")
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                task = str(row["task_id"])
                if task in selected:
                    raise ValueError(f"{arm}: duplicate task {task}")
                selected[task] = row
        if len(selected) != cases:
            raise ValueError(f"{arm}: incomplete GAIA100 {len(selected)}/{cases}")
        rows_by_arm[arm] = selected
    base, v8 = rows_by_arm["base"], rows_by_arm["qlora-v8"]
    if set(base) != set(v8):
        raise ValueError("GAIA100 Base-v8 task identity mismatch")
    base_summary = summarize_results(list(base.values()))
    v8_summary = summarize_results(list(v8.values()))
    rescues = sum(not bool(base[k]["correct"]) and bool(v8[k]["correct"]) for k in base)
    regressions = sum(bool(base[k]["correct"]) and not bool(v8[k]["correct"]) for k in base)
    if rescues - regressions != v8_summary["correct"] - base_summary["correct"]:
        raise ValueError("GAIA100 matched delta mismatch")
    return {
        "schema_version": "matched-qlora-v8-gaia100/v1",
        "benchmark": "GAIA 2023 validation / frozen Evaluation100",
        "training_run_id": 44925307580,
        "frozen_eval_commit": "08fbcaa5ed82c5238188c45d5748993bae9bf1e1",
        "same_runner_paired": True,
        "method": {"shards": shard_count, "agent_max_steps": 12,
                   "temperature": 0, "reasoning_budget": 0,
                   "context_tokens": 16384, "max_new_tokens": 512},
        "base": base_summary, "qlora_v8": v8_summary,
        "paired_rescues": rescues, "paired_regressions": regressions,
        "net_correct_gain": v8_summary["correct"] - base_summary["correct"],
        "historical_v3b_correct": 18, "historical_v3d_correct": 18,
        "historical_note": "Old v3B/v3D 18/100 figures are historical, not newly paired.",
    }


if __name__ == "__main__":
    results = aggregate(Path("evidence"))
    Path("gaia-v8-matched-summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GAIA_V8_MATCHED_VERIFIED", json.dumps({
        "base_correct": results["base"]["correct"],
        "v8_correct": results["qlora_v8"]["correct"],
        "rescues": results["paired_rescues"],
        "regressions": results["paired_regressions"],
        "completion": results["qlora_v8"]["completion_rate"],
        "tool_calls": results["qlora_v8"]["tool_calls"],
    }), flush=True)
