"""Paired GAIA Diagnostic25: replay successful duplicate tool observations.

Same v7 weights and 12-step runtime. Only the response to an exact,
already-successful duplicate changes; never executes the tool twice.
Gated questions, model answers and tool traces stay on the runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import (
    GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition,
)
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

ARMS = ("original", "replay_success")
N_SHARDS = 5


def evaluate_shard(index: int) -> dict:
    if index not in range(N_SHARDS):
        raise ValueError("invalid shard")
    dataset = load_validation()
    selection = select_gaia_partition(dataset, "diagnostic")
    if len(selection) != 25 or len({r["task_id"] for r in selection}) != 25:
        raise ValueError("GAIA diagnostic partition drift")
    cohort_digest = sha256("|".join(str(r["task_id"]) for r in selection).encode()).hexdigest()
    cases = shard_selected_rows(selection, shard_index=index, shard_count=N_SHARDS)
    by_arm = {}
    saved_selector = runner.select_gaia_partition
    try:
        for arm in ARMS:
            with TemporaryDirectory() as folder:
                runner.select_gaia_partition = lambda *args, **kwargs: cases
                runner.run_gaia100(
                    cases,
                    lambda: AgentRuntime(
                        OpenAICompatModel("qwen3.5-9b", max_new_tokens=512),
                        default_tools(),
                        max_steps=12,
                        replay_successful_duplicates=(arm == "replay_success"),
                    ),
                    folder, seed=LOCAL_PROTOCOL_SEED,
                )
                rows = [json.loads(line) for line in
                        (Path(folder) / "results.jsonl").read_text().splitlines()]
                if len(rows) != len(cases):
                    raise ValueError("incomplete evaluator output")
                by_arm[arm] = [{
                    "task_hash": sha256(r["task_id"].encode()).hexdigest(),
                    "correct": bool(r["correct"]),
                    "completed": bool(r["completed"]),
                    "stop_reason": r["stop_reason"],
                    "duplicates": r["duplicate_calls_blocked"],
                    "tool_calls": r["tool_calls"],
                    "tool_errors": r["tool_errors"],
                    "latency_ms": round(float(r["latency_ms"]), 2),
                } for r in rows]
    finally:
        runner.select_gaia_partition = saved_selector
    if [r["task_hash"] for r in by_arm["original"]] != [
        r["task_hash"] for r in by_arm["replay_success"]
    ]:
        raise ValueError("matched questions changed")
    return {"shard": index, "cohort_sha256": cohort_digest,
            "gaia_revision": GAIA_REVISION, "arms": by_arm}


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(x["shard"] for x in shards) != list(range(N_SHARDS)):
        raise ValueError("five unique shards required")
    first = shards[0]
    if any(s["gaia_revision"] != GAIA_REVISION
           or s["cohort_sha256"] != first["cohort_sha256"]
           or any(len(s["arms"].get(a, [])) != 5 for a in ARMS)
           for s in shards):
        raise ValueError("frozen cohort mismatch")
    rows = {arm: [r for s in shards for r in s["arms"][arm]] for arm in ARMS}
    before, after = rows["original"], rows["replay_success"]
    if len(before) != 25 or len({r["task_hash"] for r in before}) != 25 or (
        [r["task_hash"] for r in before] != [r["task_hash"] for r in after]
    ):
        raise ValueError("incomplete or changed matched cases")
    metrics = {}
    for arm, records in rows.items():
        metrics[arm] = {
            **{key: sum(r[key] for r in records) for key in (
                "correct", "completed", "duplicates", "tool_calls", "tool_errors",
            )},
            "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
            "mean_latency_s": round(sum(r["latency_ms"] for r in records)/25000, 2),
        }
    return {
        "n": 25, "partition": "diagnostic",
        "gaia_revision": GAIA_REVISION,
        "arms": metrics,
        "rescues": sum(not a["correct"] and b["correct"] for a, b in zip(before, after)),
        "regressions": sum(a["correct"] and not b["correct"] for a, b in zip(before, after)),
        "limitations": "Reused GAIA Diagnostic25; opt-in runtime experiment, not new model training",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        out = aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("REPLAY_GAIA25", json.dumps(out, sort_keys=True), flush=True)
    else:
        if args.shard is None:
            parser.error("specify --shard for inference")
        out = evaluate_shard(args.shard)
        print("REPLAY_SHARD_DONE", args.shard, flush=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
