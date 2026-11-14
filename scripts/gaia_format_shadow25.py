"""Paired original-v7 versus v7+generic-format-prompt on GAIA Shadow25.

Frozen GAIA partition: `shadow` (distinct from GAIA Evaluation100 and
the independent 15-case terminal suffix probe). Same v7 weights/runtime,
temperature 0, 12 tool steps. No training, no answer postprocessing.
Only hashed task IDs and numeric/categorical metrics leave the runner.
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
from gaia_small_agent.tools import default_tools
from scripts.gaia_format_reminder_ablation import FormatModel

ARM_KEYS = ("original", "format_reminder")
N_SHARDS = 5


def select_shadow(dataset: list[dict]) -> list[dict]:
    rows = select_gaia_partition(dataset, "shadow")
    other_ids = {
        r["task_id"] for partition in ("diagnostic", "evaluation")
        for r in select_gaia_partition(dataset, partition)
    }
    if len(rows) != 25 or len({r["task_id"] for r in rows}) != 25:
        raise ValueError("GAIA Shadow25 is incomplete or duplicated")
    if any(r["task_id"] in other_ids for r in rows):
        raise ValueError("GAIA Shadow25 overlaps historical diagnostic/evaluation")
    return rows


def replay(rows: list[dict], *, reminder: bool) -> list[dict]:
    original_selector = runner.select_gaia_partition
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            runner.select_gaia_partition = lambda *_args, **_kwargs: rows
            runner.run_gaia100(
                rows,
                lambda: AgentRuntime(
                    FormatModel(reminder=reminder), default_tools(), max_steps=12
                ),
                work, seed=LOCAL_PROTOCOL_SEED,
            )
        finally:
            runner.select_gaia_partition = original_selector
        results = [
            json.loads(line) for line in (work / "results.jsonl").read_text().splitlines()
        ]
    if len(results) != len(rows):
        raise ValueError("GAIA result length drift")
    return [{
        "task_hash": sha256(r["task_id"].encode()).hexdigest(),
        "level": int(r["level"]),
        "correct": bool(r["correct"]),
        "completed": bool(r["completed"]),
        "stop_reason": r["stop_reason"],
        "tool_calls": r["tool_calls"],
        "tool_errors": r["tool_errors"],
        "duplicate_calls_blocked": r["duplicate_calls_blocked"],
        "steps": r["steps"],
        "latency_ms": round(float(r["latency_ms"]), 3),
        "premature_final_proxy": "premature_final_proxy" in r["failure_signals"],
        "capability_gap": bool(r["capability_gap"]),
    } for r in results]


def run_shard(shard: int, dataset: list[dict]) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid GAIA shard")
    full = select_shadow(dataset)
    rows = shard_selected_rows(full, shard_index=shard, shard_count=N_SHARDS)
    if len(rows) != 5:
        raise ValueError("expected five GAIA cases per shard")
    results = {
        arm: replay(rows, reminder=(arm == "format_reminder"))
        for arm in ARM_KEYS
    }
    if [r["task_hash"] for r in results["original"]] != [
        r["task_hash"] for r in results["format_reminder"]
    ]:
        raise ValueError("paired case order drift")
    return {
        "shard": shard,
        "gaia_revision": GAIA_REVISION,
        "selection": "shadow25",
        "selected_identity_sha256": sha256(
            "|".join(str(r["task_id"]) for r in full).encode()
        ).hexdigest(),
        "arms": results,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("need five unique Shadow25 shards")
    first = shards[0]
    if any(s["gaia_revision"] != GAIA_REVISION
           or s["selection"] != "shadow25"
           or s["selected_identity_sha256"] != first["selected_identity_sha256"]
           or any(len(s["arms"].get(arm, [])) != 5 for arm in ARM_KEYS)
           for s in shards):
        raise ValueError("GAIA data/identity/config drift")
    by_arm = {
        arm: [r for s in sorted(shards, key=lambda s: s["shard"])
              for r in s["arms"][arm]]
        for arm in ARM_KEYS
    }
    original = by_arm["original"]
    changed = by_arm["format_reminder"]
    keys = [r["task_hash"] for r in original]
    if len(keys) != 25 or len(set(keys)) != 25 or keys != [r["task_hash"] for r in changed]:
        raise ValueError("missing or mismatched GAIA paired cases")
    totals = {}
    for arm, rows in by_arm.items():
        totals[arm] = {
            **{k: sum(r[k] for r in rows) for k in (
                "correct", "completed", "tool_calls", "tool_errors",
                "duplicate_calls_blocked", "premature_final_proxy",
                "capability_gap",
            )},
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
            "mean_latency_s": round(sum(r["latency_ms"] for r in rows) / 25000.0, 2),
        }
    return {
        "n": 25,
        "gaia_revision": GAIA_REVISION,
        "partition": "shadow",
        "arms": totals,
        "rescues": sum(not b["correct"] and a["correct"] for b, a in zip(original, changed)),
        "regressions": sum(b["correct"] and not a["correct"] for b, a in zip(original, changed)),
        "completion_rescues": sum(not b["completed"] and a["completed"]
                                   for b, a in zip(original, changed)),
        "completion_regressions": sum(b["completed"] and not a["completed"]
                                      for b, a in zip(original, changed)),
        "claim_boundary": "GAIA Shadow25 small external sample, no model update, no hidden benchmark",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.inputs:
        out = aggregate([json.loads(x.read_text()) for x in a.inputs])
    else:
        if a.shard is None:
            p.error("--shard is required")
        out = run_shard(a.shard, load_validation())
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    if a.inputs:
        print("GAIA_FORMAT_SHADOW25", json.dumps(out, sort_keys=True), flush=True)
    else:
        print("GAIA_FORMAT_SHARD_COMPLETED", a.shard, flush=True)


if __name__ == "__main__":
    main()
