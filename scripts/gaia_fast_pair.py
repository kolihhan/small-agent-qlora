"""Paired GAIA100 fast finalization experiment. No training or benchmark modifications.

Raw GAIA question/answer/trace files are private and temporary on the runner.
Only per-task SHA256 identifiers and scored metadata are retained as artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import check_default_tools, default_tools

ARMS = ("baseline", "candidate")
N_SHARDS = 20
MAX_STEPS = 12
EXPECTED_IDS_SHA256 = "869e24fd53a0eeabc2598d952d0b5f893eca279584c96eb407a72906dccba45a"


def fixed_rows(dataset):
    rows = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    digest = hashlib.sha256("|".join(str(r["task_id"]) for r in rows).encode()).hexdigest()
    if (len(rows) != 100 or len({r["task_id"] for r in rows}) != 100
            or Counter(int(r["Level"]) for r in rows) != {1: 32, 2: 52, 3: 16}
            or digest != EXPECTED_IDS_SHA256):
        raise ValueError("Frozen Evaluation100 identity/order/levels changed")
    return rows


def safe_record(row):
    return {
        "task_hash": hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
        "level": int(row["level"]),
        "correct": bool(row["correct"]),
        "completed": bool(row["completed"]),
        "stop_reason": str(row["stop_reason"]),
        "invalid_final": row["stop_reason"] == "invalid_final",
        "max_steps_stop": row["stop_reason"] == "max_steps",
        "steps": int(row["steps"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_errors": int(row["tool_errors"]),
        "duplicate_calls_blocked": int(row["duplicate_calls_blocked"]),
        "latency_ms": round(float(row["latency_ms"]), 3),
    }


def evaluate(rows, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    previous_selection = runner.select_gaia_partition
    with TemporaryDirectory(prefix="private-gaia-fast-") as root:
        work = Path(root)
        try:
            runner.select_gaia_partition = lambda *_args, **_kwargs: list(rows)
            summary = runner.run_gaia100(
                rows,
                lambda: AgentRuntime(
                    OpenAICompatModel("qwen3.5-9b", base_url="http://127.0.0.1:8080",
                                      timeout_s=180, max_new_tokens=512),
                    default_tools(), max_steps=MAX_STEPS,
                    finalization_nudge=(arm == "candidate"),
                ),
                work, seed=LOCAL_PROTOCOL_SEED, partition="evaluation",
                dataset_revision=GAIA_REVISION,
            )
        finally:
            runner.select_gaia_partition = previous_selection
        if int(summary["total"]) != len(rows):
            raise RuntimeError("Incomplete GAIA shard")
        records = [json.loads(line) for line in
                   (work / "results.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        if [r["task_id"] for r in records] != [r["task_id"] for r in rows]:
            raise RuntimeError("Scored task identities/order changed")
        return [safe_record(r) for r in records]


def run_shard(shard):
    if shard not in range(N_SHARDS):
        raise ValueError("Shard outside 0..19")
    with TemporaryDirectory(prefix="gaia-tool-check-") as tmp:
        check = check_default_tools(tmp, live_search=False)
        if not check["ready"]:
            raise RuntimeError("Tool preflight failed: " + str(check["checks"]))
    rows = fixed_rows(load_validation(token=os.environ.get("HF_TOKEN")))
    subset = shard_selected_rows(rows, shard_index=shard, shard_count=N_SHARDS)
    if len(subset) != 5:
        raise ValueError("Each shard must contain five cases")
    arms = {name: evaluate(subset, name) for name in ARMS}
    if [r["task_hash"] for r in arms["baseline"]] != [r["task_hash"] for r in arms["candidate"]]:
        raise ValueError("Unmatched paired tasks")
    return {
        "schema": "gaia-fast-v3b-paired/v1", "shard": shard,
        "source_revision": GAIA_REVISION,
        "selection": "gaia-evaluation100-seen-dev",
        "full_ids_sha256": EXPECTED_IDS_SHA256, "seed": LOCAL_PROTOCOL_SEED,
        "max_steps": MAX_STEPS, "temperature": 0, "reasoning_budget": 0,
        "same_v3b_adapter": True, "no_question_answer_trace_export": True,
        "arms": arms,
    }


def stats(rows):
    return {
        "correct": sum(r["correct"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "invalid_final": sum(r["invalid_final"] for r in rows),
        "max_steps_stop": sum(r["max_steps_stop"] for r in rows),
        "mean_steps": round(sum(r["steps"] for r in rows) / len(rows), 3),
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "mean_latency_s": round(sum(r["latency_ms"] for r in rows) / len(rows) / 1000, 3),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
    }


def aggregate(shards):
    if len(shards) != N_SHARDS or sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("Require complete 20-shard comparison")
    arms = {name: [] for name in ARMS}
    for shard in sorted(shards, key=lambda s: s["shard"]):
        if (shard.get("schema") != "gaia-fast-v3b-paired/v1"
                or shard.get("source_revision") != GAIA_REVISION
                or shard.get("selection") != "gaia-evaluation100-seen-dev"
                or shard.get("full_ids_sha256") != EXPECTED_IDS_SHA256
                or shard.get("seed") != LOCAL_PROTOCOL_SEED
                or shard.get("max_steps") != MAX_STEPS
                or shard.get("same_v3b_adapter") is not True
                or shard.get("no_question_answer_trace_export") is not True
                or set(shard.get("arms", {})) != set(ARMS)):
            raise ValueError("Protocol/adapter/source mismatch")
        for name in ARMS:
            if len(shard["arms"][name]) != 5:
                raise ValueError("Incomplete five-task arm")
            arms[name].extend(shard["arms"][name])
    orig, cand = arms["baseline"], arms["candidate"]
    original_ids = [r["task_hash"] for r in orig]
    if len(original_ids) != 100 or len(set(original_ids)) != 100:
        raise ValueError("Incomplete or repeated 100 case cohort")
    if original_ids != [r["task_hash"] for r in cand]:
        raise ValueError("Baseline/candidate mismatch")
    cases = []
    for b, c in zip(orig, cand):
        direction = ("win" if c["correct"] and not b["correct"] else
                     "loss" if b["correct"] and not c["correct"] else
                     "both_correct" if b["correct"] else "both_wrong")
        cases.append({"task_hash": b["task_hash"], "level": b["level"],
                      "direction": direction, "baseline": b, "candidate": c})
    return {
        "status": "COMPLETE_GAIA100_FAST_SCREEN_ONLY",
        "source_revision": GAIA_REVISION, "full_ids_sha256": EXPECTED_IDS_SHA256,
        "seed": LOCAL_PROTOCOL_SEED, "n": 100,
        "not_independent_heldout": True, "no_training": True,
        "arms": {name: stats(rows) for name, rows in arms.items()},
        "wins": [r for r in cases if r["direction"] == "win"],
        "losses": [r for r in cases if r["direction"] == "loss"],
        "case_table": cases,
    }


def markdown_report(result):
    b = result["arms"]["baseline"]
    c = result["arms"]["candidate"]
    lines = [
        "# GAIA100 v3B: fast terminal nudge (one paired run)", "",
        "Same v3B adapter, pinned model, tools and 12-step limit, temp=0.",
        "Seen GAIA development cohort; no held-out claim or training.", "",
        "| Metric | v3B baseline | Candidate |", "|---|---:|---:|",
    ]
    for key in ("correct", "completed", "invalid_final", "max_steps_stop",
                "mean_steps", "tool_calls", "tool_errors", "mean_latency_s"):
        lines.append(f"| {key} | {b[key]} | {c[key]} |")
    for direction in ("wins", "losses"):
        lines += ["", f"## Candidate {direction} ({len(result[direction])})", "",
                  "| SHA256(task_id) | Level | Baseline stop | Candidate stop |",
                  "|---|---:|---|---|"]
        for row in result[direction]:
            lines.append(f"| {row['task_hash']} | {row['level']} | "
                         f"{row['baseline']['stop_reason']} | "
                         f"{row['candidate']['stop_reason']} |")
    lines += ["", "All 100 task-level results are in the paired JSON artifact.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", type=int)
    group.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--markdown", type=Path)
    a = p.parse_args()
    if a.inputs:
        result = aggregate([json.loads(x.read_text(encoding="utf-8")) for x in a.inputs])
        if a.markdown:
            a.markdown.parent.mkdir(parents=True, exist_ok=True)
            a.markdown.write_text(markdown_report(result), encoding="utf-8")
    else:
        if a.markdown:
            p.error("--markdown only valid with --inputs")
        result = run_shard(a.shard)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise SystemExit("Output exists; refusing to overwrite")
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if a.inputs:
        print("GAIA_FAST_COMPLETE", json.dumps({
            "baseline_correct": result["arms"]["baseline"]["correct"],
            "candidate_correct": result["arms"]["candidate"]["correct"],
            "wins": len(result["wins"]), "losses": len(result["losses"]),
            "baseline_invalid_final": result["arms"]["baseline"]["invalid_final"],
            "candidate_invalid_final": result["arms"]["candidate"]["invalid_final"],
        }))
    else:
        print("GAIA_FAST_SHARD_COMPLETE", a.shard)


if __name__ == "__main__":
    main()
