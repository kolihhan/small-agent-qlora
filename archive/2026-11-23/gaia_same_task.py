"""Same-task Qwen3.5-9B GAIA100 comparison; keep raw GAIA data on the runner.

All arms use the same benchmark scorer, task identity, agent loop and tools.
Only original, checksum-verified GGUF LoRA adapter differs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
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

ARMS = ("base", "v3d", "v7", "v8step70")
REQUIRED_ARMS = ("base", "v7", "v8step70")
N_SHARDS = 20
MAX_STEPS = 12
EXPECTED_IDS_SHA256 = "869e24fd53a0eeabc2598d952d0b5f893eca279584c96eb407a72906dccba45a"


def fixed_rows(dataset):
    rows = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    ids = [str(r["task_id"]) for r in rows]
    digest = sha256("|".join(ids).encode()).hexdigest()
    if (
        len(rows) != 100 or len(set(ids)) != 100
        or Counter(int(r["Level"]) for r in rows) != {1: 32, 2: 52, 3: 16}
        or digest != EXPECTED_IDS_SHA256
    ):
        raise RuntimeError("GAIA100 source IDs, level quotas or order drifted")
    return rows


def safe_row(row):
    return {
        "task_hash": sha256(str(row["task_id"]).encode()).hexdigest(),
        "level": int(row["level"]),
        "correct": bool(row["correct"]),
        "completed": bool(row["completed"]),
        "stop_reason": str(row["stop_reason"]),
        "steps": int(row["steps"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_errors": int(row["tool_errors"]),
        "duplicate_blocks": int(row["duplicate_calls_blocked"]),
        "latency_ms": round(float(row["latency_ms"]), 2),
    }


def run_shard(shard_index: int, arm: str) -> dict:
    if arm not in ARMS or shard_index not in range(N_SHARDS):
        raise ValueError("invalid arm or shard")
    rows = fixed_rows(load_validation(token=os.getenv("HF_TOKEN")))
    shard = shard_selected_rows(rows, shard_index=shard_index, shard_count=N_SHARDS)
    if len(shard) != 5:
        raise RuntimeError("expected exactly 5 tasks per shard")
    with TemporaryDirectory(prefix="gaia-private-same-task-") as tmp:
        previous = runner.select_gaia_partition
        try:
            runner.select_gaia_partition = lambda *_args, **_kwargs: list(shard)
            summary = runner.run_gaia100(
                shard,
                lambda: AgentRuntime(
                    OpenAICompatModel(
                        "qwen3.5-9b", base_url="http://127.0.0.1:8080",
                        timeout_s=180, max_new_tokens=512,
                    ),
                    default_tools(), max_steps=MAX_STEPS,
                ),
                tmp,
                seed=LOCAL_PROTOCOL_SEED,
                partition="evaluation",
                dataset_revision=GAIA_REVISION,
            )
        finally:
            runner.select_gaia_partition = previous
        records = [
            json.loads(s) for s in (Path(tmp) / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        if summary["total"] != 5 or len(records) != 5:
            raise RuntimeError("incomplete scored five-task shard")
        if [r["task_id"] for r in records] != [r["task_id"] for r in shard]:
            raise RuntimeError("scored task order drift")
        return {
            "schema": "gaia-fourway-same-task/v1",
            "arm": arm, "shard": shard_index,
            "dataset_revision": GAIA_REVISION, "seed": LOCAL_PROTOCOL_SEED,
            "full_ids_sha256": EXPECTED_IDS_SHA256,
            "runtime": {
                "steps": MAX_STEPS, "temperature": 0, "reasoning_budget": 0,
                "context_tokens": 16384, "max_new_tokens": 512,
                "tools": ["search", "read", "inspect", "python"],
            },
            "cases": [safe_row(r) for r in records],
        }


def stats(rows: list[dict]) -> dict:
    return {
        "correct": sum(bool(r["correct"]) for r in rows),
        "completed": sum(bool(r["completed"]) for r in rows),
        "invalid_final": sum(r["stop_reason"] == "invalid_final" for r in rows),
        "max_steps": sum(r["stop_reason"] == "max_steps" for r in rows),
        "mean_steps": round(sum(r["steps"] for r in rows) / len(rows), 3),
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "duplicate_blocks": sum(r["duplicate_blocks"] for r in rows),
        "mean_latency_seconds": round(sum(r["latency_ms"] for r in rows) / len(rows) / 1000, 3),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
    }


def aggregate(paths: list[Path]) -> dict:
    files = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    found: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for item in files:
        arm, shard = item.get("arm"), item.get("shard")
        if (
            item.get("schema") != "gaia-fourway-same-task/v1"
            or arm not in ARMS or not isinstance(shard, int) or shard not in range(N_SHARDS)
            or shard in found[arm] or len(item.get("cases", [])) != 5
            or item.get("dataset_revision") != GAIA_REVISION
            or item.get("full_ids_sha256") != EXPECTED_IDS_SHA256
            or item.get("seed") != LOCAL_PROTOCOL_SEED
            or item.get("runtime") != {
                "steps": 12, "temperature": 0, "reasoning_budget": 0,
                "context_tokens": 16384, "max_new_tokens": 512,
                "tools": ["search", "read", "inspect", "python"],
            }
        ):
            raise ValueError("invalid or duplicated source, arm, shard or runtime")
        found[arm][shard] = item

    for arm in REQUIRED_ARMS:
        if set(found[arm]) != set(range(N_SHARDS)):
            raise ValueError("required arm incomplete: " + arm)
    if found["v3d"] and set(found["v3d"]) != set(range(N_SHARDS)):
        raise ValueError("partial v3D cohort is not accepted")

    by_arm = {
        arm: [
            case for shard in range(N_SHARDS)
            for case in found[arm][shard]["cases"]
        ]
        for arm in ARMS if found[arm]
    }
    for arm, rows in by_arm.items():
        ids = [x["task_hash"] for x in rows]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError("missing/duplicated GAIA task identities in " + arm)
        if ids != [x["task_hash"] for x in by_arm["base"]]:
            raise ValueError("arms did not evaluate exact same tasks")

    base = by_arm["base"]
    paired = {}
    for arm, rows in by_arm.items():
        if arm == "base":
            continue
        wins, losses = [], []
        for b, c in zip(base, rows):
            if not b["correct"] and c["correct"]:
                wins.append({"task_hash": b["task_hash"], "level": b["level"]})
            if b["correct"] and not c["correct"]:
                losses.append({"task_hash": b["task_hash"], "level": b["level"]})
        paired[arm] = {"wins": wins, "losses": losses}
    return {
        "status": "COMPLETE_AVAILABLE_ARMS_ONLY" if "v3d" not in by_arm else "COMPLETE_FOUR_ARMS",
        "scope": "GAIA2023 validation frozen Evaluation100, previously seen; single run",
        "dataset_revision": GAIA_REVISION,
        "full_ids_sha256": EXPECTED_IDS_SHA256,
        "n": 100, "no_new_training": True,
        "missing_versions": [a for a in ARMS if a not in by_arm],
        "arm_metrics": {a: stats(x) for a, x in by_arm.items()},
        "paired_vs_base": paired,
        "per_task": [
            {
                "task_hash": x["task_hash"], "level": x["level"],
                "arms": {a: rows[i] for a, rows in by_arm.items()},
            }
            for i, x in enumerate(base)
        ],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Qwen3.5-9B GAIA100 same-task comparison", "",
        "One physical comparison with pinned Q4 base, llama.cpp, 12 steps, identical tools/prompt/scorer.",
        "GAIA100 is a previously seen development cohort, not unseen held-out.", "",
        "| Metric | " + " | ".join(report["arm_metrics"]) + " |",
        "|---|" + "|".join("---:" for _ in report["arm_metrics"]) + "|",
    ]
    for name in ("correct", "completed", "invalid_final", "max_steps", "mean_steps",
                 "tool_calls", "tool_errors", "duplicate_blocks", "mean_latency_seconds"):
        lines.append("| " + name + " | " +
                     " | ".join(str(s[name]) for s in report["arm_metrics"].values()) + " |")
    if report["missing_versions"]:
        lines.extend(["", "Missing arms: " + ", ".join(report["missing_versions"]) +
                      ". Do not describe this as a complete four-arm experiment."])
    for arm, p in report["paired_vs_base"].items():
        lines.extend(["", "## " + arm + " vs base: " + str(len(p["wins"])) +
                      " wins, " + str(len(p["losses"])) + " losses",
                      "", "| Direction | Task SHA256 | Level |", "|---|---|---:|"])
        for direction in ("wins", "losses"):
            for item in p[direction]:
                lines.append("| " + direction + " | " + item["task_hash"] + " | " +
                             str(item["level"]) + " |")
    lines.extend(["", "Full per-task hashed records: summary JSON.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard", type=int)
    mode.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.inputs:
        if args.arm:
            parser.error("--arm only valid with --shard")
        result = aggregate(args.inputs)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown_report(result), encoding="utf-8")
        print("SAME_TASK_SUMMARY", json.dumps({
            "status": result["status"], "correct": {
                k: v["correct"] for k, v in result["arm_metrics"].items()
            }, "missing_versions": result["missing_versions"],
        }), flush=True)
    else:
        if not args.arm:
            parser.error("--arm required with --shard")
        result = run_shard(args.shard, args.arm)
        print("SAME_TASK_SHARD_COMPLETE", args.shard, args.arm, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite existing output")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
