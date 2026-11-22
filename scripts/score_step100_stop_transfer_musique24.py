"""Exact 24-pair pre-registered frozen MuSiQue transfer scorer, never a GAIA score."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

# Aggregation runs on a CPU-only job without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from scripts.eval_step100_stop_transfer_musique24 import (
    FROZEN_TASKS_SHA256, PER_SHARD, PROTOCOL, SHARDS, TASKS,
)

ALLOWED_STOP_REASONS = frozenset({
    "final", "invalid_final", "max_steps", "empty_final", "model_error",
    "model_timeout", "model_unavailable", "model_capacity",
})


def aggregate(evidence: Path) -> dict:
    grouped = {}
    for shard in range(SHARDS):
        for arm in ("control", "stop"):
            file = evidence / f"shard-{shard}-{arm}.json"
            record = json.loads(file.read_text(encoding="utf-8"))
            if (record["protocol"] != PROTOCOL
                or record["shard"] != shard or record["arm"] != arm
                or record["n_tasks"] != PER_SHARD
                or record["frozen_tasks_sha256"] != FROZEN_TASKS_SHA256
                or record["only_variable"] != "evidence_stop_contract"
                or record["max_steps"] != 12 or record["max_new_tokens"] != 512
                or record["v8_training_parent_overlap"] != 0
                or record["tool_names"] != ["search", "read", "inspect", "python"]
                or record["no_task_content_or_answers_exported"] is not True
                or len(record["cases"]) != PER_SHARD):
                raise ValueError(f"Protocol/source mismatch in {file.name}")
            d = {row["id_sha256"]: row for row in record["cases"]}
            if len(d) != PER_SHARD or any(
                row["stop_reason"] not in ALLOWED_STOP_REASONS or
                type(row["correct"]) is not bool or
                type(row["completed"]) is not bool or
                row["hop_count"] not in (3, 4) or
                row["tool_calls"] < 0 or row["steps"] not in range(1, 13) or
                row["elapsed_s"] < 0
                for row in d.values()
            ):
                raise ValueError(f"Incomplete/invalid per-task evidence: {file.name}")
            grouped[(shard, arm)] = d
    cases = []
    seen = set()
    for shard in range(SHARDS):
        a, b = grouped[(shard, "control")], grouped[(shard, "stop")]
        if a.keys() != b.keys() or seen.intersection(a):
            raise ValueError("Paired task identity mismatch or repeated tasks between shards")
        seen.update(a)
        for ident in sorted(a):
            old, new = a[ident], b[ident]
            if old["hop_count"] != new["hop_count"]:
                raise ValueError("Hops changed between matched arms")
            cases.append((old, new))
    if len(seen) != TASKS:
        raise ValueError("Missing frozen MuSiQue holdout task")

    b_correct = sum(int(x["correct"]) for x, _ in cases)
    s_correct = sum(int(y["correct"]) for _, y in cases)
    rescue = sum(int(not x["correct"] and y["correct"]) for x, y in cases)
    regression = sum(int(x["correct"] and not y["correct"]) for x, y in cases)
    if s_correct - b_correct != rescue - regression:
        raise AssertionError("Paired counts fail conservation")
    base_calls = sum(x["tool_calls"] for x, _ in cases)
    stop_calls = sum(y["tool_calls"] for _, y in cases)
    by_hop = {}
    for hop in (3, 4):
        subset = [(a, b) for a, b in cases if a["hop_count"] == hop]
        by_hop[str(hop)] = {
            "n": len(subset),
            "control_correct": sum(a["correct"] for a, _ in subset),
            "stop_correct": sum(b["correct"] for _, b in subset),
            "rescues": sum(not a["correct"] and b["correct"] for a, b in subset),
            "regressions": sum(a["correct"] and not b["correct"] for a, b in subset),
        }
    gate = {
        "net_gain_at_least_3": s_correct-b_correct >= 3,
        "regressions_at_most_1": regression <= 1,
        "rescue_in_both_hop_groups": all(x["rescues"] >= 1 for x in by_hop.values()),
        "tool_calls_at_most_125pct": stop_calls * 4 <= base_calls * 5,
    }
    return {
        "protocol": PROTOCOL,
        "source": "MuSiQue TRAIN-origin fixed 24 parent holdout; no GAIA rows",
        "task_count": TASKS,
        "frozen_tasks_sha256": FROZEN_TASKS_SHA256,
        "control_correct": b_correct,
        "stop_correct": s_correct,
        "net_gain": s_correct-b_correct,
        "rescues": rescue,
        "regressions": regression,
        "control_completed": sum(x["completed"] for x, _ in cases),
        "stop_completed": sum(y["completed"] for _, y in cases),
        "control_invalid_final": sum(x["stop_reason"]=="invalid_final" for x,_ in cases),
        "stop_invalid_final": sum(y["stop_reason"]=="invalid_final" for _,y in cases),
        "control_tool_calls": base_calls,
        "stop_tool_calls": stop_calls,
        "control_total_latency_s": round(sum(x["elapsed_s"] for x,_ in cases), 2),
        "stop_total_latency_s": round(sum(y["elapsed_s"] for _,y in cases), 2),
        "by_hop": by_hop,
        "preregistered_transfer_gates": gate,
        "transfer_gate_pass": all(gate.values()),
        "promotion_to_main_or_GAIA_claim": False,
        "limitation": "Twenty-four MuSiQue TRAIN-derived held-out multi-hop tasks, not independent GAIA or unseen-source-family proof; exploratory transfer only. Never tune on this same holdout.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing overwrite of frozen transfer result")
    report = aggregate(args.evidence_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("MUSIQUE24_PREREG_CLOSED_LOOP_TRANSFER", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
