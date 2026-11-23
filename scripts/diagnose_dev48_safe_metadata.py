"""CPU-only failure taxonomy over two frozen MuSiQue DEV24 paired result sets.

Never touch raw task content, model answers, labels beyond bool correctness or private traces.
Not a causal or supporting-evidence diagnosis: prior runs intentionally export no tool traces.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SOURCE_SHA = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SPECS = {
    "dev24a": {"protocol": "STEP100-WORKSPACE-FIND-MUSIQUE-DEV24-V1",
               "manifest": "e43aec02a481de51baeff1dc5cba67c9e64eb7eb10d463562c2bc52f78006591",
               "candidate": "find", "only_variable": "opt_in_workspace_find"},
    "dev24b": {"protocol": "STEP100-SOURCE-ANCHORED-STATE-MUSIQUE-DEV24B-V1",
               "manifest": "ac52a5682aa70c72ee0e8027c1e1f0eb62b0f5fdafb78ca53053efb73c411814",
               "candidate": "state", "only_variable": "opt_in_source_anchored_evidence_state"},
}


def load_pair(folder: Path, name: str):
    spec = SPECS[name]
    pairs = []
    seen = set()
    for shard in range(4):
        arms = {}
        for arm in ("control", spec["candidate"]):
            path = folder / f"shard-{shard}-{arm}.json"
            obj = json.loads(path.read_text(encoding="utf8"))
            if (obj.get("protocol") != spec["protocol"] or obj.get("n") != 6
                or obj.get("shard") != shard or obj.get("arm") != arm
                or obj.get("manifest_sha256") != spec["manifest"]
                or obj.get("source_sha256") != SOURCE_SHA
                or obj.get("only_variable") != spec["only_variable"]
                or obj.get("adapter") != "verified_qwen3.5_9b_v8_step100"
                or obj.get("max_steps") != 12 or obj.get("max_new_tokens") != 512
                or obj.get("answer_label_used_in_decisions") is not False
                or obj.get("no_task_or_answers_exported") is not True):
                raise ValueError(f"{name} frozen integrity mismatch at shard {shard}, arm {arm}")
            if name == "dev24b" and obj.get("same_four_tools") is not True:
                raise ValueError("DEV24B unexpectedly changed tools")
            rows = obj["cases"]
            by_id = {r["id_sha256"]: r for r in rows}
            if len(rows) != 6 or len(by_id) != 6:
                raise ValueError(f"{name}: incomplete or duplicated shard")
            for r in rows:
                if (type(r["correct"]) is not bool or type(r["completed"]) is not bool
                    or r.get("hop_count") not in (3, 4)
                    or r.get("steps") not in range(1, 13)
                    or r.get("tool_calls", -1) < 0 or r.get("tool_errors", -1) < 0
                    or r.get("elapsed_s", -1) < 0):
                    raise ValueError("Malformed safe case metadata")
            arms[arm] = by_id
        if arms["control"].keys() != arms[spec["candidate"]].keys():
            raise ValueError("Not paired")
        if seen.intersection(arms["control"]):
            raise ValueError("Duplicate identity across shards")
        seen.update(arms["control"])
        pairs.extend((arms["control"][k], arms[spec["candidate"]][k])
                     for k in sorted(arms["control"]))
    if len(seen) != 24:
        raise ValueError("Expected 24 unique task IDs")
    return pairs, seen


def describe(rows: list[dict]) -> dict:
    stops = Counter(r["stop_reason"] for r in rows)
    if sum(stops.values()) != len(rows):
        raise AssertionError("Invalid stop accounting")
    calls = [r["tool_calls"] for r in rows]
    errors = [r["tool_errors"] for r in rows]
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "completed_wrong": sum(r["completed"] and not r["correct"] for r in rows),
        "wrong_without_tool_error": sum(not r["correct"] and r["tool_errors"] == 0 for r in rows),
        "no_tool_calls": sum(x == 0 for x in calls),
        "step_budget_exhausted_or_final_step": sum(r["steps"] == 12 for r in rows),
        "total_tool_calls": sum(calls),
        "total_tool_errors": sum(errors),
        "total_duplicate_calls_blocked": sum(r.get("duplicate_calls", 0) for r in rows),
        "total_elapsed_seconds": round(sum(r["elapsed_s"] for r in rows), 2),
        "stop_reasons": dict(sorted(stops.items())),
        "by_hop": {
            str(h): {
                "n": sum(r["hop_count"] == h for r in rows),
                "correct": sum(r["correct"] and r["hop_count"] == h for r in rows),
                "invalid_final": sum(r["stop_reason"] == "invalid_final"
                                     and r["hop_count"] == h for r in rows),
                "max_steps": sum(r["stop_reason"] == "max_steps"
                                 and r["hop_count"] == h for r in rows),
            } for h in (3, 4)
        },
    }


def diagnose(a: Path, b: Path) -> dict:
    pa, ida = load_pair(a, "dev24a")
    pb, idb = load_pair(b, "dev24b")
    if ida.intersection(idb):
        raise ValueError("DEV24A/B must not share task identities")
    groups = {
        "dev24a": {
            "control": describe([x for x, _ in pa]),
            "workspace_find": describe([x for _, x in pa]),
        },
        "dev24b": {
            "control": describe([x for x, _ in pb]),
            "evidence_state": describe([x for _, x in pb]),
        },
    }
    total_base = sum(groups[g]["control"]["correct"] for g in groups)
    if total_base != 0:
        raise ValueError("Not frozen 0/48 controls")
    return {
        "protocol": "STEP100-FROZEN-DEV48-SAFE-METADATA-ROOT-CAUSE-TRIAGE-V1",
        "source_run_ids": [45100549744, 45104088600, 45104501119],
        "n_unique_tasks": 48,
        "both_original_control_zero_correct": True,
        "groups": groups,
        "trace_exported": False,
        "answer_or_gold_exported": False,
        "model_inference": False,
        "limits": [
            "No full tool-call traces in these safe JSONs: cannot identify which tools failed or whether supporting evidence was read.",
            "Wrong without tool error does not prove a semantic reasoning failure.",
            "Step 12 may be required final-response step, not proof of tool budget exhaustion.",
            "These are selected MuSiQue DEV tasks, not independent GAIA validation.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev24a", type=Path, required=True)
    p.add_argument("--dev24b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = diagnose(a.dev24a, a.dev24b)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError("Refuse overwrite")
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("DEV48_SAFE_FAILURE_TRIAGE", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
