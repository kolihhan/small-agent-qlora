"""Read-only, paired GAIA100 v8-vs-base regression audit.
Never export raw model answers, prompts, tool arguments or tool observations.
Labels are observational proxies only: cannot infer root cause from these logs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

EXPECTED_N = 100
EXPECTED_BASE = 20
EXPECTED_V8 = 21
EXPECTED_RESCUES = 8
EXPECTED_REGRESSIONS = 7

def arm_rows(root: Path, arm: str, shards: int):
    paths = sorted(root.glob("gaia-v8-paired-shard-*/" + arm + "/shard-*/results.jsonl"))
    if len(paths) != shards:
        raise ValueError(f"{arm}: expected {shards} shards, found {len(paths)}")
    results = {}
    for path in paths:
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            taskid = str(row["task_id"])
            if taskid in results:
                raise ValueError("duplicate task within arm")
            required = {"task_id", "correct", "tool_calls", "tool_errors", "stop_reason",
                        "completed", "steps", "trace", "failure_label", "capability_gap"}
            if not required <= set(row):
                raise ValueError("incomplete trace schema")
            if not isinstance(row["trace"], list):
                raise ValueError("trace is not a list")
            results[taskid] = row
    return results

def trace_signals(row):
    tools, errors = [], Counter()
    for event in row["trace"]:
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        data = event.get("data") or {}
        if kind == "tool_call":
            tools.append(str(data.get("name") or "unknown"))
        if kind == "tool_result" and data.get("ok") is False:
            errors[str(data.get("error_code") or "UNKNOWN")] += 1
    if len(tools) != int(row["tool_calls"]):
        raise ValueError("recorded tool call count mismatch")
    return tools, dict(sorted(errors.items()))

def public_metrics(row):
    tools, errors = trace_signals(row)
    return {
        "correct": bool(row["correct"]),
        "completed": bool(row["completed"]),
        "stop_reason": row["stop_reason"],
        "failure_label": row["failure_label"],
        "capability_gap": row.get("capability_gap"),
        "steps": int(row["steps"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_errors": int(row["tool_errors"]),
        "duplicate_calls_blocked": int(row.get("duplicate_calls_blocked", 0)),
        "tool_sequence": tools,
        "tool_error_codes": errors,
    }

def paired_test_pvalue(rescues, regressions):
    n = rescues + regressions
    if n == 0:
        return 1.
    one_tail = sum(math.comb(n, j) for j in range(min(rescues, regressions) + 1)) / (2 ** n)
    return min(1., 2 * one_tail)

def analyze(root: Path, expected_n=EXPECTED_N, shard_count=20, assert_frozen=True):
    base = arm_rows(root, "base", shard_count)
    v8 = arm_rows(root, "qlora-v8", shard_count)
    if set(base) != set(v8) or len(base) != expected_n:
        raise ValueError("matched GAIA identity incomplete")
    groups = Counter()
    transitions = Counter()
    error_code_changes = Counter()
    matched = []
    for tid in sorted(base):
        old, new = public_metrics(base[tid]), public_metrics(v8[tid])
        if not old["correct"] and new["correct"]:
            change = "rescue"
        elif old["correct"] and not new["correct"]:
            change = "regression"
        elif old["correct"]:
            change = "both_correct"
        else:
            change = "both_wrong"
        groups[change] += 1
        transitions[(old["failure_label"], new["failure_label"], change)] += 1
        if change in {"rescue", "regression"}:
            code = "unchanged_route" if old["tool_sequence"] == new["tool_sequence"] else "changed_route"
            if not old["tool_calls"] and new["tool_calls"]:
                code = "added_tool_calls"
            elif old["tool_calls"] and not new["tool_calls"]:
                code = "removed_tool_calls"
            error_code_changes[(change, code)] += 1
            matched.append({
                "task_sha256_16": sha256(tid.encode()).hexdigest()[:16],
                "outcome": change,
                "base": old, "v8": new,
                "tool_call_delta": new["tool_calls"] - old["tool_calls"],
                "steps_delta": new["steps"] - old["steps"],
                "interpretation": "Descriptive signal only: root cause requires content-aware review.",
            })
    score_base = groups["rescue"] + groups["both_correct"]
    score_v8 = groups["regression"] + groups["both_correct"]
    if assert_frozen and (score_base, score_v8, groups["rescue"], groups["regression"]) != (
        EXPECTED_BASE, EXPECTED_V8, EXPECTED_RESCUES, EXPECTED_REGRESSIONS
    ):
        raise ValueError("original GAIA v8 scored summary differs from downloaded traces")
    report = {
        "schema_version": "v8-gaia100-observational-paired-audit/v1",
        "run_id": 44980055775, "n": expected_n, "shards_per_arm": shard_count,
        "no_new_model_calls": True, "no_training": True,
        "privacy": "No raw prompts, final answers, tool arguments, or observations exported.",
        "base_correct": score_base, "v8_correct": score_v8,
        "outcomes": dict(sorted(groups.items())),
        "mcnemar_two_sided_exact_p": paired_test_pvalue(groups["rescue"], groups["regression"]),
        "failure_label_transitions": [
            {"base": a, "v8": b, "outcome": change, "n": n}
            for (a,b,change),n in sorted(transitions.items())
        ],
        "route_change_signals": [
            {"outcome": outcome, "signal": sig, "n": n}
            for (outcome,sig),n in sorted(error_code_changes.items())
        ],
        "changed_15_cases": matched,
        "warning": "Tool use/finalization labels cannot identify why correctness changed; manual trajectory review required.",
        "promotion_allowed": False,
    }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    report=analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n", "utf-8")
    print("GAIA_V8_REGRESSION_AUDIT", json.dumps({
        "base": report["base_correct"], "v8": report["v8_correct"],
        "rescues": report["outcomes"].get("rescue", 0),
        "regressions": report["outcomes"].get("regression", 0),
        "exact_mcnemar_p": report["mcnemar_two_sided_exact_p"],
        "route_signals": report["route_change_signals"],
    }), flush=True)

if __name__ == "__main__":
    main()
