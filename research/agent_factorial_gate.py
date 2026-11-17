"""Fail-closed 2x2 Base/Adapter x Native/Serializer result audit.

Input: JSON with model_digest, adapter_sha256, source_manifest_sha256 and
tasks: [{task_sha256, family, arms: {base_native, base_serializer,
adapter_native, adapter_serializer}}]. Each arm has correct (bool),
model_calls (int >= 0), elapsed_s (number >= 0).

CPU-only descriptive analysis. This never runs a model or promotes an adapter.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

BASE_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ARMS = ("base_native", "base_serializer", "adapter_native", "adapter_serializer")
PAIRS = (("base_native", "adapter_native"),
         ("base_native", "base_serializer"),
         ("base_serializer", "adapter_serializer"),
         ("adapter_native", "adapter_serializer"))

def _sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))

def _mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

def audit(payload):
    blockers = []
    if payload.get("model_digest") != BASE_DIGEST:
        blockers.append("pinned base digest mismatch")
    for key in ("adapter_sha256", "source_manifest_sha256"):
        if not _sha(payload.get(key)):
            blockers.append(f"{key} must be a 64-character hash")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        blockers.append("frozen per-task 2x2 trajectories unavailable")
        return {"status": "BLOCKED", "blockers": blockers, "new_model_inference": False}
    if len(tasks) < 24:
        blockers.append(f"pilot requires >=24 distinct tasks, got {len(tasks)}")
    seen, families = set(), Counter()
    valid = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            blockers.append(f"task #{idx}: not an object")
            continue
        task_sha = task.get("task_sha256")
        if not _sha(task_sha) or task_sha in seen:
            blockers.append(f"task #{idx}: invalid/duplicate task hash")
        seen.add(task_sha)
        family = task.get("family")
        if not isinstance(family, str) or not family.strip():
            blockers.append(f"task #{idx}: missing source family")
        else:
            families[family] += 1
        arms = task.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            blockers.append(f"task #{idx}: require exact four matched arms")
            continue
        good = True
        for name in ARMS:
            row = arms[name]
            if (not isinstance(row, dict) or type(row.get("correct")) is not bool or
                type(row.get("model_calls")) is not int or row["model_calls"] < 0 or
                not isinstance(row.get("elapsed_s"), (int, float)) or
                isinstance(row["elapsed_s"], bool) or row["elapsed_s"] < 0):
                blockers.append(f"task #{idx}: invalid arm {name}")
                good = False
        if good:
            valid.append(arms)
    if len(families) < 2:
        blockers.append("at least two task families required")
    if blockers:
        return {"status": "BLOCKED", "blockers": blockers,
                "new_model_inference": False, "valid_cases": len(valid)}
    metrics = {}
    for name in ARMS:
        metrics[name] = {"correct": sum(x[name]["correct"] for x in valid),
                         "model_calls": sum(x[name]["model_calls"] for x in valid),
                         "elapsed_s": sum(x[name]["elapsed_s"] for x in valid)}
    pairs = {}
    for baseline, treatment in PAIRS:
        b = sum((not x[baseline]["correct"]) and x[treatment]["correct"] for x in valid)
        c = sum(x[baseline]["correct"] and (not x[treatment]["correct"]) for x in valid)
        pairs[f"{baseline} -> {treatment}"] = {
            "rescues": b, "regressions": c, "net": b - c,
            "exact_mcnemar_two_sided": _mcnemar(b, c)}
    interaction = ((metrics["adapter_serializer"]["correct"] -
                    metrics["adapter_native"]["correct"]) -
                   (metrics["base_serializer"]["correct"] -
                    metrics["base_native"]["correct"]))
    return {"status": "DESCRIPTIVE_AUDIT_ONLY", "tasks": len(valid),
            "families": dict(families), "arms": metrics, "paired": pairs,
            "interaction_net_cases": interaction,
            "aa_variance_checked": False, "source_independence_verified": False,
            "new_model_inference": False, "promotion_allowed": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    report = audit(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "DESCRIPTIVE_AUDIT_ONLY" else 2

if __name__ == "__main__":
    raise SystemExit(main())
