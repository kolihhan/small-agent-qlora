"""Independent source-shift recovery probe: Qwen3.5-9B Base versus v7 LoRA.

ToolMisuseBench v0.2 dev only. Native benchmark scorer/environment, fixed 4 domains
x 6 fault families, same task IDs/model/server/decoding across arms. This is not
GAIA, not an evaluation of our 4-tool runtime, and not training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


DOMAINS = ("crud", "retrieval", "files", "scheduling")
FAULTS = ("clean", "timeout", "rate_limit", "schema_drift", "authz", "adversarial_error")
N_SHARDS = 4
N_TASKS = len(DOMAINS) * len(FAULTS)


def select_tasks(source: Path) -> tuple[list, str]:
    from toolmisusebench.types import Task
    raw = source.read_bytes()
    digest = sha256(raw).hexdigest()
    rows = [Task.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 900 or any(x.split != "dev" for x in rows):
        raise ValueError("expected pinned 900-row ToolMisuseBench dev source")
    if len({x.task_id for x in rows}) != len(rows):
        raise ValueError("duplicated task ids")
    grouped = {(d, f): [] for d in DOMAINS for f in FAULTS}
    for row in rows:
        family = row.fault_plan[0].fault_type if row.fault_plan else "clean"
        if row.domain in DOMAINS and (row.domain, family) in grouped:
            grouped[(row.domain, family)].append(row)
    selected = []
    for d in DOMAINS:
        for f in FAULTS:
            available = sorted(
                grouped[(d, f)], key=lambda row: (
                    sha256(f"v7-transfer24-{row.task_id}".encode()).hexdigest(),
                    row.task_id,
                )
            )
            if not available:
                raise ValueError(f"missing preregistered domain/fault pair {d}/{f}")
            selected.append(available[0])
    if len(selected) != N_TASKS or len({x.task_id for x in selected}) != N_TASKS:
        raise ValueError("invalid cross-domain sample")
    return selected, digest


def run_shard(*, source: Path, shard: int, model: str,
              endpoint: str) -> dict:
    if shard not in range(N_SHARDS) or not model:
        raise ValueError("invalid shard or model")
    from toolmisusebench.harness.runner import run_episode
    from toolmisusebench.version2.llm_agent import OpenAICompatibleToolAgent
    cases, digest = select_tasks(source)
    selection_sha = sha256("|".join(x.task_id for x in cases).encode()).hexdigest()
    agent = OpenAICompatibleToolAgent(
        base_url=endpoint, model=model, timeout_s=180,
        temperature=0.0, seed=42, max_tokens=256, disable_thinking=True,
    )
    records = []
    for task in cases[shard::N_SHARDS]:
        try:
            result = run_episode(task, agent)
            item = {
                "success": result.success, "recovery_success": result.recovery_success,
                "fault_seen": result.fault_seen,
                "invalid_calls": result.invalid_calls,
                "tool_calls": result.tool_calls,
                "budget_exceeded": result.budget_exceeded,
                "policy_violations": result.policy_violations,
                "error": None,
            }
        except Exception as exc:
            # A backend/protocol failure is counted as task failure, never skipped.
            item = {
                "success": False, "recovery_success": False, "fault_seen": False,
                "invalid_calls": 0, "tool_calls": 0, "budget_exceeded": False,
                "policy_violations": 0, "error": type(exc).__name__,
            }
        records.append({
            "id_hash": sha256(task.task_id.encode()).hexdigest(),
            "domain": task.domain,
            "fault": task.fault_plan[0].fault_type if task.fault_plan else "clean",
            **item,
        })
        print("TMB_TRANSFER_CASE", shard, task.domain, records[-1]["fault"],
              item["success"], item["error"], flush=True)
    if len(records) != N_TASKS // N_SHARDS:
        raise ValueError("incomplete transfer shard")
    return {
        "shard": shard, "source_sha": digest, "selection_sha": selection_sha,
        "records": records, "protocol": "TMB-v0.2-dev-source-shift",
    }


def aggregate(parts: list[dict]) -> dict:
    if len(parts) != 2 * N_SHARDS:
        raise ValueError("paired base/v7 all shards required")
    by_arm = {arm: sorted((x for x in parts if x["arm"] == arm),
                           key=lambda x: x["shard"]) for arm in ("base", "v7")}
    if any([x["shard"] for x in by_arm[arm]] != list(range(N_SHARDS))
           for arm in by_arm):
        raise ValueError("missing/duplicate arm shards")
    source_set = {x["source_sha"] for x in parts}
    selection_set = {x["selection_sha"] for x in parts}
    if len(source_set) != 1 or len(selection_set) != 1:
        raise ValueError("source or selection drift")
    rows = {arm: [r for x in by_arm[arm] for r in x["records"]]
            for arm in by_arm}
    if any(len(rows[a]) != N_TASKS or len({r["id_hash"] for r in rows[a]}) != N_TASKS
           for a in rows):
        raise ValueError("incomplete 24-case arms")
    if [(r["id_hash"], r["domain"], r["fault"]) for r in rows["base"]] != [
        (r["id_hash"], r["domain"], r["fault"]) for r in rows["v7"]
    ]:
        raise ValueError("paired task identities changed")
    before, after = rows["base"], rows["v7"]
    by_model = {}
    for arm, records in rows.items():
        by_model[arm] = {
            "correct": sum(r["success"] for r in records),
            "recovery_success": sum(r["recovery_success"] for r in records),
            "fault_seen": sum(r["fault_seen"] for r in records),
            "invalid_calls": sum(r["invalid_calls"] for r in records),
            "tool_calls": sum(r["tool_calls"] for r in records),
            "backend_errors": dict(Counter(r["error"] for r in records if r["error"])),
            "by_domain": {
                d: sum(r["success"] for r in records if r["domain"] == d)
                for d in DOMAINS
            },
            "by_fault": {
                f: sum(r["success"] for r in records if r["fault"] == f)
                for f in FAULTS
            },
        }
    return {
        "n": N_TASKS, "arms": by_model,
        "rescues": sum(not x["success"] and y["success"] for x, y in zip(before, after)),
        "regressions": sum(x["success"] and not y["success"] for x, y in zip(before, after)),
        "source_sha": parts[0]["source_sha"],
        "selection_sha": parts[0]["selection_sha"],
        "claim_boundary": "ToolMisuseBench synthetic deterministic dev source-shift, separate native agent/harness; not GAIA, not training, not our 4-tool runtime",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=("base", "v7"))
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        result = aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("TOOLMISUSE_TRANSFER_FINAL", json.dumps(result, sort_keys=True), flush=True)
    else:
        if args.shard is None or args.source is None or not args.arm:
            p.error("source/shard/arm required for inference")
        result = run_shard(source=args.source, shard=args.shard,
                           model=args.model, endpoint=args.endpoint)
        result["arm"] = args.arm
        print("TOOLMISUSE_TRANSFER_SHARD", args.arm, args.shard, flush=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
