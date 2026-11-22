"""Original 100-case read-only opportunity counts for the opt-in protocol repair.
No GAIA reference answers, question text, tool content, model inference or training.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json


def audit(root: Path) -> dict:
    tally = {}
    identities = {}
    for arm in ("base", "qlora-v8"):
        files = sorted(root.glob("gaia-v8-paired-shard-*/" + arm + "/shard-*/results.jsonl"))
        if len(files) != 20:
            raise ValueError("Missing one or more of the original twenty paired shards")
        seen = set()
        stops = Counter()
        opportunities = Counter()
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row["task_id"]
                if tid in seen:
                    raise ValueError("Duplicate original task ID")
                seen.add(tid)
                stop = row.get("stop_reason") or "unknown"
                stops[stop] += 1
                if stop == "model_error":
                    msgs = [
                        e.get("data", {}).get("message", "")
                        for e in row.get("trace", [])
                        if e.get("kind") == "model_error"
                    ]
                    if any(str(m).startswith("Invalid tool arguments JSON:") for m in msgs):
                        opportunities["malformed_json"] += 1
                if stop == "invalid_final":
                    observed = [
                        e.get("data", {}).get("content", "")
                        for e in row.get("trace", [])
                        if e.get("kind") == "invalid_final"
                    ]
                    if any(str(x).lstrip().startswith("<tool_call>") for x in observed):
                        opportunities["text_toolcall_at_final"] += 1
        if len(seen) != 100:
            raise ValueError("Original evaluation is not 100 unique tasks")
        identities[arm] = seen
        tally[arm] = {"n": len(seen), "stops": dict(stops), "eligible": dict(opportunities)}
    if identities["base"] != identities["qlora-v8"]:
        raise ValueError("Base and v8 tasks are not matched")
    if tally["qlora-v8"]["eligible"] != {"malformed_json": 11, "text_toolcall_at_final": 29}:
        raise ValueError("Unexpected drift from original v8 traces")
    return {
        "status": "CPU_RESEARCH_OPPORTUNITY_ONLY_NOT_RESCORING",
        "original_run": 44980055775,
        "arms": tally,
        "new_inference": 0,
        "estimated_model_call_budget": "At most 2 JSON retries plus 1 final retry per task if both opt-in flags enabled.",
        "gaia_scores_recomputed": False,
        "benchmark_leakage_decontaminated": False,
        "no_task_or_answer_content_exported": True,
        "note": "Eligible failure counts are NOT predicted additional successful answers.",
    }

def main():
    report = audit(Path("evidence"))
    target = Path("audit/protocol-recovery-original-opportunities.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("PROTOCOL_RECOVERY_OPPORTUNITIES",json.dumps(report["arms"],sort_keys=True),flush=True)

if __name__ == "__main__":
    main()
