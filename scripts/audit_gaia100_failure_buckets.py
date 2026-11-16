"""Diagnose 100 paired GAIA v7 outcomes without opening question/answer text.

Immutable artifact-only, same two arms and every frozen task hash required.
No new model calls, reruns, LoRA, answer-label access, or score improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

EXPECTED_TASK_IDS_SHA="869e24fd53a0eeabc2598d952d0b5f893eca279584c96eb407a72906dccba45a"
EXPECTED_REVISION="682dd723ee1e1697e00360edccf2366dc8418dd9"
N_SHARDS=20
ARMS=("original","format_reminder")
STOPS={"final","invalid_final","model_error","model_timeout"}


def summarize(paths:list[Path])->dict:
    if len(paths)!=N_SHARDS:
        raise ValueError("Require exactly 20 original GAIA shard artifacts")
    shards=[json.loads(p.read_text()) for p in paths]
    if sorted(s.get("shard") for s in shards)!=list(range(N_SHARDS)):
        raise ValueError("Missing/duplicate physical shard identities")
    per_arm={a:Counter() for a in ARMS}
    paired=Counter()
    paired_failure={a:Counter() for a in ARMS}
    ids=set()
    outcomes=[]
    for shard in shards:
        if (shard.get("full_hash")!=EXPECTED_TASK_IDS_SHA
            or shard.get("source_revision")!=EXPECTED_REVISION
            or shard.get("selection")!="gaia-evaluation100-seen-dev"
            or set(shard.get("arms",{}))!=set(ARMS)):
            raise ValueError("Pinned original GAIA100 source or arm metadata drift")
        one,two=(shard["arms"][arm] for arm in ARMS)
        if len(one)!=5 or len(two)!=5:
            raise ValueError("Every shard must contain five paired outcomes")
        for a,b in zip(one,two):
            if (a.get("task_hash")!=b.get("task_hash")
                or a["task_hash"] in ids
                or a.get("level")!=b.get("level")):
                raise ValueError("Cross-arm tasks do not match or are reused")
            ids.add(a["task_hash"])
            if not all(x.get("stop_reason") in STOPS for x in (a,b)):
                raise ValueError("Unknown stop reason")
            if any(x["correct"] and not x["completed"] for x in (a,b)):
                raise ValueError("Impossible correct uncompleted episode")
            paired_state=(
                "both_correct" if a["correct"] and b["correct"]
                else "original_only" if a["correct"]
                else "reminder_only" if b["correct"]
                else "both_wrong"
            )
            paired[paired_state]+=1
            for arm,x in zip(ARMS,(a,b)):
                per_arm[arm]["n"]+=1
                per_arm[arm]["correct"]+=int(x["correct"])
                per_arm[arm]["completed"]+=int(x["completed"])
                per_arm[arm]["completed_wrong"]+=int(x["completed"] and not x["correct"])
                per_arm[arm]["not_completed"]+=int(not x["completed"])
                per_arm[arm]["tool_calls"]+=int(x["tool_calls"])
                per_arm[arm]["tool_errors"]+=int(x["tool_errors"])
                per_arm[arm]["duplicates"]+=int(x["duplicates"])
                per_arm[arm][f"stop_{x['stop_reason']}"]+=1
                if paired_state=="both_wrong":
                    paired_failure[arm][x["stop_reason"]]+=1
            outcomes.append({"task_hash":a["task_hash"],
                             "level":a["level"],"paired":paired_state,
                             "original_stop":a["stop_reason"],
                             "reminder_stop":b["stop_reason"]})
    if len(ids)!=100 or sum(paired.values())!=100:
        raise ValueError("Complete distinct 100-case sample missing")
    for arm in ARMS:
        p=per_arm[arm]
        if p["n"]!=100 or p["completed"]!=p["correct"]+p["completed_wrong"]:
            raise RuntimeError("Complete/final accounting mismatch")
    return {
        "status":"COMPLETE_CPU_FAIL_BUCKETS_NO_NEW_MODEL_SCORE",
        "source":"physically completed GAIA100 #43865621132",
        "source_revision":EXPECTED_REVISION,
        "frozen_id_sha256":EXPECTED_TASK_IDS_SHA,
        "n":100,"arms":{a:dict(v) for a,v in per_arm.items()},
        "paired":dict(paired),
        "both_wrong_stop_reason":{a:dict(v) for a,v in paired_failure.items()},
        "no_gaia_answers_accessed":True,"no_new_inference":True,
        "not_hidden":True,"no_promoted_intervention":True,
        "limitations":"Stop reasons describe runtime outcomes, not independently human-labeled semantic causes. Completed-wrong can mean reasoning, unsupported evidence, or exact-output mismatch. Model timeouts may vary with host load. The cohort was previously exposed development data.",
        "cases":outcomes,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=summarize(a.inputs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("GAIA100_FAIL_BUCKETS_FINAL",
          json.dumps({k:v for k,v in out.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
