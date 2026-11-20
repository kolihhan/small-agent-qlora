"""CPU-only original AppWorld reference-trajectory source feasibility.

The OFFICIAL AppWorld train/dev ground-truth API calls are held locally on the
runner, never uploaded or printed. Only aggregated counts and SHA of task-ID
sets leave this program. No LLM teacher, GPU, GAIA text, model training or
inference. A long reference API trace does NOT imply a valid next-action SFT
label for this project's tools: verify each candidate via state tests and
run a *separate* task-family heldout comparison before training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

PROTOCOL="APPWORLD-ORIGINAL-REFERENCE-TRAIN-HORIZON-FEASIBILITY-V1"
UPSTREAM_COMMIT="42b5bcf3cd334fee33f0c37c02070a9f5807add5"
MIN_TASKS=60
MIN_NON_ADMIN_HORIZON=6
NON_BUSINESS={"admin","api_docs","supervisor"}

def _call_stats(api_calls):
    if not isinstance(api_calls,list):
        raise ValueError("official reference API calls unavailable; no proxy labels")
    business=[]
    apps=set()
    bad=0
    for call in api_calls:
        if not isinstance(call,dict):
            bad+=1
            continue
        url=str(call.get("url") or "").strip().lstrip("/")
        parts=url.split("/")
        if len(parts)<2 or not parts[0]:
            bad+=1
            continue
        app=parts[0].casefold()
        if app not in NON_BUSINESS:
            business.append(call)
            apps.add(app)
    return {"all_calls":len(api_calls),"business_calls":len(business),
            "business_apps":len(apps),"malformed_calls":bad}

def _source_sha(ids):
    unique=sorted(set(ids))
    if len(unique)!=len(ids) or not unique:
        raise ValueError("missing/duplicate task IDs")
    return sha256("\n".join(unique).encode()).hexdigest()

def summarize(split, entries):
    ids=[x[0] for x in entries]
    counts=[_call_stats(x[1]) for x in entries]
    families={x.split("_")[0] for x in ids}
    bins=Counter()
    for x in counts:
        n=x["business_calls"]
        bucket="0" if n==0 else "1-2" if n<=2 else "3-5" if n<=5 else "6-7" if n<=7 else "8-11" if n<=11 else "12+"
        bins[bucket]+=1
    return {"split":split,"n":len(entries),"task_ids_sha256":_source_sha(ids),
            "scenario_count":len(families),
            "reference_api_calls":sum(x["all_calls"] for x in counts),
            "business_api_calls":sum(x["business_calls"] for x in counts),
            "min_business_calls":min((x["business_calls"] for x in counts),default=None),
            "max_business_calls":max((x["business_calls"] for x in counts),default=None),
            "tasks_with_6plus_business_calls":sum(x["business_calls"]>=6 for x in counts),
            "tasks_with_8plus_business_calls":sum(x["business_calls"]>=8 for x in counts),
            "tasks_with_12plus_business_calls":sum(x["business_calls"]>=12 for x in counts),
            "tasks_with_2plus_business_apps":sum(x["business_apps"]>=2 for x in counts),
            "malformed_api_call_items":sum(x["malformed_calls"] for x in counts),
            "histogram":dict(sorted(bins.items())),
            "source_features_only":True}

def audit_splits(train,dev):
    train_ids={x[0] for x in train}
    dev_ids={x[0] for x in dev}
    train_scenarios={x.split("_")[0] for x in train_ids}
    dev_scenarios={x.split("_")[0] for x in dev_ids}
    if (len(train)<MIN_TASKS or len(dev)<20 or
        train_ids&dev_ids or train_scenarios&dev_scenarios):
        raise ValueError("official source split missing, too small or overlaps IDs/scenarios")
    t=summarize("train",train);d=summarize("dev",dev)
    return {"status":"COMPLETE_ORIGINAL_APPWORLD_SOURCE_CPU_FEASIBILITY",
            "protocol":PROTOCOL,"upstream_repository":"StonyBrookNLP/appworld",
            "upstream_commit":UPSTREAM_COMMIT,"source_split_separated":True,
            "no_raw_ground_truth_or_task_ids_exported":True,
            "train":t,"dev":d,"train_long_task_feasibility":t["tasks_with_6plus_business_calls"],
            "no_model":True,"no_gpu":True,"no_training":True,
            "no_gaia_answers_or_evaluation":True,"no_stronger_teacher":True,
            "training_labels_verified":False,
            "claim_boundary":"Original reference API traces count business calls and scenario diversity; no independent replay, per-state evidence sufficiency or next-action action-format conversion validated. No SFT, agent score, GAIA or holdout transfer improvement.",
            "candidate_gate":"Only if >=16 TRAIN tasks have >=6 independently required business API calls, >=2 app families and real executable state verification: manually inspect a limited source-only train sample, implement compatible adapter and reserve untouched DEV and test tasks. No replay of AppWorld TEST reference labels."}

def source_from_official(split):
    from appworld import load_task_ids
    from appworld.ground_truth import GroundTruth
    ids=load_task_ids(split)
    if not ids:
        raise RuntimeError(f"no official AppWorld {split} task identities")
    entries=[]
    for task_id in ids:
        gt=GroundTruth.load(task_id=task_id,mode="full")
        if gt.api_calls is None:
            raise RuntimeError("original official API ground truth unavailable, no proxy estimates")
        entries.append((str(task_id),gt.api_calls))
    return entries

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable source feasibility report exists")
    t=source_from_official("train")
    d=source_from_official("dev")
    result=audit_splits(t,d)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("APPWORLD_REF_SOURCE_FEASIBILITY",json.dumps({
       "status":result["status"],"train":{k:v for k,v in result["train"].items() if k not in ("histogram",)},
       "dev":{k:v for k,v in result["dev"].items() if k not in ("histogram",)}
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
