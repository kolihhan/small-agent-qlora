"""Private CPU replay of a tiny fixed TRAIN-only official AppWorld gold subset.

The reference code and task contents stay entirely in official AppWorld's
isolated runner. Only aggregate task-level *counts*, no IDs, source code,
outputs, user data or API args are exported. DEV / TEST never loaded or replayed.
This checks end-to-end state correctness, not per-action necessity or mapping
onto small-agent's search/read tool schema. No teacher, GPU or SFT.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_appworld_training_horizon import (_call_stats,_source_sha,UPSTREAM_COMMIT)

PROTOCOL="APPWORLD-ORIGINAL-TRAIN-REFERENCE-STATE-REPLAY-PILOT-V1"
SEED="state-verified-official-human-reference-8-tasks-train-only-20261117"
N=8
MIN_CALLS=12

def choose_train_entries(entries):
    eligible=[]
    for task_id,gt in entries:
        stats=_call_stats(gt.api_calls)
        if stats["business_calls"]<MIN_CALLS or stats["business_apps"]<2:
            continue
        eligible.append((sha256(f"{SEED}:{task_id}".encode()).hexdigest(),task_id))
    selected=[]
    scenarios=set()
    for _,tid in sorted(eligible):
        scenario=tid.split("_")[0]
        if scenario in scenarios:continue
        scenarios.add(scenario)
        selected.append(tid)
        if len(selected)==N:break
    if len(selected)!=N:
        raise ValueError("insufficient original TRAIN two-app, >=12-call, scenario-independent gold tasks")
    return selected

def replay(task_ids):
    from appworld import AppWorld
    counts=Counter()
    safe_stats=[]
    for task_id in task_ids:
        stage="world_construct"
        try:
            with AppWorld(task_id=task_id,experiment_name="private-train-gold-reference-replay",
                          ground_truth_mode="full",timeout_seconds=240) as world:
                stage="reference_load"
                g=world.task.ground_truth
                if g is None or not g.compiled_solution_code or g.api_calls is None:
                    raise RuntimeError("missing original official TRAIN compiled reference")
                module=g.compiled_solution_module()
                import inspect
                sig=inspect.signature(module.solution)
                counts["signature_nargs_"+str(len(sig.parameters))]+=1
                counts["signature_args_"+",".join(sig.parameters.keys())]+=1
                stage="reference_execute"
                # Official reference program (NOT external model teacher).
                # Execute inside the task's own stateful AppWorld APIs.
                if tuple(sig.parameters)!=("apis","requester"):
                    raise TypeError("pinned AppWorld original gold function signature drift")
                module.solution(world.apis,world.requester)
                stage="state_evaluate"
                tracker=world.evaluate(suppress_errors=True)
                passed=bool(tracker.success)
                counts["successful_state"]+=int(passed)
                counts["failed_state"]+=int(not passed)
                safe_stats.append({"success":passed,
                                   "state_test_count":int(tracker.num_tests),
                                   "api_calls_at_source":len(g.api_calls)})
        except Exception as exc:
            # Do not leak decrypted task or reference program via exception text.
            counts["errors"]+=1
            counts["exception_class_"+type(exc).__name__]+=1
            counts["error_stage_"+stage]+=1
    return {"n":len(task_ids),"task_subset_sha256":_source_sha(task_ids),
            "counts":dict(sorted(counts.items())),
            "mean_reference_apis":round(sum(x["api_calls_at_source"] for x in safe_stats)/len(safe_stats),2) if safe_stats else None,
            "mean_state_tests":round(sum(x["state_test_count"] for x in safe_stats)/len(safe_stats),2) if safe_stats else None,
            "attempted":len(task_ids)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable source result already exists")
    from appworld import load_task_ids
    from appworld.ground_truth import GroundTruth
    # TRAIN ONLY. Do not load DEV/TEST gold content during replay.
    rows=[(tid,GroundTruth.load(task_id=tid,mode="full")) for tid in load_task_ids("train")]
    selected=choose_train_entries(rows)
    result=replay(selected)
    result.update({"protocol":PROTOCOL,"upstream_commit":UPSTREAM_COMMIT,
                   "split":"train","state_verified_reference_calls_only":True,
                   "new_training_steps":0,"teacher_model_used":False,
                   "dev_or_test_content_loaded":False,
                   "no_raw_decrypted_task_or_code_export":True,
                   "training_labels_ready":False,
                   "claim_boundary":"State correctness of original TRAIN gold reference code, limited to 8 preselected source-only scenarios. No ablation/counterfactual evidence that all calls are necessary; no autonomous student or GAIA improvement."})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("APPWORLD_GOLD_TRAIN_ONLY_REPLAY",json.dumps({
        "n":result["n"],"attempted":result["attempted"],
        "counts":result["counts"],"mean_state_tests":result["mean_state_tests"],
        "no_raw_decrypted_task_or_code_export":True},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
