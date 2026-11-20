"""Prospective TRAIN-only AppWorld tool-action necessity counterfactual, CPU.

For each of 8 independently source-selected original TRAIN tasks: replay all
the recorded official request actions in a fresh physical world (control) and
then replay the same requests EXCEPT one deterministically selected late
non-GET, non-auth, non-supervisor business request (treatment). Compare
official end-state unit tests. No gold input is shown to any model (none
used). Only count deltas leave the runner. This measures whether a request
matters to *the recorded reference trajectory*, not whether it is uniquely
necessary or a correct next-action at a given evidence state.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_appworld_training_horizon import _call_stats,_source_sha,UPSTREAM_COMMIT
from scripts.replay_appworld_train_reference import choose_train_entries
from scripts.replay_appworld_train_api_trace import replay_calls

PROTOCOL="APPWORLD-ORIGINAL-TRAIN-LATE-NONGET-ACTION-ABLATION-V1"
AUTH_OR_READ={"login","logout","auth","authenticate","access_token",
              "refresh_token","search","find","list","get","lookup","read","query",
              "complete_task","complete"}
SEED="original-private-train-reference-late-executed-action-blinded-20261117"

def late_candidate(requests):
    if not isinstance(requests,list) or len(requests)<8:
        return None
    eligible=[]
    for i,req in enumerate(requests):
        verb=str(req.get("method") or "").lower()
        path=str(req.get("url") or "").strip("/").split("/")
        endpoint=path[-1].lower() if len(path)>1 else ""
        app=path[0].lower() if path else ""
        if (i>=6 and verb in {"post","put","patch","delete"} and
            app not in {"admin","api_docs","supervisor"} and
            endpoint not in AUTH_OR_READ and
            not any(endpoint.startswith(prefix+"_") for prefix in AUTH_OR_READ)):
            eligible.append(i)
    return eligible[-1] if eligible else None

def _execute_task(task_id,omitted):
    from appworld import AppWorld
    with AppWorld(task_id=task_id,experiment_name="private-train-late-action-necessity",
                  ground_truth_mode="full",timeout_seconds=240) as world:
        g=world.task.ground_truth
        if g is None or g.api_calls is None:
            raise ValueError("original reference API calls missing")
        reqs=g.api_calls if omitted is None else [
            r for i,r in enumerate(g.api_calls) if i!=omitted]
        replay_calls(world,reqs)
        tracker=world.evaluate(suppress_errors=True)
        return bool(tracker.success),int(tracker.pass_count),int(tracker.num_tests)

def evaluate(selected_with_calls):
    counts=Counter()
    by_depth=Counter()
    # No IDs, API names, method args, source code or test output in summary.
    for task_id,source_calls in selected_with_calls:
        candidate=late_candidate(source_calls)
        counts["source_tasks"]+=1
        counts["late_nonget_candidates"]+=int(candidate is not None)
        if candidate is None:continue
        by_depth["index_ge_6"]+=int(candidate>=6)
        try:
            base,base_pass,total=_execute_task(task_id,None)
            if not base:
                counts["baseline_reference_state_failed"]+=1
                continue
            counts["baseline_state_pass"]+=1
            test,remaining_pass,total2=_execute_task(task_id,candidate)
            if total!=total2:
                raise RuntimeError("reference official state-test count changed")
            counts["skip_one_state_still_pass"]+=int(test)
            counts["skip_one_state_fail"]+=int(not test)
            counts["lost_individual_state_tests"]+=max(0,base_pass-remaining_pass)
            counts["paired_state_tests"]+=total
        except Exception as exc:
            counts["error"]+=1
            counts["exception_"+type(exc).__name__]+=1
    return {"status":"TRAIN_ONLY_REFERENCE_ACTION_COUNTERFACTUAL_COMPLETE",
            "protocol":PROTOCOL,"upstream_commit":UPSTREAM_COMMIT,
            "selected_task_set_sha256":_source_sha([x[0] for x in selected_with_calls]),
            "counts":dict(sorted(counts.items())),
            "all_selected_late_action_indices_ge_6":bool(by_depth["index_ge_6"]==counts["late_nonget_candidates"]),
            "scope":"one prereg late non-GET non-auth action dropped from original replay; same task state, independent fresh official worlds",
            "reference_only_not_a_necessary_oracle":True,
            "no_decrypted_reference_data_exported":True,
            "no_student_model":True,"no_training":True,"no_gpu":True,
            "no_GAIA_or_DEV_or_TEST_data":True,
            "limits":"State failure after removing one call may reflect direct state effect, subsequent call dependency or error. A passing ablation means one recorded action may be redundant for end-state; neither proves unique next-action supervision. No model or GAIA improvement."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output already exists")
    from appworld import load_task_ids
    from appworld.ground_truth import GroundTruth
    source={tid:GroundTruth.load(task_id=tid,mode="full").api_calls
            for tid in load_task_ids("train")}
    selected=choose_train_entries([(tid,type("Gold",(),{"api_calls":v})())
                                   for tid,v in source.items()])
    result=evaluate([(tid,source[tid]) for tid in selected])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("APPWORLD_TRAIN_LATE_ACTION_STATE_ABLATION",
          json.dumps(result["counts"],sort_keys=True),flush=True)

if __name__=="__main__":
    main()
