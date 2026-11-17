"""CPU-only independent replay of ORIGINAL AppWorld TRAIN ground-truth HTTP API calls.

This does not run the reference Python program. Replays its recorded executed
API request method/url/data in a fresh isolated original task world, then uses
AppWorld's state test. No raw decrypted records, query parameters, responses,
or task text leave the GitHub runner; artifact is only aggregate counts.
Works only if the reference request transcript is deterministic/replayable.
No inference-time model, no teacher, no training and no DEV/TEST exposure.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.replay_appworld_train_reference import (
    choose_train_entries,PROTOCOL as REF_PROTOCOL,UPSTREAM_COMMIT)
from scripts.audit_appworld_training_horizon import _source_sha

PROTOCOL="APPWORLD-TRAIN-ORIGINAL-REQUEST-TRACE-STATE-REPLAY-V1"
ALLOWED_METHODS={"get","post","put","patch","delete"}
MAX_REQUESTS=400

def replay_calls(world,requests):
    if not isinstance(requests,list) or not requests or len(requests)>MAX_REQUESTS:
        raise ValueError("official reference request source unavailable or over capped")
    completed=0
    for req in requests:
        if not isinstance(req,dict) or set(req)<{"method","url","data"}:
            raise ValueError("malformed reference request")
        verb=str(req["method"]).lower()
        url=req["url"];data=req["data"]
        if (verb not in ALLOWED_METHODS or not isinstance(url,str)
            or not url or not isinstance(data,dict)):
            raise ValueError("unexpected original REST method/args")
        fn=getattr(world.requester,verb)
        # Real simulated REST request; NEVER show/log the returned data.
        fn(url=url,data=data,raise_on_failure=False,show=False)
        completed+=1
    return completed

def private_trace_replay(ids):
    from appworld import AppWorld
    counts=Counter()
    api_sent=0
    tests=0
    for task_id in ids:
        stage="construct"
        try:
            with AppWorld(task_id=task_id,experiment_name="private-train-http-gold-trace-replay",
                          ground_truth_mode="full",timeout_seconds=240) as world:
                stage="request_replay"
                g=world.task.ground_truth
                if g is None or g.api_calls is None:
                    raise RuntimeError("official reference request trace missing")
                api_sent+=replay_calls(world,g.api_calls)
                stage="state_evaluate"
                verdict=world.evaluate(suppress_errors=True)
                tests+=int(verdict.num_tests)
                counts["state_pass"]+=int(verdict.success)
                counts["state_fail"]+=int(not verdict.success)
        except Exception as exc:
            # Never expose error messages, potentially contain task/API secrets.
            counts["error"]+=1
            counts["error_"+stage+"_"+type(exc).__name__]+=1
    return {"attempted":len(ids),"task_ids_sha256":_source_sha(ids),
            "request_calls_replayed":api_sent,
            "state_tests_run":tests,"counts":dict(sorted(counts.items()))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable private pilot summary exists")
    from appworld import load_task_ids
    from appworld.ground_truth import GroundTruth
    # Use exactly the same SHA-frozen selection predicate as reference-program
    # replay; still source-only TRAIN, no DEV/TEST tasks loaded.
    data=[(tid,GroundTruth.load(task_id=tid,mode="full")) for tid in load_task_ids("train")]
    ids=choose_train_entries(data)
    result=private_trace_replay(ids)
    result.update({"protocol":PROTOCOL,"upstream_commit":UPSTREAM_COMMIT,
                   "reference_program_protocol":REF_PROTOCOL,
                   "split":"train","no_raw_reference_requests_or_answers_exported":True,
                   "no_gaia_data":True,"no_training":True,"no_llm":True,
                   "no_gpu":True,"official_reference_trace_is_proven_necessary":False,
                   "claim_boundary":"Replay of original TRAIN request sequence is only state/API execution evidence. It does not establish that each request is necessary, yields useful information at its prefix, or maps to current small-agent tool schema. No downstream/GAIA score."})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("APPWORLD_TRAIN_REQUEST_REPLAY_AGGREGATE",
          json.dumps({k:result[k] for k in ("attempted","request_calls_replayed","state_tests_run","counts")},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
