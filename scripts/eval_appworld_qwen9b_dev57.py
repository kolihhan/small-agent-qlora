"""Original fixed Qwen3.5-9B, official immutable AppWorld DEV57 stage evaluation.

No oracle in model context. No inference-time external model, teachers, LoRA,
hidden TEST, or tuning on previous eight exposed DEV pilots. Full roster
divided by original 19 scene IDs: exactly 3 tasks per physical shard. This is
a reproducibility/diagnostic baseline; stage metrics are not task success.

Use GitHub CI to freeze IDs first, run the exact original Qwen GGUF on an
isolated AppWorld simulator, privately grade after the student finishes,
and export only hashes and counters. An incomplete 57 is NOT a DEV57 score.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool,AppWorldCallTool
from scripts.appworld_dev57_fixed_metrics import (
    PROTOCOL, MODEL_GGUF_SHA, N, SCENARIOS, freeze_official_dev,
    extract_stage_record, aggregate_dev57
)

SHARDS=SCENARIOS
STUDENT_POLICY="same-fixed-original-Qwen9B-native-20step-AppWorld-tool-v1"
MAX_STEPS=20
MAX_API_CALLS=48


def arrange_scenarios(ids: list[str]) -> list[list[str]]:
    groups={}
    for tid in ids:
        name=tid.split("_",1)[0]
        groups.setdefault(name,[]).append(tid)
    if len(ids)!=N or len(groups)!=SCENARIOS or any(len(rows)!=3 for rows in groups.values()):
        raise ValueError("original DEV57 must contain 19 groups with three tasks each")
    # Stable source-only scenario ranking, never student success or gold.
    return [sorted(groups[name]) for name in
            sorted(groups,key=lambda x:sha256(("main-dev57-"+x).encode()).hexdigest())]


def freeze(ids: list[str]) -> dict:
    roster=freeze_official_dev(ids)
    groups=arrange_scenarios(ids)
    roster.update({
        "student_policy":STUDENT_POLICY,
        "max_steps":MAX_STEPS,"max_api_calls":MAX_API_CALLS,
        "shards":SHARDS,"tasks_per_shard":3,
        "scenario_group_sha256":sha256(json.dumps([
           [sha256(t.encode()).hexdigest() for t in group]
           for group in groups],sort_keys=True).encode()).hexdigest(),
        "original_model_no_lora":True,"ground_truth_never_in_student_messages":True
    })
    return roster


def verify_frozen(meta:dict,ids:list[str]) -> list[list[str]]:
    if meta!=freeze(ids):raise ValueError("original official DEV57 source/policy drift")
    return arrange_scenarios(ids)


def clock():
    if hasattr(time,"CLOCK_MONOTONIC_RAW"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    raise RuntimeError("raw monotonic clock required; appworld frozen perf_counter is invalid")


def task(task_id:str, workspace:Path) -> dict:
    from appworld import AppWorld
    started=clock()
    with AppWorld(task_id=task_id,experiment_name="small-agent-qwen9b-main-dev57-original-no-train",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        docs=AppWorldDocsTool(world)
        api=AppWorldCallTool(world,max_calls=MAX_API_CALLS,report_http_failures=True)
        model=OpenAICompatModel(model="qwen3.5-9b",base_url="http://127.0.0.1:8080",
                                timeout_s=240,max_new_tokens=384)
        prompt=(
            "You are operating in an isolated AppWorld simulation. "
            "Perform the following task by actually calling the available "
            "AppWorld APIs. Use appworld_docs to inspect available task apps "
            "and each API's required parameters; then use appworld_call "
            "to read and update the task state. Do not invent results. "
            "Call supervisor.complete_task when the requested changes are done. "
            "A final textual claim does not itself change app state.\n\n"
            "USER TASK:\n"+world.task.instruction)
        student=AgentRuntime(model,[docs,api],max_steps=MAX_STEPS).run(prompt,workspace)
        # Only after the student has stopped: official private state evaluator.
        judgment=world.evaluate(suppress_errors=True)
        # TRAIN reference never used. DEV reference API count is an explicitly
        # post-run diagnostic difficulty proxy, not a selection criterion.
        calls=world.task.ground_truth.api_calls
        reference_calls=len(calls) if isinstance(calls,list) else None
        record=extract_stage_record(
            task_id,student,api_calls=api.calls,real_http_failures=api.http_errors,
            official_state_success=bool(judgment.success),
            state_tests_pass=int(judgment.pass_count),
            state_tests_total=int(judgment.num_tests),
            observed_reference_call_count=reference_calls)
    record["student_elapsed_raw_wall_s"]=round((clock()-started)/1e9,2)
    return record


def run_shard(meta:dict,index:int,workspace:Path) -> dict:
    if index not in range(SHARDS):
        raise ValueError("unrecognized official DEV57 fixed source shard")
    from appworld import load_task_ids
    tasks=verify_frozen(meta,load_task_ids("dev"))[index]
    scores=[]
    for i,tid in enumerate(tasks):
        try:
            row=task(tid,workspace / str(i))
        except Exception as exc:
            # Protect user task text, decrypted source and original API inputs
            # from GitHub Actions exception traces.
            raise RuntimeError("APPWORLD_PRIVATE_DEV_TASK_FAILED:"+type(exc).__name__) from None
        scores.append(row)
        print("APPWORLD_DEV57_PHYSICAL_STAGE_SHARD",index,"task",i+1,
              "of",len(tasks),"official_state_success",int(row["official_state_success"]),
              "phase",row["stop_reason"],flush=True)
    return {"status":"COMPLETE_FIXED_APPWORLD_DEV57_ONE_SCENARIO_SHARD",
            "protocol":PROTOCOL,"fixed_source_sha256":meta["task_ids_sha256"],
            "task_hashes_sha256":meta["task_hashes_sha256"],
            "policy":STUDENT_POLICY,"model_sha256":MODEL_GGUF_SHA,
            "shard":index,"shards":SHARDS,"n":len(scores),"cases":scores,
            "no_teacher":True,"no_GAIA_score":True,"no_training":True,
            "student_never_saw_reference":True}


def combine(parts:list[dict],meta:dict,ids:list[str]) -> dict:
    frozen=freeze(ids)
    if meta!=frozen:
        raise ValueError("source or model changed between freeze and aggregation")
    if len(parts)!=SHARDS or sorted(x.get("shard") for x in parts)!=list(range(SHARDS)):
        raise ValueError("ALL 19 official DEV scenario shards required")
    if any(p.get("status")!="COMPLETE_FIXED_APPWORLD_DEV57_ONE_SCENARIO_SHARD" or
           p.get("protocol")!=PROTOCOL or
           p.get("model_sha256")!=MODEL_GGUF_SHA or
           p.get("policy")!=STUDENT_POLICY or
           p.get("fixed_source_sha256")!=meta["task_ids_sha256"] or
           p.get("task_hashes_sha256")!=meta["task_hashes_sha256"] or
           p.get("n")!=3 or len(p.get("cases",[]))!=3 or
           p.get("no_training") is not True or
           p.get("student_never_saw_reference") is not True for p in parts):
        raise ValueError("missing/corrupted actual original Qwen physical shards")
    rows=[x for p in parts for x in p["cases"]]
    expected=[sha256(t.encode()).hexdigest() for t in ids]
    summary=aggregate_dev57(rows,meta,expected_task_hashes=expected)
    return {"status":"COMPLETE_ORIGINAL_QWEN35_9B_APPWORLD_DEV57_PRIMARY",
            "protocol":PROTOCOL,"model_sha256":MODEL_GGUF_SHA,
            "frozen_source_sha256":meta["task_ids_sha256"],
            "original_physical_shards":SHARDS,"summary":summary,
            "raw_wall_seconds":round(sum(r["student_elapsed_raw_wall_s"] for r in rows),2),
            "no_train_or_adapter":True,"not_GAIA":True,
            "source_exposure_warning":"8 of 57 DEV task identities were seen in previous public 4+4 pilot; remaining 49 not previously scored in those two pilots.",
            "truth_qualifier":"AppWorld official DEV only, 57 dependent tasks from 19 scenes. Stage/HTTP recovery are descriptive proxies; source oracle only after model stopped. No model improvement or GAIA transfer claim."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","run","aggregate"])
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--workspace",type=Path,default=Path("physical-appworld-dev57"))
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("do not overwrite immutable model/source evidence")
    from appworld import load_task_ids
    ids=load_task_ids("dev")
    if args.mode=="freeze":
        result=freeze(ids)
        print("APPWORLD_DEV57_FIXED_SOURCE_LOCKED",
              result["task_ids_sha256"],result["task_hashes_sha256"],flush=True)
    elif args.mode=="run":
        if args.frozen is None or args.shard is None:parser.error("--frozen and --shard required")
        result=run_shard(json.loads(args.frozen.read_text("utf-8")),args.shard,args.workspace)
    else:
        if args.frozen is None or not args.inputs:parser.error("--frozen and --inputs required")
        result=combine([json.loads(p.read_text("utf-8")) for p in args.inputs],
                       json.loads(args.frozen.read_text("utf-8")),ids)
        print("APPWORLD_DEV57_REAL_ORIGINAL_MODEL_FULL_SCORE",
              json.dumps(result["summary"],sort_keys=True),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
