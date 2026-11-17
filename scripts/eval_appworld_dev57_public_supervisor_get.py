"""Original Qwen3.5-9B fixed AppWorld DEV57 paired PUBLIC supervisor GET experiment.

Only intervention relative to verified original DEV57 baseline: allow PUBLIC,
documented supervisor GET APIs in the SAME appworld_docs and appworld_call.
No privileged/private supervisor APIs, reference actions, hidden evaluator
access or task selection based on model outcomes. Main benchmark 57 task IDs
and 19 scenes, original SHA-frozen Qwen3.5-9B GGUF, 20 model decisions,
48 business API requests, exact same system/user prompt, temperature=0.
Aggregate strictly matched to all 19 original baseline source scene artifacts,
not an n=4 new subset. Descriptive task-paired / scenario-cluster 95% CI;
57 cases / 19 scenes cannot guarantee 5pt statistical power.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool,AppWorldCallTool
from scripts import eval_appworld_qwen9b_dev57 as base
from scripts.appworld_dev57_fixed_metrics import (
    PROTOCOL,MODEL_GGUF_SHA,N,SCENARIOS,aggregate_dev57,
    extract_stage_record
)

TREATMENT_POLICY="same-fixed-original-Qwen9B-native-20step-AppWorld-public-supervisor-GET-v1"
SOURCE_BENCHMARK="APPWORLD-ORIGINAL-FROZEN-DEV57-PUBLIC-SUPERVISOR-READS-PAIRED-V1"
SOURCE_SHA="d36670b383552f4c586baf1f84f0d53fcd3e05db5698b1532b2291ace23288a8"
SOURCE_SET_HASH="f6f4121b9f75b1d4f57ec2da12a1234966acd3bbcf7626ae117cba5b9b2bc9b1"
BOOTSTRAP_SEED=20261117
BOOTSTRAP_ROUNDS=4000
MIN_EFFECT=0.05
ARMS=("original", "supervisor_public_get")


def freeze(ids):
    roster=base.freeze(ids)
    roster["student_policy"]=TREATMENT_POLICY
    roster["original_baseline_student_policy"]=base.STUDENT_POLICY
    roster["supervisor_public_GET_only"]=True
    roster["intervention"]="Enable documented public GET supervisor APIs only; no new prompt/model/source"
    if roster["task_ids_sha256"]!=SOURCE_SHA or roster["task_hashes_sha256"]!=SOURCE_SET_HASH:
        raise ValueError("pinned DEV57 source has changed")
    return roster


def verify_frozen(meta,ids):
    if meta!=freeze(ids):
        raise ValueError("fixed source/policy/model changed after preregistration")
    return base.arrange_scenarios(ids)


def one_task(task_id,workspace):
    from appworld import AppWorld
    start=base.clock()
    with AppWorld(task_id=task_id,
                  experiment_name="small-agent-qwen9b-fixed-appworld-dev57-public-supervisor-get",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        docs=AppWorldDocsTool(world,allow_supervisor_public_reads=True)
        api=AppWorldCallTool(world,max_calls=base.MAX_API_CALLS,
                            report_http_failures=True,
                            allow_supervisor_public_reads=True)
        model=OpenAICompatModel(model="qwen3.5-9b",
                               base_url="http://127.0.0.1:8080",
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
        student=AgentRuntime(model,[docs,api],max_steps=base.MAX_STEPS).run(
            prompt,workspace)
        verdict=world.evaluate(suppress_errors=True)
        source_gold=world.task.ground_truth.api_calls
        source_depth=len(source_gold) if isinstance(source_gold,list) else None
        row=extract_stage_record(
           task_id,student,api_calls=api.calls,real_http_failures=api.http_errors,
           official_state_success=bool(verdict.success),
           state_tests_pass=int(verdict.pass_count),
           state_tests_total=int(verdict.num_tests),
           observed_reference_call_count=source_depth)
        # Count only safe tool NAME categories; no model/tool arguments exported.
        row["documented_supervisor_GET_tool_calls"]=sum(
            event.kind=="tool_call" and
            event.data.get("name")=="appworld_call" and
            isinstance(event.data.get("arguments"),dict) and
            event.data["arguments"].get("app")=="supervisor" and
            event.data["arguments"].get("api")!="complete_task"
            for event in student.trace)
        row["student_elapsed_raw_wall_s"]=round((base.clock()-start)/1e9,2)
        return row


def run_shard(meta,shard,workspace):
    if shard not in range(SCENARIOS):
        raise ValueError("must use exactly the original 19 scenario shards")
    from appworld import load_task_ids
    tasks=verify_frozen(meta,load_task_ids("dev"))[shard]
    cases=[]
    for index,tid in enumerate(tasks):
        try:
            row=one_task(tid,workspace / str(index))
        except Exception as exc:
            raise RuntimeError("private AppWorld student execution error: "+
                               type(exc).__name__) from None
        cases.append(row)
        print("APPWORLD_DEV57_PUBLIC_SUPERVISOR_GET_SCENE",shard,
              "task",index+1,"of",len(tasks),
              "state_success",int(row["official_state_success"]),
              "public_read_calls",row["documented_supervisor_GET_tool_calls"],
              flush=True)
    return {"status":"COMPLETE_ORIGINAL_QWEN_DEV57_PUBLIC_SUPERVISOR_GET_ONE_SCENE",
      "protocol":PROTOCOL,"source_sha256":SOURCE_SHA,
      "task_hashes_sha256":SOURCE_SET_HASH,
      "policy":TREATMENT_POLICY,"model_sha256":MODEL_GGUF_SHA,
      "shard":shard,"shards":SCENARIOS,"n":3,"cases":cases,
      "no_training":True,"no_oracle_in_model":True,"private_supervisor_denied":True}


def validate_scene_parts(parts,meta,*, arm):
    if len(parts)!=SCENARIOS or sorted(x.get("shard") for x in parts)!=list(range(SCENARIOS)):
        raise ValueError("need ALL original 19 three-task scenario shards")
    if arm not in ARMS:raise ValueError("unexpected paired arm")
    policy=base.STUDENT_POLICY if arm=="original" else TREATMENT_POLICY
    status=("COMPLETE_FIXED_APPWORLD_DEV57_ONE_SCENARIO_SHARD"
       if arm=="original" else "COMPLETE_ORIGINAL_QWEN_DEV57_PUBLIC_SUPERVISOR_GET_ONE_SCENE")
    for p in parts:
        if (p.get("status")!=status or p.get("policy")!=policy or
            p.get("protocol")!=PROTOCOL or
            p.get("model_sha256")!=MODEL_GGUF_SHA or
            p.get("n")!=3 or len(p.get("cases",[]))!=3 or
            p.get("task_hashes_sha256")!=SOURCE_SET_HASH or
            p.get("shards")!=SCENARIOS):
            raise ValueError("model/source/arm/shard drift")
        if arm=="original":
            if p.get("fixed_source_sha256")!=SOURCE_SHA or
               p.get("no_training") is not True or
               p.get("student_never_saw_reference") is not True:
                raise ValueError("original baseline evidence invalid")
        else:
            if (p.get("source_sha256")!=SOURCE_SHA or
                p.get("no_training") is not True or
                p.get("no_oracle_in_model") is not True or
                p.get("private_supervisor_denied") is not True):
                raise ValueError("treatment evidence invalid")
    return [row for p in parts for row in p["cases"]]


def paired_ci(pairs, *, cluster=False, rounds=BOOTSTRAP_ROUNDS):
    if not pairs or len(pairs)!=N:raise ValueError("must be exactly 57 task pairs")
    rng=random.Random(BOOTSTRAP_SEED)
    diffs=[int(b["official_state_success"])-int(a["official_state_success"])
           for a,b in pairs]
    point=sum(diffs)/len(diffs)
    by_scene=defaultdict(list)
    for a,b in pairs:
        if a["scenario_sha256"]!=b["scenario_sha256"]:
            raise ValueError("not paired same original scenario")
        by_scene[a["scenario_sha256"]].append(int(b["official_state_success"])-int(a["official_state_success"]))
    if len(by_scene)!=SCENARIOS or any(len(xs)!=3 for xs in by_scene.values()):
        raise ValueError("not 19 official three-task clusters")
    scenes=sorted(by_scene)
    draws=[]
    for _ in range(rounds):
        if cluster:
            sample=[rng.choice(scenes) for _ in scenes]
            values=[v for s in sample for v in by_scene[s]]
        else:
            values=[rng.choice(diffs) for _ in diffs]
        draws.append(sum(values)/len(values))
    draws.sort()
    return {"delta":point,"ci95_lower":draws[int(.025*rounds)],
            "ci95_upper":draws[min(rounds-1,int(.975*rounds))],
            "unit":"scenario_cluster" if cluster else "paired_task",
            "n_tasks":N,"n_scenarios":SCENARIOS,
            "bootstrap_draws":rounds}


def exact_mcnemar(rescues,regressions):
    discordant=rescues+regressions
    if not discordant:return 1.
    k=min(rescues,regressions)
    return min(1.,2*sum(math.comb(discordant,i) for i in range(k+1))/2**discordant)


def aggregate(original,treatment,meta,ids):
    if meta!=freeze(ids):
        raise ValueError("frozen source/model/policy changed")
    original_rows=validate_scene_parts(original,meta,arm="original")
    treatment_rows=validate_scene_parts(treatment,meta,arm="supervisor_public_get")
    expected={sha256(t.encode()).hexdigest() for t in ids}
    if len(expected)!=N:raise ValueError("roster task hashes duplicate")
    a={x["task_sha256"]:x for x in original_rows}
    b={x["task_sha256"]:x for x in treatment_rows}
    if len(a)!=N or len(b)!=N or set(a)!=expected or set(b)!=expected:
        raise ValueError("not all exact 57 original task IDs matched")
    pairs=[(a[k],b[k]) for k in sorted(expected)]
    ag=aggregate_dev57(original_rows,meta,expected_task_hashes=sorted(expected))
    bg=aggregate_dev57(treatment_rows,meta,expected_task_hashes=sorted(expected))
    rescues=sum(not x["official_state_success"] and y["official_state_success"]
                for x,y in pairs)
    regressions=sum(x["official_state_success"] and not y["official_state_success"]
                    for x,y in pairs)
    taskci=paired_ci(pairs)
    sceneci=paired_ci(pairs,cluster=True)
    gain=bg["task_success_rate"]-ag["task_success_rate"]
    if abs(taskci["delta"]-gain)>1e-9:raise ValueError("paired effect drift")
    return {
       "status":"COMPLETE_ORIGINAL_QWEN_DEV57_PUBLIC_SUPERVISOR_PAIRED",
       "protocol":SOURCE_BENCHMARK,
       "original_model_sha256":MODEL_GGUF_SHA,
       "fixed_main_DEV57_source_hash":SOURCE_SHA,
       "fixed_main_DEV57_task_hash_set":SOURCE_SET_HASH,
       "n_tasks":N,"n_scenarios":SCENARIOS,"physically_paired_scene_shards":SCENARIOS*2,
       "original":ag,"supervisor_public_get":bg,
       "public_supervisor_read_calls":sum(r["documented_supervisor_GET_tool_calls"] for r in treatment_rows),
       "task_rescues":rescues,"task_regressions":regressions,
       "paired_task_gain":gain,
       "mcnemar_exact_two_sided":exact_mcnemar(rescues,regressions),
       "paired_task_bootstrap95":taskci,"scenario_cluster_bootstrap95":sceneci,
       "minimal_meaningful_effect":MIN_EFFECT,
       "predeclared_positive_threshold_passed":(
          gain>=MIN_EFFECT and taskci["ci95_lower"]>0 and sceneci["ci95_lower"]>0),
       "promotion_allowed":False,"no_training":True,"no_teacher":True,
       "not_GAIA":True,"no_hidden_test":True,
       "claim_boundary":"Identical previously frozen public AppWorld DEV57 tasks and original Qwen weights; baseline sample vs one public-GET supervisor capability intervention only. All 19 scenes paired; 57 tasks are 19 correlated clusters. Reference TRAIN8 8/8 replay is source/bridge feasibility, NOT model accuracy."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["freeze","run","aggregate"])
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--workspace",type=Path,default=Path("appworld-dev57-supervisor-public-get"))
    p.add_argument("--original-inputs",nargs="+",type=Path)
    p.add_argument("--treatment-inputs",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable evidence cannot be overwritten")
    from appworld import load_task_ids
    ids=load_task_ids("dev")
    if args.mode=="freeze":
        result=freeze(ids)
        print("APPWORLD_DEV57_SUPERVISOR_GET_FROZEN",result["task_ids_sha256"],flush=True)
    elif args.mode=="run":
        if args.frozen is None or args.shard is None:p.error("need frozen/shard")
        result=run_shard(json.loads(args.frozen.read_text()),args.shard,args.workspace)
    else:
        if args.frozen is None or not args.original_inputs or not args.treatment_inputs:
            p.error("need all frozen source and paired artifacts")
        result=aggregate(
            [json.loads(x.read_text()) for x in args.original_inputs],
            [json.loads(x.read_text()) for x in args.treatment_inputs],
            json.loads(args.frozen.read_text()),ids)
        print("APPWORLD_DEV57_PUBLIC_SUPERVISOR_PAIRED_COMPLETE",
            json.dumps({k:v for k,v in result.items() if k not in ("original","supervisor_public_get")},sort_keys=True),flush=True)
        print("APPWORLD_DEV57_PUBLIC_SUPERVISOR_PAIRED_STAGE_TOTALS",
            json.dumps({a:result[a] for a in ("original","supervisor_public_get")},sort_keys=True),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    main()
