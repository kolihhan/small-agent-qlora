"""New-source paired native Qwen9B AppWorld DEV4 zero-training tool-error semantics.

Original baseline vs same Qwen + a single opt-in mechanism: classify actual
non-2xx AppWorld request failures as failed tool observations rather than
success. No prompt, model, step budget, API schema or tool definitions changed.
Original first 4 DEV source identities/scenarios excluded *before inference*.
Official AppWorld GT seen only by private state evaluator AFTER student run.
No raw protected AppWorld content exported. No model-training/GAIA claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool,AppWorldCallTool
from scripts import eval_appworld_qwen9b_dev4 as old

PROTOCOL="APPWORLD-DEV-NEW4-QWEN9B-BASE-VS-REAL-HTTP-ERROR-V1"
SEED="appworld-new4-original-qwen-baseline-vs-http-error-observation-20261117"
N=4
SHARDS=4
OLD_SOURCE_SHA="12d0a62d37b35ceb8fa5b9d8ebad2821712e28be34d14ecdca0d8240358b0e9c"
ARMS=("baseline","http_error")
GATE={"min_net_state_rescues":2,"max_state_regressions":0,
      "require_real_http_errors":True,"no_automatic_promotion":True}

def new_dev_ids(all_ids):
    old_ids=old.choose_dev_ids(all_ids)
    if old.sha_records(old_ids)!=OLD_SOURCE_SHA:
        raise ValueError("previous original four DEV source identities changed")
    used={tid.split("_")[0] for tid in old_ids}
    selected=[]
    for tid in sorted(all_ids,key=lambda s:sha256(f"{SEED}:{s}".encode()).hexdigest()):
        scene=tid.split("_")[0]
        if scene in used:continue
        used.add(scene)
        selected.append(tid)
        if len(selected)==N:break
    if len(selected)!=N or len(set(selected))!=N or set(selected)&set(old_ids):
        raise ValueError("not enough new DEV scenarios disjoint from original 4")
    return selected

def freeze(ids):
    selected=new_dev_ids(ids)
    return {"status":"FROZEN_NEW4_PRE_INFERENCE","protocol":PROTOCOL,"n":N,
        "source_sha256":old.sha_records(selected),
        "original_source_sha256":OLD_SOURCE_SHA,
        "original_model_sha256":old.MODEL_SHA,"upstream":old.UPSTREAM,
        "previous_DEV_scenarios_excluded":4,
        "new_distinct_DEV_scenarios":4,
        "selection_uses_only_task_IDs":True,
        "max_steps":old.MAX_STEPS,"max_api_calls":old.MAX_API_CALLS,
        "arms":list(ARMS),"gate":GATE,
        "no_training":True,"no_reference_inference":True,"not_GAIA":True}

def get_source(meta,ids):
    if meta!=freeze(ids):
        raise ValueError("frozen unseen source, model or protocol drift")
    return new_dev_ids(ids)

def error_bucket(stop,trace):
    if stop!="model_error":
        return stop
    data=next((e.data for e in reversed(trace) if e.kind=="model_error"),{})
    message=str(data.get("message") or "").lower()
    if "tool arguments json" in message or "invalid tool arguments" in message:
        return "malformed_tool_json"
    if "context" in message or "n_ctx" in message:
        return "model_context"
    if "timeout" in message or "timed out" in message:
        return "model_timeout"
    if "choices" in message or "response" in message or "json" in message:
        return "malformed_completion"
    if "server" in message or "http" in message:
        return "model_server"
    return "other_model_error"

def monotonic_raw():
    # AppWorld freezes ordinary time.perf_counter and datetime, yielding 0.0.
    # POSIX CLOCK_MONOTONIC_RAW is unaffected by the simulation time freezer.
    if hasattr(time,"CLOCK_MONOTONIC_RAW"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if hasattr(time,"CLOCK_BOOTTIME"):
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    raise RuntimeError("raw clock unavailable; do not publish 0s fake timings")

def one_case(task_id,arm,work):
    from appworld import AppWorld
    if arm not in ARMS:raise ValueError("unknown physically paired arm")
    started=monotonic_raw()
    with AppWorld(task_id=task_id,experiment_name="small-agent-appworld-raw-http-error-check",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        docs=AppWorldDocsTool(world)
        api=AppWorldCallTool(world,max_calls=old.MAX_API_CALLS,
                             report_http_failures=arm=="http_error")
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
            "USER TASK:\n"+world.task.instruction
        )
        result=AgentRuntime(model,[docs,api],max_steps=old.MAX_STEPS).run(prompt,work)
        evaluator=world.evaluate(suppress_errors=True)
        state=bool(evaluator.success)
        ntests=int(evaluator.num_tests)
        passed=int(evaluator.pass_count)
        bucket=error_bucket(result.stop_reason,result.trace)
        output={"state_success":state,"tests_pass":passed,"tests_total":ntests,
          "model_error_bucket":bucket,"stop_reason":result.stop_reason,
          "completed_text":bool(result.completed),
          "model_steps":result.metrics.steps,"tool_calls":result.metrics.tool_calls,
          "invalid_tool_calls":result.metrics.invalid_tool_calls,
          "tool_errors":result.metrics.tool_errors,
          "world_api_calls":api.calls,"world_api_errors":api.errors,
          "world_api_http_failures":api.http_errors}
    output["elapsed_s"]=round((monotonic_raw()-started)/1e9,2)
    return output

def run_shard(meta,arm,shard,work):
    if shard not in range(SHARDS) or arm not in ARMS:
        raise ValueError("mode/shard invalid")
    from appworld import load_task_ids
    task=get_source(meta,load_task_ids("dev"))[shard]
    out=one_case(task,arm,work)
    return {"status":"PHYSICAL_QWEN_NEW_DEV_ONE_TASK_COMPLETE",
       "protocol":PROTOCOL,"source_sha256":meta["source_sha256"],
       "model_sha256":old.MODEL_SHA,"arm":arm,"shard":shard,"n":1,
       "task_sha256":sha256(task.encode()).hexdigest(),
       "no_training":True,"no_reference_inference":True,
       "results":out}

def aggregate(parts):
    if len(parts)!=SHARDS*len(ARMS):raise ValueError("require 8 complete physical Qwen shards")
    slots={(p.get("arm"),p.get("shard")) for p in parts}
    if len(slots)!=len(parts) or slots!={(a,s) for a in ARMS for s in range(SHARDS)}:
        raise ValueError("duplicate/missing task × arm")
    if any(p.get("status")!="PHYSICAL_QWEN_NEW_DEV_ONE_TASK_COMPLETE"
           or p.get("protocol")!=PROTOCOL or
           p.get("model_sha256")!=old.MODEL_SHA or
           p.get("n")!=1 or p.get("no_training") is not True or
           p.get("no_reference_inference") is not True for p in parts):
        raise ValueError("source/model/no-oracle contract violated")
    if len({p["source_sha256"] for p in parts})!=1:raise ValueError("source differs across arms")
    joined={}
    for p in parts:
        key=p["shard"]
        task=p["task_sha256"]
        slot=joined.setdefault(key,{"task_sha256":task,"arms":{}})
        if slot["task_sha256"]!=task or p["arm"] in slot["arms"]:
            raise ValueError("mismatched task paired arms")
        slot["arms"][p["arm"]]=p["results"]
    if any(set(x["arms"])!=set(ARMS) for x in joined.values()):
        raise ValueError("not all eight outcomes matched")
    totals={}
    for arm in ARMS:
        rows=[x["arms"][arm] for x in joined.values()]
        totals[arm]={
            "state_successes":sum(x["state_success"] for x in rows),
            "official_tests_pass":sum(x["tests_pass"] for x in rows),
            "official_tests_total":sum(x["tests_total"] for x in rows),
            "student_text_completed":sum(x["completed_text"] for x in rows),
            "model_steps":sum(x["model_steps"] for x in rows),
            "tool_calls":sum(x["tool_calls"] for x in rows),
            "world_api_calls":sum(x["world_api_calls"] for x in rows),
            "http_failures_identified":sum(x["world_api_http_failures"] for x in rows),
            "model_error_categories":dict(Counter(x["model_error_bucket"] for x in rows)),
            "elapsed_s":round(sum(x["elapsed_s"] for x in rows),2)
        }
    rescues=sum(not x["arms"]["baseline"]["state_success"] and x["arms"]["http_error"]["state_success"]
                for x in joined.values())
    regressions=sum(x["arms"]["baseline"]["state_success"] and not x["arms"]["http_error"]["state_success"]
                    for x in joined.values())
    positive=(rescues-regressions>=GATE["min_net_state_rescues"] and
              regressions<=GATE["max_state_regressions"] and
              totals["http_error"]["http_failures_identified"]>0)
    return {"status":"COMPLETE_PUBLIC_DEV_NEW4_NO_TRAIN_PAIRED_DIAGNOSTIC",
        "protocol":PROTOCOL,"source_sha256":parts[0]["source_sha256"],
        "n":N,"physically_paired_arms":8,"totals":totals,
        "rescues":rescues,"regressions":regressions,
        "positive_screen_only":positive,"promotion_allowed":False,
        "model_sha256":old.MODEL_SHA,"no_training":True,
        "not_GAIA":True,"no_stronger_teacher":True,
        "boundary":"Source-disjoint four new AppWorld DEV tasks selected by ID only, same Qwen9B baseline vs true HTTP failure observation. Public pilot, not GAIA or trained-weight improvement.",
        "cases":[{"shard":k,**v} for k,v in sorted(joined.items())]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--frozen",type=Path)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--work",type=Path,default=Path("appworld-native-qwen-paired-dev4"))
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable source/outcome file exists")
    if a.source_only:
        from appworld import load_task_ids
        result=freeze(load_task_ids("dev"))
        print("APPWORLD_ERROR_SHIFT4_FROZEN",result["source_sha256"],flush=True)
    elif a.inputs:
        result=aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("APPWORLD_ERROR_SHIFT4_AGGREGATE",json.dumps({
          k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.frozen is None or a.arm is None or a.shard is None:
            p.error("need frozen, arm, shard")
        try:
            result=run_shard(json.loads(a.frozen.read_text()),a.arm,a.shard,a.work)
        except Exception as exc:
            raise RuntimeError("protected AppWorld student inference failed: "+type(exc).__name__) from None
        print("APPWORLD_ERROR_SHIFT4_ONE",a.arm,a.shard,result["results"]["state_success"],
              result["results"]["model_error_bucket"],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
