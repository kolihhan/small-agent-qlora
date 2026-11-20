"""Real AppWorld TRAIN source-frozen Qwen9B first-failure trace taxonomy.

A task-scoped original agent executes exactly its existing 20-turn/384-new-token
scaffold. Capture only anonymized numeric HTTP statuses, tool error categories,
tool order and official state outcomes; no task instructions, API names, parameter
names, values, model answers, model messages, evaluator content or API bodies
can enter uploaded JSON. This is an OBSERVATIONAL pilot, not an A/B and not
proof that the first error caused failure.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, AppWorldDocsTool
from gaia_small_agent.tools.appworld_auth_chain_audit import first_401_action_chain
from scripts.appworld_train_first_feedback_fork import (
    RecordingModel, new_model, prompt_for, task_roster,
    digest, wall_ns, UPSTREAM_APPWORLD_REV, MODEL_GGUF_SHA,
    MAX_STEPS, MAX_API_CALLS,
)

PROTOCOL="APPWORLD-QWEN35-TRAIN4-AUTH-ACTION-CHAIN-OBSERVATIONAL-V1"
ALLOWED_CODES=frozenset({
    "HTTP_ERROR","API_DENIED","APP_DENIED","API_NOT_FOUND",
    "APPWORLD_API_ERROR","BAD_ARGUMENTS","DUPLICATE_CALL",
    "UNKNOWN_TOOL","TOOL_EXCEPTION","CALL_LIMIT",
})

def safe_code(code):
    return code if code in ALLOWED_CODES else "OTHER_TOOL_ERROR"

def first_failure(trace):
    for e in trace:
        if e.kind=="tool_result" and not e.data.get("ok",False):
            return {"step":e.step,"tool":e.data.get("name") if
                    e.data.get("name") in ("appworld_docs","appworld_call")
                    else "other","code":safe_code(e.data.get("error_code"))}
    return None

def trajectory(trace,http_events,max_steps):
    errors=Counter()
    successes=Counter()
    abstract=[]
    http_by_step={}
    for ev in http_events:
        step=ev.get("model_step")
        if step is not None:
            http_by_step.setdefault(step,[]).append(int(ev["http_status"]))
    for e in trace:
        if e.kind!="tool_result":
            continue
        name=e.data.get("name")
        tool=name if name in ("appworld_docs","appworld_call") else "other"
        ok=bool(e.data.get("ok"))
        code=None if ok else safe_code(e.data.get("error_code"))
        if ok:successes[tool]+=1
        else:errors[code]+=1
        item={"step":int(e.step),"tool":tool,"outcome":"success" if ok else code}
        if code=="HTTP_ERROR":
            # Only numeric statuses on the same step; no exception/body values.
            item["http_statuses_on_step"]=http_by_step.get(e.step,[])
        abstract.append(item)
    first=first_failure(trace)
    after=[r for r in abstract if first and r["step"]>first["step"]]
    return {
        "first_observed_tool_failure":first,
        "observed_error_counts":dict(sorted(errors.items())),
        "successful_tool_result_counts":dict(sorted(successes.items())),
        "http_error_counts":dict(sorted(Counter(
            str(ev["http_status"]) for ev in http_events).items())),
        "http_error_events":[{
            "step":ev["model_step"],"status":ev["http_status"],
            "api_call_index":ev["api_call_index"]}
            for ev in http_events],
        "abstract_tool_trajectory":abstract,
        "any_successful_API_after_first_error":any(
            x["tool"]=="appworld_call" and x["outcome"]=="success"
            for x in after),
        "first_error_repeated_afterwards":sum(
            x["outcome"]==first["code"] for x in after) if first else 0,
        "multiple_errors_on_first_step":(
            sum(x["step"]==first["step"] and x["outcome"]!="success"
                for x in abstract)>1 if first else False),
        "causal_first_blocker_verified":False,
        "root_cause_requires_private_manual_adjudication":True
    }

def evaluate_index(index,private_workspace):
    from appworld import AppWorld
    roster=task_roster()
    tid=roster[index]
    t0=wall_ns()
    with AppWorld(task_id=tid,experiment_name=f"qwen35-train-anonymized-rootcause-{index}",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        recorder=RecordingModel(new_model())
        api=AppWorldCallTool(world,max_calls=MAX_API_CALLS,report_http_failures=True)
        api.model_step_counter=lambda:len(recorder.records)
        result=AgentRuntime(recorder,[AppWorldDocsTool(world),api],
                            max_steps=MAX_STEPS).run(
            prompt_for(world),private_workspace / "original")
        official=world.evaluate(suppress_errors=True)
        first_events=trajectory(result.trace,api.http_error_events,MAX_STEPS)
        auth_action_chain=first_401_action_chain(result.trace,world,api.http_error_events,MAX_STEPS,result.stop_reason)
        forced=bool(recorder.records and recorder.records[-1]["kind"]=="complete_final")
        result_data={
            "protocol":PROTOCOL,
            "scene_index":index,
            "task_sha256":sha256(tid.encode()).hexdigest(),
            "source_roster_sha256":digest(roster),
            "pinned_appworld_revision":UPSTREAM_APPWORLD_REV,
            "pinned_original_qwen_gguf_sha256":MODEL_GGUF_SHA,
            "model_max_new_tokens_per_turn":384,
            "max_agent_steps":MAX_STEPS,"max_api_calls":MAX_API_CALLS,
            "task_success_official":bool(official.success),
            "nested_state_tests":[int(official.pass_count),int(official.num_tests)],
            "model_calls":len(recorder.records),
            "last_model_call_forced_final_without_tools":forced,
            "model_384_token_stop_frequency":"NOT_INSTRUMENTED",
            "first_401_auth_action_chain":auth_action_chain,
            "typed_model_error_category":auth_action_chain["model_error_category"],
            "actual_api_calls":api.calls,
            "actual_http_errors":api.http_errors,
            "runner_stop_reason":result.stop_reason,
            "total_original_wall_s":round((wall_ns()-t0)/1e9,2),
            **first_events,
            "benchmark_A_B_counterfactual_performed":False,
            "new_adapter_training":False,
            "no_new_model_facing_observation_or_prompt":True,
            "GAIA_not_evaluated":True,
            "no_protected_task_or_raw_model_observation_export":True,
        }
    assert sum(int(v) for v in result_data["http_error_counts"].values())==api.http_errors
    return result_data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-index",required=True,type=int,choices=range(4))
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable observational output")
    try:
        row=evaluate_index(args.task_index,args.out.parent/f"private-{args.task_index}")
    except Exception as e:
        print("SAFE_TRAIN4_ROOTCAUSE_CENSUS_FAILED_TYPE",type(e).__name__,flush=True)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
    print("SAFE_TRAIN4_FIRST_ERRORS",json.dumps({
        "shard":row["scene_index"],
        "official_task_success":row["task_success_official"],
        "http_status_histogram":row["http_error_counts"],
        "first_failure":row["first_observed_tool_failure"],
        "safe_auth_chain":row["first_401_auth_action_chain"],
        "model_steps":row["model_calls"],
        "forced_last_final":row["last_model_call_forced_final_without_tools"],
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
