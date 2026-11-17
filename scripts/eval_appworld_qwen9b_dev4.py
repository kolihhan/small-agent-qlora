"""Original fixed Qwen3.5-9B AppWorld DEV closed-loop no-train baseline.

A task-set SHA and 4 distinct scenarios are frozen before any inference.
The student gets only task.instruction, task.allowed_apps and PUBLIC app docs
through regular function tools; official gold solution, API calls and
evaluator are never injected into model messages. Evaluation uses full
reference mode privately AFTER model stops, but task code/GT are not exposed
to Tool classes, model, CI logs or artifacts. No QLoRA, teacher or GAIA.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool,AppWorldCallTool

PROTOCOL="APPWORLD-DEV4-NO-TRAIN-PINNED-QWEN35-9B-CLOSEDLOOP-V1"
UPSTREAM="42b5bcf3cd334fee33f0c37c02070a9f5807add5"
MODEL_SHA="8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
N=4
SHARDS=4
SEED="appworld-dev4-taskid-source-only-pinned-20261117"
MAX_STEPS=20
MAX_API_CALLS=48

def choose_dev_ids(ids):
    if len(ids)<20 or len(set(ids))!=len(ids):
        raise ValueError("official AppWorld DEV IDs incomplete or duplicate")
    seen=set()
    selected=[]
    for tid in sorted(ids,key=lambda s:sha256(f"{SEED}:{s}".encode()).hexdigest()):
        scenario=tid.split("_")[0]
        if scenario in seen:continue
        seen.add(scenario)
        selected.append(tid)
        if len(selected)==N:break
    if len(selected)!=N:raise ValueError("insufficient distinct DEV source scenarios")
    return selected

def sha_records(tids):
    return sha256(json.dumps(tids,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def freeze_ids(ids):
    selection=choose_dev_ids(ids)
    return {"status":"SOURCE_FROZEN_BEFORE_QWEN_INFERENCE",
            "protocol":PROTOCOL,"split":"dev","n":N,
            "task_ids_sha256":sha_records(selection),
            "distinct_scenarios":N,"source_selection":"sha_task_id_only_not_gt",
            "upstream_commit":UPSTREAM,"original_model_gguf_sha256":MODEL_SHA,
            "max_steps":MAX_STEPS,"max_api_calls":MAX_API_CALLS,
            "no_training":True,"no_oracle_model_context":True,
            "dev_used_for_evaluation_only":True,
            "no_dev_ground_truth_loaded_during_selection":True}

def validate(meta, ids):
    expected=freeze_ids(ids)
    if meta!=expected:
        raise ValueError("frozen DEV source/model/tool contract changed")
    return choose_dev_ids(ids)

def physical_case(task_id,workspace):
    from appworld import AppWorld
    with AppWorld(task_id=task_id,experiment_name="small-agent-qwen9b-public-dev-no-train-baseline",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        # Ground truth is NOT referenced anywhere in model-facing code/tools.
        doc_tool=AppWorldDocsTool(world)
        api_tool=AppWorldCallTool(world,max_calls=MAX_API_CALLS)
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
        start=perf_counter()
        result=AgentRuntime(model,[doc_tool,api_tool],max_steps=MAX_STEPS).run(
            prompt,workspace)
        elapsed=round(perf_counter()-start,2)
        # Only after the model is finished do we use the private evaluator.
        evaluation=world.evaluate(suppress_errors=True)
        return {
          "task_sha256":sha256(task_id.encode()).hexdigest(),
          "state_success":bool(evaluation.success),
          "state_tests_pass":int(evaluation.pass_count),
          "state_tests_total":int(evaluation.num_tests),
          "student_completed_text":bool(result.completed),
          "stop_reason":result.stop_reason,"steps":result.metrics.steps,
          "tool_calls":result.metrics.tool_calls,
          "invalid_tool_calls":result.metrics.invalid_tool_calls,
          "tool_errors":result.metrics.tool_errors,
          "world_api_calls":api_tool.calls,"world_api_errors":api_tool.errors,
          "seconds":elapsed,
        }

def one_shard(frozen,shard,work):
    if not 0<=shard<SHARDS:raise ValueError("invalid paired shard")
    from appworld import load_task_ids
    ids=load_task_ids("dev")
    chosen=validate(frozen,ids)
    task=chosen[shard]
    try:
        record=physical_case(task,work)
    except Exception as exc:
        # Fail closed: no leaking decrypted task or model requests into logs.
        raise RuntimeError("physical original Qwen/AppWorld task runtime failed: "
                           +type(exc).__name__) from None
    return {"status":"REAL_QWEN_DEV_ONE_TASK_COMPLETE","protocol":PROTOCOL,
            "source_sha256":frozen["task_ids_sha256"],
            "original_model_gguf_sha256":MODEL_SHA,
            "shard":shard,"n":1,"cases":[record],
            "no_training":True,"no_QLoRA":True,"no_dev_groundtruth_in_model":True}

def aggregate(parts):
    if len(parts)!=SHARDS or sorted(x.get("shard") for x in parts)!=list(range(SHARDS)):
        raise ValueError("need all four original Qwen task results")
    if any(p.get("status")!="REAL_QWEN_DEV_ONE_TASK_COMPLETE" or
           p.get("protocol")!=PROTOCOL or
           p.get("original_model_gguf_sha256")!=MODEL_SHA or
           p.get("n")!=1 or len(p.get("cases",[]))!=1 or
           p.get("no_training") is not True or
           p.get("no_dev_groundtruth_in_model") is not True for p in parts):
        raise ValueError("partial model/source/groundtruth contamination")
    if len({p["source_sha256"] for p in parts})!=1:
        raise ValueError("source hash drift")
    rows=[p["cases"][0] for p in parts]
    if len({r["task_sha256"] for r in rows})!=N:
        raise ValueError("DEV task identity duplicated")
    return {"status":"COMPLETE_PUBLIC_DEV_ORIGINAL_QWEN_NO_TRAIN_BASELINE",
            "protocol":PROTOCOL,"n":N,
            "source_sha256":parts[0]["source_sha256"],
            "fixed_model_sha256":MODEL_SHA,
            "state_successes":sum(x["state_success"] for x in rows),
            "student_text_completed":sum(x["student_completed_text"] for x in rows),
            "official_tests_pass":sum(x["state_tests_pass"] for x in rows),
            "official_tests_total":sum(x["state_tests_total"] for x in rows),
            "model_steps":sum(x["steps"] for x in rows),
            "tool_calls":sum(x["tool_calls"] for x in rows),
            "world_api_calls":sum(x["world_api_calls"] for x in rows),
            "invalid_tool_calls":sum(x["invalid_tool_calls"] for x in rows),
            "elapsed_s":round(sum(x["seconds"] for x in rows),2),
            "source_only_task_selection":True,
            "reference_solution_not_shown_to_model":True,
            "no_training":True,"no_teacher":True,"no_GAIA":True,
            "not_gaia_or_hidden_gain":True,"promotion_allowed":False,
            "boundary":"4 public AppWorld DEV tasks on four unique scenarios selected by task IDs only, Qwen3.5-9B zero-shot, completely different API tools from GAIA agent. Public small-sample source diagnostic; no QLoRA gain or GAIA transfer claim.",
            "cases":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--work",type=Path,default=Path("appworld-dev-qwen9b-task"))
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("do not overwrite immutable result")
    if a.source_only:
        from appworld import load_task_ids
        res=freeze_ids(load_task_ids("dev"))
        log={k:v for k,v in res.items() if k not in ("task_ids_sha256",)}
        print("APPWORLD_QWEN9B_DEV_SOURCE_FROZEN",json.dumps(log,sort_keys=True))
    elif a.inputs:
        res=aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("APPWORLD_QWEN9B_DEV_COMPLETE",json.dumps({k:v for k,v in res.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.frozen is None or a.shard is None:p.error("--frozen and --shard required")
        res=one_shard(json.loads(a.frozen.read_text()),a.shard,a.work)
        print("APPWORLD_QWEN9B_DEV_SHARD",a.shard,
              "success",res["cases"][0]["state_success"],
              "tool_calls",res["cases"][0]["tool_calls"],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")

if __name__=="__main__":
    main()
