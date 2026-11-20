"""Prospective fixed-Qwen AppWorld Supervisor parity paired pilot.

SECURITY: Every tool response and AgentRuntime trace may contain passwords.
Hold all of them only in the sandbox memory; never print/serialize/return them.
Only write validated aggregate_private_run dict to explicitly selected files.
All exception messages are treated as secrets, caught and discarded.

Two NEW scenario families, two repeats each, both original OFF and parity ON
with fresh isolated AppWorld context per episode. Same Qwen/model/budgets.
Parity necessarily changes model-visible tool descriptions; this is the
controlled capability treatment, not an identical-prefix experiment.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr,redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool,AppWorldCallTool
from gaia_small_agent.tools.appworld_parity_safe_export import (
    aggregate_private_run,validate_safe_report,write_safe_report,safe_exception_category,
)
from scripts.appworld_parity_preregister_new_train2 import (
    require_frozen,PROTOCOL,ARMS,N_REPEATS,N_SCENES,MAX_STEPS,
    MAX_API_CALLS,MAX_MODEL_OUTPUT_TOKENS,MODEL_TEMPERATURE,
)
from scripts.appworld_train_first_feedback_fork import task_roster,prompt_for,


PROTOCOL_RUN="APPWORLD-PARITY-FIXED-QWEN-NEW-TRAIN2-PAIRED-PILOT-SHARD-V1"
PROTOCOL_FINAL="APPWORLD-PARITY-FIXED-QWEN-NEW-TRAIN2-PAIRED-PILOT-AGGREGATE-V1"


def _elapsed(start_ns:int)->float:
    # AppWorld patches perf_counter; use raw monotonic if available.
    clk=getattr(time,"CLOCK_MONOTONIC_RAW",None)
    return (time.clock_gettime_ns(clk)-start_ns)/1e9 if clk is not None else (time.monotonic_ns()-start_ns)/1e9


def _clock()->int:
    clk=getattr(time,"CLOCK_MONOTONIC_RAW",None)
    return time.clock_gettime_ns(clk) if clk is not None else time.monotonic_ns()


def _model():
    return OpenAICompatModel(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:8080",
        timeout_s=240,
        max_new_tokens=MAX_MODEL_OUTPUT_TOKENS,
    )


def _one(task_id:str,arm:str,scene:int,rep:int,private_root:Path)->dict:
    from appworld import AppWorld
    if arm not in ARMS or rep not in range(N_REPEATS) or scene not in range(N_SCENES):
        raise ValueError("invalid frozen arm, scene, or repetition")
    enabled=arm=="parity"
    # Separate new simulator with exact same task source ID, version, mode
    # and model each time. Different experiment_name avoids state contamination.
    t0=_clock()
    with AppWorld(task_id=task_id,
                  experiment_name=f"parity-paired-new-train2-scene{scene}-r{rep}-{arm}",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        api=AppWorldCallTool(
            world,max_calls=MAX_API_CALLS,
            report_http_failures=True,
            official_supervisor_parity=enabled)
        docs=AppWorldDocsTool(world,official_supervisor_parity=enabled)
        # Intentionally do NOT instantiate RecordingModel: raw model snapshots
        # duplicate credential-bearing tool messages unnecessarily.
        student=AgentRuntime(_model(),[docs,api],max_steps=MAX_STEPS).run(
            prompt_for(world),private_root/"agent")
        # Grade only after the model is done; never give evaluator to the agent.
        result=world.evaluate(suppress_errors=True)
        success=bool(result.success)
        nested=(int(result.pass_count),int(result.num_tests))
        row=aggregate_private_run(
            result=student,api_tool=api,
            arm=arm,scene_index=scene,repeat_index=rep,
            official_task_success=success,nested_tests=nested,
            elapsed_s=_elapsed(t0))
        # Any secrets in student.trace or model messages remain local.
        validate_safe_report(row)
        del student,api,docs,result
    return row


def run_scene(scene:int,freeze:dict)->dict:
    from appworld import load_task_ids
    chosen=require_frozen(freeze,load_task_ids("train"),task_roster())
    if scene not in range(N_SCENES):
        raise ValueError("scene index not frozen")
    task=chosen[scene]
    results=[]
    with tempfile.TemporaryDirectory(prefix="appworld-private-parity-") as private_dir:
        root=Path(private_dir)
        for rep in range(N_REPEATS):
            # Alternate original/treatment order to reduce order bias without
            # changing any tool/model budgets or source data.
            order=ARMS if rep%2==0 else tuple(reversed(ARMS))
            for arm in order:
                scratch_stdout,scratch_stderr=io.StringIO(),io.StringIO()
                with redirect_stdout(scratch_stdout),redirect_stderr(scratch_stderr):
                    row=_one(task,arm,scene,rep,root/f"rep{rep}-{arm}")
                # All console output from private SDK/model context discarded.
                scratch_stdout.close();scratch_stderr.close()
                results.append(row)
    assert len(results)==N_REPEATS*len(ARMS)
    assert {(x["arm"],x["repeat_index"]) for x in results}=={
        (arm,i) for arm in ARMS for i in range(N_REPEATS)}
    assert all(x["scene_index"]==scene for x in results)
    return {
        "protocol":PROTOCOL_RUN,
        "pre_registered_source_protocol":PROTOCOL,
        "scene_index":scene,
        "n_episodes":len(results),
        "complete_independent_fresh_appworld_contexts":True,
        "same_original_model_Qwen35_9B":True,
        "no_recording_model_or_sensitive_traces_written":True,
        "no_private_task_or_password_or_token_or_hash_export":True,
        "source_and_arms_frozen_before_Qwen_scoring":True,
        "episodes":results,
    }


def aggregate(meta:dict,shards:list[dict])->dict:
    if meta.get("protocol")!=PROTOCOL:raise ValueError("frozen source protocol mismatch")
    if len(shards)!=N_SCENES or {x.get("scene_index") for x in shards}!=set(range(N_SCENES)):
        raise ValueError("incomplete paired scene shards")
    if any(x.get("protocol")!=PROTOCOL_RUN or x.get("n_episodes")!=len(ARMS)*N_REPEATS
           or not x.get("no_private_task_or_password_or_token_or_hash_export") for x in shards):
        raise ValueError("unsafe or incomplete paired shards")
    paired=[]
    totals={"baseline":0,"parity":0}
    total_seconds={"baseline":0.0,"parity":0.0}
    instability=[]
    for shard in sorted(shards,key=lambda x:x["scene_index"]):
        rows=shard["episodes"]
        for x in rows:validate_safe_report(x)
        byarmrep={(x["arm"],x["repeat_index"]):x for x in rows}
        if len(byarmrep)!=len(rows):
            raise ValueError("duplicate or missing paired episode")
        for arm in ARMS:
            outcomes=[byarmrep[arm,i]["official_task_success"] for i in range(N_REPEATS)]
            instability.append({
                "scene_index":shard["scene_index"],"arm":arm,
                "official_success_disagrees_across_repeats":len(set(outcomes))>1,
            })
        for rep in range(N_REPEATS):
            a=byarmrep["baseline",rep]
            b=byarmrep["parity",rep]
            if type(a["official_task_success"]) is not bool or type(b["official_task_success"]) is not bool:
                raise ValueError("official scoring missing")
            for arm,row in (("baseline",a),("parity",b)):
                totals[arm]+=int(row["official_task_success"])
                total_seconds[arm]+=row["elapsed_wall_seconds"]
            paired.append({
                "scene_index":shard["scene_index"],"repeat_index":rep,
                "baseline_task_success":a["official_task_success"],
                "parity_task_success":b["official_task_success"],
                "rescue":not a["official_task_success"] and b["official_task_success"],
                "regression":a["official_task_success"] and not b["official_task_success"],
                "nested_baseline":a["nested_state_tests"],
                "nested_parity":b["nested_state_tests"],
                "newly_successful_supervisor_profile_calls":
                    b["auth_action_counts"]["supervisor_profile"]["succeeded"],
                "newly_successful_supervisor_password_calls":
                    b["auth_action_counts"]["supervisor_passwords"]["succeeded"],
                "actual_login_success_baseline":a["auth_action_counts"]["app_login"]["succeeded"],
                "actual_login_success_parity":b["auth_action_counts"]["app_login"]["succeeded"],
                "wall_s_baseline":a["elapsed_wall_seconds"],
                "wall_s_parity":b["elapsed_wall_seconds"],
            })
    return {
        "protocol":PROTOCOL_FINAL,
        "pre_registered_source_protocol":PROTOCOL,
        "n_unique_task_source_scenes":N_SCENES,
        "repeats_per_arm_per_scene":N_REPEATS,
        "total_matched_pairs":len(paired),
        "official_task_success_count":totals,
        "rescues":sum(x["rescue"] for x in paired),
        "regressions":sum(x["regression"] for x in paired),
        "repeat_disagreements":instability,
        "paired_task_outcomes":paired,
        "total_wall_s_by_arm":{k:round(v,2) for k,v in total_seconds.items()},
        "source_disjoint_from_prior_train8_only":True,
        "not_proof_of_generalization_or_GAIA_transfer":True,
        "no_training_or_QLoRA":True,
        "no_sensitive_task_trace_or_password_or_token_exported":True,
    }


def main():
    a=argparse.ArgumentParser()
    modes=a.add_subparsers(dest="mode",required=True)
    s=modes.add_parser("run")
    s.add_argument("--scene",type=int,required=True)
    s.add_argument("--frozen",type=Path,required=True)
    s.add_argument("--out",type=Path,required=True)
    t=modes.add_parser("aggregate")
    t.add_argument("--frozen",type=Path,required=True)
    t.add_argument("--shard-dir",type=Path,required=True)
    t.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    if args.out.exists():raise FileExistsError("immutable report path")
    try:
        frozen=json.loads(args.frozen.read_text())
        if args.mode=="run":
            report=run_scene(args.scene,frozen)
        else:
            parts=[json.loads((args.shard_dir/f"scene-{i}.json").read_text())
                   for i in range(N_SCENES)]
            report=aggregate(frozen,parts)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        # Only finite, strict safe reports or composed safe fields are dumped.
        if args.mode=="run":
            for episode in report["episodes"]:validate_safe_report(episode)
        args.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        if args.mode=="run":
            print("ORIGINAL_QWEN_PARITY_ONE_SCENE_MATCHED_4_EPISODES",json.dumps({
                "scene_index":args.scene,"complete_pairs":2,
                "official_success_values":[int(x["official_task_success"]) for x in report["episodes"]]
            },sort_keys=True),flush=True)
        else:
            print("APPWORLD_OFFICIAL_PARITY_FIXED_QWEN_PAIRED_RESULT",json.dumps({
                "matched_pairs":report["total_matched_pairs"],
                "success":report["official_task_success_count"],
                "rescues":report["rescues"],"regressions":report["regressions"],
            },sort_keys=True),flush=True)
    except Exception as exc:
        # NEVER emit raw trace, raw exception, task identity, credential, or
        # local filename in GitHub job logs. Fail closed on incomplete pairs.
        print("APPWORLD_PARITY_PAIRED_RUN_FAILED_SAFE_TYPE",
              safe_exception_category(exc),flush=True)
        raise SystemExit(1) from None


if __name__=="__main__":main()
