"""Exact-prefix, DB-snapshot-checked AppWorld TRAIN first-HTTP401 public auth hint A/B.

Original fixed Qwen3.5-9B is replayed verbatim before FIRST tool failure,
and the only altered model-visible byte sequence is ONE appended public API
authentication-contract note. A local AppWorld DB checkpoint is hashed after
the original/fork failed request and compared in RAM; private DBs, task
instructions, arguments, raw traces and hashes are NEVER uploaded.

Feasibility screen only: these TRAIN tasks were previously exposed. Official
Task Success is the success metric, not number of HTTP failures.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, AppWorldDocsTool
from scripts.appworld_train_first_feedback_fork import (
    RecordingModel,ExactPrefixModel,new_model,prompt_for,task_roster,
    digest,wall_ns,first_observed_error,UPSTREAM_APPWORLD_REV,MODEL_GGUF_SHA,
    MAX_STEPS,MAX_API_CALLS,
)

PROTOCOL="APPWORLD-QWEN35-TRAIN-EXACT-PREFIX-SNAPSHOT-PUBLIC401-AUTH-V1"


def checkpoint_state_hash(world,state_id):
    """Hash complete persisted simulator state privately; upload equality only."""
    world.save_state(state_id)
    root=Path(world.output_checkpoints_directory)/state_id
    if not root.is_dir():
        raise RuntimeError("state checkpoint directory missing")
    files=sorted(f for f in root.rglob("*") if f.is_file())
    if not files:
        raise RuntimeError("empty simulator state checkpoint")
    h=sha256()
    for f in files:
        # Reproducible path-normalized content identity, no filename/content export.
        relative=f.relative_to(root).as_posix()
        h.update(relative.encode())
        with f.open("rb") as handle:
            while True:
                chunk=handle.read(1024*1024)
                if not chunk:break
                h.update(chunk)
    return h.digest()


class CaptureFirst401Original(AppWorldCallTool):
    def __init__(self,world):
        super().__init__(world,max_calls=MAX_API_CALLS,report_http_failures=True,
                         capture_401_auth_hint=True)
        self.state_hash=None
        self.state_snapshot_taken=False

    def run(self,args,workspace):
        previous=len(self.public_401_auth_candidates)
        result=super().run(args,workspace)
        if len(self.public_401_auth_candidates)>previous and not self.state_snapshot_taken:
            self.state_hash=checkpoint_state_hash(self.world,"orig_first_public401")
            self.state_snapshot_taken=True
        return result


class OneShotPublic401AuthTool(AppWorldCallTool):
    def __init__(self,world,target_call_index):
        super().__init__(world,max_calls=MAX_API_CALLS,report_http_failures=True)
        self.target_call_index=target_call_index
        self.hint_was_delivered=False
        self.state_hash=None

    def run(self,arguments,workspace):
        current=self.calls+1
        target=current==self.target_call_index
        self.public_401_auth_hint=target
        try:
            result=super().run(arguments,workspace)
        finally:
            self.public_401_auth_hint=False
        if target:
            self.state_hash=checkpoint_state_hash(self.world,"treat_first_public401")
            self.hint_was_delivered=(result.error_code=="HTTP_ERROR"
                                     and "Public API authentication:" in result.content)
        return result


def evaluate_one(index,private_output):
    from appworld import AppWorld
    roster=task_roster()
    tid=roster[index]
    t0=wall_ns()
    with AppWorld(task_id=tid,experiment_name=f"qwen35-auth401-original-{index}",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        recorder=RecordingModel(new_model())
        api=CaptureFirst401Original(world)
        api.model_step_counter=lambda:len(recorder.records)
        result=AgentRuntime(recorder,[AppWorldDocsTool(world),api],
                            max_steps=MAX_STEPS).run(prompt_for(world),private_output/"original")
        official=world.evaluate(suppress_errors=True)
        first=first_observed_error(result.trace)
        candidates=list(api.public_401_auth_candidates)
        original_snapshot=api.state_hash
        original_http_statuses=dict(api.http_status_counts)
    record={
      "protocol":PROTOCOL,"task_sha256":sha256(tid.encode()).hexdigest(),
      "roster_sha256":digest(roster),
      "scene_index":index,"appworld_sha":UPSTREAM_APPWORLD_REV,
      "original_model_gguf_sha256":MODEL_GGUF_SHA,
      "max_steps":MAX_STEPS,"max_api_calls":MAX_API_CALLS,
      "max_new_model_tokens":384,
      "original":{
          "official_task_success":bool(official.success),
          "nested_tests":[int(official.pass_count),int(official.num_tests)],
          "model_calls":len(recorder.records),"stop_reason":result.stop_reason,
          "first_observed_tool_error":first,
          "http_status_distribution":original_http_statuses,
          "documented_401_auth_candidates":len(candidates),
          "wall_s":round((wall_ns()-t0)/1e9,2)
      },
      "same_snapshot_and_prefix_fork_completed":False,
      "treatment_status":"NO_ELIGIBLE_FIRST_401_AUTH_CONTRACT",
      "no_training":True,"no_GAIA":True,
      "original_Qwen_only":True,
      "private_task_and_DB_state_never_exported":True,
    }
    # Require the FIRST tool error to be exactly one HTTP401 call (not an
    # earlier wrong-app/docs/bad-arguments event in same model step).
    if not first or first["kind"]!="HTTP_ERROR" or first["name"]!="appworld_call":
        return record
    step=first["step"]
    in_step=[e for e in result.trace if e.kind=="tool_result" and e.step==step
             and not e.data.get("ok",False)]
    if len(in_step)!=1:
        record["treatment_status"]="FIRST_ERROR_STEP_HAS_MULTIPLE_ERRORS"
        return record
    eligible=[x for x in candidates if x["model_step"]==step]
    if len(eligible)!=1 or original_snapshot is None:
        return record
    target=eligible[0]
    record["first_401_public_reason_category"]=target["safe_category"]
    record["treatment_status"]="ELIGIBLE_CHECKING_EXACT_PREFIX_AND_PRIVATE_STATE"
    if not isinstance(step,int) or not(0<step<len(recorder.records)):
        record["treatment_status"]="NO_NEXT_MODEL_TURN_AFTER_FIRST_401"
        return record
    fork_t0=wall_ns()
    with AppWorld(task_id=tid,experiment_name=f"qwen35-auth401-public-hint-{index}",
                  ground_truth_mode="full",timeout_seconds=180) as world:
        model=ExactPrefixModel(new_model(),recorder.records,step,target["public_hint"])
        api=OneShotPublic401AuthTool(world,target["api_call_index"])
        treated=AgentRuntime(model,[AppWorldDocsTool(world),api],
                             max_steps=MAX_STEPS).run(
            prompt_for(world),private_output/"public_401_single_hint")
        if not model.one_observation_delta_verified or not api.hint_was_delivered:
            raise RuntimeError("single public401 auth observation delta failed")
        if model.verified_original_prefix_calls!=step:
            raise RuntimeError("model-visible prefix not exactly identical")
        if api.state_hash is None or api.state_hash!=original_snapshot:
            # Never score or publish a 'same state' A/B when private DB bytes drift.
            raise RuntimeError("original and fork private AppWorld DB states differ")
        verdict=world.evaluate(suppress_errors=True)
        changed={
            "official_task_success":bool(verdict.success),
            "nested_tests":[int(verdict.pass_count),int(verdict.num_tests)],
            "new_autonomous_Qwen_calls":model.new_model_calls,
            "same_private_state_snapshot_verified":True,
            "same_model_visible_prefix_verified":True,
            "exactly_one_public_401_hint_added":True,
            "stop_reason":treated.stop_reason,
            "tool_errors_after_hint":treated.metrics.tool_errors,
            "wall_s":round((wall_ns()-fork_t0)/1e9,2),
        }
    record.update({
      "same_snapshot_and_prefix_fork_completed":True,
      "treatment_status":"EXACT_PREFIX_AND_PRIVATE_SNAPSHOT_MATCH",
      "treatment":changed,
      "official_task_rescue":not record["original"]["official_task_success"] and changed["official_task_success"],
      "official_task_regression":record["original"]["official_task_success"] and not changed["official_task_success"],
    })
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-index",required=True,type=int,choices=(0,1,3))
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("original feasibility artifact immutable")
    try:
        d=evaluate_one(args.task_index,args.out.parent/f"private-{args.task_index}")
    except Exception as exc:
        print("PUBLIC401_FIRST_ERROR_FAIL_CLOSED",type(exc).__name__,flush=True)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print("PUBLIC401_FEASIBILITY",json.dumps({
        "scene":d["scene_index"],
        "original_task_success":d["original"]["official_task_success"],
        "public401_candidates":d["original"]["documented_401_auth_candidates"],
        "fork_status":d["treatment_status"],
        "paired_complete":d["same_snapshot_and_prefix_fork_completed"],
        "treatment_success":d.get("treatment",{}).get("official_task_success"),
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
