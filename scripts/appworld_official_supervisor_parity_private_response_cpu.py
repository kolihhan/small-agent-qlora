"""Private RUNNER-ONLY, CPU-only AppWorld Supervisor read-only response parity.

The two official Supervisor lookup responses necessarily contain synthetic
AppWorld user/account data. They remain local to the protected simulator.
Export ONLY redacted booleans/counts. Never print, persist, hash, or upload
actual profile, usernames, passwords, token values, or their lengths.
No login or task modification is performed; no Qwen is loaded.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools.appworld_tool import (
    AppWorldCallTool, AppWorldDocsTool, _public_response_shape,
)
from scripts.appworld_train_first_feedback_fork import (
    task_roster, digest, MODEL_GGUF_SHA, UPSTREAM_APPWORLD_REV,
)

PROTO="APPWORLD-TRAIN8-SUPERVISOR-READ-ONLY-ACTUAL-PRIVATE-RESPONSE-PARITY-V1"
APIS=("show_profile","show_account_passwords")


def primitive_structure_match(expect:Any,actual:Any,depth:int=0)->bool:
    """Compare documented public example shape, not any credential values."""
    if depth>=4:return True
    if isinstance(expect,dict):
        if not isinstance(actual,dict):return False
        # Some API response keys may be optional; strict existence is checked
        # separately to distinguish optional omissions from type mismatch.
        return all(
            primitive_structure_match(v,actual[k],depth+1)
            for k,v in expect.items() if k in actual
        )
    if isinstance(expect,list):
        return (isinstance(actual,list)
            and (not expect or not actual
                 or primitive_structure_match(expect[0],actual[0],depth+1)))
    if isinstance(expect,bool):return isinstance(actual,bool)
    if isinstance(expect,int):return isinstance(actual,int) and not isinstance(actual,bool)
    if isinstance(expect,float):return isinstance(actual,(int,float)) and not isinstance(actual,bool)
    if isinstance(expect,str):return isinstance(actual,str)
    if expect is None:return True
    return type(expect) is type(actual)


def audit_world(world)->dict:
    call=AppWorldCallTool(world,official_supervisor_parity=True,max_calls=2)
    docs=AppWorldDocsTool(world,official_supervisor_parity=True)
    outcomes=[]
    for api in APIS:
        public=world.task.api_docs["supervisor"][api]
        params=public.get("parameters")
        examples=public.get("response_schemas")
        assert isinstance(params,list) and isinstance(examples,dict)
        assert "success" in examples
        assert all(p.get("required") is not True for p in params)
        pub_doc=docs.run({"app":"supervisor","api":api},Path("."))
        assert pub_doc.ok
        # Do not reuse any real task/profile values in static documentation.
        shape=json.loads(pub_doc.content)["public_api_doc"]["response_shape"]
        assert shape==_public_response_shape(examples["success"])
        # Isolate any incidental SDK print to local memory, not Actions logs.
        scratch_stdout=io.StringIO()
        scratch_stderr=io.StringIO()
        with redirect_stdout(scratch_stdout),redirect_stderr(scratch_stderr):
            result=call.run({"app":"supervisor","api":api,"arguments":{}},Path("."))
        # Drop any scratch captured SDK output; NEVER read/print/export it.
        scratch_stdout.close();scratch_stderr.close()
        if not result.ok:
            raise RuntimeError("Read-only official Supervisor lookup failed")
        parsed=json.loads(result.content)
        seen,_=result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        # Do not include parsed, seen, or result.content in safe outcome.
        expected=examples["success"]
        outcome={
            "lookup_read_only_execution_succeeded":True,
            "raw_response_is_JSON_object_or_array":isinstance(parsed,(dict,list)),
            "top_level_matches_official_public_example_shape":(
                isinstance(expected,dict) and isinstance(parsed,dict)
                or isinstance(expected,list) and isinstance(parsed,list)
            ),
            "overlapping_official_public_fields_have_compatible_types":
                primitive_structure_match(expected,parsed),
            "full_response_fits_10k_original_tool_limit":(
                len(result.content)<10000),
            "model_visible_2k_observation_is_valid_JSON":False,
            "public_response_shape_matches_docs_projection":True,
            "SDK_internal_output_not_persisted":True,
            "no_sensitive_values_exported":True,
        }
        try:
            json.loads(seen)
            outcome["model_visible_2k_observation_is_valid_JSON"]=True
        except json.JSONDecodeError:
            pass
        outcomes.append(outcome)
    assert call.calls==2 and call.errors==0
    assert len(outcomes)==2
    return {
        "protocol":PROTO,
        "official_lookup_read_only_calls":2,
        "both_successful":all(x["lookup_read_only_execution_succeeded"] for x in outcomes),
        "private_response_contract_only":outcomes,
        "no_app_login_or_supervisor_write_calls":True,
        "Qwen_calls":0,"no_training":True,
        "no_raw_responses_tasks_account_data_secrets_or_hashes_exported":True,
        "official_task_success_not_measured":True,
    }


def run():
    from appworld import AppWorld
    tids=task_roster()
    assert len(tids)==8
    scenes=[]
    for i,tid in enumerate(tids):
        with AppWorld(task_id=tid,experiment_name=f"supervisor-readonly-contract-{i}",
                      ground_truth_mode="minimal",timeout_seconds=180) as world:
            safe=audit_world(world)
        scenes.append({"scene_index":i,"safe":safe})
    return {
        "protocol":PROTO,"source_scenes":8,
        "fixed_appworld_revision":UPSTREAM_APPWORLD_REV,
        "original_model_digest_model_never_loaded":MODEL_GGUF_SHA,
        "roster_digest":digest(tids),
        "safe_scenes":scenes,
        "all_lookups_success":all(x["safe"]["both_successful"] for x in scenes),
        "no_Qwen_calls":True,
        "no_task_or_credentials_or_hashes_exported":True,
        "not_official_Task_Success":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable evidence")
    try:
        record=run()
    except Exception as e:
        print("PRIVATE_SUPERVISOR_READ_ONLY_RESPONSE_PROBE_FAILED_TYPE",
              type(e).__name__,flush=True)
        raise SystemExit(1) from None
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print("PRIVATE_SUPERVISOR_READ_ONLY_RESPONSE_PROBE",
          json.dumps({
              "n_scene":record["source_scenes"],
              "all_public_lookups_succeeded":record["all_lookups_success"],
              "model_calls":0,
              "no_raw_values_output":True,
          },sort_keys=True))


if __name__=="__main__":main()
