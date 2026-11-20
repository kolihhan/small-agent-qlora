"""Pinned AppWorld TRAIN8: source-only OFF-vs-ON Supervisor public contract audit.

No supervisor lookup or login API is executed; there is NO raw credential,
profile, token, task instruction, private response, hash, or model invocation.
The agent-visible public API docs may be inspected locally but are not
uploaded. Export is an allowlisted set of bool/int counts only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools.appworld_tool import (
    AppWorldDocsTool, AppWorldCallTool, allowed_public_api,
)
from scripts.appworld_train_first_feedback_fork import (
    task_roster, digest, MODEL_GGUF_SHA, UPSTREAM_APPWORLD_REV,
)

PROTOCOL="APPWORLD-TRAIN8-OFFICIAL-SUPERVISOR-OPTIN-PARITY-PUBLIC-CONTRACT-V1"
AUTH_APIS=("show_profile","show_account_passwords")
ALLOWED_SUPERVISOR=set((*AUTH_APIS,"complete_task"))


def audit_world(world):
    original_docs=AppWorldDocsTool(world)
    parity_docs=AppWorldDocsTool(world,official_supervisor_parity=True)
    original_call=AppWorldCallTool(world)
    parity_call=AppWorldCallTool(world,official_supervisor_parity=True)
    original_list=json.loads(original_docs.run(
        {"app":"supervisor"},Path(".")).content)["apis"]
    parity_list=json.loads(parity_docs.run(
        {"app":"supervisor"},Path(".")).content)["apis"]
    assert original_list==["complete_task"]
    assert set(parity_list)==ALLOWED_SUPERVISOR
    public=world.task.api_docs.get("supervisor")
    assert isinstance(public,dict) and set(AUTH_APIS).issubset(public)
    assert set(world.task.allowed_apps).__contains__("supervisor")
    assert all(allowed_public_api(world,"supervisor",api,official_supervisor_parity=True)
               for api in ALLOWED_SUPERVISOR)
    assert all(not allowed_public_api(world,"supervisor",api)
               for api in AUTH_APIS)
    for other in public.keys()-ALLOWED_SUPERVISOR:
        assert not allowed_public_api(world,"supervisor",other,official_supervisor_parity=True)
        assert parity_docs.run({"app":"supervisor","api":other},Path(".")).error_code=="API_NOT_FOUND"
    contract=[]
    for api in AUTH_APIS:
        public_doc=public[api]
        assert isinstance(public_doc,dict)
        raw_params=public_doc.get("parameters")
        assert isinstance(raw_params,list)
        docs_result=parity_docs.run({"app":"supervisor","api":api},Path("."))
        assert docs_result.ok and docs_result.error_code is None
        shown=json.loads(docs_result.content)
        assert shown["app"]=="supervisor" and shown["api"]==api
        params=shown["public_api_doc"]["parameters"]
        assert len(params)==min(len(raw_params),24)
        assert all(isinstance(p,dict) and isinstance(p.get("name"),str)
                   and isinstance(p.get("type"),str)
                   and isinstance(p.get("required"),bool)
                   for p in params)
        bounded,_=docs_result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        # Public API names must survive the exact original 2k observation
        # budget, not just be present in a longer unshown response.
        assert ('"api": "'+api+'"') in bounded
        assert original_docs.run({"app":"supervisor","api":api},Path(".")).ok
        # Baseline deliberately ignores selector and returns only completion;
        # never treat that generic answer as access to the requested docs.
        original_direct=original_docs.run({"app":"supervisor","api":api},Path("."))
        assert api not in original_direct.content
        response_schemas=public_doc.get("response_schemas")
        contract.append({
            "documented":True,
            "model_can_discover_by_listing":api in parity_list,
            "model_can_read_public_parameter_contract":True,
            "api_name_survives_2000_char_tool_budget":True,
            "required_parameter_count":sum(p.get("required") is True for p in params),
            "public_response_schema_documented":isinstance(response_schemas,(list,dict)) and bool(response_schemas),
            "source_public_documented_method":isinstance(public_doc.get("method"),str),
            "source_public_documented_path":isinstance(public_doc.get("path"),str),
        })
    # Protocol parity is exactly the enabled Supervisor public documentation
    # and call allowlist. No Agent, Qwen, or Requester API execute here.
    assert original_call.calls==parity_call.calls==0
    assert original_docs.definition()["function"]["description"]!=parity_docs.definition()["function"]["description"]
    assert original_call.definition()==parity_call.definition()
    assert all(allowed_public_api(world,app,api,official_supervisor_parity=True)
               ==allowed_public_api(world,app,api)
               for app,api in (("supervisor","complete_task"),("admin","reset"),
                               ("api_docs","show_api"),("calendar","login")))
    return {
        "protocol":PROTOCOL,
        "original_supervisor_APIs_visible_count":len(original_list),
        "parity_supervisor_APIs_visible_count":len(parity_list),
        "official_lookup_APIs_accessible_in_parity_count":sum(
            allowed_public_api(world,"supervisor",api,official_supervisor_parity=True)
            for api in AUTH_APIS),
        "official_lookup_APIs_accessible_in_default_count":sum(
            allowed_public_api(world,"supervisor",api)
            for api in AUTH_APIS),
        "other_public_supervisor_APIs_remain_denied_count":len(public.keys()-ALLOWED_SUPERVISOR),
        "all_other_supervisor_public_APIs_denied":True,
        "contract_checks_only":[contract[0],contract[1]],
        "model_tool_description_changed_only_when_enabled":True,
        "tool_call_functions_schemas_unmodified":True,
        "original_Supervisor_completion_preserved":True,
        "original_Qwen_and_simulator_not_invoked":True,
        "no_real_credential_lookup_or_login_executed":True,
        "no_private_profile_credential_token_or_task_string_exported":True,
        "official_response_shape_compatibility_requires_private_runtime_verification":True,
        "task_success_not_scored":True,
    }


def run():
    from appworld import AppWorld
    tids=task_roster()
    assert len(tids)==8
    rows=[]
    for i,tid in enumerate(tids):
        with AppWorld(task_id=tid,
                      experiment_name=f"parity-public-contract-{i}",
                      ground_truth_mode="minimal",timeout_seconds=180) as world:
            data=audit_world(world)
        rows.append({"scene_index":i,"safe":data})
    result={
        "protocol":PROTOCOL,
        "source_scenes":8,
        "pinned_appworld_source_commit":UPSTREAM_APPWORLD_REV,
        "pinned_Qwen_digest_model_never_loaded":MODEL_GGUF_SHA,
        "frozen_roster_digest":digest(tids),
        "distinct_safe_public_contract_count_profiles":len({
            json.dumps(x["safe"],sort_keys=True) for x in rows}),
        "source_only_safe_rows":rows,
        "parity_default_off":True,
        "no_Qwen_or_login_or_real_credentials":True,
        "no_task_specific_instruction_or_values_exported":True,
        "study_does_not_approve_task_success_claim":True,
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable evidence file")
    try:result=run()
    except Exception as exc:
        print("SUPERVISOR_PARITY_PUBLIC_CONTRACT_FAIL_CLOSED_TYPE",
              type(exc).__name__,flush=True)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("OFF_VS_ON_PARITY_PUBLIC_CONTRACT_COUNTS",json.dumps({
        "n_scene":len(result["source_only_safe_rows"]),
        "n_profiles":result["distinct_safe_public_contract_count_profiles"],
        "on_supported_lookup_counts":[r["safe"]["official_lookup_APIs_accessible_in_parity_count"]
                 for r in result["source_only_safe_rows"]],
        "original_disabled":True,"no_Qwen":True
    },sort_keys=True))


if __name__=="__main__":main()
