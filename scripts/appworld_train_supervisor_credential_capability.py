"""CPU-only AppWorld standard credential acquisition accessibility gate.

Compare only public API docs, model-visible instructions, and the ACTUAL
AppWorld two-tool bridge; never query real passwords, call a login endpoint,
read hidden task ground truth, or export task/profile/credential identifiers.

A blocked official lookup path is a genuine scaffold capability limitation.
It does NOT prove that no alternative legal task-context path exists.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from gaia_small_agent.tools.appworld_tool import (
    AppWorldDocsTool, allowed_public_api,
)
from scripts.appworld_train_first_feedback_fork import prompt_for

PROTOCOL="APPWORLD-TRAIN8-SUPERVISOR-PUBLIC-CREDENTIAL-CAPABILITY-V1"
SENSITIVE_STANDARD_APIS=("show_profile","show_account_passwords")
SUPERVISOR="supervisor"
# The names here are public API names, never credential values.


def inspect(world:Any)->dict:
    allowed=set(world.task.allowed_apps)
    docs=world.task.api_docs
    supervisor_docs=docs.get(SUPERVISOR,{})
    documented={
        name: isinstance(supervisor_docs,dict) and name in supervisor_docs
        for name in SENSITIVE_STANDARD_APIS
    }
    bridge={
        name:bool(allowed_public_api(world,SUPERVISOR,name))
        for name in SENSITIVE_STANDARD_APIS
    }
    completion=bool(allowed_public_api(world,SUPERVISOR,"complete_task"))
    model_docs=AppWorldDocsTool(world).run(
        {"app":SUPERVISOR},Path("."))
    model_lists_credential_api=any(
        name in model_docs.content for name in SENSITIVE_STANDARD_APIS
    ) if model_docs.ok else False
    model_lists_completion=(
        model_docs.ok and "complete_task" in model_docs.content
    )
    prompt=prompt_for(world)
    # This is ONLY a provenance/availability clue, NOT credential validity.
    # No task/profile bytes or hashes ever leave the local runner.
    instruction=world.task.instruction
    prompt_contains_exact_instruction=isinstance(instruction,str) and instruction in prompt
    # Official task already has a supervisor identity in task specs. The
    # selected generic prompt does not intentionally include that profile.
    supervisor=world.task.supervisor
    email=getattr(supervisor,"email",None)
    prompt_contains_supervisor_email=(
        isinstance(email,str) and len(email)>5 and email in prompt
    )
    login_apps=0
    login_required_fields=0
    login_with_required_fields=0
    login_all_documented_required_types=0
    for app in sorted(allowed-{"admin","api_docs",SUPERVISOR}):
        public=docs.get(app,{})
        if not isinstance(public,dict) or "login" not in public:
            continue
        login_apps+=1
        login_doc=public.get("login")
        params=login_doc.get("parameters") if isinstance(login_doc,dict) else None
        if not isinstance(params,list):
            continue
        required=[p for p in params if isinstance(p,dict) and p.get("required") is True]
        login_required_fields+=len(required)
        login_with_required_fields+=bool(required)
        login_all_documented_required_types+=all(
            isinstance(x.get("name"),str) and isinstance(x.get("type"),str)
            for x in required)
    all_standard_doc=all(documented.values())
    all_standard_bridge=all(bridge.values())
    status=(
        "OFFICIAL_CREDENTIAL_LOOKUP_DOCUMENTED_BUT_BRIDGE_BLOCKED"
        if all_standard_doc and not all_standard_bridge and completion
        else "PUBLIC_SUPERVISOR_CONTRACT_INCOMPLETE"
        if not all_standard_doc
        else "NO_STANDARD_BRIDGE_BLOCK_FOUND"
    )
    return {
        "protocol":PROTOCOL,
        "public_supervisor_credential_APIs_both_documented":all_standard_doc,
        "both_credential_APIs_callable_via_current_bridge":all_standard_bridge,
        "supervisor_complete_task_callable":completion,
        "model_docs_lists_credential_APIs":model_lists_credential_api,
        "model_docs_lists_supervisor_completion":model_lists_completion,
        "original_prompt_contains_task_instruction":prompt_contains_exact_instruction,
        "original_prompt_contains_supervisor_email_value":prompt_contains_supervisor_email,
        "public_login_app_count":login_apps,
        "public_login_required_parameter_count":login_required_fields,
        "public_logins_with_required_parameters":login_with_required_fields,
        "public_logins_with_required_parameter_types_documented":login_all_documented_required_types,
        "standard_credential_retrieval_capability":status,
        # This source-only audit intentionally makes no claim about an
        # alternative legitimately granted credential in task context.
        "alternate_legal_credentials_from_instruction_or_public_APIs":"UNDETERMINED",
        "credentials_validity_or_account_scope":"UNDETERMINED",
        "official_task_success_unmeasured":True,
        "no_login_or_other_simulator_API_calls":True,
        "no_task_supervisor_or_credential_values_or_hashes_exported":True,
        "no_Qwen_or_training":True,
    }


def summarize(rows:list[dict])->dict:
    assert len(rows)==8
    assert len({r["scene_index"] for r in rows})==8
    statuses=Counter(r["safe"]["standard_credential_retrieval_capability"] for r in rows)
    profiles=Counter(str((
        r["safe"]["public_supervisor_credential_APIs_both_documented"],
        r["safe"]["both_credential_APIs_callable_via_current_bridge"],
        r["safe"]["public_login_app_count"],
        r["safe"]["public_login_required_parameter_count"],
    )) for r in rows)
    return {
        "protocol":PROTOCOL,
        "source_scenes":len(rows),
        "standard_credential_lookup_status_counts":dict(sorted(statuses.items())),
        "unique_public_contract_profiles":len(profiles),
        "original_scenes_safe_only":rows,
        "no_private_instruction_or_credential_data_exported":True,
        "no_Qwen_calls":True,
        "no_login_attempt":True,
        "main_research_decision":"BRIDGE_CAPABILITY_MISSING_BEFORE_AUTH_PLANNING"
            if set(statuses)=={"OFFICIAL_CREDENTIAL_LOOKUP_DOCUMENTED_BUT_BRIDGE_BLOCKED"}
            else "ABSTAIN_WHEN_UPSTREAM_DOCS_DIFFER",
        "can_all_user_account_credentials_be_obtained_legitimately":"UNDETERMINED",
        "source_scenes_exposed_to_prior_research":True,
    }


def run()->dict:
    from appworld import AppWorld
    from scripts.appworld_train_first_feedback_fork import task_roster
    tids=task_roster()
    assert len(tids)==8
    rows=[]
    for i,tid in enumerate(tids):
        # Task loading is read-only; no model and no API calls.
        with AppWorld(task_id=tid,
                      experiment_name=f"credential-capability-sourceonly-{i}",
                      ground_truth_mode="minimal",timeout_seconds=180) as world:
            safe=inspect(world)
        rows.append({"scene_index":i,"safe":safe})
    return summarize(rows)


def main():
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("Do not replace evidence")
    try: result=run()
    except Exception as e:
        # No private exception or task text in user-visible Actions logs.
        print("CREDENTIAL_CAPABILITY_SOURCE_AUDIT_FAILED_TYPE",type(e).__name__,flush=True)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SUPERVISOR_CREDENTIAL_CAPABILITY_ONLY",json.dumps({
        "n_scenes":result["source_scenes"],
        "status_counts":result["standard_credential_lookup_status_counts"],
        "distinct_public_profiles":result["unique_public_contract_profiles"],
        "Qwen_calls":0,
        "credential_values_exported":False,
    },sort_keys=True))


if __name__=="__main__":main()
