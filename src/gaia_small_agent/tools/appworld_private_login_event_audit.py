"""Private runner-only login attempt/status/argument contract evidence.

This is NOT a retroactive decoder for the old anonymized artifact. It requires
the original full TraceEvent stream + AppWorld public docs + requester's error
events INSIDE the same protected runtime. Output only safe counts and typed
categories, never app/API names, parameter names, credentials or value hashes.

If multiple actual HTTP errors occur on the same model step, status attribution
must fail closed unless the requester has an explicit call-id mapping.
"""
from __future__ import annotations

from collections import Counter
from typing import Any
from gaia_small_agent.tools.appworld_auth_chain_audit import _call_info

PROTO="APPWORLD-PRIVATE-LOGIN-ARGUMENT-STATUS-PER-CALL-V1"


def source_only_login_status(trace:list,world:Any,http_error_events:list[dict],
                             model_visible_task:str)->dict:
    """Available only during live private runner. No calls are executed."""
    calls={}
    for e in trace:
        c=_call_info(e)
        if c and c["id"] is not None:
            calls[c["id"]]=c
    # No status inference from mere ordered lists if step identity ambiguous.
    statuses={}
    for ev in http_error_events:
        step=ev.get("model_step")
        if not isinstance(step,int) or not isinstance(ev.get("http_status"),int):
            continue
        statuses.setdefault(step,[]).append(int(ev["http_status"]))
    first_401_step=min(
       (s for s,values in statuses.items() if 401 in values),default=None)
    records=[]
    for e in trace:
        if e.kind!="tool_result" or e.data.get("name")!="appworld_call":
            continue
        c=calls.get(e.data.get("id"))
        if not c or c["name"]!="appworld_call" or c["api"]!="login":
            continue
        app=c["app"]
        public=world.task.api_docs.get(app,{})
        spec=public.get("login") if isinstance(public,dict) else None
        params=spec.get("parameters") if isinstance(spec,dict) else None
        doc_valid=isinstance(params,list) and all(
            isinstance(x,dict) and isinstance(x.get("name"),str)
            and isinstance(x.get("required"),bool) for x in params)
        supplied=c["data"]
        fields=[x for x in params if x["required"]] if doc_valid else []
        missing=sum(
            x["name"] not in supplied or supplied[x["name"]] is None
            or (isinstance(supplied[x["name"]],str) and not supplied[x["name"]].strip())
            for x in fields)
        type_mismatch=0
        if doc_valid:
            for x in fields:
                value=supplied.get(x["name"])
                typ=x.get("type")
                if value is None:
                    continue
                if (typ=="string" and not isinstance(value,str)
                    or typ=="integer" and (not isinstance(value,int) or isinstance(value,bool))
                    or typ=="boolean" and not isinstance(value,bool)
                    or typ=="number" and (not isinstance(value,(int,float)) or isinstance(value,bool))):
                    type_mismatch+=1
        empty_sensitive=missing>0 if doc_valid else None
        visible_source_matches=0
        for x in fields:
            value=supplied.get(x["name"])
            if (isinstance(value,str) and len(value)>=8
                and isinstance(model_visible_task,str) and value in model_visible_task):
                visible_source_matches+=1
        err=e.data.get("error_code")
        matching_status=statuses.get(e.step,[])
        same_step_http_results=sum(
            x.kind=="tool_result" and x.step==e.step
            and x.data.get("name")=="appworld_call"
            and x.data.get("error_code")=="HTTP_ERROR"
            for x in trace)
        if e.data.get("ok"):
            category="SUCCESSFUL_API"
            http_status=None
        elif err=="HTTP_ERROR" and len(matching_status)==1 and same_step_http_results==1:
            category="EXACT_ONE_HTTP_PER_MODEL_STEP"
            http_status=matching_status[0]
        elif err=="HTTP_ERROR":
            category="AMBIGUOUS_OR_MISSING_CALL_HTTP_ID"
            http_status=None
        else:
            category="NON_HTTP_TOOL_ERROR"
            http_status=None
        records.append({
            "model_step":e.step,
            "relative_to_first_401":(
                "UNKNOWN" if first_401_step is None else
                "BEFORE" if e.step<first_401_step else
                "AFTER" if e.step>first_401_step else "SAME_STEP"),
            "status_attribution":category,
            "http_status":http_status,
            "public_login_doc_contract_known":doc_valid,
            "required_field_count":len(fields) if doc_valid else None,
            "missing_or_empty_required_field_count":missing if doc_valid else None,
            "required_field_public_type_mismatch_count":type_mismatch if doc_valid else None,
            "supplied_required_values_matched_task_text_count":visible_source_matches,
            "required_values_from_other_context_undetermined":True,
            "credential_validity_unverified":True,
        })
    return {
        "protocol":PROTO,
        "first401_present":first_401_step is not None,
        "first401_step":first_401_step,
        "login_attempt_count":len(records),
        "login_http_status_counts":dict(sorted(
            Counter(str(x["http_status"]) for x in records
                    if x["http_status"] is not None).items())),
        "login_contract_missing_or_empty_count":sum(
            x["missing_or_empty_required_field_count"] or 0 for x in records),
        "login_public_contract_type_mismatches":sum(
            x["required_field_public_type_mismatch_count"] or 0 for x in records),
        "ambiguous_login_http_link_count":sum(
            x["status_attribution"]=="AMBIGUOUS_OR_MISSING_CALL_HTTP_ID"
            for x in records),
        "per_login_safe":records,
        "unmatched_previous_artifacts_not_automatically_repaired":True,
        "output_contains_no_argument_values_names_credentials_hashes":True,
        "no_new_model_or_tool_action":True,
    }
