"""Privacy-safe first HTTP401 auth dependency chain from ORIGINAL agent trace.

Trace inputs stay exclusively in the private AppWorld runner. The returned
record contains only predeclared booleans, bounded counts, safe error
categories and step numbers: no task, API name, credentials, arguments, model
messages, evaluator data, request or HTTP response body may be exported.
"""
from __future__ import annotations
from collections import Counter
from typing import Any

from gaia_small_agent.tools.appworld_public_auth_observation import documented_auth_observation

SAFE_CATEGORIES=frozenset({
    "REQUIRED_ACCESS_TOKEN_ABSENT",
    "REQUIRED_ACCESS_TOKEN_PRESENT_REJECTED",
    "OPTIONAL_TOKEN_SUPPLIED_REJECTED",
    "UNDETERMINED_PUBLIC_AUTH_CAUSE",
})


def _call_info(event: Any):
    if event.kind!="tool_call":
        return None
    data=event.data if isinstance(event.data,dict) else {}
    args=data.get("arguments")
    if not isinstance(args,dict):
        return None
    app=args.get("app"); api=args.get("api")
    if not isinstance(app,str) or not isinstance(api,str):
        return None
    return {"id":data.get("id"),"name":data.get("name"),
            "app":app,"api":api,
            "data":args.get("arguments") if isinstance(args.get("arguments"),dict) else {}}


def first_401_action_chain(trace:list,world:Any,http_error_events:list[dict],
                           max_steps:int,stop_reason:str)->dict:
    """Find FIRST actual numeric HTTP401 and classify causal prerequisites.

    If multiple 401s share a model step, only report ambiguous status and
    do not infer a matching API. This function never calls tools/model.
    """
    statuses=[e for e in http_error_events if e.get("http_status")==401]
    if not statuses:
        return {"first401_observed":False,"first401_eligible_for_root_review":False,
                "stop_reason":stop_reason,"model_error_category":_model_error(trace)}
    first_status=min(e["model_step"] for e in statuses
                     if isinstance(e.get("model_step"),int))
    first_401s=[(i,e) for i,e in enumerate(trace) if
        e.kind=="tool_result" and e.step==first_status
        and e.data.get("name")=="appworld_call"
        and e.data.get("error_code")=="HTTP_ERROR"
        and "HTTP 401" in str(e.data.get("content",""))]
    if len(first_401s)!=1 or sum(e.get("model_step")==first_status for e in statuses)!=1:
        return {"first401_observed":True,"first401_eligible_for_root_review":False,
                "ambiguous_same_step_401":True,"first401_step":first_status,
                "remaining_decision_steps":max(0,max_steps-first_status),
                "stop_reason":stop_reason,"model_error_category":_model_error(trace)}
    ix, result=first_401s[0]
    # Resolve tool-call by exact call ID, never by app/identifier guessing.
    calls={}
    for e in trace[:ix]:
        c=_call_info(e)
        if c and c["id"] is not None:
            calls[c["id"]]=c
    target=calls.get(result.data.get("id"))
    if target is None or target["name"]!="appworld_call":
        return {"first401_observed":True,"first401_eligible_for_root_review":False,
                "matching_original_call_missing":True,
                "first401_step":first_status,
                "stop_reason":stop_reason,"model_error_category":_model_error(trace)}
    app=target["app"]; api=target["api"]; data=target["data"]
    public_map=world.task.api_docs.get(app,{})
    public_doc=public_map.get(api,{}) if isinstance(public_map,dict) else {}
    auth=documented_auth_observation(
        401,public_doc,data,isinstance(public_map,dict) and "login" in public_map)
    category=auth[0] if auth else "UNDETERMINED_PUBLIC_AUTH_CAUSE"
    assert category in SAFE_CATEGORIES
    # Only observations that occurred strictly BEFORE the first failing
    # tool_result, not merely generated in the same model step after it.
    prior=trace[:ix]
    docs_viewed=False
    token_was_visible=False
    login_doc_viewed=False
    login_attempted=False
    login_success=False
    docs_count=0
    call_kind_by_id={}
    for e in prior:
        c=_call_info(e)
        if c and c["name"] in ("appworld_docs","appworld_call"):
            call_kind_by_id[c["id"]]=c
            if c["name"]=="appworld_call" and c["app"]==app and c["api"]=="login":
                login_attempted=True
        if e.kind!="tool_result":
            continue
        previous=call_kind_by_id.get(e.data.get("id"))
        if not previous:
            continue
        if e.data.get("name")=="appworld_docs" and e.data.get("ok"):
            docs_count+=1
            if previous["app"]==app and previous["api"]==api:
                docs_viewed=True
                # The exact model-visible bounded text, not whole public doc.
                token_was_visible=token_was_visible or (
                    '"name": "access_token"' in str(e.data.get("content","")))
            if previous["app"]==app and previous["api"]=="login":
                login_doc_viewed=True
        if e.data.get("name")=="appworld_call" and previous["app"]==app and previous["api"]=="login":
            login_success=login_success or bool(e.data.get("ok"))
    return {
        "first401_observed":True,
        "first401_eligible_for_root_review":True,
        "first401_step":first_status,
        "remaining_decision_steps":max(0,max_steps-first_status),
        "relevant_api_docs_actually_viewed":docs_viewed,
        "access_token_requirement_visible_in_actual_view":token_was_visible,
        "login_api_docs_actually_viewed":login_doc_viewed,
        "login_api_attempted":login_attempted,
        "login_api_succeeded":login_success,
        "request_supplied_nonempty_token":bool(data.get("access_token")),
        "public_doc_has_required_token":category.startswith("REQUIRED_"),
        "public_auth_explanation_class":category,
        "other_successful_docs_before_401":docs_count,
        "stop_reason":stop_reason,
        "model_error_category":_model_error(trace),
        "first_tool_error_need_not_be_causal":True,
        "no_credentials_or_identifiers_exported":True,
    }


def _model_error(trace):
    for e in trace:
        if e.kind=="model_error":
            code=e.data.get("diagnostic_code")
            if isinstance(code,str) and code.startswith("MODEL_") and len(code)<90:
                return code
            return "MODEL_UNCLASSIFIED"
    return None
