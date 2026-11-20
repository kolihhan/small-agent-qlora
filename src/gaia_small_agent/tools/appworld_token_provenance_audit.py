"""Token-source and post-401 recovery adjudication INSIDE the private AppWorld runner.

One-sided matching only: a token matching visible task text or a prior successful
API's exact JSON field is evidence of occurrence, NOT proof of authenticity.
Unmatched != invented. No token or hash, app/API name, original task, tool
arguments or model text may leave this function. No model/tool side effects.
"""
from __future__ import annotations

import json
from typing import Any

from gaia_small_agent.tools.appworld_auth_chain_audit import _call_info

_PROTO="APPWORLD-FIRST401-TOKEN-ORIGIN-AND-RECOVERY-PRIVATE-BOOLEAN-V1"
_MAX_NESTED_JSON=32768
_MIN_TOKEN_CHARS=8


def _json_token_values(content: str) -> list[str]:
    """Accept only access_token VALUES from a successfully observed API JSON."""
    if not isinstance(content,str) or len(content)>_MAX_NESTED_JSON:
        return []
    try:
        obj=json.loads(content)
    except (ValueError,TypeError):
        return []
    found=[]
    def walk(value,depth):
        if depth>8:
            return
        if isinstance(value,dict):
            for key,v in value.items():
                if key=="access_token" and isinstance(v,str):
                    found.append(v)
                if isinstance(v,(dict,list)):
                    walk(v,depth+1)
                elif isinstance(v,str) and len(v)<_MAX_NESTED_JSON and v.lstrip().startswith(("{","[")):
                    try:
                        walk(json.loads(v),depth+1)
                    except ValueError:
                        pass
        elif isinstance(value,list):
            for v in value[:100]:
                walk(v,depth+1)
    walk(obj,0)
    return found


def token_origin_after_first_401(
    trace:list, model_visible_task:str, http_error_events:list[dict],
    *, max_steps:int=20
)->dict:
    """Report safe provenance booleans only for one uniquely mapped HTTP401."""
    default={
        "protocol":_PROTO,"first401_present":False,
        "eligible_for_provenance":False,"source_class":"NO_OBSERVED_401",
        "post401_login_attempted":False,"post401_login_succeeded":False,
        "post401_login_attempt_count":0,
        "token_value_or_hash_exported":False,
        "task_or_API_name_exported":False,
        "no_inference_or_tool_behavior_change":True,
    }
    e401=[e for e in http_error_events if e.get("http_status")==401
          and isinstance(e.get("model_step"),int)]
    if not e401:return default
    step=min(e["model_step"] for e in e401)
    result_positions=[(i,e) for i,e in enumerate(trace)
           if e.kind=="tool_result" and e.step==step
           and e.data.get("name")=="appworld_call"
           and e.data.get("error_code")=="HTTP_ERROR"
           and "HTTP 401" in str(e.data.get("content",""))]
    default.update(first401_present=True,first401_step=step,
                   remaining_decision_steps=max(0,max_steps-step))
    if sum(x["model_step"]==step for x in e401)!=1 or len(result_positions)!=1:
        default["source_class"]="AMBIGUOUS_FIRST_401_EVENT"
        return default
    i,target_result=result_positions[0]
    preceding_calls={}
    for e in trace[:i]:
        info=_call_info(e)
        if info and info["id"] is not None:
            preceding_calls[info["id"]]=info
    target=preceding_calls.get(target_result.data.get("id"))
    if not target or target["name"]!="appworld_call":
        default["source_class"]="UNMATCHED_FAILING_API_CALL"
        return default
    token=target["data"].get("access_token")
    app=target["app"]
    default["eligible_for_provenance"]=True
    if not isinstance(token,str) or not token.strip():
        default.update(source_class="NO_TOKEN_SUPPLIED",
                       supplied_nonempty_token=False,
                       exact_token_in_model_visible_task=False,
                       exact_token_in_prior_successful_API_result=False,
                       exact_token_in_prior_login_result=False)
    elif len(token.strip())<_MIN_TOKEN_CHARS:
        default.update(source_class="SHORT_TOKEN_PROVENANCE_UNDETERMINED",
                       supplied_nonempty_token=True,
                       exact_token_in_model_visible_task=False,
                       exact_token_in_prior_successful_API_result=False,
                       exact_token_in_prior_login_result=False)
    else:
        # Source text may contain a token without it being valid for the API;
        # this records *occurrence*, never credential validity.
        in_task=isinstance(model_visible_task,str) and token in model_visible_task
        in_result=False
        in_login=False
        for ev in trace[:i]:
            if ev.kind!="tool_result" or not ev.data.get("ok") or ev.data.get("name")!="appworld_call":
                continue
            parent=preceding_calls.get(ev.data.get("id"))
            if not parent or parent["name"]!="appworld_call":
                continue
            if token in _json_token_values(str(ev.data.get("content",""))):
                in_result=True
                if parent["app"]==app and parent["api"]=="login":
                    in_login=True
        sources=int(in_task)+int(in_result)
        cls=("MODEL_VISIBLE_MULTIPLE_SOURCES" if sources==2 else
             "MODEL_VISIBLE_TASK_EXACT_OCCURRENCE" if in_task else
             "PRIOR_SUCCESSFUL_API_TOKEN_FIELD" if in_result else
             "NO_IDENTIFIABLE_MODEL_VISIBLE_SOURCE")
        default.update(source_class=cls,supplied_nonempty_token=True,
                       exact_token_in_model_visible_task=in_task,
                       exact_token_in_prior_successful_API_result=in_result,
                       exact_token_in_prior_login_result=in_login)
    # Explicitly inspect recovery AFTER the failing result; not just before.
    following_calls={}
    attempted_ids=set()
    login_success=False
    for ev in trace[i+1:]:
        info=_call_info(ev)
        if info and info["name"]=="appworld_call" and info["app"]==app and info["api"]=="login":
            following_calls[info["id"]]=info
            attempted_ids.add(info["id"])
        if ev.kind=="tool_result" and ev.data.get("id") in following_calls:
            if ev.data.get("name")=="appworld_call" and ev.data.get("ok"):
                login_success=True
    default.update(post401_login_attempted=bool(attempted_ids),
                   post401_login_succeeded=login_success,
                   post401_login_attempt_count=min(20,len(attempted_ids)))
    return default
