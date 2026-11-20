"""Strict privacy boundary for any future AppWorld Supervisor Parity A/B.

The raw AgentRuntime trace and RecordingModel messages are SECRET-BEARING.
NEVER serialize, print, upload, pickle, or surface either object to GitHub.
This module consumes them *in memory* and exports only a fixed numeric /
boolean / enumerated result schema. It never exports task text, raw tool
arguments, responses, model generations, exception bodies, identifiers,
credentials or hashes. Does not call an API, model or evaluator.
"""
from __future__ import annotations

from dataclasses import fields
from math import isfinite
from pathlib import Path
import json
import re
from typing import Any

PROTOCOL="APPWORLD-SUPERVISOR-PARITY-STRICT-SAFE-AGGREGATE-V1"
_ARM=frozenset({"baseline","parity"})
_STOP=frozenset({
    "final","max_steps","model_timeout","model_unavailable",
    "model_error","model_capacity","empty_final","invalid_final",
})
_ERROR=frozenset({
    "HTTP_ERROR","API_DENIED","APP_DENIED","API_NOT_FOUND",
    "BAD_ARGUMENTS","DUPLICATE_CALL","UNKNOWN_TOOL","TOOL_EXCEPTION",
    "CALL_LIMIT","APPWORLD_API_ERROR",
})
_METRIC=frozenset({
    "steps","tool_calls","tool_successes","tool_errors","duplicate_calls_blocked",
    "invalid_tool_calls","tool_retries",
})
_PRIVILEGED=frozenset({"show_profile","show_account_passwords","complete_task"})


def _bounded_int(x:Any,lo:int=0,hi:int=100000)->int:
    if type(x) is not int or not lo<=x<=hi:
        raise ValueError("invalid strictly numeric safe field")
    return x


def _boolean(x:Any)->bool:
    if type(x) is not bool:
        raise ValueError("invalid strictly boolean safe field")
    return x


def aggregate_private_run(*,result:Any,api_tool:Any,
                          arm:str,scene_index:int,repeat_index:int,
                          official_task_success:bool|None=None,
                          nested_tests:tuple[int,int]|None=None,
                          elapsed_s:float|None=None)->dict:
    """Do not pass recorder/messages or any raw evaluation object here."""
    if arm not in _ARM:
        raise ValueError("unrecognized experiment arm")
    scene=_bounded_int(scene_index,0,9999)
    repeat=_bounded_int(repeat_index,0,99)
    if official_task_success is not None:
        _boolean(official_task_success)
    if (nested_tests is not None and (type(nested_tests) is not tuple
        or len(nested_tests)!=2)):
        raise ValueError("invalid nested-test pair")
    nested=None
    if nested_tests is not None:
        nested=[_bounded_int(x) for x in nested_tests]
        if nested[0]>nested[1]:
            raise ValueError("nested successes exceed total")
    if elapsed_s is not None:
        if (type(elapsed_s) not in (float,int) or not isfinite(elapsed_s)
            or elapsed_s<0 or elapsed_s>100000):
            raise ValueError("invalid wall-clock metric")
    metric={}
    for key in sorted(_METRIC):
        metric[key]=_bounded_int(getattr(result.metrics,key))
    steps=_bounded_int(metric["steps"],0,100000)
    raw_stop=result.stop_reason
    stop=raw_stop if raw_stop in _STOP else "other"
    # The trace contains plaintext account credentials in successful
    # supervisor tool_result.content and often in later login arguments.
    # Read only identifying API *names* for in-process categorical counts.
    # Never keep ANY original TraceEvent, arguments, call payload, or content
    # in the output object.
    call_ids={}
    outcomes={name:{"attempted":0,"succeeded":0,"failed":0}
              for name in ("supervisor_profile","supervisor_passwords","app_login")}
    errors={name:0 for name in sorted(_ERROR)}
    unknown_errors=0
    for event in result.trace:
        if event.kind=="tool_call":
            data=event.data
            if data.get("name")!="appworld_call":
                continue
            args=data.get("arguments")
            if not isinstance(args,dict):
                continue
            app=args.get("app");api=args.get("api")
            key=(("supervisor_profile" if api=="show_profile"
                    else "supervisor_passwords" if api=="show_account_passwords"
                    else None) if app=="supervisor"
                 else "app_login" if api=="login" else None)
            if key is None:
                continue
            call_id=data.get("id")
            if not isinstance(call_id,str) or call_id in call_ids:
                raise ValueError("unsafe or duplicated tool call linkage")
            call_ids[call_id]=key
            outcomes[key]["attempted"]+=1
        elif event.kind=="tool_result":
            data=event.data
            call_id=data.get("id")
            if call_id in call_ids:
                key=call_ids.pop(call_id)
                if type(data.get("ok")) is not bool:
                    raise ValueError("no boolean result for tracked API call")
                outcomes[key]["succeeded" if data["ok"] else "failed"]+=1
            if data.get("ok") is False:
                code=data.get("error_code")
                if code in _ERROR: errors[code]+=1
                else:unknown_errors+=1
    if call_ids:
        raise ValueError("missing matching private tool results")
    if any(sum(d[k] for k in ("succeeded","failed"))!=d["attempted"]
           for d in outcomes.values()):
        raise ValueError("incomplete private action accounting")
    status={}
    for k,v in api_tool.http_status_counts.items():
        # Do not allow arbitrary dictionary keys copied from request/response
        # data or from exception messages into safe report.
        if type(k) is not str or not re.fullmatch(r"[1-5]\d{2}",k):
            raise ValueError("unsafe HTTP status key")
        status[k]=_bounded_int(v)
    api_calls=_bounded_int(api_tool.calls)
    http_errors=_bounded_int(api_tool.http_errors)
    if sum(status.values())!=http_errors:
        raise ValueError("HTTP diagnostic count inconsistency")
    report={
        "protocol":PROTOCOL,
        "arm":arm,"scene_index":scene,"repeat_index":repeat,
        "official_task_success":official_task_success,
        "nested_state_tests":nested,
        "elapsed_wall_seconds":round(elapsed_s,3) if elapsed_s is not None else None,
        "stop_reason":stop,
        "agent_metrics":metric,
        "actual_api_calls":api_calls,
        "http_error_calls":http_errors,
        "http_status_histogram":dict(sorted(status.items())),
        "safe_tool_error_histogram":{**errors,"OTHER_ERROR":unknown_errors},
        "auth_action_counts":outcomes,
        "agent_steps":steps,
        "raw_agent_trace_exported":False,
        "raw_model_recording_exported":False,
        "raw_task_or_credentials_or_access_tokens_exported":False,
        "no_plaintext_or_secret_hashes_in_artifact":True,
    }
    validate_safe_report(report)
    return report


def validate_safe_report(row:dict)->None:
    """Fail closed if any caller accidentally adds raw or arbitrary fields."""
    expected={
        "protocol","arm","scene_index","repeat_index","official_task_success",
        "nested_state_tests","elapsed_wall_seconds","stop_reason","agent_metrics",
        "actual_api_calls","http_error_calls","http_status_histogram",
        "safe_tool_error_histogram","auth_action_counts","agent_steps",
        "raw_agent_trace_exported","raw_model_recording_exported",
        "raw_task_or_credentials_or_access_tokens_exported",
        "no_plaintext_or_secret_hashes_in_artifact",
    }
    if type(row) is not dict or set(row)!=expected:
        raise ValueError("unexpected aggregate field; raw value export prohibited")
    if row["protocol"]!=PROTOCOL or row["arm"] not in _ARM:
        raise ValueError("unsafe aggregate identity")
    if row["stop_reason"] not in _STOP|{"other"}:
        raise ValueError("unsafe stop reason")
    for x in ("scene_index","repeat_index","agent_steps","actual_api_calls","http_error_calls"):
        _bounded_int(row[x])
    if row["official_task_success"] is not None:_boolean(row["official_task_success"])
    if row["nested_state_tests"] is not None:
        if type(row["nested_state_tests"]) is not list or len(row["nested_state_tests"])!=2:
            raise ValueError("bad nested test representation")
        if any(type(x) is not int for x in row["nested_state_tests"]):
            raise ValueError("bad nested test values")
    if row["elapsed_wall_seconds"] is not None:
        if type(row["elapsed_wall_seconds"]) not in (float,int) or not isfinite(row["elapsed_wall_seconds"]):
            raise ValueError("unsafe elapsed seconds")
    if set(row["agent_metrics"])!=_METRIC:
        raise ValueError("unexpected metrics")
    for v in row["agent_metrics"].values():_bounded_int(v)
    if set(row["safe_tool_error_histogram"])!=_ERROR|{"OTHER_ERROR"}:
        raise ValueError("unsafe error categories")
    for v in row["safe_tool_error_histogram"].values():_bounded_int(v)
    if set(row["auth_action_counts"])!={"supervisor_profile","supervisor_passwords","app_login"}:
        raise ValueError("unexpected auth categories")
    for v in row["auth_action_counts"].values():
        if set(v)!={"attempted","succeeded","failed"}:
            raise ValueError("unexpected auth count fields")
        for k in v:_bounded_int(v[k])
    for k,v in row["http_status_histogram"].items():
        if not re.fullmatch(r"[1-5]\d{2}",k):raise ValueError("unsafe status")
        _bounded_int(v)
    for key in ("raw_agent_trace_exported","raw_model_recording_exported",
                "raw_task_or_credentials_or_access_tokens_exported"):
        if row[key] is not False:
            raise ValueError("raw private material present")
    if row["no_plaintext_or_secret_hashes_in_artifact"] is not True:
        raise ValueError("unverified private material boundary")
    # This validation rejects arbitrary user/model-provided strings or
    # unexpected nested objects, even if the caller tried to add a field.
    json.dumps(row,allow_nan=False)


def write_safe_report(path:Path,record:dict)->None:
    validate_safe_report(record)
    if path.exists():
        raise FileExistsError("refuse to overwrite an existing evidence artifact")
    path.parent.mkdir(parents=True,exist_ok=True)
    # No trace or recorder is accepted by this function.
    path.write_text(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n")


def safe_exception_category(exc:BaseException)->str:
    """Categorize only built-in type, never emit an exception message/stack."""
    if isinstance(exc,TimeoutError):return "TIMEOUT"
    if isinstance(exc,ConnectionError):return "CONNECTION"
    if isinstance(exc,ValueError):return "INVALID_DATA"
    return "OTHER_EXCEPTION"
