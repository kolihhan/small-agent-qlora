"""CPU-only gold-source contract probe for AppWorld's public tool bridge.

All official TRAIN90 reference request transcripts stay inside ephemeral GitHub
runner; public artifacts include only aggregate route and state-test counts.
This is NOT a student/GAIA/LoRA evaluation and never uses DEV labels or adjusts
DEV task selection. Determine whether the real official reference API requests
could be represented by the publicly documented app/api/argument tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, allowed_public_api

PROTOCOL = "APPWORLD-TRAIN90-PUBLIC-API-BRIDGE-CONTRACT-V1"
SOURCE_COMMIT = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
PILOT_COUNT = 8



PILOT_SEED="state-verified-official-human-reference-8-tasks-train-only-20261117"


def choose_train_entries(entries):
    """Reproduce the eight earlier source-selected original TRAIN gold tasks."""
    selected=[]
    eligible=[]
    for tid,g in entries:
        apps=set()
        business=0
        for req in g.api_calls:
            url=req.get("url") or ""
            parts=url.lstrip("/").split("/")
            app=parts[0].lower() if len(parts)>1 else ""
            if app and app not in {"admin","api_docs","supervisor"}:
                business+=1
                apps.add(app)
        if business>=12 and len(apps)>=2:
            eligible.append((sha256(f"{PILOT_SEED}:{tid}".encode()).hexdigest(),tid))
    seen=set()
    for _,tid in sorted(eligible):
        scene=tid.split("_",1)[0]
        if scene in seen:continue
        seen.add(scene)
        selected.append(tid)
        if len(selected)==PILOT_COUNT:break
    if len(selected)!=PILOT_COUNT:
        raise ValueError("the original independently selected TRAIN8 has drifted")
    return selected

def source_hash(ids):
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("official TRAIN task ID corruption")
    return sha256("\n".join(sorted(ids)).encode()).hexdigest()


def route_pattern(path: str) -> re.Pattern[str]:
    """Make an anchored path pattern for templated AppWorld public URL paths."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("invalid public API template")
    parts = re.split(r"(\{[A-Za-z_]\w*\})", path)
    regex = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            regex.append("(?P<" + p[1:-1] + ">[^/?#]+)")
        else:
            regex.append(re.escape(p))
    return re.compile("^" + "".join(regex) + "$")


def request_category(req: dict) -> str:
    """Report only high-level action classes, never API names/parameters."""
    if not isinstance(req, dict) or not isinstance(req.get("url"), str):
        return "malformed"
    path=urlsplit(req["url"]).path.strip("/")
    app=path.split("/",1)[0].casefold()
    if app in {"admin","api_docs","supervisor"}:
        return app
    return "business" if app else "malformed"

def public_route_lookup(api_docs, allowed_apps) -> dict[str, list]:
    # Docs (not hidden oracle) specify actual path and HTTP method.
    out = {}
    for app in sorted(set(allowed_apps) | {"supervisor"}):
        if app in {"admin", "api_docs"}: continue
        docs = api_docs.get(app, {})
        for api, info in docs.items():
            if app == "supervisor" and api != "complete_task":continue
            if not isinstance(info, dict):continue
            method=str(info.get("method") or "").lower()
            path=str(info.get("path") or "")
            if method not in {"get","post","put","patch","delete"} or not path.startswith("/"):
                continue
            pat=route_pattern(path)
            out.setdefault(method, []).append((app, api, pat, path))
    return out


def match_public_request(req, routes) -> tuple[str, dict | None]:
    if not isinstance(req, dict) or not all(k in req for k in ("method","url","data")):
        return "invalid_reference", None
    method=str(req["method"]).lower()
    url=req["url"]
    if method not in routes or not isinstance(url,str) or not isinstance(req["data"],dict):
        return "unmapped", None
    path=urlsplit(url).path
    found=[]
    for app,api,pat,template in routes[method]:
        match=pat.fullmatch(path)
        # Original AppWorld public docs sometimes omit the app prefix.
        if match is None and not template.startswith("/"+app+"/"):
            match=pat.fullmatch(path.removeprefix("/"+app))
        if match:
            data=dict(req["data"])
            data.update(match.groupdict())
            found.append({"app":app,"api":api,"arguments":data})
    if not found:
        return "unmapped", None
    if len(found)>1:
        return "ambiguous", None
    return "mapped", found[0]


def summarize_reference_coverage(tasks, docs_by_task):
    if len(tasks)<60:raise ValueError("original TRAIN source too small for diagnostic")
    total=Counter()
    distinct=set()
    for task_id,original_calls in tasks:
        if task_id in distinct:raise ValueError("duplicate task ID")
        distinct.add(task_id)
        routes=docs_by_task[task_id]
        for call in original_calls:
            state,_=match_public_request(call,routes)
            total[state]+=1
            total[state+"_"+request_category(call)]+=1
        total["tasks"]+=1
        total["requests"]+=len(original_calls)
    return total


def private_route_replay(task_ids):
    from appworld import AppWorld
    from appworld.ground_truth import GroundTruth
    stats=Counter()
    for task_id in task_ids:
        try:
            with AppWorld(task_id=task_id,experiment_name="private-train-bridge-route-contract",
                          ground_truth_mode="full",timeout_seconds=240) as world:
                g=world.task.ground_truth
                if g is None or g.api_calls is None:
                    raise RuntimeError("gold reference source unavailable")
                routes=public_route_lookup(world.task.api_docs,world.task.allowed_apps)
                planned=[match_public_request(req,routes) for req in g.api_calls]
                stats["tasks_attempted"]+=1
                stats["reference_requests"]+=len(planned)
                stats["public_mappable"]+=sum(status=="mapped" for status,_ in planned)
                stats["unmapped"]+=sum(status=="unmapped" for status,_ in planned)
                stats["ambiguous"]+=sum(status=="ambiguous" for status,_ in planned)
                for req,(status,_) in zip(g.api_calls,planned):
                    if status!="mapped":
                        stats["unmapped_"+request_category(req)]+=1
                # Administrative bootstrap/doc-only calls can be omitted
                # from a student-visible tool policy, never business/supervisor.
                protected=[]
                for req,(status,action) in zip(g.api_calls,planned):
                    if status=="mapped":
                        protected.append(action)
                    elif status=="unmapped" and request_category(req) in ("admin","api_docs"):
                        stats["intentionally_excluded_admin_or_docs"]+=1
                    elif status=="unmapped" and request_category(req)=="supervisor":
                        # Supplied official gold transcript may contain hidden
                        # privileged supervisor pre/post calls. The student
                        # must NEVER have access to these endpoints. We test
                        # physical BUSINESS action replay with these omitted
                        # as a strictly limited source-only diagnostic.
                        stats["intentionally_excluded_nonpublic_supervisor"]+=1
                    else:
                        stats["unreplayable_business_or_bad_calls"]+=1
                local_blocked=sum(status!="mapped" and
                    not (status=="unmapped" and request_category(req) in ("admin","api_docs","supervisor"))
                    for req,(status,_) in zip(g.api_calls,planned))
                if local_blocked or not protected:
                    stats["tasks_unreplayable_due_to_schema"]+=1
                    continue
                tool=AppWorldCallTool(world,max_calls=1000,report_http_failures=True)
                for action in protected:
                    if not allowed_public_api(world,action["app"],action["api"]):
                        stats["api_allowlist_rejected"]+=1
                        raise ValueError("request was mapped to unapproved API")
                    result=tool.run(action,Path("."))
                    if result.ok:
                        stats["bridge_call_ok"]+=1
                    else:
                        stats["bridge_call_error"]+=1
                        stats["bridge_error_"+str(result.error_code)]+=1
                verdict=world.evaluate(suppress_errors=True)
                stats["tasks_public_bridge_final_correct"]+=int(verdict.success)
                stats["tasks_public_bridge_final_wrong"]+=int(not verdict.success)
                stats["state_tests_pass"]+=int(verdict.pass_count)
                stats["state_tests_total"]+=int(verdict.num_tests)
        except Exception as exc:
            # Never print task IDs, decrypted requests/responses or exception text.
            stats["error_cases"]+=1
            stats["error_"+type(exc).__name__]+=1
    return stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("cannot mutate original evidence")
    from appworld import AppWorld,load_task_ids
    from appworld.ground_truth import GroundTruth
    ids=load_task_ids("train")
    if len(ids)!=90:raise ValueError("pinned original TRAIN90 IDs changed")
    rows=[]
    route_maps={}
    # Only original TRAIN records accessed; all action/GT data remain private.
    for tid in ids:
        g=GroundTruth.load(task_id=tid,mode="full")
        if g.api_calls is None:raise ValueError("TRAIN gold API calls missing")
        rows.append((tid,g.api_calls))
        # Public API docs; no solution content.
        with AppWorld(task_id=tid,experiment_name="private-train-bridge-coverage",
                      ground_truth_mode="full",timeout_seconds=240) as world:
            route_maps[tid]=public_route_lookup(world.task.api_docs,world.task.allowed_apps)
    totals=summarize_reference_coverage(rows,route_maps)
    # Reuse the same prospectively predeclared 8 source-only TRAIN tasks from
    # preceding gold replay, not outcome-selected dev tasks.
    original_rows=[(tid,GroundTruth.load(task_id=tid,mode="full")) for tid in ids]
    subset=choose_train_entries(original_rows)
    replay=private_route_replay(subset)
    out={"status":"COMPLETE_TRAIN_ONLY_API_TOOL_SOURCE_CONTRACT",
         "protocol":PROTOCOL,"original_upstream_commit":SOURCE_COMMIT,
         "train_id_hash":source_hash(ids),
         "train_task_count":len(ids),
         "train_request_coverage":dict(sorted(totals.items())),
         "train8_route_replay":dict(sorted(replay.items())),
         "run_contains_decrypted_source_or_references":False,
         "training":False,"models_called":0,"DEV_or_TEST_opened":False,
         "no_GAIA_score":True,
         "interpretation":"Original reference TRAIN90 business API coverage and restricted public bridge replay of earlier eight TRAIN-only traces, explicitly omitting UNAVAILABLE privileged supervisor calls. Successful state is a business-trace feasibility result only, NOT proof model could reconstruct hidden supervisor info or correct next-actions. No LLM task success, DEV/GAIA model gain, or supervised selector."}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("APPWORLD_TRAIN90_BRIDGE_COVERAGE",json.dumps(totals,sort_keys=True),flush=True)
    print("APPWORLD_TRAIN8_BRIDGE_REPLAY",json.dumps(replay,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
