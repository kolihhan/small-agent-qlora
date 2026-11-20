"""Read-only public AppWorld auth contract observability study, zero Qwen calls.

Returns COUNT-ONLY results. Neither task prompts nor API/app names, parameter
values, credentials, responses or protected evaluator content leave the runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool
from scripts.appworld_train_first_feedback_fork import (
    task_roster, digest, UPSTREAM_APPWORLD_REV, MODEL_GGUF_SHA,
)

PROTOCOL = "APPWORLD-TRAIN8-PUBLIC-AUTH-CONTRACT-VISIBLE-TO-QWEN-V1"


def source_only_auth_census(world):
    c=Counter()
    docs=AppWorldDocsTool(world)
    allowed=sorted(set(world.task.allowed_apps)-{"admin","api_docs"})
    for app in allowed:
        docs_for_app=world.task.api_docs.get(app,{})
        if not isinstance(docs_for_app,dict):
            c["invalid_app_docs"]+=1
            continue
        has_login="login" in docs_for_app
        c["public_apps"]+=1
        if has_login:c["public_apps_with_login"]+=1
        for api,payload in docs_for_app.items():
            if not isinstance(payload,dict):continue
            c["public_api_docs"]+=1
            params=payload.get("parameters",[])
            if not isinstance(params,list):continue
            tokens=[p for p in params if isinstance(p,dict)
                    and p.get("name")=="access_token"
                    and isinstance(p.get("required"),bool)]
            if not tokens:continue
            if len(tokens)!=1:
                c["ambiguous_token_parameters"]+=1
                continue
            required=bool(tokens[0]["required"])
            prefix="required" if required else "optional"
            c[prefix+"_token_api_count"]+=1
            if has_login:c[prefix+"_token_apis_with_public_login"]+=1
            result=docs.run({"app":app,"api":api},Path("."))
            if not result.ok:
                c[prefix+"_token_doc_unavailable"]+=1
                continue
            full=result.observation()
            exposed,_=result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
            exact_token_field='"name": "access_token"'
            if exact_token_field in full:
                c[prefix+"_token_in_tool_full_doc"]+=1
            if exact_token_field in exposed:
                c[prefix+"_token_visible_to_model"]+=1
            else:
                c[prefix+"_token_missing_in_model_view"]+=1
            if len(full)>MAX_TOOL_OBSERVATION_CHARS:
                c[prefix+"_token_doc_truncated"]+=1
            if required and not has_login:
                c["required_token_no_login_in_same_app"]+=1
    return dict(sorted(c.items()))


def run():
    from appworld import AppWorld
    roster=task_roster()
    assert len(roster)==8
    scenes=[]
    for index,tid in enumerate(roster):
        with AppWorld(task_id=tid,experiment_name=f"auth-contract-census-{index}",
                      ground_truth_mode="full",timeout_seconds=180) as world:
            data=source_only_auth_census(world)
        scenes.append({"scene_index":index,"public_counts_only":data})
    profiles=[r["public_counts_only"] for r in scenes]
    result={
        "protocol":PROTOCOL,"source_scenes":8,
        "appworld_revision":UPSTREAM_APPWORLD_REV,
        "model_sha_not_loaded":MODEL_GGUF_SHA,
        "roster_digest":digest(roster),
        "unique_count_profiles":len({digest(x) for x in profiles}),
        "one_scene_representative":profiles[0],
        "scenes":scenes,"source_only":True,
        "model_inference_calls":0,"model_observations_not_modified":True,
        "no_gold_or_private_response_exported":True,
        "safe_output_only_boolean_counts":True,
        "task_success_not_measured":True,
        "HTTP401_cause_not_proven":True,
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable result already exists")
    try:out=run()
    except Exception as exc:
        print("AUTH_CONTRACT_CENSUS_FAIL_CLOSED",type(exc).__name__,flush=True)
        raise SystemExit(1) from None
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("APPWORLD_AUTH_CONTRACT_SOURCE_COUNTS",
          json.dumps({"scenes":out["source_scenes"],
                      "distinct_profiles":out["unique_count_profiles"],
                      "one_scene":out["one_scene_representative"],
                      "qwen_calls":0},sort_keys=True),flush=True)


if __name__=="__main__":main()
