"""Prepare source-verified (NOT benchmark-verified) NextSearch QLoRA pilot turns.

No simulated actions: only original SFT assistant decisions and real published
search/fetch observations, plus the exact publisher gold final. The original
reasoning_content is intentionally excluded to avoid hidden-CoT imitation.

Tool observation bodies are physically clipped to native 2000-char delivery
budget; a parent is rejected if no final-answer evidence reaches the model.
No training or GPU executed by this preparer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools.base import ToolResult
from scripts.audit_nextsearch_apache_source import (
    REPO, CONFIG, FROZEN_REVISION, _call, inspect
)

PROTOCOL="nextsearch-source-verified-causal-qlora-pilot-parents-v1"
MIN_PARENTS=20


def export(row:dict)->tuple[dict|None,str]:
    scored=inspect(row)
    if not scored["long_search_fetch_candidate"]:
        return None,"not_original_executed_long_gold_exact"
    if scored["tool_error_count"]:
        return None,"original_tool_errors"
    source_user=[m["content"] for m in row["messages"]
                 if isinstance(m,dict) and m.get("role")=="user"
                 and isinstance(m.get("content"),str)]
    if not source_user:
        return None,"missing_source_question"
    source_gold=row.get("gold") or {}
    answer=source_gold.get("answer")
    if not isinstance(answer,(str,int,float)):
        return None,"missing_or_complex_reference"
    reference=str(answer).strip()
    if not 1<=len(reference)<=120:
        return None,"outlier_gold_length"
    question=source_user[-1]
    if reference.casefold() in question.casefold():
        return None,"reference_already_in_question"
    original_tools=row.get("tools")
    if not isinstance(original_tools,list) or not original_tools:
        return None,"missing_published_tool_schemas"
    tooldefs=[]
    for tool in original_tools:
        if not isinstance(tool,dict) or tool.get("name") not in ("search","fetch"):
            return None,"non_native_search_fetch_tool"
        if not isinstance(tool.get("parameters"),dict):
            return None,"incomplete_tool_definitions"
        tooldefs.append({"type":"function","function":{
            "name":tool["name"],
            "description":str(tool.get("description") or ""),
            "parameters":tool["parameters"]
        }})
    messages=[]
    for m in row["messages"]:
        if not isinstance(m,dict) or m.get("role") not in ("system","user"):
            return None,"unsupported_source_prefix"
        messages.append({"role":m["role"],"content":str(m.get("content") or "")})
    names={}
    observation_text=[]
    actions=[]
    for m in row["target"]:
        if m.get("role")=="assistant":
            msg={"role":"assistant","content":str(m.get("content") or "")}
            originals=m.get("tool_calls") or []
            if originals:
                calls=[]
                for call in originals:
                    name,ident,valid=_call(call)
                    if not valid or name not in ("search","fetch") or ident in names:
                        return None,"invalid_source_function_call"
                    args=(call.get("function") or {}).get("arguments")
                    if isinstance(args,str):
                        args=json.loads(args)
                    names[ident]=name
                    calls.append({
                        "id":ident,"type":"function",
                        "function":{"name":name,"arguments":args}
                    })
                msg["tool_calls"]=calls
                actions.append((len(messages),[x["function"]["name"] for x in calls]))
            messages.append(msg)
        elif m.get("role")=="tool":
            cid=str(m.get("tool_call_id") or "")
            if cid not in names:
                return None,"noncausal_tool_message"
            raw=m.get("content")
            if not isinstance(raw,str) or not raw.strip():
                return None,"empty_source_tool_feedback"
            observation,_=ToolResult(True,raw).bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
            observation_text.append(observation)
            messages.append({
                "role":"tool","tool_call_id":cid,
                "name":names[cid],"content":observation
            })
        else:
            return None,"unexpected_target_message_role"
    if not messages or messages[-1]["role"]!="assistant" or messages[-1].get("tool_calls"):
        return None,"no_original_final_decision"
    if len(observation_text)!=scored["tool_actions"]:
        return None,"bounded_observation_pairing_changed"
    combined="\n".join(observation_text)
    # Supports an actual value visible through bounded original web-tool
    # feedback; a gold-only final label is not enough for SFT evidence.
    if reference.casefold() not in combined.casefold():
        return None,"final_reference_not_in_bounded_original_observations"
    source_id=str(row["id"])
    if "gaia" in source_id.casefold():
        return None,"protected_source"
    # Only a deterministic, balanced subset of ACTUAL assistant actions.
    # This does not insert invented reads or a fabricated stop decision.
    search_indices=[idx for idx,names_ in actions if "search" in names_]
    fetch_indices=[idx for idx,names_ in actions if "fetch" in names_]
    if not search_indices or not fetch_indices:
        return None,"missing_search_or_fetch_decisions"
    selected=sorted(set((
        search_indices[0],search_indices[-1],
        fetch_indices[0],fetch_indices[-1],
        len(messages)-1
    )))
    return {
        "task_id":source_id,
        "source":REPO,
        "capability":"evidence_conditioned_multistep_web_research",
        "source_revision":FROZEN_REVISION,
        "source_split":"train",
        "verified":True, # SOURCE reference/observation contract only
        "verification":{
            "completed":True,
            "final_answer_correct":True,  # vs publisher's gold only
            "zero_tool_errors":True,
            "zero_duplicate_blocks":True,
            "required_tools_satisfied":True,
            "provenance_complete":True,
        },
        "provenance":{
            "license":"Apache-2.0",
            "generator":REPO,
            "generator_version":FROZEN_REVISION,
            "oracle_type":"original-publisher-reference-and-bounded-tool-observation",
            "oracle_version":PROTOCOL,
        },
        "source_success_independently_replayed":False,
        "native_gaia_policy_promoted":False,
        "required_tools":["search","fetch"],
        "expected_answer":reference,
        "tools":tooldefs,
        "messages":messages,
        "training_action_indices":selected,
        "metrics":{
            "steps":scored["assistant_action_turns"],
            "tool_calls":scored["tool_actions"],
            "tool_successes":scored["matched_nonempty_observations"],
            "tool_errors":0,
            "duplicate_calls_blocked":0,
        },
        "pilot_only":True,
        "source_tool_surface":"nextsearch-batched-search-fetch",
    },"source_verified_candidate"


def prepare(rows:list[dict],output:Path)->dict:
    if not rows:
        raise ValueError("empty original training split")
    counts=Counter()
    unique_tasks=set()
    source_counts=Counter()
    accepted=[]
    for row in rows:
        counts["seen"]+=1
        parent,reason=export(row)
        counts[reason]+=1
        if parent is None:
            continue
        if parent["task_id"] in unique_tasks:
            counts["duplicate_qualified_source_task"]+=1
            continue
        unique_tasks.add(parent["task_id"])
        source_counts[str(row.get("source") or "")]+=1
        accepted.append(parent)
    with output.open("x",encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {
        "protocol":PROTOCOL,"dataset":REPO,"revision":FROZEN_REVISION,
        "train_count":len(rows),"task_disjoint_unique_parents":len(accepted),
        "source_counts":dict(source_counts),
        "rejections":dict(counts),
        "real_tool_actions":sum(r["metrics"]["tool_calls"] for r in accepted),
        "selected_native_assistant_turns":sum(len(r["training_action_indices"]) for r in accepted),
        "bounded_observation_chars":MAX_TOOL_OBSERVATION_CHARS,
        "source_parent_jsonl_sha256":sha256(output.read_bytes()).hexdigest(),
        "meets_initial_20_parent_source_screen":len(accepted)>=MIN_PARENTS,
        "source_answer_verified_against_publisher":True,
        "task_success_independently_replayed":False,
        "training_run_completed":False,
        "gaia_validated":False,
        "production_promoted":False,
        "caveat":"SFT source-level eligibility is not GAIA evidence; tool aliases are experimental and model transfer to native search/read is unverified.",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--report",required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refuse old file overwrite")
    from datasets import load_dataset
    rows=list(islice(load_dataset(REPO,CONFIG,split="train",
                                  revision=FROZEN_REVISION,streaming=True),3270))
    if len(rows)!=3270:
        raise ValueError("original source count drift")
    summary=prepare(rows,args.output)
    args.report.write_text(json.dumps(summary,indent=2)+"\n")
    print("NEXTSEARCH_SOURCE_BOUNDED_SFT_PARENTS",
          json.dumps(summary,sort_keys=True),flush=True)
    import os
    if os.environ.get("HF_STREAM_SAFE_EXIT")=="1":
        import sys
        for path in (args.output,args.report):
            with path.open("rb") as f:
                os.fsync(f.fileno())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__=="__main__":
    main()
