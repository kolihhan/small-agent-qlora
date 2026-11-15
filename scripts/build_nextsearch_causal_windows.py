"""Causal state windows from original NextSearch source-verified train episodes.

Every supervised assistant target is an UNCHANGED original assistant action.
Each context uses only original source system/user plus the most recent 2-4
complete actual tool-call/observation groups. No synthetic tool action,
fabricated observation, gold-provided tool decision or hidden reasoning.

Window A: source reference unavailable in prior tool observations -> actual
continued tool call. Window B: source reference literally visible -> actual
original final answer (after a genuine original episode). The windows are
partial historical contexts, not reconstructed full autonomous episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

PROTOCOL="nextsearch-real-source-causal-state-windows-v1"
MAX_SOURCE_GROUPS=4


def original_groups(row:dict):
    messages=row["messages"]
    first_assistant=next((i for i,m in enumerate(messages)
                          if m.get("role")=="assistant"),None)
    if first_assistant is None or first_assistant<2:
        raise ValueError("source original prompt/action boundaries missing")
    prefix=messages[:first_assistant]
    if any(m.get("role") not in ("system","user") for m in prefix):
        raise ValueError("original source prefix not system/user")
    result=[]
    current=None
    for msg in messages[first_assistant:-1]:
        if msg.get("role")=="assistant":
            if not msg.get("tool_calls"):
                raise ValueError("nonterminal original assistant final")
            if current is not None:
                result.append(current)
            current=[msg]
        elif msg.get("role")=="tool":
            if current is None:
                raise ValueError("tool output without preceding original action")
            current.append(msg)
        else:
            raise ValueError("unrecognized original history role")
    if current is not None:
        result.append(current)
    if messages[-1].get("role")!="assistant" or messages[-1].get("tool_calls"):
        raise ValueError("original final missing")
    for group in result:
        names={c["id"] for c in group[0]["tool_calls"]}
        feedback={m.get("tool_call_id") for m in group[1:]}
        if not names or feedback!=names or len(group[1:])!=len(names):
            raise ValueError("source action/observation pairing drift")
    return prefix,result,messages[-1]


def contains_gold(groups:list[list[dict]],answer:str)->bool:
    text="\n".join(
        str(m["content"]) for group in groups
        for m in group if m.get("role")=="tool"
    )
    return answer.casefold() in text.casefold()


def make_fragment(parent:dict,source_prefix:list[dict],
                  context:list[list[dict]],target:dict,
                  state:str)->dict:
    if state not in ("evidence_missing_continue","evidence_present_finalize"):
        raise ValueError(state)
    context_messages=[dict(m) for group in context for m in group]
    dialogue=[dict(m) for m in source_prefix]+context_messages+[dict(target)]
    final_idx=len(dialogue)-1
    tools_seen=sorted({
        c["function"]["name"] for m in dialogue if m.get("role")=="assistant"
        for c in m.get("tool_calls") or []
    })
    if target.get("role")!="assistant":
        raise ValueError("only an actual original assistant action can be supervised")
    fragment=dict(parent)
    fragment.update({
        "task_id":parent["task_id"]+":"+state,
        "capability":state,
        "messages":dialogue,
        "training_action_indices":[final_idx],
        "required_tools":tools_seen,
        "source_parent_task_id":parent["task_id"],
        "source_parent_sha256":sha256(parent["task_id"].encode()).hexdigest(),
        "source_window_groups":len(context),
        "source_context_is_partial":True,
        "original_teacher_action_unchanged":True,
        "source_answer_visible_in_window":contains_gold(context,parent["expected_answer"]),
        "source_window_evidence_policy":PROTOCOL,
        # The source episode is source-gold correct, but a partial context
        # is not an independent validation of the next-action optimality.
        "task_success_independently_replayed":False,
        "training_ready_for_production":False,
    })
    return fragment


def state_windows(parent:dict)->tuple[list[dict],Counter]:
    prefix,groups,final=original_groups(parent)
    answer=str(parent["expected_answer"]).strip()
    stats=Counter()
    out=[]
    # One source genuine next tool decision after at least two already
    # received observations, before source answer is visible in any of them.
    for index in range(2,len(groups)):
        previous=groups[max(0,index-2):index]
        if not contains_gold(groups[:index],answer):
            teacher=groups[index][0]
            x=make_fragment(parent,prefix,previous,teacher,
                            "evidence_missing_continue")
            if x["source_answer_visible_in_window"]:
                raise AssertionError("missing-evidence action got answer")
            out.append(x)
            stats["missing_state"]+=1
            break
    else:
        stats["no_eligible_missing_state"]+=1

    # Reject teacher examples where the source answer was already
    # visible far earlier and unnecessary source browsing continued.
    first_reveal=next((i for i,g in enumerate(groups)
                       if contains_gold([g],answer)),None)
    if first_reveal is not None and first_reveal < len(groups)-MAX_SOURCE_GROUPS:
        stats["oversearched_after_evidence"]+=1
        return out,stats

    # Source answer must be present within a contiguous suffix of the
    # *actual* original web observations, before original recorded final.
    for count in range(2,min(MAX_SOURCE_GROUPS,len(groups))+1):
        context=groups[-count:]
        if contains_gold(context,answer):
            x=make_fragment(parent,prefix,context,final,
                            "evidence_present_finalize")
            if not x["source_answer_visible_in_window"]:
                raise AssertionError("final target has no visible source answer")
            out.append(x)
            stats["complete_state"]+=1
            break
    else:
        stats["no_eligible_complete_state"]+=1
    return out,stats


def build(parents:list[dict],out:Path)->dict:
    counts=Counter()
    tasks=set()
    exported=[]
    for parent in parents:
        base=str(parent.get("task_id") or "")
        if not base or base in tasks:
            raise ValueError("duplicate/missing original parent task")
        tasks.add(base)
        if parent.get("source_split")!="train" or parent.get("pilot_only") is not True:
            raise ValueError("unexpected source or promotion flag")
        windows,stats=state_windows(parent)
        exported.extend(windows)
        counts.update(stats)
    ids=[r["task_id"] for r in exported]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate window IDs")
    with out.open("x",encoding="utf-8") as f:
        for row in exported:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {
        "protocol":PROTOCOL,
        "source_parent_count":len(parents),
        "actual_partial_source_state_training_rows":len(exported),
        "source_missing_state_rows":counts["missing_state"],
        "source_complete_state_rows":counts["complete_state"],
        "parents_with_both_states":len({
            r["source_parent_sha256"] for r in exported if r["capability"]=="evidence_missing_continue"
        } & {
            r["source_parent_sha256"] for r in exported if r["capability"]=="evidence_present_finalize"
        }),
        "rejected_state_counts":dict(counts),
        "no_invented_tool_calls":True,
        "source_val_test_data_excluded":True,
        "actual_model_training_completed":False,
        "pilot_only":True,
        "sha256":sha256(out.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() or a.report.exists():
        raise FileExistsError("refuse overwrite")
    parents=[json.loads(line) for line in a.source.read_text().splitlines() if line.strip()]
    report=build(parents,a.output)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print("NEXTSEARCH_CAUSAL_STATE_WINDOWS",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
