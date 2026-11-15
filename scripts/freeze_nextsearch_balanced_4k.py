"""Freeze balanced, FULL-CONTEXT 4096-token Qwen9B SFT exposure from real source states.

This does not train a model. Selection depends only on source provenance,
state label, and actual token counts under the pinned Qwen3.5-9B template.
It cannot see gold answers, source task benchmark scores, or VAL outputs.

A row is included only if both original native prompt and unmodified
assistant tool/final completion fit the target length with zero truncation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.qlora import (
    _render_turn_examples, _tokenize_completion_examples, load_verified_rows
)

PROTOCOL="nextsearch-balanced-true-source-4096-qlora-v0"
MAX_LENGTH=4096
PER_STATE=24
MIN_PAIRED_PARENTS=0
MAX_PARENT_EXPOSURES=2
MODEL="Qwen/Qwen3.5-9B"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STATES=("evidence_missing_continue","evidence_present_finalize")


def order_key(name:str)->str:
    return sha256((PROTOCOL+":"+name).encode()).hexdigest()


def freeze(source_rows:list[dict],tokenizer) -> tuple[list[dict],dict]:
    if not source_rows or len({r["task_id"] for r in source_rows})!=len(source_rows):
        raise ValueError("empty or duplicate original window IDs")
    candidates=defaultdict(dict)
    counts=Counter()
    for row in source_rows:
        state=str(row.get("capability") or "")
        if state not in STATES:
            raise ValueError("unexpected teacher state")
        if row.get("source_split")!="train" or row.get("source_context_is_partial") is not True:
            raise ValueError("invalid source split or fabricated full replay")
        if row.get("training_ready_for_production") is not False:
            raise ValueError("source window incorrectly tagged for deployment")
        if row.get("original_teacher_action_unchanged") is not True:
            raise ValueError("unproven original next action")
        if bool(row.get("source_answer_visible_in_window"))!=(state==STATES[1]):
            raise ValueError("label contradicts visible-source indicator")
        if row.get("task_success_independently_replayed") is not False:
            raise ValueError("source outcome is not independently verified")
        example=_render_turn_examples(tokenizer,[row])
        if len(example)!=1:
            raise ValueError("original source window produced !=1 true next action")
        e=example[0]
        if e["metadata"]["action_type"]!=(
            "tool_call" if state==STATES[0] else "final"
        ):
            raise ValueError("source action disagrees with current evidence state")
        inp=tokenizer(e["prompt"],add_special_tokens=False)["input_ids"]
        out=tokenizer(e["completion"],add_special_tokens=False)["input_ids"]
        n=len(inp)+len(out)
        counts["source_"+state]+=1
        if not out or n>MAX_LENGTH:
            counts["too_long_"+state]+=1
            continue
        # Use same train tokenization path to guarantee real no-prompt-truncate.
        tokenized=_tokenize_completion_examples(
            tokenizer,[e],max_length=MAX_LENGTH,require_full_prompt=True
        )
        if len(tokenized)!=1 or not any(y!=-100 for y in tokenized[0]["labels"]):
            raise ValueError("complete-prompt tokenization failed")
        parent=str(row.get("source_parent_sha256") or "")
        if not parent:
            raise ValueError("source parent task missing")
        if state in candidates[parent]:
            raise ValueError("duplicate source state for one original task")
        candidates[parent][state]=(row,n)
        counts["full_length_"+state]+=1
    both=sorted(
        (parent for parent,states in candidates.items() if len(states)==2),
        key=order_key
    )
    picked=[]
    state_counts=Counter()
    touched=Counter()
    # Favor complete BEFORE/AFTER evidence observations on the same genuine
    # source task, but cap topic exposures and keep state balance.
    for parent in both:
        if any(state_counts[state]>=PER_STATE for state in STATES):
            break
        for state in STATES:
            row,_=candidates[parent][state]
            picked.append(row)
            touched[parent]+=1
            state_counts[state]+=1
    for state in STATES:
        other=sorted(
            (parent for parent,states in candidates.items()
             if state in states and not touched[parent]),
            key=order_key
        )
        for parent in other:
            if state_counts[state]>=PER_STATE:
                break
            row,_=candidates[parent][state]
            picked.append(row)
            touched[parent]+=1
            state_counts[state]+=1
    if any(state_counts[state]!=PER_STATE for state in STATES):
        raise ValueError(
            f"not enough true original 4096-token state examples: "
            f"{dict(state_counts)} vs {PER_STATE} each"
        )
    if not all(v<=MAX_PARENT_EXPOSURES for v in touched.values()):
        raise ValueError("one source task overrepresented")
    ids=[r["task_id"] for r in picked]
    if len(ids)!=len(set(ids)):
        raise ValueError("selected repeat teacher action")
    selected_samples=_render_turn_examples(tokenizer,picked)
    actual_tokenized=_tokenize_completion_examples(
        tokenizer,selected_samples,max_length=MAX_LENGTH,require_full_prompt=True
    )
    if len(actual_tokenized)!=2*PER_STATE:
        raise ValueError("training exposure changed under real SFT tokenization")
    summary={
        "protocol":PROTOCOL,
        "model":MODEL,
        "tokenizer_revision":REVISION,
        "max_length":MAX_LENGTH,
        "selected_actual_source_next_actions":len(picked),
        "states":dict(state_counts),
        "original_source_distinct_parent_tasks":len(touched),
        "paired_source_parent_tasks":sum(v==2 for v in touched.values()),
        "source_window_rows_seen":len(source_rows),
        "state_eligible_full_context":{
            state:counts["full_length_"+state] for state in STATES
        },
        "rejections":{
            "too_long_"+state:counts["too_long_"+state] for state in STATES
        },
        "example_action_types":dict(Counter(
            x["metadata"]["action_type"] for x in selected_samples
        )),
        "all_retain_full_prompt":True,
        "source_gold_available_to_selector":False,
        "native_gaia_scorer_used":False,
        "training_steps":0,
        "deployment_allowed":False,
    }
    return picked,summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("do not overwrite previous source exposure")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=REVISION,trust_remote_code=True
    )
    source=load_verified_rows(args.source)
    selected,report=freeze(source,tokenizer)
    args.output.write_text("".join(
        json.dumps(row,ensure_ascii=False)+"\n" for row in selected
    ),encoding="utf-8")
    report["output_sha256"]=sha256(args.output.read_bytes()).hexdigest()
    report["source_sha256"]=sha256(args.source.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NEXTSEARCH_FULL_CONTEXT_BALANCED_QLORA",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
