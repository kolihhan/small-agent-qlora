"""Export only WHOLE, source-attributed real tool trajectories fitting 2048 tokens.

No clipping, truncation, padding of evidence, rewording of model completions,
duplicating rare 2Wiki tasks, or training. Keeps original provenance and native
tool-message structure, including final answers, as model-facing QLoRA inputs.
Use only after independent GAIA protection and strict multi-source curation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.qlora import (
    load_verified_rows,
    _render_turn_examples,
    _tokenize_completion_examples,
)

_SOURCES={"hotpotqa-distractor-train","2wikimultihopqa-train"}


def select_whole_task_rows(*,rows:list[dict],examples:list[dict],tokenizer,
                           max_length:int=2048,min_tasks:int=30,
                           min_second_source:int=3) -> tuple[list[dict],dict]:
    if max_length<1 or min_tasks<1 or min_second_source<1:
        raise ValueError("invalid predeclared full-task resource gates")
    if len({r.get("task_id") for r in rows}) != len(rows):
        raise ValueError("duplicate task identities")
    if any(r.get("verified") is not True or r.get("source") not in _SOURCES
           or not r.get("training_action_indices") for r in rows):
        raise ValueError("only strict, source-attributed verified trajectories accepted")
    by_task=defaultdict(list)
    for ex in examples:
        metadata=ex.get("metadata") or {}
        tid=metadata.get("task_id")
        if tid is None:
            raise ValueError("unattributed action")
        by_task[str(tid)].append(ex)
    if set(by_task)!={str(r["task_id"]) for r in rows}:
        raise ValueError("incomplete or foreign rendered actions")

    selected=[]
    skip=Counter()
    per_source=Counter()
    max_action_tokens=0
    for row in rows:
        tid=str(row["task_id"])
        acts=by_task[tid]
        if len(acts)!=len(row["training_action_indices"]):
            raise ValueError("an assistant action was silently dropped")
        if sum(str(a["metadata"].get("action_type"))=="final" for a in acts)!=1:
            raise ValueError("missing or duplicated final answer model completion")
        if sum(str(a["metadata"].get("action_type"))=="tool_call" for a in acts)<2:
            raise ValueError("incomplete tool-use supervision")
        token_counts=[]
        for action in acts:
            p=tokenizer(action["prompt"],add_special_tokens=False)["input_ids"]
            c=tokenizer(action["completion"],add_special_tokens=False)["input_ids"]
            if not c:
                raise ValueError("empty model completion")
            token_counts.append(len(p)+len(c))
        if max(token_counts)>max_length:
            skip["whole_task_exceeds_context"]+=1
            continue
        # Same exact next-action tokenization as trainer, hard-fail on truncation.
        encoded=_tokenize_completion_examples(
            tokenizer,acts,max_length=max_length,require_full_prompt=True)
        assert len(encoded)==len(acts)
        if any(len(x["input_ids"])>max_length for x in encoded):
            raise AssertionError("trainer budget violated after independent preflight")
        max_action_tokens=max(max_action_tokens,*token_counts)
        selected.append(row)
        per_source[str(row["source"])]+=1
    if len(selected)<min_tasks or per_source["2wikimultihopqa-train"]<min_second_source:
        raise ValueError(f"insufficient untruncated cross-source train cohort: "
                         f"{len(selected)} tasks / {dict(per_source)}")
    return selected,{
        "schema":"real-multisource-full-task-2048-v1",
        "model":"Qwen/Qwen3.5-9B",
        "max_length":max_length,
        "original_unique_tasks":len(rows),
        "retained_unique_tasks":len(selected),
        "retained_native_actions":sum(len(x["training_action_indices"]) for x in selected),
        "retained_final_answers":len(selected),
        "source_counts":dict(sorted(per_source.items())),
        "rejected_tasks":dict(sorted(skip.items())),
        "maximum_kept_action_tokens":max_action_tokens,
        "require_full_prompt":True,
        "new_model_training_ran":False,
        "gaia_evaluation_ran":False,
        "warning":"A complete-context dataset is not a proven 2048-token GPU fit, good SFT objective, or GAIA generalization. Explicitly assess available GPU memory and run a separate matched evaluation.",
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--protected",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3.5-9B")
    p.add_argument("--min-tasks",type=int,default=30)
    p.add_argument("--min-second-source",type=int,default=3)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists() or a.output.resolve()==a.report.resolve():
        raise FileExistsError("never overwrite source/data provenance or prior artifacts")
    from transformers import AutoTokenizer
    rows=load_verified_rows(a.input,protected_questions_path=a.protected)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    acts=_render_turn_examples(tok,rows)
    selected,report=select_whole_task_rows(
        rows=rows,examples=acts,tokenizer=tok,max_length=2048,
        min_tasks=a.min_tasks,min_second_source=a.min_second_source,
    )
    if a.model!="Qwen/Qwen3.5-9B":
        raise ValueError("frozen training tokenizer identity changed")
    report["curated_source_sha256"]=sha256(a.input.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in selected),
        encoding="utf-8",
    )
    report["selected_dataset_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("REAL_2048_WHOLE_TASK_DATASET",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
