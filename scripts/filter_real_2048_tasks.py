"""Select only *complete* evidence-grounded real 2048-token tasks for QLoRA.

Operates on already curated HotpotQA / 2Wiki native tool trajectories.
Every assistant supervised turn (including the final answer) MUST fit the
token budget with its entire previous conversation. Reject whole tasks if ANY
supervised turn is too long; no slicing, repeated tasks, dropped tools, or
guessing model answers. CPU-only; no model training or GAIA evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def filter_full_tasks(rows:list[dict], examples:list[dict],tokenizer, *,
                      budget:int=2048)->tuple[list[dict],dict]:
    if budget<128:
        raise ValueError("must use an explicit legitimate full-context budget")
    if not rows or not examples:
        raise ValueError("empty curated rows or unrendered assistant turns")
    mapping={}
    for r in rows:
        tid=str(r.get("task_id") or "")
        if not tid or tid in mapping:
            raise ValueError("missing/duplicate independent task identity")
        if r.get("source") not in {"hotpotqa-distractor-train","2wikimultihopqa-train"}:
            raise ValueError("source not permitted for real QLoRA")
        if not r.get("verified"):
            raise ValueError("training row not verified")
        mapping[tid]=r
    seen=defaultdict(list)
    totals=Counter()
    for i,e in enumerate(examples):
        meta=e.get("metadata") or {}
        tid=str(meta.get("task_id") or "")
        if tid not in mapping:
            raise ValueError(f"rendered action {i} is missing source task")
        if meta.get("action_type") not in ("tool_call","final"):
            raise ValueError("unknown native assistant action")
        prompt_ids=tokenizer(str(e["prompt"]),add_special_tokens=False)["input_ids"]
        completion_ids=tokenizer(str(e["completion"]),add_special_tokens=False)["input_ids"]
        if not completion_ids:
            raise ValueError("cannot train an empty native completion")
        size=len(prompt_ids)+len(completion_ids)
        seen[tid].append((size,str(meta["action_type"])))
        totals["actions"]+=1
        if size<=budget:totals["actions_fit_individually"]+=1
    if set(seen)!=set(mapping):
        raise ValueError("rendered all tasks completely before filtering")
    accepted=[]
    rejected=[]
    source=Counter()
    for tid,row in mapping.items():
        actions=seen[tid]
        expected=row.get("training_action_indices")
        if not isinstance(expected,list) or len(actions)!=len(expected):
            raise ValueError("rendered native actions differ from curated indices")
        finals=sum(action=="final" for _,action in actions)
        if finals!=1:
            raise ValueError("complete task must have exactly one final answer")
        if all(size<=budget for size,_ in actions):
            accepted.append(row)
            source[row["source"]]+=1
        else:
            rejected.append({"task_sha256":sha256(tid.encode()).hexdigest(),
                             "action_count":len(actions),
                             "over_budget_action_count":sum(size>budget for size,_ in actions),
                             "max_context_tokens":max(size for size,_ in actions)})
    actions_selected=sum(len(r["training_action_indices"]) for r in accepted)
    if not accepted or len(accepted)+len(rejected)!=len(rows):
        raise ValueError("all real tasks rejected or internal completeness error")
    if source["hotpotqa-distractor-train"]<1 or source["2wikimultihopqa-train"]<1:
        raise ValueError("real multi-source task evidence lost by filtering")
    report={
        "schema":"real-multi-source-complete-task-filter-v1",
        "budget":budget,
        "input_unique_tasks":len(rows),
        "input_native_actions":totals["actions"],
        "actions_that_fit_in_isolation":totals["actions_fit_individually"],
        "retained_complete_tasks":len(accepted),
        "retained_native_actions":actions_selected,
        "retained_source_counts":dict(sorted(source.items())),
        "rejected_complete_tasks":len(rejected),
        "rejection_diagnostics_no_questions":rejected,
        "require_full_prompt":True,
        "truncated_prompt_or_completion_count":0,
        "model_training_ran":False,
        "gaia_evaluated":False,
    }
    return accepted,report


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--protected",type=Path,required=True)
    parser.add_argument("--budget",type=int,default=2048)
    parser.add_argument("--model",default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    a=parser.parse_args()
    if a.output.resolve()==a.report.resolve() or a.output.exists() or a.report.exists():
        raise FileExistsError("refuse to overwrite curated source of truth")
    from gaia_small_agent.training.qlora import load_verified_rows,_render_turn_examples,_tokenize_completion_examples
    from transformers import AutoTokenizer
    rows=load_verified_rows(a.source,protected_questions_path=a.protected)
    if len(rows)!=40 or len({str(r["task_id"]) for r in rows})!=40:
        raise ValueError("frozen curated source must be exactly 40 independent tasks")
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    rendered=_render_turn_examples(tokenizer,rows)
    if len(rendered)!=164:
        raise ValueError("must start with original 164 native training actions")
    retained,report=filter_full_tasks(rows,rendered,tokenizer,budget=a.budget)
    # Independent repository tokenizer must verify every remaining example
    # using its actual trainer path with require_full_prompt=True.
    keep_ids={r["task_id"] for r in retained}
    kept=[e for e in rendered if e["metadata"]["task_id"] in keep_ids]
    encoded=_tokenize_completion_examples(
        tokenizer,kept,max_length=a.budget,require_full_prompt=True)
    if len(encoded)!=report["retained_native_actions"]:
        raise ValueError("trainer's full prompt safety invariant broken")
    report["model"]=a.model
    report["curated_input_sha256"]=sha256(a.source.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in retained),
        encoding="utf-8")
    report["retained_jsonl_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("QLORA_COMPLETE_TASK_2048",json.dumps({k:v for k,v in report.items()
        if k!="rejection_diagnostics_no_questions"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
