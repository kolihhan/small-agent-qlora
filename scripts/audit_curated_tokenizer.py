"""CPU-only Qwen3.5-9B real teacher data token/coverage audit before SFT.

Never trains models, samples GAIA answers, silently truncates context, or
over-samples tiny source families. Real collected Hotpot+2Wiki read trajectories
are audited as-is, not treated as evidence of downstream GAIA success.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.qlora import load_verified_rows,_render_turn_examples


def analyze_examples(examples:list[dict],tokenizer,source_by_task:dict[str,str])->dict:
    if not examples:
        raise ValueError("No rendered native train actions")
    sizes=[]
    by_source=defaultdict(lambda:Counter())
    by_action=Counter()
    by_action_fit=defaultdict(Counter)
    by_task=defaultdict(Counter)
    for index,ex in enumerate(examples):
        prompt=ex["prompt"]
        completion=ex["completion"]
        metadata=ex["metadata"]
        task_id=metadata["task_id"]
        if task_id not in source_by_task:
            raise ValueError("unattributed source in rendered model completion")
        a=tokenizer(prompt,add_special_tokens=False)["input_ids"]
        b=tokenizer(completion,add_special_tokens=False)["input_ids"]
        if not b:raise ValueError(f"empty completion at action {index}")
        size=len(a)+len(b)
        sizes.append(size)
        source=source_by_task[task_id]
        by_source[source]["actions"]+=1
        action_type=str(metadata["action_type"])
        by_action[action_type]+=1
        by_task[task_id]["actions"]+=1
        if action_type=="final":
            by_task[task_id]["final_seen"]+=1
        if size<=1024:
            by_task[task_id]["fits_1024"]+=1
            by_action_fit[action_type]["fits_1024"]+=1
        if size<=2048:
            by_task[task_id]["fits_2048"]+=1
            by_action_fit[action_type]["fits_2048"]+=1
        for budget in (1024,1536,2048,4096):
            if size<=budget: by_source[source][f"fits_{budget}"]+=1
        if len(b)>1024:by_source[source]["completion_longer_than_t4_budget"]+=1
    ordered=sorted(sizes)
    def percentile(p):
        return ordered[min(len(ordered)-1,int((len(ordered)-1)*p))]
    return {
        "schema":"real-multisource-qwen-tokenizer-preflight-v1",
        "source_families":dict(sorted(Counter(source_by_task.values()).items())),
        "unique_tasks":len(source_by_task),
        "native_train_actions":len(examples),
        "action_type_counts":dict(sorted(by_action.items())),
        "action_type_token_fit":{k:dict(sorted(v.items())) for k,v in sorted(by_action_fit.items())},
        "fully_supported_tasks_at_1024":sum(
            v["actions"]==v["fits_1024"] and v["final_seen"]==1
            for v in by_task.values()),
        "fully_supported_tasks_at_2048":sum(
            v["actions"]==v["fits_2048"] and v["final_seen"]==1
            for v in by_task.values()),
        "tasks_with_complete_final_action":sum(
            v["final_seen"]==1 for v in by_task.values()),
        "source_token_fit":{k:dict(sorted(v.items())) for k,v in sorted(by_source.items())},
        "max_total_tokens":max(sizes),
        "median_total_tokens":percentile(0.5),
        "p90_total_tokens":percentile(0.9),
        "p99_total_tokens":percentile(0.99),
        "fits_t4_1024":sum(size<=1024 for size in sizes),
        "fits_2048":sum(size<=2048 for size in sizes),
        "fits_4096":sum(size<=4096 for size in sizes),
        "untruncated_training_ran":False,
        "gaia_evaluated":False,
        "caveat":"Observed extractive reading tool traces; without source diversity and independent evaluations, not proof of better GAIA transfer.",
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--protected",type=Path,required=True)
    p.add_argument("--expected-tasks",type=int,default=40)
    p.add_argument("--expected-actions",type=int,default=164)
    p.add_argument("--model",default="Qwen/Qwen3.5-9B")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwriting previous tokenizer evidence")
    from transformers import AutoTokenizer
    rows=load_verified_rows(args.data,protected_questions_path=args.protected)
    if len(rows)!=args.expected_tasks or len({r["task_id"] for r in rows})!=len(rows):
        raise ValueError("strict curator task identity mismatch")
    if sum(len(r.get("training_action_indices") or []) for r in rows)!=args.expected_actions:
        raise ValueError("training action index count mismatch")
    source={str(r["task_id"]):str(r["source"]) for r in rows}
    if set(source.values())!={"hotpotqa-distractor-train","2wikimultihopqa-train"}:
        raise ValueError("unexpected curated source family")
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    rendered=_render_turn_examples(tokenizer,rows)
    if len(rendered)!=args.expected_actions:
        raise ValueError("rendered action count inconsistent with curated message indices")
    report=analyze_examples(rendered,tokenizer,source)
    report["model"]=args.model
    report["data_sha256"]=sha256(args.data.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("REAL_MULTI_SOURCE_TOKEN_AUDIT",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
