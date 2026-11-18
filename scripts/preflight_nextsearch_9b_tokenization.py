"""CPU-only genuine Qwen3.5-9B tokenizer compatibility/length check for SFT.

No model weights, no GPU and no training. Refuses to suggest a 9B QLoRA
run where all genuine source-grounded prompts would be silently truncated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from gaia_small_agent.training.qlora import load_verified_rows, _render_turn_examples

MODEL="Qwen/Qwen3.5-9B"
CONTRACT="nextsearch-original-bounded-source-9b-tokenizer-preflight-v1"


def audit(data:Path,processor,revision:str)->dict:
    rows=load_verified_rows(data)
    examples=_render_turn_examples(processor,rows)
    stats=[]
    actions=Counter()
    task_types=Counter()
    for x in examples:
        prompt=processor(x["prompt"],add_special_tokens=False)["input_ids"]
        completion=processor(x["completion"],add_special_tokens=False)["input_ids"]
        if not completion:
            raise ValueError("empty supervised completion")
        stats.append({
            "prompt_tokens":len(prompt),
            "completion_tokens":len(completion),
            "total_tokens":len(prompt)+len(completion),
        })
        actions[x["metadata"]["action_type"]]+=1
        task_types[x["metadata"]["task_id"]]+=1
    lengths=sorted(s["total_tokens"] for s in stats)
    if not lengths or len(task_types)!=len(rows):
        raise ValueError("verified source data did not reach SFT tokenizer")
    eligible={}
    for maximum in (1024,2048,4096,8192):
        n=sum(x["total_tokens"]<=maximum for x in stats)
        ids=set()
        for i,x in enumerate(examples):
            if stats[i]["total_tokens"]<=maximum:
                ids.add(x["metadata"]["task_id"])
        eligible[str(maximum)]={
            "full_context_turns":n,"unique_parent_tasks":len(ids),
            "supervised_tool_turns":sum(
                e["metadata"]["action_type"]=="tool_call"
                and stats[i]["total_tokens"]<=maximum
                for i,e in enumerate(examples)
            ),
            "supervised_final_turns":sum(
                e["metadata"]["action_type"]=="final"
                and stats[i]["total_tokens"]<=maximum
                for i,e in enumerate(examples)
            ),
        }
    return {
        "protocol":CONTRACT,"model":MODEL,"model_revision":revision,
        "data_sha256":sha256(data.read_bytes()).hexdigest(),
        "source_verified_parents":len(rows),
        "original_source_selected_assistant_turns":len(examples),
        "source_action_types":dict(actions),
        "prompt_completion_total_token_min":lengths[0],
        "prompt_completion_total_token_median":median(lengths),
        "prompt_completion_total_token_max":lengths[-1],
        "full_context_eligible":eligible,
        "unsafe_truncation_forbidden":True,
        "source_eval_success_independent":False,
        "training_performed":False,
        "cuda_used":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("cannot overwrite tokenization preflight")
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    sha=HfApi().model_info(MODEL).sha
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=sha,trust_remote_code=True)
    report=audit(a.data,tokenizer,sha)
    print("NEXTSEARCH_9B_REAL_TOKENIZER_PREFLIGHT",
          json.dumps(report,sort_keys=True),flush=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
