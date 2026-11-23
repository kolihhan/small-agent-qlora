"""Posthoc outcome association AFTER blind source-coverage buckets are fixed.

This stage reads only the official boolean correctness field, never answers,
human decompositions, task text or model responses. It is descriptive and
must never guide test-set thresholds, training, or main policy promotion.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


CATEGORIES=(
    "no_read_requested",
    "read_attempted_no_success",
    "one_distinct_successful_read_source",
    "multiple_distinct_successful_read_sources",
)


def bucket(trace: list[dict]) -> str:
    requested=[]
    successes=set()
    id_sources={}
    for event in trace:
        data=event.get("data") or {}
        if event.get("kind")=="tool_call" and data.get("name")=="read":
            raw_args=data.get("arguments") or {}
            source=raw_args.get("source") if isinstance(raw_args,dict) else None
            if not isinstance(source,str) or not source:
                source=f"<unknown-id:{data.get('id')}>"
            requested.append(source)
            id_sources[str(data.get("id"))]=source
        if (event.get("kind")=="tool_result" and data.get("name")=="read"
            and data.get("ok") is True):
            source=id_sources.get(str(data.get("id")))
            if source is not None:
                successes.add(source)
    if not requested:
        return CATEGORIES[0]
    if not successes:
        return CATEGORIES[1]
    return CATEGORIES[2] if len(successes)==1 else CATEGORIES[3]


def summarize(rows) -> dict:
    groups=defaultdict(Counter)
    counts=Counter()
    n=0
    for row in rows:
        label=row["correct"]
        if type(label) is not bool or type(row["completed"]) is not bool:
            raise ValueError("Malformed benchmark bool labels")
        category=bucket(row["trace"])
        g=groups[category]
        g["n"]+=1
        g["correct"]+=label
        g["completed_wrong"]+=row["completed"] and not label
        g["invalid_final"]+=row["stop_reason"]=="invalid_final"
        counts["correct"]+=label
        n+=1
    return {
        "n":n,"correct":counts["correct"],
        "categories":{name:{
            "n":groups[name]["n"],
            "correct":groups[name]["correct"],
            "completed_wrong":groups[name]["completed_wrong"],
            "invalid_final":groups[name]["invalid_final"],
        } for name in CATEGORIES},
    }


def original_rows(root: Path, arm: str) -> list[dict]:
    paths=sorted(root.glob(f"gaia-v8-paired-shard-*/{arm}/shard-*/results.jsonl"))
    if len(paths)!=20:
        raise ValueError("Expected 20 frozen run shards per arm")
    rows=[]
    for path in paths:
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                rec=json.loads(line)
                rows.append({
                    "trace":rec["trace"],
                    "correct":rec["correct"],
                    "completed":rec["completed"],
                    "stop_reason":rec["stop_reason"],
                })
    if len(rows)!=100:
        raise ValueError("Expected exactly 100 frozen GAIA rows per arm")
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    base=summarize(original_rows(args.root,"base"))
    v8=summarize(original_rows(args.root,"qlora-v8"))
    if (base["correct"],v8["correct"])!=(20,21):
        raise ValueError("Not original pinned GAIA100 run")
    result={
        "protocol":"GAIA100-READ-SUCCESS-POSTHOC-ASSOCIATION-V1",
        "source_run_id":44980055775,
        "predefined_coverage_categories":list(CATEGORIES),
        "base":base,
        "step100":v8,
        "uses_only_boolean_scoring_label":True,
        "does_not_read_task_content_or_gold_answers":True,
        "no_model_inference":True,
        "interpretation":"Association only. Some tasks require no reads; hard tasks may produce more reads. No causal diagnosis or GAIA improvement claim. Do not use old public evaluation to tune a new policy.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("GAIA100_POSTHOC_READ_SUCCESS_ASSOCIATION",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
