"""Label-blind CPU trace census: what *happened*, never what answer was right.

Reads already-frozen real model tool event traces, but NEVER opens answer/gold,
human decomposition, correctness, or failure labels. Exports aggregates only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from gaia_small_agent.agent.evidence_state import _entities


def original_rows(root: Path, arm: str) -> list[dict]:
    paths = sorted(root.glob(f"gaia-v8-paired-shard-*/{arm}/shard-*/results.jsonl"))
    if len(paths) != 20:
        raise ValueError(f"Expected exactly 20 original paired shards, got {len(paths)}")
    rows=[]
    for p in paths:
        for line in p.read_text("utf-8").splitlines():
            if line.strip():
                raw=json.loads(line)
                # Whitelist fields BEFORE any computation; never access score/answer/gold.
                rows.append({
                    "trace":raw["trace"],
                    "stop_reason":raw["stop_reason"],
                    "tool_calls":raw["tool_calls"],
                })
    if len(rows)!=100:
        raise ValueError(f"Expected 100 original trajectories, got {len(rows)}")
    return rows


def summarize(rows: list[dict]) -> dict:
    n=len(rows)
    tally=Counter()
    stop_reasons=Counter()
    error_codes=Counter()
    total_calls=0
    for row in rows:
        events=row["trace"]
        if not isinstance(events,list):
            raise ValueError("Invalid trace")
        calls=0
        read_sources=set()
        successes=[]
        for event in events:
            data=event.get("data") or {}
            kind=event.get("kind")
            if kind=="tool_call":
                calls+=1
                if data.get("name")=="read":
                    raw_args=data.get("arguments") or {}
                    if isinstance(raw_args,dict):
                        source=raw_args.get("source")
                        if isinstance(source,str) and source:
                            read_sources.add(source)
            if kind=="tool_result":
                if data.get("ok") is False:
                    error_codes[str(data.get("error_code") or "UNKNOWN")]+=1
                elif data.get("name")=="read" and data.get("ok") is True:
                    # Source identity comes from the actual tool-call event with the same id.
                    link=next((c.get("data") or {} for c in events
                               if c.get("kind")=="tool_call" and
                               (c.get("data") or {}).get("id")==data.get("id")), {})
                    arguments=link.get("arguments") or {}
                    source=arguments.get("source") if isinstance(arguments,dict) else None
                    if isinstance(source,str):
                        successes.append((source,_entities(str(data.get("content") or "")[:1600])))
        if calls!=row["tool_calls"]:
            raise ValueError("Trace call count mismatch")
        total_calls+=calls
        if calls==0:
            tally["no_tool_calls"]+=1
        if not read_sources:
            tally["never_requested_read"]+=1
        if not successes:
            tally["no_successful_reads"]+=1
        if len({source for source,_ in successes})>=2:
            tally["two_distinct_successful_read_sources"]+=1
        bridging=any(source!=source2 and bool(names&names2)
                     for i,(source,names) in enumerate(successes)
                     for source2,names2 in successes[i+1:])
        if bridging:
            tally["two_sources_share_capitalized_name"]+=1
        if any(e.get("kind")=="tool_result" and (e.get("data") or {}).get("error_code")=="DUPLICATE_CALL" for e in events):
            tally["duplicate_tool_call_error"]+=1
        stop_reasons[str(row["stop_reason"])]+=1
    return {
        "n":n,
        "total_tool_calls":total_calls,
        "task_counts":dict(sorted(tally.items())),
        "stop_reasons":dict(sorted(stop_reasons.items())),
        "tool_error_codes":dict(sorted(error_codes.items())),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result={
        "protocol":"GAIA100-TOOL-EVIDENCE-STRUCTURE-LABELBLIND-20261123-V1",
        "source_run_id":44980055775,
        "base":summarize(original_rows(a.root,"base")),
        "step100":summarize(original_rows(a.root,"qlora-v8")),
        "no_task_content_or_gold_exported":True,
        "gold_answer_fields_read":False,
        "no_model_inference":True,
        "interpretation":"Read success and lexical overlap are proxies only. Without gold supporting passages or human judgments, cannot distinguish retrieval, extraction and semantic composition failures or assert causality.",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("GAIA100_LABELBLIND_EVIDENCE_STRUCTURE",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
