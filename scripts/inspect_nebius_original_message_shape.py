"""Inspect *structural fields only* of first 20 real Nebius execution trajectories.

No user messages, code, tool observations, patches or source text printed/exported.
Pinned source revision from preceding source audit. No model or GPU.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import islice
import json

REPO="nebius/SWE-rebench-openhands-trajectories"
REV="35455389ab51bf5e2306bfd436ef72d0f98bf882"

def shapes(rows):
    result=Counter(); examples=[]
    for row in rows:
        result["total"]+=1
        for key in sorted(row):
            val=row[key]
            result["top:"+key+":"+type(val).__name__]+=1
        seq=row.get("trajectory")
        if not isinstance(seq,list):
            result["trajectory_not_list"]+=1
            continue
        result["trajectory_len_bucket:"+str(min(len(seq)//10,30))]+=1
        for msg in seq:
            if not isinstance(msg,dict):
                result["msg_not_dict"]+=1
                continue
            role=str(msg.get("role"))
            result["role:"+role]+=1
            if "tool_calls" in msg:
                calls=msg.get("tool_calls")
                result["tool_calls_type:"+type(calls).__name__]+=1
                if isinstance(calls,list):
                    result["tool_calls_count"]+=len(calls)
                    if calls:
                        result["tool_call_keys:"+",".join(sorted(str(k) for k in calls[0])) if isinstance(calls[0],dict) else "tool_call_non_dict"]+=1
            if "tool_call_id" in msg:
                result["msg_tool_call_id"]+=1
            if "content" in msg:
                result["content_type:"+type(msg["content"]).__name__]+=1
                if isinstance(msg["content"],list) and msg["content"]:
                    v=msg["content"][0]
                    result["content_list_item_type:"+type(v).__name__]+=1
                    if isinstance(v,dict):
                        result["content_list_item_keys:"+",".join(sorted(v.keys()))]+=1
            keyset=",".join(sorted(msg))
            result["message_keys:"+keyset]+=1
        if len(examples)<3:
            examples.append({
                "row_keys":sorted(row),
                "first_eight_messages":[{
                    "role":str(msg.get("role")) if isinstance(msg,dict) else type(msg).__name__,
                    "keys":sorted(msg) if isinstance(msg,dict) else [],
                    "content_type":type(msg.get("content")).__name__ if isinstance(msg,dict) else "",
                    "content_len":len(msg.get("content")) if isinstance(msg,dict) and isinstance(msg.get("content"),(str,list)) else None,
                    "calls_type":type(msg.get("tool_calls")).__name__ if isinstance(msg,dict) else "",
                    "calls_len":len(msg.get("tool_calls")) if isinstance(msg,dict) and isinstance(msg.get("tool_calls"),list) else None
                } for msg in seq[:8]],
                "trajectory_len":len(seq),
                "resolved":int(row.get("resolved") or 0)
            })
    return {"sample_count":len(rows),"counts":dict(result),"example_structures_only":examples}

def main():
    from datasets import load_dataset
    rows=list(islice(load_dataset(REPO,split="train",revision=REV,streaming=True),20))
    print("NEBIUS_ORIGINAL_SCHEMA_ONLY",json.dumps(shapes(rows),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
