"""Public HotpotQA NEW8 Base Qwen9B vs v7 LoRA x frozen verbatim serializer.

Frozen after excluding the old Hotpot8+16 identities and annotated supports.
No GAIA, no stronger model, no gold sent to the model/serializer. Diagnostic
on public questions, cannot assert hidden-data or general reasoning transfer.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from scripts import eval_2wiki8_v7_factorial as original
from scripts import eval_hotpot_serializer_new16 as old
from scripts.eval_hotpot_source_shift import page_path,REPO,REV

N=8
SHARDS=4
PER_SHARD=2
PROTOCOL="PUBLIC-HOTPOT-NEW8-QWEN9B-V7-FACTORIAL-V1"
SEED="hotpot-disjoint-both-older-sources-2x2-v7-20261117"
PRIOR_OLD8=old.PREVIOUS_SOURCE_SHA
SOURCE_TYPES={"bridge":4,"comparison":4}

def keyed(x):
    return sha256(f"{SEED}:{x}".encode()).hexdigest()

def freeze(rows,prior8,prior16):
    if (prior8.get("cases_sha256")!=PRIOR_OLD8
        or prior8.get("n")!=8 or len(prior8.get("cases",[]))!=8
        or sha256(json.dumps(prior8["cases"],sort_keys=True).encode()).hexdigest()!=PRIOR_OLD8):
        raise ValueError("pinned historical Hotpot8 source mismatch")
    if (prior16.get("protocol")!=old.PROTOCOL
        or prior16.get("status")!="FROZEN_BEFORE_ANY_MODEL_INFERENCE"
        or prior16.get("n")!=16 or len(prior16.get("cases",[]))!=16
        or prior16.get("revision")!=REV
        or prior16.get("serializer_sha")!=old.SERIALIZER_SHA
        or sha256(json.dumps(prior16["cases"],sort_keys=True).encode()).hexdigest()!=prior16.get("new_source_sha")):
        raise ValueError("pinned previously evaluated Hotpot16 source mismatch")
    excluded_ids={c["id"] for c in prior8["cases"]+prior16["cases"]}
    excluded_support={p for c in prior8["cases"]+prior16["cases"] for p in c["support"]}
    if len(excluded_ids)!=24:
        raise ValueError("old Hotpot groups duplicate each other's identities")
    if len(excluded_support)<20:
        raise ValueError("old support document exclusions incomplete")
    frozen,used=[],set(excluded_support)
    counts=Counter()
    for row in sorted(rows,key=lambda r:keyed(str(r.get("id") or ""))):
        identity=str(row.get("id") or "")
        kind=str(row.get("type") or "")
        if not identity or identity in excluded_ids or kind not in SOURCE_TYPES or counts[kind]>=4:
            continue
        context=row.get("context") or {}
        titles=context.get("title") or []
        parts=context.get("sentences") or []
        if len(titles)!=len(parts) or len(titles)<8 or len(set(titles))!=len(titles):
            continue
        support=row.get("supporting_facts") or {}
        source_pairs=list(zip(support.get("title") or [],support.get("sent_id") or []))
        support_titles={title for title,_ in source_pairs}
        if len(support_titles)!=2 or not support_titles.issubset(set(titles)):
            continue
        paths={page_path(t) for t in support_titles}
        if paths & used:
            continue
        docs=dict(zip(titles,parts))
        src=defaultdict(list)
        for title,idx in source_pairs:
            if not isinstance(idx,int) or not 0<=idx<len(docs[title]):
                src.clear()
                break
            src[page_path(title)].append(str(docs[title][idx]))
        if len(src)!=2 or any(not t.strip() for v in src.values() for t in v):
            continue
        answer=str(row.get("answer") or "").strip()
        if not 1<=len(answer)<=80:
            continue
        if any(c["id"]==identity for c in frozen):
            continue
        frozen.append({
            "id":identity,"type":kind,"question":str(row["question"]),
            "answer":answer,
            "files":{page_path(t):t+"\n"+"\n".join(map(str,s)) for t,s in zip(titles,parts)},
            "index_titles":{t:page_path(t) for t in titles},
            "support":dict(src)
        })
        used.update(paths)
        counts[kind]+=1
        if len(frozen)==N:
            break
    if len(frozen)!=N or dict(counts)!=SOURCE_TYPES:
        raise RuntimeError(f"source-freeze insufficient balanced fresh identities: {len(frozen)}, {dict(counts)}")
    sha=original.stable_sha(frozen)
    return {
        "status":"PREINFERENCE_FROZEN_PUBLIC_HOTPOT8","protocol":PROTOCOL,
        "source_repo":REPO,"source_revision":REV,"split":"distractor/validation",
        "selection_seed":SEED,"n":N,"shards":SHARDS,
        "task_types":SOURCE_TYPES,
        "excluded_prior_task_ids":24,
        "excluded_prior_support_paths":len(excluded_support),
        "new_annotated_support_paths":len({p for c in frozen for p in c["support"]}),
        "source_sha":sha,"old8_sha":PRIOR_OLD8,"old16_sha":prior16["new_source_sha"],
        "base_gguf_sha256":original.BASE_SHA,"adapter_gguf_sha256":original.V7_SHA,
        "serializer_sha256":old.SERIALIZER_SHA,
        "public_not_hidden_or_GAIA":True,"untracked_exposure_and_pretraining_unknown":True,
        "no_training":True,"cases":frozen,
    }

def validate(obj):
    if (obj.get("status")!="PREINFERENCE_FROZEN_PUBLIC_HOTPOT8"
        or obj.get("protocol")!=PROTOCOL or obj.get("n")!=N or obj.get("shards")!=SHARDS
        or obj.get("source_revision")!=REV or obj.get("task_types")!=SOURCE_TYPES
        or obj.get("excluded_prior_task_ids")!=24
        or obj.get("base_gguf_sha256")!=original.BASE_SHA
        or obj.get("adapter_gguf_sha256")!=original.V7_SHA
        or obj.get("serializer_sha256")!=old.SERIALIZER_SHA
        or len(obj.get("cases",[]))!=N
        or obj.get("source_sha")!=original.stable_sha(obj["cases"])):
        raise ValueError("source/model SHA contract changed")
    ids=[x["id"] for x in obj["cases"]]
    if len(set(ids))!=N or Counter(x["type"] for x in obj["cases"])!=SOURCE_TYPES:
        raise ValueError("invalid balanced identities")
    if any(len(x["support"])!=2 or not set(x["support"])<=set(x["files"])
           for x in obj["cases"]):
        raise ValueError("invalid Hotpot support pointers")

def run_shard(obj,mode,shard,work):
    validate(obj)
    if mode not in ("base","adapter") or shard not in range(SHARDS):
        raise ValueError("not exactly two models and four paired shards")
    subset=obj["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    if len(subset)!=PER_SHARD:raise ValueError("missing cases")
    rows=[]
    for case in subset:
        arm_result=original.one_case(case,mode,work/keyed(case["id"])[:12])
        rows.append({"task_sha256":sha256(case["id"].encode()).hexdigest(),
                     "source_sha":obj["source_sha"],"type":case["type"],
                     "arms":arm_result})
        print("HOTPOT_FACTORIAL_QWEN",mode,shard,case["type"],
              {k:v["correct"] for k,v in arm_result.items()},flush=True)
    return {"status":"COMPLETE_PHYSICAL_ONE_MODEL_MODE","protocol":PROTOCOL,
            "source_sha":obj["source_sha"],"mode":mode,"shard":shard,"n":PER_SHARD,
            "base_gguf_sha256":original.BASE_SHA,
            "adapter_gguf_sha256":original.V7_SHA,
            "serializer_sha256":old.SERIALIZER_SHA,"cases":rows}

def aggregate(parts):
    if len(parts)!=2*SHARDS or any(p.get("protocol")!=PROTOCOL for p in parts):
        raise ValueError("need all eight physical Hotpot model/shard outputs")
    normalized=[]
    for p in parts:
        q=dict(p)
        q["protocol"]=original.PROTOCOL
        normalized.append(q)
    result=original.aggregate(normalized)
    per_id={}
    for part in parts:
        for c in part["cases"]:
            previous=per_id.setdefault(c["task_sha256"],c["type"])
            if previous!=c["type"]:
                raise ValueError("same task reported with conflicting types")
    if len(per_id)!=N or Counter(per_id.values())!=SOURCE_TYPES:
        raise ValueError("unbalanced/missing Hotpot cases")
    result["protocol"]=PROTOCOL
    result["status"]="COMPLETE_HOTPOT_DIAGNOSTIC_2X2_NO_PROMOTION"
    result["case_types"]=SOURCE_TYPES
    result["public_hotpot_not_gaia"]=True
    result["transfer_to_real_agent_tasks_not_proven"]=True
    result["claim_boundary"]="Public source-disjoint HotpotQA: same Qwen3.5-9B, original v7 LoRA and identical frozen same-model extraction rule; proof of task-family shift from earlier 2Wiki film sample, NOT GAIA or hidden transfer."
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--prior8",type=Path)
    p.add_argument("--prior16",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--mode",choices=["base","adapter"])
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--work",type=Path,default=Path("hotpot8-factorial-work"))
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("immutable source or output exists")
    if args.source_only:
        if args.prior8 is None or args.prior16 is None:p.error("must pass --prior8 and --prior16")
        from datasets import load_dataset
        rows=list(islice(load_dataset(REPO,"distractor",split="validation",revision=REV,streaming=True),2000))
        result=freeze(rows,json.loads(args.prior8.read_text()),json.loads(args.prior16.read_text()))
    elif args.inputs:
        result=aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("HOTPOT8_CROSS_FAMILY_FACTORIAL_COMPLETE",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if args.frozen is None or args.shard is None or args.mode is None:
            p.error("need frozen, shard and mode")
        result=run_shard(json.loads(args.frozen.read_text()),args.mode,args.shard,args.work)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
