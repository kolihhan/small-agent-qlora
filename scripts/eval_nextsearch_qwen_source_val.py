"""Source-disjoint Qwen9B autonomous web research baseline using original tool APIs.

Matched future QLoRA candidate uses the same frozen Apache NextSearch VAL
questions, public live web search/fetch and native AgentRuntime. Original
TRAIN answers/targets never enter the model. The source's reference answer
is only opened post hoc for strict scoring of already final outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from time import monotonic
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from scripts.nextsearch_source_tools import ResearchSearch, ResearchFetch
from scripts.audit_nextsearch_apache_source import REPO, CONFIG, FROZEN_REVISION

PROTOCOL="nextsearch-qwen9b-native-search-fetch-goldblind-source-val-v1"
MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
N=8
SHARDS=4


def key(s:str)->str:
    return sha256((PROTOCOL+":"+s).encode()).hexdigest()


def freeze(train:list[dict], val:list[dict])->dict:
    trainids={str(r.get("id") or "") for r in train}
    eligible_by_source=defaultdict(list)
    seen=set()
    for row in val:
        identity=str(row.get("id") or "")
        if not identity or identity in trainids or identity in seen:
            continue
        seen.add(identity)
        if str(row.get("license") or "").casefold()!="apache-2.0":
            continue
        gold=row.get("gold")
        reference=gold.get("answer") if isinstance(gold,dict) else None
        if not isinstance(reference,(str,int,float)):
            continue
        answer=str(reference).strip()
        prompt=row.get("messages") or []
        questions=[r["content"] for r in prompt
                   if isinstance(r,dict) and r.get("role")=="user"
                   and isinstance(r.get("content"),str)]
        if not questions or not 1<=len(answer)<=120:
            continue
        question=questions[-1].strip()
        if not 20<=len(question)<=3000:
            continue
        if answer.casefold() in question.casefold():
            continue
        source=str(row.get("source") or "")
        if source not in ("asearcher","taskcraft","quest"):
            continue
        eligible_by_source[source].append({
            "id":identity,"source":source,
            "question":question,"answer":answer,
            "band":str(row.get("band") or ""),
        })
    selected=[]
    # Include up to 2 questions per source before filling the remaining slots.
    for source in ("asearcher","taskcraft","quest"):
        selected.extend(sorted(eligible_by_source[source],
                               key=lambda r:key(r["id"]))[:2])
    used={r["id"] for r in selected}
    remaining=sorted(
        (r for rows in eligible_by_source.values() for r in rows
         if r["id"] not in used),
        key=lambda r:key(r["id"])
    )
    selected=(selected+remaining)[:N]
    if len(selected)!=N or len({r["id"] for r in selected})!=N:
        raise ValueError("insufficient original task-disjoint VAL source")
    if len({r["source"] for r in selected})<2:
        raise ValueError("source-diversity too narrow for independent baseline")
    return {
        "protocol":PROTOCOL,"source_repo":REPO,"source_revision":FROZEN_REVISION,
        "split":"validation","original_train_ids_excluded":len(trainids),
        "heldout_by_task_id":True,"no_gold_to_model":True,
        "model_digest":DIGEST,
        "cases_sha":sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
        "cases":selected
    }


def one(case:dict,work:Path)->dict:
    model=OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=512)
    prompt=(
        "Answer the question using public web research. The search tool accepts "
        "an objective and search_queries; the fetch tool reads public URLs. "
        "Check relevant sources and return only the final short answer. "
        "Do not cite an answer you have not verified.\n\nQuestion: "+
        case["question"]
    )
    start=monotonic()
    result=AgentRuntime(
        model,[ResearchSearch(),ResearchFetch()],max_steps=14
    ).run(prompt,work)
    # Do not expose reference to the model. It is used below only after
    # the model loop has already finalized.
    names=Counter(e.data.get("name") for e in result.trace if e.kind=="tool_call")
    observations=Counter()
    for e in result.trace:
        if e.kind=="tool_result":
            observations["total"]+=1
            observations["ok"]+=int(bool(e.data.get("ok")))
            observations["errors"]+=int(not bool(e.data.get("ok")))
    return {
        "id":case["id"],"source":case["source"],
        "answer":result.answer,"reference":case["answer"],
        "exact":bool(result.completed and gaia_score(result.answer,case["answer"])),
        "completed":result.completed,
        "stop_reason":result.stop_reason,
        "steps":result.metrics.steps,
        "tool_calls":result.metrics.tool_calls,
        "tool_errors":result.metrics.tool_errors,
        "actions_by_name":dict(names),
        "observations":dict(observations),
        "seconds":round(monotonic()-start,2),
    }


def shard_run(cohort:dict,shard:int,root:Path)->dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    if cohort["protocol"]!=PROTOCOL or cohort["model_digest"]!=DIGEST:
        raise ValueError("model/protocol changed")
    if sha256(json.dumps(cohort["cases"],sort_keys=True).encode()).hexdigest()!=cohort["cases_sha"]:
        raise ValueError("cohort cases drift")
    with urlopen("http://127.0.0.1:11434/api/tags",timeout=10) as response:
        models=json.load(response)["models"]
    if not any(m["name"]==MODEL and m["digest"]==DIGEST for m in models):
        raise ValueError("Qwen9B physical model checksum drift")
    out=[]
    for case in cohort["cases"][2*shard:2*shard+2]:
        record=one(case,root/key(case["id"])[:12])
        out.append(record)
        print("NEXTSEARCH_QWEN9B_ACTUAL",
              record["id"],record["exact"],record["tool_calls"],record["tool_errors"],
              flush=True)
    return {"protocol":PROTOCOL,"digest":DIGEST,"cohort_sha":cohort["cases_sha"],
            "shard":shard,"cases":out}


def aggregate(parts:list[dict])->dict:
    if len(parts)!=SHARDS or sorted(p["shard"] for p in parts)!=list(range(SHARDS)):
        raise ValueError("incomplete paired source baseline")
    if len({p["cohort_sha"] for p in parts})!=1 or any(
        p["protocol"]!=PROTOCOL or p["digest"]!=DIGEST or len(p["cases"])!=2
        for p in parts):
        raise ValueError("model/subject provenance drift")
    rows=[r for p in sorted(parts,key=lambda x:x["shard"]) for r in p["cases"]]
    if len(rows)!=N or len({r["id"] for r in rows})!=N:
        raise ValueError("missing/duplicate independent source results")
    return {
        "protocol":PROTOCOL,"model_digest":DIGEST,
        "tasks":N,
        "sources":dict(Counter(r["source"] for r in rows)),
        "exact":sum(r["exact"] for r in rows),
        "completed":sum(r["completed"] for r in rows),
        "tool_calls":sum(r["tool_calls"] for r in rows),
        "tool_errors":sum(r["tool_errors"] for r in rows),
        "search_calls":sum(r["actions_by_name"].get("search",0) for r in rows),
        "fetch_calls":sum(r["actions_by_name"].get("fetch",0) for r in rows),
        "elapsed_seconds":round(sum(r["seconds"] for r in rows),1),
        "new_adapter_training_steps":0,
        "not_GAIA":True,"promotion_allowed":False,
        "cases":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--cohort",type=Path)
    ap.add_argument("--shard",type=int)
    ap.add_argument("--inputs",type=Path,nargs="+")
    ap.add_argument("--workspace",type=Path,default=Path("nextsearch-eval-work"))
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inputs:
        result=aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("NEXTSEARCH_QWEN9B_SOURCE_VAL_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),
              flush=True)
    elif args.cohort:
        result=shard_run(json.loads(args.cohort.read_text()),args.shard,args.workspace)
    else:
        from datasets import load_dataset
        tr=list(islice(load_dataset(REPO,CONFIG,split="train",
                                   revision=FROZEN_REVISION,streaming=True),3270))
        va=list(load_dataset(REPO,CONFIG,split="validation",
                             revision=FROZEN_REVISION,streaming=True))
        if len(tr)!=3270 or len(va)<100:
            raise ValueError("original source train/val truncated")
        result=freeze(tr,va)
        print("NEXTSEARCH_QWEN_SOURCE_VAL_FROZEN",result["cases_sha"],
              [r["source"] for r in result["cases"]],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
