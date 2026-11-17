"""Physical, CPU-only, pre-registered Base vs original v7 LoRA x frozen
same-model verbatim serializer on NEW public 2Wiki validation8 tasks.
No new weights, no GAIA cases, no gold in prompts, no source tuning.
Public data / pretraining and older unrecorded exposures remain limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
import math
import os
from pathlib import Path
from time import monotonic

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts import eval_2wiki_serializer_confirm48 as original
from scripts.eval_hotpot_serializer_new16 import SERIALIZER_SHA, serialize

PROTOCOL="2WIKI-PUBLIC-NEW8-V7-GGUF-FACTORIAL-V1"
OLD48_SHA="10eb336fcec4bd171331c9922753a057d7252adcb7ec3e19351b66c5b5d6668d"
BASE_SHA="8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
V7_SHA="475f5fb59a553390a0dde9425d36b42ff85a0345bdd31b8465c22fd4d2bc02e2"
MODEL="qwen3.5-9b"
N=8
SHARDS=4
PER_SHARD=2
ARMS=("base_native","base_serializer","adapter_native","adapter_serializer")
PAIR_KEYS=(("base_native","adapter_native"),
           ("base_native","base_serializer"),
           ("adapter_native","adapter_serializer"),
           ("base_serializer","adapter_serializer"))
GATE={"min_net_adapter_native":2,"max_adapter_native_regressions":1,
      "min_net_serializer":2,"max_serializer_regressions":1}
SOURCE_REV=original.REV

def stable_sha(value):
    return sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def freeze_source(rows,old48,prior_linked,prior_novel,prior16):
    original.validate_frozen(old48)
    if old48["frozen_source_sha256"]!=OLD48_SHA or len(old48["cases"])!=48:
        raise ValueError("prior 48 scored source not exact pinned corpus")
    if original.SERIALIZER_SHA!=SERIALIZER_SHA:
        raise ValueError("formatter SHA mismatch")
    # Re-run exact hash-selected source scan with same seed and prior 40 exclusions;
    # selecting 56 rather than 48 forces the first 48 to be identical by source order.
    # This avoids post-hoc choice of new cases by their answer or prior model results.
    previous_n=original.N
    try:
        original.N=56
        chosen=original.freeze(rows,prior_linked,prior_novel,prior16)["cases"]
    finally:
        original.N=previous_n
    if len(chosen)!=56:
        raise ValueError("source scan failed to find 56 examples")
    if stable_sha(chosen[:48])!=OLD48_SHA:
        raise ValueError("historical first48 source selection drift")
    new=chosen[48:]
    if len(new)!=N or len({c["id"] for c in new})!=N:
        raise ValueError("source lacks eight unique disjoint new tasks")
    prior_ids={c["id"] for c in chosen[:48]}
    prior_support={k for c in chosen[:48] for k in c["support"]}
    if any(c["id"] in prior_ids or prior_support.intersection(c["support"])
           for c in new):
        raise ValueError("new task/support path intersects previous 48")
    source_sha=stable_sha(new)
    return {"status":"SOURCE_FROZEN_BEFORE_MODEL_INFERENCE",
            "protocol":PROTOCOL,"source_rev":SOURCE_REV,
            "task_selection":"same source-order first56; use positions 49-56, not outcome-selected",
            "source_sha":source_sha,"known_prior48_sha":OLD48_SHA,
            "excludes_recorded_40_earlier_tasks":True,
            "excludes_48_latest_scored_tasks_and_support_docs":True,
            "new_ids":len(new),"new_support_doc_paths":len({p for c in new for p in c["support"]}),
            "base_gguf_sha256":BASE_SHA,"adapter_gguf_sha256":V7_SHA,
            "serializer_sha256":SERIALIZER_SHA,"n":N,"shards":SHARDS,
            "gate":GATE,"not_GAIA_or_hidden":True,
            "pretraining_exposure_unverified":True,"scored_source_selected_public":True,
            "source_independence_vs_non_support_distractors_unverified":True,
            "no_training":True,"cases":new}

def validate_frozen(obj):
    if (obj.get("status")!="SOURCE_FROZEN_BEFORE_MODEL_INFERENCE"
        or obj.get("protocol")!=PROTOCOL or obj.get("source_rev")!=SOURCE_REV
        or obj.get("base_gguf_sha256")!=BASE_SHA
        or obj.get("adapter_gguf_sha256")!=V7_SHA
        or obj.get("serializer_sha256")!=SERIALIZER_SHA
        or obj.get("known_prior48_sha")!=OLD48_SHA
        or obj.get("n")!=N or obj.get("shards")!=SHARDS
        or obj.get("gate")!=GATE
        or len(obj.get("cases",[]))!=N
        or stable_sha(obj["cases"])!=obj.get("source_sha")):
        raise ValueError("frozen source and model contract mismatch")
    ids=[c["id"] for c in obj["cases"]]
    if len(set(ids))!=N:
        raise ValueError("duplicate source identities")
    if any(len(c["support"])!=4 or not set(c["support"])<=set(c["files"])
           for c in obj["cases"]):
        raise ValueError("invalid four-support source")

class CountingOpenAI(OpenAICompatModel):
    def __init__(self,max_new_tokens=512):
        super().__init__(model=MODEL,max_new_tokens=max_new_tokens,timeout_s=400)
        self.calls=0
    def complete(self,messages,tools):
        self.calls+=1
        return super().complete(messages,tools)
    def complete_final(self,messages,tools):
        self.calls+=1
        return super().complete_final(messages,tools)

def one_case(case,adapter_mode,work):
    _materialize_files({"files":case["files"]},work)
    model=CountingOpenAI()
    question=case["question"]
    prompt=("Answer the following question using the local article search and read tools. "
            "Search for article names and read returned paths before answering. "
            "A local source may be a distractor; decide whether more evidence is needed. "
            "Return only the answer, not reasoning.\n\nQuestion: "+question)
    start=monotonic()
    result=AgentRuntime(model,[original.LocalSearch(case["index_titles"],case["files"]),ReadTool()],
                        max_steps=12).run(prompt,work)
    agent_elapsed=round(monotonic()-start,2)
    raw=result.answer
    formatter=CountingOpenAI(max_new_tokens=128)
    changed=raw
    status="not_completed"
    format_elapsed=0.0
    if result.completed and raw.strip():
        fmt_start=monotonic()
        changed,status=serialize(question,raw,formatter)
        format_elapsed=round(monotonic()-fmt_start,2)
        if status=="accepted_verbatim" and changed.casefold() not in raw.casefold():
            raise RuntimeError("formatter output not verbatim source substring")
    base={"correct":bool(result.completed and gaia_score(raw,case["answer"])),
          "model_calls":model.calls,"elapsed_s":agent_elapsed,
          "completed":result.completed,"tool_calls":result.metrics.tool_calls,
          "tool_errors":result.metrics.tool_errors,"stop_reason":result.stop_reason}
    serialized={"correct":bool(result.completed and gaia_score(changed,case["answer"])),
                "model_calls":model.calls+formatter.calls,
                "elapsed_s":round(agent_elapsed+format_elapsed,2),
                "completed":result.completed,"tool_calls":result.metrics.tool_calls,
                "tool_errors":result.metrics.tool_errors,
                "formatter_status":status,
                "formatter_calls":formatter.calls,"formatter_seconds":format_elapsed,
                "stop_reason":result.stop_reason}
    return {f"{adapter_mode}_native":base,f"{adapter_mode}_serializer":serialized}

def run_shard(frozen,shard,mode,work):
    validate_frozen(frozen)
    if shard not in range(SHARDS) or mode not in ("base","adapter"):
        raise ValueError("invalid shard or model treatment")
    cases=frozen["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    if len(cases)!=PER_SHARD:
        raise RuntimeError("incomplete source shard")
    result=[]
    for c in cases:
        rows=one_case(c,mode,work/sha256(c["id"].encode()).hexdigest()[:16])
        result.append({"task_sha256":sha256(c["id"].encode()).hexdigest(),
                       "source_sha":frozen["source_sha"],"arms":rows})
        print("FACTORIAL_PHYSICAL",shard,mode,result[-1]["task_sha256"][:12],
              {k:v["correct"] for k,v in rows.items()},flush=True)
    return {"status":"COMPLETE_PHYSICAL_ONE_MODEL_MODE","protocol":PROTOCOL,
            "mode":mode,"shard":shard,"n":len(result),
            "source_sha":frozen["source_sha"],"base_gguf_sha256":BASE_SHA,
            "adapter_gguf_sha256":V7_SHA,"serializer_sha256":SERIALIZER_SHA,
            "cases":result}

def _p(b,c):
    d=b+c
    return min(1.,2*sum(math.comb(d,i) for i in range(min(b,c)+1))/(2**d)) if d else 1.

def aggregate(shards):
    if len(shards)!=2*SHARDS:
        raise ValueError("require all 8 base/adapter physical source shards")
    expected={(mode,i) for mode in ("base","adapter") for i in range(SHARDS)}
    got={(s.get("mode"),s.get("shard")) for s in shards}
    if got!=expected or len(got)!=len(shards):
        raise ValueError("duplicate/missing 2x4 shard identities")
    for s in shards:
        if (s.get("status")!="COMPLETE_PHYSICAL_ONE_MODEL_MODE" or
            s.get("protocol")!=PROTOCOL or s.get("n")!=PER_SHARD or
            s.get("base_gguf_sha256")!=BASE_SHA or
            s.get("adapter_gguf_sha256")!=V7_SHA or
            s.get("serializer_sha256")!=SERIALIZER_SHA or
            len(s.get("cases",[]))!=PER_SHARD):
            raise ValueError("incomplete physical result or model SHA mismatch")
    source={s["source_sha"] for s in shards}
    if len(source)!=1:
        raise ValueError("source identity mismatch")
    matched={}
    for s in shards:
        for case in s["cases"]:
            id=case["task_sha256"]
            slot=matched.setdefault(id,{"source_sha":s["source_sha"],"arms":{}})
            if any(k in slot["arms"] for k in case["arms"]):
                raise ValueError("duplicate arm on a case")
            slot["arms"].update(case["arms"])
    if len(matched)!=N or any(set(r["arms"])!=set(ARMS) for r in matched.values()):
        raise ValueError("not exactly 8 matched 2x2 tasks")
    rows=list(matched.values())
    totals={k:{"correct":sum(r["arms"][k]["correct"] for r in rows),
                "calls":sum(r["arms"][k]["model_calls"] for r in rows),
                "elapsed_s":round(sum(r["arms"][k]["elapsed_s"] for r in rows),2),
                "completed":sum(r["arms"][k]["completed"] for r in rows),
                "tools":sum(r["arms"][k]["tool_calls"] for r in rows)}
            for k in ARMS}
    paired={}
    for a,b in PAIR_KEYS:
        rescue=sum(not r["arms"][a]["correct"] and r["arms"][b]["correct"] for r in rows)
        regress=sum(r["arms"][a]["correct"] and not r["arms"][b]["correct"] for r in rows)
        paired[f"{a}->{b}"]={"rescues":rescue,"regressions":regress,
                             "net":rescue-regress,"two_sided_exact_p":_p(rescue,regress)}
    net_ad=paired["base_native->adapter_native"]
    nets=paired["base_native->base_serializer"]
    positive=(net_ad["net"]>=GATE["min_net_adapter_native"] and
              net_ad["regressions"]<=GATE["max_adapter_native_regressions"]) or (
              nets["net"]>=GATE["min_net_serializer"] and
              nets["regressions"]<=GATE["max_serializer_regressions"])
    return {"status":"COMPLETE_PUBLIC_DIAGNOSTIC_2X2_NO_PROMOTION",
            "protocol":PROTOCOL,"source_sha":next(iter(source)),
            "n":N,"total_arms":4,"physical_shards":len(shards),
            "base_gguf_sha256":BASE_SHA,"adapter_gguf_sha256":V7_SHA,
            "serializer_sha256":SERIALIZER_SHA,"totals":totals,"paired":paired,
            "interaction_net":(totals["adapter_serializer"]["correct"]-
                               totals["adapter_native"]["correct"]-
                               totals["base_serializer"]["correct"]+
                               totals["base_native"]["correct"]),
            "positive_screen_only":positive,"promotion_allowed":False,
            "aa_variance_not_yet_evaluated":True,
            "not_GAIA":True,"public_validation_not_hidden":True,
            "no_stronger_model":True,"no_training":True,
            "cases":[{"task_sha256":task,"arms":obj["arms"]} for task,obj in sorted(matched.items())]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior-linked",type=Path)
    ap.add_argument("--prior-novel",type=Path)
    ap.add_argument("--prior-confirmed",type=Path)
    ap.add_argument("--prior-48",type=Path)
    ap.add_argument("--source-only",action="store_true")
    ap.add_argument("--frozen",type=Path)
    ap.add_argument("--shard",type=int)
    ap.add_argument("--mode",choices=["base","adapter"])
    ap.add_argument("--inputs",nargs="+",type=Path)
    ap.add_argument("--work",type=Path,default=Path("factorial-work"))
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite immutable result")
    if args.source_only:
        if any(q is None for q in (args.prior_linked,args.prior_novel,args.prior_confirmed,args.prior_48)):
            ap.error("all four prior source manifests mandatory")
        from datasets import load_dataset
        rows=list(islice(load_dataset(original.REPO,split="validation",revision=SOURCE_REV,streaming=True),5000))
        result=freeze_source(rows,
                             json.loads(args.prior_48.read_text()),
                             json.loads(args.prior_linked.read_text()),
                             json.loads(args.prior_novel.read_text()),
                             json.loads(args.prior_confirmed.read_text()))
    elif args.inputs:
        result=aggregate([json.loads(q.read_text()) for q in args.inputs])
        print("FACTORIAL_FULL_PHYSICAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if args.frozen is None or args.shard is None or args.mode is None:
            ap.error("need frozen, shard and mode")
        result=run_shard(json.loads(args.frozen.read_text()),args.shard,args.mode,args.work)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
