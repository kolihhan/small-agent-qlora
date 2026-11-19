"""Prospective NEW public 2Wiki16 original v7 zero-extra-call answer-contract screen.

No new model weights, no GAIA, no runtime changes, no gold in inference.
Compare original strict, frozen last-line, new conservative final-caption rule.
All outputs are predictions from ONE original fixed Qwen3.5-9B v7 agent run.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts import eval_2wiki_serializer_confirm48 as oldsource
from scripts.eval_2wiki16_v7_evidence_prompt_ab import (
    BASE_PROMPT, BASE_SHA, V7_SHA, h, read_observations
)
from scripts.run_hotpot_original_prefix_fork8 import normalize

P="V7-QWEN9B-2WIKI-UNSEEN16-OUTPUT-CONTRACT-V1"
MODEL="qwen3.5-9b"
N=16
SHARDS=4
GATE={"min_net_gain":3,"max_regressions":0,"min_scored":16}
# Gold-blind, question-blind and conservative: anchored explicit final labels.
LABEL=re.compile(r"^(?:(?:the|my)\s+)?(?:final\s+)?answer(?:\s+is)?\s*:\s*",
                 flags=re.I)
LABEL_IS=re.compile(r"^(?:(?:the|my)\s+)?(?:final\s+)?answer\s+is\s+",
                    flags=re.I)


def clean_final_caption(answer):
    old=normalize(answer)
    if not old:
        return old
    new=LABEL.sub("",old,count=1)
    if new==old:
        new=LABEL_IS.sub("",old,count=1)
    if new==old:
        return old
    new=new.strip().strip(chr(34)+chr(39)+"*_").strip()
    return new or old


def freeze(rows,linked,novel,old16,old48,old8,previous_blind):
    if (old48.get("frozen_source_sha256")!="10eb336fcec4bd171331c9922753a057d7252adcb7ec3e19351b66c5b5d6668d"
        or old8.get("protocol")!="2WIKI-PUBLIC-NEW8-V7-GGUF-FACTORIAL-V1"
        or previous_blind.get("cases_sha256")!="a51699dcd2b470b36e162e5145985a6f9099d3254ac7906a0666e6519e6550d5"
        or len(old48.get("cases",[]))!=48 or len(old8.get("cases",[]))!=8
        or len(previous_blind.get("cases",[]))!=16):
        raise ValueError("previous published physical source invalid")
    previous_n=oldsource.N
    try:
        oldsource.N=88
        chosen=oldsource.freeze(rows,linked,novel,old16)["cases"]
    finally:
        oldsource.N=previous_n
    project=lambda c:{k:c[k] for k in ("id","question","files","index_titles")}
    if (len(chosen)!=88
        or h(chosen[:48])!=old48["frozen_source_sha256"]
        or h(chosen[48:56])!=old8["source_sha"]
        or h([project(c) for c in chosen[56:72]])!=previous_blind["cases_sha256"]):
        raise ValueError("ranked old72 source prefix not identical")
    before_ids={c["id"] for c in chosen[:72]}
    before_docs={p for c in chosen[:72] for p in c["support"]}
    new=chosen[72:88]
    if (len(new)!=16 or len({x["id"] for x in new})!=16
        or any(x["id"] in before_ids or set(x["support"])&before_docs for x in new)):
        raise ValueError("new cases overlap previous task or support documents")
    blind=[project(c) for c in new]
    assert not any("answer" in c or "support" in c for c in blind)
    src={"protocol":P,"status":"FROZEN_PRE_MODEL_SOURCE",
         "source_revision":oldsource.REV,"n":N,"shards":SHARDS,
         "source_sha":h(blind),"previous_72_and_original_40_excluded":True,
         "model_base_sha":BASE_SHA,"model_v7_sha":V7_SHA,
         "gate":GATE,"no_gold":True,"cases":blind}
    labels={"protocol":P,"source_sha":src["source_sha"],
            "references":[{"id":c["id"],"answer":c["answer"],
                           "support":sorted(c["support"])} for c in new]}
    return src,labels


def check(src):
    if (src.get("protocol")!=P or src.get("status")!="FROZEN_PRE_MODEL_SOURCE"
        or src.get("n")!=N or src.get("source_sha")!=h(src.get("cases",[]))
        or src.get("model_base_sha")!=BASE_SHA or src.get("model_v7_sha")!=V7_SHA
        or src.get("gate")!=GATE or src.get("no_gold") is not True):
        raise ValueError("frozen source and exact original v7 drift")


def infer(src,shard,work):
    check(src)
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    out=[]
    for item in src["cases"][4*shard:4*shard+4]:
        w=work/sha256(item["id"].encode()).hexdigest()[:16]
        if w.exists():
            raise FileExistsError("refuse to replay previous workspace")
        _materialize_files({"files":item["files"]},w)
        model=OpenAICompatModel(model=MODEL,timeout_s=400,max_new_tokens=512)
        runtime=AgentRuntime(model,[oldsource.LocalSearch(item["index_titles"],item["files"]),ReadTool()],max_steps=12)
        r=runtime.run(BASE_PROMPT+item["question"],w)
        paths,searches=read_observations(r.trace)
        out.append({"id":item["id"],"answer":r.answer,
                    "completed":r.completed,"stop_reason":r.stop_reason,
                    "tool_calls":r.metrics.tool_calls,"duplicate_blocks":r.metrics.duplicate_calls_blocked,
                    "tool_errors":r.metrics.tool_errors,"read_paths":paths,
                    "successful_searches":searches})
        print("NEW16_V7_ORIGINAL_OUTPUT",shard,len(out),r.completed,r.stop_reason,flush=True)
    return {"protocol":P,"status":"ORIGINAL_V7_PREDICTIONS_SEALED",
            "source_sha":src["source_sha"],"shard":shard,"n":len(out),
            "model_base_sha":BASE_SHA,"model_v7_sha":V7_SHA,
            "no_reference_loaded":True,"cases":out}


def score(src,labels,parts):
    check(src)
    if (len(parts)!=SHARDS or sorted(p["shard"] for p in parts)!=list(range(SHARDS))
        or labels.get("source_sha")!=src["source_sha"]):
        raise ValueError("missing shards or wrong labels")
    rows=[r for p in parts for r in p["cases"]]
    gold={r["id"]:r for r in labels["references"]}
    if (len(rows)!=N or len(gold)!=N or set(gold)!={r["id"] for r in rows}
        or any(p["status"]!="ORIGINAL_V7_PREDICTIONS_SEALED" or p["n"]!=4
            or p["no_reference_loaded"] is not True
            or p["source_sha"]!=src["source_sha"]
            or p["model_base_sha"]!=BASE_SHA or p["model_v7_sha"]!=V7_SHA for p in parts)):
        raise ValueError("incomplete source-matched original Qwen results")
    metrics=Counter()
    evaluated=[]
    for r in rows:
        ref=gold[r["id"]]
        flags={"raw":bool(r["completed"] and gaia_score(r["answer"],ref["answer"])),
               "old_lastline":bool(r["completed"] and gaia_score(normalize(r["answer"]),ref["answer"])),
               "new_caption":bool(r["completed"] and gaia_score(clean_final_caption(r["answer"]),ref["answer"]))}
        for label,yes in flags.items():
            metrics[label]+=int(yes)
        metrics["rescues"]+=int(not flags["old_lastline"] and flags["new_caption"])
        metrics["regressions"]+=int(flags["old_lastline"] and not flags["new_caption"])
        metrics["completed"]+=int(r["completed"])
        metrics["tool_calls"]+=int(r["tool_calls"])
        metrics["duplicate_blocks"]+=int(r["duplicate_blocks"])
        metrics["support_docs_read"]+=len(set(r["read_paths"])&set(ref["support"]))
        evaluated.append({"id":r["id"],"correct":flags,
                          "stop_reason":r["stop_reason"],
                          "support_docs_read":len(set(r["read_paths"])&set(ref["support"]))})
    net=metrics["rescues"]-metrics["regressions"]
    positive=(net>=GATE["min_net_gain"]
              and metrics["regressions"]<=GATE["max_regressions"])
    return {"protocol":P,"status":"FULL_PUBLIC_NEW16_ZERO_CALL_OUTPUT_SCREEN",
            "n":N,"source_sha":src["source_sha"],"metrics":dict(metrics),
            "net_cleaner":net,"preregistered_gate":GATE,
            "positive_screen_only":positive,"promotion_allowed":False,
            "same_original_Qwen9B_v7":True,"zero_extra_model_calls":True,
            "runtime_unchanged":True,"no_training":True,"not_GAIA":True,
            "public_pretraining_overlap_unknown":True,"cases":evaluated}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["freeze","infer","score"])
    for key in ("prior_linked","prior_novel","prior16","prior48","prior8",
                "prior_blind","blind","labels","out","work"):
        p.add_argument("--"+key.replace("_","-"),dest=key,type=Path)
    p.add_argument("--parts",type=Path,nargs="+")
    p.add_argument("--shard",type=int)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("immutable output")
    if a.mode=="freeze":
        from datasets import load_dataset
        rows=list(islice(load_dataset(oldsource.REPO,split="validation",
                                     revision=oldsource.REV,streaming=True),5000))
        inputs=[json.loads(getattr(a,k).read_text()) for k in
                ("prior_linked","prior_novel","prior16","prior48","prior8","prior_blind")]
        src,ref=freeze(rows,*inputs)
        if a.labels.exists():
            raise FileExistsError("reference output already exists")
        a.labels.write_text(json.dumps(ref,indent=2,ensure_ascii=False)+"\n")
        result=src
    elif a.mode=="infer":
        result=infer(json.loads(a.blind.read_text()),a.shard,a.work)
    else:
        result=score(json.loads(a.blind.read_text()),
                     json.loads(a.labels.read_text()),
                     [json.loads(q.read_text()) for q in a.parts])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("V7_UNSEEN16_ZERO_CALL_OUTPUT",json.dumps({
        k:v for k,v in result.items() if k!="cases"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
