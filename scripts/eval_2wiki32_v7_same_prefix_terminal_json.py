"""Fixed Qwen3.5-9B/v7 prospective NEW 2Wiki32 last-step JSON answer experiment.

Same authentic upstream tool trajectory for BOTH terminal-answer decoders:
original final vs strict JSON-schema final. One extra model call ONLY when
original AgentRuntime reaches final-only step12. Nothing else in runtime,
prompt, tools, model, weights, sources, or budgets changes. Compare accuracy
and errors, not mere finalization. Public non-GAIA, original source frozen
and publisher labels excluded from inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import requests

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts import eval_2wiki_serializer_confirm48 as source48
from scripts.eval_2wiki16_v7_evidence_prompt_ab import BASE_PROMPT, BASE_SHA, V7_SHA, h, read_observations
from scripts.run_hotpot_original_prefix_fork8 import normalize

PROTOCOL="FIXED-QWEN35-V7-2WIKI-NEW32-LAST-STEP-JSON-CONTROL-V1"
MODEL="qwen3.5-9b"
N=32
SHARDS=8
PER_SHARD=4
GATE={"min_net_correct":3,"max_regressions":1,
      "min_terminal_forks":3,"require_n":32}
PREVIOUS88_SHA="79b2413cec3c6ecf317e2dc9157c9f64f4d8f572003a82ebe6d0f079572fd47a"

ANSWER_SCHEMA={"type":"object",
 "properties":{"answer":{"type":"string"}},
 "required":["answer"],"additionalProperties":False}


def freeze(rows, linked, novel, old16, old48, old8, old72, old88):
    if (old48.get("frozen_source_sha256")!="10eb336fcec4bd171331c9922753a057d7252adcb7ec3e19351b66c5b5d6668d"
        or old8.get("protocol")!="2WIKI-PUBLIC-NEW8-V7-GGUF-FACTORIAL-V1"
        or old72.get("cases_sha256")!="a51699dcd2b470b36e162e5145985a6f9099d3254ac7906a0666e6519e6550d5"
        or old88.get("source_sha")!=PREVIOUS88_SHA
        or len(old48.get("cases",[]))!=48 or len(old8.get("cases",[]))!=8
        or len(old72.get("cases",[]))!=16 or len(old88.get("cases",[]))!=16):
        raise ValueError("pinned original source histories do not match")
    original_n=source48.N
    try:
        source48.N=120
        rows120=source48.freeze(rows,linked,novel,old16)["cases"]
    finally:
        source48.N=original_n
    project=lambda x:{k:x[k] for k in ("id","question","files","index_titles")}
    if (len(rows120)!=120
        or h(rows120[:48])!=old48["frozen_source_sha256"]
        or h(rows120[48:56])!=old8["source_sha"]
        or h([project(x) for x in rows120[56:72]])!=old72["cases_sha256"]
        or h([project(x) for x in rows120[72:88]])!=PREVIOUS88_SHA):
        raise ValueError("old-ranked full source prefix changed")
    used_ids={r["id"] for r in rows120[:88]}
    used_docs={s for r in rows120[:88] for s in r["support"]}
    new=rows120[88:120]
    if (len(new)!=N or len({r["id"] for r in new})!=N
        or any(r["id"] in used_ids or set(r["support"])&used_docs for r in new)):
        raise ValueError("new public tasks/support docs intersect previously scored sources")
    blind=[project(x) for x in new]
    if any("answer" in r or "support" in r for r in blind):
        raise ValueError("publisher gold / support ground truth leaked")
    source={"protocol":PROTOCOL,"status":"NEW32_SOURCE_FROZEN_BEFORE_MODEL",
            "source_rev":source48.REV,"n":N,"shards":SHARDS,
            "source_sha":h(blind),"model_base_sha":BASE_SHA,"v7_adapter_sha":V7_SHA,
            "previous_88_disjoint_task_and_support":True,"gate":GATE,
            "no_publisher_gold_for_model":True,"cases":blind}
    gold={"protocol":PROTOCOL,"source_sha":source["source_sha"],
          "references":[{"id":x["id"],"answer":x["answer"],
                         "support_paths":sorted(x["support"])} for x in new]}
    return source,gold


def check(src):
    if (src.get("protocol")!=PROTOCOL or
        src.get("status")!="NEW32_SOURCE_FROZEN_BEFORE_MODEL" or
        src.get("n")!=N or src.get("shards")!=SHARDS or
        src.get("source_sha")!=h(src.get("cases",[])) or
        src.get("model_base_sha")!=BASE_SHA or src.get("v7_adapter_sha")!=V7_SHA or
        src.get("gate")!=GATE or src.get("no_publisher_gold_for_model") is not True or
        len(src.get("cases",[]))!=N):
        raise ValueError("wrong frozen source or original GGUF/LoRA")


def json_terminal_request(model:OpenAICompatModel, messages:list[dict]) -> tuple[str,str]:
    # Explicit standalone same-base model call, identical messages; only
    # constrained output format changes. No extra tools, snippets or examples.
    payload={
        "model":model.model,"messages":messages,
        "max_tokens":model.max_new_tokens,
        "temperature":0,
        "response_format":{"type":"json_schema","json_schema":{
            "name":"final_answer","strict":True,"schema":ANSWER_SCHEMA
        }},
    }
    try:
        resp=requests.post(model.base_url+"/v1/chat/completions",
                           json=payload,timeout=model.timeout_s)
        resp.raise_for_status()
        raw=resp.json()["choices"][0]["message"]
        text=str(raw.get("content") or "").strip()
        obj=json.loads(text)
        if (set(obj)!={"answer"} or not isinstance(obj["answer"],str)
            or not obj["answer"].strip()):
            return "", "invalid_json_answer"
        if str(raw.get("tool_calls") or "").strip() not in ("","[]"):
            return "", "unexpected_tool_call"
        return obj["answer"].strip(),"valid_json_answer"
    except (requests.RequestException,ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
        return "", "structured_request_error_"+type(exc).__name__


class PairedLastStepModel:
    def __init__(self,original:OpenAICompatModel,structured_call=json_terminal_request):
        self.original=original
        self.structured_call=structured_call
        self.orig_terminal=None
        self.structured_answer=""
        self.structured_status="not_reached"
        self.terminal_fork_count=0

    def complete(self,messages,tools):
        return self.original.complete(messages,tools)

    def complete_final(self,messages,tools):
        # On exactly the SAME prefix, capture original step12 answer, then
        # separately issue one constrained JSON answer request. Return only
        # original turn to the unchanged AgentRuntime.
        if self.terminal_fork_count!=0:
            raise RuntimeError("unexpected multiple final-only model calls")
        self.terminal_fork_count=1
        original=self.original.complete_final(messages,tools)
        self.orig_terminal={
            "answer_text":original.content,
            "had_structured_tool_calls":bool(original.tool_calls)
        }
        answer,status=self.structured_call(self.original,messages)
        self.structured_answer,self.structured_status=answer,status
        return original


def infer(source,shard,work):
    check(source)
    if shard not in range(SHARDS):raise ValueError("invalid 4-case worker")
    out=[]
    for item in source["cases"][PER_SHARD*shard:PER_SHARD*(shard+1)]:
        w=work/sha256(item["id"].encode()).hexdigest()[:16]
        if w.exists():raise FileExistsError("do not overwrite original workspace")
        _materialize_files({"files":item["files"]},w)
        model=PairedLastStepModel(OpenAICompatModel(model=MODEL,timeout_s=400,max_new_tokens=512))
        actual=AgentRuntime(model,[source48.LocalSearch(item["index_titles"],item["files"]),ReadTool()],
                            max_steps=12).run(BASE_PROMPT+item["question"],w)
        read_paths,_=read_observations(actual.trace)
        out.append({"id":item["id"],"original_answer":actual.answer,
                    "original_completed":actual.completed,"original_stop":actual.stop_reason,
                    "structured_answer":model.structured_answer,
                    "structured_status":model.structured_status,
                    "final_forks":model.terminal_fork_count,
                    "original_last_turn":model.orig_terminal,
                    "model_calls_added":model.terminal_fork_count,
                    "tool_calls":actual.metrics.tool_calls,
                    "duplicate_blocks":actual.metrics.duplicate_calls_blocked,
                    "successful_read_paths":read_paths})
        print("QWEN_V7_SAME_PREFIX_TERMINAL32",shard,len(out),
              actual.stop_reason,model.structured_status,flush=True)
    return {"protocol":PROTOCOL,"status":"GOLD_BLIND_ALL_ORIGINAL_AND_JSON_FINALS_SEALED",
            "source_sha":source["source_sha"],"shard":shard,
            "n":len(out),"model_base_sha":BASE_SHA,"adapter_sha":V7_SHA,
            "gold_not_loaded":True,"cases":out}


def aggregate(source,labels,parts):
    check(source)
    if (len(parts)!=SHARDS or sorted(p.get("shard") for p in parts)!=list(range(SHARDS))
        or labels.get("protocol")!=PROTOCOL or labels.get("source_sha")!=source["source_sha"]):
        raise ValueError("source/publisher label mismatch")
    records=[x for p in parts for x in p["cases"]]
    refs={x["id"]:x for x in labels["references"]}
    if (len(records)!=N or len(refs)!=N or
        set(refs)!={x["id"] for x in records} or
        any(p.get("status")!="GOLD_BLIND_ALL_ORIGINAL_AND_JSON_FINALS_SEALED" or
            p.get("source_sha")!=source["source_sha"] or
            p.get("model_base_sha")!=BASE_SHA or
            p.get("adapter_sha")!=V7_SHA or p.get("gold_not_loaded") is not True or
            p.get("n")!=PER_SHARD for p in parts)):
        raise ValueError("not all 32 original physically matched final prefix")
    counts=Counter();rows=[]
    for x in records:
        ref=refs[x["id"]]
        base=bool(x["original_completed"] and
                  gaia_score(normalize(x["original_answer"]),ref["answer"]))
        if x["final_forks"]:
            candidate=x["structured_answer"]
            modified=bool(x["structured_status"]=="valid_json_answer"
                          and gaia_score(normalize(candidate),ref["answer"]))
        else:
            modified=base  # no intervention before forced terminal
        counts["baseline_correct"]+=int(base)
        counts["structured_correct"]+=int(modified)
        counts["rescues"]+=int(not base and modified)
        counts["regressions"]+=int(base and not modified)
        counts["terminal_forks"]+=int(x["final_forks"])
        counts["original_completions"]+=int(x["original_completed"])
        counts["new_terminal_valid"]+=int(x["structured_status"]=="valid_json_answer")
        counts["tool_calls"]+=int(x["tool_calls"])
        counts["duplicate_blocks"]+=int(x["duplicate_blocks"])
        counts["support_reads"]+=len(set(x["successful_read_paths"])&set(ref["support_paths"]))
        counts["extra_calls"]+=x["model_calls_added"]
        rows.append({"id":x["id"],"base_correct":base,"json_correct":modified,
                     "fork":bool(x["final_forks"]),
                     "original_stop":x["original_stop"],
                     "json_status":x["structured_status"]})
    net=counts["rescues"]-counts["regressions"]
    positive=(counts["terminal_forks"]>=GATE["min_terminal_forks"]
              and net>=GATE["min_net_correct"]
              and counts["regressions"]<=GATE["max_regressions"])
    return {"protocol":PROTOCOL,"status":"COMPLETE_REAL_NEW32_FIXED_PREFIX_V7_JSON_AB",
            "n":N,"source_sha":source["source_sha"],
            "counts":dict(counts),"net_correct":net,
            "predeclared_gate":GATE,"positive_screen_only":positive,
            "promotion_allowed":False,"no_QLoRA_training":True,
            "no_GAIA":True,"same_original_base_v7":True,
            "upstream_agent_runtime_and_tools_unchanged":True,
            "public_source_pretraining_overlap_unknown":True,"cases":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("freeze","infer","score"))
    for key in ("prior_linked","prior_novel","prior16","prior48","prior8","prior72",
                "prior88","source","labels","work","out"):
        parser.add_argument("--"+key.replace("_","-"),dest=key,type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--parts",nargs="+",type=Path)
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError("immutable output cannot be overwritten")
    if a.mode=="freeze":
        from datasets import load_dataset
        raw=list(islice(load_dataset(source48.REPO,split="validation",
                                    revision=source48.REV,streaming=True),5000))
        docs=[json.loads(getattr(a,k).read_text()) for k in
              ("prior_linked","prior_novel","prior16","prior48","prior8","prior72","prior88")]
        source,labels=freeze(raw,*docs)
        if a.labels.exists():raise FileExistsError("reference labels already saved")
        a.labels.write_text(json.dumps(labels,indent=2,ensure_ascii=False)+"\n")
        report=source
    elif a.mode=="infer":
        report=infer(json.loads(a.source.read_text()),a.shard,a.work)
    else:
        report=aggregate(json.loads(a.source.read_text()),json.loads(a.labels.read_text()),
                         [json.loads(p.read_text()) for p in a.parts])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("FIXED_V7_JSON_FORK_NEW32",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
