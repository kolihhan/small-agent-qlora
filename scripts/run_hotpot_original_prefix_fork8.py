"""Original-Qwen same-prefix tool fork on eight NEW HotpotQA natural tasks.

Freeze source and labels in separate artifacts. Original public Hotpot8+16
question IDs and annotated support articles are excluded. No GAIA, training,
new adapter, stronger teacher, or unseen-source-generalization claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn
from gaia_small_agent.benchmark.fork_probe import _prefix_digest, run_pair
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.training.trajectories import _materialize_files
from gaia_small_agent.tools import ReadTool
from scripts.eval_hotpot_serializer_new16 import freeze as freeze16, REPO, REV, MODEL, DIGEST
from scripts.eval_hotpot_source_shift import LocalSearch

PROTOCOL = "hotpot-original-qwen9b-first-read-fork8-20261119-v1"
N, SHARDS = 8, 4
OLD8_SHA = "47f52b27dc4c3a8afddabf468bac6912325ee817ddd21bbb2caffe3bf03acb5d"
OLD16_SHA = "342d70381cd48dc81026fad320fb963184150be278a2d3eb9a034c863d2955b9"
GATE = {"min_forked": 6, "net_correct_min": 2, "max_regressions": 1}


def digest(x):
    return sha256(json.dumps(x, ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def freeze_new(rows, old8, old16):
    if (old8.get("cases_sha256") != OLD8_SHA or old8.get("n")!=8
        or old16.get("new_source_sha") != OLD16_SHA or old16.get("n")!=16):
        raise ValueError("historical Hotpot8/16 immutable source mismatch")
    exposed_ids={x["id"] for x in old8["cases"]+old16["cases"]}
    exposed_support={name for x in old8["cases"]+old16["cases"]
                     for name in x["support"]}
    from scripts.eval_hotpot_source_shift import page_path
    eligible=[]
    for row in rows:
        supports={page_path(str(t)) for t in
                  (row.get("supporting_facts") or {}).get("title",[])}
        if row.get("id") in exposed_ids or supports & exposed_support:
            continue
        eligible.append(row)
    old_selector=freeze16(eligible,old8)
    chosen=[]
    counts=Counter()
    for row in old_selector["cases"]:
        if counts[row["type"]]<4:
            chosen.append(row)
            counts[row["type"]]+=1
    if len(chosen)!=N or set(counts)!={"bridge","comparison"}:
        raise ValueError("source freeze missing 4 bridge/4 comparison")
    blind=[]
    refs=[]
    for row in chosen:
        blind.append({k:row[k] for k in ("id","type","question","files","index_titles")})
        refs.append({"id":row["id"],"answer":row["answer"],
                     "support_paths":sorted(row["support"])})
    if any("answer" in c or "support" in c for c in blind):
        raise ValueError("label contamination in model source")
    frozen={"status":"FROZEN_BEFORE_MODEL","protocol":PROTOCOL,
            "source_revision":REV,"model_digest":DIGEST,"n":N,
            "prior_question_exclusions":24,"prior_support_excluded":True,
            "cases":blind,"cases_sha256":digest(blind)}
    gold={"protocol":PROTOCOL,"cases_sha256":frozen["cases_sha256"],
          "references":refs,"n":N}
    return frozen,gold


def tools(case):
    return [LocalSearch(case["index_titles"],case["files"]),ReadTool()]


def prompt(q):
    return ("Answer using the local article search and read tools. Search for "
            "relevant article titles, read source paths, then decide whether "
            "evidence suffices. Some articles are distractors. "
            "Return only the answer.\n\nQuestion: "+q)


class Recorder:
    def __init__(self,model):
        self.base=model
        self.turns=[]
    def complete(self,messages,tools):
        t=self.base.complete(messages,tools)
        self.turns.append(deepcopy(t))
        return t
    def complete_final(self,messages,tools):
        final=getattr(self.base,"complete_final",None)
        t=final(messages,tools) if callable(final) else self.base.complete(messages,[])
        self.turns.append(deepcopy(t))
        return t


def first_read(trace):
    calls={}
    for ev in trace:
        if ev.kind=="tool_call":
            calls[ev.data.get("id")]=(ev.step,ev.data.get("name"),
                                      ev.data.get("arguments") or {})
        if ev.kind=="tool_result" and ev.data.get("ok"):
            src=calls.get(ev.data.get("id"))
            if src is not None and src[1]=="read":
                return src[0]
    return None


def read_paths(trace,upto):
    calls={}
    good=[]
    for ev in trace:
        if ev.step>upto:continue
        if ev.kind=="tool_call" and ev.data.get("name")=="read":
            calls[ev.data.get("id")]=(ev.data.get("arguments") or {}).get("source")
        elif ev.kind=="tool_result" and ev.data.get("ok") and ev.data.get("id") in calls:
            good.append(str(calls[ev.data["id"]]))
    return sorted(set(good))


def check_blind(d):
    if (d.get("protocol")!=PROTOCOL or d.get("n")!=N or
        d.get("status")!="FROZEN_BEFORE_MODEL" or d.get("model_digest")!=DIGEST
        or d.get("source_revision")!=REV or digest(d.get("cases"))!=d.get("cases_sha256")
        or len({c["id"] for c in d["cases"]})!=N):
        raise ValueError("frozen source/model mismatch")


def infer(blind,shard,work):
    check_blind(blind)
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    with urlopen("http://127.0.0.1:11434/api/tags",timeout=15) as resp:
        versions=json.load(resp)["models"]
    if not any(x["name"]==MODEL and x["digest"]==DIGEST for x in versions):
        raise ValueError("wrong physical original Qwen9B model digest")
    outcomes=[]
    for case in blind["cases"][shard*2:shard*2+2]:
        case_root=work/sha256(case["id"].encode()).hexdigest()[:16]
        original=case_root/"original"
        a,b=case_root/"answer_now",case_root/"continue"
        for path in (original,a,b):
            if path.exists():raise FileExistsError("overwrite of physical source workspace")
            _materialize_files({"files":case["files"]},path)
        recorder=Recorder(OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=512))
        baseline=AgentRuntime(recorder,tools(case),max_steps=12).run(prompt(case["question"]),original)
        row={"id":case["id"],"type":case["type"],
             "baseline_answer":baseline.answer,
             "baseline_completed":baseline.completed,
             "baseline_tools":baseline.metrics.tool_calls,
             "baseline_stop":baseline.stop_reason}
        step=first_read(baseline.trace)
        if step is None or step>=12 or len(recorder.turns)<step:
            row["status"]="NO_VALID_READ_PREFIX"
        else:
            paired=run_pair(
                prompt(case["question"]),source_id=PROTOCOL+":"+case["id"],
                prefix=recorder.turns[:step],
                model_factory=lambda:OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=512),
                tools_factory=lambda:tools(case),
                answer_workspace=a,continue_workspace=b,max_steps=12)
            if paired["prefix_tool_observation_sha256"]!=_prefix_digest(baseline.trace,step):
                raise RuntimeError("original and paired tool observations diverged")
            row.update(status="MATCHED_SAME_PREFIX",
                       fork=paired,first_read_step=step,
                       pre_fork_read_paths=read_paths(baseline.trace,step))
        outcomes.append(row)
        print("HOTPOT_PREFIX_FORK",case["id"],row["status"],flush=True)
    return {"protocol":PROTOCOL,"status":"SEALED_BEFORE_GOLD","n":len(outcomes),
            "shard":shard,"model_digest":DIGEST,
            "cases_sha256":blind["cases_sha256"],"no_gold_accessed":True,
            "cases":outcomes}


def normalize(text):
    # Gold-blind, last-line-only removal of answer prefixes/markdown.
    s=str(text or "").strip()
    if not s:return s
    s=s.splitlines()[-1].strip()
    s=re.sub(r"(?i)^(?:final answer|answer)\s*:\s*","",s)
    s=re.sub(r"^\s*(?:[-*]\s+|#+\s+)","",s).strip()
    s=s.strip(" _*")
    return s.removesuffix(".").strip()


def score(shards,blind,refs):
    check_blind(blind)
    if (len(shards)!=SHARDS or sorted(x.get("shard") for x in shards)!=list(range(SHARDS))
        or refs.get("cases_sha256")!=blind["cases_sha256"] or
        refs.get("protocol")!=PROTOCOL):
        raise ValueError("missing or misaligned original source data")
    case_rows=[r for s in shards for r in s["cases"]]
    if (len(case_rows)!=N or len({r["id"] for r in case_rows})!=N
        or any(s.get("status")!="SEALED_BEFORE_GOLD" or
               s.get("no_gold_accessed") is not True or
               s.get("cases_sha256")!=blind["cases_sha256"] or
               s.get("model_digest")!=DIGEST for s in shards)):
        raise ValueError("not eight fully paired original Qwen source tasks")
    ref={r["id"]:r for r in refs["references"]}
    if set(ref)!={r["id"] for r in case_rows}:
        raise ValueError("gold identities differ from sealed model cases")
    agg=Counter()
    scored=[]
    for row in case_rows:
        label=ref[row["id"]]
        n_support=len(set(row.get("pre_fork_read_paths",[])) & set(label["support_paths"]))
        detail={"id":row["id"],"type":row["type"],"status":row["status"],
                "support_docs_read_pre_fork":n_support}
        if row["status"]=="MATCHED_SAME_PREFIX":
            a=row["fork"]["arms"]["answer_now"]
            b=row["fork"]["arms"]["continue_tools"]
            ar=bool(a["completed"] and gaia_score(a["answer"],label["answer"]))
            br=bool(b["completed"] and gaia_score(b["answer"],label["answer"]))
            an=bool(a["completed"] and gaia_score(normalize(a["answer"]),label["answer"]))
            bn=bool(b["completed"] and gaia_score(normalize(b["answer"]),label["answer"]))
            detail.update(raw_answer_now=ar,raw_continue=br,
                          normalized_answer_now=an,normalized_continue=bn,
                          added_tools=b["postfork_tool_calls"])
            agg["paired"]+=1
            for k,v in (("raw_answer_now",ar),("raw_continue",br),
                        ("normalized_answer_now",an),("normalized_continue",bn)):
                agg[k]+=int(v)
            agg["rescues"]+=int(not an and bn)
            agg["regressions"]+=int(an and not bn)
            agg["extra_tool_calls"]+=b["postfork_tool_calls"]
            agg[f"supports_read_pre_fork_{n_support}"]+=1
        else:
            agg["no_prefix"]+=1
        scored.append(detail)
    return {"protocol":PROTOCOL,"status":"COMPLETE_PHYSICAL_NATURAL_HOTPOT8_FORK",
            "source_sha256":blind["cases_sha256"],
            "n":N,"totals":dict(agg),"gate":GATE,
            "gate_pass":bool(agg["paired"]>=GATE["min_forked"] and
                             agg["rescues"]-agg["regressions"]>=GATE["net_correct_min"] and
                             agg["regressions"]<=GATE["max_regressions"]),
            "no_GAIA":True,"no_training":True,"no_adapter":True,
            "public_pretraining_overlap_unknown":True,
            "not_independent_hidden_holdout":True,
            "limitations":"New question/support-disjoint public HotpotQA validation cases; local search/read proxy, conditional first-read fork, source labels read only after both branch predictions sealed.",
            "cases":scored}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score"))
    for arg in ("previous8","previous16","blind","refs","out","work"):
        p.add_argument("--"+arg,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--shards",type=Path,nargs="+")
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable artifact already exists")
    if a.mode=="freeze":
        from datasets import load_dataset
        rows=list(islice(load_dataset(REPO,"distractor",split="validation",
                                     revision=REV,streaming=True),4000))
        blind,refs=freeze_new(rows,
                              json.loads(a.previous8.read_text()),
                              json.loads(a.previous16.read_text()))
        if a.refs.exists():raise FileExistsError("gold source already exists")
        a.refs.write_text(json.dumps(refs,ensure_ascii=False,indent=2)+"\n")
        result=blind
    elif a.mode=="infer":
        result=infer(json.loads(a.blind.read_text()),a.shard,a.work)
    else:
        result=score([json.loads(x.read_text()) for x in a.shards],
                     json.loads(a.blind.read_text()),
                     json.loads(a.refs.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("NATURAL_HOTPOT8_FIRST_READ_FORK",json.dumps({
        k:v for k,v in result.items() if k not in ("cases","references")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
