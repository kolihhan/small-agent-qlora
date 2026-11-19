"""HotpotQA original-Qwen source-backed search-to-read affordance pilot.

Exact same original model, 12-step agent scaffold, local corpus and tools.
Control: existing LocalSearch returns paths/snippets.
Treatment: SAME ranked hits + deterministic read(source=...) tool affordance,
without any publisher answers/support annotations. Gold is separated from
physical inference artifact and retrieved only AFTER model outcomes are sealed.

Fresh 8 HotpotQA question+support identities, excluding prior 8+16+8 source
uses. Pilot on public validation; no model, adapter, GAIA, teacher changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.tools.base import ToolResult
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.eval_hotpot_source_shift import LocalSearch
from scripts.run_hotpot_original_prefix_fork8 import (
    REPO, REV, MODEL, DIGEST, freeze_new, normalize, digest, prompt,
)

PROTOCOL="qwen9b-hotpot-search-read-affordance8-v1"
N=8
SHARDS=4
GATE={"min_net_normalized_correct":2,"max_normalized_regressions":1,
      "min_net_successful_read_tasks":3}
SOURCE_SEED="second-independent-hotpot8-search-to-read-only-20261119"


def freeze(rows,old8,old16,prior,prior_refs):
    if (prior.get("n")!=8 or prior.get("cases_sha256")!=
        "4ecdc54003291f60d05396c163cc691a53168965dc2a0bda4c5312d9be908bf0"
        or prior_refs.get("cases_sha256")!=prior["cases_sha256"]):
        raise ValueError("previous natural Hotpot8 original source identity drift")
    used_ids={x["id"] for x in prior["cases"]}
    support={p for x in prior_refs["references"] for p in x["support_paths"]}
    from scripts.eval_hotpot_source_shift import page_path
    candidates=[]
    for row in rows:
        if row.get("id") in used_ids:
            continue
        paths={page_path(str(t)) for t in
               (row.get("supporting_facts") or {}).get("title",[])}
        if paths & support:
            continue
        candidates.append(row)
    blind,ref=freeze_new(candidates,old8,old16)
    blind["protocol"]=PROTOCOL
    blind["prior_32_questions_excluded"]=True
    blind["previous_blind_sha256"]=prior["cases_sha256"]
    blind["intervention"]="append original search hit path as read JSON affordance only"
    ref["protocol"]=PROTOCOL
    ref["previous_blind_sha256"]=prior["cases_sha256"]
    return blind,ref


class AffordanceSearch(LocalSearch):
    """Only adds a hint derived from already-ranked original search result."""
    def run(self,arguments,workspace):
        original=super().run(arguments,workspace)
        if not original.ok:
            return original
        matches=re.findall(r"\|\s*(pages/[0-9a-f]{12}\.txt)\s*\|",original.content)
        if not matches:
            return original
        # Source metadata, not a gold/support answer; no hidden results added.
        hint=("The results above are ONLY snippets. To verify a fact, open the "
              "original file with the available read tool, for example: "
              'read({"source": "'+matches[0]+'"}).')
        return ToolResult(True,original.content+"\n"+hint,
                          original.error_code,original.metadata)


def tools(case,arm):
    search=(AffordanceSearch if arm=="affordance" else LocalSearch)(
        case["index_titles"],case["files"])
    return [search,ReadTool()]


def observed(trace):
    names={}
    reads=set()
    search_ok=0
    for e in trace:
        if e.kind=="tool_call":
            names[e.data.get("id")]=(e.data.get("name"),
                                      (e.data.get("arguments") or {}).get("source"))
        if e.kind=="tool_result" and e.data.get("ok"):
            k,source=names.get(e.data.get("id"),(None,None))
            if k=="read":
                reads.add(str(source))
            if k=="search":
                search_ok+=1
    return sorted(reads),search_ok


def check(blind):
    if (blind.get("protocol")!=PROTOCOL or blind.get("n")!=N or
        blind.get("model_digest")!=DIGEST or
        blind.get("prior_32_questions_excluded") is not True or
        digest(blind.get("cases"))!=blind.get("cases_sha256")):
        raise ValueError("source+model SHA/gate differs")


def physical(blind,shard,work):
    check(blind)
    if shard not in range(SHARDS):raise ValueError("unexpected shard")
    with urlopen("http://127.0.0.1:11434/api/tags",timeout=15) as stream:
        tags=json.load(stream)["models"]
    if not any(x["name"]==MODEL and x["digest"]==DIGEST for x in tags):
        raise ValueError("physical original Qwen digest mismatch")
    cases=[]
    for offset,case in enumerate(blind["cases"][shard*2:shard*2+2]):
        order=("original","affordance") if (shard+offset)%2==0 else ("affordance","original")
        results={}
        for arm in order:
            dest=work/sha256(case["id"].encode()).hexdigest()[:16]/arm
            if dest.exists():raise FileExistsError("model evidence workspace already exists")
            _materialize_files({"files":case["files"]},dest)
            agent=AgentRuntime(
                OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=512),
                tools(case,arm),max_steps=12)
            result=agent.run(prompt(case["question"]),dest)
            paths,search_count=observed(result.trace)
            results[arm]={
                "answer":result.answer,"completed":result.completed,
                "stop_reason":result.stop_reason,
                "successful_read_paths":paths,"successful_search_calls":search_count,
                "steps":result.metrics.steps,
                "tool_calls":result.metrics.tool_calls,
                "tool_errors":result.metrics.tool_errors,
            }
        cases.append({"id":case["id"],"type":case["type"],"arms":results})
        print("HOTPOT_TOOL_AFFORDANCE_PHYSICAL",case["id"],
              {a:(len(results[a]["successful_read_paths"]),results[a]["completed"])
               for a in ("original","affordance")},flush=True)
    return {"protocol":PROTOCOL,"status":"BOTH_ARMS_SEALED_GOLD_UNAVAILABLE",
            "n":len(cases),"shard":shard,"cases_sha256":blind["cases_sha256"],
            "model_digest":DIGEST,"gold_never_loaded":True,"cases":cases}


def score(shards,blind,ref):
    check(blind)
    if (len(shards)!=SHARDS or sorted(s["shard"] for s in shards)!=list(range(SHARDS))
        or ref.get("protocol")!=PROTOCOL or
        ref.get("cases_sha256")!=blind["cases_sha256"]):
        raise ValueError("missing or changing independent model source")
    rows=[x for s in shards for x in s["cases"]]
    if (len(rows)!=N or len({r["id"] for r in rows})!=N or
        any(s.get("status")!="BOTH_ARMS_SEALED_GOLD_UNAVAILABLE" or
            s.get("model_digest")!=DIGEST or s.get("cases_sha256")!=blind["cases_sha256"]
            or s.get("gold_never_loaded") is not True for s in shards)):
        raise ValueError("source/paired inference mismatch")
    gold={r["id"]:r for r in ref["references"]}
    if set(gold)!={r["id"] for r in rows}:
        raise ValueError("reference task ID mismatch")
    tally=Counter()
    details=[]
    for row in rows:
        a=row["arms"]["original"]
        b=row["arms"]["affordance"]
        g=gold[row["id"]]
        raw_a=bool(a["completed"] and gaia_score(a["answer"],g["answer"]))
        raw_b=bool(b["completed"] and gaia_score(b["answer"],g["answer"]))
        norm_a=bool(a["completed"] and gaia_score(normalize(a["answer"]),g["answer"]))
        norm_b=bool(b["completed"] and gaia_score(normalize(b["answer"]),g["answer"]))
        read_a=len(a["successful_read_paths"])>0
        read_b=len(b["successful_read_paths"])>0
        supports=set(g["support_paths"])
        s_a=len(set(a["successful_read_paths"])&supports)
        s_b=len(set(b["successful_read_paths"])&supports)
        for k,v in (("raw_original",raw_a),("raw_affordance",raw_b),
                    ("norm_original",norm_a),("norm_affordance",norm_b),
                    ("read_original",read_a),("read_affordance",read_b),
                    ("support_reads_original",s_a),("support_reads_affordance",s_b),
                    ("total_tool_calls_original",a["tool_calls"]),
                    ("total_tool_calls_affordance",b["tool_calls"]),
                    ("tool_errors_original",a["tool_errors"]),
                    ("tool_errors_affordance",b["tool_errors"])):
            tally[k]+=int(v)
        tally["normalized_rescues"]+=int(not norm_a and norm_b)
        tally["normalized_regressions"]+=int(norm_a and not norm_b)
        tally["new_successful_read_cases"]+=int(not read_a and read_b)
        tally["lost_successful_read_cases"]+=int(read_a and not read_b)
        details.append({"id":row["id"],"type":row["type"],
                        "raw_original":raw_a,"raw_affordance":raw_b,
                        "normalized_original":norm_a,"normalized_affordance":norm_b,
                        "original_support_docs_read":s_a,"affordance_support_docs_read":s_b,
                        "original_successful_read":read_a,"affordance_successful_read":read_b})
    net=tally["normalized_rescues"]-tally["normalized_regressions"]
    readnet=tally["new_successful_read_cases"]-tally["lost_successful_read_cases"]
    passed=(net>=GATE["min_net_normalized_correct"] and
            tally["normalized_regressions"]<=GATE["max_normalized_regressions"] and
            readnet>=GATE["min_net_successful_read_tasks"])
    return {"protocol":PROTOCOL,"status":"COMPLETE_PAIRED_ORIGINAL_QWEN_HOTPOT_TOOL_AFFORDANCE8",
            "n":N,"source_sha256":blind["cases_sha256"],"model_digest":DIGEST,
            "scores":dict(tally),"net_normalized":net,"net_read_activated":readnet,
            "predeclared_gate":GATE,"positive_screen_only":bool(passed),
            "promotion_allowed":False,"no_GAIA":True,"no_qlora":True,
            "no_stronger_teacher":True,
            "limitations":"New question-and-support-disjoint within publicly exposed HotpotQA family; no hidden OOD, one paired seed; raw and locked normalized exact both reported.",
            "cases":details}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score"))
    for f in ("old8","old16","prior","priorrefs","blind","reference","work","out"):
        p.add_argument("--"+f,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--parts",type=Path,nargs="+")
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("refuse overwrite")
    if a.mode=="freeze":
        from datasets import load_dataset
        rows=list(islice(load_dataset(REPO,"distractor",split="validation",
                                     revision=REV,streaming=True),4000))
        blind,ref=freeze(
            rows,json.loads(a.old8.read_text()),json.loads(a.old16.read_text()),
            json.loads(a.prior.read_text()),json.loads(a.priorrefs.read_text()))
        if a.reference.exists():raise FileExistsError("ref already exists")
        a.reference.write_text(json.dumps(ref,ensure_ascii=False,indent=2)+"\n")
        out=blind
    elif a.mode=="infer":
        out=physical(json.loads(a.blind.read_text()),a.shard,a.work)
    else:
        out=score([json.loads(p.read_text()) for p in a.parts],
                  json.loads(a.blind.read_text()),
                  json.loads(a.reference.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("HOTPOT_ORIGINAL_QWEN_AFFORDANCE",json.dumps({
        k:v for k,v in out.items() if k not in ("cases","references")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
