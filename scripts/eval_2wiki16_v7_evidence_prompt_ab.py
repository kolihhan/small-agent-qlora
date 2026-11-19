"""Fixed v7 QLoRA on 16 NEW public 2Wiki multihop questions: evidence checklist vs original.

No AgentRuntime modifications, new LoRA training, strong teacher, GAIA eval, or
publisher answer in inference environment. Paired actual search/read, same
Qwen3.5-9B/v7 GGUF, same 12 steps, alternate arm order. This is an exploratory
non-GAIA TASK FAMILY screen; not a GAIA82 improvement claim.
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
from scripts import eval_2wiki_serializer_confirm48 as source48
from scripts.run_hotpot_original_prefix_fork8 import normalize

PROTOCOL="ORIGINAL-QWEN9B-V7-2WIKI-NEW16-TWO-PROMPTS-EVIDENCE-STATE-20261119-V1"
MODEL="qwen3.5-9b"
BASE_SHA="8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
V7_SHA="475f5fb59a553390a0dde9425d36b42ff85a0345bdd31b8465c22fd4dc02e2"
OLD48_SHA="10eb336fcec4bd171331c9922753a057d7252adcb7ec3e19351b66c5b5d6668d"
N=16
SHARDS=4
GATE={"net_normalized_correct_min":3,"max_regressions":1,
      "min_paired":16,"max_relative_tool_cost":1.5}

BASE_PROMPT=(
    "Answer the following question using the local article search and read tools. "
    "Search for article names and read returned paths before answering. "
    "A local source may be a distractor; decide whether more evidence is needed. "
    "Return only the answer, not reasoning.\n\nQuestion: "
)
TREAT_PROMPT=(
    "Use the local article search and read tools to answer this multi-hop question. "
    "First identify the requested final entity and the intermediate facts/relations "
    "needed to identify it. Search for each missing fact by entity/title; after search "
    "READ relevant original article paths, not only snippets. Check that the article "
    "subject, direction of the relationship and dates/qualifiers match the question. "
    "Do not confuse an intermediate person's name with the final requested entity. "
    "Once the observations support both hops, STOP searching and return ONLY the "
    "short final answer. If evidence is missing, change query/source rather than "
    "repeating the same search. Do not invent facts or use unsupported guesswork.\n\nQuestion: "
)


def h(x):
    return sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def freeze(rows,linked,novel,prior16,prior48,prior8):
    if (prior48.get("frozen_source_sha256")!=OLD48_SHA
        or len(prior48.get("cases",[]))!=48
        or prior8.get("known_prior48_sha")!=OLD48_SHA
        or prior8.get("n")!=8
        or prior8.get("protocol")!="2WIKI-PUBLIC-NEW8-V7-GGUF-FACTORIAL-V1"):
        raise ValueError("previous frozen public 2Wiki48+8 source drift")
    if h(prior48["cases"])!=OLD48_SHA:
        raise ValueError("old 2Wiki48 source SHA mismatch")
    before=source48.N
    try:
        source48.N=72
        chosen=source48.freeze(rows,linked,novel,prior16)["cases"]
    finally:
        source48.N=before
    if len(chosen)!=72 or h(chosen[:48])!=OLD48_SHA:
        raise ValueError("ranked source selection changed before sample 48")
    if h(chosen[48:56])!=prior8["source_sha"]:
        raise ValueError("prior 2Wiki8 unexpected source prefix")
    old_ids={r["id"] for r in chosen[:56]}
    old_support={p for r in chosen[:56] for p in r["support"]}
    new=chosen[56:72]
    if (len(new)!=16 or len({r["id"] for r in new})!=16
        or any(r["id"] in old_ids or set(r["support"])&old_support for r in new)):
        raise ValueError("not 16 new identity/support-disjoint 2Wiki questions")
    blind_cases=[]
    ref=[]
    for c in new:
        blind_cases.append({k:c[k] for k in ("id","question","files","index_titles")})
        ref.append({"id":c["id"],"answer":c["answer"],
                    "support_paths":sorted(c["support"])})
    sha=h(blind_cases)
    blind={"protocol":PROTOCOL,"status":"SOURCE_FROZEN_BEFORE_PHYSICAL_INFERENCE",
           "source_repo":source48.REPO,"source_rev":source48.REV,
           "n":N,"shards":SHARDS,"cases_sha256":sha,
           "original_public_prior56_task_and_annotated_support_excluded":True,
           "original_base_gguf_sha256":BASE_SHA,"existing_v7_adapter_sha256":V7_SHA,
           "gate":GATE,"cases":blind_cases,
           "no_publisher_gold_in_inference":True}
    labels={"protocol":PROTOCOL,"cases_sha256":sha,"n":N,
            "references":ref}
    assert not any("answer" in c or "support" in c for c in blind_cases)
    return blind,labels


def check(blind):
    if (blind.get("protocol")!=PROTOCOL or
        blind.get("status")!="SOURCE_FROZEN_BEFORE_PHYSICAL_INFERENCE"
        or blind.get("n")!=N or blind.get("gate")!=GATE
        or blind.get("cases_sha256")!=h(blind.get("cases"))
        or blind.get("original_base_gguf_sha256")!=BASE_SHA
        or blind.get("existing_v7_adapter_sha256")!=V7_SHA
        or len(set(c["id"] for c in blind["cases"]))!=N):
        raise ValueError("locked source+model mismatch")


def read_observations(trace):
    calls={}
    read_paths=set()
    success_search=0
    for e in trace:
        if e.kind=="tool_call":
            calls[e.data.get("id")]=(e.data.get("name"),
                                      (e.data.get("arguments") or {}).get("source"))
        if e.kind=="tool_result" and e.data.get("ok"):
            name,path=calls.get(e.data.get("id"),(None,None))
            if name=="read":
                read_paths.add(str(path))
            if name=="search":
                success_search+=1
    return sorted(read_paths),success_search


def physical(blind,shard,workspace):
    check(blind)
    if shard not in range(SHARDS):raise ValueError("wrong physical shard")
    rows=[]
    for j,c in enumerate(blind["cases"][shard*4:shard*4+4]):
        order=("control","evidence") if (shard+j)%2==0 else ("evidence","control")
        arms={}
        for a in order:
            work=workspace/sha256(c["id"].encode()).hexdigest()[:16]/a
            if work.exists():raise FileExistsError("source workspace already exists")
            _materialize_files({"files":c["files"]},work)
            prompt=(BASE_PROMPT if a=="control" else TREAT_PROMPT)+c["question"]
            model=OpenAICompatModel(model=MODEL,timeout_s=400,max_new_tokens=512)
            runtime=AgentRuntime(model,[source48.LocalSearch(c["index_titles"],c["files"]),ReadTool()],max_steps=12)
            result=runtime.run(prompt,work)
            paths,ns=read_observations(result.trace)
            arms[a]={"answer":result.answer,"completed":result.completed,
                     "stop_reason":result.stop_reason,"tool_calls":result.metrics.tool_calls,
                     "tool_errors":result.metrics.tool_errors,
                     "duplicate_calls_blocked":result.metrics.duplicate_calls_blocked,
                     "steps":result.metrics.steps,
                     "successful_read_paths":paths,"successful_search_calls":ns}
        rows.append({"id":c["id"],"arms":arms})
        print("REAL_V7_2WIKI_EVIDENCE_AB",shard,len(rows),
              {k:(v["completed"],len(v["successful_read_paths"])) for k,v in arms.items()},flush=True)
    return {"protocol":PROTOCOL,"status":"BOTH_REAL_V7_ARMS_SEALED_BEFORE_REFERENCE",
            "source_sha":blind["cases_sha256"],"shard":shard,"n":len(rows),
            "original_base_gguf_sha256":BASE_SHA,"existing_v7_adapter_sha256":V7_SHA,
            "no_reference_loaded":True,"cases":rows}


def aggregate(shards,blind,labels):
    check(blind)
    if (len(shards)!=SHARDS or
        sorted(s.get("shard") for s in shards)!=list(range(SHARDS)) or
        labels.get("protocol")!=PROTOCOL or labels.get("cases_sha256")!=blind["cases_sha256"]):
        raise ValueError("score must use source-matched fully sealed two-arm model runs")
    rows=[x for s in shards for x in s["cases"]]
    if (len(rows)!=N or len({r["id"] for r in rows})!=N or
        any(s.get("status")!="BOTH_REAL_V7_ARMS_SEALED_BEFORE_REFERENCE"
            or s.get("source_sha")!=blind["cases_sha256"] or
            s.get("original_base_gguf_sha256")!=BASE_SHA or
            s.get("existing_v7_adapter_sha256")!=V7_SHA or
            s.get("no_reference_loaded") is not True or
            s.get("n")!=4 for s in shards)):
        raise ValueError("missing physical originals or wrong GGUF")
    gold={r["id"]:r for r in labels["references"]}
    if len(gold)!=N or set(gold)!={r["id"] for r in rows}:
        raise ValueError("gold source identities drifted")
    ct=Counter();reports=[]
    for row in rows:
        a,b=row["arms"]["control"],row["arms"]["evidence"]
        ref=gold[row["id"]]
        raw_a=bool(a["completed"] and gaia_score(a["answer"],ref["answer"]))
        raw_b=bool(b["completed"] and gaia_score(b["answer"],ref["answer"]))
        norm_a=bool(a["completed"] and gaia_score(normalize(a["answer"]),ref["answer"]))
        norm_b=bool(b["completed"] and gaia_score(normalize(b["answer"]),ref["answer"]))
        supports=set(ref["support_paths"])
        for key,val in (("raw_control",raw_a),("raw_evidence",raw_b),
                        ("normalized_control",norm_a),("normalized_evidence",norm_b),
                        ("completed_control",a["completed"]),("completed_evidence",b["completed"]),
                        ("tool_calls_control",a["tool_calls"]),("tool_calls_evidence",b["tool_calls"]),
                        ("duplicate_control",a["duplicate_calls_blocked"]),
                        ("duplicate_evidence",b["duplicate_calls_blocked"]),
                        ("tool_errors_control",a["tool_errors"]),("tool_errors_evidence",b["tool_errors"]),
                        ("support_docs_read_control",len(set(a["successful_read_paths"])&supports)),
                        ("support_docs_read_evidence",len(set(b["successful_read_paths"])&supports))):
            ct[key]+=int(val)
        ct["rescues"]+=int(not norm_a and norm_b)
        ct["regressions"]+=int(norm_a and not norm_b)
        reports.append({"id":row["id"],"raw_control":raw_a,"raw_evidence":raw_b,
                        "normalized_control":norm_a,"normalized_evidence":norm_b,
                        "support_docs_read_control":len(set(a["successful_read_paths"])&supports),
                        "support_docs_read_evidence":len(set(b["successful_read_paths"])&supports)})
    gain=ct["rescues"]-ct["regressions"]
    ratio=(ct["tool_calls_evidence"]/ct["tool_calls_control"]
           if ct["tool_calls_control"] else None)
    gate=bool(gain>=GATE["net_normalized_correct_min"] and
              ct["regressions"]<=GATE["max_regressions"] and
              ratio is not None and ratio<=GATE["max_relative_tool_cost"])
    return {"protocol":PROTOCOL,"status":"COMPLETE_PHYSICAL_V7_MULTIHOP_EVIDENCE_PROMPT_AB",
            "n":N,"source_sha":blind["cases_sha256"],"scores":dict(ct),
            "normalized_net":gain,"relative_tool_calls":ratio,"gate":GATE,
            "positive_screen_only":gate,"promotion_allowed":False,
            "no_GAIA_or_hidden":True,"no_QLoRA_training":True,
            "no_agent_runtime_modification":True,
            "public_validation_pretraining_overlap_unknown":True,"cases":reports}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score"))
    for key in ("prior_linked","prior_novel","prior16","prior48","prior8","blind","labels","work","out"):
        p.add_argument("--"+key.replace("_","-"),dest=key,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--parts",type=Path,nargs="+")
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("never overwrite an immutable outcome")
    if a.mode=="freeze":
        from datasets import load_dataset
        rows=list(islice(load_dataset(source48.REPO,split="validation",
                                     revision=source48.REV,streaming=True),5000))
        blind,ref=freeze(rows,*[json.loads(getattr(a,k).read_text())
                              for k in ("prior_linked","prior_novel","prior16","prior48","prior8")])
        if a.labels.exists():raise FileExistsError("reference exists")
        a.labels.write_text(json.dumps(ref,ensure_ascii=False,indent=2)+"\n")
        out=blind
    elif a.mode=="infer":
        out=physical(json.loads(a.blind.read_text()),a.shard,a.work)
    else:
        out=aggregate([json.loads(x.read_text()) for x in a.parts],
                      json.loads(a.blind.read_text()),json.loads(a.labels.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("V7_MULTIHOP_EVIDENCE_CONTROLLED",json.dumps({
        k:v for k,v in out.items() if k not in ("cases","references")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
