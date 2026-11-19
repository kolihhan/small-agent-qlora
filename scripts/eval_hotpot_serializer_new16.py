"""Prospective question-aware, evidence-preserving final-answer serialization.

New 16 HotpotQA public validation cases, no overlap with prior Hotpot8 IDs or
their annotated support document titles. Original fixed Qwen3.5-9B agent runs
ONCE; a second call to the SAME original model extracts a short answer from
its own completed reply. No gold, documents, reference, stronger teacher, or
tool use is available to the serializer. No training and not a GAIA test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from urllib.request import urlopen

from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.agent.types import ModelRuntimeError
from scripts.eval_hotpot_source_shift import (
    evaluate, page_path, REPO, REV, MODEL, DIGEST,
)
from scripts.audit_2wiki_answer_format_confound import last_line_answer

N=16
SHARDS=4
SEED="hotpot-new16-goldblind-qwen9b-final-serializer-20261117-v1"
PROTOCOL="HOTPOT16-NEW-PUBLIC-SOURCE-QWEN9B-QUESTION-CONDITIONED-FINAL-V1"
PREVIOUS_SOURCE_SHA="47f52b27dc4c3a8afddabf468bac6912325ee817ddd21bbb2caffe3bf03acb5d"
GATE={"net_exact":3, "max_regressions":1, "min_rescue_types":2}
SERIALIZER_RULE=(
    "You are a lossless answer-field serializer. You receive a question and an "
    "earlier assistant reply. You MUST NOT answer the question from memory, "
    "browse, reason with outside facts, or change the earlier reply's answer. "
    "Extract the shortest single answer span that the EARLIER REPLY explicitly "
    "asserts as the answer to the question: a proper name, number, year, "
    "place, yes/no, or short phrase. Preserve spelling and meaning. "
    "Do not select a distractor or an answer the reply rejected. "
    "If the reply is uncertain, conflicting, missing the answer, or cannot "
    "be reduced safely, output the earlier reply UNCHANGED. "
    "Do not include any explanation, quotes, prefixes, markdown or extra text."
)
SERIALIZER_SHA=sha256(SERIALIZER_RULE.encode()).hexdigest()


def ranked(value: str) -> str:
    return sha256((SEED + ":" + value).encode()).hexdigest()


def freeze(rows: list[dict], previous: dict) -> dict:
    if (previous.get("cases_sha256") != PREVIOUS_SOURCE_SHA
        or previous.get("n") != 8 or len(previous.get("cases", [])) != 8):
        raise RuntimeError("Previously exposed Hotpot8 immutable source mismatch")
    old_ids={r["id"] for r in previous["cases"]}
    old_support=set()
    for case in previous["cases"]:
        old_support.update(case["support"])
    picked, all_seen = [], set()
    for row in sorted(rows, key=lambda r: ranked(str(r.get("id")))):
        kind = str(row.get("type") or "")
        if kind not in ("bridge", "comparison") or sum(
                x["type"] == kind for x in picked) >= N//2:
            continue
        identity = str(row.get("id") or "")
        if not identity or identity in old_ids or identity in all_seen:
            continue
        context=row.get("context") or {}
        titles=context.get("title") or []
        parts=context.get("sentences") or []
        if len(titles) != len(parts) or len(titles) < 8 or len(set(titles)) != len(titles):
            continue
        facts=row.get("supporting_facts") or {}
        sources=list(zip(facts.get("title", []),facts.get("sent_id", [])))
        key_titles={title for title,_ in sources}
        if len(key_titles)!=2 or not key_titles.issubset(set(titles)):
            continue
        support_paths={page_path(t) for t in key_titles}
        if old_support & support_paths:
            continue
        docs=dict(zip(titles,parts))
        src=defaultdict(list)
        for t,i in sources:
            if not isinstance(i,int) or not (0 <= i < len(docs[t])):
                src.clear()
                break
            src[page_path(t)].append(str(docs[t][i]))
        if len(src)!=2 or any(not v.strip() for x in src.values() for v in x):
            continue
        answer=str(row.get("answer") or "").strip()
        if not 1<=len(answer)<=80:
            continue
        all_seen.add(identity)
        picked.append({
            "id":identity,"type":kind,"question":str(row["question"]),
            "answer":answer,
            "files":{page_path(t): t+"\n"+"\n".join(map(str,s))
                     for t,s in zip(titles,parts)},
            "index_titles":{t:page_path(t) for t in titles},
            "support":dict(src),
        })
        if len(picked)==N:
            break
    if len(picked)!=N or sum(x["type"]=="bridge" for x in picked)!=8:
        raise RuntimeError("Insufficient balanced independent Hotpot source cases")
    if any(r["id"] in old_ids or set(r["support"]) & old_support for r in picked):
        raise RuntimeError("Prior exposed question/support overlap")
    blob=json.dumps(picked,sort_keys=True).encode()
    return {
        "status":"FROZEN_BEFORE_ANY_MODEL_INFERENCE",
        "protocol":PROTOCOL,"repo":REPO,"revision":REV,
        "split":"distractor/validation","source_prefix":2000,
        "prior_source_sha":PREVIOUS_SOURCE_SHA,
        "new_source_sha":sha256(blob).hexdigest(),
        "model":MODEL,"model_digest":DIGEST,
        "serializer_sha":SERIALIZER_SHA,"gate":GATE,
        "n":N,"types":{"bridge":8,"comparison":8},
        "human_semantic_label_quality":"publisher_hotpot_exact_only",
        "pretraining_contamination_unverified":True,
        "no_gaia_examples":True,
        "cases":picked,
    }


def serialize(question: str, answer: str, model: OllamaModel) -> tuple[str, str]:
    """Second fixed-Qwen call plus conservative gold-blind verbatim-span gate."""
    if not answer.strip():
        return answer,"empty_original"
    user=json.dumps({"question":question,"earlier_reply":answer},
                    ensure_ascii=False,sort_keys=True)
    try:
        turn=model.complete([
            {"role":"system","content":SERIALIZER_RULE},
            {"role":"user","content":user},
        ],tools=[])
    except ModelRuntimeError:
        return answer,"model_error"
    candidate=turn.content.strip()
    if (turn.tool_calls or not candidate or len(candidate)>96
        or "\n" in candidate or
        not re.search(re.escape(candidate),answer,flags=re.IGNORECASE)):
        return answer,"rejected_not_verbatim_or_not_short"
    # Reject vacuous nonanswers and full explanations even when verbatim.
    if re.search(r"(?i)\b(therefore|because|however|uncertain|not enough information)\b",
                 candidate):
        return answer,"rejected_nonanswer"
    return candidate,"accepted_verbatim"


def one_case(case: dict, root: Path) -> dict:
    # Unchanged native agent scaffold and tools; one autonomous run per task.
    old=evaluate(case,"native",root)
    original=old["raw_answer"]
    output,reason=serialize(
        case["question"],original,
        OllamaModel(model=MODEL,timeout_s=300,max_new_tokens=128))
    if original!=old["raw_answer"]:
        raise RuntimeError("Original model reply mutated by serializer")
    return {
        "id":case["id"],"type":case["type"],"completed":old["completed"],
        "raw":original,"serialized":output,"serialization_status":reason,
        "raw_exact":old["raw_exact"],
        "lastline_exact":old["locked_lastline_exact"],
        "serialized_exact":bool(old["completed"] and gaia_score(output,case["answer"])),
        "original_support_docs":old["support_read_count"],
        "tool_calls":old["tool_calls"],"tool_errors":old["tool_errors"],
        "seconds":old["seconds"],
    }


def run_shard(frozen: dict, shard: int, root: Path) -> dict:
    if (shard not in range(SHARDS) or frozen.get("protocol")!=PROTOCOL
        or frozen.get("status")!="FROZEN_BEFORE_ANY_MODEL_INFERENCE"
        or frozen.get("serializer_sha")!=SERIALIZER_SHA
        or frozen.get("gate")!=GATE
        or frozen.get("model_digest")!=DIGEST
        or len(frozen.get("cases",[]))!=N
        or sha256(json.dumps(frozen["cases"],sort_keys=True).encode()).hexdigest()
        !=frozen.get("new_source_sha")):
        raise RuntimeError("Frozen source/model/serializer/experimental gate drift")
    with urlopen("http://127.0.0.1:11434/api/tags",timeout=15) as stream:
        models=json.load(stream)["models"]
    if not any(x["name"]==MODEL and x["digest"]==DIGEST for x in models):
        raise RuntimeError("Wrong physical Qwen3.5-9B model digest")
    subset=frozen["cases"][shard*4:(shard+1)*4]
    cases=[one_case(c,root/ranked(c["id"])[:12]) for c in subset]
    return {
        "status":"complete","protocol":PROTOCOL,"shard":shard,"n":4,
        "source_sha":frozen["new_source_sha"],"serializer_sha":SERIALIZER_SHA,
        "digest":DIGEST,"cases":cases,
    }


def aggregate(parts: list[dict]) -> dict:
    if (len(parts)!=SHARDS or sorted(p.get("shard") for p in parts)!=list(range(SHARDS))
        or len({p.get("source_sha") for p in parts})!=1
        or any(p.get("status")!="complete" or p.get("protocol")!=PROTOCOL
               or p.get("digest")!=DIGEST or p.get("n")!=4
               or p.get("serializer_sha")!=SERIALIZER_SHA
               or len(p.get("cases",[]))!=4 for p in parts)):
        raise RuntimeError("Incomplete or changed prospective physical paired cohort")
    rows=[c for p in sorted(parts,key=lambda x:x["shard"]) for c in p["cases"]]
    if len(rows)!=N or len({r["id"] for r in rows})!=N:
        raise RuntimeError("Missing or duplicate source ID")
    scores={k:sum(bool(r[k]) for r in rows)
            for k in ("raw_exact","lastline_exact","serialized_exact")}
    rescue=[r for r in rows if not r["raw_exact"] and r["serialized_exact"]]
    regression=[r for r in rows if r["raw_exact"] and not r["serialized_exact"]]
    types=sorted({r["type"] for r in rescue})
    passed=(len(rescue)-len(regression)>=GATE["net_exact"]
            and len(regression)<=GATE["max_regressions"]
            and len(types)>=GATE["min_rescue_types"])
    return {
        "status":"COMPLETE_PAIRED_PUBLIC_DEV_SCREEN_NOT_PROMOTED",
        "protocol":PROTOCOL,"source_sha":parts[0]["source_sha"],
        "same_Qwen3_5_9B_digest":DIGEST,"serializer_sha":SERIALIZER_SHA,
        "n":N,"scores":scores,
        "rescues":len(rescue),"regressions":len(regression),
        "rescue_types":types,"rescue_ids":[r["id"] for r in rescue],
        "regression_ids":[r["id"] for r in regression],
        "net_exact":len(rescue)-len(regression),
        "finished":sum(r["completed"] for r in rows),
        "agent_tools":sum(r["tool_calls"] for r in rows),
        "serializer_extra_model_calls_attempted":sum(bool(r["raw"].strip()) for r in rows),
        "accepted_verbatim":sum(r["serialization_status"]=="accepted_verbatim" for r in rows),
        "status_counts":{s:sum(r["serialization_status"]==s for r in rows)
                         for s in sorted({r["serialization_status"] for r in rows})},
        "prior_source_excluded":True,"no_GAIA_tasks_or_training":True,
        "preregistered_gate":GATE,"positive_screen_only":passed,
        "promotion_allowed":False,
        "claim_boundary":"New question+support-disjoint public HotpotQA validation 16, not GAIA and not hidden; no stronger teacher/model, only pinned base second formatter call; this tests answer serialization not new reasoning competence. Pretraining/source independence cannot be proven. An EX gain here is not a GAIA score.",
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--work",type=Path,default=Path("hotpot-serializer-work"))
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        r=aggregate([json.loads(p.read_text()) for p in a.inputs])
        print("HOTPOT16_SERIALIZER_PAIRED_FINAL",json.dumps({
            k:v for k,v in r.items() if k!="cases"},sort_keys=True),flush=True)
    elif a.frozen:
        if a.shard is None:
            p.error("--shard needed for frozen paired inference")
        r=run_shard(json.loads(a.frozen.read_text()),a.shard,a.work)
    elif a.prior:
        from datasets import load_dataset
        rows=list(islice(load_dataset(
            REPO,"distractor",split="validation",revision=REV,
            streaming=True),2000))
        r=freeze(rows,json.loads(a.prior.read_text()))
        print("HOTPOT16_SERIALIZER_SOURCE_FROZEN",r["new_source_sha"],flush=True)
    else:
        p.error("provide exactly one of --prior, --frozen, --inputs")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
