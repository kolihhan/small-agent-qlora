"""Prospective *pre-registered 48-question confirmation* of same Qwen3.5-9B lossless answer serializer.

2WikiMultihopQA validation only, original realistic local article search + read,
source-doc/ID exclusion of the 8 original-native, 8 source-link, 8 later novel,
and 16 recently scored positive-screen historical 2Wiki cases. The same original autonomous Qwen9B agent makes one reply;
one extra call to SAME model uses Hotpot16's frozen question-aware serializer
to extract only verbatim answer substrings. No external LLM, gold or tool access
during formatting. No tuning from previously exposed validation labels.
These public questions may be in pretraining; older hidden exposures cannot
be excluded from repo branches not represented by the pinned frozen cohorts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from time import monotonic
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.training.trajectories import _materialize_files

from scripts.eval_hotpot_serializer_new16 import SERIALIZER_SHA, serialize

REPO="framolfese/2WikiMultihopQA"
REV="fe713bfbd1afbca1a65246741a75890405d56a3a"
MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL="2WIKI-CONFIRM48-QWEN9B-UNCHANGED-HOTPOT-VERBATIM-SERIALIZER-V1"
SEED="qwen9b-2wiki-confirmation-new48-nonoverlap-v1-20261117"
N=48
SHARDS=8
PER_SHARD=6
MIN_GATE={"net_exact":8,"max_regressions":1,"min_completed":40,"max_exact_mcnemar_p":0.05}
LINKED_PREVIOUS_SHA="fded3c65e2ccc829ae0efdd8af448896320590b63bb69e0e9ff1d52eadc4fbb7"
NOVEL_PREVIOUS_SHA="d102a3047ed96237c987ba30fe10f95d15c8ebcd4042cf78c71f4d1ecddbb8e9"
SERIALIZER_LOCKED_SHA="a1f844029179ab91e9ff0ebd0b9b8a6293801c686e4080921edd5b3a82ef60a3"
PRIOR_POSITIVE16_SOURCE_SHA="cb661a030156a60cbfb1bb4eb17bcb06bc056dbb7920126be517226fbf3c32dd"

# Historical original native8, pinned in earlier eval_2wiki_linked_read.py.
FIRST_EXPOSED_IDS=frozenset({
    "1822cbb4089111ebbd72ac1f6bf848b6","2e419977088111ebbd6dac1f6bf848b6",
    "36fc43f808dc11ebbd9cac1f6bf848b6","d956416e086711ebbd5eac1f6bf848b6",
    "5bec3cd408a711ebbd7fac1f6bf848b6","c5601df108a411ebbd7dac1f6bf848b6",
    "ef7aac32088511ebbd6dac1f6bf848b6","a91c363708e711ebbda6ac1f6bf848b6",
})
# Only SHA256(title)[:12] identifiers; original source titles not disclosed.
FIRST_EXPOSED_SUPPORT=frozenset({
    "00dc3fb25000","13f9d4c0bda6","1bb3836144f4","1df0580c169a",
    "20bb20c7de01","25149a7ded22","26e744bf956f","296bd082a284",
    "3cbbabfd5858","3e460acefa6c","40ba7f92a16a","50c4069bb759",
    "57a46389ea3c","590895f88ed4","6024fad10db5","61bacc3869f2",
    "74468bcadf09","750bd7f28412","a0f5da80c7bd","b072d5e25ba1",
    "b761b57f29e1","b953c51803fe","bf3f09ef073a","d27b66f348ef",
    "d30458571101","d4cb27906a6d","d78db1dabb77","e3796d7f11b1",
    "ec1356cf0af7","f5f04a7ed1ca","f754e4bbde83","fe13d2a87d57",
})


def title_id(title:str)->str:
    return sha256(title.encode()).hexdigest()[:12]


def page_path(title:str)->str:
    return f"pages/{title_id(title)}.txt"


def context_support(row:dict)->tuple[dict[str,str],dict[str,list[str]]]:
    raw=row["context"]
    original=(dict(zip(raw["title"],raw["sentences"],strict=True))
              if isinstance(raw,dict) else dict(raw))
    context={str(k):" ".join(map(str,v)) for k,v in original.items()}
    support={}
    facts=row["supporting_facts"]
    pairs=(zip(facts["title"],facts["sent_id"],strict=True)
           if isinstance(facts,dict) else facts)
    for title,sent in pairs:
        if title not in original or not isinstance(sent,int) or not 0<=sent<len(original[title]):
            raise ValueError("Published 2Wiki source/support pointer invalid")
        support.setdefault(title,[]).append(str(original[title][sent]))
    return context,support


def source_rank(identity:str)->str:
    return sha256(f"{SEED}:{identity}".encode()).hexdigest()


def freeze(rows:list[dict],linked_prior:dict,novel_prior:dict,positive_prior:dict)->dict:
    if (linked_prior.get("protocol")!="2wiki-validation-qwen9b-visible-local-links-pair-v1"
        or linked_prior.get("cases_sha")!=LINKED_PREVIOUS_SHA
        or linked_prior.get("source_revision")!=REV
        or len(linked_prior.get("cases",[]))!=8
        or sha256(json.dumps(linked_prior["cases"],sort_keys=True).encode()).hexdigest()
            !=LINKED_PREVIOUS_SHA):
        raise RuntimeError("Earlier source-linked 2Wiki8 freeze missing or changed")
    if (novel_prior.get("protocol")!="qwen35-9b-v3b-vs-base-2wiki-novel-validation8-v1"
        or novel_prior.get("new8_sha")!=NOVEL_PREVIOUS_SHA
        or novel_prior.get("source8_sha")!=LINKED_PREVIOUS_SHA
        or novel_prior.get("source_revision")!=REV
        or novel_prior.get("n")!=8
        or len(novel_prior.get("cases",[]))!=8
        or sha256(json.dumps(novel_prior["cases"],sort_keys=True).encode()).hexdigest()
           !=NOVEL_PREVIOUS_SHA):
        raise RuntimeError("Third previously exposed 2Wiki NOVEL8 is missing or changed")
    novel_ids={r["id"] for r in novel_prior["cases"]}
    novel_support={p.rsplit("/",1)[-1][:-4]
                   for r in novel_prior["cases"] for p in r["support"]}
    if len(novel_ids)!=8 or len(novel_support)<8:
        raise RuntimeError("Previous novel8 source identities incomplete")
    linked_ids={r["id"] for r in linked_prior["cases"]}
    linked_support={p.rsplit("/",1)[-1][:-4]
                    for r in linked_prior["cases"] for p in r["support"]}
    if len(linked_ids)!=8 or len(linked_support)<8:
        raise RuntimeError("Earlier 2Wiki source identities incomplete")
    excluded_ids=set(FIRST_EXPOSED_IDS | linked_ids | novel_ids)
    excluded_support=set(FIRST_EXPOSED_SUPPORT | linked_support | novel_support)
    if (FIRST_EXPOSED_IDS & linked_ids or FIRST_EXPOSED_IDS & novel_ids
        or linked_ids & novel_ids
        or FIRST_EXPOSED_SUPPORT & linked_support
        or FIRST_EXPOSED_SUPPORT & novel_support
        or linked_support & novel_support):
        raise RuntimeError("Prior 2Wiki first8/linked8 overlap unexpected")
    # Fourth historically exposed group: the completed successful 2Wiki16 run.
    # This source is a GROUND-TRUTH-FREE EXCLUSION manifest only: do not read
    # previous score case labels for selection, ranking or prompt adaptation.
    if (positive_prior.get("status")!="FROZEN_BEFORE_NEW_QWEN_MODEL_INFERENCE"
        or positive_prior.get("protocol")!="2WIKI-NEW16-ORIGINAL-QWEN9B-QUESTION-CONDITIONED-SERIALIZER-V1"
        or positive_prior.get("source_revision")!=REV
        or positive_prior.get("frozen_source_sha256")!=PRIOR_POSITIVE16_SOURCE_SHA
        or positive_prior.get("n")!=16
        or positive_prior.get("serializer_sha256")!=SERIALIZER_SHA
        or positive_prior.get("model_digest")!=DIGEST
        or len(positive_prior.get("cases",[]))!=16
        or sha256(json.dumps(positive_prior["cases"],sort_keys=True).encode()).hexdigest()
            !=PRIOR_POSITIVE16_SOURCE_SHA):
        raise RuntimeError("Pinned scored 2Wiki16 prior source missing or changed")
    prior16_ids={r["id"] for r in positive_prior["cases"]}
    prior16_support={p.rsplit("/",1)[-1][:-4]
                     for r in positive_prior["cases"] for p in r["support"]}
    if len(prior16_ids)!=16 or len(prior16_support)!=64:
        raise RuntimeError("Prior 2Wiki16 identity/support set not intact")
    if prior16_ids & excluded_ids or prior16_support & excluded_support:
        raise RuntimeError("Prior 2Wiki historical source document overlap")
    excluded_ids.update(prior16_ids)
    excluded_support.update(prior16_support)
    chosen,used_support=[],set(excluded_support)
    for row in sorted(rows,key=lambda r:source_rank(str(r.get("id") or r.get("_id")))):
        identity=str(row.get("id") or row.get("_id") or "")
        if not identity or identity in excluded_ids or any(r["id"]==identity for r in chosen):
            continue
        pages,support=context_support(row)
        if len(support)!=4 or len(pages)<6:
            continue
        support_ids={title_id(title) for title in support}
        if len(support_ids)!=4 or (used_support & support_ids):
            continue
        answer=str(row.get("answer") or "").strip()
        if not 1<=len(answer)<=80 or not any(
            answer.casefold() in " ".join(spans).casefold()
            for spans in support.values()):
            continue
        chosen.append({
            "id":identity,"question":str(row["question"]),
            "answer":answer,
            "files":{page_path(title):text for title,text in pages.items()},
            "index_titles":{title:page_path(title) for title in pages},
            "support":{page_path(title):spans for title,spans in support.items()},
        })
        used_support.update(support_ids)
        if len(chosen)==N:
            break
    if len(chosen)!=N or len({r["id"] for r in chosen})!=N:
        raise RuntimeError(f"Insufficient document-disjoint 48 new 2Wiki identities: {len(chosen)}")
    if any(r["id"] in excluded_ids or
           {p.rsplit("/",1)[-1][:-4] for p in r["support"]}&excluded_support
           for r in chosen):
        raise RuntimeError("Previously exposed 2Wiki task/document contamination")
    if SERIALIZER_SHA!=SERIALIZER_LOCKED_SHA:
        raise RuntimeError("Cannot alter Hotpot16 positive serializer rule")
    source_sha=sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()
    return {
        "status":"FROZEN_BEFORE_NEW_QWEN_MODEL_INFERENCE",
        "protocol":PROTOCOL,"source_repo":REPO,"source_revision":REV,
        "split":"validation","known_prior_ids_excluded":40,
        "known_previous_document_hashes_excluded":len(excluded_support),
        "previous_positive16_source_sha":PRIOR_POSITIVE16_SOURCE_SHA,
        "previous_first8_support_count":len(FIRST_EXPOSED_SUPPORT),
        "previous_linked8_source_sha":LINKED_PREVIOUS_SHA,
        "previous_novel8_source_sha":NOVEL_PREVIOUS_SHA,
        "previous_novel8_support_count":len(novel_support),
        "previous_linked8_support_count":len(linked_support),
        "frozen_source_sha256":source_sha,
        "serializer_sha256":SERIALIZER_SHA,
        "model_digest":DIGEST,"n":N,"shards":SHARDS,
        "gate":MIN_GATE,
        "not_GAIA":True,"not_hidden":True,
        "unknown_other_historical_exposure_and_pretraining_contamination":True,
        "no_stronger_teacher_or_model":True,
        "no_training":True,
        "cases":chosen,
    }


class LocalSearch(Tool):
    name="search"
    description="Search local article titles/snippets by a named entity or topic. Returns article file paths for read. Try related names if no results."
    schema={"type":"object","properties":{"query":{"type":"string","minLength":2}},
            "required":["query"],"additionalProperties":False}
    def __init__(self,title_paths:dict[str,str],files:dict[str,str]):
        self.paths,self.files=title_paths,files

    def run(self,arguments,workspace):
        query=str(arguments.get("query") or "").casefold().strip()
        tokens=set(re.findall(r"[a-z0-9]{3,}",query))-{
            "what","which","that","whose","when","where","year","name"}
        scored=[]
        for title,path in self.paths.items():
            name=title.casefold()
            overlap=sum(tok in name for tok in tokens)
            phrase=int(bool(query and (query in name or name in query)))
            score=4*phrase+overlap
            if score:
                scored.append((score,title,path))
        scored.sort(key=lambda x:(-x[0],x[1].casefold()))
        results=[f"{title} | {path} | {self.files[path][:110]}"
                 for _,title,path in scored[:5]]
        return ToolResult(True,"\n".join(results) if results else "No matching local article found.")


def validate_frozen(frozen:dict)->None:
    if (frozen.get("status")!="FROZEN_BEFORE_NEW_QWEN_MODEL_INFERENCE"
        or frozen.get("protocol")!=PROTOCOL
        or frozen.get("source_revision")!=REV
        or frozen.get("model_digest")!=DIGEST
        or frozen.get("serializer_sha256")!=SERIALIZER_SHA
        or SERIALIZER_SHA!=SERIALIZER_LOCKED_SHA
        or frozen.get("gate")!=MIN_GATE
        or frozen.get("known_prior_ids_excluded")!=40
        or frozen.get("previous_positive16_source_sha")!=PRIOR_POSITIVE16_SOURCE_SHA
        or len(frozen.get("cases",[]))!=N
        or sha256(json.dumps(frozen["cases"],sort_keys=True).encode()).hexdigest()
           !=frozen.get("frozen_source_sha256")):
        raise RuntimeError("Locked old Qwen, old serializer, cohort and gate drift")
    if len({r["id"] for r in frozen["cases"]})!=N:
        raise RuntimeError("Source duplicate")


def one_case(case:dict,work:Path)->dict:
    _materialize_files({"files":case["files"]},work)
    question=case["question"]
    prompt=("Answer the following question using the local article search and read tools. "
            "Search for article names and read returned paths before answering. "
            "A local source may be a distractor; decide whether more evidence is needed. "
            "Return only the answer, not reasoning.\n\nQuestion: "+question)
    model=OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=512)
    start=monotonic()
    result=AgentRuntime(model,[LocalSearch(case["index_titles"],case["files"]),ReadTool()],
                        max_steps=12).run(prompt,work)
    elapsed=monotonic()-start
    original=result.answer
    candidate,status=(original,"original_not_completed")
    if result.completed and original.strip():
        candidate,status=serialize(question,original,
            OllamaModel(model=MODEL,timeout_s=400,max_new_tokens=128))
    return {
        "task_hash":sha256(case["id"].encode()).hexdigest(),
        "original_exact":bool(result.completed and gaia_score(original,case["answer"])),
        "serialized_exact":bool(result.completed and gaia_score(candidate,case["answer"])),
        "original_completed":result.completed,
        "stop_reason":result.stop_reason,
        "raw_sha":sha256(original.encode()).hexdigest(),
        "serialized_sha":sha256(candidate.encode()).hexdigest(),
        "formatter_status":status,
        "extra_formatter_call":bool(result.completed and original.strip()),
        "tool_calls":result.metrics.tool_calls,
        "tool_errors":result.metrics.tool_errors,
        "original_agent_time_seconds":round(elapsed,2),
        "not_new_reasoning":True,
    }


def run_shard(frozen:dict,shard:int,work:Path)->dict:
    validate_frozen(frozen)
    if shard not in range(SHARDS):
        raise ValueError("Invalid 2Wiki shard")
    with urlopen("http://127.0.0.1:11434/api/tags",timeout=15) as response:
        models=json.load(response)["models"]
    if not any(z["name"]==MODEL and z["digest"]==DIGEST for z in models):
        raise RuntimeError("Original Qwen3.5-9B physical model digest differs")
    cases=frozen["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    if len(cases)!=PER_SHARD:
        raise RuntimeError("Source shard incomplete")
    results=[]
    for c in cases:
        got=one_case(c,work/source_rank(c["id"])[:12])
        results.append(got)
        print("TWOWIKI48_PAIRED",shard,got["task_hash"][:10],
              "raw",got["original_exact"],"format",got["serialized_exact"],
              "finished",got["original_completed"],flush=True)
    return {
        "status":"complete","protocol":PROTOCOL,
        "shard":shard,"n":PER_SHARD,
        "source_sha":frozen["frozen_source_sha256"],
        "serializer_sha":SERIALIZER_SHA,
        "model_digest":DIGEST,
        "cases":results,
    }


def aggregate(shards:list[dict])->dict:
    if (len(shards)!=SHARDS
        or sorted(x.get("shard") for x in shards)!=list(range(SHARDS))
        or len({x.get("source_sha") for x in shards})!=1
        or any(x.get("status")!="complete" or x.get("protocol")!=PROTOCOL
               or x.get("serializer_sha")!=SERIALIZER_SHA
               or x.get("model_digest")!=DIGEST
               or x.get("n")!=PER_SHARD or len(x.get("cases",[]))!=PER_SHARD
               for x in shards)):
        raise RuntimeError("Need all 8 physical Qwen source/model-locked 2Wiki48 shards")
    cases=[r for s in sorted(shards,key=lambda z:z["shard"]) for r in s["cases"]]
    if len(cases)!=N or len({r["task_hash"] for r in cases})!=N:
        raise RuntimeError("Unmatched or duplicated physical 2Wiki case IDs")
    original=sum(r["original_exact"] for r in cases)
    after=sum(r["serialized_exact"] for r in cases)
    rescued=[r for r in cases if not r["original_exact"] and r["serialized_exact"]]
    regressed=[r for r in cases if r["original_exact"] and not r["serialized_exact"]]
    completed=sum(r["original_completed"] for r in cases)
    # Exact two-sided paired binary McNemar (binomial, no asymptotic short-cut).
    from math import comb
    disc=len(rescued)+len(regressed)
    exact_p=min(1., 2*sum(comb(disc,k) for k in range(min(len(rescued),len(regressed))+1))/(2**disc)) if disc else 1.
    positive=(len(rescued)-len(regressed)>=MIN_GATE["net_exact"]
              and len(regressed)<=MIN_GATE["max_regressions"]
              and completed>=MIN_GATE["min_completed"]
              and exact_p<=MIN_GATE["max_exact_mcnemar_p"])
    return {
        "status":"COMPLETE_PUBLIC_2WIKI16_SECOND_FAMILY_SCREEN_NOT_PROMOTED",
        "protocol":PROTOCOL,"source_sha":shards[0]["source_sha"],
        "model_digest":DIGEST,"frozen_hotpot_rule_sha":SERIALIZER_SHA,
        "n":N,"original_exact":original,"serialized_exact":after,
        "rescues":len(rescued),"regressions":len(regressed),
        "net_exact":after-original,"completed":completed,
        "exact_mcnemar_two_sided_p":exact_p,
        "tool_calls":sum(r["tool_calls"] for r in cases),
        "tool_errors":sum(r["tool_errors"] for r in cases),
        "extra_formatter_calls":sum(r["extra_formatter_call"] for r in cases),
        "accepted_verbatim":sum(r["formatter_status"]=="accepted_verbatim" for r in cases),
        "original_agent_time_seconds":round(
            sum(r["original_agent_time_seconds"] for r in cases),1),
        "preregistered_gate":MIN_GATE,"positive_screen_only":positive,
        "promotion_allowed":False,
        "not_GAIA_not_hidden_and_no_new_weights":True,
        "claim_boundary":"Public 2Wiki validation48, new publisher identities and annotated support documents disjoint from all four recorded previous 2Wiki groups (8+8+8+16=40); selection seed and untouched Hotpot verbatim serializer frozen before Qwen inference, but unknown public pretraining/other historical exposure cannot be excluded. Not GAIA, reasoning improvement, hidden model generalization or deployable policy. Same original Qwen model plus extra generation has latency cost.",
        "cases":cases,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior-linked",type=Path)
    p.add_argument("--prior-novel",type=Path)
    p.add_argument("--prior-confirmed",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--work",type=Path,default=Path("2wiki48-work"))
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inputs:
        result=aggregate([json.loads(q.read_text()) for q in args.inputs])
        print("TWOWIKI48_SERIALIZER_PAIRED_FINAL",json.dumps({
            k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    elif args.prior_linked:
        if args.prior_novel is None or args.prior_confirmed is None:
            p.error("Source freeze requires --prior-novel AND --prior-confirmed scored 16")
        from datasets import load_dataset
        dataset=list(islice(load_dataset(
            REPO,split="validation",revision=REV,streaming=True),5000))
        result=freeze(dataset,json.loads(args.prior_linked.read_text()),
                      json.loads(args.prior_novel.read_text()),
                      json.loads(args.prior_confirmed.read_text()))
        print("TWOWIKI48_SOURCE_FROZEN",result["frozen_source_sha256"],flush=True)
    else:
        if not args.frozen or args.shard is None:
            p.error("Need --prior-linked, --prior-novel, --prior-confirmed; or --inputs; or --frozen and --shard")
        result=run_shard(json.loads(args.frozen.read_text()),args.shard,args.work)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
