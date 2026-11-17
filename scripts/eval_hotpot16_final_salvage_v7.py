"""Public HotpotQA16: Base vs v7 LoRA, failure-only evidence-backed finalization.

Pre-register NEW public 8 bridge + 8 comparison cases. Exclude all tracked
Hotpot8+16+8 question IDs and support paths. Same original Qwen3.5-9B GGUF,
same original v7 adapter, same runtime 12 steps, same frozen verbatim
serializer. Intervention ONLY after objectively failed finalization
(invalid_final, empty_final, max_steps), no additional exploration/tools.
One same-model call sees Q + bounded already-observed tool messages. Accept
ONLY short answer that appears verbatim in a successful source observation.
No gold or labels in prompts; all nonfailure outputs are bitwise identical.
Public diagnostic only, not GAIA, hidden test, or training gain.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from time import monotonic

from datasets import load_dataset
from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.eval_2wiki8_v7_factorial import CountingOpenAI, BASE_SHA, V7_SHA, SERIALIZER_SHA, stable_sha
from scripts.eval_2wiki_serializer_confirm48 import LocalSearch
from scripts.eval_hotpot_source_shift import page_path, REPO, REV
from scripts.eval_hotpot_serializer_new16 import serialize, PROTOCOL as OLD16_PROTOCOL
from scripts import eval_hotpot8_v7_factorial as last

PROTOCOL="HOTPOT-PUBLIC-NEW16-BASE-V7-FAILURE-FINALIZE-V1"
N=16
SHARDS=8
PER_SHARD=2
TYPES={"bridge":8,"comparison":8}
SEED="new-independent-hotpot16-finalize-failure-v0-20261117"
PRIOR8_SHA=last.PRIOR_OLD8
PRIOR16_SHA="342d70381cd48dc81026fad320fb963184150be278a2d3eb9a034c863d2955b9"
PRIOR_LATEST8_SHA="249591c8b085a6fa16fc9092538dcd29aa117af23a8dbc31a8fdd5c37321f6e0"
FAIL_REASONS={"invalid_final","empty_final","max_steps"}
RECOVERY_POLICY=(
  "You are a strict final answer extractor for an unfinished research assistant. "
  "You may ONLY use already-observed source/tool excerpts supplied below, "
  "NOT your general knowledge, and you have NO tool access. "
  "Return one concise factual answer copied verbatim as a substring of "
  "the source excerpts (maximum 96 characters); no explanation, tool calls, "
  "markdown, or citation markers. If no answer is explicitly supported, "
  "return NO_ANSWER. Do not guess."
)
POLICY_SHA=sha256(RECOVERY_POLICY.encode()).hexdigest()
GATE={"min_net_rescues":2,"max_regressions":0,"min_fallback_uses":2,
      "no_promotion_without_diverse_holdout":True}

def key(x):
    return sha256((SEED+":"+str(x)).encode()).hexdigest()

def validate_prior(a,b,c):
    if (a.get("n")!=8 or len(a.get("cases",[]))!=8
        or a.get("cases_sha256")!=PRIOR8_SHA
        or stable_sha(a["cases"])!=PRIOR8_SHA):
        raise ValueError("prior source Hotpot8 original contract mismatch")
    if (b.get("n")!=16 or len(b.get("cases",[]))!=16
        or b.get("protocol")!=OLD16_PROTOCOL
        or b.get("new_source_sha")!=PRIOR16_SHA
        or stable_sha(b["cases"])!=PRIOR16_SHA):
        raise ValueError("prior Hotpot16 source contract mismatch")
    if (c.get("n")!=8 or len(c.get("cases",[]))!=8
        or c.get("protocol")!=last.PROTOCOL
        or c.get("source_sha")!=PRIOR_LATEST8_SHA
        or stable_sha(c["cases"])!=PRIOR_LATEST8_SHA):
        raise ValueError("prior Hotpot crossfamily8 contract mismatch")
    seen={row["id"] for q in (a,b,c) for row in q["cases"]}
    supports={p for q in (a,b,c) for row in q["cases"] for p in row["support"]}
    if len(seen)!=32 or len(supports)<48:
        raise ValueError("prior tracked Hotpot32 source overlap or missing supports")
    return seen,supports

def freeze(rows, prior8, prior16, priornew8):
    ids,used=validate_prior(prior8,prior16,priornew8)
    selected=[]
    counts=Counter()
    for row in sorted(rows,key=lambda r:key(str(r.get("id") or ""))):
        rid=str(row.get("id") or "")
        kind=str(row.get("type") or "")
        if not rid or rid in ids or kind not in TYPES or counts[kind]>=8:
            continue
        context=row.get("context") or {}
        titles=context.get("title") or []
        sentences=context.get("sentences") or []
        if len(titles)!=len(sentences) or len(titles)<8 or len(set(titles))!=len(titles):
            continue
        supports=row.get("supporting_facts") or {}
        pairs=list(zip(supports.get("title") or [],supports.get("sent_id") or []))
        support_titles={t for t,_ in pairs}
        if len(support_titles)!=2 or not support_titles.issubset(set(titles)):
            continue
        paths={page_path(t) for t in support_titles}
        if used.intersection(paths):
            continue
        docs=dict(zip(titles,sentences))
        sp=defaultdict(list)
        for title,idx in pairs:
            if not isinstance(idx,int) or not (0<=idx<len(docs[title])):
                sp.clear()
                break
            sp[page_path(title)].append(str(docs[title][idx]))
        if len(sp)!=2 or any(not text.strip() for vals in sp.values() for text in vals):
            continue
        answer=str(row.get("answer") or "").strip()
        if not 1<=len(answer)<=80:
            continue
        selected.append({
            "id":rid,"type":kind,"question":str(row["question"]),"answer":answer,
            "files":{page_path(t):t+"\n"+"\n".join(map(str,s)) for t,s in zip(titles,sentences)},
            "index_titles":{t:page_path(t) for t in titles},
            "support":dict(sp),
        })
        used.update(paths)
        counts[kind]+=1
        if len(selected)==N:
            break
    if len(selected)!=N or dict(counts)!=TYPES:
        raise RuntimeError(f"not enough new diverse source cases: {len(selected)}, {dict(counts)}")
    return {
        "status":"SOURCE_FROZEN_PREINFERENCE","protocol":PROTOCOL,
        "source_revision":REV,"source_repo":REPO,
        "source_sha":stable_sha(selected),
        "known_prior_ids_excluded":32,
        "known_prior_supports_excluded":True,
        "selected_support_paths":len({p for c in selected for p in c["support"]}),
        "types":TYPES,"n":N,"shards":SHARDS,
        "base_sha":BASE_SHA,"adapter_sha":V7_SHA,
        "serializer_sha":SERIALIZER_SHA,
        "salvage_policy_sha":POLICY_SHA,"preregistered_gate":GATE,
        "no_training":True,"not_gaia_or_hidden":True,
        "public_pretraining_exposure_unverified":True,
        "cases":selected,
    }

def validate_frozen(obj):
    if (obj.get("status")!="SOURCE_FROZEN_PREINFERENCE"
        or obj.get("protocol")!=PROTOCOL or obj.get("source_revision")!=REV
        or obj.get("source_sha")!=stable_sha(obj.get("cases",[]))
        or obj.get("n")!=N or obj.get("shards")!=SHARDS
        or obj.get("types")!=TYPES
        or obj.get("known_prior_ids_excluded")!=32
        or obj.get("base_sha")!=BASE_SHA or obj.get("adapter_sha")!=V7_SHA
        or obj.get("serializer_sha")!=SERIALIZER_SHA
        or obj.get("salvage_policy_sha")!=POLICY_SHA
        or obj.get("preregistered_gate")!=GATE
        or len(obj.get("cases",[]))!=N):
        raise ValueError("source/model/policy prereg contract drift")
    if len({x["id"] for x in obj["cases"]})!=N or Counter(x["type"] for x in obj["cases"])!=TYPES:
        raise ValueError("invalid frozen source IDs or type balance")

def _bounded_source_excerpts(trace):
    # Snapshot exactly the successful tool observations the original agent saw.
    observations=[]
    for event in trace:
        if event.kind!="tool_result" or not event.data.get("ok"):continue
        text=str(event.data.get("content") or "").strip()
        if text:
            observations.append(text[:1200])
    return observations[-10:]

def _proposal_accepted(candidate,excerpts):
    value=str(candidate or "").strip()
    if (not value or len(value)>96 or value.upper()=="NO_ANSWER"
        or value.lstrip().startswith("<tool_call>")
        or "\n" in value):
        return False
    return any(value.casefold() in obs.casefold() for obs in excerpts)

def one_case(case,mode,work):
    _materialize_files({"files":case["files"]},work)
    question=case["question"]
    prompt=("Answer the following question using local article search and read tools. "
            "Search for article names and read returned paths before answering. "
            "A local source may be a distractor; decide whether more evidence is needed. "
            "Return only the answer, not reasoning.\n\nQuestion: "+question)
    original_model=CountingOpenAI()
    start=monotonic()
    result=AgentRuntime(original_model,[LocalSearch(case["index_titles"],case["files"]),ReadTool()],
                        max_steps=12).run(prompt,work)
    native_s=round(monotonic()-start,2)
    # This uses the same previously frozen formatter and costs one call
    # for completed tasks; it is NOT applied to failed finalizations.
    formatter=CountingOpenAI(max_new_tokens=128)
    final=result.answer
    if result.completed and result.answer.strip():
        final,_=serialize(question,result.answer,formatter)
    baseline_correct=bool(result.completed and gaia_score(final,case["answer"]))
    recovered=final
    fallback_used=False
    accepted=False
    recovery_elapsed=0.
    excerpt_count=0
    if not result.completed and result.stop_reason in FAIL_REASONS:
        excerpts=_bounded_source_excerpts(result.trace)
        excerpt_count=len(excerpts)
        if excerpts:
            fallback_used=True
            recovery_model=CountingOpenAI(max_new_tokens=128)
            user=json.dumps({"question":question,"already_observed_source_excerpts":excerpts},
                            ensure_ascii=False,sort_keys=True)
            started=monotonic()
            proposal=recovery_model.complete_final([
                {"role":"system","content":RECOVERY_POLICY},
                {"role":"user","content":user}
            ],[]).content
            recovery_elapsed=round(monotonic()-started,2)
            accepted=_proposal_accepted(proposal,excerpts)
            if accepted:
                recovered=proposal.strip()
    else:
        recovery_model=None
    salvage_correct=bool((result.completed or accepted) and gaia_score(recovered,case["answer"]))
    base_calls=original_model.calls+formatter.calls
    salvage_calls=base_calls+(recovery_model.calls if recovery_model else 0)
    common={"completed":result.completed,"stop_reason":result.stop_reason,
            "steps":result.metrics.steps,
            "tool_calls":result.metrics.tool_calls,
            "tool_errors":result.metrics.tool_errors}
    return {
      f"{mode}_frozen_formatter":dict(common,correct=baseline_correct,
          calls=base_calls,elapsed_s=round(native_s,2)),
      f"{mode}_failure_salvage":dict(common,correct=salvage_correct,
          calls=salvage_calls,elapsed_s=round(native_s+recovery_elapsed,2),
          fallback_used=fallback_used,fallback_accepted=accepted,
          fallback_seconds=recovery_elapsed,source_excerpt_count=excerpt_count)
    }

def run_shard(obj,mode,shard,work):
    validate_frozen(obj)
    if mode not in ("base","adapter") or shard not in range(SHARDS):
        raise ValueError("invalid mode/shard")
    cases=obj["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    if len(cases)!=PER_SHARD:raise ValueError("incomplete source shard")
    output=[]
    for case in cases:
        arms=one_case(case,mode,work/key(case["id"])[:12])
        output.append({"task_sha256":sha256(case["id"].encode()).hexdigest(),
                       "type":case["type"],"arms":arms})
        print("HOTPOT16_FINAL_SALVAGE",mode,shard,case["type"],
              {k:{"correct":v["correct"],"stop":v["stop_reason"],"calls":v["calls"]}
               for k,v in arms.items()},flush=True)
    return {"status":"PHYSICAL_TWO_CASES_COMPLETE","protocol":PROTOCOL,
            "mode":mode,"shard":shard,"n":PER_SHARD,
            "source_sha":obj["source_sha"],"base_sha":BASE_SHA,
            "adapter_sha":V7_SHA,"serializer_sha":SERIALIZER_SHA,
            "salvage_policy_sha":POLICY_SHA,"cases":output}

def aggregate(parts):
    if len(parts)!=2*SHARDS:
        raise ValueError("need exactly 16 actual model/shard outputs")
    expected={(mode,shard) for mode in ("base","adapter") for shard in range(SHARDS)}
    got={(part.get("mode"),part.get("shard")) for part in parts}
    if len(got)!=len(parts) or got!=expected:
        raise ValueError("missing/duplicate physical mode×shard pairs")
    if any(p.get("status")!="PHYSICAL_TWO_CASES_COMPLETE"
           or p.get("n")!=PER_SHARD or p.get("protocol")!=PROTOCOL
           or p.get("base_sha")!=BASE_SHA or p.get("adapter_sha")!=V7_SHA
           or p.get("serializer_sha")!=SERIALIZER_SHA
           or p.get("salvage_policy_sha")!=POLICY_SHA
           or len(p.get("cases",[]))!=PER_SHARD for p in parts):
        raise ValueError("physical source/runtime/policy failed contract")
    if len({p["source_sha"] for p in parts})!=1:
        raise ValueError("source SHA not matched across modes")
    joined={}
    for part in parts:
        for case in part["cases"]:
            slot=joined.setdefault(case["task_sha256"],{"type":case["type"],"arms":{}})
            if slot["type"]!=case["type"] or set(slot["arms"])&set(case["arms"]):
                raise ValueError("duplicate or mismatched task arm")
            slot["arms"].update(case["arms"])
    arms=("base_frozen_formatter","base_failure_salvage",
          "adapter_frozen_formatter","adapter_failure_salvage")
    if len(joined)!=N or any(set(row["arms"])!=set(arms) for row in joined.values()):
        raise ValueError("incomplete 16×4 matched task outcomes")
    if Counter(row["type"] for row in joined.values())!=TYPES:
        raise ValueError("task family balance changed")
    results={}
    for name in arms:
        results[name]={
            "correct":sum(r["arms"][name]["correct"] for r in joined.values()),
            "calls":sum(r["arms"][name]["calls"] for r in joined.values()),
            "elapsed_s":round(sum(r["arms"][name]["elapsed_s"] for r in joined.values()),2),
            "completed":sum(r["arms"][name]["completed"] for r in joined.values()),
            "tools":sum(r["arms"][name]["tool_calls"] for r in joined.values())
        }
    comparisons={}
    for mode in ("base","adapter"):
        a,b=mode+"_frozen_formatter",mode+"_failure_salvage"
        rescued=sum(not row["arms"][a]["correct"] and row["arms"][b]["correct"] for row in joined.values())
        regressed=sum(row["arms"][a]["correct"] and not row["arms"][b]["correct"] for row in joined.values())
        uses=sum(row["arms"][b]["fallback_used"] for row in joined.values())
        accepted=sum(row["arms"][b]["fallback_accepted"] for row in joined.values())
        comparisons[mode]={"rescues":rescued,"regressions":regressed,"net":rescued-regressed,
                           "fallback_uses":uses,"accepted":accepted,
                           "review_informative":uses>=GATE["min_fallback_uses"],
                           "gate_positive":rescued-regressed>=GATE["min_net_rescues"] and
                            regressed<=GATE["max_regressions"] and uses>=GATE["min_fallback_uses"]}
    return {"status":"COMPLETE_PUBLIC_DIAGNOSTIC_NO_PROMOTION",
            "protocol":PROTOCOL,"source_sha":parts[0]["source_sha"],
            "source_repo":REPO,"n":N,"shards":len(parts),"type_counts":TYPES,
            "base_sha":BASE_SHA,"adapter_sha":V7_SHA,"serializer_sha":SERIALIZER_SHA,
            "salvage_policy_sha":POLICY_SHA,"totals":results,"comparisons":comparisons,
            "gaia_delta_claimed":False,"promotion_allowed":False,
            "no_training":True,"stronger_model_used":False,
            "public_validation_not_hidden":True,
            "cases":[{"task_sha256":k,**v} for k,v in sorted(joined.items())]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--prior8",type=Path)
    p.add_argument("--prior16",type=Path)
    p.add_argument("--prior-latest8",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--mode",choices=["base","adapter"])
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--work",type=Path,default=Path("hotpot-final-salvage-work"))
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite frozen source or outcome")
    if args.source_only:
        if any(x is None for x in (args.prior8,args.prior16,args.prior_latest8)):
            p.error("need all 3 prior artifacts to exclude 32 exposed cases")
        rows=list(islice(load_dataset(REPO,"distractor",split="validation",revision=REV,streaming=True),2500))
        result=freeze(rows,json.loads(args.prior8.read_text()),
                      json.loads(args.prior16.read_text()),
                      json.loads(args.prior_latest8.read_text()))
    elif args.inputs:
        result=aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("HOTPOT16_SALVAGE_FULL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if args.frozen is None or args.shard is None or args.mode is None:
            p.error("need frozen, shard and model treatment")
        result=run_shard(json.loads(args.frozen.read_text()),args.mode,args.shard,args.work)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
