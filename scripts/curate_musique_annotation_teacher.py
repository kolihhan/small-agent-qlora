"""Generate *annotation-oracle*, source-verified MuSiQue TRAIN tool SFT.

This is NOT observed model behavior and NOT a claim that QLoRA has improved.
Never train on MuSiQue DEV/test or the already evaluated MuSiQue TRAIN8. Keep
composition-family + physical-support-paragraph holdouts; replay every read
tool against real workspace bytes to verify tool observations. No synthetic
chain-of-thought, hidden supporting flags in prompts or GAIA examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import SYSTEM_PROMPT, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.prepare_musique_multihop_train import _hop, rows, _normal

CURATED_SOURCE = "musique-train"
SOURCE = "bdsaglam/musique@667064b514d22deecae00a3dccbd92bccf88869e"
TRAIN_FAMILIES = ("3hop1", "4hop1")
HOLDOUT_FAMILIES = ("3hop2", "4hop2", "4hop3")
SEED = "musique-source-annotated-multihop-sft-v1-20261115"
TRAIN_SHA = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
DEV_SHA = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def _prefix(r: dict) -> str:
    return str(r.get("id","")).split("__",1)[0]


def _key(r: dict) -> str:
    return sha256((SEED+":"+str(r["id"])).encode()).hexdigest()


def _strict_annotated_path(r: dict) -> list[int] | None:
    """Require each subanswer literally in its human-labeled supporting paragraph."""
    if _hop(r,excluded_ids=set(),excluded_questions=set(),
            excluded_decomp=set()) is None:
        return None
    parts={p["idx"]:p for p in r["paragraphs"]}
    path=[]
    for d in r["question_decomposition"]:
        idx=d.get("paragraph_support_idx")
        answer=str(d.get("answer") or "").strip()
        if (not isinstance(idx,int) or idx not in parts
            or parts[idx]["is_supporting"] is not True
            or idx in path or not answer
            or len(answer)>120
            or len(parts[idx]["paragraph_text"])>1800
            or _normal(answer) not in _normal(parts[idx]["paragraph_text"])):
            return None
        path.append(idx)
    if _normal(r["question_decomposition"][-1].get("answer")) != _normal(r["answer"]):
        return None
    return path


def _passage_hashes(r: dict, path: list[int]) -> tuple[str,...]:
    parts={p["idx"]:p for p in r["paragraphs"]}
    return tuple(sha256(
        " ".join(parts[idx]["paragraph_text"].split()).casefold().encode()
    ).hexdigest() for idx in path)


def make_oracle_demo(r: dict) -> dict | None:
    support=_strict_annotated_path(r)
    if support is None:
        return None
    parts=sorted(r["paragraphs"],key=lambda p:p["idx"])
    files={"passages/index.txt":"\n".join(
        f"passages/p{p['idx']}.txt | {p.get('title','')}" for p in parts)}
    for p in parts:
        files[f"passages/p{p['idx']}.txt"]=p["paragraph_text"]
    question=(
        "Answer using ONLY workspace passage files. Start with "
        "passages/index.txt, read relevant passage files using the read tool, "
        "follow multiple evidence links, and give one concise final answer. "
        "Question: "+str(r["question"])
    )
    tools=default_tools()
    reader=next(t for t in tools if t.name=="read")
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":question},
    ]
    training=[]
    with TemporaryDirectory() as tmp:
        workspace=Path(tmp)
        _materialize_files({"files":files},workspace)
        for step,source in enumerate(["passages/index.txt"]+[f"passages/p{i}.txt" for i in support]):
            arguments={"source":source}
            if reader.validate_arguments(arguments) is not None:
                raise AssertionError("invalid native read tool arguments")
            output=reader.run(arguments,workspace)
            if not output.ok:
                return None
            observation,_=output.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
            if step>0:
                expect=str(r["question_decomposition"][step-1]["answer"])
                if _normal(expect) not in _normal(observation):
                    return None
            action_id=f"oracleread_{step}"
            training.append(len(messages))
            messages.append({
                "role":"assistant","content":"",
                "tool_calls":[{"id":action_id,"type":"function",
                               "function":{"name":"read","arguments":arguments}}],
            })
            messages.append({
                "role":"tool","name":"read","tool_call_id":action_id,
                "content":observation,
            })
    training.append(len(messages))
    messages.append({"role":"assistant","content":str(r["answer"])})
    hop=len(support)
    task_id="musique-train-"+str(r["id"])
    checks={
        "completed":True, "final_answer_correct":True,
        "zero_tool_errors":True, "zero_duplicate_blocks":True,
        "required_tools_satisfied":True,"provenance_complete":True,
        "human_decomposition_all_subanswers_physically_observed":True,
    }
    return {
        "task_id":task_id,"source":CURATED_SOURCE,
        "capability":"multi_hop_evidence_to_final",
        "verified":True,"verification":checks,
        "provenance":{
            "license":"CC-BY-4.0",
            "generator":"musique-source-annotation-tool-replay",
            "generator_version":SOURCE,
            "oracle_type":"MuSiQue-human-annotation-and-physical-read-replay",
            "oracle_version":"MuSiQue-v1.0",
        },
        "required_tools":["read"],
        "tools":[t.definition() for t in tools],
        "messages":messages,
        "expected_answer":str(r["answer"]),
        "training_action_indices":training,
        "selection_method":"human_annotated_support_paths_not_model_generated",
        "derived_from_annotation_not_observed_model":True,
        "supporting_paragraph_hashes":list(_passage_hashes(r,support)),
        "source_id_sha256":sha256(str(r["id"]).encode()).hexdigest(),
        "composition_family":_prefix(r),
        "hop_count":hop,
        "metrics":{
            "steps":len(training),"tool_calls":1+hop,"tool_successes":1+hop,
            "tool_errors":0,"duplicate_calls_blocked":0,
        },
    }


def split_oracles(
    original: list[dict],
    *,
    blocked_ids: set[str],
    per_hop_train: int = 32,
    per_hop_holdout: int = 12,
    require_full: bool = True,
) -> tuple[list[dict],list[dict],dict]:
    """Source- and paragraph-disjoint family shift; full-parent teacher records."""
    groups=defaultdict(list)
    seen=set()
    counts=Counter()
    for r in original:
        rid=str(r.get("id") or "")
        if not rid or rid in seen:
            raise ValueError("missing or duplicate source MuSiQue parent ID")
        seen.add(rid)
        if sha256(rid.encode()).hexdigest() in blocked_ids:
            counts["excluded_previous_evaluation"]+=1
            continue
        f=_prefix(r)
        if f not in {*TRAIN_FAMILIES,*HOLDOUT_FAMILIES}:
            continue
        groups[f].append(r)
    for group in groups.values():
        group.sort(key=lambda r:(_key(r),r["id"]))
    selected={"train":[],"holdout":[]}
    used_support_hashes=set()
    used_question_hashes=set()
    for family,quota in [
        ("3hop1",per_hop_train),("4hop1",per_hop_train),
        ("3hop2",per_hop_holdout),("4hop2",per_hop_holdout),
    ]:
        target="train" if family in TRAIN_FAMILIES else "holdout"
        accepted=0
        for r in groups.get(family,[]):
            if accepted>=quota:
                break
            path=_strict_annotated_path(r)
            if path is None:
                counts["rejected_weak_annotation"]+=1
                continue
            hashes=set(_passage_hashes(r,path))
            qhash=sha256(_normal(r["question"]).encode()).hexdigest()
            if qhash in used_question_hashes or hashes&used_support_hashes:
                counts["rejected_duplicate_question_or_support"]+=1
                continue
            demo=make_oracle_demo(r)
            if demo is None:
                counts["rejected_failed_read_replay"]+=1
                continue
            selected[target].append(demo)
            accepted+=1
            used_question_hashes.add(qhash)
            used_support_hashes.update(hashes)
        counts["accepted_"+family]=accepted
    if require_full and (len(selected["train"])!=2*per_hop_train
                         or len(selected["holdout"])!=2*per_hop_holdout):
        raise ValueError(
            f"insufficient verified source-family/evidence-disjoint annotated tool demonstrations: "
            f"train={len(selected['train'])}/{2*per_hop_train}, "
            f"holdout={len(selected['holdout'])}/{2*per_hop_holdout}"
        )
    report={
        "protocol":"musique-source-annotated-tool-oracle-v1",
        "source":SOURCE,
        "train":len(selected["train"]),"holdout":len(selected["holdout"]),
        "train_3hop":sum(x["hop_count"]==3 for x in selected["train"]),
        "train_4hop":sum(x["hop_count"]==4 for x in selected["train"]),
        "holdout_3hop":sum(x["hop_count"]==3 for x in selected["holdout"]),
        "holdout_4hop":sum(x["hop_count"]==4 for x in selected["holdout"]),
        "train_actions":sum(len(x["training_action_indices"]) for x in selected["train"]),
        "holdout_actions":sum(len(x["training_action_indices"]) for x in selected["holdout"]),
        "train_composition_families":list(TRAIN_FAMILIES),
        "heldout_composition_families":["3hop2","4hop2"],
        "support_passages_shared_train_holdout":0,
        "questions_shared_train_holdout":0,
        "gaia_training_rows":0,
        "excluded_previous_evaluation":counts["excluded_previous_evaluation"],
        "rejection_counts":dict(sorted(counts.items())),
        "source_annotation_not_observed_model":True,
        "training_ran":False,
        "boundary":"Annotated MuSiQue TRAIN-only gold-derived teacher demonstrations, not observed model trajectories; holdout is different TRAIN composition families, MuSiQue DEV never used for SFT. No QLoRA improvement is claimed.",
    }
    return selected["train"],selected["holdout"],report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--prior-evaluation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--per-hop-train",type=int,default=32)
    p.add_argument("--per-hop-holdout",type=int,default=12)
    a=p.parse_args()
    if len({x.resolve() for x in (a.output,a.holdout,a.report)})!=3 or any(
        x.exists() for x in (a.output,a.holdout,a.report)):
        raise FileExistsError("refuse to overwrite source replay evidence")
    if sha256(a.train.read_bytes()).hexdigest()!=TRAIN_SHA or sha256(a.dev.read_bytes()).hexdigest()!=DEV_SHA:
        raise ValueError("MuSiQue source version drift")
    source=list(rows(a.train))
    dev=list(rows(a.dev))
    prior=[json.loads(line) for line in a.prior_evaluation.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prior)!=8 or any("source_id_hash" not in r for r in prior):
        raise ValueError("unrecognized previous MuSiQue eight-evaluation cohort")
    prior_hashes={r["source_id_hash"] for r in prior}
    # Never allow annotated TRAIN answers for the exact previous observed tasks.
    excluded_ids={str(r["id"]) for r in dev}
    excluded_questions={_normal(r["question"]) for r in dev}
    excluded_decomp={str(d["id"]) for r in dev for d in r.get("question_decomposition",[]) if isinstance(d,dict)}
    eligible=[]
    for r in source:
        if (_hop(r,excluded_ids=excluded_ids,excluded_questions=excluded_questions,
                 excluded_decomp=excluded_decomp) is None):
            continue
        eligible.append(r)
    train,hold,report=split_oracles(
        eligible,blocked_ids=prior_hashes,
        per_hop_train=a.per_hop_train,per_hop_holdout=a.per_hop_holdout,
    )
    # Ensure no evaluated eight source parents have leaked.
    assert not prior_hashes & {r["source_id_sha256"] for r in train+hold}
    for path,items in ((a.output,train),(a.holdout,hold)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                                for r in items),encoding="utf-8")
    report.update({
        "raw_train_sha256":TRAIN_SHA,"raw_dev_sha256":DEV_SHA,
        "train_file_sha256":sha256(a.output.read_bytes()).hexdigest(),
        "holdout_file_sha256":sha256(a.holdout.read_bytes()).hexdigest(),
    })
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("MUSIQUE_ORACLE_READ_CHAIN_GATE",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
