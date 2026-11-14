"""Frozen multisource Qwen3.5-9B QLoRA v8 candidate, no training or GPU here.

Preserve original v7 source-balanced 522 supervised actions (v3 policy + real
HotpotQA/2Wiki), add exactly 84 independently observed FinQA read+python
actions after strict curation, and 352 annotation-derived MuSiQue 3-4 hop
read-to-final actions. Hold out v7's 11 real parents and all 24 MuSiQue
different-family, evidence-disjoint parents. No GAIA / DEV training rows.
This script cannot claim that a QLoRA adapter improved generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.protection import (
    load_protected_question_hashes, question_hash,
)
from scripts.curate_finqa_teacher import audit_verified

_ALLOWED = frozenset({
    "2wikimultihopqa-train", "hotpotqa-distractor-train",
    "synthetic-small-agent-policy-v3/train", "finqa-train", "musique-train",
})
_SEED="qwen9b-v8-multisource-parent-balance-20261115"


def _messages(row: dict) -> list[dict]:
    m=row.get("messages")
    if not isinstance(m,list) or not m:
        raise ValueError("invalid messages")
    return m


def _question(row: dict) -> str:
    users=[m.get("content") for m in _messages(row)
           if isinstance(m,dict) and m.get("role")=="user"]
    if len(users)!=1 or not isinstance(users[0],str) or not users[0].strip():
        raise ValueError("non-unique user question")
    return users[0]



def compact_multihop_supervision(row: dict) -> dict:
    """Train initial evidence navigation, late-hop grounding and finalization.

    Keep full real tool observations in every prefix; do not supervise all
    repetitive read calls. v7 over-searched on GAIA. This fixes the 1:5
    final-to-tool imbalance of full MuSiQue paths without duplicating tasks.
    It is a training hypothesis, not a demonstrated model improvement.
    """
    if row.get("source") != "musique-train":
        raise ValueError("only annotated MuSiQue rows may be compacted")
    idx = row.get("training_action_indices")
    msgs = row.get("messages")
    if (not isinstance(idx,list) or len(idx) not in (5,6)
        or not isinstance(msgs,list) or len(set(idx))!=len(idx)
        or any(type(i) is not int or i<0 or i>=len(msgs) for i in idx)
        or any(msgs[i].get("role")!="assistant" for i in idx)
        or not msgs[idx[1]].get("tool_calls")
        or not msgs[idx[-2]].get("tool_calls")
        or msgs[idx[-1]].get("tool_calls")):
        raise ValueError("invalid source-verified MuSiQue tool-to-final sequence")
    result=dict(row)
    result["training_action_indices"]=[idx[1],idx[-2],idx[-1]]
    result["annotation_supervision_profile"]="first_support_last_support_final_v1"
    return result

def merge_supervision(
    base: list[dict], finqa: list[dict], musique: list[dict],
    holdout: list[dict], protected_hashes: set[str],
) -> tuple[list[dict],list[dict],dict]:
    """Fail closed on parent/question/evidence overlap, even across sources."""
    grouped=base+finqa+musique
    allrows=grouped+holdout
    ids=set()
    hashes=set()
    passage=set()
    for row in allrows:
        source=str(row.get("source") or "")
        if source not in _ALLOWED:
            raise ValueError("forbidden source/split (GAIA and DEV are never SFT)")
        if row.get("verified") is not True:
            raise ValueError("unverified training row")
        task=str(row.get("task_id") or "")
        if not task or task in ids:
            raise ValueError("duplicate or missing whole-parent task")
        ids.add(task)
        m=_messages(row)
        indices=row.get("training_action_indices")
        if (not isinstance(indices,list) or not indices
            or len(set(indices))!=len(indices)
            or any(type(i) is not int or i<0 or i>=len(m)
                   or m[i].get("role")!="assistant" for i in indices)):
            raise ValueError("invalid observed/annotated assistant action indices")
        q=_question(row)
        h=question_hash(q)
        if h in protected_hashes:
            raise ValueError("protected GAIA task question overlap")
        if h in hashes:
            raise ValueError("duplicate question across train/holdout")
        hashes.add(h)
        if source=="musique-train":
            if (row.get("derived_from_annotation_not_observed_model") is not True
                or row.get("selection_method")!="human_annotated_support_paths_not_model_generated"):
                raise ValueError("source annotation provenance not explicit")
            support=row.get("supporting_paragraph_hashes")
            if (not isinstance(support,list) or len(support)<3
                or len(support)!=len(set(support))
                or any(not isinstance(x,str) or len(x)!=64 for x in support)):
                raise ValueError("invalid physical supporting passage hashes")
            if passage & set(support):
                raise ValueError("support paragraph leaked across source/holdout")
            passage.update(support)
        elif source=="finqa-train":
            if any(row.get("derived_from_annotation_not_observed_model") is True for _ in [0]):
                raise ValueError("FinQA must be observed, not synthetic human annotation")
    for row in finqa:
        if row["source"]!="finqa-train":
            raise ValueError("forbidden FinQA split")
    if any(r["source"]!="musique-train" for r in musique):
        raise ValueError("forbidden MuSiQue training source")
    if any(r["source"] not in (_ALLOWED-{"finqa-train","musique-train"}) for r in base):
        raise ValueError("unexpected base-v7 training source")
    result=sorted(grouped,key=lambda r:(
        sha256((_SEED+":"+r["task_id"]).encode()).hexdigest(),r["task_id"]))
    dev=sorted(holdout,key=lambda r:(
        sha256((_SEED+":validation:"+r["task_id"]).encode()).hexdigest(),r["task_id"]))
    actions=lambda rows:sum(len(r["training_action_indices"]) for r in rows)
    report={
        "protocol":"qlora-v8-source-balanced-full-parent-provisional-v1",
        "train_parents":len(result),"validation_parents":len(dev),
        "train_actions":actions(result),"validation_actions":actions(dev),
        "train_source_counts":dict(sorted(Counter(r["source"] for r in result).items())),
        "train_source_actions":dict(sorted({
            s:sum(len(r["training_action_indices"]) for r in result if r["source"]==s)
            for s in {r["source"] for r in result}
        }.items())),
        "annotation_oracle_parent_count":sum(r["source"]=="musique-train" for r in result),
        "annotation_oracle_training_actions":sum(len(r["training_action_indices"]) for r in result if r["source"]=="musique-train"),
        "train_final_action_fraction":round(sum(1 for r in result for i in r["training_action_indices"] if not r["messages"][i].get("tool_calls"))/max(1,actions(result)),6),
        "observed_finqa_parent_count":sum(r["source"]=="finqa-train" for r in result),
        "protected_question_hashes_loaded":len(protected_hashes),
        "no_gaia_training":True,
        "train_holdout_question_overlap":0,
        "train_holdout_support_paragraph_overlap":0,
        "whole_parent_repeats":0,
        "train_model":"Qwen/Qwen3.5-9B",
        "training_ran":False,"gaia_evaluated":False,
        "boundary":"All TRAIN only: v7 real+synthetic observed, FinQA strict real read+python, MuSiQue source-annotated not observed. MuSiQue supervision is three deliberate decisions per whole trajectory; full prefixes preserved. GAIA/FinQA DEV must remain external.",
    }
    return result,dev,report


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _strict_finqa(path: Path) -> tuple[list[dict],int]:
    accepted=[]
    removed=0
    for row in _jsonl(path):
        try:
            indices=audit_verified(row)
        except (ValueError,TypeError,KeyError):
            removed+=1
            continue
        if row.get("training_action_indices") not in (None,indices):
            raise ValueError("verified FinQA action-index mismatch")
        rec=dict(row)
        rec["training_action_indices"]=indices
        accepted.append(rec)
    return accepted,removed


def build(
    base_file:Path,base_holdout:Path,first16:Path,fresh48:Path,
    musique_file:Path,musique_holdout:Path,mu_report_file:Path,
    protected_file:Path,
) -> tuple[list[dict],list[dict],dict]:
    base=_jsonl(base_file)
    base_dev=_jsonl(base_holdout)
    mu=[compact_multihop_supervision(row) for row in _jsonl(musique_file)]
    mu_dev=_jsonl(musique_holdout)
    mu_report=json.loads(mu_report_file.read_text(encoding="utf-8"))
    if (len(base)!=218 or sum(len(r.get("training_action_indices",[])) for r in base)!=522
        or len(base_dev)!=11 or len(mu)!=64 or len(mu_dev)!=24
        or mu_report.get("train")!=64 or mu_report.get("holdout")!=24
        or sha256(musique_file.read_bytes()).hexdigest()!=mu_report.get("train_file_sha256")
        or sha256(musique_holdout.read_bytes()).hexdigest()!=mu_report.get("holdout_file_sha256")):
        raise ValueError("frozen v7 or MuSiQue source audit identity/count changed")
    first,rej1=_strict_finqa(first16)
    fresh,rej2=_strict_finqa(fresh48)
    finqa=first+fresh
    if (len(first)!=5 or len(fresh)!=17
        or sum(len(x["training_action_indices"]) for x in finqa)!=84):
        raise ValueError(f"FinQA strict oracle parent/action numbers unexpectedly changed: "
                         f"first={len(first)} fresh={len(fresh)} "
                         f"actions={sum(len(x['training_action_indices']) for x in finqa)}")
    protected=load_protected_question_hashes(protected_file)
    if len(protected)<100:
        raise ValueError("GAIA protected validation hashes missing")
    train,dev,report=merge_supervision(base,finqa,mu,base_dev+mu_dev,protected)
    if (report["train_parents"]!=304 or report["train_actions"]!=798
        or report["validation_parents"]!=35):
        raise ValueError("unexpected v8 full-source parent/action exposure")
    report.update({
        "removed_first16_failed_python_verification":rej1,
        "removed_fresh48_failed_python_verification":rej2,
        "input_shas":{k:sha256(p.read_bytes()).hexdigest() for k,p in (
            ("v7_train",base_file),("v7_holdout",base_holdout),
            ("finqa_first16",first16),("finqa_fresh48",fresh48),
            ("musique_train",musique_file),("musique_holdout",musique_holdout)
        )},
    })
    return train,dev,report


def main():
    ap=argparse.ArgumentParser()
    for arg in ["base","base-holdout","finqa-first16","finqa-fresh48",
                "musique","musique-holdout","musique-report","protected",
                "output","validation","report"]:
        ap.add_argument("--"+arg,type=Path,required=True)
    a=ap.parse_args()
    out,valid,report_path=a.output,a.validation,a.report
    if len({p.resolve() for p in (out,valid,report_path)})!=3 or any(
        p.exists() for p in (out,valid,report_path)):
        raise FileExistsError("refuse to overwrite frozen QLoRA v8 inputs")
    train,dev,report=build(
        a.base,a.base_holdout,a.finqa_first16,a.finqa_fresh48,
        a.musique,a.musique_holdout,a.musique_report,a.protected)
    for dest,rows in ((out,train),(valid,dev)):
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows))
    report["train_sha256"]=sha256(out.read_bytes()).hexdigest()
    report["validation_sha256"]=sha256(valid.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("QLORA_V8_SOURCE_MIX_FROZEN",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
