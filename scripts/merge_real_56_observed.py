"""Merge new and earlier strictly curated observed TRAIN data before SFT.

No synthetic answers, no GAIA/validation examples, no repeated prompts,
no best-of-N replay across different source cohorts. The trainable 2048-token
subset is selected by the existing complete-parent tokenizer filter.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.protection import question_hash
from gaia_small_agent.training.qlora import load_verified_rows

_SOURCES={"hotpotqa-distractor-train","2wikimultihopqa-train"}
_CHECKS=("completed","final_answer_correct","zero_tool_errors",
         "zero_duplicate_blocks","required_tools_satisfied","provenance_complete")


def combine_strict_rows(sources: list[list[dict]]) -> tuple[list[dict],dict]:
    if len(sources)!=5 or any(not src for src in sources):
        raise ValueError("requires old 40-source and exactly four nonempty new shards")
    old, *fresh=sources
    if len(old)!=40 or [len(s) for s in fresh]!=[4,4,2,6]:
        raise ValueError("unexpected historic/expanded source row counts")
    seen_task=set()
    seen_question=set()
    merged=[]
    for shard,rows in enumerate(sources):
        for row in rows:
            tid=row.get("task_id")
            if not isinstance(tid,str) or not tid or tid in seen_task:
                raise ValueError("missing/duplicate independent task")
            msg=row.get("messages",[])
            questions=[m.get("content") for m in msg if isinstance(m,dict) and m.get("role")=="user"]
            if len(questions)!=1 or not isinstance(questions[0],str) or not questions[0]:
                raise ValueError("invalid protected task question")
            qh=question_hash(questions[0])
            if qh in seen_question:
                raise ValueError("duplicate question across distinct task IDs")
            if row.get("source") not in _SOURCES or row.get("verified") is not True:
                raise ValueError("forbidden data source / unverified row")
            verified=row.get("verification")
            if not isinstance(verified,dict) or any(verified.get(k) is not True for k in _CHECKS):
                raise ValueError("unclean observed tool policy")
            indices=row.get("training_action_indices")
            if not isinstance(indices,list) or not indices or len(indices)!=len(set(indices)):
                raise ValueError("missing/duplicated assistant action indices")
            if indices[-1] != len(msg)-1 or sum(
                isinstance(m,dict) and m.get("role")=="assistant" and not m.get("tool_calls")
                for m in msg
            )!=1:
                raise ValueError("failed native final-answer preservation")
            if shard>0 and row["source"]!="2wikimultihopqa-train":
                raise ValueError("new shards must use independent 2Wiki train tasks")
            seen_task.add(tid);seen_question.add(qh);merged.append(row)
    if len(merged)!=56:
        raise AssertionError("should combine exactly 56 independent tasks")
    by_source=Counter(row["source"] for row in merged)
    actions=sum(len(row["training_action_indices"]) for row in merged)
    if dict(by_source)!={"hotpotqa-distractor-train":36,"2wikimultihopqa-train":20} or actions!=228:
        raise ValueError("expected 36+20 source balance, 228 native assistant actions")
    return merged,{
        "schema":"qlora-real-observed-56-merged-v1",
        "old_unique_tasks":40,"new_unique_tasks":16,
        "input_unique_tasks":len(merged),
        "input_assistant_actions":actions,
        "source_task_counts":dict(sorted(by_source.items())),
        "all_traces_source_provenance_checked":True,
        "training_ran":False,"gaia_evaluated":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources",nargs=5,required=True,type=Path)
    p.add_argument("--protected",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() or a.report.exists() or a.output.resolve()==a.report.resolve():
        raise FileExistsError("do not replace already frozen evidence")
    source_rows=[
        [json.loads(s) for s in src.read_text(encoding="utf-8").splitlines() if s.strip()]
        for src in a.sources
    ]
    rows,report=combine_strict_rows(source_rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                                for r in rows),encoding="utf-8")
    checked=load_verified_rows(a.output,protected_questions_path=a.protected)
    if len(checked)!=56:
        raise ValueError("GAIA contamination or invalid native trace")
    report["gaia_protected_checked"]=True
    report["data_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("OBSERVED_56_MERGED",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
