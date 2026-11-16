#!/usr/bin/env python3
"""Auditable, fail-closed, task-question overlap screen against pinned GAIA and BIRD DEV.

Inputs MUST be original task questions, not model output from a benchmark. The
screen detects exact/strong lexical near repeats, not semantic equivalence.
No raw benchmark questions/answers/SQL are emitted in the JSON report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

TRAIN_SHA = "5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432"
BIRD_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MIN_GAIA_VALIDATION = 100
MIN_GAIA_TEST = 200

def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def norm(s: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", s.casefold(), flags=re.UNICODE))

def grams(s: str, n: int = 5) -> set[str]:
    v = norm(s).replace(" ", "")
    return {v[i:i+n] for i in range(max(0, len(v)-n+1))}

def similarity(a: str,b: str) -> tuple[float,float,float]:
    ta,tb = set(norm(a).split()),set(norm(b).split())
    lexical = len(ta&tb)/len(ta|tb) if ta|tb else 0.0
    ca,cb=grams(a),grams(b)
    char = len(ca&cb)/len(ca|cb) if ca|cb else 0.0
    ratio=SequenceMatcher(None,norm(a),norm(b),autojunk=False).ratio()
    return lexical,char,ratio

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing input: {path.name}")
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r,dict) for r in rows):
        raise ValueError(f"invalid rows: {path.name}")
    return rows

def read_json(path: Path) -> list[dict]:
    rows=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) for r in rows):
        raise ValueError(f"invalid JSON rows: {path.name}")
    return rows

def extract_train(train_path: Path) -> dict[str,str]:
    if sha(train_path)!=TRAIN_SHA:
        raise ValueError("NextSearch training artifact SHA mismatch")
    rows=read_jsonl(train_path)
    if len(rows)!=48:
        raise ValueError(f"NextSearch rows expected 48, found {len(rows)}")
    parents: dict[str,str]={}
    actions=Counter()
    for row in rows:
        if row.get("source_split")!="train" or row.get("source")!="NextTokenAI/NextSearch-1-Trajectories":
            raise ValueError("not original NextSearch TRAIN source")
        if row.get("task_success_independently_replayed") is not False:
            raise ValueError("training verification provenance changed")
        pid=str(row.get("source_parent_task_id") or "")
        users=[m.get("content") for m in row.get("messages",[]) if m.get("role")=="user"]
        if not pid or len(users)!=1 or not isinstance(users[0],str) or len(norm(users[0]))<24:
            raise ValueError("train parent/question incomplete")
        question=users[0]
        if pid in parents and norm(parents[pid])!=norm(question):
            raise ValueError("one parent task with multiple question texts")
        parents[pid]=question
        actions["final" if row.get("source_answer_visible_in_window") else "continue"]+=1
    if len(parents)!=28 or actions != Counter({"final":24,"continue":24}):
        raise ValueError("source parent/action counts disagree with signed artifact")
    return parents

def extract_reference(rows: list[dict],role:str, expected_min:int) -> dict[str,str]:
    if len(rows)<expected_min:
        raise ValueError(f"reference {role} incomplete count {len(rows)}")
    out={}
    for i,row in enumerate(rows):
        key=str(row.get("task_id") or row.get("question_id") or row.get("id") or i)
        question=row.get("Question",row.get("question"))
        if not isinstance(question,str) or len(norm(question))<12 or key in out:
            raise ValueError(f"reference {role} has duplicate/missing task ID/question")
        out[key]=question
    return out

def audit(train_path:Path, bird_path:Path, gaia_validation_path:Path,gaia_test_path:Path) -> dict:
    if sha(bird_path)!=BIRD_SHA:
        raise ValueError("pinned BIRD DEV source checksum mismatch")
    train=extract_train(train_path)
    bird_rows=read_json(bird_path)
    if len(bird_rows)!=1534 or any(not isinstance(row.get("SQL"),str) for row in bird_rows):
        raise ValueError("BIRD source requires all 1534 question+SQL pairs")
    refs={
        "bird-dev":extract_reference(bird_rows,"bird-dev",1534),
        "gaia-validation":extract_reference(read_jsonl(gaia_validation_path),"gaia-validation",MIN_GAIA_VALIDATION),
        "gaia-test":extract_reference(read_jsonl(gaia_test_path),"gaia-test",MIN_GAIA_TEST),
    }
    if set(refs["gaia-validation"]) & set(refs["gaia-test"]):
        raise ValueError("GAIA validation and test IDs unexpectedly overlap")
    observed=[]
    for p_id, question in sorted(train.items()):
        for role,ref in refs.items():
            for test_id,other in ref.items():
                exact=norm(question)==norm(other)
                lexical,char,ratio=similarity(question,other)
                # Flag *potential* near duplicates; confirmation requires blinded
                # manual review, never silently accept from a threshold alone.
                near=(lexical>=0.72 and min(len(norm(question).split()),len(norm(other).split()))>=7) or char>=0.78 or ratio>=0.89
                if exact or near:
                    observed.append({
                        "train_parent_sha256":sha256(p_id.encode()).hexdigest(),
                        "reference_task_sha256":sha256((role+":"+test_id).encode()).hexdigest(),
                        "reference_role":role,
                        "match":"exact" if exact else "near_review",
                        "token_jaccard":round(lexical,4),
                        "char5_jaccard":round(char,4),
                        "char_sequence_ratio":round(ratio,4),
                    })
    counts=Counter(item["reference_role"] for item in observed)
    return {
        "status":"REVIEW_REQUIRED" if observed else "PASS_LEXICAL_SCREEN_ONLY",
        "train_rows":48, "train_distinct_parent_tasks":len(train),
        "train_source_sha256":sha(train_path),
        "reference_provenance":{
            "bird-dev":{"count":len(refs["bird-dev"]),"sha256":sha(bird_path)},
            "gaia-validation":{"count":len(refs["gaia-validation"]),"sha256":sha(gaia_validation_path)},
            "gaia-test":{"count":len(refs["gaia-test"]),"sha256":sha(gaia_test_path)},
        },
        "flag_counts":dict(counts),
        "total_flag_count":len(observed),
        "hashed_findings":observed[:100],
        "model_calls":0,"gpu_hours":0,
        "benchmark_questions_or_answers_in_report":False,
        "independent_end_task_success_of_training_labels":"NOT_VERIFIED",
        "scope":"Exact and strong near-duplicate screen of original NextSearch parent QUESTIONS vs pinned GAIA validation/test and 1534 BIRD DEV questions. Training example assistant/tool payloads and BIRD SQL semantics are NOT comprehensively audited. Passing does not prove source-independent generalization or semantic noncontamination. Do not promote adapter."
    }

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--bird",type=Path,required=True)
    p.add_argument("--gaia-validation",type=Path,required=True)
    p.add_argument("--gaia-test",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("refuse overwrite")
    try:
        result=audit(args.train,args.bird,args.gaia_validation,args.gaia_test)
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as ex:
        result={"status":"BLOCKED","error_type":type(ex).__name__,"reason":str(ex),
                "source_content_exposed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("hashed_findings",)}))
    return 0 if result["status"]=="PASS_LEXICAL_SCREEN_ONLY" else 2

if __name__=="__main__":
    sys.exit(main())
