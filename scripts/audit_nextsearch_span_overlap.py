#!/usr/bin/env python3
"""Additional fail-closed EXACT-SUBSPAN screen for training tool/assistant payloads.

This is NOT semantic near-duplicate detection. Run alongside
audit_nextsearch_vs_gaia_bird.py; both must pass before declaring that the
supplied source pack has passed the two lexical screens.
Never print benchmark question text, reference SQL or GAIA answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from scripts.audit_nextsearch_vs_gaia_bird import (
    TRAIN_SHA, BIRD_SHA, norm, read_json, read_jsonl, extract_train,
    extract_reference, MIN_GAIA_TEST, MIN_GAIA_VALIDATION, sha
)


def scan(train: Path, bird: Path, gaia_validation: Path, gaia_test: Path) -> dict:
    if sha(train) != TRAIN_SHA or sha(bird) != BIRD_SHA:
        raise ValueError("train or BIRD reference SHA mismatch")
    extract_train(train)
    rows=read_jsonl(train)
    bird_rows=read_json(bird)
    if len(bird_rows)!=1534 or any(not isinstance(x.get("SQL"),str) for x in bird_rows):
        raise ValueError("BIRD DEV full source question/SQL missing")
    gaia={
        "gaia-validation":extract_reference(read_jsonl(gaia_validation),"gaia-validation",MIN_GAIA_VALIDATION),
        "gaia-test":extract_reference(read_jsonl(gaia_test),"gaia-test",MIN_GAIA_TEST),
    }
    # Include every message (system, user, tool, assistant) and original
    # completion/expected answer metadata, deduplicated across shared windows.
    train_spans=set()
    for row in rows:
        for m in row["messages"]:
            value=m.get("content")
            if not isinstance(value,str):
                raise ValueError("nontext message content is not audited")
            if len(norm(value)) >= 12:
                train_spans.add(norm(value))
        for field in ("expected_answer",):
            val=row.get(field)
            if isinstance(val,str) and len(norm(val)) >= 12:
                train_spans.add(norm(val))
    if not train_spans:
        raise ValueError("empty training message span set")
    refs=[]
    for role,questions in gaia.items():
        refs.extend((role,key,norm(q)) for key,q in questions.items())
    for i,row in enumerate(bird_rows):
        cid=str(row.get("question_id") or row.get("id") or i)
        refs.append(("bird-dev-question",cid,norm(row["question"])))
        refs.append(("bird-dev-SQL",cid,norm(row["SQL"])))
    evidence=[]
    for role,cid,ref_text in refs:
        # Reference questions shorter than this are handled by strict
        # parent-question comparison in the companion script, not this pass.
        if len(ref_text)<28:
            continue
        for train_text in train_spans:
            if ref_text in train_text:
                evidence.append({"role":role,
                    "reference_id_sha256":sha256((role+":"+cid).encode()).hexdigest(),
                    "train_message_sha256":sha256(train_text.encode()).hexdigest(),
                    "kind":"full_reference_subspan",
                })
                break
    by_role=dict(Counter(z["role"] for z in evidence))
    return {
        "status":"REVIEW_REQUIRED" if evidence else "PASS_EXACT_SPAN_SCREEN_ONLY",
        "train_sha256":TRAIN_SHA,
        "train_rows":len(rows),
        "unique_train_message_spans":len(train_spans),
        "gaia_validation_questions":len(gaia["gaia-validation"]),
        "gaia_test_questions":len(gaia["gaia-test"]),
        "bird_dev_questions_and_SQL":2*len(bird_rows),
        "reference_sha256":{"bird":sha(bird),
            "gaia_validation":sha(gaia_validation),"gaia_test":sha(gaia_test)},
        "flag_total":len(evidence),"flags_by_role":by_role,
        "hashed_findings":evidence[:100],
        "model_calls":0,"gpu_hours":0,
        "restricted_content_in_report":False,
        "scope":"Exact normalized subspan screen across ALL user/assistant/tool/system message strings and original expected-answer text versus all supplied GAIA question strings and BIRD DEV questions/SQL. Not a semantic-paraphrase audit, no attachments scanned, no independent label verification.",
    }


def main() -> int:
    p=argparse.ArgumentParser()
    for arg in ("train","bird","gaia-validation","gaia-test","output"):
        p.add_argument("--"+arg,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error("refusing to overwrite output")
    try:
        result=scan(a.train,a.bird,a.gaia_validation,a.gaia_test)
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as exc:
        result={"status":"BLOCKED","error_type":type(exc).__name__,"reason":str(exc)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="hashed_findings"}))
    return 0 if result["status"]=="PASS_EXACT_SPAN_SCREEN_ONLY" else 2

if __name__=="__main__":
    sys.exit(main())
