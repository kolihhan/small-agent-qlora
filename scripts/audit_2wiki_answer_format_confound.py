"""Gold-blind answer-rendering diagnostic on previously generated agent replies.

Post-hoc diagnostic only: does not show independent or GAIA improvement.
Parsing uses neither gold nor question. Gold is accessed only by the scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from gaia_small_agent.benchmark.scoring import gaia_score

SOURCE_CASES_SHA="fded3c65e2ccc829ae0efdd8af448896320590b63bb69e0e9ff1d52eadc4fbb7"
SOURCE_REV="fe713bfbd1afbca1a65246741a75890405d56a3a"
ADAPTER_SHA="678a9515d56757e7740c9fbe774ae96d37dad79155b1860b4ac016d1fefa69b0"
PROTOCOL="qwen35-9b-native40-local-2wiki-validation8-base-vs-adapter-v1"
N=8


def last_line_answer(answer: str | None) -> str | None:
    """Last nonempty line only; remove explicit Answer label and paired markdown."""
    if answer is None:
        return None
    lines=[line.strip() for line in str(answer).splitlines() if line.strip()]
    if not lines:
        return answer
    last=lines[-1]
    last=re.sub(r"(?i)^answer\s*:\s*","",last).strip()
    for wrapper in ("**","*","_",chr(96)):
        if last.startswith(wrapper) and last.endswith(wrapper) and len(last)>2*len(wrapper):
            last=last[len(wrapper):-len(wrapper)].strip()
            break
    return last


def diagnostic(paired:dict,source:dict)->dict:
    if (paired.get("protocol")!=PROTOCOL or paired.get("scorable")!=N
        or paired.get("adapter_sha")!=ADAPTER_SHA
        or paired.get("cohort_sha")!=SOURCE_CASES_SHA):
        raise ValueError("Original physical paired cohort / adapter drift")
    cases=source.get("cases")
    if (source.get("split")!="validation"
        or source.get("source_revision")!=SOURCE_REV
        or source.get("cases_sha")!=SOURCE_CASES_SHA
        or not isinstance(cases,list) or len(cases)!=N
        or sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()!=SOURCE_CASES_SHA):
        raise ValueError("Pinned original source validation cohort drift")
    gold={case["id"]:case["answer"] for case in cases}
    rows=paired.get("rows")
    if (not isinstance(rows,list) or len(rows)!=N
        or len({r["id"] for r in rows})!=N
        or set(r["id"] for r in rows)!=set(gold)):
        raise ValueError("Incomplete or duplicate paired validation rows")
    counts={arm:Counter() for arm in ("original","native40_adapter")}
    cases_out=[]
    for row in rows:
        if set(row.get("arms",{}))!=set(counts):
            raise ValueError("Missing paired arm")
        item={"task_id":row["id"]}
        for arm in counts:
            v=row["arms"][arm]
            raw=bool(v.get("strict_exact"))
            if gaia_score(v.get("answer"),gold[row["id"]])!=raw:
                raise ValueError("Raw scorer disagrees with published physical score")
            cleaned=last_line_answer(v.get("answer"))
            extracted=bool(v.get("completed") and gaia_score(cleaned,gold[row["id"]]))
            item[arm]={"raw":raw,"lastline":extracted,
                       "has_full_support":v.get("visible_support_docs")==4}
            counts[arm]["raw"]+=int(raw)
            counts[arm]["lastline"]+=int(extracted)
            counts[arm]["rendering_rescues"]+=int(not raw and extracted)
            counts[arm]["rendering_regressions"]+=int(raw and not extracted)
            counts[arm]["tool_calls"]+=int(v.get("tool_calls",0))
            counts[arm]["tool_errors"]+=int(v.get("tool_errors",0))
            counts[arm]["full_support"]+=int(v.get("visible_support_docs")==4)
        cases_out.append(item)
    for arm in counts:
        c=counts[arm]
        if c["lastline"]!=c["raw"]+c["rendering_rescues"]-c["rendering_regressions"]:
            raise AssertionError("Post-hoc score accounting failed")
    return {
        "status":"POSTHOC_FORMAT_DIAGNOSTIC_NOT_NEW_MODEL_SCORE",
        "source_protocol":PROTOCOL,"source_cohort_sha":SOURCE_CASES_SHA,
        "adapter_sha256":ADAPTER_SHA,"n":N,"no_new_inference":True,
        "no_training":True,"parser_uses_gold_or_question":False,
        "arm_counts":{a:dict(c) for a,c in counts.items()},
        "raw_paired_rescues":sum(not r["original"]["raw"] and r["native40_adapter"]["raw"] for r in cases_out),
        "lastline_paired_rescues":sum(not r["original"]["lastline"] and r["native40_adapter"]["lastline"] for r in cases_out),
        "lastline_paired_regressions":sum(r["original"]["lastline"] and not r["native40_adapter"]["lastline"] for r in cases_out),
        "claim_boundary":"These are 8 previously exposed public 2Wiki VAL questions; post-hoc last-line parsing is not independent test, trained-model gain, or GAIA score. Must freeze independent new source cohort before promoting parser.",
        "cases":cases_out,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--paired",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=diagnostic(json.loads(a.paired.read_text()),json.loads(a.source.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("FORMAT_CONFOUND_AUDIT",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
