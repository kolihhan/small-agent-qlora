"""Retrospective gold-blind scan of original GAIA100 tool observations.

Flags benchmark-solution *exposure*, NOT attribution of training or model benefit.
Tool results may be truncated; a clean result does not prove no exposure.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.agent.leakage_gate import leakage_signals

def load_rows(root: Path, arm: str):
    paths=sorted(root.glob(f"gaia-v8-paired-shard-*/{arm}/shard-*/results.jsonl"))
    if len(paths)!=20:
        raise ValueError(f"Missing original 20 {arm} shard outputs")
    rows={}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            tid=row["task_id"]
            if tid in rows:
                raise ValueError("duplicate original task id")
            rows[tid]=row
    if len(rows)!=100:
        raise ValueError(f"Expected 100 original GAIA100 cases per arm, got {len(rows)}")
    return rows

def scan_case(row):
    """Never inspect ground-truth labels when constructing leakage flags."""
    calls={}
    last_query=""
    findings=[]
    inspected=0
    for event in row["trace"]:
        kind=event.get("kind")
        data=event.get("data") or {}
        if kind=="tool_call":
            name=data.get("name")
            args=data.get("arguments") or {}
            callid=str(data.get("id",""))
            if name=="search" and isinstance(args,dict):
                q=args.get("query")
                if isinstance(q,str):
                    last_query=q
            if name in {"read","search"}:
                calls[callid]=(name,last_query)
        elif kind=="tool_result":
            callid=str(data.get("id",""))
            name,q=calls.get(callid,("", ""))
            text=data.get("content") or ""
            if name in {"search","read"} and isinstance(text,str) and bool(data.get("ok")):
                inspected+=1
                flags=leakage_signals(q,text)
                if flags:
                    findings.append({"tool":name,"flags":list(flags)})
    return {"inspected":inspected,"findings":findings}

def run(root:Path):
    a,b=load_rows(root,"base"),load_rows(root,"qlora-v8")
    if set(a)!=set(b):
        raise ValueError("Original 100 case IDs not paired")
    tally=Counter()
    samples=[]
    for tid in sorted(a):
        old,new=a[tid],b[tid]
        if not old["correct"] and new["correct"]:
            transition="rescue"
        elif old["correct"] and not new["correct"]:
            transition="regression"
        elif old["correct"]:
            transition="both_correct"
        else:
            transition="both_wrong"
        tally["cases_"+transition]+=1
        scans=[scan_case(old),scan_case(new)]
        for arm,scan in zip(("base","v8"),scans):
            tally[f"{arm}_inspected_observations"]+=scan["inspected"]
            tally[f"{arm}_flagged_observations"]+=len(scan["findings"])
            tally[f"{arm}_cases_with_flags"]+=int(bool(scan["findings"]))
            tally[f"{transition}_{arm}_flagged_cases"]+=int(bool(scan["findings"]))
            for v in scan["findings"]:
                tally[f"{arm}_flag_{v['flags'][0]}"]+=1
        if any(x["findings"] for x in scans):
            samples.append({
                "case_sha256_16":sha256(tid.encode()).hexdigest()[:16],
                "transition":transition,
                "base_flags":scans[0]["findings"],
                "v8_flags":scans[1]["findings"],
            })
    if (tally["cases_rescue"],tally["cases_regression"],tally["cases_both_correct"]) != (8,7,13):
        raise ValueError("Original matched GAIA100 transitions changed")
    return {
        "status":"RESEARCH_GOLD_BLIND_LEAKAGE_SCREEN_NOT_DECONTAMINATED_SCORE",
        "source_run_id":44980055775,
        "matched_task_count":100,
        "counts":dict(sorted(tally.items())),
        "flagged_case_hashes_only":samples,
        "no_reference_answer_access":True,
        "new_model_calls":0,
        "warning":"Conservative regex screening may miss published answer leakage; flagged records are not causal evidence or a corrected GAIA score. No source text is exported.",
    }

def main():
    result=run(Path("evidence"))
    p=Path("audit/gaia-v8-goldblind-leakage-screen.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("GAIA_GOLD_BLIND_LEAKAGE_SCAN",json.dumps(result["counts"],sort_keys=True),flush=True)

if __name__=="__main__":
    main()
