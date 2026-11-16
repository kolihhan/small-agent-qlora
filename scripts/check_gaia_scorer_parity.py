#!/usr/bin/env python3
"""Compare local GAIA scorer against an actual SHA-pinned public leaderboard scorer.

No GAIA question, answer or model prediction is loaded. Synthetic edge cases only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gaia_small_agent.benchmark.scoring import gaia_score

def reference(path:Path):
    spec=importlib.util.spec_from_file_location("official_gaia_pinned_scorer",path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import official scorer")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module,"question_scorer",None)):
        raise RuntimeError("official scorer missing question_scorer")
    return module.question_scorer

def synthetic_cases():
    basic=[
       ("New York","new york"),("N.Y.","ny"),("hello world","helloworld"),
       ("The answer is 42","42"),("42.0","42"),("$1,234","1234"),("42%","42"),
       ("1; 2","1,2"),("A; B","a;b"),("A; B","a,b"),
       ("A, B","A, C"),("A, B","A"),("", "None"),(None,"None"),
       ("0","0"),("0","-0.0"),("-32","-32"),("0.03","3%"),
       ("Hello!","hello"),("Paris","Paris."),("London;Paris","London; Paris"),
       ("abc,1","ABC,1"),("abc,1","ABC,2"),
    ]
    letters=["A","Z","New York","foo-bar","red blue","Hi!","file.pdf"]
    nums=["0","1","2","-3","12.5","1000","4e2"]
    rng=random.Random(16042)
    for _ in range(100):
        a=rng.choice(letters+nums)
        b=rng.choice(letters+nums)
        sep=rng.choice([",",";"," "])
        gt=rng.choice(letters+nums+[a+sep+b])
        answer=rng.choice([a,a.lower(),a.upper(),a+sep+b,a+".", " "+a+" ", None])
        basic.append((answer,gt))
    return basic

def compare(source:Path):
    official=reference(source)
    cases=synthetic_cases()
    mismatches=[]
    from contextlib import redirect_stdout
    import io
    import warnings
    with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for i,(answer,gt) in enumerate(cases):
            expected=bool(official(answer,gt))
            actual=bool(gaia_score(answer,gt))
            if actual != expected:
                mismatches.append({"case_index":i,"official":expected,"local":actual,
                                   "answer":answer,"synthetic_gold":gt})
    return {"status":"PASS" if not mismatches else "FAIL","n_synthetic":len(cases),
            "mismatch_count":len(mismatches),"mismatches":mismatches[:20],
            "gaia_task_ids_exposed":0,"gaia_gold_answers_exposed":0,
            "note":"Parity on these synthetic cases only; not proof across every possible official input."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-scorer-file",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error("refusing to overwrite report")
    result=compare(a.official_scorer_file)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["status"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
