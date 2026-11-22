"""Safe existing GAIA v8 discordant-case priority audit; never touches prompts or gold."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(data: dict) -> dict:
    if (data.get("n")!=100 or data.get("base_correct")!=20
        or data.get("v8_correct")!=21 or data.get("run_id")!=44980055775):
        raise ValueError("Wrong original matched v8 score source")
    rows=data.get("changed_15_cases",[])
    if len(rows)!=15 or sum(x.get("outcome")=="rescue" for x in rows)!=8 or sum(x.get("outcome")=="regression" for x in rows)!=7:
        raise ValueError("Discordant 8/7 source does not match")
    if len({x["task_sha256_16"] for x in rows})!=15:
        raise ValueError("Duplicate safe case identity")
    rescues=Counter()
    regressions=Counter()
    for row in rows:
        baseline,new=row["base"],row["v8"]
        if row["outcome"]=="rescue":
            if baseline["correct"] is not False or new["correct"] is not True:
                raise ValueError("Mislabelled rescue")
            rescues[f"base_stop_{baseline['stop_reason']}"]+=1
        else:
            if baseline["correct"] is not True or new["correct"] is not False:
                raise ValueError("Mislabelled regression")
            regressions[f"v8_stop_{new['stop_reason']}"]+=1
            regressions[f"v8_failure_{new['failure_label']}"]+=1
    return {
        "status":"RESEARCH_PRIORITY_FROM_PREVIOUSLY_EXPOSED_GAIA100_NOT_CAUSAL",
        "source_run":44980055775,
        "paired_n":100,"base_correct":20,"v8_correct":21,
        "rescues":8,"regressions":7,
        "rescue_source_stop_types":dict(rescues),
        "regression_v8_types":dict(regressions),
        "v8_regressions_eligible_for_final_text_retry":regressions["v8_stop_invalid_final"],
        "v8_regressions_normal_final_but_wrong":regressions["v8_stop_final"],
        "decision":"Do not equate protocol eligibility with correctness; prioritize answer-grounding hypothesis over extra retry calls.",
        "no_new_model_calls":True,"no_training":True,"no_new_score":True,
        "gold_or_answers_accessed":False,
        "limits":"Stop/failure labels are observational, not diagnoses. GAIA source contamination unresolved.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=summarize(json.loads(a.source.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("V8_FINAL_PRIORITY_AUDIT",json.dumps(report,sort_keys=True))


if __name__=="__main__":main()
