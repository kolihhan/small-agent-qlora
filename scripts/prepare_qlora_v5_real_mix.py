"""Deterministic, no-duplication v5 real/synthetic 3200-action SFT candidate.

Replace exactly 154 synthetic policy actions with the already-curated 154
complete real HotpotQA + 2Wiki actions. Never create repeated real examples,
never train on GAIA, and never allow the real data to exceed 5% of actions.
This creates DATA ONLY; neither starts SFT nor claims GAIA improvement.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

SEED="v5-real-4p8percent-20261113-precommitted-v1"
_ALLOWED={"hotpotqa-distractor-train","2wikimultihopqa-train"}


def _rank(task:str,namespace:str)->str:
    return sha256(f"{SEED}:{namespace}:{task}".encode()).hexdigest()


def merge_rows(synthetic:list[dict],real:list[dict],*,
               target_actions:int=3200,max_real_fraction:float=0.05)->tuple[list[dict],dict]:
    if not synthetic or not real or target_actions<1 or not 0<max_real_fraction<=0.10:
        raise ValueError("invalid data mixing bounds")
    if len({r.get("task_id") for r in synthetic+real})!=len(synthetic+real):
        raise ValueError("duplicate independent task ID across data sources")
    for row in synthetic:
        if row.get("source")!="synthetic-small-agent-policy-v3/train" or not row.get("verified"):
            raise ValueError("unprotected source or unverified synthetic policy row")
    if {r.get("source") for r in real}!=_ALLOWED or any(r.get("verified") is not True for r in real):
        raise ValueError("real source diversity / train split failed")
    def action_count(row):
        indices=row.get("training_action_indices")
        if not isinstance(indices,list) or not indices or len(set(indices))!=len(indices):
            raise ValueError("invalid native assistant action provenance")
        return len(indices)
    synthetic_actions=sum(action_count(r) for r in synthetic)
    real_actions=sum(action_count(r) for r in real)
    if synthetic_actions!=target_actions:
        raise ValueError("original synthetic control must exactly match target-action budget")
    if real_actions/target_actions > max_real_fraction:
        raise ValueError("real data would dominate synthetic reference control")
    if real_actions%2:
        raise ValueError("this frozen mixing protocol needs an even replacement budget")
    to_drop=real_actions//2
    eligible=defaultdict(list)
    for row in synthetic:
        if action_count(row)==2:
            eligible[str(row.get("capability") or "unspecified")].append(row)
    if sum(len(xs) for xs in eligible.values())<to_drop:
        raise ValueError("not enough complete 2-action synthetic tasks to replace")
    for capability,group in eligible.items():
        group.sort(key=lambda row:(_rank(row["task_id"],capability),row["task_id"]))
    selected=[]
    # Round-robin across capabilities avoids deleting all examples of one
    # rare skill. Every removed synthetic task has precisely TWO actions.
    groups=sorted(eligible)
    while len(selected)<to_drop:
        progressed=False
        for cap in groups:
            if eligible[cap] and len(selected)<to_drop:
                selected.append(eligible[cap].pop(0))
                progressed=True
        if not progressed:
            raise ValueError("insufficient balanced two-action replacement pool")
    discarded={r["task_id"] for r in selected}
    kept=[r for r in synthetic if r["task_id"] not in discarded]
    mixed=kept+real
    mixed.sort(key=lambda r:(_rank(r["task_id"],"interleave"),r["task_id"]))
    if (len(mixed)!=len(synthetic)-to_drop+len(real)
        or sum(action_count(r) for r in mixed)!=target_actions
        or len({r["task_id"] for r in mixed})!=len(mixed)):
        raise AssertionError("mixing corrupted the 3200-action control")
    sources=Counter(row["source"] for row in mixed)
    report={
        "schema":"policy-v5-synthetic-real-unique-sft-data-v1",
        "target_supervised_actions":target_actions,
        "synthetic_source_tasks":len(synthetic),
        "synthetic_source_actions":synthetic_actions,
        "real_source_tasks":len(real),
        "real_source_actions":real_actions,
        "removed_synthetic_tasks":len(selected),
        "removed_synthetic_actions":2*len(selected),
        "retained_synthetic_tasks":len(kept),
        "retained_synthetic_actions":sum(action_count(r) for r in kept),
        "unique_mixed_tasks":len(mixed),
        "mixed_actions":sum(action_count(r) for r in mixed),
        "real_action_fraction":round(real_actions/target_actions,6),
        "by_source":dict(sorted(sources.items())),
        "synthetic_capabilities_after":dict(sorted(Counter(
            r.get("capability","unspecified") for r in kept).items())),
        "omitted_synthetic_capabilities":dict(sorted(Counter(
            r.get("capability","unspecified") for r in selected).items())),
        "real_task_oversampling_factor":1,
        "dataset_has_guessed_answers":False,
        "training_ran":False,
        "gaia_evaluated":False,
        "evaluation_ready":False,
        "caveat":"Data candidate only, full 2048-token T4 memory feasibility and GAIA Shadow25 control are not yet established.",
    }
    return mixed,report


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-source",type=Path,required=True)
    p.add_argument("--protected",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve()==a.report.resolve() or a.output.exists() or a.report.exists():
        raise FileExistsError("never overwrite frozen dataset")
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import load_verified_rows,_render_turn_examples,_tokenize_completion_examples
    from scripts.prepare_full_qlora_v4 import prepare_full_clean_train
    from transformers import AutoTokenizer
    source=generate_oracle_policy_v3_dataset(Path("/tmp/qlora-v5-source"),seed="small-agent-policy-v3")
    train=[json.loads(x) for x in source["train"].read_text(encoding="utf-8").splitlines() if x.strip()]
    cleaned,synreport=prepare_full_clean_train(train)
    if len(cleaned)!=1600 or synreport["total_trainable_actions"]!=3200:
        raise ValueError("synthetic v4 control unexpectedly changed")
    real=load_verified_rows(a.real_source,protected_questions_path=a.protected)
    if (len(real)!=39 or sum(len(r["training_action_indices"]) for r in real)!=154 or
        {r["source"]:sum(x["source"]==r["source"] for x in real) for r in real}
          !={"hotpotqa-distractor-train":35,"2wikimultihopqa-train":4}):
        raise ValueError("real source not pinned 39-task/154-action input")
    if sha256(a.real_source.read_bytes()).hexdigest()!="79c22f0ef7bb6015eeb3997d11f302caf6fc77667d1494813a315ae41ceab676":
        raise ValueError("real data SHA disagrees with checked trainer artifact")
    mixed,report=merge_rows(cleaned,real)
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B",trust_remote_code=True)
    examples=_render_turn_examples(tokenizer,mixed)
    if len(examples)!=3200:
        raise ValueError("rendered native assistant actions do not match total")
    tokenized=_tokenize_completion_examples(
        tokenizer,examples,max_length=2048,require_full_prompt=True)
    if len(tokenized)!=3200:
        raise ValueError("full-prompt tokenizer failed")
    report["tokenizer"]="Qwen/Qwen3.5-9B"
    report["max_length"]=2048
    report["require_full_prompt"]=True
    report["gaia_protected_hashes"]=True
    report["real_source_sha256"]=sha256(a.real_source.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                                for r in mixed),encoding="utf-8")
    report["mixed_data_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("QLORA_MIXED_V5_GOLD_BLIND",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
