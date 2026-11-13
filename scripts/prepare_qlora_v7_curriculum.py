"""Training-set builder for *behavioral* QLoRA: disjoint observed tool traces
and broad synthetic policy actions, never memorized GAIA examples.

Different from v5's 4.8% real mix:
- 20% source-stratified observed TRAIN questions reserved for validation;
- include each retained real parent exactly once with *all* assistant turns;
- mix the remaining real states with a capability-balanced synthetic policy
  control at roughly 1:2 supervised-action ratio (no case/answer duplication);
- preserve no-tool stopping and the other seven policy capabilities;
- no silent context cropping, no GPU or success claim in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

_ALLOWED = frozenset({"hotpotqa-distractor-train", "2wikimultihopqa-train"})
_SYNTHETIC = "synthetic-small-agent-policy-v3/train"
_SEED = "small-agent-v7-source-balanced-20261113-v1"


def _key(value: str, namespace: str, seed: str) -> str:
    return sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def _validate_rows(synthetic: list[dict], real: list[dict]) -> None:
    if not synthetic or not real:
        raise ValueError("source coverage requires synthetic and two observed families")
    if {row.get("source") for row in real} != _ALLOWED:
        raise ValueError("real source family or split forbidden/missing")
    if any(row.get("source") != _SYNTHETIC for row in synthetic):
        raise ValueError("synthetic source not TRAIN")
    task_ids = []
    for row in [*synthetic, *real]:
        task_id = row.get("task_id")
        indices = row.get("training_action_indices")
        if (not isinstance(task_id, str) or not task_id
            or row.get("verified") is not True
            or not isinstance(indices, list) or not indices
            or any(type(i) is not int or i<0 for i in indices)
            or len(indices) != len(set(indices))):
            raise ValueError("unverified or malformed native assistant action indices")
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task ID across source families")
    if any(not str(r.get("capability") or "") for r in synthetic):
        raise ValueError("missing synthetic capability")
    real_hashes = []
    for row in real:
        messages = row.get("messages")
        if messages is None:
            continue  # synthetic unit fixtures; production loader verifies rows
        if not isinstance(messages, list):
            raise ValueError("invalid observed trace messages")
        if any(not isinstance(messages[i], dict) or messages[i].get("role")!="assistant"
               for i in row["training_action_indices"]):
            raise ValueError("training action not an observed assistant message")
        if any(m.get("role")=="assistant" and m.get("tool_calls")
               for m in messages[-1:]):
            raise ValueError("observed parent has no final")
        user = [m.get("content") for m in messages if m.get("role")=="user"]
        if len(user)!=1 or not isinstance(user[0],str) or not user[0]:
            raise ValueError("missing unique source task question")
        real_hashes.append(sha256(" ".join(user[0].split()).casefold().encode()).hexdigest())
    if len(real_hashes) != len(set(real_hashes)):
        raise ValueError("duplicate real question; possible train/eval leakage")


def build_curriculum(
    synthetic: list[dict],
    real: list[dict],
    *,
    seed: str = _SEED,
    holdout_fraction: float = 0.2,
    synthetic_to_real_ratio: float = 2.0,
) -> tuple[list[dict], list[dict], dict]:
    """Whole-parent training and source-stratified validation; no source replays."""
    if not 0.1 <= holdout_fraction <= 0.4 or not 1.0 <= synthetic_to_real_ratio <= 4.0:
        raise ValueError("outside bounded preregistered split/action mixing range")
    _validate_rows(synthetic, real)
    grouped = defaultdict(list)
    for row in real:
        grouped[row["source"]].append(row)
    heldout=[]
    selected_real=[]
    for source in sorted(_ALLOWED):
        group = sorted(grouped[source], key=lambda r:_key(r["task_id"],"heldout",seed))
        if len(group)<5:
            raise ValueError("need at least five real tasks per source for independent validation")
        num_val = max(1,round(len(group)*holdout_fraction))
        heldout.extend(group[:num_val])
        selected_real.extend(group[num_val:])
    real_actions = sum(len(x["training_action_indices"]) for x in selected_real)
    if real_actions < 1:
        raise ValueError("empty observed train")
    budget = round(synthetic_to_real_ratio*real_actions)
    caps=defaultdict(list)
    for row in synthetic:
        caps[row["capability"]].append(row)
    if len(caps)<3 or "no_tool_stop" not in caps:
        raise ValueError("insufficient broad policy capabilities or no-tool stopping")
    for cap in caps:
        caps[cap].sort(key=lambda row:_key(row["task_id"],f"cap:{cap}",seed))
    cap_names=sorted(caps)
    chosen=[]
    current=0
    counters=Counter()
    # Round robin over capability families; entire parent selected or skipped.
    while current < budget:
        progressed=False
        for cap in cap_names:
            while caps[cap]:
                candidate=caps[cap].pop(0)
                n=len(candidate["training_action_indices"])
                if current+n<=budget:
                    chosen.append(candidate)
                    current+=n
                    counters[cap]+=1
                    progressed=True
                    break
            if current == budget:
                break
        if not progressed:
            break
    if current < budget-1 or set(counters) != set(cap_names):
        raise ValueError("insufficient balanced synthetic policy actions")
    training = selected_real + chosen
    training.sort(key=lambda row:_key(row["task_id"],"train-order",seed))
    holdout=sorted(heldout,key=lambda row:_key(row["task_id"],"val-order",seed))
    if set(r["task_id"] for r in training) & set(r["task_id"] for r in holdout):
        raise AssertionError("holdout leakage")
    actual=sum(len(x["training_action_indices"]) for x in training)
    ratio=real_actions/actual
    if not 0.18 <=ratio<=0.55:
        raise ValueError("source/action mix out of preregistered bounds")
    report={
        "schema":"qlora-v7-whole-task-source-balanced-20261113-v1",
        "seed":seed, "holdout_fraction":holdout_fraction,
        "source_train_real_tasks":len(selected_real),
        "source_holdout_real_tasks":len(holdout),
        "source_train_real_actions":real_actions,
        "source_holdout_real_actions":sum(len(x["training_action_indices"]) for x in holdout),
        "synthetic_train_tasks":len(chosen),
        "synthetic_train_actions":current,
        "train_actions":actual,
        "train_tasks":len(training),
        "holdout_tasks":len(holdout),
        "train_real_action_fraction":round(ratio,6),
        "synthetic_capabilities_covered":len(counters),
        "synthetic_per_capability":dict(sorted(counters.items())),
        "no_tool_stop_tasks":counters.get("no_tool_stop",0),
        "train_source_counts":dict(sorted(Counter(r["source"] for r in training).items())),
        "holdout_source_counts":dict(sorted(Counter(r["source"] for r in holdout).items())),
        "gaia_training_rows":0,
        "train_parent_repetitions":0,
        "require_full_prompt":True,
        "training_ran":False, "gaia_evaluated":False,
        "claim_boundary":"No real model gain without QLoRA training and matched external GAIA; test source shift.",
    }
    return training,holdout,report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real",type=Path,required=True)
    p.add_argument("--protected",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length",type=int,default=2048)
    a=p.parse_args()
    if len({x.resolve() for x in (a.output,a.validation,a.report)})!=3 or any(
        x.exists() for x in (a.output,a.validation,a.report)):
        raise FileExistsError("refuse overwriting existing training/eval data")
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import (
        load_verified_rows,_render_turn_examples,_tokenize_completion_examples,
    )
    from scripts.prepare_full_qlora_v4 import prepare_full_clean_train
    from transformers import AutoTokenizer

    real=load_verified_rows(a.real,protected_questions_path=a.protected)
    if len(real)<50:
        raise ValueError("fewer than 50 verified complete observed source tasks")
    original=generate_oracle_policy_v3_dataset(
        Path("/tmp/qlora-v7-source"),seed="small-agent-policy-v3")
    raw=[json.loads(x) for x in original["train"].read_text().splitlines() if x.strip()]
    synthetic,_=prepare_full_clean_train(raw)
    train,val,report=build_curriculum(synthetic,real)
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    for name,rows in (("train",train),("validation",val)):
        examples=_render_turn_examples(tokenizer,rows)
        tokenized=_tokenize_completion_examples(
            tokenizer,examples,max_length=a.max_length,require_full_prompt=True)
        expected=sum(len(x["training_action_indices"]) for x in rows)
        if len(tokenized)!=expected:
            raise ValueError(f"{name}: incomplete assistant next-action training labels")
        report[name+"_max_actual_tokens"]=max(len(x["input_ids"]) for x in tokenized)
        report[name+"_fully_masked_prompts"]=len(tokenized)
    for dest,rows in ((a.output,train),(a.validation,val)):
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"
                                for r in rows),encoding="utf-8")
    report["train_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    report["holdout_sha256"]=sha256(a.validation.read_bytes()).hexdigest()
    report["base_model"]=a.model
    report["max_length"]=a.max_length
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("QLORA_V7_DATA_READY",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
